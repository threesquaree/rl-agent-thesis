"""
Hierarchical Reinforcement Learning Training Loop

This module implements the main training loop for the HRL museum dialogue agent.
It orchestrates the interaction between the agent, environment, and simulator,
managing episodes, logging, and training statistics.

Key Components:
- Episode Management: Multi-episode training with proper reset and logging
- Agent-Environment Interaction: Step-by-step dialogue simulation
- Simulator Integration: User behavior simulation and state updates
- Training Monitoring: Real-time logging and statistics tracking
- Log Management: JSON serialization and file saving
"""

import random
import numpy as np
import time
import os
from typing import Dict, Any, Optional, List
from src.environment.env import MuseumDialogueEnv
from src.simulator import get_simulator
from src.visualization.training_monitor import TrainingMonitor
from src.visualization.live_training_monitor import LiveTrainingMonitor
from src.visualization.metrics_tracker import MetricsTracker
from src.visualization.museum_map_visualizer import MuseumMapVisualizer
from src.visualization.live_progress import LiveProgressTracker
from src.agent.actor_critic_agent import ActorCriticAgent
from src.training.actor_critic_trainer import ActorCriticTrainer
from src.utils.detailed_logger import DetailedEpisodeLogger
import json
from datetime import datetime
from pathlib import Path
import torch
import sys
import io
import logging

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logger = logging.getLogger(__name__)


class ExhibitTracker:
    """Tracks agent and user navigation through exhibits and AOIs"""
    
    def __init__(self):
        self.agent_current_exhibit = None
        self.agent_current_aoi = None
        self.user_current_exhibit = None
        self.user_current_aoi = None
        
        self.agent_exhibit_history = []
        self.user_exhibit_history = []
        self.agent_aoi_history = []
        self.user_aoi_history = []
        
        self.exhibits_visited = set()
        self.aois_visited = set()
    
    def update_agent_location(self, exhibit: str, aoi: Optional[str] = None):
        """Track where agent is directing attention"""
        # Check if exhibit changed
        if exhibit != self.agent_current_exhibit:
            self.agent_exhibit_history.append({
                'exhibit': exhibit,
                'turn': len(self.agent_exhibit_history)
            })
            self.exhibits_visited.add(exhibit)
        
        if aoi and aoi != self.agent_current_aoi:
            self.agent_aoi_history.append({
                'aoi': aoi,
                'exhibit': exhibit,
                'turn': len(self.agent_aoi_history)
            })
            self.aois_visited.add(aoi)
        
        self.agent_current_exhibit = exhibit
        if aoi:
            self.agent_current_aoi = aoi
    
    def update_user_location(self, exhibit: str, aoi: str):
        """Track where user is looking"""
        # Check if exhibit changed
        if exhibit != self.user_current_exhibit:
            self.user_exhibit_history.append({
                'exhibit': exhibit,
                'turn': len(self.user_exhibit_history)
            })
        
        if aoi != self.user_current_aoi:
            self.user_aoi_history.append({
                'aoi': aoi,
                'exhibit': exhibit,
                'turn': len(self.user_aoi_history)
            })
        
        self.user_current_exhibit = exhibit
        self.user_current_aoi = aoi
    
    def get_alignment_status(self) -> str:
        """Check if agent and user are aligned on same exhibit/AOI"""
        if self.agent_current_exhibit == self.user_current_exhibit:
            if self.agent_current_aoi == self.user_current_aoi:
                return "ALIGNED"  # Same exhibit and AOI
            else:
                return "SAME_EXHIBIT"  # Same exhibit, different AOI
        else:
            return "MISALIGNED"  # Different exhibits
    
    def reset(self):
        """Reset tracker for new episode"""
        self.__init__()


class HRLTrainingLoop:
    """
    Hierarchical Reinforcement Learning Training Loop
    
    Manages the complete training process for the HRL museum dialogue agent:
    - Episode orchestration and termination
    - Agent-environment-simulator interaction
    - Training statistics and logging
    - Session management and cleanup
    """
    
    def __init__(self, max_episodes: int = 10, max_turns_per_episode: int = 20,
                 knowledge_graph_path: str = None, turn_delay: float = 1.5,
                 learning_rate: float = 5e-5, gamma: float = 0.99,  # Reduced from 1e-4 for stability
                 use_actor_critic: bool = True, device: str = 'cpu',
                 show_prompts: bool = False, force_option: str = None,
                 force_subaction: str = None, enable_live_monitor: bool = False,
                 save_metrics: bool = True, enable_map_viz: bool = True,
                 save_map_frames: bool = False, live_map_display: bool = False,
                 map_interval: int = 50, verbose: bool = False,
                 simulator_type: str = "sim8",
                 stochasticity: float = 0.5,
                 termination_strategy: str = "learned",
                 state_representation: str = "dialoguebert",
                 option_granularity: str = "medium",
                 experiment_name: str = None,
                 # H1 Termination Tuning Parameters
                 termination_reg: float = 0.01,
                 intra_option_threshold: float = 0.1,
                 intra_option_weight: float = 0.5,
                 beta_supervision_weight: float = 0.0,  # H1 Advanced: heuristic termination supervision (0.0 = pure Option-Critic)
                 deliberation_cost: float = 0.0,
                 entropy_floor: float = 0.02,
                 max_option_duration: int = None,
                 # H1 Phase 2: Entropy Control Parameters
                 entropy_coef: float = 0.08,
                 entropy_decay_start: int = 0,
                 entropy_decay_end: int = 1000,
                 adaptive_entropy: bool = False,
                 adaptive_entropy_threshold: float = 2.5,
                 adaptive_entropy_multiplier: float = 1.5,
                 # H1 Advanced: Separate learning rates and entropies
                 lr_intra_option: float = None,
                 entropy_coef_option: float = None,
                 entropy_coef_intra: float = None,
                 # H1 Advanced: Value learning parameters
                 value_loss_coef: float = 0.5,
                 target_update_interval: int = 20,
                 value_clip: float = 10.0):
        # ===== TRAINING CONFIGURATION =====
        self.experiment_name = experiment_name or "HRL_Training"
        self.training_title = "HRL MUSEUM DIALOGUE TRAINING"
        self.max_episodes = max_episodes
        self.max_turns_per_episode = max_turns_per_episode
        self.turn_delay = turn_delay
        self.use_actor_critic = use_actor_critic
        self.device = device
        self._show_prompts = show_prompts
        self.verbose = verbose  # If False, use clean progress tracker
        
        # ===== TESTING/DEBUGGING OPTIONS =====
        self.force_option = force_option
        self.force_subaction = force_subaction
        if self.force_option or self.force_subaction:
            print(f"\n[WARNING] TESTING MODE ENABLED")
            if self.force_option:
                print(f"   Force Option: {self.force_option}")
            if self.force_subaction:
                print(f"   Force Subaction: {self.force_subaction}\n")
        
        # ===== COMPONENT INITIALIZATION =====
        # Load knowledge graph (single source of truth)
        from src.utils.knowledge_graph import SimpleKnowledgeGraph
        self.knowledge_graph = SimpleKnowledgeGraph(knowledge_graph_path)
        
        # Store hypothesis-specific parameters
        self.termination_strategy = termination_strategy
        self.state_representation = state_representation
        self.option_granularity = option_granularity
        
        # Store H1 termination tuning parameters
        self.termination_reg = termination_reg
        self.intra_option_threshold = intra_option_threshold
        self.intra_option_weight = intra_option_weight
        self.beta_supervision_weight = beta_supervision_weight  # H1 Advanced: heuristic termination supervision
        self.deliberation_cost = deliberation_cost
        self.entropy_floor = entropy_floor
        self.max_option_duration = max_option_duration
        
        # Store H1 Phase 2: Entropy control parameters
        self.entropy_coef = entropy_coef
        self.entropy_decay_start = entropy_decay_start
        self.entropy_decay_end = entropy_decay_end
        self.adaptive_entropy = adaptive_entropy
        self.adaptive_entropy_threshold = adaptive_entropy_threshold
        self.adaptive_entropy_multiplier = adaptive_entropy_multiplier
        self.last_oci = 0.0  # Track OCI for adaptive entropy
        
        # H1 Advanced: Diversity reward tracking
        self.diversity_reward_coef = float(os.environ.get("HRL_DIVERSITY_REWARD_COEF", "0.0"))
        self.option_usage_window = []  # Track recent option selections for diversity calculation
        self.option_usage_window_size = 100  # Window size for option usage tracking
        
        # Store H1 Advanced parameters
        self.lr_intra_option = lr_intra_option
        self.entropy_coef_option = entropy_coef_option
        self.entropy_coef_intra = entropy_coef_intra
        self.value_loss_coef = value_loss_coef
        self.target_update_interval = target_update_interval
        
        # Initialize environment with knowledge graph (H4: state representation, H6: option granularity)
        self.env = self._create_environment(
            knowledge_graph_path=knowledge_graph_path,
            max_turns=max_turns_per_episode,
            state_representation=state_representation,
            option_granularity=option_granularity,
            deliberation_cost=deliberation_cost
        )
        
        # Initialize user simulator with knowledge graph (NOT just exhibit list)
        # Use factory function to allow switching between sim8 and state_machine simulators
        self.simulator_type = simulator_type
        self.simulator = get_simulator(
            simulator_type=simulator_type,
            knowledge_graph=self.knowledge_graph,
            stochasticity=stochasticity,
        )
        if simulator_type != "sim8":
            print(f"[Training] Using simulator: {simulator_type}")
        
        # Initialize training monitor for logging and statistics
        self.monitor = TrainingMonitor()
        
        # ===== LIVE MONITORING SYSTEM =====
        self.enable_live_monitor = enable_live_monitor
        self.save_metrics = save_metrics
        
        # Initialize live training monitor (optional)
        self.live_monitor = LiveTrainingMonitor(
            enabled=enable_live_monitor,
            log_dir="training_logs"
        )
        
        # Initialize metrics tracker (always created, saved conditionally)
        self.metrics_tracker = MetricsTracker()
        
        # Initialize live progress tracker (clean output)
        self.progress_tracker = LiveProgressTracker(max_episodes=max_episodes, experiment_name=self.experiment_name)
        
        # ===== MUSEUM MAP VISUALIZATION =====
        self.enable_map_viz = enable_map_viz
        self.save_map_frames = save_map_frames
        self.map_interval = map_interval
        
        # Determine map save directory (use experiment dir if available)
        experiment_dir = os.environ.get('EXPERIMENT_DIR', None)
        if experiment_dir:
            map_save_dir = os.path.join(experiment_dir, 'maps')
        else:
            map_save_dir = "training_logs/maps"
        
        # Initialize map visualizer (optional)
        self.map_visualizer = MuseumMapVisualizer(
            enabled=enable_map_viz,
            exhibits=self.env.exhibit_keys,
            save_dir=map_save_dir,
            live_display=False  # Save files but don't display live
        )
        
        # Initialize LLM timing trackers
        self.agent_llm_times = []
        self.simulator_llm_times = []
        
        # Initialize detailed episode logger
        experiment_dir = os.environ.get('EXPERIMENT_DIR', None)
        if experiment_dir:
            reward_mode = getattr(self.env, 'reward_mode', 'baseline')
            self.detailed_logger = DetailedEpisodeLogger(Path(experiment_dir), reward_mode=reward_mode)
        else:
            self.detailed_logger = None
        
        # Initialize exhibit/AOI tracker for navigation visualization
        
        # ===== ACTOR-CRITIC AGENT =====
        if self.use_actor_critic:
            # Get state dimension from environment (already includes H6 option config)
            state_dim = self.env.observation_space.shape[0]
            
            # Get options and subactions from environment (set during creation for H6)
            options = self.env.options
            subactions = self.env.subactions
            
            # Log option granularity if not default
            if option_granularity != "medium":
                print(f"[H6] Using {option_granularity} option granularity: {len(options)} options")
            
            # Initialize Actor-Critic agent with termination strategy (H5)
            self.agent = ActorCriticAgent(
                state_dim=state_dim,
                options=options,
                subactions=subactions,
                hidden_dim=256,
                lstm_hidden_dim=128,
                use_lstm=True,
                device=self.device,
                termination_strategy=termination_strategy
            )
            
            if termination_strategy != "learned":
                print(f"[H5] Using termination strategy: {termination_strategy}")
            
            # Initialize Actor-Critic trainer with improved stability features
            # H1 termination tuning: pass termination_reg, intra_option_threshold/weight, entropy_floor
            # H1 Phase 2: pass entropy control and adaptive entropy parameters
            # H1 Advanced: pass separate learning rates, entropies, and value learning parameters
            self.trainer = ActorCriticTrainer(
                agent=self.agent,
                learning_rate=learning_rate,
                lr_intra_option=lr_intra_option,  # H1 Advanced: Separate LR for intra-option policies
                gamma=gamma,
                device=self.device,
                max_episodes=max_episodes,
                use_target_network=True,  # Enable target network for value stability
                target_update_interval=target_update_interval,  # H1 Advanced: Configurable target update
                normalize_advantages=True,  # Normalize advantages per batch
                value_loss_coef=value_loss_coef,  # H1 Advanced: Configurable value loss coefficient
                value_clip=value_clip,  # Value clipping range for stability
                entropy_coef=self.entropy_coef,  # H1 Phase 2: Use passed parameter (fallback)
                entropy_coef_option=entropy_coef_option,  # H1 Advanced: Separate entropy for option policy
                entropy_coef_intra=entropy_coef_intra,  # H1 Advanced: Separate entropy for intra-option policies
                entropy_decay_start=self.entropy_decay_start,  # H1 Phase 2: Use passed parameter
                entropy_decay_end=self.entropy_decay_end,  # H1 Phase 2: Use passed parameter
                entropy_final=self.entropy_floor,  # H1 tuning: configurable entropy floor
                termination_reg=self.termination_reg,  # H1 tuning: termination regularization
                intra_option_threshold=self.intra_option_threshold,  # H1 tuning
                intra_option_weight=self.intra_option_weight,  # H1 tuning
                beta_supervision_weight=self.beta_supervision_weight,  # H1 Advanced: heuristic termination supervision
                adaptive_entropy=self.adaptive_entropy,  # H1 Phase 2: Adaptive entropy control
                adaptive_entropy_threshold=self.adaptive_entropy_threshold,  # H1 Phase 2
                adaptive_entropy_multiplier=self.adaptive_entropy_multiplier  # H1 Phase 2
            )
            
            # Episode buffer for training
            self.episode_buffer = {
                'states': [],
                'options': [],
                'subactions': [],
                'rewards': [],
                'next_states': [],
                'dones': [],
                'available_subactions': [],
                'beta_targets': []  # H1 Advanced: heuristic termination targets
            }
        
        # ===== TRAINING STATE =====
        self.current_episode = 0
        self.total_episodes = 0
        self.total_turns = 0
        self.episode_rewards = []
    
    def _create_environment(self, knowledge_graph_path: str, max_turns: int, state_representation: str, 
                              option_granularity: str = "medium", deliberation_cost: float = 0.0):
        """
        Factory function to create environment with appropriate state representation (H4) and option granularity (H6).
        
        Args:
            knowledge_graph_path: Path to knowledge graph JSON
            max_turns: Maximum turns per episode
            state_representation: One of 'dialoguebert' (149-d), 'dialogue_act' (23-d)
            option_granularity: One of 'medium' (4 options), 'coarse' (2 options), 'coarse_3opt' (3 options), 'coarse_4opt' (4 options reward-aligned)
            deliberation_cost: Per-step penalty for option persistence (H1 termination tuning)
            
        Returns:
            Environment instance with appropriate state representation and option config
        """
        # Get option configuration for H6 if not using default
        options_override = None
        subactions_override = None
        if option_granularity != "medium":
            from src.agent.option_configs import get_option_config
            options_override, subactions_override = get_option_config(option_granularity)
        
        if state_representation == "dialogue_act":
            # H4: Compact dialogue-act classification (23-d)
            try:
                from src.environment.h4_env import H5StateAblationEnv
                print("[H4] Using dialogue-act state representation (23-d)")
                env = H5StateAblationEnv(
                    knowledge_graph_path=knowledge_graph_path,
                    max_turns=max_turns,
                    deliberation_cost=deliberation_cost
                )
            except ImportError:
                print("[WARNING] H4 dialogue-act environment not found, using default")
                env = MuseumDialogueEnv(
                    knowledge_graph_path=knowledge_graph_path,
                    max_turns=max_turns,
                    options_override=options_override,
                    subactions_override=subactions_override,
                    deliberation_cost=deliberation_cost
                )
        else:
            # Default: DialogueBERT (baseline, 149-d)
            env = MuseumDialogueEnv(
                knowledge_graph_path=knowledge_graph_path,
                max_turns=max_turns,
                options_override=options_override,
                subactions_override=subactions_override,
                deliberation_cost=deliberation_cost
            )
        
        return env
        
    def run_training(self):
        """Execute the complete training loop"""
        # Record start time for overnight session tracking
        self.start_time = time.time()
        
        # Show configuration
        print("=" * 80)
        print(self.training_title)
        print("=" * 80)
        print(f"Episodes: {self.max_episodes} | Max Turns: {self.max_turns_per_episode} | Device: {self.device}")
        print(f"Exhibits: {len(self.env.exhibit_keys)} | Checkpoints: Every 50 episodes")
        print(f"Started: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        if not self.verbose:
            print("\n[Progress Mode: Episode-level tracking. Use --verbose for turn details]\n")
        
        try:
            # Run training episodes
            for episode in range(self.max_episodes):
                self.current_episode = episode + 1
                
                # Start episode tracking
                self.progress_tracker.start_episode(self.current_episode)
                
                if self.verbose:
                    print(f"\n[{self.experiment_name}] [EPISODE] {self.current_episode}/{self.max_episodes}")
                    print("-" * 40)
                
                # Run single episode
                episode_reward, episode_length, episode_time, episode_timing = self._run_episode()
                self.episode_rewards.append(episode_reward)
                
                # Get coverage information for this episode
                coverage = self.env._get_museum_exhibit_coverage()
                
                # End episode tracking (prints summary)
                self.progress_tracker.end_episode(episode_reward, episode_length, episode_time, coverage_dict=coverage, timing=episode_timing)
                
                if self.verbose:
                    # Print verbose episode summary
                    self._print_episode_summary(episode_reward)
                
                # Progress report every 50 episodes for overnight monitoring
                if self.current_episode % 50 == 0:
                    # Always print progress report (not just in verbose mode)
                    self._print_progress_report()
                    
                    # Save incremental checkpoint (metrics + model)
                    self._save_checkpoint(f"checkpoint_ep{self.current_episode}")
                
                # Check for early termination
                if self._should_terminate_early():
                    print("\n[STOP] Early termination condition met")
                    break
            
            # Finalize training
            self._finalize_training()
            
        except KeyboardInterrupt:
            print("\n[WARNING] Training interrupted by user")
            self._finalize_training()
        except Exception as e:
            # Check if it's an LLM critical error
            from src.utils.llm_handler import LLMCriticalError
            if isinstance(e, LLMCriticalError):
                print("\n" + "=" * 80)
                print("CRITICAL LLM ERROR - GRACEFUL SHUTDOWN")
                print("=" * 80)
                print(f"\nError: {e}")
                print(f"\nTraining stopped at Episode {self.current_episode}/{self.max_episodes}")
                print(f"Saving model and metrics up to this point...\n")
                self._finalize_training()
                print("\n[SUCCESS] Model and metrics saved successfully!")
                print("Training can resume from this checkpoint later.")
                print("=" * 80)
            else:
                print(f"\n[ERROR] Training failed with error: {e}")
                import traceback
                traceback.print_exc()
                self._finalize_training()

    def _run_episode(self) -> tuple:
        """Run a single training episode"""
        # Track episode start time for efficiency metrics
        episode_start_time = time.time()
        
        # Initialize episode
        obs, info = self.env.reset()
        self.simulator.initialize_session(persona="Agreeable")
        
        # ===== INJECT INTRODUCTION EXCHANGE =====
        # Add scripted greeting and response before RL turns begin
        # This ensures Turn 1 has proper dialogue context
        # Note: inject_introduction only exists on StateMachineSimulator, not Sim8Simulator
        if hasattr(self.simulator, 'inject_introduction'):
            starting_exhibit = self.simulator.get_current_aoi()
            intro = self.simulator.inject_introduction(starting_exhibit)
            self.env.set_initial_dialogue(intro["agent_greeting"], intro["user_response"])
        
        # Start detailed logging for this episode
        if self.detailed_logger:
            self.detailed_logger.start_episode(self.current_episode)
        
        # Sync initial simulator state to environment
        # This ensures the agent sees the correct starting exhibit focus
        self._update_environment_state()
        obs = self.env._get_obs()  # Get fresh observation with correct focus
        
        # Reset tracker for new episode
        
        # Reset agent if using Actor-Critic
        if self.use_actor_critic:
            self.agent.reset()
            self.episode_buffer = {
                'states': [],
                'options': [],
                'subactions': [],
                'rewards': [],
                'next_states': [],
                'dones': [],
                'available_subactions': [],
                'beta_targets': []  # H1 Advanced: heuristic termination targets
            }
        
        # Track option switches and durations for this episode
        self.episode_option_sequence = []  # List of (option_name, duration)
        self.current_option_start_turn = 1
        self.previous_option = None
        
        # Track advanced HRL metrics for this episode
        self.episode_policy_confidences = []  # max(π) for each action
        self.episode_greedy_actions = []      # Was action greedy? (True/False)
        self.episode_termination_advantages = []  # A_O when terminating
        
        # ===== LIVE MONITOR: Episode Start =====
        self.live_monitor.on_episode_start(
            episode_num=self.current_episode,
            persona=self.simulator.current_persona,
            exhibits=self.env.exhibit_keys
        )
        
        # ===== MAP VISUALIZER: Reset for Episode =====
        self.map_visualizer.reset()
        
        # Print Groq API status at start of episode
        # Every 10 episodes, show detailed API health stats
        if self.current_episode % 10 == 0 or self.current_episode == 1:
            from src.utils.llm_handler import get_llm_handler
            llm = get_llm_handler()
            stats = llm.get_api_stats()
            print(f"[Groq API] Episode {self.current_episode} - API Health: "
                  f"{stats['total_calls']} calls, {stats['total_retries']} retries "
                  f"({stats['retry_rate']:.1%} rate), "
                  f"avg {stats['avg_time_per_call']:.2f}s/call, "
                  f"max {stats['max_consecutive_failures']} consecutive failures")
        else:
            print(f"[Groq API] Groq API okay! (Episode {self.current_episode})")
        
        episode_reward = 0.0
        turn_count = 0
        
        # Track reward components for this episode
        episode_reward_components = {
            "engagement": 0.0,
            "novelty": 0.0,
            "responsiveness": 0.0,
            "transition": 0.0,
            "conclude": 0.0
        }
        
        # Track component timing for this episode
        episode_timing = {
            "bert_time": 0.0,
            "agent_llm_time": 0.0,
            "agent_template_time": 0.0,
            "simulator_llm_time": 0.0,
            "env_step_time": 0.0,
            "nn_forward_time": 0.0,
        }
        
        if self.verbose:
            print("=" * 80)
            print(" " * 25 + "EPISODE INITIALIZATION" + " " * 25)
            print("=" * 80)
            print(f"   [PERSONA] Simulator Persona: Agreeable")
            print(f"   [AOI] Starting AOI: {self.simulator.get_current_aoi()}")
            print(f"   [GOAL] Episode Goal: Explore exhibits and engage visitor")
            print(f"   [EXHIBITS] Available Exhibits: {', '.join(self.env.exhibit_keys)}")
            print("═" * 82)
            print()
            print(f"   [START] Starting episode loop (max {self.max_turns_per_episode} turns)...")
        
        # Episode loop
        while turn_count < self.max_turns_per_episode:
            turn_count += 1
            
            # Verbose: Print turn header and current state FIRST (before action selection)
            if self.verbose:
                print("-" * 80)
                print(f" TURN {turn_count:2d} " + "-" * 66)
                print("-" * 80)
                self._print_state_vector(obs)
            
            # Update environment with simulator state
            self._update_environment_state()
            
            # Generate agent action (Actor-Critic or random)
            if self.use_actor_critic:
                action = self._generate_action_actor_critic(obs)
            else:
                action = self._generate_action_random()
            
            # Duration limit tracking (Sutton et al. 1999) - add before env.step
            if not hasattr(self, 'steps_in_current_option'):
                self.steps_in_current_option = 0
                self.last_option_name = None
            
            # Get option name from action
            option_name = action.get('option', None)
            
            # Track option switches and durations for metrics
            if option_name != self.previous_option and self.previous_option is not None:
                # Option switched! Record duration of previous option
                duration = turn_count - self.current_option_start_turn
                self.episode_option_sequence.append((self.previous_option, duration))
                self.current_option_start_turn = turn_count
            self.previous_option = option_name
            
            # Increment or reset counter
            if option_name == self.last_option_name:
                self.steps_in_current_option += 1
            else:
                self.steps_in_current_option = 1
                self.last_option_name = option_name
            
            # Apply duration limit if configured (default: None = disabled)
            if self.max_option_duration and self.steps_in_current_option >= self.max_option_duration:
                action['terminate_option'] = True
                if self.verbose:
                    print(f"   [DURATION LIMIT] Forced termination after {self.steps_in_current_option} steps in {option_name}")
                # CRITICAL FIX: Reset counter so if agent re-selects this option, it starts fresh
                self.steps_in_current_option = 0
                self.last_option_name = None
            
            # Execute environment step with timing
            _step_start = time.time()
            next_obs, reward, done, truncated, info = self.env.step(action)
            episode_timing["env_step_time"] += time.time() - _step_start
            
            # Collect component timing from environment
            episode_timing["bert_time"] += getattr(self.env, '_last_bert_time', 0.0)
            episode_timing["agent_llm_time"] += info.get('agent_llm_time', 0.0)
            episode_timing["agent_template_time"] += getattr(self.env, '_last_agent_template_time', 0.0)
            
            # H1 Advanced: Calculate and add diversity reward (Kamat & Precup 2020)
            diversity_reward = 0.0
            if self.diversity_reward_coef > 0.0:
                # Track option usage in sliding window
                option_name = info.get('option', 'Unknown')
                self.option_usage_window.append(option_name)
                if len(self.option_usage_window) > self.option_usage_window_size:
                    self.option_usage_window.pop(0)
                
                # Calculate option entropy: H(π_Ω) = -Σ p(ω) log p(ω)
                if len(self.option_usage_window) >= 4:  # Need at least a few samples
                    from collections import Counter
                    import math
                    option_counts = Counter(self.option_usage_window)
                    total = len(self.option_usage_window)
                    option_probs = [count / total for count in option_counts.values()]
                    # Calculate entropy
                    option_entropy = -sum(p * math.log(p) for p in option_probs if p > 0)
                    # Diversity reward: r_div = w_div × H(π_Ω)
                    diversity_reward = self.diversity_reward_coef * option_entropy
                    # Add to reward
                    reward += diversity_reward
                    # Store in info for logging
                    info['reward_diversity'] = diversity_reward
                    info['option_entropy'] = option_entropy
            
            # Update progress tracker with turn info (no printing)
            option = info.get('option', 'Unknown')
            subaction = info.get('subaction', None)
            self.progress_tracker.update_turn(reward, option, subaction)
            
            # Track LLM timing
            agent_time = info.get('agent_llm_time', 0.0)
            if agent_time > 0:
                self.agent_llm_times.append(agent_time)
            
            # Store experience in buffer for Actor-Critic training
            if self.use_actor_critic:
                self.episode_buffer['states'].append(obs)
                self.episode_buffer['options'].append(action['option'])
                self.episode_buffer['subactions'].append(action['subaction'])
                self.episode_buffer['rewards'].append(reward)  # Reward now includes diversity bonus
                self.episode_buffer['next_states'].append(next_obs)
                self.episode_buffer['dones'].append(done)
                available_subs = info.get('available_subactions', [])
                self.episode_buffer['available_subactions'].append(available_subs)
                # H1 Advanced: Store heuristic termination target for beta supervision
                beta_target = info.get('beta_target', 0.0)
                self.episode_buffer['beta_targets'].append(beta_target)
            
            # Verbose: Print agent decision and utterance (after action execution)
            if self.verbose:
                self._print_agent_decision(action, info)
            
            # Update simulator with agent response (happens after agent speaks)
            sim_llm_times_before = len(self.simulator_llm_times)
            self._update_simulator_state(info)
            
            # Collect simulator timing (after _update_simulator_state processes it)
            if len(self.simulator_llm_times) > sim_llm_times_before:
                episode_timing["simulator_llm_time"] += self.simulator_llm_times[-1]
            
            # Facts already extracted in env.step(), just retrieve from info
            # Add DialogueBERT insights after user state is updated
            self.env.add_dialoguebert_insights_to_info(info)
            
            # Log training step
            simulator_data = self._get_simulator_data()
            self.monitor.update_training_step(
                state=obs,
                action=action,
                reward=reward,
                done=done,
                info=info,
                simulator_data=simulator_data
            )
            
            # Update episode state
            episode_reward += reward
            
            # Accumulate reward components for this episode
            episode_reward_components["engagement"] += info.get("reward_engagement", 0.0)
            episode_reward_components["novelty"] += info.get("reward_novelty", 0.0)
            episode_reward_components["responsiveness"] += info.get("reward_responsiveness", 0.0)
            # Sum all transition-related rewards
            episode_reward_components["transition"] += (
                info.get("reward_transition_insufficiency", 0.0) +
                info.get("reward_transition_exploration", 0.0) +
                info.get("reward_transition_sufficiency", 0.0) +
                info.get("reward_transition_frequency", 0.0)
            )
            episode_reward_components["conclude"] += info.get("reward_conclude", 0.0)
            
            # ===== LIVE MONITOR & METRICS: Turn Update =====
            # Get simulator data for turn tracking
            simulator_state = self.simulator.get_current_state()
            user_response = simulator_state.get("last_user_response", {})
            
            turn_data = {
                "agent_utterance": info.get("agent_utterance", ""),
                "user_utterance": user_response.get("utterance", ""),
                "option": info.get("option", "Unknown"),
                "subaction": info.get("subaction", "Unknown"),
                "dwell": user_response.get("gaze_features", [0.0])[0],
                "response_type": user_response.get("response_type", "unknown"),
                "current_exhibit": info.get("current_exhibit", "Unknown"),
                "facts_shared": info.get("facts_shared", 0),
                "reward_engagement": info.get("reward_engagement", 0.0),
                "reward_novelty": info.get("reward_novelty", 0.0),
                "reward_responsiveness": info.get("reward_responsiveness", 0.0),
                "reward_transition_insufficiency": info.get("reward_transition_insufficiency", 0.0),
                "reward_transition_exploration": info.get("reward_transition_exploration", 0.0),
                "reward_transition_sufficiency": info.get("reward_transition_sufficiency", 0.0),
                "reward_transition_frequency": info.get("reward_transition_frequency", 0.0),
                "reward_conclude": info.get("reward_conclude", 0.0),
                "total_reward": reward,
                "exhibit_coverage": self.env._get_museum_exhibit_coverage()
            }
            
            # Update live monitor
            self.live_monitor.on_turn(turn_data)
            
            # Update metrics tracker
            self.metrics_tracker.update_turn(turn_data)
            
            # Log detailed episode data
            if self.detailed_logger:
                # Get prompts from environment and simulator
                agent_prompt = getattr(self.env, '_last_llm_prompt', None)
                agent_system_prompt = getattr(self.env, '_last_agent_system_prompt', None)
                simulator_prompt = getattr(self.simulator, '_last_simulator_prompt', None)
                simulator_system_prompt = getattr(self.simulator, '_last_simulator_system_prompt', None)
                
                self.detailed_logger.log_turn(
                    turn_num=turn_count,
                    state=obs,
                    action=action,
                    reward=reward,
                    info=info,
                    next_state=next_obs,
                    agent_prompt=agent_prompt,
                    agent_system_prompt=agent_system_prompt,
                    simulator_prompt=simulator_prompt,
                    simulator_system_prompt=simulator_system_prompt,
                    user_response=user_response
                )
            
            # ===== MAP VISUALIZER: Update Turn =====
            if self.enable_map_viz:
                agent_exhibit = info.get("current_exhibit", "Unknown")
                visitor_exhibit = simulator_state.get("current_exhibit", agent_exhibit)
                current_dwell = user_response.get("gaze_features", [0.0])[0]
                current_option = info.get("option", "Unknown")
                current_subaction = info.get("subaction", "Unknown")
                
                # Get completion rates for all exhibits
                coverage = self.env._get_museum_exhibit_coverage()
                exhibit_completion = {
                    exhibit_name: coverage.get(exhibit_name, {"coverage": 0.0})["coverage"]
                    for exhibit_name in self.env.exhibit_keys
                }
                
                # Only capture frames for episodes that will be saved (massive speedup)
                should_capture = (self.current_episode % self.map_interval == 0) or self.save_map_frames
                
                if should_capture:
                    self.map_visualizer.update(
                        agent_exhibit=agent_exhibit,
                        visitor_exhibit=visitor_exhibit,
                        dwell=current_dwell,
                        turn_num=turn_count,
                        option=current_option,
                        subaction=current_subaction,
                        exhibit_completion=exhibit_completion,
                        turn_reward=reward  # Add total reward display
                    )
                    
                    # Capture frame for animation
                    self.map_visualizer.capture_frame()
                    
                    # Save snapshot if requested
                    if self.save_map_frames:
                        # Save every turn when --save-map-frames is enabled
                        episode_dir = f"episode_{self.current_episode:03d}"
                        self.map_visualizer.save_snapshot(
                            f"{episode_dir}/turn_{turn_count:02d}.png"
                        )
            
            obs = next_obs
            
            # Verbose: Print rest of turn status (simulator response, rewards, facts, stats)
            if self.verbose:
                self._print_turn_details(reward, info)
            
            # Check episode termination
            if done:
                if self.verbose:
                    print("=" * 80)
                    print(f"[{self.experiment_name}] " + " " * 24 + "EPISODE COMPLETED" + " " * 24)
                    print("=" * 80)
                    print(f"   [DONE] Episode finished after {turn_count} turns")
                    print(f"   [REWARD] Final reward: {episode_reward:.3f}")
                    print()
                
                # End detailed logging
                if self.detailed_logger:
                    episode_stats = {
                        "total_turns": turn_count,
                        "episode_reward": episode_reward,
                        "avg_reward_per_turn": episode_reward / turn_count if turn_count > 0 else 0.0
                    }
                    self.detailed_logger.end_episode(episode_reward, episode_stats)
                
                break
        
        # End detailed logging if episode ended without done flag (max turns reached)
        if self.detailed_logger and turn_count >= self.max_turns_per_episode:
            episode_stats = {
                "total_turns": turn_count,
                "episode_reward": episode_reward,
                "avg_reward_per_turn": episode_reward / turn_count if turn_count > 0 else 0.0
            }
            self.detailed_logger.end_episode(episode_reward, episode_stats)
        
        # Track episode time for efficiency metrics
        episode_time = time.time() - episode_start_time
        self.metrics_tracker.update_training_efficiency(time_per_ep=episode_time)
        
        # Calculate samples per second
        if episode_time > 0:
            samples_per_sec = turn_count / episode_time
            self.metrics_tracker.update_training_efficiency(samples_per_sec=samples_per_sec)
        
        # Track updates per episode
        num_updates = 1 if self.use_actor_critic and len(self.episode_buffer['states']) > 0 else 0
        if hasattr(self.metrics_tracker, 'updates_per_episode'):
            self.metrics_tracker.updates_per_episode.append(num_updates)
        
        # Train Actor-Critic agent on collected experience
        train_stats = None
        if self.use_actor_critic and len(self.episode_buffer['states']) > 0:
            train_stats = self.trainer.update(
                states=self.episode_buffer['states'],
                options=self.episode_buffer['options'],
                subactions=self.episode_buffer['subactions'],
                rewards=self.episode_buffer['rewards'],
                next_states=self.episode_buffer['next_states'],
                dones=self.episode_buffer['dones'],
                available_subactions=self.episode_buffer['available_subactions'],
                episode=self.current_episode,  # Pass episode number for decay and scheduling
                beta_targets=self.episode_buffer['beta_targets']  # H1 Advanced: heuristic termination targets
            )
            
            # Update target network periodically (every target_update_interval episodes)
            if self.use_actor_critic and hasattr(self.trainer, 'use_target_network') and self.trainer.use_target_network:
                if self.current_episode % self.trainer.target_update_interval == 0:
                    self.trainer.update_target_network()
                    if self.verbose:
                        print(f"   [TRAINING] Target network updated at episode {self.current_episode}")
            
            # Update metrics tracker with RL training stats
            self.metrics_tracker.update_training_stats(train_stats)
            
            # Track detailed HRL-specific metrics per episode (Bacon et al. 2017)
            if 'option_advantages_all' in train_stats:
                self.metrics_tracker.episode_option_advantages.append({
                    'mean': train_stats['option_advantage_mean'],
                    'std': train_stats['option_advantage_std'],
                    'values': train_stats['option_advantages_all']
                })
            # Track per-option advantages (all options) for collapse analysis
            if 'all_option_advantages' in train_stats:
                all_opt_advs = np.array(train_stats['all_option_advantages'])  # Shape: (batch_size, num_options)
                # Compute per-option statistics
                per_option_stats = {}
                for opt_idx in range(all_opt_advs.shape[1]):
                    opt_advs = all_opt_advs[:, opt_idx]
                    per_option_stats[f'option_{opt_idx}'] = {
                        'mean': float(np.mean(opt_advs)),
                        'std': float(np.std(opt_advs)),
                        'min': float(np.min(opt_advs)),
                        'max': float(np.max(opt_advs))
                    }
                # Store in episode-level tracking
                if not hasattr(self.metrics_tracker, 'episode_per_option_advantages'):
                    self.metrics_tracker.episode_per_option_advantages = []
                self.metrics_tracker.episode_per_option_advantages.append(per_option_stats)
            if 'termination_probs' in train_stats:
                self.metrics_tracker.episode_termination_probs.append({
                    'mean': np.mean(train_stats['termination_probs']),
                    'values': train_stats['termination_probs']
                })
            if 'action_advantages_all' in train_stats:
                self.metrics_tracker.episode_intra_advantages.append({
                    'mean': train_stats['action_advantage_mean'],
                    'std': train_stats['action_advantage_std'],
                    'values': train_stats['action_advantages_all']
                })
            if 'option_qvalues' in train_stats:
                self.metrics_tracker.episode_option_qvalues.append({
                    'mean': train_stats['mean_option_value'],
                    'values': train_stats['option_qvalues']
                })
            # Track per-option Q-values (all options) for collapse analysis
            if 'all_option_qvalues' in train_stats:
                all_opt_qvals = np.array(train_stats['all_option_qvalues'])  # Shape: (batch_size, num_options)
                per_option_qvals = {}
                for opt_idx in range(all_opt_qvals.shape[1]):
                    opt_qvals = all_opt_qvals[:, opt_idx]
                    per_option_qvals[f'option_{opt_idx}'] = {
                        'mean': float(np.mean(opt_qvals)),
                        'std': float(np.std(opt_qvals)),
                        'min': float(np.min(opt_qvals)),
                        'max': float(np.max(opt_qvals))
                    }
                if not hasattr(self.metrics_tracker, 'episode_per_option_qvalues'):
                    self.metrics_tracker.episode_per_option_qvalues = []
                self.metrics_tracker.episode_per_option_qvalues.append(per_option_qvals)
            
            # Log training statistics
            if self.verbose:
                print(f"   [TRAINING] Training Update:")
                print(f"      Policy Loss: {train_stats['policy_loss']:.4f}")
                print(f"      Value Loss: {train_stats['value_loss']:.4f}")
                print(f"      Entropy: {train_stats['entropy']:.4f}")
                print(f"      Mean Advantage: {train_stats['mean_advantage']:.4f}")
                print()
        
        # ===== LIVE MONITOR & METRICS: Episode End =====
        # Count exhibits covered
        exhibits_covered = sum(1 for fact_set in self.env.facts_mentioned_per_exhibit.values() 
                              if len(fact_set) > 0)
        total_facts = sum(len(fact_set) for fact_set in self.env.facts_mentioned_per_exhibit.values())
        
        # Calculate coverage ratio
        total_available_facts = sum(len(self.env.knowledge_graph.get_exhibit_facts(ex)) 
                                   for ex in self.env.exhibit_keys)
        coverage_ratio = total_facts / total_available_facts if total_available_facts > 0 else 0.0
        
        # Record final option duration
        if hasattr(self, 'previous_option') and self.previous_option is not None:
            final_duration = turn_count - self.current_option_start_turn + 1
            self.episode_option_sequence.append((self.previous_option, final_duration))
        
        # Compute option switches and duration stats
        num_switches = len(self.episode_option_sequence) - 1 if len(self.episode_option_sequence) > 0 else 0
        option_durations = {}
        for option, duration in self.episode_option_sequence:
            if option not in option_durations:
                option_durations[option] = []
            option_durations[option].append(duration)
        
        # Compute advanced HRL metrics for this episode
        policy_confidence = np.mean(self.episode_policy_confidences) if self.episode_policy_confidences else 0.5
        exploration_rate = (1.0 - np.mean(self.episode_greedy_actions)) * 100 if self.episode_greedy_actions else 0.0
        termination_effectiveness = np.mean(self.episode_termination_advantages) if self.episode_termination_advantages else 0.0
        
        # Build episode summary
        episode_summary = {
            "cumulative_reward": episode_reward,
            "turns": turn_count,
            "coverage_ratio": coverage_ratio,
            "total_facts": total_facts,
            "exhibits_covered": exhibits_covered,
            "mean_value": train_stats.get('mean_value', 0.0) if train_stats else 0.0,
            "reward_engagement": episode_reward_components["engagement"],
            "reward_novelty": episode_reward_components["novelty"],
            "reward_responsiveness": episode_reward_components["responsiveness"],
            "reward_transition": episode_reward_components["transition"],
            "reward_conclude": episode_reward_components["conclude"],
            "option_switches": num_switches,
            "option_durations": option_durations,
            # Advanced HRL metrics
            "policy_confidence": policy_confidence,
            "exploration_rate": exploration_rate,
            "termination_effectiveness": termination_effectiveness
        }
        
        # Update live monitor
        self.live_monitor.on_episode_end(episode_summary)
        
        # Update metrics tracker
        self.metrics_tracker.update_episode(episode_summary)
        
        # Update trainer with latest OCI for adaptive entropy control
        if hasattr(self, 'metrics_tracker') and len(self.metrics_tracker.option_collapse_index) > 0:
            latest_oci = self.metrics_tracker.option_collapse_index[-1]
            self.trainer.update_oci(latest_oci)
        
        # ===== MAP VISUALIZER: Save Episode Animation (at specified interval) =====
        if self.enable_map_viz and (self.current_episode % self.map_interval == 0):
            animation_filename = f"episode_{self.current_episode:03d}_animation.gif"
            self.map_visualizer.save_animation(animation_filename, fps=2)
            print(f"   [MAP] Map animation saved: training_logs/maps/{animation_filename}")
        elif self.enable_map_viz:
            print(f"   [MAP] Map episode {self.current_episode} (animation saved every {self.map_interval} episodes)")
        
        # ===== EPSILON DECAY (H6 exploration fix) =====
        # Decay epsilon for epsilon-greedy exploration after each episode
        if hasattr(self.agent, 'decay_epsilon'):
            self.agent.decay_epsilon()
            if self.verbose and self.current_episode % 50 == 0:
                print(f"   [EXPLORATION] Epsilon: {self.agent.get_epsilon():.3f}")
        
        return episode_reward, turn_count, episode_time, episode_timing

    def _update_environment_state(self):
        """Update environment with current simulator state"""
        # Get current exhibit from simulator and map to focus index
        current_exhibit = self.simulator.get_current_aoi()
        focus = 0
        if current_exhibit in self.env.exhibit_keys:
            focus = self.env.exhibit_keys.index(current_exhibit) + 1
        
        # Update environment with available simulator data
        # NOTE: Don't clear utterance - it should persist from the previous turn's user response
        # The utterance will be updated later when the simulator responds to the agent
        self.env.update_user_state(
            focus=focus
            # utterance is NOT cleared here - it should carry over from previous turn
        )
        
    def _generate_action_actor_critic(self, obs: np.ndarray) -> Dict[str, Any]:
        """Generate agent action using Actor-Critic policy"""
        # ===== TESTING MODE: Force specific actions =====
        if self.force_option or self.force_subaction:
            available_options = self.env._get_available_options()
            
            # Determine which option to use
            if self.force_option:
                if self.force_option not in available_options:
                    print(f"[WARNING] Force option '{self.force_option}' not available. "
                          f"Available: {available_options}")
                    option_idx = available_options.index(available_options[0])
                    option = available_options[0]
                else:
                    option = self.force_option
                    option_idx = available_options.index(self.force_option)
            else:
                option = available_options[0]
                option_idx = 0
            
            # Determine which subaction to use
            available_subactions = self.env._get_available_subactions(option)
            if self.force_subaction:
                if self.force_subaction not in available_subactions:
                    print(f"[WARNING] Force subaction '{self.force_subaction}' not available for '{option}'. "
                          f"Available: {available_subactions}")
                    subaction_idx = 0
                    subaction = available_subactions[0]
                else:
                    subaction = self.force_subaction
                    subaction_idx = available_subactions.index(self.force_subaction)
            else:
                subaction_idx = 0
                subaction = available_subactions[0]
            
            print(f"[FORCED] FORCED ACTION: Option={option}, Subaction={subaction}")
            
            return {
                "option": option_idx,
                "subaction": subaction_idx,
                "terminate_option": False
            }
        
        # ===== NORMAL MODE: Use Actor-Critic policy =====
        # Get available options and subactions
        available_options = self.env._get_available_options()
        if not available_options:
            return {"option": 0, "subaction": 0, "terminate_option": False}
        
        # Build available subactions dict
        available_subactions_dict = {}
        for opt in available_options:
            available_subactions_dict[opt] = self.env._get_available_subactions(opt)
        
        # Select action using Actor-Critic agent
        action_info = self.agent.select_action(
            state=obs,
            available_options=available_options,
            available_subactions_dict=available_subactions_dict,
            deterministic=False  # Use stochastic policy during training
        )
        
        return {
            "option": action_info['option'],
            "subaction": action_info['subaction'],
            "terminate_option": action_info['terminated']
        }
    
    def _generate_action_random(self) -> Dict[str, Any]:
        """Generate random action (baseline/fallback)"""
        # Get available options and subactions
        available_options = self.env._get_available_options()
        if not available_options:
            return {"option": 0, "subaction": 0, "terminate_option": False}
        
        # Select random option
        option = random.choice(available_options)
        option_idx = self.env.options.index(option)
        
        # Get available subactions for selected option
        available_subactions = self.env._get_available_subactions(option)
        if not available_subactions:
            return {"option": option_idx, "subaction": 0, "terminate_option": False}
        
        # Select random subaction
        subaction = random.choice(available_subactions)
        subaction_idx = self.env.subactions[option].index(subaction)
        
        # Random termination (10% chance)
        terminate_option = random.random() < 0.1
        
        return {
            "option": option_idx,
            "subaction": subaction_idx,
            "terminate_option": terminate_option
        }

    def _update_simulator_state(self, info: Dict[str, Any]):
        """Update simulator with agent response"""
        agent_utterance = info.get("agent_utterance", "")
        agent_option = info.get("option", None)  # Get agent's chosen option
        agent_subaction = info.get("subaction", None)
        target_exhibit = info.get("target_exhibit", None)  # For OfferTransition
        current_exhibit_completion = info.get("current_exhibit_completion", 0.0)  # For transition probability
        exhibit_exhausted = info.get("exhibit_exhausted", False)  # Whether exhibit has no new facts
        target_exhibit_completion = info.get("target_exhibit_completion", 0.0)  # Target exhibit completion
        target_exhibit_exhausted = info.get("target_exhibit_exhausted", False)  # Whether target is exhausted
        
        # FIX: Map coarse options to "OfferTransition" for simulator transition logic
        # Handles both:
        # - coarse (2-opt): "Engage" + subaction "SuggestMove" -> "OfferTransition"
        # - coarse_3opt (3-opt): "Transition" option -> "OfferTransition"
        effective_option = agent_option
        if agent_option == "Transition":
            # H6 coarse_3opt: "Transition" option maps directly to "OfferTransition"
            effective_option = "OfferTransition"
        elif agent_subaction == "SuggestMove" and agent_option != "OfferTransition":
            # H6 coarse (2-opt): "SuggestMove" under "Engage" maps to "OfferTransition"
            effective_option = "OfferTransition"
        
        if agent_utterance:
            # Generate user response (now with transition logic)
            user_response = self.simulator.generate_user_response(
                agent_utterance, 
                agent_option=effective_option,  # Use effective option for transition logic
                agent_subaction=agent_subaction,
                target_exhibit=target_exhibit,
                current_exhibit_completion=current_exhibit_completion,
                exhibit_exhausted=exhibit_exhausted,
                target_exhibit_completion=target_exhibit_completion,
                target_exhibit_exhausted=target_exhibit_exhausted
            )
            
            # Track simulator LLM timing
            sim_time = user_response.get('simulator_llm_time', 0.0)
            if sim_time > 0:
                self.simulator_llm_times.append(sim_time)

            # Update environment with utterance and response type
            if user_response.get("utterance") is not None:
                self.env.update_user_state(
                    utterance=user_response["utterance"],
                    response_type=user_response.get("response_type", "statement"),
                    visitor_state=user_response.get("visitor_state")  # For agent template responsiveness
                )

            # Dwell time reward: take first gaze feature as dwell
            gaze_feats = user_response.get("gaze_features") or []
            if gaze_feats:
                dwell_time = float(gaze_feats[0])
                self.env.update_user_state(dwell=dwell_time)
            
            # Update focus based on simulator's new exhibit (if changed)
            new_exhibit = self.simulator.get_current_aoi()
            new_aoi = user_response.get("aoi", new_exhibit)  # Get AOI from response
            if new_exhibit in self.env.exhibit_keys:
                new_focus = self.env.exhibit_keys.index(new_exhibit) + 1
                self.env.update_user_state(focus=new_focus)
            
            # Check for transition success using effective option (handles coarse config)
            is_transition = info.get("option") == "OfferTransition" or info.get("subaction") == "SuggestMove"
            if is_transition:
                success = user_response.get("transition_success", False)
                if success:
                    self.env.record_successful_transition()

    def _get_simulator_data(self) -> Dict[str, Any]:
        """Get current simulator data for logging"""
        return {
            "aoi": self.simulator.get_current_aoi(),
            "persona": getattr(self.simulator, 'current_persona', 'Unknown'),
            "utterance": "",
            "dwell_time": 0.5,
            "engagement_level": 0.5
        }

    def _print_state_vector(self, state: np.ndarray):
        """Print state vector at the beginning of turn (before action selection)"""
        if state is None:
            return
        
        # Calculate dimensions dynamically based on environment
        n_exhibits = self.env.n_exhibits
        focus_dim = n_exhibits + 1
        history_dim = n_exhibits + len(self.env.options)
        intent_dim = 64
        context_dim = 64
        total_dim = focus_dim + history_dim + intent_dim + context_dim
        
        print(f"[STATE] STATE VECTOR ({total_dim}-d) [Before Action Selection]:")
        
        # Focus vector
        focus_end = focus_dim
        focus_vec = state[0:focus_end]
        focus_idx = np.argmax(focus_vec)
        if focus_idx == n_exhibits:  # Last index is "no focus"
            focus_name = "No focus"
        else:
            focus_name = self.env.exhibit_keys[focus_idx]
        print(f"   [FOCUS] Focus (0-{focus_end-1}): {focus_vec.round(2)} -> {focus_name}")
        
        # History vector - Exhibit completion ratios
        completion_start = focus_end
        completion_end = completion_start + n_exhibits
        completion_vec = state[completion_start:completion_end]
        print(f"   [COMPLETION] Completion ({completion_start}-{completion_end-1}):")
        for i, exhibit_name in enumerate(self.env.exhibit_keys):
            print(f"      [{i}] {exhibit_name:15s}: {completion_vec[i]:.2f}")
        
        # History vector - Option usage
        option_start = completion_end
        option_end = option_start + len(self.env.options)
        option_vec = state[option_start:option_end]
        print(f"   [OPTIONS] Option Usage ({option_start}-{option_end-1}):")
        for i, opt_name in enumerate(self.env.options):
            print(f"      [{i}] {opt_name:15s}: {option_vec[i]:.3f}")
        
        # Intent embedding
        intent_start = option_end
        intent_end = intent_start + intent_dim
        intent_vec = state[intent_start:intent_end]
        print(f"   [INTENT] Intent Emb ({intent_start}-{intent_end-1}): [{intent_vec[0]:.3f}, {intent_vec[1]:.3f}, ..., {intent_vec[-1]:.3f}] ({intent_dim}-d)")
        
        # Context embedding (may be empty for state ablation experiments)
        context_start = intent_end
        context_end = context_start + context_dim
        if context_end <= len(state):
            context_vec = state[context_start:context_end]
            if len(context_vec) > 0:
                print(f"   [CONTEXT] Context Emb ({context_start}-{context_end-1}): [{context_vec[0]:.3f}, {context_vec[1]:.3f}, ..., {context_vec[-1]:.3f}] ({context_dim}-d)")
            else:
                print(f"   [CONTEXT] Context Emb: [EMPTY] (state ablation)")
        else:
            print(f"   [CONTEXT] Context Emb: [NOT PRESENT] (state ablation)")
        print()
    
    def _print_agent_decision(self, action: Dict[str, Any], info: Dict[str, Any]):
        """Print agent decision and utterance (after action execution)"""
        option = info.get("option", "Unknown")
        subaction = info.get("subaction", "Unknown")
        agent_utterance = info.get("agent_utterance", "")
        terminated_option = info.get("terminated_option", False)
        current_option = info.get("current_option", None)
        turns_in_option = info.get("turns_in_option", 0)
        
        # Determine option status
        # Check if this is truly a new option or just continuing with different subaction
        if hasattr(self, '_previous_option_name'):
            is_same_option = (self._previous_option_name == option)
        else:
            is_same_option = (current_option == option)
        
        option_status = "[CONTINUING]" if is_same_option else "[NEW]"
        termination_text = " | [TERMINATED]" if terminated_option else ""
        
        # Store for next turn comparison
        if terminated_option:
            # Store the option name even though it terminated
            self._previous_option_name = option
        else:
            self._previous_option_name = option
        
        # AGENT DECISION SECTION
        print("[AGENT] AGENT DECISION:")
        print(f"   [OPTION] Option: {option} ({option_status}) | Turns in option: {turns_in_option}{termination_text}")
        print(f"   [SUBACTION] Subaction: {subaction}")
        print(f"   [ACTION] Raw Action: option={action.get('option', '?')}, subaction={action.get('subaction', '?')}, terminate={action.get('terminate_option', '?')}")
        print()
        
        # AGENT UTTERANCE SECTION
        print("[AGENT] AGENT UTTERANCE:")
        if agent_utterance:
            # Word wrap the utterance nicely
            words = agent_utterance.split()
            lines = []
            current_line = "   \"" 
            for word in words:
                if len(current_line + word + " ") > 75:
                    lines.append(current_line)
                    current_line = "   " + word + " "
                else:
                    current_line += word + " "
            lines.append(current_line.rstrip() + "\"")
            
            for line in lines:
                print(line)
        else:
            print("   \"[No utterance generated]\"")
        print()
        
        # Show LLM prompt (optional, can be toggled)
        llm_prompt = info.get("llm_prompt", "")
        if llm_prompt and hasattr(self, '_show_prompts') and self._show_prompts:
            print("[PROMPT] LLM PROMPT:")
            prompt_lines = llm_prompt.split('\n')
            for line in prompt_lines[:15]:  # Show first 15 lines
                if line.strip():
                    print(f"   {line[:75]}")
            if len(prompt_lines) > 15:
                print(f"   ... ({len(prompt_lines) - 15} more lines)")
            print()
    
    def _print_turn_details(self, reward: float, info: Dict[str, Any]):
        """Print simulator response, rewards, facts, and stats (happens AFTER simulator responds)"""
        current_focus = info.get("current_focus", 0)
        current_exhibit = info.get("current_exhibit", "Unknown")
        facts_shared = info.get("facts_shared", 0)
        exhibits_covered = info.get("exhibits_covered", 0)
        
        # Get simulator data
        simulator_state = self.simulator.get_current_state()
        user_response = simulator_state.get("last_user_response", {})
        # SIMULATOR RESPONSE SECTION
        print("[SIMULATOR] SIMULATOR RESPONSE:")
        user_utterance = user_response.get("utterance")
        user_aoi = user_response.get("aoi", "Unknown")
        user_persona = user_response.get("persona", "Unknown")
        response_type = user_response.get("response_type", "unknown")
        gaze_features = user_response.get("gaze_features", [])
        
        # Get visitor's current exhibit from simulator
        sim_state = self.simulator.get_current_state()
        visitor_exhibit = sim_state.get("current_exhibit", "Unknown")
        
        if user_utterance:
            print(f"   [USER] User Says: \"{user_utterance}\"")
        else:
            print(f"   [USER] User Says: [SILENT] ({response_type})")
        
        print(f"   [VISITOR] Visitor at Exhibit: {visitor_exhibit} | Looking at AOI: {user_aoi}")
        print(f"   [PERSONA] Persona: {user_persona} | Response Type: {response_type}")
        
        # Gaze features display
        if gaze_features and len(gaze_features) >= 6:
            dwell = gaze_features[0]
            saccade = gaze_features[1] 
            entropy = gaze_features[2]
            fix_rate = gaze_features[3]
            dom_ratio = gaze_features[4]
            entry_lat = gaze_features[5]
            
            print(f"   [GAZE] Gaze: Dwell={dwell:.3f} | Saccade={saccade:.3f} | Entropy={entropy:.3f}")
            print(f"           FixRate={fix_rate:.3f} | DomRatio={dom_ratio:.3f} | EntryLat={entry_lat:.3f}")
        else:
            print(f"   [GAZE] Gaze: [No gaze data available]")
        
        # Show simulator engagement metrics
        engagement_level = user_response.get("engagement_level", 1.0)
        off_topic_strikes = user_response.get("off_topic_strikes", 0)
        agent_option = user_response.get("agent_option", "Unknown")
        response_type = user_response.get("response_type", "unknown")
        transition_success = user_response.get("transition_success", None)
        
        transition_status = ""
        if agent_option == "OfferTransition" and transition_success is not None:
            transition_status = f" | Transition: {'[SUCCESS]' if transition_success else '[REJECTED]'}"
        
        print(f"   [OPTION] Agent Option: {agent_option} | Response Type: {response_type}{transition_status} | Engagement: {engagement_level:.2f} | Strikes: {off_topic_strikes}")
        print()
        
        # REWARD BREAKDOWN SECTION
        print("[REWARD] REWARD BREAKDOWN:")
        engagement_reward = info.get("reward_engagement", 0.0)
        novelty_reward = info.get("reward_novelty", 0.0)
        responsiveness_reward = info.get("reward_responsiveness", 0.0)
        conclude_bonus = info.get("reward_conclude", 0.0)
        transition_penalty = info.get("reward_transition_penalty", 0.0)
        spam_penalty = info.get("reward_spam_penalty", 0.0)
        
        print(f"   [ENGAGEMENT] Engagement:    {engagement_reward:+7.3f} (lagged from prev turn)")
        print(f"   [NOVELTY] Novelty:       {novelty_reward:+7.3f} (verified facts)")
        if responsiveness_reward != 0:
            if responsiveness_reward > 0:
                print(f"   [SUCCESS] Responsiveness: {responsiveness_reward:+7.3f} (answered question)")
            else:
                print(f"   [FAIL] Responsiveness: {responsiveness_reward:+7.3f} (deflected question)")
        if conclude_bonus > 0:
            print(f"   [CONCLUDE] Conclude:      {conclude_bonus:+7.3f} (exhibits covered)")
        if transition_penalty < 0:
            print(f"   [WARNING] Transition:    {transition_penalty:+7.3f} (insufficient facts)")
        if spam_penalty < 0:
            print(f"   [SPAM] Spam:          {spam_penalty:+7.3f} (too many transitions)")
        print(f"   [TOTAL] TOTAL:         {reward:+7.3f}")
        print()
        
        # FACT IDs EXTRACTED
        mentioned_fact_ids = info.get("facts_mentioned_in_utterance", [])
        hallucinated_fact_ids = info.get("hallucinated_facts", [])
        
        if mentioned_fact_ids or hallucinated_fact_ids:
            if mentioned_fact_ids:
                print(f"[FACTS] NEW FACTS: {len(mentioned_fact_ids)} fact(s) - {mentioned_fact_ids}")
            if hallucinated_fact_ids:
                print(f"[ERROR] HALLUCINATED: {len(hallucinated_fact_ids)} fact(s) - {hallucinated_fact_ids} (no reward)")
            print()
        
        # DialogueBERT INSIGHTS SECTION
        dialoguebert = info.get("dialoguebert_insights")
        if dialoguebert:
            print("[DIALOGUEBERT] DialogueBERT INSIGHTS:")
            print(f"   [INTENT] Intent: {dialoguebert.get('intent_category', 'unknown')}")
            print(f"   [NORM] ||intent||={dialoguebert.get('intent_norm', 0.0):.3f} | ||context||={dialoguebert.get('context_norm', 0.0):.3f}")
            print(f"   [COSINE] cos(intent, context)={dialoguebert.get('cosine_intent_context', 0.0):+.3f}")
            print(f"   [COSINE_PREV] cos(intent, prev_intent)={dialoguebert.get('cosine_intent_prev', 0.0):+.3f} | cos(context, prev_context)={dialoguebert.get('cosine_context_prev', 0.0):+.3f}")
            print()
        
        # TRAINING STATS SECTION
        print("[STATS] TRAINING STATS:")
        print(f"   [EXHIBIT] Current Exhibit: {current_exhibit} | Focus Index: {current_focus}")
        print(f"   [FACTS] Facts Shared: {facts_shared} | Exhibits Covered: {exhibits_covered}")
        
        # Show available actions for next turn
        available_options = info.get("available_options", [])
        available_subactions = info.get("available_subactions", [])
        print(f"   [ACTIONS] Next Available - Options: {available_options}")
        print(f"                      Subactions: {available_subactions}")
        
        print("─" * 82)
        print()  # Add spacing between turns
        
        # Small delay to make live viewing easier
        time.sleep(self.turn_delay)

    def _print_episode_summary(self, episode_reward: float):
        """Print detailed episode summary"""
        if not self.monitor.training_history:
            return
            
        last_turn = self.monitor.training_history[-1]
        info = last_turn['info']
        
        # Calculate episode statistics
        episode_turns = [step for step in self.monitor.training_history if step['episode'] == self.current_episode]
        total_turns = len(episode_turns)
        avg_reward_per_turn = episode_reward / total_turns if total_turns > 0 else 0.0
        
        # Count action types used in this episode
        action_counts = {}
        for turn in episode_turns:
            option = turn['info'].get('option', 'Unknown')
            action_counts[option] = action_counts.get(option, 0) + 1
        
        print(f"   [{self.experiment_name}] [SUMMARY] Episode {self.current_episode} Summary:")
        print(f"      [PERFORMANCE] Total Reward={episode_reward:.3f} | Avg/Turn={avg_reward_per_turn:.3f} | Turns={total_turns}")
        print(f"      [CONTENT] Facts Shared={info.get('facts_shared', 0)} | Exhibits Covered={info.get('exhibits_covered', 0)}")
        print(f"      [STATE] Final State: Focus={info.get('current_exhibit', 'Unknown')} | Dwell={info.get('dwell', 0.0):.2f}")
        
        # Show action distribution
        if action_counts:
            action_str = " | ".join([f"{opt}={count}" for opt, count in action_counts.items()])
            print(f"      [ACTIONS] Actions: {action_str}")
        
        # Print detailed exhibit/facts progress
        self._print_exhibit_facts_progress()
        
        print()  # Add spacing between episodes

    def _print_exhibit_facts_progress(self):
        """SIMPLE: Print which facts were mentioned per exhibit"""
        print()
        print("      " + "=" * 100)
        print(f"      [PROGRESS] EXHIBIT PROGRESS [Episode {self.current_episode}]")
        print("      " + "=" * 100)
        
        total_mentioned = 0
        total_facts = 0
        
        for exhibit in sorted(self.env.exhibit_keys):
            all_facts = self.env.knowledge_graph.get_exhibit_facts(exhibit)
            mentioned_ids = self.env.facts_mentioned_per_exhibit[exhibit]
            
            total_facts += len(all_facts)
            total_mentioned += len(mentioned_ids)
            
            print(f"\n      [EXHIBIT] {exhibit} ({len(mentioned_ids)}/{len(all_facts)} facts)")
            print(f"      {'-' * 96}")
            
            for fact in all_facts:
                fact_id = self.env.knowledge_graph.extract_fact_id(fact)
                fact_text = self.env.knowledge_graph.strip_fact_id(fact)
                if fact_id in mentioned_ids:
                    print(f"      [OK] {fact_id}  [mentioned]     {fact_text[:70]}")
                else:
                    print(f"      [ ] {fact_id}  [not mentioned] {fact_text[:70]}")
        
        print("      " + "=" * 100)
        pct = (total_mentioned / total_facts * 100) if total_facts > 0 else 0
        print(f"      [TOTAL] TOTAL: {total_mentioned}/{total_facts} facts ({pct:.1f}%)")
        print("      " + "=" * 100)
        print()

    def _should_terminate_early(self) -> bool:
        """Check if training should terminate early"""
        # DISABLED: Don't terminate early for thesis-level experiments
        # We need complete training runs for proper analysis
        return False

    def _save_checkpoint(self, checkpoint_name: str):
        """Save incremental checkpoint (metrics + model) for crash recovery - ALWAYS SAVE."""
        try:
            experiment_dir = os.environ.get('EXPERIMENT_DIR', None)
            if experiment_dir:
                checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
            else:
                checkpoint_dir = "training_logs/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save metrics (ALWAYS)
            metrics_file = os.path.join(checkpoint_dir, f"{checkpoint_name}_metrics.json")
            self.metrics_tracker.save_to_json(metrics_file)
            
            # Also save RL metrics at checkpoint
            checkpoint_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            training_time_so_far = time.time() - self.start_time if hasattr(self, 'start_time') else 0
            self._save_rl_metrics(checkpoint_dir, checkpoint_timestamp, training_time_so_far)
            
            # Save model if using Actor-Critic (ALWAYS)
            if self.use_actor_critic and hasattr(self, 'agent'):
                model_file = os.path.join(checkpoint_dir, f"{checkpoint_name}_model.pt")
                self.trainer.save_checkpoint(model_file, self.current_episode)
            
            print(f"   [SAVED] Checkpoint saved: {checkpoint_name}")
            
        except Exception as e:
            print(f"   [WARNING] Failed to save checkpoint: {e}")
    
    def _print_progress_report(self):
        """Print comprehensive progress report every 50 episodes"""
        elapsed_time = time.time() - self.start_time if hasattr(self, 'start_time') else 0
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        
        # Calculate recent performance
        recent_50 = self.episode_rewards[-50:] if len(self.episode_rewards) >= 50 else self.episode_rewards
        avg_recent = np.mean(recent_50) if recent_50 else 0
        
        # Calculate overall trend
        if len(self.episode_rewards) >= 100:
            first_50 = np.mean(self.episode_rewards[:50])
            trend = "[IMPROVING]" if avg_recent > first_50 else "[STABLE]" if abs(avg_recent - first_50) < 0.5 else "[DECLINING]"
        else:
            trend = "[BUILDING] Building baseline"
        
        # Estimate completion time
        if elapsed_time > 0 and len(self.episode_rewards) > 0:
            # Calculate average turns per episode so far
            total_turns_so_far = sum([len([h for h in self.monitor.training_history if h.get('episode') == ep]) 
                                     for ep in range(1, self.current_episode + 1)])
            avg_turns_per_episode = total_turns_so_far / self.current_episode if self.current_episode > 0 else self.max_turns_per_episode
            
            # Estimate remaining time based on turns (realistic timing)
            remaining_episodes = self.max_episodes - self.current_episode
            remaining_turns = remaining_episodes * avg_turns_per_episode
            eta_seconds = remaining_turns * 1.2  # More realistic: 1.2 seconds per turn
            eta_hours = eta_seconds / 3600
            eta_h = int(eta_hours)
            eta_m = int((eta_hours % 1) * 60)
            
            # Calculate current pace
            turns_per_hour = total_turns_so_far / (elapsed_time / 3600) if elapsed_time > 0 else 0
        else:
            eta_h = eta_m = 0
            turns_per_hour = 0
        
        print("\n" + "=" * 80)
        print(f"[{self.experiment_name}] [CHECKPOINT] PROGRESS CHECKPOINT - Episode {self.current_episode:,}/{self.max_episodes:,}")
        print("=" * 80)
        print(f"[TIME] Elapsed: {hours}h {minutes}m | ETA: {eta_h}h {eta_m}m")
        print(f"[REWARD] Recent 50 Episodes Avg Reward: {avg_recent:.3f}")
        print(f"[TREND] Learning Trend: {trend}")
        print(f"� Current Pace: {turns_per_hour:.0f} turns/hour")
        if self.map_interval > 0:
            print(f"�🗺️  Maps Saved: {(self.current_episode // self.map_interval)} animations")
        print(f"[PROGRESS] Progress: {(self.current_episode/self.max_episodes)*100:.1f}% complete")
        print("=" * 80 + "\n")

    def _finalize_training(self):
        """Finalize training and save results"""
        print("\n" + "=" * 80)
        print("[COMPLETE] TRAINING SESSION COMPLETE!")
        print("=" * 80)
        
        # Calculate total training time
        training_time = time.time() - self.start_time if hasattr(self, 'start_time') else 0
        self.training_duration_seconds = training_time  # Store for summary.json
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        
        # Update final statistics
        self.total_episodes = self.current_episode
        self.total_turns = len(self.monitor.training_history)
        
        # Get training summary
        summary = self.monitor.get_training_summary()
        
        # Print comprehensive statistics
        print(f"[PERFORMANCE] TRAINING PERFORMANCE:")
        print(f"   - Total Episodes: {summary['total_episodes']:,}")
        print(f"   - Total Turns: {summary['total_turns']:,}")
        print(f"   - Training Time: {hours}h {minutes}m {seconds}s")
        print(f"   - Episodes/Hour: {summary['total_episodes']/max(training_time/3600, 0.1):.1f}")
        print(f"   - Turns/Hour: {summary['total_turns']/max(training_time/3600, 0.1):.1f}")
        
        print(f"\n[LEARNING] LEARNING METRICS:")
        print(f"   - Avg Reward/Turn: {summary['average_reward_per_turn']:.4f}")
        print(f"   - Avg Reward/Episode: {summary['average_reward_per_episode']:.4f}")
        print(f"   - Best Episode Reward: {summary['best_episode_reward']:.4f}")
        if hasattr(self, 'episode_rewards') and len(self.episode_rewards) > 0:
            print(f"   - Final 100 Episodes Avg: {np.mean(self.episode_rewards[-100:]):.4f}")
            trend_text = "[IMPROVING]" if len(self.episode_rewards) > 100 and np.mean(self.episode_rewards[-100:]) > np.mean(self.episode_rewards[:100]) else "[STABLE/DECLINING]"
        else:
            print(f"   - Final 100 Episodes Avg: N/A (episode_rewards not available)")
            trend_text = "[N/A]"
        print(f"   - Learning Curve Trend: {trend_text}")
        
        print(f"\n[STRATEGY] STRATEGY ANALYSIS:")
        print(f"   - Most Used Option: {summary['most_used_option']}")
        print(f"   - Most Used Subaction: {summary['most_used_subaction']}")
        print(f"   - Strategy Diversity: {len(set([h['option'] for h in self.monitor.training_history if 'option' in h]))} unique options used")
        
        # Map generation summary
        maps_saved = (self.current_episode // self.map_interval) if self.enable_map_viz else 0
        print(f"\n[VISUALIZATION] VISUALIZATION:")
        print(f"   - Map Episodes Saved: {maps_saved} (every {self.map_interval} episodes)")
        print(f"   - Total Frames Captured: {len(self.map_visualizer.frames) if self.enable_map_viz else 0}")
        opt_status = "[ENABLED]" if maps_saved < self.current_episode else "[DISABLED]"
        print(f"   - Storage Optimization: {opt_status}")
        
        # Generate comprehensive RL analysis report (ENABLED for thesis)
        # Only generate if we have enough data
        if hasattr(self, 'episode_rewards') and len(self.episode_rewards) >= 50:
            self._generate_rl_analysis_report(training_time, summary)
        
        # Print LLM timing summary
        if self.agent_llm_times or self.simulator_llm_times:
            print(f"\n[TIMING] LLM Timing Summary:")
            if self.agent_llm_times:
                avg_agent = sum(self.agent_llm_times) / len(self.agent_llm_times)
                total_agent = sum(self.agent_llm_times)
                print(f"   [AGENT] Agent LLM:     {len(self.agent_llm_times)} calls | Avg: {avg_agent:.2f}s | Total: {total_agent:.1f}s")
            if self.simulator_llm_times:
                avg_sim = sum(self.simulator_llm_times) / len(self.simulator_llm_times)
                total_sim = sum(self.simulator_llm_times)
                print(f"   [SIMULATOR] Simulator LLM: {len(self.simulator_llm_times)} calls | Avg: {avg_sim:.2f}s | Total: {total_sim:.1f}s")
            if self.agent_llm_times and self.simulator_llm_times:
                total_all = sum(self.agent_llm_times) + sum(self.simulator_llm_times)
                count_all = len(self.agent_llm_times) + len(self.simulator_llm_times)
                avg_all = total_all / count_all if count_all > 0 else 0
                print(f"   [TOTAL] Total LLM:     {count_all} calls | Avg: {avg_all:.2f}s | Total: {total_all:.1f}s")
        
        # Save training log (ALWAYS SAVE)
        self._save_training_log()
        
        # ===== SAVE METRICS (ALWAYS SAVE FOR THESIS) =====
        print("\n[SAVING] Saving comprehensive metrics...")
        
        # Determine save directory
        experiment_dir = os.environ.get('EXPERIMENT_DIR', None)
        if experiment_dir:
            logs_dir = os.path.join(experiment_dir, 'logs')
            os.makedirs(logs_dir, exist_ok=True)
        else:
            logs_dir = "training_logs"
            os.makedirs(logs_dir, exist_ok=True)
        
        # Save from live monitor (only if enabled and has data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.enable_live_monitor and (self.live_monitor.episode_metrics or self.live_monitor.turn_metrics):
            if experiment_dir:
                from pathlib import Path
                self.live_monitor.log_dir = Path(logs_dir)
            self.live_monitor.save_metrics(f"monitor_{timestamp}")
        
        # Save from metrics tracker (ALWAYS SAVE)
        if experiment_dir:
            metrics_file = os.path.join(logs_dir, f"metrics_tracker_{timestamp}.json")
        else:
            metrics_file = os.path.join(logs_dir, f"metrics_tracker_{timestamp}.json")
        self.metrics_tracker.save_to_json(metrics_file)
        
        # Save RL metrics (comprehensive RL analysis)
        self._save_rl_metrics(logs_dir, timestamp, training_time)
        
        # Save learning curves data
        self._save_learning_curves(logs_dir, timestamp)
        
        # Save convergence report
        self._save_convergence_report(logs_dir, timestamp, training_time)
        
        # ===== ORGANIZE RESULTS IN MAJOR_RESULTS/ =====
        # Only organize to major_results/ if explicitly requested (via environment variable)
        # This is set by train_all_variations.py, not by individual train.py scripts
        if experiment_dir and os.environ.get('ORGANIZE_TO_MAJOR_RESULTS', '').lower() == 'true':
            try:
                self._organize_major_results(experiment_dir, training_time, timestamp)
            except Exception as e:
                print(f"\n[WARNING] Failed to organize major_results: {e}")
                import traceback
                traceback.print_exc()
            
        # Print summary statistics (ALWAYS)
        summary = self.metrics_tracker.get_summary_statistics()
        print("\n[METRICS] Training Metrics Summary:")
        print(f"   Mean Return: {summary.get('mean_return', 0.0):.3f} ± {summary.get('std_return', 0.0):.3f}")
        print(f"   Recent Mean Return (last 100): {summary.get('recent_mean_return', 0.0):.3f}")
        print(f"   Mean Episode Length: {summary.get('mean_length', 0.0):.1f} turns")
        print(f"   Mean Coverage: {summary.get('mean_coverage', 0.0):.1%}")
        print(f"   Transition Success Rate: {summary.get('transition_success_rate', 0.0):.1%}")
        print(f"   Question Answer Rate: {summary.get('question_answer_rate', 0.0):.1%}")
        
        option_usage = summary.get('option_usage', {})
        if option_usage:
            print(f"\n   Option Usage:")
            for opt, prop in sorted(option_usage.items(), key=lambda x: -x[1]):
                print(f"      {opt}: {prop:.1%}")
        
        # ===== GENERATE RL PLOTS AUTOMATICALLY =====
        if experiment_dir and len(self.metrics_tracker.episode_returns) > 0:
            print("\n[PLOTS] Generating RL/HRL evaluation plots...")
            try:
                # Import here to avoid circular dependencies
                import sys
                from pathlib import Path
                
                # Find RL_plots directory (should be at project root)
                project_root = Path(__file__).parent.parent.parent
                rl_plots_module = project_root / 'RL_plots' / 'generate_rl_plots.py'
                
                if rl_plots_module.exists():
                    # Add project root to path and import
                    if str(project_root) not in sys.path:
                        sys.path.insert(0, str(project_root))
                    
                    # Import the plot generator
                    from RL_plots.generate_rl_plots import RLPlotGenerator
                    
                    # Generate plots automatically in experiment_dir/RL_plots
                    generator = RLPlotGenerator(experiment_dir, output_dir=None)  # None = auto to experiment_dir/RL_plots
                    generator.generate_all_plots()
                    print(f"   [SUCCESS] RL plots saved to: {generator.output_dir}")
                else:
                    print(f"   [WARNING] RL plot generator not found at {rl_plots_module}")
                    print(f"      (Project root: {project_root})")
            except Exception as e:
                print(f"   [WARNING] Failed to generate RL plots: {e}")
                import traceback
                traceback.print_exc()
        
        # ===== SAVE FINAL MODEL =====
        if self.use_actor_critic and hasattr(self, 'agent'):
            print("\n[SAVING] Saving final trained model...")
            experiment_dir = os.environ.get('EXPERIMENT_DIR', None)
            if experiment_dir:
                model_dir = os.path.join(experiment_dir, 'models')
                os.makedirs(model_dir, exist_ok=True)
                final_model_path = os.path.join(model_dir, f'final_model_ep{self.current_episode}.pt')
                self.trainer.save_checkpoint(final_model_path, self.current_episode)
                print(f"   [SUCCESS] Final model saved: {final_model_path}")
            else:
                # Fallback to default location
                os.makedirs('models', exist_ok=True)
                final_model_path = f'models/final_model_ep{self.current_episode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
                self.trainer.save_checkpoint(final_model_path, self.current_episode)
                print(f"   [SUCCESS] Final model saved: {final_model_path}")
        
        # Clean up
        self.monitor.close()
        self.map_visualizer.close()
        print("\n[SUCCESS] Training completed successfully!")

    def _generate_rl_analysis_report(self, training_time, summary):
        """Generate comprehensive RL analysis report with convergence analysis"""
        print("\n" + "=" * 80)
        print(" " * 25 + "REINFORCEMENT LEARNING ANALYSIS REPORT" + " " * 25)
        print("=" * 80)
        
        # Sample Efficiency Analysis
        print(f"\n[EFFICIENCY] SAMPLE EFFICIENCY:")
        total_samples = len(self.monitor.training_history)
        episodes_to_convergence = self._analyze_convergence()
        print(f"   ├─ Total Samples (Turns): {total_samples:,}")
        print(f"   ├─ Sample Efficiency: {total_samples/self.current_episode:.1f} turns/episode")
        print(f"   ├─ Convergence Episode: {episodes_to_convergence if episodes_to_convergence else 'Not achieved'}")
        print(f"   └─ Samples to Convergence: {episodes_to_convergence * (total_samples/self.current_episode) if episodes_to_convergence else 'N/A'}")
        
        # Learning Curve Analysis
        print(f"\n[ANALYSIS] LEARNING CURVE ANALYSIS:")
        if len(self.episode_rewards) >= 100:
            early_performance = np.mean(self.episode_rewards[:100])
            late_performance = np.mean(self.episode_rewards[-100:])
            improvement = late_performance - early_performance
            improvement_rate = improvement / len(self.episode_rewards) * 100
            
            print(f"   ├─ Early Performance (first 100): {early_performance:.3f}")
            print(f"   ├─ Late Performance (last 100): {late_performance:.3f}")
            print(f"   ├─ Total Improvement: {improvement:.3f}")
            print(f"   ├─ Improvement Rate: {improvement_rate:.4f}/100 episodes")
            stability_status = "[STABLE]" if np.std(self.episode_rewards[-100:]) < 2.0 else "[UNSTABLE]"
            print(f"   └─ Learning Stability: {stability_status}")
        
        # Policy Analysis
        print(f"\n[POLICY] POLICY ANALYSIS:")
        option_counts = {}
        termination_counts = {}
        option_durations = []
        
        current_option_start = 0
        for i, turn in enumerate(self.monitor.training_history):
            option = turn.get('info', {}).get('option', 'Unknown')
            if option and option != 'Unknown':
                option_counts[option] = option_counts.get(option, 0) + 1
            
            # Track option terminations and durations
            if turn.get('option_terminated', False):
                duration = i - current_option_start + 1
                option_durations.append(duration)
                termination_counts[option] = termination_counts.get(option, 0) + 1
                current_option_start = i + 1
        
        print(f"   ├─ Option Usage Distribution:")
        for option, count in sorted(option_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(self.monitor.training_history) * 100
            print(f"   │    • {option}: {count:,} turns ({pct:.1f}%)")
        
        if option_durations:
            print(f"   ├─ Average Option Duration: {np.mean(option_durations):.1f} turns")
            print(f"   ├─ Option Duration Std: {np.std(option_durations):.1f}")
            termination_status = "[ACTIVE]" if len(termination_counts) > 0 else "[NONE] No terminations detected"
            print(f"   └─ Termination Learning: {termination_status}")
        
        # Exploration vs Exploitation Analysis
        print(f"\n[EXPLORATION] EXPLORATION-EXPLOITATION BALANCE:")
        if len(self.episode_rewards) >= 200:
            # Convert deque to list for slicing
            history_list = list(self.monitor.training_history)
            quarter_size = len(history_list) // 4
            entropy_early = self._calculate_action_entropy(history_list[:quarter_size])
            entropy_late = self._calculate_action_entropy(history_list[-quarter_size:])
            print(f"   ├─ Early Action Entropy: {entropy_early:.3f}")
            print(f"   ├─ Late Action Entropy: {entropy_late:.3f}")
            print(f"   ├─ Exploration Decay: {entropy_early - entropy_late:.3f}")
            balance_status = "[GOOD] Good balance" if 0.1 < entropy_late < entropy_early else "[WARNING] Check entropy schedule"
            print(f"   └─ Exploitation Status: {balance_status}")
        
        # Credit Assignment Analysis (HRL specific)
        print(f"\n[CREDIT] HIERARCHICAL CREDIT ASSIGNMENT:")
        option_rewards = {}
        for turn in self.monitor.training_history:
            option = turn['info'].get('option', 'Unknown')  # Fixed: get option from 'info' dict
            reward = turn.get('reward', 0)
            if option not in option_rewards:
                option_rewards[option] = []
            option_rewards[option].append(reward)
        
        for option, rewards in option_rewards.items():
            if rewards:
                avg_reward = np.mean(rewards)
                print(f"   ├─ {option} Average Reward: {avg_reward:.3f}")
        
        # Generate and save convergence plots
        self._generate_convergence_plots()
        
        # Generate comprehensive analysis with individual plots
        self._generate_comprehensive_analysis()
        
        print("=" * 80)

    def _analyze_convergence(self):
        """Analyze when the policy converged"""
        if len(self.episode_rewards) < 200:
            return None
        
        # Use sliding window approach to detect convergence
        window_size = 50
        threshold = 0.05  # 5% change threshold
        
        for i in range(100, len(self.episode_rewards) - window_size):
            current_window = self.episode_rewards[i:i+window_size]
            next_window = self.episode_rewards[i+window_size//2:i+window_size//2+window_size]
            
            current_mean = np.mean(current_window)
            next_mean = np.mean(next_window)
            
            if abs(next_mean - current_mean) / (abs(current_mean) + 1e-8) < threshold:
                # Check if this stability continues
                stable_count = 0
                for j in range(i, min(i+100, len(self.episode_rewards)-window_size)):
                    test_window = self.episode_rewards[j:j+window_size]
                    if abs(np.mean(test_window) - current_mean) / (abs(current_mean) + 1e-8) < threshold:
                        stable_count += 1
                
                if stable_count > 50:  # Stable for 50+ episodes
                    return i
        
        return None

    def _calculate_action_entropy(self, history):
        """Calculate action entropy for exploration analysis"""
        actions = [turn.get('subaction', 'Unknown') for turn in history]
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        total = len(actions)
        entropy = 0
        for count in action_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy

    def _generate_convergence_plots(self):
        """Generate comprehensive convergence and learning analysis plots"""
        try:
            import matplotlib.pyplot as plt
            
            # Create comprehensive analysis figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Learning Curve with Moving Averages
            episodes = list(range(1, len(self.episode_rewards) + 1))
            ax1.plot(episodes, self.episode_rewards, alpha=0.3, color='lightblue', label='Episode Rewards')
            
            # Moving averages
            if len(self.episode_rewards) >= 50:
                ma_50 = []
                for i in range(len(self.episode_rewards)):
                    start = max(0, i - 49)
                    ma_50.append(np.mean(self.episode_rewards[start:i+1]))
                ax1.plot(episodes, ma_50, color='orange', linewidth=2, label='50-Episode MA')
            
            if len(self.episode_rewards) >= 100:
                ma_100 = []
                for i in range(len(self.episode_rewards)):
                    start = max(0, i - 99)
                    ma_100.append(np.mean(self.episode_rewards[start:i+1]))
                ax1.plot(episodes, ma_100, color='red', linewidth=2, label='100-Episode MA')
            
            convergence_episode = self._analyze_convergence()
            if convergence_episode:
                ax1.axvline(x=convergence_episode, color='green', linestyle='--', linewidth=2, label=f'Convergence (ep {convergence_episode})')
            
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.set_title('Learning Curve Analysis')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Convergence Analysis
            if len(self.episode_rewards) >= 100:
                convergence_analysis = []
                window = 100
                for i in range(window, len(self.episode_rewards)):
                    recent_performance = np.mean(self.episode_rewards[i-window:i])
                    convergence_analysis.append(recent_performance)
                
                conv_episodes = list(range(window, len(self.episode_rewards)))
                ax2.plot(conv_episodes, convergence_analysis, 'go-', linewidth=2)
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('100-Episode Moving Average')
                ax2.set_title('Convergence Analysis')
                ax2.grid(True, alpha=0.3)
                
                if convergence_episode:
                    ax2.axvline(x=convergence_episode, color='red', linestyle='--', label='Detected Convergence')
                    ax2.legend()
            
            # Plot 3: Option Usage Distribution
            option_counts = {}
            for turn in self.monitor.training_history:
                option = turn.get('info', {}).get('option', 'Unknown')
                if option and option != 'Unknown':
                    option_counts[option] = option_counts.get(option, 0) + 1
            
            if option_counts:
                options = list(option_counts.keys())
                counts = list(option_counts.values())
                colors = plt.cm.Set3(np.linspace(0, 1, len(options)))
                
                wedges, texts, autotexts = ax3.pie(counts, labels=options, autopct='%1.1f%%', colors=colors)
                ax3.set_title('Option Usage Distribution')
            
            # Plot 4: Training Stability Analysis
            if len(self.episode_rewards) >= 200:
                stability_window = 50
                stability_scores = []
                
                for i in range(stability_window, len(self.episode_rewards)):
                    window_data = self.episode_rewards[i-stability_window:i]
                    stability = 1.0 / (1.0 + np.std(window_data))  # Higher = more stable
                    stability_scores.append(stability)
                
                stability_episodes = list(range(stability_window, len(self.episode_rewards)))
                ax4.plot(stability_episodes, stability_scores, 'b-', linewidth=2)
                ax4.set_xlabel('Episode')
                ax4.set_ylabel('Stability Score')
                ax4.set_title('Training Stability Analysis')
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim(0, 1)
            
            plt.tight_layout()
            
            # Save plots to experiment directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_dir = os.environ.get('EXPERIMENT_DIR', None)
            if experiment_dir:
                plot_filename = os.path.join(experiment_dir, f"rl_analysis_{timestamp}.png")
            else:
                # Fallback to training_logs if EXPERIMENT_DIR not set
                plot_filename = f"training_logs/rl_analysis_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            
            print(f"\n[PLOTS] RL Analysis plots saved to: {plot_filename}")
            
            plt.close()
            
        except ImportError:
            print("[WARNING] Matplotlib not available - skipping plot generation")
        except Exception as e:
            print(f"[WARNING] Failed to generate plots: {e}")

    def _generate_comprehensive_analysis(self):
        """Generate comprehensive analysis with individual and combined plots"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            
            experiment_dir = os.environ.get('EXPERIMENT_DIR', None)
            if not experiment_dir:
                print("[WARNING] No experiment directory found - skipping comprehensive analysis")
                return
            
            plots_dir = os.path.join(experiment_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            print("\n[PLOTS] Generating Comprehensive Analysis Plots...")
            
            # ===== INDIVIDUAL PLOTS =====
            
            # 1. Learning Curve (Individual)
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            episodes = range(1, len(self.episode_rewards) + 1)
            ax1.plot(episodes, self.episode_rewards, alpha=0.3, color='blue', label='Raw Returns')
            
            # Moving average
            if len(self.episode_rewards) >= 50:
                window = 50
                moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                ax1.plot(range(window, len(self.episode_rewards) + 1), moving_avg, 
                        linewidth=3, color='red', label='50-Episode Moving Average')
            
            # Detect hypothesis from experiment directory
            hypothesis = 'H1'  # Default
            if experiment_dir:
                exp_path = Path(experiment_dir)
                dir_name = exp_path.name.lower()
                for h in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    if dir_name.startswith(h) or f'_{h}_' in dir_name:
                        hypothesis = h.upper()
                        break
                # Also check metadata
                metadata_file = exp_path / 'metadata.json'
                if metadata_file.exists():
                    import json
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        exp_name = metadata.get('experiment_name', '').lower()
                        for h in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                            if h in exp_name:
                                hypothesis = h.upper()
                                break
            
            ax1.set_xlabel('Episode', fontsize=14)
            ax1.set_ylabel('Total Reward', fontsize=14)
            ax1.set_title(f'{hypothesis}: Learning Curve', fontsize=16, fontweight='bold')
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'learning_curve_{timestamp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   [SAVED] Saved: learning_curve_{timestamp}.png")
            
            # 2. Option Usage Distribution (Individual)
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            option_counts = {}
            for turn in self.monitor.training_history:
                option = turn['info'].get('option', 'Unknown')
                option_counts[option] = option_counts.get(option, 0) + 1
            
            if option_counts:
                options = list(option_counts.keys())
                counts = list(option_counts.values())
                colors = plt.cm.Set3(np.linspace(0, 1, len(options)))
                
                wedges, texts, autotexts = ax2.pie(counts, labels=options, autopct='%1.1f%%',
                                                   colors=colors, textprops={'fontsize': 12})
                ax2.set_title('Option Usage Distribution (All Episodes)', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'option_distribution_{timestamp}.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   [SAVED] Saved: option_distribution_{timestamp}.png")
            
            # 3. Option Evolution Over Time (Per-Episode with Rolling Window)
            fig3, ax3 = plt.subplots(figsize=(14, 8))
            
            # Get per-episode option usage data from metrics_tracker
            episode_option_usage = self.metrics_tracker.episode_option_usage
            
            # Initialize variables for mega plot (will be populated if data exists)
            option_evolution = {}
            colors_map = {'Explain': 'blue', 'AskQuestion': 'green', 'OfferTransition': 'orange', 'Conclude': 'red'}
            
            if episode_option_usage and len(episode_option_usage) > 0:
                # Get all unique options
                all_options = set()
                for ep_usage in episode_option_usage:
                    all_options.update(ep_usage.keys())
                
                # Calculate percentages per episode
                episodes = np.arange(1, len(episode_option_usage) + 1)
                option_percentages = {opt: [] for opt in all_options}
                
                for ep_usage in episode_option_usage:
                    total = sum(ep_usage.values())
                    if total == 0:
                        for opt in all_options:
                            option_percentages[opt].append(0)
                    else:
                        for opt in all_options:
                            pct = (ep_usage.get(opt, 0) / total) * 100
                            option_percentages[opt].append(pct)
                
                # Apply rolling window smoothing for cleaner visualization
                n_episodes = len(episode_option_usage)
                window_size = max(3, min(50, n_episodes // 20))
                
                option_percentages_smoothed = {}
                for opt in all_options:
                    data = np.array(option_percentages[opt])
                    if len(data) >= window_size:
                        # Pad with edge values to avoid boundary effects
                        padded = np.pad(data, (window_size//2, window_size//2), mode='edge')
                        smoothed_full = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
                        # Trim to original length
                        smoothed = smoothed_full[:len(data)]
                        option_percentages_smoothed[opt] = smoothed
                    else:
                        option_percentages_smoothed[opt] = data
                
                # Store for mega plot
                option_evolution = option_percentages_smoothed
                
                # Sort options by mean usage for legend ordering
                option_means = {opt: np.mean(pcts) for opt, pcts in option_percentages_smoothed.items()}
                sorted_options = sorted(option_means.keys(), key=lambda o: option_means[o], reverse=True)
                
                # Plot evolution for each option
                colors_map = {'Explain': 'blue', 'AskQuestion': 'green', 'OfferTransition': 'orange', 'Conclude': 'red'}
                for opt in sorted_options:
                    pcts = option_percentages_smoothed[opt]
                    mean_pct = option_means[opt]
                    color = colors_map.get(opt, 'gray')
                    # Only show in legend if mean > 0.5%
                    label = f"{opt} ({mean_pct:.1f}%)" if mean_pct > 0.5 else None
                    ax3.plot(episodes, pcts, color=color, linewidth=2.5, label=label, alpha=0.8)
                
                ax3.set_xlabel('Episode', fontsize=12)
                ax3.set_ylabel('Usage Percentage (%)', fontsize=12)
                ax3.set_title('Option Usage Evolution Over Training', fontsize=14, fontweight='bold', pad=15)
                ax3.legend(loc='best', fontsize=10, framealpha=0.9)
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim(-2, 102)
                ax3.set_xlim(1, len(episodes))
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'option_evolution_{timestamp}.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   [SAVED] Saved: option_evolution_{timestamp}.png")
            else:
                print(f"   [SKIP] No episode option usage data available for option evolution plot")
                plt.close()
            
            # 4. Reward Distribution (Individual)
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            ax4.hist(self.episode_rewards, bins=30, alpha=0.7, color='purple', edgecolor='black')
            ax4.axvline(np.mean(self.episode_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.episode_rewards):.2f}')
            ax4.set_xlabel('Total Reward', fontsize=14)
            ax4.set_ylabel('Frequency', fontsize=14)
            ax4.set_title('Reward Distribution', fontsize=16, fontweight='bold')
            ax4.legend(fontsize=12)
            ax4.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'reward_distribution_{timestamp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   [SAVED] Saved: reward_distribution_{timestamp}.png")
            
            # ===== COMBINED MEGA PLOT =====
            fig = plt.figure(figsize=(20, 12))
            gs = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.3)
            
            # Learning Curve (large, top left)
            ax1 = fig.add_subplot(gs[0, :2])
            ax1.plot(episodes, self.episode_rewards, alpha=0.3, color='blue')
            if len(self.episode_rewards) >= 50:
                ax1.plot(range(50, len(self.episode_rewards) + 1), moving_avg, linewidth=3, color='red')
            ax1.set_xlabel('Episode', fontsize=12)
            ax1.set_ylabel('Reward', fontsize=12)
            ax1.set_title('Learning Curve', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Option Distribution (top right)
            ax2 = fig.add_subplot(gs[0, 2])
            # Calculate option_counts if not already done
            if 'option_counts' not in locals() or not option_counts:
                option_counts = {}
                for turn in self.monitor.training_history:
                    option = turn.get('info', {}).get('option', 'Unknown')
                    if option and option != 'Unknown':
                        option_counts[option] = option_counts.get(option, 0) + 1
            
            if option_counts:
                options = list(option_counts.keys())
                counts = list(option_counts.values())
                colors = plt.cm.Set3(np.linspace(0, 1, len(options)))
                wedges, texts, autotexts = ax2.pie(counts, labels=options, autopct='%1.1f%%', colors=colors)
                ax2.set_title('Option Distribution', fontsize=12, fontweight='bold')
            
            # Option Evolution (middle, full width)
            ax3 = fig.add_subplot(gs[1, :])
            if option_evolution:
                for option, percentages in option_evolution.items():
                    color = colors_map.get(option, 'gray')
                    ax3.plot(episodes, percentages, marker='o', linewidth=2, label=option, color=color)
                ax3.legend(fontsize=10)
            else:
                ax3.text(0.5, 0.5, 'No option evolution data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_xlabel('Episode', fontsize=12)
            ax3.set_ylabel('Usage %', fontsize=12)
            ax3.set_title('Option Evolution Over Time', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Reward Distribution (bottom left)
            ax4 = fig.add_subplot(gs[2, 0])
            ax4.hist(self.episode_rewards, bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax4.set_xlabel('Reward', fontsize=10)
            ax4.set_ylabel('Frequency', fontsize=10)
            ax4.set_title('Reward Distribution', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Convergence Analysis (bottom middle)
            ax5 = fig.add_subplot(gs[2, 1])
            if len(self.episode_rewards) >= 10:
                rolling_best = np.maximum.accumulate(self.episode_rewards)
                ax5.plot(episodes, rolling_best, linewidth=2, color='green')
                ax5.set_xlabel('Episode', fontsize=10)
                ax5.set_ylabel('Best So Far', fontsize=10)
                ax5.set_title('Convergence', fontsize=12, fontweight='bold')
                ax5.grid(True, alpha=0.3)
            
            # Statistics (bottom right)
            ax6 = fig.add_subplot(gs[2, 2])
            ax6.axis('off')
            stats_text = f"Statistics\n{'='*30}\n\n"
            stats_text += f"Episodes: {len(self.episode_rewards)}\n"
            if len(self.episode_rewards) > 0:
                stats_text += f"Avg Reward: {np.mean(self.episode_rewards):.2f}\n"
                stats_text += f"Std Reward: {np.std(self.episode_rewards):.2f}\n"
                stats_text += f"Best Reward: {max(self.episode_rewards):.2f}\n"
                stats_text += f"Worst Reward: {min(self.episode_rewards):.2f}\n"
            else:
                stats_text += "No episodes completed\n"
            ax6.text(0.1, 0.9, stats_text, fontsize=11, verticalalignment='top',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.suptitle('Comprehensive Training Analysis', fontsize=18, fontweight='bold')
            plt.savefig(os.path.join(plots_dir, f'comprehensive_analysis_{timestamp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   [SAVED] Saved: comprehensive_analysis_{timestamp}.png")
            
            print(f"\n[SUCCESS] All analysis plots saved to: {plots_dir}")
            
        except Exception as e:
            print(f"[WARNING] Failed to generate comprehensive analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_training_log(self):
        """Save training results to JSON file - ALWAYS SAVE to experiment directory"""
        try:
            # Determine save location (prefer experiment directory)
            experiment_dir = os.environ.get('EXPERIMENT_DIR', None)
            now = datetime.now()
            timestamp = now.strftime("%H%M%S")
            
            if experiment_dir:
                # Save to experiment directory
                logs_dir = os.path.join(experiment_dir, 'logs')
                os.makedirs(logs_dir, exist_ok=True)
                filename = f"training_log_{timestamp}.json"
                filepath = os.path.join(logs_dir, filename)
            else:
                # Fallback to date-organized directory under experiments/
                date_str = now.strftime("%Y%m%d")
                timestamp_full = now.strftime("%Y%m%d_%H%M%S")
                exp_base = Path("training_logs/experiments")
                date_folder = exp_base / date_str
                date_folder.mkdir(parents=True, exist_ok=True)
                
                # Create a fallback experiment folder for this date
                existing = list(date_folder.glob("exp_*"))
                if existing:
                    numbers = []
                    for e in existing:
                        parts = e.name.split('_')
                        if len(parts) > 1 and parts[1].isdigit():
                            numbers.append(int(parts[1]))
                    next_num = max(numbers) + 1 if numbers else 1
                else:
                    next_num = 1
                
                fallback_exp_dir = date_folder / f"exp_{next_num:03d}_{timestamp_full}"
                fallback_exp_dir.mkdir(exist_ok=True)
                logs_dir = fallback_exp_dir / "logs"
                logs_dir.mkdir(exist_ok=True)
                
                print(f"[WARNING] EXPERIMENT_DIR not set, using fallback: {fallback_exp_dir}")
                
                filename = f"training_log_{timestamp}.json"
                filepath = logs_dir / filename
            
            # Prepare data for JSON serialization
            training_data = {
                "training_summary": self.monitor.get_training_summary(),
                "episode_rewards": self.episode_rewards,
                "training_history": []
            }
            
            # Convert training history to JSON-serializable format
            for step in self.monitor.training_history:
                serializable_step = {
                    "episode": step["episode"],
                    "turn": step["turn"],
                    "action": step["action"],
                    "reward": step["reward"],
                    "done": step["done"],
                    "info": {
                        "option": step["info"].get("option"),
                        "subaction": step["info"].get("subaction"),
                        "agent_utterance": step["info"].get("agent_utterance"),
                        "facts_shared": step["info"].get("facts_shared"),
                        "exhibits_covered": step["info"].get("exhibits_covered"),
                        "current_focus": step["info"].get("current_focus"),
                        "current_exhibit": step["info"].get("current_exhibit")
                    }
                }
                
                # Add simulator data if available
                if step.get("simulator_data"):
                    serializable_step["simulator_data"] = {
                        "aoi": step["simulator_data"].get("aoi"),
                        "persona": step["simulator_data"].get("persona"),
                        "utterance": step["simulator_data"].get("utterance")
                    }
                
                training_data["training_history"].append(serializable_step)
            
            # Save to file (ensure filepath is a string)
            filepath_str = str(filepath) if isinstance(filepath, Path) else filepath
            with open(filepath_str, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2)
            
            print(f"[SAVED] Training log saved to: {filepath_str}")
            
        except Exception as e:
            print(f"[ERROR] Error saving training log: {e}")
    
    def _save_rl_metrics(self, logs_dir: str, timestamp: str, training_time: float):
        """Save comprehensive RL metrics JSON file"""
        try:
            # Update total training time
            self.metrics_tracker.total_training_time_seconds = training_time
            
            # Get RL metrics summary
            rl_metrics = self.metrics_tracker.get_rl_metrics_summary()
            
            # Add training efficiency data
            rl_metrics["training_efficiency"]["total_time_seconds"] = training_time
            rl_metrics["training_efficiency"]["updates_per_episode"] = self.metrics_tracker.updates_per_episode
            
            # Save to JSON
            rl_metrics_file = os.path.join(logs_dir, f"rl_metrics_{timestamp}.json")
            with open(rl_metrics_file, 'w') as f:
                json.dump(rl_metrics, f, indent=2)
            
            print(f"   [SAVED] Saved: rl_metrics_{timestamp}.json")
        except Exception as e:
            print(f"   [WARNING] Failed to save RL metrics: {e}")
    
    def _save_learning_curves(self, logs_dir: str, timestamp: str):
        """Save learning curves data JSON file"""
        try:
            learning_curves = {
                "episode_returns": self.episode_rewards,
                "value_estimates": self.metrics_tracker.value_estimates,
                "policy_entropy": self.metrics_tracker.entropies,
                "value_losses": self.metrics_tracker.value_losses,
                "policy_losses": self.metrics_tracker.policy_losses,
                "termination_losses": self.metrics_tracker.termination_losses,
                "advantages": self.metrics_tracker.advantages
            }
            
            # Get smoothed learning curve
            smoothed, stds = self.metrics_tracker.get_learning_curve(window=50)
            learning_curves["smoothed_returns"] = smoothed
            learning_curves["return_stds"] = stds
            
            learning_curves_file = os.path.join(logs_dir, f"learning_curves_{timestamp}.json")
            with open(learning_curves_file, 'w') as f:
                json.dump(learning_curves, f, indent=2)
            
            print(f"   [SAVED] Saved: learning_curves_{timestamp}.json")
        except Exception as e:
            print(f"   [WARNING] Failed to save learning curves: {e}")
    
    def _save_convergence_report(self, logs_dir: str, timestamp: str, training_time: float):
        """Save convergence analysis report JSON file"""
        try:
            convergence_episode = self._analyze_convergence()
            
            # Calculate convergence metrics
            convergence_data = {
                "episode": convergence_episode,
                "samples": None,
                "time_seconds": None,
                "window_mean": None,
                "window_std": None,
                "criterion": "sliding_window",
                "threshold": 0.05
            }
            
            if convergence_episode is not None:
                # Calculate samples to convergence
                total_samples = len(self.monitor.training_history)
                samples_per_episode = total_samples / self.current_episode if self.current_episode > 0 else 0
                convergence_data["samples"] = int(convergence_episode * samples_per_episode)
                
                # Calculate time to convergence
                time_per_episode = training_time / self.current_episode if self.current_episode > 0 else 0
                convergence_data["time_seconds"] = convergence_episode * time_per_episode
                
                # Calculate window statistics
                window_size = 50
                if convergence_episode + window_size <= len(self.episode_rewards):
                    convergence_window = self.episode_rewards[convergence_episode:convergence_episode+window_size]
                    convergence_data["window_mean"] = float(np.mean(convergence_window))
                    convergence_data["window_std"] = float(np.std(convergence_window))
            
            # Update metrics tracker with convergence data
            self.metrics_tracker.update_convergence_metrics(convergence_data)
            
            # Save convergence report
            convergence_file = os.path.join(logs_dir, f"convergence_report_{timestamp}.json")
            with open(convergence_file, 'w') as f:
                json.dump(convergence_data, f, indent=2)
            
            print(f"   [SAVED] Saved: convergence_report_{timestamp}.json")
        except Exception as e:
            print(f"   [WARNING] Failed to save convergence report: {e}")
    
    def _organize_major_results(self, experiment_dir: str, training_time: float, timestamp: str):
        """
        Organize training results in major_results/ directory.
        
        Args:
            experiment_dir: Path to experiment directory
            training_time: Total training time in seconds
            timestamp: Timestamp string
        """
        from pathlib import Path
        import json
        from src.utils.major_results_manager import MajorResultsManager
        from src.utils.major_results_templates import create_readme_file
        from src.utils.metrics_validator import MetricsValidator
        
        print("\n[ORGANIZING] Organizing results in major_results/...")
        
        # Initialize manager
        manager = MajorResultsManager()
        
        # Detect model name from experiment directory or metadata
        exp_path = Path(experiment_dir)
        model_name = self._detect_model_name(exp_path)
        
        print(f"   Detected model: {model_name}")
        
        # Extract date from timestamp (YYYYMMDD_HHMMSS -> YYYYMMDD)
        date_str = timestamp.split('_')[0] if '_' in timestamp else datetime.now().strftime("%Y%m%d")
        
        # Create experiment folder with date and auto-incrementing number
        exp_dir = manager.create_experiment_folder(model_name, date_str=date_str)
        
        print(f"   Created experiment folder: {exp_dir.name}")
        
        # Load metadata for README
        metadata = {}
        metadata_path = exp_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Create README
        create_readme_file(exp_dir, model_name, metadata)
        print(f"   [SUCCESS] Created README.md")
        
        # Copy training results
        success = manager.copy_training_results(str(exp_path), exp_dir, copy_maps=True)
        if success:
            print(f"   [SUCCESS] Copied training results")
        else:
            print(f"   [WARNING] Failed to copy some training results")
        
        # Consolidate metrics
        training_dir = exp_dir / "training"
        manager.consolidate_metrics(exp_dir, training_dir)
        print(f"   [SUCCESS] Consolidated metrics")
        
        # Generate basic visualizations
        self._generate_basic_visualizations(exp_dir, model_name)
        
        # Run evaluation automatically
        self._run_automatic_evaluation(exp_dir, model_name)
        
        # Validate metrics
        validator = MetricsValidator()
        validation_results = validator.validate_all(exp_dir)
        
        # Save validation report
        validator.save_validation_report(exp_dir)
        
        # Print validation summary
        missing = []
        for category, results in validation_results.items():
            for item, exists in results.items():
                if not exists and 'count' not in item.lower():
                    missing.append(f"{category}/{item}")
        
        if missing:
            print(f"   [WARNING] Missing items: {len(missing)} (see validation_report.txt)")
        else:
            print(f"   [SUCCESS] All required metrics present")
        
        print(f"\n[SUCCESS] Results organized in: {exp_dir}")
        print(f"   View README: {exp_dir / 'README.md'}")
        print(f"   Validation: {exp_dir / 'validation_report.txt'}")
    
    def _detect_model_name(self, experiment_dir: Path) -> str:
        """
        Detect model name from experiment directory name or metadata.
        
        Args:
            experiment_dir: Path to experiment directory
            
        Returns:
            Normalized model name
        """
        # Try to extract from directory name
        dir_name = experiment_dir.name.lower()
        
        # Check for model indicators in directory name (aligned with paper.tex hypotheses)
        if 'baseline' in dir_name and 'h' not in dir_name:
            return 'baseline'
        elif 'h1' in dir_name or 'flat_mdp' in dir_name:
            return 'h1_flat_mdp'
        elif 'h2' in dir_name or 'augmented' in dir_name:
            return 'h2_augmented_rewards'
        elif 'h3' in dir_name or 'simulator' in dir_name or 'signal' in dir_name:
            return 'h3_simulator_signal'
        elif 'h4' in dir_name or 'state_efficiency' in dir_name or 'compact' in dir_name:
            return 'h4_state_efficiency'
        elif 'h5' in dir_name or 'termination' in dir_name:
            return 'h5_termination'
        
        # Try metadata
        metadata_path = experiment_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    exp_name = metadata.get('experiment_name', '').lower()
                    if exp_name:
                        # Use MajorResultsManager to normalize
                        from src.utils.major_results_manager import MajorResultsManager
                        manager = MajorResultsManager()
                        return manager.normalize_model_name(exp_name)
            except:
                pass
        
        # Default to baseline
        return 'baseline'
    
    def _generate_basic_visualizations(self, model_dir: Path, model_name: str):
        """
        Generate basic visualizations for major_results.
        
        Args:
            model_dir: Path to model directory in major_results/
            model_name: Model name
        """
        try:
            viz_basic_dir = model_dir / "visualizations" / "basic"
            
            # Generate learning curve
            if len(self.episode_rewards) > 0:
                self._plot_basic_learning_curve(viz_basic_dir)
            
            # Generate convergence analysis (always generate, even if convergence not detected)
            if len(self.episode_rewards) > 0:
                self._plot_basic_convergence(viz_basic_dir)
            
            # Generate RL metrics summary
            if len(self.metrics_tracker.value_losses) > 0:
                self._plot_basic_rl_metrics(viz_basic_dir)
            
            print(f"   [SUCCESS] Generated basic visualizations")
            
        except Exception as e:
            print(f"   [WARNING] Failed to generate some visualizations: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_basic_learning_curve(self, output_dir: Path):
        """Generate basic learning curve plot."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            episodes = np.arange(1, len(self.episode_rewards) + 1)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(episodes, self.episode_rewards, alpha=0.3, color='steelblue', linewidth=0.5, label='Episode Returns')
            
            # Moving average
            if len(self.episode_rewards) >= 50:
                window = 50
                ma = []
                for i in range(len(self.episode_rewards)):
                    start = max(0, i - window + 1)
                    ma.append(np.mean(self.episode_rewards[start:i+1]))
                ax.plot(episodes, ma, color='orange', linewidth=2, label=f'{window}-Episode MA')
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Return')
            ax.set_title('Learning Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'learning_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to plot learning curve: {e}")
    
    def _plot_basic_convergence(self, output_dir: Path):
        """Generate basic convergence analysis plot."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            if len(self.episode_rewards) == 0:
                return
            
            episodes = np.arange(1, len(self.episode_rewards) + 1)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(episodes, self.episode_rewards, alpha=0.3, color='steelblue', linewidth=0.5, label='Episode Returns')
            
            # Add moving average if we have enough episodes
            if len(self.episode_rewards) >= 10:
                window = min(10, len(self.episode_rewards))
                ma = []
                for i in range(len(self.episode_rewards)):
                    start = max(0, i - window + 1)
                    ma.append(np.mean(self.episode_rewards[start:i+1]))
                ax.plot(episodes, ma, color='orange', linewidth=2, label=f'{window}-Episode MA')
            
            # Mark convergence point if detected
            conv_ep = None
            if hasattr(self.metrics_tracker, 'convergence_episode') and self.metrics_tracker.convergence_episode:
                conv_ep = self.metrics_tracker.convergence_episode
                if conv_ep is not None and conv_ep <= len(self.episode_rewards):
                    ax.axvline(x=conv_ep, color='red', linestyle='--', linewidth=2, label=f'Convergence (ep {conv_ep})')
                    ax.scatter([conv_ep], [self.episode_rewards[conv_ep-1]], color='red', s=100, zorder=5)
            else:
                # Show that convergence hasn't been detected yet
                ax.text(0.5, 0.95, 'Convergence not yet detected\n(requires more episodes)', 
                       transform=ax.transAxes, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Return')
            ax.set_title('Convergence Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to plot convergence: {e}")
    
    def _plot_basic_rl_metrics(self, output_dir: Path):
        """Generate basic RL metrics summary plot."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Value loss
            if self.metrics_tracker.value_losses:
                axes[0, 0].plot(self.metrics_tracker.value_losses, alpha=0.7)
                axes[0, 0].set_title('Value Loss')
                axes[0, 0].set_xlabel('Update')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].grid(True, alpha=0.3)
            
            # Policy loss
            if self.metrics_tracker.policy_losses:
                axes[0, 1].plot(self.metrics_tracker.policy_losses, alpha=0.7)
                axes[0, 1].set_title('Policy Loss')
                axes[0, 1].set_xlabel('Update')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Entropy
            if self.metrics_tracker.entropies:
                axes[1, 0].plot(self.metrics_tracker.entropies, alpha=0.7)
                axes[1, 0].set_title('Policy Entropy')
                axes[1, 0].set_xlabel('Update')
                axes[1, 0].set_ylabel('Entropy')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Value estimates
            if self.metrics_tracker.value_estimates:
                axes[1, 1].plot(self.metrics_tracker.value_estimates, alpha=0.7)
                axes[1, 1].set_title('Value Estimates')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Value')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'rl_metrics_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to plot RL metrics: {e}")
    
    def _run_automatic_evaluation(self, exp_dir: Path, model_name: str):
        """
        Automatically run evaluation script for the model after training.
        
        Args:
            exp_dir: Path to experiment directory in major_results/
            model_name: Normalized model name
        """
        import subprocess
        import sys
        
        # Map model names to their evaluation scripts (aligned with paper.tex hypotheses)
        evaluation_scripts = {
            'baseline': 'experiments/baseline/evaluate.py',
            'h1_flat_mdp': 'experiments/h1_flat_mdp/evaluate.py',
            'h2_augmented_rewards': 'experiments/h2_augmented_rewards/evaluate.py',
            'h3_simulator_signal': 'experiments/h3_simulator_signal/evaluate.py',
            'h4_state_efficiency': 'experiments/h4_state_efficiency/evaluate.py',
            'h5_termination': 'experiments/h5_termination/evaluate.py',
        }
        
        eval_script = evaluation_scripts.get(model_name)
        if not eval_script:
            # No evaluation script for this model
            print(f"   [INFO] No evaluation script for model: {model_name}")
            return
        
        eval_script_path = Path(eval_script)
        if not eval_script_path.exists():
            print(f"   [WARNING] Evaluation script not found: {eval_script}")
            return
        
        print(f"\n[EVALUATION] Running automatic evaluation...")
        print(f"   Script: {eval_script}")
        
        # Create evaluation output directory
        eval_output_dir = exp_dir / 'evaluation'
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run evaluation script
        try:
            cmd = [
                sys.executable,
                str(eval_script_path),
                '--experiment-dir', str(exp_dir),
                '--output-dir', str(eval_output_dir)
            ]
            
            result = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode == 0:
                print(f"   [SUCCESS] Evaluation complete")
                if result.stdout:
                    # Print last few lines of output
                    lines = result.stdout.strip().split('\n')
                    for line in lines[-5:]:
                        if line.strip():
                            print(f"   {line}")
            else:
                print(f"   [WARNING] Evaluation failed with return code {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}")
        except Exception as e:
            print(f"   [WARNING] Failed to run evaluation: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main training function"""
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create and run training loop
    training_loop = HRLTrainingLoop(max_episodes=5, max_turns_per_episode=15)
    training_loop.run_training()


if __name__ == "__main__":
    main()
