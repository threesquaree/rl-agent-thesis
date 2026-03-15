import os
import time
import numpy as np
from typing import Dict, Any

from src.training.training_loop import HRLTrainingLoop
from src.visualization.museum_map_visualizer import MuseumMapVisualizer

from .agent import FlatActorCriticAgent
from .env import FlatDialogueEnv
from .trainer import FlatActorCriticTrainer


class FlatTrainingLoop(HRLTrainingLoop):
    """
    Training loop wrapper that reuses the hierarchical infrastructure but swaps
    in a flat policy/agent whose action space is the set of primitive subactions.
    """

    def __init__(self, *args, **kwargs):
        # Force hierarchical components to stay uninitialized in parent
        kwargs = dict(kwargs)
        kwargs.setdefault("use_actor_critic", False)
        self._flat_knowledge_graph_path = kwargs.get("knowledge_graph_path", None)
        self._state_representation = kwargs.get("state_representation", "dialoguebert")
        super().__init__(*args, **kwargs)
        
        # Ensure episode_rewards is initialized (parent should do this, but ensure it exists)
        if not hasattr(self, 'episode_rewards'):
            self.episode_rewards = []
        
        # Override progress tracker with Flat RL flag
        from src.visualization.live_progress import LiveProgressTracker
        max_episodes = kwargs.get("max_episodes", self.max_episodes)
        self.progress_tracker = LiveProgressTracker(max_episodes=max_episodes, is_flat_rl=True)

        # Replace environment with flat variant (same simulator + rewards)
        # Support dialogue act classification for H4
        max_turns = kwargs.get("max_turns_per_episode", self.max_turns_per_episode)
        deliberation_cost = kwargs.get("deliberation_cost", 0.0)
        
        if self._state_representation == "dialogue_act":
            # H4: Use dialogue act environment as base for flat wrapper
            try:
                from src.environment.h4_env import H5StateAblationEnv
                from src.flat_rl.env_dialogue_act import FlatDialogueActEnv
                print("[H4] Using dialogue-act state representation for MDP (23-d)")
                base_env = H5StateAblationEnv(
                    knowledge_graph_path=self._flat_knowledge_graph_path,
                    max_turns=max_turns,
                    deliberation_cost=deliberation_cost
                )
                self.env = FlatDialogueActEnv(base_env)
            except ImportError:
                print("[WARNING] H4 dialogue-act environment not found, using default")
                self.env = FlatDialogueEnv(
                    knowledge_graph_path=self._flat_knowledge_graph_path,
                    max_turns=max_turns,
                )
        else:
            # Default: Standard flat environment
            self.env = FlatDialogueEnv(
                knowledge_graph_path=self._flat_knowledge_graph_path,
                max_turns=max_turns,
            )
        
        # CRITICAL FIX: Update simulator reference to new environment's simulator
        # The parent class sets self.simulator = self.env.simulator, but we replaced self.env
        # so we need to update the simulator reference too
        if hasattr(self.env, 'simulator'):
            self.simulator = self.env.simulator

        # Rebuild map visualizer with new environment reference
        map_save_dir = "training_logs/maps"
        experiment_dir = os.environ.get("EXPERIMENT_DIR", None)
        if experiment_dir:
            map_save_dir = os.path.join(experiment_dir, "maps")

        self.map_visualizer = MuseumMapVisualizer(
            enabled=self.enable_map_viz,
            exhibits=self.env.exhibit_keys,
            save_dir=map_save_dir,
            live_display=False,
        )

        # Initialize flat agent/trainer
        state_dim = self.env.observation_space.shape[0]
        self.agent = FlatActorCriticAgent(
            state_dim=state_dim,
            options=self.env.options,
            subactions=self.env.subactions,
            hidden_dim=256,
            lstm_hidden_dim=128,
            use_lstm=True,
            device=self.device,
        )

        learning_rate = kwargs.get("learning_rate", 1e-4)
        gamma = kwargs.get("gamma", 0.99)

        self.trainer = FlatActorCriticTrainer(
            agent=self.agent,
            learning_rate=learning_rate,
            gamma=gamma,
            device=self.device,
        )

        # Episode buffer mirrors base loop for compatibility
        self.use_actor_critic = True
        self.episode_buffer = {
            "states": [],
            "options": [],
            "subactions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
        }

        self.training_title = "FLAT RL MUSEUM DIALOGUE TRAINING"

    # ------------------------------------------------------------------ #
    # Action selection override
    # ------------------------------------------------------------------ #
    def _generate_action_actor_critic(self, obs) -> Dict[str, Any]:
        available_options = self.env._get_available_options()
        if not available_options:
            return {"option": 0, "subaction": 0, "terminate_option": False}

        available_subactions_dict = {
            opt: self.env._get_available_subactions(opt) for opt in available_options
        }

        action_info = self.agent.select_action(
            state=obs,
            available_options=available_options,
            available_subactions_dict=available_subactions_dict,
            deterministic=False,
        )

        return {
            "option": action_info["option"],
            "option_name": action_info["option_name"],
            "subaction": action_info["subaction"],
            "subaction_name": action_info["subaction_name"],
            "terminate_option": False,
            "flat_action": action_info["flat_action"],
        }
    
    # ------------------------------------------------------------------ #
    # Override _run_episode to handle flat action space
    # ------------------------------------------------------------------ #
    def _run_episode(self):
        """Override to pass flat_action integer to env.step() instead of dict."""
        # Get initial observation
        obs, info = self.env.reset()
        
        # CRITICAL FIX: Initialize simulator (required for proper tracking)
        # The parent class does this, but we override _run_episode so we need to do it too
        if hasattr(self, 'simulator'):
            self.simulator.initialize_session(persona="Agreeable")
        else:
            # If simulator isn't initialized, initialize it from environment
            # FlatDialogueEnv should have a simulator attached
            if hasattr(self.env, 'simulator'):
                self.simulator = self.env.simulator
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
        if hasattr(self, '_update_environment_state'):
            self._update_environment_state()
            obs = self.env._get_obs()  # Get fresh observation with correct focus
        
        # Reset agent state
        if self.use_actor_critic:
            self.agent.reset()
        
        # Reset episode buffer
        self.episode_buffer = {
            "states": [],
            "options": [],
            "subactions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
        }
        
        # Reset map visualizer
        self.map_visualizer.reset()
        
        episode_start_time = time.time()
        episode_reward = 0.0
        turn_count = 0
        
        # Track reward components for episode summary
        episode_reward_components = {
            "engagement": 0.0,
            "novelty": 0.0,
            "responsiveness": 0.0,
            "transition": 0.0,
            "conclude": 0.0
        }
        
        # Track component timing for this episode (for compatibility with parent class)
        episode_timing = {
            "bert_time": 0.0,
            "agent_llm_time": 0.0,
            "agent_template_time": 0.0,
            "simulator_llm_time": 0.0,
            "env_step_time": 0.0,
            "nn_forward_time": 0.0,
        }
        
        # Episode loop
        while turn_count < self.max_turns_per_episode:
            turn_count += 1
            
            # Update environment with simulator state
            self._update_environment_state()
            
            # Generate agent action
            if self.use_actor_critic:
                action_dict = self._generate_action_actor_critic(obs)
                # Extract flat_action integer for flat environment
                flat_action = action_dict.get("flat_action", 0)
            else:
                # Random action for flat space
                flat_action = self.env.action_space.sample()
                action_dict = {"flat_action": flat_action}
            
            # Execute environment step with flat action integer
            next_obs, reward, done, truncated, info = self.env.step(flat_action)
            
            # CRITICAL: Update simulator state (generate user response with gaze features)
            # This is where dwell/gaze_features are generated!
            self._update_simulator_state(info)
            
            # Store transition (for compatibility, store as if hierarchical)
            self.episode_buffer["states"].append(obs)
            self.episode_buffer["options"].append(action_dict.get("option", 0))
            self.episode_buffer["subactions"].append(action_dict.get("subaction", 0))
            self.episode_buffer["rewards"].append(reward)
            self.episode_buffer["next_states"].append(next_obs)
            self.episode_buffer["dones"].append(done or truncated)
            
            # Get simulator data for metrics tracking (now includes user response with gaze features)
            simulator_state = self.simulator.get_current_state()
            user_response = simulator_state.get("last_user_response", {})
            
            # CRITICAL FIX: Update parent's monitor for training history tracking
            # This is needed for total_turns and total_episodes calculation in finalization
            # Convert flat action to hierarchical format for monitor compatibility
            monitor_action = {
                "option": action_dict.get("option", 0),
                "subaction": action_dict.get("subaction", 0),
                "flat_action": flat_action
            }
            simulator_data = self._get_simulator_data() if hasattr(self, '_get_simulator_data') else simulator_state
            self.monitor.update_training_step(
                state=obs,
                action=monitor_action,
                reward=reward,
                done=done or truncated,
                info=info,
                simulator_data=simulator_data
            )
            
            # Build turn_data for metrics tracking (include flat_action_name)
            turn_data = {
                "agent_utterance": info.get("agent_utterance", ""),
                "user_utterance": user_response.get("utterance", ""),
                "option": info.get("option", "Unknown"),  # For compatibility
                "subaction": info.get("subaction", "Unknown"),  # For compatibility
                "flat_action_name": info.get("flat_action_name", ""),  # Flat RL specific
                "dwell": user_response.get("gaze_features", [0.0])[0],
                "response_type": user_response.get("response_type", "unknown"),
                "current_exhibit": info.get("current_exhibit", "Unknown"),
                "facts_shared": info.get("facts_shared", 0),
                "reward_engagement": info.get("reward_engagement", 0.0),
                "reward_novelty": info.get("reward_novelty", 0.0),
                "reward_responsiveness": info.get("reward_responsiveness", 0.0),
                "reward_transition_insufficiency": info.get("reward_transition_insufficiency", 0.0),
                "reward_transition_sufficiency": info.get("reward_transition_sufficiency", 0.0),
                "reward_transition_frequency": info.get("reward_transition_frequency", 0.0),
                "reward_conclude": info.get("reward_conclude", 0.0),
                "total_reward": reward,
                "exhibit_coverage": self.env._get_museum_exhibit_coverage()
            }
            
            # Update metrics tracker with turn data (includes flat_action_name)
            self.metrics_tracker.update_turn(turn_data)
            
            # Log detailed episode data (same as HRL but with flat action)
            if self.detailed_logger:
                # Get prompts from environment and simulator
                agent_prompt = getattr(self.env, '_last_llm_prompt', None)
                agent_system_prompt = getattr(self.env, '_last_agent_system_prompt', None)
                simulator_prompt = getattr(self.simulator, '_last_simulator_prompt', None)
                simulator_system_prompt = getattr(self.simulator, '_last_simulator_system_prompt', None)
                
                # Convert flat action to dict for compatibility with detailed logger
                action_for_log = {
                    "option": action_dict.get("option", 0),
                    "option_name": action_dict.get("option_name", ""),
                    "subaction": action_dict.get("subaction", 0),
                    "subaction_name": action_dict.get("subaction_name", ""),
                    "flat_action": flat_action,
                    "flat_action_name": info.get("flat_action_name", "Unknown")
                }
                
                self.detailed_logger.log_turn(
                    turn_num=turn_count,
                    state=obs,
                    action=action_for_log,
                    reward=reward,
                    info=info,
                    next_state=next_obs,
                    agent_prompt=agent_prompt,
                    agent_system_prompt=agent_system_prompt,
                    simulator_prompt=simulator_prompt,
                    simulator_system_prompt=simulator_system_prompt,
                    user_response=user_response
                )
            
            # Get flat action name for tracking (before using it)
            flat_action_name = info.get("flat_action_name", "Unknown")
            
            # Update map visualizer if enabled
            if self.enable_map_viz:
                agent_exhibit = info.get("current_exhibit", "Unknown")
                visitor_exhibit = simulator_state.get("current_exhibit", agent_exhibit)
                current_dwell = user_response.get("gaze_features", [0.0])[0] if user_response else 0.0
                
                # Get exhibit completion rates
                coverage = self.env._get_museum_exhibit_coverage()
                exhibit_completion = {
                    exhibit_name: coverage.get(exhibit_name, {"coverage": 0.0})["coverage"]
                    for exhibit_name in self.env.exhibit_keys
                }
                
                # Only capture frames for episodes that will be saved (massive speedup)
                should_capture = (self.current_episode % self.map_interval == 0) or self.save_map_frames
                
                if should_capture:
                    # For flat RL, we display the flat action name as "option" with no subaction
                    self.map_visualizer.update(
                        agent_exhibit=agent_exhibit,
                        visitor_exhibit=visitor_exhibit,
                        dwell=current_dwell,
                        turn_num=turn_count,
                        option=flat_action_name,
                        subaction=None,
                        exhibit_completion=exhibit_completion,
                        turn_reward=reward  # Pass turn reward for cumulative display
                    )
                    
                    # Capture frame for animation
                    self.map_visualizer.capture_frame()
                    
                    # Save snapshot if requested
                    if self.save_map_frames:
                        episode_dir = f"episode_{self.current_episode:03d}"
                        self.map_visualizer.save_snapshot(
                            f"{episode_dir}/turn_{turn_count:02d}.png"
                        )
            
            # Update progress tracker with flat_action_name (for live display)
            # For Flat RL, we only track the flat action name (no option/subaction hierarchy)
            # Pass flat_action_name as "option" and None as "subaction" to avoid redundancy
            self.progress_tracker.update_turn(reward, flat_action_name, None)
            
            # Accumulate reward components for this episode
            episode_reward_components["engagement"] += turn_data.get("reward_engagement", 0.0)
            episode_reward_components["novelty"] += turn_data.get("reward_novelty", 0.0)
            episode_reward_components["responsiveness"] += turn_data.get("reward_responsiveness", 0.0)
            # Sum all transition-related rewards
            episode_reward_components["transition"] += (
                turn_data.get("reward_transition_insufficiency", 0.0) +
                turn_data.get("reward_transition_exploration", 0.0) +
                turn_data.get("reward_transition_sufficiency", 0.0) +
                turn_data.get("reward_transition_frequency", 0.0)
            )
            episode_reward_components["conclude"] += turn_data.get("reward_conclude", 0.0)
            
            episode_reward += reward
            obs = next_obs
            
            if done or truncated:
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
        
        # Update trainer if using actor-critic
        train_stats = None
        if self.use_actor_critic and len(self.episode_buffer["states"]) > 0:
            train_stats = self.trainer.update(
                states=self.episode_buffer["states"],
                options=self.episode_buffer["options"],
                subactions=self.episode_buffer["subactions"],
                rewards=self.episode_buffer["rewards"],
                next_states=self.episode_buffer["next_states"],
                dones=self.episode_buffer["dones"]
            )
            
            # Update metrics tracker with RL training stats
            if train_stats:
                self.metrics_tracker.update_training_stats(train_stats)
            
            # Track per-action advantages for MDP collapse analysis
            if 'action_advantages_all' in train_stats:
                if not hasattr(self.metrics_tracker, 'episode_action_advantages'):
                    self.metrics_tracker.episode_action_advantages = []
                self.metrics_tracker.episode_action_advantages.append({
                    'mean': train_stats.get('mean_advantage', 0.0),
                    'std': train_stats.get('advantage_std', 0.0),
                    'values': train_stats['action_advantages_all']
                })
            # Track action logits (proxy for action preferences/Q-values)
            if 'action_logits_all' in train_stats:
                action_logits = np.array(train_stats['action_logits_all'])  # Shape: (batch_size, num_actions)
                # Compute per-action statistics
                per_action_stats = {}
                for action_idx in range(action_logits.shape[1]):
                    action_logs = action_logits[:, action_idx]
                    per_action_stats[f'action_{action_idx}'] = {
                        'mean': float(np.mean(action_logs)),
                        'std': float(np.std(action_logs)),
                        'min': float(np.min(action_logs)),
                        'max': float(np.max(action_logs))
                    }
                if not hasattr(self.metrics_tracker, 'episode_per_action_logits'):
                    self.metrics_tracker.episode_per_action_logits = []
                self.metrics_tracker.episode_per_action_logits.append(per_action_stats)
        
        # Build episode summary for metrics
        exhibits_covered = sum(1 for fact_set in self.env.facts_mentioned_per_exhibit.values() 
                              if len(fact_set) > 0)
        total_facts = sum(len(fact_set) for fact_set in self.env.facts_mentioned_per_exhibit.values())
        
        # Calculate coverage ratio
        total_available_facts = sum(len(self.env.knowledge_graph.get_exhibit_facts(ex)) 
                                   for ex in self.env.exhibit_keys)
        coverage_ratio = total_facts / total_available_facts if total_available_facts > 0 else 0.0
        
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
            "reward_conclude": episode_reward_components["conclude"]
        }
        
        # Update metrics tracker with episode summary
        self.metrics_tracker.update_episode(episode_summary)
        
        # Save map animation if enabled and at the right interval
        if self.enable_map_viz and (self.current_episode % self.map_interval == 0):
            animation_filename = f"episode_{self.current_episode:03d}_animation.gif"
            self.map_visualizer.save_animation(animation_filename, fps=2)
            print(f"   [MAP] Map animation saved: {animation_filename}")
        
        episode_time = time.time() - episode_start_time
        return episode_reward, turn_count, episode_time, episode_timing
    
    # ------------------------------------------------------------------ #
    # Override plot generation to skip HRL-specific plots
    # ------------------------------------------------------------------ #
    def _generate_comprehensive_analysis(self):
        """Override to skip option-specific plots that don't apply to flat RL."""
        import matplotlib.pyplot as plt
        import os
        from datetime import datetime
        
        if not hasattr(self, 'monitor') or not self.monitor.training_history:
            print("[!] No training history available for comprehensive analysis")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_dir = os.environ.get("EXPERIMENT_DIR", ".")
        plots_dir = os.path.join(experiment_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        print("\nGenerating comprehensive analysis plots...")
        
        # Plot 1: Learning curve (same as HRL)
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            episodes = list(range(1, len(self.episode_rewards) + 1))
            
            # Moving average
            window = min(50, len(self.episode_rewards) // 10)
            if window > 1:
                import numpy as np
                smoothed = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                ax.plot(range(window, len(self.episode_rewards) + 1), smoothed, 
                       linewidth=2.5, label=f'{window}-Episode MA', color='#2E86AB')
            
            ax.plot(episodes, self.episode_rewards, alpha=0.3, linewidth=1, 
                   label='Raw Returns', color='gray')
            ax.set_xlabel('Episode', fontsize=14)
            ax.set_ylabel('Cumulative Return', fontsize=14)
            ax.set_title('Flat RL Learning Curve', fontsize=16, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'learning_curve_{timestamp}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   [+] Learning curve -> learning_curve_{timestamp}.png")
        except Exception as e:
            print(f"   [!] Failed to generate learning curve: {e}")
        
        # Plot 2: Flat action distribution over time (NOT option evolution!)
        try:
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Track flat action usage over time windows
            window_size = max(100, len(self.monitor.training_history) // 20)
            action_evolution = {}
            
            history_list = list(self.monitor.training_history)
            for i in range(0, len(history_list), window_size):
                window = history_list[i:i+window_size]
                window_counts = {}
                for turn in window:
                    # Get flat action name (stored in info)
                    action_name = turn.get('info', {}).get('flat_action_name', 'Unknown')
                    if action_name and action_name != 'Unknown':
                        window_counts[action_name] = window_counts.get(action_name, 0) + 1
                
                # Normalize to percentages
                total = sum(window_counts.values())
                for action, count in window_counts.items():
                    if action not in action_evolution:
                        action_evolution[action] = []
                    action_evolution[action].append((count / total * 100) if total > 0 else 0)
            
            # Plot evolution for each action
            if action_evolution:
                window_episodes = [i * window_size / len(history_list) * len(self.episode_rewards)
                                 for i in range(len(next(iter(action_evolution.values()))))]
                
                # Sort actions by average usage for legend clarity
                sorted_actions = sorted(action_evolution.items(), 
                                      key=lambda x: sum(x[1])/len(x[1]), 
                                      reverse=True)
                
                for action, percentages in sorted_actions:
                    ax.plot(window_episodes, percentages, marker='o', linewidth=2, 
                           label=action, alpha=0.8)
                
                ax.set_xlabel('Episode', fontsize=14)
                ax.set_ylabel('Usage Percentage (%)', fontsize=14)
                ax.set_title('Flat Action Distribution Evolution Over Training', 
                           fontsize=16, fontweight='bold')
                ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'action_evolution_{timestamp}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   [+] Action evolution -> action_evolution_{timestamp}.png")
            else:
                print("   [!] No action data available for evolution plot")
        except Exception as e:
            print(f"   [!] Failed to generate action evolution: {e}")
        
        # Plot 3: Reward distribution (same as HRL)
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(self.episode_rewards, bins=30, color='#A23B72', alpha=0.7, edgecolor='black')
            ax.axvline(sum(self.episode_rewards) / len(self.episode_rewards), 
                      color='red', linestyle='--', linewidth=2, label='Mean')
            ax.set_xlabel('Cumulative Return', fontsize=14)
            ax.set_ylabel('Frequency', fontsize=14)
            ax.set_title('Flat RL Reward Distribution', fontsize=16, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'reward_distribution_{timestamp}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   [+] Reward distribution -> reward_distribution_{timestamp}.png")
        except Exception as e:
            print(f"   [!] Failed to generate reward distribution: {e}")
        
        print(f"[OK] Flat RL analysis plots saved to: {plots_dir}")

__all__ = ["FlatTrainingLoop"]


