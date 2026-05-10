"""
Hierarchical Reinforcement Learning Environment for Museum Dialogue Agent

This module implements a Semi-Markov Decision Process (SMDP) environment for museum
dialogue agents using the Options Framework (Sutton et al., 1999). The environment
supports temporal abstraction through high-level options and low-level subactions,
enabling efficient learning in sparse-reward, long-horizon dialogue settings.

Key HRL Components:
- Options Framework: High-level strategies (Explain, Ask, Transition, Conclude) per paper.tex
- Intra-option policies: Low-level subactions within each option
- Learned termination: Option-Critic style termination functions
- Action masking: Prevents invalid actions based on dialogue state
- Gaze-based rewards: Engagement signals from dwell time (Bozkir et al., 2021)
- Dynamic Knowledge Graph: Flexible loading from Neo4j or fallback sources

References:
- Sutton, R. S., Precup, D., & Singh, S. (1999). Between MDPs and semi-MDPs: A framework 
  for temporal abstraction in reinforcement learning. Artificial Intelligence.
- Bacon, P. L., Harb, J., & Precup, D. (2017). The option-critic architecture. AAAI.
- Bozkir, E., et al. (2021). Eye tracking in virtual learning environments.
"""

from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import re
import os
import time
from typing import Dict, Set
from src.utils.dialogue_planner import build_prompt 
from src.utils.knowledge_graph import SimpleKnowledgeGraph
from src.utils.dialoguebert_intent_recognizer import get_dialoguebert_recognizer
from typing import Dict, List, Optional, Any


class MuseumDialogueEnv(gym.Env):
    """
    Hierarchical Reinforcement Learning Environment for Museum Dialogue Agent
    
    Implements the Options Framework as a Semi-Markov Decision Process (SMDP):
    - High-level options (temporally extended actions) select dialogue strategies
    - Low-level subactions specify concrete utterance types within each option
    - Learned termination functions determine when to switch options
    - Action masking prevents invalid actions based on dialogue state
    
    State Space: Focus snapshot + dialogue history + intent label + trend features
    Action Space: Hierarchical (option, subaction, terminate_option)
    Reward: Engagement (dwell) + Novelty (coverage) + deliberation cost
    """
    
    def __init__(self, deliberation_cost=0.0, knowledge_graph_path=None, max_turns=20, 
                 options_override=None, subactions_override=None):
        super().__init__()
        
        # ===== CONFIGURATION =====
        self.deliberation_cost = deliberation_cost
        self.max_turns = max_turns  # Maximum turns per episode
        
        # ===== REWARD MODE (per paper.tex Section 4.7) =====
        # "baseline" = Engagement + Novelty ONLY (for H1, H3, H4, H5)
        # "augmented" = + Responsiveness, Transition, Conclude, Question-asking (for H2)
        self.reward_mode = os.environ.get("HRL_REWARD_MODE", "baseline")
        
        # ===== REWARD PARAMETERS (configurable via environment variables) =====
        # Default values per paper.tex Section 4.7, lines 681-709
        
        # Engagement: r^eng_t = dwell_t × w_engagement (paper.tex default: 1.0)
        self.w_engagement = float(os.environ.get("HRL_W_ENGAGEMENT", "1.0"))

        # Centred engagement reward (thesis Section 5.1): r^ceng_t = w_e * (dwell_t - d_bar_t)
        # Enabled via HRL_CENTRED_ENGAGEMENT=1; EMA decay alpha via HRL_DWELL_EMA_ALPHA (default 0.1)
        self.centred_engagement = os.environ.get("HRL_CENTRED_ENGAGEMENT", "0") == "1"
        self.dwell_ema_alpha = float(os.environ.get("HRL_DWELL_EMA_ALPHA", "0.1"))
        self._dwell_ema = 0.5  # Running EMA of dwell; initialised at neutral midpoint
        
        # Novelty: r^nov_t = novelty_per_fact × |new_facts| (paper.tex default: 1.0)
        self.novelty_per_fact = float(os.environ.get("HRL_NOVELTY_PER_FACT", "1.0"))

        # Engagement-gated novelty: r_t = dwell_t * novelty_credit (engagement modulates novelty)
        # When enabled, engagement and novelty are combined multiplicatively instead of additively
        self.engagement_gated_novelty = os.environ.get("HRL_ENGAGEMENT_GATED_NOVELTY", "0") == "1"

        # Broadened novelty reward (thesis Eq. 5): replaces standard novelty when enabled
        self.broadened_novelty = os.environ.get("HRL_BROADENED_NOVELTY", "0") == "1"
        self.alpha_new = float(os.environ.get("HRL_ALPHA_NEW", "1.0"))
        self.alpha_rep = float(os.environ.get("HRL_ALPHA_REP", "0.3"))
        self.alpha_clar = float(os.environ.get("HRL_ALPHA_CLAR", "0.3"))
        self.alpha_ask = float(os.environ.get("HRL_ALPHA_ASK", "0.2"))
        self.alpha_stale = float(os.environ.get("HRL_ALPHA_STALE", "1.0"))  # Increased from 0.5 to make staying at exhausted exhibits costly
        self.alpha_transition = float(os.environ.get("HRL_ALPHA_TRANSITION", "0.4"))  # Novelty reward for transitions

        # ===== TRAJECTORY REWARD PARAMETERS (prospect theory) =====
        # R_t = α·max(0, Δdwell) − β·max(0, −Δdwell) + R_terminal·coverage
        # β/α = 2.25 from Kahneman & Tversky (1979)
        self.alpha = float(os.environ.get("HRL_ALPHA", "1.0"))
        self.beta = float(os.environ.get("HRL_BETA", "2.25"))
        self.terminal_coverage_weight = float(os.environ.get("HRL_TERMINAL_COVERAGE_WEIGHT", "5.0"))

        # === AUGMENTED REWARD COMPONENTS (only used when reward_mode="augmented") ===
        # Responsiveness: +w_responsiveness (answer) / -0.6*w_responsiveness (deflect)
        self.w_responsiveness = float(os.environ.get("HRL_W_RESPONSIVENESS", "0.5"))
        
        # Conclude: w_conclude × |exhibits_covered|
        self.w_conclude = float(os.environ.get("HRL_W_CONCLUDE", "0.4"))
        
        # Question-asking: hybrid reward
        self.w_ask = float(os.environ.get("HRL_W_ASK", "0.5"))
        
        # Completion bonus: flat bonus when all exhibits reach 100% coverage (automatic conclusion)
        # Encourages agent to complete full tour coverage
        self.w_completion_bonus = float(os.environ.get("HRL_W_COMPLETION_BONUS", "10.0"))
        
        # ===== SMDP COVERAGE FIX: REWARD STRUCTURE PARAMETERS =====
        # Exhaustion penalty: penalty for Explain at exhausted exhibits (increased from -0.5)
        self.exhaustion_penalty_value = float(os.environ.get("HRL_EXHAUSTION_PENALTY", "-1.0"))

        # Transition bonus: immediate bonus for successful exhibit transitions (increased from 0.0)
        self.transition_bonus = float(os.environ.get("HRL_TRANSITION_BONUS", "0.3"))
        
        # Zero engagement at exhausted: set engagement to 0 for Explain at exhausted (default: False)
        self.zero_engagement_exhausted = os.environ.get("HRL_ZERO_ENGAGEMENT_EXHAUSTED", "0") == "1"
        
        # ===== ACTION REPETITION PENALTY (anti-spam fix) =====
        # Penalizes consecutive selection of the same subaction to break spam patterns
        # e.g. ExplainNewFact spam or AskClarification spam after exhaustion
        self.action_repeat_penalty = float(os.environ.get("HRL_ACTION_REPEAT_PENALTY", "0.15"))
        self.action_repeat_threshold = int(os.environ.get("HRL_ACTION_REPEAT_THRESHOLD", "2"))  # Penalty starts after N consecutive

        # ===== ExplainNewFact DIMINISHING RETURNS (anti-spam, reward-shaping) =====
        # Novelty reward for ExplainNewFact decays geometrically with consecutive uses:
        #   Turn 1: novelty × 1.00  (no decay)
        #   Turn 2: novelty × decay_rate          (default 0.65)
        #   Turn 3: novelty × decay_rate²         (default 0.42)
        #   Turn N: novelty × max(floor, decay_rate^(N-1))
        # Applies in BOTH standard and broadened novelty modes.
        # Encourages the agent to interleave AskQuestion / ClarifyFact / RepeatFact
        # rather than monologue-lecturing through every fact sequentially.
        self.enf_decay_rate = float(os.environ.get("HRL_ENF_DECAY_RATE", "0.65"))
        self.enf_decay_floor = float(os.environ.get("HRL_ENF_DECAY_FLOOR", "0.25"))
        self.enf_window = int(os.environ.get("HRL_ENF_WINDOW", "6"))

        # ===== RESPONSE TYPE FEATURES (thesis extension) =====
        # Response type as state feature: 6-dim one-hot (acknowledgment, follow_up_question, question, statement, confusion, silence)
        self.response_type_feature = os.environ.get("HRL_RESPONSE_TYPE_FEATURE", "0") == "1"

        # Response type reward component: reward/penalty based on visitor's response type
        self.response_type_reward = os.environ.get("HRL_RESPONSE_TYPE_REWARD", "0") == "1"
        self.w_response_type = float(os.environ.get("HRL_W_RESPONSE_TYPE", "0.3"))

        # Reward values per response type (positive = good visitor reaction, negative = bad)
        self.response_type_reward_values = {
            "acknowledgment": +0.3,
            "follow_up_question": +0.25,
            "question": +0.1,
            "statement": 0.0,
            "confusion": -0.3,
            "silence": -0.2,
        }

        # Action masking parameters for AskQuestion option
        self.ask_question_dwell_threshold = float(os.environ.get("HRL_ASKQUESTION_DWELL_THRESHOLD", "0.35"))
        self.ask_question_explain_threshold = int(os.environ.get("HRL_ASKQUESTION_EXPLAIN_THRESHOLD", "3"))
        
        # ===== HIERARCHICAL ACTION SPACE =====
        # High-level options represent dialogue strategies (per paper.tex)
        # Can be overridden for H6 (option granularity experiments)
        if options_override is not None:
            self.options = options_override
        else:
            self.options = ["Explain", "AskQuestion", "OfferTransition", "Conclude", "Engage"]

        # Low-level subactions within each option (per paper.tex Table 1)
        # Can be overridden for H6 (option granularity experiments)
        if subactions_override is not None:
            self.subactions = subactions_override
        else:
            self.subactions = {
                "Explain":         ["ExplainNewFact"],
                "AskQuestion":     ["AskOpinion", "AskClarification"],
                "OfferTransition": ["SummarizeAndSuggest"],
                "Conclude":        ["WrapUp"],
                "Engage":          ["RecoverEngagement"],
            }
        
        # Action masking parameters for dialogue coherence
        self.min_facts_before_conclude = 3
        self.min_exhibits_before_conclude = 2
        
        # ===== SIMPLIFIED KNOWLEDGE GRAPH =====
        # Load knowledge graph from JSON or use default
        self.knowledge_graph = SimpleKnowledgeGraph(knowledge_graph_path)
        
        # Extract knowledge graph data
        self.exhibit_keys = self.knowledge_graph.get_exhibit_names()
        self.n_exhibits = len(self.exhibit_keys)
        
        # ===== OBSERVATION SPACE =====
        # State representation with DialogueBERT as per paper formalization (Section 4.6):
        # s_t = [f_t, h_t, i_t, c_t]
        # - f_t: focus vector (n_exhibits + 1 for no-focus)
        # - h_t: dialogue history (n_exhibits completion ratios + 4 option usage)
        # - i_t: intent embedding (64-d projection from 768-d DialogueBERT)
        # - c_t: dialogue context (64-d projection from 768-d DialogueBERT)
        # Total dimensions calculated dynamically based on number of exhibits
        # Example: 5 exhibits → (5+1) + (5+4) + 64 + 64 = 143-d
        
        focus_dim = self.n_exhibits + 1  # +1 for "no focus" state
        # Flat subaction list for history tracking (deterministic ordering)
        self._all_subactions = [sa for opt in self.options for sa in self.subactions[opt]]
        history_dim = self.n_exhibits + len(self._all_subactions)  # completion ratios + per-subaction usage
        intent_dim = 64  # Projected DialogueBERT intent embedding
        context_dim = 64  # Projected DialogueBERT dialogue context
        subaction_availability_dim = 4  # Subaction availability indicators
        response_type_dim = 6 if self.response_type_feature else 0  # One-hot response type

        # Canonical ordering of response types for one-hot encoding
        self.response_type_labels = ["acknowledgment", "follow_up_question", "question", "statement", "confusion", "silence"]

        total_obs_dim = focus_dim + history_dim + intent_dim + context_dim + subaction_availability_dim + response_type_dim

        print(f"[Environment] Observation space: {total_obs_dim}-d "
              f"(focus={focus_dim}, history={history_dim}, intent={intent_dim}, context={context_dim}, "
              f"subaction_availability={subaction_availability_dim}, response_type={response_type_dim})")
        
        self.observation_space = spaces.Box(
            low=-10.0,  # Allow negative values after projection
            high=10.0, 
            shape=(total_obs_dim,), 
            dtype=np.float32
        )
        
        # ===== DIALOGUEBERT PROJECTION MATRIX =====
        # Fixed, offline-trained linear projection P: R^768 -> R^64
        # This implements the projection: i_t = P * e_t where e_t is the 768-d DialogueBERT output
        # Using a fixed random projection with normalization (Johnson-Lindenstrauss type)
        np.random.seed(42)  # Fixed seed for reproducibility
        self.projection_matrix = np.random.randn(64, 768).astype(np.float32) / np.sqrt(768)
        
        # ===== HIERARCHICAL ACTION SPACE =====
        # SMDP action space with options, subactions, and termination
        self.action_space = spaces.Dict({
            "option": spaces.Discrete(len(self.options)),
            "subaction": spaces.Discrete(max(len(subacts) for subacts in self.subactions.values())),
            "terminate_option": spaces.Discrete(2)
        })

        # Initialize environment state
        self.reset()

    # ===== ENVIRONMENT LIFECYCLE =====
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Core dialogue state
        self.focus = 0  # Current exhibit focus (0 = none)
        self.dwell = 0.0  # Current dwell time (engagement signal)
        self.last_user_utterance = ""
        self._previous_dwell = 0.0  # Store previous dwell for lagged rewards
        self._dwell_ema = 0.5  # Reset EMA to neutral midpoint at episode start
        self._last_user_intent = "statement"  # Track user intent for responsiveness
        self._last_user_response_type = "statement"  # Track simulator's response type (question, confusion, etc.)
        self._last_visitor_state = None  # Track visitor state for agent template responsiveness
        
        # Dialogue history for context
        self.dialogue_history = []  # List of (role, utterance, turn_number) tuples
        self.max_dialogue_history = 6  # Keep last 6 utterances (3 exchanges)
        self.dialogue_turn_counter = 0  # Track turn number for DialogueBERT
        
        # SIMPLE FACT TRACKING: exhibit_name -> set of fact IDs mentioned
        self.facts_mentioned_per_exhibit: Dict[str, Set[str]] = {ex: set() for ex in self.exhibit_keys}
        self.facts_mentioned_this_turn: Set[str] = set()  # Fact IDs mentioned this turn only
        
        # Dialogue history tracking
        self.explained = [0] * self.n_exhibits  # Which exhibits have been explained
        self.actions_used = {sa: 0 for sa in self._all_subactions}  # Count of each flat subaction used
        self.consecutive_explain_turns = 0
        
        # Option tracking (for termination learning)
        self.current_option = "Explain"
        self.turns_in_option = 0
        self._previous_option = None

        # Action repetition tracking (anti-spam)
        self._last_subaction = None
        self._consecutive_same_action = 0

        # ENF rolling window for diminishing returns (replaces consecutive counter)
        self._enf_history = deque(maxlen=self.enf_window)
        
        # Transition insufficiency tracking (per paper.tex Section 4.7, line 707)
        # Track last successful transition turn for 3-turn exemption rule
        self.last_successful_transition_turn = -999  # Initialize to far past
        
        # Question-asking tracking (for spacing and hybrid reward)
        self.last_question_turn = -10  # Initialize far in past for spacing check
        
        # Component contribution tracking (for analysis)
        self.engagement_sum = 0.0
        self.novelty_sum = 0.0
        # Broadened novelty sub-component tracking
        self.bnov_new_sum = 0.0
        self.bnov_rep_sum = 0.0
        self.bnov_clar_sum = 0.0
        self.bnov_ask_sum = 0.0
        self.bnov_stale_sum = 0.0
        self.bnov_transition_sum = 0.0
        self.responsiveness_sum = 0.0
        self.conclude_sum = 0.0
        self.transition_sum = 0.0
        self.question_sum = 0.0
        self.exploration_sum = 0.0
        self.exhaustion_sum = 0.0  # H6 fix: track exhaustion penalties
        self.response_type_sum = 0.0  # Response type reward tracking
        self.deliberation_sum = 0.0  # H1 termination tuning: track deliberation cost
        self.action_repeat_sum = 0.0  # Action repetition penalty tracking
        self.trajectory_reward_sum = 0.0
        self.terminal_bonus_sum = 0.0
        
        # Note: Trend tracking removed to match paper specification
        
        # Session tracking
        self.turn_count = 0
        self.session_reward = 0.0
        
        # DialogueBERT embeddings for insight reporting (stored as full 768-d for visualization)
        # Note: The state uses 64-d projected versions, but we keep 768-d for diagnostics
        self._last_intent_embedding = np.zeros(768, dtype=np.float32)
        self._last_dialogue_context = np.zeros(768, dtype=np.float32)
        self._prev_intent_embedding = np.zeros(768, dtype=np.float32)
        self._prev_dialogue_context = np.zeros(768, dtype=np.float32)
        
        # Initialize action masks
        self.action_masks = [True] * len(self.options)  # All options available initially
        
        return self._get_obs(), {}
    
    def set_initial_dialogue(self, agent_greeting: str, user_response: str):
        """Set initial dialogue context from scripted introduction.
        
        Call this after reset() and simulator.inject_introduction() to sync
        the environment's dialogue history with the introduction exchange.
        
        Args:
            agent_greeting: Scripted agent welcome message
            user_response: Scripted visitor initial response
        """
        # Add introduction to dialogue history (turn 0 for intro)
        self.dialogue_history.append(("agent", agent_greeting, 0))
        self.dialogue_history.append(("user", user_response, 0))
        
        # Set last user utterance so Turn 1 prompts have context
        self.last_user_utterance = user_response
        
        # Set a positive initial dwell to reflect engaged visitor
        self.dwell = 0.6  # Moderately engaged baseline
        self._previous_dwell = 0.6
    
    def _compute_question_asking_reward(self, option, spacing, response_length, verbose=False):
        """
        Hybrid question-asking reward (Option 4):
        1. Spacing requirement (avoid spam)
        2. Engagement impact (dwell change)
        3. Response quality (visitor engagement)
        """
        if option != "AskQuestion":
            return 0.0
        
        w_ask = self.w_ask  # From environment config
        
        # 1. Spacing check - penalize question spam
        if spacing < 3:
            penalty = -0.5 * w_ask
            if verbose:
                print(f"❌ QUESTION SPAM: {penalty:.2f} (spacing={spacing} < 3)")
            return penalty
        
        # 2. Engagement impact (dwell change)
        if self._previous_dwell > 0:
            dwell_change = self.dwell - self._previous_dwell
            dwell_pct_change = dwell_change / self._previous_dwell
            engagement_component = w_ask * np.clip(dwell_pct_change, -1.0, 2.0)
        else:
            # First question - give moderate reward
            engagement_component = 0.3 * w_ask
        
        # 3. Response quality (visitor engagement level)
        response_component = 0.0
        if response_length > 10:
            # Substantive response (visitor is engaged)
            response_component = 0.4 * w_ask
        elif response_length > 5:
            # Moderate response
            response_component = 0.2 * w_ask
        elif response_length <= 2:
            # Minimal response (poor question)
            response_component = -0.3 * w_ask
        
        # Combined reward
        total_reward = engagement_component + response_component
        
        if verbose:
            print(f"💬 QUESTION REWARD: {total_reward:.2f} "
                  f"(engagement={engagement_component:.2f}, response={response_component:.2f})")
        
        return total_reward

    def step(self, action_dict):
        """Execute one step in the SMDP environment"""
        # Apply action masking
        masked_action = self._apply_action_masks(action_dict)
        
        # Extract actions
        option_idx = masked_action["option"]
        subaction_idx = masked_action["subaction"]
        terminate_option = masked_action["terminate_option"]
        
        # Get available options and subactions
        available_options = self._get_available_options()
        option = available_options[option_idx]
        available_subactions = self._get_available_subactions(option)
        
        # Safety: if no subactions available, use the first subaction from the full list
        if not available_subactions:
            available_subactions = self.subactions[option]
        
        # Clamp subaction index to valid range
        if subaction_idx >= len(available_subactions):
            subaction_idx = 0
        
        subaction = available_subactions[subaction_idx]
        
        # FIX: Check if this is effectively a transition action (handles coarse option config)
        # In coarse config, "SuggestMove" subaction under "Engage" option = transition
        is_transition_action = option == "OfferTransition" or subaction == "SuggestMove"
        
        # Handle option termination and transitions (Option-Critic style)
        if self.current_option != option or (terminate_option and self.current_option is not None):
            # Option is switching/terminating
            self.current_option = option
            self.option_start_turn = self.turn_count
            self.turns_in_option = 0
        else:
            # Continue current option
            self.turns_in_option += 1
        
        # Update action usage (per flat subaction)
        self.actions_used[subaction] += 1
        if option == "Explain":
            self.consecutive_explain_turns += 1
        else:
            self.consecutive_explain_turns = 0

        # Track ENF usage in rolling window for diminishing returns
        self._enf_history.append(1 if subaction == "ExplainNewFact" else 0)
        
        # Get current exhibit ID for response generation and reward calculation
        ex_id = self._get_current_exhibit()
        
        # Generate agent response using dialogue planner and LLM
        agent_response, target_exhibit = self._generate_agent_response(option, subaction)
        
        # Store agent utterance in dialogue history
        self._update_dialogue_history("agent", agent_response)
        
        # === SIMPLE FACT EXTRACTION WITH VALIDATION ===
        # Find [ID] patterns in agent response
        fact_ids_mentioned = re.findall(r'\[([A-Z]{2}_\d{3})\]', agent_response)
        
        # Check verbose mode
        import os
        verbose = os.environ.get('HRL_VERBOSE', '0') == '1'
        
        # Get all valid fact IDs for current exhibit
        valid_fact_ids = set()
        for fact_with_id in self.knowledge_graph.get_exhibit_facts(ex_id):
            fact_id = self.knowledge_graph.extract_fact_id(fact_with_id)
            if fact_id:
                valid_fact_ids.add(fact_id)
        
        # Track NEW and VALID facts only
        new_fact_ids = []
        hallucinated_ids = []
        for fact_id in fact_ids_mentioned:
            # Check if fact ID is valid for this exhibit
            if fact_id not in valid_fact_ids:
                hallucinated_ids.append(fact_id)
                if verbose:
                    print(f"🚨 HALLUCINATED: [{fact_id}] (not a real fact for {ex_id})")
                continue
            
            # Check if already mentioned
            if fact_id not in self.facts_mentioned_per_exhibit[ex_id]:
                new_fact_ids.append(fact_id)
                self.facts_mentioned_per_exhibit[ex_id].add(fact_id)
                self.facts_mentioned_this_turn.add(fact_id)
                if verbose:
                    print(f"✅ NEW FACT: [{fact_id}]")
            else:
                if verbose:
                    print(f"🔁 REPEAT FACT: [{fact_id}]")
        
        # ===== CHECK FOR AUTOMATIC COMPLETION (all exhibits at 100%) =====
        # Check if all exhibits have reached 100% coverage
        coverage = self._get_museum_exhibit_coverage()
        all_exhibits_complete = all(
            cov_data["coverage"] >= 1.0 
            for cov_data in coverage.values() 
            if cov_data["total"] > 0  # Only check exhibits with facts
        )
        
        # If all exhibits are complete, automatically conclude the episode
        # (unless agent already chose to conclude manually)
        completion_bonus = 0.0
        auto_concluded = False
        if all_exhibits_complete and option != "Conclude":  # Don't override if already concluded manually
            auto_concluded = True
            # Generate automatic conclusion remark
            from src.utils.dialogue_planner import build_wrap_up_prompt
            
            # Get context for conclusion
            all_facts = self.knowledge_graph.get_exhibit_facts(ex_id)
            mentioned_facts = [f for f in all_facts
                              if self.knowledge_graph.extract_fact_id(f) in self.facts_mentioned_per_exhibit[ex_id]]
            
            # Build context section (simplified for conclusion)
            context_section = f"[CONTEXT] Museum guide at: {ex_id}\n"
            if self.last_user_utterance:
                context_section += f"Last visitor comment: {self.last_user_utterance}\n"
            
            # Build conclusion prompt
            current_completion = 1.0  # All exhibits are complete
            conclusion_prompt = build_wrap_up_prompt(
                ex_id=ex_id,
                context_section=context_section,
                facts_all=all_facts,
                facts_used=mentioned_facts,
                current_completion=current_completion
            )
            
            # Generate conclusion remark
            import time
            import os
            start_time = time.time()
            
            # Check for template mode - skip LLM if enabled
            if os.environ.get('HRL_TEMPLATE_MODE') == '1':
                # Use template-based conclusion (no LLM call)
                conclusion_templates = [
                    "Thank you for exploring all our exhibits today! It's been wonderful sharing these treasures with you.",
                    "We've covered everything! I hope you enjoyed discovering all the exhibits with me.",
                    "That completes our tour! It was a pleasure guiding you through our collection.",
                    "What a journey we've had! Thank you for letting me share these amazing pieces with you.",
                    "You've seen it all! I hope this tour has given you a deeper appreciation for our collection.",
                ]
                agent_response = self.np_random.choice(conclusion_templates)
                elapsed = time.time() - start_time
                if verbose:
                    print(f"[Auto-Conclusion] Template mode - using template conclusion")
                self._last_agent_llm_time = 0.0  # No LLM time
            else:
                # Use LLM-based conclusion
                if verbose:
                    print(f"[Auto-Conclusion] All exhibits at 100% coverage - generating conclusion remark...")
                    print(f"[Auto-Conclusion] PROMPT:\n{'-'*60}\n{conclusion_prompt}\n{'-'*60}")
                
                from LLM_CONFIG import get_agent_llm
                llm = get_agent_llm()
                system_prompt = """You are a natural, conversational museum guide.

IMPORTANT GUIDELINES:
- Be natural and conversational - be concise and engaging
- DO NOT quote or repeat what the visitor said verbatim (avoid phrases like "You said...", "I see you...")
- Use conversation history to maintain natural flow and continuity
- Reference past topics naturally when relevant, but don't quote them
- Be warm, informative, and engaging

Thank the visitor for exploring all exhibits. Keep it warm and brief (2-3 sentences)."""
                
                agent_response = llm.generate(conclusion_prompt, system_prompt=system_prompt).strip()
                elapsed = time.time() - start_time
                
                if verbose:
                    print(f"[Auto-Conclusion] Response received in {elapsed:.2f}s: {agent_response}")
                
                # Store timing
                self._last_agent_llm_time = elapsed
            
            # Update dialogue history with conclusion remark (replace the original response)
            # Remove the last entry (original response) and add the conclusion
            if self.dialogue_history and self.dialogue_history[-1][0] == "agent":
                self.dialogue_history.pop()
            self._update_dialogue_history("agent", agent_response)
            
            # Award flat completion bonus
            completion_bonus = self.w_completion_bonus
            if verbose:
                print(f"🎉 COMPLETION BONUS: +{completion_bonus:.2f} (all exhibits at 100% coverage!)")
            
            # Override done flag to end episode
            done = True
        
        exhibit_exhausted_initial = self._is_exhibit_exhausted(ex_id)
        
        # Clear this turn's tracking for next turn
        self.facts_mentioned_this_turn = set()

        # ===== REWARD CALCULATION (prospect theory asymmetric delta) =====
        # R_t = α·max(0,Δdwell) − β·max(0,−Δdwell) + R_terminal·coverage
        # β/α = 2.25 from Kahneman & Tversky (1979) prospect theory
        delta_dwell = self.dwell - self._previous_dwell
        trajectory_reward = (
            self.alpha * max(0.0, delta_dwell)
            - self.beta * max(0.0, -delta_dwell)
        )

        # Terminal coverage bonus — absorbs novelty signal at episode end
        exhibits_covered = 0
        terminal_bonus = 0.0
        if done:
            exhibits_covered = sum(
                1 for ex in self.exhibit_keys
                if len(self.facts_mentioned_per_exhibit[ex]) > 0
            )
            terminal_bonus = self.terminal_coverage_weight * (exhibits_covered / self.n_exhibits)

        step_reward = trajectory_reward + terminal_bonus - self.deliberation_cost

        if verbose:
            print(f"📈 TRAJECTORY REWARD: {trajectory_reward:.3f} "
                  f"(Δdwell={delta_dwell:+.3f}, dwell={self.dwell:.3f}, prev={self._previous_dwell:.3f})")
            if terminal_bonus > 0:
                print(f"🏁 TERMINAL BONUS: +{terminal_bonus:.3f} "
                      f"({exhibits_covered}/{self.n_exhibits} exhibits covered)")
        
        # Track reward components for analysis
        self.trajectory_reward_sum = getattr(self, 'trajectory_reward_sum', 0.0) + trajectory_reward
        self.terminal_bonus_sum = getattr(self, 'terminal_bonus_sum', 0.0) + terminal_bonus
        self.deliberation_sum -= self.deliberation_cost
        
        # Store current dwell for NEXT turn's reward
        self._previous_dwell = self.dwell
        
        # Update session reward
        self.session_reward += step_reward
        
        # Update exhibits covered count
        exhibits_covered = sum(1 for exp in self.explained if exp > 0)
        
        # Update turn count
        self.turn_count += 1
        
        # Check termination conditions
        # Note: done may already be True if auto_concluded above
        if auto_concluded:
            done = True  # Already set above, but ensure it's True
        else:
            done = (option == "Conclude" or self.turn_count >= self.max_turns)
        
        # Calculate current exhibit completion rate for transition logic
        current_exhibit_completion = len(self.facts_mentioned_per_exhibit[ex_id]) / len(self.knowledge_graph.get_exhibit_facts(ex_id)) if len(self.knowledge_graph.get_exhibit_facts(ex_id)) > 0 else 0.0
        
        # Check if exhibit is exhausted (no new facts available)
        all_facts = self.knowledge_graph.get_exhibit_facts(ex_id)
        mentioned_ids = self.facts_mentioned_per_exhibit[ex_id]
        unmentioned_facts = [f for f in all_facts 
                           if self.knowledge_graph.extract_fact_id(f) not in mentioned_ids]
        exhibit_exhausted = len(unmentioned_facts) == 0
        
        # Calculate target exhibit completion (for transition quality assessment)
        target_exhibit_completion = 0.0
        target_exhibit_exhausted = False
        if target_exhibit and target_exhibit in self.exhibit_keys:
            target_facts = self.knowledge_graph.get_exhibit_facts(target_exhibit)
            target_mentioned = len(self.facts_mentioned_per_exhibit[target_exhibit])
            target_exhibit_completion = target_mentioned / len(target_facts) if len(target_facts) > 0 else 0.0
            target_exhibit_exhausted = self._is_exhibit_exhausted(target_exhibit)
        
        # Build info dictionary
        info = {
            "agent_utterance": agent_response,
            "option": "Conclude" if auto_concluded else option,  # Mark as Conclude if auto-concluded
            "subaction": "WrapUp" if auto_concluded else subaction,  # Mark as WrapUp if auto-concluded
            "terminated_option": terminate_option,
            "reward_trajectory": trajectory_reward,
            "reward_terminal_bonus": terminal_bonus,
            "reward_deliberation_cost": -self.deliberation_cost,
            "auto_concluded": auto_concluded,  # Whether episode was automatically concluded due to 100% coverage
            "dwell": self.dwell,
            "total_reward": step_reward,
            "session_reward": self.session_reward,
            "turn": self.turn_count,
            "facts_shared": len(new_fact_ids),
            "facts_mentioned_in_utterance": new_fact_ids,
            "hallucinated_facts": hallucinated_ids,
            "exhibits_covered": exhibits_covered,
            "current_option": self.current_option,
            "turns_in_option": self.turns_in_option,
            "current_focus": self.focus,
            "current_exhibit": ex_id,
            "target_exhibit": target_exhibit,  # For OfferTransition: where agent wants to go
            "current_exhibit_completion": current_exhibit_completion,  # For transition probability
            "exhibit_exhausted": exhibit_exhausted,  # Whether current exhibit has no new facts
            "beta_target": 1.0 if (current_exhibit_completion >= 0.7 or exhibit_exhausted) else 0.0,  # Heuristic termination target
            "target_exhibit_completion": target_exhibit_completion,  # Target exhibit completion (for quality assessment)
            "target_exhibit_exhausted": target_exhibit_exhausted,  # Whether target exhibit is exhausted
            "available_options": self._get_available_options(),
            "available_subactions": self._get_available_subactions(option),
            "action_masks": self.get_action_masks(),
            "agent_llm_time": getattr(self, '_last_agent_llm_time', 0.0)
        }
        
        # Add component breakdown when episode ends
        if done:
            info['component_breakdown'] = {
                'total_reward': self.session_reward,
                'trajectory_reward_contribution': self.trajectory_reward_sum,
                'terminal_bonus_contribution': self.terminal_bonus_sum,
                'deliberation_contribution': self.deliberation_sum  # H1 termination tuning
            }
        
        # ===== DialogueBERT INSIGHTS (for visualization) =====
        # Note: DialogueBERT insights will be added after user state is updated
        
        return self._get_obs(), step_reward, done, False, info

    # ===== OBSERVATION CONSTRUCTION =====
    
    def _get_obs(self):
        """
        Construct observation vector with DialogueBERT as per paper formalization (Section 4.6):
        s_t = [f_t, h_t, i_t, c_t]
        
        Where:
        - f_t: focus vector (one-hot over exhibits + no-focus)
        - h_t: dialogue history (exhibits explained + per-subaction counts)
        - i_t: projected intent embedding = P * DialogueBERT(u_t, role="user", turn_number)
        - c_t: projected dialogue context = P * DialogueBERT(recent_utterances with turn/role)
        
        DialogueBERT includes turn and role embeddings:
        - Turn embeddings: Track turn position in dialogue (0-indexed)
        - Role embeddings: Distinguish user (0) vs system/agent (1)
        
        Projection: 768-d -> 64-d using fixed matrix P
        """
        
        # 1. Focus vector f_t (n_exhibits + 1-d, e.g., 6-d for 5 exhibits)
        focus_snapshot = np.zeros(self.n_exhibits + 1)
        if self.focus > 0:
            focus_snapshot[self.focus - 1] = 1.0
        else:
            focus_snapshot[-1] = 1.0  # No focus
        
        # 2. Dialogue history vector h_t (n_exhibits + n_subactions-d)
        # First n_exhibits: exhibit completion ratios (0-1 for facts shared per exhibit)
        # Next n_subactions: per-subaction usage (normalized counts)
        history = np.zeros(self.n_exhibits + len(self._all_subactions))

        # Get current exhibit completion data
        coverage = self._get_museum_exhibit_coverage()

        # Exhibit completion ratios (0-1 for facts shared)
        for i, exhibit_name in enumerate(self.exhibit_keys):
            completion_ratio = coverage.get(exhibit_name, {"coverage": 0.0})["coverage"]
            history[i] = completion_ratio

        # Per-subaction usage (normalized counts)
        total_actions = sum(self.actions_used.values()) or 1
        for i, sa in enumerate(self._all_subactions):
            history[self.n_exhibits + i] = self.actions_used[sa] / total_actions
        
        # 3. Intent embedding i_t (64-d projected from 768-d)
        # Get DialogueBERT embedding: e_t = DialogueBERT(u_t, role="user", turn_number)
        _bert_start = time.perf_counter()
        intent_recognizer = get_dialoguebert_recognizer()
        # Get turn number from last user utterance in dialogue history
        # Find the most recent user utterance's turn number
        current_turn = 0
        if self.dialogue_history:
            # Look for last user utterance in history
            for entry in reversed(self.dialogue_history):
                if len(entry) >= 3 and entry[0] == "user":
                    current_turn = entry[2]  # Get turn_number
                    break
            # If no user utterance found, use current counter
            if current_turn == 0 and self.dialogue_turn_counter > 0:
                current_turn = self.dialogue_turn_counter
        
        intent_embedding_768 = intent_recognizer.get_intent_embedding(
            self.last_user_utterance, role="user", turn_number=current_turn
        )
        
        # Apply projection: i_t = P * e_t
        intent_embedding_64 = np.dot(self.projection_matrix, intent_embedding_768).astype(np.float32)
        
        # 4. Dialogue context c_t (64-d projected from 768-d)
        # Get DialogueBERT context embedding (average of last 3 turns)
        # dialogue_history already contains (role, utterance) tuples
        dialogue_context_768 = intent_recognizer.get_dialogue_context(
            self.dialogue_history, max_turns=3
        )
        self._last_bert_time = time.perf_counter() - _bert_start
        
        # Apply projection: c_t = P * context_768
        dialogue_context_64 = np.dot(self.projection_matrix, dialogue_context_768).astype(np.float32)
        
        # Track embeddings for insights (keep full 768-d for visualization)
        prev_intent = getattr(self, '_last_intent_embedding', np.zeros(768, dtype=np.float32))
        prev_context = getattr(self, '_last_dialogue_context', np.zeros(768, dtype=np.float32))
        self._prev_intent_embedding = prev_intent.astype(np.float32)
        self._prev_dialogue_context = prev_context.astype(np.float32)
        self._last_intent_embedding = intent_embedding_768.astype(np.float32)
        self._last_dialogue_context = dialogue_context_768.astype(np.float32)
        
        # 5. Subaction availability indicators (4-d binary vector)
        # [0]: ExplainNewFact available (1.0) or masked (0.0)
        # [1]: AskOpinion available (1.0) or masked (0.0)
        # [2]: RecoverEngagement available (1.0) or masked (0.0)
        # [3]: Exhibit exhausted indicator (1.0 if exhausted, 0.0 otherwise)
        subaction_availability = np.zeros(4, dtype=np.float32)
        available_subs = self._get_available_subactions("Explain")
        subaction_availability[0] = 1.0 if "ExplainNewFact" in available_subs else 0.0
        available_subs_ask = self._get_available_subactions("AskQuestion")
        subaction_availability[1] = 1.0 if "AskOpinion" in available_subs_ask else 0.0
        available_subs_engage = self._get_available_subactions("Engage")
        subaction_availability[2] = 1.0 if "RecoverEngagement" in available_subs_engage else 0.0
        # Get current exhibit first to avoid warning when focus=0 (passes exhibit explicitly)
        current_exhibit = self._get_current_exhibit()  # Get once, handles focus=0 gracefully
        subaction_availability[3] = 1.0 if self._is_exhibit_exhausted(current_exhibit) else 0.0
        
        # Build state components list
        state_components = [
            focus_snapshot,        # (n_exhibits + 1)-d
            history,               # (n_exhibits + 4)-d
            intent_embedding_64,   # 64-d
            dialogue_context_64,   # 64-d
            subaction_availability # 4-d
        ]

        # 6. Response type one-hot (6-d, optional)
        if self.response_type_feature:
            response_type_onehot = np.zeros(len(self.response_type_labels), dtype=np.float32)
            if self._last_user_response_type in self.response_type_labels:
                idx = self.response_type_labels.index(self._last_user_response_type)
                response_type_onehot[idx] = 1.0
            else:
                # Default to "statement" for unknown types
                response_type_onehot[self.response_type_labels.index("statement")] = 1.0
            state_components.append(response_type_onehot)

        obs = np.concatenate(state_components).astype(np.float32)

        return obs

    # ===== MUSEUM OVERVIEW =====
    
    def _get_museum_exhibit_coverage(self):
        """
        Calculate coverage stats for all exhibits in the museum.
        Returns dict: {exhibit_name: {"total": int, "mentioned": int, "coverage": float}}
        """
        coverage = {}
        for exhibit_name in self.exhibit_keys:
            all_facts = self.knowledge_graph.get_exhibit_facts(exhibit_name)
            total_facts = len(all_facts)
            mentioned_facts = len(self.facts_mentioned_per_exhibit[exhibit_name])
            coverage[exhibit_name] = {
                "total": total_facts,
                "mentioned": mentioned_facts,
                "coverage": mentioned_facts / total_facts if total_facts > 0 else 0.0
            }
        return coverage
    
    def _select_least_discussed_exhibit(self, current_exhibit: str) -> str:
        """
        Select the exhibit with the most remaining facts (excluding current exhibit).
        Per paper.tex Section 4.8: select e* = argmax_e (|F_e^total| - |F_e^used|)
        
        Prioritizes completely empty exhibits (0 facts mentioned) first, then exhibits
        with the most remaining facts. Explicitly avoids exhausted exhibits (no unmentioned facts)
        unless there are no other options. If multiple exhibits tie, choose randomly among them.
        """
        coverage = self._get_museum_exhibit_coverage()
        
        # Remove current exhibit from consideration
        other_exhibits = {k: v for k, v in coverage.items() if k != current_exhibit}
        
        if not other_exhibits:
            # Fallback: stay at current exhibit (shouldn't happen)
            return current_exhibit
        
        # Calculate remaining facts and check if exhausted for all exhibits
        for k, v in other_exhibits.items():
            v["remaining"] = v["total"] - v["mentioned"]
            # Use _is_exhibit_exhausted to accurately check if exhibit has no unmentioned facts
            v["is_exhausted"] = self._is_exhibit_exhausted(k)
        
        # PRIORITY 1: Find completely empty exhibits (0 facts mentioned, but has facts available)
        # Must have total > 0 to ensure it's a real exhibit with facts, not just an exhibit with no facts
        empty_exhibits = [k for k, v in other_exhibits.items() 
                         if v["mentioned"] == 0 and v["total"] > 0]
        
        if empty_exhibits:
            # If there are empty exhibits, choose randomly among them
            import os
            verbose = os.environ.get('HRL_VERBOSE', '0') == '1'
            if verbose:
                print(f"\n🔍 TRANSITION SELECTION (from {current_exhibit}):")
                print(f"  ✓ Found {len(empty_exhibits)} empty exhibit(s): {empty_exhibits}")
                for k, v in sorted(other_exhibits.items()):
                    status = "EXHAUSTED" if v["is_exhausted"] else f"{v['remaining']} remaining"
                    marker = "⭐ EMPTY" if k in empty_exhibits else ""
                    print(f"  {k}: {v['mentioned']}/{v['total']} facts, {status} {marker}")
            selected = self.np_random.choice(empty_exhibits)
            if verbose:
                print(f"  → Selected: {selected} (random choice from empty exhibits)")
            return selected
        
        # PRIORITY 2: Filter out exhausted exhibits (unless all are exhausted)
        # CRITICAL: Double-check that we didn't miss any empty exhibits (safety check)
        remaining_empty = [k for k, v in other_exhibits.items() 
                          if v["mentioned"] == 0 and v["total"] > 0 and k not in empty_exhibits]
        if remaining_empty:
            # This should never happen, but if it does, use them
            import os
            verbose = os.environ.get('HRL_VERBOSE', '0') == '1'
            if verbose:
                print(f"⚠️ WARNING: Found {len(remaining_empty)} empty exhibits that were missed: {remaining_empty}")
            return self.np_random.choice(remaining_empty)
        
        non_exhausted_exhibits = {k: v for k, v in other_exhibits.items() if not v["is_exhausted"]}
        
        if non_exhausted_exhibits:
            # Use only non-exhausted exhibits
            candidates = non_exhausted_exhibits
        else:
            # Edge case: all other exhibits are exhausted, use all of them
            # (This shouldn't happen in normal operation, but handle gracefully)
            candidates = other_exhibits
        
        # Find maximum remaining facts among candidates
        max_remaining = max(v["remaining"] for v in candidates.values())
        
        # Get all exhibits with maximum remaining facts
        exhibits_with_most_remaining = [k for k, v in candidates.items() if v["remaining"] == max_remaining]
        
        # CRITICAL SAFETY CHECK: Verify we're not selecting a non-empty exhibit when empty ones exist
        # This should never happen if the logic above is correct, but double-check
        final_empty_check = [k for k, v in other_exhibits.items() 
                           if v["mentioned"] == 0 and v["total"] > 0]
        if final_empty_check and exhibits_with_most_remaining:
            # If we have empty exhibits but are about to select a non-empty one, that's a bug!
            selected_exhibit = exhibits_with_most_remaining[0]
            selected_mentioned = other_exhibits[selected_exhibit]["mentioned"]
            if selected_mentioned > 0:
                import os
                verbose = os.environ.get('HRL_VERBOSE', '0') == '1'
                if verbose:
                    print(f"\n🚨 CRITICAL BUG DETECTED: About to select {selected_exhibit} ({selected_mentioned} facts) when empty exhibits exist: {final_empty_check}")
                    print(f"   This should never happen! Using empty exhibit instead.")
                # Force selection of empty exhibit
                return self.np_random.choice(final_empty_check)
        
        # Debug logging (if verbose mode enabled)
        import os
        verbose = os.environ.get('HRL_VERBOSE', '0') == '1'
        if verbose:
            print(f"\n🔍 TRANSITION SELECTION (from {current_exhibit}):")
            for k, v in sorted(other_exhibits.items()):
                status = "EXHAUSTED" if v["is_exhausted"] else f"{v['remaining']} remaining"
                marker = "⭐ EMPTY" if (v["mentioned"] == 0 and v["total"] > 0) else ""
                print(f"  {k}: {v['mentioned']}/{v['total']} facts, {status} {marker}")
            print(f"  → Selected: {exhibits_with_most_remaining[0] if exhibits_with_most_remaining else 'NONE'}")
            if non_exhausted_exhibits:
                print(f"  ✓ Filtered out {len(other_exhibits) - len(non_exhausted_exhibits)} exhausted exhibit(s)")
        
        # Random choice if multiple exhibits tie
        return self.np_random.choice(exhibits_with_most_remaining)
    
    # ===== ACTION MASKING =====
    
    def _get_available_options(self):
        """Get available options based on current state (action masking)"""
        available_options = self.options.copy()
        coverage_dict = self._get_museum_exhibit_coverage() if self.exhibit_keys else {}
        
        # Count total facts mentioned across all exhibits
        total_facts_mentioned = sum(len(ids) for ids in self.facts_mentioned_per_exhibit.values())
        
        # Check if enough facts have been shared to allow conclusion
        if total_facts_mentioned < self.min_facts_before_conclude:
            if "Conclude" in available_options:
                available_options.remove("Conclude")
                # Diagnostic logging
                import os
                if os.environ.get('HRL_VERBOSE', '0') == '1':
                    print(f"🔒 CONCLUDE MASKED: Only {total_facts_mentioned} facts shared (need {self.min_facts_before_conclude})")
        
        # Check if enough exhibits have been covered
        exhibits_covered = sum(1 for ids in self.facts_mentioned_per_exhibit.values() if len(ids) > 0)
        if exhibits_covered < self.min_exhibits_before_conclude:
            if "Conclude" in available_options:
                available_options.remove("Conclude")
                # Diagnostic logging
                import os
                if os.environ.get('HRL_VERBOSE', '0') == '1':
                    print(f"🔒 CONCLUDE MASKED: Only {exhibits_covered} exhibits covered (need {self.min_exhibits_before_conclude})")
        
        # NOTE: AskQuestion masking REMOVED per state machine design
        # Agent should learn when to ask questions via dwell signal from simulator
        # State machine will provide clear reward signals for appropriate question timing
        
        # Mask Explain option when exhibit is exhausted (ONLY for reward-aligned configs)
        # In reward-aligned configs (coarse_4opt, coarse_3opt, coarse), Explain only has ExplainNewFact.
        # When exhibit is exhausted, the agent should NOT be able to select Explain at all.
        # This forces the agent to Transition or Engage instead of wasting turns.
        # Function-based baseline (medium) has RepeatFact/ClarifyFact under Explain, so this doesn't apply.
        if "Explain" in available_options:
            explain_subactions = self.subactions.get("Explain", [])
            # Only apply to reward-aligned configs where Explain = [ExplainNewFact] only
            if explain_subactions == ["ExplainNewFact"]:
                current_exhibit = self.exhibit_keys[self.focus - 1] if self.focus > 0 else None
                if current_exhibit and self._is_exhibit_exhausted(current_exhibit):
                    available_options.remove("Explain")
                    import os
                    if os.environ.get('HRL_VERBOSE', '0') == '1':
                        print(f"🔒 EXPLAIN MASKED: Exhibit {current_exhibit} exhausted (reward-aligned config)")
        
        return available_options

    def _get_available_subactions(self, option):
        """Get available subactions for a given option (action masking)"""
        if option not in self.subactions:
            return []
        
        subactions = self.subactions[option].copy()
        
        # Only apply basic fact availability masking for Explain option
        if option == "Explain":
            current_exhibit = self.exhibit_keys[self.focus - 1] if self.focus > 0 else None
            if current_exhibit:
                all_facts = self.knowledge_graph.get_exhibit_facts(current_exhibit)
                mentioned_ids = self.facts_mentioned_per_exhibit[current_exhibit]
                
                # Check if there are unmentioned facts
                has_unmentioned = any(self.knowledge_graph.extract_fact_id(f) not in mentioned_ids 
                                     for f in all_facts)
                
                # Remove ExplainNewFact if all facts mentioned
                if not has_unmentioned and "ExplainNewFact" in subactions:
                    subactions.remove("ExplainNewFact")
                
                # Remove RepeatFact if no facts mentioned yet
                if len(mentioned_ids) == 0 and "RepeatFact" in subactions:
                    subactions.remove("RepeatFact")
        
        return subactions

    def _apply_action_masks(self, action_dict):
        """Apply action masks to ensure valid actions"""
        masked_action = action_dict.copy()
        
        # Mask options
        available_options = self._get_available_options()
        if masked_action["option"] >= len(available_options):
            masked_action["option"] = 0  # Default to first available
        
        # Mask subactions
        option = available_options[masked_action["option"]]
        available_subactions = self._get_available_subactions(option)
        if masked_action["subaction"] >= len(available_subactions):
            masked_action["subaction"] = 0  # Default to first available
        
        return masked_action

    def get_action_masks(self):
        """Get current action masks for training"""
        available_options = self._get_available_options()
        option_mask = [1 if opt in available_options else 0 for opt in self.options]
        
        # Get subaction mask for first available option
        if available_options:
            first_option = available_options[0]
            available_subactions = self._get_available_subactions(first_option)
            subaction_mask = [1 if sub in available_subactions else 0 for sub in self.subactions[first_option]]
        else:
            subaction_mask = [0] * max(len(subacts) for subacts in self.subactions.values())
        
        return {
            "option_mask": option_mask,
            "subaction_mask": subaction_mask
        }

    # ===== AGENT RESPONSE GENERATION =====
    
    def _generate_agent_response(self, option: str, subaction: str) -> tuple:
        """Generate agent response using dialogue planner and LLM
        
        Returns:
            tuple: (response_str, target_exhibit_str or None)
        """
        try:
            # Get current exhibit ID
            ex_id = self._get_current_exhibit()
            all_facts = self.knowledge_graph.get_exhibit_facts(ex_id)
            
            # SIMPLE: Get unmentioned facts for this exhibit
            mentioned_ids = self.facts_mentioned_per_exhibit[ex_id]
            unmentioned_facts = [f for f in all_facts 
                               if self.knowledge_graph.extract_fact_id(f) not in mentioned_ids]
            
            # FOR TRANSITION: Calculate target exhibit and museum coverage
            # FIX: Also handle coarse config where "Engage" + "SuggestMove" = transition
            target_exhibit = None
            coverage_dict = None
            is_transition_action = option == "OfferTransition" or subaction == "SuggestMove"
            if is_transition_action:
                # Always select the least discussed/empty exhibit for transitions
                target_exhibit = self._select_least_discussed_exhibit(ex_id)
                coverage_dict = self._get_museum_exhibit_coverage()
                
                # Validate that target exhibit is valid and different from current
                # CRITICAL: If validation fails, re-run selection instead of random fallback
                if target_exhibit not in self.exhibit_keys:
                    import os
                    verbose = os.environ.get('HRL_VERBOSE', '0') == '1'
                    if verbose:
                        print(f"⚠️ WARNING: Invalid target exhibit '{target_exhibit}', re-running selection")
                    # Re-run selection to get a valid exhibit (don't use random fallback)
                    target_exhibit = self._select_least_discussed_exhibit(ex_id)
                    # If still invalid, use last resort
                    if target_exhibit not in self.exhibit_keys:
                        other_exhibits = [e for e in self.exhibit_keys if e != ex_id]
                        if other_exhibits:
                            # Even in last resort, try to select empty exhibit if possible
                            coverage = self._get_museum_exhibit_coverage()
                            empty = [e for e in other_exhibits 
                                   if coverage.get(e, {}).get("mentioned", 1) == 0 
                                   and coverage.get(e, {}).get("total", 0) > 0]
                            target_exhibit = empty[0] if empty else other_exhibits[0]
                        else:
                            target_exhibit = ex_id  # Last resort: stay at current
                
                # Ensure target is different from current (shouldn't happen, but safety check)
                if target_exhibit == ex_id:
                    import os
                    verbose = os.environ.get('HRL_VERBOSE', '0') == '1'
                    if verbose:
                        print(f"⚠️ WARNING: Target exhibit same as current, re-running selection")
                    # Re-run selection to get a different exhibit
                    target_exhibit = self._select_least_discussed_exhibit(ex_id)
                    # If still same, use last resort
                    if target_exhibit == ex_id:
                        other_exhibits = [e for e in self.exhibit_keys if e != ex_id]
                        if other_exhibits:
                            # Even in last resort, try to select empty exhibit if possible
                            coverage = self._get_museum_exhibit_coverage()
                            empty = [e for e in other_exhibits 
                                   if coverage.get(e, {}).get("mentioned", 1) == 0 
                                   and coverage.get(e, {}).get("total", 0) > 0]
                            target_exhibit = empty[0] if empty else other_exhibits[0]
            
            # FINAL VALIDATION for transition actions: Ensure we never select non-empty when empty exists
            if is_transition_action and target_exhibit:
                final_coverage = self._get_museum_exhibit_coverage()
                target_stats = final_coverage.get(target_exhibit, {})
                target_mentioned = target_stats.get("mentioned", 0)
                target_total = target_stats.get("total", 0)
                
                # Check if there are empty exhibits we should be using instead
                empty_exhibits_final = [k for k, v in final_coverage.items() 
                                       if k != ex_id and v.get("mentioned", 1) == 0 and v.get("total", 0) > 0]
                
                if empty_exhibits_final and target_mentioned > 0:
                    import os
                    verbose = os.environ.get('HRL_VERBOSE', '0') == '1'
                    if verbose:
                        print(f"\n🚨 FINAL VALIDATION FAILED: Target {target_exhibit} has {target_mentioned} facts, but empty exhibits exist: {empty_exhibits_final}")
                        print(f"   FORCING selection of empty exhibit instead!")
                    # Force selection of empty exhibit
                    target_exhibit = self.np_random.choice(empty_exhibits_final)
                    if verbose:
                        print(f"   → Corrected to: {target_exhibit}")
                
                # CRITICAL FIX: NEVER transition to exhausted exhibit
                if self._is_exhibit_exhausted(target_exhibit):
                    import os
                    verbose = os.environ.get('HRL_VERBOSE', '0') == '1'
                    if verbose:
                        print(f"\n🚨 EXHAUSTED EXHIBIT BLOCKED: {target_exhibit} is fully covered!")
                    # Force re-selection to non-exhausted exhibit
                    non_exhausted = [e for e in self.exhibit_keys 
                                    if e != ex_id and not self._is_exhibit_exhausted(e)]
                    if non_exhausted:
                        target_exhibit = self.np_random.choice(non_exhausted)
                        if verbose:
                            print(f"   → Redirected to: {target_exhibit}")
                    # If all exhausted, keep the selection (rare edge case - tour nearly complete)
            
            # Get facts based on action type
            if option == "Explain":
                # For Explain actions, show unmentioned facts
                facts_for_prompt = unmentioned_facts
            else:
                # For other actions, show all facts (mentioned + unmentioned)
                facts_for_prompt = all_facts

            # Get mentioned facts for RepeatFact actions
            mentioned_facts = [f for f in all_facts
                              if self.knowledge_graph.extract_fact_id(f) in self.facts_mentioned_per_exhibit[ex_id]]

            # Use compositional templates if template mode is enabled (skip LLM entirely)
            import os
            import time as time_mod
            if os.environ.get('HRL_TEMPLATE_MODE') == '1':
                _template_start = time_mod.perf_counter()
                from src.utils.agent_templates import generate_agent_response_template
                response = generate_agent_response_template(
                    option=option,
                    subaction=subaction,
                    unmentioned_facts=unmentioned_facts,
                    mentioned_facts=mentioned_facts,
                    target_exhibit=target_exhibit,
                    rng=self.np_random,
                    visitor_state=self._last_visitor_state,
                    last_visitor_utterance=self.last_user_utterance
                )
                self._last_agent_template_time = time_mod.perf_counter() - _template_start
                self._last_agent_llm_time = 0.0  # No LLM time
                return response.strip(), target_exhibit

            # Build prompt with dialogue history
            prompt = build_prompt(
                option=option,
                subaction=subaction,
                ex_id=ex_id,
                last_utt=self.last_user_utterance,
                facts_all=facts_for_prompt,
                facts_used=mentioned_facts,  # Pass mentioned facts for RepeatFact
                selected_fact=None,
                dialogue_history=self.dialogue_history,
                exhibit_names=self.exhibit_keys,
                knowledge_graph=self.knowledge_graph,
                target_exhibit=target_exhibit,  # NEW: For transitions
                coverage_dict=coverage_dict    # NEW: For transitions
            )
            
            # Generate response using LLM from centralized config
            import time
            import os
            verbose = os.environ.get('HRL_VERBOSE', '0') == '1'
            start_time = time.time()
            if verbose:
                print(f"[Agent LLM] Generating response for {option}/{subaction}...", flush=True)
                if is_transition_action:
                    print(f"[Agent LLM] TARGET EXHIBIT: {target_exhibit}", flush=True)
                print(f"[Agent LLM] PROMPT:\n{'-'*60}\n{prompt}\n{'-'*60}", flush=True)
            from LLM_CONFIG import get_agent_llm
            llm = get_agent_llm()
            
            # Adjust system prompt based on subaction
            if subaction == "ExplainNewFact":
                system_prompt = """You are a natural, conversational museum guide.

IMPORTANT GUIDELINES:
- Be natural and conversational - be concise and engaging
- DO NOT quote or repeat what the visitor said verbatim (avoid phrases like "You said...", "I see you...")
- Use conversation history to maintain natural flow and continuity
- Reference past topics naturally when relevant, but don't quote them

🚨 CRITICAL RULE: Use ONLY fact IDs from the AVAILABLE FACTS list in the prompt.
- If the list shows [DM_001], [DM_002], [DM_003] - use ONLY these IDs
- NEVER create new IDs like [DM_014], [DM_020], etc.
- You MUST select at least 1 fact from the list provided
- Even if facts seem off-topic, use them - do NOT make up relevant-sounding facts

Example: "Created by Gerrit Dou in 1635 [TU_003]"
Keep responses 2-3 sentences. NO HALLUCINATED FACT IDs."""
            elif subaction in ("RepeatFact", "ClarifyFact"):
                system_prompt = """You are a natural, conversational museum guide.

IMPORTANT GUIDELINES:
- Be natural and conversational - be concise and engaging
- DO NOT quote or repeat what the visitor said verbatim (avoid phrases like "You said...", "I see you...")
- Use conversation history to maintain natural flow and continuity
- Reference past topics naturally when relevant, but don't quote them

For RepeatFact: Use the EXACT fact ID provided in the prompt - do NOT modify it or create new ones.
For ClarifyFact: Do NOT include fact IDs - just clarify the concept naturally.
Keep responses 2-3 sentences."""
            else:
                system_prompt = """You are a natural, conversational museum guide.

IMPORTANT GUIDELINES:
- Be natural and conversational - be concise and engaging
- DO NOT quote or repeat what the visitor said verbatim (avoid phrases like "You said...", "I see you...")
- Use conversation history to maintain natural flow and continuity
- Reference past topics naturally when relevant, but don't quote them

Keep responses 2-3 sentences."""
            
            response = llm.generate(prompt, system_prompt=system_prompt)
            elapsed = time.time() - start_time
            if verbose:
                print(f"[Agent LLM] Response received in {elapsed:.2f}s ({len(response)} chars)", flush=True)
            
            # Store prompt and timing for debugging and logging
            self._last_llm_prompt = prompt
            self._last_agent_system_prompt = system_prompt
            self._last_agent_llm_time = elapsed
            
            return response.strip(), target_exhibit
            
        except Exception as e:
            print(f"Error generating agent response: {e}")
            raise e



    # ===== HELPER METHODS =====
    
    def _get_current_exhibit(self) -> str:
        """Get current exhibit ID based on focus"""
        if self.focus > 0 and self.focus <= len(self.exhibit_keys):
            return self.exhibit_keys[self.focus - 1]
        # Fallback to first exhibit if focus is invalid (should not happen in normal operation)
        if len(self.exhibit_keys) > 0:
            print(f"⚠️  WARNING: Invalid focus {self.focus}, falling back to {self.exhibit_keys[0]}")
            return self.exhibit_keys[0]
        return "Unknown"  # Should never happen if knowledge graph is loaded
    
    def _is_exhibit_exhausted(self, exhibit_id: Optional[str] = None) -> bool:
        """Check if all facts for an exhibit have been mentioned"""
        ex_id = exhibit_id or self._get_current_exhibit()
        if ex_id not in self.facts_mentioned_per_exhibit:
            return False
        mentioned = self.facts_mentioned_per_exhibit[ex_id]
        all_facts = self.knowledge_graph.get_exhibit_facts(ex_id)
        for fact in all_facts:
            fact_id = self.knowledge_graph.extract_fact_id(fact)
            if fact_id not in mentioned:
                return False
        return True
    
    


    def record_successful_transition(self):
        """Record that a transition was successful (called from training loop after simulator responds).
        
        This is used for the 3-turn exemption rule (paper.tex line 707): if a transition succeeds,
        the insufficiency penalty is not applied for the next 3 turns.
        """
        self.last_successful_transition_turn = self.turn_count
    
    def _update_dialogue_history(self, role: str, utterance: str):
        """Update dialogue history with new utterance
        
        Args:
            role: 'agent' or 'user'
            utterance: The text utterance
        """
        if utterance and utterance.strip():
            # Increment turn counter for each utterance
            self.dialogue_turn_counter += 1
            self.dialogue_history.append((role, utterance, self.dialogue_turn_counter))
            # Keep only recent utterances
            if len(self.dialogue_history) > self.max_dialogue_history:
                self.dialogue_history = self.dialogue_history[-self.max_dialogue_history:]

    def update_user_state(self, focus: int = None, dwell: float = None, utterance: str = None, 
                          response_type: str = None, simulator=None, visitor_state: Optional[str] = None):
        """Update user state from simulator
        
        Args:
            focus: Current exhibit focus index
            dwell: Current dwell time (engagement signal)
            utterance: User's utterance text
            response_type: Simulator's response type (question, statement, confusion, etc.)
            simulator: Simulator instance for state synchronization
            visitor_state: Visitor's current state (for agent template responsiveness)
        """
        if focus is not None:
            self.focus = focus
        if dwell is not None:
            self.dwell = dwell
        if utterance is not None:
            self.last_user_utterance = utterance
            self._update_dialogue_history("user", utterance)
        if visitor_state is not None:
            self._last_visitor_state = visitor_state

            # Track user intent for responsiveness checking
            if utterance.strip():
                recognizer = get_dialoguebert_recognizer()
                self._last_user_intent = recognizer.classify_intent_category(utterance)
        
        # Track response type for responsiveness / repeat handling
        if response_type is not None:
            self._last_user_response_type = response_type
            if response_type == "repeat_request":
                self.repeat_request_active = True

        # NEW: Sync simulator with state information for transitions
        if simulator and hasattr(simulator, 'update_from_state'):
            # Pass current focus and any target exhibit from transition logic
            target_exhibit = None
            if self.focus > 0 and self.focus <= len(self.exhibit_keys):
                target_exhibit = self.exhibit_keys[self.focus - 1]

            simulator.update_from_state(self.focus, target_exhibit)
    



    def add_dialoguebert_insights_to_info(self, info: Dict[str, Any]):
        """Add DialogueBERT insights to info dict after user state is updated"""
        try:
            def _cos(a, b):
                an = float(np.linalg.norm(a) + 1e-8)
                bn = float(np.linalg.norm(b) + 1e-8)
                return float(np.dot(a, b) / (an * bn)) if an > 0.0 and bn > 0.0 else 0.0

            # Re-compute embeddings with updated user utterance
            recognizer = get_dialoguebert_recognizer()
            
            # Store previous embeddings
            prev_intent = getattr(self, '_last_intent_embedding', np.zeros(768, dtype=np.float32))
            prev_context = getattr(self, '_last_dialogue_context', np.zeros(768, dtype=np.float32))
            self._prev_intent_embedding = prev_intent.astype(np.float32)
            self._prev_dialogue_context = prev_context.astype(np.float32)
            
            # Compute new embeddings with current utterance
            # Find the most recent user utterance's turn number
            current_turn = 0
            if self.dialogue_history:
                # Look for last user utterance in history
                for entry in reversed(self.dialogue_history):
                    if len(entry) >= 3 and entry[0] == "user":
                        current_turn = entry[2]  # Get turn_number
                        break
                # If no user utterance found, use current counter
                if current_turn == 0 and self.dialogue_turn_counter > 0:
                    current_turn = self.dialogue_turn_counter
            
            intent_embedding = recognizer.get_intent_embedding(
                self.last_user_utterance, role="user", turn_number=current_turn
            )
            # dialogue_history already contains (role, utterance, turn_number) tuples
            dialogue_context = recognizer.get_dialogue_context(
                self.dialogue_history, max_turns=3
            )
            
            # Update stored embeddings
            self._last_intent_embedding = intent_embedding.astype(np.float32)
            self._last_dialogue_context = dialogue_context.astype(np.float32)
            
            # Compute insights
            intent_category = recognizer.classify_intent_category(self.last_user_utterance)
            intent_norm = float(np.linalg.norm(intent_embedding))
            context_norm = float(np.linalg.norm(dialogue_context))
            cosine_intent_context = _cos(intent_embedding, dialogue_context)
            cosine_intent_prev = _cos(intent_embedding, prev_intent)
            cosine_context_prev = _cos(dialogue_context, prev_context)

            info["dialoguebert_insights"] = {
                "intent_category": intent_category,
                "intent_norm": intent_norm,
                "context_norm": context_norm,
                "cosine_intent_context": cosine_intent_context,
                "cosine_intent_prev": cosine_intent_prev,
                "cosine_context_prev": cosine_context_prev
            }
        except Exception:
            # Insights are optional; ignore failures
            pass


# Alias for backward compatibility and test imports
MuseumEnvironment = MuseumDialogueEnv
