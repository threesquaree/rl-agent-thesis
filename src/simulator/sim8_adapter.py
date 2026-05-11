"""
Sim8 Simulator Adapter for HRL Museum Agent

This adapter implements a production-friendly subset of the sim8_1 notebook logic:
- AOI detection via sentence-transformers (keyword + semantic fallback)
- Persona-aware response types (question, confusion, reference, statement, silence)
- Gaze feature synthesis with dwell time as reward signal
- Clean API: initialize_session, get_current_aoi, generate_user_response, get_current_state

Design goals:
- No Colab, HF login, or large LLM dependencies
- Deterministic-ish behavior with randomness for diversity
- Compatible with existing training loop and environment
"""

import random
import re
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util  # Optional but recommended
    _has_st = True
except Exception:
    _has_st = False
    SentenceTransformer = None
    util = None


class Sim8Simulator:
    """Simulator with AOI detection, persona behavior, and gaze synthesis.
    
    ALL exhibit and AOI mappings are derived from the knowledge graph - no hardcoded values.
    """

    PERSONAS = ["Agreeable", "Conscientious", "Neurotic"]

    # Gaze feature labels - FIRST FEATURE IS DWELL TIME FOR REWARD
    GAZE_LABELS = [
        "DwellTime", "SaccadeSpan", "TurnGazeEntropy",
        "TurnFixChangeRate", "DominantObjectRatio", "GazeEntryLatency"
    ]
    SILENCE_STATS = {
        "Agreeable": {"TurnScanpathLength": (78.809, 132.598), "SaccadeSpan": (0.1030, 0.0556),
                       "TurnGazeEntropy": (0.8501, 0.4581), "TurnFixChangeRate": (2.1581, 0.8471),
                       "DominantObjectRatio": (0.7233, 0.1832), "GazeEntryLatency": (6.4418, 12.5257)},
        "Conscientious": {"TurnScanpathLength": (49.190, 60.532), "SaccadeSpan": (0.1224, 0.0673),
                           "TurnGazeEntropy": (0.5848, 0.5409), "TurnFixChangeRate": (2.0858, 1.4099),
                           "DominantObjectRatio": (0.7938, 0.2126), "GazeEntryLatency": (4.6317, 6.6963)},
        "Neurotic": {"TurnScanpathLength": (41.272, 49.602), "SaccadeSpan": (0.1249, 0.0652),
                      "TurnGazeEntropy": (1.0182, 0.3763), "TurnFixChangeRate": (2.8544, 0.7449),
                      "DominantObjectRatio": (0.6576, 0.2000), "GazeEntryLatency": (2.3105, 3.7783)}
    }

    def __init__(self, knowledge_graph=None, exhibits: Optional[List[str]] = None, seed: int = 42):
        """Initialize simulator with knowledge graph as source of truth.
        
        Args:
            knowledge_graph: SimpleKnowledgeGraph instance (PRIMARY source of truth)
            exhibits: List of exhibit names (fallback if no knowledge graph)
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
        
        # ===== BUILD ALL MAPPINGS FROM KNOWLEDGE GRAPH =====
        if knowledge_graph:
            self._init_from_knowledge_graph(knowledge_graph)
        elif exhibits:
            # Fallback: use provided exhibit list without AOI mappings
            self.exhibits = exhibits
            self.aoi_to_exhibit = {}
            self.exhibit_to_aois = {ex: [] for ex in exhibits}
        else:
            raise ValueError("Must provide either knowledge_graph or exhibits list")
        
        # Sentence transformer model
        self._st_model = None
        self._aoi_list = list(self.aoi_to_exhibit.keys())
        self._aoi_embeddings = None
        if _has_st and self._aoi_list:
            try:
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
                self._aoi_embeddings = self._st_model.encode(self._aoi_list, convert_to_tensor=True)
            except Exception:
                self._st_model = None
                self._aoi_embeddings = None

        # Session state
        self.current_persona: Optional[str] = None
        self.current_exhibit: Optional[str] = None
        self.current_aoi: Optional[str] = None
        self.aoi_usage_count: Dict[str, int] = {}
        self.seen_aois: set = set()
        self.consecutive_silence_count: int = 0
        self.last_user_response: Dict[str, Any] = {}
        
        # Dwell dynamics parameters (for action variety incentive)
        self.base_dwell = 0.8  # Baseline for ExplainNewFact
        self.max_dwell = 1.0   # Maximum with good question variety
        self.min_dwell = 0.4   # Minimum with explain spam
        self.dwell_boost_per_question = 0.1  # Boost from asking question
        self.dwell_decay_rate = 0.02  # Decay per explain/transition without questions
        
        # Track recent action variety
        self.recent_actions = []  # Last N actions (option, subaction)
        self.action_variety_window = 10  # Consider last 10 actions
        
        # Track conversation context for realistic responses
        self.last_user_question: Optional[str] = None  # What user asked
        self.last_user_utterance: Optional[str] = None  # What user last said
        self.last_agent_utterance: Optional[str] = None  # What agent said
        self.last_agent_option: Optional[str] = None  # What option agent used (Explain, AskQuestion, etc.)
        self.conversation_flow: List[str] = []  # Track conversation quality
        
        # Track question spam for engagement reduction (simulator-level only)
        self.consecutive_ask_questions = 0  # Track consecutive AskQuestion actions
        self.repeat_request_probability = 0.2
        
        # Track transition spam for engagement reduction (simulator-level only)
        self.consecutive_transitions = 0  # Track consecutive OfferTransition actions
        self.confusion_active = False
        
        # Store prompts for detailed logging
        self._last_simulator_prompt: Optional[str] = None
        self._last_simulator_system_prompt: Optional[str] = None
        
        # NEW: Museum context for grounding user responses
        self.museum_context = self._build_museum_context()
        
        # NEW: Disengagement tracking
        self.engagement_level = 1.0  # 0.0 = fully disengaged, 1.0 = fully engaged
        self.off_topic_strikes = 0  # Track consecutive off-topic responses
        self._consecutive_recover_count = 0

        # Question pacing reward parameters (make well-spaced questions boost engagement)
        self.turns_since_last_question: int = 999
        self.question_bonus_cooldown: int = 3  # Require at least this many non-question turns
        self.question_bonus_boost: float = 1.6  # Large multiplier when spacing respected
        self.question_penalty_multiplier: float = 0.85  # Slight penalty when spammed

        # Explain spam deterrent when exhibit already complete
        self.consecutive_explains_on_completed = 0
        self.explain_completion_threshold = 0.98  # Treat >=98% as complete to allow rounding noise
        self.explain_spam_penalty_step = 0.2  # Each extra spam turn reduces multiplier by 0.2
        self.explain_spam_min_multiplier = 0.35  # Floor for multiplier impact
        self.explain_spam_recovery_rate = 0.05  # Recover engagement when agent switches strategies
        
        # Dwell stagnation tracking: penalty increases with time spent at same exhibit
        self.turns_at_current_exhibit = 0
        self.previous_exhibit = None
    
    def _init_from_knowledge_graph(self, knowledge_graph):
        """Build ALL mappings from knowledge graph (single source of truth)"""
        # Get all exhibits from knowledge graph
        self.exhibits = knowledge_graph.get_exhibit_names()
        
        # Build AOI → Exhibit mapping from knowledge graph
        self.aoi_to_exhibit: Dict[str, str] = {}
        self.exhibit_to_aois: Dict[str, List[str]] = {}
        
        for exhibit_name in self.exhibits:
            aois = knowledge_graph.get_exhibit_aois(exhibit_name)
            self.exhibit_to_aois[exhibit_name] = aois
            
            # Map each AOI to this exhibit
            for aoi in aois:
                self.aoi_to_exhibit[aoi] = exhibit_name
        
        print(f"[Simulator] Initialized from knowledge graph:")
        print(f"   - {len(self.exhibits)} exhibits: {', '.join(self.exhibits)}")
        print(f"   - {len(self.aoi_to_exhibit)} AOIs mapped to exhibits")
    
    def _build_museum_context(self) -> Dict[str, Any]:
        """Build knowledge of museum structure for context-aware responses"""
        context = {
            "exhibits": self.exhibit_to_aois,
            "aoi_descriptions": {}
        }
        
        # Simple descriptions for AOIs (for future semantic grounding)
        for aoi in self.aoi_to_exhibit.keys():
            context["aoi_descriptions"][aoi] = f"Part of {self.aoi_to_exhibit[aoi]} exhibit"
            
        return context

    # ===== Public API =====
    def reset(self):
        """Reset transient simulator state for a new episode/test.

        Re-initializes counters and session trackers without requiring a
        knowledge graph. Safe to call standalone (e.g. from unit tests).
        """
        self.current_persona = None
        self.current_exhibit = None
        self.current_aoi = None
        self.aoi_usage_count = {}
        self.seen_aois = set()
        self.consecutive_silence_count = 0
        self.last_user_response = {}
        self.consecutive_ask_questions = 0
        self.consecutive_transitions = 0
        self.confusion_active = False
        self.engagement_level = 1.0
        self.off_topic_strikes = 0
        self._consecutive_recover_count = 0
        self.recent_actions = []
        self.turns_at_current_exhibit = 0
        self.previous_exhibit = None
        self.turns_since_last_question = self.question_bonus_cooldown
        self.consecutive_explains_on_completed = 0
        self.last_user_question = None
        self.last_user_utterance = None
        self.last_agent_utterance = None
        self.last_agent_option = None
        self.conversation_flow = []
        self.dialogue_history = []
        self.facts_learned = set()
        self.exhibits_visited = set()
        self.max_history_length = 8
        self.late_phase_questions_asked = 0
        self.current_exhibit_completion_last = 0.0
        self.exhibit_exhausted = False

    def initialize_session(self, persona: Optional[str] = None):
        self.current_persona = persona or self.rng.choice(self.PERSONAS)
        self.current_exhibit = self.rng.choice(self.exhibits)
        self.current_aoi = self._pick_initial_aoi(self.current_exhibit)
        self.aoi_usage_count.clear()
        self.seen_aois.clear()
        self.consecutive_silence_count = 0
        self.last_user_response = {}
        
        # Reset question spam tracking for new session
        self.consecutive_ask_questions = 0
        
        # Reset transition spam tracking for new session
        self.consecutive_transitions = 0
        
        # Initialize dialogue history and learning
        self.dialogue_history = []
        self.facts_learned = set()
        self.exhibits_visited = set()
        self.max_history_length = 8

        # Reset question pacing trackers
        self.turns_since_last_question = self.question_bonus_cooldown
        self.consecutive_explains_on_completed = 0
        self.engagement_level = 1.0
        
        # Reset action variety tracking (for dynamic dwell adjustment)
        self.recent_actions = []
        self._consecutive_recover_count = 0
        
        # Reset exhibit duration tracking
        self.turns_at_current_exhibit = 0
        self.previous_exhibit = self.current_exhibit
        
        # Track late-phase questions for engagement drop
        self.late_phase_questions_asked = 0
        self.current_exhibit_completion_last = 0.0

    def get_current_aoi(self) -> str:
        """Return current exhibit (for env focus) to match previous simulator contract."""
        return self.current_exhibit or self.exhibits[0]

    def generate_user_response(self, agent_utterance: str, agent_option: str = None,
                             agent_subaction: str = None, target_exhibit: str = None,
                             current_exhibit_completion: float = 0.0,
                             exhibit_exhausted: bool = False,
                             target_exhibit_completion: float = 0.0,
                             target_exhibit_exhausted: bool = False) -> Dict[str, Any]:
        """Generate a response dict with utterance, aoi, persona, gaze_features, response_type.
        
        Args:
            agent_utterance: The agent's dialogue response
            agent_option: The agent's chosen option (Explain, AskQuestion, OfferTransition, Conclude)
            target_exhibit: For OfferTransition, the exhibit the agent wants to transition to
            current_exhibit_completion: Completion rate (0.0-1.0) of current exhibit
            exhibit_exhausted: Whether the current exhibit has no new facts available
        """
        # Store agent's utterance and option for context tracking
        self.last_agent_utterance = agent_utterance
        self.last_agent_option = agent_option
        self.exhibit_exhausted = exhibit_exhausted  # Store for dwell time calculation

        # Update spacing tracker for AskQuestion option (counts non-question turns)
        self.turns_since_last_question = min(self.turns_since_last_question + 1, 1000)
        
        # VERY RARE silence (1% only, prevents Turn 2 silence issue)
        # Only in first turn OR if too many consecutive silences
        if self.consecutive_silence_count >= 2:
            # Force non-silence to prevent dialogue breakdown
            pass
        elif len(self.dialogue_history) < 2 and self.rng.random() < 0.01:
            # Only in very first turn, 1% chance
            self.consecutive_silence_count += 1
            response = self._make_silence_response()
            self.last_user_response = response
            return response
        else:
            # Never silence after first turn
            pass

        # Reset silence streak on any non-silence response
        self.consecutive_silence_count = 0

        # STEP 0: Check agent relevance to current context
        relevance_score = self._check_agent_relevance(agent_utterance)
        
        # Track agent utterance in dialogue history
        self.dialogue_history.append({"role": "agent", "text": agent_utterance})
        if len(self.dialogue_history) > self.max_history_length:
            self.dialogue_history.pop(0)
        
        # NEW TRANSITION LOGIC: Probability-based transition success
        transition_success = False
        detected_aoi = self.current_aoi  # Default: stay at current AOI
        
        if agent_option == "OfferTransition" and target_exhibit is not None:
            # Calculate transition success probability based on current exhibit completion
            # Scale: 0 facts = 20% success, 1 fact = 50%, 2 facts = 80%, 3+ facts = 95%
            if current_exhibit_completion == 0.0:
                transition_prob = 0.20
            elif current_exhibit_completion < 0.33:  # 1/3 facts (e.g., 1 of 3 or 1-2 of 5)
                transition_prob = 0.50
            elif current_exhibit_completion < 0.67:  # 2/3 facts (e.g., 2 of 3 or 3-4 of 5)
                transition_prob = 0.80
            else:  # 3+ facts or high completion
                transition_prob = 0.95
            
            # PENALTY: Reduce success probability if target exhibit is already visited/exhausted
            # Visitor is less interested in already-explored exhibits
            target_quality_multiplier = 1.0
            if target_exhibit_exhausted:
                # Target is completely exhausted - major penalty
                target_quality_multiplier = 0.15  # 85% reduction in success probability
                import os
                verbose = os.environ.get('HRL_VERBOSE', '0') == '1'
                if verbose:
                    print(f"⚠️ POOR TARGET: {target_exhibit} is EXHAUSTED, reducing transition prob by 85%")
            elif target_exhibit_completion >= 0.67:
                # Target is mostly complete (67%+) - moderate penalty
                target_quality_multiplier = 0.50  # 50% reduction in success probability
                import os
                verbose = os.environ.get('HRL_VERBOSE', '0') == '1'
                if verbose:
                    print(f"⚠️ POOR TARGET: {target_exhibit} is {target_exhibit_completion:.0%} complete, reducing transition prob by 50%")
            elif target_exhibit_completion >= 0.33:
                # Target is partially visited (33-67%) - mild penalty
                target_quality_multiplier = 0.75  # 25% reduction in success probability
                import os
                verbose = os.environ.get('HRL_VERBOSE', '0') == '1'
                if verbose:
                    print(f"⚠️ POOR TARGET: {target_exhibit} is {target_exhibit_completion:.0%} complete, reducing transition prob by 25%")
            
            # Apply target quality penalty to transition probability
            transition_prob *= target_quality_multiplier
            
            # Roll for transition success
            if self.rng.random() < transition_prob:
                # TRANSITION SUCCEEDS: Move to target exhibit
                transition_success = True
                
                # Validate target exhibit exists AND has AOIs
                if target_exhibit in self.exhibits and target_exhibit in self.exhibit_to_aois and self.exhibit_to_aois[target_exhibit]:
                    # Select a random AOI from the target exhibit
                    detected_aoi = self.rng.choice(self.exhibit_to_aois[target_exhibit])
                    self.current_exhibit = target_exhibit  # Only update after successful AOI selection
                    import os
                    verbose = os.environ.get('HRL_VERBOSE', '0') == '1'
                    if verbose:
                        print(f"🔄 TRANSITION SUCCESS! → {target_exhibit} (prob={transition_prob:.1%}, curr={current_exhibit_completion:.1%}, target={target_exhibit_completion:.1%})")
                elif target_exhibit not in self.exhibits:
                    # Invalid target exhibit
                    transition_success = False
                    print(f"⚠️  TRANSITION FAILED: Invalid target exhibit {target_exhibit}")
                else:
                    # No AOIs found for this exhibit
                    transition_success = False
                    print(f"⚠️  TRANSITION FAILED: No AOIs found for {target_exhibit}")
            else:
                # TRANSITION FAILS: Stay at current exhibit
                transition_success = False
                import os
                verbose = os.environ.get('HRL_VERBOSE', '0') == '1'
                if verbose:
                    print(f"❌ TRANSITION REJECTED! Visitor wants to stay at {self.current_exhibit} (prob={transition_prob:.1%}, curr={current_exhibit_completion:.1%}, target={target_exhibit_completion:.1%})")
        else:
            # Not a transition action, stay at current AOI
            # Allow occasional random exploration to keep engagement diverse
            if self.rng.random() < 0.15:  # 15% chance of natural gaze wander
                detected_aoi = self._switch_to_related_aoi(detected_aoi)
        
        # Update session mapping
        self.current_aoi = detected_aoi
        previous_exhibit = self.current_exhibit
        if detected_aoi in self.aoi_to_exhibit:
            self.current_exhibit = self.aoi_to_exhibit[detected_aoi]
        
        # Track turns at current exhibit (for dwell stagnation penalty)
        exhibit_changed = (self.current_exhibit != previous_exhibit and self.current_exhibit is not None and previous_exhibit is not None)
        if exhibit_changed:
            # Exhibit changed - reset counter and update exhausted state
            self.turns_at_current_exhibit = 1
            self.previous_exhibit = self.current_exhibit
            # CRITICAL FIX: Reset exhibit_exhausted flag when transitioning to new exhibit
            # The new exhibit's exhaustion state will be updated on next turn via environment
            # For now, reset it so dwell stagnation doesn't carry over from previous exhibit
            if transition_success:
                # Successful transition - reset exhausted flag (will be updated by env next turn)
                self.exhibit_exhausted = False
        elif self.current_exhibit == previous_exhibit and self.current_exhibit is not None:
            # Same exhibit - increment counter
            self.turns_at_current_exhibit += 1
        else:
            # Initial state or no exhibit yet
            self.turns_at_current_exhibit = 1
            self.previous_exhibit = self.current_exhibit
        
        self.aoi_usage_count[detected_aoi] = self.aoi_usage_count.get(detected_aoi, 0) + 1
        self.seen_aois.add(detected_aoi)

        # STEP 2: Choose response type based on context and agent quality
        # For failed transitions, FORCE confusion response
        if agent_option == "OfferTransition" and not transition_success:
            rtype = "confusion"
        else:
            rtype = self._determine_response_type_contextual(
                agent_utterance, agent_option,
                current_exhibit_completion=current_exhibit_completion,
                turns_at_current_exhibit=self.turns_at_current_exhibit
            )
        
        if rtype == "confusion":
            self.confusion_active = True
        
        repeat_request_triggered = False
        clarify_success = False
        clarify_misfire = False
        if agent_option == "Explain" and rtype != "confusion":
            if self.rng.random() < self.repeat_request_probability:
                rtype = "repeat_request"
                repeat_request_triggered = True
        if agent_option == "Explain" and agent_subaction == "ClarifyFact":
            if self.confusion_active:
                clarify_success = True
                self.confusion_active = False
                rtype = "acknowledgment"
            else:
                clarify_misfire = True
        
        # STEP 3: Generate utterance with LLM-guided response (NEW!)
        import time
        sim_start = time.time()
        # Pass transition context to utterance generation
        utterance = self._synthesize_llm_guided_utterance(
            agent_utterance, rtype, detected_aoi,
            transition_rejected=(agent_option == "OfferTransition" and not transition_success),
            transition_success=(agent_option == "OfferTransition" and transition_success),
            target_exhibit=target_exhibit,
            current_exhibit_completion=current_exhibit_completion,
            turns_at_current_exhibit=self.turns_at_current_exhibit,
            agent_option=agent_option,
            agent_subaction=agent_subaction
        )
        self._last_sim_llm_time = time.time() - sim_start
        
        # STEP 4: Track question spam and transition spam (reduces engagement/dwell)
        import os
        verbose = os.environ.get('HRL_VERBOSE', '0') == '1'

        engagement_adjust_multiplier = 1.0
        if agent_option == "AskQuestion":
            self.consecutive_ask_questions += 1
            spacing_ok = self.turns_since_last_question >= self.question_bonus_cooldown
            if spacing_ok:
                question_multiplier = self.question_bonus_boost
                if verbose:
                    print(f"✨ QUESTION BOOST: Engagement multiplier {question_multiplier:.2f} (spacing {self.turns_since_last_question} turns)", flush=True)
            else:
                question_multiplier = self.question_penalty_multiplier
                if verbose:
                    print(f"⚠️ QUESTION TOO SOON: Applying multiplier {question_multiplier:.2f} (spacing {self.turns_since_last_question} turns)", flush=True)
            engagement_adjust_multiplier *= question_multiplier
            self.turns_since_last_question = 0
        else:
            self.consecutive_ask_questions = 0
        
        explain_spam_multiplier = 1.0
        if agent_option == "Explain":
            # Check if agent is sharing new facts (detect fact IDs in brackets)
            fact_ids_in_utterance = re.findall(r'\[([^\]]+)\]', agent_utterance or "")
            has_new_facts = len(fact_ids_in_utterance) > 0
            
            if current_exhibit_completion >= self.explain_completion_threshold:
                # Only penalize if NO new facts are being shared (spam = repeating old info)
                if not has_new_facts:
                    self.consecutive_explains_on_completed += 1
                    penalty = 1.0 - (self.explain_spam_penalty_step * self.consecutive_explains_on_completed)
                    explain_spam_multiplier = max(self.explain_spam_min_multiplier, penalty)
                    if verbose:
                        completion_pct = current_exhibit_completion * 100.0
                        print(
                            f"⚠️ EXPLAIN SPAM: Completion {completion_pct:.1f}% → multiplier {explain_spam_multiplier:.2f} "
                            f"(streak {self.consecutive_explains_on_completed}, no new facts)",
                            flush=True
                        )
                else:
                    # Agent is sharing new facts even at high completion - reset spam counter
                    self.consecutive_explains_on_completed = 0
                    if verbose:
                        print(f"✨ NEW FACTS DETECTED: Resetting explain spam counter (completion={current_exhibit_completion:.1%})", flush=True)
            else:
                self.consecutive_explains_on_completed = 0
        else:
            self.consecutive_explains_on_completed = 0

        engagement_adjust_multiplier *= explain_spam_multiplier

        # NOVELTY BOOST: Boost engagement when agent shares new facts (per paper.tex lines 828-830)
        # Novel facts boost dwell by +0.05 to +0.15, scaled by visitor curiosity
        if agent_utterance:
            # Detect fact IDs in brackets like [FACT_ID]
            fact_ids = re.findall(r'\[([^\]]+)\]', agent_utterance)
            num_new_facts = len(fact_ids)
            
            if num_new_facts > 0:
                # Boost engagement when sharing new facts (especially if exhibit not exhausted)
                # Scale boost: 1 fact = +0.05, 2 facts = +0.10, 3+ facts = +0.15
                if not exhibit_exhausted:
                    # Exhibit still has new facts - stronger boost (visitor is learning)
                    novelty_boost = 1.0 + (0.05 * min(num_new_facts, 3))  # Max +0.15 boost
                else:
                    # Exhibit exhausted but agent still sharing facts (might be repeats) - smaller boost
                    novelty_boost = 1.0 + (0.03 * min(num_new_facts, 2))  # Max +0.06 boost
                
                engagement_adjust_multiplier *= novelty_boost
                if verbose:
                    print(f"📚 NOVELTY BOOST: +{((novelty_boost - 1.0) * 100):.1f}% engagement ({num_new_facts} fact(s), exhausted={exhibit_exhausted})", flush=True)

        if agent_option == "OfferTransition":
            if transition_success:
                # Successful transition - reset spam counter
                self.consecutive_transitions = 0
                
                # RECOVERY BONUS: Boost engagement when transitioning to fresh exhibits
                # This helps dwell ratio recover after staying at exhausted exhibit
                recovery_bonus = 1.0
                
                # Check if target exhibit is poor quality (exhausted or mostly complete)
                # Apply additional penalty even on successful transitions to poor targets
                if target_exhibit_exhausted:
                    # Transitioned to exhausted exhibit - strong penalty, no recovery
                    engagement_adjust_multiplier *= 0.60
                    if verbose:
                        print(f"⚠️ BAD TARGET (EXHAUSTED): multiplier 0.60, target={target_exhibit} ({target_exhibit_completion:.0%})", flush=True)
                elif target_exhibit_completion >= 0.67:
                    # Transitioned to mostly-complete exhibit - moderate penalty, slight recovery
                    engagement_adjust_multiplier *= 0.75
                    recovery_bonus = 1.05  # Small recovery bonus
                    if verbose:
                        print(f"⚠️ BAD TARGET (HIGH COMPLETION): multiplier 0.75, target={target_exhibit} ({target_exhibit_completion:.0%})", flush=True)
                elif target_exhibit_completion >= 0.33:
                    # Transitioned to partially-visited exhibit - mild penalty, moderate recovery
                    engagement_adjust_multiplier *= 0.85
                    recovery_bonus = 1.10  # Moderate recovery bonus
                    if verbose:
                        print(f"⚠️ BAD TARGET (PARTIAL COMPLETION): multiplier 0.85, target={target_exhibit} ({target_exhibit_completion:.0%})", flush=True)
                elif current_exhibit_completion < self.explain_completion_threshold:
                    # Good target but premature departure from current - mild penalty, recovery
                    engagement_adjust_multiplier *= 0.85
                    recovery_bonus = 1.10  # Moderate recovery bonus
                    if verbose:
                        print(f"⚠️ TRANSITION DISENGAGEMENT: multiplier 0.85 (premature, coverage={current_exhibit_completion:.2f})", flush=True)
                else:
                    # EXCELLENT: Smooth handoff to fresh/empty exhibit - strong recovery bonus
                    # This helps engagement recover after being at exhausted exhibit
                    recovery_bonus = 1.20  # Strong recovery bonus for fresh exhibits
                    if verbose:
                        print(f"✨ TRANSITION RECOVERY: multiplier 1.20 (fresh target={target_exhibit}, coverage={target_exhibit_completion:.0%})", flush=True)
                
                # Apply recovery bonus (helps dwell ratio recover after exhaustion)
                engagement_adjust_multiplier *= recovery_bonus
            else:
                # Failed transition - increment spam counter and apply penalty
                self.consecutive_transitions += 1
                engagement_adjust_multiplier *= 0.85
                if verbose:
                    print(f"⚠️ TRANSITION REJECTED: multiplier 0.85, spam count={self.consecutive_transitions} (coverage={current_exhibit_completion:.2f})", flush=True)
        else:
            self.consecutive_transitions = 0
        
        # Track late-phase questions and apply engagement drop after 2-3 questions
        if current_exhibit_completion >= 0.80:
            # Reset counter if we just entered late phase
            if self.current_exhibit_completion_last < 0.80:
                self.late_phase_questions_asked = 0
            
            # Track questions asked by visitor in late phase
            if rtype in ["question", "follow_up_question"]:
                self.late_phase_questions_asked += 1
                
                # After 2-3 questions in late phase, reduce engagement
                if self.late_phase_questions_asked >= 2:
                    engagement_drop = 0.75 - (0.05 * (self.late_phase_questions_asked - 2))
                    engagement_drop = max(0.60, engagement_drop)  # Floor at 0.60
                    engagement_adjust_multiplier *= engagement_drop
                    if verbose:
                        print(f"📉 LATE-PHASE QUESTIONS: {self.late_phase_questions_asked} questions asked, multiplier {engagement_drop:.2f}", flush=True)
        else:
            # Not in late phase, reset counter
            self.late_phase_questions_asked = 0
        
        # Store current completion for next turn
        self.current_exhibit_completion_last = current_exhibit_completion
        
        if agent_option == "Explain" and agent_subaction == "ClarifyFact":
            if clarify_success:
                engagement_adjust_multiplier *= 1.15
                if verbose:
                    print("✨ CLARIFICATION SUCCESS: engagement boost", flush=True)
            elif clarify_misfire:
                engagement_adjust_multiplier *= 0.8
                if verbose:
                    print("⚠️ UNNEEDED CLARIFICATION: engagement drop", flush=True)
        
        self.engagement_level = max(
            0.2,
            min(1.0, self.engagement_level + (engagement_adjust_multiplier - 1.0) * 0.2)
        )
        
        # STEP 5: Generate gaze features based on response type (question spam and transition spam reduce dwell)
        gaze_features = self._synthesize_contextual_gaze(rtype, agent_option, agent_subaction, engagement_adjust_multiplier=engagement_adjust_multiplier)
        
        # Store user's question if they asked one
        if rtype == "question" or rtype == "repeat_request":
            self.last_user_question = utterance
        
        # Store last user utterance for context tracking
        self.last_user_utterance = utterance

        response = {
            "utterance": utterance,
            "aoi": detected_aoi,
            "persona": self.current_persona,
            "gaze_features": gaze_features,
            "response_type": rtype,
            "engagement_level": self.engagement_level,  # Track disengagement
            "off_topic_strikes": self.off_topic_strikes,  # Track penalties
            "agent_option": self.last_agent_option,  # Track agent's strategy
            "transition_success": transition_success,  # Whether transition succeeded (if OfferTransition)
            "simulator_llm_time": getattr(self, '_last_sim_llm_time', 0.0),
            "repeat_request": repeat_request_triggered
        }
        self.last_user_response = response
        
        # Track in memory (NEW!)
        self.exhibits_visited.add(self.current_exhibit)
        self.dialogue_history.append({"role": "user", "text": utterance or ""})
        
        # Extract fact IDs if present
        fact_ids = re.findall(r'\[([A-Z]{2}_\d{3})\]', agent_utterance)
        for fact_id in fact_ids:
            self.facts_learned.add(fact_id)
        
        return response
    
    def _adjust_dwell_for_action_variety(self, option, subaction, base_dwell):
        """
        Dynamically adjust dwell based on action variety:
        - Questions boost dwell (up to max_dwell=1.0)
        - Explain/Transition spam reduces dwell (down to min_dwell=0.4)
        - Baseline ExplainNewFact maintains base_dwell=0.8
        """
        # Track recent actions
        self.recent_actions.append((option, subaction))
        if len(self.recent_actions) > self.action_variety_window:
            self.recent_actions.pop(0)
        
        # Count action types in recent window
        question_count = sum(1 for opt, _ in self.recent_actions if opt == "AskQuestion")
        explain_new_count = sum(1 for opt, sub in self.recent_actions 
                               if opt == "Explain" and sub == "ExplainNewFact")
        transition_count = sum(1 for opt, _ in self.recent_actions if opt == "OfferTransition")
        
        # Calculate target dwell based on variety
        if question_count >= 2:
            # Good variety - boost toward max (questions maintain engagement)
            target_dwell = self.max_dwell
        elif question_count == 1:
            # Some variety - slight boost above base
            target_dwell = self.base_dwell + (self.dwell_boost_per_question * 0.5)
        elif explain_new_count >= 7:
            # Explain spam - decay toward min (information overload)
            decay_amount = self.dwell_decay_rate * (explain_new_count - 5)
            target_dwell = self.base_dwell - decay_amount
        elif (explain_new_count + transition_count) >= 8:
            # Mixed spam (explain + transition without questions) - moderate decay
            target_dwell = self.base_dwell - (self.dwell_decay_rate * 3)
        else:
            # Normal variety - maintain base
            target_dwell = self.base_dwell
        
        # Clamp to valid range
        target_dwell = np.clip(target_dwell, self.min_dwell, self.max_dwell)
        
        # Smooth transition (don't jump immediately - 30% adjustment per turn)
        adjusted_dwell = base_dwell + 0.3 * (target_dwell - base_dwell)
        
        # Clamp final result
        adjusted_dwell = np.clip(adjusted_dwell, self.min_dwell, self.max_dwell)
        
        return adjusted_dwell

    def get_current_state(self) -> Dict[str, Any]:
        return {
            "aoi": self.current_aoi,
            "current_exhibit": self.current_exhibit,
            "persona": self.current_persona,
            "seen_aois": list(self.seen_aois),
            "aoi_usage_count": dict(self.aoi_usage_count),
            "consecutive_silence_count": self.consecutive_silence_count,
            "last_user_response": dict(self.last_user_response) if self.last_user_response else {}
        }

    def update_from_state(self, state_focus: int, target_exhibit: str = None):
        """Update simulator state based on environment state information"""
        # If we have a target exhibit from transition logic, prioritize it
        if target_exhibit and target_exhibit != self.current_exhibit:
            # Find an AOI for the target exhibit
            for parent_code, parent_exhibit in self.PARENT_TO_EXHIBIT.items():
                if parent_exhibit == target_exhibit:
                    candidate_aois = [a for a, p in self.AOI_TO_PARENT.items() if p == parent_code]
                    if candidate_aois:
                        new_aoi = self.rng.choice(candidate_aois)
                        self.current_aoi = new_aoi
                        self.current_exhibit = target_exhibit
                        self.aoi_usage_count[new_aoi] = self.aoi_usage_count.get(new_aoi, 0) + 1
                        self.seen_aois.add(new_aoi)
                        print(f"STATE INFLUENCE: Transitioned to {target_exhibit} (AOI: {new_aoi})")
                        return

        # Otherwise, use focus from state if it's different from current
        if state_focus > 0 and state_focus <= len(self.exhibits):
            target_exhibit = self.exhibits[state_focus - 1]
            if target_exhibit != self.current_exhibit:
                # Find an AOI for the target exhibit
                for parent_code, parent_exhibit in self.PARENT_TO_EXHIBIT.items():
                    if parent_exhibit == target_exhibit:
                        candidate_aois = [a for a, p in self.AOI_TO_PARENT.items() if p == parent_code]
                        if candidate_aois:
                            new_aoi = self.rng.choice(candidate_aois)
                            self.current_aoi = new_aoi
                            self.current_exhibit = target_exhibit
                            self.aoi_usage_count[new_aoi] = self.aoi_usage_count.get(new_aoi, 0) + 1
                            self.seen_aois.add(new_aoi)
                            print(f"STATE SYNC: Updated to {target_exhibit} (AOI: {new_aoi})")

    # ===== Internals =====
    def _pick_initial_aoi(self, exhibit: str) -> str:
        """Pick a random AOI from the given exhibit"""
        if exhibit in self.exhibit_to_aois and self.exhibit_to_aois[exhibit]:
            return self.rng.choice(self.exhibit_to_aois[exhibit])
        # Fallback: pick any AOI from any exhibit
        if self.aoi_to_exhibit:
            return self.rng.choice(list(self.aoi_to_exhibit.keys()))
        return "Unknown"

    def _detect_aoi_and_parent(self, text: str, sim_threshold: float = 0.35) -> Tuple[Optional[str], Optional[str]]:
        t = (text or "").lower()

        # First, try exhibit names (for cross-exhibit transitions) - improved matching
        # Check both PARENT_TO_EXHIBIT and AOI_TO_EXHIBIT mappings
        all_exhibit_names = set(self.PARENT_TO_EXHIBIT.values()) | set(self.AOI_TO_EXHIBIT.values())

        for exhibit_name in all_exhibit_names:
            # Remove underscores and check for exhibit name in text (more flexible matching)
            exhibit_clean = exhibit_name.replace("_", " ").lower()

            # Direct match
            if exhibit_clean in t:
                # Find an AOI for this exhibit
                if exhibit_name in self.AOI_TO_EXHIBIT:
                    # It's a direct AOI-to-exhibit mapping
                    aoi = exhibit_name
                    parent_code = self.AOI_TO_PARENT.get(aoi, "C6")  # Default to C6 if not found
                    return aoi, parent_code
                else:
                    # It's a parent exhibit, find an AOI within it
                    for parent_code, parent_exhibit in self.PARENT_TO_EXHIBIT.items():
                        if parent_exhibit == exhibit_name:
                            # Get random AOI from this exhibit
                            candidate_aois = [a for a, p in self.AOI_TO_PARENT.items() if p == parent_code]
                            if candidate_aois:
                                aoi = self.rng.choice(candidate_aois)
                                return aoi, parent_code

            # Partial word matching for exhibit names (e.g., "caspar" matches "King Caspar")
            exhibit_words = exhibit_clean.split()
            for word in exhibit_words:
                if len(word) > 3 and word in t:  # Only match longer words to avoid false positives
                    if exhibit_name in self.AOI_TO_EXHIBIT:
                        # It's a direct AOI-to-exhibit mapping
                        aoi = exhibit_name
                        parent_code = self.AOI_TO_PARENT.get(aoi, "C6")
                        return aoi, parent_code
                    else:
                        # It's a parent exhibit
                        for parent_code, parent_exhibit in self.PARENT_TO_EXHIBIT.items():
                            if parent_exhibit == exhibit_name:
                                candidate_aois = [a for a, p in self.AOI_TO_PARENT.items() if p == parent_code]
                                if candidate_aois:
                                    aoi = self.rng.choice(candidate_aois)
                                    return aoi, parent_code

        # Keyword match for AOIs (handle possessives like "caspar's")
        for aoi in self._aoi_list:
            parts = aoi.lower().split()
            for part in parts:
                # Match with optional possessive 's
                if re.search(rf"\b{re.escape(part)}('s)?\b", t):
                    return aoi, self.AOI_TO_PARENT[aoi]

        # Enhanced semantic fallback for transition contexts
        if self._st_model is not None and self._aoi_embeddings is not None:
            try:
                query_emb = self._st_model.encode(text, convert_to_tensor=True)
                from torch import max as tmax  # local import to avoid hard dep if torch missing
                cos_scores = util.cos_sim(query_emb, self._aoi_embeddings)[0]
                top_score, top_idx = tmax(cos_scores, dim=0)

                # Lower threshold for transition contexts (agent suggesting movement)
                transition_keywords = ["visit", "see", "check out", "explore", "move", "next", "let's", "shall we"]
                is_transition_context = any(keyword in t for keyword in transition_keywords)
                effective_threshold = sim_threshold * 0.7 if is_transition_context else sim_threshold

                if float(top_score) >= effective_threshold:
                    aoi = self._aoi_list[int(top_idx)]
                    return aoi, self.AOI_TO_PARENT[aoi]
            except Exception:
                pass
        return None, None

    def _switch_to_related_aoi(self, current_aoi: str) -> str:
        """Switch to a related AOI (70% same exhibit, 30% different exhibit)"""
        current_exhibit = self.aoi_to_exhibit.get(current_aoi, self.current_exhibit)
        
        if self.rng.random() < 0.7 and current_exhibit:
            # Switch within same exhibit (sibling AOIs)
            siblings = [aoi for aoi in self.exhibit_to_aois.get(current_exhibit, []) if aoi != current_aoi]
            if siblings:
                return self.rng.choice(siblings)
        
        # Switch to different exhibit - ONLY use exhibits from self.exhibits list
        other_exhibits = [ex for ex in self.exhibits if ex != current_exhibit]
        if other_exhibits:
            new_exhibit = self.rng.choice(other_exhibits)
            new_aois = self.exhibit_to_aois.get(new_exhibit, [])
            if new_aois:
                self.current_exhibit = new_exhibit
                return self.rng.choice(new_aois)
        
        # Fallback: stay with current
        return current_aoi

    def _is_suggesting_movement(self, text: str) -> bool:
        """Detect if agent is suggesting moving to another exhibit"""
        text_lower = (text or "").lower()
        move_keywords = [
            "check out", "explore", "visit", "see", "move", "head to",
            "would you like to", "shall we", "let's", "next", "another",
            "recommend", "suggest", "ready to", "finished", "done"
        ]
        return any(keyword in text_lower for keyword in move_keywords)
    
    def _detect_hallucinations(self, agent_utterance: str) -> tuple:
        """Detect if agent claims conversations that didn't happen."""
        agent_lower = (agent_utterance or "").lower()
        
        # Detect memory-seeking language
        memory_patterns = ["recall", "remember", "discussed", "we talked about", "earlier", "before"]
        has_memory_claim = any(pattern in agent_lower for pattern in memory_patterns)
        
        if not has_memory_claim:
            return False, ""
        
        # If claiming past discussion but dialogue history is empty = hallucination!
        if not self.dialogue_history:
            return True, "first_turn_memory_claim"
        
        # Extract claimed topic from utterance
        import re
        topic_patterns = [
            r'(?:about|regarding|on)\s+([a-z\s]+?)(?:\?|\.)',
            r'our\s+(?:discussion|conversation)\s+([a-z\s]+?)(?:\?|\.)',
        ]
        
        claimed_topic = None
        for pattern in topic_patterns:
            match = re.search(pattern, agent_lower)
            if match:
                claimed_topic = match.group(1).strip()
                break
        
        if claimed_topic:
            # Search dialogue history for this topic
            history_text = " ".join([turn.get("text", "").lower() for turn in self.dialogue_history])
            if claimed_topic not in history_text:
                return True, f"topic_not_discussed_{claimed_topic}"
        
        return False, ""

    def _check_agent_relevance(self, agent_utterance: str) -> float:
        """
        Check if agent's response is relevant to current context.
        Returns relevance score 0.0-1.0 and updates engagement metric.
        
        Checks:
        - Is agent talking about current AOI/exhibit?
        - Is agent using meta-commentary (bad)?
        - Is agent being natural vs robotic?
        """
        relevance_score = 1.0
        agent_lower = (agent_utterance or "").lower()
        
        # CHECK 0: Hallucination detection (NEW!)
        is_hallucinating, hallucination_reason = self._detect_hallucinations(agent_utterance)
        if is_hallucinating:
            relevance_score -= 0.5  # MAJOR PENALTY
            self.off_topic_strikes += 2  # Double strike
            self.engagement_level *= 0.4  # Significant drop
        
        # Check current AOI mention
        current_aoi_clean = self.current_aoi.lower().replace("_", " ") if self.current_aoi else ""
        current_exhibit_clean = self.current_exhibit.lower().replace("_", " ") if self.current_exhibit else ""
        
        aoi_mentioned = current_aoi_clean in agent_lower
        exhibit_mentioned = current_exhibit_clean in agent_lower
        
        if not (aoi_mentioned or exhibit_mentioned):
            relevance_score -= 0.3  # Penalty: not mentioning current focus
        
        # Check for meta-commentary (BAD: "here's a response", "visitor:", "guide:")
        meta_patterns = ["here's a response", "here is a response", "the guide says", 
                        "visitor:", "guide:", "museum guide:", "assistant:"]
        meta_count = sum(1 for pattern in meta_patterns if pattern in agent_lower)
        if meta_count > 0:
            relevance_score -= 0.2 * meta_count
        
        # Check for natural language markers (GOOD)
        natural_markers = ["actually", "honestly", "you know", "interesting", "fascinating"]
        natural_count = sum(1 for marker in natural_markers if marker in agent_lower)
        relevance_score += 0.1 * min(natural_count, 2)
        
        # Clamp score
        relevance_score = max(0.0, min(1.0, relevance_score))
        
        # Update disengagement metric
        if relevance_score < 0.5:
            self.off_topic_strikes += 1
            if self.off_topic_strikes >= 2:
                self.engagement_level *= 0.6  # Drop engagement significantly
        else:
            self.off_topic_strikes = 0
            self.engagement_level = min(1.0, self.engagement_level + 0.05)
        
        return relevance_score
    
    def _check_exhibit_mismatch(self, agent_utterance: str) -> bool:
        """Check if agent is talking about a different exhibit than visitor is at.
        
        Returns True if there's a mismatch (should trigger confusion).
        """
        if not agent_utterance or not self.current_exhibit:
            return False
        
        agent_lower = agent_utterance.lower()
        
        # Get all exhibit names from knowledge graph
        all_exhibits = self.exhibits
        
        # Check which exhibits are mentioned in agent's utterance
        mentioned_exhibits = []
        for exhibit in all_exhibits:
            exhibit_clean = exhibit.lower().replace("_", " ")
            if exhibit_clean in agent_lower:
                mentioned_exhibits.append(exhibit)
        
        # If agent mentioned a different exhibit than where visitor is
        if mentioned_exhibits and self.current_exhibit not in mentioned_exhibits:
            import os
            verbose = os.environ.get('HRL_VERBOSE', '0') == '1'
            if verbose:
                print(f"[SIM] Exhibit mismatch: Agent talks about {mentioned_exhibits}, visitor at {self.current_exhibit}")
            return True
        
        return False
    
    def _determine_response_type_contextual(self, agent_utterance: str, agent_option: str = None,
                                           current_exhibit_completion: float = 0.0,
                                           turns_at_current_exhibit: int = 0) -> str:
        """Determine response type based on agent's utterance and option choice.
        
        Response types directly signal quality:
        - acknowledgment, follow_up_question = positive engagement
        - confusion = negative (agent off-topic or deflecting)
        - question, statement = neutral engagement
        
        Args:
            agent_utterance: The agent's dialogue response
            agent_option: The agent's chosen option (Explain, AskQuestion, etc.)
            current_exhibit_completion: Completion rate (0.0-1.0) of current exhibit
            turns_at_current_exhibit: Number of turns spent at current exhibit
        """
        text = (agent_utterance or "").lower()
        
        # 1. Check for exhibit mismatch (agent talks about wrong exhibit)
        exhibit_mismatch = self._check_exhibit_mismatch(agent_utterance)
        if exhibit_mismatch:
            # High chance of confusion when exhibits don't match
            if self.rng.random() < 0.7:
                return "confusion"
        
        # 2. Check if agent deflected with questions when user requested info
        if agent_option == "AskQuestion" and self.last_user_utterance:
            info_request_phrases = [
                "tell me", "could you tell", "can you tell", "share more", 
                "i'd love to learn", "explain", "what about", "how does",
                "could you share", "what is", "what are"
            ]
            user_requested_info = any(phrase in self.last_user_utterance.lower() 
                                     for phrase in info_request_phrases)
            
            # Penalize if user explicitly asked for info and agent deflected (after turn 2)
            if user_requested_info and len(self.dialogue_history) > 2:
                import os
                verbose = os.environ.get('HRL_VERBOSE', '0') == '1'
                if verbose:
                    print(f"[SIM] User requested info, agent used AskQuestion → confusion")
                if self.rng.random() < 0.6:
                    return "confusion"
        
        # 3. If user asked a question, check if agent provided substantive response
        if self.last_user_question:
            # Check if agent provided information (has [FACT_ID] or info keywords)
            has_fact_id = "[" in text and "]" in text
            info_keywords = ["is", "was", "are", "made", "created", "from", "century", "period"]
            has_info = sum(1 for kw in info_keywords if kw in text.split()) >= 2
            
            if has_fact_id or has_info:
                # Agent provided good answer → positive response
                if self.rng.random() < 0.4:
                    return "acknowledgment"
                elif self.rng.random() < 0.6:
                    return "follow_up_question"
                else:
                    return "statement"
            else:
                # Agent didn't really answer → confusion or re-ask
                if self.rng.random() < 0.5:
                    return "confusion"
                else:
                    return "question"
        
        # 4. Agent asked a question - respond appropriately
        if "?" in text:
            if self.rng.random() < 0.7:
                return "question"
            else:
                return "statement"
        
        # 5. Neurotic persona gets confused sometimes
        if self.current_persona == "Neurotic" and self.rng.random() < 0.15:
            return "confusion"
        
        # 6. Frustration when overstaying at exhausted exhibit
        if turns_at_current_exhibit >= 6 and current_exhibit_completion >= 0.80:
            if self.rng.random() < 0.40:
                return "confusion"  # Frustration signal
        
        # 7. Progressive questioning based on exhibit completion
        # Early phase: new exhibit, visitor just acknowledges/listens
        if turns_at_current_exhibit <= 2 or current_exhibit_completion < 0.30:
            rand = self.rng.random()
            if rand < 0.75:
                return "acknowledgment"
            elif rand < 0.90:
                return "statement"
            else:
                return "question"
        
        # Middle phase: visitor processing information, moderate questions
        elif current_exhibit_completion < 0.80:
            rand = self.rng.random()
            if rand < 0.35:
                return "question"
            elif rand < 0.70:
                return "acknowledgment"
            elif rand < 0.90:
                return "statement"
            else:
                return "follow_up_question"
        
        # Late phase: high completion, visitor asks clarifying questions
        else:
            rand = self.rng.random()
            if rand < 0.55:
                return "question"
            elif rand < 0.75:
                return "follow_up_question"
            elif rand < 0.90:
                return "acknowledgment"
            else:
                return "statement"

    def _synthesize_contextual_utterance(self, rtype: str, aoi: str) -> str:
        """Create a plausible utterance for the given response_type and AOI."""
        if rtype == "silence":
            return ""
        
        # Clean AOI name for better templates
        clean_aoi = aoi.replace("_", " ")
        
        # Question templates - much more variety
        question_templates = [
            f"What does the {clean_aoi} signify?",
            f"Can you tell me more about the {clean_aoi}?",
            f"Why is the {clean_aoi} important?",
            f"What's special about the {clean_aoi}?",
            f"Who created the {clean_aoi}?",
            f"When was the {clean_aoi} made?",
            f"Where did the {clean_aoi} come from?",
            f"How was the {clean_aoi} used?",
            f"What's the story behind the {clean_aoi}?",
            f"What materials is the {clean_aoi} made from?",
            f"What's the significance of the {clean_aoi}?",
            f"Is there symbolism in the {clean_aoi}?",
            f"What culture does the {clean_aoi} represent?",
            f"Tell me about the history of the {clean_aoi}.",
            f"What period is the {clean_aoi} from?",
        ]
        
        # Statement templates - more natural
        statement_templates = [
            f"That's interesting about the {clean_aoi}.",
            f"I like the {clean_aoi}.",
            f"The {clean_aoi} looks beautiful.",
            f"I find the {clean_aoi} fascinating.",
            f"The {clean_aoi} is quite striking.",
            f"I'm drawn to the {clean_aoi}.",
            f"The craftsmanship of the {clean_aoi} is impressive.",
            f"I appreciate the detail in the {clean_aoi}.",
            f"The {clean_aoi} has such rich colors.",
            f"The {clean_aoi} stands out to me.",
        ]
        
        # Reference templates (referring back)
        reference_templates = [
            f"You mentioned the {clean_aoi} earlier.",
            f"Going back to the {clean_aoi}...",
            f"About the {clean_aoi}...",
            f"Regarding the {clean_aoi}...",
            f"I was thinking about the {clean_aoi}.",
            f"Can we return to the {clean_aoi}?",
            f"I have another question about the {clean_aoi}.",
        ]
        
        # Confusion templates
        confusion_templates = [
            f"I'm not sure I understand about the {clean_aoi}.",
            f"Could you clarify about the {clean_aoi}?",
            f"I'm confused about the {clean_aoi}.",
            f"What did you mean about the {clean_aoi}?",
            f"I didn't quite follow that about the {clean_aoi}.",
            f"Could you explain the {clean_aoi} again?",
        ]
        
        repeat_request_templates = [
            f"Sorry, who created the {clean_aoi} again?",
            f"Could you remind me of that detail about the {clean_aoi}?",
            f"I missed the name—can you repeat it?",
            f"Would you mind repeating that last fact?",
            f"I didn't quite catch that, could you say it again?",
        ]
        
        # NEW: Acknowledgment templates (when agent answers well)
        acknowledgment_templates = [
            f"Oh, that's fascinating about the {clean_aoi}!",
            f"I see, that makes sense about the {clean_aoi}.",
            f"That's really interesting, thank you!",
            f"Wow, I didn't know that about the {clean_aoi}.",
            f"That's helpful, I understand better now.",
            f"Interesting perspective on the {clean_aoi}.",
            f"Thank you for explaining that!",
            f"That clears things up about the {clean_aoi}.",
        ]
        
        # NEW: Follow-up question templates (when engaged and curious)
        followup_templates = [
            f"What else can you tell me about the {clean_aoi}?",
            f"How does that relate to other pieces in the collection?",
            f"Can you elaborate on that?",
            f"What's the historical context for this?",
            f"Are there other similar {clean_aoi} pieces?",
            f"What influenced the artist's choice here?",
        ]
        
        if rtype == "acknowledgment":
            return self.rng.choice(acknowledgment_templates)
        elif rtype == "follow_up_question":
            return self.rng.choice(followup_templates)
        elif rtype == "question":
            return self.rng.choice(question_templates)
        elif rtype == "statement":
            return self.rng.choice(statement_templates)
        elif rtype == "reference":
            return self.rng.choice(reference_templates)
        elif rtype == "confusion":
            return self.rng.choice(confusion_templates)
        elif rtype == "repeat_request":
            return self.rng.choice(repeat_request_templates)
        else:
            return self.rng.choice(question_templates)

    def _synthesize_llm_guided_utterance(self, agent_utterance: str, rtype: str, aoi: str,
                                         transition_rejected: bool = False,
                                         transition_success: bool = False,
                                         target_exhibit: Optional[str] = None,
                                         current_exhibit_completion: float = 0.0,
                                         turns_at_current_exhibit: int = 0,
                                         agent_option: Optional[str] = None,
                                         agent_subaction: Optional[str] = None) -> str:
        """
        Generate user response using LLM guidance to ensure responses directly address agent utterances.
        Falls back to templates if LLM is unavailable.
        """
        import os
        
        # Use compositional templates if template mode is enabled
        if os.environ.get('HRL_TEMPLATE_MODE') == '1':
            from src.simulator.visitor_templates import generate_visitor_utterance
            return generate_visitor_utterance(
                response_type=rtype, 
                visitor_state=None, 
                aoi=aoi, 
                rng=self.rng,
                agent_option=agent_option,
                agent_subaction=agent_subaction,
                transition_success=transition_success,
                transition_rejected=transition_rejected,
                target_exhibit=target_exhibit
            )
        
        try:
            # Try to use LLM for guided response - uses simulator LLM from centralized config
            from LLM_CONFIG import get_simulator_llm
            
            # Skip if in fast mode (use templates only)
            if os.environ.get('HRL_FAST_MODE') == '1':
                return self._synthesize_contextual_utterance(rtype, aoi)
            
            import time
            start_time = time.time()
            import os
            verbose = os.environ.get('HRL_VERBOSE', '0') == '1'
            if verbose:
                print(f"[Simulator LLM] Generating user response ({rtype})...", flush=True)
            llm = get_simulator_llm()
            clean_aoi = aoi.replace("_", " ")
            
            # Build a prompt that guides the LLM to generate contextually appropriate responses
            # Check for frustration when overstaying at exhausted exhibit
            if rtype == "confusion" and turns_at_current_exhibit >= 6 and current_exhibit_completion >= 0.80:
                # Frustration at staying too long at completed exhibit
                frustration_templates = [
                    f"We've been looking at {clean_aoi} for a while now...",
                    f"Haven't we covered everything about {clean_aoi}?",
                    f"I think I've learned all I can here, should we move on?",
                    f"Are there any other exhibits we should see?",
                    f"This has been interesting, but maybe we should explore something else?",
                    f"I feel like we've spent enough time on {clean_aoi}.",
                ]
                system_prompt = f"""You are a museum visitor with persona: {self.current_persona}.
You've been looking at the same exhibit ({clean_aoi}) for {turns_at_current_exhibit} turns and feel like you've learned everything about it.
Generate a SHORT (1-2 sentence) response expressing mild frustration or restlessness about staying at the same exhibit.

RULES:
- Express polite restlessness or desire to move on
- Indicate you feel you've covered this exhibit thoroughly
- Suggest seeing other exhibits
- Be polite but clearly ready to move on
- Use phrases like: "we've been here a while", "covered everything", "should we move on", "see something else"

Current exhibit: {clean_aoi}"""
            elif transition_rejected:
                # Special handling for rejected transitions - always confusion
                if current_exhibit_completion < 0.33:
                    confusion_reason = f"we've barely discussed anything about {clean_aoi} yet"
                elif current_exhibit_completion < 0.67:
                    confusion_reason = f"there's still so much more to learn about {clean_aoi}"
                else:
                    confusion_reason = f"I'd like to hear a bit more about {clean_aoi} before moving on"
                
                system_prompt = f"""You are a museum visitor with persona: {self.current_persona}.
The guide just suggested moving to another exhibit, but you're CONFUSED because {confusion_reason}.
Generate a SHORT (1-2 sentence) response expressing confusion about leaving so soon.

RULES:
- Express confusion or surprise at the suggestion to move
- Mention that you feel like you haven't learned enough here yet
- Ask to stay longer or learn more about the current exhibit
- Be polite but clearly confused about the rush
- Response type: confusion

Current exhibit: {clean_aoi}"""
            else:
                if rtype == "repeat_request":
                    system_prompt = f"""You are a museum visitor with persona: {self.current_persona}.
You didn't catch the last detail the guide shared. Politely ask them to repeat the fact.
Keep your response short (1 sentence) and clearly signal you'd like the information again."""
                else:
                    system_prompt = f"""You are a museum visitor with persona: {self.current_persona}.
Generate a SHORT (1-2 sentence) NATURAL response to what the museum guide just said.
Be conversational and genuine. React specifically to their statement.
Response type should be: {rtype}
Current exhibit: {clean_aoi}"""

            user_prompt = f"""Guide said: "{agent_utterance}"

Your {rtype} response (1-2 sentences):"""

            response = llm.generate(user_prompt, system_prompt=system_prompt)
            elapsed = time.time() - start_time
            if verbose:
                print(f"[Simulator LLM] User response received in {elapsed:.2f}s ({len(response)} chars)", flush=True)
            response = response.strip().strip('"')
            
            # Store prompts for detailed logging
            self._last_simulator_prompt = user_prompt
            self._last_simulator_system_prompt = system_prompt
            self._last_sim_llm_time = elapsed
            
            return response[:300]  # Limit length
            
        except Exception:
            # Fallback to template-based approach
            return self._synthesize_contextual_utterance(rtype, aoi)

    def _get_question_spam_multiplier(self) -> float:
        """Calculate penalty multiplier for question spam (reduces dwell).
        
        Returns a multiplier (0.0-1.0) that reduces dwell time.
        Lower values = more penalty = lower engagement.
        """
        multiplier = 1.0
        
        # Penalty for question spam: -0.15 per consecutive question after first
        if self.consecutive_ask_questions > 1:
            penalty = 0.15 * (self.consecutive_ask_questions - 1)
            multiplier = max(0.3, multiplier - penalty)  # Cap at 30% minimum
        
        return multiplier
    
    def _get_transition_spam_multiplier(self) -> float:
        """Calculate penalty multiplier for transition spam (reduces dwell).
        
        Returns a multiplier (0.0-1.0) that reduces dwell time.
        Lower values = more penalty = lower engagement.
        Transition spam penalty: -0.20 per consecutive transition after first (strengthened from -0.15).
        """
        multiplier = 1.0
        
        # Penalty for transition spam: -0.20 per consecutive transition after first (strengthened)
        if self.consecutive_transitions > 1:
            penalty = 0.20 * (self.consecutive_transitions - 1)  # Increased from 0.15
            multiplier = max(0.25, multiplier - penalty)  # Lower cap (25% vs 30% minimum)
        
        return multiplier
    
    def _get_dwell_stagnation_multiplier(self) -> float:
        """Calculate penalty multiplier for staying at same exhibit too long (reduces dwell).
        
        IMPORTANT: Dwell time should only decrease if:
        1. Visitor stays at the same exhibit (tracks via turns_at_current_exhibit)
        2. That exhibit is exhausted of new facts (exhibit_exhausted = True)
        
        If exhibit still has new facts, no penalty is applied (visitor is still engaged).
        Returns a multiplier (0.0-1.0) that reduces dwell time.
        Lower values = more penalty = lower engagement.
        
        FIXED: More gradual penalties to avoid sudden drops, and better recovery.
        """
        # Only apply penalty if exhibit is exhausted (no new facts available)
        if not getattr(self, 'exhibit_exhausted', False):
            # Exhibit still has new facts - no penalty, visitor is still engaged
            return 1.0
        
        # Exhibit is exhausted - apply GRADUAL penalty based on turns spent
        # More gradual than before to avoid sudden drops in engagement
        # IMPORTANT: Penalty should be very gradual to prevent dramatic drops
        if self.turns_at_current_exhibit <= 4:
            # No penalty for first 4 turns even if exhausted (reasonable time to wrap up)
            return 1.0
        elif self.turns_at_current_exhibit <= 6:
            # Very light penalty: 5-6 turns at exhausted exhibit (gradual start)
            return 0.98  # Minimal penalty to start
        elif self.turns_at_current_exhibit <= 8:
            # Light penalty: 7-8 turns at exhausted exhibit
            return 0.95  # Still minimal
        elif self.turns_at_current_exhibit <= 10:
            # Moderate-light penalty: 9-10 turns
            return 0.90  # Gradual decrease
        elif self.turns_at_current_exhibit <= 12:
            # Moderate penalty: 11-12 turns at exhausted exhibit
            return 0.85  # More noticeable but still reasonable
        elif self.turns_at_current_exhibit <= 15:
            # Moderate-strong penalty: 13-15 turns at exhausted exhibit
            return 0.75  # Encouraging transition but not dramatic
        elif self.turns_at_current_exhibit <= 20:
            # Strong penalty: 16-20 turns at exhausted exhibit
            return 0.60  # Strong encouragement to transition
        else:
            # Very strong penalty: 21+ turns at exhausted exhibit (strongly encourage transition)
            return 0.40  # Very low but not zero (still allows some engagement)
    
    def _synthesize_contextual_gaze(self, rtype: str, agent_option: str = None, agent_subaction: str = None, engagement_adjust_multiplier: float = 1.0) -> List[float]:
        """Generate synthetic gaze features based on response type.
        
        Response types directly encode quality:
        - acknowledgment, follow_up_question = HIGH engagement (agent did well)
        - confusion = LOW engagement (agent was off-topic or deflecting)
        - question, statement = MODERATE engagement (neutral)
        
        Question spam and transition spam reduce dwell time (simulator-level penalties).
        Transition insufficiency penalty is handled at environment/reward level.
        """
        # Apply engagement level multiplier from disengagement tracking
        engagement_multiplier = self.engagement_level
        
        # Apply penalty multiplier for question spam (reduces engagement)
        question_spam_multiplier = self._get_question_spam_multiplier()
        
        # Apply penalty multiplier for transition spam (reduces engagement)
        transition_spam_multiplier = self._get_transition_spam_multiplier()
        
        # Apply penalty multiplier for dwell stagnation (reduces engagement when staying at same exhibit too long)
        dwell_stagnation_multiplier = self._get_dwell_stagnation_multiplier()
        
        # Combined penalty multiplier (question spam + transition spam + dwell stagnation)
        combined_spam_multiplier = question_spam_multiplier * transition_spam_multiplier * dwell_stagnation_multiplier

        # Aggregate multiplier including pacing bonuses/penalties (questions, explain spam, etc.)
        effective_multiplier = engagement_multiplier * combined_spam_multiplier * engagement_adjust_multiplier

        if agent_subaction == "RecoverEngagement":
            # RecoverEngagement action: higher dwell with diminishing returns on consecutive uses
            self._consecutive_recover_count += 1
            base_dwell = self._randf(0.55, 0.75)
            if self._consecutive_recover_count == 2:
                base_dwell *= 0.70
            elif self._consecutive_recover_count >= 3:
                base_dwell *= 0.40
            dwell_time = self._clip(base_dwell, 0.10, 1.0)
            saccade_span = max(0.05, np.random.normal(0.07, 0.03))
        else:
            # Reset diminishing-returns counter for any non-RecoverEngagement action
            self._consecutive_recover_count = 0

            # Different patterns for different response types
            if rtype in ["acknowledgment", "follow_up_question"]:
                # HIGH engagement - agent did well, user is satisfied and curious
                base_dwell = self._randf(0.75, 0.95)
                dwell_time = self._clip(base_dwell * effective_multiplier, 0.2, 1.0)
                saccade_span = max(0.05, np.random.normal(0.07, 0.03))  # Low saccades (focused)

            elif rtype == "question":
                # MODERATE-HIGH engagement - genuinely curious
                base_dwell = self._randf(0.4, 0.7)
                dwell_time = self._clip(base_dwell * effective_multiplier, 0.2, 1.0)
                saccade_span = max(0.05, np.random.normal(0.08, 0.04))

            elif rtype == "statement":
                # MODERATE engagement - making statements
                base_dwell = self._randf(0.3, 0.6)
                dwell_time = self._clip(base_dwell * effective_multiplier, 0.2, 1.0)
                saccade_span = max(0.05, np.random.normal(0.09, 0.04))

            elif rtype == "confusion":
                # LOW engagement - agent was unhelpful, off-topic, or deflecting
                base_dwell = self._randf(0.25, 0.50)
                # Confusion gets extra penalty from engagement tracking AND spam penalties
                dwell_time = self._clip(base_dwell * effective_multiplier * 0.8, 0.1, 0.6)
                saccade_span = max(0.05, np.random.normal(0.12, 0.05))  # Higher saccades (unfocused)

            else:
                # Default: MODERATE engagement
                base_dwell = self._randf(0.3, 0.6)
                dwell_time = self._clip(base_dwell * effective_multiplier, 0.2, 1.0)
                saccade_span = max(0.05, np.random.normal(0.09, 0.04))

            # Apply action variety adjustment (questions boost, explain spam decays)
            dwell_time = self._adjust_dwell_for_action_variety(agent_option, agent_subaction, dwell_time)
        
        # Common gaze features (persona-influenced)
        persona = self.current_persona or "Agreeable"
        stats = self.SILENCE_STATS.get(persona, self.SILENCE_STATS["Agreeable"])
        
        gaze_entropy = self._clip(np.random.normal(stats["TurnGazeEntropy"][0], stats["TurnGazeEntropy"][1]), 0.0, 2.5)
        fix_change_rate = self._clip(np.random.normal(stats["TurnFixChangeRate"][0], stats["TurnFixChangeRate"][1]), 0.2, 4.0)
        dom_ratio = self._clip(dwell_time * self._randf(0.6, 0.95), 0.0, 1.0)
        entry_latency = self._clip(np.random.normal(stats["GazeEntryLatency"][0], stats["GazeEntryLatency"][1]), 0.1, 12.0)

        return [
            float(dwell_time),
            float(saccade_span),
            float(gaze_entropy),
            float(fix_change_rate),
            float(dom_ratio),
            float(entry_latency),
        ]

    def _make_silence_response(self) -> Dict[str, Any]:
        persona = self.current_persona or "Agreeable"
        
        # For silence, low engagement (dwell time should be low)
        # Sample other gaze features from persona stats
        dwell_time = self._randf(0.1, 0.4)  # Low engagement during silence
        
        # Sample other features from stats
        stats = self.SILENCE_STATS[persona]
        saccade_span = max(0.05, np.random.normal(0.1, 0.05))  # Higher saccades during silence
        gaze_entropy = self._clip(np.random.normal(stats["TurnGazeEntropy"][0], stats["TurnGazeEntropy"][1]), 0.0, 2.5)
        fix_change_rate = self._clip(np.random.normal(stats["TurnFixChangeRate"][0], stats["TurnFixChangeRate"][1]), 0.2, 4.0)
        dom_ratio = self._clip(self._randf(0.4, 0.7), 0.0, 1.0)  # Lower dominance during silence
        entry_latency = self._clip(np.random.normal(stats["GazeEntryLatency"][0], stats["GazeEntryLatency"][1]), 0.1, 12.0)
        
        feats = [
            float(dwell_time),
            float(saccade_span),
            float(gaze_entropy),
            float(fix_change_rate),
            float(dom_ratio),
            float(entry_latency),
        ]
        
        return {
            "utterance": None,
            "aoi": self.current_aoi,
            "persona": persona,
            "gaze_features": feats,
            "response_type": "silence",
        }
    def _clip(self, v: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, v)))

    def _randf(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.rng.random()


# Public alias used by tests and external callers
Sim8Adapter = Sim8Simulator

