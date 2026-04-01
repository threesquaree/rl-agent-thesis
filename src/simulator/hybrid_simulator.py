"""
Hybrid Simulator for HRL Museum Agent

Combines the state machine's discrete behavioral backbone with sim8's continuous
engagement dynamics to produce cleaner, richer training signals.

Architecture (5 layers):
  1. State Machine Backbone: 9 visitor states, deterministic triggers, recovery mechanics
  2. Engagement Dynamics: Continuous engagement_level, multiplicative cascades, novelty boost
  3. Hybrid Dwell Computation: State band + engagement position + penalties + noise
  4. Transition Model: State machine triggers + sim8 probability model
  5. Response Generation: State-aware LLM/template utterances

Key innovation: The state machine determines the reward "band" (e.g., ENGAGED = 0.75-0.90),
while sim8's engagement dynamics determine the position within that band. This gives clean
reward signals (non-overlapping bands) with rich within-state nuance.

See HYBRID_SIMULATOR_ARCHITECTURE.md for full design rationale.
"""

import random
import re
import os
import time
from typing import Dict, Any, List, Optional
from collections import deque

import numpy as np

# Reuse state definitions from state_machine_simulator
from src.simulator.state_machine_simulator import (
    VisitorState,
    DWELL_RANGES,
    STATE_PROMPTS,
    FALLBACK_TEMPLATES,
)


class HybridSimulator:
    """
    Hybrid visitor simulator: state machine backbone + sim8 engagement dynamics.

    Same interface as Sim8Simulator and StateMachineSimulator for drop-in use.

    Args:
        knowledge_graph: SimpleKnowledgeGraph instance
        exhibits: List of exhibit names (fallback)
        seed: Random seed for reproducibility
        stochasticity: 0.0-1.0 controls sim8 influence
            0.0 = pure state machine (deterministic, uniform within bands)
            0.5 = balanced hybrid (recommended default)
            1.0 = maximum sim8 influence (engagement dominates positioning)
    """

    PERSONAS = ["Agreeable", "Conscientious", "Neurotic"]

    GAZE_LABELS = [
        "DwellTime", "SaccadeSpan", "TurnGazeEntropy",
        "TurnFixChangeRate", "DominantObjectRatio", "GazeEntryLatency",
    ]

    # --- State machine thresholds (literature-grounded) ---
    OVERLOAD_THRESHOLD = 3
    FATIGUE_THRESHOLD = 3
    CURIOUS_PROBABILITY = 0.30
    CONFUSED_PROBABILITY = 0.20
    READY_COVERAGE = 0.80
    READY_TURNS = 4
    BORED_TOPIC_THRESHOLD = 3
    HIGHLY_ENGAGED_THRESHOLD = 3

    # --- Recovery rates (from state_machine) ---
    RECOVERY_RATES = {
        VisitorState.HIGHLY_ENGAGED: {"OfferTransition": 0.85},
        VisitorState.CONFUSED: {
            "ClarifyFact": 0.90, "AskClarification": 0.70,
            "OfferTransition": 0.60, "ExplainNewFact": 0.40,
        },
        VisitorState.OVERLOADED: {"AskQuestion": 0.85, "OfferTransition": 0.75},
        VisitorState.CURIOUS: {"Explain": 0.90},
        VisitorState.BORED_OF_TOPIC: {"Explain": 0.85, "OfferTransition": 0.90},
        VisitorState.FATIGUED: {"AskQuestion": 0.80, "OfferTransition": 0.85},
        VisitorState.READY_TO_MOVE: {"OfferTransition": 0.95},
        VisitorState.DISENGAGED: {"OfferTransition": 0.50},
    }

    # --- Persona profiles (from state_machine) ---
    PERSONA_PROFILES = {
        "Explorer": {
            "curious_prob": 0.40,
            "overload_threshold": 5,
            "fatigue_threshold": 5,
            "ready_turns": 5,
            "recovery_modifier": 1.0,
        },
        "Focused": {
            "curious_prob": 0.20,
            "overload_threshold": 3,
            "fatigue_threshold": 6,
            "ready_turns": 4,
            "recovery_modifier": 1.0,
        },
        "Impatient": {
            "curious_prob": 0.30,
            "overload_threshold": 3,
            "fatigue_threshold": 3,
            "ready_turns": 3,
            "recovery_modifier": 0.90,
        },
    }

    # --- Sim8 persona gaze statistics ---
    GAZE_STATS = {
        "Agreeable": {
            "SaccadeSpan": (0.103, 0.056),
            "TurnGazeEntropy": (0.850, 0.458),
            "TurnFixChangeRate": (2.158, 0.847),
            "DominantObjectRatio": (0.723, 0.183),
            "GazeEntryLatency": (6.442, 12.526),
        },
        "Conscientious": {
            "SaccadeSpan": (0.122, 0.067),
            "TurnGazeEntropy": (0.585, 0.541),
            "TurnFixChangeRate": (2.086, 1.410),
            "DominantObjectRatio": (0.794, 0.213),
            "GazeEntryLatency": (4.632, 6.696),
        },
        "Neurotic": {
            "SaccadeSpan": (0.125, 0.065),
            "TurnGazeEntropy": (1.018, 0.376),
            "TurnFixChangeRate": (2.854, 0.745),
            "DominantObjectRatio": (0.658, 0.200),
            "GazeEntryLatency": (2.311, 3.778),
        },
    }

    # ================================================================
    # Initialisation
    # ================================================================

    def __init__(
        self,
        knowledge_graph=None,
        exhibits: Optional[List[str]] = None,
        seed: int = 42,
        stochasticity: float = 0.5,
    ):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.stochasticity = max(0.0, min(1.0, stochasticity))

        # --- Knowledge graph ---
        if knowledge_graph:
            self._init_from_knowledge_graph(knowledge_graph)
        elif exhibits:
            self.exhibits = exhibits
            self.aoi_to_exhibit: Dict[str, str] = {}
            self.exhibit_to_aois: Dict[str, List[str]] = {ex: [ex] for ex in exhibits}
            self.knowledge_graph = None
        else:
            raise ValueError("Must provide either knowledge_graph or exhibits list")

        # LLM handler (lazy)
        self._llm = None

        # --- Session state (initialised properly in initialize_session) ---
        self.current_persona: Optional[str] = None
        self.current_exhibit: Optional[str] = None
        self.current_aoi: Optional[str] = None
        self.persona_profile: Optional[str] = None

        # State machine backbone
        self.visitor_state: VisitorState = VisitorState.ENGAGED

        # Action tracking (state machine)
        self.consecutive_explain_count = 0
        self.turns_without_question = 0
        self.turns_at_current_exhibit = 0
        self.current_exhibit_completion = 0.0
        self.turns_in_ready_to_move = 0
        self.recent_actions: deque = deque(maxlen=10)
        self.turn_counter = 0
        self.turns_since_recovery = 999
        self.consecutive_lecture_turns = 0
        self.lecture_fatigue_penalty = 0.0
        self.consecutive_same_topic_turns = 0
        self.last_fact_category: Optional[str] = None
        self.consecutive_engaged_turns = 0
        self.unique_actions_in_engaged: set = set()
        self.turns_since_highly_engaged = 0
        self.recovery_count = 0
        self.overload_episodes = 0
        self.transition_rejected = False
        self.transition_accepted = False
        self.transition_target: Optional[str] = None
        self.turns_at_exhausted_exhibit = 0
        self.engagement_boost_pending = 0.0
        self.turns_without_new_fact = 0
        self.consecutive_questions = 0
        self.mentioned_facts: set = set()
        self.repeated_fact_count = 0

        # Sim8 engagement dynamics
        self.engagement_level = 1.0  # continuous 0.0-1.0
        self.off_topic_strikes = 0
        self.consecutive_ask_questions = 0
        self.consecutive_transitions = 0

        # Dialogue tracking
        self.dialogue_history: List[Dict[str, str]] = []
        self.last_user_utterance: str = ""
        self.last_agent_option: Optional[str] = None
        self.last_agent_subaction: Optional[str] = None
        self.aoi_usage_count: Dict[str, int] = {}
        self.seen_aois: set = set()
        self.last_user_response: Dict[str, Any] = {}
        self._last_sim_llm_time: float = 0.0

        # Compatibility stubs
        self.AOI_TO_PARENT: Dict = {}
        self.PARENT_TO_EXHIBIT: Dict = {}

    def _init_from_knowledge_graph(self, kg):
        self.exhibits = kg.get_exhibit_names()
        self.aoi_to_exhibit: Dict[str, str] = {}
        self.exhibit_to_aois: Dict[str, List[str]] = {ex: [ex] for ex in self.exhibits}
        for ex in self.exhibits:
            self.aoi_to_exhibit[ex] = ex
        self.knowledge_graph = kg

    # ================================================================
    # Public API
    # ================================================================

    def initialize_session(
        self,
        persona: Optional[str] = None,
        persona_profile: Optional[str] = None,
    ):
        """Initialise a new dialogue session."""
        self.current_persona = persona or self.rng.choice(self.PERSONAS)
        self.current_exhibit = self.rng.choice(self.exhibits)
        self.current_aoi = self.current_exhibit

        # Persona profile (state machine axis)
        if persona_profile:
            self.persona_profile = persona_profile
        else:
            self.persona_profile = self.rng.choice(list(self.PERSONA_PROFILES.keys()))

        # Reset state machine backbone
        self.visitor_state = VisitorState.ENGAGED
        self.consecutive_explain_count = 0
        self.turns_without_question = 0
        self.turns_at_current_exhibit = 0
        self.current_exhibit_completion = 0.0
        self.turns_in_ready_to_move = 0
        self.recent_actions.clear()
        self.turn_counter = 0
        self.turns_since_recovery = 999
        self.consecutive_lecture_turns = 0
        self.lecture_fatigue_penalty = 0.0
        self.consecutive_same_topic_turns = 0
        self.last_fact_category = None
        self.consecutive_engaged_turns = 0
        self.unique_actions_in_engaged.clear()
        self.turns_since_highly_engaged = 0
        self.recovery_count = 0
        self.overload_episodes = 0
        self.transition_rejected = False
        self.transition_accepted = False
        self.transition_target = None
        self.turns_at_exhausted_exhibit = 0
        self.engagement_boost_pending = 0.0
        self.turns_without_new_fact = 0
        self.consecutive_questions = 0
        self.mentioned_facts.clear()
        self.repeated_fact_count = 0

        # Reset sim8 engagement dynamics
        self.engagement_level = 1.0
        self.off_topic_strikes = 0
        self.consecutive_ask_questions = 0
        self.consecutive_transitions = 0

        # Reset dialogue tracking
        self.aoi_usage_count.clear()
        self.seen_aois.clear()
        self.dialogue_history.clear()
        self.last_user_utterance = ""
        self.last_user_response = {}
        self.last_agent_option = None
        self.last_agent_subaction = None

        return {
            "persona": self.current_persona,
            "persona_profile": self.persona_profile,
            "exhibit": self.current_exhibit,
            "aoi": self.current_aoi,
        }

    def get_current_aoi(self) -> str:
        return self.current_exhibit or self.exhibits[0]

    def get_introduction_exchange(self, exhibit_name: str = None) -> Dict[str, str]:
        exhibit = exhibit_name or self.current_exhibit or "this artwork"
        exhibit_display = exhibit.replace("_", " ")
        return {
            "agent_greeting": (
                f"Welcome! I'm delighted to be your guide today. "
                f"We're starting here at {exhibit_display}. "
                f"Take a moment to look at it - what catches your eye first?"
            ),
            "user_response": (
                "Oh, this is lovely! I'm excited to learn more about it. "
                "There's something about the composition that draws me in."
            ),
        }

    def inject_introduction(self, exhibit_name: str = None) -> Dict[str, str]:
        intro = self.get_introduction_exchange(exhibit_name)
        self.dialogue_history.append({"role": "agent", "text": intro["agent_greeting"]})
        self.dialogue_history.append({"role": "user", "text": intro["user_response"]})
        self.last_user_utterance = intro["user_response"]
        return intro

    # ================================================================
    # Main response generation
    # ================================================================

    def generate_user_response(
        self,
        agent_utterance: str,
        agent_option: str = None,
        agent_subaction: str = None,
        target_exhibit: str = None,
        current_exhibit_completion: float = 0.0,
        exhibit_exhausted: bool = False,
        target_exhibit_completion: float = 0.0,
        target_exhibit_exhausted: bool = False,
    ) -> Dict[str, Any]:
        """Generate visitor response using hybrid logic.

        Flow:
        1. Update action tracking (both state machine + sim8)
        2. Check state triggers (state machine deterministic + engagement-modulated random)
        3. Process recovery (state machine base + engagement bonus)
        4. Compute hybrid dwell (state band + engagement position + penalties + noise)
        5. Generate utterance (state-aware LLM / templates)
        6. Update sim8 engagement dynamics
        7. Return response
        """
        verbose = os.environ.get("HRL_VERBOSE", "0") == "1"

        # --- Update action tracking ---
        self.last_agent_option = agent_option
        self.last_agent_subaction = agent_subaction
        self.recent_actions.append((agent_option, agent_subaction))
        self.current_exhibit_completion = current_exhibit_completion
        self.turns_at_current_exhibit += 1
        self.turn_counter += 1

        # Exhausted exhibit tracking
        if exhibit_exhausted:
            self.turns_at_exhausted_exhibit += 1
        else:
            self.turns_at_exhausted_exhibit = 0

        # Content starvation
        if agent_subaction == "ExplainNewFact":
            self.turns_without_new_fact = 0
            self.consecutive_questions = 0
        else:
            self.turns_without_new_fact += 1

        # Consecutive questions (diminishing returns)
        if agent_option == "AskQuestion":
            self.consecutive_questions += 1
        # (reset handled above when ExplainNewFact)

        # Consecutive explain count (for OVERLOADED)
        if agent_option == "Explain" and agent_subaction == "ExplainNewFact":
            self.consecutive_explain_count += 1
        else:
            self.consecutive_explain_count = 0

        # Turns without question (for FATIGUED)
        if agent_option == "AskQuestion":
            self.turns_without_question = 0
        else:
            self.turns_without_question += 1

        # Lecture fatigue
        if agent_option in ("Explain", "Conclude"):
            self.consecutive_lecture_turns += 1
            self.lecture_fatigue_penalty = min(0.40, self.consecutive_lecture_turns * 0.08)
        elif agent_option == "AskQuestion":
            self.consecutive_lecture_turns = 0
            self.lecture_fatigue_penalty = 0.0

        self.turns_since_recovery += 1

        # Topic tracking (BORED_OF_TOPIC)
        self._update_topic_tracking(agent_utterance)
        # Fact repetition tracking
        self._track_fact_repetition(agent_utterance)

        # Sim8 spam counters
        if agent_option == "AskQuestion":
            self.consecutive_ask_questions += 1
            self.consecutive_transitions = 0
        elif agent_option == "OfferTransition" or agent_subaction == "SuggestMove":
            self.consecutive_transitions += 1
            self.consecutive_ask_questions = 0
        else:
            self.consecutive_ask_questions = 0
            self.consecutive_transitions = 0

        # Reset transition flags
        self.transition_rejected = False
        self.transition_accepted = False
        self.transition_target = None

        prev_state = self.visitor_state

        # Hallucination detection
        hallucination_detected = self._check_agent_hallucination(agent_utterance)
        if hallucination_detected:
            # Sim8 engagement drop for hallucination
            self.engagement_level = max(0.1, self.engagement_level * 0.4)
            if verbose:
                print("[Hybrid] HALLUCINATION: engagement_level dropped")

        # === STEP 1: State triggers (Layer 1) ===
        self._check_state_triggers(
            agent_option, agent_subaction, agent_utterance,
            current_exhibit_completion, exhibit_exhausted, verbose,
            hallucination_detected=hallucination_detected,
        )

        # === STEP 2: Recovery (Layer 1 + Layer 2 enhancement) ===
        transition_success = self._process_recovery(
            agent_option, agent_subaction, agent_utterance,
            target_exhibit, current_exhibit_completion, verbose,
        )

        if verbose and prev_state != self.visitor_state:
            print(f"[Hybrid] {prev_state.value} -> {self.visitor_state.value}")

        # === STEP 3: Update sim8 engagement dynamics (Layer 2) ===
        engagement_adjust = self._compute_engagement_adjustment(
            agent_option, agent_subaction, agent_utterance,
            current_exhibit_completion, exhibit_exhausted,
            target_exhibit, target_exhibit_completion, target_exhibit_exhausted,
            transition_success, verbose,
        )
        self.engagement_level = max(
            0.1,
            min(1.0, self.engagement_level + (engagement_adjust - 1.0) * 0.2),
        )

        # === STEP 4: Hybrid dwell computation (Layer 3) ===
        dwell_time = self._compute_dwell(verbose)
        gaze_features = self._synthesize_gaze(dwell_time)

        # === STEP 5: Utterance generation (Layer 5) ===
        sim_start = time.time()
        utterance = self._generate_utterance(agent_utterance, agent_option, agent_subaction)
        self._last_sim_llm_time = time.time() - sim_start

        # === STEP 6: Response type ===
        response_type = self._state_to_response_type()

        # === Handle exhibit transition ===
        is_transition_attempt = (
            agent_option == "OfferTransition" or agent_subaction == "SuggestMove"
        )
        if is_transition_attempt and transition_success and target_exhibit:
            if self.turns_at_exhausted_exhibit >= 3:
                self.engagement_boost_pending = 0.25
                if verbose:
                    print(f"[Hybrid] ESCAPE BOOST: +0.25 for leaving exhausted exhibit")
            self.current_exhibit = target_exhibit
            self.current_aoi = target_exhibit
            self.turns_at_current_exhibit = 0
            self.turns_at_exhausted_exhibit = 0
            self.consecutive_lecture_turns = 0
            self.lecture_fatigue_penalty = 0.0
            self.consecutive_same_topic_turns = 0
            self.last_fact_category = None
            # Sim8 engagement recovery on fresh exhibit
            self.engagement_level = min(1.0, self.engagement_level + 0.10)
            if verbose:
                print(f"[Hybrid] Transitioned to: {target_exhibit}")

        # === Build response ===
        response = {
            "utterance": utterance,
            "aoi": self.current_aoi,
            "persona": self.current_persona,
            "gaze_features": gaze_features,
            "response_type": response_type,
            "visitor_state": self.visitor_state.value,
            "engagement_level": dwell_time,  # primary reward signal
            "off_topic_strikes": self.off_topic_strikes,
            "transition_success": transition_success,
            "simulator_llm_time": self._last_sim_llm_time,
            "repeat_request": False,
        }

        self.last_user_response = response
        self.last_user_utterance = utterance
        self.dialogue_history.append({"role": "agent", "text": agent_utterance})
        self.dialogue_history.append({"role": "user", "text": utterance})

        return response

    # ================================================================
    # Layer 1: State Machine Backbone
    # ================================================================

    def _get_effective_threshold(self, base_threshold: str) -> float:
        profile = self.PERSONA_PROFILES.get(self.persona_profile, {})
        threshold_map = {
            "overload": ("overload_threshold", self.OVERLOAD_THRESHOLD),
            "fatigue": ("fatigue_threshold", self.FATIGUE_THRESHOLD),
            "curious": ("curious_prob", self.CURIOUS_PROBABILITY),
            "ready": ("ready_turns", self.READY_TURNS),
        }
        key, default = threshold_map.get(base_threshold, (None, None))
        return profile.get(key, default) if key else default

    def _check_state_triggers(
        self,
        agent_option: str,
        agent_subaction: str,
        agent_utterance: str,
        current_completion: float,
        exhibit_exhausted: bool,
        verbose: bool,
        hallucination_detected: bool = False,
    ):
        """State triggers: deterministic (state machine) + engagement-modulated random."""
        overload_thresh = self._get_effective_threshold("overload")
        fatigue_thresh = self._get_effective_threshold("fatigue")
        base_curious_prob = self._get_effective_threshold("curious")
        ready_turns = self._get_effective_threshold("ready")

        # --- HIGHLY_ENGAGED decay ---
        if self.visitor_state == VisitorState.HIGHLY_ENGAGED:
            self.turns_since_highly_engaged += 1
            if self.turns_since_highly_engaged >= 2:
                self.visitor_state = VisitorState.ENGAGED
                self.turns_since_highly_engaged = 0
                if verbose:
                    print("[Hybrid] HIGHLY_ENGAGED decayed to ENGAGED")
            return

        # Only trigger from ENGAGED
        if self.visitor_state != VisitorState.ENGAGED:
            return

        # --- Positive escalation: HIGHLY_ENGAGED ---
        self.consecutive_engaged_turns += 1
        self.unique_actions_in_engaged.add(agent_option)
        if (
            self.consecutive_engaged_turns >= self.HIGHLY_ENGAGED_THRESHOLD
            and len(self.unique_actions_in_engaged) >= 2
        ):
            self.visitor_state = VisitorState.HIGHLY_ENGAGED
            self.turns_since_highly_engaged = 0
            if verbose:
                print(f"[Hybrid] HIGHLY_ENGAGED triggered: {self.consecutive_engaged_turns} varied turns")
            return

        # === DETERMINISTIC TRIGGERS (priority order) ===
        if self.consecutive_explain_count >= overload_thresh:
            self.visitor_state = VisitorState.OVERLOADED
            self.overload_episodes += 1
            self._reset_engaged_tracking()
            if verbose:
                print(f"[Hybrid] OVERLOADED: {self.consecutive_explain_count} consecutive explains")
            return

        if self.turns_without_question >= fatigue_thresh:
            self.visitor_state = VisitorState.FATIGUED
            self._reset_engaged_tracking()
            if verbose:
                print(f"[Hybrid] FATIGUED: {self.turns_without_question} turns without question")
            return

        if self.consecutive_same_topic_turns >= self.BORED_TOPIC_THRESHOLD:
            self.visitor_state = VisitorState.BORED_OF_TOPIC
            self._reset_engaged_tracking()
            if verbose:
                print(f"[Hybrid] BORED_OF_TOPIC: {self.consecutive_same_topic_turns} same topic turns")
            return

        if current_completion >= self.READY_COVERAGE and self.turns_at_current_exhibit >= ready_turns:
            self.visitor_state = VisitorState.READY_TO_MOVE
            self._reset_engaged_tracking()
            if verbose:
                print(f"[Hybrid] READY_TO_MOVE: {current_completion:.0%} coverage, {self.turns_at_current_exhibit} turns")
            return

        # === ENGAGEMENT-MODULATED RANDOM TRIGGERS (hybrid innovation) ===
        # Sim8's engagement_level modulates the random trigger probabilities.
        # High engagement -> more curiosity, less confusion.
        # Low engagement -> less curiosity, more confusion.
        # The stochasticity parameter controls how much modulation is applied.

        eng = self.engagement_level
        mod_strength = self.stochasticity  # 0.0 = no modulation, 1.0 = full modulation

        # CURIOUS: base_prob * (0.5 + 0.5 * engagement) when fully modulated
        # At stochasticity=0.0, uses base_prob unchanged
        curious_mod = 1.0 + mod_strength * (0.5 + 0.5 * eng - 1.0)
        curious_prob = base_curious_prob * curious_mod

        if self.turn_counter > 3 and self.rng.random() < curious_prob:
            self.visitor_state = VisitorState.CURIOUS
            self._reset_engaged_tracking()
            if verbose:
                print(f"[Hybrid] CURIOUS triggered (prob={curious_prob:.2f}, eng={eng:.2f})")
            return

        # CONFUSED: base_prob * (1.5 - engagement) when fully modulated
        base_confused = self.CONFUSED_PROBABILITY
        if hallucination_detected:
            base_confused = 0.60
        confused_mod = 1.0 + mod_strength * (1.5 - eng - 1.0)
        confused_prob = base_confused * confused_mod

        if self.turn_counter > 1 and self.turns_since_recovery >= 2:
            if self.rng.random() < confused_prob:
                self.visitor_state = VisitorState.CONFUSED
                self._reset_engaged_tracking()
                if verbose:
                    reason = "hallucination" if hallucination_detected else "random"
                    print(f"[Hybrid] CONFUSED triggered ({reason}, prob={confused_prob:.2f}, eng={eng:.2f})")

    def _reset_engaged_tracking(self):
        self.consecutive_engaged_turns = 0
        self.unique_actions_in_engaged.clear()

    def _process_recovery(
        self,
        agent_option: str,
        agent_subaction: str,
        agent_utterance: str,
        target_exhibit: str,
        current_completion: float,
        verbose: bool,
    ) -> bool:
        """Recovery with state machine base rates + engagement bonus (hybrid enhancement)."""
        transition_success = False
        is_transition_attempt = (
            agent_option == "OfferTransition" or agent_subaction == "SuggestMove"
        )

        # --- HIGHLY_ENGAGED: high acceptance for transitions ---
        if self.visitor_state == VisitorState.HIGHLY_ENGAGED and is_transition_attempt:
            transition_prob = 0.85 if current_completion >= 0.40 else 0.50
            if self.rng.random() < transition_prob:
                transition_success = True
                self.transition_accepted = True
                self.transition_target = target_exhibit
            else:
                self.transition_rejected = True
            return transition_success

        # --- ENGAGED: sim8-style transition probability ---
        if self.visitor_state == VisitorState.ENGAGED and is_transition_attempt:
            if current_completion < 0.20:
                transition_prob = 0.20
            elif current_completion < 0.40:
                transition_prob = 0.50
            elif current_completion < 0.60:
                transition_prob = 0.80
            else:
                transition_prob = 0.95
            if self.rng.random() < transition_prob:
                transition_success = True
                self.transition_accepted = True
                self.transition_target = target_exhibit
            else:
                self.visitor_state = VisitorState.CURIOUS
                self.transition_rejected = True
            return transition_success

        if self.visitor_state == VisitorState.ENGAGED:
            return transition_success

        # --- Negative states: recovery with fatigue + engagement bonus ---
        fatigue_penalty = min(0.45, 0.15 * self.recovery_count)
        base_rates = self.RECOVERY_RATES.get(self.visitor_state, {})
        recovery_rates = {k: max(0.10, v - fatigue_penalty) for k, v in base_rates.items()}

        # Hybrid: engagement bonus (+0-10% based on engagement momentum)
        engagement_bonus = 0.1 * self.engagement_level * self.stochasticity

        recovered = False

        if self.visitor_state == VisitorState.CONFUSED:
            rate = None
            if agent_subaction == "ClarifyFact":
                rate = recovery_rates.get("ClarifyFact", 0.90)
            elif agent_subaction == "AskClarification":
                rate = recovery_rates.get("AskClarification", 0.70)
            elif is_transition_attempt:
                rate = recovery_rates.get("OfferTransition", 0.60)
                if rate and self.rng.random() < rate + engagement_bonus:
                    self.visitor_state = VisitorState.ENGAGED
                    self.turns_since_recovery = 0
                    transition_success = True
                    self.transition_accepted = True
                    self.transition_target = target_exhibit
                    recovered = True
                rate = None  # handled
            elif agent_option == "Explain" and agent_subaction == "ExplainNewFact":
                rate = recovery_rates.get("ExplainNewFact", 0.40)

            if rate is not None and self.rng.random() < rate + engagement_bonus:
                self.visitor_state = VisitorState.ENGAGED
                self.turns_since_recovery = 0
                recovered = True

        elif self.visitor_state == VisitorState.OVERLOADED:
            if agent_option == "AskQuestion":
                if self.rng.random() < recovery_rates.get("AskQuestion", 0.85) + engagement_bonus:
                    self.visitor_state = VisitorState.ENGAGED
                    self.consecutive_explain_count = 0
                    self.turns_since_recovery = 0
                    recovered = True
            elif is_transition_attempt:
                if self.rng.random() < recovery_rates.get("OfferTransition", 0.75) + engagement_bonus:
                    self.visitor_state = VisitorState.ENGAGED
                    self.consecutive_explain_count = 0
                    self.turns_since_recovery = 0
                    transition_success = True
                    self.transition_accepted = True
                    self.transition_target = target_exhibit
                    recovered = True

        elif self.visitor_state == VisitorState.CURIOUS:
            if agent_option == "Explain":
                has_fact = bool(re.search(r'\[[A-Z]{2}_\d{3}\]', agent_utterance or ""))
                if has_fact and self.rng.random() < recovery_rates.get("Explain", 0.90) + engagement_bonus:
                    self.visitor_state = VisitorState.ENGAGED
                    self.turns_since_recovery = 0
                    recovered = True
            elif agent_option == "AskQuestion":
                if self.rng.random() < 0.50:
                    self.visitor_state = VisitorState.CONFUSED
                    if verbose:
                        print("[Hybrid] DEFLECTION: AskQuestion when CURIOUS -> CONFUSED")

        elif self.visitor_state == VisitorState.BORED_OF_TOPIC:
            if agent_option == "Explain":
                current_fact = self._extract_fact_category(agent_utterance)
                is_different = current_fact and current_fact != self.last_fact_category
                profile = self.PERSONA_PROFILES.get(self.persona_profile, {})
                recovery_mod = profile.get("recovery_modifier", 1.0)
                base_rate = recovery_rates.get("Explain", 0.85)
                if is_different and self.rng.random() < base_rate * recovery_mod + engagement_bonus:
                    self.visitor_state = VisitorState.ENGAGED
                    self.consecutive_same_topic_turns = 0
                    self.last_fact_category = current_fact
                    self.turns_since_recovery = 0
                    recovered = True
            elif is_transition_attempt:
                base_trans = self.RECOVERY_RATES.get(VisitorState.BORED_OF_TOPIC, {}).get("OfferTransition", 0.90)
                trans_rate = max(0.10, base_trans - fatigue_penalty)
                if self.rng.random() < trans_rate + engagement_bonus:
                    self.visitor_state = VisitorState.ENGAGED
                    self.consecutive_same_topic_turns = 0
                    self.turns_since_recovery = 0
                    transition_success = True
                    self.transition_accepted = True
                    self.transition_target = target_exhibit
                    recovered = True

        elif self.visitor_state == VisitorState.FATIGUED:
            if agent_option == "AskQuestion":
                if self.rng.random() < recovery_rates.get("AskQuestion", 0.80) + engagement_bonus:
                    self.visitor_state = VisitorState.ENGAGED
                    self.turns_without_question = 0
                    self.turns_since_recovery = 0
                    recovered = True
            elif is_transition_attempt:
                if self.rng.random() < recovery_rates.get("OfferTransition", 0.85) + engagement_bonus:
                    self.visitor_state = VisitorState.ENGAGED
                    self.turns_since_recovery = 0
                    transition_success = True
                    self.transition_accepted = True
                    self.transition_target = target_exhibit
                    recovered = True

        elif self.visitor_state == VisitorState.READY_TO_MOVE:
            self.turns_in_ready_to_move += 1
            if is_transition_attempt:
                if self.rng.random() < recovery_rates.get("OfferTransition", 0.95) + engagement_bonus:
                    self.visitor_state = VisitorState.ENGAGED
                    self.turns_in_ready_to_move = 0
                    self.turns_since_recovery = 0
                    transition_success = True
                    self.transition_accepted = True
                    self.transition_target = target_exhibit
                    recovered = True
            elif agent_option == "Explain":
                self.visitor_state = VisitorState.FATIGUED
                if verbose:
                    print(f"[Hybrid] WRONG ACTION: {agent_subaction} when READY_TO_MOVE -> FATIGUED")
            elif self.turns_in_ready_to_move >= 3:
                self.visitor_state = VisitorState.DISENGAGED
                if verbose:
                    print(f"[Hybrid] ESCALATION: READY_TO_MOVE x{self.turns_in_ready_to_move} -> DISENGAGED")

        elif self.visitor_state == VisitorState.DISENGAGED:
            if is_transition_attempt:
                if self.rng.random() < recovery_rates.get("OfferTransition", 0.50) + engagement_bonus:
                    self.visitor_state = VisitorState.ENGAGED
                    self.turns_in_ready_to_move = 0
                    self.turns_since_recovery = 0
                    transition_success = True
                    self.transition_accepted = True
                    self.transition_target = target_exhibit
                    recovered = True
                else:
                    self.transition_rejected = True

        if recovered:
            self.recovery_count += 1
            self._reset_engaged_tracking()
            # Hybrid: engagement level recovery boost on successful recovery
            self.engagement_level = min(1.0, self.engagement_level + 0.15)
            if verbose:
                eff_penalty = min(0.45, 0.15 * self.recovery_count)
                print(f"[Hybrid] Recovery #{self.recovery_count} (fatigue -{eff_penalty:.0%}), eng={self.engagement_level:.2f}")

        return transition_success

    # ================================================================
    # Layer 2: Sim8 Engagement Dynamics
    # ================================================================

    def _compute_engagement_adjustment(
        self,
        agent_option, agent_subaction, agent_utterance,
        current_completion, exhibit_exhausted,
        target_exhibit, target_completion, target_exhausted,
        transition_success, verbose,
    ) -> float:
        """Compute sim8-style engagement adjustment multiplier.

        Returns a multiplier around 1.0:
          >1.0 = engagement improves
          <1.0 = engagement degrades
        The caller applies: eng += (mult - 1.0) * 0.2
        """
        mult = 1.0

        # Question pacing bonus/penalty (sim8)
        if agent_option == "AskQuestion":
            turns_since_q = self.turns_without_question  # already reset, so use pre-reset
            # Good pacing: ask after 3+ non-question turns
            if self.consecutive_ask_questions <= 1:
                mult *= 1.3  # well-spaced question bonus
            elif self.consecutive_ask_questions == 2:
                mult *= 1.0  # neutral
            else:
                mult *= 0.80  # question spam penalty

        # Explain spam on exhausted exhibit (sim8)
        if agent_option == "Explain" and exhibit_exhausted:
            mult *= 0.85

        # Novelty: new facts boost engagement (sim8)
        if agent_utterance:
            fact_ids = re.findall(r'\[[A-Z]{2}_\d{3}\]', agent_utterance)
            new_facts = [f for f in fact_ids if f not in self.mentioned_facts]
            if new_facts and not exhibit_exhausted:
                mult *= 1.15  # novelty boost
            elif new_facts and exhibit_exhausted:
                mult *= 1.06  # diminished novelty

        # Transition quality (sim8)
        is_transition = agent_option == "OfferTransition" or agent_subaction == "SuggestMove"
        if is_transition and transition_success:
            if target_exhausted:
                mult *= 0.85  # bad target
            elif target_completion and target_completion >= 0.67:
                mult *= 0.90  # partially done target
            else:
                mult *= 1.15  # good target
        elif is_transition and not transition_success:
            mult *= 0.85  # rejected

        # Off-topic / meta-commentary detection (simplified sim8 check)
        if agent_utterance:
            meta_patterns = [
                r"(?i)as an ai", r"(?i)i don't have personal",
                r"(?i)let me think", r"(?i)that's a great question",
            ]
            for pat in meta_patterns:
                if re.search(pat, agent_utterance):
                    self.off_topic_strikes += 1
                    mult *= 0.80
                    break
            else:
                # Gradual recovery from off-topic
                if self.off_topic_strikes > 0:
                    self.off_topic_strikes = max(0, self.off_topic_strikes - 1)

        return mult

    def _get_question_spam_multiplier(self) -> float:
        """Sim8: penalty for consecutive AskQuestion."""
        if self.consecutive_ask_questions > 1:
            penalty = 0.15 * (self.consecutive_ask_questions - 1)
            return max(0.30, 1.0 - penalty)
        return 1.0

    def _get_transition_spam_multiplier(self) -> float:
        """Sim8: penalty for consecutive OfferTransition."""
        if self.consecutive_transitions > 1:
            penalty = 0.20 * (self.consecutive_transitions - 1)
            return max(0.25, 1.0 - penalty)
        return 1.0

    def _get_dwell_stagnation_multiplier(self) -> float:
        """Sim8: penalty for staying at exhausted exhibit too long."""
        if self.turns_at_exhausted_exhibit <= 4:
            return 1.0
        elif self.turns_at_exhausted_exhibit <= 6:
            return 0.98
        elif self.turns_at_exhausted_exhibit <= 8:
            return 0.95
        elif self.turns_at_exhausted_exhibit <= 10:
            return 0.90
        elif self.turns_at_exhausted_exhibit <= 12:
            return 0.85
        elif self.turns_at_exhausted_exhibit <= 15:
            return 0.75
        elif self.turns_at_exhausted_exhibit <= 20:
            return 0.60
        else:
            return 0.40

    # ================================================================
    # Layer 3: Hybrid Dwell Computation
    # ================================================================

    def _compute_dwell(self, verbose: bool = False) -> float:
        """Hybrid dwell: state band + engagement position + penalties + noise.

        Steps:
        1. State determines the band [low, high]
        2. Engagement dynamics position within band (scaled by stochasticity)
        3. State machine additive penalties/boosts
        4. Sim8 multiplicative cascades
        5. Novelty boost (from pending engagement)
        6. Persona noise (scaled by stochasticity)
        7. Final clamp [0.0, 1.0]
        """
        low, high = DWELL_RANGES[self.visitor_state]
        band_width = high - low

        # STEP 2: Position within band
        # At stochasticity=0.0: uniform random (pure state machine)
        # At stochasticity=1.0: fully engagement-driven
        uniform_pos = self.rng.uniform(0.0, 1.0)
        engagement_pos = self.engagement_level
        # Interpolate between uniform and engagement-driven
        position = (1.0 - self.stochasticity) * uniform_pos + self.stochasticity * engagement_pos
        base_dwell = low + position * band_width

        # STEP 3: State machine penalties/boosts (additive)
        # 1. Explain ratio penalty
        if len(self.recent_actions) >= 5:
            explain_count = sum(
                1 for opt, sub in list(self.recent_actions)[-5:]
                if opt == "Explain" and sub == "ExplainNewFact"
            )
            if explain_count >= 3:
                base_dwell *= 0.90

        # 2. Cumulative overload penalty
        if self.overload_episodes > 0:
            base_dwell = max(0.15, base_dwell - 0.05 * self.overload_episodes)

        # 3. READY_TO_MOVE escalation penalty
        if self.visitor_state == VisitorState.READY_TO_MOVE:
            base_dwell = max(0.10, base_dwell - 0.10 * self.turns_in_ready_to_move)

        # 4. Fact repetition penalty
        if self.repeated_fact_count >= 2:
            base_dwell = max(0.20, base_dwell - 0.10 * self.repeated_fact_count)

        # 5. Topic staleness decay
        if self.turns_at_current_exhibit >= 8 and self.visitor_state == VisitorState.ENGAGED:
            base_dwell = max(0.30, base_dwell - 0.05 * (self.turns_at_current_exhibit - 7))

        # 6. Lecture fatigue penalty
        if self.lecture_fatigue_penalty > 0:
            base_dwell = max(0.25, base_dwell - self.lecture_fatigue_penalty)

        # 7. AskQuestion boost with diminishing returns
        if (
            self.last_agent_option == "AskQuestion"
            and self.visitor_state in (VisitorState.FATIGUED, VisitorState.OVERLOADED, VisitorState.ENGAGED)
        ):
            if self.consecutive_questions == 1:
                base_dwell = min(1.0, base_dwell + 0.30)
            elif self.consecutive_questions == 2:
                base_dwell = min(1.0, base_dwell + 0.10)
            else:
                base_dwell = max(0.0, base_dwell - 0.25)

        # 7b. AskOpinion boost
        if (
            self.last_agent_subaction == "AskOpinion"
            and self.visitor_state in (VisitorState.FATIGUED, VisitorState.OVERLOADED, VisitorState.ENGAGED)
        ):
            if self.consecutive_questions == 1:
                base_dwell = min(1.0, base_dwell + 0.25)
            elif self.consecutive_questions == 2:
                base_dwell = min(1.0, base_dwell + 0.08)
            else:
                base_dwell = max(0.0, base_dwell - 0.20)

        # 8. ExplainNewFact recovery boost after slump
        if (
            self.last_agent_subaction == "ExplainNewFact"
            and self.lecture_fatigue_penalty >= 0.16
        ):
            base_dwell = min(0.80, base_dwell + 0.20)

        # 9. Exhausted exhibit penalty
        if self.turns_at_exhausted_exhibit >= 5:
            exhaustion_penalty = min(0.35, 0.08 * (self.turns_at_exhausted_exhibit - 4))
            base_dwell = max(0.10, base_dwell - exhaustion_penalty)

        # 10. Transition escape boost
        if self.engagement_boost_pending > 0:
            base_dwell = min(0.90, base_dwell + self.engagement_boost_pending)
            self.engagement_boost_pending = 0

        # 11. Content starvation penalty
        if self.turns_without_new_fact >= 3:
            starvation = min(0.50, 0.10 * (self.turns_without_new_fact - 2))
            base_dwell = max(0.0, base_dwell - starvation)

        # STEP 4: Sim8 multiplicative cascades
        sim8_mult = (
            self._get_question_spam_multiplier()
            * self._get_transition_spam_multiplier()
            * self._get_dwell_stagnation_multiplier()
        )
        base_dwell *= sim8_mult

        # STEP 6: Persona noise (prevents threshold gaming)
        if self.stochasticity > 0:
            noise_std = self.stochasticity * 0.05
            noise = self.np_rng.normal(0, noise_std)
            base_dwell += noise

        # STEP 7: Final clamp
        return float(max(0.0, min(1.0, base_dwell)))

    # ================================================================
    # Layer 4 helpers (transition is handled in _process_recovery)
    # ================================================================

    def _extract_fact_category(self, utterance: str) -> Optional[str]:
        if not utterance:
            return None
        match = re.search(r'\[([A-Z]{2})_\d{3}\]', utterance)
        return match.group(1) if match else None

    def _update_topic_tracking(self, agent_utterance: str):
        current_category = self._extract_fact_category(agent_utterance)
        if current_category:
            if current_category == self.last_fact_category:
                self.consecutive_same_topic_turns += 1
            else:
                self.consecutive_same_topic_turns = 1
                self.last_fact_category = current_category

    def _track_fact_repetition(self, agent_utterance: str):
        if not agent_utterance:
            return
        current_facts = re.findall(r'\[([A-Z]{2}_\d{3})\]', agent_utterance)
        if not current_facts:
            return
        any_new = False
        for fact in current_facts:
            if fact not in self.mentioned_facts:
                any_new = True
                self.mentioned_facts.add(fact)
        if any_new:
            self.repeated_fact_count = 0
        else:
            self.repeated_fact_count += 1

    def _check_agent_hallucination(self, utterance: str) -> bool:
        if not utterance:
            return False
        mentioned = re.findall(r'\[([A-Z]{2}_\d{3})\]', utterance)
        if not mentioned:
            return False
        if self.knowledge_graph:
            valid_facts = self.knowledge_graph.get_all_fact_ids()
            return any(f not in valid_facts for f in mentioned)
        return False

    # ================================================================
    # Layer 5: Response Generation
    # ================================================================

    def _generate_utterance(
        self,
        agent_utterance: str,
        agent_option: Optional[str] = None,
        agent_subaction: Optional[str] = None,
    ) -> str:
        """Generate visitor utterance using LLM or fallback templates."""
        # Template mode
        if os.environ.get("HRL_TEMPLATE_MODE") == "1":
            from src.simulator.visitor_templates import generate_visitor_utterance
            return generate_visitor_utterance(
                response_type=None,
                visitor_state=self.visitor_state.value,
                aoi=self.current_exhibit,
                rng=self.rng,
                agent_option=agent_option,
                agent_subaction=agent_subaction,
                transition_success=self.transition_accepted,
                transition_rejected=self.transition_rejected,
                target_exhibit=self.transition_target,
            )

        # Transition rejection (hardcoded)
        if self.transition_rejected:
            return self.rng.choice([
                "Wait, I want to learn more about this first.",
                "Hold on, can you tell me more before we move?",
                "Actually, I'm still curious about this.",
                "I'd like to stay here a bit longer.",
                "Not yet, I have more questions about this one.",
            ])

        # Try LLM
        if os.environ.get("HRL_FAST_MODE") != "1":
            try:
                utterance = self._generate_llm_utterance(agent_utterance)
                if utterance:
                    return utterance
            except Exception as e:
                if os.environ.get("HRL_VERBOSE") == "1":
                    print(f"[Hybrid] LLM failed, using fallback: {e}")

        # Fallback templates
        templates = FALLBACK_TEMPLATES.get(
            self.visitor_state, FALLBACK_TEMPLATES[VisitorState.ENGAGED]
        )
        return self.rng.choice(templates)

    def _generate_llm_utterance(self, agent_utterance: str) -> Optional[str]:
        try:
            from LLM_CONFIG import get_simulator_llm

            if self._llm is None:
                self._llm = get_simulator_llm()

            history_context = ""
            if len(self.dialogue_history) >= 2:
                recent = self.dialogue_history[-4:]
                lines = [
                    f"{'Guide' if t['role'] == 'agent' else 'Visitor'}: {t['text']}"
                    for t in recent
                ]
                history_context = "Recent conversation:\n" + "\n".join(lines)

            exhibit_name = self.current_exhibit or "the artwork"
            prompt_template = STATE_PROMPTS.get(
                self.visitor_state, STATE_PROMPTS[VisitorState.ENGAGED]
            )
            prompt = prompt_template.format(
                agent_utterance=agent_utterance[:200],
                exhibit_name=exhibit_name,
                history_context=history_context,
            )

            response = self._llm.generate(
                prompt,
                system_prompt=(
                    "You are a museum visitor. Generate ONLY ONE short response "
                    "(1-2 sentences max). No lists, no multiple statements, no imagining "
                    "future dialogue. Just one natural reply."
                ),
            )

            if response:
                response = response.strip().strip('"').strip("'")
                for prefix in ("Visitor:", "Response:", "Answer:"):
                    if response.startswith(prefix):
                        response = response[len(prefix):].strip()
                response = response.split("\n")[0].strip()
                sentences = re.split(r'(?<=[.!?])\s+', response)
                if len(sentences) > 2:
                    response = " ".join(sentences[:2])
                if len(response) > 200:
                    response = response[:200].rsplit(" ", 1)[0]
                return response

        except Exception as e:
            if os.environ.get("HRL_VERBOSE") == "1":
                print(f"[Hybrid] LLM error: {e}")
        return None

    def _state_to_response_type(self) -> str:
        mapping = {
            VisitorState.HIGHLY_ENGAGED: "acknowledgment",
            VisitorState.ENGAGED: "acknowledgment",
            VisitorState.CONFUSED: "confusion",
            VisitorState.OVERLOADED: "statement",
            VisitorState.CURIOUS: "question",
            VisitorState.BORED_OF_TOPIC: "question",
            VisitorState.FATIGUED: "statement",
            VisitorState.READY_TO_MOVE: "statement",
            VisitorState.DISENGAGED: "statement",
        }
        return mapping.get(self.visitor_state, "statement")

    # ================================================================
    # Gaze synthesis
    # ================================================================

    def _synthesize_gaze(self, dwell_time: float) -> List[float]:
        """Generate 6D gaze vector with persona-specific statistics."""
        persona = self.current_persona or "Agreeable"
        stats = self.GAZE_STATS.get(persona, self.GAZE_STATS["Agreeable"])

        saccade = max(0.05, self.np_rng.normal(*stats["SaccadeSpan"]))
        entropy = float(np.clip(self.np_rng.normal(*stats["TurnGazeEntropy"]), 0.0, 2.5))
        fix_rate = float(np.clip(self.np_rng.normal(*stats["TurnFixChangeRate"]), 0.2, 4.0))
        dom_ratio = float(np.clip(dwell_time * self.rng.uniform(0.6, 0.95), 0.0, 1.0))
        latency = float(np.clip(self.np_rng.normal(*stats["GazeEntryLatency"]), 0.1, 12.0))

        return [
            float(dwell_time),
            float(saccade),
            entropy,
            fix_rate,
            dom_ratio,
            latency,
        ]

    # ================================================================
    # State accessors
    # ================================================================

    def get_current_state(self) -> Dict[str, Any]:
        return {
            "aoi": self.current_aoi,
            "current_exhibit": self.current_exhibit,
            "persona": self.current_persona,
            "persona_profile": self.persona_profile,
            "visitor_state": self.visitor_state.value,
            "engagement_level": self.engagement_level,
            "stochasticity": self.stochasticity,
            "consecutive_explain_count": self.consecutive_explain_count,
            "turns_without_question": self.turns_without_question,
            "turns_at_current_exhibit": self.turns_at_current_exhibit,
            "seen_aois": list(self.seen_aois),
            "last_user_response": dict(self.last_user_response) if self.last_user_response else {},
        }

    def update_from_state(self, state_focus: int, target_exhibit: str = None):
        if target_exhibit and target_exhibit != self.current_exhibit:
            if target_exhibit in self.exhibits:
                self.current_exhibit = target_exhibit
                self.current_aoi = target_exhibit
                self.turns_at_current_exhibit = 0
                self.turns_in_ready_to_move = 0
                self.visitor_state = VisitorState.ENGAGED
