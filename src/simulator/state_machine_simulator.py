"""
State Machine Simulator for HRL Museum Agent

A literature-grounded visitor simulator with coherent state machine design:
- Deterministic state transitions based on agent actions
- Non-overlapping dwell ranges for clear reward signals
- LLM-generated utterances that match visitor state sentiment
- Recovery actions that the agent can learn

Literature grounding:
- OVERLOADED: Working memory ~4 items (Bitgood 2013, Cowan 2001)
- FATIGUED: Question spacing 3-5 turns (Woo et al. 2024)
- CURIOUS: Users expect responses to questions (Grice 1975)
- CONFUSED: Conversational repair sequences (Schegloff et al. 1977)
- READY_TO_MOVE: Variety seeking after satiation (Bitgood 2013)
"""

import random
import re
import os
import time
from enum import Enum
from typing import Dict, Any, List, Optional
from collections import deque

import numpy as np


class VisitorState(Enum):
    """Visitor engagement states with associated dwell ranges."""
    HIGHLY_ENGAGED = "highly_engaged"  # Exceptional engagement (0.90-1.00)
    ENGAGED = "engaged"           # Normal, attentive (0.75-0.90)
    CONFUSED = "confused"         # Needs clarification (0.20-0.35)
    OVERLOADED = "overloaded"     # Too many facts (0.30-0.45)
    CURIOUS = "curious"           # Asked a question (0.55-0.70)
    BORED_OF_TOPIC = "bored_of_topic"  # Wants variety in content (0.45-0.60)
    FATIGUED = "fatigued"         # Bored, wants variety (0.40-0.55)
    READY_TO_MOVE = "ready_to_move"  # Exhibit exhausted (0.50-0.65)
    DISENGAGED = "disengaged"     # Visitor has given up (0.05-0.15)


# Dwell ranges per state (from Daniel's thesis Table 7 - non-overlapping design)
DWELL_RANGES = {
    VisitorState.HIGHLY_ENGAGED: (0.90, 1.00),  # Best possible - visitor is captivated
    VisitorState.ENGAGED: (0.75, 0.90),          # Normal, attentive
    VisitorState.CURIOUS: (0.55, 0.70),           # Asked a question
    VisitorState.READY_TO_MOVE: (0.50, 0.65),     # Exhibit exhausted
    VisitorState.BORED_OF_TOPIC: (0.45, 0.60),    # Wants variety in content
    VisitorState.FATIGUED: (0.40, 0.55),           # Bored, wants variety
    VisitorState.OVERLOADED: (0.30, 0.45),         # Too many facts
    VisitorState.CONFUSED: (0.20, 0.35),           # Needs clarification
    VisitorState.DISENGAGED: (0.05, 0.15),         # Visitor has given up
}

# State-specific prompts for LLM utterance generation
# Includes emotional context and encourages varied response lengths
STATE_PROMPTS = {
    VisitorState.HIGHLY_ENGAGED: """MUSEUM VISITOR SIMULATION
Exhibit: "{exhibit_name}"
Guide said: "{agent_utterance}"
{history_context}

YOUR EMOTIONAL STATE: Genuinely interested and enjoying this. You find this topic fascinating.

Generate a natural visitor response. VARY your style randomly:
- Sometimes enthusiastic: "Oh that's so cool!" or "No way, really?"
- Sometimes brief but positive: "Nice." or "Huh, interesting."
- Sometimes ask a follow-up: "Wait so how does that work?"

Keep it casual and realistic. 1-2 sentences max.
Output ONLY the visitor's words, nothing else.""",

    VisitorState.ENGAGED: """MUSEUM VISITOR SIMULATION
Exhibit: "{exhibit_name}"
Guide said: "{agent_utterance}"
{history_context}

YOUR EMOTIONAL STATE: Interested, paying attention, but normal energy level.

Generate a natural visitor response. VARY your style randomly:
- Sometimes one word: "Cool." "Nice." "Huh." "Yeah."
- Sometimes a short reaction: "Oh okay, that makes sense."
- Sometimes a brief question: "What about the colors?"

Be casual, not theatrical. Real museum visitors aren't overly enthusiastic.
Output ONLY the visitor's words, nothing else.""",

    VisitorState.CONFUSED: """MUSEUM VISITOR SIMULATION
Exhibit: "{exhibit_name}"
Guide said: "{agent_utterance}"
{history_context}

YOUR EMOTIONAL STATE: Confused. Something the guide said didn't make sense to you.

Generate a confused response. VARY your style:
- Direct: "Wait, what?" or "I don't get it."
- Specific: "What do you mean by [term from utterance]?"
- Hesitant: "Um... I'm not sure I follow."

Sound like a real confused person, not a dramatic actor.
Output ONLY the visitor's words, nothing else.""",

    VisitorState.OVERLOADED: """MUSEUM VISITOR SIMULATION
Exhibit: "{exhibit_name}"
Guide said: "{agent_utterance}"
{history_context}

YOUR EMOTIONAL STATE: Overwhelmed. Too much information. Brain is full. Need a pause.

Generate an overwhelmed response. VARY your style:
- Minimal: "Okay." "Right." "Mmhm."
- Honest: "That's a lot." or "Okay slow down."
- Polite: "Let me think about that for a sec."

You're mentally tired. Keep it SHORT. Don't ask questions.
Output ONLY the visitor's words, nothing else.""",

    VisitorState.CURIOUS: """MUSEUM VISITOR SIMULATION
Exhibit: "{exhibit_name}"
Guide said: "{agent_utterance}"
{history_context}

YOUR EMOTIONAL STATE: Curious. Something caught your attention and you want to know more.

Generate a question. VARY your style:
- Direct: "Why is that?" or "How come?"
- Specific: "What happened to [thing mentioned]?"
- Casual: "So like, was that common back then?"

One question only. Be specific to what was just said.
Output ONLY the visitor's question, nothing else.""",

    VisitorState.BORED_OF_TOPIC: """MUSEUM VISITOR SIMULATION
Exhibit: "{exhibit_name}"
Guide said: "{agent_utterance}"
{history_context}

YOUR EMOTIONAL STATE: Bored of THIS topic. You want to hear about something else at this exhibit.

Generate a redirect. VARY your style:
- Direct: "What else is there?" or "Anything else about this?"
- Hint: "I think I got that part."
- Blunt: "Yeah yeah, what about the other stuff?"

You're not rude, just ready for variety.
Output ONLY the visitor's words, nothing else.""",

    VisitorState.FATIGUED: """MUSEUM VISITOR SIMULATION
Exhibit: "{exhibit_name}"
Guide said: "{agent_utterance}"

YOUR EMOTIONAL STATE: Fatigued. Low energy. The guide has been talking too much.

Generate a MINIMAL response. Pick ONE:
"Mm." / "Yeah." / "Sure." / "Okay." / "Right." / "Uh huh." / "Mhm." / "K."

ONLY 1-2 words. You're tired of talking.
Output ONLY the visitor's brief acknowledgment, nothing else.""",

    VisitorState.READY_TO_MOVE: """MUSEUM VISITOR SIMULATION
Exhibit: "{exhibit_name}"
Guide said: "{agent_utterance}"
{history_context}

YOUR EMOTIONAL STATE: Ready to move on. You've seen enough of this exhibit.

Generate a hint to move. VARY your style:
- Direct: "Can we see something else?" or "What's next?"
- Polite: "This was great, what else is there?"
- Casual: "Cool, should we keep going?"

You're not being rude, just ready for the next thing.
Output ONLY the visitor's words, nothing else.""",

    VisitorState.DISENGAGED: """MUSEUM VISITOR SIMULATION
Exhibit: "{exhibit_name}"
Guide said: "{agent_utterance}"
{history_context}

YOUR EMOTIONAL STATE: Annoyed/disengaged. You've lost interest. The guide won't stop talking.

Generate an annoyed response. VARY your style:
- Blunt: "Okay." or "Sure." (flat, uninterested)
- Irritated: "Can we please move on?"
- Checked out: "Mhm." or just silence cue like "..."

You're done with this. Don't pretend to be interested.
Output ONLY the visitor's words, nothing else.""",
}

# Fallback templates when LLM is unavailable
FALLBACK_TEMPLATES = {
    VisitorState.HIGHLY_ENGAGED: [
        "Oh cool!",
        "No way, really?",
        "That's awesome.",
        "Huh, I didn't know that!",
        "Nice!",
    ],
    VisitorState.ENGAGED: [
        "Cool.",
        "Nice.",
        "Huh.",
        "Yeah, interesting.",
        "Okay.",
    ],
    VisitorState.CONFUSED: [
        "Wait, what?",
        "I don't get it.",
        "Huh?",
        "What do you mean?",
        "Sorry, what?",
    ],
    VisitorState.OVERLOADED: [
        "Okay.",
        "Right.",
        "That's a lot.",
        "Mm.",
        "Got it.",
    ],
    VisitorState.CURIOUS: [
        "Why's that?",
        "How come?",
        "Who made it?",
        "When was this?",
        "Really? Why?",
    ],
    VisitorState.BORED_OF_TOPIC: [
        "What else?",
        "Anything else about this?",
        "Got it, what about the rest?",
        "Okay. Next?",
        "Yeah, what else is there?",
    ],
    VisitorState.FATIGUED: [
        "Mm.",
        "Yeah.",
        "Okay.",
        "Sure.",
        "Mhm.",
    ],
    VisitorState.READY_TO_MOVE: [
        "What's next?",
        "Can we move on?",
        "Should we keep going?",
        "What else is there?",
        "Cool, next?",
    ],
    VisitorState.DISENGAGED: [
        "Okay.",
        "Sure.",
        "Can we go?",
        "Mhm.",
        "...",
    ],
}


class StateMachineSimulator:
    """
    Literature-grounded visitor simulator with coherent state machine.
    
    Key features:
    - Deterministic state transitions based on agent behavior
    - Non-overlapping dwell ranges for clear reward signals
    - LLM-generated utterances matching visitor sentiment
    - Recovery actions the agent can learn
    
    Same interface as Sim8Simulator for drop-in replacement.
    """
    
    PERSONAS = ["Agreeable", "Conscientious", "Neurotic"]
    
    # Gaze feature labels (for compatibility)
    GAZE_LABELS = [
        "DwellTime", "SaccadeSpan", "TurnGazeEntropy",
        "TurnFixChangeRate", "DominantObjectRatio", "GazeEntryLatency"
    ]
    
    # Thresholds (literature-grounded, lowered to prevent ExplainNewFact spam)
    OVERLOAD_THRESHOLD = 3      # Consecutive ExplainNewFact before OVERLOADED
    FATIGUE_THRESHOLD = 3       # Turns without AskQuestion before FATIGUED
    CURIOUS_PROBABILITY = 0.30  # Thesis Table 7: 30% random from ENGAGED
    CONFUSED_PROBABILITY = 0.20 # Thesis Table 7: 20% random + deflection
    READY_COVERAGE = 0.80       # 80% coverage before READY_TO_MOVE
    READY_TURNS = 4             # Minimum turns at exhibit before READY_TO_MOVE
    
    # Recovery success rates
    RECOVERY_RATES = {
        VisitorState.HIGHLY_ENGAGED: {"OfferTransition": 0.85},  # Fresh content appeal
        VisitorState.CONFUSED: {"ClarifyFact": 0.90, "AskClarification": 0.70, "OfferTransition": 0.60, "ExplainNewFact": 0.40},
        VisitorState.OVERLOADED: {"AskQuestion": 0.85, "OfferTransition": 0.75},
        VisitorState.CURIOUS: {"Explain": 0.90},  # Any explain with fact
        VisitorState.BORED_OF_TOPIC: {"Explain": 0.85, "OfferTransition": 0.90},  # New topic or new exhibit
        VisitorState.FATIGUED: {"AskQuestion": 0.80, "OfferTransition": 0.85},
        VisitorState.READY_TO_MOVE: {"OfferTransition": 0.95},
        VisitorState.DISENGAGED: {"OfferTransition": 0.50},  # Hard to recover - only 50%
    }
    
    # Additional thresholds for new states
    BORED_TOPIC_THRESHOLD = 3  # Same topic 3+ turns triggers BORED_OF_TOPIC
    HIGHLY_ENGAGED_THRESHOLD = 3  # 3+ engaged turns with varied actions
    
    def __init__(self, knowledge_graph=None, exhibits: Optional[List[str]] = None, seed: int = 42):
        """Initialize state machine simulator.
        
        Args:
            knowledge_graph: SimpleKnowledgeGraph instance
            exhibits: List of exhibit names (fallback)
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        
        # Build mappings from knowledge graph
        if knowledge_graph:
            self._init_from_knowledge_graph(knowledge_graph)
        elif exhibits:
            self.exhibits = exhibits
            self.aoi_to_exhibit = {}
            self.exhibit_to_aois = {ex: [] for ex in exhibits}
        else:
            raise ValueError("Must provide either knowledge_graph or exhibits list")
        
        # LLM handler (lazy loaded)
        self._llm = None
        
        # Session state
        self.current_persona: Optional[str] = None
        self.current_exhibit: Optional[str] = None
        self.current_aoi: Optional[str] = None
        self.visitor_state: VisitorState = VisitorState.ENGAGED
        
        # State tracking
        self.consecutive_explain_count = 0
        self.turns_without_question = 0
        self.turns_at_current_exhibit = 0
        self.current_exhibit_completion = 0.0
        self.turns_in_ready_to_move = 0  # Escalation counter for exhausted exhibits
        self.recent_actions: deque = deque(maxlen=10)
        
        # Turn counter for absolute episode position (used for Turn 1 protection)
        self.turn_counter = 0
        
        # Recovery buffer - immunity from random confusion after recovery
        self.turns_since_recovery = 999  # Start high so no immunity initially
        
        # Lecture fatigue tracking (consecutive passive actions without AskQuestion)
        self.consecutive_lecture_turns = 0
        self.lecture_fatigue_penalty = 0.0
        
        # New state tracking for BORED_OF_TOPIC and HIGHLY_ENGAGED
        self.consecutive_same_topic_turns = 0
        self.last_fact_category: Optional[str] = None
        self.consecutive_engaged_turns = 0
        self.unique_actions_in_engaged: set = set()
        self.turns_since_highly_engaged = 0  # Decay counter for HIGHLY_ENGAGED
        
        # Persona modulation (affects thresholds)
        self.persona_profile: Optional[str] = None  # Explorer, Focused, or Impatient
        
        # Anti-spam tracking (prevent ExplainNewFact spam)
        self.recovery_count = 0  # Tracks recoveries this session (reduces success rate)
        self.overload_episodes = 0  # Tracks OVERLOADED episodes (escalating penalty)
        
        # Transition tracking
        self.transition_rejected = False  # Flag for response generation when visitor wants to stay
        self.transition_accepted = False  # Flag for response generation when visitor accepts transition
        self.transition_target = None  # Target exhibit for accepted transition
        
        # Exhausted exhibit tracking
        self.turns_at_exhausted_exhibit = 0  # Turns spent at 100% completion exhibit
        self.engagement_boost_pending = 0.0  # Boost applied after escaping exhausted exhibit
        
        # Content starvation tracking
        self.turns_without_new_fact = 0  # Turns since last ExplainNewFact
        
        # Question spam prevention (diminishing returns)
        self.consecutive_questions = 0  # Consecutive AskQuestion turns
        
        # Fact repetition tracking
        self.mentioned_facts: set = set()  # All facts mentioned this session
        self.repeated_fact_count = 0  # Consecutive repetitions of already-mentioned facts
        
        # Dialogue tracking
        self.dialogue_history: List[Dict[str, str]] = []
        self.last_user_utterance: str = ""
        self.last_agent_option: Optional[str] = None
        self.last_agent_subaction: Optional[str] = None
        
        # Response tracking
        self.aoi_usage_count: Dict[str, int] = {}
        self.seen_aois: set = set()
        self.last_user_response: Dict[str, Any] = {}
        self._last_sim_llm_time: float = 0.0
        
        # Mappings for compatibility
        self.AOI_TO_PARENT = {}
        self.PARENT_TO_EXHIBIT = {}
    
    def _init_from_knowledge_graph(self, kg):
        """Initialize exhibit/AOI mappings from knowledge graph."""
        self.exhibits = kg.get_exhibit_names()
        self.aoi_to_exhibit = {}
        self.exhibit_to_aois = {ex: [ex] for ex in self.exhibits}
        
        # Simple mapping: each exhibit is its own AOI
        for ex in self.exhibits:
            self.aoi_to_exhibit[ex] = ex
        
        self.knowledge_graph = kg
    
    # Persona profiles with threshold modulation
    PERSONA_PROFILES = {
        "Explorer": {
            "curious_prob": 0.40,      # Higher curiosity (40% vs 30%)
            "overload_threshold": 5,   # More tolerant of info dumps
            "fatigue_threshold": 5,    # More patient
            "ready_turns": 5,          # Spends more time per exhibit
            "recovery_modifier": 1.0,  # Normal recovery
        },
        "Focused": {
            "curious_prob": 0.20,      # Lower curiosity (20% vs 30%)
            "overload_threshold": 3,   # Gets overwhelmed faster
            "fatigue_threshold": 6,    # Doesn't need questions as much
            "ready_turns": 4,          # Standard
            "recovery_modifier": 1.0,  # Normal recovery
        },
        "Impatient": {
            "curious_prob": 0.30,      # Normal curiosity
            "overload_threshold": 3,   # Low patience for info
            "fatigue_threshold": 3,    # Gets bored fast
            "ready_turns": 3,          # Wants to move on quickly
            "recovery_modifier": 0.90, # 10% harder to recover
        },
    }
    
    def initialize_session(self, persona: Optional[str] = None, persona_profile: Optional[str] = None):
        """Initialize a new dialogue session.
        
        Args:
            persona: Big Five persona (Agreeable, Conscientious, Neurotic)
            persona_profile: Behavior profile (Explorer, Focused, Impatient)
        """
        self.current_persona = persona or self.rng.choice(self.PERSONAS)
        self.current_exhibit = self.rng.choice(self.exhibits)
        self.current_aoi = self.current_exhibit
        
        # Select persona profile (affects thresholds)
        if persona_profile:
            self.persona_profile = persona_profile
        else:
            self.persona_profile = self.rng.choice(list(self.PERSONA_PROFILES.keys()))
        
        # Reset state
        self.visitor_state = VisitorState.ENGAGED
        self.consecutive_explain_count = 0
        self.turns_without_question = 0
        self.turns_at_current_exhibit = 0
        self.current_exhibit_completion = 0.0
        self.turns_in_ready_to_move = 0
        self.recent_actions.clear()
        
        # Reset turn counter and new tracking
        self.turn_counter = 0
        self.turns_since_recovery = 999  # No immunity at start
        self.consecutive_lecture_turns = 0
        self.lecture_fatigue_penalty = 0.0
        
        # Reset new state tracking
        self.consecutive_same_topic_turns = 0
        self.last_fact_category = None
        self.consecutive_engaged_turns = 0
        self.unique_actions_in_engaged.clear()
        self.turns_since_highly_engaged = 0
        
        # Reset anti-spam tracking
        self.recovery_count = 0
        self.overload_episodes = 0
        self.transition_rejected = False
        self.transition_accepted = False
        self.transition_target = None
        self.mentioned_facts.clear()
        self.repeated_fact_count = 0
        
        # Reset exhausted exhibit tracking
        self.turns_at_exhausted_exhibit = 0
        self.engagement_boost_pending = 0.0
        
        # Reset content starvation tracking
        self.turns_without_new_fact = 0
        
        # Reset question spam prevention
        self.consecutive_questions = 0
        
        # Reset tracking
        self.aoi_usage_count.clear()
        self.seen_aois.clear()
        self.dialogue_history.clear()
        self.last_user_utterance = ""
        self.last_user_response = {}
        
        return {
            "persona": self.current_persona,
            "persona_profile": self.persona_profile,
            "exhibit": self.current_exhibit,
            "aoi": self.current_aoi,
        }
    
    def get_current_aoi(self) -> str:
        """Return current exhibit for env focus."""
        return self.current_exhibit or self.exhibits[0]
    
    def get_introduction_exchange(self, exhibit_name: str = None) -> Dict[str, str]:
        """Get scripted introduction exchange for episode start.
        
        Returns a fixed agent greeting and visitor response to establish
        dialogue context before RL-controlled turns begin.
        
        Args:
            exhibit_name: Name of the starting exhibit (uses current if None)
            
        Returns:
            Dict with 'agent_greeting' and 'user_response' keys
        """
        exhibit = exhibit_name or self.current_exhibit or "this artwork"
        # Format exhibit name for display (replace underscores with spaces)
        exhibit_display = exhibit.replace("_", " ")
        
        # Scripted agent greeting - warm, natural, inviting engagement
        agent_greeting = (
            f"Welcome! I'm delighted to be your guide today. "
            f"We're starting here at {exhibit_display}. "
            f"Take a moment to look at it - what catches your eye first?"
        )
        
        # Scripted visitor response - positive, curious, sets good tone
        user_response = (
            "Oh, this is lovely! I'm excited to learn more about it. "
            "There's something about the composition that draws me in."
        )
        
        return {
            "agent_greeting": agent_greeting,
            "user_response": user_response
        }
    
    def inject_introduction(self, exhibit_name: str = None) -> Dict[str, str]:
        """Inject introduction exchange into dialogue history.
        
        Call this after initialize_session() to populate dialogue history
        with the scripted introduction before RL turns begin.
        
        Args:
            exhibit_name: Name of the starting exhibit
            
        Returns:
            Dict with 'agent_greeting' and 'user_response' for env sync
        """
        intro = self.get_introduction_exchange(exhibit_name)
        
        # Add to dialogue history
        self.dialogue_history.append({"role": "agent", "text": intro["agent_greeting"]})
        self.dialogue_history.append({"role": "user", "text": intro["user_response"]})
        
        # Update last user utterance so Turn 1 has context
        self.last_user_utterance = intro["user_response"]
        
        return intro
    
    def generate_user_response(
        self,
        agent_utterance: str,
        agent_option: str = None,
        agent_subaction: str = None,
        target_exhibit: str = None,
        current_exhibit_completion: float = 0.0,
        exhibit_exhausted: bool = False,
        target_exhibit_completion: float = 0.0,
        target_exhibit_exhausted: bool = False
    ) -> Dict[str, Any]:
        """
        Generate visitor response based on state machine logic.
        
        Flow:
        1. Update action tracking
        2. Check state triggers (BEFORE response)
        3. Check recovery from agent action
        4. Generate dwell from state
        5. Generate utterance from state
        6. Return response
        """
        verbose = os.environ.get('HRL_VERBOSE', '0') == '1'
        
        # Track agent action
        self.last_agent_option = agent_option
        self.last_agent_subaction = agent_subaction
        self.recent_actions.append((agent_option, agent_subaction))
        self.current_exhibit_completion = current_exhibit_completion
        self.turns_at_current_exhibit += 1
        
        # Increment absolute turn counter
        self.turn_counter += 1
        
        # Track exhausted exhibit (100% completion)
        if exhibit_exhausted:
            self.turns_at_exhausted_exhibit += 1
        else:
            self.turns_at_exhausted_exhibit = 0
        
        # Track content starvation (turns without sharing new facts)
        if agent_subaction == "ExplainNewFact":
            self.turns_without_new_fact = 0
            self.consecutive_questions = 0  # Reset question spam counter
        else:
            self.turns_without_new_fact += 1
        
        # Track consecutive questions (for diminishing returns)
        if agent_option == "AskQuestion":
            self.consecutive_questions += 1
        
        # Update action counters
        if agent_option == "Explain" and agent_subaction == "ExplainNewFact":
            self.consecutive_explain_count += 1
        else:
            self.consecutive_explain_count = 0
        
        if agent_option == "AskQuestion":
            self.turns_without_question = 0
        else:
            self.turns_without_question += 1
        
        # Track lecture fatigue (Explain + Conclude are "passive" actions)
        # Note: OfferTransition is interactive, not passive
        if agent_option in ["Explain", "Conclude"]:
            self.consecutive_lecture_turns += 1
            # Each lecture turn adds 0.08 penalty, capped at 0.40
            self.lecture_fatigue_penalty = min(0.40, self.consecutive_lecture_turns * 0.08)
        elif agent_option == "AskQuestion":
            # AskQuestion RESTORES engagement and clears fatigue
            self.consecutive_lecture_turns = 0
            self.lecture_fatigue_penalty = 0.0
        
        # Increment recovery buffer counter
        self.turns_since_recovery += 1
        
        # Track topic for BORED_OF_TOPIC detection
        self._update_topic_tracking(agent_utterance)
        
        # Track fact repetition for engagement penalty
        self._track_fact_repetition(agent_utterance)
        
        # Reset transition flags
        self.transition_rejected = False
        self.transition_accepted = False
        self.transition_target = None
        
        # Store previous state for logging
        prev_state = self.visitor_state
        
        # Check for agent hallucination (fake fact IDs)
        hallucination_detected = self._check_agent_hallucination(agent_utterance)
        if hallucination_detected and verbose:
            print(f"[StateMachine] HALLUCINATION DETECTED in agent utterance")
        
        # === STEP 1: Check state triggers ===
        self._check_state_triggers(
            agent_option, agent_subaction, agent_utterance,
            current_exhibit_completion, exhibit_exhausted, verbose,
            hallucination_detected=hallucination_detected
        )
        
        # === STEP 2: Check recovery from agent action ===
        transition_success = self._process_recovery(
            agent_option, agent_subaction, agent_utterance,
            target_exhibit, current_exhibit_completion, verbose
        )
        
        # Log state transition
        if verbose and prev_state != self.visitor_state:
            print(f"[StateMachine] {prev_state.value} -> {self.visitor_state.value}")
        
        # === STEP 3: Generate dwell from state ===
        dwell_time = self._compute_dwell()
        gaze_features = self._synthesize_gaze(dwell_time)
        
        # === STEP 4: Generate utterance from state ===
        sim_start = time.time()
        utterance = self._generate_utterance(agent_utterance, agent_option, agent_subaction)
        self._last_sim_llm_time = time.time() - sim_start
        
        # === STEP 5: Map state to response type ===
        response_type = self._state_to_response_type()
        
        # Handle exhibit transition (check both OfferTransition and SuggestMove for coarse config)
        is_transition_attempt = agent_option == "OfferTransition" or agent_subaction == "SuggestMove"
        if is_transition_attempt and transition_success and target_exhibit:
            # Massive engagement boost for escaping exhausted exhibit
            if self.turns_at_exhausted_exhibit >= 3:
                self.engagement_boost_pending = 0.25
                if verbose:
                    print(f"[StateMachine] ESCAPE BOOST: +0.25 for leaving exhausted exhibit after {self.turns_at_exhausted_exhibit} turns")
            
            self.current_exhibit = target_exhibit
            self.current_aoi = target_exhibit
            self.turns_at_current_exhibit = 0
            self.turns_at_exhausted_exhibit = 0  # Reset exhausted counter
            # Reset exhibit-specific counters for fresh start at new exhibit
            self.consecutive_lecture_turns = 0
            self.lecture_fatigue_penalty = 0.0
            self.consecutive_same_topic_turns = 0
            self.last_fact_category = None
            if verbose:
                print(f"[StateMachine] Transitioned to: {target_exhibit}")
        
        # Build response
        response = {
            "utterance": utterance,
            "aoi": self.current_aoi,
            "persona": self.current_persona,
            "gaze_features": gaze_features,
            "response_type": response_type,
            "visitor_state": self.visitor_state.value,
            "engagement_level": dwell_time,
            "transition_success": transition_success,
            "simulator_llm_time": self._last_sim_llm_time,
        }
        
        self.last_user_response = response
        self.last_user_utterance = utterance
        self.dialogue_history.append({"role": "agent", "text": agent_utterance})
        self.dialogue_history.append({"role": "user", "text": utterance})
        
        return response
    
    def _get_effective_threshold(self, base_threshold: str) -> float:
        """Get threshold modified by persona profile."""
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
        hallucination_detected: bool = False
    ):
        """Check and trigger state transitions based on conditions.
        
        Order matters: deterministic triggers (OVERLOADED, FATIGUED, READY_TO_MOVE)
        are checked before random triggers (CURIOUS, CONFUSED) to ensure predictable
        behavior when thresholds are met.
        
        Args:
            hallucination_detected: If True, increases confusion probability to 60%
        """
        
        # Get effective thresholds based on persona profile
        overload_thresh = self._get_effective_threshold("overload")
        fatigue_thresh = self._get_effective_threshold("fatigue")
        curious_prob = self._get_effective_threshold("curious")
        ready_turns = self._get_effective_threshold("ready")
        
        # === HIGHLY_ENGAGED DECAY ===
        # If HIGHLY_ENGAGED, decay after 2 turns without varied stimulus
        if self.visitor_state == VisitorState.HIGHLY_ENGAGED:
            self.turns_since_highly_engaged += 1
            if self.turns_since_highly_engaged >= 2:
                self.visitor_state = VisitorState.ENGAGED
                self.turns_since_highly_engaged = 0
                if verbose:
                    print(f"[StateMachine] HIGHLY_ENGAGED decayed to ENGAGED")
            return
        
        # Only trigger new states from ENGAGED
        if self.visitor_state != VisitorState.ENGAGED:
            return
        
        # === POSITIVE ESCALATION: HIGHLY_ENGAGED ===
        # Track varied actions while engaged
        self.consecutive_engaged_turns += 1
        self.unique_actions_in_engaged.add(agent_option)
        
        # 3+ engaged turns with 2+ different action types -> HIGHLY_ENGAGED
        if (self.consecutive_engaged_turns >= self.HIGHLY_ENGAGED_THRESHOLD and 
            len(self.unique_actions_in_engaged) >= 2):
            self.visitor_state = VisitorState.HIGHLY_ENGAGED
            self.turns_since_highly_engaged = 0
            if verbose:
                print(f"[StateMachine] HIGHLY_ENGAGED triggered: {self.consecutive_engaged_turns} varied turns")
            return
        
        # === DETERMINISTIC TRIGGERS (checked first, in priority order) ===
        
        # 1. OVERLOADED: 3+ consecutive ExplainNewFact (highest priority - information overload)
        if self.consecutive_explain_count >= overload_thresh:
            self.visitor_state = VisitorState.OVERLOADED
            self.overload_episodes += 1  # Track for cumulative penalty
            self._reset_engaged_tracking()
            if verbose:
                print(f"[StateMachine] OVERLOADED triggered: {self.consecutive_explain_count} consecutive explains (episode {self.overload_episodes})")
            return
        
        # 2. FATIGUED: 3+ turns without AskQuestion (monotony detection)
        if self.turns_without_question >= fatigue_thresh:
            self.visitor_state = VisitorState.FATIGUED
            self._reset_engaged_tracking()
            if verbose:
                print(f"[StateMachine] FATIGUED triggered: {self.turns_without_question} turns without question")
            return
        
        # 3. BORED_OF_TOPIC: 3+ turns on same topic/fact category
        if self.consecutive_same_topic_turns >= self.BORED_TOPIC_THRESHOLD:
            self.visitor_state = VisitorState.BORED_OF_TOPIC
            self._reset_engaged_tracking()
            if verbose:
                print(f"[StateMachine] BORED_OF_TOPIC triggered: {self.consecutive_same_topic_turns} same topic turns")
            return
        
        # 4. READY_TO_MOVE: 80%+ coverage AND 4+ turns (exhibit exhaustion)
        if current_completion >= self.READY_COVERAGE and self.turns_at_current_exhibit >= ready_turns:
            self.visitor_state = VisitorState.READY_TO_MOVE
            self._reset_engaged_tracking()
            if verbose:
                print(f"[StateMachine] READY_TO_MOVE triggered: {current_completion:.0%} coverage, {self.turns_at_current_exhibit} turns")
            return
        
        # === RANDOM TRIGGERS (checked after deterministic triggers) ===
        
        # 5. CURIOUS: 15% random chance (visitor asks question)
        # Protected: First 3 turns are immune to prevent early CURIOUS→CONFUSED loops
        if self.turn_counter > 3 and self.rng.random() < curious_prob:
            self.visitor_state = VisitorState.CURIOUS
            self._reset_engaged_tracking()
            if verbose:
                print(f"[StateMachine] CURIOUS triggered (random)")
            return
        
        # 6. CONFUSED: 10% base rate (visitor didn't understand)
        # Protected: Turn 1 is immune, and 2 turns after recovery are immune
        # Hallucination increases confusion probability to 60%
        if self.turn_counter > 1 and self.turns_since_recovery >= 2:
            confusion_prob = 0.60 if hallucination_detected else self.CONFUSED_PROBABILITY
            if self.rng.random() < confusion_prob:
                self.visitor_state = VisitorState.CONFUSED
                self._reset_engaged_tracking()
                if verbose:
                    trigger_reason = "hallucination" if hallucination_detected else "random"
                    print(f"[StateMachine] CONFUSED triggered ({trigger_reason})")
    
    def _reset_engaged_tracking(self):
        """Reset tracking for HIGHLY_ENGAGED when leaving ENGAGED."""
        self.consecutive_engaged_turns = 0
        self.unique_actions_in_engaged.clear()
    
    def _extract_fact_category(self, utterance: str) -> Optional[str]:
        """Extract fact category from utterance (e.g., 'HI' from '[HI_001]').
        
        Fact IDs follow format: [XX_NNN] where XX is category code.
        """
        if not utterance:
            return None
        match = re.search(r'\[([A-Z]{2})_\d{3}\]', utterance)
        return match.group(1) if match else None
    
    def _update_topic_tracking(self, agent_utterance: str):
        """Track consecutive turns on same topic for BORED_OF_TOPIC detection."""
        current_category = self._extract_fact_category(agent_utterance)
        
        if current_category:
            if current_category == self.last_fact_category:
                self.consecutive_same_topic_turns += 1
            else:
                self.consecutive_same_topic_turns = 1
                self.last_fact_category = current_category
        # If no fact in utterance, don't reset (questions, transitions don't count)
    
    def _track_fact_repetition(self, agent_utterance: str):
        """Track fact repetition for engagement penalty.
        
        When agent repeats already-mentioned facts, increment repetition counter.
        New facts reset the counter.
        """
        if not agent_utterance:
            return
        
        # Find all fact IDs in utterance
        current_facts = re.findall(r'\[([A-Z]{2}_\d{3})\]', agent_utterance)
        
        if not current_facts:
            return  # No facts in this utterance
        
        # Check each fact - if ANY is new, reset counter
        any_new = False
        for fact in current_facts:
            if fact not in self.mentioned_facts:
                any_new = True
                self.mentioned_facts.add(fact)
        
        if any_new:
            self.repeated_fact_count = 0
        else:
            # All facts were already mentioned = repetition
            self.repeated_fact_count += 1
    
    def _check_agent_hallucination(self, utterance: str) -> bool:
        """Check if agent mentioned fact IDs not in knowledge base.
        
        This detects when the LLM agent hallucinates fake fact IDs,
        which causes visitor confusion because the facts don't exist.
        
        Returns:
            True if hallucination detected (invalid fact ID found)
        """
        if not utterance:
            return False
        
        # Find all fact IDs in utterance (format: [XX_NNN])
        mentioned = re.findall(r'\[([A-Z]{2}_\d{3})\]', utterance)
        if not mentioned:
            return False  # No fact IDs mentioned
        
        # Get valid fact IDs from knowledge graph
        if self.knowledge_graph:
            valid_facts = self.knowledge_graph.get_all_fact_ids()
            return any(f not in valid_facts for f in mentioned)
        
        return False  # Can't validate without knowledge graph
    
    def _process_recovery(
        self,
        agent_option: str,
        agent_subaction: str,
        agent_utterance: str,
        target_exhibit: str,
        current_completion: float,
        verbose: bool
    ) -> bool:
        """
        Process agent action for potential state recovery.
        
        Also handles Sim8-style transition probability in ENGAGED state:
        - Low completion (0-20%): 20% acceptance
        - Medium (20-40%): 50% acceptance  
        - Good (40-60%): 80% acceptance
        - High (60%+): 95% acceptance
        
        Applies recovery fatigue: each recovery this session reduces success rate by 15%,
        capped at 45% reduction (3 recoveries). This prevents spam-recover-repeat cycles.
        
        Returns:
            transition_success: Whether a transition offer was accepted
        """
        transition_success = False
        
        # Check if this is a transition attempt (handles coarse config where SuggestMove is under Engage)
        is_transition_attempt = agent_option == "OfferTransition" or agent_subaction == "SuggestMove"
        
        # === HIGHLY_ENGAGED STATE: High acceptance for transitions ===
        if self.visitor_state == VisitorState.HIGHLY_ENGAGED and is_transition_attempt:
            # Highly engaged visitors are receptive to fresh content
            transition_prob = 0.85 if current_completion >= 0.40 else 0.50
            
            if self.rng.random() < transition_prob:
                transition_success = True
                self.transition_accepted = True
                self.transition_target = target_exhibit
                if verbose:
                    print(f"[StateMachine] TRANSITION ACCEPTED in HIGHLY_ENGAGED state (prob={transition_prob:.0%})")
            else:
                self.transition_rejected = True
                if verbose:
                    print(f"[StateMachine] TRANSITION REJECTED: Highly engaged visitor wants more")
            return transition_success
        
        # === ENGAGED STATE: Sim8-style transition probability ===
        if self.visitor_state == VisitorState.ENGAGED and is_transition_attempt:
            # Calculate probability based on exhibit completion (Sim8-style)
            if current_completion < 0.20:
                transition_prob = 0.20  # Too early - visitor wants to learn more
            elif current_completion < 0.40:
                transition_prob = 0.50  # Some coverage, might accept
            elif current_completion < 0.60:
                transition_prob = 0.80  # Good coverage, likely to accept
            else:
                transition_prob = 0.95  # Well covered, almost always accept
            
            if self.rng.random() < transition_prob:
                transition_success = True
                self.transition_accepted = True
                self.transition_target = target_exhibit
                if verbose:
                    print(f"[StateMachine] TRANSITION ACCEPTED in ENGAGED state (prob={transition_prob:.0%}, coverage={current_completion:.0%})")
            else:
                # REJECTED: visitor wants to stay and learn more
                self.visitor_state = VisitorState.CURIOUS
                self.transition_rejected = True  # Flag for response generation
                if verbose:
                    print(f"[StateMachine] TRANSITION REJECTED: Visitor wants to stay (prob={transition_prob:.0%}, coverage={current_completion:.0%})")
            return transition_success
        
        if self.visitor_state == VisitorState.ENGAGED:
            return transition_success
        
        # Apply recovery fatigue (15% reduction per recovery, max 45%)
        fatigue_penalty = min(0.45, 0.15 * self.recovery_count)
        
        # Get base recovery rates and apply fatigue
        base_rates = self.RECOVERY_RATES.get(self.visitor_state, {})
        recovery_rates = {k: max(0.10, v - fatigue_penalty) for k, v in base_rates.items()}
        
        # Track if we recovered this turn (to increment fatigue counter)
        recovered = False
        prev_state = self.visitor_state
        
        # === CONFUSED: ClarifyFact, AskClarification, OfferTransition, or ExplainNewFact recovers ===
        if self.visitor_state == VisitorState.CONFUSED:
            if agent_subaction == "ClarifyFact":
                if self.rng.random() < recovery_rates.get("ClarifyFact", 0.90):
                    self.visitor_state = VisitorState.ENGAGED
                    self.turns_since_recovery = 0  # Reset recovery buffer
                    recovered = True
                    if verbose:
                        print(f"[StateMachine] RECOVERY: ClarifyFact resolved confusion (fatigue: {self.recovery_count})")
            elif agent_subaction == "AskClarification":
                if self.rng.random() < recovery_rates.get("AskClarification", 0.70):
                    self.visitor_state = VisitorState.ENGAGED
                    self.turns_since_recovery = 0  # Reset recovery buffer
                    recovered = True
                    if verbose:
                        print(f"[StateMachine] RECOVERY: AskClarification resolved confusion")
            elif is_transition_attempt:
                # Moving to a new exhibit can help confused visitor reset
                if self.rng.random() < recovery_rates.get("OfferTransition", 0.60):
                    self.visitor_state = VisitorState.ENGAGED
                    self.turns_since_recovery = 0
                    transition_success = True
                    self.transition_accepted = True
                    self.transition_target = target_exhibit
                    recovered = True
                    if verbose:
                        print(f"[StateMachine] RECOVERY: Transition helped confused visitor reset")
            elif agent_option == "Explain" and agent_subaction == "ExplainNewFact":
                # Explaining a new fact clearly can sometimes help confusion
                if self.rng.random() < recovery_rates.get("ExplainNewFact", 0.40):
                    self.visitor_state = VisitorState.ENGAGED
                    self.turns_since_recovery = 0
                    recovered = True
                    if verbose:
                        print(f"[StateMachine] RECOVERY: New fact explanation helped confusion")
        
        # === OVERLOADED: AskQuestion or Transition recovers ===
        elif self.visitor_state == VisitorState.OVERLOADED:
            if agent_option == "AskQuestion":
                if self.rng.random() < recovery_rates.get("AskQuestion", 0.85):
                    self.visitor_state = VisitorState.ENGAGED
                    self.consecutive_explain_count = 0
                    self.turns_since_recovery = 0
                    recovered = True
                    if verbose:
                        print(f"[StateMachine] RECOVERY: AskQuestion resolved overload (fatigue: {self.recovery_count})")
            elif is_transition_attempt:
                if self.rng.random() < recovery_rates.get("OfferTransition", 0.75):
                    self.visitor_state = VisitorState.ENGAGED
                    self.consecutive_explain_count = 0
                    self.turns_since_recovery = 0
                    transition_success = True
                    self.transition_accepted = True
                    self.transition_target = target_exhibit
                    recovered = True
                    if verbose:
                        print(f"[StateMachine] RECOVERY: Transition resolved overload (fatigue: {self.recovery_count})")
        
        # === CURIOUS: Explain with fact recovers ===
        elif self.visitor_state == VisitorState.CURIOUS:
            if agent_option == "Explain":
                # Check if agent provided a fact (has fact ID in utterance)
                has_fact = bool(re.search(r'\[[A-Z]{2}_\d{3}\]', agent_utterance or ""))
                if has_fact and self.rng.random() < recovery_rates.get("Explain", 0.90):
                    self.visitor_state = VisitorState.ENGAGED
                    self.turns_since_recovery = 0
                    recovered = True
                    if verbose:
                        print(f"[StateMachine] RECOVERY: Explain with fact answered question")
            elif agent_option == "AskQuestion":
                # Deflection - 50% chance instead of 100% (gives agent another chance)
                if self.rng.random() < 0.50:
                    self.visitor_state = VisitorState.CONFUSED
                    if verbose:
                        print(f"[StateMachine] DEFLECTION: AskQuestion when CURIOUS -> CONFUSED")
                else:
                    # Lucky! Stay CURIOUS, give agent another chance
                    if verbose:
                        print(f"[StateMachine] CURIOUS visitor tolerates question (50% luck)")
        
        # === BORED_OF_TOPIC: Explain with different topic recovers ===
        elif self.visitor_state == VisitorState.BORED_OF_TOPIC:
            if agent_option == "Explain":
                # Check if this is a different fact category
                current_fact = self._extract_fact_category(agent_utterance)
                is_different_topic = current_fact and current_fact != self.last_fact_category
                
                # Apply persona recovery modifier
                profile = self.PERSONA_PROFILES.get(self.persona_profile, {})
                recovery_mod = profile.get("recovery_modifier", 1.0)
                base_rate = recovery_rates.get("Explain", 0.85)
                
                if is_different_topic and self.rng.random() < base_rate * recovery_mod:
                    self.visitor_state = VisitorState.ENGAGED
                    self.consecutive_same_topic_turns = 0
                    self.last_fact_category = current_fact
                    self.turns_since_recovery = 0
                    recovered = True
                    if verbose:
                        print(f"[StateMachine] RECOVERY: Different topic resolved boredom (fatigue: {self.recovery_count})")
                elif not is_different_topic:
                    if verbose:
                        print(f"[StateMachine] FAILED RECOVERY: Same topic doesn't help boredom")
            elif is_transition_attempt:
                # Transition also works for boredom (apply fatigue to this rate too)
                base_trans_rate = self.RECOVERY_RATES.get(VisitorState.BORED_OF_TOPIC, {}).get("OfferTransition", 0.90)
                trans_rate = max(0.10, base_trans_rate - fatigue_penalty)
                if self.rng.random() < trans_rate:
                    self.visitor_state = VisitorState.ENGAGED
                    self.consecutive_same_topic_turns = 0
                    self.turns_since_recovery = 0
                    transition_success = True
                    self.transition_accepted = True
                    self.transition_target = target_exhibit
                    recovered = True
                    if verbose:
                        print(f"[StateMachine] RECOVERY: Transition resolved boredom (fatigue: {self.recovery_count})")
        
        # === FATIGUED: AskQuestion or Transition recovers ===
        elif self.visitor_state == VisitorState.FATIGUED:
            if agent_option == "AskQuestion":
                if self.rng.random() < recovery_rates.get("AskQuestion", 0.80):
                    self.visitor_state = VisitorState.ENGAGED
                    self.turns_without_question = 0
                    self.turns_since_recovery = 0
                    recovered = True
                    if verbose:
                        print(f"[StateMachine] RECOVERY: AskQuestion resolved fatigue (fatigue: {self.recovery_count})")
            elif is_transition_attempt:
                if self.rng.random() < recovery_rates.get("OfferTransition", 0.85):
                    self.visitor_state = VisitorState.ENGAGED
                    self.turns_since_recovery = 0
                    transition_success = True
                    self.transition_accepted = True
                    self.transition_target = target_exhibit
                    recovered = True
                    if verbose:
                        print(f"[StateMachine] RECOVERY: Transition resolved fatigue (fatigue: {self.recovery_count})")
        
        # === READY_TO_MOVE: Transition recovers, escalates to DISENGAGED ===
        elif self.visitor_state == VisitorState.READY_TO_MOVE:
            self.turns_in_ready_to_move += 1
            
            if is_transition_attempt:
                if self.rng.random() < recovery_rates.get("OfferTransition", 0.95):
                    self.visitor_state = VisitorState.ENGAGED
                    self.turns_in_ready_to_move = 0
                    self.turns_since_recovery = 0
                    transition_success = True
                    self.transition_accepted = True
                    self.transition_target = target_exhibit
                    recovered = True
                    if verbose:
                        print(f"[StateMachine] RECOVERY: Transition satisfied ready-to-move")
            elif agent_option == "Explain":
                # ALL explain subactions are wrong when visitor wants to leave
                self.visitor_state = VisitorState.FATIGUED
                if verbose:
                    print(f"[StateMachine] WRONG ACTION: {agent_subaction} when READY_TO_MOVE -> FATIGUED")
            # Escalate to DISENGAGED after 3 turns of ignoring (only if not already changed)
            elif self.turns_in_ready_to_move >= 3:
                self.visitor_state = VisitorState.DISENGAGED
                if verbose:
                    print(f"[StateMachine] ESCALATION: {self.turns_in_ready_to_move} turns ignoring READY_TO_MOVE -> DISENGAGED")
        
        # === DISENGAGED: Only Transition can recover (50% success) ===
        elif self.visitor_state == VisitorState.DISENGAGED:
            if is_transition_attempt:
                if self.rng.random() < recovery_rates.get("OfferTransition", 0.50):
                    self.visitor_state = VisitorState.ENGAGED
                    self.turns_in_ready_to_move = 0
                    self.turns_since_recovery = 0
                    transition_success = True
                    self.transition_accepted = True
                    self.transition_target = target_exhibit
                    recovered = True
                    if verbose:
                        print(f"[StateMachine] RECOVERY: Transition rescued disengaged visitor (fatigue: {self.recovery_count})")
                else:
                    self.transition_rejected = True
                    if verbose:
                        print(f"[StateMachine] FAILED RECOVERY: Transition rejected by disengaged visitor")
            # All other actions keep visitor disengaged (no state change)
        
        # Increment recovery fatigue counter if we recovered
        if recovered:
            self.recovery_count += 1
            self._reset_engaged_tracking()  # Fresh start when returning to ENGAGED
            if verbose and self.recovery_count > 0:
                effective_penalty = min(0.45, 0.15 * self.recovery_count)
                print(f"[StateMachine] Recovery fatigue now at {self.recovery_count} (-{effective_penalty:.0%} to future recoveries)")
        
        return transition_success
    
    def _compute_dwell(self) -> float:
        """Compute dwell time from current state (deterministic range).
        
        Applies multiple penalties/boosts to prevent spam and encourage variety:
        1. Explain ratio penalty: 0.90x multiplier if >60% of last 5 actions are ExplainNewFact
        2. Cumulative overload penalty: -0.05 per OVERLOADED episode this session
        3. READY_TO_MOVE escalation: -0.10 per turn ignoring ready state
        4. Fact repetition penalty: -0.10 per consecutive repetition of already-mentioned facts
        5. Topic staleness decay: -0.05 per turn after 8+ turns at same exhibit
        6. Lecture fatigue penalty: -0.08 per consecutive passive action (capped at -0.40)
        7. AskQuestion boost: +0.30 when used during fatigue/overload (aggressive)
        7b. AskOpinion boost: +0.25 when used during fatigue/overload
        8. ExplainNewFact boost: +0.20 when used after 2+ passive turns (fresh content helps)
        9. Exhausted exhibit penalty: -0.08 per turn after 5+ turns at exhausted exhibit (capped at -0.35)
        10. Transition escape boost: +0.25 for successfully leaving exhausted exhibit
        11. Content starvation penalty: -0.10 per turn after 3+ turns without ExplainNewFact (capped at -0.50)
        """
        low, high = DWELL_RANGES[self.visitor_state]
        base_dwell = self.rng.uniform(low, high)
        
        # 1. Explain ratio penalty (applies even in ENGAGED state)
        if len(self.recent_actions) >= 5:
            explain_count = sum(1 for opt, sub in list(self.recent_actions)[-5:] 
                               if opt == "Explain" and sub == "ExplainNewFact")
            if explain_count >= 3:  # >60% of last 5 actions
                base_dwell *= 0.90
        
        # 2. Cumulative overload penalty (escalating cost for repeated overload)
        if self.overload_episodes > 0:
            overload_penalty = 0.05 * self.overload_episodes
            base_dwell = max(0.15, base_dwell - overload_penalty)
        
        # 3. Escalating penalty for ignoring ready-to-move
        if self.visitor_state == VisitorState.READY_TO_MOVE:
            penalty = 0.10 * self.turns_in_ready_to_move
            base_dwell = max(0.10, base_dwell - penalty)
        
        # 4. Fact repetition penalty (agent repeating same facts is boring)
        if self.repeated_fact_count >= 2:
            repetition_penalty = 0.10 * self.repeated_fact_count
            base_dwell = max(0.20, base_dwell - repetition_penalty)
        
        # 5. Topic staleness decay (too long at same exhibit without new content)
        if self.turns_at_current_exhibit >= 8 and self.visitor_state == VisitorState.ENGAGED:
            staleness_penalty = 0.05 * (self.turns_at_current_exhibit - 7)
            base_dwell = max(0.30, base_dwell - staleness_penalty)
        
        # 6. Lecture fatigue penalty (consecutive passive actions without questions)
        # This implements the "engagement drops during lectures" behavior
        if self.lecture_fatigue_penalty > 0:
            base_dwell = max(0.25, base_dwell - self.lecture_fatigue_penalty)
        
        # 7. AskQuestion engagement boost with DIMINISHING RETURNS
        # Sharp falloff to prevent question spam: 1st full boost, 2nd reduced, 3rd+ penalty
        # This enforces: Question(s) → ExplainNewFact(s) → Fatigue → Question(s) → repeat
        if (self.last_agent_option == "AskQuestion" and 
            self.visitor_state in [VisitorState.FATIGUED, VisitorState.OVERLOADED, VisitorState.ENGAGED]):
            if self.consecutive_questions == 1:
                base_dwell = min(1.0, base_dwell + 0.30)  # Full boost
            elif self.consecutive_questions == 2:
                base_dwell = min(1.0, base_dwell + 0.10)  # Reduced boost
            else:  # 3+ consecutive questions = penalty
                base_dwell = max(0.0, base_dwell - 0.25)  # Question spam penalty
        
        # 7b. AskOpinion also boosts engagement (same diminishing returns as AskQuestion)
        if (self.last_agent_subaction == "AskOpinion" and 
            self.visitor_state in [VisitorState.FATIGUED, VisitorState.OVERLOADED, VisitorState.ENGAGED]):
            if self.consecutive_questions == 1:
                base_dwell = min(1.0, base_dwell + 0.25)  # Full boost (slightly less than AskQuestion)
            elif self.consecutive_questions == 2:
                base_dwell = min(1.0, base_dwell + 0.08)  # Reduced boost
            else:  # 3+ consecutive = penalty
                base_dwell = max(0.0, base_dwell - 0.20)  # Question spam penalty
        
        # 8. ExplainNewFact engagement boost after slump
        # Fresh factual content can help restore engagement after passive period
        if (self.last_agent_subaction == "ExplainNewFact" and 
            self.lecture_fatigue_penalty >= 0.16):  # At least 2 consecutive passive turns
            base_dwell = min(0.80, base_dwell + 0.20)  # +0.20 engagement boost
        
        # 9. EXHAUSTED EXHIBIT PENALTY - staying too long at completed exhibit
        # Massive penalty kicks in after 5+ turns at exhausted exhibit
        if self.turns_at_exhausted_exhibit >= 5:
            exhaustion_penalty = 0.08 * (self.turns_at_exhausted_exhibit - 4)  # -0.08, -0.16, -0.24...
            exhaustion_penalty = min(0.35, exhaustion_penalty)  # Cap at -0.35
            base_dwell = max(0.10, base_dwell - exhaustion_penalty)
        
        # 10. Transition escape boost - successfully leaving exhausted exhibit
        if self.engagement_boost_pending > 0:
            base_dwell = min(0.90, base_dwell + self.engagement_boost_pending)
            self.engagement_boost_pending = 0  # Consume the boost
        
        # 11. Content starvation penalty - must share facts regularly
        # Agent can't game engagement by spamming AskQuestion without teaching
        if self.turns_without_new_fact >= 3:
            starvation_penalty = 0.10 * (self.turns_without_new_fact - 2)  # -0.10, -0.20, -0.30...
            starvation_penalty = min(0.5, starvation_penalty)  # Cap at -0.50 (halves engagement)
            base_dwell = max(0.0, base_dwell - starvation_penalty)
        
        # FINAL SAFETY CLAMP: Ensure dwell is never negative (minimum is 0.0)
        return max(0.0, base_dwell)
    
    def _synthesize_gaze(self, dwell_time: float) -> List[float]:
        """Generate gaze features with dwell as first element."""
        # Simplified gaze features - dwell is the primary signal
        saccade_span = max(0.05, self.np_rng.normal(0.08, 0.03))
        gaze_entropy = max(0.0, min(2.5, self.np_rng.normal(0.8, 0.4)))
        fix_change_rate = max(0.2, min(4.0, self.np_rng.normal(2.2, 0.8)))
        dom_ratio = dwell_time * self.rng.uniform(0.6, 0.95)
        entry_latency = max(0.1, min(12.0, self.np_rng.normal(4.0, 3.0)))
        
        return [
            float(dwell_time),
            float(saccade_span),
            float(gaze_entropy),
            float(fix_change_rate),
            float(dom_ratio),
            float(entry_latency),
        ]
    
    def _generate_utterance(self, agent_utterance: str, agent_option: Optional[str] = None, agent_subaction: Optional[str] = None) -> str:
        """Generate visitor utterance using LLM or fallback templates."""
        
        # Use compositional templates if template mode is enabled
        if os.environ.get('HRL_TEMPLATE_MODE') == '1':
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
                target_exhibit=self.transition_target
            )
        
        # Special case: transition was rejected - visitor wants to stay (non-template mode)
        if self.transition_rejected:
            rejection_responses = [
                "Wait, I want to learn more about this first.",
                "Hold on, can you tell me more before we move?",
                "Actually, I'm still curious about this.",
                "I'd like to stay here a bit longer.",
                "Not yet, I have more questions about this one.",
            ]
            return self.rng.choice(rejection_responses)
        
        # Try LLM first
        if os.environ.get('HRL_FAST_MODE') != '1':
            try:
                utterance = self._generate_llm_utterance(agent_utterance)
                if utterance:
                    return utterance
            except Exception as e:
                if os.environ.get('HRL_VERBOSE') == '1':
                    print(f"[StateMachine] LLM failed, using fallback: {e}")
        
        # Fallback to templates
        templates = FALLBACK_TEMPLATES.get(self.visitor_state, FALLBACK_TEMPLATES[VisitorState.ENGAGED])
        return self.rng.choice(templates)
    
    def _generate_llm_utterance(self, agent_utterance: str) -> Optional[str]:
        """Generate utterance using LLM with state-specific, context-aware prompt."""
        try:
            from LLM_CONFIG import get_simulator_llm
            
            if self._llm is None:
                self._llm = get_simulator_llm()
            
            # Build context from recent dialogue history
            history_context = ""
            if len(self.dialogue_history) >= 2:
                recent = self.dialogue_history[-4:]  # Last 2 exchanges
                history_lines = []
                for turn in recent:
                    role = "Guide" if turn["role"] == "agent" else "Visitor"
                    history_lines.append(f"{role}: {turn['text']}")
                history_context = "Recent conversation:\n" + "\n".join(history_lines)
            
            # Get exhibit name for context
            exhibit_name = self.current_exhibit or "the artwork"
            
            prompt_template = STATE_PROMPTS.get(self.visitor_state, STATE_PROMPTS[VisitorState.ENGAGED])
            prompt = prompt_template.format(
                agent_utterance=agent_utterance[:200],
                exhibit_name=exhibit_name,
                history_context=history_context
            )
            
            response = self._llm.generate(
                prompt,
                system_prompt="You are a museum visitor. Generate ONLY ONE short response (1-2 sentences max). No lists, no multiple statements, no imagining future dialogue. Just one natural reply."
            )
            
            # Clean up response
            if response:
                response = response.strip().strip('"').strip("'")
                # Remove common prefixes
                for prefix in ["Visitor:", "Response:", "Answer:"]:
                    if response.startswith(prefix):
                        response = response[len(prefix):].strip()
                
                # CRITICAL: Take only the FIRST line/sentence to prevent LLM rambling
                # Split on newlines first
                lines = response.split('\n')
                response = lines[0].strip()
                
                # If still too long (multiple sentences), take first 1-2 sentences
                # Split on sentence-ending punctuation
                sentences = re.split(r'(?<=[.!?])\s+', response)
                if len(sentences) > 2:
                    response = ' '.join(sentences[:2])
                
                # Hard cap at 200 chars as final safety net
                if len(response) > 200:
                    response = response[:200].rsplit(' ', 1)[0]  # Don't cut mid-word
                
                return response
            
        except Exception as e:
            if os.environ.get('HRL_VERBOSE') == '1':
                print(f"[StateMachine] LLM error: {e}")
        
        return None
    
    def _state_to_response_type(self) -> str:
        """Map visitor state to response type for compatibility."""
        mapping = {
            VisitorState.HIGHLY_ENGAGED: "acknowledgment",
            VisitorState.ENGAGED: "acknowledgment",
            VisitorState.CONFUSED: "confusion",
            VisitorState.OVERLOADED: "statement",
            VisitorState.CURIOUS: "question",
            VisitorState.BORED_OF_TOPIC: "question",  # Asks for different content
            VisitorState.FATIGUED: "statement",
            VisitorState.READY_TO_MOVE: "statement",
            VisitorState.DISENGAGED: "statement",
        }
        return mapping.get(self.visitor_state, "statement")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current simulator state for logging."""
        return {
            "aoi": self.current_aoi,
            "current_exhibit": self.current_exhibit,
            "persona": self.current_persona,
            "visitor_state": self.visitor_state.value,
            "consecutive_explain_count": self.consecutive_explain_count,
            "turns_without_question": self.turns_without_question,
            "turns_at_current_exhibit": self.turns_at_current_exhibit,
            "seen_aois": list(self.seen_aois),
            "last_user_response": dict(self.last_user_response) if self.last_user_response else {},
        }
    
    def update_from_state(self, state_focus: int, target_exhibit: str = None):
        """Update simulator state from environment (for transitions)."""
        if target_exhibit and target_exhibit != self.current_exhibit:
            if target_exhibit in self.exhibits:
                self.current_exhibit = target_exhibit
                self.current_aoi = target_exhibit
                self.turns_at_current_exhibit = 0
                self.turns_in_ready_to_move = 0
                self.visitor_state = VisitorState.ENGAGED

