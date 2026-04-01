# Hybrid Simulator Architecture

A layered visitor simulator that combines the state machine's discrete behavioral backbone with sim8's continuous engagement dynamics to produce cleaner, richer training signals for the RL agent.

## Table of Contents

1. [Motivation](#motivation)
2. [Comparative Analysis](#comparative-analysis)
3. [Architecture Overview](#architecture-overview)
4. [Layer 1: State Machine Backbone](#layer-1-state-machine-backbone)
5. [Layer 2: Engagement Dynamics](#layer-2-engagement-dynamics)
6. [Layer 3: Hybrid Dwell Computation](#layer-3-hybrid-dwell-computation)
7. [Layer 4: Transition Model](#layer-4-transition-model)
8. [Layer 5: Response Generation](#layer-5-response-generation)
9. [Persona System](#persona-system)
10. [The Stochasticity Parameter](#the-stochasticity-parameter)
11. [Anti-Gaming Mechanisms](#anti-gaming-mechanisms)
12. [Integration](#integration)
13. [Expected Training Improvements](#expected-training-improvements)

---

## Motivation

The two existing simulators are complementary, not redundant:

- **State machine** answers: *"What state is the visitor in?"* (the band)
- **Sim8** answers: *"How well is the agent performing within that state?"* (the position within the band)

Neither alone produces an optimal training signal. The state machine gives clean reward bands but treats all turns within a state equally. Sim8 gives rich within-turn engagement nuance but its overlapping dwell ranges create noisy learning signals. A hybrid that layers sim8's continuous engagement dynamics on top of state machine's discrete state backbone addresses both weaknesses simultaneously.

### Goals

1. **Better training signal**: Non-overlapping state bands (from state machine) with engagement-driven positioning within bands (from sim8)
2. **Harder to game**: Double anti-gaming layers from both simulators compound
3. **Ablation control**: A single `stochasticity` parameter (0.0-1.0) interpolates between pure state machine and maximum sim8 influence
4. **Backward compatible**: At `stochasticity=0.0`, reverts to pure state machine behavior

---

## Comparative Analysis

### What Sim8 Does Well (for training signal)

| Feature | Training benefit |
|---------|-----------------|
| Continuous engagement tracking (0.0-1.0) | Smooth gradient signal; agent learns incremental improvements |
| Multiplicative cascades (question spam x transition spam x stagnation) | Multiple anti-gaming layers compound |
| Novelty boost for `[FACT_ID]` sharing | Explicit incentive for information coverage |
| Question spacing bonus (x1.6) | Teaches proper pacing without hard thresholds |
| Persona variance (Agreeable/Conscientious/Neurotic) | Prevents overfitting to one visitor type |

### What Sim8 Does Poorly (for training signal)

| Feature | Training cost |
|---------|--------------|
| Overlapping dwell ranges across response types | Noisy signal: acknowledgment (0.75-0.95) overlaps with question (0.40-0.70) |
| No explicit visitor states | Agent cannot learn "visitor is confused, do X to recover" |
| No recovery mechanics | No incentive to learn rescue strategies |
| Response type determination is a complex cascade | Hard to trace why a particular reward was given |

### What State Machine Does Well (for training signal)

| Feature | Training benefit |
|---------|-----------------|
| Non-overlapping dwell ranges | Clean signal: ENGAGED (0.75-0.90) vs CONFUSED (0.20-0.35), no ambiguity |
| Deterministic triggers | Clear cause-effect: 3x ExplainNewFact -> OVERLOADED (learnable) |
| Recovery mechanics with fatigue | Agent learns to rescue disengaged visitors; fatigue prevents gaming |
| 11 penalty/boost components | Rich, interpretable reward shaping |
| Escalation paths (READY_TO_MOVE -> DISENGAGED) | Teaches urgency: ignoring signals has consequences |

### What State Machine Does Poorly (for training signal)

| Feature | Training cost |
|---------|--------------|
| Uniform random within bands | All ENGAGED turns feel the same (0.75-0.90 uniform): no within-state nuance |
| No continuous engagement momentum | Jumping between states is abrupt; real visitors have gradual engagement shifts |
| No novelty boost for new facts | Agent has no explicit reward for sharing information beyond coverage |
| Predictable thresholds | Agent can learn exact boundaries and game them (e.g., explain exactly 2x then ask) |

### Verdict

The two simulators address each other's gaps. A hybrid that preserves the state machine's clean reward bands while adding sim8's continuous engagement dynamics within those bands produces a strictly better training signal than either alone.

---

## Architecture Overview

```
+-------------------------------------------------------------+
|                    HYBRID SIMULATOR                           |
|                                                               |
|  +-------------------------------------------------------+   |
|  |  Layer 1: STATE MACHINE BACKBONE                       |   |
|  |  9 visitor states with deterministic triggers           |   |
|  |  ENGAGED --> OVERLOADED (3+ explains)                   |   |
|  |  ENGAGED --> FATIGUED   (3+ turns no question)          |   |
|  |  ENGAGED --> CURIOUS    (random, persona-modified)      |   |
|  |  ENGAGED --> HIGHLY_ENGAGED (3+ varied action turns)    |   |
|  |  Recovery: CONFUSED --> ENGAGED (ClarifyFact = 90%)     |   |
|  +-------------------------------------------------------+   |
|                          | state determines band              |
|                          v                                    |
|  +-------------------------------------------------------+   |
|  |  Layer 2: ENGAGEMENT DYNAMICS (from sim8)               |   |
|  |  Continuous engagement_level: 0.0-1.0                   |   |
|  |  Positions reward WITHIN the state band                 |   |
|  |  Multiplicative cascades:                               |   |
|  |    x question_spam_multiplier                           |   |
|  |    x transition_spam_multiplier                         |   |
|  |    x dwell_stagnation_multiplier                        |   |
|  |  Novelty boost: +0.05-0.15 for [FACT_ID]               |   |
|  |  Question spacing bonus: x1.6                           |   |
|  +-------------------------------------------------------+   |
|                          | engagement positions in band       |
|                          v                                    |
|  +-------------------------------------------------------+   |
|  |  Layer 3: HYBRID DWELL COMPUTATION                      |   |
|  |                                                         |   |
|  |  base = state_low + engagement_level x (high - low)     |   |
|  |  base += state_machine_penalties()    # 11 components   |   |
|  |  base *= sim8_multiplier_cascade()    # spam/pacing     |   |
|  |  base += novelty_boost()              # fact sharing    |   |
|  |  base += gaussian_noise(s=stochasticity x 0.05)        |   |
|  |  return clamp(base, 0.0, 1.0)                          |   |
|  +-------------------------------------------------------+   |
|                                                               |
|  +-------------------------------------------------------+   |
|  |  Layer 4: TRANSITION MODEL (hybrid)                     |   |
|  |  Trigger: state_machine's READY_TO_MOVE                 |   |
|  |  Success: sim8's probability model                      |   |
|  |    base_prob x target_quality_penalty                   |   |
|  |  Recovery: state_machine mechanics + fatigue             |   |
|  +-------------------------------------------------------+   |
|                                                               |
|  +-------------------------------------------------------+   |
|  |  Layer 5: RESPONSE GENERATION                           |   |
|  |  State-aware LLM prompts (from state_machine)           |   |
|  |  Template fallbacks (from visitor_templates.py)          |   |
|  |  Response type = state-to-type mapping                  |   |
|  +-------------------------------------------------------+   |
|                                                               |
|  CONFIG: stochasticity=0.5, persona_profile, persona_type    |
+-------------------------------------------------------------+
```

---

## Layer 1: State Machine Backbone

The state machine provides the behavioral skeleton. Nine visitor states, each mapped to a non-overlapping dwell range, determine the "band" of possible engagement for a given turn.

### Visitor States

| State | Dwell Range | Meaning |
|-------|-------------|---------|
| HIGHLY_ENGAGED | 0.90-1.00 | Visitor is captivated, peak engagement |
| ENGAGED | 0.75-0.90 | Normal, attentive (hub state) |
| CURIOUS | 0.55-0.70 | Visitor asked a question, wants an answer |
| READY_TO_MOVE | 0.50-0.65 | Exhibit exhausted, wants variety |
| BORED_OF_TOPIC | 0.45-0.60 | Same content too long |
| FATIGUED | 0.40-0.55 | Too many monologues without interaction |
| OVERLOADED | 0.30-0.45 | Information overload, working memory full |
| CONFUSED | 0.20-0.35 | Did not understand, needs clarification |
| DISENGAGED | 0.05-0.15 | Terminal state, hard to recover |

### Transition Triggers (from ENGAGED)

Checked in priority order. Deterministic triggers fire first; random triggers are checked only if no deterministic trigger activated.

**Deterministic triggers:**

1. **OVERLOADED**: `consecutive_explain_count >= overload_threshold` (default 3)
2. **FATIGUED**: `turns_without_question >= fatigue_threshold` (default 3)
3. **BORED_OF_TOPIC**: `consecutive_same_topic_turns >= 3`
4. **READY_TO_MOVE**: `exhibit_coverage >= 80%` AND `turns_at_exhibit >= ready_turns` (default 4)
5. **HIGHLY_ENGAGED** (positive escalation): `consecutive_engaged_turns >= 3` with `2+` different action types

**Engagement-modulated random triggers (hybrid innovation):**

6. **CURIOUS**: `base_curious_prob x (0.5 + 0.5 x engagement_level)`
   - High engagement increases curiosity (visitor is invested, asks more questions)
   - Low engagement decreases curiosity (visitor is checking out)

7. **CONFUSED**: `base_confused_prob x (1.5 - engagement_level)`
   - Low engagement increases confusion probability (compounding disengagement)
   - High engagement decreases confusion (visitor gives benefit of the doubt)

This is the core hybrid innovation for transitions: sim8's continuous engagement level modulates the state machine's random trigger probabilities. A visitor with declining engagement is more likely to become confused and less likely to become curious, matching real behavior patterns.

### Literature Grounding (preserved from state_machine)

- **OVERLOADED**: Working memory approximately 4 items (Bitgood 2013, Cowan 2001)
- **FATIGUED**: Question spacing 3-5 turns (Woo et al. 2024)
- **CURIOUS**: Users expect responses to questions (Grice 1975)
- **CONFUSED**: Conversational repair sequences (Schegloff et al. 1977)
- **READY_TO_MOVE**: Variety seeking after satiation (Bitgood 2013)

---

## Layer 2: Engagement Dynamics

Ported from sim8, the engagement dynamics layer tracks a continuous `engagement_level` (0.0-1.0) that represents the visitor's cumulative satisfaction with the agent's behavior over the session. Unlike the discrete visitor states, this value changes gradually and reflects the quality of the agent's recent actions.

### Engagement Level Updates

```
engagement_level starts at 1.0

Decreases when:
  - Agent is off-topic or irrelevant (x0.6 drop via off_topic_strikes)
  - Agent hallucinates fake fact IDs (x0.4 drop)
  - Agent uses meta-commentary instead of content

Increases when:
  - Agent shares relevant new facts (+0.05)
  - Agent successfully recovers from negative state (+0.15)
  - Agent transitions to fresh exhibit (+0.10)
```

### Multiplicative Cascades

Three independent multipliers from sim8, applied to the dwell computation:

**Question spam multiplier:**
```
consecutive_ask_questions = 0  -> 1.00
consecutive_ask_questions = 1  -> 1.00
consecutive_ask_questions = 2  -> 0.85
consecutive_ask_questions = 3+ -> 0.85 - 0.15 per additional (min 0.30)
```

**Transition spam multiplier:**
```
consecutive_transitions = 0  -> 1.00
consecutive_transitions = 1  -> 1.00
consecutive_transitions = 2+ -> 1.00 - 0.20 per additional (min 0.25)
```

**Dwell stagnation multiplier** (penalizes staying at exhausted exhibit):
```
turns_at_exhausted = 1-4   -> 1.00
turns_at_exhausted = 5-6   -> 0.98
turns_at_exhausted = 7-8   -> 0.95
turns_at_exhausted = 9-10  -> 0.90
turns_at_exhausted = 11-12 -> 0.85
turns_at_exhausted = 13-15 -> 0.75
turns_at_exhausted = 16-20 -> 0.60
turns_at_exhausted = 21+   -> 0.40
```

### Novelty Boost

From sim8. When the agent shares new facts (detected via `[FACT_ID]` tags in utterance):

```
If exhibit NOT exhausted: +0.05 to +0.15 dwell boost
If exhibit IS exhausted:  +0.03 to +0.06 dwell boost (diminished)
```

### Question Spacing Bonus

From sim8. When the agent spaces questions well (3+ non-question turns before asking):

```
Well-spaced question: x1.6 engagement_adjust_multiplier
Poorly-spaced question: x0.85 engagement_adjust_multiplier
```

---

## Layer 3: Hybrid Dwell Computation

The central innovation. The state machine determines the reward band; engagement dynamics determine the position within that band; penalties and noise shape the final value.

### Computation Steps

```
STEP 1: State determines the band
  state_low, state_high = DWELL_RANGES[visitor_state]

STEP 2: Engagement dynamics position within band
  engagement_position = state_low + engagement_level x (state_high - state_low)
  Example: ENGAGED + engagement_level=0.8 -> 0.75 + 0.8 x 0.15 = 0.87

STEP 3: State machine penalties/boosts (additive, can push outside band)
  11 components from state_machine._compute_dwell():
    1. Explain ratio penalty (0.90x if >60% of last 5 are ExplainNewFact)
    2. Cumulative overload penalty (-0.05 per OVERLOADED episode)
    3. READY_TO_MOVE escalation (-0.10 per turn ignoring ready state)
    4. Fact repetition penalty (-0.10 per consecutive repetition)
    5. Topic staleness decay (-0.05 per turn after 8+ at same exhibit)
    6. Lecture fatigue penalty (-0.08 per passive turn, capped -0.40)
    7. AskQuestion boost (+0.30/+0.10/-0.25 diminishing returns)
    7b. AskOpinion boost (+0.25/+0.08/-0.20 diminishing returns)
    8. ExplainNewFact recovery boost (+0.20 after passive slump)
    9. Exhausted exhibit penalty (-0.08 per turn after 5+, capped -0.35)
    10. Transition escape boost (+0.25 for leaving exhausted exhibit)
    11. Content starvation penalty (-0.10 per turn without facts, capped -0.50)

STEP 4: Sim8 multiplicative cascades
  sim8_multiplier = question_spam x transition_spam x dwell_stagnation
  base_dwell *= sim8_multiplier

STEP 5: Novelty boost
  base_dwell += novelty_boost (if agent shared new facts)

STEP 6: Persona noise
  noise = gaussian(mean=0, std=stochasticity x 0.05)
  base_dwell += noise

STEP 7: Final clamp
  return clamp(base_dwell, 0.0, 1.0)
```

### Example Outputs for ENGAGED State (band 0.75-0.90)

| Agent behavior | engagement_level | Multipliers | Final dwell | Signal |
|----------------|------------------|-------------|-------------|--------|
| Varied actions, new facts, good pacing | 0.9 | 1.0 | ~0.88 + novelty | Excellent |
| Standard explaining | 0.6 | 1.0 | ~0.84 | Good |
| Explain spam (3+ in a row) | 0.4 | 0.90 | ~0.72 | Declining |
| Question spam | 0.3 | 0.70 | ~0.56 | Poor |
| Off-topic + spam | 0.2 | 0.50 | ~0.39 | Very poor |

Compare to the current state machine, which returns `uniform(0.75, 0.90)` for all ENGAGED turns regardless of agent quality. The hybrid differentiates good agent behavior from bad behavior within the same visitor state.

---

## Layer 4: Transition Model

Combines the state machine's transition triggers with sim8's probability-based transition success model.

### When Transitions Happen

The state machine's READY_TO_MOVE trigger fires deterministically when exhibit coverage reaches 80% and the agent has spent enough turns. The agent must then use `OfferTransition` to move.

### Transition Success Probability

From sim8. When the agent uses `OfferTransition`:

```
Base probability (based on current exhibit completion):
  0% complete   -> 20% success
  <33% complete -> 50% success
  <67% complete -> 80% success
  >=67% complete -> 95% success

Target quality penalty (multiplicative):
  Target exhausted    -> x0.15 (visitor does not want to revisit)
  Target >=67% done   -> x0.50
  Target >=33% done   -> x0.75
  Target fresh        -> x1.0

Final: base_probability x target_quality_penalty
```

### Transition Failure

If the visitor rejects the transition, they remain at the current exhibit and generate a confusion/rejection response. The agent must try again (possibly with a different target) or continue at the current exhibit.

### Escalation

From state_machine: if the visitor is in READY_TO_MOVE and the agent fails to transition for 3+ turns, the visitor escalates to DISENGAGED (0.05-0.15 dwell). This teaches the agent urgency.

---

## Layer 5: Response Generation

### Utterance Generation

State-aware LLM prompts from the state machine, with template fallbacks:

```
Priority 1: LLM mode (default)
  - State-specific prompts with emotional context
  - Recent dialogue history for coherence
  - 1-2 sentence limit
  - Persona-influenced tone

Priority 2: Template mode (HRL_TEMPLATE_MODE=1)
  - Compositional templates from visitor_templates.py
  - 80+ templates per state/engagement level
  - Fast, no API calls

Priority 3: Fallback templates (if LLM fails)
  - Short, natural responses: "Wait, what?" (CONFUSED), "Oh cool!" (HIGHLY_ENGAGED)
```

### Response Type Mapping

Each visitor state maps deterministically to a response type for state vector construction:

```
HIGHLY_ENGAGED  -> "acknowledgment"
ENGAGED         -> "acknowledgment"
CURIOUS         -> "question"
BORED_OF_TOPIC  -> "question"
CONFUSED        -> "confusion"
OVERLOADED      -> "statement"
FATIGUED        -> "statement"
READY_TO_MOVE   -> "statement"
DISENGAGED      -> "statement"
```

---

## Persona System

The hybrid unifies both simulators' persona systems into a two-axis model.

### Axis 1: Persona Profiles (from state_machine)

Affect state transition thresholds:

| Profile | overload_thresh | fatigue_thresh | curious_prob | ready_turns |
|---------|-----------------|----------------|--------------|-------------|
| Explorer | 5 | 5 | 0.40 | 5 |
| Focused | 3 | 6 | 0.20 | 4 |
| Impatient | 3 | 3 | 0.30 | 3 |

### Axis 2: Persona Types (from sim8)

Affect engagement dynamics, gaze statistics, and behavioral variance:

| Type | Engagement variance | Behavior |
|------|--------------------| ---------|
| Agreeable | High exploration variance | Tolerant, engaged easily |
| Conscientious | Focused, less variable | Consistent, attentive |
| Neurotic | Unpredictable | Anxious, prone to confusion |

### Cross-Product: 9 Visitor Personalities

The combination produces 9 distinct visitor personalities, each with unique behavioral characteristics:

| Combination | Character |
|-------------|-----------|
| Agreeable Explorer | High curiosity, high engagement variance, tolerant |
| Agreeable Focused | Engaged easily, consistent attention, moderate curiosity |
| Agreeable Impatient | Easily pleased but wants to move fast |
| Conscientious Explorer | Thorough, methodical, asks good questions |
| Conscientious Focused | Most patient visitor, deep attention |
| Conscientious Impatient | Efficient, wants facts quickly |
| Neurotic Explorer | Curious but easily confused, high variance |
| Neurotic Focused | Anxious attention, sensitive to overload |
| Neurotic Impatient | Hardest visitor: low patience, unpredictable, easily confused |

During training, personas are sampled randomly per episode, exposing the agent to diverse visitor behaviors and preventing overfitting to a single personality.

---

## The Stochasticity Parameter

A single parameter controls the balance between deterministic state machine behavior and stochastic sim8 influence.

```
--stochasticity 0.0   Pure state machine (deterministic triggers, uniform within bands)
--stochasticity 0.3   Light noise (recommended for interpretable experiments)
--stochasticity 0.5   Balanced hybrid (recommended default)
--stochasticity 0.7   Heavy sim8 influence (recommended for robustness training)
--stochasticity 1.0   Maximum noise (engagement dynamics dominate positioning)
```

### Where Stochasticity Applies

| Component | stochasticity=0.0 | stochasticity=0.5 | stochasticity=1.0 |
|-----------|--------------------|--------------------|---------------------|
| Dwell noise (Gaussian jitter) | sigma=0.00 | sigma=0.025 | sigma=0.05 |
| Engagement influence on band position | 0% (uniform random) | 50% engagement + 50% uniform | 100% engagement-driven |
| Random trigger modulation by engagement | None (base rates only) | Moderate modulation | Full modulation |
| Persona noise on gaze features | None | Moderate | Full sim8 statistics |

### Ablation Utility

Sweeping over stochasticity values isolates which components matter for training:

```bash
for s in 0.0 0.25 0.5 0.75 1.0; do
    python train.py --simulator hybrid --stochasticity $s --episodes 500
done
```

If `stochasticity=0.0` and `stochasticity=1.0` produce similar coverage and return, the sim8 dynamics do not add training value. If there is a clear optimum in the middle, the hybrid is capturing something neither simulator captures alone.

---

## Anti-Gaming Mechanisms

The hybrid inherits anti-gaming protections from both simulators. These compound rather than overlap, making it substantially harder for the RL agent to find degenerate shortcuts.

### From State Machine

| Mechanism | What it prevents |
|-----------|-----------------|
| Recovery fatigue (-15% per recovery, up to -45%) | Spam-recover-repeat cycles |
| Content starvation penalty (-0.10/turn, cap -0.50) | AskQuestion spam without teaching |
| Exhausted exhibit penalty (-0.08/turn, cap -0.35) | Staying at completed exhibit |
| READY_TO_MOVE -> DISENGAGED escalation | Ignoring transition signals |
| Cumulative overload penalty (-0.05 per episode) | Repeated information dumping |

### From Sim8

| Mechanism | What it prevents |
|-----------|-----------------|
| Question spam multiplier (min 0.30) | Consecutive AskQuestion spam |
| Transition spam multiplier (min 0.25) | Consecutive OfferTransition spam |
| Dwell stagnation multiplier (min 0.40) | Overstaying at any exhibit |
| Off-topic strikes (engagement x0.6) | Irrelevant or hallucinated content |
| Explain spam multiplier (min 0.35) | Consecutive ExplainNewFact spam |

### Combined Effect

An agent that tries to spam ExplainNewFact faces:
1. State machine triggers OVERLOADED after 3 consecutive explains (drops to 0.30-0.45 band)
2. Sim8's explain ratio penalty (x0.90 multiplier)
3. Sim8's engagement_level decay (reduces position within band)
4. Cumulative overload penalty if it happens repeatedly (-0.05 per episode)
5. Persona noise prevents learning exact trigger boundaries

No single trick works. The agent must learn genuinely varied, responsive behavior.

---

## Integration

### Factory Registration

The hybrid registers as a third simulator option alongside sim8 and state_machine:

```python
# src/simulator/__init__.py
def get_simulator(simulator_type="sim8", ..., stochasticity=0.5):
    if simulator_type == "hybrid":
        from .hybrid_simulator import HybridSimulator
        return HybridSimulator(
            knowledge_graph=knowledge_graph,
            stochasticity=stochasticity,
            seed=seed,
        )
```

### CLI Arguments

```bash
python train.py --simulator hybrid                      # default stochasticity=0.5
python train.py --simulator hybrid --stochasticity 0.0  # pure state machine
python train.py --simulator hybrid --stochasticity 1.0  # max sim8 influence
```

### API Contract

Same interface as existing simulators. The response dict includes fields from both:

```python
response = simulator.generate_user_response(
    agent_utterance="The ring dates to the 15th century...",
    agent_option="Explain",
    agent_subaction="ExplainNewFact",
    target_exhibit="King_Caspar",
    current_exhibit_completion=0.45,
    exhibit_exhausted=False,
    target_exhibit_completion=0.12,
    target_exhibit_exhausted=False,
)

# Returns:
{
    "utterance": "Oh interesting, was it made locally?",
    "aoi": "Ring",
    "persona": "Agreeable",
    "gaze_features": [0.84, 0.07, 0.82, 2.1, 0.71, 5.3],  # 6D vector
    "response_type": "follow_up_question",
    "visitor_state": "engaged",           # From state machine backbone
    "engagement_level": 0.84,             # From sim8 dynamics (primary reward)
    "off_topic_strikes": 0,
    "transition_success": False,
    "simulator_llm_time": 0.45,
    "repeat_request": False,
}
```

### Recovery Model

Combines state machine recovery rates with engagement-modulated success:

```
Recovery probability = base_rate x fatigue_penalty + engagement_bonus

Where:
  base_rate       = state_machine recovery rate (e.g., ClarifyFact on CONFUSED = 0.90)
  fatigue_penalty = max(0, 1.0 - 0.15 x recovery_count)
  engagement_bonus = 0.1 x engagement_level

On successful recovery:
  visitor_state -> ENGAGED
  engagement_level += 0.15 (recovery boost)
  recovery_count += 1
```

Recovery rates by state and action (base rates):

| State | ClarifyFact | AskClarification | OfferTransition | ExplainNewFact | AskQuestion |
|-------|-------------|------------------|-----------------|----------------|-------------|
| CONFUSED | 0.90 | 0.70 | 0.60 | 0.40 | - |
| OVERLOADED | - | - | 0.75 | - | 0.85 |
| CURIOUS | - | - | - | 0.90 (with fact) | 0.50 (deflection risk) |
| BORED_OF_TOPIC | - | - | 0.90 | 0.85 (different topic) | - |
| FATIGUED | - | - | 0.85 | - | 0.80 |
| READY_TO_MOVE | - | - | 0.95 | - | - |
| DISENGAGED | - | - | 0.50 | - | - |

---

## Expected Training Improvements

| Metric | State Machine | Sim8 | Hybrid (predicted) |
|--------|--------------|------|---------------------|
| Reward clarity | High (non-overlapping bands) | Low (overlapping ranges) | High (bands preserved) |
| Within-state signal | None (uniform random) | High (engagement dynamics) | High (engagement positions in band) |
| Gaming resistance | Medium (predictable thresholds) | Medium (no state structure) | High (double anti-gaming layers + noise) |
| Policy robustness | Low (3 persona profiles) | Medium (3 persona types) | High (9 combinations) |
| Ablation utility | None | None | High (stochasticity sweep) |
| Interpretability | High (named states) | Low (continuous only) | High (named states + continuous) |
| Recovery learning | Yes (state machine) | No | Yes (enhanced with engagement) |
| Novelty incentive | No | Yes (fact boost) | Yes (preserved from sim8) |

### Key Prediction

The hybrid should reduce degenerate behaviors (ExplainNewFact spam, question spam) more effectively than either simulator alone, because the agent faces both:
- Clear state-level consequences (OVERLOADED, FATIGUED) from the state machine
- Continuous within-state degradation from sim8's engagement dynamics

The agent cannot avoid one penalty system by exploiting the other, since both apply simultaneously.

---

## Implementation Estimate

- **New file**: `src/simulator/hybrid_simulator.py` (~800-1000 lines)
- **Modified files**: `src/simulator/__init__.py` (factory registration), `train.py` (CLI arg)
- **Reused**: `visitor_templates.py` (shared with state_machine), `DWELL_RANGES` and `VisitorState` (imported from state_machine_simulator)
- **No changes to**: environment, training loops, reward computation, or state builder
