# Thesis Direction — Modified

**Title:** Simulator Fidelity, Reward Design, and Gaze-Based Engagement Signals for RL-Driven Museum Guide Agents  
**Framework:** Flat MDP (Actor-Critic / A2C) throughout — no hierarchical RL

---

## Baseline

Daniel Bourdon's flat Actor-Critic agent trained on **Sim8** (probabilistic response simulator).  
Sim8 uses overlapping dwell distributions per response type (e.g., confusion [0.25, 0.50] overlapping with question [0.40, 0.70]), making credit assignment difficult. This baseline achieves high content coverage (≈98.7%) but limited engagement-adaptive behaviour due to simulator signal ambiguity.

The baseline is the fixed point of comparison across all three pillars.

---

## Pillar 2 — Simulator Fidelity Ladder *(Main Contribution)*

**Core claim:** The ceiling of what a flat RL agent can learn is determined by simulator signal quality.  
**Approach:** A hybrid simulator fidelity ladder — three progressively richer simulator versions, each adding one fidelity dimension to the base.

### Rung 0 — Baseline
Flat MDP trained on Sim8 (Bourdon's baseline).

### Rung 1 — Hybrid Simulator
A **Hybrid Simulator** (`src/simulator/hybrid_simulator.py`) that blends Sim8's continuous engagement dynamics with the State Machine's discrete state backbone. The core innovation: the State Machine determines the *reward band* (e.g., ENGAGED → dwell ∈ [0.75, 0.90]) while Sim8's continuous `engagement_level` (0.0–1.0) determines the *position within that band*. This preserves non-overlapping dwell ranges (clean credit assignment) while adding rich within-state nuance.

#### `stochasticity` parameter ∈ [0, 1]
- `0.0` = pure State Machine: uniform random positioning within band, no engagement modulation
- `1.0` = full Sim8 influence: `engagement_level` drives band position, random triggers fully modulated
- Intermediate values linearly interpolate both the band-positioning and the random trigger probabilities

#### 5-Layer Architecture
**Layer 1 — State Machine Backbone:** 9 visitor states (HIGHLY_ENGAGED, ENGAGED, CURIOUS, CONFUSED, OVERLOADED, BORED_OF_TOPIC, FATIGUED, READY_TO_MOVE, DISENGAGED). Deterministic triggers fire from the ENGAGED state:

| Trigger | Condition |
|---|---|
| OVERLOADED | `consecutive_explain_count ≥ overload_threshold` (default 3) |
| FATIGUED | `turns_without_question ≥ fatigue_threshold` (default 3) |
| BORED_OF_TOPIC | `consecutive_same_topic_turns ≥ 3` |
| READY_TO_MOVE | exhibit completion ≥ 80% AND turns at exhibit ≥ `ready_turns` |
| HIGHLY_ENGAGED | 3+ consecutive engaged turns with ≥ 2 distinct action types used |

Additionally, two **engagement-modulated random triggers** (the hybrid innovation) fire with probabilities scaled by `stochasticity` and `engagement_level`:
- CURIOUS: `base_prob × (0.5 + 0.5 × engagement)` at full stochasticity → higher engagement → more curiosity
- CONFUSED: `base_prob × (1.5 − engagement)` at full stochasticity → lower engagement → more confusion

Recovery rates use State Machine base rates (e.g., ClarifyFact → CONFUSED recovery at 90%), reduced by −15% per prior recovery attempt (max −45% fatigue), plus an engagement bonus of `+0.1 × engagement_level × stochasticity`.

**Layer 2 — Sim8 Engagement Dynamics:** A continuous `engagement_level` (initialised at 1.0) is updated each turn via a multiplicative adjustment. Multipliers >1.0 improve engagement, <1.0 degrade it. Key adjustments:
- Well-spaced AskQuestion (≤1 consecutive): ×1.3; question spam (3+): ×0.80
- New fact on non-exhausted exhibit: ×1.15; on exhausted: ×1.06
- Good transition target: ×1.15; exhausted target: ×0.85; rejected: ×0.85
- Off-topic/meta-commentary detected: ×0.80 per occurrence
- Hallucination detected: `engagement_level ×= 0.4` (immediate drop)

**Layer 3 — Hybrid Dwell Computation:** The primary reward signal, computed in 4 steps:
1. *Band*: `[low, high]` from State Machine's DWELL_RANGES
2. *Position*: `(1 − stochasticity) × uniform + stochasticity × engagement_level`
3. *State Machine additive penalties/boosts* (11 total): explain ratio penalty, cumulative overload floor reduction (−0.05/episode), READY_TO_MOVE escalation, fact repetition penalty, topic staleness decay (after 8 turns at exhibit), lecture fatigue penalty (up to −0.40), AskQuestion boost with diminishing returns (+0.30 → +0.10 → −0.25), ExplainNewFact recovery boost, exhausted exhibit penalty (up to −0.35), transition escape boost (+0.25), content starvation penalty (−0.10/turn after 3 turns without new fact)
4. *Sim8 multiplicative cascades*: question spam multiplier, transition spam multiplier, dwell stagnation multiplier (0.40× at 20+ exhausted exhibit turns)
5. *Persona noise*: N(0, stochasticity × 0.05) — prevents threshold gaming

**Layer 4 — Transition Model:** Coverage-dependent acceptance probability for ENGAGED state (20% at <20% completion → 95% at 60%+); state-specific recovery rates for negative states.

**Layer 5 — Response Generation:** State-aware LLM utterances or fallback templates; response type mapped deterministically from visitor state (CURIOUS → "question", CONFUSED → "confusion", etc.).

#### Two Independent Persona Axes
- **Sim8 personas** (Agreeable, Conscientious, Neurotic): govern gaze feature statistics (SaccadeSpan, TurnGazeEntropy, TurnFixChangeRate, GazeEntryLatency) drawn from persona-specific Gaussian distributions
- **State Machine profiles** (Explorer, Focused, Impatient): govern behavioural thresholds (Explorer: overload=5, fatigue=5; Impatient: overload=3, fatigue=3, recovery_modifier=0.90)

#### Note on Gaze Feature Independence
DominantObjectRatio is derived as `dwell_time × U(0.6, 0.95)` — it is *not* independently modelled. The other 4 non-dwell features (SaccadeSpan, Entropy, FixChangeRate, GazeEntryLatency) are drawn from persona statistics independently of dwell. This partial independence matters for Pillar 3 comparisons.

This is the primary simulator for Rungs 2 and 3.

### Rung 2 — Hybrid + Gradual Engagement Drift
Adds a **gradual engagement drift** mechanism to the Hybrid Simulator's engagement signal.  
Rather than threshold-triggered discrete state jumps (e.g., 3 consecutive `ExplainNewFact` → `Overloaded`), engagement drifts continuously based on accumulated dialogue history. This produces more realistic, temporally smooth reward signals and tests whether the agent can learn proactive maintenance strategies (e.g., asking questions before disengagement reaches a threshold).

### Rung 3 — Hybrid + Drift + Cross-Exhibit Visitor Memory
Adds **cross-exhibit visitor memory**: visitor engagement state is no longer reset when transitioning between exhibits. A visitor who was overloaded at Exhibit 1 carries that history into Exhibit 2. This tests whether the agent learns exhibit-sequencing strategies that account for cumulative visitor fatigue.

### Evaluation Metrics (Pillar 2)
- Cumulative reward and coverage
- Action diversity (entropy, per-action usage distribution)
- Learning efficiency (episodes to convergence)
- Qualitative behaviours: recovery action usage, engagement-contingent question-asking, proactive transition decisions

---

## Pillar 1 — Reward Function Design *(Supporting)*

**Core claim:** The structural non-negativity of the baseline reward prevents the agent from learning that disengagement is bad.

### Reward Pathologies in Bourdon's Baseline
1. **Asymmetric engagement reward**: `r_eng = w_e · dwell ≥ 0` always — near-disengagement (dwell = 0.10) still yields positive reward
2. **Structural novelty bias**: novelty reward `r_nov ≥ 0` always, and only `ExplainNewFact` can trigger it — creates incentive for action repetition

### Proposed Corrections
1. **Centred engagement reward**: subtract a running baseline from dwell time to produce genuine negative signals for below-average engagement  
   `r_eng' = w_e · (dwell_t − d̄)`  
   where `d̄` is a running average of recent dwell values

2. **Broadened novelty reward**: a full redesign of the novelty component that expands which actions earn credit and adds corrective penalties. Six sub-components (from `src/environment/env.py`):

   | Sub-component | Effect | Default weight |
   |---|---|---|
   | `bnov_new` | Reward for `ExplainNewFact` (with ENF geometric decay applied) | α_new = 1.0 |
   | `bnov_rep` | Reward for `RepeatFact` | α_rep = 0.3 |
   | `bnov_clar` | Reward for `ClarifyFact` | α_clar = 0.3 |
   | `bnov_ask` | Reward for ask actions — full → half → zero on consecutive uses | α_ask = 0.2 |
   | `bnov_transition` | Reward for `SuggestMove`, higher when exhibit is exhausted | α_transition = 0.4 |
   | `bnov_stale` | **Penalty** for taking non-explain, non-transition actions at an exhausted exhibit | α_stale = 1.0 |

   **ENF geometric decay** (applies in both standard and broadened modes): `ExplainNewFact`'s novelty multiplier decays as `decay_rate^(n−1)` for the nth consecutive use, floored at `enf_decay_floor`. Resets to 1.0 the moment any other action is taken (default: decay_rate = 0.65, floor = 0.25).

   The key structural change: novelty is no longer exclusive to `ExplainNewFact` — `RepeatFact`, `ClarifyFact`, and ask actions all earn credit, directly addressing the baseline's action-collapse bias.

Both corrections stay within the flat MDP framework and are grounded in potential-based reward shaping theory (Ng et al., 1999).

---

## Pillar 3 — Gaze Feature Representations *(Exploratory)*

**Core claim:** Using only dwell time discards potentially informative eye-tracking dimensions.

### Six Available Gaze Features
| Feature | Captures |
|---|---|
| Dwell time | Proportion of turn with gaze in exhibit AOI |
| Saccade span | Average amplitude of eye movements |
| Gaze entropy | Spatial dispersion of fixations across AOIs |
| Fixation change rate | Temporal dynamics — gaze shifts per turn |
| Dominant object ratio | Proportion of fixation time on single most-attended object |
| Gaze entry latency | Time to first fixation on exhibit AOI after turn start |

### Approach
Compare individual features and selected combinations as RL engagement signals. Evaluate policy quality, learning speed, and action diversity under matched conditions.

**Dependency on Pillar 2**: Meaningful multi-feature comparison requires a simulator that generates features with independent information content. In Bourdon's Sim8, all six features are derived from dwell time and response type — effectively transformations of the same signal. The Hybrid Simulator (Rung 1+) must be extended to generate features with independent modelling for Pillar 3 to produce interpretable results.

---

## Research Questions (Updated)

**RQ1 (Pillar 2 — Main):** How does the fidelity of the visitor simulator — from Sim8, to Hybrid, to Hybrid+drift, to Hybrid+drift+memory — affect the emergence of engagement-adaptive dialogue policies in a flat RL agent?

**RQ2 (Pillar 1 — Supporting):** Do bipolar reward signals (centred engagement + broadened novelty) improve policy diversity and reduce action repetition compared to the non-negative baseline?

**RQ3 (Pillar 3 — Exploratory):** Which gaze features, individually and in combination, provide the most informative engagement signal for RL policy learning?

---

## Key Deviations from Proposal v2

| Aspect | Proposal v2 | Modified Direction |
|---|---|---|
| RL framework | HRL (Option-Critic) vs flat MDP comparison | Flat MDP only |
| Baseline | Bourdon's flat A2C (general) | Explicitly Bourdon's flat A2C on Sim8 |
| Pillar 2 mechanism | SM-v1 → SM-v4 State Machine ladder | Hybrid Simulator → +drift → +cross-exhibit memory |
| Pillar 2 base simulator | State Machine (theory-driven) | Hybrid Simulator (SM + Sim8 blend) |
| Pillar 1 | Bipolar reward (same) | Bipolar reward (same) |
| Pillar 3 | Multi-feature gaze (same) | Multi-feature gaze (same) |
| Option collapse stat | 88.3% (from HRL hierarchical design) | Not applicable to flat MDP — remove from framing |
