# Reward Reformalization & Action Space Redesign
**Date:** 2026-05-09
**Scope:** Flat MDP only
**Repository:** Duplicate of existing Thesis repo (existing repo untouched)
**Status:** Approved for implementation planning

---

## 1. Motivation

The existing reward function suffers from a novelty-dominates-engagement imbalance:
- `ExplainNewFact` yields `alpha_new = 1.0` novelty reward unconditionally
- Engagement reward at dwell=0.2 yields at most `0.2 × 1.0 = 0.20`
- Result: agent earns +0.70 reward while visitor is disengaged → action collapse toward ExplainNewFact spam

The reformalization replaces level-based rewards (dwell × weight) with trajectory-based rewards (Δdwell), grounded in prospect theory.

---

## 2. Reward Formalization

### Core Equation

```
R_t = α · max(0, dwell_t − dwell_{t−1})
    − β · max(0, dwell_{t−1} − dwell_t)
    + R_terminal · (exhibits_covered / total_exhibits)    [episode end only]
```

### Parameters

| Parameter | Default | Source |
|-----------|---------|--------|
| α | 1.0 | Gain weight |
| β | 2.25 | Loss weight — ratio β/α = 2.25 from Kahneman & Tversky (1979) |
| R_terminal | 5.0 | Terminal coverage bonus (tunable) |

### Theoretical Grounding

The ratio β/α = 2.25 is taken directly from Kahneman & Tversky's (1979) prospect theory — their empirical finding that losses are weighted approximately 2.25× more heavily than equivalent gains. This provides a principled, theoretically motivated default rather than an arbitrary hyperparameter.

### Example Values

| Situation | Calculation | R_t |
|-----------|-------------|-----|
| dwell 0.2 → 0.7 (recovery) | α × 0.5 | +0.50 |
| dwell 0.7 → 0.2 (decline) | −β × 0.5 | −1.125 |
| dwell flat 0.5 → 0.5 | 0 | 0.00 |
| dwell 0.8 → 0.9 (fine-tuning) | α × 0.1 | +0.10 |

### Coverage Absorption

Novelty is not a separate reward component. New facts naturally spike dwell → positive delta → positive reward. The terminal bonus ensures broad exhibit coverage without per-turn pressure.

---

## 3. Action Space Redesign (Flat MDP)

### Removed Actions

| Action | Reason |
|--------|--------|
| RepeatFact | Spammed, causes action cycling, low dwell signal |
| ClarifyFact | Functionally overlaps ExplainNewFact in practice |
| AskMemory | Indistinguishable from AskOpinion in simulator response |

### Final Flat Action Space (6 actions)

| Action | Role |
|--------|------|
| ExplainNewFact | Primary content delivery |
| AskOpinion | Engagement probe, invites reflection |
| AskClarification | Handles confusion signal from visitor |
| SummarizeAndSuggest | Exhibit transition |
| WrapUp | Episode conclusion |
| **RecoverEngagement** | **NEW: explicit trajectory-reversal action** |

### RecoverEngagement — Specification

**Purpose:** When dwell is falling, the agent needs a tool that does not advance content (risking overload of a disengaged visitor) but actively pivots toward re-engagement via a lighter, relatable response — a surprising connection, a brief anecdote, or a "did you know?" hook loosely tied to the current exhibit.

**Simulator behaviour:**
```
base_dwell_response:  randf(0.55, 0.75)   ← above disengaged range
diminishing returns:  ×0.70 on 2nd consecutive use
                      ×0.40 on 3rd+ use   ← severe penalty, not a spam action
intended use:         1 strategic use per exhibit when Δdwell < −0.1
```

**Key ablation:** Does the trained agent learn to select RecoverEngagement specifically when Δdwell < 0 in the prior turn? If yes, the trajectory reward is teaching trajectory-responsive behaviour. If no, the reward shaping has failed.

---

## 4. State Space Addition

### Trajectory Feature τ_t

```
Current:  s_t = [f_t, h_t, i_t, c_t]

Proposed: s_t = [f_t, h_t, i_t, c_t, τ_t]
```

`τ_t` is a 2-dimensional vector: `[dwell_t_norm, Δdwell_t]` where `dwell_t_norm = 2 × dwell_t − 1` maps [0,1] → [−1,1], and `Δdwell_t = dwell_t − dwell_{t−1}` is naturally in [−1,1].

**Rationale:** Without `τ_t`, the agent must infer trajectory entirely from visitor language (i_t). Adding `τ_t` makes state and reward consistent — the agent explicitly observes what it is being rewarded on. This is theoretically cleaner and produces a more interpretable ablation.

**VR deployment:** In a VR environment, `τ_t` is computed from the headset's eye tracker gaze fixation data — the one continuous engagement signal VR makes easily measurable.

---

## 5. System Changes Required

### env.py
- Add `τ_t = [dwell_t, Δdwell_t]` to observation vector (2-d addition)
- Replace 8-action flat space with 6-action space
- Replace `R_t = engagement + novelty` with asymmetric delta formula
- Add CLI-configurable `--alpha`, `--beta`, `--terminal_coverage_weight`

### hybrid_simulator.py / sim8_adapter.py
- Add `RecoverEngagement` response handler with specified dwell curve
- Diminishing returns tracking for consecutive RecoverEngagement use

### train.py
- New CLI flags: `--alpha` (default 1.0), `--beta` (default 2.25), `--terminal_coverage_weight` (default 5.0)

---

## 6. Thesis Contribution Summary

| Dimension | Current | Proposed |
|-----------|---------|----------|
| Reward signal | Engagement level (dwell) + novelty credits | Engagement trajectory (Δdwell), asymmetric |
| Theoretical grounding | Engagement-novelty balance (ad hoc weights) | Prospect theory (Kahneman & Tversky, 1979) |
| Action space | 8 flat actions (3 redundant) | 6 flat actions + trajectory-targeted RecoverEngagement |
| State | No engagement signal | Explicit trajectory feature τ_t |
| Key ablation | — | Does agent deploy RecoverEngagement when Δdwell < 0? |

---

## 7. Implementation Notes

- All changes go in a **duplicate repository** — the existing Thesis repo is not modified
- Existing experiments remain reproducible from the original repo
- New experiments run from the duplicate with the reformalized reward
