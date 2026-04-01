# Action Spam Fixes: ExplainNewFact + AskClarification Spam

## Problem Description

The `H1_MDP_StateMachine_CentredEng_BroadNov` model exhibits a degenerate two-phase policy when tested on real users:

1. **Phase 1**: Spams `ExplainNewFact` until all facts at the current exhibit are exhausted
2. **Phase 2**: Switches to spamming `AskClarification` indefinitely (never transitions)

### Root Cause

**ExplainNewFact spam**: `alpha_new = 1.0` is the highest broadened novelty reward, so the agent greedily picks it every turn. The state machine triggers OVERLOADED after 3 consecutive ExplainNewFacts, but the novelty bonus (+1.0) dominates the dwell penalty.

**AskClarification spam after exhaustion**: Once ExplainNewFact is masked (exhibit exhausted), the agent evaluates remaining actions:

| Action at Exhausted Exhibit | Broadened Nov | Exhaustion Penalty | Stale Penalty | Dwell Boost | Approx Net |
|-----------------------------|---------------|--------------------|---------------|-------------|------------|
| RepeatFact / ClarifyFact    | +0.3          | **-0.5**           | --            | none        | -0.2       |
| AskClarification            | +0.2          | none               | -0.5          | +0.30       | 0.0 to +0.1|
| SuggestMove                 | **0.0**       | none               | none          | none        | ~0.0       |

AskClarification wins because: (a) avoids the exhaustion penalty (only applies to Explain subactions), (b) the state machine question boost (+0.30 dwell) partially offsets the stale penalty, and (c) SuggestMove has **zero intrinsic reward** (`HRL_TRANSITION_BONUS` was 0.0).

---

## Fixes Implemented (in `src/environment/env.py`)

### Fix 1: Action Repetition Penalty (new)

**What**: Penalizes consecutive selection of the same subaction. Breaks both ExplainNewFact and AskClarification spam patterns.

**How**: After `N` consecutive uses of the same subaction (default `N=2`), a linearly scaling penalty is applied:
```
penalty = -action_repeat_penalty * max(0, consecutive_count - threshold)
```

**Default values**:
- `HRL_ACTION_REPEAT_PENALTY = 0.15` (penalty per repeat over threshold)
- `HRL_ACTION_REPEAT_THRESHOLD = 2` (penalty starts after 2 consecutive)

**Example**: 4 consecutive AskClarification actions:
- Turn 1-2: No penalty (within threshold)
- Turn 3: -0.15 (1 over threshold)
- Turn 4: -0.30 (2 over threshold)

**Config**:
```bash
HRL_ACTION_REPEAT_PENALTY=0.15  # Penalty magnitude per repeat
HRL_ACTION_REPEAT_THRESHOLD=2   # Starts penalizing after N consecutive
```

---

### Fix 2: Broadened Novelty Transition Component (new)

**What**: Added `alpha_transition` term to broadened novelty so transitions get a positive novelty signal. Previously, SuggestMove had zero broadened novelty reward -- no gradient toward transitioning.

**How**:
- Full `alpha_transition` (+0.4) for transitioning FROM exhausted exhibits
- Half `alpha_transition` (+0.2) for transitioning from non-exhausted exhibits
- Transitions are also **exempt from stale penalty** (they are the solution, not the problem)

**Config**:
```bash
HRL_ALPHA_TRANSITION=0.4   # Novelty reward for transitions
```

**Effect on reward table at exhausted exhibit**:

| Action              | Before Fix | After Fix |
|---------------------|------------|-----------|
| AskClarification    | ~0.0       | **-0.95** (stale + repeat penalty + diminishing ask) |
| SuggestMove         | ~0.0       | **+0.7**  (transition novelty + transition bonus) |

---

### Fix 3: Diminishing Returns on Ask Bonus (modified)

**What**: `alpha_ask` now has diminishing returns for consecutive AskQuestion subactions, matching the state machine's dwell-level diminishing returns with reward-level diminishing returns.

**How** (within broadened novelty computation):
- 1st consecutive ask: Full `alpha_ask` (+0.2)
- 2nd consecutive ask: Half `alpha_ask` (+0.1)
- 3rd+ consecutive ask: Zero bonus

**No new config needed** -- uses existing `HRL_ALPHA_ASK`.

---

### Fix 4: Updated Default Reward Parameters

Three defaults were too weak to create proper gradients:

| Parameter | Old Default | New Default | Rationale |
|-----------|-------------|-------------|-----------|
| `HRL_ALPHA_STALE` | 0.5 | **1.0** | Staying at exhausted exhibits must be clearly worse than transitioning |
| `HRL_EXHAUSTION_PENALTY` | -0.5 | **-1.0** | Explain at exhausted exhibits must strongly push toward other actions |
| `HRL_TRANSITION_BONUS` | 0.0 | **0.3** | Agent needs a direct positive signal for successful transitions |

---

## Will Sim8 Work Better?

The `H1_MDP_Sim8_CentredEng_BroadNov_RespType` model will show **less severe AskClarification spam** due to Sim8's multiplicative `question_spam_multiplier` (min 0.30x dwell) and `dwell_stagnation_multiplier`. However:

- **ExplainNewFact spam persists** -- still the highest-reward action
- **Core issue remains** -- the missing transition incentive is in `env.py`, not the simulator
- **Prediction**: Same qualitative pattern, less severe quantitatively

The fixes above address the root cause at the reward level, so they work with **any** simulator backend.

---

## Recommended Training Configs

### Config A: Minimal Fix (baseline reward mode)
Best for quick iteration -- uses only the new anti-spam mechanisms:
```bash
python3 train.py \
  --simulator state_machine \
  --episodes 500 \
  HRL_CENTRED_ENGAGEMENT=1 \
  HRL_BROADENED_NOVELTY=1 \
  HRL_RESPONSE_TYPE_REWARD=1
```
The new defaults (stale=1.0, exhaustion=-1.0, transition_bonus=0.3, action_repeat_penalty=0.15, alpha_transition=0.4) are active automatically.

### Config B: Full Fix (augmented reward mode)
Adds transition insufficiency penalties and exploration bonuses for maximum coverage:
```bash
python3 train.py \
  --simulator hybrid --stochasticity 0.5 \
  --reward_mode augmented \
  --episodes 500 \
  HRL_CENTRED_ENGAGEMENT=1 \
  HRL_BROADENED_NOVELTY=1 \
  HRL_RESPONSE_TYPE_REWARD=1
```

### Config C: Conservative (if existing models break)
If the new defaults are too aggressive, dial them back:
```bash
HRL_ALPHA_STALE=0.7 \
HRL_EXHAUSTION_PENALTY=-0.7 \
HRL_TRANSITION_BONUS=0.2 \
HRL_ACTION_REPEAT_PENALTY=0.10 \
HRL_ALPHA_TRANSITION=0.3
```

---

## New Environment Variables Summary

| Variable | Default | Description |
|----------|---------|-------------|
| `HRL_ACTION_REPEAT_PENALTY` | 0.15 | Penalty per consecutive same-subaction over threshold |
| `HRL_ACTION_REPEAT_THRESHOLD` | 2 | Number of consecutive repeats before penalty kicks in |
| `HRL_ALPHA_TRANSITION` | 0.4 | Broadened novelty reward for transition actions |

## Modified Defaults

| Variable | Old Default | New Default |
|----------|-------------|-------------|
| `HRL_ALPHA_STALE` | 0.5 | 1.0 |
| `HRL_EXHAUSTION_PENALTY` | -0.5 | -1.0 |
| `HRL_TRANSITION_BONUS` | 0.0 | 0.3 |
