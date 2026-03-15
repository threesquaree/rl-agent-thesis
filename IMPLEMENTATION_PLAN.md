# Plan: Implement Centred Engagement + Broadened Novelty Reward

## Context

The thesis has shifted from a pure bipolar reward focus to a three-pillar approach. Pillar 1 (supporting contribution) involves two reward modifications:
1. **Centred engagement**: `r^ceng_t = w_e * (dwell_t - EMA(dwell))` — already implemented, just needs enabling
2. **Broadened novelty**: distributes novelty credit across all content-advancing actions, not just ExplainNewFact — **needs implementation**

After implementation, retrain the flat Actor-Critic agent with both modifications enabled to compare against the standard baseline.

---

## What Already Exists

- **Centred engagement** is fully implemented in `env.py:72-76` and `env.py:532-538`. Activated via `--centred-engagement` flag. No code changes needed.

---

## Implementation: Broadened Novelty Reward

### Formula (Thesis Proposal Eq. 5)
```
r^bnov = α_new × 1[ExplainNewFact ∧ new_facts>0]
       + α_rep × 1[RepeatFact]
       + α_clar × 1[ClarifyFact]
       + α_ask × 1[AskOpinion ∨ AskMemory ∨ AskClarification]
       - α_stale × 1[ExhaustedExhibit ∧ non-Explain action]
```

### Default Weights
| Weight | Default | Rationale |
|--------|---------|-----------|
| `α_new` | 1.0 | Matches current `novelty_per_fact` default |
| `α_rep` | 0.3 | Partial credit for reinforcement |
| `α_clar` | 0.3 | Partial credit for depth |
| `α_ask` | 0.2 | Mild reward for engagement variety |
| `α_stale` | 0.5 | Staleness penalty at exhausted exhibits |

### Design Decisions
- **Replaces** standard novelty when `--broadened-novelty` flag is set (not additive — α_new subsumes the old novelty)
- `--novelty-per-fact` is ignored when broadened novelty is active (warning printed)
- `α_stale` only fires on **non-Explain** actions at exhausted exhibits (Explain actions already penalized by the existing `exhaustion_penalty`)
- `ExplainNewFact` reward requires `len(new_fact_ids) > 0` (no reward for failed fact delivery)

---

## Files to Modify (3 files)

### 1. `train.py` — CLI args + env var passthrough
- **Lines ~161-163**: Add 6 CLI arguments after `--novelty-per-fact`:
  - `--broadened-novelty` (store_true)
  - `--alpha-new` (float, default 1.0)
  - `--alpha-rep` (float, default 0.3)
  - `--alpha-clar` (float, default 0.3)
  - `--alpha-ask` (float, default 0.2)
  - `--alpha-stale` (float, default 0.5)
- **Line ~264**: Warning if both `--broadened-novelty` and non-default `--novelty-per-fact`
- **Lines ~401-409**: Add broadened novelty fields to `metadata["reward_parameters"]`
- **Lines ~448**: Update novelty console output (conditional on broadened flag)
- **Lines ~469-473**: Add env var passthrough (`HRL_BROADENED_NOVELTY`, `HRL_ALPHA_*`)

### 2. `src/environment/env.py` — Core reward logic
- **Lines ~78-79** (`__init__`): Add 7 instance vars (broadened_novelty flag + 5 alpha weights)
- **Lines ~231-238** (`reset()`): Add 5 tracking accumulators (`bnov_*_sum`)
- **Lines ~542-547** (`step()`): Replace novelty block with conditional:
  - If `self.broadened_novelty`: compute 5 sub-components based on subaction indicators
  - Else: keep existing `len(new_fact_ids) * self.novelty_per_fact`
- **Lines ~728-764** (`step()` info dict): Add per-turn broadened novelty fields
- **Lines ~768-779** (`step()` component breakdown): Add end-of-episode broadened novelty breakdown

### 3. `run_reward_comparison.sh` (NEW) — Experiment script
Run 4 conditions × 3 seeds for systematic comparison:
- A: Standard engagement + standard novelty (current baseline)
- B: Centred engagement + standard novelty
- C: Standard engagement + broadened novelty
- D: Centred engagement + broadened novelty (full proposed system)

---

## Step-by-Step Execution

### Step 1: Add CLI args to `train.py`
After `--novelty-per-fact` argument (line ~162), add the 6 new arguments.

### Step 2: Add warning + env var passthrough in `train.py`
After `args = parser.parse_args()`, add conflict warning.
After `HRL_DWELL_EMA_ALPHA` passthrough (line ~473), add 6 new env vars.

### Step 3: Update metadata + console output in `train.py`
Add broadened novelty params to `reward_parameters` dict (line ~406).
Update novelty print line (line ~448) to be conditional.

### Step 4: Add broadened novelty params in `env.py` `__init__`
After `self.novelty_per_fact` (line ~79), add:
```python
self.broadened_novelty = os.environ.get("HRL_BROADENED_NOVELTY", "0") == "1"
self.alpha_new = float(os.environ.get("HRL_ALPHA_NEW", "1.0"))
self.alpha_rep = float(os.environ.get("HRL_ALPHA_REP", "0.3"))
self.alpha_clar = float(os.environ.get("HRL_ALPHA_CLAR", "0.3"))
self.alpha_ask = float(os.environ.get("HRL_ALPHA_ASK", "0.2"))
self.alpha_stale = float(os.environ.get("HRL_ALPHA_STALE", "0.5"))
```

### Step 5: Add accumulators in `env.py` `reset()`
After `self.novelty_sum = 0.0` (line ~231), add 5 broadened novelty accumulators.

### Step 6: Replace novelty computation in `env.py` `step()`
Replace lines ~542-547 with conditional block:
- Initialize `bnov_*` locals to 0.0 (for scope safety)
- If `self.broadened_novelty`: compute each indicator-based sub-component
- Else: original computation
- Track sub-component sums

### Step 7: Extend info dict + component breakdown in `env.py`
Add `reward_bnov_*` fields to per-turn info dict (line ~734 area).
Add `bnov_*_contribution` fields to end-of-episode component breakdown (line ~768 area).

### Step 8: Create experiment comparison script
Create `run_reward_comparison.sh` with the 4 conditions.

---

## Verification

1. **Unit check**: Run a quick 5-episode training with `--broadened-novelty --centred-engagement --verbose` and verify:
   - Broadened novelty sub-components print correctly per turn
   - RepeatFact/ClarifyFact/AskQuestion actions receive reward (not just ExplainNewFact)
   - Staleness penalty fires at exhausted exhibits for non-Explain actions
   - Centred engagement shows EMA-based values
2. **Backward compatibility**: Run 5 episodes WITHOUT the new flags and confirm identical behavior to current code
3. **Full experiment**: Run the 4-condition comparison script with 300+ episodes per seed
