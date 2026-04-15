# Adaptive Stochasticity Schedule for HybridSimulator

**Date:** 2026-04-15
**Status:** Approved, pending implementation
**Scope:** `src/simulator/stochasticity_schedule.py`, `src/training/training_loop.py`, launch scripts

---

## Background

`HybridSimulator` controls visitor behaviour realism through a `stochasticity` parameter (range 0–1):

- **High stochasticity (≈0.8):** Dwell is positioned aggressively by the continuous `engagement_level` signal, and CURIOUS/CONFUSED state transitions are strongly modulated by engagement. The simulator produces high-variance, exploration-rich reward signals.
- **Low stochasticity (≈0.2):** Dwell is mostly sampled uniformly within the state band, and state-transition modulation is weak. The simulator produces low-variance, stable reward signals.

Current experiments fix `stochasticity=0.5` for all 500 episodes. Analysis shows the Hybrid's converged reward window has σ=10.52 vs Sim8's σ=3.55, suggesting the fixed high stochasticity is inflating variance late in training when the agent needs stability.

**Hypothesis:** A curriculum that starts with high stochasticity (rich exploration signal) and decays to low stochasticity (stable convergence signal) will preserve Hybrid's faster convergence advantage while closing the variance gap.

---

## Design

### Schedule Shape: Cosine Annealing

```
stoch(t) = end + 0.5 * (start - end) * (1 + cos(π · t / T))
```

Where `t` is the current episode (0-indexed) and `T` is total episodes.

Properties:
- At `t=0`: returns `start` (e.g. 0.8)
- At `t=T-1`: returns `end` (e.g. 0.2)
- Slow at both ends, fast in the middle — preserves exploration early, stabilises late
- Mirrors cosine LR annealing (standard in ML curriculum literature)
- When `start == end`: flat schedule — backward compatible with static mode

### Component 1: `StochasticityScheduler`

**File:** `src/simulator/stochasticity_schedule.py`

```python
class StochasticityScheduler:
    def __init__(self, start: float, end: float, total_episodes: int):
        self.start = start
        self.end = end
        self.total = total_episodes
        self.current_value = start  # primed at start before episode 0

    def step(self, episode: int) -> float:
        """Advance schedule. Call once per episode (0-indexed). Returns new stochasticity."""
        t = min(episode, self.total - 1)
        import math
        self.current_value = (
            self.end + 0.5 * (self.start - self.end) * (1 + math.cos(math.pi * t / self.total))
        )
        return self.current_value
```

Responsibilities: schedule arithmetic only. No dependency on simulator or training loop.

### Component 2: Training Loop Integration

**File:** `src/training/training_loop.py`

**Constructor additions:**
```python
stochasticity: float = 0.5,         # static fallback (non-hybrid or no schedule)
stochasticity_start: float = None,  # None = use static stochasticity
stochasticity_end: float = None,    # None = use static stochasticity
```

**Scheduler construction** (after simulator creation in `__init__`):
```python
if simulator_type == "hybrid" and stochasticity_start is not None:
    self.stoch_scheduler = StochasticityScheduler(
        start=stochasticity_start,
        end=stochasticity_end,
        total_episodes=max_episodes,
    )
    self.simulator.stochasticity = stochasticity_start  # prime before episode 1
else:
    self.stoch_scheduler = None  # static mode
```

**Per-episode update** (top of `for episode in range(...)` loop in `run_training`):
```python
if self.stoch_scheduler is not None:
    self.simulator.stochasticity = self.stoch_scheduler.step(episode)
```

### Component 3: Logging

**`_save_learning_curves`:** Add `stochasticity_per_episode: [float, ...]` list alongside `episode_rewards`.

**`_print_progress_report`** (every 50 episodes): When scheduler is active, print:
```
Stochasticity:   0.623 → 0.412  (start → current)
```

**`_finalize_training` / `summary.json`:** Add optional block when scheduler active:
```json
"stochasticity_schedule": {
    "type": "cosine",
    "start": 0.8,
    "end": 0.2
}
```

### Component 4: Launch Script (`train.py`)

Two new `argparse` arguments alongside the existing `--stochasticity`:
```
--stochasticity_start 0.8
--stochasticity_end   0.2
```

Both default to `None`. When provided, they are passed to `HRLTrainingLoop` and logged to `summary.json`. The existing `--stochasticity 0.5` argument is retained and used as the constructor's static fallback when no schedule is active. Zero breaking changes.

---

## Integration Points Summary

| File | Change type | Size |
|---|---|---|
| `src/simulator/stochasticity_schedule.py` | New file | ~25 lines |
| `src/training/training_loop.py` | Additive | ~20 lines across 4 methods |
| `train.py` | Additive | ~6 lines |

No changes to `HybridSimulator`, `get_simulator`, or any other simulator.

---

## Success Criteria

1. At episode 0, `simulator.stochasticity == stochasticity_start`
2. At episode T-1, `simulator.stochasticity ≈ stochasticity_end` (within floating point)
3. `learning_curves_*.json` contains `stochasticity_per_episode` array of length == total episodes
4. Progress report at episode 50/100/150... prints current stochasticity when schedule is active
5. `summary.json` contains `stochasticity_schedule` block when schedule params are provided
6. Static mode (`--stochasticity 0.5`, no start/end): behaviour identical to current

---

## Backward Compatibility

- All existing experiments unaffected: `stochasticity_start=None` → scheduler is None → no change
- `stochasticity` parameter retained for static use (Sim8, StateMachine, or non-scheduled Hybrid)
- No changes to `HybridSimulator` internals — it already reads `self.stochasticity` per call
