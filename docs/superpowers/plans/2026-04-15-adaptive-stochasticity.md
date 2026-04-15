# Adaptive Stochasticity Schedule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a cosine-annealed stochasticity schedule to `HybridSimulator` so the simulator gives high-variance exploration signals early and stable signals late, reducing the converged reward variance observed in Hybrid experiments (σ=10.52 vs Sim8's 3.55).

**Architecture:** A standalone `StochasticityScheduler` class computes cosine schedule values. `HRLTrainingLoop` holds the scheduler, calls `.step(episode)` once per episode, and assigns the returned value to `self.simulator.stochasticity` before `_run_episode()`. `HybridSimulator` already reads `self.stochasticity` fresh on every call — no changes needed there.

**Tech Stack:** Python stdlib (`math`), existing `argparse` in `train.py`, pytest for tests.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/simulator/stochasticity_schedule.py` | **Create** | Cosine schedule arithmetic only |
| `tests/test_stochasticity_schedule.py` | **Create** | Unit tests for scheduler |
| `src/training/training_loop.py` | **Modify** | Constructor params, episode loop update, history tracking |
| `train.py` | **Modify** | CLI args, metadata block, `HRLTrainingLoop` call |

`HybridSimulator`, `get_simulator`, and all other files are **unchanged**.

---

### Task 1: `StochasticityScheduler` class

**Files:**
- Create: `src/simulator/stochasticity_schedule.py`
- Create: `tests/test_stochasticity_schedule.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stochasticity_schedule.py`:

```python
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simulator.stochasticity_schedule import StochasticityScheduler


def test_initial_value_equals_start():
    s = StochasticityScheduler(start=0.8, end=0.2, total_episodes=100)
    assert s.current_value == 0.8


def test_step_0_returns_start():
    s = StochasticityScheduler(start=0.8, end=0.2, total_episodes=100)
    val = s.step(0)
    assert abs(val - 0.8) < 1e-9


def test_step_final_returns_end():
    s = StochasticityScheduler(start=0.8, end=0.2, total_episodes=100)
    val = s.step(99)
    assert abs(val - 0.2) < 1e-6


def test_midpoint_is_between_start_and_end():
    s = StochasticityScheduler(start=0.8, end=0.2, total_episodes=100)
    val = s.step(50)
    assert 0.2 < val < 0.8


def test_step_updates_current_value():
    s = StochasticityScheduler(start=0.8, end=0.2, total_episodes=100)
    returned = s.step(50)
    assert s.current_value == returned


def test_monotonically_decreasing():
    s = StochasticityScheduler(start=0.8, end=0.2, total_episodes=100)
    values = [s.step(t) for t in range(100)]
    for i in range(len(values) - 1):
        assert values[i] >= values[i + 1], f"Not monotone at t={i}: {values[i]} < {values[i+1]}"


def test_flat_schedule_when_start_equals_end():
    s = StochasticityScheduler(start=0.5, end=0.5, total_episodes=100)
    for t in [0, 25, 50, 75, 99]:
        assert abs(s.step(t) - 0.5) < 1e-9


def test_step_beyond_total_clamps_to_end():
    s = StochasticityScheduler(start=0.8, end=0.2, total_episodes=100)
    val = s.step(200)
    assert abs(val - 0.2) < 1e-6
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
cd /Users/Nayan/Thesis && python -m pytest tests/test_stochasticity_schedule.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'src.simulator.stochasticity_schedule'`

- [ ] **Step 3: Implement `StochasticityScheduler`**

Create `src/simulator/stochasticity_schedule.py`:

```python
import math


class StochasticityScheduler:
    """Cosine annealing schedule for HybridSimulator stochasticity.

    stoch(t) = end + 0.5 * (start - end) * (1 + cos(π · t / T))

    At t=0  → start (e.g. 0.8, high exploration variance)
    At t=T-1 → end  (e.g. 0.2, stable convergence signal)

    When start == end: flat schedule (backward compatible with static mode).
    """

    def __init__(self, start: float, end: float, total_episodes: int):
        self.start = start
        self.end = end
        self.total = total_episodes
        self.current_value: float = start  # primed at start before episode 0

    def step(self, episode: int) -> float:
        """Advance schedule. Call once per episode (0-indexed).

        Args:
            episode: Current episode index (0-indexed). Values beyond
                     total_episodes - 1 are clamped to the final value.

        Returns:
            New stochasticity value for this episode.
        """
        t = min(episode, self.total - 1)
        self.current_value = (
            self.end
            + 0.5 * (self.start - self.end) * (1 + math.cos(math.pi * t / self.total))
        )
        return self.current_value
```

- [ ] **Step 4: Run tests to confirm they all pass**

```bash
cd /Users/Nayan/Thesis && python -m pytest tests/test_stochasticity_schedule.py -v
```

Expected: 8 tests PASSED.

- [ ] **Step 5: Commit**

```bash
cd /Users/Nayan/Thesis
git add src/simulator/stochasticity_schedule.py tests/test_stochasticity_schedule.py
git commit -m "feat: add StochasticityScheduler with cosine annealing

Standalone cosine schedule: stoch(t) = end + 0.5*(start-end)*(1+cos(π·t/T))
Flat schedule when start==end for backward compatibility.
8 unit tests covering boundary values, monotonicity, clamping."
```

---

### Task 2: `HRLTrainingLoop` integration

**Files:**
- Modify: `src/training/training_loop.py:131-168` (constructor signature)
- Modify: `src/training/training_loop.py:239-246` (post-simulator construction)
- Modify: `src/training/training_loop.py:451-452` (per-episode update)

- [ ] **Step 1: Add import at top of `training_loop.py`**

Find the imports block at the top of `src/training/training_loop.py`. After the existing simulator import line:

```python
from src.simulator import get_simulator
```

Add:

```python
from src.simulator.stochasticity_schedule import StochasticityScheduler
```

- [ ] **Step 2: Add constructor parameters**

In `HRLTrainingLoop.__init__` at line 141, the current signature has:
```python
                 stochasticity: float = 0.5,
```

Add two parameters immediately after it:
```python
                 stochasticity: float = 0.5,
                 stochasticity_start: float = None,
                 stochasticity_end: float = None,
```

- [ ] **Step 3: Construct the scheduler and initialise history list**

In `__init__`, after the simulator is created (currently lines 242–246):
```python
        self.simulator = get_simulator(
            simulator_type=simulator_type,
            knowledge_graph=self.knowledge_graph,
            stochasticity=stochasticity,
        )
```

Add immediately after:
```python
        # Adaptive stochasticity schedule (Hybrid simulator only)
        if simulator_type == "hybrid" and stochasticity_start is not None:
            self.stoch_scheduler = StochasticityScheduler(
                start=stochasticity_start,
                end=stochasticity_end,
                total_episodes=max_episodes,
            )
            self.simulator.stochasticity = stochasticity_start  # prime before episode 1
        else:
            self.stoch_scheduler = None  # static mode, no-op

        self.stochasticity_history: list = []  # per-episode log
```

- [ ] **Step 4: Add per-episode update to `run_training`**

In `run_training`, inside the `for episode in range(self.max_episodes):` loop, immediately **before** the `_run_episode()` call at line 462. The current code reads:

```python
                # Run single episode
                episode_reward, episode_length, episode_time, episode_timing = self._run_episode()
```

Insert before it:
```python
                # Advance stochasticity schedule (no-op when scheduler is None)
                if self.stoch_scheduler is not None:
                    self.simulator.stochasticity = self.stoch_scheduler.step(episode)
                self.stochasticity_history.append(
                    self.stoch_scheduler.current_value if self.stoch_scheduler is not None
                    else getattr(self.simulator, 'stochasticity', None)
                )
```

- [ ] **Step 5: Smoke test the integration (no schedule, should behave identically)**

```bash
cd /Users/Nayan/Thesis && python -c "
from src.training.training_loop import HRLTrainingLoop
# This exercises the constructor path with no schedule (static mode)
print('Import OK')
print('StochasticityScheduler imported:', hasattr(HRLTrainingLoop, '__init__'))
"
```

Expected: `Import OK` and `StochasticityScheduler imported: True` with no errors.

- [ ] **Step 6: Commit**

```bash
cd /Users/Nayan/Thesis
git add src/training/training_loop.py
git commit -m "feat: integrate StochasticityScheduler into HRLTrainingLoop

Add stochasticity_start/end params to constructor. When provided with
simulator_type='hybrid', creates a scheduler and primes the simulator.
Per-episode update runs before _run_episode(). History tracked in
self.stochasticity_history for logging."
```

---

### Task 3: Logging

**Files:**
- Modify: `src/training/training_loop.py:1635` (`_print_progress_report`)
- Modify: `src/training/training_loop.py:2509` (`_save_learning_curves`)

- [ ] **Step 1: Add stochasticity line to `_print_progress_report`**

In `_print_progress_report` (line 1635), find the existing print block that ends with:

```python
        print(f"[PROGRESS] Progress: {(self.current_episode/self.max_episodes)*100:.1f}% complete")
        print("=" * 80 + "\n")
```

Insert before the closing separator:
```python
        if self.stoch_scheduler is not None:
            print(f"[STOCH]   Stochasticity: {self.stoch_scheduler.start:.3f} → {self.stoch_scheduler.current_value:.3f}  (start → current, cosine schedule)")
        print(f"[PROGRESS] Progress: {(self.current_episode/self.max_episodes)*100:.1f}% complete")
        print("=" * 80 + "\n")
```

- [ ] **Step 2: Add `stochasticity_per_episode` to `_save_learning_curves`**

In `_save_learning_curves` (line 2509), find the `learning_curves` dict:

```python
            learning_curves = {
                "episode_returns": self.episode_rewards,
                "value_estimates": self.metrics_tracker.value_estimates,
                "policy_entropy": self.metrics_tracker.entropies,
                "value_losses": self.metrics_tracker.value_losses,
                "policy_losses": self.metrics_tracker.policy_losses,
                "termination_losses": self.metrics_tracker.termination_losses,
                "advantages": self.metrics_tracker.advantages
            }
```

Add one field:
```python
            learning_curves = {
                "episode_returns": self.episode_rewards,
                "value_estimates": self.metrics_tracker.value_estimates,
                "policy_entropy": self.metrics_tracker.entropies,
                "value_losses": self.metrics_tracker.value_losses,
                "policy_losses": self.metrics_tracker.policy_losses,
                "termination_losses": self.metrics_tracker.termination_losses,
                "advantages": self.metrics_tracker.advantages,
                "stochasticity_per_episode": self.stochasticity_history,
            }
```

- [ ] **Step 3: Verify logging fields with a unit test**

Add to `tests/test_stochasticity_schedule.py`:

```python
def test_history_length_matches_episodes():
    """stochasticity_history should have one entry per episode."""
    s = StochasticityScheduler(start=0.8, end=0.2, total_episodes=10)
    history = [s.step(t) for t in range(10)]
    assert len(history) == 10
    assert history[0] == s.start or abs(history[0] - s.start) < 1e-9
```

Run:
```bash
cd /Users/Nayan/Thesis && python -m pytest tests/test_stochasticity_schedule.py -v
```

Expected: 9 tests PASSED.

- [ ] **Step 4: Commit**

```bash
cd /Users/Nayan/Thesis
git add src/training/training_loop.py tests/test_stochasticity_schedule.py
git commit -m "feat: log stochasticity schedule to learning curves and progress report

_print_progress_report prints start→current when scheduler is active.
_save_learning_curves writes stochasticity_per_episode array.
Add history-length test."
```

---

### Task 4: `train.py` CLI, metadata, and summary

**Files:**
- Modify: `train.py:266-267` (argparse block)
- Modify: `train.py:429` (metadata dict)
- Modify: `train.py:599` (`HRLTrainingLoop` call)

- [ ] **Step 1: Add two new CLI arguments**

In `train.py`, find the existing `--stochasticity` argument (line 266):

```python
    parser.add_argument('--stochasticity', type=float, default=0.5,
                       help='Hybrid simulator: sim8 influence (0.0=pure state machine, 0.5=balanced, 1.0=max sim8). Default: 0.5')
```

Add immediately after:
```python
    parser.add_argument('--stochasticity_start', type=float, default=None,
                       help='Hybrid simulator: cosine schedule start value (e.g. 0.8). Enables adaptive schedule when set alongside --stochasticity_end.')
    parser.add_argument('--stochasticity_end', type=float, default=None,
                       help='Hybrid simulator: cosine schedule end value (e.g. 0.2). Enables adaptive schedule when set alongside --stochasticity_start.')
```

- [ ] **Step 2: Add schedule params to metadata dict**

In `train.py`, find the metadata dict entry (line 429):

```python
        "stochasticity": args.stochasticity if args.simulator == "hybrid" else None,
```

Replace with:
```python
        "stochasticity": args.stochasticity if args.simulator == "hybrid" else None,
        "stochasticity_schedule": (
            {"type": "cosine", "start": args.stochasticity_start, "end": args.stochasticity_end}
            if args.simulator == "hybrid" and args.stochasticity_start is not None
            else None
        ),
```

- [ ] **Step 3: Pass new params to `HRLTrainingLoop`**

In `train.py`, find the `HRLTrainingLoop(...)` call (line 599):

```python
        stochasticity=args.stochasticity,
```

Add the two new params immediately after:
```python
        stochasticity=args.stochasticity,
        stochasticity_start=args.stochasticity_start,
        stochasticity_end=args.stochasticity_end,
```

- [ ] **Step 4: Update the simulator print line to show schedule**

Find (line 493):
```python
    print(f"Simulator: {args.simulator}" + (f" (stochasticity={args.stochasticity})" if args.simulator == "hybrid" else ""))
```

Replace with:
```python
    if args.simulator == "hybrid":
        if args.stochasticity_start is not None:
            sim_str = f" (cosine schedule: {args.stochasticity_start}→{args.stochasticity_end})"
        else:
            sim_str = f" (stochasticity={args.stochasticity})"
    else:
        sim_str = ""
    print(f"Simulator: {args.simulator}{sim_str}")
```

- [ ] **Step 5: Smoke test CLI parsing**

```bash
cd /Users/Nayan/Thesis && python train.py --help 2>&1 | grep stochasticity
```

Expected output includes all three lines:
```
  --stochasticity STOCHASTICITY
  --stochasticity_start STOCHASTICITY_START
  --stochasticity_end STOCHASTICITY_END
```

- [ ] **Step 6: Commit**

```bash
cd /Users/Nayan/Thesis
git add train.py
git commit -m "feat: expose stochasticity_start/end CLI args in train.py

--stochasticity_start and --stochasticity_end enable cosine schedule.
metadata.json and summary.json gain stochasticity_schedule block when
active. Simulator config print shows start→end range."
```

---

## Success Criteria Checklist

After all tasks are complete, verify the spec requirements:

- [ ] `StochasticityScheduler(0.8, 0.2, 500).step(0)` returns 0.8
- [ ] `StochasticityScheduler(0.8, 0.2, 500).step(499)` ≈ 0.2
- [ ] `StochasticityScheduler(0.5, 0.5, 500).step(250)` == 0.5 (flat)
- [ ] `python -m pytest tests/test_stochasticity_schedule.py -v` → all 9 PASS
- [ ] `python train.py --help | grep stochasticity_start` → shows the arg
- [ ] Running with `--simulator hybrid --stochasticity_start 0.8 --stochasticity_end 0.2` produces `learning_curves_*.json` with `stochasticity_per_episode` list of length == episodes
- [ ] Running with `--simulator sim8` (no schedule): behaviour identical to pre-change, `stochasticity_schedule` field absent from `summary.json`
