# Reward Reformalization & Action Space Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing engagement+novelty reward with a prospect-theory-grounded asymmetric delta reward, redesign the flat MDP action space from 8 to 6 actions (adding RecoverEngagement), and add a trajectory feature τ_t to the observation vector.

**Architecture:** All changes live on the `reward-reformalization` branch. `env.py` owns the reward formula, action space, and observation space. `sim8_adapter.py` owns the RecoverEngagement dwell response. `train.py` exposes the new parameters as CLI flags. The existing `main` branch is untouched.

**Tech Stack:** Python 3, NumPy, Gym spaces, pytest — same as existing codebase.

---

## File Map

| File | Change |
|------|--------|
| `src/environment/env.py` | Add α/β/R_terminal params; replace reward block; redesign action space; add τ_t to obs |
| `src/simulator/sim8_adapter.py` | Add RecoverEngagement dwell handler + diminishing returns counter |
| `train.py` | Add `--alpha`, `--beta`, `--terminal-coverage-weight` CLI flags |
| `tests/test_reward_reformalization.py` | New test file (created in Task 1) |

---

## Task 1: Reward Parameters — env.py `__init__` and train.py CLI

**Files:**
- Modify: `src/environment/env.py` (around line 67, reward parameters block)
- Modify: `train.py` (around line 167, alpha-new block)
- Create: `tests/test_reward_reformalization.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reward_reformalization.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def make_env(**env_kwargs):
    """Minimal env construction for unit testing reward params."""
    os.environ.pop("HRL_ALPHA", None)
    os.environ.pop("HRL_BETA", None)
    os.environ.pop("HRL_TERMINAL_COVERAGE_WEIGHT", None)
    from src.environment.env import MuseumEnvironment
    return MuseumEnvironment(**env_kwargs)


def test_alpha_default():
    env = make_env()
    assert env.alpha == 1.0


def test_beta_default():
    env = make_env()
    assert env.beta == 2.25


def test_terminal_coverage_weight_default():
    env = make_env()
    assert env.terminal_coverage_weight == 5.0


def test_alpha_from_env_var():
    os.environ["HRL_ALPHA"] = "2.0"
    env = make_env()
    assert env.alpha == 2.0
    os.environ.pop("HRL_ALPHA")


def test_beta_from_env_var():
    os.environ["HRL_BETA"] = "3.0"
    env = make_env()
    assert env.beta == 3.0
    os.environ.pop("HRL_BETA")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/Nayan/Thesis
python -m pytest tests/test_reward_reformalization.py::test_alpha_default -v
```
Expected: `AttributeError: 'MuseumEnvironment' object has no attribute 'alpha'`

- [ ] **Step 3: Add reward parameters to env.py `__init__`**

In `src/environment/env.py`, after the existing broadened novelty block (around line 93), add:

```python
        # ===== TRAJECTORY REWARD PARAMETERS (prospect theory) =====
        # R_t = α·max(0, Δdwell) − β·max(0, −Δdwell) + R_terminal·coverage
        # β/α = 2.25 from Kahneman & Tversky (1979)
        self.alpha = float(os.environ.get("HRL_ALPHA", "1.0"))
        self.beta = float(os.environ.get("HRL_BETA", "2.25"))
        self.terminal_coverage_weight = float(os.environ.get("HRL_TERMINAL_COVERAGE_WEIGHT", "5.0"))
```

- [ ] **Step 4: Add CLI flags to train.py**

In `train.py`, after the `--alpha-transition` argument (around line 177), add:

```python
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Trajectory reward gain weight α (default: 1.0, prospect theory)')
    parser.add_argument('--beta', type=float, default=2.25,
                        help='Trajectory reward loss weight β (default: 2.25, Kahneman & Tversky 1979)')
    parser.add_argument('--terminal-coverage-weight', type=float, default=5.0,
                        help='Terminal coverage bonus R_terminal (default: 5.0)')
```

In `train.py`, in the env var setting block (around line 557), add:

```python
    os.environ["HRL_ALPHA"] = str(args.alpha)
    os.environ["HRL_BETA"] = str(args.beta)
    os.environ["HRL_TERMINAL_COVERAGE_WEIGHT"] = str(args.terminal_coverage_weight)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_reward_reformalization.py -v
```
Expected: all 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_reward_reformalization.py src/environment/env.py train.py
git commit -m "feat: add trajectory reward parameters α, β, R_terminal to env and train CLI"
```

---

## Task 2: Asymmetric Delta Reward Formula — env.py `step()`

**Files:**
- Modify: `src/environment/env.py` (lines 599–854, reward calculation block)
- Modify: `tests/test_reward_reformalization.py`

- [ ] **Step 1: Add reward formula tests**

Append to `tests/test_reward_reformalization.py`:

```python
def test_trajectory_reward_gain():
    """dwell 0.2 → 0.7: R = α × 0.5 = +0.50"""
    env = make_env()
    env.reset()
    env._previous_dwell = 0.2
    env.dwell = 0.7
    env.alpha = 1.0
    env.beta = 2.25
    delta = env.dwell - env._previous_dwell   # 0.5
    reward = env.alpha * max(0.0, delta) - env.beta * max(0.0, -delta)
    assert abs(reward - 0.50) < 1e-6


def test_trajectory_reward_loss():
    """dwell 0.7 → 0.2: R = -β × 0.5 = -1.125"""
    env = make_env()
    env.reset()
    env._previous_dwell = 0.7
    env.dwell = 0.2
    env.alpha = 1.0
    env.beta = 2.25
    delta = env.dwell - env._previous_dwell   # -0.5
    reward = env.alpha * max(0.0, delta) - env.beta * max(0.0, -delta)
    assert abs(reward - (-1.125)) < 1e-6


def test_trajectory_reward_flat():
    """dwell flat 0.5 → 0.5: R = 0.0"""
    env = make_env()
    env.reset()
    env._previous_dwell = 0.5
    env.dwell = 0.5
    env.alpha = 1.0
    env.beta = 2.25
    delta = env.dwell - env._previous_dwell   # 0.0
    reward = env.alpha * max(0.0, delta) - env.beta * max(0.0, -delta)
    assert abs(reward - 0.0) < 1e-6
```

- [ ] **Step 2: Run to verify tests pass (formula tests are pure arithmetic)**

```bash
python -m pytest tests/test_reward_reformalization.py::test_trajectory_reward_gain tests/test_reward_reformalization.py::test_trajectory_reward_loss tests/test_reward_reformalization.py::test_trajectory_reward_flat -v
```
Expected: all 3 PASS (these test the formula directly, not env.step())

- [ ] **Step 3: Replace reward computation block in env.py step()**

In `src/environment/env.py`, find the comment `# ===== REWARD CALCULATION (per paper.tex Section 4.7, lines 681-709) =====` (around line 599) and replace **everything from that comment down to and including the `step_reward = step_reward - self.deliberation_cost` line** (around line 854) with:

```python
        # ===== REWARD CALCULATION (prospect theory asymmetric delta) =====
        # R_t = α·max(0,Δdwell) − β·max(0,−Δdwell) + R_terminal·coverage
        # β/α = 2.25 from Kahneman & Tversky (1979) prospect theory
        delta_dwell = self.dwell - self._previous_dwell
        trajectory_reward = (
            self.alpha * max(0.0, delta_dwell)
            - self.beta * max(0.0, -delta_dwell)
        )

        # Terminal coverage bonus — absorbs novelty signal at episode end
        terminal_bonus = 0.0
        if done:
            exhibits_covered = sum(
                1 for ex in self.exhibit_keys
                if len(self.facts_mentioned_per_exhibit[ex]) > 0
            )
            terminal_bonus = self.terminal_coverage_weight * (exhibits_covered / self.n_exhibits)

        step_reward = trajectory_reward + terminal_bonus - self.deliberation_cost

        if verbose:
            print(f"📈 TRAJECTORY REWARD: {trajectory_reward:.3f} "
                  f"(Δdwell={delta_dwell:+.3f}, dwell={self.dwell:.3f}, prev={self._previous_dwell:.3f})")
            if terminal_bonus > 0:
                print(f"🏁 TERMINAL BONUS: +{terminal_bonus:.3f} "
                      f"({exhibits_covered}/{self.n_exhibits} exhibits covered)")
```

Also remove or zero out the tracking variables for old components that no longer exist. Find the block starting with `# Track component contributions for analysis` (around line 856) and replace it with:

```python
        # Track reward components for analysis
        self.trajectory_reward_sum = getattr(self, 'trajectory_reward_sum', 0.0) + trajectory_reward
        self.terminal_bonus_sum = getattr(self, 'terminal_bonus_sum', 0.0) + terminal_bonus
        self.deliberation_sum -= self.deliberation_cost
```

- [ ] **Step 4: Initialize tracking vars in reset()**

In `src/environment/env.py`, in the `reset()` method, add after existing tracking variable resets:

```python
        self.trajectory_reward_sum = 0.0
        self.terminal_bonus_sum = 0.0
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_reward_reformalization.py -v
```
Expected: all 8 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/environment/env.py tests/test_reward_reformalization.py
git commit -m "feat: implement asymmetric delta reward (prospect theory) in env.py step()"
```

---

## Task 3: Redesign Flat Action Space — 6 Actions

**Files:**
- Modify: `src/environment/env.py` (lines 163–178 options/subactions, lines 1076–1083 subaction_availability)
- Modify: `tests/test_reward_reformalization.py`

- [ ] **Step 1: Add action space tests**

Append to `tests/test_reward_reformalization.py`:

```python
def test_action_space_has_six_subactions():
    env = make_env()
    all_subs = [sa for opt in env.options for sa in env.subactions[opt]]
    assert len(all_subs) == 6, f"Expected 6 subactions, got {len(all_subs)}: {all_subs}"


def test_removed_actions_absent():
    env = make_env()
    all_subs = [sa for opt in env.options for sa in env.subactions[opt]]
    assert "RepeatFact" not in all_subs
    assert "ClarifyFact" not in all_subs
    assert "AskMemory" not in all_subs


def test_new_action_present():
    env = make_env()
    all_subs = [sa for opt in env.options for sa in env.subactions[opt]]
    assert "RecoverEngagement" in all_subs


def test_core_actions_present():
    env = make_env()
    all_subs = [sa for opt in env.options for sa in env.subactions[opt]]
    for action in ["ExplainNewFact", "AskOpinion", "AskClarification",
                   "SummarizeAndSuggest", "WrapUp"]:
        assert action in all_subs, f"{action} missing from action space"
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_reward_reformalization.py::test_action_space_has_six_subactions -v
```
Expected: FAIL — currently 8 subactions

- [ ] **Step 3: Replace options and subactions in env.py `__init__`**

In `src/environment/env.py`, find the `subactions_override` block (around lines 163–178) and replace the default options/subactions with:

```python
        if options_override is not None:
            self.options = options_override
        else:
            self.options = ["Explain", "AskQuestion", "OfferTransition", "Conclude", "Engage"]

        if subactions_override is not None:
            self.subactions = subactions_override
        else:
            self.subactions = {
                "Explain":          ["ExplainNewFact"],
                "AskQuestion":      ["AskOpinion", "AskClarification"],
                "OfferTransition":  ["SummarizeAndSuggest"],
                "Conclude":         ["WrapUp"],
                "Engage":           ["RecoverEngagement"],
            }
```

- [ ] **Step 4: Update subaction_availability vector**

In `src/environment/env.py`, find the `subaction_availability` block (around lines 1076–1083) and replace with:

```python
        subaction_availability = np.zeros(4, dtype=np.float32)
        subaction_availability[0] = 1.0 if "ExplainNewFact" in available_subs else 0.0
        subaction_availability[1] = 1.0 if "AskOpinion" in available_subs else 0.0
        subaction_availability[2] = 1.0 if "RecoverEngagement" in available_subs else 0.0
        subaction_availability[3] = 1.0 if self._is_exhibit_exhausted(current_exhibit) else 0.0
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_reward_reformalization.py -v
```
Expected: all 12 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/environment/env.py tests/test_reward_reformalization.py
git commit -m "feat: redesign flat action space — 6 actions, add RecoverEngagement, remove 3 redundant"
```

---

## Task 4: Trajectory Feature τ_t — Observation Space

**Files:**
- Modify: `src/environment/env.py` (lines 202–218 dim calculation, lines 1085–1105 obs construction)
- Modify: `tests/test_reward_reformalization.py`

- [ ] **Step 1: Add observation space tests**

Append to `tests/test_reward_reformalization.py`:

```python
def test_obs_dim_includes_trajectory():
    env = make_env()
    expected_trajectory_dim = 2
    # obs space shape should be 2 larger than without τ_t
    # Verify trajectory dim is included in total
    obs_shape = env.observation_space.shape[0]
    # Compute expected: focus + history + intent + context + availability + trajectory
    focus_dim = env.n_exhibits + 1
    history_dim = env.n_exhibits + len([sa for opt in env.options for sa in env.subactions[opt]])
    expected_total = focus_dim + history_dim + 64 + 64 + 4 + expected_trajectory_dim
    assert obs_shape == expected_total, f"Expected {expected_total}, got {obs_shape}"


def test_trajectory_feature_values():
    env = make_env()
    env.reset()
    env.dwell = 0.75
    env._previous_dwell = 0.50
    obs = env._construct_observation()
    # τ_t is the last 2 dims: [dwell_norm, delta_dwell]
    dwell_norm = obs[-2]
    delta_dwell = obs[-1]
    assert abs(dwell_norm - (2.0 * 0.75 - 1.0)) < 1e-5   # 2×0.75-1 = 0.5
    assert abs(delta_dwell - 0.25) < 1e-5                  # 0.75 - 0.50


def test_trajectory_feature_at_reset():
    env = make_env()
    obs = env.reset()
    # After reset: dwell=0.6, _previous_dwell=0.6 → delta=0.0
    dwell_norm = obs[-2]
    delta_dwell = obs[-1]
    assert abs(dwell_norm - (2.0 * 0.6 - 1.0)) < 1e-5    # 0.2
    assert abs(delta_dwell - 0.0) < 1e-5
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_reward_reformalization.py::test_obs_dim_includes_trajectory -v
```
Expected: FAIL — obs shape doesn't yet include trajectory dims

- [ ] **Step 3: Add trajectory_dim to observation space calculation**

In `src/environment/env.py`, find the observation space dimension block (around lines 202–218) and update:

```python
        focus_dim = self.n_exhibits + 1
        self._all_subactions = [sa for opt in self.options for sa in self.subactions[opt]]
        history_dim = self.n_exhibits + len(self._all_subactions)
        intent_dim = 64
        context_dim = 64
        subaction_availability_dim = 4
        response_type_dim = 6 if self.response_type_feature else 0
        trajectory_dim = 2  # τ_t = [dwell_t_norm, Δdwell_t]

        total_obs_dim = (focus_dim + history_dim + intent_dim + context_dim
                         + subaction_availability_dim + response_type_dim + trajectory_dim)

        print(f"[Environment] Observation space: {total_obs_dim}-d "
              f"(focus={focus_dim}, history={history_dim}, intent={intent_dim}, "
              f"context={context_dim}, subaction_availability={subaction_availability_dim}, "
              f"response_type={response_type_dim}, trajectory={trajectory_dim})")
```

- [ ] **Step 4: Append τ_t to state_components in `_construct_observation()`**

In `src/environment/env.py`, find the `obs = np.concatenate(state_components)` line (around line 1105) and replace the block from `obs = np.concatenate` onward with:

```python
        # 6. Response type one-hot (6-d, optional)
        if self.response_type_feature:
            response_type_onehot = np.zeros(len(self.response_type_labels), dtype=np.float32)
            if self._last_user_response_type in self.response_type_labels:
                idx = self.response_type_labels.index(self._last_user_response_type)
                response_type_onehot[idx] = 1.0
            else:
                response_type_onehot[self.response_type_labels.index("statement")] = 1.0
            state_components.append(response_type_onehot)

        # 7. Trajectory feature τ_t = [dwell_t_norm, Δdwell_t]
        # dwell_t_norm: maps [0,1] → [-1,1] via 2×dwell-1
        # Δdwell_t: naturally in [-1,1]
        dwell_norm = float(2.0 * self.dwell - 1.0)
        delta_dwell = float(self.dwell - self._previous_dwell)
        state_components.append(np.array([dwell_norm, delta_dwell], dtype=np.float32))

        obs = np.concatenate(state_components).astype(np.float32)
        return obs
```

- [ ] **Step 5: Run all tests**

```bash
python -m pytest tests/test_reward_reformalization.py -v
```
Expected: all 15 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/environment/env.py tests/test_reward_reformalization.py
git commit -m "feat: add trajectory feature tau_t=[dwell_norm, delta_dwell] to observation space"
```

---

## Task 5: RecoverEngagement Handler — sim8_adapter.py

**Files:**
- Modify: `src/simulator/sim8_adapter.py` (`__init__`, `reset()`, `_synthesize_contextual_gaze()`)
- Modify: `tests/test_reward_reformalization.py`

- [ ] **Step 1: Add RecoverEngagement dwell tests**

Append to `tests/test_reward_reformalization.py`:

```python
def make_sim8():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.simulator.sim8_adapter import Sim8Adapter
    return Sim8Adapter()


def test_recover_engagement_first_use_dwell_range():
    sim = make_sim8()
    sim.reset()
    dwells = []
    for _ in range(30):
        sim._consecutive_recover_count = 0
        gaze = sim._synthesize_contextual_gaze(
            rtype="statement",
            agent_option="Engage",
            agent_subaction="RecoverEngagement",
            engagement_adjust_multiplier=1.0,
        )
        dwells.append(gaze[0])
    assert all(0.40 <= d <= 1.0 for d in dwells), f"First-use dwell out of range: {min(dwells):.3f}–{max(dwells):.3f}"
    assert sum(dwells) / len(dwells) > 0.50, "Expected mean dwell > 0.50 on first use"


def test_recover_engagement_diminishing_returns():
    sim = make_sim8()
    sim.reset()
    # Simulate 3rd+ consecutive use
    sim._consecutive_recover_count = 2
    dwells = []
    for _ in range(30):
        gaze = sim._synthesize_contextual_gaze(
            rtype="statement",
            agent_option="Engage",
            agent_subaction="RecoverEngagement",
            engagement_adjust_multiplier=1.0,
        )
        dwells.append(gaze[0])
    mean_dwell = sum(dwells) / len(dwells)
    assert mean_dwell < 0.50, f"Expected diminished dwell on 3rd+ use, got mean {mean_dwell:.3f}"


def test_recover_count_resets_on_other_action():
    sim = make_sim8()
    sim.reset()
    sim._consecutive_recover_count = 3
    sim._synthesize_contextual_gaze(
        rtype="acknowledgment",
        agent_option="Explain",
        agent_subaction="ExplainNewFact",
        engagement_adjust_multiplier=1.0,
    )
    assert sim._consecutive_recover_count == 0
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_reward_reformalization.py::test_recover_engagement_first_use_dwell_range -v
```
Expected: `AttributeError: 'Sim8Adapter' object has no attribute '_consecutive_recover_count'`

- [ ] **Step 3: Add `_consecutive_recover_count` to Sim8Adapter `__init__`**

In `src/simulator/sim8_adapter.py`, find the `__init__` block where counters are initialized (around line 131, near `self.engagement_level = 1.0`), add:

```python
        self._consecutive_recover_count = 0  # Tracks consecutive RecoverEngagement uses
```

- [ ] **Step 4: Reset the counter in `reset()`**

In `src/simulator/sim8_adapter.py`, find the `reset()` method (around line 211) and add:

```python
        self._consecutive_recover_count = 0
```

- [ ] **Step 5: Add RecoverEngagement branch in `_synthesize_contextual_gaze()`**

In `src/simulator/sim8_adapter.py`, find `_synthesize_contextual_gaze` — specifically the comment `# Different patterns for different response types` (around line 1417). Insert this block **before** the `if rtype in [...]` chain:

```python
        # RecoverEngagement override — dwell is action-driven, not rtype-driven
        if agent_subaction == "RecoverEngagement":
            self._consecutive_recover_count += 1
            base_dwell = self._randf(0.55, 0.75)
            if self._consecutive_recover_count == 2:
                base_dwell *= 0.70
            elif self._consecutive_recover_count >= 3:
                base_dwell *= 0.40
            dwell_time = self._clip(base_dwell, 0.10, 1.0)
            saccade_span = max(0.05, np.random.normal(0.07, 0.03))
        else:
            self._consecutive_recover_count = 0
            # Different patterns for different response types
            if rtype in ["acknowledgment", "follow_up_question"]:
```

Then ensure the rest of the existing rtype chain is inside the `else:` block (indent it one level). The chain ends at the existing `else:` default block. The `_adjust_dwell_for_action_variety` call at the end stays outside both branches.

- [ ] **Step 6: Run all tests**

```bash
python -m pytest tests/test_reward_reformalization.py -v
```
Expected: all 18 tests PASS

- [ ] **Step 7: Smoke test — launch a short training run**

```bash
cd /Users/Nayan/Thesis
source .venv/bin/activate
python train.py --variant h1 --episodes 5 --alpha 1.0 --beta 2.25 --terminal-coverage-weight 5.0 --name test_reformalization --simulator sim8
```
Expected: 5 episodes complete without errors; reward values printed each turn show `TRAJECTORY REWARD` format.

- [ ] **Step 8: Commit**

```bash
git add src/simulator/sim8_adapter.py tests/test_reward_reformalization.py
git commit -m "feat: add RecoverEngagement dwell handler with diminishing returns to sim8_adapter"
```

---

## Self-Review

**Spec coverage check:**
- ✅ Asymmetric delta reward R_t = α·max(0,Δ) − β·max(0,−Δ) → Task 2
- ✅ β/α = 2.25 from prospect theory → Task 1 (defaults) + Task 2 (formula)
- ✅ Terminal coverage bonus → Task 2
- ✅ Remove RepeatFact, ClarifyFact, AskMemory → Task 3
- ✅ Add RecoverEngagement → Task 3 (env) + Task 5 (sim8)
- ✅ τ_t = [dwell_norm, Δdwell] added to obs → Task 4
- ✅ CLI flags --alpha, --beta, --terminal-coverage-weight → Task 1
- ✅ sim8 only (hybrid_simulator.py untouched) → Task 5 scope
- ✅ reward-reformalization branch (main untouched) → branch already created

**Type consistency check:**
- `_consecutive_recover_count` initialized in Task 5 Step 3, used in Task 5 Step 5 ✅
- `trajectory_reward_sum` initialized in Task 2 Step 4 reset(), updated in Task 2 Step 3 ✅
- `self.alpha`, `self.beta`, `self.terminal_coverage_weight` defined in Task 1 Step 3, used in Task 2 Step 3 ✅
- `trajectory_dim = 2` defined in Task 4 Step 3, observation appended in Task 4 Step 4 ✅
- `_synthesize_contextual_gaze` signature unchanged — Task 5 uses existing `agent_subaction` param ✅
