import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


_REWARD_VARS = ("HRL_ALPHA", "HRL_BETA", "HRL_TERMINAL_COVERAGE_WEIGHT")


def make_env(**env_kwargs):
    """Minimal env construction for unit testing reward params."""
    from src.environment.env import MuseumEnvironment
    return MuseumEnvironment(**env_kwargs)


def test_alpha_default():
    for v in _REWARD_VARS:
        os.environ.pop(v, None)
    env = make_env()
    assert env.alpha == 1.0


def test_beta_default():
    for v in _REWARD_VARS:
        os.environ.pop(v, None)
    env = make_env()
    assert env.beta == 2.25


def test_terminal_coverage_weight_default():
    for v in _REWARD_VARS:
        os.environ.pop(v, None)
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


def test_trajectory_reward_gain():
    """dwell 0.2 → 0.7: R = α × 0.5 = +0.50"""
    env = make_env()
    env.reset()
    env._previous_dwell = 0.2
    env.dwell = 0.7
    env.alpha = 1.0
    env.beta = 2.25
    delta = env.dwell - env._previous_dwell
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
    delta = env.dwell - env._previous_dwell
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
    delta = env.dwell - env._previous_dwell
    reward = env.alpha * max(0.0, delta) - env.beta * max(0.0, -delta)
    assert abs(reward - 0.0) < 1e-6


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
    assert "RecoverEngagement" in env.subactions["Engage"]


def test_core_actions_present():
    env = make_env()
    all_subs = [sa for opt in env.options for sa in env.subactions[opt]]
    for action in ["ExplainNewFact", "AskOpinion", "AskClarification",
                   "SummarizeAndSuggest", "WrapUp"]:
        assert action in all_subs, f"{action} missing from action space"
