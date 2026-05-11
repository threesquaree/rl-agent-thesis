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


def test_obs_dim_includes_trajectory():
    os.environ.pop("HRL_RESPONSE_TYPE_FEATURE", None)  # ensure default (off)
    env = make_env()
    focus_dim = env.n_exhibits + 1
    history_dim = env.n_exhibits + len([sa for opt in env.options for sa in env.subactions[opt]])
    expected_total = focus_dim + history_dim + 64 + 64 + 4 + 2  # +2 for τ_t
    assert env.observation_space.shape[0] == expected_total, (
        f"Expected {expected_total}, got {env.observation_space.shape[0]}"
    )


def test_trajectory_feature_values():
    env = make_env()
    env.reset()
    env.dwell = 0.75
    env._previous_dwell = 0.50
    env._last_delta_dwell = 0.25  # cache what step() would have stored
    obs = env._get_obs()
    dwell_norm = obs[-2]
    delta_dwell = obs[-1]
    assert abs(dwell_norm - (2.0 * 0.75 - 1.0)) < 1e-5, f"dwell_norm wrong: {dwell_norm}"
    assert abs(delta_dwell - 0.25) < 1e-5, f"delta_dwell wrong: {delta_dwell}"


def test_trajectory_feature_at_reset():
    env = make_env()
    obs, _ = env.reset()
    # After reset: dwell=0.0, _previous_dwell=0.0 → dwell_norm=-1.0, delta=0.0
    dwell_norm = obs[-2]
    delta_dwell = obs[-1]
    assert abs(dwell_norm - (2.0 * 0.0 - 1.0)) < 1e-5, f"dwell_norm at reset wrong: {dwell_norm}"
    assert abs(delta_dwell - 0.0) < 1e-5, f"delta_dwell at reset wrong: {delta_dwell}"


def test_trajectory_feature_upper_boundary():
    env = make_env()
    env.reset()
    env.dwell = 1.0
    env._previous_dwell = 0.5
    env._last_delta_dwell = 0.5  # cache what step() would have stored
    obs = env._get_obs()
    assert abs(obs[-2] - 1.0) < 1e-5   # dwell_norm = 2×1.0-1 = 1.0
    assert abs(obs[-1] - 0.5) < 1e-5   # delta = 1.0 - 0.5


def test_trajectory_feature_negative_delta():
    env = make_env()
    env.reset()
    env.dwell = 0.2
    env._previous_dwell = 0.8
    env._last_delta_dwell = -0.6  # cache what step() would have stored
    obs = env._get_obs()
    assert abs(obs[-2] - (2.0 * 0.2 - 1.0)) < 1e-5   # dwell_norm = -0.6
    assert abs(obs[-1] - (-0.6)) < 1e-5                # delta = 0.2 - 0.8


def test_delta_dwell_reads_from_cache_not_recomputed():
    """_get_obs() must read _last_delta_dwell, not recompute dwell - _previous_dwell.

    After step() runs, _previous_dwell == dwell, so recomputing gives 0.
    The cache holds the actual per-step delta — this test verifies _get_obs uses it.
    """
    env = make_env()
    env.reset()
    env.dwell = 0.7
    env._previous_dwell = 0.7   # equal: recomputing would give 0
    env._last_delta_dwell = 0.4  # cache holds what step() measured
    obs = env._get_obs()
    assert abs(obs[-1] - 0.4) < 1e-5, f"Expected 0.4 from cache, got {obs[-1]}"


def make_sim8():
    from src.simulator.sim8_adapter import Sim8Adapter
    return Sim8Adapter(exhibits=["exhibit_1"])


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
    assert all(0.10 <= d <= 1.0 for d in dwells)
    assert sum(dwells) / len(dwells) > 0.50


def test_recover_engagement_second_use_dwell_range():
    """2nd consecutive use: base × 0.70 → expected mean ~0.455, distinct from 1st (>0.50) and 3rd+ (<0.45)"""
    sim = make_sim8()
    sim.reset()
    dwells = []
    for _ in range(30):
        sim._consecutive_recover_count = 1  # always test the 2nd-use (×0.70) path
        gaze = sim._synthesize_contextual_gaze(
            rtype="statement",
            agent_option="Engage",
            agent_subaction="RecoverEngagement",
            engagement_adjust_multiplier=1.0,
        )
        dwells.append(gaze[0])
    mean_dwell = sum(dwells) / len(dwells)
    # base ∈ [0.55, 0.75] × 0.70 = [0.385, 0.525]; bounds chosen to be distinct from 1st-use (>0.50) and 3rd+ (<0.45)
    assert 0.30 < mean_dwell < 0.55, f"2nd-use mean dwell {mean_dwell:.3f} outside expected (0.30, 0.55)"


def test_recover_engagement_diminishing_returns():
    sim = make_sim8()
    sim.reset()
    sim._consecutive_recover_count = 2  # simulate 3rd+ use
    dwells = []
    for _ in range(30):
        gaze = sim._synthesize_contextual_gaze(
            rtype="statement",
            agent_option="Engage",
            agent_subaction="RecoverEngagement",
            engagement_adjust_multiplier=1.0,
        )
        dwells.append(gaze[0])
    assert sum(dwells) / len(dwells) < 0.45


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
