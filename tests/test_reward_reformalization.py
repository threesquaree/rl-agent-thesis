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
