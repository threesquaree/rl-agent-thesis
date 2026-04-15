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


def test_raises_for_single_episode():
    import pytest
    with pytest.raises(ValueError, match="total_episodes"):
        StochasticityScheduler(start=0.8, end=0.2, total_episodes=1)


def test_history_length_matches_episodes():
    """stochasticity_history should have one entry per episode."""
    s = StochasticityScheduler(start=0.8, end=0.2, total_episodes=10)
    history = [s.step(t) for t in range(10)]
    assert len(history) == 10
    assert abs(history[0] - s.start) < 1e-9
