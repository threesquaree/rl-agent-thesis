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
        if total_episodes < 2:
            raise ValueError(f"total_episodes must be >= 2, got {total_episodes}")
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
        # Normalize t to [0, 1] range. At t=total-1, we should get end value.
        # Using progress parameter: when t=0 → progress=0 (cos(0)=1, gives start)
        # when t=total-1 → progress=1 (cos(π)=-1, gives end)
        progress = t / (self.total - 1) if self.total > 1 else 1.0
        self.current_value = (
            self.end
            + 0.5 * (self.start - self.end) * (1 + math.cos(math.pi * progress))
        )
        return self.current_value
