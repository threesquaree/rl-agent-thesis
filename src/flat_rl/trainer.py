from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


class FlatActorCriticTrainer:
    """
    TD(0) actor-critic trainer for the flat action policy.

    The public interface mirrors `ActorCriticTrainer` so the existing training
    loop can reuse the same call-sites.
    """

    def __init__(
        self,
        agent,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        entropy_decay_start: int = 0,
        entropy_decay_end: int = 500,
        entropy_final: float = 0.005,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
    ):
        self.agent = agent
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.value_loss_coef = value_loss_coef
        self.initial_entropy_coef = entropy_coef
        self.entropy_coef = entropy_coef
        self.entropy_decay_start = entropy_decay_start
        self.entropy_decay_end = entropy_decay_end
        self.entropy_final = entropy_final
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.optimizer = optim.Adam(self.agent.network.parameters(), lr=learning_rate)
        self.stats = defaultdict(list)

    def set_episode(self, episode: int):
        """Linearly decay entropy coefficient from initial to entropy_final."""
        if episode < self.entropy_decay_start or self.entropy_decay_end <= self.entropy_decay_start:
            return
        if episode >= self.entropy_decay_end:
            self.entropy_coef = self.entropy_final
            return
        progress = (episode - self.entropy_decay_start) / (self.entropy_decay_end - self.entropy_decay_start)
        self.entropy_coef = self.initial_entropy_coef * (1 - progress) + self.entropy_final * progress

    def update(
        self,
        states: List[np.ndarray],
        options: List[int],
        subactions: List[int],
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[bool],
    ) -> Dict[str, float]:
        # Convert data to tensors
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        dones_t = torch.FloatTensor([1.0 if d else 0.0 for d in dones]).to(self.device)

        flat_indices = [
            self.agent.map_option_subaction_to_flat(opt_idx, sub_idx)
            for opt_idx, sub_idx in zip(options, subactions)
        ]
        actions_t = torch.LongTensor(flat_indices).to(self.device)

        # Reset hidden state before batching
        self.agent.network.reset_hidden_state()
        outputs = self.agent.network.forward(states_t)

        with torch.no_grad():
            self.agent.network.reset_hidden_state()
            next_outputs = self.agent.network.forward(next_states_t)
            next_values = next_outputs["state_value"]

        values = outputs["state_value"]
        targets = rewards_t + self.gamma * next_values * (1.0 - dones_t)
        value_loss = F.mse_loss(values, targets.detach())

        advantages = (targets - values).detach()

        logits = outputs["action_logits"]
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        policy_loss = -(selected_log_probs * advantages).mean()

        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        total_loss = (
            self.value_loss_coef * value_loss
            + policy_loss
            - self.entropy_coef * entropy
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Compute per-action advantages for collapse analysis
        # For each action, compute advantage as Q(s,a) - V(s)
        # Q(s,a) = targets (TD target), V(s) = values
        # For all actions, we need action values - but we only have state values
        # So we'll track: selected action advantages and action logits (which indicate preference)
        action_logits_all = logits  # Shape: (batch_size, num_actions)
        # Compute action values from logits (softmax gives probabilities, but we need Q-values)
        # For now, track logits as proxy for action values, and selected advantages
        selected_advantages = advantages  # Advantages for selected actions
        
        stats = {
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "mean_advantage": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
            # Per-action tracking for collapse analysis
            "action_advantages_all": selected_advantages.cpu().detach().numpy().tolist(),
            "action_logits_all": action_logits_all.cpu().detach().numpy().tolist(),  # Proxy for action preferences
            "state_values": values.cpu().detach().numpy().tolist()
        }

        for k, v in stats.items():
            self.stats[k].append(v)

        return stats

    def save_checkpoint(self, path: str, episode: int):
        torch.save(
            {
                "episode": episode,
                "agent_state": self.agent.network.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "stats": dict(self.stats),
            },
            path,
        )


__all__ = ["FlatActorCriticTrainer"]


