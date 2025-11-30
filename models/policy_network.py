"""MLP policy/value networks used by PPO."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class MLPActorCritic(nn.Module):
    """Simple shared-architecture actor-critic network."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: nn.Module = nn.Tanh,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = list(hidden_sizes)
        self.device = torch.device(
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.actor = self._build_mlp(obs_dim, action_dim, activation)
        self.critic = self._build_mlp(obs_dim, 1, activation)
        self.to(self.device)

    def _build_mlp(
        self,
        input_dim: int,
        output_dim: int,
        activation: nn.Module,
    ) -> nn.Sequential:
        layers: Iterable[nn.Module] = []
        prev_dim = input_dim
        for hidden in self.hidden_sizes:
            layers += [nn.Linear(prev_dim, hidden), activation()]
            prev_dim = hidden
        layers += [nn.Linear(prev_dim, output_dim)]
        return nn.Sequential(*layers)

    def act(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """Sample or select an action along with log-probability and value."""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.actor(obs_tensor)
        dist = Categorical(logits=logits)
        if deterministic:
            action_tensor = torch.argmax(logits, dim=-1)
        else:
            action_tensor = dist.sample()
        log_prob = dist.log_prob(action_tensor)
        value = self.critic(obs_tensor)
        return (
            int(action_tensor.item()),
            float(log_prob.item()),
            float(value.squeeze(-1).item()),
        )

    def get_value(self, obs: np.ndarray) -> float:
        """Return the critic value estimate for the given observation."""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        value = self.critic(obs_tensor)
        return float(value.squeeze(-1).item())

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-probs, entropy, and value estimates for PPO updates."""
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(obs).squeeze(-1)
        return log_probs, entropy, values
