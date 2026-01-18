"""Multi-head actor for scenario/reward-specific PPO (MPPO)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class MPPOPolicy(nn.Module):
    """Shared torso with multiple actor heads and a shared critic."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (1024, 1024, 512, 512),
        head_keys: Sequence[str] | None = None,
        active_head_key: str | None = None,
        activation: nn.Module = nn.ReLU,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = list(hidden_sizes)
        self.head_keys: List[str] = list(head_keys) if head_keys else ["default"]
        self.head_index: Dict[str, int] = {key: idx for idx, key in enumerate(self.head_keys)}
        self.active_head_key = active_head_key or (self.head_keys[0] if self.head_keys else "default")
        self.device = torch.device(
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        layers = []
        prev_dim = obs_dim
        for width in self.hidden_sizes:
            layers.append(nn.Linear(prev_dim, width))
            layers.append(nn.LayerNorm(width))
            layers.append(activation())
            prev_dim = width
        self.body = nn.Sequential(*layers)

        self.actor_heads = nn.ModuleList([nn.Linear(prev_dim, action_dim) for _ in self.head_keys])
        self.critic_head = nn.Linear(prev_dim, 1)
        self.to(self.device)

    def _select_head(self, head_key: str | None = None) -> nn.Linear:
        key = head_key or self.active_head_key
        if key not in self.head_index:
            key = self.head_keys[0]
        idx = self.head_index.get(key, 0)
        return self.actor_heads[idx]

    def set_active_head(self, head_key: str) -> None:
        if head_key in self.head_index:
            self.active_head_key = head_key

    def forward(self, obs: torch.Tensor, head_key: str | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.body(obs)
        logits = self._select_head(head_key)(features)
        values = self.critic_head(features).squeeze(-1)
        return logits, values

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, values = self.forward(obs_tensor)
        dist = Categorical(logits=logits)
        if deterministic:
            action_tensor = torch.argmax(logits, dim=-1)
        else:
            action_tensor = dist.sample()
        log_prob = dist.log_prob(action_tensor)
        return (
            int(action_tensor.item()),
            float(log_prob.item()),
            float(values.squeeze(-1).item()),
        )

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values

    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> float:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, values = self.forward(obs_tensor)
        return float(values.squeeze(-1).item())
