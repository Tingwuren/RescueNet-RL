"""Multi-head value (coverage/throughput/cost) actor-critic for A3C."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class A3CPolicy(nn.Module):
    """Actor with triple critics aggregated by configurable weights."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (1024, 1024, 512, 512),
        activation: nn.Module = nn.ReLU,
        device: str = "cpu",
        value_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = list(hidden_sizes)
        self.device = torch.device(
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.value_weights = value_weights or {"coverage": 1.0, "throughput": 0.5, "cost": 0.2}

        layers = []
        prev_dim = obs_dim
        for width in self.hidden_sizes:
            layers.append(nn.Linear(prev_dim, width))
            layers.append(nn.LayerNorm(width))
            layers.append(activation())
            prev_dim = width
        self.body = nn.Sequential(*layers)

        self.actor_head = nn.Linear(prev_dim, action_dim)
        self.coverage_value = nn.Linear(prev_dim, 1)
        self.throughput_value = nn.Linear(prev_dim, 1)
        self.cost_value = nn.Linear(prev_dim, 1)
        self.to(self.device)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.body(obs)
        logits = self.actor_head(features)
        v_cov = self.coverage_value(features).squeeze(-1)
        v_thr = self.throughput_value(features).squeeze(-1)
        v_cost = self.cost_value(features).squeeze(-1)
        v_aggr = self._aggregate_values(v_cov, v_thr, v_cost)
        return logits, v_cov, v_thr, v_cost, v_aggr

    def _aggregate_values(self, v_cov: torch.Tensor, v_thr: torch.Tensor, v_cost: torch.Tensor) -> torch.Tensor:
        w_cov = float(self.value_weights.get("coverage", 1.0))
        w_thr = float(self.value_weights.get("throughput", 0.5))
        w_cost = float(self.value_weights.get("cost", 0.2))
        return w_cov * v_cov + w_thr * v_thr - w_cost * v_cost

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, v_cov, v_thr, v_cost, v_aggr = self.forward(obs_tensor)
        dist = Categorical(logits=logits)
        if deterministic:
            action_tensor = torch.argmax(logits, dim=-1)
        else:
            action_tensor = dist.sample()
        log_prob = dist.log_prob(action_tensor)
        return (
            int(action_tensor.item()),
            float(log_prob.item()),
            float(v_aggr.squeeze(-1).item()),
        )

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, v_cov, v_thr, v_cost, v_aggr = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, v_aggr

    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> float:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, _, _, _, v_aggr = self.forward(obs_tensor)
        return float(v_aggr.squeeze(-1).item())

    def update_value_weights(self, value_weights: Dict[str, float]) -> None:
        self.value_weights = value_weights


# Backward compatibility alias.
N3CPolicy = A3CPolicy
