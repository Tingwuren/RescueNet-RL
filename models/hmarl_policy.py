"""Hierarchical actor-critic policy for HMARL resource orchestration."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class HMARLPolicy(nn.Module):
    """Shared encoder with L1/L2 auxiliary heads and an L3 action head."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (768, 512, 256),
        l1_regions: int = 9,
        l2_link_types: int = 4,
        activation: nn.Module = nn.ReLU,
        device: str = "cpu",
        prior_weight: float = 1.25,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = list(hidden_sizes)
        self.l1_regions = int(l1_regions)
        self.l2_link_types = int(l2_link_types)
        self.prior_weight = float(prior_weight)
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
        self.l1_head = nn.Linear(prev_dim, self.l1_regions)
        self.l2_head = nn.Linear(prev_dim, self.l2_link_types)
        self.l3_actor_head = nn.Linear(prev_dim, action_dim)
        self.critic_head = nn.Linear(prev_dim, 1)
        self.to(self.device)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.body(obs)
        l1_logits = self.l1_head(features)
        l2_logits = self.l2_head(features)
        l3_logits = self.l3_actor_head(features)
        values = self.critic_head(features).squeeze(-1)
        return l3_logits, values, l1_logits, l2_logits

    def _apply_prior(self, logits: torch.Tensor, action_prior: torch.Tensor | None) -> torch.Tensor:
        if action_prior is None:
            return logits
        prior = action_prior.to(device=logits.device, dtype=logits.dtype)
        if prior.ndim == 1:
            prior = prior.unsqueeze(0)
        if prior.shape[-1] != logits.shape[-1]:
            return logits
        prior = torch.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=-1e9)
        return logits + self.prior_weight * prior

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        action_prior: np.ndarray | None = None,
    ) -> Tuple[int, float, float]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        prior_tensor = (
            torch.as_tensor(action_prior, dtype=torch.float32, device=self.device).unsqueeze(0)
            if action_prior is not None
            else None
        )
        logits, values, l1_logits, l2_logits = self.forward(obs_tensor)
        logits = self._apply_prior(logits, prior_tensor)
        dist = Categorical(logits=logits)
        if deterministic:
            action_tensor = torch.argmax(logits, dim=-1)
        else:
            action_tensor = dist.sample()
        log_prob = dist.log_prob(action_tensor)
        self.last_decision: Dict[str, int] = {
            "l1_region": int(torch.argmax(l1_logits, dim=-1).item()),
            "l2_link_type": int(torch.argmax(l2_logits, dim=-1).item()),
            "l3_action": int(action_tensor.item()),
        }
        return (
            int(action_tensor.item()),
            float(log_prob.item()),
            float(values.squeeze(-1).item()),
        )

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs, entropy, values, _, _ = self.evaluate_actions_with_prior(obs, actions)
        return log_probs, entropy, values

    def evaluate_actions_with_prior(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_prior: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values, l1_logits, l2_logits = self.forward(obs)
        logits = self._apply_prior(logits, action_prior)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values, l1_logits, l2_logits

    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> float:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, values, _, _ = self.forward(obs_tensor)
        return float(values.squeeze(-1).item())
