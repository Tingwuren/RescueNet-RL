"""High-capacity policy network for multimodal communication control."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class MultimodalPolicy(nn.Module):
    """Shared body actor-critic with >1M parameters for multi-modal scenarios."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (1024, 1024, 512, 512),
        activation: nn.Module = nn.ReLU,
        device: str = "cpu",
        min_parameter_budget: int = 1_000_000,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = list(hidden_sizes)
        self.device = torch.device(
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.body = self._build_body(obs_dim, activation)
        self.actor_head = nn.Linear(self.hidden_sizes[-1], action_dim)
        self.critic_head = nn.Linear(self.hidden_sizes[-1], 1)
        self.to(self.device)

        total_params = sum(p.numel() for p in self.parameters())
        if total_params < min_parameter_budget:
            raise ValueError(
                f"MultimodalPolicy requires >= {min_parameter_budget} parameters, "
                f"but only {total_params} were created. Increase hidden_sizes."
            )

    def _build_body(self, input_dim: int, activation: nn.Module) -> nn.Sequential:
        layers = []
        prev_dim = input_dim
        for width in self.hidden_sizes:
            layers.append(nn.Linear(prev_dim, width))
            layers.append(nn.LayerNorm(width))
            layers.append(activation())
            prev_dim = width
        return nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.body(obs)
        logits = self.actor_head(features)
        values = self.critic_head(features).squeeze(-1)
        return logits, values

    def act(self, obs, deterministic: bool = False) -> Tuple[int, float, float]:
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

    def get_value(self, obs) -> float:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, values = self.forward(obs_tensor)
        return float(values.squeeze(-1).item())
