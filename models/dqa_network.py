"""Q-network for decomposed/discrete action control."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


class DQANetwork(nn.Module):
    """Feed-forward Q-network with epsilon-greedy action selection."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: nn.Module = nn.ReLU,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = list(hidden_sizes)
        self.device = torch.device(
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        layers = []
        prev = obs_dim
        for width in self.hidden_sizes:
            layers.extend([nn.Linear(prev, width), activation()])
            prev = width
        layers.append(nn.Linear(prev, action_dim))
        self.q_net = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_net(obs)

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: float = 0.0, deterministic: bool = False) -> Tuple[int, float]:
        """Return an action index and max-Q value."""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.q_net(obs_tensor)
        if deterministic or torch.rand(1).item() > epsilon:
            action_tensor = torch.argmax(q_values, dim=-1)
        else:
            action_tensor = torch.randint(0, self.action_dim, (1,), device=self.device)
        q_max = float(torch.max(q_values).item())
        return int(action_tensor.item()), q_max

    def hard_update(self, target_net: "DQANetwork") -> None:
        target_net.load_state_dict(self.state_dict())

    def soft_update(self, target_net: "DQANetwork", tau: float) -> None:
        with torch.no_grad():
            for target_param, param in zip(target_net.parameters(), self.parameters()):
                target_param.data.mul_(1.0 - tau).add_(param.data * tau)
