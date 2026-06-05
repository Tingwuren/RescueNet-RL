"""L2 区域策略网络（Actor）。"""

from typing import Tuple

import torch
from torch.distributions import Normal

from ..common.mlp import MLP
from .l2_spaces import L2Config


class L2Actor(torch.nn.Module):
    def __init__(self, cfg: L2Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = MLP(cfg.obs_dim, cfg.hidden_dims, cfg.action_dim)
        self.log_std = torch.nn.Parameter(
            torch.full((cfg.action_dim,), cfg.log_std_init)
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.backbone(obs)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        mean, std = self.forward(obs)
        return Normal(mean, std)

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_distribution(obs)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return mean
