"""
L1 策略网络（Actor）：高斯策略，输出 N×5 维连续动作，经 l1_spaces 解码为配额矩阵。
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from ..common.mlp import MLP
from .l1_spaces import L1Config


class L1Actor(nn.Module):
    def __init__(self, cfg: L1Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = MLP(cfg.obs_dim, cfg.hidden_dims, cfg.action_dim)
        self.log_std = nn.Parameter(torch.full((cfg.action_dim,), cfg.log_std_init))

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

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_distribution(obs)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return mean
