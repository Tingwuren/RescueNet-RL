"""
L1 价值网络（Critic）：估计全局状态价值 V(s)。
"""

import torch
import torch.nn as nn

from ..common.mlp import MLP
from .l1_spaces import L1Config


class L1Critic(nn.Module):
    def __init__(self, cfg: L1Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = MLP(cfg.obs_dim, cfg.hidden_dims, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs).squeeze(-1)
