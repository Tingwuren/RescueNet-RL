"""L3 价值网络（Critic）。"""

import torch
import torch.nn as nn

from ..common.mlp import MLP
from .l3_spaces import L3Config


class L3Critic(nn.Module):
    def __init__(self, cfg: L3Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = MLP(cfg.obs_dim, cfg.hidden_dims, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs).squeeze(-1)
