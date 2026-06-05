"""L2 区域价值网络（Critic）。"""

import torch
import torch.nn as nn

from ..common.mlp import MLP
from .l2_spaces import L2Config


class L2Critic(nn.Module):
    def __init__(self, cfg: L2Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = MLP(cfg.obs_dim, cfg.hidden_dims, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs).squeeze(-1)
