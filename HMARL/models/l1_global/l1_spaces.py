"""
L1 全局统筹层：观测/动作空间定义与编解码。

观测（不直接使用子区域 32 维细粒度特征）：
  - 灾害类型 one-hot (3)
  - 全局网格摘要 (4)
  - 全局设备库存 (5)
  - 各区域灾情严重度 (N)
  - 各区域用户总数 (N)
  - 各区域高优先级用户占比 (N)

动作：
  - 连续松弛向量 dim = N * 5，解码为 N×5 整数配额矩阵 Q
  - 列和不超过全局库存，作为 L2/L3 硬约束
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class DisasterType(IntEnum):
    RAINSTORM_FLOOD = 0  # 暴雨
    TYPHOON_STORM = 1    # 台风风暴潮
    LANDSLIDE = 2        # 滑坡


class DeviceType(IntEnum):
    EMERGENCY_BS = 0           # 应急基站
    PORTABLE_BROADCAST_GW = 1  # 便携式广播通信网关
    RELAY_5G = 2               # 5G 中继
    RELAY_MESH = 3             # Mesh 中继
    UAV_COMM = 4               # 通信 UAV


DISASTER_NAMES = ["暴雨", "台风风暴潮", "滑坡"]
DEVICE_NAMES = [
    "应急基站",
    "便携式广播通信网关",
    "5G中继",
    "Mesh中继",
    "通信UAV",
]


@dataclass
class L1Config:
    n_regions: int = 5
    n_disaster_types: int = 3
    n_device_types: int = 5
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    log_std_init: float = 0.0

    @property
    def obs_dim(self) -> int:
        # 3 + 4 + 5 + 3*N
        return 12 + 3 * self.n_regions

    @property
    def action_dim(self) -> int:
        return self.n_regions * self.n_device_types


@dataclass
class L1GlobalState:
    """环境传入的 L1 全局状态（字典或本 dataclass）。"""

    disaster_type: int
    global_inventory: np.ndarray  # shape (5,)
    region_severity: np.ndarray   # shape (N,)
    region_user_count: np.ndarray # shape (N,)
    region_high_priority_ratio: np.ndarray  # shape (N,), in [0,1]
    num_regions: int = 5
    grid_rows: int = 24
    grid_cols: int = 24
    total_area_km2: float = 100.0
    region_area_ratio: Optional[np.ndarray] = None  # shape (N,), sum≈1

    def __post_init__(self):
        n = self.num_regions
        for name, arr in [
            ("global_inventory", self.global_inventory),
            ("region_severity", self.region_severity),
            ("region_user_count", self.region_user_count),
            ("region_high_priority_ratio", self.region_high_priority_ratio),
        ]:
            a = np.asarray(arr, dtype=np.float32).reshape(-1)
            if a.shape[0] != n and name != "global_inventory":
                raise ValueError(f"{name} length {a.shape[0]} != num_regions {n}")
        self.global_inventory = np.asarray(self.global_inventory, dtype=np.float32).reshape(5)
        self.region_severity = np.asarray(self.region_severity, dtype=np.float32).reshape(n)
        self.region_user_count = np.asarray(self.region_user_count, dtype=np.float32).reshape(n)
        self.region_high_priority_ratio = np.clip(
            np.asarray(self.region_high_priority_ratio, dtype=np.float32).reshape(n), 0.0, 1.0
        )
        if self.region_area_ratio is None:
            self.region_area_ratio = np.ones(n, dtype=np.float32) / n
        else:
            self.region_area_ratio = np.asarray(self.region_area_ratio, dtype=np.float32).reshape(n)
            s = self.region_area_ratio.sum()
            if s > 0:
                self.region_area_ratio = self.region_area_ratio / s


def encode_observation(state: L1GlobalState, cfg: L1Config) -> np.ndarray:
    """将全局状态编码为 L1 观测向量。"""
    n = cfg.n_regions
    disaster_oh = np.zeros(cfg.n_disaster_types, dtype=np.float32)
    d = int(np.clip(state.disaster_type, 0, cfg.n_disaster_types - 1))
    disaster_oh[d] = 1.0

    inv = state.global_inventory.astype(np.float32)
    inv_norm = inv / (inv.max() + 1e-6) if inv.max() > 0 else inv

    grid_summary = np.array(
        [
            float(n) / max(n, 1),
            state.grid_rows / 24.0,
            state.grid_cols / 24.0,
            min(state.total_area_km2 / 1000.0, 1.0),
        ],
        dtype=np.float32,
    )

    severity = state.region_severity[:n]
    users = state.region_user_count[:n]
    umax = users.max() + 1e-6
    users_norm = users / umax
    priority = state.region_high_priority_ratio[:n]

    obs = np.concatenate(
        [disaster_oh, grid_summary, inv_norm, severity, users_norm, priority]
    )
    assert obs.shape[0] == cfg.obs_dim, f"obs {obs.shape[0]} != {cfg.obs_dim}"
    return obs


def state_from_dict(raw: Dict[str, Any], cfg: L1Config) -> L1GlobalState:
    """从环境 raw_state 字典构造 L1GlobalState。"""
    n = cfg.n_regions
    return L1GlobalState(
        disaster_type=int(raw.get("disaster_type", 0)),
        global_inventory=np.asarray(
            raw.get(
                "global_inventory",
                raw.get("global_resource_total", np.zeros(5)),
            ),
            dtype=np.float32,
        ),
        region_severity=np.asarray(
            raw.get("region_severity", raw.get("global_disaster_distribution", np.zeros(n))),
            dtype=np.float32,
        ),
        region_user_count=np.asarray(
            raw.get("region_user_count", np.zeros(n)),
            dtype=np.float32,
        ),
        region_high_priority_ratio=np.asarray(
            raw.get("region_high_priority_ratio", np.full(n, 0.3)),
            dtype=np.float32,
        ),
        num_regions=n,
        grid_rows=int(raw.get("grid_rows", 24)),
        grid_cols=int(raw.get("grid_cols", 24)),
        total_area_km2=float(raw.get("total_area_km2", 100.0)),
        region_area_ratio=raw.get("region_area_ratio"),
    )


def decode_action_to_quota(
    action: np.ndarray,
    global_inventory: np.ndarray,
    cfg: L1Config,
) -> np.ndarray:
    """
    将 Actor 输出的连续动作解码为 N×5 整数配额矩阵 Q。

    对每一类设备 j：在 N 个区域上做 softmax，再按全局库存 G_j 分配整数，保证 sum_i Q_ij <= G_j。
    """
    n, m = cfg.n_regions, cfg.n_device_types
    G = np.asarray(global_inventory, dtype=np.float32).reshape(m)
    G = np.maximum(G, 0)

    logits = np.asarray(action, dtype=np.float32).reshape(n, m)
    Q = np.zeros((n, m), dtype=np.int32)

    for j in range(m):
        col = logits[:, j]
        w = np.exp(col - col.max())
        w = w / (w.sum() + 1e-8)
        total = int(round(G[j]))
        if total <= 0:
            continue
        raw_alloc = w * total
        floor_alloc = np.floor(raw_alloc).astype(np.int32)
        remainder = total - floor_alloc.sum()
        frac = raw_alloc - floor_alloc
        order = np.argsort(-frac)
        for k in range(int(remainder)):
            floor_alloc[order[k % n]] += 1
        Q[:, j] = floor_alloc

    return Q


def project_quota_to_inventory(
    Q: np.ndarray,
    global_inventory: np.ndarray,
) -> np.ndarray:
    """确保列和不超过库存（硬约束投影）。"""
    Q = Q.astype(np.int32).copy()
    G = np.maximum(np.asarray(global_inventory, dtype=np.int32).reshape(-1), 0)
    for j in range(Q.shape[1]):
        col_sum = Q[:, j].sum()
        cap = int(G[j])
        if col_sum > cap:
            excess = col_sum - cap
            for _ in range(excess):
                idx = np.argmax(Q[:, j])
                if Q[idx, j] > 0:
                    Q[idx, j] -= 1
    return Q


def quota_to_dict(Q: np.ndarray) -> Dict[str, Any]:
    """配额矩阵转为可读结构，供 L2/L3 使用。"""
    n, m = Q.shape
    return {
        "quota_matrix": Q.tolist(),
        "per_region": [
            {DEVICE_NAMES[j]: int(Q[i, j]) for j in range(m)}
            for i in range(n)
        ],
        "column_sums": {DEVICE_NAMES[j]: int(Q[:, j].sum()) for j in range(m)},
    }
