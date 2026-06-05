"""
L2 区域调控层：观测/动作空间与编解码。

每个区域一个 L2 智能体；观测为区域聚合特征 + 邻居通信摘要 + L1 配额。
动作解码为：资源迁移指令 (M×[src,tgt,5])、跨区域链路 (K×[A,B,type,pos])。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 与 L1 设备类型一致
DEVICE_NAMES = [
    "应急基站",
    "便携式广播通信网关",
    "5G中继",
    "Mesh中继",
    "通信UAV",
]
N_DEVICES = 5


class CrossRegionLinkType(IntEnum):
    SATELLITE_BACKHAUL = 0  # 卫星回传链路
    MICROWAVE_RELAY = 1     # 微波中继链路
    UAV_RELAY = 2           # UAV 中继链路


LINK_TYPE_NAMES = ["卫星回传链路", "微波中继链路", "UAV中继链路"]


@dataclass
class L2Config:
    n_regions: int = 5
    max_migrations: int = 3       # M：每区域最多提出 M 条迁出任务
    max_links: int = 2            # K：每区域最多提出 K 条跨区链路
    max_neighbors: int = 4        # 通信邻居上限
    neighbor_msg_dim: int = 6     # 邻居交换特征维度
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    log_std_init: float = 0.0

    @property
    def local_obs_dim(self) -> int:
        # 用户需求3 + 残余资源7 + 环境3 + L1配额5
        return 18

    @property
    def obs_dim(self) -> int:
        return self.local_obs_dim + self.max_neighbors * self.neighbor_msg_dim

    @property
    def action_dim(self) -> int:
        # 每条迁移: 目标区域选择1 + 设备清单5；每条链路: A侧1 + B侧1 + 类型1 + 位置1
        # 本智能体迁出：tgt(1)+5；链路：peer(1)+link_type(1)+deploy_pos(1) 但 A 为本区域固定
        return self.max_migrations * 6 + self.max_links * 3


@dataclass
class L2RegionState:
    """单个区域的聚合状态（由子区域 32 维特征聚合得到）。"""

    region_id: int
    user_total: float
    high_priority_ratio: float
    avg_demand_intensity: float
    residual_public_bw: float
    residual_broadcast: float
    deployed_counts: np.ndarray  # (5,)
    severity: float
    road_pass_rate: float
    power_recovery_rate: float
    l1_quota: np.ndarray  # (5,) L1 分配上限
    neighbor_ids: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.deployed_counts = np.asarray(self.deployed_counts, dtype=np.float32).reshape(5)
        self.l1_quota = np.asarray(self.l1_quota, dtype=np.float32).reshape(5)
        self.high_priority_ratio = float(np.clip(self.high_priority_ratio, 0.0, 1.0))
        self.severity = float(np.clip(self.severity, 0.0, 1.0))
        self.road_pass_rate = float(np.clip(self.road_pass_rate, 0.0, 1.0))
        self.power_recovery_rate = float(np.clip(self.power_recovery_rate, 0.0, 1.0))


def compute_neighbor_message(state: L2RegionState) -> np.ndarray:
    """
    生成对外广播的 6 维邻居通信摘要。
    [severity, user_norm, residual_bw_norm, resource_gap, surplus, deployed_sum_norm]
    """
    deployed_sum = float(state.deployed_counts.sum())
    quota_sum = float(state.l1_quota.sum()) + 1e-6
    gap = max(0.0, (state.user_total * state.avg_demand_intensity) - state.residual_public_bw)
    gap = gap / (gap + 1.0)
    surplus = max(0.0, float(state.l1_quota.sum() - deployed_sum)) / quota_sum
    bw_norm = state.residual_public_bw / (state.residual_public_bw + 100.0)
    user_norm = state.user_total / (state.user_total + 1000.0)
    dep_norm = deployed_sum / quota_sum
    return np.array(
        [state.severity, user_norm, bw_norm, gap, surplus, dep_norm],
        dtype=np.float32,
    )


def encode_local_observation(state: L2RegionState) -> np.ndarray:
    """编码本区域 18 维局部观测。"""
    user_block = np.array(
        [
            state.user_total / (state.user_total + 1000.0),
            state.high_priority_ratio,
            state.avg_demand_intensity,
        ],
        dtype=np.float32,
    )
    residual_block = np.concatenate(
        [
            np.array(
                [
                    state.residual_public_bw / (state.residual_public_bw + 100.0),
                    state.residual_broadcast / (state.residual_broadcast + 50.0),
                ],
                dtype=np.float32,
            ),
            state.deployed_counts / (state.l1_quota + 1.0),
        ]
    )
    env_block = np.array(
        [state.severity, state.road_pass_rate, state.power_recovery_rate],
        dtype=np.float32,
    )
    quota_block = state.l1_quota / (state.l1_quota.max() + 1.0)
    return np.concatenate([user_block, residual_block, env_block, quota_block])


def encode_observation(
    state: L2RegionState,
    neighbor_messages: Dict[int, np.ndarray],
    cfg: L2Config,
) -> np.ndarray:
    """局部观测 + 邻居通信（不足补零）。"""
    local = encode_local_observation(state)
    msgs = []
    for nid in state.neighbor_ids[: cfg.max_neighbors]:
        msgs.append(neighbor_messages.get(int(nid), np.zeros(cfg.neighbor_msg_dim, dtype=np.float32)))
    while len(msgs) < cfg.max_neighbors:
        msgs.append(np.zeros(cfg.neighbor_msg_dim, dtype=np.float32))
    return np.concatenate([local, np.concatenate(msgs)])


def region_state_from_dict(raw: Dict[str, Any], region_id: int, cfg: L2Config) -> L2RegionState:
    return L2RegionState(
        region_id=region_id,
        user_total=float(raw.get("user_total", 0)),
        high_priority_ratio=float(raw.get("high_priority_ratio", 0.3)),
        avg_demand_intensity=float(raw.get("avg_demand_intensity", 0.5)),
        residual_public_bw=float(raw.get("residual_public_bw", 0)),
        residual_broadcast=float(raw.get("residual_broadcast", 0)),
        deployed_counts=np.asarray(raw.get("deployed_counts", np.zeros(5)), dtype=np.float32),
        severity=float(raw.get("severity", 0.5)),
        road_pass_rate=float(raw.get("road_pass_rate", 0.5)),
        power_recovery_rate=float(raw.get("power_recovery_rate", 0.5)),
        l1_quota=np.asarray(raw.get("l1_quota", np.zeros(5)), dtype=np.float32),
        neighbor_ids=list(raw.get("neighbor_ids", [])),
    )


def _pick_neighbor_index(selector: float, neighbor_ids: List[int]) -> int:
    if not neighbor_ids:
        return -1
    idx = int(np.clip(selector, 0.0, 0.999) * len(neighbor_ids))
    return int(neighbor_ids[min(idx, len(neighbor_ids) - 1)])


def decode_regional_action(
    action: np.ndarray,
    region_id: int,
    state: L2RegionState,
    cfg: L2Config,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    解码单个 L2 智能体动作。

    Returns:
        migrations: 迁出任务列表，每项 {src, tgt, devices[5]}
        links: 链路列表，每项 {region_a, region_b, link_type, deploy_position}
    """
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    migrations: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []

    off = 0
    for _ in range(cfg.max_migrations):
        sel = float(a[off])
        counts_raw = a[off + 1 : off + 6]
        off += 6
        tgt = _pick_neighbor_index(sel, state.neighbor_ids)
        if tgt < 0:
            continue
        available = np.maximum(state.deployed_counts - 0, 0).astype(np.float32)
        w = np.exp(counts_raw - counts_raw.max())
        transfer = np.floor(w / (w.sum() + 1e-8) * available.sum()).astype(np.int32)
        # 按可用量分配
        for j in range(5):
            cap = int(available[j])
            transfer[j] = min(int(transfer[j]), cap)
        if transfer.sum() <= 0:
            continue
        migrations.append(
            {
                "src": int(region_id),
                "tgt": int(tgt),
                "devices": transfer.copy(),
            }
        )

    for _ in range(cfg.max_links):
        peer_sel = float(a[off])
        link_sel = float(a[off + 1])
        pos = float(a[off + 2])
        off += 3
        if not state.neighbor_ids:
            continue
        peer = _pick_neighbor_index(peer_sel, state.neighbor_ids)
        if peer < 0:
            continue
        link_type = int(np.clip(link_sel * 3, 0, 2.999))
        links.append(
            {
                "region_a": int(region_id),
                "region_b": int(peer),
                "link_type": link_type,
                "link_type_name": LINK_TYPE_NAMES[link_type],
                "deploy_position": float(np.clip(pos, 0.0, 1.0)),
            }
        )

    return migrations, links


def merge_migrations(all_migrations: List[Dict[str, Any]]) -> np.ndarray:
    """合并为 M×(2+5) 矩阵表示（变长列表也可，此处返回结构化数组）。"""
    if not all_migrations:
        return np.zeros((0, 7), dtype=np.float32)
    rows = []
    for m in all_migrations:
        row = [m["src"], m["tgt"], *m["devices"].tolist()]
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def merge_links(all_links: List[Dict[str, Any]]) -> np.ndarray:
    """合并为 K×4 矩阵。"""
    if not all_links:
        return np.zeros((0, 4), dtype=np.float32)
    rows = []
    for lk in all_links:
        rows.append([lk["region_a"], lk["region_b"], lk["link_type"], lk["deploy_position"]])
    return np.asarray(rows, dtype=np.float32)


def apply_migrations_to_quotas(
    l1_quota: np.ndarray,
    migrations: List[Dict[str, Any]],
    n_regions: int,
) -> np.ndarray:
    """
    根据迁移方案动态调整各区域有效配额（在 L1 硬上限内调剂）。

    Returns:
        adjusted_quota: (N, 5)
    """
    Q = np.asarray(l1_quota, dtype=np.int32).copy()
    if Q.ndim == 1:
        Q = Q.reshape(1, 5)
    if Q.shape[0] < n_regions:
        pad = np.zeros((n_regions - Q.shape[0], 5), dtype=np.int32)
        Q = np.vstack([Q, pad])

    for m in migrations:
        src, tgt = int(m["src"]), int(m["tgt"])
        dev = np.asarray(m["devices"], dtype=np.int32).reshape(5)
        if src >= n_regions or tgt >= n_regions:
            continue
        move = np.minimum(dev, Q[src])
        Q[src] -= move
        Q[tgt] += move

    return Q
