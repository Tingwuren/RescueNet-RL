"""
L3 用户配置/执行层：32 维子区域细粒度观测 + 上层约束 + 72 维标准动作。

每个 L3 智能体对应约 10 km² 子区域，独立接收该子区域 32 维特征。
动作解码后生成组网拓扑结构（节点、链路、覆盖）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DEVICE_NAMES = [
    "应急基站",
    "便携式广播通信网关",
    "5G中继",
    "Mesh中继",
    "通信UAV",
]
N_DEVICES = 5
N_GRIDS = 12

# 32 维特征分组索引（便于文档对照）
USER_SLICE = slice(0, 8)
RESOURCE_SLICE = slice(8, 16)
DEVICE_SLICE = slice(16, 24)
ENV_SLICE = slice(24, 32)


@dataclass
class L3Config:
    n_device_types: int = 5
    n_deploy_grids: int = 12
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    log_std_init: float = 0.0
    max_deploy_per_grid: int = 3  # 单网格单类设备数量上限

    @property
    def base_obs_dim(self) -> int:
        return 32

    @property
    def constraint_dim(self) -> int:
        # L1配额5 + L2调入5 + L2调出5 + L2链路端点4
        return 19

    @property
    def obs_dim(self) -> int:
        return self.base_obs_dim + self.constraint_dim

    @property
    def action_dim(self) -> int:
        return 72  # 60 + 10 + 2


@dataclass
class L3SubRegionState:
    """子区域 32 维细粒度状态 + 元数据。"""

    subregion_id: int
    region_id: int
    # --- 用户特征 8 ---
    user_total: float
    high_priority_ratio: float
    avg_demand_intensity: float
    user_concentration: float
    rescue_personnel: float
    affected_population: float
    command_personnel: float
    demand_growth_rate: float
    # --- 资源特征 8 ---
    bw_5g_600: float
    bw_5g_700: float
    bw_satellite: float
    bw_wifi6: float
    bw_shortwave: float
    bw_uav: float
    bw_residual: float
    bw_total_available: float
    # --- 设备特征 8 ---
    avail_emergency_bs: float
    avail_portable_gw: float
    avail_relay: float
    avail_uav: float
    avg_battery: float
    avg_tx_power: float
    device_fault_rate: float
    deploy_difficulty: float
    # --- 环境特征 8 ---
    severity: float
    terrain_complexity: float
    road_pass_rate: float
    power_recovery_rate: float
    hours_since_disaster: float
    neighbor_resource_state: float
    secondary_disaster_risk: float
    rescue_progress: float
    # 网格中心坐标（拓扑用） shape (12, 2) 或 flatten
    grid_centers: Optional[np.ndarray] = None

    def to_base_observation(self) -> np.ndarray:
        """编码标准 32 维向量。"""
        vec = np.array(
            [
                self.user_total,
                self.high_priority_ratio,
                self.avg_demand_intensity,
                self.user_concentration,
                self.rescue_personnel,
                self.affected_population,
                self.command_personnel,
                self.demand_growth_rate,
                self.bw_5g_600,
                self.bw_5g_700,
                self.bw_satellite,
                self.bw_wifi6,
                self.bw_shortwave,
                self.bw_uav,
                self.bw_residual,
                self.bw_total_available,
                self.avail_emergency_bs,
                self.avail_portable_gw,
                self.avail_relay,
                self.avail_uav,
                self.avg_battery,
                self.avg_tx_power,
                self.device_fault_rate,
                self.deploy_difficulty,
                self.severity,
                self.terrain_complexity,
                self.road_pass_rate,
                self.power_recovery_rate,
                self.hours_since_disaster,
                self.neighbor_resource_state,
                self.secondary_disaster_risk,
                self.rescue_progress,
            ],
            dtype=np.float32,
        )
        assert vec.shape[0] == 32
        return _normalize_base_obs(vec)


@dataclass
class L3UpperConstraints:
    """来自 L1/L2 的上层约束指令。"""

    l1_quota: np.ndarray           # (5,) 本区域（或子区域所属区域）设备配额上限
    l2_transfer_in: np.ndarray     # (5,) 调入
    l2_transfer_out: np.ndarray    # (5,) 调出
    link_active: float = 0.0       # 是否承担跨区链路端点
    link_type: int = 0             # 0/1/2
    link_peer_region: int = -1
    link_deploy_grid: int = 0      # 建议部署网格 0-11

    def effective_quota(self) -> np.ndarray:
        """L1 配额经 L2 调剂后的可部署上限。"""
        q = np.asarray(self.l1_quota, dtype=np.float32).reshape(5)
        tin = np.asarray(self.l2_transfer_in, dtype=np.float32).reshape(5)
        tout = np.asarray(self.l2_transfer_out, dtype=np.float32).reshape(5)
        return np.maximum(q + tin - tout, 0)


def _normalize_base_obs(vec: np.ndarray) -> np.ndarray:
    """轻量归一化，避免量级差异影响训练。"""
    out = vec.astype(np.float32).copy()
    # 用户数相关
    out[0:8] = out[0:8] / (np.abs(out[0:8]).max() + 1e-6)
    out[8:16] = out[8:16] / (out[8:16].max() + 1e-6)
    out[16:24] = out[16:24] / (out[16:24].max() + 1e-6)
    out[24:32] = np.clip(out[24:32], 0.0, 1.0)
    return out


def encode_constraint_vector(constraints: L3UpperConstraints, cfg: L3Config) -> np.ndarray:
    eff = constraints.effective_quota()
    eff_norm = eff / (eff.max() + 1.0)
    tin = np.asarray(constraints.l2_transfer_in, dtype=np.float32).reshape(5)
    tout = np.asarray(constraints.l2_transfer_out, dtype=np.float32).reshape(5)
    tin_norm = tin / (tin.max() + 1.0)
    tout_norm = tout / (tout.max() + 1.0)
    link_block = np.array(
        [
            float(constraints.link_active),
            constraints.link_type / 2.0,
            constraints.link_peer_region / max(constraints.link_peer_region, 1),
            constraints.link_deploy_grid / max(cfg.n_deploy_grids - 1, 1),
        ],
        dtype=np.float32,
    )
    return np.concatenate([eff_norm, tin_norm, tout_norm, link_block])


def encode_observation(
    state: L3SubRegionState,
    constraints: L3UpperConstraints,
    cfg: L3Config,
) -> np.ndarray:
    base = state.to_base_observation()
    cons = encode_constraint_vector(constraints, cfg)
    return np.concatenate([base, cons])


def subregion_from_dict(raw: Dict[str, Any], subregion_id: int, region_id: int) -> L3SubRegionState:
    def g(key: str, default: float) -> float:
        return float(raw.get(key, default))

    return L3SubRegionState(
        subregion_id=subregion_id,
        region_id=region_id,
        user_total=g("user_total", 100),
        high_priority_ratio=g("high_priority_ratio", 0.3),
        avg_demand_intensity=g("avg_demand_intensity", 0.5),
        user_concentration=g("user_concentration", 0.5),
        rescue_personnel=g("rescue_personnel", 10),
        affected_population=g("affected_population", 80),
        command_personnel=g("command_personnel", 5),
        demand_growth_rate=g("demand_growth_rate", 0.1),
        bw_5g_600=g("bw_5g_600", 10),
        bw_5g_700=g("bw_5g_700", 20),
        bw_satellite=g("bw_satellite", 5),
        bw_wifi6=g("bw_wifi6", 15),
        bw_shortwave=g("bw_shortwave", 2),
        bw_uav=g("bw_uav", 8),
        bw_residual=g("bw_residual", 12),
        bw_total_available=g("bw_total_available", 50),
        avail_emergency_bs=g("avail_emergency_bs", 2),
        avail_portable_gw=g("avail_portable_gw", 1),
        avail_relay=g("avail_relay", 2),
        avail_uav=g("avail_uav", 1),
        avg_battery=g("avg_battery", 0.8),
        avg_tx_power=g("avg_tx_power", 0.7),
        device_fault_rate=g("device_fault_rate", 0.05),
        deploy_difficulty=g("deploy_difficulty", 0.4),
        severity=g("severity", 0.5),
        terrain_complexity=g("terrain_complexity", 0.5),
        road_pass_rate=g("road_pass_rate", 0.5),
        power_recovery_rate=g("power_recovery_rate", 0.3),
        hours_since_disaster=g("hours_since_disaster", 12),
        neighbor_resource_state=g("neighbor_resource_state", 0.5),
        secondary_disaster_risk=g("secondary_disaster_risk", 0.2),
        rescue_progress=g("rescue_progress", 0.3),
        grid_centers=raw.get("grid_centers"),
    )


def decode_action(
    action: np.ndarray,
    constraints: L3UpperConstraints,
    cfg: L3Config,
) -> Dict[str, Any]:
    """
    72 维动作解码并投影到上层约束可行域。

    Returns:
        deployment (5,12) int, work_params (5,2), global_params (2,)
    """
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    assert a.shape[0] == cfg.action_dim

    deploy_raw = a[0:60].reshape(cfg.n_device_types, cfg.n_deploy_grids)
    params_raw = a[60:70].reshape(cfg.n_device_types, 2)
    global_raw = a[70:72]

    # 部署数量：非负整数，先 softmax 按设备类型分配有效配额，再按网格分配
    eff_quota = constraints.effective_quota().astype(np.int32)
    deployment = np.zeros((cfg.n_device_types, cfg.n_deploy_grids), dtype=np.int32)

    for j in range(cfg.n_device_types):
        cap = int(eff_quota[j])
        if cap <= 0:
            continue
        grid_w = np.exp(deploy_raw[j] - deploy_raw[j].max())
        grid_w = grid_w / (grid_w.sum() + 1e-8)
        counts = np.floor(grid_w * cap).astype(np.int32)
        rem = cap - counts.sum()
        order = np.argsort(-(grid_w * cap - counts))
        for k in range(int(rem)):
            gidx = order[k % cfg.n_deploy_grids]
            if counts[gidx] < cfg.max_deploy_per_grid:
                counts[gidx] += 1
        deployment[j] = counts

    work_params = np.clip(params_raw, 0.0, 1.0)
    global_params = np.clip(global_raw, 0.0, 1.0)

    return {
        "deployment": deployment,
        "work_params": work_params,
        "global_params": global_params,
        "rescue_priority_weight": float(global_params[0]),
        "cross_region_reserve_ratio": float(global_params[1]),
    }


def check_constraints(
    decoded: Dict[str, Any],
    constraints: L3UpperConstraints,
) -> Tuple[bool, str]:
    """检查部署总量是否超过有效配额。"""
    dep = decoded["deployment"]
    eff = constraints.effective_quota().astype(np.int32)
    for j, name in enumerate(DEVICE_NAMES):
        total = int(dep[j].sum())
        if total > int(eff[j]):
            return False, f"{name}: 部署{total} > 有效配额{eff[j]}"
    return True, "ok"
