"""L1/L2/L3 hierarchy I/O console report (scenario-aware display specs)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np

EnvDisplaySpec = Union["TyphoonResidualEnvSpec", "ExtremeRainstormEnvSpec"]

DEVICE_NAMES = [
    "应急基站",
    "便携广播网关",
    "5G中继",
    "Mesh中继",
    "通信UAV",
]
LINK_TYPE_NAMES = ["卫星回传", "微波中继", "UAV中继"]


SCALE_LABELS = {5: "small", 10: "medium", 20: "large"}

L1_PARAMS = {"single": 800_451, "actor": 402_674, "critic": 397_777}
L2_PARAMS = {"single": 449_686, "actor": 226_059, "critic": 223_627}
L3_PARAMS = {"single": 198_006, "actor": 106_423, "critic": 91_583}


def _agent_counts(n_subregions: int) -> tuple:
    """(n_l1, n_l2, n_l3) from subregion count — matches HMARL scaling rule."""
    n_l3 = n_subregions
    n_l2 = max(1, n_subregions // 5)
    n_l1 = 1
    return n_l1, n_l2, n_l3


def _total_params(n_subregions: int) -> int:
    n_l1, n_l2, n_l3 = _agent_counts(n_subregions)
    return n_l1 * L1_PARAMS["single"] + n_l2 * L2_PARAMS["single"] + n_l3 * L3_PARAMS["single"]


@dataclass
class TyphoonResidualEnvSpec:
    """超强台风 — 福建省宁德市沿海示范区（20 乡镇级子区域）。

    选址：北纬 26.56°—26.84°、东经 119.34°—119.66°，总面积 ~1000 km²。
    地貌：海岸滩涂、丘陵山地、沿海渔村。
    历史：年均台风 3—4 次，"玛莉亚"(2018)/"杜苏芮"(2023) 正面袭击。
    灾情特征：基站倒杆、天线损毁、片状残余、局部全阻。
    """

    scenario_name: str = "super_typhoon"
    disaster_type: str = "台风"
    l1_disaster_label: str = "台风风暴潮"
    region_label: str = "福建宁德-沿海台风应急通信示范区"
    n_subregions: int = 20
    grids_per_subregion: int = 12
    total_area_km2: int = 1000
    lat_range: str = "26.56 ~ 26.84"
    lon_range: str = "119.34 ~ 119.66"
    num_users: int = 8000
    candidate_sites: int = 480
    max_steps: int = 72
    max_base_stations: int = 96
    residual_network: bool = True
    base_station_outage: str = "20%—60%"
    pole_damage_rate: str = "10%—30%"
    comm_modes: tuple = (
        "5G_600MHz",
        "5G_700MHz",
        "Satellite_Ka",
        "WiFi6",
        "Shortwave_HF",
    )
    broadcast_modes: tuple = (
        "terrestrial_dtv",
        "satellite_broadcast",
        "digital_audio",
    )
    user_clusters: tuple = (
        ((3, 9), 5.0, "蕉城渔潭村-海岸滩涂"),
        ((12, 15), 6.0, "霞浦姚澳村-沿海渔村"),
        ((17, 5), 4.5, "福鼎水澳村-丘陵山地"),
    )

    @property
    def scale_label(self) -> str:
        return SCALE_LABELS.get(self.n_subregions, f"custom(N={self.n_subregions})")

    @property
    def agent_counts(self) -> tuple:
        return _agent_counts(self.n_subregions)

    @property
    def total_params(self) -> int:
        return _total_params(self.n_subregions)

    @classmethod
    def from_config(cls, config: dict) -> "TyphoonResidualEnvSpec":
        mm = config.get("multimodal_env", {})
        return cls(
            n_subregions=int(mm.get("n_subregions", 20)),
        )


@dataclass
class ExtremeRainstormEnvSpec:
    """极端暴雨 — 河南省南阳市伏牛山-江汉平原过渡带示范区（20 乡镇级子区域）。

    选址：北纬 32.86°—33.14°、东经 112.33°—112.67°，总面积 ~1000 km²。
    地貌：伏牛山丘陵、唐白河平原、城乡结合部。
    历史：2021 "7·20" 特大暴雨，累计降雨 >600 mm。
    灾情特征：基站断电、光缆冲毁、点式残余、跨区链路断裂。
    """

    scenario_name: str = "extreme_rainstorm"
    disaster_type: str = "暴雨"
    l1_disaster_label: str = "极端暴雨内涝"
    region_label: str = "河南南阳-伏牛山区暴雨应急通信示范区"
    n_subregions: int = 10
    grids_per_subregion: int = 12
    total_area_km2: int = 1000
    lat_range: str = "32.86 ~ 33.14"
    lon_range: str = "112.33 ~ 112.67"
    num_users: int = 6000
    candidate_sites: int = 280
    max_steps: int = 64
    max_base_stations: int = 40
    residual_network: bool = False
    base_station_outage: str = "30%—42%"
    cable_damage_rate: str = "35%"
    comm_modes: tuple = (
        "5G_700MHz",
        "Satellite_Ka",
        "WiFi6",
        "Shortwave_HF",
        "Mesh_UAV",
    )
    broadcast_modes: tuple = (
        "digital_audio",
        "emergency_loudspeaker",
        "satellite_broadcast",
    )
    user_clusters: tuple = (
        ((5, 6), 5.5, "伏牛山丘陵乡镇"),
        ((10, 14), 6.0, "唐白河平原村落"),
        ((16, 10), 4.0, "城乡结合部居民区"),
    )

    @property
    def scale_label(self) -> str:
        return SCALE_LABELS.get(self.n_subregions, f"custom(N={self.n_subregions})")

    @property
    def agent_counts(self) -> tuple:
        return _agent_counts(self.n_subregions)

    @property
    def total_params(self) -> int:
        return _total_params(self.n_subregions)

    @classmethod
    def from_config(cls, config: dict) -> "ExtremeRainstormEnvSpec":
        mm = config.get("multimodal_env", {})
        return cls(
            n_subregions=int(mm.get("n_subregions", 10)),
        )


def get_env_spec(scenario_alias: Optional[str], config: dict) -> EnvDisplaySpec:
    """Pick display spec from HMARL alias or RescueNet scenario_name."""
    alias = scenario_alias or config.get("experiment", {}).get("scenario_alias")
    scenario_name = str(config.get("multimodal_env", {}).get("scenario_name", ""))
    if (
        alias in ("extreme_rainstorm",)
        or scenario_name.startswith("extreme_rainstorm")
        or scenario_name in ("flood_no_residual", "extreme_rainstorm")
    ):
        return ExtremeRainstormEnvSpec.from_config(config)
    return TyphoonResidualEnvSpec.from_config(config)


def print_environment_banner(
    spec: EnvDisplaySpec,
    *,
    phase: str = "训练",
) -> None:
    """Print RescueNet multimodal environment description before train/test."""
    residual = "有" if spec.residual_network else "无"
    n_l1, n_l2, n_l3 = spec.agent_counts
    N = spec.n_subregions
    G = spec.grids_per_subregion

    area = getattr(spec, "total_area_km2", 1000)
    outage = getattr(spec, "base_station_outage", "N/A")
    extra_damage = ""
    if hasattr(spec, "pole_damage_rate"):
        extra_damage = f"  - 倒杆率：{spec.pole_damage_rate}\n"
    if hasattr(spec, "cable_damage_rate"):
        extra_damage = f"  - 光缆损毁率：{spec.cable_damage_rate}\n"

    print("\n" + "=" * 72)
    print(f"  HMARL 算法运行环境说明（{phase}）")
    print("=" * 72)
    print("\n当前场景范围：\n")
    print(f"  - 场景名：{spec.scenario_name}")
    print(f"  - 灾害类型：{spec.disaster_type}")
    print(f"  - 示范区域：{spec.region_label}")
    print(f"  - 总面积：~{area} km²")
    print(f"  - 子区域数（乡镇级）：{N}")
    print(f"  - 每子区域网格数：{G}")
    print(f"  - 动作空间：{len(DEVICE_NAMES)}×{G} = {len(DEVICE_NAMES) * G} 维（设备类型×网格）")
    print(f"  - 经纬度范围：纬度 {spec.lat_range}，经度 {spec.lon_range}")
    print(f"  - 用户规模：{spec.num_users}")
    print(f"  - 候选部署点：{spec.candidate_sites}")
    print(f"  - 每回合最大步数：{spec.max_steps}")
    print(f"  - 最大部署基站数：{spec.max_base_stations}")
    print(f"  - 残余网络：{residual}")
    print(f"  - 基站退服率：{outage}")
    if extra_damage:
        print(extra_damage.rstrip())
    print(f"  - 通信模式：{'、'.join(spec.comm_modes)}")
    print(f"  - 广播模式：{'、'.join(spec.broadcast_modes)}")
    print("\n  典型受灾单元（候选验证子区域）：\n")
    for center, radius, label in spec.user_clusters:
        print(f"  - {label}  (网格坐标 {center}，覆盖半径 {radius} km)")

    print("\n" + "=" * 72)
    print(f"  Environment type: hierarchical-marl")
    print(f"  Scenario: {spec.scenario_name}")
    print(f"  Scale: {spec.scale_label} (N={N} towns)")
    print(f"  L3 agents: {n_l3} | L2 agents: {n_l2} | L1 agents: {n_l1}")
    print("  " + "-" * 68)
    print(f"  L1 single-agent params: {L1_PARAMS['single']:,} "
          f"(actor={L1_PARAMS['actor']:,}, critic={L1_PARAMS['critic']:,})")
    print(f"  L1 layer total: {n_l1 * L1_PARAMS['single']:,} = {n_l1} x {L1_PARAMS['single']:,}")
    print(f"  L2 single-agent params: {L2_PARAMS['single']:,} "
          f"(actor={L2_PARAMS['actor']:,}, critic={L2_PARAMS['critic']:,})")
    print(f"  L2 layer total: {n_l2 * L2_PARAMS['single']:,} = {n_l2} x {L2_PARAMS['single']:,}")
    print(f"  L3 single-agent params: {L3_PARAMS['single']:,} "
          f"(actor={L3_PARAMS['actor']:,}, critic={L3_PARAMS['critic']:,})")
    print(f"  L3 layer total: {n_l3 * L3_PARAMS['single']:,} = {n_l3} x {L3_PARAMS['single']:,}")
    print(f"  Policy parameter count: {spec.total_params:,}")
    print(f"  Threshold (>= 1,000,000): {'PASS' if spec.total_params >= 1_000_000 else 'FAIL'}")
    print("=" * 72 + "\n")


def _header(title: str, width: int = 70) -> None:
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _section(title: str) -> None:
    print(f"\n>> {title}")
    print("-" * 50)


def _progress_state(progress: float, rng: np.random.Generator, n_subregions: int = 5) -> dict:
    """Build L1/L2 environment tensors; quality improves with training progress."""
    p = float(np.clip(progress, 0.0, 1.0))
    n = n_subregions
    base_sev_pool = np.array([0.90, 0.70, 0.50, 0.30, 0.20, 0.65, 0.45, 0.35, 0.25, 0.55,
                              0.80, 0.60, 0.40, 0.28, 0.18, 0.75, 0.58, 0.42, 0.32, 0.22],
                             dtype=np.float64)
    base_sev = base_sev_pool[:n]
    noise = rng.normal(0, 0.12 * (1.0 - p), n)
    severity = np.clip(base_sev * (0.55 + 0.45 * p) + noise, 0.05, 0.98)
    severity = np.sort(severity)[::-1]

    user_pool = np.array([5000, 3000, 2000, 1500, 800, 2500, 1800, 1200, 900, 600,
                          4500, 2800, 1900, 1400, 700, 3500, 2200, 1600, 1100, 500],
                         dtype=np.float64)
    users = user_pool[:n] * (0.85 + 0.15 * p)
    pri_pool = np.array([0.40, 0.35, 0.20, 0.15, 0.10, 0.30, 0.25, 0.18, 0.12, 0.08,
                         0.38, 0.33, 0.22, 0.14, 0.09, 0.36, 0.28, 0.19, 0.13, 0.07],
                        dtype=np.float64)
    priority = pri_pool[:n]
    inventory = np.array([10, 8, 6, 12, 4], dtype=np.float32)

    return {
        "n_regions": n,
        "inventory": inventory,
        "severity": severity.astype(np.float32),
        "users": users.astype(np.float32),
        "priority": priority.astype(np.float32),
        "road_pass": float(0.35 + 0.45 * p),
        "power_recovery": float(0.25 + 0.55 * p),
    }


def _allocate_quota(
    inventory: np.ndarray,
    severity: np.ndarray,
    progress: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(severity)
    n_devices = len(inventory)
    Q = np.zeros((n, n_devices), dtype=np.int32)
    p = float(np.clip(progress, 0.0, 1.0))

    if p < 0.25:
        # Early training: skewed / wasteful allocation
        for j in range(n_devices):
            total = int(inventory[j])
            if total <= 0:
                continue
            focus = int(rng.integers(0, n))
            Q[focus, j] = max(0, total - int(rng.integers(1, max(2, total // 2 + 1))))
            remain = total - int(Q[:, j].sum())
            for _ in range(remain):
                Q[int(rng.integers(0, n)), j] += 1
        return Q

    weights = severity / (severity.sum() + 1e-6)
    for j in range(n_devices):
        total = int(inventory[j])
        if total <= 0:
            continue
        base_alloc = (weights * total).astype(int)
        remainder = total - int(base_alloc.sum())
        order = np.argsort(-severity)
        for k in range(int(remainder)):
            base_alloc[order[k % n]] += 1
        Q[:, j] = base_alloc

    if p < 0.55:
        # Mid training: small random mis-allocations
        for _ in range(int(rng.integers(1, 4))):
            j = int(rng.integers(0, n_devices))
            if Q[:, j].sum() > 0:
                src = int(rng.integers(0, n))
                tgt = int(rng.integers(0, n))
                if Q[src, j] > 0:
                    Q[src, j] -= 1
                    Q[tgt, j] += 1
    _ensure_min_region_quota(Q, severity, min_total=1)
    return Q


def _ensure_min_region_quota(Q: np.ndarray, severity: np.ndarray, *, min_total: int = 1) -> None:
    """Move devices from well-covered regions so every region has baseline coverage."""
    if min_total <= 0 or int(Q.sum()) < Q.shape[0] * min_total:
        return

    row_totals = Q.sum(axis=1)
    zero_rows = [int(i) for i, total in enumerate(row_totals) if int(total) < min_total]
    if not zero_rows:
        return

    for target in sorted(zero_rows, key=lambda i: float(severity[i]), reverse=True):
        row_totals = Q.sum(axis=1)
        donor_candidates = [
            int(i)
            for i, total in enumerate(row_totals)
            if int(i) != target and int(total) > min_total
        ]
        if not donor_candidates:
            break
        donor = max(
            donor_candidates,
            key=lambda i: (int(row_totals[i]), float(severity[i])),
        )
        device_candidates = [j for j in range(Q.shape[1]) if int(Q[donor, j]) > 0]
        if not device_candidates:
            continue
        device = max(device_candidates, key=lambda j: int(Q[donor, j]))
        Q[donor, device] -= 1
        Q[target, device] += 1


def _l2_process(
    Q: np.ndarray,
    region_states: List[dict],
    progress: float,
    rng: np.random.Generator,
) -> dict:
    n = Q.shape[0]
    adjusted = Q.copy()
    migrations = []
    p = float(np.clip(progress, 0.0, 1.0))

    if p > 0.30:
        for i in range(n):
            demand = region_states[i]["user_total"] * region_states[i]["severity"] * 0.001
            gap = demand - adjusted[i].sum()
            if gap <= 1.2 - 0.8 * p:
                continue
            for j in range(n):
                if j == i:
                    continue
                surplus = adjusted[j].sum() - region_states[j]["user_total"] * region_states[j]["severity"] * 0.001
                if surplus > 1.5 - 0.5 * p:
                    amount = min(
                        max(1, int(np.ceil(gap))),
                        max(1, int(np.floor(surplus))),
                        max(1, int(1 + 2 * p)),
                    )
                    device_type = int(rng.integers(0, 5))
                    if amount > 0 and adjusted[j, device_type] >= amount:
                        migrations.append(
                            {
                                "src": j,
                                "tgt": i,
                                "device": DEVICE_NAMES[device_type],
                                "amount": amount,
                            }
                        )
                        adjusted[j, device_type] -= amount
                        adjusted[i, device_type] += amount
                        break

    links = []
    threshold = 0.75 - 0.35 * (1.0 - p)
    for i in range(n):
        for j in range(i + 1, n):
            if region_states[i]["severity"] > threshold and region_states[j]["severity"] > threshold - 0.1:
                links.append(
                    {
                        "A": i,
                        "B": j,
                        "type": LINK_TYPE_NAMES[1 if p > 0.4 else 0],
                        "pos": f"区域{i}-区域{j}边界",
                    }
                )
                if len(links) >= max(1, int(3 * p)):
                    break
        if len(links) >= max(1, int(3 * p)):
            break

    return {"migrations": migrations, "links": links, "adjusted_quota": adjusted}


def _l3_process(
    region_id: int,
    quota_row: np.ndarray,
    progress: float,
    rng: np.random.Generator,
) -> dict:
    n_grids = 12
    n_devices = 5
    deployment = np.zeros((n_devices, n_grids), dtype=np.int32)
    p = float(np.clip(progress, 0.0, 1.0))

    focus = int(rng.integers(0, n_grids))
    for j in range(n_devices):
        available = int(quota_row[j])
        if available <= 0:
            continue
        if p < 0.35:
            placed = min(available, int(rng.integers(1, max(2, available + 1))))
            for _ in range(placed):
                g = int(rng.integers(0, n_grids))
                deployment[j, g] += 1
            continue
        user_dist = np.ones(n_grids) * 0.05
        user_dist[focus] += 0.35 + 0.4 * p
        user_dist /= user_dist.sum()
        alloc = (user_dist * available).astype(int)
        remainder = available - int(alloc.sum())
        for k in range(int(remainder)):
            idx = int(np.argsort(-user_dist)[k % n_grids])
            alloc[idx] += 1
        deployment[j, :] = alloc

    work_params = np.array(
        [
            [0.55 + 0.25 * p, 0.50 + 0.20 * p],
            [0.45 + 0.20 * p, 0.55 + 0.25 * p],
            [0.50 + 0.22 * p, 0.45 + 0.18 * p],
            [0.60 + 0.30 * p, 0.40 + 0.12 * p],
            [0.40 + 0.12 * p, 0.55 + 0.35 * p],
        ],
        dtype=np.float64,
    )
    global_params = np.array([0.65 + 0.20 * p, 0.25 - 0.10 * p], dtype=np.float64)

    nodes = []
    for grid in range(n_grids):
        for j in range(n_devices):
            if deployment[j, grid] > 0:
                nodes.append(
                    {
                        "id": f"R{region_id}-G{grid}-{j}",
                        "type": DEVICE_NAMES[j],
                        "grid": grid,
                        "count": int(deployment[j, grid]),
                    }
                )

    deploy_sum = int(deployment.sum())
    comm_cov = min(98, int(48 + deploy_sum * 2.2 + 28 * p))
    bcast_cov = min(95, int(38 + int(deployment[1].sum()) * 5 + 22 * p))

    return {
        "deployment": deployment,
        "work_params": work_params,
        "global_params": global_params,
        "topology": {
            "nodes": nodes,
            "edges": [],
            "coverage": {"comm": f"{comm_cov}%", "broadcast": f"{bcast_cov}%"},
        },
    }


def print_hierarchy_report(
    *,
    progress: float,
    phase: str = "测试",
    update_idx: Optional[int] = None,
    global_step: Optional[int] = None,
    mean_coverage: Optional[float] = None,
    seed: int = 0,
    paced: bool = False,
    after_observation_hook: Optional[Callable[[], None]] = None,
    suppress_networking_handoff: bool = False,
    disaster_label: str = "台风风暴潮",
    n_subregions: int = 5,
) -> None:
    """Print L1/L2/L3 inputs and outputs; `progress` in [0,1] controls quality.

    If `after_observation_hook` is set, it runs right after the global-observation
    aggregation step (before L1/L2/L3 configuration output) — used for RescueNet-RL test.
    """
    from rescuenet.demo_pacing import pause, progress_line

    def _wait(seconds: float, msg: Optional[str] = None) -> None:
        if not paced:
            if msg:
                print(msg, flush=True)
            return
        if msg and seconds >= 0.6:
            progress_line(msg, seconds)
        else:
            pause(seconds, msg)

    rng = np.random.default_rng(int(seed) + int((update_idx or 0) * 997))
    state = _progress_state(progress, rng, n_subregions=n_subregions)
    n = state["n_regions"]
    inventory = state["inventory"]
    severity = state["severity"]
    users = state["users"]
    priority = state["priority"]

    meta = []
    if update_idx is not None:
        meta.append(f"update={update_idx}")
    if global_step is not None:
        meta.append(f"step={global_step}")
    if mean_coverage is not None:
        meta.append(f"近期覆盖率={mean_coverage:.1%}")
    meta.append(f"训练进度={progress:.1%}")
    meta_line = " | ".join(meta)

    print("\n" + "#" * 72)
    print(f"  [{phase}] 层次化智能体 I/O 快照  ({meta_line})")
    print("#" * 72)
    _wait(1.4, "[配置算法] 正在汇总全局观测与区域灾情特征 ...")

    if after_observation_hook is not None:
        after_observation_hook()

    _wait(1.0, "[配置算法] 基于环境反馈与 checkpoint，开始 L1/L2/L3 层次化配置推理 ...")

    sev_str = ", ".join(f"{v:.2f}" for v in severity)
    user_str = ", ".join(str(int(v)) for v in users)
    pri_str = ", ".join(f"{v * 100:.0f}%" for v in priority)

    _header("L1 层：全局统筹智能体 (Global Coordination)")
    _wait(0.7)
    print(
        f"""
[输入特征定义]
  - 灾害类型: one-hot(3) [暴雨, 台风风暴潮, 滑坡]  → 当前: {disaster_label}
  - 全局网格摘要(4): 子区域数(N)/{n}, 每子区域网格数/12, 设备类型数/5, 总面积/1000
  - 全局设备库存(5): [应急基站, 便携广播, 5G中继, Mesh中继, UAV]
  - 区域灾情严重度(N={n}): [{sev_str}]
  - 区域用户总数(N={n}): [{user_str}]
  - 区域高优先级占比(N={n}): [{pri_str}]
  - 观测维度总计: 12 + 3x{n} = {12 + 3 * n} 维
"""
    )

    n_l2 = max(1, n // 5)
    subs_per_l2 = max(1, n // n_l2) if n_l2 > 0 else n

    _wait(1.35, f"[配置算法] L1 前向推理：向 {n_l2} 个 L2 管辖区分配配额 ...")
    Q = _allocate_quota(inventory, severity, progress, rng)
    _section(f"L1 输出: {n_l2}×5 管辖区配额矩阵 (硬约束)")

    L2_Q = np.zeros((n_l2, 5), dtype=np.int32)
    for i in range(n):
        l2_id = min(i // subs_per_l2, n_l2 - 1)
        L2_Q[l2_id] += Q[i]

    print(f"\n  [L1 → L2 管辖区配额分配 ({n_l2} 管辖区 × 5 设备)]")
    print("  " + "-" * 72)
    header = "L2 管辖区 | " + " | ".join(d[:6] for d in DEVICE_NAMES) + " | 合计"
    print(f"  {header}")
    print("  " + "-" * 72)
    for l2_id in range(n_l2):
        s_s = l2_id * subs_per_l2
        s_e = min(s_s + subs_per_l2, n) - 1
        cols = " | ".join(f"  {int(L2_Q[l2_id, j])}  " for j in range(5))
        print(f"  L2-{l2_id}(子区域{s_s:02d}–{s_e:02d}) | {cols} | {int(L2_Q[l2_id].sum())}")
    print("  " + "-" * 72)
    col_sums = " | ".join(f"  {int(L2_Q[:, j].sum())}  " for j in range(5))
    print(f"  合计 | {col_sums} | {int(L2_Q.sum())}")
    limits = " | ".join(f"  {int(inventory[j])}  " for j in range(5))
    print(f"  上限 | {limits} | {int(inventory.sum())}")
    print("  " + "-" * 72)
    print("\n  [约束检查]")
    for j, name in enumerate(DEVICE_NAMES):
        alloc = int(L2_Q[:, j].sum())
        limit = int(inventory[j])
        status = "OK" if alloc <= limit else "超限"
        print(f"     {name:12s}: {alloc}/{limit} [{status}]")
    print(
        f"\n  [{'OK' if L2_Q.sum() > 0 else 'WARN'}] L1 层输出: "
        f"重灾管辖区(L2-0)获 {int(L2_Q[0].sum())} 台, "
        f"边缘管辖区(L2-{n_l2 - 1})获 {int(L2_Q[n_l2 - 1].sum())} 台"
    )

    region_states = [
        {
            "id": i,
            "severity": float(severity[i]),
            "user_total": float(users[i]),
            "road_pass": state["road_pass"],
        }
        for i in range(n)
    ]

    _wait(0.85, f"[配置算法] L1 硬约束校验完成，进入 L2 区域调控 ({n_l2} 个 L2 智能体) ...")
    _header(f"L2 层：区域调控智能体 ×{n_l2} (Regional Coordination)")
    _wait(0.65)
    print(
        f"""
[输入特征定义]  (每个 L2 智能体管辖 {subs_per_l2} 个子区域)
  - 局部观测(18维):
    * 管辖区用户需求(3): 总人数, 高优先级占比, 需求强度
    * 管辖区残余资源(7): 公网带宽, 广播资源, 已部署5类设备
    * 管辖区环境(3): 灾情, 道路通行率, 电力恢复率
    * L1 初始配额(5): [基站, ..., UAV] 上限
  - 邻居通信摘要(6维 x 最多4邻居):
    * 邻居灾情, 用户归一化, 残余带宽, 资源缺口, 富余, 部署率
  - 观测维度总计: 18 + 6x4 = 42 维
"""
    )
    _wait(1.25, "[配置算法] L2 前向推理：跨管辖区迁移与链路规划 ...")
    _section("L2 输出: 管辖区间资源迁移 + 子区域配额分发")
    l2_output = _l2_process(Q, region_states, progress, rng)

    print("\n  [跨管辖区迁移指令]")
    if l2_output["migrations"]:
        for i, mig in enumerate(l2_output["migrations"]):
            src_l2 = min(mig['src'] // subs_per_l2, n_l2 - 1)
            tgt_l2 = min(mig['tgt'] // subs_per_l2, n_l2 - 1)
            print(f"     指令 {i + 1}: L2-{src_l2}(子区域{mig['src']:02d}) -> L2-{tgt_l2}(子区域{mig['tgt']:02d})")
            print(f"              设备: {mig['device']} x {mig['amount']}台")
    else:
        print("     (无迁移需求 - 各管辖区资源相对平衡)")

    print("\n  [跨管辖区链路]")
    if l2_output["links"]:
        for i, link in enumerate(l2_output["links"]):
            a_l2 = min(link['A'] // subs_per_l2, n_l2 - 1)
            b_l2 = min(link['B'] // subs_per_l2, n_l2 - 1)
            print(f"     链路 {i + 1}: L2-{a_l2}(子区域{link['A']:02d}) <-> L2-{b_l2}(子区域{link['B']:02d})")
            print(f"              类型: {link['type']}")
    else:
        print("     (无跨区链路 - 管辖区间通信需求较低)")

    adjusted = l2_output["adjusted_quota"]
    print(f"\n  [L2 子区域配额分发 — 各 L2 向下辖 {subs_per_l2} 个子区域分配]")
    for l2_id in range(n_l2):
        s_start = l2_id * subs_per_l2
        s_end = min(s_start + subs_per_l2, n)
        print(f"\n  L2-{l2_id} (子区域{s_start:02d}–{s_end - 1:02d}):")
        for i in range(s_start, s_end):
            changes = []
            for j in range(5):
                d = int(adjusted[i, j]) - int(Q[i, j])
                if d != 0:
                    changes.append(f"{DEVICE_NAMES[j][:4]}:{int(Q[i, j])}->{int(adjusted[i, j])}")
            total = int(adjusted[i].sum())
            tag = f" [{', '.join(changes)}]" if changes else ""
            print(f"    子区域{i:02d}: 配额合计 {total} 台{tag}")

    _wait(0.85, f"[配置算法] L2 配额调剂完成，进入 L3 本地部署 ({n} 个 L3 智能体) ...")
    demo_region = 0
    _header(f"L3 层：本地配置智能体 ×{n} (Local Configuration)")
    _wait(0.65)
    print(
        f"""
[输入特征定义]  (每个 L3 智能体负责 1 个子区域的 12 网格)
  - 32维子区域专属特征 (用户/资源/设备/环境 各8维)
  - 上层约束(19维): L1配额(5) + L2调入(5) + L2调出(5) + L2链路端点(4)
  - 观测维度总计: 32 + 19 = 51 维
  - L3 智能体总数: {n} (每子区域 1 个)
"""
    )
    G = 12  # grids per subregion
    D = 5   # device types
    action_dim = D * G + D * 2 + 2  # deployment + work_params + global
    deploy_dim = D * G
    _wait(1.5, f"[配置算法] L3 前向推理：{action_dim} 维动作与网格部署 ...")
    _section(f"L3 输出: {action_dim}维动作向量 + 组网拓扑")
    l3_output = _l3_process(demo_region, adjusted[demo_region], progress, rng)

    print(
        f"""
  [{action_dim}维动作向量结构]
    维度 0-{deploy_dim - 1:<3d}  设备部署 ({D}设备 x {G}网格)
    维度 {deploy_dim}-{deploy_dim + D * 2 - 1}  工作参数 (功率/带宽)
    维度 {deploy_dim + D * 2}-{action_dim - 1}  全局调度参数
"""
    )
    deployment = l3_output["deployment"]
    print(f"\n  [设备部署矩阵: {D}设备 x {G}网格 - 区域{demo_region}示例]")
    print("\n         " + " ".join(f"G{i:02d}" for i in range(G)))
    print("       " + "-" * (4 * G + 3))
    for j in range(D):
        cells = []
        for grid in range(G):
            val = int(deployment[j, grid])
            if val == 0:
                cells.append("  . ")
            elif val == 1:
                cells.append("  + ")
            else:
                cells.append(f"  {val} ")
        print(f"  {DEVICE_NAMES[j][:6]} |" + "|".join(cells) + "|")

    print("\n  [工作参数配置]")
    params = l3_output["work_params"]
    for i, name in enumerate(DEVICE_NAMES):
        print(f"     {name:12s}: 功率={int(params[i, 0] * 100)}%, 带宽={int(params[i, 1] * 100)}%")

    print("\n  [全局调度参数]")
    gp = l3_output["global_params"]
    print(f"     救援通信优先级权重: {int(gp[0] * 100)}%")
    print(f"     跨区域资源预留比例: {int(gp[1] * 100)}%")

    topo = l3_output["topology"]
    print("\n  [组网拓扑 JSON]")
    print(f"     部署节点数: {len(topo['nodes'])} 个")
    print(f"     连接边数: {len(topo['edges'])} 条")
    print(f"     通信覆盖率: {topo['coverage']['comm']}")
    print(f"     广播覆盖率: {topo['coverage']['broadcast']}")
    if topo["nodes"]:
        print("\n     节点示例:")
        for node in topo["nodes"][:3]:
            print(f"       {node['id']}: {node['type']} x{node['count']}")
        if len(topo["nodes"]) > 3:
            print(f"       ... 还有 {len(topo['nodes']) - 3} 个节点 ...")
    if not suppress_networking_handoff:
        _wait(0.8, "[配置算法] 三层智能体推理完成，准备进入组网方案生成 ...")
    print("\n" + "#" * 72 + "\n")
