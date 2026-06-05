"""Pydantic schemas for the RescueNet-RL API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


AlgorithmKey = Literal["ppo", "dqn", "a3c", "mppo", "hmarl"]
EvaluationProtocol = Literal["standard", "earthquake_stress"]

DisasterScenarioKey = str
DisasterSeverityKey = str
DeviceTypeKey = Literal["5G_700MHz", "Satellite_Ka", "WiFi6", "Shortwave_HF", "custom"]
DeviceStatusKey = Literal["active", "inactive"]


class TrainRequest(BaseModel):
    scenario_name: str = Field(..., description="Scenario key defined in data/scenarios.json")
    env_type: Literal["baseline", "multimodal"] = Field("multimodal", description="Environment variant to train")
    algorithm: AlgorithmKey = Field("ppo", description="Training algorithm.")
    total_timesteps: Optional[int] = Field(None, ge=1000, description="Override PPO total timesteps")
    stochastic_eval: bool = Field(True, description="Use stochastic actions during eval")
    reward_mode: Optional[str] = Field(
        None, description="Reward profile key to override scenario defaults when env-type=multimodal."
    )
    evaluation_protocol: Optional[EvaluationProtocol] = Field(
        None, description="Named training/evaluation protocol; earthquake_stress enables the high-intensity earthquake benchmark."
    )
    learning_rate: Optional[float] = Field(None, gt=0, description="Optimizer learning rate override.")
    discount_factor: Optional[float] = Field(None, gt=0, le=1, description="Discount factor gamma override.")
    batch_size: Optional[int] = Field(None, ge=1, description="Mini-batch or replay batch size override.")
    rollout_steps: Optional[int] = Field(None, ge=1, description="On-policy rollout step override.")
    entropy_coef: Optional[float] = Field(None, ge=0, description="Entropy coefficient override.")
    clip_range: Optional[float] = Field(None, gt=0, description="Policy clip coefficient override.")
    eval_interval: Optional[int] = Field(None, ge=1, description="Evaluation interval override.")
    custom_base_stations: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Residual base-station deployments for this training run; omitted uses saved scenario device state.",
    )


class TrainResponse(BaseModel):
    run_id: str


class TrainingStatus(BaseModel):
    run_id: str
    status: str
    scenario_name: str
    env_type: str
    algorithm: str
    reward_mode: Optional[str] = None
    evaluation_protocol: Optional[str] = None
    started_at: float
    updated_at: float
    error: Optional[str] = None


class CustomDevice(BaseModel):
    x: int = Field(..., description="Region grid row index (formerly X)")
    y: int = Field(..., description="Region grid column index (formerly Y)")
    demand: float = Field(10.0, description="Demand in Mbps")
    connected: bool = Field(False, description="Initial connectivity flag")
    broadcast_served: bool = Field(False, description="Initial broadcast coverage flag")
    device_id: Optional[str] = Field(None, description="Dedicated device ID snapshot.")
    device_name: Optional[str] = Field(None, description="Dedicated device name snapshot.")
    device_type: Optional[str] = Field(None, description="Dedicated device communication type snapshot.")
    is_dedicated: bool = Field(False, description="Whether this user/device entry comes from a dedicated device.")


class CustomBaseStation(BaseModel):
    model_config = ConfigDict(extra="allow")

    device_uid: Optional[str] = Field(None, description="Stable per-scene device identifier.")
    deployment_id: Optional[str] = Field(None, description="Source deployment identifier from disaster imports.")
    x: int = Field(..., description="Region grid row index for the residual base station")
    y: int = Field(..., description="Region grid column index for the residual base station")
    base_station: str = Field(..., description="Base-station profile key defined in the scenario dataset.")
    mode: Optional[str] = Field(
        None, description="Communication mode to activate; defaults to the first supported mode of the base-station type."
    )
    status: Optional[str] = Field(
        None, description="Operational status for the station; offline stations are retained but do not cover users."
    )
    device_name: Optional[str] = Field(None, description="Operator-facing device instance name.")
    device_category: Optional[str] = Field(None, description="Operator-facing device category.")
    station_type: Optional[str] = Field(None, description="Source station type from disaster dataset.")
    station_label: Optional[str] = Field(None, description="Source station label from disaster dataset.")
    cell_user_count: Optional[int] = Field(None, ge=0, description="Users attached to this cell in the imported dataset.")
    coverage_radius: Optional[float] = Field(None, ge=0, description="Coverage radius in environment grid units.")
    coverage_radius_km: Optional[float] = Field(None, ge=0, description="Coverage radius in kilometers.")
    max_throughput: Optional[float] = Field(None, ge=0, description="Downlink throughput capacity in Mbps.")
    max_users: Optional[int] = Field(None, ge=0, description="Maximum served users for this device instance.")
    downlink_bandwidth_mbps: Optional[float] = Field(None, ge=0)
    uplink_bandwidth_mbps: Optional[float] = Field(None, ge=0)
    tx_power_watt: Optional[float] = Field(None, ge=0)
    battery_duration_h: Optional[float] = Field(None, ge=0)
    notes: Optional[str] = None


class ScenarioBaseStationUpdate(BaseModel):
    base_stations: List[CustomBaseStation] = Field(
        default_factory=list,
        description="Complete base-station deployment list to persist for the scenario.",
    )


class ScenarioBaseStationResponse(BaseModel):
    scenario_name: str
    base_stations: List[Dict[str, Any]]
    updated_at: Optional[float] = None


class ScenarioDeviceTypeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    device_name: Optional[str] = None
    device_category: Optional[str] = None
    mode: Optional[str] = None
    status: Optional[str] = None
    coverage_radius: Optional[float] = Field(None, ge=0)
    coverage_radius_km: Optional[float] = Field(None, ge=0)
    max_throughput: Optional[float] = Field(None, ge=0)
    max_users: Optional[int] = Field(None, ge=0)
    downlink_bandwidth_mbps: Optional[float] = Field(None, ge=0)
    uplink_bandwidth_mbps: Optional[float] = Field(None, ge=0)
    tx_power_watt: Optional[float] = Field(None, ge=0)
    battery_duration_h: Optional[float] = Field(None, ge=0)
    notes: Optional[str] = None


class ScenarioDeviceStateUpdate(BaseModel):
    base_stations: Optional[List[CustomBaseStation]] = Field(
        None, description="Complete per-device station list for this scenario."
    )
    type_overrides: Optional[Dict[str, ScenarioDeviceTypeConfig]] = Field(
        None, description="Parameter overrides applied by base-station type or concrete device model."
    )
    operation: Optional[str] = Field(None, description="Audit trail operation label.")


class ScenarioDeviceBlockUpdate(BaseModel):
    x: int = Field(..., description="Region grid row index.")
    y: int = Field(..., description="Region grid column index.")
    base_station: str = Field(..., description="Base-station profile key.")
    mode: Optional[str] = None
    status: Optional[str] = "active"
    quantity: int = Field(..., ge=0, description="Desired station count in this grid block.")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    operation: Optional[str] = None


class ScenarioDeviceStateResponse(BaseModel):
    scenario_name: str
    display_name: Optional[str] = None
    grid: Dict[str, Any]
    device_types: List[Dict[str, Any]]
    device_models: List[Dict[str, Any]] = Field(default_factory=list)
    devices: List[Dict[str, Any]]
    blocks: List[Dict[str, Any]]
    status_counts: Dict[str, int]
    type_overrides: Dict[str, Dict[str, Any]]
    history: List[Dict[str, Any]]
    updated_at: Optional[float] = None


class DedicatedDeviceCreate(BaseModel):
    device_name: str = Field(..., min_length=1, max_length=100, description="设备名称")
    device_type: DeviceTypeKey = Field("custom", description="通信制式")
    device_category: str = Field("其他", description="设备类别")
    coverage_radius_km: float = Field(1.0, ge=0.01, description="覆盖半径（km）")
    downlink_bandwidth_mbps: float = Field(10.0, ge=0, description="下行带宽（Mbps）")
    uplink_bandwidth_mbps: float = Field(5.0, ge=0, description="上行带宽（Mbps）")
    max_users: int = Field(50, ge=1, description="最大接入用户数")
    tx_power_watt: Optional[float] = Field(None, ge=0)
    battery_duration_h: Optional[float] = Field(None, ge=0)
    supported_modes: List[str] = Field(default_factory=list)
    image_url: Optional[str] = None
    bound_scenario: Optional[str] = None


class DedicatedDeviceUpdate(BaseModel):
    device_name: Optional[str] = Field(None, min_length=1, max_length=100)
    device_type: Optional[DeviceTypeKey] = None
    device_category: Optional[str] = None
    coverage_radius_km: Optional[float] = Field(None, ge=0.01)
    downlink_bandwidth_mbps: Optional[float] = Field(None, ge=0)
    uplink_bandwidth_mbps: Optional[float] = Field(None, ge=0)
    max_users: Optional[int] = Field(None, ge=1)
    tx_power_watt: Optional[float] = Field(None, ge=0)
    battery_duration_h: Optional[float] = Field(None, ge=0)
    supported_modes: Optional[List[str]] = None
    image_url: Optional[str] = None
    bound_scenario: Optional[str] = None
    deploy_position: Optional[Dict[str, float]] = None


class DedicatedDeviceStatusUpdate(BaseModel):
    status: DeviceStatusKey


class DedicatedDevice(DedicatedDeviceCreate):
    device_id: str
    is_dedicated: bool = True
    status: DeviceStatusKey = "active"
    deploy_position: Optional[Dict[str, float]] = None
    created_at: str
    updated_at: str


class DedicatedDeviceListResponse(BaseModel):
    devices: List[DedicatedDevice]
    total: int
    active_count: int


class SceneImportRequest(BaseModel):
    scenario_name: str = Field("typhoon_residual", description="Scenario to import as a disaster scene.")
    env_type: Literal["baseline", "multimodal"] = Field("multimodal", description="Environment variant.")
    evaluation_protocol: Optional[EvaluationProtocol] = Field(
        None, description="Named protocol used to materialize the imported scene."
    )
    custom_base_stations: Optional[List[CustomBaseStation]] = Field(
        default=None,
        description="Residual base stations to materialize in the imported scene; empty list means fully damaged.",
    )
    dataset_import_ids: List[str] = Field(
        default_factory=list,
        description="Disaster dataset import session ids selected as the source for this scene preview.",
    )


class SceneImportResponse(BaseModel):
    scenario: Dict[str, Any]
    initial_state: Dict[str, Any]
    scene: Dict[str, Any]


class SimulationRequest(BaseModel):
    scenario_name: str = Field("typhoon_residual", description="Scenario to use as baseline.")
    checkpoint_path: str = Field("artifacts/ppo_policy.pt", description="Policy checkpoint to load.")
    env_type: Literal["baseline", "multimodal"] = Field("multimodal", description="Environment variant.")
    algorithm: AlgorithmKey = Field("ppo", description="Policy algorithm.")
    reward_mode: Optional[str] = Field(
        None, description="Reward profile key to align evaluation with the training configuration."
    )
    evaluation_protocol: Optional[EvaluationProtocol] = Field(
        None, description="Named evaluation protocol; use earthquake_stress for the high-intensity earthquake benchmark."
    )
    episodes: int = Field(1, ge=1, description="Evaluation episodes to run.")
    stochastic_eval: bool = Field(True, description="Sample actions during evaluation.")
    eval_seed: Optional[int] = Field(None, ge=0, description="Optional random seed for reproducible stochastic evaluation.")
    custom_devices: List[CustomDevice] = Field(default_factory=list, description="Custom device definitions.")
    custom_base_stations: Optional[List[CustomBaseStation]] = Field(
        default=None,
        description="Residual base-station deployments; empty list means fully damaged with no residual network.",
    )
    dataset_import_ids: List[str] = Field(
        default_factory=list,
        description="Disaster dataset import session ids selected as simulation data sources.",
    )
    replay_source: Optional[Literal["test", "training", "manual"]] = Field(
        None,
        description="Replay session source label; defaults to test for strategy simulations.",
    )


class SceneExport(BaseModel):
    disaster_scene_path: str
    deployment_scene_path: str
    deployment_plan_path: Optional[str] = None
    disaster_scene: Dict[str, Any]
    deployment_scene: Dict[str, Any]
    deployment_plan: Optional[Dict[str, Any]] = None


class SimulationResponse(BaseModel):
    avg_reward: float
    avg_final_coverage: float
    reports: List[Dict[str, Any]]
    deployment_plan: Optional[Dict[str, Any]] = None
    scene_export: Optional[SceneExport] = None
    replay_session_id: Optional[str] = None
    replay_session_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Disaster data import schemas
# ---------------------------------------------------------------------------

class GeoBounds(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class GridSize(BaseModel):
    rows: int
    cols: int


class GridPosition(BaseModel):
    row: int
    col: int


class CoverageRadiusRange(BaseModel):
    min: float
    max: float


class StationCounts(BaseModel):
    total: int
    active: int
    degraded: int
    offline: int


class DisasterImportRequest(BaseModel):
    disaster_scenario: DisasterScenarioKey = Field(..., description="灾害场景名称")
    disaster_severity: DisasterSeverityKey = Field(..., description="灾害烈度等级")
    session_sample_limit: int = Field(100, ge=1, le=500, description="每个基站采样的最大用户会话数")


class DeploymentItem(BaseModel):
    deployment_id: str
    station_type: str
    station_label: str
    comm_type: str
    comm_label: str
    status: str
    grid_position: GridPosition
    downlink_bandwidth_mbps_avg: float
    coverage_radius_km: float
    cell_user_count: int


class HeatmapCell(BaseModel):
    grid_row: int
    grid_col: int
    user_count: int


class DisasterImportSummary(BaseModel):
    import_id: str
    disaster_scenario: str
    disaster_scenario_label: str
    disaster_severity: str
    disaster_severity_label: str
    session_sample_limit: int
    status: str
    imported_at: str
    effective_geo_bounds: GeoBounds
    grid_size: GridSize
    station_counts: StationCounts
    unique_user_count: int
    total_sessions_sampled: int
    comm_type_breakdown: Dict[str, int]


class DisasterImportDetail(DisasterImportSummary):
    deployments: List[DeploymentItem]
    user_heatmap: List[HeatmapCell]


class DisasterImportListResponse(BaseModel):
    imports: List[DisasterImportSummary]
    total: int


# ---------------------------------------------------------------------------
# Mahimahi schemas
# ---------------------------------------------------------------------------

class MahimahiSimulateRequest(BaseModel):
    trace_name: str = Field(..., description="Name of the trace file (without .trace extension)")
    duration_s: float = Field(60.0, ge=1, le=300, description="Simulation duration in seconds")
    rtt_ms: float = Field(80.0, ge=1, le=2000, description="Round-trip time in milliseconds")
    buffer_packets: int = Field(100, ge=1, le=10000, description="Buffer size in packets")
    window_ms: int = Field(500, ge=100, le=5000, description="Aggregation window in milliseconds")


class MahimahiTraceInfo(BaseModel):
    name: str
    filename: str
    period_ms: int
    total_packets: int
    avg_throughput_mbps: float


class MahimahiTimePoint(BaseModel):
    time_s: float
    value: float


class MahimahiStats(BaseModel):
    avg_capacity_mbps: float
    avg_throughput_mbps: float
    avg_sending_rate_mbps: float
    utilization: float
    loss_rate: float
    total_delivered_mb: float


class MahimahiSimulateResponse(BaseModel):
    trace_name: str
    duration_s: float
    rtt_ms: float
    buffer_packets: int
    window_ms: int
    mahimahi_native: bool
    capacity: List[Dict[str, Any]]
    throughput: List[Dict[str, Any]]
    sending_rate: List[Dict[str, Any]]
    stats: MahimahiStats
