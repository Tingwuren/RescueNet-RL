"""Pydantic schemas for the RescueNet-RL API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    scenario_name: str = Field(..., description="Scenario key defined in data/scenarios.json")
    env_type: Literal["baseline", "multimodal"] = Field("multimodal", description="Environment variant to train")
    algorithm: Literal["ppo", "dqn", "a3c", "mppo"] = Field("ppo", description="Training algorithm.")
    total_timesteps: Optional[int] = Field(None, ge=1000, description="Override PPO total timesteps")
    stochastic_eval: bool = Field(True, description="Use stochastic actions during eval")
    reward_mode: Optional[str] = Field(
        None, description="Reward profile key to override scenario defaults when env-type=multimodal."
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
    started_at: float
    updated_at: float
    error: Optional[str] = None


class CustomDevice(BaseModel):
    x: int = Field(..., description="Region grid row index (formerly X)")
    y: int = Field(..., description="Region grid column index (formerly Y)")
    demand: float = Field(10.0, description="Demand in Mbps")
    connected: bool = Field(False, description="Initial connectivity flag")
    broadcast_served: bool = Field(False, description="Initial broadcast coverage flag")


class CustomBaseStation(BaseModel):
    x: int = Field(..., description="Region grid row index for the residual base station")
    y: int = Field(..., description="Region grid column index for the residual base station")
    base_station: str = Field(..., description="Base-station profile key defined in the scenario dataset.")
    mode: Optional[str] = Field(
        None, description="Communication mode to activate; defaults to the first supported mode of the base-station type."
    )


class SceneImportRequest(BaseModel):
    scenario_name: str = Field("typhoon_residual", description="Scenario to import as a disaster scene.")
    env_type: Literal["baseline", "multimodal"] = Field("multimodal", description="Environment variant.")
    custom_base_stations: Optional[List[CustomBaseStation]] = Field(
        default=None,
        description="Residual base stations to materialize in the imported scene; empty list means fully damaged.",
    )


class SceneImportResponse(BaseModel):
    scenario: Dict[str, Any]
    initial_state: Dict[str, Any]
    scene: Dict[str, Any]


class SimulationRequest(BaseModel):
    scenario_name: str = Field("typhoon_residual", description="Scenario to use as baseline.")
    checkpoint_path: str = Field("artifacts/ppo_policy.pt", description="Policy checkpoint to load.")
    env_type: Literal["baseline", "multimodal"] = Field("multimodal", description="Environment variant.")
    algorithm: Literal["ppo", "dqn", "a3c", "mppo"] = Field("ppo", description="Policy algorithm.")
    reward_mode: Optional[str] = Field(
        None, description="Reward profile key to align evaluation with the training configuration."
    )
    episodes: int = Field(1, ge=1, description="Evaluation episodes to run.")
    stochastic_eval: bool = Field(True, description="Sample actions during evaluation.")
    eval_seed: Optional[int] = Field(None, ge=0, description="Optional random seed for reproducible stochastic evaluation.")
    custom_devices: List[CustomDevice] = Field(default_factory=list, description="Custom device definitions.")
    custom_base_stations: Optional[List[CustomBaseStation]] = Field(
        default=None,
        description="Residual base-station deployments; empty list means fully damaged with no residual network.",
    )


class SceneExport(BaseModel):
    disaster_scene_path: str
    deployment_scene_path: str
    disaster_scene: Dict[str, Any]
    deployment_scene: Dict[str, Any]


class SimulationResponse(BaseModel):
    avg_reward: float
    avg_final_coverage: float
    reports: List[Dict[str, Any]]
    scene_export: Optional[SceneExport] = None
