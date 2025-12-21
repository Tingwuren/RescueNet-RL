"""Pydantic schemas for the RescueNet-RL API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    scenario_name: str = Field(..., description="Scenario key defined in data/scenarios.json")
    env_type: Literal["baseline", "multimodal"] = Field("multimodal", description="Environment variant to train")
    total_timesteps: Optional[int] = Field(None, ge=1000, description="Override PPO total timesteps")
    stochastic_eval: bool = Field(True, description="Use stochastic actions during eval")


class TrainResponse(BaseModel):
    run_id: str


class TrainingStatus(BaseModel):
    run_id: str
    status: str
    scenario_name: str
    env_type: str
    started_at: float
    updated_at: float
    error: Optional[str] = None


class CustomDevice(BaseModel):
    x: int = Field(..., description="Grid X coordinate")
    y: int = Field(..., description="Grid Y coordinate")
    demand: float = Field(10.0, description="Demand in Mbps")
    connected: bool = Field(False, description="Initial connectivity flag")
    broadcast_served: bool = Field(False, description="Initial broadcast coverage flag")


class SimulationRequest(BaseModel):
    scenario_name: str = Field("typhoon_residual", description="Scenario to use as baseline.")
    checkpoint_path: str = Field("artifacts/ppo_policy.pt", description="Policy checkpoint to load.")
    env_type: Literal["baseline", "multimodal"] = Field("multimodal", description="Environment variant.")
    episodes: int = Field(1, ge=1, description="Evaluation episodes to run.")
    stochastic_eval: bool = Field(True, description="Sample actions during evaluation.")
    custom_devices: List[CustomDevice] = Field(default_factory=list, description="Custom device definitions.")


class SimulationResponse(BaseModel):
    avg_reward: float
    avg_final_coverage: float
    reports: List[Dict[str, Any]]
