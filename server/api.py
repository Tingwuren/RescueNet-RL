"""FastAPI application exposing training and simulation endpoints."""

from __future__ import annotations

import asyncio
import json
import queue
from pathlib import Path
from typing import AsyncGenerator, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from configs.default_config import get_default_config
from data.resource_dataset import ResourceDataset
from server.schemas import (
    SimulationRequest,
    SimulationResponse,
    TrainRequest,
    TrainResponse,
    TrainingStatus,
)
from server.training_manager import TrainingManager
from services.evaluation import build_env, evaluate_policy, load_policy

app = FastAPI(title="RescueNet-RL API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

training_manager = TrainingManager()
default_config = get_default_config()
dataset_path = Path(default_config["multimodal_env"]["dataset_path"])
dataset = ResourceDataset(dataset_path)


@app.get("/api/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/scenarios")
def list_scenarios() -> Dict[str, List[Dict[str, object]]]:
    scenarios = []
    for name in dataset.list_scenarios():
        record = dataset.get(name)
        scenarios.append(
            {
                "name": record.name,
                "disaster_type": record.disaster_type,
                "num_users": record.num_users,
                "candidate_sites": record.candidate_sites,
                "max_steps": record.max_steps,
                "has_residual_network": record.has_residual_network,
            }
        )
    return {"scenarios": scenarios}


@app.post("/api/train", response_model=TrainResponse)
def start_training(request: TrainRequest) -> TrainResponse:
    run = training_manager.start_run(
        scenario_name=request.scenario_name,
        env_type=request.env_type,
        total_timesteps=request.total_timesteps,
        stochastic_eval=request.stochastic_eval,
    )
    return TrainResponse(run_id=run.run_id)


@app.get("/api/train/{run_id}", response_model=TrainingStatus)
def get_training_status(run_id: str) -> TrainingStatus:
    run = training_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found.")
    return TrainingStatus(
        run_id=run.run_id,
        status=run.status,
        scenario_name=run.scenario_name,
        env_type=run.env_type,
        started_at=run.started_at,
        updated_at=run.updated_at,
        error=run.error,
    )


@app.get("/api/train/{run_id}/stream")
async def stream_training_events(run_id: str):
    run = training_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found.")

    async def event_generator() -> AsyncGenerator[str, None]:
        while True:
            if run.status in {"completed", "failed"} and run.events.empty():
                break
            try:
                event = await asyncio.to_thread(run.events.get, True, 0.5)
            except queue.Empty:
                continue
            payload = json.dumps(event)
            yield f"data: {payload}\n\n"
        yield f"data: {json.dumps({'type': 'end', 'status': run.status})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/simulate", response_model=SimulationResponse)
def simulate_strategy(request: SimulationRequest) -> SimulationResponse:
    config = get_default_config()
    config["experiment"]["env_type"] = request.env_type
    if request.env_type == "multimodal":
        config["multimodal_env"]["scenario_name"] = request.scenario_name

    checkpoint_path = Path(request.checkpoint_path)
    env = build_env(config, request.env_type)
    policy = load_policy(checkpoint_path, env, config, request.env_type)

    custom_state = [device.dict() for device in request.custom_devices]
    rewards, coverages, reports = evaluate_policy(
        env=env,
        policy=policy,
        episodes=request.episodes,
        deterministic=not request.stochastic_eval,
        render=False,
        custom_user_state=custom_state or None,
    )

    return SimulationResponse(
        avg_reward=float(np.mean(rewards)),
        avg_final_coverage=float(np.mean(coverages)),
        reports=reports,
    )
