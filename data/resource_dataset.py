"""Scenario-aware communication resource dataset utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union


@dataclass
class ModeSnapshot:
    """Stores per-mode availability metadata for a specific timestamp."""

    available_bandwidth: float
    availability: float


@dataclass
class BroadcastSnapshot:
    """Broadcast network snapshot for a timestamp."""

    available_bandwidth: float
    coverage: float


@dataclass
class TimeStepRecord:
    """Aggregated resource metrics for one timestamp."""

    time: int
    mode_metrics: Dict[str, ModeSnapshot]
    broadcast_metrics: Dict[str, BroadcastSnapshot]


@dataclass
class DisasterScenario:
    """Parsed disaster scenario ready for environment consumption."""

    name: str
    disaster_type: str
    grid_size: int
    num_users: int
    candidate_sites: int
    max_steps: int
    has_residual_network: bool
    communication_modes: List[str]
    broadcast_modes: List[str]
    mode_profiles: Dict[str, Dict[str, float]]
    broadcast_profiles: Dict[str, Dict[str, float]]
    user_clusters: List[Dict[str, float]]
    time_series: List[TimeStepRecord]


class ResourceDataset:
    """Loads and validates the multi-mode disaster communication dataset."""

    MIN_COMM_MODES = 4

    def __init__(self, dataset_path: Union[str, Path]) -> None:
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found at {self.dataset_path}")
        raw = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        self._scenarios = {
            entry["name"]: self._parse_scenario(entry) for entry in raw.get("scenarios", [])
        }
        if not self._scenarios:
            raise ValueError("Dataset does not contain any scenarios.")

    def list_scenarios(self) -> List[str]:
        return sorted(self._scenarios.keys())

    def get(self, name: str) -> DisasterScenario:
        if name not in self._scenarios:
            raise KeyError(f"Scenario '{name}' not found in dataset.")
        return self._scenarios[name]

    def _parse_scenario(self, data: Dict) -> DisasterScenario:
        comm_modes = data.get("communication_modes", [])
        if len(comm_modes) < self.MIN_COMM_MODES:
            raise ValueError(
                f"Scenario {data.get('name')} must define at least {self.MIN_COMM_MODES} communication modes"
            )
        broadcast_modes = data.get("broadcast_modes", [])
        time_series = [self._parse_time_step(step) for step in data.get("time_series", [])]
        if not time_series:
            raise ValueError(f"Scenario {data.get('name')} must include time-series metrics")
        return DisasterScenario(
            name=data["name"],
            disaster_type=data["disaster_type"],
            grid_size=int(data["grid_size"]),
            num_users=int(data["num_users"]),
            candidate_sites=int(data["candidate_sites"]),
            max_steps=int(data["max_steps"]),
            has_residual_network=bool(data.get("has_residual_network", False)),
            communication_modes=list(comm_modes),
            broadcast_modes=list(broadcast_modes),
            mode_profiles=data.get("mode_profiles", {}),
            broadcast_profiles=data.get("broadcast_profiles", {}),
            user_clusters=data.get("user_clusters", []),
            time_series=time_series,
        )

    def _parse_time_step(self, step: Dict) -> TimeStepRecord:
        mode_metrics = {
            mode: ModeSnapshot(
                available_bandwidth=float(values["available_bandwidth"]),
                availability=float(values["availability"]),
            )
            for mode, values in step.get("mode_metrics", {}).items()
        }
        broadcast_metrics = {
            mode: BroadcastSnapshot(
                available_bandwidth=float(values["available_bandwidth"]),
                coverage=float(values["coverage"]),
            )
            for mode, values in step.get("broadcast_metrics", {}).items()
        }
        if not mode_metrics:
            raise ValueError("Time step is missing mode metrics")
        if not broadcast_metrics:
            raise ValueError("Time step is missing broadcast metrics")
        return TimeStepRecord(
            time=int(step["time"]),
            mode_metrics=mode_metrics,
            broadcast_metrics=broadcast_metrics,
        )


def load_dataset(dataset_path: Union[str, Path]) -> ResourceDataset:
    """Helper to load the dataset from disk."""
    return ResourceDataset(dataset_path)
