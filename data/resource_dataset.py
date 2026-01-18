"""Scenario-aware communication resource dataset utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


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
    mode_profiles: Dict[str, Dict[str, Any]]
    broadcast_profiles: Dict[str, Dict[str, Any]]
    user_clusters: List[Dict[str, float]]
    time_series: List[TimeStepRecord]
    base_station_profiles: Dict[str, "BaseStationProfile"]
    reward_profiles: Dict[str, "RewardProfile"]
    default_reward_profile: Optional[str] = None

    def get_reward_profile(self, key: Optional[str] = None) -> "RewardProfile":
        """Return reward profile by key or fallback to scenario default."""
        if key and key in self.reward_profiles:
            return self.reward_profiles[key]
        if self.default_reward_profile and self.default_reward_profile in self.reward_profiles:
            return self.reward_profiles[self.default_reward_profile]
        if not self.reward_profiles:
            raise ValueError(f"Scenario {self.name} does not define reward profiles.")
        return next(iter(self.reward_profiles.values()))

    def get_base_station_for_mode(self, mode: str) -> Optional["BaseStationProfile"]:
        """Resolve the base-station profile that backs a communication mode."""
        profile = self.mode_profiles.get(mode)
        if not profile:
            return None
        key = profile.get("base_station")
        if key and key in self.base_station_profiles:
            return self.base_station_profiles[key]
        # Fall back to first profile that lists the mode under supported modes.
        for entry in self.base_station_profiles.values():
            if mode in entry.supported_modes:
                return entry
        return None


@dataclass
class BaseStationProfile:
    """Describes deployable base-station hardware limits and costs."""

    name: str
    label: str
    max_throughput: float
    max_users: int
    device_cost: float
    bandwidth_cost: float
    supported_modes: List[str]


@dataclass
class RewardProfile:
    """Scenario-specific reward configuration weights."""

    key: str
    label: str
    description: str
    coverage_weight: float
    bandwidth_weight: float
    throughput_weight: float
    broadcast_weight: float
    device_cost_weight: float
    bandwidth_cost_weight: float


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
        base_stations = self._parse_base_stations(data.get("base_stations", {}))
        reward_profiles = self._parse_reward_profiles(data.get("reward_profiles", {}), data.get("name"))
        default_reward = data.get("default_reward_profile")
        if not default_reward or default_reward not in reward_profiles:
            default_reward = next(iter(reward_profiles.keys()))
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
            base_station_profiles=base_stations,
            reward_profiles=reward_profiles,
            default_reward_profile=default_reward,
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

    def _parse_base_stations(self, table: Dict[str, Dict[str, Any]]) -> Dict[str, BaseStationProfile]:
        profiles: Dict[str, BaseStationProfile] = {}
        for name, entry in table.items():
            profiles[name] = BaseStationProfile(
                name=name,
                label=str(entry.get("label", name)),
                max_throughput=float(entry.get("max_throughput", entry.get("max_bandwidth", 0.0))),
                max_users=int(entry.get("max_users", 50)),
                device_cost=float(entry.get("device_cost", 1.0)),
                bandwidth_cost=float(entry.get("bandwidth_cost", 0.0)),
                supported_modes=list(entry.get("supported_modes", [])),
            )
        return profiles

    def _parse_reward_profiles(
        self, table: Dict[str, Dict[str, Any]], scenario_name: Optional[str]
    ) -> Dict[str, RewardProfile]:
        profiles: Dict[str, RewardProfile] = {}
        if not table:
            raise ValueError(f"Scenario {scenario_name} must define at least one reward profile.")
        for key, entry in table.items():
            profiles[key] = RewardProfile(
                key=key,
                label=str(entry.get("label", key)),
                description=str(entry.get("description", "")),
                coverage_weight=float(entry.get("coverage_weight", 1.0)),
                bandwidth_weight=float(entry.get("bandwidth_weight", 0.0)),
                throughput_weight=float(entry.get("throughput_weight", 0.0)),
                broadcast_weight=float(entry.get("broadcast_weight", 0.0)),
                device_cost_weight=float(entry.get("device_cost_weight", 0.0)),
                bandwidth_cost_weight=float(entry.get("bandwidth_cost_weight", 0.0)),
            )
        return profiles


def load_dataset(dataset_path: Union[str, Path]) -> ResourceDataset:
    """Helper to load the dataset from disk."""
    return ResourceDataset(dataset_path)
