"""Scenario-aware communication resource dataset utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


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
    region_grid: "RegionGrid"
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
        region_grid = self._parse_region_grid(data.get("region_grid"), int(data["grid_size"]), data.get("name"))
        return DisasterScenario(
            name=data["name"],
            disaster_type=data["disaster_type"],
            grid_size=int(data["grid_size"]),
            region_grid=region_grid,
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

    def _parse_region_grid(self, raw: Optional[Dict[str, Any]], grid_size: int, scenario_name: Optional[str]) -> "RegionGrid":
        """Map grid cells to real-world latitude/longitude bounds and human labels."""
        rows = int(raw.get("rows", grid_size)) if raw else grid_size
        cols = int(raw.get("cols", grid_size)) if raw else grid_size
        bounds = raw.get("geo_bounds", {}) if raw else {}
        lat_min = float(bounds.get("lat_min", 0.0))
        lat_max = float(bounds.get("lat_max", float(rows)))
        lon_min = float(bounds.get("lon_min", 0.0))
        lon_max = float(bounds.get("lon_max", float(cols)))
        labels = {str(key): str(value) for key, value in (raw.get("cell_labels", {}) if raw else {}).items()}
        name = str(raw.get("name", f"{scenario_name}_grid")) if raw else f"{scenario_name}_grid"
        return RegionGrid(
            name=name,
            rows=rows,
            cols=cols,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            cell_labels=labels,
        )


def load_dataset(dataset_path: Union[str, Path]) -> ResourceDataset:
    """Helper to load the dataset from disk."""
    return ResourceDataset(dataset_path)


@dataclass
class RegionGrid:
    """Semantic mapping from grid cells to geo bounds and labels."""

    name: str
    rows: int
    cols: int
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    cell_labels: Dict[str, str]

    @property
    def cell_count(self) -> int:
        return self.rows * self.cols

    def cell_index(self, row: int, col: int) -> int:
        return row * self.cols + col

    def normalize_cell_index(self, row: int, col: int) -> float:
        return self.cell_index(row, col) / max(1, self.cell_count - 1)

    def normalize_row(self, row: int) -> float:
        return row / max(1, self.rows - 1)

    def normalize_col(self, col: int) -> float:
        return col / max(1, self.cols - 1)

    def cell_bounds(self, row: int, col: int) -> Tuple[float, float, float, float]:
        lat_step = (self.lat_max - self.lat_min) / max(1, self.rows)
        lon_step = (self.lon_max - self.lon_min) / max(1, self.cols)
        lat0 = self.lat_min + row * lat_step
        lat1 = lat0 + lat_step
        lon0 = self.lon_min + col * lon_step
        lon1 = lon0 + lon_step
        return lat0, lat1, lon0, lon1

    def cell_label(self, row: int, col: int) -> str:
        return self.cell_labels.get(f"{row},{col}", f"cell-{row}-{col}")

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rows": self.rows,
            "cols": self.cols,
            "geo_bounds": {
                "lat_min": self.lat_min,
                "lat_max": self.lat_max,
                "lon_min": self.lon_min,
                "lon_max": self.lon_max,
            },
            "cell_labels": dict(self.cell_labels),
        }
