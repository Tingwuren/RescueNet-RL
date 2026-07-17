"""FastAPI application exposing training and simulation endpoints."""

from __future__ import annotations

import asyncio
import json
import queue
import re
import threading
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

from configs.default_config import apply_evaluation_protocol, apply_level4_algorithm_profile, get_default_config
from data.resource_dataset import ResourceDataset
from server.schemas import (
    DedicatedDevice,
    DedicatedDeviceCreate,
    DedicatedDeviceListResponse,
    DedicatedDeviceStatusUpdate,
    DedicatedDeviceUpdate,
    DeviceStatusKey,
    DeviceTypeKey,
    CustomBaseStation,
    DisasterImportDetail,
    DisasterImportListResponse,
    DisasterImportRequest,
    DisasterImportSummary,
    MahimahiSimulateRequest,
    MahimahiSimulateResponse,
    MahimahiTraceInfo,
    ScenarioBaseStationResponse,
    ScenarioBaseStationUpdate,
    ScenarioDeviceBlockUpdate,
    ScenarioDeviceStateResponse,
    ScenarioDeviceStateUpdate,
    SceneImportRequest,
    SceneImportResponse,
    SimulationRequest,
    SimulationResponse,
    TrainRequest,
    TrainResponse,
    TrainingStatus,
)
from server.device_manager import DeviceManager
from server.scenario_device_manager import ScenarioDeviceManager
from server.training_manager import TrainingManager
from server.mahimahi_manager import MahimahiManager
from server.ns3_replay_manager import Ns3ReplayManager
from server.replay_manager import ReplaySessionManager
from server.disaster_import_manager import DisasterImportManager
from services.evaluation import build_env, build_scene_preview, evaluate_policy, export_episode_scene, load_policy

app = FastAPI(title="RescueNet-RL API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

training_manager = TrainingManager()
mahimahi_manager = MahimahiManager()
ns3_replay_manager = Ns3ReplayManager()
replay_session_manager = ReplaySessionManager()
default_config = get_default_config()
dataset_path = Path(default_config["multimodal_env"]["dataset_path"])
dataset = ResourceDataset(dataset_path)
disaster_import_manager = DisasterImportManager(Path("data/extreme_disaster_resources"))
device_manager = DeviceManager(Path(os.environ.get("RESCUENET_DEVICE_DB", "data/dedicated_devices.db")))
scenario_device_manager = ScenarioDeviceManager(
    Path(os.environ.get("RESCUENET_SCENARIO_DEVICE_DB", "data/scenario_devices.db"))
)
TRAINING_RUN_DIR_RE = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{6})_")
SCENE_EXPORT_DIR_RE = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{6})_")
scenario_state_path = Path(os.environ.get("RESCUENET_SCENARIO_STATE", "data/scenario_state.json"))
scenario_state_lock = threading.Lock()
scenarios_cache_lock = threading.RLock()
scenarios_response_cache: Optional[bytes] = None
candidate_site_preview_cache: Dict[str, List[Dict[str, object]]] = {}

DEVICE_CONFIG_FIELDS = (
    "device_name",
    "device_category",
    "coverage_radius",
    "coverage_radius_km",
    "max_throughput",
    "max_users",
    "downlink_bandwidth_mbps",
    "uplink_bandwidth_mbps",
    "tx_power_watt",
    "battery_duration_h",
    "notes",
)
DEVICE_PRESERVED_FIELDS = (
    "device_uid",
    "deployment_id",
    "station_type",
    "station_label",
    "cell_user_count",
    "override_fields",
    *DEVICE_CONFIG_FIELDS,
)
TYPE_OVERRIDE_FIELDS = tuple(field for field in DEVICE_CONFIG_FIELDS if field != "device_name")
DEVICE_STATUS_VALUES = {"active", "degraded", "offline", "planned", "deployed", "unknown"}

STATION_TYPE_LABELS = {
    "backpack_micro_cell": "背负式 5G 700MHz 微站",
    "low_band_macro_cell": "低频宏基站",
    "temporary_macro_cell": "临时宏基站",
    "fixed_satellite_gateway": "固定 Ka 应急卫星网关",
    "vehicle_satellite_terminal": "车载 Ka 卫星终端",
    "command_vehicle_radio": "指挥车短波电台",
    "field_shortwave_station": "野战短波台",
    "portable_hotspot": "便携式 WiFi 6 热点",
    "shelter_mesh_node": "避难所 WiFi 6 Mesh 节点",
    "vehicle_wifi_node": "车载 WiFi 6 节点",
}


def _dedicated_device_log_messages(custom_state: List[Dict[str, Any]]) -> List[str]:
    messages: List[str] = []
    for device in custom_state:
        if not bool(device.get("is_dedicated")):
            continue
        row = device.get("x", "--")
        col = device.get("y", "--")
        messages.append(
            "[专用设备] "
            f"device_id={device.get('device_id') or '--'} "
            f"device_name={device.get('device_name') or '--'} "
            f"device_type={device.get('device_type') or '--'} "
            f"部署位置=(row={row}, col={col}) 参与仿真"
        )
    return messages


def _attach_replay_session(request: SimulationRequest, response: SimulationResponse) -> SimulationResponse:
    session = replay_session_manager.create_from_simulation(
        request.model_dump(),
        response.model_dump(),
        source=request.replay_source or "test",
    )
    response.replay_session_id = str(session.get("replay_id") or "")
    response.replay_session_path = str((session.get("artifacts") or {}).get("session_dir") or "")
    return response


STREAM_USER_DETAIL_LIMIT = 120


def _compact_stream_state(state: Optional[Dict[str, Any]], *, keep_user_sample: bool = False) -> Dict[str, Any]:
    compact = dict(state or {})
    user_details = compact.get("user_details")
    if isinstance(user_details, list):
        compact["user_details_total"] = len(user_details)
        compact["user_details"] = user_details[:STREAM_USER_DETAIL_LIMIT] if keep_user_sample else []
        compact["user_details_truncated"] = len(user_details) > len(compact["user_details"])
    return compact


def _compact_stream_report(report: Dict[str, Any]) -> Dict[str, Any]:
    compact = dict(report or {})
    compact["initial_state"] = _compact_stream_state(compact.get("initial_state"), keep_user_sample=False)
    compact["final_state"] = _compact_stream_state(compact.get("final_state"), keep_user_sample=True)
    compact_steps = []
    for step in compact.get("steps") or []:
        next_step = dict(step or {})
        if isinstance(next_step.get("post_state"), dict):
            next_step["post_state"] = _compact_stream_state(next_step.get("post_state"), keep_user_sample=False)
        compact_steps.append(next_step)
    compact["steps"] = compact_steps
    return compact


def _compact_stream_response(response: SimulationResponse) -> SimulationResponse:
    payload = response.model_dump()
    payload["reports"] = [_compact_stream_report(report) for report in payload.get("reports") or []]
    return SimulationResponse(**payload)


def _read_scenario_state() -> Dict[str, Any]:
    if not scenario_state_path.exists():
        return {}
    try:
        raw = json.loads(scenario_state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _write_scenario_state(state: Dict[str, Any]) -> None:
    scenario_state_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _legacy_scenario_state_entry(scenario_name: str) -> Optional[Dict[str, Any]]:
    state = _read_scenario_state()
    entry = state.get(scenario_name)
    return entry if isinstance(entry, dict) else None


def _scenario_grid_public(record: Any) -> Dict[str, Any]:
    return record.region_grid.to_public_dict() if getattr(record, "region_grid", None) else {
        "rows": record.grid_size,
        "cols": record.grid_size,
    }


def _scenario_state_entry(scenario_name: str) -> Optional[Dict[str, Any]]:
    record = dataset.get(scenario_name)
    scenario_exists = scenario_device_manager.has_scenario(scenario_name)
    default_specs: List[Dict[str, Any]] = []
    legacy_entry = None
    if not scenario_exists:
        default_specs = _normalize_scenario_base_stations(
            scenario_name,
            _default_scenario_base_stations(scenario_name),
        )
        legacy_entry = _legacy_scenario_state_entry(scenario_name)
        if legacy_entry and isinstance(legacy_entry.get("base_stations"), list):
            legacy_entry = {
                **legacy_entry,
                "base_stations": _normalize_scenario_base_stations(
                    scenario_name,
                    legacy_entry["base_stations"],
                ),
            }
    scenario_device_manager.ensure_scenario(
        scenario_name,
        display_name=record.display_name or record.name,
        disaster_type=record.disaster_type,
        source_scenario=getattr(record, "source_scenario", None),
        severity_level=getattr(record, "severity_level", None),
        grid=_scenario_grid_public(record),
        default_base_stations=default_specs,
        legacy_entry=legacy_entry,
    )
    entry = scenario_device_manager.get_state(scenario_name)
    return entry if isinstance(entry, dict) else None


def _clean_device_config(data: Optional[Dict[str, Any]], allowed_fields: tuple[str, ...] = DEVICE_CONFIG_FIELDS) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    cleaned: Dict[str, Any] = {}
    for key in allowed_fields:
        if key not in data:
            continue
        value = data.get(key)
        if value is None or value == "":
            continue
        if key in {"max_users", "cell_user_count"}:
            cleaned[key] = max(0, int(value))
        elif key in {
            "coverage_radius",
            "coverage_radius_km",
            "max_throughput",
            "downlink_bandwidth_mbps",
            "uplink_bandwidth_mbps",
            "tx_power_watt",
            "battery_duration_h",
        }:
            cleaned[key] = max(0.0, float(value))
        elif key == "override_fields" and isinstance(value, list):
            cleaned[key] = [str(item) for item in value if item]
        else:
            cleaned[key] = value
    return cleaned


def _scenario_type_overrides(scenario_name: str) -> Dict[str, Dict[str, Any]]:
    entry = _scenario_state_entry(scenario_name)
    raw = entry.get("type_overrides") if entry else {}
    if not isinstance(raw, dict):
        return {}
    return {
        str(base_key): _clean_device_config(config, TYPE_OVERRIDE_FIELDS)
        for base_key, config in raw.items()
        if isinstance(config, dict)
    }


def _base_station_device_uid(scenario_name: str, spec: Dict[str, Any], index: int) -> str:
    if spec.get("device_uid"):
        return str(spec["device_uid"])
    if spec.get("deployment_id"):
        return f"deployment:{spec['deployment_id']}"
    return (
        f"{scenario_name}:{index}:{spec.get('base_station', 'station')}:"
        f"{spec.get('mode', 'mode')}:{spec.get('x', 0)}:{spec.get('y', 0)}"
    )


def _ensure_unique_device_uids(scenario_name: str, specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[str, int] = {}
    for index, spec in enumerate(specs):
        base_uid = _base_station_device_uid(scenario_name, spec, index)
        count = seen.get(base_uid, 0)
        seen[base_uid] = count + 1
        spec["device_uid"] = base_uid if count == 0 else f"{base_uid}:{count + 1}"
    return specs


def _grid_radius_from_km(record: Any, mode: Optional[str], radius_km: Optional[float]) -> Optional[float]:
    if radius_km is None or not mode:
        return None
    mode_profile = record.mode_profiles.get(str(mode), {})
    source_km = float(mode_profile.get("source_coverage_radius_km") or 0.0)
    default_radius = float(mode_profile.get("coverage_radius") or 0.0)
    if source_km <= 0 or default_radius <= 0:
        return None
    return float(radius_km) / source_km * default_radius


def _default_scenario_base_stations(scenario_name: str) -> List[Dict[str, Any]]:
    record = dataset.get(scenario_name)
    measured_specs = _measured_scenario_base_stations(record)
    if measured_specs:
        return measured_specs

    if not record.has_residual_network:
        return []
    locations = list(record.candidate_locations or [])
    profiles = list(record.base_station_profiles.values())
    if not locations or not profiles:
        return []

    specs: List[Dict[str, Any]] = []
    for index, location in enumerate(locations):
        if not isinstance(location, (list, tuple)) or len(location) < 2:
            continue
        profile = profiles[index % len(profiles)]
        mode = profile.supported_modes[0] if profile.supported_modes else None
        if not mode:
            continue
        specs.append(
            {
                "base_station": profile.name,
                "mode": mode,
                "x": int(location[0]),
                "y": int(location[1]),
                "status": "active",
            }
        )
    return specs


def _deployment_sample_paths_for_record(record: Any) -> List[Path]:
    source_scenario = getattr(record, "source_scenario", None)
    severity_level = getattr(record, "severity_level", None)
    if not source_scenario or not severity_level:
        return []

    scenario_dir = dataset_path.parent / str(source_scenario) / str(severity_level)
    if not scenario_dir.exists():
        return []

    paths: List[Path] = []
    for station_dir in sorted(item for item in scenario_dir.glob("*/*") if item.is_dir()):
        path = station_dir / "deployment_samples.jsonl"
        if not path.exists():
            path = station_dir / "cell_info.jsonl"
        if path.exists():
            paths.append(path)
    return paths


def _sample_station_status(sample: Dict[str, Any]) -> str:
    status = str(sample.get("operational_status") or "").strip().lower()
    if status in {"active", "degraded", "offline"}:
        return status
    damage_level = str(sample.get("damage_level") or "").strip().lower()
    if damage_level in {"offline", "destroyed"}:
        return "offline"
    if damage_level and damage_level not in {"intact", "normal"}:
        return "degraded"
    return "active"


def _sample_numeric_metric(sample: Dict[str, Any], key: str) -> Optional[float]:
    value = sample.get(key)
    if isinstance(value, dict):
        for metric_key in ("avg", "mean", "max", "min"):
            metric_value = value.get(metric_key)
            if metric_value is not None:
                try:
                    return float(metric_value)
                except (TypeError, ValueError):
                    continue
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _station_type_label(station_type: object, fallback: object = None) -> str:
    key = str(station_type or "").strip()
    return STATION_TYPE_LABELS.get(key) or str(fallback or key or "场景基站")


def _sample_mode_name(sample: Dict[str, Any], record: Any) -> Optional[str]:
    mode = sample.get("communication_type")
    if mode in getattr(record, "mode_profiles", {}):
        return str(mode)

    directory = str(sample.get("communication_directory") or "").lower()
    for candidate in getattr(record, "communication_modes", []):
        compact = str(candidate).lower().replace("_", "")
        if compact and compact in directory.replace("_", ""):
            return str(candidate)
    return None


def _measured_scenario_base_stations(record: Any) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for path in _deployment_sample_paths_for_record(record):
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle):
                    if not line.strip():
                        continue
                    sample = json.loads(line)
                    mode = _sample_mode_name(sample, record)
                    if not mode:
                        continue
                    base_profile = record.get_base_station_for_mode(mode)
                    if base_profile is None:
                        continue
                    grid = sample.get("grid_position", {})
                    deployment_id = str(
                        sample.get("deployment_id")
                        or f"{path.parent.name}:{path.stem}:{line_number}:{mode}:{grid.get('row', 0)}:{grid.get('col', 0)}"
                    )
                    if deployment_id in seen:
                        continue
                    seen.add(deployment_id)
                    specs.append(
                        {
                            "deployment_id": deployment_id,
                            "base_station": base_profile.name,
                            "mode": mode,
                            "x": int(grid.get("row", 0)),
                            "y": int(grid.get("col", 0)),
                            "status": _sample_station_status(sample),
                            "station_type": sample.get("base_station_type"),
                            "station_label": sample.get("base_station_label"),
                            "cell_user_count": int(sample.get("cell_user_count") or 0),
                            "coverage_radius_km": float(sample.get("coverage_radius_km") or 0.0),
                            "downlink_bandwidth_mbps": _sample_numeric_metric(sample, "downlink_bandwidth_mbps"),
                            "uplink_bandwidth_mbps": _sample_numeric_metric(sample, "uplink_bandwidth_mbps"),
                            "tx_power_watt": _sample_numeric_metric(sample, "tx_power_watt"),
                        }
                    )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    return specs


def _normalize_scenario_base_stations(
    scenario_name: str,
    base_stations: List[Dict[str, Any]],
    *,
    strict: bool = False,
) -> List[Dict[str, Any]]:
    record = dataset.get(scenario_name)
    rows = int(getattr(record.region_grid, "rows", record.grid_size))
    cols = int(getattr(record.region_grid, "cols", record.grid_size))
    normalized: List[Dict[str, Any]] = []

    for index, entry in enumerate(base_stations):
        base_key = str(entry.get("base_station", "")).strip()
        profile = record.base_station_profiles.get(base_key)
        if not profile:
            if strict:
                raise ValueError(f"Unknown base_station at index {index}: {base_key or '<empty>'}")
            continue
        supported_modes = list(profile.supported_modes)
        mode = entry.get("mode")
        if mode not in supported_modes:
            mode = supported_modes[0] if supported_modes else None
        if not mode:
            if strict:
                raise ValueError(f"Base station {base_key} has no supported communication mode")
            continue
        status = str(entry.get("status") or "active").strip().lower()
        if status not in DEVICE_STATUS_VALUES:
            status = "active"
        spec = {
            "base_station": base_key,
            "mode": str(mode),
            "x": int(np.clip(int(entry.get("x", 0)), 0, max(0, rows - 1))),
            "y": int(np.clip(int(entry.get("y", 0)), 0, max(0, cols - 1))),
            "status": status,
        }
        for key in DEVICE_PRESERVED_FIELDS:
            value = entry.get(key)
            if value is not None and value != "":
                spec[key] = value
        if "coverage_radius" not in spec and "coverage_radius_km" in spec:
            radius = _grid_radius_from_km(record, str(mode), float(spec["coverage_radius_km"]))
            if radius is not None:
                spec["coverage_radius"] = radius
        normalized.append(spec)
    return _ensure_unique_device_uids(scenario_name, normalized)


def _apply_scenario_device_type_overrides(scenario_name: str, specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    overrides = _scenario_type_overrides(scenario_name)
    if not overrides:
        return specs
    merged_specs: List[Dict[str, Any]] = []
    for spec in specs:
        merged = dict(spec)
        explicit_fields = set(spec.get("override_fields") or [])
        applied = False
        for override_key in (str(spec.get("base_station") or ""), _device_model_override_key(spec)):
            override = overrides.get(override_key)
            if not override:
                continue
            for key, value in override.items():
                if key not in explicit_fields:
                    merged[key] = value
                    applied = True
        if applied:
            merged["type_override_applied"] = True
        merged_specs.append(merged)
    return merged_specs


def _device_model_override_key(device: Dict[str, Any]) -> str:
    station_type = str(device.get("station_type") or "").strip()
    if station_type:
        return f"station_type:{station_type}"
    station_label = str(device.get("station_label") or "").strip()
    if station_label:
        return f"station_label:{station_label}"
    device_name = str(device.get("device_name") or device.get("label") or "").strip()
    if device_name:
        return f"device_name:{device_name}"
    return f"base_station:{device.get('base_station') or 'unknown'}"


def _enrich_scenario_base_stations(scenario_name: str, specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    record = dataset.get(scenario_name)
    enriched: List[Dict[str, Any]] = []
    for index, spec in enumerate(specs):
        profile = record.base_station_profiles.get(spec.get("base_station"))
        if profile is None:
            continue
        mode = spec.get("mode") or (profile.supported_modes[0] if profile.supported_modes else None)
        mode_profile = record.mode_profiles.get(str(mode), {}) if mode else {}
        max_throughput = float(
            spec.get("max_throughput")
            or spec.get("downlink_bandwidth_mbps")
            or profile.max_throughput
        )
        max_users = int(spec.get("max_users") or spec.get("cell_user_count") or profile.max_users)
        coverage_radius = float(spec.get("coverage_radius") or mode_profile.get("coverage_radius") or 0.0)
        source_coverage_radius_km = float(
            spec.get("coverage_radius_km")
            or mode_profile.get("source_coverage_radius_km")
            or 0.0
        )
        enriched.append(
            {
                **spec,
                "id": spec.get("device_uid")
                or spec.get("deployment_id")
                or f"{scenario_name}:{index}:{spec.get('base_station')}:{spec.get('x')}:{spec.get('y')}",
                "label": spec.get("device_name")
                or _station_type_label(spec.get("station_type"), spec.get("station_label") or profile.label),
                "mode_label": mode_profile.get("label", mode),
                "coverage_radius": coverage_radius,
                "source_coverage_radius_km": source_coverage_radius_km,
                "coverage_radius_km": source_coverage_radius_km,
                "max_throughput": max_throughput,
                "downlink_bandwidth_mbps": float(spec.get("downlink_bandwidth_mbps") or max_throughput),
                "uplink_bandwidth_mbps": float(spec.get("uplink_bandwidth_mbps") or max(0.0, max_throughput * 0.3)),
                "max_users": max_users,
                "device_cost": profile.device_cost,
                "bandwidth_cost": profile.bandwidth_cost,
                "status": spec.get("status") or "active",
                "device_category": spec.get("device_category") or profile.label,
            }
        )
    return enriched


def _scenario_base_station_specs_raw(scenario_name: str) -> List[Dict[str, Any]]:
    entry = _scenario_state_entry(scenario_name)
    if entry is not None and isinstance(entry.get("base_stations"), list):
        return _normalize_scenario_base_stations(scenario_name, entry["base_stations"])
    return _normalize_scenario_base_stations(scenario_name, _default_scenario_base_stations(scenario_name))


def _scenario_base_station_specs(scenario_name: str) -> List[Dict[str, Any]]:
    return _apply_scenario_device_type_overrides(scenario_name, _scenario_base_station_specs_raw(scenario_name))


def _scenario_base_station_response(scenario_name: str) -> ScenarioBaseStationResponse:
    entry = _scenario_state_entry(scenario_name)
    specs = _scenario_base_station_specs(scenario_name)
    return ScenarioBaseStationResponse(
        scenario_name=scenario_name,
        base_stations=_enrich_scenario_base_stations(scenario_name, specs),
        updated_at=float(entry.get("updated_at")) if entry and entry.get("updated_at") is not None else None,
    )


def _scenario_device_type_rows(scenario_name: str, devices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    record = dataset.get(scenario_name)
    overrides = _scenario_type_overrides(scenario_name)
    counts: Dict[str, Dict[str, int]] = {}
    for device in devices:
        base_key = str(device.get("base_station") or "")
        if not base_key:
            continue
        status = str(device.get("status") or "unknown")
        counts.setdefault(base_key, {"total": 0, "active": 0, "degraded": 0, "offline": 0})
        counts[base_key]["total"] += 1
        if status in counts[base_key]:
            counts[base_key][status] += 1

    rows: List[Dict[str, Any]] = []
    for profile in record.base_station_profiles.values():
        override = overrides.get(profile.name, {})
        supported_modes = list(profile.supported_modes)
        default_mode = supported_modes[0] if supported_modes else None
        mode_profile = record.mode_profiles.get(str(default_mode), {}) if default_mode else {}
        max_throughput = float(
            override.get("max_throughput")
            or override.get("downlink_bandwidth_mbps")
            or profile.max_throughput
        )
        coverage_radius = float(override.get("coverage_radius") or mode_profile.get("coverage_radius") or 0.0)
        coverage_radius_km = float(
            override.get("coverage_radius_km")
            or mode_profile.get("source_coverage_radius_km")
            or 0.0
        )
        rows.append(
            {
                "base_station": profile.name,
                "label": profile.label,
                "supported_modes": supported_modes,
                "mode": override.get("mode") or default_mode,
                "device_category": override.get("device_category") or profile.label,
                "coverage_radius": coverage_radius,
                "coverage_radius_km": coverage_radius_km,
                "max_throughput": max_throughput,
                "downlink_bandwidth_mbps": float(override.get("downlink_bandwidth_mbps") or max_throughput),
                "uplink_bandwidth_mbps": float(override.get("uplink_bandwidth_mbps") or max(0.0, max_throughput * 0.3)),
                "max_users": int(override.get("max_users") or profile.max_users),
                "tx_power_watt": override.get("tx_power_watt"),
                "battery_duration_h": override.get("battery_duration_h"),
                "notes": override.get("notes"),
                "counts": counts.get(profile.name, {"total": 0, "active": 0, "degraded": 0, "offline": 0}),
                "has_override": bool(override),
            }
        )
    return rows


def _scenario_device_model_rows(scenario_name: str, devices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    record = dataset.get(scenario_name)
    overrides = _scenario_type_overrides(scenario_name)
    grouped: Dict[str, Dict[str, Any]] = {}

    for device in devices:
        model_key = _device_model_override_key(device)
        profile = record.base_station_profiles.get(device.get("base_station"))
        group = grouped.setdefault(
            model_key,
            {
                "model_key": model_key,
                "station_type": device.get("station_type"),
                "station_label": device.get("station_label"),
                "base_station": device.get("base_station"),
                "base_station_label": profile.label if profile else device.get("base_station"),
                "supported_modes": list(profile.supported_modes) if profile else [],
                "devices": [],
                "counts": {"total": 0, "active": 0, "degraded": 0, "offline": 0},
            },
        )
        group["devices"].append(device)
        status = str(device.get("status") or "unknown")
        group["counts"]["total"] += 1
        if status in group["counts"]:
            group["counts"][status] += 1

    rows: List[Dict[str, Any]] = []
    numeric_fields = (
        "coverage_radius",
        "coverage_radius_km",
        "max_throughput",
        "downlink_bandwidth_mbps",
        "uplink_bandwidth_mbps",
        "max_users",
        "tx_power_watt",
        "battery_duration_h",
    )

    for model_key, group in grouped.items():
        model_devices = group.pop("devices")
        sample = model_devices[0] if model_devices else {}
        override = overrides.get(model_key, {})

        def average_number(field: str, fallback: Any = None) -> Optional[float]:
            values: List[float] = []
            for item in model_devices:
                value = item.get(field)
                if value is None and field == "max_throughput":
                    value = item.get("downlink_bandwidth_mbps")
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(number):
                    values.append(number)
            if values:
                return float(sum(values) / len(values))
            try:
                return float(fallback) if fallback is not None else None
            except (TypeError, ValueError):
                return None

        row: Dict[str, Any] = {
            **group,
            "label": _station_type_label(
                group.get("station_type"),
                sample.get("device_name") or sample.get("label") or group.get("station_label") or group.get("base_station_label"),
            ),
            "mode": override.get("mode") or sample.get("mode"),
            "device_category": override.get("device_category") or sample.get("device_category") or sample.get("label"),
            "notes": override.get("notes"),
            "has_override": bool(override),
            "sample_device_id": sample.get("id") or sample.get("device_uid"),
        }
        for field in numeric_fields:
            value = override.get(field)
            if value is None and field == "max_throughput":
                value = override.get("downlink_bandwidth_mbps")
            if value is None:
                value = average_number(field)
            if field == "max_users" and value is not None:
                row[field] = int(round(float(value)))
            elif value is not None:
                row[field] = float(value)
        rows.append(row)

    return sorted(rows, key=lambda item: (str(item.get("base_station") or ""), str(item.get("label") or "")))


def _scenario_device_blocks(devices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    for device in devices:
        key = (
            int(device.get("x", 0)),
            int(device.get("y", 0)),
            str(device.get("base_station") or ""),
            str(device.get("mode") or ""),
            str(device.get("status") or "unknown"),
        )
        block = grouped.setdefault(
            key,
            {
                "x": key[0],
                "y": key[1],
                "base_station": key[2],
                "mode": key[3],
                "status": key[4],
                "quantity": 0,
                "device_ids": [],
                "label": device.get("label") or device.get("base_station"),
                "mode_label": device.get("mode_label") or device.get("mode"),
            },
        )
        block["quantity"] += 1
        block["device_ids"].append(device.get("id") or device.get("device_uid"))
    return sorted(grouped.values(), key=lambda item: (item["x"], item["y"], item["base_station"], item["status"]))


def _scenario_device_status_counts(devices: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"total": len(devices), "active": 0, "degraded": 0, "offline": 0, "planned": 0, "unknown": 0}
    for device in devices:
        status = str(device.get("status") or "unknown")
        counts[status if status in counts else "unknown"] += 1
    return counts


def _scenario_state_history(scenario_name: str) -> List[Dict[str, Any]]:
    entry = _scenario_state_entry(scenario_name)
    history = entry.get("history") if entry else []
    return history if isinstance(history, list) else []


def _scenario_device_state_payload(scenario_name: str) -> ScenarioDeviceStateResponse:
    record = dataset.get(scenario_name)
    entry = _scenario_state_entry(scenario_name)
    specs = _scenario_base_station_specs(scenario_name)
    devices = _enrich_scenario_base_stations(scenario_name, specs)
    return ScenarioDeviceStateResponse(
        scenario_name=scenario_name,
        display_name=record.display_name or record.name,
        grid=_scenario_grid_public(record),
        device_types=_scenario_device_type_rows(scenario_name, devices),
        device_models=_scenario_device_model_rows(scenario_name, devices),
        devices=devices,
        blocks=_scenario_device_blocks(devices),
        status_counts=_scenario_device_status_counts(devices),
        type_overrides=_scenario_type_overrides(scenario_name),
        history=_scenario_state_history(scenario_name)[-80:],
        updated_at=float(entry.get("updated_at")) if entry and entry.get("updated_at") is not None else None,
    )


def _persist_scenario_device_state(
    scenario_name: str,
    base_stations: List[Dict[str, Any]],
    *,
    type_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    operation: str = "update",
) -> ScenarioDeviceStateResponse:
    specs = _normalize_scenario_base_stations(scenario_name, base_stations, strict=True)
    merged_type_overrides = (
        {
            str(base_key): _clean_device_config(config, TYPE_OVERRIDE_FIELDS)
            for base_key, config in type_overrides.items()
            if isinstance(config, dict)
        }
        if type_overrides is not None
        else None
    )
    scenario_device_manager.replace_state(
        scenario_name,
        specs,
        type_overrides=merged_type_overrides,
        operation=operation,
    )
    _invalidate_scenarios_cache()
    return _scenario_device_state_payload(scenario_name)


def _request_base_stations_or_scenario_state(
    scenario_name: str,
    requested_base_stations: Optional[List[Dict[str, Any]]],
    *,
    use_scenario_state: bool = True,
) -> Optional[List[Dict[str, Any]]]:
    if requested_base_stations is not None:
        return _normalize_scenario_base_stations(scenario_name, requested_base_stations)
    if not use_scenario_state:
        return None
    specs = _scenario_base_station_specs(scenario_name)
    return specs if specs else None


def _strategy_station_key(station: Dict[str, Any]) -> str:
    deployment_id = station.get("deployment_id")
    if deployment_id:
        return f"deployment:{deployment_id}"
    device_uid = station.get("device_uid")
    if device_uid:
        return f"device:{device_uid}"
    return "{}:{}:{}:{}".format(
        station.get("base_station") or station.get("station_type") or "station",
        station.get("mode") or "",
        station.get("x") if station.get("x") is not None else station.get("row"),
        station.get("y") if station.get("y") is not None else station.get("col"),
    )


def _strategy_status_for_db(status: Any) -> str:
    normalized = str(status or "unknown").strip().lower()
    if normalized in {"active", "deployed", "restored", "online"}:
        return "active"
    if normalized in {"degraded", "partial", "limited"}:
        return "degraded"
    if normalized in {"offline", "inactive", "down"}:
        return "offline"
    if normalized in {"planned"}:
        return "planned"
    return "unknown"


def _strategy_recovery_events_by_key(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    events_by_key: Dict[str, Dict[str, Any]] = {}
    for event in summary.get("events") or []:
        if not isinstance(event, dict):
            continue
        keys = [event.get("station_key")]
        if event.get("deployment_id"):
            keys.append(f"deployment:{event.get('deployment_id')}")
        if event.get("device_uid"):
            keys.append(f"device:{event.get('device_uid')}")
        for key in keys:
            if key:
                events_by_key[str(key)] = event
    return events_by_key


def _station_status_counts_for_specs(specs: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"total": len(specs), "active": 0, "degraded": 0, "offline": 0, "planned": 0, "unknown": 0}
    for spec in specs:
        status = _strategy_status_for_db(spec.get("status"))
        counts[status if status in counts else "unknown"] += 1
    return counts


def _sync_scenario_devices_from_strategy_result(
    scenario_name: str,
    base_stations: Optional[List[Dict[str, Any]]],
    reports: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Persist station recovery statuses back to the scenario device database."""
    report = reports[0] if reports else {}
    summary = (report or {}).get("station_recovery_summary") or {}
    events_by_key = _strategy_recovery_events_by_key(summary)
    if not events_by_key:
        return {"updated": False, "changed_count": 0, "status_counts": {}}

    source_specs = base_stations if base_stations is not None else _scenario_base_station_specs_raw(scenario_name)
    specs = _normalize_scenario_base_stations(scenario_name, [dict(spec) for spec in (source_specs or [])])
    changed: List[Dict[str, Any]] = []

    for spec in specs:
        event = events_by_key.get(_strategy_station_key(spec))
        if not event and spec.get("deployment_id"):
            event = events_by_key.get(f"deployment:{spec.get('deployment_id')}")
        if not event and spec.get("device_uid"):
            event = events_by_key.get(f"device:{spec.get('device_uid')}")
        if not event:
            continue
        next_status = _strategy_status_for_db(event.get("to_status"))
        if next_status not in DEVICE_STATUS_VALUES:
            continue
        previous_status = _strategy_status_for_db(spec.get("status"))
        spec["status"] = next_status
        if previous_status != next_status:
            changed.append(
                {
                    "device_uid": spec.get("device_uid"),
                    "deployment_id": spec.get("deployment_id"),
                    "from_status": previous_status,
                    "to_status": next_status,
                    "grid": {"row": spec.get("x"), "col": spec.get("y")},
                }
            )

    if not changed:
        return {"updated": False, "changed_count": 0, "status_counts": _station_status_counts_for_specs(specs)}

    scenario_device_manager.replace_state(
        scenario_name,
        specs,
        operation="strategy_result_sync",
    )
    _invalidate_scenarios_cache()
    return {
        "updated": True,
        "changed_count": len(changed),
        "changes": changed,
        "status_counts": _station_status_counts_for_specs(specs),
    }


def _is_level4_scenario_name(scenario_name: object) -> bool:
    text = str(scenario_name or "").lower()
    return "level_4" in text or "level-4" in text or "特别严重" in text


def _scenario_disaster_type(scenario_name: object) -> str | None:
    if not scenario_name:
        return None
    try:
        return dataset.get(str(scenario_name)).disaster_type
    except Exception:  # pylint: disable=broad-except
        return None


def _artifact_protocol(meta: Dict[str, object], config: Dict[str, object]) -> str:
    evaluation_cfg = config.get("evaluation", {}) if isinstance(config, dict) else {}
    if not isinstance(evaluation_cfg, dict):
        evaluation_cfg = {}
    return str(meta.get("evaluation_protocol") or evaluation_cfg.get("protocol") or "standard")


def _artifact_created_at(meta: Dict[str, object], run_dir: Path) -> float:
    match = TRAINING_RUN_DIR_RE.match(run_dir.name)
    if match:
        try:
            return datetime.strptime(match.group("date") + match.group("time"), "%Y%m%d%H%M%S").timestamp()
        except ValueError:
            pass

    value = _finite_float(meta.get("created_at") if isinstance(meta, dict) else None)
    if value is not None and value > 0:
        return value

    try:
        return run_dir.stat().st_ctime
    except OSError:
        return 0.0


def _artifact_run_dir_from_policy_path(meta: Dict[str, object], fallback_dir: Path) -> Path:
    policy_path = meta.get("policy_path") if isinstance(meta, dict) else None
    if isinstance(policy_path, str) and policy_path:
        candidate = Path(policy_path).expanduser().parent
        if TRAINING_RUN_DIR_RE.match(candidate.name):
            return candidate
    return fallback_dir


def _artifact_modified_at(meta_path: Path, metrics_path: Path) -> float:
    return max(
        meta_path.stat().st_mtime if meta_path.exists() else 0.0,
        metrics_path.stat().st_mtime if metrics_path.exists() else 0.0,
    )


def _timestamp_from_dir_name(path: Path, pattern: re.Pattern[str]) -> Optional[float]:
    match = pattern.match(path.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group("date") + match.group("time"), "%Y%m%d%H%M%S").timestamp()
    except ValueError:
        return None


def _scene_export_created_at(meta: Dict[str, object], export_dir: Path, metadata_path: Path) -> float:
    value = _finite_float(meta.get("created_at") if isinstance(meta, dict) else None)
    if value is not None and value > 0:
        return value

    parsed = _timestamp_from_dir_name(export_dir, SCENE_EXPORT_DIR_RE)
    if parsed is not None:
        return parsed

    try:
        return metadata_path.stat().st_mtime
    except OSError:
        return 0.0


def _scenario_display_label(value: object) -> str:
    text = str(value or "").strip()
    lower = text.lower()
    if "typhoon" in lower or "台风" in text:
        return "台风灾后残余网络"
    if "earthquake" in lower or "地震" in text:
        return "地震灾后断链恢复"
    if (
        "rainstorm" in lower
        or "flood" in lower
        or "water_disaster" in lower
        or "water disaster" in lower
        or "暴雨" in text
        or "洪水" in text
        or "水灾" in text
    ):
        return "洪水孤岛通信恢复"
    return text or "未选择场景"


def _scene_export_record(metadata_path: Path) -> Optional[Dict[str, object]]:
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict):
        return None

    export_dir = metadata_path.parent
    created_at = _scene_export_created_at(meta, export_dir, metadata_path)
    return {
        "id": export_dir.name,
        "source": meta.get("source") or "test",
        "created_at": created_at,
        "created_at_iso": meta.get("created_at_iso")
        or (datetime.fromtimestamp(created_at).isoformat(timespec="seconds") if created_at > 0 else ""),
        "modified_at": metadata_path.stat().st_mtime if metadata_path.exists() else 0.0,
        "scenario_name": meta.get("scenario_name"),
        "scenario_label": meta.get("scenario_label") or _scenario_display_label(meta.get("scenario_name")),
        "episode": meta.get("episode"),
        "export_dir": str(export_dir),
        "metadata_path": str(metadata_path),
        "disaster_scene_path": meta.get("disaster_scene_path"),
        "deployment_scene_path": meta.get("deployment_scene_path"),
        "deployment_plan_path": meta.get("deployment_plan_path"),
    }


def _default_protocol_for_scenario(scenario_name: object) -> str:
    return "earthquake_stress" if _scenario_disaster_type(scenario_name) == "earthquake" else "standard"


def _classify_candidate_site(region_grid, row: int, col: int) -> str:
    label = region_grid.cell_label(row, col)
    if label and not label.startswith("cell-"):
        return "重点保障"

    row_norm = row / max(1, region_grid.rows - 1)
    col_norm = col / max(1, region_grid.cols - 1)
    is_edge = row_norm <= 0.16 or row_norm >= 0.84 or col_norm <= 0.16 or col_norm >= 0.84
    is_center = 0.35 <= row_norm <= 0.65 and 0.35 <= col_norm <= 0.65
    is_corridor = abs(row_norm - col_norm) <= 0.12 or abs((row_norm + col_norm) - 1.0) <= 0.12

    if is_center:
        return "核心覆盖"
    if is_corridor:
        return "中继转发"
    if is_edge:
        return "边缘补盲"
    return "机动接入"


def _build_candidate_site_preview(scenario_name: str) -> List[Dict[str, object]]:
    config = get_default_config()
    config["experiment"]["env_type"] = "multimodal"
    config["multimodal_env"]["scenario_name"] = scenario_name
    config["multimodal_env"]["seed"] = 42
    env = build_env(config, "multimodal")
    try:
        preview = []
        for site_index, coords in enumerate(env.candidate_locations):
            row = int(coords[0])
            col = int(coords[1])
            lat_min, lat_max, lon_min, lon_max = env.region_grid.cell_bounds(row, col)
            region_label = env.region_grid.cell_label(row, col)
            preview.append(
                {
                    "site_index": site_index,
                    "x": row,
                    "y": col,
                    "region_label": region_label,
                    "category": _classify_candidate_site(env.region_grid, row, col),
                    "lat_lon_bounds": {
                        "lat_min": lat_min,
                        "lat_max": lat_max,
                        "lon_min": lon_min,
                        "lon_max": lon_max,
                    },
                }
            )
        return preview
    finally:
        env.close()


def _cached_candidate_site_preview(scenario_name: str) -> List[Dict[str, object]]:
    with scenarios_cache_lock:
        cached = candidate_site_preview_cache.get(scenario_name)
        if cached is not None:
            return cached
        preview = _build_candidate_site_preview(scenario_name)
        candidate_site_preview_cache[scenario_name] = preview
        return preview


def _invalidate_scenarios_cache() -> None:
    global scenarios_response_cache
    with scenarios_cache_lock:
        scenarios_response_cache = None


@app.get("/api/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/train/latest-artifact")
def latest_training_artifact() -> Dict[str, object]:
    artifact_dir = Path(default_config["logging"]["artifact_dir"])
    meta_path = artifact_dir / "policy_meta.json"
    metrics_path = artifact_dir / "training_metrics.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="No training artifact metadata found.")

    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    metrics = {}
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

    config = metrics.get("config", {}) if isinstance(metrics, dict) else {}
    experiment_cfg = config.get("experiment", {}) if isinstance(config, dict) else {}
    multimodal_cfg = config.get("multimodal_env", {}) if isinstance(config, dict) else {}
    run_dir = _artifact_run_dir_from_policy_path(meta, artifact_dir)
    created_at = _artifact_created_at(meta, run_dir)
    modified_at = _artifact_modified_at(meta_path, metrics_path)
    scenario_name = multimodal_cfg.get("scenario_name")
    evaluation_protocol = _artifact_protocol(meta, config)
    return {
        "algorithm": meta.get("algorithm") or experiment_cfg.get("algorithm") or "ppo",
        "env_type": meta.get("env_type") or experiment_cfg.get("env_type") or "multimodal",
        "checkpoint_path": meta.get("policy_path"),
        "scenario_name": scenario_name,
        "disaster_type": _scenario_disaster_type(scenario_name),
        "reward_mode": multimodal_cfg.get("reward_mode"),
        "evaluation_protocol": evaluation_protocol,
        "created_at": created_at,
        "updated_at": created_at,
        "modified_at": modified_at,
        "status": "completed",
    }


@app.get("/api/train/artifacts")
def list_training_artifacts() -> Dict[str, List[Dict[str, object]]]:
    artifact_dir = Path(default_config["logging"]["artifact_dir"])
    run_roots = [
        artifact_dir / "runs",
        artifact_dir / "real_level4_benchmark" / "runs",
    ]
    artifacts: List[Dict[str, object]] = []

    for runs_dir in run_roots:
        if not runs_dir.exists():
            continue
        for meta_path in runs_dir.glob("*/policy_meta.json"):
            metrics_path = meta_path.with_name("training_metrics.json")
            try:
                with meta_path.open("r", encoding="utf-8") as handle:
                    meta = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue

            metrics = {}
            if metrics_path.exists():
                try:
                    with metrics_path.open("r", encoding="utf-8") as handle:
                        metrics = json.load(handle)
                except (OSError, json.JSONDecodeError):
                    metrics = {}

            config = metrics.get("config", {}) if isinstance(metrics, dict) else {}
            experiment_cfg = config.get("experiment", {}) if isinstance(config, dict) else {}
            multimodal_cfg = config.get("multimodal_env", {}) if isinstance(config, dict) else {}
            policy_path = meta.get("policy_path")
            if not policy_path:
                continue

            created_at = _artifact_created_at(meta, meta_path.parent)
            modified_at = _artifact_modified_at(meta_path, metrics_path)
            scenario_name = multimodal_cfg.get("scenario_name")
            evaluation_protocol = _artifact_protocol(meta, config)
            artifacts.append(
                {
                    "algorithm": meta.get("algorithm") or experiment_cfg.get("algorithm") or "ppo",
                    "env_type": meta.get("env_type") or experiment_cfg.get("env_type") or "multimodal",
                    "checkpoint_path": policy_path,
                    "scenario_name": scenario_name,
                    "disaster_type": _scenario_disaster_type(scenario_name),
                    "reward_mode": multimodal_cfg.get("reward_mode"),
                    "evaluation_protocol": evaluation_protocol,
                    "updated_at": created_at,
                    "created_at": created_at,
                    "modified_at": modified_at,
                    "status": "completed",
                    "operator": "系统",
                    "run_dir": str(meta_path.parent),
                }
            )

    artifacts.sort(key=lambda item: float(item.get("created_at") or item.get("updated_at") or 0), reverse=True)
    return {"artifacts": artifacts}


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _finite_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _sample_metric_indices(length: int, max_points: int = 20) -> List[int]:
    if length <= 0:
        return []
    if length <= max_points:
        return list(range(length))
    if max_points <= 1:
        return [length - 1]
    return sorted({round(index * (length - 1) / (max_points - 1)) for index in range(max_points)})


def _normalize_eval_history(eval_history: Any) -> List[Dict[str, float]]:
    if not isinstance(eval_history, list):
        return []

    rows: List[Dict[str, float]] = []
    for item in eval_history:
        if not isinstance(item, dict):
            continue
        step = _finite_float(item.get("step", item.get("global_step")))
        if step is None:
            continue
        row: Dict[str, float] = {"step": step}
        reward = _finite_float(item.get("avg_reward", item.get("reward")))
        coverage = _finite_float(item.get("avg_coverage", item.get("coverage")))
        broadcast = _finite_float(item.get("avg_broadcast", item.get("broadcast")))
        if reward is not None:
            row["avg_reward"] = reward
        if coverage is not None:
            row["avg_coverage"] = coverage
        if broadcast is not None:
            row["avg_broadcast"] = broadcast
        if len(row) > 1:
            rows.append(row)
    return rows


def _build_episode_curve_history(
    episode_rewards: Any,
    episode_coverages: Any,
    episode_broadcasts: Any,
    episode_timesteps: Any,
) -> List[Dict[str, float]]:
    rewards = episode_rewards if isinstance(episode_rewards, list) else []
    coverages = episode_coverages if isinstance(episode_coverages, list) else []
    broadcasts = episode_broadcasts if isinstance(episode_broadcasts, list) else []
    timesteps = episode_timesteps if isinstance(episode_timesteps, list) else []
    total_rows = max(len(rewards), len(coverages), len(broadcasts), len(timesteps))

    rows: List[Dict[str, float]] = []
    for index in _sample_metric_indices(total_rows):
        step = _finite_float(timesteps[index] if index < len(timesteps) else index + 1)
        if step is None:
            step = float(index + 1)
        row: Dict[str, float] = {"step": step}
        reward = _finite_float(rewards[index] if index < len(rewards) else None)
        coverage = _finite_float(coverages[index] if index < len(coverages) else None)
        broadcast = _finite_float(broadcasts[index] if index < len(broadcasts) else None)
        if reward is not None:
            row["avg_reward"] = reward
        if coverage is not None:
            row["avg_coverage"] = coverage
        if broadcast is not None:
            row["avg_broadcast"] = broadcast
        if len(row) > 1:
            rows.append(row)
    return rows


def _training_curve_history(
    eval_history: Any,
    episode_rewards: Any,
    episode_coverages: Any,
    episode_broadcasts: Any,
    episode_timesteps: Any,
) -> tuple[List[Dict[str, float]], str]:
    normalized_eval_history = _normalize_eval_history(eval_history)
    if any("avg_coverage" in item or "avg_broadcast" in item for item in normalized_eval_history):
        return normalized_eval_history, "eval_history"

    episode_curve_history = _build_episode_curve_history(
        episode_rewards,
        episode_coverages,
        episode_broadcasts,
        episode_timesteps,
    )
    if episode_curve_history:
        return episode_curve_history, "episode_history"
    return normalized_eval_history, "eval_history"


@app.get("/api/train/artifacts/detail")
def training_artifact_detail(run_dir: str) -> Dict[str, object]:
    artifact_dir = Path(default_config["logging"]["artifact_dir"]).resolve()
    allowed_run_roots = [
        (artifact_dir / "runs").resolve(),
        (artifact_dir / "real_level4_benchmark" / "runs").resolve(),
    ]
    requested_dir = Path(run_dir).resolve()

    if not any(_is_relative_to(requested_dir, root) for root in allowed_run_roots):
        raise HTTPException(status_code=400, detail="Invalid training run directory.")

    meta_path = requested_dir / "policy_meta.json"
    metrics_path = requested_dir / "training_metrics.json"
    test_results_path = requested_dir / "test_results.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Training artifact metadata not found.")

    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail="Failed to read policy metadata.") from error

    metrics = {}
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
        except (OSError, json.JSONDecodeError):
            metrics = {}

    test_results: Dict[str, object] = {}
    if test_results_path.exists():
        try:
            with test_results_path.open("r", encoding="utf-8") as handle:
                raw_test_results = json.load(handle)
            test_results = raw_test_results if isinstance(raw_test_results, dict) else {}
        except (OSError, json.JSONDecodeError):
            test_results = {}

    episode_rewards = metrics.get("episode_rewards") if isinstance(metrics, dict) else []
    episode_coverages = metrics.get("episode_coverages") if isinstance(metrics, dict) else []
    episode_broadcasts = metrics.get("episode_broadcasts") if isinstance(metrics, dict) else []
    episode_timesteps = metrics.get("episode_timesteps") if isinstance(metrics, dict) else []
    eval_history = metrics.get("eval_history") if isinstance(metrics, dict) else []
    config = metrics.get("config", {}) if isinstance(metrics, dict) else {}
    experiment_cfg = config.get("experiment", {}) if isinstance(config, dict) else {}
    multimodal_cfg = config.get("multimodal_env", {}) if isinstance(config, dict) else {}
    algorithm_key = meta.get("algorithm") or experiment_cfg.get("algorithm") or "ppo"
    algorithm_cfg = config.get(str(algorithm_key).lower(), {}) if isinstance(config, dict) else {}
    train_cfg = config.get("train", {}) if isinstance(config, dict) else {}
    curve_history, curve_history_source = _training_curve_history(
        eval_history,
        episode_rewards,
        episode_coverages,
        episode_broadcasts,
        episode_timesteps,
    )
    created_at = _artifact_created_at(meta, requested_dir)
    modified_at = _artifact_modified_at(meta_path, metrics_path)

    scenario_name = multimodal_cfg.get("scenario_name")
    evaluation_protocol = _artifact_protocol(meta, config)
    return {
        "algorithm": algorithm_key,
        "env_type": meta.get("env_type") or experiment_cfg.get("env_type") or "multimodal",
        "checkpoint_path": meta.get("policy_path"),
        "scenario_name": scenario_name,
        "disaster_type": _scenario_disaster_type(scenario_name),
        "reward_mode": multimodal_cfg.get("reward_mode"),
        "evaluation_protocol": evaluation_protocol,
        "updated_at": created_at,
        "created_at": created_at,
        "modified_at": modified_at,
        "status": "completed",
        "operator": "系统",
        "run_dir": str(requested_dir),
        "episode_count": len(episode_rewards) if isinstance(episode_rewards, list) else 0,
        "total_timesteps": episode_timesteps[-1] if isinstance(episode_timesteps, list) and episode_timesteps else train_cfg.get("total_timesteps"),
        "last_reward": episode_rewards[-1] if isinstance(episode_rewards, list) and episode_rewards else None,
        "best_reward": max(episode_rewards) if isinstance(episode_rewards, list) and episode_rewards else None,
        "last_coverage": episode_coverages[-1] if isinstance(episode_coverages, list) and episode_coverages else None,
        "best_coverage": max(episode_coverages) if isinstance(episode_coverages, list) and episode_coverages else None,
        "last_broadcast": episode_broadcasts[-1] if isinstance(episode_broadcasts, list) and episode_broadcasts else None,
        "best_broadcast": max(episode_broadcasts) if isinstance(episode_broadcasts, list) and episode_broadcasts else None,
        "eval_history": curve_history,
        "raw_eval_history": eval_history if isinstance(eval_history, list) else [],
        "curve_history": curve_history,
        "curve_history_source": curve_history_source,
        "test_results": test_results,
        "config": {
            "experiment": experiment_cfg if isinstance(experiment_cfg, dict) else {},
            "train": train_cfg if isinstance(train_cfg, dict) else {},
            "multimodal_env": multimodal_cfg if isinstance(multimodal_cfg, dict) else {},
            "algorithm": algorithm_cfg if isinstance(algorithm_cfg, dict) else {},
            "evaluation": config.get("evaluation", {}) if isinstance(config, dict) else {},
        },
    }


def _build_scenarios_payload() -> Dict[str, List[Dict[str, object]]]:
    scenarios = []
    for name in dataset.list_scenarios():
        record = dataset.get(name)
        candidate_site_preview = _cached_candidate_site_preview(name)
        region_grid = record.region_grid.to_public_dict() if getattr(record, "region_grid", None) else None
        reward_profiles = [
            {
                "key": key,
                "label": profile.label,
                "description": profile.description,
                "coverage_weight": profile.coverage_weight,
                "bandwidth_weight": profile.bandwidth_weight,
                "throughput_weight": profile.throughput_weight,
                "broadcast_weight": profile.broadcast_weight,
                "device_cost_weight": profile.device_cost_weight,
                "bandwidth_cost_weight": profile.bandwidth_cost_weight,
            }
            for key, profile in sorted(record.reward_profiles.items())
        ]
        base_stations = [
            {
                "name": profile.name,
                "label": profile.label,
                "max_throughput": profile.max_throughput,
                "max_users": profile.max_users,
                "device_cost": profile.device_cost,
                "bandwidth_cost": profile.bandwidth_cost,
                "supported_modes": profile.supported_modes,
            }
            for profile in record.base_station_profiles.values()
        ]
        communication_modes = [
            {
                "key": key,
                "name": key,
                "label": profile.get("label", key),
                "coverage_radius": profile.get("coverage_radius"),
                "source_coverage_radius_km": profile.get("source_coverage_radius_km"),
                "base_station": profile.get("base_station"),
            }
            for key, profile in sorted(record.mode_profiles.items())
        ]
        base_station_deployments = _enrich_scenario_base_stations(name, _scenario_base_station_specs(name))
        try:
            severity_meta = disaster_import_manager.get_severity_meta(record.severity_level or "")
        except KeyError:
            severity_meta = {}
        scenarios.append(
            {
                "name": record.name,
                "display_name": record.display_name,
                "source_scenario": record.source_scenario,
                "severity_level": record.severity_level,
                "severity_label": record.severity_label,
                "severity_description": severity_meta.get("description"),
                "damage_rate": severity_meta.get("damage_rate"),
                "offline_rate": severity_meta.get("offline_rate"),
                "severity_meta": severity_meta,
                "disaster_type": record.disaster_type,
                "grid_size": record.grid_size,
                "region_grid": region_grid,
                "num_users": record.num_users,
                "candidate_sites": record.candidate_sites,
                "max_steps": record.max_steps,
                "has_residual_network": record.has_residual_network,
                "reward_profiles": reward_profiles,
                "default_reward_profile": record.default_reward_profile,
                "base_stations": base_stations,
                "base_station_deployments": base_station_deployments,
                "residual_base_stations": base_station_deployments,
                "communication_modes": communication_modes,
                "user_clusters": record.user_clusters,
                "candidate_site_preview": candidate_site_preview,
            }
        )
    return {"scenarios": scenarios}


def _cached_scenarios_response_body() -> bytes:
    global scenarios_response_cache
    with scenarios_cache_lock:
        if scenarios_response_cache is None:
            scenarios_response_cache = json.dumps(
                _build_scenarios_payload(),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        return scenarios_response_cache


@app.get("/api/scenarios")
def list_scenarios() -> Response:
    return Response(
        content=_cached_scenarios_response_body(),
        media_type="application/json",
    )


@app.get("/api/devices", response_model=DedicatedDeviceListResponse)
def list_dedicated_devices(
    status: Optional[DeviceStatusKey] = Query(None, description="Filter by active/inactive status."),
    device_type: Optional[DeviceTypeKey] = Query(None, description="Filter by communication type."),
) -> DedicatedDeviceListResponse:
    return device_manager.list(status=status, device_type=device_type)


@app.get("/api/device-schema")
def get_device_schema() -> Dict[str, Any]:
    return {
        "database": str(scenario_device_manager.db_path),
        "tables": scenario_device_manager.schema_overview(),
        "relationships": [
            "scenario_registry 1 -> N scenario_grid_cells",
            "scenario_registry 1 -> N scenario_devices",
            "scenario_grid_cells 1 -> N scenario_devices via (scenario_name,row_index,col_index)",
            "scenario_registry 1 -> N scenario_device_type_configs",
            "scenario_registry 1 -> N scenario_device_events",
        ],
    }


@app.post("/api/devices", response_model=DedicatedDevice, status_code=201)
def create_dedicated_device(request: DedicatedDeviceCreate) -> DedicatedDevice:
    return device_manager.create(request)


@app.get("/api/devices/{device_id}", response_model=DedicatedDevice)
def get_dedicated_device(device_id: str) -> DedicatedDevice:
    device = device_manager.get(device_id)
    if device is None:
        raise HTTPException(status_code=404, detail=f"Device not found: {device_id}")
    return device


@app.put("/api/devices/{device_id}", response_model=DedicatedDevice)
def update_dedicated_device(device_id: str, request: DedicatedDeviceUpdate) -> DedicatedDevice:
    device = device_manager.update(device_id, request)
    if device is None:
        raise HTTPException(status_code=404, detail=f"Device not found: {device_id}")
    return device


@app.patch("/api/devices/{device_id}/status", response_model=DedicatedDevice)
def update_dedicated_device_status(device_id: str, request: DedicatedDeviceStatusUpdate) -> DedicatedDevice:
    device = device_manager.update_status(device_id, request.status)
    if device is None:
        raise HTTPException(status_code=404, detail=f"Device not found: {device_id}")
    return device


@app.delete("/api/devices/{device_id}", status_code=204)
def delete_dedicated_device(device_id: str) -> Response:
    deleted = device_manager.delete(device_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Device not found: {device_id}")
    return Response(status_code=204)


@app.get("/api/scenarios/{scenario_name}/base-stations", response_model=ScenarioBaseStationResponse)
def get_scenario_base_stations(scenario_name: str) -> ScenarioBaseStationResponse:
    try:
        dataset.get(scenario_name)
        return _scenario_base_station_response(scenario_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_name}") from exc


@app.put("/api/scenarios/{scenario_name}/base-stations", response_model=ScenarioBaseStationResponse)
def update_scenario_base_stations(
    scenario_name: str,
    request: ScenarioBaseStationUpdate,
) -> ScenarioBaseStationResponse:
    try:
        dataset.get(scenario_name)
        raw_specs = [station.model_dump() for station in request.base_stations]
        _persist_scenario_device_state(scenario_name, raw_specs, operation="replace_base_stations")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_name}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _scenario_base_station_response(scenario_name)


@app.delete("/api/scenarios/{scenario_name}/base-stations", response_model=ScenarioBaseStationResponse)
def reset_scenario_base_stations(scenario_name: str) -> ScenarioBaseStationResponse:
    try:
        dataset.get(scenario_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_name}") from exc

    scenario_device_manager.reset_scenario(
        scenario_name,
        _normalize_scenario_base_stations(
            scenario_name,
            _default_scenario_base_stations(scenario_name),
        ),
        operation="reset_base_stations",
    )
    _invalidate_scenarios_cache()

    return _scenario_base_station_response(scenario_name)


@app.get("/api/scenarios/{scenario_name}/device-state", response_model=ScenarioDeviceStateResponse)
def get_scenario_device_state(scenario_name: str) -> ScenarioDeviceStateResponse:
    try:
        dataset.get(scenario_name)
        return _scenario_device_state_payload(scenario_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_name}") from exc


@app.put("/api/scenarios/{scenario_name}/device-state", response_model=ScenarioDeviceStateResponse)
def update_scenario_device_state(
    scenario_name: str,
    request: ScenarioDeviceStateUpdate,
) -> ScenarioDeviceStateResponse:
    try:
        dataset.get(scenario_name)
        current_specs = _scenario_base_station_specs_raw(scenario_name)
        raw_specs = [station.model_dump() for station in request.base_stations] if request.base_stations is not None else current_specs
        raw_type_overrides = (
            {key: value.model_dump(exclude_unset=True) for key, value in request.type_overrides.items()}
            if request.type_overrides is not None
            else None
        )
        return _persist_scenario_device_state(
            scenario_name,
            raw_specs,
            type_overrides=raw_type_overrides,
            operation=request.operation or "update_device_state",
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_name}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/scenarios/{scenario_name}/devices", response_model=ScenarioDeviceStateResponse, status_code=201)
def add_scenario_device(scenario_name: str, request: CustomBaseStation) -> ScenarioDeviceStateResponse:
    try:
        dataset.get(scenario_name)
        specs = _scenario_base_station_specs_raw(scenario_name)
        specs.append(request.model_dump(exclude_unset=True))
        return _persist_scenario_device_state(scenario_name, specs, operation="add_device")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_name}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch("/api/scenarios/{scenario_name}/devices/{device_uid}", response_model=ScenarioDeviceStateResponse)
def update_scenario_device(
    scenario_name: str,
    device_uid: str,
    request: Dict[str, Any],
) -> ScenarioDeviceStateResponse:
    try:
        dataset.get(scenario_name)
        specs = _scenario_base_station_specs_raw(scenario_name)
        updated = False
        for spec in specs:
            if str(spec.get("device_uid") or spec.get("deployment_id") or "") != device_uid:
                continue
            override_fields = set(spec.get("override_fields") or [])
            for key, value in request.items():
                if value is not None and key not in {"id"}:
                    spec[key] = value
                    if key in DEVICE_CONFIG_FIELDS:
                        override_fields.add(key)
            if override_fields:
                spec["override_fields"] = sorted(override_fields)
            updated = True
            break
        if not updated:
            raise HTTPException(status_code=404, detail=f"Device not found: {device_uid}")
        return _persist_scenario_device_state(scenario_name, specs, operation="update_device")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_name}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/scenarios/{scenario_name}/devices/{device_uid}", response_model=ScenarioDeviceStateResponse)
def delete_scenario_device(scenario_name: str, device_uid: str) -> ScenarioDeviceStateResponse:
    try:
        dataset.get(scenario_name)
        specs = _scenario_base_station_specs_raw(scenario_name)
        remaining = [
            spec for spec in specs
            if str(spec.get("device_uid") or spec.get("deployment_id") or "") != device_uid
        ]
        if len(remaining) == len(specs):
            raise HTTPException(status_code=404, detail=f"Device not found: {device_uid}")
        return _persist_scenario_device_state(scenario_name, remaining, operation="delete_device")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_name}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch("/api/scenarios/{scenario_name}/device-blocks", response_model=ScenarioDeviceStateResponse)
def update_scenario_device_block(
    scenario_name: str,
    request: ScenarioDeviceBlockUpdate,
) -> ScenarioDeviceStateResponse:
    try:
        dataset.get(scenario_name)
        specs = _scenario_base_station_specs_raw(scenario_name)
        mode = request.mode
        if not mode:
            profile = dataset.get(scenario_name).base_station_profiles.get(request.base_station)
            mode = profile.supported_modes[0] if profile and profile.supported_modes else None
        kept = [
            spec for spec in specs
            if not (
                int(spec.get("x", -1)) == request.x
                and int(spec.get("y", -1)) == request.y
                and str(spec.get("base_station")) == request.base_station
                and (not mode or str(spec.get("mode")) == str(mode))
            )
        ]
        for index in range(request.quantity):
            kept.append(
                {
                    **request.parameters,
                    "device_uid": f"{scenario_name}:block:{request.x}:{request.y}:{request.base_station}:{mode}:{index + 1}",
                    "base_station": request.base_station,
                    "mode": mode,
                    "x": request.x,
                    "y": request.y,
                    "status": request.status or "active",
                }
            )
        return _persist_scenario_device_state(
            scenario_name,
            kept,
            operation=request.operation or "update_block_quantity",
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_name}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/train", response_model=TrainResponse)
def start_training(request: TrainRequest) -> TrainResponse:
    run = training_manager.start_run(
        scenario_name=request.scenario_name,
        env_type=request.env_type,
        algorithm=request.algorithm,
        total_timesteps=request.total_timesteps,
        stochastic_eval=request.stochastic_eval,
        reward_mode=request.reward_mode,
        evaluation_protocol=request.evaluation_protocol or _default_protocol_for_scenario(request.scenario_name),
        learning_rate=request.learning_rate,
        discount_factor=request.discount_factor,
        batch_size=request.batch_size,
        rollout_steps=request.rollout_steps,
        entropy_coef=request.entropy_coef,
        clip_range=request.clip_range,
        eval_interval=request.eval_interval,
        custom_base_stations=_request_base_stations_or_scenario_state(
            request.scenario_name,
            request.custom_base_stations,
            use_scenario_state=not _is_level4_scenario_name(request.scenario_name),
        ),
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
        algorithm=run.algorithm,
        reward_mode=run.reward_mode,
        evaluation_protocol=run.evaluation_protocol,
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
    env_type = "multimodal" if request.algorithm == "hmarl" else request.env_type
    config["experiment"]["env_type"] = env_type
    config["experiment"]["algorithm"] = request.algorithm
    if env_type == "multimodal":
        config["multimodal_env"]["scenario_name"] = request.scenario_name
        if request.reward_mode is not None:
            config["multimodal_env"]["reward_mode"] = request.reward_mode
    apply_evaluation_protocol(config, request.evaluation_protocol or _default_protocol_for_scenario(request.scenario_name))
    apply_level4_algorithm_profile(config, request.algorithm)
    level4_benchmark = bool(config.get("evaluation", {}).get("level4_benchmark", False))
    if request.eval_seed is not None:
        torch.manual_seed(request.eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request.eval_seed)
        np.random.seed(request.eval_seed)

    checkpoint_path = Path(request.checkpoint_path)
    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")

    env = build_env(config, env_type)
    try:
        policy = load_policy(checkpoint_path, env, config, env_type, algorithm=request.algorithm)

        custom_state = [device.model_dump() for device in request.custom_devices]
        custom_base_stations = (
            [station.model_dump() for station in request.custom_base_stations]
            if request.custom_base_stations is not None
            else None
        )
        custom_base_stations = _request_base_stations_or_scenario_state(
            request.scenario_name,
            custom_base_stations,
            use_scenario_state=True,
        )
        rewards, coverages, reports = evaluate_policy(
            env=env,
            policy=policy,
            episodes=request.episodes,
            deterministic=not request.stochastic_eval,
            render=False,
            custom_user_state=custom_state or None,
            custom_base_stations=custom_base_stations,
            dqn_use_lookahead=bool(config.get("evaluation", {}).get("dqn_use_lookahead", True)),
        )
        scene_export = None
        if reports:
            export_dir = Path(config["logging"]["artifact_dir"]) / "scene_exports"
            scene_export = export_episode_scene(reports[0], env, export_dir)

        response = SimulationResponse(
            avg_reward=float(np.mean(rewards)),
            avg_final_coverage=float(np.mean(coverages)),
            reports=reports,
            deployment_plan=reports[0].get("deployment_plan") if reports else None,
            scene_export=scene_export,
        )
        _sync_scenario_devices_from_strategy_result(request.scenario_name, custom_base_stations, reports)
        return _attach_replay_session(request, response)
    finally:
        env.close()


@app.post("/api/simulate/scene", response_model=SceneImportResponse)
def import_simulation_scene(request: SceneImportRequest) -> SceneImportResponse:
    config = get_default_config()
    config["experiment"]["env_type"] = request.env_type
    if request.env_type == "multimodal":
        config["multimodal_env"]["scenario_name"] = request.scenario_name
    apply_evaluation_protocol(config, request.evaluation_protocol or _default_protocol_for_scenario(request.scenario_name))
    apply_level4_algorithm_profile(config, None)
    level4_benchmark = bool(config.get("evaluation", {}).get("level4_benchmark", False))

    env = build_env(config, request.env_type)
    custom_base_stations = (
        [station.model_dump() for station in request.custom_base_stations]
        if request.custom_base_stations is not None
        else None
    )
    custom_base_stations = _request_base_stations_or_scenario_state(
        request.scenario_name,
        custom_base_stations,
        use_scenario_state=True,
    )

    try:
        preview = build_scene_preview(env, custom_base_stations=custom_base_stations)
        return SceneImportResponse(**preview)
    finally:
        env.close()


@app.post("/api/simulate/stream")
def stream_simulation(request: SimulationRequest):
    def encode_sse(event: Dict[str, object]) -> str:
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    def build_event(event_type: str, payload: Dict[str, object]) -> Dict[str, object]:
        return {
            "type": event_type,
            "payload": payload,
            "timestamp": time.time(),
        }

    event_queue: "queue.Queue[Dict[str, object]]" = queue.Queue()

    def push_event(event_type: str, payload: Dict[str, object]) -> None:
        event_queue.put(build_event(event_type, payload))

    def run_simulation() -> None:
        env = None
        final_state = "failed"
        try:
            push_event("status", {"state": "initializing"})
            push_event("log", {"message": f"开始加载场景 {request.scenario_name}。"})
            if request.dataset_import_ids:
                push_event("log", {"message": f"灾害数据来源 import_id={', '.join(request.dataset_import_ids)}。"})
            config = get_default_config()
            env_type = "multimodal" if request.algorithm == "hmarl" else request.env_type
            config["experiment"]["env_type"] = env_type
            config["experiment"]["algorithm"] = request.algorithm
            if env_type == "multimodal":
                config["multimodal_env"]["scenario_name"] = request.scenario_name
                if request.reward_mode is not None:
                    config["multimodal_env"]["reward_mode"] = request.reward_mode
            apply_evaluation_protocol(
                config,
                request.evaluation_protocol or _default_protocol_for_scenario(request.scenario_name),
            )
            apply_level4_algorithm_profile(config, request.algorithm)
            level4_benchmark = bool(config.get("evaluation", {}).get("level4_benchmark", False))
            if request.eval_seed is not None:
                torch.manual_seed(request.eval_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(request.eval_seed)
                np.random.seed(request.eval_seed)

            checkpoint_path = Path(request.checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            env = build_env(config, env_type)
            custom_state = [device.model_dump() for device in request.custom_devices]
            for message in _dedicated_device_log_messages(custom_state):
                push_event("log", {"message": message, "event_type": "dedicated_device"})
            custom_base_stations = (
                [station.model_dump() for station in request.custom_base_stations]
                if request.custom_base_stations is not None
                else None
            )
            custom_base_stations = _request_base_stations_or_scenario_state(
                request.scenario_name,
                custom_base_stations,
                use_scenario_state=True,
            )

            push_event(
                "log",
                {
                    "message": f"加载模型 {checkpoint_path}，算法={request.algorithm}，协议={config.get('evaluation', {}).get('protocol', 'standard')}，episodes={request.episodes}，stochastic={request.stochastic_eval}，seed={request.eval_seed if request.eval_seed is not None else 'auto'}。"
                },
            )
            policy = load_policy(checkpoint_path, env, config, env_type, algorithm=request.algorithm)
            push_event("status", {"state": "running"})
            push_event("log", {"message": "测试开始，终端将持续输出策略执行过程。"})

            def push_progress(event: Dict[str, object]) -> None:
                message = str(event.get("message", "")).strip()
                if not message:
                    return
                push_event(
                    "log",
                    {
                        "message": message,
                        "event_type": str(event.get("type", "log")),
                    },
                )

            rewards, coverages, reports = evaluate_policy(
                env=env,
                policy=policy,
                episodes=request.episodes,
                deterministic=not request.stochastic_eval,
                render=False,
                custom_user_state=custom_state or None,
                custom_base_stations=custom_base_stations,
                dqn_use_lookahead=bool(config.get("evaluation", {}).get("dqn_use_lookahead", True)),
                progress_callback=push_progress,
            )

            scene_export = None
            if reports:
                export_dir = Path(config["logging"]["artifact_dir"]) / "scene_exports"
                scene_export = export_episode_scene(reports[0], env, export_dir)

            response = SimulationResponse(
                avg_reward=float(np.mean(rewards)),
                avg_final_coverage=float(np.mean(coverages)),
                reports=reports,
                deployment_plan=reports[0].get("deployment_plan") if reports else None,
                scene_export=scene_export,
            )
            sync_result = _sync_scenario_devices_from_strategy_result(
                request.scenario_name,
                custom_base_stations,
                reports,
            )
            if sync_result.get("updated"):
                counts = sync_result.get("status_counts") or {}
                push_event(
                    "log",
                    {
                        "message": (
                            "设备数据库已按策略测试终态同步："
                            f"更新 {sync_result.get('changed_count', 0)} 台；"
                            f"在线 {counts.get('active', 0)} / 降级 {counts.get('degraded', 0)} / 离线 {counts.get('offline', 0)}。"
                        ),
                        "event_type": "device_state_sync",
                    },
                )
            push_event("result", _compact_stream_response(response).model_dump())
            push_event("status", {"state": "completed"})
            final_state = "completed"
        except Exception as exc:  # pylint: disable=broad-except
            push_event("error", {"message": str(exc)})
            push_event("status", {"state": "failed"})
        finally:
            if env is not None:
                env.close()
            push_event("end", {"state": final_state})

    def event_generator():
        worker = threading.Thread(target=run_simulation, daemon=True)
        worker.start()

        while True:
            try:
                event = event_queue.get(timeout=0.5)
            except queue.Empty:
                if not worker.is_alive():
                    break
                continue
            yield encode_sse(event)
            if event.get("type") == "end":
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Scene replay endpoints
# ---------------------------------------------------------------------------

@app.get("/api/test/outputs")
def list_test_outputs(limit: int = Query(50, ge=1, le=200)) -> Dict[str, object]:
    scene_export_root = Path(default_config["logging"]["artifact_dir"]) / "scene_exports"
    outputs: List[Dict[str, object]] = []
    if scene_export_root.exists():
        for metadata_path in scene_export_root.glob("*/metadata.json"):
            record = _scene_export_record(metadata_path)
            if record is not None:
                outputs.append(record)

    outputs.sort(key=lambda item: float(item.get("created_at") or 0), reverse=True)
    return {"outputs": outputs[:limit], "total": len(outputs)}


@app.get("/api/replay/sessions")
def list_replay_sessions(
    limit: int = Query(20, ge=1, le=100),
    source: Optional[str] = Query(None),
) -> Dict[str, object]:
    return replay_session_manager.list_sessions(limit=limit, source=source)


@app.get("/api/replay/sessions/{replay_id}")
def get_replay_session(replay_id: str) -> Dict[str, object]:
    try:
        return replay_session_manager.get_session(replay_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/replay/sessions/{replay_id}/frames")
def get_replay_frames(
    replay_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    stride: int = Query(1, ge=1),
) -> Dict[str, object]:
    try:
        return replay_session_manager.get_frames(replay_id, offset=offset, limit=limit, stride=stride)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/replay/sessions/{replay_id}/frames/{frame_index}")
def get_replay_frame(
    replay_id: str,
    frame_index: int,
    sample_ratio: int = Query(30, ge=1, le=200),
    include_links: bool = Query(True),
) -> Dict[str, object]:
    try:
        return replay_session_manager.get_frame(
            replay_id,
            frame_index,
            sample_ratio=sample_ratio,
            include_links=include_links,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/replay/sessions/{replay_id}/logs")
def get_replay_logs(
    replay_id: str,
    limit: int = Query(200, ge=1, le=1000),
) -> Dict[str, object]:
    try:
        return replay_session_manager.get_logs(replay_id, limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/replay/sessions/{replay_id}/link-metrics")
def get_replay_link_metrics(
    replay_id: str,
    frame_index: Optional[int] = Query(None, ge=0),
) -> Dict[str, object]:
    try:
        return replay_session_manager.get_link_metrics(replay_id, frame_index=frame_index)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/replay/sessions/{replay_id}/download")
def download_replay_artifact(
    replay_id: str,
    type: str = Query("log"),  # pylint: disable=redefined-builtin
) -> FileResponse:
    try:
        path = replay_session_manager.artifact_path(replay_id, type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path=path, filename=path.name)


# ---------------------------------------------------------------------------
# Disaster data import endpoints
# ---------------------------------------------------------------------------

@app.get("/api/disaster-scenarios")
def list_disaster_scenarios() -> Dict[str, object]:
    return disaster_import_manager.list_scenarios()


@app.get("/api/disaster-scenarios/{scenario}")
def get_disaster_scenario(scenario: str) -> Dict[str, object]:
    try:
        return disaster_import_manager.get_scenario(scenario)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/disaster-scenarios/{scenario}/severity-levels/{severity}")
def get_disaster_severity_overview(scenario: str, severity: str) -> Dict[str, object]:
    try:
        return disaster_import_manager.get_severity_overview(scenario, severity)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get(
    "/api/disaster-scenarios/{scenario}/severity-levels/{severity}"
    "/station-profiles/{comm_type}/{station_type}"
)
def get_disaster_station_profile(
    scenario: str, severity: str, comm_type: str, station_type: str
) -> Dict[str, object]:
    try:
        return disaster_import_manager.get_station_profile(scenario, severity, comm_type, station_type)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/disaster-imports", response_model=DisasterImportSummary, status_code=201)
def create_disaster_import(request: DisasterImportRequest) -> DisasterImportSummary:
    try:
        detail = disaster_import_manager.create_import(
            disaster_scenario=request.disaster_scenario,
            disaster_severity=request.disaster_severity,
            session_sample_limit=request.session_sample_limit,
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Dataset read error: {exc}") from exc
    # Return summary (without heavy deployments/heatmap list)
    return DisasterImportSummary(
        import_id=detail.import_id,
        disaster_scenario=detail.disaster_scenario,
        disaster_scenario_label=detail.disaster_scenario_label,
        disaster_severity=detail.disaster_severity,
        disaster_severity_label=detail.disaster_severity_label,
        session_sample_limit=detail.session_sample_limit,
        status=detail.status,
        imported_at=detail.imported_at,
        effective_geo_bounds=detail.effective_geo_bounds,
        grid_size=detail.grid_size,
        station_counts=detail.station_counts,
        unique_user_count=detail.unique_user_count,
        total_sessions_sampled=detail.total_sessions_sampled,
        comm_type_breakdown=detail.comm_type_breakdown,
    )


@app.get("/api/disaster-imports", response_model=DisasterImportListResponse)
def list_disaster_imports() -> DisasterImportListResponse:
    return disaster_import_manager.list_imports()


@app.get("/api/disaster-imports/{import_id}", response_model=DisasterImportDetail)
def get_disaster_import(import_id: str) -> DisasterImportDetail:
    try:
        return disaster_import_manager.get_import(import_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Import session not found: {import_id}")


@app.delete("/api/disaster-imports/{import_id}", status_code=204)
def delete_disaster_import(import_id: str) -> None:
    try:
        disaster_import_manager.delete_import(import_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Import session not found: {import_id}")


# ---------------------------------------------------------------------------
# Mahimahi endpoints
# ---------------------------------------------------------------------------

@app.get("/api/mahimahi/status")
def mahimahi_status() -> Dict[str, object]:
    return {
        "mahimahi_available": mahimahi_manager.mahimahi_available,
        "traces_dir": str(mahimahi_manager.traces_dir),
    }


@app.get("/api/mahimahi/traces")
def list_mahimahi_traces() -> Dict[str, List[Dict[str, object]]]:
    return {"traces": mahimahi_manager.list_traces()}


@app.get("/api/mahimahi/traces/{trace_name}")
def get_trace_analysis(
    trace_name: str,
    duration_s: float = 60,
    window_ms: int = 500,
) -> Dict[str, object]:
    try:
        return mahimahi_manager.analyze_trace(trace_name, duration_s, window_ms)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Trace '{trace_name}' not found")


@app.post("/api/mahimahi/simulate", response_model=MahimahiSimulateResponse)
def mahimahi_simulate(request: MahimahiSimulateRequest) -> MahimahiSimulateResponse:
    try:
        result = mahimahi_manager.simulate(
            trace_name=request.trace_name,
            duration_s=request.duration_s,
            rtt_ms=request.rtt_ms,
            buffer_packets=request.buffer_packets,
            window_ms=request.window_ms,
        )
        return MahimahiSimulateResponse(**result)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Trace '{request.trace_name}' not found")


@app.post("/api/mahimahi/simulate/stream")
def mahimahi_simulate_stream(request: MahimahiSimulateRequest):
    """SSE stream that pushes simulation progress and final results."""

    def encode_sse(event: Dict[str, object]) -> str:
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    event_queue: "queue.Queue[Dict[str, object]]" = queue.Queue()

    def run() -> None:
        try:
            event_queue.put({"type": "status", "payload": {"state": "initializing"}, "timestamp": time.time()})
            event_queue.put({
                "type": "log",
                "payload": {"message": f"加载 trace: {request.trace_name}, RTT={request.rtt_ms}ms, 时长={request.duration_s}s"},
                "timestamp": time.time(),
            })

            result = mahimahi_manager.simulate(
                trace_name=request.trace_name,
                duration_s=request.duration_s,
                rtt_ms=request.rtt_ms,
                buffer_packets=request.buffer_packets,
                window_ms=request.window_ms,
            )

            n = len(result.get("capacity", []))
            chunk_size = max(1, n // 20)
            for i in range(0, n, chunk_size):
                chunk = {
                    "capacity": result["capacity"][i : i + chunk_size],
                    "throughput": result["throughput"][i : i + chunk_size],
                    "sending_rate": result["sending_rate"][i : i + chunk_size],
                }
                event_queue.put({"type": "data_chunk", "payload": chunk, "timestamp": time.time()})
                time.sleep(0.05)

            event_queue.put({"type": "result", "payload": {"stats": result["stats"]}, "timestamp": time.time()})
            event_queue.put({"type": "end", "payload": {"state": "completed"}, "timestamp": time.time()})
        except Exception as exc:
            event_queue.put({"type": "error", "payload": {"message": str(exc)}, "timestamp": time.time()})
            event_queue.put({"type": "end", "payload": {"state": "failed"}, "timestamp": time.time()})

    def generator():
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        while True:
            try:
                event = event_queue.get(timeout=0.5)
            except queue.Empty:
                if not worker.is_alive():
                    break
                continue
            yield encode_sse(event)
            if event.get("type") == "end":
                break

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.on_event("startup")
async def startup_event() -> None:
    ns3_replay_manager.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    ns3_replay_manager.stop()


@app.get("/api/ns3/status")
def ns3_status() -> Dict[str, object]:
    return ns3_replay_manager.status()


@app.post("/api/ns3/run")
def ns3_run() -> Dict[str, object]:
    return ns3_replay_manager.start_simulation()


@app.post("/api/import")
def ns3_manual_import() -> Dict[str, object]:
    exp_id = ns3_replay_manager.import_trace()
    return {"success": exp_id is not None, "exp_id": exp_id}


@app.get("/api/experiments")
def list_ns3_experiments() -> List[Dict[str, object]]:
    return ns3_replay_manager.list_experiments()


@app.get("/api/exp/{exp_id}/charts")
def get_ns3_charts(exp_id: int) -> Dict[str, List[float]]:
    return ns3_replay_manager.get_charts(exp_id)


@app.get("/api/exp/{exp_id}/frame/{frame_idx}")
def get_ns3_frame(exp_id: int, frame_idx: int) -> Dict[str, object]:
    frame = ns3_replay_manager.get_frame(exp_id, frame_idx)
    if frame is None:
        raise HTTPException(status_code=404, detail="Frame not found.")
    return frame


def _resolve_frontend_dist() -> Path:
    configured = Path(os.environ.get("FRONTEND_DIST", "frontend/dist"))
    if configured.is_absolute():
        return configured
    project_root = Path(__file__).resolve().parents[1]
    return project_root / configured


def _mount_frontend_static() -> None:
    frontend_dist = _resolve_frontend_dist()
    if not frontend_dist.exists():
        return

    # Keep API routes unchanged, and serve built frontend for root/static files.
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


def _mount_ns3_native_static() -> None:
    if not ns3_replay_manager.ns3_root.exists():
        return
    app.mount("/ns3-native", StaticFiles(directory=ns3_replay_manager.ns3_root, html=True), name="ns3-native")


_mount_ns3_native_static()
_mount_frontend_static()
