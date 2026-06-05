"""Resolve HMARL scenario aliases to RescueNet multimodal dataset keys."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import yaml

from rescuenet.bootstrap import REPO_ROOT, RESCUENET_DIR

LEGACY_DATASET = "data/scenarios.json"
EXTREME_DATASET = "data/extreme_disaster_resources/regions.json"
DEFAULT_SEVERITY = "level_4"

_LEGACY_MAP: Dict[str, str] = {
    "extreme_rainstorm": "flood_no_residual",
    "super_typhoon": "typhoon_residual",
    "typhoon_residual": "typhoon_residual",
    "flood_no_residual": "flood_no_residual",
    "earthquake_residual": "earthquake_residual",
    "destructive_earthquake": "earthquake_residual",
}

_EXTREME_MAP: Dict[str, str] = {
    "extreme_rainstorm": f"extreme_rainstorm__{DEFAULT_SEVERITY}",
    "super_typhoon": f"super_typhoon__{DEFAULT_SEVERITY}",
    "destructive_earthquake": f"destructive_earthquake__{DEFAULT_SEVERITY}",
    "typhoon_residual": f"super_typhoon__{DEFAULT_SEVERITY}",
    "flood_no_residual": f"extreme_rainstorm__{DEFAULT_SEVERITY}",
    "earthquake_residual": f"destructive_earthquake__{DEFAULT_SEVERITY}",
}


def extreme_dataset_path(*, repo_root: Path | None = None) -> Path:
    return (repo_root or REPO_ROOT) / EXTREME_DATASET


def legacy_dataset_path(*, repo_root: Path | None = None) -> Path:
    return (repo_root or REPO_ROOT) / LEGACY_DATASET


def active_dataset_path(*, repo_root: Path | None = None) -> str:
    """Prefer the imported extreme-disaster bundle when present."""
    if extreme_dataset_path(repo_root=repo_root).exists():
        return EXTREME_DATASET
    return LEGACY_DATASET


def load_scenario_map(*, dataset_path: str | None = None) -> Dict[str, str]:
    """Return alias -> scenario_name for the active or requested dataset."""
    path = dataset_path or active_dataset_path()
    base = _EXTREME_MAP if path == EXTREME_DATASET else _LEGACY_MAP
    yaml_path = RESCUENET_DIR / "scenarios.yaml"
    if yaml_path.exists():
        with yaml_path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        merged = dict(base)
        merged.update({str(k): str(v) for k, v in data.items()})
        return merged
    return dict(base)


def resolve_scenario(name: str, *, dataset_path: str | None = None) -> str:
    key = name.strip()
    mapping = load_scenario_map(dataset_path=dataset_path)
    if key in mapping:
        return mapping[key]
    if key in mapping.values():
        return key
    known = ", ".join(sorted(mapping))
    raise ValueError(f"Unknown scenario {name!r}. Known HMARL aliases: {known}")


def resolve_multimodal_env(name: str, *, repo_root: Path | None = None) -> Tuple[str, str]:
    """Return (dataset_path, scenario_name) for HMARL / RescueNet multimodal training."""
    dataset_path = active_dataset_path(repo_root=repo_root)
    scenario_name = resolve_scenario(name, dataset_path=dataset_path)
    return dataset_path, scenario_name


def apply_multimodal_scenario(config: dict, alias: str, *, repo_root: Path | None = None) -> Tuple[str, str]:
    dataset_path, scenario_name = resolve_multimodal_env(alias, repo_root=repo_root)
    config["multimodal_env"]["dataset_path"] = dataset_path
    config["multimodal_env"]["scenario_name"] = scenario_name
    return dataset_path, scenario_name
