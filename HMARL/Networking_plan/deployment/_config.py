"""Shared configuration loading for metric2 network architecture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
ARCH_DIR = ROOT / "architecture"
SCENARIO_DIR = ROOT / "scenarios"
MODE_DIR = ROOT / "network_modes"
DEPLOY_DIR = ROOT / "deployment"
OUTPUT_DIR = ROOT / "outputs"
PROOFS_DIR = ROOT / "proofs"
HMARL_ROOT = ROOT.parent

SCENARIO_IDS = ("extreme_rainstorm", "super_typhoon")
NETWORK_MODES = ("with_residual", "no_residual")

OUTPUT_DIR_MAP = {
    ("extreme_rainstorm", "with_residual"): "rainstorm_with_residual",
    ("extreme_rainstorm", "no_residual"): "rainstorm_no_residual",
    ("super_typhoon", "with_residual"): "typhoon_with_residual",
    ("super_typhoon", "no_residual"): "typhoon_no_residual",
}


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML required: pip install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_architecture() -> Dict[str, Any]:
    return {
        "L1": load_yaml(ARCH_DIR / "l1_global_layer.yaml"),
        "L2": load_yaml(ARCH_DIR / "l2_fusion_layer.yaml"),
        "L3": load_yaml(ARCH_DIR / "l3_execution_layer.yaml"),
        "comm_modes": load_yaml(ARCH_DIR / "comm_modes.yaml"),
    }


def load_scenario(scenario_id: str) -> Dict[str, Any]:
    path = SCENARIO_DIR / f"{scenario_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Scenario config not found: {path}")
    return load_yaml(path)


def load_network_mode(mode: str) -> Dict[str, Any]:
    base = MODE_DIR / mode
    return {
        "mode_config": load_yaml(base / "mode_config.yaml"),
        "deploy_rules": load_yaml(base / "deploy_rules.yaml"),
        "topology_template": load_json(base / "topology_template.json"),
    }


def load_phased_deploy() -> Dict[str, Any]:
    return load_yaml(DEPLOY_DIR / "phased_deploy.yaml")


def get_comm_mode_ids(arch: Optional[Dict[str, Any]] = None) -> List[str]:
    arch = arch or load_architecture()
    modes = arch.get("comm_modes", {}).get("comm_modes", [])
    return [m["id"] for m in modes if isinstance(m, dict) and "id" in m]
