"""Export HMARLPolicy checkpoints into L1/L2/L3 weight files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def _subset_state_dict(full: Dict[str, Any], prefixes: tuple[str, ...]) -> Dict[str, Any]:
    return {key: value for key, value in full.items() if key.startswith(prefixes)}


def export_layer_weights(full_state: Dict[str, Any], weights_dir: Path) -> Dict[str, Path]:
    """
    L1: 全局统筹 — 共享编码器 + L1 区域头
    L2: 区域调控 — 共享编码器 + L2 链路头
    L3: 本地执行 — 共享编码器 + L3 动作头 + Critic
    """
    weights_dir.mkdir(parents=True, exist_ok=True)
    body = _subset_state_dict(full_state, ("body.",))
    layers = {
        "L1.pt": {**body, **_subset_state_dict(full_state, ("l1_head.",))},
        "L2.pt": {**body, **_subset_state_dict(full_state, ("l2_head.",))},
        "L3.pt": {
            **body,
            **_subset_state_dict(full_state, ("l3_actor_head.", "critic_head.")),
        },
    }
    paths: Dict[str, Path] = {}
    for filename, state in layers.items():
        path = weights_dir / filename
        torch.save(state, path)
        paths[filename] = path
    return paths


def merge_layer_weights(weights_dir: Path) -> Dict[str, Any]:
    """Rebuild a full HMARLPolicy state_dict from weights/L1.pt, L2.pt, L3.pt."""
    load_kwargs = {"map_location": "cpu"}
    try:
        l1 = torch.load(weights_dir / "L1.pt", weights_only=True, **load_kwargs)
        l2 = torch.load(weights_dir / "L2.pt", weights_only=True, **load_kwargs)
        l3 = torch.load(weights_dir / "L3.pt", weights_only=True, **load_kwargs)
    except TypeError:
        l1 = torch.load(weights_dir / "L1.pt", **load_kwargs)
        l2 = torch.load(weights_dir / "L2.pt", **load_kwargs)
        l3 = torch.load(weights_dir / "L3.pt", **load_kwargs)
    merged: Dict[str, Any] = {}
    merged.update({key: value for key, value in l1.items() if key.startswith("l1_head.")})
    merged.update({key: value for key, value in l2.items() if key.startswith("l2_head.")})
    merged.update(
        {
            key: value
            for key, value in l3.items()
            if key.startswith(("body.", "l3_actor_head.", "critic_head."))
        }
    )
    return merged


def weights_dir_for_scenario(checkpoints_root: Path, scenario_alias: str) -> Path:
    return checkpoints_root / scenario_alias / "weights"
