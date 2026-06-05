"""Parse HMARL checkpoint / train log for optional RL-enhanced plan generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ._config import HMARL_ROOT


def find_checkpoint_dir(scenario_id: str) -> Optional[Path]:
    ckpt = HMARL_ROOT / "checkpoints" / scenario_id
    if ckpt.is_dir() and (ckpt / "train_log.json").exists():
        return ckpt
    return None


def parse_rl_output(scenario_id: str) -> Dict[str, Any]:
    """
    Read HMARL training artifacts if available.
    Returns metadata and final metrics; does not load torch weights.
    """
    ckpt_dir = find_checkpoint_dir(scenario_id)
    result: Dict[str, Any] = {
        "scenario_id": scenario_id,
        "checkpoint_available": ckpt_dir is not None,
        "checkpoint_dir": str(ckpt_dir) if ckpt_dir else None,
        "source": "rule_based",
    }

    if ckpt_dir is None:
        return result

    log_path = ckpt_dir / "train_log.json"
    try:
        with log_path.open("r", encoding="utf-8") as f:
            log = json.load(f)
    except (json.JSONDecodeError, OSError):
        return result

    result["source"] = "hmari_checkpoint_enhanced"
    episodes = log if isinstance(log, list) else log.get("episodes", [])
    if episodes:
        last = episodes[-1] if isinstance(episodes[-1], dict) else {}
        result["final_metrics"] = {
            "episode": last.get("episode", len(episodes)),
            "reward": last.get("reward") or last.get("total_reward"),
            "coverage": last.get("coverage"),
            "broadcast_coverage": last.get("broadcast_coverage"),
        }
    result["train_log_entries"] = len(episodes) if isinstance(episodes, list) else 0

    for name in ("actor_l1.pt", "actor_l2.pt", "actor_l3.pt"):
        p = ckpt_dir / name
        if p.exists():
            result.setdefault("weight_files", []).append(name)

    return result
