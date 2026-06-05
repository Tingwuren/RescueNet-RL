"""Shared scenario validation logic."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment._config import HMARL_ROOT, OUTPUT_DIR, OUTPUT_DIR_MAP, load_scenario
from deployment.export_plan import export_plan
from validation._utils import print_header


def _load_hmarl_scenario(scenario_id: str) -> Dict[str, Any]:
    path = HMARL_ROOT / "configs" / "scenarios" / f"{scenario_id}.yaml"
    if not path.exists():
        return {}
    try:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def validate_scenario(scenario_id: str, scenario_label: str) -> bool:
    print_header(f"指标二：{scenario_label}场景验证")

    scenario = load_scenario(scenario_id)
    hmarl = _load_hmarl_scenario(scenario_id)

    print(f"\n[场景标识] {scenario_id} ({scenario.get('scenario_name', '')})")
    print(f"[残余形态] {scenario.get('residual_pattern')}")
    print(f"[基站退服率] {scenario.get('base_station_outage_min')}-{scenario.get('base_station_outage_max')}")
    print(f"[道路通行率] {scenario.get('road_pass_rate')}")

    if scenario_id == "extreme_rainstorm":
        print(f"[链路断裂率] {scenario.get('link_breakage_rate')}")
        print(f"[核心特征] {scenario.get('core_feature', '')}")
    else:
        print(f"[倒杆率] {scenario.get('pole_damage_rate_min')}-{scenario.get('pole_damage_rate_max')}")
        print(f"[局部全阻区] {scenario.get('local_blackout_zones')}")
        print(f"[核心特征] {scenario.get('core_feature', '')}")

    if hmarl:
        print(f"\n[HMARL 交叉引用] configs/scenarios/{scenario_id}.yaml")
        print(f"  HMARL network_mode: {hmarl.get('network_mode', 'N/A')}")
        ckpt = HMARL_ROOT / "checkpoints" / scenario_id / "train_log.json"
        print(f"  训练日志: {'存在' if ckpt.exists() else '缺失'} ({ckpt})")

    print("\n[生成组网方案]")
    paths = []
    for mode in ("with_residual", "no_residual"):
        p = export_plan(scenario_id, mode, verbose=False)
        paths.append(p)
        print(f"  {mode}: {p}")

    all_ok = True
    print("\n[方案关键字段]")
    for mode in ("with_residual", "no_residual"):
        sub = OUTPUT_DIR_MAP[(scenario_id, mode)]
        plan_path = OUTPUT_DIR / sub / "network_plan.json"
        with plan_path.open("r", encoding="utf-8") as f:
            plan = json.load(f)
        residual = plan.get("residual_nodes_reused", -1)
        emergency = plan.get("emergency_nodes_deployed", -1)
        backhaul = plan.get("primary_backhaul", "")
        mode_ok = (residual > 0 if mode == "with_residual" else residual == 0)
        print(f"  {mode}: residual={residual} emergency={emergency} backhaul={backhaul} [{'OK' if mode_ok else 'FAIL'}]")
        all_ok = all_ok and mode_ok

    status = "PASS" if all_ok else "FAIL"
    print(f"\n[{status}] {scenario_label}场景 + 双模式组网方案验证通过")
    return all_ok
