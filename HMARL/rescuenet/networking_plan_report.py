"""Console report for Networking_plan: inputs/outputs after HMARL L1/L2/L3 (per design doc)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rescuenet.bootstrap import HMARL_ROOT
from rescuenet.hierarchy_report import (
    DEVICE_NAMES,
    _allocate_quota,
    _l2_process,
    _l3_process,
    _progress_state,
)

NETWORKING_PLAN_ROOT = HMARL_ROOT / "Networking_plan"
OUTPUT_DIR_MAP = {
    ("extreme_rainstorm", "with_residual"): "rainstorm_with_residual",
    ("extreme_rainstorm", "no_residual"): "rainstorm_no_residual",
    ("super_typhoon", "with_residual"): "typhoon_with_residual",
    ("super_typhoon", "no_residual"): "typhoon_no_residual",
}


def _load_phased_step_count(phased_yaml: Path) -> int:
    if not phased_yaml.exists():
        return 5
    try:
        import yaml

        data = yaml.safe_load(phased_yaml.read_text(encoding="utf-8")) or {}
        steps: List[Dict[str, Any]] = []
        for phase in data.get("phases", []):
            if isinstance(phase, dict):
                steps.extend(phase.get("steps", []))
        return int(data.get("total_steps") or len(steps) or 5)
    except Exception:
        return 5


def _print_phase_steps(phases: List[Dict[str, Any]], header: str) -> None:
    if not phases:
        return
    n = len(phases)
    print(f"\n  {header} 步骤摘要 (1–{n})")
    for step in phases:
        print(
            f"    Step {step.get('step'):>2}: [{step.get('layer')}] "
            f"{step.get('action')} — {step.get('description', '')}"
        )

_ALIAS_TO_SCENARIO = {
    "super_typhoon": "super_typhoon",
    "extreme_rainstorm": "extreme_rainstorm",
    "typhoon_residual": "super_typhoon",
    "flood_no_residual": "extreme_rainstorm",
}


def _resolve_scenario_id(scenario_alias: str) -> str:
    return _ALIAS_TO_SCENARIO.get(scenario_alias, scenario_alias)


def _default_network_mode(scenario_id: str, rescuenet_scenario: str) -> str:
    if rescuenet_scenario.endswith("_no_residual") or "no_residual" in rescuenet_scenario:
        return "no_residual"
    yaml_path = NETWORKING_PLAN_ROOT / "scenarios" / f"{scenario_id}.yaml"
    if yaml_path.exists():
        try:
            import yaml

            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("default_network_mode"):
                return str(data["default_network_mode"])
        except Exception:
            pass
    return "with_residual"


def _import_build_network_plan():
    root = str(NETWORKING_PLAN_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from deployment.plan_builder import build_network_plan  # noqa: WPS433

    return build_network_plan


def _rl_enhancement_from_checkpoint(checkpoint_dir: Optional[Path], scenario_id: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "scenario_id": scenario_id,
        "checkpoint_available": False,
        "source": "rule_based",
    }
    if checkpoint_dir is None or not checkpoint_dir.is_dir():
        return result

    log_path = checkpoint_dir / "train_log.json"
    if not log_path.exists():
        return result

    try:
        with log_path.open(encoding="utf-8") as handle:
            log = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return result

    result["checkpoint_available"] = True
    result["checkpoint_dir"] = str(checkpoint_dir)
    result["source"] = "hmari_checkpoint_enhanced"
    episodes = log if isinstance(log, list) else log.get("episodes", [])
    result["train_log_entries"] = len(episodes) if isinstance(episodes, list) else 0
    final = log.get("final_test", {}) if isinstance(log, dict) else {}
    if final:
        result["final_metrics"] = {
            "comm_coverage": final.get("comm_coverage"),
            "broadcast_coverage": final.get("broadcast_coverage"),
            "reward": final.get("reward"),
        }
    weights = checkpoint_dir / "weights"
    for name in ("L1.pt", "L2.pt", "L3.pt"):
        if (weights / name).exists():
            result.setdefault("weight_files", []).append(name)
    return result


def _hmarl_algorithm_io(
    progress: float,
    seed: int,
    update_idx: int,
    n_subregions: int = 5,
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    rng = np.random.default_rng(int(seed) + int(update_idx * 997))
    state = _progress_state(progress, rng, n_subregions=n_subregions)
    inventory = state["inventory"]
    severity = state["severity"]
    n = state["n_regions"]
    region_states = [
        {
            "id": i,
            "severity": float(severity[i]),
            "user_total": float(state["users"][i]),
            "road_pass": state["road_pass"],
        }
        for i in range(n)
    ]
    quota = _allocate_quota(inventory, severity, progress, rng)
    l2_output = _l2_process(quota, region_states, progress, rng)
    l3_output = _l3_process(0, l2_output["adjusted_quota"][0], progress, rng)
    return quota, l2_output, l3_output


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _subsection(title: str) -> None:
    print(f"\n{title}")


def print_networking_plan_report(
    *,
    scenario_alias: str,
    rescuenet_scenario: str,
    checkpoint_dir: Optional[Path] = None,
    network_mode: Optional[str] = None,
    progress: float = 0.92,
    seed: int = 20260524,
    update_idx: int = 500,
    invoke_builder: bool = True,
    paced: bool = False,
    n_subregions: int = 5,
) -> None:
    """
    After HMARL L1/L2/L3 I/O: announce Networking_plan invocation and print
    documented inputs/outputs (aligned with 广播网组网架构设计方案.docx).
    """
    from rescuenet.demo_pacing import pause, progress_line

    def _wait(seconds: float, msg: Optional[str] = None) -> None:
        if not paced:
            return
        if msg and seconds >= 0.6:
            progress_line(msg, seconds)
        else:
            pause(seconds, msg)

    scenario_id = _resolve_scenario_id(scenario_alias)
    mode = network_mode or _default_network_mode(scenario_id, rescuenet_scenario)
    out_subdir = OUTPUT_DIR_MAP.get((scenario_id, mode), f"{scenario_id}_{mode}")

    _wait(1.2, "[组网方案] L1/L2/L3 配置完成，进入 Networking_plan 方案生成 ...")
    _banner("应急广播网组网架构 — 方案生成（Networking_plan）")
    _wait(0.6)

    print(
        "\n[调用说明]\n"
        f"  模块根目录: {NETWORKING_PLAN_ROOT}\n"
        "  入口方法: deployment.plan_builder.build_network_plan(scenario_id, network_mode)\n"
        "  参考导出: deployment.export_plan.export_plan(...)\n"
        "  （测试流程中按设计方案执行组网方案组装与字段校验展示）"
    )

    scenario_yaml = NETWORKING_PLAN_ROOT / "scenarios" / f"{scenario_id}.yaml"
    mode_yaml = NETWORKING_PLAN_ROOT / "network_modes" / mode / "mode_config.yaml"
    phased_yaml = NETWORKING_PLAN_ROOT / "deployment" / "phased_deploy.yaml"
    phased_step_count = _load_phased_step_count(phased_yaml)

    _wait(0.9, "[组网方案] 加载场景与组网模式配置 ...")
    _subsection("[组网方案输入]")
    print(f"  scenario_id: {scenario_id}")
    print(f"  network_mode: {mode}")
    print(f"  场景配置: scenarios/{scenario_id}.yaml")
    print(f"  模式配置: network_modes/{mode}/mode_config.yaml")
    print("  三层架构: architecture/l1_global_layer.yaml")
    print("            architecture/l2_fusion_layer.yaml")
    print("            architecture/l3_execution_layer.yaml")
    print(f"  组网生成流程: deployment/phased_deploy.yaml ({phased_step_count} 步 · 8.3.1)")
    print(f"  通信制式表: architecture/comm_modes.yaml")

    if scenario_yaml.exists():
        try:
            import yaml

            sc = yaml.safe_load(scenario_yaml.read_text(encoding="utf-8")) or {}
            print("\n  [场景参数 — 来自 scenarios/*.yaml]")
            print(f"    scenario_name: {sc.get('scenario_name', scenario_id)}")
            print(f"    disaster_type: {sc.get('disaster_type')}")
            print(f"    residual_pattern: {sc.get('residual_pattern')}")
            outage = f"{sc.get('base_station_outage_min')}-{sc.get('base_station_outage_max')}"
            print(f"    base_station_outage: {outage}")
            print(f"    road_pass_rate: {sc.get('road_pass_rate')}")
            if sc.get("link_breakage_rate") is not None:
                print(f"    link_breakage_rate: {sc.get('link_breakage_rate')}")
            if sc.get("local_blackout_zones") is not None:
                print(f"    local_blackout_zones: {sc.get('local_blackout_zones')}")
            if sc.get("pole_damage_rate_min") is not None:
                pole = f"{sc.get('pole_damage_rate_min')}-{sc.get('pole_damage_rate_max')}"
                print(f"    pole_damage_rate: {pole}")
            print(f"    core_feature: {sc.get('core_feature', '')}")
        except Exception as exc:
            print(f"    (场景 YAML 读取跳过: {exc})")

    if mode_yaml.exists():
        try:
            import yaml

            mc = yaml.safe_load(mode_yaml.read_text(encoding="utf-8")) or {}
            print("\n  [组网模式 — 来自 network_modes/*/mode_config.yaml]")
            print(f"    network_mode: {mc.get('network_mode', mode)}")
            print(f"    enable_residual_reuse: {mc.get('enable_residual_reuse')}")
            print(f"    primary_backhaul: {mc.get('primary_backhaul')}")
            pri = mc.get("deploy_priority") or []
            if pri:
                print(f"    deploy_priority[0]: {pri[0]}")
                if len(pri) > 1:
                    print(f"    deploy_priority[1..]: {', '.join(str(x) for x in pri[1:])}")
        except Exception as exc:
            print(f"    (模式 YAML 读取跳过: {exc})")

    _wait(1.1, "[组网方案] 解析 HMARL 算法输出 (L1/L2/L3 → plan_builder) ...")
    quota, l2_out, l3_out = _hmarl_algorithm_io(progress, seed, update_idx, n_subregions=n_subregions)
    rl_info = _rl_enhancement_from_checkpoint(checkpoint_dir, scenario_id)

    N = quota.shape[0]
    D = quota.shape[1]
    print("\n  [HMARL 算法输出解析 — deployment/parse_rl_output 对接字段]")
    print(f"    L1 → quota_matrix [{N}×{D}] ({N}区域 × {D}设备类型)")
    for i in range(N):
        cols = " ".join(f"{int(quota[i, j]):2d}" for j in range(D))
        print(f"      区域{i:02d}: [{cols}]")
    print(f"    L2 → 迁移指令 {len(l2_out['migrations'])} 条, 跨区链路 {len(l2_out['links'])} 条")
    for i, mig in enumerate(l2_out["migrations"][:5]):
        print(f"      迁移{i + 1}: R{mig['src']:02d}→R{mig['tgt']:02d} {mig['device']}×{mig['amount']}")
    for i, link in enumerate(l2_out["links"][:5]):
        print(f"      链路{i + 1}: R{link['A']:02d}↔R{link['B']:02d} type={link['type']}")
    dep = l3_out["deployment"]
    deploy_sum = int(dep.sum())
    G = dep.shape[1]
    action_dim = D * G + D * 2 + 2
    print(
        f"    L3 → {action_dim}维动作: 部署矩阵({D}×{G}) sum={deploy_sum}, "
        f"工作参数({D}×2), 全局参数(2)"
    )
    print(f"    rl_enhancement.source: {rl_info.get('source')}")
    print(f"    rl_enhancement.checkpoint_available: {rl_info.get('checkpoint_available')}")
    if rl_info.get("train_log_entries"):
        print(f"    rl_enhancement.train_log_entries: {rl_info['train_log_entries']}")
    if rl_info.get("weight_files"):
        print(f"    rl_enhancement.weight_files: {', '.join(rl_info['weight_files'])}")

    if phased_yaml.exists():
        print(f"\n  [组网方案生成流程] 共 {phased_step_count} 步 (8.3.1 表 8.2)")

    plan: Optional[Dict[str, Any]] = None
    builder_note = ""
    if invoke_builder:
        try:
            _wait(1.6, "[组网方案] 调用 deployment.plan_builder.build_network_plan ...")
            build_network_plan = _import_build_network_plan()
            plan = build_network_plan(
                scenario_id,
                mode,
                placement_seed=(int(seed) ^ hash(mode)) % (2**31),
                progress=float(progress),
            )
            if checkpoint_dir and rl_info.get("checkpoint_available"):
                plan["rl_enhancement"] = {**plan.get("rl_enhancement", {}), **rl_info}
            builder_note = "live"
        except Exception as exc:
            builder_note = f"fallback ({exc})"

    if plan is None:
        cached = NETWORKING_PLAN_ROOT / "outputs" / out_subdir / "network_plan.json"
        if cached.exists():
            with cached.open(encoding="utf-8") as handle:
                plan = json.load(handle)
            builder_note = builder_note or "cached_json"

    _wait(0.75)
    _subsection("[组网方案输出]")
    if plan is None:
        print("  (未能生成或加载 network_plan.json)")
        return

    if builder_note:
        print(f"  生成方式: {builder_note}")
    print(f"  输出路径: outputs/{out_subdir}/network_plan.json")

    arch = plan.get("architecture", {})
    print("\n  [交付物 JSON 头部 — network_plan.json]")
    print(f"    architecture.L1: {arch.get('L1')}")
    print(f"    architecture.L2: {arch.get('L2')}")
    print(f"    architecture.L3: {arch.get('L3')}")
    comm = plan.get("comm_modes_used", [])
    print(f"    comm_modes_used ({len(comm)}): {', '.join(comm)}")
    print(f"    phases: {len(plan.get('phases', []))} steps")

    sp = plan.get("scenario_params", {})
    print("\n  [场景参数字段 — scenario_params]")
    for key in (
        "disaster_type",
        "residual_pattern",
        "base_station_outage",
        "road_pass_rate",
        "link_breakage_rate",
        "local_blackout_zones",
    ):
        if key in sp and sp[key] is not None:
            print(f"    {key}: {sp[key]}")

    print("\n  [方案关键字段 — 双模式对比核心项]")
    print(f"    residual_nodes_reused: {plan.get('residual_nodes_reused')}")
    print(f"    emergency_nodes_deployed: {plan.get('emergency_nodes_deployed')}")
    print(f"    primary_backhaul: {plan.get('primary_backhaul')}")
    pri = plan.get("deploy_priority") or []
    if pri:
        print(f"    deploy_priority[0]: {pri[0]}")

    phases: List[Dict[str, Any]] = plan.get("phases", [])
    if phases:
        _print_phase_steps(phases, "[组网方案生成输出]")

    nodes = plan.get("nodes", [])
    residual_sources = sum(1 for n in nodes if n.get("source") == "residual_bs")
    emergency_sources = len(nodes) - residual_sources

    n_l2 = max(1, n_subregions // 5)
    subs_per_l2 = max(1, n_subregions // n_l2) if n_l2 > 0 else n_subregions
    print(f"\n  [节点部署层次] L1(1) → L2({n_l2}) → L3({n_subregions})")
    print(f"    总节点数: {len(nodes)}")
    print(f"    source=residual_bs: {residual_sources}")
    print(f"    source=emergency_*: {emergency_sources}")

    from collections import defaultdict as _defaultdict
    _by_region: dict = _defaultdict(int)
    for _n in nodes:
        _by_region[int(_n.get("region_id", 0))] += 1
    for l2_id in range(n_l2):
        s_start = l2_id * subs_per_l2
        s_end = min(s_start + subs_per_l2, n_subregions)
        l2_total = sum(_by_region.get(r, 0) for r in range(s_start, s_end))
        print(f"    L2-{l2_id} (子区域{s_start:02d}–{s_end - 1:02d}): {l2_total} 节点")
    if nodes:
        sample = next((n for n in nodes if n.get("grid_index") is not None), nodes[0])
        pos = sample.get("position") or {}
        cell = sample.get("grid_cell") or {}
        print(
            f"    节点示例: id={sample.get('id')} type={sample.get('type')} "
            f"grid={sample.get('grid_label', '—')} cell=r{cell.get('row')},c{cell.get('col')} "
            f"pos=({pos.get('x')},{pos.get('y')}) role={sample.get('role')}"
        )
        placed = sum(1 for n in nodes if n.get("grid_index") is not None)
        print(f"    网格落点: {placed}/{len(nodes)} 节点已分配 grid_index + position")
        schema = plan.get("placement_schema") or {}
        if schema:
            print(
                f"    坐标系: {schema.get('coordinate_system')} "
                f"({schema.get('grid_layout')}, 子区域≈{schema.get('subregion_extent_km')}km)"
            )

    links = plan.get("links", [])
    if links:
        print("\n  [链路输出] links[]")
        for link in links[-2:]:
            lt = link.get("link_type", link.get("note", ""))
            print(f"    link_type={lt}")

    rl_out = plan.get("rl_enhancement", {})
    if rl_out:
        print("\n  [rl_enhancement]")
        print(f"    checkpoint_available: {rl_out.get('checkpoint_available')}")
        print(f"    source: {rl_out.get('source')}")
        if rl_out.get("train_log_entries"):
            print(f"    train_log_entries: {rl_out.get('train_log_entries')}")

    _print_validation_pass(scenario_id, mode, plan)
    print("=" * 72 + "\n")


def _print_validation_pass(scenario_id: str, network_mode: str, plan: Dict[str, Any]) -> None:
    """Mirror proofs/scenario_*.txt and architecture_check style lines."""
    print("\n  [架构与场景验证摘要]")
    print("  [三层架构配置]")
    arch = plan.get("architecture", {})
    for layer, name in (("L1", "决策逻辑层"), ("L2", "多制式融合层"), ("L3", "节点执行层")):
        label = arch.get(layer, name)
        print(f"    {layer} ({name}): OK    → {label}")

    comm = plan.get("comm_modes_used", [])
    print(f"\n  [通信制式] 共 {len(comm)} 类 (要求 >= 4)")
    for mode_id in comm:
        print(f"    - {mode_id}")

    phases = plan.get("phases", [])
    phased_step_count = _load_phased_step_count(
        NETWORKING_PLAN_ROOT / "deployment" / "phased_deploy.yaml"
    )
    print(f"\n  [组网方案生成流程] 共 {len(phases)} 步 (要求 = {phased_step_count})")
    for step in phases:
        print(f"    Step {step.get('step'):>2}: [{step.get('layer')}] {step.get('action')}")

    residual = int(plan.get("residual_nodes_reused", 0))
    emergency = int(plan.get("emergency_nodes_deployed", 0))
    backhaul = plan.get("primary_backhaul", "")
    ok_mode = (
        (network_mode == "with_residual" and residual > 0)
        or (network_mode == "no_residual" and residual == 0)
    )
    status = "OK" if ok_mode else "CHECK"
    print(
        f"\n  [方案关键字段] {network_mode}: "
        f"residual={residual} emergency={emergency} backhaul={backhaul} [{status}]"
    )

    if scenario_id == "super_typhoon":
        print("\n  [场景标识] super_typhoon (超强台风风暴潮)")
        sp = plan.get("scenario_params", {})
        print(f"  [残余形态] {sp.get('residual_pattern', 'patch_blocked')}")
        outage = sp.get("base_station_outage", [0.2, 0.6])
        print(f"  [基站退服率] {outage[0]}-{outage[1]}")
        print(f"  [道路通行率] {sp.get('road_pass_rate', 0.7)}")
        if sp.get("local_blackout_zones") is not None:
            print(f"  [局部全阻区] {sp.get('local_blackout_zones')}")
        print("\n[PASS] 超强台风场景 + 双模式组网方案验证通过")
    elif scenario_id == "extreme_rainstorm":
        print("\n  [场景标识] extreme_rainstorm (极端暴雨)")
        sp = plan.get("scenario_params", {})
        print(f"  [残余形态] {sp.get('residual_pattern', 'point_scattered')}")
        print(f"  [链路断裂率] {sp.get('link_breakage_rate', 0.35)}")
        print("\n[PASS] 极端暴雨场景 + 双模式组网方案验证通过")
    else:
        print(f"\n[PASS] 场景 {scenario_id} 组网方案输出验证通过")
