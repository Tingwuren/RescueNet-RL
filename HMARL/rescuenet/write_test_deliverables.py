"""Write test-run deliverables per 广播网组网架构设计方案.docx (proofs + outputs + tables)."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from rescuenet.bootstrap import HMARL_ROOT

NETWORKING_PLAN_ROOT = HMARL_ROOT / "Networking_plan"
PROOFS_DIR = NETWORKING_PLAN_ROOT / "proofs"
OUTPUTS_DIR = NETWORKING_PLAN_ROOT / "outputs"

SCENARIO_LABELS = {
    "super_typhoon": "超强台风风暴潮",
    "extreme_rainstorm": "极端暴雨",
}

PRIMARY_MODE = {
    "super_typhoon": "with_residual",
    "extreme_rainstorm": "no_residual",
}


def _ensure_networking_path() -> None:
    root = str(NETWORKING_PLAN_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _export_networking_bundle(
    scenario_id: str,
    *,
    run_seed: int,
    progress: float,
) -> Dict[str, Path]:
    """Export network_plan.json + deployment tables for both modes."""
    _ensure_networking_path()
    from deployment._config import OUTPUT_DIR_MAP
    from deployment.export_deployment_docs import export_deployment_documents
    from deployment.export_plan import export_plan

    plans: Dict[str, Path] = {}
    for mode in ("with_residual", "no_residual"):
        mode_seed = int(run_seed) ^ (0x5F3759DF if mode == "with_residual" else 0x1B873593)
        plan_path = export_plan(
            scenario_id,
            mode,
            verbose=False,
            placement_seed=mode_seed,
            progress=float(progress),
        )
        sub = OUTPUT_DIR_MAP[(scenario_id, mode)]
        out_dir = OUTPUTS_DIR / sub
        export_deployment_documents(
            json.loads(plan_path.read_text(encoding="utf-8")),
            out_dir,
        )
        plans[mode] = plan_path
    return plans


def _capture_scenario_validation(scenario_id: str, label: str) -> str:
    _ensure_networking_path()
    from validation.validate_scenario_common import validate_scenario

    buf = StringIO()
    import contextlib

    with contextlib.redirect_stdout(buf):
        validate_scenario(scenario_id, label)
    return buf.getvalue()


def _capture_architecture_check() -> str:
    _ensure_networking_path()
    from validation.validate_architecture import validate_architecture

    buf = StringIO()
    import contextlib

    with contextlib.redirect_stdout(buf):
        validate_architecture()
    return buf.getvalue()


def _build_dual_mode_matrix() -> str:
    _ensure_networking_path()
    from deployment._config import OUTPUT_DIR, OUTPUT_DIR_MAP, SCENARIO_IDS

    lines = [
        "",
        "=" * 70,
        "  指标三：残余/无残余双模式组网验证",
        "=" * 70,
        "",
        f"{'场景':<22}{'模式':<18}{'残余复用':>8}{'应急部署':>10}  {'主回传':<18}{'状态':>6}",
        "-" * 90,
    ]
    all_pass = True
    for scenario_id in SCENARIO_IDS:
        for mode in ("with_residual", "no_residual"):
            sub = OUTPUT_DIR_MAP[(scenario_id, mode)]
            plan_path = OUTPUT_DIR / sub / "network_plan.json"
            with plan_path.open(encoding="utf-8") as handle:
                plan = json.load(handle)
            residual = int(plan.get("residual_nodes_reused", -1))
            emergency = int(plan.get("emergency_nodes_deployed", -1))
            backhaul = str(plan.get("primary_backhaul", ""))
            ok = (residual > 0 if mode == "with_residual" else residual == 0)
            all_pass = all_pass and ok
            status = "PASS" if ok else "FAIL"
            lines.append(
                f"{scenario_id:<22}{mode:<18}{residual:>8}{emergency:>10}  {backhaul:<18}{status:>6}"
            )
    lines.append("")
    lines.append("[PASS] 双模式组网 4/4 组合验证" if all_pass else "[FAIL] 双模式组网存在未通过组合")
    lines.append("")
    return "\n".join(lines)


def _save_proof(name: str, content: str) -> Path:
    PROOFS_DIR.mkdir(parents=True, exist_ok=True)
    path = PROOFS_DIR / name
    path.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")
    return path


def _copy_output_tree_to_deliverables(scenario_id: str, dest: Path) -> List[Path]:
    _ensure_networking_path()
    from deployment._config import OUTPUT_DIR_MAP

    copied: List[Path] = []
    for mode in ("with_residual", "no_residual"):
        sub = OUTPUT_DIR_MAP[(scenario_id, mode)]
        src = OUTPUTS_DIR / sub
        dst = dest / sub
        if dst.exists():
            shutil.rmtree(dst)
        if src.exists():
            shutil.copytree(src, dst)
            copied.append(dst)
    return copied


def _write_test_report(
    path: Path,
    *,
    scenario_alias: str,
    scenario_id: str,
    checkpoint_dir: Path,
    rescuenet_scenario: str,
    run_seed: int,
    rollout: Optional[Dict[str, Any]],
) -> None:
    label = SCENARIO_LABELS.get(scenario_id, scenario_id)
    lines = [
        "HMARL RescueNet 测试验收报告",
        "=" * 72,
        f"生成时间: {datetime.now(timezone.utc).isoformat()}",
        f"场景别名: {scenario_alias} ({label})",
        f"RescueNet 场景: {rescuenet_scenario}",
        f"Checkpoint: {checkpoint_dir}",
        f"run_seed: {run_seed}",
        "",
        "[指标一] 组网架构设计方案",
        f"  归档: Networking_plan/proofs/architecture_check.txt",
        "",
        f"[指标二] {label}场景 + 双模式组网",
        f"  归档: Networking_plan/proofs/scenario_{'rainstorm' if scenario_id == 'extreme_rainstorm' else 'typhoon'}.txt",
        f"  方案目录: Networking_plan/outputs/*_{'rainstorm' if scenario_id == 'extreme_rainstorm' else 'typhoon'}_*/",
        "",
        "[指标三] 双模式矩阵",
        "  归档: Networking_plan/proofs/dual_mode_matrix.txt",
        "",
        "[HMARL 强化学习测试]",
    ]
    if rollout:
        lines.extend(
            [
                f"  episodes: {rollout.get('episodes', 1)}",
                f"  avg_reward: {rollout.get('avg_reward', 0):.4f}",
                f"  avg_final_coverage: {rollout.get('avg_final_coverage', 0):.2%}",
                f"  demo_mode: {rollout.get('demo_mode', False)}",
            ]
        )
    else:
        lines.append("  (已跳过 rollout, --skip-eval)")
    lines.extend(
        [
            "",
            "[交付物清单 — 每种场景×模式各 4 类文件]",
            "  - network_plan.json",
            "  - 01_全局设备分配清单.txt",
            "  - 02_区域资源调度表.txt",
            "  - 03_子区域设备部署明细表.txt",
            "  - 04_设备点位与拓扑连接表.txt",
            "",
            f"[本目录副本] {path.parent / 'networking'}/",
            "[PASS] 测试流程与设计方案归档完成",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_deliverables_index(path: Path, scenario_id: str) -> None:
    proof_name = "scenario_rainstorm.txt" if scenario_id == "extreme_rainstorm" else "scenario_typhoon.txt"
    content = f"""本目录为 test_checkpoint 自动生成的验收交付物索引
场景: {scenario_id}

文件说明:
  test_report.txt              — 本次测试总览
  hmarl_rollout_summary.json   — RL rollout 指标
  networking/                  — 组网方案副本 (2 模式 × 4 文件)
  ../Networking_plan/proofs/{proof_name}  — 场景验证归档
  ../Networking_plan/proofs/architecture_check.txt
  ../Networking_plan/proofs/dual_mode_matrix.txt
"""
    path.write_text(content, encoding="utf-8")


def write_test_deliverables(
    *,
    scenario_alias: str,
    checkpoint_dir: Path,
    rescuenet_scenario: str,
    run_seed: int,
    progress: float = 0.92,
    rollout: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Generate all design-doc deliverables for one test run.
    Returns deliverables root directory under checkpoint_dir.
    """
    from rescuenet.networking_plan_report import _resolve_scenario_id

    scenario_id = _resolve_scenario_id(scenario_alias)
    deliverables = checkpoint_dir / "deliverables"
    deliverables.mkdir(parents=True, exist_ok=True)
    networking_dest = deliverables / "networking"

    _export_networking_bundle(scenario_id, run_seed=run_seed, progress=float(progress))

    scenario_proof = _capture_scenario_validation(
        scenario_id,
        SCENARIO_LABELS.get(scenario_id, scenario_id),
    )
    proof_file = "scenario_rainstorm.txt" if scenario_id == "extreme_rainstorm" else "scenario_typhoon.txt"
    _save_proof(proof_file, scenario_proof)
    _save_proof("architecture_check.txt", _capture_architecture_check())
    _save_proof("dual_mode_matrix.txt", _build_dual_mode_matrix())

    networking_dest.mkdir(parents=True, exist_ok=True)
    (networking_dest / proof_file).write_text(scenario_proof, encoding="utf-8")
    _copy_output_tree_to_deliverables(scenario_id, networking_dest)

    if rollout is not None:
        (deliverables / "hmarl_rollout_summary.json").write_text(
            json.dumps(rollout, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    _write_test_report(
        deliverables / "test_report.txt",
        scenario_alias=scenario_alias,
        scenario_id=scenario_id,
        checkpoint_dir=checkpoint_dir,
        rescuenet_scenario=rescuenet_scenario,
        run_seed=run_seed,
        rollout=rollout,
    )
    _write_deliverables_index(deliverables / "README.txt", scenario_id)

    print("\n" + "=" * 72)
    print("  测试交付物已归档（对齐《广播网组网架构设计方案》）")
    print("=" * 72)
    print(f"  checkpoint 副本: {deliverables.resolve()}")
    print(f"  Networking_plan: {NETWORKING_PLAN_ROOT.resolve()}")
    print(f"    proofs/{proof_file}")
    print("    proofs/architecture_check.txt")
    print("    proofs/dual_mode_matrix.txt")
    _ensure_networking_path()
    from deployment._config import OUTPUT_DIR_MAP  # noqa: WPS433

    for mode in ("with_residual", "no_residual"):
        sub = OUTPUT_DIR_MAP[(scenario_id, mode)]
        print(f"    outputs/{sub}/network_plan.json + 01–04 部署表")
    print("=" * 72 + "\n")

    return deliverables
