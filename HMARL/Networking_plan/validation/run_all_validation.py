"""Run all validations and generate proof artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment._config import OUTPUT_DIR, OUTPUT_DIR_MAP, SCENARIO_IDS, NETWORK_MODES
from deployment.export_plan import export_all
from validation._utils import capture_output, print_header, save_proof
from validation.validate_architecture import validate_architecture
from validation.validate_scenario_common import validate_scenario


def validate_output_deliverables() -> bool:
    """Each scenario×mode directory must contain JSON + 01–04 deployment tables."""
    from deployment._config import OUTPUT_DIR, OUTPUT_DIR_MAP, SCENARIO_IDS, NETWORK_MODES

    required = [
        "network_plan.json",
        "01_全局设备分配清单.txt",
        "02_区域资源调度表.txt",
        "03_子区域设备部署明细表.txt",
        "04_设备点位与拓扑连接表.txt",
    ]
    print_header("交付物完整性验证（JSON + 01–04 表）")
    all_ok = True
    for scenario_id in SCENARIO_IDS:
        for mode in NETWORK_MODES:
            sub = OUTPUT_DIR_MAP[(scenario_id, mode)]
            out_dir = OUTPUT_DIR / sub
            missing = [name for name in required if not (out_dir / name).exists()]
            ok = not missing
            all_ok = all_ok and ok
            status = "PASS" if ok else "FAIL"
            suffix = "" if ok else f" missing: {', '.join(missing)}"
            print(f"  {sub:28s} [{status}]{suffix}")
    final = "PASS" if all_ok else "FAIL"
    print(f"\n[{final}] 4 套 outputs 交付物完整性验证")
    return all_ok


def validate_dual_mode_matrix() -> bool:
    print_header("指标三：残余/无残余双模式组网验证")

    print("\n场景                  模式              残余复用  应急部署  主回传              状态")
    print("-" * 90)

    all_ok = True
    rows = []

    for scenario_id in SCENARIO_IDS:
        for mode in NETWORK_MODES:
            sub = OUTPUT_DIR_MAP[(scenario_id, mode)]
            plan_path = OUTPUT_DIR / sub / "network_plan.json"
            if not plan_path.exists():
                export_all(verbose=False)

            with plan_path.open("r", encoding="utf-8") as f:
                plan = json.load(f)

            residual = plan.get("residual_nodes_reused", 0)
            emergency = plan.get("emergency_nodes_deployed", 0)
            backhaul = plan.get("primary_backhaul", "")

            if mode == "with_residual":
                ok = residual > 0
            else:
                ok = residual == 0 and emergency > 0

            status = "PASS" if ok else "FAIL"
            all_ok = all_ok and ok
            row = f"{scenario_id:20s}  {mode:16s}  {residual:8d}  {emergency:8d}  {backhaul:16s}  {status}"
            print(row)
            rows.append(row)

    final = "PASS" if all_ok else "FAIL"
    print(f"\n[{final}] 双模式组网 4/4 组合验证")
    return all_ok


def run_all() -> bool:
    print_header("应急广播网组网架构指标 — 一键验证")

    results = {}

    arch_content = capture_output(validate_architecture)
    print(arch_content, end="")
    save_proof("architecture_check.txt", arch_content)
    results["architecture"] = "[PASS]" in arch_content

    rain_content = capture_output(lambda: validate_scenario("extreme_rainstorm", "极端暴雨"))
    print(rain_content, end="")
    save_proof("scenario_rainstorm.txt", rain_content)
    results["rainstorm"] = "[PASS]" in rain_content

    typhoon_content = capture_output(lambda: validate_scenario("super_typhoon", "超强台风"))
    print(typhoon_content, end="")
    save_proof("scenario_typhoon.txt", typhoon_content)
    results["typhoon"] = "[PASS]" in typhoon_content

    dual_content = capture_output(validate_dual_mode_matrix)
    print(dual_content, end="")
    save_proof("dual_mode_matrix.txt", dual_content)
    results["dual_mode"] = "[PASS]" in dual_content

    deliverables_content = capture_output(validate_output_deliverables)
    print(deliverables_content, end="")
    save_proof("deliverables_check.txt", deliverables_content)
    results["deliverables"] = "[PASS]" in deliverables_content

    print_header("验证汇总")
    for name, ok in results.items():
        print(f"  {name:15s}: {'PASS' if ok else 'FAIL'}")

    all_pass = all(results.values())
    summary = "\n".join(
        [
            "应急广播网组网架构指标验证汇总",
            "=" * 40,
            *(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in results.items()),
            "",
            f"总体: {'ALL PASS (4/4)' if all_pass else 'SOME FAILED'}",
        ]
    )
    save_proof("verification_summary.txt", summary)
    print(f"\n总体: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"证明归档目录: {ROOT / 'proofs'}")
    return all_pass


def main() -> None:
    ok = run_all()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
