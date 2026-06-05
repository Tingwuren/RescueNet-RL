"""Validate L1/L2/L3 architecture configuration completeness."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment._config import ARCH_DIR, DEPLOY_DIR, get_comm_mode_ids, load_architecture, load_phased_deploy
from validation._utils import capture_output, print_header, save_proof


def validate_architecture() -> bool:
    print_header("指标一：组网架构设计方案验证")

    arch = load_architecture()
    phased = load_phased_deploy()
    comm_ids = get_comm_mode_ids(arch)

    layer_files = {
        "L1": "l1_global_layer.yaml",
        "L2": "l2_fusion_layer.yaml",
        "L3": "l3_execution_layer.yaml",
    }

    print("\n[三层架构配置加载]")
    for layer, fname in layer_files.items():
        cfg = arch[layer]
        print(f"  {layer} ({cfg.get('layer_name', layer)}): OK")
        print(f"    文件: architecture/{fname}")

    print(f"\n[通信制式] 共 {len(comm_ids)} 类 (要求 >= 4)")
    for cid in comm_ids:
        print(f"  - {cid}")
    comm_ok = len(comm_ids) >= 4

    steps = []
    for phase in phased.get("phases", []):
        steps.extend(phase.get("steps", []))
    expected_steps = int(phased.get("total_steps", 5))
    print(f"\n[组网方案生成流程] 共 {len(steps)} 步 (要求 = {expected_steps})")
    for s in steps:
        print(f"  Step {s.get('step'):2d}: [{s.get('layer')}] {s.get('action')}")

    steps_ok = len(steps) == expected_steps
    files_ok = all((ARCH_DIR / f).exists() for f in layer_files.values()) and (
        ARCH_DIR / "comm_modes.yaml"
    ).exists() and (ARCH_DIR / "overview.md").exists() and (DEPLOY_DIR / "phased_deploy.yaml").exists()

    passed = comm_ok and steps_ok and files_ok
    status = "PASS" if passed else "FAIL"
    print(f"\n[{status}] 组网架构设计方案完整可执行")
    print(f"  制式数量: {len(comm_ids)}/4  生成步骤: {len(steps)}/{expected_steps}  配置文件: {'OK' if files_ok else 'MISSING'}")
    return passed


def main() -> None:
    content = capture_output(validate_architecture)
    print(content, end="")
    save_proof("architecture_check.txt", content)
    sys.exit(0 if "[PASS]" in content else 1)


if __name__ == "__main__":
    main()
