"""Export network plan JSON for scenario x mode combinations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .export_deployment_docs import export_deployment_documents
from .plan_builder import build_network_plan
from ._config import OUTPUT_DIR, OUTPUT_DIR_MAP, NETWORK_MODES, SCENARIO_IDS


def export_plan(
    scenario_id: str,
    network_mode: str,
    verbose: bool = True,
    *,
    placement_seed: int | None = None,
    progress: float = 0.92,
) -> Path:
    out_subdir = OUTPUT_DIR_MAP[(scenario_id, network_mode)]
    out_dir = OUTPUT_DIR / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "network_plan.json"

    plan = build_network_plan(
        scenario_id,
        network_mode,
        placement_seed=placement_seed,
        progress=progress,
    )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    export_deployment_documents(plan, out_dir)

    if verbose:
        print(f"[export_plan] scenario={scenario_id} mode={network_mode}")
        print(f"  output: {out_path}")
        print(f"  residual_nodes_reused: {plan['residual_nodes_reused']}")
        print(f"  emergency_nodes_deployed: {plan['emergency_nodes_deployed']}")
        print(f"  primary_backhaul: {plan['primary_backhaul']}")
        print(f"  comm_modes: {', '.join(plan['comm_modes_used'])}")
        print(f"  phases: {len(plan['phases'])} steps")
        print(f"  rl_source: {plan['rl_enhancement'].get('source', 'unknown')}")

    return out_path


def export_all(verbose: bool = True) -> list[Path]:
    paths = []
    for scenario_id in SCENARIO_IDS:
        for mode in NETWORK_MODES:
            paths.append(export_plan(scenario_id, mode, verbose=verbose))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Export emergency broadcast network plan JSON")
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIO_IDS),
        help="Disaster scenario id",
    )
    parser.add_argument(
        "--mode",
        choices=list(NETWORK_MODES),
        help="Network mode: with_residual or no_residual",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all 4 scenario x mode combinations",
    )
    args = parser.parse_args()

    if args.all:
        export_all()
        return

    if not args.scenario or not args.mode:
        parser.error("Specify --scenario and --mode, or use --all")

    export_plan(args.scenario, args.mode)


if __name__ == "__main__":
    main()
