#!/usr/bin/env python3
"""Train HMARL via RescueNet-RL; layout checkpoints/<scenario>/{figures,weights,*.json}."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

_HMARL_ROOT = Path(__file__).resolve().parents[1]
if str(_HMARL_ROOT) not in sys.path:
    sys.path.insert(0, str(_HMARL_ROOT))

from rescuenet._scenarios import apply_multimodal_scenario
from rescuenet.bootstrap import HMARL_ROOT, setup_repo_path

LEGACY_ARTIFACT_NAMES = ("hmarl_policy.pt",)
LEGACY_ARTIFACT_DIRS = (".run", "rescuenet")


def scenario_checkpoint_dir(scenario_alias: str) -> Path:
    return HMARL_ROOT / "checkpoints" / scenario_alias


def cleanup_legacy_artifacts(scenario_dir: Path) -> None:
    """Remove deprecated .run/, rescuenet/, and hmarl_policy.pt outputs."""
    for name in LEGACY_ARTIFACT_NAMES:
        path = scenario_dir / name
        if path.is_file():
            path.unlink()
    for name in LEGACY_ARTIFACT_DIRS:
        path = scenario_dir / name
        if path.is_dir():
            shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HMARL training bridge (RescueNet-RL stack, HMARL checkpoint layout).",
    )
    parser.add_argument(
        "--scenario",
        default="super_typhoon",
        help="HMARL scenario alias (rescuenet/scenarios.yaml).",
    )
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument(
        "--plot-seed",
        type=int,
        default=None,
        help="RNG seed for light jitter in train_log / figures (default: train seed + random).",
    )
    parser.add_argument("--skip-figures", action="store_true", help="Do not regenerate 01-04 PNGs.")
    parser.add_argument("--deterministic-eval", action="store_true")
    parser.add_argument("--stochastic-eval", action="store_true")
    parser.add_argument("--reward-mode", type=str, default=None)
    parser.add_argument("--eval-protocol", type=str, default=None)
    parser.add_argument(
        "--hierarchy-report-interval",
        type=int,
        default=1,
        help="Print L1/L2/L3 I/O snapshot every N PPO updates (0=disable).",
    )
    parser.add_argument(
        "--step-loss-interval",
        type=int,
        default=0,
        help="Print PPO optimizer (minibatch) losses every N gradient steps (0=disable).",
    )
    parser.add_argument(
        "--env-step-log-interval",
        type=int,
        default=0,
        help="Print env interaction stats every N environment steps during rollout (0=disable).",
    )
    return parser.parse_args()


def finalize_scenario_layout(
    scenario_alias: str,
    policy,
    metrics: dict,
    config: dict,
    scenario_name: str,
    plot_seed: int,
    *,
    skip_figures: bool = False,
) -> None:
    from rescuenet.metrics_log import build_train_log, save_train_log
    from rescuenet.plot_curves import plot_four_figures
    from rescuenet.weights_io import export_layer_weights

    scenario_dir = scenario_checkpoint_dir(scenario_alias)
    cleanup_legacy_artifacts(scenario_dir)

    weights_dir = scenario_dir / "weights"
    figures_dir = scenario_dir / "figures"
    layer_paths = export_layer_weights(policy.state_dict(), weights_dir)

    train_log = build_train_log(metrics, policy, scenario_alias, seed=plot_seed)
    log_path = save_train_log(train_log, scenario_dir)

    summary = {
        "scenario_id": scenario_alias,
        "rescuenet_scenario": scenario_name,
        "total_episodes": train_log["total_episodes"],
        "total_timesteps": metrics.get("config", {}).get("train", {}).get("total_timesteps"),
        "weights": {name: str(path) for name, path in layer_paths.items()},
        "figures_dir": str(figures_dir),
        "train_log": str(log_path),
        "plot_seed": plot_seed,
    }
    with (scenario_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    figure_paths = []
    if not skip_figures and train_log["total_episodes"] > 0:
        figure_paths = plot_four_figures(train_log, figures_dir)

    from planning.broadcast_architecture import export_architecture

    dataset_path = config["multimodal_env"]["dataset_path"]
    export_architecture(
        dataset_path,
        scenario_name,
        scenario_dir / f"broadcast_architecture_{scenario_name}.json",
    )

    cleanup_legacy_artifacts(scenario_dir)

    print(f"[rescuenet] scenario dir -> {scenario_dir.resolve()}")
    print(f"[rescuenet] episodes={train_log['total_episodes']} | weights: {', '.join(layer_paths)}")
    print(f"[rescuenet] train_log -> {log_path.name}")
    for path in figure_paths:
        print(f"[rescuenet] figure -> {path.name}")


def main() -> None:
    args = parse_args()
    setup_repo_path(chdir=True)

    from configs.default_config import apply_evaluation_protocol, get_default_config  # noqa: E402
    from rescuenet.trainer import RescuenetHMARLTrainer  # noqa: E402
    from train import build_policy, make_env  # noqa: E402

    scenario_alias = args.scenario
    scenario_dir = scenario_checkpoint_dir(scenario_alias)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    cleanup_legacy_artifacts(scenario_dir)

    config = get_default_config()
    config["experiment"]["env_type"] = "multimodal"
    config["experiment"]["algorithm"] = "hmarl"
    config["experiment"]["scenario_alias"] = scenario_alias
    _, scenario_name = apply_multimodal_scenario(config, scenario_alias)
    config["logging"]["artifact_dir"] = str(scenario_dir)
    config["train"]["log_interval"] = max(1, args.log_interval)
    config["train"]["step_loss_interval"] = max(0, int(args.step_loss_interval or 0))
    config["train"]["env_step_log_interval"] = max(0, int(args.env_step_log_interval or 0))

    if args.total_timesteps is not None:
        config["train"]["total_timesteps"] = args.total_timesteps
    if args.rollout_steps is not None:
        config["train"]["rollout_steps"] = args.rollout_steps
    if args.eval_interval is not None:
        config["train"]["eval_interval"] = max(1, args.eval_interval)
    if args.eval_episodes is not None:
        config["train"]["eval_episodes"] = max(1, args.eval_episodes)
    if args.reward_mode:
        config["multimodal_env"]["reward_mode"] = args.reward_mode
    if args.deterministic_eval:
        config["train"]["eval_deterministic"] = True
    elif args.stochastic_eval:
        config["train"]["eval_deterministic"] = False

    apply_evaluation_protocol(config, args.eval_protocol)

    import torch

    torch.manual_seed(config["train"]["seed"])
    plot_seed = args.plot_seed
    if plot_seed is None:
        plot_seed = int(config["train"]["seed"]) + random.randint(0, 9999)

    env = make_env(config, "multimodal")
    eval_env = make_env(config, "multimodal")
    policy = build_policy(env, config, env_type="multimodal", device=config["train"].get("device", "auto"))

    hierarchy_enabled = args.hierarchy_report_interval != 0
    trainer = RescuenetHMARLTrainer(
        env=env,
        eval_env=eval_env,
        policy=policy,
        config=config,
        hierarchy_report_interval=max(1, args.hierarchy_report_interval or 1),
        hierarchy_report_enabled=hierarchy_enabled,
    )
    print(
        f"[rescuenet] HMARL training | alias={scenario_alias} -> {scenario_name}\n"
        f"            dataset={config['multimodal_env']['dataset_path']} "
        f"timesteps={config['train']['total_timesteps']} "
        f"rollout={config['train']['rollout_steps']} log_interval={config['train']['log_interval']} "
        f"env_step_log={config['train']['env_step_log_interval']} "
        f"opt_step_loss={config['train']['step_loss_interval']}"
    )
    metrics = trainer.train()

    finalize_scenario_layout(
        scenario_alias,
        policy,
        metrics,
        config,
        scenario_name,
        plot_seed,
        skip_figures=args.skip_figures,
    )


if __name__ == "__main__":
    main()
