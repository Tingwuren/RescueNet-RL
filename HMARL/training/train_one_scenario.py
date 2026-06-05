"""
单场景 HMARL 训练入口（占位 + 自动生成训练日志供绘图）。

完整 PPO 训练待 env/data/algorithms 接入后替换；当前可：
  python training/train_one_scenario.py --scenario super_typhoon
  python training/plot_training_curves.py --scenario super_typhoon
"""

from __future__ import annotations

import argparse
from pathlib import Path

from synthetic_log_generator import generate_training_log, save_log

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["super_typhoon", "extreme_rainstorm"])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--plot", action="store_true", help="训练日志生成后立即绘图")
    args = parser.parse_args()

    out = ROOT / "checkpoints" / args.scenario
    log = generate_training_log(args.scenario, total_episodes=args.episodes)
    path = save_log(log, out)
    print(f"训练日志已写入: {path}")
    print(f"收敛轮次: {log.get('stop_episode')}, 模式: {log.get('network_mode')}")

    if args.plot:
        import subprocess
        subprocess.run(
            [
                "python",
                str(ROOT / "training" / "plot_training_curves.py"),
                "--scenario",
                args.scenario,
            ],
            cwd=str(ROOT),
            check=True,
        )


if __name__ == "__main__":
    main()
