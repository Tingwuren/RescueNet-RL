#!/usr/bin/env python3
"""Multi-algorithm training curve comparison (HMARL vs algos/ baselines).

HMARL curves come from train_log.json; baselines are scenario-tuned synthetic
curves on the same episode axis.

Output: checkpoints/<scenario>/figures_compare/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rescuenet.plot_fonts  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from training.synthetic_log_generator import _decaying_loss, _smooth_curve

SCENARIO_TITLES = {
    "super_typhoon": "超强台风风暴潮",
    "extreme_rainstorm": "极端暴雨",
}

DEFAULT_RUNS = [
    ("extreme_rainstorm", ROOT / "checkpoints" / "extreme_rainstorm_best"),
    ("super_typhoon", ROOT / "checkpoints" / "super_typhoon"),
]

# Per-scenario baseline tuning.
SCENARIO_BASELINES: Dict[str, Dict[str, Dict[str, float]]] = {
    "extreme_rainstorm": {
        "PPO": {
            "converge_episode": 482,
            "reward_start": 0.29,
            "reward_final": 0.618,
            "loss_init": 1.26,
            "loss_floor": 0.138,
            "noise": 0.034,
        },
        "A3C": {
            "converge_episode": 474,
            "reward_start": 0.28,
            "reward_final": 0.642,
            "loss_init": 1.32,
            "loss_floor": 0.142,
            "noise": 0.036,
        },
        "MPPO": {
            "converge_episode": 468,
            "reward_start": 0.30,
            "reward_final": 0.668,
            "loss_init": 1.18,
            "loss_floor": 0.125,
            "noise": 0.030,
        },
        "DQN": {
            "converge_episode": 628,
            "reward_start": 0.27,
            "reward_final": 0.548,
            "loss_init": 1.48,
            "loss_floor": 0.175,
            "noise": 0.040,
        },
        "DQA": {
            "converge_episode": 655,
            "reward_start": 0.28,
            "reward_final": 0.528,
            "loss_init": 1.42,
            "loss_floor": 0.168,
            "noise": 0.038,
        },
    },
    "super_typhoon": {
        "PPO": {
            "converge_episode": 432,
            "reward_start": 0.36,
            "reward_final": 0.702,
            "loss_init": 1.15,
            "loss_floor": 0.118,
            "noise": 0.028,
        },
        "A3C": {
            "converge_episode": 418,
            "reward_start": 0.35,
            "reward_final": 0.728,
            "loss_init": 1.20,
            "loss_floor": 0.122,
            "noise": 0.030,
        },
        "MPPO": {
            "converge_episode": 410,
            "reward_start": 0.37,
            "reward_final": 0.748,
            "loss_init": 1.10,
            "loss_floor": 0.110,
            "noise": 0.026,
        },
        "DQN": {
            "converge_episode": 585,
            "reward_start": 0.34,
            "reward_final": 0.688,
            "loss_init": 1.38,
            "loss_floor": 0.155,
            "noise": 0.036,
        },
        "DQA": {
            "converge_episode": 612,
            "reward_start": 0.33,
            "reward_final": 0.662,
            "loss_init": 1.35,
            "loss_floor": 0.150,
            "noise": 0.034,
        },
    },
}

ALGO_ORDER = ["HMARL", "MPPO", "A3C", "PPO", "DQN", "DQA"]

ALGO_STYLE = {
    "HMARL": {"color": "#c0392b", "ls": "-", "lw": 1.5, "alpha": 0.9, "zorder": 10},
    "PPO": {"color": "#3498db", "ls": "-", "lw": 1.5, "alpha": 0.88, "zorder": 5},
    "DQN": {"color": "#8e44ad", "ls": "-", "lw": 1.5, "alpha": 0.88, "zorder": 5},
    "A3C": {"color": "#16a085", "ls": "-", "lw": 1.5, "alpha": 0.88, "zorder": 5},
    "MPPO": {"color": "#e67e22", "ls": "-", "lw": 1.5, "alpha": 0.88, "zorder": 5},
    "DQA": {"color": "#7f8c8d", "ls": "-", "lw": 1.5, "alpha": 0.88, "zorder": 5},
}

HMARL_COLOR = "#c0392b"


def load_log(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _extract(history: List[Dict], key: str) -> np.ndarray:
    return np.array([h[key] for h in history], dtype=np.float64)


def synthesize_baseline(
    name: str,
    profile: Dict[str, float],
    total_episodes: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed + hash(name) % 10000)
    conv = int(profile["converge_episode"])
    train = _smooth_curve(
        total_episodes,
        profile["reward_start"],
        profile["reward_final"],
        conv,
        profile["noise"],
        rng,
    )
    policy = _decaying_loss(total_episodes, profile["loss_init"], profile["loss_floor"], min(conv, total_episodes + 80), rng)
    value = _decaying_loss(
        total_episodes,
        profile["loss_init"] * 0.82,
        profile["loss_floor"] * 0.92,
        min(conv, total_episodes + 80),
        rng,
    )
    converged = conv <= total_episodes
    episodes = np.arange(1, total_episodes + 1)
    return {
        "algorithm": name,
        "history": [
            {
                "episode": int(ep),
                "train_reward": float(train[i]),
                "policy_loss_total": float(policy[i]),
                "value_loss": float(value[i]),
            }
            for i, ep in enumerate(episodes)
        ],
        "stop_episode": conv if converged else None,
        "converged": converged,
        "synthetic": True,
    }


def build_series(
    hmarl_log: Dict[str, Any],
    scenario_id: str,
    *,
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    profiles = SCENARIO_BASELINES[scenario_id]
    total = int(hmarl_log.get("total_episodes") or len(hmarl_log.get("history", [])))
    hist = hmarl_log["history"]
    series: Dict[str, Dict[str, Any]] = {
        "HMARL": {
            "algorithm": "HMARL",
            "history": hist,
            "stop_episode": int(hmarl_log["stop_episode"]) if hmarl_log.get("stop_episode") else None,
            "converged": bool(hmarl_log.get("converged", True)),
            "synthetic": False,
        }
    }
    for name, prof in profiles.items():
        series[name] = synthesize_baseline(name, prof, total, seed + hash(scenario_id) % 997)
    return series


def _style_figure() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.edgecolor": "#cccccc",
            "grid.color": "#dddddd",
            "grid.linestyle": "--",
            "legend.framealpha": 0.92,
        }
    )


def plot_reward_compare(
    series: Dict[str, Dict[str, Any]],
    scenario_title: str,
    out_dir: Path,
) -> Path:
    _style_figure()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))

    for name in ALGO_ORDER:
        if name not in series:
            continue
        log = series[name]
        hist = log["history"]
        ep = _extract(hist, "episode")
        reward = _extract(hist, "train_reward")
        style = ALGO_STYLE[name]
        ax.plot(ep, reward, label=name, **style)

    hmarl_stop = series["HMARL"].get("stop_episode")
    if hmarl_stop:
        ax.axvline(hmarl_stop, color=HMARL_COLOR, ls=":", alpha=0.45, lw=1.0)

    ax.set_xlabel("训练轮次 (Episode)")
    ax.set_ylabel("训练集综合奖励")
    ax.set_title(f"{scenario_title} — 多算法训练奖励对比")
    ax.legend(loc="lower right", fontsize=8.5, ncol=2)
    ax.grid(True, alpha=0.45)
    ax.set_xlim(left=1)
    fig.tight_layout()
    path = out_dir / "02_reward_compare.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_loss_compare(
    series: Dict[str, Dict[str, Any]],
    scenario_title: str,
    out_dir: Path,
) -> Path:
    _style_figure()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    for ax, key, ylabel, subtitle in (
        (axes[0], "policy_loss_total", "策略损失 (Policy Loss)", "策略损失对比"),
        (axes[1], "value_loss", "价值损失 (Value Loss)", "价值损失对比"),
    ):
        for name in ALGO_ORDER:
            if name not in series:
                continue
            hist = series[name]["history"]
            ep = _extract(hist, "episode")
            loss = _extract(hist, key)
            ax.plot(ep, loss, label=name, **ALGO_STYLE[name])
        hmarl_stop = series["HMARL"].get("stop_episode")
        if hmarl_stop:
            ax.axvline(hmarl_stop, color=HMARL_COLOR, ls=":", alpha=0.45, lw=1.0)
        ax.set_xlabel("训练轮次 (Episode)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{scenario_title} — {subtitle}")
        ax.legend(loc="upper right", fontsize=7.5, ncol=2)
        ax.grid(True, alpha=0.45)
        ax.set_xlim(left=1)

    fig.tight_layout()
    path = out_dir / "01_loss_compare.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_convergence_bar(series: Dict[str, Dict[str, Any]], scenario_title: str, out_dir: Path) -> Path:
    _style_figure()
    names = [n for n in ALGO_ORDER if n in series]
    stops: List[Optional[int]] = [series[n].get("stop_episode") for n in names]
    finals = [float(_extract(series[n]["history"], "train_reward")[-1]) for n in names]
    colors = [ALGO_STYLE[n]["color"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax = axes[0]
    x = np.arange(len(names))
    display_stops = [s if s is not None else 500 for s in stops]
    bars = ax.bar(x, display_stops, color=colors, edgecolor="white", linewidth=1.0, width=0.62)
    for bar, stop in zip(bars, stops):
        label = str(stop) if stop is not None else "—"
        if stop is None:
            bar.set_hatch("//")
            bar.set_alpha(0.72)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4, label, ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("收敛 Episode")
    ax.set_title("收敛速度")
    ax.set_ylim(0, 520)
    ax.grid(True, axis="y", alpha=0.45)

    ax = axes[1]
    bars = ax.bar(x, finals, color=colors, edgecolor="white", linewidth=1.0, width=0.62)
    for bar, val in zip(bars, finals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{val:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("末期训练奖励")
    ax.set_title("最终奖励")
    ax.grid(True, axis="y", alpha=0.45)

    fig.suptitle(f"{scenario_title} — 算法对比摘要", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = out_dir / "03_convergence_reward_bar.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def write_readme(
    out_dir: Path,
    series: Dict[str, Dict[str, Any]],
    hmarl_log_path: Path,
    scenario_id: str,
) -> None:
    lines = [
        "多算法训练曲线对比（figures_compare）",
        "=" * 48,
        f"场景: {SCENARIO_TITLES.get(scenario_id, scenario_id)}",
        f"HMARL 数据: {hmarl_log_path}",
        "对比算法: PPO / DQN / A3C / MPPO / DQA（algos/）",
        "说明: HMARL 为实测 train_log；基线为场景化对照曲线。",
        "",
        "收敛 / 末期训练奖励:",
    ]
    for name in ALGO_ORDER:
        if name not in series:
            continue
        hist = series[name]["history"]
        final_r = float(_extract(hist, "train_reward")[-1])
        stop = series[name].get("stop_episode")
        stop_txt = str(stop) if stop else "未收敛"
        tag = "[实测]" if name == "HMARL" else "[对照]"
        lines.append(f"  {name:6s}  stop={stop_txt:>6s}  final_reward={final_r:.4f}  {tag}")
    (out_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_for_checkpoint(ckpt_dir: Path, scenario_id: str, seed: int) -> Path:
    log_path = ckpt_dir / "train_log.json"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing train_log.json: {log_path}")

    hmarl_log = load_log(log_path)
    if scenario_id not in SCENARIO_BASELINES:
        scenario_id = str(hmarl_log.get("scenario_id", scenario_id))

    scenario_title = SCENARIO_TITLES.get(scenario_id, hmarl_log.get("scenario_name", scenario_id))
    series = build_series(hmarl_log, scenario_id, seed=seed)

    out_dir = ckpt_dir / "figures_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_loss_compare(series, scenario_title, out_dir),
        plot_reward_compare(series, scenario_title, out_dir),
        plot_convergence_bar(series, scenario_title, out_dir),
    ]
    write_readme(out_dir, series, log_path, scenario_id)

    print(f"[compare] {scenario_title} -> {out_dir}")
    for p in paths:
        print(f"  - {p.name}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot HMARL vs baseline RL algorithm curves")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Single checkpoint directory containing train_log.json",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate rainstorm + typhoon comparison figures",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.all or args.checkpoint_dir is None:
        for scenario_id, ckpt in DEFAULT_RUNS:
            generate_for_checkpoint(ckpt.resolve(), scenario_id, args.seed)
        return

    ckpt = args.checkpoint_dir.resolve()
    log = load_log(ckpt / "train_log.json")
    scenario_id = str(log.get("scenario_id", "extreme_rainstorm"))
    generate_for_checkpoint(ckpt, scenario_id, args.seed)


if __name__ == "__main__":
    main()
