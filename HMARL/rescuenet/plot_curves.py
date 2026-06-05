"""Plot 01–04 figures (matplotlib/numpy only; mirrors training/plot_training_curves.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import rescuenet.plot_fonts  # noqa: F401  — configure CJK fonts before pyplot
import matplotlib.pyplot as plt
import numpy as np

SCENARIO_TITLES = {
    "super_typhoon": "超强台风风暴潮",
    "extreme_rainstorm": "极端暴雨",
}


def _extract(history: List[Dict], key: str) -> np.ndarray:
    return np.array([h[key] for h in history], dtype=np.float64)


def plot_loss_curves(log: Dict[str, Any], out_dir: Path) -> Path:
    hist = log["history"]
    ep = _extract(hist, "episode")
    title = SCENARIO_TITLES.get(log["scenario_id"], log["scenario_name"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(ep, _extract(hist, "policy_loss_l1"), label="L1 策略损失", alpha=0.9, lw=1.2)
    ax.plot(ep, _extract(hist, "policy_loss_l2"), label="L2 策略损失", alpha=0.9, lw=1.2)
    ax.plot(ep, _extract(hist, "policy_loss_l3"), label="L3 策略损失", alpha=0.9, lw=1.2)
    ax.plot(
        ep,
        _extract(hist, "policy_loss_total"),
        label="合计策略损失",
        color="#333",
        ls="--",
        lw=1.5,
    )
    ax.set_xlabel("训练轮次 (Episode)")
    ax.set_ylabel("策略损失 (Policy Loss)")
    ax.set_title(f"{title} — 分层策略损失")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    if log.get("stop_episode"):
        ax.axvline(log["stop_episode"], color="green", ls=":", alpha=0.7)

    ax = axes[1]
    ax.plot(ep, _extract(hist, "value_loss"), color="#c0392b", lw=1.5, label="价值函数损失")
    ax.set_xlabel("训练轮次 (Episode)")
    ax.set_ylabel("价值损失 (Value Loss)")
    ax.set_title(f"{title} — Critic 价值损失")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "01_loss_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_reward_curves(log: Dict[str, Any], out_dir: Path) -> Path:
    hist = log["history"]
    ep = _extract(hist, "episode")
    title = SCENARIO_TITLES.get(log["scenario_id"], log["scenario_name"])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(ep, _extract(hist, "train_reward"), label="训练集综合奖励", color="#2980b9", lw=1.5)
    ax.plot(ep, _extract(hist, "val_reward"), label="验证集综合奖励", color="#e67e22", lw=1.5, alpha=0.9)
    if log.get("stop_episode"):
        ax.axvline(
            log["stop_episode"],
            color="green",
            ls="--",
            alpha=0.8,
            label=f"收敛 @ Ep{log['stop_episode']}",
        )
    ax.set_xlabel("训练轮次 (Episode)")
    ax.set_ylabel("综合奖励值")
    ax.set_title(f"{title} — 训练/验证奖励曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "02_reward_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_weight_norms(log: Dict[str, Any], out_dir: Path) -> Path:
    hist = log["history"]
    ep = _extract(hist, "episode")
    title = SCENARIO_TITLES.get(log["scenario_id"], log["scenario_name"])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(ep, _extract(hist, "weight_norm_l1"), label="L1 Actor 权重范数", lw=1.5)
    ax.plot(ep, _extract(hist, "weight_norm_l2"), label="L2 Actor 权重范数", lw=1.5)
    ax.plot(ep, _extract(hist, "weight_norm_l3"), label="L3 Actor 权重范数", lw=1.5)

    w1 = _extract(hist, "weight_norm_l1")
    delta_pct = (w1 - w1[0]) / (w1[0] + 1e-8) * 100
    ax2 = ax.twinx()
    ax2.plot(ep, delta_pct, color="#7f8c8d", ls=":", alpha=0.6, label="L1 相对变化 %")
    ax2.set_ylabel("权重相对变化 (%)", color="#7f8c8d")

    ax.set_xlabel("训练轮次 (Episode)")
    ax.set_ylabel("权重 L2 范数 ||θ||")
    ax.set_title(f"{title} — 三级智能体权重演化")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "03_weight_norms.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_test_metrics(log: Dict[str, Any], out_dir: Path) -> Path:
    tests = log.get("test_evaluations", [])
    path = out_dir / "04_test_metrics.png"
    if not tests:
        return path

    title = SCENARIO_TITLES.get(log["scenario_id"], log["scenario_name"])
    ep = np.array([t["episode"] for t in tests])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#fafafa")
    fig.patch.set_facecolor("#fafafa")
    fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.9, bottom=0.08)

    metrics = [
        ("comm_coverage", "通信覆盖率", "#1e88e5", "#64b5f6"),
        ("broadcast_coverage", "广播覆盖率", "#7b1fa2", "#ba68c8"),
        ("high_priority_satisfaction", "高优先级用户满足率", "#2e7d32", "#81c784"),
        ("throughput_mbps", "业务吞吐量 (Mbps)", "#d32f2f", "#ef5350"),
    ]

    for ax, (key, label, color_main, color_light) in zip(axes.flat, metrics):
        vals = np.array([t[key] for t in tests])
        y_min = 0.0
        y_max = max(vals) * 1.2 if key != "throughput_mbps" else max(vals) * 1.15

        ax.fill_between(ep, vals, alpha=0.25, color=color_main)
        ax.plot(
            ep,
            vals,
            "o-",
            color=color_main,
            lw=2.5,
            markersize=8,
            markerfacecolor="white",
            markeredgewidth=2,
            markeredgecolor=color_main,
            zorder=5,
        )
        if len(ep) > 3:
            ep_smooth = np.linspace(ep.min(), ep.max(), 200)
            vals_smooth = np.interp(ep_smooth, ep, vals)
            ax.plot(ep_smooth, vals_smooth, "--", color=color_light, lw=1.5, alpha=0.6, zorder=3)

        ax.set_title(label, fontsize=13, fontweight="bold", color="#2c3e50", pad=10)
        ax.set_xlabel("训练轮次 (评估点)", fontsize=11, color="#555")
        ylabel_text = "覆盖率" if "coverage" in key or "satisfaction" in key else "吞吐量 (Mbps)"
        ax.set_ylabel(ylabel_text, fontsize=11, color="#555")
        ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(ep.min() - max(2, ep.min() * 0.05), ep.max() + max(2, ep.max() * 0.05))
        final_val = vals[-1]
        ax.axhline(y=final_val, color=color_main, linestyle=":", alpha=0.5, linewidth=1.5)

    fig.suptitle(
        f"{title} — 测试集指标演化 (Episodes={log.get('total_episodes', len(ep))})",
        fontsize=16,
        fontweight="bold",
        color="#2c3e50",
        y=0.98,
    )
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="#fafafa")
    plt.close(fig)
    return path


def plot_four_figures(log: Dict[str, Any], figures_dir: Path) -> List[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_loss_curves(log, figures_dir),
        plot_reward_curves(log, figures_dir),
        plot_weight_norms(log, figures_dir),
        plot_test_metrics(log, figures_dir),
    ]
