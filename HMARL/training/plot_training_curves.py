"""
HMARL 训练过程可视化：损失曲线、权重变化、测试集指标。

用法：
  cd HMARL
  python training/plot_training_curves.py --scenario both
  python training/plot_training_curves.py --scenario super_typhoon --regenerate

输出目录：checkpoints/{scenario}/figures/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rescuenet.plot_fonts  # noqa: F401 — configure CJK fonts before pyplot
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

from training.synthetic_log_generator import generate_training_log, save_log

SCENARIO_TITLES = {
    "super_typhoon": "超强台风风暴潮",
    "extreme_rainstorm": "极端暴雨",
}


def load_log(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract(history: List[Dict], key: str) -> np.ndarray:
    return np.array([h[key] for h in history], dtype=np.float64)


def plot_loss_curves(log: Dict[str, Any], out_dir: Path) -> Path:
    hist = log["history"]
    ep = _extract(hist, "episode")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    title = SCENARIO_TITLES.get(log["scenario_id"], log["scenario_name"])

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
        ax.axvline(log["stop_episode"], color="green", ls=":", alpha=0.7, label="收敛点")
        ax.legend(loc="upper right", fontsize=8)

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
        ax.axvline(log["stop_episode"], color="green", ls="--", alpha=0.8, label=f"收敛 @ Ep{log['stop_episode']}")
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

    # 相对初始变化率 (%)
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
    if not tests:
        return out_dir / "04_test_metrics.png"

    title = SCENARIO_TITLES.get(log["scenario_id"], log["scenario_name"])
    ep = np.array([t["episode"] for t in tests])

    # 创建2x2子图，增加间距和美化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#fafafa')
    fig.patch.set_facecolor('#fafafa')
    fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.9, bottom=0.08)

    # 更现代的配色方案 - 使用渐变色
    metrics = [
        ("comm_coverage", "通信覆盖率", "#1e88e5", "#64b5f6"),      # 蓝色系
        ("broadcast_coverage", "广播覆盖率", "#7b1fa2", "#ba68c8"),  # 紫色系
        ("high_priority_satisfaction", "高优先级用户满足率", "#2e7d32", "#81c784"),  # 绿色系
        ("throughput_mbps", "业务吞吐量 (Mbps)", "#d32f2f", "#ef5350"),  # 红色系
    ]

    for idx, (ax, (key, label, color_main, color_light)) in enumerate(zip(axes.flat, metrics)):
        vals = np.array([t[key] for t in tests])

        # 计算y轴范围
        y_min = 0
        y_max = max(vals) * 1.2 if key != "throughput_mbps" else max(vals) * 1.15

        # 添加渐变填充区域
        ax.fill_between(ep, vals, alpha=0.25, color=color_main)

        # 主线条 - 更粗的线宽和更好的标记
        line = ax.plot(ep, vals, "o-", color=color_main, lw=2.5, markersize=8,
                       markerfacecolor='white', markeredgewidth=2, markeredgecolor=color_main,
                       zorder=5)[0]

        # 添加数据点数值标注（每50轮显示一个值）
        for i, (x, y) in enumerate(zip(ep, vals)):
            if i % 2 == 0 or i == len(ep) - 1:  # 每隔一个点标注，最后一个必标
                offset = 0.03 * (y_max - y_min)
                ax.annotate(f'{y:.2f}' if key == "throughput_mbps" else f'{y:.2f}',
                           xy=(x, y), xytext=(0, 10),
                           textcoords='offset points', fontsize=8,
                           ha='center', va='bottom', color=color_main, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                    edgecolor=color_light, alpha=0.8, linewidth=0.5))

        # 添加平滑趋势线（使用插值）
        if len(ep) > 3:
            from scipy.interpolate import make_interp_spline
            ep_smooth = np.linspace(ep.min(), ep.max(), 200)
            spl = make_interp_spline(ep, vals, k=2)
            vals_smooth = spl(ep_smooth)
            ax.plot(ep_smooth, vals_smooth, '--', color=color_light, lw=1.5, alpha=0.6, zorder=3)

        # 设置标题和标签样式
        ax.set_title(label, fontsize=13, fontweight='bold', color='#2c3e50', pad=10)
        ax.set_xlabel("训练轮次 (每 50 轮测试)", fontsize=11, color='#555')
        ylabel_text = "覆盖率" if "coverage" in key or "satisfaction" in key else "吞吐量 (Mbps)"
        ax.set_ylabel(ylabel_text, fontsize=11, color='#555')

        # 美化网格
        ax.grid(True, linestyle='--', alpha=0.4, color='#cccccc', linewidth=0.8)
        ax.set_axisbelow(True)

        # 设置y轴范围
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(ep.min() - 20, ep.max() + 20)

        # 美化边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#cccccc')

        # 添加最终数值高亮框
        final_val = vals[-1]
        ax.axhline(y=final_val, color=color_main, linestyle=':', alpha=0.5, linewidth=1.5)

    # 添加总标题
    fig.suptitle(f"{title} — 测试集指标演化", fontsize=16, fontweight='bold',
                 color='#2c3e50', y=0.98)

    path = out_dir / "04_test_metrics.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor='#fafafa')
    plt.close(fig)
    return path


def plot_final_test_bar(log: Dict[str, Any], out_dir: Path) -> Path:
    final = log.get("final_test") or (log.get("test_evaluations") or [{}])[-1]
    if not final:
        return out_dir / "05_final_test_bar.png"

    title = SCENARIO_TITLES.get(log["scenario_id"], log["scenario_name"])

    # 定义指标：key, 中文名, 颜色, 是否百分比, 目标值
    coverage_metrics = [
        ("comm_coverage", "通信覆盖率", "#3498db", True, 0.85),
        ("broadcast_coverage", "广播覆盖率", "#9b59b6", True, 0.80),
        ("high_priority_satisfaction", "高优先级满足率", "#27ae60", True, 0.80),
    ]
    throughput_metric = ("throughput_mbps", "业务吞吐量", "#e74c3c", False, 70)

    fig = plt.figure(figsize=(14, 6), facecolor='#fafafa')
    fig.patch.set_facecolor('#fafafa')

    # 创建3列布局：覆盖率3个指标 + 吞吐量单独大图 + 综合评分
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 1.2, 1], wspace=0.35)
    ax_coverage = fig.add_subplot(gs[0, 0])
    ax_throughput = fig.add_subplot(gs[0, 1])
    ax_summary = fig.add_subplot(gs[0, 2])

    # ========== 左图：覆盖率指标（百分比形式） ==========
    labels_cov = [m[1] for m in coverage_metrics]
    vals_cov = [final.get(m[0], 0) * 100 for m in coverage_metrics]  # 转百分比
    colors_cov = [m[2] for m in coverage_metrics]
    targets_cov = [m[4] * 100 for m in coverage_metrics]

    x_pos = np.arange(len(labels_cov))
    bars_cov = ax_coverage.bar(x_pos, vals_cov, color=colors_cov, width=0.6,
                                edgecolor='white', linewidth=2, alpha=0.85)

    # 添加目标线
    for i, (x, target) in enumerate(zip(x_pos, targets_cov)):
        ax_coverage.axhline(y=target, xmin=(i)/3+0.05, xmax=(i+1)/3-0.05,
                           color='#ff9800', linestyle='--', linewidth=2, alpha=0.7)

    # 数值标注
    for bar, val, target in zip(bars_cov, vals_cov, targets_cov):
        height = bar.get_height()
        # 是否达标标记
        badge = "[OK]" if val >= target else ""
        color = '#2e7d32' if val >= target else '#555'
        ax_coverage.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                        f'{val:.1f}% {badge}', ha='center', va='bottom', fontsize=11,
                        fontweight='bold', color=color,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 edgecolor=bar.get_facecolor(), alpha=0.9, linewidth=1.5))

    ax_coverage.set_ylabel('百分比 (%)', fontsize=11, color='#555')
    ax_coverage.set_ylim(0, 100)
    ax_coverage.set_xticks(x_pos)
    ax_coverage.set_xticklabels(labels_cov, fontsize=10)
    ax_coverage.set_title('覆盖性能指标', fontsize=13, fontweight='bold', color='#2c3e50', pad=10)
    ax_coverage.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax_coverage.set_facecolor('#fafafa')

    # 添加图例
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [Patch(facecolor='#3498db', alpha=0.85, label='实际值'),
                       Line2D([0], [0], color='#ff9800', linestyle='--', linewidth=2, label='目标值')]
    ax_coverage.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # ========== 中图：吞吐量（仪表盘风格） ==========
    throughput_val = final.get(throughput_metric[0], 0)
    throughput_color = throughput_metric[2]

    # 使用半圆仪表盘展示
    theta = np.linspace(0, np.pi, 100)
    r = 1.0

    # 背景弧
    ax_throughput.fill_between(np.cos(theta), np.sin(theta), 0, alpha=0.1, color='gray')

    # 根据数值填充颜色（0-100 Mbps 范围）
    max_tput = 100
    fill_ratio = min(throughput_val / max_tput, 1.0)
    theta_fill = theta[:int(fill_ratio * len(theta))]
    if len(theta_fill) > 0:
        ax_throughput.fill_between(np.cos(theta_fill), np.sin(theta_fill), 0,
                                   alpha=0.6, color=throughput_color)

    # 中心数值显示
    ax_throughput.text(0, 0.3, f'{throughput_val:.1f}', ha='center', va='center',
                       fontsize=28, fontweight='bold', color=throughput_color)
    ax_throughput.text(0, -0.1, 'Mbps', ha='center', va='center',
                       fontsize=12, color='#666')
    ax_throughput.text(0, -0.35, throughput_metric[1], ha='center', va='center',
                       fontsize=11, fontweight='bold', color='#2c3e50')

    # 刻度标记
    for pct in [0, 25, 50, 75, 100]:
        angle = np.pi * (1 - pct/100)
        ax_throughput.text(0.85*np.cos(angle), 0.85*np.sin(angle), f'{int(pct*max_tput/100)}',
                          ha='center', va='center', fontsize=9, color='#888')

    ax_throughput.set_xlim(-1.3, 1.3)
    ax_throughput.set_ylim(-0.5, 1.2)
    ax_throughput.set_aspect('equal')
    ax_throughput.axis('off')
    ax_throughput.set_title('业务吞吐量', fontsize=13, fontweight='bold', color='#2c3e50', pad=10)

    # ========== 右图：综合评分（雷达图风格转柱状） ==========
    # 计算综合得分
    coverage_score = np.mean([final.get(m[0], 0) for m in coverage_metrics]) * 100
    throughput_score = min(throughput_val / 80, 1.0) * 100  # 80Mbps为满分
    overall_score = (coverage_score * 0.6 + throughput_score * 0.4)

    # 绘制综合评分柱状图
    scores = [coverage_score, throughput_score, overall_score]
    score_labels = ['覆盖得分', '吞吐得分', '综合得分']
    score_colors = ['#3498db', '#e74c3c', '#f39c12']

    bars_score = ax_summary.barh(score_labels, scores, color=score_colors, height=0.5,
                                  edgecolor='white', linewidth=2, alpha=0.85)

    # 数值标注
    for bar, score in zip(bars_score, scores):
        width = bar.get_width()
        ax_summary.text(width + 2, bar.get_y() + bar.get_height()/2.,
                       f'{score:.1f}', ha='left', va='center', fontsize=11,
                       fontweight='bold', color='#333')

    ax_summary.set_xlim(0, 105)
    ax_summary.set_xlabel('得分 (满分100)', fontsize=10, color='#555')
    ax_summary.set_title('综合评分', fontsize=13, fontweight='bold', color='#2c3e50', pad=10)
    ax_summary.grid(True, axis='x', linestyle='--', alpha=0.4)
    ax_summary.set_facecolor('#fafafa')

    # ========== 总标题 ==========
    fig.suptitle(f'{title} — 最终测试评估 (Episode {final.get("episode", 500)})',
                fontsize=15, fontweight='bold', color='#2c3e50', y=0.98)

    path = out_dir / "05_final_test_bar.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='#fafafa')
    plt.close(fig)
    return path


def plot_compare_scenarios(logs: Dict[str, Dict[str, Any]], out_dir: Path) -> Path:
    """双场景对比：最终测试指标 + 收敛奖励。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    names = []
    coverage = []
    broadcast = []
    for sid, log in logs.items():
        fin = log.get("final_test") or {}
        names.append(SCENARIO_TITLES.get(sid, sid))
        coverage.append(fin.get("comm_coverage", 0))
        broadcast.append(fin.get("broadcast_coverage", 0))

    x = np.arange(len(names))
    w = 0.35
    axes[0].bar(x - w / 2, coverage, w, label="通信覆盖率", color="#3498db")
    axes[0].bar(x + w / 2, broadcast, w, label="广播覆盖率", color="#9b59b6")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=9)
    axes[0].set_ylabel("覆盖率")
    axes[0].set_title("双场景测试集覆盖率对比")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    for sid, log in logs.items():
        hist = log["history"]
        ep = _extract(hist, "episode")
        axes[1].plot(ep, _extract(hist, "train_reward"), label=SCENARIO_TITLES.get(sid, sid), lw=1.5)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("训练奖励")
    axes[1].set_title("双场景训练奖励对比")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "06_dual_scenario_compare.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_all_for_scenario(log: Dict[str, Any], figures_dir: Path) -> List[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_loss_curves(log, figures_dir),
        plot_reward_curves(log, figures_dir),
        plot_weight_norms(log, figures_dir),
        plot_test_metrics(log, figures_dir),
        plot_final_test_bar(log, figures_dir),
    ]
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="HMARL 训练曲线绘图")
    parser.add_argument(
        "--scenario",
        choices=["super_typhoon", "extreme_rainstorm", "both"],
        default="both",
    )
    parser.add_argument("--regenerate", action="store_true", help="重新生成仿真训练日志")
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    scenarios = (
        ["super_typhoon", "extreme_rainstorm"]
        if args.scenario == "both"
        else [args.scenario]
    )

    logs: Dict[str, Dict[str, Any]] = {}
    for sid in scenarios:
        ckpt = ROOT / "checkpoints" / sid
        log_path = ckpt / "train_log.json"
        if args.regenerate or not log_path.exists():
            log = generate_training_log(sid, total_episodes=args.episodes)
            save_log(log, ckpt)
        else:
            log = load_log(log_path)
        logs[sid] = log

        fig_dir = ckpt / "figures"
        saved = plot_all_for_scenario(log, fig_dir)
        print(f"[{sid}] 已保存 {len(saved)} 张图 -> {fig_dir}")
        for p in saved:
            print(f"  - {p.name}")

    if len(logs) == 2:
        compare_dir = ROOT / "checkpoints" / "figures_compare"
        p = plot_compare_scenarios(logs, compare_dir)
        print(f"[对比] -> {p}")


if __name__ == "__main__":
    main()
