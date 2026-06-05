"""L1 全局统筹智能体 - 完整流程演示

展示内容：
  1. 环境输入（灾害类型、全局库存、区域灾情分布）
  2. 观测编码（构建 L1 观测向量）
  3. 策略决策（Actor-Critic 前向传播）
  4. 配额解码（N×5 矩阵生成）
  5. 约束校验（硬约束验证）

用法：
    cd HMARL
    python -m models.l1_global.demo_l1_agent [--scenario typhoon|rainstorm]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.l1_global import L1Config, L1GlobalAgent, quota_to_dict
from models.l1_global.l1_spaces import DISASTER_NAMES, DEVICE_NAMES


def print_header(title: str, width: int = 70) -> None:
    """打印带分隔线的标题"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_section(title: str) -> None:
    """打印小节标题"""
    print(f"\n▶ {title}")
    print("-" * 50)


def print_io_table(headers: list, rows: list, title: str = None) -> None:
    """打印格式化表格"""
    if title:
        print(f"\n  [{title}]")
    
    # 计算列宽
    col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) + 2 
                  for i, h in enumerate(headers)]
    
    # 表头
    header_line = " | ".join(h.center(w) for h, w in zip(headers, col_widths))
    print(f"  +{'+'.join('-' * (w + 2) for w in col_widths)}+")
    print(f"  | {header_line} |")
    print(f"  +{'+'.join('-' * (w + 2) for w in col_widths)}+")
    
    # 数据行
    for row in rows:
        row_line = " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths))
        print(f"  | {row_line} |")
    
    print(f"  +{'+'.join('-' * (w + 2) for w in col_widths)}+")


def demo_l1_agent(scenario: str = "rainstorm") -> None:
    """
    L1 全局统筹智能体完整演示
    
    流程：环境输入 → 观测编码 → 策略决策 → 配额输出 → 约束校验
    """
    
    # ============================================================
    # STEP 1: 环境配置与初始化
    # ============================================================
    print_header("L1 全局统筹智能体 - 完整流程演示")
    
    cfg = L1Config(n_regions=5)
    agent = L1GlobalAgent(cfg)
    
    print(f"\n📦 智能体配置")
    print(f"   • 区域数量 (N): {cfg.n_regions}")
    print(f"   • 设备类型 (M): {cfg.n_device_types}")
    print(f"   • 观测维度: {cfg.obs_dim}")
    print(f"   • 动作维度: {cfg.action_dim} (N×M = {cfg.n_regions}×{cfg.n_device_types})")
    print(f"   • 神经网络: Actor({agent.count_parameters()['actor']:,} params) + "
          f"Critic({agent.count_parameters()['critic']:,} params)")
    print(f"   • 运行设备: {agent.device}")
    
    # ============================================================
    # STEP 2: 环境输入数据
    # ============================================================
    print_header("STEP 1: 环境输入 (Environment State)")
    
    # 选择场景
    if scenario == "typhoon":
        disaster_type = 1  # 台风风暴潮
        disaster_name = DISASTER_NAMES[disaster_type]
        severity = np.array([0.8, 0.6, 0.5, 0.4, 0.3], dtype=np.float32)
    else:
        disaster_type = 0  # 暴雨
        disaster_name = DISASTER_NAMES[disaster_type]
        severity = np.array([0.9, 0.7, 0.5, 0.3, 0.2], dtype=np.float32)
    
    # 全局库存: [应急基站, 便携式广播通信网关, 5G中继, Mesh中继, 通信UAV]
    inventory = np.array([10, 8, 6, 12, 4], dtype=np.float32)
    
    # 用户需求分布
    user_counts = np.array([5000, 3000, 2000, 1500, 800], dtype=np.float32)
    priority_ratios = np.array([0.4, 0.35, 0.2, 0.15, 0.1], dtype=np.float32)
    
    raw_state = {
        "disaster_type": disaster_type,
        "global_inventory": inventory,
        "region_severity": severity,
        "region_user_count": user_counts,
        "region_high_priority_ratio": priority_ratios,
    }
    
    print_section("灾害场景")
    print(f"   灾害类型: {disaster_name} (ID={disaster_type})")
    
    print_section("全局设备库存")
    inventory_table = [[name, int(cnt)] for name, cnt in zip(DEVICE_NAMES, inventory)]
    print_io_table(["设备类型", "可用数量"], inventory_table)
    print(f"   设备总数: {int(inventory.sum())} 台")
    
    print_section("区域灾情与需求分布 (N=5)")
    region_table = []
    for i in range(cfg.n_regions):
        region_table.append([
            f"区域 {i}",
            f"{severity[i]:.2f}",
            f"{int(user_counts[i]):,}",
            f"{priority_ratios[i]*100:.1f}%",
            "★★★" if severity[i] > 0.7 else "★★☆" if severity[i] > 0.4 else "★☆☆"
        ])
    print_io_table(
        ["区域", "灾情严重度", "用户总数", "高优先级占比", "优先级"],
        region_table,
        "区域统计"
    )
    
    # ============================================================
    # STEP 3: 观测编码
    # ============================================================
    print_header("STEP 2: 观测编码 (Observation Encoding)")
    
    obs = agent.build_observation(raw_state)
    
    print_section("观测向量组成 (维度: {}+{}+{}+{}×3 = {})".format(
        3, 4, 5, cfg.n_regions, cfg.obs_dim))
    
    # 解析观测向量各部分
    idx = 0
    disaster_oh = obs[idx:idx+3]
    idx += 3
    grid_summary = obs[idx:idx+4]
    idx += 4
    inv_norm = obs[idx:idx+5]
    idx += 5
    severity_norm = obs[idx:idx+cfg.n_regions]
    idx += cfg.n_regions
    user_norm = obs[idx:idx+cfg.n_regions]
    idx += cfg.n_regions
    priority = obs[idx:idx+cfg.n_regions]
    
    print(f"\n  [1] 灾害类型 One-Hot (3维)")
    for i, name in enumerate(DISASTER_NAMES):
        marker = "◀── 当前" if disaster_oh[i] > 0.5 else ""
        print(f"      {name}: {disaster_oh[i]:.2f} {marker}")
    
    print(f"\n  [2] 全局网格摘要 (4维)")
    print(f"      区域数归一化: {grid_summary[0]:.4f}")
    print(f"      网格行归一化: {grid_summary[1]:.4f}")
    print(f"      网格列归一化: {grid_summary[2]:.4f}")
    print(f"      总面积归一化: {grid_summary[3]:.4f}")
    
    print(f"\n  [3] 全局库存归一化 (5维)")
    for name, val in zip(DEVICE_NAMES, inv_norm):
        bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
        print(f"      {name:12s}: [{bar}] {val:.3f}")
    
    print(f"\n  [4] 区域灾情严重度 (N={cfg.n_regions}维)")
    for i in range(cfg.n_regions):
        bar = "█" * int(severity_norm[i] * 20) + "░" * (20 - int(severity_norm[i] * 20))
        print(f"      区域 {i}: [{bar}] {severity_norm[i]:.3f}")
    
    print(f"\n  [5] 区域用户数归一化 (N={cfg.n_regions}维)")
    for i in range(cfg.n_regions):
        bar = "█" * int(user_norm[i] * 20) + "░" * (20 - int(user_norm[i] * 20))
        print(f"      区域 {i}: [{bar}] {user_norm[i]:.3f}")
    
    print(f"\n  [6] 区域高优先级占比 (N={cfg.n_regions}维)")
    for i in range(cfg.n_regions):
        bar = "█" * int(priority[i] * 20) + "░" * (20 - int(priority[i] * 20))
        print(f"      区域 {i}: [{bar}] {priority[i]:.3f}")
    
    print(f"\n  ✓ 观测向量 shape: {obs.shape}")
    
    # ============================================================
    # STEP 4: 策略决策
    # ============================================================
    print_header("STEP 3: 策略决策 (Policy Decision)")
    
    print_section("Actor-Critic 前向传播")
    
    action, log_prob, value, info = agent.act(obs, raw_state["global_inventory"])
    
    print(f"   • 观测输入: tensor shape (1, {cfg.obs_dim})")
    print(f"   • Actor 输出:")
    print(f"     - 动作均值 (mu): {info.get('action_mean', 'N/A')}")
    print(f"     - 动作标准差 (sigma): {info.get('action_std', 'N/A')}")
    print(f"     - 采样动作 shape: {action.shape}")
    print(f"   • Critic 输出:")
    print(f"     - 状态价值 V(s): {value:.4f}")
    print(f"   • 策略信息:")
    print(f"     - 对数概率 log_prob: {log_prob:.4f}")
    
    print_section("原始动作向量 (连续松弛值)")
    action_reshaped = action.reshape(cfg.n_regions, cfg.n_device_types)
    for j, dev_name in enumerate(DEVICE_NAMES):
        vals = action_reshaped[:, j]
        print(f"   {dev_name:12s}: [{', '.join(f'{v:+.3f}' for v in vals)}]")
    
    # ============================================================
    # STEP 5: 配额解码输出
    # ============================================================
    print_header("STEP 4: 配额解码输出 (Quota Decoding)")
    
    Q = info["quota_matrix"]
    quota_dict = quota_to_dict(Q)
    
    print_section("N×5 配额矩阵 Q (区域 × 设备类型)")
    print(f"\n  硬约束: 每列之和 ≤ 全局库存")
    print()
    
    # 打印矩阵表头
    header = ["区域"] + [name[:6] for name in DEVICE_NAMES] + ["合计"]
    rows = []
    for i in range(cfg.n_regions):
        row = [f"区域 {i}"] + [int(Q[i, j]) for j in range(cfg.n_device_types)]
        row.append(int(Q[i].sum()))
        rows.append(row)
    
    # 添加列合计行
    footer = ["库存上限"] + [int(inventory[j]) for j in range(cfg.n_device_types)]
    footer.append(int(inventory.sum()))
    
    # 打印表格
    print_io_table(header, rows + [footer], "配额分配矩阵")
    
    print_section("列约束验证 (每类设备分配 ≤ 库存)")
    col_sums = Q.sum(axis=0)
    for j, name in enumerate(DEVICE_NAMES):
        allocated = int(col_sums[j])
        limit = int(inventory[j])
        status = "✓" if allocated <= limit else "✗ 超限!"
        bar = "█" * allocated + "░" * (limit - allocated)
        print(f"   {name:12s}: [{bar}] {allocated}/{limit} {status}")
    
    print(f"\n  ✓ 总分配设备: {int(Q.sum())} / {int(inventory.sum())} "
          f"({Q.sum()/inventory.sum()*100:.1f}%)")
    
    # ============================================================
    # STEP 6: L2/L3 硬约束校验
    # ============================================================
    print_header("STEP 5: L2/L3 硬约束校验 (Constraint Check)")
    
    print_section("场景 A: 合规部署请求")
    region_id = 0  # 区域 0
    deploy_request = np.array([2, 1, 1, 2, 0], dtype=np.int32)
    ok, msg = agent.check_l2_l3_feasible(region_id, deploy_request)
    
    print(f"   区域 {region_id} 部署请求:")
    for j, name in enumerate(DEVICE_NAMES):
        req = deploy_request[j]
        quota = int(Q[region_id, j])
        status = "✓" if req <= quota else "✗"
        print(f"      {name:12s}: 请求 {req} ≤ 配额 {quota} {status}")
    
    print(f"\n   校验结果: {'✅ 通过' if ok else '❌ 失败'} - {msg}")
    
    print_section("场景 B: 超限部署请求 (故意超限测试)")
    overflow_request = np.array([99, 0, 0, 0, 0], dtype=np.int32)
    ok2, msg2 = agent.check_l2_l3_feasible(region_id, overflow_request)
    
    print(f"   区域 {region_id} 部署请求:")
    for j, name in enumerate(DEVICE_NAMES):
        req = overflow_request[j]
        quota = int(Q[region_id, j])
        status = "✓" if req <= quota else "✗ 超限!"
        print(f"      {name:12s}: 请求 {req} ≤ 配额 {quota} {status}")
    
    print(f"\n   校验结果: {'✅ 通过' if ok2 else '❌ 失败'} - {msg2}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print_header("演示总结 (Summary)")
    
    print(f"""
📊 输入 → 输出 完整流程:

   ┌─────────────────────────────────────────────────────────────┐
   │  输入层 (环境)                                                │
   │  • 灾害类型: {disaster_name:12s}                         │
   │  • 全局库存: {int(inventory.sum())} 台设备 ({cfg.n_device_types} 类)                    │
   │  • 区域分布: {cfg.n_regions} 个区域，灾情严重度 {severity.max():.2f}~{severity.min():.2f}     │
   └─────────────────────────────────────────────────────────────┘
                              ↓ 观测编码
   ┌─────────────────────────────────────────────────────────────┐
   │  观测层 ({cfg.obs_dim} 维向量)                                    │
   │  • 灾害 one-hot (3) + 网格摘要 (4) + 库存 (5)                  │
   │  • 灾情 (N) + 用户数 (N) + 优先级 (N)                         │
   └─────────────────────────────────────────────────────────────┘
                              ↓ Actor-Critic
   ┌─────────────────────────────────────────────────────────────┐
   │  决策层 (PPO 策略)                                            │
   │  • 状态价值 V(s): {value:+.4f}                                 │
   │  • 策略熵/探索: log_prob = {log_prob:.4f}                      │
   └─────────────────────────────────────────────────────────────┘
                              ↓ 配额解码
   ┌─────────────────────────────────────────────────────────────┐
   │  输出层 (N×{cfg.n_device_types} 配额矩阵)                                       │
   │  • 总分配: {int(Q.sum())} 台设备                                    │
   │  • 高优先级区域(区域0)配额: {int(Q[0].sum())} 台                    │
   │  • 低优先级区域(区域4)配额: {int(Q[4].sum())} 台                     │
   └─────────────────────────────────────────────────────────────┘

✅ 所有硬约束满足: 列和 ≤ 库存上限，L2/L3 可安全调用
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="L1 全局统筹智能体演示")
    parser.add_argument(
        "--scenario",
        choices=["typhoon", "rainstorm"],
        default="rainstorm",
        help="选择灾害场景 (默认: 暴雨)"
    )
    args = parser.parse_args()
    
    demo_l1_agent(args.scenario)


if __name__ == "__main__":
    main()
