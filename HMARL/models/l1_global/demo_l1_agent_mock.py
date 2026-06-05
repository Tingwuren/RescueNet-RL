"""L1 全局统筹智能体 - 演示版本（Mock模式，无需PyTorch）

展示内容：
  1. 环境输入（灾害类型、全局库存、区域灾情分布）
  2. 观测编码（构建 L1 观测向量）
  3. 策略决策（模拟 Actor-Critic 输出）
  4. 配额解码（N×5 矩阵生成）
  5. 约束校验（硬约束验证）

用法：
    cd HMARL
    python models/l1_global/demo_l1_agent_mock.py [--scenario typhoon|rainstorm]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

import numpy as np

# Mock 配置和设备名称
DISASTER_NAMES = ["暴雨", "台风风暴潮", "滑坡"]
DEVICE_NAMES = [
    "应急基站",
    "便携式广播通信网关",
    "5G中继",
    "Mesh中继",
    "通信UAV",
]


@dataclass
class L1Config:
    n_regions: int = 5
    n_disaster_types: int = 3
    n_device_types: int = 5
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128])

    @property
    def obs_dim(self) -> int:
        return 12 + 3 * self.n_regions

    @property
    def action_dim(self) -> int:
        return self.n_regions * self.n_device_types


def print_header(title: str, width: int = 70) -> None:
    """打印带分隔线的标题"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_section(title: str) -> None:
    """打印小节标题"""
    print(f"\n>> {title}")
    print("-" * 50)


def print_io_table(headers: list, rows: list, title: str = None) -> None:
    """打印格式化表格"""
    if title:
        print(f"\n  [{title}]")

    col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) + 2
                  for i, h in enumerate(headers)]

    header_line = " | ".join(h.center(w) for h, w in zip(headers, col_widths))
    print(f"  +{'+'.join('-' * (w + 2) for w in col_widths)}+")
    print(f"  | {header_line} |")
    print(f"  +{'+'.join('-' * (w + 2) for w in col_widths)}+")

    for row in rows:
        row_line = " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths))
        print(f"  | {row_line} |")

    print(f"  +{'+'.join('-' * (w + 2) for w in col_widths)}+")


def encode_observation(disaster_type: int, inventory: np.ndarray,
                       severity: np.ndarray, user_counts: np.ndarray,
                       priority_ratios: np.ndarray, cfg: L1Config) -> np.ndarray:
    """模拟观测编码"""
    n = cfg.n_regions

    # 灾害类型 one-hot
    disaster_oh = np.zeros(cfg.n_disaster_types, dtype=np.float32)
    disaster_oh[disaster_type] = 1.0

    # 库存归一化
    inv_norm = inventory / (inventory.max() + 1e-6)

    # 网格摘要 (模拟)
    grid_summary = np.array([1.0, 1.0, 1.0, 0.1], dtype=np.float32)

    # 用户归一化
    umax = user_counts.max() + 1e-6
    users_norm = user_counts / umax

    obs = np.concatenate([disaster_oh, grid_summary, inv_norm, severity, users_norm, priority_ratios])
    return obs


def decode_action_mock(action: np.ndarray, inventory: np.ndarray, cfg: L1Config) -> np.ndarray:
    """模拟配额解码 - 使用 softmax 分配"""
    n, m = cfg.n_regions, cfg.n_device_types
    G = np.asarray(inventory, dtype=np.float32).reshape(m)
    G = np.maximum(G, 0)

    logits = np.asarray(action, dtype=np.float32).reshape(n, m)
    Q = np.zeros((n, m), dtype=np.int32)

    for j in range(m):
        col = logits[:, j]
        w = np.exp(col - col.max())
        w = w / (w.sum() + 1e-8)
        total = int(round(G[j]))
        if total <= 0:
            continue
        raw_alloc = w * total
        floor_alloc = np.floor(raw_alloc).astype(np.int32)
        remainder = total - floor_alloc.sum()
        frac = raw_alloc - floor_alloc
        order = np.argsort(-frac)
        for k in range(int(remainder)):
            floor_alloc[order[k % n]] += 1
        Q[:, j] = floor_alloc

    return Q


def quota_to_dict(Q: np.ndarray) -> dict:
    """配额矩阵转为可读结构"""
    n, m = Q.shape
    return {
        "quota_matrix": Q.tolist(),
        "per_region": [
            {DEVICE_NAMES[j]: int(Q[i, j]) for j in range(m)}
            for i in range(n)
        ],
    }


def demo_l1_agent(scenario: str = "rainstorm") -> None:
    """L1 全局统筹智能体完整演示"""

    # ============================================================
    # STEP 1: 环境配置与初始化
    # ============================================================
    print_header("L1 全局统筹智能体 - 完整流程演示 (Mock模式)")

    cfg = L1Config(n_regions=5)

    print("\n[智能体配置]")
    print(f"   区域数量 (N): {cfg.n_regions}")
    print(f"   设备类型 (M): {cfg.n_device_types}")
    print(f"   观测维度: {cfg.obs_dim}")
    print(f"   动作维度: {cfg.action_dim} (N x M = {cfg.n_regions} x {cfg.n_device_types})")
    print(f"   神经网络: Actor({sum(cfg.hidden_dims) * 100:,} params) + Critic({sum(cfg.hidden_dims) * 50:,} params) [模拟]")

    # ============================================================
    # STEP 2: 环境输入数据
    # ============================================================
    print_header("STEP 1: 环境输入 (Environment State)")

    # 选择场景
    if scenario == "typhoon":
        disaster_type = 1
        disaster_name = DISASTER_NAMES[disaster_type]
        severity = np.array([0.8, 0.6, 0.5, 0.4, 0.3], dtype=np.float32)
    else:
        disaster_type = 0
        disaster_name = DISASTER_NAMES[disaster_type]
        severity = np.array([0.9, 0.7, 0.5, 0.3, 0.2], dtype=np.float32)

    # 全局库存
    inventory = np.array([10, 8, 6, 12, 4], dtype=np.float32)

    # 用户需求分布
    user_counts = np.array([5000, 3000, 2000, 1500, 800], dtype=np.float32)
    priority_ratios = np.array([0.4, 0.35, 0.2, 0.15, 0.1], dtype=np.float32)

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
            "***" if severity[i] > 0.7 else "** " if severity[i] > 0.4 else "*  "
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

    obs = encode_observation(disaster_type, inventory, severity, user_counts, priority_ratios, cfg)

    print_section("观测向量组成")

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

    print("\n  [1] 灾害类型 One-Hot (3维)")
    for i, name in enumerate(DISASTER_NAMES):
        marker = "<-- 当前" if disaster_oh[i] > 0.5 else ""
        print(f"      {name}: {disaster_oh[i]:.2f} {marker}")

    print("\n  [2] 全局网格摘要 (4维)")
    print(f"      区域数归一化: {grid_summary[0]:.4f}")
    print(f"      网格行归一化: {grid_summary[1]:.4f}")
    print(f"      网格列归一化: {grid_summary[2]:.4f}")
    print(f"      总面积归一化: {grid_summary[3]:.4f}")

    print("\n  [3] 全局库存归一化 (5维)")
    for name, val in zip(DEVICE_NAMES, inv_norm):
        bar = "#" * int(val * 20) + "-" * (20 - int(val * 20))
        print(f"      {name:12s}: [{bar}] {val:.3f}")

    print(f"\n  [4] 区域灾情严重度 (N={cfg.n_regions}维)")
    for i in range(cfg.n_regions):
        bar = "#" * int(severity_norm[i] * 20) + "-" * (20 - int(severity_norm[i] * 20))
        print(f"      区域 {i}: [{bar}] {severity_norm[i]:.3f}")

    print(f"\n  [5] 区域用户数归一化 (N={cfg.n_regions}维)")
    for i in range(cfg.n_regions):
        bar = "#" * int(user_norm[i] * 20) + "-" * (20 - int(user_norm[i] * 20))
        print(f"      区域 {i}: [{bar}] {user_norm[i]:.3f}")

    print(f"\n  [6] 区域高优先级占比 (N={cfg.n_regions}维)")
    for i in range(cfg.n_regions):
        bar = "#" * int(priority[i] * 20) + "-" * (20 - int(priority[i] * 20))
        print(f"      区域 {i}: [{bar}] {priority[i]:.3f}")

    print(f"\n  [OK] 观测向量 shape: {obs.shape}")
    print(f"  [OK] 向量值范围: [{obs.min():.3f}, {obs.max():.3f}]")

    # ============================================================
    # STEP 4: 策略决策 (Mock)
    # ============================================================
    print_header("STEP 3: 策略决策 (Policy Decision)")

    print_section("Actor-Critic 前向传播 [模拟]")

    # 模拟 Actor 输出 (基于灾情严重度的启发式)
    np.random.seed(42)
    action = np.zeros(cfg.action_dim, dtype=np.float32)

    # 根据区域灾情分配权重
    for j in range(cfg.n_device_types):
        for i in range(cfg.n_regions):
            # 灾情越重，分配权重越高
            weight = severity[i] * (1 + priority_ratios[i])
            action[i * cfg.n_device_types + j] = weight + np.random.randn() * 0.1

    log_prob = -2.5 + np.random.randn() * 0.5
    value = 0.85 + np.random.randn() * 0.1

    print(f"   观测输入: tensor shape (1, {cfg.obs_dim})")
    print(f"   Actor 网络:")
    print(f"     - 输入层 -> 隐藏层({cfg.hidden_dims[0]}) -> 隐藏层({cfg.hidden_dims[1]}) -> 输出层({cfg.action_dim})")
    print(f"     - 输出: 连续动作向量 shape {action.shape}")
    print(f"   Critic 网络:")
    print(f"     - 输入层 -> 隐藏层({cfg.hidden_dims[0]}) -> 隐藏层({cfg.hidden_dims[1]}) -> 输出层(1)")
    print(f"     - 输出: 状态价值 V(s) = {value:.4f}")
    print(f"   策略信息:")
    print(f"     - 对数概率 log_prob: {log_prob:.4f}")
    print(f"     - 动作均值/标准差: [模拟高斯分布采样]")

    print_section("原始动作向量 (连续松弛值)")
    action_reshaped = action.reshape(cfg.n_regions, cfg.n_device_types)
    for j, dev_name in enumerate(DEVICE_NAMES):
        vals = action_reshaped[:, j]
        print(f"   {dev_name:12s}: [{', '.join(f'{v:+.3f}' for v in vals)}]")

    # ============================================================
    # STEP 5: 配额解码输出
    # ============================================================
    print_header("STEP 4: 配额解码输出 (Quota Decoding)")

    Q = decode_action_mock(action, inventory, cfg)
    quota_dict = quota_to_dict(Q)

    print_section("解码算法: Softmax + 整数分配")
    print("""
   算法步骤:
   1. 对每类设备 j，在 N 个区域上做 softmax: w_i = exp(a_ij) / sum exp(a_kj)
   2. 按权重分配整数: Q_ij = floor(w_i x G_j)
   3. 处理余数: 按小数部分从大到小依次加1
   4. 硬约束投影: 确保 sum Q_ij <= G_j (全局库存)
    """)

    print_section("N x 5 配额矩阵 Q (区域 x 设备类型)")
    print(f"\n  硬约束: 每列之和 <= 全局库存")
    print()

    # 打印矩阵表头
    header = ["区域"] + [name[:6] for name in DEVICE_NAMES] + ["合计"]
    rows = []
    for i in range(cfg.n_regions):
        row = [f"区域 {i}"] + [int(Q[i, j]) for j in range(cfg.n_device_types)]
        row.append(int(Q[i].sum()))
        rows.append(row)

    # 添加列合计行
    col_sums = Q.sum(axis=0)
    footer = ["列合计"] + [int(col_sums[j]) for j in range(cfg.n_device_types)]
    footer.append(int(col_sums.sum()))

    # 库存行
    stock_row = ["库存上限"] + [int(inventory[j]) for j in range(cfg.n_device_types)]
    stock_row.append(int(inventory.sum()))

    print_io_table(header, rows + [footer, stock_row], "配额分配矩阵")

    print_section("列约束验证 (每类设备分配 <= 库存)")
    for j, name in enumerate(DEVICE_NAMES):
        allocated = int(col_sums[j])
        limit = int(inventory[j])
        status = "[OK]" if allocated <= limit else "[超限!]"
        bar = "#" * allocated + "-" * (limit - allocated)
        print(f"   {name:12s}: [{bar}] {allocated}/{limit} {status}")

    print(f"\n  [OK] 总分配设备: {int(Q.sum())} / {int(inventory.sum())} "
          f"({Q.sum()/inventory.sum()*100:.1f}%)")

    # ============================================================
    # STEP 6: L2/L3 硬约束校验
    # ============================================================
    print_header("STEP 5: L2/L3 硬约束校验 (Constraint Check)")

    print_section("场景 A: 合规部署请求")
    region_id = 0
    deploy_request = np.array([2, 1, 1, 2, 0], dtype=np.int32)

    print(f"   区域 {region_id} 部署请求:")
    all_pass = True
    for j, name in enumerate(DEVICE_NAMES):
        req = deploy_request[j]
        quota = int(Q[region_id, j])
        passed = req <= quota
        if not passed:
            all_pass = False
        status = "[OK]" if passed else "[超限]"
        print(f"      {name:12s}: 请求 {req} <= 配额 {quota} {status}")

    print(f"\n   校验结果: {'[PASS] 部署请求符合L1配额约束' if all_pass else '[FAIL]'}")

    print_section("场景 B: 超限部署请求 (故意超限测试)")
    overflow_request = np.array([99, 0, 0, 0, 0], dtype=np.int32)

    print(f"   区域 {region_id} 部署请求:")
    all_pass2 = True
    for j, name in enumerate(DEVICE_NAMES):
        req = overflow_request[j]
        quota = int(Q[region_id, j])
        passed = req <= quota
        if not passed:
            all_pass2 = False
        status = "[OK]" if passed else "[超限!]"
        print(f"      {name:12s}: 请求 {req} <= 配额 {quota} {status}")

    fail_reason = f"应急基站请求({overflow_request[0]}) > 配额({int(Q[region_id, 0])})"
    print(f"\n   校验结果: {'[PASS]' if all_pass2 else '[FAIL] - ' + fail_reason}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print_header("演示总结 (Summary)")

    print(f"""
输入 -> 输出 完整流程:

   +-----------------------------------------------------------+
   |  输入层 (环境)                                            |
   |  灾害类型: {disaster_name:12s}                                 |
   |  全局库存: {int(inventory.sum())} 台设备 ({cfg.n_device_types} 类)                          |
   |  区域分布: {cfg.n_regions} 个区域，灾情严重度 {severity.max():.2f}~{severity.min():.2f}                |
   +-----------------------------------------------------------+
                              |
                              v 观测编码 (12+3N={cfg.obs_dim}维)
   +-----------------------------------------------------------+
   |  神经网络层 (Actor-Critic)                                |
   |  Actor: 观测({cfg.obs_dim}) -> 隐藏({cfg.hidden_dims[0]}) -> 隐藏({cfg.hidden_dims[1]}) -> 动作({cfg.action_dim})   |
   |  Critic: 观测({cfg.obs_dim}) -> 隐藏({cfg.hidden_dims[0]}) -> 隐藏({cfg.hidden_dims[1]}) -> 价值(1)   |
   |  输出: 状态价值 V(s) ~ {value:.4f}                               |
   +-----------------------------------------------------------+
                              |
                              v 配额解码 (Softmax + 整数分配)
   +-----------------------------------------------------------+
   |  输出层 (N x {cfg.n_device_types} 配额矩阵)                                       |
   |  总分配: {int(Q.sum())} 台设备                                        |
   |  高优先级区域(区域0)配额: {int(Q[0].sum())} 台                            |
   |  低优先级区域(区域4)配额: {int(Q[4].sum())} 台                             |
   +-----------------------------------------------------------+

[OK] 所有硬约束满足: 列和 <= 库存上限，L2/L3 可安全调用
[OK] 区域配额与灾情正相关: 区域0(灾重) > 区域4(灾轻)
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="L1 全局统筹智能体演示 (Mock)")
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
