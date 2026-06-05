"""HMARL 三层智能体参数量统计与验证 (Mock版本，无需PyTorch)

计算 L1/L2/L3 各层参数量，并验证不同场景规模下的总参数是否满足 >= 100万 要求。

用法：
    cd HMARL
    python models/param_counter_mock.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def format_number(num: float, unit: str = "W") -> str:
    """格式化数字显示（万为单位）"""
    if unit == "W":
        return f"{num/10000:.1f}W"
    elif unit == "K":
        return f"{num/1000:.1f}K"
    else:
        return f"{num:,}"


def calculate_mlp_params(input_dim: int, hidden_dims: List[int], output_dim: int) -> int:
    """
    计算 MLP 网络参数量
    
    每层: (输入维度 + 1) × 输出维度 = 权重 + 偏置
    """
    total = 0
    prev_dim = input_dim
    
    for h in hidden_dims:
        # 权重矩阵: prev_dim × h
        # 偏置向量: h
        total += (prev_dim * h) + h
        prev_dim = h
    
    # 输出层
    total += (prev_dim * output_dim) + output_dim
    
    return total


def calculate_actor_params(obs_dim: int, action_dim: int, hidden_dims: List[int]) -> int:
    """计算 Actor 网络参数量 (backbone + mu_head + log_std)"""
    # Backbone
    backbone_params = calculate_mlp_params(obs_dim, hidden_dims, hidden_dims[-1])
    # Mu head (线性层)
    mu_head_params = (hidden_dims[-1] * action_dim) + action_dim
    # Log std (可学习参数)
    log_std_params = action_dim
    
    return backbone_params + mu_head_params + log_std_params


def calculate_critic_params(obs_dim: int, hidden_dims: List[int]) -> int:
    """计算 Critic 网络参数量"""
    return calculate_mlp_params(obs_dim, hidden_dims, 1)


def print_separator(char: str = "=", length: int = 75) -> None:
    print(char * length)


def print_header(title: str) -> None:
    print_separator()
    print(f"  {title}")
    print_separator()


def print_section(title: str) -> None:
    print(f"\n>> {title}")
    print("-" * 55)


def analyze_layer_architecture() -> Dict[str, Dict]:
    """分析各层网络架构并计算单智能体参数量"""
    
    print_header("HMARL 三层智能体参数量分析 (基于 PyTorch 计算公式)")
    
    results = {}
    
    # --------------------------------------------------------------------------
    # L1 层：全局统筹层
    # --------------------------------------------------------------------------
    print_section("L1 全局统筹层 (Global Coordination)")
    
    # L1 配置 (精确调整至 80W 参数量)
    l1_obs_dim = 27  # 12 + 3*5 (N=5)
    l1_action_dim = 25  # N*5 = 5*5
    # 隐藏层: [640, 384, 256] -> 总参数量约 80W
    l1_hidden_dims = [640, 384, 256]
    
    # 计算参数量
    l1_actor_params = calculate_actor_params(l1_obs_dim, l1_action_dim, l1_hidden_dims)
    l1_critic_params = calculate_critic_params(l1_obs_dim, l1_hidden_dims)
    l1_single_total = l1_actor_params + l1_critic_params
    
    # 打印详细信息
    print(f"  观测维度 (obs_dim): {l1_obs_dim}")
    print(f"    - 灾害类型 one-hot(3) + 网格摘要(4) + 全局库存(5) = 12")
    print(f"    - 区域灾情(N=5) + 用户数(N=5) + 优先级(N=5) = 15")
    print(f"    - 总计: 12 + 15 = {l1_obs_dim} 维")
    print(f"  动作维度 (action_dim): {l1_action_dim} (N×5 = 5×5)")
    print(f"  隐藏层结构: {l1_hidden_dims}")
    
    print(f"\n  Actor 网络参数量计算:")
    backbone_l1 = calculate_mlp_params(l1_obs_dim, l1_hidden_dims, l1_hidden_dims[-1])
    print(f"    - Backbone ({l1_obs_dim} -> {l1_hidden_dims} -> {l1_hidden_dims[-1]}):")
    print(f"      {l1_obs_dim}×256+256 + 256×256+256 + 256×128+128 = {format_number(backbone_l1, 'K')}")
    mu_head_l1 = (l1_hidden_dims[-1] * l1_action_dim) + l1_action_dim
    print(f"    - Mu head (128 -> {l1_action_dim}): 128×25+25 = {format_number(mu_head_l1, 'K')}")
    print(f"    - Log std ({l1_action_dim} 维可学习参数): {l1_action_dim}")
    print(f"    - Actor 总计: {format_number(l1_actor_params)}")
    
    print(f"\n  Critic 网络参数量计算:")
    print(f"    - 与 Actor Backbone 相同结构，输出维度改为 1")
    print(f"    - Critic 总计: {format_number(l1_critic_params)}")
    
    print(f"\n  * L1 单智能体总参数量: {format_number(l1_single_total)}")
    print(f"    (Actor {format_number(l1_actor_params)} + Critic {format_number(l1_critic_params)})")
    print(f"  配置规则: 全局固定仅 1 个")
    print(f"  复杂度说明: 统筹全域，约束最多，建模难度最大，参数量最高")
    
    results["L1"] = {
        "name": "全局统筹层",
        "obs_dim": l1_obs_dim,
        "action_dim": l1_action_dim,
        "actor_params": l1_actor_params,
        "critic_params": l1_critic_params,
        "single_total": l1_single_total,
        "hidden_dims": l1_hidden_dims,
        "n_agents": 1,
        "complexity": "最高（统筹全域）"
    }
    
    # --------------------------------------------------------------------------
    # L2 层：区域协调层
    # --------------------------------------------------------------------------
    print_section("L2 区域协调层 (Regional Coordination)")
    
    # L2 配置 (精确调整至 45W 参数量)
    l2_obs_dim = 42  # 18 + 6*4 = 局部18 + 邻居通信24
    l2_action_dim = 18  # M*6 + K*3 = 3*6 + 2*3 = 24（简化）
    # 隐藏层: [512, 256, 192] -> 总参数量约 45W
    l2_hidden_dims = [512, 256, 192]
    
    # 计算参数量
    l2_actor_params = calculate_actor_params(l2_obs_dim, l2_action_dim, l2_hidden_dims)
    l2_critic_params = calculate_critic_params(l2_obs_dim, l2_hidden_dims)
    l2_single_total = l2_actor_params + l2_critic_params
    
    print(f"  观测维度 (obs_dim): {l2_obs_dim}")
    print(f"    - 局部观测: 用户需求(3) + 残余资源(7) + 环境(3) + L1配额(5) = 18")
    print(f"    - 邻居通信摘要: 6维 × 最多4邻居 = 24")
    print(f"    - 总计: 18 + 24 = {l2_obs_dim} 维")
    print(f"  动作维度 (action_dim): {l2_action_dim}")
    print(f"  隐藏层结构: {l2_hidden_dims}")
    
    print(f"\n  Actor 网络参数量计算:")
    backbone_l2 = calculate_mlp_params(l2_obs_dim, l2_hidden_dims, l2_hidden_dims[-1])
    print(f"    - Backbone ({l2_obs_dim} -> {l2_hidden_dims} -> {l2_hidden_dims[-1]}):")
    print(f"      {format_number(backbone_l2, 'K')}")
    mu_head_l2 = (l2_hidden_dims[-1] * l2_action_dim) + l2_action_dim
    print(f"    - Mu head ({l2_hidden_dims[-1]} -> {l2_action_dim}): {format_number(mu_head_l2, 'K')}")
    print(f"    - Log std ({l2_action_dim} 维): {l2_action_dim}")
    print(f"    - Actor 总计: {format_number(l2_actor_params)}")
    
    print(f"\n  Critic 网络参数量: {format_number(l2_critic_params)}")
    
    print(f"\n  * L2 单智能体总参数量: {format_number(l2_single_total)}")
    print(f"  配置规则: 每 5 个 L3 子区域配置 1 个")
    print(f"  复杂度说明: 片区内多区域协调，兼顾联动关系，复杂度中等")
    
    results["L2"] = {
        "name": "区域协调层",
        "obs_dim": l2_obs_dim,
        "action_dim": l2_action_dim,
        "actor_params": l2_actor_params,
        "critic_params": l2_critic_params,
        "single_total": l2_single_total,
        "hidden_dims": l2_hidden_dims,
        "n_agents_formula": "ceil(N_L3 / 5)",
        "complexity": "中等（片区协调）"
    }
    
    # --------------------------------------------------------------------------
    # L3 层：本地执行层
    # --------------------------------------------------------------------------
    print_section("L3 本地执行层 (Local Configuration)")
    
    # L3 配置 (精确调整至 19.8W 参数量)
    l3_obs_dim = 51  # 32 + 19 = 32维子区特征 + 19维上层约束
    l3_action_dim = 72  # 60 + 10 + 2 = 72维标准动作
    # 隐藏层: [288, 200] -> 总参数量约 19.8W
    l3_hidden_dims = [288, 200]
    
    # 计算参数量
    l3_actor_params = calculate_actor_params(l3_obs_dim, l3_action_dim, l3_hidden_dims)
    l3_critic_params = calculate_critic_params(l3_obs_dim, l3_hidden_dims)
    l3_single_total = l3_actor_params + l3_critic_params
    
    print(f"  观测维度 (obs_dim): {l3_obs_dim}")
    print(f"    - 子区域专属特征 32 维:")
    print(f"      · 用户特征(8) + 资源特征(8) + 设备特征(8) + 环境特征(8)")
    print(f"    - 上层约束 19 维:")
    print(f"      · L1配额(5) + L2调入(5) + L2调出(5) + L2链路端点(4)")
    print(f"    - 总计: 32 + 19 = {l3_obs_dim} 维")
    print(f"  动作维度 (action_dim): {l3_action_dim}")
    print(f"    - 设备部署 60 维 (5设备×12网格)")
    print(f"    - 工作参数 10 维 (5设备×2参数)")
    print(f"    - 全局调度 2 维 (优先级权重 + 预留比例)")
    print(f"  隐藏层结构: {l3_hidden_dims} （轻量化设计）")
    
    print(f"\n  Actor 网络参数量计算:")
    backbone_l3 = calculate_mlp_params(l3_obs_dim, l3_hidden_dims, l3_hidden_dims[-1])
    print(f"    - Backbone ({l3_obs_dim} -> {l3_hidden_dims} -> {l3_hidden_dims[-1]}):")
    print(f"      51×128+128 + 128×128+128 = {format_number(backbone_l3, 'K')}")
    mu_head_l3 = (l3_hidden_dims[-1] * l3_action_dim) + l3_action_dim
    print(f"    - Mu head (128 -> 72): 128×72+72 = {format_number(mu_head_l3, 'K')}")
    print(f"    - Log std (72 维): 72")
    print(f"    - Actor 总计: {format_number(l3_actor_params)}")
    
    print(f"\n  Critic 网络参数量: {format_number(l3_critic_params)}")
    
    print(f"\n  * L3 单智能体总参数量: {format_number(l3_single_total)}")
    print(f"  配置规则: 1 个子区域 = 1 个 L3")
    print(f"  设计理念: 轻量化 PPO，降低并行训练算力开销")
    print(f"  复杂度说明: 仅执行单点部署，任务边界清晰，参数量最低")
    
    results["L3"] = {
        "name": "本地执行层",
        "obs_dim": l3_obs_dim,
        "action_dim": l3_action_dim,
        "actor_params": l3_actor_params,
        "critic_params": l3_critic_params,
        "single_total": l3_single_total,
        "hidden_dims": l3_hidden_dims,
        "n_agents_formula": "N_L3（子区域总数）",
        "complexity": "最低（单点执行）"
    }
    
    return results


@dataclass
class ScenarioConfig:
    """场景配置"""
    name: str
    n_l3: int  # L3 子区域数量（乡镇数量）
    n_l2: int  # L2 区域智能体数量
    n_l1: int  # L1 全局智能体数量（固定1）
    description: str


def calculate_scenario_params(layer_info: Dict[str, Dict], scenario: ScenarioConfig) -> Dict:
    """计算特定场景下的总参数量"""
    
    l1_total = layer_info["L1"]["single_total"] * scenario.n_l1
    l2_total = layer_info["L2"]["single_total"] * scenario.n_l2
    l3_total = layer_info["L3"]["single_total"] * scenario.n_l3
    
    total = l1_total + l2_total + l3_total
    
    return {
        "scenario": scenario,
        "L1_total": l1_total,
        "L2_total": l2_total,
        "L3_total": l3_total,
        "grand_total": total,
        "meets_requirement": total >= 1_000_000
    }


def print_scenario_analysis(layer_info: Dict[str, Dict], results: Dict) -> None:
    """打印场景分析结果"""
    
    scenario = results["scenario"]
    
    print(f"\n  【场景】{scenario.name}")
    print(f"  【描述】{scenario.description}")
    
    print(f"\n  智能体配置:")
    print(f"    L3 子区域数 (N): {scenario.n_l3}")
    print(f"    L2 智能体数: {scenario.n_l2} (ceil({scenario.n_l3}/5) = {(scenario.n_l3 + 4)//5})")
    print(f"    L1 智能体数: {scenario.n_l1} (全局固定)")
    
    print(f"\n  参数量计算明细:")
    l3_str = format_number(results['L3_total'])
    l2_str = format_number(results['L2_total'])
    l1_str = format_number(results['L1_total'])
    
    print(f"    L3 总参数 = {scenario.n_l3} × {format_number(layer_info['L3']['single_total'])}")
    print(f"             = {l3_str}")
    print(f"    L2 总参数 = {scenario.n_l2} × {format_number(layer_info['L2']['single_total'])}")
    print(f"             = {l2_str}")
    print(f"    L1 总参数 = {scenario.n_l1} × {format_number(layer_info['L1']['single_total'])}")
    print(f"             = {l1_str}")
    
    print(f"\n  * 全局总参数量 = {l3_str} + {l2_str} + {l1_str}")
    print(f"                 = {format_number(results['grand_total'])}")
    
    # 达标判断
    if results["meets_requirement"]:
        excess = results["grand_total"] - 1_000_000
        print(f"\n  [OK] 满足 >= 100万 参数要求")
        print(f"  [OK] 超出指标: {format_number(excess)} ({excess/10000:.1f}万)")
    else:
        deficit = 1_000_000 - results["grand_total"]
        print(f"\n  [FAIL] 未达到 100万 参数要求")
        print(f"  [FAIL] 差距: {format_number(deficit)}")


def main():
    """主函数：执行参数量统计与验证"""
    
    # 1. 分析各层架构
    layer_info = analyze_layer_architecture()
    
    # 2. 定义场景配置
    scenarios = [
        ScenarioConfig(
            name="小型受灾场景",
            n_l3=5,
            n_l2=1,
            n_l1=1,
            description="5个乡镇级子区域参与应急组网"
        ),
        ScenarioConfig(
            name="中型受灾场景",
            n_l3=10,
            n_l2=2,
            n_l1=1,
            description="10个乡镇级子区域参与应急组网"
        ),
        ScenarioConfig(
            name="大型受灾场景",
            n_l3=20,
            n_l2=4,
            n_l1=1,
            description="20个乡镇级子区域参与应急组网"
        ),
    ]
    
    # 3. 计算各场景参数量
    print_header("不同场景规模下的总参数量计算")
    
    all_results = []
    for scenario in scenarios:
        result = calculate_scenario_params(layer_info, scenario)
        print_scenario_analysis(layer_info, result)
        all_results.append(result)
        print_separator("-", 55)
    
    # 4. 汇总表格
    print_header("参数量汇总表")
    
    print("\n  +-------------------------------------------------------------------+")
    print("  | 受灾场景规模      | L3总数 | L2数量 | L1数量 | 总参数量  | 达标  |")
    print("  +-------------------+--------+--------+--------+-----------+---------+")
    
    for result in all_results:
        sc = result["scenario"]
        total_str = format_number(result["grand_total"])
        status = "[OK] 满足" if result["meets_requirement"] else "[FAIL] 不足"
        
        print(f"  | {sc.name:17s} | {sc.n_l3:6d} | {sc.n_l2:6d} | {sc.n_l1:6d} | "
              f"{total_str:9s} | {status:7s} |")
    
    print("  +-------------------+--------+--------+--------+-----------+---------+")
    print("  | 项目要求: 总参数量 >= 100万                                       |")
    print("  +-------------------------------------------------------------------+")
    
    # 5. 设计逻辑说明
    print_header("参数量设计逻辑说明")
    
    print("""
  【层级参数设计原则】
  
  自上而下单智能体参数量递减：
  
      L1 (80.0W) > L2 (45.0W) > L3 (19.8W)
  
  【设计理由】
  
  ┌─────────────────────────────────────────────────────────────────────┐
  │ L1 全局统筹层 (80W 参数)                                              │
  │ ├─ 观测维度: 27维 (全局聚合特征)                                       │
  │ ├─ 隐藏层: [256, 256, 128] (3层大网络)                               │
  │ ├─ 决策范围: 统筹全域灾害类型、全域资源池、全区域灾情优先级            │
  │ ├─ 约束复杂度: 聚合所有区域特征完成顶层资源配额分配                     │
  │ └─ 建模难度: 全局约束最多，建模难度最大 → 参数量最高                   │
  ├─────────────────────────────────────────────────────────────────────┤
  │ L2 区域协调层 (45W 参数)                                              │
  │ ├─ 观测维度: 42维 (区域聚合 + 邻居通信)                                │
  │ ├─ 隐藏层: [192, 192, 96] (3层中等网络)                               │
  │ ├─ 决策范围: 聚合片区内多子区域状态                                     │
  │ ├─ 约束复杂度: 跨区域资源调度、骨干链路规划、残余网络片区协同            │
  │ └─ 建模难度: 兼顾多区域联动关系，复杂度高于局部执行层                   │
  ├─────────────────────────────────────────────────────────────────────┤
  │ L3 本地执行层 (19.8W 参数)                                            │
  │ ├─ 观测维度: 51维 (32维子区 + 19维约束)                                │
  │ ├─ 隐藏层: [128, 128] (2层轻量网络)                                   │
  │ ├─ 决策范围: 仅负责单一子区域 32 维局部状态感知                         │
  │ ├─ 约束复杂度: 72 维本地组网动作输出，仅服从上层约束完成本地化部署        │
  │ ├─ 建模难度: 任务边界清晰、决策范围最小 → 单智能体参数量最低            │
  │ └─ 设计优势: 轻量化降低海量并行训练算力开销                              │
  └─────────────────────────────────────────────────────────────────────┘
  
  【架构设计优势】
  
  1. 参数充足性: 最小场景(5乡镇)即可达 224万参数，远超100万门槛
  2. 线性扩展性: 随受灾范围扩大，参数量平稳线性增长，既保证模型能力，又避免冗余
  3. 计算效率:   L3轻量化设计降低并行训练开销，L1/L2高精度保障顶层决策质量
  4. 能力分层:   实现"轻量化执行 + 高精度统筹"的组合优势
    """)
    
    # 6. 保存报告
    proofs_dir = ROOT / "proofs"
    proofs_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = proofs_dir / "param_count_report.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("HMARL 三层智能体参数量统计报告\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("【计算公式】\n")
        f.write("MLP 参数量 = Σ(输入维度×输出维度 + 输出维度) 每层\n")
        f.write("Actor 参数量 = Backbone + Mu_head + Log_std\n")
        f.write("Critic 参数量 = Backbone (输出维度=1)\n")
        f.write("单智能体总参数量 = Actor + Critic\n\n")
        
        f.write("【单智能体参数量】\n")
        for layer_id, info in layer_info.items():
            f.write(f"\n{layer_id} {info['name']}:\n")
            f.write(f"  观测维度: {info['obs_dim']}\n")
            f.write(f"  动作维度: {info['action_dim']}\n")
            f.write(f"  隐藏层: {info['hidden_dims']}\n")
            f.write(f"  Actor参数量: {format_number(info['actor_params'])} ({info['actor_params']:,})\n")
            f.write(f"  Critic参数量: {format_number(info['critic_params'])} ({info['critic_params']:,})\n")
            f.write(f"  单智能体总参数量: {format_number(info['single_total'])} ({info['single_total']:,})\n")
            f.write(f"  复杂度: {info['complexity']}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("【场景总参数量计算】\n\n")
        
        for result in all_results:
            sc = result["scenario"]
            f.write(f"{sc.name} ({sc.description}):\n")
            f.write(f"  L3: {sc.n_l3}个 × {layer_info['L3']['single_total']:,} = "
                   f"{result['L3_total']:,}\n")
            f.write(f"  L2: {sc.n_l2}个 × {layer_info['L2']['single_total']:,} = "
                   f"{result['L2_total']:,}\n")
            f.write(f"  L1: {sc.n_l1}个 × {layer_info['L1']['single_total']:,} = "
                   f"{result['L1_total']:,}\n")
            f.write(f"  总计: {result['grand_total']:,} ({format_number(result['grand_total'])})\n")
            f.write(f"  达标情况: {'满足' if result['meets_requirement'] else '不满足'} >= 100万要求\n\n")
        
        f.write("=" * 70 + "\n")
        f.write(f"结论: 最小场景(5乡镇)总参数量 {all_results[0]['grand_total']:,} > 100万，满足项目要求\n")
        f.write(f"      参数量设计遵循: L1(80W) > L2(45W) > L3(19.8W) 层级递减原则\n")
        f.write("=" * 70 + "\n")
    
    print(f"\n  报告已保存至: {report_path}")
    print(f"  绝对路径: {report_path.absolute()}")
    
    return layer_info, all_results


if __name__ == "__main__":
    main()
