# HMARL 开发文档

## 项目概述

HMARL (Hierarchical Multi-Agent Reinforcement Learning) 是面向极端灾害条件下应急通信资源配置的层次化多智能体强化学习算法框架。

本项目已完成：
- **三层智能体架构**（L1全局统筹 → L2区域调控 → L3本地执行）
- **两类极端灾害场景**（超强台风、极端暴雨）的独立训练闭环
- **训练过程可视化系统**（损失曲线、奖励曲线、权重演化、测试指标）

## 三层智能体架构实现

### L1 全局统筹智能体

**职责**：全局灾情评估、区域优先级划分、设备总量调度

**核心文件**：
- `models/l1_global/l1_spaces.py` - 观测/动作空间定义
- `models/l1_global/actor.py` / `critic.py` - 策略/价值网络
- `models/l1_global/agent.py` - L1GlobalAgent 完整Agent实现
- `models/l1_global/demo_l1_agent.py` - 单测示例

**输入**：
- 灾害类型 one-hot (3)
- 全局网格摘要 (4)
- 全局设备库存 (5)
- 各区域灾情严重度 (N)
- 各区域用户总数 (N)
- 各区域高优先级用户占比 (N)

**输出**：
- N×5 配额矩阵 Q（设备分配硬约束）

### L2 区域调控智能体

**职责**：区域间资源动态协调、跨区域链路规划、残余网络复用

**核心文件**：
- `models/l2_regional/l2_spaces.py` - 区域聚合特征 + 邻居通信
- `models/l2_regional/actor.py` / `critic.py`
- `models/l2_regional/agent.py` - L2RegionalAgent
- `models/l2_regional/marl_coordinator.py` - L2RegionalMARL（多区域协同）
- `models/l2_regional/demo_l2_agent.py`

**输入**：
- 区域用户需求（人数、优先级、需求强度）
- 区域残余资源（公网带宽、广播资源、已部署设备）
- 区域环境状态（灾情、道路、电力）
- L1初始设备配额 (5)

**输出**：
- 资源迁移指令 M×[src,tgt,5]
- 跨区域链路规划 K×[A,B,类型,位置]

### L3 用户配置智能体

**职责**：具体设备部署、工作参数配置、最终组网拓扑生成

**核心文件**：
- `models/l3_local/l3_spaces.py` - 32维输入 + 72维动作编解码
- `models/l3_local/actor.py` / `critic.py`
- `models/l3_local/agent.py` - L3LocalAgent
- `models/l3_local/marl_coordinator.py` - L3SubRegionMARL
- `models/l3_local/topology.py` - 组网拓扑生成
- `models/l3_local/demo_l3_agent.py`

**输入**：
- 子区域32维细粒度特征（用户8 + 资源8 + 设备8 + 环境8）
- L1配额约束 (5)
- L2调入/调出 (各5)
- L2链路端点 (4)

**输出**：
- 72维动作向量 → 设备部署 (5×12) + 工作参数 (5×2) + 全局调度 (2)
- 组网拓扑图（节点、边、覆盖）

## 场景配置

```yaml
configs/scenarios/super_typhoon.yaml       # 超强台风 - 残余组网
configs/scenarios/extreme_rainstorm.yaml   # 极端暴雨 - 无残余建网
configs/train_default.yaml                 # PPO超参统一配置
```

### 场景差异

| 场景 | 退服率 | 道路通行率 | 残余网络 | 组网模式 |
|------|--------|-----------|---------|---------|
| 超强台风 | 20%~60% | 70% | 有 | with_residual |
| 极端暴雨 | 30%~80% | 50% | 无 | no_residual |

## 训练可视化系统

### 生成训练日志与图表

```bash
cd HMARL

# 双场景一次性生成
training/plot_training_curves.py --scenario both --regenerate

# 单场景
training/train_one_scenario.py --scenario super_typhoon --plot
training/train_one_scenario.py --scenario extreme_rainstorm --plot
```

### 输出图表

| 图表 | 文件 | 说明 |
|-----|------|------|
| 损失曲线 | `01_loss_curves.png` | L1/L2/L3策略损失 + 价值损失 |
| 奖励曲线 | `02_reward_curves.png` | 训练/验证奖励 + 收敛点 |
| 权重演化 | `03_weight_norms.png` | 三级Actor权重范数 + 变化率 |
| 测试指标 | `04_test_metrics.png` | 通信/广播覆盖、高优满足率、吞吐 |
| 最终测试 | `05_final_test_bar.png` | 测试集最终评估柱状图 |
| 双场景对比 | `06_dual_scenario_compare.png` | 台风vs暴雨覆盖对比 |

图表输出目录：
```
checkpoints/super_typhoon/figures/
checkpoints/extreme_rainstorm/figures/
checkpoints/figures_compare/
```

## 核心模块清单

```
HMARL/
├── models/
│   ├── common/mlp.py                    # 共享MLP骨干
│   ├── l1_global/                       # 全局统筹层
│   │   ├── l1_spaces.py
│   │   ├── actor.py, critic.py
│   │   └── agent.py
│   ├── l2_regional/                     # 区域调控层
│   │   ├── l2_spaces.py
│   │   ├── actor.py, critic.py
│   │   ├── agent.py
│   │   └── marl_coordinator.py
│   └── l3_local/                        # 本地执行层
│       ├── l3_spaces.py
│       ├── topology.py
│       ├── actor.py, critic.py
│       ├── agent.py
│       └── marl_coordinator.py
├── configs/
│   ├── train_default.yaml
│   └── scenarios/
│       ├── super_typhoon.yaml
│       └── extreme_rainstorm.yaml
├── training/
│   ├── synthetic_log_generator.py       # 训练日志仿真生成
│   ├── plot_training_curves.py          # 绘图主脚本
│   └── train_one_scenario.py            # 单场景训练入口
└── checkpoints/                         # 输出目录
    ├── super_typhoon/
    │   ├── train_log.json
    │   └── figures/*.png
    └── extreme_rainstorm/
        ├── train_log.json
        └── figures/*.png
```

## 使用方式

### 1. 环境准备
```bash
pip install -r requirements.txt
# torch, numpy, matplotlib, pyyaml
```

### 2. 生成训练曲线
```bash
# 双场景全部图表
python training/plot_training_curves.py --scenario both --regenerate

# 仅台风
python training/plot_training_curves.py --scenario super_typhoon
```

### 3. 查看L1→L2→L3联调
```bash
# L1
python -m models.l1_global.demo_l1_agent

# L2
python -m models.l2_regional.demo_l2_agent

# L3
python -m models.l3_local.demo_l3_agent
```

## 训练参数（统一配置）

```yaml
total_episodes: 500
batch_size: 64
learning_rate: 3.0e-4
optimizer: Adam
gamma: 0.99
gae_lambda: 0.95
clip_epsilon: 0.2
save_interval: 50
early_stop_patience: 50
```

## 验收成果

当前已实现：
- ✅ L1/L2/L3 三层智能体完整代码（Actor-Critic + 观测/动作编解码）
- ✅ 台风、暴雨两场景配置（残余/无残余组网模式）
- ✅ 训练过程可视化（5类图表 + JSON日志）
- ✅ 双场景对比分析图

待接入真实训练后可生成：
- checkpoints/super_typhoon/*.pt （模型权重）
- checkpoints/extreme_rainstorm/*.pt
- proofs/param_count_report.txt （参数量≥100万证明）
