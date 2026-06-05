# 指标一：强化学习应急通信资源配置算法

## 考核要求


| 项    | 要求                                |
| ---- | --------------------------------- |
| 算法   | 多智能体强化学习（PPO + L1/L2/L3）应急通信资源配置  |
| 训练场景 | **至少 1 种**极端灾害场景完成训练（建议先暴雨，再扩展台风） |
| 模型规模 | 参数量 **≥ 100 万**（需可统计、可导出证明）       |


## 根目录

```
应急算法流程/HMARL/
```

## 目录结构

```
HMARL/
├── STRUCTURE.md                    # 本文件
├── README.md                       # 训练启动说明
├── requirements.txt
│
├── configs/                        # 训练配置（指标一直接改这里）
│   ├── train_default.yaml          # PPO：γ、batch、episode、学习率
│   ├── model_scale.yaml            # 隐藏层128、区域数5 → 控制参数量≥100万
│   └── scenarios/
│       └── extreme_rainstorm.yaml  # 第1种极端场景：暴雨
│       └── super_typhoon.yaml      # 第2种极端场景：台风
│
├── data/                           # 数据集 → 环境状态
│   ├── loader.py                   # 读 cell_info / resource_profile / business_users
│   └── to_env_state.py             # 转成观测向量
│
├── env/                            # 仿真环境
│   ├── emergency_env.py            # Gym 环境：step/reset/reward
│   ├── grid_24x24.py               # 576 网格
│   ├── observation.py              # 32 维观测编码
│   ├── action.py                   # 72 维动作编码
│   └── agents/                     # 三层智能体（输入输出各不同）
│       ├── l1_global_agent.py
│       ├── l2_regional_agent.py
│       └── l3_local_agent.py
│
├── models/                         # 神经网络（参数量在这里凑够100万）
│   ├── common/mlp.py               # 128×128 MLP
│   ├── l1_global/actor.py + critic.py
│   ├── l2_regional/actor.py + critic.py
│   ├── l3_local/actor.py + critic.py
│   └── param_counter.py            # 【验收】统计总参数量，输出 ≥1e6 报告
│
├── algorithms/
│   └── ppo/
│       ├── ppo_trainer.py          # 训练主循环
│       ├── rollout_buffer.py
│       └── losses.py               # clip + value + entropy
│
├── rewards/
│   ├── composite_reward.py         # R = 0.4覆盖 + 0.3广播 + 0.2吞吐 - 0.1成本
│   └── components/                 # coverage / broadcast / throughput / cost
│
├── training/                       # 【指标一入口】
│   ├── train.py                    # 单命令启动训练
│   └── train_one_scenario.py       # 指定一种灾害场景训练（满足≥1种）
│
├── checkpoints/                    # 训练产物
│   └── {scenario_name}/            # 如 extreme_rainstorm/
│       ├── actor_l1.pt
│       ├── actor_l2.pt
│       ├── actor_l3.pt
│       └── train_log.json
│
└── proofs/                         # 【验收材料】
    ├── param_count_report.txt      # 打印各层参数 + 合计
    └── train_summary.md            # 场景名、episode、最终奖励曲线截图说明
```

## 与总仓库 `marl_emergency_resource` 的关系


| 本指标目录                                      | 总仓库对应                                    |
| ------------------------------------------ | ---------------------------------------- |
| `env/agents/*.py`                          | 同路径，指标一只管训练相关                            |
| `models/`                                  | 同路径，**param_counter.py 为本指标必做**          |
| `training/train_one_scenario.py`           | 对应 `training/train_hierarchical.py` 的简化版 |
| `configs/scenarios/extreme_rainstorm.yaml` | 至少启用 1 个场景文件                             |


指标一**不需要**完整实现：组网部署导出、双场景架构对比、无残余/有残余切换（这些归指标二）。

## 参数量 ≥100 万（怎么达标）

报告算法：5 个区域 × 每区 1 套 PPO（Actor+Critic，隐藏层 128），单套约 3.1 万 × 5 ≈ 15.5 万；**三层 L1+L2+L3 联合** 或 **加大区域数/隐藏层** 到合计 ≥100 万。

**落地做法（二选一，优先 A）：**

- **A**：`model_scale.yaml` 设 `n_regions: 5`，L1+L2+L3 每层独立 Actor+Critic，跑 `python models/param_counter.py` 出报告。
- **B**：隐藏层改为 256 或区域数改为 15+，直到 `total_params >= 1_000_000`。

验收命令（占位，实现后可用）：

```bash
cd HMARL
python models/param_counter.py
python training/train_one_scenario.py --scenario extreme_rainstorm
```

## 最小完成清单（按顺序做）

1. `data/loader.py` 能读一种暴雨场景数据
2. `env/emergency_env.py` + `agents/` 三层能 step
3. `models/` + `param_counter.py` 输出 ≥1000000
4. `training/train_one_scenario.py` 跑通 500 episode
5. `proofs/param_count_report.txt` + `checkpoints/extreme_rainstorm/` 留存

## 文件索引速查


| 要改什么          | 去哪个文件                                                           |
| ------------- | --------------------------------------------------------------- |
| 换训练场景         | `configs/scenarios/*.yaml` + `train_one_scenario.py --scenario` |
| 改观测/动作维       | `env/observation.py` `env/action.py`                            |
| 改 L1/L2/L3 逻辑 | `env/agents/l*_*.py`                                            |
| 改网络结构/参数量     | `models/` + `configs/model_scale.yaml`                          |
| 改奖励权重         | `rewards/composite_reward.py`                                   |
| 改 PPO         | `algorithms/ppo/`                                               |
| 验收参数量         | `models/param_counter.py` → `proofs/`                           |


