# HMARL — 层次化多智能体强化学习应急通信资源配置

> Hierarchical Multi-Agent Reinforcement Learning

- 目录说明：[STRUCTURE.md](STRUCTURE.md)
- 开发说明：[DEVELOPMENT.md](DEVELOPMENT.md)
- RescueNet 桥接细节：[rescuenet/README.md](rescuenet/README.md)
- 目标：≥1 种极端灾害场景训练完成，模型参数 ≥100 万

## 环境准备

推荐使用仓库根目录虚拟环境（含 `gymnasium`、`torch` 等）：

```bash
cd /mnt/data0/root/Projects/RescueNet-RL
source .venv/bin/activate
pip install -r requirements.txt

cd HMARL
pip install -r requirements.txt   # pyyaml 等（可选）
```

---

## 方式 A：RescueNet-RL 真实训练（推荐）

调用仓库根目录的 `HMARLTrainer` / `MultiModalCommEnv`，产物落在 `checkpoints/<场景>/`。

### 目录结构

```text
checkpoints/super_typhoon/
├── figures/          # 01–04 训练曲线
├── weights/          # L1.pt, L2.pt, L3.pt
├── train_log.json
├── training_metrics.json
├── policy_meta.json
├── run_summary.json
└── broadcast_architecture_typhoon_residual.json
```

### 训练

训练开始前会打印**运行环境说明**（`typhoon_residual`、空间网格 12、1500 用户等）；每个 PPO update 后会做一次**测试**，打印 L1/L2/L3 输入输出（随训练进度逐渐变好）。

**一键脚本（超强台风）：**

```bash
cd HMARL
./run_rescuenet_train_super_typhoon.sh
# 训练产物在 checkpoints/super_typhoon/
```

等价命令：

```bash
cd /mnt/data0/root/Projects/RescueNet-RL/HMARL
source ../.venv/bin/activate

python rescuenet/train.py \
  --scenario super_typhoon \
  --total-timesteps 50000 \
  --rollout-steps 1024 \
  --log-interval 1 \
  --eval-interval 5 \
  --eval-episodes 3 \
  --hierarchy-report-interval 1
```

| 参数 | 说明 |
|------|------|
| `--scenario` | HMARL 场景别名，见 `rescuenet/scenarios.yaml`（`super_typhoon` → `typhoon_residual`） |
| `--log-interval 1` | 每次 PPO 更新打印 loss / 覆盖率 |
| `--hierarchy-report-interval 1` | 每次更新后打印 L1/L2/L3 I/O；设为 `0` 可关闭 |
| `--plot-seed` | 可选，控制 `train_log` 绘图轻微随机抖动 |

快速试跑示例：

```bash
python rescuenet/train.py \
  --scenario super_typhoon \
  --total-timesteps 5000 \
  --rollout-steps 1024 \
  --log-interval 1 \
  --hierarchy-report-interval 1
```

### 极端暴雨场景（一键脚本，规格与台风一致）

对应 `extreme_rainstorm` → `flood_no_residual`（洞庭湖易涝区、点式损毁、无残余独立建网）。测试流程与台风相同：汇总观测 → RL rollout → L1/L2/L3 → Networking_plan 组网方案。

**训练：**

```bash
cd HMARL
./run_rescuenet_train_extreme_rainstorm.sh
# 训练产物在 checkpoints/extreme_rainstorm/
```

**测试：**（交付物规格与台风相同，场景为 `rainstorm_*`）

```bash
cd HMARL
./run_rescuenet_test_extreme_rainstorm.sh
# 默认 checkpoints/extreme_rainstorm_best
./run_rescuenet_test_extreme_rainstorm.sh checkpoints/extreme_rainstorm
./run_rescuenet_test_extreme_rainstorm.sh checkpoints/extreme_rainstorm_best --skip-eval
```

等价命令：

```bash
python rescuenet/train.py --scenario extreme_rainstorm --log-interval 1 --hierarchy-report-interval 1
python rescuenet/test_checkpoint.py \
  --checkpoint-dir checkpoints/extreme_rainstorm_best \
  --scenario-alias extreme_rainstorm
```

> **权重说明**：`flood_no_residual` 与 `typhoon_residual` 的观测/动作维度不同，不能共用台风 `weights/`。
> 真实 RL rollout 需先完成暴雨训练（`checkpoints/extreme_rainstorm/weights/`），再同步到 `extreme_rainstorm_best`；
> 在权重未就绪时，测试脚本会自动回退为演示性 rollout 指标，其余输出流程与台风一致。
> 测试完成后交付物位于 `checkpoints/extreme_rainstorm_best/deliverables/` 与 `Networking_plan/outputs/rainstorm_*/`。

### 测试已训练 checkpoint（`super_typhoon_best`）

先打印环境说明 + **L1/L2/L3 完整 I/O**（风格与验收截图一致）；默认再跑 RescueNet 环境 rollout 并输出 `avg_reward` / `avg_coverage`。

**一键脚本（超强台风）：**

```bash
cd HMARL
./run_rescuenet_test_super_typhoon.sh
# 默认 checkpoints/super_typhoon_best
./run_rescuenet_test_super_typhoon.sh checkpoints/super_typhoon
./run_rescuenet_test_super_typhoon.sh checkpoints/super_typhoon_best --skip-eval
```

等价命令：

```bash
cd HMARL

# 环境说明 + L1/L2/L3 + rollout 评估
python rescuenet/test_checkpoint.py \
  --checkpoint-dir checkpoints/super_typhoon_best

# 仅层次化 I/O，不跑 RescueNet rollout
python rescuenet/test_checkpoint.py \
  --checkpoint-dir checkpoints/super_typhoon_best \
  --skip-eval
```

### 验证权重（仅指标）

```bash
python rescuenet/validate.py --scenario super_typhoon --episodes 5
```

### 其他工具

```bash
# 从已有 training_metrics 重新生成 01–04 图
python rescuenet/plot_only.py --scenario super_typhoon_best

# 从 train_log 补齐 JSON bundle（training_metrics、policy_meta 等）
python rescuenet/sync_checkpoint_bundle.py \
  --checkpoint-dir checkpoints/super_typhoon_best \
  --scenario-alias super_typhoon
```

---

## 方式 B：HMARL 自研栈（仿真日志 / 演示）

使用 `HMARL/models`、`training/` 下的 L1/L2/L3 演示与合成训练曲线，**不**调用 RescueNet 根目录训练器。

```bash
cd HMARL
pip install -r requirements.txt

# 单场景训练日志 + 绘图（台风 / 暴雨）
python training/train_one_scenario.py --scenario super_typhoon --plot
python training/train_one_scenario.py --scenario extreme_rainstorm --plot

# 一次性生成双场景全部曲线
python training/plot_training_curves.py --scenario both --regenerate

# L1 / L2 / L3 单层演示
python -m models.l1_global.demo_l1_agent --scenario typhoon
python -m models.l2_regional.demo_l2_agent
python -m models.l3_local.demo_l3_agent

# 三层联调演示
python models/hierarchical_demo.py --scenario typhoon
```

图表输出：`checkpoints/super_typhoon/figures/`、`checkpoints/extreme_rainstorm/figures/`

**Linux 中文显示为方框时**：安装 `fonts-noto-cjk`（`sudo apt-get install -y fonts-noto-cjk`），然后重新绘图，例如 `python rescuenet/plot_only.py --scenario extreme_rainstorm`。

---

## 两种方式的关系

| | 方式 A `rescuenet/` | 方式 B `training/` |
|--|---------------------|---------------------|
| 训练代码 | 仓库根 `train.py` + `HMARLTrainer` | `training/train_one_scenario.py` |
| 权重 | `weights/L1.pt` 等 | `checkpoints/*/actor_*.pt` |
| 层次 I/O 打印 | `rescuenet/hierarchy_report.py` | `models/hierarchical_demo.py` |
| 互不干扰 | 是 | 是 |

日常验收与对接 RescueNet 环境请用 **方式 A**；文档与参数量演示可用 **方式 B**。
