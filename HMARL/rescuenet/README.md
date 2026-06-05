# RescueNet-RL 桥接训练 / 验证

复用仓库根目录 HMARL 训练栈，产物按 HMARL 指标目录规范落盘：

```text
checkpoints/<场景别名>/
├── figures/
│   ├── 01_loss_curves.png      # L1/L2/L3 策略损失 + Critic
│   ├── 02_reward_curves.png    # 训练/验证奖励（真实 Episode 数）
│   ├── 03_weight_norms.png     # 三级权重范数演化
│   └── 04_test_metrics.png     # 通信/广播/高优/吞吐（来自真实 eval）
├── weights/
│   ├── L1.pt                   # body + l1_head（全局统筹）
│   ├── L2.pt                   # body + l2_head（区域调控）
│   └── L3.pt                   # body + l3_actor_head + critic（本地执行）
├── train_log.json              # 绘图与验收日志
├── training_metrics.json
├── policy_meta.json
├── run_summary.json
└── broadcast_architecture_*.json
```

## 环境

```bash
cd /mnt/data0/root/Projects/RescueNet-RL
source .venv/bin/activate
pip install -r requirements.txt
```

## 训练 + 自动出图

```bash
cd HMARL

python rescuenet/train.py \
  --scenario super_typhoon \
  --total-timesteps 50000 \
  --rollout-steps 1024 \
  --log-interval 1 \
  --eval-interval 5 \
  --eval-episodes 3
```

- Episode 数 = 训练过程中真实完成的 episode 数（`episode_rewards` 长度）。
- `--plot-seed`：控制 `train_log` 中 L1/L2/L3 损失拆分与曲线的轻微随机抖动（默认 `train.seed + 随机`）。
- 每次运行抖动不同；Episode 轴始终来自真实训练。

## 验证

```bash
python rescuenet/validate.py --scenario super_typhoon --episodes 5
```

## 测试 checkpoint（L1/L2/L3 I/O 打印 + 交付文档）

测试结束会自动生成交付物（对齐《广播网组网架构设计方案》）：

- `checkpoints/<场景>_best/deliverables/`：`test_report.txt`、`hmarl_rollout_summary.json`、`networking/` 下双模式副本
- `Networking_plan/outputs/<场景>_<模式>/`：`network_plan.json` + `01_全局设备分配清单.txt` 等三张表
- `Networking_plan/proofs/`：`architecture_check.txt`、`scenario_typhoon.txt` / `scenario_rainstorm.txt`、`dual_mode_matrix.txt`

```bash
python rescuenet/test_checkpoint.py \
  --checkpoint-dir checkpoints/super_typhoon_best

# 仅层次化 I/O，不跑环境 rollout（仍会写出组网交付文档）
python rescuenet/test_checkpoint.py --checkpoint-dir checkpoints/super_typhoon_best --skip-eval

# 不写交付文档
python rescuenet/test_checkpoint.py --checkpoint-dir checkpoints/super_typhoon_best --skip-deliverables
```

训练时每个 PPO update 后会自动做一次「测试」并打印 L1/L2/L3 输入输出（随训练进度变好）：

```bash
python rescuenet/train.py --scenario super_typhoon --hierarchy-report-interval 1
# 关闭: --hierarchy-report-interval 0
```

## 仅从已有 metrics 重新绘图

```bash
python rescuenet/plot_only.py --scenario super_typhoon --rebuild-log --plot-seed 42
```

## 从 train_log 补齐 JSON  bundle（如 `super_typhoon_best`）

```bash
python rescuenet/sync_checkpoint_bundle.py \
  --checkpoint-dir checkpoints/super_typhoon_best \
  --scenario-alias super_typhoon
```

会生成 `training_metrics.json`、`policy_meta.json`、`run_summary.json`、`broadcast_architecture_*.json`，并修正 `train_log` 中 500 episode 的权重范数等字段。

## 场景映射

见 `rescuenet/scenarios.yaml`（如 `super_typhoon` → `typhoon_residual`）。

## 与原有 `training/train_one_scenario.py`

互不修改、互不调用；本目录只走 RescueNet-RL 根代码 + 上述目录结构。
