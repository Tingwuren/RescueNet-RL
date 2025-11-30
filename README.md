# RescueNet-RL

面向灾区移动基站部署策略优化的强化学习 Baseline，实现了一个可复现的 PPO 训练流程。该项目以 Gymnasium 风格的环境和 PyTorch 实现的策略网络为核心，可在本地 `pytorch` Conda 环境中直接运行。

## 项目结构

```
RescueNet-RL/
├── envs/                    # 灾区环境定义
├── models/                  # Actor-Critic 策略网络
├── algos/                   # PPO 算法实现
├── configs/default_config.py# 统一超参/环境配置
├── train.py                 # 训练入口
├── eval.py                  # 评估与可视化
└── artifacts/               # 训练输出（模型、日志、曲线）
```

## 环境设定概述

- **空间建模**：10×10 网格表示灾区，候选部署点随机挑选 `candidate_sites` 个格点。
- **用户建模**：每次 reset 随机生成 `num_users` 个坐标，并按 `initial_outage_fraction` 标记断网状态。
- **动作空间**：离散动作索引到候选点列表；若重复部署或预算耗尽，给予 `invalid_action_penalty`。
- **奖励**：`reward = coverage_reward * newly_covered - deployment_cost`，按新增覆盖用户数加分，同时支付部署成本。
- **终止条件**：达到 `max_steps`、预算用尽或覆盖率达到 100%。

详细实现位于 `envs/disaster_cellular_env.py`，并在注释中说明可扩展到多类型基站或多智能体。

## 配置说明

所有实验配置集中在 `configs/default_config.py`：

- `env`：网格大小、用户数、候选点数量、奖励系数等。
- `model`：MLP Actor-Critic 的隐层宽度。
- `ppo`：学习率、折扣 γ、GAE λ、Clip 系数、更新轮数、熵/价值损失权重等。
- `train`：总步数、每次 rollout 步数、日志/评估间隔、评估 episode 数、设备。
- `logging`：输出目录（默认 `artifacts/`）。

运行脚本时可通过 CLI 参数覆写关键配置，例如：

```bash
python train.py \
  --total-timesteps 50000 \
  --rollout-steps 512 \
  --log-interval 1 \
  --eval-interval 1 \
  --eval-episodes 5
```

## 训练流程

1. **构建环境**：`train.py` 根据配置分别实例化训练与评估环境，保证随机种子一致性。
2. **初始化策略**：`models/MLPActorCritic` 根据观测维度和动作维度创建共享体网络。
3. **滚动采样**：`algos/PPOTrainer` 以 `rollout_steps` 为单位采集 trajectories，记录 `obs/actions/rewards/dones/values`。
4. **优势计算**：使用 GAE(`gamma`, `gae_lambda`) 计算 `advantages` 与 `returns`。
5. **PPO 更新**：每次采样后执行 `update_epochs` 轮 mini-batch 更新，裁剪策略比率以稳定训练。
6. **日志与评估**：每个 `log_interval` 打印滚动平均奖励/覆盖率与 loss；`eval_interval` 触发确定性评估，记录“真实”覆盖表现。
7. **产出 Artifact**：
   - `artifacts/ppo_policy.pt`：训练后的策略参数。
   - `artifacts/training_metrics.json`：episode 奖励、覆盖率、时间步及配置快照。
   - `artifacts/training_coverage_curve.png`：由 `eval.py` 生成的覆盖率曲线。

## 评估与可视化

运行：

```bash
python eval.py \
  --checkpoint artifacts/ppo_policy.pt \
  --metrics artifacts/training_metrics.json \
  --episodes 10 \
  --coverage-plot artifacts/training_coverage_curve.png/
  --plot-skip 50
```

该脚本将：

1. 加载配置与训练好的策略权重。
2. 以确定性动作执行指定数量的 episode，输出平均奖励与最终覆盖率。
3. 根据 `training_metrics.json` 绘制训练过程中覆盖率随时间步的变化曲线。

可通过 `--render` 选项输出逐步部署情况，便于调试策略。

## 实验建议

- **收敛性**：推荐 `total_timesteps ≥ 50k`，并适当提高 `rollout_steps` 以获得更平稳的优势估计。
- **奖励设计**：若部署成本过高导致奖励偏负，可调低 `deployment_cost` 或提高 `coverage_reward`。
- **多基站类型**：可在 `DisasterCellularEnv` 中扩展动作编码（位置 + 类型），并在策略网络输出中使用多维离散建模。
- **多智能体扩展**：目前为集中式控制。可在未来将候选点拆分给多 UAV，配合多头策略或 MARL 框架进行扩展。

## 快速验证

若只需检查环境是否正常，可运行随机策略：

```bash
python - <<'PY'
from envs import DisasterCellularEnv
env = DisasterCellularEnv()
obs, info = env.reset()
print('obs_dim:', obs.shape[0], 'init_coverage:', info['coverage_ratio'])
for step in range(5):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(step, reward, info)
    if terminated or truncated:
        break
PY
```

该脚本可快速验证动作空间、奖励和终止条件是否匹配预期。训练前建议先执行一次，确保环境与依赖安装完好。***
