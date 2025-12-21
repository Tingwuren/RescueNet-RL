# RescueNet-RL

面向灾区移动基站部署策略优化的强化学习 Baseline，实现了一个可复现的 PPO 训练流程。该项目以 Gymnasium 风格的环境和 PyTorch 实现的策略网络为核心，可在本地 `pytorch` Conda 环境中直接运行。

## 项目结构

```
RescueNet-RL/
├── data/                       # 极端灾害通信资源数据集
├── envs/                       # 灾区环境定义
├── models/                     # Actor-Critic 策略网络
├── planning/broadcast_arch...  # 广播/通信组网架构输出
├── algos/                      # PPO 算法实现
├── configs/default_config.py   # 统一超参/环境配置
├── train.py                    # 训练入口
├── eval.py                     # 评估与可视化
└── artifacts/                  # 训练输出（模型、日志、曲线、组网方案）
```

## 环境设定概述

- **空间建模**：10×10 网格表示灾区，候选部署点随机挑选 `candidate_sites` 个格点。
- **用户建模**：每次 reset 随机生成 `num_users` 个坐标，并按 `initial_outage_fraction` 标记断网状态。
- **动作空间**：离散动作索引到候选点列表；若重复部署或预算耗尽，给予 `invalid_action_penalty`。
- **奖励**：`reward = coverage_reward * newly_covered - deployment_cost`，按新增覆盖用户数加分，同时支付部署成本。
- **终止条件**：达到 `max_steps`、预算用尽或覆盖率达到 100%。

详细实现位于 `envs/disaster_cellular_env.py`，并在注释中说明可扩展到多类型基站或多智能体。若需要调度多制式通信/广播资源，请切换到 `envs/multimodal_comm_env.py`，该环境会读取 `data/scenarios.json` 中定义的极端灾害场景，包含≥4 种通信制式、广播方式及残余/无残余网络的切换逻辑。

## 数据集与组网架构

- `data/scenarios.json`：覆盖台风、洪水、地震等场景，每个场景包含通信制式 ≥4、广播方式 ≥2 的时间序列资源变化数据（可扩展）。
- `data/resource_dataset.py`：数据访问与校验工具，保障多制式指标满足 ≥4 的硬约束。
- `planning/broadcast_architecture.py`：基于场景数据自动生成“残余网络/无残余网络”双方案的智能广播与通信组网架构（含单用户理论带宽、资源利用率等指标），训练完成后会将方案导出至 `artifacts/broadcast_architecture_<scenario>.json`。

## 配置说明

所有实验配置集中在 `configs/default_config.py`：

- `env`：网格大小、用户数、候选点数量、奖励系数等。
- `model`：`hidden_sizes` 用于基础 PPO；`multimodal_hidden_sizes`（默认为 `[1024, 1024, 512, 512]`）驱动 >100 万参数的强化学习模型，满足“模型参数规模 ≥100 万”的指标，实际参数量 >2M。
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
  --eval-episodes 3
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
   - `artifacts/training_coverage_curve.png`：训练脚本根据每次评估 `avg_final_coverage` 自动绘制的曲线。

## 评估与可视化

若需要在训练完成后复现策略表现，可运行：

```bash
python eval.py \
  --checkpoint artifacts/ppo_policy.pt \
  --episodes 10
```

脚本会加载训练好的策略，针对同一环境配置随机生成的新灾情执行确定性 rollout，并给出平均奖励与最终覆盖率。若需要逐步观测部署情况，可追加 `--render` 输出以打印每个部署动作。训练期间生成的 `artifacts/training_coverage_curve.png` 仍由 `train.py` 自动输出，无需在评估阶段读取 `training_metrics.json`。

> **提示（多制式环境）**：联合通信/广播环境的策略在训练时依赖随机采样来探索巨大的组合动作空间。如果使用默认的贪心评估（`deterministic`），会因为始终选择同一组合动作而难以复现训练期表现。请在多制式场景评估时追加 `--stochastic-eval`，或在训练命令中使用 `--stochastic-eval` 让 `train.py` 的内部评估也采样动作：

```bash
python eval.py \
  --env-type multimodal \
  --scenario-name typhoon_residual \
  --checkpoint artifacts/ppo_policy.pt \
  --episodes 5 \
  --stochastic-eval
```

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

该脚本可快速验证动作空间、奖励和终止条件是否匹配预期。训练前建议先执行一次，确保环境与依赖安装完好。
若需启动多制式/广播一体化训练并完成至少 1 种极端场景的训练，可执行：

```bash
python train.py \
  --env-type multimodal \
  --scenario-name typhoon_residual \
  --total-timesteps 12000 \
  --stochastic-eval
```

该流程会：

1. 从数据集中加载指定场景；
2. 启动 `MultimodalPolicy`（参数量 >1M）训练；
3. 训练完成后在 `artifacts/` 中生成 `training_coverage_curve.png`、`ppo_policy.pt` 以及 `broadcast_architecture_typhoon_residual.json`，用于技术进展报告及专家评审。
   - `--stochastic-eval` 会让 `train.py` 在周期性评估时同样采用采样动作，避免“贪心动作重复部署”导致的覆盖率偏低。

## Web 仪表盘（Vue + FastAPI）

为了让非命令行用户也能操作训练和测试流程，新增了一个基于 FastAPI + Vue3 的前端界面：

1. **启动后端 API**
   ```bash
   conda activate pytorch
   pip install -r requirements.txt
   uvicorn server.api:app --reload --port 8000
   ```
   - `/api/train` 支持选择 `typhoon_residual`、`flood_no_residual`、`earthquake_residual` 等场景触发 PPO 训练；
   - `EventSource` (`/api/train/{run_id}/stream`) 会实时推送 episode/update/eval 事件，前端即可监控训练过程；
   - `/api/simulate` 接收自定义设备清单，载入训练好的策略，输出逐步组网策略和恢复状态。

2. **启动前端（Vue 3 + Vite）**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   - “灾害场景训练” 面板：选择台风/洪水/地震场景、配置训练步数、实时查看训练事件。
   - “自定义环境测试” 面板：自由添加受灾设备（坐标/需求/初始状态），并查看模型在该环境中的组网策略与设备恢复情况。

> 默认 `VITE_API_BASE=http://localhost:8000/api`，若后端端口不同，可通过 `.env.local` 覆写。
