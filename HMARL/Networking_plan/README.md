# 指标二：应急广播网组网架构

- 目录说明：[STRUCTURE.md](STRUCTURE.md)
- **指标验证报告（可粘贴中期材料）**：[docs/metric_verification.md](docs/metric_verification.md)
- 2×2 场景模式对照：[docs/scenario_mode_matrix.md](docs/scenario_mode_matrix.md)

## 目标

2 种灾害场景 × 有/无残余网络，共 4 套 outputs 交付目录（每套含 `network_plan.json` + 01–04 部署表）。

## 快速开始

```bash
cd metric2_network_architecture

# 导出 4 份 network_plan.json
python -m deployment.export_plan --all

# 一键验证 + 生成 proofs/
python validation/run_all_validation.py
```

## 单项命令

```bash
python -m deployment.export_plan --scenario extreme_rainstorm --mode with_residual
python -m deployment.export_plan --scenario extreme_rainstorm --mode no_residual
python -m deployment.export_plan --scenario super_typhoon --mode with_residual
python -m deployment.export_plan --scenario super_typhoon --mode no_residual

python validation/validate_architecture.py
python validation/validate_rainstorm.py
python validation/validate_typhoon.py
```

## 交付物

| 类型 | 路径 |
|------|------|
| 组网方案 | `outputs/*/`（4 套：JSON + 01–04 部署表） |
| 验证证明 | `proofs/*.txt` |
| 验收清单 | `validation/checklist.md` |

## 与指标一关系

可选读取 `HMARL/checkpoints/` 训练产物增强方案（`deployment/parse_rl_output.py`）；无 checkpoint 时按架构配置与部署规则独立生成方案。
