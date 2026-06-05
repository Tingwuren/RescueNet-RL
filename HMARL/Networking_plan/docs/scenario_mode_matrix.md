# 2 场景 × 2 模式 组网方案对照表

| 灾害场景 | 场景 ID | 有残余网络 | 无残余网络 |
|---------|---------|-----------|-----------|
| 极端暴雨 | `extreme_rainstorm` | [outputs/rainstorm_with_residual/network_plan.json](../outputs/rainstorm_with_residual/network_plan.json) | [outputs/rainstorm_no_residual/network_plan.json](../outputs/rainstorm_no_residual/network_plan.json) |
| 超强台风 | `super_typhoon` | [outputs/typhoon_with_residual/network_plan.json](../outputs/typhoon_with_residual/network_plan.json) | [outputs/typhoon_no_residual/network_plan.json](../outputs/typhoon_no_residual/network_plan.json) |

## 场景参数差异

| 参数 | 极端暴雨 | 超强台风 |
|------|---------|---------|
| 残余形态 | 点式分布 `point_scattered` | 片状/局部全阻 `patch_blocked` |
| 基站退服率 | 30%–80% | 20%–60% |
| 道路通行率 | 50% | 70% |
| 特有指标 | 链路断裂率 35% | 倒杆率 10%–30%，局部全阻区 4 个 |
| 设计文档依据 | 表3-1 | 表3-2 |

## 双模式关键字段对比（实测）

| 场景 | 模式 | 残余复用 | 应急部署 | 主回传 | 部署优先级首项 |
|------|------|---------|---------|--------|---------------|
| extreme_rainstorm | with_residual | 32 | 16 | 5G_residual_link | activate_residual_bs |
| extreme_rainstorm | no_residual | 0 | 59 | Satellite_Ka | satellite_backhaul |
| super_typhoon | with_residual | 57 | 28 | 5G_residual_link | activate_residual_bs |
| super_typhoon | no_residual | 0 | 105 | Satellite_Ka | satellite_backhaul |

## 生成命令

```bash
cd metric2_network_architecture
python -m deployment.export_plan --all
```

## 验证命令

```bash
python validation/run_all_validation.py
```

验证归档：`proofs/dual_mode_matrix.txt`、`proofs/verification_summary.txt`
