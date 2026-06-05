# 组网架构验收勾选表

| # | 验收项 | 验证方式 | 状态 |
|---|--------|----------|------|
| 1 | L1/L2/L3 三层架构配置完整 | `python validation/validate_architecture.py` | 运行后查看 proofs/architecture_check.txt |
| 2 | 四类通信制式 (>=4) | architecture/comm_modes.yaml | 同上 |
| 3 | 组网方案生成 5 步（8.3.1） | deployment/phased_deploy.yaml | 同上 |
| 4 | 暴雨场景 + 双模式 | `python validation/validate_rainstorm.py` | proofs/scenario_rainstorm.txt |
| 5 | 台风场景 + 双模式 | `python validation/validate_typhoon.py` | proofs/scenario_typhoon.txt |
| 6 | 4 套 outputs 交付目录 | `python -m deployment.export_plan --all` | 每套含 network_plan.json + 01–04 表 |
| 7 | 残余/无残余字段可区分 | `python validation/run_all_validation.py` | proofs/dual_mode_matrix.txt |

一键验收：

```bash
cd metric2_network_architecture
python validation/run_all_validation.py
```
