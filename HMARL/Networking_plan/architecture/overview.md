# 应急广播网三层组网架构总览

## 架构分层

| 层级 | 名称 | 职责 | 配置文件 |
|------|------|------|----------|
| L1 | 决策逻辑层 | 全域优先级、资源配额、骨干拓扑 | `l1_global_layer.yaml` |
| L2 | 多制式融合层 | 5G/卫星/WiFi/短波链路选择与带宽协调 | `l2_fusion_layer.yaml` |
| L3 | 节点执行层 | 设备部署、广播激活、接入与回传 | `l3_execution_layer.yaml` |

## 数据流

```
灾害场景参数 → L1(全局配额矩阵 Q) → L2(区域迁移+跨区链路) → L3(72维动作+拓扑)
                    ↓                      ↓                        ↓
              硬约束传递            制式融合与链路规划          可执行组网方案 JSON
```

## 与设计文档对应关系

- **8.1** 组网部署总体框架 → 本目录三层 yaml
- **8.2** 算法输出解析与任务拆解 → `deployment/parse_rl_output.py`、`deployment/task_split.py`
- **8.3.1** 组网方案生成步骤（5 步 · 表 8.2）→ `deployment/phased_deploy.yaml`
- **8.4** 双模式差异化策略 → `network_modes/with_residual/`、`network_modes/no_residual/`

## 通信制式

四类制式能力定义见 `comm_modes.yaml`：5G_700MHz、Satellite_Ka、WiFi6、Shortwave_HF。
