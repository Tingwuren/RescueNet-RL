# 组网方案生成数据流（当前实现）

> 可视化版本：在浏览器打开 [network_plan_dataflow.html](./network_plan_dataflow.html) 可截图插入 Word。  
> 旧版验证报告中的数据流图**已过时**，请勿继续使用。

## 旧图问题

| 旧图表述 | 实际情况 |
|----------|----------|
| plan_builder 仅接收 L1 + L3 | 读取 **L1/L2/L3 + comm_modes** 全部 architecture |
| 无 scenarios / network_modes | **必须**读取 `scenarios/*.yaml` 与 `network_modes/{mode}/` |
| plan_builder 后直接 export | 中间还有 **grid_placement → topology_builder** |
| 输出仅 `network_plan.json` | 还有 **01–04 四张部署表** |

## 新数据流（Mermaid）

```mermaid
flowchart TB
  subgraph IN["① 配置输入"]
    direction LR
    A1["architecture/"] ~~~ A2["scenarios/"] ~~~ A3["network_modes/"] ~~~ A4["phased_deploy"] ~~~ A5["RL 可选"]
  end

  subgraph RUN["② 生成引擎 → ③ 五步流水线"]
    direction LR
    EP(["export_plan"]) --> PB["plan_builder"] --> S1["全局清单"] --> S2["区域调度"]
    S2 --> S3["设备点位"] --> S4["三级拓扑"] --> P{{plan}}
  end

  subgraph DEL["④ 导出 → ⑤ 交付物"]
    direction LR
    P --> E["交付导出"] --> J["JSON"] & T["01–04 表"]
    J --> O0["network_plan.json"]
    T --> O1["01"] & O2["02"] & O3["03"] & O4["04"]
  end

  IN --> PB
```

## 架构分层（概念图，可与实现图并存）

L1→L2→L3 描述**职责分层**，与代码加载路径不同：

```mermaid
flowchart LR
  L1["L1 决策逻辑层<br/>全局配额 · 优先级"]
  L2["L2 多制式融合层<br/>5G / 卫星 / WiFi / 短波"]
  L3["L3 节点执行层<br/>部署 · 广播 · 回传"]
  CM["comm_modes.yaml"]

  L1 ==> L2
  CM ==> L2
  L2 ==> L3
```

此概念图可保留在报告「架构设计」章节；**方案生成数据流**请用上一节的完整流程图。

## 代码锚点

| 步骤 | 文件 |
|------|------|
| 入口 | `deployment/export_plan.py` |
| 组装 | `deployment/plan_builder.py` → `build_network_plan()` |
| 点位 | `deployment/grid_placement.py` → `assign_grid_placements()` |
| 拓扑 | `deployment/topology_builder.py` → `build_topology()` |
| 表导出 | `deployment/export_deployment_docs.py` → `export_deployment_documents()` |
