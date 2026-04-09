# Mahimahi 网络仿真功能说明

## 功能概述

在原有训练/测试功能基础上，新增 Mahimahi 网络仿真模块。通过回放 mahimahi 格式的 trace 文件，在前端可视化灾后通信链路的带宽容量随时间变化过程，并模拟在该链路条件下的发送速率曲线。

## 新增文件清单

```
server/mahimahi_manager.py          # Trace 解析与容量分析
data/traces/generate_traces.py      # Trace 文件生成脚本
data/traces/*.trace                 # 5 个预置 trace 文件
frontend/src/components/MahimahiSimulator.vue  # 前端仿真组件
demo_server.py                      # 独立演示服务器（无需 FastAPI 依赖）
```

## 修改文件清单

```
frontend/src/App.vue                # 新增"网络仿真"路由与导航入口
server/api.py                       # 新增 /api/mahimahi/* 接口
server/schemas.py                   # 新增 Mahimahi 相关数据模型
Dockerfile                          # 添加 mahimahi 构建依赖
docker-compose.yml                  # 添加 NET_ADMIN 权限和 traces 挂载
```

## Trace 文件说明

Mahimahi trace 格式：每行一个整数，表示一个 1500 字节包的传输时间点（毫秒）。trace 在 mahimahi 中自动循环。包的密度决定了每个时间窗口的链路容量。

预置 trace 对应以下灾后通信场景：

| 文件名 | 前端标签 | 场景说明 |
|--------|---------|---------|
| emergency-command.trace | 应急指挥中心链路 (10Mbps) | 恒定带宽，指挥中心稳定链路 |
| damaged-station.trace | 震后受损基站链路 | 带宽波动，受损基站信号不稳 |
| mobile-patrol.trace | 灾区巡查车载链路 | 移动场景，含基站切换和信号盲区 |
| flood-emergency.trace | 洪灾应急通信链路 | 低带宽应急通信 |
| temp-relay.trace | 临时中继站链路 | 突发型带宽，临时中继 |

如需添加新 trace，将符合格式的 `.trace` 文件放入 `data/traces/` 目录，并在 `server/mahimahi_manager.py` 的 `TRACE_DESCRIPTIONS` 中添加对应的中文标签即可。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/mahimahi/traces | 获取可用 trace 列表 |
| GET | /api/mahimahi/traces/{name} | 获取指定 trace 详情与容量序列 |
| POST | /api/mahimahi/simulate | 运行仿真，返回容量时间序列 |

POST /api/mahimahi/simulate 请求体：

```json
{
  "trace_name": "damaged-station",
  "duration_s": 60,
  "window_ms": 500
}
```

返回：

```json
{
  "trace_name": "damaged-station",
  "duration_s": 60,
  "window_ms": 500,
  "capacity": [
    {"time_s": 0.0, "value": 7.68},
    {"time_s": 0.5, "value": 9.12},
    ...
  ]
}
```

