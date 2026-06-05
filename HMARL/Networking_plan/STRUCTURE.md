# 指标二：应急广播网组网架构设计方案

## 考核要求

| 项 | 要求 |
|----|------|
| 交付物 | 应急广播网 **组网架构设计方案**（可配置、可执行、可验证） |
| 灾害场景 | **2 种**极端灾害：暴雨、台风 |
| 组网模式 | **有残余网络** / **无残余网络** 两种能力 |

## 根目录

```
应急算法流程/metric2_network_architecture/
```

## 目录结构

```
metric2_network_architecture/
├── STRUCTURE.md                    # 本文件
├── README.md                       # 架构方案使用说明
│
├── architecture/                   # 【核心】三层组网架构定义
│   ├── overview.md                 # 总体架构图说明（L1/L2/L3 职责）
│   ├── l1_global_layer.yaml        # L1 决策逻辑层：全局资源与优先级
│   ├── l2_fusion_layer.yaml        # L2 多制式融合层：5G/卫星/WiFi/短波
│   ├── l3_execution_layer.yaml     # L3 节点执行层：广播/救援/告警/回传
│   └── comm_modes.yaml             # 四类制式能力表（带宽/覆盖/时延/能耗）
│
├── scenarios/                      # 【必做】2 种极端灾害场景
│   ├── extreme_rainstorm.yaml      # 暴雨：点式残余、链路断裂、退服30%-80%
│   └── super_typhoon.yaml          # 台风：片状残余、局部全阻、倒杆10%-30%
│
├── network_modes/                  # 【必做】两种组网模式
│   ├── with_residual/              # 有残余网络：复用可用基站/链路
│   │   ├── mode_config.yaml        # 启用残余节点、负载迁移、补丁路由
│   │   ├── topology_template.json  # 拓扑模板：残余+应急设备混合
│   │   └── deploy_rules.yaml       # 部署规则：优先盘活残余
│   └── no_residual/                # 无残余网络：纯应急组网
│       ├── mode_config.yaml        # 卫星/短波/UAV/便携网关拉起
│       ├── topology_template.json  # 拓扑模板：机动节点+回传
│       └── deploy_rules.yaml       # 部署规则：快速广覆盖
│
├── fusion/                         # 广播通信融合配置（L2 落地）
│   ├── link_selector.py            # 多制式链路选择
│   ├── broadcast_path.py           # 广播路径 / 多跳树
│   └── policy_merge.py             # L1/L2/L3 策略合并为可执行方案
│
├── deployment/                     # 方案生成与分阶段执行
│   ├── parse_rl_output.py          # 解析指标一训练出的策略/checkpoint
│   ├── task_split.py               # 任务拆解：区域→节点→业务
│   ├── phased_deploy.yaml          # 组网方案生成 5 步流程（8.3.1 表 8.2）
│   └── export_plan.py              # 导出组网方案 JSON/表格（交付物）
│
├── validation/                     # 组网验证
│   ├── validate_rainstorm.py       # 暴雨场景 + 两种模式 冒烟
│   ├── validate_typhoon.py         # 台风场景 + 两种模式 冒烟
│   └── checklist.md                # 验收勾选表
│
├── outputs/                        # 【交付目录】生成的设计方案
│   ├── rainstorm_with_residual/
│   ├── rainstorm_no_residual/
│   ├── typhoon_with_residual/
│   └── typhoon_no_residual/
│       └── network_plan.json + 01–04 部署表
│
└── docs/                           # 设计说明文档（可贴中期报告）
    ├── design_principles.md        # 设计原则
    └── scenario_mode_matrix.md     # 2场景×2模式 对照表
```

## 2 场景 × 2 模式（必须凑齐 4 份方案）

| 灾害场景 | 有残余网络 | 无残余网络 |
|---------|-----------|-----------|
| 暴雨 `extreme_rainstorm` | `outputs/rainstorm_with_residual/` | `outputs/rainstorm_no_residual/` |
| 台风 `super_typhoon` | `outputs/typhoon_with_residual/` | `outputs/typhoon_no_residual/` |

生成命令（占位，实现后可用）：

```bash
cd metric2_network_architecture
python deployment/export_plan.py --scenario extreme_rainstorm --mode with_residual
python deployment/export_plan.py --scenario extreme_rainstorm --mode no_residual
python deployment/export_plan.py --scenario super_typhoon --mode with_residual
python deployment/export_plan.py --scenario super_typhoon --mode no_residual
```

## 与指标一的关系

| 指标一 | 指标二 |
|--------|--------|
| 训练 RL 策略、参数量≥100万 | **不训练**，用架构+配置+规则生成组网方案 |
| 至少 1 种场景训练 | **2 种场景**都要出方案 |
| 可不区分残余/无残余 | **必须**两种组网模式 |

指标二可**读取**指标一产物：`HMARL/checkpoints/` → `deployment/parse_rl_output.py`，没有 checkpoint 时先用 `architecture/*.yaml` + 默认规则生成方案，保证 4 份 `network_plan.json` 能出。

## 三层架构在本指标中的含义（设计方案，不是训练代码）

| 层 | 配置文件 | 干什么 |
|----|---------|--------|
| L1 | `architecture/l1_global_layer.yaml` | 全域优先级、资源配额、骨干拓扑 |
| L2 | `architecture/l2_fusion_layer.yaml` | 制式选择、链路、广播路径、带宽协调 |
| L3 | `architecture/l3_execution_layer.yaml` | 节点部署、广播激活、接入与回传 |

## 最小完成清单（按顺序做）

1. 写好 `architecture/` 三层 yaml + `comm_modes.yaml`  
2. 写好 `scenarios/` 暴雨、台风两份参数  
3. 写好 `network_modes/with_residual` 与 `no_residual` 各一套配置+拓扑模板  
4. `deployment/export_plan.py` 跑出 **4 个** `outputs/*/network_plan.json`  
5. `validation/checklist.md` 四项打勾 + `docs/scenario_mode_matrix.md`  

## 文件索引速查

| 要改什么 | 去哪个文件 |
|---------|-----------|
| 改 L1/L2/L3 架构定义 | `architecture/l*_*.yaml` |
| 改暴雨/台风参数 | `scenarios/*.yaml` |
| 改有残余组网逻辑 | `network_modes/with_residual/` |
| 改无残余组网逻辑 | `network_modes/no_residual/` |
| 改多制式融合 | `fusion/` |
| 接指标一策略 | `deployment/parse_rl_output.py` |
| 导出交付方案 | `deployment/export_plan.py` → `outputs/` |
| 验收 4 组合 | `validation/` + `outputs/` 四个目录 |
