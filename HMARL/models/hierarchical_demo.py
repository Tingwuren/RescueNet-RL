"""HMARL 三层智能体联调演示 (L1->L2->L3)

展示内容：
  L1 层：全局统筹 -> 输出 Nx5 配额矩阵 Q
  L2 层：区域调控 -> 输出迁移指令 + 跨区链路
  L3 层：本地配置 -> 输出 72 维动作 + 组网拓扑

用法：
    cd HMARL
    python models/hierarchical_demo.py [--scenario rainstorm|typhoon]
"""

from __future__ import annotations

import argparse
import numpy as np

# 设备名称定义
DEVICE_NAMES = [
    "应急基站",
    "便携广播网关", 
    "5G中继",
    "Mesh中继",
    "通信UAV",
]

LINK_TYPE_NAMES = ["卫星回传", "微波中继", "UAV中继"]


def print_header(title: str, width: int = 70) -> None:
    """打印带分隔线的标题"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_section(title: str) -> None:
    """打印小节标题"""
    print(f"\n>> {title}")
    print("-" * 50)


class L1GlobalAgentMock:
    """L1 全局统筹智能体 Mock"""
    
    def __init__(self, n_regions: int = 5):
        self.n_regions = n_regions
        self.n_devices = 5
    
    def process(self, disaster_type: str, inventory: np.ndarray,
                severity: np.ndarray, users: np.ndarray) -> dict:
        """L1 层处理：输入环境状态 -> 输出 Nx5 配额矩阵"""
        
        weights = severity / (severity.sum() + 1e-6)
        
        Q = np.zeros((self.n_regions, self.n_devices), dtype=np.int32)
        for j in range(self.n_devices):
            total = int(inventory[j])
            if total > 0:
                base_alloc = (weights * total).astype(int)
                remainder = total - base_alloc.sum()
                for k in range(int(remainder)):
                    idx = np.argsort(-severity)[k % self.n_regions]
                    base_alloc[idx] += 1
                Q[:, j] = base_alloc
        
        return {
            "quota_matrix": Q,
            "total_allocated": int(Q.sum()),
            "constraint_satisfied": all(Q.sum(axis=0) <= inventory)
        }


class L2RegionalAgentMock:
    """L2 区域调控智能体 Mock"""
    
    def __init__(self, n_regions: int = 5):
        self.n_regions = n_regions
        self.n_devices = 5
    
    def process(self, l1_quota: np.ndarray, region_states: list) -> dict:
        """L2 层处理：输入 L1配额 + 区域状态 -> 输出迁移指令 + 跨区链路"""
        
        n = self.n_regions
        adjusted_quota = l1_quota.copy()
        
        migrations = []
        
        for i in range(n):
            state = region_states[i]
            demand = state["user_total"] * state["severity"] * 0.001
            gap = demand - adjusted_quota[i].sum()
            
            if gap > 2:
                for j in range(n):
                    if j != i:
                        neighbor_state = region_states[j]
                        surplus = adjusted_quota[j].sum() - neighbor_state["user_total"] * neighbor_state["severity"] * 0.001
                        
                        if surplus > 2:
                            transfer_amount = min(int(gap), int(surplus), 2)
                            if transfer_amount > 0:
                                device_type = 0
                                if adjusted_quota[j, device_type] >= transfer_amount:
                                    migrations.append({
                                        "src": j,
                                        "tgt": i,
                                        "device": DEVICE_NAMES[device_type],
                                        "amount": transfer_amount
                                    })
                                    adjusted_quota[j, device_type] -= transfer_amount
                                    adjusted_quota[i, device_type] += transfer_amount
                                    break
        
        links = []
        for i in range(n):
            for j in range(i+1, n):
                if region_states[i]["severity"] > 0.6 and region_states[j]["severity"] > 0.6:
                    links.append({
                        "A": i,
                        "B": j,
                        "type": LINK_TYPE_NAMES[1],
                        "pos": f"区域{i}-区域{j}边界"
                    })
                    if len(links) >= 3:
                        break
            if len(links) >= 3:
                break
        
        return {
            "migrations": migrations,
            "links": links,
            "adjusted_quota": adjusted_quota
        }


class L3LocalAgentMock:
    """L3 本地配置智能体 Mock"""
    
    def __init__(self, n_grids: int = 12):
        self.n_grids = n_grids
        self.n_devices = 5
    
    def process(self, region_id: int, l2_adjusted_quota: np.ndarray) -> dict:
        """L3 层处理：输入配额 -> 输出 72维动作 + 组网拓扑"""
        
        deployment = np.zeros((self.n_devices, self.n_grids), dtype=np.int32)
        user_dist = np.random.dirichlet(np.ones(self.n_grids))
        
        for j in range(self.n_devices):
            available = int(l2_adjusted_quota[j])
            if available > 0:
                alloc = (user_dist * available).astype(int)
                remainder = available - alloc.sum()
                for k in range(int(remainder)):
                    idx = np.argsort(-user_dist)[k % self.n_grids]
                    alloc[idx] += 1
                deployment[j, :] = alloc
        
        work_params = np.array([
            [0.8, 0.7],
            [0.6, 0.8],
            [0.7, 0.6],
            [0.9, 0.5],
            [0.5, 0.9],
        ])
        
        global_params = np.array([0.85, 0.15])
        
        topology = {
            "nodes": [],
            "edges": [],
            "coverage": {}
        }
        
        for grid in range(self.n_grids):
            for j in range(self.n_devices):
                if deployment[j, grid] > 0:
                    topology["nodes"].append({
                        "id": f"R{region_id}-G{grid}-{j}",
                        "type": DEVICE_NAMES[j],
                        "grid": grid,
                        "count": int(deployment[j, grid])
                    })
        
        topology["coverage"] = {
            "comm": f"{min(95, 60 + int(deployment.sum() * 2))}%",
            "broadcast": f"{min(90, 50 + int(deployment[1].sum() * 5))}%"
        }
        
        return {
            "deployment": deployment,
            "work_params": work_params,
            "global_params": global_params,
            "topology": topology,
        }


def demo_hierarchical(scenario: str = "rainstorm") -> None:
    """HMARL L1->L2->L3 三层联调演示"""
    
    print_header("HMARL 三层智能体联调演示 (L1->L2->L3)")
    
    n_regions = 5
    disaster_name = "暴雨" if scenario == "rainstorm" else "台风风暴潮"
    
    print(f"\n[系统配置]")
    print(f"   场景: {disaster_name}")
    print(f"   区域数 (N): {n_regions}")
    print(f"   设备类型 (M): 5")
    print(f"   层级: L1(全局) -> L2(区域) -> L3(本地)")
    
    inventory = np.array([10, 8, 6, 12, 4], dtype=np.float32)
    severity = np.array([0.9, 0.7, 0.5, 0.3, 0.2], dtype=np.float32)
    users = np.array([5000, 3000, 2000, 1500, 800], dtype=np.float32)
    
    region_states = [
        {"id": i, "severity": severity[i], "user_total": users[i], 
         "road_pass": 0.4 + np.random.rand() * 0.4}
        for i in range(n_regions)
    ]
    
    # ============================================================
    # L1 层
    # ============================================================
    print_header("L1 层: 全局统筹智能体 (Global Coordination)")
    
    print("""
[输入特征定义]
  - 灾害类型: one-hot(3) [暴雨, 台风风暴潮, 滑坡]
  - 全局网格摘要(4): 区域数/24, 网格行列/24, 总面积/1000
  - 全局设备库存(5): [应急基站, 便携广播, 5G中继, Mesh中继, UAV]
  - 区域灾情严重度(N=5): [0.90, 0.70, 0.50, 0.30, 0.20]
  - 区域用户总数(N=5): [5000, 3000, 2000, 1500, 800]
  - 区域高优先级占比(N=5): [40%, 35%, 20%, 15%, 10%]
  - 观测维度总计: 12 + 3x5 = 27 维
    """)
    
    l1_agent = L1GlobalAgentMock(n_regions)
    l1_output = l1_agent.process(disaster_name, inventory, severity, users)
    Q = l1_output["quota_matrix"]
    
    print_section("L1 输出: Nx5 配额矩阵 Q (硬约束)")
    
    print("\n  [配额分配矩阵]")
    print("  " + "-" * 65)
    header = "区域 | " + " | ".join([d[:6] for d in DEVICE_NAMES]) + " | 合计"
    print(f"  {header}")
    print("  " + "-" * 65)
    
    for i in range(n_regions):
        cols = " | ".join([f"  {int(Q[i,j])}  " for j in range(5)])
        print(f"  区域{i} | {cols} | {int(Q[i].sum())}")
    
    print("  " + "-" * 65)
    col_sums = " | ".join([f"  {int(Q[:,j].sum())}  " for j in range(5)])
    print(f"  合计 | {col_sums} | {int(Q.sum())}")
    
    limits = " | ".join([f"  {int(inventory[j])}  " for j in range(5)])
    print(f"  上限 | {limits} | {int(inventory.sum())}")
    print("  " + "-" * 65)
    
    print("\n  [约束检查]")
    for j, name in enumerate(DEVICE_NAMES):
        alloc = int(Q[:, j].sum())
        limit = int(inventory[j])
        status = "OK" if alloc <= limit else "超限"
        print(f"     {name:12s}: {alloc}/{limit} [{status}]")
    
    print(f"\n  [OK] L1 层输出: 高优先级区域(区域0)获 {int(Q[0].sum())} 台, 低优先级(区域4)获 {int(Q[4].sum())} 台")
    
    # ============================================================
    # L2 层
    # ============================================================
    print_header("L2 层: 区域调控智能体 (Regional Coordination)")
    
    print("""
[输入特征定义]
  - 局部观测(18维):
    * 区域用户需求(3): 总人数, 高优先级占比, 需求强度
    * 区域残余资源(7): 公网带宽, 广播资源, 已部署5类设备
    * 区域环境(3): 灾情, 道路通行率, 电力恢复率
    * L1 初始配额(5): [基站, ..., UAV] 上限
  - 邻居通信摘要(6维 x 最多4邻居):
    * 邻居灾情, 用户归一化, 残余带宽, 资源缺口, 富余, 部署率
  - 观测维度总计: 18 + 6x4 = 42 维
    """)
    
    print_section("L2 输出: 区域间资源迁移 + 跨区链路规划")
    
    l2_agent = L2RegionalAgentMock(n_regions)
    l2_output = l2_agent.process(Q, region_states)
    
    # 迁移指令
    print("\n  [迁移指令 Mx[src, tgt, device]]")
    if l2_output["migrations"]:
        for i, mig in enumerate(l2_output["migrations"]):
            print(f"     指令 {i+1}: 区域{mig['src']} -> 区域{mig['tgt']}")
            print(f"              设备: {mig['device']} x {mig['amount']}台")
    else:
        print("     (无迁移需求 - 各区域资源相对平衡)")
    
    # 跨区链路
    print("\n  [跨区域链路 Kx[A, B, type, pos]]")
    if l2_output["links"]:
        for i, link in enumerate(l2_output["links"]):
            print(f"     链路 {i+1}: 区域{link['A']} <-> 区域{link['B']}")
            print(f"              类型: {link['type']}")
            print(f"              位置: {link['pos']}")
    else:
        print("     (无跨区链路 - 区域间通信需求较低)")
    
    # 调剂后配额对比
    print("\n  [配额调剂对比: L1配额 -> L2调剂后]")
    adjusted = l2_output["adjusted_quota"]
    
    print("\n  " + "-" * 55)
    for i in range(n_regions):
        l1_row = [int(Q[i,j]) for j in range(5)]
        l2_row = [int(adjusted[i,j]) for j in range(5)]
        diff = [l2_row[j] - l1_row[j] for j in range(5)]
        
        changes = []
        for j in range(5):
            if diff[j] != 0:
                changes.append(f"{DEVICE_NAMES[j][:4]}:{l1_row[j]}->{l2_row[j]}")
        
        if changes:
            change_str = ", ".join(changes)
        else:
            change_str = "(无变化)"
        
        print(f"  区域{i}: {change_str}")
    print("  " + "-" * 55)
    
    # ============================================================
    # L3 层
    # ============================================================
    print_header("L3 层: 本地配置智能体 (Local Configuration)")
    
    print("""
[输入特征定义]
  - 32维子区域专属特征:
    * 用户特征(8): 总人数, 高优先级占比, 需求强度, 集中度, 救援人员, 受灾人口, 指挥人员, 需求增长率
    * 资源特征(8): 5G-600M带宽, 卫星带宽, WiFi6, 短波, UAV带宽, 残余带宽, 总可用带宽, 预留带宽
    * 设备特征(8): 可用设备数, 平均电量, 发射功率, 故障率, 部署难度, 维护状态, 运输条件, 优先级
    * 环境特征(8): 灾情严重度, 地形复杂度, 道路通行率, 电力恢复率, 灾后时长, 邻居资源状态, 次生灾害风险, 救援进度
  - 上层约束(19维):
    * L1配额(5) + L2调入(5) + L2调出(5) + L2链路端点(4)
  - 观测维度总计: 32 + 19 = 51 维
    """)
    
    print_section("L3 输出: 72维动作向量 + 组网拓扑")
    
    demo_region = 0
    l3_agent = L3LocalAgentMock(n_grids=12)
    l3_output = l3_agent.process(demo_region, l2_output["adjusted_quota"][demo_region])
    
    print("""
  [72维动作向量结构]
    维度范围      内容                    形状
    --------------------------------------------------
    0-59    设备部署数量(5设备x12网格)     (5, 12)
    60-69   设备工作参数(5设备x2参数)      (5, 2)
            - 参数1: 发射功率调整比例
            - 参数2: 带宽分配比例
    70-71   全局调度参数                   (2,)
            - 参数1: 救援通信优先级权重
            - 参数2: 跨区域资源预留比例
    """)
    
    deployment = l3_output["deployment"]
    print(f"\n  [设备部署矩阵: 5设备 x 12网格 - 区域{demo_region}示例]")
    
    # 简化的热力图显示
    print("\n         " + " ".join([f"G{i:02d}" for i in range(12)]))
    print("       " + "-" * (4 * 12 + 3))
    
    for j in range(5):
        cells = []
        for grid in range(12):
            val = deployment[j, grid]
            if val == 0:
                cells.append("  . ")
            elif val == 1:
                cells.append("  + ")
            else:
                cells.append(f"  {val} ")
        print(f"  {DEVICE_NAMES[j][:6]} |" + "|".join(cells) + "|")
    
    print(f"\n  [工作参数配置]")
    params = l3_output["work_params"]
    for i, name in enumerate(DEVICE_NAMES):
        print(f"     {name:12s}: 功率={int(params[i,0]*100)}%, 带宽={int(params[i,1]*100)}%")
    
    print(f"\n  [全局调度参数]")
    gp = l3_output["global_params"]
    print(f"     救援通信优先级权重: {int(gp[0]*100)}%")
    print(f"     跨区域资源预留比例: {int(gp[1]*100)}%")
    
    # 组网拓扑
    topo = l3_output["topology"]
    print(f"\n  [组网拓扑 JSON]")
    print(f"     部署节点数: {len(topo['nodes'])} 个")
    print(f"     连接边数: {len(topo['edges'])} 条")
    print(f"     通信覆盖率: {topo['coverage']['comm']}")
    print(f"     广播覆盖率: {topo['coverage']['broadcast']}")
    
    if topo['nodes']:
        print(f"\n     节点示例:")
        for node in topo['nodes'][:3]:
            print(f"       {node['id']}: {node['type']} x{node['count']}")
        if len(topo['nodes']) > 3:
            print(f"       ... 还有 {len(topo['nodes'])-3} 个节点 ...")
    
    # ============================================================
    # 总结
    # ============================================================
    print_header("三层联调流程总结")
    
    print(f"""
[L1 全局统筹层]
  - 输入: 全局灾情摘要 (27维)
  - 决策: 全局资源配额
  - 输出: Nx5 配额矩阵 Q
    区域0配额: {int(Q[0].sum())}台  区域4配额: {int(Q[4].sum())}台
    [硬约束: 列和 <= 全局库存]

       |
       v 硬约束传递

[L2 区域调控层]
  - 输入: 区域聚合特征 + L1配额 + 邻居通信 (42维)
  - 决策: 区域间资源调剂 + 跨区链路规划
  - 输出: M条迁移指令 + K条跨区链路
    迁移: {len(l2_output['migrations'])}条指令  链路: {len(l2_output['links'])}条连接
    [软约束: 根据区域需求动态调剂]

       |
       v 约束 + 链路端点

[L3 本地配置层]
  - 输入: 32维子区特征 + 上层约束19维 = 51维
  - 决策: 设备部署位置 + 工作参数 + 组网拓扑
  - 输出: 72维动作向量 + 组网拓扑JSON
    部署: 60维矩阵  参数: 10维  全局: 2维
    通信覆盖: {topo['coverage']['comm']}  广播覆盖: {topo['coverage']['broadcast']}
    [最终输出: 可直接指导现场应急部署]

[数据流完整闭环]
  L1(全局配额) -> L2(区域调剂) -> L3(本地部署)
  每层输出作为下层输入约束，形成层级化决策链
    """)


def main() -> None:
    parser = argparse.ArgumentParser(description="HMARL 三层智能体联调演示")
    parser.add_argument(
        "--scenario",
        choices=["rainstorm", "typhoon"],
        default="rainstorm",
        help="选择灾害场景 (默认: 暴雨)"
    )
    args = parser.parse_args()
    
    demo_hierarchical(args.scenario)


if __name__ == "__main__":
    main()
