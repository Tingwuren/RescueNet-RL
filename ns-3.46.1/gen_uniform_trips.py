import os
import sys
import math
import random

# 尝试导入 sumolib
try:
    import sumolib
except ImportError:
    sys.exit("错误：找不到 'sumolib'。请确保已安装 SUMO 并设置了 SUMO_HOME 环境变量。")

# === 配置 ===
NET_FILE = "map.net.xml"
ROUTE_FILE = "routes.xml"
NODE_COUNT = 100  # 节点数量

def generate_uniform_routes():
    print(f"正在读取路网 {NET_FILE} ...")
    try:
        net = sumolib.net.readNet(NET_FILE)
    except Exception as e:
        print(f"读取路网失败: {e}")
        return

    # 1. 获取地图边界
    bbox = net.getBBoxXY()
    if len(bbox) == 2:
        x_min, y_min = bbox[0]
        x_max, y_max = bbox[1]
    else:
        x_min, y_min, x_max, y_max = bbox
    
    width = x_max - x_min
    height = y_max - y_min
    
    # 2. 计算网格
    grid_side = int(math.ceil(math.sqrt(NODE_COUNT)))
    if grid_side == 0: grid_side = 1
    step_x = width / grid_side
    step_y = height / grid_side
    
    trips = []
    
    print(f"正在计算 {NODE_COUNT} 个均匀分布的出生点 (无视道路类型)...")
    
    count = 0
    # 3. 遍历网格
    for r in range(grid_side):
        for c in range(grid_side):
            if count >= NODE_COUNT:
                break
            
            center_x = x_min + (c + 0.5) * step_x
            center_y = y_min + (r + 0.5) * step_y
            
            # 搜索半径 500m
            try:
                edges = net.getNeighboringEdges(center_x, center_y, r=500)
            except:
                continue

            # === [核心修改] ===
            # 不再检查 allows("passenger")
            # 只要有路 (edges不为空)，就直接取最近的一条
            if len(edges) > 0:
                # edges[0] 是最近的，格式为 (edge_object, distance)
                start_edge = edges[0][0]
                
                trips.append({
                    "id": str(count),
                    "depart": "0.00", 
                    "from": start_edge.getID(),
                })
                count += 1
            else:
                pass

    # 4. 分配终点并写入
    # 获取所有可能的边，不管是人行道还是铁路
    all_edges = net.getEdges()
    
    if not all_edges:
        print("错误：地图中没有任何边！")
        return

    with open(ROUTE_FILE, "w") as f:
        f.write("<routes>\n")
        
        # === [核心修改] ===
        # vClass="ignoring": 让车辆无视道路权限，哪里都能开
        f.write('    <vType id="car" vClass="ignoring" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="15" guiShape="passenger"/>\n')
        
        for trip in trips:
            end_edge = random.choice(all_edges)
            retry = 0
            while end_edge.getID() == trip["from"] and retry < 10:
                end_edge = random.choice(all_edges)
                retry += 1
            
            f.write(f'    <trip id="{trip["id"]}" type="car" depart="{trip["depart"]}" from="{trip["from"]}" to="{end_edge.getID()}" />\n')
            
        f.write("</routes>\n")
    
    print(f"成功生成 {len(trips)} 条路线到 {ROUTE_FILE} (vClass=ignoring)")
    
    if len(trips) < NODE_COUNT:
        print(f"警告：只生成了 {len(trips)} 个节点（原计划 {NODE_COUNT}）。请调整 NS-3 代码中的 g_nNodes。")

if __name__ == "__main__":
    generate_uniform_routes()
