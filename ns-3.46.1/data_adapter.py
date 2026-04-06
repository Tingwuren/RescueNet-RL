import json
import os

INPUT_FILE = "deployment.json"
OUTPUT_FILE = "scenario.txt"

# 类型映射: 0=USER(会动), 1=MACRO_ENB(不动,灾后死), 2=MANPACK_ENB(不动,灾前隐身灾后活)
TYPE_MAP = {"USER": 0, "MACRO_ENB": 1, "MANPACK_ENB": 2}

def convert():
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到 {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    nodes = data['nodes']
    
    with open(OUTPUT_FILE, 'w') as out:
        # 第一行写地图宽、高、节点总数
        out.write(f"{data['map_width']} {data['map_height']} {len(nodes)}\n")
        
        for n in nodes:
            t = TYPE_MAP.get(n['type'], 0)
            out.write(f"{n['id']} {t} {n['x']} {n['y']}\n")
            
    print(f"成功将 {len(nodes)} 个节点转换为 scenario.txt 供 NS-3 读取。")

if __name__ == "__main__":
    convert()
