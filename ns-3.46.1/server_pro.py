#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应急通信数字孪生平台 - 后端服务
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import sqlite3
import json
import os
import time
import threading
from contextlib import asynccontextmanager

DB_FILE = "simulation_history.db"
TRACE_FILE = "trace.json"

# ============================================================================
# 数据库初始化
# ============================================================================

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # 实验记录表
    c.execute('''CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        date TEXT NOT NULL,
        duration REAL DEFAULT 0,
        total_nodes INTEGER DEFAULT 0,
        disaster_time REAL DEFAULT 0,
        frames INTEGER DEFAULT 0,
        notes TEXT
    )''')

    # 帧数据表
    c.execute('''CREATE TABLE IF NOT EXISTS frame_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        exp_id INTEGER NOT NULL,
        frame_idx INTEGER NOT NULL,
        time REAL NOT NULL,
        tp REAL DEFAULT 0,
        loss REAL DEFAULT 0,
        disaster INTEGER DEFAULT 0,
        data_json TEXT,
        FOREIGN KEY (exp_id) REFERENCES experiments (id)
    )''')

    # 节点统计表
    c.execute('''CREATE TABLE IF NOT EXISTS node_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        exp_id INTEGER NOT NULL,
        node_type INTEGER NOT NULL,
        total_count INTEGER DEFAULT 0,
        FOREIGN KEY (exp_id) REFERENCES experiments (id)
    )''')

    # 索引
    c.execute('CREATE INDEX IF NOT EXISTS idx_frame_exp ON frame_data(exp_id)')

    conn.commit()
    conn.close()
    print("[DB] 数据库初始化完成: " + DB_FILE)

# ============================================================================
# 数据导入
# ============================================================================

def import_trace():
    if not os.path.exists(TRACE_FILE):
        print("[IMPORT] trace.json 不存在")
        return None

    print("[IMPORT] 发现 trace.json，正在导入...")

    try:
        with open(TRACE_FILE, 'r') as f:
            content = f.read()

        content = content.strip()
        if content.endswith(','): content = content[:-1]
        if not content.startswith('['): content = '[' + content
        if not content.endswith(']'): content = content + ']'

        frames = json.loads(content)
        if not frames:
            print("[IMPORT] 数据为空")
            return None

        first = frames[0]
        node_count = len(first.get('nodes', []))

        disaster_time = 0
        for f in frames:
            if f.get('disaster', 0) == 1:
                disaster_time = f.get('time', 0)
                break

        duration = frames[-1].get('time', 0) if frames else 0

        # 统计节点类型
        node_types = {}
        for node in first.get('nodes', []):
            t = node[1]
            node_types[t] = node_types.get(t, 0) + 1

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()

        date_str = time.strftime("%Y-%m-%d %H:%M:%S")
        c.execute('''INSERT INTO experiments (name, date, duration, total_nodes, disaster_time, frames)
                    VALUES (?, ?, ?, ?, ?, ?)''',
                 (f"演练_{int(time.time())}", date_str, duration, node_count, disaster_time, len(frames)))
        exp_id = c.lastrowid

        # 节点统计
        for node_type, count in node_types.items():
            c.execute('''INSERT INTO node_stats (exp_id, node_type, total_count)
                        VALUES (?, ?, ?)''', (exp_id, node_type, count))

        # 帧数据
        for idx, frame in enumerate(frames):
            data_json = json.dumps({'nodes': frame.get('nodes', []), 'links': frame.get('links', [])})
            c.execute('''INSERT INTO frame_data (exp_id, frame_idx, time, tp, loss, disaster, data_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?)''',
                     (exp_id, idx, frame.get('time', 0), frame.get('tp', 0),
                      frame.get('loss', 0), frame.get('disaster', 0), data_json))

        conn.commit()
        conn.close()

        print(f"[IMPORT] 成功! 实验ID: {exp_id}, 帧数: {len(frames)}, 节点数: {node_count}")
        os.remove(TRACE_FILE)
        return exp_id

    except Exception as e:
        print(f"[IMPORT] 失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# 文件监控
# ============================================================================

class Watcher:
    def __init__(self):
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print("[WATCHER] 启动文件监控")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        print("[WATCHER] 停止文件监控")

    def _loop(self):
        last_mtime = 0
        while self.running:
            try:
                if os.path.exists(TRACE_FILE):
                    mtime = os.path.getmtime(TRACE_FILE)
                    if mtime != last_mtime and last_mtime > 0:
                        time.sleep(0.5)
                        import_trace()
                        last_mtime = 0
                    else:
                        last_mtime = mtime
            except Exception as e:
                print(f"[WATCHER] 错误: {e}")
            time.sleep(1)

# ============================================================================
# FastAPI 应用
# ============================================================================

init_db()
watcher = Watcher()

@asynccontextmanager
async def lifespan(app: FastAPI):
    import_trace()
    watcher.start()
    yield
    watcher.stop()

app = FastAPI(title="应急通信数字孪生平台 API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"])
app.mount("/ns3-native", StaticFiles(directory=os.path.dirname(os.path.abspath(__file__)), html=True), name="ns3-native")

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/")
async def root():
    return {
        "name": "应急通信数字孪生平台 API",
        "version": "2.0.0",
        "description": "支持多实验管理和历史回放"
    }

@app.get("/api/health")
async def health():
    return {"status": "ok", "db_exists": os.path.exists(DB_FILE)}

# ========================================================================
# 实验管理 API
# ========================================================================

@app.get("/api/experiments")
async def get_experiments():
    """获取所有实验列表"""
    conn = get_db()

    experiments = []
    for row in conn.execute('''
        SELECT e.*, GROUP_CONCAT(ns.node_type || ':' || ns.total_count) as node_stats
        FROM experiments e
        LEFT JOIN node_stats ns ON e.id = ns.exp_id
        GROUP BY e.id
        ORDER BY e.id DESC
    ''').fetchall():
        exp = dict(row)
        # 解析节点统计
        if exp.get('node_stats'):
            stats = {}
            for stat in exp['node_stats'].split(','):
                if ':' in stat:
                    t, c = stat.split(':')
                    type_names = {0: "用户", 1: "宏基站", 2: "背负式", 3: "小型基站", 4: "中继"}
                    stats[type_names.get(int(t), f"类型{t}")] = int(c)
            exp['node_types'] = stats
        else:
            exp['node_types'] = {}
        experiments.append(exp)

    conn.close()
    return experiments

@app.get("/api/exp/{exp_id}")
async def get_experiment(exp_id: int):
    """获取单个实验详情"""
    conn = get_db()

    exp = conn.execute('SELECT * FROM experiments WHERE id = ?', (exp_id,)).fetchone()
    if not exp:
        conn.close()
        return {"error": "实验不存在"}

    exp = dict(exp)

    # 节点统计
    node_stats = {}
    for row in conn.execute('SELECT node_type, total_count FROM node_stats WHERE exp_id = ?', (exp_id,)).fetchall():
        exp['node_types'] = {**exp.get('node_types', {}), row['node_type']: row['total_count']}

    conn.close()
    return exp

@app.get("/api/exp/{exp_id}/charts")
async def get_charts(exp_id: int):
    """获取图表数据"""
    conn = get_db()
    rows = conn.execute('''SELECT time, tp, loss, disaster FROM frame_data
                          WHERE exp_id = ? ORDER BY frame_idx''', (exp_id,)).fetchall()
    conn.close()

    return {
        "times": [r['time'] for r in rows],
        "tps": [r['tp'] for r in rows],
        "losses": [r['loss'] for r in rows],
        "disasters": [r['disaster'] for r in rows]
    }

@app.get("/api/exp/{exp_id}/frame/{frame_idx}")
async def get_frame(exp_id: int, frame_idx: int):
    """获取指定帧"""
    conn = get_db()
    row = conn.execute('''SELECT * FROM frame_data WHERE exp_id = ? AND frame_idx = ?''',
                       (exp_id, frame_idx)).fetchone()
    conn.close()

    if not row: return {"error": "帧不存在"}

    data = json.loads(row['data_json'])
    return {
        "frame_idx": row['frame_idx'],
        "time": row['time'],
        "tp": row['tp'],
        "loss": row['loss'],
        "disaster": row['disaster'],
        "nodes": data['nodes'],
        "links": data['links']
    }

@app.get("/api/exp/{exp_id}/stats")
async def get_stats(exp_id: int):
    """获取实验统计"""
    conn = get_db()

    exp = conn.execute('SELECT * FROM experiments WHERE id = ?', (exp_id,)).fetchone()
    if not exp:
        conn.close()
        return {"error": "实验不存在"}

    stats = dict(exp)

    # 帧统计
    frame_stats = conn.execute('''SELECT
        MAX(tp) as max_tp, AVG(tp) as avg_tp,
        MAX(loss) as max_loss, AVG(loss) as avg_loss
        FROM frame_data WHERE exp_id = ?''', (exp_id,)).fetchone()

    stats.update(dict(frame_stats))

    # 节点统计
    node_stats = {}
    for row in conn.execute('SELECT node_type, total_count FROM node_stats WHERE exp_id = ?', (exp_id,)).fetchall():
        type_names = {0: "用户", 1: "宏基站", 2: "背负式基站", 3: "小型基站", 4: "中继节点"}
        node_stats[type_names.get(row['node_type'], f"类型{row['node_type']}")] = row['total_count']

    stats['node_types'] = node_stats

    conn.close()
    return stats

@app.post("/api/import")
async def manual_import():
    """手动触发导入"""
    exp_id = import_trace()
    if exp_id:
        return {"success": True, "exp_id": exp_id}
    return {"success": False, "message": "没有找到 trace.json"}

# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("  应急通信数字孪生平台 - 后端服务")
    print("  端口: 8000")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8000)
