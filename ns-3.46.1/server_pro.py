#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应急通信数字孪生平台 - 后端服务
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import json
import os
import time
import threading
from typing import Optional

DB_FILE = "simulation_history.db"
TRACE_FILE = "trace.json"

# ============================================================================
# 数据库初始化
# ============================================================================

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS experiments 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, date TEXT, 
                  duration REAL, total_nodes INTEGER, disaster_time REAL, frames INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS frame_data 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, exp_id INTEGER, frame_idx INTEGER, 
                  time REAL, tp REAL, loss REAL, disaster INTEGER, data_json TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS node_stats 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, exp_id INTEGER, node_type INTEGER, total_count INTEGER)''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_frame_exp ON frame_data(exp_id)')
    conn.commit()
    conn.close()
    print("[DB] 数据库初始化完成")

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
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        date_str = time.strftime("%Y-%m-%d %H:%M:%S")
        c.execute('''INSERT INTO experiments (name, date, duration, total_nodes, disaster_time, frames)
                    VALUES (?, ?, ?, ?, ?, ?)''',
                 (f"演练_{int(time.time())}", date_str, duration, node_count, disaster_time, len(frames)))
        exp_id = c.lastrowid
        
        for idx, frame in enumerate(frames):
            data_json = json.dumps({'nodes': frame.get('nodes', []), 'links': frame.get('links', [])})
            c.execute('''INSERT INTO frame_data (exp_id, frame_idx, time, tp, loss, disaster, data_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?)''',
                     (exp_id, idx, frame.get('time', 0), frame.get('tp', 0), 
                      frame.get('loss', 0), frame.get('disaster', 0), data_json))
        
        conn.commit()
        conn.close()
        
        print(f"[IMPORT] 成功! 实验ID: {exp_id}, 帧数: {len(frames)}")
        os.remove(TRACE_FILE)
        return exp_id
        
    except Exception as e:
        print(f"[IMPORT] 失败: {e}")
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
        print("[WATCHER] 停止文件监控")
    
    def _loop(self):
        while self.running:
            if os.path.exists(TRACE_FILE):
                time.sleep(1)
                import_trace()
            time.sleep(2)

# ============================================================================
# FastAPI 应用
# ============================================================================

init_db()

app = FastAPI(title="应急通信数字孪生平台 API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"])

watcher = Watcher()

@app.on_event("startup")
async def startup():
    import_trace()
    watcher.start()

@app.on_event("shutdown")
async def shutdown():
    watcher.stop()

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/")
async def root():
    return {"name": "应急通信数字孪生平台 API", "version": "1.0.0"}

@app.get("/api/health")
async def health():
    return {"status": "ok", "db_exists": os.path.exists(DB_FILE)}

@app.get("/api/experiments")
async def get_experiments():
    conn = get_db()
    exps = [dict(r) for r in conn.execute("SELECT * FROM experiments ORDER BY id DESC").fetchall()]
    conn.close()
    return exps

@app.get("/api/exp/{exp_id}/charts")
async def get_charts(exp_id: int):
    conn = get_db()
    rows = conn.execute('''SELECT time, tp, loss FROM frame_data 
                          WHERE exp_id = ? ORDER BY frame_idx''', (exp_id,)).fetchall()
    conn.close()
    return {"times": [r['time'] for r in rows], "tps": [r['tp'] for r in rows], 
            "losses": [r['loss'] for r in rows]}

@app.get("/api/exp/{exp_id}/frame/{frame_idx}")
async def get_frame(exp_id: int, frame_idx: int):
    conn = get_db()
    row = conn.execute('''SELECT * FROM frame_data WHERE exp_id = ? AND frame_idx = ?''',
                       (exp_id, frame_idx)).fetchone()
    conn.close()
    if not row: return {"error": "帧不存在"}
    data = json.loads(row['data_json'])
    return {"time": row['time'], "tp": row['tp'], "loss": row['loss'], 
            "disaster": row['disaster'], "nodes": data['nodes'], "links": data['links']}

@app.post("/api/import")
async def manual_import():
    exp_id = import_trace()
    if exp_id: return {"success": True, "exp_id": exp_id}
    return {"success": False}

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("  应急通信数字孪生平台 - 后端服务")
    print("  端口: 8000")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
