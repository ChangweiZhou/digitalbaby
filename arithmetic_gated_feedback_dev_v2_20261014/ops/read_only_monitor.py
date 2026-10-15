"""Loopback-only read-only status, no experiment control endpoints."""
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def status():
    result = dict(root=read(ROOT / 'STATUS.json'), time=time.time(), technical_only=True)
    for stage in ('cycle2', 'cycle3'):
        folder = ROOT / 'cycles' / stage
        s = read(folder / 'STATUS.json')
        active = [read(p) for p in sorted((folder / 'active').glob('*.json'))]
        active = [a for a in active if a is not None]
        launch = read(folder / 'LAUNCH.json')
        pid = None if launch is None else launch['supervisor_pid']
        alive = False
        if pid:
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            except PermissionError:
                alive = None  # permission denied is not evidence of a dead process
        result[stage] = dict(status=s, active=active, supervisor_pid=pid, supervisor_alive=alive)
    return result


PAGE = '''<!doctype html><meta charset="utf-8"><title>GATED-FEEDBACK V2 技术资格</title>
<style>body{font:16px system-ui;background:#101820;color:#edf3f8;margin:32px;max-width:1100px}pre{white-space:pre-wrap;background:#172632;padding:20px;border-radius:10px}a{color:#9ed0ff}</style>
<h1>GATED-FEEDBACK V2 · 三轮技术练习</h1><p>只读监视器。完整提交计数与进行中的记录分开；这里没有启动、重跑或更改实验的操作。</p>
<p id="head"></p><a href="/report">完成报告</a><pre id="data"></pre>
<script>async function refresh(){try{let s=await(await fetch('/status',{cache:'no-store'})).json();
document.getElementById('head').textContent='根状态：'+s.root.status+'；第三轮：'+(s.cycle3.status?.status||'待启动')+'；正式科学作业：0';
document.getElementById('data').textContent=JSON.stringify(s,null,2);}catch(e){document.getElementById('head').textContent='状态读取失败：'+e;}}refresh();setInterval(refresh,5000)</script>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            data, content = PAGE.encode(), 'text/html; charset=utf-8'
        elif self.path == '/status':
            data, content = json.dumps(status(), ensure_ascii=False).encode(), 'application/json; charset=utf-8'
        elif self.path == '/report' and (ROOT / 'REPORT.md').exists():
            data, content = (ROOT / 'REPORT.md').read_bytes(), 'text/plain; charset=utf-8'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', content)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', 8881), Handler).serve_forever()
