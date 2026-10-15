"""Local read-only status surface; never imports or calls scientific code."""
import json
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer

ROOT=Path(__file__).resolve().parents[1]
PAGE='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>MiniFly 加法三轮技术验证</title>
<style>body{font:17px system-ui;max-width:1050px;margin:40px auto;background:#f5f5f1;color:#182a35}pre{background:white;padding:24px;white-space:pre-wrap;border-radius:12px}h1{font-size:28px}.note{color:#536570}</style>
<h1>MiniFly · 加法核心谱系比较</h1><p class="note">三轮技术验证 · 五个固定版本 · 正式科学实验尚未启动。局部训练进度不计为已提交作业。</p><pre id="s">读取状态…</pre>
<script>async function tick(){try{let d=await(await fetch('/api',{cache:'no-store'})).json();document.getElementById('s').textContent=JSON.stringify(d,null,2)}catch(e){document.getElementById('s').textContent+='\\n状态服务暂时不可达；请查看本地 results/STATUS.json。'}}tick();setInterval(tick,3000)</script></html>'''


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path=='/api':
            value={'status':read(ROOT/'results/STATUS.json'),
                   'cycle1':read(ROOT/'cycles/cycle1/TEST_RESULTS.json'),
                   'cycle2':{k:v for k,v in (read(ROOT/'cycles/cycle2/SUMMARY.json') or {}).items() if k!='results'},
                   'active':[read(p) for p in (ROOT/'cycles/cycle3/active').glob('*.json')]}
            payload=json.dumps(value,ensure_ascii=False,indent=2).encode();mime='application/json'
        elif self.path=='/':payload=PAGE.encode();mime='text/html; charset=utf-8'
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(payload)
    def log_message(self,*args):pass


if __name__=='__main__':
    server=HTTPServer(('127.0.0.1',8921),Handler);server.timeout=2
    start=time.monotonic();finished=None
    while time.monotonic()-start<36000:
        server.handle_request()
        state=(read(ROOT/'results/STATUS.json') or {}).get('state','')
        if state in ('THREE_CYCLES_COMPLETE','FAILED','PAUSED_SESSION_LIMIT'):
            if finished is None:finished=time.monotonic()
            if time.monotonic()-finished>120:break
    server.server_close()
