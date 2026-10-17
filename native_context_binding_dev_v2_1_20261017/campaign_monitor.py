"""Read-only loopback status monitor, independent of the scientific process."""
import argparse
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import campaign_common as c

HTML='''<!doctype html><meta charset="utf-8"><title>MiniFly NATIVE-BIND</title>
<style>body{font:16px system-ui;background:#101827;color:#e5edf7;margin:32px;max-width:1150px}h1{font-size:26px}.card{padding:18px;background:#1b293d;border-radius:12px;margin:14px 0}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:9px;border-bottom:1px solid #354359}.muted{color:#a0b2c9}b{color:#85dbb5}pre{white-space:pre-wrap}a{color:#94caff}</style>
<h1>MiniFly · NATIVE-BIND V2.1 · DEV_A</h1><p class="muted">原生核心上下文绑定 · 未见加法泛化 / 样本 / 计算 / OLD保持</p>
<div class="card" id="headline">Loading…</div><div class="card" id="counts"></div><div class="card"><table><thead><tr><th>Worker</th><th>配置</th><th>历史 / 阶段</th><th>本历史记录</th><th>心跳</th></tr></thead><tbody id="workers"></tbody></table></div>
<p class="muted">只读监视器。仅完整且审计通过的job计入已提交；在途记录不计入已完成历史。4进程；8小时准入 / 10小时硬边界。确认有条件进行，最大规模不是必须执行量。</p>
<div class="card"><a href="/report">查看当前报告</a><pre id="detail"></pre></div>
<script>const esc=s=>String(s??'—').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
async function tick(){try{const r=await fetch('/api/status',{cache:'no-store'}),d=await r.json();
document.getElementById('headline').innerHTML='<b>'+esc(d.status)+'</b> · '+esc(d.stage)+'<br><span class="muted">'+esc(d.updated_utc)+'</span>';
document.getElementById('counts').innerHTML='已提交完整 jobs：<b>'+esc(d.committed_full_jobs)+'</b>　持续历史：<b>'+esc(d.committed_histories)+'</b>　外部记录：<b>'+esc(d.committed_records)+'</b><br>当前阶段：'+esc(d.stage_committed_jobs)+' / '+esc(d.stage_total_jobs)+'　待分配：'+esc(d.pending_jobs);
document.getElementById('workers').innerHTML=(d.active_jobs||[]).map(x=>'<tr><td>'+esc(x.pid)+'</td><td>'+esc(x.arm)+'</td><td>'+esc(x.history)+' / '+esc(x.phase)+'</td><td>'+esc(x.history_records)+'</td><td>'+esc(x.heartbeat_epoch?Math.round(Date.now()/1000-x.heartbeat_epoch)+'s ago':'starting')+'</td></tr>').join('');
document.getElementById('detail').textContent=d.error||d.verdict||(d.next_stage?'Next: '+d.next_stage:'');
}catch(e){document.getElementById('detail').textContent='Monitor fetch failed: '+e;} }tick();setInterval(tick,5000);</script>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path=="/api/status":
            path=c.RESULTS/"STATUS.json"
            data=path.read_bytes() if path.exists() else b'{"status":"PREPARING"}'
            kind="application/json"
        elif self.path=="/report":
            path=c.RESULTS/"REPORT.md"
            if not path.exists():path=c.RESULTS/"DEV_REPORT.md"
            data=path.read_bytes() if path.exists() else "报告尚未生成。".encode();kind="text/plain; charset=utf-8"
        elif self.path=="/":data=HTML.encode();kind="text/html; charset=utf-8"
        else:self.send_error(404);return
        self.send_response(200);self.send_header("Content-Type",kind);self.send_header("Cache-Control","no-store")
        self.send_header("Content-Length",str(len(data)));self.end_headers();self.wfile.write(data)
    def log_message(self,*args):pass


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--port",type=int,default=8889);args=p.parse_args()
    ThreadingHTTPServer(("127.0.0.1",args.port),Handler).serve_forever()
