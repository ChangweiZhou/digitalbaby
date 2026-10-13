"""Local read-only status page. Never imports or invokes scientific workers."""
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
import json,time,os
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results'
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return None

def snapshot():
 status=read(R/'STATUS.json') or {};launch=read(R/'LAUNCH.json') or {}
 pid=launch.get('pid');alive=None
 if pid:
  try:os.kill(pid,0);alive=True
  except ProcessLookupError:alive=False
  except PermissionError:alive=None
 committed=sum(1 for p in (R/'certificates').glob('*.json') if (R/'receipts'/(p.stem+'.json.gz')).exists())
 active=[]
 for p in (R/'active').glob('*.json'):
  item=read(p)
  if item:item['age_seconds']=round(time.time()-item.get('epoch',time.time()),1);active.append(item)
 stage=read(R/'ANALYSIS_HEARTBEAT.json')
 return {'status':status,'committed_jobs':committed,'planned_jobs':240,'committed_histories':committed*3,'committed_records':committed*1536,'planned_records':368640,'supervisor_pid':pid,'supervisor_alive':alive,'status_age_seconds':round(time.time()-status.get('epoch',time.time()),1),'active':active,'analysis':stage,'failure':read(R/'FAILURE.json'),'package':read(R/'PACKAGE.json'),'time':time.time()}
HTML='''<!doctype html><meta charset="utf-8"><title>ADAPT-2 实验监视器</title><style>body{font:17px system-ui;background:#111827;color:#e5e7eb;margin:40px;max-width:1000px}h1{font-size:28px}.card{padding:20px;background:#1f2937;border-radius:12px;margin:14px 0}progress{width:100%;height:24px}pre{white-space:pre-wrap;font-size:14px}small{color:#9ca3af}</style><h1>ADAPT-2 · 未见四则题样本效率</h1><small>只读实时页面 · 每3秒刷新 · 2个后台工作进程 · 48配对世界</small><div class="card"><h2 id="status">连接中…</h2><progress id="bar" max="240" value="0"></progress><p id="counts"></p><p id="live"></p></div><div class="card"><h3>当前任务（部分工作，不计入完成）</h3><pre id="active"></pre></div><div class="card"><h3>分析 / 审核 / 打包</h3><pre id="final"></pre></div><small>完成数以已提交完整收据与对应审核证书存在为准。720条历史相互相关，不是720个独立样本。失败不会自动重启；此页面不修改科学程序。</small><script>async function tick(){try{let r=await fetch('/status.json',{cache:'no-store'});if(!r.ok)throw Error(r.status);let d=await r.json();document.getElementById('status').textContent=d.status.status||'等待主管状态';document.getElementById('bar').value=d.committed_jobs;document.getElementById('counts').textContent=`${d.committed_jobs} / 240 完整任务 · ${d.committed_histories} / 720 相关历史 · ${d.committed_records.toLocaleString()} / 368,640 外部记录`;document.getElementById('live').textContent=`主管 PID ${d.supervisor_pid} · 进程${d.supervisor_alive===true?'存活':d.supervisor_alive===false?'已退出':'未知'} · 状态记录 ${d.status_age_seconds} 秒前更新`;document.getElementById('active').textContent=JSON.stringify(d.active,null,2);document.getElementById('final').textContent=JSON.stringify({analysis:d.analysis,failure:d.failure,package:d.package},null,2)}catch(e){document.getElementById('status').textContent='监视器连接失败：'+e.message}}tick();setInterval(tick,3000)</script>'''
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  if self.path=='/status.json':body=json.dumps(snapshot()).encode();kind='application/json'
  elif self.path in ('/','/index.html'):body=HTML.encode();kind='text/html; charset=utf-8'
  else:self.send_error(404);return
  self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
 def log_message(self,*args):pass
if __name__=='__main__':
 server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
 (Path(__file__).parent/'MONITOR.json').write_text(json.dumps({'pid':os.getpid(),'url':f'http://127.0.0.1:{server.server_port}/','read_only':True,'epoch':time.time()}))
 server.serve_forever()
