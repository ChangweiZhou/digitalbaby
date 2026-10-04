"""Loopback-only, read-only monitor. Never imports a scientific runner."""
import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from urllib.parse import urlsplit

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
RESULTS=ROOT/'results'
ARMS=('V1','ERROR','REPLACE','BOTH')
RECEIPT=re.compile(r'^(501\d{3})_(V1|ERROR|REPLACE|BOTH)_(lifetime|reuse)\.json\.gz$')
CACHE_LOCK=threading.Lock()
CACHE={'at':0.,'data':None}


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def alive(pid):
    try:os.kill(int(pid),0);return True
    except (OSError,TypeError,ValueError):return False


def timestamp(value):
    try:return datetime.fromisoformat(value).timestamp()
    except (ValueError,TypeError):return None


def tail(path):
    try:
        path=path.resolve()
        if not path.is_relative_to(RESULTS.resolve()):return 'Log path is outside this experiment.'
        with path.open('rb') as f:
            f.seek(max(0,path.stat().st_size-6000))
            return '\n'.join(f.read().decode(errors='replace').splitlines()[-22:])
    except OSError:return ''


def snapshot():
    with CACHE_LOCK:
        now=time.time()
        if CACHE['data'] is not None and now-CACHE['at']<1.5:return CACHE['data']
        status=read(RESULTS/'STATUS.json');launch=read(RESULTS/'LAUNCH.json')
        started=timestamp(status.get('science_started_utc'))
        heartbeat=timestamp(status.get('heartbeat_utc'))
        heartbeat_age=max(0,now-heartbeat) if heartbeat is not None else None
        observed=status.get('state','WAITING');notice='';failure=read(RESULTS/'FAILURE.json')
        if failure:
            observed='FAILED';notice=failure.get('error','Execution failure recorded.')
        elif observed=='RUNNING' and (heartbeat_age is None or heartbeat_age>30):
            # A stale file alone cannot prove that the runner stopped.
            pid=status.get('pid');live=alive(pid)
            command=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True,timeout=2).stdout.strip() if live else ''
            expected=launch.get('supervisor',str(ROOT/'supervisor.py'))
            live=live and expected in command
            observed='HEARTBEAT LATE' if live else 'STOPPED'
            notice='Supervisor heartbeat is late; its process is still alive.' if live else 'Supervisor heartbeat is stale and the expected process is not alive. No restart is performed by this monitor.'
        counters={w:0 for w in range(501001,501097)}
        arms={a:{'lifetime':0,'reuse':0} for a in ARMS}
        for p in (RESULTS/'science/receipts').glob('*.json.gz'):
            match=RECEIPT.fullmatch(p.name)
            if match:
                w,a,assay=match.groups();w=int(w)
                if w in counters:counters[w]+=1;arms[a][assay]+=1
        jobs=sum(counters.values());lives=sum(x['lifetime']*4+x['reuse']*2 for x in arms.values())
        active=[]
        for job in status.get('active_jobs',[]):
            key=f"{job['world']}_{job['arm']}_{job['assay']}"
            progress=read(RESULTS/'science/active'/f'{key}.json')
            active.append({**job,'branch':progress.get('branch'),'cursor':progress.get('committed_record_cursor'),
                           'records_per_life':864 if job['assay']=='lifetime' else 216,
                           'heartbeat_age_s':max(0,now-progress['heartbeat_unix']) if progress.get('heartbeat_unix') else None})
        qualification=read(RESULTS/'QUALIFICATION.json')
        eq=read(ROOT/'ops/workers8/QUALIFICATION.json')
        data={'now_unix':now,'status':status,'observed_state':observed,'heartbeat_age_s':heartbeat_age,
              'elapsed_s':now-started if started is not None else 0,'notice':notice,'failure':bool(failure),
              'caffeinate_alive':status.get('caffeinate_attached',False) and alive(status.get('caffeinate_pid')),
              'active':active,'committed':{'jobs':jobs,'lives':lives,'full_worlds':sum(x==8 for x in counters.values()),
                                          'worlds':[{'world':w,'jobs':n} for w,n in counters.items()],'arms':arms},
              'log':tail(Path(launch['log'])) if launch.get('log') else '',
              'qualification':f"3 cycles: {qualification.get('verdict','pending')} · 8 workers: {eq.get('verdict','pending')}",
              'report_available':(RESULTS/'REPORT.md').exists(),'bundle_available':(RESULTS/'RESULT_BUNDLE.zip').exists()}
        CACHE.update(at=now,data=data)
        return data


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        host=self.headers.get('Host','').split(':')[0]
        if host not in ('127.0.0.1','localhost'):
            self.send_error(403);return
        path=urlsplit(self.path).path
        if path=='/':body=(HERE/'index.html').read_bytes();kind='text/html; charset=utf-8'
        elif path=='/api/status':body=json.dumps(snapshot()).encode();kind='application/json; charset=utf-8'
        elif path=='/report' and (RESULTS/'REPORT.md').exists():body=(RESULTS/'REPORT.md').read_bytes();kind='text/plain; charset=utf-8'
        elif path=='/bundle' and (RESULTS/'RESULT_BUNDLE.zip').exists():
            file=RESULTS/'RESULT_BUNDLE.zip'
            self.send_response(200);self.send_header('Content-Type','application/zip');self.send_header('Content-Length',str(file.stat().st_size));self.send_header('Content-Disposition','attachment; filename="RESULT_BUNDLE.zip"');self.end_headers()
            with file.open('rb') as f:shutil.copyfileobj(f,self.wfile,262144)
            return
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8874);args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Read-only persistent-core monitor: http://127.0.0.1:{args.port}/',flush=True)
    server.serve_forever()


if __name__=='__main__':main()
