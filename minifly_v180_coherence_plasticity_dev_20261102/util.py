import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,time,hashlib,resource
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parent
START=time.time();CPU0=time.process_time()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(b):return hashlib.sha256(b).hexdigest()
def save(n,x):
 p=H/n;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def load(n):return json.loads((H/n).read_text())
def budget():
 a=load('AUTHORIZATION.json');cpu=time.process_time()+sum(v['CPU'] for v in load('PROCESS_LEDGER.json').get('completed',[]));rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 size=sum(p.stat().st_size for p in H.rglob('*') if p.is_file() and not p.is_symlink())
 if size>a['output_bytes_cap']:raise RuntimeError('OUTPUT_BOUNDARY')
 if cpu>a['cpu_seconds_cap'] or time.time()-a['epoch']>a['wall_seconds_cap'] or rss>a['rss_bytes_cap']:raise RuntimeError('RESOURCE_BOUNDARY')
 return dict(CPU_current_process=time.process_time(),CPU_accounted=cpu,RSS_peak_bytes=rss,wall_seconds=time.time()-a['epoch'])
def finish_process(name):
 l=load('PROCESS_LEDGER.json');l['completed'].append(dict(name=name,CPU=time.process_time(),wall=time.time()-START,RSS_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));save('PROCESS_LEDGER.json',l)
