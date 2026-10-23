import time,os,json,hashlib,resource,subprocess,ctypes,re
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
START=time.monotonic();CPU=0.;PREV=json.loads((P/'results/RESOURCE_LEDGER.json').read_text()) if (P/'results/RESOURCE_LEDGER.json').exists() else dict(CPU_seconds=0,wall_seconds=0)
LIB=ctypes.CDLL('/usr/lib/libproc.dylib');awake=None;peak=0;last=0

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,v):
 p=P/name;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.pending');q.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n');q.replace(p)
def stats():
 global peak
 rss=0
 for pid in [os.getpid()]+([awake.pid] if awake is not None and awake.poll() is None else []):
  buf=ctypes.create_string_buffer(1024)
  if LIB.proc_pid_rusage(pid,0,ctypes.byref(buf))!=0:raise RuntimeError('RSS_UNAVAILABLE')
  rss+=int(ctypes.cast(ctypes.byref(buf,16),ctypes.POINTER(ctypes.c_uint64))[6])
 peak=max(peak,rss);ch=resource.getrusage(resource.RUSAGE_CHILDREN)
 return dict(CPU_seconds=PREV['CPU_seconds']+time.process_time()-CPU+ch.ru_utime+ch.ru_stime,wall_seconds=PREV['wall_seconds']+time.monotonic()-START,peak_tree_RSS_bytes=peak)
def check(message=None,admission=False,**kw):
 global last
 st=stats()
 if st['CPU_seconds']>=7200 or st['wall_seconds']>=10800 or peak>=2*1024**3:raise RuntimeError('RESOURCE_LIMIT')
 if admission:
  s=subprocess.check_output(['/usr/bin/vm_stat'],text=True);pg=int(re.search(r'page size of (\d+)',s).group(1));head=sum(int(re.search(re.escape(k)+r':\s+(\d+)',s).group(1)) for k in ('Pages free','Pages inactive','Pages speculative'))*pg
  if head<2*1024**3:raise RuntimeError('INSUFFICIENT_HEADROOM')
 if message or time.monotonic()-last>20:
  save('results/STATUS.json',dict(message=message,**kw,**st));print(json.dumps(dict(message=message,**kw,**st)),flush=True);last=time.monotonic()
def begin():
 global awake
 awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())]);check('START',admission=True)
def end():
 save('results/RESOURCE_LEDGER.json',stats())
 if awake is not None:awake.terminate();awake.wait()
