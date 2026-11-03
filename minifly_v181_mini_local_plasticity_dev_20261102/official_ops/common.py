"""Small official runtime: atomic receipts and unchanged hard resource limits."""
import os,json,time,hashlib,resource,sys
from pathlib import Path
OPS=Path(__file__).resolve().parent;DEV=OPS.parent;ROOT=DEV.parent
sys.path.insert(0,str(DEV))
CAM=DEV/'campaign_20261102'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def load(p):return json.loads(Path(p).read_text())
def footprint():return sum(p.stat().st_size for p in CAM.rglob('*') if p.is_file())
def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
def check():
 launch=load(CAM/'LAUNCH.json');wall=time.time()-launch['started_unix'];cpu=time.process_time()+launch['carried_CPU_seconds']
 if cpu>900 or wall>1800 or rss()>1073741824 or footprint()>268435456:raise RuntimeError(f'RESOURCE_LIMIT cpu={cpu} wall={wall} RSS={rss()} output={footprint()}')
 return dict(CPU_seconds=cpu,wall_seconds=wall,peak_RSS_bytes=rss(),output_bytes=footprint())
def source_check():
 for rel,h in load(CAM/'SOURCE_LOCK.json')['files'].items():assert sha(ROOT/rel)==h,rel
