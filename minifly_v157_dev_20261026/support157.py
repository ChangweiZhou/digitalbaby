"""A1 accounting and admission. No arithmetic or candidate learning equation."""
from pathlib import Path
from collections import Counter
import os,json,time,pickle,hashlib,resource
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parent;D=ROOT/'minifly_v157_design_20261026'
START=time.monotonic();COUNTS=Counter();TIMES=Counter();STAGE='qualification';STAGE_START=0.;STAGE_USED=Counter();ACTIVE=None;ACTIVE_INFO=None
CAP={'qualification':120.,'execution':900.,'audit':180.}
class Limit(RuntimeError):pass
class AdmissionStop(RuntimeError):pass
def cpu():
 c=resource.getrusage(resource.RUSAGE_CHILDREN);return time.process_time()+c.ru_utime+c.ru_stime
def stage_cpu():return STAGE_USED[STAGE]+cpu()-STAGE_START
def stage(name):
 global STAGE,STAGE_START
 STAGE_USED[STAGE]+=cpu()-STAGE_START;STAGE=name;STAGE_START=cpu();budget()
def budget():
 if cpu()>=1200:raise Limit('TOTAL_CPU_CAP')
 if stage_cpu()>=CAP[STAGE]:raise Limit(STAGE+'_CPU_CAP')
 if time.monotonic()-START>=3600:raise Limit('WALL_CAP')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>1073741824:raise Limit('RSS_CAP')
 assert COUNTS['observed_records']<=3120 and COUNTS['observed_targets']<=9360
 assert COUNTS['technical_records']<=16 and COUNTS['technical_pairs']<=64
 assert COUNTS['science_pairs']<=9696 and COUNTS['held_queries']<=800
 assert COUNTS['evaluator_distributions']<=71520 and COUNTS['logical_evaluator_distributions']<=71520
def clean(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [clean(v) for v in x]
 return x
def save(path,obj):
 p=H/path;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
 with q.open('w') as f:json.dump(clean(obj),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 q.replace(p)
def load(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def log(event,**kw):
 with (H/'LEDGER.jsonl').open('a') as f:f.write(json.dumps(clean(dict(event=event,CPU=cpu(),wall=time.monotonic()-START,**kw)),allow_nan=False)+'\n')
def digest(m):
 t=time.process_time();b=pickle.dumps(m,protocol=5);COUNTS['digest_bytes']+=len(b);z=hashlib.sha256(b).hexdigest();TIMES['digest_CPU']+=time.process_time()-t;return z
def checkpoint(m,name):
 p=H/'local'/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('wb') as f:pickle.dump(m,f,protocol=5)
 return dict(path=str(p.relative_to(H)),sha256=sha(p),bytes=p.stat().st_size)
def restore(meta):
 p=H/meta['path'];assert sha(p)==meta['sha256']
 with p.open('rb') as f:return pickle.load(f)
def record(row,kind,write):
 p=bytes.fromhex(row['prompt']);a=bytes.fromhex(row['answer']);COUNTS['observed_records']+=1;COUNTS['observed_targets']+=len(a)+1;COUNTS[kind+'_records']+=1;COUNTS[kind+'_targets']+=len(a)+1;COUNTS['updated_records' if write else 'inhibited_records']+=1;COUNTS['observed_prompt_bytes']+=len(p);COUNTS['feedback_payload_bytes']+=len(a);COUNTS['feedback_EOS']+=1;budget()
def state():
 return dict(CPU=cpu(),wall=time.monotonic()-START,counts=dict(COUNTS),stage=STAGE,stage_CPU={**dict(STAGE_USED),STAGE:stage_cpu()},timings=dict(TIMES),RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
