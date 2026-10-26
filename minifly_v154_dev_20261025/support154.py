"""V154 evaluator accounting and task data; no task information in the learner."""
from pathlib import Path
import time,os,json,hashlib,pickle,resource
from collections import Counter
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parent;D=ROOT/'minifly_v154_design_20261025';START=time.monotonic();WALL=time.time();COUNTS=Counter();TIMES=Counter();ACTIVE_STATE=None
class BudgetStop(RuntimeError):pass

def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x

def save(name,x):
 p=H/name;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp')
 with q.open('w') as f:json.dump(plain(x),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 q.replace(p)
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cpu():
 c=resource.getrusage(resource.RUSAGE_CHILDREN);return time.process_time()+c.ru_utime+c.ru_stime

def budget(reserve=600):
 if cpu()+reserve>=7200:raise BudgetStop('PAUSED_CPU_BUDGET')
 if time.monotonic()-START+reserve>=10800:raise BudgetStop('PAUSED_WALL_BUDGET')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>1073741824:raise BudgetStop('PAUSED_MEMORY_TARGET')
 assert COUNTS['record_applications']<=43328 and COUNTS['observed_targets']<=129984
 assert COUNTS['pair_queries']<=43676 and COUNTS['probe_TF_records']<=8704
 assert COUNTS['technical_records']<=64 and COUNTS['technical_pair_queries']<=128
 assert COUNTS['batch_evaluations']<=3840 and COUNTS['batch_virtual_target_reuses']<=11796480
 assert COUNTS['native_records']<=25600 and COUNTS['feature_records']<=1280 and COUNTS['online_probe_records']<=16384 and COUNTS['probe_TF_distributions']<=26112

def record(prompt,answer,kind,write=False):
 budget();COUNTS['record_applications']+=1;COUNTS['observed_targets']+=len(answer)+1;COUNTS[kind+'_records']+=1;COUNTS[kind+'_targets']+=len(answer)+1;COUNTS['observed_prompt_bytes']+=len(prompt);COUNTS['updating_records' if write else 'nonupdating_records']+=1

def query(kind):COUNTS['pair_queries']+=1;COUNTS[kind+'_pair_queries']+=1;budget()
def log(event,**x):
 with (H/'LEDGER.jsonl').open('a') as f:f.write(json.dumps(plain(dict(event=event,CPU=cpu(),wall=time.monotonic()-START,**x)),allow_nan=False)+'\n')
def digest(obj):
 t=time.process_time();raw=pickle.dumps(obj,protocol=5);COUNTS['digest_bytes']+=len(raw);out=hashlib.sha256(raw).hexdigest();TIMES['digest_CPU']+=time.process_time()-t;return out

def checkpoint(obj,name):
 p=H/'local'/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('wb') as f:pickle.dump(obj,f,protocol=5)
 return dict(path=str(p.relative_to(H)),sha256=sha(p),bytes=p.stat().st_size)
def rng(world,stream):return np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(f'V154|{world}|{stream}'.encode()).digest()[:16],'big')))
def dataset(world):
 orbits=[(i,j) for i in range(20) for j in range(i,20)];order=rng(world,'split').permutation(210);orient=rng(world,'orientation').integers(0,2,210);rows=[]
 for k in order:
  i,j=orbits[int(k)];a,b=(j,i) if orient[k] else (i,j);rows.append(dict(id=f'{world}:{k}',orbit=[i,j],prompt=f'{a}+{b}='.encode().hex(),answer=str(a+b).encode().hex()))
 return dict(world=world,OLD=rows[:64],NEW=rows[64:128],validation=rows[128:160],assessment=rows[160:],orders={phase:[rng(world,f'{phase}|pass|{p}').permutation(64).tolist() for p in range(1,9)] for phase in ['OLD','NEW']})
def unpack(row):return bytes.fromhex(row['prompt']),bytes.fromhex(row['answer'])
