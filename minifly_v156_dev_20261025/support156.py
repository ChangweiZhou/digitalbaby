"""Bounded diagnostic accounting; no task computation or candidate update."""
from pathlib import Path
from collections import Counter
import os,time,json,hashlib,resource,pickle
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parent;D=ROOT/'minifly_v156_design_20261025'
START=time.monotonic();COUNTS=Counter();TIMES=Counter();STAGE='qualification'
START_CPU=time.process_time();STAGE_CPU=START_CPU;ACTIVE_MODEL=None;ACTIVE_CASE=None;_INPUTS=None
CAPS={'qualification':300.,'F':900.,'M':1800.,'audit':600.}
class BudgetStop(RuntimeError):pass
def cpu():return time.process_time()-START_CPU
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x
def save(path,x):
 p=H/path;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
 with q.open('w') as f:json.dump(plain(x),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 q.replace(p)
def load(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def log(event,**kw):
 with (H/'LEDGER.jsonl').open('a') as f:f.write(json.dumps(plain(dict(event=event,cpu=cpu(),wall=time.monotonic()-START,**kw)),allow_nan=False)+'\n')
def stage(name):
 global STAGE,STAGE_CPU
 TIMES[STAGE]+=time.process_time()-STAGE_CPU;STAGE=name;STAGE_CPU=time.process_time();budget()
def budget():
 if cpu()>=3600:raise BudgetStop('TOTAL_CPU_LIMIT')
 if time.monotonic()-START>=7200:raise BudgetStop('WALL_LIMIT')
 if TIMES[STAGE]+time.process_time()-STAGE_CPU>=CAPS[STAGE]:raise BudgetStop(STAGE+'_CPU_LIMIT')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>1073741824:raise BudgetStop('RSS_LIMIT')
 assert COUNTS['observed_records']<=144 and COUNTS['observed_targets']<=432
 assert COUNTS['technical_records']<=16 and COUNTS['technical_pairs']<=64
 assert COUNTS['native_pairs']<=16320 and COUNTS['witness_queries']<=1536
 assert COUNTS['evaluator_distributions']<=120832
def record(row,kind,write):
 a=bytes.fromhex(row['answer']);p=bytes.fromhex(row['prompt']);COUNTS['observed_records']+=1;COUNTS['observed_targets']+=len(a)+1;COUNTS[kind+'_records']+=1;COUNTS[kind+'_targets']+=len(a)+1;COUNTS['updated_records' if write else 'inhibited_records']+=1;COUNTS['feedback_payload_bytes']+=len(a);COUNTS['feedback_EOS']+=1;COUNTS['observed_prompt_bytes']+=len(p);budget()
def digest(m):return hashlib.sha256(pickle.dumps(m,protocol=5)).hexdigest()
def checkpoint(m,name):
 p=H/'local'/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('wb') as f:pickle.dump(m,f,protocol=5)
 return dict(path=str(p.relative_to(H)),sha256=sha(p),bytes=p.stat().st_size)
def inputs():
 global _INPUTS
 if _INPUTS is not None:return _INPUTS
 # Filter before any learner/fit sees records. Do not retain held labels in the run inputs.
 raw=load(ROOT/'minifly_v154_dev_20261025/local/DATA_MANIFEST.json');out={}
 for w in [202610255401,202610255402]:
  j=raw[str(w)];new=sorted(j['NEW'],key=lambda r:(hashlib.sha256(f"V156|{w}|margin-order|{r['id']}".encode()).digest(),r['id']))[:16]
  out[str(w)]=dict(world=w,OLD=j['OLD'],NEW=j['NEW'],selected_NEW=new,panel=j['OLD']+new)
 _INPUTS=out
 return out
def snapshot_counts():
 return dict(counts=dict(COUNTS),cpu=cpu(),stage_cpu={**dict(TIMES),STAGE:TIMES[STAGE]+time.process_time()-STAGE_CPU},wall=time.monotonic()-START,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
