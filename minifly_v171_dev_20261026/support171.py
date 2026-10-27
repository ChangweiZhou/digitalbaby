"""V171 accounting only. Source experiment V169; its salts/caps are unchanged."""
from pathlib import Path
from collections import Counter
import os,time,json,hashlib,pickle,resource,sys
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parent
SPEC=json.loads((H/'EXECUTION_SPEC.json').read_text());START=json.loads((H/'START.json').read_text())
prior=json.loads((H/'COUNTS.json').read_text()) if (H/'COUNTS.json').exists() else {}
COUNTS=Counter(prior.get('counts',{}));TIMES=Counter(prior.get('timings',{}));PRIOR_CPU=prior.get('CPU',0.)
PROCESS_START=time.process_time();STAGE='initialization'
class Stop(RuntimeError):pass
def cpu():return PRIOR_CPU+time.process_time()-PROCESS_START
def wall():return time.time()-START['wall_epoch']
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x
def save(path,x):
 p=H/path;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(plain(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def budget(reserve=0):
 if cpu()+reserve>=14400:raise Stop('PAUSED_CPU_BUDGET')
 if wall()+reserve>=21600:raise Stop('PAUSED_WALL_BUDGET')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>1073741824:raise Stop('PAUSED_RSS_BUDGET')
 for key,cap in [('logical_records',38432),('arriving_targets',115296),('technical_records',32),('native_vector_reads',400000),('distributions',280000),('paired_queries',11104),('held_base_queries',2400),('held_intervention_queries',800),('taught_intervention_queries',2880),('synthetic_loss_evaluations',512)]:
  if COUNTS[key]>cap:raise Stop('CAP_'+key)
def log(event,**kw):
 with (H/'LEDGER.jsonl').open('a') as f:f.write(json.dumps(plain(dict(event=event,stage=STAGE,CPU=cpu(),wall=wall(),**kw)),allow_nan=False)+'\n')
def finish(status=None):
 save('COUNTS.json',dict(counts=dict(COUNTS),timings=dict(TIMES),CPU=cpu(),wall=wall(),RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
 if status:save('STATUS.json',dict(version='V171',status=status,stage=STAGE,model_runs=COUNTS['model_births'],logical_records=COUNTS['logical_records'],CPU=cpu(),wall=wall()))
def records(prompt,answer,n,technical=False):
 COUNTS['logical_records']+=n;COUNTS['arriving_targets']+=n*(len(answer)+1);COUNTS['arriving_prompt_bytes']+=n*len(prompt);COUNTS['revealed_payload_bytes']+=n*len(answer);COUNTS['revealed_EOS']+=n
 if technical:COUNTS['technical_records']+=n
 else:COUNTS['calibration_records' if STAGE in ('cycle2','calibration') else 'main_records']+=n
 budget()
def digest(obj):
 raw=pickle.dumps(obj,protocol=5);COUNTS['digest_bytes']+=len(raw);return hashlib.sha256(raw).hexdigest()
def checkpoint(obj,name):
 p=H/'local'/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('wb') as f:pickle.dump(obj,f,protocol=5)
 return dict(path=str(p.relative_to(H)),sha256=sha(p),bytes=p.stat().st_size)
def rng(world,name):return np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(f'V169|{world}|{name}'.encode()).digest()[:16],'big')))
def source_lock():
 files={}
 old=json.loads((ROOT/'minifly_v157_dev_20261026/CONTINUATION_SOURCE_LOCK.json').read_text())
 for p,h in old.items():
  if p.startswith(('minifly_a_v3/','minifly_cls_dev_20261017/','minifly_v152_dev_20261025/')) or p in ['minifly_v154_dev_20261025/native154.py','minifly_v154_dev_20261025/support154.py']:
   q=ROOT/p
   if not q.is_file() or sha(q)!=h:raise RuntimeError('SOURCE_MISSING_OR_CHANGED '+p)
   files[p]=h
 for q in (ROOT/'minifly_a_v3/package').rglob('*'):
  if q.is_file() and q.suffix in ('.py','.json','.npz'):files[str(q.relative_to(ROOT))]=sha(q)
 for q in (ROOT/'minifly_v169_design_20261026').glob('*'):
  if q.is_file():files[str(q.relative_to(ROOT))]=sha(q)
 for q in H.glob('*.py'):files[str(q.relative_to(ROOT))]=sha(q)
 for name in ['EXECUTION_SPEC.json','AUTHORIZATION.json','DESIGN.md']:
  files[str((H/name).relative_to(ROOT))]=sha(H/name)
 return files
class Tape:
 """Saved observations, not state accessible by a learner."""
 def __init__(self,tag):self.tag=tag;self.rows=[];self.arr=[]
 def add(self,metadata,**arrays):self.rows.append(plain(metadata));self.arr.append({k:np.asarray(v).copy() for k,v in arrays.items()})
 def flush(self):
  p=H/'local'/f'{self.tag}.npz';p.parent.mkdir(exist_ok=True,parents=True)
  if self.arr:
   fields=set(self.arr[0]);assert all(set(x)==fields for x in self.arr)
   np.savez_compressed(p,metadata=np.frombuffer(json.dumps(self.rows,allow_nan=False).encode(),dtype=np.uint8),**{k:np.stack([x[k] for x in self.arr]) for k in fields})
   rec=dict(path=str(p.relative_to(H)),sha256=sha(p),rows=len(self.rows),fields=sorted(fields));log('TAPE_COMMIT',**rec);self.rows=[];self.arr=[];return rec
  return None
