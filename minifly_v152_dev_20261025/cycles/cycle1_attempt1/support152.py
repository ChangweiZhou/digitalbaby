"""V152 bounded accounting and evaluator data. No arithmetic in learner modules."""
from pathlib import Path
import os, json, time, hashlib, resource, pickle
from collections import Counter
import numpy as np
H=Path(__file__).resolve().parent; ROOT=H.parent; DESIGN=ROOT/'minifly_v152_design_20261025'
START=time.monotonic(); WALL_START=time.time(); COUNTS=Counter(); TIMES=Counter(); STAGE='preflight'
class BudgetStop(RuntimeError):pass

def clean(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [clean(v) for v in x]
 return x

def save(name,x):
 p=H/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w') as f:json.dump(clean(x),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(m):
 t=time.process_time();b=pickle.dumps(m,protocol=5);COUNTS['digest_bytes']+=len(b);z=hashlib.sha256(b).hexdigest();TIMES['digest_CPU']+=time.process_time()-t;return z

def log(event,**kw):
 with (H/'LEDGER.jsonl').open('a') as f:f.write(json.dumps(clean(dict(event=event,CPU=cpu(),wall=time.monotonic()-START,**kw)),allow_nan=False)+'\n')
def cpu():
 c=resource.getrusage(resource.RUSAGE_CHILDREN);return time.process_time()+c.ru_utime+c.ru_stime

def budget(reserve=600):
 if cpu()+reserve>=7200:raise BudgetStop('PAUSED_CPU_BUDGET')
 if time.monotonic()-START+reserve>=10800:raise BudgetStop('PAUSED_WALL_BUDGET')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>1073741824:raise BudgetStop('PAUSED_MEMORY_TARGET')
 assert COUNTS['records']<=69280 and COUNTS['targets']<=207840
 assert COUNTS['query_pairs']<=116042 and COUNTS['query_output_distributions']<=812294
 assert COUNTS['preflight_records']<=64 and COUNTS['preflight_targets']<=192 and COUNTS['preflight_query_pairs']<=128

def target_record(answer,kind='main',write=False,prompt=b''):
 COUNTS['records']+=1;COUNTS['targets']+=len(answer)+1;COUNTS[kind+'_records']+=1;COUNTS[kind+'_targets']+=len(answer)+1
 COUNTS['training_prompt_bytes']+=len(prompt);COUNTS['revealed_reply_payload_bytes']+=len(answer);COUNTS['revealed_EOS']+=1
 COUNTS['learning_records' if write else 'inhibited_or_diagnostic_records']+=1
 budget()
def query_count(kind):
 COUNTS['query_pairs']+=1;COUNTS[kind+'_query_pairs']+=1;budget()
def rng(world,stream):
 seed=int.from_bytes(hashlib.sha256(f'V152|{world}|{stream}'.encode()).digest()[:16],'big');return np.random.Generator(np.random.PCG64(seed))
def dataset(world):
 orbits=[(i,j) for i in range(20) for j in range(i,20)];perm=rng(world,'split').permutation(210);ori=rng(world,'orientation').integers(0,2,210)
 rows=[]
 for k in perm:
  i,j=orbits[int(k)];a,b=(j,i) if ori[k] else (i,j)
  rows.append(dict(id=f'{world}:{int(k)}',orbit=[i,j],prompt=f'{a}+{b}='.encode().hex(),answer=str(a+b).encode().hex()))
 d=dict(world=world,OLD=rows[:64],NEW=rows[64:128],validation=rows[128:160],assessment=rows[160:])
 d['orders']={phase:[rng(world,f'{phase}|pass|{p}').permutation(64).tolist() for p in range(1,9)] for phase in ['OLD','NEW']}
 return d

def read_record(rec):return bytes.fromhex(rec['prompt']),bytes.fromhex(rec['answer'])
def checkpoint(m,name):
 p=H/'local_states'/name;p.parent.mkdir(exist_ok=True);t=time.process_time()
 with p.open('wb') as f:pickle.dump(m,f,protocol=5)
 TIMES['checkpoint_CPU']+=time.process_time()-t;COUNTS['checkpoint_bytes']+=p.stat().st_size
 return {'path':str(p.relative_to(H)),'sha256':sha(p),'bytes':p.stat().st_size}
def reset_transient(m):m.begin()
