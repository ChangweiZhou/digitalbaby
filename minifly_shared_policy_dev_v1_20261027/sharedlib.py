"""SP-COND-V1 instruments. No native teaching, attention or task semantics."""
import os,sys,time,json,pickle,hashlib,resource
from pathlib import Path
from collections import Counter,deque
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
T0=time.process_time();W0=time.monotonic();H=Path(__file__).resolve().parent;ROOT=H.parent
import numpy as np
SPEC=json.loads((H/'EXECUTION_SPEC.json').read_text());COUNTS=Counter();TIMES=Counter();PRIOR_CPU=0.;TIER='initialization'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [plain(v) for v in x]
 return x
def save(name,x):
 p=H/name;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(plain(x),indent=2,allow_nan=False)+'\n');q.replace(p)
def cpu():
 r=resource.getrusage(resource.RUSAGE_CHILDREN)
 return PRIOR_CPU+time.process_time()-T0+r.ru_utime+r.ru_stime
def wall():return time.monotonic()-W0
def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
def budget(*a,**kw):
 assert cpu()<3600,'CPU_CAP';assert wall()<7200,'WALL_CAP';assert rss()<1073741824,'RSS_CAP'
 assert COUNTS['distributions']<=120000,'DISTRIBUTION_CAP';assert COUNTS['native_vector_reads']<=120000,'NATIVE_READ_CAP'
def log(event,**x):
 with (H/'LEDGER.jsonl').open('a') as f:f.write(json.dumps(plain(dict(event=event,tier=TIER,CPU=cpu(),wall=wall(),**x)),allow_nan=False)+'\n')
def checkpoint(x,name):
 p=H/'local'/name;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp')
 with q.open('wb') as f:pickle.dump(x,f,protocol=5)
 q.replace(p);return dict(path=str(p.relative_to(H)),sha256=sha(p),bytes=p.stat().st_size)
def finish(status):save('COUNTS.json',dict(counts=dict(COUNTS),timings=dict(TIMES),CPU=cpu(),wall=wall(),peak_RSS_bytes=rss()));save('STATUS.json',dict(identity='SP-COND-V1',status=status,CPU=cpu(),wall=wall(),counts=dict(COUNTS)))
sys.path.insert(0,str(ROOT/'minifly_v171_dev_20261026'))
import support171 as parent_support
parent_support.COUNTS=COUNTS;parent_support.TIMES=TIMES;parent_support.budget=budget;parent_support.H=H;parent_support.cpu=cpu;parent_support.wall=wall
import native171 as N
from controller171 import Controller,PARAMS,CAPS,warm
ARMS=('SGD3','ADAM3','ADAM1','ADAM3-R0','ADAM3-CONST')
class Shared(Controller):
 def __init__(self,world,arm,carrier):
  assert arm in ARMS;super().__init__(world,'EXACT3',N.C);self.arm=arm;self.carrier=np.asarray(carrier).copy();self.n_train=0;self.opt_t=0
  self.mom1={k:np.zeros_like(getattr(self,k)) for k in PARAMS};self.mom2={k:np.zeros_like(getattr(self,k)) for k in PARAMS}
  if arm=='ADAM3-R0':self.R.fill(0)
 def effective(self,v):return self.carrier if self.arm=='ADAM3-CONST' else v
 def step(self,b,v):return super().step(b,self.effective(v))
 def refresh(self,v):return super().refresh(self.effective(v))
 def exact_credit(self,y):
  n=self.n_train if self.arm=='SGD3' else 2048
  if self.arm=='ADAM1':
   frames=self.frames;self.frames=deque(list(frames)[-1:]);out=super().credit(y,n,mode='EXACT3');self.frames=frames
  else:out=super().credit(y,n,mode='EXACT3')
  if self.arm=='ADAM3-R0':self.grads['R'].fill(0)
  return out
 def update(self,ws):
  norms={k:float(np.linalg.norm(self.grads[k])) for k in PARAMS};changes={};project=0;clipped=0;eta=2**-10
  if self.arm=='SGD3':
   before={k:getattr(self,k).copy() for k in PARAMS};r=super().apply()
   changes={k:float(np.linalg.norm(getattr(self,k)-before[k])) for k in PARAMS};project=r['projected'];clipped=r['clipped']
  else:
   self.opt_t+=1;w=float(np.mean(ws));self.last_w=w
   for k in PARAMS:
    a=getattr(self,k);g=self.grads[k]
    if self.arm=='ADAM3-R0' and k=='R':self.mom1[k].fill(0);self.mom2[k].fill(0);a.fill(0);changes[k]=0.;continue
    self.mom1[k]*=.9;self.mom1[k]+=.1*g;self.mom2[k]*=.999;self.mom2[k]+=.001*g*g
    direction=(self.mom1[k]/(1-.9**self.opt_t))/(np.sqrt(self.mom2[k]/(1-.999**self.opt_t))+1e-8)
    clipped+=int(np.count_nonzero(abs(direction)>1));old=a.copy();proposed=a-eta*w*np.clip(direction,-1,1);a[:]=np.clip(proposed,-CAPS[k],CAPS[k]);project+=int(np.count_nonzero(proposed!=a));changes[k]=float(np.linalg.norm(a-old));assert np.max(abs(a-old))<=eta*w+1e-12
   COUNTS['Adam_record_updates']+=1
  if self.arm=='ADAM3-R0':assert not self.R.any() and not self.mom1['R'].any() and not self.mom2['R'].any()
  assert all(np.isfinite(getattr(self,k)).all() and np.max(abs(getattr(self,k)))<=CAPS[k]+1e-14 for k in PARAMS)
  COUNTS['optimizer_parameter_visits']+=sum(getattr(self,k).size for k in PARAMS)
  return dict(gradient_norm=norms,applied_step_norm=changes,projected=project,direction_or_gradient_clipped=clipped,opt_t=self.opt_t,mean_w=float(np.mean(ws)),min_w=float(min(ws)),max_w=float(max(ws)),parameter_hash=self.state_hash())
def collect(model,prompt,answer,identifier):
 """A new counted no-write native episode, independent of controller and target selection."""
 q=model.clone();q._control(True);pv=[]
 for b in prompt:q.step(b);pv.append(N.raw(q,'cache_prompt'))
 q._control(False);boundary=N.raw(q,'cache_boundary');targets=[];v=boundary
 for y in list(answer)+[256]:
  targets.append(v.copy())
  if y!=256:q.step(y);v=N.raw(q,'cache_reply_prefix')
 COUNTS[TIER+'_cache_episodes']+=1;COUNTS[TIER+'_cache_prompt_bytes']+=len(prompt);COUNTS[TIER+'_cache_target_sites']+=len(targets)
 return dict(id=identifier,prompt=prompt.hex(),answer=answer.hex(),prompt_v=np.asarray(pv),boundary_v=boundary.copy(),target_v=np.asarray(targets))
def fit_record(c,row):
 c.reset();start=c.n_train
 for b,v in zip(bytes.fromhex(row['prompt']),row['prompt_v']):c.step(b,v)
 c.refresh(row['boundary_v']);ws=[];targetrows=[]
 for j,y in enumerate(list(bytes.fromhex(row['answer']))+[256]):
  w=warm(c.n_train);info,arrays=c.exact_credit(y);ws.append(w);targetrows.append((dict(site=j,y=y,n=c.n_train,actual_w=w,**{k:v for k,v in info.items() if k not in ('y',)}),{k:np.asarray(arrays[k]).copy() for k in ('p','v','g','h','a','direct_error','signal_norms')}));c.n_train+=1;COUNTS[TIER+'_fit_targets']+=1
  if y!=256:c.step(y,row['target_v'][j+1])
 update=c.update(ws);COUNTS[TIER+'_fit_records']+=1;out=dict(n_start=start,n_end=c.n_train,loss_nats=sum(-x[0]['lp'] for x in targetrows),TF_correct=sum(x[0]['argmax']==x[0]['y'] for x in targetrows),targets=len(targetrows),**update);c.reset();budget();return out,targetrows
def lock():
 p=ROOT/'minifly_v171_dev_20261026/CALIBRATION_SOURCE_LOCK.json';old=json.loads(p.read_text());assert all(sha(ROOT/f)==v for f,v in old.items())
 files=dict(old)
 for module in list(sys.modules.values()):
  f=getattr(module,'__file__',None)
  if f and Path(f).resolve().is_relative_to(ROOT) and Path(f).suffix=='.py':files[str(Path(f).resolve().relative_to(ROOT))]=sha(f)
 for f in H.glob('*.py'):files[str(f.relative_to(ROOT))]=sha(f)
 for f in ('EXECUTION_SPEC.json','AUTHORIZATION.json'):files[str((H/f).relative_to(ROOT))]=sha(H/f)
 return files
def evaluate(model,controllers,rows,tag):
 receipts={}
 for arm,c in controllers.items():
  values=[]
  for row in rows:
   o=N.pair(model,c,bytes.fromhex(row['prompt']),bytes.fromhex(row['answer']));values.append(dict(id=row['id'],**o))
  path=f'local/panels/{tag}_{arm}.json';save(path,dict(tag=tag,arm=arm,rows=values));receipts[arm]=dict(path=path,sha256=sha(H/path),correct=sum(x['correct'] for x in values),correct_ids=[x['id'] for x in values if x['correct']],records=len(values),TF_targets=sum(len(x['teacher_forced']) for x in values),TF_nll=sum(-t['lp'] for x in values for t in x['teacher_forced']),empty_EOS=sum(x['EOS'] and not x['output_hex'] for x in values),CAP=sum(x['CAP'] for x in values))
  budget()
 return receipts
