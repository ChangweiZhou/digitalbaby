"""Bounded technical exercise, not the V145 main screen."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,time,json,hashlib,pickle,copy,resource,subprocess
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(ROOT/'minifly_v144_r2_exercise_20261024'),str(ROOT/'minifly_v144_dev_20261024')]
from r2_core import trace_from_parent,controls
from reference import ByteTrace,Reference,seed
import task
from kernel import Code
START=time.process_time(); ledger=[]
def save(n,v):
 (HERE/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def clock():return time.process_time()-START
def lock():
 files=list(HERE.glob('*.py'))+[ROOT/'minifly_v144_dev_20261024'/x for x in ['reference.py','task.py']]
 for m in list(sys.modules.values()):
  p=getattr(m,'__file__',None)
  if p and Path(p).suffix=='.py' and Path(p).resolve().is_relative_to(ROOT):files.append(Path(p).resolve())
 return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in set(files)}
base=trace_from_parent(); old=base.table.copy();lb=sys.modules['lb1'];FM=lb.fastmem
class Trace(ByteTrace):
 def __init__(self,table):
  self.cache={};tab=table.copy()
  class FE(FM.FEPerm):
   def feed(self,b,t):self.advance(t);self.p=self.p[FM.PERM]+tab[int(b)]
  super().__init__(tab,FM.PERM,lb.DELTA,FE)
 def context(self,p,prefix=b''):
  key=(bytes(p),bytes(prefix))
  if key not in self.cache:self.cache[key]=super().context(*key)
  return self.cache[key]
def table(c):
 raw=c.z[:256]@c.U.T
 # Birth calibration, never learned/held-state normalization.
 b=Code(seed('code:'+CURRENT));birth=b.z[:256]@b.U.T
 return raw*np.sqrt(np.mean(old**2)/np.mean(birth**2))
def traincode(arm,name,records):
 global CURRENT
 CURRENT=name;c=Code(seed('code:'+name),mode='BP' if arm=='ORDER-BP' else 'LOCAL');r=np.random.default_rng(seed('count:'+name));perm=r.permutation(list(b'0123456789'))
 while list(perm) in [list(b'0123456789'),list(b'9876543210')]:perm=r.permutation(list(b'0123456789'))
 t=time.process_time()
 for _ in range(records):
  seq=list(b'0123456789')
  if arm=='PERM':seq=list(perm)
  if arm=='BAG':seq=list(r.permutation(seq))
  if arm!='NO-WRITE':
   for x,y in zip(seq,seq[1:]+[256]):c.observe(x,y)
 ledger.append(dict(kind='count',arm=arm,world=name,records=records,presented_tokens=11*records,targets=10*records,cpu=time.process_time()-t))
 return c,table(c)
def score(m,w):
 before=m.state_hash();v={k:task.score(m,task.oriented(p)) for k,p in [('held',w['held']),('taught',w['training_order'])]};assert before==m.state_hash()
 return v
def fit(tab,w,arm,mode,passes):
 m=Reference(Trace(tab),'V145:'+w['name'],128,.03,.001,mode);r=np.random.default_rng(seed('order:'+w['name']));pairs=task.oriented(w['training_order']);t=time.process_time()
 # Never pass evaluator targets to generation; cache is inputs only.
 for i in range(passes):
  for j in r.permutation(len(pairs)):
   p=pairs[j];m.train(task.prompt(p),task.answer(p))
  if clock()>1200:raise RuntimeError('technical exercise CPU boundary')
 v=score(m,w);ledger.append(dict(kind='addition',arm=arm,mode=mode,world=w['name'],records=m.records,targets=m.target_tokens,cpu=time.process_time()-t))
 return dict(arm=arm,mode=mode,passes=passes,records=m.records,targets=m.target_tokens,scores=v),m,r
# Cycle 1: equation and complete byte-interface audits.
t=time.process_time();c=Code(7,mode='BP');target=c.z[19].copy();g=c.gradients(8,19);err=0
for k in ['D','T','c','b','z']:
 a=c.z[8] if k=='z' else getattr(c,k)
 for ix in np.ndindex(a.shape):
  x=a[ix];eps=1e-6;a[ix]=x+eps;plus=c.loss(8,19,target);a[ix]=x-eps;minus=c.loss(8,19,target);a[ix]=x;err=max(err,abs((plus-minus)/(2*eps)-g[k][ix]))
assert err<1e-7
tr=Trace(old)
for x in range(256):assert np.array_equal(tr.context(bytes([x])),base.context(bytes([x])))
before=c.digest();z=c.z.copy();c.observe(8,19);assert np.array_equal(z,c.z) # zero-rate first update
c.observe(8,19);assert np.array_equal(c.z[256],z[256])
a=copy.deepcopy(c);b=pickle.loads(pickle.dumps(c));a.observe(8,256);b.observe(8,256);assert a.digest()==b.digest()
world=task.world('V145-EXERCISE-C2');assert not set(map(tuple,world['held']))&set(map(tuple,world['training_order']))
locks=lock();save('SOURCE_LOCK.json',locks)
save('CYCLE1.json',dict(gradient_max_error=err,all_256_bytes_parent_equal=True,first_update_zero=True,EOS_target_fixed=True,pickle_resume_equal=True,record_applications=4,cpu=time.process_time()-t))
# Cycle2 actual-interface oracle check; independent evaluator priors only.
results=[]
for arm in ['SYMBOL','ORACLE-LINE','ORACLE-CIRCLE']:
 tab=old.copy()
 if arm!='SYMBOL':
  raw=np.zeros((10,4))
  for a in range(10):
   if arm=='ORACLE-LINE':raw[a,0]=a/9
   else:raw[a,:2]=[np.cos(2*np.pi*a/10),np.sin(2*np.pi*a/10)]
  U=Code(seed('oracle')).U;rr=raw@U.T;rr*=np.sqrt(np.mean(old**2)/np.mean(rr**2));tab[np.array(list(b'0123456789'))]=rr
 v,_,_=fit(tab,world,arm,'BP',256);results.append(v);save('PROGRESS.json',dict(stage='C2 oracle',arm=arm,cpu=clock()))
save('CYCLE2_ORACLE.json',results)
# Branch explicitly: weak oracle is scoped; still qualify the proposed predictor,
# but do not run a full menu/main or claim pretraining efficacy.
for arm in ['ORDER','PERM','BAG','NO-WRITE','ORDER-BP']:
 c,tab=traincode(arm,'V145-EXERCISE-C2',2048)
 for mode in ['BP','DFA']:
  v,_,_=fit(tab,world,arm,mode,64);results.append(v)
 save('PROGRESS.json',dict(stage='C2 predictor',arm=arm,cpu=clock()))
save('CYCLE2.json',results)
# C3 fresh developmental rehearsal; shortened dose, not exact main protocol.
w=task.world('V145-EXERCISE-C3');c,tab=traincode('ORDER',w['name'],128)
v,m,r=fit(tab,w,'ORDER','BP',4)
state=pickle.dumps((m,r));m1,r1=pickle.loads(state);m2,r2=pickle.loads(state)
pairs=task.oriented(w['training_order'])
for model,rng in [(m1,r1),(m2,r2)]:
 for j in rng.permutation(len(pairs)):p=pairs[j];model.train(task.prompt(p),task.answer(p))
assert m1.state_hash()==m2.state_hash()
ledger.append(dict(kind='resume_rehearsal',records=2*len(pairs),targets=2*sum(len(task.answer(p))+1 for p in pairs),cpu=0,cpu_included_in_total=True))
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in locks.items())
# recompute every emitted answer from raw hex; no best-so-far aggregation.
for row in results+[v]:
 for group in row['scores'].values():
  for rr in group['rows']:assert rr['exact']==(rr['status']=='EOS' and bytes.fromhex(rr['output_hex'])==task.answer(rr['pair']))
save('CYCLE3.json',dict(fresh_developmental_world=w['name'],shortened_rehearsal=True,resume_equal=True,source_lock_matches=True,saved_output_scores_verified=True,main_worlds_opened=False,full_main_dispatcher_qualified=False))
save('LEDGER.json',ledger)
save('SUMMARY.json',dict(status='THREE_BOUNDED_CYCLES_COMPLETE_WITH_LIMITATIONS',main_worlds_opened=False,formal_jobs=0,records=sum(x['records'] for x in ledger)+4,targets=sum(x.get('targets',0) for x in ledger)+4,process_cpu=clock(),peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,results=[dict(arm=x['arm'],mode=x['mode'],passes=x['passes'],held=x['scores']['held']['orbit_exact'],taught=x['scores']['taught']['orbit_exact']) for x in results]))
