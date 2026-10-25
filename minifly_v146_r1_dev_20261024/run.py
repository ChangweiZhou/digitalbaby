import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,time,json,pickle,hashlib,copy,resource
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H.parent;START=time.process_time()
sys.path[:0]=[str(R/'minifly_learned_bank_v1_20261024'),str(R/'minifly_v144_dev_20261024'),str(R/'minifly_v144_r2_exercise_20261024')]
import lb1,task
from reference import seed,ByteTrace
from r2_core import controls
from core import Local,Map
FM=lb1.fastmem;md=FM.Compartment().fly.m;counts={s:int(np.ceil(md.active_fraction*np.sum(md.kc_side==s))) for s in [0,1]}
def save(n,v):(H/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def event(v):
 with (H/'LEDGER.jsonl').open('a') as f:f.write(json.dumps(dict(cpu=time.process_time()-START,**v))+'\n');f.flush();os.fsync(f.fileno())
class Trace:
 def __init__(self,table):self.table=table.copy();self.perm=np.random.default_rng(146).permutation(32);self.cache={};self.g=np.sqrt(np.mean(np.sum(FM.BC.R_TABLE**2,axis=1)))
 def context(self,p,prefix=b''):
  key=(bytes(p),bytes(prefix))
  if key in self.cache:return self.cache[key]
  fe=FM.FEPerm();fe.t=0.;v=np.zeros(32);t=0.
  for b in bytes(p):
   t+=lb1.DELTA;fe.advance(t);fe.feed(b,t);v=.95*v;v=v[self.perm]+self.table[b]
  t+=2*lb1.DELTA;fe.advance(t);v*=np.exp(-2*lb1.DELTA/fe.tau)
  for b in bytes(prefix):
   fe.feed(b,t);v=v[self.perm]+self.table[b];t+=lb1.DELTA;fe.advance(t);v*=np.exp(-lb1.DELTA/fe.tau)
  out=np.r_[fe.read(),self.g*v]
  if len(self.cache)>=4096:self.cache.pop(next(iter(self.cache)))
  self.cache[key]=out;return out
 def __getstate__(self):return dict(table=self.table,perm=self.perm,g=self.g,cache={})
 def __setstate__(self,d):self.__dict__.update(d)
def born(w,alpha,rho):return Local(seed('V146:'+w),md.pn_type_index,md.B,md.kc_side,counts,alpha,rho)
def gen(m,t,p):
 out=bytearray()
 for _ in range(8):
  v=m.margins(t.context(p,out));y=sum(1<<j for j,a in enumerate(v) if a>0)
  if y==256:return bytes(out),'EOS'
  if y>256:return bytes(out),'INVALID'
  out.append(y)
 return bytes(out),'TRUNCATED'
def score(m,t,w,which):
 before=m.digest();rows=[];orbits=w['held'] if which=='held' else w['training_order']
 for pair in task.oriented(orbits):
  out,status=gen(m,t,task.prompt(pair));rows.append(dict(pair=list(pair),output_hex=out.hex(),status=status,exact=status=='EOS' and out==task.answer(pair)))
 groups={}
 for x in rows:groups.setdefault(tuple(sorted(x['pair'])),[]).append(x['exact'])
 assert before==m.digest()
 return dict(accuracy=float(np.mean([np.mean(v) for v in groups.values()])),rows=rows)
def one(m,t,p):
 pre=bytearray()
 for y in list(task.answer(p))+[256]:
  m.observe(t.context(task.prompt(p),pre),y)
  if y!=256:pre.append(y)
 m.records+=1
# C1 invariants, no arithmetic rule in core.
tr=Trace(np.zeros((256,32)));parent=ByteTrace(FM.BC.R_TABLE,FM.PERM,lb1.DELTA,FM.FEPerm)
for x in range(256):assert np.array_equal(tr.context(bytes([x]))[:88],parent.context(bytes([x])))
m=born('fixture',.03,.03);p=np.random.default_rng(10).normal(size=120);a,ix=m.code(p);raw=np.asarray(md.B.T@(m.V@np.maximum(a,0))[md.pn_type_index]).ravel();native=lb1.topk_side(md,raw,counts,False);assert np.array_equal(ix,np.flatnonzero(native))
for j in range(10):m.observe(p,j)
ix=m.code(p)[1];v=m.margins(p);m.rest();assert np.array_equal(ix,m.code(p)[1]);assert np.allclose(m.margins(p),v*.97)
x=Map(FM.BC.R_TABLE,88)
for j in range(32):x.observe(j,np.zeros(88))
assert np.allclose(np.linalg.norm(x.recall(255)),1)
save('CYCLE1.json',dict(identity_all_256_equal=True,topk_matches_parent=True,shrink_KC_membership_invariant=True,shrink_margins_scale=True,generic_map_fixture_updates=32,naming_fixture_targets=10))
files={str(Path(v.__file__).resolve()) for v in list(sys.modules.values()) if getattr(v,'__file__',None) and str(v.__file__).endswith('.py') and Path(v.__file__).resolve().is_relative_to(R)}
files|={str(p) for p in H.glob('*.py')};files|={str(R/'minifly_a_v3/package/FULL151_CANONICAL_B.npz')}
lock={str(Path(p).relative_to(R)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(files)};save('SOURCE_LOCK.json',lock)
results=[]
for wn in ['V146-D1','V146-D2']:
 w=task.world(wn);pairs=task.oriented(w['training_order']);short=controls(w,40,parent);best=max(v['orbit_complete'] for v in short.values());save(wn+'_CONTROLS.json',short)
 for alpha in [.003,.03]:
  for arm in ['SYMBOL','ORACLE']:
   tab=np.zeros((256,32))
   if arm=='ORACLE':
    for k,b in enumerate(b'0123456789'):
     bump=np.exp(-(np.arange(32)-31*k/9)**2/2);tab[b]=bump/np.linalg.norm(bump)
    for b in set(range(256))-set(b'0123456789'):tab[b]=np.ones(32)/np.sqrt(32)
   for rho in [0,.01,.03]:
    ident=f'{wn}_{arm}_{alpha}_{rho}';event(dict(event='START',id=ident));t=Trace(tab);m=born(wn,alpha,rho);rng=np.random.default_rng(seed('order:'+wn));curve=[];begin=time.process_time()
    for pas in range(1,1025):
     for j in rng.permutation(len(pairs)):one(m,t,pairs[j])
     if pas>256 and (pas-256)%5==0:m.rest()
     if pas in [16,64,256,512,1024]:
      ss={label:score(m,t,w,label) for label in ['taught','held']};curve.append(dict(pas=pas,**ss));save('ACTIVE.json',dict(id=ident,pass_number=pas,cpu=time.process_time()-START))
     if pas%16==0:event(dict(event='PROGRESS',id=ident,passes=pas,records=m.records,targets=m.targets))
     if time.process_time()-START>21000:raise RuntimeError('Six CPU-hour admission reserve reached')
    row=dict(id=ident,world=wn,arm=arm,alpha=alpha,rho=rho,records=m.records,targets=m.targets,cpu=time.process_time()-begin,strongest_shortcut=best,curve=curve);save(ident+'.json',row);results.append(row);event(dict(event='COMMIT',id=ident,records=m.records,targets=m.targets));save('COMPACT.json',[dict(id=z['id'],held=z['curve'][-1]['held']['accuracy'],taught=z['curve'][-1]['taught']['accuracy'],cpu=z['cpu']) for z in results])
# C3 fresh resume fixture, not a full selected ten-arm rehearsal.
w=task.world('V146-TECH-C3');pairs=task.oriented(w['training_order']);t=Trace(np.zeros((256,32)));m=born(w['name'],.03,.01);rng=np.random.default_rng(42)
def pass1(m,t,rng):
 for j in rng.permutation(len(pairs)):one(m,t,pairs[j])
pass1(m,t,rng);blob=pickle.dumps((m,t,rng));pass1(m,t,rng);m2,t2,rng2=pickle.loads(blob);pass1(m2,t2,rng2);assert m.digest()==m2.digest()
for name,h in lock.items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h
ready=[]
for alpha in [.003,.03]:
 for rho in [.01,.03]:
  cells=[z for z in results if z['arm']=='ORACLE' and z['alpha']==alpha and z['rho']==rho];ready.append(dict(alpha=alpha,rho=rho,pass_ready=all(z['curve'][-1]['taught']['accuracy']>=.9 and z['curve'][-1]['held']['accuracy']>=z['strongest_shortcut']+.2 for z in cells)))
save('SUMMARY.json',dict(status='REFERENCE_PIPELINE_READY' if any(x['pass_ready'] for x in ready) else 'REFERENCE_PIPELINE_NOT_READY',readiness=ready,completed_readiness_fits=len(results),addition_records=sum(z['records'] for z in results)+3*len(pairs),observed_addition_targets=sum(z['targets'] for z in results)+3*sum(len(task.answer(p))+1 for p in pairs),synthetic_map_payload_updates=32,synthetic_naming_targets=10,cpu_including_imports=time.process_time()-START,resume_equal=True,main_worlds_opened=False,map_curriculum_trained=False,full_D4_rehearsal=False))
