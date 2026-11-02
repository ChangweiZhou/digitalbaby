"""V175 native-only LPS-16. L learns solely from pre-target cue activity."""
import os,sys,time,json,hashlib,resource,pickle
from pathlib import Path
from collections import Counter
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parent;START=time.process_time()
sys.path.insert(0,str(ROOT/'minifly_writer_interference_dev_v1_20261027'))
import writerlib as W
import numpy as np
F=W.F;C=W.C;EOS=256
SPEC=json.loads((H/'EXECUTION_SPEC.json').read_text());AUTH=json.loads((H/'AUTHORIZATION.json').read_text())
prior=json.loads((H/'COUNTS.json').read_text()) if (H/'COUNTS.json').exists() else {};PRIOR_CPU=prior.get('CPU',0.)
W.H=H;W.AUTH=AUTH;W.SPEC=SPEC;W.COUNTS=Counter(prior.get('counts',{}));W.TIMES=Counter(prior.get('timings',{}));W.PRIOR_CPU=0.;W.TIER='C0';W.ASSETS={};W.ASSET_IDS={}
COUNTS=W.COUNTS;TIMES=W.TIMES;PEAK_PRIOR=prior.get('peak_RSS_bytes',0);ACTIVE=None
# No historical module is edited; redirect only its telemetry globals before any model birth.
def cpu():return PRIOR_CPU+time.process_time()-START
def budget():
 checks={'CPU_CAP':cpu()<7200,'WALL_CAP':time.time()-AUTH['epoch']<14400,'RSS_CAP':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1073741824,'RECORD_CAP':COUNTS['full_records']<=5000,'TECH_CAP':COUNTS['technical_records']<=32,'READ_CAP':COUNTS['native_vector_reads']<=100000,'PROBE_CAP':COUNTS['own_TF_pairs']<=2700,'SHADOW_CAP':COUNTS['shadow_targets']<=160}
 for k,v in checks.items():
  if not v:raise W.StopProtocol(k)
W.cpu=cpu;W.budget=budget
def status(state,**kw):
 W.save('COUNTS.json',dict(counts=dict(COUNTS),timings=dict(TIMES),CPU=cpu(),wall=time.time()-AUTH['epoch'],peak_RSS_bytes=max(PEAK_PRIOR,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))
 W.save('STATUS.json',dict(identity=SPEC['identity'],status=state,tier=W.TIER,**kw))
W.status=status
def rng(world,stream):return np.random.default_rng(int.from_bytes(hashlib.sha256(f'V175|{world}|{stream}'.encode()).digest()[:16],'big'))
W.rng=rng
def selected(rows,w,purpose,n):return sorted(rows,key=lambda r:hashlib.sha256(f'V175|{w}|{purpose}|{r["id"]}'.encode()).hexdigest())[:n]
W.selected=selected

def topology(side):
 edges=[]
 for s in (0,1):
  ix=np.flatnonzero(side==s);g=np.random.default_rng(int.from_bytes(hashlib.sha256(f'V175|topology|{s}'.encode()).digest()[:16],'big'));ix=g.permutation(ix)
  for shift in range(1,9):
   j=np.roll(ix,shift);edges.extend(zip(np.minimum(ix,j).tolist(),np.maximum(ix,j).tolist()))
 e=np.array(sorted(set(edges)),dtype=np.int32);assert len(e)==8*len(side)
 deg=np.bincount(e.ravel(),minlength=len(side));assert np.all(deg==16) and np.all(side[e[:,0]]==side[e[:,1]])
 return e[:,0].copy(),e[:,1].copy()

def update_edges(weights,i,j,z0):
 """All-edge, cue-only coactivation trace. No targets or phase metadata available."""
 out=np.minimum(1.,(1.-1/2048)*weights.astype(np.float64)+(1/32)*z0[i]*z0[j]);weights[:]=out.astype(np.float32)

class Model(W.Arm):
 def __init__(self,enabled,gamma=(0.,0.)):
  super().__init__();self.enabled=bool(enabled);self.gamma=np.array(gamma,dtype=np.float64);self.L=None;self.i=None;self.j=None;self.update_count=0;self.suppress_updates=False
  if self.enabled:
   self.i,self.j=topology(self.bank[0].fly.m.kc_side);self.L=np.zeros(len(self.i),dtype=np.float32)
 def clone(self):
  out=super().clone();out.gamma=self.gamma.copy()
  if self.enabled:out.L=self.L.copy();COUNTS['lateral_clone_bytes']+=self.L.nbytes
  return out
 def rawcode(self,p):
  COUNTS['KC_encodings']+=1;return F.FastMemory.encode(self,p)
 def encode(self,p):
  t=time.process_time();z0,u=self.rawcode(p)
  if not self.enabled:TIMES['BASE_encoding_CPU']+=time.process_time()-t;return z0,u
  side=self.bank[0].fly.m.kc_side
  d=np.bincount(self.i,weights=self.L*z0[self.j],minlength=len(z0))+np.bincount(self.j,weights=self.L*z0[self.i],minlength=len(z0))
  score=u-self.gamma[side]*d;x=np.zeros(len(u))
  for s in (0,1):
   ix=np.flatnonzero((side==s)&(u>0));order=np.lexsort((ix,-score[ix]));x[ix[order[:min(len(ix),self.side_n[s])]]]=1.
   assert x[side==s].sum()==z0[side==s].sum()
  COUNTS['inhibition_edge_visits']+=len(self.i);TIMES['LPS_encoding_CPU']+=time.process_time()-t;return x,u
 def cue_before_target(self):
  f=self.fe.clone();f.advance(self.time+F.DELTA);return self.rawcode(f.read())[0]
 def after_observed_record(self,z0):
  if self.enabled and not self.suppress_updates:
   t=time.process_time();update_edges(self.L,self.i,self.j,z0);self.update_count+=1;COUNTS['lateral_record_updates']+=1;COUNTS['lateral_update_edge_visits']+=len(self.i);TIMES['lateral_update_CPU']+=time.process_time()-t
 def train_row(self,row,write=True,shadow=None):
  global ACTIVE
  ACTIVE=self;t=time.process_time();self._prompt(bytes.fromhex(row['prompt']));cue=self.cue_before_target() if self.enabled else None;beforeL=None if self.L is None else hashlib.sha256(self.L.tobytes()).hexdigest();sites=[]
  for site,y in enumerate([*bytes.fromhex(row['answer']),EOS]):
   v=self.values();p,lp=W.prob(v);coef=W.warm(self.n)*W.direction(v,p,y,'D')
   if site==0 and shadow is not None:shadow(self,row,y,coef)
   self.teach_once(y,coef,write)
   if y!=EOS:self.fe.feed(y,self.time)
   sites.append(dict(y=y,n=self.n,w=W.warm(self.n),lp=float(lp[y]),argmax=int(p.argmax()),coef_max=float(np.max(abs(coef))),time=self.time));self.n+=1;COUNTS[W.TIER+'_targets']+=1
  if self.L is not None:assert hashlib.sha256(self.L.tobytes()).hexdigest()==beforeL
  self.after_observed_record(cue);self.records+=1;COUNTS['full_records']+=1;COUNTS[W.TIER+'_records']+=1;TIMES[('LPS' if self.enabled else 'BASE')+'_train_CPU']+=time.process_time()-t
  W.log('train_record',id=row['id'],LPS=self.enabled,write=write,n=self.n,native_time=self.time,L_updates=self.update_count,sites=sites);ACTIVE=None;return sites

parent_digest=W.digest
def digest(m):
 base=parent_digest(m);z=hashlib.sha256(base.encode())
 if hasattr(m,'enabled'):
  z.update(pickle.dumps((m.enabled,m.gamma,m.update_count,m.suppress_updates),protocol=5))
  if m.enabled:z.update(m.L.tobytes())
 return z.hexdigest()
W.digest=digest

def codes(m,rows):
 before=digest(m);a=[];raw=[]
 for row in rows:
  q=m.clone();q._prompt(bytes.fromhex(row['prompt']));f=q.fe.clone();f.advance(q.time+F.DELTA);x,u=q.encode(f.read());a.append(x);raw.append(u);COUNTS['code_probe_episodes']+=1
 assert digest(m)==before;return np.array(a),np.array(raw)
def jaccard(A,B):
 inter=A@B.T;den=A.sum(1)[:,None]+B.sum(1)[None,:]-inter;return np.divide(inter,den,out=np.zeros_like(inter),where=den>0)
def conflicts(X,rows):
 found=[]
 for i in range(len(X)):
  for j in range(i):
   if np.array_equal(X[i],X[j]) and rows[i]['answer']!=rows[j]['answer']:found.append([rows[i]['id'],rows[j]['id']])
 return found

def self_margin(v,y):
 p,lp=W.prob(v);return float(lp[y]-np.max(np.delete(lp,y)))
def shadow(pre,row,y,coef,sentinels,tag):
 zero=pre.clone();written=pre.clone();pre_syn=W.synstate(pre);zero.teach_once(y,np.zeros(16),False);written.teach_once(y,coef,True);COUNTS['shadow_targets']+=2
 x=W.querycode(zero);xx=W.querycode(written);assert np.array_equal(x,xx) and zero.time==written.time and W.nonvalue(zero)==W.nonvalue(written)
 if pre.enabled:assert np.array_equal(pre.L,zero.L) and np.array_equal(pre.L,written.L)
 z=W.fixedread(zero,x,zero.time+F.DELTA);v=W.fixedread(written,x,written.time+F.DELTA)
 a=W.panel(zero,sentinels,tag+'_zero');b=W.panel(written,sentinels,tag+'_written')
 rr0=json.loads((H/a['path']).read_text());rr1=json.loads((H/b['path']).read_text());damage=[dict(id=r0['id'],zero_margin=sum(t['margin'] for t in r0['tf'])/2,written_margin=sum(t['margin'] for t in r1['tf'])/2) for r0,r1 in zip(rr0,rr1)]
 return dict(tag=tag,id=row['id'],same_code=True,same_time=True,time=zero.time+F.DELTA,query_code_hash=hashlib.sha256(x.tobytes()).hexdigest(),zero_panel=a,written_panel=b,sentinels=damage,D=float(np.mean([r['zero_margin']-r['written_margin'] for r in damage])),self_target_margin_change=self_margin(v,y)-self_margin(z,y),delivered_d=W.displacement(v,z),passive_zero_synaptic_L2=float(np.linalg.norm(W.synstate(zero)-pre_syn)),write_minus_zero_synaptic_L2=float(np.linalg.norm(W.synstate(written)-W.synstate(zero))))
