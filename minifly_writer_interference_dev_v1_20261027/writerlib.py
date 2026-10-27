"""WCFI-V1: local native writers and separately charged clone-only instrumentation."""
import os,sys,time,json,hashlib,pickle,resource,copy,itertools
from pathlib import Path
from collections import Counter
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parent;START=time.process_time()
import numpy as np
SPEC=json.loads((H/'EXECUTION_SPEC.json').read_text());AUTH=json.loads((H/'AUTHORIZATION.json').read_text())
PRIOR=json.loads((H/'COUNTS.json').read_text()) if (H/'COUNTS.json').exists() else {}
COUNTS=Counter(PRIOR.get('counts',{}));TIMES=Counter(PRIOR.get('timings',{}));PRIOR_CPU=PRIOR.get('CPU',0.);TIER='qualification'
sys.path.insert(0,str(ROOT/'minifly_cls_dev_20261017'))
import fastmem as F
C=F.CODE;EOS=256;ASSETS={};ASSET_IDS={}
class StopProtocol(Exception):pass
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [plain(v) for v in x]
 return x
def save(name,x):
 p=H/name;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(plain(x),indent=2,allow_nan=False)+'\n');q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cpu():return PRIOR_CPU+time.process_time()-START
def budget():
 limits={'CPU_CAP':cpu()<3600,'WALL_CAP':time.time()-AUTH['epoch']<7200,'RSS_CAP':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1073741824,'READ_CAP':COUNTS['native_vector_reads']<=160000,'DIST_CAP':COUNTS['distributions']<=120000,'FULL_RECORD_CAP':COUNTS['full_records']<=3472,'SHADOW_CAP':COUNTS['shadow_targets']<=4096,'PROBE_CAP':COUNTS['own_TF_pairs']<=5408}
 for k,v in limits.items():
  if not v:raise StopProtocol(k)
def status(s,**kw):
 save('COUNTS.json',dict(counts=dict(COUNTS),timings=dict(TIMES),CPU=cpu(),wall=time.time()-AUTH['epoch'],peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
 save('STATUS.json',dict(identity='WCFI-V1',status=s,tier=TIER,**kw))
def log(event,**kw):
 with (H/'LEDGER.jsonl').open('a') as f:f.write(json.dumps(plain(dict(event=event,tier=TIER,CPU=cpu(),**kw)),allow_nan=False)+'\n')
 budget()
def rng(w,s):return np.random.default_rng(int.from_bytes(hashlib.sha256(f'WCFI-V1|{w}|{s}'.encode()).digest()[:16],'big'))
def selected(rows,w,purpose,n):return sorted(rows,key=lambda r:hashlib.sha256(f'WCFI-V1|{w}|{purpose}|{r["id"]}'.encode()).hexdigest())[:n]
def data(w):
 a=rng(w,'alphabet').permutation(256).tolist();cue=a[:8];suffix=a[8:10];reply=a[10:26]
 pool=list(itertools.product(cue,repeat=3));ix=rng(w,'prefixes').choice(len(pool),80,replace=False)
 oldlab=rng(w,'OLD-labels').permutation(np.repeat(reply,4));newlab=rng(w,'NEW-labels').permutation(reply)
 rows=[dict(id=f'{w}:{i}',prompt=bytes((*pool[int(j)],*suffix)).hex(),answer=bytes([int((oldlab if i<64 else newlab)[i if i<64 else i-64])]).hex(),cohort='OLD' if i<64 else 'NEW') for i,j in enumerate(ix)]
 return dict(world=w,rows=rows,old_orders=[rng(w,f'OLD-order-{i}').permutation(64).tolist() for i in range(8)]*2,new_orders=[rng(w,f'NEW-order-{i}').permutation(16).tolist() for i in range(8)],alphabet=dict(cue=cue,suffix=suffix,reply=reply))
def prob(v):
 COUNTS['distributions']+=1;z=C@(v-v.mean())/4;lp=z-z.max();lp-=np.log(np.exp(lp).sum());return np.exp(lp),lp
def warm(n):return 1/16+15/16*min(n/1024,1)
def direction(v,p,y,form):
 if form=='N':
  e=p.copy();e[y]-=1;g=C.T@e/4;return g-g.mean()
 assert form=='D';return (np.tanh((v-v.mean())/2)-C[y])/2
class Arm(F.FastMemory):
 def __init__(self):
  super().__init__(front='perm',teach='delta',fresh_adapt=True,homeo=False);self.n=0;COUNTS['births']+=1
 def clone(self):
  t=time.process_time();out=super().clone();out.theta=self.theta.copy();out.freq=self.freq.copy();COUNTS['memory_clones']+=1;COUNTS['mutable_clone_bytes']+=sum(s.fly.m.fast.nbytes+s.fly.m.slow.nbytes+s.fly.m.adapt.nbytes for s in self.bank)+self.theta.nbytes+self.freq.nbytes;TIMES['clone_CPU']+=time.process_time()-t;return out
 def encode(self,p):COUNTS['KC_encodings']+=1;return super().encode(p)
 def values(self):
  t=time.process_time();out=super().values();COUNTS['native_vector_reads']+=1;COUNTS['native_compartment_reads']+=16;TIMES['read_CPU']+=time.process_time()-t;return out
 def teach_once(self,y,coef,write=True):
  assert np.isfinite(coef).all() and np.max(abs(coef))<=1+1e-14
  self._tick();x=self._code()
  for j,s in enumerate(self.bank):s.pending_t,s.pending_x=self.time,x;s.teach_signed(0.,float(coef[j]),self.time,write=write)
  COUNTS['signed_teacher_calls']+=16;COUNTS['enabled_signed_calls' if write else 'inhibited_signed_calls']+=16
  return x
 def train_row(self,row,form='D',gain=1.,write=True,measure=False):
  t=time.process_time();self._prompt(bytes.fromhex(row['prompt']));sites=[]
  for y in [*bytes.fromhex(row['answer']),EOS]:
   v=self.values();p,lp=prob(v);w=warm(self.n);c=w*gain*direction(v,p,y,form)
   if measure:
    zero=self.clone();zero.teach_once(y,np.zeros(16),False);COUNTS['shadow_targets']+=1;oldstate=synstate(zero)
   x=self.teach_once(y,c,write)
   sites.append(dict(y=y,n=self.n,w=w,argmax=int(p.argmax()),lp=float(lp[y]),coef_RMS=float(np.sqrt(np.mean(c*c))),coef_max=float(np.max(abs(c))),time=self.time,code_hash=hashlib.sha256(x.tobytes()).hexdigest()))
   if measure:
    xx=querycode(self);z=fixedread(zero,xx,zero.time+F.DELTA);vv=fixedread(self,xx,self.time+F.DELTA);sites[-1].update(delivered_d=displacement(vv,z),synaptic_L2=float(np.linalg.norm(synstate(self)-oldstate)))
   if y!=EOS:self.fe.feed(y,self.time)
   self.n+=1;COUNTS[TIER+'_targets']+=1
  self.records+=1;COUNTS['full_records']+=1;COUNTS[TIER+'_records']+=1;TIMES['train_CPU']+=time.process_time()-t
  log('train_record',id=row['id'],form=form,gain=gain,write=write,n=self.n,native_time=self.time,sites=sites);return sites

def fixedread(m,x,when):
 """Explicit same-condition reader, independent of front-end prospective scheduling."""
 t=time.process_time();out=[]
 for s in m.bank:
  fly=s.fly.clone();fly.rest(when-m.time);dx=F.NATIVE.common.observed_activity(fly.m,x[None,:]);out.append(float((fly.m.expression(dx)-fly.reader.predict(dx)).mean()))
 COUNTS['native_vector_reads']+=1;COUNTS['fixed_condition_reads']+=1;COUNTS['native_compartment_reads']+=16;TIMES['read_CPU']+=time.process_time()-t;return np.array(out)
def querycode(m):
 f=m.fe.clone();f.advance(m.time+F.DELTA);return m.encode(f.read())[0]
def context(model,row,site):
 q=model.clone();q._prompt(bytes.fromhex(row['prompt']))
 if site:q._event(bytes.fromhex(row['answer'])[0])
 COUNTS['context_episodes']+=1;return q

def synstate(m):return np.concatenate([np.concatenate((s.fly.m.fast.ravel(),s.fly.m.slow.ravel())) for s in m.bank])
def digest(m):
 z=hashlib.sha256()
 for s in m.bank:
  for v in (s.fly.m.fast,s.fly.m.slow,s.fly.m.adapt):z.update(v.tobytes())
  z.update(repr((s.brain_t,s.teach_seen,s.fly.m.elapsed)).encode())
 for v in (m.theta,m.freq):z.update(v.tobytes())
 if m.fe is not None:z.update(pickle.dumps(m.fe,protocol=5))
 z.update(repr((m.time,m.n,m.records)).encode());return z.hexdigest()
def nonvalue(m):
 return hashlib.sha256(pickle.dumps(([s.fly.m.adapt for s in m.bank],[s.fly.m.elapsed for s in m.bank],m.time,m.n,m.fe,m.theta,m.freq),protocol=5)).hexdigest()
def register_assets(m):
 for s in m.bank:
  for k in ('T','Q','F','MM','k0','B'):
   v=getattr(s.fly.m,k);b=pickle.dumps(v,protocol=5);key=hashlib.sha256(b).hexdigest();ASSETS.setdefault(key,v);ASSET_IDS[id(v)]=key
 class P(pickle.Pickler):
  def persistent_id(self,obj):return ('fixed',ASSET_IDS[id(obj)]) if id(obj) in ASSET_IDS else None
 return P
def checkpoint(m,name):
 P=register_assets(m);p=H/'local'/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('wb') as f:P(f,protocol=5).dump(m)
 ap=H/'local/FIXED_ASSETS.pkl'
 if not ap.exists():
  with ap.open('wb') as f:pickle.dump(ASSETS,f,protocol=5)
 COUNTS['checkpoint_bytes']+=p.stat().st_size;return dict(path=str(p.relative_to(H)),sha256=sha(p),bytes=p.stat().st_size,state_hash=digest(m))
def restore(receipt):
 global ASSETS
 if not ASSETS:
  with (H/'local/FIXED_ASSETS.pkl').open('rb') as f:ASSETS=pickle.load(f)
 class U(pickle.Unpickler):
  def persistent_load(self,pid):assert pid[0]=='fixed';return ASSETS[pid[1]]
 p=H/receipt['path'];assert sha(p)==receipt['sha256']
 with p.open('rb') as f:m=U(f).load()
 register_assets(m);assert digest(m)==receipt['state_hash'];return m

def pair(m,row):
 COUNTS['own_TF_pairs']+=1;own=m.clone();own._prompt(bytes.fromhex(row['prompt']));out=[];ended=False
 for k in range(4):
  p,_=prob(own.values());y=int(p.argmax())
  if y==EOS:ended=True;break
  out.append(y)
  if k<3:own._event(y)
 tf=m.clone();tf._prompt(bytes.fromhex(row['prompt']));sites=[]
 for y in [*bytes.fromhex(row['answer']),EOS]:
  p,lp=prob(tf.values());r=lp.copy();r[y]=-np.inf;sites.append(dict(y=y,argmax=int(p.argmax()),lp=float(lp[y]),margin=float(lp[y]-r.max())))
  if y!=EOS:tf._event(y)
 return dict(id=row['id'],cohort=row['cohort'],output_hex=bytes(out).hex(),EOS=ended,CAP=not ended,correct=ended and bytes(out).hex()==row['answer'],tf=sites)
def panel(m,rows,tag):
 before=digest(m);t=time.process_time();r=[pair(m,row) for row in rows];assert digest(m)==before;TIMES['panels_CPU']+=time.process_time()-t
 path=f'local/panels/{tag}.json';save(path,r);budget();return dict(path=path,sha256=sha(H/path),correct=sum(x['correct'] for x in r),OLD=sum(x['correct'] for x in r if x['cohort']=='OLD'),NEW=sum(x['correct'] for x in r if x['cohort']=='NEW'),correct_ids=[x['id'] for x in r if x['correct']],empty_EOS=sum(x['EOS'] and not x['output_hex'] for x in r),CAP=sum(x['CAP'] for x in r))

def pulse(pre,y,form,gain,force_warm=1.,keep=False):
 """Shadow target: no observed target feed, held post-event query condition."""
 COUNTS['shadow_targets']+=1;budget();q=pre.clone();v=q.values();p,_=prob(v);coef=force_warm*gain*direction(v,p,y,form)
 q.teach_once(y,coef,gain!=0);x=querycode(q);when=q.time+F.DELTA;post=fixedread(q,x,when)
 return dict(model=q if keep else None,v=post,coef=coef,x=x,time=when,nonvalue=nonvalue(q))
def displacement(v,z):
 dv=v-z;zz=C@(dv-dv.mean())/4;zz-=zz.mean();return float(np.sqrt(np.mean(zz*zz)))
def solve(response,target,upper,floor):
 evaluations=[]
 def evaluate(a):
  if len(evaluations)>=20:raise StopProtocol('GAIN_EVALUATION_CAP')
  d,extra=response(a);evaluations.append(dict(gain=a,d=d));return d,extra
 dlo,_=evaluate(0.);dhi,hiout=evaluate(upper);mid=upper/2;dm,mo=evaluate(mid)
 if not (dlo-floor<=dm<=dhi+floor):raise StopProtocol('NONMONOTONIC_CALIBRATION')
 if target>dhi*1.01+floor or target<=floor:raise StopProtocol('UNATTAINABLE_OR_UNRESOLVED_DOSE')
 if abs(dhi-target)<=max(.01*target,floor):return upper,hiout,evaluations
 low,high=0.,upper
 for iteration in range(17):
  d,out=(dm,mo) if iteration==0 else evaluate(mid)
  if not dlo-floor<=d<=dhi+floor:raise StopProtocol('NONMONOTONIC_CALIBRATION')
  if abs(d-target)<=max(.01*target,floor):return mid,out,evaluations
  if d<target:low,dlo=mid,d
  else:high,dhi=mid,d
  mid=(low+high)/2
 raise StopProtocol('DOSE_MATCH_FAILED_20')
def lock():
 old=json.loads((ROOT/'minifly_v171_dev_20261026/CALIBRATION_SOURCE_LOCK.json').read_text())
 assert all(sha(ROOT/f)==h for f,h in old.items()),'historical source mismatch'
 files=dict(old)
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f and Path(f).resolve().is_relative_to(ROOT) and Path(f).suffix=='.py':files[str(Path(f).resolve().relative_to(ROOT))]=sha(f)
 for p in [*H.glob('*.py'),H/'EXECUTION_SPEC.json',H/'AUTHORIZATION.json',H/'MANIFEST.json']:
  files[str(p.relative_to(ROOT))]=sha(p)
 return files
