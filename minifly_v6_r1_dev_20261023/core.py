"""Generic byte learner; no evaluator, parser, answer table or task rules."""
import os,sys,copy,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
sys.path.insert(0,str(R/'minifly_v5_1_dev_20261020'))
import v51_common
import numba
numba.config.CACHE_DIR=str(P/'scratch');os.environ['NUMBA_CACHE_DIR']=str(P/'scratch')
import numpy as np
from engine import Engine,targets,residual,decode,DT
H=88;D=348;SIGMA=.1

def ramp(n):return 1/16+15/16*min(n/1024,1)
class Controller:
 def __init__(self,seed):
  self.rng=np.random.default_rng(seed);self.Wv=self.rng.normal(0,.05/np.sqrt(D),(H,D));self.Wg=self.rng.normal(0,.05/np.sqrt(D),(H,D))
  self.Wv[:,-1]=0;self.Wg[:,-1]=np.log(.1/.9);self.b=0.;self.n=0;self.movement=0.;self.clipped=0;self.updates=0;self.reset()
 def reset(self):
  self.h=np.zeros(H);self.Ev=np.zeros((H,D));self.Eg=np.zeros((H,D));self.Gv=np.zeros((H,D));self.Gg=np.zeros((H,D));self.b_start=self.b;self.rewards=[];self.entropy=0.;self.steps=0
 def step(self,symbol,trace=True):
  if not isinstance(symbol,(int,np.integer)) or not 0<=symbol<259:raise ValueError('invalid observed symbol')
  u=np.zeros(D);u[int(symbol)]=1;u[259:347]=self.h;u[-1]=1
  mu=self.Wv@u;z=self.Wg@u;p=1/(1+np.exp(-np.clip(z,-700,700)));g=(self.rng.random(H)<p).astype(float);v=mu+SIGMA*self.rng.normal(size=H)
  self.h=(1-g)*self.h+g*np.tanh(v)
  if trace:
   self.Ev+=np.outer(g*(v-mu)/SIGMA**2,u);self.Eg+=np.outer(g-p,u)
   self.entropy+=float(np.mean(-p*np.log(p)-(1-p)*np.log1p(-p)));self.steps+=1
  return mu,p,g,v,u
 def credit(self,m,t):
  y=targets(t);eligible=slice(8,9) if t==256 else slice(None)
  loss=float(np.logaddexp(0,-y[eligible]*m[eligible]/2).mean());r=-min(loss,20)/20
  self.Gv+=(r-self.b_start)*self.Ev;self.Gg+=(r-self.b_start)*self.Eg;self.rewards.append(r);return loss
 def finish(self,enabled=True):
  q=max(1,len(self.rewards));gv=self.Gv/q;gg=self.Gg/q;norm=float(np.sqrt(np.sum(gv*gv)+np.sum(gg*gg)));scale=max(1.,norm);eta=.001*ramp(self.n)
  if enabled:
   self.Wv+=eta/scale*gv;self.Wg+=eta/scale*gg;self.b+=(float(np.mean(self.rewards))-self.b)/1024;self.movement+=eta*min(norm,1);self.clipped+=int(norm>1);self.updates+=1
  st=dict(gradient_norm=norm,trace_norm=float(np.sqrt(np.sum(self.Ev**2)+np.sum(self.Eg**2))),gate_entropy=self.entropy/max(1,self.steps),baseline=self.b,eta=eta,mean_loss=float(-20*np.mean(self.rewards)))
  self.n+=1;return st
class Fly:
 def __init__(self,seed):
  self.c=Controller(seed);self.e=Engine(5177,seed,False)
  with np.load(P/'assets/calibration.npz') as z:self.mu=z['mu'];self.sg=np.maximum(z['sigma'],z['floor'])
  self.sides=[np.flatnonzero(self.e.bank.kc_side==s) for s in (0,1)];self.code_stats=dict(frames=0,active_sum=0,shortfalls=0,ties=0);self.initial=(self.c.Wv.copy(),self.c.Wg.copy())
 def code(self,observe=False):
  b=self.e.bank;a=(np.asarray(b.B.T@self.c.h[b.pn_type_index]).ravel()-self.mu)/self.sg;x=np.zeros(5177)
  for s,ix in enumerate(self.sides):
   ids=ix[a[ix]>0];ids=ids[np.lexsort((ids,-a[ids]))];x[ids[:b.side_n[s]]]=1
   if observe:
    self.code_stats['shortfalls']+=int(len(ids)<b.side_n[s]);self.code_stats['ties']+=int(len(ids)>1 and np.any(np.diff(a[ids])==0))
  if observe:self.code_stats['frames']+=1;self.code_stats['active_sum']+=int(x.sum())
  return x
 def begin(self):self.c.reset();self.e.begin();self.c.step(256)
 def observe(self,t,enabled=True):
  x=self.code(True);m=self.e.margins(x)[0];loss=self.c.credit(m,t);self.e.tick(x,residual(m,t,ramp(self.c.n)/16) if enabled else None);self.c.step(258 if t==256 else t);return loss
 def _generate_private(self,tape,cap=16):
  v=self;v.c.rng=np.random.default_rng(tape);out=[];marg=[]
  for j in range(cap+1):
   x=v.code();m=v.e.margins(x)[0];t=decode(m);marg.append(m.tolist());v.e.tick(x,None)
   if t==256:return out,True,False,marg
   if j==cap:return out,False,True,marg
   out.append(t);v.c.step(t,False)
  raise AssertionError
 def generation(self,tape,cap=16):
  v=copy.copy(self);v.e=self.e.clone();v.c=copy.copy(self.c);v.c.h=self.c.h.copy()
  return v._generate_private(tape,cap)
 def query(self,prompt,tape):
  v=copy.copy(self);v.e=self.e.clone();v.c=copy.deepcopy(self.c);v.c.rng=np.random.default_rng(tape);v.begin()
  for t in prompt:
   x=v.code();v.e.tick(x,None);v.c.step(t,False)
  v.c.step(257,False)
  return v._generate_private(tape+1)
 def record(self,prompt,answer,native=True,controller=True,tape=0):
  if len(prompt)>256 or len(answer)>16:raise ValueError('observed payload cap')
  self.begin();loss=[]
  for t in prompt:loss.append(self.observe(t,native))
  self.c.step(257);gen=[self.generation(tape+k) for k in range(3)]
  for t in list(answer)+[256]:loss.append(self.observe(t,native))
  self.e.end(bytes(prompt),write=native);st=self.c.finish(controller);st['actual_loss']=float(np.mean(loss));return gen,st
 def digest(self):
  h=hashlib.sha256()
  for a in (self.e.bank.fast,self.e.bank.slow,self.e.bank.adapt,self.e.q,self.e.q_at,self.e.last_at,self.e.ring,self.e.ring_pos,self.e.signs,self.c.Wv,self.c.Wg,self.c.h,self.c.Ev,self.c.Eg,self.c.Gv,self.c.Gg):h.update(a.tobytes())
  h.update(repr((self.e.time,self.e.events,self.e.write_l1,self.e.slow_write_l1,self.e.confirmations,self.c.b,self.c.n,self.c.movement,self.c.clipped,self.c.updates,self.c.b_start,self.c.rewards,self.c.rng.bit_generator.state,self.code_stats)).encode());return h.hexdigest()
 def checkpoint(self,path):
  a={k:getattr(self.e,k) for k in ('q','q_at','last_at','ring','ring_pos','signs')};a.update(fast=self.e.bank.fast,slow=self.e.bank.slow,adapt=self.e.bank.adapt)
  for k in ('Wv','Wg','h','Ev','Eg','Gv','Gg'):a['c_'+k]=getattr(self.c,k)
  a['metadata']=np.array(json.dumps(dict(engine={k:getattr(self.e,k) for k in ('time','events','write_l1','slow_write_l1','confirmations')},controller={k:getattr(self.c,k) for k in ('b','n','movement','clipped','updates','b_start','rewards','entropy','steps')},rng=self.c.rng.bit_generator.state,code_stats=self.code_stats)))
  np.savez_compressed(path,**a)
 def restore(self,path):
  with np.load(path) as a:
   for k in ('q','q_at','last_at','ring','ring_pos','signs'):setattr(self.e,k,a[k].copy())
   for k in ('fast','slow','adapt'):setattr(self.e.bank,k,a[k].copy())
   for k in ('Wv','Wg','h','Ev','Eg','Gv','Gg'):setattr(self.c,k,a['c_'+k].copy())
   m=json.loads(str(a['metadata']))
  for k,v in m['engine'].items():setattr(self.e,k,v)
  for k,v in m['controller'].items():setattr(self.c,k,v)
  self.c.rng.bit_generator.state=m['rng'];self.code_stats=m['code_stats']
