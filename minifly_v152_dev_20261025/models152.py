"""Generic raw-byte learners; no task parsing, relation targets or learned evaluator."""
import sys,copy,time,math
import numpy as np
from support152 import ROOT,rng,COUNTS,TIMES
sys.path.insert(0,str(ROOT/'minifly_cls_dev_20261017'))
import fastmem
C=fastmem.CODE.copy();EOS=256

def warmup(n):return 1/16+15/16*min(n/1024,1)
def categorical(v,q):
 u=(v-v.mean())/4;z=C@u+q;z-=z.max();lp=z-np.log(np.exp(z).sum());return np.exp(lp),lp

def value_error(p,y):
 e=p.copy();e[y]-=1;g=C.T@e/4;g-=g.mean();return g,e

class Recurrent:
 PARAMS=('U','b','theta','R','q')
 def __init__(self,arm,world,eta):
  assert arm in ('R_BP','R_LE','R_LE0');self.arm=arm;self.eta=eta;self.n=0;self.records=0;self.gain=.5
  gen=rng(world,'graph');self.ids=np.array([[i]+gen.choice(np.delete(np.arange(64),i),3,replace=False).tolist() for i in range(64)])
  self.sign=gen.choice([-1.,1.],(64,4));gen=rng(world,'weights')
  self.U=gen.normal(0,1/16,(64,256));self.b=np.zeros(64);self.theta=np.zeros((64,4));self.R=gen.normal(0,1/np.sqrt(128),(16,128));self.q=np.zeros(257)
  self.clip_count=0;self.grad_count=0;self.project_count=0;self.begin()
 def clone(self):
  t=time.process_time();o=copy.deepcopy(self);TIMES['reference_clone_CPU']+=time.process_time()-t;COUNTS['reference_clones']+=1;return o
 def begin(self):self.h=np.zeros(64);self.r=np.zeros(64)
 def edge(self):
  a=np.exp(self.theta-self.theta.max(1,keepdims=True));return self.sign*a/a.sum(1,keepdims=True)
 def step(self,byte):
  assert 0<=int(byte)<256
  w=self.edge();self.h=.75*self.h+.25*np.tanh(self.U[:,int(byte)]+self.b+self.gain*np.sum(w*self.h[self.ids],axis=1));self.r=.95*self.r+.05*self.h
  COUNTS['reference_forward_byte_steps']+=1
 def read(self):
  COUNTS['output_distributions']+=1;return categorical(self.R@np.r_[self.h,self.r],self.q)
 def gradients(self,prompt,answer,method=None):
  method=method or self.arm;assert method in ('R_BP','R_LE','R_LE0')
  if method=='R_BP':return self._bptt(prompt,answer)
  return self._eligibility(prompt,answer,method)
 def _forward_record(self,prompt,answer):
  self.begin();cache=[];sites=[];w=self.edge();n=self.n
  def advance(byte):
   hp=self.h.copy();rp=self.r.copy();inc=np.sum(w*hp[self.ids],axis=1);z=np.tanh(self.U[:,byte]+self.b+self.gain*inc);self.h=.75*hp+.25*z;self.r=.95*rp+.05*self.h
   cache.append((int(byte),hp,z,inc));COUNTS['reference_forward_byte_steps']+=1
  for b in prompt:advance(b)
  for y in list(answer)+[EOS]:
   p,lp=self.read();gv,e=value_error(p,y);sites.append(dict(t=len(cache),y=y,p=p,lp=float(lp[y]),w=warmup(n),gv=gv,e=e,h=self.h.copy(),r=self.r.copy()));n+=1
   if y!=EOS:advance(y)
  return cache,sites,w
 def _bptt(self,prompt,answer):
  """Exact analytic reverse differentiation, diagnostic reference only."""
  cache,sites,w=self._forward_record(prompt,answer);g={k:np.zeros_like(getattr(self,k)) for k in self.PARAMS};lh=np.zeros((len(cache)+1,64));lr=lh.copy()
  for s in sites:
   gv=s['w']*s['gv'];g['R']+=np.outer(gv,np.r_[s['h'],s['r']]);g['q']+=s['w']*s['e'];lh[s['t']]+=self.R[:,:64].T@gv;lr[s['t']]+=self.R[:,64:].T@gv
  bh=np.zeros(64);br=bh.copy()
  for t in range(len(cache),0,-1):
   byte,hp,z,inc=cache[t-1];br=br+lr[t];bh=bh+lh[t]+.05*br;dz=.25*(1-z*z)*bh
   g['U'][:,byte]+=dz;g['b']+=dz;g['theta']+=self.gain*dz[:,None]*(w*hp[self.ids]-abs(w)*inc[:,None])
   prev=.75*bh;np.add.at(prev,self.ids.ravel(),(self.gain*dz[:,None]*w).ravel());bh=prev;br=.95*br
  return g,sites
 def _eligibility(self,prompt,answer,method):
  """Analytic neuron-local history; no BPTT/autograd calls in this path."""
  self.begin();w=self.edge();eh=np.zeros((64,261));er=eh.copy();g={k:np.zeros_like(getattr(self,k)) for k in self.PARAMS};sites=[];n=self.n
  def advance(byte):
   nonlocal eh,er
   hp=self.h.copy();inc=np.sum(w*hp[self.ids],axis=1);z=np.tanh(self.U[:,byte]+self.b+self.gain*inc);d=.25*(1-z*z)
   if method=='R_LE0':eh.fill(0);er.fill(0)
   eh*= (.75+self.gain*d*w[:,0])[:,None]
   eh[:,byte]+=d;eh[:,256]+=d;eh[:,257:]+=self.gain*d[:,None]*(w*hp[self.ids]-abs(w)*inc[:,None])
   er=.95*er+.05*eh;self.h=.75*hp+.25*z;self.r=.95*self.r+.05*self.h;COUNTS['reference_forward_byte_steps']+=1
   COUNTS['eligibility_values_updated']+=33408
  for b in prompt:advance(b)
  for y in list(answer)+[EOS]:
   p,lp=self.read();gv,e=value_error(p,y);s=warmup(n);gh=self.R[:,:64].T@gv;gr=self.R[:,64:].T@gv;gg=s*(gh[:,None]*eh+gr[:,None]*er)
   g['U']+=gg[:,:256];g['b']+=gg[:,256];g['theta']+=gg[:,257:];g['R']+=s*np.outer(gv,np.r_[self.h,self.r]);g['q']+=s*e
   sites.append(dict(y=y,lp=float(lp[y]),w=s,p=p,h=self.h.copy(),r=self.r.copy()));n+=1
   if y!=EOS:advance(y)
  return g,sites
 def apply(self,g,only=None):
  for k in self.PARAMS:
   if only is not None and k not in only:continue
   raw=g[k];self.clip_count+=int((abs(raw)>1).sum());self.grad_count+=raw.size;setattr(self,k,getattr(self,k)-self.eta*np.clip(raw,-1,1))
  before=self.theta.copy();self.theta=np.clip(self.theta,-3,3);self.project_count+=int((before!=self.theta).sum())
 def train(self,prompt,answer,write=True):
  t=time.process_time()
  if write:g,sites=self.gradients(prompt,answer);self.apply(g)
  else:_,sites,_=self._forward_record(prompt,answer)
  self.n+=len(answer)+1;self.records+=1;TIMES['reference_training_CPU']+=time.process_time()-t
  return [{'y':s['y'],'lp':s['lp'],'argmax':int(s['p'].argmax()),'warmup':s['w']} for s in sites]
 def rest(self,steps):self.begin()

class Native(fastmem.FastMemory):
 def __init__(self,arm):
  assert arm in ('N_A','N_2B','N_2B_Q');super().__init__(homeo=False);self.arm=arm;self.q=np.zeros(257);self.n=0;self.value_disabled=False
 def clone(self):
  t=time.process_time();o=super().clone();o.theta=self.theta.copy();o.freq=self.freq.copy();o.q=self.q.copy();COUNTS['native_model_clones']+=1
  COUNTS['native_minimum_clone_bytes']+=sum(s.fly.m.fast.nbytes+s.fly.m.slow.nbytes+s.fly.m.adapt.nbytes for s in self.bank)
  TIMES['native_clone_CPU']+=time.process_time()-t;return o
 def begin(self):self._prompt(b'')
 def step(self,byte):super()._event(int(byte),None)
 def encode(self,p):
  t=time.process_time();v=super().encode(p);TIMES['native_encoding_CPU']+=time.process_time()-t;COUNTS['native_encoding_global_sorts']+=2;return v
 def read(self):
  t=time.process_time();v=np.zeros(16) if self.value_disabled else self.values();COUNTS['output_distributions']+=1;COUNTS['native_vector_readouts']+=not self.value_disabled;TIMES['native_readout_CPU']+=time.process_time()-t;return categorical(v,self.q)
 def train(self,prompt,answer,write=True):
  t=time.process_time();self._prompt(prompt);sites=[]
  for y in list(answer)+[EOS]:
   v=self.values();p,lp=categorical(v,self.q);COUNTS['output_distributions']+=1;COUNTS['native_vector_readouts']+=1;gv,e=value_error(p,y)
   coef=.1*warmup(self.n)*(np.tanh((v-v.mean())/2)-C[y])/2 if self.arm=='N_A' else .2*warmup(self.n)*gv
   assert np.isfinite(coef).all() and np.max(abs(coef))<=1
   if write:
    self._tick();x=self._code()
    for j,s in enumerate(self.bank):
     s.pending_t=self.time;s.pending_x=x;s.teach_signed(0.,float(coef[j]),self.time,write=True);s.brain_t=self.time
    COUNTS['native_signed_API_calls']+=16
    if self.arm=='N_2B_Q':self.q-=.03*warmup(self.n)*e
    if y!=EOS:self.fe.feed(y,self.time)
   else:super()._event(y,None)
   sites.append(dict(y=y,lp=float(lp[y]),argmax=int(p.argmax()),warmup=warmup(self.n)));self.n+=1
  self.records+=1;TIMES['native_training_CPU']+=time.process_time()-t;return sites
 def rest(self,steps):super().rest(steps*fastmem.DELTA)

class Bias:
 def __init__(self):self.q=np.zeros(257);self.n=0;self.records=0;self.arm='Q_ONLY'
 def clone(self):return copy.deepcopy(self)
 def begin(self):pass
 def step(self,byte):assert 0<=int(byte)<256
 def read(self):COUNTS['output_distributions']+=1;return categorical(np.zeros(16),self.q)
 def train(self,prompt,answer,write=True):
  sites=[]
  for y in list(answer)+[EOS]:
   p,lp=self.read();_,e=value_error(p,y);sites.append(dict(y=y,lp=float(lp[y]),argmax=int(p.argmax()),warmup=warmup(self.n)))
   if write:self.q-=.03*warmup(self.n)*e
   self.n+=1
  self.records+=1;return sites
 def rest(self,steps):pass

def make(arm,world,eta):
 return Recurrent(arm,world,eta) if arm.startswith('R_') else Bias() if arm=='Q_ONLY' else Native(arm)

def prepare(m,prompt):
 m.begin()
 for b in prompt:m.step(b)

def generate(m,prompt,cap=4,ablate=None):
 q=m.clone()
 if ablate=='no_native':q.value_disabled=True
 if ablate=='no_q':q.q.fill(0)
 if isinstance(q,Native):q._prompt(prompt)
 else:prepare(q,prompt)
 out=[];ended=False;choices=[]
 for _ in range(cap):
  p,_=q.read();y=int(p.argmax());choices.append(y)
  if y==EOS:ended=True;break
  out.append(y);q.step(y)
 return dict(output_hex=bytes(out).hex(),EOS=ended,choices=choices)

def teacher_forced(m,prompt,answer,ablate=None):
 q=m.clone()
 if ablate=='no_native':q.value_disabled=True
 if ablate=='no_q':q.q.fill(0)
 if isinstance(q,Native):q._prompt(prompt)
 else:prepare(q,prompt)
 sites=[]
 for y in list(answer)+[EOS]:
  p,lp=q.read();sites.append(dict(y=y,lp=float(lp[y]),argmax=int(p.argmax())))
  if y!=EOS:q.step(y)
 return sites
