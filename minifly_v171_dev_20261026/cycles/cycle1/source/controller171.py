"""Raw-byte recurrent controller and literal V169 local feedback circuit.

No arithmetic task, query/key memory, autograd or exact derivative appears in
candidate credit(). Exact transposes are an explicitly selected instrument.
"""
from collections import deque
import copy
import numpy as np
import support171 as s
PARAMS=('U','W','R','b','G','c');CAPS=dict(U=1.,W=1/8,R=1/32,b=1/4,G=1/8,c=1/4)
def warm(n):return 1/16+15/16*min(n/2048,1.)
def center(x):return x-x.mean()
def probs(v,C):
 z=C@center(v)/4;z=z-z.max();lp=z-np.log(np.exp(z).sum());s.COUNTS['distributions']+=1;return np.exp(lp),lp
class Controller:
 def __init__(self,world,role,C):
  assert role in ('ON3','ON1','MIS3','OFF3','CAP-OFF3','EXACT3','EXACT_RECORD')
  self.world=world;self.role=role;self.C=C;self.H=69 if role=='CAP-OFF3' else 64;h=self.H
  r=s.rng(world,'recurrent_edges');self.ids=np.array([[i]+r.choice(np.delete(np.arange(h),i),3,replace=False).tolist() for i in range(h)])
  r=s.rng(world,'gate_edges');self.gids=np.array([r.choice(h,8,replace=False) for _ in range(16)])
  self.U=s.rng(world,'U').uniform(-1/16,1/16,(h,256));self.W=s.rng(world,'W').uniform(-1/64,1/64,(h,4));self.R=s.rng(world,'R').uniform(-1/128,1/128,(h,16));self.b=np.zeros(h);self.G=s.rng(world,'G').uniform(-1/32,1/32,(16,8));self.c=np.zeros(16)
  self.off=role in ('OFF3','CAP-OFF3')
  if self.off:self.R.fill(0)
  dw=np.bincount(self.ids.ravel(),minlength=h);dg=np.bincount(self.gids.ravel(),minlength=h)
  self.FW=s.rng(world,'F_W').choice([-1.,1.],self.W.shape)/np.sqrt(dw[self.ids]);self.FR=s.rng(world,'F_R').choice([-1.,1.],self.R.shape)/np.sqrt(h);self.FG=s.rng(world,'F_G').choice([-1.,1.],self.G.shape)/np.sqrt(dg[self.gids])
  self.reset();s.COUNTS['controller_births']+=1
 def reset(self):
  self.h=np.zeros(self.H);self.r=np.zeros(16);self.frames=deque(maxlen=None if self.role=='EXACT_RECORD' else 3);self.grads={k:np.zeros_like(getattr(self,k)) for k in PARAMS};self.touched=set();self.latch=None;self.steps=0
 def state_hash(self):return s.digest({k:getattr(self,k) for k in PARAMS})
 def clone(self):return copy.deepcopy(self)
 def gate(self,h,v):
  a=np.sum(self.G*h[self.gids],1)+self.c;g=1+.5*np.tanh(a);u=np.tanh(center(v)/4);return a,g,u,g*u
 def step(self,byte,v):
  assert 0<=int(byte)<256
  hp=self.h.copy();rp=self.r.copy();z=np.tanh(self.U[:,byte]+np.sum(self.W*hp[self.ids],1)+(0 if self.off else self.R@rp)+self.b);self.h=.75*hp+.25*z
  a,g,u,self.r=self.gate(self.h,v);f=dict(byte=int(byte),hp=hp,rp=rp,z=z,h=self.h.copy(),v=np.asarray(v).copy(),a=a,g=g,u=u,index=self.steps);self.steps+=1;self.frames.append(f);self.latch=f
  s.COUNTS['controller_byte_steps']+=1;s.COUNTS['controller_forward_edges']+=self.H*4+16*8+(0 if self.off else self.H*16)
 def refresh(self,v):
  a,g,u,self.r=self.gate(self.h,v)
  if self.frames:f=self.frames[-1];f.update(v=np.asarray(v).copy(),a=a,g=g,u=u);self.latch=f
  else:self.latch=dict(v=np.asarray(v).copy(),a=a,g=g,u=u,h=self.h.copy())
 def read(self):return probs(self.latch['g']*self.latch['v'],self.C)
 def scatter(self,kind,msg,tied=False):
  if kind=='G':ids=self.gids;weights=self.G if tied else self.FG
  elif kind=='W':ids=self.ids;weights=self.W if tied else self.FW
  elif kind=='R':return (self.R if tied else self.FR).T@msg
  return np.bincount(ids.ravel(),weights=(msg[:,None]*weights).ravel(),minlength=self.H)
 def credit(self,y,n,mode=None):
  """One target's local signal. Nonlocal EXACT modes are reference-only."""
  mode=mode or self.role;tied=mode in ('EXACT3','EXACT_RECORD');K=1 if mode=='ON1' else (len(self.frames) if mode=='EXACT_RECORD' else 3)
  p,lp=self.read();e=p.copy();e[y]-=1;dv=center(self.C.T@e/4);f=self.latch;da=.5*(1-np.tanh(f['a'])**2)*f['v']*dv;w=warm(n);self.grads['G']+=w*da[:,None]*self.h[self.gids];self.grads['c']+=w*da
  m=self.scatter('G',da,tied);frames=list(self.frames);last=len(frames)-1;signal=np.zeros(3);ages=[]
  for age in range(min(K,len(frames))):
   j=last-age;fj=frames[j];xi=.25*(1-fj['z']**2)*m
   source=frames[last-(3-age)] if mode=='MIS3' and len(frames)>=3 and age in (1,2) else fj
   self.grads['U'][:,source['byte']]+=w*xi;self.touched.add(source['byte']);self.grads['W']+=w*xi[:,None]*source['hp'][self.ids];self.grads['b']+=w*xi
   if not self.off:self.grads['R']+=w*np.outer(xi,source['rp'])
   signal[age]=float(np.linalg.norm(xi));ages.append(fj['index']);s.COUNTS['local_credit_rounds' if not tied else 'reference_credit_rounds']+=1
   if age+1<min(K,len(frames)):
    prev=frames[j-1]
    ea=np.zeros(16) if self.off else .5*(1-np.tanh(prev['a'])**2)*prev['u']*self.scatter('R',xi,tied)
    # MIS3 breaks only delayed update/input pairing; propagated modulators retain true ages.
    gsource=frames[last-2] if mode=='MIS3' and len(frames)>=3 and age==0 else (frames[last-1] if mode=='MIS3' and len(frames)>=3 and age==1 else prev)
    self.grads['G']+=w*ea[:,None]*gsource['h'][self.gids];self.grads['c']+=w*ea
    m=.75*m+self.scatter('W',xi,tied)+self.scatter('G',ea,tied)
  masked=p.copy();masked[y]=-np.inf;rival=int(masked.argmax());margin=float(lp[y]-lp[rival])
  return dict(y=int(y),argmax=int(p.argmax()),lp=float(lp[y]),rival=rival,margin=margin,warmup=w,ages=ages),dict(p=p,h=self.h,z=f.get('z',np.zeros(self.H)),a=f['a'],u=f['u'],g=f['g'],v=f['v'],direct_error=da,signal_norms=signal)
 def apply(self,write=True):
  clips=0;items=0;project=0
  if write:
   for k in PARAMS:
    if self.off and k=='R':continue
    a=getattr(self,k);grad=self.grads[k]
    if k=='U':
     ix=sorted(self.touched);old=a[:,ix].copy();gg=grad[:,ix];v=old-2**-10*np.clip(gg,-1,1);a[:,ix]=np.clip(v,-CAPS[k],CAPS[k]);project+=int(np.count_nonzero(v!=a[:,ix]))
    else:
     gg=grad;v=a-2**-10*np.clip(gg,-1,1);a[:]=np.clip(v,-CAPS[k],CAPS[k]);project+=int(np.count_nonzero(v!=a))
    clips+=int(np.count_nonzero(abs(gg)>1));items+=gg.size
  out=dict(clipped=clips,update_values=items,projected=project,hash=self.state_hash(),write=bool(write));s.COUNTS['controller_update_values']+=items;s.COUNTS['controller_clipped_values']+=clips;s.COUNTS['controller_projected_values']+=project;return out
 def vector(self):return np.concatenate([getattr(self,k).ravel() for k in PARAMS])
 def gradvector(self):return np.concatenate([self.grads[k].ravel() for k in PARAMS])
 def setvector(self,v):
  pos=0
  for k in PARAMS:
   a=getattr(self,k);a[:]=v[pos:pos+a.size].reshape(a.shape);pos+=a.size
 def dense(self):
  W=np.zeros((self.H,self.H));G=np.zeros((16,self.H));W[np.arange(self.H)[:,None],self.ids]=self.W;G[np.arange(16)[:,None],self.gids]=self.G;return W,G
