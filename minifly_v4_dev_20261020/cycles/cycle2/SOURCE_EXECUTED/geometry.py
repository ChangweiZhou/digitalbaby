"""Generic bit/segment front end. No task generator, numeric parser or answer values."""
from v4_common import np
import math,copy,hashlib,json
BITS=((np.arange(256)[:,None]>>np.arange(8))&1).astype(float)

class SegmentState:
 def __init__(self,slots=3,limit=16):
  self.slots,self.limit=slots,limit;self.closed=np.zeros((slots,limit,9));self.open=[];self.steps=0;self.last=np.zeros(9);self.trace=np.zeros((4,9));self.overflow=0;self.closures=0
 def feed(self,b,boundary):
  raw=np.r_[BITS[b],1.]
  # Boundary byte is a separator; it closes the preceding generic segment.
  if boundary:
   self.closed[1:]=self.closed[:-1];self.closed[0].fill(0)
   for i,v in enumerate(reversed(self.open[-self.limit:])):self.closed[0,i]=v
   self.open=[];self.steps=0;self.closures+=1
  else:
   self.open.append(raw)
   if len(self.open)>self.limit:self.open.pop(0);self.overflow+=1
   self.steps+=1
  self.last=raw
  self.trace=self.trace*np.array([.2,.5,.8,.95])[:,None]+raw
 def feature(self,segmented=True):
  # Four finite time-cell bands cover generic positions, not operand columns.
  phase=np.zeros(17);phase[min(self.steps,16)]=1
  closed=self.closed.ravel() if segmented else np.zeros(self.closed.size)
  return np.r_[closed,phase,self.last,self.trace.ravel()/np.array([1.25,2,5,20]) .repeat(9)]

class Geometry:
 def __init__(self,n,seed,topographic=True,segmented=True,fan=12):
  rng=np.random.default_rng(seed);self.n=n;self.segmented=segmented;self.topographic=topographic
  d=3*16*9+17+9+36;self.dimension=d
  self.index=rng.integers(0,d,(n,fan),dtype=np.int32)
  if topographic:
   center=rng.integers(0,16,n);width=rng.choice([1,2,4,8],n)
   for j in range(fan-4):
    pos=np.minimum(15,center+rng.integers(0,8,n)%width)
    self.index[:,j]=rng.integers(0,3,n)*16*9+pos*9+rng.integers(0,9,n)
   # Time/last/trace channels are uniformly accessible without aligning output k.
   self.index[:,-4:]=rng.integers(3*16*9,d,(n,4))
  self.weights=rng.choice([-1.,1.],(n,fan))/math.sqrt(fan)
  self.theta=np.zeros(n);self.frequency=np.zeros(n);self.active=max(1,int(n*.02))
 def encode(self,state):
  f=state.feature(self.segmented);raw=(f[self.index]*self.weights).sum(1)-self.theta
  # Stable tie-breaking in sparse top-k; zero/negative drives are ineligible.
  eligible=np.flatnonzero(raw>0);k=min(self.active,len(eligible));x=np.zeros(self.n)
  if k:
   order=np.lexsort((eligible,-raw[eligible]));x[eligible[order[:k]]]=1
  return x,raw
 def update(self,x,raw,a):
  self.frequency+=.01*(x-self.frequency)
  self.theta+=.005*a*np.maximum(0,self.frequency-.1)*max(float(raw.std()),1e-9)

class Boundary:
 def __init__(self,seed):
  self.w=np.zeros(9);self.rng=np.random.default_rng(seed);self.baseline=0.;self.last=[]
 def decide(self,b,explore=False):
  r=np.r_[BITS[b]*2-1,1.];p=1/(1+math.exp(-float(np.clip(self.w@r,-4,4))))
  decision=bool(self.rng.random()<p) if explore else bool(p>.5)
  if explore:self.last.append((float(decision)-p)*r)
  return decision
 def update(self,loss,a):
  advantage=self.baseline-loss
  if self.last:
   e=np.clip(np.sum(self.last,axis=0),-8,8);self.w=np.clip(self.w+.001*a*advantage*e,-4,4)
  self.baseline=.95*self.baseline+.05*loss;self.last=[]
