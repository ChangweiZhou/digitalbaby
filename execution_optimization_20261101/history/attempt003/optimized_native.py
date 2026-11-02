"""Exact native execution specialization. No learning-rule or precision changes.

Only canonical rep0/feedback0 flies are supported. Passive active-code events
skip calculations whose outputs cannot affect state when plastic=False. Derived
encoding caches are bounded, label-free and absent from learned-state digests.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,copy,math
from pathlib import Path
import numpy as np
from numba import njit
H=Path(__file__).resolve().parent;ROOT=H.parent
sys.path.insert(0,str(ROOT/'minifly_v176_feature_deployment_dev_20261101'))
import native_stage as N
N.W.H=H;N.W.TIER='engineering'
F=N.F
import model_evo as E

@njit(cache=True,fastmath=False)
def passive_active(x,dt,ta,tg,tf,ts,fast,slow,adapt):
 # Same scalar expressions/order as advance73 with rawfast/rawslow identically0.
 for i in range(len(x)):
  end=adapt[i]*np.exp(-.05*dt*x[i]);end=1-(1-end)*np.exp(-dt/ta);adapt[i]=end
  fast[i,0]=(fast[i,0]+0.)*np.exp(-dt/tg)
  fast[i,1]=(fast[i,1]+0.)*np.exp(-dt/tf)
 for i in range(len(slow)):slow[i]=(slow[i]+0.)*np.exp(-dt/ts)

@njit(cache=True,fastmath=False)
def passive_banks(x,dt,ta,tg,tf,ts,fast,slow,adapt):
 for j in range(len(fast)):passive_active(x,dt,ta,tg,tf,ts,fast[j],slow[j],adapt[j])

def clone_fly(fly):
 out=copy.copy(fly);out.genes=fly.genes.copy();out.m=copy.copy(fly.m)
 for k in ('fast','slow','adapt'):setattr(out.m,k,getattr(fly.m,k).copy())
 out.m.kernel=dict(fly.m.kernel);out.m.settings=dict(fly.m.settings);out.m.diagnostics=[]
 out.phenotype=dict(fly.phenotype);out.stats=E.empty_stats();return out

class PassiveEvo(E.EvoLearner):
 def clone(self):return clone_fly(self)
 def event(self,dt,x=None,punishment=0.,plastic=True):
  dt=float(dt)
  if not math.isfinite(dt) or dt<0 or not math.isfinite(punishment):raise ValueError('Invalid event')
  if x is None or not np.any(x):return E.raw_event_scaled(self.m,dt,scale=self.alpha_scale)
  if plastic:return super().event(dt,x,punishment,True)
  x=np.asarray(x,dtype=np.float64)
  if x.shape!=self.m.adapt.shape or not np.isfinite(x).all() or np.any(x<0):raise ValueError('Invalid activity')
  m=self.m;a=m.adapt[x>0];summary=dict(adaptation_mean=float(a.mean()),adaptation_min=float(a.min()),adaptation_max=float(a.max()))
  f0=m.kernel['fraction'];g=self.genes
  # With rawalpha=0, sign-dependent allocation terms are exactly zero.
  fraction=1./(1.+np.exp(-(np.log(f0/(1.-f0))+g[0]+g[1]*(2.*m.adapt-1.))))
  passive_active(x,dt,m.ta,m.tg,m.kernel['fast_tau'],m.kernel['slow_tau'],m.fast,m.slow,m.adapt)
  m.elapsed=np.float64(m.elapsed+dt);m.event_count+=np.uint64(1);m.presentation_count+=np.uint64(1)
  stats=self.stats;stats['active_events']+=1;stats['no_write_events']+=1
  z=np.zeros(len(x))
  return dict(dt=dt,active=True,rawslow=z.copy(),rawfast=z.copy(),rawalpha=z,gamma_L1=0.,allocation_min=float(fraction.min()),allocation_max=float(fraction.max()),split_error=0.,state_patch_error=0.,**summary)

def install(m):
 for bank in m.bank:
  fly=bank.fly
  if fly.m.rep!=0 or fly.m.feedback!=0:raise ValueError('Outside canonical rep0/feedback0 specialization')
  fly.__class__=PassiveEvo
 return m

def topk_exact(raw,side,side_n):
 x=np.zeros(len(raw))
 for s in (0,1):
  ix=np.flatnonzero((side==s)&(raw>0));n=min(len(ix),side_n[s])
  if n==len(ix):x[ix]=1.;continue
  if not n:continue
  cut=np.partition(raw[ix],len(ix)-n)[len(ix)-n];above=ix[raw[ix]>cut];equal=ix[raw[ix]==cut]
  x[above]=1.;x[equal[:n-len(above)]]=1.
 return x

class Model(N.Model):
 """Native state/update route, with exact TopK and bounded immutable-code cache."""
 def __init__(self,asset=None,cache_size=64):
  super().__init__(asset);install(self);self._bt=self.bank[0].fly.m.B.T;self.cache_size=cache_size;self._encoding_cache={};self.cache_hits=0;self.cache_misses=0
  self._packed={k:np.stack([getattr(s.fly.m,k) for s in self.bank]) for k in ('fast','slow','adapt')};self._bind()
  ref=self.bank[0].fly.m
  for s in self.bank:
   m=s.fly.m
   if (m.ta,m.tg,m.kernel)!=(ref.ta,ref.tg,ref.kernel):raise ValueError('Compartment dynamics differ')
 def _bind(self):
  for j,s in enumerate(self.bank):
   for k,v in self._packed.items():setattr(s.fly.m,k,v[j])
 def __setstate__(self,state):
  self.__dict__.update(state);self._bind()
 def rawencode(self,p):
  native=self.bank[0].fly.m;raw=np.asarray(self._bt@np.asarray(p,dtype=np.float64)[native.pn_type_index]).ravel()-self.theta
  return topk_exact(raw,native.kc_side,self.side_n),raw
 def encode(self,p):
  # All state that influences encoding must match the exact cache key.
  key=(np.asarray(p,dtype=np.float64).tobytes(),self.theta.tobytes())
  if self.cache_size and key in self._encoding_cache:self.cache_hits+=1;return self._encoding_cache[key]
  self.cache_misses+=1;x,u=self.rawencode(p)
  if self.asset is not None:
   wh,b=self.asset;h=np.maximum(0,wh[np.flatnonzero(x)].sum(0)+b);x,u=self.rawencode(F.BC._norm(self.bridge@h))
  if self.cache_size:
   if len(self._encoding_cache)>=self.cache_size:self._encoding_cache.pop(next(iter(self._encoding_cache)))
   x.flags.writeable=False;u.flags.writeable=False;self._encoding_cache[key]=(x,u)
  return x,u
 def clone(self):
  t=N.time.process_time();q=copy.copy(self);q._packed={k:v.copy() for k,v in self._packed.items()};q.bank=[]
  for s in self.bank:
   b=copy.copy(s);b.fly=copy.copy(s.fly);b.fly.genes=s.fly.genes.copy();b.fly.m=copy.copy(s.fly.m)
   b.fly.m.kernel=dict(s.fly.m.kernel);b.fly.m.settings=dict(s.fly.m.settings);b.fly.m.diagnostics=[];b.fly.phenotype=dict(s.fly.phenotype);b.fly.stats=E.empty_stats()
   b.pending_x=None if s.pending_x is None else s.pending_x.copy();q.bank.append(b)
  q._bind();q.fe=None if self.fe is None else self.fe.clone();q.theta=self.theta.copy();q.freq=self.freq.copy();q._encoding_cache=dict(self._encoding_cache)
  N.W.COUNTS['memory_clones']+=1;N.W.COUNTS['mutable_clone_bytes']+=sum(v.nbytes for v in q._packed.values())+q.theta.nbytes+q.freq.nbytes;N.W.TIMES['clone_CPU']+=N.time.process_time()-t
  return q
 def _event(self,token=None,target=None):
  if target is not None:return super()._event(token,target)
  dt=self._tick();x=self._code()
  if np.any(x):
   m=self.bank[0].fly.m;passive_banks(x,dt,m.ta,m.tg,m.kernel['fast_tau'],m.kernel['slow_tau'],self._packed['fast'],self._packed['slow'],self._packed['adapt'])
   for s in self.bank:
    state=s.fly.m;state.elapsed=np.float64(state.elapsed+dt);state.event_count+=np.uint64(1);state.presentation_count+=np.uint64(1)
    s.fly.stats['active_events']+=1;s.fly.stats['no_write_events']+=1;s.brain_t=self.time
  else:
   for s in self.bank:s.fly.event(dt,x,0.,False);s.brain_t=self.time
  if token is not None and token!=N.W.EOS:self.fe.feed(token,self.time)
