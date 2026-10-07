"""Causal two-module learning reference derived from Huang equations (GPL-3.0-or-later).
Only forgetting is changed: local updates enter positive exponential components.
This is an effective model with signed net inputs, not physical KC synapse weights.
"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent/'vendor'));sys.path.insert(0,str(ROOT.parent/'paper_refinement/src'))
import numpy as np
from numba import njit
import memory_model as original

@njit(cache=True)
def advance(k0,W,fw0,fwd,adapt_tau,gamma_tau,fast_tau,slow_tau,fraction,fast,slow,adapt,events):
 N=len(adapt);E=len(events);responses=np.zeros((E,6))
 for ei in range(E):
  dt=events[ei,0];shock=events[ei,1];activity=events[ei,2:];dx=np.zeros(N)
  for i in range(N):
   end=adapt[i]*np.exp(-.05*dt*activity[i]);end=1-(1-end)*np.exp(-dt/adapt_tau)
   dx[i]=.5*(adapt[i]+end)*activity[i];adapt[i]=end
  inp=np.zeros(6)
  for i in range(N):
   for j in range(6):inp[j]+=k0[i,j]*dx[i]
   inp[3]+=fast[i,0]*dx[i];inp[5]+=(fast[i,1]+slow[i])*dx[i]
  inp[0]+=27.85*shock;inp[2]+=11.38*shock
  r=np.zeros(6);r[3]=min(max(inp[3]+35.2,0),71.66)-35.2
  r[4]=min(max(inp[4]+9.+W[3,4]*r[3],0),17.9)-9.
  r[5]=min(max(inp[5]+11.2+W[3,5]*r[3],0),31.16)-11.2
  for j in range(3):
   r[j]=inp[j]
   for k in range(3,6):r[j]+=W[k,j]*r[k]
  responses[ei]=r
  for i in range(N):
   for c,j in enumerate((0,2)):
    teaching=0.
    for q in range(N):teaching+=k0[q,j]*dx[q]
    for k in range(3,6):teaching+=W[k,j]*r[k]
    shockrate=27.85 if j==0 else 11.38
    delta=(fw0*teaching+fwd*shockrate*shock)*dx[i]*dt/90.
    if c==0:fast[i,0]+=delta
    else:fast[i,1]+=(1-fraction)*delta;slow[i]+=fraction*delta
   fast[i,0]*=np.exp(-dt/gamma_tau);fast[i,1]*=np.exp(-dt/fast_tau);slow[i]*=np.exp(-dt/slow_tau)
 return responses,fast,slow,adapt

def parameter_set(valence='attractive',reset=False,shock_scale=1.):
 return original.specialize(original.load_parameters(2),valence,reset=reset,shock_scale=shock_scale)

def pack(events,N=2):
 z=np.zeros((len(events),N+2))
 for i,e in enumerate(events):
  z[i,0]=e['duration'];z[i,1]=e['punishment']
  if e['odor']>=0:z[i,2+e['odor']]=1
 return z

class MemoryCore:
 def __init__(self,p,kernel,N=2):
  self.k0=np.tile(p[0][0,0],(N,1));self.W=p[3][0].copy();self.fw0=float(p[1][0,0,0]);self.fwd=float(p[2][0,0,0]);self.adapt_tau=float(p[5][0,0,0]);self.gamma_tau=float(p[4][0,0,0]);self.kernel=dict(kernel)
  self.fast=np.zeros((N,2));self.slow=np.zeros(N);self.adapt=np.ones(N);self.elapsed=0.
 def step(self,duration,activity=None,punishment=0.):
  N=len(self.adapt);x=np.zeros(N) if activity is None else np.asarray(activity,float)
  if not np.isfinite(duration) or duration<0 or x.shape!=(N,) or np.any(~np.isfinite(x)) or np.any(x<0) or not np.isfinite(punishment):raise ValueError('Require finite nonnegative duration/activity and matching cue count')
  return self.run_packed(np.array([[duration,punishment,*x]]))[0]
 def run_packed(self,events):
  events=np.asarray(events,float)
  if events.ndim!=2 or events.shape[1]!=len(self.adapt)+2 or np.any(~np.isfinite(events)) or np.any(events[:,0]<0) or np.any(events[:,2:]<0):raise ValueError('Expected finite event rows [nonnegative seconds, punishment, nonnegative cue activities]')
  k=self.kernel
  r,self.fast,self.slow,self.adapt=advance(self.k0,self.W,self.fw0,self.fwd,self.adapt_tau,self.gamma_tau,k['fast_tau'],k['slow_tau'],k['fraction'],self.fast,self.slow,self.adapt,np.asarray(events,float))
  self.elapsed+=float(np.sum(events[:,0]));return r
 def state(self):
  return dict(k0=self.k0.tolist(),W=self.W.tolist(),fw0=self.fw0,fwd=self.fwd,adapt_tau=self.adapt_tau,gamma_tau=self.gamma_tau,kernel=dict(self.kernel),fast=self.fast.tolist(),slow=self.slow.tolist(),adapt=self.adapt.tolist(),elapsed=self.elapsed,scope='Effective causal cue-addressed state; no physical synapse interpretation')
 @classmethod
 def restore(cls,s):
  m=cls.__new__(cls)
  for k,v in s.items():
   if k!='scope':setattr(m,k,np.array(v) if k in ['k0','W','fast','slow','adapt'] else v)
  return m

def run_protocol(kernel,events,valence='attractive',reset=False,shock_scale=1.,ablate_gamma_to_alpha3=False):
 p=parameter_set(valence,reset,shock_scale)
 if ablate_gamma_to_alpha3:p[3][:,3,2]=0
 core=MemoryCore(p,kernel);r=core.run_packed(pack(events));return r,core

def images(r,events):
 i=[j for j,e in enumerate(events) if e['imaging']];return r[i].reshape(-1,2,6)
