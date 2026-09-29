"""Fixed-capacity FlyWire/effective bout learner. See PROTOCOL.md and NOTICE.md.
Huang-derived dynamics: GPL-3.0-or-later. This is not a spiking or synapse-physical model.
"""
from pathlib import Path
import sys,json,copy,hashlib
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'vendor'));sys.path.insert(0,str(BASE/'iteration4/src'))
import numpy as np
from numba import njit
from scipy.sparse import load_npz
from causal_memory import parameter_set
from prepare_interface import encode
NAMES=['D_gamma_left','D_gamma_right','D_alpha3_left','D_alpha3_right','M_gamma_left','M_gamma_right','M_alpha3_left','M_alpha3_right']
MODES=['full','no_learning','teaching_block','feedback_block']

@njit(cache=True)
def advance_step(x,dt,pun,Q,T,F,MM,k0,fw0,fwd,ta, tg,tf,ts,fraction,fast,slow,adapt,plasticity,teaching):
 N=len(x);dx=np.empty(N);agg=np.zeros(4);inp=np.zeros(4)
 for i in range(N):
  end=adapt[i]*np.exp(-.05*dt*x[i]);end=1-(1-end)*np.exp(-dt/ta);dx[i]=.5*(adapt[i]+end)*x[i];adapt[i]=end
  for j in range(4):
   agg[j]+=Q[i,j]*dx[i]
   w=fast[i,0] if j<2 else fast[i,1]+slow[i]
   inp[j]+=Q[i,j]*dx[i]*(k0[3 if j<2 else 5]+w)
 rm=np.zeros(4)
 for j in range(2):rm[j]=min(max(inp[j]+35.2,0),71.66)-35.2
 for j in range(2):
  for s in range(2):inp[j+2]+=MM[s,j]*rm[s]
  rm[j+2]=min(max(inp[j+2]+11.2,0),31.16)-11.2
 d0=np.zeros(4);payload=np.zeros(4);rd=np.zeros(4)
 for j in range(4):
  d0[j]=k0[0 if j<2 else 2]*agg[j]
  for s in range(4):d0[j]+=F[s,j]*rm[s]
  shock=27.85 if j<2 else 11.38;rd[j]=d0[j]+shock*pun
  payload[j]=(fw0*d0[j]+fwd*shock*pun) if teaching else 0.
 eg=np.exp(-dt/tg);ef=np.exp(-dt/tf);es=np.exp(-dt/ts)
 for i in range(N):
  if plasticity:
   dg=(T[i,0]*payload[0]+T[i,1]*payload[1])*dx[i]*dt/90.
   da=(T[i,2]*payload[2]+T[i,3]*payload[3])*dx[i]*dt/90.
   fast[i,0]+=dg;fast[i,1]+=(1-fraction)*da;slow[i]+=fraction*da
  fast[i,0]*=eg;fast[i,1]*=ef;slow[i]*=es
 return np.concatenate((rd,rm)),payload

class HybridLearner:
 def __init__(self,kernel='bi',mode='full',valence='attractive',reset_baseline=False,shock_scale=1.):
  if kernel not in ['bi','single'] or mode not in MODES:raise ValueError('Unsupported frozen kernel/control')
  data=np.load(ROOT/'data/interface.npz');self.Q=data['output_weights'];self.T=data['teacher_routes'];self.B=load_npz(ROOT/'data/pn_to_kc.npz');self.pn_type_index=data['pn_type_index'];self.kc_side=data['kc_side'];self.input_channels=int(self.pn_type_index.max())+1
  self.kernel_name=kernel;self.mode=mode;self.valence=valence;self.reset_baseline=reset_baseline;self.shock_scale=shock_scale
  self.interface_sha=hashlib.sha256((ROOT/'data/interface.npz').read_bytes()).hexdigest()
  p=parameter_set(valence,reset_baseline,shock_scale);self.k0=p[0][0,0].copy();self.fw0=float(p[1][0,0,0]);self.fwd=float(p[2][0,0,0]);self.ta=float(p[5][0,0,0]);self.tg=float(p[4][0,0,0]);self.F=data['feedback_routes'].copy()
  for s in range(4):
   for j in range(4):self.F[s,j]*=p[3][0,3 if s<2 else 5,0 if j<2 else 2]
  if mode=='feedback_block':self.F[:2,:]=0
  self.MM=data['mbon_routes']*p[3][0,3,5]
  self.kernel=json.loads((BASE/'iteration4/results/fitted_kernels.json').read_text())['fits'][kernel]['kernel']
  N=len(self.kc_side);self.fast=np.zeros((N,2));self.slow=np.zeros(N);self.adapt=np.ones(N);self.elapsed=np.float64(0);self.event_count=np.uint64(0);self.presentation_count=np.uint64(0)
 def encode(self,p):
  p=np.asarray(p,float)
  if p.shape!=(self.input_channels,) or np.any(~np.isfinite(p)) or np.any(p<0):raise ValueError('Expected fixed-length finite nonnegative PN-channel activity')
  return encode(self.B,self.pn_type_index,self.kc_side,p)
 def step(self,seconds,pn_activity=None,punishment=0.):
  if not np.isfinite(seconds) or seconds<0 or not np.isfinite(punishment):raise ValueError('Finite nonnegative bout seconds and finite punishment required')
  p=np.zeros(self.input_channels) if pn_activity is None else np.asarray(pn_activity,float);x=self.encode(p);k=self.kernel
  r,payload=advance_step(x,float(seconds),float(punishment),self.Q,self.T,self.F,self.MM,self.k0,self.fw0,self.fwd,self.ta,self.tg,k['fast_tau'],k['slow_tau'],k['fraction'],self.fast,self.slow,self.adapt,self.mode!='no_learning',self.mode!='teaching_block')
  self.elapsed=np.float64(self.elapsed+seconds);self.event_count+=np.uint64(1);self.presentation_count+=np.uint64(np.any(p>0))
  return dict(rates=r,teaching_payload=payload,active_KCs=int(x.sum()))
 def clone(self,reset_plastic=False):
  m=copy.copy(self)
  for key in ['fast','slow','adapt']:setattr(m,key,getattr(self,key).copy())
  if reset_plastic:m.fast[:]=0;m.slow[:]=0
  return m
 def observe(self,pn_activity,seconds=5.):return self.clone().step(seconds,pn_activity)['rates']
 def memory_contribution(self,pn_activity):return self.observe(pn_activity)-self.clone(True).observe(pn_activity)
 def mutable_bytes(self):return int(self.fast.nbytes+self.slow.nbytes+self.adapt.nbytes+self.elapsed.nbytes+self.event_count.nbytes+self.presentation_count.nbytes)
 def save(self,path):
  path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
  np.savez(path,fast=self.fast,slow=self.slow,adapt=self.adapt,elapsed=self.elapsed,event_count=self.event_count,presentation_count=self.presentation_count,interface_sha=self.interface_sha,kernel=self.kernel_name,mode=self.mode,valence=self.valence,reset_baseline=self.reset_baseline,shock_scale=self.shock_scale)
 @classmethod
 def restore(cls,path):
  s=np.load(path);m=cls(str(s['kernel']),str(s['mode']),str(s['valence']),bool(s['reset_baseline']),float(s['shock_scale']))
  if str(s['interface_sha'])!=m.interface_sha:raise ValueError('Checkpoint and anatomical interface hashes differ')
  for key in ['fast','slow','adapt']:
   if s[key].shape!=getattr(m,key).shape:raise ValueError('Checkpoint shape mismatch')
   setattr(m,key,s[key].copy())
  m.elapsed=np.float64(s['elapsed']);m.event_count=np.uint64(s['event_count']);m.presentation_count=np.uint64(s['presentation_count']);return m
