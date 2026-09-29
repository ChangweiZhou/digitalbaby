"""Huang et al. (2024) bout-level memory model, Python port.

Derived from Luo/Huang/Schnitzer source, GPL-3.0-or-later; see sources/huang_model/README.md.
This is an effective rate/weight model of selected conditioning protocols,
NOT synapse-resolved FlyWire dynamics and NOT newly fitted biology.
"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];MINIFLY=ROOT.parent
sys.path.insert(0,str(MINIFLY/'vendor'))
import numpy as np
from scipy.io import loadmat
from numba import njit
BASE=np.array([35.2,9.,11.2]);CAP=BASE+np.array([36.46,8.9,19.96])
NAMES=['D_gamma1','D_alpha2','D_alpha3','M_gamma1','M_alpha2','M_alpha3']

def load_parameters(modules=3,draws=False):
 d=loadmat(ROOT/f'sources/huang_model/data_and_parameters/Dx_steady_state_nonlinear_3_27-Mar-2023_{modules}modules.mat')
 vectors=d['para_rand'].T if draws else d['para_mu'].reshape(1,-1);out=[];pos=0
 for a in d['mat_lu_cell'].ravel():
  lo,hi=a[...,0],a[...,1];mask=(lo!=hi).ravel(order='F');n=int(mask.sum());v=np.tile(lo.ravel(order='F'),(len(vectors),1));v[:,mask]=vectors[:,pos:pos+n];pos+=n
  # Fortran ordering is within each individual parameter matrix.
  out.append(np.stack([x.reshape(lo.shape,order='F') for x in v]))
 assert pos==vectors.shape[1]
 return out

def specialize(p,valence='attractive',reset=False,shock_scale=1.,ablation=None):
 p=[a.copy() for a in p];W=p[3]
 if reset:
  target=np.array([-3.42,-4.58,-5.60,31.6,5.8,17.8]) if valence=='attractive' else np.array([2.57,2.24,3.35,31.6,5.8,17.8])
  p[0]=np.array([((np.eye(6)-w.T)@target)*(2/(1+np.exp(-.05*5))) for w in W])[:,None,:]
 else:p[0]=p[0][...,list(range(6)) if valence=='attractive' else [6,7,8,3,4,5]]
 p[2][:,0,0]*=shock_scale
 if ablation=='no_gamma_feedback':p[3][:,3,:3]=0
 if ablation=='no_plasticity':p[1][:]=0;p[2][:]=0
 if ablation=='no_late_tau_switch':p[4][:,0,2]=p[4][:,0,1]
 if ablation=='scalar_tau':p[4][:,0,:]=p[4][:,0,0,None]
 return p

def session(name,n=1,isi=None):
 isi=(135 if name in ['training','extinction'] else 120) if isi is None else isi
 length=30 if name in ['training','extinction'] else 5
 return [dict(name=name,duration=float(d),odor=o,punishment=int(name=='training' and o==0),imaging=int(name in ['imaging','extinction'] and o>=0)) for _ in range(n) for d,o in [(length,0),(isi,-1),(length,1),(isi,-1)]]
def rest(t):
 assert t>=0;return [dict(name='rest',duration=float(t),odor=-1,punishment=0,imaging=0)]
def protocol_isi(isi=135):
 e=session('imaging')+session('training',6,isi)+session('imaging')+rest(86400-300-250)+session('imaging');e[3]['duration']=300;e[4*(6+1)-1]['duration']=300;return e

def protocol_extinction(valence='attractive',extinction_min=None,end_min=180):
 e=session('imaging')+session('training',3)+session('imaging');remaining=end_min*60-300-250
 if extinction_min is None:e+=rest(remaining)
 else:
  first=extinction_min*60-300-250;ext=session('extinction',3);remaining-=first+sum(x['duration'] for x in ext);e+=rest(first)+ext+rest(remaining)
 e+=session('imaging');e[3]['duration']=300;e[4*(3+1)-1]['duration']=300;return e

def protocol_fit():
 e=session('imaging')+session('training',3)+session('imaging')+session('training',3)+session('imaging')+rest(3600-250-300)+session('imaging')+rest(7200-250)+session('imaging')+rest(21*3600-250)+session('imaging')
 for i in [4,16,20,32]:e[i-1]['duration']=300
 return e

def packed(e):
 return np.array([[x['duration'],x['odor'],x['punishment'],x['imaging'],int(x['name']=='training')] for x in e],float)

@njit(cache=True)
def simulate_batch(k0s,fw0s,fwds,Ws,taus,adapts,events,no_adaptation=False):
 """Exact fast response elimination for the published acyclic rate matrix.
 Output: draw x event x six delta-rates, plus final adaptation and plastic state.
 """
 B=len(Ws);E=len(events);out=np.zeros((B,E,6));final_dw=np.zeros((B,2,3));final_adapt=np.ones((B,2));last_training=-1
 for i in range(E):
  if events[i,4]>0:last_training=i
 for b in range(B):
  dw=np.zeros((2,3));ad=np.ones(2);W=Ws[b];k0=k0s[b,0];fw0=fw0s[b,0,0];fwd=fwds[b,0,0];tau=taus[b,0];adapt=adapts[b,0,0];since_training=0.
  for ei in range(E):
   length=events[ei,0];odor=int(events[ei,1]);pun=events[ei,2];dxkc=np.zeros(2)
   for oi in range(2):
    end=ad[oi]*np.exp(-(.0 if no_adaptation else .05)*length*(oi==odor));end=1-(1-end)*np.exp(-length/adapt)
    dxkc[oi]=.5*(ad[oi]+end)*(oi==odor);ad[oi]=end
   inp=np.zeros(6)
   for j in range(6):
    for oi in range(2):inp[j]+=(k0[j]+(dw[oi,j-3] if j>=3 else 0.))*dxkc[oi]
   inp[0]+=27.85*pun;inp[2]+=11.38*pun
   r=np.zeros(6)
   r[3]=min(max(inp[3]+35.2,0.),71.66)-35.2
   r[4]=min(max(inp[4]+9.+W[3,4]*r[3],0.),17.9)-9.
   r[5]=min(max(inp[5]+11.2+W[3,5]*r[3],0.),31.16)-11.2
   for j in range(3):
    r[j]=inp[j]
    for k in range(3,6):r[j]+=W[k,j]*r[k]
   out[b,ei]=r
   for oi in range(2):
    for j in range(3):
     integrated=k0[j]*(dxkc[0]+dxkc[1])
     for k in range(3,6):integrated+=W[k,j]*r[k]
     shock=27.85 if j==0 else (11.38 if j==2 else 0.)
     dw[oi,j]+=(fw0*integrated+fwd*shock*pun)*dxkc[oi]*length/90.
   before=since_training
   if ei>last_training:since_training+=length
   for oi in range(2):
    for j in range(3):
     if j==0:exponent=length/tau[0]
     elif since_training<=10800.:exponent=length/tau[1]
     elif before<10800.:exponent=(10800.-before)/tau[1]+(since_training-10800.)/tau[2]
     else:exponent=length/tau[2]
     dw[oi,j]*=np.exp(-exponent)
  final_dw[b]=dw;final_adapt[b]=ad
 return out,final_dw,final_adapt

def run(p,e,no_adaptation=False):
 W=p[3]
 assert np.all(W[:,:3,:]==0) and np.all(W[:,4:,3:]==0) and np.all(W[:,3,3]==0),'Fast elimination only supports the published rate graph'
 return simulate_batch(*p,packed(e),no_adaptation)

def imaging(out,e):
 inds=[i for i,x in enumerate(e) if x['imaging']];assert len(inds)%2==0
 return out[:,inds].reshape(len(out),len(inds)//2,2,6)
