"""37,313 parameters; manual source-local updates, no autograd or native banks."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,hashlib,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'minifly_v178_recurrent_failure_localization_dev_20261102/local/compat'))
import torch as T
import numpy as np
T.set_num_threads(1)
def seed(s):return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],'big')%(2**63-1)
class Learner:
 def __init__(self,world='M1'):
  pi=np.random.default_rng(seed('V181|KC-H-birth')).permutation(5177)
  self.indices=pi[(256*np.arange(64)[:,None]+np.arange(256))%5177].astype(np.uint16)
  self.ix=T.tensor(self.indices.astype(np.int64));g=T.Generator().manual_seed(seed('V181|'+world+'|birth'))
  self.p={k:T.randn(shape,generator=g)*std if std else T.zeros(shape) for k,shape,std in [('H',(64,256),1/3),('Hb',(64,),0),('T',(64,64),1/8),('Tb',(64,),0),('O',(257,64),1/8),('Ob',(257,),0)]}
  self.fh=T.randn((257,64),generator=T.Generator().manual_seed(21))/math.sqrt(257)
  self.ft=T.randn((257,64),generator=T.Generator().manual_seed(22))/math.sqrt(257)
  self.m={k:T.zeros_like(v) for k,v in self.p.items()};self.v={k:T.zeros_like(v) for k,v in self.p.items()};self.clock=0
  self.rng=T.Generator().manual_seed(seed('V181|'+world+'|dropout'))
 def forward(self,x,training=False):
  xx=x[:,self.ix];ah=(xx*self.p['H']).sum(2)+self.p['Hb'];h=ah.clamp_min(0);at=h@self.p['T'].T+self.p['Tb'];mask=(T.rand(at.shape,generator=self.rng)>=.1).float()/.9 if training else T.ones_like(at);z=at.clamp_min(0)*mask;logits=z@self.p['O'].T+self.p['Ob']
  return logits,(xx,ah,h,at,mask,z)
 def gradients(self,x,y,ids):
  logits,(xx,ah,h,at,mask,z)=self.forward(x,True);e=logits.softmax(1);e[T.arange(len(y)),y]-=1;e/=len(y);dt=(e@self.ft)*(at>0)*mask;dh=(e@self.fh)*(ah>0)
  def hidden(sel,scale):return {'H':T.einsum('nj,njr->jr',dh[sel],xx[sel])*scale,'Hb':dh[sel].sum(0)*scale,'T':dt[sel].T@h[sel]*scale,'Tb':dt[sel].sum(0)*scale}
  full=hidden(T.ones(len(y),dtype=T.bool),1);full.update(O=e.T@z,Ob=e.sum(0))
  unique=sorted(set(ids),key=lambda v:hashlib.sha256(('V180|micro|'+v).encode()).hexdigest());left=set(unique[::2]);sel=T.tensor([i in left for i in ids]);n1=int(sel.sum());n2=len(y)-n1
  a=hidden(sel,len(y)/n1) if n1 else {k:T.zeros_like(v) for k,v in full.items() if k not in ('O','Ob')};b=hidden(~sel,len(y)/n2) if n2 else {k:T.zeros_like(v) for k,v in a.items()}
  err=max(float(((a[k]*n1+b[k]*n2)/len(y)-full[k]).abs().max()) for k in a)
  for k in a:assert T.allclose((a[k]*n1+b[k]*n2)/len(y),full[k],atol=1e-7,rtol=1e-6),k
  reliable=min(len(unique[::2]),len(unique[1::2]))>=2;gains={}
  for k in ('H','T'):
   bias=k+'b';dot=(a[k]*b[k]).sum(1)+a[bias]*b[bias];na=(a[k].square().sum(1)+a[bias].square()).sqrt();nb=(b[k].square().sum(1)+b[bias].square()).sqrt();co=(dot/(na*nb+1e-12)).clamp(0,1) if reliable else T.zeros(64);gains[k]=.25+.75*co
  return full,gains,{'reliable':reliable,'group_targets':[n1,n2],'group_records':[len(unique[::2]),len(unique[1::2])],'gradient_recombination_max_error':err}
 def step(self,x,y,ids,arm):
  grad,gain,info=self.gradients(x,y,ids);self.clock+=1;d={}
  for k,p in self.p.items():
   self.m[k].mul_(.9).add_(grad[k],alpha=.1);self.v[k].mul_(.999).addcmul_(grad[k],grad[k],value=.001)
   d[k]=-.001*(self.m[k]/(1-.9**self.clock)/(self.v[k]/(1-.999**self.clock)).sqrt().add(1e-8))-.001*.0001*p
  info['energy']={};info['gain']={}
  for k in ('H','T'):
   en=d[k].square().sum(1)+d[k+'b'].square();cand=(en*gain[k].square()).sum()
   if arm=='NORMAL':a=T.ones(64)
   elif arm=='FIXED-LOW':a=T.full((64,),.35)
   elif arm=='CGLP':a=gain[k]
   elif arm=='RMS-SHAM':a=T.ones(64)*((cand/en.sum()).sqrt() if en.sum()>0 else .25)
   else:raise ValueError(arm)
   info['energy'][k]={'candidate':float(cand),'actual':float((en*a.square()).sum())};info['gain'][k]=[float(a.min()),float(a.max()),float(a.mean())]
   d[k]*=a[:,None];d[k+'b']*=a
  for k in self.p:self.p[k].add_(d[k])
  assert all(T.isfinite(p).all() for p in self.p.values())
  return info
 def state(self):return {'weights':self.p,'moment1':self.m,'moment2':self.v,'clock':self.clock,'dropout_rng':self.rng.get_state()}
 def restore(self,s):
  self.p={k:v.clone() for k,v in s['weights'].items()};self.m={k:v.clone() for k,v in s['moment1'].items()};self.v={k:v.clone() for k,v in s['moment2'].items()};self.clock=s['clock'];self.rng.set_state(s['dropout_rng'])
