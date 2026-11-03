"""V183: unchanged forward circuit, only H credit assignment changes."""
import bootstrap as B
T=B.T;P=B.P
HIDDEN=('H','Hb')
def hnorm(delta):return T.sqrt(sum(delta[k].double().square().sum() for k in HIDDEN))
def proposals(net,grad):
 clock=net.clock+1;m={};v={};delta={}
 for k,p in net.p.items():
  m[k]=net.m[k].clone().mul_(.9).add_(grad[k],alpha=.1);v[k]=net.v[k].clone().mul_(.999).addcmul_(grad[k],grad[k],value=.001)
  delta[k]=-.001*(m[k]/(1-.9**clock)/(v[k]/(1-.999**clock)).sqrt().add(1e-8))-.001*.0001*p
 return m,v,delta

def base_gradients(net,x,y):
 logits,cache=net.forward(x,True);xx,ah,h,at,mask,z=cache;e=logits.softmax(1);e[T.arange(len(y)),y]-=1;e/=len(y);dt=(e@net.ft)*(at>0)*mask;dh=(e@net.fh)*(ah>0)
 grad={'H':T.einsum('nj,njr->jr',dh,xx),'Hb':dh.sum(0),'T':dt.T@h,'Tb':dt.sum(0),'O':e.T@z,'Ob':e.sum(0)}
 return grad,cache

def paired(net,cache,y,noise):
 xx,ah,h,at,mask,z=cache;n=len(y);a=ah.double();sigma=.01*max(float(a.square().mean().sqrt()),1e-3)
 # Stored float32 parameters are cast read-only. Sigma, weights and mask are
 # shared and frozen across plus/minus. There is no derivative through RMS.
 wt,bt,wo,bo=[net.p[k].double() for k in ('T','Tb','O','Ob')];drop=mask.double()
 def losses(v):return T.nn.functional.cross_entropy(((v.clamp_min(0)@wt.T+bt).clamp_min(0)*drop)@wo.T+bo,y,reduction='none')
 plus=losses(a+sigma*noise);minus=losses(a-sigma*noise);q=(plus-minus)/(2*sigma)
 return q,sigma,losses

def hcredit(xx,q,noise):
 delta=q[:,None]*noise/len(q)
 return {'H':T.einsum('nj,njr->jr',delta,xx.double()).float(),'Hb':delta.sum(0).float()}

class Learner(P.Learner):
 def __init__(self,world='M1'):
  super().__init__(world);self.credit_rng=T.Generator().manual_seed(P.seed('V183|'+world+'|H-credit'));self.shuffle_rng=T.Generator().manual_seed(P.seed('V183|'+world+'|H-shuffle'))
 def step(self,x,y,ids,arm):
  if arm=='NORMAL':return super().step(x,y,ids,'NORMAL')
  if arm not in ('HPC','SHUFFLED'):raise ValueError(arm)
  grad,cache=base_gradients(self,x,y);noise=T.randn(cache[1].shape,dtype=T.float64,generator=self.credit_rng);q,sigma,_=paired(self,cache,y,noise);real=hcredit(cache[0],q,noise);realgrad={**grad,**real};mp,vp,dp=proposals(self,realgrad);shift=None;scale=1.
  if arm=='SHUFFLED':
   shift=int(T.randint(1,len(y),(1,),generator=self.shuffle_rng)) if len(y)>1 else 0
   qs=T.roll(q,shift) if len(y)>1 else T.zeros_like(q);shuf=hcredit(cache[0],qs,noise);mg,vg,dg=proposals(self,{**grad,**shuf});np,ns=hnorm(dp),hnorm(dg)
   if ns<1e-20 and np>=1e-20:raise RuntimeError('SHAM_STRENGTH_INVALID')
   scale=float(np/ns) if ns>=1e-20 else 0.
   for k in HIDDEN:dg[k]=dg[k]*scale
  else:mg,vg,dg=mp,vp,dp
  before={k:self.p[k].clone() for k in HIDDEN};target_energy=float(hnorm(dp)**2);proposed_energy=float(hnorm(dg)**2)
  for k in self.p:
   self.m[k]=mg[k];self.v[k]=vg[k];self.p[k].add_(dg[k])
  self.clock+=1
  actual=sum((self.p[k]-before[k]).double().square().sum() for k in HIDDEN)
  if not all(T.isfinite(p).all() for d in (self.p,self.m,self.v) for p in d.values()):raise RuntimeError('NUMERIC_INVALID')
  return {'arm':arm,'n':len(y),'sigma':sigma,'shuffle_shift':shift,'sham_scale':scale,'HPC_counterfactual_H_energy':target_energy,'applied_proposed_H_energy':proposed_energy,'realized_H_energy':float(actual),'one_noise_pair':True,'loss_target_evaluations':3*len(y),'auxiliary_float64':True,'extra_H_relu_derivative':False}
 def state(self):return {**super().state(),'credit_rng':self.credit_rng.get_state(),'shuffle_rng':self.shuffle_rng.get_state()}
 def restore(self,s):
  super().restore(s)
  for name in ('credit_rng','shuffle_rng'):
   if name in s:getattr(self,name).set_state(s[name])
