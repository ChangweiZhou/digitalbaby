"""V182: remove T, keep H/output births, local signals and AdamW constants.

Dropout remains immediately before the readout, now on H. This tests the
whole declared shallow package, not depth independently of dropout placement.
"""
import parent_learner as P
T=P.T
class Learner(P.Learner):
 def __init__(self,world='M1'):
  super().__init__(world)
  for k in ('T','Tb'):
   del self.p[k];del self.m[k];del self.v[k]
  del self.ft
 def forward(self,x,training=False):
  xx=x[:,self.ix];ah=(xx*self.p['H']).sum(2)+self.p['Hb'];h=ah.clamp_min(0)
  mask=(T.rand(h.shape,generator=self.rng)>=.1).float()/.9 if training else T.ones_like(h)
  z=h*mask;return z@self.p['O'].T+self.p['Ob'],(xx,ah,h,mask,z)
 def gradients(self,x,y,ids=None):
  logits,(xx,ah,h,mask,z)=self.forward(x,True)
  e=logits.softmax(1);e[T.arange(len(y)),y]-=1;e/=len(y)
  dh=(e@self.fh)*(ah>0)*mask
  return {'H':T.einsum('nj,njr->jr',dh,xx),'Hb':dh.sum(0),'O':e.T@z,'Ob':e.sum(0)}
 def step(self,x,y,ids=None,arm=None):
  grad=self.gradients(x,y,ids);self.clock+=1
  for k,p in self.p.items():
   self.m[k].mul_(.9).add_(grad[k],alpha=.1);self.v[k].mul_(.999).addcmul_(grad[k],grad[k],value=.001)
   d=-.001*(self.m[k]/(1-.9**self.clock)/(self.v[k]/(1-.999**self.clock)).sqrt().add(1e-8))-.001*.0001*p
   p.add_(d)
  assert all(T.isfinite(p).all() for p in self.p.values())
  return {'gain':1.0}
