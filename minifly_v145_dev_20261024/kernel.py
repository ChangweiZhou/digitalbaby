"""Generic predictive code learner. No task generator or arithmetic imports."""
import copy, hashlib
import numpy as np

def orth(r,n,d):
 q,v=np.linalg.qr(r.normal(size=(n,d)));return q*np.where(np.diag(v)<0,-1,1)
def bit(y): return np.array([(int(y)>>j)&1 for j in range(9)],float)
class Code:
 def __init__(self,seed,d=4,lr=.01,mode='LOCAL'):
  r=np.random.default_rng(seed);self.z=r.normal(0,1/np.sqrt(d),(257,d));self.U=orth(r,88,d);self.C=orth(r,d,d);self.B=r.normal(0,1/3,(d,9));self.D=r.normal(0,1/np.sqrt(d),(9,d));self.T=np.eye(d);self.c=np.zeros(d);self.b=np.zeros(9);self.n=0;self.clips=0;self.lr=lr;self.mode=mode;self.d=d
 def gradients(self,x,y):
  z=self.z[x];q=self.T@z+self.c;u=(q-self.z[y])/self.d;l=self.D@q+self.b;e=(1/(1+np.exp(-np.clip(l,-700,700)))-bit(y))/9
  v=(self.D.T@e if self.mode=='BP' else self.B@e)+u
  return {'D':np.outer(e,q),'b':e,'T':np.outer(v,z),'c':v,'z':self.T.T@v if self.mode=='BP' else self.C@v}
 def loss(self,x,y,target=None):
  q=self.T@self.z[x]+self.c;l=self.D@q+self.b;t=self.z[y] if target is None else target
  return float(np.mean(np.logaddexp(0,l)-bit(y)*l)+np.sum((q-t)**2)/(2*self.d))
 def observe(self,x,y):
  g=self.gradients(x,y);eta=self.lr*min(1,self.n/1024)
  for k,v in g.items():
   update=eta*v;norm=np.linalg.norm(update)
   if norm>1:self.clips+=1;update/=norm
   if k=='z':self.z[x]-=update
   else:setattr(self,k,getattr(self,k)-update)
  norm=np.linalg.norm(self.z[x]);self.z[x]*=min(1,4/max(norm,1e-30));self.n+=1
  assert all(np.isfinite(v).all() for v in [self.z,self.D,self.T,self.c,self.b])
 def digest(self):return hashlib.sha256(b''.join(v.tobytes() for v in [self.z,self.U,self.C,self.B,self.D,self.T,self.c,self.b])+str((self.n,self.clips)).encode()).hexdigest()
