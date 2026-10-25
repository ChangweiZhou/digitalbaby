"""Output-local learning only. No native/encoder/time backpropagation."""
import copy,hashlib,json
import numpy as np
V,D=257,16
def warmup(n):return 1/16+15/16*min(n/1024,1)
def logits(C,v,beta=None):
 u=(v-v.mean())/4;z=C@u
 return z if beta is None else z+beta

def probabilities(C,v,beta=None):
 z=logits(C,v,beta);z=z-z.max();lp=z-np.log(np.exp(z).sum());return np.exp(lp),lp

def gradients(C,v,y,beta=None):
 p,lp=probabilities(C,v,beta);r=p.copy();r[y]-=1;gv=C.T@r/4;gv-=gv.mean();u=(v-v.mean())/4
 return gv,np.outer(r,u),r,float(-lp[y])

def project(C):
 C=C-C.mean(1,keepdims=True);return C/np.maximum(1,np.linalg.norm(C,axis=1,keepdims=True)/4)

class Output:
 def __init__(self,C,contexts,arm,eta):
  self.C=C.copy();self.v=np.zeros((1 if arm=='C0' else contexts,D));self.beta=np.zeros(V) if arm=='BIAS' else None;self.n=0;self.arm=arm;self.eta=eta
 def values(self,ctx):return self.v[0 if self.arm=='C0' else ctx]
 def read(self,ctx):return probabilities(self.C,self.values(ctx),self.beta)
 def observe(self,ctx,y):
  idx=0 if self.arm=='C0' else ctx;v=self.v[idx];gv,gc,r,loss=gradients(self.C,v,y,self.beta);s=warmup(self.n)
  e=(np.tanh((v-v.mean())/2)-self.C[y])/2 if self.arm=='A' else gv
  self.v[idx]-=self.eta*s*e
  if self.arm in ['C','C0']:self.C=project(self.C-self.eta/16*s*gc)
  if self.beta is not None:self.beta-=self.eta*s*r
  self.n+=1;return loss
 def digest(self):
  b=self.C.tobytes()+self.v.tobytes()+(self.beta.tobytes() if self.beta is not None else b'')+str((self.n,self.arm,self.eta)).encode();return hashlib.sha256(b).hexdigest()
 def clone(self):return copy.deepcopy(self)
