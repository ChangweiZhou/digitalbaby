"""Generic local learner adapter; no task/evaluator arithmetic imports."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'minifly_v146_r1_dev_20261024'),str(ROOT/'minifly_learned_bank_v1_20261024'),str(ROOT/'minifly_v144_dev_20261024')]
from core import Local
import lb1
from reference import ByteTrace
class Trace:
 def __init__(self):self.base=ByteTrace(lb1.BC.R_TABLE,lb1.fastmem.PERM,lb1.DELTA,lb1.fastmem.FEPerm);self.cache={}
 def context(self,p,prefix=b''):
  key=(bytes(p),bytes(prefix))
  if key not in self.cache:
   if len(self.cache)>=4096:self.cache.pop(next(iter(self.cache)))
   self.cache[key]=np.r_[self.base.context(*key),np.zeros(32)]
  return self.cache[key]
class Model(Local):
 def __init__(self,*args,features=True,writes=True,**kwargs):super().__init__(*args,**kwargs);self.features=features;self.writes=writes
 def observe(self,p,y):
  if self.writes:
   a,ix=self.code(p);v=2*self.theta[:,ix].sum(axis=1)/np.sqrt(len(ix));target=np.array([1 if y&(1<<j) else -1 for j in range(9)]);e=(np.tanh(v/2)-target)/2;s=1/16+15/16*min(1,self.records/1024)
   self.theta[:,ix]-=self.alpha*s*e[:,None]/np.sqrt(len(ix))
   if self.features:
    d=(self.G@e)*(a>0);self.W-=.003*s*np.outer(d,p);self.b-=.003*s*d
  self.targets+=1
 def arrays_hash(self):
  import hashlib
  return hashlib.sha256(b''.join(x.tobytes() for x in [self.W,self.b,self.theta,self.V,self.G])).hexdigest()
def teach(m,tr,p,observed):
 prefix=bytearray()
 for y in list(bytes(observed))+[256]:
  m.observe(tr.context(p,prefix),y)
  if y!=256:prefix.append(y)
 m.records+=1
def generate(m,tr,p):
 out=bytearray()
 for _ in range(8):
  v=m.margins(tr.context(p,out));y=sum(1<<j for j,x in enumerate(v) if x>0)
  if y==256:return bytes(out),'EOS'
  if y>256:return bytes(out),'INVALID'
  out.append(y)
 return bytes(out),'TRUNCATED'
