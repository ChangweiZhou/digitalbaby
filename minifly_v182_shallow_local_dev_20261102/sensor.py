"""One immutable FE-PERM/KC circuit. No native memory model is constructed."""
import hashlib,json,math,ctypes
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
ROOT=Path(__file__).resolve().parent.parent
P=ROOT/'minifly_v182_shallow_local_dev_20261102/assets/native'
def sha(b):return hashlib.sha256(b).hexdigest()
class Sensor:
 def __init__(self):
  f=json.loads((P/'FULL151_BIRTH_FINGERPRINT.json').read_text())['other_birth_state']
  z=np.load(P/'REFERENCE_SOURCE/minifly/iteration10/models/reference.npz')
  self.pn=z['pn_type_index'];self.side=z['kc_side']
  for name,a in [('pn_type_index',self.pn),('kc_side',self.side)]:
   assert sha(a.tobytes())==f['arrays']['fly.m.'+name]['sha256']
  b=np.load(P/'FULL151_CANONICAL_B.npz');self.B=csr_matrix((b['data'],b['indices'],b['indptr']),shape=tuple(b['shape']))
  self.frac=float.fromhex(f['scalars']['fly.m.active_fraction']);self.perm=np.random.default_rng(20261017).permutation(88)
  rng=np.random.default_rng(int(sha(b'BYTE-CORE-1|RECEPTOR|v1')[:16],16));self.R=np.zeros((256,88))
  # Recover the exact ARM/FMA birth, rather than x86's two-round uniform.
  fma=ctypes.CDLL('libm.so.6').fma;fma.argtypes=[ctypes.c_double]*3;fma.restype=ctypes.c_double
  for i in range(256):
   idx=rng.choice(88,6,replace=False);self.R[i,idx]=[fma(1.-.4,float(v),.4) for v in rng.random(6)]
  self.limits=[math.ceil(self.frac*sum(self.side==s)) for s in (0,1)]
  self.arrays=[self.pn,self.side,self.B.data,self.B.indices,self.B.indptr,self.perm,self.R]
  for a in self.arrays:a.flags.writeable=False
  self.digest=sha(b''.join(a.tobytes() for a in self.arrays)+self.frac.hex().encode())
 def encode(self,p):
  q=p/p.max() if p.max()>0 else p.copy();raw=np.asarray(self.B.T@q[self.pn]).ravel();out=np.zeros(5177,np.float32)
  for s in (0,1):
   ix=np.flatnonzero((self.side==s)&(raw>0));ix=ix[np.lexsort((ix,-raw[ix]))[:self.limits[s]]];out[ix]=1
  return out
 def codes(self,r):
  # Prediction happens before the target byte enters FE. One observation per byte.
  p=np.zeros(88);out=[]
  for i in range(len(r)+1):
   out.append(self.encode(p*.95))
   if i<len(r):p=(p*.95)[self.perm]+self.R[r[i]]
  return np.array(out)
