"""Fixed independent role/magnitude receptors. Contains no addition operation."""
import copy
from collections import OrderedDict
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
import dependencies

class ByteRoles:
    def __init__(self): self.line=b''
    def feed(self,byte):
        if type(byte) is not int or byte not in b'012345678+=\n ':
            raise ValueError('undeclared byte')
        if byte==10:
            self.line=b'';return False
        if byte==32: return False
        if len(self.line)>=5: raise ValueError('line too long')
        self.line+=bytes([byte])
        if byte==61:
            if len(self.line)!=4 or self.line[1]!=43 or self.line[0] not in b'01234' or self.line[2] not in b'01234':
                raise ValueError('not a declared single-digit expression')
            return True
        return False
    def operands(self):
        if len(self.line)!=4 or self.line[-1]!=61: raise ValueError('no prompt')
        return self.line[0]-48,self.line[2]-48

def receptors(operands):
    if len(operands)!=2 or any(type(d) is not int or not 0<=d<=4 for d in operands):
        raise ValueError('two declared operand digits required')
    pn=np.zeros(88,np.float64)
    for role,digit in enumerate(operands):
        base=32*role
        pn[base+digit]=.5
        for threshold in range(4):
            pn[base+8+threshold]=float(digit>threshold)
            pn[base+16+threshold]=float(digit<=threshold)
        pn[base+20]=.25
    return pn

class SharedEncoder:
    def __init__(self):
        s=dependencies.inherited.Compartment();m=s.fly.m
        self.template=SimpleNamespace(B=m.B,pn_type_index=m.pn_type_index,kc_side=m.kc_side,active_fraction=m.active_fraction)
        self.mask=(np.abs(np.asarray(m.T)[:,2:]).sum(1)>0)&(np.abs(np.asarray(m.Q)[:,2:]).sum(1)>0)
        self.mask.flags.writeable=False
        self.n=len(self.mask);self.birth=s.birth_record
        self.model=dependencies.inherited.stores.bb.bc.native().model
        self.cache=OrderedDict()
        # Shared U069/U070 structure; sampling knows no labels, sums or worlds.
        features=[32*r+j for r in range(2) for j in list(range(5))+list(range(8,12))+list(range(16,20))+[20]]
        chosen=np.random.default_rng(20261007).permutation(np.flatnonzero(self.mask))[:len(features)*15]
        rows=[];cols=[];weights=[]
        for i,pn in enumerate(features):
            gain=2. if pn%32<5 else (4. if pn%32==20 else 1.)
            rows.extend([pn]*15);cols.extend(chosen[i*15:(i+1)*15]);weights.extend([gain]*15)
        self.projection=csr_matrix((weights,(rows,cols)),shape=(88,self.n),dtype=np.float64)
        self.projection.data.flags.writeable=False
        self.mode='ROLE_PARTITIONED_FIXED_PN_KC_V2'
    def code(self,operands):
        key=tuple(operands)
        if key not in self.cache:
            x=np.asarray(receptors(key)@self.projection,dtype=np.float64)
            x.flags.writeable=False;self.cache[key]=x
            if len(self.cache)>25: self.cache.popitem(last=False)
        return self.cache[key]
    def clone(self):
        c=copy.copy(self);c.cache=self.cache.copy();return c
