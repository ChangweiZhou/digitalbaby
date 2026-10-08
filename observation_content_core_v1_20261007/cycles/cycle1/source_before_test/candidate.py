"""U045 instance: fixed KC address; bounded locally accessed fast/slow contents.

No task IDs, teacher bits, context dictionary of learned values or trainable
head. Four output columns ARE the operative content memory. The new rule is an
engineering structural alternative, not a source-faithful biological claim.
"""
import copy, hashlib, json, math
from collections import OrderedDict
from types import SimpleNamespace
import numpy as np
import environment
import config

class FrozenEncoder:
    def __init__(self):
        compartment=environment.native_core.Compartment()
        m=compartment.fly.m
        self.m=SimpleNamespace(B=m.B,pn_type_index=m.pn_type_index,
            kc_side=m.kc_side,active_fraction=m.active_fraction)
        self.model=environment.native_core.stores.bb.bc.native().model
        # Keep the native read/write-capable coordinates; no extra eligible KCs.
        self.mask=(np.abs(np.asarray(m.T)[:,2:]).sum(1)>0)&(np.abs(np.asarray(m.Q)[:,2:]).sum(1)>0)
        self.mask.flags.writeable=False
        self.n=len(self.mask)
        self.birth_record=compartment.birth_record
        self.cache=OrderedDict()
    def code(self,history):
        if history not in self.cache:
            seed=int.from_bytes(hashlib.sha256(b'RAW-CONTEXT-4-v1|'+history).digest()[:8],'little')
            rng=np.random.default_rng(seed)
            p=np.zeros(88,dtype=np.float64)
            idx=rng.choice(88,24,replace=False)
            p[idx]=rng.uniform(.4,1.,size=24);p/=p.max()
            x=np.asarray(self.model.encode_sparse(self.m,p),dtype=np.float64)
            x.flags.writeable=False
            self.cache[history]=x
            if len(self.cache)>256:self.cache.popitem(last=False)
        return self.cache[history]

class LocalContentCore:
    def __init__(self,*,plastic=True):
        if type(plastic) is not bool:raise ValueError('plastic must be bool')
        self.encoder=FrozenEncoder()
        self.fast=np.zeros((self.encoder.n,4),np.float64)
        self.slow=np.zeros_like(self.fast)
        self.last=np.zeros(self.encoder.n,np.float64)
        self.history=b'';self.t=0.;self.bytes_seen=0;self.plastic=plastic;self.cached=None
    def clone(self):
        c=copy.copy(self)
        c.fast=self.fast.copy();c.slow=self.slow.copy();c.last=self.last.copy()
        c.cached=copy.deepcopy(self.cached)
        # Address memoization is bounded and contains no learned state.
        return c
    def _time(self,t):
        if isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or t<self.t:
            raise ValueError('finite monotonic time required')
        return float(t)
    def _view(self,ids,t):
        dt=t-self.last[ids]
        if np.any(dt<0):raise ValueError('row clock reversed')
        return self.fast[ids]*np.exp(-dt[:,None]/config.FAST_TAU), self.slow[ids]*np.exp(-dt[:,None]/config.SLOW_TAU)
    def predict(self,t):
        t=self._time(t)
        if self.cached is not None:raise ValueError('pending prediction')
        code=self.encoder.code(self.history)
        ids=np.flatnonzero((code>0)&self.encoder.mask)
        z=np.ones(len(ids),np.float64)/math.sqrt(len(ids)) if len(ids) else np.zeros(0)
        f,s=self._view(ids,t)
        values=z@(f+s)
        logits=values/config.SCALE; lp=logits-np.logaddexp.reduce(logits);p=np.exp(lp)
        self.cached=dict(t=t,history=self.history,ids=ids,z=z,p=p.copy(),code=code)
        return dict(probabilities=p.tolist(),emitted=config.ALPHABET[int(np.argmax(p))],context=self.history.hex(),native_values=values.tolist())
    def observe(self,byte,t):
        if type(byte) is not int or byte not in config.ALPHABET:raise ValueError('raw byte outside 0123')
        t=self._time(t)
        c=self.cached
        if c is None or c['t']!=t or c['history']!=self.history:raise ValueError('prediction must precede observation')
        ids,z,p=c['ids'],c['z'],c['p']; signs=p.copy();signs[config.ALPHABET.index(byte)]-=1.
        f,s=self._view(ids,t); before_f=f.copy();before_s=s.copy()
        if self.plastic:
            delta=-config.ETA*z[:,None]*signs[None,:]
            f=np.clip(f+(1.-config.SLOW_SHARE)*delta,-config.ROW_BOUND,config.ROW_BOUND)
            s=np.clip(s+config.SLOW_SHARE*delta,-config.ROW_BOUND,config.ROW_BOUND)
        self.fast[ids]=f;self.slow[ids]=s;self.last[ids]=t
        dose=np.abs(f-before_f).sum(0)+np.abs(s-before_s).sum(0)
        row=dict(byte=byte,t=t,context=self.history.hex(),probabilities=p.tolist(),emitted=config.ALPHABET[int(np.argmax(p))],
            signs=signs.tolist(),native_write_l1=dose.tolist(),plastic=self.plastic,
            loss_bits=-math.log2(float(p[config.ALPHABET.index(byte)])),visited_rows=len(ids))
        self.history=(self.history+bytes([byte]))[-4:];self.t=t;self.bytes_seen+=1;self.cached=None
        return row
    def rest(self,seconds):
        if self.cached is not None or isinstance(seconds,bool) or not math.isfinite(seconds) or seconds<0:raise ValueError('invalid rest')
        self.t+=float(seconds)
    def erase_native_content(self):
        if self.cached is not None:raise ValueError('pending prediction')
        self.fast.fill(0.);self.slow.fill(0.)
    def mutable_bytes(self):return self.fast.nbytes+self.slow.nbytes+self.last.nbytes+4
    def state_digest(self):
        h=hashlib.sha256()
        for a in (self.fast,self.slow,self.last):h.update(a.tobytes())
        h.update(json.dumps([self.history.hex(),self.t,self.bytes_seen,self.plastic],sort_keys=True).encode())
        if self.cached is not None:
            h.update(str((self.cached['t'],self.cached['history'].hex())).encode())
            for name in ('ids','z','p','code'):h.update(self.cached[name].tobytes())
        return h.hexdigest()
