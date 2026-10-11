"""Six bounded engineered cores. Learner API receives bytes/time/permission only."""
import copy
import hashlib
import json
import math
import random
import numpy as np
import config

def emitted(value):
    if not math.isfinite(value):raise ValueError('nonfinite output')
    return min(range(9),key=lambda k:((k-value)**2,k))

def features(left,right):
    if type(left) is not int or type(right) is not int:raise ValueError('byte identity')
    return (config.OPERANDS.index(left),15+config.OPERANDS.index(right),30)

def state_hash(state):
    h=hashlib.sha256()
    for name in sorted(state):
        v=state[name];h.update(name.encode())
        if isinstance(v,np.ndarray):
            h.update(str((v.shape,v.dtype.str)).encode());a=v.copy();a[a==0]=0
            h.update(a.tobytes())
        else:h.update(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    return h.hexdigest()

class Core:
    def __init__(self,arm):
        if arm not in config.ARMS:raise ValueError('arm roster')
        self.arm=arm;self.w=np.zeros(31)
        self.p=np.ones(31)*10 if arm.startswith('DIAG') else np.eye(31)*10
        self.rng=random.Random(20261011047);self.cache=[]
        self.updates=self.extra_updates=0
        self.scalar_ops=self.traffic_bytes=0
        self.birth=copy.deepcopy(self.state())
        for v in self.birth.values():
            if isinstance(v,np.ndarray):v.flags.writeable=False

    def state(self):
        return dict(w=self.w,p=self.p,cache=self.cache,updates=self.updates,
                    extra_updates=self.extra_updates,rng=self.rng.getstate())
    def digest(self):return state_hash(self.state())
    def clone(self):return copy.deepcopy(self)
    def restore(self,state):
        def tuples(v):return tuple(tuples(x) for x in v) if isinstance(v,(list,tuple)) else v
        for k,v in state.items():
            if k=='rng':self.rng.setstate(tuples(v))
            else:setattr(self,k,copy.deepcopy(v))
    def erase(self):self.restore(self.birth)
    def payload_bytes(self):
        s=self.state()
        return sum(v.nbytes for v in s.values() if isinstance(v,np.ndarray))+len(json.dumps(
            {k:v for k,v in s.items() if not isinstance(v,np.ndarray)},separators=(',',':')).encode())
    def value(self,ids):return math.fsum(float(self.w[i]) for i in ids)

    def correction(self,ids,target,value,information):
        r=target-value
        if self.arm.startswith('DIAG'):
            q=np.zeros(31);q[list(ids)]=self.p[list(ids)]
        else:q=self.p[:,ids[0]]+self.p[:,ids[1]]+self.p[:,ids[2]]
        u=math.fsum(float(q[i]) for i in ids)
        if u<0 or not math.isfinite(u):raise ValueError('invalid confidence')
        threshold=math.sqrt(1+u) if self.arm.endswith('-I') else 1.
        gamma=1. if abs(r)<=threshold else threshold/abs(r)
        d=1+gamma*u
        if not math.isfinite(d) or d<=0:raise ValueError('invalid denominator')
        self.w=np.clip(self.w+(gamma*r/d)*q,-4*config.KAPPA,4*config.KAPPA)
        if information:
            factor=gamma/d
            if self.arm.startswith('DIAG'):self.p=self.p-factor*q**2
            else:
                self.p=self.p-factor*np.outer(q,q);self.p=(self.p+self.p.T)/2
        self.scalar_ops += (600 if self.arm.startswith('DIAG') else (11000 if information else 400))
        self.traffic_bytes += (2400 if self.arm.startswith('DIAG') else (96000 if information else 4000))

    def update(self,ids,target,allowed,cached_value):
        if type(allowed) is not bool or type(target) is not int or not 0<=target<=8:
            raise ValueError('observed target/permission')
        if not allowed:return dict(performed=False,replay_index=None,extra_performed=False,extra_ids=None,extra_target=None)
        self.correction(ids,target,cached_value,True);self.updates+=1
        index=chosen=None;extra=False
        if self.arm in ('COV-REPLAY','COV-EXTRA'):
            if self.cache:
                index=self.rng.randrange(len(self.cache))
                chosen=self.cache[index] if self.arm=='COV-REPLAY' else dict(ids=list(ids),target=target)
                self.correction(chosen['ids'],chosen['target'],self.value(chosen['ids']),False)
                self.extra_updates+=1;extra=True
            self.cache.append(dict(ids=list(ids),target=target));self.cache=self.cache[-32:]
            self.scalar_ops+=50;self.traffic_bytes+=32
        return dict(performed=True,replay_index=index,extra_performed=extra,
                    extra_ids=chosen['ids'] if chosen else None,extra_target=chosen['target'] if chosen else None)


class ByteLearner:
    """Only feed(byte, ticks, allowed) receives experimental input."""
    def __init__(self, arm):
        self.core = Core(arm)
        self.line = b''
        self.pending = None
        self.ticks = 0
        self.bytes_seen = 0

    def clone(self):
        return copy.deepcopy(self)

    def rest(self, ticks):
        if type(ticks) is not int or ticks < 0 or self.line or self.pending is not None:
            raise ValueError('rest must occur at complete-record boundary')
        self.ticks += ticks

    def feed(self, byte, ticks, allowed):
        if type(byte) is not int or type(ticks) is not int or ticks <= self.ticks or type(allowed) is not bool:
            raise ValueError('public input type/time')
        prediction = update = None
        if byte == 10:
            if len(self.line) not in (4, 5):
                raise ValueError('incomplete line')
            self.line = b''
            self.pending = None
        elif len(self.line) == 0:
            if byte not in config.OPERANDS:
                raise ValueError('operand')
            self.line += bytes([byte])
        elif len(self.line) == 1:
            if byte != 43:
                raise ValueError('plus')
            self.line += b'+'
        elif len(self.line) == 2:
            if byte not in config.OPERANDS:
                raise ValueError('operand')
            self.line += bytes([byte])
        elif len(self.line) == 3:
            if byte != 61:
                raise ValueError('prompt')
            self.line += b'='
            ids = features(self.line[0], self.line[2])
            value = self.core.value(ids)
            prediction = {'value': value, 'emitted': emitted(value) + 48}
            self.pending = (ids, value)
        elif len(self.line) == 4:
            if byte not in config.ANSWERS or self.pending is None:
                raise ValueError('actual observed answer required')
            ids, value = self.pending
            update = self.core.update(ids, byte-48, allowed, value)
            self.pending = None
            self.line += bytes([byte])
        else:
            raise ValueError('overlong record')
        self.ticks = ticks
        self.bytes_seen += 1
        return {'prediction': prediction, 'update': update}
