"""Six bounded engineered cores. Learner API receives bytes/time/permission only."""
import copy
import hashlib
import json
import math
import random
import numpy as np
from . import config

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
        self.arm=arm;self.w=np.zeros(31);self.p=np.eye(31)*10
        self.rng=random.Random(20261011047);self.cache=[];self.history=[]
        self.updates=self.extra_updates=0;self.scalar_ops=self.traffic_bytes=0
        self.birth=copy.deepcopy(self.state())
        for v in self.birth.values():
            if isinstance(v,np.ndarray):v.flags.writeable=False
    def state(self):
        return dict(w=self.w,p=self.p,cache=self.cache,history=self.history,updates=self.updates,
                    extra_updates=self.extra_updates,rng=self.rng.getstate())
    def digest(self):return state_hash(self.state())
    def clone(self):return copy.deepcopy(self)
    def restore(self,state):
        def tuples(v):return tuple(tuples(x) for x in v) if isinstance(v,(list,tuple)) else v
        if set(state)!=set(self.state()):raise ValueError('complete learned state required')
        for k,v in state.items():
            if k=='rng':self.rng.setstate(tuples(v))
            else:setattr(self,k,copy.deepcopy(v))
    def erase(self):self.restore(self.birth)
    def payload_bytes(self):
        s=self.state()
        return sum(v.nbytes for v in s.values() if isinstance(v,np.ndarray))+len(json.dumps(
            {k:v for k,v in s.items() if not isinstance(v,np.ndarray)},separators=(',',':')).encode())
    def value(self,ids):return math.fsum(float(self.w[i]) for i in ids)
    def support(self,ids,target):
        key=[int(config.OPERANDS[ids[0]]),int(config.OPERANDS[ids[1]-15])]
        index=next((i for i,row in enumerate(self.cache) if row['key']==key),None)
        past=self.cache[index]['answers'] if index is not None else []
        votes=[past.count(y) for y in range(9)]
        n=len(past);count=max(votes);mode=votes.index(count) if count>=2 and 2*count>n else None
        eligible=mode is not None;conflict=eligible and target!=mode
        probability=sum(self.history)/len(self.history) if self.history else 0.
        draw=self.rng.random()
        factor=.25 if ((conflict if self.arm=='COV-SUPPORT' else eligible and draw<probability)) else 1.
        return key,index,dict(eligible=eligible,prior_count=n,mode=mode,mode_count=count,
            conflict=bool(conflict),probability=probability,random=draw,factor=factor)
    def update(self,ids,target,allowed,cached_value):
        if type(allowed) is not bool or type(target) is not int or not 0<=target<=8:
            raise ValueError('observed target/permission')
        if len(ids)!=3 or any(type(i) is not int for i in ids) or not(0<=ids[0]<15 and 15<=ids[1]<30 and ids[2]==30):
            raise ValueError('ordered sensory features')
        out=dict(performed=False,replay_index=None,extra_performed=False,extra_ids=None,extra_target=None,gate=None)
        if not allowed:return out
        gate=dict(eligible=False,prior_count=0,mode=None,mode_count=0,conflict=False,probability=0.,random=None,factor=1.)
        if self.arm!='COV-R':key,index,gate=self.support(ids,target)
        r=target-cached_value;q=self.p[:,ids[0]]+self.p[:,ids[1]]+self.p[:,ids[2]]
        u=math.fsum(float(q[i]) for i in ids)
        if u<0 or not math.isfinite(u):raise ValueError('invalid confidence')
        gamma_r=1. if abs(r)<=1. else 1./abs(r)
        gamma=gate['factor']*gamma_r;d=1.+gamma*u
        if not math.isfinite(d) or d<=0:raise ValueError('invalid denominator')
        self.w=np.clip(self.w+(gamma*r/d)*q,-4*config.KAPPA,4*config.KAPPA)
        self.p=self.p-(gamma/d)*np.outer(q,q);self.p=(self.p+self.p.T)/2
        self.updates+=1
        # These are explicit conservative algorithmic estimates, not measured bandwidth.
        self.scalar_ops+=11000+(256 if self.arm!='COV-R' else 0)
        self.traffic_bytes+=96000+(1024 if self.arm!='COV-R' else 0)
        if self.arm!='COV-R':
            if gate['eligible']:self.history=(self.history+[int(gate['conflict'])])[-32:]
            if index is None:
                self.cache.append(dict(key=key,answers=[target]));self.cache=self.cache[-32:]
            else:self.cache[index]['answers']=(self.cache[index]['answers']+[target])[-8:]
        gate['gamma']=gamma;gate['gamma_r']=gamma_r
        out.update(performed=True,gate=gate);return out


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
