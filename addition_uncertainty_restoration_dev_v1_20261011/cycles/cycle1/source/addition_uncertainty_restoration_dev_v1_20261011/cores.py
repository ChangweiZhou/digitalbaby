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
    def __init__(self,arm,world_id=0):
        if arm not in config.ARMS:raise ValueError('arm roster')
        self.arm=arm;self.w=np.zeros(31);self.p=np.eye(31)*10
        self.rng=random.Random(20261011047);self.cache=[];self.history=[]
        self.updates=self.extra_updates=0;self.scalar_ops=self.traffic_bytes=0
        if arm in config.RESTORE_ARMS:
            self.world_id=world_id
            seed=int.from_bytes(hashlib.sha256(b"U034-DIRECTION-V1:"+str(world_id).encode('ascii')).digest()[:8],'big')
            self.direction_rng=random.Random(seed);self.buffer=[];self.qualified=0;self.restorations=0;self.fallbacks=0
        self.birth=copy.deepcopy(self.state())
        for v in self.birth.values():
            if isinstance(v,np.ndarray):v.flags.writeable=False
    def state(self):
        out=dict(w=self.w,p=self.p,cache=self.cache,history=self.history,updates=self.updates,
                    extra_updates=self.extra_updates,rng=self.rng.getstate())
        if self.arm in config.RESTORE_ARMS:
            out.update(world_id=self.world_id,direction_rng=self.direction_rng.getstate(),buffer=self.buffer,
                qualified=self.qualified,restorations=self.restorations,fallbacks=self.fallbacks)
        return out
    def digest(self):return state_hash(self.state())
    def clone(self):return copy.deepcopy(self)
    def restore(self,state):
        def tuples(v):return tuple(tuples(x) for x in v) if isinstance(v,(list,tuple)) else v
        if set(state)!=set(self.state()):raise ValueError('complete learned state required')
        for k,v in state.items():
            if k in ('rng','direction_rng'):getattr(self,k).setstate(tuples(v))
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
        factor=.25 if conflict else 1.
        return key,index,dict(eligible=eligible,prior_count=n,mode=mode,mode_count=count,
            conflict=bool(conflict),probability=probability,random=draw,factor=factor)
    def restore_direction(self,ids,target,gate):
        if self.arm not in config.RESTORE_ARMS:return None
        self.scalar_ops+=600;self.traffic_bytes+=2048
        qualified=bool(gate['eligible'] and not gate['conflict'])
        if qualified:
            self.qualified+=1;self.buffer=(self.buffer+[{'ids':list(ids),'target':target}])[-8:]
        event=dict(qualified_added=qualified,qualified_count=self.qualified,performed=False,
            restoration_count=self.restorations,delta=0.,trace_added=0.,direction=None,fallback=False)
        keys={tuple(row['ids'][:2]) for row in self.buffer}
        if not (qualified and self.qualified%8==0 and len(keys)>=2 and self.restorations<64):return event
        g=np.zeros(31)
        for row in self.buffer:
            e=max(-1.,min(1.,row['target']-self.value(row['ids'])))
            for i in row['ids']:g[i]+=e
        norm=math.sqrt(math.fsum(float(v*v) for v in g));fallback=norm<=1e-12
        if fallback:
            z=np.zeros(31);z[list(ids)]=1./math.sqrt(3)
        else:z=g/norm
        random_direction=np.array([1. if self.direction_rng.getrandbits(1) else -1. for _ in range(31)])/math.sqrt(31)
        chosen=z if self.arm=='SUPPORT-DIR' else random_direction
        before=self.p.diagonal().copy()
        if self.arm=='SUPPORT-ISO':self.p=self.p+(.25/31)*np.eye(31)
        else:self.p=self.p+.25*np.outer(chosen,chosen)
        self.restorations+=1;self.fallbacks+=int(fallback)
        self.scalar_ops+=3200 if self.arm=='SUPPORT-ISO' else 4000
        self.traffic_bytes+=24000 if self.arm=='SUPPORT-ISO' else 32000
        event.update(performed=True,restoration_count=self.restorations,delta=.25,
            trace_added=math.fsum(float(a-b) for a,b in zip(self.p.diagonal(),before)),
            direction=None if self.arm=='SUPPORT-ISO' else chosen.tolist(),fallback=fallback)
        return event
    def update(self,ids,target,allowed,cached_value):
        if type(allowed) is not bool or type(target) is not int or not 0<=target<=8:
            raise ValueError('observed target/permission')
        if len(ids)!=3 or any(type(i) is not int for i in ids) or not(0<=ids[0]<15 and 15<=ids[1]<30 and ids[2]==30):
            raise ValueError('ordered sensory features')
        out=dict(performed=False,replay_index=None,extra_performed=False,extra_ids=None,extra_target=None,gate=None,restoration=None)
        if not allowed:return out
        gate=dict(eligible=False,prior_count=0,mode=None,mode_count=0,conflict=False,probability=0.,random=None,factor=1.)
        if self.arm!='COV-R':key,index,gate=self.support(ids,target)
        restoration=self.restore_direction(ids,target,gate)
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
        out.update(performed=True,gate=gate,restoration=restoration);return out


class ByteLearner:
    """Only feed(byte, ticks, allowed) receives experimental input."""
    def __init__(self, arm, world_id=0):
        self.core = Core(arm,world_id)
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
