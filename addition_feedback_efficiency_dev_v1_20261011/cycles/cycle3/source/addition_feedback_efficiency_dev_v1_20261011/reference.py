"""Independent transition reconstruction. Does not import production cores."""
import copy
import hashlib
import json
import math
import random
import numpy as np
import config

def checksum(s):
    h=hashlib.sha256()
    for k,v in sorted(s.items()):
        h.update(k.encode())
        if isinstance(v,np.ndarray):
            h.update(str((v.shape,v.dtype.str)).encode());a=np.array(v,copy=True);a[a==0.]=0.
            h.update(a.tobytes())
        else:h.update(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    return h.hexdigest()
def answer(v):
    values=[(k-v)*(k-v) for k in range(9)]
    return values.index(min(values))

class Reference:
    def __init__(self,arm):
        if arm not in config.ARMS:raise ValueError('arm')
        self.arm=arm
        self.a={'w':np.zeros(31),'p':np.full(31,10.) if arm[:4]=='DIAG' else np.diag([10.]*31)}
        self.cache=[];self.rng=random.Random(20261011047);self.updates=self.extra_updates=0
    def state(self):
        return dict(self.a,cache=self.cache,rng=self.rng.getstate(),updates=self.updates,extra_updates=self.extra_updates)
    def clone(self):return copy.deepcopy(self)
    def digest(self):return checksum(self.state())
    def predict(self,ids):return math.fsum(float(self.a['w'][i]) for i in ids)
    def adjust(self,ids,target,prediction,update_information):
        residual=target-prediction
        if self.arm[:4]=='DIAG':
            q=np.zeros(31)
            for i in ids:q[i]=self.a['p'][i]
        else:
            col=[self.a['p'][:,i] for i in ids];q=col[0]+col[1]+col[2]
        uncertainty=math.fsum(float(q[i]) for i in ids)
        if uncertainty<0 or not math.isfinite(uncertainty):raise AssertionError('confidence')
        cap=math.sqrt(uncertainty+1.) if self.arm[-2:]=='-I' else 1.
        robust=1. if abs(residual)<=cap else cap/abs(residual)
        denominator=1.+robust*uncertainty
        delta=(robust*residual/denominator)*q
        self.a['w']=np.minimum(4*config.KAPPA,np.maximum(-4*config.KAPPA,self.a['w']+delta))
        if update_information:
            scale=robust/denominator
            if self.arm[:4]=='DIAG':self.a['p']=self.a['p']-scale*np.square(q)
            else:
                lowered=self.a['p']-scale*(q[:,None]*q[None,:])
                self.a['p']=(lowered+lowered.transpose())/2.
    def step(self,ids,target,allowed,predicted):
        if not allowed:return {'performed':False,'replay_index':None,'extra_performed':False,'extra_ids':None,'extra_target':None}
        self.adjust(ids,target,predicted,True);self.updates+=1
        selected=None;used=None
        if self.arm in ('COV-REPLAY','COV-EXTRA'):
            if len(self.cache)>0:
                selected=self.rng.randrange(len(self.cache))
                used=copy.deepcopy(self.cache[selected]) if self.arm=='COV-REPLAY' else {'ids':list(ids),'target':target}
                self.adjust(used['ids'],used['target'],self.predict(used['ids']),False)
                self.extra_updates+=1
            self.cache=self.cache+[{'ids':list(ids),'target':target}]
            if len(self.cache)>32:self.cache=self.cache[1:]
        return {'performed':True,'replay_index':selected,'extra_performed':used is not None,
                'extra_ids':used['ids'] if used else None,'extra_target':used['target'] if used else None}
    def require_seal(self,actual):
        if actual!=self.digest():raise AssertionError('independent full learned-state transition seal mismatch')
    def invariants(self):
        for a in self.a.values():
            if not np.isfinite(a).all():raise AssertionError('nonfinite state')
        if self.arm[:4]=='DIAG':
            if np.min(self.a['p'])<=0:raise AssertionError('positive diagonal')
        else:
            if not np.array_equal(self.a['p'],self.a['p'].T):raise AssertionError('symmetry')
            np.linalg.cholesky(self.a['p'])
        if len(self.cache)>32 or self.extra_updates>self.updates:raise AssertionError('bounded reuse')
