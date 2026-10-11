"""Independent full transition reconstruction; no production core import."""
import copy, hashlib, json, math, random
from collections import Counter
import numpy as np
import config

def checksum(s):
    h=hashlib.sha256()
    for k,v in sorted(s.items()):
        h.update(k.encode())
        if isinstance(v,np.ndarray):
            h.update(str((v.shape,v.dtype.str)).encode());a=np.array(v,copy=True);a[a==0.]=0.;h.update(a.tobytes())
        else:h.update(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    return h.hexdigest()
def answer(v):
    values=[(k-v)*(k-v) for k in range(9)]
    return values.index(min(values))
class Reference:
    def __init__(self,arm):
        if arm not in config.ARMS:raise ValueError('arm')
        self.arm=arm;self.a={'w':np.zeros(31),'p':np.diag([10.]*31)}
        self.cache=[];self.history=[];self.rng=random.Random(20261011047)
        self.updates=self.extra_updates=0
    def state(self):
        return dict(self.a,cache=self.cache,history=self.history,rng=self.rng.getstate(),updates=self.updates,extra_updates=self.extra_updates)
    def clone(self):return copy.deepcopy(self)
    def digest(self):return checksum(self.state())
    def predict(self,ids):return math.fsum(float(self.a['w'][i]) for i in ids)
    def step(self,ids,target,allowed,predicted):
        result={'performed':False,'replay_index':None,'extra_performed':False,'extra_ids':None,'extra_target':None,'gate':None}
        if not allowed:return result
        gate={'eligible':False,'prior_count':0,'mode':None,'mode_count':0,'conflict':False,'probability':0.,'random':None,'factor':1.}
        if self.arm!='COV-R':
            alphabet=b'01234ABCDEFGHIJ';key=[alphabet[ids[0]],alphabet[ids[1]-15]]
            found=None
            for ix,entry in enumerate(self.cache):
                if entry['key']==key:found=ix;break
            observations=[] if found is None else list(self.cache[found]['answers'])
            tally=Counter(observations);n=len(observations)
            highest=max(tally.values(),default=0)
            winners=[y for y,count in tally.items() if count>=2 and count*2>n]
            mode=winners[0] if winners else None
            eligible=mode is not None;conflict=bool(eligible and target!=mode)
            frequency=math.fsum(self.history)/len(self.history) if self.history else 0.
            uniform=self.rng.random()
            gate.update(eligible=eligible,prior_count=n,mode=mode,mode_count=highest,
                        conflict=conflict,probability=frequency,random=uniform)
            if self.arm=='COV-SUPPORT':reduce=conflict
            else:reduce=bool(eligible and uniform<frequency)
            gate['factor']=.25 if reduce else 1.
        residual=target-predicted
        col=[self.a['p'][:,i] for i in ids];q=col[0]+col[1]+col[2]
        uncertainty=math.fsum(float(q[i]) for i in ids)
        if uncertainty<0 or not math.isfinite(uncertainty):raise AssertionError('confidence')
        robust=1. if abs(residual)<=1. else 1./abs(residual)
        weighted=gate['factor']*robust;denominator=1.+weighted*uncertainty
        self.a['w']=np.minimum(4*config.KAPPA,np.maximum(-4*config.KAPPA,
            self.a['w']+(weighted*residual/denominator)*q))
        lowered=self.a['p']-(weighted/denominator)*(q[:,None]*q[None,:])
        self.a['p']=(lowered+lowered.transpose())/2.
        self.updates+=1
        if self.arm!='COV-R':
            if gate['eligible']:
                self.history.append(int(gate['conflict']))
                if len(self.history)>32:del self.history[0]
            if found is None:
                self.cache.append({'key':key,'answers':[target]})
                if len(self.cache)>32:del self.cache[0]
            else:
                self.cache[found]['answers'].append(target)
                if len(self.cache[found]['answers'])>8:del self.cache[found]['answers'][0]
        gate.update(gamma=weighted,gamma_r=robust)
        result.update(performed=True,gate=gate);return result
    def require_seal(self,actual):
        if actual!=self.digest():raise AssertionError('independent full learned-state transition seal mismatch')
    def invariants(self):
        for a in self.a.values():
            if not np.isfinite(a).all():raise AssertionError('nonfinite state')
        if not np.array_equal(self.a['p'],self.a['p'].T):raise AssertionError('symmetry')
        np.linalg.cholesky(self.a['p'])
        if len(self.cache)>32 or len(self.history)>32 or self.extra_updates!=0:raise AssertionError('bounded auxiliary state')
        if len({tuple(x['key']) for x in self.cache})!=len(self.cache):raise AssertionError('unique keys')
        for x in self.cache:
            if not 1<=len(x['answers'])<=8 or any(type(y) is not int or not 0<=y<=8 for y in x['answers']):raise AssertionError('observed answer window')
        if any(type(y) is not int or y not in (0,1) for y in self.history):raise AssertionError('conflict flags')
