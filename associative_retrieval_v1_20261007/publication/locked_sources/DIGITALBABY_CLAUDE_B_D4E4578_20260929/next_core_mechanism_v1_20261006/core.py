"""Byte-only core; no task, world, stage or target-bearing evaluator imports."""
import copy
import time
from dataclasses import dataclass
import numpy as np
import bootstrap
from centered_core import CenteredCore
import stores
from common_platform import clone_model
from mechanisms import convert,projections
from spec import ALPHABET,SCALES,PHYSICAL_ARMS,DOSES,EPSILONS
from io_utils import digest

@dataclass(frozen=True)
class Prediction:
    emitted:int
    shared:tuple
    private:tuple
    combined:tuple
    extra:tuple=()

def pi_of(v):
    v=np.asarray(v)/SCALES[1];e=np.exp(v-v.max());return e/e.sum()

class Core(CenteredCore):
    def __init__(self,arm):
        if arm not in PHYSICAL_ARMS:raise ValueError('unknown physical arm')
        self.extra=[]
        super().__init__();self.arm=arm;self.private_prediction=None;self.yoked_counts=None
        if arm in EPSILONS or arm in ('S3_CUE','S3_RAND'):
            self.shared=[convert(m,arm,j) for j,m in enumerate(self.shared)]
        if arm in DOSES:
            for j in range(4):
                m,rc=stores.birth('native');self.extra.append(convert(m,arm,j))
                self.births.append({'bank':'extra','store':j,**rc})
        if len({id(m.fly) for m in self.models})!=len(self.models):raise AssertionError('aliased birth')

    @property
    def models(self):return self.shared+self.private+self.extra

    def clone(self):
        out=super().clone();out.extra=[clone_model(m) for m in self.extra]
        out.private_prediction=None if self.private_prediction is None else self.private_prediction.copy()
        out.yoked_counts=copy.deepcopy(self.yoked_counts)
        for m in out.models:
            for n in ('teach_logged','teach_signed'):m.__dict__.pop(n,None)
        return out

    def predict(self,t):
        t=self._time(t)
        if self.cached is not None or self.awaiting_newline or self.cue_count!=12:raise ValueError('prediction boundary')
        before=self.bank_digests()
        s=np.array([m.association_value(t) for m in self.shared]);p=np.array([m.association_value(t) for m in self.private])
        r=np.array([m.association_value(t) for m in self.extra])
        u=s/SCALES[0]+p/SCALES[1]
        if len(r):u+=r/SCALES[0]
        if before!=self.bank_digests() or not np.isfinite(u).all():raise AssertionError('read mutated bank')
        for m in self.shared:
            if hasattr(m,'cache_cue'):m.cache_cue(t)
        self.cached=s.copy();self.private_prediction=pi_of(p)
        self.prediction_time=self.last_time=t
        return Prediction(int(ALPHABET[int(np.argmax(u))]),tuple(s),tuple(p),tuple(u),tuple(r))

    def observe_outcome(self,byte,t,*,learn=True):
        byte,t=self._byte(byte),self._time(t)
        if type(learn) is not bool:raise ValueError('learn must be bool')
        if self.cached is None or self.private_prediction is None or byte not in ALPHABET or t!=self.prediction_time:raise ValueError('outcome boundary')
        y=np.array([float(b==byte) for b in ALPHABET])
        pi=self.private_prediction.copy();signs=(np.full(4,.25)-y if self.arm=='CENTER' else pi-y)
        rel_cache=[m.rel_features(t) for m in self.extra]
        for m in self.models:m.byte(byte,t,learn=False)
        x=self.private[0].pending_x
        if x is None or any(not np.array_equal(m.pending_x,x) for m in self.private[1:]):raise AssertionError('private addresses differ')
        for m,(h,q,xr) in zip(self.extra,rel_cache):
            if not np.array_equal(m.last_h,h) or not np.array_equal(m.last_q,q) or not np.array_equal(m.pending_x,xr):raise AssertionError('extra read/write address mismatch')
            if self.arm=='REL_PERM10':
                perm=projections()[2];written=np.empty_like(q);written[perm]=q
                m.pending_x=np.asarray(m.model.encode_sparse(m.fly.m,written),dtype=float)
        sh=[m.teach_logged(int(b!=byte),t,write=learn) for b,m in zip(ALPHABET,self.shared)]
        pr=[m.teach_signed(0.,float(s),t,write=learn) for s,m in zip(signs,self.private)]
        ex=[m.teach_logged(int(b!=byte),t,write=learn) for b,m in zip(ALPHABET,self.extra)]
        adaptations=[]
        for j,m in enumerate(self.shared):
            if hasattr(m,'adapt_cue'):
                n=None
                if self.arm=='S3_RAND':
                    if self.yoked_counts is None or self.records>=len(self.yoked_counts):raise ValueError('missing diagnostic schedule')
                    n=self.yoked_counts[self.records][j]
                adaptations.append(m.adapt_cue(t,n))
        if not learn and any(sh+pr+ex):raise AssertionError('disabled supervisory write')
        self.last_write={'outcome':byte,'time':t,'learn':learn,'c':0.,'s':signs.tolist(),
                         'private_pre_outcome_probabilities':pi.tolist(),'shared_l1':sh,'private_l1':pr,
                         'extra_l1':ex,'adaptations':adaptations}
        self.cached=self.private_prediction=self.prediction_time=None
        self.awaiting_newline=True;self.last_time=t;self.records+=1
        return copy.deepcopy(self.last_write)

    def clear_unreinforced_prediction(self):
        self.cached=self.private_prediction=self.prediction_time=None;self.cue_count=0

    def state_digest(self):
        return digest([super().state_digest(),self.arm,
                       None if self.private_prediction is None else self.private_prediction.tolist(),self.yoked_counts])

    def base_state_digest(self):
        # Source parent parity excludes wrapper metadata but includes all 8 operative stores.
        return digest([m.state_digest() for m in self.shared+self.private])

def policies(pred):
    s=np.asarray(pred.shared)/SCALES[0];p=np.asarray(pred.private)/SCALES[1]
    r=np.asarray(pred.extra)/SCALES[0] if pred.extra else np.zeros(4)
    return {name:{'combined':u.tolist(),'emitted':int(ALPHABET[int(np.argmax(u))])}
            for name,u in [('native',s+p+r),('Q_HALF',s+.5*p+r)]}

class ChoiceOrgan:
    def __init__(self,core):self.core=core.clone()
    def choose(self,first,second,at,dt):
        ps=[];vals=[];half=[];self.half_cpu_s=0.
        for k,cue in enumerate((first,second)):
            start=at+13*k*dt
            for i,b in enumerate(cue):self.core.feed(b,start+i*dt)
            p=self.core.predict(start+12*dt);ps.append(p);vals.append(p.combined[1]-p.combined[0])
            started=time.process_time();q=policies(p)['Q_HALF']['combined'];half.append(q[1]-q[0]);self.half_cpu_s+=time.process_time()-started
            self.core.clear_unreinforced_prediction()
            self.core.feed(ord('|') if k==0 else ord('?'),start+12*dt);self.core.cue_count=0
        return (ord('L') if vals[0]>=vals[1] else ord('R')),vals,ps,(ord('L') if half[0]>=half[1] else ord('R'))
