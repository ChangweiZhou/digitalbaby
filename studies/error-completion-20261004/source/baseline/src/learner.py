"""Byte/timing-only learner. No environment, world, stage, mapping or evaluator imports.

The event clock is an explicit fixed protocol: 12 cue bytes, predict, one actual
outcome byte, newline. Four outcomes are fixed for the entire life.
"""
import copy
import numpy as np
import bootstrap
import stores
from integrity import require_technical, require_science, require_supervised
import os
from common_platform import clone_model

ALPHABET=b'0123'
ARMS=('R3','R_center','R0_signed')

class Learner:
    def __init__(self, arm, scales):
        require_supervised()
        if os.environ.get('R3_OBS_MODE')=='science': require_science()
        else: require_technical()
        if arm not in ARMS: raise ValueError(arm)
        self.arm=arm
        self.scales=tuple(scales)
        self.shared=[]; self.private=[]; self.births=[]
        for name,kind in [('shared','native'),('private','content')]:
            for j in range(4):
                model,rc=stores.birth(kind)
                getattr(self,name).append(model)
                self.births.append({'bank':name,'store':j,**rc})
        if len({b['fly_id'] for b in self.births})!=8: raise AssertionError('stores must have independent births')
        self.cached=None
        self.prediction_time=None
        self.cue_count=0
        self.audit=[]
    def clone(self):
        out=copy.copy(self)
        out.shared=[clone_model(x) for x in self.shared]
        out.private=[clone_model(x) for x in self.private]
        out.cached=None if self.cached is None else self.cached.copy()
        out.audit=list(self.audit)
        return out
    def digests(self): return [m.state_digest() for m in self.shared+self.private]
    def feed(self,b,t):
        if self.cached is not None: raise ValueError('outcome must follow prediction')
        if b==10:
            self.cue_count=0
        else:
            self.cue_count+=1
        for m in self.shared+self.private: m.byte(b,t,learn=False)
    def predict(self,t):
        if self.cached is not None: raise ValueError('prediction already pending')
        if self.cue_count!=12: raise ValueError('prediction requires twelve observed cue bytes')
        before=self.digests()
        s=np.array([m.association_value(t) for m in self.shared],dtype=float)
        p=np.array([m.association_value(t) for m in self.private],dtype=float)
        u=s/self.scales[0]+p/self.scales[1]
        assert self.digests()==before and np.isfinite(u).all()
        self.cached=s.copy()
        self.prediction_time=float(t)
        return {'emitted':int(ALPHABET[int(np.argmax(u))]),'shared':s.tolist(),'private':p.tolist(),'combined':u.tolist()}
    def observe_outcome(self,byte,t):
        if self.cached is None or byte not in ALPHABET: raise ValueError('actual outcome required after prediction')
        if float(t)!=self.prediction_time: raise ValueError('outcome must occupy the scheduled post-cue slot')
        # Target is constructed only here from the received byte.
        y=np.zeros(4); y[ALPHABET.index(byte)]=1.
        v=self.cached/self.scales[0]; e=np.exp(v-v.max()); p=e/e.sum()
        if self.arm=='R3': c,s=0.,p-y
        elif self.arm=='R_center': c,s=0.,np.full(4,.25)-y
        else: c,s=1.,1.-y
        for m in self.shared+self.private: m.byte(byte,t,learn=False)
        x=self.private[0].pending_x
        if x is None or any(not np.array_equal(m.pending_x,x) for m in self.private[1:]): raise AssertionError('private pre-outcome codes differ')
        shared=[];private=[]
        for j,m in enumerate(self.shared): shared.append(m.teach_logged(int(ALPHABET[j]!=byte),t,write=True))
        for j,m in enumerate(self.private): private.append(m.teach_signed(c,float(s[j]),t,write=True))
        self.audit.append({'outcome':byte,'prediction_time':self.prediction_time,'outcome_time':float(t),'cached_shared':self.cached.tolist(),'p':p.tolist(),'c':c,'s':s.tolist(),'shared_l1':shared,'private_l1':private,'shared_state':[m.state_digest() for m in self.shared]})
        self.cached=None
        self.prediction_time=None
    def flush(self,t):
        if self.cached is not None: raise ValueError('cannot flush pending outcome')
        return max(m.flush(t) for m in self.shared+self.private)
