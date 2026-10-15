"""Explicitly separate retained native and recurrent models. No task imports."""
import copy
import hashlib
import math
import time
from dataclasses import dataclass
import bootstrap
import numpy as np
import torch
import stores
from centered_core import CenteredCore
from centered_core.core import SCALES,ALPHABET

PARENT=bootstrap.source_module('sequence_port_parent','addition_core_lineage_comparison_dev_v1_20261015/core.py')
CODE, MU, EOS=PARENT.CODE,PARENT.MU,PARENT.EOS


@dataclass(frozen=True)
class ScheduledPrediction:
    emitted:int
    shared:tuple
    private:tuple
    combined:tuple
    logits:tuple


class ScheduledC16(CenteredCore):
    """Only change the number/teaching/readout of channels; keep legacy framing."""
    variant='R_CENTER_SCHEDULED_C16_V1'

    def __init__(self):
        self.shared=[];self.private=[];self.births=[]
        for bank,kind in [('shared','native'),('private','content')]:
            for j in range(16):
                model,receipt=stores.birth(kind)
                getattr(self,bank).append(model)
                self.births.append(dict(bank=bank,store=j,**receipt))
        if len({id(m.fly) for m in self.models})!=32:raise AssertionError('separate canonical births required')
        self.cached=self.prediction_time=None
        self.cue_count=0;self.awaiting_newline=False;self.last_time=0.;self.records=0;self.last_write=None

    def state_digest(self):
        return hashlib.sha256((self.variant+super().state_digest()).encode()).hexdigest()

    def predict(self,t):
        t=self._time(t)
        if self.cached is not None or self.awaiting_newline or self.cue_count!=12:
            raise ValueError('unchanged twelve-byte scheduled protocol required')
        before=self.bank_digests()
        shared=np.array([m.association_value(t) for m in self.shared])
        private=np.array([m.association_value(t) for m in self.private])
        values=shared/SCALES[0]+private/SCALES[1]
        scores=PARENT.logits(values)
        if before!=self.bank_digests() or not np.isfinite(scores).all():
            raise AssertionError('readout mutated native memories or became nonfinite')
        self.cached=shared.copy();self.prediction_time=self.last_time=t
        return ScheduledPrediction(int(scores.argmax()),tuple(shared),tuple(private),tuple(values),tuple(scores))

    def observe_outcome(self,byte,t,*,learn=True):
        byte=self._byte(byte);t=self._time(t)
        if type(learn) is not bool:raise ValueError('explicit learning permission')
        if self.cached is None or t!=self.prediction_time:
            raise ValueError('scheduled actual target must follow prediction')
        for m in self.models:m.byte(byte,t,learn=False)
        shared=[m.teach_logged(int((1-CODE[byte,j])/2),t,write=learn) for j,m in enumerate(self.shared)]
        private=[m.teach_signed(0.,float((MU[j]-CODE[byte,j])/2),t,write=learn) for j,m in enumerate(self.private)]
        if not learn and any(x!=0. for x in shared+private):raise AssertionError('forbidden write')
        self.last_write=dict(outcome=byte,time=t,learn=learn,shared_l1=shared,private_l1=private)
        self.cached=self.prediction_time=None;self.awaiting_newline=True
        self.last_time=t;self.records+=1
        return copy.deepcopy(self.last_write)


class SequenceC16(PARENT.Native):
    """Retained RC151-C16 equations, now actually predict before each target."""
    variant='R_CENTER_SEQUENCE_C16_CLOSED_LOOP_V1'

    def __init__(self):
        super().__init__('RC151-C16')
        self.resource.update(prefeedback_predictions=0,prefeedback_readout_cpu=0.,
                             forbidden_write_audit_cpu=0.,idle_cpu=0.,idle_seconds=0.)

    def forecast(self):
        # No target/answer/evaluator argument is available to this method.
        values=self.values();scores=PARENT.logits(values)
        if not np.isfinite(scores).all():raise FloatingPointError('nonfinite native output')
        return int(scores.argmax()),scores

    def train(self,prompt,answer,allowed=True):
        if type(prompt) is not bytes or type(answer) is not bytes or len(answer)>16 or type(allowed) is not bool:
            raise ValueError('bounded observed bytes and explicit permission')
        audit_start=time.process_time()
        pre=[(s.fly.m.fast.copy(),s.fly.m.slow.copy()) for b in self.banks for s in b] if not allowed else None
        audit_cpu=time.process_time()-audit_start
        start=time.process_time();old_time=self.time
        self._prompt(prompt);applied=0.;trace=[]
        for target in (*answer,EOS):
            read_start=time.process_time()
            predicted,scores=self.forecast()
            self.resource['prefeedback_readout_cpu']+=time.process_time()-read_start
            self.resource['prefeedback_predictions']+=1
            # Only now is the arriving observation consumed by teaching.
            applied+=self._event(target,target,allowed)
            trace.append(dict(predicted=predicted,target=target,logits=scores.tolist()))
        self.resource['production_cpu']+=time.process_time()-start
        audit_start=time.process_time();off_error=0.
        if not allowed:
            elapsed=self.time-old_time
            for (fast,slow),s in zip(pre,[s for b in self.banks for s in b]):
                m=s.fly.m
                factors=np.array([math.exp(-elapsed/m.tg),math.exp(-elapsed/m.kernel['fast_tau'])])
                off_error=max(off_error,float(np.abs(m.fast-fast*factors).max()),
                              float(np.abs(m.slow-slow*math.exp(-elapsed/m.kernel['slow_tau'])).max()))
            if off_error>1e-9:raise AssertionError(('forbidden native plasticity',off_error))
        self.resource['forbidden_write_audit_cpu']+=audit_cpu+time.process_time()-audit_start
        self.resource['records']+=1;self.resource['answer_bytes']+=len(answer)
        self.resource['EOS_targets']+=1;self.resource['input_bytes']+=len(prompt)
        return dict(allowed=allowed,raw_alpha_l1=applied,forbidden_delta=off_error,trace=trace)

    def rest(self,seconds):
        start=time.process_time();super().rest(seconds)
        self.resource['idle_cpu']+=time.process_time()-start;self.resource['idle_seconds']+=seconds

    def generate_trace(self,prompt,cap=16):
        if type(cap) is not int or not 0<=cap<=16:raise ValueError('bounded payload cap')
        view=self.clone();view._prompt(prompt);out=bytearray();trace=[]
        for _ in range(cap+1):
            predicted,scores=view.forecast()
            trace.append(dict(prefix=bytes(out).hex(),predicted=predicted,logits=scores.tolist()))
            if predicted==EOS:return bytes(out),'EOS',trace
            if len(out)==cap:return bytes(out),'TRUNCATED',trace
            out.append(predicted);view._event(predicted)
        raise AssertionError('unreachable generation state')

    def teacher_conditioned_trace(self,prompt,answer):
        """Diagnostic conditional margins; never graded as autonomous answers."""
        view=self.clone();view._prompt(prompt);rows=[]
        for target in (*answer,EOS):
            predicted,scores=view.forecast()
            rows.append(dict(predicted=predicted,target=target,logits=scores.tolist()))
            view._event(target)
        return rows


class AdditionReference(PARENT.Recurrent):
    """Unchanged KEEP/G62 C16 CE model, explicitly independent of Full151."""
    def __init__(self,version,seed):
        if version not in ('KEEP-C16-CE','G62-C16-CE'):raise ValueError('fixed two-reference roster')
        super().__init__(version,seed)
        self.rest_cpu=0.;self.rest_seconds=0.

    def rest(self,seconds):
        start=time.process_time();super().rest(seconds)
        self.rest_cpu+=time.process_time()-start;self.rest_seconds+=seconds


def margin(scores,target):
    scores=np.asarray(scores,float)
    if not 0<=target<len(scores):raise ValueError('target outside actual output universe')
    others=np.delete(scores,target)
    return float(scores[target]-others.max())
