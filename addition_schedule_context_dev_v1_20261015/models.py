"""Operative byte models, without evaluator/task/operand imports."""
import copy
import hashlib
import json
import math
import time
from collections import OrderedDict
import bootstrap
import numpy as np
import torch
import stores

PREVIOUS=bootstrap.source_module('previous_interface','native_interface_transfer_dev_v1_20261015/models.py')
PARENT=PREVIOUS.PARENT
CODE,MU,EOS=PARENT.CODE,PARENT.MU,PARENT.EOS
GROUPS={'E':('E',),'R':('Wz','Uz','bz','Wr','Ur','br','Wg','Ug','bg'),'C':('C',),'V':('V',)}


class Keep(PREVIOUS.AdditionReference):
    """Exact inherited CE/RMS update with optional external proposal capture."""
    def __init__(self,seed):
        super().__init__('KEEP-C16-CE',seed)
        self.last_capture=None

    def state(self):
        return dict(p={k:v.detach().numpy().copy() for k,v in self.p.items()},
             mom={k:v.detach().numpy().copy() for k,v in self.mom.items()},
             actor_steps=self.actor_steps,critic_steps=self.critic_steps,last=copy.deepcopy(self.last),
             resource=copy.deepcopy(self.resource),rest_cpu=self.rest_cpu,rest_seconds=self.rest_seconds)

    def set_state(self,state):
        if set(state['p'])!=set(self.p) or set(state['mom'])!=set(self.mom):raise ValueError('parameter schema mismatch')
        with torch.no_grad():
            for k,p in self.p.items():p.copy_(torch.as_tensor(state['p'][k]))
            for k,m in self.mom.items():m.copy_(torch.as_tensor(state['mom'][k]))
        for key in ('actor_steps','critic_steps','last','resource','rest_cpu','rest_seconds'):
            setattr(self,key,copy.deepcopy(state[key]))
        self.last_capture=None

    def train(self,prompt,answer,allowed=True,*,capture=False):
        if type(prompt) is not bytes or type(answer) is not bytes or len(answer)>16 or type(allowed) is not bool:
            raise ValueError('raw byte feedback and explicit permission')
        if capture and not allowed:raise ValueError('capture an actual permitted update only')
        guard_start=time.process_time();pre=self.state() if capture else None
        guard_cpu=time.process_time()-guard_start
        self.last_capture=None
        start=time.process_time()
        self.resource['records']+=1;self.resource['observed_answer_bytes']+=len(answer)
        self.resource['observed_EOS_targets']+=1
        if not allowed:
            with torch.no_grad():loss=float(self.ce_loss(prompt,answer))
            self.resource['production_cpu']+=time.process_time()-start
            return dict(allowed=False,actor_steps=0,loss=loss,forbidden_delta=0.,raw_alpha_l1=0.)
        loss=self.ce_loss(prompt,answer)
        gradients=torch.autograd.grad(loss,list(self.p.values()))
        norm=math.sqrt(sum(float((g*g).sum()) for g in gradients))
        if not math.isfinite(norm):raise FloatingPointError('nonfinite actor gradient')
        clip=min(1.,1/max(norm,1e-30));self.actor_steps+=1
        with torch.no_grad():
            for (k,p),gradient in zip(self.p.items(),gradients):
                g=gradient*clip
                self.mom[k].mul_(.99).add_(g*g,alpha=.01)
                p.sub_(.001*g/(torch.sqrt(self.mom[k]/(1-.99**self.actor_steps))+1e-8))
        if not all(torch.isfinite(p).all() for p in self.p.values()):raise FloatingPointError('nonfinite actor state')
        self.resource['actor_steps']+=1;self.resource['production_cpu']+=time.process_time()-start
        record=dict(allowed=True,actor_steps=1,loss=float(loss.detach()),gradient_norm=norm,forbidden_delta=0.,raw_alpha_l1=0.)
        if capture:
            at=time.process_time();post=self.state();gs={k:g.detach().numpy().copy() for k,g in zip(self.p,gradients)}
            delta={k:post['p'][k]-pre['p'][k] for k in self.p}
            metrics={}
            for name,keys in GROUPS.items():
                displacement=np.concatenate([delta[k].ravel() for k in keys])
                raw=np.concatenate([gs[k].ravel() for k in keys])
                denominator=np.concatenate([(np.sqrt(post['mom'][k]/(1-.99**self.actor_steps))+1e-8).ravel() for k in keys])
                pnorm=math.sqrt(sum(float((pre['p'][k]**2).sum()) for k in keys))
                dnorm=float(np.linalg.norm(displacement))
                metrics[name]=dict(raw_gradient_l2=float(np.linalg.norm(raw)),clipped_gradient_l2=float(np.linalg.norm(raw))*clip,
                    delta_l2=dnorm,delta_abs_max=float(np.abs(displacement).max()),parameter_l2=pnorm,
                    relative_delta_l2=None if pnorm==0 else dnorm/pnorm,
                    RMS_denominator_quantiles=np.quantile(denominator,[0,.5,1]).tolist())
            self.last_capture=dict(pre=pre,post=post,delta=delta,raw_gradients=gs,clip=clip,metrics=metrics)
            record['capture_cpu']=guard_cpu+time.process_time()-at
        return record


class Front(stores.content_model.VisibleContextFE):
    """Only the private receptor sketch changes. Shared FE0 remains unmodified."""
    def __init__(self,variant):
        super().__init__();self.variant=variant;self.q=b'';self.prefix=b'';self.frozen=False
        self.cache=OrderedDict();self.cache_hits=self.cache_misses=0

    def freeze_prompt(self):self.frozen=True

    def feed(self,b,t):
        super().feed(b,t)
        if self.frozen:
            if len(self.prefix)>=16:raise ValueError('prefix cap')
            self.prefix+=bytes([b])
        else:
            if len(self.q)>=256:raise ValueError('prompt cap')
            self.q+=bytes([b])

    def read(self):
        if self.variant=='ROLLING-C16' or not self.prefix:
            if len(self.visible)<4:return stores.bb.bc.FE0.read(self)
            key=('rolling',self.visible)
            def make():return stores.bb.bc._norm(stores.content_model.context_receptors.__wrapped__(self.visible))
        else:
            key=('bound',self.q,self.prefix)
            def make():
                material=b'RCPROMPTPREFIX-v1'+len(self.q).to_bytes(2,'little')+self.q+len(self.prefix).to_bytes(1,'little')+self.prefix
                rng=np.random.default_rng(int.from_bytes(hashlib.sha256(material).digest()[:8],'little'))
                z=np.zeros(88,dtype=np.float64);indices=rng.choice(88,24,replace=False)
                z[indices]=rng.uniform(.4,1.,size=24)
                return stores.bb.bc._norm(z)
        if key in self.cache:
            self.cache_hits+=1;self.cache.move_to_end(key);return self.cache[key]
        self.cache_misses+=1;out=make();out.flags.writeable=False;self.cache[key]=out
        if len(self.cache)>65536:self.cache.popitem(last=False)
        return out

    def clone(self):
        out=super().clone();out.cache=OrderedDict();out.cache_hits=out.cache_misses=0
        return out


class Native(PREVIOUS.SequenceC16):
    def __init__(self,variant):
        if variant not in ('ROLLING-C16','BOUND-C16'):raise ValueError('fixed native variants')
        self.context_variant=variant
        super().__init__()

    def _event(self,token=None,target=None,allowed=False):
        if token==PARENT.REPLY:self.fronts[1].freeze_prompt()
        applied=super()._event(token,target,allowed)
        if token==PARENT.BEGIN:
            self.fronts[1]=Front(self.context_variant);self.fronts[1].t=self.time
        return applied

    def digest(self):
        h=hashlib.sha256((super().digest()+self.context_variant).encode())
        for fe in self.fronts:
            h.update(fe.p.tobytes());h.update(repr(fe.t).encode())
            if isinstance(fe,Front):h.update(repr((fe.visible,fe.q,fe.prefix,fe.frozen)).encode())
        for bank in self.banks:
            for s in bank:
                h.update(repr((s.brain_t,s.pending_t,s.teach_seen,s.fly.m.elapsed)).encode())
                if s.pending_x is not None:h.update(s.pending_x.tobytes())
        return h.hexdigest()
