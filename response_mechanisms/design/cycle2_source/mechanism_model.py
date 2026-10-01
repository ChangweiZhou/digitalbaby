"""Source-matched Full151 response mechanisms. Evaluator labels never enter sensory updates."""
from __future__ import annotations
import copy
import hashlib
import math
import sys
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
A3 = ROOT.parent / 'minifly_a_v3'
sys.path.insert(0, str(A3 / 'src'))
import stores
import brain_byte as bb
from common_platform import clone_model

ARMS = ('FE0', 'T_OFF', 'T', 'H', 'J_ADD', 'J', 'J_SHUFFLE')
DEFAULT = dict(timing_lr=.05, timing_tau=10., homeo_beta=1/16,
               eligibility_tau=10., recent_tau=10., j_lr=.25, j_gain=1.,
               j_bound=16., j_tau=86400., revision=2)


def digest(*arrays, meta=()):
    h = hashlib.sha256(repr(meta).encode())
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(repr((a.shape, a.dtype.str)).encode()); h.update(a.tobytes())
    return h.hexdigest()


class System:
    """Four native output compartments and strictly cue-only added plasticity.

    Allocation is identical in all arms. Dormant states are counted as dormant,
    not claimed as equal effective capacity. The added synapses are a declared
    architecture intervention, with J_ADD and J_SHUFFLE matched active controls.
    """
    def __init__(self, arm, params=None):
        if arm not in ARMS: raise ValueError(arm)
        self.arm = arm
        self.p = {**DEFAULT, **(params or {})}
        pairs = [stores.birth('native') for _ in range(4)]
        self.stores = [x[0] for x in pairs]
        self.births = [{k:v for k,v in x[1].items() if k != 'fly_id'} for x in pairs]
        B = self.stores[0].fly.m.B
        self.rows = np.repeat(np.arange(B.shape[0]), np.diff(B.indptr))
        self.cols = B.indices.copy()
        self.budget = np.bincount(self.cols, weights=B.data, minlength=B.shape[1])
        deg = np.bincount(self.cols, minlength=B.shape[1])
        self.mean = np.divide(self.budget, deg, out=np.ones_like(self.budget), where=deg>0)
        self.tpre = np.zeros(B.shape[0]); self.tpost = np.zeros(B.shape[1])
        self.tlast = None
        self.telig = np.zeros(len(self.cols))
        self.h = np.zeros(4)
        self.jw = np.zeros((4, 88*88))
        self.jlast = 0.
        self.recent = np.zeros(88)
        self.elig = np.zeros((88,88))
        self.sens = bb.bc.FE0()
        self.cue_last = None
        self.pending_timing = None
        self.events = []

    def clone(self):
        out = copy.copy(self)
        out.stores = [clone_model(s) for s in self.stores]
        for name in ('tpre','tpost','telig','h','jw','recent','elig'):
            setattr(out, name, getattr(self,name).copy())
        out.sens = self.sens.clone()
        out.pending_timing = None if self.pending_timing is None else self.pending_timing.copy()
        out.events = []
        return out

    def state_digest(self):
        return digest(self.tpre,self.tpost,self.telig,self.h,self.jw,self.recent,self.elig,self.sens.p,
                      *[s.fly.m.B.data for s in self.stores],
                      meta=(tuple(s.state_digest() for s in self.stores), self.arm, self.p,
                            self.tlast,self.jlast,self.cue_last,self.sens.t,
                            None if self.pending_timing is None else digest(self.pending_timing)))

    def sensory_digest(self):
        return digest(self.tpre,self.tpost,self.telig,self.recent,self.elig,self.sens.p,
                      self.stores[0].fly.m.B.data,
                      meta=(self.tlast,self.cue_last,self.sens.t))

    def begin_cue(self, at):
        # Explicit record boundary is the same timing signal for every arm.
        # Sensor reset excludes previous teachers, never receives task/answer IDs.
        self.recent.fill(0); self.elig.fill(0)
        self.tpre.fill(0); self.tpost.fill(0); self.telig.fill(0); self.tlast=None
        self.sens = bb.bc.FE0(); self.sens.t=float(at)
        self.cue_last = None

    def cue_byte(self, b, t):
        dt = 0. if self.cue_last is None else float(t-self.cue_last)
        self.recent *= math.exp(-dt/self.p['recent_tau'])
        self.elig *= math.exp(-dt/self.p['eligibility_tau'])
        current = bb.bc.R_TABLE[b]
        if self.arm == 'J_ADD':
            # Linear additive control, no cue-dependent normalization.
            self.elig += .5*(self.recent[:,None]+current[None,:])
        else:
            self.elig += self.recent[:,None]*current[None,:]
        self.recent += current
        self.sens.feed(b,t); self.cue_last=float(t)
        if self.arm in ('T','T_OFF'):
            decay=0. if self.tlast is None else math.exp(-(t-self.tlast)/self.p['timing_tau'])
            self.tpre*=decay;self.tpost*=decay
            sensory=self.sens.read()
            a=sensory[self.stores[0].fly.m.pn_type_index]
            x=self.stores[0].model.encode_sparse(self.stores[0].fly.m,sensory)
            self.telig+=self.p['timing_lr']*(self.tpre[self.rows]*x[self.cols]-a[self.rows]*self.tpost[self.cols])/12.
            self.tpre+=a;self.tpost+=x;self.tlast=float(t)
        for s in self.stores: s.byte(b,t,learn=False)

    def features(self, t):
        dt=0. if self.cue_last is None else t-self.cue_last
        return (self.elig*math.exp(-dt/self.p['eligibility_tau'])).ravel().copy()

    def raw_values(self,t):
        return np.array([s.association_value(t) for s in self.stores])

    def read(self,t):
        raw=self.raw_values(t)
        phi=self.features(t)
        j_decay=math.exp(-max(0.,t-self.jlast)/self.p['j_tau'])
        extra=self.p['j_gain']*(self.jw@phi)*j_decay
        out=raw.copy()
        if self.arm == 'H': out-=self.h
        elif self.arm.startswith('J'): out+=extra
        if not np.isfinite(out).all(): raise FloatingPointError('nonfinite output')
        return out,raw,phi,extra

    def observe_cue(self,t,raw):
        # No teacher argument. No mutable supervised state contributes to T code.
        B=self.stores[0].fly.m.B
        dv=self.telig
        dw=0.
        if self.arm == 'T':
            w=np.maximum(B.data/self.mean[self.cols]+dv,1e-9)*self.mean[self.cols]
            sums=np.bincount(self.cols,weights=w,minlength=B.shape[1])
            scale=np.divide(self.budget,sums,out=np.ones_like(sums),where=sums>0)
            w*=scale[self.cols]
            dw=float(np.abs(w-B.data).sum())
            self.pending_timing = w
        # Only its own channel's pre-teacher output enters each homeostatic cell.
        if self.arm == 'H': self.h+=(raw-self.h)*self.p['homeo_beta']
        return dw

    def teach(self,answer,t,write,world,record,raw,phi):
        # Unsupervised mechanisms have completed before this method receives label.
        for s in self.stores: s.byte(answer,t,learn=False)
        native=[]
        for j,s in enumerate(self.stores):
            native.append(s.teach_logged(int(48+j!=answer),t,write=bool(write)))
        if self.pending_timing is not None:
            B=self.stores[0].fly.m.B
            for s in self.stores:
                s.fly.m.B=csr_matrix((self.pending_timing.copy(),B.indices.copy(),B.indptr.copy()),shape=B.shape)
            self.pending_timing=None
        before=self.jw.copy()
        self.jw*=math.exp(-max(0.,t-self.jlast)/self.p['j_tau']); self.jlast=float(t)
        drift=float(np.abs(self.jw-before).sum())
        j_write=0.;j_step_l2=0.;j_proposed_l1=0.;j_clipped=0
        if self.arm.startswith('J') and write:
            e=phi.copy()
            if self.arm == 'J_SHUFFLE':
                seed=int.from_bytes(hashlib.sha256(f'JSHUF-v1|{world}|{record}'.encode()).digest()[:8],'big')
                e=e[np.random.default_rng(seed).permutation(len(e))]
            target=np.array([1. if 48+j==answer else -1. for j in range(4)])
            pred=self.jw@e
            delta=self.p['j_lr']*(target-pred)[:,None]*e[None,:]/max(1.,float(e@e))
            old_weights=self.jw.copy()
            self.jw+=delta
            j_clipped=int(np.sum(np.abs(self.jw)>self.p['j_bound']))
            np.clip(self.jw,-self.p['j_bound'],self.p['j_bound'],out=self.jw)
            actual=self.jw-old_weights
            j_write=float(np.abs(actual).sum());j_step_l2=float(np.linalg.norm(actual))
            j_proposed_l1=float(np.abs(delta).sum())
        return dict(write=int(write),native_l1=native,j_write_l1=j_write,j_decay_l1=drift,j_step_l2=j_step_l2,j_proposed_l1=j_proposed_l1,
                    j_clipped_coordinates=j_clipped,
                    j_max=float(np.max(np.abs(self.jw))),h=self.h.tolist(),
                    phi_sha=digest(phi),sensory_sha=self.sensory_digest())

    def end_record(self,t,newline_at):
        for s in self.stores:
            s.byte(10,newline_at,learn=False)
            if s.flush(t)>1e-6: raise AssertionError('native clock mismatch')

    def flush(self,t):
        for s in self.stores:
            if s.flush(t)>1e-6: raise AssertionError('native clock mismatch')

    def state_budget(self):
        return dict(added_allocated_bytes=sum(getattr(self,n).nbytes for n in
                    ('tpre','tpost','telig','h','jw','recent','elig')),
                    conjunctive_synapses=int(self.jw.size),
                    timing_edges=int(len(self.cols)),homeostatic_scalars=4,
                    operative_components=('timing' if self.arm=='T' else 'homeostasis' if self.arm=='H'
                                          else 'added_output_bank' if self.arm.startswith('J') else 'none'))
