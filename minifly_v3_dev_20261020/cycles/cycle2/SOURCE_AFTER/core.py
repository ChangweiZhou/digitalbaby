"""Generic byte learner: gated local plasticity feeding native value memory.

This module cannot import the task generator/evaluator. The sole answer route
is native values decoded through a fixed, task-independent byte identity port.
"""
import copy
import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import bootstrap
from native import NativeBank

UP = bootstrap.upstream_fastmem()
BC, CODE, EOS, DELTA = UP.BC, UP.CODE, 256, 30/14
for value in (CODE,):
    value.flags.writeable = False
N_OLD, HORIZON, START = 7680, 960, 1/16


def ramp(n):
    u = min(max(n/HORIZON, 0.), 1.)
    return START+(1-START)*(3*u*u-2*u*u*u)


_RATES = np.array([ramp(n) for n in range(1, N_OLD+1)])
MEAN_RATE, MEAN_SQUARE = float(_RATES.mean()), float((_RATES*_RATES).mean())


def multipliers(n, gradual):
    if n > N_OLD:
        return 1., 1.
    a = ramp(n)
    return (a, a*a) if gradual else (MEAN_RATE, MEAN_SQUARE)


def hash_arrays(arrays):
    out = hashlib.sha256()
    for a in arrays:
        out.update(np.ascontiguousarray(a).tobytes())
    return out.hexdigest()


@dataclass(frozen=True)
class Limits:
    eta_max: float = .00003
    eligibility_clip: float = 8.
    relative_row_cap: float = .001
    projection_bound: float = .125


class Learner:
    def __init__(self, seed, plastic=True, gradual=True, limits=None, source_parent=False):
        parent = UP.FastMemory()
        self.native = NativeBank(parent)
        self.seed = int(seed)
        self.births = self.native.births
        self.plastic, self.gradual, self.source_parent = bool(plastic), bool(gradual), bool(source_parent)
        self.limits = limits or Limits()
        rng = np.random.default_rng(seed)
        self.params = {}
        for name, shape in (('W', (64, 88)), ('U', (64, 64)), ('b', (64,)),
                            ('G', (64, 88)), ('H', (64, 64)), ('c', (64,))):
            if len(shape) == 1:
                a = np.full(shape, -1. if name == 'c' else 0.)
            else:
                a = rng.uniform(-math.sqrt(6/sum(shape)), math.sqrt(6/sum(shape)), shape)
            if name in ('U', 'H'):
                np.fill_diagonal(a, 0.)
            self.params[name] = a
        self.row_anchor = {k: np.maximum(1., np.linalg.norm(v.reshape(64, -1), axis=1))
                           for k, v in self.params.items()}
        self.D = rng.choice(np.array([-1., 1.]), (88, 64))/math.sqrt(64)
        self.Fback = np.stack([rng.permutation(np.r_[np.ones(8), -np.ones(8)])/4 for _ in range(64)])
        self.D.flags.writeable = self.Fback.flags.writeable = False
        self.initial_parameter_hash = self.parameter_hash()
        self.fe = UP.FEPerm()
        self.h = np.zeros(64)
        self.theta, self.freq = parent.theta.copy(), parent.freq.copy()
        self.eligibility = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.pending = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.time, self.records, self.parameter_steps, self.version = 0., 0, 0, 0
        self.trace_active = False
        self.last_trace = []
        self.last_update = {}
        self.stats = dict(native_alpha_l1=0., native_teach_events=0, observed_target_events=0,
                          local_update_l1=0., row_clips=0, queries=0, eligibility_clips=0,
                          total_records=0, supplied_answer_bytes=0)

    def parameter_hash(self):
        return hash_arrays([v for _, v in sorted(self.params.items())])

    def learning_hash(self):
        return hash_arrays([self.native.fast, self.native.slow, self.theta, self.freq,
                            *[v for _, v in sorted(self.params.items())]])

    def digest(self):
        out = hash_arrays([self.native.fast, self.native.slow, self.native.adapt, self.theta,
                           self.freq, self.h, self.fe.p, *[v for _, v in sorted(self.params.items())],
                           *[v for _, v in sorted(self.eligibility.items())],
                           *[v for _, v in sorted(self.pending.items())]])
        meta = (self.time, self.fe.t, self.records, self.parameter_steps, self.version, self.stats)
        return hashlib.sha256((out+json.dumps(meta, sort_keys=True)).encode()).hexdigest()

    def clone(self):
        out = copy.copy(self)
        out.native = self.native.clone()
        out.h, out.fe = self.h.copy(), self.fe.clone()
        out.theta, out.freq = self.theta.copy(), self.freq.copy()
        out.params = {k: v.view() for k, v in self.params.items()}
        for value in out.params.values():
            value.flags.writeable = False
        out.stats = self.stats.copy()
        out.eligibility = {k: np.zeros_like(v) for k, v in self.params.items()}
        out.pending = {k: np.zeros_like(v) for k, v in self.params.items()}
        out.trace_active = False
        out.last_trace = []
        return out

    def _context(self, byte):
        if self.source_parent:
            return
        r, old, p = np.asarray(BC.R_TABLE[byte]), self.h.copy(), self.params
        v = np.tanh(p['W']@r+p['U']@old+p['b'])
        g = 1/(1+np.exp(-np.clip(p['G']@r+p['H']@old+p['c'], -40, 40)))
        self.h = (1-g)*old+g*v
        if self.trace_active:
            drive = g*(1-v*v)
            gate = g*(1-g)*(v-old)
            for name, term in (('W', drive[:, None]*r), ('U', drive[:, None]*old), ('b', drive),
                               ('G', gate[:, None]*r), ('H', gate[:, None]*old), ('c', gate)):
                e = self.eligibility[name]
                e *= (1-g)[:, None] if e.ndim == 2 else 1-g
                e += term
                if name in ('U', 'H'):
                    np.fill_diagonal(e, 0.)
                self.stats['eligibility_clips'] += int(np.count_nonzero(abs(e) > self.limits.eligibility_clip))
                np.clip(e, -self.limits.eligibility_clip, self.limits.eligibility_clip, out=e)

    def _drive(self, front=None):
        f = self.fe if front is None else front
        p = f.read()
        if not self.source_parent:
            p = np.maximum(0., p+self.limits.projection_bound*np.tanh(self.D@self.h))
            p = p/max(float(p.max()), 1e-300)
        return p

    def _control(self, begin):
        self.time += DELTA
        self.native.rest(DELTA)
        if begin:
            self.fe = UP.FEPerm()
            self.fe.t = self.time
            self.native.adapt[:] = 1.
            self.h = np.zeros(64)
            for value in self.eligibility.values():
                value.fill(0.)
            for value in self.pending.values():
                value.fill(0.)
        self.version += 1

    def _frame(self):
        f = self.fe.clone()
        f.advance(self.time+DELTA)
        p = self._drive(f)
        x, raw = self.native.encode(p, self.theta)
        z = self.native.values(x, DELTA)
        return dict(x=x, raw=raw, drive=p, z=z, version=self.version, at=self.time+DELTA)

    def _homeostasis(self, frame, scale):
        x = frame['x']
        self.freq += .01*(x-self.freq)
        over = np.maximum(0., self.freq-5*self.native.active_fraction)
        if over.any():
            raw_sd = float((frame['raw']+self.theta).std())
            self.theta += (.05*scale)*over*raw_sd/self.native.active_fraction*.1

    def _event(self, token, target=None, allowed=False, frame=None):
        if type(token) is not int or not 0 <= token <= EOS:
            raise ValueError('byte or EOS')
        if target is not None and (type(target) is not int or target != token):
            raise ValueError('only arriving observed target may teach')
        if allowed and target is None:
            raise ValueError('unobserved bytes cannot teach')
        coeff = None
        if target is not None:
            if frame is None or frame['version'] != self.version or frame['at'] != self.time+DELTA:
                raise ValueError('stale prediction frame')
            s = (np.tanh((frame['z']-frame['z'].mean())/2)-CODE[target])/2
            scale, _ = (1., 1.) if self.source_parent else multipliers(self.records+1, self.gradual)
            if allowed:
                coeff = scale*s
                if self.trace_active:
                    signal = self.Fback@s
                    for name in self.pending:
                        self.pending[name] -= signal[:, None]*self.eligibility[name] if self.pending[name].ndim == 2 else signal*self.eligibility[name]
            self.stats['observed_target_events'] += 1
        self.time += DELTA
        self.fe.advance(self.time)
        if frame is None:
            x, raw = self.native.encode(self._drive(), self.theta)
        else:
            x = frame['x']
            if allowed:
                self._homeostasis(frame, scale)
        applied = self.native.event(x, DELTA, coeff)
        self.stats['native_alpha_l1'] += applied
        self.stats['native_teach_events'] += int(coeff is not None)
        self.version += 1
        if token != EOS:
            self.fe.feed(token, self.time)
            self._context(token)
        return coeff

    def _prompt(self, prompt):
        if type(prompt) is not bytes or len(prompt) > 256:
            raise ValueError('INPUT_LIMIT')
        self._control(True)
        for byte in prompt:
            self._event(byte)
        self._control(False)

    def train(self, prompt, answer, allowed=True):
        if type(answer) is not bytes or len(answer) > 16 or type(allowed) is not bool:
            raise ValueError('bounded observed payload and explicit permission required')
        self.trace_active = self.plastic and allowed and not self.source_parent
        self._prompt(prompt)
        self.last_trace, self.last_update = [], {}
        n_targets = len(answer)+1
        for target in (*answer, EOS):
            frame = self._frame()
            predicted = int(UP.logits(frame['z']).argmax())
            coeff = self._event(target, target, allowed, frame)
            self.last_trace.append(dict(target=target, predicted=predicted,
                                        pre_version=frame['version'], post_version=self.version,
                                        pretime=frame['at']-DELTA, posttime=self.time,
                                        address_hash=hash_arrays([frame['x']]),
                                        coefficient=None if coeff is None else coeff.tolist(),
                                        values=frame['z'].tolist(), observed=True, allowed=allowed))
        if self.trace_active:
            _, rate = multipliers(self.records+1, self.gradual)
            for name, value in self.params.items():
                update = self.pending[name]*(self.limits.eta_max*rate/n_targets)
                flat = update.reshape(64, -1)
                norm = np.linalg.norm(flat, axis=1)
                cap = self.limits.relative_row_cap*self.row_anchor[name]
                factor = np.minimum(1., cap/np.maximum(norm, 1e-300))
                update *= factor[:, None] if update.ndim == 2 else factor
                self.stats['row_clips'] += int(np.count_nonzero(factor < 1))
                value += update
                if name in ('U', 'H'):
                    np.fill_diagonal(value, 0.)
                if not np.isfinite(value).all():
                    raise FloatingPointError('nonfinite local update')
                size = float(abs(update).sum())
                self.stats['local_update_l1'] += size
                self.last_update[name] = dict(l1=size, max_row_norm=float(np.linalg.norm(update.reshape(64, -1), axis=1).max()),
                                              max_allowed=float(cap.max()))
            self.parameter_steps += 1
        self.records += 1
        self.stats['total_records'] += 1
        self.stats['supplied_answer_bytes'] += len(answer)
        self.trace_active = False
        return self.last_trace

    def generate(self, prompt, cap=16):
        if type(cap) is not int or not 0 <= cap <= 16:
            raise ValueError('output cap')
        view = self.clone()
        view._prompt(prompt)
        out, tokens = bytearray(), []
        while True:
            frame = view._frame()
            token = int(UP.logits(frame['z']).argmax())
            tokens.append(token)
            if token == EOS:
                return dict(payload=bytes(out), ended_with_EOS=True, truncated=False, tokens=tokens)
            if len(out) == cap:
                return dict(payload=bytes(out), ended_with_EOS=False, truncated=True, tokens=tokens)
            out.append(token)
            view._event(token)

    def rest(self, seconds):
        self.native.rest(seconds)
        self.time += seconds
        self.version += 1

    def diagnostics(self):
        return dict(records=self.records, time=self.time, parameter_steps=self.parameter_steps,
                    parameter_hash=self.parameter_hash(), homeostasis_mean=float(self.theta.mean()),
                    homeostasis_max=float(self.theta.max()), frequent_cells=int((self.freq>.8).sum()),
                    fast_l1=float(abs(self.native.fast).sum()), slow_l1=float(abs(self.native.slow).sum()),
                    mutable_bytes=int(sum(v.nbytes for v in (self.native.fast,self.native.slow,self.native.adapt,
                                                             self.theta,self.freq,self.h,*self.params.values(),
                                                             *self.eligibility.values(),*self.pending.values()))),
                    finite=bool(all(np.isfinite(v).all() for v in (self.native.fast,self.native.slow,
                                                                 self.native.adapt,self.theta,*self.params.values()))),
                    stats=self.stats.copy())
