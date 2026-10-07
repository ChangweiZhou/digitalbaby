"""Continuous observed-byte errors write ONLY native Full151 fast/slow state.

No task boundary, label, phase, cue ID or external Teach input exists. Four
canonical stores are born separately. The independent V9 predictive matrices
are discarded with the temporary birth wrappers and never retained or used.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import OrderedDict

import numpy as np

import bootstrap  # noqa: F401
import stores
import spec


def digest_arrays(arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class Compartment(stores._Teach):
    def __init__(self):
        temporary, self.birth_record = stores.birth('native')
        self.fly = temporary.fly
        self.brain_t = 0.
        self.elapsed_base = float(self.fly.m.elapsed)
        self.pending_x = self.pending_t = None
        self.teach_seen = 0


class ObservationCore:
    def __init__(self, *, plastic=True):
        if type(plastic) is not bool:
            raise ValueError('plastic must be bool')
        self.stores = [Compartment() for _ in spec.ALPHABET]
        self.model = stores.bb.bc.native().model
        self.common = stores.bb.bc.native().common
        self.plastic = plastic
        self.history = b''
        self.t = 0.
        self.bytes_seen = 0
        self.cached = None
        self.cache = OrderedDict()  # <=256 deterministic receptor vectors, no learned table.
        for i, a in enumerate(self.stores):
            for b in self.stores[i + 1:]:
                for name in ('fast', 'slow', 'adapt'):
                    if np.shares_memory(getattr(a.fly.m, name), getattr(b.fly.m, name)):
                        raise AssertionError('native newborn states alias')

    def clone(self):
        out = copy.copy(self)
        out.stores = []
        for s in self.stores:
            c = copy.copy(s)
            c.fly = s.fly.clone()
            c.pending_x = None if s.pending_x is None else s.pending_x.copy()
            out.stores.append(c)
        out.cache = self.cache.copy()
        out.cached = copy.deepcopy(self.cached)
        return out

    def native_digest(self):
        return digest_arrays([a for s in self.stores for a in
                              (s.fly.m.fast, s.fly.m.slow, s.fly.m.adapt)])

    def state_digest(self):
        scalars = [self.history.hex(), self.t, self.bytes_seen, self.plastic,
                   [(s.brain_t, s.teach_seen, float(s.fly.m.elapsed),
                     int(s.fly.m.event_count), int(s.fly.m.presentation_count)) for s in self.stores]]
        if self.cached is not None:
            scalars += [self.cached['t'], self.cached['history'].hex(), self.cached['p'].tolist(),
                        digest_arrays(self.cached['codes'])]
        return hashlib.sha256((self.native_digest() + json.dumps(scalars, sort_keys=True)).encode()).hexdigest()

    def _time(self, t):
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < self.t:
            raise ValueError('finite monotonic observation time required')
        return float(t)

    def receptors(self):
        key = self.history
        if key not in self.cache:
            seed = int.from_bytes(hashlib.sha256(b'RAW-CONTEXT-4-v1|' + key).digest()[:8], 'little')
            rng = np.random.default_rng(seed)
            p = np.zeros(88, dtype=np.float64)
            idx = rng.choice(88, spec.CONTACTS, replace=False)
            p[idx] = rng.uniform(.4, 1., size=spec.CONTACTS)
            p /= p.max()
            p.flags.writeable = False
            self.cache[key] = p
            if len(self.cache) > 256:
                self.cache.popitem(last=False)
        return self.cache[key]

    def predict(self, t):
        t = self._time(t)
        if self.cached is not None:
            raise ValueError('observe the pending byte before predicting again')
        codes, values = [], []
        receptors = self.receptors()
        for s in self.stores:
            x = np.asarray(self.model.encode_sparse(s.fly.m, receptors), dtype=np.float64)
            n = s.fly.clone()
            n.rest(t - self.t)
            dx = self.common.observed_activity(n.m, np.atleast_2d(x))
            values.append(float((n.m.expression(dx) - n.reader.predict(dx)).mean(1)[0]))
            codes.append(x)
        logits = np.asarray(values) / spec.SCALE
        if not np.isfinite(logits).all():
            raise FloatingPointError('nonfinite native values')
        p = np.exp(logits - logits.max())
        p /= p.sum()
        self.cached = dict(t=t, history=self.history, codes=codes, p=p.copy())
        return dict(probabilities=p.tolist(), emitted=int(spec.ALPHABET[int(np.argmax(p))]),
                    native_values=values, context=self.history.hex())

    def observe(self, byte, t):
        if type(byte) is not int or byte not in spec.ALPHABET:
            raise ValueError('raw byte outside declared alphabet 0123')
        t = self._time(t)
        if self.cached is None or t != self.cached['t']:
            raise ValueError('prediction must precede this observation at the same time')
        if self.history != self.cached['history']:
            raise AssertionError('pre-arrival history changed')
        p = self.cached['p']
        signs = p.copy()
        signs[spec.ALPHABET.index(byte)] -= 1.
        dose = []
        for i, s in enumerate(self.stores):
            s.pending_x = self.cached['codes'][i].copy()
            s.pending_t = t
            dose.append(s.teach_signed(0., float(signs[i]), t, write=self.plastic))
        if not self.plastic and any(v != 0. for v in dose):
            raise AssertionError('disabled native learning applied a write')
        result = dict(byte=byte, t=t, context=self.history.hex(), probabilities=p.tolist(),
                      emitted=int(spec.ALPHABET[int(np.argmax(p))]), signs=signs.tolist(),
                      native_write_l1=dose, plastic=self.plastic,
                      loss_bits=-math.log2(float(p[spec.ALPHABET.index(byte)])))
        self.history = (self.history + bytes([byte]))[-spec.WINDOW:]
        self.t = t
        self.bytes_seen += 1
        self.cached = None
        return result

    def rest(self, seconds):
        if self.cached is not None:
            raise ValueError('cannot rest with a pending prediction')
        if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('invalid rest')
        for s in self.stores:
            s.fly.rest(float(seconds))
            s.brain_t += float(seconds)
        self.t += float(seconds)

    def erase_native_content(self):
        if self.cached is not None:
            raise ValueError('erase only at an observation boundary')
        for s in self.stores:
            s.fly.m.fast.fill(0.)
            s.fly.m.slow.fill(0.)

    def mutable_bytes(self):
        return sum(a.nbytes for s in self.stores for a in
                   (s.fly.m.fast, s.fly.m.slow, s.fly.m.adapt)) + spec.WINDOW
