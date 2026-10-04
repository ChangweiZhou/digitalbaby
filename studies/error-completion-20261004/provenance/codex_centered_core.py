"""R_center v1: fixed byte bridge and eight independently born Full151 stores.

The core receives only bytes and their times. Output precedes the arriving
teacher byte. No task map, stage, world, evaluator, or learned p enters it.
"""
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from . import bootstrap  # noqa: F401
import stores
from common_platform import clone_model

VERSION = 'R_CENTER_ENGINEERING_V1'
ALPHABET = b'0123'
SCALES = (1.4911274663291492, 1.3452365735750882)


@dataclass(frozen=True)
class Prediction:
    emitted: int
    shared: tuple
    private: tuple
    combined: tuple


class CenteredCore:
    """Frozen 12-byte cue protocol; predict, observe outcome, newline.

    A newline can discard an incomplete cue. Learning is controlled explicitly
    at observe_outcome(), identically for all eight stores. Four fixed ASCII
    output channels, first-maximum tie rule, and upstream scales are unchanged.
    """
    def __init__(self):
        from .integrity import verify_sources
        verify_sources()
        self.shared, self.private, self.births = [], [], []
        for bank, kind in [('shared', 'native'), ('private', 'content')]:
            for j in range(4):
                m, receipt = stores.birth(kind)
                getattr(self, bank).append(m)
                self.births.append({'bank': bank, 'store': j, **receipt})
        if len({id(m.fly) for m in self.models}) != 8:
            raise AssertionError('eight independent births required')
        self.cached = None
        self.prediction_time = None
        self.cue_count = 0
        self.awaiting_newline = False
        self.last_time = 0.0
        self.records = 0
        self.last_write = None

    @property
    def models(self):
        return self.shared + self.private

    @staticmethod
    def _byte(b):
        if isinstance(b, bool) or not isinstance(b, Integral) or not 0 <= b <= 255:
            raise ValueError('byte must be an integer in 0..255')
        return int(b)

    def _time(self, t):
        if isinstance(t, bool) or not isinstance(t, Real) or not math.isfinite(t) or t < self.last_time:
            raise ValueError('finite monotonic time required')
        return float(t)

    def clone(self):
        out = copy.copy(self)
        out.shared = [clone_model(m) for m in self.shared]
        out.private = [clone_model(m) for m in self.private]
        out.cached = None if self.cached is None else self.cached.copy()
        out.births = copy.deepcopy(self.births)
        out.last_write = copy.deepcopy(self.last_write)
        return out

    def bank_digests(self):
        return [m.state_digest() for m in self.models]

    def state_digest(self):
        meta = {'banks': self.bank_digests(), 'cached': None if self.cached is None else self.cached.tolist(),
                'prediction_time': self.prediction_time, 'cue_count': self.cue_count,
                'awaiting_newline': self.awaiting_newline, 'last_time': self.last_time,
                'records': self.records, 'last_write': self.last_write, 'version': VERSION}
        return hashlib.sha256(json.dumps(meta, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def feed(self, byte, t):
        byte, t = self._byte(byte), self._time(t)
        if self.cached is not None:
            raise ValueError('observe the outcome before another byte')
        if self.awaiting_newline and byte != 10:
            raise ValueError('newline required after outcome')
        if byte != 10 and self.cue_count >= 12:
            raise ValueError('twelve cue bytes already observed')
        for m in self.models:
            m.byte(byte, t, learn=False)
        self.cue_count = 0 if byte == 10 else self.cue_count + 1
        self.awaiting_newline = False
        self.last_time = t

    def predict(self, t):
        t = self._time(t)
        if self.cached is not None or self.awaiting_newline or self.cue_count != 12:
            raise ValueError('prediction requires twelve cue bytes and no pending outcome')
        before = self.bank_digests()
        s = np.array([m.association_value(t) for m in self.shared], dtype=float)
        p = np.array([m.association_value(t) for m in self.private], dtype=float)
        u = s / SCALES[0] + p / SCALES[1]
        if before != self.bank_digests() or not np.isfinite(u).all():
            raise AssertionError('prediction changed memory or produced nonfinite values')
        self.cached, self.prediction_time, self.last_time = s.copy(), t, t
        return Prediction(int(ALPHABET[int(np.argmax(u))]), tuple(s), tuple(p), tuple(u))

    def observe_outcome(self, byte, t, *, learn=True):
        byte, t = self._byte(byte), self._time(t)
        if type(learn) is not bool:
            raise ValueError('learn must be bool')
        if self.cached is None or byte not in ALPHABET or t != self.prediction_time:
            raise ValueError('actual output byte required at the prediction slot')
        # The only teacher source is the byte that has actually arrived.
        signs = [.25 - float(b == byte) for b in ALPHABET]
        for m in self.models:
            m.byte(byte, t, learn=False)
        x = self.private[0].pending_x
        if x is None or any(not np.array_equal(m.pending_x, x) for m in self.private[1:]):
            raise AssertionError('private pre-outcome addresses differ')
        shared = [m.teach_logged(int(b != byte), t, write=learn) for b, m in zip(ALPHABET, self.shared)]
        private = [m.teach_signed(0., s, t, write=learn) for s, m in zip(signs, self.private)]
        self.last_write = {'outcome': byte, 'time': t, 'learn': learn, 'c': 0., 's': signs,
                           'shared_l1': shared, 'private_l1': private}
        if not learn and any(v != 0. for v in shared + private):
            raise AssertionError('disabled teaching applied a write')
        self.cached, self.prediction_time = None, None
        self.awaiting_newline = True
        self.last_time, self.records = t, self.records + 1
        return copy.deepcopy(self.last_write)

    def flush(self, t):
        t = self._time(t)
        if self.cached is not None:
            raise ValueError('cannot advance time with a pending outcome')
        error = max(m.flush(t) for m in self.models)
        self.last_time = t
        return error

    def rest(self, seconds):
        if isinstance(seconds, bool) or not isinstance(seconds, Real) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('finite nonnegative rest required')
        return self.flush(self.last_time + float(seconds))

    def save(self, path):
        from .checkpoint import save
        save(self, path)

    @classmethod
    def load(cls, path):
        from .checkpoint import load
        return load(path)
