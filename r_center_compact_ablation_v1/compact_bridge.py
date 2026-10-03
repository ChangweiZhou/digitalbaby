"""One physical deletion; the parent engineering release remains immutable."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / 'r_center_core_v1'
sys.path.insert(0, str(PARENT))
from centered_core import CenteredCore, Prediction
from centered_core.core import ALPHABET, SCALES
from centered_core.integrity import verify_sources as verify_parent
import stores
import numpy as np

ARMS = ('R_center_8', 'CONTENT_4')


class CompactCore(CenteredCore):
    def __init__(self):
        verify_parent()
        self.shared, self.private, self.births = [], [], []
        for j in range(4):
            model, receipt = stores.birth('content')
            self.private.append(model)
            self.births.append({'bank': 'private', 'store': j, **receipt})
        assert len({id(m.fly) for m in self.models}) == 4
        assert len({id(m.fly.m.B.data) for m in self.models}) == 4
        self.cached = None
        self.prediction_time = None
        self.cue_count = 0
        self.awaiting_newline = False
        self.last_time = 0.0
        self.records = 0
        self.last_write = None

    def predict(self, t):
        t = self._time(t)
        if self.cached is not None or self.awaiting_newline or self.cue_count != 12:
            raise ValueError('prediction requires twelve cue bytes and no pending outcome')
        before = self.bank_digests()
        p = np.array([m.association_value(t) for m in self.private], dtype=float)
        u = p / SCALES[1]
        if before != self.bank_digests() or not np.isfinite(u).all():
            raise AssertionError('mutating/nonfinite prediction')
        self.cached = np.zeros(4)
        self.prediction_time = self.last_time = t
        return Prediction(int(ALPHABET[int(np.argmax(u))]), (), tuple(p), tuple(u))

    def observe_outcome(self, byte, t, *, learn=True):
        byte, t = self._byte(byte), self._time(t)
        if type(learn) is not bool:
            raise ValueError('learn must be bool')
        if self.cached is None or byte not in ALPHABET or t != self.prediction_time:
            raise ValueError('actual output byte required at prediction slot')
        signs = [.25 - float(b == byte) for b in ALPHABET]
        for m in self.private:
            m.byte(byte, t, learn=False)
        x = self.private[0].pending_x
        if x is None or any(not np.array_equal(m.pending_x, x) for m in self.private[1:]):
            raise AssertionError('private pre-outcome addresses differ')
        applied = [m.teach_signed(0., s, t, write=learn) for s, m in zip(signs, self.private)]
        self.last_write = {'outcome': byte, 'time': t, 'learn': learn, 'c': 0., 's': signs,
                           'shared_l1': [], 'private_l1': applied}
        if not learn and any(applied):
            raise AssertionError('disabled teaching applied a write')
        self.cached = self.prediction_time = None
        self.awaiting_newline = True
        self.last_time, self.records = t, self.records + 1
        import copy
        return copy.deepcopy(self.last_write)

    def save(self, path):
        from compact_checkpoint import save
        from compact_integrity import verify
        save(self, path, {'component_only': True}, verify())

    @classmethod
    def load(cls, path):
        from compact_checkpoint import load
        from compact_integrity import verify
        core, _ = load(path, 'CONTENT_4', verify())
        return core


def make_core(arm):
    if arm == 'R_center_8': return CenteredCore()
    if arm == 'CONTENT_4': return CompactCore()
    raise ValueError(arm)
