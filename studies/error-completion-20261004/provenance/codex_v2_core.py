"""Two declared core factors; no fixture, stage, key table or evaluator enters here."""
import copy
import hashlib
import json
import numpy as np
import bootstrap
from centered_core import CenteredCore
from centered_core.core import ALPHABET, SCALES

ARMS = ('V1', 'ERROR', 'REPLACE', 'BOTH')
CONFLICT_COUNT = 3
REPLACE_FRACTION = .5


def probabilities(private):
    z = np.asarray(private, dtype=float) / SCALES[1]
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


class TrialCore(CenteredCore):
    def __init__(self, arm):
        if arm not in ARMS: raise ValueError('unknown arm')
        super().__init__()
        self.arm = arm
        self.conflicts = np.zeros((4, len(self.private[0].fly.m.slow)), dtype=np.uint8)
        self.private_prediction = None

    def clone(self):
        out = super().clone()
        out.conflicts = self.conflicts.copy()
        out.private_prediction = None if self.private_prediction is None else self.private_prediction.copy()
        return out

    def state_digest(self):
        meta = (super().state_digest(), self.arm,
                None if self.private_prediction is None else self.private_prediction.tolist())
        return hashlib.sha256(repr(meta).encode() + self.conflicts.tobytes()).hexdigest()

    def predict(self, t):
        result = super().predict(t)
        self.private_prediction = probabilities(result.private)
        return result

    def observe_outcome(self, byte, t, *, learn=True):
        byte, t = self._byte(byte), self._time(t)
        if type(learn) is not bool: raise ValueError('learn must be bool')
        if self.cached is None or self.private_prediction is None or byte not in ALPHABET or t != self.prediction_time:
            raise ValueError('observed byte must follow a prediction at the same slot')
        y = np.array([float(b == byte) for b in ALPHABET])
        pi = self.private_prediction.copy()
        signs = pi - y if self.arm in ('ERROR', 'BOTH') else .25 - y
        before_evidence = hashlib.sha256(self.conflicts.tobytes()).hexdigest()
        for m in self.models: m.byte(byte, t, learn=False)
        x = self.private[0].pending_x
        if x is None or any(not np.array_equal(m.pending_x, x) for m in self.private[1:]):
            raise AssertionError('different private pre-outcome addresses')
        shared = [m.teach_logged(int(b != byte), t, write=learn) for b, m in zip(ALPHABET, self.shared)]
        private, replacements = [], []
        for j, (s, m) in enumerate(zip(signs, self.private)):
            # The reference advances native non-plastic time, so decay is not
            # mistaken for an opposing learning update.
            enabled = learn and self.arm in ('REPLACE', 'BOTH')
            reference = None
            if enabled:
                reference = m.fly.clone()
                reference.event(m._teach_prologue(t), m.pending_x, 0., False)
            applied = m.teach_signed(0., float(s), t, write=learn)
            private.append(applied)
            info = {'enabled': bool(enabled), 'changed': 0, 'removed_l1': 0.,
                    'ids': [], 'old': [], 'delta': [], 'after': [], 'evidence': [],
                    'outside_conflict_unchanged': True, 'certificate_error': 0.}
            if enabled:
                old = reference.m.slow
                delta = m.fly.m.slow - old
                active = x != 0
                conflict = active & (old * delta < 0.)
                self.conflicts[j, active & ~conflict] = 0
                self.conflicts[j, conflict] = np.minimum(
                    self.conflicts[j, conflict].astype(np.int16) + 1, CONFLICT_COUNT)
                ids = np.flatnonzero(conflict & (self.conflicts[j] >= CONFLICT_COUNT))
                pre = m.fly.m.slow.copy()
                removed = REPLACE_FRACTION * old[ids]
                m.fly.m.slow[ids] -= removed
                outside = np.ones(len(old), dtype=bool); outside[ids] = False
                unchanged = np.array_equal(pre[outside], m.fly.m.slow[outside])
                expected = old[ids] + delta[ids] - removed
                error = float(np.max(np.abs(m.fly.m.slow[ids] - expected))) if len(ids) else 0.
                if not unchanged or error > 1e-12 or not np.isfinite(m.fly.m.slow).all():
                    raise AssertionError('selective replacement certificate failed')
                info.update(changed=len(ids), removed_l1=float(np.abs(removed).sum()), ids=ids.tolist(),
                            old=old[ids].tolist(), delta=delta[ids].tolist(), after=m.fly.m.slow[ids].tolist(),
                            evidence=self.conflicts[j, ids].tolist(), outside_conflict_unchanged=unchanged,
                            certificate_error=error)
            replacements.append(info)
        after_evidence = hashlib.sha256(self.conflicts.tobytes()).hexdigest()
        if not learn and (any(shared + private) or before_evidence != after_evidence):
            raise AssertionError('disabled write or evidence update')
        self.last_write = {'outcome': byte, 'time': t, 'learn': learn, 'c': 0., 's': signs.tolist(),
                           'private_pre_outcome_probabilities': pi.tolist(),
                           'shared_l1': shared, 'private_l1': private, 'replacements': replacements,
                           'evidence_before': before_evidence, 'evidence_after': after_evidence}
        self.cached = self.private_prediction = self.prediction_time = None
        self.awaiting_newline = True
        self.last_time, self.records = t, self.records + 1
        return copy.deepcopy(self.last_write)


class ChoiceOrgan:
    """Inherited two-option protocol, comparing the model's own binary utility.

    Both option streams advance one disposable core clone. No reset between
    options, no label and no taught/held-out status enters the organ. Both orders
    are scored. The ordinary four-output organ remains unchanged for facts.
    """
    def __init__(self, core):
        self.core = core.clone()

    def choose(self, first, second, at, dt):
        values, predictions = [], []
        for k, cue in enumerate((first, second)):
            start = at + 13 * k * dt
            for i, byte in enumerate(cue): self.core.feed(byte, start + i * dt)
            p = self.core.predict(start + 12 * dt)
            predictions.append(p)
            values.append(p.combined[1] - p.combined[0])
            # Separator is sensory, unreinforced; it is not an answer byte.
            self.core.cached = self.core.private_prediction = self.core.prediction_time = None
            self.core.cue_count = 0
            self.core.feed(ord('|') if k == 0 else ord('?'), start + 12 * dt)
            self.core.cue_count = 0
        emitted = ord('L') if values[0] >= values[1] else ord('R')
        return emitted, values, predictions
