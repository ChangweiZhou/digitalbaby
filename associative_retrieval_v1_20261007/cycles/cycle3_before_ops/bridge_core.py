"""Integrated associative retrieval, with unchanged ERROR native writes."""
import copy
import hashlib
import time
from dataclasses import dataclass, asdict
import runtime
import numpy as np
from core import Core as ParentCore
from spec import ALPHABET, SCALES
from association import Associations, code_ids, GAIN

@dataclass(frozen=True)
class Prediction:
    emitted: int
    shared: tuple
    private: tuple
    combined: tuple
    retrieval: tuple
    permutation_retrieval: tuple
    shared_ids: tuple
    private_ids: tuple
    query: tuple
    policies: dict
    costs: dict

def address(brain, t):
    fe = brain.fe.clone()
    fe.advance(t)
    return code_ids(brain.model.encode_sparse(brain.fly.m, fe.read()), brain.n_native_kc)

def read_addresses(brain, t, addresses):
    if not addresses:
        return np.empty(0)
    n = brain.fly.clone()
    if brain.pending_t is not None:
        n.rest(brain.pending_t - brain.brain_t)
        n.rest(t - brain.pending_t)
    else:
        n.rest(t - brain.brain_t)
    x = np.zeros((len(addresses), brain.n_native_kc))
    for j, ids in enumerate(addresses):
        x[j, ids] = 1.
    dx = brain.common.observed_activity(n.m, x)
    alpha = n.m.expression(dx)
    pred = brain.fly.reader.predict(dx)
    return np.asarray((alpha - pred).mean(1), dtype=float)

class RetrievalCore(ParentCore):
    def __init__(self):
        super().__init__('ERROR')
        n = self.private[0].n_native_kc
        self.associations = Associations(n)
        self.association_pending = None
        # Diagnostic preserves anatomical side, encoder eligibility and writable support.
        m = self.private[0].fly.m
        side = np.asarray(m.kc_side)
        eligible = np.asarray(m.B.sum(axis=0)).ravel() > 0
        writable = np.abs(np.asarray(m.T)[:, 2:4]).sum(axis=1) > 0
        rng = np.random.default_rng(0x41535231)
        self.permutation = np.arange(n, dtype=np.int32)
        for s in (0, 1, 2):
            for e in (False, True):
                for w in (False, True):
                    ids = np.flatnonzero((side == s) & (eligible == e) & (writable == w))
                    self.permutation[ids] = rng.permutation(ids)
        self.permutation.flags.writeable = False
        self.last_costs = {}

    def clone(self):
        out = super().clone()
        out.associations = self.associations.clone()
        out.association_pending = copy.deepcopy(self.association_pending)
        out.last_costs = self.last_costs.copy()
        return out

    def state_digest(self):
        return hashlib.sha256((super().state_digest() + self.associations.digest()
                               + repr(self.association_pending)).encode()).hexdigest()

    def native_digest(self):
        return ParentCore.state_digest(self)

    def predict(self, t):
        start = time.process_time()
        p = super().predict(t)
        base_s = time.process_time() - start
        start = time.process_time()
        shared_ids = address(self.shared[0], t)
        private_ids = address(self.private[0], t)
        self.association_pending = (shared_ids.tolist(), private_ids.tolist())
        query = self.associations.query(shared_ids)
        query_s = time.process_time() - start
        before = self.bank_digests()
        start = time.process_time()
        r = self._retrieve(query, t, False)
        retrieval_s = time.process_time() - start
        start = time.process_time()
        rp = self._retrieve(query, t, True)
        control_s = time.process_time() - start
        if before != self.bank_digests():
            raise AssertionError('association retrieval mutated native memory')
        u0 = np.asarray(p.combined)
        u = u0 + GAIN * (r - r.mean()) / SCALES[1]
        up = u0 + GAIN * (rp - rp.mean()) / SCALES[1]
        policies = {name: {'combined': a.tolist(), 'emitted': int(ALPHABET[int(np.argmax(a))])}
                    for name, a in [('ERROR', u0), ('LINK', u), ('PERM', up)]}
        costs = {'parent_prediction_cpu_s': base_s, 'query_cpu_s': query_s,
                 'retrieval_cpu_s': retrieval_s, 'control_retrieval_cpu_s': control_s}
        return Prediction(policies['LINK']['emitted'], p.shared, p.private, tuple(u),
                          tuple(r), tuple(rp), tuple(map(int, shared_ids)), tuple(map(int, private_ids)),
                          tuple(query), policies, costs)

    def _retrieve(self, query, t, permute):
        if not query:
            return np.zeros(4)
        codes = [np.asarray(r['private_ids'], dtype=np.int32) for r in query]
        if permute:
            codes = [np.sort(self.permutation[c]) for c in codes]
        weights = np.asarray([r['weight'] for r in query])
        r = np.array([float(read_addresses(m, t, codes) @ weights) for m in self.private])
        if not np.isfinite(r).all():
            raise FloatingPointError('nonfinite retrieval')
        return r

    def observe_outcome(self, byte, t, *, learn=True):
        if self.association_pending is None:
            raise ValueError('missing pre-answer association cache')
        s, p = copy.deepcopy(self.association_pending)
        start = time.process_time()
        w = super().observe_outcome(byte, t, learn=learn)
        base_s = time.process_time() - start
        start = time.process_time()
        before = self.associations.digest()
        update = self.associations.observe(np.asarray(s, np.int32), np.asarray(p, np.int32))
        update_s = time.process_time() - start
        self.association_pending = None
        w['association'] = {'before': before, 'after': self.associations.digest(), **update,
                            'shared_ids': s, 'private_ids': p, 'cue_only': True}
        self.last_costs = {'parent_outcome_cpu_s': base_s, 'association_update_cpu_s': update_s}
        return w

    def clear_unreinforced_prediction(self):
        super().clear_unreinforced_prediction()
        self.association_pending = None

class ChoiceOrgan:
    def __init__(self, core):
        self.core = core.clone()

    def choose(self, first, second, at, dt):
        ps = []
        vals = {k: [] for k in ('ERROR', 'LINK', 'PERM')}
        for k, cue in enumerate((first, second)):
            start = at + 13 * k * dt
            for i, b in enumerate(cue):
                self.core.feed(b, start + i * dt)
            p = self.core.predict(start + 12 * dt)
            ps.append(asdict(p))
            for name, v in vals.items():
                u = p.policies[name]['combined']
                v.append(u[1] - u[0])
            self.core.clear_unreinforced_prediction()
            self.core.feed(ord('|') if k == 0 else ord('?'), start + 12 * dt)
            self.core.cue_count = 0
        return {name: {'emitted': ord('L') if v[0] >= v[1] else ord('R'), 'values': v}
                for name, v in vals.items()}, ps
