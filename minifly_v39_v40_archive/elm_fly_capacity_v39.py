#!/usr/bin/env python3
"""V39: Bounded superposed memory.

Tests whether code separation is a *consequence* of bounded superposed storage
rather than an independent design goal.

    python3 elm_fly_capacity_v39.py --self-test
    python3 elm_fly_capacity_v39.py --stage g2  --out-dir v39_results
    python3 elm_fly_capacity_v39.py --stage full --out-dir v39_results

Label-free throughout: the only teaching signal is the next-byte residual.
Workload construction is V38's, extended with a load parameter and verified
byte-identical to V38 at V38's own settings (see self_test).

Design: v39_experiment_recommendation.md.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np
from scipy.stats import t as student_t

SOURCE = Path(__file__).resolve()
V38_SOURCE = SOURCE.parent / 'elm_v38_predictive_evidence.py'

VERSION = 'v39.1.0'
SEEDS = tuple(range(39101, 39117))          # 16 seeds; V38's 8 could not resolve its controls

CONTEXT_BYTES = 12
CODE_BITS = CONTEXT_BYTES * 8               # 96, the V33/V35/V38 lag code
OUTPUTS = 256
RATE = .6                                   # V36/V38 write rate, unchanged
TEMP = .1                                   # V38 readout temperature, unchanged

UNITS = 4096                                # KC-layer size, fixed: this is the capacity under test
CODING = 0.05                               # 5% coding level -> K = 205 active, fly-like
K_ACTIVE = int(round(UNITS * CODING))
MAX_PERIOD = 32
VARIANCE_FLOOR = 1e-6
WARMUP_BYTES = 4096                         # phase discovery uses only bytes already streamed

# Connectome and prior-version reference points. Report context only, never a learner input.
REFERENCE = dict(
    fly_kc_overlap_same=0.417, fly_kc_overlap_diff=0.032, fly_ratio=13.5,
    fly_pn_claws_median=5, fly_kc_count=2000,
    v36_chatgpt_gate=dict(j_same_min=0.30, j_diff_max=0.04, ratio_min=10.0),
    v36_chatgpt_measured=dict(ratio_range=[4.01, 4.91], j_diff_range=[0.096, 0.129]),
    v38_pred_v_slots_after_D=1504, v38_pred_v_payload_bytes_after_D=9723904,
    v38_e1_heldout=0.0080, v38_e2_heldout=1.0,
)


# ----------------------------------------------------------------------------
# 0. Substrate import
# ----------------------------------------------------------------------------

def _load_v38():
    if not V38_SOURCE.exists():
        raise SystemExit(
            'V39 reuses the V38 workload and slot-store architecture unchanged but\n'
            f'could not find {V38_SOURCE}\nPlace this file beside it.')
    spec = importlib.util.spec_from_file_location('elm_v39_v38', V38_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # V38 defers numpy and scipy into its own dependency bootstrap and binds them as
    # module globals. V39 needs only its stream, Encoder, Learner and ByteModel, all
    # of which are pure numpy, so inject directly rather than running that bootstrap.
    if getattr(module, 'np', None) is None:
        module.np = np
    if getattr(module, 'student_t', None) is None:
        module.student_t = student_t
    return module


V38 = _load_v38()
context = V38.context
PREFIX, HEADERS, SEPARATORS = V38.PREFIX, V38.HEADERS, V38.SEPARATORS
softmax, loss = V38.softmax, V38.loss

assert V38.D == CODE_BITS and V38.OUTPUTS == OUTPUTS and V38.TEMP == TEMP, \
    'V38 constants moved; V39 comparability assumptions are void.'


def sha(data):
    if isinstance(data, (bytes, bytearray)):
        return hashlib.sha256(bytes(data)).hexdigest()
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(values):
    """Paired Student 95%. Seeds are the sampling units; a missing seed voids the interval."""
    vals = [None if v is None else float(v) for v in values]
    good = [v for v in vals if v is not None and math.isfinite(v)]
    out = {'n': len(good), 'expected_n': len(vals),
           'mean': float(np.mean(good)) if good else None, 'low': None, 'high': None}
    if len(good) >= 2 and len(good) == len(vals):
        half = float(student_t.ppf(.975, len(good) - 1) * np.std(good, ddof=1) / math.sqrt(len(good)))
        out.update(low=out['mean'] - half, high=out['mean'] + half)
    return out


def paired(per_seed, left, right, key):
    """Difference of two cells on the same seeds. Never averages across seeds first."""
    seeds = sorted(set(per_seed.get(left, {})) & set(per_seed.get(right, {})))
    if not seeds:
        return interval([])
    return interval([per_seed[left][s].get(key) if per_seed[left][s].get(key) is None or
                     per_seed[right][s].get(key) is None else
                     per_seed[left][s][key] - per_seed[right][s][key] for s in seeds])


# ----------------------------------------------------------------------------
# 1. Workload: V38's stream, with a load parameter
# ----------------------------------------------------------------------------

def make_stream(seed, keys_per_wave=16, repeats=32, smoke=False):
    """V38.make_stream with `count` exposed. Byte-identical to V38 at (16, 32).

    The RNG call order is preserved exactly, so the only difference at other loads
    is the one the load axis intends: more keys per wave, identical training per key.
    `nrev` scales with the load so that the untouched retention cohort is always half
    of wave A. Fixing it at 4 instead would let the untouched fraction drift from 50%
    at load 64 to 94% at load 512, which is a second uncontrolled change along the
    axis under test.
    """
    rr = np.random.default_rng(seed + 810007)
    count, repeats = (4, 4) if smoke else (keys_per_wave, repeats)
    waves = 4
    key_alphabet = np.frombuffer(b'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', dtype=np.uint8)
    keys = []
    while len(keys) < count * waves:
        key = bytes(rr.choice(key_alphabet, size=6))
        if key not in keys:
            keys.append(key)
    labels = rr.choice(np.frombuffer(b'abcdefghijklmnopqrstuvwx', dtype=np.uint8),
                       size=len(keys)).astype(int)
    # Scaled with the load so the untouched (retention) cohort stays a constant half of
    # each wave. At V38's count=16 this is 4, and at its smoke count=4 it is 1, so the
    # stream stays byte-identical to V38 at both of V38's own settings.
    nrev = max(1, count // 4)
    revised = list(range(nrev))
    noisy = list(range(nrev, 2 * nrev))
    untouched = list(range(2 * nrev, count))
    changed = {i: int(ord('a') + (int(labels[i]) - ord('a') + 1 + int(rr.integers(23))) % 24)
               for i in revised}
    rows, checkpoints = [], []
    render_counts = collections.Counter()

    def add(i, target, kind, exposure, wave, alias):
        render_counts[(i, alias, kind)] += 1
        rows.append({'key_id': int(i), 'target': int(target), 'kind': kind, 'exposure': exposure,
                     'wave': wave, 'alias': alias,
                     'rendering_exposure': render_counts[(i, alias, kind)],
                     'raw': context(keys[i], alias) + bytes([target]) + b'\n'})

    for wave in range(waves):
        cohort = chr(65 + wave)
        for exposure in range(1, repeats + 1):
            for i in rr.permutation(np.arange(wave * count, (wave + 1) * count)):
                add(int(i), int(labels[i]), 'acquisition', exposure, cohort, (exposure - 1) % 2)
            if exposure in (1, 2, 4, 8, 16, 32):
                checkpoints.append({'record_count': len(rows), 'name': f'{cohort}_acq_{exposure}',
                                    'type': 'acquisition', 'wave': cohort, 'exposure': exposure})
        checkpoints.append({'record_count': len(rows), 'name': f'after_{cohort}',
                            'type': 'wave', 'wave': cohort})
        if wave == 1:
            for i in noisy:
                wrong = ord('a') + (int(labels[i]) - ord('a') + 1) % 24
                add(i, wrong, 'one_off_noise', 1, cohort, 0)
            checkpoints.append({'record_count': len(rows), 'name': 'after_noise',
                                'type': 'noise', 'wave': cohort})
        if wave == 2:
            for exposure in range(1, 9):
                for i in rr.permutation(revised):
                    add(int(i), changed[int(i)], 'revision', exposure, cohort, 0)
                if exposure in (1, 2, 4, 8):
                    checkpoints.append({'record_count': len(rows), 'name': f'revision_{exposure}',
                                        'type': 'revision', 'wave': cohort, 'exposure': exposure})
    raw = b''.join(row['raw'] for row in rows)
    assert all(len(row['raw']) == 14 for row in rows)
    assert len(set(keys)) == len(keys)
    return {'seed': seed, 'keys': keys, 'labels': labels, 'revised': revised, 'noisy': noisy,
            'untouched': untouched, 'revisions': changed, 'rows': rows, 'raw': raw,
            'checkpoints': checkpoints, 'keys_per_wave': count, 'repeats': repeats,
            'raw_sha256': sha(raw)}


# ----------------------------------------------------------------------------
# 2. Front ends: what the KC layer reads
# ----------------------------------------------------------------------------

def lag_bits(window):
    """The exact V33/V35/V38 lag96 code: 12 raw bytes, 8 signed bits each."""
    h = bytes(window)[-CONTEXT_BYTES:]
    out = np.zeros(CODE_BITS)
    if h:
        out[-8 * len(h):] = np.unpackbits(
            np.frombuffer(h, dtype=np.uint8)).astype(np.float64) * 2. - 1.
    return out


class LagFront:
    """Raw lag96, normalised. The V33/V35/V38 default, and V39's low-separation end."""

    name = 'lag'

    def __init__(self, seed):
        self.encoder = V38.Encoder(seed, 'lag')

    def discover(self, warmup):
        return 1

    def observe(self, window, t):
        pass

    def encode(self, window, t):
        return self.encoder.encode(bytes(window))[0]

    def state_bytes(self):
        return 0


class PhaseFront:
    """Phase-conditioned normalisation of lag96, from V36.

    A phase-blind normaliser cannot remove framing boilerplate, because a byte that is
    constant at a fixed record offset is not constant across offsets. V36 measured
    phase-blind at 1.44x separation and phase-conditioned at 5.71x on this workload.

    The period is discovered LABEL-FREE by byte autocorrelation over bytes the model
    has already streamed; on this workload it recovers 14, the true record length,
    without being told the record structure.
    """

    name = 'phase'

    def __init__(self, seed, max_period=MAX_PERIOD, floor=VARIANCE_FLOOR):
        self.max_period, self.floor = int(max_period), float(floor)
        self.period, self.n = 1, None

    def discover(self, warmup):
        a = np.frombuffer(bytes(warmup), dtype=np.uint8).astype(np.float64)
        if len(a) < 4 * self.max_period:
            self.period = 1
        else:
            a = a - a.mean()
            best, score = 1, -2.
            for lag in range(2, self.max_period + 1):
                c = float(np.corrcoef(a[:-lag], a[lag:])[0, 1])
                if math.isfinite(c) and c > score:
                    best, score = lag, c
            self.period = best
        self.n = np.zeros(self.period)
        self.mean = np.zeros((self.period, CODE_BITS))
        self.m2 = np.zeros((self.period, CODE_BITS))
        return self.period

    def observe(self, window, t):
        if self.n is None or len(bytes(window)) < CONTEXT_BYTES:
            return
        ph = t % self.period
        z = lag_bits(window)
        self.n[ph] += 1
        d = z - self.mean[ph]
        self.mean[ph] += d / self.n[ph]
        self.m2[ph] += d * (z - self.mean[ph])

    def encode(self, window, t):
        z = lag_bits(window)
        if self.n is not None:
            ph = t % max(1, self.period)
            if self.n[ph] >= 2:
                sd = np.sqrt(self.m2[ph] / (self.n[ph] - 1))
                live = sd > self.floor
                z = np.where(live, (z - self.mean[ph]) / np.where(live, sd, 1.), 0.)
        norm = float(np.linalg.norm(z))
        return z / norm if norm > 1e-12 else z

    def state_bytes(self):
        return 0 if self.n is None else int(self.mean.nbytes + self.m2.nbytes + self.n.nbytes)


FRONTS = {'lag': LagFront, 'phase': PhaseFront}


# ----------------------------------------------------------------------------
# 3. The sparse KC code
# ----------------------------------------------------------------------------

class SparseCode:
    """Fixed sensory code, 96 -> UNITS, K active. No learning in the code.

    `structure='fly'` is V36's SparseCode: `contacts` distinct lag96 coordinates per
    KC, seeded Gaussian contact weights normalised once, tanh drive, top-K by |activity|.
    `structure='hash'` is V36's HASH-O control: the exact context is hashed to a random
    K-subset with independent signed amplitudes, which destroys all input similarity
    and therefore pins the separation ratio at ~1.

    Selection uses argpartition rather than V36's full stable argsort. The selected set
    is identical whenever the K-th and (K+1)-th activities differ, which self_test
    checks; ties occur only on degenerate all-equal drives.
    """

    def __init__(self, seed, front, units=UNITS, k=K_ACTIVE, contacts=5, structure='fly'):
        if structure not in ('fly', 'hash'):
            raise ValueError(structure)
        if not 1 <= k <= units:
            raise ValueError((k, units))
        self.units, self.k, self.structure, self.front = int(units), int(k), structure, front
        self.n_contacts = int(contacts)
        rr = np.random.default_rng(seed + 3604096)
        self.contacts = np.array([rr.choice(CODE_BITS, contacts, replace=False)
                                  for _ in range(units)])
        self.weights = rr.normal(size=(units, contacts))
        self.weights /= np.linalg.norm(self.weights, axis=1, keepdims=True)

    def encode(self, window, t):
        if self.structure == 'hash':
            digest = hashlib.sha256(bytes(window)[-CONTEXT_BYTES:]).digest()[:16]
            rr = np.random.default_rng(int.from_bytes(digest, 'little'))
            # A uniformly random K-subset, drawn as argpartition over uniform draws.
            # Same distribution as choice(replace=False) and ~10x faster per byte,
            # which matters because HASH-O rehashes on every byte by construction.
            ix = np.sort(np.argpartition(rr.random(self.units), self.k - 1)[:self.k])
            vals = (2 * rr.integers(0, 2, self.k) - 1) / math.sqrt(self.k)
            return ix, vals.astype(np.float64)
        z = self.front.encode(window, t)
        u = np.tanh(np.einsum('ij,ij->i', self.weights, z[self.contacts]))
        ix = np.sort(np.argpartition(-np.abs(u), self.k - 1)[:self.k])
        vals = u[ix]
        norm = float(np.linalg.norm(vals))
        return ix, (vals / norm if norm > 1e-12 else np.zeros(self.k))

    def fixed_bytes(self):
        return int(self.contacts.nbytes + self.weights.nbytes)


# ----------------------------------------------------------------------------
# 4. Architectures
# ----------------------------------------------------------------------------

class SuperposedModel:
    """Bounded store: one fixed UNITS x 256 matrix. Every fact writes on top of
    every other; interference is code overlap and nothing else. No shared base
    learner and no slots, so this is the pure superposed comparison."""

    bounded = True

    def __init__(self, seed, code):
        self.code = code
        self.output = np.zeros((code.units, OUTPUTS))
        self.ops = 0
        self.updates = 0

    def scores(self, window, t):
        ix, v = self.code.encode(window, t)
        return v @ self.output[ix]

    def observe(self, window, t, y):
        ix, v = self.code.encode(window, t)
        block = self.output[ix]
        used = v @ block
        if not np.isfinite(used).all():
            raise FloatingPointError('Nonfinite superposed prediction.')
        pre_loss, correct = loss(used, y), int(np.argmax(used) == y)
        vv = float(v @ v)
        if vv > 1e-15:
            error = -used
            error[y] += 1.
            self.output[ix] = block + RATE * np.outer(v / vv, error)
            self.updates += 1
        self.code.front.observe(window, t)
        self.ops += (2 * self.code.k * OUTPUTS + self.code.units * self.code.n_contacts
                     + self.code.units)
        return pre_loss, correct

    def state_hash(self):
        return sha(self.output.tobytes())

    def costs(self):
        return {'bounded': True, 'store_bytes': int(self.output.nbytes),
                'fixed_code_bytes': self.code.fixed_bytes(),
                'front_state_bytes': self.code.front.state_bytes(),
                'slots': None, 'mean_members': None,
                'total_payload_bytes': int(self.output.nbytes + self.code.fixed_bytes()
                                           + self.code.front.state_bytes()),
                'operation_proxy_multiply_add_units': int(self.ops)}


class SlotModel:
    """Unbounded reference: V38's ByteModel, unchanged. Shared RLS base learner plus
    one value vector per slot, addressed by an O(N) similarity scan at read time."""

    bounded = False

    def __init__(self, seed, arm):
        if arm not in V38.MAIN_ARMS:
            raise ValueError(arm)
        self.model = V38.ByteModel(seed, arm)

    def scores(self, window, t):
        return self.model.scores(bytes(window))

    def observe(self, window, t, y):
        pre_loss, correct, _ = self.model.observe(bytes(window), y)
        return pre_loss, correct

    def state_hash(self):
        return self.model.state_hash()

    def costs(self):
        c = self.model.costs()
        return {'bounded': False, 'store_bytes': int(c['local_value_bytes']),
                'fixed_code_bytes': int(c['fixed_representation_bytes']),
                'front_state_bytes': 0,
                'slots': int(c['slots']), 'mean_members': float(c['mean_members']),
                'total_payload_bytes': int(c['total_payload_bytes']),
                'operation_proxy_multiply_add_units': int(c['operation_proxy_multiply_add_units'])}


# ----------------------------------------------------------------------------
# 5. Arms and the grid
# ----------------------------------------------------------------------------

class Cell:
    __slots__ = ('arm', 'load', 'kind', 'front', 'contacts', 'k', 'structure', 'slot_arm', 'block')

    def __init__(self, arm, load, kind, block, front=None, contacts=None, k=K_ACTIVE,
                 structure='fly', slot_arm=None):
        self.arm, self.load, self.kind, self.block = arm, load, kind, block
        self.front, self.contacts, self.k, self.structure = front, contacts, k, structure
        self.slot_arm = slot_arm

    @property
    def name(self):
        return f'{self.arm}@{self.load}'

    @property
    def keys_per_wave(self):
        return self.load // 4

    def build(self, seed):
        if self.kind == 'slot':
            return SlotModel(seed, self.slot_arm)
        front = FRONTS[self.front](seed)
        return SuperposedModel(seed, SparseCode(seed, front, UNITS, self.k,
                                                self.contacts, self.structure))

    def spec(self):
        return {'arm': self.arm, 'load': self.load, 'kind': self.kind, 'block': self.block,
                'front': self.front, 'contacts': self.contacts, 'k': self.k,
                'structure': self.structure, 'slot_arm': self.slot_arm, 'units': UNITS}


# Loads chosen from a 2-seed calibration: at UNITS=4096 the store is already past the
# collapse point for a low-separation code at 64, and past it for every code by 512,
# so a grid starting at 64 would sit entirely in the collapsed regime.
LOADS = (32, 64, 128, 256)
SLOT_LOADS = (32, 64)

# Separation ladder at a fixed coding level (K = 205), so cost per byte is constant
# across the axis and the expensive cells are not also the low-separation cells.
# Calibrated ratios on seed 38101, 32 keys: see SEPARATION.csv for the run's own values.
LADDER = (
    ('S1-lag-c2',    'lag',   2),      # ~1.48x
    ('S2-lag-c12',   'lag',   12),     # ~2.82x
    ('S3-phase-c5',  'phase', 5),      # ~5.11x
    ('S4-phase-c12', 'phase', 12),     # ~5.62x
)
# Supplementary high point. K differs, so it is excluded from the primary contrast.
SUPPLEMENTARY = ('S5-phase-c12-k82', 'phase', 12, 82)


def build_grid(stage):
    cells = []
    for name, front, contacts in LADDER:
        for load in LOADS:
            cells.append(Cell(name, load, 'sparse', 'A', front=front, contacts=contacts))
    for load in LOADS:
        cells.append(Cell('HASH-O', load, 'sparse', 'B', front='lag', contacts=5,
                          structure='hash'))
    name, front, contacts, k = SUPPLEMENTARY
    for load in (LOADS[0], LOADS[-1]):
        cells.append(Cell(name, load, 'sparse', 'B', front=front, contacts=contacts, k=k))
    for load in SLOT_LOADS:
        cells.append(Cell('PRED-V', load, 'slot', 'C', slot_arm='PRED-V'))
        cells.append(Cell('EXACT-V', load, 'slot', 'D', slot_arm='EXACT-V'))
    if stage == 'g2':
        # G0, G1 and G2 only: the extremes of the ladder at the extremes of the load
        # axis. This is the cheapest set that can void the design, so it runs first.
        keep = {LADDER[0][0], LADDER[-1][0]}
        cells = [c for c in cells if c.arm in keep and c.load in (LOADS[0], LOADS[-1])]
    return cells


# ----------------------------------------------------------------------------
# 6. Geometry: the separation the run actually achieved
# ----------------------------------------------------------------------------

def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.


def separation(model, stream, t_phase, sample=32):
    """Same-key (different rendering) versus different-key overlap.

    Jaccard on active sets for the sparse codes -- the fly-comparable metric, and the
    one ChatGPT's V36 preflight gate was written against. Cosine on the dense front-end
    code for every arm, so the slot arms have a number too. The two are different
    metrics and are never compared to each other numerically.
    """
    keys = stream['keys'][:min(sample, stream['keys_per_wave'])]
    out = {'jaccard_same': None, 'jaccard_diff': None, 'jaccard_ratio': None,
           'cosine_same': None, 'cosine_diff': None, 'cosine_ratio': None}
    if not keys:
        return out
    if isinstance(model, SuperposedModel):
        sets, vecs = {}, {}
        for i, key in enumerate(keys):
            for a in (0, 1, 2):
                ix, _ = model.code.encode(context(key, a), t_phase)
                sets[(i, a)] = set(ix.tolist())
                if model.code.structure == 'fly':
                    vecs[(i, a)] = model.code.front.encode(context(key, a), t_phase)
        same = [jaccard(sets[(i, 0)], sets[(i, a)]) for i in range(len(keys)) for a in (1, 2)]
        diff = [jaccard(sets[(i, 0)], sets[(j, 0)])
                for i in range(len(keys)) for j in range(len(keys)) if i != j]
        out.update(jaccard_same=float(np.mean(same)), jaccard_diff=float(np.mean(diff)))
        if out['jaccard_diff'] and out['jaccard_diff'] > 0:
            out['jaccard_ratio'] = out['jaccard_same'] / out['jaccard_diff']
    else:
        vecs = {(i, a): model.model.address_encoder.encode(context(key, a))[0]
                for i, key in enumerate(keys) for a in (0, 1, 2)}
    if vecs:
        cs = [float(vecs[(i, 0)] @ vecs[(i, a)]) for i in range(len(keys)) for a in (1, 2)]
        cd = [float(vecs[(i, 0)] @ vecs[(j, 0)])
              for i in range(len(keys)) for j in range(len(keys)) if i != j]
        out.update(cosine_same=float(np.mean(cs)), cosine_diff=float(np.mean(cd)))
        if out['cosine_diff'] and abs(out['cosine_diff']) > 1e-9:
            out['cosine_ratio'] = out['cosine_same'] / out['cosine_diff']
    return out


# ----------------------------------------------------------------------------
# 7. One trajectory
# ----------------------------------------------------------------------------

class Group:
    __slots__ = ('n', 'bpb', 'correct')

    def __init__(self):
        self.n = self.bpb = self.correct = 0.

    def add(self, bpb, correct):
        self.n += 1
        self.bpb += bpb
        self.correct += correct

    def result(self):
        return {'n': int(self.n), 'bpb': self.bpb / self.n if self.n else None,
                'accuracy': self.correct / self.n if self.n else None}


def run_trial(seed, cell, smoke=False, warmup=WARMUP_BYTES):
    start = time.perf_counter()
    stream = make_stream(seed, keys_per_wave=cell.keys_per_wave, smoke=smoke)
    model = cell.build(seed)
    raw, rows = stream['raw'], stream['rows']

    # Label-free period discovery over bytes the model has already streamed.
    if isinstance(model, SuperposedModel):
        model.code.front.discover(raw[:min(warmup, len(raw))])

    groups = collections.defaultdict(Group)
    probes = []
    checkpoints = collections.defaultdict(list)
    for cp in stream['checkpoints']:
        checkpoints[cp['record_count']].append(cp)
    history = bytearray()
    t = 0
    last_target_t = CONTEXT_BYTES        # phase of a target byte; refreshed as they occur

    def probe(name, kind, ids, aliases=(0, 1), revised=False):
        before = model.state_hash()
        for i in ids:
            y = (stream['revisions'][i] if revised and i in stream['revisions']
                 else int(stream['labels'][i]))
            for a in aliases:
                s = model.scores(context(stream['keys'][i], a), last_target_t)
                probes.append({'checkpoint': name, 'kind': kind, 'key_id': int(i), 'alias': int(a),
                               'accuracy': float(int(np.argmax(s)) == y),
                               'target_probability': float(softmax(s)[y]),
                               'bpb': float(loss(s, y))})
        if model.state_hash() != before:
            raise AssertionError(f'Probe mutated model state at {name}')

    emerged = {}
    for number, row in enumerate(rows, 1):
        for pos, y in enumerate(row['raw']):
            pre_loss, correct = model.observe(history, t, y)
            groups['whole'].add(pre_loss, correct)
            groups['target' if pos == CONTEXT_BYTES else 'nontarget'].add(pre_loss, correct)
            if pos == CONTEXT_BYTES:
                last_target_t = t
            history.append(y)
            if len(history) > CONTEXT_BYTES:
                del history[0]
            t += 1
            # E1/E2 by stream position, so both architectures compute the same thing.
            # E1: alias 0 of this key has been trained, alias 1 has not. E2: both have.
            if pos == CONTEXT_BYTES and row['kind'] == 'acquisition':
                i, phase = row['key_id'], emerged.setdefault(row['key_id'], set())
                if row['exposure'] == 1 and 'E1' not in phase:
                    phase.add('E1')
                    probe('E1', 'emergence', [i], (2,))
                elif row['exposure'] == 2 and 'E2' not in phase:
                    phase.add('E2')
                    probe('E2', 'emergence', [i], (2,))
        for cp in checkpoints.get(number, []):
            name, wi, k = cp['name'], ord(cp['wave']) - 65, stream['keys_per_wave']
            if cp['type'] == 'wave':
                probe(name, 'untouched_A', stream['untouched'], aliases=(0, 1))
                probe(name, 'untouched_A_heldout', stream['untouched'], aliases=(2,))
                probe(name, 'current', range(wi * k, (wi + 1) * k))
                if cp['wave'] == 'A':
                    probe(name, 'all_A', range(k))
            elif cp['type'] == 'revision':
                probe(name, 'revision', stream['revised'], aliases=(0, 1, 2), revised=True)

    geometry = separation(model, stream, last_target_t)
    costs = model.costs()
    byte_events = len(raw)
    result = {
        'seed': seed, 'cell': cell.name, 'spec': cell.spec(), 'smoke': smoke,
        'stream_sha256': stream['raw_sha256'], 'byte_events': byte_events,
        'target_events': len(rows), 'keys': len(stream['keys']),
        'groups': {g: s.result() for g, s in groups.items()},
        'geometry': geometry, 'costs': costs,
        'ops_per_byte': costs['operation_proxy_multiply_add_units'] / byte_events,
        'period': getattr(getattr(model, 'code', None), 'front', None)
                  and getattr(model.code.front, 'period', None),
        'probes': probes, 'elapsed_seconds': time.perf_counter() - start,
    }
    result['metrics'] = metrics(result)
    return result


def probe_mean(probes, checkpoint, kind, field='accuracy', aliases=(0, 1)):
    vals = [p[field] for p in probes
            if p['checkpoint'] == checkpoint and p['kind'] == kind and p['alias'] in aliases]
    return float(np.mean(vals)) if vals else None


def metrics(result):
    p, g = result['probes'], result['groups']
    m = {f'{key}_bpb': g[key]['bpb'] for key in ('whole', 'target', 'nontarget') if key in g}
    m['target_accuracy'] = g['target']['accuracy'] if 'target' in g else None
    m['A_initial'] = probe_mean(p, 'after_A', 'all_A')
    for w in 'ABCD':
        m[f'A_after_{w}'] = probe_mean(p, f'after_{w}', 'untouched_A')
    m['D_current'] = probe_mean(p, 'after_D', 'current')
    for w in 'ABCD':
        m[f'A_after_{w}_heldout'] = probe_mean(p, f'after_{w}', 'untouched_A_heldout',
                                               aliases=(2,))
    # The V35 wall as one number, at matched delay: an arm scores only if it holds the
    # trained renderings AND the never-trained one after three intervening waves.
    r, h = m.get('A_after_D'), m.get('A_after_D_heldout')
    m['joint_min'] = None if r is None or h is None else min(r, h)
    m['retention_ratio'] = (None if not m.get('A_initial') or m.get('A_after_D') is None
                            else m['A_after_D'] / m['A_initial'])
    for ex in (1, 2, 4, 8):
        for a in range(3):
            m[f'revision_{ex}_alias_{a}'] = probe_mean(p, f'revision_{ex}', 'revision',
                                                       aliases=(a,))
    for e in ('E1', 'E2'):
        m[f'{e}_heldout_accuracy'] = probe_mean(p, e, 'emergence', aliases=(2,))
    m.update({f'separation_{k}': v for k, v in result['geometry'].items()})
    m['total_payload_bytes'] = result['costs']['total_payload_bytes']
    m['ops_per_byte'] = result['ops_per_byte']
    m['slots'] = result['costs']['slots']
    return m


# ----------------------------------------------------------------------------
# 8. Gates
# ----------------------------------------------------------------------------

G0_ACQUISITION = 0.90      # an arm must learn wave A AT THE LOAD WHERE IT IS COMPARED
G1_SPAN = 2.0              # the separation axis must span at least 2x
G2_PRESSURE = 0.10         # load must cost the best arm at least 10 pp on the joint endpoint
G3_MARGIN = 0.10           # the high-ratio arm must beat both corners by 10 pp
PRIMARY = 'joint_min'      # min(retention, held-out transfer), both at wave-D delay


def cell_level(per_seed, cell, key):
    return interval([per_seed[cell][s].get(key) for s in sorted(per_seed.get(cell, {}))])


def evaluate(per_seed, cells, stage):
    """Gates are ordered; a failure voids the levels beneath it.

    The primary endpoint is `joint_min` = min(retention, held-out transfer), both
    measured on the untouched wave-A cohort after wave D so they share a delay. A
    single-component endpoint cannot express the claim: retention alone is maximised
    by destroying input similarity (HASH-O), and transfer alone is maximised by
    destroying separation (S1). Only a code with a high same/different ratio holds
    both, which is the V35 four-way wall restated inside a bounded store.
    """
    names = [c.name for c in cells]
    ladder = [n for n, _, _ in LADDER]
    present = {c.arm for c in cells}
    loads = sorted({c.load for c in cells})
    lo_load, hi_load = loads[0], loads[-1]
    gates = {}

    # G0 -- acquisition validity, evaluated at EVERY load. An arm that cannot acquire
    # at a load is excluded at that load; scoring retention for it would repeat V31's
    # error of rewarding arms that never learned.
    g0 = {}
    for arm in sorted(present):
        for load in loads:
            cell = f'{arm}@{load}'
            if cell in per_seed:
                lv = cell_level(per_seed, cell, 'A_initial')
                g0[cell] = {'level': lv, 'pass': bool(lv['mean'] is not None
                                                      and lv['mean'] >= G0_ACQUISITION)}
    valid = {c for c, v in g0.items() if v['pass']}
    gates['G0'] = {'threshold': G0_ACQUISITION, 'per_cell': g0,
                   'valid_cells': sorted(valid),
                   'pass': bool(g0) and any(v['pass'] for v in g0.values()),
                   'note': 'Checked at every load, not only the lowest. Cells below the '
                           'threshold are invalid_acquisition and enter no contrast.'}

    # G1 -- the separation axis is real, measured at the lowest load.
    ratios = {arm: cell_level(per_seed, f'{arm}@{lo_load}', 'separation_jaccard_ratio')
              for arm in sorted(present) if f'{arm}@{lo_load}' in per_seed}
    usable = {a: r for a, r in ratios.items() if r['mean'] is not None and a in ladder}
    low_arm = high_arm = None
    if len(usable) >= 2:
        low_arm = min(usable, key=lambda a: usable[a]['mean'])
        high_arm = max(usable, key=lambda a: usable[a]['mean'])
        span = usable[high_arm]['mean'] / usable[low_arm]['mean']
        gap = paired(per_seed, f'{high_arm}@{lo_load}', f'{low_arm}@{lo_load}',
                     'separation_jaccard_ratio')
        gates['G1'] = {'threshold': G1_SPAN, 'low_arm': low_arm, 'high_arm': high_arm,
                       'span': span, 'paired_gap': gap, 'ratios': ratios,
                       'pass': bool(span >= G1_SPAN and gap['low'] is not None and gap['low'] > 0)}
    else:
        gates['G1'] = {'threshold': G1_SPAN, 'ratios': ratios, 'pass': False,
                       'note': 'Fewer than two ladder arms produced a Jaccard ratio.'}

    # G2 -- capacity pressure exists. Measured on the arm with the most to lose, which
    # is the one with the best joint score at the lowest load. Testing the weakest arm
    # instead cannot work: it is already at the floor and has no room to fall.
    best = None
    # Only arms that actually run at BOTH extremes can be scored for a change across
    # the load range. The slot arms run at SLOT_LOADS only, so including them here
    # selects an arm that has no high-load cell and voids the gate spuriously.
    spanning = [a for a in present
                if f'{a}@{lo_load}' in per_seed and f'{a}@{hi_load}' in per_seed]
    scored = {a: cell_level(per_seed, f'{a}@{lo_load}', PRIMARY)['mean']
              for a in spanning if f'{a}@{lo_load}' in valid}
    scored = {a: v for a, v in scored.items() if v is not None}
    if scored:
        best = max(scored, key=scored.get)
    if best and f'{best}@{hi_load}' in per_seed and lo_load != hi_load:
        drop = paired(per_seed, f'{best}@{hi_load}', f'{best}@{lo_load}', PRIMARY)
        gates['G2'] = {'threshold': -G2_PRESSURE, 'arm': best, 'endpoint': PRIMARY,
                       'loads': [lo_load, hi_load], 'change': drop,
                       'pass': bool(drop['mean'] is not None and drop['mean'] <= -G2_PRESSURE
                                    and drop['high'] is not None and drop['high'] < 0)}
    else:
        gates['G2'] = {'threshold': -G2_PRESSURE, 'pass': False,
                       'spanning_arms': spanning,
                       'note': 'No arm both acquires at the lowest load and has a cell '
                               'at the highest load.'}

    # G3 -- primary: at the largest load where the high-ratio arm and both corners are
    # valid, the high-ratio arm must beat BOTH corners on the joint endpoint.
    corners = [a for a in ('HASH-O',) if a in present] + ([low_arm] if low_arm else [])
    target_load, contrasts = None, {}
    if high_arm and corners:
        for load in reversed(loads):
            need = [f'{a}@{load}' for a in [high_arm] + corners]
            if all(c in valid for c in need):
                target_load = load
                break
    if target_load is not None:
        contrasts = {a: paired(per_seed, f'{high_arm}@{target_load}', f'{a}@{target_load}',
                               PRIMARY) for a in corners}
        ok = all(d['mean'] is not None and d['mean'] >= G3_MARGIN
                 and d['low'] is not None and d['low'] > 0 for d in contrasts.values())
        gates['G3'] = {'margin': G3_MARGIN, 'endpoint': PRIMARY, 'high_arm': high_arm,
                       'corners': corners, 'load': target_load, 'contrasts': contrasts,
                       'components': {f'{a}@{target_load}': {
                           'retention': cell_level(per_seed, f'{a}@{target_load}', 'A_after_D'),
                           'heldout': cell_level(per_seed, f'{a}@{target_load}', 'A_after_D_heldout'),
                           'joint': cell_level(per_seed, f'{a}@{target_load}', PRIMARY)}
                           for a in [high_arm] + corners},
                       'pass': bool(ok and gates['G1']['pass'] and gates['G2']['pass'])}
    else:
        gates['G3'] = {'margin': G3_MARGIN, 'pass': False,
                       'note': 'No load where the high-ratio arm and both corners acquire.'}

    # G3b -- capacity curve, descriptive: the joint score at every valid load.
    gates['G3b'] = {'endpoint': PRIMARY, 'curves': {
        arm: {load: cell_level(per_seed, f'{arm}@{load}', PRIMARY)
              for load in loads if f'{arm}@{load}' in per_seed}
        for arm in sorted(present)}}

    # G4 -- the two frontier corners, reported as levels rather than as a control.
    gates['G4'] = {'note': 'HASH-O destroys input similarity (ratio ~1) and is the '
                           'retention corner; the lowest-contact code destroys separation '
                           'and is the transfer corner. Neither is a matched control; '
                           'both are frontier endpoints.',
                   'corners': {a: {'ratio': ratios.get(a),
                                   'retention': cell_level(per_seed, f'{a}@{lo_load}', 'A_after_D'),
                                   'heldout': cell_level(per_seed, f'{a}@{lo_load}', 'A_after_D_heldout')}
                               for a in corners if f'{a}@{lo_load}' in per_seed}}

    # G5 -- cost accounting, descriptive.
    gates['G5'] = {'per_cell': {n: {'total_payload_bytes': cell_level(per_seed, n, 'total_payload_bytes'),
                                    'ops_per_byte': cell_level(per_seed, n, 'ops_per_byte'),
                                    'slots': cell_level(per_seed, n, 'slots')}
                                for n in names if n in per_seed}}
    gates['stage'] = stage
    gates['verdict'] = verdict(gates)
    return gates


def verdict(gates):
    if not gates['G0']['pass']:
        return 'invalid_acquisition'
    if not gates['G1']['pass']:
        return 'separation_axis_void'
    if not gates['G2']['pass']:
        return 'no_capacity_pressure'
    if gates['stage'] == 'g2':
        return 'gates_cleared_run_full_grid'
    g3 = gates['G3']
    if g3['pass']:
        return 'qualified'
    ds = list((g3.get('contrasts') or {}).values())
    if not ds or any(d['mean'] is None for d in ds):
        return 'invalid_acquisition'
    if all(d['mean'] > 0 and (d['low'] or -1) > 0 for d in ds):
        return 'supported_below_qualification'
    if all(d['mean'] > 0 for d in ds):
        return 'favorable_unresolved_precision'
    if any((d['high'] or 1) < 0 for d in ds):
        return 'adverse'
    return 'no_mean_advantage'


# ----------------------------------------------------------------------------
# 9. Outputs
# ----------------------------------------------------------------------------

def fmt(s, places=4):
    if not s or s.get('mean') is None:
        return '--'
    if s.get('low') is None:
        return f'{s["mean"]:+.{places}f}'
    return f'{s["mean"]:+.{places}f} [{s["low"]:+.{places}f}, {s["high"]:+.{places}f}]'


def write_outputs(out_dir, cells, results, per_seed, gates, manifest):
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ['seed', 'cell', 'arm', 'load', 'kind', 'front', 'contacts', 'k', 'structure',
              'bounded', 'A_initial', 'A_after_A', 'A_after_D', 'A_after_D_heldout',
              'joint_min', 'retention_ratio', 'D_current',
              'revision_4_alias_0', 'revision_4_alias_2', 'E1_heldout_accuracy',
              'E2_heldout_accuracy', 'target_accuracy', 'target_bpb', 'whole_bpb',
              'separation_jaccard_same', 'separation_jaccard_diff', 'separation_jaccard_ratio',
              'separation_cosine_ratio', 'total_payload_bytes', 'ops_per_byte', 'slots',
              'period', 'elapsed_seconds']
    with (out_dir / 'TERMINAL.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sorted(results, key=lambda r: (r['spec']['block'], r['cell'], r['seed'])):
            row = {'seed': r['seed'], 'cell': r['cell'], 'arm': r['spec']['arm'],
                   'load': r['spec']['load'], 'kind': r['spec']['kind'],
                   'front': r['spec']['front'], 'contacts': r['spec']['contacts'],
                   'k': r['spec']['k'], 'structure': r['spec']['structure'],
                   'bounded': r['costs']['bounded'], 'period': r['period'],
                   'elapsed_seconds': round(r['elapsed_seconds'], 2)}
            row.update({k: r['metrics'].get(k) for k in fields if k not in row})
            w.writerow(row)
    (out_dir / 'GATES.json').write_text(json.dumps(gates, indent=1, default=str))
    (out_dir / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1, default=str))
    (out_dir / 'PER_SEED.json').write_text(json.dumps(per_seed, indent=1, default=str))
    (out_dir / 'REPORT.md').write_text(report(cells, per_seed, gates, manifest))


def report(cells, per_seed, gates, manifest):
    L = ['# V39 results — bounded superposed memory', '',
         f"**Source SHA256** `{manifest['identity']['source_sha256']}`  ",
         f"**Stage** `{gates['stage']}` · **seeds** {len(manifest['plan']['seeds'])} · "
         f"**verdict** **`{gates['verdict']}`**", '',
         'Label-free: the only teaching signal is the next-byte residual. Workload is '
         "V38's, extended with a load axis and byte-identical to V38 at V38's settings.",
         '',
         'Primary endpoint `joint_min` = min(retention, held-out transfer), both on the '
         'untouched wave-A cohort after wave D, so the two share a delay.', '']

    L += ['## The frontier', '',
          'Retention tracks different-key overlap; transfer tracks same-key overlap. A '
          'single-component endpoint is maximised by destroying one of them.', '',
          '| cell | J_same | J_diff | ratio | A initial | retention | held-out | '
          '**joint** | payload | ops/byte |',
          '|---|---:|---:|---:|---|---|---|---|---:|---:|']
    for c in sorted(cells, key=lambda c: (c.block, c.arm, c.load)):
        if c.name not in per_seed:
            continue
        g = lambda k: cell_level(per_seed, c.name, k)
        num = lambda k, d=3: ('--' if g(k)['mean'] is None else f"{g(k)['mean']:.{d}f}")
        pay, ops = g('total_payload_bytes')['mean'], g('ops_per_byte')['mean']
        L.append(f"| `{c.name}` | {num('separation_jaccard_same')} | "
                 f"{num('separation_jaccard_diff')} | {num('separation_jaccard_ratio', 2)} | "
                 f"{fmt(g('A_initial'), 3)} | {fmt(g('A_after_D'), 3)} | "
                 f"{fmt(g('A_after_D_heldout'), 3)} | **{fmt(g('joint_min'), 3)}** | "
                 f"{(pay or 0)/1024/1024:.1f} MB | {(ops or 0):,.0f} |")

    L += ['', '## Gates', '']
    g0 = gates['G0']
    L += [f"### G0 — acquisition validity (≥ {g0['threshold']:.2f}, at every load) · "
          f"**{'pass' if g0['pass'] else 'FAIL'}**", '',
          '| cell | A initial | valid |', '|---|---|:-:|']
    for cell, v in sorted(g0['per_cell'].items()):
        L.append(f"| `{cell}` | {fmt(v['level'], 3)} | {v['pass']} |")
    L += ['', g0['note'], '']

    g1 = gates['G1']
    L += [f"### G1 — separation axis is real (span ≥ {g1['threshold']}×) · "
          f"**{'pass' if g1['pass'] else 'FAIL'}**", '']
    if 'span' in g1:
        L += [f"Span **{g1['span']:.2f}×** from `{g1['low_arm']}` to `{g1['high_arm']}`; "
              f"paired gap {fmt(g1['paired_gap'], 3)}.", '',
              '| arm | Jaccard ratio |', '|---|---|']
        for arm, r in sorted(g1['ratios'].items(),
                             key=lambda kv: (kv[1]['mean'] is None, kv[1]['mean'] or 0)):
            L.append(f"| {arm} | {fmt(r, 3)} |")
    else:
        L.append(g1.get('note', ''))

    g2 = gates['G2']
    L += ['', f"### G2 — capacity pressure exists · **{'pass' if g2['pass'] else 'FAIL'}**", '']
    if 'change' in g2:
        L += [f"`{g2['arm']}` (the best arm at the lowest load, so the one with the most "
              f"to lose) on `{g2['endpoint']}`, load {g2['loads'][0]} → {g2['loads'][1]}: "
              f"**{fmt(g2['change'])}**, required ≤ {g2['threshold']:+.2f} with the upper "
              f"bound below zero.", '',
              "If load hurts nobody there is no pressure and every contrast below is void. "
              "This is V27's headroom screen applied to capacity."]
    else:
        L.append(g2.get('note', ''))

    g3 = gates['G3']
    L += ['', f"### G3 — primary: the high-ratio code holds both corners' properties · "
          f"**{'pass' if g3['pass'] else 'FAIL'}**", '']
    if 'contrasts' in g3:
        L += [f"At load **{g3['load']}**, the largest where every compared cell acquires. "
              f"`{g3['high_arm']}` must beat each corner on `{g3['endpoint']}` by "
              f"≥ {g3['margin']:.2f} with the paired lower bound above zero.", '',
              '| contrast | joint_min difference |', '|---|---|']
        for arm, d in sorted(g3['contrasts'].items()):
            L.append(f"| `{g3['high_arm']}` − `{arm}` | {fmt(d)} |")
        L += ['', '| cell | retention | held-out | joint |', '|---|---|---|---|']
        for cell, v in g3['components'].items():
            L.append(f"| `{cell}` | {fmt(v['retention'], 3)} | {fmt(v['heldout'], 3)} | "
                     f"{fmt(v['joint'], 3)} |")
    else:
        L.append(g3.get('note', ''))

    L += ['', '### G3b — capacity curves (descriptive)', '',
          f"`{gates['G3b']['endpoint']}` by load.", '']
    loads = sorted({l for cur in gates['G3b']['curves'].values() for l in cur})
    L += ['| arm | ' + ' | '.join(str(l) for l in loads) + ' |',
          '|---|' + '---:|' * len(loads)]
    for arm, cur in sorted(gates['G3b']['curves'].items()):
        cells_ = [(f"{cur[l]['mean']:.3f}" if l in cur and cur[l]['mean'] is not None else '--')
                  for l in loads]
        L.append(f"| {arm} | " + ' | '.join(cells_) + ' |')

    L += ['', '### G4 — the two frontier corners (descriptive)', '', gates['G4']['note'], '',
          '| corner | ratio | retention | held-out |', '|---|---|---|---|']
    for arm, v in sorted(gates['G4']['corners'].items()):
        L.append(f"| {arm} | {fmt(v['ratio'], 3)} | {fmt(v['retention'], 3)} | "
                 f"{fmt(v['heldout'], 3)} |")

    L += ['', '## Reference points', '', '| quantity | value |', '|---|---:|',
          f"| fly KC overlap, same odour (real wiring, 5% coding) | {REFERENCE['fly_kc_overlap_same']} |",
          f"| fly KC overlap, different odour | {REFERENCE['fly_kc_overlap_diff']} |",
          f"| **fly separation ratio** | **{REFERENCE['fly_ratio']}×** |",
          f"| ChatGPT V36 preflight gate | ratio ≥ {REFERENCE['v36_chatgpt_gate']['ratio_min']}, "
          f"J_diff ≤ {REFERENCE['v36_chatgpt_gate']['j_diff_max']} |",
          f"| ChatGPT V36 measured (lag96 front end) | "
          f"{REFERENCE['v36_chatgpt_measured']['ratio_range'][0]}–"
          f"{REFERENCE['v36_chatgpt_measured']['ratio_range'][1]}× |",
          f"| V38 PRED-V slots after D (unbounded) | {REFERENCE['v38_pred_v_slots_after_D']} |",
          f"| V38 E1 / E2 held-out transfer | {REFERENCE['v38_e1_heldout']} / {REFERENCE['v38_e2_heldout']} |",
          '', '## Interpretation limits', '',
          '- G0, G1 and G2 are hard gates in that order. A failure at any level voids the',
          '  levels beneath it; running mechanism contrasts under a void gate is how V27',
          '  and V31 produced uninterpretable results.',
          '- Retention alone is the wrong endpoint. It is maximised by destroying input',
          '  similarity, which HASH-O does by construction, at the cost of all transfer.',
          '  `joint_min` is used precisely so that neither corner can win by giving up the',
          '  other property.',
          '- The superposed arms carry no shared base learner, so their whole-stream bpb is',
          '  not comparable to the slot arms\'. Retention and transfer are the endpoints.',
          '- Jaccard on sparse active sets and cosine on dense codes are different metrics.',
          '  They are reported side by side and never compared numerically.',
          '- Revision endpoints are measured shortly after the revision and are therefore',
          '  recency-protected; the primary endpoint deliberately uses the wave-D delay for',
          '  both of its components instead.',
          '- Storage is bounded only for the superposed arms. The slot arms are the',
          '  unbounded reference and run only at the two lowest loads.',
          '- Synthetic stream with controlled renderings. No scaling or language claim.']
    return '\n'.join(L) + '\n'


# ----------------------------------------------------------------------------
# 10. Self-test
# ----------------------------------------------------------------------------

def self_test(full=False):
    checks = {}

    # 1. The workload is V38's, exactly, at V38's own settings.
    for seed in (38101, 38105):
        assert make_stream(seed)['raw_sha256'] == V38.make_stream(seed)['raw_sha256'], seed
        assert make_stream(seed, smoke=True)['raw_sha256'] == \
            V38.make_stream(seed, smoke=True)['raw_sha256'], seed
    checks['stream_matches_v38'] = True

    # 2. The load axis changes only the load.
    sizes = {}
    for load in LOADS:
        count = load // 4
        s = make_stream(39101, keys_per_wave=count)
        assert len(s['keys']) == load, (load, len(s['keys']))
        # nrev scales with the load, so the untouched retention cohort stays half of
        # wave A at every load. Only the load itself varies along the axis.
        assert len(s['revised']) == max(1, count // 4) == len(s['noisy'])
        assert len(s['untouched']) == count - 2 * max(1, count // 4)
        assert 2 * len(s['untouched']) == count or count < 4
        per_key = collections.Counter(r['key_id'] for r in s['rows'] if r['kind'] == 'acquisition')
        assert set(per_key.values()) == {32}, sorted(set(per_key.values()))
        sizes[load] = len(s['raw'])
    assert sorted(sizes.values()) == [sizes[l] for l in LOADS]
    checks['load_axis'] = sizes

    # 3. Period discovery is label-free and finds the record length.
    raw = make_stream(39101)['raw']
    front = PhaseFront(39101)
    checks['discovered_period'] = front.discover(raw[:WARMUP_BYTES])
    assert checks['discovered_period'] == 14, checks['discovered_period']

    # 4. argpartition selects the same set as V36's stable argsort when there is no tie.
    rr = np.random.default_rng(0)
    for _ in range(20):
        u = rr.normal(size=UNITS)
        a = set(np.argsort(-np.abs(u), kind='stable')[:K_ACTIVE].tolist())
        b = set(np.argpartition(-np.abs(u), K_ACTIVE - 1)[:K_ACTIVE].tolist())
        assert a == b
    checks['selection_matches_argsort'] = True

    # 5. Separation ordering: phase front end beats lag at matched contacts and K.
    stream = make_stream(39101)
    ratios = {}
    for name, kind, contacts, structure in (('lag', 'lag', 12, 'fly'),
                                            ('phase', 'phase', 12, 'fly'),
                                            ('hash', 'lag', 5, 'hash')):
        f = FRONTS[kind](39101)
        if kind == 'phase':
            f.discover(stream['raw'][:WARMUP_BYTES])
            for t in range(CONTEXT_BYTES, min(len(stream['raw']), 20000)):
                f.observe(stream['raw'][t - CONTEXT_BYTES:t], t)
        m = SuperposedModel(39101, SparseCode(39101, f, UNITS, K_ACTIVE, contacts, structure))
        ratios[name] = separation(m, stream, CONTEXT_BYTES)['jaccard_ratio']
    assert ratios['phase'] > ratios['lag'] > 1.2, ratios
    # The hash control destroys input similarity, so same-key and different-key overlap
    # coincide. This is what makes it the separation floor rather than a matched control.
    assert abs(ratios['hash'] - 1.) < 0.35, ratios
    checks['separation_ratios'] = ratios

    # 6. The bounded store does not grow with load, and the slot store does.
    small = run_trial(39101, Cell('S4-phase-c12', 64, 'sparse', 'A', front='phase', contacts=12),
                      smoke=True)
    assert small['costs']['bounded'] and small['costs']['slots'] is None
    assert small['costs']['store_bytes'] == UNITS * OUTPUTS * 8
    slot = run_trial(39101, Cell('PRED-V', 64, 'slot', 'C', slot_arm='PRED-V'), smoke=True)
    assert not slot['costs']['bounded'] and slot['costs']['slots'] > 0
    checks['bounded_store_bytes'] = small['costs']['store_bytes']
    checks['slot_store_slots'] = slot['costs']['slots']

    # 7. Probes are read-only (run_trial asserts this internally) and the delta rule learns.
    assert small['metrics']['A_initial'] is not None
    assert small['groups']['target']['bpb'] < 8.0, small['groups']['target']['bpb']
    checks['smoke_target_bpb'] = small['groups']['target']['bpb']

    # 8. E1/E2 are computed for both architectures.
    for r in (small, slot):
        assert r['metrics']['E1_heldout_accuracy'] is not None
        assert r['metrics']['E2_heldout_accuracy'] is not None
        assert r['metrics']['A_after_D_heldout'] is not None
        assert r['metrics']['joint_min'] == min(r['metrics']['A_after_D'],
                                                r['metrics']['A_after_D_heldout'])
    checks['emergence_probes'] = True
    checks['joint_min_smoke'] = {'sparse': small['metrics']['joint_min'],
                                 'slot': slot['metrics']['joint_min']}

    # 9. Determinism.
    again = run_trial(39101, Cell('S4-phase-c12', 64, 'sparse', 'A', front='phase', contacts=12),
                      smoke=True)
    assert again['metrics'] == small['metrics']
    checks['deterministic'] = True

    if full:
        t0 = time.perf_counter()
        r = run_trial(39101, Cell('S4-phase-c12', 64, 'sparse', 'A', front='phase', contacts=12))
        checks['full_length'] = {'seconds': round(time.perf_counter() - t0, 1),
                                 'bytes': r['byte_events'],
                                 'A_initial': r['metrics']['A_initial'],
                                 'A_after_D': r['metrics']['A_after_D'],
                                 'separation': r['metrics']['separation_jaccard_ratio'],
                                 'target_bpb': r['metrics']['target_bpb']}
    return checks


# ----------------------------------------------------------------------------
# 11. Runner
# ----------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--verify-full', action='store_true',
                        help='Self-test plus one full-length engineering trajectory.')
    parser.add_argument('--smoke', action='store_true',
                        help='Tiny engineering run; never scientific evidence.')
    parser.add_argument('--stage', choices=('g2', 'full'), default='full',
                        help="'g2' runs only the cells that can void the design.")
    parser.add_argument('--out-dir', '--out', type=Path, default=Path('v39_results'))
    parser.add_argument('--seeds', type=int, help='Override the seed count (non-scientific).')
    parser.add_argument('--max-new-jobs', type=int)
    args = parser.parse_args(argv)

    if args.self_test or args.verify_full:
        print(json.dumps(self_test(args.verify_full), indent=2, default=str))
        return

    seeds = list(SEEDS[:args.seeds] if args.seeds else SEEDS)
    scientific = not args.smoke and not args.seeds and args.stage == 'full'
    cells = build_grid(args.stage)
    out_dir = args.out_dir
    jobs_dir = out_dir / 'jobs'
    jobs_dir.mkdir(parents=True, exist_ok=True)

    plan = {'seeds': seeds, 'cells': [c.spec() for c in cells], 'stage': args.stage,
            'smoke': args.smoke, 'units': UNITS, 'coding': CODING, 'k_active': K_ACTIVE,
            'loads': list(LOADS), 'slot_loads': list(SLOT_LOADS), 'rate': RATE,
            'ladder': [list(x) for x in LADDER]}
    identity = {'source_sha256': sha_file(SOURCE), 'v38_sha256': sha_file(V38_SOURCE),
                'plan_sha256': sha(plan), 'version': VERSION}
    manifest = {'version': VERSION, 'identity': identity, 'plan': plan,
                'scientific': scientific,
                'environment': {'python': sys.version, 'numpy': np.__version__,
                                'platform': platform.platform(), 'blas_threads': 1}}

    total = len(cells) * len(seeds)
    print(f'V39 {args.stage}: {len(cells)} cells x {len(seeds)} seeds = {total} trajectories'
          f"{'' if scientific else '  (ENGINEERING ONLY)'}", flush=True)

    results, new, reused = [], 0, 0
    for cell in cells:
        for seed in seeds:
            path = jobs_dir / f'{seed}_{cell.name.replace("@", "_at_")}.json'
            if path.exists():
                cached = json.loads(path.read_text())
                if cached.get('identity') == identity:
                    results.append(cached['result'])
                    reused += 1
                    continue
            if args.max_new_jobs is not None and new >= args.max_new_jobs:
                print(f'Paused at --max-new-jobs={args.max_new_jobs}.', flush=True)
                break
            t0 = time.perf_counter()
            r = run_trial(seed, cell, smoke=args.smoke)
            tmp = path.with_suffix('.tmp')
            tmp.write_text(json.dumps({'identity': identity, 'result': r}, default=str))
            tmp.replace(path)
            results.append(r)
            new += 1
            print(f'  {cell.name:22s} seed {seed}  '
                  f"A_after_D={r['metrics']['A_after_D']:.3f}  "
                  f"sep={r['metrics']['separation_jaccard_ratio'] or float('nan'):.2f}  "
                  f'{time.perf_counter()-t0:.1f}s  [{new+reused}/{total}]', flush=True)
        else:
            continue
        break

    per_seed = collections.defaultdict(dict)
    for r in results:
        per_seed[r['cell']][r['seed']] = r['metrics']
    per_seed = {k: v for k, v in per_seed.items()}
    gates = evaluate(per_seed, cells, args.stage)
    manifest['status'] = {'completed': len(results), 'new': new, 'reused': reused,
                          'expected': total}
    write_outputs(out_dir, cells, results, per_seed, gates, manifest)
    print(f"\nVerdict: {gates['verdict']}\nWrote {out_dir}/REPORT.md", flush=True)


if __name__ == '__main__':
    main()
