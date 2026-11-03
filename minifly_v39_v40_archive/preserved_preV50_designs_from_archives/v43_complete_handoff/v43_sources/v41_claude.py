#!/usr/bin/env python3
"""V40: Complementary two-layer memory.

    python3 elm_fly_complementary_v40.py --self-test
    python3 elm_fly_complementary_v40.py --stage g1   --out-dir v40_results
    python3 elm_fly_complementary_v40.py --stage full --out-dir v40_results

V39 left two architectures with exactly complementary failures. Admission-controlled
slots bound storage but cannot generalise to an unseen rendering before it has been
seen twice (E1 ~ 0.03 at any load). A bounded superposed store generalises one-shot
(E1 ~ 0.998) but collapses under load. V40 composes them and arbitrates with V37's
corroboration authority, unchanged.

Label-free throughout: the only teaching signal is the next-byte residual.
Workload and slot layer are V39's, imported; the load axis is added here and the
stream is verified byte-identical to V39 at V39's own settings (see self_test).

Design: v40_experiment_recommendation.md.
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
V39_SOURCE = SOURCE.parent / 'elm_v39_earned_consolidation.py'

VERSION = 'v40.1.0'
SEEDS = tuple(range(40101, 40117))          # 16; V38 at 8 could not resolve its controls

CONTEXT_BYTES = 12
CODE_BITS = CONTEXT_BYTES * 8               # 96
OUTPUTS = 256
RATE = .6
UNITS = 4096                                # KC layer, V39's bounded superposed store
K_ACTIVE = 205                              # 5% coding
CONTACTS = 12                               # V39's S2-lag-c12, the best bounded arm
SHUFFLE_BUFFER = 512

# Pre-run calibration of the read composition, seed 40101, load 64, through wave D.
# E1 is one-shot held-out accuracy over 64 keys; bpb is the whole stream under that
# composition alone, before memory authority. Recorded because the choice is a design
# decision the run must not be allowed to relitigate: the rules that predict best are
# exactly the ones that do not generalise one-shot.
COMPOSITION_CALIBRATION = {
    'base':      {'E1': 0.0000, 'whole_bpb': 1.4346},
    'super':     {'E1': 1.0000, 'whole_bpb': 1.3556},
    'add':       {'E1': 0.1875, 'whole_bpb': 1.1680},
    'written':   {'E1': 1.0000, 'whole_bpb': 1.3556},
    'confident': {'E1': 0.3281, 'whole_bpb': 1.1344},
    'chosen': 'written',
    'note': 'written == super in practice: the layer is written almost everywhere after '
            'warmup. Chosen because it is the only rule that delivers one-shot transfer, '
            'which is the property the hybrid exists to test. It costs ~0.19 bpb against '
            'the additive rule; the prediction non-inferiority gate scores that cost.',
}

REFERENCE = dict(
    v39_super_wall={'32': 0.936, '64': 0.871, '128': 0.488, '256': 0.092},
    v39_super_E1=0.998, v39_slots_E1=0.027,
    v39_trace_corr_peak_bytes=2097152, v39_legacy_peak_bytes=23658060,
    v39_trace_corr_post_clutter=0.9082, v39_legacy_post_clutter=0.9199,
    v38_e1_heldout=0.0080,
)


# ----------------------------------------------------------------------------
# 0. Substrate import
# ----------------------------------------------------------------------------

def _load_v39():
    if not V39_SOURCE.exists():
        raise SystemExit(
            'V40 reuses the V39 workload and slot layer unchanged but could not find\n'
            f'{V39_SOURCE}\nPlace this file beside it.')
    spec = importlib.util.spec_from_file_location('elm_v40_v39', V39_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # V39 defers numpy and scipy into its own dependency bootstrap and binds them as
    # module globals. V40 needs its stream, Encoder, Learner and Memory, all of which
    # are pure numpy, so inject directly rather than running that bootstrap.
    if getattr(module, 'np', None) is None:
        module.np = np
    if getattr(module, 'student_t', None) is None:
        module.student_t = student_t
    return module


V39 = _load_v39()
context = V39.context
Encoder, Learner, Memory = V39.Encoder, V39.Learner, V39.Memory
softmax, loss = V39.softmax, V39.loss
TAU = V39.TAU

assert V39.D == CODE_BITS and V39.OUTPUTS == OUTPUTS and V39.RATE == RATE, \
    'V39 constants moved; V40 comparability assumptions are void.'


def sha(data):
    if isinstance(data, (bytes, bytearray)):
        return hashlib.sha256(bytes(data)).hexdigest()
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(values):
    """Paired Student 95%. Seeds are the sampling units; a missing seed voids it."""
    vals = [None if v is None else float(v) for v in values]
    good = [v for v in vals if v is not None and math.isfinite(v)]
    out = {'n': len(good), 'expected_n': len(vals),
           'mean': float(np.mean(good)) if good else None, 'low': None, 'high': None}
    if len(good) >= 2 and len(good) == len(vals):
        half = float(student_t.ppf(.975, len(good) - 1) * np.std(good, ddof=1) / math.sqrt(len(good)))
        out.update(low=out['mean'] - half, high=out['mean'] + half)
    return out


def paired(per_seed, left, right, key):
    seeds = sorted(set(per_seed.get(left, {})) & set(per_seed.get(right, {})))
    diffs = []
    for s in seeds:
        a, b = per_seed[left][s].get(key), per_seed[right][s].get(key)
        diffs.append(None if a is None or b is None else a - b)
    return interval(diffs)


# ----------------------------------------------------------------------------
# 1. Workload: V39's, with a load parameter
# ----------------------------------------------------------------------------

def make_stream(seed, keys_per_wave=16, repeats=32, smoke=False):
    """V39.make_stream with `count` exposed. Byte-identical to V39 at (16, 32).

    RNG call order is preserved exactly, so the only difference at other loads is the
    one the load axis intends. `nrev` scales with the load so the untouched retention
    cohort stays half of wave A; at V39's count=16 that is 4 and at its smoke count=4
    it is 1, so both of V39's own settings reproduce byte for byte.
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
        checkpoints.append({'record_count': len(rows), 'name': f'after_{cohort}',
                            'type': 'wave', 'wave': cohort})
        if wave == 1:
            for i in noisy:
                wrong = ord('a') + (int(labels[i]) - ord('a') + 1) % 24
                add(i, wrong, 'one_off_noise', 1, cohort, 0)
        if wave == 2:
            for exposure in range(1, 9):
                for i in rr.permutation(revised):
                    add(int(i), changed[int(i)], 'revision', exposure, cohort, 0)
                if exposure in (1, 2, 4, 8):
                    checkpoints.append({'record_count': len(rows), 'name': f'revision_{exposure}',
                                        'type': 'revision', 'wave': cohort, 'exposure': exposure})
    raw = b''.join(row['raw'] for row in rows)
    assert all(len(row['raw']) == 14 for row in rows)
    return {'seed': seed, 'keys': keys, 'labels': labels, 'revised': revised, 'noisy': noisy,
            'untouched': untouched, 'revisions': changed, 'rows': rows, 'raw': raw,
            'checkpoints': checkpoints, 'keys_per_wave': count, 'repeats': repeats,
            'raw_sha256': sha(raw)}


def workload(seed, keys_per_wave=16, smoke=False):
    """V39's clutter tail on top of the load-parameterised stream.

    The clutter cohort is held FIXED across loads (768 single-exposure keys plus 128
    seen twice, as in V39). Scaling it with the load would move two stressors at once;
    holding it fixed means clutter is a constant absolute perturbation and is therefore
    proportionally milder at high load. Stated as a limitation, not hidden.
    """
    s = make_stream(seed, keys_per_wave=keys_per_wave, smoke=smoke)
    s['standard_rows'] = len(s['rows'])
    rr = np.random.default_rng(seed + 390081)
    single_n, repeat_n = (24, 8) if smoke else (768, 128)
    used = set(s['keys'])
    keys = []
    alphabet = np.frombuffer(b'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', dtype=np.uint8)
    while len(keys) < single_n + repeat_n:
        k = bytes(rr.choice(alphabet, size=6))
        if k not in used:
            used.add(k)
            keys.append(k)
    targets = rr.choice(np.frombuffer(b'abcdefghijklmnopqrstuvwx', dtype=np.uint8),
                        size=len(keys)).astype(int)
    tail = []
    for i in rr.permutation(len(keys)):
        tail.append({'raw': context(keys[i], 0) + bytes([targets[i]]) + b'\n',
                     'kind': 'singleton' if i < single_n else 'exact_repeat',
                     'target': int(targets[i]), 'exposure': 1, 'key_id': -1,
                     'wave': 'T', 'alias': 0, 'rendering_exposure': 1})
    for i in rr.permutation(np.arange(single_n, len(keys))):
        tail.append({'raw': context(keys[i], 0) + bytes([targets[i]]) + b'\n',
                     'kind': 'exact_repeat', 'target': int(targets[i]), 'exposure': 2,
                     'key_id': -1, 'wave': 'T', 'alias': 0, 'rendering_exposure': 2})
    s['rows'] = s['rows'] + tail
    s['checkpoints'] = s['checkpoints'] + [{'record_count': len(s['rows']),
                                            'name': 'after_clutter', 'type': 'clutter',
                                            'wave': 'T'}]
    s['raw'] = b''.join(r['raw'] for r in s['rows'])
    s['raw_sha256'] = sha(s['raw'])
    s['tail_rows'] = len(tail)
    return s


# ----------------------------------------------------------------------------
# 2. The superposed layer (V39's S2-lag-c12)
# ----------------------------------------------------------------------------

class SuperposedLayer:
    """Fixed UNITS x 256 store over a sparse KC code. Bounded by construction.

    Front end is the raw lag96 code, which V39 measured as perfectly stationary
    (same-context self-overlap 1.0000 across a run). The phase-conditioned front end
    reached a higher separation ratio but drifted to 0.62, so a fact was read through
    a code sharing 62% of its units with the one it was written under; that is why V39
    ranked it below this one despite the better geometry.
    """

    def __init__(self, seed, units=UNITS, k=K_ACTIVE, contacts=CONTACTS):
        self.units, self.k, self.n_contacts = int(units), int(k), int(contacts)
        rr = np.random.default_rng(seed + 3604096)
        self.contacts = np.array([rr.choice(CODE_BITS, contacts, replace=False)
                                  for _ in range(units)])
        self.weights = rr.normal(size=(units, contacts))
        self.weights /= np.linalg.norm(self.weights, axis=1, keepdims=True)
        self.encoder = Encoder(seed, 'lag')
        self.output = np.zeros((units, OUTPUTS))
        self.frozen = False
        self.ops = 0
        self.updates = 0

    def encode(self, window):
        z = self.encoder.encode(bytes(window))[0]
        u = np.tanh(np.einsum('ij,ij->i', self.weights, z[self.contacts]))
        ix = np.sort(np.argpartition(-np.abs(u), self.k - 1)[:self.k])
        vals = u[ix]
        norm = float(np.linalg.norm(vals))
        return ix, (vals / norm if norm > 1e-12 else np.zeros(self.k))

    def scores(self, window):
        ix, v = self.encode(window)
        self.ops += self.k * OUTPUTS + self.units * self.n_contacts + self.units
        return v @ self.output[ix]

    def observe(self, window, y):
        if self.frozen:
            return
        ix, v = self.encode(window)
        block = self.output[ix]
        used = v @ block
        if not np.isfinite(used).all():
            raise FloatingPointError('Nonfinite superposed prediction.')
        vv = float(v @ v)
        if vv > 1e-15:
            error = -used
            error[y] += 1.
            self.output[ix] = block + RATE * np.outer(v / vv, error)
            self.updates += 1
        self.ops += self.k * OUTPUTS + self.units * self.n_contacts + self.units

    def cost(self):
        return {'store_bytes': int(self.output.nbytes),
                'fixed_code_bytes': int(self.contacts.nbytes + self.weights.nbytes),
                'operation_proxy': int(self.ops), 'bounded': True}

    def state_hash(self):
        return sha(self.output.tobytes())


class ShuffledSuperposed:
    """Nondeployable control: the superposed layer's own outputs, decoupled from context.

    Keeps a rotating buffer of recent genuine predictions and returns one drawn from it
    instead of the prediction for this context. Magnitude, timing and marginal
    distribution are those of the real layer; only the content-context correspondence
    is destroyed. This is V38's shuffled-evidence discipline applied to the new layer.
    """

    def __init__(self, layer, seed, size=SHUFFLE_BUFFER):
        self.layer = layer
        self.rng = np.random.default_rng(seed + 400077)
        self.buffer = collections.deque(maxlen=size)

    def scores(self, window):
        real = self.layer.scores(window)
        self.buffer.append(real.copy())
        if len(self.buffer) < 2:
            return np.zeros(OUTPUTS)
        return self.buffer[int(self.rng.integers(len(self.buffer) - 1))]


# ----------------------------------------------------------------------------
# 3. Arms
# ----------------------------------------------------------------------------

ARMS = ('SUPER', 'SLOTS', 'HYBRID', 'HYBRID-SHUFFLE', 'HYBRID-FROZEN', 'HYBRID-RANDOM')
MEMORY_ARMS = ('SLOTS', 'HYBRID', 'HYBRID-SHUFFLE', 'HYBRID-FROZEN')
NONDEPLOYABLE = ('HYBRID-SHUFFLE', 'HYBRID-RANDOM')
SLOT_POLICY = 'TRACE-CORR'                  # V39's qualified bounded policy, unchanged

LOADS = (64, 128, 256)
PRIMARY_LOAD_CANDIDATES = (256, 128, 64)


class Branches:
    """One shared base trajectory; every arm reads and writes its own memory branch.

    Sharing the base is V39's design and makes every contrast exactly paired: the arms
    differ only in what sits between the base predictor and the byte.
    """

    def __init__(self, seed, freeze_after):
        self.base = Learner('full')
        self.encoder = Encoder(seed, 'last_fd')
        self.live = SuperposedLayer(seed)
        self.frozen_layer = SuperposedLayer(seed)
        self.shuffled = ShuffledSuperposed(self.live, seed)
        self.memories = {a: Memory(SLOT_POLICY, seed) for a in MEMORY_ARMS}
        self.freeze_after = freeze_after
        self.rand_rng = np.random.default_rng(seed + 400101)
        self.authority_hits = 0
        self.authority_total = 0

    def fallback(self, arm, window, base_scores):
        """The three-way read chain below memory authority: superposed layer where it
        has content, shared base underneath.

        The composition is `written`, fixed by the pre-run calibration in
        COMPOSITION_CALIBRATION and not tuned afterwards. Adding the layer to the base
        instead predicts better but destroys the mechanism under test: at E1 the base
        has seen a key once and is still noisy, and the sum's argmax follows the base
        rather than the layer. In practice the layer is written for nearly every
        context after warmup, so the chain collapses to two levels and the base fires
        only very early -- which is reported, not hidden.
        """
        if arm == 'SLOTS':
            return base_scores
        layer = (self.shuffled if arm == 'HYBRID-SHUFFLE' else
                 self.frozen_layer if arm == 'HYBRID-FROZEN' else self.live)
        s = layer.scores(window)
        return s if float(np.linalg.norm(s)) > 1e-9 else base_scores

    def read(self, window, base_scores):
        """Every arm's scores for this context. No arm sees the byte."""
        out = {}
        packets = {}
        for arm in MEMORY_ARMS:
            fb = self.fallback(arm, window, base_scores)
            p = self.memories[arm].lookup(window, fb)
            packets[arm] = p
            out[arm] = p['scores']
        out['SUPER'] = self.live.scores(window)
        # HYBRID-RANDOM is an offline, count-matched read control over HYBRID's own
        # state, exactly as V37's RAND-MATCH was: it never learns and never writes.
        h = packets['HYBRID']
        self.authority_total += 1
        self.authority_hits += int(h['authority'])
        rate = self.authority_hits / max(self.authority_total, 1)
        use_slot = bool(self.rand_rng.random() < rate) and h['id'] is not None
        out['HYBRID-RANDOM'] = h['scores'] if use_slot else self.fallback('HYBRID', window,
                                                                         base_scores)
        return out, packets

    def write(self, window, base_scores, packets, y, record_number):
        z, b = self.encoder.encode(bytes(window))
        self.base.observe(z, b, y)
        self.live.observe(window, y)
        if record_number <= self.freeze_after:
            self.frozen_layer.observe(window, y)
        elif not self.frozen_layer.frozen:
            self.frozen_layer.frozen = True
        for arm in MEMORY_ARMS:
            self.memories[arm].observe(packets[arm], y)

    def state_hash(self):
        return sha([self.base.state_hash(), self.live.state_hash(),
                    self.frozen_layer.state_hash()]
                   + [self.memories[a].state_hash() for a in MEMORY_ARMS])


# ----------------------------------------------------------------------------
# 4. One trajectory
# ----------------------------------------------------------------------------

def run_trial(seed, load, smoke=False):
    start = time.perf_counter()
    stream = workload(seed, keys_per_wave=load // 4, smoke=smoke)
    rows = stream['rows']
    freeze_after = next(cp['record_count'] for cp in stream['checkpoints']
                        if cp['name'] == 'after_A')
    br = Branches(seed, freeze_after)
    enc = br.encoder

    groups = collections.defaultdict(lambda: collections.defaultdict(
        lambda: {'n': 0, 'bpb': 0., 'correct': 0}))
    probes = []
    cps = collections.defaultdict(list)
    for cp in stream['checkpoints']:
        cps[cp['record_count']].append(cp)
    history = bytearray()

    def probe(name, kind, ids, aliases=(0, 1), revised=False):
        before = br.state_hash()
        for i in ids:
            y = (stream['revisions'][i] if revised and i in stream['revisions']
                 else int(stream['labels'][i]))
            for alias in aliases:
                c = context(stream['keys'][i], alias)
                z, b = enc.encode(c)
                sb = br.base.scores(z, b)
                scored, _ = br.read(c, sb)
                scored['BASE'] = sb
                for arm, s in scored.items():
                    probes.append({'checkpoint': name, 'kind': kind, 'arm': arm,
                                   'key_id': int(i), 'alias': int(alias),
                                   'accuracy': float(int(np.argmax(s)) == y),
                                   'bpb': float(loss(s, y))})
        if br.state_hash() != before:
            raise AssertionError(f'Probe mutated state at {name}')

    emerged = {}
    for number, row in enumerate(rows, 1):
        for pos, y in enumerate(row['raw']):
            z, b = enc.encode(bytes(history))
            sb = br.base.scores(z, b)
            scored, packets = br.read(history, sb)
            category = 'target' if pos == CONTEXT_BYTES else 'nontarget'
            tail = 'clutter' if number > stream['standard_rows'] else 'standard'
            for arm, s in list(scored.items()) + [('BASE', sb)]:
                for cat in ('whole', category, tail + '_' + category):
                    g = groups[arm][cat]
                    g['n'] += 1
                    g['bpb'] += loss(s, y)
                    g['correct'] += int(np.argmax(s) == y)
            br.write(history, sb, packets, y, number)
            history.append(y)
            if len(history) > CONTEXT_BYTES:
                del history[0]
            # E1/E2 by stream position so both architectures compute the same thing:
            # E1 after the key's first exposure (alias 0 trained, alias 1 not), E2 after
            # the second (both trained). V38 defined these by slot-binding events, which
            # a superposed store does not have.
            if pos == CONTEXT_BYTES and row['kind'] == 'acquisition':
                i = row['key_id']
                phase = emerged.setdefault(i, set())
                if row['exposure'] == 1 and 'E1' not in phase:
                    phase.add('E1')
                    probe('E1', 'emergence', [i], (2,))
                elif row['exposure'] == 2 and 'E2' not in phase:
                    phase.add('E2')
                    probe('E2', 'emergence', [i], (2,))
        for cp in cps.get(number, []):
            name, k = cp['name'], stream['keys_per_wave']
            wi = ord(cp['wave']) - 65
            if cp['type'] in ('wave', 'clutter'):
                probe(name, 'untouched_A', stream['untouched'], aliases=(0, 1))
                probe(name, 'untouched_A_heldout', stream['untouched'], aliases=(2,))
                if cp['type'] == 'wave':
                    probe(name, 'current', range(wi * k, (wi + 1) * k), aliases=(0, 1))
                    if cp['wave'] == 'A':
                        probe(name, 'all_A', range(k), aliases=(0, 1))
            elif cp['type'] == 'revision':
                probe(name, 'revision', stream['revised'], aliases=(0, 1, 2), revised=True)

    costs = {}
    for arm in ARMS:
        c = {'superposed': None, 'memory': None}
        if arm in MEMORY_ARMS:
            m = br.memories[arm].cost()
            c['memory'] = {k: m[k] for k in ('traces', 'slots', 'anchors', 'semantic_bytes',
                                             'bounded')}
            c['memory']['peak'] = dict(br.memories[arm].peak)
            c['memory']['ops'] = int(sum(br.memories[arm].ops.values()))
        if arm != 'SLOTS':
            layer = br.frozen_layer if arm == 'HYBRID-FROZEN' else br.live
            c['superposed'] = layer.cost()
        costs[arm] = c

    result = {'seed': seed, 'load': load, 'smoke': smoke,
              'untouched_ids': [int(i) for i in stream['untouched']],
              'stream_sha256': stream['raw_sha256'], 'byte_events': len(stream['raw']),
              'standard_rows': stream['standard_rows'], 'tail_rows': stream['tail_rows'],
              'keys': len(stream['keys']),
              'groups': {a: {c: {'n': g['n'], 'bpb': g['bpb'] / g['n'] if g['n'] else None,
                                 'accuracy': g['correct'] / g['n'] if g['n'] else None}
                             for c, g in d.items()} for a, d in groups.items()},
              'costs': costs, 'probes': probes,
              'elapsed_seconds': time.perf_counter() - start}
    result['metrics'] = {a: metrics(result, a) for a in list(ARMS) + ['BASE']}
    return result


def probe_mean(probes, arm, checkpoint, kind, aliases=(0, 1), field='accuracy'):
    vals = [p[field] for p in probes if p['arm'] == arm and p['checkpoint'] == checkpoint
            and p['kind'] == kind and p['alias'] in aliases]
    return float(np.mean(vals)) if vals else None


def metrics(result, arm):
    p = result['probes']
    g = result['groups'].get(arm, {})
    m = {f'{k}_bpb': (g[k]['bpb'] if k in g else None)
         for k in ('whole', 'target', 'nontarget', 'clutter_target', 'standard_target')}
    m['A_initial'] = probe_mean(p, arm, 'after_A', 'all_A')
    for w in 'ABCD':
        m[f'A_after_{w}'] = probe_mean(p, arm, f'after_{w}', 'untouched_A')
        m[f'A_after_{w}_heldout'] = probe_mean(p, arm, f'after_{w}', 'untouched_A_heldout',
                                               aliases=(2,))
    m['post_clutter'] = probe_mean(p, arm, 'after_clutter', 'untouched_A')
    m['post_clutter_heldout'] = probe_mean(p, arm, 'after_clutter', 'untouched_A_heldout',
                                           aliases=(2,))
    m['D_current'] = probe_mean(p, arm, 'after_D', 'current')
    for ex in (1, 2, 4, 8):
        for a in range(3):
            m[f'revision_{ex}_alias_{a}'] = probe_mean(p, arm, f'revision_{ex}', 'revision',
                                                       aliases=(a,))
    # E1/E2 fire for every key at its first/second exposure, so the stream-wide figure
    # spans all four waves. The wall's other two components are the untouched wave-A
    # cohort, so the wall uses the wave-A restriction and the stream-wide value is
    # reported alongside it. Without this, HYBRID-FROZEN scores ~0.25 on the wall purely
    # because three of the four cohorts emerge after its layer is frozen -- a real
    # deficit, but one about later cohorts rather than about retention of wave A.
    cohort = set(result.get('untouched_ids', []))
    for e in ('E1', 'E2'):
        m[f'{e}_heldout'] = probe_mean(p, arm, e, 'emergence', aliases=(2,))
        vals = [x['accuracy'] for x in p if x['arm'] == arm and x['checkpoint'] == e
                and x['kind'] == 'emergence' and x['alias'] == 2 and x['key_id'] in cohort]
        m[f'{e}_heldout_A'] = float(np.mean(vals)) if vals else None
    # The V35 four-way wall as one number, over the three architecture-discriminating
    # components, all on the untouched wave-A cohort so retention and transfer share a
    # delay. A single component is maximised by surrendering another: slots win
    # retention with E1 ~ 0, superposition wins E1 and loses retention under load.
    for tag, ret, held in (('wall', 'A_after_D', 'A_after_D_heldout'),
                           ('wall_post_clutter', 'post_clutter', 'post_clutter_heldout')):
        parts = [m.get(ret), m.get(held), m.get('E1_heldout_A')]
        m[tag] = None if any(x is None for x in parts) else min(parts)
    c = result['costs'].get(arm, {})
    sup, mem = c.get('superposed'), c.get('memory')
    m['store_bytes'] = ((sup['store_bytes'] + sup['fixed_code_bytes']) if sup else 0) + \
                       (mem['semantic_bytes'] if mem else 0)
    m['peak_semantic_bytes'] = mem['peak']['semantic_bytes'] if mem else (
        sup['store_bytes'] + sup['fixed_code_bytes'] if sup else None)
    m['slots'] = mem['slots'] if mem else None
    m['traces'] = mem['traces'] if mem else None
    m['ops_per_byte'] = (((sup['operation_proxy'] if sup else 0) +
                          (mem['ops'] if mem else 0)) / result['byte_events'])
    m['bounded'] = bool((mem['bounded'] if mem else True) and (sup is None or sup['bounded']))
    return m


# ----------------------------------------------------------------------------
# 5. Gates
# ----------------------------------------------------------------------------

G0_ACQUISITION = 0.90
G1_SLOTS_E1_MAX = 0.10       # the slot layer must actually fail one-shot transfer
G1_SUPER_DECAY = 0.30        # superposition must actually collapse across the load axis
G2_MARGIN = 0.10             # hybrid over each single layer
G3_MARGIN = 0.10             # hybrid over each specificity control
BPB_NONINFERIORITY = 0.10
PRIMARY = 'wall'


def cell(arm, load):
    return f'{arm}@{load}'


def level(per_seed, arm, load, key):
    return interval([per_seed[cell(arm, load)][s].get(key)
                     for s in sorted(per_seed.get(cell(arm, load), {}))])


def evaluate(per_seed, loads, stage):
    loads = sorted(loads)
    lo, hi = loads[0], loads[-1]
    gates = {}

    # G0 -- acquisition validity at every load, for every arm.
    g0 = {}
    for arm in ARMS:
        for load in loads:
            if cell(arm, load) in per_seed:
                lv = level(per_seed, arm, load, 'A_initial')
                g0[cell(arm, load)] = {'level': lv,
                                       'pass': bool(lv['mean'] is not None
                                                    and lv['mean'] >= G0_ACQUISITION)}
    valid = {c for c, v in g0.items() if v['pass']}
    gates['G0'] = {'threshold': G0_ACQUISITION, 'per_cell': g0, 'valid_cells': sorted(valid),
                   'pass': bool(g0) and any(v['pass'] for v in g0.values()),
                   'note': 'Checked at every load. Cells below the threshold are '
                           'invalid_acquisition and enter no contrast.'}

    # G1 -- the complementary gap exists. If either single layer already holds the wall
    # at high load there is nothing for a hybrid to fill.
    slots_e1 = (level(per_seed, 'SLOTS', hi, 'E1_heldout_A')
                if cell('SLOTS', hi) in per_seed else interval([]))
    decay = (paired(per_seed, cell('SUPER', lo), cell('SUPER', hi), PRIMARY)
             if cell('SUPER', lo) in per_seed and cell('SUPER', hi) in per_seed else interval([]))
    slots_fails = bool(slots_e1['mean'] is not None and slots_e1['mean'] <= G1_SLOTS_E1_MAX
                       and slots_e1['high'] is not None and slots_e1['high'] < G1_SLOTS_E1_MAX * 2)
    super_fails = bool(decay['mean'] is not None and decay['mean'] >= G1_SUPER_DECAY
                       and decay['low'] is not None and decay['low'] > 0)
    gates['G1'] = {'slots_E1_at_max_load': slots_e1, 'slots_E1_max': G1_SLOTS_E1_MAX,
                   'super_wall_decay': decay, 'super_decay_min': G1_SUPER_DECAY,
                   'slots_fails_one_shot': slots_fails, 'super_collapses_under_load': super_fails,
                   'pass': bool(slots_fails and super_fails),
                   'note': 'Both single layers must fail in the predicted complementary '
                           'way. If either already holds the wall there is no gap and '
                           'the design is void.'}

    # G2 -- primary: the hybrid beats BOTH single layers on the wall.
    target_load, singles = None, ('SUPER', 'SLOTS')
    for load in reversed(loads):
        if all(cell(a, load) in valid for a in ('HYBRID',) + singles):
            target_load = load
            break
    if target_load is not None and gates['G1']['pass']:
        contrasts = {a: paired(per_seed, cell('HYBRID', target_load), cell(a, target_load),
                               PRIMARY) for a in singles}
        ok = all(d['mean'] is not None and d['mean'] >= G2_MARGIN
                 and d['low'] is not None and d['low'] > 0 for d in contrasts.values())
        gates['G2'] = {'margin': G2_MARGIN, 'endpoint': PRIMARY, 'load': target_load,
                       'contrasts': contrasts,
                       'components': {cell(a, target_load): {
                           k: level(per_seed, a, target_load, k)
                           for k in ('A_after_D', 'A_after_D_heldout', 'E1_heldout_A',
                                     'E1_heldout', PRIMARY)}
                           for a in ('HYBRID',) + singles},
                       'pass': bool(ok)}
    else:
        gates['G2'] = {'margin': G2_MARGIN, 'pass': False,
                       'note': 'Not evaluated: G1 failed or no load where the hybrid and '
                               'both single layers acquire.'}

    # G3 -- specificity. A shuffled layer or a random arbitration that matches the
    # hybrid means this is ensembling, not complementary memory.
    if target_load is not None and gates['G2']['pass']:
        controls = [a for a in ('HYBRID-SHUFFLE', 'HYBRID-RANDOM')
                    if cell(a, target_load) in per_seed]
        cs = {a: paired(per_seed, cell('HYBRID', target_load), cell(a, target_load), PRIMARY)
              for a in controls}
        ok = bool(cs) and all(d['mean'] is not None and d['mean'] >= G3_MARGIN
                              and d['low'] is not None and d['low'] > 0 for d in cs.values())
        gates['G3'] = {'margin': G3_MARGIN, 'load': target_load, 'contrasts': cs, 'pass': ok}
    else:
        gates['G3'] = {'margin': G3_MARGIN, 'pass': False,
                       'note': 'Not evaluated: G2 did not pass.'}

    # G4 -- cost. An unbounded hybrid cannot qualify.
    def split_cell(c):
        arm, load = c.rsplit('@', 1)
        return arm, int(load)
    gates['G4'] = {'per_cell': {c: {k: level(per_seed, *split_cell(c), k)
                                    for k in ('store_bytes', 'peak_semantic_bytes',
                                              'ops_per_byte', 'slots', 'traces')}
                                for c in sorted(per_seed)},
                   'bounded': {c: all(m.get('bounded') for m in per_seed[c].values())
                               for c in sorted(per_seed)}}
    gates['G4']['pass'] = bool(target_load is None
                               or gates['G4']['bounded'].get(cell('HYBRID', target_load), False))

    # Prediction non-inferiority: a hybrid that buys the wall by wrecking prediction has
    # solved nothing.
    if target_load is not None:
        d = paired(per_seed, cell('HYBRID', target_load), cell('SLOTS', target_load), 'whole_bpb')
        gates['prediction'] = {'margin': BPB_NONINFERIORITY, 'hybrid_minus_slots_whole_bpb': d,
                               'pass': bool(d['mean'] is not None
                                            and d['mean'] <= BPB_NONINFERIORITY)}
    gates['stage'] = stage
    gates['verdict'] = verdict(gates)
    return gates


def verdict(gates):
    if not gates['G0']['pass']:
        return 'invalid_acquisition'
    if not gates['G1']['pass']:
        return 'no_complementary_gap'
    if gates['stage'] == 'g1':
        return 'gap_confirmed_run_full_grid'
    if not gates['G4'].get('pass', True):
        return 'unbounded_hybrid'
    g2 = gates['G2']
    if not g2['pass']:
        ds = list((g2.get('contrasts') or {}).values())
        if not ds or any(d['mean'] is None for d in ds):
            return 'invalid_acquisition'
        if all(d['mean'] > 0 for d in ds):
            return 'favorable_unresolved_precision'
        if any((d['high'] or 1) < 0 for d in ds):
            return 'adverse'
        return 'no_mean_advantage'
    if not gates['G3']['pass']:
        return 'ensembling_not_mechanism'
    if not gates.get('prediction', {}).get('pass', True):
        return 'supported_below_qualification'
    return 'qualified'


# ----------------------------------------------------------------------------
# 6. Outputs
# ----------------------------------------------------------------------------

def fmt(s, places=4):
    if not s or s.get('mean') is None:
        return '--'
    if s.get('low') is None:
        return f'{s["mean"]:+.{places}f}'
    return f'{s["mean"]:+.{places}f} [{s["low"]:+.{places}f}, {s["high"]:+.{places}f}]'


def write_outputs(out_dir, results, per_seed, gates, manifest):
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ['seed', 'load', 'arm', 'A_initial', 'A_after_D', 'A_after_D_heldout',
              'E1_heldout_A', 'E2_heldout_A', 'E1_heldout', 'E2_heldout',
              'wall', 'post_clutter', 'post_clutter_heldout',
              'wall_post_clutter', 'D_current', 'revision_4_alias_0', 'revision_4_alias_2',
              'whole_bpb', 'target_bpb', 'nontarget_bpb', 'clutter_target_bpb',
              'store_bytes', 'peak_semantic_bytes', 'slots', 'traces', 'ops_per_byte',
              'bounded', 'elapsed_seconds']
    with (out_dir / 'TERMINAL.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sorted(results, key=lambda r: (r['load'], r['seed'])):
            for arm in list(ARMS) + ['BASE']:
                row = {'seed': r['seed'], 'load': r['load'], 'arm': arm,
                       'elapsed_seconds': round(r['elapsed_seconds'], 2)}
                row.update({k: r['metrics'][arm].get(k) for k in fields if k not in row})
                w.writerow(row)
    (out_dir / 'GATES.json').write_text(json.dumps(gates, indent=1, default=str))
    (out_dir / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1, default=str))
    (out_dir / 'PER_SEED.json').write_text(json.dumps(per_seed, indent=1, default=str))
    (out_dir / 'REPORT.md').write_text(report(per_seed, gates, manifest))


def report(per_seed, gates, manifest):
    loads = sorted({int(c.split('@')[1]) for c in per_seed})
    L = ['# V40 results — complementary two-layer memory', '',
         f"**Source SHA256** `{manifest['identity']['source_sha256']}`  ",
         f"**Stage** `{gates['stage']}` · **seeds** {len(manifest['plan']['seeds'])} · "
         f"**verdict** **`{gates['verdict']}`**", '',
         'Label-free: the only teaching signal is the next-byte residual. Workload and '
         "slot layer are V39's, imported; the load axis is added here and the stream is "
         "byte-identical to V39 at V39's own settings.", '',
         '`wall` = min(retention, delayed held-out transfer, E1 one-shot transfer), all '
         'on the untouched wave-A cohort. A single component is maximised by '
         'surrendering another, so the wall is the endpoint.', '']

    L += ['## The wall by arm and load', '',
          '| arm | ' + ' | '.join(str(x) for x in loads) + ' |',
          '|---|' + '---:|' * len(loads)]
    for arm in ARMS:
        vals = []
        for load in loads:
            s = level(per_seed, arm, load, PRIMARY) if cell(arm, load) in per_seed else None
            vals.append('--' if not s or s['mean'] is None else f"{s['mean']:.3f}")
        tag = f'`{arm}`' + (' *(control)*' if arm in NONDEPLOYABLE else '')
        L.append(f'| {tag} | ' + ' | '.join(vals) + ' |')

    L += ['', '## Components at each load', '',
          '| cell | A initial | retention | delayed held-out | E1 | **wall** | store | ops/byte |',
          '|---|---|---|---|---|---|---:|---:|']
    for load in loads:
        for arm in ARMS:
            if cell(arm, load) not in per_seed:
                continue
            g = lambda k: level(per_seed, arm, load, k)
            store = g('store_bytes')['mean']
            ops = g('ops_per_byte')['mean']
            L.append(f"| `{cell(arm, load)}` | {fmt(g('A_initial'), 3)} | "
                     f"{fmt(g('A_after_D'), 3)} | {fmt(g('A_after_D_heldout'), 3)} | "
                     f"{fmt(g('E1_heldout_A'), 3)} | **{fmt(g(PRIMARY), 3)}** | "
                     f"{(store or 0)/1048576:.1f} MB | {(ops or 0):,.0f} |")

    L += ['', '## Gates', '']
    g0 = gates['G0']
    L += [f"### G0 — acquisition validity (≥ {g0['threshold']:.2f}, every load) · "
          f"**{'pass' if g0['pass'] else 'FAIL'}**", '',
          '| cell | A initial | valid |', '|---|---|:-:|']
    for c, v in sorted(g0['per_cell'].items()):
        L.append(f"| `{c}` | {fmt(v['level'], 3)} | {v['pass']} |")

    g1 = gates['G1']
    L += ['', f"### G1 — the complementary gap exists · **{'pass' if g1['pass'] else 'FAIL'}**",
          '', g1['note'], '',
          f"- `SLOTS` E1 at the highest load: {fmt(g1['slots_E1_at_max_load'], 3)} "
          f"(must be ≤ {g1['slots_E1_max']:.2f}) — **{g1['slots_fails_one_shot']}**",
          f"- `SUPER` wall decay across the load axis: {fmt(g1['super_wall_decay'], 3)} "
          f"(must be ≥ {g1['super_decay_min']:.2f}) — **{g1['super_collapses_under_load']}**"]

    g2 = gates['G2']
    L += ['', f"### G2 — primary: the hybrid beats both single layers · "
          f"**{'pass' if g2['pass'] else 'FAIL'}**", '']
    if 'contrasts' in g2:
        L += [f"At load **{g2['load']}**, on `{g2['endpoint']}`, margin ≥ "
              f"{g2['margin']:.2f} with the paired lower bound above zero.", '',
              '| contrast | wall difference |', '|---|---|']
        for a, d in sorted(g2['contrasts'].items()):
            L.append(f"| `HYBRID` − `{a}` | {fmt(d)} |")
        L += ['', '| cell | retention | delayed | E1 (wave A) | E1 (all waves) | wall |',
              '|---|---|---|---|---|---|']
        for c, v in g2['components'].items():
            L.append(f"| `{c}` | {fmt(v['A_after_D'], 3)} | {fmt(v['A_after_D_heldout'], 3)} | "
                     f"{fmt(v['E1_heldout_A'], 3)} | {fmt(v['E1_heldout'], 3)} | "
                     f"{fmt(v[PRIMARY], 3)} |")
    else:
        L.append(g2.get('note', ''))

    g3 = gates['G3']
    L += ['', f"### G3 — specificity · **{'pass' if g3['pass'] else 'FAIL'}**", '']
    if 'contrasts' in g3:
        L += ['A shuffled layer or a count-matched random arbitration that matches the '
              'hybrid means this is ensembling, not complementary memory.', '',
              '| contrast | wall difference |', '|---|---|']
        for a, d in sorted(g3['contrasts'].items()):
            L.append(f"| `HYBRID` − `{a}` | {fmt(d)} |")
    else:
        L.append(g3.get('note', ''))

    if 'prediction' in gates:
        p = gates['prediction']
        L += ['', f"### Prediction non-inferiority · **{'pass' if p['pass'] else 'FAIL'}**", '',
              f"`HYBRID` − `SLOTS` whole bpb: {fmt(p['hybrid_minus_slots_whole_bpb'])} "
              f"(must be ≤ {p['margin']:+.2f}; lower is better)."]

    L += ['', '## Reference points', '', '| quantity | value |', '|---|---:|',
          f"| V39 superposed wall @32 / @256 | {REFERENCE['v39_super_wall']['32']} / "
          f"{REFERENCE['v39_super_wall']['256']} |",
          f"| V39 superposed E1 / slot E1 | {REFERENCE['v39_super_E1']} / "
          f"{REFERENCE['v39_slots_E1']} |",
          f"| V38 E1 held-out (all policies) | {REFERENCE['v38_e1_heldout']} |",
          f"| V39 TRACE-CORR / LEGACY-U peak store | "
          f"{REFERENCE['v39_trace_corr_peak_bytes']/1048576:.2f} MB / "
          f"{REFERENCE['v39_legacy_peak_bytes']/1048576:.2f} MB |",
          '', '## Interpretation limits', '',
          '- G0 and G1 are hard gates in that order. G1 exists because a hybrid is only',
          '  worth testing if both single layers fail in the predicted complementary way;',
          '  if either already holds the wall, the premise dissolves.',
          '- `HYBRID-SHUFFLE` and `HYBRID-RANDOM` are nondeployable controls and can never',
          '  qualify. `HYBRID-RANDOM` is an offline count-matched read policy over the',
          "  hybrid's own state, as V37's RAND-MATCH was; it never learns or writes.",
          '- The clutter cohort is fixed across loads, so it is proportionally milder at',
          '  high load. Load and clutter are not varied together on purpose.',
          '- `SUPER` carries no shared base learner, matching V39, so its whole-stream bpb',
          '  is not comparable to the arms that do. The wall is the endpoint for that reason.',
          '- The read composition below memory authority is `written`, fixed by a pre-run',
          f"  calibration ({COMPOSITION_CALIBRATION['chosen']}): the additive rule predicts",
          f"  {COMPOSITION_CALIBRATION['add']['whole_bpb']:.3f} bpb against",
          f"  {COMPOSITION_CALIBRATION['written']['whole_bpb']:.3f} but reaches only",
          f"  {COMPOSITION_CALIBRATION['add']['E1']:.3f} one-shot transfer against 1.000, so it",
          '  cannot express the mechanism. The table is in MANIFEST.json.',
          '- The superposed layer uses the stationary lag front end. V39 measured the',
          '  phase-conditioned alternative at 0.62 same-context self-overlap across a run,',
          '  which is why the higher-separation code is not used here.',
          '- E1/E2 fire for every key at its first exposure, so the stream-wide figure',
          '  spans all four cohorts while the wall\'s other components are the untouched',
          '  wave-A cohort. The wall uses the wave-A restriction so all three components',
          '  share a cohort; both figures are reported.',
          '- Revision endpoints are recency-protected; the wall deliberately uses the',
          '  wave-D delay for both of its retention components.',
          '- Synthetic stream with controlled renderings. No scaling or language claim.']
    return '\n'.join(L) + '\n'


# ----------------------------------------------------------------------------
# 7. Self-test
# ----------------------------------------------------------------------------

def self_test(full=False):
    checks = {}

    for seed in (39101, 39105):
        assert make_stream(seed)['raw_sha256'] == V39.make_stream(seed)['raw_sha256'], seed
        assert make_stream(seed, smoke=True)['raw_sha256'] == \
            V39.make_stream(seed, smoke=True)['raw_sha256'], seed
    checks['stream_matches_v39'] = True

    w = workload(39101)
    wv = V39.workload(39101)
    assert w['raw_sha256'] == wv['raw_sha256'], 'clutter tail diverged from V39'
    checks['workload_matches_v39'] = True

    sizes = {}
    for load in LOADS:
        count = load // 4
        s = workload(40101, keys_per_wave=count)
        assert len(s['keys']) == load
        assert len(s['untouched']) == count - 2 * max(1, count // 4)
        per_key = collections.Counter(r['key_id'] for r in s['rows']
                                      if r['kind'] == 'acquisition')
        assert set(per_key.values()) == {32}
        sizes[load] = len(s['raw'])
    assert sorted(sizes.values()) == [sizes[l] for l in LOADS]
    checks['load_axis'] = sizes

    # The superposed layer is bounded and stationary.
    layer = SuperposedLayer(40101)
    assert layer.output.nbytes == UNITS * OUTPUTS * 8
    ctx = context(w['keys'][0], 0)
    before = layer.encode(ctx)[0]
    for r in w['rows'][:400]:
        for pos, y in enumerate(r['raw']):
            layer.observe(r['raw'][max(0, pos - 12):pos], y)
    assert layer.output.nbytes == UNITS * OUTPUTS * 8
    assert set(before.tolist()) == set(layer.encode(ctx)[0].tolist()), 'lag front end drifted'
    checks['superposed_bounded_and_stationary'] = True

    # The shuffled control keeps the layer's magnitudes and destroys its correspondence.
    sh = ShuffledSuperposed(layer, 40101)
    outs = [sh.scores(context(w['keys'][i], 0)) for i in range(40)]
    real = [layer.scores(context(w['keys'][i], 0)) for i in range(40)]
    assert not np.allclose(outs[-1], real[-1])
    assert abs(float(np.mean([np.linalg.norm(o) for o in outs[5:]]))
               - float(np.mean([np.linalg.norm(r) for r in real[5:]]))) < 0.5 * \
        float(np.mean([np.linalg.norm(r) for r in real[5:]])), 'shuffle changed magnitude'
    checks['shuffle_control'] = True

    r = run_trial(40101, 64, smoke=True)
    for arm in ARMS:
        assert r['metrics'][arm]['A_initial'] is not None, arm
        assert r['metrics'][arm]['E1_heldout'] is not None, arm
        assert r['metrics'][arm]['E1_heldout_A'] is not None, arm
        assert r['metrics'][arm]['wall'] is not None, arm
        m = r['metrics'][arm]
        assert m['wall'] == min(m['A_after_D'], m['A_after_D_heldout'], m['E1_heldout_A'])
    assert r['metrics']['SLOTS']['bounded'] and r['metrics']['HYBRID']['bounded']
    checks['smoke_walls'] = {a: round(r['metrics'][a]['wall'], 4) for a in ARMS}

    again = run_trial(40101, 64, smoke=True)
    assert again['metrics'] == r['metrics']
    checks['deterministic'] = True

    if full:
        t0 = time.perf_counter()
        rr = run_trial(40101, 64)
        checks['full_length'] = {
            'seconds': round(time.perf_counter() - t0, 1), 'bytes': rr['byte_events'],
            'walls': {a: round(rr['metrics'][a]['wall'], 4) for a in ARMS},
            'E1_waveA': {a: round(rr['metrics'][a]['E1_heldout_A'], 4) for a in ARMS},
            'E1_all': {a: round(rr['metrics'][a]['E1_heldout'], 4) for a in ARMS},
            'retention': {a: round(rr['metrics'][a]['A_after_D'], 4) for a in ARMS},
            'store_MB': {a: round(rr['metrics'][a]['store_bytes'] / 1048576, 2) for a in ARMS}}
    return checks


# ----------------------------------------------------------------------------
# 8. Runner
# ----------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--verify-full', action='store_true',
                        help='Self-test plus one full-length engineering trajectory.')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stage', choices=('g1', 'full'), default='full',
                        help="'g1' runs only the load extremes, which can void the design.")
    parser.add_argument('--out-dir', '--out', type=Path, default=Path('v40_results'))
    parser.add_argument('--seeds', type=int, help='Override the seed count (non-scientific).')
    parser.add_argument('--loads', type=str, help='Comma-separated load override (non-scientific).')
    parser.add_argument('--max-new-jobs', type=int)
    args = parser.parse_args(argv)

    if args.self_test or args.verify_full:
        print(json.dumps(self_test(args.verify_full), indent=2, default=str))
        return

    seeds = list(SEEDS[:args.seeds] if args.seeds else SEEDS)
    loads = ([int(x) for x in args.loads.split(',')] if args.loads
             else ([LOADS[0], LOADS[-1]] if args.stage == 'g1' else list(LOADS)))
    scientific = not args.smoke and not args.seeds and not args.loads and args.stage == 'full'
    out_dir, jobs_dir = args.out_dir, args.out_dir / 'jobs'
    jobs_dir.mkdir(parents=True, exist_ok=True)

    plan = {'seeds': seeds, 'loads': loads, 'stage': args.stage, 'smoke': args.smoke,
            'arms': list(ARMS), 'slot_policy': SLOT_POLICY, 'units': UNITS,
            'k_active': K_ACTIVE, 'contacts': CONTACTS, 'rate': RATE,
            'composition_calibration': COMPOSITION_CALIBRATION}
    identity = {'source_sha256': sha_file(SOURCE), 'v39_sha256': sha_file(V39_SOURCE),
                'plan_sha256': sha(plan), 'version': VERSION}
    manifest = {'version': VERSION, 'identity': identity, 'plan': plan,
                'scientific': scientific,
                'environment': {'python': sys.version, 'numpy': np.__version__,
                                'platform': platform.platform(), 'blas_threads': 1}}

    total = len(loads) * len(seeds)
    print(f'V40 {args.stage}: {len(loads)} loads x {len(seeds)} seeds = {total} trajectories, '
          f"{len(ARMS)} arms each{'' if scientific else '  (ENGINEERING ONLY)'}", flush=True)

    results, new, reused, stop = [], 0, 0, False
    for load in loads:
        for seed in seeds:
            path = jobs_dir / f'{seed}_L{load}.json'
            if path.exists():
                cached = json.loads(path.read_text())
                if cached.get('identity') == identity:
                    results.append(cached['result'])
                    reused += 1
                    continue
            if args.max_new_jobs is not None and new >= args.max_new_jobs:
                print(f'Paused at --max-new-jobs={args.max_new_jobs}.', flush=True)
                stop = True
                break
            t0 = time.perf_counter()
            r = run_trial(seed, load, smoke=args.smoke)
            tmp = path.with_suffix('.tmp')
            tmp.write_text(json.dumps({'identity': identity, 'result': r}, default=str))
            tmp.replace(path)
            results.append(r)
            new += 1
            m = r['metrics']
            print(f"  L{load:<4d} seed {seed}  "
                  f"SUPER={m['SUPER']['wall']:.3f} SLOTS={m['SLOTS']['wall']:.3f} "
                  f"HYBRID={m['HYBRID']['wall']:.3f}  "
                  f'{time.perf_counter()-t0:.1f}s  [{new+reused}/{total}]', flush=True)
        if stop:
            break

    per_seed = collections.defaultdict(dict)
    for r in results:
        for arm in ARMS:
            per_seed[cell(arm, r['load'])][r['seed']] = r['metrics'][arm]
    per_seed = dict(per_seed)
    gates = evaluate(per_seed, sorted({r['load'] for r in results}), args.stage)
    manifest['status'] = {'completed': len(results), 'new': new, 'reused': reused,
                          'expected': total}
    write_outputs(out_dir, results, per_seed, gates, manifest)
    print(f"\nVerdict: {gates['verdict']}\nWrote {out_dir}/REPORT.md", flush=True)


if __name__ == '__main__':
    main()
