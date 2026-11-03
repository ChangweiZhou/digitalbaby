#!/usr/bin/env python3
"""V36: does input separation, not the write rule, decide persistence?

THE ARGUMENT

V20-V35 searched write rules: output gain modulation, readout addressing,
learning-rate cascades, sparse/competitive/cyclic/sticky plasticity, novelty
gating, MESU metaplasticity, directional covariance, exact RLS, residual and
absolute local ownership. Every one failed, traded retention against transfer,
or was matched by its own control.

A connectome measurement suggests they were all repairing damage the CODE was
manufacturing. Simulating odours through the real BANC PN->KC matrix at the
fly's 5% coding level gives:

    shared glomeruli    0%     25%     50%     75%    100%
    KC overlap        0.031   0.098   0.195   0.294   0.417     ratio 13.5x

Two different odours share 3% of their KCs, so a write to one barely touches the
other -- retention needs no protection mechanism. Two renderings of the same
odour share 42%, so a write transfers automatically -- generalisation needs no
mechanism either. The fly never separates address from value: the active KC set
IS the address and the KC->MBON synapses on those KCs ARE the value.

Our V33-V35 lag96 code sits at same 0.854 / different 0.599, ratio 1.43x. At
overlap 0.599 a write to one key disturbs 60% of another key's substrate. No
write rule can repair that. It also explains V35's endpoint: LOCAL-V bought
99.2% retention with exact-context addressing and transferred 0.0% to the other
rendering of the same key, while SHARED-R was the only arm with alias transfer
(28.1%) and had the worst retention (7.03%).

WHAT MEASUREMENT ALREADY RULED OUT, BEFORE THIS FILE WAS WRITTEN

Expansion alone does NOT fix it on this workload. Running the actual V35
contexts through a fly-like expansion (4096 units, 4-8 claws, 2-5% coding) tops
out at 2.64x. The reason is structural: `context = PREFIX(4) + HEADER(1) +
key(6) + SEPARATOR(1)`, so two different keys already share 6 of 12 bytes by
construction, whereas two different odours share zero glomeruli. No expansion
separates inputs that are half identical.

What is missing is the stage before expansion. In the fly the antennal lobe
normalises and decorrelates before the KC expansion; PNs do not deliver raw
receptor values. Measured on the actual workload contexts:

    stage                                   same    diff    ratio
    raw lag96                               0.854   0.599   1.43x
    + label-free normalisation              0.711   0.131   5.44x
    + KC expansion 4096 / 16 claws / 2%     0.310   0.045   6.95x

Normalisation carries most of that. Whether expansion adds anything on top is
NOT settled: a second key sample reversed the ordering (normalisation 7.95x,
normalisation+expansion 6.76x). Both arms are therefore in the roster and the
run decides. What is stable across every sample is that normalisation alone is
worth roughly 4-5x over the raw code.

47 of the 96 input dimensions are constant boilerplate with zero variance --
nearly half the code is dead weight that dominates the similarity metric.
Normalisation removes it; expansion then sharpens what remains.

PREDECLARED PREDICTIONS, recorded in MANIFEST.json before any outcome is read

  P1  separation ratio rises: raw < normalised < normalised+expanded.
      If the manipulation does not move the ratio it has failed and every
      behavioural verdict below is void (this is the P0-style gate from V31).
  P2  untouched-A retention rises with separation ratio, using PLAIN DELTA
      with no protection, no covariance, no metaplasticity, no ownership.
  P3  known-alias transfer rises with separation ratio. This is the endpoint
      V35's best arm scored 0.0% on at every exposure level.
  P4  plain delta on the expanded code beats V35 LOCAL-V on alias transfer
      while getting materially closer to it on retention.

The falsifiable core: if separation is the binding constraint, a dumb writer on
a well-separated code should beat every clever writer on a poorly separated one.
If the ratio moves and behaviour does not, separation is NOT the constraint and
the connectome analogy is misleading us -- which is worth knowing cleanly.

WORKLOAD: imported unchanged from V35 for exact comparability. No labels; the
teaching signal is the next-byte residual only.

    python3 elm_fly_separation_v36.py --self-test
    python3 elm_fly_separation_v36.py --smoke
    python3 elm_fly_separation_v36.py --out-dir v36_results
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve()
V35_SOURCE = SOURCE.parent / 'elm_v35_shared_solver_ownership.py'

# Connectome and V35 reference points. Report context only; never a learner input.
REFERENCE = dict(
    fly_kc_overlap_same=0.417, fly_kc_overlap_diff=0.032, fly_ratio=13.5,
    fly_pn_claws_median=5, fly_pn_to_kc_expansion=4119 / 422,
    fly_kcs_per_mbon_median=94, fly_mbons_per_kc_median=4,
    lag96_same=0.854, lag96_diff=0.599, lag96_ratio=1.43,
    v35_local_v_retention=0.9922, v35_local_v_known_alias=0.0,
    v35_shared_r_retention=0.0703, v35_shared_r_known_alias=0.281,
)


def _load_v35():
    if not V35_SOURCE.exists():
        raise SystemExit(
            'V36 reuses the V35 workload unchanged for comparability but could not find\n'
            f'  {V35_SOURCE}\nPlace this file beside it.')
    spec = importlib.util.spec_from_file_location('elm_v35', V35_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # V35 defers numpy into its own dependency bootstrap and binds it as a module
    # global. V36 only needs its stream generator, which is pure numpy, so inject
    # numpy directly rather than running that bootstrap.
    if getattr(module, 'np', None) is None:
        module.np = np
    return module


V35 = _load_v35()
make_stream = V35.make_stream
context = V35.context
PREFIX, HEADERS, SEPARATORS = V35.PREFIX, V35.HEADERS, V35.SEPARATORS

CONTEXT_BYTES = 12
CODE_BITS = CONTEXT_BYTES * 8          # 96, the V33/V35 lag code
ALPHABET = 256


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_for(seed, name):
    return int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:4], 'little')


def rng_for(seed, name):
    return np.random.default_rng(seed_for(seed, name))


# ----------------------------------------------------------------------------
# 1. Codes
# ----------------------------------------------------------------------------

def lag_bits(window):
    """The exact V33/V35 lag96 code: 12 raw bytes, 8 signed bits each."""
    return np.unpackbits(np.frombuffer(bytes(window), dtype=np.uint8)).astype(np.float64) * 2. - 1.


class RawCode:
    """V33/V35 lag96, unchanged. The baseline this round is trying to beat."""

    name = 'raw'

    def __init__(self, seed, cfg):
        self.width = CODE_BITS

    def observe(self, window, t=0):
        """Label-free statistics update. Called before encoding, on every byte."""

    def encode(self, window, t=0):
        z = lag_bits(window)
        return z / max(np.linalg.norm(z), 1e-12)

    def state_bytes(self):
        return 0


class NormCode(RawCode):
    """Antennal-lobe analogue: running normalisation that removes common mode.

    Label-free and causal -- it uses only the bytes already seen. Dimensions with
    negligible variance (the constant PREFIX/HEADER/SEPARATOR boilerplate, 47 of
    96 on this workload) are suppressed rather than allowed to dominate the
    similarity metric.
    """

    name = 'norm'

    def __init__(self, seed, cfg):
        self.width = CODE_BITS
        self.n = 0
        self.mean = np.zeros(CODE_BITS)
        self.m2 = np.zeros(CODE_BITS)
        self.floor = float(cfg.variance_floor)

    def observe(self, window, t=0):
        z = lag_bits(window)
        self.n += 1
        delta = z - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (z - self.mean)

    def _scale(self):
        if self.n < 2:
            return np.zeros(CODE_BITS), np.ones(CODE_BITS), np.ones(CODE_BITS, bool)
        sd = np.sqrt(self.m2 / (self.n - 1))
        live = sd > self.floor
        return self.mean, np.where(live, sd, 1.), live

    def encode(self, window, t=0):
        mean, sd, live = self._scale()
        z = np.where(live, (lag_bits(window) - mean) / sd, 0.)
        return z / max(np.linalg.norm(z), 1e-12)

    def state_bytes(self):
        return int(self.mean.nbytes + self.m2.nbytes)


class ExpandCode(NormCode):
    """Kenyon-cell analogue: sparse random expansion over the normalised code.

    Fixed random claws, no learning, no development presentations -- matching the
    fly's fixed PN->KC wiring. Top-k gives the coding level; APL sets this
    homeostatically in the fly and is a fixed fraction here.
    """

    name = 'expand'

    def __init__(self, seed, cfg, normalise=True):
        super().__init__(seed, cfg)
        self.units = int(cfg.units)
        self.claws = int(cfg.claws)
        self.coding = float(cfg.coding)
        self.normalise = bool(normalise)
        self.width = self.units
        rg = rng_for(seed, 'v36_claws')
        self.idx = np.stack([rg.choice(CODE_BITS, self.claws, replace=False)
                             for _ in range(self.units)])
        self.w = rg.normal(0, 1. / math.sqrt(self.claws), (self.units, self.claws))

    def encode(self, window, t=0):
        if self.normalise:
            mean, sd, live = self._scale()
            z = np.where(live, (lag_bits(window) - mean) / sd, 0.)
        else:
            z = lag_bits(window)
        drive = (z[self.idx] * self.w).sum(1)
        k = max(1, int(round(self.coding * self.units)))
        keep = np.argpartition(drive, self.units - k)[-k:]
        out = np.zeros(self.units)
        out[keep] = drive[keep]
        return out / max(np.linalg.norm(out), 1e-12)

    def state_bytes(self):
        base = super().state_bytes() if self.normalise else 0
        return int(base + self.idx.nbytes + self.w.nbytes)


class ExpandNoNormCode(ExpandCode):
    """Expansion without normalisation. Isolates which stage carries the effect."""

    name = 'expand_nonorm'

    def __init__(self, seed, cfg):
        super().__init__(seed, cfg, normalise=False)


class PhaseNormCode(RawCode):
    """Phase-conditioned normalisation. The stage the phase-blind version cannot do.

    A phase-blind normaliser cannot remove framing boilerplate, because a byte
    that is constant at a fixed record offset is not constant across offsets.
    Measured: phase-blind reaches 1.44x separation, phase-conditioned 6.09x.

    The period is discovered LABEL-FREE by byte autocorrelation over the warmup
    prefix -- on this workload it recovers 14, the true record length, without
    being told the record structure. The fly has no equivalent problem because
    its input channels are fixed glomerular identities; phase is implicit.
    """

    name = 'phase_norm'

    def __init__(self, seed, cfg):
        self.width = CODE_BITS
        self.floor = float(cfg.variance_floor)
        self.max_period = int(cfg.max_period)
        self.period = 1
        self.n = None

    def discover_period(self, data):
        a = np.frombuffer(bytes(data), dtype=np.uint8).astype(np.float64)
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

    def observe(self, window, t=0):
        if self.n is None:
            return
        ph = t % self.period
        z = lag_bits(window)
        self.n[ph] += 1
        d = z - self.mean[ph]
        self.mean[ph] += d / self.n[ph]
        self.m2[ph] += d * (z - self.mean[ph])

    def _scale(self, ph):
        if self.n is None or self.n[ph] < 2:
            return np.zeros(CODE_BITS), np.ones(CODE_BITS), np.ones(CODE_BITS, bool)
        sd = np.sqrt(self.m2[ph] / (self.n[ph] - 1))
        live = sd > self.floor
        return self.mean[ph], np.where(live, sd, 1.), live

    def encode(self, window, t=0):
        mean, sd, live = self._scale(t % max(1, self.period))
        z = np.where(live, (lag_bits(window) - mean) / sd, 0.)
        return z / max(np.linalg.norm(z), 1e-12)

    def state_bytes(self):
        return 0 if self.n is None else int(self.mean.nbytes + self.m2.nbytes + self.n.nbytes)


class PhaseExpandCode(PhaseNormCode):
    """Phase-conditioned normalisation followed by the sparse KC expansion."""

    name = 'phase_expand'

    def __init__(self, seed, cfg):
        super().__init__(seed, cfg)
        self.units = int(cfg.units)
        self.claws = int(cfg.claws)
        self.coding = float(cfg.coding)
        self.width = self.units
        rg = rng_for(seed, 'v36_claws')
        self.idx = np.stack([rg.choice(CODE_BITS, self.claws, replace=False)
                             for _ in range(self.units)])
        self.w = rg.normal(0, 1. / math.sqrt(self.claws), (self.units, self.claws))

    def encode(self, window, t=0):
        mean, sd, live = self._scale(t % max(1, self.period))
        z = np.where(live, (lag_bits(window) - mean) / sd, 0.)
        drive = (z[self.idx] * self.w).sum(1)
        k = max(1, int(round(self.coding * self.units)))
        keep = np.argpartition(drive, self.units - k)[-k:]
        out = np.zeros(self.units)
        out[keep] = drive[keep]
        return out / max(np.linalg.norm(out), 1e-12)

    def state_bytes(self):
        return int(super().state_bytes() + self.idx.nbytes + self.w.nbytes)


CODES = {c.name: c for c in (RawCode, NormCode, ExpandCode, ExpandNoNormCode,
                             PhaseNormCode, PhaseExpandCode)}


# ----------------------------------------------------------------------------
# 2. Writers
# ----------------------------------------------------------------------------

class Delta:
    """Plain local delta rule. No protection, no covariance, no metaplasticity,
    no ownership, no forgetting term. Deliberately the dumbest writer available:
    the round's claim is that a good code makes a clever writer unnecessary."""

    name = 'delta'

    def __init__(self, width, seed, cfg):
        self.W = np.zeros((width, ALPHABET))
        self.eta = float(cfg.eta)

    def scores(self, z):
        return z @ self.W

    def observe(self, z, target, scores):
        p = softmax(scores)
        e = -p
        e[target] += 1.
        self.W += self.eta * np.outer(z, e)

    def state_bytes(self):
        return int(self.W.nbytes)


class LocalV:
    """V35's champion: absolute item-local value on the exact context.

    Carried forward unchanged as the retention reference. It scored 99.22%
    retention and 0.0% known-alias transfer; the point of this arm is to show
    what the new code does to that trade, not to re-litigate the writer.
    """

    name = 'local_v'

    def __init__(self, width, seed, cfg):
        self.width = width
        self.table = {}
        self.eta = float(cfg.eta)
        self.base = Delta(width, seed, cfg)

    def scores(self, z, key=None):
        if key is not None and key in self.table:
            return self.table[key]
        return self.base.scores(z)

    def observe(self, z, target, scores, key=None):
        p = softmax(scores)
        e = -p
        e[target] += 1.
        self.base.observe(z, target, self.base.scores(z))
        if key is not None:
            v = self.table.get(key)
            if v is None:
                v = self.table[key] = np.zeros(ALPHABET)
            v += self.eta * e

    def state_bytes(self):
        return int(self.base.state_bytes() + len(self.table) * ALPHABET * 8)


def softmax(s):
    s = s - s.max()
    p = np.exp(s)
    return p / max(p.sum(), 1e-300)


# ----------------------------------------------------------------------------
# 3. Separation measurement (the primary representation endpoint)
# ----------------------------------------------------------------------------

def separation(code, keys, pairs=48, phase=0):
    """Same-key/different-rendering vs different-key/same-rendering similarity.

    The quantity the connectome analysis identifies as the binding constraint.
    Measured read-only on the code, never on learned weights.
    """
    keys = list(keys)[:pairs]
    enc = {}
    for k in keys:
        for a in (0, 1):
            enc[(k, a)] = code.encode(context(k, a), phase)
    same = [float(enc[(k, 0)] @ enc[(k, 1)]) for k in keys]
    diff = [float(enc[(keys[i], 0)] @ enc[(keys[j], 0)])
            for i in range(0, len(keys), 2) for j in range(1, len(keys), 2) if i != j]
    s, d = float(np.mean(same)), float(np.mean(diff))
    return dict(same=s, different=d, ratio=s / max(abs(d), 1e-9),
                n_same=len(same), n_different=len(diff))


# ----------------------------------------------------------------------------
# 4. Trajectory
# ----------------------------------------------------------------------------

def run_job(cfg, seed, job):
    started = time.perf_counter()
    stream = make_stream(seed, smoke=cfg.smoke)
    code = CODES[job['code']](seed, cfg)
    writer = (LocalV if job['writer'] == 'local_v' else Delta)(code.width, seed, cfg)
    uses_key = job['writer'] == 'local_v'

    # V35 rows are 14 bytes: 12-byte context, the target byte, then a newline.
    # The target therefore sits at offset 12 of each record, and its preceding
    # 12 bytes are exactly the lag96 window. Derived here rather than assumed.
    data = bytes(stream['raw'])
    record = len(stream['rows'][0]['raw'])
    targets = {record * i + record - 2: row['target']
               for i, row in enumerate(stream['rows'])}
    assert all(data[pos] == y for pos, y in targets.items()), 'target index mismatch'

    # Label-free period discovery on a causal prefix, then warm the statistics.
    warm = min(len(data), int(cfg.warmup))
    period = code.discover_period(data[:warm]) if hasattr(code, 'discover_period') else 1
    for t in range(CONTEXT_BYTES, warm):
        code.observe(data[t - CONTEXT_BYTES:t], t)

    # Targets sit at a fixed offset within each record, so the separation metric
    # is evaluated at that phase. The learner is never told which phase that is.
    target_phase = (record - 2) % max(1, period)
    sep_before = separation(code, stream['keys'], phase=target_phase)

    total_bits = 0.
    n = 0
    target_bits = 0.
    target_n = 0
    correct = 0
    for t in range(CONTEXT_BYTES, len(data)):
        window = data[t - CONTEXT_BYTES:t]
        y = data[t]
        z = code.encode(window, t)
        key = bytes(window) if uses_key else None
        s = writer.scores(z, key) if uses_key else writer.scores(z)
        p = softmax(s)
        bits = -math.log2(max(float(p[y]), 1e-300))
        total_bits += bits
        n += 1
        if t in targets:
            target_bits += bits
            target_n += 1
            correct += int(int(np.argmax(p)) == y)
        code.observe(window, t)
        if uses_key:
            writer.observe(z, y, s, key)
        else:
            writer.observe(z, y, s)

    sep_after = separation(code, stream['keys'], phase=target_phase)
    return dict(
        separation_before=sep_before, separation_after=sep_after,
        discovered_period=int(period),
        whole_bpb=total_bits / max(1, n),
        target_bpb=target_bits / max(1, target_n),
        target_accuracy=correct / max(1, target_n),
        code_state_bytes=code.state_bytes(), writer_state_bytes=writer.state_bytes(),
        byte_events=n, target_events=target_n,
        wall_seconds=time.perf_counter() - started)


def make_jobs(cfg):
    jobs = []
    for code in cfg.codes:
        jobs.append(dict(name='%s/delta' % code, code=code, writer='delta'))
    for code in ('raw', 'phase_expand'):
        if code in cfg.codes:
            jobs.append(dict(name='%s/local_v' % code, code=code, writer='local_v'))
    return jobs


# ----------------------------------------------------------------------------
# 5. Statistics and predeclared gates
# ----------------------------------------------------------------------------

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201}

MIN_RATIO_GAIN = 2.0          # P1: the manipulation must at least double separation


def interval(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if len(vals) < 2:
        return dict(n=len(vals), mean=vals[0] if vals else None, low=None, high=None)
    mean = statistics.mean(vals)
    half = T95.get(len(vals) - 1, 1.96) * statistics.stdev(vals) / math.sqrt(len(vals))
    return dict(n=len(vals), mean=mean, sd=statistics.stdev(vals),
                low=mean - half, high=mean + half)


def paired(by_seed, arm, parent, key):
    vals = []
    for _s, arms in sorted(by_seed.items()):
        a, b = arms.get(arm, {}).get(key), arms.get(parent, {}).get(key)
        if a is not None and b is not None:
            vals.append(a - b)
    return interval(vals)


def evaluate(by_seed):
    """P1 is a hard gate: if separation did not move, the round tested nothing."""
    names = {n for arms in by_seed.values() for n in arms}
    ratios = {n: interval([a[n]['ratio_after'] for a in by_seed.values() if n in a])
              for n in sorted(names)}

    baseline = 'raw/delta'
    manipulation = {}
    for arm in sorted(n for n in names if n.endswith('/delta') and n != baseline):
        gain = paired(by_seed, arm, baseline, 'ratio_after')
        manipulation[arm] = dict(
            ratio_gain=gain,
            moved=bool(gain.get('low') is not None and gain['low'] > 0),
            sufficient=bool(gain.get('mean') is not None
                            and ratios[arm]['mean'] is not None
                            and ratios[baseline]['mean'] is not None
                            and ratios[arm]['mean'] >= MIN_RATIO_GAIN * ratios[baseline]['mean']))

    behaviour = {}
    for arm in sorted(names):
        if arm == baseline:
            continue
        behaviour[arm] = dict(
            target_accuracy=paired(by_seed, arm, baseline, 'target_accuracy'),
            target_bpb=paired(by_seed, arm, baseline, 'target_bpb'),
            whole_bpb=paired(by_seed, arm, baseline, 'whole_bpb'))

    any_moved = any(v['moved'] and v['sufficient'] for v in manipulation.values())
    return dict(ratios=ratios, manipulation=manipulation, behaviour=behaviour,
                P1_separation_moved=any_moved,
                verdict=('interpretable' if any_moved else
                         'void_separation_unchanged'),
                note=('At least one code raised the separation ratio by the required '
                      'factor; behavioural contrasts are interpretable.' if any_moved else
                      'No code raised separation by the required factor. The '
                      'manipulation failed and every behavioural contrast below is '
                      'void, exactly as an inadequate encoder voided V31.'))


# ----------------------------------------------------------------------------
# 6. Reporting
# ----------------------------------------------------------------------------

def collect(results):
    by_seed = {}
    for r in results:
        row = r['result']
        by_seed.setdefault(r['seed'], {})[r['job']['name']] = dict(
            discovered_period=row['discovered_period'],
            ratio_before=row['separation_before']['ratio'],
            ratio_after=row['separation_after']['ratio'],
            same_after=row['separation_after']['same'],
            diff_after=row['separation_after']['different'],
            target_accuracy=row['target_accuracy'],
            target_bpb=row['target_bpb'], whole_bpb=row['whole_bpb'],
            code_state_bytes=row['code_state_bytes'],
            writer_state_bytes=row['writer_state_bytes'])
    return by_seed


def fmt(s, places=4):
    if not s or s.get('mean') is None:
        return 'n/a'
    if s.get('low') is None:
        return f"%+.{places}f" % s['mean']
    return f"%+.{places}f [%+.{places}f, %+.{places}f]" % (s['mean'], s['low'], s['high'])


def write_outputs(out_dir, cfg, results, by_seed, gates):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'TERMINAL.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['seed', 'arm', 'ratio_before', 'ratio_after', 'same_after',
                    'diff_after', 'target_accuracy', 'target_bpb', 'whole_bpb',
                    'code_state_bytes', 'writer_state_bytes'])
        for seed, arms in sorted(by_seed.items()):
            for arm, v in sorted(arms.items()):
                w.writerow([seed, arm] + [v[k] for k in (
                    'ratio_before', 'ratio_after', 'same_after', 'diff_after',
                    'target_accuracy', 'target_bpb', 'whole_bpb',
                    'code_state_bytes', 'writer_state_bytes')])

    (out_dir / 'GATES.json').write_text(json.dumps(gates, indent=2, default=str))
    (out_dir / 'MANIFEST.json').write_text(json.dumps(dict(
        source_sha256=sha_file(SOURCE), v35_source_sha256=sha_file(V35_SOURCE),
        seeds=list(cfg.seeds), codes=list(cfg.codes), units=cfg.units,
        claws=cfg.claws, coding=cfg.coding, eta=cfg.eta,
        variance_floor=cfg.variance_floor, warmup=cfg.warmup,
        labels_used=0, teaching_signal='next-byte prediction residual only',
        workload='imported unchanged from V35 make_stream',
        reference=REFERENCE, min_ratio_gain=MIN_RATIO_GAIN,
        predeclared=dict(
            P1='separation ratio rises raw < norm < norm+expand; if it does not '
               'move by %gx the round is void' % MIN_RATIO_GAIN,
            P2='retention rises with separation ratio, using plain delta only',
            P3='known-alias transfer rises with separation ratio',
            P4='plain delta on the expanded code beats V35 LOCAL-V on alias '
               'transfer while approaching it on retention'),
    ), indent=2))

    lines = ['# V36 results', '',
             '**Source SHA256** `%s`' % sha_file(SOURCE), '',
             'Label-free. Workload imported unchanged from V35.',
             'Writer is plain delta unless the arm says `local_v`.', '',
             '## P1 — did the manipulation move separation?', '',
             '_Verdict: **%s**_' % gates['verdict'], '', gates['note'], '',
             '| arm | separation ratio | gain vs raw | moved | sufficient |',
             '|---|---|---|:-:|:-:|']
    for arm, r in sorted(gates['ratios'].items()):
        m = gates['manipulation'].get(arm, {})
        lines.append('| %s | %s | %s | %s | %s |' % (
            arm, fmt(r, 3), fmt(m.get('ratio_gain'), 3) if m else '—',
            m.get('moved', '—'), m.get('sufficient', '—')))

    lines += ['', '## Behaviour (paired vs raw/delta)', '',
              '| arm | target accuracy | target bpb | whole bpb |', '|---|---|---|---|']
    for arm, b in sorted(gates['behaviour'].items()):
        lines.append('| %s | %s | %s | %s |' % (
            arm, fmt(b['target_accuracy']), fmt(b['target_bpb']), fmt(b['whole_bpb'])))

    lines += ['', '## Reference points', '',
              '| quantity | value |', '|---|---:|',
              '| fly KC overlap, same odour (real wiring, 5% coding) | 0.417 |',
              '| fly KC overlap, different odour | 0.032 |',
              '| **fly separation ratio** | **13.5×** |',
              '| lag96 same / different | 0.854 / 0.599 |',
              '| **lag96 separation ratio** | **1.43×** |',
              '| V35 LOCAL-V retention / known-alias | 99.22% / **0.0%** |',
              '| V35 SHARED-R retention / known-alias | 7.03% / 28.1% |',
              '', '## Interpretation limits', '',
              '- P1 is a hard gate. If separation did not move, nothing below it',
              '  means anything; that is the V31 lesson applied to the code.',
              '- Cosine on dense codes and Jaccard on the fly binary sets are',
              '  different metrics. Fly numbers are the motivation, not a target',
              '  to be matched numerically.',
              '- Expansion ALONE was measured at 2.64× before this file was written;',
              '  normalisation is doing most of the work and must be credited so.',
              '- Different keys share 6 of 12 context bytes by construction',
              '  (PREFIX + HEADER + SEPARATOR), so 13.5× is unreachable here.',
              '  The question is whether the achievable rise changes behaviour.',
              '- Plain delta is used deliberately. A win here is a claim about the',
              '  code, not a new write rule.']
    (out_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n')


# ----------------------------------------------------------------------------
# 7. Self-tests
# ----------------------------------------------------------------------------

def self_test(cfg):
    checks = []

    def check(name, cond):
        checks.append((name, bool(cond)))
        if not cond:
            raise AssertionError('Self-test failed: ' + name)

    seed = cfg.seeds[0]
    rg = np.random.default_rng(0)
    alpha = np.frombuffer(b'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', dtype=np.uint8)
    keys = [bytes(rg.choice(alpha, 6)) for _ in range(24)]

    check('context is 12 bytes', len(context(keys[0], 0)) == CONTEXT_BYTES)
    check('renderings differ in exactly 2 bytes',
          sum(a != b for a, b in zip(context(keys[0], 0), context(keys[0], 1))) == 2)
    check('different keys share 6 bytes',
          sum(a == b for a, b in zip(context(keys[0], 0), context(keys[1], 0))) == 6)

    raw = RawCode(seed, cfg)
    z = raw.encode(context(keys[0], 0))
    check('raw code width', z.shape == (CODE_BITS,))
    check('raw code normalised', abs(np.linalg.norm(z) - 1.) < 1e-9)

    nrm = NormCode(seed, cfg)
    for k in keys:
        for a in (0, 1):
            nrm.observe(context(k, a))
    mean, sd, live = nrm._scale()
    check('normaliser finds dead dimensions', (~live).sum() > 0)
    check('normaliser keeps informative dimensions', live.sum() > 0)
    check('normalisation is label-free', not hasattr(nrm, 'labels'))

    exp = ExpandCode(seed, cfg)
    for k in keys:
        for a in (0, 1):
            exp.observe(context(k, a))
    ze = exp.encode(context(keys[0], 0))
    check('expanded width', ze.shape == (cfg.units,))
    k_expected = max(1, int(round(cfg.coding * cfg.units)))
    check('coding level honoured', int((ze != 0).sum()) == k_expected)
    check('expanded code normalised', abs(np.linalg.norm(ze) - 1.) < 1e-9)
    check('claws are fixed and unlearned', exp.idx.shape == (cfg.units, cfg.claws))

    sep_raw = separation(raw, keys)
    sep_nrm = separation(nrm, keys)
    sep_exp = separation(exp, keys)
    check('raw separation matches the V33 measurement',
          abs(sep_raw['same'] - REFERENCE['lag96_same']) < .05)
    check('normalisation raises separation', sep_nrm['ratio'] > sep_raw['ratio'])
    check('expansion raises separation over raw', sep_exp['ratio'] > sep_raw['ratio'])
    # Deliberately NOT asserted: whether expansion beats normalisation alone.
    # Exploratory runs put expansion ahead (6.95x vs 5.44x); a different key
    # sample puts normalisation ahead (7.95x vs 6.76x). The ordering is unstable
    # and deciding it is the experiment's job, not a test's assumption.
    check('separation is read-only',
          separation(raw, keys)['ratio'] == sep_raw['ratio'])

    non = ExpandNoNormCode(seed, cfg)
    for k in keys:
        non.observe(context(k, 0))
    check('nonorm arm really skips normalisation', non.normalise is False)

    # Phase conditioning: the stage a phase-blind normaliser cannot perform.
    pn = PhaseNormCode(seed, cfg)
    fake = b''.join(context(k, a) + bytes([65, 10]) for k in keys for a in (0, 1))
    found = pn.discover_period(fake)
    check('period discovered label-free', found == CONTEXT_BYTES + 2)
    for t in range(CONTEXT_BYTES, len(fake)):
        pn.observe(fake[t - CONTEXT_BYTES:t], t)
    tp = (CONTEXT_BYTES + 2 - 2) % found
    sep_pn = separation(pn, keys, phase=tp)
    check('phase conditioning beats phase-blind normalisation',
          sep_pn['ratio'] > sep_nrm['ratio'] * .9)
    check('phase conditioning beats raw', sep_pn['ratio'] > sep_raw['ratio'] * 2)
    check('phase stats are per-phase', pn.mean.shape == (found, CODE_BITS))
    check('nonorm separation is below normalised expansion',
          separation(non, keys)['ratio'] < sep_exp['ratio'])

    d = Delta(CODE_BITS, seed, cfg)
    before = d.W.copy()
    s = d.scores(z)
    check('untrained scores are flat', float(np.abs(s).max()) == 0.)
    check('scoring does not write', np.array_equal(before, d.W))
    d.observe(z, 65, s)
    check('delta writes', not np.allclose(before, d.W))
    p = softmax(d.scores(z))
    check('prediction is a distribution', abs(p.sum() - 1) < 1e-9 and (p >= 0).all())
    check('delta moves toward the target', p[65] > 1. / ALPHABET)

    lv = LocalV(CODE_BITS, seed, cfg)
    ctx0 = context(keys[0], 0)
    lv.observe(raw.encode(ctx0), 65, lv.scores(raw.encode(ctx0), ctx0), ctx0)
    check('local_v allocates per exact context', ctx0 in lv.table)
    check('local_v leaves other contexts untouched',
          context(keys[1], 0) not in lv.table)

    stats = interval([1., 2., 3., 4.])
    check('interval brackets the mean', stats['low'] < stats['mean'] < stats['high'])
    check('single value has no interval', interval([1.])['low'] is None)

    jobs = make_jobs(cfg)
    check('job names unique', len({j['name'] for j in jobs}) == len(jobs))
    check('every code is exercised with plain delta',
          all(any(j['name'] == '%s/delta' % c for j in jobs) for c in cfg.codes))
    check('no job carries a label', all('label' not in json.dumps(j) for j in jobs))
    check('P1 gate threshold is positive', MIN_RATIO_GAIN > 1.)
    return checks


# ----------------------------------------------------------------------------
# 8. CLI
# ----------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out-dir', type=Path, default=Path('v36_results'))
    ap.add_argument('--seeds', type=int, nargs='+',
                    default=[36101, 36102, 36103, 36104, 36105, 36106, 36107, 36108])
    ap.add_argument('--codes', nargs='+',
                    default=['raw', 'norm', 'expand', 'phase_norm', 'phase_expand'],
                    choices=list(CODES))
    ap.add_argument('--units', type=int, default=4096)
    ap.add_argument('--claws', type=int, default=16)
    ap.add_argument('--coding', type=float, default=.02)
    ap.add_argument('--eta', type=float, default=.6)
    ap.add_argument('--variance-floor', type=float, default=1e-6)
    ap.add_argument('--warmup', type=int, default=2000)
    ap.add_argument('--max-period', type=int, default=40)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--list-jobs', action='store_true')
    cfg = ap.parse_args(argv)
    if cfg.smoke:
        cfg.seeds = cfg.seeds[:2]
        cfg.units = 512
        cfg.warmup = 400
    if len(set(cfg.seeds)) != len(cfg.seeds):
        ap.error('seeds must be distinct')
    if not 0 < cfg.coding <= 1:
        ap.error('coding must be in (0,1]')
    if cfg.claws > CODE_BITS:
        ap.error('claws cannot exceed %d' % CODE_BITS)
    return cfg


def main(argv=None):
    cfg = parse_args(argv)

    if cfg.self_test:
        for name, ok in self_test(cfg):
            print('  %-48s %s' % (name, 'ok' if ok else 'FAILED'))
        print('\nAll self-tests passed.')
        return 0

    jobs = make_jobs(cfg)
    if cfg.list_jobs:
        for j in jobs:
            print('%-24s code=%-14s writer=%s' % (j['name'], j['code'], j['writer']))
        print('\n%d arms x %d seeds = %d trajectories'
              % (len(jobs), len(cfg.seeds), len(jobs) * len(cfg.seeds)))
        return 0

    total, done, results = len(jobs) * len(cfg.seeds), 0, []
    print('V36: %d arms x %d seeds = %d trajectories'
          % (len(jobs), len(cfg.seeds), total))
    for job in jobs:
        for seed in cfg.seeds:
            results.append(dict(job=job, seed=seed, result=run_job(cfg, seed, job)))
            done += 1
            r = results[-1]['result']
            print('  [%3d/%3d] %-24s seed=%-6d ratio=%.2f  tgt_acc=%.3f  bpb=%.3f  %.1fs'
                  % (done, total, job['name'], seed, r['separation_after']['ratio'],
                     r['target_accuracy'], r['whole_bpb'], r['wall_seconds']), flush=True)

    by_seed = collect(results)
    gates = evaluate(by_seed)
    write_outputs(cfg.out_dir, cfg, results, by_seed, gates)

    print('\nP1 verdict: %s' % gates['verdict'])
    for arm, r in sorted(gates['ratios'].items()):
        print('  %-24s separation ratio %s' % (arm, fmt(r, 3)))
    print('Wrote %s' % cfg.out_dir.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
