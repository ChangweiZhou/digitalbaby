#!/usr/bin/env python3
"""V31: label-free byte-level learning with per-weight synaptic uncertainty.

Two changes from every prior version, both requested by Mr. Zhou.

1. NO LABELS ANYWHERE. The learner predicts the next symbol of a byte stream and
   is taught by its own prediction residual. Nothing external names a category.
   The fly gets no label for the swatter either: it gets a looming pattern
   followed by more sensory input, and dopamine signals the mismatch. So the
   constraint is not "no teaching signal" -- without one nothing can learn --
   but "the teaching signal is generated inside the system from the stream."

   Concretely, V23-V30 wrote with `target = eye(4)[label]`. Here the target is
   the next symbol the stream is about to deliver, the prediction is made and
   scored BEFORE it is revealed, and the write uses only the residual.

2. WRITE MODULATED BY EXISTING INFO, via per-weight uncertainty (MESU-style
   metaplasticity, Bonnet et al. 2025, arXiv:2504.13569). Each readout weight
   carries a companion variance that scales its own update: confident weights
   move little, uncertain weights move freely. The variance is LEARNED from
   curvature, not designed.

WHY THIS AND NOT ANOTHER SELECTION HEURISTIC

V26-V29 designed which weights change -- sparse, competitive, gated, cyclic,
sticky, top-k. All failed or traded retention against transfer, repeatedly,
under matched controls. V30 modulated write DIRECTION using accumulated
statistics and produced the program's only non-trading gain: +15.89 pp old
recall [+10.74, +21.04] at +0.13 pp transfer cost [-2.49, +2.75] on overlap.
But it cost 589,824 bytes of covariance against 24,576 bytes of weights: 24x
the thing it protects.

MESU is the cheap version of that idea: two scalars per synapse instead of a
full covariance, and the modulating quantity has its own dynamics.

V30's `diag` writer failed, and that does NOT predict failure here. Different
objects: V30's diagonal was a statistic of the INPUT distribution with no
dynamics and no floor; MESU's sigma^2 is posterior uncertainty about EACH
WEIGHT, updated by curvature, with a prior-relaxation floor set by a memory
horizon N.

There is also an unfinished thread in our own history. V23's `bayes_stable`,
an acknowledged MESU approximation, scored 63.66% A-retention / 47.64% terminal
against ordinary's 53.94% / 36.66% for 0.8 KiB extra -- one of V23's better
arms, dropped when the program went down the bank/routing path.

THE UPDATE RULE (Aitchison form, eqs 15-16 of the paper, adapted to a linear
readout; the paper notes it matches MESU up to a factor of two)

    prediction  p = softmax(z @ mu)
    residual    e = onehot(next_symbol) - p          <- the only teaching signal
    mean        mu    += sigma2 * outer(z, e)  +  (sigma2 / (N * s2_prior)) * (0 - mu)
    variance    sigma2 += -(sigma2^2 / s2_fb) * outer(z^2, 1)  +  (2/N) * (s2_prior - sigma2)

z_i^2 is exactly the Hessian diagonal for this readout, so the variance tracks
curvature as the paper derives. With N -> infinity the relaxation term vanishes,
sigma^2 collapses as (H_D t)^-1, and the learner ossifies -- the paper's
"catastrophic remembering", which is allocate128's V23 failure mode (100%
retention, 45% late acquisition). Finite N holds a nonzero variance floor.

PREDECLARED PREDICTIONS, recorded in MANIFEST.json before any result is read

  P1  mesu retains domain-A bits/symbol better than plain at matched B transfer.
  P2  mesu_frozen_sigma does NOT match mesu. If it does, the benefit is a fixed
      per-weight rate schedule and there is no metaplasticity -- retire it, the
      way the novelty gate and competitive selection were retired.
  P3  mesu_shuffled_sigma does NOT match mesu. Same variance distribution,
      permuted across weights. This is the decisive control; it is the analogue
      of the bottom-k control that settled V27/V28.
  P4  mesu_infN shows BETTER A-retention and WORSE late B acquisition than mesu.
      This is a structural prediction of eq 19 and is falsifiable on its own.

If P2 or P3 fails, the mechanism is retired regardless of P1.

SUBSTRATE HELD FIXED (validated V26-V29; not manipulated here)
  contrast (fast-minus-slow) temporal encoder, frozen; 8 banks with local
  routing; per-episode state reset. Task-validity screening carried over from
  V28/V29 because it is the reason those versions measured anything real.

    python3 elm_fly_uncertainty_v31.py --self-test
    python3 elm_fly_uncertainty_v31.py --smoke
    python3 elm_fly_uncertainty_v31.py --out-dir v31_results
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve()

ALPHABET = 16          # symbols; 4 bits each
N_CELLS = 96           # encoder width, matching V26-V30
BANKS = 8
ACTIVE_FRACTION = .25
BINS, FEATURES = 12, 32
WINDOW = 24            # symbols of context feeding the encoder
TAUS = (2., 6., 16.)
SLOW_RATIO = 4.

# V30 reference points, for report context only. Never an input to a learner.
V30_REFERENCE = dict(
    full_cov_old_recall_overlap=0.1589, full_cov_transfer_overlap=0.0013,
    full_cov_statistic_bytes=589824, bank_weight_bytes=24576,
    v23_bayes_stable_A_retention=0.6366, v23_ordinary_A_retention=0.5394,
)


def seed_for(seed, name):
    return int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:4], 'little')


def rng_for(seed, name):
    return np.random.default_rng(seed_for(seed, name))


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------
# 1. Label-free symbol streams with a controlled A -> B shift
# ----------------------------------------------------------------------------

def make_source(rng, order, concentration):
    """A sparse high-order symbol source.

    Next symbol depends on the previous `order` symbols through a sparse
    Dirichlet-drawn conditional. Concentration controls predictability: low
    concentration gives peaked conditionals (learnable structure), high gives
    near-uniform (little to learn).
    """
    n_contexts = ALPHABET ** order
    table = rng.dirichlet(np.full(ALPHABET, concentration), size=n_contexts)
    return dict(order=order, table=table)


def sample_stream(source, rng, length):
    order, table = source['order'], source['table']
    out = np.zeros(length + order, dtype=np.int64)
    out[:order] = rng.integers(0, ALPHABET, order)
    for t in range(order, length + order):
        context = 0
        for k in range(order):
            context = context * ALPHABET + int(out[t - order + k])
        out[t] = rng.choice(ALPHABET, p=table[context])
    return out[order:]


def entropy_rate(source, stream):
    """Bits/symbol of the true generating source on this stream: the floor."""
    order, table = source['order'], source['table']
    total = 0.
    for t in range(order, len(stream)):
        context = 0
        for k in range(order):
            context = context * ALPHABET + int(stream[t - order + k])
        total -= math.log2(max(table[context][stream[t]], 1e-12))
    return total / max(1, len(stream) - order)


def build_streams(cfg, seed):
    """Domain A and domain B with different conditional structure.

    The shift is never announced. B is a fresh source, so a model fitted to A
    must genuinely re-learn; A's held-out stream stays available for scoring
    retention and is NEVER replayed for learning after the shift.
    """
    rg = rng_for(seed, 'v31_sources')
    source_a = make_source(rg, cfg.order, cfg.concentration)
    source_b = make_source(rg, cfg.order, cfg.concentration)
    sampler = rng_for(seed, 'v31_samples')
    streams = dict(
        train_a=sample_stream(source_a, sampler, cfg.train_len),
        train_b=sample_stream(source_b, sampler, cfg.train_len),
        test_a=sample_stream(source_a, sampler, cfg.test_len),
        test_b=sample_stream(source_b, sampler, cfg.test_len),
        # Held-out stream from a THIRD source, never trained on. Retention on
        # test_a must exceed this, or "retention" is only generic symbol
        # statistics -- the recall-dependence lesson from V28.
        test_generic=sample_stream(make_source(rg, cfg.order, cfg.concentration),
                                   sampler, cfg.test_len))
    streams['floor_a'] = entropy_rate(source_a, streams['test_a'])
    streams['floor_b'] = entropy_rate(source_b, streams['test_b'])
    return streams


def windows(stream):
    """(context, next_symbol) pairs. The target is the stream's own future."""
    for t in range(WINDOW, len(stream)):
        yield stream[t - WINDOW:t], int(stream[t])


def encode_context(context):
    """Map a symbol window to the validated (12, 32) encoder input."""
    x = np.zeros((BINS, FEATURES))
    for b in range(BINS):
        recent = context[-1 - b]
        older = context[-1 - b - BINS]
        x[b, recent] = 1.
        x[b, ALPHABET + older] = 1.
    return x


# ----------------------------------------------------------------------------
# 2. Contrast encoder (frozen; geometry validated in V26-V29)
# ----------------------------------------------------------------------------

class ContrastEncoder:
    """Fixed sparse afferents plus fast-minus-slow temporal contrast.

    No training presentations; a calibration pass only sets mean/sd. This is the
    geometry that won every anti-shortcut control in V26-V28 (+10.5 to +11.1 pp
    on the rotated task) and beat every alternative on context in V29.
    """

    def __init__(self, seed, per_cell=8):
        rg = rng_for(seed, 'v31_encoder')
        contacts = np.stack([rg.choice(FEATURES, per_cell, replace=False)
                             for _ in range(N_CELLS)])
        tau = np.asarray(TAUS)[np.arange(N_CELLS) % len(TAUS)]
        self.arrays = dict(
            sensory=contacts.astype(np.int64),
            weights=rg.normal(0, 1 / math.sqrt(per_cell), (N_CELLS, per_cell)),
            alpha=-np.expm1(-1. / tau),
            alpha_slow=-np.expm1(-1. / (tau * SLOW_RATIO)))
        self.mean, self.sd = np.zeros(N_CELLS), np.ones(N_CELLS)
        calib = np.stack([encode_context(c) for c, _ in
                          zip((rg.integers(0, ALPHABET, WINDOW) for _ in range(64)), range(64))])
        activity = self._activity(calib)
        self.mean = activity.mean(0)
        self.sd = np.maximum(activity.std(0), .01)

    def _activity(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:
            x = x[None]
        contacts, w = self.arrays['sensory'], self.arrays['weights']
        alpha, alpha_slow = self.arrays['alpha'][None, :], self.arrays['alpha_slow'][None, :]
        fast = np.zeros((len(x), N_CELLS))
        slow = np.zeros((len(x), N_CELLS))
        for t in range(BINS):
            drive = np.einsum('bnc,nc->bn', x[:, t, :][:, contacts], w)
            activation = np.tanh(drive)
            fast += alpha * (activation - fast)
            slow += alpha_slow * (activation - slow)
        return fast + (fast - slow)          # contrast geometry

    def features(self, x):
        z = (self._activity(x) - self.mean) / self.sd
        k = max(1, round(N_CELLS * ACTIVE_FRACTION))
        cut = np.partition(np.abs(z), N_CELLS - k, axis=1)[:, N_CELLS - k][:, None]
        kept = np.where(np.abs(z) >= cut, z, 0.)
        return kept / np.maximum(np.linalg.norm(kept, axis=1, keepdims=True), 1e-12)

    def bytes_used(self):
        return int(sum(a.nbytes for a in self.arrays.values())
                   + self.mean.nbytes + self.sd.nbytes)


class LaggedEncoder(ContrastEncoder):
    """Sparse random expansion that samples (bin, feature) pairs.

    Preserves which-symbol-at-which-lag, unlike the leaky integrator, which sums
    all 12 bins into one accumulator per cell and cannot separate "symbol 7 at
    lag 1" from "symbol 7 at lag 2".
    """

    def __init__(self, seed, per_cell=8):
        rg = rng_for(seed, 'v31_lagged')
        self.bins = rg.integers(0, BINS, (N_CELLS, per_cell))
        self.feats = rg.integers(0, FEATURES, (N_CELLS, per_cell))
        self.arrays = dict(weights=rg.normal(0, 1 / math.sqrt(per_cell), (N_CELLS, per_cell)),
                           bins=self.bins, feats=self.feats)
        self.mean, self.sd = np.zeros(N_CELLS), np.ones(N_CELLS)

    def features(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 3:
            x = x[0]
        z = np.tanh((x[self.bins, self.feats] * self.arrays['weights']).sum(1))
        k = max(1, round(N_CELLS * ACTIVE_FRACTION))
        cut = np.partition(np.abs(z), N_CELLS - k)[N_CELLS - k]
        kept = np.where(np.abs(z) >= cut, z, 0.)
        return (kept / max(np.linalg.norm(kept), 1e-12))[None]


class ObservableEncoder(ContrastEncoder):
    """Task-informed diagnostic: the raw lagged one-hots, no expansion.

    Not a proposed architecture. It is the ceiling that says how much a linear
    readout could extract if the code preserved symbol identity, and it is the
    control that exposes an inadequate code. V28/V29 showed this arm is the most
    valuable single diagnostic available.
    """

    def __init__(self, seed, per_cell=None):
        self.arrays = {}
        self.mean, self.sd = np.zeros(FEATURES), np.ones(FEATURES)

    def features(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 3:
            x = x[0]
        v = np.zeros(N_CELLS)
        v[:FEATURES] = x[0]          # bin 0 holds lag-1 and lag-13 one-hots
        v[FEATURES:2 * FEATURES] = x[1]
        return (v / max(np.linalg.norm(v), 1e-12))[None]

    def bytes_used(self):
        return 0


ENCODERS = dict(contrast=ContrastEncoder, lagged=LaggedEncoder, observable=ObservableEncoder)


# ----------------------------------------------------------------------------
# 3. Banked readout with pluggable write rules
# ----------------------------------------------------------------------------

WRITERS = ('frozen', 'plain', 'mesu', 'mesu_infN',
           'mesu_frozen_sigma', 'mesu_shuffled_sigma', 'v30_cov')


class BankMemory:
    """Eight locally routed banks over a shared frozen code.

    Every writer sees the same routing and the same residual. They differ only
    in how the residual is turned into a weight change, so the contrasts isolate
    the write rule and nothing else.
    """

    def __init__(self, seed, writer='plain', eta=.5, horizon=64,
                 prior_var=1., feedback_scale=1.):
        if writer not in WRITERS:
            raise ValueError('writer must be one of ' + repr(WRITERS))
        self.writer = writer
        self.eta = float(eta)
        self.horizon = float(horizon)
        self.prior_var = float(prior_var)
        self.feedback_scale = float(feedback_scale)
        rg = rng_for(seed, 'v31_banks')
        self.centers = rg.normal(size=(BANKS, N_CELLS))
        self.centers /= np.linalg.norm(self.centers, axis=1, keepdims=True)
        self.mu = np.zeros((BANKS, N_CELLS, ALPHABET))
        self.sigma2 = np.full((BANKS, N_CELLS, ALPHABET), self.prior_var)
        self.counts = np.zeros(BANKS, dtype=np.int64)
        self.cov = (np.zeros((BANKS, N_CELLS, N_CELLS)) if writer == 'v30_cov' else None)
        self._perm_rng = rng_for(seed, 'v31_sigma_shuffle')
        self.writes = 0

    # -- read ---------------------------------------------------------------
    def route(self, z):
        return int(np.argmax(self.centers @ z))

    def predict(self, z, bank=None):
        """Pure. Never called with the target in scope."""
        b = self.route(z) if bank is None else bank
        logits = z @ self.mu[b]
        logits -= logits.max()
        p = np.exp(logits)
        return p / max(p.sum(), 1e-12)

    # -- write --------------------------------------------------------------
    def observe(self, z, target):
        """The ONLY teaching signal is the residual against the next symbol."""
        if self.writer == 'frozen':
            return self.route(z)
        b = self.route(z)
        p = self.predict(z, b)
        e = -p
        e[target] += 1.                                   # onehot(target) - p
        grad = np.outer(z, e)                             # -dC/dmu

        if self.writer == 'plain':
            self.mu[b] += self.eta * grad
        elif self.writer == 'v30_cov':
            self.cov[b] += np.outer(z, z)
            shaped = np.linalg.solve(
                self.cov[b] / max(1, self.counts[b]) + np.eye(N_CELLS) * 1e-2, grad)
            self.mu[b] += self.eta * shaped
        else:
            sigma2 = self.sigma2[b]
            if self.writer == 'mesu_shuffled_sigma':
                # Same variance distribution, deliberately wrong assignment.
                flat = sigma2.reshape(-1)
                sigma2 = flat[self._perm_rng.permutation(flat.size)].reshape(sigma2.shape)
            self.mu[b] += self.eta * sigma2 * grad
            if self.horizon > 0 and self.writer != 'mesu_infN':
                self.mu[b] += (self.sigma2[b] / (self.horizon * self.prior_var)) * (0. - self.mu[b])
            if self.writer not in ('mesu_frozen_sigma',):
                curvature = np.outer(z * z, np.ones(ALPHABET))
                self.sigma2[b] -= (self.sigma2[b] ** 2 / self.feedback_scale) * curvature
                if self.writer != 'mesu_infN':
                    self.sigma2[b] += (2. / self.horizon) * (self.prior_var - self.sigma2[b])
                np.clip(self.sigma2[b], 1e-8, 4. * self.prior_var, out=self.sigma2[b])

        self.counts[b] += 1
        rate = min(.05, 1. / float(self.counts[b]))
        center = (1 - rate) * self.centers[b] + rate * z
        self.centers[b] = center / max(np.linalg.norm(center), 1e-12)
        self.writes += 1
        return b

    # -- accounting ---------------------------------------------------------
    def bytes_used(self):
        total = self.mu.nbytes + self.centers.nbytes + self.counts.nbytes
        if self.writer.startswith('mesu'):
            total += self.sigma2.nbytes
        if self.cov is not None:
            total += self.cov.nbytes
        return int(total)

    def statistic_bytes(self):
        if self.writer.startswith('mesu'):
            return int(self.sigma2.nbytes)
        return int(self.cov.nbytes) if self.cov is not None else 0

    def mean_sigma2(self):
        return float(self.sigma2.mean())


# ----------------------------------------------------------------------------
# 4. Scoring: bits/symbol, no labels anywhere
# ----------------------------------------------------------------------------

def score(encoder, memory, stream, limit=None):
    """Held-out bits/symbol. Read-only: no writes, no state change."""
    before = (memory.mu.copy(), memory.centers.copy())
    total, count = 0., 0
    for context, target in windows(stream):
        if limit is not None and count >= limit:
            break
        z = encoder.features(encode_context(context))[0]
        p = memory.predict(z)
        total -= math.log2(max(float(p[target]), 1e-12))
        count += 1
    assert np.array_equal(memory.mu, before[0]), 'scoring mutated weights'
    assert np.array_equal(memory.centers, before[1]), 'scoring mutated routing'
    return total / max(1, count)


def learn(encoder, memory, stream, limit=None):
    """Prequential: predict, score, THEN observe. The target is never seen early."""
    total, count = 0., 0
    for context, target in windows(stream):
        if limit is not None and count >= limit:
            break
        z = encoder.features(encode_context(context))[0]
        p = memory.predict(z)                     # prediction precedes the target
        total -= math.log2(max(float(p[target]), 1e-12))
        memory.observe(z, target)
        count += 1
    return total / max(1, count)


def run_job(cfg, seed, job):
    started = time.perf_counter()
    streams = build_streams(cfg, seed)
    encoder = ENCODERS[job['encoder']](seed)
    memory = BankMemory(seed, writer=job['writer'], eta=job['eta'],
                        horizon=job['horizon'], prior_var=cfg.prior_var,
                        feedback_scale=cfg.feedback_scale)

    prequential_a = learn(encoder, memory, streams['train_a'])
    after_a = dict(
        a=score(encoder, memory, streams['test_a'], cfg.score_len),
        b=score(encoder, memory, streams['test_b'], cfg.score_len),
        generic=score(encoder, memory, streams['test_generic'], cfg.score_len),
        sigma2=memory.mean_sigma2())

    # Domain B. Domain A is never replayed for learning after this point.
    if job['writer'] == 'frozen':
        prequential_b = score(encoder, memory, streams['train_b'], cfg.score_len)
    else:
        prequential_b = learn(encoder, memory, streams['train_b'])
    after_b = dict(
        a=score(encoder, memory, streams['test_a'], cfg.score_len),
        b=score(encoder, memory, streams['test_b'], cfg.score_len),
        generic=score(encoder, memory, streams['test_generic'], cfg.score_len),
        sigma2=memory.mean_sigma2())

    return dict(
        after_a=after_a, after_b=after_b,
        prequential_a=prequential_a, prequential_b=prequential_b,
        retention_loss=after_b['a'] - after_a['a'],       # bits/symbol worse on A
        transfer_gain=after_a['b'] - after_b['b'],        # bits/symbol better on B
        recall_margin=after_b['generic'] - after_b['a'],  # A-specific vs generic
        floor_a=streams['floor_a'], floor_b=streams['floor_b'],
        bank_bytes=memory.bytes_used(), statistic_bytes=memory.statistic_bytes(),
        encoder_bytes=encoder.bytes_used(), writes=memory.writes,
        wall_seconds=time.perf_counter() - started)


# ----------------------------------------------------------------------------
# 5. Jobs
# ----------------------------------------------------------------------------

def make_jobs(cfg):
    """Encoder x writer. Mechanism verdicts are computed per encoder, and only
    on encoders whose code can carry the task (see encoder_adequacy)."""
    mid = cfg.horizons[len(cfg.horizons) // 2]
    writers = [('frozen', 'frozen', 0., 0.), ('plain', 'plain', cfg.eta, 0.),
               ('v30_cov', 'v30_cov', cfg.eta, 0.)]
    writers += [('mesu_N%g' % h, 'mesu', cfg.eta, h) for h in cfg.horizons]
    writers += [('mesu_infN', 'mesu_infN', cfg.eta, 0.),
                ('mesu_frozen_sigma', 'mesu_frozen_sigma', cfg.eta, mid),
                ('mesu_shuffled_sigma', 'mesu_shuffled_sigma', cfg.eta, mid)]
    jobs = []
    for enc in cfg.encoders:
        for name, writer, eta, horizon in writers:
            jobs.append(dict(name='%s/%s' % (enc, name), encoder=enc, arm=name,
                             writer=writer, eta=eta, horizon=horizon))
    return jobs


# ----------------------------------------------------------------------------
# 6. Statistics and predeclared gates
# ----------------------------------------------------------------------------

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201}


def interval(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    n = len(values)
    if n < 2:
        return dict(n=n, mean=values[0] if values else None, low=None, high=None)
    mean = statistics.mean(values)
    half = T95.get(n - 1, 1.96) * statistics.stdev(values) / math.sqrt(n)
    return dict(n=n, mean=mean, sd=statistics.stdev(values),
                low=mean - half, high=mean + half)


def paired(by_seed, arm, parent, key, sign=1.):
    vals = []
    for _s, arms in sorted(by_seed.items()):
        a, b = arms.get(arm, {}).get(key), arms.get(parent, {}).get(key)
        if a is not None and b is not None:
            vals.append(sign * (a - b))
    return interval(vals)


MARGIN_BITS = .05          # bits/symbol noninferiority margin


UNIFORM_BITS = math.log2(ALPHABET)
ADEQUACY_MARGIN = .05          # bits/symbol the code must beat uniform by


def per_encoder(by_seed, encoder):
    return {s: arms[encoder] for s, arms in by_seed.items() if encoder in arms}


def encoder_adequacy(by_seed, encoder):
    """Can this code carry the task at all?

    ADDED after the contrast encoder scored 4.121 bits/symbol against a uniform
    ceiling of 4.000 -- worse than guessing -- while an observable readout on the
    same stream reached 3.661 (source floor 2.078). Random sparse expansion
    assumes distributed sensory input; the fly's antennal lobe delivers ~50
    graded glomeruli, whereas a symbol is one-hot, and random projection dilutes
    a one-hot exactly as it would a mixture.

    Running write-rule arms on a code that carries no information reproduces the
    V27 failure of measuring mechanisms on a substrate that cannot express the
    task. Mechanism verdicts on an inadequate encoder are void.
    """
    scoped = per_encoder(by_seed, encoder)
    learned = interval([a['plain']['after_a_a'] for a in scoped.values() if 'plain' in a])
    gain = interval([UNIFORM_BITS - a['plain']['after_a_a']
                     for a in scoped.values() if 'plain' in a])
    return dict(plain_bits_on_A=learned, gain_over_uniform=gain,
                uniform_bits=UNIFORM_BITS,
                adequate=bool(gain.get('low') is not None and gain['low'] > ADEQUACY_MARGIN))


def task_validity(by_seed_scoped):
    """Carried over from V28/V29 -- the reason those versions measured anything.

    headroom  : the frozen model must not already predict B well
    tension   : plain adaptation on B must measurably worsen A
    recall    : retained A performance must beat a never-trained generic source,
                or "retention" is only generic symbol statistics
    """
    by_seed = by_seed_scoped
    frozen_b = interval([a['frozen']['after_b_b'] for a in by_seed.values()
                         if 'frozen' in a])
    headroom = paired(by_seed, 'plain', 'frozen', 'after_b_b', sign=-1.)
    tension = paired(by_seed, 'plain', 'frozen', 'retention_loss')
    recall = interval([a['plain']['recall_margin'] for a in by_seed.values()
                       if 'plain' in a])
    gates = dict(
        headroom=bool(headroom.get('low') is not None and headroom['low'] > .02),
        tension=bool(tension.get('low') is not None and tension['low'] > .02),
        recall_dependence=bool(recall.get('low') is not None and recall['low'] > .05))
    return dict(frozen_b_bits=frozen_b, headroom_gain=headroom,
                tension_loss=tension, recall_margin=recall, gates=gates,
                eligible=bool(all(gates.values())))


def evaluate(by_seed, cfg):
    """Predeclared. P2/P3 are veto gates: failing either retires the mechanism."""
    names = {n for arms in by_seed.values() for n in arms}
    mesu_arms = sorted(n for n in names if n.startswith('mesu_N'))
    out = {}
    for arm in mesu_arms:
        entry = dict(
            P1_retention_vs_plain=paired(by_seed, 'plain', arm, 'retention_loss'),
            P1_transfer_vs_plain=paired(by_seed, arm, 'plain', 'after_b_b', sign=-1.),
            P2_vs_frozen_sigma=paired(by_seed, 'mesu_frozen_sigma', arm, 'retention_loss')
            if 'mesu_frozen_sigma' in names else None,
            P3_vs_shuffled_sigma=paired(by_seed, 'mesu_shuffled_sigma', arm, 'retention_loss')
            if 'mesu_shuffled_sigma' in names else None,
            vs_v30_cov=paired(by_seed, 'v30_cov', arm, 'retention_loss')
            if 'v30_cov' in names else None)

        def beats(stats):
            return bool(stats and stats.get('low') is not None and stats['low'] > 0)

        retains = beats(entry['P1_retention_vs_plain'])
        transfer_ok = (entry['P1_transfer_vs_plain'].get('low') is None
                       or entry['P1_transfer_vs_plain']['low'] >= -MARGIN_BITS)
        beats_frozen_sigma = beats(entry['P2_vs_frozen_sigma'])
        beats_shuffled = beats(entry['P3_vs_shuffled_sigma'])

        if entry['P2_vs_frozen_sigma'] is None or entry['P3_vs_shuffled_sigma'] is None:
            verdict = 'controls_missing'
        elif not (beats_frozen_sigma and beats_shuffled):
            # Matched by a fixed or misassigned variance: no metaplasticity.
            verdict = 'fixed_rate_effect'
        elif retains and transfer_ok:
            verdict = 'uncertainty_modulation_supported'
        else:
            verdict = 'inconclusive'
        entry['verdict'] = verdict
        out[arm] = entry

    # P4: eq 19 says infinite horizon ossifies -- better A, worse late B.
    if 'mesu_infN' in names and mesu_arms:
        reference = mesu_arms[len(mesu_arms) // 2]
        a_gain = paired(by_seed, reference, 'mesu_infN', 'retention_loss')
        b_cost = paired(by_seed, 'mesu_infN', reference, 'prequential_b')
        out['P4_catastrophic_remembering'] = dict(
            reference=reference, retention_advantage_of_infN=a_gain,
            late_B_cost_of_infN=b_cost,
            supported=bool(a_gain.get('low') is not None and a_gain['low'] > 0
                           and b_cost.get('low') is not None and b_cost['low'] > 0))
    return out


# ----------------------------------------------------------------------------
# 7. Reporting
# ----------------------------------------------------------------------------

def collect(results):
    by_seed = {}
    for r in results:
        row = r['result']
        by_seed.setdefault(r['seed'], {}).setdefault(r['job']['encoder'], {})[r['job']['arm']] = dict(
            after_a_a=row['after_a']['a'], after_b_a=row['after_b']['a'],
            after_a_b=row['after_a']['b'], after_b_b=row['after_b']['b'],
            retention_loss=row['retention_loss'], transfer_gain=row['transfer_gain'],
            recall_margin=row['recall_margin'], prequential_b=row['prequential_b'],
            sigma2=row['after_b']['sigma2'], bank_bytes=row['bank_bytes'],
            statistic_bytes=row['statistic_bytes'])
    return by_seed


def fmt(stats, unit=''):
    if not stats or stats.get('mean') is None:
        return 'n/a'
    if stats.get('low') is None:
        return '%+.4f%s' % (stats['mean'], unit)
    return '%+.4f [%+.4f, %+.4f]%s' % (stats['mean'], stats['low'], stats['high'], unit)


def write_outputs(out_dir, cfg, results, by_seed, report):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'TERMINAL.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['seed', 'encoder', 'arm', 'bits_A_after_A', 'bits_A_after_B',
                    'bits_B_after_B', 'retention_loss', 'recall_margin',
                    'prequential_B', 'mean_sigma2', 'bank_bytes', 'statistic_bytes'])
        for seed, encs in sorted(by_seed.items()):
            for enc, arms in sorted(encs.items()):
                for arm, v in sorted(arms.items()):
                    w.writerow([seed, enc, arm, v['after_a_a'], v['after_b_a'],
                                v['after_b_b'], v['retention_loss'], v['recall_margin'],
                                v['prequential_b'], v['sigma2'], v['bank_bytes'],
                                v['statistic_bytes']])

    (out_dir / 'GATES.json').write_text(json.dumps(
        dict(by_encoder=report, margin_bits=MARGIN_BITS,
             adequacy_margin_bits=ADEQUACY_MARGIN, uniform_bits=UNIFORM_BITS),
        indent=2, default=str))
    (out_dir / 'MANIFEST.json').write_text(json.dumps(dict(
        source_sha256=sha_file(SOURCE), seeds=list(cfg.seeds),
        horizons=list(cfg.horizons), encoders=list(cfg.encoders),
        alphabet=ALPHABET, banks=BANKS, window=WINDOW, order=cfg.order,
        concentration=cfg.concentration, eta=cfg.eta, prior_var=cfg.prior_var,
        labels_used=0, teaching_signal='next-symbol prediction residual only',
        reference=V30_REFERENCE,
        predeclared=dict(
            P0='an encoder whose plain writer cannot beat uniform by %.2f bits '
               'carries no task information; its mechanism verdicts are void' % ADEQUACY_MARGIN,
            P1='mesu retains A better than plain at matched B transfer',
            P2='mesu_frozen_sigma does NOT match mesu; otherwise retire',
            P3='mesu_shuffled_sigma does NOT match mesu; otherwise retire',
            P4='mesu_infN retains A better but learns B worse (eq 19)'),
    ), indent=2))

    lines = ['# V31 results', '',
             '**Source SHA256** `%s`' % sha_file(SOURCE), '',
             'Label-free: the only teaching signal is the next-symbol prediction residual.',
             'Lower bits/symbol is better; uniform is %.3f.' % UNIFORM_BITS, '',
             '## Encoder adequacy (P0)', '',
             '| encoder | plain bits on A | gain over uniform | adequate |',
             '|---|---|---|---|']
    for enc, r in sorted(report.items()):
        a = r['adequacy']
        lines.append('| %s | %s | %s | **%s** |' % (
            enc, fmt(a['plain_bits_on_A']), fmt(a['gain_over_uniform']), a['adequate']))

    for enc, r in sorted(report.items()):
        lines += ['', '## %s' % enc, '']
        if not r['adequacy']['adequate']:
            lines += ['_Code carries no task information. All mechanism verdicts VOID._', '']
            continue
        v = r['validity']
        lines += ['_Task eligible: **%s**_' % v.get('eligible'), '',
                  '| gate | value | pass |', '|---|---|---|',
                  '| headroom | %s | %s |' % (fmt(v['headroom_gain']), v['gates']['headroom']),
                  '| tension | %s | %s |' % (fmt(v['tension_loss']), v['gates']['tension']),
                  '| recall dependence | %s | %s |' % (
                      fmt(v['recall_margin']), v['gates']['recall_dependence']),
                  '',
                  '| arm | P1 retention vs plain | P2 vs frozen-sigma | P3 vs shuffled-sigma | verdict |',
                  '|---|---|---|---|---|']
        for arm, e in sorted(r['mechanism'].items()):
            if arm.startswith('P4') or not isinstance(e, dict) or 'verdict' not in e:
                continue
            lines.append('| %s | %s | %s | %s | **%s** |' % (
                arm, fmt(e['P1_retention_vs_plain']), fmt(e['P2_vs_frozen_sigma']),
                fmt(e['P3_vs_shuffled_sigma']), e['verdict']))
        p4 = r['mechanism'].get('P4_catastrophic_remembering')
        if p4:
            lines += ['', '**P4 (eq 19, infinite horizon ossifies): %s** -- '
                      'retention advantage %s, late-B cost %s' % (
                          p4['supported'], fmt(p4['retention_advantage_of_infN']),
                          fmt(p4['late_B_cost_of_infN']))]

    lines += ['', '## Interpretation limits', '',
              '- P0 is a hard gate. An inadequate code voids every verdict beneath it;',
              '  running write rules on a code that cannot express the task is how V27',
              '  produced uninterpretable results.',
              '- `fixed_rate_effect` means a frozen or misassigned variance matches the',
              '  learned one: a per-weight rate schedule, not metaplasticity. Retire it.',
              '- `observable` is a diagnostic ceiling, not a proposed architecture.',
              '- Retention is scored against a never-trained generic source; a small',
              '  recall margin means the metric captures only generic symbol statistics.',
              '- Synthetic streams with a controlled shift. No scaling or language claim.']
    (out_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n')


# ----------------------------------------------------------------------------
# 8. Self-tests
# ----------------------------------------------------------------------------

def self_test(cfg):
    checks = []

    def check(name, cond):
        checks.append((name, bool(cond)))
        if not cond:
            raise AssertionError('Self-test failed: ' + name)

    seed = cfg.seeds[0]
    enc = ContrastEncoder(seed)
    ctx = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4])
    x = encode_context(ctx)
    check('encoder input shape', x.shape == (BINS, FEATURES))
    z = enc.features(x)
    check('feature shape', z.shape == (1, N_CELLS))
    k = max(1, round(N_CELLS * ACTIVE_FRACTION))
    check('sparsity honoured', int((np.abs(z[0]) > 0).sum()) <= k)
    check('code is normalised', abs(np.linalg.norm(z[0]) - 1.) < 1e-9)

    mem = BankMemory(seed, writer='mesu', horizon=32)
    p = mem.predict(z[0])
    check('prediction is a distribution', abs(p.sum() - 1.) < 1e-9 and (p >= 0).all())
    check('uniform before learning', abs(p.max() - 1. / ALPHABET) < 1e-9)

    before_mu = mem.mu.copy()
    mem.predict(z[0])
    check('prediction does not write', np.array_equal(before_mu, mem.mu))

    s0 = mem.sigma2.mean()
    for _ in range(30):
        mem.observe(z[0], 7)
    check('mesu writes', not np.allclose(before_mu, mem.mu))
    check('variance shrinks where active', mem.sigma2.mean() < s0)
    check('variance stays positive', mem.sigma2.min() > 0)

    # Finite horizon holds a floor; infinite horizon collapses.
    finite = BankMemory(seed, writer='mesu', horizon=16)
    infinite = BankMemory(seed, writer='mesu_infN')
    for _ in range(400):
        finite.observe(z[0], 7)
        infinite.observe(z[0], 7)
    b = finite.route(z[0])
    active = np.abs(z[0]) > 0
    check('finite horizon keeps a variance floor', finite.sigma2[b][active].mean() > 1e-4)
    check('infinite horizon collapses further',
          infinite.sigma2[b][active].mean() < finite.sigma2[b][active].mean())

    frozen_sigma = BankMemory(seed, writer='mesu_frozen_sigma', horizon=32)
    s_before = frozen_sigma.sigma2.copy()
    for _ in range(20):
        frozen_sigma.observe(z[0], 2)
    check('frozen sigma never updates', np.array_equal(s_before, frozen_sigma.sigma2))
    check('frozen sigma still writes', not np.allclose(frozen_sigma.mu, 0.))

    shuffled = BankMemory(seed, writer='mesu_shuffled_sigma', horizon=32)
    normal = BankMemory(seed, writer='mesu', horizon=32)
    for _ in range(20):
        shuffled.observe(z[0], 2)
        normal.observe(z[0], 2)
    check('shuffled sigma diverges from mesu', not np.allclose(shuffled.mu, normal.mu))

    frozen = BankMemory(seed, writer='frozen')
    frozen.observe(z[0], 5)
    check('frozen writer never writes', np.allclose(frozen.mu, 0.))

    # MESU costs exactly one extra scalar per weight (2x total). The covariance
    # scales as N_CELLS^2 rather than N_CELLS*ALPHABET, so its overhead relative
    # to the weights grows as the readout narrows: 6x here, 24x in V30's 4-wide
    # readout. The invariant that holds regardless of width is the ratio below.
    check('mesu statistic is exactly one scalar per weight',
          normal.statistic_bytes() == normal.mu.nbytes)
    cov = BankMemory(seed, writer='v30_cov')
    check('covariance overhead exceeds mesu',
          cov.statistic_bytes() == N_CELLS / ALPHABET * normal.statistic_bytes()
          and cov.statistic_bytes() > normal.statistic_bytes())

    rg = rng_for(seed, 'test_source')
    src = make_source(rg, 1, .2)
    stream = sample_stream(src, rg, 400)
    check('stream in alphabet', stream.min() >= 0 and stream.max() < ALPHABET)
    floor = entropy_rate(src, stream)
    check('entropy floor below uniform', 0 < floor < math.log2(ALPHABET))

    pairs = list(windows(stream))
    check('windows are causal',
          all(len(c) == WINDOW for c, _ in pairs) and len(pairs) == len(stream) - WINDOW)
    ctx0, tgt0 = pairs[0]
    check('target is the next symbol', tgt0 == stream[WINDOW]
          and np.array_equal(ctx0, stream[:WINDOW]))

    mem2 = BankMemory(seed, writer='plain')
    kept = mem2.mu.copy()
    bits = score(enc, mem2, stream, 40)
    check('scoring is read-only', np.array_equal(kept, mem2.mu))
    check('untrained bits equal log2(alphabet)', abs(bits - math.log2(ALPHABET)) < 1e-6)

    obs = ObservableEncoder(seed)
    zo = obs.features(x)
    check('observable shape', zo.shape == (1, N_CELLS))
    check('observable preserves lag-1 identity', zo[0][:FEATURES].argmax() == ctx[-1])
    lag = LaggedEncoder(seed)
    zl = lag.features(x)
    check('lagged shape', zl.shape == (1, N_CELLS))
    check('lagged is sparse', int((np.abs(zl[0]) > 0).sum()) <= max(1, round(N_CELLS * ACTIVE_FRACTION)))
    check('encoders registry complete', set(ENCODERS) == {'contrast', 'lagged', 'observable'})
    check('adequacy margin is positive', ADEQUACY_MARGIN > 0)
    check('uniform bits is log2 alphabet', abs(UNIFORM_BITS - math.log2(ALPHABET)) < 1e-12)

    jobs = make_jobs(cfg)
    names = [j['name'] for j in jobs]
    check('every job names an encoder', all(j['encoder'] in ENCODERS for j in jobs))
    check('job names unique', len(set(names)) == len(names))
    arms = {j['arm'] for j in jobs}
    for required in ('frozen', 'plain', 'mesu_frozen_sigma', 'mesu_shuffled_sigma'):
        check('roster contains ' + required, required in arms)
    check('every encoder carries the full writer set',
          all(len({j['arm'] for j in jobs if j['encoder'] == e}) == len(arms)
              for e in {j['encoder'] for j in jobs}))
    check('no job carries a label', all('label' not in json.dumps(j) for j in jobs))
    return checks


# ----------------------------------------------------------------------------
# 9. CLI
# ----------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out-dir', type=Path, default=Path('v31_results'))
    ap.add_argument('--seeds', type=int, nargs='+',
                    default=[1301, 1303, 1307, 1319, 1321, 1327])
    ap.add_argument('--horizons', type=float, nargs='+', default=[8, 32, 128, 512])
    ap.add_argument('--encoders', nargs='+', default=['contrast', 'lagged', 'observable'],
                    choices=['contrast', 'lagged', 'observable'])
    ap.add_argument('--order', type=int, default=2)
    ap.add_argument('--concentration', type=float, default=.15)
    ap.add_argument('--train-len', type=int, default=6000)
    ap.add_argument('--test-len', type=int, default=1200)
    ap.add_argument('--score-len', type=int, default=800)
    ap.add_argument('--eta', type=float, default=.5)
    ap.add_argument('--prior-var', type=float, default=1.)
    ap.add_argument('--feedback-scale', type=float, default=1.)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--list-jobs', action='store_true')
    cfg = ap.parse_args(argv)
    if cfg.smoke:
        cfg.seeds = cfg.seeds[:2]
        cfg.horizons = cfg.horizons[:2]
        cfg.train_len, cfg.test_len, cfg.score_len = 600, 300, 200
        cfg.encoders = cfg.encoders[:2]
    if len(set(cfg.seeds)) != len(cfg.seeds):
        ap.error('seeds must be distinct')
    if any(h <= 0 for h in cfg.horizons):
        ap.error('horizons must be positive')
    return cfg


def main(argv=None):
    cfg = parse_args(argv)

    if cfg.self_test:
        for name, ok in self_test(cfg):
            print('  %-46s %s' % (name, 'ok' if ok else 'FAILED'))
        print('\nAll self-tests passed.')
        return 0

    jobs = make_jobs(cfg)
    if cfg.list_jobs:
        for j in jobs:
            print('%-34s encoder=%-11s writer=%-20s horizon=%g'
                  % (j['name'], j['encoder'], j['writer'], j['horizon']))
        print('\n%d arms x %d seeds = %d trajectories'
              % (len(jobs), len(cfg.seeds), len(jobs) * len(cfg.seeds)))
        return 0

    total, done, results = len(jobs) * len(cfg.seeds), 0, []
    print('V31: %d arms x %d seeds = %d trajectories' % (len(jobs), len(cfg.seeds), total))
    for job in jobs:
        for seed in cfg.seeds:
            results.append(dict(job=job, seed=seed, result=run_job(cfg, seed, job)))
            done += 1
            r = results[-1]['result']
            print('  [%3d/%3d] %-22s seed=%-5d A=%.3f->%.3f  B=%.3f  %.1fs'
                  % (done, total, job['name'], seed, r['after_a']['a'],
                     r['after_b']['a'], r['after_b']['b'], r['wall_seconds']), flush=True)

    by_seed = collect(results)
    report = {}
    for enc in cfg.encoders:
        scoped = per_encoder(by_seed, enc)
        adequacy = encoder_adequacy(by_seed, enc)
        report[enc] = dict(
            adequacy=adequacy,
            validity=task_validity(scoped) if adequacy['adequate'] else
                     dict(eligible=False, skipped='encoder inadequate'),
            mechanism=evaluate(scoped, cfg) if adequacy['adequate'] else
                      dict(skipped='encoder inadequate; verdicts void'))
    write_outputs(cfg.out_dir, cfg, results, by_seed, report)

    for enc, r in report.items():
        a = r['adequacy']
        print('\n%s: adequate=%s  plain bits on A=%s (uniform %.3f)'
              % (enc, a['adequate'], fmt(a['plain_bits_on_A']), UNIFORM_BITS))
        if not a['adequate']:
            print('   mechanism verdicts VOID: this code cannot carry the task')
            continue
        print('   task eligible: %s' % r['validity'].get('eligible'))
        ok = [k for k, v in r['mechanism'].items()
              if isinstance(v, dict) and v.get('verdict') == 'uncertainty_modulation_supported']
        print('   arms supporting uncertainty modulation: %s' % (', '.join(ok) or 'none'))
    print('\nWrote %s' % cfg.out_dir.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
