#!/usr/bin/env python3
"""V30: controlled directional bank writes, then conditional practical confirmation.

Run: python3 elm_fly_directional_memory_v30.py --install-deps
Same command resumes. Automatic macOS caffeinate. One compact results ZIP.
160 diagnostic trajectories;256 maximum with fresh practical confirmation.
--smoke:64 tiny engineering jobs covering BOTH writer paths;no scientific verdict.
--self-test checks equations and reporting;--list-jobs displays the fixed plan.

Preserves V29 temporal encoder and nuisance/overlap tasks. Compares ordinary,
diagonal and full co-activation writes at equal immediate correction, with
misassigned-statistic and own-state displacement controls. Mandatory revision.
Full covariance adds576KiB;diagonal adds6KiB. No lifetime qualification claim.
Requires NumPy/SciPy;no earlier files, datasets, Transformer or replay buffer.
"""
from __future__ import annotations
import argparse
import atexit
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import signal
import shlex
import subprocess
import sys
import tempfile
import time
import zipfile

for _thread_var in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS',
                   'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_thread_var]='1'  # tiny matrices: avoid BLAS oversubscription
np=None
student_t=None

VERSION="30.0.0"
TASKS=("nuisance","overlap")
VIEWS=("hard","blended")
_v29_math=math

def dependencies(install=False):
    global np, student_t
    try:
        import numpy as np
        from scipy.stats import t as student_t
    except ImportError:
        if not install:
            raise RuntimeError('Run again with --install-deps (requires numpy and scipy).')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy', 'scipy'])
        import numpy as np
        from scipy.stats import t as student_t


def rng(seed, name):
    n = int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:4], 'little')
    return np.random.default_rng(n)


def norm_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [clean(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, np.generic):
        return clean(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        raise FloatingPointError('Nonfinite result; refusing to save scientific output')
    return obj


def serialized(obj):
    return json.dumps(clean(obj), sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(obj):
    return hashlib.sha256(serialized(obj).encode()).hexdigest()


def atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, obj):
    atomic(path, json.dumps(clean(obj), indent=2, sort_keys=True, allow_nan=False)+'\n')


class Encoder:
    """Only x reaches plasticity. All probe operations are pure.

    Selection RNG and residence clocks belong to this learner, not the evaluator.
    Original weights/diagnostic snapshots are NOT retained in this class.
    """
    def __init__(self, seed, spec, calibration):
        self.spec = dict(spec)
        rr = rng(seed, 'v25_minimal_sparse_afferents')
        self.contacts = np.stack([rr.choice(32, 8, replace=False) for _ in range(96)])
        self.w = rr.normal(0, 1/math.sqrt(8), (96, 8))
        if spec.get('unit', False):
            self.w = norm_rows(self.w)
        self.radii = np.linalg.norm(self.w, axis=1)
        self.alpha = -np.expm1(-1/np.resize(np.array([1., 2., 4.]), 96))
        self.slow_alpha = -np.expm1(-1/(4*np.resize(np.array([1., 2., 4.]), 96)))
        self.stable = np.zeros(96, bool)
        order = rng(seed, 'v28_mask_partition')
        self.groups = []
        for start in (0, 48):
            ids = start+order.permutation(48)
            n = round(48*spec.get('stable', 0.))
            self.stable[ids[:n]] = True
            self.groups.append(ids[n:])
        self.fixed = np.zeros(96, bool)
        for ids in self.groups:
            self.fixed[ids[:max(1, round(len(ids)*spec.get('fraction', .25)))]] = True
        self.select_rng = rng(seed, 'v28_mask_rng')
        self.resident = self.fixed.copy()
        self.experience = 0
        self.tick = 0
        self.counts = np.zeros(96, np.int64)
        self.last_changed = np.full(96, -1, np.int64)
        self.cell_path = np.zeros(96)
        self.budget = np.zeros(5)  # raw proposal, submitted, actual, max match error, calls
        a = self.raw(calibration)
        self.mean = a.mean(0)
        self.sd = np.maximum(a.std(0), .01)

    def raw(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 2:
            x = x[None]
        if x.ndim != 3 or x.shape[-1] != 32 or not np.isfinite(x).all():
            raise ValueError('Encoder input is a finite (batch,time,32) observation')
        fast = np.zeros((len(x), 96))
        slow = np.zeros_like(fast)
        for t in range(x.shape[1]):
            drive = np.tanh((x[:, t][:, self.contacts]*self.w).sum(-1))
            fast += self.alpha*(drive-fast)
            slow += self.slow_alpha*(drive-slow)
        return {'F': fast, 'D': fast-slow, 'R': 2*fast-slow}[self.spec['formula']]

    def features(self, x, address=False):
        z = np.clip((self.raw(x)-self.mean)/self.sd, -5, 5)
        if address and self.spec.get('path') == 'scaffold':
            # Stable outputs before top-k: plastic competitors cannot change keys.
            return norm_rows(z*self.stable)
        pieces = []
        for start in (0, 48):
            a = z[:, start:start+48]
            ids = np.argpartition(np.abs(a), -12, axis=1)[:, -12:]
            h = np.zeros_like(a)
            np.put_along_axis(h, ids, np.take_along_axis(a, ids, 1), 1)
            pieces.append(norm_rows(h)/math.sqrt(2))
        return np.concatenate(pieces, axis=1)

    def mask(self, response):
        rule = self.spec['rule']
        out = np.zeros(96, bool)
        if rule == 'dense':
            return ~self.stable
        if rule == 'fixed':
            return self.fixed.copy()
        if rule == 'sticky':
            if self.experience % 8 == 0 and self.tick == 0:
                self.resident[:] = False
                for ids in self.groups:
                    k = max(1, round(len(ids)*self.spec['fraction']))
                    self.resident[self.select_rng.choice(ids, k, replace=False)] = True
            return self.resident.copy()
        for ids in self.groups:
            k = max(1, round(len(ids)*self.spec['fraction']))
            if rule == 'random':
                chosen = self.select_rng.choice(ids, k, replace=False)
            elif rule == 'cyclic':
                # Fixed, evenly spaced group schedule in each eligible half.
                start = ((self.experience//8)*k) % len(ids)
                chosen = ids[(start+np.arange(k)) % len(ids)]
            elif rule in ('top', 'bottom'):
                order = np.argsort(np.abs(response[ids]), kind='stable')
                chosen = ids[order[-k:] if rule == 'top' else order[:k]]
            else:
                raise ValueError('Unknown plasticity rule '+rule)
            out[chosen] = True
        return out

    def update(self, x):
        if self.spec['rule'] == 'frozen' or self.spec['rate'] == 0:
            self.experience += 1
            return
        for tick, row in enumerate(np.asarray(x)):
            self.tick = tick
            u = norm_rows(row[self.contacts])
            unit = self.w/self.radii[:, None]
            y = (unit*u).sum(1)
            p = y[:, None]*(u-y[:, None]*unit)
            p[self.stable] = 0
            mask = self.mask(y)
            sparse = p*mask[:, None]
            desired = np.linalg.norm(sparse)
            if self.spec.get('matched', False):
                total = np.linalg.norm(p)
                proposal = p*(desired/total) if total > 0 else np.zeros_like(p)
            else:
                proposal = sparse
            delta = .001*self.spec['rate']*proposal
            before = self.w.copy()
            # Unmodified rows remain bitwise untouched, avoiding rounding drift.
            changed = np.any(delta != 0, axis=1)
            self.w[changed] = norm_rows(self.w[changed]+delta[changed])*self.radii[changed, None]
            mag = np.linalg.norm(self.w-before, axis=1)
            self.counts += mag > 0
            self.last_changed[mag > 0] = self.experience
            self.cell_path += mag
            self.budget[:3] += (np.linalg.norm(p), np.linalg.norm(delta), np.linalg.norm(mag))
            if self.spec.get('matched', False):
                self.budget[3] = max(self.budget[3], abs(np.linalg.norm(delta)-.001*self.spec['rate']*desired))
            self.budget[4] += 1
        self.experience += 1

    def digest(self):
        h = hashlib.sha256()
        for k, v in sorted(vars(self).items()):
            h.update(k.encode())
            if isinstance(v, np.ndarray):
                h.update(v.tobytes())
            elif k == 'select_rng':
                h.update(serialized(v.bit_generator.state).encode())
            elif k == 'groups':
                for a in v:
                    h.update(a.tobytes())
            else:
                h.update(serialized(v).encode())
        return h.hexdigest()

    def costs(self):
        arrays = {k: int(v.nbytes) for k, v in vars(self).items() if isinstance(v, np.ndarray)}
        arrays['groups'] = sum(x.nbytes for x in self.groups)
        arrays['rng_numeric_payload'] = 48  # PCG64 state/inc (128 bits each), cached uint32 flag/value
        arrays['clock_scalars'] = 16
        return {'array_bytes': arrays, 'total_numeric_bytes': sum(arrays.values()),
                'plastic_sensory_weight_bytes': int(self.w.nbytes),
                'mask_residence_bytes': int(self.fixed.nbytes+self.resident.nbytes+self.stable.nbytes)
                    + arrays['groups']+arrays['rng_numeric_payload']+16,
                'note': 'Numeric payload only; includes diagnostic counters, excludes Python overhead/spec constants'}

    def audit(self, initial):
        displacement = np.linalg.norm(self.w-initial, axis=1)
        return {'cumulative_actual_frobenius': self.budget[2], 'net_displacement': np.linalg.norm(displacement),
                'cumulative_submitted_frobenius': self.budget[1], 'max_proposal_match_error': self.budget[3],
                'update_calls': int(self.budget[4]), 'cell_update_events': int(self.counts.sum()),
                'ever_modified': float(np.mean(self.counts > 0)),
                'modified_last_16': float(np.mean(self.last_changed >= max(0, self.experience-16))),
                'per_cell_displacement': displacement, 'per_cell_path_length': self.cell_path,
                'original_norm_max_error': float(np.max(np.abs(np.linalg.norm(self.w, axis=1)-self.radii)))}


class Bundle:
    def __init__(self, seed, spec, calibration):
        self.spec = dict(spec)
        self.value = Encoder(seed, spec, calibration)
        if spec['path'] in ('FF', 'FL', 'LL'):
            key = dict(spec)
            if spec['path'] in ('FF', 'FL'):
                key.update(rule='frozen', rate=0.)
            self.address = Encoder(seed, key, calibration)
        else:
            self.address = self.value
        if spec['path'] == 'FF':
            self.value.spec.update(rule='frozen', rate=0.)

    def features(self, x):
        z = self.value.features(x)
        r = z if self.address is self.value and self.spec['path'] != 'scaffold' else self.address.features(x, True)
        return r, z

    def update(self, x):
        self.value.update(x)
        if self.address is not self.value:
            self.address.update(x)

    def digest(self):
        return self.address.digest()+self.value.digest()


class Banks:
    def __init__(self, seed):
        self.centers = norm_rows(rng(seed, 'v28_banks').normal(size=(8, 96)))
        self.w = np.zeros((8, 96, 4))
        self.counts = np.zeros(8, np.int64)

    def routes(self, r):
        return np.argmax(np.asarray(r)@self.centers.T, axis=-1)

    def scores(self, r, z, view):
        r, z = np.atleast_2d(r), np.atleast_2d(z)
        local = np.stack([z@w for w in self.w], axis=1)
        if view == 'hard':
            return local[np.arange(len(z)), self.routes(r)]
        score = 8*(norm_rows(r)@self.centers.T)
        p = np.exp(score-score.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        return np.einsum('nb,nbc->nc', p, local)

    def learn(self, r, z, label):
        b = int(self.routes(r))
        target = np.eye(4)[int(label)]
        error = target-z@self.w[b]
        self.w[b] += .6*np.outer(z, error)/max(float(z@z), 1e-12)
        self.counts[b] += 1
        rate = min(.05, 1/float(self.counts[b]))
        self.centers[b] = norm_rows((1-rate)*self.centers[b]+rate*norm_rows(r))
        return b

    def digest(self):
        return hashlib.sha256(self.w.tobytes()+self.centers.tobytes()+self.counts.tobytes()).hexdigest()


def dimensions(smoke):
    return dict(group=4, reverse=2, new=12, calibration=8, probes=16, passes=1) if smoke else \
        dict(group=24, reverse=12, new=192, calibration=96, probes=192, passes=3)


def render(bases, variants, rr, noise=.08, horizon=12):
    x = np.zeros((len(bases), variants, horizon, 32))
    a = bases[:, None, 0]+rr.normal(0, noise, (len(bases), variants, 32))
    b = bases[:, None, 1]+rr.normal(0, noise, (len(bases), variants, 32))
    x[:, :, :3] = a[:, :, None]
    x[:, :, horizon-7:horizon-4] = b[:, :, None]
    return x


def spec(rule='frozen', rate=0., fraction=.25, **kw):
    s = dict(formula='D', rule=rule, rate=float(rate), fraction=float(fraction),
             path='one', stable=0., matched=False, unit=False)
    s.update(kw)
    return s


def measure(bank, r, z, labels, view):
    return float(np.mean(np.argmax(bank.scores(r, z, view), axis=1) == labels))


class RunLock:
    def __init__(self, out):
        self.path = Path(out).resolve()/'.run.lock'
        self.file = None
    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a+')
        try:
            import fcntl
            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.close()
            raise RuntimeError('Another process is already running this output directory')
        except ImportError:
            self.file.close()
            raise RuntimeError('This runner supports macOS/Linux process locking')
        return self
    def __exit__(self, *args):
        self.file.close()


def keep_awake():
    if sys.platform != 'darwin':
        return None
    try:
        child = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print('macOS caffeinate active for this run.', flush=True)
        atexit.register(lambda: child.terminate() if child.poll() is None else None)
        return child
    except OSError as e:
        print('Could not start caffeinate: '+str(e), file=sys.stderr)
        return None


def make_stream_v29(seed, task, smoke=False, lifetime=False, horizon=12):
    """Environment/evaluator only. Never passed to a learner object."""
    if task not in ('nuisance', 'overlap'):
        raise ValueError('V29 task must be nuisance or overlap')
    d = dimensions(smoke)
    nA, nB = 3*d['group'], d['new']
    structured = True
    domains = 3 if lifetime else 1
    rr = rng(seed, 'v28_environment_transform')
    if task == 'rotated':
        q, r = np.linalg.qr(rr.normal(size=(32, 32)))
        transform = q*np.where(np.diag(r) >= 0, 1., -1.)[None]
    else:
        transform = np.eye(32)[:, rr.permutation(32)]

    def bases(n, domain, key, labels=None):
        rg = rng(seed, key)
        x = rg.normal(0, .12, (n, 2, 32))
        start = domain*(4 if task == 'overlap' else 8)
        x[:, :, start:start+8] = rg.normal(0, .65, (n, 2, 8))
        if labels is not None:
            bits = np.stack((labels//2, labels % 2), 1)
            if task == 'equal':
                a = rg.uniform(.75, 1.15, (n, 2, 8))
                x[:, :, start:start+8] = (2*rg.integers(0, 2, a.shape)-1)*a
                x[:, :, start] = (2*bits-1)*a[:, :, 0]
            else:
                if task in ('nuisance', 'overlap'):
                    x[:, :, start+1:start+8] *= 2
                x[:, :, start] = (2*bits-1)*rg.uniform(.75, 1.15, (n, 2))
        return x

    count = nA+nB*domains if structured else nA+nB
    labels = rng(seed, 'arbitrary_labels').integers(0, 4, count)
    old = bases(nA, 0, 'A_latents')
    all_bases = [old]
    independent = {}
    if structured:
        for domain in range(1, domains+1):
            ids = slice(nA+(domain-1)*nB, nA+domain*nB)
            # Balanced labels, randomly ordered; latent RNGs separate from labels.
            labels[ids] = rng(seed, f'labels_{domain}').permutation(np.tile(np.arange(4), math.ceil(nB/4))[:nB])
            all_bases.append(bases(nB, domain, f'train_{domain}', labels[ids]))
            y = np.tile(np.arange(4), math.ceil(d['probes']/4))[:d['probes']]
            latent = bases(len(y), domain, f'unseen_{domain}', y)
            independent[chr(65+domain)+'_unseen'] = (
                render(latent, 1, rng(seed, f'unseen_noise_{domain}'))[:, 0]@transform, y)
    else:
        all_bases.append(bases(nB, 0, 'association_extra'))
    latent = np.concatenate(all_bases)
    train = render(latent, 2, rng(seed, 'train_noise'))@transform
    probe = render(latent, 2, rng(seed, 'probe_noise'))@transform
    calibration = render(bases(d['calibration'], 0, 'calibration'), 1,
        rng(seed, 'calibration_noise'))[:, 0]@transform
    current = labels.copy()
    events, checkpoints, shifts = [], {}, []
    def phase(name, ids, domain=0, revision=False):
        ids = np.asarray(ids, dtype=int)
        if revision:
            current[ids] = (current[ids]+1) % 4
        order = rng(seed, 'order_'+name)
        for _ in range(d['passes']):
            pairs = [(int(i), v) for i in ids for v in (0, 1)]
            for j in order.permutation(len(pairs)):
                i, v = pairs[j]
                events.append((i, v, int(current[i]), domain))
        checkpoints[len(events)] = name
    rev = np.arange(d['reverse'])
    phase('acquire_A', np.arange(d['group']))
    phase('interfere_A2', np.arange(d['group'], 2*d['group']))
    phase('revise_A', rev, revision=True)
    phase('interfere_A3', np.arange(2*d['group'], nA))
    phase('revise_A_again', rev, revision=True)
    shifts.append(len(events))
    if structured:
        for domain in range(1, domains+1):
            ids = np.arange(nA+(domain-1)*nB, nA+domain*nB)
            half = len(ids)//2
            phase(chr(65+domain)+'_mid', ids[:half], domain)
            phase(chr(65+domain)+'_end', ids[half:], domain)
            if domain < domains:
                shifts.append(len(events))
    else:
        phase('association_end', np.arange(nA, count))
    probes = dict(independent)
    probes['old_exact'] = (probe[:nA].reshape(-1, 12, 32), np.repeat(current[:nA], 2))
    fresh = bases(nA, 0, 'old_fresh_latents')
    probes['old_fresh'] = (render(fresh, 2, rng(seed, 'old_fresh_noise')).reshape(-1, 12, 32)@transform,
        np.repeat(current[:nA], 2))
    probes['trained_new'] = (probe[nA:].reshape(-1, 12, 32), np.repeat(current[nA:], 2))
    probes['association'] = (probe.reshape(-1, 12, 32), np.repeat(current, 2))
    # Programmatic provenance: exact probes retain the train latent array; unseen
    # B is generated by independent named streams, not another rendering of B.
    assert np.array_equal(latent[:nA], old)
    if structured:
        assert all(i >= nA for i, _, _, _ in events[shifts[0]:])
        for domain in range(2, domains+1):
            assert all(dd >= domain for _, _, _, dd in events[shifts[domain-1]:])
    return dict(train=train, calibration=calibration, events=events, checkpoints=checkpoints,
        probes=probes, old_n=nA, shifts=shifts, labels=current, train_latents=latent,
        exact_probe_latents=latent.copy(), latent_probe_overlap=0,
        meta=dict(task=task, active_block_stride=4 if task == 'overlap' else 8, A_arbitrary=True, A_label_revisions=2, A_revision_identities=len(rev),
            old_exact_latent_overlap=nA, independent_unseen_overlap=0,
            unique_A=nA, unique_new_per_domain=nB, structured_domains=domains if structured else 0,
            independent_unseen_per_domain=d['probes'] if structured else 0,
            calibration=d['calibration'], calibration_domain='A only',
            equal_marginals='8 active coordinates exchangeable; inactive background .12' if task == 'equal' else None,
            old_replay_after_shift=0 if structured else None,
            training_observations=2*count, total_labeled_presentations=len(events)))


def _v29_operational_spec(arm):
    """Keep control labels outside the learner; preserve V28 operations exactly."""
    keys = ('formula', 'rule', 'rate', 'fraction', 'path', 'stable', 'matched', 'unit')
    return {k: arm[k] for k in keys}


def _v29_factorial(bank0, bank1, r0, z0, r1, z1, labels):
    values, effects = {}, {}
    for view in VIEWS:
        scores = {}
        for e in (0, 1):
            r, z = (r0, z0) if e == 0 else (r1, z1)
            for router in (0, 1):
                for weights in (0, 1):
                    # An evaluator copy only. Scores do not use route counts.
                    proxy = copy.copy(bank0)
                    proxy.centers = (bank0 if router == 0 else bank1).centers
                    proxy.w = (bank0 if weights == 0 else bank1).w
                    value = measure(proxy, r, z, labels, view)
                    scores[e, router, weights] = value
                    values[f'{view}_E{e}_R{router}_W{weights}'] = value
        base = scores[0, 0, 0]
        effect = dict(encoder=scores[1, 0, 0]-base,
                      router=scores[0, 1, 0]-base,
                      weights=scores[0, 0, 1]-base)
        effect['encoder_router'] = scores[1, 1, 0]-scores[1, 0, 0]-scores[0, 1, 0]+base
        effect['encoder_weights'] = scores[1, 0, 1]-scores[1, 0, 0]-scores[0, 0, 1]+base
        effect['router_weights'] = scores[0, 1, 1]-scores[0, 1, 0]-scores[0, 0, 1]+base
        effect['triple'] = (scores[1, 1, 1]-scores[1, 1, 0]-scores[1, 0, 1]
            -scores[0, 1, 1]+scores[1, 0, 0]+scores[0, 1, 0]+scores[0, 0, 1]-base)
        effect['total'] = scores[1, 1, 1]-base
        if abs(sum(v for k, v in effect.items() if k != 'total')-effect['total']) > 1e-12:
            raise AssertionError('Factorial decomposition failed to reconstruct total')
        effects[view] = effect
    return values, effects


def _v29_stat(values, seeds, confidence=.95):
    values, seeds = list(values), list(seeds)
    if len(values) != len(seeds):
        raise ValueError('One statistic value per declared seed is required')
    good = [v is not None and _v29_math.isfinite(float(v)) for v in values]
    out = dict(n=sum(good), expected=len(seeds), confidence=confidence,
               missing_seeds=[int(s) for s, ok in zip(seeds, good) if not ok],
               mean=None, low=None, high=None, status='incomplete')
    if not all(good) or len(seeds) < 2:
        return out
    a = np.asarray(values, dtype=float)
    constant = bool(np.all(a == a[0]))
    mean = float(a[0]) if constant else float(a.mean())
    half = (0. if constant else float(student_t.ppf((1.+confidence)/2., len(a)-1)
                 * a.std(ddof=1)/_v29_math.sqrt(len(a))))
    out.update(mean=mean, low=mean-half, high=mean+half, status='complete')
    return out


def _v29_index(records):
    result = {}
    for r in records:
        key = (r['task'], r['spec']['name'], int(r['seed']))
        if key in result:
            raise ValueError('Duplicate task/arm/seed in V29 report: '+repr(key))
        result[key] = r
    return result


def _v29_value(record, key, view='hard'):
    if record is None:
        return None
    return (record.get('terminal') or {}).get(view, {}).get(key)


def _v29_boundary(record, cohort='A', key=None, view='hard'):
    if record is None:
        return None
    key = key or ('old_exact' if cohort == 'A' else cohort+'_unseen')
    if cohort == 'A':
        return (record.get('a_boundary') or {}).get(view, {}).get(key)
    found = [r for r in record.get('rows', [])
             if r.get('checkpoint') == cohort+'_end' and r.get('view') == view]
    if len(found) != 1:
        return None
    return found[0].get(key)


def _v29_return(record, key, view='hard'):
    if record is None:
        return None
    return (record.get('return_summary') or {}).get(view, {}).get(key)


def _v29_eval_stat(index, task, seeds, arms, fn, confidence=.95):
    values = []
    for seed in seeds:
        records = [index.get((task, arm, int(seed))) for arm in arms]
        if any(r is None for r in records):
            values.append(None)
            continue
        try:
            values.append(fn(*records))
        except (KeyError, TypeError, ValueError, IndexError):
            values.append(None)
    return _v29_stat(values, seeds, confidence)


def _v29_level(index, task, arm, seeds, key, confidence=.95, view='hard'):
    return _v29_eval_stat(index, task, seeds, (arm,),
                          lambda a: _v29_value(a, key, view), confidence)


def _v29_diff(index, task, a, b, seeds, key, confidence=.95, view='hard'):
    return _v29_eval_stat(index, task, seeds, (a, b),
                          lambda x, y: _v29_value(x, key, view)-_v29_value(y, key, view),
                          confidence)


def _v29_lower_gate(stat, limit=0., strict=False, mean_min=None):
    """Pass iff the exact declared rule holds; otherwise distinguish uncertainty."""
    out = dict(stat=stat, threshold=limit, strict=strict, mean_min=mean_min,
               status='incomplete')
    if stat['status'] != 'complete':
        return out
    lower_ok = stat['low'] > limit if strict else stat['low'] >= limit
    mean_ok = mean_min is None or stat['mean'] >= mean_min
    if lower_ok and mean_ok:
        out['status'] = 'pass'
    elif ((stat['high'] <= limit if strict else stat['high'] < limit)
          or (mean_min is not None and stat['high'] < mean_min)):
        out['status'] = 'fail'
    else:
        out['status'] = 'inconclusive'
    return out


def _v29_conjunction(statuses):
    statuses = list(statuses)
    if not statuses or 'incomplete' in statuses:
        return 'incomplete'
    if 'fail' in statuses:
        return 'fail'
    if all(s == 'pass' for s in statuses):
        return 'pass'
    return 'inconclusive'


def _v29_complete(index, task, seeds, arms):
    missing = [dict(seed=int(s), arm=a) for s in seeds for a in arms
               if (task, a, int(s)) not in index]
    return dict(status='incomplete' if missing else 'pass', missing=missing,
                expected=len(seeds)*len(arms), present=len(seeds)*len(arms)-len(missing))


def csv_file(path, rows):
    if not rows:
        atomic(path, '')
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: serialized(v) if isinstance(v, (dict,list,tuple)) else v
                         for k,v in row.items()})
    atomic(path, out.getvalue())


class Paused(Exception):
    pass

SEED_AUDIT={'checked_sources': ['v29_dev_smoke/PLAN.json', 'v29_dev_smoke/MANIFEST.json', 'v29_final_validation/PLAN.json', 'v29_final_validation/MANIFEST.json', 'v28_audit_data/PLAN.json', 'v28_audit_data/MANIFEST.json', 'v28_validation_smoke/PLAN.json', 'v28_validation_smoke/MANIFEST.json', 'v30_v29_data/PLAN.json', 'v30_v29_data/MANIFEST.json', 'v28_release_smoke/PLAN.json', 'v28_release_smoke/MANIFEST.json', 'v28_final_smoke/PLAN.json', 'v28_final_smoke/MANIFEST.json', 'project_sources/09-RUN_MANIFEST.json', 'project_sources/05-RUN_MANIFEST.json', 'project_sources/11-RUN_MANIFEST.json', 'elm_fly_minimal_core_v29.py', 'elm_fly_persistent_sparse_v28.py'], 'planned_seed_collisions': [], 'scope': 'Available previous manifests and V28/V29 source literals;not a claim to have every historical seed file'}
DESIGN_SHA256='d948a32959a300e91f9a42646e524662d24563ca737cb445e8c962a9c6904e3f'


# V30 bank writers. No identity, task, boundary or evaluator probes enter learning.
def write_direction(z, inverse_action):
    zz = float(z @ z)
    if zz == 0.:
        return np.zeros_like(z), False
    den = float(z @ inverse_action)
    if not math.isfinite(den) or den <= 0.:
        raise FloatingPointError('Nonpositive/nonfinite directional denominator')
    q0 = z/zz
    d = inverse_action/den-q0
    d -= z*(float(z@d)/zz)  # remove roundoff normal component
    dn = float(np.linalg.norm(d))
    limit = float(np.linalg.norm(q0))*math.sqrt(1/.6**2-1)
    clipped = dn > limit
    if clipped:
        d *= limit/dn
    q = q0+d
    if (not np.isfinite(q).all() or abs(float(z@q)-1) > 1e-10
            or .6*np.linalg.norm(q) > np.linalg.norm(q0)*(1+1e-10)):
        raise FloatingPointError('Write-direction identity/norm-bound violation')
    return q, clipped


class DirectionBanks(Banks):
    def __init__(self, seed, writer='ordinary', shadow=False):
        super().__init__(seed)
        self.writer = writer
        self.freeze_router = False
        self.P = np.repeat(np.eye(96)[None], 8, axis=0) if shadow or writer.startswith('full') else None
        self.h = np.ones((8,96)) if shadow or writer.startswith('diag') else None
        self.permutations = np.stack([rng(seed, 'v30_permutation_'+str(b)).permutation(96)
            for b in range(8)]) if shadow or writer.endswith('_shuffle') else None
        # calls, zero, capped, path, maxstep, maxcorrectionerror, maxnormerror,
        # inactive squared norm, total squared norm, alpha sum
        self.metrics = np.zeros(10)
        self.last_delta = np.zeros((96,4))  # transient write audit, never read by learner

    def learn(self, r, z, label):
        b = int(self.routes(r))
        self.metrics[0] += 1
        zz = float(z@z)
        if not np.isfinite(z).all() or not math.isfinite(zz):
            raise FloatingPointError('Nonfinite code')
        self.last_delta.fill(0)
        if zz == 0.:
            self.metrics[1] += 1
            return b
        e = np.eye(4)[int(label)]-z@self.w[b]
        q0 = z/zz
        v = np.einsum('ij,j->i', self.P[b], z) if self.P is not None else None
        q, clipped, alpha = q0, False, .6
        if self.writer != 'ordinary':
            if self.writer.startswith('full'):
                if self.P is None:
                    raise RuntimeError('Full statistic missing')
                u = v
                if self.writer.endswith('_shuffle'):
                    ix = self.permutations[b]
                    u = np.einsum('ij,j->i', self.P[b][np.ix_(ix,ix)], z)
            elif self.writer.startswith('diag'):
                if self.h is None:
                    raise RuntimeError('Diagonal statistic missing')
                h = self.h[b]
                if self.writer.endswith('_shuffle'):
                    h = h[self.permutations[b]]
                u = z/h
            else:
                raise ValueError('Unknown writer: '+self.writer)
            q, clipped = write_direction(z,u)
            if self.writer.endswith('_norm'):
                alpha = .6*float(np.linalg.norm(q)/np.linalg.norm(q0))
                q = q0
        delta = alpha*np.outer(q,e)
        # Ordinary arithmetic order is identical to V29 for exact continuity.
        if self.writer == 'ordinary':
            delta = .6*np.outer(z,e)/max(zz,1e-12)
        self.w[b] += delta
        self.last_delta = delta
        correction = float(np.max(np.abs((np.eye(4)[int(label)]-z@self.w[b])-(1-alpha)*e)))
        step = float(np.linalg.norm(delta))
        self.metrics[2] += int(clipped)
        self.metrics[3] += step
        self.metrics[4] = max(self.metrics[4],step)
        self.metrics[5] = max(self.metrics[5],correction)
        self.metrics[6] = max(self.metrics[6], max(0.,step-float(np.linalg.norm(q0)*np.linalg.norm(e))))
        self.metrics[7] += float(np.sum(delta[z==0]**2))
        self.metrics[8] += float(np.sum(delta**2))
        self.metrics[9] += alpha
        if correction > 1e-9 or not np.isfinite(self.w[b]).all():
            raise FloatingPointError('Invalid bank correction')
        if self.P is not None:
            den = 1+float(z@v)
            if not math.isfinite(den) or den <= 0.:
                raise FloatingPointError('Invalid Sherman-Morrison denominator')
            self.P[b] -= np.outer(v,v)/den
            if not np.isfinite(self.P[b]).all():
                raise FloatingPointError('Nonfinite inverse statistic')
        if self.h is not None:
            self.h[b] += z*z
        self.counts[b] += 1
        if not self.freeze_router:
            rate = min(.05,1/float(self.counts[b]))
            self.centers[b] = norm_rows((1-rate)*self.centers[b]+rate*norm_rows(r))
        return b

    def digest(self):
        h = hashlib.sha256(super().digest().encode())
        for a in (self.P,self.h,self.permutations,self.metrics,self.last_delta):
            if a is not None:
                h.update(a.tobytes())
        h.update((self.writer+str(self.freeze_router)).encode())
        return h.hexdigest()

    def summary(self):
        m = self.metrics
        return dict(calls=int(m[0]),zero_codes=int(m[1]),capped_writes=int(m[2]),
            cap_fraction=float(m[2]/max(1,m[0]-m[1])),cumulative_displacement=float(m[3]),
            maximum_step=float(m[4]),maximum_correction_error=float(m[5]),
            maximum_norm_bound_error=float(m[6]),
            inactive_row_squared_norm_fraction=float(m[7]/max(m[8],1e-30)),
            mean_write_coefficient=float(m[9]/max(1,m[0]-m[1])),
            per_bank_presentations=self.counts.tolist(),writer=self.writer)

    def costs_v30(self):
        state = (self.P.nbytes if self.writer.startswith('full') else
                 self.h.nbytes if self.writer.startswith('diag') else 0)
        perm = self.permutations.nbytes if self.writer.endswith('_shuffle') else 0
        allocated = sum(a.nbytes for a in (self.P,self.h,self.permutations) if a is not None)
        return dict(functional_statistic_bytes=int(state),functional_permutation_bytes=int(perm),
            shadow_statistic_bytes=int(allocated-state-perm),
            bank_monitor_bytes=self.metrics.nbytes,transient_last_write_bytes=self.last_delta.nbytes,
            update_scratch_note='O(96^2) full or O(96) diagonal; temporary arrays and process peak reported separately')


def resource_peak():
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(v if sys.platform=='darwin' else 1024*v)
    except (ImportError,AttributeError):
        return None


def bank_diagnostics(bank, zA):
    out = bank.summary()
    out['diagonal_range'] = [float(bank.h.min()),float(bank.h.max())] if bank.h is not None else None
    if bank.h is not None:
        out['current_A_diagonal_inverse_energy_per_bank'] = np.einsum('pi,bi->b',zA*zA,1/bank.h)/max(1,len(zA))
    if bank.P is not None:
        eigen = np.linalg.eigvalsh(bank.P)
        if eigen.min() <= 0 or not np.isfinite(eigen).all():
            raise FloatingPointError('Inverse covariance lost positive definiteness')
        out.update(inverse_eigen_min=float(eigen.min()),inverse_eigen_max=float(eigen.max()),
            maximum_condition=float(np.max(eigen[:,-1]/eigen[:,0])),
            symmetry_error=float(np.max(np.abs(bank.P-bank.P.transpose(0,2,1)))),
            current_A_inverse_energy_per_bank=np.einsum('pi,bij,pj->b',zA,bank.P,zA)/max(1,len(zA)))
    return clean(out)


class WriteLedger:
    """Evaluator only. Frozen A probe codes NEVER enter DirectionBanks."""
    def __init__(self,r,z,y,bank):
        self.routes=bank.routes(r).copy()
        self.z=z.copy(); self.y=y.copy()
        self.delta_sum=np.zeros((len(z),4))
        self.margin_sum=0.;self.true_logit_sum=0.;self.energy=0.;self.adverse=0;self.calls=0
        self.initial_w=bank.w.copy()
        self.code_route_hash=hashlib.sha256()
        self.max_identity_error=0.
        self.old_new_support_overlap_sum=0.;self.overlap_pairs=0

    def observe(self,b,r,z,delta,bank):
        self.code_route_hash.update(r.tobytes()+z.tobytes()+int(b).to_bytes(4,'little'))
        mask=self.routes==b
        za=self.z[mask]; ya=self.y[mask]
        change=za@delta
        post=za@bank.w[b];pre=post-change
        def margin(scores):
            other=scores.copy(); other[np.arange(len(scores)),ya]=-np.inf
            return scores[np.arange(len(scores)),ya]-other.max(1)
        dm=margin(post)-margin(pre) if len(za) else np.array([])
        self.delta_sum[mask]+=change
        self.margin_sum+=float(dm.sum()); self.adverse+=int(np.sum(dm<0))
        self.true_logit_sum+=float(change[np.arange(len(za)),ya].sum())
        self.energy+=float(np.sum(change**2));self.calls+=1
        self.old_new_support_overlap_sum+=float(np.sum((za!=0)&(z[None]!=0))/24)
        self.overlap_pairs+=len(za)
        # Telescoping fixed-code logits, independent of later probe re-encoding.
        expected=np.einsum('pi,pic->pc',self.z,bank.w[self.routes]-self.initial_w[self.routes])
        err=float(np.max(np.abs(expected-self.delta_sum)))
        self.max_identity_error=max(self.max_identity_error,err)
        if err>1e-8:
            raise AssertionError('Per-write logit ledger failed telescoping identity')

    def summary(self):
        return dict(writes=self.calls,true_class_logit_change_sum=self.true_logit_sum,
            true_class_margin_change_sum=self.margin_sum,adverse_probe_write_pairs=self.adverse,
            squared_logit_change_sum=self.energy,maximum_telescoping_error=self.max_identity_error,
            mean_old_new_active_support_overlap=self.old_new_support_overlap_sum/max(1,self.overlap_pairs),
            old_new_overlap_probe_write_pairs=self.overlap_pairs,
            B_code_route_hash=self.code_route_hash.hexdigest(),
            evaluation_only_bytes=self.z.nbytes+self.y.nbytes+self.routes.nbytes+
                self.delta_sum.nbytes+self.initial_w.nbytes,
            interpretation='Fixed A-boundary codes and routes; actual delta for every writer; no probe feedback to learning')


def run_trial_v30(seed, task, arm, smoke=False, lifetime=False):
    started = time.perf_counter()
    stream = make_stream_v29(seed, task, smoke=smoke, lifetime=lifetime)
    operational = _v29_operational_spec(arm)
    bundle = Bundle(seed, operational, stream['calibration'])
    if lifetime:
        raise ValueError('V30 has no multi-shift stage')
    diagnostic = arm.get('diagnostic', False)
    bank = DirectionBanks(seed, 'ordinary' if diagnostic else arm['writer'], shadow=diagnostic)
    ledger = None
    initial = bundle.value.w.copy()
    initial_key = bundle.address.w.copy()
    n = len(stream['train'])
    counts = np.zeros(n, int)
    last_labels = np.full(n, -1)
    last_routes = np.full(n, -1)
    hits = {v: {} for v in VIEWS}
    rows, diagnostics = [], []
    snapshots, cohort_boundaries = {}, {}
    a_boundary, a_boundary_hash = None, None
    strict_exposures = len(stream['events'])
    checkpoints = dict(stream['checkpoints'])
    early = (2, 4, 8) if smoke else (24, 96, 192)
    for domain, start in enumerate(stream['shifts'], 1):
        stop = stream['shifts'][domain] if domain < len(stream['shifts']) else strict_exposures
        for exposed in early:
            if start+exposed <= stop:
                checkpoints.setdefault(start+exposed, f'{chr(65+domain)}_early{exposed}')
    probe_bytes = sum(x.nbytes+y.nbytes for x, y in stream['probes'].values())

    def add(view, name, value):
        a = hits[view].setdefault(name, [0, 0])
        a[0] += int(value)
        a[1] += 1

    for t, (identity, variant, label, domain) in enumerate(stream['events'], 1):
        x = stream['train'][identity, variant]
        r, z = bundle.features(x)
        r, z = r[0], z[0]
        for view in VIEWS:
            hit = int(np.argmax(bank.scores(r, z, view)[0]) == label)
            add(view, 'prequential', hit)
            add(view, f'domain_{domain}_prequential', hit)
            if counts[identity] == 0:
                add(view, f'domain_{domain}_first', hit)
            if domain == 0:
                add(view, 'A_prequential', hit)
            if counts[identity] == 1:
                if last_labels[identity] != label:
                    raise AssertionError('Second presentation unexpectedly revised')
                add(view, 'second', hit)
                add(view, f'domain_{domain}_second', hit)
                if domain == 0:
                    add(view, 'A_second', hit)
        last_routes[identity] = bank.learn(r, z, label)
        if ledger is not None:
            ledger.observe(int(last_routes[identity]), r, z, bank.last_delta, bank)
        bundle.update(x)
        counts[identity] += 1
        last_labels[identity] = label
        if t not in checkpoints:
            continue
        digest = bundle.digest()+bank.digest()
        cache = {}
        if t >= stream['shifts'][0]:
            allowed = ['old_exact', 'old_fresh']
            if domain > 0:
                allowed += [chr(65+k)+'_unseen' for k in range(1, domain+1)]
            if t == strict_exposures:
                allowed += ['trained_new', 'association']
            for key in allowed:
                if key in stream['probes']:
                    cache[key] = bundle.features(stream['probes'][key][0])
        checkpoint_rows = {}
        for view in VIEWS:
            row = dict(view=view, checkpoint=checkpoints[t], presentations=t,
                domain=domain, unique_identities=int(np.sum(counts > 0)),
                domain_presentations=t if domain == 0 else t-stream['shifts'][domain-1])
            if t == stream['shifts'][0]:
                row['boundary'] = 'A_end'
            for key, (num, den) in hits[view].items():
                row[key] = num/den
                row[key+'_count'] = den
            for key, (rr, zz) in cache.items():
                row[key] = measure(bank, rr, zz, stream['probes'][key][1], view)
            if domain > 0:
                row['new_unseen'] = row[chr(65+domain)+'_unseen']
            rows.append(row)
            checkpoint_rows[view] = row
        if t == stream['shifts'][0]:
            a_boundary = copy.deepcopy(checkpoint_rows)
            # Operational specs have no descriptive arm/control fields. Full digest
            # therefore verifies weights, normalization, RNG, clocks, and banks.
            a_boundary_hash = hashlib.sha256(digest.encode()).hexdigest()
        acquired = None
        if t == stream['shifts'][0]:
            acquired = 'old_exact'
        elif checkpoints[t] == chr(65+domain)+'_end':
            acquired = chr(65+domain)+'_unseen'
        if acquired is not None:
            rr, zz = cache[acquired]
            snapshots[acquired] = (rr.copy(), zz.copy(), copy.deepcopy(bank))
            cohort_boundaries[acquired] = dict(checkpoint=checkpoints[t], presentations=t,
                **{view: checkpoint_rows[view][acquired] for view in VIEWS})
        if 'old_exact' in cache:
            R, Z = cache['old_exact']
            r0, z0, m0 = snapshots['old_exact']
            y = stream['probes']['old_exact'][1]
            cf, effects = _v29_factorial(m0, bank, r0, z0, R, Z, y)
            same = m0.routes(R) == m0.routes(r0)
            diagnostics.append(dict(checkpoint=checkpoints[t], presentations=t, domain=domain,
                bank_write=bank_diagnostics(bank,Z),
                feature_active_overlap=float(np.mean(np.sum((Z!=0)&(z0!=0),axis=1)/24)),
                sensory_clip_fraction=float(np.mean(np.abs((bundle.value.raw(stream['probes']['old_exact'][0])-bundle.value.mean)/bundle.value.sd)>=5)),
                write_ledger=ledger.summary() if ledger is not None else None,
                encoder=copy.deepcopy(bundle.value.audit(initial)),
                key_encoder=copy.deepcopy(bundle.address.audit(initial_key)),
                feature_code_drift=float(np.mean((Z-z0)**2)),
                route_change=float(np.mean(bank.routes(R) != m0.routes(r0))),
                address_only_route_change=float(np.mean(m0.routes(R) != m0.routes(r0))),
                versus_last_write_route_change=float(np.mean(bank.routes(R) != np.repeat(last_routes[:stream['old_n']], 2))),
                same_route_content_drift=float(np.mean((Z[same]-z0[same])**2)) if same.any() else None,
                same_route_count=int(same.sum()), counterfactual=cf, effects=effects))
        if digest != bundle.digest()+bank.digest():
            raise AssertionError('Evaluator mutated learner state')
        if t == stream['shifts'][0]:
            ledger = WriteLedger(cache['old_exact'][0],cache['old_exact'][1],
                stream['probes']['old_exact'][1],bank)
            if diagnostic:
                bank.writer = arm['writer']
            bank.freeze_router = bool(arm.get('freeze_router_after_A',False))
        if t == stream['shifts'][0] and arm.get('freeze_after_A', False):
            # Diagnostic boundary control; practical arms receive no phase signal.
            # one-path address aliases value, so both representations are frozen.
            bundle.value.spec['rate'] = 0.

    strict_terminal = {v: copy.deepcopy([r for r in rows if r['view'] == v][-1]) for v in VIEWS}
    strict_audit = copy.deepcopy(bundle.value.audit(initial))
    strict_update_calls = sum(int(e.budget[4]) for e in
        ([bundle.value] if bundle.address is bundle.value else [bundle.value, bundle.address]))
    strict_bank_hash = bank.digest()
    strict_writer = bank.summary()
    strict_ledger = ledger.summary()
    strict_w = bank.w.copy()
    return_summary = None
    return_rows = []
    return_exposures = 0
    if True:  # Revision is mandatory, regardless of screening outcome.
        revise_n = 2 if smoke else 6
        revised = np.arange(revise_n)
        old_labels = stream['labels'][revised].copy()
        new_labels = (old_labels+1) % 4
        # Exact return probes are independent noisy views, as for strict retention.
        old_x, old_y = stream['probes']['old_exact']
        revised_probe = old_x[:2*revise_n]
        revised_target = np.repeat(new_labels, 2)
        unchanged_probe, unchanged_target = old_x[2*revise_n:], old_y[2*revise_n:]
        return_summary = {v: dict() for v in VIEWS}
        revision_counts = np.zeros(revise_n, int)
        return_hits = {v: {} for v in VIEWS}

        def return_add(view, name, hit):
            pair = return_hits[view].setdefault(name, [0, 0])
            pair[0] += int(hit)
            pair[1] += 1

        def return_probe(round_number):
            before = bundle.digest()+bank.digest()
            ur, uz = bundle.features(unchanged_probe)
            rr, zz = bundle.features(revised_probe)
            for view in VIEWS:
                unchanged_score = measure(bank, ur, uz, unchanged_target, view)
                revised_score = measure(bank, rr, zz, revised_target, view)
                row = dict(view=view, checkpoint=f'return_after_{round_number}',
                    presentations=strict_exposures+return_exposures,
                    return_presentations=return_exposures,
                    labels_per_revised_identity=round_number,
                    old_unchanged=unchanged_score, revised_exact=revised_score)
                for name, (num, den) in return_hits[view].items():
                    row[name] = num/den
                    row[name+'_count'] = den
                return_rows.append(row)
                if round_number == 0:
                    return_summary[view]['pre_return_unchanged'] = unchanged_score
                    return_summary[view]['pre_return_revised_against_new_label'] = revised_score
                elif round_number in (1, 2, 6):
                    label = {1: 'one', 2: 'two', 6: 'six'}[round_number]
                    return_summary[view]['revised_after_'+label] = revised_score
                    return_summary[view]['unchanged_after_'+label] = unchanged_score
                if round_number == 6:
                    return_summary[view]['post_return_unchanged'] = unchanged_score
            if before != bundle.digest()+bank.digest():
                raise AssertionError('Return evaluator mutated learner state')

        return_probe(0)
        for round_index in range(6):
            order = rng(seed, 'v29_return_round_'+str(round_index)).permutation(revised)
            variant = round_index % 2
            for identity in order:
                identity = int(identity)
                x = stream['train'][identity, variant]
                r, z = bundle.features(x)
                r, z = r[0], z[0]
                label = int(new_labels[identity])
                for view in VIEWS:
                    predicted = int(np.argmax(bank.scores(r, z, view)[0]))
                    hit = int(predicted == label)
                    return_add(view, 'revision_prequential', hit)
                    if revision_counts[identity] == 0:
                        return_add(view, 'first_return_old_label', predicted == int(old_labels[identity]))
                        return_add(view, 'first_return_new_label', hit)
                    elif revision_counts[identity] == 1:
                        return_add(view, 'revision_second', hit)
                last_routes[identity] = bank.learn(r, z, label)
                bundle.update(x)
                counts[identity] += 1
                last_labels[identity] = label
                revision_counts[identity] += 1
                return_exposures += 1
            if not np.all(revision_counts == round_index+1):
                raise AssertionError('Return endpoint mixes exposure counts')
            if round_index+1 in (1, 2, 6):
                return_probe(round_index+1)
        for view in VIEWS:
            for name, (num, den) in return_hits[view].items():
                return_summary[view][name] = num/den
                return_summary[view][name+'_count'] = den
            return_summary[view].update(exposures=return_exposures,
                unique_revised_identities=revise_n, labels_per_revised_identity=6,
                first_return_protocol='chronological first prediction before that identity receives its changed label',
                schedule='six identity rounds; alternate existing variant 0/1; identity order shuffled per round',
                labels_changed=True, unchanged_old_training_presentations=0)

    encoders = [bundle.value] if bundle.address is bundle.value else [bundle.value, bundle.address]
    snapshot_bytes = initial.nbytes+initial_key.nbytes
    for rr, zz, mm in snapshots.values():
        snapshot_bytes += rr.nbytes+zz.nbytes+mm.w.nbytes+mm.centers.nbytes+mm.counts.nbytes
        snapshot_bytes += sum(a.nbytes for a in (mm.P,mm.h,mm.permutations,mm.metrics,mm.last_delta) if a is not None)
    total_exposures = strict_exposures+return_exposures
    if int(bank.counts.sum()) + int(bank.metrics[1]) != total_exposures:
        raise AssertionError('Bank updates do not match labeled exposure ledger')
    metadata = dict(stream['meta'])
    metadata.update(strict_labeled_presentations=strict_exposures,
        return_labeled_presentations=return_exposures, total_labeled_presentations=total_exposures,
        old_replay_before_strict_endpoint=0,
        return_is_new_revised_experience=True,
        freeze_after_A_is_diagnostic=bool(arm.get('freeze_after_A', False)))
    return clean(dict(seed=seed, task=task, spec=dict(arm), horizon=12, lifetime=lifetime,
        rows=rows, diagnostics=diagnostics, terminal=strict_terminal,
        a_boundary=a_boundary, a_boundary_hash=a_boundary_hash,
        cohort_boundaries=cohort_boundaries, return_summary=return_summary, return_rows=return_rows,
        audit=strict_audit, final_after_return_audit=bundle.value.audit(initial),
        strict_bank_hash=strict_bank_hash, writer_summary=strict_writer,
        after_return_writer_summary=bank.summary(),strict_write_ledger=strict_ledger,
        strict_weight_net_displacement=float(np.linalg.norm(strict_w-snapshots['old_exact'][2].w)),
        total_write_net_displacement=float(np.linalg.norm(bank.w)),
        process_peak_rss_bytes=resource_peak(),
        numeric_memory=bank.costs_v30(),
        baseline_total_learner_numeric_bytes=sum(e.costs()['total_numeric_bytes'] for e in encoders)
            +bank.w.nbytes+bank.centers.nbytes+bank.counts.nbytes,
        costs=dict(encoder=[e.costs() for e in encoders], bank_weights_bytes=bank.w.nbytes,
            router_bytes=bank.centers.nbytes+bank.counts.nbytes,
            total_learner_numeric_bytes=sum(e.costs()['total_numeric_bytes'] for e in encoders)
                +bank.w.nbytes+bank.centers.nbytes+bank.counts.nbytes
                +bank.costs_v30()['functional_statistic_bytes']+bank.costs_v30()['functional_permutation_bytes'],
            diagnostic_shadow_statistic_bytes=bank.costs_v30()['shadow_statistic_bytes'],
            writer_monitor_bytes=bank.metrics.nbytes, transient_last_delta_bytes=bank.last_delta.nbytes,
            evaluator_write_ledger_bytes=strict_ledger['evaluation_only_bytes'],
            evaluator_probe_bytes=probe_bytes, evaluator_snapshot_bytes=snapshot_bytes,
            environment_training_array_bytes=stream['train'].nbytes+stream['train_latents'].nbytes+stream['exact_probe_latents'].nbytes,
            environment_calibration_array_bytes=stream['calibration'].nbytes,
            evaluator_id_ledger_bytes=counts.nbytes+last_labels.nbytes+last_routes.nbytes,
            labeled_presentations=total_exposures, strict_labeled_presentations=strict_exposures,
            return_labeled_presentations=return_exposures,
            bank_update_events=int(bank.counts.sum()), strict_bank_update_events=int(strict_writer["calls"]-strict_writer["zero_codes"]),
            unlabeled_calibration_presentations=len(stream['calibration']),
            calibration_encoder_evaluations=len(stream['calibration'])*len(encoders),
            representation_update_calls=sum(int(e.budget[4]) for e in encoders),
            strict_representation_update_calls=strict_update_calls),
        stream_metadata=metadata, seconds=time.perf_counter()-started))


SCREEN_SEEDS=tuple(range(30001,30009))
FRESH_SEEDS=tuple(range(31001,31013))
SMOKE_SCREEN=(94103,94109)
SMOKE_FRESH=(95107,95111)


def arm(name,writer='ordinary',diagnostic=False,freeze=False,router_freeze=False):
    return dict(spec('top',2),name=name,writer=writer,diagnostic=diagnostic,
        freeze_after_A=freeze,freeze_router_after_A=router_freeze)


def screen_arms():
    return [arm('L',diagnostic=True),arm('F',diagnostic=True,freeze=True),
        arm('R',diagnostic=True,router_freeze=True),
        arm('S',diagnostic=True,freeze=True,router_freeze=True)]+[
        arm(name,writer,True,True,True) for name,writer in
        [('D','diag'),('P','full'),('DX','diag_shuffle'),('PX','full_shuffle'),
         ('DN','diag_norm'),('PN','full_norm')]]


def practical_arms(writer):
    tag={'diag':'D','full':'P'}[writer]
    return [arm('O_live'),arm('O_freeze',freeze=True),
            arm(tag+'_live',writer),arm(tag+'_freeze',writer,freeze=True)]


POLICY={
 'question':'One bank-write family: ordinary vs diagonal/full historical co-activation direction, with equal immediate correction and bounded displacement.',
 'substrate':'Unchanged V29 D encoder,96 cells,8 contacts,top12 per48-cell half,top25/rate2 sensory adaptation,8x96x4 banks,ordinary correction.6.',
 'tasks':'Unchanged V29 nuisance and adjacent50% overlap;1728 labeled presentations,96 A-only calibration;36-label revised-A tail after strict evaluation.',
 'screen':'Identical top+ordinary A prefix with shadow P/h,10 branches including encoder/router2x2 factorial, diagonal/full and their own misassignment/norm controls.8 paired seeds per task.',
 'direction':'P=(I+sum zzT)^-1 or h=1+sum z^2; qraw=Pz/(zTPz) or (z/h)/(sum z^2/h); clip orthogonal(qraw-q0) to ||q||<=||q0||/.6, retaining zTq=1. Update statistic after write.',
 'memory':'Full adds589824bytes;diag adds6144bytes float64. Functional/shadow/evaluator/temporary costs separate. Full is not a minimal local core.',
 'misassignment':'Fixed per-bank permutation conjugates P or permutes h at read only; statistics continue on actual unpermuted codes.',
 'norm_control':'Isotropic alpha=.6||q||/||q0|| in[.6,1] at comparator own state; not equal achieved cross-trajectory cumulative displacement.',
 'screen_gates':'98.75% paired intervals. Overlap old gain vsS mean>=.05 and lower>0; nuisance old lower>=-.03. Both tasks B transfer/prequential/second lower>=-.03. Overlap old>own norm/misassignment controls lower>0 and B transfer lower>=-.03. Revision2/6 lower>=-.05 vsS; unchanged post/pre lower>=-.05;revision6 lower>.25.',
 'selection':'Prefer diagonal if eligible;otherwise full if eligible;otherwise stop. No tuning or added seeds. Full-only selection labeled larger reference.',
 'practical':'12 fresh paired seeds/task;ordinary and selected writer crossed with live/freeze-after-A encoder;router always live;selected writer operates from first A label.97.5% intervals.',
 'practical_bank_gate':'Overlap old mean gain>=.05,lower>0;nuisance old lower>=-.03. B transfer/prequential/second and A acquired/prequential/second lower>=-.03 vsordinary-live. Revision2/6 lower>=-.05 vsordinary-live;unchanged withinarm lower>=-.05;revision6 lower>.25.',
 'representation_gate':'Selected live vs identical-A freeze: B transfer mean>=.05,lower>0;old and Bprequential/second lower>=-.05;A acquired lower>.25;paired retained-excess lower>=0 for postB-.25-.75*(A_boundary-.25).',
 'return':'First6 A identities,change current label+1mod4,6 shuffled rounds alternating two existing views;score0/1/2/6 labels and first prediction against old/new labels. No unchanged-A training or return-before-strict.',
 'statistics':'Paired seed unit;every expected seed required;all endpoints conjunction;missing/fail/inconclusive separate. Smoke verdicts disabled;no best read view.',
 'limits':'No longevity experiment;no unlimited memory claim;full statistics can update inactive rows;history contains write-time codes,not current probe representations;statistics weight exposures not semantic utility.',
 'runtime':'One file/command;macOS caffeinate;atomic checksummed records;locked output directory;strict resume identity including source/dependencies/plan;compact ZIP.',
}


def plan_description():
    return dict(version=VERSION,tasks=list(TASKS),screen=dict(arms=screen_arms(),seeds=SCREEN_SEEDS,trajectories=160),
        practical=dict(arms_by_writer={w:practical_arms(w) for w in ('diag','full')},seeds=FRESH_SEEDS,
            condition='Exactly one writer selected by fixed Stage I gates;from-start practical test',trajectories=96),
        minimum_trajectories=160,maximum_trajectories=256,
        smoke=dict(screen_seeds=SMOKE_SCREEN,fresh_seeds=SMOKE_FRESH,
            trajectories=64,interpretation='40 screening+24 practical: BOTH writer paths forced, shared ordinary references;engineering only'),
        prior_seed_audit=SEED_AUDIT,policy=POLICY)


def hash_gate(index,task,seeds,arms,field='a_boundary_hash'):
    rows=[]
    for s in seeds:
        vals=[index.get((task,a,s),{}).get(field) for a in arms]
        rows.append(dict(seed=s,status='incomplete' if not all(vals) else
            'pass' if len(set(vals))==1 else 'fail',hashes=dict(zip(arms,vals))))
    return dict(status=_v29_conjunction(x['status'] for x in rows),per_seed=rows)


def endpoint(r,key,kind='terminal'):
    return r[{'terminal':'terminal','boundary':'a_boundary','return':'return_summary'}[kind]]['hard'][key]


def compare(index,task,seeds,a,b,key,confidence,kind='terminal'):
    return _v29_eval_stat(index,task,seeds,(a,b),
        lambda x,y:endpoint(x,key,kind)-endpoint(y,key,kind),confidence)


def revision_gates(index,task,seeds,a,b,confidence):
    g={k:_v29_lower_gate(compare(index,task,seeds,a,b,k,confidence,'return'),-.05)
       for k in ('revised_after_two','revised_after_six')}
    g['unchanged_preserved']=_v29_lower_gate(_v29_eval_stat(index,task,seeds,(a,),
        lambda r:endpoint(r,'post_return_unchanged','return')-endpoint(r,'pre_return_unchanged','return'),confidence),-.05)
    g['revision_above_chance']=_v29_lower_gate(_v29_eval_stat(index,task,seeds,(a,),
        lambda r:endpoint(r,'revised_after_six','return'),confidence),.25,True)
    return g


def levels(index,task,seeds,arms):
    keys=('old_exact','old_fresh','new_unseen','A_prequential','A_second',
          'domain_1_prequential','domain_1_second')
    return {a:{v:{k:_v29_level(index,task,a,seeds,k,.95,v) for k in keys} for v in VIEWS} for a in arms}


def screen_report(records,seeds):
    ix=_v29_index(records);names=[a['name'] for a in screen_arms()]
    tasks={}; candidates={}
    for task in TASKS:
        hashes=hash_gate(ix,task,seeds,names)
        stable=['S','D','P','DX','PX','DN','PN']
        code_ix={k:dict(v,B_code_route_hash=v.get('strict_write_ledger',{}).get('B_code_route_hash')) for k,v in ix.items()}
        codes=hash_gate(code_ix,task,seeds,stable,'B_code_route_hash')
        tasks[task]=dict(records=_v29_complete(ix,task,seeds,names),identical_A=hashes,
            stable_codes_routes=codes,levels=levels(ix,task,seeds,names),
            intervention_contrasts={a+'-'+b:{k:compare(ix,task,seeds,a,b,k,.9875) for k in
                ('old_exact','new_unseen','domain_1_prequential','domain_1_second')}
                for a,b in [('L','F'),('R','S'),('L','R'),('F','S')]})
        tasks[task]['status']=_v29_conjunction(tasks[task][k]['status'] for k in
            ('records','identical_A','stable_codes_routes'))
    for tag,writer in [('D','diag'),('P','full')]:
        ct={}
        for task in TASKS:
            g={'integrity':{'status':tasks[task]['status']}}
            s=compare(ix,task,seeds,tag,'S','old_exact',.9875)
            g['old_recall']=_v29_lower_gate(s,0.,True,.05) if task=='overlap' else _v29_lower_gate(s,-.03)
            for key in ('new_unseen','domain_1_prequential','domain_1_second'):
                g[key]=_v29_lower_gate(compare(ix,task,seeds,tag,'S',key,.9875),-.03)
            if task=='overlap':
                for comp in (tag+'N',tag+'X'):
                    g['old_vs_'+comp]=_v29_lower_gate(compare(ix,task,seeds,tag,comp,'old_exact',.9875),0,True)
                    g['transfer_vs_'+comp]=_v29_lower_gate(compare(ix,task,seeds,tag,comp,'new_unseen',.9875),-.03)
            g.update(revision_gates(ix,task,seeds,tag,'S',.9875))
            ct[task]=dict(gates=g,status=_v29_conjunction(v['status'] for v in g.values()))
        candidates[writer]=dict(tasks=ct,status=_v29_conjunction(v['status'] for v in ct.values()))
    selected=next((w for w in ('diag','full') if candidates[w]['status']=='pass'),None)
    return dict(tasks=tasks,candidates=candidates,selected_writer=selected,seeds=list(seeds),
        status='pass' if selected else _v29_conjunction(v['status'] for v in candidates.values()),
        interval=.9875,interpretation='Conjunction gates,diagonal preference;no size equivalence claim')


def practical_report(records,seeds,writer):
    ix=_v29_index(records);tag={'diag':'D','full':'P'}[writer]
    live,frozen=tag+'_live',tag+'_freeze';names=[a['name'] for a in practical_arms(writer)]
    tasks={}
    for task in TASKS:
        integrity={'records':_v29_complete(ix,task,seeds,names),
            'selected_A_hash':hash_gate(ix,task,seeds,[live,frozen]),
            'ordinary_A_hash':hash_gate(ix,task,seeds,['O_live','O_freeze'])}
        g={k:{'status':v['status']} for k,v in integrity.items()}
        old=compare(ix,task,seeds,live,'O_live','old_exact',.975)
        g['old_recall']=_v29_lower_gate(old,0,True,.05) if task=='overlap' else _v29_lower_gate(old,-.03)
        for key in ('new_unseen','domain_1_prequential','domain_1_second','A_prequential','A_second'):
            g[key]=_v29_lower_gate(compare(ix,task,seeds,live,'O_live',key,.975),-.03)
        g['A_acquisition']=_v29_lower_gate(compare(ix,task,seeds,live,'O_live','old_exact',.975,'boundary'),-.03)
        g.update(revision_gates(ix,task,seeds,live,'O_live',.975))
        rg={k:{'status':v['status']} for k,v in integrity.items()}
        rg['transfer_gain']=_v29_lower_gate(compare(ix,task,seeds,live,frozen,'new_unseen',.975),0,True,.05)
        for key in ('old_exact','domain_1_prequential','domain_1_second'):
            rg[key]=_v29_lower_gate(compare(ix,task,seeds,live,frozen,key,.975),-.05)
        rg['A_above_chance']=_v29_lower_gate(_v29_eval_stat(ix,task,seeds,(live,),
            lambda r:endpoint(r,'old_exact','boundary'),.975),.25,True)
        rg['A_retained_excess']=_v29_lower_gate(_v29_eval_stat(ix,task,seeds,(live,),
            lambda r:endpoint(r,'old_exact')-.25-.75*(endpoint(r,'old_exact','boundary')-.25),.975),0)
        tasks[task]=dict(integrity=integrity,bank_gates=g,representation_gates=rg,
            bank_status=_v29_conjunction(v['status'] for v in g.values()),
            representation_status=_v29_conjunction(v['status'] for v in rg.values()),
            levels=levels(ix,task,seeds,names))
    return dict(writer=writer,tasks=tasks,seeds=list(seeds),
        bank_status=_v29_conjunction(v['bank_status'] for v in tasks.values()),
        representation_status=_v29_conjunction(v['representation_status'] for v in tasks.values()),
        scope='Two synthetic single-shift conditions;no lifetime qualification',
        memory_scope='small diagonal candidate' if writer=='diag' else 'larger covariance reference;not a tiny local core')


def reporting_checks():
    names=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        names.append(name)
    seeds=(97001,97003,97007)
    check('missing_seed_blocks_interval',_v29_stat([.1,None,.2],seeds)['status']=='incomplete')
    zero=_v29_stat([0.,0.,0.],seeds,.9875)
    check('exact_zero_not_superiority',_v29_lower_gate(zero,0,True)['status']=='fail')
    check('noninferiority_includes_boundary',_v29_lower_gate(zero,0)['status']=='pass')
    check('uncertainty_not_equivalence',_v29_lower_gate(_v29_stat([-.2,0,.2],seeds))['status']=='inconclusive')
    def fake(t,s,a,old=.5,new=.8):
        m=dict(old_exact=old,new_unseen=new,old_fresh=.25,A_prequential=.8,A_second=.8,
               domain_1_prequential=.8,domain_1_second=.8)
        ret=dict(revised_after_two=.8,revised_after_six=.9,pre_return_unchanged=.6,post_return_unchanged=.6)
        return dict(seed=s,task=t,spec=a,terminal={v:dict(m) for v in VIEWS},
            a_boundary={v:dict(old_exact=.9) for v in VIEWS},
            a_boundary_hash='A'+str(s),return_summary={v:dict(ret) for v in VIEWS},
            strict_write_ledger={'B_code_route_hash':'same'+str(s)})
    records=[fake(t,s,a,.7 if a['name'] in ('D','P') else .5)
        for t in TASKS for a in screen_arms() for s in seeds]
    sr=screen_report(records,seeds)
    check('screen_all_gates_can_pass',all(v['status']=='pass' for v in sr['candidates'].values()))
    check('diagonal_preference',sr['selected_writer']=='diag')
    bad=copy.deepcopy(records)
    for r in bad:
        if r['spec']['name']=='D':r['terminal']['hard']['old_exact']=.4
    check('full_only_selection',screen_report(bad,seeds)['selected_writer']=='full')
    for r in bad:
        if r['spec']['name']=='P':r['terminal']['hard']['old_exact']=.4
    check('all_fail_stop',screen_report(bad,seeds)['selected_writer'] is None)
    check('missing_control_blocks_promotion',screen_report(records[:-1],seeds)['selected_writer'] is None)
    bad=copy.deepcopy(records);bad[0]['a_boundary_hash']='different'
    check('different_A_blocks_promotion',screen_report(bad,seeds)['selected_writer'] is None)
    bad=copy.deepcopy(records)
    next(r for r in bad if r['spec']['name']=='S')['strict_write_ledger']['B_code_route_hash']='different'
    check('different_codes_blocks_promotion',screen_report(bad,seeds)['selected_writer'] is None)
    bad=copy.deepcopy(records)
    for r in bad:
        if r['spec']['name'] in ('D','P'):r['terminal']['hard'].pop('domain_1_second')
    check('missing_second_blocks_promotion',screen_report(bad,seeds)['selected_writer'] is None)
    bad=copy.deepcopy(records)
    for r in bad:
        if r['spec']['name'] in ('D','P'):r['return_summary']['hard']['revised_after_two']=.1
    check('revision_failure_blocks_promotion',screen_report(bad,seeds)['selected_writer'] is None)
    rec=[]
    for t in TASKS:
        for a in practical_arms('diag'):
            for s in seeds:
                r=fake(t,s,a,.8 if a['writer']=='diag' else .55,
                    .85 if a['name']=='D_live' else .70 if a['name']=='D_freeze' else .8)
                rec.append(r)
    pr=practical_report(rec,seeds,'diag')
    check('practical_bank_can_pass',pr['bank_status']=='pass')
    check('representation_can_pass',pr['representation_status']=='pass')
    for r in rec:
        if r['spec']['name']=='D_freeze':r['terminal']['hard']['new_unseen']=.85
    pr=practical_report(rec,seeds,'diag')
    check('bank_success_separate_from_representation',pr['bank_status']=='pass' and pr['representation_status']!='pass')
    next(r for r in rec if r['spec']['name']=='D_live')['a_boundary']['hard']['old_exact']=.25
    check('chance_A_not_retained_knowledge',practical_report(rec,seeds,'diag')['representation_status']!='pass')
    return names


def startup_checks():
    names=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        names.append(name)
    check('10_screen_arms',len(screen_arms())==len({a['name'] for a in screen_arms()})==10)
    check('160_256_plan',plan_description()['minimum_trajectories']==160 and plan_description()['maximum_trajectories']==256)
    check('64_smoke_jobs',10*2*2+6*2*2==64)
    sets=[set(SCREEN_SEEDS),set(FRESH_SEEDS),set(SMOKE_SCREEN),set(SMOKE_FRESH),{96101}]
    check('disjoint_seed_sets',all(not a&b for i,a in enumerate(sets) for b in sets[i+1:]))
    check('prior_seed_audit',not SEED_AUDIT['planned_seed_collisions'])
    rg=rng(96101,'v30_unit_checks')
    P=np.eye(7);C=np.eye(7)
    for _ in range(30):
        z=rg.normal(size=7);z/=np.linalg.norm(z)
        q,cap=write_direction(z,P@z)
        check_once=abs(z@q-1)<1e-12 and .6*np.linalg.norm(q)<=np.linalg.norm(z/(z@z))+1e-12
        if not check_once:raise AssertionError('direction identities')
        q0=z/(z@z)
        if q@C@q>q0@C@q0+1e-10:raise AssertionError('regularized disturbance')
        v=P@z;P-=np.outer(v,v)/(1+z@v);C+=np.outer(z,z)
    check('Sherman_Morrison_matches_inverse',np.max(np.abs(P-np.linalg.inv(C)))<1e-12)
    names+=['unit_current_correction','unit_norm_cap','regularized_disturbance_no_worse']
    z=norm_rows(np.array([1.,.001]));u=np.diag([1e-10,1.])@z
    q,cap=write_direction(z,u)
    check('extreme_direction_is_capped',cap and abs(z@q-1)<1e-10)
    for writer in ('ordinary','diag','full','diag_shuffle','full_shuffle','diag_norm','full_norm'):
        b=DirectionBanks(96101,writer);b.freeze_router=True
        r=b.centers[0].copy()
        for n in range(12):
            z=np.zeros(96);z[rg.choice(96,24,replace=False)]=rg.normal(size=24);z=norm_rows(z)
            e=np.eye(4)[n%4]-z@b.w[0];q0=z/(z@z)
            before=b.w[0].copy();P0=b.P[0].copy() if b.P is not None else None
            h0=b.h[0].copy() if b.h is not None else None
            bank_id=b.learn(r,z,n%4)
            if bank_id!=0:raise AssertionError('routing unit test')
            delta=b.w[0]-before
            actual_alpha=float(np.dot(z@delta,e)/max(float(e@e),1e-30))
            if not writer.endswith('_norm') and abs(actual_alpha-.6)>1e-11:raise AssertionError('current fit')
            if writer.endswith('_norm') and not .6-1e-11<=actual_alpha<=1+1e-11:raise AssertionError('norm control range')
            if np.linalg.norm(delta)>np.linalg.norm(q0)*np.linalg.norm(e)+1e-10:raise AssertionError('step cap')
            if writer.startswith('diag') or writer=='ordinary':
                if np.any(delta[z==0]!=0):raise AssertionError('inactive rows changed')
            if P0 is not None:
                v=P0@z
                if np.max(np.abs(b.P[0]-(P0-np.outer(v,v)/(1+z@v))))>1e-12:raise AssertionError('permutation changed statistic')
            if h0 is not None and not np.allclose(b.h[0],h0+z*z,rtol=0,atol=1e-14):raise AssertionError('diagonal update')
        d=b.digest();b.scores(r,z,'hard');b.scores(r,z,'blended');bank_diagnostics(b,np.atleast_2d(z))
        check(writer+'_pure_evaluation',b.digest()==d)
        check(writer+'_correction',b.metrics[5]<1e-10)
    check('diagonal_bytes',DirectionBanks(96101,'diag').costs_v30()['functional_statistic_bytes']==6144)
    check('full_bytes',DirectionBanks(96101,'full').costs_v30()['functional_statistic_bytes']==589824)
    b=DirectionBanks(96101,'diag');d=b.w.copy();h=b.h.copy();b.learn(b.centers[0],np.zeros(96),0)
    check('zero_code_no_write_or_statistic',np.array_equal(b.w,d) and np.array_equal(b.h,h) and b.metrics[1]==1)
    # Deliberately anisotropic statistic makes a permutation observable.
    b=DirectionBanks(96101,'full_shuffle');b.P[0]=np.diag(np.linspace(.01,1,96));z=norm_rows(rg.normal(size=96))
    ix=b.permutations[0];expected,_=write_direction(z,b.P[0][np.ix_(ix,ix)]@z)
    qtrue,_=write_direction(z,b.P[0]@z)
    check('misassignment_does_not_cancel',np.linalg.norm(expected-qtrue)>1e-4)
    b.freeze_router=True;b.learn(b.centers[0],z,0)
    check('misassignment_actual_write',np.max(np.abs(b.last_delta-.6*np.outer(expected,np.eye(4)[0])))<1e-12)
    names.extend(reporting_checks())
    with tempfile.TemporaryDirectory(prefix='v30_checks_') as tmp:
        p=Path(tmp)/'target';p.mkdir();alias=Path(tmp)/'alias';alias.symlink_to(p,target_is_directory=True)
        check('resolved_path_alias',p.resolve()==alias.resolve())
        write_json(alias/'atomic.json',{'a':1})
        check('atomic_roundtrip',json.loads((p/'atomic.json').read_text())=={'a':1})
        rejected=False
        with RunLock(p):
            try:
                with RunLock(alias):pass
            except RuntimeError:rejected=True
        check('concurrent_writer_rejected',rejected)
    return dict(count=len(names),checks=names,scientific_seeds_executed=False)


class V30Runner:
    def __init__(self,out,smoke=False,stop_after=None):
        self.out=Path(out).resolve();self.out.mkdir(parents=True,exist_ok=True)
        self.smoke=bool(smoke);self.stop_after=stop_after
        self.source=Path(__file__).read_bytes();self.source_hash=hashlib.sha256(self.source).hexdigest()
        import scipy
        self.identity=dict(version=VERSION,source_sha256=self.source_hash,numpy=np.__version__,
            scipy=scipy.__version__,smoke=self.smoke,dimensions=dimensions(self.smoke),plan=plan_description())
        self.identity_hash=sha(self.identity)
        mp=self.out/'MANIFEST.json'
        if mp.exists():
            saved=json.loads(mp.read_text())
            if saved.get('identity_hash')!=self.identity_hash or sha(saved['identity'])!=self.identity_hash:
                raise ValueError('Resume identity mismatch: source/settings/dependencies changed. Use a new --out directory.')
        elif any((self.out/k).exists() for k in ('jobs','REPORT.json','STATUS.json')):
            raise ValueError('Existing results without a manifest;refusing to overwrite')
        else:
            write_json(mp,dict(identity=self.identity,identity_hash=self.identity_hash,
                python=sys.version,platform=platform.platform(),invocation=sys.argv))
        self.source_name='elm_fly_directional_memory_v30.py'
        dest=self.out/self.source_name
        if dest.exists() and dest.read_bytes()!=self.source:
            raise ValueError('Archived source differs from executing source')
        if not dest.exists():atomic(dest,self.source.decode())
        self.records=[];self.reports={};self.new=self.reused=0

    def job(self,stage,seed,task,arm):
        descriptor=dict(stage=stage,seed=int(seed),task=task,spec=arm)
        key=sha(descriptor);path=self.out/'jobs'/(key+'.json')
        if path.exists():
            saved=json.loads(path.read_text())
            if (saved.get('identity_hash')!=self.identity_hash or saved.get('job')!=clean(descriptor)
                    or saved.get('payload_sha256')!=sha(saved['payload'])):
                raise ValueError('Invalid cached trajectory: '+str(path))
            record=saved['payload']
            if any(record.get(k)!=v for k,v in descriptor.items()):
                raise ValueError('Cached payload does not match job descriptor')
            self.reused+=1
        else:
            if self.stop_after is not None and self.new>=self.stop_after:raise Paused()
            record=run_trial_v30(seed,task,arm,self.smoke)
            record.update(stage=stage,job_key=key)
            write_json(path,dict(identity_hash=self.identity_hash,job=descriptor,
                payload=record,payload_sha256=sha(record)))
            self.new+=1
            print(f'{self.new} new,{self.reused} resumed | {stage} {task} seed{seed} {arm["name"]}',flush=True)
        self.records.append(record)
        return record

    def save_decision(self,name,value):
        value=clean(value);path=self.out/(name+'.json')
        if path.exists() and json.loads(path.read_text())!=value:
            raise ValueError('Sealed decision changed on resume: '+name)
        write_json(path,value)

    def outputs(self,status):
        # Partial reports must never inherit an earlier completed verdict.
        write_json(self.out/'STATUS.json',dict(status=status,new=self.new,reused=self.reused,
            visited_completed=len(self.records),smoke=self.smoke,scientific_verdicts_enabled=not self.smoke,
            source_sha256=self.source_hash))
        reports=copy.deepcopy(self.reports)
        if status!='complete':reports['verdict']='incomplete'
        if self.smoke:reports.update(verdict='engineering_smoke_only',scientific_claims_enabled=False)
        write_json(self.out/'REPORT.json',reports)
        terminals=[];checkpoints=[];returns=[];diagnostics=[];costs=[]
        for r in self.records:
            base=dict(stage=r['stage'],seed=r['seed'],task=r['task'],arm=r['spec']['name'])
            for view,row in r['terminal'].items():
                terminals.append({**base,**row,'view':view,'endpoint':'strict_pre_return',
                    'learner_numeric_bytes':r['costs']['total_learner_numeric_bytes']})
            checkpoints.extend(dict(base,**row) for row in r['rows'])
            returns.extend({**base,**row,'view':v} for v,row in r['return_summary'].items())
            diagnostics.extend(dict(base,**row) for row in r['diagnostics'])
            costs.append({**base,**r['costs'],**r['numeric_memory'],
                'seconds':r['seconds'],'process_peak_rss_bytes':r['process_peak_rss_bytes'],
                'rss_scope':'process high-water mark,includes prior jobs;not per-learner peak',
                'strict_writer':r['writer_summary'],'strict_write_ledger':r['strict_write_ledger']})
        for name,rows in [('TERMINAL',terminals),('CHECKPOINTS',checkpoints),('RETURN',returns),
                          ('DIAGNOSTICS',diagnostics),('COSTS',costs)]:
            csv_file(self.out/(name+'.csv'),rows)
        text=['# V30 directional-memory results','',f'Status: {status}.',
            f'Completed trajectories visited: {len(self.records)}.',
            '**ENGINEERING ONLY; NO SCIENTIFIC VERDICT.**' if self.smoke else
            'Stage I is an identical-A controlled bank-write experiment;Stage II is fresh from-start learning.',
            '',f'Workflow verdict: {reports.get("verdict","incomplete")}.',
            'TERMINAL.csv is pre-return;RETURN.csv separately measures 36 new revision labels (12 in smoke).',
            'Hard-read paired seed comparisons are primary;blended is descriptive.',
            'REPORT.json contains every gate,interval,missing seed and selection.',
            'COSTS.csv separates functional statistics,shadow statistics,writer monitors and evaluator storage.',
            'Full covariance adds576KiB;diagonal adds6KiB. Full-only success is not a tiny local core.',
            'No lifetime or unlimited persistence claim is made.',
            'Completed job records contain source/settings identity and payload checksums.','']
        sr=reports.get('screen',{})
        if sr:
            text+=['## Screening','',f'Selected writer: {sr.get("selected_writer")}.','',
                '| Writer | Status |','|---|---|']
            for w,v in sr.get('candidates',{}).items():text.append(f'| {w} | {v["status"]} |')
        for w,p in reports.get('practical',{}).items():
            text+=['',f'## Practical {w}','',f'Bank improvement: {p["bank_status"]}.',
                f'Continued representation learning: {p["representation_status"]}.']
        atomic(self.out/'README.md','\n'.join(text)+'\n')
        archive=self.out.parent/(self.out.name+'_results.zip');tmp=archive.with_suffix('.zip.tmp')
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
            for f in sorted(self.out.rglob('*')):
                if f.is_file() and f.name!='.run.lock' and not f.name.endswith('.tmp'):
                    z.write(f,str(f.relative_to(self.out)))
        os.replace(tmp,archive)
        return archive


def workflow(run):
    ss=SMOKE_SCREEN if run.smoke else SCREEN_SEEDS
    fs=SMOKE_FRESH if run.smoke else FRESH_SEEDS
    records=[run.job('screen',s,t,a) for t in TASKS for a in screen_arms() for s in ss]
    sr=screen_report(records,ss)
    run.reports.update(screen=sr,verdict='screen_complete')
    writers=['diag','full'] if run.smoke else ([sr['selected_writer']] if sr['selected_writer'] else [])
    run.save_decision('SCREEN_DECISION',dict(selected_writer=sr['selected_writer'],execution_writers=writers,
        smoke_override=run.smoke,selection='diagonal if eligible;otherwise full;otherwise stop'))
    if not writers:
        run.reports.update(verdict='bank_direction_not_established',downstream='No writer met fixed gates;practical stage skipped')
        return
    practical=[];names=set();arms=[]
    for w in writers:
        for a in practical_arms(w):
            if a['name'] not in names:arms.append(a);names.add(a['name'])
    for t in TASKS:
        for a in arms:
            for s in fs:practical.append(run.job('practical',s,t,a))
    pr={w:practical_report(practical,fs,w) for w in writers}
    run.reports['practical']=pr
    if run.smoke:
        run.reports['verdict']='engineering_smoke_only'
    else:
        result=pr[writers[0]]
        run.reports['verdict']=('ready_for_future_multishift_test' if
            result['bank_status']==result['representation_status']=='pass' else
            'bank_improvement_only' if result['bank_status']=='pass' else 'practical_improvement_not_established')
        run.reports['memory_scope']=result['memory_scope']
    run.save_decision('PRACTICAL_DECISION',dict(verdict=run.reports['verdict'],
        outcomes={w:{k:v[k] for k in ('bank_status','representation_status','memory_scope')} for w,v in pr.items()},
        smoke=run.smoke))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--install-deps',action='store_true')
    parser.add_argument('--self-test',action='store_true',help='engineering invariants only;exit afterward')
    parser.add_argument('--smoke',action='store_true',help='64 tiny engineering trajectories;forces both practical writers')
    parser.add_argument('--list-jobs',action='store_true',help='print fixed policy and roster;no experiments')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--stop-after',type=int,help='pause after this many new jobs;completed jobs preserved')
    a=parser.parse_args(argv)
    if a.stop_after is not None and a.stop_after<0:parser.error('--stop-after must be nonnegative')
    if a.out and a.resume and a.out.resolve()!=a.resume.resolve():parser.error('--out and --resume disagree')
    if a.resume:
        mp=a.resume/'MANIFEST.json'
        if not mp.exists():parser.error('--resume requires an existing V30 manifest')
        saved=json.loads(mp.read_text())
        if not str(saved.get('identity',{}).get('version','')).startswith('30.'):
            parser.error('Not a V30 results directory')
        if a.smoke and not saved['identity']['smoke']:parser.error('Cannot resume a scientific run as smoke')
        a.smoke=bool(saved['identity']['smoke'])
    if a.list_jobs:
        print(json.dumps(plan_description(),indent=2));return 0
    dependencies(a.install_deps)
    child=keep_awake()
    try:
        checks=startup_checks()
        print(f'V30 startup checks: {checks["count"]} passed.',flush=True)
        if a.self_test:return 0
        out=a.resume or a.out or Path('v30_smoke' if a.smoke else 'v30_directional_memory_results')
        with RunLock(out):
            run=V30Runner(out,a.smoke,a.stop_after)
            write_json(run.out/'PREFLIGHT.json',checks)
            write_json(run.out/'PLAN.json',plan_description())
            write_json(run.out/'POLICY.json',POLICY)
            exit_code=0
            try:
                workflow(run);status='complete'
                if (run.out/'ERROR.txt').exists():(run.out/'ERROR.txt').unlink()
            except (Paused,KeyboardInterrupt):status='paused;completed trajectories preserved'
            except Exception as e:
                import traceback
                status='failed: '+type(e).__name__+': '+str(e)
                atomic(run.out/'ERROR.txt',traceback.format_exc());print(status,file=sys.stderr,flush=True);exit_code=1
            archive=run.outputs(status)
            print(f'{status}\nResults: {archive}\nResume: python3 {shlex.quote(Path(__file__).name)} --resume {shlex.quote(str(run.out))}',flush=True)
            return exit_code
    finally:
        if child is not None and child.poll() is None:child.terminate()


if __name__=='__main__':
    raise SystemExit(main())
