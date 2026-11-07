"""Locked path-integration engineering exercise on the v630 PFN/hDeltaB/PFR route."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np
import torch

import fit_gains as first
import fit_gains_followup as followup
from qualify import rewire
from qualify_dynamics import projection

LOCK = 'PI_LOCK.json'
DT = .05
POINTS = (40, 80, 120, 160)
GROUPS = ('EPG->PFNd', 'EPG->PFNv', 'PFNd->hDeltaB', 'PFNv->hDeltaB',
          'hDeltaB->hDeltaB', 'hDeltaB->PFR')
ADAPTERS = ('forward_velocity_input', 'hDeltaB_retention', 'displacement_scale', 'displacement_phase_rad')
NAMES = GROUPS + ADAPTERS
RANGES = np.array([
    (0., 3., 1.), (0., 3., 1.), (0., 3., 1.), (0., 3., 1.),
    (0., .025, .002), (0., 3., 1.), (.1, 3., 1.), (.96, 1., .998),
    (1., 180., 35.), (-np.pi, np.pi, 0.),
])
HAND = [1., 1., 1., 1., 0., 1., 1., 1., 35., 0.]
TRAIN_SEED, MAIN_SEED, RETURN_SEED = 40129, 202611071, 202611072
SOURCES = ('https://www.nature.com/articles/s41586-021-04067-0',
           'https://pmc.ncbi.nlm.nih.gov/articles/PMC10759448/')


def digest(name):
    return sha256((first.ROOT / name).read_bytes()).hexdigest()


def headings(rng):
    return first.PANEL + (2 * np.pi / 16) * rng.random(16)


def panel(seed, repetitions, returning):
    rng = np.random.default_rng(seed)
    panels = []
    for _ in range(repetitions):
        starts = headings(rng)
        if returning:
            start = np.tile(starts, 3)
            yaw = np.zeros((48, 160))
            directions = np.r_[np.ones(16), -np.ones(16), rng.permutation(np.repeat([-1., 1.], 8))]
            yaw[:, 50:110] = directions[:, None] * np.pi / 60
            speed = np.zeros_like(yaw)
            speed[:, :50] = rng.uniform(.5, 1.5, (48, 1))
            speed[:, 110:] = speed[:, :50]
        else:
            start = np.tile(starts, 6)
            yaw = np.zeros((96, 160))
            legacy_yaw = first.sealed_panels()['compass'][1][16:]
            yaw[:, 40:80] = legacy_yaw
            speed = np.repeat(rng.uniform(.5, 1.5, (96, 2)), 80, axis=1)
        panels.append((start, yaw, speed))
    return tuple(np.concatenate(arrays) for arrays in zip(*panels))


def datasets():
    train_main = panel(TRAIN_SEED, 3, False)
    train_return = panel(TRAIN_SEED + 1, 2, True)
    train = tuple(np.concatenate((a, b)) for a, b in zip(train_main, train_return))
    main = panel(MAIN_SEED, 1, False)
    returning = panel(RETURN_SEED, 1, True)
    old = first.load('FIT_OFFICIAL_LOCK.json')['panels']['arrays']
    earlier = [first.train_panel()[0], *(p[0] for p in first.sealed_panels().values()),
               *(np.array(r['values'][0]) for r in old.values())]
    seen = np.concatenate([train[0], *earlier])
    for new in (main, returning):
        assert np.min(abs(np.angle(np.exp(1j * (new[0][:, None] - seen))))) > 1e-6
        seen = np.r_[seen, new[0]]
    return dict(train=train, main=main, returning=returning)


def routes(graph, model):
    labels, sides = graph['labels'], graph['sides']
    masks = {name: labels == name for name in ('PFNd', 'PFNv', 'hDeltaB', 'PFR')}
    masks['EPG'] = model.epg
    u, v, usable, w, sign = (graph[k] for k in ('u', 'v', 'usable', 'w', 'sign'))
    selected = np.zeros(len(u), dtype=bool)
    for group in GROUPS:
        a, b = group.split('->')
        selected |= masks[a][u] & masks[b][v]
    selected &= usable
    u, v, w, sign = u[selected], v[selected], w[selected], sign[selected]
    assert np.all(sign == np.sign(graph['signed_raw'][selected]))
    labels_side = np.char.add(np.char.add(labels, '/'), sides)
    control, swaps = rewire(u, v, sign, labels_side)
    assert np.array_equal(labels_side[v], labels_side[control])
    coords = {row['root_id']: row for row in first.load('COMPARTMENT_MAP.json')['cells']}
    x = np.array([coords[k]['FB']['centroid_um'][0] for k in graph['ids'][masks['PFR']]])
    phase = model.fb_phase_slope * x + model.fb_phase_offset
    observer = np.linalg.pinv(np.column_stack((np.ones(len(x)), np.cos(phase), np.sin(phase))))[1:].T
    matrices = {}
    for arm, target in (('native', v), ('rewired', control)):
        weights = w * sign
        raw = {}
        for group in GROUPS:
            a, b = group.split('->')
            raw[group] = projection(u, target, weights, masks[a], masks[b], normalize=False)
        # As in the prior velocity probe, the PFN branches share the same target denominator.
        mass = sum(abs(raw[key]).sum(axis=0) for key in GROUPS[2:4]).clip(1e-12)
        for key, m in raw.items():
            scale = mass if key in GROUPS[2:4] else abs(m).sum(axis=0).clip(1e-12)
            raw[key] = first.tensor(m / scale)
        matrices[arm] = raw
    facts = dict(
        nodes={k: int(m.sum()) for k, m in masks.items()},
        signed_edges={group: int((masks[a][u] & masks[b][v]).sum())
                      for group in GROUPS for a, b in [group.split('->')]},
        signed_edge_count=len(u), rewire_swaps=swaps, changed_edges=int((control != v).sum()),
        rewired_target_sha256=first.hash_arrays((control,)),
        native_target_sha256=first.hash_arrays((v,)), signed_degree_preserved=True)
    # Forward projection on both body axes of each type. PFNv's negative values are
    # deviations below tonic firing, not negative rates or inhibitory synapses.
    axis_gain = (float(np.cos(np.pi / 4)), float(np.cos(3 * np.pi / 4)))
    return matrices, first.tensor(observer), axis_gain, facts


def make_lock(graph, model):
    prior = first.load('FIT_OFFICIAL_LOCK.json')
    official = first.load('FIT_OFFICIAL_RESULTS.json')
    assert digest('FIT_OFFICIAL_LOCK.json') == official['lock_sha256']
    assert official['all_seeds_passed']
    _, _, _, facts = routes(graph, model)
    return dict(
        scope='Three-cycle engineering exercise, frozen native compass for both PI wirings',
        sources=SOURCES,
        files={n: digest(n) for n in (*first.INPUTS, 'fit_gains_followup.py',
                                     'FIT_OFFICIAL_LOCK.json', 'FIT_OFFICIAL_RESULTS.json',
                                     'path_integration.py')},
        frozen_compass=dict(order=prior['frozen_parameters']['order'],
                            native_by_seed={s: r['native'] for s, r in prior['frozen_parameters']['fit2_vectors'].items()},
                            official_panel_hashes={k: v['sha256'] for k, v in prior['panels']['arrays'].items()}),
        parameters=[dict(name=n, count=1, min=float(lo), max=float(hi), center=float(center), hand_set=value)
                    for n, (lo, hi, center), value in zip(NAMES, RANGES, HAND)],
        free_role_pair_gain_count=len(GROUPS), free_adapter_and_dynamics_count=len(ADAPTERS),
        free_parameter_count=len(NAMES), graph=facts,
        dynamics=dict(
            compass='One initial landmark, original frozen native EPG state update with scripted yaw; no PI feedback',
            PFN='EPG signed projections with forward-component amplitude. PFNd receives positive and PFNv negative '
                'tonic-subtracted modulation using the original +/-45 and +/-135 degree body axes; no sideslip.',
            hDeltaB='Signed deviation state. Next = retention * current + mix * (current @ normalized recurrence '
                    '- current) + dt * incoming PFN velocity drive. Reset to zero at episode start.',
            PFR='Only the hDeltaB to PFR signed projection. Use the previous FB geometry first harmonic, '
                'a common trainable phase offset and a positive scale. No cumulative state in the decoder.',
            scope_limit='The temporal state update is an engineering assumption. The published pathway demonstrates '
                        'instantaneous travel direction and motivates the cell types.'),
        panels=dict(
            hashes={k: first.hash_arrays(v) for k, v in datasets().items()},
            generation='PCG64, independently sample one heading per equal-width sector; reuse across '
                       'yaw conditions within each draw. Score yaw before translation in every step.',
            main='96 episodes. Six legacy dark-turn schedules at steps 41-80; zero yaw elsewhere. '
                 'Independent Uniform(.5,1.5) forward speed in body lengths/s in each 80-step half.',
            returning='48 episodes. Outbound 50 steps, +/-pi over 60 steps in place, 50 steps back at '
                      'same Uniform(.5,1.5) speed. True final displacement is zero up to roundoff.',
            training='288 main episodes plus 96 returns with independent headings and speeds from the same recipes.',
            note='Prior navigation probes do not specify a translational speed distribution. The declared uniform '
                 'speed law is shared by training and both evaluation panels.',
            train_seeds=[TRAIN_SEED, TRAIN_SEED+1], main_seed=MAIN_SEED, return_seed=RETURN_SEED,
            dt_seconds=DT, checkpoints=POINTS, episodes=dict(train=384, main=96, returning=48)),
        training=dict(optimizer='Adam', updates=followup.UPDATES, batch=first.BATCH, initial_lr=first.LR,
                      lr_halving_interval=followup.HALF_LIFE, curve_interval=followup.INTERVAL,
                      permutation_seed=41053, seeds=first.SEEDS, gradient_clip=5.,
                      init='Common centers in logit coordinates plus iid N(0,.12) using the paired fitting seed.',
                      loss='Mean squared vector norm error at 40/80/120/160 plus mean squared wrapped compass '
                           'error across all 160 steps. Frozen compass term is cached and parameter-independent.',
                      evaluation='Final iterate only; both panels score after all six fits are saved.'),
        gates=dict(main_native_angle_count_min=72, within_angle_deg=20., magnitude_ratio_bounds=[.75, 1.25],
                   native_return_median_fraction_strictly_below=.3, paired_angle_count_lead_min=30,
                   compass_max_regression_deg=3.,
                   joint_count='Also report count within both angle and magnitude limits and its rewired lead.',
                   all_seeds_required=True),
        contributions='Report near-zero gains below .01 and RMS of each PFN branch drive at the sealed main '
                      'episodes, plus removal of hDeltaB recurrence and each PFN branch using the same locked data '
                      'as a descriptive dependence check. No refitting or gate changes.',
        cycles=['Wire/sign audit, zero-velocity and exact-return geometry checks, frozen compass replay',
                'Blocked recurrence parity, finite-difference gradients and deterministic Adam replay on training data',
                'Six matched locked fits, then sealed evaluation and fixed-gain comparison'])


def params(z):
    lo, hi, _ = first.tensor(RANGES).T
    return lo + (hi - lo) * z.sigmoid()


def init(seed):
    lo, hi, center = RANGES.T
    x = (center - lo) / (hi - lo)
    return first.tensor(np.log(x / (1 - x)) + np.random.default_rng(seed).normal(0., .12, len(x)))


@torch.no_grad()
def cache(loop, compass, data):
    start, yaw, speed = map(first.tensor, data)
    # The model sees only the cue, yaw and speed. The world trajectory is a label.
    e = compass[11] * first.phase_bump(loop.phase, start)
    truth_heading = start[:, None] + yaw.cumsum(1)
    translation = DT * speed[..., None] * torch.stack((truth_heading.cos(), truth_heading.sin()), dim=-1)
    truth = translation.cumsum(1)[:, np.array(POINTS) - 1]
    state, error = [], []
    for step in range(160):
        e = loop.step(e, yaw[:, step], compass)
        state.append(e)
        error.append(first.wrap(first.decode(e @ loop.observer) - truth_heading[:, step]).square())
    return torch.stack(state, dim=1), speed, truth, torch.stack(error, dim=1).mean(1)


class Integrator:
    def __init__(self, matrices, observer, axis_gain):
        self.in_d, self.in_v, self.out_d, self.out_v, self.recurrent, self.out = [
            matrices[name] for name in GROUPS]
        self.observer = observer
        self.axis_d, self.axis_v = axis_gain
        self.identity = torch.eye(len(self.recurrent), dtype=first.DTYPE)

    @torch.no_grad()
    def pfns(self, states, speed):
        return ((states @ self.in_d) * (self.axis_d * speed[..., None]),
                (states @ self.in_v) * (self.axis_v * speed[..., None]))

    def inputs(self, features, p):
        d, v = features
        return ((d @ self.out_d) * (p[0] * p[2] * p[6]),
                (v @ self.out_v) * (p[1] * p[3] * p[6]))

    def predict(self, features, p, explicit=False):
        d, v = self.inputs(features, p)
        drive = DT * (d + v)
        a = p[7] * self.identity + p[4] * (self.recurrent - self.identity)
        if explicit:
            state = torch.zeros_like(drive[:, 0])
            output = []
            for t in range(160):
                state = state @ a + drive[:, t]
                if t + 1 in POINTS:
                    output.append(state)
            checkpoints = torch.stack(output, dim=1)
        else:
            # Exact state at four checkpoints of the 160-step recurrence.
            powers = [self.identity]
            for _ in range(39):
                powers.append(powers[-1] @ a)
            increment = torch.einsum('bkti,tij->bkj', drive.reshape(-1, 4, 40, len(a)).flip(2),
                                     torch.stack(powers))
            step = powers[-1] @ a
            state = increment[:, 0]
            states = [state]
            for part in increment.unbind(dim=1)[1:]:
                state = state @ step + part
                states.append(state)
            checkpoints = torch.stack(states, dim=1)
        xy = checkpoints @ (self.out @ self.observer) * (p[5] * p[8])
        c, s = p[9].cos(), p[9].sin()
        rotation = torch.stack((torch.stack((c, s)), torch.stack((-s, c))))
        return xy @ rotation


def get_model(graph):
    return first.Interface(graph, *(first.load(name) for name in first.FILES[1:]))


def get_arm(graph, model):
    matrices, observer, axis, _ = routes(graph, model)
    return {arm: Integrator(m, observer, axis) for arm, m in matrices.items()}


def get_compass(lock, seed):
    return first.tensor(lock['frozen_compass']['native_by_seed'][str(seed)])


@torch.no_grad()
def training_features(net, loop, compass, data):
    e, speed, truth, compass_loss = cache(loop, compass, data)
    return (*net.pfns(e, speed), truth, compass_loss)


def loss(net, batch, z):
    d, v, truth, compass_loss = batch
    vector_loss = (net.predict((d, v), params(z)) - truth).square().sum(-1).mean()
    return vector_loss, compass_loss.mean()


def fit(net, data, seed, updates, arm):
    z = torch.nn.Parameter(init(seed))
    opt = torch.optim.Adam([z], lr=first.LR)
    rng = np.random.default_rng(41053)
    curve = []

    @torch.no_grad()
    def checkpoint(step, lr, norm):
        vector, compass = loss(net, data, z)
        curve.append(dict(update=step, vector_mse=vector.item(), compass_mse_rad2=compass.item(),
                          lr=lr, preceding_batch_grad_norm=norm))

    checkpoint(0, first.LR, None)
    for i in range(1, updates + 1):
        if (i - 1) % 8 == 0:
            order = rng.permutation(len(data[0])).reshape(8, first.BATCH)
        batch = tuple(x[order[(i - 1) % 8]] for x in data)
        lr = followup.rate(i)
        opt.param_groups[0]['lr'] = lr
        opt.zero_grad()
        vector, compass = loss(net, batch, z)
        (vector + compass).backward()
        norm = torch.nn.utils.clip_grad_norm_([z], 5., error_if_nonfinite=True)
        opt.step()
        if i % followup.INTERVAL == 0 or i == updates:
            checkpoint(i, lr, norm.item())
            if updates == followup.UPDATES and i % 512 == 0:
                print(seed, arm, i, round(curve[-1]['vector_mse'], 5), flush=True)
    return params(z.detach()).tolist(), curve


def official_compass(model, lock):
    original = first.load('FIT_OFFICIAL_LOCK.json')
    panels = {k: tuple(np.array(a) for a in r['values']) for k, r in original['panels']['arrays'].items()}
    assert all(first.hash_arrays(p) == lock['frozen_compass']['official_panel_hashes'][k]
               for k, p in panels.items())
    baseline = first.load('FIT_OFFICIAL_RESULTS.json')['fitted']
    loop = first.Loop(model, 'native')
    out = {}
    keys = ('stationary_median_deg', 'dark_turn_deg', 'combined_compass_deg')
    for seed in first.SEEDS:
        scores = first.score(loop, get_compass(lock, seed), panels)
        reference = baseline[str(seed)]['scores']['native']
        deltas = {k: scores[k] - reference[k] for k in keys}
        assert np.allclose(list(scores.values()), list(reference.values()), atol=1e-7, rtol=0)
        out[str(seed)] = dict(scores=scores, heading_median_deltas_deg=deltas,
                              passed=all(x <= 3. for x in deltas.values()))
    return out


def cycle1(net, loop, lock, data):
    # Check on training recipes only. Scored holdout is untouched.
    returns = tuple(x[-96:] for x in data)
    cache_return = cache(loop, get_compass(lock, first.SEEDS[0]), returns)
    assert torch.linalg.vector_norm(cache_return[2][:, -1], dim=1).max() < 1e-12
    e, speed, _, _ = cache_return
    f = net.pfns(e[:2], torch.zeros_like(speed[:2]))
    assert torch.count_nonzero(net.predict(f, first.tensor(HAND))) == 0
    assert all(lock['graph']['signed_edges'].values())
    return dict(max_true_return_residual_bl=float(torch.linalg.vector_norm(cache_return[2][:, -1], dim=1).max()),
                zero_motion_exact=True, signed_edges=lock['graph']['signed_edges'])


def cycle2(net, data):
    seed = first.SEEDS[0]
    z = init(seed).requires_grad_()
    batch = tuple(x[:8] for x in data)
    p = params(z)
    a = net.predict(batch[:2], p)
    b = net.predict(batch[:2], p, explicit=True)
    max_state_diff = float((a - b).abs().max())
    assert torch.allclose(a, b, atol=1e-10, rtol=1e-10)
    gradient = torch.autograd.grad(sum(loss(net, batch, z)), z)[0]
    with torch.no_grad():
        eps = 1e-6
        numbers = torch.stack([(sum(loss(net, batch, z + eps * d)) -
                                sum(loss(net, batch, z - eps * d))) / (2 * eps)
                               for d in torch.eye(len(z), dtype=first.DTYPE)])
    assert torch.allclose(gradient, numbers, atol=1e-7, rtol=.003)
    assert torch.all(gradient.abs() > 1e-9)
    assert fit(net, data, seed, 2, '') == fit(net, data, seed, 2, '')
    return dict(max_recurrence_difference_bl=max_state_diff,
                max_gradient_difference=float((gradient - numbers).abs().max()), deterministic_replay=True)


def fit_worker(seed, arm, dest):
    graph = first.read_graph()
    model = get_model(graph)
    lock = first.load(LOCK)
    assert lock == json.loads(json.dumps(make_lock(graph, model)))
    net = get_arm(graph, model)[arm]
    train = training_features(net, first.Loop(model, 'native'), get_compass(lock, seed), datasets()['train'])
    start = time.monotonic()
    p, curve = fit(net, train, seed, followup.UPDATES, arm)
    dest.write_text(json.dumps(dict(seed=seed, arm=arm, parameters=dict(zip(NAMES, p)), curve=curve,
                                    seconds=time.monotonic()-start), allow_nan=False))


def fit_all(jobs):
    pairs = [(seed, arm) for seed in first.SEEDS for arm in first.ARMS]
    with TemporaryDirectory(prefix='minifly-pi-') as work, ThreadPoolExecutor(max_workers=jobs) as pool:
        def call(pair):
            seed, arm = pair
            dest = Path(work) / f'{seed}-{arm}.json'
            subprocess.run([sys.executable, __file__, '--worker', str(seed), arm, str(dest)], check=True)
            return json.loads(dest.read_text())
        return list(pool.map(call, pairs))


@torch.no_grad()
def measure(net, data, p, outbound, lock):
    pred = net.predict(data[:2], p)
    truth = data[2]
    absolute = torch.linalg.vector_norm(pred - truth, dim=-1).numpy()
    truth_mag = torch.linalg.vector_norm(truth, dim=-1).numpy()
    magnitude = torch.linalg.vector_norm(pred, dim=-1).numpy()
    angles = torch.rad2deg(first.wrap(first.decode(pred.reshape(-1, 2)) -
                                      first.decode(truth.reshape(-1, 2))).abs()).reshape(-1, 4).numpy()
    curve = [dict(step=t, vector_rmse_bl=float(np.sqrt((absolute[:, i] ** 2).mean())),
                  median_vector_error_bl=float(np.median(absolute[:, i])),
                  median_predicted_magnitude_bl=float(np.median(magnitude[:, i])),
                  median_true_magnitude_bl=float(np.median(truth_mag[:, i])))
             for i, t in enumerate(POINTS)]
    if outbound is not None:
        ratio = magnitude[:, -1] / outbound
        return dict(median_residual_over_outbound=float(np.median(ratio)),
                    max_true_residual_bl=float(truth_mag[:, -1].max()),
                    curve=curve, episode_residual_over_outbound=ratio.tolist())
    assert np.all(truth_mag > 1e-9)
    ratio = magnitude / truth_mag
    bounds = lock['gates']['magnitude_ratio_bounds']
    angle_pass = angles <= lock['gates']['within_angle_deg']
    magnitude_pass = (ratio >= bounds[0]) & (ratio <= bounds[1])
    for i, row in enumerate(curve):
        row.update(median_angle_deg=float(np.median(angles[:, i])), median_magnitude_ratio=float(np.median(ratio[:, i])))
    return dict(angle_count=int(angle_pass[:, -1].sum()), joint_count=int((angle_pass & magnitude_pass)[:, -1].sum()),
                median_angle_deg=curve[-1]['median_angle_deg'], median_magnitude_ratio=curve[-1]['median_magnitude_ratio'],
                curve=curve, episode_angle_deg=angles[:, -1].tolist(), episode_magnitude_ratio=ratio[:, -1].tolist())


def gate(native, rewired, compass, lock):
    g = lock['gates']
    main, control = native['main'], rewired['main']
    lead = main['angle_count'] - control['angle_count']
    angle = main['angle_count'] >= g['main_native_angle_count_min']
    lo, hi = g['magnitude_ratio_bounds']
    magnitude = lo <= main['median_magnitude_ratio'] <= hi
    returning = native['returning']['median_residual_over_outbound'] < g['native_return_median_fraction_strictly_below']
    margin = lead >= g['paired_angle_count_lead_min']
    return dict(angle=angle, magnitude=magnitude, returning=returning, lead=lead, margin=margin,
                joint_lead=main['joint_count'] - control['joint_count'], compass=compass,
                heading_tracking_stop=angle and magnitude and not returning,
                passed=angle and magnitude and returning and margin and compass)


@torch.no_grad()
def score_all(nets, loop, lock, records, compass_scores, panels):
    assert len(records) == 6
    scores = {}
    contributions = {}
    for seed in first.SEEDS:
        c = get_compass(lock, seed)
        caches = {k: cache(loop, c, panels[k]) for k in ('main', 'returning')}
        outbound = DT * panels['returning'][2][:, :50].sum(1)
        s = scores[str(seed)] = {}
        for arm, net in nets.items():
            data = {k: (*net.pfns(e, speed), truth, loss) for k, (e, speed, truth, loss) in caches.items()}
            record = next(r for r in records if r['seed'] == seed and r['arm'] == arm)
            fitted = first.tensor([record['parameters'][k] for k in NAMES])
            for tag, p in (('hand_set', first.tensor(HAND)), ('fitted', fitted)):
                s[f'{arm}_{tag}'] = dict(main=measure(net, data['main'], p, None, lock),
                                        returning=measure(net, data['returning'], p, outbound, lock))
            if arm == 'native':
                d, v = net.inputs(data['main'][:2], fitted)
                use = contributions[str(seed)] = dict(
                    near_zero_gains=[n for n, x in zip(NAMES, fitted) if abs(x) < .01],
                    pfnd_drive_rms=float(d.square().mean().sqrt()),
                    pfnv_drive_rms=float(v.square().mean().sqrt()))
                for label, index in (('without_PFNd', 2), ('without_PFNv', 3), ('without_hDeltaB_recurrence', 4)):
                    p = fitted.clone()
                    p[index] = 0.
                    use[label] = measure(net, data['main'], p, None, lock)['angle_count']
        s['gate'] = gate(s['native_fitted'], s['rewired_fitted'], compass_scores[str(seed)]['passed'], lock)
    return scores, contributions


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--freeze', action='store_true')
    cli.add_argument('--jobs', type=int, default=1)
    cli.add_argument('--worker', nargs=3)
    args = cli.parse_args()
    torch.set_num_threads(1)
    if args.worker:
        seed, arm, dest = args.worker
        fit_worker(int(seed), arm, Path(dest))
        return
    graph = first.read_graph()
    model = get_model(graph)
    expected = make_lock(graph, model)
    if args.freeze:
        if (first.ROOT / LOCK).exists():
            raise FileExistsError(LOCK)
        first.save(LOCK, expected)
        return
    lock = first.load(LOCK)
    assert lock == json.loads(json.dumps(expected)), 'Lock mismatch'
    panels = datasets()
    nets = get_arm(graph, model)
    loop = first.Loop(model, 'native')
    compass = official_compass(model, lock)
    cycles = {'cycle1': cycle1(nets['native'], loop, lock, panels['train'])}
    print('Cycle 1 passed. Original compass and steering scores reproduced.', flush=True)
    c = get_compass(lock, first.SEEDS[0])
    cycles['cycle2'] = {arm: cycle2(net, training_features(net, loop, c, panels['train']))
                        for arm, net in nets.items()}
    print('Cycle 2 passed. Starting six locked fits.', flush=True)
    records = fit_all(args.jobs)
    first.save('PI_CURVES.json', dict(lock_sha256=digest(LOCK), torch=str(torch.__version__),
                                      numpy=np.__version__, records=records))
    scores, contributions = score_all(nets, loop, lock, records, compass, panels)
    cycles['cycle3'] = dict(fits=len(records), all_seeds_passed=all(s['gate']['passed'] for s in scores.values()))
    first.save('PI_RESULTS.json', dict(lock_sha256=digest(LOCK), cycles=cycles, scores=scores,
                                       compass=compass, contributions=contributions))
    print('Cycle 3 complete. All seeds passed:', cycles['cycle3']['all_seeds_passed'], flush=True)


if __name__ == '__main__':
    main()
