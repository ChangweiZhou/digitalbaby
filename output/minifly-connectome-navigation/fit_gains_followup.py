"""Repeat the paired gain fit and trace failures on the existing steering panel."""
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

LOCK = 'FIT2_LOCK.json'
UPDATES = 2048
INTERVAL = 128
HALF_LIFE = 512
TAIL = 20
NAMES = first.GROUPS + first.ADAPTERS
NUMERIC_ATOL = 1e-7


def make_lock(graph):
    lock = first.make_lock(graph)
    lock['files'].update({name: sha256((first.ROOT / name).read_bytes()).hexdigest()
                          for name in ('FIT_LOCK.json', 'FIT_RESULTS.json', 'fit_gains_followup.py')})
    lock['training'].update(updates=UPDATES, lr_halving_interval=HALF_LIFE, curve_interval=INTERVAL,
                            evaluation='Final iterate only. Historical fits may be diagnosed before training. '
                                       'Score new fits after all six finish.')
    lock['loss'] = ('Mean squared wrapped compass error over all 80 post-update steps plus mean squared '
                    'wrapped goal error over post-update steps 61–80. Both terms average over episodes '
                    'and are in rad^2. Full closed-loop autodiff.')
    lock['scope'] = ('Follow-up engineering exercise on the five unchanged historical probes. No evaluation '
                     'panel scores enter fitting. Active circuit is EPG, PEN, Delta7, PFL3 and FC2.')
    lock['absent_role_groups'] = lock.pop('frozen_role_groups')
    lock['absent_role_gain_count'] = lock.pop('frozen_role_gain_count')
    del lock['frozen_role_gain_value']
    lock['gates'] = dict(native_clamped_minus_combined_at_most=10,
                         native_combined_compass_median_strictly_below_deg=15.,
                         min_paired_native_minus_rewired=30, each_seed_required=True,
                         native_dark_turn_regression_over_3_deg='Report as a trade-off, outside pass predicate',
                         steering_failure_table_required=True)
    lock['ceiling_diagnosis'] = dict(
        reference='Four-second clamped convergence count for the same arm and parameters, on the legacy '
                  'clamped panel. This is an empirical reference across two panels with different goal offsets.',
        stable='Tail absolute error range <=1 deg, final commanded yaw <=0.05 deg per step in magnitude',
        approaching='Late signed absolute-error slope < -1 deg/s and net absolute-error drop >1 deg',
        departing='Late signed absolute-error slope > 1 deg/s and net absolute-error rise >1 deg',
        slope='Least squares over post-update steps 61–80, 0.05 seconds per step',
        other='Unresolved when stable, approaching and departing criteria all fail',
        angle_bins='Original signed offsets -120, -65, -30, +30, +65, +120 degrees; report misses out of 16 each')
    lock['cycles'] = ['Check fixed wiring and legacy parity; diagnose archived fits on the original clamped panel',
                      'Verify tail loss and full-horizon gradients and deterministic optimizer replay on training episodes',
                      'Fit all pairs, save final iterates and curves, score the historical probes and diagnose steering misses']
    return lock


def verify_lock(graph):
    assert first.load(LOCK) == json.loads(json.dumps(make_lock(graph))), 'Lock mismatch'


def get_loops(graph):
    model = first.Interface(graph, *(first.load(name) for name in first.FILES[1:]))
    return model, {arm: first.Loop(model, arm) for arm in first.ARMS}


def rollout(loop, starts, goals, p):
    # Same single initial cue and simulator as Loop.run. Collect goal error at each step.
    e = p[11] * first.phase_bump(loop.phase, starts)
    position = starts
    heading = first.decode(e @ loop.observer)
    steering = loop.prepare_steering(goals, p)
    heading_errors, positions = [], []
    for _ in range(80):
        yaw = (p[14] * loop.command(heading, steering)).clamp(-.12, .12)
        position = position + yaw
        e = loop.step(e, yaw, p)
        heading = first.decode(e @ loop.observer)
        heading_errors.append(first.wrap(heading - position))
        positions.append(position)
    return torch.stack(heading_errors), first.wrap(goals - torch.stack(positions))


def loss(loop, panel, z):
    heading, goal = rollout(loop, *panel, first.physical(z))
    return heading.square().mean(), goal[-TAIL:].square().mean()


def training_data():
    panel = tuple(map(first.tensor, first.train_panel()))
    rng = np.random.default_rng(41053)
    indices = np.concatenate([rng.permutation(len(panel[0])).reshape(-1, first.BATCH)
                              for _ in range(UPDATES // 8)])
    data = [tuple(x[batch] for x in panel) for batch in indices]
    assert len(data) == UPDATES
    return panel, data


def rate(step):
    return first.LR * .5 ** ((step - 1) // HALF_LIFE)


def optimize(loop, seed, full, batches):
    z = torch.nn.Parameter(first.init(seed))
    adam = torch.optim.Adam([z], lr=first.LR)
    curve = []

    @torch.no_grad()
    def record(update, lr, norm):
        h, g = loss(loop, full, z)
        curve.append(dict(update=update, heading_rad2=h.item(), goal_rad2=g.item(),
                          learning_rate=lr, preceding_batch_grad_norm=norm))

    record(0, first.LR, None)
    for step, batch in enumerate(batches, 1):
        lr = rate(step)
        adam.param_groups[0]['lr'] = lr
        adam.zero_grad()
        h, g = loss(loop, batch, z)
        (h + g).backward()
        norm = torch.nn.utils.clip_grad_norm_([z], 5., error_if_nonfinite=True)
        adam.step()
        if step % INTERVAL == 0 or step == len(batches):
            record(step, lr, norm.item())
            if len(batches) == UPDATES and step % HALF_LIFE == 0:
                print(seed, step, curve[-1], flush=True)
    return first.physical(z.detach()).tolist(), curve


def cycle2(loops, full, batches):
    reference_full, reference_batches = first.batches()
    assert all(torch.equal(a, b) for a, b in zip(full, reference_full))
    assert all(torch.equal(a, b) for new, old in zip(batches, reference_batches) for a, b in zip(new, old))
    assert [rate(i) for i in (1, 512, 513, 1025, 1537, 2048)] == [.025, .025, .0125, .00625, .003125, .003125]
    batch = tuple(x[:12] for x in batches[0])
    result = {}
    for arm, loop in loops.items():
        z = first.init(first.SEEDS[0]).requires_grad_()
        p = first.physical(z)
        with torch.no_grad():
            h, g = rollout(loop, *batch, p)
            old_h, old_g = loop.run(*batch, p)
            assert torch.equal(h, old_h) and torch.equal(g[-1], old_g)
            assert not torch.isclose(g[-TAIL:].square().mean(), g[-1].square().mean())
        gradient = torch.autograd.grad(sum(loss(loop, batch, z)), z)[0]
        with torch.no_grad():
            eps = 1e-6
            numerical = torch.stack([(sum(loss(loop, batch, z + eps * d)) -
                                      sum(loss(loop, batch, z - eps * d))) / (2 * eps)
                                     for d in torch.eye(len(z), dtype=first.DTYPE)])
        assert torch.allclose(gradient, numerical, atol=2e-5, rtol=.003), (arm, gradient, numerical)
        assert optimize(loop, first.SEEDS[0], batch, batches[:2]) == optimize(loop, first.SEEDS[0], batch, batches[:2])
        result[arm] = dict(max_abs_gradient_difference=float((gradient - numerical).abs().max()),
                           legacy_trajectory_exact=True, training_order_prefix_exact=True, replay_exact=True)
    return result


@torch.no_grad()
def trace_steering(loop, p, panel):
    starts, goals = map(first.tensor, panel)
    steering = loop.prepare_steering(goals, p)
    state = starts
    initial = first.wrap(goals - state)
    trail = []
    for _ in range(80):
        state = state + (p[14] * loop.command(state, steering)).clamp(-.12, .12)
        trail.append(first.wrap(goals - state))
    tail = torch.rad2deg(torch.stack(trail[-TAIL:])).numpy()
    offsets = torch.rad2deg(initial).numpy()
    headings = torch.rad2deg(starts % (2 * torch.pi)).numpy()
    final_yaws = torch.rad2deg((p[14] * loop.command(state, steering)).clamp(-.12, .12)).numpy()
    x = np.arange(TAIL) * .05
    centered = x - x.mean()
    misses = []
    counts = Counter({k: 0 for k in (-120, -65, -30, 30, 65, 120)})
    for i in np.flatnonzero(np.abs(tail[-1]) > 20.):
        error = np.abs(tail[:, i])
        change = error[-1] - error[0]
        slope = float(centered @ error / (centered @ centered))
        span = float(np.ptp(error))
        yaw = float(final_yaws[i])
        if span <= 1. and abs(yaw) <= .05:
            behavior = 'stable wrong heading'
        elif slope < -1. and change < -1.:
            behavior = 'still approaching'
        elif slope > 1. and change > 1.:
            behavior = 'moving away'
        else:
            behavior = 'unresolved'
        offset = round(float(offsets[i]))
        counts[offset] += 1
        misses.append(dict(episode=int(i + 1), start_deg=float(headings[i]), initial_offset_deg=offset,
                           final_signed_error_deg=float(tail[-1, i]), final_abs_error_deg=float(error[-1]),
                           late_change_deg=float(change), late_slope_deg_s=slope, late_span_deg=span,
                           final_yaw_deg_step=yaw, behavior=behavior))
    return dict(converged=len(starts) - len(misses), miss_by_offset=dict(sorted(counts.items())),
                miss_by_behavior=dict(Counter(row['behavior'] for row in misses)), misses=misses)


def diagnosis(loops, parameters, scores, panel):
    out = {}
    for arm, p in parameters.items():
        out[arm] = trace_steering(loops[arm], p, panel)
        assert out[arm]['converged'] == scores[arm]['convergence']
    return out


def archived_diagnosis(loops, history):
    panel = first.sealed_panels()['steering']
    archived = first.load('FIT_RESULTS.json')
    hand = diagnosis(loops, {arm: first.old_params(history, arm) for arm in loops}, archived['hand_set'], panel)
    prior = {}
    for seed, arms in archived['fitted'].items():
        params = {}
        for arm in loops:
            row = arms[arm]
            p = row['role_gains'] | row['adapter_and_dynamics']
            params[arm] = first.tensor([p[name] for name in NAMES])
        prior[seed] = diagnosis(loops, params, {arm: arms[arm]['scores'] for arm in loops}, panel)
    return dict(hand_set=hand, fit1=prior)


def fit_worker(seed, arm, target):
    graph = first.read_graph()
    verify_lock(graph)
    _, loops = get_loops(graph)
    full, batches = training_data()
    start = time.monotonic()
    p, curve = optimize(loops[arm], seed, full, batches)
    row = dict(seed=seed, arm=arm, parameters=p, curve=curve, seconds=time.monotonic() - start)
    target.write_text(json.dumps(row, allow_nan=False))


def fit_all(jobs):
    pairs = [(seed, arm) for seed in first.SEEDS for arm in first.ARMS]
    with TemporaryDirectory(prefix='minifly-fit2-') as scratch, ThreadPoolExecutor(max_workers=jobs) as pool:
        def call(pair):
            seed, arm = pair
            dest = Path(scratch) / f'{seed}-{arm}.json'
            subprocess.run([sys.executable, __file__, '--worker', str(seed), arm, str(dest)], check=True)
            return json.loads(dest.read_text())

        return list(pool.map(call, pairs))


def score_all(loops, history, records, traces, cycles):
    panels = first.sealed_panels()
    hand = {arm: first.score(loop, first.old_params(history, arm), panels) for arm, loop in loops.items()}
    for arm in loops:
        first.check_historical(hand[arm], history, arm)
    fitted = {}
    traces['fit2'] = {}
    for record in records:
        arm = record['arm']
        p = first.tensor(record['parameters'])
        fitted.setdefault(str(record['seed']), {})[arm] = dict(
            scores=first.score(loops[arm], p, panels),
            role_gains=dict(zip(first.GROUPS, record['parameters'])),
            adapter_and_dynamics=dict(zip(first.ADAPTERS, record['parameters'][len(first.GROUPS):])))
    for seed, row in fitted.items():
        params = {r['arm']: first.tensor(r['parameters']) for r in records if str(r['seed']) == seed}
        traces['fit2'][seed] = diagnosis(loops, params, {arm: row[arm]['scores'] for arm in loops}, panels['steering'])
        native, rewired = (row[arm]['scores'] for arm in first.ARMS)
        gap = native['convergence'] - native['combined']
        margin = native['combined'] - rewired['combined']
        dark_delta = native['dark_turn_deg'] - hand['native']['dark_turn_deg']
        row['gate'] = dict(clamped_minus_combined=gap, native_minus_rewired=margin,
                           native_dark_turn_delta_deg=dark_delta, dark_turn_tradeoff=dark_delta > 3. + NUMERIC_ATOL,
                           gap_pass=gap <= 10, compass_pass=native['combined_compass_deg'] < 15.,
                           wiring_margin_pass=margin >= 30,
                           pass_all_required=gap <= 10 and native['combined_compass_deg'] < 15. and margin >= 30)
    cycles['cycle3'] = dict(fits_completed=len(records), historical_parity=True,
                            diagnosis_complete=True,
                            all_seeds_passed=all(row['gate']['pass_all_required'] for row in fitted.values()))
    first.save('FIT2_STEERING.json', traces)
    first.save('FIT2_RESULTS.json', dict(cycles=cycles, hand_set=hand, fitted=fitted))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--worker', nargs=3, metavar=('SEED', 'ARM', 'PATH'))
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.worker:
        seed, arm, dest = args.worker
        fit_worker(int(seed), arm, Path(dest))
        return
    graph = first.read_graph()
    if args.freeze:
        if (first.ROOT / LOCK).exists():
            raise FileExistsError(LOCK)
        first.save(LOCK, make_lock(graph))
        return
    verify_lock(graph)
    model, loops = get_loops(graph)
    history = first.load(first.OLD)
    cycles = {'cycle1': first.cycle1(model, graph['labels'], loops, history)}
    traces = archived_diagnosis(loops, history)
    print('Cycle 1 passed. Archived steering misses traced.', flush=True)
    cycles['cycle2'] = cycle2(loops, *training_data())
    print('Cycle 2 passed. Starting six locked fits.', flush=True)
    records = fit_all(args.jobs)
    first.save('FIT2_CURVES.json', dict(
        lock_sha256=sha256((first.ROOT / LOCK).read_bytes()).hexdigest(),
        torch=str(torch.__version__), numpy=np.__version__, parameter_order=NAMES, records=records))
    score_all(loops, history, records, traces, cycles)
    print('Cycle 3 complete.', flush=True)


if __name__ == '__main__':
    main()
