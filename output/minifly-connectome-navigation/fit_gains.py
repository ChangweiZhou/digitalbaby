"""Three-cycle engineering check of development-time gains on the existing loop."""
import argparse
from hashlib import sha256
import json
import platform
import time

import numpy as np
import torch

from qualify_dynamics import (Compass, FILES, Interface, N, PANEL, ROOT, Steering,
                              steering_panel, turn_panel)

ARMS = ('native', 'rewired')
SEEDS = (197, 281, 389)
UPDATES = 128
BATCH = 48
LR = .025
DTYPE = torch.float64
GROUPS = ('EPG->EPG', 'EPG->PEN', 'PEN->EPG', 'EPG->D7', 'D7->EPG',
          'EPG->PFL', 'D7->PFL', 'FC2->PFL')
ADAPTERS = ('rest_update', 'active_update', 'yaw_input', 'landmark_amplitude',
            'goal_phase_offset_rad', 'pfl3_threshold', 'motor_readout')
# Each row is lower bound, upper bound, common initialization center.
RANGES = np.array([
    (.1, 4., 1.55), (.1, 3., 1.), (.1, 3., 1.), (.1, 3., 1.),
    (.05, 2., .55), (.1, 3., 1.), (.01, 3., .75), (.1, 4., 1.4),
    (.005, .4, .055), (.1, .85, .45), (1., 70., 16.), (.2, 1., .95),
    (-np.pi, np.pi, .25), (-1., 2., .25), (.025, .45, .14),
])
INPUTS = (*FILES, 'qualify_dynamics.py', 'qualify_interfaces.py', 'qualify.py',
          'DYNAMICS_QUALIFICATION.json', 'fit_gains.py')
OLD = 'DYNAMICS_QUALIFICATION.json'


def tensor(values):
    return torch.as_tensor(values, dtype=DTYPE)


def phase_bump(phases, angles):
    return torch.exp(4 * (torch.cos(angles[..., None] - phases) - 1))


def wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def decode(xy):
    x, y = xy.unbind(dim=1)
    # Match atan2(0,0)=0 while avoiding an undefined derivative for collapsed bumps.
    return torch.atan2(y, torch.where((x == 0) & (y == 0), 1e-15, x))


def save(name, obj):
    path = ROOT / name
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load(name):
    return json.loads((ROOT / name).read_text())


def sealed_panels():
    turns = turn_panel(PANEL + .31, [np.zeros(N), np.full(N, .027), np.full(N, -.027),
        np.r_[np.full(17, .054), np.full(23, -.014)],
        np.r_[np.full(17, -.054), np.full(23, .014)],
        np.r_[np.zeros(12), np.full(28, .054)], np.r_[np.zeros(12), np.full(28, -.054)]])
    steering = steering_panel(PANEL + .19, np.deg2rad([-30, 30, -65, 65, -120, 120]))
    combined = steering_panel(PANEL + .47, np.deg2rad([-42, 42, -80, 80, -110, 110]))
    return dict(compass=turns, steering=steering, combined=combined)


def train_panel():
    return steering_panel(np.arange(64) * 2 * np.pi / 64 + .013,
                          np.deg2rad([-42, 42, -80, 80, -110, 110]))


def hash_arrays(panel):
    digest = sha256()
    for values in panel:
        array = np.asarray(values, dtype='<f8', order='C')
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def read_graph():
    with np.load(ROOT / FILES[0], allow_pickle=False) as source:
        return dict(source)


def extract_groups(graph):
    usable = graph['usable']
    roles = graph['roles']
    u, v = (graph[key][usable] for key in ('u', 'v'))
    names = sorted({a + '->' + b for a, b in zip(roles[u], roles[v])})
    assert len(names) == 106 and set(GROUPS) <= set(names)
    return names


def make_lock(graph):
    panels = sealed_panels()
    training = train_panel()
    for held in panels.values():
        assert np.min(abs(np.angle(np.exp(1j * (training[0][:, None] - held[0]))))) > 1e-6
    return dict(
        scope='Three-cycle engineering exercise on the historical five probes. '
              'The historical panels were previously reported. No held scores enter fitting.',
        files={name: sha256((ROOT / name).read_bytes()).hexdigest() for name in INPUTS},
        sealed_panels={name: hash_arrays(panel) for name, panel in panels.items()},
        training=dict(sha256=hash_arrays(training), episodes=384, steps_per_episode=80,
                      dt_seconds=.05, optimizer='Adam', lr=LR, updates=UPDATES, batch=BATCH,
                      permutation_seed=41053, gradient_clip_norm=5.,
                      init='Common centers in unconstrained logit coordinates plus iid N(0,.12) per seed. Same initialization across arms.',
                      seeds=SEEDS, evaluation='Final iterate only. All six fits before score.'),
        loss='Mean over episodes and all 80 post-update time steps of squared wrapped compass error, '
             'plus mean squared wrapped final goal error. Both terms in rad^2. Full closed-loop autodiff.',
        numerical='CPU float64, one Torch thread, fixed 0.12-rad yaw clamp, fixed normalized projections '
                  'and bump width. Continuous version of the historical steering calibration over the same 64 '
                  'uniform phases and 0.04 standard-deviation floor. Recompute from the current two path scales.',
        wiring='Historical Compass observed_contacts and Steering aggregate, native and deterministic '
               'subtype/side/signed-degree rewires. Historical Delta7 inhibitory sign override stays fixed. '
               'Existing adapters, neuron subsets and compass/steering projection normalization stay fixed.',
        free_parameter_count=len(RANGES), active_role_gain_count=len(GROUPS),
        adapter_and_dynamics_count=len(ADAPTERS), extracted_role_gain_count=106,
        frozen_role_gain_count=106 - len(GROUPS),
        frozen_role_gain_value=1.,
        parameters=[dict(name=name, count=1, lower=float(lo), upper=float(hi), init_center=float(center))
                    for name, (lo, hi, center) in zip(GROUPS + ADAPTERS, RANGES)],
        frozen_role_groups=sorted(set(extract_groups(graph)) - set(GROUPS)),
        gates=dict(native_combined_min_goals=72, native_combined_compass_median_strictly_below_deg=15.,
                   min_paired_native_minus_rewired=30, each_seed_required=True,
                   no_regression='Native-fitted versus native-hand-set on stationary count, dark-turn median '
                                 'and qualified count, clamped steering direction count and convergence count. '
                                 'Rewired probe deltas are reported.'),
        cycles=['Graph audit and hand-set parity on development fixtures only',
                'Full-horizon numerical gradient and deterministic replay on training episodes only',
                'Three identical-budget paired fits, then five historical probes and frozen gates'])


class Loop:
    def __init__(self, model, arm):
        self.compass = Compass(model, 'observed_contacts', arm)
        self.steering = Steering(model, arm)
        for name in ('local', 'enter', 'exit', 'inhib', 'side'):
            setattr(self, name, tensor(getattr(self.compass, name)))
        for name in ('head', 'delta', 'goal'):
            setattr(self, name, tensor(getattr(self.steering, name)))
        self.motor = tensor(self.steering.motor[0] - self.steering.motor[1])
        self.phase = tensor(model.phase)
        self.goal_phase = tensor(model.fc2_phase)
        self.observer = tensor(model.observer.T)
        phases = tensor(np.arange(64) * (2 * np.pi / 64))
        self.head_reference = phase_bump(self.phase, phases)
        goal_reference = phase_bump(self.goal_phase, phases) @ self.goal
        self.goal_mean = goal_reference.mean(0)
        self.goal_std = goal_reference.std(0, unbiased=False).clamp_min(.04)

    def step(self, e, yaw, p):
        relay = (p[1] * (e @ self.enter)).clamp_min(0.)
        modulation = (p[10] * yaw[:, None] * self.side).clamp(-1., 1.)
        rest = p[0] * (e @ self.local) + p[2] * (relay @ self.exit) + (p[3] * p[4]) * (e @ self.inhib)
        active = torch.tanh((rest + p[2] * ((relay * modulation) @ self.exit)).clamp_min(0.))
        rest = torch.tanh(rest.clamp_min(0.))
        return (e + p[8] * (rest - e) + p[9] * (active - rest)).clamp(0., 1.)

    def prepare_steering(self, goals, p):
        projection = p[5] * self.head + (p[3] * p[6]) * self.delta
        reference = self.head_reference @ projection
        mean = reference.mean(0)
        # Clamp the variance before sqrt, matching the old scale floor and retaining finite derivatives.
        scale = reference.var(0, unbiased=False).clamp_min(.04 ** 2).sqrt()
        goal_signal = (phase_bump(self.goal_phase, goals + p[12]) @ self.goal - self.goal_mean) / self.goal_std
        return projection / scale, p[7] * goal_signal - mean / scale - p[13]

    def command(self, heading, steering):
        projection, bias = steering
        signal = phase_bump(self.phase, heading) @ projection + bias
        return (torch.logaddexp(torch.zeros_like(signal), 3 * signal) / 3) @ self.motor

    def run(self, starts, goals, p):
        e = p[11] * phase_bump(self.phase, starts)
        position = starts
        heading = decode(e @ self.observer)
        steering = self.prepare_steering(goals, p)
        errors = []
        for _ in range(80):
            yaw = (p[14] * self.command(heading, steering)).clamp(-.12, .12)
            position = position + yaw
            e = self.step(e, yaw, p)
            heading = decode(e @ self.observer)
            errors.append(wrap(heading - position))
        return torch.stack(errors), wrap(goals - position)


def old_params(history, arm):
    compass = history['compass_separate_timescales']['observed_contacts'][arm]
    steer = history['steering'][arm]['params']
    local, inhib, feedback, yaw = compass['params']
    offset, delta, goal_gain, threshold = steer
    return tensor([local, 1., feedback, 1., inhib, 1., delta, goal_gain,
                   compass['rest_rate'], .45, yaw, 1., offset, threshold, .14])


def cycle1(model, labels, loops, history):
    result = {}
    for route in ('observed_contacts', 'aggregate'):
        u, v, w = model.variants[route]['']
        ru, rv, rw = model.variants[route]['_rewired']
        assert np.array_equal(u, ru) and np.array_equal(w, rw)
        assert np.array_equal(labels[v], labels[rv])
        assert np.array_equal(model.sides[v], model.sides[rv])
        for sign in (-1, 1):
            keep = np.sign(w) == sign
            assert np.array_equal(np.sort(v[keep]), np.sort(rv[keep]))
        result[route] = dict(edges=len(v), moved=int(np.count_nonzero(v != rv)),
                             signed_degrees_preserved=True)
    starts, goals = train_panel()
    starts, goals = starts[7:19], goals[7:19]
    for arm, loop in loops.items():
        p = old_params(history, arm)
        compass = history['compass_separate_timescales']['observed_contacts'][arm]
        steer = history['steering'][arm]['params']
        e = np.exp(4 * (np.cos(starts[:, None] - model.phase) - 1))
        position = starts.copy()
        expected = []
        for _ in range(80):
            x, y = (e @ model.observer.T).T
            command = loop.steering.command(np.arctan2(y, x), goals, steer)
            yaw = np.clip(.14 * command, -.12, .12)
            e = loop.compass.step(e, yaw, compass['params'], 'opponent', compass['rest_rate'])
            position += yaw
            x, y = (e @ model.observer.T).T
            expected.append(np.angle(np.exp(1j * (np.arctan2(y, x) - position))))
        h, g = loop.run(tensor(starts), tensor(goals), p)
        heading_delta = np.max(abs(np.angle(np.exp(1j * (h.numpy() - expected)))))
        goal_delta = np.max(abs(np.angle(np.exp(1j * (g.numpy() - (goals - position))))))
        assert max(heading_delta, goal_delta) < 1e-7, (arm, heading_delta, goal_delta)
        result[arm] = dict(max_trajectory_heading_difference_rad=heading_delta,
                           max_terminal_goal_difference_rad=goal_delta)
    return result


def init(seed):
    lo, hi, center = RANGES.T
    z = (center - lo) / (hi - lo)
    return tensor(np.log(z / (1. - z)) + np.random.default_rng(seed).normal(0., .12, len(z)))


def physical(z):
    lo, hi, _ = tensor(RANGES).T
    return lo + (hi - lo) * z.sigmoid()


def loss_terms(loop, starts, goals, z):
    h, g = loop.run(starts, goals, physical(z))
    return h.square().mean(), g.square().mean()


def optimize(loop, seed, batches, steps):
    z = torch.nn.Parameter(init(seed))
    adam = torch.optim.Adam([z], lr=LR)
    curve = []
    for i, (starts, goals) in enumerate(batches[:steps], 1):
        adam.zero_grad()
        h, g = loss_terms(loop, starts, goals, z)
        (h + g).backward()
        norm = torch.nn.utils.clip_grad_norm_([z], 5., error_if_nonfinite=True)
        curve.append(dict(update=i, heading_rad2=h.item(), goal_rad2=g.item(), grad_norm=norm.item()))
        adam.step()
    return physical(z.detach()), curve


def batches():
    starts, goals = map(tensor, train_panel())
    rng = np.random.default_rng(41053)
    result = []
    while len(result) < UPDATES:
        order = rng.permutation(len(starts)).reshape(-1, BATCH)
        result.extend((starts[indices], goals[indices]) for indices in order)
    return (starts, goals), result[:UPDATES]


def cycle2(loops, training_batches):
    result = {}
    starts, goals = (x[:12] for x in training_batches[0])
    for arm, loop in loops.items():
        z = init(SEEDS[0]).requires_grad_()
        gradient = torch.autograd.grad(sum(loss_terms(loop, starts, goals, z)), z)[0]
        numbers = []
        with torch.no_grad():
            for direction in torch.eye(len(z), dtype=DTYPE):
                eps = 1e-6
                a = sum(loss_terms(loop, starts, goals, z + eps * direction))
                b = sum(loss_terms(loop, starts, goals, z - eps * direction))
                numbers.append((a - b) / (2 * eps))
        numerical = torch.stack(numbers)
        assert torch.allclose(gradient, numerical, atol=2e-5, rtol=.003), (arm, gradient, numerical)
        assert torch.all(gradient.abs() > 1e-9), (arm, gradient)
        first, c1 = optimize(loop, SEEDS[0], training_batches, 2)
        second, c2 = optimize(loop, SEEDS[0], training_batches, 2)
        assert torch.equal(first, second) and c1 == c2
        result[arm] = dict(max_abs_autodiff_minus_finite_difference=float((gradient - numerical).abs().max()),
                           gradient_by_parameter=dict(zip(GROUPS + ADAPTERS, gradient.tolist())),
                           two_update_replay_bitwise_equal=True)
    return result


@torch.no_grad()
def score(loop, p, panels):
    starts, yaw = map(tensor, panels['compass'])
    initial = p[11] * phase_bump(loop.phase, starts)
    e = initial
    for dt in yaw.T:
        e = loop.step(e, dt, p)
    xy = e @ loop.observer
    retention = torch.linalg.vector_norm(xy, dim=1) / torch.linalg.vector_norm(initial @ loop.observer, dim=1)
    error = torch.rad2deg(wrap(decode(xy) - starts - yaw.sum(1)).abs()).numpy()
    moving = (yaw != 0).any(1).numpy()
    retained = retention.numpy() >= .25
    passes = retained & (error <= 20.)
    starts, goals = map(tensor, panels['steering'])
    steering = loop.prepare_steering(goals, p)
    correct = loop.command(starts, steering) * wrap(goals - starts) > 0.
    state = starts
    for _ in range(80):
        state = state + (p[14] * loop.command(state, steering)).clamp(-.12, .12)
    clamped = torch.rad2deg(wrap(goals - state).abs()).numpy()
    h, g = loop.run(*map(tensor, panels['combined']), p)
    goal_error = torch.rad2deg(g.abs()).numpy()
    return dict(stationary=int(passes[~moving].sum()), stationary_median_deg=float(np.median(error[~moving])),
                dark_turn_deg=float(np.median(error[moving])), dark_turn_pass=int(passes[moving].sum()),
                dark_turn_retained=int(retained[moving].sum()), steering=int(correct.sum()),
                convergence=int((clamped <= 20.).sum()), convergence_deg=float(np.median(clamped)),
                combined=int((goal_error <= 20.).sum()), combined_goal_deg=float(np.median(goal_error)),
                combined_compass_deg=float(np.median(torch.rad2deg(h[-1].abs()).numpy())))


def check_historical(row, history, arm):
    q = history['compass_separate_timescales']['observed_contacts'][arm]['qualification']
    s = history['steering'][arm]['qualification']
    c = history['combined_supplied_goal'][arm]['80']
    expected = [q['stationary_pass'], q['stationary_median_deg'], q['turn_median_deg'], q['turn_pass'],
                q['turn_retained'], s['correct'], s['closed_loop_within_20_deg'],
                s['closed_loop_median_error_deg'], c['within_20_deg'], c['goal_error_median_deg'],
                c['compass_error_median_deg']]
    assert np.allclose(list(row.values()), expected, atol=1e-7, rtol=0), (arm, row, expected)


def gate(native, rewired, baseline):
    isolated = {key: native[key] >= baseline[key] for key in
                ('stationary', 'dark_turn_pass', 'steering', 'convergence')}
    isolated['dark_turn_deg'] = native['dark_turn_deg'] <= baseline['dark_turn_deg'] + 1e-7
    return dict(native_goals=native['combined'] >= 72,
                native_compass=native['combined_compass_deg'] < 15.,
                wiring_margin=native['combined'] - rewired['combined'] >= 30,
                rewired_also_meets_combined_target=rewired['combined'] >= 72 and rewired['combined_compass_deg'] < 15.,
                isolated_no_regression=isolated,
                passed=(native['combined'] >= 72 and native['combined_compass_deg'] < 15. and
                        native['combined'] - rewired['combined'] >= 30 and all(isolated.values())))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--freeze', action='store_true', help='write the lock without fitting or scoring')
    args = cli.parse_args()
    torch.set_num_threads(1)
    graph = read_graph()
    lock = make_lock(graph)
    if args.freeze:
        if (ROOT / 'FIT_LOCK.json').exists():
            raise FileExistsError('FIT_LOCK.json already exists')
        save('FIT_LOCK.json', lock)
        return
    assert load('FIT_LOCK.json') == json.loads(json.dumps(lock)), 'Lock mismatch'
    model = Interface(graph, *(load(name) for name in FILES[1:]))
    loops = {arm: Loop(model, arm) for arm in ARMS}
    history = load(OLD)
    cycles = {'cycle1': cycle1(model, graph['labels'], loops, history)}
    print('Cycle 1 passed', flush=True)
    full_training, train_batches = batches()
    cycles['cycle2'] = cycle2(loops, train_batches)
    print('Cycle 2 passed', flush=True)
    records = []
    for seed in SEEDS:
        for arm, loop in loops.items():
            start = time.monotonic()
            fitted, curve = optimize(loop, seed, train_batches, UPDATES)
            with torch.no_grad():
                initial_loss = loss_terms(loop, *full_training, init(seed))
                # Evaluate final loss directly from physical gains.
                h, g = loop.run(*full_training, fitted)
                terminal_loss = (h.square().mean(), g.square().mean())
            records.append(dict(seed=seed, arm=arm, parameters=fitted.tolist(), curve=curve,
                                train_initial_rad2=[x.item() for x in initial_loss],
                                train_final_rad2=[x.item() for x in terminal_loss],
                                seconds=time.monotonic() - start))
            print(seed, arm, records[-1]['train_initial_rad2'], records[-1]['train_final_rad2'], flush=True)
    # Persist all fitted values before opening the historical panels.
    save('FIT_CURVES.json', dict(
        lock_sha256=sha256((ROOT / 'FIT_LOCK.json').read_bytes()).hexdigest(),
        platform=platform.platform(), torch=str(torch.__version__), numpy=np.__version__,
        parameter_order=GROUPS + ADAPTERS, records=records))
    panels = sealed_panels()
    hand = {arm: score(loop, old_params(history, arm), panels) for arm, loop in loops.items()}
    for arm in ARMS:
        check_historical(hand[arm], history, arm)
    names = extract_groups(graph)
    rows = {}
    for record in records:
        seed, arm = record['seed'], record['arm']
        p = record['parameters']
        rows.setdefault(str(seed), {})[arm] = dict(
            scores=score(loops[arm], tensor(p), panels),
            role_gains=dict(zip(names, (dict(zip(GROUPS, p)).get(name, 1.) for name in names))),
            adapter_and_dynamics=dict(zip(ADAPTERS, p[len(GROUPS):])))
    for row in rows.values():
        row['gate'] = gate(row['native']['scores'], row['rewired']['scores'], hand['native'])
    cycles['cycle3'] = dict(fits_completed=len(records), historical_parity=True,
                            all_seeds_passed=all(row['gate']['passed'] for row in rows.values()))
    save('FIT_RESULTS.json', dict(cycles=cycles, hand_set=hand, fitted=rows))
    print('Cycle 3 complete. All-seed qualification:', cycles['cycle3']['all_seeds_passed'], flush=True)


if __name__ == '__main__':
    main()
