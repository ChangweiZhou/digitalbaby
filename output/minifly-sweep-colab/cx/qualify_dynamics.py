"""Calibrate explicit compass and steering subcircuits, then qualify on fresh panels."""
from hashlib import sha256
from itertools import product
import json
import numpy as np
from qualify_interfaces import Interface, ROOT, wrap

FILES = ('EXTRACT.npz', 'COMPARTMENT_MAP.json', 'RESOLVED_BOUNDARY.json', 'SYNAPSE_COMPARTMENTS.json')
N = 40
PANEL = np.arange(16) * 2 * np.pi / 16


def bump(phases, angles):
    return np.exp(4 * (np.cos(np.asarray(angles)[..., None] - phases) - 1))


def projection(source, target, weights, pre, post, normalize=True):
    a = np.flatnonzero(pre)
    b = np.flatnonzero(post)
    mask = pre[source] & post[target]
    matrix = np.zeros((len(a), len(b)))
    np.add.at(matrix, (np.searchsorted(a, source[mask]), np.searchsorted(b, target[mask])), weights[mask])
    if normalize:
        matrix /= np.abs(matrix).sum(axis=0).clip(1e-12)
    return matrix


class Compass:
    def __init__(self, model, route, arm):
        a, b, w = model.variants[route]['_rewired' if arm == 'rewired' else '']
        e, p = model.epg, model.pen_a
        d = model.roles == 'D7'
        self.local = projection(a, b, w, e, e)
        self.enter = projection(a, b, w, e, p)
        # Normalize by the combined population, so the two PEN branches compete.
        self.exit = projection(a, b, w, p, e)
        self.inhib = projection(a, b, w, e, d) @ projection(a, b, w, d, e)
        self.side = np.where(model.sides[p] == 'left', 1., -1.)
        self.model = model
        if arm == 'recurrence_off':
            self.local *= 0
            self.inhib *= 0
            self.exit *= 0

    def step(self, e, yaw, params, drive_mode, rest_rate):
        local, inhibit, feedback, yaw_gain = params
        p = np.maximum(e @ self.enter, 0.)
        modulation = np.clip(yaw[:, None] * yaw_gain * self.side, -1., 1.)
        if drive_mode == 'single_side':
            modulation = np.maximum(modulation, 0.)
        resting = local * (e @ self.local) + feedback * (p @ self.exit) + inhibit * (e @ self.inhib)
        active = np.tanh(np.maximum(resting + feedback * ((p * modulation) @ self.exit), 0.))
        if rest_rate == .45:
            return .55 * e + .45 * active
        resting = np.tanh(np.maximum(resting, 0.))
        return np.clip(e + rest_rate * (resting - e) + .45 * (active - resting), 0., 1.)

    def run(self, angles, yaw, params, drive_mode='opponent', rest_rate=.45):
        e = bump(self.model.phase, angles)
        initial = e
        for dt in yaw.T:
            e = self.step(e, dt, params, drive_mode, rest_rate)
        xy = e @ self.model.observer.T
        baseline = np.linalg.norm(initial @ self.model.observer.T, axis=1)
        return np.arctan2(xy[:, 1], xy[:, 0]), np.linalg.norm(xy, axis=1) / baseline


def turn_panel(angles, sequences):
    return np.tile(angles, len(sequences)), np.repeat(sequences, len(angles), axis=0)


def compass_score(compass, angles, yaw, params, mode, rest_rate=.45):
    phase, retention = compass.run(angles, yaw, params, mode, rest_rate)
    moving = np.any(yaw != 0, axis=1)
    error = np.rad2deg(abs(wrap(phase - angles - yaw.sum(axis=1))))
    passes = (retention >= .25) & (error <= 20.)
    return dict(stationary_pass=int(passes[~moving].sum()), stationary_n=int((~moving).sum()),
                turn_median_deg=float(np.median(error[moving])), turn_n=int(moving.sum()),
                turn_pass=int(passes[moving].sum()),
                turn_retained=int((retention[moving] >= .25).sum()),
                stationary_median_deg=float(np.median(error[~moving])))


def compass_experiment(model):
    dev = turn_panel(PANEL + .04, [np.zeros(N), np.full(N, .04), np.full(N, -.04)])
    held = turn_panel(PANEL + .19, [np.zeros(N), np.full(N, .03), np.full(N, -.03),
                                    np.r_[np.full(10, .065), np.full(30, -.009)],
                                    np.r_[np.full(10, -.065), np.full(30, .009)],
                                    np.r_[np.zeros(16), np.full(24, .06)],
                                    np.r_[np.zeros(16), np.full(24, -.06)]])
    report = {}
    for route in ('aggregate', 'pb_proximity', 'observed_contacts'):
        report[route] = {}
        for arm in ('native', 'rewired'):
            compass = Compass(model, route, arm)
            candidates = []
            for params in product((.6, .95, 1.25, 1.55), (.4, .8, 1.2), (.3, .65, 1.), (6., 15., 30., 55.)):
                for mode in ('opponent', 'single_side'):
                    row = compass_score(compass, *dev, params, mode)
                    # Penalize compass collapse and score angle only after stationary coverage.
                    key = (-min(row['stationary_pass'], 13), -row['turn_pass'], row['turn_median_deg'])
                    candidates.append((key, params, mode, row))
            _, params, mode, calibration = min(candidates, key=lambda row: row[0])
            qualification = compass_score(compass, *held, params, mode)
            report[route][arm] = dict(params=params, yaw_mode=mode, calibration=calibration, qualification=qualification)
            if arm == 'native':
                for lesion in ('recurrence_off', 'rewired'):
                    report[route][lesion + '_native_parameters'] = compass_score(
                        Compass(model, route, lesion), *held, params, mode)
                report[route]['no_yaw_native_parameters'] = compass_score(compass, *held, (*params[:3], 0.), mode)
    return report


def separate_timescale_experiment(model):
    dev = turn_panel(PANEL + .07, [np.zeros(N), np.full(N, .04), np.full(N, -.04),
                                   np.r_[np.full(18, .055), np.full(22, -.017)],
                                   np.r_[np.full(18, -.055), np.full(22, .017)]])
    # Fresh headings and yaw schedules for the follow-up after the first panel was examined.
    held = turn_panel(PANEL + .31, [np.zeros(N), np.full(N, .027), np.full(N, -.027),
                                    np.r_[np.full(17, .054), np.full(23, -.014)],
                                    np.r_[np.full(17, -.054), np.full(23, .014)],
                                    np.r_[np.zeros(12), np.full(28, .054)],
                                    np.r_[np.zeros(12), np.full(28, -.054)]])
    report = {}
    for route in ('pb_proximity', 'observed_contacts'):
        results = report[route] = {}
        for arm in ('native', 'rewired'):
            compass = Compass(model, route, arm)
            best = None
            for rest, local, inhibition, feedback, gain in product(
                (.055, .085, .13), (.75, 1.25, 1.55), (.4, .8), (.3, .65, 1.), (8., 16., 30.)
            ):
                params = (local, inhibition, feedback, gain)
                row = compass_score(compass, *dev, params, 'opponent', rest)
                key = (-min(row['stationary_pass'], 13), -row['turn_pass'], row['turn_median_deg'])
                if best is None or key < best[0]:
                    best = (key, params, rest, row)
            _, params, rest, row = best
            results[arm] = dict(params=params, rest_rate=rest, calibration=row,
                                qualification=compass_score(compass, *held, params, 'opponent', rest))
            if arm == 'native':
                for control in ('rewired', 'recurrence_off', 'no_yaw'):
                    net = compass if control == 'no_yaw' else Compass(model, route, control)
                    p = (*params[:3], 0.) if control == 'no_yaw' else params
                    results[control + '_native_parameters'] = compass_score(net, *held, p, 'opponent', rest)
    return report


class Steering:
    def __init__(self, model, arm):
        a, b, w = model.variants['aggregate']['_rewired' if arm == 'rewired' else '']
        e, f, p = model.epg, model.fc2, model.pfl3
        d = model.roles == 'D7'
        self.head = projection(a, b, w, e, p)
        self.delta = projection(a, b, w, e, d) @ projection(a, b, w, d, p)
        self.goal = projection(a, b, w, f, p)
        self.motor = model.motor[:, p]
        self.model = model
        # Center and scale the two input streams independently. Normalization uses
        # a uniform phase panel and no turn outcomes or target offsets.
        ph = np.arange(64) * (2 * np.pi / 64)
        self.stats = [[(x.mean(axis=0), x.std(axis=0).clip(.04)) for x in self.streams(ph, ph, weight)]
                      for weight in (0., .5, 1.)]

    def streams(self, headings, goals, weight):
        state = bump(self.model.phase, headings)
        return state @ (self.head + weight * self.delta), bump(self.model.fc2_phase, goals) @ self.goal

    def command(self, headings, goals, params, lesion=None):
        offset, weight, goal_gain, threshold = params
        h, g = self.streams(headings, goals + offset, weight)
        (h_mean, h_scale), (g_mean, g_scale) = self.stats[int(weight * 2)]
        h, g = (h - h_mean) / h_scale, (g - g_mean) / g_scale
        if lesion == 'no_goal':
            g *= 0.
        if lesion == 'no_heading':
            h *= 0.
        response = np.logaddexp(0., 3 * (h + goal_gain * g - threshold)) / 3
        return response @ (self.motor[0] - self.motor[1])


def steering_panel(headings, offsets):
    h = np.repeat(headings, len(offsets))
    return h, h + np.tile(offsets, len(headings))


def steering_score(steering, headings, goals, params, lesion=None, closed=True):
    truth = wrap(goals - headings)
    u = steering.command(headings, goals, params, lesion)
    correct = int(np.count_nonzero(u * truth > 0.))
    pair_correct = int(np.all((u * truth > 0).reshape(-1, 2), axis=1).sum())
    result = dict(correct=correct, n=len(u), opposite_goal_pairs=pair_correct,
                  median_abs_output=float(np.median(abs(u))))
    if closed:
        state = headings.copy()
        for _ in range(80):
            command = steering.command(state, goals, params, lesion)
            state += np.clip(.14 * command, -.12, .12)
        error = np.rad2deg(abs(wrap(goals - state)))
        result.update(closed_loop_median_error_deg=float(np.median(error)),
                      closed_loop_within_20_deg=int((error <= 20.).sum()))
    return result


def steering_experiment(model):
    dev = steering_panel(PANEL + .04, np.deg2rad([-40, 40, -95, 95]))
    held = steering_panel(PANEL + .19, np.deg2rad([-30, 30, -65, 65, -120, 120]))
    report = {}
    for arm in ('native', 'rewired'):
        s = Steering(model, arm)
        rows = []
        # Motor polarity stays fixed to DNa02 laterality. The phase offset calibrates
        # the FB goal coordinate into the EPG coordinate on the development panel.
        for params in product(np.arange(24) * np.pi / 12, (0., .5, 1.), (.5, 1., 2.), (.25, .75, 1.25)):
            row = steering_score(s, *dev, params, closed=False)
            if row['correct'] >= 42:
                closed = steering_score(s, *dev, params)
                rows.append(((-closed['closed_loop_within_20_deg'], -row['opposite_goal_pairs'],
                              -row['correct'], closed['closed_loop_median_error_deg']), params, closed))
            else:
                rows.append(((0, -row['opposite_goal_pairs'], -row['correct'], 180.), params, row))
        _, params, calibration = min(rows, key=lambda row: row[0])
        report[arm] = dict(params=params, calibration=calibration,
                           qualification=steering_score(s, *held, params),
                           no_goal=steering_score(s, *held, params, 'no_goal'),
                           no_heading=steering_score(s, *held, params, 'no_heading'))
        if arm == 'native':
            report['rewired_native_parameters'] = steering_score(Steering(model, 'rewired'), *held, params)
    return report


def combined_probe(model, turn, steering, starts, goals, yaw_limit):
    """Close the loop using decoded compass state after a single initial cue."""
    report = {}
    for arm in ('native', 'no_yaw', 'no_goal', 'rewired'):
        graph_arm = 'rewired' if arm == 'rewired' else 'native'
        compass = Compass(model, 'observed_contacts', graph_arm)
        controller = Steering(model, graph_arm)
        turn_params = turn[graph_arm]['params']
        steering_params = steering[graph_arm]['params']
        if arm == 'no_yaw':
            turn_params = (*turn_params[:3], 0.)
        e = bump(model.phase, starts)
        position = starts.copy()  # Simulator heading, used for scoring only after the initial cue.
        panel = {}
        for step in range(1, 81):
            x, y = (e @ model.observer.T).T
            heading = np.arctan2(y, x)
            action = controller.command(heading, goals, steering_params, 'no_goal' if arm == 'no_goal' else None)
            yaw = np.clip(.14 * action, -yaw_limit, yaw_limit)
            position += yaw
            e = compass.step(e, yaw, turn_params, 'opponent', turn[graph_arm]['rest_rate'])
            if step in (40, 80):
                error = np.rad2deg(abs(wrap(goals - position)))
                x, y = (e @ model.observer.T).T
                compass_error = np.rad2deg(abs(wrap(np.arctan2(y, x) - position)))
                panel[str(step)] = dict(n=len(starts), within_20_deg=int((error <= 20.).sum()),
                                        goal_error_median_deg=float(np.median(error)),
                                        compass_error_median_deg=float(np.median(compass_error)))
        report[arm] = panel
    return report


def main():
    with np.load(ROOT / FILES[0], allow_pickle=False) as source:
        graph = dict(source)
    model = Interface(graph, *(json.loads((ROOT / name).read_text()) for name in FILES[1:]))
    turn = separate_timescale_experiment(model)
    steering = steering_experiment(model)
    initial_combined = steering_panel(PANEL + .47, np.deg2rad([-42, 42, -80, 80, -110, 110]))
    rate_panel = steering_panel(PANEL + .59, np.deg2rad([-36, 36, -71, 71, -105, 105]))
    report = dict(graph_inputs={name: sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES},
                  parameter_order=dict(compass=['EPG recurrence', 'Delta7 inhibition', 'PEN feedback', 'yaw gain'],
                                       steering=['FB phase offset in radians', 'Delta7 gain', 'FC2 gain', 'PFL3 threshold']),
                  compass=compass_experiment(model), steering=steering,
                  compass_separate_timescales=turn,
                  combined_supplied_goal=combined_probe(model, turn['observed_contacts'], steering, *initial_combined, .12),
                  combined_rate_check={str(limit): combined_probe(model, turn['observed_contacts'], steering, *rate_panel, limit)
                                       for limit in (.054, .12)},
                  scope='Engineering calibration and disjoint qualification panels. No learned-goal or displacement run.')
    (ROOT / 'DYNAMICS_QUALIFICATION.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
