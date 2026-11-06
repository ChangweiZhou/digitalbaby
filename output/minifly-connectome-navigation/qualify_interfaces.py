"""Fixed-gain v630 compass and steering probes on development fixtures."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from qualify import rewire

ROOT = Path(__file__).resolve().parent
HEADINGS = 2 * np.pi * (np.arange(16) + .25) / 16
STEP_SECONDS = .05
HOLD_STEPS = 40


def wrap(angle):
    return np.angle(np.exp(1j * angle))


class Interface:
    def __init__(self, graph, coordinates, boundary, synapses):
        labels, self.roles, self.sides = (graph[k] for k in ('labels', 'roles', 'sides'))
        self.epg = self.roles == 'EPG'
        self.pen_a = np.char.startswith(labels, 'PEN_a')
        self.er = labels == 'ER4d'
        self.fc2 = self.roles == 'FC2'
        self.pfl3 = labels == 'PFL3'
        cells = {r['root_id']: r for r in coordinates['cells']}
        ids = graph['ids']
        eb = {row['root_id']: row['EB'] for row in synapses['epg']}
        self.phase = np.deg2rad([eb[k]['phase_deg'] for k in ids[self.epg]])
        # Least-squares spatial first harmonic with a DC column. Its design matrix uses only anatomy.
        self.observer = np.linalg.pinv(np.column_stack((np.ones(self.epg.sum()), np.cos(self.phase), np.sin(self.phase))))[1:]
        pb = np.array([cells[k]['PB']['position_um'] for k in ids[self.epg]])
        pb_position = np.array([cells.get(k, {}).get('PB', {}).get('position_um', np.nan) for k in ids])
        pfl3_pb = np.array([cells[k]['PB']['position_um'] for k in ids[self.pfl3]])
        pfl3_x = np.array([cells[k]['FB']['centroid_um'][0] for k in ids[self.pfl3]])
        fc2_x = np.array([cells[k]['FB']['centroid_um'][0] for k in ids[self.fc2]])
        # One rotation of FB phase across its PFL3-covered x span. Fit a circular offset from
        # PB/FB geometry on each side separately, then take the midpoint of the opponent shifts.
        pfl3_phase = np.angle(np.exp(-.5 * ((pfl3_pb[:, None] - pb) / 3.)**2) @ np.exp(1j * self.phase))
        slope = 2 * np.pi / np.ptp(pfl3_x)
        by_side = self.sides[self.pfl3]
        offsets = [np.mean(np.exp(1j * (pfl3_phase - slope * pfl3_x))[by_side == s]) for s in ('left', 'right')]
        self.fb_phase_slope = slope
        self.fb_phase_offset = np.angle(sum(z / abs(z) for z in offsets))
        self.fc2_phase = wrap(slope * fc2_x + self.fb_phase_offset)
        self.fc2_alignment = dict(spatial_period_um=float(np.ptp(pfl3_x)),
                                  side_alignment_resultants=[float(abs(z)) for z in offsets])

        u, v, w, signs, usable = (graph[k] for k in ('u', 'v', 'w', 'sign', 'usable'))
        er_phase = np.zeros(len(labels), dtype=complex)
        phase_vector = np.zeros(len(labels), dtype=complex)
        phase_vector[self.epg] = np.exp(1j * self.phase)
        for source in np.flatnonzero(self.er):
            a = (u == source) & self.epg[v] & usable
            er_phase[source] = w[a] @ phase_vector[v[a]] / max(w[a].sum(), 1)
        self.er_phase = np.angle(er_phase[self.er])
        self.er_selectivity = abs(er_phase[self.er])
        motor = np.zeros((2, len(labels)))
        idx = {k: i for i, k in enumerate(ids)}
        for row in boundary['pfl3_to_dna02']:
            if row['consistent']:
                side = ('left', 'right').index(row['target_side'])
                motor[side, idx[row['source']]] = row['raw_signed_weight']
        assert np.all(motor.sum(axis=1) > 0) and np.all(np.count_nonzero(motor, axis=0)[self.pfl3] == 1)
        self.motor = motor / motor.sum(axis=1, keepdims=True)
        self.motor_edge_count = int(np.count_nonzero(motor))
        self.glno_current = np.zeros(len(labels))
        for row in boundary['glno_to_pen_a']:
            if row['consistent']:
                source_sign = 1. if row['source_side'] == 'left' else -1.
                self.glno_current[idx[row['target']]] += source_sign * row['raw_signed_weight']
        self.glno_current /= np.median(np.abs(self.glno_current[self.pen_a]))
        assert np.isfinite(self.glno_current).all()

        core = self.epg | self.er | self.fc2 | self.pfl3 | np.isin(self.roles, ['D7', 'PEG', 'PEN'])
        graph_scope = core[u] & core[v]
        sensitivity = np.isin(ids[u], boundary['glutamatergic_delta7']) & (self.epg[v] | self.pfl3[v])
        assert not np.any(usable & sensitivity)
        self.compass = np.isin(self.roles, ['EPG', 'PEN', 'PEG', 'D7'])
        self.circuits = {}
        self.variants = {method: {} for method in ('aggregate', 'pb_proximity', 'observed_contacts')}
        for name, mask in [('native', usable), ('delta7_inhibitory', usable | sensitivity)]:
            sel = graph_scope & mask
            sign = np.where(sensitivity[sel], -1, signs[sel])
            a, b, weight = u[sel], v[sel], w[sel] * sign
            # Rewire within anatomical subtype and side, retaining the same boundary neurons.
            null_labels = np.char.add(np.char.add(labels, '/'), self.sides)
            null, swaps = rewire(a, b, sign, null_labels)
            self.circuits[name] = self.normalize(a, b, weight)
            self.circuits[name + '_rewired'] = self.normalize(a, null, weight)
            self.circuits[name + '_rewired_swaps'] = dict(accepted=swaps, changed=int(np.count_nonzero(b != null)))
            if name == 'delta7_inhibitory':
                filtered = weight.astype(float)
                epg_to_pen = self.epg[a] & self.pen_a[b]
                filtered[epg_to_pen] *= np.exp(-.5 * ((pb_position[a[epg_to_pen]] - pb_position[b[epg_to_pen]]) / 3.)**2)
                self.pen_pb_phase_shift = self.pen_offsets(a, b, weight, filtered)
                for method, effective in [('aggregate', weight), ('pb_proximity', filtered)]:
                    self.variants[method] = {'': (a, b, effective), '_rewired': (a, null, effective)}
                self.register_observed_compartments(a, b, weight, sign, ids, synapses, null_labels)
                epg_excit = self.epg[b] & self.compass[a] & (weight > 0)
                epg = self.epg[b]
                fractions = (np.bincount(b[epg_excit], weights=weight[epg_excit], minlength=len(labels)) /
                             np.bincount(b[epg], weights=abs(weight[epg]), minlength=len(labels)).clip(1e-12))
                self.epg_excitatory_fraction = float(np.median(fractions[self.epg]))
        # Removing the compass recurrent inputs keeps cue and yaw drives identical across arms.
        a, b, w = self.circuits['native']
        keep = ~(self.compass[a] & self.compass[b])
        self.circuits['recurrence_off'] = a[keep], b[keep], w[keep]

    def normalize(self, source, target, w):
        mass = np.bincount(target, weights=abs(w), minlength=len(self.roles))
        return source, target, 1.2 * w / np.maximum(mass[target], 1e-12)

    def register_observed_compartments(self, a, b, weight, sign, ids, synapses, null_labels):
        sites = {(row['source'], row['target']): row for row in synapses['epg_pen_a_connections']}
        exact = weight.astype(float)
        afferent, efferent = self.epg[a] & self.pen_a[b], self.pen_a[a] & self.epg[b]
        stats = {}
        for route, roi in [(afferent, 'pb'), (efferent, 'eb')]:
            indices = np.flatnonzero(route)
            found = 0
            for i in indices:
                row = sites.get((ids[a[i]], ids[b[i]]))
                if row is not None:
                    found += 1
                    exact[i] *= row[roi] / row['total']
                else:
                    exact[i] = 0
            stats[roi] = dict(aggregate_edges=len(indices), coordinate_export_pairs=found,
                              edges_with_contacts_in_compartment=int(np.count_nonzero(exact[indices])))
        keep = exact != 0
        q, swaps = rewire(a[keep], b[keep], sign[keep], null_labels)
        self.variants['observed_contacts'] = {'': (a[keep], b[keep], exact[keep]), '_rewired': (a[keep], q, exact[keep])}
        self.observed_compartments = dict(stats, rewired_swaps=swaps, changed=int(np.count_nonzero(b[keep] != q)),
                                          median_feedback_phase_shift_deg=self.pen_offsets(a, b, weight, exact, exact))

    def pen_offsets(self, source, target, weights, filtered, filtered_outputs=None):
        phases = np.zeros(len(self.roles), dtype=complex)
        phases[self.epg] = np.exp(1j * self.phase)
        offsets = {'left': [[], []], 'right': [[], []]}
        if filtered_outputs is None:
            filtered_outputs = weights
        for i in np.flatnonzero(self.pen_a):
            entering = (target == i) & self.epg[source]
            exiting = (source == i) & self.epg[target]
            for result, w_in, w_out in zip(offsets[self.sides[i]], (weights, filtered), (weights, filtered_outputs)):
                in_phase = w_in[entering] @ phases[source[entering]]
                out_phase = w_out[exiting] @ phases[target[exiting]]
                result.append(np.rad2deg(np.angle(out_phase * in_phase.conjugate())))
        return {s: dict(aggregate=float(np.median(raw)), compartment_weighted=float(np.median(pb)))
                for s, (raw, pb) in offsets.items()}

    def separate_compass_gains(self, excitation, inhibition, suffix, method='aggregate'):
        source, target, weight = self.variants[method][suffix]
        a, b, w = self.normalize(source, target, weight)
        recurrent = self.compass[source] & self.compass[target]
        landmark = self.er[source] & self.epg[target]
        # An inactive sensory pathway should not dilute compass recurrence. Keep the observed
        # ratios within each of the three independently normalized incoming route classes.
        for mask, gain in [(recurrent & (weight > 0), excitation), (recurrent & (weight < 0), inhibition),
                           (landmark, .4)]:
            mass = np.bincount(target[mask], weights=abs(weight[mask]), minlength=len(self.roles))
            w[mask] = gain * weight[mask] / mass[target[mask]]
        self.circuits['separated' + suffix] = a, b, w

    def initial_state(self):
        return np.zeros(len(self.roles))

    def step(self, state, circuit, landmark_bearing=None, yaw=0., supplied_goal=None, sensory='epg',
             yaw_scale=.06, yaw_adapter='pen_gain'):
        a, b, w = self.circuits[circuit]
        current = np.bincount(b, weights=w * state[a], minlength=len(state))
        if landmark_bearing is not None:
            # The fixture has one landmark at phase zero and supplies its egocentric bearing.
            if sensory == 'epg':
                current[self.epg] += .9 * np.exp(4 * (np.cos(self.phase + landmark_bearing) - 1))
            elif sensory in ('er', 'er_constant'):
                contrast = self.er_selectivity * np.cos(self.er_phase + landmark_bearing) if sensory == 'er' else 0.
                current[self.er] += .8 * (1 - contrast)
                # Inhibition can modulate an active compass. It cannot seed activity from rest.
                current[self.epg] += .65
            else:
                raise ValueError(f'Unknown sensory adapter {sensory}')
        # Engineered velocity gain at PEN_a. Positive yaw is a left turn in the fixture.
        if yaw_adapter == 'glno':
            # A centered perturbation around tonic GLNO activity. Its assumed opponent sensory
            # drive is held fixed across graphs, with the signed v630 GLNO→PEN1 weights.
            current += np.clip(yaw / yaw_scale, -1, 1) * self.glno_current
        else:
            pen_side = np.where(self.sides[self.pen_a] == 'left', 1., -1.)
            current[self.pen_a] *= 1 + np.maximum(0, yaw * pen_side / yaw_scale)
        if supplied_goal is not None:
            current[self.fc2] += .9 * np.exp(4 * (np.cos(self.fc2_phase - supplied_goal) - 1))
        return .8 * state + .2 * np.tanh(np.maximum(0, current))

    def heading(self, state):
        x, y = self.observer @ state[self.epg]
        return np.arctan2(y, x), np.hypot(x, y)

    def steer(self, state):
        left, right = self.motor @ state
        return left - right


def compass_probe(model, circuit, sensory, headings=HEADINGS, yaw_scale=.06, yaw_adapter='pen_gain'):
    visible, stationary, turning = [], [], []
    for angle in headings:
        initial = model.initial_state()
        for _ in range(HOLD_STEPS):
            initial = model.step(initial, circuit, landmark_bearing=-angle, sensory=sensory)
        locked, strength = model.heading(initial)
        visible.append((abs(wrap(locked - angle)), strength))
        stationary_phase = None
        for yaw in (0., -.04, .04):
            state = initial.copy()
            for _ in range(HOLD_STEPS):
                state = model.step(state, circuit, yaw=yaw, yaw_scale=yaw_scale, yaw_adapter=yaw_adapter)
            phase, remaining = model.heading(state)
            retention = remaining / max(strength, 1e-12)
            if yaw == 0:
                stationary_phase = phase
                stationary.append((abs(wrap(phase - locked)), retention))
            else:
                turning.append((abs(wrap(phase - angle - HOLD_STEPS * yaw)), retention,
                                np.sign(yaw) * wrap(phase - stationary_phase)))
    visible, stationary, turning = map(np.asarray, (visible, stationary, turning))
    return dict(landmark_median_error_deg=float(np.rad2deg(np.median(visible[:, 0]))),
                landmark_median_harmonic=float(np.median(visible[:, 1])),
                landmark_n_with_harmonic_at_least_0_02=int(np.count_nonzero(visible[:, 1] >= .02)),
                stationary_with_at_least_quarter_strength=int(np.count_nonzero(stationary[:, 1] >= .25)),
                stationary_with_quarter_strength_and_drift_at_most_20_deg=int(np.count_nonzero(
                    (stationary[:, 1] >= .25) & (stationary[:, 0] <= np.deg2rad(20)))),
                stationary_n=len(stationary), stationary_median_strength_ratio=float(np.median(stationary[:, 1])),
                turn_median_error_deg=float(np.rad2deg(np.median(turning[:, 0]))),
                turn_median_directional_response_deg=float(np.rad2deg(np.median(turning[:, 2]))),
                turn_median_strength_ratio=float(np.median(turning[:, 1])), turn_n=len(turning))


def steering_probe(model, circuit):
    rows = []
    for heading in HEADINGS:
        for goal_offset in (-np.pi / 2, -np.pi / 4, np.pi / 4, np.pi / 2):
            state = model.initial_state()
            no_goal = model.initial_state()
            for _ in range(HOLD_STEPS):
                state = model.step(state, circuit, landmark_bearing=-heading, supplied_goal=heading + goal_offset)
                no_goal = model.step(no_goal, circuit, landmark_bearing=-heading)
            rows.append((goal_offset, model.steer(state), model.steer(no_goal)))
    target, supplied, absent = np.asarray(rows).T
    return dict(correct_turn_sign=int(np.count_nonzero(target * supplied > 0)),
                no_goal_correct_turn_sign=int(np.count_nonzero(target * absent > 0)),
                goal_effect_correct_turn_sign=int(np.count_nonzero(target * (supplied - absent) > 0)),
                n=len(rows), median_abs_goal_effect=float(np.median(abs(supplied - absent))))


def gain_probe(model):
    # A bounded development grid. Probe a different angular panel from the original smoke test.
    headings = HEADINGS + np.pi / 16
    trials = []
    for excite in (1.2, 1.8, 2.4):
        for inhibit in (.4, .8, 1.2):
            for suffix in ('', '_rewired'):
                model.separate_compass_gains(excite, inhibit, suffix)
            trials.append(dict(excitatory_gain=excite, inhibitory_gain=inhibit,
                               native=compass_probe(model, 'separated', 'epg', headings),
                               rewired=compass_probe(model, 'separated_rewired', 'epg', headings)))
    return trials


def compartment_probe(model, method):
    # Follow-up on a shifted panel after the separated-gain development grid.
    headings = HEADINGS + np.pi / 32
    trials = []
    for inhibition in (.4, .8):
        for suffix in ('', '_rewired'):
            model.separate_compass_gains(1.2, inhibition, suffix, method=method)
        for yaw_scale in (.02, .04, .06):
            trials.append(dict(excitatory_gain=1.2, inhibitory_gain=inhibition, yaw_scale=yaw_scale,
                               native=compass_probe(model, 'separated', 'epg', headings, yaw_scale),
                               rewired=compass_probe(model, 'separated_rewired', 'epg', headings, yaw_scale)))
    return trials


def glno_probe(model):
    headings = HEADINGS + 3 * np.pi / 64
    for suffix in ('', '_rewired'):
        model.separate_compass_gains(1.2, .8, suffix, method='observed_contacts')
    return [dict(yaw_scale=scale,
                 native=compass_probe(model, 'separated', 'epg', headings, scale, yaw_adapter='glno'),
                 rewired=compass_probe(model, 'separated_rewired', 'epg', headings, scale, yaw_adapter='glno'))
            for scale in (.04, .08, .16)]


def velocity_probe(model, graph, coordinates):
    """Test instantaneous velocity with a clamped compass and no displacement state."""
    roles, labels, sides, ids = (graph[k] for k in ('roles', 'labels', 'sides', 'ids'))
    pfnd, pfnv, hdelta, pfr = (labels == t for t in ('PFNd', 'PFNv', 'hDeltaB', 'PFR'))
    pfn = pfnd | pfnv
    cells = {row['root_id']: row for row in coordinates['cells']}
    pfr_x = np.array([cells[k]['FB']['centroid_um'][0] for k in ids[pfr]])
    phase = model.fb_phase_slope * pfr_x + model.fb_phase_offset
    observer = np.linalg.pinv(np.column_stack((np.ones(pfr.sum()), np.cos(phase), np.sin(phase))))[1:]
    # Forward positive, left positive. The four body axes follow Lyu et al. Fig. 4.
    left = sides[pfn] == 'left'
    axes = np.where(pfnd[pfn], np.where(left, 45., -45.), np.where(left, -135., 135.))
    axes = np.deg2rad(axes)
    u, v, w, sign, usable = (graph[k] for k in ('u', 'v', 'w', 'sign', 'usable'))
    route = (model.epg[u] & pfn[v]) | (pfn[u] & hdelta[v]) | (hdelta[u] & pfr[v])
    a, b, w = u[route & usable], v[route & usable], w[route & usable] * sign[route & usable]
    q, swaps = rewire(a, b, np.sign(w), np.char.add(np.char.add(labels, '/'), sides))
    n = len(roles)
    report = {}
    directions = np.pi * np.arange(8) / 4
    for name, target in [('native', b), ('rewired', q)]:
        mass = np.bincount(target, weights=np.abs(w), minlength=n).clip(1e-12)

        def current(source):
            return np.bincount(target, weights=w * source[a], minlength=n) / mass

        errors, strength, zero = [], [], []
        for heading in HEADINGS:
            epg = np.zeros(n)
            epg[model.epg] = np.exp(4 * (np.cos(model.phase - heading) - 1))
            heading_current = current(epg)[pfn]
            for direction in np.r_[directions, np.nan]:
                pfn_state = np.zeros(n)
                if np.isfinite(direction):
                    pfn_state[pfn] = heading_current * np.maximum(0, np.cos(direction - axes))
                hdelta_state = current(pfn_state) * hdelta
                x, y = observer @ current(hdelta_state)[pfr]
                if np.isfinite(direction):
                    errors.append(abs(wrap(np.arctan2(y, x) - heading - direction)))
                    strength.append(np.hypot(x, y))
                else:
                    zero.append(np.hypot(x, y))
        report[name] = dict(n=len(errors), median_direction_error_deg=float(np.rad2deg(np.median(errors))),
                            median_harmonic=float(np.median(strength)), max_zero_motion_harmonic=float(max(zero)))
    return dict(report, changed_edges=int(np.count_nonzero(b != q)), accepted_rewire_swaps=swaps,
                description='Feedforward EPG→PFNd/v→hDeltaB→PFR with externally clamped compass and no integrated position')


def main():
    names = ['EXTRACT.npz', 'COMPARTMENT_MAP.json', 'RESOLVED_BOUNDARY.json', 'SYNAPSE_COMPARTMENTS.json']
    extract, map_file, boundary_file, synapse_file = (ROOT / name for name in names)
    with np.load(extract, allow_pickle=False) as source:
        graph = dict(source)
    coordinates = json.loads(map_file.read_text())
    model = Interface(graph, coordinates, json.loads(boundary_file.read_text()), json.loads(synapse_file.read_text()))
    # The motor signs are fixed by the anatomical output sides and the published ipsiversive convention.
    for side, expected_sign in [('left', 1), ('right', -1)]:
        pulse = (model.motor[('left', 'right').index(side)] > 0).astype(float)
        assert np.isclose(model.steer(pulse), expected_sign)
    arms = ('native', 'native_rewired', 'recurrence_off', 'delta7_inhibitory', 'delta7_inhibitory_rewired')
    report = dict(fixture='fixed-gain engineering development only', step_seconds=STEP_SECONDS, fitted_scalars=0,
                  graph_inputs={name: sha256((ROOT / name).read_bytes()).hexdigest() for name in names},
                  dna02_motor_edge_count=model.motor_edge_count, dna02_left_right_sign_canary=True,
                  epg_median_compass_excitation_fraction_of_incoming_absolute_mass=model.epg_excitatory_fraction,
                  fc2_geometry=model.fc2_alignment,
                  swaps={k: v for k, v in model.circuits.items() if k.endswith('_swaps')},
                  heading={arm: compass_probe(model, arm, 'epg') for arm in arms},
                  er4d_only={arm: compass_probe(model, arm, 'er') for arm in ('native', 'native_rewired')},
                  er4d_constant_drive_control=compass_probe(model, 'native', 'er_constant'),
                  steering={arm: steering_probe(model, arm) for arm in arms if arm != 'recurrence_off'},
                  separated_compass_gain_development_grid=gain_probe(model),
                  pen1_median_feedback_phase_shift_deg=model.pen_pb_phase_shift,
                  pb_weighted_compass_development_grid=compartment_probe(model, 'pb_proximity'),
                  observed_compartments=model.observed_compartments,
                  observed_compartments_development_grid=compartment_probe(model, 'observed_contacts'),
                  glno_yaw_development_grid=glno_probe(model),
                  clamped_heading_velocity=velocity_probe(model, graph, coordinates),
                  displacement='unopened pending compass qualification', learned_navigation='unopened')
    (ROOT / 'INTERFACE_PROBE.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('heading', 'er4d_only', 'steering')}, indent=2))


if __name__ == '__main__':
    main()
