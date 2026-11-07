"""Score the six frozen gains once on fresh stratified instances of the same five probes."""
import argparse
from hashlib import sha256
import json

import numpy as np
import torch

import fit_gains as first
import fit_gains_followup as followup

LOCK = 'FIT_OFFICIAL_LOCK.json'
PANEL_SEED = 20261107


def make_panels():
    rng = np.random.default_rng(PANEL_SEED)
    width = 2 * np.pi / 16
    # Keep the yaw schedules and goal offsets from the legacy panels.
    # Sample one heading inside each equal-width sector independently per panel.
    starts = [first.PANEL + width * rng.random(16) for _ in range(3)]
    historical = first.sealed_panels()
    _, yaw = historical['compass']
    panels = dict(
        compass=(np.tile(starts[0], len(yaw) // 16), yaw),
        steering=first.steering_panel(starts[1], np.deg2rad([-30, 30, -65, 65, -120, 120])),
        combined=first.steering_panel(starts[2], np.deg2rad([-42, 42, -80, 80, -110, 110])),
    )
    forbidden = np.concatenate([first.train_panel()[0], *(p[0] for p in historical.values())])
    all_starts = np.concatenate(starts)
    assert np.min(abs(np.angle(np.exp(1j * (all_starts[:, None] - forbidden))))) > 1e-6
    for name, pair in panels.items():
        assert tuple(x.shape for x in pair) == tuple(x.shape for x in historical[name])
        assert first.hash_arrays(pair) != first.hash_arrays(historical[name])
    return panels


def digest(name):
    return sha256((first.ROOT / name).read_bytes()).hexdigest()


def make_lock(graph):
    followup.verify_lock(graph)
    previous = first.load('FIT2_LOCK.json')
    fits = first.load('FIT2_RESULTS.json')['fitted']
    curves = first.load('FIT2_CURVES.json')
    assert curves['lock_sha256'] == digest('FIT2_LOCK.json')
    assert curves['parameter_order'] == list(followup.NAMES)
    vectors = {}
    for record in curves['records']:
        seed, arm = str(record['seed']), record['arm']
        row = fits[seed][arm]
        gains = row['role_gains'] | row['adapter_and_dynamics']
        values = [gains[name] for name in followup.NAMES]
        assert values == record['parameters'] and record['curve'][-1]['update'] == previous['training']['updates']
        vectors.setdefault(seed, {})[arm] = values
    assert tuple(map(int, vectors)) == first.SEEDS and all(tuple(v) == first.ARMS for v in vectors.values())
    return dict(
        scope='Official confirmation with frozen fitted vectors. No gradient steps and no model selection. '
              'Independent starting headings conditional on the fixed probe recipes.',
        frozen_parameters=dict(order=followup.NAMES, fit2_vectors=vectors,
                               baseline='Original hand-set values for each arm, re-scored on the fresh panels'),
        files={name: digest(name) for name in (*previous['files'], 'FIT2_LOCK.json', 'FIT2_RESULTS.json',
                                               'FIT2_CURVES.json', 'score_gains_official.py')},
        panels=dict(seed=PANEL_SEED, generator='NumPy default_rng (PCG64)',
                    scheme='One uniform heading in each of sixteen equal-width sectors per panel. '
                           'Same discrete yaw schedules, goal offsets, horizons and sizes as FIT2.',
                    arrays={name: dict(sha256=first.hash_arrays(pair), values=[a.tolist() for a in pair])
                            for name, pair in make_panels().items()}),
        gates=previous['gates'],
        dark_turn_reference='Native hand-set median on the same fresh dark-turn panel. '
                            'Also report delta versus the historical native hand-set median.',
        ceiling_diagnosis=previous['ceiling_diagnosis'],
    )


def gate(native, rewired, hand, historical_hand, thresholds):
    gap = native['convergence'] - native['combined']
    lead = native['combined'] - rewired['combined']
    dark_delta = native['dark_turn_deg'] - hand['dark_turn_deg']
    gap_pass = gap <= thresholds['native_clamped_minus_combined_at_most']
    compass_pass = native['combined_compass_deg'] < thresholds['native_combined_compass_median_strictly_below_deg']
    lead_pass = lead >= thresholds['min_paired_native_minus_rewired']
    return dict(gap=gap, lead=lead, compass_deg=native['combined_compass_deg'],
                dark_turn_delta_deg=dark_delta,
                dark_turn_delta_vs_historical_hand_set_deg=native['dark_turn_deg'] - historical_hand['dark_turn_deg'],
                dark_turn_tradeoff=dark_delta > 3. + followup.NUMERIC_ATOL,
                gap_pass=gap_pass, compass_pass=compass_pass, lead_pass=lead_pass,
                passed=gap_pass and compass_pass and lead_pass)


def score_all(lock, graph):
    panels = {}
    for name, record in lock['panels']['arrays'].items():
        pair = tuple(np.array(a) for a in record['values'])
        assert first.hash_arrays(pair) == record['sha256']
        panels[name] = pair
    _, loops = followup.get_loops(graph)
    history = first.load(first.OLD)
    params = {arm: first.old_params(history, arm) for arm in first.ARMS}
    hand = {arm: first.score(loops[arm], p, panels) for arm, p in params.items()}
    prior_hand = first.load('FIT2_RESULTS.json')['hand_set']['native']
    traces = {'hand_set': followup.diagnosis(loops, params, hand, panels['steering'])}
    fitted = {}
    for seed, vectors in lock['frozen_parameters']['fit2_vectors'].items():
        params = {arm: first.tensor(p) for arm, p in vectors.items()}
        scores = {arm: first.score(loops[arm], p, panels) for arm, p in params.items()}
        traces[seed] = followup.diagnosis(loops, params, scores, panels['steering'])
        fitted[seed] = dict(scores=scores, gate=gate(*[scores[a] for a in first.ARMS], hand['native'],
                                                   prior_hand, lock['gates']))
    return dict(lock_sha256=digest(LOCK), hand_set=hand, fitted=fitted, clamped_diagnosis=traces,
                all_seeds_passed=all(row['gate']['passed'] for row in fitted.values()))


def report(result):
    seeds = [k for k in result['fitted']]
    count = sum(row['gate']['passed'] for row in result['fitted'].values())
    lines = ['# MiniFly official gain-fit confirmation', '', '## Outcome', '',
             f'**{count}/{len(seeds)} fitting seeds met the revised combined-loop gates on the fresh panels.** '
             'The six gain vectors came from the 2,048-update fit and were frozen before scoring. '
             'The panels use one newly sampled heading per angular sector and preserve the original '
             'yaw schedules, goal offsets, episode lengths and sample sizes. Both wiring arms and both '
             'hand-set references received the same fresh panels.',
             '', '## Five probes', '',
             '| Arm | Seed | Stationary heading | Dark turns | Clamped steering | Clamped convergence | Combined loop |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    groups = {'all': result['hand_set'], **{seed: r['scores'] for seed, r in result['fitted'].items()}}
    for seed, arms in groups.items():
        for arm, s in arms.items():
            label = f'{arm.title()} {"hand-set" if seed == "all" else "fitted"}'
            lines.append(f"| {label} | {seed} | {s['stationary']}/16 | {s['dark_turn_deg']:.2f}° · "
                         f"{s['dark_turn_pass']}/96 | {s['steering']}/96 | {s['convergence']}/96 · "
                         f"{s['convergence_deg']:.2f}° | {s['combined']}/96 · {s['combined_compass_deg']:.2f}° |")
    lines += ['', 'Dark turns report median heading error and qualified count. Clamped convergence reports '
              'successes within 20° and median terminal goal error. Combined reports successes within 20° '
              'and median terminal compass error.',
              '', '## Gates', '',
              '| Seed | Native gap ≤10 | Native compass <15° | Native lead ≥30 | Dark-turn delta vs fresh hand-set | Verdict |',
              '| --- | ---: | ---: | ---: | ---: | --- |']
    for seed, row in result['fitted'].items():
        g = row['gate']
        tradeoff = ' (trade-off)' if g['dark_turn_tradeoff'] else ''
        lines.append(f"| {seed} | {g['gap']} | {g['compass_deg']:.2f}° | {g['lead']:+d} | "
                     f"{g['dark_turn_delta_deg']:+.2f}°{tradeoff} | {'Pass' if g['passed'] else 'Miss'} |")
    lines += ['', 'A dark-turn regression above 3° is a reportable trade-off. The clamped and combined panels '
              'use different offset sets, so the gap is an empirical comparison.',
              '', '## Clamped steering diagnosis', '',
              'The bins show misses out of 16 episodes. The four-second trend classifier uses the locked '
              'criteria from the 2,048-update exercise. Every miss has signed final error and late trend '
              'in the result file.',
              '', '| Arm | Seed | −120° | −65° | −30° | +30° | +65° | +120° | Negative final residual | '
              'Stable wrong | Approaching | Moving away | Unresolved |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for seed in ['hand_set', *seeds]:
        for arm, row in result['clamped_diagnosis'][seed].items():
            n = len(row['misses'])
            negative = sum(m['final_signed_error_deg'] < 0 for m in row['misses'])
            counts = [row['miss_by_offset'][i] for i in (-120, -65, -30, 30, 65, 120)]
            behavior = [row['miss_by_behavior'].get(k, 0) for k in
                        ('stable wrong heading', 'still approaching', 'moving away', 'unresolved')]
            label = f'{arm.title()} {"hand-set" if seed == "hand_set" else "fitted"}'
            cells = [label, 'all' if seed == 'hand_set' else seed, *counts, f'{negative}/{n}', *behavior]
            lines.append('| ' + ' | '.join(map(str, cells)) + ' |')
    lines += ['', '[Frozen protocol and panels](FIT_OFFICIAL_LOCK.json) · '
              '[Scores and individual steering misses](FIT_OFFICIAL_RESULTS.json) · '
              '[2,048-update fitting report](FIT2_REPORT.md)', '']
    (first.ROOT / 'FIT_OFFICIAL_REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    graph = first.read_graph()
    expected = make_lock(graph)
    if args.freeze:
        if (first.ROOT / LOCK).exists():
            raise FileExistsError(LOCK)
        first.save(LOCK, expected)
        return
    lock = first.load(LOCK)
    assert lock == json.loads(json.dumps(expected)), 'Lock mismatch'
    result = score_all(lock, graph)
    first.save('FIT_OFFICIAL_RESULTS.json', result)
    report(result)
    print('Official confirmation. All seeds met revised gates:', result['all_seeds_passed'])


if __name__ == '__main__':
    main()
