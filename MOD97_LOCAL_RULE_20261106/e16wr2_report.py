"""Score exact-job receipts and verify a completed E16-WR2 campaign."""

import csv
import io
from datetime import datetime, timezone
from fractions import Fraction
from statistics import mean

from e16wr2_common import OUT, ROOT, SEEDS, read, save, sha, validate_receipt, verify_lock


PAIRS = ('+-', '-+', '><', '<>')
OPERATIONS = ('add', 'sub', 'max', 'min')
METRICS = ('exact', 'final', 'intermediate')
EXPECTED_VERDICTS = {'P1': 3, 'P2': 3, 'P3': 6, 'P4': 3, 'P5': 3}
FINISHED = 'RUNS_COMPLETE_VERIFICATION_PENDING'


def percent(score, metric='exact'):
    return Fraction(100 * score[metric], score['n'])


def held(stage, metric='final'):
    return mean(percent(stage['scores']['compose'][pair], metric) for pair in PAIRS)


def operations(stage, moduli):
    return mean(percent(stage['scores'][f'{op}{mod}']) for mod in moduli for op in OPERATIONS)


def compose(receipt):
    stages = [stage for stage in receipt['stages'] if stage['task'] == 'compose']
    assert len(stages) == 1
    return stages[0]


def summarize(receipt):
    acquired = compose(receipt)
    final = receipt['stages'][-1]
    weakest = min(PAIRS[:2], key=lambda pair: percent(final['scores']['compose'][pair], 'final'))
    operation_scores = {f'{op}{mod}': percent(final['scores'][f'{op}{mod}'])
                        for mod in (97, 73) for op in OPERATIONS}
    return dict(
        post_compose=held(acquired), final_held=held(final), held_loss=held(acquired) - held(final),
        held_full=held(final, 'exact'), old97=operations(final, (97,)),
        operations8=operations(final, (97, 73)),
        weakest_carry=weakest, weakest_carry_final=percent(final['scores']['compose'][weakest], 'final'),
        operation_scores=operation_scores,
    )


def markdown_table(headers, rows):
    def line(values):
        return '| ' + ' | '.join(map(str, values)) + ' |'

    return '\n'.join([line(headers), line(['---'] * len(headers)), *(line(row) for row in rows)])


def write_csv(name, headers, rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)
    (ROOT / name).write_text(output.getvalue())


def score_verdicts(summaries, controls):
    verdicts = []
    for key, row in summaries.items():
        seed = row['seed']
        if row['role'] == 'A':
            if row['capacity'] == 65536:
                verdicts.append(dict(prediction='P1', key=key, seed=seed,
                                     passed=row['held_loss'] <= 8 and row['old97'] >= 93,
                                     held_loss_pp=float(row['held_loss']), final_mod97=float(row['old97'])))
            else:
                reference = controls[('WR_observed', seed)]
                held_delta = row['final_held'] - reference['final_held']
                old97_delta = row['old97'] - reference['old97']
                verdicts.append(dict(prediction='P2', key=key, seed=seed,
                                     passed=abs(held_delta) <= 5 and abs(old97_delta) <= 5,
                                     final_held_delta_pp=float(held_delta), final_mod97_delta_pp=float(old97_delta)))
            delta = row['post_compose'] - controls[('W_uniform', seed)]['post_compose']
            verdicts.append(dict(prediction='P3', key=key, seed=seed, passed=abs(delta) <= 3,
                                 post_compose_delta_pp=float(delta)))
        else:
            reference = controls[('E13_A1', seed)]
            operations_delta = row['operations8'] - reference['operations8']
            held_delta = row['final_held'] - reference['final_held']
            verdicts.append(dict(prediction='P4', key=key, seed=seed, passed=abs(operations_delta) <= 2,
                                 final_operations_delta_pp=float(operations_delta)))
            verdicts.append(dict(prediction='P5', key=key, seed=seed, passed=0 <= held_delta <= 5,
                                 final_held_delta_pp=float(held_delta)))
    return verdicts


def collect(lock, state, final):
    controls = {}
    for control in lock['reused_controls']:
        path = OUT / control['receipt']
        assert sha(path) == control['sha256'], path
        receipt = read(path)
        assert receipt['seed'] == control['seed'], path
        controls[control['role'], control['seed']] = summarize(receipt)

    snapshots, receipts, summaries, costs = {}, {}, {}, {}
    for expected in lock['jobs']:
        key = expected['key']
        path = OUT / f'E16WR2_JOB_{key}.json'
        if not path.exists():
            continue
        job = read(path)
        if job['status'] != 'complete':
            continue
        for field in ('key', 'seed', 'role', 'capacity', 'env', 'receipt', 'metadata'):
            assert job[field] == expected[field], (key, field)
        assert job['command'] == [lock['runtime']['executable'], 'e16wr2_worker.py', key], key
        assert job['returncode'] == 0, key
        assert sha(OUT / job['receipt']) == job['receipt_sha256'], key
        receipt = validate_receipt(job)
        meta = read(OUT / job['metadata'])
        assert meta['lock_sha256'] == sha(ROOT / 'E16WR2_LOCK.json'), key
        assert meta['updates'] == 57344 and meta['parameters'] == 470801, key
        assert meta['gate']['active_other_producers'] == [], key
        assert meta['python_version'] == lock['runtime']['python_version'], key
        assert meta['torch_version'] == lock['runtime']['torch_distribution'], key
        sidecar = read((OUT / job['receipt']).with_suffix('.replay.json'))
        assert len(sidecar) == 9 and all(row['wrong_against_observed'] == 0 for row in sidecar), key
        snapshots[key] = sidecar
        receipts[key] = receipt
        summaries[key] = dict(seed=job['seed'], role=job['role'], capacity=job['capacity'], **summarize(receipt))
        costs[key] = dict(life_cpu_seconds=meta['process_cpu_seconds'], wall_seconds=job['wall_seconds'],
                          peak_rss=meta['peak_rss_raw'], peak_rss_units=meta['peak_rss_units'],
                          raw_output_bytes=sum(artifact.stat().st_size for artifact in (
                              OUT / job['receipt'], OUT / job['metadata'],
                              (OUT / job['receipt']).with_suffix('.replay.json'),
                              OUT / 'logs' / f'e16wr2_{key}.log', path,
                          )))
    if final:
        assert len(receipts) == len(lock['jobs']) == 9
        assert read(ROOT / 'E16WR2_CHECKS.json')['status'] == 'PASS'
        assert read(ROOT / 'E16WR2_TECHNICAL_JOB.json')['returncode'] == 0
        assert state['source_lock_sha256'] == sha(ROOT / 'E16WR2_LOCK.json')
        assert set(receipts) == {job['key'] for job in state['jobs'] if job['status'] == 'complete'}
        for job in state['jobs']:
            assert job == read(OUT / f'E16WR2_JOB_{job["key"]}.json'), job['key']
        for name, digest in lock['source_hashes'].items():
            assert sha(ROOT / 'E16WR2_SOURCE_SNAPSHOT' / name) == digest, name
    return controls, receipts, summaries, snapshots, costs


def export_curves(receipts, summaries):
    curves, scores = [], []
    for key, receipt in receipts.items():
        identity = {field: summaries[key][field] for field in ('role', 'capacity', 'seed')}
        for stage in receipt['stages']:
            base = dict(key=key, **identity, stage=stage['stage'], task=stage['task'])
            has_compose = 'compose' in stage['scores']
            curves.append(dict(**base, held_final=float(held(stage)) if has_compose else None,
                               held_full=float(held(stage, 'exact')) if has_compose else None,
                               mod97=float(operations(stage, (97,))), mod73=float(operations(stage, (73,))),
                               operations8=float(operations(stage, (97, 73)))))
            for panel, values in stage['scores'].items():
                subpanels = values if panel == 'compose' else {panel: values}
                for subpanel, value in subpanels.items():
                    for metric in METRICS:
                        if value.get(metric) is not None:
                            scores.append(dict(**base, panel=panel, subpanel=subpanel, metric=metric,
                                               correct=value[metric], n=value['n'], percent=float(percent(value, metric))))
    prefix = ('key', 'role', 'capacity', 'seed', 'stage', 'task')
    write_csv('E16WR2_CURVES.csv', (*prefix, 'held_final', 'held_full', 'mod97', 'mod73', 'operations8'), curves)
    write_csv('E16WR2_ALL_SCORES.csv', (*prefix, 'panel', 'subpanel', 'metric', 'correct', 'n', 'percent'), scores)
    return curves


def export_exposures(snapshots):
    rows = []
    for key, stages in snapshots.items():
        for stage in stages:
            row = {name: value for name, value in stage.items() if name not in ('origin_counts', 'realized_draw_counts')}
            rows.append(dict(key=key, **row, **{
                f'{kind}_stage_{index}': stage[field].get(str(index), 0)
                for kind, field in (('retained', 'origin_counts'), ('drawn', 'realized_draw_counts'))
                for index in range(1, 10)
            }))
    headers = ('key', 'stage', 'task', 'current_presentations', 'presentations_total', 'retained_slots',
               'distinct_prompts', 'replay_count', 'wrong_against_observed', 'transient_pending_records_peak')
    origins = tuple(f'{kind}_stage_{index}' for kind in ('retained', 'drawn') for index in range(1, 10))
    write_csv('E16WR2_EXPOSURES.csv', (*headers, *origins), rows)


def plot_curves(curves):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    for column, seed in enumerate(SEEDS):
        axes[0, column].set_title(f'Seed {seed}')
        for cap in (16384, 65536):
            rows = [row for row in curves if row['key'] == f'A{cap}_{seed}']
            for index, metric in enumerate(('held_final', 'mod97')):
                axis = axes[index, column]
                axis.plot([row['stage'] for row in rows], [row[metric] for row in rows],
                          marker='.', label=f'Total {cap:,}')
        rows = [row for row in curves if row['key'] == f'B_{seed}']
        axes[2, column].plot([row['stage'] for row in rows], [row['operations8'] for row in rows],
                             marker='.', label='Eight operations')
        axes[2, column].scatter([rows[-1]['stage']], [rows[-1]['held_final']], label='Held composition')
        axes[2, column].set_xlabel('Completed training stage')
        for row in range(3):
            axis = axes[row, column]
            axis.axvline(5 if row < 2 else 9, color='gray', linestyle=':', linewidth=1)
            axis.set_ylim(0, 100)
            if column == 2:
                axis.legend(fontsize=7, loc='best')
    for axis, label in zip(axes[:, 0], ('Part A held composition (%)', 'Part A mod-97 operations (%)', 'Part B accuracy (%)')):
        axis.set_ylabel(label)
    fig.tight_layout()
    fig.savefig(ROOT / 'E16WR2_ACCURACY_CURVES.png', dpi=170)
    plt.close(fig)


def main():
    lock = verify_lock()
    status_path = ROOT / 'E16WR2_STATUS.json'
    state = read(status_path) if status_path.exists() else {'status': 'PREPARED'}
    final = state['status'] == FINISHED
    controls, receipts, summaries, snapshots, costs = collect(lock, state, final)
    verdicts = score_verdicts(summaries, controls)
    curves = export_curves(receipts, summaries)
    export_exposures(snapshots)
    save(ROOT / 'E16WR2_PREDICTION_VERDICTS.json', verdicts)
    save(ROOT / 'E16WR2_EXPOSURES.json', snapshots)
    save(ROOT / 'E16WR2_COSTS.json', dict(
        workers=costs, campaign_reaped_child_cpu_seconds=state.get('reaped_child_cpu_seconds'),
        campaign_wall_seconds=state.get('wall_seconds'), controller_cpu_seconds=state.get('controller_cpu_seconds'),
        technical_job=read(ROOT / 'E16WR2_TECHNICAL_JOB.json') if (ROOT / 'E16WR2_TECHNICAL_JOB.json').exists() else None,
    ))
    verdict_counts = {
        name: dict(passes=sum(v['passed'] for v in verdicts if v['prediction'] == name),
                   evaluated=sum(v['prediction'] == name for v in verdicts), expected=expected)
        for name, expected in EXPECTED_VERDICTS.items()
    }
    endpoint_table = markdown_table(
        ('Job', 'Post-compose held %', 'Final held %', 'Loss pp', 'Final held full %',
         'Final mod-97 %', 'Final eight operations %', 'Weakest carry', 'Carry final %'),
        ([key, *('—' if field == 'held_loss' and row['role'] == 'B' else f'{float(row[field]):.4f}'
                 for field in ('post_compose', 'final_held', 'held_loss',
                'held_full', 'old97', 'operations8')), row['weakest_carry'], f'{float(row["weakest_carry_final"]):.4f}']
         for key, row in summaries.items()),
    )
    verdict_table = markdown_table(('Prediction', 'Passing', 'Evaluated', 'Expected'), (
        [name, row['passes'], row['evaluated'], row['expected']] for name, row in verdict_counts.items()
    ))
    operation_names = [f'{op}{mod}' for mod in (97, 73) for op in OPERATIONS]
    b_deltas = markdown_table(('Seed', *operation_names), (
        [row['seed'], *(f'{float(row["operation_scores"][op] - controls[("E13_A1", row["seed"])]["operation_scores"][op]):+.4f}'
                        for op in operation_names)]
        for row in summaries.values() if row['role'] == 'B'
    ))
    lines = [
        '# E16-WR2 — observed replay targets and total capacity', '',
        f'{len(receipts)} of 9 scientific lives have validated receipts.' if final else
        f'{len(receipts)} of 9 scientific lives have validated receipts. Controller status is `{state["status"]}`.', '',
        'Part A uses one total reservoir of 16,384 or 65,536 slots. Replay samples uniformly from the retained observations, with origin-stage tags recorded for diagnosis. Part B retains the original 1,024-per-stage sampling schedule. All arms replay saved observed answer bytes at a one-half share after the first stage and use the sign rule, decay 0.25, batch 64 and the three standing seeds.', '',
        endpoint_table, '', verdict_table, '',
        markdown_table(('Prediction', 'Job', 'Pass', 'Measurements'), (
            [v['prediction'], v['key'], v['passed'], ', '.join(f'{field} {value:.4f}' for field, value in v.items()
                                                             if field not in ('prediction', 'key', 'seed', 'passed'))]
            for v in verdicts
        )), '',
        'P1 requires held loss ≤ 8 points and final mod-97 ≥ 93%. P2 requires both final endpoints within 5 points of the prior per-stage observed arm. P3 uses a two-sided 3-point acquisition band. P4 uses a two-sided 2-point band for the mean of eight operations. P5 requires a held-composition gain in [0, 5] points. Verdicts use exact panel counts before display rounding.', '',
        'Part B final operation accuracy differences from E13-A1, in percentage points.', '', b_deltas, '',
        '## Reused controls', '',
        markdown_table(('Role', 'Seed', 'Post-compose held %', 'Final held %', 'Final mod-97 %', 'Final eight operations %'), (
            [role, seed, *(f'{float(row[field]):.4f}' for field in ('post_compose', 'final_held', 'old97', 'operations8'))]
            for (role, seed), row in sorted(controls.items())
        )), '',
        'The score CSV records every stage and scored panel, including individual operations, held pairs, within-pair scores and intermediate and full scratchpad accuracy. The exposure ledger records retained slots, distinct prompts, transient pending observations and realized draws by origin stage. Replay target agreement measures byte equality with the stored observed answer. [Score CSV](E16WR2_ALL_SCORES.csv) · [Exposure CSV](E16WR2_EXPOSURES.csv) · [Exposure ledger](E16WR2_EXPOSURES.json)', '',
        'The equal-numeric-capacity P2 comparison changes both retained memory and sampling. The earlier per-stage arm could retain 147,456 slots across nine stages. Pending observations add transient memory to the new total reservoir before they become eligible at the stage boundary. These synthetic tasks use a staged training schedule and the existing Store. The cell and window results use different schedules, representations and tasks.', '',
        'Each life has 470,801 trainable weights and the existing 65,536-key Store. Worker RSS includes the independent observation map used to check replay targets. The historical comparators, births and WR full-default reproduction are reused. WR2 budgets three short qualification cycles and zero additional full-profile lives. The locked configuration saves scores, traces and provenance, with final model weight export disabled. [Cost ledger](E16WR2_COSTS.json)',
    ]
    if final:
        assert all(row['evaluated'] == row['expected'] for row in verdict_counts.values())
        total_cpu = state['reaped_child_cpu_seconds']
        wall = state['wall_seconds']
        tech = read(ROOT / 'E16WR2_TECHNICAL_JOB.json')
        checked_targets = sum(stage['replay_count'] for stages in snapshots.values() for stage in stages)
        largest = [row for row in summaries.values() if row['role'] == 'A' and row['capacity'] == 65536]
        smaller = {row['seed']: row for row in summaries.values() if row['role'] == 'A' and row['capacity'] == 16384}
        paired_gains = [float(row['final_held'] - smaller[row['seed']]['final_held']) for row in largest]
        distinct = [stages[-1]['distinct_prompts'] for key, stages in snapshots.items() if key.startswith('A65536_')]
        peak_pending = max(stage['transient_pending_records_peak'] for key, stages in snapshots.items()
                           if summaries[key]['role'] == 'A' for stage in stages)
        losses = ' / '.join(f'{float(row["held_loss"]):.2f}' for row in sorted(largest, key=lambda row: row['seed']))
        operation_results = ' / '.join(f'{float(row["old97"]):.2f}%' for row in sorted(largest, key=lambda row: row['seed']))
        b_rows = [row for row in verdicts if row['prediction'] == 'P5']
        b_gains = ' / '.join(f'{row["final_held_delta_pp"]:+.2f}' for row in sorted(b_rows, key=lambda row: row['seed']))
        findings = [
            f'Increasing the total reservoir to 65,536 slots raised final held-composition accuracy by {min(paired_gains):.2f}–{max(paired_gains):.2f} percentage points across the three seeds. It still lost {losses} points after acquisition, in seed order. Final mod-97 accuracy was {operation_results}. P1 met both thresholds in {verdict_counts["P1"]["passes"]} of 3 seeds.',
            f'At 16,384 total slots, {verdict_counts["P2"]["passes"]} of 3 seeds met both endpoint bands against the prior per-stage reservoir. Across the two Part A capacities, {verdict_counts["P3"]["passes"]} of 6 lives met the acquisition band against E16-W uniform replay. The larger reservoir finished with {min(distinct):,}–{max(distinct):,} distinct prompts. In both capacity arms the stage-end design held up to {peak_pending:,} pending records in addition to the retained reservoir.',
            f'Part B reproduced the E8 demonstration with observed answers. It changed final held-composition accuracy by {b_gains} points relative to E13-A1. The eight-operation means met P4 in {verdict_counts["P4"]["passes"]} of 3 seeds. Composition is taught last in Part B, so the endpoint measures acquisition and retention of prior operations. All three full Part B input streams matched prompt hashes independently reconstructed from the original schedule and sampler. The archived E13-A1 runs supply scores, and their original prompt streams were not recorded.',
        ]
        lines[4:4] = [item for finding in findings for item in (finding, '')]
        lines += ['', f'All {checked_targets:,} replay draws across the nine lives matched previously observed answer bytes.', '',
                  f'The campaign took {wall / 60:.2f} minutes of controller wall time and {total_cpu / 3600:.4f} CPU-hours across reaped child processes.', '',
                  f'The technical cycles used {tech["process_tree_cpu_seconds"]:.2f} child CPU seconds and {tech["wall_seconds"]:.2f} wall seconds. Scientific worker timers sum to {sum(row["life_cpu_seconds"] for row in costs.values()) / 3600:.4f} CPU-hours and exclude imports and metadata writing. Controller CPU was {state["controller_cpu_seconds"]:.2f} seconds. The raw scientific receipt, metadata, replay ledger, job and log files total {sum(row["raw_output_bytes"] for row in costs.values()):,} bytes.', '',
                  markdown_table(('Job', 'Worker CPU seconds', 'Job wall seconds', 'Peak RSS', 'RSS units'), (
                      [key, f'{row["life_cpu_seconds"]:.2f}', f'{row["wall_seconds"]:.2f}',
                       f'{row["peak_rss"]:,}', row['peak_rss_units']] for key, row in costs.items()
                  )), '',
                  '![Held composition and operation accuracy for each seed](E16WR2_ACCURACY_CURVES.png)']
        plot_curves(curves)
        review = ['# 9.40 — E16-WR2 observed replay', '',
                  *[item for finding in findings for item in (finding, '')],
                  endpoint_table, '', verdict_table, '',
                  f'The campaign used {total_cpu / 3600:.4f} child CPU-hours and {wall / 60:.2f} minutes of wall time. The total and per-stage arms differ in retained capacity as well as sampling. The observed targets agree byte for byte with the saved observations.', '',
                  'See [the full report](E16WR2_REPORT.md) and [verification receipt](E16WR2_VERIFICATION.json).']
        (ROOT / 'E16WR2_REVIEW_SECTION_9_40.md').write_text('\n'.join(review) + '\n')
    (ROOT / 'E16WR2_REPORT.md').write_text('\n'.join(lines) + '\n')
    if final:
        save(ROOT / 'E16WR2_VERIFICATION.json', dict(
            status='PASS', verified_at=datetime.now(timezone.utc).isoformat(),
            scientific_lives=9, all_observed_returncodes_zero=True,
            exact_job_receipts_and_locked_inputs_verified=True,
            locked_source_snapshot_hashes_verified=True, all_updates_57344=True,
            full_part_b_prompt_hashes_match_reconstructed_reference=True,
            observed_replay_targets_checked_against_saved_observations=checked_targets,
            observed_replay_targets_wrong_against_saved_observations=0,
            prediction_counts=verdict_counts, receipt_hashes={
                job['key']: job['receipt_sha256'] for job in state['jobs'] if job['status'] == 'complete'
            },
        ))
    print(f'Validated {len(receipts)}/9 scientific lives. Report updated.')


if __name__ == '__main__':
    main()
