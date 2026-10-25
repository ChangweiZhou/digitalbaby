"""Re-reduce immutable compact Recoverability artifacts; no model imports/execution.

Run: python3 PROJECT_DOCUMENTS/MINIFLY_PREDICTIVE_DIAGNOSIS_20261025/recoverability/analyze.py
Only reads git blobs explicitly listed below. No native tensors, checkpoints, heads,
learners, archives or held V147 artifacts are opened. All outputs stay beside this file.
"""
from pathlib import Path
import collections
import hashlib
import json
import math
import resource
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
COMMIT = '9803c3d4b8e5428abc54e263c32daa2cb12d8846'
PREFIX = 'minifly_recoverability_v1_dev_20261022/'
INPUTS = [
    'REPORT.md', 'DETAIL_METRICS.json', 'DIAGNOSTIC_PHASE_ROWS.json',
    'PAYLOAD_EOS_DIAGNOSIS.json', 'SUMMARY.json', 'RESOURCE_LEDGER.json',
    'reduce_details.py', 'diagnose.py', 'rc_common.py', 'run.py', 'artifact_inference.py',
]
METRICS = ('nonempty_exact', 'forced_length_payload_exact', 'bit_accuracy',
           'byte_accuracy', 'early_EOS', 'late_EOS', 'truncation')
PHASES = ('OLD_pass1', 'OLD_pass4', 'OLD_end', 'OLD_rest1',
          'OLD_after_NEW_pass1', 'OLD_after_NEW_pass4', 'OLD_NEW_end',
          'OLD_final', 'OLD_S_NEW', 'NEW_pass1', 'NEW_pass4', 'NEW_end',
          'NEW_final', 'NEW_S_NEW')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def mean(values):
    return statistics.fmean(values)


def group(rows, keys):
    result = collections.defaultdict(list)
    for row in rows:
        result[tuple(row[k] for k in keys)].append(row)
    return result


def units(rows):
    """Average head seeds within a life first; preserve paired worlds as units."""
    life = []
    for key, rr in sorted(group(rows, ('seed', 'life', 'STD', 'readout', 'pool')).items()):
        expected = 1 if key[3] == 'DIRECT' else 3
        assert len(rr) == expected, (key, len(rr))
        life.append(dict(zip(('seed', 'life', 'STD', 'readout', 'pool'), key),
                         **{m: mean(r[m] for r in rr) for m in METRICS}))
    world = []
    for key, rr in sorted(group(life, ('seed', 'STD', 'readout', 'pool')).items()):
        assert len(rr) == 2, (key, len(rr))
        world.append(dict(zip(('seed', 'STD', 'readout', 'pool'), key),
                          **{m: mean(r[m] for r in rr) for m in METRICS}))
    return life, world


def aggregate(world):
    result = []
    for key, rr in sorted(group(world, ('STD', 'readout', 'pool')).items()):
        assert len(rr) == 4
        result.append(dict(zip(('STD', 'readout', 'pool'), key), worlds=4,
                           **{m: mean(r[m] for r in rr) for m in METRICS}))
    return result


def contrast(world, kind, left, right):
    """Signed left-minus-right at matched world; no independent-query inference."""
    result = []
    for std in (False, True):
        for view in ('DIRECT', 'MEAN', 'SEP'):
            paired = []
            for seed in range(202610229301, 202610229305):
                select = lambda pool: next(r for r in world if r['seed'] == seed and
                    r['STD'] == std and r['readout'] == view and r['pool'] == pool)
                l, r = select(left), select(right)
                paired.append(dict(world=seed, **{m: l[m]-r[m] for m in METRICS}))
            result.append(dict(kind=kind, STD=std, readout=view, left=left, right=right,
                world_differences=paired, **{m: mean(r[m] for r in paired) for m in METRICS}))
    return result


def main():
    start_cpu, start_wall = time.process_time(), time.monotonic()
    provenance, objects = [], {}
    for name in INPUTS:
        path = PREFIX + name
        blob = git('show', f'{COMMIT}:{path}')
        oid = git('rev-parse', f'{COMMIT}:{path}').decode().strip()
        # Independently check the git object identifier and plain-content digest.
        assert hashlib.sha1(f'blob {len(blob)}\0'.encode()+blob).hexdigest() == oid
        objects[name] = json.loads(blob) if name.endswith('.json') else blob.decode()
        provenance.append(dict(path=path, git_blob_oid=oid, bytes=len(blob),
                               sha256=hashlib.sha256(blob).hexdigest()))
    detail = objects['DETAIL_METRICS.json']
    rows = detail['rows']
    for r in rows:
        for m in METRICS:
            assert math.isfinite(r[m]) and 0 <= r[m] <= 1, (m, r[m])
    # Assessment only. Development episodes never join this reduction.
    selected = [r for r in rows if 202610229301 <= r['seed'] <= 202610229304
                and r['arm'] == 'W' and r['lesion'] == 'NONE']
    assert len(selected) == 1568
    identity_keys = ('seed', 'life', 'STD', 'arm', 'pool', 'readout', 'head_seed', 'lesion')
    assert len({tuple(r[k] for k in identity_keys) for r in rows}) == len(rows)
    assert set(r['pool'] for r in selected) == set(PHASES)
    life, world = units(selected)
    avg = aggregate(world)
    idx = {(r['STD'], r['readout'], r['pool']): r for r in avg}
    # Separate compact diagnostics were produced by a different reduction path.
    dx = objects['PAYLOAD_EOS_DIAGNOSIS.json']['rows']
    diagnostic_checks = 0
    for r in selected:
        if r['pool'] not in ('OLD_end', 'OLD_final', 'NEW_final', 'OLD_S_NEW'):
            continue
        other = next(x for x in dx if x['pair'] == r['seed'] and x['life'] == r['life']
            and bool(x['STD']) == r['STD'] and x['view'] == r['readout']
            and x['head_seed'] == r['head_seed'] and x['pool'] == r['pool'])
        assert abs(other['payload_exact_nonempty']-r['forced_length_payload_exact']) < 1e-14
        assert abs(other['bit_accuracy']-r['bit_accuracy']) < 1e-14
        diagnostic_checks += 1
    # Phase summaries used identical assessment W/NONE aggregate weighting.
    summary_checks = 0
    for r in objects['DIAGNOSTIC_PHASE_ROWS.json']:
        head = idx[bool(r['STD']), r['view'], r['pool']]
        direct = idx[bool(r['STD']), 'DIRECT', r['pool']]
        assert abs(r['head_nonempty_exact']-head['nonempty_exact']) < 1e-14
        assert abs(r['head_bits']-head['bit_accuracy']) < 1e-14
        assert abs(r['direct_nonempty_exact']-direct['nonempty_exact']) < 1e-14
        summary_checks += 1
    differences = []
    for args in [('NEW_write_effect', 'OLD_final', 'OLD_S_NEW'),
                 ('total_OLD_change', 'OLD_final', 'OLD_end'),
                 ('matched_no_NEW_write_OLD_change', 'OLD_S_NEW', 'OLD_end'),
                 ('first_rest_OLD_change', 'OLD_rest1', 'OLD_end'),
                 ('late_rest_OLD_change', 'OLD_final', 'OLD_NEW_end'),
                 ('NEW_vs_unwritten_NEW', 'NEW_final', 'NEW_S_NEW')]:
        differences.extend(contrast(world, *args))
    # Learned joint heads see the same native response record as DIRECT.
    head_differences = []
    for std in (False, True):
        for view in ('MEAN', 'SEP'):
            for pool in ('OLD_end', 'OLD_final', 'OLD_S_NEW', 'NEW_final'):
                p = []
                for seed in range(202610229301, 202610229305):
                    get = lambda ro: next(r for r in world if r['seed'] == seed and
                        r['STD'] == std and r['readout'] == ro and r['pool'] == pool)
                    h, d = get(view), get('DIRECT')
                    p.append(dict(world=seed, **{m: h[m]-d[m] for m in METRICS}))
                head_differences.append(dict(STD=std, head=view, pool=pool,
                    sign='joint_head minus DIRECT on same native responses',
                    world_differences=p, **{m: mean(r[m] for r in p) for m in METRICS}))
    # Neither readout seeds nor facts create additional independent native lives.
    zero_variance = []
    for std in (False, True):
        rr = [r for r in world if r['STD'] == std and r['readout'] == 'DIRECT' and r['pool'] == 'OLD_final']
        for m in ('nonempty_exact', 'forced_length_payload_exact', 'bit_accuracy'):
            values = [r[m] for r in rr]
            zero_variance.append(dict(STD=std, outcome=m, values=values,
                sample_variance=statistics.variance(values)))
    result = dict(
        schema='RECOVERABILITY_BOUNDED_RETROSPECTIVE_REDUCTION_V1',
        source_commit=COMMIT, status='COMPLETE_ARTIFACT_ONLY_E1',
        evidence_level='E1 taught-association diagnosis; no new learning/generalization evidence',
        source_rows=len(rows), assessment_W_NONE_rows=len(selected),
        assessment_worlds=4, correlated_lives_per_world=2, head_seeds_per_view=3,
        denominators=dict(facts_per_cohort_life=18, nonempty_facts_per_cohort_life=16,
            payload_bytes_per_cohort_life=72, payload_bits_per_cohort_life=576,
            total_assessment_nonempty_facts_per_cohort=128,
            total_assessment_payload_bits_per_cohort=4608,
            note='Complete and forced-length payload rates share 16 nonempty facts; bits weight the 576 actual payload bits. Head variants reuse responses.'),
        weighting='head seeds within life, two lives within world, four paired worlds equal; all same-native comparisons preserve this hierarchy',
        checks=dict(compact_payload_diagnostic_rows_crosschecked=diagnostic_checks,
            phase_summaries_crosschecked=summary_checks,
            duplicate_rows=False, source_blob_oid_and_sha_verified=True),
        phase_aggregates=avg, world_phase_rows=world,
        signed_phase_differences=differences, same_native_response_joint_head_effects=head_differences,
        outcome_variance_diagnostic=zero_variance,
        available_early_information=dict(OLD_passes=[1,4,14], OLD_after_NEW_passes=[1,4,14],
            metrics=list(METRICS), native_margin_values_in_read_artifacts=False,
            individual_teaching_events=False, address_overlap_values=False,
            early_training_head_probes=False),
        limits=[
            'No S_OLD branch: N disables OLD and NEW writes and is not a later OLD-episode lesion.',
            'Signed matched W-S_NEW identifies the net NEW-write effect for this fixed branch, including all downstream state consequences, not a per-write linear operator.',
            'Native response tensors are tracked historically but intentionally not read or restored; compact summaries do not expose raw margins or query-wise flips.',
            'Marginal accuracy changes are net changes, not counts of damaged versus rescued individual bits.',
            'Four worlds are the grouped units; heads/lives/queries are correlated. No confidence or causal-generalization claim from treating facts as independent.',
            'All worlds have the same time and write-dose schedule, so time-only and write-count predictors have no cross-world dose variation.',
            'Final complete accuracy is at floor; STD direct final OLD payload is also constant zero across worlds, precluding useful discrimination on those endpoints.',
            'This is retrospective evidence already used to construct hypotheses. No prospective holdout, no tuning, and no claim of predictive validation.',
            'A key lookup table can solve the reinforced facts; this evidence does not establish E3 unseen addition or reliable arbitrary-input memory service.',
        ],
        new_exposure=dict(native_teaching_records=0, native_events=0, head_fits=0,
            model_inference=0, training_trajectory_replays=0, V147_held_scores=0),
        inputs=provenance,
    )
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    result['analysis_cost'] = dict(process_cpu_seconds=time.process_time()-start_cpu,
        git_read_child_cpu_seconds=child.ru_utime+child.ru_stime,
        active_wall_seconds=time.monotonic()-start_wall,
        note='Includes git-show/rev-parse reads separately; excludes interactive source inspection/preparation and output serialization. Reduction costs, not native learning costs.')
    (OUT/'RESULTS.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], diagnostic_rows=diagnostic_checks,
        phase_checks=summary_checks, source_commit=COMMIT,
        cpu_seconds=result['analysis_cost']['process_cpu_seconds']), indent=2))


if __name__ == '__main__':
    main()
