"""Finalizer tests: synthetic fixtures and locked world-190000 receipts only.

No science score reduction, analyzer import, learner execution or production publication.
"""
import ast
import copy
import gzip
import json
import math
import sys
import zipfile
from pathlib import Path

import pytest
import finalize_a_v3 as final

ROOT = final.ROOT
sys.path.insert(0, str(ROOT / 'src'))
import paths
from fixture import make_world


@pytest.fixture(scope='module')
def technical():
    import numpy as np
    with np.load(ROOT / 'package/FULL151_CANONICAL_B.npz') as anchor:
        support = [final.sha(np.ascontiguousarray(anchor[k]).tobytes()) for k in ('indices', 'indptr')]
    receipts = {a: json.loads(gzip.decompress((ROOT / f'results/technical_final/{a}/190000.json.gz').read_bytes()))
                for a in final.ARMS}
    return receipts, make_world(190000), support


def validate_technical(r, fixture, support):
    # Technical receipts predate SOURCE_LOCK creation; their own saved source map
    # is passed only as test fixture context, then deliberately changed below.
    return final.validate_receipt(r, r['arm'], 190000, fixture, r['source_sha256'], None,
                                  support, kind='technical_final')


@pytest.mark.parametrize('arm', final.ARMS)
def test_all_frozen_technical_receipts_pass_new_arithmetic(technical, arm):
    receipts, fixture, support = technical
    result = validate_technical(receipts[arm], fixture, support)
    assert sum(map(len, result.values())) == 1380
    metrics = final.reduce_bins(result)
    assert set(metrics) == set(final.SELECTORS) | {'E1', 'E3'}
    assert metrics['E1'] == metrics['old_fact_final_W'] - metrics['old_fact_final_N']
    assert metrics['E3'] == metrics['heldout_final_W'] - metrics['heldout_final_N_old_rel']


@pytest.mark.parametrize('mutation,fragment', [
    ('answer', 'tampered emitted'), ('target', 'probe target'), ('correct', 'correctness flag'),
    ('branch', 'branch'), ('world', 'world'), ('dropped_probe', 'dropped/extra'),
    ('graph', 'graph'), ('birth_graph', 'wrong graph'), ('support', 'graph support'),
    ('source', 'source'), ('first_order', 'first-response'), ('first_time', 'teacher timing'),
    ('first_answer', 'first-response'), ('option_score', 'relation score'),
    ('nonfinite_value', 'nonfinite'),
])
def test_required_tampering_is_rejected(technical, mutation, fragment):
    receipts, fixture, support = technical
    base = receipts['P1']
    receipt = copy.deepcopy(base)
    row = receipt['probes'][0]
    held = next(r for r in receipt['probes'] if r['set'] == 'old_relation_heldout')
    if mutation == 'answer': row['emitted'] = 48 + (row['emitted'] - 48 + 1) % 4
    elif mutation == 'target': row['target'] = 48 + (row['target'] - 48 + 1) % 4
    elif mutation == 'correct': row['correct'] = 1 - row['correct']
    elif mutation == 'branch': row['branch'] = 'N_old_fact'
    elif mutation == 'world': receipt['world'] = 190001
    elif mutation == 'dropped_probe': receipt['probes'].pop()
    elif mutation == 'graph': receipt['canonical_B_sha256'] = '0' * 64
    elif mutation == 'birth_graph': receipt['births'][0]['canonical_B_sha256'] = '0' * 64
    elif mutation == 'support': receipt['params']['support_sha256'][0] = '0' * 64
    elif mutation == 'source': receipt['source_sha256']['src/analyze_a.py'] = '0' * 64
    elif mutation == 'first_order': receipt['first'][0]['branch'] = 'N_old_fact'
    elif mutation == 'first_time': receipt['first'][0]['teacher_at'] += 1
    elif mutation == 'first_answer': receipt['first'][0]['emitted'] = 99
    elif mutation == 'option_score': held['option_scores'][0] += .01
    elif mutation == 'nonfinite_value': row['values'][0] = float('nan')
    with pytest.raises(final.FinalAuditError, match=fragment):
        final.validate_receipt(receipt, 'P1', 190000, fixture, base['source_sha256'], None,
                               support, kind='technical_final')


def test_answer_plus_correct_flag_still_rejected(technical):
    receipts, fixture, support = technical
    receipt = copy.deepcopy(receipts['P1'])
    row = receipt['probes'][0]
    row['emitted'] = 48 + (row['emitted'] - 48 + 1) % 4
    row['correct'] = int(row['emitted'] == row['target'])
    with pytest.raises(final.FinalAuditError, match='tampered emitted'):
        validate_technical(receipt, fixture, support)


def test_roster_constructor_is_independent_and_complete(technical):
    _, fixture, _ = technical
    rows = list(final.expected_probes(fixture))
    assert len(rows) == 1380
    for branch in final.BRANCHES:
        for stage, n in [('old_end', 34), ('old_day', 34), ('new_end', 56), ('final', 152)]:
            assert sum(r['branch'] == branch and r['stage'] == stage for r in rows) == n
    held = [r for r in rows if r['branch'] == 'W' and r['stage'] == 'final' and r['set'] == 'old_relation_heldout']
    assert [r['target'] for r in held] == [76, 82] * 3
    source = (ROOT / 'ops/finalize_a_v3.py').read_text()
    tree = ast.parse(source)
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not set(imports) & {'analyze_a', 'runner', 'systems', 'stores', 'common_platform', 'audit_round'}


def test_exposure_and_countermodel_checks(technical):
    result = final.fixture_evidence(technical[1])
    assert result['pass'] and len(result['exposure_records']) == 600
    facts = result['fact_address_countermodels']
    for cohort in ('old', 'new'):
        assert facts[cohort + '_canonical']['exact_table_accuracy'] == 1
        assert facts[cohort + '_spacing']['content_table_accuracy'] == 1
        assert facts[cohort + '_prefix']['content_table_accuracy'] == 1
        assert facts[cohort + '_inner_marker']['content_key_hits'] == 0
        assert facts[cohort + '_inner_marker']['content_table_accuracy'] == .25
    assert result['heldout']['exact_taught_pair_hits'] == 0
    assert result['heldout']['learned_first_symbol_class_accuracy'] == 1
    assert result['heldout']['constant_option_L_accuracy'] == .5


def test_hand_computed_score_arithmetic():
    bins = {}
    for key, selector in final.SELECTORS.items():
        n = 16 if 'fact' in selector[2] else 12 if selector[2] == 'old_relation_taught' else 6
        bins[selector] = [1] * (n // 2) + [0] * (n - n // 2)
    bins[final.SELECTORS['old_fact_final_W']] = [1] * 12 + [0] * 4
    bins[final.SELECTORS['old_fact_final_N']] = [1] * 4 + [0] * 12
    bins[final.SELECTORS['heldout_final_W']] = [1] * 4 + [0] * 2
    bins[final.SELECTORS['heldout_final_N_old_rel']] = [1] * 2 + [0] * 4
    metrics = final.reduce_bins(bins)
    assert metrics['E1'] == .5
    assert metrics['E3'] == 1 / 3
    bins[final.SELECTORS['heldout_final_W']].pop()
    with pytest.raises(final.FinalAuditError, match='endpoint count'):
        final.reduce_bins(bins)


def test_interval_family_and_zero_variance():
    from scipy.stats import t
    values = [-1., 1.] * 32
    result = final.interval(values)
    expected_half = float(t.ppf(1 - .05 / 52, 63)) * math.sqrt(64 / 63) / 8
    assert result['mean'] == 0 and result['n_worlds'] == 64
    assert result['upper'] == pytest.approx(expected_half)
    zero = final.interval([0.] * 64)
    assert zero['upper'] == pytest.approx(4 * math.sqrt(math.log(1040) / 128))
    assert zero['lower'] < 0 < zero['upper'] and 'Hoeffding' in zero['interval']
    assert zero['upper'] > 0


def test_compare_rejects_changed_axis_family_or_item():
    final.compare({'a': [1., False]}, {'a': [1., False]})
    for candidate in ({'a': [1.01, False]}, {'a': [1., True]}, {'a': [1.]}, {'b': [1., False]}):
        with pytest.raises(final.FinalAuditError):
            final.compare(candidate, {'a': [1., False]})


def write_preflight(root, complete=False):
    directory = root / 'results/science'
    directory.mkdir(parents=True)
    (directory / 'RUN_STATUS.json').write_text(json.dumps({'validated_receipts': 832 if complete else 831,
        'roster': 832, 'running': [], 'failure': None}))
    (directory / 'RUN_LEDGER.json').write_text(json.dumps({'persisted_receipts': 832, 'failure': None}))


def test_incomplete_admission_does_not_decode_receipts(tmp_path, monkeypatch):
    write_preflight(tmp_path)
    monkeypatch.setattr(gzip, 'decompress', lambda _: pytest.fail('must not decode science receipt'))
    with pytest.raises(final.FinalAuditError, match='full 832'):
        final.preflight_roster(tmp_path)


def test_complete_metadata_without_receipts_or_analysis_rejected(tmp_path):
    write_preflight(tmp_path, True)
    with pytest.raises(final.FinalAuditError, match='exact receipt roster'):
        final.preflight_roster(tmp_path)
    for name in final.roster_names():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'Not read during gate')
    with pytest.raises(final.FinalAuditError, match='qualified final analysis absent'):
        final.preflight_roster(tmp_path)


def test_active_worker_prevents_admission(tmp_path):
    write_preflight(tmp_path, True)
    status = tmp_path / 'results/science/RUN_STATUS.json'
    data = json.loads(status.read_text()); data['running'] = [['P1', 190064]]
    status.write_text(json.dumps(data))
    with pytest.raises(final.FinalAuditError, match='active or failed'):
        final.preflight_roster(tmp_path)


def test_bundle_is_complete_and_write_once(tmp_path):
    source = tmp_path / 'source'; source.write_bytes(b'original frozen source')
    receipt = tmp_path / 'receipt'; receipt.write_bytes(b'original receipt')
    output = tmp_path / 'return.zip'
    files = {'src/source.py': source, 'results/receipt.gz': receipt}
    verified = final.build_bundle(output, files, 'LOCK')
    assert verified['files'] == 2
    assert final.verify_bundle(output) == verified
    with pytest.raises(final.FinalAuditError, match='overwrite'):
        final.build_bundle(output, files, 'LOCK')
    with zipfile.ZipFile(output, 'a') as archive:
        archive.writestr('unmanifested.txt', 'tamper')
    with pytest.raises(final.FinalAuditError, match='manifest ZIP coverage'):
        final.verify_bundle(output)


def test_bundle_blocks_paths_outside_package(tmp_path):
    for name in ('../escape', '/absolute', 'a/../../escape'):
        with pytest.raises(final.FinalAuditError, match='unsafe'):
            final.safe_path(tmp_path, name)


def test_off_host_channels_are_not_added_as_independent_receipts(tmp_path):
    directory = tmp_path / 'results/recovery'; directory.mkdir(parents=True)
    hashes = {f'{a}/{w}': f'h{a}{w}' for a in final.ARMS for w in final.WORLDS}
    table = {final.receipt_name(*k.split('/')): v for k, v in hashes.items()}
    common = {'verified_at_utc': '2026-10-02T00:00:00Z', 'lock_digest': 'L'}
    (directory / 'BACKUP_STATUS.json').write_text(json.dumps({**common,
        'schema': 'MINIFLY-A3-VERIFIED-BACKUP-v1', 'receipt_sha256': dict(list(table.items())[:694])}))
    (directory / 'PRIVATE_BACKUP_STATUS.json').write_text(json.dumps({**common,
        'schema': 'MINIFLY-A3-VERIFIED-PRIVATE-BACKUP-v1', 'receipt_sha256': table}))
    result = final.coverage(tmp_path, hashes, 'L')
    assert result['combined_receipts'] == 832
    assert result['channels']['github']['receipts'] == 694
    assert result['channels']['private_recoverable']['receipts'] == 832
    assert result['new_final_outputs_off_host'] is False
    bad = read = json.loads((directory / 'PRIVATE_BACKUP_STATUS.json').read_text())
    bad['receipt_sha256'][next(iter(table))] = 'wrong'
    (directory / 'PRIVATE_BACKUP_STATUS.json').write_text(json.dumps(bad))
    with pytest.raises(final.FinalAuditError, match='does not match'):
        final.coverage(tmp_path, hashes, 'L')


# The following fixtures contain fabricated metadata and tiny fabricated receipts.
# They never read, reduce, or simulate any science trajectory.
def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def rewrite(path, edit):
    value = final.read_json(path)
    edit(value)
    path.write_text(json.dumps(value, sort_keys=True))


def synthetic_bins(arm, world):
    bins = {}
    for selector in final.SELECTORS.values():
        n = 16 if 'fact' in selector[2] else 12 if selector[2] == 'old_relation_taught' else 6
        correct = (final.ARMS.index(arm) + world - 190001) % (n + 1)
        bins[selector] = [1] * correct + [0] * (n - correct)
    return bins


def synthetic_replay(root, locked, hashes):
    wrapper = root / 'ops/audit_science_replay.py'
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_bytes(b'synthetic wrapper, never executed')
    source = {'synthetic_source': 'synthetic_hash'}
    ctx = {'lock_digest': locked['lock_digest'], 'locked_environment': locked['environment'],
        'receipt_source_sha256': source, 'wrapper_sha256': final.file_sha(wrapper),
        'scope': copy.deepcopy(final.REPLAY_SCOPE), 'runtime': {'synthetic': True}}
    records = []
    for world in range(190001, 190005):
        for arm in final.ARMS:
            identity = {**ctx, 'arm': arm, 'world': world,
                'receipt_sha256': {f'{arm}/{world}': hashes[f'{arm}/{world}']}}
            if arm in final.DEPENDENCIES:
                key = f'{final.DEPENDENCIES[arm]}/{world}'
                identity['receipt_sha256'][key] = hashes[key]
            replay = {'arm': arm}
            if arm.startswith('P'):
                replay.update(branch='W', store=0, p_updates_checked=600, pd_checkpoints_checked=2)
            elif arm.startswith('Z'):
                replay.update(branch='W', store=0, z_events_checked=600)
            else:
                replay['novelty_checked'] = 600
                if arm in ('R0_signed', 'R3', 'R3_randtarget'):
                    replay['shared_values'] = {'arm': arm, 'records_checked': 600}
            key = final.sha(final.canonical(identity))
            name = f'results/recovery/science_replay/{arm}/{world}.{key}.json'
            path = put(root, name, {'schema': 'MINIFLY-A3-SCIENCE-REPLAY-RECOVERY-v1',
                'key': key, 'identity': identity, 'pass': True, 'error': None,
                'checks': {'independent_log_audit': {'pass': True, 'arm': arm, 'world': world},
                           'independent_replay': replay},
                'resources': {'elapsed_s': 1., 'process_peak_rss_bytes': 1000}})
            records.append({'arm': arm, 'world': world, 'key': key, 'record': name,
                'record_sha256': final.file_sha(path), 'pass': True, 'cached': False})
    summary = put(root, 'results/recovery/science_replay/SUMMARY.synthetic.json', {
        'schema': 'MINIFLY-A3-SCIENCE-REPLAY-RECOVERY-v1-SUMMARY', 'pass': True,
        'sample_arms': list(final.ARMS), 'sample_worlds': [190001, 190002, 190003, 190004],
        'context': ctx, 'records': records})
    return summary, source


@pytest.fixture
def replay_admission(tmp_path):
    locked = {'lock_digest': 'synthetic-lock', 'environment': {'synthetic': 'environment'}}
    hashes = {f'{a}/{w}': final.sha(f'{a}/{w}'.encode()) for a in final.ARMS for w in final.WORLDS}
    summary, source = synthetic_replay(tmp_path, locked, hashes)
    return tmp_path, summary, locked, source, hashes


def test_synthetic_replay_admission_accepts_all_52(replay_admission):
    result = final.verify_replay(*replay_admission)
    assert result['records_checked'] == 52
    assert len(result['mechanism_checks']) == 52
    assert result['summed_record_elapsed_s'] == 52


@pytest.mark.parametrize('mutation', ['summary_pass', 'scope', 'source', 'duplicate', 'record_hash',
    'record_pass', 'identity', 'dependency', 'key', 'log', 'p_count', 'p_branch', 'z_count',
    'r_count', 'signed_count', 'signed_arm', 'resource'])
def test_synthetic_replay_admission_rejects_mutations(replay_admission, mutation):
    root, summary_path, locked, source, hashes = replay_admission
    summary = final.read_json(summary_path)
    if mutation == 'summary_pass': summary['pass'] = False
    elif mutation == 'scope': summary['context']['scope']['records'] = 1
    elif mutation == 'source': summary['context']['receipt_source_sha256'] = {}
    elif mutation == 'duplicate': summary['records'][-1] = summary['records'][0]
    elif mutation == 'record_hash': summary['records'][0]['record_sha256'] = 'wrong'
    else:
        target = {'dependency': 'Z2_rand', 'p_count': 'P1', 'p_branch': 'P1',
            'z_count': 'Z2', 'r_count': 'R1', 'signed_count': 'R3', 'signed_arm': 'R3'}.get(mutation, 'R0')
        entry = next(r for r in summary['records'] if r['arm'] == target)
        path = root / entry['record']; record = final.read_json(path)
        replay = record['checks']['independent_replay']
        if mutation == 'record_pass': record['pass'] = False
        elif mutation == 'identity': record['identity']['world'] += 1
        elif mutation == 'dependency': record['identity']['receipt_sha256'].pop('Z2/190001')
        elif mutation == 'key': record['key'] = 'wrong'
        elif mutation == 'log': record['checks']['independent_log_audit']['pass'] = False
        elif mutation == 'p_count': replay['pd_checkpoints_checked'] = 1
        elif mutation == 'p_branch': replay['branch'] = 'N_old_fact'
        elif mutation == 'z_count': replay['z_events_checked'] = 0
        elif mutation == 'r_count': replay['novelty_checked'] = 599
        elif mutation == 'signed_count': replay['shared_values']['records_checked'] = 599
        elif mutation == 'signed_arm': replay['shared_values']['arm'] = 'R0'
        elif mutation == 'resource': record['resources']['elapsed_s'] = -1
        path.write_text(json.dumps(record, sort_keys=True))
        entry['record_sha256'] = final.file_sha(path)
    summary_path.write_text(json.dumps(summary, sort_keys=True))
    with pytest.raises(final.FinalAuditError):
        final.verify_replay(root, summary_path, locked, source, hashes)


@pytest.fixture
def analysis_admission(tmp_path, monkeypatch):
    root = tmp_path
    locked = {'lock_digest': 'synthetic-lock', 'environment': {'synthetic': 'environment'},
              'files': {'src/analyze_a.py': 'synthetic-analyzer'}}
    for name in ('ops/bounded_analysis.py', 'ops/test_bounded_analysis.py', 'ops/test_finalize_a_v3.py'):
        p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'synthetic, never executed')
    runtime = {'lock_digest': locked['lock_digest'], 'source_lock_sha256': final.LOCK_SHA,
        'analyzer_sha256': locked['files']['src/analyze_a.py'], 'environment': locked['environment']}
    q = {'schema': 'MINIFLY-A3-BOUNDED-ANALYSIS-QUALIFICATION-v1', 'runtime': runtime,
        'adapter_sha256': final.file_sha(root / 'ops/bounded_analysis.py'),
        'test_sha256': final.file_sha(root / 'ops/test_bounded_analysis.py'), 'numerical_tolerance_used': None}
    for k in ('pass', 'test_only', 'byte_exact_final_metrics', 'byte_exact_stdout',
              'identical_scientific_call_order_arguments_and_returns', 'original_audits_enabled',
              'source_immutable', 'source_lock_immutable', 'qualification_code_identity_unchanged'):
        q[k] = True
    qualification = put(root, 'ops/QUALIFICATION.synthetic.json', q)
    monkeypatch.setattr(final, 'QUALIFICATION_SHA', final.file_sha(qualification))
    hashes, per = {}, {a: {} for a in final.ARMS}
    for arm in final.ARMS:
        for world in final.WORLDS:
            r = {'arm': arm, 'world': world, 'probes': [], 'params': {},
                 'resources': {'life_s': 1., 'peak_rss_bytes': 1000}}
            if arm in final.DEPENDENCIES:
                dep = final.DEPENDENCIES[arm]
                r['params'][f'paired_{dep}_receipt_sha256'] = hashes[f'{dep}/{world}']
            if arm.startswith('Z'):
                r['mechanism_events'] = {'W': [['Z', i, 0] for i in range(600)]}
            path = root / final.receipt_name(arm, world); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(gzip.compress(final.canonical(r), mtime=0))
            hashes[f'{arm}/{world}'] = final.file_sha(path)
            per[arm][str(world)] = {**final.reduce_bins(synthetic_bins(arm, world)), **r['resources']}
    from scipy.stats import t
    metrics = {'schema': 'MINIFLY-A3-CLAUDE-FINAL-METRICS-v1', 'lock_digest': locked['lock_digest'],
        'family_size_m': 26, 'package_contrasts_available': 12,
        'package_contrasts_unavailable': {'P3': {k: 'NOT_INSTANTIATED (technically unqualified/unresolved; no replacement)'
                                              for k in ('dE1', 'dE3')}},
        't_critical': float(t.ppf(1-.05/52, 63)), 'receipt_sha256': hashes, **final.aggregate(per)}
    metrics_path = put(root, 'results/FINAL_METRICS.json', metrics)
    marker = put(root, 'results/FINAL_ANALYSIS_STORAGE_VERIFICATION.json', {
        'schema': 'MINIFLY-A3-BOUNDED-ANALYSIS-ACCEPTANCE-v1', 'pass': True,
        'runtime': runtime, 'adapter_sha256': q['adapter_sha256'],
        'qualification_sha256': final.file_sha(qualification), 'final_metrics_sha256': final.file_sha(metrics_path),
        'storage': {'registered': 832, 'complete_rechecks': 1, 'decoded_receipts_cached': 0,
                    'receipt_sha256': hashes}})
    return root, qualification, locked, metrics_path, marker


def test_synthetic_analysis_admission_accepts_832_without_decode(analysis_admission, monkeypatch):
    root, qualification, locked, _, _ = analysis_admission
    monkeypatch.setattr(gzip, 'decompress', lambda _: pytest.fail('admission must not decode receipts'))
    metrics, marker = final.verify_analysis(root, qualification, locked)
    assert len(metrics['receipt_sha256']) == marker['storage']['registered'] == 832


@pytest.mark.parametrize('mutation', ['qualification', 'qualification_hash', 'adapter', 'test', 'runtime',
    'metrics_bytes', 'pass', 'registered', 'rechecks', 'cached', 'coverage', 'receipt', 'family', 'extra_field'])
def test_synthetic_analysis_admission_rejects_mutations(analysis_admission, mutation):
    root, qualification, locked, metrics_path, marker_path = analysis_admission
    marker, metrics = final.read_json(marker_path), final.read_json(metrics_path)
    if mutation == 'qualification': rewrite(qualification, lambda q: q.update(byte_exact_stdout=False))
    elif mutation == 'qualification_hash': marker['qualification_sha256'] = 'wrong'
    elif mutation == 'adapter': (root / 'ops/bounded_analysis.py').write_bytes(b'changed')
    elif mutation == 'test': (root / 'ops/test_bounded_analysis.py').write_bytes(b'changed')
    elif mutation == 'runtime': marker['runtime']['lock_digest'] = 'wrong'
    elif mutation == 'metrics_bytes': metrics_path.write_text('changed'); metrics = None
    elif mutation == 'pass': marker['pass'] = False
    elif mutation == 'registered': marker['storage']['registered'] = 831
    elif mutation == 'rechecks': marker['storage']['complete_rechecks'] = 0
    elif mutation == 'cached': marker['storage']['decoded_receipts_cached'] = 1
    elif mutation == 'coverage': marker['storage']['receipt_sha256'].pop('P4/190064')
    elif mutation == 'receipt': (root / final.receipt_name('P4', 190064)).write_bytes(b'changed')
    elif mutation == 'family': metrics['family_size_m'] = 12
    elif mutation == 'extra_field': metrics['unexpected_result'] = 'must reject'
    if metrics is not None:
        metrics_path.write_text(json.dumps(metrics, sort_keys=True))
        marker['final_metrics_sha256'] = final.file_sha(metrics_path)
    marker_path.write_text(json.dumps(marker, sort_keys=True))
    with pytest.raises((final.FinalAuditError, json.JSONDecodeError)):
        final.verify_analysis(root, qualification, locked)


@pytest.fixture
def orchestration(analysis_admission, monkeypatch):
    import numpy as np
    import audit_a
    import fixture
    root, qualification, locked, metrics_path, marker = analysis_admission
    hashes = final.read_json(metrics_path)['receipt_sha256']
    summary, source = synthetic_replay(root, locked, hashes)
    for name in ('scratch/supervision/supervisor.lock', 'scratch/bounded_analysis_runs/FULL_ANALYSIS.lock'):
        p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.touch()
    put(root, 'results/science/RUN_STATUS.json', {'lock_digest': locked['lock_digest'],
        'validated_receipts': 832, 'roster': 832, 'running': [], 'failure': None,
        'active_wall_s': 100., 'worker_s': 200.})
    put(root, 'results/science/RUN_LEDGER.json', {'lock_digest': locked['lock_digest'],
        'persisted_receipts': 832, 'failure': None, 'active_wall_s': 100., 'worker_s': 200.})
    ack = put(root, 'results/recovery/PRIVATE_BACKUP_STATUS.json', {
        'schema': 'MINIFLY-A3-VERIFIED-PRIVATE-BACKUP-v1', 'verified_at_utc': '2026-10-02T00:00:00Z',
        'lock_digest': locked['lock_digest'],
        'receipt_sha256': {final.receipt_name(*k.split('/')): v for k, v in hashes.items()}})
    cal = put(root, 'results/calibration/R_CALIBRATION.json', {'theta': .1, 'scales': {'shared': 1., 'private': 1.}})
    tech = put(root, 'results/technical_final/TECHNICAL_AUDIT.json', {'pass': True})
    budget = put(root, 'RESOURCE_BUDGET.json', {'synthetic': True})
    p = root / 'src/analyze_a.py'; p.parent.mkdir(parents=True); p.write_bytes(b'synthetic, never executed')
    (root / 'package').mkdir()
    np.savez(root / 'package/FULL151_CANONICAL_B.npz', indices=np.array([0]), indptr=np.array([0, 1]))
    # Source/runtime validation is a test seam. Admission, full iteration,
    # paired hashes, arithmetic comparisons, rechecks, packaging and publication run normally.
    locked['files'] = {p.relative_to(root).as_posix(): final.file_sha(p) for p in
                       [cal, tech, budget, root / 'package/FULL151_CANONICAL_B.npz']}
    locked['files']['src/analyze_a.py'] = final.file_sha(root / 'src/analyze_a.py')
    put(root, 'SOURCE_LOCK.json', locked)
    rewrite(qualification, lambda q: q['runtime'].update(analyzer_sha256=locked['files']['src/analyze_a.py']))
    monkeypatch.setattr(final, 'QUALIFICATION_SHA', final.file_sha(qualification))
    rewrite(marker, lambda m: (m.update(qualification_sha256=final.file_sha(qualification)),
                              m['runtime'].update(analyzer_sha256=locked['files']['src/analyze_a.py'])))
    monkeypatch.setattr(final, 'verify_source', lambda r: (locked, source))
    monkeypatch.setattr(fixture, 'make_world', lambda w: {'world': w})
    monkeypatch.setattr(final, 'fixture_evidence', lambda f: {'synthetic': True})
    checked = []
    def validate(r, arm, world, *args):
        checked.append((arm, world))
        return synthetic_bins(arm, world)
    monkeypatch.setattr(final, 'validate_receipt', validate)
    monkeypatch.setattr(audit_a, 'audit_receipt', lambda r, **kwargs: {'pass': True, 'arm': r['arm'], 'world': r['world']})
    return root, summary, qualification, root / 'return.zip', checked, metrics_path, marker


def test_synthetic_whole_finalizer_all_832_and_write_once(orchestration):
    root, summary, qualification, bundle, checked, _, _ = orchestration
    result = final.finalize(root, summary, qualification, bundle)
    assert result['pass'] and result['local_only'] and not result['off_host_final_bundle_verified']
    assert len(checked) == 832 and set(checked) == {(a, w) for a in final.ARMS for w in final.WORLDS}
    assert final.verify_bundle(bundle)['bundle_sha256'] == result['bundle_sha256']
    commit = final.read_json(root / 'results/FINALIZATION_COMMIT.json')
    assert commit == result
    for name, digest in commit['final_outputs_sha256'].items():
        assert final.file_sha(root / 'results' / name) == digest
    with zipfile.ZipFile(bundle) as archive:
        assert final.roster_names() <= set(archive.namelist())
        assert 'TECHNICAL_AUDIT.json' in archive.namelist()
    text = (root / 'results/REPORT.md').read_text()
    assert 'P3 and P3_shuffle: NOT_INSTANTIATED' in text
    assert 'Package B V2 E3 is invalid' in text
    with pytest.raises(final.FinalAuditError, match='new and outside'):
        final.finalize(root, summary, qualification, bundle)


@pytest.mark.parametrize('mutation', ['per_world', 'contrasts', 'diagnostics', 'absolute', 'bool_type'])
def test_synthetic_whole_finalizer_compares_complete_metrics(orchestration, mutation):
    root, summary, qualification, bundle, checked, metrics, marker = orchestration
    def change(m):
        if mutation == 'per_world': m['per_world']['P4']['190064']['E1'] += .001
        elif mutation == 'contrasts': m['contrasts']['P4']['per_world_dE3'][-1] += .001
        elif mutation == 'diagnostics': m['diagnostics']['Z2']['dE1']['mean'] += .001
        elif mutation == 'absolute': m['absolute']['P4']['old_fact_marker_final_W']['mean'] += .001
        elif mutation == 'bool_type': m['contrasts']['P4']['dE1_improvement_claim'] = 0
    rewrite(metrics, change)
    rewrite(marker, lambda m: m.update(final_metrics_sha256=final.file_sha(metrics)))
    with pytest.raises(final.FinalAuditError):
        final.finalize(root, summary, qualification, bundle)
    assert len(checked) == 832 and not bundle.exists()
    assert not any((root / 'results' / name).exists() for name in final.OUTPUTS)


@pytest.mark.parametrize('target', ['receipt', 'metrics', 'marker', 'summary', 'qualification', 'status'])
def test_synthetic_whole_finalizer_rejects_input_change_after_scoring(orchestration, monkeypatch, target):
    root, summary, qualification, bundle, checked, metrics, marker = orchestration
    path = {'receipt': root / final.receipt_name('R0', 190001), 'metrics': metrics,
            'marker': marker, 'summary': summary, 'qualification': qualification,
            'status': root / 'results/science/RUN_STATUS.json'}[target]
    original = final.make_report
    def changed(*args):
        result = original(*args)
        path.write_bytes(path.read_bytes() + b' ')
        return result
    monkeypatch.setattr(final, 'make_report', changed)
    with pytest.raises(final.FinalAuditError, match='admitted input changed'):
        final.finalize(root, summary, qualification, bundle)
    assert len(checked) == 832 and not bundle.exists()
    assert not any((root / 'results' / name).exists() for name in final.OUTPUTS)


def test_synthetic_whole_finalizer_lock_blocks_before_decode(orchestration, monkeypatch):
    import fcntl
    root, summary, qualification, bundle, checked, _, _ = orchestration
    with (root / 'scratch/supervision/supervisor.lock').open('ab') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(gzip, 'decompress', lambda _: pytest.fail('locked gate must not decode'))
        with pytest.raises(final.FinalAuditError, match='still active'):
            final.finalize(root, summary, qualification, bundle)
    assert checked == [] and not bundle.exists()


def test_synthetic_whole_finalizer_incomplete_gate_before_decode(orchestration, monkeypatch):
    root, summary, qualification, bundle, checked, _, _ = orchestration
    rewrite(root / 'results/science/RUN_STATUS.json', lambda s: s.update(validated_receipts=831))
    monkeypatch.setattr(gzip, 'decompress', lambda _: pytest.fail('incomplete gate must not decode'))
    with pytest.raises(final.FinalAuditError, match='full 832'):
        final.finalize(root, summary, qualification, bundle)
    assert checked == [] and not bundle.exists()


def test_synthetic_whole_finalizer_rejects_input_change_during_bundle(orchestration, monkeypatch):
    root, summary, qualification, bundle, checked, _, _ = orchestration
    original = final.build_bundle
    def changed(*args):
        result = original(*args)
        path = root / final.receipt_name('P4', 190064)
        path.write_bytes(path.read_bytes() + b' ')
        return result
    monkeypatch.setattr(final, 'build_bundle', changed)
    with pytest.raises(final.FinalAuditError, match='input changed during bundle'):
        final.finalize(root, summary, qualification, bundle)
    assert len(checked) == 832 and not bundle.exists()
    assert not any((root / 'results' / name).exists() for name in final.OUTPUTS)


def test_synthetic_publication_failure_never_gets_acceptance_commit(orchestration, monkeypatch):
    root, summary, qualification, bundle, checked, _, _ = orchestration
    original = final.os.link
    called = []
    def interrupted(src, dest):
        called.append(str(dest))
        if len(called) == 3:
            raise OSError('synthetic interrupted publication')
        return original(src, dest)
    monkeypatch.setattr(final.os, 'link', interrupted)
    with pytest.raises(OSError, match='interrupted publication'):
        final.finalize(root, summary, qualification, bundle)
    assert bundle.exists() and (root / 'results/FINAL_AUDIT.json').exists()
    assert not (root / 'results/FINALIZATION_COMMIT.json').exists()
    # Surviving evidence is retained; partial publication cannot masquerade as
    # completion and a retry cannot silently overwrite it.
    with pytest.raises(final.FinalAuditError, match='new and outside'):
        final.finalize(root, summary, qualification, bundle)


def test_unpinned_qualification_cannot_self_authorize(analysis_admission):
    root, qualification, locked, _, marker = analysis_admission
    rewrite(qualification, lambda q: q.update(extra_claim='new evidence'))
    rewrite(marker, lambda m: m.update(qualification_sha256=final.file_sha(qualification)))
    with pytest.raises(final.FinalAuditError, match='approved storage qualification'):
        final.verify_analysis(root, qualification, locked)


@pytest.fixture
def terminal_persistence(orchestration):
    root, summary, qualification, bundle, checked, metrics, marker = orchestration
    rewrite(root / 'results/science/RUN_LEDGER.json', lambda l: l.update(persisted_receipts=829))
    rewrite(root / 'results/science/RUN_STATUS.json', lambda s: s.update(state='complete'))
    ledger = final.read_json(root / 'results/science/RUN_LEDGER.json')
    status = final.read_json(root / 'results/science/RUN_STATUS.json')
    hashes = final.read_json(metrics)['receipt_sha256']
    roster = {final.receipt_name(*k.split('/')): v for k, v in hashes.items()}
    common = {'schema': 'MINIFLY-A3-LOCAL-CHECKPOINT-v2', 'lock_digest': ledger['lock_digest'],
              'ledger_snapshot': ledger, 'status_snapshot': status}
    first = put(root, 'results/recovery/local_queue/first.json', {**common, 'sequence': 1,
        'previous_checkpoint': None, 'validated_receipts': 829, 'receipt_sha256': dict(list(roster.items())[:829])})
    tip = put(root, 'results/recovery/local_queue/terminal.json', {**common, 'sequence': 2,
        'previous_checkpoint': {'path': first.relative_to(root).as_posix(), 'sha256': final.file_sha(first), 'sequence': 1},
        'validated_receipts': 832, 'receipt_sha256': roster})
    local = put(root, 'results/recovery/LOCAL_STATUS.json', {'local_validated_receipts': 832,
        'checkpoint': tip.relative_to(root).as_posix(), 'checkpoint_sha256': final.file_sha(tip)})
    archive = root / 'independent-private-readback.zip'
    names = ('SOURCE_LOCK.json', 'results/science/RUN_LEDGER.json', 'results/science/RUN_STATUS.json')
    manifest = {'schema': 'MINIFLY-A3-PRIVATE-INCREMENTAL-v1', 'lock_digest': ledger['lock_digest'],
        'receipt_sha256': roster, 'total_recoverable_with_base_and_prior_backups': 832,
        'file_sha256': {'minifly_a_v3/' + n: final.file_sha(root / n) for n in names}}
    with zipfile.ZipFile(archive, 'x') as z:
        for name in names: z.write(root / name, 'minifly_a_v3/' + name)
        z.writestr('RECOVERY_MANIFEST.json', json.dumps(manifest))
    ack = root / 'results/recovery/PRIVATE_BACKUP_STATUS.json'
    rewrite(ack, lambda a: a.update(archive_sha256=final.file_sha(archive), library_file_id='synthetic-library-id'))
    return orchestration, ledger, status, first, tip, local, archive, ack


def test_terminal_persistence_proves_stale_cache_without_decoding(terminal_persistence, monkeypatch):
    orchestration, ledger, status, _, _, _, archive, _ = terminal_persistence
    root = orchestration[0]
    before = final.file_sha(root / 'results/science/RUN_LEDGER.json')
    monkeypatch.setattr(gzip, 'decompress', lambda _: pytest.fail('persistence gate must not decode'))
    evidence, pinned = final.verify_terminal_persistence(root, status, ledger, archive)
    assert evidence['authoritative_persisted_receipts'] == 832
    assert evidence['ledger_cached_persisted_receipts'] == 829 and evidence['cache_is_stale']
    assert evidence['chain_records'] == 2
    assert len([p for p in pinned if p.suffix == '.gz']) == 832
    assert final.preflight_roster(root, archive) == (status, ledger)
    assert final.file_sha(root / 'results/science/RUN_LEDGER.json') == before


@pytest.mark.parametrize('mutation', ['local_count', 'tip_hash', 'duplicate_sequence', 'previous',
    'prior_replacement', 'missing_receipt', 'wrong_world', 'receipt_bytes', 'snapshot', 'ack_map', 'archive_bytes'])
def test_terminal_persistence_mutations_fail_closed(terminal_persistence, mutation):
    orchestration, ledger, status, first, tip, local, archive, ack = terminal_persistence
    root = orchestration[0]
    if mutation == 'local_count': rewrite(local, lambda x: x.update(local_validated_receipts=831))
    elif mutation == 'tip_hash': rewrite(local, lambda x: x.update(checkpoint_sha256='wrong'))
    elif mutation == 'duplicate_sequence': (first.parent / 'duplicate.json').write_bytes(first.read_bytes())
    elif mutation == 'previous': rewrite(tip, lambda x: x['previous_checkpoint'].update(sha256='wrong'))
    elif mutation == 'prior_replacement':
        rewrite(tip, lambda x: x['receipt_sha256'].update({next(iter(x['receipt_sha256'])): 'wrong'}))
    elif mutation == 'missing_receipt': rewrite(tip, lambda x: x['receipt_sha256'].pop(next(iter(x['receipt_sha256']))))
    elif mutation == 'wrong_world': rewrite(tip, lambda x: x['receipt_sha256'].update({'results/science/P4/190065.json.gz': 'wrong'}))
    elif mutation == 'receipt_bytes': (root / final.receipt_name('P4', 190064)).write_bytes(b'changed')
    elif mutation == 'snapshot': rewrite(tip, lambda x: x['status_snapshot'].update(worker_s=-1))
    elif mutation == 'ack_map': rewrite(ack, lambda x: x['receipt_sha256'].pop(next(iter(x['receipt_sha256']))))
    elif mutation == 'archive_bytes': archive.write_bytes(archive.read_bytes() + b'changed')
    if mutation in ('previous', 'prior_replacement', 'missing_receipt', 'wrong_world', 'snapshot'):
        rewrite(local, lambda x: x.update(checkpoint_sha256=final.file_sha(tip)))
    with pytest.raises(final.FinalAuditError):
        final.verify_terminal_persistence(root, status, ledger, archive)


def test_synthetic_finalizer_admits_terminal_832_preserving_cache_829(terminal_persistence):
    orchestration, ledger, status, first, tip, _, archive, _ = terminal_persistence
    root, summary, qualification, bundle, checked, _, _ = orchestration
    ledger_hash = final.file_sha(root / 'results/science/RUN_LEDGER.json')
    result = final.finalize(root, summary, qualification, bundle, terminal_backup_path=archive)
    assert result['pass'] and len(checked) == 832
    assert final.file_sha(root / 'results/science/RUN_LEDGER.json') == ledger_hash
    audit = final.read_json(root / 'results/FINAL_AUDIT.json')
    resources = final.read_json(root / 'results/RESOURCE_REPORT.json')
    assert audit['terminal_persistence'] == resources['terminal_persistence']
    assert audit['terminal_persistence']['ledger_cached_persisted_receipts'] == 829
    assert 'ledger cache reports 829' in (root / 'results/REPORT.md').read_text()
    with zipfile.ZipFile(bundle) as z:
        assert first.relative_to(root).as_posix() in z.namelist()
        assert tip.relative_to(root).as_posix() in z.namelist()
        assert 'results/recovery/FINAL_PRIMARY_PRIVATE_READBACK.zip' in z.namelist()
