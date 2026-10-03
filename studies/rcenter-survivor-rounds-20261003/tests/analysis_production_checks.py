# SPDX-License-Identifier: GPL-3.0-or-later
"""Production terminal-analysis checks on a tiny synthetic cohort; no learner."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import recovery as r
import analyze as frozen
from durable import create, replace, sha
from integrity import digest_map
from validate_analysis import inputs, load


def run_checks(temp_root):
    root = Path(temp_root) / 'production-analysis'
    root.mkdir(parents=True, exist_ok=False)
    source = {'synthetic/source.py': '1' * 64}
    runtime = copy.deepcopy(r.RUNTIME)
    create(root / 'protocol/OFFICIAL_LOCK.json', {'hashes': source})
    acceptance = sha(root / 'protocol/OFFICIAL_LOCK.json')
    persistence = {'worlds': {}, 'jobs': {}}
    attempts = []
    checks = []

    def reject(fn):
        try:
            fn()
        except (RuntimeError, AssertionError, OSError, KeyError, ValueError):
            return
        raise AssertionError('terminal-analysis production guard accepted forbidden synthetic input')

    def fake_loader(path, world, expected_source, **kwargs):
        path = Path(path)
        manifest = r.read(path / 'manifest.json')
        assert manifest['world'] == world and manifest['source_digest'] == digest_map(expected_source)
        for name, part in manifest['parts'].items():
            assert sha(path / name) == part['sha256'] and (path / name).stat().st_size == part['bytes']
        header = r.read(path / 'header.json')
        assert header['source'] == expected_source
        if kwargs.get('expected_kind'):
            assert manifest['kind'] == kwargs['expected_kind']
        if kwargs.get('expected_acceptance'):
            assert header['acceptance_sha256'] == kwargs['expected_acceptance']
        return {'manifest': manifest, 'manifest_sha256': sha(path / 'manifest.json'), 'header': header,
                'probes': r.read(path / 'probes.json'), 'scientific_digest': 'f' * 64}

    def probes(world):
        out = []
        for branch in ('W', 'N_old_relation', 'N_new'):
            rows = []
            for name, n in (('old', 12), ('heldout', 4), ('new', 16)):
                for i in range(n):
                    rows.append({'set': name, 'policies': {policy: {'correct': int((world + i + len(branch) + len(policy)) % 3 == 0)}
                                                          for policy in ('alpha1', 'alpha0', 'alpha_half')}})
            out.append({'branch': branch, 'when': 'final', 'rows': rows})
        return out

    def seed_attempt(key, index):
        a = {'key': key, 'attempt': 1, 'status': 'completed', 'reservation_id': f'{index:032x}', 'host_token': 'mock-historical-host',
             'charged_s': 1.0, 'cap_reserved_s': r.CAP, 'source_digest': digest_map(source), 'runtime': runtime,
             'receipt': r.receipt_path(key), 'log': f'operations/attempts/{key}/attempt-1/worker.log'}
        path = root / a['receipt']
        if key == 'cycle3/operations':
            create(path, {'schema': 'RC-SURVIVOR-CYCLE3-OPERATIONS-v1', 'passed': True, 'mock_only': True, 'native_teaching_events': 0,
                          'source': source, 'runtime': runtime, 'key': key, 'attempt': 1, 'reservation_id': a['reservation_id'],
                          'resources': {'wall_s': 1.0, 'peak_rss_bytes': 1024}})
            a['receipt_sha256'] = sha(path)
        else:
            world = int(key.split('/')[1]) if key.startswith('science/') else 310200
            create(path / 'header.json', {'source': source, 'runtime': runtime, 'acceptance_sha256': acceptance})
            create(path / 'probes.json', probes(world))
            create(path / 'manifest.json', {'complete': True, 'world': world, 'kind': 'science' if key.startswith('science/') else 'technical',
                   'source_digest': digest_map(source), 'resources': {'wall_s': 1.0, 'peak_rss_bytes': 1024},
                   'parts': {name: {'sha256': sha(path / name), 'bytes': (path / name).stat().st_size} for name in ('header.json', 'probes.json')}})
            a['receipt_sha256'] = sha(path / 'manifest.json')
        group, item = ('worlds', key.split('/')[1]) if key.startswith('science/') else ('jobs', key)
        persistence[group][item] = {'manifest_sha256': a['receipt_sha256'], 'private_readback_verified': True, 'github_tree_verified': True}
        attempts.append(a)
        return a

    for i, key in enumerate(r.QUALIFICATION_KEYS, 1):
        seed_attempt(key, i)
    create(root / 'receipts/cycle3/replay_comparison.json', {'passed': True, 'fresh_process': True, 'same_world': 310200,
           'first_manifest_sha256': attempts[1]['receipt_sha256'], 'replay_manifest_sha256': attempts[2]['receipt_sha256'], 'scientific_digest': 'f' * 64})
    for i, world in enumerate(r.OFFICIAL_WORLDS[:-1], 4):
        seed_attempt(f'science/{world}', i)
    replace(root / 'operations/PERSISTENCE_STATE.json', persistence)
    with r.exclusive_lock(root) as lock:
        base = r.load_ledger(root, lock=lock)
        ledger = copy.deepcopy(base)
        ledger.update(attempts=copy.deepcopy(attempts), worker_s=float(len(attempts)))
        ledger = r.commit_ledger(root, base, ledger, 'isolated_synthetic_setup_63_worlds', lock=lock)
        plan = r.reconcile_and_plan(root, source, runtime, 'science', lock=lock, receipt_loader=fake_loader)
        assert plan['missing'] == ['science/310064', 'analysis/final'] and plan['next_key'] == 'science/310064'
        reject(lambda: r.prepare_attempt(root, source, runtime, 'science', 'analysis/final', lock=lock,
                                         host_token='mock-final-host', receipt_loader=fake_loader))
        reject(lambda: inputs(root, source, runtime, receipt_loader=fake_loader))
        last = seed_attempt('science/310064', 67)
        ledger_next = copy.deepcopy(ledger)
        ledger_next['attempts'].append(last)
        ledger_next['worker_s'] += 1
        ledger = r.commit_ledger(root, ledger, ledger_next, 'isolated_synthetic_setup_final_world', lock=lock)
        assert r.reconcile_and_plan(root, source, runtime, 'science', lock=lock, receipt_loader=fake_loader)['next_key'] is None
        reject(lambda: inputs(root, source, runtime, receipt_loader=fake_loader))
        replace(root / 'operations/PERSISTENCE_STATE.json', persistence)
        plan = r.reconcile_and_plan(root, source, runtime, 'science', lock=lock, receipt_loader=fake_loader)
        assert plan['missing'] == ['analysis/final'] and plan['next_key'] == 'analysis/final'
        reserved = r.prepare_attempt(root, source, runtime, 'science', 'analysis/final', lock=lock,
                                     host_token='mock-final-host', receipt_loader=fake_loader)
        a = reserved['attempt']
        assert a['key'] == 'analysis/final' and a['charged_s'] == 900 and a['receipt'] == 'results/final'
        replace(root / 'operations/RESERVATION_ACK.json', {'reservation_id': a['reservation_id'], 'key': a['key'],
                'source_digest': a['source_digest'], 'charged_s': 900, 'private_readback_verified': True,
                'ledger_sha256': reserved['ledger_sha256'], 'checkpoint_sha256': 'b' * 64, 'content_identity': 'c' * 64,
                'library_file_id': 'mock-final-library', 'version': 0})
        ledger, a = r.start_reserved_attempt(root, a['reservation_id'], source, runtime, 'science', lock=lock,
                                            child_identity=r.process_identity(), host_token='mock-final-host', receipt_loader=fake_loader)
        path = root / a['receipt']
        with patch.object(frozen, 'load', side_effect=fake_loader):
            result = frozen.analyze(root / 'receipts/science', source, path / 'analysis.json')
        manifest = {'schema': 'RC-SURVIVOR-ANALYSIS-RECEIPT-v1', 'complete': True, 'kind': 'analysis', 'key': a['key'],
                    'attempt': a['attempt'], 'reservation_id': a['reservation_id'], 'native_teaching_events': 0,
                    'source': source, 'source_digest': digest_map(source), 'runtime': runtime, 'lock_sha256': acceptance,
                    'input_manifest': result['input_manifest'], 'resources': {'wall_s': 1.0, 'peak_rss_bytes': 1024},
                    'parts': {'analysis.json': {'sha256': sha(path / 'analysis.json'), 'bytes': (path / 'analysis.json').stat().st_size}}}
        create(path / 'manifest.json', manifest)
        assert load(root, path, source, runtime, attempt=a, receipt_loader=fake_loader)['result'] == result
        # Deliberately rehash the part and receipt after semantic tampering.
        # Rejection therefore proves formula/cohort checks, not only hashing.
        for mutate in (
            lambda d: d['worlds'].pop(),
            lambda d: d['input_manifest'].pop('310064'),
            lambda d: d['world_values'][0].update(E3_1=0.987654),
            lambda d: d['intervals']['E3_1'].update(lower=0.987654),
            lambda d: d['decision'].update(later_round_launch_authorized=True),
        ):
            changed = copy.deepcopy(result)
            mutate(changed)
            replace(path / 'analysis.json', changed)
            changed_manifest = copy.deepcopy(manifest)
            changed_manifest['parts']['analysis.json'] = {'sha256': sha(path / 'analysis.json'), 'bytes': (path / 'analysis.json').stat().st_size}
            replace(path / 'manifest.json', changed_manifest)
            reject(lambda: load(root, path, source, runtime, attempt=a, receipt_loader=fake_loader))
        replace(path / 'analysis.json', result)
        replace(path / 'manifest.json', manifest)
        for field, bad in (('source', {}), ('runtime', {}), ('lock_sha256', '0' * 64), ('reservation_id', '0' * 32), ('input_manifest', {}), ('resources', {'wall_s':0.,'peak_rss_bytes':1024}), ('resources', {'wall_s':1.,'peak_rss_bytes':True}), ('resources', {'wall_s':1.,'peak_rss_bytes':0})):
            changed = copy.deepcopy(manifest)
            changed[field] = bad
            replace(path / 'manifest.json', changed)
            reject(lambda: load(root, path, source, runtime, attempt=a, receipt_loader=fake_loader))
        replace(path / 'manifest.json', manifest)
        completed = r.finish_attempt(root, a['reservation_id'], source, runtime, 1.0, 1024, None, 0, lock=lock, receipt_loader=fake_loader)
        plan = r.reconcile_and_plan(root, source, runtime, 'science', lock=lock, receipt_loader=fake_loader)
        assert plan['missing'] == [] and plan['pending_barriers'][0]['key'] == 'analysis/final' and plan['blocked'] == ['persistence_barrier']
        persistence['jobs']['analysis/final'] = {'manifest_sha256': completed['receipt_sha256'], 'private_readback_verified': True, 'github_tree_verified': True}
        replace(root / 'operations/PERSISTENCE_STATE.json', persistence)
        final = r.reconcile_and_plan(root, source, runtime, 'science', lock=lock, receipt_loader=fake_loader)
        assert final['missing'] == [] and final['blocked'] == [] and final['next_key'] is None
        assert final['ledger']['worker_s'] == 68.0
        reject(lambda: r.prepare_attempt(root, source, runtime, 'science', 'analysis/final', lock=lock,
                                         host_token='mock-final-host', receipt_loader=fake_loader))
        checks += ['production_analysis_requires_all64_worlds_and_barriers', 'production_analysis_reserved_supervised_and_charged',
                   'production_analysis_rehash_semantic_subset_formula_decision_tampering_rejected',
                   'production_analysis_source_runtime_lock_attempt_input_identity_rejected',
                   'production_final_analysis_persistence_barrier_and_completed_noop']
    return checks
