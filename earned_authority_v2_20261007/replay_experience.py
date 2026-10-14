"""Exploratory observer learning; never constructs or teaches native Core.

Ordinary recorded teacher bytes update authority AFTER pre-feedback predictions.
Scored probe answers are evaluator-only and never update authority.
"""
import gzip
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from experience_policy import ExperienceAuthority, Observation, POLICIES, sensory_key

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / 'next_core_mechanism_v1_20261006'
sys.dont_write_bytecode = True
sys.path.insert(0, str(PARENT))
from assays import make_world, permitted
from io_utils import read_receipt, atomic_json


def observation(d):
    return Observation(tuple(d['shared']), tuple(d['private']))


def audit_record(rec, event, branch):
    expected = permitted(branch, event['stage'])
    assert rec['index'] == event['index'] and rec['learn'] is expected
    assert rec['prediction_precedes_outcome'] is True
    assert rec['write']['outcome'] == event['outcome']
    calls = rec['actual_calls']
    assert len(calls) == 8 and [r['store'] for r in calls] == list(range(8))
    assert all(r['write'] is expected for r in calls)
    if not expected:
        assert all(r['applied_l1'] == 0 and r['no_write_reference_equal'] is True for r in calls)


def baseline_parity(row, result, *, record=False):
    e = row['prediction']['emitted'] if record else row.get('emitted')
    if e is None:
        e = row['readouts']['ERROR']['emitted']
    assert result['ERROR']['emitted'] == e, 'ERROR exact parity'
    h = row.get('Q_HALF_emitted')
    if h is None:
        h = row.get('policies', {}).get('Q_HALF', {}).get('emitted')
    if h is None:
        h = row.get('readouts', {}).get('Q_HALF', {}).get('emitted')
    assert h is not None and result['Q_HALF']['emitted'] == h, 'Q_HALF exact parity'


def probe_inputs(row, w):
    if 'cue_hex' in row:
        cues = [bytes.fromhex(c) for c in row['cue_hex']]
    elif w['assay'] == 'reuse':
        pair = row.get('pair', row.get('item'))
        a, b = w['sets'][row['stage']][2 * pair: 2 * pair + 2]
        r = (a, b) if row['order'] == 0 else (b, a)
        cues = [bytes.fromhex(x['cue_hex']) for x in r]
    else:
        cues = [bytes.fromhex(w['sets'][row['stage']][row['item']]['cue_hex'])]
    predictions = row.get('predictions') or [row['prediction']]
    return tuple(sensory_key(c) for c in cues), tuple(observation(p) for p in predictions)


def main():
    started = time.process_time()
    lock = ROOT / 'RULE_LOCK.json'
    lock_before = hashlib.sha256(lock.read_bytes()).hexdigest()
    candidates = list(sorted((ROOT.parent / 'next_core_budget_v2_20261006/science/confirm/receipts').glob('*_ERROR_*.json.gz')))
    candidates += list(sorted((ROOT.parent / 'earned_authority_v1_20261007/cycles/cycle3/full').glob('*/receipts/*.json.gz')))
    assert len(candidates) == 99, 'expected 96 archived jobs and three DEV jobs'
    rows, histories, sources = [], [], []
    parity_records = parity_probes = allowed_observer_records = disabled_records = 0
    literal_state_size = 0
    for path in candidates:
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        d = read_receipt(path)
        assert d['complete'] and d['arm'] == 'ERROR'
        w = make_world(d['world'], d['assay'])  # frozen inputs only, no native learner
        assert d['fixture_sha256'] == w['sha256']
        scope = 'DEV_ONE_WORLD' if d['development'] else 'HISTORICAL_48_WORLDS'
        sources.append({'path': str(path), 'sha256': source_hash, 'world': d['world'], 'assay': d['assay'], 'scope': scope})
        for branch, bd in d['branches'].items():
            c = ExperienceAuthority()
            # Explicit diagnostic: teach observer old records on the native
            # N_old trajectory. This is never the primary causal control.
            observer_only = ExperienceAuthority() if branch == 'N_old' else None
            records = bd['records']
            assert len(records) == len(w['events'])
            probes = {p['name']: p for p in bd['probes']}
            visited = []
            for i, (rec, event) in enumerate(zip(records, w['events'])):
                audit_record(rec, event, branch)
                key = sensory_key(bytes.fromhex(event['cue_hex']))
                ob = observation(rec['prediction'])
                forecast = c.predict(key, ob)
                baseline_parity(rec, forecast, record=True)
                before = c.snapshot()
                feedback = c.observe(rec['write']['outcome'], learn=rec['learn'])
                if rec['learn']:
                    allowed_observer_records += 1
                    assert feedback['updated']
                else:
                    disabled_records += 1
                    before['pending'] = None
                    assert c.snapshot() == before, 'disabled observer write changed learning state'
                if observer_only is not None:
                    observer_only.predict(key, ob)
                    observer_only.observe(rec['write']['outcome'], learn=True)
                parity_records += 1
                for name in w['boundaries'].get(str(i + 1), []):
                    probe = probes[name]
                    visited.append(name)
                    assert probe['at'] == w['clocks'][name]
                    assert probe['state_before'] == probe['state_after']
                    initial = c.digest()
                    for row in probe['rows']:
                        assert row['first_pre_feedback']
                        keys, observations = probe_inputs(row, w)
                        result = c.query(keys, observations)
                        baseline_parity(row, result)
                        parity_probes += 1
                        if w['assay'] == 'lifetime' and row['stage'] == 'old' and row['item'] not in w['intact_old_items']:
                            continue
                        # Target is only used AFTER all readouts have emitted.
                        correct = {p: int(r['emitted'] == row['target']) for p, r in result.items()}
                        if observer_only is not None:
                            answer = observer_only.query(keys, observations)['EXPERIENCE']['emitted']
                            correct['OBSERVER_ONLY_OLD'] = int(answer == row['target'])
                        rows.append({'scope': scope, 'world': d['world'], 'assay': d['assay'], 'branch': branch,
                                     'checkpoint': name, 'group': row['stage'],
                                     'item': row.get('item', row.get('pair')), 'order': row.get('order'),
                                     'supported': result['EXPERIENCE']['supported'],
                                     'alpha': result['EXPERIENCE']['alpha'], 'correct': correct,
                                     'different_from_ERROR': result['EXPERIENCE']['emitted'] != result['ERROR']['emitted']})
                    assert c.digest() == initial, 'query changed continuing authority'
                    histories.append({'scope': scope, 'world': d['world'], 'assay': d['assay'], 'branch': branch,
                                      'checkpoint': name, 'records_seen': i + 1, 'state': c.snapshot(),
                                      'state_sha256': c.digest()})
                    literal_state_size = max(literal_state_size, len(json.dumps(c.snapshot()).encode()))
            assert set(visited) == set(probes), 'probe chronology incomplete'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash, 'archived receipt changed'
        print(json.dumps({'replayed_jobs': len(sources), 'of': len(candidates), 'assay': d['assay']}), flush=True) if len(sources) % 16 == 0 else None
    assert hashlib.sha256(lock.read_bytes()).hexdigest() == lock_before
    groups = defaultdict(list)
    for r in rows:
        groups[(r['scope'], r['assay'], r['world'], r['branch'], r['checkpoint'], r['group'])].append(r)
    per_world = []
    for k, rs in sorted(groups.items()):
        policies = list(POLICIES) + (['OBSERVER_ONLY_OLD'] if k[3] == 'N_old' else [])
        per_world.append(dict(zip(('scope', 'assay', 'world', 'branch', 'checkpoint', 'group'), k),
                              n=len(rs), accuracy={p: float(np.mean([r['correct'][p] for r in rs])) for p in policies},
                              half_weight_rate=float(np.mean([r['alpha'] == .5 for r in rs])),
                              changed_choices=sum(r['different_from_ERROR'] for r in rs),
                              supported_rate=float(np.mean([r['supported'] for r in rs]))))
    ag = defaultdict(list)
    for r in per_world:
        ag[(r['scope'], r['assay'], r['branch'], r['checkpoint'], r['group'])].append(r)
    aggregates = []
    for k, rs in sorted(ag.items()):
        aggregates.append(dict(zip(('scope', 'assay', 'branch', 'checkpoint', 'group'), k),
                               worlds=len(rs), accuracy={p: float(np.mean([r['accuracy'][p] for r in rs])) for p in rs[0]['accuracy']},
                               half_weight_rate=float(np.mean([r['half_weight_rate'] for r in rs])),
                               changed_choices=sum(r['changed_choices'] for r in rs),
                               supported_rate=float(np.mean([r['supported_rate'] for r in rs]))))
    finals = [r for r in rows if r['checkpoint'] == 'final']
    result = {'status': 'EXPLORATORY_REPLAY_NOT_CONFIRMATION', 'rule_sha256_before_after': lock_before,
              'no_new_native_trajectories': True, 'new_observer_learning': True,
              'no_threshold_search': True, 'sources': sources, 'native_jobs_replayed': len(sources),
              'baseline_parity': {'teacher_records': parity_records, 'probe_rows': parity_probes},
              'observer_records_allowed': allowed_observer_records, 'observer_records_disabled': disabled_records,
              'authority_logical_packed_bytes': ExperienceAuthority().packed_bytes(),
              'authority_snapshot_max_bytes': literal_state_size, 'cpu_s': time.process_time() - started,
              'final_half_weight_queries': sum(r['alpha'] == .5 for r in finals),
              'final_changed_choices': sum(r['different_from_ERROR'] for r in finals),
              'aggregate': aggregates, 'per_world': per_world,
              'latin_limitation': 'one new V1 technical DEVELOPMENT world, not64 confirmation worlds'}
    out = ROOT / 'replay'
    atomic_json(out / 'SUMMARY.json', result)
    atomic_json(out / 'HISTORIES.json', histories)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ROWS.json.gz').write_bytes(gzip.compress(json.dumps(rows, separators=(',', ':')).encode(), mtime=0))
    print(json.dumps({k: result[k] for k in ('status', 'baseline_parity', 'final_half_weight_queries', 'final_changed_choices', 'cpu_s')}))
    for a in aggregates:
        if a['checkpoint'] == 'final' and a['group'] in ('old', 'heldout', 'new', 'revision'):
            print(json.dumps(a))


if __name__ == '__main__':
    main()
