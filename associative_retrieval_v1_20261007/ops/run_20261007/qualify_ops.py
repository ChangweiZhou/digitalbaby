"""Short registered E0 prefixes and operations tests, not new science."""
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
import bootstrap_ops as boot
from contracts import lock_sources, require_sources, require_scope, roster
from io_utils import atomic_json, read_receipt
from analysis import evaluate, query_cost
from science_audit import audit_science

def rejected(call):
    try:
        call()
    except (AssertionError, ValueError):
        return True
    raise AssertionError('tamper or invalid operation not rejected')

def synthetic_worlds(n, gain=True):
    out = []
    for i in range(n):
        # Nonzero variance with an unambiguous known favorable profile.
        jitter = ((i % 4)-1.5)*.005
        error = {'old_E1': .98, 'new_E1': .98, 'revision_E1': .98,
                 'reuse_W': .6+jitter, 'reuse_N_old': .5,
                 'reuse_effect': .1+jitter, 'operational_W_cpu_s': 100.,
                 'capability_per_1000_CPU_s': 1.+10*jitter}
        link = {**error, 'reuse_W': .8+2*jitter if gain else .5,
                'reuse_effect': .3+2*jitter if gain else 0.,
                'operational_W_cpu_s': 105.,
                'capability_per_1000_CPU_s': 1000*(.3+2*jitter)/105 if gain else 0.}
        perm = {**error, 'reuse_W': .6, 'reuse_effect': .1,
                'capability_per_1000_CPU_s': 1., 'operational_W_cpu_s': 105.}
        out.append({'world': i, 'conditions': {'ERROR': error, 'LINK': link, 'PERM': perm}})
    return out

def native_args(folder, limit=2, branch='W'):
    return [sys.executable, '-u', str(boot.OPS/'science_worker.py'),
            '--qualification', '--stage', 'qualification', '--world', '71008003',
            '--assay', 'reuse', '--branch', branch, '--limit', str(limit), '--folder', str(folder)]

def run_command(argv, log, expected=0):
    with open(log, 'w') as f:
        p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT)
    rc = p.wait(timeout=600)
    if rc != expected:
        raise RuntimeError(f'qualification exit {rc}, expected {expected}; {log}')

def receipt(folder):
    return read_receipt(Path(folder)/'receipts/71008003_reuse.json.gz')

def stable_record(r):
    r = copy.deepcopy(r)
    for k in list(r):
        if 'cpu_s' in k:
            r.pop(k)
    r['prediction'].pop('costs')
    return r

def main():
    root = boot.OPS/'qualification'
    if root.exists():
        raise ValueError('qualification folder already exists; do not rerun its prefixes silently')
    root.mkdir()
    lock = lock_sources()
    started = time.monotonic()
    checks = {}
    assert len(roster('screen')) == 16 and len(roster('confirm')) == 128
    rejected(lambda: require_scope(71008003, 'reuse', 'screen'))
    rejected(lambda: require_scope(71009001, 'reuse', 'qualification', True))
    rejected(lambda: require_scope(71009009, 'reuse', 'screen'))
    checks['strict_science_and_development_rosters'] = 'PASS'
    positive = evaluate(synthetic_worlds(8), 'screen')
    assert positive['advance'] and abs(positive['mean_causal_gain']-.2) < 1e-12
    assert abs(positive['CPU_ratio']-1.05) < 1e-12
    assert not evaluate(synthetic_worlds(8, False), 'screen')['advance']
    assert evaluate(synthetic_worlds(64), 'confirm')['adopted']
    assert not evaluate(synthetic_worlds(64, False), 'confirm')['adopted']
    bad = synthetic_worlds(8)
    for w in bad:
        w['conditions']['LINK']['old_E1'] = .8
    assert not evaluate(bad, 'screen')['advance']
    bad = synthetic_worlds(8)
    for w in bad:
        w['conditions']['LINK']['operational_W_cpu_s'] = 200.
    assert not evaluate(bad, 'screen')['advance']
    constant = synthetic_worlds(64)
    for w in constant:
        for c in w['conditions'].values():
            c['reuse_W'] = .6
            c['reuse_effect'] = .1
    z = evaluate(constant, 'confirm')
    assert not z['adopted'] and z['bounds']['causal_gain']['method'] == 'Hoeffding_registered_range'
    assert z['bounds']['causal_gain']['lower'] < 0
    checks['registered_analysis_positive_negative_harm_CPU_and_zero_variance'] = 'PASS'
    # Query accounting: common work plus each route's own measured operations.
    q = {'physical_probe_cpu_s': 10., 'rows': [{'predictions': [{'costs': {
        'query_cpu_s': 1., 'retrieval_cpu_s': 2., 'control_retrieval_cpu_s': 3.}}]}]}
    assert query_cost(q) == {'ERROR': 4., 'LINK': 7., 'PERM': 8.}
    checks['policy_CPU_accounting'] = 'PASS'
    # Eight actual processes start before waiting; each executes a disposable
    # two-record prefix plus read-only technical queries.
    children = []
    peak = 0
    for i in range(8):
        folder = root/f'concurrent8_{i}'
        log = open(root/f'concurrent8_{i}.log', 'w')
        p = subprocess.Popen(native_args(folder), stdout=log, stderr=subprocess.STDOUT)
        children.append((p, log, folder))
        peak = max(peak, sum(x.poll() is None for x, _, _ in children))
    for p, log, folder in children:
        rc = p.wait(timeout=600)
        log.close()
        if rc:
            raise RuntimeError(f'eight-worker adapter qualification failed: {folder}')
    assert peak == 8
    docs = [receipt(folder) for _, _, folder in children]
    reference = docs[0]['branches']['W']
    for d in docs:
        audit_science(d, lock)
        b = d['branches']['W']
        assert b['final_state_digest'] == reference['final_state_digest']
        assert b['association_final_digest'] == reference['association_final_digest']
        assert [stable_record(r) for r in b['records']] == [stable_record(r) for r in reference['records']]
    historical = read_receipt(boot.ROOT/'technical/cycle3/71008003_reuse_W.json.gz')
    assert ([stable_record(r) for r in historical['branches']['W']['records'][:2]] ==
            [stable_record(r) for r in reference['records']])
    checks['eight_process_native_state_and_historical_record_parity'] = 'PASS'
    # Safe boundary resume uses only short technical prefixes, never a full life.
    straight, resumed = root/'straight8', root/'resume8'
    run_command(native_args(straight, 8), root/'straight8.log')
    run_command(native_args(resumed, 8)+['--stop-after', '4'], root/'resume_stop.log', 75)
    run_command(native_args(resumed, 8)+['--resume'], root/'resume_continue.log')
    a, b = receipt(straight)['branches']['W'], receipt(resumed)['branches']['W']
    assert a['final_state_digest'] == b['final_state_digest']
    assert [stable_record(r) for r in a['records']] == [stable_record(r) for r in b['records']]
    before = (straight/'receipts/71008003_reuse.json.gz').read_bytes()
    again = subprocess.run(native_args(straight, 8), capture_output=True, text=True, timeout=120)
    assert again.returncode == 0 and 'SKIPPED_COMMITTED' in again.stdout
    assert (straight/'receipts/71008003_reuse.json.gz').read_bytes() == before
    checks['safe_boundary_resume_and_no_repeated_committed_records'] = 'PASS'
    disabled = root/'N_old'
    run_command(native_args(disabled, 2, 'N_old'), root/'N_old.log')
    nd = receipt(disabled)
    audit_science(nd, lock)
    for r in nd['branches']['N_old']['records']:
        assert not r['learn'] and all(c['no_write_reference_equal'] for c in r['actual_calls'])
    attacks = []
    d = copy.deepcopy(nd)
    d['branches']['N_old']['records'][0]['actual_calls'][0]['write'] = True
    attacks.append(d)
    d = copy.deepcopy(nd)
    d['branches']['N_old']['records'][0]['actual_calls'][0]['delta_l1'] = .1
    attacks.append(d)
    d = copy.deepcopy(docs[0]); d['operations_identity'] = 'wrong'; attacks.append(d)
    d = copy.deepcopy(docs[0]); d['fixture_sha256'] = 'wrong'; attacks.append(d)
    d = copy.deepcopy(docs[0]); d['branches']['W']['births'][1]['fly_id'] = d['branches']['W']['births'][0]['fly_id']; attacks.append(d)
    d = copy.deepcopy(docs[0]); d['branches']['W']['probes'][0]['at'] += 1; attacks.append(d)
    d = copy.deepcopy(docs[0]); d['branches']['W']['probes'][0]['rows'].pop(); attacks.append(d)
    d = copy.deepcopy(docs[0]); d['branches']['W']['records'][0]['prediction']['query'].append({}); attacks.append(d)
    d = copy.deepcopy(docs[0]); d['world'] = 71009001; attacks.append(d)
    d = copy.deepcopy(docs[0]); d['development'] = True; attacks.append(d)
    for d in attacks:
        rejected(lambda d=d: audit_science(d, lock))
    checks['actual_native_no_write_and_tamper_controls'] = f'PASS ({len(attacks)} rejected)'
    # Atomic publication filenames cannot be counted as committed receipts.
    t = root/'suffix_test'; t.mkdir()
    (t/'a.json.gz.pending-race').write_bytes(b'not a committed receipt')
    (t/'a.json.gz').write_bytes(b'committed')
    assert [p.name for p in t.glob('*.json.gz')] == ['a.json.gz']
    checks['atomic_pending_files_excluded_from_receipt_inventory'] = 'PASS'
    require_sources()
    d = {'verdict': 'PASS', 'operations_identity': lock['operations_identity'],
         'native_identity': lock['native_identity'], 'checks': checks,
         'actual_concurrency_peak': peak, 'science_trajectories_executed': 0,
         'short_disposable_technical_records': 16+8+8+2,
         'RSS_peak_bytes': max(x['peak_rss_bytes'] for x in docs),
         'elapsed_s': time.monotonic()-started, 'native_code_unchanged': True}
    atomic_json(boot.OPS/'QUALIFICATION.json', d)
    print(json.dumps(d, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
