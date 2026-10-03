"""Bounded engineering trials and adversarial auditors; no science-world entry."""
import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from compact_bridge import ROOT, ARMS, make_core, CompactCore, stores
from compact_fixture import make_world, BRANCHES, permitted
from compact_experiment import instrument, record
from compact_integrity import identity, source_map
from compact_audit import audit_receipt, read, paired_private_equal
from compact_worker import run_job
from compact_checkpoint import load
from compact_storage import atomic_json
from compact_analysis import cp_upper, cp_lower


def rejected(call):
    try: call()
    except (ValueError, AssertionError, KeyError): return True
    raise AssertionError('hostile input was accepted')


def tamper_suite(doc):
    cases = {}
    def trial(name, edit):
        changed = json.loads(json.dumps(doc)); edit(changed)
        cases[name] = rejected(lambda: audit_receipt(changed, doc['identity'], complete=False))
    branch = next(iter(doc['branches']))
    r = lambda d: d['branches'][branch]['records'][0]
    trial('wrong_fixture', lambda d: d.update(fixture_sha256='0' * 64))
    trial('wrong_source', lambda d: d.update(identity='0' * 64))
    trial('hidden_wrong_teacher', lambda d: r(d)['write'].update(outcome=255))
    trial('wrong_branch_permission', lambda d: r(d).update(learn=not r(d)['learn']))
    trial('actual_call_permission', lambda d: r(d)['actual_calls'][0].update(write=not r(d)['learn']))
    trial('missing_store_call', lambda d: r(d)['actual_calls'].pop())
    trial('wrong_signed_coefficient', lambda d: r(d)['actual_calls'][-1].update(coefficients=[0., .5]))
    trial('wrong_emitted_byte', lambda d: r(d)['prediction'].update(emitted=255))
    trial('nonfinite_value', lambda d: r(d)['prediction']['combined'].__setitem__(0, float('nan')))
    trial('target_before_prediction', lambda d: r(d).update(prediction_precedes_outcome=False))
    trial('shifted_clock', lambda d: r(d).update(observed_at=r(d)['observed_at'] + 1))
    trial('duplicate_birth', lambda d: d['births'].__setitem__(1, dict(d['births'][0])))
    trial('wrong_record_index', lambda d: r(d).update(index=1))
    return cases


def cycle1():
    began = time.monotonic(); world = make_world(490101)
    with patch.object(stores, 'birth', wraps=stores.birth) as births:
        candidate = CompactCore()
        assert births.call_count == 4 and all(call.args == ('content',) for call in births.call_args_list)
    baseline = make_core(ARMS[0]); core_by_arm = {ARMS[0]: baseline, ARMS[1]: candidate}
    docs = {}
    for arm, core in core_by_arm.items():
        rows = []; trace = instrument(core)
        for event in world['events'][:12]: rows.append(record(core, event, 'N_old', trace))
        docs[arm] = {'identity': identity(), 'arm': arm, 'world': world['world'], 'births': core.births,
                     'fixture_sha256': world['sha256'],
                     'branches': {'N_old': {'records': rows, 'probes': [], 'births': core.births}}}
        audit_receipt(docs[arm], identity(), complete=False)
    assert baseline.bank_digests()[4:] == candidate.bank_digests()
    # Follow writes after the explicit off-history; same private trajectory, different fixed output.
    bt, ct = instrument(baseline), instrument(candidate)
    for event in world['events'][12:24]:
        a = record(baseline, event, 'W', bt); b = record(candidate, event, 'W', ct)
        assert a['prediction']['private'] == b['prediction']['private']
        assert baseline.bank_digests()[4:] == candidate.bank_digests()
    matrix = {branch: {stage: permitted(branch, stage) for stage in ('old', 'new', 'revision')} for branch in BRANCHES}
    assert matrix['N_revision'] == {'old': True, 'new': True, 'revision': False}
    assert cp_upper(0, 96, .05 / 6) < .05 and cp_upper(1, 96, .05 / 6) > .05
    assert cp_lower(96, 96, .05 / 6) > .9
    assert cp_upper(96, 96, .05 / 6) == 1. and cp_lower(0, 96, .05 / 6) == 0.
    result = {'cycle': 1, 'verdict': 'PASS', 'evidence': 'E0 physical deletion/causal instrumentation',
              'actual_birth_calls': 4, 'no_shared_store_born': True, 'private_parity_records': 24,
              'branch_matrix': matrix, 'tamper_rejections': tamper_suite(docs[ARMS[1]]),
              'exact_primary_boundary_tests': True, 'elapsed_s': time.monotonic() - began,
              'identity_at_trial': identity()}
    atomic_json(ROOT / 'results/CYCLE1.json', result)
    return result


def dev_child(world, arm, folder, *, resume=False, stop=None, short=False):
    args = [sys.executable, '-u', str(ROOT / 'cycle.py'), '--dev-job', str(world), '--arm', arm, '--folder', str(folder)]
    if resume: args.append('--resume')
    if stop: args += ['--stop', str(stop)]
    if short: args.append('--short')
    log = folder / ('child-' + arm + ('-resume' if resume else '-start') + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    with log.open('w') as f:
        child = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, timeout=2400)
    if child.returncode: raise RuntimeError(f'development worker failed: {log}')
    return time.monotonic() - began


def cycle2():
    began = time.monotonic(); folder = ROOT / 'results/development/cycle2'; docs = []; elapsed = {}
    for arm in ARMS:
        elapsed[arm] = dev_child(490102, arm, folder)
        doc = read(folder / 'receipts' / f'490102_{arm}.json.gz'); audit_receipt(doc, identity())
        docs.append(doc)
    paired_private_equal(*docs)
    estimate = sum(d['worker_s'] for d in docs) * 96
    if estimate > 144000: raise AssertionError('measured science work estimate exceeds 40-hour cap')
    if any(d['peak_rss_bytes'] > 1024**3 for d in docs): raise AssertionError('worker RSS cap')
    sizes = {arm: (folder / 'receipts' / f'490102_{arm}.json.gz').stat().st_size for arm in ARMS}
    result = {'cycle': 2, 'verdict': 'PASS', 'evidence': 'E0 complete life/branch/clock/private parity',
              'world': 490102, 'lives': 8, 'records': 6912, 'private_parity_every_checkpoint': True,
              'resource_trials': {d['arm']: {'worker_s': d['worker_s'], 'peak_rss_bytes': d['peak_rss_bytes']} for d in docs},
              'receipt_bytes': sizes, 'estimated_science_worker_h': estimate / 3600.,
              'child_elapsed_s': elapsed, 'elapsed_s': time.monotonic() - began, 'identity_at_trial': identity()}
    atomic_json(ROOT / 'results/CYCLE2.json', result)
    return result


def cycle3():
    began = time.monotonic(); results = {}; folder = ROOT / 'results/development/cycle3'
    for arm in ARMS:
        uninterrupted = folder / arm / 'uninterrupted'; resumed = folder / arm / 'resumed'
        dev_child(490103, arm, uninterrupted, short=True)
        dev_child(490103, arm, resumed, stop=32, short=True)
        cp = resumed / 'checkpoints' / f'490103_{arm}.npz'
        core, cursor = load(cp, arm, identity())
        assert cursor['next_record'] == 32 and len(cursor['branches']['W']['records']) == 32
        rejected(lambda: load(cp, arm, '0' * 64))
        # A freshly recomputed metadata checksum must not legitimise a skipped cursor.
        from compact_checkpoint import seal
        with np.load(cp, allow_pickle=False) as z: changed = {k: z[k].copy() for k in z.files}
        meta = json.loads(changed['metadata_json'].tobytes())
        meta.pop('seal'); meta['cursor']['next_record'] += 1; meta['seal'] = seal(meta)
        changed['metadata_json'] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        bad_cursor = folder / ('tamper-cursor-' + arm + '.npz'); np.savez_compressed(bad_cursor, **changed)
        rejected(lambda: load(bad_cursor, arm, identity()))
        dev_child(490103, arm, resumed, resume=True, short=True)
        a = read(uninterrupted / 'receipts' / f'490103_{arm}.json.gz')
        b = read(resumed / 'receipts' / f'490103_{arm}.json.gz')
        assert a['branches']['W']['records'] == b['branches']['W']['records']
        assert a['branches']['W']['final_private_digest'] == b['branches']['W']['final_private_digest']
        receipt = resumed / 'receipts' / f'490103_{arm}.json.gz'
        before = receipt.read_bytes()
        dev_child(490103, arm, resumed, resume=True, short=True)
        assert receipt.read_bytes() == before, 'completed receipt rerun/overwrite'
        with np.load(cp, allow_pickle=False) as z: arrays = {k: z[k].copy() for k in z.files}
        arrays['s0_fly_fast'].flat[0] += 1.
        bad = folder / ('tamper-' + arm + '.npz'); np.savez_compressed(bad, **arrays)
        rejected(lambda: load(bad, arm, identity()))
        results[arm] = {'fresh_process_resume_exact': True, 'checkpoint_cursor': 32,
                        'records_compared': 64, 'completed_receipt_unchanged': True,
                        'wrong_identity_rejected': True, 'tampered_array_rejected': True}
        results[arm]['resealed_skipped_cursor_rejected'] = True
    result = {'cycle': 3, 'verdict': 'PASS', 'evidence': 'E0 durable fresh-process recovery',
              'arms': results, 'elapsed_s': time.monotonic() - began, 'identity_at_trial': identity()}
    atomic_json(ROOT / 'results/CYCLE3.json', result)
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--cycle', type=int)
    p.add_argument('--dev-job', type=int); p.add_argument('--arm'); p.add_argument('--folder', type=Path)
    p.add_argument('--resume', action='store_true'); p.add_argument('--stop', type=int); p.add_argument('--short', action='store_true')
    args = p.parse_args()
    if args.dev_job:
        if args.dev_job not in (490101, 490102, 490103): raise ValueError('only development worlds')
        _, state = run_job(args.dev_job, args.arm, args.folder, identity(), resume=args.resume,
                           stop_after=args.stop, record_limit=64 if args.short else 864,
                           branch_limit=1 if args.short else 4,
                           progress_callback=lambda b,c: print(json.dumps({'branch': b, 'cursor': c}), flush=True))
        print(json.dumps(state)); return
    result = {1: cycle1, 2: cycle2, 3: cycle3}[args.cycle]()
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
