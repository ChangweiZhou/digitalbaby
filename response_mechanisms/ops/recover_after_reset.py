"""Operational reset recovery; the original scientific source and lock stay immutable.

The explicitly authorized cumulative worker cap is 12 hours (superseding 8).
The original 12-hour elapsed-wall deadline and all other bounds remain in force.
Simulation subprocesses use the original locked assay and exact original arguments.
"""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from locked_run import verify_lock, atomic, rss, stop_process, EXPECTED_PARAMS
from audit_receipts import load, validate

LOCK_SHA = '24943155d4b010c8e17746a265803155e13585a7b0b23fe261f3c0bae95a63fe'
APPROVED_WORKER_HOURS = 12.0
LOSS_REPORT = ROOT / 'results/final/operations/RESET_RECOVERY_20261001.json'
AMENDMENT = ROOT / 'RECOVERY_AMENDMENT_20261001.md'
DEST = ROOT / 'results/final'


def drive():
    guard = ROOT / 'scratch/response-supervisor.lock'
    guard.parent.mkdir(parents=True, exist_ok=True)
    guard_handle = guard.open('a')
    fcntl.flock(guard_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock, digest = verify_lock()
    assert digest == LOCK_SHA
    assert AMENDMENT.exists(), 'Missing approved operational amendment'
    assert lock['resource_caps']['worker_hours'] == 8.0
    assert lock['resource_caps']['workers'] == 1
    caps = dict(lock['resource_caps'], worker_hours=APPROVED_WORKER_HOURS)
    ledger_path = DEST / 'RUN_LEDGER.json'
    status_path = DEST / 'RUN_STATUS.json'
    history = json.loads(ledger_path.read_text())
    assert history['lock_sha256'] == digest
    loss = json.loads(LOSS_REPORT.read_text())
    assert history['started_unix'] == loss['accounting']['original_started_unix']
    if not history.get('reset_recovery_accounting_applied'):
        assert history['active_job'] is None
        assert history['worker_seconds'] == loss['accounting']['restored_remote_ledger_worker_seconds']
        assert len(history['jobs']) == 14
        original = DEST / 'operations/PRE_RESET_REMOTE_LEDGER.json'
        if original.exists():
            assert json.loads(original.read_text()) == history
        else:
            with original.open('x') as f:
                json.dump(history, f, indent=2, sort_keys=True); f.write('\n')
        charge = loss['accounting']['conservative_prior_charge_seconds'] - history['worker_seconds']
        assert charge >= 0
        history['jobs'].append(dict(world=None, arm=None, replay=False, seconds=charge,
            exit_code=None, reason='Conservative lost-work/reset accounting adjustment; not a measured life runtime',
            accounting_only=True, source='operations/RESET_RECOVERY_20261001.json'))
        history['worker_seconds'] += charge
        history['reset_recovery_accounting_applied'] = True
        history['recovered_receipts_at_reset'] = dict(primary=7, replay=7)
        history['missing_receipts_at_reset'] = 217
    if history.get('active_job'):
        prior = history['active_job']
        charge = max(0., time.time() - prior['started_unix'])
        history['jobs'].append({**prior, 'seconds': charge, 'exit_code': None,
            'reason': 'Unobserved interruption conservatively charged'})
        history['worker_seconds'] += charge
        history['active_job'] = None
    history['effective_worker_hours_cap'] = APPROVED_WORKER_HOURS
    history['original_worker_hours_cap_superseded'] = 8.0
    history['operational_amendment'] = 'RECOVERY_AMENDMENT_20261001.md'
    history['operational_launcher_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    atomic(ledger_path, history)
    done = []

    def checked(path, world, arm):
        doc = load(path)
        validate(doc, expected_world=world, expected_arm=arm, expected_bouts=lock['bouts'],
            expected_sources=lock['source_hashes'], expected_params=lock['params'],
            expected_runtime=lock['expected_receipt_runtime'], lock_sha256=digest)
        return doc

    def cap_reason(elapsed=0., pid=None):
        if elapsed > caps['per_life_seconds']: return 'per-life time cap'
        if pid is not None and rss(pid) > caps['peak_rss_bytes']: return 'worker RSS cap'
        if history['worker_seconds'] + elapsed > caps['worker_hours'] * 3600: return 'amended cumulative worker-hours cap'
        if time.time() - history['started_unix'] > caps['active_session_hours'] * 3600: return 'original elapsed wall cap'
        if sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()) > caps['results_bytes']: return 'results disk cap'
        return None

    # Original preselected replays survive; validate rather than repeat them.
    for world, arm in lock['replay_jobs']:
        first = checked(DEST / arm / f'{world}.json.gz', world, arm)
        again = checked(DEST / 'replays' / arm / f'{world}.json.gz', world, arm)
        assert first['resource']['process_id'] != again['resource']['process_id']
        first.pop('resource'); again.pop('resource'); assert first == again
    assert json.loads((DEST / 'REPLAY_AUDIT.json').read_text())['pass_all']

    for world in lock['worlds']:
        for arm in lock['arms']:
            verify_lock()
            target = DEST / arm / f'{world}.json.gz'
            if target.exists():
                checked(target, world, arm)
                done.append([world, arm])
                continue
            reason = cap_reason()
            if reason:
                atomic(status_path, dict(state='budget_stop', reason=reason, completed=len(done), expected=224))
                raise RuntimeError(reason)
            tag = f'{world}-{arm}-attempt-{time.time_ns()}'
            stdout = DEST / 'operations' / f'{tag}.stdout.txt'
            stderr = DEST / 'operations' / f'{tag}.stderr.txt'
            command = [sys.executable, str(ROOT / 'src/assay.py'), '--arm', arm, '--world', str(world),
                '--bouts', str(lock['bouts']), '--kind', 'final', '--params', json.dumps(EXPECTED_PARAMS, separators=(',', ':'))]
            history['active_job'] = dict(world=world, arm=arm, replay=False, started_unix=time.time(),
                stdout=str(stdout.relative_to(DEST)), stderr=str(stderr.relative_to(DEST)))
            atomic(ledger_path, history)
            began = time.monotonic(); reason = None
            with stdout.open('x') as out, stderr.open('x') as err:
                process = subprocess.Popen(command, stdout=out, stderr=err,
                    pass_fds=(guard_handle.fileno(),))
                history['active_job']['process_id'] = process.pid
                atomic(ledger_path, history)
                while process.poll() is None:
                    elapsed = time.monotonic() - began
                    reason = cap_reason(elapsed, process.pid)
                    atomic(status_path, dict(state='running', world=world, arm=arm, replay=False,
                        completed=len(done), replayed=7, expected=224, elapsed_current_seconds=elapsed,
                        worker_seconds=history['worker_seconds']+elapsed, effective_worker_hours_cap=12,
                        recovery='Recomputed missing receipt after workspace loss; original science lock unchanged'))
                    if reason:
                        stop_process(process); break
                    time.sleep(2)
            elapsed = time.monotonic() - began
            # Post-exit checks close the original launcher's final-poll accounting gap.
            reason = reason or cap_reason(elapsed)
            entry = dict(world=world, arm=arm, replay=False, exit_code=process.returncode,
                seconds=elapsed, reason=reason, reset_recovery=True,
                stdout=str(stdout.relative_to(DEST)), stderr=str(stderr.relative_to(DEST)))
            history['jobs'].append(entry); history['worker_seconds'] += elapsed
            history['active_job'] = None; atomic(ledger_path, history)
            if reason or process.returncode != 0:
                atomic(status_path, dict(state='technical_stop', job=entry, completed=len(done), expected=224))
                raise RuntimeError(json.dumps(entry))
            doc = checked(target, world, arm)
            if doc['resource']['wall_seconds'] > caps['per_life_seconds'] or doc['resource']['peak_rss_bytes'] > caps['peak_rss_bytes']:
                atomic(status_path, dict(state='technical_stop', reason='completed receipt resource cap', completed=len(done), expected=224))
                raise RuntimeError('Completed receipt resource cap')
            done.append([world, arm])
            print(json.dumps(dict(completed=len(done), world=world, arm=arm, seconds=elapsed,
                worker_hours=history['worker_seconds']/3600)), flush=True)
        atomic(DEST / 'publication_queue' / f'{world}.json', dict(world=world, completed=len(done),
            lock_sha256=digest, source='Recovery launcher completed and validated full world; publication is independent'))
    reason = cap_reason()
    if reason: raise RuntimeError(reason)
    atomic(status_path, dict(state='complete_pending_final_audit', completed=len(done), replayed=7,
        expected=224, lock_sha256=digest, effective_worker_hours_cap=12,
        worker_seconds=history['worker_seconds'], original_started_unix=history['started_unix']))
    print(json.dumps(dict(completed=len(done), replayed=7, worker_hours=history['worker_seconds']/3600)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['verify', 'run'])
    args = parser.parse_args()
    if args.action == 'verify':
        print(json.dumps(dict(lock_sha256=verify_lock()[1], operational_launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))
    else:
        drive()
