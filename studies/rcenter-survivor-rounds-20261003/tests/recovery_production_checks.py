# SPDX-License-Identifier: GPL-3.0-or-later
"""Small isolated mocks for the actual production recovery/admission functions.

No entry-point side effects, runtime numerical import, native learner, real
ledger, real receipt or external operation. Called once by the approved pure
Cycle-3 operations job; also suitable for bounded development mock checks.
"""
import copy
import fcntl
import importlib.util
import time
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import recovery as r
from durable import create, replace, journal, sha
from integrity import digest_map


def run_checks(temp_root):
    base = Path(temp_root) / 'production-recovery'
    base.mkdir(parents=True, exist_ok=False)
    source = {'mock/source.py': 'a' * 64}
    runtime = copy.deepcopy(r.RUNTIME)
    checks = []
    counter = 0

    def case():
        nonlocal counter
        counter += 1
        root = base / str(counter)
        (root / 'operations').mkdir(parents=True)
        return root

    def reject(fn, text=None):
        try:
            fn()
        except (RuntimeError, AssertionError, FileNotFoundError, FileExistsError, KeyError, ValueError, BlockingIOError) as exc:
            if text is not None:
                assert text in str(exc), (text, str(exc))
            return
        raise AssertionError('production guard accepted a forbidden mock transition')

    def loader(path, world, expected_source):
        # Minimal independent transport/manifest loader, no scientific engine.
        path = Path(path)
        m = r.read(path / 'manifest.json')
        r.check(m['complete'] is True and m['world'] == world, 'mock manifest identity')
        r.check(set(p.name for p in path.iterdir()) == set(m['parts']) | {'manifest.json'}, 'mock part set')
        for name, part in m['parts'].items():
            r.check(sha(path / name) == part['sha256'] and (path / name).stat().st_size == part['bytes'], 'mock part tamper')
        header = r.read(path / 'header.json')
        r.check(header['source'] == expected_source and m['source_digest'] == digest_map(expected_source), 'mock source')
        return {'manifest': m, 'manifest_sha256': sha(path / 'manifest.json'), 'header': header, 'scientific_digest': 'f' * 64}

    def plan(root, lock, mode='cycle3', **kwargs):
        return r.reconcile_and_plan(root, source, runtime, mode, lock=lock, receipt_loader=loader, **kwargs)

    def reserve(root, lock, key):
        return r.prepare_attempt(root, source, runtime, key.split('/')[0], key, lock=lock, host_token='mock-host-session', receipt_loader=loader)

    def ack_reservation(root, reservation):
        a = reservation['attempt']
        replace(root / 'operations/RESERVATION_ACK.json', {
            'reservation_id': a['reservation_id'], 'key': a['key'], 'source_digest': a['source_digest'],
            'charged_s': r.CAP, 'private_readback_verified': True,
            'ledger_sha256': reservation['ledger_sha256'], 'checkpoint_sha256': 'b' * 64,
            'content_identity': 'c' * 64, 'library_file_id': 'mock-owned-library-file', 'version': 1})

    def start(root, lock, reservation):
        ack_reservation(root, reservation)
        a = reservation['attempt']
        return r.start_reserved_attempt(root, a['reservation_id'], source, runtime, a['key'].split('/')[0],
                                        lock=lock, child_identity=r.process_identity(), host_token='mock-host-session', receipt_loader=loader)

    def write_receipt(root, a):
        p = root / a['receipt']
        resources = {'wall_s': 1.0, 'peak_rss_bytes': 1024}
        if a['key'] == 'cycle3/operations':
            create(p, {'schema': 'RC-SURVIVOR-CYCLE3-OPERATIONS-v1', 'passed': True, 'mock_only': True,
                       'native_teaching_events': 0, 'source': source, 'runtime': runtime, 'key': a['key'],
                       'attempt': a['attempt'], 'reservation_id': a['reservation_id'], 'resources': resources})
        else:
            world = int(a['key'].split('/')[1]) if a['key'].startswith('science/') else 310200
            create(p / 'header.json', {'source': source, 'runtime': runtime})
            create(p / 'payload.json', {'synthetic': True, 'world': world})
            parts = {name: {'sha256': sha(p / name), 'bytes': (p / name).stat().st_size} for name in ('header.json', 'payload.json')}
            create(p / 'manifest.json', {'complete': True, 'world': world,
                                       'kind': 'science' if a['key'].startswith('science/') else 'technical',
                                       'source_digest': digest_map(source), 'parts': parts, 'resources': resources})
            if a['key'] == 'cycle3/replay':
                create(root / 'receipts/cycle3/replay_comparison.json', {
                    'passed': True, 'fresh_process': True, 'same_world': 310200,
                    'first_manifest_sha256': sha(root / 'receipts/cycle3/pilot-310200/manifest.json'),
                    'replay_manifest_sha256': sha(p / 'manifest.json'), 'scientific_digest': 'f' * 64})

    def barrier(root, a):
        p = root / 'operations/PERSISTENCE_STATE.json'
        state = r.read(p) if p.exists() else {'worlds': {}, 'jobs': {}}
        group, key = ('worlds', a['key'].split('/')[1]) if a['key'].startswith('science/') else ('jobs', a['key'])
        state[group][key] = {'manifest_sha256': a['receipt_sha256'], 'private_readback_verified': True, 'github_tree_verified': True}
        replace(p, state)

    def complete(root, lock, key, persisted=True):
        reservation = reserve(root, lock, key)
        ledger, a = start(root, lock, reservation)
        write_receipt(root, a)
        out = r.finish_attempt(root, a['reservation_id'], source, runtime, 1.0, 1024, None, 0, lock=lock, receipt_loader=loader)
        if persisted:
            barrier(root, out)
        return out

    def approval(root, a, count=1, disposition='retry'):
        replace(root / 'operations/RECOVERY_ACCEPTED.json', {'accepted': True, 'independent_review': True,
            'classification': 'infrastructure', 'key': a['key'], 'attempt_count': count,
            'source_digest': a['source_digest'], 'runtime': runtime, 'reservation_id': a['reservation_id'], 'disposition': disposition})

    def simulate_dead_writer(root, lock):
        # The production /proc identity check is exercised with a disappeared
        # process table, after testing the real live-identity rejection.
        with patch.object(r, 'process_identity', return_value=None):
            return plan(root, lock)

    root = case()
    with r.exclusive_lock(root) as lock:
        p = plan(root, lock)
        assert p['missing'] == list(r.QUALIFICATION_KEYS) and p['next_key'] == 'cycle3/operations'
        reject(lambda: reserve(root, lock, 'cycle3/replay'), 'next admitted missing')
        reject(lambda: reserve(root, lock, 'cycle3/pilot'), 'next admitted missing')
        assert 'qualification_prerequisites_missing' in plan(root, lock, 'science')['blocked']
        a = complete(root, lock, 'cycle3/operations', False)
        assert plan(root, lock)['next_key'] is None and plan(root, lock)['pending_barriers'][0]['key'] == a['key']
        reject(lambda: reserve(root, lock, 'cycle3/pilot'))
        barrier(root, a)
        assert plan(root, lock)['next_key'] == 'cycle3/pilot'
        replace(root / 'operations/TRANSPORT_PENDING.json', {'phase': 'uncertain_upload'})
        assert 'unresolved_transport' in plan(root, lock)['blocked']
        replace(root / 'operations/TRANSPORT_PENDING.json', None)
        complete(root, lock, 'cycle3/pilot')
        complete(root, lock, 'cycle3/replay')
        p = plan(root, lock, 'science')
        assert p['missing'] == [f'science/{w}' for w in range(310001, 310065)] + ['analysis/final'] and p['next_key'] == 'science/310001'
        reject(lambda: reserve(root, lock, 'science/310002'))
        first = complete(root, lock, 'science/310001', False)
        assert plan(root, lock, 'science')['next_key'] is None
        barrier(root, first)
        assert plan(root, lock, 'science')['next_key'] == 'science/310002'
        original = (root / 'receipts/science/310001/payload.json').read_bytes()
        (root / 'receipts/science/310001/payload.json').write_bytes(b'{}')
        reject(lambda: plan(root, lock, 'science'), 'mock part tamper')
        (root / 'receipts/science/310001/payload.json').write_bytes(original)
        mp = root / 'receipts/science/310001/manifest.json'
        manifest_bytes = mp.read_bytes()
        mp.unlink()
        reject(lambda: plan(root, lock, 'science'))
        mp.write_bytes(manifest_bytes)
        assert plan(root, lock, 'science')['next_key'] == 'science/310002'
        checks += ['production_missing_only_fixed_64_order', 'production_operations_pilot_replay_prerequisites',
                   'production_private_public_pending_barriers', 'production_completed_part_tamper_rejected',
                   'production_deleted_completed_manifest_rejected']

    root = case()
    with r.exclusive_lock(root) as lock:
        reservation = reserve(root, lock, 'cycle3/operations')
        saved_before_worker = (root / 'operations/RUN_LEDGER.json').read_bytes()
        assert json.loads(saved_before_worker)['worker_s'] == r.CAP
        reject(lambda: r.start_reserved_attempt(root, reservation['attempt']['reservation_id'], source, runtime, 'cycle3',
                                                lock=lock, child_identity=r.process_identity(), host_token='mock-host-session', receipt_loader=loader))
        ledger, a = start(root, lock, reservation)
        reject(lambda: plan(root, lock), 'old writer still alive')
        create(root / a['receipt'], {'incomplete': True})
        p = simulate_dead_writer(root, lock)
        interrupted = p['ledger']['attempts'][0]
        assert interrupted['status'] == 'interrupted' and interrupted['charged_s'] == r.CAP and p['ledger']['worker_s'] == r.CAP
        assert not (root / a['receipt']).exists()
        assert r.read(root / interrupted['quarantine_path']) == {'incomplete': True}
        anchor = p['ledger']['journal_anchor']
        p2 = plan(root, lock)
        assert p2['ledger']['worker_s'] == r.CAP and p2['ledger']['journal_anchor'] == anchor
        assert 'retry_requires_independent_infrastructure_approval' in p2['blocked']
        reject(lambda: reserve(root, lock, 'cycle3/operations'))
        approval(root, interrupted)
        prior_approval = r.read(root / 'operations/RECOVERY_ACCEPTED.json')
        replace(root / 'operations/RECOVERY_ACCEPTED.json', dict(prior_approval, attempt_count=2))
        reject(lambda: reserve(root, lock, 'cycle3/operations'))
        replace(root / 'operations/RECOVERY_ACCEPTED.json', prior_approval)
        second = reserve(root, lock, 'cycle3/operations')
        assert second['attempt']['attempt'] == 2 and r.load_ledger(root, lock=lock)['worker_s'] == 2 * r.CAP
        assert (root / interrupted['quarantine_path']).exists()
        ledger, second_attempt = start(root, lock, second)
        write_receipt(root, second_attempt)
        finished = r.finish_attempt(root, second_attempt['reservation_id'], source, runtime, 1.0, 1024, None, 0, lock=lock, receipt_loader=loader)
        barrier(root, finished)
        assert plan(root, lock)['next_key'] == 'cycle3/pilot'
        assert (root / finished['receipt']).exists() and (root / interrupted['quarantine_path']).exists()
        assert plan(root, lock)['ledger']['worker_s'] == r.CAP + 1
        checks.append('production_retry_receipt_not_quarantined_as_previous_attempt')
        checks += ['production_no_run_before_backed_reservation', 'production_live_writer_identity_rejected',
                   'production_unknown_duration_exactly_900_once', 'production_partial_quarantine_preserved',
                   'production_exact_key_count_source_approval_required', 'production_prejob_checkpoint_contains_full_charge']

    root = case()
    with r.exclusive_lock(root) as lock:
        reservation = reserve(root, lock, 'cycle3/operations')
        ledger, a = start(root, lock, reservation)
        write_receipt(root, a)
        p = simulate_dead_writer(root, lock)
        pending = p['ledger']['attempts'][0]
        assert pending['status'] == 'receipt_complete_pending_review' and p['ledger']['worker_s'] == r.CAP
        assert p['next_key'] is None and (root / a['receipt']).exists()
        assert plan(root, lock)['ledger']['journal_anchor'] == p['ledger']['journal_anchor']
        approval(root, pending, disposition='accept_completed_receipt')
        p = plan(root, lock)
        assert p['ledger']['attempts'][0]['status'] == 'completed' and p['ledger']['worker_s'] == r.CAP
        assert p['pending_barriers'][0]['key'] == 'cycle3/operations'
        assert plan(root, lock)['ledger']['worker_s'] == r.CAP
        checks.append('production_complete_receipt_crash_requires_independent_classification')

    root = case()
    with r.exclusive_lock(root) as lock:
        reservation = reserve(root, lock, 'cycle3/operations')
        reject(lambda: r.reconcile_and_plan(root, {'different': 'd' * 64}, runtime, 'cycle3', lock=lock, receipt_loader=loader), 'source transition')
        reject(lambda: r.reconcile_and_plan(root, source, dict(runtime, python='0'), 'cycle3', lock=lock, receipt_loader=loader), 'runtime mismatch')
        reject(lambda: r.limits(root, {'worker_s': r.TOTAL + 1, 'active_wall_s': 0}), 'worker cap')
        reject(lambda: r.limits(root, {'worker_s': 0, 'active_wall_s': 0}, active_wall_s=r.WALL + 1), 'host-wall cap')
        reject(lambda: r.limits(root, {'worker_s': 0, 'active_wall_s': 0}, output_bytes=r.OUTPUT_CAP + 1), 'output cap')
        reject(lambda: r.limits(root, {'worker_s': 0, 'active_wall_s': 0}, free_bytes=r.RESERVE - 1), 'disk reserve')
        r.mkdir(root / 'operations/mock-tests')
        (root / 'operations/mock-tests/artifact').write_bytes(b'x' * 13)
        before = r.output_size(root)
        r.mkdir(root / 'operations/unknown-artifacts')
        (root / 'operations/unknown-artifacts/file').write_bytes(b'x' * 17)
        assert r.output_size(root) == before + 17
        checks += ['production_source_runtime_mismatch_rejected', 'production_worker_wall_storage_budgets', 'production_mocks_and_new_artifacts_counted']

    # Corrupt/drop immutable WAL entries, alter the mutable anchored ledger, and
    # recover a real append-before-replace crash without charging a second time.
    root = case()
    with r.exclusive_lock(root) as lock:
        reservation = reserve(root, lock, 'cycle3/operations')
        lp = root / 'operations/RUN_LEDGER.json'
        saved = lp.read_bytes()
        ledger = r.load_ledger(root, lock=lock)
        replace(lp, dict(ledger, worker_s=0))
        reject(lambda: r.load_ledger(root, lock=lock), 'tampered')
        lp.write_bytes(saved)
        j = root / 'operations/journal/000000.json'
        jb = j.read_bytes()
        j.unlink()
        reject(lambda: r.load_ledger(root, lock=lock), 'tail deleted')
        j.write_bytes(jb)
        old = r.load_ledger(root, lock=lock)
        new = copy.deepcopy(old)
        new['attempts'][0]['status'] = 'interrupted'
        real_replace = r.replace
        def crash_replace(path, data):
            if Path(path) == lp:
                raise OSError('isolated mock crash before ledger replace')
            return real_replace(path, data)
        with patch.object(r, 'replace', side_effect=crash_replace):
            try:
                r.commit_ledger(root, old, new, 'mock_uncommitted_tail', lock=lock)
            except OSError:
                pass
            else:
                raise AssertionError('mock crash not injected')
        recovered = r.load_ledger(root, lock=lock)
        assert recovered['attempts'][0]['status'] == 'interrupted' and recovered['worker_s'] == r.CAP
        assert r.load_ledger(root, lock=lock) == recovered
        checks += ['production_ledger_and_journal_anchor_tamper_rejected', 'production_wal_crash_tail_replayed_once']

    root = case()
    with r.exclusive_lock(root) as lock:
        for number in range(1, 4):
            reservation = reserve(root, lock, 'cycle3/operations')
            ledger, a = start(root, lock, reservation)
            out = r.finish_attempt(root, a['reservation_id'], source, runtime, 0.1, 1024, 'mock infrastructure failure', 1, lock=lock, receipt_loader=loader)
            approval(root, out, number)
        assert 'exact_key_attempt_cap' in plan(root, lock)['blocked']
        reject(lambda: reserve(root, lock, 'cycle3/operations'))
        assert abs(r.load_ledger(root, lock=lock)['worker_s'] - 0.3) < 1e-12
        checks.append('production_maximum_three_attempts_and_no_double_worker_charge')

    root = case()
    with r.exclusive_lock(root) as lock:
        reservation = reserve(root, lock, 'cycle3/operations')
        create(root / reservation['attempt']['receipt'], {'partial': True})
        with patch.object(r.os, 'rename', side_effect=OSError('mock crash before quarantine rename')):
            try:
                plan(root, lock)
            except OSError:
                pass
            else:
                raise AssertionError('quarantine crash not exercised')
        pending = r.load_ledger(root, lock=lock)['attempts'][0]
        assert pending['quarantine_path'] and (root / pending['receipt']).exists()
        p = plan(root, lock)
        assert (root / pending['quarantine_path']).exists() and not (root / pending['receipt']).exists()
        assert p['ledger']['worker_s'] == r.CAP
        checks.append('production_quarantine_intent_crash_recovery_idempotent')

    # A restored reservation is already charged and cannot be started under a
    # new keeper token, even if the caller bypasses its normal initial plan.
    root = case()
    with r.exclusive_lock(root) as lock:
        reservation = reserve(root, lock, 'cycle3/operations')
        ack_reservation(root, reservation)
        reject(lambda: r.start_reserved_attempt(root, reservation['attempt']['reservation_id'], source, runtime, 'cycle3',
            lock=lock, child_identity=r.process_identity(), host_token='different-new-host', receipt_loader=loader), 'lost host session')
        p = plan(root, lock)
        assert p['ledger']['worker_s'] == r.CAP and p['ledger']['attempts'][0]['status'] == 'interrupted'
        assert 'retry_requires_independent_infrastructure_approval' in p['blocked']
        checks.append('production_restored_prejob_reservation_cannot_forget_or_restart_charge')

    # Exercise the production budget gate at a small isolated ceiling. This
    # does not edit the programme's constant or write a real ledger.
    root = case()
    with r.exclusive_lock(root) as lock:
        with patch.object(r, 'TOTAL', r.CAP - 1):
            reject(lambda: reserve(root, lock, 'cycle3/operations'), 'total worker cap')
        assert not (root / 'operations/RUN_LEDGER.json').exists()
        checks.append('production_budget_exhaustion_blocks_prejob_reservation')

    # Small accepted historical source maps stand in for the immutable C1/C2
    # files. The real historical validator is used without an override.
    root = case()
    historical = {'schema': 'RC-SURVIVOR-LEDGER-v1', 'worker_s': 0.0, 'active_wall_s': 0.0, 'attempts': []}
    for cycle in (1, 2):
        prefix = f'CYCLE{cycle}'
        snapshot = root / f'history/cycle{cycle}'
        create(snapshot / 'source_snapshot/mock/accepted.json', {'cycle': cycle})
        old_source = {'mock/accepted.json': sha(snapshot / 'source_snapshot/mock/accepted.json')}
        create(root / f'receipts/cycle{cycle}/smoke.json', {'passed': True, 'source': old_source})
        receipt_sha = sha(root / f'receipts/cycle{cycle}/smoke.json')
        a = {'key': f'cycle{cycle}/test', 'status': 'completed', 'charged_s': 1.0, 'runtime': runtime,
             'source_digest': digest_map(old_source), 'receipt_sha256': receipt_sha}
        historical['attempts'].append(a)
        historical['worker_s'] += 1
        historical['active_wall_s'] += 1
        create(root / f'audits/{prefix}_LEDGER_SNAPSHOT.json', historical)
        create(root / f'audits/{prefix}_SOURCE_REVIEW.md', {'report': cycle})
        review = {'accepted': True, 'source_hashes': old_source,
                  'report_sha256': sha(root / f'audits/{prefix}_SOURCE_REVIEW.md')}
        create(root / f'audits/{prefix}_SOURCE_REVIEW.json', review)
        review_sha = sha(root / f'audits/{prefix}_SOURCE_REVIEW.json')
        create(snapshot / 'SNAPSHOT.json', {'source_review_sha256': review_sha, 'exact_accepted_source_map': old_source})
        create(root / f'audits/{prefix}_ACCEPTED.md', {'accepted_report': cycle})
        acceptance = {'accepted': True, 'source_review_sha256': review_sha, 'source_digest': digest_map(old_source),
            'receipt_sha256': receipt_sha, 'report_file': f'audits/{prefix}_ACCEPTED.md',
            'report_sha256': sha(root / f'audits/{prefix}_ACCEPTED.md'),
            'ledger_snapshot': f'audits/{prefix}_LEDGER_SNAPSHOT.json',
            'ledger_snapshot_sha256': sha(root / f'audits/{prefix}_LEDGER_SNAPSHOT.json'),
            'input_hashes': {f'receipts/cycle{cycle}/smoke.json': receipt_sha, f'audits/{prefix}_SOURCE_REVIEW.json': review_sha}}
        create(root / f'audits/{prefix}_ACCEPTED.json', acceptance)
    replace(root / 'operations/RUN_LEDGER.json', historical)
    historical_bytes = (root / 'operations/RUN_LEDGER.json').read_bytes()
    with r.exclusive_lock(root) as lock:
        assert plan(root, lock)['next_key'] == 'cycle3/operations'
        receipt = root / 'receipts/cycle1/smoke.json'
        saved = receipt.read_bytes()
        receipt.unlink()
        reject(lambda: plan(root, lock))
        receipt.write_bytes(saved)
        replace(receipt, {'passed': True, 'source': source})
        reject(lambda: plan(root, lock), 'drift')
        receipt.write_bytes(saved)
        snap = root / 'history/cycle2/source_snapshot/mock/accepted.json'
        snap_bytes = snap.read_bytes()
        snap.write_bytes(b'{}')
        reject(lambda: plan(root, lock), 'source bytes drift')
        snap.write_bytes(snap_bytes)
        assert (root / 'operations/RUN_LEDGER.json').read_bytes() == historical_bytes
        assert plan(root, lock)['ledger']['attempts'] == historical['attempts']
        checks += ['production_historical_accepted_source_maps_and_receipts_checked',
                   'production_historical_attempts_preserved_byte_for_byte']

    # Host lifetime/18h checking is an actual supervisor call with a local
    # disposable coordinator lock and tiny JSON state, without admission/run.
    root = case()
    spec = importlib.util.spec_from_file_location('mock_production_supervisor', Path(__file__).resolve().parents[1] / 'operations/supervise.py')
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    supervisor.ROOT = root
    with (root / 'operations/HOST_COORDINATOR.lock').open('a') as keeper:
        fcntl.flock(keeper, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {'effective_s': 7.0, 'cumulative_s': 2.0, 'active': True, 'host_token': 'mock-host',
                 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'updated_monotonic_s': time.monotonic()}
        replace(root / 'operations/HOST_WALL.json', state)
        with patch.dict(os.environ, {'SURVIVOR_HOST_TOKEN': 'mock-host'}):
            assert supervisor.host_wall() == 7.0
            replace(root / 'operations/HOST_WALL.json', dict(state, updated_monotonic_s=time.monotonic() - 6))
            reject(supervisor.host_wall, 'heartbeat stale')
            replace(root / 'operations/HOST_WALL.json', dict(state, host_token='different'))
            reject(supervisor.host_wall, 'session token mismatch')
            replace(root / 'operations/HOST_WALL.json', dict(state, updated_monotonic_s=time.monotonic()))
            fcntl.flock(keeper, fcntl.LOCK_UN)
            reject(supervisor.host_wall, 'lock is not held')
    checks.append('production_supervisor_host_heartbeat_token_and_lock_guards')

    # Run the exact production startup wrapper with a tiny marker entry. Reuse
    # its live PID in stale records: no source/key/reservation/host mismatch may
    # permit the entry to execute before its current admission is published.
    root = case()
    marker = root / 'entered.json'
    entry = root / 'marker_entry.py'
    entry.write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('entered')\n")
    admission = root / 'operations/ACTIVE_JOB.json'
    expected = {'source_digest': 'a' * 64, 'reservation_id': 'b' * 32,
                'key': 'cycle3/operations', 'host_token': 'fresh-wrapper-host'}
    env = dict(os.environ, SUPERVISOR_PID=str(os.getpid()), ADMISSION=str(admission), ENTRY=str(entry),
               SURVIVOR_SOURCE_DIGEST=expected['source_digest'], SURVIVOR_RESERVATION_ID=expected['reservation_id'],
               SURVIVOR_JOB_KEY=expected['key'], SURVIVOR_HOST_TOKEN=expected['host_token'])
    with r.exclusive_lock(root) as lock:
        env['SUPERVISOR_LOCK_FD'] = str(lock.handle.fileno())
        child = subprocess.Popen([sys.executable, '-c', supervisor.WRAPPER], env=env,
                                 pass_fds=(lock.handle.fileno(),), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            expected['child_pid'] = child.pid
            for field in ('reservation_id', 'key', 'source_digest', 'host_token', 'child_pid'):
                stale = dict(expected, **{field: child.pid + 1 if field == 'child_pid' else 'stale-value'})
                replace(admission, stale)
                time.sleep(0.08)
                assert child.poll() is None and not marker.exists(), ('stale admission entered worker', field)
            replace(admission, expected)
            child.wait(timeout=5)
            assert child.returncode == 0 and marker.read_text() == 'entered'
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
            child.stderr.close()
    checks.append('production_reused_pid_stale_reservation_source_key_host_startup_rejected')

    # Independently accepted exact old->new revision transition on one failed
    # pure operations attempt. All files below are synthetic private temp data.
    root = case()
    snapshot = root / 'history/failed-pure-attempt'
    create(snapshot / 'source_snapshot/mock/old.json', {'revision': 'old'})
    old_source = {'mock/old.json': sha(snapshot / 'source_snapshot/mock/old.json')}
    new_source = {'mock/new.json': '2' * 64}
    with r.exclusive_lock(root) as lock:
        reservation = r.prepare_attempt(root, old_source, runtime, 'cycle3', 'cycle3/operations', lock=lock,
                                        host_token='old-mock-host', receipt_loader=loader)
        ack_reservation(root, reservation)
        ledger, old = r.start_reserved_attempt(root, reservation['attempt']['reservation_id'], old_source, runtime, 'cycle3',
                                               lock=lock, child_identity=r.process_identity(), host_token='old-mock-host', receipt_loader=loader)
        (root / old['log']).write_text('synthetic infrastructure-only failure\n')
        old = r.finish_attempt(root, old['reservation_id'], old_source, runtime, 3.25, 1024, 'synthetic infrastructure failure', 1,
                               lock=lock, receipt_loader=loader)
        failed = r.load_ledger(root, lock=lock)
        create(snapshot / 'FAILED_LEDGER.json', failed)
        (snapshot / 'worker.log').write_bytes((root / old['log']).read_bytes())
        create(snapshot / 'CYCLE3_SOURCE_REVIEW.md', {'old_source_independently_accepted': True})
        create(snapshot / 'CYCLE3_SOURCE_REVIEW.json', {'accepted': True, 'source_hashes': old_source,
                'source_digest': digest_map(old_source), 'report_sha256': sha(snapshot / 'CYCLE3_SOURCE_REVIEW.md')})
        create(snapshot / 'SNAPSHOT.json', {'source_hashes': old_source, 'source_digest': digest_map(old_source),
                'source_review_sha256': sha(snapshot / 'CYCLE3_SOURCE_REVIEW.json'),
                'report_sha256': sha(snapshot / 'CYCLE3_SOURCE_REVIEW.md'),
                'failed_ledger_sha256': sha(snapshot / 'FAILED_LEDGER.json'), 'worker_log_sha256': sha(snapshot / 'worker.log')})
        create(root / 'audits/CYCLE3_SOURCE_REVIEW.md', {'new_source_independently_accepted': True})
        create(root / 'audits/CYCLE3_SOURCE_REVIEW.json', {'accepted': True, 'source_hashes': new_source,
                'source_digest': digest_map(new_source), 'report_sha256': sha(root / 'audits/CYCLE3_SOURCE_REVIEW.md')})
        def changed_plan(candidate=new_source):
            return r.reconcile_and_plan(root, candidate, runtime, 'cycle3', lock=lock, receipt_loader=loader)
        reject(changed_plan, 'source transition missing')
        transition_path = root / f'operations/RECOVERY_TRANSITIONS/{old["reservation_id"]}.json'
        transition = {'schema': 'RC-SURVIVOR-SOURCE-TRANSITION-v1', 'accepted': True, 'independent_review': True,
                'classification': 'infrastructure', 'disposition': 'retry', 'key': old['key'], 'attempt_count': 1,
                'runtime': runtime, 'reservation_id': old['reservation_id'], 'source_digest': digest_map(old_source),
                'target_source_digest': digest_map(new_source), 'snapshot_dir': str(snapshot.relative_to(root)),
                'snapshot_sha256': sha(snapshot / 'SNAPSHOT.json'), 'attempt_entry_sha256': digest_map(old),
                'current_source_review_sha256': sha(root / 'audits/CYCLE3_SOURCE_REVIEW.json')}
        create(transition_path, transition)
        assert 'retry_requires_independent_infrastructure_approval' in changed_plan()['blocked']
        approval(root, old)
        top = r.read(root / 'operations/RECOVERY_ACCEPTED.json')
        top.update(target_source_digest=digest_map(new_source), source_transition_sha256=sha(transition_path))
        replace(root / 'operations/RECOVERY_ACCEPTED.json', top)
        assert changed_plan()['next_key'] == 'cycle3/operations'
        for path in (snapshot / 'source_snapshot/mock/old.json', snapshot / 'worker.log', root / old['log'], snapshot / 'FAILED_LEDGER.json'):
            saved = path.read_bytes()
            path.write_bytes(b'changed')
            reject(changed_plan)
            path.write_bytes(saved)
        for key, value in (('charged_s', 0.0), ('reservation_id', 'f' * 32), ('source_digest', 'f' * 64), ('status', 'completed')):
            changed = copy.deepcopy(old)
            changed[key] = value
            reject(lambda: r.source_transition(root, changed, new_source, runtime))
        reject(lambda: changed_plan({'mock/wrong-new.json': '3' * 64}))
        for name, value in (('target_source_digest', '0' * 64), ('source_transition_sha256', '0' * 64), ('attempt_count', 2),
                            ('source_digest', digest_map(new_source)), ('reservation_id', '0' * 32)):
            replace(root / 'operations/RECOVERY_ACCEPTED.json', dict(top, **{name: value}))
            assert changed_plan()['next_key'] is None
        replace(root / 'operations/RECOVERY_ACCEPTED.json', top)
        second = r.prepare_attempt(root, new_source, runtime, 'cycle3', 'cycle3/operations', lock=lock,
                                   host_token='new-mock-host', receipt_loader=loader)
        after = r.load_ledger(root, lock=lock)
        assert after['attempts'][0] == old and after['worker_s'] == 903.25
        assert second['attempt']['source_digest'] == digest_map(new_source) and second['attempt']['attempt'] == 2
        # The historical per-reservation proof survives replacement of the
        # current retry approval; it grants no permission to run another key.
        replace(root / 'operations/RECOVERY_ACCEPTED.json', {'unrelated': True})
        r.validate_attempts(root, after, new_source, runtime)
        checks += ['production_failed_old_source_requires_independent_exact_transition',
                   'production_source_transition_verifies_full_old_source_log_and_attempt',
                   'production_source_transition_rejects_changed_charge_identity_and_completed_status',
                   'production_source_transition_requires_exact_new_revision_and_retry_approval',
                   'production_source_transition_preserves_old_attempt_and_charges_new_reservation']

    from analysis_production_checks import run_checks as analysis_checks
    checks += analysis_checks(temp_root)
    return checks
