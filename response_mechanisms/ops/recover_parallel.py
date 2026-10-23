"""Operational two-worker recovery; immutable scientific source/receipts stay untouched.

Candidate only. `run` requires separate explicit approval and independent review
records bound to this exact source hash. `preflight` is strictly read-only.
No simulation runs on import, verification, preflight, or mock tests.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path('/workspace/scratch/cbb599bc73b9/minifly-response/response_mechanisms')
PYTHON = '/workspace/scratch/cbb599bc73b9/minifly-env/bin/python'
LOCK_SHA = '24943155d4b010c8e17746a265803155e13585a7b0b23fe261f3c0bae95a63fe'
ORIGINAL_START = 1790872804.660057
DEADLINE = ORIGINAL_START + 12 * 3600
PRIOR_LOSS_CHARGE = 17245.33994293213
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')}
WORKER_CAP = 12 * 3600.
WORKERS = 2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, data):
    """Durable single-writer replacement, without predictable shared temp names."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x') as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_once(path, data):
    """Immutable operational evidence, separate from scientific receipts."""
    with Path(path).open('x') as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def acquire_guard(root, *, create=False):
    path = root / 'scratch/response-supervisor.lock'
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a' if create else 'r')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError('Launch blocked: existing supervisor or inherited child still holds the exclusive flock')
    # NEVER LOCK_UN: inherited children must retain the shared open-file-description
    # lock even if this supervisor dies. Closing only our fd is safe.
    return handle


def matching_processes(root):
    """Supplement flock with exact script-argument detection of legacy launchers."""
    scripts = {str(root / 'src/assay.py'), str(root / 'src/locked_run.py'),
               str(root / 'ops/recover_after_reset.py'), str(root / 'ops/recover_parallel.py')}
    found = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if int(path.parent.name) == os.getpid():
            continue
        try:
            args = path.read_bytes().decode().rstrip('\0').split('\0')
            if scripts.intersection(args):
                found.append(dict(pid=int(path.parent.name), command=args))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (PermissionError, UnicodeDecodeError):
            raise RuntimeError(f'Cannot inspect process {path.parent.name}; unresolved launch overlap')
    return found


def rss(pid):
    try:
        lines = Path(f'/proc/{pid}/status').read_text().splitlines()
        return max([int(line.split()[1]) * 1024 for line in lines
                    if line.startswith(('VmRSS:', 'VmHWM:'))] or [0])
    except (FileNotFoundError, ProcessLookupError):
        return 0


def result_bytes(dest):
    total = 0
    for path in dest.rglob('*'):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass  # A publisher's atomic temp was renamed during enumeration.
    return total


def active_records(history):
    records = list(history.get('active_jobs', []))
    if history.get('active_job'):
        if records:
            raise RuntimeError('Ambiguous legacy and parallel active jobs; do not discard either')
        records.append(history['active_job'])
    return records


def reconcile(history, now, *, start=ORIGINAL_START, minimum=PRIOR_LOSS_CHARGE):
    """Pure migration; caller holds flock and persists the before image first."""
    h = copy.deepcopy(history)
    assert h['lock_sha256'] == LOCK_SHA
    assert h['started_unix'] == start, 'Original elapsed start must never reset'
    assert h.get('reset_recovery_accounting_applied') is True
    assert math.isfinite(h['worker_seconds']) and h['worker_seconds'] >= minimum
    amounts = [j['seconds'] for j in h['jobs']]
    assert all(math.isfinite(x) and x >= 0 for x in amounts)
    assert abs(math.fsum(amounts) - h['worker_seconds']) < 1e-6, 'Ledger does not reconcile'
    if h.get('terminal_failure'):
        raise RuntimeError('Existing technical failure requires independent disposition before restart')
    records = active_records(h)
    assert len(records) <= WORKERS
    identities = [(r['world'], r['arm']) for r in records]
    assert len(set(identities)) == len(identities), 'Duplicate active job reservation'
    for prior in records:
        assert math.isfinite(prior['started_unix']) and prior['started_unix'] <= now
        charge = now - prior['started_unix']
        # Include any persisted running lower bound when clocks jump backwards.
        charge = max(charge, prior.get('elapsed_observed_seconds', 0.))
        h['jobs'].append({**prior, 'seconds': charge, 'exit_code': None,
            'reason': 'Unobserved interruption conservatively charged from saved start; not measured life runtime',
            'unobserved_interruption': True, 'charged_through_unix': now})
        h['worker_seconds'] += charge
    h['active_job'] = None
    h['active_jobs'] = []
    return h


class Science:
    """Only production adapter. It always invokes the original exact child command."""
    def __init__(self, root):
        assert root == ROOT
        assert sys.executable == PYTHON, 'Use the exact existing locked Python executable'
        assert sys.version_info[:3] == (3, 11, 15)
        assert all(os.environ.get(k) == v for k, v in THREAD_ENV.items())
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(root / 'src'))
        from locked_run import verify_lock, EXPECTED_PARAMS
        from audit_receipts import load, validate
        self.verify_lock, self.load, self.validate = verify_lock, load, validate
        self.params = EXPECTED_PARAMS
        self.root = root
        self.lock, digest = verify_lock()
        assert digest == LOCK_SHA
        assert self.lock['resource_caps'] == dict(workers=1, worker_hours=8., per_life_seconds=300.,
            peak_rss_bytes=800000000, results_bytes=200000000, active_session_hours=12.)
        assert (root / 'RECOVERY_AMENDMENT_20261001.md').is_file()

    def verify(self):
        assert self.verify_lock()[1] == LOCK_SHA

    def command(self, world, arm):
        return [PYTHON, str(self.root / 'src/assay.py'), '--arm', arm, '--world', str(world),
                '--bouts', str(self.lock['bouts']), '--kind', 'final', '--params',
                json.dumps(self.params, separators=(',', ':'))]

    def checked(self, path, world, arm):
        doc = self.load(path)
        self.validate(doc, expected_world=world, expected_arm=arm, expected_bouts=self.lock['bouts'],
            expected_sources=self.lock['source_hashes'], expected_params=self.lock['params'],
            expected_runtime=self.lock['expected_receipt_runtime'], lock_sha256=LOCK_SHA)
        caps = self.lock['resource_caps']
        assert doc['resource']['wall_seconds'] <= caps['per_life_seconds'], 'Receipt per-life cap exceeded'
        assert doc['resource']['peak_rss_bytes'] <= caps['peak_rss_bytes'], 'Receipt RSS cap exceeded'
        return doc


def authorization(approval_path, review_path):
    current = sha(__file__)
    approval = json.loads(Path(approval_path).read_text())
    review = json.loads(Path(review_path).read_text())
    required = dict(approved=True, effective_workers=2, effective_worker_hours_cap=12,
        lock_sha256=LOCK_SHA, original_started_unix=ORIGINAL_START, deadline_unix=DEADLINE,
        operational_launcher_sha256=current, python_executable=PYTHON)
    for key, value in required.items():
        assert approval.get(key) == value, f'Missing/mismatched explicit authorization: {key}'
    assert approval.get('user_approval_reference'), 'Approval needs verified user-message evidence'
    assert review.get('pass_all') is True and review.get('operational_launcher_sha256') == current
    assert review.get('independent_reviewer') and review.get('review_reference')
    return dict(approval=approval, independent_review=review)


class Live:
    def __init__(self, record, began):
        self.record, self.began = record, began
        self.process = None
        self.finished = None
        self.reason = None
        self.peak_rss = 0
        self.monitor = None

    def elapsed(self):
        return max(0., (self.finished if self.finished is not None else time.monotonic()) - self.began)

    def kill(self, reason):
        self.reason = self.reason or reason
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.kill()
            except ProcessLookupError:
                pass


class Coordinator:
    def __init__(self, root, science, guard, *, evidence, start=ORIGINAL_START,
                 deadline=DEADLINE, minimum=PRIOR_LOSS_CHARGE, worker_cap=WORKER_CAP,
                 poll_seconds=.05, stop_margin=.25):
        self.root, self.science, self.guard = root, science, guard
        self.dest = root / 'results/final'
        self.ledger_path = self.dest / 'RUN_LEDGER.json'
        self.status_path = self.dest / 'RUN_STATUS.json'
        self.lock = science.lock
        self.caps = self.lock['resource_caps']
        self.deadline, self.worker_cap = deadline, worker_cap
        self.poll_seconds, self.stop_margin = poll_seconds, stop_margin
        self.live = {}
        self.done = set()
        self.session = uuid.uuid4().hex
        self.operational_sha = sha(__file__)
        self.prior = json.loads(self.ledger_path.read_text())
        self.history = reconcile(self.prior, time.time(), start=start, minimum=minimum)
        self.history.update(effective_workers=2, effective_worker_hours_cap=12.,
            original_workers_cap_superseded=1, parallel_ledger_schema='RESPONSE-PARALLEL-LEDGER-v1',
            operational_launcher_sha256=self.operational_sha,
            original_elapsed_deadline_unix=deadline, parallel_authorization=evidence)
        self.expected = [(w, a) for w in self.lock['worlds'] for a in self.lock['arms']]
        assert all((r['world'], r['arm']) in self.expected and r.get('replay') is False
                   for r in active_records(self.prior)), 'Interrupted job outside original roster'
        assert self.caps['workers'] == 1 and self.caps['worker_hours'] == 8.

    def save(self):
        self.history['active_jobs'] = [dict(x.record, elapsed_observed_seconds=x.elapsed())
                                      for x in self.live.values()]
        atomic(self.ledger_path, self.history)

    def global_reason(self):
        if time.time() >= self.deadline:
            return 'original elapsed wall cap'
        if self.history['worker_seconds'] + sum(x.elapsed() for x in self.live.values()) > self.worker_cap:
            return 'amended cumulative worker-hours cap'
        if result_bytes(self.dest) > self.caps['results_bytes']:
            return 'results disk cap'
        return None

    def status(self, state, **extra):
        atomic(self.status_path, dict(state=state, completed=len(self.done), expected=len(self.expected),
            replayed=len(self.lock['replay_jobs']), effective_workers=2, effective_worker_hours_cap=12.,
            worker_seconds=self.history['worker_seconds'] + sum(x.elapsed() for x in self.live.values()),
            reserved_seconds=sum(x.record['reserved_seconds'] for x in self.live.values()),
            active_jobs=[dict(x.record, elapsed_current_seconds=x.elapsed()) for x in self.live.values()],
            lock_sha256=LOCK_SHA, operational_launcher_sha256=self.operational_sha,
            original_started_unix=self.history['started_unix'], original_elapsed_deadline_unix=self.deadline,
            **extra))

    def validate_existing(self):
        expected_paths = {f'{a}/{w}.json.gz' for w, a in self.expected}
        expected_paths |= {f'replays/{a}/{w}.json.gz' for w, a in self.lock['replay_jobs']}
        actual = {str(p.relative_to(self.dest)) for p in self.dest.rglob('*.json.gz')}
        assert actual <= expected_paths, 'Unexpected extra receipt'
        for world, arm in self.expected:
            path = self.dest / arm / f'{world}.json.gz'
            if path.exists():
                self.science.checked(path, world, arm)
                self.done.add((world, arm))
        for world, arm in self.lock['replay_jobs']:
            first = self.science.checked(self.dest / arm / f'{world}.json.gz', world, arm)
            again = self.science.checked(self.dest / 'replays' / arm / f'{world}.json.gz', world, arm)
            assert first['resource']['process_id'] != again['resource']['process_id']
            first = {k: v for k, v in first.items() if k != 'resource'}
            again = {k: v for k, v in again.items() if k != 'resource'}
            assert first == again, 'Exact replay mismatch'
        audit = json.loads((self.dest / 'REPLAY_AUDIT.json').read_text())
        assert audit['pass_all'] is True
        assert audit['jobs'] == self.lock['replay_jobs'] and audit['excluded_fields'] == ['resource']

    def queue_worlds(self):
        # Contiguous-prefix count preserves prepare_recovery_checkpoint.py's contract.
        for index, world in enumerate(self.lock['worlds'], 1):
            if not all((world, arm) in self.done for arm in self.lock['arms']):
                break
            path = self.dest / 'publication_queue' / f'{world}.json'
            count = index * len(self.lock['arms'])
            if path.exists():
                prior = json.loads(path.read_text())
                assert prior['world'] == world and prior['completed'] == count and prior['lock_sha256'] == LOCK_SHA
            else:
                atomic(path, dict(world=world, completed=count, lock_sha256=LOCK_SHA,
                    source='Recovery launcher completed and validated full world; publication is independent'))

    def watch(self, live):
        """Independent cap monitor remains active during source/receipt validation."""
        try:
            while live.process.poll() is None:
                live.peak_rss = max(live.peak_rss, rss(live.process.pid))
                if live.elapsed() >= self.caps['per_life_seconds'] - self.stop_margin:
                    live.kill('per-life time cap safety stop')
                elif time.time() >= self.deadline - self.stop_margin:
                    live.kill('original elapsed wall cap safety stop')
                elif live.peak_rss > self.caps['peak_rss_bytes']:
                    live.kill('worker RSS cap')
                elif result_bytes(self.dest) > self.caps['results_bytes']:
                    live.kill('results disk cap')
                time.sleep(self.poll_seconds)
        except BaseException as exc:
            live.kill('cap monitor failure: ' + repr(exc))
        finally:
            code = live.process.wait()
            if code != 0:
                live.reason = live.reason or f'nonzero child exit: {code}'
            live.finished = time.monotonic()

    def check_child_failures(self):
        # Poll the processes directly as well as the monitor flags. A monitor may
        # not yet have been scheduled after an exit, so its flag alone is unsafe.
        for live in self.live.values():
            if live.process is not None:
                code = live.process.poll()
                if code is not None and code != 0:
                    live.reason = live.reason or f'nonzero child exit: {code}'
            if live.reason:
                raise RuntimeError(live.reason)

    def drain_completed(self):
        # Receipt validation may be slow. Recompute readiness AFTER every settle
        # instead of admitting work from a stale snapshot of completed children.
        settled = 0
        while True:
            self.check_child_failures()
            ready = next((attempt for attempt, live in self.live.items()
                          if live.process is not None and live.process.poll() is not None), None)
            if ready is None:
                return settled
            self.settle(ready)
            settled += 1

    def can_reserve(self):
        return (self.history['worker_seconds'] + sum(x.record['reserved_seconds'] for x in self.live.values())
                + self.caps['per_life_seconds'] <= self.worker_cap)

    def launch(self, world, arm):
        self.drain_completed()
        assert len(self.live) < WORKERS
        assert self.can_reserve(), 'Insufficient remaining budget for another complete per-life reservation'
        assert (world, arm) not in self.done
        assert all((x.record['world'], x.record['arm']) != (world, arm) for x in self.live.values())
        self.check_child_failures()
        self.science.verify()
        self.drain_completed()  # Source verification can span another child's exit.
        assert not (self.dest / arm / f'{world}.json.gz').exists(), 'Never overwrite a receipt'
        reason = self.global_reason()
        if reason:
            raise RuntimeError(reason)
        attempt = f'{world}-{arm}-attempt-{time.time_ns()}-{uuid.uuid4().hex[:8]}'
        record = dict(attempt_id=attempt, world=world, arm=arm, replay=False,
            started_unix=time.time(), reserved_seconds=self.caps['per_life_seconds'],
            stdout=f'operations/{attempt}.stdout.txt', stderr=f'operations/{attempt}.stderr.txt',
            operational_launcher_sha256=self.operational_sha, effective_workers=2)
        live = Live(record, time.monotonic())
        self.live[attempt] = live
        self.save()  # Persist each reservation BEFORE spawn, even if spawn later fails.
        with (self.dest / record['stdout']).open('x') as out, (self.dest / record['stderr']).open('x') as err:
            # Reservation fsync/log setup can also span another child's exit.
            # Drain success receipts (including invalid ones) and surface failures
            # immediately before constructing the exact command and spawning.
            self.drain_completed()
            reason = self.global_reason()
            if reason:
                raise RuntimeError(reason)
            if live.elapsed() >= self.caps['per_life_seconds'] - self.stop_margin:
                raise RuntimeError('Unspawned reservation exceeded per-life safety window')
            self.check_child_failures()
            live.process = subprocess.Popen(self.science.command(world, arm), stdout=out, stderr=err,
                env=dict(os.environ, **THREAD_ENV), pass_fds=(self.guard.fileno(),))
        record['process_id'] = live.process.pid
        live.monitor = threading.Thread(target=self.watch, args=(live,), daemon=True)
        live.monitor.start()
        self.save()

    def settle(self, attempt, *, forced_reason=None):
        live = self.live[attempt]
        if live.monitor:
            live.monitor.join()
        elif live.process:
            live.process.wait()
        if live.finished is None:
            live.finished = time.monotonic()
        elapsed = live.elapsed()
        reason = forced_reason or live.reason or self.global_reason()
        if elapsed > self.caps['per_life_seconds']:
            reason = reason or 'post-exit per-life time cap'
        if live.peak_rss > self.caps['peak_rss_bytes']:
            reason = reason or 'post-exit worker RSS cap'
        code = live.process.returncode if live.process else None
        if code != 0:
            reason = reason or 'nonzero child exit or spawn failure'
        if not reason:
            try:
                self.science.checked(self.dest / live.record['arm'] / f"{live.record['world']}.json.gz",
                                     live.record['world'], live.record['arm'])
                reason = self.global_reason()  # Validation itself can cross global bounds.
            except BaseException as exc:
                reason = 'post-exit receipt validation failure: ' + repr(exc)
        entry = dict(live.record, seconds=elapsed, exit_code=code, reason=reason,
            finished_unix=time.time(), observed_peak_rss_bytes=live.peak_rss,
            receipt_validated=not bool(reason), parallel_recovery=True)
        self.history['jobs'].append(entry)
        self.history['worker_seconds'] += elapsed
        del self.live[attempt]
        if reason:
            self.history['terminal_failure'] = dict(reason=reason, attempt_id=attempt)
        self.save()
        if reason:
            raise RuntimeError(json.dumps(entry))
        self.done.add((live.record['world'], live.record['arm']))
        print(json.dumps(dict(completed=len(self.done), world=live.record['world'],
            arm=live.record['arm'], seconds=elapsed, worker_hours=self.history['worker_seconds'] / 3600)), flush=True)

    def stop_all(self, reason):
        # Kill all first; never let the second child run during a first-child wait.
        for live in self.live.values():
            live.kill(reason)
        failures = []
        for attempt in list(self.live):
            try:
                self.settle(attempt, forced_reason=reason)
            except BaseException as exc:
                failures.append(repr(exc))
        return failures

    def run(self):
        (self.dest / 'operations').mkdir(exist_ok=True)
        write_once(self.dest / 'operations' / f'PRE_PARALLEL_LEDGER-{self.session}.json', self.prior)
        write_once(self.dest / 'operations' / f'PARALLEL_AUTHORIZATION-{self.session}.json',
                   self.history['parallel_authorization'])
        self.save()  # Original entries/start/lost-work charge plus every stale child charge.
        try:
            self.validate_existing()
            self.queue_worlds()
            pending = [(w, a) for w, a in self.expected if (w, a) not in self.done]
            last_status = 0.
            while pending or self.live:
                reason = self.global_reason()
                if reason:
                    raise RuntimeError(reason)
                # Repeatedly drain: validation can make new exits observable.
                settled = self.drain_completed()
                self.queue_worlds()
                while pending and len(self.live) < WORKERS and self.can_reserve():
                    self.launch(*pending.pop(0))
                if pending and not self.live:
                    self.status('budget_stop', reason='Remaining cumulative budget cannot reserve a complete life')
                    return False
                if time.monotonic() - last_status >= 2. or settled:
                    self.status('running')
                    self.save()
                    last_status = time.monotonic()
                time.sleep(self.poll_seconds)
            self.science.verify()
            self.validate_existing()
            reason = self.global_reason()
            if reason:
                raise RuntimeError(reason)
            assert len(self.done) == len(self.expected)
            self.queue_worlds()
            self.status('complete_pending_final_audit')
            return True
        except BaseException as exc:
            reason = 'Coordinator stopped: ' + repr(exc)
            cleanup = self.stop_all(reason)
            self.history['terminal_failure'] = self.history.get('terminal_failure') or dict(reason=reason)
            self.save()
            self.status('technical_stop', reason=reason, cleanup_errors=cleanup)
            raise


def preflight(approval_path=None, review_path=None):
    """Read-only: do not create/replace evidence, queues, receipts, or ledger."""
    science = Science(ROOT)
    blockers = []
    try:
        guard = acquire_guard(ROOT)
    except (RuntimeError, FileNotFoundError) as exc:
        blockers.append(str(exc))
    else:
        guard.close()
    processes = matching_processes(ROOT)
    if processes:
        blockers.append('Unresolved scientific child or supervisor process remains')
    evidence = None
    if approval_path and review_path:
        try:
            evidence = authorization(approval_path, review_path)
        except (AssertionError, OSError, ValueError) as exc:
            blockers.append(str(exc))
    else:
        blockers.append('Explicit two-worker authorization and independent exact-source review are required')
    history = json.loads((ROOT / 'results/final/RUN_LEDGER.json').read_text())
    migrated = reconcile(history, time.time())
    if time.time() >= DEADLINE:
        blockers.append('Original elapsed deadline has expired')
    if migrated['worker_seconds'] + science.lock['resource_caps']['per_life_seconds'] > WORKER_CAP:
        blockers.append('Insufficient cumulative budget for a complete life reservation')
    return dict(launch_ready=not blockers, blockers=blockers, matching_processes=processes,
        operational_launcher_sha256=sha(__file__), lock_sha256=LOCK_SHA,
        effective_workers_requested=2, deadline_unix=DEADLINE,
        persisted_worker_seconds=history['worker_seconds'],
        conservative_worker_seconds_if_resumed_now=migrated['worker_seconds'],
        unobserved_active_jobs=len(active_records(history)), authorization_checked=evidence is not None)


def main():
    if not __debug__:
        raise RuntimeError('Optimized Python disables mandatory checks; launch refused')
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['preflight', 'run'])
    parser.add_argument('--approval')
    parser.add_argument('--review')
    args = parser.parse_args()
    if args.action == 'preflight':
        result = preflight(args.approval, args.review)
        print(json.dumps(result, indent=2))
        return 0 if result['launch_ready'] else 2
    assert args.approval and args.review, 'Launch authorization and independent review paths are required'
    evidence = authorization(args.approval, args.review)
    guard = acquire_guard(ROOT, create=True)
    try:
        assert not matching_processes(ROOT), 'Unresolved active old supervisor or scientific child'
        science = Science(ROOT)
        coordinator = Coordinator(ROOT, science, guard, evidence=evidence)
        def stop(signum, _frame):
            raise InterruptedError(f'Supervisor signal {signum}; stopping both children')
        signal.signal(signal.SIGTERM, stop)
        return 0 if coordinator.run() else 3
    finally:
        guard.close()


if __name__ == '__main__':
    raise SystemExit(main())
