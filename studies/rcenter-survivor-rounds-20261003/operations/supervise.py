# SPDX-License-Identifier: GPL-3.0-or-later
"""Same-lock production admission and bounded, pre-reserved worker execution.

The active tool host must call plan -> reserve -> private readback/ACK -> run.
This process never calls remote tools or assumes it survives executor loss.
"""
import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source/runtime'))
from integrity import hashes, digest_map, require_runtime, sha
from durable import replace
from recovery import (CAP, RSS, TOTAL, WALL, RESERVE, check, read, exclusive_lock,
                      reconcile_and_plan, prepare_attempt, start_reserved_attempt,
                      finish_attempt, require_reservation_ack, load_ledger,
                      output_size as recovery_output_size, limits, process_identity,
                      validate_completed)


def peak(pid):
    try:
        return max(int(line.split()[1]) * 1024 for line in Path(f'/proc/{pid}/status').read_text().splitlines()
                   if line.startswith(('VmRSS:', 'VmHWM:')))
    except (FileNotFoundError, ValueError):
        return 0


def memory_available():
    return next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def output_size():
    return recovery_output_size(ROOT)


def host_wall():
    # This counter includes supervision, checkpoint, restore, readback and tools.
    # It is owned exclusively by the host keeper, never incremented here.
    state = read(ROOT / 'operations/HOST_WALL.json')
    value = state['effective_s']
    check(math.isfinite(value) and value >= 0, 'invalid active host-wall counter')
    check(state.get('active') is True, 'active tool-host accounting session required')
    check(os.environ.get('SURVIVOR_HOST_TOKEN') and os.environ['SURVIVOR_HOST_TOKEN'] == state.get('host_token'), 'tool-host session token mismatch')
    check(state.get('boot_id') == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'host-wall boot identity mismatch')
    age = time.monotonic() - state['updated_monotonic_s']
    check(0 <= age <= 5.0, 'host-wall keeper heartbeat stale')
    path = ROOT / 'operations/HOST_COORDINATOR.lock'
    check(path.exists(), 'host coordinator lock missing')
    with path.open('r') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(handle, fcntl.LOCK_UN)
            raise RuntimeError('active host coordinator lock is not held')
    return float(value)


def validate_limits(dt, high, ledger):
    check(math.isfinite(dt) and 0 <= dt <= CAP and 0 <= high <= RSS, 'per-job time/RSS cap')
    # The ledger already includes this running job's full cap reservation.
    limits(ROOT, ledger, active_wall_s=host_wall())


def governance(mode, source):
    if mode == 'cycle3':
        review = read(ROOT / 'audits/CYCLE3_SOURCE_REVIEW.json')
        check(review.get('accepted') is True, 'independent cycle3 source admission missing')
        check(review.get('source_hashes') == source, 'incomplete/stale cycle3 closure')
        check(sha(ROOT / 'protocol' / review['design_file']) == review['design_sha256'], 'stale cycle3 design')
        check(sha(ROOT / 'audits/CYCLE3_SOURCE_REVIEW.md') == review['report_sha256'], 'cycle3 source report drift')
    elif mode == 'science':
        lock = read(ROOT / 'protocol/OFFICIAL_LOCK.json')
        launch = read(ROOT / 'audits/LAUNCH_ACCEPTED.json')
        check(lock['hashes'] == source, 'official source drift')
        check(launch.get('accepted') is True and launch['lock_sha256'] == sha(ROOT / 'protocol/OFFICIAL_LOCK.json'), 'launch acceptance missing')
        for rel, digest in lock['acceptances'].items():
            check(sha(ROOT / rel) == digest, 'prior acceptance or evidence drift')
        state = read(ROOT / 'operations/PERSISTENCE_STATE.json')
        check(state.get('prelaunch_verified') is True and state.get('prelaunch_source_digest') == digest_map(source), 'prelaunch persistence missing/stale')
    else:
        raise RuntimeError('unsupported mode')


def key_for(mode, job, world):
    if mode == 'science':
        if job == 'analysis':
            check(world is None, 'analysis is not a native world')
            return 'analysis/final'
        check(world in range(310001, 310065), 'invalid official world')
        return f'science/{world}'
    check(mode == 'cycle3' and job in ('operations', 'pilot', 'replay'), 'invalid qualification job')
    check(world is None or world == 310200, 'invalid qualification world')
    return f'cycle3/{job}'


def admission(mode, job, world, ledger=None, *, lock=None):
    """Public admission path always reopens/validates local completed receipts."""
    if lock is None:
        with exclusive_lock(ROOT) as acquired:
            return admission(mode, job, world, ledger, lock=acquired)
    runtime = require_runtime()
    source = hashes()
    governance(mode, source)
    plan = reconcile_and_plan(ROOT, source, runtime, mode, lock=lock, active_wall_s=host_wall())
    if ledger is not None:
        check(ledger == plan['ledger'], 'caller ledger is stale')
    key = key_for(mode, job, world)
    check(plan['next_key'] == key, 'missing-only admission blocked: ' + repr(plan['blocked']))
    check(memory_available() >= int(2.5 * 1024**3) + RSS, 'insufficient fresh memory reserve')
    validate_limits(0.0, 0, plan['ledger'])
    return key, runtime, source


WRAPPER = """import ctypes,os,signal,time,runpy,json
from pathlib import Path
expected=int(os.environ['SUPERVISOR_PID'])
if os.getppid()!=expected:raise SystemExit('parent lost before startup')
if ctypes.CDLL(None).prctl(1,signal.SIGKILL)!=0:raise SystemExit('parent-death guard failed')
if os.getppid()!=expected:raise SystemExit('parent lost during startup')
if 'SUPERVISOR_LOCK_FD' in os.environ:os.fstat(int(os.environ['SUPERVISOR_LOCK_FD']))
p=Path(os.environ['ADMISSION'])
wanted={'child_pid':os.getpid(),'source_digest':os.environ['SURVIVOR_SOURCE_DIGEST'],
        'reservation_id':os.environ['SURVIVOR_RESERVATION_ID'],'key':os.environ['SURVIVOR_JOB_KEY'],
        'host_token':os.environ['SURVIVOR_HOST_TOKEN']}
if not all(wanted.values()):raise SystemExit('empty startup admission identity')
while True:
 if os.getppid()!=expected:raise SystemExit('parent lost waiting for admission')
 try:active=json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):active={}
 if isinstance(active,dict) and all(active.get(k)==v for k,v in wanted.items()):break
 time.sleep(.02)
if os.getppid()!=expected:raise SystemExit('parent lost before admitted entry')
runpy.run_path(os.environ['ENTRY'],run_name='__main__')
"""


def run_reserved(mode, job, world, reservation_id, *, lock):
    check(reservation_id, 'durable pre-job reservation required; reserve and read back before run')
    runtime = require_runtime()
    source = hashes()
    governance(mode, source)
    key = key_for(mode, job, world)
    plan = reconcile_and_plan(ROOT, source, runtime, mode, lock=lock, keep_reservation_id=reservation_id,
                              active_wall_s=host_wall())
    check(set(plan['blocked']) <= {'reserved_attempt'}, 'pending barrier or recovery blocks run')
    matches = [a for a in plan['ledger']['attempts'] if a.get('reservation_id') == reservation_id]
    check(len(matches) == 1 and matches[0]['status'] == 'reserved' and matches[0]['key'] == key, 'reservation identity mismatch')
    reserved = matches[0]
    check(reserved['host_token'] == os.environ.get('SURVIVOR_HOST_TOKEN'), 'reservation belongs to a lost host session')
    require_reservation_ack(ROOT, reserved)
    check(memory_available() >= int(2.5 * 1024**3) + RSS, 'insufficient fresh memory reserve')
    check(host_wall() + CAP <= WALL, 'insufficient live active-wall reserve')
    validate_limits(0.0, 0, plan['ledger'])
    dest = ROOT / reserved['receipt']
    check(not dest.exists(), 'attempt output must remain fresh')
    entry_script = ROOT / ('tests/cycle3_recovery.py' if key == 'cycle3/operations' else
                           'source/runtime/analysis_entry.py' if key == 'analysis/final' else 'source/runtime/world_entry.py')
    env = dict(os.environ, SURVIVOR_SUPERVISED='1', SURVIVOR_MODE=mode,
               R3_OBS_MODE='science' if mode == 'science' else 'technical',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
    env.update(ADMISSION=str(ROOT / 'operations/ACTIVE_JOB.json'), ENTRY=str(entry_script),
               SUPERVISOR_PID=str(os.getpid()), SUPERVISOR_LOCK_FD=str(lock.handle.fileno()),
               SURVIVOR_RESERVATION_ID=reservation_id, SURVIVOR_JOB_KEY=key,
               SURVIVOR_HOST_TOKEN=reserved['host_token'], SURVIVOR_SOURCE_DIGEST=reserved['source_digest'],
               SURVIVOR_WORLD=str(world if mode == 'science' else 310200), SURVIVOR_DEST=str(dest), SURVIVOR_JOB=job)
    begin = time.monotonic()
    high = 0
    failure = None
    proc = None
    entry = None
    ledger = plan['ledger']
    try:
        with (ROOT / reserved['log']).open('x') as log:
            proc = subprocess.Popen([sys.executable, '-c', WRAPPER], stdout=log, stderr=subprocess.STDOUT,
                                    env=env, pass_fds=(lock.handle.fileno(),))
            child = process_identity(proc.pid)
            check(child is not None, 'child vanished before recorded admission')
            ledger, entry = start_reserved_attempt(ROOT, reservation_id, source, runtime, mode, lock=lock,
                                                    child_identity=child, host_token=os.environ.get('SURVIVOR_HOST_TOKEN'), active_wall_s=host_wall())
            replace(ROOT / 'operations/ACTIVE_JOB.json', {'child_pid': proc.pid, 'source_digest': entry['source_digest'],
                                                          'runtime': runtime, 'mode': mode, 'key': key,
                                                          'attempt': entry['attempt'], 'reservation_id': reservation_id, 'host_token': reserved['host_token']})
            while proc.poll() is None:
                high = max(high, peak(proc.pid))
                validate_limits(time.monotonic() - begin, high, ledger)
                time.sleep(0.25)
        check(proc.returncode == 0, 'worker failed; preserve exact attempt artifacts')
        result = validate_completed(ROOT, entry, source, runtime)
        high = max(high, int(result['manifest']['resources']['peak_rss_bytes']))
        validate_limits(time.monotonic() - begin, high, ledger)
        check(hashes() == source, 'source changed during worker')
    except BaseException as exc:
        failure = repr(exc)
    finally:
        if proc and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        duration = time.monotonic() - begin
        try:
            validate_limits(duration, high, ledger)
        except Exception as exc:
            failure = failure or repr(exc)
        if entry is not None:
            entry = finish_attempt(ROOT, reservation_id, source, runtime, duration, high, failure,
                                   None if proc is None else proc.returncode, lock=lock)
    if entry is None:
        raise RuntimeError('worker never admitted; cap reservation retained: ' + str(failure))
    return entry


def main(mode, job='operations', world=None, action='run', reservation_id=None):
    check(sys.flags.optimize == 0, 'optimized Python prohibited')
    with exclusive_lock(ROOT) as lock:
        if action == 'plan':
            return reconcile_and_plan(ROOT, hashes(), require_runtime(), mode, lock=lock, active_wall_s=host_wall())
        if action == 'reserve':
            key, runtime, source = admission(mode, job, world, lock=lock)
            return prepare_attempt(ROOT, source, runtime, mode, key, lock=lock, host_token=os.environ.get('SURVIVOR_HOST_TOKEN'), active_wall_s=host_wall())
        check(action == 'run', 'invalid supervisor action')
        return run_reserved(mode, job, world, reservation_id, lock=lock)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['cycle3', 'science'])
    p.add_argument('--job', choices=['operations', 'pilot', 'replay', 'analysis'], default='operations')
    p.add_argument('--world', type=int)
    p.add_argument('--action', choices=['plan', 'reserve', 'run'], default='run')
    p.add_argument('--reservation-id')
    a = p.parse_args()
    result = main(a.mode, a.job, a.world, a.action, a.reservation_id)
    print(json.dumps(result, separators=(',', ':')), flush=True)
    if a.action == 'run' and result['status'] != 'completed':
        raise SystemExit(1)
