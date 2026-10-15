"""Two-worker bounded technical-only dispatcher. No automatic retries."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from protocol import ROOT, D, atomic, identity


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('cycle2', 'cycle3'), required=True)
    a = p.parse_args()
    out = ROOT / 'cycles' / a.stage
    launch = out / 'LAUNCH.json'
    if launch.exists():
        raise RuntimeError('stage already dispatched; no automatic repeats')
    lock = json.loads((ROOT / 'SOURCE_LOCK.json').read_text())
    assert identity() == lock['source']
    jobs = [dict(world=w, model=m) for w in D['technical'][a.stage]['worlds'] for m in ('KEEP', 'G62')]
    awake = json.loads((ROOT / 'ops/WORK_AWAKE.json').read_text())
    session_start = awake['started_epoch']
    caffeinate = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-s', '-w', str(os.getpid())],
                                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    running, completed = [], []
    stopped = False

    def interrupt(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)

    def status(state, **extra):
        atomic(out / 'STATUS.json', dict(status=state, heartbeat=time.time(), supervisor_pid=os.getpid(),
            workers=[dict(**j, pid=proc.pid) for j, proc, log in running], committed_jobs=len(completed),
            committed_records=sum(x['records'] for x in completed),
            committed_related_histories=7 * len(completed), planned_jobs=len(jobs), **extra))

    try:
        atomic(launch, dict(stage=a.stage, supervisor_pid=os.getpid(), caffeinate_pid=caffeinate.pid,
            python=sys.executable, workers=2, source=lock['source'], jobs=jobs,
            detached=True, session_started_epoch=session_start, heartbeat=time.time(), technical_only=True))
        pending = list(jobs)
        while pending or running:
            if stopped:
                raise RuntimeError('technical dispatcher received stop signal')
            elapsed = time.time() - session_start
            if elapsed > 10 * 3600:
                raise RuntimeError('hard ten-hour session boundary')
            admit = elapsed < 8 * 3600
            # Conservative cost bound from the completed short qualification.
            prior = ROOT / 'cycles/cycle2/STATUS.json'
            budget = 3600.
            if a.stage == 'cycle3' and prior.exists():
                s = json.loads(prior.read_text())
                per = s.get('max_job_wall', 100.)
                budget = max(900., per * 40.) + 900.
            admit = admit and 10 * 3600 - elapsed > budget
            while pending and len(running) < 2 and admit:
                j = pending.pop(0)
                logpath = out / 'logs' / f"{j['world']}_{j['model']}.log"
                logpath.parent.mkdir(parents=True, exist_ok=True)
                log = logpath.open('w')
                proc = subprocess.Popen([sys.executable, '-u', str(ROOT / 'runner.py'), '--stage', a.stage,
                    '--world', str(j['world']), '--model', j['model']], cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
                running.append((j, proc, log))
            for j, proc, log in list(running):
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                running.remove((j, proc, log))
                if code != 0:
                    raise RuntimeError(f'worker {j} exited {code}; see its preserved log')
                target = out / 'receipts' / f"{j['world']}_{j['model']}.json.audit.json"
                certificate = json.loads(target.read_text())
                assert certificate['accepted']
                completed.append(dict(j, records=certificate['records']))
                atomic(out / 'COMMIT_INDEX.json', dict(jobs=completed, source=lock['source']))
            if pending and not running and not admit:
                status('PAUSED_SESSION_LIMIT', next_jobs=pending)
                return
            status('RUNNING_TECHNICAL' if pending or running else 'EXECUTION_ACCEPTED')
            if pending or running:
                time.sleep(2)
        import gzip
        walls = []
        for j in completed:
            with gzip.open(out / 'receipts' / f"{j['world']}_{j['model']}.json.gz", 'rt') as f:
                walls.append(json.load(f)['wall'])
        assert identity() == lock['source']
        status('EXECUTION_ACCEPTED', max_job_wall=max(walls), source_unchanged=True)
    except BaseException:
        error = traceback.format_exc()
        atomic(out / 'FAILURE.json', dict(error=error, time=time.time(), committed_jobs=len(completed),
            committed_records=sum(x['records'] for x in completed), source=lock['source']))
        status('FAILED', error=error)
        print(error, flush=True)
        raise
    finally:
        for j, proc, log in running:
            if proc.poll() is None:
                proc.terminate()
        for j, proc, log in running:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.close()
        if caffeinate.poll() is None:
            caffeinate.terminate()
        caffeinate.wait()


if __name__ == '__main__':
    main()
