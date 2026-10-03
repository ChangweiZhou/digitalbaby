"""Detached, two-worker supervisor; no automatic retries and no partial analysis."""
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from compact_bridge import ROOT, ARMS
from compact_fixture import SCIENCE_WORLDS
from compact_integrity import verify
from compact_audit import read, audit_receipt
from compact_storage import atomic_json


def now(): return datetime.now(timezone.utc).isoformat()


def git_upload(message):
    repo = ROOT.parent
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=repo, text=True).strip() != 'codex/r-center-compact-ablation-20261003':
        raise RuntimeError('publication branch changed')
    subprocess.run(['git', 'add', '--', ROOT.name], cwd=repo, check=True)
    changed = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=repo).returncode
    if changed:
        subprocess.run(['git', 'commit', '-m', message], cwd=repo, check=True)
    subprocess.run(['git', 'push', 'origin', 'HEAD:refs/heads/codex/r-center-compact-ablation-20261003'], cwd=repo, check=True, timeout=300)
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()


def main():
    import fcntl
    mutex = (ROOT / 'results/supervisor.lock').open('a')
    fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.chdir(ROOT); identity = verify()
    qual = json.loads((ROOT / 'results/QUALIFICATION.json').read_text())
    if qual['verdict'] != 'PASS' or qual['identity'] != identity: raise ValueError('unqualified final source')
    spec = json.loads((ROOT / 'SPEC.json').read_text())
    results = ROOT / 'results'; folder = results / 'science'; active = {}
    jobs = [(w, a) for w in SCIENCE_WORLDS for a in ARMS]
    completed = []
    for job in list(jobs):
        path = folder / 'receipts' / f'{job[0]}_{job[1]}.json.gz'
        if path.exists():
            audit_receipt(read(path), identity); completed.append(job); jobs.remove(job)
        elif (folder / 'checkpoints' / f'{job[0]}_{job[1]}.npz').exists():
            raise ValueError('partial science cursor requires explicit resume authorization')
    status = {'state': 'RUNNING', 'stage': 'execution', 'pid': os.getpid(), 'identity': identity,
              'started_utc': now(), 'completed_world_arm_jobs': len(completed), 'total_world_arm_jobs': 192,
              'total_lives': 768, 'completed_lives': len(completed) * 4, 'caffeinate_attached': False}
    caff = None
    if platform.system() == 'Darwin' and shutil.which('caffeinate'):
        caff = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())]); status['caffeinate_attached'] = True
        status['caffeinate_pid'] = caff.pid
    def signal_stop(signum, frame): raise RuntimeError(f'supervisor interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, signal_stop); signal.signal(signal.SIGINT, signal_stop)
    worker_s = 0.
    try:
        while jobs or active:
            while jobs and len(active) < spec['workers']:
                world, arm = jobs.pop(0); logpath = folder / 'logs' / f'{world}_{arm}.log'
                logpath.parent.mkdir(parents=True, exist_ok=True); log = logpath.open('w')
                env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           NUMBA_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
                proc = subprocess.Popen([sys.executable, '-u', str(ROOT / 'compact_worker.py'), '--world', str(world), '--arm', arm],
                                        stdout=log, stderr=subprocess.STDOUT, env=env)
                active[proc.pid] = {'process': proc, 'log': log, 'job': (world, arm), 'started': time.monotonic(),
                                    'logpath': str(logpath.relative_to(ROOT))}
            for pid, entry in list(active.items()):
                proc = entry['process']; code = proc.poll()
                if time.monotonic() - entry['started'] > spec['max_job_seconds']:
                    raise RuntimeError(f'worker deadline: {entry["job"]}')
                if code is None:
                    rss = subprocess.run(['ps', '-p', str(pid), '-o', 'rss='], capture_output=True, text=True)
                    if rss.returncode == 0 and rss.stdout.strip().isdigit() and int(rss.stdout.strip()) * 1024 > spec['max_worker_rss_bytes']:
                        raise RuntimeError(f'worker RSS cap: {entry["job"]}')
                    continue
                entry['log'].close()
                if code != 0:
                    raise RuntimeError(f'worker failed {entry["job"]}, exit {code}; {entry["logpath"]}')
                world, arm = entry['job']; doc = read(folder / 'receipts' / f'{world}_{arm}.json.gz')
                audit_receipt(doc, identity); worker_s += doc['worker_s']; completed.append(entry['job']); del active[pid]
                print(json.dumps({'committed_world_arm_jobs': len(completed), 'total': 192, 'world': world, 'arm': arm}), flush=True)
            bytes_used = sum(p.stat().st_size for p in results.rglob('*') if p.is_file())
            charged = worker_s + sum(time.monotonic() - e['started'] for e in active.values())
            if charged > spec['worker_budget_seconds'] or bytes_used > spec['disk_cap_bytes']:
                raise RuntimeError('cumulative worker or storage cap exceeded')
            status.update(heartbeat_utc=now(), completed_world_arm_jobs=len(completed), completed_lives=len(completed) * 4,
                          active_jobs=[{'pid': p, 'world': e['job'][0], 'arm': e['job'][1]} for p,e in active.items()],
                          worker_s_completed=worker_s, bytes_used=bytes_used)
            atomic_json(results / 'STATUS.json', status)
            time.sleep(5)
        verify(); status.update(stage='analysis'); atomic_json(results / 'STATUS.json', status)
        for script in ('compact_analysis.py', 'final_audit.py'):
            with (results / (script + '.log')).open('w') as log:
                subprocess.run([sys.executable, '-u', str(ROOT / script)], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        status.update(state='COMPLETE', stage='accepted', completed_utc=now(), github_upload='PENDING')
        atomic_json(results / 'STATUS.json', status)
        try:
            commit = git_upload('Complete compact R_center physical deletion study and audited results')
            atomic_json(results / 'PUBLICATION.json', {'state': 'UPLOADED', 'commit': commit,
                        'branch': 'codex/r-center-compact-ablation-20261003', 'utc': now()})
            status.update(github_upload='UPLOADED', published_commit=commit)
        except Exception:
            status.update(github_upload='FAILED', publication_error=traceback.format_exc())
            atomic_json(results / 'PUBLICATION.json', {'state': 'FAILED', 'error': status['publication_error']})
    except Exception:
        status.update(state='FAILED', error=traceback.format_exc(), stopped_utc=now())
        atomic_json(results / 'FAILURE.json', status)
        print(status['error'], file=sys.stderr, flush=True)
        for entry in active.values():
            if entry['process'].poll() is None: entry['process'].terminate()
        for entry in active.values():
            try: entry['process'].wait(timeout=30)
            except subprocess.TimeoutExpired: entry['process'].kill(); entry['process'].wait()
            entry['log'].close()
        # Preserve and upload the actual failure, without retrying science or packaging a partial result as complete.
        try: git_upload('Preserve compact R_center execution failure and committed receipts')
        except Exception: pass
        return 1
    finally:
        if caff is not None:
            caff.terminate(); caff.wait(); status['caffeinate_released'] = True
        atomic_json(results / 'STATUS.json', status)
    return 0


if __name__ == '__main__': sys.exit(main())
