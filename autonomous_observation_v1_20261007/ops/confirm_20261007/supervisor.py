"""Single detached dispatcher, no retries; completes analysis and final package."""
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback
import zipfile
from ops_common import OPS, ROOT, RESULTS, WORLDS, atomic, counts, read, sha, verify_sources


def status(stage, **extra):
    value = dict(stage=stage, supervisor_pid=os.getpid(), workers=1,
                 heartbeat_unix=time.time(), **counts(), **extra)
    atomic(RESULTS / 'STATUS.json', value)
    return value


def package(summary):
    # Fixed source, qualification, formal receipts/audits/logs; no parent data duplication.
    paths = [ROOT / n for n in read(ROOT / 'SOURCE_LOCK.json')['files']]
    paths += [ROOT / n for n in ('SOURCE_LOCK.json', 'QUALIFICATION.json', 'FINAL_AUDIT.json')]
    paths += [p for p in OPS.iterdir() if p.is_file() and p.name not in ('supervisor.log','supervisor.lock')]
    paths += [p for p in RESULTS.rglob('*') if p.is_file() and p.name not in ('RESULT_BUNDLE.zip','PACKAGE.json','STATUS.json')]
    paths = sorted(set(paths))
    def arcname(path):
        if path.is_relative_to(RESULTS):
            return 'results/confirm/' + str(path.relative_to(RESULTS))
        return str(path.relative_to(ROOT))
    manifest = {arcname(p): sha(p) for p in paths}
    atomic(RESULTS / 'BUNDLE_MANIFEST.json', manifest, exclusive=True)
    bundle = RESULTS / 'RESULT_BUNDLE.zip'
    temp = bundle.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in paths:
            z.write(p, arcname(p))
        z.write(RESULTS / 'BUNDLE_MANIFEST.json', 'results/confirm/BUNDLE_MANIFEST.json')
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:
            raise ValueError('result ZIP failed integrity read')
    os.link(temp, bundle)
    temp.unlink()
    atomic(RESULTS / 'PACKAGE.json', dict(status='COMPLETE', path=str(bundle),
           sha256=sha(bundle), bytes=bundle.stat().st_size, payload_files=len(paths)+1,
           verdict=summary['verdict'], native_parent_external_dependency=str(ROOT.parent/'r_center_core_v1'),
           partial_data_packaged=False, GitHub_upload_authorized=False), exclusive=True)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    lockfile = (OPS / 'supervisor.lock').open('a')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (RESULTS / 'LAUNCH.json').exists() or counts()['committed_worlds']:
        raise ValueError('already launched; no automatic restart or repeated trajectories')
    source = verify_sources()
    if read(OPS/'QUALIFICATION.json')['verdict'] != 'PASS':
        raise ValueError('operations qualification not accepted')
    def interrupted(signum, frame):
        raise InterruptedError(f'supervisor received signal {signum}; no automatic restart')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    start = time.monotonic()
    caffeine = subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())], stdin=subprocess.DEVNULL)
    atomic(RESULTS/'LAUNCH.json', dict(supervisor_pid=os.getpid(), caffeinate_pid=caffeine.pid,
           interpreter=sys.executable, workers=1, time_unix=time.time(), session_limit_seconds=36000,
           estimated_wall_minutes=read(ROOT/'QUALIFICATION.json')['conservative_32_world_single_worker_wall_minutes'],
           supervisor_log=str(OPS/'supervisor.log'), source_lock_sha256=source,
           worlds=list(WORLDS), total_lives=96, retries=0, GitHub_upload_authorized=False), exclusive=True)
    child = None
    index = {}
    try:
        for world in WORLDS:
            if time.monotonic()-start > 36000-300:
                status('PAUSED_SESSION_LIMIT', next_world=world,
                       pause_at_whole_world_boundary=True, resume_requires_explicit_authorization=True)
                return
            verify_sources()
            logpath = RESULTS/'logs'/f'{world}.log'
            logpath.parent.mkdir(parents=True, exist_ok=True)
            with logpath.open('x') as log:
                child = subprocess.Popen([sys.executable,'-u',str(OPS/'worker.py'),'--world',str(world)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
                print(f'DISPATCH world={world} pid={child.pid}',flush=True)
                while child.poll() is None:
                    status('EXECUTION', current_world=world, worker_pid=child.pid,
                           elapsed_wall_seconds=time.monotonic()-start)
                    if time.monotonic()-start >= 36000:
                        raise TimeoutError('10-hour session hard limit reached inside unfinished world')
                    time.sleep(2)
                if child.returncode != 0:
                    failure = RESULTS/'failures'/f'{world}.json'
                    error = read(failure)['error'] if failure.exists() else logpath.read_text()[-5000:]
                    raise RuntimeError(f'world {world} worker exit {child.returncode}: {error}')
            path = RESULTS/'receipts'/f'WORLD_{world}.json.gz'
            if not path.exists() or read(path)['independent_audit']['verdict'] != 'PASS':
                raise ValueError('worker exited without accepted full-world commit')
            index[str(world)] = dict(path=str(path), sha256=sha(path), training_lives=3, training_bytes=3840)
            atomic(RESULTS/'COMMIT_INDEX.json', index)
            status('EXECUTION', last_committed_world=world)
            print(f'COMMITTED {counts()}',flush=True)
            child = None
        status('ANALYSIS')
        from analysis import analyze
        summary = analyze()
        status('PACKAGING', verdict=summary['verdict'])
        package(summary)
        status('COMPLETE', verdict=summary['verdict'], final_audit='ACCEPTED_COMPLETE_EXPERIMENT',
               package_complete=True, elapsed_wall_seconds=time.monotonic()-start)
        print(f"COMPLETE {summary['verdict']}",flush=True)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        atomic(RESULTS/'FAILURE.json', dict(error=repr(exc), traceback=traceback.format_exc(),
               time_unix=time.time(), **counts(), retries=0), exclusive=True)
        status('FAILED', error=repr(exc), retries=0)
        raise
    finally:
        caffeine.terminate()
        caffeine.wait(timeout=10)


if __name__ == '__main__':
    main()
