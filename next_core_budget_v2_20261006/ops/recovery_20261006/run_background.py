"""Explicitly authorized detached continuation after reviewed operations repair."""
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OPS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def save(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def main():
    lock = (ROOT / 'ops/background_20261006/supervisor.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    launch = json.loads((OPS / 'LAUNCH_REQUEST.json').read_text())
    launch.update(pid=os.getpid(), process_group=os.getpgrp(), started_unix=started,
                  state='STARTING', detached=True, automatic_restart=False)
    save(ROOT / 'results/LAUNCH.json', launch)
    save(ROOT / 'results/STATUS.json', {
        'state': 'STARTING_AUTHORIZED_RESUME', 'pid': os.getpid(),
        'updated_unix': time.time(), 'current_stage_status': 'science/screen/STATUS.json',
    })
    try:
        import pipeline
        from runtime import execution_identity
        if execution_identity() != launch['execution_source_identity']:
            raise ValueError('recovery launch source identity changed')
        recovery = json.loads((OPS / 'RECOVERY_AUDIT.json').read_text())
        if (recovery['verdict'] != 'PASS' or
                recovery['new_execution_source_identity'] != execution_identity()):
            raise ValueError('recovery audit not current')
        if (ROOT / 'results/FAILURE.json').exists():
            raise ValueError('new unresolved execution failure; no automatic restart')
        out = pipeline.run(resume=True)
        save(OPS / 'SESSION_RESULT.json', out)
        save(ROOT / 'results/STATUS.json', dict(out, pid=os.getpid(), updated_unix=time.time()))
        print(json.dumps(out, ensure_ascii=False), flush=True)
        return 1 if out['state'] in ('FAILED', 'STOP_NEEDS_REVIEW') else 0
    except Exception as exc:
        traceback.print_exc()
        failure = {'state': 'FAILED', 'category': 'EXECUTION_FAILURE',
                   'error': f'{type(exc).__name__}: {exc}', 'pid': os.getpid(),
                   'updated_unix': time.time(), 'automatic_restart': False}
        save(ROOT / 'results/FAILURE.json', failure)
        save(ROOT / 'results/STATUS.json', failure)
        save(OPS / 'SESSION_RESULT.json', failure)
        return 1
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    sys.exit(main())
