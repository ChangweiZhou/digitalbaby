"""Launch in a separate session with file handles, then return immediately."""
import json
import os
import subprocess
import sys
from compact_bridge import ROOT
from compact_integrity import verify
from compact_storage import atomic_json


def main():
    identity = verify()
    status_path = ROOT / 'results/STATUS.json'
    if status_path.exists():
        old = json.loads(status_path.read_text())
        if old['state'] in ('RUNNING', 'FAILED', 'COMPLETE'):
            raise ValueError('existing run requires review; no duplicate/automatic restart')
    log = (ROOT / 'results/supervisor.log').open('a')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMBA_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    proc = subprocess.Popen([sys.executable, '-u', str(ROOT / 'supervisor.py')], cwd=ROOT,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, close_fds=True, env=env)
    record = {'pid': proc.pid, 'identity': identity, 'detached_session': True, 'log': 'results/supervisor.log',
              'python': sys.executable, 'authorized_science': True}
    atomic_json(ROOT / 'results/LAUNCH.json', record)
    print(json.dumps(record))


if __name__ == '__main__': main()
