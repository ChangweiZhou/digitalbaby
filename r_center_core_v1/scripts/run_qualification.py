"""Bounded engineering-only acceptance. No automatic retries or science entry."""
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from centered_core.integrity import verify_sources


def write(path, doc):
    temp = path.with_suffix(path.suffix + '.pending')
    temp.write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    os.replace(temp, path)


def main():
    os.chdir(ROOT)
    identity = verify_sources()
    started = time.monotonic()
    results = ROOT / 'results'; results.mkdir(exist_ok=True)
    status = {'kind': 'engineering_only', 'pid': os.getpid(), 'state': 'RUNNING', 'source_identity': identity,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'cycles_completed': 0}
    write(results / 'STATUS.json', status)
    caffeinate = None
    if platform.system() == 'Darwin' and shutil.which('caffeinate'):
        caffeinate = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
        status['caffeinate_attached'] = True
    else: status['caffeinate_attached'] = False
    try:
        for cycle in (1, 2, 3):
            status['cycle'] = cycle
            write(results / 'STATUS.json', status)
            with (results / f'final_cycle{cycle}.log').open('w') as log:
                child = subprocess.run([sys.executable, '-u', '-m', f'tests.check_cycle{cycle}'],
                                       stdout=log, stderr=subprocess.STDOUT)
            if child.returncode:
                raise RuntimeError(f'cycle {cycle} failed: exit {child.returncode}; see final_cycle{cycle}.log')
            receipt = json.loads((results / f'CYCLE{cycle}.json').read_text())
            if receipt['verdict'] != 'PASS': raise RuntimeError(f'cycle {cycle} failed acceptance')
            status['cycles_completed'] = cycle
        import numpy, scipy, numba
        receipt = {'verdict': 'PASS', 'kind': 'E0 engineering qualification', 'source_identity': identity,
                   'cycles_completed': 3, 'elapsed_s': time.monotonic() - started,
                   'python': platform.python_version(), 'numpy': numpy.__version__, 'scipy': scipy.__version__,
                   'numba': numba.__version__, 'platform': platform.platform(),
                   'completed_utc': datetime.now(timezone.utc).isoformat(),
                   'no_science_worlds_or_new_mechanism_variants': True}
        verify_sources()
        write(results / 'QUALIFICATION.json', receipt)
        status.update(state='COMPLETE', elapsed_s=receipt['elapsed_s'])
        print(json.dumps(receipt), flush=True)
        return 0
    except Exception as exc:
        status.update(state='FAILED', error=str(exc), elapsed_s=time.monotonic() - started)
        print(json.dumps(status), file=sys.stderr, flush=True)
        return 1
    finally:
        write(results / 'STATUS.json', status)
        if caffeinate is not None:
            caffeinate.terminate(); caffeinate.wait()


if __name__ == '__main__': sys.exit(main())
