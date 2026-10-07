"""Launch authorization and atomic evidence I/O; no native learning code."""
import contextlib
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

OPS = Path(__file__).resolve().parent
ROOT = OPS.parent.parent
RESULTS = ROOT / 'results' / 'confirm'
WORLDS = tuple(range(811001, 811033))
EVIDENCE = 'CONFIRM_E0_E1_ORDINARY_OBSERVATION'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    return json.loads(path.read_text())


def atomic(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()
    if path.suffix == '.gz':
        data = gzip.compress(data, mtime=0)
    try:
        with temp.open('xb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if exclusive:
            os.link(temp, path)  # Atomic no-overwrite commit, including racing dispatchers.
            temp.unlink()
        else:
            os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@contextlib.contextmanager
def authorized_worlds(worlds=WORLDS):
    """Rebind only the DEV admission check; restore it on exit."""
    import spec
    if tuple(worlds) != WORLDS:
        raise ValueError('authorization roster must equal the frozen 32-world roster')
    original = spec.require_dev

    def require_confirm(world):
        if type(world) is not int or world not in WORLDS:
            raise ValueError('not an authorized fresh confirmation world')

    spec.require_dev = require_confirm
    try:
        yield
    finally:
        spec.require_dev = original


def verify_sources():
    lock = read(OPS / 'SOURCE_LOCK.json')
    for base, key in ((ROOT, 'qualified_files'), (OPS, 'operation_files')):
        for name, expected in lock[key].items():
            if sha(base / name) != expected:
                raise AssertionError('frozen source changed: ' + str(base / name))
    parent = ROOT.parent / 'r_center_core_v1'
    if sha(parent / 'SOURCE_LOCK.json') != lock['parent_lock_sha256']:
        raise AssertionError('native parent lock changed')
    for name, expected in read(parent / 'SOURCE_LOCK.json')['files'].items():
        if sha(parent / name) != expected:
            raise AssertionError('native parent dependency changed: ' + name)
    return sha(OPS / 'SOURCE_LOCK.json')


def counts():
    files = sorted((RESULTS / 'receipts').glob('WORLD_*.json.gz'))
    return dict(committed_worlds=len(files), total_worlds=32,
                committed_training_lives=3 * len(files), total_training_lives=96,
                committed_training_bytes=3840 * len(files), total_training_bytes=122880)
