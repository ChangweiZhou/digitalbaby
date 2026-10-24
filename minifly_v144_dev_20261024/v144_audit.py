"""V144 audit/accounting; unique module name avoids native common.py shadowing."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


class Ledger:
    def __init__(self, name, stage):
        self.path = HERE / 'ops' / (name + '.jsonl')
        self.path.parent.mkdir(exist_ok=True)
        if self.path.exists():
            raise RuntimeError('attempt IDs are immutable: ' + name)
        self.name, self.stage = name, stage
        self.start = time.monotonic()
        self.emit('start')

    def emit(self, event, **fields):
        row = dict(attempt=self.name, stage=self.stage, pid=os.getpid(), event=event,
                   wall_s=time.monotonic()-self.start, process_cpu_s=time.process_time(), **fields)
        with self.path.open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')


def source_lock(output):
    files = {Path(m.__file__).resolve() for m in list(sys.modules.values())
             if getattr(m, '__file__', None) and Path(m.__file__).resolve().is_relative_to(ROOT)}
    files |= set(HERE.glob('*.py'))
    files |= set((ROOT / 'minifly_a_v3/package').rglob('*.json'))
    files |= set((ROOT / 'minifly_a_v3/package').rglob('*.npz'))
    files |= {ROOT/'minifly_cls_dev_20261017/OUTPUT_CODEBOOK.json',
              ROOT/'minifly_v144_addition_readiness_design_20261024/SPEC.json'}
    value = {'files': {str(p.relative_to(ROOT)): sha(p) for p in sorted(files) if p.is_file()},
             'python': sys.version, 'numerical_threads': 1}
    write_json(output, value)
    return value


def validate(lock, rows, required, closure_ok=True, expected_settings=None):
    problems = []
    for rel, digest in lock['files'].items():
        p = ROOT / rel
        if not p.is_file() or sha(p) != digest:
            problems.append('source/asset mismatch: ' + rel)
    if not closure_ok:
        problems.append('runtime closure failed')
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path:
            p = Path(path).resolve()
            if p.is_relative_to(ROOT) and p.suffix == '.py' and str(p.relative_to(ROOT)) not in lock['files']:
                problems.append('unlocked runtime module: ' + str(p.relative_to(ROOT)))
    ids = [r.get('id') for r in rows]
    if len(ids) != len(set(ids)):
        problems.append('duplicate history')
    if set(ids) != set(required):
        problems.append('missing/extra history')
    for row in rows:
        if row.get('complete') is not True or not row.get('metrics'):
            problems.append('missing completeness/metrics: ' + str(row.get('id')))
        if expected_settings is not None and row.get('settings') != expected_settings:
            problems.append('invalid settings: ' + str(row.get('id')))
    return {'valid': not problems, 'problems': problems,
            'verdict': 'QUALIFIED' if not problems else 'INVALID_FOR_VERDICT'}
