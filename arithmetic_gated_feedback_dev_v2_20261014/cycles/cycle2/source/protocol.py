"""Environment-side schedules, permissions and immutable source identity."""
import hashlib
import json
import os
import random
from pathlib import Path
from worlds import make_world, schedule, shuffle_targets, probes

ROOT = Path(__file__).resolve().parent
D = json.loads((ROOT / 'DESIGN.json').read_text())
LOCAL_FILES = ('core.py', 'protocol.py', 'worlds.py', 'runner.py', 'runner_helpers.py',
               'auditor.py', 'independent_truth.py', 'analysis.py', 'tests.py',
               'tamper_tests.py', 'legacy_keep_reference.py', 'supervisor.py',
               'DESIGN.json', 'EXPERIMENT.md', 'IMPLEMENTATION_NOTES.md')


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.pending-' + str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def seed(world, index, kind='action'):
    return int.from_bytes(hashlib.sha256(f'MF|{world}|{index}|{kind}'.encode()).digest()[:8], 'big') % (2 ** 63)


def show(world, index, phase):
    return random.Random(seed(world, index, 'visibility')).random() < D['course'][phase]['answer_probability']


def plans(world, nb, nm, stages):
    w = make_world(world)
    binary, mixed = schedule(w, 'binary', nb), schedule(w, 'mixed', nm)
    first = nb - 3 * nm
    assert first >= 0 and first % 4 == 0
    warm = binary[:first]
    j = first
    for row in mixed:
        warm.extend(binary[j:j + 3])
        warm.append(row)
        j += 3
    assert len(warm) == nb + nm
    binary = schedule(w, 'binary', sum(stages) // 2)
    mixed = schedule(w, 'mixed', sum(stages) // 2)
    tail = [x for pair in zip(binary, mixed) for x in pair]
    ends, position = {}, 0
    for phase, count in zip(('HALF', 'SPARSE', 'SCALAR'), stages):
        ends[phase] = tail[position:position + count]
        position += count
    return w, warm, shuffle_targets(warm, world + 19), ends


def identity():
    result = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in LOCAL_FILES}
    p = ROOT.parent / 'arithmetic_end_to_end_design_v1_20261013/validate_design.py'
    result['external:evaluator_data_source'] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result
