"""Evaluator-owned observation generator. Not imported by the learning core."""
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
from vendor.task_source import make_split, prompt, answer

OPS = ('+', '-', '*', '/')
CHECKPOINTS = (4, 8, 16, 32, 64, 128)
TECH_MIN, TECH_MAX = 202610130100, 202610130199
SCI_MIN, SCI_MAX = 202610130201, 202610130208


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def make_world(world, technical=True, checkpoints=CHECKPOINTS):
    if not technical: raise PermissionError("preparation runner admits reserved technical worlds only")
    if technical:
        if not TECH_MIN <= world <= TECH_MAX:
            raise ValueError('technical namespace')
    elif not SCI_MIN <= world <= SCI_MAX or tuple(checkpoints) != CHECKPOINTS:
        raise ValueError('scientific roster or budget changed')
    split = make_split(world)
    train = [r for x in split.values() for r in x['train']]
    test = [r for x in split.values() for r in x['test']]
    rng = random.Random(world + 771)
    pools = {op: sorted(r for r in train if r[0] == op) for op in OPS}
    loops = {op: [] for op in OPS}
    while any(len(rows) < max(checkpoints) for rows in loops.values()):
        for op in OPS:
            rows = pools[op].copy()
            rng.shuffle(rows)
            loops[op].extend(rows)
    records, blocks = [], []
    start = 0
    rotation = (world - (TECH_MIN if technical else SCI_MIN)) % 4
    for ci, stop in enumerate(checkpoints):
        order = OPS[rotation:] + OPS[:rotation]
        for op in order:
            indices = []
            for row in loops[op][start:stop]:
                index = len(records)
                records.append({'index': index, 'operation': op, 'checkpoint': stop,
                    'prompt': prompt(row), 'answer': answer(row), 'row': list(row)})
                indices.append(index)
            blocks.append({'operation': op, 'checkpoint': stop, 'record_indices': indices})
        start = stop
        rotation = (rotation + 1) % 4
    # Each operation/checkpoint increment has a fixed answer permutation.
    shuffled = {}
    for block in blocks:
        answers = [records[i]['answer'] for i in block['record_indices']]
        random.Random(world*10000 + block['checkpoint']*10 + OPS.index(block['operation'])).shuffle(answers)
        shuffled.update(zip(block['record_indices'], answers))
    obj = {'world': world, 'technical': technical, 'checkpoints': list(checkpoints),
           'held_families': {k: v['held'] for k, v in split.items()},
           'training_pool': [list(r) for r in train], 'held_pool': [list(r) for r in test],
           'records': records, 'blocks': blocks,
           'shuffled_answers': {str(i): a for i, a in shuffled.items()},
           'shuffle_fixed_points': sum(records[i]['answer'] == a for i, a in shuffled.items())}
    # Canonical primitive types BEFORE returning: tuples must not only turn into
    # lists when a receipt is written. The auditor reconstructs from disk.
    obj = json.loads(json.dumps(obj))
    cohort_rng = random.Random(world + 19088743)
    keys = {bytes([0,10,61,255])}
    while len(keys) < 8: keys.add(bytes(cohort_rng.randrange(256) for _ in range(4)))
    pairs = []
    for i,key in enumerate(sorted(keys)):
        value = bytes(cohort_rng.randrange(256) for _ in range(1+cohort_rng.randrange(3)))
        if i == 0: value = bytes([0,10,61])
        pairs.append({'prompt_hex':key.hex(),'answer_hex':value.hex()})
    cohort = []
    for repeat in range(4):
        order = list(range(8)); cohort_rng.shuffle(order)
        cohort.extend(dict(pairs[i], index=len(cohort)+j) for j,i in enumerate(order))
    obj['cohort_pairs'] = pairs
    obj['cohort_records'] = cohort
    obj['stream_digest'] = digest(obj)
    return obj
