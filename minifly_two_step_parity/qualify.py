"""Engineering-only qualification for the two-step parity experiment."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ARMS = ('aligned', 'timescale', 'shuffled', 'no_feedback')

def world(seed, suite, arm):
    rng = np.random.default_rng(seed)
    order = rng.permutation(12)
    parity = np.argsort(order) % 2
    polarity = int(rng.integers(2))
    omitted = int(rng.integers(12))
    permutation = rng.permutation(12)
    while np.array_equal(parity[permutation], parity) or np.array_equal(parity[permutation], 1-parity):
        permutation = rng.permutation(12)
    maps = np.zeros((2, 12, 12))
    eye = np.eye(12)
    for start in rng.integers(12, size=1200):
        run = order[(start+np.arange(16)) % 12]
        for index, stride in enumerate((1, 1 if arm == 'timescale' else 2)):
            sequence = permutation[run] if index == 1 and arm == 'shuffled' else run
            gamma = .65 if index == 0 else .85
            for x, y in zip(sequence, sequence[stride:]):
                maps[index, x] += .12*(eye[x]+gamma*maps[index, y]-maps[index, x])
    projection = rng.normal(size=(2000, 48))
    pairs = [(int(x), int(y), d) for i,x in enumerate(order)
             for d in range(1,12) for y in [order[(i+d)%12]]]
    def held(pair):
        x,y,d = pair
        return d in (2,3,9,10) if suite == 'distance' else omitted in (x,y)
    train = [(x,y) for x,y,d in pairs if not held((x,y,d))]
    test = [(x,y) for x,y,d in pairs if held((x,y,d))]
    assert not (set(train) & (set(test) | {(y,x) for x,y in test}))
    def answer(x,y):
        return int(parity[x] == parity[y]) ^ polarity
    assert {answer(x,y) for x,y in train} == {0,1}
    code = {}
    for x,y,_ in pairs:
        values = np.tanh(projection @ maps[:, [x,y], :].ravel())
        ids = np.argsort(-np.abs(values), kind='stable')[:100]
        code[x,y] = (ids,values[ids]/np.linalg.norm(values[ids]))
    weights = np.zeros((2000,256))
    eos = np.zeros((256,256), dtype=int)
    for _ in range(24):
        for i in rng.permutation(len(train)):
            x,y = train[i]
            target = 60+2*answer(x,y)
            eos[target,10] += 1
            if arm != 'no_feedback':
                ids,v = code[x,y]
                err = -(v @ weights[ids])
                err[target] += 1
                weights[ids] += .3*np.outer(v,err)
    before = hashlib.sha256(weights.tobytes()).hexdigest()
    hits = joint = 0
    for x,y in test:
        ids,v = code[x,y]
        emitted = int(np.argmax(v @ weights[ids]))
        hits += emitted == 60+2*answer(x,y)
        joint += emitted == 60+2*answer(x,y) and np.argmax(eos[emitted]) == 10
    assert before == hashlib.sha256(weights.tobytes()).hexdigest()
    assert arm != 'no_feedback' or not weights.any()
    return dict(seed=seed,suite=suite,arm=arm,test=len(test),hits=int(hits),joint=int(joint),state=before)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print(json.dumps(world(int(sys.argv[1]), *sys.argv[2:])))
        sys.exit()
    a = world(101,'distance','no_feedback')
    assert a['hits'] == a['joint'] == 0
    jobs = [(seed,suite,arm) for seed in (101,102) for suite in ('distance','item') for arm in ARMS]
    rows = [world(*job) for job in jobs]
    check = json.loads(subprocess.check_output([sys.executable,__file__,'101','distance','aligned']))
    assert check == rows[0]
    result = {'status':'ENGINEERING_PARTIAL', 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'cycle_1':'Holdout, reverse-pair exclusion, feedback clamp and read-only scoring checks passed',
              'cycle_2':rows, 'cycle_3':'Fresh-process result and state equality passed for one engineering job'}
    Path(__file__).with_name('ENGINEERING.json').write_text(json.dumps(result,indent=2)+'\n')
