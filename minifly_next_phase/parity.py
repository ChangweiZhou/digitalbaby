"""Engineering-only qualification for the two-step parity experiment."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from engineering import parse

ARMS = ('aligned', 'timescale', 'shuffled', 'no_feedback')

def world(seed, suite, arm):
    rng = np.random.default_rng(seed)
    order = rng.permutation(12)
    symbols = rng.choice(np.arange(97,123),size=12,replace=False)
    byte_index = {int(b):i for i,b in enumerate(symbols)}
    def query(x,y): return b"par "+bytes([symbols[x],symbols[y]])+b"="
    def operands(raw): return tuple(byte_index[b] for b in parse(raw))
    even = int(rng.choice([2,4]))
    odd = int(rng.choice([1,3,5]))
    distances = {even,12-even,odd,12-odd}
    parity = np.argsort(order) % 2
    polarity = int(rng.integers(2))
    omitted = int(rng.integers(12))
    permutation = rng.permutation(12)
    while np.array_equal(parity[permutation], parity) or np.array_equal(parity[permutation], 1-parity):
        permutation = rng.permutation(12)
    maps = np.zeros((2, 12, 12))
    eye = np.eye(12)
    for start in rng.integers(12, size=1200):
        raw = bytes(symbols[order[(start+np.arange(16)) % 12]].tolist())
        run = np.array([byte_index[b] for b in raw])
        for index, stride in enumerate((1, 1 if arm == 'timescale' else 2)):
            sequence = permutation[run] if index == 1 and arm == 'shuffled' else run
            gamma = .65 if index == 0 else .85
            for x, y in zip(sequence[:14], sequence[stride:]):
                maps[index, x] += .12*(eye[x]+gamma*maps[index, y]-maps[index, x])
    projection = rng.normal(size=(2000, 48))
    pairs = [(int(x), int(y), d) for i,x in enumerate(order)
             for d in range(1,12) for y in [order[(i+d)%12]]]
    def held(pair):
        x,y,d = pair
        return d in distances if suite == 'distance' else omitted in (x,y)
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
            x,y = operands(query(*train[i]))
            target = 60+2*answer(x,y)
            eos[target,10] += 1
            if arm != 'no_feedback':
                ids,v = code[x,y]
                err = -(v @ weights[ids])
                err[target] += 1
                weights[ids] += .3*np.outer(v,err)
    before = hashlib.sha256(weights.tobytes()).hexdigest()
    hits = joint = 0
    for pair in test:
        x,y = operands(query(*pair))
        ids,v = code[x,y]
        emitted = int(rng.choice([60,62])) if arm == "no_feedback" else int(np.argmax(v @ weights[ids]))
        hits += emitted == 60+2*answer(x,y)
        joint += emitted == 60+2*answer(x,y) and np.argmax(eos[emitted]) == 10
    assert before == hashlib.sha256(weights.tobytes()).hexdigest()
    assert arm != 'no_feedback' or not weights.any()
    # A two-step edge joins items in the same alternating group.
    adjacency = {i:set() for i in range(12)}
    for i,x in enumerate(order):
        y=order[(i+2)%12]
        adjacency[x].add(y)
    def group(x):
        result={x}
        for _ in range(12):
            result |= {y for a in result for y in adjacency[a]}
        return result
    x,y=train[0]
    mapping=answer(x,y)^int(y in group(x))
    assert all(answer(x,y)==(mapping^int(y in group(x))) for x,y in test)
    baselines={}
    for mode in ('constant','first','second','ordered','unordered'):
        def key(x,y):
            return {'constant':0,'first':x,'second':y,'ordered':(x,y),'unordered':tuple(sorted((x,y)))}[mode]
        counts={}
        for x,y in train:
            counts.setdefault(key(x,y),[0,0])[answer(x,y)] += 1
        baselines[mode]=sum(int(np.argmax(counts.get(key(x,y),[0,0])))==answer(x,y) for x,y in test)/len(test)
    return dict(baselines=baselines, seed=seed,suite=suite,arm=arm,test=len(test),hits=int(hits),joint=int(joint),state=before)

