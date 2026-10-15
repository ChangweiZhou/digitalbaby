"""Evaluator-only arithmetic data; never imported by operative core.py."""
import random
from validate_design import split, ordered

CHECKPOINTS = [0,32,128,512,2048,8192,16384]
TECH_WORLDS = [202610143901,202610143902,202610143903]


def seed_for(world):
    worlds = TECH_WORLDS + list(range(202610144001,202610144017))
    rng = random.Random(2026101443001)
    mapping = {w:rng.getrandbits(32) for w in worlds}
    return mapping[world]


def render(pair):
    a,b = pair
    return f'{a}+{b}='.encode(), str(a+b).encode()


def stream(pairs, count, seed):
    rng = random.Random(seed)
    result = []
    while len(result) < count:
        block = pairs.copy()
        rng.shuffle(block)
        result.extend(block)
    return result[:count]


def prepare(world, old_n, new_n):
    if world not in TECH_WORLDS:
        raise ValueError('this runner authorizes technical worlds only')
    groups = {k:ordered(v) for k,v in split(world).items() if k in ('old','new','held')}
    old = [render(p) for p in stream(groups['old'],old_n,(world<<8)+1)]
    new = [render(p) for p in stream(groups['new'],new_n,(world<<8)+2)]
    shuffled = []
    rng = random.Random((world<<8)+3)
    stops = sorted(set([0,old_n]+[n for n in CHECKPOINTS if 0 < n < old_n]))
    for a,b in zip(stops,stops[1:]):
        answers = [answer for _,answer in old[a:b]]
        rng.shuffle(answers)
        shuffled.extend((prompt,answer) for (prompt,_),answer in zip(old[a:b],answers))
    return dict(world=world, seed=seed_for(world), groups=groups, old=old,new=new,shuffled=shuffled,
                checkpoints=stops)
