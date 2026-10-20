"""Evaluator-only arithmetic data. Never imported by core.py or native.py."""
import random


def split(world):
    rng = random.Random(world)
    original = [(a, b) for a in range(10) for b in range(a+1, 10)]
    diagonal = [(d, d) for d in range(10)]
    for _ in range(10000):
        off = original.copy()
        rng.shuffle(off)
        held, new, old = off[:10], off[10:20], diagonal+off[20:]
        if {a+b for a,b in old} == set(range(19)) and min(sum(d in p for p in old) for d in range(10)) >= 3:
            ordered = lambda ps: sorted({p for a,b in ps for p in ((a,b),(b,a))})
            pairs = dict(OLD=ordered(old), NEW=ordered(new), HELD=ordered(held))
            return {k: [(f'{a}+{b}='.encode(), str(a+b).encode()) for a,b in v] for k,v in pairs.items()}, pairs
    raise ValueError('finite split rejection limit')


def records(world, history, phase):
    sets, _ = split(world)
    recs = sets[phase]
    passes = 128 if phase == 'OLD' else 32
    rng = random.Random((world << 8)+(1 if phase == 'OLD' else 2))
    sham = random.Random((world << 8)+3)
    for epoch in range(passes):
        order = list(range(len(recs)))
        rng.shuffle(order)
        arranged = [recs[i] for i in order]
        answers = [a for _,a in arranged]
        if phase == 'OLD' and history == 'SH_OLD':
            sham.shuffle(answers)
        for idx, ((p, truth), answer) in enumerate(zip(arranged, answers)):
            yield dict(prompt=p, answer=answer, truth=truth, epoch=epoch, offset=idx,
                       allowed=not (phase == 'OLD' and history == 'N_OLD'))


def metadata(world):
    sets, pairs = split(world)
    return dict(world=world, pairs=pairs,
                counts={k:len(v) for k,v in sets.items()},
                disjoint=not any(set(pairs[a]) & set(pairs[b]) for a,b in [('OLD','NEW'),('OLD','HELD'),('NEW','HELD')]),
                held_never_taught_in_either_order=all((b,a) not in pairs['OLD']+pairs['NEW'] for a,b in pairs['HELD']))
