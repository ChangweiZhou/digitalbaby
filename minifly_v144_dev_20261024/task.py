"""Evaluator-only addition data. Never imported by the reference learner."""
import numpy as np
from reference import seed

ORBITS = [(a,b) for a in range(10) for b in range(a,10)]


def prompt(pair):
    return ('%d+%d=' % tuple(pair)).encode()


def answer(pair):
    return str(sum(pair)).encode()


def oriented(orbits):
    return [p for a,b in orbits for p in ([(a,b)] if a==b else [(a,b),(b,a)])]


def coverage(orbits):
    return min(sum(d in pair for pair in orbits) for d in range(10))


def world(name):
    rng = np.random.default_rng(seed('split:' + name))
    for _ in range(10000):
        perm = [ORBITS[i] for i in rng.permutation(55)]
        held, train = perm[:15], perm[15:]
        short = sum(len(answer(p))==1 for p in held)
        if 5 <= short <= 10 and coverage(train)>=4 and coverage(train[:20])>=2:
            return {'name':name,'held':sorted(held),'training_order':train,
                    'doses':{str(n):train[:n] for n in (20,30,40)}}
    raise RuntimeError('no admissible split; do not change constraints')


def score(model, pairs):
    rows=[]
    for pair in pairs:
        output,status=model.generate(prompt(pair))
        target=answer(pair)
        rows.append({'pair':list(pair),'output_hex':output.hex(),'status':status,
                     'exact':status=='EOS' and output==target,
                     'payload':output==target,'own_eos':status=='EOS'})
    groups={}
    for row in rows:
        groups.setdefault(tuple(sorted(row['pair'])),[]).append(float(row['exact']))
    return {'orbit_exact':float(np.mean([np.mean(x) for x in groups.values()])),
            'oriented_exact':float(np.mean([x['exact'] for x in rows])),
            'payload_only':float(np.mean([x['payload'] for x in rows])),
            'own_eos':float(np.mean([x['own_eos'] for x in rows])),
            'invalid':float(np.mean([x['status']=='INVALID' for x in rows])),
            'truncated':float(np.mean([x['status']=='TRUNCATED' for x in rows])),
            'rows':rows}
