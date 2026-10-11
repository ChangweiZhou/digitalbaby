"""Truth and contamination exist only on the environment/auditor side."""
import itertools
import numpy as np
import settings

HELD_ORBITS=((1,1),(1,2),(0,4),(2,3),(3,3))
HELD=tuple(sorted(set(HELD_ORBITS)|{(b,a) for a,b in HELD_ORBITS}))
TRAIN=tuple(p for p in itertools.product(range(5),repeat=2) if p not in HELD)
NEW=((1,3),(3,1),(2,4),(3,0))
OLD=tuple(p for p in TRAIN if p not in NEW)
ALL=tuple(itertools.product(range(5),repeat=2))

def expression(pair):
    return bytes([48+pair[0],43,48+pair[1],61])

def generate(world,quick=False):
    settings.require_dev(world)
    repeats=10 if quick else settings.REPEATS
    result={}
    for phase,keys,offset in (('old',OLD,0),('new',NEW,1)):
        rng=np.random.default_rng(world*100+offset);records=[]
        for pair in keys:
            kinds=rng.permutation([0]*(repeats*8//10)+[1]*(repeats//10)+[2]*(repeats//10))
            truth=48+pair[0]+pair[1]
            for kind in kinds:
                target=None if kind==2 else truth
                if kind==1:target=int(rng.choice([d for d in settings.ALPHABET if d!=truth]))
                records.append(dict(pair=list(pair),truth=truth,noise_kind=int(kind),target=target))
        order=rng.permutation(len(records));records=[records[i] for i in order]
        present=[i for i,r in enumerate(records) if r['target'] is not None]
        targets=[records[i]['target'] for i in present]
        shuffled=rng.permutation(targets).tolist()
        for i,target in zip(present,shuffled):records[i]['shuffled_target']=target
        for r in records:
            r.setdefault('shuffled_target',None)
            r['raw']= (expression(r['pair'])+(b'' if r['target'] is None else bytes([r['target']]))+b'\n').hex()
            st=r['shuffled_target'] if phase=='old' else r['target']
            r['shuffled_raw']=(expression(r['pair'])+(b'' if st is None else bytes([st]))+b'\n').hex()
        result[phase]=records
    return dict(world=world,quick=bool(quick),old_keys=[list(x) for x in OLD],new_keys=[list(x) for x in NEW],held_keys=[list(x) for x in HELD],records=result)

def write_allowed(branch,phase):
    if branch not in settings.BRANCHES or phase not in ('old','new'):raise ValueError('branch/phase')
    return branch!='N_OLD' or phase=='new'

