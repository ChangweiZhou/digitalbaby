"""Environment side only. A master observation is never an input to the learner."""
import itertools
import numpy as np
import settings
ALL=tuple(itertools.product(range(5),repeat=2))
HELD=((0,4),(1,1),(1,2),(2,1),(2,3),(3,2),(3,3),(4,0))
NEW=((1,3),(3,1),(2,4),(3,0))
OLD=tuple(p for p in ALL if p not in HELD and p not in NEW)
def expression(pair):return bytes((48+pair[0],43,48+pair[1],61))
def generate(world):
    settings.require_world(world)
    master={}
    for pc,(phase,keys) in enumerate((('OLD',OLD),('NEW',NEW))):
        blocks=[]
        for block in range(4):
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261011,world,pc,block])))
            records=[]
            for pair in sorted(keys):
                truth=48+pair[0]+pair[1]
                for kind in rng.permutation([0]*8+[1,2]):
                    target=None if kind==2 else truth
                    if kind==1:target=int(rng.choice([d for d in settings.ALPHABET if d!=truth]))
                    records.append(dict(pair=list(pair),truth=truth,kind=int(kind),target=target))
            records=[records[i] for i in rng.permutation(len(records))]
            present=[i for i,r in enumerate(records) if r['target'] is not None]
            targets=[records[i]['target'] for i in present]
            sr=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261011,world,pc,block,99])))
            shifted=sr.permutation(targets).tolist()
            for i,t in zip(present,shifted):records[i]['shuffled_target']=int(t)
            for record in records:
                record.setdefault('shuffled_target',None)
                record['raw']=(expression(record['pair'])+(b'' if record['target'] is None else bytes([record['target']]))+b'\n').hex()
                target=record['shuffled_target'] if phase=='OLD' else record['target']
                record['shuffled_raw']=(expression(record['pair'])+(b'' if target is None else bytes([target]))+b'\n').hex()
            blocks.append(records)
        master[phase]=blocks
    return dict(world=world,blocks=master)
def permitted(branch,phase):
    if branch not in settings.BRANCHES or phase not in ('OLD','NEW'):raise ValueError('branch/phase')
    return not (branch=='N_OLD' and phase=='OLD')
