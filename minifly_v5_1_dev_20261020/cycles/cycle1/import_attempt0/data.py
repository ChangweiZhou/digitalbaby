"""Evaluator-owned addition generator and globally excluded orbit partitions."""
from common import np
import hashlib
def orbit(a,b):return tuple(sorted((int(a),int(b))))
def part(a,b,seed):
    h=hashlib.blake2b(f'{seed}:{min(a,b)}:{max(a,b)}'.encode(),digest_size=8).digest()
    return int.from_bytes(h,'little')%10
def stream(seed,count,partition='train',maxdigits=4):
    rng=np.random.default_rng(seed);allowed={'train':set(range(8)),'dev':{8},'held':{9}}[partition]
    out=[]
    while len(out)<count:
        ds=rng.integers(1,maxdigits+1,2)
        a,b=[int(rng.integers(0 if d==1 else 10**(d-1),10**d)) for d in ds]
        if part(a,b,seed) in allowed:out.append((f'{a}+{b}='.encode(),str(a+b).encode()))
    return out
def score(model,records):
    before=model.digest();rows=[]
    for p,a in records:
        o=model.generate(p);rows.append(dict(prompt=p.hex(),want=a.hex(),**o,exact=bool(o['payload']==a.hex() and o['ended'] and not o['truncated'])))
    assert before==model.digest(),'query changed persistent state'
    return dict(exact=sum(x['exact'] for x in rows)/max(1,len(rows)),rows=rows)
