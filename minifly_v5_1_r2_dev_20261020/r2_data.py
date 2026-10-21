"""Evaluator-owned addition data. Never imported by the learner."""
from r2_common import np
import hashlib,collections
BIRTHS=[202610205301,202610205302]
def orbit(a,b):return tuple(sorted((int(a),int(b))))
def partition(a,b,seed):
    return int.from_bytes(hashlib.blake2b(f'{seed}:{min(a,b)}:{max(a,b)}'.encode(),digest_size=8).digest(),'little')%10
def fact(a,b):return f'{a}+{b}='.encode(),str(a+b).encode()
def key(p):
    a,b=p[:-1].split(b'+');return orbit(int(a),int(b))
def build(seed):
    rng=np.random.default_rng(seed);used=set()
    def draw(ds,held):
        while True:
            a,b=[int(rng.integers(0 if d==1 else 10**(d-1),10**d)) for d in ds]
            k=orbit(a,b)
            if (partition(a,b,seed)==9)==held and k not in used:
                used.add(k);return fact(a,b)
    # Exactly four held orbits for each of the sixteen ordered length strata.
    held=[draw((d,e),True) for d in range(1,5) for e in range(1,5) for _ in range(4)]
    old=[draw((d,e),False) for d in range(1,5) for e in range(1,5)]
    new=[draw(tuple(rng.integers(1,5,2)),False) for _ in range(8)]
    listening={};caps={'A':4096,'B':16384,'C':4096,'D':4096}
    for phase,cap in caps.items():
        records=[];total=0
        while True:
            # D0 may repeat real observations but never a globally held orbit.
            ds=rng.integers(1,5,2)
            a,b=[int(rng.integers(0 if d==1 else 10**(d-1),10**d)) for d in ds]
            if partition(a,b,seed)==9:continue
            rec=fact(a,b);n=sum(map(len,rec))
            if total+n>cap:break
            records.append(rec);total+=n
        listening[phase]=records
    shuffled=list(old)
    for length in sorted(set(len(a) for _,a in old)):
        ix=[i for i,(_,a) in enumerate(old) if len(a)==length];order=list(rng.permutation(ix))
        for j,i in enumerate(order):shuffled[i]=(old[i][0],old[order[(j+1)%len(order)]][1])
    old_order=[int(i) for _ in range(32) for i in rng.permutation(16)]
    new_order=[int(i) for _ in range(16) for i in rng.permutation(8)]
    d=dict(seed=seed,old=old,new=new,held=held,listening=listening,shuffled=shuffled,old_order=old_order,new_order=new_order)
    assert len({key(p) for p,_ in held})==64
    h={key(p) for p,_ in held}
    assert not h&{key(p) for records in [old,new,*listening.values()] for p,_ in records}
    assert not {key(p) for p,_ in old}&{key(p) for p,_ in new}
    assert collections.Counter(a for _,a in old)==collections.Counter(a for _,a in shuffled)
    assert all(len(a)==len(b) for (_,a),(_,b) in zip(old,shuffled))
    return d
def jsonable(d):
    def cv(x):
        if isinstance(x,bytes):return {'hex':x.hex()}
        if isinstance(x,dict):return {str(k):cv(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return [cv(v) for v in x]
        return x
    return cv(d)
def orientations(records):
    out=[]
    for p,a in records:
        u,v=p[:-1].split(b'+');out.append((p,a))
        if u!=v:out.append((v+b'+'+u+b'=',a))
    return out
def countermodels(exposures,probes):
    raw={};canonical={};freq=collections.Counter();bylen=collections.defaultdict(collections.Counter)
    suffix=[collections.defaultdict(collections.Counter) for _ in range(5)]
    for p,a in exposures:
        raw[p]=a;canonical[key(p)]=a;freq[a]+=1;bylen[len(p)][a]+=1
        for k in range(5):suffix[k][p[-k:] if k else b''][a]+=1
    def top(c):return min(c,key=lambda a:(-c[a],a)) if c else None
    scores=collections.Counter()
    for p,a in probes:
        scores['raw_lookup']+=raw.get(p)==a;scores['canonical_lookup']+=canonical.get(key(p))==a
        scores['frequency']+=top(freq)==a;scores['prompt_length_frequency']+=top(bylen[len(p)])==a
        for k in range(5):scores['suffix'+str(k)]+=top(suffix[k][p[-k:] if k else b''])==a
    return {k:v/len(probes) for k,v in scores.items()}
