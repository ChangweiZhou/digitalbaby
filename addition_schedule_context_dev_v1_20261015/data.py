"""Evaluator-only task and schedule construction; never imported by models."""
import hashlib
import random
from collections import Counter,defaultdict
import config


def seed(world,role):
    return int.from_bytes(hashlib.sha256(f'ADDITION-ORDER-V1|{world}|{role}'.encode()).digest()[:4],'little')


def split(world):
    if world not in (*config.SCIENCE_WORLDS,*config.TECH_WORLDS):raise ValueError('world outside fixed roster')
    rng=random.Random(world)
    diagonals=[(a,a) for a in range(10)];off=[(a,b) for a in range(10) for b in range(a+1,10)]
    for attempt in range(1,10001):
        x=off.copy();rng.shuffle(x)
        groups=dict(old=diagonals+x[20:],new=x[10:20],held=x[:10])
        if {a+b for a,b in groups['old']}==set(range(19)) and min(sum(d in p for p in groups['old']) for d in range(10))>=3:
            return {k:sorted({p for a,b in pairs for p in ((a,b),(b,a))}) for k,pairs in groups.items()},attempt
    raise ValueError('fixed coverage constraints unsatisfied')


def example(pair):return f'{pair[0]}+{pair[1]}='.encode(),str(sum(pair)).encode()


def shuffled_cycles(items,n,rng):
    result=[]
    while len(result)<n:
        block=list(items);rng.shuffle(block);result.extend(block)
    return result[:n]


def make_addition(world,technical=False):
    if technical:
        if world not in config.TECH_WORLDS:raise ValueError('technical namespace required')
        old_n,new_n=2,2
    else:
        if world not in config.SCIENCE_WORLDS:raise ValueError('production namespace required')
        old_n,new_n=16384,1024
    groups,attempt=split(world)
    streams={}
    for kind,n,offset in [('OLD',old_n,1),('NEW',new_n,2)]:
        pairs=shuffled_cycles(groups[kind.lower()],n,random.Random((world<<8)+offset))
        streams[kind]=[dict(id=f'{kind}:{i}',kind=kind,source_index=i,prompt=example(p)[0].hex(),answer=example(p)[1].hex()) for i,p in enumerate(pairs,1)]
    shuffled=[];rng=random.Random((world<<8)+3)
    boundaries=sorted({0,old_n,*[n for n in (32,128,512,2048,8192) if n<old_n]})
    for left,right in zip(boundaries,boundaries[1:]):
        labels=[r['answer'] for r in streams['OLD'][left:right]];rng.shuffle(labels)
        shuffled.extend(dict(r,answer=y) for r,y in zip(streams['OLD'][left:right],labels))
    return dict(world=world,technical=technical,groups=groups,split_attempts=attempt,
                streams=streams,shuffled=shuffled,old_n=old_n,new_n=new_n)


def ordered(source,schedule,history):
    if schedule not in config.SCHEDULES or history not in config.HISTORIES:raise ValueError('fixed condition')
    O=source['shuffled'] if history=='SH_OLD' else source['streams']['OLD'];N=source['streams']['NEW']
    if len(O)%len(N):raise ValueError('complete fixed interleave ratio required')
    ratio=len(O)//len(N)
    rows=list(O)+list(N) if schedule=='BLOCKED' else [r for j,n in enumerate(N) for r in [*O[j*ratio:(j+1)*ratio],n]]
    return [dict(r,allowed=history!='N_OLD' or r['kind']=='NEW') for r in rows]


def checkpoint_plan(source,schedule):
    rows=ordered(source,schedule,'W');total=len(rows);new=0;old=0;plan=defaultdict(list)
    global_grid=(0,total) if source['technical'] else config.GLOBAL_GRID
    for t in global_grid:plan[t].append('GLOBAL')
    new_grid=(0,1,2) if source['technical'] else config.NEW_GRID
    first=next(i for i,r in enumerate(rows,1) if r['kind']=='NEW')
    plan[first-1].append('NEW:0')
    for t,r in enumerate(rows,1):
        if r['kind']=='NEW':
            new+=1
            if new in new_grid:plan[t].append(f'NEW:{new}')
    return dict(sorted(plan.items()))


def native_fixture(world):
    if world not in (*config.NATIVE_MAPS,*config.TECH_WORLDS):raise ValueError('fixed native namespace')
    rng=random.Random(seed(world,'native-fixture'));letters=list(b'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ');rng.shuffle(letters)
    labels=list(b'0123');rng.shuffle(labels);rows=[]
    for j,p in enumerate(labels):
        prefix=bytes(rng.choices(letters,k=8));a,b=letters[8+2*j:10+2*j]
        for layer in ('collision','matched'):
            c=letters[j if layer=='collision' else 4+j]
            suffixes=[bytes([a,c,c,c]),bytes([b,c,c,c])] if layer=='collision' else [bytes([c,c,c,a]),bytes([c,c,c,b])]
            answers=[bytes([p,labels[(j+1)%4]]),bytes([p,labels[(j+2)%4]])] if j<2 else [bytes([p]),bytes([p,labels[(j+1)%4]])]
            rows.extend(dict(id=f'{layer}:{j}:{k}',cue=prefix+s,answer=y,stratum=layer,pair=j) for k,(s,y) in enumerate(zip(suffixes,answers)))
    return rows


def native_aliases(fixture):
    grouped=defaultdict(list)
    for item in fixture:
        for i,target in enumerate((*item['answer'],256)):
            prefix=item['answer'][:i];key=(item['cue']+prefix)[-4:]
            grouped[key].append(dict(id=item['id'],prefix=prefix.hex(),target=target,stratum=item['stratum']))
    return [dict(key=k.hex(),rows=v) for k,v in sorted(grouped.items()) if len({r['target'] for r in v})>1]
