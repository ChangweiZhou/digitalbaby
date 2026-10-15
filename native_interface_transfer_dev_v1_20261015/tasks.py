"""Evaluator-only data generators. Operative models must never import this file."""
import random
from collections import Counter
import config


def native_fixture():
    rng=random.Random(config.TRACK_A_MAP_SEED)
    alphabet=b'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    clean=[];used=set()
    # Preserve the released wrapper's eight-space/four-visible-byte framing.
    # Labels are arbitrary and no arithmetic symbols or numeric roles occur.
    for i in range(8):
        while True:
            cue=b' '*8+bytes(rng.choices(alphabet,k=4))
            if cue[-4:] not in used:break
        used.add(cue[-4:])
        first=bytes([48+i%4])
        response=first if i<4 else first+bytes([48+((i+1)%4)])
        clean.append(dict(id=f'clean{i}',cue=cue,answer=response,stratum='clean'))
    challenge=[dict(id='suffix0',cue=b'mNqRstuvAxyz',answer=b'01',stratum='suffix_collision'),
               dict(id='suffix1',cue=b'wXyZabCdBxyz',answer=b'02',stratum='suffix_collision'),
               dict(id='suffix2',cue=b'eFgHijKlCuvw',answer=b'3',stratum='suffix_collision'),
               dict(id='suffix3',cue=b'nOpQrsTuDuvw',answer=b'30',stratum='suffix_collision')]
    result=clean+challenge
    assert len({x['cue'] for x in result})==12
    assert all(len(x['cue'])==12 and 10 not in x['cue'] for x in result)
    return result


def ordered_stream(items,n,seed):
    rng=random.Random(seed);result=[]
    while len(result)<n:
        block=list(items);rng.shuffle(block);result.extend(block)
    return result[:n]


def addition_split(world):
    if world not in config.TRACK_B_WORLDS:raise ValueError('development worlds only')
    rng=random.Random(world)
    diagonals=[(a,a) for a in range(10)]
    off=[(a,b) for a in range(10) for b in range(a+1,10)]
    for _ in range(10000):
        x=off.copy();rng.shuffle(x)
        groups=dict(old=diagonals+x[20:],new=x[10:20],held=x[:10])
        if {a+b for a,b in groups['old']}==set(range(19)) and min(sum(d in p for p in groups['old']) for d in range(10))>=3:
            return {k:sorted({p for a,b in pairs for p in ((a,b),(b,a))}) for k,pairs in groups.items()}
    raise ValueError('split construction failed independently of model')


def render(pair):
    a,b=pair
    return f'{a}+{b}='.encode(),str(a+b).encode()


def addition_streams(world,old_n=16384,new_n=1024):
    groups=addition_split(world)
    old=[render(p) for p in ordered_stream(groups['old'],old_n,(world<<8)+1)]
    new=[render(p) for p in ordered_stream(groups['new'],new_n,(world<<8)+2)]
    rng=random.Random((world<<8)+3);shuffled=[]
    boundaries=sorted({0,old_n,*[x for x in config.TRACK_B_OLD_DOSES if x<old_n]})
    for left,right in zip(boundaries,boundaries[1:]):
        labels=[y for _,y in old[left:right]];rng.shuffle(labels)
        assert Counter(labels)==Counter(y for _,y in old[left:right])
        shuffled.extend((p,y) for (p,_),y in zip(old[left:right],labels))
    return dict(groups=groups,old=old,new=new,shuffled=shuffled)


def model_seed(world):
    seeds=random.Random(202610155100)
    return {w:seeds.getrandbits(32) for w in config.TRACK_B_WORLDS}[world]
