"""V213 world construction. Simulator truth is isolated from the learner."""
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import random

def seed(label):
    return int(sha256(label.encode()).hexdigest()[:16],16)

@dataclass(frozen=True)
class World:
    key: int
    entities: tuple
    positions: tuple
    opcodes: tuple
    attach: int
    held: tuple
    training: tuple
    twins: tuple

def generate(key):
    r=random.Random(key)
    ids=r.sample(range(16,256),14)
    entities,positions,ops=tuple(ids[:8]),tuple(ids[8:12]),tuple(ids[12:])
    attach=r.choice(ops)
    release=next(x for x in ops if x!=attach)
    first,second=entities[:4],entities[4:]
    held=tuple(zip(first,r.sample(second,4)))
    taught=[pair for pair in product(first,second) if pair not in held]
    r.shuffle(taught)

    def init(a,b):
        x,y=r.choices(positions,k=2)
        return (("obs",a,x),("obs",b,y)),x,y

    def revealing(a,b,x,y):
        z=r.choice([p for p in positions if p not in (x,y)])
        d=r.choice([i for i in entities if i not in (a,b)])
        e=r.choice([i for i in entities if i not in (a,b,d)])
        return (("obs",d,r.choice(positions)),("obs",e,r.choice(positions)),("move",a,z)),z

    training=[]
    for a,b in taught:
        sequences=list(product(ops,repeat=2))*2
        r.shuffle(sequences)
        for o1,o2 in sequences:
            packet,x,y=init(a,b)
            for o in (o1,o2):
                suffix,z=revealing(a,b,x,y)
                y=z if o==attach else y
                packet+=(("event",o,a,b),)+suffix+(("obs",b,y),)
                x=z
            training.append(packet)
    twins=[]
    for subset,pairs in (("held",held),("taught",taught[:4])):
        for a,b in pairs:
            for earlier in (attach,release):
                for _ in range(2):
                    prefix,x,y=init(a,b)
                    prefix+=(("event",earlier,a,b),)
                    suffix,z=revealing(a,b,x,y)
                    branches=((prefix+(("event",attach,a,b),)+suffix,z),
                              (prefix+(("event",release,a,b),)+suffix,y))
                    twins.append((subset,earlier,branches))
    return World(key,entities,positions,ops,attach,held,tuple(training),tuple(twins))

def oracle(prefix, attach):
    values={}
    last=None
    for record in prefix:
        tag,*payload=record
        if tag=="obs":
            e,p=payload
            values[e]=p
        elif tag=="event":
            last=tuple(payload)
        elif tag=="move":
            e,p=payload
            values[e]=p
    o,a,b=last
    return values[a] if o==attach else values[b]

def audit(w):
    assert len(w.training)==96 and len(w.twins)==32
    seen=set()
    for offset in range(0,96,8):
        block=w.training[offset:offset+8]
        for packet in block:
            assert len(packet)==12
            for cut in (6,11):
                assert packet[cut][0]=="obs"
                assert oracle(packet[:cut],w.attach)==packet[cut][2]
            seen.update(tuple(r[-2:]) for r in packet if r[0]=="event")
        for cut in (2,7):
            assert all(sum(p[cut][1]==o for p in block)==4 for o in w.opcodes)
    assert len(seen)==12 and not seen.intersection(w.held)
    assert {a for a,_ in seen}=={a for a,_ in w.held}
    assert {b for _,b in seen}=={b for _,b in w.held}
    for panel,earlier,branches in w.twins:
        (a,x),(b,y)=branches
        assert len(a)==len(b)==7 and a[-3:]==b[-3:] and a[:3]==b[:3] and x!=y
        assert a[2][1]==earlier and a[3][1]!=b[3][1]
        assert (tuple(a[3][-2:]) in w.held)==(panel=="held")
        assert oracle(a,w.attach)==x and oracle(b,w.attach)==y
    return seen

def official_roster():
    return tuple(generate(seed(f"MINIFLY|V213|WORLD|{i}")) for i in range(8))
