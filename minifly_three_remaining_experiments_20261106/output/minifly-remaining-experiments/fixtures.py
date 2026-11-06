"""Engineering fixtures for the three remaining MiniFly protocols."""
from collections import Counter
from itertools import combinations_with_replacement
from pathlib import Path
from random import Random
import sys

OUTPUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OUTPUT / "minifly-recurrent-worked-examples"))
from qualify_worked import fixture, parse
from benchmark import expected

EOS = 256
SEED = 57467
ACTORS = bytes.maketrans(bytes(range(128,136)),bytes(range(160,168)))


def graph_bank(phase):
    bank, probes = fixture(SEED + (phase == "B"), "A")
    if phase=="B":
        bank = [(k,p.translate(ACTORS),ys) for k,p,ys in bank]
        probes = [(k,[p.translate(ACTORS) for p in ps],ys) for k,ps,ys in probes]
    return bank,probes


def graph_stream(bank,phase,law,consistent,count):
    rng=Random(f"{SEED}:{phase}:life")
    for offset in range(0,count,len(bank)):
        for i in rng.sample(range(len(bank)),len(bank))[:count-offset]:
            k,p,ys=bank[i]
            choice=law if consistent else (law+(k%64>=32)+offset//len(bank))%2
            yield list(p)+[ys[choice],EOS]


def development_windows():
    book=(OUTPUT/"minifly-pretraining"/"books"/"2600.txt").read_bytes()
    base=book[4096:4096+16*129]
    text=[list(base[i:i+129]) for i in range(0,len(base),129)]
    rng=Random(SEED)
    synth=[]
    for _ in range(16):
        key=bytes(rng.sample(range(97,123),4))
        s=bytearray()
        while len(s)<129:
            s.extend(b"["+key+b"]"+bytes(rng.choices(range(65,91),k=rng.randint(2,7)))+key)
        synth.append(list(s[:129]))
    return text,synth


def carry(pair):
    a,b=pair
    return int(a%10+b%10>=10)


def mod_split(seed=SEED):
    rng=Random(seed)
    diag=set(rng.sample(range(97),48))
    extra=set(rng.sample(range(97),48))
    out=[[],[],[]]
    for residue in range(97):
        groups=[[p for p in combinations_with_replacement(range(97),2)
                 if sum(p)%97==residue and p[0]!=p[1] and carry(p)==c] for c in range(2)]
        d=(residue*49)%97
        n=24+(residue in extra)-(residue in diag)
        first=round(n*len(groups[0])/48)
        for g,k in zip(groups,(first,n-first)):
            rng.shuffle(g)
            out[0].extend(g[:k])
            rest=g[k:]
            side=1 if len(out[1])<=len(out[2]) else 2
            out[side].extend(rest[::2])
            out[3-side].extend(rest[1::2])
        side=1 if len(out[1])<=len(out[2]) else 2
        out[0 if residue in diag else side].append((d,d))
    return out


def expand(pairs):
    return [p for a,b in pairs for p in (((a,b),(b,a)) if a!=b else ((a,b),))]


def arithmetic(pair,mod=0,mark=b"="):
    a,b=pair
    prompt=f"{a:02d}+{b:02d}".encode()+mark
    y=(a+b)%mod if mod else a+b
    return list(prompt)+list(f"{y:02d}".encode())+[EOS]


def tally(n,width):
    return b"|"*n+b"."*(width-n)


def counting(pairs,arm,count):
    rng=Random(SEED+91)
    order=list(pairs)
    for offset in range(0,count,len(pairs)):
        rng.shuffle(order)
        perm=order[1:]+order[:1]
        while any(sorted(a)==sorted(b) for a,b in zip(order,perm)):
            rng.shuffle(perm)
        for i,(pair,other) in enumerate(zip(order,perm)):
            if i+offset>=count:
                break
            n=(i+offset)%19
            a,b=pair if arm=="worked" else other
            count_row=list(b"C"+tally(n,18)+b"?")+list(f"{n:02d}".encode())+[EOS]
            trace=arithmetic(pair,mark=b"W")[:6]+list(tally(a,9)+b"/"+tally(b,9))+[EOS]
            yield count_row,arithmetic(pair),trace


def cycle1():
    graph={}
    for phase in ("A","B"):
        bank,probes=graph_bank(phase)
        exposed={p for _,p,_ in bank}
        assert len(exposed)==512 and len(probes)==64
        assert all(p not in exposed for _,ps,_ in probes for p in ps[2:])
        assert all(expected(parse(p),op)==ys[law]
                   for _,p,ys in bank for law,op in enumerate(b"XY"))
        a=list(graph_stream(bank,phase,0,True,1024))
        b=list(graph_stream(bank,phase,0,False,1024))
        assert [r[:-2] for r in a]==[r[:-2] for r in b]
        assert sum(x[-2]!=y[-2] for x,y in zip(a,b))==256
        graph[phase]={"feedback_prefixes":512,"held_keys":64,"same_law_controls":True}
    assert {a for _,p,_ in graph_bank("A")[0] for a in parse(p).locations}.isdisjoint(
            {a for _,p,_ in graph_bank("B")[0] for a in parse(p).locations})
    train,monitor,sealed=mod_split()
    assert len(train)==2376 and sorted((len(monitor),len(sealed)))==[1188,1189]
    assert len(set(train+monitor+sealed))==4753
    x,y,z=map(expand,(train,monitor,sealed))
    assert len(x)==4704 and len(y)+len(z)==4705
    assert set(x).isdisjoint(y+z) and set(y).isdisjoint(z)
    trained=Counter(sum(p)%97 for p in x)
    assert max(trained.values())-min(trained.values())<=3
    assert all(int(bytes(arithmetic(p,97)[6:8]))==sum(p)%97 for p in x+y+z)
    train3=[(a,b) for a in range(10) for b in range(10) if min(a,b)<=4]
    held3=[(a,b) for a in range(5,10) for b in range(5,10)]
    sets=[list(counting(train3,arm,75)) for arm in ("answer","shuffled","worked")]
    assert all([(c,a) for c,a,_ in part]==[(c,a) for c,a,_ in sets[0]] for part in sets)
    assert Counter(tuple(t[6:]) for _,_,t in sets[0])==Counter(tuple(t[6:]) for _,_,t in sets[2])
    assert not set(tuple(arithmetic(p)[:6]) for p in held3)&{tuple(a[:6]) for _,a,_ in sets[0]}
    assert all(not set(t[6:-1])&set(range(48,58)) for _,_,t in sets[2])
    assert all(sum(map(len,row))==58 for row in sets[0])
    return {"encoder_graph":graph,"mod97":{"train":len(x),"monitor":len(y),"sealed":len(z),
            "ordered_oracle_checks":len(x+y+z),"carry_fractions":[sum(map(carry,p))/len(p) for p in (x,y,z)]},
            "counting":{"taught":len(train3),"held":len(held3),"symbols_per_group":58,
            "held_equation_overlap":0,"matched_trace_marginals":True,
            "held_answer_values_in_counting":sorted({sum(p) for p in held3})}}
