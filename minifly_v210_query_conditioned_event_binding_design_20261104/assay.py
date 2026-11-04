"""Bounded V210 supplied-pool readout and inherited vote adapter."""
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import random
import string

import numpy as np
from bridge import v,votes

ARMS=("VOTE","POSITION","POSITION+VOTE","QUERY-ANCHOR","QUERY-RECENT","QUESTION-BLIND","SAME-TYPE-SHUFFLE")
LEARNED=ARMS[3:]

def h(b):
    return int.from_bytes(hashlib.blake2s(b,digest_size=8).digest(),"big")

@dataclass(frozen=True)
class Candidate:
    word: str
    recent: str
    anchor: str
    local: str
    position: int
    source_key: tuple

def roster(seed, n, training):
    rng=random.Random(seed)
    assert n%16==0
    used=set()
    def token(prefix):
        while True:
            name=prefix+"".join(rng.choices(string.ascii_lowercase,k=5))
            if name not in used:
                used.add(name)
                return name
    # Pair identical strata so swaps can be exactly balanced.
    for j in range(n):
        k=j%16
        f,order,person,pad=[(k>>b)&1 for b in range(4)]
        names=[token("T" if training else "H") for _ in range(2)]
        places=[token("t" if training else "h") for _ in range(4)]
        yield (names,places,f,order,person,40*pad,(j//16)%2)

def render(record,pad=None,training=True):
    names,places,f,order,person,length,swap=record
    length=length if pad is None else pad
    sentences=[]
    candidates=[]
    for i,name in enumerate(names):
        text=("Then " if f else "")+name+(" "+("~^"*20)[:length-1] if length else "")
        roles=[1,0] if order^i else [0,1]
        positions=[]
        for j,role in enumerate(roles):
            text+=(" and" if j else "")+(" went to the " if role==0 else " avoided the ")
            word=places[2*i+role]
            positions.append((word,text,j))
            text+=word
        text+="."
        sentences.append(text)
        payload=dict(v.payload(text))
        for word,before,j in positions:
            assert name in before[:16] and (not length or name not in before[-32:])
            candidates.append(Candidate(word,before[-32:],before[:16],before[-12:],j,(word,payload[word])))
    frame=f if training else 1-f
    name=names[person]
    questions=(f"Where did {name} go?",f"Which place did {name} avoid?") if frame==0 else (
        f"For {name}, where did they go?",f"For {name}, which place did they avoid?")
    targets=tuple(places[2*person:2*person+2])
    assert len({c.word for c in candidates})==4 and set(targets)<={c.word for c in candidates}
    return sentences,tuple(candidates),questions,targets,swap

def gram(s):
    return {s[i:i+k] for k in (2,3,4) for i in range(len(s)-k+1)}

def take(keys,n):
    return sorted(keys,key=lambda k:(h(k.encode()),k))[:n]

@lru_cache(maxsize=4096)
def features(arm,q,c):
    local=gram(c.local)
    keys=["L|"+x for x in take(local,16)]
    if arm!="QUESTION-BLIND":
        anchor=gram(c.recent if arm=="QUERY-RECENT" else c.anchor)
        question=gram(q)
        keys+=["A|"+x for x in take(anchor&question,16)]
        keys+=["R|"+x for x in take(gram(c.recent)&question,16)]
        cross={a+"|"+b for a in take(question,16) for b in take(local,16)}
        keys+=["C|"+x for x in take(cross,48)]
    return tuple(sorted({h(x.encode())%65536 for x in keys}))

def normalized(model,q,candidates):
    raw=votes(model,q)
    weights=[raw.get(c.source_key,0) for c in candidates]
    total=sum(weights)
    return {c.word:w/total if total else 0 for c,w in zip(candidates,weights)}

def choose(scores):
    best=max(scores.values())
    tied={w for w,s in scores.items() if s==best}
    return min(tied,key=lambda w:(h(w.encode()),w)),tied

class Selector:
    def __init__(self,arm):
        self.arm=arm
        self.w=np.zeros(65536,np.float32)

    def score(self,q,c):
        ix=features(self.arm,q,c)
        return float(self.w[list(ix)].sum())/math.sqrt(len(ix))

    def teach(self,q,candidates,r,target,eta):
        byword={c.word:c for c in candidates}
        scores={c.word:r[c.word]+self.score(q,c) for c in candidates}
        wrong,_=choose({k:v for k,v in scores.items() if k!=target})
        if scores[target]-scores[wrong]<1:
            for word,direction in ((target,1),(wrong,-1)):
                ix=features(self.arm,q,byword[word])
                self.w[list(ix)]+=direction*eta/math.sqrt(len(ix))
