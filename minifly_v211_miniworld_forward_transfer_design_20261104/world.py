"""Fixed MiniWorld schedules. Packet metadata belongs to the evaluator."""
from dataclasses import dataclass
from hashlib import sha256
import random
import string

FRAMES=("{p} went to the {l}","{p} is now in the {l}",
        "In the {l} is where {p} went","At the {l} is where {p} can be found")
ENDINGS=("", " today", " for now", " at present")

def seed(label):
    return int(sha256(label.encode()).hexdigest()[:16],16)

def rng(s,channel):
    return random.Random(seed(f"{s}|{channel}"))

@dataclass(frozen=True)
class Packet:
    events: tuple
    questions: tuple
    answers: tuple
    forms: tuple
    swap: bool

def balanced(r, seq, n):
    assert n%len(seq)==0
    values=list(seq)*(n//len(seq))
    r.shuffle(values)
    return values

def life(s, past=32, target_packets=32):
    used=set()
    vocab=rng(s,"vocab")
    def token(capital):
        while True:
            x="".join(vocab.choices(string.ascii_lowercase,k=6))
            if x not in used:
                used.add(x)
                return x.capitalize() if capital else x
    worlds=[]
    for i in range(past+1):
        target=i==past
        packets=target_packets if target else 8
        people=[token(True) for _ in range(2)]
        places=[token(False) for _ in range(4)]
        sizes=balanced(rng(s,f"{i}|sizes"),(2,6),packets)
        shapes=balanced(rng(s,f"{i}|shapes"),
                        [(a,b) for a in range(4) for b in range(4) if (a+b)%2==target],sum(sizes))
        movers=balanced(rng(s,f"{i}|movers"),(0,1),packets)
        question_orders=balanced(rng(s,f"{i}|q-order"),(0,1),packets)
        question_forms=balanced(rng(s,f"{i}|q-form"),(0,1),packets)
        swaps=balanced(rng(s,f"{i}|swap"),(False,True),packets) if not target else [False]*packets
        movement=rng(s,f"{i}|movement")
        latest={}
        pos=0
        world=[]
        for j,size in enumerate(sizes):
            events=[]
            forms=[]
            last=movers[j]
            for k in range(size):
                actor=people[(last+1+k)%2]
                candidates=[p for p in places if p not in latest.values()]
                loc=movement.choice(candidates)
                latest[actor]=loc
                a,b=shapes[pos]
                pos+=1
                forms.append((a,b))
                events.append(FRAMES[a].format(p=actor,l=loc)+ENDINGS[b]+".")
            ordered=people[question_orders[j]:]+people[:question_orders[j]]
            qform=("Where is {}?","Where can {} be found?")[question_forms[j]]
            world.append(Packet(tuple(events),tuple(qform.format(p) for p in ordered),
                                tuple(latest[p] for p in ordered),tuple(forms),swaps[j]))
        worlds.append(tuple(world))
    return tuple(worlds)
