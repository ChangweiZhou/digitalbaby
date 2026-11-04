"""Local teaching orchestration. No hidden state is passed to learners."""
from random import Random

def shuffled(w,packet_count):
    r=Random(w.key^213217)
    slots=[]
    for start in range(0,packet_count,8):
        block=w.training[start:start+8]
        a=[p[2][1] for p in block]
        b=[p[7][1] for p in block]
        r.shuffle(a)
        r.shuffle(b)
        slots.extend(zip(a,b))
    return slots

def fit(model,w,packets):
    order=shuffled(w,packets)
    for i,records in enumerate(w.training[:packets]):
        model.reset()
        eta=.02 if i<8 else .05 if i<16 else .10
        for j,r in enumerate(records):
            if j in (6,11):
                assert model.pending is not None
                model.reveal(r,eta,order[i][j==11])
            else:
                model.consume(r)
    return order
