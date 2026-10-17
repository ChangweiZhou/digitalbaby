"""Evaluator-only data and scores. Operative kernel never imports this file."""
import random
from collections import Counter
from bind_kernel import seed


def split(world):
    rng=random.Random(seed(world,"split"))
    off=[(a,b) for a in range(10) for b in range(a+1,10)]
    for attempt in range(1,10001):
        q=off.copy();rng.shuffle(q)
        groups={"old":[(a,a) for a in range(10)]+q[20:],"new":q[10:20],"held":q[:10]}
        if {a+b for a,b in groups["old"]}!=set(range(19)):
            continue
        if min(sum(d in p for p in groups["old"]) for d in range(10))<3:
            continue
        result={k:sorted({p for a,b in rows for p in ((a,b),(b,a))}) for k,rows in groups.items()}
        return result
    raise ValueError("input coverage construction failed")


def stream(world,group,n):
    data=split(world)[group];rng=random.Random(seed(world,group+"_stream"))
    rows=[]
    while len(rows)<n:
        block=data.copy();rng.shuffle(block);rows.extend(block)
    return [(f"{a}+{b}=".encode(),str(a+b).encode()) for a,b in rows[:n]]


def score(model,items,readout="MEM",cap=16):
    before=model.digest();rows=[]
    for prompt,answer in items:
        got,status,trace=model.generate(prompt,cap,readout)
        rows.append(dict(prompt=prompt.hex(),answer=answer.hex(),emitted=got.hex(),status=status,
                         correct=(status=="EOS" and got==answer),emitted_tokens=len(trace)))
    if model.digest()!=before:
        raise AssertionError("evaluation changed production state")
    return dict(accuracy=sum(r["correct"] for r in rows)/len(rows),rows=rows)


def sample_auc(points):
    import math
    x=[math.log2(n+1) for n,_ in points];y=[a for _,a in points]
    if x[-1]<=x[0]:
        return 0.
    return sum((x[i+1]-x[i])*(y[i]+y[i+1])/2 for i in range(len(x)-1))/(x[-1]-x[0])


def cpu_auc(points,budget):
    # Available state is held constant between saved completed-window states.
    if budget<=0:
        raise ValueError("positive common CPU budget required")
    points=sorted(points,key=lambda row:row[0]);area=0.
    for i,(cost,value) in enumerate(points):
        end=points[i+1][0] if i+1<len(points) else budget
        width=max(0.,min(end,budget)-min(cost,budget))
        area+=width*value
    return area/budget


def exact_correct(emitted,status,answer):
    return status=="EOS" and emitted==answer
