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
    if n%4:
        raise ValueError("complete four-record windows required")
    data=split(world)[group]
    pairs=[(f"{a}+{b}=".encode(),str(a+b).encode()) for a,b in data]
    return repeat_stream(world,group,pairs,n)


def repeat_stream(world,group,pairs,n):
    # Group only by exact raw prompt; answers do not affect order or counts.
    ordered=sorted(pairs,key=lambda r:r[0]);by_prompt=dict(ordered)
    if len(by_prompt)!=len(ordered) or n%4:
        raise ValueError("unique raw prompts and full windows required")
    rng=random.Random(seed(world,group+"_counts"))
    extra=[p for p,_ in ordered];rng.shuffle(extra)
    q,rem=divmod(n,len(ordered));counts={p:q+int(p in extra[:rem]) for p,_ in ordered}
    pool={p:c//2 for p,c in counts.items()};rows=[]
    rng=random.Random(seed(world,group+"_pair_layers"))
    for level in range(max(pool.values(),default=0)):
        layer=sorted(p for p,c in pool.items() if c>level);rng.shuffle(layer)
        if len(layer)%2:
            raise ValueError("odd pair layer not admitted by frozen doses")
        for index in range(0,len(layer),2):
            a,b=layer[index:index+2];rows.extend([a,b,a,b])
    tail=sorted(p for p,c in counts.items() if c%2)
    random.Random(seed(world,group+"_singleton_tail")).shuffle(tail);rows.extend(tail)
    if len(tail)%4 or len(rows)!=n:
        raise ValueError("wrong frozen tail/dose")
    return [(p,by_prompt[p]) for p in rows]


class SuffixBackoff:
    """Evaluator-only observed byte suffix counts; no task parser."""
    def __init__(self):
        self.counts={n:{} for n in range(5)}
    def observe(self,prompt,answer):
        prefix=tuple(prompt)
        for target in (*answer,256):
            for n in range(5):
                if n<=len(prefix):
                    key=prefix[-n:] if n else ()
                    self.counts[n].setdefault(key,Counter())[target]+=1
            if target!=256:prefix+= (target,)
    def next(self,prefix):
        prefix=tuple(prefix)
        for n in range(min(4,len(prefix)),-1,-1):
            row=self.counts[n].get(prefix[-n:] if n else ())
            if row:return min(row,key=lambda y:(-row[y],y))
        return 256
    def generate(self,prompt,cap=16):
        answer=bytearray()
        for _ in range(cap+1):
            byte=self.next(tuple(prompt)+tuple(answer))
            if byte==256:return bytes(answer),"EOS"
            if len(answer)==cap:return bytes(answer),"TRUNCATED"
            answer.append(byte)



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
