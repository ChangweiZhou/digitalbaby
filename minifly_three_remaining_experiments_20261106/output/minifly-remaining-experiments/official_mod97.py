"""Staged modular-addition downscaling campaign."""
from hashlib import sha256
from pathlib import Path
from random import Random
import sys
import time

from campaign_common import (RUN,SEEDS,checkpoint,lock,measures,read,responses,restore,
                             usage,verify,write)
from fixtures import arithmetic,expand,mod_split
from models import Modular,Sensory,T,digest
from qualify_remaining import clock_update

ROOT=RUN/"mod97"
ARMS=("none","periodic","smooth")
BLOCK=512
DOSE=4096
BATCH=64


def model(seed,sensor):
    T.manual_seed(seed)
    m=Modular(sensor)
    return m,T.optim.Adam(m.parameters(),lr=.003)


def panels(seed):
    return tuple(map(expand,mod_split(seed)))


def stream(train,seed,sham=False):
    targets=[arithmetic(p,97) for p in train]
    if sham:
        perm=list(range(len(train)))
        rng=Random(seed+7401)
        rng.shuffle(perm)
        targets=[a[:6]+targets[j][6:] for a,j in zip(targets,perm)]
    order=Random(seed+1)
    chunk=[]
    while True:
        for row in order.sample(targets,len(targets)):
            chunk.append(row)
            if len(chunk)==BATCH:
                yield chunk
                chunk=[]


def score(m,pairs,raw=False):
    rows=[arithmetic(p,97) for p in pairs]
    prompts=[r[:6] for r in rows]
    gold=[r[6:] for r in rows]
    outs=responses(m,prompts,5)
    data=measures(prompts,gold,outs)
    exact=dict(zip(pairs,[a==b for a,b in zip(outs,gold)]))
    data.update(both_orders=sum(exact[a,b] and exact[b,a] for a,b in pairs if a<b),
                both_order_n=sum(a<b for a,b in pairs),
                wrap=[sum(exact[a,b] for a,b in pairs if a+b>=97),sum(a+b>=97 for a,b in pairs)],
                decimal_carry=[sum(exact[a,b] for a,b in pairs if a%10+b%10>=10),
                               sum(a%10+b%10>=10 for a,b in pairs)])
    if not raw:data.pop("rows")
    return data


def prepare(archive):
    split={str(seed):[{"n":len(panel),"sha256":sha256(bytes(x for p in panel for x in p)).hexdigest()}
                      for panel in panels(seed)] for seed in SEEDS}
    assert all([x["n"] for x in val][0]==4704 and sum(x["n"] for x in val[1:])==4705
               for val in split.values())
    lock(ROOT,Path(__file__).name,archive,dict(split=split,arms=ARMS,batch=BATCH,
         checkpoints=BLOCK,acquisition_cap=DOSE,continuation=DOSE,period=128,gamma=.98,
         planned_worker_rss_mib=2100,planned_worker_cpu_seconds=800,
         gates=dict(taught=.95,early=.50,rise=.20,over_none=.10,over_smooth=.05)))


def advance(m,opt,batches,first,last,arm):
    for i in range(first+1,last+1):
        clock_update(m,opt,batches[i-1],arm)
        if arm=="smooth": m.shrink(.98**(1/128))
        elif arm=="periodic" and i%128==0: m.shrink(.98)


def run(archive,seed):
    verify(ROOT,Path(__file__).name)
    assert seed in SEEDS
    out=ROOT/str(seed)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"COMPLETE.json").exists():return
    sensor=Sensory(archive)
    m,opt=model(seed,sensor)
    train,monitor,_=panels(seed)
    from itertools import islice
    batches=list(islice(stream(train,seed),2*DOSE))
    start=time.process_time()
    path=out/"warm-resume.pt"
    cursor,cpu=restore(path,m,opt) if path.exists() else (0,0)
    rows=[read(out/f"warm-{i}.json") for i in range(BLOCK,cursor+1,BLOCK)]
    if len(rows)<2 or any(r["taught"]["exact"]/len(train)<.95 for r in rows[-2:]):
        for stop in range(cursor+BLOCK,DOSE+1,BLOCK):
            advance(m,opt,batches,stop-BLOCK,stop,"none")
            a=score(m,train)
            b=score(m,monitor)
            row=dict(step=stop,taught=a,monitor=b)
            write(out/f"warm-{stop}.json",row)
            checkpoint(path,m,opt,stop,cpu+time.process_time()-start)
            rows.append(row)
            print(seed,"warm",stop,a["exact"],b["exact"],flush=True)
            if len(rows)>=2 and all(r["taught"]["exact"]/len(train)>=.95 for r in rows[-2:]):break
    at=rows[-1]["step"]
    if len(rows)<2 or any(r["taught"]["exact"]/len(train)<.95 for r in rows[-2:]):
        status="ACQUISITION_GATE_FAILED"
    elif rows[-1]["monitor"]["exact"]/len(monitor)>.50:
        status="EARLY_TRANSFER"
    else:
        status="BRANCHES_COMPLETE"
        if not (out/"fork.pt").exists():checkpoint(out/"fork.pt",m,opt,at)
        for arm in ARMS:
            m,opt=model(seed,sensor)
            restore(out/"fork.pt",m,opt)
            rp=out/f"{arm}-resume.pt"
            offset,_=restore(rp,m,opt) if rp.exists() else (0,0)
            for stop in range(offset+BLOCK,DOSE+1,BLOCK):
                advance(m,opt,batches[at:],stop-BLOCK,stop,arm)
                taught=score(m,train)
                result=dict(step=stop,taught=taught,monitor=score(m,monitor),
                            norm=float(m.shared[0].weight.detach().norm()),state=digest(m))
                write(out/f"{arm}-{stop}.json",result)
                checkpoint(rp,m,opt,stop)
                print(seed,arm,stop,taught["exact"],result["monitor"]["exact"],flush=True)
            if not (out/f"{arm}-final.pt").exists():rp.replace(out/f"{arm}-final.pt")
    write(out/"COMPLETE.json",dict(status=status,fork_step=at,warmup_updates=at,
         branch_updates=DOSE*len(ARMS) if status=="BRANCHES_COMPLETE" else 0,
         **usage(start),prior_cpu_seconds=cpu))
    path.unlink(missing_ok=True)


def sealed(archive):
    verify(ROOT,Path(__file__).name)
    results={}
    for seed in SEEDS:
        out=ROOT/str(seed)
        state=read(out/"COMPLETE.json")
        if state["status"]!="BRANCHES_COMPLETE":continue
        _,_,panel=panels(seed)
        sensor=Sensory(archive)
        m,opt=model(seed,sensor)
        for arm in ("fork",*ARMS):
            restore(out/("fork.pt" if arm=="fork" else f"{arm}-final.pt"),m,opt)
            write(out/f"sealed-{arm}.json",score(m,panel,raw=True))
        s={arm:read(out/f"sealed-{arm}.json")["exact"]/len(panel) for arm in ("fork",*ARMS)}
        taught=read(out/f"periodic-{DOSE}.json")["taught"]
        results[seed]=dict(exact=s,delayed=s["periodic"]-s["fork"]>=.20 and
                       s["periodic"]-s["none"]>=.10 and taught["exact"]/taught["n"]>=.95,
                       timing=s["periodic"]-s["smooth"]>=.05)
    write(ROOT/"SEALED_DECISION.json",dict(seeds=results,
          launch_sham=len(results)==3 and all(r["delayed"] for r in results.values())))


def sham(archive,seed):
    verify(ROOT,Path(__file__).name)
    assert read(ROOT/"SEALED_DECISION.json")["launch_sham"]
    out=ROOT/str(seed)
    at=read(out/"COMPLETE.json")["fork_step"]
    start=time.process_time()
    train,_,held=panels(seed)
    m,opt=model(seed,Sensory(archive))
    from itertools import islice
    batches=list(islice(stream(train,seed,True),at+DOSE))
    advance(m,opt,batches,0,at,"none")
    advance(m,opt,batches[at:],0,DOSE,"periodic")
    write(out/"sealed-sham.json",dict(score=score(m,held,raw=True),updates=at+DOSE,**usage(start)))


if __name__=="__main__":
    action,archive=sys.argv[1:3]
    a=Path(archive)
    if action=="prepare":prepare(a)
    elif action=="run":run(a,int(sys.argv[3]))
    elif action=="sealed":sealed(a)
    else:sham(a,int(sys.argv[3]))
