"""Locked topology and teaching comparison."""
from pathlib import Path
from random import Random
import sys
import time

from campaign_common import (RUN,SEEDS,checkpoint,lock,measures,read,responses,restore,
                             usage,verify,write)
from fixtures import arithmetic,counting
from models import Attractor,Sensory,T,digest
from qualify_remaining import clock_update,PAIR,HELD

ROOT=RUN/"attractor"
ARMS=("answer","shuffled","worked")
CHECKS=(2048,8192,16384)
CAP=5


def model(seed,mode,sensor):
    T.manual_seed(seed)
    m=Attractor(sensor,mode=="rewired")
    return m,T.optim.Adam(m.parameters(),lr=.003)


def score(m,pairs,wrapper=False):
    rows=[arithmetic(p) for p in pairs]
    prompts=[r[:6] if not wrapper else [*r[:2],32,*r[2:5],32,r[5]] for r in rows]
    targets=[r[6:] for r in rows]
    outputs=responses(m,prompts,CAP)
    data=measures(prompts,targets,outputs)
    pairscores=dict(zip(pairs,[x==y for x,y in zip(outputs,targets)]))
    data.update(both_orders=sum(pairscores[a,b] and pairscores[b,a] for a,b in pairs if a<b),
                both_order_n=sum(a<b for a,b in pairs),
                low_sum=[sum(pairscores[a,b] for a,b in pairs if a+b<=13),sum(a+b<=13 for a,b in pairs)],
                high_sum=[sum(pairscores[a,b] for a,b in pairs if a+b>=14),sum(a+b>=14 for a,b in pairs)])
    return data


def prepare(archive):
    lock(ROOT,Path(__file__).name,archive,dict(checkpoints=CHECKS,cap=CAP,
         arms=ARMS,groups_per_history=16384,groups_main=294912,actual_symbols_per_group=58,
         planned_worker_rss_mib=1800,planned_worker_cpu_seconds=600,
         gates=dict(taught=60,held=20,advantage=5)))


def run(archive,seed,mode,arm):
    verify(ROOT,Path(__file__).name)
    assert seed in SEEDS and mode in ("ring","rewired") and arm in ARMS
    out=ROOT/f"{seed}-{mode}-{arm}"
    out.mkdir(parents=True,exist_ok=True)
    if (out/"COMPLETE.json").exists(): return
    sensor=Sensory(archive)
    m,opt=model(seed,mode,sensor)
    start=time.process_time()
    bank=list(PAIR)
    Random(seed).shuffle(bank)
    partial=out/"resume.pt"
    offset,cpu=restore(partial,m,opt) if partial.exists() else (0,0)
    for i,row in enumerate(counting(bank,arm,CHECKS[-1]),1):
        if i<=offset: continue
        clock_update(m,opt,row,arm)
        if i%512==0:
            if i in CHECKS:
                scores=dict(dose=i,state=digest(m),taught=score(m,PAIR),held=score(m,HELD),
                            dynamics=m.persists())
                checkpoint(out/f"A-{i}.pt",m,opt,i,cpu+time.process_time()-start)
                write(out/f"A-{i}.json",scores)
            checkpoint(partial,m,opt,i,cpu+time.process_time()-start)
            print(seed,mode,arm,i,flush=True)
    write(out/"COMPLETE.json",dict(groups=CHECKS[-1],**usage(start),prior_cpu_seconds=cpu))
    partial.unlink(missing_ok=True)


if __name__=="__main__":
    action,archive=sys.argv[1:3]
    if action=="prepare":prepare(Path(archive))
    else:run(Path(archive),int(sys.argv[3]),*sys.argv[4:6])
