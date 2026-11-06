"""Evaluate final dynamics and apply the counting attribution gates."""
from pathlib import Path
from random import Random
import sys
import time

from campaign_common import RUN,SEEDS,checkpoint,read,restore,sha,usage,verify,write
from fixtures import EOS,counting
from models import Sensory,T,digest,loss,step
from official_attractor import ARMS,CHECKS,HELD,PAIR,model,score

ROOT=RUN/"attractor"


@T.no_grad()
def movement(m):
    h=T.zeros(8,32)
    for i in range(8):h[i,[(4*i+j)%32 for j in range(3)]]=1
    snapshots={}
    def center(values):
        theta=T.arange(32)*2*T.pi/32
        vec=values@T.polar(T.ones_like(theta),theta).real+1j*(values@theta.sin())
        return T.angle(vec)*32/(2*T.pi)
    start=center(h)
    for symbol in ("|","."):
        state=h.clone()
        code=m.book[ord(symbol)].expand(8,8)
        for _ in range(8):state=m.transition(state,m.drive(code),m.gate(code).sigmoid())
        change=(center(state)-start+16)%32-16
        snapshots[symbol]=dict(delta=change.tolist(),norm=state.norm(dim=1).tolist())
    return snapshots


def ablation(m):
    state=digest(m)
    full=score(m,HELD)
    saved=m.head.weight.detach().clone()
    with T.no_grad():m.head.weight.zero_()
    try:muted=score(m,HELD)
    finally:
        with T.no_grad():m.head.weight.copy_(saved)
    assert digest(m)==state
    return dict(full=full["exact"],working_state_muted=muted["exact"],n=len(HELD))


def qualification(archive):
    verify(ROOT,"official_attractor.py")
    eligible=[]
    results={}
    for mode in ("ring","rewired"):
        for arm in ARMS:
            rows=[]
            for seed in SEEDS:
                d=ROOT/f"{seed}-{mode}-{arm}"
                assert read(d/"COMPLETE.json")["groups"]==CHECKS[-1]
                rows.append(read(d/f"A-{CHECKS[-1]}.json"))
            results[f"{mode}-{arm}"]=rows
            if all(r["taught"]["exact"]>=60 and r["held"]["exact"]>=20 for r in rows):
                eligible.append((min(r["held"]["exact"] for r in rows),mode=="rewired",-ARMS.index(arm),mode,arm))
    diagnostics={}
    sensor=Sensory(archive)
    for mode in ("ring","rewired"):
        for arm in ARMS:
            for seed in SEEDS:
                d=ROOT/f"{seed}-{mode}-{arm}"
                m,opt=model(seed,mode,sensor)
                restore(d/f"A-{CHECKS[-1]}.pt",m,opt)
                assert digest(m)==read(d/f"A-{CHECKS[-1]}.json")["state"]
                diagnostics[d.name]=dict(movement=movement(m),ablation=ablation(m))
    selected=max(eligible)[-2:] if eligible else None
    write(ROOT/"DECISION.json",dict(selected=selected,eligible=[list(r[-2:]) for r in eligible],
          diagnostics=diagnostics,runner_sha256=sha(Path(__file__))))
    print("Attractor selected",selected,flush=True)
    return selected


def continue_selected(archive,seed):
    mode,arm=read(ROOT/"DECISION.json")["selected"]
    out=ROOT/f"{seed}-{mode}-{arm}"
    sensor=Sensory(archive)
    m,opt=model(seed,mode,sensor)
    bank=list(PAIR)
    Random(seed).shuffle(bank)
    born=score(m,HELD)
    start=time.process_time()
    for i,row in enumerate(counting(bank,arm,CHECKS[-1]),1):
        step(opt,sum(loss(m,[r],6 if j else None) for j,r in enumerate(row)))
        if i%1024==0:print(seed,"count-only",i,flush=True)
    count_only=score(m,HELD)
    checkpoint(out/"count-only.pt",m,opt,CHECKS[-1])
    restore(out/f"A-{CHECKS[-1]}.pt",m,opt)
    rng=Random(seed+11)
    for _ in range(2048):
        k=rng.randrange(26)
        record=[ord("K"),65+k,ord("?"),97+k,EOS]
        step(opt,loss(m,[record]))
    write(out/"CONDITIONAL.json",dict(birth=born,count_only=count_only,
         after_B=score(m,HELD),after_B_fresh_wrapper=score(m,HELD,True),**usage(start)))


if __name__=="__main__":
    a=Path(sys.argv[2])
    if sys.argv[1]=="evaluate":qualification(a)
    else:continue_selected(a,int(sys.argv[3]))
