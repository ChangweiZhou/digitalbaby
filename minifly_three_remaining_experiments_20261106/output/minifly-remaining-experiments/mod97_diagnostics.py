"""Read-only shortcut and pathway diagnostics for frozen modular checkpoints."""
from collections import Counter
from pathlib import Path
import sys

from campaign_common import SEEDS,read,restore,verify,write
from models import Sensory,T,digest
from official_mod97 import ROOT,model,panels,score


def shortcuts(train,held):
    taught={tuple(sorted(p)):sum(p)%97 for p in train}
    frequency=Counter(sum(p)%97 for p in train).most_common(1)[0][0]
    targets=[sum(p)%97 for p in held]
    ordered=sorted(taught)
    nearest=[taught[min(ordered,key=lambda q:(abs(q[0]-min(p))+abs(q[1]-max(p)),q))] for p in held]
    predictions={"frequency":[frequency]*len(held),"nearest_unordered":nearest,
                 "no_wrap":[sum(p) for p in held],
                 "no_decimal_carry":[(10*(a//10+b//10)+(a%10+b%10)%10)%97 for a,b in held]}
    assert all(tuple(sorted(p)) not in taught for p in held)
    return dict(n=len(held),exact_lookup=0,canonical_lookup=0,oracle=len(held),
                **{k:sum(y==t for y,t in zip(v,targets)) for k,v in predictions.items()})


@T.no_grad()
def masks(m,held):
    before=digest(m)
    result={}
    for label,parameters in (("shared_only",[m.association.weight]),
                             ("association_only",[m.shared[-1].weight,m.shared[-1].bias])):
        saved=[p.clone() for p in parameters]
        try:
            for p in parameters:p.zero_()
            result[label]=score(m,held)
        finally:
            for p,value in zip(parameters,saved):p.copy_(value)
    assert digest(m)==before
    return result


def main(archive):
    verify(ROOT,"official_mod97.py")
    report={}
    for seed in SEEDS:
        out=ROOT/str(seed)
        state=read(out/"COMPLETE.json")
        train,_,held=panels(seed)
        report[str(seed)]={"shortcut_scores":shortcuts(train,held)}
        if state["status"]=="ACQUISITION_GATE_FAILED":continue
        if state["status"]=="EARLY_TRANSFER":out=out/"early-branches"
        m,opt=model(seed,Sensory(archive))
        for arm in ("fork","none","periodic","smooth"):
            restore(out/("fork.pt" if arm=="fork" else f"{arm}-final.pt"),m,opt)
            report[str(seed)][arm]=masks(m,held)
            print(seed,arm,"masked",flush=True)
    write(ROOT/"PATH_AND_SHORTCUT_AUDIT.json",report)


if __name__=="__main__":
    main(Path(sys.argv[1]))
