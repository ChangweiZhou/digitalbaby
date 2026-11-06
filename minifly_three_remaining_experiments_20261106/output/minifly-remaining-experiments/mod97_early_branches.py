"""Complete the registered shrinkage contrasts on early-transfer seeds.

These seeds are excluded from the delayed-rise decision, independent of branch outcomes.
"""
from itertools import islice
from pathlib import Path
import sys
import time

from campaign_common import SEEDS,checkpoint,read,restore,sha,usage,verify,write
from models import Sensory,digest
from official_mod97 import ARMS,BLOCK,DOSE,ROOT,advance,model,panels,score,stream


def run(archive,seed):
    verify(ROOT,"official_mod97.py")
    original=ROOT/str(seed)
    r=read(original/"COMPLETE.json")
    if r["status"]!="EARLY_TRANSFER":return
    out=original/"early-branches"
    out.mkdir(exist_ok=True)
    if (out/"COMPLETE.json").exists():return
    start=time.process_time()
    train,monitor,_=panels(seed)
    at=r["fork_step"]
    sensor=Sensory(archive)
    m,opt=model(seed,sensor)
    checkpoint_path=out/"fork.pt"
    batches=list(islice(stream(train,seed),at+DOSE))
    if not checkpoint_path.exists():
        backup=original/"recovery-warm.pt"
        offset,_=restore(backup,m,opt) if backup.exists() else (0,0)
        assert offset<=at
        advance(m,opt,batches,offset,at,"none")
        warm=read(original/f"warm-{at}.json")
        assert score(m,train)==warm["taught"] and score(m,monitor)==warm["monitor"]
        checkpoint(checkpoint_path,m,opt,at)
    for arm in ARMS:
        final=out/f"{arm}-final.pt"
        if final.exists():continue
        m,opt=model(seed,sensor)
        restore(checkpoint_path,m,opt)
        partial=out/f"{arm}-resume.pt"
        offset,_=restore(partial,m,opt) if partial.exists() else (0,0)
        for stop in range(offset+BLOCK,DOSE+1,BLOCK):
            advance(m,opt,batches[at:],stop-BLOCK,stop,arm)
            taught=score(m,train)
            result=dict(step=stop,taught=taught,monitor=score(m,monitor),
                        norm=float(m.shared[0].weight.detach().norm()),state=digest(m))
            write(out/f"{arm}-{stop}.json",result)
            checkpoint(partial,m,opt,stop)
            print(seed,arm,stop,taught["exact"],result["monitor"]["exact"],flush=True)
        partial.replace(final)
    write(out/"COMPLETE.json",dict(fork_step=at,updates=3*DOSE,eligible_for_delayed_rise=False,
              runner_sha256=sha(Path(__file__)),**usage(start)))


def sealed(archive):
    verify(ROOT,"official_mod97.py")
    # Open the sealed panels only once all seeds are frozen.
    for seed in SEEDS:
        orig=ROOT/str(seed)
        if read(orig/"COMPLETE.json")["status"]!="EARLY_TRANSFER":continue
        out=orig/"early-branches"
        assert read(out/"COMPLETE.json")["eligible_for_delayed_rise"] is False
    for seed in SEEDS:
        orig=ROOT/str(seed)
        if read(orig/"COMPLETE.json")["status"]!="EARLY_TRANSFER":continue
        out=orig/"early-branches"
        _,_,held=panels(seed)
        m,opt=model(seed,Sensory(archive))
        for arm in ("fork",*ARMS):
            restore(out/("fork.pt" if arm=="fork" else f"{arm}-final.pt"),m,opt)
            write(out/f"sealed-{arm}.json",score(m,held,raw=True))


if __name__=="__main__":
    a=Path(sys.argv[2])
    if sys.argv[1]=="sealed":sealed(a)
    else:run(a,int(sys.argv[3]))
