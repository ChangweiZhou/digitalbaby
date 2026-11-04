"""V213 full-path technical rehearsal and dry census."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import resource
import statistics as st
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from learner import ARMS,Learner
from train import fit
from world import audit,generate,official_roster,oracle,seed

OUT=Path("rehearsal")
REFS=("PERSIST","FOLLOW","RECENT","EXACT-PAIR","POOLED","ORACLE")

def sha(b):
    return hashlib.sha256(b).hexdigest()

def state(model):
    return sha(model.weights.tobytes()+model.theta.tobytes())

def keys(prefix):
    o,a,b=next(r[1:] for r in reversed(prefix) if r[0]=="event")
    last=next(r[2] for r in reversed(prefix) if r[0]=="obs" and r[1]==b)
    z=prefix[-1][2]
    return last,z,(prefix[-3:],(a,b,o,last,z),(o,last,z))

def references(w,packets):
    tables=[defaultdict(Counter) for _ in range(3)]
    for records in w.training[:packets]:
        for cut in (6,11):
            _,_,xs=keys(records[:cut])
            for table,key in zip(tables,xs):
                table[key][records[cut][2]]+=1
    return tables

def predict_ref(w,tables,ref,prefix):
    last,z,xs=keys(prefix)
    if ref in ("PERSIST","FOLLOW","ORACLE"):
        y={"PERSIST":last,"FOLLOW":z,"ORACLE":oracle(prefix,w.attach)}[ref]
        return np.array([float(p==y) for p in w.positions])
    i=("RECENT","EXACT-PAIR","POOLED").index(ref)
    counts=tables[i].get(xs[i],{})
    values=np.array([1+counts.get(pos,0) for pos in w.positions],float)
    return values/values.sum()

def strict(p,target):
    return int(p[target]==max(p) and sum(p==p[target])==1)

def row(w,name,predict):
    groups=defaultdict(list)
    predictions=[]
    for panel,earlier,branches in w.twins:
        correct=[]
        for prefix,y in branches:
            p=np.asarray(predict(prefix))
            assert p.shape==(4,) and np.isfinite(p).all() and np.all(p>=0)
            assert abs(float(p.sum())-1)<1e-7
            t=w.positions.index(y)
            correct.append(strict(p,t))
            predictions.append(dict(panel=panel,earlier=earlier,target=y,prob=list(map(float,p)),
                                    strict=correct[-1]))
        value=int(all(correct))
        groups[panel].append(value)
        groups[f"{panel}_{'linked' if earlier==w.attach else 'independent'}"].append(value)
    scores={k:st.mean(v) for k,v in groups.items()}
    ll=-st.mean(math.log(max(1e-8,p["prob"][w.positions.index(p["target"])])) for p in predictions)
    return dict(arm=name,scores=scores,mean_log_loss=ll,predictions=predictions)

def world_run(w,packets):
    audit(w)
    tables=references(w,packets)
    results={}
    for arm in ARMS:
        m=Learner(w.positions,w.opcodes,w.key,arm)
        order=fit(m,w,packets)
        expected=[sum(pair.count(o) for pair in order) for o in w.opcodes]
        assert m.gate_updates==(expected if arm in ARMS[:2] else [0,0])
        assert m.emission_updates==(0 if arm=="NO-PRIOR-WRITE" else packets*2)
        before=state(m)
        def predict(prefix):
            m.reset()
            for r in prefix:
                p=m.consume(r)
            return p
        r=row(w,arm,predict)
        assert state(m)==before
        results[arm]={**r,"state_sha256":before,"emission_updates":m.emission_updates,
                      "gate_updates":m.gate_updates}
    for a in REFS:
        results[a]=row(w,a,lambda prefix:predict_ref(w,tables,a,prefix))
    assert results["ORACLE"]["scores"]["held"]==1
    assert all(results[a]["scores"]["held"]==0 for a in ("PERSIST","FOLLOW","RECENT","EXACT-PAIR"))
    return dict(world=w.key,training_packets=packets,rows=results)

def decision(worlds):
    assert len(worlds)==len({w["world"] for w in worlds})==8
    assert all(set(w["rows"])==set(ARMS+REFS) for w in worlds)
    def avg(arm,key):
        return st.mean(w["rows"][arm]["scores"][key] for w in worlds)
    # 1 - 0.05/3 one-sided t quantile, 7 df.
    t=2.641859673
    effects={}
    for arm in ARMS[1:]:
        d=[w["rows"]["ALIGNED"]["scores"]["held"]-w["rows"][arm]["scores"]["held"] for w in worlds]
        se=st.stdev(d)/math.sqrt(8)
        effects[arm]=dict(mean=st.mean(d),lower=st.mean(d)-t*se,
                          ci95=[st.mean(d)-2.364624251*se,st.mean(d)+2.364624251*se])
    guards=dict(held=avg("ALIGNED","held")>=.75,taught=avg("ALIGNED","taught")>=.80,
        strata=all(avg("ALIGNED",k)>=.65 for k in ("held_linked","held_independent")),
        contrast=all(x["mean"]>=.15 and x["lower"]>0 for x in effects.values()),
        shortcut=all(avg("ALIGNED","held")>avg(a,"held") for a in ("PERSIST","RECENT")),
        oracle=avg("ORACLE","held")==1)
    return dict(passed=all(guards.values()),guards=guards,effects=effects)

def reducer_tests():
    def fixture(sham):
        return [dict(world=i,rows={a:dict(scores={k:(1 if a in ("ALIGNED","ORACLE") else sham if a=="SHUFFLED-OP" else 0)
            for k in ("held","taught","held_linked","held_independent")}) for a in ARMS+REFS}) for i in range(8)]
    assert decision(fixture(0))["passed"] and not decision(fixture(1))["passed"]
    try:
        decision(fixture(0)[:-1])
    except AssertionError:
        pass
    else:
        raise AssertionError("incomplete roster accepted")

def save(name,obj):
    p=OUT/name
    p.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
    return sha(p.read_bytes())

def main():
    start=time.process_time()
    wall=time.monotonic()
    reducer_tests()
    OUT.mkdir(exist_ok=False)
    dry=official_roster()
    lock=[]
    for w in dry:
        audit(w)
        lock.append(dict(seed=w.key,sha256=sha(repr(w).encode())))
    sources={p:sha(Path(p).read_bytes()) for p in ("DESIGN.md","world.py","learner.py","train.py",
                                                  "test_world.py","test_learning.py","rehearse.py")}
    hashes={}
    panels=[]
    start_live=time.process_time()
    for i,packets in enumerate((48,32)):
        w=generate(seed(f"V213|ENGINEERING|ENTRY|{i}"))
        assert w.key not in {x.key for x in dry}
        r=world_run(w,packets)
        hashes[f"TECH_{i}.json"]=save(f"TECH_{i}.json",r)
        panels.append({a:x["scores"] for a,x in r["rows"].items()})
    cpu=time.process_time()-start_live
    cumulative=dict(records=100352+8*12*4+80*12*7+2*64*7*10,
                    outcomes=10752+64+80*2*7,probes=5120+2*64*10)
    assert cumulative["records"]<=115000 and cumulative["outcomes"]<=12000 and cumulative["probes"]<=8000
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    cpu_estimate=30+cpu*max(8*96/80,4)*3
    assert rss<512*2**20 and cpu_estimate<300
    save("CYCLE3_TEST.json",dict(status="TECHNICAL_COMPLETE",official_worlds_opened=0,
        dry_roster=lock,sources=sources,receipts=hashes,engineering_panels=panels,
        charged_with_official=cumulative,projected_cpu_seconds=cpu_estimate,
        measured_cpu_seconds=time.process_time()-start,measured_wall_seconds=time.monotonic()-wall,
        rss_bytes=rss))
    paths=sorted(OUT.glob("*.json"))
    with ZipFile(OUT/"TECHNICAL_BUNDLE.zip","w",ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p,p.name)
    with ZipFile(OUT/"TECHNICAL_BUNDLE.zip") as z:
        assert z.testzip() is None and all(z.read(p.name)==p.read_bytes() for p in paths)
        assert all(sha(z.read(n))==h for n,h in hashes.items())
    print(json.dumps([{a:(p["held"],p["taught"]) for a,p in panel.items()} for panel in panels]))

if __name__=="__main__":
    main()
