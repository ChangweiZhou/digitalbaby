"""Run the fixed V200 roster after a budgeted launch profile."""
import json
import math
import os
import random
import statistics as stats
import time
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from learner import ARMS, EOS, Learner, band, chain, exact_chain, rows
from rehearse import FORMS, census, decision, digest, rate, roster, state
import resource

OUT=Path("official")
START=time.monotonic()
CARRY_CPU=60
CARRY_WALL=120
FILLER=(b"we said so. ",b"it was on the way. ",b"and then they ran. ")
INPUTS=("DESIGN.md","OFFICIAL_PROTOCOL.md","learner.py","rehearse.py","official.py",
        "CYCLE1_TEST.json","CYCLE2_TEST.json","rehearsal/CYCLE3_TEST.json")

def save(name,data):
    p=OUT/name
    tmp=p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    os.replace(tmp,p)

def seed(s):
    return int(digest(s.encode())[:16],16)

def limits():
    cost=dict(cpu=time.process_time()+CARRY_CPU,wall=time.monotonic()-START+CARRY_WALL,
              rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              output=sum(p.stat().st_size for p in OUT.iterdir()))
    assert cost["cpu"]<=900 and cost["wall"]<=1800 and cost["rss"]<=2**30
    assert cost["output"]<=256*2**20
    return cost

def stream(cohorts,world):
    phases=[]
    filler=random.Random(seed(f"{world}|filler"))
    for index,cohort in enumerate(cohorts):
        order_rng=random.Random(seed(f"{world}|order|{index}"))
        links=[link for a,b,d in cohort for link in ((a,b),(b,d))]
        records=[]
        for rep in range(12):
            order=links.copy()
            order_rng.shuffle(order)
            lr=(.1,.3)[rep] if index==0 and rep<2 else .6
            records.extend((filler.choice(FILLER),FORMS[rep%2](k),y,lr) for k,y in order)
        phases.append(records)
    return phases,[filler.choice(FILLER) for _ in range(256)]

def teach(m, records, exact):
    for _,query,answer,lr in records:
        m.teach(query,answer,lr)
        exact[query]=answer

def panel(m, originals, exact):
    before=state(m)
    canonical={k:exact[FORMS[0](k)] for a,b,_ in originals for k in (a,b)}
    digits=[v for v in exact.values() if len(v)==1]
    modal=max(set(digits),key=lambda v:(digits.count(v),v))
    buckets={}
    for key in canonical:
        v=canonical[key]
        if len(v)==1:
            buckets.setdefault((len(key),key[:1]),[]).append(v)
    results=[]
    for w,form in enumerate(FORMS):
        for i,(a,b,d) in enumerate(originals):
            q=form(a)
            first,second=chain(m.generate,form,a)
            gold=(*b,EOS)
            trace=[int(m.score(q,b[:t])[1].argmax()) for t in range(len(gold))]
            oracle=m.generate(form(b))
            if w>=2:
                assert q not in exact and form(b) not in exact
            train=set(rows(m.arm,FORMS[0](a),b""))
            test=set(rows(m.arm,q,b""))
            frequent=buckets.get((len(b),b[:1]),[modal])
            results.append(dict(item=i,wrapper=w,length=len(b),first=first[0].hex(),
                eos=first[1],second=None if second is None else [second[0].hex(),second[1]],
                hop1=first==(b,True),strict=exact_chain(first,second,b,d),prefix=first[0].startswith(b),
                gold_stop=trace[-1]==EOS,gold_second=oracle==(d,True),
                gold_position=[g==v for g,v in zip(trace,gold)],
                first_error=next((t for t,g in enumerate(gold)
                                  if t>=len(first[0])+int(first[1]) or
                                  g!=(first[0][t] if t<len(first[0]) else EOS)),None),
                overlap=len(train&test),union=len(train|test),same_address=train==test,
                canonical=canonical[canonical[a]]==d,
                exact_table=exact.get(q)==b and exact.get(form(b))==d,
                modal=modal==d,position_shortcut=max(set(frequent),key=lambda v:(frequent.count(v),v))==d,
                wrong_intermediate_correct_digit=first!=(b,True) and second==(d,True)))
    assert before==state(m)
    return results

def profile(roster_data):
    # No training records. Measure cold scoring, synthetic row arithmetic and audit hashing separately.
    import numpy as np
    timings=[]
    for arm in ARMS:
        band.cache_clear()
        m=Learner(arm)
        a_rows=roster_data[0]
        start=time.process_time()
        calls=0
        for form in FORMS:
            for a,b,_ in a_rows:
                for t in range(len(b)+1):
                    m.score(form(a),b[:t])
                    calls+=1
        score_cost=(time.process_time()-start)/calls
        ix=np.arange(48)
        error=np.zeros(257,np.float32)
        error[256]=1
        start=time.process_time()
        for _ in range(4096):
            delta=.6*error[None,:]/(48*(1+m.importance[ix,None]/5))
            m.weights[ix]+=delta.astype(np.float32)
            m.importance[ix]+=1/48
        arithmetic=(time.process_time()-start)/4096
        start=time.process_time()
        state(m)
        audit_cost=time.process_time()-start
        assert m.weights.nbytes+m.importance.nbytes+3*2**20<=72*2**20
        timings.append(dict(arm=arm,score=score_cost,update=arithmetic,audit=audit_cost,read_only_calls=calls))
    return timings

def summarize(worlds):
    paired={}
    for base in ARMS[:2]:
        delta=[rate(w["SPLIT-HOLD"],"hop1",True)-rate(w[base],"hop1",True) for w in worlds]
        m=stats.mean(delta)
        half=2.364624*stats.stdev(delta)/math.sqrt(8)
        paired[base]=dict(mean=m,interval=[m-half,m+half],worlds=delta)
    scores={arm:{key:[stats.mean(rate(w[arm],key,held) for w in worlds) for held in (False,True)]
                  for key in ("hop1","strict","prefix","gold_stop","gold_second")} for arm in ARMS}
    return scores,paired

def main():
    assert os.environ["OPENBLAS_NUM_THREADS"]=="1"
    previous=json.loads(Path("rehearsal/CYCLE3_TEST.json").read_text())
    for path,sha in previous["sources"].items():
        assert digest(Path(path).read_bytes())==sha
    seeds=[seed(f"MINIFLY|V200|WORLD|{i}") for i in range(8)]
    rosters=[roster(s,128,32) for s in seeds]
    assert [digest(repr(w).encode()) for w in rosters]==previous["dry_roster"]
    schedules=[stream(w,s) for w,s in zip(rosters,seeds)]
    marked=sum(census(w)[0] for w in rosters)*3
    targets=sum(census(w)[1] for w in rosters)*3
    filler_count=sum(sum(map(len,p))+len(d) for p,d in schedules)*3
    filler_bytes=sum(sum(len(f) for phase in p for f,*_ in phase)+sum(map(len,d)) for p,d in schedules)*3
    probe_cap=8*3*3*4*128*32
    assert marked==92160 and targets==366912
    assert previous["charged_records"]+120==105000
    assert previous["charged_targets"]+480<=600000 and probe_cap<4000000
    save("LOCK.json",dict(seeds=seeds,sources={p:digest(Path(p).read_bytes()) for p in INPUTS},
        schedule_hashes=[digest(repr(s).encode()) for s in schedules],marked=marked,targets=targets,
        filler_records=filler_count,filler_bytes=filler_bytes,probe_upper=probe_cap))
    prior=Path("preflight_projection_stop")
    assert json.loads((prior/"FAILURE.json").read_text())["error"]=="AssertionError()"
    assert not (prior/"COMMIT_INDEX.json").exists()
    timings=profile(rosters[0])
    projection=(max(t["score"] for t in timings)*(targets+probe_cap)
                +max(t["update"] for t in timings)*targets
                +max(t["audit"] for t in timings)*24*7)*3+60
    save("ADMISSION.json",dict(status="PASS" if projection+limits()["cpu"]<900 else "STOP",
        prior_technical_presentations=120,prior_technical_targets=480,additional_presentations=0,
        read_only_profile=timings,projected_cpu=projection,cost=limits(),
        first_attempt_receipt_sha=digest((prior/"LOCK.json").read_bytes())))
    assert projection+limits()["cpu"]<900
    commits=[]
    for i,((originals,new),(phases,delay)) in enumerate(zip(rosters,schedules)):
        for arm in ARMS:
            band.cache_clear()
            m=Learner(arm)
            exact={}
            teach(m,phases[0],exact)
            stages=[panel(m,originals,exact)]
            assert len(delay)==256
            stages.append(panel(m,originals,exact))
            assert stages[0]==stages[1]
            teach(m,phases[1],exact)
            stages.append(panel(m,originals,exact))
            assert m.sites==census((originals,new))[1]
            limits()
            name=f"{i}_{arm}_RECEIPT.json"
            save(name,dict(world=i,arm=arm,stages=stages,state=state(m),sites=m.sites,
                presentations=sum(map(len,phases)),probe_calls=m.decisions-m.sites,
                rows_read=m.read_rows,rows_written=m.write_rows))
            commits.append(dict(file=name,sha=digest((OUT/name).read_bytes())))
            save("COMMIT_INDEX.json",commits)
            print("committed",name,flush=True)
    worlds=[{} for _ in seeds]
    receipts=[]
    for row in commits:
        raw=(OUT/row["file"]).read_bytes()
        assert digest(raw)==row["sha"]
        r=json.loads(raw)
        assert r["arm"] not in worlds[r["world"]]
        worlds[r["world"]][r["arm"]]=r["stages"][-1]
        receipts.append(r)
    assert all(set(w)==set(ARMS) and all(len(p)==512 for p in w.values()) for w in worlds)
    assert sum(r["presentations"] for r in receipts)==marked
    assert sum(r["sites"] for r in receipts)==targets
    probes=sum(r["probe_calls"] for r in receipts)
    assert probes<=probe_cap
    scores,paired=summarize(worlds)
    status="ADVANCE" if decision(worlds) else "DID_NOT_ADVANCE"
    save("SUMMARY.json",dict(status=status,scores_trained_held=scores,paired_hold_minus=paired,
        lives=len(receipts),marked=marked,targets=targets,probe_calls=probes,cost=limits()))
    (OUT/"REPORT.md").write_text(f"# V200 official result\n\n**{status}** · E2 bounded changed-wrapper development.\n\n"
        f"Eight paired worlds, {len(receipts)} accepted arm lives, {marked:,} marked presentations, "
        f"{targets:,} supervised byte/EOS targets and {probes:,} read-only decisions.\n\n"
        "| Arm | Trained strict chain | Held exact hop one + EOS | Held strict chain |\n|---|---:|---:|---:|\n"+
        "".join(f"| {a} | {s['strict'][0]:.2%} | {s['hop1'][1]:.2%} | {s['strict'][1]:.2%} |\n" for a,s in scores.items())+
        "\nThe controller is fixed. See SUMMARY.json for paired contrasts.\n")
    files=sorted(p for p in OUT.iterdir() if p.suffix in (".json",".md"))
    with ZipFile(OUT/"RESULT_BUNDLE.zip","w",ZIP_DEFLATED) as z:
        for p in files:
            z.write(p,p.name)
    with ZipFile(OUT/"RESULT_BUNDLE.zip") as z:
        assert z.testzip() is None and all(z.read(p.name)==p.read_bytes() for p in files)
    save("PACKAGE.json",dict(status="COMPLETE",sha=digest((OUT/"RESULT_BUNDLE.zip").read_bytes()),cost=limits()))

if __name__=="__main__":
    OUT.mkdir(exist_ok=False)
    try:
        main()
    except Exception as e:
        save("FAILURE.json",dict(error=repr(e)))
        raise
