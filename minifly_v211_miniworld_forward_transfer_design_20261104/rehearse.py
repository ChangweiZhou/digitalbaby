"""Driver qualification with a canonical witness. No learner training."""
import hashlib
import json
from pathlib import Path
import statistics as st
from zipfile import ZipFile, ZIP_DEFLATED

from audit import answer, audit, observe
from world import life,seed

ARMS=("PRIOR-TRUE","PRIOR-TYPE-SHAM","PRIOR-OBSERVE","FRESH")
OUT=Path("rehearsal")

class Witness:
    def __init__(self):
        self.state={}
        self.calls=[]
    def observe(self,raw):
        observe(raw.decode(),self.state)
        self.calls.append("observe")
    def choose(self,raw):
        self.calls.append("choose")
        return answer(raw.decode(),self.state)
    def feedback(self,raw):
        assert isinstance(raw,bytes)
        self.calls.append("feedback")

def drive(worlds,arm,backend):
    trace=[]
    for i,packets in enumerate(worlds):
        target=i==len(worlds)-1
        if not target and arm=="FRESH":
            continue
        for p in packets:
            for e in p.events:
                backend.observe(e.encode())
            choices=[backend.choose(q.encode()) for q in p.questions]
            labels=()
            if target or arm!="PRIOR-OBSERVE":
                labels=p.answers[::-1] if p.swap and arm=="PRIOR-TYPE-SHAM" and not target else p.answers
                for y in labels:
                    backend.feedback(y.encode())
            trace.append(dict(target=target,events=p.events,questions=p.questions,selected=choices,
                              feedback=labels,correct=[x==y for x,y in zip(choices,p.answers)]))
    return trace

def reduce(lives):
    assert len(lives)==16 and len({r["id"] for r in lives})==16
    assert all(set(r["scores"])==set(ARMS) and all(len(x)==64 for x in r["scores"].values()) for r in lives)
    def effect(arm,start,end):
        vals=[st.mean(r["scores"]["PRIOR-TRUE"][start:end])-st.mean(r["scores"][arm][start:end])
              for r in lives]
        center=st.mean(vals)
        lower=center-2.13144955*st.stdev(vals)/4
        return center,lower
    fresh=effect("FRESH",0,32)
    matched=[effect(a,0,32) for a in ARMS[1:3]]
    late=effect("FRESH",48,64)
    return (fresh[0]>=.10 and fresh[1]>0 and
            all(x[0]>=.05 and x[1]>0 for x in matched) and late[1]>-.05)

def mock(sham=False, late=False):
    scores={a:[0]*48+[1]*16 for a in ARMS}
    scores["PRIOR-TRUE"]=[1]*48+[int(not late)]*16
    if sham:
        scores["PRIOR-TYPE-SHAM"]=[1]*64
    return [dict(id=i,scores=scores) for i in range(16)]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(name,obj):
    p=OUT/name
    p.write_text(json.dumps(obj,indent=2)+"\n")
    return sha(p)

def main():
    worlds=life("V211-TECH-WITNESS",past=2,target_packets=4)
    audit(worlds)
    assert reduce(mock()) and not reduce(mock(sham=True)) and not reduce(mock(late=True))
    try:
        reduce(mock()[:-1])
    except AssertionError:
        pass
    else:
        raise AssertionError("incomplete roster accepted")
    OUT.mkdir(exist_ok=False)
    hashes={}
    target_ref=None
    for arm in ARMS:
        backend=Witness()
        r=drive(worlds,arm,backend)
        expected=[]
        for p in r:
            assert all(p["correct"])
            expected+=["observe"]*len(p["events"])+["choose"]*2+["feedback"]*len(p["feedback"])
            assert not p["feedback"] or set(p["feedback"])==set(p["selected"])
        assert backend.calls==expected
        target=[p for p in r if p["target"]]
        target_ref=target_ref or target
        assert target==target_ref and len(target)==4
        assert len(r)==(4 if arm=="FRESH" else 20)
        name=f"{arm}_WITNESS.json"
        hashes[name]=write(name,r)
    dry=[]
    for i in range(16):
        s=seed(f"MINIFLY|V211|LIFE|{i}")
        schedule=life(s)
        audit(schedule)
        dry.append(dict(seed=s,sha256=hashlib.sha256(repr(schedule).encode()).hexdigest()))
    write("CYCLE3_TEST.json",dict(status="DRIVER_PASS_PARENT_UNQUALIFIED",science_worlds_run=0,
          learner_training_questions=0,dry_roster=dry,receipts=hashes,
          sources={p:sha(Path(p)) for p in ("DESIGN.md","parent_gate.py","world.py","audit.py","rehearse.py")}))
    files=sorted(OUT.glob("*.json"))
    with ZipFile(OUT/"TECHNICAL_BUNDLE.zip","w",ZIP_DEFLATED) as z:
        for p in files:
            z.write(p,p.name)
    with ZipFile(OUT/"TECHNICAL_BUNDLE.zip") as z:
        assert z.testzip() is None and all(z.read(p.name)==p.read_bytes() for p in files)
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in hashes.items())

if __name__=="__main__":
    main()
