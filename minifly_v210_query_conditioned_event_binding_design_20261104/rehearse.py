"""V210 bounded engineering comparison. No official worlds are trained."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from assay import ARMS, LEARNED, Selector, choose, normalized, render, roster, v

OUT=Path("rehearsal")
PANEL=(*ARMS,"ANCHOR-NO-VOTES")

def digest(obj):
    return sha256(repr(obj).encode()).hexdigest()

def set_store(model,sentences):
    model.store.clear()
    for s in sentences:
        model.write(s)

def check(scores,target):
    first,tied=choose(scores)
    assert (first,tied)==choose(dict(reversed(tuple(scores.items()))))
    return tied=={target},first==target

def exercise(i):
    train=list(roster(f"V210-RUN-TRAIN|{i}",32,True))
    held=list(roster(f"V210-RUN-HELD|{i}",16,False))
    m=v.Learner("context")
    arms={a:Selector(a) for a in LEARNED}
    prior=Counter()
    # Fit only question frequencies before training the local readouts.
    for rec in train:
        for q in render(rec)[2]:
            m.feedback(q,[],None)
    for j,rec in enumerate(train):
        sentences,cs,qs,ys,swap=render(rec)
        set_store(m,sentences)
        for t,(q,y) in enumerate(zip(qs,ys)):
            prior[next(c.position for c in cs if c.word==y)]+=1
            r=normalized(m,q,cs)
            for a,model in arms.items():
                label=ys[1-t] if swap and a=="SAME-TYPE-SHUFFLE" else y
                model.teach(q,cs,r,label,.1 if j<16 else .3)
    before={a:sha256(m.w.tobytes()).hexdigest() for a,m in arms.items()}
    rows=[]
    for l in (0,40):
        strict=Counter()
        emitted=Counter()
        coverage=0
        for rec in held:
            sentences,cs,qs,ys,_=render(rec,pad=l,training=False)
            set_store(m,sentences)
            pair={a:[] for a in PANEL}
            for q,y in zip(qs,ys):
                coverage+=any(w==y for w,_ in m.recall(q))
                r=normalized(m,q,cs)
                scores={"VOTE":r,"POSITION":{c.word:prior[c.position]/64 for c in cs},
                        "POSITION+VOTE":{c.word:prior[c.position]/64+r[c.word] for c in cs}}
                scores.update({a:{c.word:model.score(q,c)+r[c.word] for c in cs} for a,model in arms.items()})
                scores["ANCHOR-NO-VOTES"]={c.word:arms["QUERY-ANCHOR"].score(q,c) for c in cs}
                for a,s in scores.items():
                    unique,correct=check(s,y)
                    pair[a].append(unique)
                    emitted[a]+=correct
            for a,x in pair.items():
                assert len(x)==2
                strict[a]+=all(x)
        rows.append(dict(padding=l,strict_pairs=dict(strict),emitted_correct=dict(emitted),
                         story_count=16,source_coverage=coverage,source_questions=32))
    assert before=={a:sha256(m.w.tobytes()).hexdigest() for a,m in arms.items()}
    return dict(rows=rows,readout_sha256=before)

def main():
    OUT.mkdir(exist_ok=False)
    dry=[]
    for i in range(8):
        seed=int(sha256(f"MINIFLY|V210|WORLD|{i}".encode()).hexdigest()[:16],16)
        train=list(roster(f"{seed}|train",128,True))
        held=list(roster(f"{seed}|held",64,False))
        for r in train:
            render(r)
        for r in held:
            for l in (0,40):
                render(r,pad=l,training=False)
        dry.append(dict(seed=seed,train_sha256=digest(train),held_sha256=digest(held)))
    tests=[exercise(i) for i in range(2)]
    assert 8192+1200+512<=10000
    p=OUT/"CYCLE3_TEST.json"
    p.write_text(json.dumps(dict(status="TECHNICAL_ONLY_BRIDGE_UNRESOLVED",
        dry=dry,source_lock={n:sha256(Path(n).read_bytes()).hexdigest()
          for n in ("DESIGN.md","bridge.py","assay.py","audit.py","rehearse.py")},
        full_learned_question_applications=8192,technical_learned_question_applications=512,
        cumulative_with_historical_fits=9904,worlds=tests),indent=2)+"\n")
    with ZipFile(OUT/"TECHNICAL_BUNDLE.zip","w",ZIP_DEFLATED) as z:
        for path in ("CYCLE1_TEST.json","CYCLE2_TEST.json",str(p)):
            z.write(path,Path(path).name)
    with ZipFile(OUT/"TECHNICAL_BUNDLE.zip") as z:
        assert z.testzip() is None and z.read(p.name)==p.read_bytes()
    for r in tests:
        print(r["rows"])

if __name__=="__main__":
    main()
