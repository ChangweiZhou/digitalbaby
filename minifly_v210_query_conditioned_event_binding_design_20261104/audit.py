from pathlib import Path
import json
from collections import Counter
from assay import LEARNED,choose,features,render,roster
train=list(roster("V210-TECH-AUDIT",32,True))
held=list(roster("V210-TECH-HELD",16,False))
counts=Counter((tuple(x[2:6]),x[-1]) for x in train)
assert all(counts[k,0]==counts[k,1]==1 for k,_ in counts)
for record in train+held:
    training=record in train
    _,cs,qs,ys,_=render(record,training=training)
    assert ys[0]!=ys[1] and len(set(qs))==2
    assert len({c.word for c in cs})==4
    assert choose(dict.fromkeys((c.word for c in cs),0))==choose(dict.fromkeys((c.word for c in reversed(cs)),0))
    for a in LEARNED:
        for c in cs:
            for q in qs:
                assert 0<len(features(a,q,c))<=96
        if a=="QUESTION-BLIND":
            assert all(features(a,qs[0],c)==features(a,qs[1],c) for c in cs)
Path("CYCLE2_TEST.json").write_text(json.dumps(dict(status="PASS",untrained_canary_stories=48))+"\n")
