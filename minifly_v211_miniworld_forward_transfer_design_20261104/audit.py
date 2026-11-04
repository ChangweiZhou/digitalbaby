"""Independent, evaluator-only audit of generated worlds."""
import json
import re
from pathlib import Path
from world import life,seed

MOVE=re.compile(r"(?:^([A-Z][a-z]+) (?:went to|is now in) the ([a-z]+)|^(?:In|At) the ([a-z]+),? is where ([A-Z][a-z]+))")
QUERY=re.compile(r"^Where (?:is|can) ([A-Z][a-z]+)")

def observe(sentence,state):
    a,b,c,d=MOVE.match(sentence).groups()
    state[a or d]=b or c

def answer(question,state):
    return state[QUERY.match(question)[1]]

def audit(worlds):
    tokens=set()
    shapes=[set(),set()]
    total_events=total_questions=0
    state={}
    for i,w in enumerate(worlds):
        target=i==len(worlds)-1
        current=set()
        for p in w:
            for e in p.events:
                observe(e,state)
            assert tuple(answer(q,state) for q in p.questions)==p.answers
            assert len(set(p.answers))==2
            current.update(QUERY.match(q)[1] for q in p.questions)
            for e in p.events:
                _,b,c,_=MOVE.match(e).groups()
                current.add(b or c)
            total_events+=len(p.events)
            total_questions+=len(p.questions)
            assert all((a+b)%2==target for a,b in p.forms)
            shapes[target].update(p.forms)
        assert len(current)==6 and not current&tokens
        tokens|=current
        assert sum(p.swap for p in w)==(0 if target else len(w)//2)
    assert shapes[0].isdisjoint(shapes[1])
    assert all(len(x)==8 for x in shapes)
    return total_events,total_questions

if __name__=="__main__":
    counts=[audit(life(seed(f"MINIFLY|V211|LIFE|{i}"))) for i in range(16)]
    assert counts==[(1152,576)]*16
    Path("CYCLE2_TEST.json").write_text(json.dumps(dict(status="PASS",dry_lives=16,
        physical_event_records=sum(a for a,b in counts),physical_questions=sum(b for a,b in counts)))+"\n")
