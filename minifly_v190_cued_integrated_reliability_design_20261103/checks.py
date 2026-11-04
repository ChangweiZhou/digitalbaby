"""V190 bounded engineering checks. No learner training."""
from collections import Counter
from pathlib import Path
import json,sys

def roster():
    held={3,6,10,13}
    domain=set(range(1,16))-{8}
    train={(a,(a+d)%16) for a in range(16) for d in domain-held}
    probe={(a,(a+d)%16) for a in range(16) for d in held}
    assert len(train)==160 and len(probe)==64 and not train&probe
    assert all((b,a) in probe for a,b in probe)
    for label in (lambda d:d<8,lambda d:d%2):
        assert sorted(Counter(label((b-a)%16) for a,b in probe).values())==[32,32]
    assert 3*(8+256+640+32+232)==3504
    return {'status':'PASS','held_per_relation':64,'taught_per_relation':160}

def reliability():
    votes=[[(1,2),(1,2)] for _ in range(2)]
    seen=Counter()
    for tag,prompt,y,preds in [(7,b'x',1,(1,0)),(9,b'x',0,(1,0)),
                                (7,b'x',1,(0,1)),(9,b'x',0,(0,1))]:
        key=tag,prompt
        first=seen[key]
        seen[key]+=1
        for expert,pred in enumerate(preds):
            c,n=votes[first][expert]
            votes[first][expert]=(c+int(pred==y),n+1)
    assert all(n==4 for policy in votes for _,n in policy)
    def complete(out,y):return out==[y,256]
    assert complete([19,256],19)
    assert not any(complete(out,19) for out in ([19],[256],[19,0,256]))
    return {'status':'PASS','vote_ledgers':votes,'EOS_canaries':4}

def source_admission():
    root=Path('/Users/pencilbard/Downloads')
    paths={name:[str(p) for p in root.rglob(name) if 'round4_20261104' in p.parts]
           for name in ('integ.py','bytecyc4.py')}
    return {'status':'SOURCE_CLOSURE_PENDING' if all(paths.values()) else 'SOURCE_NOT_ADMITTED',
            'paths':paths,'scientific_records':0}
cycle=int(sys.argv[1])
result=[roster,reliability,source_admission][cycle-1]()
Path(f'CYCLE{cycle}_TEST.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
