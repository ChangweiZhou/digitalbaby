"""Evaluator-owned fixed environment. Hidden map NEVER passed to Learner.

Observable events are cue bytes then the actual received outcome byte; no
relation task, hidden answer construction, or task-dependent output mode.
"""
import hashlib,json,random
ALPHABET=b'0123'
DT=30./14
RECORD_SECONDS=165.
DAY=86400.
BOUTS=12

def rng(world,key): return random.Random(int.from_bytes(hashlib.sha256(f'R3-OBS-v1|{world}|{key}'.encode()).digest()[:16],'big'))
def make_world(world):
    sets={}
    for name,digits in [('old',b'0123'),('new',b'4567')]:
        cues=[(b' '*8+bytes((a,43,b,61))).hex() for a in digits for b in digits]
        outcomes=list(ALPHABET)*4; rng(world,name+'map').shuffle(outcomes)
        sets[name]=dict(cues=cues,outcomes=outcomes)
    events=[]
    for name in ('old','new'):
        for bout in range(BOUTS):
            order=list(range(16));rng(world,f'{name}schedule{bout}').shuffle(order)
            for item in order:
                i=len(events);at=i*RECORD_SECONDS+(DAY if name=='new' else 0)
                events.append(dict(cue_hex=sets[name]['cues'][item],outcome=sets[name]['outcomes'][item],at=at,stage=name,item=item))
    doc=dict(world=world,sets=sets,events=events,old_end=192*RECORD_SECONDS,new_start=192*RECORD_SECONDS+DAY,new_end=384*RECORD_SECONDS+DAY,final=384*RECORD_SECONDS+2*DAY)
    doc['sha256']=hashlib.sha256(json.dumps(doc,sort_keys=True).encode()).hexdigest()
    return doc
