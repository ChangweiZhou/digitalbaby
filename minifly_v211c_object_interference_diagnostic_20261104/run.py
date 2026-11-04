"""Read-only probes of object-event interference using the unchanged V211B learner."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics as st
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

PARENT=Path("../minifly_v211b_recovered_miniworld_20261104")
OUT=Path("results")
SEEDS=tuple(range(40,48))
NAMES=4
PLACES=6
OBJECTS=4

def sha(b):
    return hashlib.sha256(b).hexdigest()

def grams(s):
    return {s[i:i+n] for n in (2,3,4) for i in range(len(s)-n+1)}

def overlap(words):
    g=[grams(w) for w in words]
    return sum(len(a&b) for i,a in enumerate(g) for b in g[i+1:])

def vocab(ctx,seed,forbidden):
    rng=random.Random(80000+seed)
    regular=[]
    while len(regular)<NAMES+PLACES+OBJECTS:
        w="".join(rng.choices(ctx["SYL"],k=3))
        if w not in forbidden and w not in regular:
            regular.append(w)
    best=regular
    for _ in range(128):
        proposal=[]
        for start,end in ((0,NAMES),(NAMES,NAMES+PLACES),(NAMES+PLACES,len(regular))):
            letters=list("".join(regular[start:end]))
            rng.shuffle(letters)
            proposal.extend("".join(letters[i:i+6]) for i in range(0,len(letters),6))
        if len(set(proposal))==len(regular) and not set(proposal)&forbidden and overlap(proposal)<overlap(best):
            best=proposal
    assert overlap(best)<overlap(regular)
    return [tuple(w.capitalize() if i<NAMES else w for i,w in enumerate(group)) for group in (regular,best)]

def schedule(seed):
    r=random.Random(20000+seed)
    order=[0,1]*16
    r.shuffle(order)
    return [(a,b,x,y,r.randrange(OBJECTS),bool(j%2),last)
            for j,last in enumerate(order)
            for a,b in [r.sample(range(NAMES),2)] for x,y in [r.sample(range(PLACES),2)]]

def save(name,data):
    p=OUT/name
    p.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    return sha(p.read_bytes())

def probe(m,q,target):
    ix,v=m.units(q.rstrip("?").encode())
    bag=v@m.M[ix]
    cand=np.flatnonzero(bag>0)
    p0=(m.H.sum()+1)/(m.T.sum()+2)
    def logit(x):
        return np.log(x/(1-x))
    scores={}
    for i in cand:
        word=m.words[i]
        fs=[(m.H[i]+2*p0)/(m.T[i]+2)]
        fs.extend((m.CH.get(f,0)+2*p0)/(m.CT.get(f,0)+2) for f in m.CF.get(word,[]))
        score=bag[i]/(1+np.exp(-(logit(p0)+sum(logit(x)-logit(p0) for x in fs))))
        scores[word]=float(score)
    rng=m.rng.getstate()
    chosen,actual=m.ask(q)
    m.rng.setstate(rng)
    assert np.array_equal(cand,actual)
    assert chosen is None or scores[chosen]>=max(scores.values())-1e-12
    return dict(chosen=chosen,correct=chosen==target,available=target in scores,candidate_scores=scores)

def test(ctx,sd,arm,words,prior,packets):
    m=ctx["Learner"](sd)
    for w in prior:
        ctx["live"](m,random.Random(w[0][0]+str(sd)),w,40,"true")
    assert not {w.lower() for w in words}&{w.lower() for w in m.lex}
    ids=[set(m.units(w.lower().encode(),True)[0]) for w in words]
    hashed=sum(len(a&b) for i,a in enumerate(ids) for b in ids[i+1:])
    frozen=(m.NQ,m.H.copy(),m.T.copy(),m.CH.copy(),m.CT.copy(),m.DF.copy())
    rows=[]
    for a,b,x,y,o,drop,last in packets:
        name,other=words[a],words[b]
        place,alt=words[NAMES+x],words[NAMES+y]
        obj=words[NAMES+PLACES+o]
        movements=(f"{name} went to the {place}.",f"{other} went to the {alt}.")
        for i in (1-last,last):
            m.see(movements[i])
        q=f"Where is {name}?"
        before=probe(m,q,place)
        action="dropped" if drop else "picked up"
        foreign=f"{other} {action} the {obj}."
        local=f"{name} {action} the {obj}."
        touched=m.units(foreign.encode(),True)[0]
        restored=m.M[touched].copy()
        lex,allwords,context=m.lex.copy(),m.words.copy(),m.CF.copy()
        m.see(foreign)
        cross=probe(m,q,place)
        m.M[touched]=restored
        m.lex,m.words,m.CF=lex,allwords,context
        assert probe(m,q,place)==before
        m.see(local)
        after=probe(m,q,place)
        qrows=set(m.units(q.rstrip("?").encode())[0])
        shared=[len(qrows&set(m.units(event.encode(),True)[0])) for event in (foreign,local)]
        rows.append(dict(target=place,query=q,before=before,other=cross,same=after,
                         object_action=action,query_object_row_overlap=shared))
    assert frozen[0]==m.NQ and np.array_equal(frozen[1],m.H) and np.array_equal(frozen[2],m.T)
    assert frozen[3:]==(m.CH,m.CT,m.DF)
    return dict(world=sd,arm=arm,raw_overlap=overlap([w.lower() for w in words]),
                hashed_overlap=hashed,rows=rows)

def stats(results):
    def mean(rows,stage,key):
        return st.mean(r[stage][key] for r in rows)
    output={}
    for arm,items in results.items():
        rows=[r for w in items for r in w["rows"]]
        output[arm]=dict(raw_overlap=st.mean(w["raw_overlap"] for w in items),
                         hashed_overlap=st.mean(w["hashed_overlap"] for w in items),
                         stages={s:{**{k:mean(rows,s,k) for k in ("correct","available")},
                                    "present_but_lost":sum(r[s]["available"] and not r[s]["correct"] for r in rows)}
                                 for s in ("before","other","same")})
    effects={}
    for key in ("correct","available"):
        for stage in ("before","other","same"):
            d=[mean(b["rows"],stage,key)-mean(a["rows"],stage,key)
               for a,b in zip(results["SYLLABLE"],results["LOW-OVERLAP"])]
            margin=2.364624*st.stdev(d)/math.sqrt(8)
            effects[f"low_minus_syllable_{stage}_{key}"]=[st.mean(d),st.mean(d)-margin,st.mean(d)+margin]
        for arm,items in results.items():
            d=[mean(w["rows"],"same",key)-mean(w["rows"],"before",key) for w in items]
            margin=2.364624*st.stdev(d)/math.sqrt(8)
            effects[f"{arm}_own_minus_before_{key}"]=[st.mean(d),st.mean(d)-margin,st.mean(d)+margin]
    return output,effects

def main():
    assert os.environ["PYTHONHASHSEED"]=="1" and not os.environ.get("NOCTX")
    source=(PARENT/"miniworld.py").read_bytes()
    expected=json.loads((PARENT/"RECOVERY.json").read_text())["source_sha256"]
    assert sha(source)==expected
    ctx={}
    code=source.decode().split("K=int(E('K','30'))",1)[0]
    exec(compile(code,"miniworld.py","exec"),ctx)
    OUT.mkdir(exist_ok=False)
    prepared={}
    for sd in SEEDS:
        r=random.Random(sd)
        prior=[ctx["world"](r,ctx["POOL_TRAIN"]) for _ in range(5)]
        forbidden={w.lower() for world in prior for group in world[:2] for w in group}
        prepared[sd]=(prior,vocab(ctx,sd,forbidden),schedule(sd))
    save("LOCK.json",dict(seeds=SEEDS,parent_sha256=expected,driver_sha256=sha(Path(__file__).read_bytes()),
        schedule_sha256=sha(repr(prepared).encode()),prior_questions=3200,planned_probe_questions=1536))
    results={a:[] for a in ("SYLLABLE","LOW-OVERLAP")}
    commits={}
    for sd,(prior,vocabularies,packets) in prepared.items():
        for arm,words in zip(results,vocabularies):
            r=test(ctx,sd,arm,words,prior,packets)
            name=f"{sd}_{arm}.json"
            commits[name]=save(name,r)
            results[arm].append(r)
        print(f"Committed paired world {sd}",flush=True)
    assert all(sha((OUT/n).read_bytes())==h for n,h in commits.items())
    cells,effects=stats(results)
    save("SUMMARY.json",dict(status="DIRECT_PATH_DIAGNOSTIC_COMPLETE_ROUND17_SOURCE_MISSING",
        worlds=8,packets=512,scored_probes=1536,extra_restore_checks=512,prior_questions=3200,
        cells=cells,effects=effects,receipt_sha256=commits))
    print(json.dumps(cells,indent=2))

if __name__=="__main__":
    main()
