"""V200 technical runner, fixed dry census and paired reducer."""
import hashlib
import json
import random
import resource
import statistics
import time
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from learner import ARMS, EOS, Learner, chain, exact_chain

FORMS = (
    lambda k: b"the code of "+k+b" is ",
    lambda k: b"recall the code for "+k+b" => ",
    lambda k: b"remember "+k+b": ",
    lambda k: b"tell me, what is "+k+b"? ",
)
ALPHABET = b"mnopqrstuvwxyz"

def digest(b):
    return hashlib.sha256(b).hexdigest()

def roster(seed, n, new):
    streams = [random.Random(int(digest(f"{seed}|{tag}".encode())[:16],16))
               for tag in ("keys","links","digits")]
    keys, links, digits = streams
    used = set()
    def key(length):
        while True:
            k = bytes(keys.choices(ALPHABET,k=length))
            if k not in used:
                used.add(k)
                return k
    cohorts = []
    for count in (n,new):
        a = [key(5) for _ in range(count)]
        b = [key(3+i%5) for i in range(count)]
        d = [bytes([48+i%10]) for i in range(count)]
        links.shuffle(b)
        digits.shuffle(d)
        cohorts.append(list(zip(a,b,d)))
    return cohorts

def census(cohorts):
    return sum(2*len(c)*12 for c in cohorts), sum((len(b)+3)*12 for c in cohorts for _,b,_ in c)

def state(m):
    h=hashlib.sha256()
    for a in (m.weights,m.importance):
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()

def teach(m, cohort, seed, warm, table):
    rng = random.Random(seed)
    links=[link for a,b,d in cohort for link in ((a,b),(b,d))]
    for rep in range(12):
        order=links.copy()
        rng.shuffle(order)
        for k,y in order:
            q=FORMS[rep%2](k)
            m.teach(q,y,(.1,.3)[rep] if warm and rep<2 else .6)
            table[q]=y

def probe(m, originals, observed):
    start=state(m)
    canonical={key: observed[FORMS[0](key)] for a,b,_ in originals for key in (a,b)}
    rows=[]
    for j,form in enumerate(FORMS):
        for i,(a,b,d) in enumerate(originals):
            first, second=chain(m.generate,form,a)
            gold_second=m.generate(form(b))
            gold_stop=int(m.score(form(a),b)[1].argmax())==EOS
            assert (form(a) in observed)==(j<2)
            assert (form(b) in observed)==(j<2)
            rows.append(dict(item=i, wrapper=j, length=len(b), first=first[0].hex(),
                             eos=first[1], prefix=first[0].startswith(b), hop1=first==(b,True),
                             strict=exact_chain(first,second,b,d), gold_stop=gold_stop,
                             gold_second=gold_second==(d,True), canonical=canonical[canonical[a]]==d,
                             exact_table=observed.get(form(a))==b and observed.get(form(b))==d))
    assert start==state(m)
    return rows

def rate(items, key, held):
    return statistics.mean(i[key] for i in items if (i["wrapper"]>=2)==held)

def decision(worlds):
    assert len(worlds)==8
    hold="SPLIT-HOLD"
    effects={}
    for base in ("ROLL","SPLIT-ROLL"):
        d=[rate(w[hold],"hop1",True)-rate(w[base],"hop1",True) for w in worlds]
        effects[base]=dict(mean=statistics.mean(d),lower=statistics.mean(d)-2.364624*statistics.stdev(d)/8**.5)
    def mean(arm,key,held):
        return statistics.mean(rate(w[arm],key,held) for w in worlds)
    return (effects["ROLL"]["mean"]>=.10 and all(v["lower"]>0 for v in effects.values())
            and all(sum(sum(i["hop1"] for i in w[hold] if i["wrapper"]==j)
                        -sum(i["hop1"] for i in w["ROLL"] if i["wrapper"]==j) for w in worlds)>=0 for j in (2,3))
            and mean("ROLL","strict",False)>=.8
            and all(mean(hold,k,False)>=mean("ROLL",k,False)-.05 for k in ("hop1","strict"))
            and mean(hold,"strict",True)>=.2
            and mean(hold,"strict",True)-mean("ROLL","strict",True)>=.05)

def main():
    cpu=time.process_time()
    def seed(i):
        return int(digest(f"MINIFLY|V200|WORLD|{i}".encode())[:16],16)
    full=[roster(seed(i),128,32) for i in range(8)]
    records,targets=[sum(x)*len(ARMS) for x in zip(*(census(w) for w in full))]
    assert records==92160 and targets<=460800
    originals,new=cohorts=roster("V200-TECHNICAL",5,1)
    technical,technical_targets=[x*3 for x in census(cohorts)]
    bridge=json.loads(Path("CYCLE1_TEST.json").read_text())
    assert bridge["status"]=="PASS"
    assert records+technical+bridge["presentations"]<=105000
    assert targets+technical_targets+bridge["targets"]<=600000
    receipts={}
    for arm in ARMS:
        m=Learner(arm)
        table={}
        teach(m,originals,10,True,table)
        initial=probe(m,originals,table)
        # The query buffer starts fresh and unmarked filler has no writer.
        delay=[b"we said so. "]*256
        assert not any(p in delay for p in table)
        delayed=probe(m,originals,table)
        assert initial==delayed
        teach(m,new,11,False,table)
        final=probe(m,originals,table)
        assert m.sites==technical_targets//3
        receipts[arm]=dict(stages=[initial,delayed,final], state=state(m), sites=m.sites,
                           rows_read=m.read_rows,rows_written=m.write_rows,decisions=m.decisions)
    def mock(trained,held):
        return [dict(wrapper=j,hop1=trained if j<2 else held,strict=trained if j<2 else held) for j in range(4)]
    positive={"ROLL":mock(1,0),"SPLIT-ROLL":mock(1,0),"SPLIT-HOLD":mock(1,1)}
    assert decision([positive]*8)
    assert not decision([dict(positive, **{"SPLIT-ROLL":mock(1,1)})]*8)
    out=Path("rehearsal")
    out.mkdir(exist_ok=False)
    hashes={}
    for arm,r in receipts.items():
        p=out/f"{arm}_RECEIPT.json"
        p.write_text(json.dumps(r,indent=2)+"\n")
        hashes[p.name]=digest(p.read_bytes())
    result=dict(status="TECHNICAL_PASS_OFFICIAL_PENDING", fixture_records=technical,
                fixture_targets=technical_targets, full_records=records, full_targets=targets,
                charged_records=technical+records+bridge["presentations"], charged_targets=targets+technical_targets+bridge["targets"],
                dry_roster=[digest(repr(w).encode()) for w in full], receipt_hashes=hashes,
                sources={p:digest(Path(p).read_bytes()) for p in ("DESIGN.md","parent_hop.py","bridge.py","learner.py","audit.py","rehearse.py")},
                fixture_cpu=time.process_time()-cpu,fixture_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                fixture_scores={arm:{k:rate(r["stages"][-1],k,True) for k in ("hop1","strict")} for arm,r in receipts.items()})
    (out/"CYCLE3_TEST.json").write_text(json.dumps(result,indent=2)+"\n")
    bundle=out/"TECHNICAL_BUNDLE.zip"
    with ZipFile(bundle,"w",ZIP_DEFLATED) as z:
        for p in out.glob("*.json"):
            z.write(p,p.name)
    with ZipFile(bundle) as z:
        assert z.testzip() is None
        assert all(digest(z.read(name))==sha for name,sha in hashes.items())
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
