"""V206 reduced inherited-source replay and dry science roster."""
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

from address import ARMS
from bridge import CACHE, digest, load

LENGTHS=(0,16,40,64)
OUT=Path("rehearsal")

def sha(b):
    return hashlib.sha256(b).hexdigest()

def stories(seed,count,base):
    rng=random.Random(seed)
    result=[]
    for j in range(count):
        people=rng.sample(base.PEOPLE,2)
        places=rng.sample(base.PLACES,4)
        moves=[(people[i%2],places[i],rng.choice(base.TEMPLATES["original"]),
                "".join(rng.choices("xzvq",k=63))) for i in range(4)]
        result.append(moves)
    return result

def render(roster,length):
    lines=[]
    labels=[]
    for story in roster:
        i=0
        last={}
        for at,(person,place,verb,noise) in enumerate(story):
            i+=1
            pad=" "+noise[:length-1] if length else ""
            lines.append(f"{i} {person}{pad} {verb} {place}.")
            last[person]=place
            if at in (1,3):
                for who,value in last.items():
                    i+=1
                    lines.append(f"{i} Where is {who}? \t{value}\t1")
                    labels.append(value)
    return "\n".join(lines)+"\n",labels

def code(v,arm):
    src=v.source("position")
    assert src.count("    STREAM=[]")==1 and src.count("del hist[:-64]")==1
    src=src.replace("    STREAM=[]","    global WIN, bag\n    STREAM=[]")
    src=src.replace("del hist[:-64]","del hist[:-(128 if FROZ[0] else 64)]")
    src=src.replace("            frozen=learned_hash()",f"""            frozen=learned_hash()
            from address import make
            WIN={96 if arm=="RECENT96" else 32}
            bag=make(bag,{arm!r})""")
    assert "q,a=x; feed" in src and "feed(b'\\n',store=False)" in src
    src=src.replace("q,a=x; feed","q,a=x; prior_hist=hist[:]; feed")
    return src.replace("feed(b'\\n',store=False)","hist[:]=prior_hist")

def main():
    start=time.monotonic()
    v=load()
    previous=json.loads(Path("CYCLE1_TEST.json").read_text())
    assert digest(CACHE/"parent.npz")==previous["checkpoint_sha256"]
    OUT.mkdir(exist_ok=False)
    dry=[]
    for i in range(8):
        seed=int(sha(f"MINIFLY|V206|WORLD|{i}".encode())[:16],16)
        roster=stories(seed,32,v.base)
        dry.append([sha(render(roster,l)[0].encode()) for l in LENGTHS])
    receipt={}
    expected_frozen=None
    for i in range(2):
        roster=stories(f"V206-TECH|{i}",4,v.base)
        for length in LENGTHS:
            raw,labels=render(roster,length)
            case=CACHE/f"technical-{i}-{length}.txt"
            case.write_text(raw)
            for arm in ARMS:
                env=dict(os.environ,PYTHONHASHSEED="1",OPENBLAS_NUM_THREADS="1",
                         QA_GOLD="0",QA_LOAD=str(CACHE/"parent.npz"),QA_SAVE="",QA_CONTROL="oracle",
                         FREEZE="1",BIN="1",VALID="0",STORE="gated",EPOCHS="1",GORD="3,4,5",
                         GTH=".15",QNOSTORE="1",STEPN="2",MATCHN="3",UNIT="1",SEG="1",WIN="32",
                         HABT="0",CREDIT="match",RATE="1")
                child=subprocess.check_output([sys.executable,"-B","-c",code(v,arm),str(v.base.TRAIN),str(case)],env=env,text=True)
                row=json.loads(child)
                expected_frozen=expected_frozen or row["frozen"]
                assert row["frozen"]==expected_frozen and row["n"]==len(labels)==16
                diag=row["gate_audit"]
                assert diag["hit"]==16 and diag["follow"]==32 and diag["false_write"]==diag["miss"]==0
                exact=[bytes.fromhex(x)==(y+"\n").encode() for x,y in zip(row["replies"],labels)]
                row["final_exact"]=sum(x for j,x in enumerate(exact) if j%4>=2)
                assert round(sum(exact)/16,3)==row["exact_answer_with_own_newline"]
                p=OUT/f"{i}_{length}_{arm}.json"
                p.write_text(json.dumps(row,indent=2)+"\n")
                receipt[p.name]=digest(p)
    cells={arm:[sum(json.loads((OUT/f"{i}_{l}_{arm}.json").read_text())["final_exact"] for i in range(2))
                for l in LENGTHS] for arm in ARMS}
    summary=dict(status="TECHNICAL_COMPLETE",final_exact_per_16=cells,fixture_story_arm_applications=128,
                 full_story_arm_applications=4096,full_write_opportunities=16384,full_answers=16384,
                 source_hashes={p:digest(Path(p)) for p in ("DESIGN.md","bridge.py","address.py","test_address.py","rehearse.py")},
                 locked_dry_renderings=dry,receipts=receipt,elapsed_seconds=time.monotonic()-start,
                 child_peak_rss=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    (OUT/"CYCLE3_TEST.json").write_text(json.dumps(summary,indent=2)+"\n")
    with ZipFile(OUT/"TECHNICAL_BUNDLE.zip","w",ZIP_DEFLATED) as z:
        for p in OUT.glob("*.json"):
            z.write(p,p.name)
    with ZipFile(OUT/"TECHNICAL_BUNDLE.zip") as z:
        assert z.testzip() is None
        assert all(sha(z.read(name))==h for name,h in receipt.items())
    print(cells)

if __name__=="__main__":
    main()
