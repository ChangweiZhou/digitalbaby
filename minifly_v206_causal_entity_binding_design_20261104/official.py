"""Locked eight-world V206 comparison using the qualified inherited path."""
import json
import math
import os
from pathlib import Path
import resource
import statistics as st
import subprocess
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

from rehearse import ARMS, LENGTHS, CACHE, code, digest, load, render, sha, stories

OUT=Path("official")
START=time.monotonic()
CARRY=60

def save(name,data):
    p=OUT/name
    t=p.with_suffix(".tmp")
    t.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    t.replace(p)
    return digest(p)

def resources():
    own=resource.getrusage(resource.RUSAGE_SELF)
    child=resource.getrusage(resource.RUSAGE_CHILDREN)
    result=dict(cpu_seconds=CARRY+own.ru_utime+own.ru_stime+child.ru_utime+child.ru_stime,
                wall_seconds=CARRY+time.monotonic()-START,peak_child_rss=child.ru_maxrss,
                output_bytes=sum(p.stat().st_size for p in OUT.iterdir()))
    assert result["cpu_seconds"]<=1200 and result["wall_seconds"]<=2400
    assert max(own.ru_maxrss,child.ru_maxrss)<=2**30
    assert result["output_bytes"]<=128*2**20
    return result

def env():
    return dict(os.environ,PYTHONHASHSEED="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",
         QA_GOLD="0",QA_LOAD=str(CACHE/"parent.npz"),QA_SAVE="",QA_CONTROL="oracle",
         FREEZE="1",BIN="1",VALID="0",STORE="gated",EPOCHS="1",GORD="3,4,5",
         GTH=".15",QNOSTORE="1",STEPN="2",MATCHN="3",UNIT="1",SEG="1",WIN="32",
         HABT="0",CREDIT="match",RATE="1")

def reducer(lives):
    hold="SPLIT-ANCHOR"
    def score(i,arm,l,kind="exact"):
        return st.mean(x[kind] for x in lives[i,l,arm]["final"])
    def mean(arm,l):
        return st.mean(score(i,arm,l) for i in range(8))
    effects={}
    for base in ("RECENT32","SPLIT-RECENT"):
        d=[st.mean(score(i,hold,l)-score(i,base,l) for l in (40,64)) for i in range(8)]
        center=st.mean(d)
        half=2.364624*st.stdev(d)/math.sqrt(8)
        effects[base]=dict(mean=center,interval=[center-half,center+half],worlds=d)
    def panel(arm):
        return [x for i in range(8) for l in (40,64) for x in lives[i,l,arm]["final"]]
    a,b=panel(hold),panel("SPLIT-RECENT")
    guards=dict(anchor_effect=effects["SPLIT-RECENT"]["mean"]>=.10 and
                               all(e["interval"][0]>0 for e in effects.values()),
                long_floor=all(mean(hold,l)>=.75 for l in (40,64)),
                reference=mean("RECENT32",0)>=.90 and mean(hold,0)>=mean("RECENT32",0)-.05,
                both_subjects=all(st.mean(x["exact"] for x in a if x["slot"]==s)>
                                  st.mean(x["exact"] for x in b if x["slot"]==s) for s in (0,1)),
                cross_person=st.mean(x["other"] for x in a)<st.mean(x["other"] for x in b))
    return guards,effects,{arm:{str(l):mean(arm,l) for l in LENGTHS} for arm in ARMS}

def main():
    q=json.loads(Path("rehearsal/CYCLE3_TEST.json").read_text())
    b=json.loads(Path("CYCLE1_TEST.json").read_text())
    assert q["status"]=="TECHNICAL_COMPLETE" and b["published_row_exact"]
    assert digest(CACHE/"parent.npz")==b["checkpoint_sha256"]
    for p,h in q["source_hashes"].items():
        assert digest(Path(p))==h
    for p,h in q["receipts"].items():
        assert digest(Path("rehearsal")/p)==h
    v=load()
    from bridge import PATHS, GENERAL, PARENT
    for src,dst in [*PATHS.values(),(GENERAL,"minifly-qa-generalization/qualify.py")]:
        assert digest(PARENT/dst)==b["sources"][src]
    tapes={}
    payload_bytes=0
    for i in range(8):
        seed=int(sha(f"MINIFLY|V206|WORLD|{i}".encode())[:16],16)
        roster=stories(seed,32,v.base)
        for j,length in enumerate(LENGTHS):
            raw,labels=render(roster,length)
            assert sha(raw.encode())==q["locked_dry_renderings"][i][j]
            lines=[line.split(" ",1)[1].split("\t")[0].strip() for line in raw.splitlines()]
            assert len(labels)==128 and len(lines)==256
            payload_bytes+=sum(len(line.encode())+1 for line in lines)
            for story in roster:
                for who,_,verb,pad in story:
                    prefix=who+(" "+pad[:length-1] if length else "")+" "+verb+" "
                    assert who in prefix[-96:] and who in prefix[:16]
                    if length>=40:
                        assert who not in prefix[-32:]
            tapes[i,length]=(raw,labels)
    projected=q["elapsed_seconds"]*32*3+CARRY
    assert projected<=1200 and q["child_peak_rss"]<=2**30
    assert 4096+128+20<=5000 and 16384+512+200<=20000 and 16384+512+100<=20000
    save("LOCK.json",dict(status="ADMITTED",sources={**q["source_hashes"],"official.py":digest(Path(__file__))},
        checkpoint=b["checkpoint_sha256"],roster=q["locked_dry_renderings"],projected_cpu=projected,
        story_arm_applications=4096,write_opportunities=16384,answer_generations=16384,
        output_cap=12,unique_render_payload_bytes=payload_bytes,charged_payload_bytes=4*payload_bytes,
        historical_checkpoint_training_lines=3000))
    commits=[]
    frozen=None
    for (i,length),(raw,labels) in tapes.items():
        case=CACHE/f"official-{i}-{length}.txt"
        case.write_text(raw)
        for arm in ARMS:
            resources()
            result=subprocess.check_output([sys.executable,"-B","-c",code(v,arm),str(v.base.TRAIN),str(case)],
                                           env=env(),text=True,timeout=30)
            r=json.loads(result)
            frozen=frozen or r["frozen"]
            assert r["frozen"]==frozen and r["n"]==len(labels)==128
            d=r["gate_audit"]
            assert d["hit"]==128 and d["follow"]==256 and d["false_write"]==d["miss"]==0
            emitted=[bytes.fromhex(x) for x in r["replies"]]
            assert len(emitted)==128 and all(0<len(x)<=12 for x in emitted)
            exact=[x==(y+"\n").encode() for x,y in zip(emitted,labels)]
            assert round(st.mean(exact),3)==r["exact_answer_with_own_newline"]
            r["final"]=[dict(slot=j%2,exact=exact[j],other=x==(labels[j^1]+"\n").encode())
                        for j,x in enumerate(emitted) if j%4>=2]
            r.update(world=i,length=length,arm=arm,emitted_bytes=sum(map(len,emitted)))
            name=f"{i}_{length}_{arm}_RECEIPT.json"
            commits.append(dict(file=name,sha256=save(name,r)))
        save("COMMIT_INDEX.json",commits)
        print(f"Committed world {i}, padding {length}, all four arms",flush=True)
    lives={}
    for entry in commits:
        p=OUT/entry["file"]
        assert digest(p)==entry["sha256"]
        r=json.loads(p.read_text())
        k=r["world"],r["length"],r["arm"]
        assert k not in lives
        lives[k]=r
    assert len(lives)==128 and all(len(r["final"])==64 for r in lives.values())
    guards,effects,scores=reducer(lives)
    verdict="ADVANCE_TO_GATE_STUDY" if all(guards.values()) else "DID_NOT_ADVANCE"
    summary=dict(verdict=verdict,guards=guards,effects=effects,scores=scores,
        story_arm_applications=4096,answer_generations=sum(r["n"] for r in lives.values()),
        write_opportunities=sum(r["gate_audit"]["hit"] for r in lives.values()),
        committed_value_bytes=sum(r["gate_audit"]["hit"]+r["gate_audit"]["follow"] for r in lives.values()),
        row_writes=sum(r["gate_audit"]["row_writes"] for r in lives.values()),
        emitted_bytes=sum(r["emitted_bytes"] for r in lives.values()),
        shortcuts=dict(query_only_current_label_coverage=0,canonical_latest=1,
                       subject_blind_latest=.5),resources=resources())
    save("SUMMARY.json",summary)
    table="".join(f"| {a} | "+" | ".join(f"{scores[a][str(l)]:.2%}" for l in LENGTHS)+" |\n" for a in ARMS)
    (OUT/"REPORT.md").write_text(f"# V206 official result\n\n**{verdict}** · Eight paired worlds and 4,096 story-arm applications.\n\n"
        "| Arm | 0 bytes | 16 bytes | 40 bytes | 64 bytes |\n|---|---:|---:|---:|---:|\n"+table+
        "\nScores require the exact final destination and own newline. Write timing was supplied.\n")
    paths=sorted(OUT.iterdir())
    with ZipFile(OUT/"RESULT_BUNDLE.zip","w",ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p,p.name)
    with ZipFile(OUT/"RESULT_BUNDLE.zip") as z:
        assert z.testzip() is None and all(z.read(p.name)==p.read_bytes() for p in paths)
    save("PACKAGE.json",dict(status="COMPLETE",sha256=digest(OUT/"RESULT_BUNDLE.zip"),resources=resources()))

if __name__=="__main__":
    OUT.mkdir(exist_ok=False)
    try:
        main()
    except Exception as e:
        save("FAILURE.json",dict(error=repr(e)))
        raise
