"""Authorized fixed-roster V208 confirmation."""
import hashlib
import json
from pathlib import Path
import resource
import statistics as st
import time
from zipfile import ZipFile, ZIP_DEFLATED

from launch import execute_world, tape
from protocol import ARMS, HELD, World, decision, interval, manifest, parent, rate, sha

OUT=Path("official")
START=time.monotonic()

def save(name,data):
    p=OUT/name
    tmp=p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    tmp.replace(p)
    return sha(p)

def limits():
    cost=dict(cpu_seconds=time.process_time()+10,wall_seconds=time.monotonic()-START+30,
              peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              output_bytes=sum(p.stat().st_size for p in OUT.iterdir()))
    assert cost["cpu_seconds"]<=300 and cost["wall_seconds"]<=600
    assert cost["peak_rss_bytes"]<=512*2**20 and cost["output_bytes"]<=64*2**20
    return cost

def main():
    base=manifest()
    technical=Path("rehearsal")
    locked=json.loads((technical/"DRY_LOCK.json").read_text())
    q=json.loads((technical/"CYCLE3_TEST.json").read_text())
    assert q["status"]=="TECHNICAL_PASS" and locked["parent_sources"]==base["sources"]
    assert q["projected_cpu"]<300 and q["projected_wall"]<600
    for p,h in locked["sources"].items():
        assert sha(Path(p))==h
    for r in q["commits"]:
        assert sha(technical/r["path"])==r["sha256"]
    for k,cap in dict(episodes=24000,arm_applications=72000,updates=48000,probes=10000).items():
        assert q["cumulative_with_science"][k]<=cap
    worlds=[World(r["seed"]) for r in locked["roster"]]
    assert len({w.world for w in worlds})==32
    assert not {w.world for w in worlds}&{r["seed"] for r in base["roster"]}
    for w,r in zip(worlds,locked["roster"]):
        events=tape(w)
        assert len(events)==640 and sum(len(e.prompt)+len(e.neutral)+1 for e in events)==5760
        assert hashlib.sha256(repr(events).encode()).hexdigest()==r["schedule_sha256"]
        assert not {w.prompt(c,f) for c in range(4) for f in HELD}.intersection(e.prompt for e in events)
    save("LOCK.json",dict(**locked,authorization="Do the official run.",official_sha256=sha(Path(__file__)),
                          technical_result_sha256=sha(technical/"CYCLE3_TEST.json"),
                          cumulative=q["cumulative_with_science"]))
    index=[]
    for i,w in enumerate(worlds):
        limits()
        r=execute_world(w)
        p=f"WORLD_{i:02d}_RECEIPT.json"
        index.append(dict(path=p,sha256=save(p,r),world=w.world))
        save("COMMIT_INDEX.json",index)
        print(f"Committed {i+1}/32",flush=True)
    lives=[]
    for w,e in zip(worlds,index):
        p=OUT/e["path"]
        assert sha(p)==e["sha256"]
        r=json.loads(p.read_text())
        assert parent.audit(w,r)==r["stats"]
        lives.append(r)
    result=decision(lives)
    def average(stage,arm,held):
        return st.mean(rate(r["stages"][stage],arm,held) for r in lives)
    scores={stage:{a:dict(trained=average(stage,a,False),held=average(stage,a,True)) for a in ARMS}
            for stage in parent.STAGES}
    online={}
    for phase in parent.PHASES:
        episodes,safe,preferred,explored=[sum(r["stats"][phase][j] for r in lives) for j in range(4)]
        online[phase]=dict(episodes=episodes,safety=safe/episodes,preferred_actions=preferred/episodes,
                           exploration=explored/episodes)
    assert sum(p["episodes"] for p in online.values())==20480
    bands={}
    for lo,hi in ((1,4),(5,8),(9,16)):
        s=[st.mean(y for b in r["history"] if b["phase"]=="R" and lo<=b["block"]+1<=hi
                   for y in b["outcomes"]) for r in lives]
        bands[f"{lo}-{hi}"]=dict(mean=st.mean(s),interval=interval(s))
    shortcuts={key:st.mean(r["shortcuts"][key] for r in lives) for key in lives[0]["shortcuts"]}
    verdict="CONFIRMED_E2" if result["passed"] else "DID_NOT_CONFIRM"
    save("SUMMARY.json",dict(verdict=verdict,worlds=32,physical_actions=20480,arm_applications=61440,
         updates=40960,probes=6144,interface_bytes=184320,generated_placeholder_bytes=2048,
         **result,scores=scores,online=online,reversal_bands=bands,shortcuts=shortcuts,cost=limits()))
    table="".join(f"| {a} | {scores['FINAL'][a]['trained']:.2%} | {scores['FINAL'][a]['held']:.2%} |\n" for a in ARMS)
    safe=result["live_safety"]
    contrasts="".join(f"- ALIGNED minus {a} {d['mean']:+.2%} (paired 95% interval {d['interval'][0]:+.2%} to {d['interval'][1]:+.2%}).\n" for a,d in result["contrasts"].items())
    (OUT/"REPORT.md").write_text(f"# V208 consequence-driven action confirmation\n\n**{verdict}** · 32 fresh worlds, 20,480 learner-chosen actions.\n\n"
        "| Arm | Final trained | Final held |\n|---|---:|---:|\n"+table+"\n"+contrasts+
        f"\nLive reversal safety was {safe['mean']:.2%} (95% world interval {safe['interval'][0]:.2%} to {safe['interval'][1]:.2%}). "
        f"The evaluator-only canonical recency table scored {shortcuts['canonical_recency']:.2%} on the final held panel.\n\n"
        "The learner chose its own actions and learned from their sampled consequences. This assay uses one pending action and fixed exploration. "
        "The nominal neutral gap was generated without a learner transition. Retention follows actual interfering action experience.\n")
    files=sorted(OUT.iterdir())
    with ZipFile(OUT/"RESULT_BUNDLE.zip","w",ZIP_DEFLATED) as z:
        for p in files:
            z.write(p,p.name)
    with ZipFile(OUT/"RESULT_BUNDLE.zip") as z:
        assert z.testzip() is None and all(z.read(p.name)==p.read_bytes() for p in files)
    save("PACKAGE.json",dict(status="COMPLETE",sha256=sha(OUT/"RESULT_BUNDLE.zip"),cost=limits()))

if __name__=="__main__":
    OUT.mkdir(exist_ok=False)
    try:
        main()
    except Exception as exc:
        save("FAILURE.json",dict(error=repr(exc)))
        raise
