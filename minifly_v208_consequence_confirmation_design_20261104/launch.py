"""V208 technical entry point. Confirmation schedules are generated dry."""
import json
from pathlib import Path
import resource
import statistics as st
import time
from zipfile import ZipFile, ZIP_DEFLATED

from protocol import HELD, TRAIN, World, manifest, parent, rate, run, seed, sha

OUT=Path("rehearsal")

def write(name, data):
    p=OUT/name
    t=p.with_suffix(".tmp")
    t.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    t.replace(p)
    return sha(p)

def tape(world):
    return [e for p,n in zip(parent.PHASES,parent.LENGTHS) for i in range(n) for e in world.block(p,i)]

def shortcut(world, receipt):
    lookup={world.prompt(c,w):c for c in range(4) for w in TRAIN}
    values=[[.5,.5] for _ in range(4)]
    exact={}
    for block in receipt["history"]:
        for e,a,y in zip(world.block(block["phase"],block["block"]),block["actions"],block["outcomes"]):
            v=values[lookup[e.prompt]]
            v[a]+=.2*(y-v[a])
            exact.setdefault(e.prompt,[[],[]])[a].append(y)
    held=[world.prompt(c,w) for c in (0,1) for w in HELD]
    assert not set(held)&exact.keys()
    def correct(c):
        a=world.preferred(c,True)
        m=values[c][a]-values[c][1-a]
        return 1 if m>0 else 0 if m<0 else .5
    return dict(canonical_recency=st.mean(correct(c) for c in (0,1)),
                uniform=.5,fixed_global=.5,wrapper_only=.5,held_exact_coverage=0)

def execute_world(world):
    receipt=run(world,parent.LENGTHS)
    stats=parent.audit(world,receipt)
    return {**receipt,"stats":stats,"shortcuts":shortcut(world,receipt)}

def main():
    start=time.process_time()
    out_time=time.monotonic()
    lock=manifest()
    historical={r["seed"] for r in lock["roster"]}
    technical=["V208-TECH-AUDIT","V208-TECH-RUN-0","V208-TECH-RUN-1"]
    worlds=[World(seed(f"MINIFLY|V208|CONFIRM|{i}")) for i in range(32)]
    assert len({w.world for w in worlds})==32
    assert not historical.intersection(w.world for w in worlds)
    roster=[]
    import hashlib
    for w in worlds:
        scheduled=tape(w)
        assert len(scheduled)==640 and sum(len(e.prompt)+len(e.neutral)+1 for e in scheduled)==5760
        assert not {w.prompt(c,f) for c in range(4) for f in HELD}.intersection(e.prompt for e in scheduled)
        roster.append(dict(seed=w.world,schedule_sha256=hashlib.sha256(repr(scheduled).encode()).hexdigest()))
    OUT.mkdir(exist_ok=False)
    sources={p:sha(Path(p)) for p in ("DESIGN.md","protocol.py","bridge.py","audit.py","launch.py")}
    write("DRY_LOCK.json",dict(roster=roster,parent_sources=lock["sources"],
        sources=sources,engineering_seeds=technical,episodes=20480,arm_applications=61440,
        updates=40960,probes=6144,interface_bytes=184320,generated_placeholder_bytes=2048))
    commits=[]
    summary=[]
    measured=time.process_time()
    for i,name in enumerate(technical[1:]):
        receipt=execute_world(World(name))
        path=f"TECH_{i}_RECEIPT.json"
        commits.append(dict(path=path,sha256=write(path,receipt)))
        summary.append(dict(held={a:rate(receipt["stages"]["FINAL"],a,True) for a in protocol_arms},
                            reversal_safety=receipt["stats"]["R"][1]/256,shortcuts=receipt["shortcuts"]))
    world_cpu=time.process_time()-measured
    for item in commits:
        assert sha(OUT/item["path"])==item["sha256"]
        r=json.loads((OUT/item["path"]).read_text())
        assert parent.audit(World(r["world"]),r)==r["stats"]
    cumulative=dict(episodes=20480+2080,arm_applications=61440+3*2080,
                    updates=40960+2*2080,probes=6144+3*192+168)
    caps=dict(episodes=24000,arm_applications=72000,updates=48000,probes=10000)
    assert all(cumulative[k]<=v for k,v in caps.items())
    # Conservative multiplier on full-size technical worlds. No new science is performed.
    projection=world_cpu*16*3+30
    wall_projection=(time.monotonic()-out_time)*16*3+30
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert projection<300 and wall_projection<600 and rss<512*2**20
    write("CYCLE3_TEST.json",dict(status="TECHNICAL_PASS",technical_episodes=2080,
        cumulative_with_science=cumulative,engineering_summary=summary,commits=commits,
        projected_cpu=projection,projected_wall=wall_projection,
        cpu_seconds=time.process_time()-start,peak_rss_bytes=rss))
    paths=sorted(OUT.glob("*.json"))
    with ZipFile(OUT/"TECHNICAL_BUNDLE.zip","w",ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p,p.name)
    with ZipFile(OUT/"TECHNICAL_BUNDLE.zip") as z:
        assert z.testzip() is None and all(z.read(p.name)==p.read_bytes() for p in paths)
    print(json.dumps(summary))

if __name__=="__main__":
    from protocol import ARMS as protocol_arms
    main()
