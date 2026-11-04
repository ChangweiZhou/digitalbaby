import json
from pathlib import Path
from protocol import ARMS, HELD, World, decision, parent, run

world=World("V208-TECH-AUDIT")
tape=[e for p,n in zip(parent.PHASES,parent.LENGTHS) for i in range(n) for e in world.block(p,i)]
assert not {world.prompt(c,w) for c in range(4) for w in HELD}.intersection(e.prompt for e in tape)
assert sum(len(e.prompt)+len(e.neutral)+1 for e in tape)==5760
r=run(world,parent.LENGTHS)
stats=parent.audit(world,r)
assert sum(x[0] for x in stats.values())==640
def panels(stage):
    return {a:[{**x,"correct":int(stage!="R0") if a=="ALIGNED" else .5}
               for x in r["stages"][stage][a]] for a in ARMS}
def fixture(safe):
    return [dict(world=i,stages={k:panels(k) for k in parent.STAGES},stats={"R":[256,safe]})
            for i in range(32)]
assert decision(fixture(160))["passed"]
failure=decision(fixture(128))
assert not failure["passed"] and not failure["guards"]["live_safety"]
assert sum(not v for v in failure["guards"].values())==1
try:
    decision(fixture(160)[:-1])
except AssertionError:
    pass
else:
    raise AssertionError("Incomplete roster accepted")
Path("CYCLE2_TEST.json").write_text(json.dumps(dict(status="PASS",live_actions=640,
    consumed_interface_bytes=5760,generated_placeholder_bytes=64,stats=stats))+"\n")
