import json
from pathlib import Path
from bridge import load
from address import ARMS,make,segment

raw=load().base.SOURCE.read_text()
env={"__name__":"parent"}
import os
os.environ.update(SEG="1",HABT="0",WIN="32")
exec(raw[:raw.index("print(json.dumps(")],env)
contexts=0
for length in (0,16,40,64):
    for person in (b"Mary",b"Daniel"):
        history=b"Previous sentence. "+person+b" "+b"x"*length+b" went to the "
        assert person in segment(history)[:16] and person in history[-96:]
        if length>=40:
            assert person not in history[-32:]
        for step in range(3):
            for arm in ARMS:
                env["WIN"]=96 if arm=="RECENT96" else 32
                ids,weights=make(env["bag"],arm)(history,step=step)
                assert 0<len(ids)<=360 and len(ids)==len(set(ids))==len(weights)
                if arm=="RECENT32":
                    assert list(ids)==list(env["bag"](history,step=step)[0])
                if arm.startswith("SPLIT"):
                    assert sum(ids<32768)<=240 and sum(ids>=32768)<=120
                contexts+=1
Path("CYCLE2_TEST.json").write_text(json.dumps(dict(status="PASS",address_checks=contexts))+"\n")
