import json
import math
from pathlib import Path
from protocol import ARMS,CHECKPOINTS,ByteCounter,decision,efficiency,score,uniform_word_bits

history=b"earlier byte stream|"
target=b"query|attic|"
alt=target[:5]+b"another suffix"
root=ByteCounter()
score(root,history)
checkpoint=root.snapshot()
a=score(ByteCounter.restore(checkpoint),target,((6,12,"answer"),),True)
b=score(ByteCounter.restore(checkpoint),alt,capture=True)
assert a["predictions"][:6]==b["predictions"][:6]
assert root.snapshot()==checkpoint
joined=score(ByteCounter(),history+target,capture=True)
assert joined["predictions"][len(history):]==a["predictions"]
options=[x+b"|" for x in (b"attic",b"arcade",b"barn",b"cellar",b"deck",b"den")]
assert all(math.isclose(uniform_word_bits(x,options),math.log2(6)) for x in options)

early=efficiency(100,80,1024,40)
late=efficiency(100,75,3072,80)
assert early["gain_bits"]<late["gain_bits"] and early["own_code_ratio"]>late["own_code_ratio"]
assert efficiency(100,100,0,0)["own_code_ratio"] is None

def fixture(tied=False,decline=False):
    bits={"BASE":[100,100,100,100],"RECALL":[100,70 if decline else 90,85,80],
          "CUE-SHAM":[100,100,100,80 if tied else 100]}
    return [dict(id=i,arms={a:{k:dict(bits=b,bytes=100,sha256=f"T{i}")
            for k,b in zip(CHECKPOINTS,bits[a])} for a in ARMS}) for i in range(16)]
good=[dict(BASE=2.,RECALL=2.01)]*8
bad=[dict(BASE=2.,RECALL=2.03)]*8
assert decision(fixture(),good)["recall_pass"]
assert not decision(fixture(tied=True),good)["recall_pass"]
assert decision(fixture(decline=True),good)["eligible_to_design_stage_b"]
assert not decision(fixture(decline=True),bad)["eligible_to_design_stage_b"]
invalid=fixture()
invalid[0]["arms"]["BASE"][24]["sha256"]="different"
try:
    decision(invalid,good)
except AssertionError:
    pass
else:
    raise AssertionError("unpaired target accepted")
Path("CYCLE2_TEST.json").write_text(json.dumps(dict(status="PASS",backend="BYTE_COUNTER_TEST_DOUBLE",
    uniform_six_word_bits=math.log2(6),parent_byte_applications=0))+"\n")
