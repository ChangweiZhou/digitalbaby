"""Read-only science accounting after the engineering run stopped at admission."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from world import official_roster,audit

OUT=Path("rehearsal")

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    files=[OUT/"TECH_0.json",OUT/"TECH_1.json"]
    rows=[json.loads(p.read_text()) for p in files]
    packets=sum(w["training_packets"] for w in rows)
    assert packets==80
    cumulative=dict(records=100352+8*12*4+packets*12*7+2*64*7*10,
                    outcomes=10752+64+packets*2*7,probes=5120+2*64*10)
    assert cumulative==dict(records=116416,outcomes=11936,probes=6400)
    roster=official_roster()
    for w in roster:
        audit(w)
    src=("DESIGN.md","world.py","learner.py","train.py","test_world.py","test_learning.py",
         "rehearse.py","finalize_rehearsal.py")
    result=dict(status="TECHNICAL_COMPLETE_OFFICIAL_NOT_ADMITTED",official_worlds_opened=0,
        reason="Cumulative record cap exceeded",record_cap=115000,cumulative_with_official=cumulative,
        source_hashes={p:sha(Path(p).read_bytes()) for p in src},
        dry_roster=[dict(seed=w.key,sha256=sha(repr(w).encode())) for w in roster],
        receipts={p.name:sha(p.read_bytes()) for p in files},
        scores=[{a:r["scores"] for a,r in w["rows"].items()} for w in rows])
    summary=OUT/"CYCLE3_TEST.json"
    summary.write_text(json.dumps(result,indent=2)+"\n")
    files.extend((summary,Path("CYCLE1_TEST.json"),Path("CYCLE2_TEST.json")))
    bundle=OUT/"TECHNICAL_BUNDLE.zip"
    with ZipFile(bundle,"w",ZIP_DEFLATED) as z:
        for p in files:
            z.write(p,p.name)
    with ZipFile(bundle) as z:
        assert z.testzip() is None and all(z.read(p.name)==p.read_bytes() for p in files)
    print("Verified bundle",sha(bundle.read_bytes()))

if __name__=="__main__":
    main()
