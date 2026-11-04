import json
from pathlib import Path
from world import generate,official_roster,audit,seed
test=[generate(seed(f"V213|ENGINEERING|INTERFACE|{i}")) for i in range(4)]
official=official_roster()
assert len({w.key for w in test+list(official)})==12
assert {w.attach==w.opcodes[0] for w in test+list(official)}=={False,True}
for w in test+list(official):
    audit(w)
Path("CYCLE1_TEST.json").write_text(json.dumps(dict(status="PASS",engineering_worlds=4,
    dry_official_worlds=8,oracle_twins=384,learner_updates=0))+"\n")
