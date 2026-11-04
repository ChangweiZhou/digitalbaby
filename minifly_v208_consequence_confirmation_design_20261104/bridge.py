import json
from pathlib import Path
from protocol import PARENT, World, manifest, run, sha
manifest()
source=PARENT/"rehearsal/TECH0_RECEIPT.json"
lock=json.loads((PARENT/"rehearsal/CYCLE3_TEST.json").read_text())
assert sha(source)==lock["receipt_hashes"][source.name]
reference=json.loads(source.read_text())
result=run(World(reference["world"]),(4,4,2))
assert json.loads(json.dumps(result))==reference
Path("CYCLE1_TEST.json").write_text(json.dumps(dict(status="PASS",historical_live_actions=160,
    historical_receipt_sha256=sha(source)))+"\n")
