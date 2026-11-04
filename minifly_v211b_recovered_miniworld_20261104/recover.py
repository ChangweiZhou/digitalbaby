"""Recover code bodies from the session export without executing recorded commands."""
import ast
import hashlib
import json
from pathlib import Path
import sys

def sha(data):
    return hashlib.sha256(data).hexdigest()

export=Path(sys.argv[1]).read_bytes()
events=json.loads(export)

def event(i, expected):
    e=events[i]
    assert e["id"]==expected
    return e["input"]["command"]

cmd=event(3009,"sevt_01TN95GdNDnKvbjKQ4JUJVxf")
base=cmd.split("cat > /tmp/mw/miniworld.py <<'EOF'\n",1)[1].split("\nEOF\n",1)[0]+"\n"
patch=event(3014,"sevt_01HyzXHABoYoG9PJoPFBNyjH")
assert "NOCTX" in patch and "FBS" in patch
replacements=[
    ("        for i,w in enumerate(ws): s.CF[w]=", "        if E('NOCTX')=='1': return\n        for i,w in enumerate(ws): s.CF[w]="),
    ("for fb in ('true','type','none'):", "for fb in E('FBS','true,type,none').split(','):"),
]
final=base
for before,after in replacements:
    assert final.count(before)==1
    final=final.replace(before,after)
ast.parse(final)
Path("miniworld.py").write_text(final)
run=events[3010]
assert run["type"]=="agent.tool_result"
original=run["content"][0]["text"]
reference={}
for line in original.splitlines():
    tag,_,payload=line.partition(" ")
    if tag in ("true","type","none") and payload.startswith("{"):
        reference[tag]=json.loads(payload)
assert set(reference)=={"true","type","none"}
Path("HISTORICAL_RESULTS.json").write_text(json.dumps(reference,indent=2)+"\n")
Path("RECOVERY.json").write_text(json.dumps(dict(export_sha256=sha(export),
    source_sha256=sha(final.encode()),initial_sha256=sha(base.encode()),
    source_event=events[3009]["id"],amendment_event=events[3014]["id"],
    historical_result_event=run["id"],recovered_stage="Round 16 MiniWorld v0"),indent=2)+"\n")
