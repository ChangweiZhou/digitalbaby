from pathlib import Path
import json,hashlib
p=Path(__file__).resolve().parent
m=json.loads((p/"SPLIT_MANIFEST.json").read_text())
out=p.parent/Path(m["original"]).name
if out.exists(): raise SystemExit("Output exists; preserving it: "+str(out))
h=hashlib.sha256()
with out.open("xb") as w:
 for x in m["parts"]:
  data=(p/Path(x["path"]).name).read_bytes()
  assert len(data)==x["bytes"]
  w.write(data);h.update(data)
assert out.stat().st_size==m["original_bytes"]
assert h.hexdigest()==m["original_sha256"]
print(out)
