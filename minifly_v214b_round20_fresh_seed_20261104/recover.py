"""Extract only code bodies from the user-supplied session export."""
import ast
import hashlib
import json
from pathlib import Path
import sys

def sha(b):
    return hashlib.sha256(b).hexdigest()

def main():
    raw=Path(sys.argv[1]).read_bytes()
    events=json.loads(raw)
    def extract(index,path):
        e=events[index]
        content=e["input"]["command"].split(f"cat > {path} <<'EOF'\n",1)[1].split("\nEOF\n",1)[0]+"\n"
        return e["id"],content
    sources={}
    for name,i,path in (("cmr.py",3168,"cmr.py"),("ansbits.py",3168,"ansbits.py"),
                        ("world_source.py",3033,"/tmp/mw/miniworld2.py")):
        event,code=extract(i,path)
        if name=="world_source.py":
            # Round 20 executes only this unchanged generator prefix.
            code=code.split("class Learner",1)[0]
        ast.parse(code)
        Path(name).write_text(code)
        sources[name]=dict(event=event,sha256=sha(code.encode()))
    archived={str(i):dict(event=events[i]["id"],text=events[i]["content"][0]["text"])
              for i in (3169,3174)}
    Path("HISTORICAL_RESULTS.json").write_text(json.dumps(archived,indent=2)+"\n")
    Path("RECOVERY.json").write_text(json.dumps(dict(export_sha256=sha(raw),sources=sources),indent=2)+"\n")

if __name__=="__main__":
    main()
