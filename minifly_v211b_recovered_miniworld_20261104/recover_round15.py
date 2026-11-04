"""Apply only the logged literal string replacements to the recovered Round 14 parent."""
import ast
import hashlib
import json
from pathlib import Path
import sys

export, parent=map(Path,sys.argv[1:])
events=json.loads(export.read_bytes())
code=parent.read_text()
before=hashlib.sha256(parent.read_bytes()).hexdigest()
assert before=="e18c8e797f42e3189340aa9d92666b45168d6c2ba65bda3b3d8a2b9d73db665a"
indices=(2980,2985)
ids=("sevt_01Hp6ZKFuUTbidRYWDZx1Zs2","sevt_0129x72ev49wCJBW2FAdfJgg")
counts=[]
for i,expected in zip(indices,ids):
    assert events[i]["id"]==expected
    cmd=events[i]["input"]["command"]
    tree=ast.parse(cmd.split("python - <<'EOF'\n",1)[1].split("\nEOF\n",1)[0])
    replacements=0
    for statement in tree.body:
        if not isinstance(statement,ast.Assign) or not isinstance(statement.value,ast.Call):
            continue
        call=statement.value
        if not isinstance(call.func,ast.Attribute) or call.func.attr!="replace":
            continue
        old,new=map(ast.literal_eval,call.args)
        assert code.count(old)==1,old
        code=code.replace(old,new)
        replacements+=1
    counts.append(replacements)
ast.parse(code)
Path("round15_learner.py").write_text(code)
Path("ROUND15_RECOVERY.json").write_text(json.dumps(dict(
    parent_sha256=before,recovered_sha256=hashlib.sha256(code.encode()).hexdigest(),
    amendment_events=ids,replacements=counts,status="SOURCE_RECONSTRUCTED_BEHAVIOR_NOT_QUALIFIED"),indent=2)+"\n")
