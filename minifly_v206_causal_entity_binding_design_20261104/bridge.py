"""V204 engineering bridge. Parent paths are isolated from published files."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
UPSTREAM=Path(os.environ.get("V204_SOURCE","/private/tmp/minifly-v188-remote"))
CACHE=Path(os.environ["V206_CACHE"]).resolve()
PARENT=CACHE/"parent"
PATHS={
    "qualify.py":("minifly_v204_delayed_write_selection/qualify.py","v204/qualify.py"),
    "official.py":("minifly_v204_delayed_write_selection/official.py","v204/official.py"),
    "babi16.py":("minifly_v203_babi_qa1_binary_gate/babi16.py","minifly-v203/babi16.py"),
    "qa1_single-supporting-fact_train.txt":("minifly_v202_babi_qa1/data/qa1_single-supporting-fact_train.txt",
                                         "minifly-v202/data/qa1_single-supporting-fact_train.txt"),
}
GENERAL="minifly_qa1_frozen_generalization/qualify.py"

def digest(p):
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load():
    spec=importlib.util.spec_from_file_location("v204",PARENT/"v204/qualify.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    root=UPSTREAM/"minifly_v204_delayed_write_selection"
    plan=json.loads((root/"OFFICIAL_PLAN.json").read_text())
    hashes={}
    for key,(src,dst) in PATHS.items():
        path=UPSTREAM/src
        assert digest(path)==plan["sources"][key]
        target=PARENT/dst
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        hashes[src]=digest(path)
    target=PARENT/"minifly-qa-generalization/qualify.py"
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(UPSTREAM/GENERAL,target)
    hashes[GENERAL]=digest(target)
    v=load()
    folder=CACHE/"fixture"
    folder.mkdir()
    v.base.materialize(302,folder)
    row=v.run(("position","original",folder,CACHE/"parent.npz",None,1,1))
    published=json.loads((root/"ENGINEERING.json").read_text())
    ref=next(x for x in published["rows"] if x["arm"]=="position" and x["view"]=="original")
    assert row==ref
    assert row["gate_audit"]["hit"]==200 and row["gate_audit"]["miss"]==0
    (HERE/"CYCLE1_TEST.json").write_text(json.dumps(dict(status="PASS",sources=hashes,
        checkpoint_sha256=digest(CACHE/"parent.npz"),n=row["n"],accuracy=row["exact_answer_with_own_newline"],
        write_audit=row["gate_audit"],published_row_exact=True),indent=2)+"\n")

if __name__=="__main__":
    main()
