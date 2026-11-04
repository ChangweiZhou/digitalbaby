"""Dry regeneration of the original world's declared vocabularies and templates."""
import json
import os
from pathlib import Path
import random

assert os.environ["PYTHONHASHSEED"]=="1"
prefix=Path("miniworld.py").read_text().split("K=int(E('K','30'))",1)[0]
ctx={}
exec(compile(prefix,"miniworld.py","exec"),ctx)
rows=[]
for s in range(10,20):
    r=random.Random(s)
    prior=[ctx["world"](r,ctx["POOL_TRAIN"]) for _ in range(30)]
    target=ctx["world"](random.Random(10000+s),ctx["POOL_TEST"])
    known={w for world in prior for words in world[:2] for w in words}
    folded={w.lower() for w in known}
    tokens=[x for words in target[:2] for x in words]
    assert not set(target[2])&set(ctx["POOL_TRAIN"])
    rows.append(dict(seed=s,prior_token_overlap=[x for x in tokens if x.lower() in folded],
                     exact_case_overlap=[x for x in tokens if x in known]))
out=dict(target_template_disjoint=True,worlds_with_prior_token_overlap=sum(bool(r["prior_token_overlap"]) for r in rows),
         worlds_with_exact_case_overlap=sum(bool(r["exact_case_overlap"]) for r in rows),worlds=rows)
Path("EXPOSURE_AUDIT.json").write_text(json.dumps(out,indent=2)+"\n")
print(out["worlds_with_exact_case_overlap"])
