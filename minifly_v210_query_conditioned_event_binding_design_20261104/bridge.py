import collections
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

ROOT=Path(os.environ.get("V209_SOURCE","/private/tmp/minifly-v188-remote/minifly_v209_contextual_answer_selection"))
SOURCE_SHA="f6af81f76724f834e20ca8b5962df6e76438329c07da7d87a09201221c2b0a93"
source=ROOT/"qualify.py"
assert hashlib.sha256(source.read_bytes()).hexdigest()==SOURCE_SHA
spec=importlib.util.spec_from_file_location("v209_source",source)
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

def votes(model,query):
    out=collections.Counter()
    for row in sorted(v.addresses(query)):
        if row in model.store:
            weight=math.log((1+model.questions)/(1+model.df[row]))
            for candidate in model.store[row]:
                out[candidate]+=weight
    return out

def main():
    refs=json.loads((ROOT/"ENGINEERING.json").read_text())["rows"]
    original=v.Learner.recall
    def checked(model,q):
        answer=original(model,q)
        vals=votes(model,q)
        assert sorted(vals,key=lambda c:(-vals[c],c))[:v.CANDIDATES]==answer
        return answer
    v.Learner.recall=checked
    model=v.train("context","true")
    out=[]
    for permute in (False,True):
        ref=next(r for r in refs if r["selector"]=="context" and r["feedback"]=="true"
                 and r["view"]=="joint" and r["permuted"]==permute)
        r=v.evaluate(model,"joint",permute)
        assert all(ref[k]==value for k,value in r.items()), {k:(str(ref[k])[:100],str(value)[:100]) for k,value in r.items() if ref[k]!=value}
        out.append(dict(permuted=permute,correct=r["correct"],n=r["total"]))
    Path("CYCLE1_TEST.json").write_text(json.dumps(dict(status="PASS",source=SOURCE_SHA,
        training_questions=600,views=out))+"\n")

if __name__=="__main__":
    main()
