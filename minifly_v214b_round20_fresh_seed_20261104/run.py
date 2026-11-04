"""Fresh-seed reproduction of the Round 20 MiniWorld recall and transfer rosters."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
from world_source import world,stream,MOVE_TR,MOVE_TE

OUT=Path("results")
ENV=dict(os.environ,PYTHONHASHSEED="7",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",
         VECLIB_MAXIMUM_THREADS="1",LAM="0.7")

def sha(b):
    return sha256(b).hexdigest()

def render(r,w,nq):
    out=[]
    ans=[]
    length=0
    for k,x in stream(r,w,nq):
        if k=="s":
            s=x+"\n"
        else:
            q,a,kind=x
            ans.append((length+len(q)+1,len(a),kind))
            s=q+" "+a+"\n"
        out.append(s)
        length+=len(s)
    return "".join(out).encode("ascii"),ans

def write(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")

def execute(item):
    stem,mode=item
    p=OUT/(stem+".txt")
    r=subprocess.run([sys.executable,"-B","cmr.py",str(p)],env=dict(ENV,REC=mode),
                     capture_output=True,text=True,check=True)
    row=json.loads(r.stdout)
    (OUT/f"{stem}_{mode}.log").write_text(r.stdout+r.stderr)
    bits=np.load(str(p)+f".bits_{mode}.npy",allow_pickle=False)
    assert bits.shape==(p.stat().st_size,) and np.isfinite(bits).all() and (bits>=0).all()
    assert round(float(bits.mean()),4)==row["bpb"]
    print("Committed",stem,mode,row["bpb"],flush=True)
    return stem,mode,row,bits

def spans(bits,ans):
    return {kind:float(np.mean([bits[p:p+l].sum() for p,l,k in ans if k==kind]))
            for kind in ("per","obj")}

def main():
    assert os.environ["PYTHONHASHSEED"]=="7"
    OUT.mkdir(exist_ok=False)
    r=random.Random(17)
    log,marks=[],[]
    at=0
    for _ in range(25):
        b,a=render(r,world(r,MOVE_TR),60)
        log.append(b)
        marks.extend((p+at,l,k) for p,l,k in a)
        at+=len(b)
    (OUT/"mw.txt").write_bytes(b"".join(log))
    r=random.Random(23)
    prior=[world(r,MOVE_TR) for _ in range(24)]
    # Preserve the original interleaving of world generation and render RNG consumption.
    r=random.Random(23)
    prior=[]
    hist=[]
    for _ in range(24):
        w=world(r,MOVE_TR)
        prior.append(w)
        hist.append(render(r,w,60)[0])
    tw=world(random.Random(197),MOVE_TE)
    target,ans=render(random.Random(198),tw,80)
    old={x.lower() for w in prior for xs in w[:3] for x in xs}
    overlap=sorted({x for xs in tw[:3] for x in xs if x.lower() in old})
    offsets={}
    for k in (0,2,8,24):
        pre=b"".join(hist[:k])
        offsets[k]=len(pre)
        (OUT/f"tr_{k}.txt").write_bytes(pre+target)
    jobs=[("mw",r) for r in ("0","shuf","1")]+[(f"tr_{k}","1") for k in offsets]
    write("LOCK.json",dict(seeds=dict(mixed=17,history=23,target_world=197,target_stream=198,python_hash=7),
        files={p.name:dict(sha256=sha(p.read_bytes()),bytes=p.stat().st_size) for p in OUT.glob("*.txt")},
        cmr_sha256=sha(Path("cmr.py").read_bytes()),target_sha256=sha(target),offsets=offsets,
        declared_target_token_overlap=overlap,exact_template_overlap=sorted(set(tw[3])&set(MOVE_TR)),
        jobs=jobs,answer_span_rule="Payload bytes, original Round 20; delimiter excluded"))
    write("SPAN_INDEX.json",dict(mixed=marks,target=ans))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(execute,jobs))
    mixed={}
    transfer={}
    for stem,mode,row,bits in results:
        if stem=="mw":
            mixed[mode]=dict(**row,answer_bits=[spans(bits,a) for a in (marks[:len(marks)//2],marks[len(marks)//2:])])
        else:
            k=int(stem.split("_")[1])
            o=offsets[k]
            assert (OUT/(stem+".txt")).read_bytes()[o:]==target
            transfer[k]=dict(history_bytes=o,history_bits=float(bits[:o].sum()),
                             target_bytes=len(target),target_bits=float(bits[o:].sum()),
                             answer_bits=spans(bits[o:],ans))
    base=transfer[0]["target_bits"]
    for r in transfer.values():
        r["gain_bits"]=base-r["target_bits"]
        r["own_code_ratio"]=r["gain_bits"]/r["history_bits"] if r["history_bits"] else None
    write("SUMMARY.json",dict(status="FRESH_SEED_DEVELOPMENT_COMPLETE",mixed=mixed,transfer=transfer,
        declared_target_token_overlap=overlap,charged_bytes=sum((OUT/(stem+".txt")).stat().st_size for stem,_ in jobs),
        prose_run=False))

if __name__=="__main__":
    main()
