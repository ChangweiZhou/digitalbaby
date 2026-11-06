"""Shared receipts and native scoring for the official development runs."""
from hashlib import sha256
import json
from pathlib import Path
import time
import resource

from models import T,digest

ROOT=Path(__file__).resolve().parent
RUN=ROOT/"official"
SEEDS=(73129,93683,116089)
CAP=8


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def sources(runner):
    base=ROOT.parent
    paths=[ROOT/n for n in ("fixtures.py","models.py","qualify_remaining.py","campaign_common.py",runner)]
    paths.extend(base/n for n in ("minifly-recurrent-worked-examples/qualify_worked.py",
        "minifly-ordered-front-end/qualify.py","minifly-ordered-front-end/revised_panel.py",
        "minifly-reasoning-scaling/backend.py","minifly-reasoning-scaling/task.py",
        "minifly-reasoning-scaling/benchmark.py","minifly-reasoning-scaling/cycles.py",
        "minifly-whole-repo-audit/MINIFLY_THREE_REMAINING_EXPERIMENTS_DESIGN_20261106.md"))
    return {str(p.relative_to(base)):sha(p) for p in paths}


def lock(root,runner,archive,spec):
    from cycles import source_receipt
    data=dict(seeds=SEEDS,sources=sources(runner),archive=source_receipt(archive),**spec)
    path=root/"LOCK.json"
    if path.exists():
        assert read(path)==json.loads(json.dumps(data))
    else:write(path,data)


def verify(root,runner,archive=None):
    data=read(root/"LOCK.json")
    assert data["sources"]==sources(runner)
    if archive is not None:
        from cycles import source_receipt
        assert data["archive"]==source_receipt(archive)
    return data


def checkpoint(path,model,opt,step,cpu_seconds=0):
    tmp=path.with_suffix(".tmp")
    T.save(dict(model=model.state_dict(),opt=opt.state_dict(),step=step,cpu_seconds=cpu_seconds),tmp)
    tmp.replace(path)


def restore(path,model,opt):
    data=T.load(path,weights_only=True,map_location="cpu")
    model.load_state_dict(data["model"])
    opt.load_state_dict(data["opt"])
    return data["step"],data["cpu_seconds"]


def usage(start):
    return dict(cpu_seconds=time.process_time()-start,
                rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)


@T.no_grad()
def responses(model,prompts,cap=CAP,batch=64):
    assert len({len(p) for p in prompts})==1
    before=digest(model)
    outputs=[]
    for i in range(0,len(prompts),batch):
        rows=[[257,*p] for p in prompts[i:i+batch]]
        out=[[] for _ in rows]
        active=list(range(len(rows)))
        for _ in range(cap):
            if not active: break
            values=model(T.tensor([rows[j] for j in active]))[:,-1].argmax(1).tolist()
            next_active=[]
            for j,y in zip(active,values):
                out[j].append(y)
                if y!=256:
                    rows[j].append(y)
                    next_active.append(j)
            active=next_active
        outputs.extend(out)
    assert digest(model)==before
    return outputs


def measures(prompts,targets,outputs):
    return dict(n=len(outputs),exact=sum(x==y for x,y in zip(outputs,targets)),
                first=sum(x[0]==y[0] for x,y in zip(outputs,targets)),
                eos=sum(x[-1]==256 for x in outputs),
                rows=[dict(prompt=bytes(p).hex(),target=y,output=x) for p,y,x in zip(prompts,targets,outputs)])
