"""Run the frozen A comparison and score conditional read-only endpoints."""
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import resource
import sys
import time

from qualify_worked import (ARMS, MODES, EOS, GRAPHS, Model, T, digest, fixture,
                            parse, render, sequence, structure_key, truth, update)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "official"
BLOCKS = (93157, 104729)
DOSES = (2048,8192)
FLOORS = (52,48)


def write(path,obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,indent=2)+"\n")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def receipt():
    files = [Path(__file__),ROOT/"qualify_worked.py",
             ROOT.parent/"minifly-ordered-front-end"/"revised_panel.py",
             ROOT.parent/"minifly-ordered-front-end"/"adaptive_audit.py",
             ROOT.parent/"minifly-reasoning-scaling"/"benchmark.py",
             ROOT.parent/"minifly-reasoning-scaling"/"task.py",
             ROOT.parent/"minifly-whole-repo-audit"/"MINIFLY_RECURRENT_WORKED_EXAMPLES_EXPERIMENT_DESIGN_20261106.md"]
    return {str(p.relative_to(ROOT.parent)):sha256(p.read_bytes()).hexdigest() for p in files}


def held_all(block, family, label):
    graph = GRAPHS[family]
    gold = truth(graph)
    rng = Random(f"{block}:{family}:{label}:all-masks")
    for mask in range(256):
        actors = rng.sample(range(128,136),8)
        places = rng.sample(range(48,58),10)
        order = rng.sample(range(8),8)
        prefix = render(graph,mask,b"XY",actors,places,order)
        assert structure_key(parse(prefix)) not in ((1,2),(2,),(3,))
        moves = [gold[mask],gold[mask^255]]
        yield prefix, [places[8 if m else 3] for m in moves], moves


def e2(model, probes, law):
    records=[]
    before=digest(model)
    for key,ps,ys in probes:
        target=[ys[law],EOS]
        prompts=[ps[0],ps[2],ps[3]]
        answers=[model.generate(p) for p in prompts]
        records.append(dict(key=key,prefixes=[p.hex() for p in prompts],target=target,answers=answers))
    assert digest(model)==before
    return dict(keys=len(records),taught=sum(r["answers"][0]==r["target"] for r in records),
                held_both=sum(r["answers"][1:]==[r["target"]]*2 for r in records),
                first=sum(a[0]==r["target"][0] for r in records for a in r["answers"]),
                eos=sum(a[-1]==EOS for r in records for a in r["answers"]),
                cap=sum(len(a)==16 and a[-1]!=EOS for r in records for a in r["answers"]),
                raw=records)


def e3(model,block,law,label):
    reports=[]
    before=digest(model)
    for i in range(len(GRAPHS)):
        raw=[]
        count=[0,0]
        correct=[0,0]
        for prefix,ys,moves in held_all(block,i,label):
            answer=model.generate(prefix)
            target=[ys[law],EOS]
            count[moves[law]]+=1
            correct[moves[law]]+=answer==target
            raw.append(dict(prefix=prefix.hex(),target=target,flip=ys[0]!=ys[1],output=answer))
        reports.append(dict(correct=correct,count=count,
                            balanced_exact=sum(x/n for x,n in zip(correct,count))/2,raw=raw))
    assert digest(model)==before
    return reports


def prepare():
    from adaptive_audit import balanced_accuracy_ceiling
    OUT.mkdir(exist_ok=True)
    path=OUT/"LOCK.json"
    assert not path.exists()
    blocks={}
    for block in BLOCKS:
        blocks[str(block)]={}
        for phase in ("A","B"):
            bank,probes=fixture(block,phase)
            feedback={p for _,p,_ in bank}
            assert len(feedback)==512 and len(probes)==64
            assert all(p not in feedback for _,ps,_ in probes for p in ps[2:])
            streams={}
            for law in range(2):
                for arm in ARMS:
                    rows=sequence(bank,block,law,arm,phase,8192 if phase=="A" else 2048)
                    streams[f"{law}-{arm}"]=sha256(bytes() .join(
                        bytes(x%256 for x in a+b) for a,b in rows)).hexdigest()
            blocks[str(block)][phase]={"stream_sha256_eos_mod256":streams,
                 "held_presentation_sha256":sha256(b"".join(p for _,ps,_ in probes for p in ps)).hexdigest()}
        blocks[str(block)]["E3_sha256"]={label:sha256(b"".join(
            p+bytes(ys) for i in range(2) for p,ys,_ in held_all(block,i,label))).hexdigest()
            for label in ("A","post_B")}
    write(path,dict(sources=receipt(), blocks=blocks,doses=DOSES, floors=FLOORS,
                    adaptive_ceiling=[balanced_accuracy_ceiling(g)["3"] for g in GRAPHS],
                    planned_A_pairs=196608, planned_A_symbols=24182784,
                    planned_cpu_limit_seconds=2400,planned_worker_rss_limit_mib=1024,
                    qualification_cpu_seconds=read(ROOT/"CYCLE3.json")["cpu_seconds"]))
    print("locked",flush=True)


def run(block,mode,arm):
    lock=read(OUT/"LOCK.json")
    assert lock["sources"]==receipt() and block in BLOCKS and mode in MODES and arm in ARMS
    directory=OUT/f"{block}-{mode}-{arm}"
    directory.mkdir(exist_ok=True)
    bank,probes=fixture(block,"A")
    start=time.process_time()
    for law in range(2):
        model=Model(mode,block)
        opt=T.optim.Adam(model.parameters(),lr=.003)
        stream=list(sequence(bank,block,law,arm,"A",8192))
        recent=0
        for step,row in enumerate(stream,1):
            recent+=update(model,opt,row,arm)
            if step%512==0:
                print(block,mode,arm,law,step,round(recent/512,4),flush=True)
                recent=0
            if step in DOSES:
                path=directory/f"law{law}-{step}.pt"
                tmp=path.with_suffix(".tmp")
                T.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),tmp)
                tmp.replace(path)
                score=e2(model,probes,law)
                score.update(state=digest(model),dose=step,law=law)
                write(path.with_suffix(".json"),score)
    write(directory/"COMPLETE.json",dict(pairs=16384,cpu_seconds=time.process_time()-start,
            rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2))


def summarize():
    lock=read(OUT/"LOCK.json")
    assert lock["sources"]==receipt()
    rows=[]
    winners=[]
    complete=[]
    for mode in MODES:
        for arm in ARMS:
            eligible=True
            for b in BLOCKS:
                d=OUT/f"{b}-{mode}-{arm}"
                complete.append(read(d/"COMPLETE.json"))
                for law in range(2):
                    for dose in DOSES:
                        r=read(d/f"law{law}-{dose}.json")
                        assert r["keys"]==64 and r["dose"]==dose and r["law"]==law
                        rows.append(dict(block=b,mode=mode,arm=arm,**{k:v for k,v in r.items() if k!="raw"}))
                        if dose==8192:
                            eligible &= r["taught"]>=52 and r["held_both"]>=48
            if eligible:
                winners.append([mode,arm])
    assert sum(r["pairs"] for r in complete)==196608
    obj=dict(status="A_COMPLETE",pairs=196608,delivered_symbols=24182784,results=rows,
             E2_eligible=winners,cpu_seconds=sum(x["cpu_seconds"] for x in complete),
             max_worker_rss_mib=max(x["rss_mib"] for x in complete))
    if not winners:
        obj["status"]="COMPLETE_STOPPED_AT_A"
    write(OUT/"SUMMARY.json",obj)
    print(obj["status"],winners,obj["cpu_seconds"],flush=True)


if __name__=="__main__":
    action=sys.argv[1]
    if action=="prepare":
        prepare()
    elif action=="summarize":
        summarize()
    else:
        run(int(sys.argv[2]),*sys.argv[3:5])
