"""Admit and supervise the fixed V216 development roster."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
PARENT=BASE/"minifly_v181_mini_local_plasticity_dev_20261102"
PKG=BASE/"minifly_a_v3/package"
BUDGET=dict(events=1600000,observed_bytes=60000000,target_sites=64000000,
            cpu_seconds=28800,wall_seconds=36000,rss_bytes=1<<30,output_bytes=256<<20)
PRE=ROOT/"prelaunch_final_20261104"
OUT=ROOT/"campaign_20261104"


def read(path):
    return json.loads(path.read_text())


def write(path,obj):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def plan():
    pilots=[read(PRE/f"family_{i}.json") for i in range(2)]
    parity=read(ROOT/"prelaunch_20261104/PARITY.json")
    assert parity["bitwise_training_parity"] and parity["events"]==1200
    assert all(r["trial"] and r["tally"]["events"]==512 for r in pilots)
    compute=max(sum(a["train_cpu"] for a in row["arms"].values())/512 for row in pilots)
    projection=(compute*1600000 + max(r["cpu_seconds"] for r in pilots)*8*3)*1.25
    extras={k:max(row["tally"][k]-512*v for row in pilots)*8*3
            for k,v in (("observed_bytes",36),("target_sites",38))}
    projected=dict(cpu_seconds=projection,observed_bytes=1600000*36+extras["observed_bytes"],
                   target_sites=1600000*38+extras["target_sites"],
                   rss_bytes=sum(r["rss_bytes"] for r in pilots)+(64<<20))
    assert all(value < BUDGET[k] for k,value in projected.items()),projected
    paths=[ROOT/x for x in ("DESIGN.md","campaign.py","launch.py","engine.py","fast.py","run.py","environment.py")]
    paths += [PARENT/x for x in ("learner.py","sensor.py")]
    paths += [PKG/x for x in ("FULL151_BIRTH_FINGERPRINT.json","FULL151_CANONICAL_B.npz",
                             "REFERENCE_SOURCE/minifly/iteration10/models/reference.npz")]
    return dict(status="ADMITTED",worlds=list(range(8)),workers=2,checkpoints=[1000,10000,100000],
                budgets=BUDGET,projected=projected,python=sys.executable,
                sources={str(p.relative_to(BASE)):digest(p) for p in paths},
                scope="fixed V216 development roster; no million-event escalation")


def summarize(rows):
    import numpy as np
    from scipy.stats import t

    def contrast(values):
        mean=float(np.mean(values))
        radius=float(t.ppf(.975,7)*np.std(values,ddof=1)/math.sqrt(8))
        return dict(mean=mean,ci95=[mean-radius,mean+radius],families=values)

    dose, diversity, adapt, write_effect=[],[],[],[]
    for row in rows:
        few,many=(row["arms"][name] for name in ("FEW","MANY"))
        def acc(arm,k):
            return arm["checkpoints"][str(k)]["shared"]["exact"]/32
        def avg(panel):
            return sum(panel["changed"][str(k)]["exact"] for k in (4,16,64))/96
        dose.append(acc(many,100000)-acc(many,1000))
        diversity.append(acc(many,100000)-acc(few,100000))
        warm=avg(many["adaptation"]["matched"])
        adapt.append(warm-avg(row["cold_adapt"]["matched"]))
        write_effect.append(warm-avg(many["adaptation"]["frozen"]))
    return dict(dose=contrast(dose),diversity=contrast(diversity),
                adaptation=contrast(adapt),write_effect=contrast(write_effect))


def run():
    lock=read(OUT/"LOCK.json")
    assert all(digest(BASE/p)==v for p,v in lock["sources"].items())
    active={}
    done=[]
    pending=list(lock["worlds"])
    start=time.monotonic()
    def status(phase,**kw):
        write(OUT/"STATUS.json",dict(state=phase,committed=done,active={k:p.pid for k,(p,_) in active.items()},**kw))
    try:
        while pending or active:
            while pending and len(active)<2:
                i=pending.pop(0)
                log=(OUT/f"worker_{i}.log").open("w")
                active[i]=(subprocess.Popen([sys.executable,"-B",str(ROOT/"campaign.py"),str(i),str(OUT)],
                                           stdout=log,stderr=subprocess.STDOUT,cwd=ROOT),log)
            assert time.monotonic()-start<BUDGET["wall_seconds"],"wall cap"
            assert sum(p.stat().st_size for p in OUT.iterdir())<BUDGET["output_bytes"],"output cap"
            for i,(p,log) in list(active.items()):
                code=p.poll()
                if code is None:
                    continue
                log.close()
                del active[i]
                if code!=0:
                    raise RuntimeError(f"family {i} exit {code}; inspect worker_{i}.log")
                receipt=read(OUT/f"family_{i}.json")
                assert receipt["index"]==i and not receipt["trial"] and receipt["tally"]["events"]==200000
                done.append(i)
            status("RUNNING")
            if active:
                time.sleep(2)
        rows=[read(OUT/f"family_{i}.json") for i in range(8)]
        totals={k:sum(row["tally"][k] for row in rows)
                for k in ("events","observed_bytes","target_sites")}
        totals["cpu_seconds"]=sum(row["cpu_seconds"] for row in rows)
        assert totals["events"]==1600000 and all(v<=BUDGET[k] for k,v in totals.items())
        metrics=summarize(rows)
        decision=(metrics["dose"]["mean"]>=.0625 and metrics["dose"]["ci95"][0]>0
                  and sum(x>0 for x in metrics["dose"]["families"])>=6
                  and metrics["adaptation"]["ci95"][0]>-.0625)
        result=dict(totals=totals,metrics=metrics,eligible_for_followup=decision,
                    receipts={f"family_{i}.json":digest(OUT/f"family_{i}.json") for i in range(8)})
        write(OUT/"SUMMARY.json",result)
        (OUT/"REPORT.md").write_text("# V216 eight-family development result\n\n"+
            ("Scaling and adaptation screens passed." if decision else "The prespecified follow-up screen was not met.")+
            "\n\nSee SUMMARY.json for paired family contrasts, intervals, and accounting.\n")
        with zipfile.ZipFile(OUT/"RESULT_BUNDLE.zip","w",zipfile.ZIP_DEFLATED) as archive:
            for name in (*result["receipts"],"SUMMARY.json","REPORT.md","LOCK.json"):
                archive.write(OUT/name,name)
        with zipfile.ZipFile(OUT/"RESULT_BUNDLE.zip") as archive:
            assert archive.testzip() is None
        status("COMPLETED",totals=totals)
    except BaseException as error:
        for p,log in active.values():
            if p.poll() is None:
                p.terminate()
            log.close()
        status("FAILED",error=str(error))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("admit","run"))
    if parser.parse_args().mode=="admit":
        lock=plan()
        OUT.mkdir(exist_ok=False)
        write(OUT/"LOCK.json",lock)
        print(json.dumps(lock["projected"]))
    else:
        run()
