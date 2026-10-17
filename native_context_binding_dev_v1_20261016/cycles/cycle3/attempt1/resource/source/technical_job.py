"""Short technical histories only. Reject every DEV/CONFIRM world."""
import os
import time
from pathlib import Path
import bootstrap
import torch
from bind_kernel import Learner,ARMS,array_hash
from bind_artifacts import Ledger,write,source_identity
import bind_evaluator as evaluator

TECH_WORLDS=(202610166001,202610166002,202610166003)


def run_job(arm,world,directory):
    if arm not in ARMS or world not in TECH_WORLDS:
        raise ValueError("technical-only admission; no science-world launch")
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/"RESULT.json").exists() or (directory/"EXECUTION_LEDGER.json").exists():
        raise ValueError("never overwrite an existing executed technical job")
    ledger=Ledger(directory);started=time.time()
    model=Learner(arm,world)
    initial=model.digest()
    old=evaluator.stream(world,"old",4);new=evaluator.stream(world,"new",4)
    groups=evaluator.split(world)
    held=[(f"{a}+{b}=".encode(),str(a+b).encode()) for a,b in groups["held"][:2]]
    assert not {p for p,_ in old+new}&{p for p,_ in held}
    histories={}
    for history in ("W","N_OLD"):
        current=model.clone()
        assert current.digest()==initial
        updates=[]
        old_allowed=history=="W"
        theta0=current.theta_digest();state0=current.state.detached(True);time0=current.time
        updates.append(ledger.call(history+"-OLD",current,lambda:current.train_window(old,old_allowed)))
        if not old_allowed:
            assert theta0==current.theta_digest()
            expected=current.anatomy.rest(state0,current.time-time0)
            assert torch.allclose(current.state.fast,expected.fast,atol=1e-9,rtol=1e-9)
            assert torch.allclose(current.state.slow,expected.slow,atol=1e-9,rtol=1e-9)
        current.rest(86400)
        cpu0=time.process_time();before=evaluator.score(current,held)
        probe_cpu=time.process_time()-cpu0
        cursor=dict(next_record=4,stage="NEW",history=history,world=world,arm=arm)
        path=directory/f"{history}.pt"
        digest=current.digest();current.checkpoint(path,cursor)
        restored,got_cursor=Learner.restore(path)
        assert got_cursor==cursor and restored.digest()==digest
        # Continue from exactly the next unexecuted record, never replay OLD.
        current=restored
        pre_theta=current.theta_digest()
        updates.append(ledger.call(history+"-NEW",current,lambda:current.train_window(new,True)))
        assert current.resource["records"]==8
        if current.learns:
            assert pre_theta!=current.theta_digest()
        current.rest(86400)
        cpu0=time.process_time();after=evaluator.score(current,held);probe_cpu+=time.process_time()-cpu0
        legacy=None
        if arm=="NATIVE":
            cpu0=time.process_time();legacy=evaluator.score(current,held,"LEGACY");probe_cpu+=time.process_time()-cpu0
            r=current.resource
            assert abs(r["logical_MEM_cpu"]+r["legacy_readout_cpu"]-r["production_cpu"])<1e-8
            assert abs(r["logical_LEGACY_cpu"]+r["mem_readout_cpu"]-r["production_cpu"])<1e-8
        histories[history]=dict(updates=updates,before=before,after=after,legacy_final=legacy,
            final_digest=current.digest(),records=current.resource["records"],
            resource=current.resource.copy(),probe_cpu=probe_cpu,checkpoint_digest=digest)
    result=dict(passed=True,arm=arm,world=world,histories=histories,
                records_executed=16,independent_histories=2,science_jobs=0,
                behavioral_classification="TECHNICAL_FIXTURE_ONLY_NOT_EFFICACY",
                worker_pid=os.getpid(),elapsed_seconds=time.time()-started,source_identity=source_identity())
    write(directory/"RESULT.json",result)
    return dict(arm=arm,passed=True,records=16,pid=os.getpid(),seconds=time.time()-started)
