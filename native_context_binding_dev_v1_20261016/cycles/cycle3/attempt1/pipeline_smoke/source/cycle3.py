"""Four-worker technical pipeline, checkpoint, scorer and ledger qualification."""
import argparse
import hashlib
import math
import multiprocessing
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import bootstrap
import numpy as np
import torch
from bind_kernel import Learner,ARMS,tensor
from bind_artifacts import Ledger,write,source_identity,admit_technical
from technical_job import run_job
import bind_evaluator as evaluator


def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    consumed=admit_technical()
    if consumed+128+4>256:
        raise RuntimeError("not enough remaining technical budget for fixed cycle3")
    start=time.time();checks={};ledger=Ledger(out)
    snapshot=source_identity()
    try:
        # Grade exact output and EOS, not equality before a false stop.
        assert evaluator.exact_correct(b"11","EOS",b"11")
        assert not evaluator.exact_correct(b"11","TRUNCATED",b"11")
        assert not evaluator.exact_correct(b"1","EOS",b"11")
        assert not evaluator.exact_correct(b"11x","EOS",b"11")
        assert abs(evaluator.sample_auc([(0,0),(3,1)])-.5)<1e-12
        assert abs(evaluator.cpu_auc([(0,0),(1,1)],2)-.5)<1e-12
        checks["scorer_and_auc_fixtures"]=True
        # Full 256-byte vocabulary passes through context; EOS remains a typed control.
        io=Learner("BIND",202610166003)
        with torch.no_grad():
            io.prompt(bytes(range(256)))
            assert io.h.isfinite().all() and io.h.abs().max()<=1
            code=io._encode([f.clone() for f in io.fronts],io.time+30/14)
            assert code.shape==(2,1,5177)
        checks["all_256_payloads_no_hidden_protocol"]=True
        # Finite difference of the declared sigmoid with frozen threshold and scale.
        anatomy=io.anatomy
        p=torch.full((2,88),.5,dtype=torch.float64,requires_grad=True)
        weighted=torch.arange(anatomy.N,dtype=torch.float64)
        y=anatomy.encode(p,True)
        gradient=torch.autograd.grad((y*weighted).sum(),p)[0]
        pn0=p.detach().numpy()
        limits=[]
        for bank in range(2):
            raw=np.asarray(anatomy.reference.B.T@pn0[bank][anatomy.reference.pn_type_index]).ravel()
            native_x=anatomy.encode(p.detach(),False)[bank,0].numpy()
            for side in (0,1):
                ids=np.flatnonzero((anatomy.side==side)&(raw>0))
                limits.append((bank,ids,raw[ids[native_x[ids]>0]].min(),max(float(raw[ids].std()),1e-6)))
        def relaxed(pn):
            total=0.
            for bank,ids,kappa,scale in limits:
                raw=np.asarray(anatomy.reference.B.T@pn[bank][anatomy.reference.pn_type_index]).ravel()
                z=(raw[ids]-kappa)/scale
                total+=float(np.sum((1/(1+np.exp(-z)))*ids))
            return total
        flat=int(gradient.abs().argmax());index=np.unravel_index(flat,tuple(p.shape));eps=1e-6
        pa=pn0.copy();pb=pn0.copy();pa[index]+=eps;pb[index]-=eps
        finite=(relaxed(pa)-relaxed(pb))/(2*eps)
        error=abs(finite-float(gradient[index]))/max(1.,abs(finite))
        assert error<1e-5
        checks["declared_STE_frozen_surrogate_relative_error"]=error
        # Bounded four-worker native histories. Completed artifacts are immutable.
        jobs=out/"jobs";jobs.mkdir(exist_ok=True)
        completions=[]
        ctx=multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=4,mp_context=ctx) as pool:
            futures={pool.submit(run_job,arm,202610166003,str(jobs/arm)):arm for arm in ARMS}
            for future in as_completed(futures):
                result=future.result();completions.append(result)
                write(out/"STATUS.json",dict(status="RUNNING_TECHNICAL",completed=len(completions),
                      roster=list(ARMS),completed_jobs=completions,science_jobs=0))
        assert len(completions)==8
        assert source_identity()==snapshot
        checks["four_worker_jobs"]=completions
        # The same four next records after checkpoint, compared once; both applications charged.
        job=__import__("json").loads((jobs/"BIND"/"RESULT.json").read_text())
        restored,cursor=Learner.restore(jobs/"BIND"/"W.pt")
        records=evaluator.stream(202610166003,"new",4)
        before=restored.clone()
        nextstep=ledger.call("checkpoint-reference-continuation-four-records",restored,
                            lambda:restored.train_window(records))
        restored.rest(86400)
        assert restored.digest()==job["histories"]["W"]["final_digest"]
        # Actual parameter update must change hard production addresses, not only a proxy loss.
        changed=0
        # Early theta + late native state: a descriptive same-memory intervention.
        early=restored.clone()
        for k,v in early.p.items():
            with torch.no_grad():v.copy_(before.p[k])
        for prompt,_ in records:
            with torch.no_grad():
                va=restored.clone();vb=early.clone();va.prompt(prompt);vb.prompt(prompt)
                xa=va._encode([f.clone() for f in va.fronts],va.time+30/14)
                xb=vb._encode([f.clone() for f in vb.fronts],vb.time+30/14)
                changed+=int((xa!=xb).sum())
        assert changed>0
        checks["hard_KC_flips_after_real_update"]=changed
        checks["checkpoint_continuation_exact"]=True
        # Source-lock tampering is rejected, without launching another training record.
        saved=torch.load(jobs/"BIND"/"W.pt",weights_only=False)
        saved["source_identity"]["bind_kernel.py"]="tampered"
        torch.save(saved,out/"TAMPERED_CHECKPOINT.pt")
        try:Learner.restore(out/"TAMPERED_CHECKPOINT.pt")
        except ValueError:checks["tampered_source_checkpoint_rejected"]=True
        else:raise AssertionError("source tamper accepted")
        report=dict(passed=True,checks=checks,source_identity=snapshot,
                    full_record_applications=128+4,elapsed_seconds=time.time()-start,
                    science_jobs=0,evidence="E0 technical pipeline only")
        write(out/"RESULT.json",report)
        write(out/"STATUS.json",dict(status="COMPLETED_TECHNICAL",completed=8,science_jobs=0,
                                   full_record_applications=132))
        print(dict(passed=True,records=132,seconds=time.time()-start,hard_KC_flips=changed))
    except Exception as exc:
        write(out/"FAILURE.json",dict(error=repr(exc),traceback=traceback.format_exc(),checks=checks))
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);main(p.parse_args().out)
