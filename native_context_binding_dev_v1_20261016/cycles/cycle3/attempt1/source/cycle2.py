"""Gradient, causal permissions and distinct specialized-mechanism checks."""
import argparse
import ast
import math
import time
import traceback
from pathlib import Path
import bootstrap
import numpy as np
import torch
from bind_native import State,tensor
from bind_kernel import Learner,ARMS,DELTA
from bind_artifacts import Ledger,write,source_identity,admit_technical


def numerical_gradient(call,value,index,eps=1e-6):
    a=value.detach().clone();b=a.clone()
    a[index]+=eps;b[index]-=eps
    return float((call(a)-call(b))/(2*eps))


def main(out):
    admit_technical();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ledger=Ledger(out);checks={};started=time.time()
    try:
        base=Learner("BIND",202610166002)
        records=[(b"2+3=",b"5"),(b"4+5=",b"9"),(b"9+2=",b"11"),(b"3+4=",b"7")]
        # Same hard forward states; only the previous-record derivative is cut.
        other=base.clone()
        full=ledger.call("full-native-write-tape",base,lambda:base.train_window(records,apply_update=False))
        cut=ledger.call("cut-previous-record-native-write-credit",other,
                        lambda:other.train_window(records,apply_update=False,detach_between_records=True))
        assert base.digest()==other.digest()
        diff=math.sqrt(sum(float(((base.last_grads[k]-other.last_grads[k])**2).sum()) for k in base.p))
        assert diff>1e-10
        checks["across_record_write_credit_gradient_difference"]=diff
        # Exact native differentiable arithmetic, independent of top-k surrogate.
        anatomy=base.anatomy
        x=torch.full((2,1,anatomy.N),.01,dtype=torch.float64,requires_grad=True)
        fixed=base.state.detached(True)
        call=lambda u:(anatomy.expression(fixed,u)-anatomy.expression(fixed,u,True)).sum()
        grad=torch.autograd.grad(call(x),x)[0]
        ids=torch.topk(grad.abs().flatten(),3).indices.tolist()
        errors=[]
        for flat in ids:
            index=np.unravel_index(flat,tuple(x.shape))
            finite=numerical_gradient(call,x,index)
            errors.append(abs(finite-float(grad[index])))
        assert max(errors)<1e-5
        checks["exact_readout_derivative_max_abs"]=max(errors)
        # Exact native write derivatives at a nonzero, fixed-activity state.
        xw=torch.full((2,1,anatomy.N),.002,dtype=torch.float64,requires_grad=True)
        coefficients=dict(shared=torch.arange(16,dtype=torch.float64)%2,
                          private=torch.linspace(-.4,.4,16,dtype=torch.float64))
        def write_objective(z):
            state,_,_=anatomy.teach(fixed,z,DELTA,coefficients,True)
            return state.fast.square().sum()+state.slow.square().sum()
        gw=torch.autograd.grad(write_objective(xw),xw)[0]
        flat=int(gw.abs().argmax());index=np.unravel_index(flat,tuple(xw.shape))
        finite=numerical_gradient(write_objective,xw,index,eps=1e-7)
        error=abs(finite-float(gw[index]))/max(1.,abs(finite),abs(float(gw[index])))
        assert error<1e-5
        checks["exact_native_write_derivative_relative_error"]=error
        # The declared STE is checked against its frozen-mask sigmoid, not hard finite differences.
        pn=torch.full((2,88),.5,dtype=torch.float64,requires_grad=True)
        y=anatomy.encode(pn,True)
        hard=anatomy.encode(pn,False)
        assert torch.equal(y,hard)
        gs=torch.autograd.grad((y*torch.arange(anatomy.N,dtype=torch.float64)).sum(),pn)[0]
        assert torch.isfinite(gs).all() and float(gs.abs().sum())>0
        checks["STE_hard_forward_exact"]=True
        checks["STE_nonzero_declared_backward"]=float(gs.abs().sum())
        # All eight physical variants, including both conditional frozen controls.
        results={}
        for arm in ARMS:
            model=Learner(arm,202610166002)
            pre=model.theta_digest()
            w=ledger.call(f"{arm}-W",model,lambda:model.train_window(records))
            after=model.theta_digest()
            assert (pre!=after)==model.learns
            state=model.state.detached(True);before=model.theta_digest()
            time0=model.time
            n=ledger.call(f"{arm}-NONZERO_N",model,lambda:model.train_window(records,False))
            assert before==model.theta_digest()
            expected=model.anatomy.rest(state,model.time-time0)
            assert torch.allclose(model.state.fast,expected.fast,atol=1e-9,rtol=1e-9)
            assert torch.allclose(model.state.slow,expected.slow,atol=1e-9,rtol=1e-9)
            assert n["applied_update"] is False
            before=model.digest();model.generate(b"2+3=",cap=1)
            assert before==model.digest()
            assert all(row["prefeedback"] for row in model.last_trace)
            results[arm]=dict(parameters=sum(p.numel() for p in model.p.values()),
                W=w,N=n,feedback_max=getattr(model,"last_feedback_max",None),
                production_cpu=model.resource["production_cpu"])
            if model.feedback:
                assert model.last_feedback_max<=.5
                # Fixed-prestate rm must not depend on current punishment or write permission.
                xx=anatomy.encode(torch.ones((2,88),dtype=torch.float64),False)
                _,rm0,_=anatomy.event(model.state,xx,DELTA,torch.zeros((2,16)),False)
                _,rm1,_=anatomy.event(model.state,xx,DELTA,torch.ones((2,16)),True)
                assert torch.equal(rm0,rm1)
                checks["prewrite_MBON_target_invariance"]=True
                # A trainable G is amplitude bounded even under extreme coefficients.
                with torch.no_grad():
                    model.p["G"].fill_(1e6)
                    model._context(49,rm0)
                    assert model.last_feedback_max<=.5
            if model.adaptive:
                assert torch.any(model.a!=0)
                assert results[arm]["parameters"]==23424+128
            if model.feedback:
                assert results[arm]["parameters"]==23424+768
            if arm in ("BIND","BIND-1STEP"):
                # Same last byte, different earlier h: full carry responds; 1STEP forgets new h.
                with torch.no_grad():
                    model.h.fill_(.5);model._context(50,torch.zeros((2,16,4)));ha=model.h.clone()
                    model.h.fill_(-.5);model._context(50,torch.zeros((2,16,4)));hb=model.h.clone()
                    assert torch.equal(ha,hb)==model.one_step
        checks["all_eight_configs"]=results
        for name in ("bind_kernel.py","bind_native.py"):
            tree=ast.parse((bootstrap.HERE/name).read_text())
            imports={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
            imports|={a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names}
            assert not imports&{"tasks","evaluator","data","scipy.optimize"}
        report=dict(passed=True,checks=checks,full_record_applications=sum(r["records_executed"] for r in ledger.rows),
                    source_identity=source_identity(),elapsed_seconds=time.time()-started,
                    evidence="E0 source derivatives/causal controls; no efficacy or E3 claim",science_jobs=0)
        write(out/"RESULT.json",report)
        print(dict(passed=True,records=report["full_record_applications"],seconds=report["elapsed_seconds"],
                   gradient_path_difference=diff))
    except Exception as exc:
        write(out/"FAILURE.json",dict(error=repr(exc),traceback=traceback.format_exc(),
                                    checks=checks,source_identity=source_identity()))
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);main(p.parse_args().out)
