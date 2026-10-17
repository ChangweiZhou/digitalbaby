"""Source parity and actual-native cold-start credit qualification."""
import argparse
import time
from pathlib import Path
import bootstrap
import numpy as np
import torch
from bind_native import tensor
from bind_kernel import Learner, DELTA, CODE
from bind_artifacts import Ledger, write, source_identity, admit_technical


def assert_close(actual, expected, label, atol=1e-9):
    a = actual.detach().numpy() if isinstance(actual,torch.Tensor) else np.asarray(actual)
    e = np.asarray(expected)
    np.testing.assert_allclose(a,e,atol=atol,rtol=1e-9,err_msg=label)
    return float(np.max(np.abs(a-e))) if a.size else 0.


def parent_mem(parent):
    import stores
    values=[]
    for fe,bank in zip(parent.fronts,parent.banks):
        f=fe.clone();f.advance(parent.time+DELTA)
        x=stores.bb.bc.native().model.encode_sparse(bank[0].fly.m,f.read())
        row=[]
        for s in bank:
            fly=s.fly.clone();fly.rest(DELTA)
            dx=stores.bb.bc.native().common.observed_activity(fly.m,x[None,:])
            row.append(float((fly.m.expression(dx)-fly.m.expression(dx,True)).mean()))
        values.append(np.array(row))
    return values[0]/1.4911274663291492+values[1]/1.3452365735750882


def main(out):
    admit_technical()
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ledger=Ledger(out);started=time.time()
    results={}
    try:
        parent_module=bootstrap.source_module("native_parity_parent","native_interface_transfer_dev_v1_20261015/models.py")
        model=Learner("NATIVE",202610166001)
        parent=parent_module.SequenceC16()
        results["separate_canonical_births"]=len(model.births)==32
        records=[(b"2+3=",b"5"),(b"4+5=",b"9"),(b"9+2=",b"11"),(b"\x00\n =\xff",b"\x00\xff")]
        for allowed in (True,False):
            ledger.call(f"mirror-{'W' if allowed else 'NONZERO_N'}",model,lambda:model.train_window(records,allowed))
            for i,(prompt,answer) in enumerate(records):
                ledger.call(f"parent-{'W' if allowed else 'NONZERO_N'}-{i}",parent,
                            lambda p=prompt,a=answer:parent.train(p,a,allowed),planned=1)
            for name in ("fast","slow","adapt"):
                expected=np.stack([getattr(s.fly.m,name) for bank in parent.banks for s in bank])
                expected=expected.reshape(tuple(getattr(model.state,name).shape))
                results[f"{allowed}_{name}_max_abs"]=assert_close(getattr(model.state,name),expected,name)
            assert model.time==parent.time
            with torch.no_grad():
                results[f"{allowed}_legacy_max_abs"]=assert_close(model.values("LEGACY"),parent.values(),"legacy")
                results[f"{allowed}_memory_max_abs"]=assert_close(model.values("MEM"),parent_mem(parent),"memory")
        model.rest(86400);parent.rest(86400)
        for name in ("fast","slow","adapt"):
            expected=np.stack([getattr(s.fly.m,name) for bank in parent.banks for s in bank])
            results[f"rest_{name}_max_abs"]=assert_close(getattr(model.state,name),
                expected.reshape(tuple(getattr(model.state,name).shape)),name)
        # Raw-byte and codebook interface without training 257 target records.
        assert all(int((CODE@(CODE[y]-CODE[y].mean())/4).argmax())==y for y in range(257))
        before=model.digest();model.generate(b"\x00\n =\xff",cap=2)
        assert before==model.digest()
        results["all_257_output_rows_decode"]=True
        results["readonly_own_generation"]=True
        # One chronological four-record native-write tape, not invented initial memory.
        bind=Learner("BIND",202610166001)
        summary=ledger.call("BIND-cold-start-four-record-native-write",bind,
                           lambda:bind.train_window(records))
        assert summary["gradient_norm"]>0
        assert bind.actor_steps==1
        results["cold_start_native_write_tape"]=summary
        results["parameter_count"]=sum(p.numel() for p in bind.p.values())
        assert results["parameter_count"]==23424
        report=dict(passed=True,checks=results,source_identity=source_identity(),
                    elapsed_seconds=time.time()-started,
                    full_record_applications=sum(x["records_executed"] for x in ledger.rows),
                    science_jobs=0,evidence="E0 technical parity/credit, not arithmetic learning")
        write(out/"RESULT.json",report)
        print({"passed":True,"records":report["full_record_applications"],"seconds":report["elapsed_seconds"],"checks":results})
    except Exception as exc:
        write(out/"FAILURE.json",dict(error=repr(exc),checks=results,source_identity=source_identity()))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",required=True)
    main(parser.parse_args().out)
