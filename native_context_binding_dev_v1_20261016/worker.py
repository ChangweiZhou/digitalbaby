"""Manifest-admitted native experiment worker; no automatic authorization.

SMOKE uses the same history/score/permission/checkpoint path as DEV/CONFIRM,
but accepts only the three reserved technical worlds and 4+4 records.
"""
import argparse
import hashlib
import json
import signal
import time
from pathlib import Path
import bootstrap
import torch
from bind_kernel import Learner,ARMS
from bind_artifacts import Ledger,write,source_identity
import bind_evaluator as evaluator

STOP=False


def stop_requested(signum,frame):
    global STOP
    STOP=True


def admission(stage,arm,world,authorization):
    design=json.loads((bootstrap.ROOT/"native_context_binding_design_v1_20261016/DESIGN.json").read_text())
    if stage=="SMOKE":
        if world not in (202610166001,202610166002,202610166003) or arm not in ARMS:
            raise ValueError("technical namespace rejected")
        return 4,4,[0,4,8],None
    if stage not in ("DEV","CONFIRM") or authorization is None:
        raise PermissionError("science requires explicit human authorization and locked stage")
    auth=json.loads(Path(authorization).read_text())
    lock_path=bootstrap.HERE/"SOURCE_LOCK.json"
    if auth.get("schema")!="NATIVE_BIND_RUN_AUTHORIZATION_V1" or auth.get("human_authorized") is not True or auth.get("experiment")!="NATIVE-BIND-V1" or stage not in auth.get("stages",[]):
        raise PermissionError("science authorization rejected")
    lock=json.loads(lock_path.read_text())
    for relative,digest in lock["operative_sources"].items():
        path=bootstrap.ROOT/relative
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise PermissionError("source lock mismatch")
    if stage=="DEV":
        spec=design["dev"];roster=spec["roster"];cpu_budget=None
    else:
        selection=json.loads((bootstrap.HERE/"results/DEV/SELECTION_LOCK.json").read_text())
        if selection.get("selected") not in design["confirm"]["rosters"]:
            raise PermissionError("no selected confirmation recipe")
        spec=design["confirm"];roster=spec["rosters"][selection["selected"]]
        cpu_budget=selection["confirmation_CPU_budget"]
    if world not in spec["worlds"] or arm not in roster:
        raise PermissionError("world/arm outside frozen stage manifest")
    old_n,new_n=spec["old_records_per_history"],spec["new_records_per_history"]
    points=[0,16,64,256,512,768,1024] if stage=="DEV" else [0,16,64,256,512,1024,1536,2048]
    return old_n,new_n,points,cpu_budget


def run(stage,arm,world,directory,authorization=None):
    old_n,new_n,checkpoints,cpu_budget=admission(stage,arm,world,authorization)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/"RECEIPT.json").exists() or (directory/"EXECUTION_LEDGER.json").exists():
        raise ValueError("existing executed job is immutable; never replay")
    ledger=Ledger(directory);started=time.time()
    birth=Learner(arm,world)
    old=evaluator.stream(world,"old",old_n);new=evaluator.stream(world,"new",new_n)
    groups=evaluator.split(world)
    def rendered(name):
        return [(f"{a}+{b}=".encode(),str(a+b).encode()) for a,b in groups[name]]
    held=rendered("held")[:2] if stage=="SMOKE" else rendered("held")
    history_results={}
    for history in ("W","N_OLD"):
        model=birth.clone();points=[];windows=[];boundaries=[]
        thresholds=[cpu_budget*f for f in (1/16,1/8,1/4,1/2,3/4,1)] if cpu_budget else []
        def measured(position,tag):
            before=model.digest();time0=time.process_time()
            readings={"MEM":evaluator.score(model,held)}
            if arm=="NATIVE":readings["LEGACY"]=evaluator.score(model,held,"LEGACY")
            assert before==model.digest()
            points.append(dict(records=position,tag=tag,readings=readings,
                CPU_MEM=model.resource["logical_MEM_cpu"],
                CPU_LEGACY=model.resource["logical_LEGACY_cpu"],
                probe_cpu=time.process_time()-time0))
        measured(0,"SAMPLE")
        for phase,records in (("OLD",old),("NEW",new)):
            allowed=phase=="NEW" or history=="W"
            for i in range(0,len(records),4):
                before_theta=model.theta_digest()
                old_state=model.state.detached(True) if not allowed else None
                before_time=model.time
                alpha0=model.resource["raw_alpha_l1"];actors=model.actor_steps
                result=ledger.call(f"{history}-{phase}-{i}",model,
                    lambda rows=records[i:i+4],allow=allowed:model.train_window(rows,allow))
                if not allowed:
                    expected=model.anatomy.rest(old_state,model.time-before_time)
                    assert before_theta==model.theta_digest()
                    assert torch.allclose(model.state.fast,expected.fast,atol=1e-9,rtol=1e-9)
                    assert torch.allclose(model.state.slow,expected.slow,atol=1e-9,rtol=1e-9)
                    assert model.resource["raw_alpha_l1"]==alpha0
                assert model.actor_steps-actors==int(model.learns and allowed)
                windows.append(dict(phase=phase,history=history,start=i,n=4,allowed=allowed,
                                    actor_steps=model.actor_steps-actors,
                                    alpha_l1=model.resource["raw_alpha_l1"]-alpha0))
                position=i+4+(old_n if phase=="NEW" else 0)
                threshold_reached=bool(thresholds and model.resource["logical_MEM_cpu"]>=thresholds[0])
                if position in checkpoints or threshold_reached:
                    measured(position,"SAMPLE" if position in checkpoints else "CPU")
                while thresholds and model.resource["logical_MEM_cpu"]>=thresholds[0]:
                    thresholds.pop(0)
                write(directory/"ACTIVE.json",dict(stage=stage,arm=arm,world=world,history=history,
                    position=position,old_n=old_n,new_n=new_n,heartbeat_epoch=time.time(),
                    committed_full_jobs=0,completed_inflight_records=len(windows)*4))
                if STOP:
                    model.checkpoint(directory/"PAUSE.pt",dict(history=history,phase=phase,
                        next_record=i+4,position=position,points=points,windows=windows,
                        boundaries=boundaries,completed_histories=history_results))
                    write(directory/"SESSION_RESULT.json",dict(status="PAUSED_SESSION_LIMIT",
                        last_position=position,history=history,checkpoint="PAUSE.pt"))
                    return dict(status="PAUSED_SESSION_LIMIT",science_job_committed=False)
            before=model.digest()
            before_boundary=dict(phase=phase,tag="BEFORE_REST",scores={
                k:evaluator.score(model,rendered(k)[:2] if stage=="SMOKE" else rendered(k))
                for k in ("old","new")})
            assert model.digest()==before
            boundaries.append(before_boundary)
            model.rest(86400)
            measured(old_n if phase=="OLD" else old_n+new_n,phase+"_REST")
            boundaries.append(dict(phase=phase,tag="AFTER_REST",scores={
                k:evaluator.score(model,rendered(k)[:2] if stage=="SMOKE" else rendered(k))
                for k in ("old","new")}))
        model.checkpoint(directory/f"{history}_FINAL.pt",dict(history=history,position=old_n+new_n))
        history_results[history]=dict(points=points,boundaries=boundaries,windows=windows,
            final_digest=model.digest(),resource=model.resource.copy())
    receipt=dict(schema="NATIVE_BIND_SCIENCE_V1" if stage!="SMOKE" else "NATIVE_BIND_TECHNICAL_SMOKE_V1",
        stage=stage,arm=arm,world=world,histories=history_results,old_n=old_n,new_n=new_n,
        records_executed=2*(old_n+new_n),source_identity=source_identity(),
        elapsed_seconds=time.time()-started)
    audit_receipt(receipt)
    write(directory/"RECEIPT.json",receipt)
    write(directory/"AUDIT.json",dict(accepted=True,stage=stage,records=receipt["records_executed"],
                                    science=stage!="SMOKE",history_permissions_checked=True))
    return receipt


def audit_receipt(receipt):
    if receipt["stage"] not in ("DEV","CONFIRM","SMOKE"):
        raise AssertionError("invalid classification")
    if set(receipt["histories"])!={"W","N_OLD"}:
        raise AssertionError("missing causal history")
    expected=2*(receipt["old_n"]+receipt["new_n"])
    if receipt["records_executed"]!=expected:
        raise AssertionError("wrong complete-job record count")
    for history,data in receipt["histories"].items():
        expected_positions={"OLD":0,"NEW":0}
        for row in data["windows"]:
            allowed=row["phase"]=="NEW" or history=="W"
            if row["allowed"] is not allowed or row["history"]!=history or row["n"]!=4 or row["start"]!=expected_positions[row["phase"]]:
                raise AssertionError("causal branch permission/count violation")
            expected_positions[row["phase"]]+=4
            if not allowed and (row["actor_steps"]!=0 or row["alpha_l1"]!=0):
                raise AssertionError("forbidden native/theta learning")
        if expected_positions!={"OLD":receipt["old_n"],"NEW":receipt["new_n"]}:
            raise AssertionError("missing/duplicate history windows")
        if data["resource"]["records"]!=expected//2:
            raise AssertionError("ledger/resource mismatch")
        for point in data["points"]:
            for value in point["readings"].values():
                score=sum(r["correct"] for r in value["rows"])/len(value["rows"])
                if score!=value["accuracy"]:
                    raise AssertionError("inconsistent exact-output score")
                for row in value["rows"]:
                    correct=row["status"]=="EOS" and row["emitted"]==row["answer"]
                    if row["correct"] is not correct:
                        raise AssertionError("wrong own-output grading")
    return True


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("SMOKE","DEV","CONFIRM"),required=True)
    p.add_argument("--arm",choices=ARMS,required=True);p.add_argument("--world",type=int,required=True)
    p.add_argument("--out",required=True);p.add_argument("--authorization")
    args=p.parse_args()
    signal.signal(signal.SIGUSR1,stop_requested)
    result=run(args.stage,args.arm,args.world,args.out,args.authorization)
    print(dict(stage=args.stage,records=result.get("records_executed",0),status=result.get("status","COMPLETED")))
