"""Operational completion of the qualified single-job path; kernel is unchanged."""
import argparse
import copy
import hashlib
import os
import signal
import sys
import resource
import time
from pathlib import Path
import bootstrap
import torch
from bind_kernel import Learner
from campaign_ledger import Ledger
import bind_evaluator as evaluator
import campaign_common as c
import campaign_audit
import campaign_diagnostics
import gzip
import audit_technical


STOP=False


def request_stop(signum, frame):
    global STOP
    STOP=True


def resume_cursor(path, stage, arm, world, ledger):
    model,cursor=Learner.restore(path)
    if (cursor["stage"],cursor["arm"],cursor["world"])!=(stage,arm,world):
        raise PermissionError("Checkpoint job identity mismatch")
    if cursor["campaign_source_identity"] != c.identity():
        raise PermissionError("Checkpoint campaign source changed")
    completed=sum(h["resource"]["records"] for h in cursor["completed_histories"].values())
    used=sum(row["records_executed"] for row in ledger.rows)
    if completed+model.resource["records"] != used:
        raise AssertionError("Checkpoint does not bind the next unexecuted record")
    if any(row["status"]!="COMPLETED" for row in ledger.rows):
        raise PermissionError("An incomplete training window cannot be resumed/replayed")
    return model,cursor


def run(stage,arm,world,directory,authorization,resume=False):
    scientific_stage=stage
    old_n,new_n,sample_points,cpu_budget=c.admission(scientific_stage,arm,world,authorization)
    selected=None
    manifest=c.manifest(stage,selected)
    if world not in manifest["worlds"]:raise PermissionError("Wrong confirmation session")
    auth=c.read(authorization)
    if stage not in auth.get("sessions",[]):
        raise PermissionError("This finite session is not human-authorized")
    if resume and not auth.get("resume_authorized",False):
        raise PermissionError("Resume requires separate explicit human authorization")
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/"RECEIPT.json").exists():raise PermissionError("Committed job is immutable")
    start=time.time();process_cpu0=time.process_time();init_cpu0=process_cpu0
    ledger_path=directory/"EXECUTION_LEDGER.json"
    current=None;cursor=None;histories={}
    if ledger_path.exists():
        if not resume:raise PermissionError("Never automatically restart an executed job")
        saved=c.read(ledger_path)
        ledger=Ledger.__new__(Ledger);ledger.path=ledger_path;ledger.rows=saved["rows"];ledger.started=saved["started_epoch"]
        current,cursor=resume_cursor(directory/"PAUSE.pt",stage,arm,world,ledger)
        histories=cursor["completed_histories"]
        birth=None
    else:
        if resume:raise PermissionError("No checkpoint to resume")
        ledger=Ledger(directory)
        birth=Learner(arm,world)
        birth.checkpoint(directory/"BIRTH.pt",dict(stage=stage,arm=arm,world=world,kind="BIRTH",next_record=0))
    init_cpu=time.process_time()-init_cpu0
    old=evaluator.stream(world,"old",old_n);new=evaluator.stream(world,"new",new_n)
    groups=evaluator.split(world)
    rendered={k:[(f"{a}+{b}=".encode(),str(a+b).encode()) for a,b in rows] for k,rows in groups.items()}
    readings=("MEM","LEGACY") if arm=="NATIVE" else ("MEM",)
    for hi,history in enumerate(("W","N_OLD")):
        if history in histories:continue
        if cursor is not None:
            if cursor["history"]!=history:raise AssertionError("Resume history cursor mismatch")
            model=current;points=cursor["points"];windows=cursor["windows"];boundaries=cursor["boundaries"]
            thresholds=cursor["thresholds"];phase_start=cursor["phase_index"];record_start=cursor["phase_next_record"]
            theta_birth={k:torch.tensor(v,dtype=torch.float64) for k,v in cursor["theta_birth"].items()};theta_old={k:torch.tensor(v,dtype=torch.float64) for k,v in cursor["theta_old"].items()};diagnostics=cursor["diagnostics"]
            cursor=None
        else:
            if birth is None:
                t=time.process_time();birth,_=Learner.restore(directory/"BIRTH.pt");init_cpu+=time.process_time()-t
            model=birth.clone();points=[];windows=[];boundaries=[];diagnostics={}
            thresholds={r:[cpu_budget*f for f in (1/16,1/8,1/4,1/2,3/4,1)] if cpu_budget else [] for r in readings}
            phase_start=0;record_start=0
            theta_birth={k:v.detach().clone() for k,v in model.p.items()};theta_old={}

        def measured(position,tag):
            before=model.digest();resource=copy.deepcopy(model.resource);t=time.process_time()
            scores={r:evaluator.score(model,rendered["held"],r) for r in readings}
            if before!=model.digest() or model.resource!=resource:raise AssertionError("Probe wrote production state")
            points.append(dict(records=position,tag=tag,readings=scores,
                 CPU_MEM=model.resource["logical_MEM_cpu"],CPU_LEGACY=model.resource["logical_LEGACY_cpu"],probe_cpu=time.process_time()-t))

        def boundary(phase,tag):
            before=model.digest();t=time.process_time()
            scores={r:{k:evaluator.score(model,rendered[k],r) for k in ("old","new")} for r in readings}
            if before!=model.digest():raise AssertionError("Boundary probe changed model")
            boundaries.append(dict(phase=phase,tag=tag,readings=scores,probe_cpu=time.process_time()-t))

        if not points:
            measured(0,"SAMPLE");boundary("BIRTH","BIRTH")
        for pi,(phase,records) in enumerate((("OLD",old),("NEW",new))):
            if pi<phase_start:continue
            first=record_start if pi==phase_start else 0
            for j in range(first,len(records),4):
                allowed=history=="W" or phase=="NEW"
                theta=model.theta_digest();actors=model.actor_steps;alpha=model.resource["raw_alpha_l1"]
                oldstate=model.state.detached(True) if not allowed else None;oldtime=model.time
                result=ledger.call(f"{history}-{phase}-{j}",model,
                    lambda rows=records[j:j+4],allow=allowed:model.train_window(rows,allow))
                audit_technical.audit_traces(arm,[model.last_trace],[result],[allowed])
                tracefile=directory/f'{history}_{phase}_{j:04d}_TRACE.json.gz'
                if tracefile.exists():raise PermissionError('Never overwrite executed trace')
                with gzip.open(tracefile,'wt') as out:
                    __import__('json').dump(dict(history=history,phase=phase,start=j,update=result,trace=model.last_trace),out,allow_nan=False)
                unchanged=theta==model.theta_digest();passive=True
                if not allowed:
                    expected=model.anatomy.rest(oldstate,model.time-oldtime)
                    passive=bool(torch.allclose(model.state.fast,expected.fast,atol=1e-9,rtol=1e-9) and
                                 torch.allclose(model.state.slow,expected.slow,atol=1e-9,rtol=1e-9))
                    if not unchanged or not passive:raise AssertionError("N_OLD changed native facts/theta/optimizer")
                delta_actor=model.actor_steps-actors;alpha_l1=model.resource["raw_alpha_l1"]-alpha
                if delta_actor!=int(model.learns and allowed) or (not allowed and alpha_l1!=0):
                    raise AssertionError("Forbidden update")
                windows.append(dict(history=history,phase=phase,start=j,n=4,allowed=allowed,
                    actor_steps=delta_actor,alpha_l1=alpha_l1,theta_unchanged=unchanged,
                    theta_before=theta,theta_after=model.theta_digest(),
                    native_passive_only=passive,trace_sha256=c.sha(tracefile),trace_file=tracefile.name,loss=result["loss"],gradient_norm=result["gradient_norm"],
                    CPU_MEM=model.resource["logical_MEM_cpu"],CPU_LEGACY=model.resource["logical_LEGACY_cpu"],
                    parameter_gradient_norms={k:float(g.norm()) for k,g in model.last_grads.items()}))
                position=j+4+(old_n if phase=="NEW" else 0)
                reached=any(v and model.resource["logical_"+r+"_cpu"]>=v[0] for r,v in thresholds.items())
                if position in sample_points or reached:
                    measured(position,"SAMPLE" if position in sample_points else "CPU")
                for r,values in thresholds.items():
                    while values and model.resource["logical_"+r+"_cpu"]>=values[0]:values.pop(0)
                c.write(directory/"ACTIVE.json",dict(stage=stage,arm=arm,world=world,history=history,phase=phase,
                    position=position,history_records=model.resource["records"],
                    completed_inflight_records=sum(h["resource"]["records"] for h in histories.values())+model.resource["records"],
                    production_cpu=model.resource["production_cpu"],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024),heartbeat_epoch=time.time(),pid=os.getpid()))
                if STOP:
                    model.checkpoint(directory/"PAUSE.pt",dict(stage=stage,arm=arm,world=world,
                        campaign_source_identity=c.identity(),history=history,phase_index=pi,phase_next_record=j+4,next_record=model.resource["records"],
                        points=points,windows=windows,boundaries=boundaries,thresholds=thresholds,
                        theta_birth={k:v.tolist() for k,v in theta_birth.items()},theta_old={k:v.tolist() for k,v in theta_old.items()},diagnostics=diagnostics,completed_histories=histories))
                    c.write(directory/"SESSION_RESULT.json",dict(status="PAUSED_SESSION_LIMIT",stage=stage,
                        committed_full_jobs=0,completed_inflight_records=sum(r["records_executed"] for r in ledger.rows)))
                    return dict(status="PAUSED_SESSION_LIMIT")
            boundary(phase,"BEFORE_REST")
            if phase=="OLD":model.checkpoint(directory/f"{history}_OLD_END.pt",dict(kind="OLD_END",history=history,next_record=model.resource["records"]))
            model.rest(86400)
            measured(old_n if phase=="OLD" else old_n+new_n,phase+"_REST")
            boundary(phase,"AFTER_REST")
            if phase=="OLD":
                theta_old={k:v.detach().clone() for k,v in model.p.items()}
                model.checkpoint(directory/f"{history}_OLD_REST.pt",dict(kind="OLD_REST",history=history,next_record=model.resource["records"]))
                diagnostics["OLD_REST"]=campaign_diagnostics.measure(model,rendered["old"][:2]+rendered["held"][:2],{"BIRTH_THETA":theta_birth})
        diagnostics["FINAL"]=campaign_diagnostics.measure(model,rendered["old"][:2]+rendered["held"][:2],
                                             {"BIRTH_THETA":theta_birth,"OLD_THETA":theta_old})
        model.checkpoint(directory/f"{history}_FINAL.pt",dict(kind="FINAL",history=history,position=old_n+new_n,next_record=model.resource["records"]))
        histories[history]=dict(points=points,windows=windows,boundaries=boundaries,diagnostics=diagnostics,
             final_digest=model.digest(),resource=model.resource.copy(),birth_receipts=model.births)
    receipt=dict(schema="NATIVE_BIND_SCIENCE_V2_1",stage=stage,arm=arm,world=world,old_n=old_n,new_n=new_n,
        histories=histories,records_executed=2*(old_n+new_n),source_identity=c.identity(),
        kernel_source_identity=__import__("bind_kernel").science_identity(),
        data_stream_sha256=hashlib.sha256(repr(old+new).encode()).hexdigest(),
        initialization_CPU_seconds=init_cpu,elapsed_seconds=time.time()-start,
        actual_worker_CPU_seconds=time.process_time()-process_cpu0,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024),
        resumed=resume,source_lock_sha256=c.sha(c.HERE/"CAMPAIGN_SOURCE_LOCK.json"))
    receipt["confirmation_admission_sha256"]=manifest["confirmation_admission_sha256"]
    audit_start=time.process_time();certificate=campaign_audit.audit(receipt,directory=directory)
    receipt["audit_CPU_seconds"]=time.process_time()-audit_start
    receipt["actual_worker_CPU_seconds"]=time.process_time()-process_cpu0
    c.write(directory/"RECEIPT.json",receipt)
    certificate["receipt_sha256"]=c.sha(directory/"RECEIPT.json")
    c.write(directory/"AUDIT.json",certificate)
    c.write(directory/"ACTIVE.json",dict(status="COMPLETED",stage=stage,world=world,arm=arm,
                                             heartbeat_epoch=time.time(),completed_inflight_records=0))
    return dict(status="COMPLETED",records=receipt["records_executed"])


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("DEV_A",),required=True)
    parser.add_argument("--arm",required=True);parser.add_argument("--world",required=True,type=int)
    parser.add_argument("--out",required=True);parser.add_argument("--authorization",required=True)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args();signal.signal(signal.SIGUSR1,request_stop)
    try:
        print(run(args.stage,args.arm,args.world,args.out,args.authorization,args.resume),flush=True)
    except BaseException as exc:
        import traceback
        Path(args.out).mkdir(parents=True,exist_ok=True)
        c.write(Path(args.out)/"FAILURE.json",dict(error=repr(exc),traceback=traceback.format_exc(),utc=c.now()))
        raise
