"""Zero-training operations fixtures and read-only qualified-checkpoint tests."""
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(HERE))
import bootstrap
import torch
import bind_kernel
import bind_evaluator as evaluator
import campaign_common as c
import campaign_worker as cw
import campaign_audit as auditor
import campaign_analysis as analysis

BASE=Path(__file__).resolve().parent
attempt=1
while (BASE/f"qualification_attempt{attempt}").exists():attempt+=1
OUT=BASE/f"qualification_attempt{attempt}"
OUT.mkdir(parents=True)


class State:
    def __init__(self,v=0.):self.fast=torch.tensor([v]);self.slow=torch.tensor([v]);self.adapt=torch.tensor([1.])
    def detached(self,clone=False):return copy.deepcopy(self)


class FakeLearner:
    """Counter fixture, never a neural learner or a science trajectory."""
    saved={}
    calls=0
    def __init__(self,arm,world):
        self.arm=arm;self.world=world;self.learns=arm!="NATIVE" and "FROZEN" not in arm
        self.actor_steps=0;self.state=State();self.time=0.;self.last_grads={}
        self.anatomy=SimpleNamespace(rest=lambda state,dt:copy.deepcopy(state))
        self.p={} if arm=="NATIVE" else {"p":torch.tensor([0.])}
        self.births=[dict(fly_id=j,canonical_B_sha256="32a3726cb25a8255c1aef5696ff40aa303138c5d061a598004d2087d4da5f964") for j in range(32)]
        self.resource=dict(records=0,input_bytes=0,answer_bytes=0,EOS_targets=0,native_store_events=0,
            actor_steps=0,production_cpu=0.,raw_alpha_l1=0.,logical_MEM_cpu=0.,logical_LEGACY_cpu=0.)
    def clone(self):return copy.deepcopy(self)
    def digest(self):return hashlib.sha256(repr((self.actor_steps,self.time,float(self.state.fast[0]),self.resource["records"])).encode()).hexdigest()
    def theta_digest(self):return str(self.actor_steps)
    def train_window(self,records,allowed=True):
        type(self).calls+=1
        if len(records)!=4:raise AssertionError("wrong mock window")
        if self.learns and allowed:self.actor_steps+=1
        if allowed:self.state.fast+=1;self.state.slow+=1;self.resource["raw_alpha_l1"]+=1
        self.time+=len(records)
        for prompt,answer in records:
            self.resource["records"]+=1;self.resource["input_bytes"]+=len(prompt)
            self.resource["answer_bytes"]+=len(answer);self.resource["EOS_targets"]+=1
            self.resource["native_store_events"]+=(2+len(prompt)+len(answer)+1)*32
        self.resource["actor_steps"]=self.actor_steps
        self.resource["production_cpu"]+=.04;self.resource["logical_MEM_cpu"]+=.04
        self.resource["logical_LEGACY_cpu"]+=.03 if self.arm=="NATIVE" else 0.
        return dict(loss=1.,gradient_norm=1.)
    def checkpoint(self,path,cursor):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        self.saved[str(path)]=(self.clone(),copy.deepcopy(cursor))
        path.write_text("SYNTHETIC_COUNTER_FIXTURE_NO_NATIVE_TRAINING\n")
    @classmethod
    def restore(cls,path):return copy.deepcopy(cls.saved[str(path)])
    def rest(self,seconds):self.time+=seconds


def fake_score(model,items,readout="MEM",cap=16):
    # Oracle emitter fixture for the scorer only, not claimed learned behavior.
    rows=[dict(prompt=p.hex(),answer=a.hex(),emitted=a.hex(),status="EOS",correct=True,emitted_tokens=len(a)+1) for p,a in items]
    return dict(accuracy=1.,rows=rows)


def reject(fn):
    try:fn()
    except (AssertionError,PermissionError,ValueError):return True
    raise AssertionError("A tampered fixture was accepted")


def main():
    checks={}
    core_identity=bind_kernel.science_identity()
    original=c.read(HERE/"cycles/cycle3/attempt1/pipeline_smoke/RECEIPT.json")["source_identity"]
    assert all(original[k]==v for k,v in core_identity.items())
    checks["qualified_kernel_bytes_unchanged"]=True
    # Any accidental real training call in this qualification is fatal.
    real_train=bind_kernel.Learner.train_window
    def forbidden(*args,**kwargs):raise AssertionError("No native training is authorized by operations qualification")
    bind_kernel.Learner.train_window=forbidden
    checkpoint=HERE/"cycles/cycle3/attempt1/pipeline_smoke/W_FINAL.pt"
    digest=c.sha(checkpoint)
    model,cursor=bind_kernel.Learner.restore(checkpoint)
    items=[(b"2+3=",b"5"),(b"3+2=",b"5")]
    before=model.digest()
    import campaign_diagnostics
    measured=campaign_diagnostics.measure(model,items,{"SAME_THETA":{k:v.detach().clone() for k,v in model.p.items()}})
    assert measured["training_records"]==0 and before==model.digest() and digest==c.sha(checkpoint)
    checks["real_snapshot_readonly_diagnostics"]=True
    for reading in ("MEM","LEGACY"):
        v=evaluator.score(model,items,reading)
        auditor.score_check(v,{p.hex():a.hex() for p,a in items})
    checks["both_boundary_readouts_real_checkpoint"]=True
    auth=OUT/"FIXTURE_AUTHORIZATION.json"
    c.write(auth,dict(human_authorized=False,sessions=["DEV"],resume_authorized=True,
                     synthetic_fixture_only=True,external_training_records=0))
    reject(lambda: cw.qualified_worker.admission("DEV","BIND",202610166101,auth))
    checks["fixture_auth_cannot_launch_science"]=True

    # Full orchestration through the real worker with a counter object replacing
    # all learner calls. Data worlds below only define evaluator fixtures.
    original_learner=cw.Learner;original_admission=cw.qualified_worker.admission
    original_score=evaluator.score;original_diagnostics=cw.campaign_diagnostics.measure
    cw.Learner=FakeLearner
    cw.qualified_worker.admission=lambda *args:(768,256,[0,16,64,256,512,768,1024],None)
    evaluator.score=fake_score
    cw.campaign_diagnostics.measure=lambda *args:dict(synthetic=True,training_records=0)
    native=cw.run("DEV","NATIVE",202610166101,OUT/"native_counter",auth)
    assert native["records"]==2048
    checks["complete_worker_both_readouts_fixture"]=True
    cw.STOP=True
    paused=cw.run("DEV","BIND",202610166101,OUT/"resumed_counter",auth)
    assert paused["status"]=="PAUSED_SESSION_LIMIT"
    cw.STOP=False
    resumed=cw.run("DEV","BIND",202610166101,OUT/"resumed_counter",auth,resume=True)
    reference=cw.run("DEV","BIND",202610166101,OUT/"reference_counter",auth)
    rr=c.read(OUT/"resumed_counter/RECEIPT.json");ref=c.read(OUT/"reference_counter/RECEIPT.json")
    assert {h:v["final_digest"] for h,v in rr["histories"].items()}=={h:v["final_digest"] for h,v in ref["histories"].items()}
    ledger=c.read(OUT/"resumed_counter/EXECUTION_LEDGER.json")
    assert ledger["full_record_applications"]==2048 and len(ledger["rows"])==512
    checks["worker_resume_next_record_without_replay_fixture"]=True
    original_fake_train=FakeLearner.train_window
    def pause_in_nold(self,records,allowed=True):
        result=original_fake_train(self,records,allowed)
        if not allowed:cw.STOP=True
        return result
    FakeLearner.train_window=pause_in_nold
    paused=cw.run("DEV","BIND",202610166101,OUT/"resumed_N_OLD_counter",auth)
    assert paused["status"]=="PAUSED_SESSION_LIMIT"
    FakeLearner.train_window=original_fake_train;cw.STOP=False
    cw.run("DEV","BIND",202610166101,OUT/"resumed_N_OLD_counter",auth,resume=True)
    rrn=c.read(OUT/"resumed_N_OLD_counter/RECEIPT.json")
    assert {h:v["final_digest"] for h,v in rrn["histories"].items()}=={h:v["final_digest"] for h,v in ref["histories"].items()}
    checks["resume_preserves_completed_prior_history_fixture"]=True
    reject(lambda:cw.run("DEV","BIND",202610166101,OUT/"resumed_counter",auth))
    checks["committed_job_replay_rejected"]=True
    mutations={
        "N_OLD_permission":lambda r:r["histories"]["N_OLD"]["windows"][0].__setitem__("allowed",True),
        "N_OLD_theta":lambda r:r["histories"]["N_OLD"]["windows"][0].__setitem__("actor_steps",1),
        "N_OLD_native":lambda r:r["histories"]["N_OLD"]["windows"][0].__setitem__("alpha_l1",1.),
        "missing_boundary_readout":lambda r:r["histories"]["W"]["boundaries"][0]["readings"].pop("LEGACY"),
        "wrong_grade":lambda r:r["histories"]["W"]["points"][0]["readings"]["MEM"]["rows"][0].__setitem__("status","TRUNCATED"),
        "duplicate_window":lambda r:r["histories"]["W"]["windows"].append(copy.deepcopy(r["histories"]["W"]["windows"][0])),
        "wrong_stream":lambda r:r.__setitem__("data_stream_sha256","bad"),
        "wrong_byte_count":lambda r:r["histories"]["W"]["resource"].__setitem__("answer_bytes",0),
        "wrong_birth":lambda r:r["histories"]["W"]["birth_receipts"][0].__setitem__("canonical_B_sha256","bad"),
        "changed_source":lambda r:r["source_identity"].__setitem__("bind_kernel.py","bad"),
        "technical_as_science":lambda r:r.__setitem__("schema","SYNTHETIC_ONLY"),
    }
    receipt=c.read(OUT/"native_counter/RECEIPT.json")
    rejected=[]
    for name,mutate in mutations.items():
        view=copy.deepcopy(receipt);mutate(view);reject(lambda v=view:auditor.audit(v));rejected.append(name)
    checks["independent_audit_tamper_rejections"]=rejected
    cw.Learner=original_learner;cw.qualified_worker.admission=original_admission
    evaluator.score=original_score;cw.campaign_diagnostics.measure=original_diagnostics

    # Constant and nonconstant world intervals, no invented efficacy data.
    assert analysis.interval([.1]*16)["lower"]==.1
    assert analysis.interval([0.]*16)["two_sided_p"]==1.
    assert analysis.interval([-.1,.1]*8)["lower"]<0
    def m(g=.1,q=.1,old=1.,held=.1):
        return dict(G=g,Q=q,OLD_end=old,OLD_rest=old,OLD_final=old,NEW_final=old,final_HELD=held)
    worlds={str(w):dict(arms={
        "NATIVE":{"W":{"MEM":m(),"LEGACY":m(.12,.12)},"N_OLD":{"MEM":m(old=.1),"LEGACY":m(old=.1)}},
        "BIND":{"W":{"MEM":m(.5,.5,held=.8)},"N_OLD":{"MEM":m(old=.1)}},
        "BIND-ADAPT":{"W":{"MEM":m(.4,.4,held=.8)},"N_OLD":{"MEM":m(old=.1)}},
        "BIND-FB":{"W":{"MEM":m(.3,.3,held=.8)},"N_OLD":{"MEM":m(old=.1)}},
        "BIND-FROZEN":{"W":{"MEM":m()},"N_OLD":{"MEM":m(old=.1)}},
        "BIND-1STEP":{"W":{"MEM":m()},"N_OLD":{"MEM":m(old=.1)}}},
        shortcut=dict(G=.1,final_HELD=.1,OLD_frequency_end=.1,OLD_frequency_final=.1)) for w in range(16)}
    assert analysis.select(worlds)["selected"]=="BIND"
    assert analysis.confirm(worlds,"BIND")["global_conjunctive_claim_passed"]
    worlds["0"]["arms"]["BIND"]["W"]["MEM"]["OLD_final"]=0.
    assert not analysis.confirm(worlds,"BIND")["global_conjunctive_claim_passed"]
    checks["selection_confirmation_retention_Holm_fixtures"]=True
    data=[(b"a",b"1"),(b"b",b"2")]
    scores=analysis.shortcut_scores(data,data)
    assert scores["components"]["exact"]==1.
    assert analysis.shortcut_scores([],data)["envelope"]==0.
    assert evaluator.sample_auc([(0,0),(3,1)])==.5
    assert evaluator.cpu_auc([(0,0),(1,1)],2)==.5
    checks["shortcuts_and_common_axis_fixtures"]=True
    bind_kernel.Learner.train_window=real_train
    result=dict(schema="NATIVE_BIND_CAMPAIGN_QUALIFICATION_V1",accepted=True,
                scope="Read-only native checkpoint + synthetic worker/statistical/permission fixtures",
                checks=checks,additional_native_training_records=0,science_jobs=0,
                technical_training_records_remain=256,source_identity=c.identity(),
                kernel_identity=core_identity,fixture_counter_objects_are_not_science=True,
                automatic_launch_authorized=False,utc=c.now())
    c.write(HERE/"CAMPAIGN_QUALIFICATION.json",result)
    print(json.dumps(dict(accepted=True,checks=len(checks),native_training_records=0,science_jobs=0)))


if __name__=="__main__":main()
