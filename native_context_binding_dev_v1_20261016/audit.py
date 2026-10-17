"""Independent read-only receipt checks; never reruns a learner."""
import copy
import hashlib
import json
from pathlib import Path
import bootstrap
import bind_evaluator as evaluator
from worker import audit_receipt


def audit(receipt):
    audit_receipt(receipt)
    stage=receipt["stage"]
    expected_schema="NATIVE_BIND_TECHNICAL_SMOKE_V1" if stage=="SMOKE" else "NATIVE_BIND_SCIENCE_V1"
    if receipt["schema"]!=expected_schema:
        raise AssertionError("technical/science envelope mismatch")
    if stage=="SMOKE" and receipt["world"] not in (202610166001,202610166002,202610166003):
        raise AssertionError("wrong technical world")
    for name in ("bootstrap.py","bind_kernel.py","bind_native.py","OUTPUT_CODEBOOK.json","bind_evaluator.py","worker.py"):
        current=hashlib.sha256((bootstrap.HERE/name).read_bytes()).hexdigest()
        if receipt["source_identity"].get(name)!=current:
            raise AssertionError("operative source mismatch")
    groups=evaluator.split(receipt["world"])
    truth={k:{f"{a}+{b}=".encode().hex():str(a+b).encode().hex() for a,b in rows} for k,rows in groups.items()}
    for data in receipt["histories"].values():
        for point in data["points"]:
            for scores in point["readings"].values():
                if {r["prompt"] for r in scores["rows"]}-truth["held"].keys():
                    raise AssertionError("score is not a never-taught relation")
                for row in scores["rows"]:
                    if row["answer"]!=truth["held"][row["prompt"]]:
                        raise AssertionError("wrong evaluator answer")
        for boundary in data["boundaries"]:
            for name,scores in boundary["scores"].items():
                for row in scores["rows"]:
                    if truth[name].get(row["prompt"])!=row["answer"]:
                        raise AssertionError("wrong boundary membership/answer")
    return dict(accepted=True,records=receipt["records_executed"],stage=stage,
                science=stage!="SMOKE",scored_relations_checked=True)


def tamper_suite(receipt):
    mutations={
        "N_old_enabled":lambda r:r["histories"]["N_OLD"]["windows"][0].__setitem__("allowed",True),
        "N_old_theta_update":lambda r:r["histories"]["N_OLD"]["windows"][0].__setitem__("actor_steps",1),
        "N_old_native_write":lambda r:r["histories"]["N_OLD"]["windows"][0].__setitem__("alpha_l1",1.),
        "duplicate_window":lambda r:r["histories"]["W"]["windows"].append(copy.deepcopy(r["histories"]["W"]["windows"][0])),
        "wrong_count":lambda r:r.__setitem__("records_executed",r["records_executed"]+4),
        "technical_as_science":lambda r:r.__setitem__("schema","NATIVE_BIND_SCIENCE_V1"),
        "changed_kernel":lambda r:r["source_identity"].__setitem__("bind_kernel.py","tampered"),
        "wrong_held_answer":lambda r:r["histories"]["W"]["points"][0]["readings"]["MEM"]["rows"][0].__setitem__("answer","ffff"),
        "wrong_status_grade":lambda r:r["histories"]["W"]["points"][0]["readings"]["MEM"]["rows"][0].__setitem__("correct",True),
    }
    rejected=[]
    for name,mutate in mutations.items():
        view=copy.deepcopy(receipt);mutate(view)
        try:audit(view)
        except AssertionError:rejected.append(name)
        else:raise AssertionError("tamper accepted: "+name)
    return rejected
