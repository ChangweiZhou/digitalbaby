import hashlib
import json
import os
import time
from pathlib import Path
import bootstrap


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    os.replace(temporary, path)


def source_identity():
    root = bootstrap.HERE
    files = sorted(root.glob("*.py")) + [root/"OUTPUT_CODEBOOK.json"]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


class Ledger:
    def __init__(self,directory,cycle):
        self.path=Path(directory)/"EXECUTION_LEDGER.json"
        if self.path.exists():raise ValueError("never overwrite executed attempt")
        self.cycle=cycle;self.rows=[];self.fixtures=[];self.started=time.time();self.flush()
    def flush(self):
        write(self.path,dict(cycle=self.cycle,rows=self.rows,fixtures=self.fixtures,
            full_record_applications=sum(r.get("records_executed",0) for r in self.rows),
            charged_record_applications=sum(max(r["records_planned"],r.get("records_executed",0)) for r in self.rows),
            started_epoch=self.started,updated_epoch=time.time()))
    def call(self,label,model,fn,planned=4):
        all_paths=list((bootstrap.HERE/"cycles").glob("cycle*/attempt*/EXECUTION_LEDGER.json"))
        totals=[json.loads(p.read_text()) for p in all_paths]
        total=sum(v["charged_record_applications"] for v in totals)
        subtotal=sum(v["charged_record_applications"] for v in totals if v["cycle"]==self.cycle)
        if total+planned>512 or subtotal+planned>{1:128,2:128,3:256}[self.cycle]:
            raise ValueError("technical record cap;no automatic extension")
        before=model.resource["records"]
        row=dict(label=label,status="STARTED",records_planned=planned,records_executed=0,started_epoch=time.time())
        self.rows.append(row);self.flush()
        try:
            result=fn();row.update(status="COMPLETED",records_executed=model.resource["records"]-before)
            if row["records_executed"]!=planned:raise AssertionError("full record count mismatch")
            return result
        except Exception as exc:
            row.update(status="FAILED",records_executed=model.resource["records"]-before,error=repr(exc));raise
        finally:
            row["ended_epoch"]=time.time();self.flush()
    def fixture(self,label,fn,native_events=0,payload_bytes=0):
        row=dict(label=label,native_event_calls=native_events,payload_bytes=payload_bytes,status="STARTED")
        self.fixtures.append(row);self.flush();start=time.process_time()
        try:
            result=fn();row["status"]="COMPLETED";return result
        except Exception as exc:
            row.update(status="FAILED",error=repr(exc));raise
        finally:
            row["CPU_seconds"]=time.process_time()-start;self.flush()


def admit_technical():
    return True
