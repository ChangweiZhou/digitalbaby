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
    def __init__(self, directory):
        self.path = Path(directory)/"EXECUTION_LEDGER.json"
        self.rows = []
        self.started = time.time()
        self.flush()

    def flush(self):
        write(self.path, dict(rows=self.rows,
            full_record_applications=sum(r.get("records_executed",0) for r in self.rows),
            started_epoch=self.started, updated_epoch=time.time()))

    def call(self, label, model, fn, planned=4):
        before = model.resource["records"]
        row = dict(label=label, status="STARTED", records_planned=planned,
                   records_executed=0, started_epoch=time.time())
        self.rows.append(row); self.flush()
        try:
            result = fn()
            row.update(status="COMPLETED", records_executed=model.resource["records"]-before)
            return result
        except Exception as exc:
            row.update(status="FAILED", records_executed=model.resource["records"]-before,
                       error=repr(exc))
            raise
        finally:
            row["ended_epoch"]=time.time(); self.flush()


def admit_technical():
    paths = list((bootstrap.HERE/"cycles").glob("cycle*/attempt*/**/EXECUTION_LEDGER.json"))
    consumed = sum(json.loads(p.read_text())["full_record_applications"] for p in paths)
    if consumed >= 256:
        raise RuntimeError("technical training-record cap exhausted")
    return consumed
