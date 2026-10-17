"""Whole-window write-ahead accounting; no retry or replay admission."""
import time
from pathlib import Path
import campaign_common as c
class Ledger:
    def __init__(self,directory):
        self.path=Path(directory)/'EXECUTION_LEDGER.json'
        if self.path.exists():raise PermissionError('Never overwrite executed science')
        self.rows=[];self.started=time.time();self.flush()
    def flush(self):
        c.write(self.path,dict(schema='NATIVE_BIND_SCIENCE_LEDGER_V2_1',science=True,
            rows=self.rows,started_epoch=self.started,updated_epoch=time.time(),
            full_record_applications=sum(r['records_executed'] for r in self.rows)))
    def call(self,label,model,fn):
        if any(r['label']==label for r in self.rows):raise PermissionError('Duplicate window')
        if any(r['status']!='COMPLETED' for r in self.rows):raise PermissionError('Incomplete previous window')
        before=model.resource['records']
        row=dict(label=label,status='STARTED',records_planned=4,records_executed=0,started_epoch=time.time())
        self.rows.append(row);self.flush()
        try:
            result=fn();row['records_executed']=model.resource['records']-before
            if row['records_executed']!=4:raise AssertionError('Incomplete training window')
            row['status']='COMPLETED';return result
        except BaseException as exc:
            row.update(status='FAILED',records_executed=model.resource['records']-before,error=repr(exc));raise
        finally:self.flush()
