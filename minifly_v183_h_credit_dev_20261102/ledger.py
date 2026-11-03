"""Persist every charge before executing a fixture; no uncharged synthetic work."""
import json,time,resource,sys,math
from pathlib import Path
H=Path(__file__).resolve().parent
class Ledger:
 def __init__(self):
  self.path=H/'EXPOSURE_LEDGER.json';self.start=time.process_time();self.wall=time.time();self.rows=[]
  if self.path.exists():raise RuntimeError('Existing qualification ledger: no automatic rerun')
  self.write()
 def write(self):
  technical=[r for r in self.rows if r['category']=='technical'];frozen=[r for r in self.rows if r['category']=='frozen_assay'];counts={'technical_base_record_equivalents':sum(r['base_record_equivalents'] for r in technical),'technical_loss_record_equivalents':sum(r['loss_record_equivalents'] for r in technical),'technical_target_evaluations':sum(r['target_evaluations'] for r in technical),'technical_optimizer_updates':sum(r.get('optimizer_updates',0) for r in technical),'frozen_assay_record_equivalents':sum(r['loss_record_equivalents'] for r in frozen),'frozen_assay_target_evaluations':sum(r['target_evaluations'] for r in frozen)}
  assert counts['technical_base_record_equivalents']<=32 and counts['technical_loss_record_equivalents']<=96 and counts['technical_target_evaluations']<=12384
  assert counts['frozen_assay_record_equivalents']<=636 and counts['frozen_assay_target_evaluations']<=24576
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024);cpu=time.process_time()-self.start+45;wall=time.time()-self.wall+300
  if cpu>900 or wall>1800 or rss>1073741824:raise RuntimeError('RESOURCE_NOT_ADMITTED')
  self.path.write_text(json.dumps({'started_unix':self.wall,'rows':self.rows,'totals':counts,'charged_CPU_seconds':cpu,'wall_seconds_including_300_setup_allowance':wall,'peak_RSS_bytes':rss,'historical_probe_CPU_carried':30,'setup_CPU_allowance':15,'technical_includes_synthetic_and_gradient_only':True},indent=2)+'\n');return counts
 def add(self,label,n=0,paths=1,base_records=1,updates=0,category='technical',loss_records=None):
  row={'label':label,'category':category,'sites':n,'paths':paths,'base_record_equivalents':base_records,'loss_record_equivalents':base_records*paths if loss_records is None else loss_records,'target_evaluations':n*paths,'optimizer_updates':updates};self.rows.append(row);self.write()
