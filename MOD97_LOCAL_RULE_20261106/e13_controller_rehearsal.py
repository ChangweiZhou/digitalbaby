"""Production subprocess/controller rehearsal on explicitly synthetic technical fixtures."""
import tempfile,json
from pathlib import Path
import e13_campaign as C

def main():
 root=Path(tempfile.mkdtemp(prefix='e13-controller-'));C.R=root;C.O=root/'exp45';C.O.mkdir();(C.O/'logs').mkdir()
 def job(key,part,delay=0.,exitcode=0):
  receipt=f'{key}.json';p=C.O/receipt
  code=f"import json,time,pathlib,sys;p=pathlib.Path({str(p)!r});p.write_text(json.dumps(dict(stages=[])));time.sleep({delay});p.write_text(json.dumps(dict(stages=[dict(task='compose',scores={{}})])));sys.exit({exitcode})"
  return dict(key=key,part=part,receipt=receipt,stages=1,command=['-c',code],env={})
 fast=job('same_fast','A',.2);slow=job('same_slow','A',1.)
 done,failed=C.queue([fast,slow],C.run,C.validate,lambda *a:None,2)
 assert len(done)==2 and not failed and all(x['status']=='complete' for x in done)
 fail=job('intentional_exit7','F',0,7);skip=job('must_skip','F');keep=job('other_part','G')
 done,failed=C.queue([fail,skip,keep],C.run,C.validate,lambda *a:None,1)
 assert failed=={'F'} and len(done)==2 and not (C.O/'E13_JOB_must_skip.json').exists()
 result=dict(passed=True,actual_production_subprocesses=True,unfinished_sibling_during_fast_completion=True,exact_receipt_validation=True,intentional_exit7_isolates_part=True,other_part_completed=True,scientific_training=False,fixture_directory=str(root))
 target=Path(__file__).resolve().parent/'E13_CONTROLLER_REHEARSAL.json';target.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
