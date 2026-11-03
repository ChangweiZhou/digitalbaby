"""Operations wrapper only: no learning-code, dose or gate changes."""
from pathlib import Path
import sys,json,traceback,hashlib,zipfile,runpy,os
HERE=Path(__file__).resolve().parent;CAM=HERE.parent;DEV=CAM.parent
sys.path.insert(0,str(DEV))
def save(name,v):
 p=CAM/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
try:
 save('STATUS.json',{'status':'RUNNING','stage':'FIXED_MAIN','scientific_lives':3})
 runpy.run_path(str(HERE/'run_authorized.py'),run_name='__main__')
 audit=json.loads((CAM/'FINAL_AUDIT.json').read_text());assert audit['status']=='ACCEPTED_RUN_LOCAL'
 states=json.loads((CAM/'STATE_BUNDLE.json').read_text());assert states['all30_states_and_site_arrays']
 files=['AUTHORIZATION.json','OPERATIONS_AMENDMENT.json','QUALIFICATION.json','SOURCE_LOCK.json','LAUNCH.json','MANIFEST.json','COUNTERS.json','COMMIT_INDEX.json','RUN_COMPLETE.json','DECISION.json','SAVED_SITE_AUDIT.json','SUMMARY.json','REPORT.md','STATE_BUNDLE.json','FINAL_AUDIT.json']
 with zipfile.ZipFile(CAM/'RESULT_BUNDLE.zip','w',zipfile.ZIP_DEFLATED) as z:
  for name in files:z.write(CAM/name,name)
  for name in ('bootstrap.py','learner.py','metrics.py','run.py','reduce.py','finalize.py','REVISED_SPEC.json','REVISED_DESIGN.md'):
   z.write(DEV/name,'source/'+name)
  for p in HERE.glob('*.py'):z.write(p,'source/operations/'+p.name)
 p=CAM/'RESULT_BUNDLE.zip';save('PACKAGE.json',{'status':'COMPLETE','path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'owner_state_archive':states,'raw_states_and_site_arrays_in_compact_bundle':False})
 current=json.loads((CAM/'STATUS.json').read_text());current.update(package='COMPLETE');save('STATUS.json',current);print('V183_COMPLETED',audit['verdict'],flush=True)
except BaseException as e:
 save('FAILURE.json',{'error':repr(e),'traceback':traceback.format_exc(),'counters':json.loads((CAM/'COUNTERS.json').read_text()) if (CAM/'COUNTERS.json').exists() else {},'committed_lives':json.loads((CAM/'COMMIT_INDEX.json').read_text()) if (CAM/'COMMIT_INDEX.json').exists() else []});save('STATUS.json',{'status':'FAILED','error':repr(e)});raise
