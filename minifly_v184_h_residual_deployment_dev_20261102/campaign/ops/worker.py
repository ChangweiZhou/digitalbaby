"""Launch and compact packaging only; qualified learning/decision code unchanged."""
from pathlib import Path
import sys,json,os,traceback,hashlib,zipfile,runpy,time
OPS=Path(__file__).resolve().parent;CAM=OPS.parent;DEV=CAM.parent;sys.path.insert(0,str(DEV))
def read(n):return json.loads((CAM/n).read_text())
def save(n,v):
 p=CAM/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def committed():return read('COMMIT_INDEX.json') if (CAM/'COMMIT_INDEX.json').exists() else []
try:
 save('STATUS.json',{'status':'RUNNING','stage':'FIXED_M1_MAIN','arms':['NORMAL','HRES','NORM_SHAM']})
 runpy.run_path(str(OPS/'run_authorized.py'),run_name='__main__')
 audit=read('FINAL_AUDIT.json');states=read('STATE_BUNDLE.json');assert audit['status']=='ACCEPTED_RUN_LOCAL' and states['all15_full_states_and_required_site_arrays']
 names=['AUTHORIZATION.json','OPERATIONS_CONVENTIONS.json','QUALIFICATION.json','SOURCE_LOCK.json','LAUNCH.json','MANIFEST.json','COUNTERS.json','COMMIT_INDEX.json','RUN_COMPLETE.json','DECISION.json','SAVED_SITE_AUDIT.json','SUMMARY.json','REPORT.md','STATE_BUNDLE.json','FINAL_AUDIT.json']
 p=CAM/'RESULT_BUNDLE.zip'
 with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
  for n in names:z.write(CAM/n,n)
  for n in ('bootstrap.py','learner.py','data.py','evaluate.py','protocol.py','run.py','finalize.py','REVISED_DESIGN.md','REVISED_SPEC.json'):z.write(DEV/n,'source/'+n)
  for f in OPS.glob('*.py'):z.write(f,'source/operations/'+f.name)
 save('PACKAGE.json',{'status':'COMPLETE','path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'owner_state_archive':states,'raw_states_sites_or_corpora_in_compact_bundle':False})
 state=read('STATUS.json');state.update(package='COMPLETE');save('STATUS.json',state);print('V184_COMPLETED',audit['verdict'],flush=True)
except BaseException as e:
 kind='CONTROL_NOT_QUALIFIED' if type(e).__name__=='ControlNotQualified' else 'EXECUTION_FAILURE'
 save('FAILURE.json',{'kind':kind,'error':repr(e),'traceback':traceback.format_exc(),'committed_full_lives':committed(),'counters':read('COUNTERS.json') if (CAM/'COUNTERS.json').exists() else {},'no_retry_or_partial_efficacy_claim':True});save('STATUS.json',{'status':'STOPPED_CONTROL_NOT_QUALIFIED' if kind=='CONTROL_NOT_QUALIFIED' else 'FAILED','error':repr(e),'data_preserved':True});raise
