"""Independent algebra/checksum audit and compact non-raw result package."""
import os,time,json,hashlib,zipfile,resource
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parent;P=ROOT/'minifly_v171_dev_20261026'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads((H/p).read_text())
def save(p,a):(H/p).write_text(json.dumps(a,indent=2,allow_nan=False)+'\n')
def main():
 t=time.process_time();old=json.loads((P/'CALIBRATION_SOURCE_LOCK.json').read_text());assert all(sha(ROOT/p)==h for p,h in old.items())
 a=load('ALL80_RESULTS.json');c=load('CREDIT_RESULTS.json');b=load('CEILING_RESULTS.json');f=load('FROZEN_CEILING_RESULTS.json')
 for report in (a,c,b,f):
  for p,h in report.get('input_sha256',{}).items():assert sha(ROOT/p)==h,p
  for p,h in report.get('detail_sha256',{}).items():assert sha(H/p)==h,p
 for p,h in load('CREDIT_SOURCE_LOCK.json').items():assert sha(ROOT/p)==h,p
 words=json.loads((ROOT/'minifly_cls_dev_20261017/OUTPUT_CODEBOOK.json').read_text())['words_by_class'];C=np.array([[1 if w&(1<<j) else -1 for j in range(16)] for w in words]);assert np.all(C.sum(1)==0)
 reads=load('local/FROZEN_PREFIX_READS.json');lookup={(x['world'],x['stage'],x['id'],x['site']):x for x in reads};checks=0
 for row in load('local/FROZEN_CEILING_DETAILS.json'):
  r=lookup[(row['world'],row['stage'],row['id'],row['site'])];v=np.array(r['v']);y=r['y'];A=(C[y]-C)*v/4;mask=np.arange(257)!=y
  for variant,rad in [('original_box',.5),('parameter_cap_envelope',.5*np.tanh(1.25))]:
   g=1+rad*np.sign(C[y]*v);margin=float((A@g)[mask].min());rr=row['variants'][variant]
   assert rr['certificate_valid'] and abs(margin-rr['optimal_min_margin'])<1e-7
   checks+=1
 convex_checks=0
 for row in load('local/FROZEN_CONVEX_DETAILS.json'):
  r=lookup[(row['world'],row['stage'],row['id'],row['site'])];v=np.array(r['v']);y=r['y'];g=1+.5*np.sign(C[y]*v);z=C@(g*v)/4;zz=z-z.max();ce=float(np.log(np.exp(zz).sum())-zz[y]);assert abs(ce-row['result']['CE_nats'])<1e-7;assert row['objective_gap_qualified'];convex_checks+=1
 details=load('local/CREDIT_DETAILS.json');rec=[];late=[]
 for x in details:
  if x['feedback']!='ON':continue
  den=sum(x['blocks'][k]['full_L2']**2 for k in ('U','W','R','b'))**.5
  rec.append(sum(x['blocks'][k]['three_minus_one_L2']**2 for k in ('U','W','R','b'))**.5/den)
  late.append(sum(x['blocks'][k]['full_minus_three_L2']**2 for k in ('U','W','R','b'))**.5/den)
 assert c['own_query_authentications']==960 and c['teacher_prefix_records']==320 and c['native_target_prefix_sites']==920
 assert c['new_training_records']==0 and a['qualification_changed']==False
 assert sum(b['counts'][k] for k in ('data_LP_solves','synthetic_LP_solves'))+f['counts']['LP_solves']+751==4434
 failed_cpu=sum(json.loads(p.read_text()).get('CPU_seconds',0) for p in (H/'attempts').rglob('CREDIT_FAILURE.json'))
 metered=c['CPU_seconds']+a['CPU_seconds']+b['resources']['CPU_seconds']+f['resources']['CPU_seconds']+failed_cpu
 audit={'status':'ACCEPTED_CORRECTIVE_DIAGNOSTIC_WITH_DECLARED_PROCESS_LIMITS','parent_source_hashes_verified':len(old),'parent_source_changed':False,'source_and_input_hashes_pass':True,'checkpoint_and_saved_panel_authentications':960,'EXACT1_independent_numerical_check':load('EXACT1_NUMERICAL.json'),'frozen_LP_independent_analytic_corner_checks':checks,'convex_analytic_optimum_checks':convex_checks,'codebook_balanced':True,'recurrent_block_three_minus_one_over_full_median':float(np.median(rec)),'recurrent_block_full_minus_three_over_full_median':float(np.median(late)),'new_native_training_records':0,'new_held_assessment_decisions':0,'parent_histories_replayed':False,'original_lost_nine_records_remain_missing':True,'new_native_own_queries':960,'new_native_TF_records':320,'new_native_TF_target_sites':920,'gradient_target_reuses':5520,'discarded_clone_steps':1920,'synthetic_loss_evaluations_including_failed_attempt':24,'LP_solves_including_failed_attempt':4434,'convex_fits':122,'convex_objective_gradient_target_reuses':1473,'analytic_corner_verifications':checks+convex_checks,'verified_saved_panel_rows':a['saved_rows_verified'],'verified_saved_panel_TF_sites':a['saved_TF_target_reuses'],'metered_CPU_before_final_audit_seconds':metered,'final_audit_CPU_seconds':time.process_time()-t,'peak_RSS_native_runner_bytes':c['peak_RSS_bytes'],'limits':['First failed LP reduction CPU unretained','Interactive authoring and shell commands not fully metered; no end-to-end efficiency claim','No fitted EXACT1 history; fixed-checkpoint derivative probe only','Independent target-informed gates are not shared-policy learning'],'historical_verdict':'REFERENCE_GATE_NOT_PASSED','new_core_adopted':False}
 save('FINAL_AUDIT.json',audit)
 save('SUMMARY.json',{'version':'V171-R1','status':'CORRECTIVE_DIAGNOSTIC_COMPLETE','historical_verdict':'REFERENCE_GATE_NOT_PASSED','new_training_records':0,'unseen_addition_improvement_established':False,'complaint_note_commit':'fd80e118761d5106d804f7b39f3cab40e424d5f8','report':'REPORT.md','revision':'REVISED_DESIGN.md','audit':'FINAL_AUDIT.json','GitHub_pushed':False})
 files=[p for p in H.rglob('*') if p.is_file() and 'local' not in p.relative_to(H).parts and '__pycache__' not in p.parts and p.name not in ('ACTIVE.json','credit.log','RESULT_BUNDLE.zip','PACKAGE.json','BUNDLE_MANIFEST.json') and p.suffix!='.log']
 manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)};save('BUNDLE_MANIFEST.json',manifest)
 with zipfile.ZipFile(H/'RESULT_BUNDLE.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in files+[H/'BUNDLE_MANIFEST.json']:z.write(p,p.relative_to(ROOT))
 with zipfile.ZipFile(H/'RESULT_BUNDLE.zip') as z:assert z.testzip() is None
 save('PACKAGE.json',{'status':'COMPLETE','files':len(files)+1,'bundle':'RESULT_BUNDLE.zip','sha256':sha(H/'RESULT_BUNDLE.zip'),'contains_raw_data':False,'contains_checkpoints':False,'CRC_verified':True})
 print(json.dumps({'status':audit['status'],'metered_CPU':metered,'recurrent_delayed_median':audit['recurrent_block_three_minus_one_over_full_median'],'bundle_sha256':sha(H/'RESULT_BUNDLE.zip')}))
if __name__=='__main__':main()
