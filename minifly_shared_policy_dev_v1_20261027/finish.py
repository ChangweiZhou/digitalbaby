"""Endpoint comparisons and compact package; no model imports or training."""
import json,hashlib,zipfile,time,itertools
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parent;T0=time.process_time()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads((H/p).read_text())
def save(p,obj):(H/p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def main():
 r=load('RUN_RESULTS.json');a=load('FINAL_AUDIT.json');assert a['status']=='ACCEPTED';spec=load('EXECUTION_SPEC.json');data=json.loads((ROOT/'minifly_v171_dev_20261026/local/TRAIN_DATA.json').read_text());comparisons=[]
 for world,w in r['worlds'].items():
  old={x['id'] for x in data[world]['OLD']};initial={k:set(v['correct_ids'])&old for k,v in w['panels']['OLD16'].items()};final={k:set(v['correct_ids'])&old for k,v in w['panels']['NEW16'].items()};initial['NATIVE']=set(w['native']['correct_ids'])&old;final['NATIVE']=initial['NATIVE']
  for x,y in itertools.combinations(initial,2):
   common=initial[x]&initial[y];comparisons.append(dict(world=int(world),arm_a=x,arm_b=y,common_initial_OLD=len(common),retention_measurable=bool(common),a_common_lost=len(common-final[x]),b_common_lost=len(common-final[y])))
  final_sets={k:set(v['correct_ids']) for k,v in w['panels']['NEW16'].items()};assert final_sets['ADAM3']==final_sets['ADAM1']==final_sets['ADAM3-R0']
 save('ENDPOINT_COMPARISONS.json',dict(paired_common_initial_retention=comparisons,final_Adam_one_three_R0_correct_sets_identical=True,new_model_queries=0,source_run_results_sha256=sha(H/'RUN_RESULTS.json'),source_audit_sha256=sha(H/'FINAL_AUDIT.json')))
 save('SUMMARY.json',dict(identity='SP-COND-V1',status='THREE_CYCLES_COMPLETE_AUDITED',descriptive_result='No complete-answer gain in world711; one shared NEW answer gained in world712 under ADAM3/ADAM1/R0',binary_efficacy_gate=False,all_planned_fits_completed=True,worlds_previously_exposed=True,independent_worlds=2,main_fitted_records=12800,main_fitted_target_exposures=36800,technical_fitted_records=5,technical_targets=10,new_native_teaching_records=0,held_assessment_decisions=0,all_fitted_arms_reference_instruments=True,gradual_warmup_preserved=True,attention_added=False,arithmetic_structure_added=False,new_core_adopted=False,unseen_addition_improvement_established=False,main_CPU_seconds=r['CPU'],audit_CPU_seconds=a['CPU_seconds'],report='REPORT.md',revision='REVISED_DESIGN.md',audit='FINAL_AUDIT.json',GitHub_pushed=False))
 save('STATUS.json',dict(identity='SP-COND-V1',status='THREE_CYCLES_COMPLETE_AUDITED',all_fits_completed=True,new_native_teaching_records=0,held_assessment_decisions=0,new_core_adopted=False))
 # Keep the completed historical experiment and its design pointers untouched.
 p=ROOT/'PROJECT_DOCUMENTS/CURRENT_OUTPUT_DIAGNOSTIC.json';history=ROOT/'PROJECT_DOCUMENTS/OUTPUT_DIAGNOSTIC_POINTER_HISTORY/pre_SHARED_POLICY_COMPLETION_20261027.json';assert not history.exists();history.write_bytes(p.read_bytes());p.write_text(json.dumps(dict(identity='SP-COND-V1',status='THREE_CYCLES_COMPLETE_AUDITED',report=str((H/'REPORT.md').relative_to(ROOT)),revision=str((H/'REVISED_DESIGN.md').relative_to(ROOT)),audit=str((H/'FINAL_AUDIT.json').relative_to(ROOT)),package=str((H/'RESULT_BUNDLE.zip').relative_to(ROOT)),main_fitted_records=12800,main_fitted_target_exposures=36800,new_native_teaching_records=0,held_assessment_decisions=0,new_core_adopted=False,unseen_addition_improvement_established=False,arithmetic_pointer_replaced=False,previous_pointer=str(history.relative_to(ROOT)),GitHub_pushed=False),indent=2)+'\n')
 assert all(sha(ROOT/p)==v for p,v in load('MAIN_SOURCE_LOCK.json').items())
 save('DELIVERY_AUDIT.json',dict(status='PASS',fixed_scientific_source_unchanged=True,raw_arrays_and_checkpoints_local_only=True,scalar_finite_difference_loss_evaluations=24,synthetic_derivative_target_fixtures=3,optimizer_unit_fixtures='SGD parity on discarded clones; three imposed-gradient Adam tests, not arriving training records',supplemental_model_queries=0,CPU_seconds=time.process_time()-T0))
 paths=[]
 for base in [H,ROOT/'minifly_shared_policy_design_v1_20261027']:
  for p in base.rglob('*'):
   if p.is_file() and 'local' not in p.relative_to(base).parts and '__pycache__' not in p.parts and p.suffix!='.log' and p.name not in ('ACTIVE.json','LAUNCH.json','BUNDLE_MANIFEST.json','PACKAGE.json','RESULT_BUNDLE.zip'):paths.append(p)
 manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)};save('BUNDLE_MANIFEST.json',manifest)
 with zipfile.ZipFile(H/'RESULT_BUNDLE.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in paths+[H/'BUNDLE_MANIFEST.json']:z.write(p,p.relative_to(ROOT))
 with zipfile.ZipFile(H/'RESULT_BUNDLE.zip') as z:assert z.testzip() is None and all(not x.endswith(('.npz','.pkl','.log')) and 'local' not in Path(x).parts for x in z.namelist())
 save('PACKAGE.json',dict(status='COMPLETE',bundle='RESULT_BUNDLE.zip',files=len(paths)+1,sha256=sha(H/'RESULT_BUNDLE.zip'),bytes=(H/'RESULT_BUNDLE.zip').stat().st_size,contains_raw_data=False,contains_checkpoints=False,CRC_verified=True))
 print(json.dumps(dict(status='COMPLETE',package_bytes=(H/'RESULT_BUNDLE.zip').stat().st_size,files=len(paths)+1)))
if __name__=='__main__':main()
