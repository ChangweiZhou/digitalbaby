"""Saved-only certificate checks. Never trains or reruns a scientific trajectory."""
import json,hashlib,zipfile
from pathlib import Path
H=Path(__file__).resolve().parent
load=lambda n:json.loads((H/n).read_text())
for name in ['CYCLE1.json','CYCLE2.json','CYCLE3.json','SOURCE_PARITY.json','INDEPENDENT_NUMERIC_AUDIT.json','STRESS_AUDIT.json','EVALUATOR_AUDIT.json']:
 assert load(name)['status']=='PASS',name
s=load('SUMMARY.json');r=load('RESOURCE_ADMISSION.json');a=load('EXPOSURE_AUDIT.json');c=load('CYCLE3.json')
assert s['science_records']==0 and s['science_targets']==0
assert s['technical_records']==a['neural_optimizer_record_applications']==c['technical_records']==32
assert s['technical_targets']==a['neural_optimizer_target_applications']==c['technical_target_applications']==1064
assert load('CYCLE1.json')['learned_scalars']==37313
assert r['complete_plan_admitted'] and r['full_CPU_projection_with_1_5_reserve']<=900 and r['output_projection_bytes']<=268435456
assert r['checkpoint_max_bytes']<=1048576 and r['inference_bytes']<=8388608 and max(r['peak_RSS_bytes'],r['profile_peak_RSS_bytes'],r['full_batch_stress_peak_RSS_bytes'])<=1073741824
assert r['scheduled_science_records']==4480 and r['scheduled_science_target_applications']==227148
assert all(v['correction']['qualified'] for v in load('DATA_AUDIT.json')['worlds'].values())
assert not (H/'FAILURE.json').exists()
lock=load('SOURCE_LOCK.json')
for name,h in lock['files'].items():assert hashlib.sha256((H/name).read_bytes()).hexdigest()==h,name
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(H.iterdir()) if p.suffix in ('.json','.md','.py','.txt') and p.name not in ('FINAL_AUDIT.json','PACKAGE.json')}
cert={'status':'ACCEPTED_TECHNICAL_ONLY','saved_only':True,'new_training_or_validation_trajectories':0,'technical_records':32,'technical_targets':1064,'scientific_records':0,'scientific_targets':0,'resource_admitted':True,'whole_science_run_completed':False,'official_lifecycle_integration_still_required':True,'external_raw_reproduction':False,'files':files}
(H/'FINAL_AUDIT.json').write_text(json.dumps(cert,indent=2)+'\n')
with zipfile.ZipFile(H/'RESULT_BUNDLE.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in [*files,'FINAL_AUDIT.json']:z.write(H/name,name)
(H/'PACKAGE.json').write_text(json.dumps({'status':'COMPLETE_TECHNICAL_PACKAGE','filename':'RESULT_BUNDLE.zip','sha256':hashlib.sha256((H/'RESULT_BUNDLE.zip').read_bytes()).hexdigest(),'bytes':(H/'RESULT_BUNDLE.zip').stat().st_size,'raw_data_or_weights_included':False,'source_corpora_or_framework_packages_included':False,'not_scientific_efficacy_package':True},indent=2)+'\n')
print('SAVED_ONLY_AUDIT_ACCEPTED_TECHNICAL_ONLY',len(files),'files')
