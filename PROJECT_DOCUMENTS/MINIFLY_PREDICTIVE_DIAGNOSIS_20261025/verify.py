"""Independent structural/hash/arithmetic audit. No fitting, learner import or writes."""
from pathlib import Path
import hashlib,json,math,subprocess,time,resource
H=Path(__file__).resolve().parent;R=H.parents[1];start=time.process_time()
load=lambda p:json.loads((H/p).read_text())
l=load('law/RESULTS.json');c=load('capacity/RESULTS.json');r=load('recoverability/RESULTS.json');v=load('v147/RESULTS.json');availability=load('ARTIFACT_AVAILABILITY.json')
assert l['ledger']['completed_fixture_target_updates']==12 and l['ledger']['failed_fixture_target_updates']==0
assert l['ledger']['external_record_applications']==l['ledger']['replayed_record_applications']==l['ledger']['held_inputs_used']==0
assert l['source_and_input_hashes_unchanged'] and l['original_model_and_trace_state_unchanged'] and l['forward_probes_no_persistent_mutation']
assert l['max_equation_absolute_error']<1e-10 and not l['held_inputs_scored'] and not l['world_generator_imported']
for p,digest in l['hashes'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==digest,p
for x in availability['local_checkpoints']:assert hashlib.sha256((R/x['path']).read_bytes()).hexdigest()==x['sha256']
assert c['external_teaching_record_applications']==c['native_or_candidate_fits']==c['historical_trajectories_replayed']==0
for model in c['prediction_results'].values():
 folds=model['folds'];assert len(folds)==8;errors=[]
 for fold in folds:
  assert fold['held_world'] not in fold['training_worlds'] and len(set(fold['training_worlds']))==7
  assert {q['N'] for q in fold['predictions']}=={5177,20708}
  ee=[q['predicted']-q['actual'] for q in fold['predictions']]
  assert all(abs(x-q['signed_error'])<1e-14 for x,q in zip(ee,fold['predictions']))
  assert abs(sum(abs(x) for x in ee)/2-fold['mean_absolute_error'])<1e-14
  assert abs(sum(x*x for x in ee)/2-fold['mean_squared_error'])<1e-14;errors+=ee
 assert abs(sum(abs(x) for x in errors)/16-model['MAE'])<1e-14
 assert abs(math.sqrt(sum(x*x for x in errors)/16)-model['RMSE'])<1e-14
assert c['prediction_results']['geometry_added']['RMSE']>c['prediction_results']['baseline']['RMSE']
for x in r['signed_phase_differences']:
 for key in ['nonempty_exact','forced_length_payload_exact','bit_accuracy','byte_accuracy']:
  assert abs(sum(z[key] for z in x['world_differences'])/4-x[key])<1e-12
std=next(x for x in r['signed_phase_differences'] if x['kind']=='NEW_write_effect' and x['STD'] and x['readout']=='DIRECT')
assert std['forced_length_payload_exact']==-.5390625
assert v['held_inputs_scored']==v['new_training_records']==v['new_target_events']==v['new_generated_outputs']==0
for fit in v['fits']:
 for z in fit['curves']:
  if z['complete_given_correct_first'] is not None:assert abs(z['complete']-z['first_payload_correct']*z['complete_given_correct_first'])<1e-12
for p,sha in v['source_sha256'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==sha
# Git sources stay in memory and must match both their object identity and bytes.
sources={x['path']:x for x in r['inputs']};sources.update(c['provenance']);read_bytes=0
for p,d in sources.items():
 b=subprocess.check_output(['git','show',f"{c['source_commit']}:{p}"],cwd=R)
 assert len(b)==d['bytes'] and hashlib.sha256(b).hexdigest()==d['sha256'],p
 oid=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();assert oid==d['git_blob_oid'],p;read_bytes+=len(b)
assert hashlib.sha256((H/'PREDICTIONS.json').read_bytes()).hexdigest()==load('PREDICTION_ADJUDICATION.json')['register_sha256']
# Reject nonfinite compact output values without executing any analysis again.
def finite(x):
 if isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,list):
  for v in x:finite(v)
for x in [l,c,r,v]:finite(x)
assert not subprocess.check_output(['git','diff','--name-only'],cwd=R).strip(),'Tracked historical files changed'
summary={'status':'PASS','source_git_blobs_checked':len(sources),'source_git_bytes_rechecked_in_memory':read_bytes,'local_calibration_checkpoints_hash_checked':8,'single_target_fixture_events':12,'fixture_failures':0,'held_scores_opened':False,'new_scientific_training_records':0,'new_trajectory_replays':0,'prediction_fold_arithmetic_recomputed_without_refitting':True,'all_worlds_kept_grouped':True,'signed_historical_contrasts_checked':True,'original_tracked_files_unchanged':True,'review_scope':'Source hashes, JSON arithmetic, fold grouping, finite metrics, reported ledger. Does not certify prospective theory validity.','CPU_seconds':time.process_time()-start}
(H/'AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
