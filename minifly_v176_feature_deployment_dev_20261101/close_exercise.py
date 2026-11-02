"""Close the bounded technical exercise; no model invocation or scientific validation."""
import hashlib,json,platform,time
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parent
def read(n):return json.loads((H/n).read_text())
def save(n,v):(H/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 data=read('DATA_AUDIT.json');nq=read('NATIVE_QUALIFICATION.json');fq=read('FEATURE_QUALIFICATION.json');b=read('ROSTER_BENCHMARK.json');ex=read('EXPORT.json')
 manifest=read('local/DATA_MANIFEST.json');mid=b['benchmarks'][1];high=b['benchmarks'][2]
 def interpolate(field,length):return mid[field]+(length-mid['prompt_length'])*(high[field]-mid[field])/(high['prompt_length']-mid['prompt_length'])
 native_train=native_eval=0.
 for entry in manifest.values():
  train=entry['rows']['train'];held=entry['rows']['held']
  native_train+=6*8*sum(interpolate('train_record_CPU',len(bytes.fromhex(r['prompt']))) for r in train)
  native_eval+=6*sum(interpolate('own_TF_pair_CPU',len(bytes.fromhex(r['prompt']))) for r in train*4+held*3)
 native=native_train+native_eval
 source_batches=sum((v['targets']+255)//256 for k,v in ex['splits'].items() if k.endswith('train'))*9
 # Gradient-only timing multiplied by four explicitly reserves optimizer/copy work.
 source=source_batches*fq['source_batch_CPU']*4
 probe=12*3*8*fq['probe_batch_CPU']
 prior_upper=30.;actual_accounted=prior_upper+nq['CPU']+fq['CPU']+b['CPU']+ex['CPU']
 forecast=1.5*(native+source+probe+actual_accounted)
 admission=dict(status='RESOURCE_NOT_ADMITTED',CPU_cap=7200,multiplier=1.5,native_training_CPU=native_train,native_panels_CPU=native_eval,native_only_forecast_with_margin=1.5*native,source_training_estimate_CPU=source,probe_fit_estimate_CPU=probe,accounted_technical_CPU_upper=actual_accounted,whole_plan_forecast_CPU=forecast,estimated_CPU_hours=forecast/3600,important='Native-only forecast already exceeds cap; verdict does not depend on approximate source estimate.',method='Interpolate between label-independent engineering prompt lengths35 and71; actual selected roster lengths and all12 lives/panels; single numerical thread. Extrapolation below35 is disclosed.',benchmark_files=['ROSTER_BENCHMARK.json','NATIVE_QUALIFICATION.json','FEATURE_QUALIFICATION.json','EXPORT.json'])
 assert admission['native_only_forecast_with_margin']>7200
 save('RESOURCE_ADMISSION.json',admission)
 prior=json.loads((ROOT/'minifly_v171_dev_20261026/CALIBRATION_SOURCE_LOCK.json').read_text());mismatches=[p for p,d in prior.items() if not (ROOT/p).exists() or sha(ROOT/p)!=d];assert not mismatches
 lock={str(p.relative_to(ROOT)):sha(p) for p in H.glob('*.py')}
 lock.update({str((ROOT/p).relative_to(ROOT)):d for p,d in prior.items()})
 for p in (ROOT/'minifly_cls_dev_20261017').glob('*.py'):lock[str(p.relative_to(ROOT))]=sha(p)
 for p in (H/'assets/source').rglob('*.py'):lock[str(p.relative_to(ROOT))]=sha(p)
 save('IMPLEMENTATION_SOURCE_LOCK.json',dict(files=lock,historical_closure_verified=len(prior),mismatches=[],timing='post-technical, before any main fitting; not a prospective technical source lock'))
 counts=dict(scientific_source_records=0,scientific_native_records=0,source_fits_completed=0,native_lives_completed=0,probe_fits_completed=0,native_technical_training_records=10,technical_sensory_record_passes=16,synthetic_feature_record_touch_upper=4,total_technical_record_touch_upper=30,technical_training_target_applications=788,own_TF_pairs=16,native_vector_reads=92,raw_data_published=False)
 save('COUNTS.json',dict(counts=counts,accounted_CPU_upper=actual_accounted,peak_RSS_bytes=max(nq['peak_RSS_bytes'],fq['peak_RSS_bytes'],ex['peak_RSS_bytes'],b['peak_RSS_bytes']),non_scientific=True))
 save('STATUS.json',dict(identity='V176-NATIVE-FEATURE-BRIDGE-V1',status='STOPPED_C1_RESOURCE_NOT_ADMITTED',three_scientific_cycles_complete=False,design_audit_technical_run_revision_complete=True,main_fitting_authorized_but_not_admitted=True,core_adopted=False,counts=counts,next_stage='Prospectively qualify an exact acceleration or explicitly amend compute budget; no automatic continuation.'))
 save('FINAL_AUDIT.json',dict(status='ACCEPTED_TECHNICAL_STOP_RECEIPT',scientific_final_audit=False,independent_audit=False,verdict='RESOURCE_NOT_ADMITTED',data_integrity='PASS',native_and_local_update_technical_checks='PASS_WITH_RECORDED_ASSERTION_CORRECTION',full_C1_qualification=False,C2A_run=False,C2B_run=False,counts=counts,scope='E0 only; no feature formation, cross-text transfer, native deployment or arithmetic efficacy result.',limitations=['First failed technical process did not save its partial state; failure preserved and charged; no scientific state existed.','Existing writerlib CPU ledger initially inherited historical offset; do not use those timestamps as V176 CPU. Stage-local measured CPU and conservative prior upper are authoritative.','Some engineering and full-driver checks remain unqualified because the resource gate stopped admission.','Full fit/probe routines are draft and CLI-disabled; bridged feature export and native main driver are not qualified.']))
 print(json.dumps(admission,indent=2))
if __name__=='__main__':main()
