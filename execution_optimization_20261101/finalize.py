"""Read-only evidence reconciliation and compact execution-amendment package."""
import json,hashlib,time,zipfile
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parent;OLD=ROOT/'minifly_v176_feature_deployment_dev_20261101'
def read(p):return json.loads(p.read_text())
def save(n,v):(H/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 prior=read(H/'RESOURCE_REFORECAST.json');bench=read(H/'BENCHMARK_V2.json');panel=read(H/'SHARED_PANEL_QUALIFICATION.json');manifest=read(OLD/'local/DATA_MANIFEST.json')
 r={arm:{v['prompt_length']:v for v in panel['results'] if v['arm']==arm} for arm in ('RAW','H')}
 def cost(arm,n):
  a,b=r[arm][35],r[arm][71];return a['own_TF_pair_CPU']+(n-35)*(b['own_TF_pair_CPU']-a['own_TF_pair_CPU'])/36
 panels=0.
 for entry in manifest.values():
  for arm,mult in [('RAW',1),('H',5)]:panels+=mult*sum(cost(arm,len(bytes.fromhex(v['prompt']))) for v in entry['rows']['train']*4+entry['rows']['held']*3)
 forecast=1.5*(prior['native_training_CPU']+panels+prior['source_training_CPU']+prior['probe_fit_CPU']+prior['prior_technical_CPU_allowance']+prior['supplemental_engineering_CPU_reserve'])
 prior.update(selected_panel_backend='pair_shared',native_panels_CPU=panels,whole_plan_forecast_CPU=forecast,whole_plan_forecast_minutes=forecast/60,conservative_no_shared_panel_forecast_CPU=2505.9439868750023,forecast_reduction=1-forecast/prior['previous_forecast_CPU'])
 save('RESOURCE_REFORECAST.json',prior)
 files=[H/'PROFILE.json',H/'history/cpu_v1/QUALIFICATION.json',H/'history/attempt001/FAILURE.json',H/'history/attempt002/FAILURE.json',H/'history/attempt003/FAILURE.json',H/'QUALIFICATION.json',H/'BENCHMARK.json',H/'BENCHMARK_V2.json',H/'GPU_FEASIBILITY.json',H/'OWNERSHIP_QUALIFICATION.json',H/'SHARED_PANEL_QUALIFICATION.json']
 measured=sum(read(p).get('CPU',0.) for p in files)
 records=len((H/'LEDGER.jsonl').read_text().splitlines())+3
 save('COUNTS.json',dict(engineering_teacher_records_upper=records,engineering_target_applications_upper=2*records,own_TF_pairs=161,own_TF_reconciliation='profile3 + initial qualification32 + initial benchmark32 + packed-state failed qualification32 + fixture failed qualification16 + saved-state continuation16 + optimized benchmark12 + shared-panel qualification18',stage_CPU_measured=measured,all_engineering_CPU_reserved=300,imports_setup_and_unmeasured_checks_included_in_reserve=True,peak_RSS_bytes=max(read(p).get('peak_RSS_bytes',0) for p in files),science_source_records=0,science_native_records=0,scope='Supplemental engineering requested after V176 stop; all previous costs retained.'))
 lock=read(OLD/'IMPLEMENTATION_SOURCE_LOCK.json')['files'];mismatch=[p for p,d in lock.items() if p.startswith('minifly_a_v3/') and sha(ROOT/p)!=d];assert not mismatch
 sources={str(p.relative_to(ROOT)):sha(p) for p in H.glob('*.py')}
 save('SOURCE_LOCK.json',dict(files=sources,parent_native_files_verified=sum(p.startswith('minifly_a_v3/') for p in lock),parent_mismatches=mismatch,original_science_kernel_edited=False,scope='Final selected optimization code; qualification version history retained.'))
 save('INTEGRATION.json',dict(native_module='execution_optimization_20261101/optimized_native.py',native_class='Model',evaluation='pair_shared',selected_backend='CPU float64, compiled passive updates; no fastmath',source_feature_training='unchanged CPU local updates',source_probe='unchanged evaluator-only BP',GPU_native_selected=False,source_head_used_for_native_output=False,full_scientific_driver_qualified=False,main_fit_CLI_remains_blocked=True))
 save('EXECUTION_AMENDMENT.json',dict(authorization='User: Let us pull all levers to simplify, optimize the execution machinery etc before we run it.',scientific_roster_dose_gates_unchanged=True,supplemental_engineering_only=True,previous_stops_preserved=True,original_scientific_own_TF_pairs=10752,supplemental_engineering_own_TF_ceiling=192,prospective_all_query_pair_ceiling_including_supplement=11008,original_global_record_cap=28000,original_global_target_cap=1200000,original_total_CPU_cap=7200,engineering_CPU_inherited=True,changes=['exact passive-event specialization','fused compartment state layout','fixed-asset-sharing clones','bounded encoding cache','exact cutoff TopK with original ties','common-prefix evaluation','packed-view restore binding'],no_main_run_started=True))
 save('STATUS.json',dict(status='OPTIMIZED_NATIVE_BACKEND_QUALIFIED_MAIN_NOT_STARTED',native_equivalence='PASS_BITWISE',common_prefix_panel_equivalence='PASS_BITWISE',resource_forecast='FITS_CPU_BUDGET',full_driver_qualified=False,scientific_training_records=0,automatic_core_adoption=False))
 save('FINAL_AUDIT.json',dict(status='ACCEPTED_ENGINEERING_PACKAGE',independent_audit=False,evidence_level='E0 engineering only',selected_native_state_and_output_tests='PASS_BITWISE',whole_scientific_run_not_performed=True,arithmetic_efficacy_not_tested=True,failures_preserved=['history/attempt001/','history/attempt002/','history/attempt003/'],limitations=['Finite engineering equivalence coverage, not proof over every possible state.','Full source/probe/native main driver still needs prospective qualification.','Timing samples are short; forecast includes1.5x and300-second engineering reserve.','Initial supplemental engineering own/TF cap128 was exceeded by manual miscount; reconciled161 and disclosed prospective192 ceiling. No scientific record was generated.','First profiling pass appended three engineering rows to ignored old telemetry; those are counted here, not science; later telemetry isolated.']))
 print(json.dumps(prior,indent=2))
if __name__=='__main__':main()
