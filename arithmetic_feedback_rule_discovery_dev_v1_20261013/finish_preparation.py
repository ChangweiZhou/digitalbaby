"""Build descriptive preparation artifacts from saved receipts. No learner execution."""
import csv,gzip,hashlib,json,platform,sys,zipfile
from pathlib import Path
import numpy as np,scipy,threadpoolctl
from metrics import J,G

ROOT=Path(__file__).parent

def encode_numpy(x):
 if isinstance(x,np.generic):return x.item()
 raise TypeError(f'unsupported artifact value {type(x)}')

def write(path,value):path.write_text(json.dumps(value,indent=2,default=encode_numpy)+'\n')

def main():
 audit=json.loads((ROOT/'FINAL_AUDIT.json').read_text());assert audit['accepted']
 certs={};receipts={}
 for p in (ROOT/'cycles/cycle3/final_receipts').glob('*.json.gz'):
  with gzip.open(p,'rt') as f:r=json.load(f)
  a=json.loads(p.with_suffix('.audit.json').read_text());assert a['accepted']
  receipts[r['config']]=r;certs[r['config']]=a
 roster=('R1-DIAG','R1-FULL','R4-DIAG','R4-FULL','F1-FULL','F4-FULL','LEGACY-B')
 rows=[];diagnostics={}
 for name in roster:
  a=certs[name];s=a['summary'];w=s['W'];final=w[-1];r=receipts[name]
  features=np.array([x['feature'] for x in r['histories']['W']['checkpoints'][-1]['held_pre']])
  gram=features@features.T;distance=np.sqrt(np.maximum(0.,2-2*gram));np.fill_diagonal(distance,np.inf)
  gref=G(w,certs['LEGACY-B']['summary']['W'])
  grad=[x['encoder_update']['unclipped_gradient_norm'] for x in r['histories']['W']['root_records'] if x.get('encoder_update')]
  diagnostics[name]={'mean_nearest_held_feature_distance_pre_cohort':float(distance.min(axis=1).mean()),
      'median_unclipped_encoder_gradient':float(np.median(grad)) if grad else None,
      'learning_state_storage':r['histories']['W']['storage'],
      'first_step_diagnostics_only':True,'feature_metrics_are_not_causal_explanations':True}
  rows.append({'config':name,'terminal_taught_before_cohort':final['taught'],
       'terminal_taught_retained':final['taught_retained'],
       'terminal_held_pre':sum(final['held_pre'].values())/4,
       'terminal_held_retained':sum(final['held_retained'].values())/4,
       'terminal_held_retained_by_operation':final['held_retained'],
       'terminal_causal_W_minus_N_OLD':{op:final['held_retained'][op]-s['N_OLD'][-1]['held_retained'][op] for op in '+-*/'},
       'J_W':J(s),'J_N_OLD':J(s,'N_OLD'),'J_SHUFFLE_OLD':J(s,'SHUFFLE_OLD'),
       'paired_G_vs_legacy':{'candidate':gref[0],'legacy':gref[1]},
       'terminal_cohort_post':final['cohort_post'],'W_terminal_scenario_training_cpu':final['cpu'],
       'job_wall_seconds':r['wall_seconds'],'job_resource':a['resource']})
 # The predeclared DEV signs are displayed for the reserved fixture only, not
 # used to select a candidate, estimate p-values or replace fresh DEV worlds.
 signs={}
 for name in ('R1-FULL','R4-FULL'):
  s=certs[name]['summary'];w=s['W'];n=s['N_OLD'];sh=s['SHUFFLE_OLD'];fixed='F'+name[1:] 
  gw,gf=G(w,certs[fixed]['summary']['W']);gl,gb=G(w,certs['LEGACY-B']['summary']['W'])
  signs[name]={'taught_W_above_N_at_every_budget':all(a['taught']>b['taught'] for a,b in zip(w,n)),
    'terminal_each_operation_above_N_and_shuffle':all(w[-1]['held_retained'][op]>n[-1]['held_retained'][op] and w[-1]['held_retained'][op]>sh[-1]['held_retained'][op] for op in '+-*/'),
    'J_above_frozen_and_legacy':J(s)>J(certs[fixed]['summary']) and J(s)>J(certs['LEGACY-B']['summary']),
    'G_above_frozen_and_legacy':gw>gf and gl>gb,
    'cohort_acquired':all(x['cohort_post']>x['cohort_pre'] for x in w),
    'not_a_scientific_selection':True}
 assert not all(v for k,v in signs['R1-FULL'].items() if k!='not_a_scientific_selection')
 assert not all(v for k,v in signs['R4-FULL'].items() if k!='not_a_scientific_selection')
 summary={'scope':'reserved technical fixtures only; no formal population inference',
  'goal':'Learn arithmetic from observed byte feedback and improve retained unseen-case generalization per sample and compute, without supplying arithmetic rules',
  'verdict':'TECHNICAL_PASS_BEHAVIORAL_NOT_READY','three_cycles_completed':True,
  'scientific_jobs':0,'full_technical_jobs':audit['all_full_technical_jobs'],
  'technical_record_applications':audit['all_full_technical_record_applications'],
  'separate_binary_fixture_record_applications':128,'final_technical_world':202610130103,
  'final_source_comparisons':rows,'reserved_fixture_screen_signs':signs,
  'behavioral_diagnostics':diagnostics,'formal_DEV_and_confirmation_not_run':True,
  'family_wide_exclusions':0,'adopted_core_changed':False}
 write(ROOT/'SUMMARY.json',summary)
 with (ROOT/'BUDGET_CURVES.csv').open('w',newline='') as f:
  writer=csv.writer(f);writer.writerow(['config','history','n_per_operation','scenario_records','taught_pre','taught_retained','held_retained_macro','held_add','held_sub','held_mul','held_div','cohort_pre','cohort_post','training_cpu'])
  for name in roster:
   for history,points in certs[name]['summary'].items():
    for x in points:writer.writerow([name,history,x['n'],x['scenario_records'],x['taught'],x['taught_retained'],sum(x['held_retained'].values())/4,*[x['held_retained'][op] for op in '+-*/'],x['cohort_pre'],x['cohort_post'],x['cpu']])
 maximum=max(r['wall_seconds'] for r in receipts.values())
 runtime={'technical_single_thread_full_job_wall_seconds':{k:r['wall_seconds'] for k,r in receipts.items()},
  'conditional_maximum_provisional_hours_two_workers_with_3x_allowance':296*maximum/2*3/3600,
  'not_an_authorized_launch':True,'two_worker_parallel_parity_not_qualified':True,
  'probe_snapshot_audit_and_commit_cpu_are_separate_from_training_score':True,
  'future_background_dispatcher_qualification_deferred_due_to_behavioral_failure':True}
 write(ROOT/'RUNTIME_ESTIMATE.json',runtime)
 parameter_report={'encoder_parameters':19696,'readout_parameters':16705,'full_confidence_entries':4225,
  'learned_full_persistent_array_bytes':482576,'learned_diagonal_persistent_array_bytes':449296,
  'gradient_tape_max_events':32,'transient_gradient_buffers_extra':True,
  'global_dense_engineered_model_not_native_scalar_neurons':True,
  'GRU_forward_dense_multiply_add_flops_per_transition':'6*d*(m+d), plus O(d) gate arithmetic',
  'categorical_prediction_dense_flops_per_decision':'2*(d+1)*257, plus loss/probability arithmetic',
  'confidence_update_full_complexity':'O((d+1)^2 + (d+1)*257)',
  'encoder_update_complexity':'O(window*d*(m+d) + permitted_targets*(K-1)*d*(m+d) + encoder_parameters)',
  'logical_array_traffic':'Each dense transition reads input/recurrent matrices; each readout decision reads V. Cache/DRAM traffic not measured; never label logical reads as measured DRAM bandwidth.',
  'all_runtime_costs_charged_in_saved_resources':True}
 write(ROOT/'RESOURCE_MODEL.json',parameter_report)
 qualification={'schema':'RAW_RULE_PREPARATION_QUALIFICATION_V1R1','completed_cycles':[1,2,3],
  'technical_correctness_passed':True,'final_unit_tests_passed':20,
  'independent_receipt_source_tamper_rejections':13,
  'final_source_full_jobs':7,'all_full_technical_jobs':16,
  'all_full_technical_record_applications':33792,
  'checkpoint_files_verified':audit['final_checkpoints_independently_verified'],
  'ready_for_formal_science':False,'science_launch_authorized':False,'scientific_jobs_executed':0,
  'reason':'Neither learned FULL recipe earns retained four-operation/sample/compute evidence in reserved actual-budget tests; incomplete future background/selection/confirmation qualification is explicitly deferred.',
  'namespace_census_limit':'No foreign match in local uncompressed metadata/source census; unavailable remote data/compressed receipt-body namespaces not exhaustively certified. Science admission blocked.',
  'no_arithmetic_priors_in_new_cores':True,'legacy_numerical_prior_disclosed':True,
  'dependencies':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'threadpoolctl':threadpoolctl.__version__,'platform':platform.platform()},
  'cycle_audits':['cycles/cycle1/AUDIT.json','cycles/cycle2/AUDIT.json','cycles/cycle3/AUDIT.json'],
  'source_identity':audit['final_source_identity'],'adopted_core_unchanged':True}
 write(ROOT/'QUALIFICATION.json',qualification)
 write(ROOT/'cycles/cycle3/AUDIT.json',{'cycle':3,'technical_correctness_passed':True,
  'final_unit_tests_passed':20,'initial_jobs':7,'final_source_jobs':7,'record_applications':29568,
  'initial_fixture_failure':'EOS column fixture incorrectly assumed positive sum of signed features; corrected to bias-only forcing',
  'operations_revision':'Measured separated CPU and actual per-record component counters; fresh reserved fixtures, no learning change',
  'full_checkpoint_audit_count':98,'behavioral_readiness':False,'formal_science_jobs':0})
 source={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))}
 source.update(audit['final_source_identity'])
 write(ROOT/'SOURCE_LOCK.json',{'purpose':'final technical preparation sources, not authorization to start science',
       'files':source,'receipt_execution_sources':audit['final_source_identity']})
 table=['| Configuration | Taught, before cohort | Taught, retained | Held, retained (four-op macro) | J over six budgets |',
        '| --- | ---: | ---: | ---: | ---: |']
 for x in rows:table.append(f"| {x['config']} | {x['terminal_taught_before_cohort']:.1%} | {x['terminal_taught_retained']:.1%} | {x['terminal_held_retained']:.2%} | {x['J_W']:.2%} |")
 report='''# RAW-RULE V1 — three-cycle preparation result

**Three cycles completed. Mathematical/interface/write-control qualification passes; the tested arithmetic recipe is not ready for formal science.** The goal remains retained unseen four-operation ability from fewer samples and compute, learned from observed byte feedback without supplied arithmetic rules. No DEV or confirmation was run, and the adopted core is unchanged.

## What was built and audited

A generic byte embedding → 64-wide GRU → one 257-category plastic readout. Learned/frozen encoders, one/four tied processing steps, and full/diagonal confidence form the declared seven-arm roster with the unchanged legacy reference. New cores have no integer parser, digit values, operand slots, operation IDs, numeric losses, arithmetic answer functions, inverse relations, numeric formatter, vocabulary mask or pretrained helper. The legacy reference retains its disclosed numerical features solely as a historical comparator. This is engineered backpropagation, not unchanged Full151 or a fly-local rule.

Cycle 1 independently finite-differenced the encoder at both depths using each target's pre-update readout, checked binary acquisition and legal byte interface, and repaired identity-dependent audit hashing. Cycle 2 ran two complete actual-budget jobs, checked root/fork/write branches and exposed weak arithmetic behavior; revisions strengthened guards, source checking and cap parity without changing learning parameters. Cycle 3 ran the full seven-arm roster, revised an incorrect EOS test fixture and missing CPU categories, then qualified the final runner on seven fresh technical jobs. Initial sources, failures and trajectories remain preserved; none were reused or overwritten.

Twenty final unit tests pass. Thirteen independent receipt/source tamper cases are rejected, with actual array write-protection also tested. The final auditor checks all 16 technical receipts, all component counter continuities in the final seven jobs, family holdout, resource accounting, immutable parent forks and 98 saved final-source checkpoints. Long-stream finite differences independently verify truncation: older prefix state is detached, evicted loss derivatives are discarded, and the mean divisor includes all permitted targets. Generated EOS cannot flush training; an actual no-write EOS cannot step the optimizer.

## Actual-budget behavior

The table is **one reserved final technical world**, not a population result or a scientific selection. Each configuration used identical external records and held families; 128 examples per operation in the terminal root, followed by the 32-record binary cohort. Held accuracy is an equal average over operations; J averages retained scores over all six budgets.

'''+ '\n'.join(table)+'''

The two earlier reserved arithmetic fixtures were similarly weak. For example, depth-1 FULL first recalled 13/16 taught cases, then only 31/200 at its terminal prefix; both selectable FULL recipes finished that fixture at 0/104 retained held cases. In the final world, depth-1 FULL has zero terminal retained held answers; depth-4 FULL has a small subtraction-only terminal score, with addition, multiplication and division still zero. Neither beats the legacy sample score or satisfies the declared four-operation signs in these fixtures.

This is not evidence that the byte bridge is broken. Both depths acquired the two-key binary fixture (2/2 after 64 records); the later binary cohort is also learnable, and the unchanged legacy core retains 100% of terminal taught cases in the final world. Yet the generic recurrent recipe recalls only about 20% of terminal taught cases before the cohort and barely generalizes. Its learned encoders give little acquisition advantage over their frozen controls. Extra tied processing and full confidence do not establish the required retained arithmetic gain at the tested budgets.

The failure already exists before intervening learning: depth-1 has only a small subtraction score on held cases and depth-4 has none at the terminal pre-cohort probe. Thus persistence alone cannot explain the result. Features, gradients, probabilities and saved states allow inspection; feature geometry and confidence drift remain hypotheses rather than established causal mechanisms. Fixing output termination or adding arithmetic relationships would not be a justified success claim.

## Cost and scope

There are 16 accepted complete technical jobs: 48 root histories and 288 inherited cohort forks (336 correlated trajectories), 33,792 external record applications. The separate binary fixture used 128 records. Unit/synthetic fixtures are recorded separately; there are **zero scientific jobs**. Final-source qualification covers seven jobs/14,784 records. Correlated trajectories are not independent worlds.

Teaching CPU is recorded separately from probes, snapshots, independent auditing, orchestration and commit costs. Dense recurrence, readout, confidence and optimizer work are disclosed in RESOURCE_MODEL.json; logical matrix traffic is not presented as measured hardware bandwidth. Learned FULL persistent arrays occupy 482,576 bytes; bounded tapes and gradient buffers are additional. No free replay, arithmetic warmup or independent-logit combination was added.

## Readiness and revised next action

Keep the public goal and the delivered sample/compute and four-operation criteria. Record this recipe as **technically verified, behaviorally insufficient**; do not start the proposed 56-job DEV as if qualification had shown promising rule learning. No percentage gate was lowered, helper supplied or training budget increased to manufacture readiness. These reserved fixtures justify a preparation stop and reconsidering the feedback/representation learning recipe, not a statistical exclusion of generic recurrence or an entire mechanism family.

The current implementation admits reserved technical namespaces only. A future formal dispatcher, DEV selection and confirmation pipeline has not been qualified; the local namespace census also cannot certify unavailable remote/compressed-body history. These are explicit future admission requirements, not hidden claims of launch readiness. No mechanism inventory family is scientifically crossed out from these technical results. The next design revision must address learned representation/readout coordination under the same raw-byte feedback contract, and earn actual-budget unseen-case evidence before another formal launch.

Artifacts: QUALIFICATION.json, FINAL_AUDIT.json, SUMMARY.json, BUDGET_CURVES.csv, SOURCE_LOCK.json, RUNTIME_ESTIMATE.json, RESOURCE_MODEL.json and the three cycle directories. No GitHub upload or background science was performed.
'''
 (ROOT/'PREPARATION_REPORT.md').write_text(report)
 (ROOT/'REPORT.md').write_text(report)
 revision='''# RAW-RULE V1 preparation revision V1R1

Three authorized design–audit–test–revise cycles are complete. See the implementation folder's PREPARATION_REPORT.md and QUALIFICATION.json. The original EXPERIMENT.md is preserved in implementation/design/.

The learning recipe, seven configurations, sample budgets, family holdout, metrics and twelve primary comparisons remain unchanged. Clarifications: cap means sixteen payload bytes plus one final EOS decision, matching the historical reference; the gradient divisor includes all permitted frame targets even when old derivatives leave the 32-event tape; prompt and query arrays are independently guarded; both source identity and actual component counter continuity are audited. Depth-1 unused direct-gradient buffers are removed. Separate teaching, probe, snapshot, audit and commit CPU are retained.

The tested generic recurrent learners are behaviorally insufficient in reserved actual-budget tests. Formal science readiness is false; no automatic DEV/confirmation or background launch. This is a preparation outcome, not scientific family-wide exclusion. Do not add arithmetic priors, lower gates or inflate training to turn it into a pass. Future refinement must retain the same arithmetic-from-feedback goal.
'''
 (ROOT/'design/PREPARATION_REVISION_V1R1.md').write_text(revision)
 parent_design=ROOT.parent/'arithmetic_feedback_rule_discovery_design_v1_20261013'
 (parent_design/'PREPARATION_REVISION_V1R1.md').write_text(revision)
 metadata=json.loads((parent_design/'DESIGN.json').read_text())
 metadata.update(status='THREE_CYCLES_COMPLETE_TECHNICAL_PASS_BEHAVIORAL_NOT_READY',technical_cycles_executed=3,
  implementation='../'+ROOT.name,preparation_report='../'+ROOT.name+'/PREPARATION_REPORT.md',
  qualification='../'+ROOT.name+'/QUALIFICATION.json',ready_for_formal_science=False,
  scientific_jobs_executed=0,preparation_revision='PREPARATION_REVISION_V1R1.md')
 write(parent_design/'DESIGN.json',metadata)
 pointer_path=ROOT.parent/'PROJECT_DOCUMENTS/CURRENT_ARITHMETIC_EXPERIMENT_DESIGN.json'
 pointer=json.loads(pointer_path.read_text());history=pointer_path.parent/'ARITHMETIC_DESIGN_POINTER_HISTORY/pre_RAW_RULE_three_cycles_20261013.json'
 if not history.exists():write(history,pointer)
 pointer.update(status=metadata['status'],technical_cycles_executed=3,qualification_completed=True,
  technical_correctness_passed=True,ready_for_formal_science=False,implementation=ROOT.name,
  preparation_report=ROOT.name+'/PREPARATION_REPORT.md',qualification=ROOT.name+'/QUALIFICATION.json',
  preparation_revision='arithmetic_feedback_rule_discovery_design_v1_20261013/PREPARATION_REVISION_V1R1.md')
 write(pointer_path,pointer)
 for source_name in ('core.py','interface.py','runner.py','auditor.py','test_core.py','test_pipeline.py','metrics.py'):
  (ROOT/'cycles/cycle3'/('final_'+source_name)).write_bytes((ROOT/source_name).read_bytes())
 # Immutable preparation archive: includes initial failures and all saved jobs.
 files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ('PREPARATION_BUNDLE.zip','PACKAGE.json','PACKAGE_VERIFICATION.json','packaging.log')]
 manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 bundle=ROOT/'PREPARATION_BUNDLE.zip'
 with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files:z.write(p,p.relative_to(ROOT))
 write(ROOT/'PACKAGE.json',{'scope':'technical preparation only','complete':True,'bundle':bundle.name,
  'sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'bytes':bundle.stat().st_size,'files':manifest,
  'science_executed':False,'upload_authorized':False})
 print(json.dumps({'verdict':summary['verdict'],'technical_jobs':16,'technical_records':33792,
  'science_jobs':0,'bundle_bytes':bundle.stat().st_size},indent=2))

if __name__=='__main__':main()
