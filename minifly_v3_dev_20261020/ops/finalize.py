"""Finish three-cycle artifacts from immutable engineering data; no training."""
from pathlib import Path
import sys,subprocess,os,json,hashlib,zipfile,shutil,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from artifacts import HERE,ROOT,atomic_json,sha,source_identity,source_digest

def run():
 audit=json.loads((HERE/'cycles/cycle3/AUDIT.json').read_text());assert audit['accepted']
 expected=json.loads((HERE/'cycles/cycle3/SOURCE_LOCK.json').read_text());assert expected['digest']==source_digest()
 snapshot=HERE/'cycles/cycle3/SOURCE_EXECUTED';snapshot.mkdir(exist_ok=True)
 for p in HERE.glob('*.py'):shutil.copy2(p,snapshot/p.name)
 for filename in ('SPEC.json','PARAMETER_LOCK.json'):shutil.copy2(HERE/filename,snapshot/filename)
 env=os.environ.copy()
 for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
 for script in ('report.py','ops/supplementary_audit.py','ops/resource_profile.py'):
  with open(HERE/'cycles/cycle3'/f"{Path(script).stem}.log",'w') as out:
   subprocess.run([sys.executable,'-u',str(HERE/script)],cwd=HERE,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
 summary=json.loads((HERE/'results/SUMMARY.json').read_text())
 supplemental=json.loads((HERE/'cycles/cycle3/SUPPLEMENTARY_AUDIT.json').read_text());assert supplemental['accepted']
 resource=json.loads((HERE/'cycles/cycle3/RESOURCE_PROFILE.json').read_text())
 receipts=[json.loads(p.read_text()) for p in sorted((HERE/'cycles/cycle3/jobs').glob('*/RECEIPT.json'))]
 production=sum(sum(tr['points'][-1]['production_cpu'] for tr in r['trajectories'].values())-2*r['initialization_cpu'] for r in receipts)
 evaluation=sum(r['evaluation_cpu'] for r in receipts)
 atomic_json(HERE/'results/RESOURCE_REPORT.json',dict(
  recorded_full_life_production_CPU_seconds=production,recorded_full_life_evaluation_CPU_seconds=evaluation,
  engineering_dispatcher_wall_seconds=audit['elapsed_wall_s'],actual_concurrency=2,numerical_threads_per_worker=1,
  production_CPU_scope='Record training including native prediction needed for teaching, local updates and native idle advance; one physical initialization per world/arm job. Standalone-history logical costs include one initialization each.',
  accounting_limit='Full worker process CPU and historical peak RSS were not metered. Diagnostic prefix probes, checkpoint/trace compression and operation overhead are outside the recorded production/evaluation categories; do not call their sum total physical compute.',
  resource_profile=resource,future_operations_requirement='Use an OS-timed worker wrapper and explicit live memory admission before a pilot. The current engineering run used a fixed two-worker fallback; four-worker admission is not qualified.'))
 ledger=dict(preliminary_native_smoke=2,cycle1_native_and_interface=18,cycle2_finite_calibration=64,
  cycle2_acquisition=3840,cycle2_resume_both_branches=6,cycle3_parent_control=3840,cycle3_full_engineering=199680)
 assert sum(ledger.values())==summary['total_record_applications_so_far']
 atomic_json(HERE/'TRAINING_LEDGER.json',dict(record_applications_by_stage=ledger,total_record_applications=sum(ledger.values()),
  partial_prompt_fixtures=18,partial_target_fixtures=2,scientific_records=0,
  readonly_reducers_and_supplemental_audits_training_records=0,
  definition='One full external record application per learner history; native output compartments, prediction bytes and derived time-only views are not independent lives.'))
 qualification=dict(status='THREE_CYCLES_COMPLETE_NO_ADVANCE',technical_cycles_completed=3,
  implementation_and_chronology_checks_passed=True,unseen_addition_efficacy_demonstrated=False,
  formal_campaign_ready=False,science_jobs_executed=0,pilot_started=False,confirmation_started=False,
  native_answer_path=True,gradual_warmup_preserved=True,arithmetic_specific_structure=False,
  attention=False,transformers=False,ELM=False,BPTT=False,automatic_adoption=False,
  engineering_full_jobs=8,continuous_histories=24,full_life_record_applications=199680,
  total_technical_record_applications=sum(ledger.values()),verdict=summary['verdict'],
  reason='No final held-out directional advantage; technical qualification and taught memory do not establish E3.',
  remaining_operations_limit='Exact total-worker CPU/peak-RSS and live resource admission must be instrumented for any future campaign; no campaign is recommended from these efficacy results.',
  sources=expected['digest'],supplementary_audit_sha256=sha(HERE/'cycles/cycle3/SUPPLEMENTARY_AUDIT.json'))
 atomic_json(HERE/'QUALIFICATION.json',qualification)
 atomic_json(HERE/'MODEL_MANIFEST.json',dict(version='MiniFly v3',upstream_commit='2c217b2716fc1cc0796f73a9fcc288f8aca864da',
  retained='Canonical Full151 native value equations, 16 C16 output compartments, FE-PERM, residual signed teaching, homeostasis and adaptation reset',
  added='64-unit gated sequence adaptor with local eligibility and fixed feedback; scheduled gradual native/representation/homeostatic updates',
  active_representation_parameters=19456,stored_parameter_entries_including_masked_diagonals=19584,
  frozen_lesions=['F-C','F-G'],locally_plastic_arms=['P-C','P-G'],
  no_separate_answer_head=True,no_arithmetic_parser_or_features=True,external_episode_cache=False,
  native_immutable_asset_audit=str((HERE/'cycles/cycle1/NATIVE_ASSET_AUDIT.json').relative_to(ROOT))))
 reports={
  'cycle1':'# Cycle 1 Design audit, tiny tests and revision\n\nThe source/native boundary was implemented and audited before comparisons. The original archived package was recovered after strict hashing identified sanitized metadata/helper differences in the local mirror. An initial readout mirror used m.Q instead of the frozen reader’s own Q; this was corrected, with the failed fixture preserved. Scalar/batched state and readout errors then stayed below 5e-14. Direct local derivatives, byte controls, read-only queries, disabled learning and EOS/truncation passed. Two preliminary plus 18 formal technical record applications were used. Exact executed source snapshots are hash-verified.\n',
  'cycle2':'# Cycle 2 Warmup, local updates and checkpoint audit\n\nThe first preregistered local-rate candidate qualified on arbitrary-byte taught fixtures without HELD feedback: eta_max=3e-5, eligibility clip=8, row cap=0.001 of the fixed birth-row anchor, projection bound=0.125. Four arms each received 960 OLD records, with six additional record applications across the two resume branches. Including calibration, this cycle used 3,910 record applications. P-G taught recall was 61.7% versus P-C 86.7% at this dose. Warmup was retained; that cost was not hidden or tuned away. Exact checkpoint/resume passed. A subsequent metric fixture corrected omission of birth-policy compute before the first timed checkpoint.\n',
  'cycle3':'# Cycle 3 Complete engineering lives, independent audit and revision\n\nTwo registered engineering worlds, four arms and three histories per job were executed through OLD/rest/NEW/rest. All eight immutable jobs passed the separate trace/receipt reducer: 24 continuous histories and 199,680 external record applications. Native-parent positive controls added 3,840 applications. The learner, roster, dose and source remained locked during these runs.\n\nNo final held-out gain was observed, so the result is NO_ADVANCE, not arithmetic-learning success. The additional active-receptor derivative fixture repairs the limited zero-input W/G derivative check in cycle 1; explicit eligibility recurrence checks also passed. Saved-checkpoint native-state erasure removed taught recall without changing production states. These were mathematical/read-only audits, not additional training or a fourth experimental cycle.\n\nResource review distinguishes exact production timings from unmetered full-process overhead and historical peak RSS. A conservative read-only profiling workload supplies a future reservation estimate; it does not fabricate retrospective measurements. Any future pilot must use an OS-timed wrapper and explicit live admission. No further campaign is authorized or recommended from the current efficacy result.\n'}
 for c,text in reports.items():(HERE/'cycles'/c/'REVIEW.md').write_text(text)
 # Amend the delivered design without deleting its original cycle-1 snapshot.
 design=ROOT/'minifly_v3_design_20261020/DESIGN.md';text=design.read_text()
 text=text.replace('20 October 2026. **Design delivered; not implemented, qualified or launched.** This is the v3 revision of Claude\'s 18 October v2 proposal. It authorizes no training, automatic continuation, adoption or GitHub upload. Historical experiments and their verdicts remain unchanged.',
  '20 October 2026. **Three engineering cycles completed; no demonstrated unseen-addition improvement; NO_ADVANCE.** The later user request authorized implementation and bounded technical tests of this v3 revision. No formal pilot or confirmation was launched. No automatic continuation, adoption or GitHub upload is authorized. Historical experiments and their verdicts remain unchanged. The originally delivered design is preserved in the first-cycle snapshot.')
 text=text.replace('2. Fixed forward-projection amplitude and sparse connectivity, including full initialization algorithms and seeds.',
  '2. Fixed forward-projection amplitude and connection masks, including full initialization algorithms and seeds. The bounded adaptor is dense except its masked recurrent diagonals; inherited native PN/KC sparsity is unchanged.')
 text=text.replace('Implementation and a three-cycle exercise require a subsequent request. Their intended sequence is:',
  'The user subsequently authorized implementation and this three-cycle exercise. The executed sequence was:')
 appendix='\n\n## 14 Three-cycle revision and measured outcome\n\nThe [engineering report](../minifly_v3_dev_20261020/REPORT.md) and [qualification record](../minifly_v3_dev_20261020/QUALIFICATION.json) are the executed status authority. The original design and exact executed source versions remain archived under cycles/.\n\nThe selected engineering parameters are eta_max=0.00003, eligibility clipping at 8, per-row update cap 0.001 times max(1, birth row norm), projection bound 0.125, uniform fan-scaled input/recurrent initialization and gate bias −1. Local updates remain delayed until EOS, with no BPTT or independent answer head. Gate logits are clipped at ±40 for numerical stability. The steady OLD multipliers are 0.94146728515625 for native/homeostatic updates and 0.9236837114606627 for representation learning. These are tested settings, not established optima.\n\nAll eight complete engineering jobs and their 24 histories were audited. The full-life budget was 199,680 record applications; the total including parent, calibration, preliminary and resume fixtures was 207,450. All four W arms ended at 0% HELD in both engineering worlds. Strong taught acquisition and delayed recall therefore remain E1 results; the desired E3 gain and an efficiency advantage were not demonstrated. Gradual warmup, exact output scoring, corpus boundaries and the no-arithmetic-structure rule were preserved rather than adjusted to manufacture a winner.\n\nThe completed tests do not justify the conditional pilot in Section 10. A future mechanism revision would need a new explicit design and a concrete hypothesis about representation/credit quality; simply extending training, increasing n, supplying arithmetic state or relaxing exact accuracy is not a permitted repair. Any subsequent run must also close the OS-level resource-telemetry/admission gap documented in the resource report. No such run or new mechanism was executed here.\n'
 assert '## 14 Three-cycle revision' not in text
 design.write_text(text+appendix)
 # Explicit limitations and targeted revisions accompany the generated summary.
 with open(HERE/'REPORT.md','a') as out:
  out.write('\n## Final audit additions and limits\n\nActive-receptor W/G derivative checks and explicit local trace recurrence passed in the supplementary audit. Erasing native fast/slow state on read-only final P-G checkpoints reduced taught recall to zero in both worlds, confirming that the native stores supplied the scored memory. These are E1 attribution checks, not E3 success.\n\nThe resource report preserves exact recorded production costs, measured dispatcher wall time and a conservative read-only memory profile. It does not invent missing full-worker process CPU or retrospective peak RSS. Future operations need explicit telemetry/admission qualification; no pilot is currently justified.\n\nThe design was revised with the measured parameter lock and the negative outcome. No arithmetic-specific component, extra sleep replay, longer budget, changed held-out set or approximate-answer gate was introduced.\n')
 atomic_json(HERE/'SOURCE_LOCK.json',dict(operative_files=source_identity(),operative_digest=source_digest(),
  cycle3_executed_source_digest=expected['digest'],operative_source_unchanged_during_tests=True,
  operations_files={str(p.relative_to(ROOT)):sha(p) for p in (HERE/'ops').glob('*.py')},
  parameter_lock_sha256=sha(HERE/'PARAMETER_LOCK.json'),design_sha256=sha(design)))
 p=ROOT/'PROJECT_DOCUMENTS/CURRENT_ARITHMETIC_EXPERIMENT_DESIGN.json';d=json.loads(p.read_text());n=d['next_design']
 n.update(status='THREE_CYCLES_COMPLETE_NO_ADVANCE',implemented=True,technical_cycles_executed=3,
  training_record_applications_executed=207450,science_jobs_executed=0,technical_qualification_passed=True,
  formal_campaign_ready=False,launch_authorized=False,pilot_started=False,new_core_adopted=False,
  qualification=str(HERE/'QUALIFICATION.json'),report=str(HERE/'REPORT.md'),
  source_lock=str(HERE/'SOURCE_LOCK.json'),parameter_lock=str(HERE/'PARAMETER_LOCK.json'),
  technical_scope='Two engineering worlds; eight full jobs; 24 correlated histories; no pilot or confirmation',
  verdict=summary['verdict'])
 p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
 atomic_json(HERE/'FINAL_AUDIT.json',dict(accepted=True,scope='Engineering accounting, source identity, primitive parity, local-rule/interface fixtures, eight independent receipt reductions and read-only attribution; not an E3 efficacy pass',
  scientific_records=0,technical_record_applications=207450,full_engineering_jobs=8,
  source_digest=source_digest(),qualification_sha256=sha(HERE/'QUALIFICATION.json'),
  summary_sha256=sha(HERE/'results/SUMMARY.json'),resource_limitations_documented=True))
 atomic_json(HERE/'STATUS.json',dict(status='THREE_CYCLES_COMPLETE_NO_ADVANCE',technical_only=True,
  accepted_engineering_jobs=8,continuous_histories=24,technical_record_applications=207450,
  scientific_records=0,active_workers=[],pilot_started=False,automatic_adoption=False))
 # Bundle the new implementation plus its small exact parent-source dependency.
 files=[p for p in HERE.rglob('*') if p.is_file() and not any(k in p.relative_to(HERE).parts for k in ['scratch','__pycache__']) and p.name not in ('RESULT_BUNDLE.zip','PACKAGE.json','BUNDLE_MANIFEST.json') and not p.name.endswith('.pending')]
 extras=[design,ROOT/'minifly_a_v3/SOURCE_LOCK.json']+list((ROOT/'minifly_a_v3/src').glob('*.py'))
 files+=extras
 manifest={str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(set(files))}
 atomic_json(HERE/'BUNDLE_MANIFEST.json',dict(schema='MINIFLY_V3_TECHNICAL_BUNDLE',files=manifest,
  roles={'upstream':'Historical immutable source/assets; not new scientific observations','cycles':'Engineering tests and correlated histories','results':'Derived engineering metrics'},scientific_records=0))
 bundle=HERE/'RESULT_BUNDLE.zip'
 with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(set(files+[HERE/'BUNDLE_MANIFEST.json'])):z.write(p,str(p.relative_to(ROOT)))
 with zipfile.ZipFile(bundle) as z:assert z.testzip() is None;entries=len(z.infolist())
 atomic_json(HERE/'PACKAGE.json',dict(completed=True,path=str(bundle),sha256=sha(bundle),bytes=bundle.stat().st_size,
  entries=entries,manifest_sha256=sha(HERE/'BUNDLE_MANIFEST.json'),scientific_records=0,technical_record_applications=207450))
 print(json.dumps(dict(verdict=summary['verdict'],record_applications=207450,bundle_bytes=bundle.stat().st_size,report=str(HERE/'REPORT.md'))),flush=True)
if __name__=='__main__':run()
