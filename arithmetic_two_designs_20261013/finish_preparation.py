"""Package existing technical evidence only; never starts a learning trajectory."""
import json,gzip,hashlib,zipfile,statistics,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent.parent
names=['arithmetic_feedback_rule_discovery_dev_v2_20261013','arithmetic_stream_transfer_dev_v1_20261013']
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
for i,name in enumerate(names):
 root=R/name;certs=[];resources=[]
 assert json.loads((root/'cycles/cycle3/RECEIPT_AUDIT.json').read_text())['accepted']
 if i==0:assert json.loads((root/'cycles/cycle3/FINAL_WINDOW_GRADIENT.json').read_text())['accepted']
 else:assert json.loads((root/'cycles/cycle2/INDEPENDENT_CHECKS.json').read_text())['accepted']
 for cycle in ('cycle2','cycle3'):
  exit_file=root/'cycles'/cycle/'EXIT_STATUS.json'
  status=json.loads(exit_file.read_text());assert status and all(v==0 for v in status.values())
  for p in sorted((root/'cycles'/cycle/'receipts').glob('*.audit.json')):
   c=json.loads(p.read_text());assert c['accepted'] and c['technical'];certs.append(c);resources.append(c['resource'])
 assert len(certs)==(10 if i==0 else 5)
 final=[c for c in certs if (c.get('world')==202610139803 if i==0 else c['label'].endswith('CYCLE_3'))]
 rows=[]
 for c in final:
  if i==0:
   w=c['summary']['W'][-1];rows.append({'config':c['config'],'taught':w['taught'],'taught_retained':w['taught_retained'],'held_retained_by_op':w['held_retained']})
  else:
   s=c['summary'];a=s['addition']['W'][-1];m=s['multiplication']['W'][-1]
   rows.append({'config':c['config'],'addition_taught':a['taught'],'addition_held_pre':a['held_pre'],'addition_held_retained':a['held_retained'],
    'multiplication_taught':m['taught'],'multiplication_held_retained':m['held_retained'],'addition_after_multiplication':m['addition_retained'],
    'scratch_multiplication_held_retained':s['multiplication']['N_ADD'][-1]['held_retained'],
    'no_write_multiplication_held_retained':s['multiplication']['N_MUL'][-1]['held_retained'],
    'shuffled_multiplication_held_retained':s['multiplication']['SH_MUL'][-1]['held_retained']})
 n=sum(c['record_applications'] for c in certs)
 summary={'status':'THREE_CYCLES_COMPLETE','technical_qualification':'PASS','behavioral_adoption':'NOT_ESTABLISHED','ready_for_formal_science':False,
  'scientific_jobs':0,'technical_full_jobs':len(certs),'technical_full_job_external_record_applications':n,
  'partial_cycle1_fixture_records':256 if i==0 else 104,'unit_fixtures_counted_as_science':False,'final_cycle_rows':rows,
  'warning':'One final technical world per recipe, not statistical evidence; no scientific selection or adoption.'}
 write(root/'SUMMARY.json',summary)
 maxcpu=max(r['total_job_cpu']+r.get('commit_cpu',0) for r in resources);maxwall=max(r['wall_seconds']+r.get('commit_cpu',0) for r in resources)
 projectedjobs=352 if i==0 else 56
 runtime={'measured_full_jobs':len(certs),'maximum_cpu_seconds_per_job_including_commit':maxcpu,'maximum_wall_seconds_plus_commit_cpu':maxwall,
  'formal_conditional_max_jobs':projectedjobs,'planning_workers':2,'conservative_3x_measured_max_seconds':3*max(maxcpu,maxwall)*projectedjobs/2,
  'actual_technical_total_cpu_seconds':sum(r['total_job_cpu']+r.get('commit_cpu',0) for r in resources),
  'caveat':'Technical timings, not a guarantee; conditional science roster remains unlaunched. Repeated observations are charged as external samples; correlated branches are not independent lives.'}
 write(root/'RUNTIME_ESTIMATE.json',runtime)
 cycletexts=([
 ('Joint byte learning and ordered generic readers.','Finite-difference checks cover GRU/LAG32 with CE/SQ; binary payload fixtures exercise NUL and 255.','12 numerical/interface tests and 256 binary observations.','Specify A final-window encoder credit separately from B per-target credit; keep every head target and the full mean denominator.'),
 ('Two full-budget joint-CE jobs on technical world 202610139802.','Independent gate, split, snapshot, output and resource audit.','G-CE and L-CE: 2 jobs / 4224 external applications. Sixteen revised unit/pipeline tests pass.','Close counter-ledger loophole: anchor each root at zero and each cohort fork at the parent prefix. Preserve initial sources and receipts; no scientific kernel or threshold tuning.'),
 ('All eight prespecified recipes on technical world 202610139803.','Independent audit of all 10 saved full jobs against archived sources, exact RAW reference identity, long final-window finite differences.','8 final jobs / 16896 applications; the legacy anchor is disclosed as engineered and nonselectable.','Short-budget generic CE acquisition remains weak. Do not promote a core or launch a large run from this fixture; gates unchanged.')
 ] if i==0 else [
 ('Same generic recurrent core for counting, addition and multiplication; four objective/decay recipes.','All-byte loss, bounded per-target gradients, actual EOS update, pure generation and no-write including decay.','12 numerical/interface tests; 4 stream fixtures / 104 external applications including disabled records.','Implement bounded online per-target credit for long counting strings; task changes never reset parameters or optimizer.'),
 ('One complete 256-pass ALL-0 technical life tree.','Independent full receipt/resource audit plus 8 tamper rejections, arithmetic labels, shuffle lengths, scratch-state parity and import boundary checks.','1 full job / 144768 applications; 12 saved independent checks pass.','Separate taught acquisition from unseen transfer and retention. Charge repeated external presentations and prior addition compute; do not call shared trajectories independent lives.'),
 ('All four fixed recipes, fresh technical label CYCLE_3, two concurrent technical workers.','Full branch, source, permissions, optimizer, retained-query and resource reconciliation; source code unchanged from cycle 2.','4 full jobs / 579072 applications.','Do not infer a general arithmetic law, cross-task benefit or adoption from one technical world. Keep the joint transfer/retention gates; bounded +/* feasibility remains distinct from four-operation ambition.')
 ])
 for number,(design,audit,test,revise) in enumerate(cycletexts,1):
  p=root/'cycles'/f'cycle{number}'/'CYCLE_REPORT.md'
  p.write_text(f'# Cycle {number}\n\n## Design\n\n{design}\n\n## Audit\n\n{audit}\n\n## Test run\n\n{test}\n\n## Revise\n\n{revise}\n\nAll execution in this cycle is technical qualification, not formal DEV or confirmation science.\n')
 source={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*.py')) if '__pycache__' not in p.parts and 'cycles' not in p.parts}
 (root/'REVISED_EXPERIMENT.md').write_text('# Three-cycle revision of the experiment\n\nThe full original specification is preserved at design/EXPERIMENT.md and design/DESIGN.json. The roster, optimizer settings, sample budgets, holdouts and adoption gates are unchanged.\n\n' + '\n\n'.join(f'Cycle {j+1}: {x[3]}' for j,x in enumerate(cycletexts)) + '\n\nTechnical qualification passed; behavioral adoption is not established. No formal DEV or confirmation run is admitted by this preparation. Any later redesign requires an explicit specification rather than post-hoc gate relaxation.\n')
 write(root/'SOURCE_LOCK.json',{'scope':'Final implementation and qualification scripts; archived per-cycle runner kernel identities remain in receipts.','files':source,'science_authorized':False})
 write(root/'QUALIFICATION.json',{'accepted':True,'scope':'Three technical cycles and full-job runner qualification only','cycle_count':3,'full_jobs':len(certs),'record_applications':n,'formal_science_ready':False,'behavioral_adoption':False})
 if i==0:
  table='| Recipe | Taught | Retained taught | Retained unseen (+,−,×,÷) |\n|---|---:|---:|---|\n'
  for x in rows:
   vals=x['held_retained_by_op'];table+=f"| {x['config']} | {100*x['taught']:.1f}% | {100*x['taught_retained']:.1f}% | "+', '.join(f'{100*vals[op]:.1f}%' for op in ('+','-','*','/'))+' |\n'
  interpretation='The generic short-budget learners do not establish usable acquisition or retained four-operation generalization. The nonselectable legacy engineered anchor recalls 99.5% in this fixture; that is not evidence for generic raw-byte rule learning. Gradient correctness and causal no-write controls pass, so these results cannot be dismissed as a missing write gate. They also do not exclude entire mechanism families.'
 else:
  table='| Recipe | Taught + | Retained unseen + | Taught × | Retained unseen × | Unseen + after × |\n|---|---:|---:|---:|---:|---:|\n'
  for x in rows:table+='| '+x['config']+' | '+' | '.join(f"{100*x[k]:.1f}%" for k in ('addition_taught','addition_held_retained','multiplication_taught','multiplication_held_retained','addition_after_multiplication'))+' |\n'
  interpretation='Repetition and joint generic learning can recover taught-item acquisition, but taught recall is distinct from a retained transferable arithmetic law. Inspect all no-write and shuffled histories in the receipts; a single technical world supports no significance or adoption claim. Counting support and byte rendering are external data only. The core contains no arithmetic parser, numerical features, operand roles, arithmetic functions or supplied arithmetic targets beyond actual answer bytes.'
 report=f'# {"RAW-RULE V2" if i==0 else "STREAM-TRANSFER V1"}: three-cycle preparation\n\nThree cycles completed; technical checks pass. No formal science, selected candidate, adopted core or GitHub upload. The goal remains exact own-output unseen arithmetic learned from feedback per sample and compute.\n\n'+table+'\n'+interpretation+f'\n\nScope: A tests binary four-operation facts; B tests counting, addition and multiplication, not mixed expressions or arbitrary NLP. All 256 payload values are supported; scored generation has a fixed 16-byte cap plus its own EOS decision. The nominal day delay performs no parameter evolution, so retention here measures intervening-cohort interference rather than biological time-dependent forgetting. B has 35 trained exchange orbits per operation; repeated presentations are counted again, not reported as distinct facts.\n\nCompleted full technical jobs: {len(certs)}; external record applications: {n}. Cycle-1 partial fixtures ({256 if i==0 else 104} records) and numerical tests are reported separately.\n\nRevisions and evidence: cycles/cycle1–3/CYCLE_REPORT.md, per-job receipts/certificates, SOURCE_LOCK.json, QUALIFICATION.json, RUNTIME_ESTIMATE.json. Initial sources and negative outputs are preserved.\n\nThe complete overlapping mechanism inventories remain referenced in arithmetic_feedback_rule_discovery_dev_v1_20261013/ops/read_only_review_20261013/inventory/FULL_INVENTORY_REVIEW.md. No inventory family was crossed out by these technical fixtures.\n\nDecision: do not expand this unchanged preparation merely to obtain more near-zero outputs. Any next scientific design must be explicit; no thresholds, sample budgets or architecture were changed in response to these scores.\n'
 (root/'PREPARATION_REPORT.md').write_text(report)
 write(root/'FINAL_AUDIT.json',{'accepted':True,'scope':'Technical preparation only','complete_cycles':3,'source_lock_present':True,'all_full_job_certificates_accepted':True,'scientific_jobs':0,'counts':summary,'behavioral_claim':'NOT_ESTABLISHED','report_generated_from_saved_certificates':True})
 zip_path=root/'PREPARATION_BUNDLE.zip'
 with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(root.rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts and p.name not in ('PREPARATION_BUNDLE.zip','PACKAGE.json'):z.write(p,p.relative_to(root))
 write(root/'PACKAGE.json',{'complete':True,'file':'PREPARATION_BUNDLE.zip','bytes':zip_path.stat().st_size,'sha256':hashlib.sha256(zip_path.read_bytes()).hexdigest(),'scientific_results':False})
 print(name,len(certs),n,'packaged')
