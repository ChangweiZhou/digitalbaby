"""Metadata-only consistency checks. Never imports a learner, array loader or solver."""
from pathlib import Path
import hashlib, json, re

D=Path(__file__).resolve().parent
R=D.parent
def load(name):return json.loads((D/name).read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
checks=[]
def check(name,condition):
 assert condition,name
 checks.append(name)

S=load('SPEC.json');A=load('ASSETS.json');P=load('PROVENANCE.json');T=load('STATUS.json')
check('version_and_design_only',all(x['version']=='V156' for x in [S,A,P,T]) and S['status']==T['status']=='DESIGN_ONLY')
check('zero_execution',not S['execution_authorized'] and not T['execution_authorized'] and T['model_queries']==T['solver_runs']==T['new_training_record_applications']==0)
check('unchanged_goal_constraints',S['constraints']['gradual_warmup'] and S['constraints']['local_candidate_updates'] and not any(S['constraints'][k] for k in ['arithmetic_specific_structure','attention','transformers','ELM','assessment_labels_opened','core_adoption']))
check('fixed_nested_families',[x['id'] for x in S['lane_F']['families']]==['C0','CQ','FQ'] and S['lane_F']['unit_margin']==1)
check('four_cases_twelve_checks',len(S['worlds'])*len(S['fronts'])==4 and S['lane_F']['family_case_checks']==12)
check('solver_bound',12*64*384*256==S['lane_F']['solver']['separation_constraint_evaluations_max']==75497472)
pairs=4*80+64*3*80+4*80+4*80
check('panel_query_arithmetic',pairs==S['lane_M']['native_panel_query_pairs_max']==16320)
check('output_distribution_arithmetic',pairs*7==S['totals']['native_panel_output_distributions_max'] and pairs*7+1536*4+64*7==S['totals']['all_output_distributions_max']==120832)
check('record_target_arithmetic',64+64+16==S['totals']['new_observed_record_applications_max']==144 and 144*3==S['totals']['new_reply_EOS_targets_max']==432 and S['technical']['payload_bytes_max']==2)
check('shared_cpu_budget',sum(S['resources'][k] for k in ['cpu_F','cpu_M','cpu_qualification','cpu_audit_package'])==S['resources']['cpu_seconds_total']==3600)
check('no_retry_or_budget_reset',not S['resources']['automatic_retries'] and not S['resources']['automatic_resumption'])
check('eight_checkpoints',sum(x['role']=='immutable_checkpoint' for x in A['assets'])==8)
for x in A['assets']:
 p=Path(x['path']);p=p if p.is_absolute() else R/p
 check('asset:'+x['path'],p.is_file() and sha(p)==x['sha256'])
for x in P['local_sources']:
 check('source:'+x['path'],sha(R/x['path'])==x['sha256'])
for x in P['parent_source_lock_checks']:
 check('parent_source:'+x['path'],x['matches'])
lock=json.loads((R/'minifly_v154_dev_20261025/SOURCE_LOCK.json').read_text())
check('parent_source_lock_unchanged',all(sha(R/p)==h for p,h in lock.items()))
history=P['previous_output_pointer']
check('prior_pointer_preserved',sha(R/history['path'])==history['sha256'])
ptr=json.loads((R/'PROJECT_DOCUMENTS/CURRENT_OUTPUT_DIAGNOSTIC.json').read_text())
check('completed_V154_preserved',ptr['version']=='V154' and ptr['status']=='THREE_CYCLES_COMPLETE' and ptr['next_design']['version']=='V156' and not ptr['next_design']['execution_authorized'])
check('remote_followup_pinned',S['remote_followup']['commit']=='2501683d4867d2111a004f060edd28aaee67ffb4' and S['remote_followup']['no_text_task_added'])
doc=(D/'DESIGN.md').read_text()
check('correct_units_and_scope','FastMemory.rest(idle, actual_end.time - parent.time)' in doc and 'single-record counterfactuals' in doc and 'initially-correct **OLD**' in doc)
check('feature_parity_and_unit_margin','bitwise' in doc and 'rescaled parameters' in doc and 'exact rational' in doc)
check('new_followup_in_design','2.699' in doc and 'V155 local-key-learning' in doc and 'not a BPS ceiling' in doc)
for md in ['DESIGN.md','EVIDENCE_REVIEW.md']:
 for target in re.findall(r'\]\(([^)]+)\)',(D/md).read_text()):
  if '://' not in target:check('local_link:'+target,(D/target.split('#')[0]).exists())
out=dict(status='PASS',scope='Static design metadata, count arithmetic, file/hash and link checks only; not technical qualification or scientific validation',version='V156',checks=checks,models_executed=0,checkpoints_deserialized=0,feature_arrays_loaded=0,solvers_executed=0,new_observed_record_applications=0)
(D/'STATIC_CHECKS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(status='PASS',version='V156',checks=len(checks),models_executed=0,solvers_executed=0)))
