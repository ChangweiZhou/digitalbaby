"""Static document/schema/count/source checks only. No learners, arrays or solvers."""
from pathlib import Path
import json,re,hashlib
D=Path(__file__).resolve().parent;R=D.parent
def load(n):return json.loads((D/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def ck(name,ok):
 assert ok,name
 checks.append(name)
S=load('SPEC.json');A=S['A1'];T=load('STATUS.json');P=load('PROVENANCE.json')
ck('design_only',S['status']==T['status']=='DESIGN_ONLY' and not S['execution_authorized'] and T['models_run']==0)
ck('fixed_goal',all(S['constraints'][k] for k in ['gradual_warmup','local_candidate_updates','own_EOS']) and not any(S['constraints'][k] for k in ['arithmetic_specific_structure','attention','Transformers','ELM']))
ck('no_evidence_drift','retention' in S['evidence_levels']['E1'] and 'presentation' in S['evidence_levels']['E2'])
ck('lane_independence',S['lanes']['A1']['independent_of_B_and_BP'] and S['lanes']['C']['independent_of_A_and_B'] and not S['lanes']['B']['blocking_other_lanes'])
ck('C_not_implicitly_runnable',not S['C']['candidate_local_rule_locked'] and not S['C']['runtime_budget_authorized'])
ck('science_observations',sum(A['scientific_record_components'].values())==A['scientific_records']==2*(512+512+4*128)+32==3104)
ck('all_observations',A['scientific_records']+A['technical_record_reserve']==A['all_record_cap']==3120)
ck('target_upper_bound',A['all_record_cap']*3==A['reply_EOS_target_cap']==9360)
ck('query_pairs',sum(A['paired_query_components'].values())==A['scientific_pair_cap']==9696)
ck('technical_query_reserve',A['scientific_pair_cap']+A['technical_pair_reserve']==A['all_pair_cap']==9760)
ck('own_only_held_queries',len(A['worlds'])*4*2*50==A['held_own_only_queries']==800)
ck('distribution_upper_bound',A['all_pair_cap']*7+A['held_own_only_queries']*4==A['evaluator_distribution_cap']==71520)
ck('CPU_partition',sum(A['resources'][k] for k in ['qualification_CPU','execution_CPU','audit_package_CPU'])==A['resources']['CPU_seconds']==1200)
ck('no_automatic_followup',not A['automatic_retries'] and not A['automatic_budget_expansion'] and not A['automatic_followup'] and not A['automatic_adoption'])
ck('shared_prefix_accounting',A['physical_OLD_preparations']==4 and A['main_NEW_branches']==8)
ck('probe_roster',A['sampled_passive_probe_positions']==A['NEW_boundary_records']==[16,32,64,128])
ck('warmup_preserved',512*2>=1024 and A['NEW_writes_in_both_prior_states'])
for x in P['local_sources']:
 p=R/x.get('preserved_snapshot',x['path']);ck('source:'+x['path'],p.is_file() and sha(p)==x['sha256'])
for x in P['standing_instructions']:ck('instruction:'+x['path'],sha(Path(x['path']))==x['sha256'])
for name in ['DESIGN.md','A1_NATIVE_SCHEDULE.md','SCORING_CONTRACT.md','ARCHITECTURE_PROTOCOL.md']:
 txt=(D/name).read_text();ck('text_nonempty:'+name,len(txt)>500)
 for target in re.findall(r'\]\(([^)]+)\)',txt):
  if '://' not in target:ck('link:'+target,(D/target.split('#')[0]).exists())
ptr=json.loads((R/'PROJECT_DOCUMENTS/CURRENT_OUTPUT_DIAGNOSTIC.json').read_text())
ck('completed_V156_preserved',ptr['version']=='V156' and ptr['status']=='THREE_CYCLES_COMPLETE_DIAGNOSTIC_ONLY' and ptr['next_design']['version']=='V157')
ck('B_source_failed_fixture_preserved',S['lanes']['B']['source_commit']=='04c2873efb5ca8cc70eacc45052ec9e71231a93f' and 'FAILED' in S['lanes']['B']['status'])
out=dict(version='V157',status='PASS',scope='static documents/count arithmetic/source hashes only; not implementation or scientific qualification',checks=checks,models_run=0,solvers_run=0,new_observed_records=0)
(D/'STATIC_CHECKS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(status='PASS',checks=len(checks),models_run=0,records=0)))
