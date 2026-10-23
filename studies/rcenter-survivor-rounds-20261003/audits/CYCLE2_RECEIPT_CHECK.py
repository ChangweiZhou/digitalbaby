"""Independent pure receipt validation; never import or re-run native learner."""
import importlib.util,json,hashlib,math,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(),parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
i=load('receipt_integrity',ROOT/'source/runtime/integrity.py');f=load('receipt_fixture',ROOT/'source/runtime/survivor_fixture.py')
review=read(ROOT/'audits/CYCLE2_SOURCE_REVIEW.json');receipt=ROOT/'receipts/cycle2/smoke.json';d=read(receipt);ledger=read(ROOT/'operations/RUN_LEDGER.json');w=f.make_world(310101)
assert sha(receipt)=='661ca626edd7aee7427b85a64c1f66034d2b4fc06dd404bcf20becdd6381ccef'
assert review['accepted'] is True and review['source_hashes']==i.hashes()==d['source']
assert sha(ROOT/'protocol'/review['design_file'])==review['design_sha256']
assert sha(ROOT/'audits/CYCLE2_SOURCE_REVIEW.md')==review['report_sha256']
assert d['passed'] is True and d['schema']=='RC-SURVIVOR-CYCLE2-TEST-v1' and d['fixture']==w['sha256']
assert d['state_equivalence_scope']=='fixed open-loop observed-outcome stream; returned policy output only'
expected_checks=['learner_boundary_and_poisoned_inherited_generators','invalid_outcome_sequence_and_time_rejected','operative_mutable_clone_isolation','independent_brain_elapsed_fe_corruption_detected','full_and_fixed_fingerprint_sensitivity','48_locked_teaching_records','all_policy_operative_states_identical','causal_nonplastic_states_identical','all_branch_clocks_and_delays','write_clamps_correct','96_disposable_first_pre_feedback_probes','scoring_label_independence','source_and_report_tamper_rejected_and_restored']
assert d['checks']==expected_checks
branches=('W','N_old_relation','N_new','half','shared','forced');policy=('W','half','shared','forced')
assert len(d['steps'])==48 and len(d['phase_rows'])==192
steps={(x['record'],x['branch']):x for x in d['steps']};phases={(x['record'],x['phase'],x['branch']):x for x in d['phase_rows']}
assert len(steps)==48 and len(phases)==192
fixed={x['fixed'] for x in d['phase_rows']};assert len(fixed)==1;fixed=next(iter(fixed));assert re.fullmatch('[a-f0-9]{64}',fixed)
for j in range(8):
 stage='old' if j<4 else 'new'
 for b in branches:
  x=steps[j,b];assert x['stage']==stage and x['clock_error']==0 and x['predicted'] in f.ALPHABET
  assert x['blocked']==((b=='N_old_relation' and stage=='old')or(b=='N_new' and stage=='new'))
  assert x['state']==phases[j,'flush',b]['operative'] and x['nonplastic']==phases[j,'flush',b]['nonplastic']
 assert steps[j,'forced']['predicted']==f.ALPHABET[(f.ALPHABET.index(steps[j,'W']['predicted'])+1)%4]
 assert len({steps[j,b]['state'] for b in policy})==1
 for phase in ('prediction','outcome','newline','flush'):
  assert len({phases[j,phase,b]['operative'] for b in policy})==1
  assert len({phases[j,phase,b]['nonplastic'] for b in branches})==1
  if j<4:assert phases[j,phase,'W']['operative']==phases[j,phase,'N_new']['operative']
 assert steps[j,'W']['state']!=steps[j,'N_old_relation']['state']
 if j>=4:assert steps[j,'W']['state']!=steps[j,'N_new']['state']
assert len({phases[0,'prediction',b]['operative'] for b in branches})==1
clockrows={(x['at'],x['branch']):x['clocks'] for x in d['clocks']};assert len(d['clocks'])==len(clockrows)==24
for point in ('old_end','new_start','new_end','final'):
 at=w[point];n=4 if point in ('old_end','new_start') else 8
 last=w['events'][3 if n==4 else 195]['at']+13*f.DT
 for b in branches:
  rows=clockrows[at,b];assert len(rows)==8
  for x in rows:
   assert x['brain_t']==x['fe_t']==at and x['elapsed']-x['elapsed_base']==at
   assert x['elapsed_base']==0 and x['pending_t'] is None and x['last_byte_t']==last
   assert x['teach_seen']==n and x['bytes_seen']==14*n
SCALES=(1.4911274663291492,1.3452365735750882)
def probecheck(q,b):
 assert q['state_before']==q['state_after'] and q['fixed_identity']==fixed
 assert q['clocks']==clockrows[w['final'],b]
 for x in q['rows']:
  assert x['at']==w['final'] and x['first_pre_feedback'] is True
  assert x['continuing_state_before']==x['continuing_state_after']==q['state_before'] and x['fixed_identity']==fixed
  s=[v/SCALES[0] for v in x['shared']];p=[v/SCALES[1] for v in x['private']]
  assert len(s)==len(p)==4 and all(math.isfinite(v) for v in s+p)
  scores={'alpha1':[a+b for a,b in zip(s,p)],'alpha_half':[a+.5*b for a,b in zip(s,p)],'alpha0':s,'private_only':p}
  assert x['combined']==scores['alpha1'] and set(x['policies'])==set(scores)
  for name,u in scores.items():
   item=x['policies'][name];ties=[j for j,v in enumerate(u) if v==max(u)];emitted=f.ALPHABET[ties[0]]
   assert item=={'scores':u,'emitted':emitted,'ties':ties,'correct':int(emitted==x['target'])}
  assert x['emitted']==x['policies']['alpha1']['emitted']
assert set(d['probes'])=={'W','N_old_relation','N_new'}
for b,q in d['probes'].items():
 probecheck(q,b);assert len(q['rows'])==32
 expected=[(name,j,ch,w['sets'][name]['outcomes'][j]) for name in ('old','heldout','new') for j,ch in enumerate(w['sets'][name]['cues'])]
 assert [(x['set'],x['item'],x['cue_hex'],x['target']) for x in q['rows']]==expected
lab=d['label_check'];a=lab['first_probe'];b=lab['second_probe'];probecheck(a,'W');probecheck(b,'W')
assert lab['prediction_unchanged'] is True and len(a['rows'])==len(b['rows'])==1
assert a['state_before']==b['state_before']==lab['continuing_state']==d['probes']['W']['state_before']
x=a['rows'][0];y=b['rows'][0];actual=w['sets']['heldout']['outcomes'][0]
assert x['cue_hex']==y['cue_hex']==w['sets']['heldout']['cues'][0]
assert x['target']==actual and y['target']==f.ALPHABET[(f.ALPHABET.index(actual)+1)%4]
for field in ('emitted','shared','private','combined'):assert x[field]==y[field]
for name in x['policies']:
 for field in ('scores','emitted','ties'):assert x['policies'][name][field]==y['policies'][name][field]
attempts=[a for a in ledger['attempts'] if a['key']=='cycle2/test'];assert len(attempts)==1;a=attempts[0]
assert a['status']=='completed' and a['error'] is None and a['receipt_sha256']==sha(receipt) and a['source_digest']==review['source_digest']
assert a['runtime']=={'python':'3.11.15','numpy':'2.2.6','scipy':'1.14.1','numba':'0.61.2'}
assert 0<d['resources']['wall_s']<=a['charged_s']<=900
assert 0<d['resources']['peak_rss_bytes']<=a['peak_rss_bytes']<=512*1024**2
assert ledger['worker_s']==ledger['active_wall_s']==sum(x['charged_s'] for x in ledger['attempts'])<12*3600
assert all(x['status']=='completed' for x in ledger['attempts'])
log=(ROOT/'operations/cycle2.log').read_text().splitlines();assert len(log)==1;logged=json.loads(log[0]);assert logged=={'passed':True,'resources':d['resources'],'checks':d['checks']}
active=read(ROOT/'operations/ACTIVE_JOB.json');assert active['mode']=='cycle2' and active['source_digest']==review['source_digest']
assert sorted(str(p.relative_to(ROOT/'receipts')) for p in (ROOT/'receipts').rglob('*.json'))==['cycle1/smoke.json','cycle2/smoke.json']
# Capture an immutable audit view before later phases add ledger entries.
snapshot=ROOT/'audits/CYCLE2_LEDGER_SNAPSHOT.json';assert not snapshot.exists() or snapshot.read_bytes()==(ROOT/'operations/RUN_LEDGER.json').read_bytes();
if not snapshot.exists():snapshot.write_bytes((ROOT/'operations/RUN_LEDGER.json').read_bytes())
inputs=['audits/CYCLE2_SOURCE_REVIEW.md','audits/CYCLE2_SOURCE_REVIEW.json','protocol/ROUND1_DESIGN_CYCLE2.md','receipts/cycle2/smoke.json','operations/cycle2.log','audits/CYCLE2_LEDGER_SNAPSHOT.json']
result={'schema':'rcenter-cycle2-receipt-check-v1','passed':True,'reviewer_executed_native_learner':False,'receipt_sha256':sha(receipt),'source_digest':review['source_digest'],'source_closure_files':len(d['source']),'fixture':w['sha256'],'checks':expected_checks,'steps':48,'phase_rows':192,'clock_checkpoints':24,'store_clock_rows':192,'causal_probe_rows':96,'target_label_probe_rows':2,'policy_row_recomputations':392,'fixed_identity':fixed,'forced_outputs_changed':8,'old_w_equals_n_new_before_new':True,'old_control_full_certificate_differs':True,'new_control_full_certificate_differs':True,'flush_error_max':0.0,'attempts':1,'charged_s':a['charged_s'],'peak_rss_bytes':a['peak_rss_bytes'],'programme_worker_s':ledger['worker_s'],'input_hashes':{p:sha(ROOT/p) for p in inputs}}
(ROOT/'audits/CYCLE2_RECEIPT_CHECK.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps(result))
