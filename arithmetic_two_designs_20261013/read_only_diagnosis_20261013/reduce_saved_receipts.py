"""Read saved receipts only. No learner imports, forward passes, updates or selection."""
import collections,gzip,json,re,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[2]
def load(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def outputs(rows):
 errors=collections.Counter();diff=[];first=0;mode=collections.Counter()
 for r in rows:
  g=bytes.fromhex(r['payload_hex']);a=bytes.fromhex(r['answer_hex']);mode[(g.hex(),r['status'])]+=1
  good=r['status']=='EOS' and g==a
  first+=bool(g) and bool(a) and g[0]==a[0]
  if good:errors['exact']+=1
  elif not g:errors['empty_EOS']+=1
  elif not a or g[0]!=a[0]:errors['wrong_first_byte']+=1
  elif g==a:errors['wrong_end']+=1
  else:errors['correct_first_wrong_suffix']+=1
  if r['status']=='EOS' and re.fullmatch(rb'[0-9]+',g) and re.fullmatch(rb'[0-9]+',a):diff.append(abs(int(g)-int(a)))
 expected_first=collections.Counter(bytes.fromhex(r['answer_hex'])[0] for r in rows if r['answer_hex'])
 return {'n':len(rows),'errors':dict(errors),'first_byte_correct':first,'target_first_byte_mode_count':max(expected_first.values(),default=0),
  'valid_nonnegative_integer_EOS':len(diff),'numeric_MAE_diagnostic_only':statistics.mean(diff) if diff else None,
  'within_one_diagnostic_only':sum(v<=1 for v in diff),'most_common_outputs':[(x,n) for x,n in mode.most_common(3)]}
def norms(records):
 updates=[r['update'] for r in records if r.get('update')]
 return {k:statistics.mean(u[k] for u in updates) for k in ('mean_loss','unclipped_gradient_norm')}|{'mean_step_'+k:statistics.mean(u['component_step_norms'][k] for u in updates) for k in ('E','V')}
A=[];B=[]
for p in sorted((R/'arithmetic_feedback_rule_discovery_dev_v2_20261013/cycles/cycle3/receipts').glob('*.json.gz')):
 r=load(p);h=r['histories']['W'];records=h['records'];permitted=[];correct_payload=0;payload_targets=0;correct_EOS=0
 for rec in records:
  permitted.extend(t for t,m in zip(rec['targets'],rec['permissions']) if m)
 for rec in records[-128:]:
  start=len(bytes.fromhex(rec['prompt_hex']));a=bytes.fromhex(rec['answer_hex'])
  for target,pred in zip(a,rec['choices'][start:start+len(a)]):payload_targets+=1;correct_payload+=target==pred
  correct_EOS+=rec['choices'][-1]==256
 A.append({'config':r['config'],'source_receipt':str(p.relative_to(R)),'permitted_targets':len(permitted),'EOS_targets':permitted.count(256),
  'exposure_histogram':dict(collections.Counter(collections.Counter(rec['prompt_hex'] for rec in records).values())),
  'last128_prearrival_payload_correct':correct_payload,'last128_prearrival_payload_total':payload_targets,'last128_prearrival_EOS_correct':correct_EOS,
  'taught_final':outputs(h['checkpoints'][-1]['taught_pre']),
  'mean_loss_first32':statistics.mean(rec['update']['mean_loss'] for rec in records[:32]) if records[0]['update'] else None,
  'mean_loss_last32':statistics.mean(rec['update']['mean_loss'] for rec in records[-32:]) if records[0]['update'] else None})
for p in sorted((R/'arithmetic_stream_transfer_dev_v1_20261013/cycles/cycle3/receipts').glob('*.json.gz')):
 r=load(p);phases={}
 for phase in ('addition','multiplication'):
  phases[phase]={}
  for history,h in r[phase].items():
   phases[phase][history]=[{'pass':cp['pass'],'taught_pre':outputs(cp['taught_pre']),'held_pre':outputs(cp['held_pre']),
    'held_retained':outputs(cp['held_retained']),'cohort_post':outputs(cp['cohort_post']),
    'addition_after_multiplication':outputs(cp['addition_retained']) if phase=='multiplication' else None} for cp in h['checkpoints']]
 w=r['addition']['W'];B.append({'config':r['config'],'source_receipt':str(p.relative_to(R)),'phases':phases,
  'last32_addition_update_norms':norms(w['records'][-32:]),'final_addition_cohort_update_norms':norms(w['checkpoints'][-1]['cohort_records'])})
result={'analysis_scope':'Read-only descriptive reduction of existing cycle-3 technical receipts; no new model probes or training.',
 'scientific_inference':False,'exact_own_output_gate_unchanged':True,'numeric_error_is_diagnostic_not_success':True,
 'A':A,'B':B,'interpretation_boundaries':['Final-cycle comparisons share one technical world, not independent statistical replications.',
 'B does not measure taught addition after multiplication; only held addition retention is recorded.',
 'Core math passing tests is not proof of adequate optimization or law extraction.',
 'A/B short-versus-long repetition comparison confounds dose, task/data distribution and objective; it is not a matched causal mechanism comparison.']}
out=Path(__file__).with_name('SAVED_RECEIPT_DIAGNOSIS.json');out.write_text(json.dumps(result,indent=2)+'\n');print(out)
