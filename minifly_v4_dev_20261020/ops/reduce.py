"""Independent receipt/exposure reduction. No learner training or trajectory replay."""
from pathlib import Path
import sys,json,hashlib,math,collections
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v4_common import HERE,ROOT,save,sha,lock
from data_v4 import dataset

def run():
 c1=json.loads((HERE/'cycles/cycle1/AUDIT.json').read_text());c2=json.loads((HERE/'cycles/cycle2/AUDIT.json').read_text());c3=json.loads((HERE/'cycles/cycle3/AUDIT.json').read_text());cfg=json.loads((HERE/'SPEC.json').read_text())
 assert c1['accepted'] and c2['accepted'] and c3['accepted']
 assert lock()==json.loads((HERE/'cycles/cycle3/SOURCE_LOCK.json').read_text())
 index=json.loads((HERE/'cycles/cycle3/COMMIT_INDEX.json').read_text());audits=[];metrics=[];target_events=0;evalcpu=prodCPU=wholeCPU=0.
 for item in index['jobs']:
  p=HERE/item['path'];assert sha(p)==item['receipt_sha256'];r=json.loads(p.read_text())
  tr,he=dataset(r['world'],64,16);lookup={p.hex():a.hex() for p,a in tr+he}
  train_keys={tuple(sorted(map(int,p.decode().rstrip('=').split('+')))) for p,_ in tr};held_keys={tuple(sorted(map(int,p.decode().rstrip('=').split('+')))) for p,_ in he};assert not train_keys&held_keys
  assert len(r['traces'])==r['record_applications']==72;assert sum(t['target_events'] for t in r['traces'])==r['target_events']
  assert sum(t['replayed'] for t in r['traces'])==8;assert sum(not t['replayed'] for t in r['traces'])==64
  assert [t['external'] for t in r['traces'][:64]]==list(range(1,65));assert all(t['external']==64 for t in r['traces'][64:])
  for t in r['traces']:assert 1/16<=t['a']<=1 and t['overflow']==0
  for phase,point in r['points'].items():
   for setname in ('taught','held'):
    ss=point[setname];actual=0
    for row in ss['rows']:
     assert lookup[row['prompt']]==row['want']
     correct=row['payload']==row['want'] and row['ended'] and not row['truncated'];assert correct==row['exact'];actual+=correct
    assert actual/len(ss['rows'])==ss['exact'];evalcpu+=ss['evaluation_CPU_seconds']
  if r['arm']=='B3_N':
   assert all(r['final_stats'][k]==0 for k in ('fast_l1','slow_l1','q_nonzero','native_write_l1'))
   assert all(v==0 for v in r['final_stats']['boundary_weights'])
  for phase in r['points']:
   point=r['points'][phase];metrics.append(dict(world=r['world'],arm=r['arm'],phase=phase,taught_exact=point['taught']['exact'],held_exact=point['held']['exact']))
  audits.append(dict(world=r['world'],arm=r['arm'],accepted=True,record_applications=72,receipt_sha256=sha(p),hash_verified=True,clock=r['points']['rest_after_replay']['clock']))
  prodCPU+=r['final_stats']['production_CPU_seconds'];wholeCPU+=r['total_worker_phase_CPU_seconds'];target_events+=r['target_events']
 # Matched lengths preserve clocks even for shuffled labels.
 for world in cfg['G1_worlds']:
  clocks=[a['clock'] for a in audits if a['world']==world];assert max(clocks)-min(clocks)<1e-6
 failed_serial=json.loads((HERE/'cycles/cycle3_preflight_serialization_failure/ACCOUNTING.json').read_text())
 failed_replay=json.loads((HERE/'cycles/cycle3_replay_zero_failure/ACCOUNTING.json').read_text())
 assert failed_serial['full_technical_record_applications']==4 and failed_replay['full_technical_record_applications']==108
 failed_train,_=dataset(202610204401,64,16)
 failed_B0_events=sum(len(a)+1 for _,a in failed_train)+sum(len(a)+1 for _,a in failed_train[-32:])+sum(len(a)+1 for _,a in failed_train[-8:])
 invalid_files=list((HERE/'cycles/cycle3_invalid_teaching_sign/jobs').glob('*/RECEIPT.json'));assert len(invalid_files)==6
 invalid_events=sum(json.loads(p.read_text())['target_events'] for p in invalid_files)
 invalid_train,_=dataset(202610204451,64,16);invalid_events+=sum(len(a)+1 for _,a in invalid_train[:16])
 native_total=c1['full_record_applications']+4+108+452+4+c3['full_native_record_applications']
 assert native_total==1602
 ledger=dict(native_record_applications=native_total,breakdown=dict(cycle1_generic=6,failed_empty_digest_preflight=4,failed_replay_zero_preflight=4,failed_replay_zero_B0_external=64,failed_replay_zero_B0_replay=40,invalid_sign_preflight=4,invalid_sign_completed_external=384,invalid_sign_completed_replay=48,invalid_sign_partial_SH_external=16,failed_passive_control_preflight=4,corrected_cycle3_preflight=4,corrected_positive_control=16,corrected_G1_external=14*64,corrected_G1_replay=14*8),
  G0_teacher_forced_oracle_record_presentations=c2['oracle_record_presentations'],G0_teacher_forced_target_events=c2['oracle_target_events'],
  all_native_plus_oracle_record_presentations=native_total+c2['oracle_record_presentations'],accepted_G1_jobs=14,correlated_histories=14,
  accepted_G1_record_applications=1008,native_observed_target_events=12+8+8+failed_B0_events+8+invalid_events+8+8+32+target_events,
  scientific_records=0,partial_native_event_fixture_count=34,
  definition='Whole record applications count per learner history, including failed technical work and replay. Oracle code-extraction presentations are separate labeled diagnostics, not native learning. 12 paired parity events, four confirmation events, one standalone sign event, one failed directional event, two matched-baseline diagnosis events and two final directional events are native partial fixtures; decode-only checks are not training.')
 assert sum(ledger['breakdown'].values())==native_total
 save(Path('TRAINING_LEDGER.json'),ledger)
 result=dict(accepted=True,scope='Separate immutable-receipt, exact-output, exposure, replay-boundary, control and clock audit; not efficacy acceptance',jobs=audits,ledger=ledger,metrics=metrics,
  verdict='NO_ADVANCE_G0_FAILED_G1_ONLY',large_P3_started=False,scientific_records=0,
  resource=dict(recorded_native_production_CPU_seconds=prodCPU,recorded_primary_evaluation_CPU_seconds=evalcpu,accepted_job_phase_CPU_seconds=wholeCPU,
   corrected_cycle3_CPU_seconds=c3['CPU_seconds'],G0_cycle_CPU_seconds=c2['CPU_seconds'],cycle1_CPU_seconds=c1['CPU_seconds'],
   corrected_cycle3_peak_RSS_bytes=c3['peak_RSS_bytes'],G0_cycle_peak_RSS_bytes=c2['peak_RSS_bytes'],
   limitation='Cycle/job intervals exclude startup imports; historical CPU of failed attempts and secondary-greedy timing are not independently itemized. No end-to-end efficiency advantage is claimed. ru_maxrss is a cumulative process high-water mark, not an isolated per-arm peak.'),
  operative_sources_unchanged=True,recorded_G0_range=[min(c['held_digit'] for c in c2['cases']),max(c['held_digit'] for c in c2['cases'])])
 save(Path('results/SUMMARY.json'),result);save(Path('FINAL_AUDIT.json'),dict(accepted=True,scope=result['scope'],summary_sha256=sha(HERE/'results/SUMMARY.json'),native_record_applications=native_total,scientific_records=0))
 print(json.dumps(dict(accepted=True,native_record_applications=native_total,G0_oracle_presentations=c2['oracle_record_presentations'],verdict=result['verdict'])),flush=True)
if __name__=='__main__':run()
