"""Independent reduction of saved records and states; never runs learning or probes."""
import sys,json,time,hashlib,collections,pickle,math,zipfile
from pathlib import Path
START=time.process_time()
P=Path(__file__).resolve().parents[1];ROOT=P.parent
sys.path.insert(0,str(P))
def read(f):return json.loads((P/f).read_text())
def rows(f):
 p=P/'results'/f
 return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
def save(f,x):(P/f).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cv(x):
 if isinstance(x,dict) and set(x)=={'hex'}:return bytes.fromhex(x['hex'])
 if isinstance(x,dict):return {k:cv(v) for k,v in x.items()}
 if isinstance(x,list):return [cv(v) for v in x]
 return x
def key(p):return tuple(sorted(map(int,p[:-1].split(b'+'))))
def partition(k,s):return int.from_bytes(hashlib.blake2b(f'{s}:{k[0]}:{k[1]}'.encode(),digest_size=8).digest(),'little')%10
def target(p):
 a,b=map(int,p[:-1].split(b'+'));return str(a+b).encode()
result=read('RUN_RESULT.json');assert result['status'] in ('THREE_CYCLES_COMPLETE','RESOURCE_LIMIT')
source=read('SOURCE_LOCK.json')
assert all(sha(ROOT/f)==h for f,h in source.items()),'changed source'
for c in ('cycle1','cycle2','cycle3'):
 assert read(f'cycles/{c}/SOURCE_LOCK.json')==source
 assert read(f'cycles/{c}/AUDIT.json')['accepted']
parent=read('PARENT_ASSET_AUDIT.json');assert all(sha(ROOT/f)==h for f,h in parent['files'].items())
datasets=cv(read('DATA_LOCK.json'))['datasets'];training=rows('TRAINING.jsonl');exposure=rows('EXPOSURES.jsonl');probes=rows('PROBES.jsonl');curves=rows('CURVES.jsonl')
D0=[];models={};exposure_counts={};states=[];groups=collections.defaultdict(list)
for r in training:groups[(r['seed'],r['arm'])].append(r)
for seed,raw in datasets.items():
 s=int(seed);d=raw;h={key(p) for p,a in d['held']};assert len(h)==64
 assert all(partition(k,s)==9 for k in h)
 allowed=d['old']+d['new']+[r for batch in d['listening'].values() for r in batch]
 assert not h&{key(p) for p,a in allowed}
 assert all(target(p)==a for p,a in allowed)
 assert not {key(p) for p,a in d['old']}&{key(p) for p,a in d['new']}
 assert collections.Counter(a for p,a in d['old'])==collections.Counter(a for p,a in d['shuffled'])
 assert all(len(a)==len(b) for (p,a),(q,b) in zip(d['old'],d['shuffled']))
 for phase,recs in d['listening'].items():
  observed=[r for r in exposure if r['seed']==s and r['phase']=='D0-'+phase]
  assert len(observed)==len(recs)
  assert all((bytes.fromhex(r['prompt']),bytes.fromhex(r['observed_answer']))==tuple(rec) for r,rec in zip(observed,recs))
 x=read(f'results/D0_{s}.json');assert x['codes']['alias_count']==0
 assert all(abs(v-1)<1e-12 for v in x['codes']['profile_sums'])
 assert x['calibration']['events']==sum(len(p)+len(a)+1 for p,a in d['listening']['D'])
 assert abs(x['native']['active_fraction']-.03535533905932738)<1e-15
 D0.append(x)
 exposure_counts[seed]={'physical_records':sum(len(r) for r in d['listening'].values()),'payload_bytes':sum(sum(map(len,r)) for batch in d['listening'].values() for r in batch),'answer_bytes':sum(len(a) for batch in d['listening'].values() for p,a in batch)}
 for arm in ('R-W','A-W','R-N_OLD','R-SH_OLD'):
  rs=groups[(s,arm)];old=[r for r in rs if r['phase']=='OLD'];new=[r for r in rs if r['phase']=='NEW']
  cohort=d['shuffled'] if arm=='R-SH_OLD' else d['old']
  for index,r in enumerate(old):
   assert r['index']==index+1
   assert (bytes.fromhex(r['prompt']),bytes.fromhex(r['observed_answer']))==tuple(cohort[d['old_order'][index]])
   assert r['write']==(arm!='R-N_OLD')
  for index,r in enumerate(new):
   assert r['index']==index+1 and r['write']
   assert (bytes.fromhex(r['prompt']),bytes.fromhex(r['observed_answer']))==tuple(d['new'][d['new_order'][index]])
  for index,r in enumerate(rs):assert abs(r['strength']-(1/16)*(1/16+15/16*min(index/1024,1)))<1e-15
  if result['status']=='THREE_CYCLES_COMPLETE':assert len(old)==512 and len(new)==128
# Every exact score is independently reconstructed from the actual emitted bytes/EOS.
for r in probes:
 p=bytes.fromhex(r['prompt']);expected=bytes.fromhex(r['expected']);assert target(p)==expected
 assert r['exact']==(r['payload']==r['expected'] and r['ended'] and not r['truncated'])
 if r['cohort']=='HELD':assert key(p) in {key(p) for p,a in datasets[str(r['seed'])]['held']}
for c in curves:
 subset=[r for r in probes if all(r[k]==c[k] for k in ('seed','arm','phase','cohort'))]
 assert len(subset)==c['records'];g=collections.defaultdict(list)
 for r in subset:g[key(bytes.fromhex(r['prompt']))].append(int(r['exact']))
 exact=sum(sum(v)/len(v) for v in g.values())/len(g)
 assert abs(exact-c['exact'])<1e-15 and c['readonly']
# Inspect saved final states, never execute their predictions or updates.
from r2_common import np
for entry in result['completed_histories']:
 s,arm=entry['seed'],entry['arm'];f=P/'results'/f'FINAL_{s}_{arm}.pkl'
 with f.open('rb') as inp:m=pickle.load(inp)
 with (P/'results'/f'PRESTATE_{s}.pkl').open('rb') as inp:codes,g,d0events=pickle.load(inp)
 assert pickle.dumps(m.codes,protocol=5)==pickle.dumps(codes,protocol=5)
 assert pickle.dumps(m.geometry,protocol=5)==pickle.dumps(g,protocol=5)
 rs=groups[(s,arm)];events=sum(len(bytes.fromhex(r['prompt']))+len(bytes.fromhex(r['observed_answer']))+1 for r in rs)
 assert abs(m.engine.time-((d0events+events)*(30/14)+7200))<1e-6
 assert m.external==len(rs)==640 and m.engine.gated==(arm!='A-W')
 for a in [m.engine.bank.fast,m.engine.bank.slow,m.engine.bank.adapt,m.engine.q]:assert np.isfinite(a).all()
 states.append({'seed':s,'arm':arm,'record_count':m.external,'bytes':f.stat().st_size,'sha256':sha(f),
 'native_time':m.engine.time,'q_abs_quantiles':np.quantile(abs(m.engine.q),[0,.5,.9,.99,1]).tolist(),
 'q_nonzero_by_compartment':np.sum(m.engine.q!=0,axis=1).tolist(),
 'fast_l1_by_compartment':np.sum(abs(m.engine.bank.fast),axis=(1,2)).tolist(),
 'slow_l1_by_compartment':np.sum(abs(m.engine.bank.slow),axis=1).tolist(),
 'confirmations':m.engine.confirmations,'write_l1':m.engine.write_l1,'slow_write_l1':m.engine.slow_write_l1})
 del m,codes,g
# Recompute the recitation envelope on exactly the observed stream for each arm.
from r2_data import countermodels as make_countermodels,orientations,build,jsonable
for seed,d in datasets.items():assert jsonable(build(int(seed)))==read('DATA_LOCK.json')['datasets'][seed], 'deterministic data mismatch'
for (s,arm),rs in groups.items():
 if not rs:continue
 d=datasets[str(s)];observed=[r for batch in d['listening'].values() for r in batch]
 observed += [(bytes.fromhex(r['prompt']),bytes.fromhex(r['observed_answer'])) for r in rs]
 models[f'{s}/{arm}']=make_countermodels(observed,orientations(d['held']))
 production=sum(r['cpu'] for r in rs)
 entry=next((e for e in result['completed_histories'] if (e['seed'],e['arm'])==(s,arm)),None)
 if entry:assert abs(entry['production_CPU_seconds']-production)<1e-7
started_keys={k for k,rs in groups.items() if rs}
if result.get('partial_history'):started_keys.add((result['partial_history']['seed'],result['partial_history']['arm']))
charged_executed_D0_records=sum(exposure_counts[str(seed)]['physical_records'] for seed,arm in started_keys)
charged_executed_D0_payload=sum(exposure_counts[str(seed)]['payload_bytes'] for seed,arm in started_keys)
charged_executed_D0_CPU=sum(next(x['CPU_seconds'] for x in D0 if x['seed']==seed) for seed,arm in started_keys)
main_records=len(training);physicalD0=sum(x['physical_records'] for x in exposure_counts.values());D0bytes=sum(x['payload_bytes'] for x in exposure_counts.values());answers=sum(x['answer_bytes'] for x in exposure_counts.values())
physicalD0CPU=sum(x['CPU_seconds'] for x in D0);chargedD0CPU=physicalD0CPU*4
complete=result['status']=='THREE_CYCLES_COMPLETE'
final=[c for c in curves if c['phase']=='FINAL'];full_held=[c for c in final if c['cohort']=='HELD']
all_final_zero=bool(full_held) and all(c['exact']==0 for c in full_held)
verdict=('NO_ADVANCE_NO_EXACT_ADDITION' if all_final_zero else 'ENGINEERING_REVIEW_REQUIRED') if complete else 'RESOURCE_LIMIT_NO_EFFICACY_VERDICT'
summary={'status':result['status'],'verdict':verdict,'cycles_executed':3,'independent_engineering_births':len(D0),'completed_correlated_histories':len(result['completed_histories']),
 'native_main_full_record_applications':main_records,'native_main_payload_bytes':sum(len(bytes.fromhex(r['prompt']))+len(bytes.fromhex(r['observed_answer'])) for r in training),
 'native_main_answer_bytes':sum(len(bytes.fromhex(r['observed_answer'])) for r in training),'native_main_EOS':main_records,
 'native_main_events':sum(len(bytes.fromhex(r['prompt']))+len(bytes.fromhex(r['observed_answer']))+1 for r in training),
 'D0_physical_observed_records':physicalD0,'D0_physical_payload_bytes':D0bytes,'D0_physical_answer_bytes':answers,
 'D0_charged_records_all_8_histories':physicalD0*4,'D0_charged_payload_bytes_all_8_histories':D0bytes*4,
 'D0_charged_records_executed_histories':charged_executed_D0_records,'D0_charged_payload_executed_histories':charged_executed_D0_payload,'started_histories':len(started_keys),'total_charged_observed_records':charged_executed_D0_records+main_records,'D0_CPU_charged_executed_histories':charged_executed_D0_CPU,'D0_model_selection_record_views':sum(x['model_selection_views'] for x in D0),
 'additional_synthetic_vector_fixture_views':320,'M3_profile_views_physical':sum(x['codes']['profile_views'] for x in D0),'M3_candidate_profile_views_physical':sum(sum(m['reservoir_views'] for m in x['codes']['M3'].values()) for x in D0),'synthetic_component_payload_bytes':read('cycles/cycle1/AUDIT.json')['synthetic_payload_exposures'],
 'synthetic_component_validation_payload_bytes':read('cycles/cycle1/AUDIT.json')['synthetic_validation_payload'],
 'partial_native_fixture_events':read('cycles/cycle1/AUDIT.json')['partial_native_events']+sum(x['response']['partial_native_events'] for x in D0),
 'query_records':len(probes),'query_native_events':sum(r['events'] for r in probes),'production_CPU_seconds':sum(r['cpu'] for r in training),
 'evaluation_CPU_seconds':sum(c['CPU_seconds'] for c in curves),'D0_physical_CPU_seconds':physicalD0CPU,'D0_CPU_charged_all_8_histories':chargedD0CPU,
 'executed_resources':result['resources'],'import_failure_before_training':read('ops/import_failure_1/FAILURE.json'),
 'scientific_jobs':0,'new_core_adopted':False,'formal_campaign_ready':False,'rate_or_sample_plan_changed':False,'replay_records':0,
 'evidence':'E0 component/interface repairs; complete E1 acquisition diagnostic and exploratory E3 test, no positive addition generalization established' if complete else 'E0 component/interface repairs and negative addition acquisition/held observations on completed histories; full planned two-birth paired E3 harness incomplete',
 'final_curves':final,'countermodels':models,'D0_by_birth':[{'seed':x['seed'],'byte_aliases':x['codes']['alias_count'],'selected_C':x['codes']['selection']['selected'],'occupied_classes':x['codes']['occupied_classes'],'before_adjacent_overlap':x['geometry_before']['adjacent_overlap_mean'],'after_adjacent_overlap':x['geometry_after']['adjacent_overlap_mean'],'physical_exposure':exposure_counts[str(x['seed'])]} for x in D0],
 'teacher_forced':[],
 'limitations':['Two engineering births, correlated arms, small number of facts; no confirmatory statistics','M3 C=1 remains allowed and occurred in birth2; not evidence for general impossibility','Actual-frame response tested one event at initial strength, not full safe/effective dose ceiling','Only aggregate final q/fast/slow allocation statistics retained; per-event mixed-sign/spacing rejection rates were not instrumented','Global projections/sorts/native sweeps remain; no scalable sparse-compute claim','Import-only failed startup CPU was not instrumented; successful controller CPU includes imports, compilation and child memory probes']}
tf=[]
for (s,arm),rs in groups.items():
 for phase in ('OLD','NEW'):
  z=[r for r in rs if r['phase']==phase]
  if z:tf.append(dict(seed=s,arm=arm,phase=phase,records=len(z),mean_loss=sum(r['loss'] for r in z)/len(z),byte_accuracy=sum(r['correct_teacher_prefix'] for r in z)/sum(r['target_events'] for r in z),bit_accuracy=sum(r['bit_correct'] for r in z)/sum(r['bit_count'] for r in z)))
summary['teacher_forced']=tf
partial_audit=None
if result.get('partial_history'):
 entry=result['partial_history'];seed,arm=entry['seed'],entry['arm']
 with (P/'results/PARTIAL_STATE.pkl').open('rb') as inp:m=pickle.load(inp)
 with (P/'results'/f'PRESTATE_{seed}.pkl').open('rb') as inp:codes,g,d0events=pickle.load(inp)
 assert pickle.dumps(m.codes,protocol=5)==pickle.dumps(codes,protocol=5) and pickle.dumps(m.geometry,protocol=5)==pickle.dumps(g,protocol=5)
 rs=groups[(seed,arm)];assert m.external==len(rs)==entry['records']
 for ar in [m.engine.bank.fast,m.engine.bank.slow,m.engine.bank.adapt,m.engine.q]:assert np.isfinite(ar).all()
 events=sum(len(bytes.fromhex(r['prompt']))+len(bytes.fromhex(r['observed_answer']))+1 for r in rs)
 rest=m.engine.time-(d0events+events)*(30/14);assert min(abs(rest-v) for v in (0,3600,7200))<1e-6
 partial_audit=dict(seed=seed,arm=arm,full_record_applications=m.external,rest_seconds=round(rest),state_sha256=sha(P/'results/PARTIAL_STATE.pkl'),frozen_representation_preserved=True,finite=True)
 save('results/PARTIAL_STATE_AUDIT.json',partial_audit)
 del m,codes,g
summary['partial_history']=partial_audit
save('SUMMARY.json',summary);save('results/STATE_AUDIT.json',states)
audit={'accepted':True,'meaning':'record/source/exposure/state integrity; not model efficacy qualification','complete_harness':complete,'no_held_answer_exposure':True,'own_output_scores_recomputed':True,'source_and_parent_hashes_match':True,'D0_and_main_streams_match_locked_data':True,'warmup_schedule_verified':True,'final_states_match_D0_freezes_and_clocks':True,'records':main_records,'histories':len(states),'probe_records':len(probes),'scientific_jobs':0,'auditor_sha256':sha(Path(__file__)),'artifact_reduction_CPU_seconds':time.process_time()-START}
save('FINAL_AUDIT.json',audit)
print(json.dumps({'verdict':verdict,'records':main_records,'histories':len(states),'D0':summary['D0_by_birth'],'teacher_forced':tf,'audit_CPU':audit['artifact_reduction_CPU_seconds']},indent=2))
