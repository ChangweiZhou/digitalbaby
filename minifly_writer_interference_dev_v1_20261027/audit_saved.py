"""Independent saved-artifact reducer: no learner imports, queries, replay or training."""
import json,hashlib,math,time,statistics,collections
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parent;T=time.process_time()
def read(n):return json.loads((H/n).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):(H/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def panel(p,rows):
 assert sha(H/p['path'])==p['sha256'];raw=read(p['path']);lookup={r['id']:r for r in rows};assert {r['id'] for r in raw}==set(lookup) and len(raw)==len(rows)
 for r in raw:
  rr=lookup[r['id']];assert r['correct']==(r['EOS'] and r['output_hex']==rr['answer']);assert r['CAP']==(not r['EOS'])
  assert len(bytes.fromhex(r['output_hex']))<=4 and len(r['tf'])==2
  for s,y in zip(r['tf'],[*bytes.fromhex(rr['answer']),256]):assert s['y']==y and 0<=s['argmax']<=256 and math.isfinite(s['lp']) and math.isfinite(s['margin'])
 assert p['correct']==sum(r['correct'] for r in raw)
 assert set(p['correct_ids'])=={r['id'] for r in raw if r['correct']}
 return raw
def run():
 assert not (H/'STOP.json').exists(), 'execution stopped; preserve partial artifacts'
 assert read('STATUS.json')['status']=='MAIN_COMPLETED'
 lock=read('SOURCE_LOCK.json');assert all(sha(R/p)==v for p,v in lock.items())
 cal=read('CALIBRATION_LOCK.json');assert cal['accepted']
 for s in cal['solves'].values():
  assert len(s['evaluations'])<=20 and abs(statistics.median(s['doses'])/s['target']-1)<=.01
  trace=sorted(s['evaluations'],key=lambda r:r['gain']);assert all(a['d']<=b['d']+1e-12 for a,b in zip(trace,trace[1:]))
 data=read('MANIFEST.json');results=read('RESULTS.json');assert len(results)==2
 ledger=[json.loads(line) for line in (H/'LEDGER.jsonl').read_text().splitlines()];training=[r for r in ledger if r['event']=='train_record'];counts=read('COUNTS.json');c=counts['counts']
 assert len(training)==3466 and c['full_records']==3466
 assert sum(len(r['sites']) for r in training)==6932
 assert c['qualification_records']==10 and c['calibration_records']==128 and c['OLD_records']==2048
 assert sum(c['NEW_'+a+'_records'] for a in ('N-low','N-high','D-low','D-high','OBS'))==1280
 assert c['shadow_targets']<=4096 and c['own_TF_pairs']<=5408
 ownpairs=0;worldsummary=[];pulse_summary=[];dose_rows=[]
 for wr in results:
  w=wr['world'];d=data[str(w)];rows=d['rows'];old=rows[:64];new=rows[64:]
  assert len(rows)==80 and len({r['prompt'] for r in rows})==80
  assert all(len(bytes.fromhex(r['prompt']))==5 and len(bytes.fromhex(r['answer']))==1 for r in rows)
  assert sorted(collections.Counter(r['answer'] for r in old).values())==[4]*16 and sorted(collections.Counter(r['answer'] for r in new).values())==[1]*16
  panel(wr['pre'],rows);ownpairs+=80;initial=set(wr['initial_OLD_correct']);assert initial==set(wr['pre']['correct_ids'])&{r['id'] for r in old}
  clock0=None;finals={};contrasts={};states={}
  for arm,stages in wr['branches'].items():
   assert set(stages)=={'1','2','4','8'}
   if arm!='IDLE':
    cc=read(f'EVENT_CLOCK_{w}_{arm}.json');assert len(cc)==128
    if clock0 is None:clock0=cc
    else:assert cc==clock0
   for k,p in stages.items():
    raw=panel(p,rows);ownpairs+=80
    assert p['OLD']==sum(r['correct'] for r in raw if r['cohort']=='OLD') and p['NEW']==sum(r['correct'] for r in raw if r['cohort']=='NEW')
    assert p['retained']==len(initial&set(p['correct_ids'])) and p['lost']==len(initial-set(p['correct_ids']))
    assert abs(p['delta_t']-int(k)*16*9*(30/14))<1e-6
    assert p['n']==(2048 if arm=='IDLE' else 2048+int(k)*32)
    states.setdefault(k,[]).append((arm,p))
   finals[arm]={k:stages['8'][k] for k in ('OLD','NEW','correct','retained','lost','empty_EOS','CAP')}
  for k,pairs in states.items():
   ps=dict(pairs);obs=ps['OBS'];idle=ps['IDLE'];assert len({p['nonvalue_hash'] for a,p in pairs if a!='IDLE'})==1
   contrasts[k]=dict(observation_OLD=obs['OLD']-idle['OLD'],time_OLD=idle['OLD']-wr['pre']['OLD'],write_OLD={a:ps[a]['OLD']-obs['OLD'] for a in ('N-low','N-high','D-low','D-high')},write_retained={a:ps[a]['retained']-obs['retained'] for a in ('N-low','N-high','D-low','D-high')})
  worldsummary.append(dict(world=w,pre_OLD=wr['pre']['OLD'],pre_NEW=wr['pre']['NEW'],initial_OLD_correct=len(initial),final=finals,contrasts=contrasts))
  pulses=read(f'PULSES_{w}.json');assert len(pulses)==16
  for pulse in pulses:
   sentinelids=set(pulse['zero']['correct_ids']);sentraw=read(pulse['zero']['path']);sentrows=[r for r in old if r['id'] in {s['id'] for s in sentraw}];assert len(sentrows)==8;panel(pulse['zero'],sentrows);ownpairs+=8
   for arm,p in pulse['interventions'].items():
    if not p['matched']:continue
    assert len(p['trace'])<=20 and abs(p['actual_d']/p['target_d']-1)<=.01
    panel(p['sentinels'],sentrows);ownpairs+=8
    pulse_summary.append(dict(world=w,id=pulse['id'],site=pulse['site'],arm=arm,d=p['actual_d'],target=p['target_d'],target_logp_change=p['target_logp_change'],margin_change=p['margin_change'],sentinel_count_change=p['sentinels']['OLD']-pulse['zero']['OLD'] if 'OLD' in pulse['zero'] else p['sentinels']['correct']-pulse['zero']['correct']))
  for rr in training:
   if not rr['tier'].startswith('NEW_') or not rr['id'].startswith(str(w)+':'):continue
   for site in rr['sites']:
    if 'delivered_d' in site:dose_rows.append(dict(world=w,arm=rr['tier'][4:],n=site['n'],y=site['y'],d=site['delivered_d'],coef_RMS=site['coef_RMS'],synaptic_L2=site['synaptic_L2']))
 assert ownpairs==5280 and c['own_TF_pairs']==ownpairs+8
 assert len(pulse_summary)==128 and len(dose_rows)==64
 assert counts['CPU']<3600 and counts['peak_RSS_bytes']<1073741824
 files=list(H.rglob('*'));total=sum(p.stat().st_size for p in files if p.is_file());assert total<1073741824
 for name in ('OLD_202610277311.json','OLD_202610277312.json','CALIBRATION_PARENT.json'):
  rr=read(name);assert sha(H/rr['path'])==rr['sha256']
 aggregates={}
 for a in ('N-low','N-high','D-low','D-high'):
  ps=[p for p in pulse_summary if p['arm']==a];ds=[r['d'] for r in dose_rows if r['arm']==a]
  aggregates[a]=dict(pulses=len(ps),mean_target_logp_change=statistics.mean(p['target_logp_change'] for p in ps),mean_margin_change=statistics.mean(p['margin_change'] for p in ps),mean_OLD_sentinel_count_change=statistics.mean(p['sentinel_count_change'] for p in ps),median_longitudinal_checkpoint_d=statistics.median(ds),min_longitudinal_checkpoint_d=min(ds),max_longitudinal_checkpoint_d=max(ds))
 by_context={(p['world'],p['id'],p['site'],p['arm']):p for p in pulse_summary}
 contrasts=[]
 for w in (202610277311,202610277312):
  for site in (0,1):
   for level in ('low','high'):
    ns=[p for p in pulse_summary if p['world']==w and p['site']==site and p['arm']=='N-'+level]
    ds=[by_context[(w,p['id'],site,'D-'+level)] for p in ns]
    contrasts.append(dict(world=w,site=site,level=level,contexts=len(ns),mean_D_minus_N_logp=statistics.mean(d['target_logp_change']-n['target_logp_change'] for n,d in zip(ns,ds)),mean_D_minus_N_margin=statistics.mean(d['margin_change']-n['margin_change'] for n,d in zip(ns,ds)),mean_D_minus_N_sentinel=statistics.mean(d['sentinel_count_change']-n['sentinel_count_change'] for n,d in zip(ns,ds)),median_actual_D_to_N_d=statistics.median(d['d']/n['d'] for n,d in zip(ns,ds))))
 save('PAIRED_FORM_CONTRASTS.json',contrasts)
 summary=dict(identity='WCFI-V1',evidence=['E0','E1','E1-R'],main_records=3328,main_targets=6656,all_full_records=3466,all_full_targets=6932,worlds=worldsummary,pulses=aggregates,calibration_gains=cal['gains'],low_d=cal['low'],high_d=cal['high'],matching_max_relative_error=max(abs(p['d']/p['target']-1) for p in pulse_summary),arithmetic_generalization_tested=False,new_core_adopted=False)
 save('SUMMARY.json',summary);save('PULSE_EFFECTS.json',pulse_summary);save('LONGITUDINAL_DOSES.json',dose_rows)
 audit=dict(accepted=True,source_files=len(lock),source_hashes_valid=True,saved_artifact_reduction_only=True,new_model_queries=0,all_full_records=3466,main_full_records=3328,main_targets=6656,shadow_targets=c['shadow_targets'],own_TF_pairs=c['own_TF_pairs'],matched_pulses=len(pulse_summary),longitudinal_dose_observations=len(dose_rows),matched_clock_and_nonvalue_checks=True,calibration_frozen=True,output_bytes_before_package=total,reducer_CPU=time.process_time()-T,reducer_sha256=sha(__file__),arithmetic_pointer_modified=False)
 save('FINAL_AUDIT.json',audit);print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
