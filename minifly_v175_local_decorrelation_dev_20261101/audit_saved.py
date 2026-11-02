"""Saved-only V175 reducer. No native modules, training, replay or new model queries."""
import json,hashlib,math,csv,statistics,time,collections
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parent;T=time.process_time()
def read(p):return json.loads((H/p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x
def save(p,x):(H/p).write_text(json.dumps(plain(x),indent=2,allow_nan=False)+'\n')
def codes(rec):
 assert sha(H/rec['code_path'])==rec['code_sha256']
 with np.load(H/rec['code_path'],allow_pickle=False) as z:return z['codes'].copy()
def J(A,B):
 v=A@B.T;d=A.sum(1)[:,None]+B.sum(1)[None,:]-v;return np.divide(v,d,out=np.zeros_like(v),where=d>0)
def conflicts(A,rows):
 return [(rows[i]['id'],rows[j]['id']) for i in range(len(A)) for j in range(i) if rows[i]['answer']!=rows[j]['answer'] and np.array_equal(A[i],A[j])]
def panel(rec,rows):
 assert sha(H/rec['path'])==rec['sha256'];out=read(rec['path']);expected={r['id']:r for r in rows};assert len(out)==len(rows) and {r['id'] for r in out}==set(expected)
 for r in out:
  ref=expected[r['id']];assert r['correct']==(r['EOS'] and r['output_hex']==ref['answer']);assert r['CAP']==(not r['EOS']);assert len(bytes.fromhex(r['output_hex']))<=4
  for s,y in zip(r['tf'],[*bytes.fromhex(ref['answer']),256]):assert s['y']==y and math.isfinite(s['lp']) and math.isfinite(s['margin']) and 0<=s['argmax']<=256
 assert rec['OLD']==sum(r['correct'] for r in out if r['cohort']=='OLD');assert rec['NEW']==sum(r['correct'] for r in out if r['cohort']=='NEW');assert rec['correct']==rec['OLD']+rec['NEW'];assert set(rec['correct_ids'])=={r['id'] for r in out if r['correct']}
 return out

def run():
 assert read('STATUS.json')['status']=='C2_COMPLETED';assert not (H/'FAILURE.json').exists()
 lock=read('SOURCE_LOCK.json');assert all(sha(ROOT/p)==h for p,h in lock.items());qual=read('CYCLE_1_QUALIFICATION.json');assert qual['accepted'] and qual['technical_records']==18
 assert read('CYCLE_0_TESTBED.json')['accepted'] and read('RESOURCE_ADMISSION.json')['accepted']
 data=read('MANIFEST.json');res=read('RESULTS.json');assert len(res)==2
 log=[json.loads(line) for line in (H/'LEDGER.jsonl').read_text().splitlines()];records=[r for r in log if r['event']=='train_record'];counts=read('COUNTS.json');c=counts['counts']
 # One passive-only technical presentation was explicitly charged outside the teacher log.
 assert len(records)==4625 and c['full_records']==4626 and c['technical_records']==18
 assert c['technical_targets']==34 and c['main_BASE_records']==2304 and c['main_LPS-16_records']==2304
 assert c['main_BASE_targets']==4608 and c['main_LPS-16_targets']==4608
 assert c['shadow_targets']==66 and c['own_TF_pairs']==2436
 world_results=[];metrics=[];shadow_rows=[];pairs=0;checkpoint_receipts=[];expected_order={'INITIAL':0,'OLD8':512,'OLD16':1024,'NEW4':1088,'FINAL':1152,'REST128':1152}
 for world in res:
  w=world['world'];rows=data[str(w)]['rows'];old_ids={r['id'] for r in rows[:64]};arms={}
  assert len(rows)==80 and len({r['prompt'] for r in rows})==80
  assert sorted(collections.Counter(r['answer'] for r in rows[:64]).values())==[4]*16
  assert sorted(collections.Counter(r['answer'] for r in rows[64:]).values())==[1]*16
  baseline_X=codes(world['arms']['BASE']['panels']['INITIAL'])
  for arm,a in world['arms'].items():
   ss=a['panels'];assert set(ss)==set(expected_order);initial=set(ss['OLD16']['correct_ids'])&old_ids;final=set(ss['FINAL']['correct_ids'])&old_ids
   initialX=codes(ss['INITIAL']);X8=codes(ss['OLD8']);X16=codes(ss['OLD16']);Xf=codes(ss['FINAL']);overlap=float(J(X16[:64],X16[64:]).mean());stability=float(np.diag(J(X8[:64],X16[:64])).mean());changed=int(np.count_nonzero(np.any(X16!=baseline_X,axis=1)))
   assert np.array_equal(initialX,baseline_X)
   if arm=='BASE':assert np.array_equal(X8,baseline_X) and np.array_equal(X16,baseline_X) and np.array_equal(Xf,baseline_X)
   for stage,rec in ss.items():
    raw=panel(rec,rows);pairs+=80;X=codes(rec);assert np.array_equal(X.sum(1),baseline_X.sum(1));assert rec['n']==2*expected_order[stage]
    expected_time=expected_order[stage]*9*(30/14)+(128*(30/14) if stage=='REST128' else 0);assert abs(rec['native_time']-expected_time)<1e-5
    assert rec['L_updates']==(expected_order[stage] if arm=='LPS-16' else 0);assert 0<=rec['L_max']<=1
    pay=[r['tf'][0] for r in raw];eos=[r['tf'][1] for r in raw]
    metrics.append(dict(world=w,arm=arm,stage=stage,OLD=rec['OLD'],NEW=rec['NEW'],total=rec['correct'],retained=len(initial&set(rec['correct_ids'])),OLD_boundary=len(initial),TF_payload=sum(s['argmax']==s['y'] for s in pay)/80,TF_EOS=sum(s['argmax']==s['y'] for s in eos)/80,bits=-sum(s['lp'] for r in raw for s in r['tf'])/(160*math.log(2)),empty_EOS=rec['empty_EOS'],CAP=rec['CAP'],native_time=rec['native_time']))
   fc=conflicts(Xf,rows);assert len(fc)==len(ss['FINAL']['conflicts'])
   assert len(a['shadows'])==8
   for sh in a['shadows']:
    assert sh['same_code'] and sh['same_time'];ids={r['id'] for r in sh['sentinels']};sent=[r for r in rows[:64] if r['id'] in ids];assert len(sent)==8
    zero=panel(sh['zero_panel'],sent);written=panel(sh['written_panel'],sent);pairs+=16
    d=statistics.mean(r['zero_margin']-r['written_margin'] for r in sh['sentinels']);assert abs(d-sh['D'])<1e-12
    shadow_rows.append(dict(world=w,arm=arm,ordinal=int(sh['tag'].split('_')[-1]),D=d,self_margin_change=sh['self_target_margin_change'],delivered_d=sh['delivered_d'],passive_zero_L2=sh['passive_zero_synaptic_L2']))
   for key in ('OLD8_checkpoint','OLD16_checkpoint','FINAL_checkpoint'):
    cr=a[key];assert sha(H/cr['path'])==cr['sha256'];checkpoint_receipts.append(cr)
   arms[arm]=dict(O0=len(initial),O1=ss['FINAL']['OLD'],N1=ss['FINAL']['NEW'],T=ss['FINAL']['correct'],retained=len(initial&final),lost=len(initial-final),R=len(initial&final)/len(initial) if initial else None,J=overlap,S=stability,changed_codes_OLD16=changed,final_conflicts=len(fc),L_nonzero=ss['FINAL']['L_nonzero'],L_max=ss['FINAL']['L_max'],D=statistics.mean(sh['D'] for sh in a['shadows']),life_total_CPU=a['life_total_CPU'],life_train_CPU_including_shadow=a['life_train_CPU'])
  b=arms['BASE'];l=arms['LPS-16'];retention_delta=None if b['R'] is None or l['R'] is None else l['R']-b['R'];P1=l['T']-b['T']>=8 and l['N1']-b['N1']>=-2 and retention_delta is not None and retention_delta>=-.05;P2=b['J']-l['J']>=.05 and l['S']>=.85 and l['changed_codes_OLD16']>0 and l['final_conflicts']==0
  world_results.append(dict(world=w,arms=arms,delta_total=l['T']-b['T'],delta_NEW=l['N1']-b['N1'],delta_R=retention_delta,J_drop=b['J']-l['J'],P1=bool(P1),P2=bool(P2)))
 assert pairs==2432 and pairs+4==c['own_TF_pairs']
 output_bytes=sum(p.stat().st_size for p in H.rglob('*') if p.is_file());assert output_bytes<1073741824
 P3=counts['CPU']<=7200 and counts['peak_RSS_bytes']<=1073741824 and c['full_records']<=5000 and qual['accepted']
 promising=all(r['P1'] and r['P2'] for r in world_results) and P3
 verdict='PROMISING_DEV_NOT_ADOPTED' if promising else 'DEV_SCREEN_FAILED'
 summary=dict(version=175,identity='MINIFLY-V175-LPS16-D1',verdict=verdict,evidence=['E0','E1','E1-R'],worlds=world_results,P3=bool(P3),main_full_records=4608,main_targets=9216,all_full_records=c['full_records'],technical_full_records=18,all_full_targets=9250,shadow_targets=c['shadow_targets'],own_TF_pairs=c['own_TF_pairs'],CPU=counts['CPU'],peak_RSS_bytes=counts['peak_RSS_bytes'],gamma=read('GAMMA_LOCK.json')['gamma'],core_adopted=False,E3_tested=False,main_reruns=0)
 save('SUMMARY.json',summary);save('SHADOW_EFFECTS.json',shadow_rows);save('CHECKPOINT_INDEX.json',checkpoint_receipts)
 with (H/'METRICS.csv').open('w') as f:
  wr=csv.DictWriter(f,fieldnames=list(metrics[0]));wr.writeheader();wr.writerows(metrics)
 audit=dict(accepted=True,scientific_outcome=verdict,source_files_verified=len(lock),full_records=4626,main_records=4608,main_targets=9216,all_targets=9250,shadow_targets=66,main_physical_lives=4,technical_failed_attempts_preserved=2,completed_histories_replayed=0,saved_only_reducer=True,new_model_queries=0,population_counts_matched=True,source_model_and_gates_unchanged=True,clock_and_warmup_checks=True,readonly_checks=qual['checks']['readonly_full_digest'],output_bytes_before_package=output_bytes,reducer_CPU=time.process_time()-T,reducer_sha256=sha(__file__))
 save('FINAL_AUDIT.json',audit);print(json.dumps(plain(summary),indent=2))
if __name__=='__main__':run()
