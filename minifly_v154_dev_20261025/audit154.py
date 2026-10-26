"""Independent saved-response/receipt reducer, no learning or model inference."""
from pathlib import Path
from collections import Counter
import json,hashlib,math,copy
H=Path(__file__).resolve().parent;ROOT=H.parent
load=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_eval(e):
 rows=e['rows'];correct=0;targets=0;nats=0
 for r in rows:
  cc=r['choices'];assert 0<len(cc)<=4 and all(0<=x<=256 for x in cc);assert not any(x==256 for x in cc[:-1]);eos=cc[-1]==256;assert eos==r['EOS'];assert eos or len(cc)==4;out=bytes(cc[:-1] if eos else cc);expected=bytes.fromhex(r['expected_hex']);assert out.hex()==r['output_hex'];right=eos and out==expected;assert right==r['correct'];correct+=right
  tf=r['teacher_forced'];assert [x['y'] for x in tf]==list(expected)+[256];assert all(math.isfinite(x['lp']) and x['lp']<=0 and 0<=x['argmax']<=256 for x in tf);targets+=len(tf);nats-=sum(x['lp'] for x in tf)
  for x in tf:
   if 'context_vs_EOS' in x:assert abs(x['context_vs_EOS']+x['bias_vs_EOS']-x['full_vs_EOS'])<1e-12
 m=e['metrics'];assert m['records']==len(rows) and m['correct']==correct and m['targets']==targets;assert abs(m['BPS']-nats/targets/math.log(2))<1e-12;return len(rows)
def run():
 lock=load(H/'SOURCE_LOCK.json');assert all(sha(ROOT/p)==d for p,d in lock.items());summary=load(H/'SUMMARY.json');tot=Counter();group_out=[];allheld=0
 cfile=H/'local/LANE_C_PREDICTIONS.json'
 if cfile.exists():
  c=load(cfile)
  for entry in c:
   for condition in entry['conditions']:tot['checkpoint_pairs']+=check_eval(condition)
  assert tot['checkpoint_pairs']==1728
 for p in sorted((H/'groups').glob('*/RECEIPT.json')) if (H/'groups').exists() else []:
  d=p.parent;rc=load(p);a=load(d/'ACQUISITION.json');r=load(d/'RESULT.json');data=load(d/'DATA.json')
  assert sha(d/'ACQUISITION.json')==rc['acquisition_sha256'] and sha(d/'RESULT.json')==rc['result_sha256'];assert rc['physical_records']==len(a['training_records'])==2560
  keys={k:{x['id'] for x in data[k]} for k in ['OLD','NEW','validation','assessment']};assert len(set.union(*keys.values()))==210
  assert len({tuple(x['orbit']) for k in keys for x in data[k]})==210
  for phase,branch,write in [('OLD','W',True),('OLD','SH_OLD',False),('NEW','W',True),('NEW','N_OLD',False),('NEW','SH_OLD',True)]:
   for pp in range(1,9):
    rows=[x for x in a['training_records'] if x['phase']==phase and x['branch']==branch and x['pass_number']==pp];assert [x['id'] for x in rows]==[data[phase][i]['id'] for i in data['orders'][phase][pp-1]];assert all(x['write']==write for x in rows)
    for x in rows:
     rec=next(v for v in data[phase] if v['id']==x['id']);ys=list(bytes.fromhex(rec['answer']))+[256];assert [v['y'] for v in x['sites']]==ys;start=x['n']-len(ys)
     for j,v in enumerate(x['sites']):assert abs(v['warmup']-(1/16+15/16*min((start+j)/1024,1)))<1e-14
  tot['native_records']+=2560;tot['native_targets']+=sum(len(x['sites']) for x in a['training_records']);assert len(a['immediate'])==1536
  for e in a['curves']+a['immediate']:
   tot['native_pairs']+=check_eval(e);assert not any(x['id'] in keys['assessment'] for x in e['rows'])
  held=load(d/'HELD.json');alock=load(H/'ACQUISITION_LOCK.json')[d.name];assert alock['sha256']==sha(d/'ACQUISITION.json') and alock['ready']==a['ready']
  for e in held['branches']:
   tot['native_pairs']+=check_eval(e);assert {v['id'] for v in e['rows']}==keys['assessment'];allheld+=len(e['rows'])
  assert bool(held['branches'])==a['ready'];group_out.append(dict(group=d.name,records=2560,ready=a['ready'],immediate_exact=r['immediate_exact']))
  for state in a['states'].values():assert sha(H/state['path'])==state['sha256']
 for p in (H/'local').glob('PROBE_*.json'):
  probe=load(p)
  for e in probe['scoring']:
   rr=e['records'];tf=[z for x in rr for z in x['targets']];bps=-sum(x['lp'] for x in tf)/len(tf)/math.log(2);assert abs(bps-e['BPS'])<1e-12;assert abs(sum(x['argmax']==x['y'] for x in tf)/len(tf)-e['argmax_accuracy'])<1e-14;tot['probe_TF_records']+=len(rr)
 counts=load(H/'COUNTS.json')['counts'];assert counts.get('native_records',0)==tot['native_records'] and counts.get('native_targets',0)==tot['native_targets'];assert counts['pair_queries']==tot['native_pairs']+tot['checkpoint_pairs']+counts['technical_pair_queries'];assert counts['probe_TF_records']==tot['probe_TF_records']
 assert counts['record_applications']==sum(counts.get(x+'_records',0) for x in ['native','feature','online_probe','technical']);assert counts['observed_targets']==sum(counts.get(x+'_targets',0) for x in ['native','feature','online_probe','technical'])
 assert counts['record_applications']<=43328 and counts['observed_targets']<=129984 and counts['pair_queries']<=43676 and counts['batch_evaluations']<=3840
 # Negative saved-output fixtures: no learner calls or extra information exposure.
 negative={}
 if group_out:
  base=load(next((H/'groups').glob('*/ACQUISITION.json')))['curves'][0]
  for kind in ['EOS','CORRECT','CLASS','LOSS']:
   fixture=copy.deepcopy(base)
   if kind=='EOS':fixture['rows'][0]['EOS']=not fixture['rows'][0]['EOS']
   elif kind=='CORRECT':fixture['rows'][0]['correct']=not fixture['rows'][0]['correct']
   elif kind=='CLASS':fixture['rows'][0]['choices'][0]=257
   else:fixture['rows'][0]['teacher_forced'][0]['lp']=float('nan')
   try:check_eval(fixture)
   except (AssertionError,ValueError):negative[kind]='REJECTED'
   else:raise AssertionError('Bad response accepted')
 out=dict(status='ACCEPTED',source_files_checked=len(lock),native_groups=group_out,counts_recomputed=dict(tot),held_questions_scored=allheld,negative_response_fixtures=negative,training_rerun=False,external_readout_is_native=False,new_core_adopted=False)
 (H/'FINAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'audit':'ACCEPTED','groups':len(group_out),'held_questions':allheld}))
if __name__=='__main__':run()
