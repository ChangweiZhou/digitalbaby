"""Independent saved-response reducer; no learner calls or scientific reruns."""
import json,hashlib,math,collections
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H.parent

def load(p):return json.loads(Path(p).read_text())
def check_eval(e):
 rows=e['rows'];n=0;nats=0;correct=0
 for row in rows:
  choices=row['choices'];expected=bytes.fromhex(row['expected_hex'])
  assert len(choices)<=4 and all(0<=y<=256 for y in choices)
  eos=bool(choices and choices[-1]==256);assert not any(y==256 for y in choices[:-1]);assert eos==row['EOS'];assert eos or len(choices)==4
  payload=bytes(choices[:-1] if eos else choices);assert payload.hex()==row['output_hex']
  right=eos and payload==expected;assert right==row['correct'];correct+=right
  tf=row['teacher_forced'];assert [x['y'] for x in tf]==list(expected)+[256]
  n+=len(tf);nats-=sum(x['lp'] for x in tf);assert all(math.isfinite(x['lp']) and x['lp']<=0 for x in tf)
 met=e['metrics'];assert met['records']==len(rows) and met['correct']==correct and met['target_positions']==n
 assert abs(met['accuracy']-correct/len(rows))<1e-14 and abs(met['BPS']-nats/n/math.log(2))<1e-12
 return len(rows),n

def run(summary):
 hashes=load(H/'SOURCE_LOCK.json');bad=[p for p,d in hashes.items() if hashlib.sha256((R/p).read_bytes()).hexdigest()!=d];assert not bad,bad
 counts=collections.Counter();groups=[];unseen=0
 for directory in sorted((H/'groups').iterdir()) if (H/'groups').exists() else []:
  if not (directory/'RECEIPT.json').exists():continue
  receipt=load(directory/'RECEIPT.json');assert receipt['status']=='COMPLETE'
  for file,key in [('RESULT.json','result_sha256'),('ACQUISITION.json','acquisition_sha256')]:assert hashlib.sha256((directory/file).read_bytes()).hexdigest()==receipt[key]
  result=load(directory/'RESULT.json');a=load(directory/'ACQUISITION.json');data=load(directory/'DATA.json');assert not a['sealed_assessment_scored']
  partitions=[data[k] for k in ['OLD','NEW','validation','assessment']];orbits=[tuple(x['orbit']) for rows in partitions for x in rows];assert len(set(orbits))==210
  ids={k:{x['id'] for x in data[k]} for k in ['OLD','NEW','validation','assessment']};assert all(not ids[x]&ids[y] for x in ids for y in ids if x<y)
  train=a['training_records'];assert len(train)==2560==receipt['physical_records']==a['physical_records'];counts['main_records']+=len(train)
  branch_records=collections.Counter(x['branch'] for x in train);assert branch_records=={'W':1024,'SH_OLD':1024,'N_OLD':512}
  write=collections.Counter((x['phase'],x['branch'],x['write']) for x in train);assert write=={('OLD','W',True):512,('OLD','SH_OLD',False):512,('NEW','W',True):512,('NEW','SH_OLD',True):512,('NEW','N_OLD',False):512}
  for row in train:
   assert row['id'] in ids[row['phase']];record=next(z for z in data[row['phase']] if z['id']==row['id']);answer=bytes.fromhex(record['answer']);assert [x['y'] for x in row['targets']]==list(answer)+[256]
   n=row['observed_targets']-len(row['targets'])
   for j,z in enumerate(row['targets']):assert abs(z['warmup']-(1/16+15/16*min((n+j)/1024,1)))<1e-14
   counts['main_targets']+=len(row['targets'])
  for row in a['curves']+a['immediate']+result['held']+result['ablations']:
   q,n=check_eval(row);counts['main_query_pairs']+=q
  assert len(a['immediate'])==1536
  for x in a['curves']:assert not any(y['id'] in ids['assessment'] for y in x['rows'])
  for x in result['held']:
   allowed=ids['validation'] if result['calibration'] else ids['assessment'];assert {z['id'] for z in x['rows']}==allowed
   if not result['calibration']:unseen+=len(x['rows'])
  groups.append(dict(arm=result['arm'],world=result['world'],eta=result['eta'],calibration=result['calibration'],physical_records=len(train)))
 c=load(H/'COUNTS.json')['counts'];assert c['records']<=69280 and c['targets']<=207840 and c['query_pairs']<=116042
 assert c['records']==counts['main_records']+c.get('preflight_records',0)+c.get('credit_records',0)
 assert c['targets']==counts['main_targets']+c.get('preflight_targets',0)+c.get('credit_targets',0)
 assert c['query_pairs']==counts['main_query_pairs']+c.get('preflight_query_pairs',0)+c.get('credit_query_pairs',0)
 if summary['verdict']=='REFERENCE_NOT_READY':
  assert len(groups)==6 and all(x['calibration'] for x in groups) and unseen==0
  rs=[load(p/'RESULT.json') for p in (H/'groups').iterdir() if (p/'RECEIPT.json').exists()];rates={e:np.mean([next(h['metrics']['BPS'] for h in x['held'] if h['branch']=='W') for x in rs if x['eta']==e]) for e in [.01,.03]};eta=min(rates,key=lambda e:(rates[e],e));assert eta==summary['readiness']['selected_eta']
  bp=next(p for p in (H/'groups').iterdir() if load(p/'RESULT.json')['arm']=='R_BP' and load(p/'RESULT.json')['eta']==eta);a=load(bp/'ACQUISITION.json')
  def accuracy(tag,phase=None):
   row=next(x for x in a['curves'] if x['branch']=='W' and x['tag']==tag);rr=row['rows'][64:] if phase=='NEW' else row['rows'];return sum(z['correct'] for z in rr)/len(rr)
  old0,old1,new0,new1=accuracy('INITIAL_OLD'),accuracy('OLD8'),accuracy('BEFORE_NEW'),accuracy('NEW8','NEW')
  assert not ((old1>old0 or old0==1) and (new1>new0 or new0==1))
  assert [old0,old1,new0,new1]==[summary['readiness'][k] for k in ['BP_OLD_before','BP_OLD_after','BP_NEW_before','BP_NEW_after']]
 return dict(status='ACCEPTED',audit_type='independent_saved_response_and_exposure_reduction',source_hashes_checked=len(hashes),groups=groups,counts_recomputed=dict(counts),sealed_assessment_queries=unseen,held_addition_efficacy_claim=False,new_core_adopted=False,training_rerun=False)

if __name__=='__main__':
 audit=run(load(H/'SUMMARY.json'));(H/'FINAL_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({'audit':audit['status'],'groups':len(audit['groups']),'sealed_queries':audit['sealed_assessment_queries']}))
