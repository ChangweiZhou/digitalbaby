"""Independent saved-response/exposure audit. No model queries or trajectory replay."""
import json
import numpy as np
import support157 as s
def score(choices,answer):
 payload=[];eos=False
 for value in choices:
  if value<0:break
  assert 0<=value<=256
  if value==256:eos=True;break
  payload.append(int(value))
 return eos and bytes(payload)==answer,eos,bytes(payload).hex()
def panel(receipt,rows):
 p=s.H/receipt['path'];assert s.sha(p)==receipt['sha256']
 with np.load(p,allow_pickle=False) as a:z={k:a[k] for k in a.files}
 assert len(rows)==len(z['correct'])==receipt['records']
 assert z['z'].shape==z['lp'].shape==(len(z['y']),257)
 logits=z['z'];zz=logits-logits.max(1,keepdims=True);expected=zz-np.log(np.exp(zz).sum(1,keepdims=True))
 assert np.max(abs(z['lp']-expected))<1e-11
 mask=logits.copy();mask[np.arange(len(mask)),z['y']]=-np.inf;rival=mask.argmax(1);margin=logits[np.arange(len(mask)),z['y']]-logits[np.arange(len(mask)),rival]
 assert np.array_equal(rival,z['rival']) and np.array_equal(margin,z['margin'])
 for i,r in enumerate(rows):
  lo,hi=z['offsets'][i:i+2];answer=bytes.fromhex(r['answer']);assert list(z['y'][lo:hi])==list(answer)+[256] and str(z['ids'][i])==r['id']
  correct,eos,_=score(z['choices'][i],answer);assert correct==bool(z['correct'][i]) and eos==bool(z['EOS'][i])
 assert int(z['correct'].sum())==receipt['correct'];return z
def trace(path,rows,expected_ids,write,n0=0):
 by={r['id']:r for r in rows};rr=[json.loads(x) for x in (s.H/path).read_text().splitlines()];assert [x['id'] for x in rr]==expected_ids
 targets=0
 for record in rr:
  r=by[record['id']];ys=list(bytes.fromhex(r['answer']))+[256];assert [x['y'] for x in record['sites']]==ys and record['write']==write
  for offset,site in enumerate(record['sites']):assert abs(site['warmup']-(1/16+15/16*min((n0+offset)/1024,1)))<1e-14
  n0+=len(ys);targets+=len(ys);assert record['n']==n0
 return dict(records=len(rr),targets=targets,final_n=n0)
def world(d,results):
 output=[];roots={}
 for prior in ['W','SH_OLD']:
  root=s.load(s.H/f'roots/{d["world"]}_{prior}.json');order=[d['OLD'][i]['id'] for one in d['OLD_orders'] for i in one];tt=trace(root['trace'],d['OLD'],order,prior=='W');assert tt['final_n']==root['n'];pa=panel(root['panel'],d['panel']);assert np.flatnonzero(pa['correct'][:64]).tolist()==root['initial_correct_OLD'];roots[prior]=root
 for result in results:
  key=result['case'];schedule=result['schedule'];prior=result['prior'];root=roots[prior];tt=trace(result['trace'],d['NEW'],d['schedules'][schedule],True,root['n']);assert tt['final_n']==result['final_n']
  assert len(result['immediate'])==128
  for e,rid in zip(result['immediate'],d['schedules'][schedule]):
   row=next(r for r in d['NEW'] if r['id']==rid);correct,eos,body=score(e['choices'],bytes.fromhex(row['answer']));assert e['correct']==correct and e['EOS']==eos and e['output_hex']==body
  for checkpoint in result['boundaries']:
   aa=panel(checkpoint['panel'],d['panel']);assert sum(checkpoint['exposure'].values())==checkpoint['index'] and len(checkpoint['exposure'])==checkpoint['unique_taught'];assert checkpoint['OLD_retained']==sum(bool(aa['correct'][i]) for i in root['initial_correct_OLD'])
  for pr in result['probes']:
   aa=panel(pr['actual'],d['panel']);zero=panel(pr['inhibited'],d['panel']);idle=panel(pr['idle'],d['panel'])
   assert abs(float((aa['margin']-zero['margin']).mean())-pr['write_mean_margin'])<1e-12
   assert abs(float((zero['margin']-idle['margin']).mean())-pr['observed_mean_margin'])<1e-12
   assert int(np.sum(zero['correct']&~aa['correct']))==pr['write_complete_losses']
   row=next(r for r in d['NEW'] if r['id']==pr['id']);co,en,body=score(pr['parent_current']['choices'],bytes.fromhex(row['answer']));assert co==pr['parent_current']['correct'] and en==pr['parent_current']['EOS']
  panel(result['delayed_panel'],d['panel'])
  for state in ['final','delayed']:assert s.sha(s.H/result[state]['path'])==result[state]['sha256']
  output.append(dict(case=key,status='ACCEPTED',NEW_records=128,NEW_targets=tt['targets'],main_n_preserved=True,source_training_replayed=False))
 audit=dict(status='ACCEPTED',world=d['world'],cases=output,held_scores_opened=False,scope='taught histories, immutable output arrays and source counter checks')
 s.save(f'worlds/{d["world"]}/TAUGHT_AUDIT.json',audit);return audit
def negative_fixtures():
 assert score([0,255,256],bytes([0,255]))[0]
 assert not score([0,254,256],bytes([0,255]))[0]
 assert not score([0,255,1,2],bytes([0,255]))[0]
 return dict(correct='ACCEPTED',mutated_answer='REJECTED',missing_EOS='REJECTED')
