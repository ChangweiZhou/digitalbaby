"""Native W/SH_OLD/N_OLD learning; held scoring separated until global freeze."""
import time
import support154 as s
import native154 as n

def run_group(config,data):
 world=data['world'];name=f"{config['id']}_{world}";dest=s.H/'groups'/name;assert not dest.exists();dest.mkdir(parents=True)
 s.save(f'groups/{name}/DATA.json',data);w=n.Native(config['teacher'],config['front']);sh=w.clone();curves=[];immediate=[];train=[];training_cpu=0.;records=0
 def ev(model,rr,tag,branch):
  out=n.evaluate(model,rr);out.update(tag=tag,branch=branch,observed_targets=model.n,training_CPU=training_cpu);curves.append(out);return out
 def physical(model,phase,branch,write):
  nonlocal training_cpu,records
  for p in range(1,9):
   for index in data['orders'][phase][p-1]:
    s.ACTIVE_STATE=dict(group=name,model=model,curves=curves,immediate=immediate,train=train,records=records,phase=phase,branch=branch,pass_number=p)
    s.budget();row=data[phase][index];prompt,answer=s.unpack(row);s.record(prompt,answer,'native',write);t=time.process_time();sites=model.train(prompt,answer,write);training_cpu+=time.process_time()-t;records+=1
    train.append(dict(id=row['id'],phase=phase,branch=branch,pass_number=p,write=write,sites=sites,n=model.n))
    if write:
     out=n.evaluate(model,[row]);out.update(phase=phase,branch=branch,pass_number=p,n=model.n);immediate.append(out)
   s.log('NATIVE_PASS_COMMIT',group=name,branch=branch,phase=phase,pass_number=p,physical_records=records,n=model.n)
   s.save('ACTIVE.json',dict(stage='NATIVE',group=name,phase=phase,branch=branch,pass_number=p,physical_records=records,CPU=s.cpu(),counts=dict(s.COUNTS)))
   if p in [1,2,4,8]:
    if phase=='OLD' and branch=='W':ev(model,data['OLD'],f'OLD{p}','W')
    if phase=='NEW':ev(model,data['OLD']+data['NEW'],f'NEW{p}',branch)
  return model
 ev(w,data['OLD'],'INITIAL_OLD','W');physical(w,'OLD','W',True);physical(sh,'OLD','SH_OLD',False);ev(sh,data['OLD'],'NO_PRIOR_OLD','SH_OLD')
 states={'OLD_W':s.checkpoint(w,f'{name}_OLD_W.pkl'),'OLD_SH':s.checkpoint(sh,f'{name}_OLD_SH.pkl')};no=w.clone();models={'W':w,'SH_OLD':sh,'N_OLD':no}
 for branch,model in models.items():ev(model,data['NEW'],'BEFORE_NEW',branch)
 for branch,model in models.items():physical(model,'NEW',branch,branch!='N_OLD')
 for branch,model in models.items():model.rest(128);ev(model,data['OLD']+data['NEW'],'POST_IDLE',branch);states[branch]=s.checkpoint(model,f'{name}_FINAL_{branch}.pkl')
 def count(tag,part=None):
  row=next(c for c in curves if c['tag']==tag and c['branch']=='W');rr=row['rows'][64:] if part=='NEW' else row['rows'];return sum(x['correct'] for x in rr)
 old0,old1,new0,new1=count('INITIAL_OLD'),count('OLD8'),count('BEFORE_NEW'),count('NEW8','NEW');ready=(old1>old0 or old0==64) and (new1>new0 or new0==64)
 acq=dict(version='V154',config=config,world=world,physical_records=records,targets=sum(len(x['sites']) for x in train),training_CPU=training_cpu,curves=curves,immediate=immediate,training_records=train,states=states,ready=ready,readiness_counts=dict(OLD_before=old0,OLD_after=old1,NEW_before=new0,NEW_after=new1),assessment_scored=False)
 s.save(f'groups/{name}/ACQUISITION.json',acq)
 result={k:v for k,v in acq.items() if k not in ['curves','immediate','training_records']};result['curves']=[{k:v for k,v in c.items() if k!='rows'} for c in curves];result['immediate_exact']=sum(x['metrics']['correct'] for x in immediate);result['immediate_probes']=len(immediate)
 s.save(f'groups/{name}/RESULT.json',result);s.save(f'groups/{name}/RECEIPT.json',dict(status='ACQUISITION_COMPLETE',acquisition_sha256=s.sha(dest/'ACQUISITION.json'),result_sha256=s.sha(dest/'RESULT.json'),physical_records=records,targets=acq['targets']))
 s.log('NATIVE_GROUP_COMMIT',group=name,ready=ready,physical_records=records,targets=acq['targets']);return result

def score_held(groups,datasets):
 import pickle
 # All acquisition files and readiness decisions are immutable before any held score.
 lock={f"{x['config']['id']}_{x['world']}":dict(ready=x['ready'],sha256=s.sha(s.H/'groups'/f"{x['config']['id']}_{x['world']}"/'ACQUISITION.json')) for x in groups};s.save('ACQUISITION_LOCK.json',lock);s.log('GLOBAL_ACQUISITION_FROZEN',groups=len(groups),ready=sum(x['ready'] for x in groups))
 out=[]
 for result in groups:
  name=f"{result['config']['id']}_{result['world']}";rows=[]
  if result['ready']:
   for branch in ['W','SH_OLD','N_OLD']:
    with (s.H/result['states'][branch]['path']).open('rb') as f:model=pickle.load(f)
    ev=n.evaluate(model,datasets[result['world']]['assessment']);ev['branch']=branch;rows.append(ev)
  countermodels=countermodel_scores(s.load(s.H/'groups'/name/'ACQUISITION.json'),datasets[result['world']],result['ready']);s.save(f'groups/{name}/COUNTERMODELS.json',countermodels)
  s.save(f'groups/{name}/HELD.json',dict(status='SCORED' if result['ready'] else 'NOT_ADMITTED',branches=rows,admission_locked_before_scoring=True));out.append(dict(group=name,ready=result['ready'],branches=[dict(branch=x['branch'],metrics=x['metrics']) for x in rows]))
 return out

def countermodel_scores(acq,data,ready):
 from collections import Counter
 by={x['id']:x for k in ['OLD','NEW'] for x in data[k]};out=[]
 for branch in ['W','SH_OLD','N_OLD']:
  histories=[r for r in acq['training_records'] if r['write'] and (branch=='W' and r['branch']=='W' or branch=='SH_OLD' and r['branch']=='SH_OLD' or branch=='N_OLD' and r['branch']=='W' and r['phase']=='OLD')]
  raw={};canonical={};favorite=Counter();positions=[Counter() for _ in range(4)]
  for row in histories:
   r=by[row['id']];answer=bytes.fromhex(r['answer']);raw[r['prompt']]=answer;canonical[tuple(r['orbit'])]=answer;favorite[answer]+=1
   for i,y in enumerate(list(answer)+[256]):positions[i][y]+=1
  fav=min(favorite,key=lambda x:(-favorite[x],x)) if favorite else b'';predpos=[]
  for i in range(4):
   yy=min(positions[i],key=lambda x:(-positions[i][x],x)) if positions[i] else 256;predpos.append(yy)
   if yy==256:break
  scores={key:[] for key in ['RAW','CANONICAL','FAVORITE','POSITION']}
  if ready:
   for r in data['assessment']:
    assert r['prompt'] not in raw and tuple(r['orbit']) not in canonical
    for key,choices in [('RAW',list(raw.get(r['prompt'],fav))+[256]),('CANONICAL',list(canonical.get(tuple(r['orbit']),fav))+[256]),('FAVORITE',list(fav)+[256]),('POSITION',predpos)]:
     eos=choices[-1]==256;answer=bytes(choices[:-1] if eos else choices);scores[key].append(dict(id=r['id'],output_hex=answer.hex(),EOS=eos,correct=eos and answer.hex()==r['answer']));s.COUNTS['countermodel_record_evaluations']+=1
  out.append(dict(branch=branch,learned_record_presentations=len(histories),unique_raw_keys=len(raw),unique_canonical_keys=len(canonical),scores={key:dict(records=len(rr),correct=sum(x['correct'] for x in rr),rows=rr) for key,rr in scores.items()}))
 return dict(status='SCORED' if ready else 'FITTED_NO_HELD_ADMISSION',branch_results=out,trained_only_from_logged_updating_presentations=True)
