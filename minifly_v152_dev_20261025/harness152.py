"""Fixed OLD/NEW causal histories, sealed final assessment and paid queries."""
import time,math,collections
import numpy as np
import support152 as s
import models152 as m

def evaluate(model,records,tag,kind='main',ablate=None):
 t=time.process_time();rows=[]
 for rec in records:
  s.query_count(kind);prompt,answer=s.read_record(rec);before_outputs=s.COUNTS['output_distributions'];out=m.generate(model,prompt,ablate=ablate);tf=m.teacher_forced(model,prompt,answer,ablate=ablate);s.COUNTS['query_output_distributions']+=s.COUNTS['output_distributions']-before_outputs
  rows.append(dict(id=rec['id'],expected_hex=answer.hex(),**out,correct=bool(out['EOS'] and out['output_hex']==answer.hex()),teacher_forced=tf))
 n=sum(len(x['teacher_forced']) for x in rows);nats=sum(-y['lp'] for x in rows for y in x['teacher_forced']);metrics=dict(records=len(rows),correct=sum(x['correct'] for x in rows),accuracy=sum(x['correct'] for x in rows)/len(rows),target_positions=n,BPS=nats/n/math.log(2))
 s.TIMES['evaluation_CPU']+=time.process_time()-t
 return dict(tag=tag,metrics=metrics,rows=rows)

def group(arm,world,eta,calibration=False):
 name=f'{arm}_{world}_{eta:g}';dest=s.H/'groups'/name
 assert not dest.exists(),f'No automatic rerun of {name}'
 dest.mkdir(parents=True);data=s.dataset(world);s.save(f'groups/{name}/DATA.json',data)
 kinds='calibration' if calibration else 'main';begin=s.cpu();initial=m.make(arm,world,eta);w=initial;sh=initial.clone();curves=[];immediate=[];trainrows=[];training_cpu=0.;committed_records=0
 def ev(model,records,tag,branch):
  result=evaluate(model,records,tag,kinds);row=dict(branch=branch,training_CPU=training_cpu,observed_targets=model.n,**result);curves.append(row);return row
 def state(model,label):return s.checkpoint(model,f'{name}_{label}.pkl')
 def physical_phase(model,phase,branch,write):
  nonlocal training_cpu,committed_records
  for p in range(1,9):
   for index in data['orders'][phase][p-1]:
    s.ACTIVE_STATE=dict(group=name,model=model,trainrows=trainrows,curves=curves,immediate=immediate,physical_records=committed_records,branch=branch,phase=phase,pass_number=p);s.budget();rec=data[phase][index];prompt,answer=s.read_record(rec);s.target_record(answer,kinds,write=write,prompt=prompt);t=time.process_time();sites=model.train(prompt,answer,write=write);training_cpu+=time.process_time()-t;committed_records+=1
    trainrows.append(dict(branch=branch,phase=phase,pass_number=p,id=rec['id'],write=write,targets=sites,observed_targets=model.n))
    if write:
     probe=evaluate(model,[rec],f'{phase}{p}_immediate',kinds);immediate.append(dict(branch=branch,phase=phase,pass_number=p,observed_targets=model.n,**probe))
   s.log('PASS_COMMIT',group=name,branch=branch,phase=phase,pass_number=p,records=committed_records,targets=model.n)
   s.save('ACTIVE.json',dict(stage='CALIBRATION' if calibration else 'ASSESSMENT',group=name,branch=branch,phase=phase,pass_number=p,committed_physical_records=committed_records,counts=dict(s.COUNTS),CPU=s.cpu()))
   if p in [1,2,4,8]:
    if phase=='OLD' and branch=='W':ev(model,data['OLD'],f'OLD{p}','W')
    if phase=='NEW':
     ev(model,data['OLD']+data['NEW'],f'NEW{p}',branch)
    if arm=='R_LE' and not calibration and branch=='W' and p in [1,8]:s.save(f'groups/{name}/{phase}{p}_SNAPSHOT.json',state(model,f'{phase}{p}'))
  return model
 baseline_old=ev(w,data['OLD'],'INITIAL_OLD','W')
 physical_phase(w,'OLD','W',True);physical_phase(sh,'OLD','SH_OLD',False);no_prior=ev(sh,data['OLD'],'OLD8_NO_PRIOR','SH_OLD')
 no=w.clone();models={'W':w,'N_OLD':no,'SH_OLD':sh};states={'OLD_W':state(w,'OLD_W'),'OLD_SH':state(sh,'OLD_SH')}
 for branch,model in models.items():ev(model,data['NEW'],'BEFORE_NEW',branch)
 for branch,model in models.items():physical_phase(model,'NEW',branch,branch!='N_OLD')
 for branch,model in models.items():
  model.rest(128);ev(model,data['OLD']+data['NEW'],'POST_IDLE',branch);states[branch]=state(model,'FINAL_'+branch)
 # Freeze acquisition before any assessment labels are opened by scorer.
 s.save(f'groups/{name}/ACQUISITION.json',dict(arm=arm,world=world,eta=eta,curves=curves,immediate=immediate,training_records=trainrows,states=states,training_CPU=training_cpu,physical_records=committed_records,sealed_assessment_scored=False))
 s.log('ACQUISITION_FROZEN',group=name,sha256=s.sha(dest/'ACQUISITION.json'))
 held=[];ablations=[]
 for branch,model in models.items():held.append(dict(branch=branch,**evaluate(model,data['validation'] if calibration else data['assessment'],'VALIDATION' if calibration else 'SEALED_ASSESSMENT',kinds)))
 if arm=='N_2B_Q':
  for branch,model in models.items():
   for ablate in ['no_native','no_q']:ablations.append(dict(branch=branch,ablation=ablate,**evaluate(model,data['OLD']+data['NEW']+data['assessment'],'FINAL_ABLATION','hybrid',ablate)))
 summary=dict(arm=arm,world=world,eta=eta,calibration=calibration,physical_records=committed_records,observed_target_applications=sum(len(x['targets']) for x in trainrows),training_CPU=training_cpu,total_CPU=s.cpu()-begin,curves=[{k:v for k,v in x.items() if k not in ['rows']} for x in curves],held=held,ablations=ablations,states=states)
 s.save(f'groups/{name}/RESULT.json',summary);s.save(f'groups/{name}/RECEIPT.json',dict(status='COMPLETE',result_sha256=s.sha(dest/'RESULT.json'),acquisition_sha256=s.sha(dest/'ACQUISITION.json'),physical_records=committed_records))
 s.log('GROUP_COMMIT',group=name,physical_records=committed_records,result_sha256=s.sha(dest/'RESULT.json'))
 return summary

def readiness(results):
 rates=[]
 for eta in [.01,.03]:
  r=[x for x in results if x['eta']==eta];score=np.mean([next(z['metrics']['BPS'] for z in x['held'] if z['branch']=='W') for x in r]);rates.append((float(score),eta))
 score,eta=min(rates);bp=next(x for x in results if x['eta']==eta and x['arm']=='R_BP');c=bp['curves']
 def cell(tag):return next(x['metrics']['accuracy'] for x in c if x['tag']==tag and x['branch']=='W')
 old_before,old_after=cell('INITIAL_OLD'),cell('OLD8');new_before=cell('BEFORE_NEW')
 row=next(x for x in c if x['tag']=='NEW8' and x['branch']=='W');new_after=sum(y['correct'] for y in s_load_acq_new(bp))/64
 ok_old=old_after>old_before or old_before==1;ok_new=new_after>new_before or new_before==1
 return dict(status='READY' if ok_old and ok_new else 'REFERENCE_NOT_READY',selected_eta=eta,validation_mean_BPS=score,rate_scores=[dict(eta=e,BPS=z) for z,e in rates],BP_OLD_before=old_before,BP_OLD_after=old_after,BP_NEW_before=new_before,BP_NEW_after=new_after,OLD_ready=ok_old,NEW_ready=ok_new)
def s_load_acq_new(bp):
 import json
 name=f"{bp['arm']}_{bp['world']}_{bp['eta']:g}";a=json.loads((s.H/'groups'/name/'ACQUISITION.json').read_text());row=next(x for x in a['curves'] if x['tag']=='NEW8' and x['branch']=='W');return row['rows'][64:]
