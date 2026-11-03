"""The fixed 4,480-record V181 official run; one pass through each frozen life."""
from common import *
import copy,io
import learner as L
import evaluator as E
from sensor import Sensor
from data import prepare,corrected
import metrics as M
T=L.T;np=L.np
ARMS=['NORMAL','FIXED-LOW','RMS-SHAM','CGLP']

def execute():
 source_check();assert load(CAM/'AUTHORIZATION.json')['user_instruction']=='Run it'
 with (CAM/'RUN_STARTED').open('x') as f:f.write(str(time.time()))
 sensor=Sensor();data,meta=prepare(sensor);assert meta==load(DEV/'DATA_AUDIT.json')['worlds']
 save(CAM/'DATA_AUDIT.json',{'status':'PASS','worlds':meta,'sensor_sha256':sensor.digest})
 manifest={'worlds':{},'scheduled_science_records':4480,'scheduled_science_targets':227148,'record_order':'original chronological; fixed repeated passes; C original ID retained'}
 for w,sp in data.items():
  manifest['worlds'][w]={name:{'record_sha256':[__import__('hashlib').sha256(r).hexdigest() for r in rs],'record_lengths':[len(r) for r in rs],'targets':sum(len(r)+1 for r in rs)} for name,rs in sp.items()}
 save(CAM/'MANIFEST.json',manifest)
 result={};counter={'science_records':0,'science_targets':0,'technical_records_carried':32,'technical_targets_carried':1064,'committed_jobs':0,'checkpoint_count':0};receipts=[];panels={};birth={}
 save(CAM/'COUNTERS.json',counter);save(CAM/'STATUS.json',{'status':'RUNNING','stage':'COUNT_CONTROLS'})
 save(CAM/'COUNT_CONTROLS.json',M.counts(data))
 for w,sp0 in data.items():
  sp=dict(sp0);rulemeta=meta[w]['correction'];rule=(bytes.fromhex(rulemeta['context_hex']),rulemeta['old'],rulemeta['new'])
  for split in ('calib','held'):sp['EDIT_'+split]=[corrected(r,rule) for r in sp['A_'+split]]
  packed={}
  for name,rs in sp.items():
   save(CAM/'ACTIVE.json',{'stage':'SENSORY_CACHE','world':w,'split':name,**check()});packed[name]=E.pack(sensor,rs)
  original_C_ids=rulemeta['selected_train_ids'];result[w]={}
  for arm in ARMS:
   net=L.Learner(w);job=f'{w}_{arm}';jobdir=CAM/'jobs'/job;jobdir.mkdir(parents=True,exist_ok=False);local={'records':0,'targets':0};saved={};all_updates=[];check()
   h=E.state_hash(net)
   if w in birth:assert birth[w]==h
   else:birth[w]=h
   def snapshot(stage):
    path=jobdir/(stage+'.pt');T.save(net.state(),path);assert path.stat().st_size<=1048576
    # A hash of all mutable values before/after serialization; moments/RNG included.
    restored=T.load(path,weights_only=True);probe=L.Learner(w);probe.restore(restored);assert E.state_hash(probe)==E.state_hash(net)
    info={'stage':stage,'state_sha256':E.state_hash(net),'checkpoint':str(path.relative_to(CAM)),'checkpoint_sha256':sha(path),'checkpoint_bytes':path.stat().st_size,'clock':net.clock,'local_records':local['records'],'local_targets':local['targets'],'metrics':{}}
    for name,rs in sp.items():
     out,site=M.score(net,rs,packed[name],sp['A_held'] if name=='EDIT_held' else None)
     if name=='EDIT_held':out['partitions']=M.corrected_partitions(rs,sp['A_held'],packed[name],packed['A_held'],site)
     info['metrics'][name]=out
    if stage in ('A2','B3','EDIT2'):info['generation']=M.generate(net,sensor,sp['A_held'][:16]+sp['B_held'][:16])
    save(jobdir/(stage+'.json'),info);saved[stage]=info;counter['checkpoint_count']+=1;save(CAM/'COUNTERS.json',counter);check()
   def train_pass(role,stage):
    rs=sp[role+'_train'];rows=[];ys=[];ids=[];ends=[];updates=[]
    sourceids=original_C_ids if role in ('EDIT','NOEDIT') else list(range(len(rs)))
    ids_role='A' if role in ('EDIT','NOEDIT') else role
    def flush():
     if not ys:return
     assert len(ys)<=256 and counter['science_targets']+len(ys)+1064<=600000
     x=T.zeros((len(ys),5177))
     for i,row in enumerate(rows):x[i,T.tensor(row.astype(np.int64))]=1
     info=net.step(x,T.tensor(ys),ids,arm);updates.append(info);counter['science_targets']+=len(ys);counter['science_records']+=sum(ends);local['records']+=sum(ends);local['targets']+=len(ys)
     assert counter['science_records']+32<=5000
     rows.clear();ys.clear();ids.clear();ends.clear();save(CAM/'COUNTERS.json',counter);save(CAM/'ACTIVE.json',{'stage':'TRAINING','world':w,'arm':arm,'phase':stage,'clock':net.clock,'local':local.copy(),**check()})
    for ri,(r,code) in enumerate(zip(rs,packed[role+'_train'])):
     for pos,(y,row) in enumerate(zip([*r,256],code)):
      rows.append(row);ys.append(y);ids.append(f'V181|{w}|{ids_role}|{sourceids[ri]}');ends.append(pos==len(r))
      if len(ys)==256:flush()
    flush();all_updates.extend(updates);snapshot(stage)
    save(jobdir/(stage+'_PASS_RECEIPT.json'),{'status':'ACCEPTED_COMPLETE_PASS','stage':stage,'records':len(rs),'targets':sum(len(r)+1 for r in rs),'checkpoint_sha256':saved[stage]['checkpoint_sha256'],'metrics_sha256':sha(jobdir/(stage+'.json')),'source_lock_sha256':sha(CAM/'SOURCE_LOCK.json')})
   snapshot('birth')
   for i in (1,2):train_pass('A',f'A{i}')
   for i in (1,2,3):train_pass('B',f'B{i}')
   bstate=copy.deepcopy(net.state());bhash=E.state_hash(net)
   for role in ('EDIT','NOEDIT'):
    net.restore(bstate);assert E.state_hash(net)==bhash
    for i in (1,2):train_pass(role,f'{role}{i}')
   assert local['records']==560
   save(jobdir/'UPDATE_AUDIT.json',{'batches':len(all_updates),'reliable_batches':sum(i['reliable'] for i in all_updates),'fallback_batches':sum(not i['reliable'] for i in all_updates),'max_gradient_recombination_error':max(i['gradient_recombination_max_error'] for i in all_updates),'mean_applied_H_gain':sum(i['gain']['H'][2] for i in all_updates)/len(all_updates),'mean_applied_T_gain':sum(i['gain']['T'][2] for i in all_updates)/len(all_updates),'EDIT_NOEDIT_same_B_state_hash':bhash})
   result[w][arm]={stage:info['metrics'] for stage,info in saved.items()};save(CAM/'METRICS.json',result)
   receipt={'status':'ACCEPTED_FULL_LIFE','world':w,'arm':arm,'records':local['records'],'targets':local['targets'],'checkpoints':{stage:{k:v for k,v in info.items() if k in ('checkpoint','checkpoint_sha256','checkpoint_bytes','clock','local_records','local_targets','state_sha256')} for stage,info in saved.items()},'same_birth_hash':birth[w],'correlated_C_forks':2,'source_lock_sha256':sha(CAM/'SOURCE_LOCK.json')}
   save(jobdir/'RECEIPT.json',receipt);receipts.append({'path':str((jobdir/'RECEIPT.json').relative_to(CAM)),'sha256':sha(jobdir/'RECEIPT.json'),'records':local['records'],'targets':local['targets']});counter['committed_jobs']+=1;save(CAM/'COMMIT_INDEX.json',{'receipts':receipts,'records':sum(r['records'] for r in receipts),'targets':sum(r['targets'] for r in receipts)});save(CAM/'COUNTERS.json',counter);check()
 assert counter['science_records']==4480 and counter['science_targets']==227148 and counter['checkpoint_count']==80
 save(CAM/'STATUS.json',{'status':'ANALYZING','stage':'SAVED_ONLY_REDUCTION',**check()})
 import reduce as R
 R.finish()
if __name__=='__main__':
 try:execute()
 except BaseException as e:
  import traceback
  save(CAM/'FAILURE.json',{'error':repr(e),'traceback':traceback.format_exc(),'counter':load(CAM/'COUNTERS.json') if (CAM/'COUNTERS.json').exists() else {},'CPU_seconds':time.process_time()});save(CAM/'STATUS.json',{'status':'FAILED','error':repr(e)});raise
