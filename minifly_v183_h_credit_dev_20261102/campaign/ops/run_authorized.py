from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
"""Future fixed M1 lifetime runner. Requires a separate owner launch record."""
import bootstrap as B
from learner import Learner
import metrics as M
from pathlib import Path
import json,time,copy,resource,os
T=B.T;np=B.np;H=B.H;OUT=H/'campaign';ARMS=('NORMAL','HPC','SHUFFLED')

def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def load(path):return json.loads(Path(path).read_text())
def batches(records,codes,sourceids,role):
 rows=[];ys=[];ids=[];ends=[]
 for ri,(r,code) in enumerate(zip(records,codes)):
  for pos,(row,y) in enumerate(zip(code,[*r,256])):
   rows.append(row);ys.append(y);ids.append(f'V181|M1|{role}|{sourceids[ri]}');ends.append(pos==len(r))
   if len(ys)==256:yield rows,ys,ids,ends;rows=[];ys=[];ids=[];ends=[]
 if ys:yield rows,ys,ids,ends

def execute():
 auth=load(OUT/'AUTHORIZATION.json');assert auth['scientific_training_authorized'] and auth['user_instruction'] in ('Run it','Start it')
 assert not (OUT/'RUN_STARTED').exists(),'No rerun or automatic retry'
 OUT.mkdir(exist_ok=True)
 for rel,h in load(OUT/'SOURCE_LOCK.json')['files'].items():assert B.digest(B.ROOT/rel)==h,rel
 (OUT/'RUN_STARTED').write_text(str(time.time()));start=time.monotonic();cpu=time.process_time();carried=auth['charged_CPU_carried'];wallcarried=auth['active_wall_carried'];qledger=load(H/'EXPOSURE_LEDGER.json');baseq=qledger['totals'];statecost={'optimizer_records':0,'optimizer_targets':0,'learning_loss_targets':0,'learning_record_equivalents':0,'read_only_targets':0,'read_only_record_views':0,'generation_outputs':0,'generation_prefix_bytes':0}
 def budget():
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if B.sys.platform=='darwin' else 1024);spent=time.process_time()-cpu+carried;wall=time.monotonic()-start+wallcarried;disk=sum(p.stat().st_size for p in H.rglob('*') if p.is_file())+sum(p.stat().st_size for p in Path('/private/tmp/minifly_v183_runtime').rglob('*') if p.is_file())
  assert spent<=900 and wall<=1800 and rss<=1073741824 and disk<=268435456,(spent,wall,rss,disk)
  assert statecost['learning_record_equivalents']+baseq['technical_loss_record_equivalents']+baseq.get('conservative_all_frozen_forward_record_equivalents',baseq['frozen_assay_record_equivalents'])<=5000
  assert statecost['learning_loss_targets']+baseq['technical_target_evaluations']+baseq.get('conservative_all_frozen_forward_target_evaluations',baseq['frozen_assay_target_evaluations'])<=600000
  return {'CPU_seconds':spent,'wall_seconds':wall,'peak_RSS_bytes':rss,'output_bytes_including_runtime':disk}
 sensor=B.sensor.Sensor();data,meta=B.D.prepare(sensor);sp=dict(data['M1']);rm=meta['M1']['correction'];rule=(bytes.fromhex(rm['context_hex']),rm['old'],rm['new'])
 for split in ('calib','held'):sp['EDIT_'+split]=[B.D.corrected(r,rule) for r in sp['A_'+split]]
 codes={name:M.pack(sensor,rs) for name,rs in sp.items()};assert sum(len(r)+1 for rs in sp.values() for r in rs)==33307
 manifest={name:{'record_sha256':[B.hashlib.sha256(r).hexdigest() for r in rs],'targets':[len(r)+1 for r in rs]} for name,rs in sp.items()};save(OUT/'MANIFEST.json',manifest);all_metrics={};all_receipts=[]
 for arm in ARMS:
  net=Learner();job=OUT/arm;job.mkdir(exist_ok=False);stage_metrics={};checkpoints={};localrec=localtar=0;updates=[]
  def snapshot(stage):
   cp=job/(stage+'.pt');T.save(net.state(),cp);assert cp.stat().st_size<=1048576;test=Learner();test.restore(T.load(cp,weights_only=True));assert B.statehash(test)==B.statehash(net)
   panels={};sitearrays={};forecasts=None;held=None
   for name,rs in sp.items():
    out,sites=M.score(net,rs,codes[name],sp['A_held'] if name=='EDIT_held' else None,record_log_sums=stage=='A2' and name=='A_train')
    if name=='EDIT_held':out['partitions']=M.partitions(sites,sp['A_held'],codes[name],codes['A_held'])
    if name=='A_train' and stage=='A2':forecasts=M.forecast_from_summaries(sites)
    if name=='A_held':held=sites
    panels[name]=out
    for k,v in sites.items():sitearrays[name+'__'+k]=v
    statecost['read_only_targets']+=len(sites['y']);statecost['read_only_record_views']+=len(rs)
   if stage=='A2':
    assert forecasts is not None and held is not None;panels['_own_mean_forecast_A_held_bits']=M.forecast_loss(forecasts,held);sitearrays['A2_mean_logq']=forecasts
   np.savez_compressed(job/(stage+'_sites.npz'),**sitearrays)
   if stage in ('A2','B3','EDIT2'):
    panels['_generation']=M.generate(net,sensor,sp['A_held'][:16]+sp['B_held'][:16]);statecost['generation_outputs']+=sum(v['model_outputs'] for v in panels['_generation']);statecost['generation_prefix_bytes']+=sum(len(bytes.fromhex(v['prefix_hex'])) for v in panels['_generation'])
   stage_metrics[stage]=panels;save(job/(stage+'.json'),panels);checkpoints[stage]={'state_sha256':B.statehash(net),'checkpoint_sha256':B.digest(cp),'checkpoint_bytes':cp.stat().st_size,'sites_sha256':B.digest(job/(stage+'_sites.npz')),'metrics_sha256':B.digest(job/(stage+'.json')),'clock':net.clock,'optimizer_records':localrec,'optimizer_targets':localtar};save(OUT/'COUNTERS.json',statecost);budget()
  def train(role,stage):
   nonlocal localrec,localtar
   sourceids=rm['selected_train_ids'] if role in ('EDIT','NOEDIT') else range(len(sp[role+'_train']));origrole='A' if role in ('EDIT','NOEDIT') else role;factor=1 if arm=='NORMAL' else 3
   for rows,ys,ids,ends in batches(sp[role+'_train'],codes[role+'_train'],sourceids,origrole):
    x=T.zeros((len(ys),5177))
    for i,row in enumerate(rows):x[i,T.tensor(row.astype(np.int64))]=1
    info=net.step(x,T.tensor(ys),ids,arm);nr=sum(ends);localrec+=nr;localtar+=len(ys);statecost['optimizer_records']+=nr;statecost['optimizer_targets']+=len(ys);statecost['learning_loss_targets']+=factor*len(ys);statecost['learning_record_equivalents']+=factor*nr;updates.append(info);save(OUT/'COUNTERS.json',statecost);save(OUT/'ACTIVE.json',{'arm':arm,'stage':stage,'cost':budget()})
   snapshot(stage)
  snapshot('birth')
  for i in (1,2):train('A',f'A{i}')
  for i in (1,2,3):train('B',f'B{i}')
  st=copy.deepcopy(net.state());finger=B.statehash(net)
  for role in ('EDIT','NOEDIT'):
   net.restore(st);assert B.statehash(net)==finger
   for i in (1,2):train(role,f'{role}{i}')
  assert localrec==560 and localtar==34364 and checkpoints['EDIT2']['clock']==checkpoints['NOEDIT2']['clock']
  receipt={'arm':arm,'records':localrec,'targets':localtar,'checkpoints':checkpoints,'same_B_fork_state_hash':finger,'source_lock_sha256':B.digest(OUT/'SOURCE_LOCK.json')};save(job/'RECEIPT.json',receipt);save(job/'UPDATE_DIAGNOSTICS.json',updates);all_receipts.append(receipt);all_metrics[arm]=stage_metrics;save(OUT/'METRICS.json',all_metrics);save(OUT/'COMMIT_INDEX.json',all_receipts)
 assert statecost['optimizer_records']==1680 and statecost['optimizer_targets']==103092 and statecost['learning_loss_targets']==240548 and statecost['learning_record_equivalents']==3920
 assert statecost['read_only_targets']==999210 and statecost['read_only_record_views']==16320 and statecost['generation_outputs']<=18432
 save(OUT/'RUN_COMPLETE.json',{'status':'COMPLETE','counts':statecost,'cost':budget(),'full_checkpoints':30,'scientific_lives':3,'correlated_C_child_branches':6,'new_countermodel_fits':0})
 from finalize import finish
 finish(OUT,budget)
if __name__=='__main__':execute()
