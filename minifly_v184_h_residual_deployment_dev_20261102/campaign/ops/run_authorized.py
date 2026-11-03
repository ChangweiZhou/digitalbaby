from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
"""Future owner-authorized fixed M1 run; never launched by qualification."""
import bootstrap as B
from learner import Learner,ControlNotQualified
from data import prepare
import evaluate as E
from pathlib import Path
import os,json,time,copy,resource
T=B.T;np=B.np;H=B.H;OUT=H/'campaign';FULL={'birth','A2','B3','EDIT2','NOEDIT2'};ARMS=('NORMAL','HRES','NORM_SHAM')

def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def load(p):return json.loads(Path(p).read_text())
def batches(rs,codes,ids,role):
 rows=[];ys=[];names=[];ends=[]
 for ri,(r,code) in enumerate(zip(rs,codes)):
  for j,(row,y) in enumerate(zip(code,[*r,256])):
   rows.append(row);ys.append(y);names.append(f'V181|M1|{role}|{ids[ri]}');ends.append(j==len(r))
   if len(ys)==256:yield rows,ys,names,ends;rows=[];ys=[];names=[];ends=[]
 if ys:yield rows,ys,names,ends

def execute():
 auth=load(OUT/'AUTHORIZATION.json');assert auth['scientific_training_authorized'] and auth['owner_request_verbatim'] in ('Run it','Start it','Launch it')
 assert not (OUT/'RUN_STARTED').exists(),'No restart or retry'
 for rel,d in load(OUT/'SOURCE_LOCK.json')['files'].items():assert B.digest(B.ROOT/rel)==d,rel
 (OUT/'RUN_STARTED').write_text(str(time.time()));start=time.monotonic();cpu=time.process_time();counter={'optimizer_records':0,'optimizer_targets':0,'read_only_targets':0,'read_only_record_views':0,'generation_outputs':0,'generation_prefix_bytes':0,'alignment_backbone_sites':0,'alignment_scored_target_evaluations':0,'full_states':0,'summary_snapshots':0};qcounts=load(H/'EXPOSURE_LEDGER.json')['totals']
 def budget():
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if B.sys.platform=='darwin' else 1024);spent=time.process_time()-cpu+auth['charged_CPU_carried'];wall=time.monotonic()-start+auth['active_execution_wall_carried'];disk=sum(p.stat().st_size for base in (H,Path('/private/tmp/minifly_v183_runtime')) for p in base.rglob('*') if p.is_file())
  assert spent<=900 and wall<=1800 and rss<=1073741824 and disk<=268435456,(spent,wall,rss,disk)
  assert counter['optimizer_records']+qcounts['technical_record_equivalents']<=5000 and counter['optimizer_targets']+qcounts['technical_supervised_target_evaluations']<=600000
  return {'CPU_seconds':spent,'active_execution_wall_seconds':wall,'peak_RSS_bytes':rss,'output_bytes_including_runtime':disk}
 s=B.sensor.Sensor();assert s.digest=='7e0b78ae71cd9f508e9069018ec2343a88ab8bfd9263067506ee6eb3fae4962b' and B.scipy.__version__=='1.17.0';sp,cids,manifest=prepare();save(OUT/'MANIFEST.json',manifest);codes={k:E.pack(s,rs) for k,rs in sp.items()};metrics={};receipts=[]
 for arm in ARMS:
  net=Learner(arm);assert net.alpha==.5 and all(p.dtype==T.float32 for p in net.p.values());job=OUT/arm;job.mkdir(exist_ok=False);panels={};states={};lr=lt=0;updates=[];norm={'sites':0,'clipped_sites':0,'zero_denominator_sites':0,'harmless_both_zero_sites':0,'material_norm_mismatch_sites':0,'gain_max':0.};orig=net.forward
  def observed(*args,**kw):
   try:ans=orig(*args,**kw)
   except ControlNotQualified:
    save(OUT/'CONTROL_FAILURE.json',{'arm':arm,'diagnostics':net.last,'no_fallback_or_gain_cap_change':True});raise
   for k in ('sites','clipped_sites','zero_denominator_sites','harmless_both_zero_sites','material_norm_mismatch_sites'):norm[k]+=net.last[k]
   norm['gain_max']=max(norm['gain_max'],net.last['gain_max']);return ans
  net.forward=observed
  def snap(stage):
   result={};raw={};fore=None;held=None
   if stage in FULL:
    cp=job/(stage+'.pt');T.save(net.state(),cp);assert cp.stat().st_size<=1048576;t=Learner(arm);t.restore(T.load(cp,weights_only=True));assert B.statehash(t)==B.statehash(net);counter['full_states']+=1
   for name,rs in sp.items():
    out,sites=E.score(net,rs,codes[name],sp['A_held'] if name=='EDIT_held' else None,record_log_sums=stage=='A2' and name=='A_train')
    if name=='EDIT_held':out['partitions']=E.partitions(sites,sp['A_held'],codes[name],codes['A_held'])
    if name=='A_train' and stage=='A2':fore=E.forecast_from_summaries(sites)
    if name=='A_held':held=sites
    y=sites['y'];ids=sites['record_id'];payload=y!=256;b=-sites['logp_true']/np.log(2);out['payload_record_bit_sums']=[float(b[(ids==i)&payload].sum()) for i in range(len(rs))];out['payload_record_target_counts']=[int(((ids==i)&payload).sum()) for i in range(len(rs))];result[name]=out;counter['read_only_targets']+=len(y);counter['read_only_record_views']+=len(rs)
    if stage in FULL:
     for k,v in sites.items():raw[name+'__'+k]=v
   if stage=='A2':result['_own_mean_forecast_A_held_bits']=E.forecast_loss(fore,held);raw['A2_mean_logq']=fore
   if stage in FULL:np.savez_compressed(job/(stage+'_sites.npz'),**raw)
   if stage in ('A2','B3','EDIT2'):
    result['_generation']=E.generate(net,s,sp['A_held'][:16]+sp['B_held'][:16]);counter['generation_outputs']+=sum(v['model_outputs'] for v in result['_generation']);counter['generation_prefix_bytes']+=sum(len(bytes.fromhex(v['prefix_hex'])) for v in result['_generation'])
   if arm=='HRES' and stage=='A2':
    align=E.alignment(net,sp['A_held'],codes['A_held']);save(job/'A2_ALIGNMENT.json',align);counter['alignment_backbone_sites']+=align['held_sites'];counter['alignment_scored_target_evaluations']+=2*align['eligible_sites']
   panels[stage]=result;save(job/(stage+'.json'),result)
   if stage in FULL:states[stage]={'checkpoint_sha256':B.digest(job/(stage+'.pt')),'checkpoint_bytes':(job/(stage+'.pt')).stat().st_size,'state_sha256':B.statehash(net),'sites_sha256':B.digest(job/(stage+'_sites.npz')),'metrics_sha256':B.digest(job/(stage+'.json')),'clock':net.clock,'optimizer_records':lr,'optimizer_targets':lt}
   counter['summary_snapshots']+=1;save(OUT/'COUNTERS.json',counter);save(job/'NORM_CENSUS.json',norm);budget()
  def train(role,stage):
   nonlocal lr,lt
   ids=cids if role in ('EDIT','NOEDIT') else list(range(len(sp[role+'_train'])));origrole='A' if role in ('EDIT','NOEDIT') else role
   for rows,ys,tags,ends in batches(sp[role+'_train'],codes[role+'_train'],ids,origrole):
    x=T.zeros((len(ys),5177))
    for i,row in enumerate(rows):x[i,T.tensor(row.astype(np.int64))]=1
    info=net.step(x,T.tensor(ys),tags);updates.append(info);nr=sum(ends);lr+=nr;lt+=len(ys);counter['optimizer_records']+=nr;counter['optimizer_targets']+=len(ys);save(OUT/'COUNTERS.json',counter);save(OUT/'ACTIVE.json',{'arm':arm,'phase':stage,'cost':budget()})
   snap(stage)
  snap('birth')
  for i in (1,2):train('A',f'A{i}')
  for i in (1,2,3):train('B',f'B{i}')
  st=copy.deepcopy(net.state());finger=B.statehash(net)
  for role in ('EDIT','NOEDIT'):
   net.restore(st);assert B.statehash(net)==finger
   for i in (1,2):train(role,f'{role}{i}')
  assert lr==560 and lt==34364 and states['EDIT2']['clock']==states['NOEDIT2']['clock'];receipt={'arm':arm,'records':lr,'targets':lt,'checkpoints':states,'same_B_fork_hash':finger,'source_lock_sha256':B.digest(OUT/'SOURCE_LOCK.json')};save(job/'RECEIPT.json',receipt);save(job/'UPDATE_DIAGNOSTICS.json',updates);receipts.append(receipt);metrics[arm]=panels;save(OUT/'COMMIT_INDEX.json',receipts);save(OUT/'METRICS.json',metrics)
 assert counter['optimizer_records']==1680 and counter['optimizer_targets']==103092 and counter['full_states']==15 and counter['summary_snapshots']==30 and counter['read_only_targets']==999210
 save(OUT/'RUN_COMPLETE.json',{'status':'COMPLETE','counts':counter,'cost':budget(),'new_countermodel_fits':0})
 from finalize import finish
 finish(OUT,budget)
if __name__=='__main__':execute()
