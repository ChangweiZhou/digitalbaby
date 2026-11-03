"""Source-locked V182 fixed-dose lifetime. No efficacy stopping or rescue."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,hashlib,copy,resource,io,platform
from pathlib import Path
import numpy as np
import torch as T
import learner as L,parent_learner as P,evaluator as E,metrics as M
from sensor import Sensor
from data import prepare,corrected
T.set_num_threads(1)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=HERE/'results';OLD=ROOT/'minifly_v181_mini_local_plasticity_dev_20261102/campaign_20261102'
START=time.monotonic();CPU=time.process_time()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def read(p):return json.loads(p.read_text())
def budget():
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
 carried=read(OUT/'TECHNICAL.json')['cost']['CPU_seconds'] if (HERE/'LAUNCH.json').exists() else 0.0
 cpu=time.process_time()-CPU+carried+60.0;wall=time.monotonic()-START
 assert rss<=1073741824 and size<=268435456 and cpu<=850 and wall<=1800,(rss,size,cpu,wall)
 return dict(CPU_seconds=cpu,wall_seconds=wall,peak_RSS_bytes=rss,output_bytes=size)
def source_lock():
 files={str(p.relative_to(ROOT)):digest(p) for p in HERE.rglob('*') if p.is_file() and 'results' not in p.parts and '__pycache__' not in p.parts and p.name not in ('SOURCE_LOCK.json','LAUNCH.json')}
 deps=[ROOT/'minifly_v181_mini_local_plasticity_design_20261102/SPEC.json']+[OLD/n for n in ('DATA_AUDIT.json','MANIFEST.json','COUNT_CONTROLS.json','M1_RESULTS_COMPACT.json','M2_RESULTS_COMPACT.json')]
 files.update({str(p.relative_to(ROOT)):digest(p) for p in deps})
 save(HERE/'SOURCE_LOCK.json',{'source_commit':'5871b412c9582fbdec72165193294ae9b3db5226','files':files,'locked_before_scientific_training':True})
def source_check():
 for rel,h in read(HERE/'SOURCE_LOCK.json')['files'].items():assert digest(ROOT/rel)==h,rel

def recover():
 s=Sensor();assert s.digest==read(OLD/'DATA_AUDIT.json')['sensor_sha256']
 data,audit=prepare(s);assert audit==read(OLD/'DATA_AUDIT.json')['worlds']
 manifest=read(OLD/'MANIFEST.json')['worlds']
 for w,sp in data.items():
  for name,rs in sp.items():
   assert [hashlib.sha256(r).hexdigest() for r in rs]==manifest[w][name]['record_sha256']
   assert [len(r) for r in rs]==manifest[w][name]['record_lengths']
 save(OUT/'RECOVERY.json',{'sensor_digest':s.digest,'data_audit_equal':True,'manifest_equal':True,'worlds':audit})
 return s,data,audit

def qualify():
 OUT.mkdir(exist_ok=True);s,data,meta=recover();x=T.from_numpy(s.codes(data['M1']['A_train'][0]));y=T.tensor([*data['M1']['A_train'][0],256]);ids=['technical-record-0']*len(y)
 a=L.Learner('M1');n=P.Learner('M1');assert sum(v.numel() for v in a.p.values())==33153;assert sum(v.numel() for v in n.p.values())==37313
 assert all(T.equal(a.p[k],n.p[k]) for k in a.p) and T.equal(a.fh,n.fh) and np.array_equal(a.indices,n.indices)
 st=copy.deepcopy(a.state());g=a.gradients(x,y);a.restore(st);logit,(xx,ah,h,mask,z)=a.forward(x,True)
 hp=a.p['H'].double().clone().requires_grad_();hb=a.p['Hb'].double().clone().requires_grad_();op=a.p['O'].double().clone().requires_grad_();ob=a.p['Ob'].double().clone().requires_grad_()
 active=((xx.double()*hp).sum(2)+hb).clamp_min(0)
 logits=(active.detach()*mask.double())@op.T+ob
 loss=T.nn.functional.cross_entropy(logits,y);go,gob=T.autograd.grad(loss,(op,ob))
 error=logit.double().softmax(1);error[T.arange(len(y)),y]-=1;error/=len(y)
 surrogate=(active*(error@a.fh.double())*mask.double()).sum();gh,ghb=T.autograd.grad(surrogate,(hp,hb))
 errs={k:float((g[k].double()-v).abs().max()) for k,v in [('H',gh),('Hb',ghb),('O',go),('Ob',gob)]};assert max(errs.values())<1e-6,errs
 # One complete real record, then an exact full-state roundtrip and equal continuation.
 a.restore(st);a.step(x,y,ids);buf=io.BytesIO();T.save(a.state(),buf);assert buf.tell()<1048576;buf.seek(0);b=L.Learner('M1');b.restore(T.load(buf,weights_only=True));a.step(x,y,ids);b.step(x,y,ids);assert E.state_hash(a)==E.state_hash(b)
 zero=L.Learner('M1');zero.fh.zero_();birth=copy.deepcopy(zero.p);zero.step(x,y,ids);assert T.allclose(zero.p['H'],birth['H']*(1-1e-7),atol=1e-7);assert T.equal(zero.p['Hb'],birth['Hb']);assert not T.equal(zero.p['O'],birth['O'])
 inf_sensor=sum(v.nbytes for v in s.arrays);census={}
 for name,net in [('NORMAL',n),('SHALLOW',L.Learner('M1'))]:
  buf=io.BytesIO();T.save(net.state(),buf);cp=buf.tell();inf=inf_sensor+net.indices.nbytes+net.ix.numel()*net.ix.element_size()+sum(v.numel()*v.element_size() for v in net.p.values())+net.fh.numel()*4+(net.ft.numel()*4 if hasattr(net,'ft') else 0)+200000
  assert cp<1048576 and inf<8388608
  census[name]={'parameters':sum(v.numel() for v in net.p.values()),'checkpoint_bytes':cp,'conservative_inference_bytes':inf}
 before=E.state_hash(a);M.score(a,data['M1']['A_held'][:1],E.pack(s,data['M1']['A_held'][:1]));M.generate(a,s,data['M1']['A_held'][:1]);assert E.state_hash(a)==before
 save(OUT/'TECHNICAL.json',{'status':'PASS','gradient_max_errors':errs,'gradient_reference':'autograd technical instrument; no learned-layer BP in candidate','roundtrip_continuation_equal':True,'zero_feedback_liveness':True,'readonly_preserves_state':True,'technical_record_applications':4,'technical_target_applications':4*len(y),'census':census,'runtime':{'python':platform.python_version(),'torch':T.__version__,'numpy':np.__version__},'cost':budget()})
 source_lock();print('QUALIFIED',census,flush=True)

def execute():
 assert read(OUT/'TECHNICAL.json')['status']=='PASS';source_check();assert not (HERE/'LAUNCH.json').exists(),'No rerun'
 save(HERE/'LAUNCH.json',{'request':'Let us activate the newly built skill, and do an new round iteration now as V181 is out.','source_lock_sha256':digest(HERE/'SOURCE_LOCK.json'),'time_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
 resource.setrlimit(resource.RLIMIT_CPU,(850,900));s,data,meta=recover();metrics={};receipts=[];records=targets=0
 for w,sp0 in data.items():
  sp=dict(sp0);rm=meta[w]['correction'];rule=(bytes.fromhex(rm['context_hex']),rm['old'],rm['new'])
  for split in ('calib','held'):sp['EDIT_'+split]=[corrected(r,rule) for r in sp['A_'+split]]
  codes={name:E.pack(s,rs) for name,rs in sp.items()};metrics[w]={}
  for arm,cls in [('NORMAL',P.Learner),('SHALLOW',L.Learner)]:
   net=cls(w);job={};localrec=localtar=0;checkpoints={}
   def snap(stage):
    nonlocal checkpoints
    file=OUT/'checkpoints'/f'{w}_{arm}_{stage}.pt';file.parent.mkdir(exist_ok=True);T.save(net.state(),file);assert file.stat().st_size<1048576
    statehash=E.state_hash(net);loaded=cls(w);loaded.restore(T.load(file,weights_only=True));assert E.state_hash(loaded)==statehash
    panels={};sites={}
    for name,rs in sp.items():
     out,site=M.score(net,rs,codes[name],sp['A_held'] if name=='EDIT_held' else None)
     if name=='EDIT_held':out['partitions']=M.corrected_partitions(rs,sp['A_held'],codes[name],codes['A_held'],site)
     panels[name]=out
     if name=='EDIT_held':sites[name+'__same_KC']=np.array([np.array_equal(a,b) for ca,cb in zip(codes[name],codes['A_held']) for a,b in zip(ca,cb)])
     for k,v in site.items():sites[name+'__'+k]=v
    sf=OUT/'sites'/f'{w}_{arm}_{stage}.npz';sf.parent.mkdir(exist_ok=True);np.savez_compressed(sf,**sites)
    if stage in ('A2','B3','EDIT2'):panels['_generation']=M.generate(net,s,sp['A_held'][:16]+sp['B_held'][:16])
    job[stage]=panels;checkpoints[stage]={'state_sha256':statehash,'checkpoint_sha256':digest(file),'checkpoint_bytes':file.stat().st_size,'clock':net.clock,'cumulative_record_applications':localrec,'cumulative_target_applications':localtar,'sites_sha256':digest(sf)};budget()
   def train(role,stage):
    nonlocal records,targets,localrec,localtar
    rows=[];ys=[];ids=[];ends=[]
    sourceids=rm['selected_train_ids'] if role in ('EDIT','NOEDIT') else list(range(len(sp[role+'_train'])))
    def flush():
     nonlocal records,targets,localrec,localtar
     if not ys:return
     x=T.zeros((len(ys),5177))
     for i,row in enumerate(rows):x[i,T.tensor(row.astype(np.int64))]=1
     net.step(x,T.tensor(ys),ids,'NORMAL');nr=sum(ends);records+=nr;targets+=len(ys);localrec+=nr;localtar+=len(ys)
     assert records+8<=5000 and targets+read(OUT/'TECHNICAL_AGGREGATE.json')['technical_target_applications']<=600000
     rows.clear();ys.clear();ids.clear();ends.clear();budget()
    for ri,(r,code) in enumerate(zip(sp[role+'_train'],codes[role+'_train'])):
     for pos,(y,row) in enumerate(zip([*r,256],code)):
      rows.append(row);ys.append(y);ids.append(f'V181|{w}|'+('A' if role in ('EDIT','NOEDIT') else role)+f'|{sourceids[ri]}');ends.append(pos==len(r))
      if len(ys)==256:flush()
    flush();snap(stage);print(w,arm,stage,job[stage]['A_held']['bits'],job[stage]['B_held']['bits'],flush=True)
   snap('birth')
   for i in (1,2):train('A',f'A{i}')
   for i in (1,2,3):train('B',f'B{i}')
   bstate=copy.deepcopy(net.state());bhash=E.state_hash(net)
   for role in ('EDIT','NOEDIT'):
    net.restore(bstate);assert E.state_hash(net)==bhash
    for i in (1,2):train(role,f'{role}{i}')
   assert localrec==560 and checkpoints['EDIT2']['clock']==checkpoints['NOEDIT2']['clock']
   metrics[w][arm]=job;save(OUT/f'{w}_{arm}_METRICS.json',job)
   receipt={'world':w,'arm':arm,'records':localrec,'targets':localtar,'same_B_fork_hash':bhash,'checkpoints':checkpoints};save(OUT/f'{w}_{arm}_RECEIPT.json',receipt);receipts.append(receipt)
 assert records==2240 and targets==113574;source_check()
 save(OUT/'RUN_COMPLETE.json',{'status':'COMPLETE','records':records,'targets':targets,'receipts':len(receipts),'checkpoints':sum(len(x['checkpoints']) for x in receipts),'cost':budget(),'technical_cost':read(OUT/'TECHNICAL.json')['cost'],'new_countermodel_fits':0})
 print('RUN_COMPLETE',budget(),flush=True)
if __name__=='__main__':
 if sys.argv[1]=='qualify':qualify()
 elif sys.argv[1]=='run':execute()

