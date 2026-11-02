"""V176 explicit local DFA updates; BP is confined to disposable evaluator probes."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,time,math,json,hashlib,resource,gc
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
H=Path(__file__).resolve().parent;START=time.process_time()
def save(n,v):
 p=H/n;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def seed(s):return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],'big')&((1<<63)-1)
def rng(s):return np.random.default_rng(int.from_bytes(hashlib.sha256(s.encode()).digest()[:16],'big'))
def gen(s):return torch.Generator().manual_seed(seed(s))
def warm(n):return 1/16+15/16*np.minimum(np.asarray(n)/1024,1)
def init(pair):
 prefix=f'V176|{pair}|'
 wh=torch.randn(5177,512,generator=gen(prefix+'H-init'))/math.sqrt(184)
 with torch.random.fork_rng():
  torch.manual_seed(seed(prefix+'T-init'));t=torch.nn.Linear(512,2048)
  torch.manual_seed(seed(prefix+'O-init'));o=torch.nn.Linear(2048,257)
 return [wh,torch.zeros(512),t.weight.detach().T.contiguous(),t.bias.detach().clone(),o.weight.detach().T.contiguous(),o.bias.detach().clone()]
def dense(idx):
 idx=torch.as_tensor(idx,dtype=torch.long);x=torch.zeros((len(idx),5177));rows=torch.arange(len(idx))[:,None].expand_as(idx);good=idx>=0;x[rows[good],idx[good]]=1;return x
def local_grads(p,x,y,FH,FT,mask,perm=None):
 # No autograd graph exists for candidate training.
 wh,b,wt,bt,wo,bo=p;ah=x@wh+b;h=ah.relu();at=h@wt+bt;z=at.relu()*mask;logits=z@wo+bo
 probs=logits.softmax(1);e=probs.clone();e[torch.arange(len(y)),y]-=1;e/=len(y)
 eh=e if perm is None else e[perm];dt=(eh@FT)*mask*(at>0);dh=(eh@FH)*(ah>0)
 return [x.T@dh,dh.sum(0),h.T@dt,dt.sum(0),z.T@e,e.sum(0)],float(torch.nn.functional.cross_entropy(logits,y)),e
class Adam:
 def __init__(self,p):self.p=p;self.m=[torch.zeros_like(v) for v in p];self.v=[torch.zeros_like(v) for v in p];self.t=0
 def step(self,g,w,frozen=False,decay=.0001):
  self.t+=1;proposed=applied=0.
  for j,(p,d) in enumerate(zip(self.p,g)):
   if frozen and j<2:continue
   self.m[j].mul_(.9).add_(d,alpha=.1);self.v[j].mul_(.999).addcmul_(d,d,value=.001)
   step=(self.m[j]/(1-.9**self.t))/(torch.sqrt(self.v[j]/(1-.999**self.t))+1e-8)
   proposed+=float(step.square().sum());p.mul_(1-.001*w*decay).add_(step,alpha=-.001*w);applied+=float((.001*w*step).square().sum())
   assert torch.isfinite(p).all()
  return math.sqrt(proposed),math.sqrt(applied)
def checkpoint(p,opt,name,n):
 torch.save(dict(parameters=p,m=opt.m,v=opt.v,steps=opt.t,targets=n),H/'local'/name)
def qualify():
 t=time.process_time();p=init('technical');x=dense(np.tile(np.arange(184),(256,1)));y=torch.arange(256)
 fh=torch.randn(257,512,generator=gen('V176|technical|F-H'))/math.sqrt(257);ft=torch.randn(257,2048,generator=gen('V176|technical|F-T'))/math.sqrt(257)
 mask=(torch.rand(256,2048,generator=gen('V176|technical|dropout'))>=.5).float()*2
 grads,loss,e=local_grads(p,x,y,fh,ft,mask)
 # Independent synapse formulas on selected coordinates; no candidate backward call.
 ah=x@p[0]+p[1];hh=ah.relu();at=hh@p[2]+p[3];dd=(e@fh)*(ah>0)
 assert torch.allclose(grads[0][13,7],sum((x[i,13]*dd[i,7] for i in range(256))),atol=1e-6)
 assert torch.allclose(grads[1],dd.sum(0))
 zero,_,_=local_grads(p,x,y,fh*0,ft*0,mask);assert zero[0].abs().max()==0 and zero[2].abs().max()==0
 opt=Adam(p);before=p[0].clone();g0=grads[0].clone();w=float(warm(np.arange(256)).mean());opt.step(grads,w)
 expected=before*(1-.001*w*.0001)-.001*w*g0/(g0.abs()+1e-8);assert torch.allclose(p[0],expected,atol=2e-7)
 frozen=p[0].clone();opt.step(grads,1,frozen=True);assert torch.equal(p[0],frozen)
 tt=time.process_time();local_grads(p,x,y,fh,ft,mask);source_batch=time.process_time()-tt
 # Full evaluator update at largest (bridged KC) input dimension.
 del opt,grads,before,g0,expected;gc.collect()
 torch.manual_seed(seed('V176|technical|probe'));net=torch.nn.Sequential(torch.nn.Linear(5177,2048),torch.nn.ReLU(),torch.nn.Dropout(.5),torch.nn.Linear(2048,257))
 po=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);tt=time.process_time();po.zero_grad();pl=torch.nn.functional.cross_entropy(net(x),y);pl.backward();po.step();probe_batch=time.process_time()-tt
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 receipt=dict(status='PASS',candidate_cross_layer_autograd=False,independent_local_formula=True,zero_feedback_test=True,actual_step_warmup=True,frozen_excludes_decay=True,source_batch_CPU=source_batch,probe_batch_CPU=probe_batch,technical_targets=256+256,technical_record_touches=4,CPU=time.process_time()-t,peak_RSS_bytes=rss,torch=torch.__version__,numpy=np.__version__,probe_BP_evaluator_only=True)
 save('FEATURE_QUALIFICATION.json',receipt);print(json.dumps(receipt))
def fit():
 receipts=[]
 for pair in ('P1','P2'):
  data=np.load(H/f'local/{pair}_train.npz');ix=data['idx'];y=torch.from_numpy(data['y']);birth=init(pair)
  for arm in ('TRUE-H','FROZEN-H','SHUFFLED-H'):
   t=time.process_time();p=[v.clone() for v in birth];opt=Adam(p);n=0;curve=[]
   fh=torch.randn(257,512,generator=gen(f'V176|{pair}|F-H'))/math.sqrt(257);ft=torch.randn(257,2048,generator=gen(f'V176|{pair}|F-T'))/math.sqrt(257)
   order_rng=rng(f'V176|{pair}|batch');drop_rng=gen(f'V176|{pair}|dropout');shuf_rng=rng(f'V176|{pair}|shuffle')
   for ep in range(3):
    total=0.;order=order_rng.permutation(len(y))
    for start in range(0,len(order),256):
     sel=order[start:start+256];x=dense(ix[sel]);yy=y[sel];mask=(torch.rand(len(sel),2048,generator=drop_rng)>=.5).float()*2
     perm=torch.from_numpy(shuf_rng.permutation(len(sel))) if arm=='SHUFFLED-H' else None
     g,loss,_=local_grads(p,x,yy,fh,ft,mask,perm);w=float(warm(np.arange(n,n+len(sel))).mean());prop,applied=opt.step(g,w,arm=='FROZEN-H');n+=len(sel);total+=loss*len(sel)
     if start%4096==0:
      save('ACTIVE.json',dict(stage='C2A',pair=pair,arm=arm,pass_index=ep+1,targets=n,CPU=time.process_time()-START,last_applied_norm=applied,warmup=w))
      if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>1073741824:checkpoint(p,opt,f'{pair}_{arm}_partial.pt',n);raise RuntimeError('RSS_CAP')
    curve.append(total/len(y))
   checkpoint(p,opt,f'{pair}_{arm}_final.pt',n);np.savez(H/f'local/{pair}_{arm}_H.npz',W=p[0].numpy(),b=p[1].numpy())
   item=dict(pair=pair,arm=arm,passes=3,targets=n,records=len(np.unique(data['record_id']))*3,loss_curve=curve,CPU=time.process_time()-t,checkpoint='final',H_sha256=hashlib.sha256((H/f'local/{pair}_{arm}_H.npz').read_bytes()).hexdigest())
   receipts.append(item);save('SOURCE_TRAINING.json',dict(status='RUNNING',receipts=receipts));del p,opt,g;gc.collect()
 save('SOURCE_TRAINING.json',dict(status='COMPLETE',receipts=receipts,CPU=sum(r['CPU'] for r in receipts),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
def probe():
 results=[]
 for pair in ('P1','P2'):
  raw=np.load(H/f'local/{pair}_calib.npz');y=torch.from_numpy(raw['y']);rid=raw['record_id']
  for space in ('direct','bridged'):
   for arm in ('TRUE-H','FROZEN-H','SHUFFLED-H'):
    t=time.process_time();asset=np.load(H/f'local/{pair}_{arm}_H.npz');wh=torch.from_numpy(asset['W']);bh=torch.from_numpy(asset['b'])
    if space=='direct':
     feats=torch.cat([(dense(raw['idx'][i:i+256])@wh+bh).relu() for i in range(0,len(y),256)])
    else:feats=torch.from_numpy(np.load(H/f'local/{pair}_{arm}_bridged.npy'))
    with torch.random.fork_rng():
     torch.manual_seed(seed(f'V176|{pair}|probe|{space}'));net=torch.nn.Sequential(torch.nn.Linear(feats.shape[1],2048),torch.nn.ReLU(),torch.nn.Dropout(.5),torch.nn.Linear(2048,257))
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);n=0;curve=[]
    # Fixed random stream reset per arm; all comparisons have the same dropout draws.
    with torch.random.fork_rng():
     torch.manual_seed(seed(f'V176|{pair}|probe|{space}|dropout'))
     for ep in range(3):
      total=0.
      for i in range(0,2048,256):
       xx=feats[i:i+256];yy=y[i:i+256];opt.zero_grad();loss=torch.nn.functional.cross_entropy(net(xx),yy);loss.backward()
       w=float(warm(np.arange(n,n+len(yy))).mean());opt.param_groups[0]['lr']=.001*w;opt.step();n+=len(yy);total+=float(loss.detach())*len(yy)
      curve.append(total/2048)
    net.eval()
    with torch.no_grad():losses=torch.cat([torch.nn.functional.cross_entropy(net(feats[i:i+256]),y[i:i+256],reduction='none') for i in range(2048,len(y),256)]).numpy()
    record_losses={str(int(r)):float(losses[rid[2048:]==r].mean()) for r in np.unique(rid[2048:])}
    item=dict(pair=pair,space=space,arm=arm,eval_NLL=float(losses.mean()),eval_targets=len(losses),fit_targets=n,record_touches=3*len(np.unique(rid[:2048])),train_curve=curve,record_losses=record_losses,CPU=time.process_time()-t)
    results.append(item);save('PROBE_RESULTS.json',dict(status='RUNNING',results=results));del net,opt,feats;gc.collect()
 checks={}
 for pair in ('P1','P2'):
  checks[pair]={}
  for space in ('direct','bridged'):
   v={r['arm']:r['eval_NLL'] for r in results if r['pair']==pair and r['space']==space};checks[pair][space]=dict(true=v['TRUE-H'],frozen=v['FROZEN-H'],benefit=v['FROZEN-H']-v['TRUE-H'],passed=v['TRUE-H']<v['FROZEN-H'])
 passed=all(c['passed'] for p in checks.values() for c in p.values())
 verdict='QUALIFIED_FOR_NATIVE_B' if passed else ('SOURCE_FEATURE_SIGNAL_NOT_QUALIFIED_WITH_THIS_PROBE' if not all(p['direct']['passed'] for p in checks.values()) else 'BRIDGED_SIGNAL_NOT_QUALIFIED_WITH_THIS_PROBE')
 save('PROBE_RESULTS.json',dict(status='COMPLETE',results=results,checks=checks,verdict=verdict,CPU=sum(r['CPU'] for r in results),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));print(verdict)
if __name__=='__main__':
 try:
  command=sys.argv[1]
  if command!='qualify':
   admission=json.loads((H/'RESOURCE_ADMISSION.json').read_text())
   if admission.get('status')!='PASS':raise RuntimeError('MAIN_FITTING_NOT_ADMITTED')
   raise RuntimeError('MAIN_DRIVER_NOT_YET_QUALIFIED: no scientific fitting under this implementation')
  qualify()
 except BaseException as e:save('FEATURE_FAILURE.json',dict(error=repr(e),CPU=time.process_time()-START,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));raise
