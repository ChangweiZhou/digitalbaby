"""Fixed local source learner and evaluator-only probe, with persistent accounting."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,hashlib,gc,resource
import runtime as C
sys.path.insert(0,str(C.DEV));import feature_stage as S
import numpy as np
import torch
torch.set_num_threads(1)
ACTIVE=None
def rawhash(t):return hashlib.sha256(t.numpy().tobytes()).hexdigest()
def cp(name,state):
 path=C.R/'local'/name;torch.save(state,path);return path
def initialize():
 locks={}
 for pair in ('P1','P2'):
  p=S.init(pair);locks[pair]=[rawhash(v) for v in p]
 C.save('BIRTH_LOCKS.json',dict(parameters=locks,torch=torch.__version__,numpy=np.__version__,candidate_cross_layer_autograd=False))
def qualify():
 global ACTIVE
 initialize();p=S.init('technical_pipeline');opt=S.Adam(p);g=S.gen('V176|technical_pipeline|dropout')
 ix=np.tile(np.arange(184,dtype=np.int16),(64,1));x=S.dense(ix);y=torch.tensor([*range(63),256]);fh=torch.randn(257,512,generator=S.gen('V176|technical_pipeline|F-H'))/257**.5;ft=torch.randn(257,2048,generator=S.gen('V176|technical_pipeline|F-T'))/257**.5
 mask=(torch.rand(64,2048,generator=g)>=.5).float()*2;grads,loss,e=S.local_grads(p,x,y,fh,ft,mask)
 ACTIVE=dict(parameters=p,optimizer=opt,targets=64)
 C.COUNTS['qualification_targets']+=64;C.COUNTS['probability_evaluations']+=64;C.COUNTS['technical_feature_touches']+=2
 w=float(S.warm(np.arange(64)).mean());opt.step(grads,w)
 path=cp('qualification_source.pt',dict(parameters=p,m=opt.m,v=opt.v,step=opt.t,n=64,dropout=g.get_state()))
 restored=torch.load(path,weights_only=False);assert all(torch.equal(a,b) for a,b in zip(p,restored['parameters']))
 np.savez(C.R/'local/qualification_H.npz',W=p[0].numpy(),b=p[1].numpy())
 # A probe is the only training component allowed to call backward across its own layers.
 with torch.random.fork_rng():
  torch.manual_seed(S.seed('V176|technical_pipeline|probe'));probe=torch.nn.Sequential(torch.nn.Linear(512,2048),torch.nn.ReLU(),torch.nn.Dropout(.5),torch.nn.Linear(2048,257))
 popt=torch.optim.AdamW(probe.parameters(),lr=.001*w,weight_decay=.0001);popt.zero_grad()
 feats=(x@p[0]+p[1]).relu().detach();pl=torch.nn.functional.cross_entropy(probe(feats),y);pl.backward();popt.step()
 np.savez(C.R/'local/qualification_features.npz',idx=ix,h=feats.numpy())
 assert all(v.grad is None for v in p)
 C.COUNTS['qualification_targets']+=64;C.COUNTS['probability_evaluations']+=64
 C.save('SOURCE_DRIVER_QUALIFICATION.json',dict(status='PASS',local_step=True,applied_warmup=True,checkpoint_roundtrip=True,evaluator_only_BP=True,EOS_class_tested=True,CPU=C.usage()))
 C.done()
def fit():
 global ACTIVE
 assert json.loads((C.R/'QUALIFICATION.json').read_text())['status']=='PASS'
 locks=json.loads((C.R/'BIRTH_LOCKS.json').read_text())['parameters'];summary=[]
 for pair in ('P1','P2'):
  data=np.load(C.DEV/f'local/{pair}_train.npz');ix=data['idx'];y=torch.from_numpy(data['y']);rids=data['record_id'];birth=S.init(pair)
  assert [rawhash(v) for v in birth]==locks[pair]
  for arm in ('TRUE-H','FROZEN-H','SHUFFLED-H'):
   start=C.usage();p=[v.clone() for v in birth];opt=S.Adam(p);n=0;curve=[];order_rng=S.rng(f'V176|{pair}|batch');drop=S.gen(f'V176|{pair}|dropout');shuffle=S.rng(f'V176|{pair}|shuffle')
   fh=torch.randn(257,512,generator=S.gen(f'V176|{pair}|F-H'))/257**.5;ft=torch.randn(257,2048,generator=S.gen(f'V176|{pair}|F-T'))/257**.5
   ACTIVE=dict(pair=pair,arm=arm,parameters=p,optimizer=opt,n=n,order_rng=order_rng,dropout=drop,shuffle=shuffle,FH=fh,FT=ft)
   for ep in range(3):
    order=order_rng.permutation(len(y));counts=np.zeros(int(rids.max())+1,np.int32);required=np.bincount(rids);total=0.
    for pos in range(0,len(order),256):
     sel=order[pos:pos+256];xx=S.dense(ix[sel]);yy=y[sel];mask=(torch.rand(len(sel),2048,generator=drop)>=.5).float()*2
     perm=torch.from_numpy(shuffle.permutation(len(sel))) if arm=='SHUFFLED-H' else None
     grads,loss,_=S.local_grads(p,xx,yy,fh,ft,mask,perm)
     C.COUNTS['source_targets']+=len(sel);C.COUNTS['probability_evaluations']+=len(sel)
     w=float(S.warm(np.arange(n,n+len(sel))).mean());proposed,applied=opt.step(grads,w,arm=='FROZEN-H');n+=len(sel);total+=loss*len(sel)
     before=int(np.sum(counts==required));np.add.at(counts,rids[sel],1);C.COUNTS['source_records']+=int(np.sum(counts==required))-before
     ACTIVE.update(n=n,pass_index=ep+1,batch_position=pos,record_completion=counts)
     C.pulse('SOURCE_FITTING',pair=pair,arm=arm,pass_index=ep+1,targets=n,warmup=w,proposed_norm=proposed,applied_norm=applied)
    assert np.array_equal(counts,required);curve.append(total/len(y))
   if arm=='FROZEN-H':assert rawhash(p[0])==locks[pair][0] and rawhash(p[1])==locks[pair][1] and opt.m[0].abs().max()==0 and opt.v[0].abs().max()==0
   path=cp(f'{pair}_{arm}_source_final.pt',dict(parameters=p,m=opt.m,v=opt.v,steps=opt.t,n=n,FH=fh,FT=ft,order_rng=order_rng.bit_generator.state,dropout=drop.get_state(),shuffle=shuffle.bit_generator.state))
   hp=C.R/f'local/{pair}_{arm}_H.npz';np.savez(hp,W=p[0].numpy(),b=p[1].numpy())
   item=dict(pair=pair,arm=arm,passes=3,records=len(required)*3,targets=n,loss_curve=curve,CPU=C.usage()-start,H_array_sha256=rawhash(p[0]),H_bias_sha256=rawhash(p[1]),checkpoint='fixed_final')
   C.receipt('SOURCE',f'{pair}_{arm}_source',item['records'],n,[path,hp],metrics=item);summary.append(item);C.save('SOURCE_TRAINING.json',dict(status='RUNNING',receipts=summary))
   ACTIVE=None;del p,opt,grads,xx,mask;gc.collect()
 C.save('SOURCE_TRAINING.json',dict(status='COMPLETE',receipts=summary));C.done()
def probe():
 global ACTIVE
 rows=[]
 for pair in ('P1','P2'):
  data=np.load(C.DEV/f'local/{pair}_calib.npz');y=torch.from_numpy(data['y']);rid=data['record_id']
  for space in ('direct','bridged'):
   for arm in ('TRUE-H','FROZEN-H','SHUFFLED-H'):
    start=C.usage();f=np.load(C.R/f'local/{pair}_{arm}_{space}.npy',mmap_mode='r');dim=512 if space=='direct' else 5177
    with torch.random.fork_rng():
     torch.manual_seed(S.seed(f'V176|{pair}|probe|{space}'));net=torch.nn.Sequential(torch.nn.Linear(dim,2048),torch.nn.ReLU(),torch.nn.Dropout(.5),torch.nn.Linear(2048,257))
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);n=0;curve=[]
    ACTIVE=dict(pair=pair,space=space,arm=arm,probe=net,optimizer=opt,n=n)
    def features(i,j):
     return torch.from_numpy(np.array(f[i:j],copy=True)) if space=='direct' else S.dense(f[i:j])
    with torch.random.fork_rng():
     torch.manual_seed(S.seed(f'V176|{pair}|probe|{space}|dropout'))
     for ep in range(3):
      total=0.
      for i in range(0,2048,256):
       xx=features(i,i+256);yy=y[i:i+256];opt.zero_grad();loss=torch.nn.functional.cross_entropy(net(xx),yy);loss.backward();w=float(S.warm(np.arange(n,n+len(yy))).mean());opt.param_groups[0]['lr']=.001*w;opt.step();n+=len(yy)
       C.COUNTS['probe_fit_targets']+=len(yy);C.COUNTS['probability_evaluations']+=len(yy);total+=float(loss.detach())*len(yy);ACTIVE['n']=n
      C.COUNTS['probe_record_touches']+=len(np.unique(rid[:2048]));curve.append(total/2048);C.pulse('PROBE_FITTING',pair=pair,space=space,arm=arm,pass_index=ep+1)
    net.eval();losses=[]
    with torch.no_grad():
     for i in range(2048,len(y),256):
      vals=torch.nn.functional.cross_entropy(net(features(i,min(i+256,len(y)))),y[i:i+256],reduction='none').numpy();losses.extend(vals.tolist());C.COUNTS['probability_evaluations']+=len(vals)
    losses=np.array(losses);per_record={str(int(r)):dict(loss_sum=float(losses[rid[2048:]==r].sum()),targets=int(np.sum(rid[2048:]==r))) for r in np.unique(rid[2048:])}
    item=dict(pair=pair,space=space,arm=arm,eval_NLL=float(losses.mean()),eval_targets=len(losses),fit_targets=n,source_record_touches=3*len(np.unique(rid[:2048])),loss_curve=curve,per_record=per_record,CPU=C.usage()-start)
    path=C.R/f'local/{pair}_{space}_{arm}_probe.pt';torch.save(dict(state=net.state_dict(),n=n),path)
    C.receipt('PROBE',f'{pair}_{space}_{arm}_probe',item['source_record_touches'],n,[path],metrics=item);rows.append(item);C.save('PROBE_RESULTS.json',dict(status='RUNNING',results=rows));ACTIVE=None;del net,opt,f;gc.collect()
 checks={}
 for pair in ('P1','P2'):
  checks[pair]={}
  for space in ('direct','bridged'):
   v={r['arm']:r['eval_NLL'] for r in rows if r['pair']==pair and r['space']==space};checks[pair][space]=dict(TRUE=v['TRUE-H'],FROZEN=v['FROZEN-H'],benefit=v['FROZEN-H']-v['TRUE-H'],passed=v['TRUE-H']<v['FROZEN-H'])
 qualified=all(x['passed'] for v in checks.values() for x in v.values());verdict='QUALIFIED_FOR_NATIVE_B' if qualified else ('SOURCE_FEATURE_SIGNAL_NOT_QUALIFIED_WITH_THIS_PROBE' if not all(v['direct']['passed'] for v in checks.values()) else 'BRIDGED_SIGNAL_NOT_QUALIFIED_WITH_THIS_PROBE')
 C.save('PROBE_RESULTS.json',dict(status='COMPLETE',results=rows,checks=checks,verdict=verdict));C.save('ADMISSION.json',dict(verdict=verdict,native_B_admitted=qualified,checks=checks));C.done(verdict=verdict)
def partial(e):
 p=C.R/'local/FAILED_source_state.pt'
 if ACTIVE is not None:
  state=dict(ACTIVE)
  if 'optimizer' in state:
   op=state.pop('optimizer');state['optimizer']=dict(m=op.m,v=op.v,t=op.t) if isinstance(op,S.Adam) else op.state_dict()
  for k in ('order_rng','shuffle'):
   if k in state:state[k]=state[k].bit_generator.state
  if 'dropout' in state:state['dropout']=state['dropout'].get_state()
  torch.save(state,p)
 C.fail(e,str(p) if p.exists() else None)
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:partial(e);raise
