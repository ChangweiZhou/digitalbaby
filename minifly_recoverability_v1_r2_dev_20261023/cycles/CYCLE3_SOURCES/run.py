from rc_common import *
import rc_common as C
import head as H
OLD=R/'minifly_recoverability_v1_dev_20261022'

def infer(D,std,hs):
 scale=np.load(P/'heads'/f'scale_{int(std)}.npy');X,base=H.features(D,scale,'MEAN')
 with np.load(P/'heads'/f'{int(std)}_{hs}.npz') as f:p={k:f[k] for k in H.NAMES}
 z=base.copy();z[:,:,8:9]=H.forward(p,X,base)[0];return z

def metric(z,vals):
 out=[];payload=[];stop=[];correct=[]
 for q,v in zip(z,vals):
  L=len(v);bs=[];ended=False
  for j in range(9):
   if q[j,8]>0:ended=True;break
   if j==8:break
   bs.append(sum(1<<b for b in range(8) if q[j,b]>0))
  out.append(dict(bytes=bs,EOS=ended));correct.append(ended and bs==v)
  payload.append(all(sum(1<<b for b in range(8) if q[j,b]>0)==t for j,t in enumerate(v)))
  stop.append(bool(np.all(q[:L,8]<=0) and q[L,8]>0))
 ix=[i for i,v in enumerate(vals) if v];eligible=[i for i in ix if payload[i]]
 return dict(nonempty_exact=float(np.mean([correct[i] for i in ix])) if ix else None,payload_exact=float(np.mean([payload[i] for i in ix])) if ix else None,conditional_n=len(eligible),conditional_stop=float(np.mean([stop[i] for i in eligible])) if eligible else None,exact=float(np.mean(correct)),outputs=out)

def evaluate(tag):
 meta=json.loads((P/'traces'/f'{tag}.json').read_text());rows=[]
 with np.load(P/'traces'/f'{tag}.npz') as f:
  for i,(pool,vals) in enumerate(zip(meta['pools'],meta['labels'])):
   D=f[f'D{i}'];rows.append(dict(pool=pool,head='DIRECT',lesion='NONE',metrics=metric(D.mean(-1),vals)))
   for hs in range(202610239401,202610239404):
    for lesion in (('NONE','ZERO','FOREIGN') if meta['arm']=='W' else ('NONE',)):
     src=D if lesion=='NONE' else np.zeros_like(D) if lesion=='ZERO' else np.roll(D,1,axis=0)
     z=infer(src,meta['STD'],hs);assert np.array_equal(z[:,:,:8]>0,src.mean(-1)[:,:,:8]>0)
     rows.append(dict(pool=pool,head=hs,lesion=lesion,metrics=metric(z,vals)))
 save(f'traces/{tag}_outputs.json',dict(**meta,results=rows))

def cycle1():
 rng=np.random.default_rng(202610239501);D=rng.normal(size=(2,9,9,2));X,base=H.features(D,np.ones(9),'MEAN');p=H.initialize(202610239501);assert sum(x.size for x in p.values())==5537
 assert np.array_equal(H.forward(p,X,base)[0],base[:,:,8:9]);p['w2'][:]=rng.normal(0,.1,(32,1));y,w=H.targets([[],list(range(8))]);_,g=H.loss_grad(p,X,base,y,w);errs=[]
 for k in H.NAMES:
  ix=tuple(0 for _ in p[k].shape);a=p[k][ix];p[k][ix]=a+1e-6;hi=H.loss_grad(p,X,base,y,w)[0];p[k][ix]=a-1e-6;lo=H.loss_grad(p,X,base,y,w)[0];p[k][ix]=a;errs.append(abs((hi-lo)/2e-6-g[k][ix]))
 assert max(errs)<1e-7
 e=Engine(5177,202610239501,False);x,_=encode(e,keys(202610239501)[:1],True)
 # Nonzero prestate, g=1 and g=0 parity; four partial fixture events.
 e.begin();e.tick(x[0,0],np.linspace(-.01,.01,9));C.events+=1
 for gate in (1.,0.):
  a=e.clone();ref=e.clone();c=np.linspace(-.003,.003,9);protected_tick(a,x[0,1],c,np.full((9,5177),gate));ref.tick(x[0,1],c if gate else None);C.events+=1
  for k in ('fast','slow','adapt'):assert np.max(abs(getattr(a.bank,k)-getattr(ref.bank,k)))<1e-12
 before=digest(e);probe(e,x);assert digest(e)==before
 h=np.array([0.,.3,1.]);v=np.array([0.,.5,.9]);assert np.all((h+(1-h)*v)>=0) and np.all((h+(1-h)*v)<=1)
 save('cycles/CYCLE1_RESULT.json',dict(status='PASS',native_partial_events=C.events,gradient_max_error=max(errs),parameters=5537,payload_sign_invariant=True,g1_g0_parity=True,readonly=True))

def cycle2():
 for std in (0,1):
  Ds=[];vs=[]
  for seed in range(202610229101,202610229113):
   meta=json.loads((OLD/'traces'/f'{seed}_0_{std}_W.json').read_text())
   with np.load(OLD/'traces'/f'{seed}_0_{std}_W.npz') as z:
    for i,vals in enumerate(meta['labels']):Ds.append(z[f'D{i}']);vs+=vals
  D=np.concatenate(Ds);scale=np.load(P/'heads'/f'scale_{std}.npy');X,base=H.features(D,scale,'MEAN');Y,W=H.targets(vs)
  for seed in range(202610239401,202610239404):
   p=H.initialize(seed);st={k:[np.zeros_like(a),np.zeros_like(a)] for k,a in p.items()};rng=np.random.default_rng(seed+44);terms=0;curve=[];final=int((W>0).sum())
   for step in range(1,2001):
    check();ids=rng.integers(0,len(D),32);cost=int((W[ids]>0).sum())
    if terms+cost+final>2000000:break
    loss,g=H.loss_grad(p,X[ids],base[ids],Y[ids],W[ids]);H.adam(p,g,st,step);terms+=cost
    if step%100==0:curve.append(dict(step=step,loss=loss))
   loss=H.loss_grad(p,X,base,Y,W)[0];terms+=final
   np.savez_compressed(P/'heads'/f'{std}_{seed}.npz',**p);save(f'heads/{std}_{seed}.json',dict(updates=step,terms=terms,loss=loss,curve=curve));check(f'head frozen {std}/{seed}')
 save('HEAD_FREEZE.json',{x.name:sha(x) for x in (P/'heads').iterdir() if x.is_file()})
 save('cycles/CYCLE2_RESULT.json',dict(status='COMPLETE',fits=6,new_native_records=0,training_source='immutable V1 twelve episodes',revision='Freeze EOS heads; native protection does not depend on favorable stopping results'))

# Diagnostic decomposition, deliberately using the unchanged native primitive.
# Production-efficiency claim is forbidden; clone/proposal overhead is charged.
def proposal(e,x,c):
 b=e.clone();q=e.clone();b.tick(x,None);q.tick(x,c)
 delta=[getattr(q.bank,k)-getattr(b.bank,k) for k in ('fast','slow')]
 return b,q,delta

def apply(e,b,q,delta,g):
 for k,a in zip(('fast','slow'),delta):getattr(e.bank,k)[:]=getattr(b.bank,k)+a*(g[:,:,None] if a.ndim==3 else g)
 e.bank.adapt[:]=b.bank.adapt;e.signs[:]=q.signs;e.time=q.time;e.events=q.events
 e.write_l1+=float(sum(abs(a*(g[:,:,None] if a.ndim==3 else g)).sum() for a in delta));e.slow_write_l1+=float(abs(delta[1]*g).sum())

def protected_tick(e,x,c,g):
 b,q,delta=proposal(e,x,c);apply(e,b,q,delta,g)

def paired_protection(seed,lid):
 ks=keys(seed);old,new=values(seed,lid);eng={a:Engine(5177,seed,False) for a in ('CREDIT','UNIFORM','SHUFFLED')};x,_=encode(eng['CREDIT'],ks,True)
 h=np.zeros((9,5177));at=np.zeros_like(h);pool={a:[] for a in eng};names=[];labels=[];logs=[]
 def snap(name,ids,vals):
  names.append(name);labels.append(vals)
  for a,e in eng.items():pool[a].append(probe(e,x[ids]))
 def teach_group(vals,offset,n):
  nonlocal h,at
  for pas in range(14):
   for idx in np.random.default_rng(7000+pas+n//252).permutation(18):
    check();k=offset+idx
    for e in eng.values():e.begin()
    strength=(1/16)*(1/16+(15/16)*min(n/1024,1))
    for slot,token in enumerate(vals[idx]+[256]):
     activity=x[k,slot];e=eng['CREDIT'];b=e.bank;mask=np.ones(9,dtype=bool)
     if token==256:mask[:8]=False
     active=np.flatnonzero(activity);now=e.time+DT
     h[:,active]*=np.exp(-(now-at[:,active])/86400);at[:,active]=now
     m=e.margins(activity)[0];t=targets(token)
     aa=1-(1-b.adapt[:,active])*np.exp(-DT/b.ta);end=1-(1-aa*np.exp(-.25*activity[active]))*np.exp(-5/b.ta);dx=.5*(aa+end)*activity[active]
     s=dx*(b.Q[active,2]+b.Q[active,3])[None,:]/2*(b.fast[:,active,1]*np.exp(-DT/b.tf)+b.slow[:,active]*np.exp(-DT/b.ts))
     credit=(t*m>0)[:,None]*np.tanh(np.maximum(0,t[:,None]*s)/2)*mask[:,None]
     g=np.ones_like(h);g[:,active]=1/(1+h[:,active]);pg={};norm={};res={}
     for a,q in eng.items():
      res[a]=residual(q.margins(activity)[0],token,strength);pg[a]=proposal(q,activity,res[a]);ds=pg[a][2];norm[a]=abs(ds[0]).sum(-1)+abs(ds[1])
     target=(g*norm['CREDIT']).sum(1);un=norm['UNIFORM'].sum(1);rho=np.minimum(1,np.divide(target,un,out=np.zeros(9),where=un>0));gu=np.repeat(rho[:,None],5177,1)
     gs=g.copy()
     for side in (0,1):
      ids=active[b.kc_side[active]==side]
      if len(ids)>1:gs[:,ids]=np.roll(g[:,ids],1,axis=1)
     for a,gate in [('CREDIT',g),('UNIFORM',gu),('SHUFFLED',gs)]:apply(eng[a],*pg[a],gate)
     h[:,active]+=(1-h[:,active])*credit
     logs.append(dict(n=n,slot=slot,a=strength,target_L1=target.tolist(),uniform_L1=(gu*norm['UNIFORM']).sum(1).tolist(),shuffle_L1=(gs*norm['SHUFFLED']).sum(1).tolist(),proposed_L1={a:v.sum(1).tolist() for a,v in norm.items()},residual={a:v.tolist() for a,v in res.items()},credit_mean=float(h[:,active].mean()),gate_mean=float(g[:,active].mean())))
     C.events+=3
    for a,q in eng.items():q.end(ks[k]);C.records+=1
    n+=1
   if pas+1 in (1,4):
    snap(('OLD' if offset==0 else 'NEW')+f'_pass{pas+1}',slice(offset,offset+18),vals)
  return n
 n=teach_group(old,0,0);snap('OLD_end',slice(0,18),old)
 for e in eng.values():e.rest(3600)
 snap('OLD_rest1',slice(0,18),old);shadows={a:e.clone() for a,e in eng.items()}
 n=teach_group(new,18,n);snap('OLD_NEW_end',slice(0,18),old);snap('NEW_end',slice(18,36),new)
 for e in eng.values():e.rest(3600)
 snap('OLD_final',slice(0,18),old);snap('NEW_final',slice(18,36),new)
 for a,e in shadows.items():
  teach(e,x[18:36],new,252,False,ks[18:36]);e.rest(3600)
  pool[a]+=[probe(e,x[:18]),probe(e,x[18:36])]
 for name,vals in [('OLD_S_NEW',old),('NEW_S_NEW',new)]:names.append(name);labels.append(vals)
 for a,e in eng.items():
  tag=f'{seed}_{lid}_1_{a}';np.savez_compressed(P/'traces'/f'{tag}.npz',**{f'D{i}':v for i,v in enumerate(pool[a])});save(f'traces/{tag}.json',dict(seed=seed,life=lid,STD=True,arm=a,pools=names,labels=labels));checkpoint(e,tag+'_FINAL.npz');evaluate(tag)
 np.savez_compressed(P/'checkpoints'/f'{seed}_{lid}_credit.npz',h=h,at=at)
 save(f'cycles/{seed}_{lid}_write_ledger.json',logs)

def cycle3():
 phase_start=time.process_time();freeze=json.loads((P/'HEAD_FREEZE.json').read_text());assert all(sha(P/'heads'/n)==v for n,v in freeze.items())
 oldkeys=set()
 for f in (OLD/'traces').glob('*.json'):
  if f.stem.endswith('_decoded'):continue
  oldkeys.update(json.loads(f.read_text()).get('keys_hex',[]))
 assert all(k.hex() not in oldkeys for seed in range(202610239301,202610239305) for k in keys(seed))
 for seed in range(202610239301,202610239305):
  for lid in (0,1):
   for std in (False,True):
    for arm in ('W','N','SH'):
     life(seed,lid,std,arm,False);evaluate(f'{seed}_{lid}_{int(std)}_{arm}')
   paired_protection(seed,lid);check(f'pair progress {seed}/{lid}')
   if seed==202610239301 and lid==1:
    if resource_stats()['CPU_seconds']+3*(time.process_time()-phase_start)*1.25>1800:raise RuntimeError('RESOURCE_PROFILE_CANNOT_FIT')
 assert C.records==46368 and C.events==231840,(C.records,C.events)
 assert all(sha(P/'heads'/n)==v for n,v in freeze.items())
 save('cycles/CYCLE3_RESULT.json',dict(status='COMPLETE',native_records=C.records,native_events=C.events,head_hashes_unchanged=True))

if __name__=='__main__':
 awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())]);C.awake=awake
 try:
  check('start',admission=True)
  if PHASE==1:cycle1()
  elif PHASE==2:cycle2()
  else:cycle3()
  assert sha(R/'PROJECT_DOCUMENTS/CURRENT_ARITHMETIC_EXPERIMENT_DESIGN.json')==json.loads((P/'AUTHORIZATION.json').read_text())['arithmetic_pointer_sha256']
 except BaseException as ex:
  save(f'cycles/CYCLE{PHASE}_FAILURE.json',dict(error=repr(ex),records=C.records,events=C.events));raise
 finally:awake.terminate();awake.wait();save('RESOURCE_LEDGER.json',resource_stats())
