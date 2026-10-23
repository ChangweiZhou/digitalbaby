"""Three foreground bounded cycles, original pilot roster and cumulative budget."""
from rc_common import *
import rc_common as common
import head as H

def headhash(p):
 h=hashlib.sha256()
 for k in H.NAMES:h.update(p[k].tobytes())
 return h.hexdigest()
def cycle1():
 calibrate();e=Engine(5177,202610229501,False);ks=keys(202610229501);x,_=encode(e,ks[:2],False);ref=e.bank.clone();errors=[]
 for i in range(4):
  a=x[0,i];D=read_one(e,x[0]);q=e.clone();q.begin();errors.append(float(abs(D.mean(-1)[i]-q.margins(a)[0]).max()))
  c=np.linspace(-.001,.001,9) if i%2 else np.zeros(9)
  e.tick(a,c);common.events+=1;event_batch(ref.fast,ref.slow,ref.adapt,a,DT,c,True,ref.Q,ref.T,ref.F,ref.MM,ref.k0,ref.fw0,ref.fwd,ref.ta,ref.tg,ref.tf,ref.ts,ref.fraction,ref.alpha_scale,ref.genes,ref.sf,ref.ss)
  for k in ('fast','slow','adapt'):assert np.max(abs(getattr(e.bank,k)-getattr(ref,k)))<1e-12
 assert max(errors)<1e-12
 before=digest(e);probe(e,x);assert digest(e)==before
 checkpoint(e,'E0_restore.npz');r=Engine(5177,202610229501,False)
 with np.load(P/'checkpoints/E0_restore.npz') as z:
  for k in ('fast','slow','adapt'):getattr(r.bank,k)[:]=z[k]
  for k in ('q','q_at','last_at','ring','ring_pos','signs'):getattr(r,k)[:]=z[k]
  r.time=float(z['time'][0]);r.events=int(z['time'][1]);r.write_l1=float(z['time'][2]);r.slow_write_l1=float(z['time'][3]);r.confirmations=int(z['time'][4])
 assert digest(e)==digest(r);assert np.array_equal(read_one(e,x[0]),read_one(r,x[0]))
 # Meaningful gradient audit: nonzero residual branch, empty and length-eight masks.
 rng=np.random.default_rng(202610229501);D=rng.normal(size=(2,9,9,2));v=[[],list(range(8))];y,w=head_targets(v);scale=np.ones(9)
 p=initialize(202610229501);assert sum(a.size for a in p.values())==5801
 for view in ('MEAN','SEP'):
  X,base=features(D,scale,view);logits,_=forward(p,X,base);assert np.array_equal(logits,base)
 p['w2'][:]=rng.normal(0,.05,p['w2'].shape);X,base=features(D,scale,'SEP');loss,g=loss_grad(p,X,base,y,w);errs=[]
 for k in H.NAMES:
  for _ in range(8):
   idx=tuple(int(rng.integers(n)) for n in p[k].shape);original=p[k][idx];p[k][idx]=original+1e-6;l1=loss_grad(p,X,base,y,w)[0];p[k][idx]=original-1e-6;l0=loss_grad(p,X,base,y,w)[0];p[k][idx]=original;errs.append(abs((l1-l0)/2e-6-g[k][idx]))
 assert max(errs)<1e-7
 assert all(decode(targets(t))==t for t in range(257));assert np.all(w[0,1:]==0) and np.all(w[1,:8,:8]>0)
 save('cycles/CYCLE1_RESULT.json',dict(status='PASS',native_partial_events=4,full_records=0,codec257=True,restore_readonly=True,parameters=5801,mean_component_error=max(errors),gradient_max_absolute_error=max(errs),arithmetic_pointer_unchanged=sha(R/'PROJECT_DOCUMENTS/CURRENT_ARITHMETIC_EXPERIMENT_DESIGN.json')==json.loads((P/'AUTHORIZATION.json').read_text())['arithmetic_pointer_sha256']))

def load_training(std):
 Ds=[];Ys=[];Ws=[]
 for seed in range(202610229101,202610229113):
  meta=json.loads((P/'traces'/f'{seed}_0_{int(std)}_W.json').read_text())
  with np.load(P/'traces'/f'{seed}_0_{int(std)}_W.npz') as z:
   for i,vals in enumerate(meta['labels']):
    assert len(vals)==18;Ds.append(z[f'D{i}']);y,w=head_targets(vals);Ys.append(y);Ws.append(w)
 return np.concatenate(Ds),np.concatenate(Ys),np.concatenate(Ws)

def fit_heads():
 for std in (False,True):
  D,Y,W=load_training(std);scale=np.maximum(np.sqrt(np.mean(D*D,axis=(0,1,3))),np.sqrt(np.finfo(float).eps));np.save(P/'heads'/f'scale_{int(std)}.npy',scale)
  for view in ('MEAN','SEP'):
   X,base=features(D,scale,view)
   for seed in range(202610229401,202610229404):
    check(f'fit {int(std)} {view} {seed}',admission=True);p=initialize(seed);st={k:[np.zeros_like(a),np.zeros_like(a)] for k,a in p.items()};rng=np.random.default_rng(seed+44);terms=updates=0;curve=[]
    for step in range(1,2001):
     check();ids=rng.integers(0,len(D),32);cost=int(np.sum(W[ids]>0))
     if terms+cost>2000000:break
     loss,g=loss_grad(p,X[ids],base[ids],Y[ids],W[ids]);assert np.isfinite(loss);gn=H.adam(p,g,st,step);terms+=cost;updates=step
     if step==1 or step%100==0:curve.append(dict(step=step,loss=loss,gradient_norm=gn,terms=terms))
    loss=loss_grad(p,X,base,Y,W)[0];name=f'{int(std)}_{view}_{seed}';np.savez_compressed(P/'heads'/f'{name}.npz',**p)
    save(f'heads/{name}.json',dict(updates=updates,terms=terms,training_loss=loss,curve=curve,parameter_sha256=headhash(p),native_gradient=False));check(f'fit done {name}: loss {loss:.5f}')
 freeze={q.name:sha(q) for q in (P/'heads').iterdir() if q.is_file()};save('HEAD_FREEZE.json',dict(files=freeze,phase='before all development/assessment values',native_records_completed=parent_records+common.records))

def cycle2():
 for seed in range(202610229101,202610229113):
  for std in (False,True):life(seed,0,std,'W',True)
 assert common.records==12096 and common.events==60480
 fit_heads();save('cycles/CYCLE2_RESULT.json',dict(status='COMPLETE',native_records=common.records,native_events=common.events,trained_heads=12,freeze_sha256=sha(P/'HEAD_FREEZE.json')))

def decode_outputs(z):
 out=[]
 for q in z:
  bs=[];EOS=False
  for j,a in enumerate(q):
   if a[8]>0:EOS=True;break
   if j==8:break
   bs.append(int(sum(1<<k for k in range(8) if a[k]>0)))
  out.append(dict(bytes=bs,EOS=EOS,truncation=not EOS))
 return out

def score(logits,vals,absent=False):
 out=decode_outputs(logits)
 if absent:return dict(outputs=out,empty_emission=float(np.mean([o['EOS'] and not o['bytes'] for o in out])),absence_recognition_claim=False)
 correct=[o['bytes']==v and o['EOS'] for o,v in zip(out,vals)];non=[i for i,v in enumerate(vals) if len(v)>0]
 bit_n=bit_d=0
 for i,v in enumerate(vals):
  for j,t in enumerate(v):bit_n+=int(np.sum((logits[i,j,:8]>0)==((int(t)>>np.arange(8))&1)));bit_d+=8
 return dict(exact=float(np.mean(correct)),nonempty_exact=float(np.mean([correct[i] for i in non])),bits=bit_n/bit_d if bit_d else None,outputs=out,per_length={str(L):float(np.mean([correct[i] for i,v in enumerate(vals) if len(v)==L])) for L in range(9) if any(len(v)==L for v in vals)})

def evaluate_trace(seed,life_id,std,arm):
 tag=f'{seed}_{life_id}_{int(std)}_{arm}';meta=json.loads((P/'traces'/f'{tag}.json').read_text());outputs=[];saved={}
 scale=np.load(P/'heads'/f'scale_{int(std)}.npy')
 with np.load(P/'traces'/f'{tag}.npz') as raw:
  for i,(pool,vals) in enumerate(zip(meta['pools'],meta['labels'])):
   D=raw[f'D{i}'];absent=pool=='ABSENT';outputs.append(dict(pool=pool,readout='DIRECT',lesion='NONE',metrics=score(D.mean(-1),vals,absent)))
   for view in ('MEAN','SEP'):
    for hs in range(202610229401,202610229404):
     with np.load(P/'heads'/f'{int(std)}_{view}_{hs}.npz') as f:p={k:f[k] for k in H.NAMES}
     before=headhash(p)
     lesions=['NONE','ZERO','FOREIGN'] if arm=='W' and not absent else ['NONE']
     for lesion in lesions:
      source=D if lesion=='NONE' else np.zeros_like(D) if lesion=='ZERO' else np.roll(D,1,axis=0)
      X,base=features(source,scale,view);z,_=forward(p,X,base);assert before==headhash(p)
      key=f'logits{len(saved)}';saved[key]=z;outputs.append(dict(pool=pool,readout=view,head_seed=hs,lesion=lesion,array=key,metrics=score(z,vals,absent)))
 np.savez_compressed(P/'traces'/f'{tag}_decoded.npz',**saved);save(f'traces/{tag}_decoded.json',dict(seed=seed,life=life_id,STD=std,arm=arm,pools=meta['pools'],labels=meta['labels'],results=outputs))

def cycle3():
 freeze=json.loads((P/'HEAD_FREEZE.json').read_text());assert all(sha(P/'heads'/n)==h for n,h in freeze['files'].items())
 roster=[(seed,0) for seed in range(202610229201,202610229205)]+[(seed,j) for seed in range(202610229301,202610229305) for j in (0,1)]
 # Check disjoint key identities before any evaluation facts are generated.
 trainkeys={k for seed in range(202610229101,202610229113) for k in keys(seed)};seen=set()
 for seed in range(202610229201,202610229205):
  kk=set(keys(seed));assert not kk&trainkeys and not kk&seen;seen|=kk
 for seed in range(202610229301,202610229305):
  kk=set(keys(seed));assert not kk&trainkeys and not kk&seen;seen|=kk
 save('SPLIT_AUDIT.json',dict(disjoint=True,intentional_assessment_key_reuse_only=True,head_freeze_sha256=sha(P/'HEAD_FREEZE.json')))
 for seed,lid in roster:
  if seed>=202610229301:
   old0,new0=values(seed,0);old1,new1=values(seed,1);save(f'assets/{seed}_counterfactual.json',dict(identical_values=sum(a==b for a,b in zip(old0+new0,old1+new1)),different_lengths=sum(len(a)!=len(b) for a,b in zip(old0+new0,old1+new1))))
  for std in (False,True):
   for arm in ('W','N','SH'):
    life(seed,lid,std,arm,False);evaluate_trace(seed,lid,std,arm)
  if seed==202610229201:
   check('complete development profile')
   # Conservative extrapolation includes observed evaluation and training costs.
   if resource_stats()['CPU_seconds']+11*(time.process_time()-phase3start)*1.25>1800:raise RuntimeError('RESOURCE_PROFILE_CANNOT_FIT_REMAINING_ROSTER')
 assert common.records==42336 and common.events==211680
 assert all(sha(P/'heads'/n)==h for n,h in freeze['files'].items())
 save('cycles/CYCLE3_RESULT.json',dict(status='COMPLETE',native_records=common.records,native_events=common.events,head_weights_unchanged=True,development_episodes=4,assessment_pairs=4,assessment_value_lives=8))

if __name__=='__main__':
 auth=json.loads((P/'AUTHORIZATION.json').read_text());parent_records=sum(json.loads(q.read_text()).get('native_records',0) for q in (P/'cycles').glob('CYCLE*_RESULT.json'))
 awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
 # common.check tracks the same lease.
 import rc_common as common;common.awake=awake
 try:
  check(f'cycle {PHASE} start',admission=True);phase3start=time.process_time()
  # globals imported with * do not track counters rebound in common.
  if PHASE==1:cycle1()
  elif PHASE==2:cycle2()
  else:cycle3()
  assert sha(R/'PROJECT_DOCUMENTS/CURRENT_ARITHMETIC_EXPERIMENT_DESIGN.json')==auth['arithmetic_pointer_sha256']
 except BaseException as ex:
  save(f'cycles/CYCLE{PHASE}_FAILURE.json',dict(error=repr(ex),native_records=common.records,native_events=common.events,resources=resource_stats()));raise
 finally:
  awake.terminate();awake.wait();save('RESOURCE_LEDGER.json',resource_stats())
