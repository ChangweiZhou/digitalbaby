import os,sys,time,json,hashlib,copy,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
sys.path.insert(0,str(R/'minifly_v6_r1_dev_20261023'))
import core as K
import v6_data as D
import numpy as np
start=time.process_time();awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
def save(n,x):(P/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def seed(i):return int.from_bytes(hashlib.sha256(f'NATIVE_NEXT_V1_20261023/B|{i}|manifest'.encode()).digest()[:4],'big')
def rho(P,M):
 r=min(1.,(P+.5)/(M+.5));p=(P+.5)/(P+M+1);return r,p+(1-p)*r
def fly(seed,scaled):
 f=K.Fly(seed)
 if scaled:
  for w in [f.c.Wv,f.c.Wg]:w[:,:259]*=np.sqrt(348);w[:,259:347]*=np.sqrt(348/88)
 return f
def code(f,p,t):
 f.c.rng=np.random.default_rng(t);f.c.reset();f.c.step(256,False)
 for b in p:f.c.step(b,False)
 f.c.step(257,False);return f.code(),f.c.h.copy()
def jac(a,b):
 u=np.count_nonzero(a|b);return np.count_nonzero(a&b)/u if u else 0.
def run():
 # Cycle 1: prospective equation/count/source audit; no assessment learning.
 rows=[]
 for pp,mm in [(0,0),(1,4),(18,72),(252,1008)]:
  rr,bb=rho(pp,mm);assert 0<rr<=1 and rr<=bb<=1;rows.append(dict(P=pp,M=mm,rho=rr,beta=bb))
 assets=R/'minifly_address_capacity_v1_dev_20261023';lock=json.loads(subprocess.check_output(['git','show','HEAD:minifly_address_capacity_v1_dev_20261023/PRE_VALUE_ASSET_LOCK.json']))
 checks=[]
 for name,sha in lock.items():
  if name.startswith('assets/'):
   p=assets/name;assert p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==sha;checks.append(name)
 save('CYCLE1.json',dict(status='PASS_STATIC_AND_ASSET_QUALIFICATION',EOS_equations=rows,expanded_assets_hash_verified=checks,new_native_records=0,revision='Resolve actual upstream package locations and qualified Python environment before native tests; no scientific parameter change.'))
 # Cycle 2: fixed birth supports on OLD prompts only.
 results=[];fixtures=[];partial=0
 for world in [0,1]:
  s=seed(world);data=D.build(s);save(f'manifest_{world}.json',data)
  for scaled in [False,True]:
   f=fly(s,scaled);groups=[];anchors=[]
   for length in [1,4]:
    prompts=[bytes(x['prompt']) for x in data['old'] if all(len(str(a))==length for a in x['operands'])];assert len(prompts)==8
    xs=[]
    for i,p in enumerate(prompts):
     if i%2:p=bytes(D.flip(next(x for x in data['old'] if bytes(x['prompt'])==p))['prompt'])
     xs.append([code(f,p,9000+t)[0]>0 for t in range(3)])
    xs=np.array(xs);same=[jac(xs[i,t],xs[i,u]) for i in range(8) for t in range(3) for u in range(t)]
    other=[jac(xs[i,t],xs[j,u]) for i in range(8) for j in range(8) if i!=j for t in range(3) for u in range(3) if t!=u]
    accuracy=[]
    for i in range(8):
     for t in range(3):
      scores=np.array([np.mean([jac(xs[i,t],xs[j,u]) for u in range(3) if u!=t]) for j in range(8)]);best=scores==scores.max();accuracy.append(float(best[i])/best.sum())
    groups.append(dict(length=length,J_same=float(np.mean(same)),J_other=float(np.mean(other)),DeltaJ=float(np.mean(same)-np.mean(other)),ID_accuracy=float(np.mean(accuracy)),min_active=int(xs.sum(-1).min())))
    anchors.extend(xs[:4])
   for comp in [0,8]:
    vals=[]
    for i,x in enumerate(anchors):
     for sign in [-1,1]:
      e=f.e.clone();passive=f.e.clone();c=np.zeros(9);c[comp]=sign/512
      e.tick(x[0].astype(float),c);passive.tick(x[0].astype(float),None);partial+=1
      effects=[]
      for j,y in enumerate(anchors):effects.append(sign*float((e.margins(y[1].astype(float))[0]-passive.margins(y[1].astype(float))[0])[comp]))
      # Compare only equal-length neighbors.
      ix=list(range(0,4)) if i<4 else list(range(4,8));vals.append(effects[i]-float(np.mean([effects[j] for j in ix if j!=i])))
    fixtures.append(dict(world=world,scaled=scaled,compartment=comp,self_minus_other=float(np.mean(vals))))
   results.append(dict(world=world,seed=s,scaled=scaled,groups=groups))
   print('birth',world,scaled,groups,flush=True)
 gates=[]
 for world in [0,1]:
  rr=next(x for x in results if x['world']==world and x['scaled']);ff=[x for x in fixtures if x['world']==world and x['scaled']]
  gates.append(all(g['DeltaJ']>0 and g['ID_accuracy']>.125 and g['min_active']>0 for g in rr['groups']) and all(x['self_minus_other']>0 for x in ff))
 save('CYCLE2.json',dict(status='ADMITTED' if all(gates) else 'INPUT_STATE_NOT_ADMITTED',world_gates=gates,results=results,native_margin_fixtures=fixtures,partial_native_events=partial,new_native_records=0,revision='Preserve failed gate without retuning; no HELD answers or full arithmetic roster opened.'))
 # Cycle 3: real canonical EOS-only coefficient comparison and passive telemetry.
 f=fly(771,False);base=f.e.clone();bal=f.e.clone();uni=f.e.clone();passive=f.e.clone();events=0;tele=[];pc=mc=0
 x=code(f,b'generic',77)[0]
 for record in range(4):
  rr,bb=rho(pc,mc)
  for target in [17,250,0,61,256]:
   m=base.margins(x)[0];coef=K.residual(m,target,K.ramp(record)/16)
   prev=base.clone();c1=coef.copy();c2=coef.copy();c1[8]*=1 if target==256 else rr;c2[8]*=bb
   for e,c in [(base,coef),(bal,c1),(uni,c2),(passive,None)]:e.tick(x,c);events+=1
   assert np.array_equal(base.bank.fast[:8],bal.bank.fast[:8]) and np.array_equal(base.bank.fast[:8],uni.bank.fast[:8])
   assert np.array_equal(base.bank.slow[:8],bal.bank.slow[:8]) and np.array_equal(base.bank.slow[:8],uni.bank.slow[:8])
   # A pure read-only telemetry operation must leave the actual state unchanged.
   dg=hashlib.sha256(base.bank.fast.tobytes()+base.bank.slow.tobytes()).hexdigest();q=prev.clone();q.tick(x,None)
   increment=np.r_[ (base.bank.fast-q.bank.fast).ravel(),(base.bank.slow-q.bank.slow).ravel()]
   tele.append(dict(target=target,rho=rr,beta=bb,applied_vs_passive_L1=float(np.abs(increment).sum()),applied_vs_passive_L2=float(np.linalg.norm(increment))))
   assert dg==hashlib.sha256(base.bank.fast.tobytes()+base.bank.slow.tobytes()).hexdigest()
  pc+=1;mc+=4
 save('CYCLE3.json',dict(status='PASS_BOUNDED_NATIVE_COMPONENT_TEST',synthetic_record_applications=16,observed_native_events=events,payload_states_exactly_equal=True,telemetry=tele,full_EOS_assessment='NOT_ADMITTED_WITHOUT_A_CONFIRMATION',A_confirmation='NOT_LAUNCHED',A_full_signed_clipping_instrumentation='NOT_YET_FULLY_QUALIFIED',revision='Separate observational delta telemetry from mechanistic signed/clipping decomposition; do not present limited fixtures as formal A qualification.'))
 save('SUMMARY.json',dict(status='THREE_BOUNDED_CYCLES_COMPLETE',B_admitted=all(gates),A_formal_confirmation=False,C_full_assessment=False,new_scientific_jobs=0,synthetic_record_applications=16,partial_native_events=partial,CPU_seconds=time.process_time()-start))
try:run()
except Exception as e:save('FAILURE.json',dict(error=repr(e),CPU_seconds=time.process_time()-start));raise
finally:awake.terminate()
