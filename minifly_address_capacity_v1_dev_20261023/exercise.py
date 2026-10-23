"""Bounded ADDRESS-CAPACITY exercise; no head fitting or arithmetic edits."""
from rc_common import *
import rc_common as C
import engine as E
from scipy.sparse import csc_matrix,hstack,save_npz,load_npz
import head as H
from decode_saved import metric,infer,evaluate
SOURCE=R/'minifly_stop_only_confirm_v1_20261023'
SIZE=5177
ORIGINAL=E.fresh_bank

def jsonout(name,v):save(name,v)
def feature_infer(D,donor,hs):
 real=infer(D,True,hs);eos=infer(donor,True,hs)
 real[:,:,8]=eos[:,:,8];assert np.array_equal(real[:,:,:8]>0,D.mean(-1)[:,:,:8]>0)
 return real

def saved_controls():
 start=time.process_time();rows=[]
 for seed in range(202610239601,202610239633):
  for lid in (0,1):
   tag=f'{seed}_{lid}_1_W';meta=json.loads((SOURCE/'traces'/f'{tag}.json').read_text())
   with np.load(SOURCE/'traces'/f'{tag}.npz') as f:
    for i,(pool,vals) in enumerate(zip(meta['pools'],meta['labels'])):
     D=f[f'D{i}'];base=D.mean(-1);zs={'DIRECT':base}
     for length in range(9):
      z=base.copy();z[:,:,8]=-1;z[:,length,8]=1;zs[f'FIXED_{length}']=z
     for hs in range(202610239401,202610239404):
      zs[f'STOP_{hs}']=infer(D,True,hs)
      zs[f'POSITION_{hs}']=feature_infer(D,np.zeros_like(D),hs)
      zs[f'FOREIGN_{hs}']=feature_infer(D,np.roll(D,1,axis=0),hs)
     eligibility=None
     for name,z in zs.items():
      assert np.array_equal(z[:,:,:8]>0,base[:,:,:8]>0)
      m=metric(z,vals)
      if eligibility is None:eligibility=m['conditional_n']
      assert m['conditional_n']==eligibility
      rows.append(dict(pair=seed,life=lid,pool=pool,readout=name,metrics=m))
   assert time.process_time()-start<300
 jsonout('SAVED_CONTROL_ROWS.json',rows)
 # Equal life weighting, then equal pair weighting; no winner selection.
 table=[]
 for pool in sorted({x['pool'] for x in rows}):
  for group in ['DIRECT']+[f'FIXED_{j}' for j in range(9)]+['STOP','POSITION','FOREIGN']:
   rr=[x for x in rows if x['pool']==pool and (x['readout']==group or x['readout'].startswith(group+'_'))]
   vals=[x['metrics']['nonempty_exact'] for x in rr];q=[x['metrics']['conditional_stop'] for x in rr if x['metrics']['conditional_stop'] is not None]
   table.append(dict(pool=pool,readout=group,complete_mean=float(np.mean(vals)) if vals[0] is not None else None,conditional_mean=float(np.mean(q)) if q else None,undefined=sum(x['metrics']['conditional_stop'] is None for x in rr)))
 jsonout('SAVED_CONTROL_SUMMARY.json',table)
 jsonout('cycles/CYCLE1.json',dict(status='PASS',audit='payload signs and real-payload denominator identical across readouts; all fixed lengths reported',test='all STD W saved checkpoints, 32 pairs/two lives/three frozen heads',revision='foreign EOS input separated from real payload; unchanged confirmation verdict',new_native_events=0,CPU_seconds=time.process_time()-start))

def build():
 b=ORIGINAL(5177,0);rng=np.random.default_rng(202610238201);B=b.B.tocsc();cols=[B];donors=[np.arange(5177)]
 freqs={}
 for side in (0,1):
  freq=np.asarray((B[:,b.kc_side==side]!=0).sum(axis=1)).ravel();ix=np.flatnonzero(freq);freqs[side]=(ix,freq[ix]/freq[ix].sum())
 for block in range(3):
  order=np.empty(5177,dtype=int);newcols=[]
  for side in (0,1):
   ix=np.flatnonzero(b.kc_side==side);order[ix]=rng.permutation(ix)
  for i,d in enumerate(order):
   side=b.kc_side[d];ix,prob=freqs[side];weights=B[:,d].data.copy();chosen=rng.choice(ix,len(weights),replace=False,p=prob)
   newcols.append(csc_matrix((rng.permutation(weights),(chosen,np.zeros(len(chosen),dtype=int))),shape=(B.shape[0],1)))
  cols.append(hstack(newcols,format='csc'));donors.append(order)
 expanded=hstack(cols,format='csr');ds=np.concatenate(donors)
 assert expanded.shape[1]==20708 and expanded.nnz==4*B.nnz
 save_npz(P/'assets/expanded_B.npz',expanded);np.save(P/'assets/donors.npy',ds)
 v=np.load(SOURCE/'assets/calibration.npz')['receptors'];z=np.asarray(expanded.T@v[:,b.pn_type_index].T);mu=z.mean(1);sigma=z.std(1);floor=float(np.load(SOURCE/'assets/calibration.npz')['floor'])
 orig=np.load(SOURCE/'assets/calibration.npz');assert np.allclose(mu[:5177],orig['mu'],rtol=1e-12,atol=1e-12)
 mu[:5177]=orig['mu'];sigma[:5177]=orig['sigma'];np.savez(P/'assets/calibration_20708.npz',mu=mu,sigma=sigma,floor=floor,receptors=v)
 def activities(mat,side,mu,sigma):
  a=np.asarray(mat.T@v[:,b.pn_type_index].T).T;a=(a-mu)/np.maximum(sigma,floor);out=np.zeros_like(a)
  for row,x in zip(a,out):
   for s in (0,1):
    ix=np.flatnonzero((side==s)&(row>0));x[ix[np.lexsort((ix,-row[ix]))[:b.side_n[s]]]]=1
  return out
 x0=activities(B,b.kc_side,orig['mu'],orig['sigma']);xn=activities(expanded,b.kc_side[ds],mu,sigma)
 gains={}
 for name in ('Q','reader_Q'):
  q=getattr(b,name);g=(x0@q).mean(0)/(xn@q[ds]).mean(0);assert np.all(np.isfinite(g)) and np.all(g>0);gains[name]=g.tolist()
  np.save(P/'assets'/f'expanded_{name}.npy',q[ds]*g)
 np.save(P/'assets/calibration_x0.npy',x0[:8]);np.save(P/'assets/calibration_xn.npy',xn[:8])
 jsonout('CONSTRUCTION.json',dict(N0=5177,N=20708,edges0=B.nnz,edges=expanded.nnz,K=b.side_n,gains=gains,active_mean0=x0.sum(1).mean(),active_meanN=xn.sum(1).mean(),effective_N0=float(x0.sum(0).sum()**2/np.square(x0.sum(0)).sum()),effective_N=float(xn.sum(0).sum()**2/np.square(xn.sum(0)).sum())))

def bank(n,seed):
 b=ORIGINAL(5177,seed)
 if SIZE==5177:return b
 ds=np.load(P/'assets/donors.npy');b.B=load_npz(P/'assets/expanded_B.npz');b.kc_side=b.kc_side[ds];b.T=b.T[ds].copy()
 for key in ('Q','reader_Q'):setattr(b,key,np.load(P/'assets'/f'expanded_{key}.npy'))
 b.fast=np.zeros((9,SIZE,2));b.slow=np.zeros((9,SIZE));b.adapt=np.ones((9,SIZE));return b
E.fresh_bank=bank
C.Engine=lambda n,seed,gated=False:E.Engine(SIZE,seed,gated)

def qualify():
 global SIZE
 build();rows=[]
 for SIZE in (5177,20708):
  xs=np.load(P/'assets'/('calibration_x0.npy' if SIZE==5177 else 'calibration_xn.npy'))
  e=E.Engine(SIZE,0,False);assert e.bank.B.shape[1]==e.q.shape[1]==SIZE
  before=digest(e);probe(e,np.array([xs]*9).transpose(1,0,2));assert before==digest(e)
  for i,x in enumerate(xs):
   for c in range(9):
    for sign in (-1,1):
     a=E.Engine(SIZE,0,False);passive=a.clone();coeff=np.zeros(9);coeff[c]=sign/512
     a.tick(x,coeff);passive.tick(x);C.events+=1
     changes=[]
     for target in xs:changes.append((a.margins(target)[0]-passive.margins(target)[0]).tolist())
     rows.append(dict(N=SIZE,address=i,compartment=c,coefficient=float(coeff[c]),changes=changes))
  for token in range(257):assert decode(targets(token))==token
 jsonout('CALIBRATION_TRANSFER.json',rows)
 jsonout('cycles/CYCLE2.json',dict(status='PASS',test='constructed dimensions, finite gains, deterministic counts, read-only, codec and 288 partial self/cross-write events',revision='expanded B and all bookkeeping dimensions now explicit; same side_n preserved',partial_native_events=C.events,records=0))

def pilot():
 global SIZE
 # Final pre-value checkpoint qualification: two full one-byte synthetic records.
 for SIZE in (5177,20708):
  e=E.Engine(SIZE,0,False);x=np.load(P/'assets'/('calibration_x0.npy' if SIZE==5177 else 'calibration_xn.npy'))[0]
  for token in (32,256):e.tick(x,residual(e.margins(x)[0],token,1/256))
  checkpoint(e,f'E0_{SIZE}.npz');other=E.Engine(SIZE,0,False)
  with np.load(P/'checkpoints'/f'E0_{SIZE}.npz') as z:
   for key in ('fast','slow','adapt'):getattr(other.bank,key)[:]=z[key]
   for key in ('q','q_at','last_at','ring','ring_pos','signs'):getattr(other,key)[:]=z[key]
   other.time=float(z['time'][0]);other.events=int(z['time'][1]);other.write_l1=float(z['time'][2]);other.slow_write_l1=float(z['time'][3]);other.confirmations=int(z['time'][4])
  assert digest(e)==digest(other)
  assert np.array_equal(e.margins(x)[0],other.margins(x)[0])
 jsonout('FINAL_QUALIFICATION.json',dict(full_synthetic_records=2,full_synthetic_events=4,restore_all_fields=True,partial_events_preceding=288))
 closure={str(Path(m.__file__).resolve()):sha(Path(m.__file__)) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and Path(m.__file__).suffix=='.py' and str(Path(m.__file__).resolve()).startswith(str(R))}
 jsonout('RUNTIME_MODULE_CLOSURE.json',closure)
 jsonout('PRE_VALUE_ASSET_LOCK.json',{str(f.relative_to(P)):sha(f) for d in ('assets','heads') for f in (P/d).glob('*') if f.is_file()})
 # Exact key collision audit before generating any fresh values.
 prior=set()
 for root in R.glob('minifly*recoverability*dev*'):
  for f in (root/'traces').glob('*.json'):
   d=json.loads(f.read_text());prior.update(d.get('keys_hex',[]))
 for f in (SOURCE/'traces').glob('*.json'):
  d=json.loads(f.read_text());prior.update(d.get('keys_hex',[]))
 seen=set()
 for seed in range(202610238101,202610238109):
  kk={x.hex() for x in keys(seed)};assert not (kk&prior or kk&seen);seen|=kk
 jsonout('KEY_AUDIT.json',dict(unique_keys=len(seen),prior_keys=len(prior),pass_=True))
 start=time.process_time()
 for seed in range(202610238101,202610238109):
  for lid in (0,1):
   for SIZE in (5177,20708):
    for arm in ('W','N','SH'):
     C.life(seed,lid,True,arm);evaluate(f'{SIZE}_{seed}_{lid}_1_{arm}')
  check(f'pair {seed} complete')
  if seed==202610238101 and resource_stats()['CPU_seconds']+7*(time.process_time()-start)*1.25>3300:raise RuntimeError('PROFILE_CANNOT_FIT_ROSTER')
 assert C.records==56448 and C.events==282240
 jsonout('cycles/CYCLE3.json',dict(status='COMPLETE',records=C.records,events=C.events,revision='No post-outcome changes; native E1 pilot only',resources=resource_stats()))

if __name__=='__main__':
 C.awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
 try:
  if PHASE==1:saved_controls()
  elif PHASE==2:qualify()
  elif PHASE==3:pilot()
 except BaseException as ex:jsonout(f'CYCLE{PHASE}_FAILURE.json',dict(error=repr(ex),records=C.records,events=C.events,resources=resource_stats()));raise
 finally:
  C.awake.terminate();C.awake.wait();jsonout('RESOURCE_LEDGER.json',resource_stats())
