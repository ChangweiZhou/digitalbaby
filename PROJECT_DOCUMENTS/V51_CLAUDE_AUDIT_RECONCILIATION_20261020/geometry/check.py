import os,sys,time,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
R=Path('/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929');D=R/'minifly_v5_1_dev_20261020'
sys.path.insert(0,str(D))
from v51_common import np,fresh_bank
import numba
numba.config.CACHE_DIR='/tmp/v51_claude_audit_geometry_20261020/numba'
os.environ['NUMBA_CACHE_DIR']=numba.config.CACHE_DIR
from representation import Codes,Sequence,Geometry,BITS
start=time.process_time();out={'scope':'encoding-only diagnostic; perfect identity/classes are evaluator fixture, not a learned capability','native_value_teaching_events':0,'calibration_byte_cap_per_seed':2048,'results':[]}
alpha=sorted(set(b'0123456789+='))
rng=np.random.default_rng(8100)
records=[]
for i in range(400):
 ds=rng.integers(1,5,2);a,b=[int(rng.integers(0 if d==1 else 10**(d-1),10**d)) for d in ds]
 records.append((f'{a}+{b}='.encode(),str(a+b).encode()))

def state(codes,p):
 s=Sequence(codes)
 for b in p:s.feed(b)
 s.begin_reply();return s

def metrics(g,codes):
 xs=[];ties=[[],[]];halves=[];operand=[];common=[];zero_operand=[]
 for p,a in records[:64]:
  s=state(codes,p);x,raw=g.encode(s);active=np.flatnonzero(x);xs.append(x)
  halves.append(float(np.mean(active>=g.n//2)))
  f=s.feature();contrib=f[g.index]*g.weights
  # Perfect class fixture puts second and first operand in generic closed
  # registers 1 and 3 at the first answer frame. This annotation is evaluator-only.
  slots=g.index//(16*73);pos=g.index<5*16*73
  op=pos&np.isin(slots,[1,3]);com=g.index>=5*16*73
  den=np.sum(abs(contrib[active]))
  operand.append(float(np.sum(abs(contrib[active])*op[active])/max(den,1e-30)))
  common.append(float(np.sum(abs(contrib[active])*com[active])/max(den,1e-30)))
  zero_operand.append(float(np.mean(np.sum(abs(contrib[active])*op[active],axis=1)==0)))
  for side in (0,1):
   chosen=np.flatnonzero((g.side==side)&(x>0));cut=float(np.min(raw[chosen]))
   avail=int(np.sum((g.side==side)&(raw>0)&(raw==cut)))
   picked=int(np.sum(raw[chosen]==cut))
   ties[side].append([len(chosen),cut,picked,avail])
 X=np.array(xs);inter=X@X.T;k=X.sum(1)
 pair=(inter/np.minimum(k[:,None],k[None,:]))[np.triu_indices(len(X),1)]
 return {'active_count':k.tolist(),'mean_pairwise_intersection_fraction':float(np.mean(pair)),
 'all_prompts_common_fraction':float(np.sum(np.all(X>0,axis=0))/np.mean(k)),
 'mean_active_clock_last_half_fraction':float(np.mean(halves)),
 'mean_operand_afferent_absolute_drive_fraction':float(np.mean(operand)),
 'mean_clock_last_afferent_absolute_drive_fraction':float(np.mean(common)),
 'mean_active_without_operand_afferent_drive':float(np.mean(zero_operand)),
 'cutoff_ties_first_prompt_side0':ties[0][0],'cutoff_ties_first_prompt_side1':ties[1][0],
 'mean_selected_cutoff_tie_cells_per_side':[float(np.mean([r[2] for r in t])) for t in ties],
 'mean_available_cutoff_tie_cells_per_side':[float(np.mean([r[3] for r in t])) for t in ties]}

for seed in (202610205101,202610205102):
 bank=fresh_bank(65536,seed);codes=Codes(seed)
 for i,b in enumerate(alpha):codes.w[i]=BITS[b];codes.classes[i]=0 if 48<=b<=57 else 1
 codes.identity_frozen=codes.context_frozen=True
 assert len(set(codes.winner(b) for b in alpha))==12
 g=Geometry(65536,seed+17,bank.kc_side,bank.active_fraction,True)
 row={'seed':seed,'active_fraction':bank.active_fraction,'side_counts':[int(np.sum(bank.kc_side==s)) for s in (0,1)],'before':metrics(g,codes)}
 heard=0;complete=0
 for p,a in records:
  s=Sequence(codes)
  for b in p+a:
   if heard>=2048:break
   s.feed(b);g.calibrate(s);heard+=1
  if heard>=2048:break
  complete+=1
  if time.process_time()-start>25:break
 row.update(calibration_payload_bytes=heard,complete_calibration_records=complete,after=metrics(g,codes),theta_max=float(g.theta.max()),nonzero_theta=int(np.sum(g.theta>0)))
 out['results'].append(row)
 print(json.dumps({'seed':seed,'before_overlap':row['before']['mean_pairwise_intersection_fraction'],'after_overlap':row['after']['mean_pairwise_intersection_fraction'],'CPU':time.process_time()-start}),flush=True)
 if time.process_time()-start>30:break
out['CPU_seconds']=time.process_time()-start
out['source_hashes']={str(D/f):hashlib.sha256((D/f).read_bytes()).hexdigest() for f in ('representation.py','engine.py','cycle2.py','v51_core.py')}
Path('/tmp/v51_claude_audit_geometry_20261020/RESULT.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
