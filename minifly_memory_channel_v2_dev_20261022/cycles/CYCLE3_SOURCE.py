"""Bounded technical qualification; never launches the V2 campaign."""
import sys,os,json,time,hashlib,resource,subprocess,importlib.util
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
spec=importlib.util.spec_from_file_location('parent_diagnostic',R/'minifly_transformer_external_memory_v1_r1_20261022/diagnostic.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
import numpy as np
import numba
numba.config.CACHE_DIR=str(P/'scratch');os.environ['NUMBA_CACHE_DIR']=numba.config.CACHE_DIR
from engine import targets
Engine=d.Engine;DT=30/14
START=time.process_time();WALL=time.monotonic();records=0;synthetic=0;events=[];rows=[];arrays={}
def save(n,v):(P/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def budget():
 if time.process_time()-START>600 or time.monotonic()-WALL>900:raise RuntimeError('TECHNICAL_CYCLE_RESOURCE_LIMIT')
def frame(e,seed,count,cal=256,std=False):
 rng=np.random.default_rng(seed);b=e.bank
 train=np.tanh(rng.normal(size=(cal,88)));z=np.asarray(b.B.T@train[:,b.pn_type_index].T);mu=z.mean(1);sigma=z.std(1)
 floor=np.sqrt(np.finfo(float).eps)*np.median(sigma[sigma>0]);assert floor>0
 raw=np.tanh(rng.normal(size=(count,9,88)));out=[]
 for row in raw:
  codes=[]
  for v in row:
   drive=np.asarray(b.B.T@v[b.pn_type_index]).ravel()-mu
   if std:drive/=np.maximum(sigma,floor)
   x=np.zeros(len(drive))
   for side in (0,1):
    ix=np.flatnonzero((b.kc_side==side)&(drive>0));order=np.lexsort((ix,-drive[ix]));x[ix[order[:b.side_n[side]]]]=1
   codes.append(x)
  out.append(codes)
 return np.array(out)
def read(e,x,view):
 before=d.digest(e);v=e.clone();v.begin();out=[]
 for j,a in enumerate(x):
  if view=='TIME':v=e.clone();v.rest(j*DT);v.begin()
  out.append(v.margins(a)[0])
  if view=='SEQ':v.tick(a,None)
 assert d.digest(e)==before
 return np.array(out)
def metrics(m,values,threshold=0):
 exact=bits=den=0;early=late=0;em=[]
 for row,val in zip(m,values):
  out=[];end=False
  for j,z in enumerate(row):
   if z[8]>threshold:end=True;break
   if j==8:break
   out.append(int(np.sum((z[:8]>0)*(1<<np.arange(8)))))
  ok=end and out==list(val);exact+=ok;early+=end and len(out)<len(val);late+=not end or len(out)>len(val)
  for j,t in enumerate(val):bits+=int(np.sum((row[j,:8]>0)==((int(t)>>np.arange(8))&1)));den+=8
  em.append({'bytes':out,'EOS':end,'truncation':not end})
 return {'exact':exact/len(values),'bit_accuracy':bits/den if den else None,'early_EOS':early/len(values),'late_EOS':late/len(values),'outputs':em}
def probe(e,x,vals,label):
 for view in ('SNAP','SEQ','TIME'):
  m=np.array([read(e,k,view) for k in x]);key=f'm{len(rows)}';arrays[key]=m
  rows.append(dict(label=label,view=view,array=key,values=[list(map(int,v)) for v in vals],metrics=metrics(m,vals)))
def teach(e,x,vals,passes,start,enabled=True,add=False):
 global records
 n=start
 for pas in range(passes):
  for i in np.random.default_rng(9900+start+pas).permutation(len(vals)):
   budget();e.begin();a=(1/16)*(1/16+(15/16)*min(n/1024,1))
   for j,t in enumerate(list(vals[i])+[256]):
    m=e.margins(x[i,j])[0];y=targets(int(t));target=-a*y/2;pred=a*np.tanh(m/2)/2
    if t==256:target[:8]=0;pred[:8]=0
    coeff=target+(0 if add else pred)
    events.append(np.r_[n,j,t,a,m,target,pred,coeff if enabled else np.zeros(9)])
    e.tick(x[i,j],coeff if enabled else None)
   e.end(str(i).encode(),write=enabled);n+=1;records+=1
 return n
def checkpoint(e,name):
 a={k:getattr(e,k) for k in ('q','q_at','last_at','ring','ring_pos','signs')}
 a.update(fast=e.bank.fast,slow=e.bank.slow,adapt=e.bank.adapt,time=np.array([e.time,e.events,e.write_l1,e.slow_write_l1,e.confirmations]))
 np.savez_compressed(P/'cycles'/name,**a)
def influence(e,x,name,n):
 global synthetic
 out=[];a=(1/16)*(1/16+(15/16)*min(n/1024,1))
 for key in (0,1):
  for slot in (0,3,4):
   for ch in (0,8):
    for y in (-1,1):
     v=e.clone();v.begin()
     for j in range(slot):v.tick(x[key,j],None);synthetic+=1
     active=v.clone();passive=v.clone();coeff=np.zeros(9);coeff[ch]=a*(np.tanh(v.margins(x[key,slot])[0][ch]/2)-y)/2
     active.tick(x[key,slot],coeff);passive.tick(x[key,slot],None);synthetic+=2
     delta=np.array([read(active,k,'SNAP')-read(passive,k,'SNAP') for k in x]);out.append(delta)
 arrays[name]=np.array(out)
def life(seed,m,cal,std,add,arm,variable=False):
 e=Engine(5177,seed,False);x=frame(e,seed,2*m+8,cal,std)
 rng=np.random.default_rng(seed+77)
 lengths=np.tile(np.arange(9),m//9) if variable else np.full(m,4)
 old=[rng.integers(0,256,int(k),dtype=np.uint8) for k in lengths];new=[rng.integers(0,256,int(k),dtype=np.uint8) for k in lengths]
 taught=old
 if arm=='SH':
  if variable:
   taught=list(old)
   for L in range(9):
    ids=np.flatnonzero(lengths==L)
    for j,i in enumerate(ids):taught[i]=old[ids[(j+1)%len(ids)]]
  else:taught=old[-1:]+old[:-1]
 enabled=arm!='N';passes=(252 if variable else 256)//m;tag=f'{seed}_{std}_{add}_{arm}'
 if int(sys.argv[1])==1 and arm=='W':influence(e,x[:m],tag+'_influence_empty',0)
 n=0
 # Cumulative first/fourth/final pass grid without reteaching a checkpoint.
 done=0
 for stop in sorted(set([1,min(4,passes),passes])):
  n=teach(e,x[:m],taught,stop-done,n,enabled,add);done=stop;probe(e,x[:m],old,tag+f'_OLD_pass{stop}')
 checkpoint(e,tag+'_OLD.npz')
 if arm=='W':
  q=e.clone();r=e.clone();q.tick(x[0,0],None);r.rest(DT)
  assert np.allclose(q.bank.fast,r.bank.fast,rtol=0,atol=1e-14) and np.allclose(q.bank.slow,r.bank.slow,rtol=0,atol=1e-14)
 if int(sys.argv[1])==1 and arm=='W':influence(e,x[:m],tag+'_influence_OLD',n)
 e.rest(3600);probe(e,x[:m],old,tag+'_rest1');shadow=e.clone() if arm=='W' else None
 n0=n;done=0
 for stop in sorted(set([1,min(4,passes),passes])):
  n=teach(e,x[m:2*m],new,stop-done,n,enabled,add);done=stop
  probe(e,x[m:2*m],new,tag+f'_NEW_pass{stop}');probe(e,x[:m],old,tag+f'_OLD_afterNEW{stop}')
 checkpoint(e,tag+'_NEW.npz');e.rest(3600)
 for cohort,codes,vals in [('OLD',x[:m],old),('NEW',x[m:2*m],new)]:probe(e,codes,vals,tag+'_final_'+cohort)
 probe(e,x[2*m:],[[] for _ in range(8)],tag+'_ABSENT')
 checkpoint(e,tag+'_final.npz')
 if int(sys.argv[1])==1 and arm=='W':influence(e,x[:m],tag+'_influence_final',n)
 if shadow is not None:
  teach(shadow,x[m:2*m],new,passes,n0,False,add);shadow.rest(3600)
  probe(shadow,x[:m],old,tag+'_S_NEW_final_OLD');probe(shadow,x[m:2*m],new,tag+'_S_NEW_final_NEW');checkpoint(shadow,tag+'_S_NEW_final.npz')
 return x

def main(c):
 checks={}
 if c==1:
  e=Engine(5177,202610228101,False);x=frame(e,202610228101,16)
  a=e.clone();b=e.clone();a.tick(x[0,0],None);b.rest(DT)
  checks['passive_decay_parity']=np.array_equal(a.bank.fast,b.bank.fast) and np.array_equal(a.bank.slow,b.bank.slow)
  checks['codec257']=all(d.decode(targets(t))==t for t in range(257))
  assert all(checks.values())
  for arm in ('W','N','SH'):life(202610228101,16,256,False,False,arm)
 elif c==2:
  for std in (False,True):
   for add in (False,True):
    for arm in ('W','SH'):life(202610228201,16,2048,std,add,arm)
   life(202610228201,16,2048,std,False,'N')
 else:
  for seed,m in ((202610228301,18),(202610228311,18),(202610228321,36)):
   for arm in ('W','N','SH'):life(seed,m,2048,False,False,arm,True)
  # Fixed calibration-only EOS threshold per read protocol. No native learning here.
  thresholds={}
  for view in ('SNAP','SEQ','TIME'):
   pools=[r for r in rows if r['view']==view and r['label'].startswith('202610228301_False_False_W') and any(r['label'].endswith(z) for z in ('OLD_pass14','OLD_afterNEW14','NEW_pass14','final_OLD','final_NEW'))]
   v=[];truth=[];weights=[]
   for r in pools:
    m=arrays[r['array']];vals=r['values'];v.extend(m[:,:,8].ravel());truth.extend([j==len(val) for val in vals for j in range(9)]);weights.extend([(1 if r['label'].endswith('OLD_pass14') else .5)/len(vals)]*(len(vals)*9))
   v=np.array(v);truth=np.array(truth);w=np.array(weights);w[truth]/=w[truth].sum();w[~truth]/=w[~truth].sum()
   u=np.unique(v);candidates=np.r_[0,np.nextafter(u[0],-np.inf),(u[:-1]+u[1:])/2,np.nextafter(u[-1],np.inf)]
   thresholds[view]=float(min(candidates,key=lambda t:(float(np.sum(w*((v>t)!=truth))),abs(t),t)))
  for r in rows:
   if not r['label'].startswith('202610228301'):r['calibrated']=metrics(arrays[r['array']],r['values'],thresholds[r['view']])
  checks['thresholds']=thresholds
 np.savez_compressed(P/'cycles'/f'CYCLE{c}_arrays.npz',**arrays,event_terms=np.array(events))
 save(f'cycles/CYCLE{c}_RESULT.json',dict(cycle=c,records=records,synthetic_events=synthetic,CPU_seconds=time.process_time()-START,wall_seconds=time.monotonic()-WALL,checks=checks,rows=rows,science_jobs=0))
 print(json.dumps(dict(cycle=c,records=records,CPU_seconds=time.process_time()-START)),flush=True)
if __name__=='__main__':
 awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
 try:main(int(sys.argv[1]))
 except Exception as ex:save(f'cycles/CYCLE{sys.argv[1]}_FAILURE.json',dict(error=repr(ex),records=records,synthetic_events=synthetic));raise
 finally:awake.terminate()
