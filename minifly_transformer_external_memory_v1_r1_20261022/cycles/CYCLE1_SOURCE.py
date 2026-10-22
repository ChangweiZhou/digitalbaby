"""Technical native-memory diagnostics; random PN surrogate, no trained host."""
import os,sys,json,time,hashlib,resource
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.path.insert(0,str(ROOT/'minifly_v5_1_dev_20261020'))
import v51_common as common
import numba
numba.config.CACHE_DIR=str(HERE/'scratch/numba');os.environ['NUMBA_CACHE_DIR']=numba.config.CACHE_DIR
from engine import Engine,residual,decode
import numpy as np
START=time.process_time()
def save(p,d):p.write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+'\n')
def digest(e):
 h=hashlib.sha256()
 for x in (e.bank.fast,e.bank.slow,e.bank.adapt,e.q,e.q_at,e.last_at,e.ring,e.ring_pos,e.signs):h.update(x.tobytes())
 h.update(repr((e.time,e.events,e.write_l1,e.slow_write_l1,e.confirmations)).encode());return h.hexdigest()
def fresh(seed):return Engine(5177,seed,gated=False)
def frames(e,seed,m,shared=False):
 rng=np.random.default_rng(seed);b=e.bank
 train=np.tanh(rng.normal(size=(256,88)))
 theta=np.mean(np.asarray(b.B.T@train[:,b.pn_type_index].T),axis=1)
 p=np.tanh(rng.normal(size=(m,9,88)))
 if shared:p[:]=p[:,:1]
 x=np.array([[b.encode(v,theta)[0] for v in row] for row in p])
 return x,theta
def overlap(x):
 def ov(a,b):return float(np.dot(a,b)/max(1,min(a.sum(),b.sum())))
 return dict(between_keys_same_slot=float(np.mean([ov(x[i,j],x[k,j]) for j in range(9) for i in range(len(x)) for k in range(i)])),within_key_between_slots=float(np.mean([ov(row[j],row[k]) for row in x for j in range(9) for k in range(j)])),active_mean=float(x.sum(2).mean()),identical_pairs=sum(np.array_equal(x[i,j],x[k,j]) for j in range(9) for i in range(len(x)) for k in range(i)))
def read(e,x):
 before=digest(e);v=e.clone();v.begin();m=np.array([[v.margins(a)[0] for a in row] for row in x]);assert digest(e)==before
 return m
def score(e,x,values):
 m=read(e,x);outs=[]
 for row in m:
  out=[];eos=False
  for z in row:
   t=decode(z)
   if t==256:eos=True;break
   out.append(t)
  outs.append(dict(payload=out,eos=eos,truncated=not eos))
 exact=np.mean([r['payload']==list(v) and r['eos'] for r,v in zip(outs,values)])
 bits=np.array([[(int(t)>>np.arange(8))&1 for t in v] for v in values])
 return dict(exact=float(exact),bit_accuracy=float(np.mean((m[:,:4,:8]>0)==bits)),EOS_at_position4=float(np.mean(m[:,4,8]>0)),outputs=outs,margin_abs_mean=float(abs(m).mean()),fast_l1=float(abs(e.bank.fast).sum()),slow_l1=float(abs(e.bank.slow).sum()))
def teach(e,x,values,passes,start=0,enabled=True):
 n=start
 for pas in range(passes):
  for i in np.random.default_rng(8800+pas+start).permutation(len(values)):
   if time.process_time()-START>240:raise RuntimeError('CPU_LIMIT_240_SECONDS')
   e.begin();s=(1/16)*(1/16+(15/16)*min(n/1024,1))
   for j,t in enumerate(list(values[i])+[256]):
    a=x[i,j];m=e.margins(a)[0];e.tick(a,residual(m,int(t),s) if enabled else None)
   e.end(str(i).encode(),write=enabled);n+=1
 return n

def main(c):
 seed=202610226100+c;e=fresh(seed);x,theta=frames(e,seed,16)
 rng=np.random.default_rng(seed+90);values=rng.integers(0,256,(16,4),dtype=np.uint8)
 result=dict(cycle=c,seed=seed,kind='TECHNICAL_E1_SURROGATE_NOT_HOST_OR_SCIENCE',records=0,host_training_updates=0,geometry=overlap(x),fast_tau=e.bank.tf,slow_tau=e.bank.ts,initial_strength=1/256)
 if c==1:
  # Causal write influence, measured before/after one record at a pinned read clock.
  a=e.clone();b=e.clone();teach(a,x[:1],values[:1],1);teach(b,x[:1],values[:1],1,enabled=False)
  delta=read(a,x)-read(b,x)
  result.update(records=2,self_change_norm=float(np.linalg.norm(delta[0,:5])),other_key_change_norm_mean=float(np.mean(np.linalg.norm(delta[1:,:5],axis=(1,2)))),readonly=True,no_write_zero=bool(np.all(b.bank.fast==0) and np.all(b.bank.slow==0)),positive_write_mass=a.write_l1)
  xs,_=frames(e,seed,16,True);result['collapsed_position_geometry']=overlap(xs)
  # Exact byte/EOS finite codec fixture, independent of learned storage.
  from engine import targets
  result['codec_all257']=all(decode(targets(t))==t for t in range(257))
 elif c==2:
  n=0;curves=[]
  for stop in (1,4,16,64):
   n=teach(e,x,values,stop-n//16,n);curves.append(dict(passes=stop,records=n,last_strength=(1/16)*(1/16+(15/16)*min((n-1)/1024,1)),score=score(e,x,values)))
  result.update(records=n,curves=curves)
 elif c==3:
  # Fresh technical seed, original dose, paired full causal branches.
  xs,_=frames(e,seed,32);old,new=xs[:16],xs[16:];newval=rng.integers(0,256,(16,4),dtype=np.uint8)
  rows=[]
  for arm in ('W','N','SH'):
   b=e.clone();vv=np.roll(values,1,axis=0) if arm=='SH' else values
   n=teach(b,old,vv,16,enabled=arm!='N');stages=[dict(stage='immediate',score=score(b,old,values))]
   b.rest(3600);stages.append(dict(stage='rest1',score=score(b,old,values)))
   n=teach(b,new,newval,1,n,enabled=arm!='N');stages.append(dict(stage='new_end',score=score(b,old,values)))
   b.rest(3600);stages.append(dict(stage='rest2',score=score(b,old,values)))
   rows.append(dict(arm=arm,records=n,stages=stages));result['records']+=n
  result['arms']=rows
 result.update(CPU_seconds=time.process_time()-START,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 save(HERE/'cycles'/f'CYCLE{c}_RESULT.json',result);print(json.dumps({k:result[k] for k in ('cycle','records','CPU_seconds','geometry')}),flush=True)
if __name__=='__main__':
 try:main(int(sys.argv[1]))
 except Exception as ex:
  save(HERE/'cycles'/f'CYCLE{sys.argv[1]}_FAILURE.json',dict(error=repr(ex),CPU_seconds=time.process_time()-START));raise
