import os,sys,time,json,hashlib,resource,subprocess,ctypes,re
from pathlib import Path
START_CPU=time.process_time();START_WALL=time.monotonic()
P=Path(__file__).resolve().parent;R=P.parent;sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.path.insert(0,str(R/'minifly_v5_1_dev_20261020'))
import v51_common as parent
import numba
numba.config.CACHE_DIR=str(P/'scratch');os.environ['NUMBA_CACHE_DIR']=numba.config.CACHE_DIR
from engine import Engine,targets,residual,decode,neutral_query,DT
from native import query_batch,event_batch
import numpy as np
from head import initialize,features,forward,targets as head_targets,loss_grad,adam
PHASE=int(sys.argv[1]) if len(sys.argv)>1 else 0
PREV=json.loads((P/'RESOURCE_LEDGER.json').read_text()) if (P/'RESOURCE_LEDGER.json').exists() else {'CPU_seconds':0,'wall_seconds':0}
records=0;events=0;queries=0;awake=None;peak=0;lastlog=0;lasthead=0
LIB=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,v):
 q=P/name;q.parent.mkdir(parents=True,exist_ok=True);t=q.with_suffix(q.suffix+'.pending');t.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n');t.replace(q)
def stats_pid(pid):
 buf=ctypes.create_string_buffer(1024)
 if LIB.proc_pid_rusage(pid,0,ctypes.byref(buf))!=0:raise RuntimeError('PROCESS_RSS_MEASUREMENT_UNAVAILABLE')
 a=ctypes.cast(ctypes.byref(buf,16),ctypes.POINTER(ctypes.c_uint64));return int(a[6])
def resource_stats():
 global peak
 rss=stats_pid(os.getpid())+(stats_pid(awake.pid) if awake is not None and awake.poll() is None else 0)
 peak=max(peak,rss);child=resource.getrusage(resource.RUSAGE_CHILDREN)
 return dict(CPU_seconds=PREV['CPU_seconds']+time.process_time()-START_CPU+child.ru_utime+child.ru_stime,wall_seconds=PREV['wall_seconds']+time.monotonic()-START_WALL,peak_tree_RSS_bytes=peak)
def check(message='',admission=False):
 global lastlog,lasthead
 st=resource_stats()
 if st['CPU_seconds']>=1800 or st['wall_seconds']>=2700 or peak>=2*1024**3:raise RuntimeError('RESOURCE_LIMIT')
 if admission:
  out=subprocess.check_output(['/usr/bin/vm_stat'],text=True);pg=int(re.search(r'page size of (\d+)',out).group(1));n=0
  for label in ('Pages free','Pages inactive','Pages speculative'):n+=int(re.search(re.escape(label)+r':\s+(\d+)',out).group(1))
  if n*pg<2*1024**3:raise RuntimeError('RESOURCE_ADMISSION_HEADROOM_BELOW_2GIB')
 if message or time.monotonic()-lastlog>20:
  v=dict(phase=PHASE,message=message,records_this_phase=records,events_this_phase=events,queries_this_phase=queries,**st);save('STATUS.json',v);print(json.dumps(v),flush=True);lastlog=time.monotonic()
def digest(e):
 h=hashlib.sha256()
 for a in (e.bank.fast,e.bank.slow,e.bank.adapt,e.q,e.q_at,e.last_at,e.ring,e.ring_pos,e.signs):h.update(a.tobytes())
 h.update(repr((e.time,e.events,e.write_l1,e.slow_write_l1,e.confirmations)).encode());return h.hexdigest()
def calibrate():
 e=Engine(5177,202610229001,False);rng=np.random.default_rng(202610229001);v=np.tanh(rng.normal(size=(2048,88)));b=e.bank
 z=np.asarray(b.B.T@v[:,b.pn_type_index].T);mu=z.mean(1);sg=z.std(1);floor=np.sqrt(np.finfo(float).eps)*np.median(sg[sg>0]);assert floor>0
 np.savez_compressed(P/'assets/calibration.npz',mu=mu,sigma=sg,floor=floor,receptors=v)
def keys(seed):return [hashlib.blake2b(f'RECOVERABILITY_V1:key:{seed}:{i}'.encode(),digest_size=16).digest() for i in range(44)]
def encode(e,ks,std):
 with np.load(P/'assets/calibration.npz',allow_pickle=False) as z:mu=z['mu'];sg=np.maximum(z['sigma'],z['floor'])
 b=e.bank;xs=[];receptors=[]
 for key in ks:
  seed=int.from_bytes(hashlib.blake2b(b'RECOVERABILITY_V1_FIXED_SALT\x00'+key,digest_size=16).digest(),'little')
  v=np.tanh(np.random.default_rng(seed).normal(size=(9,88)));receptors.append(v);cs=[]
  for p in v:
   a=np.asarray(b.B.T@p[b.pn_type_index]).ravel()-mu
   if std:a/=sg
   x=np.zeros(5177)
   for side in (0,1):
    ix=np.flatnonzero((b.kc_side==side)&(a>0));order=np.lexsort((ix,-a[ix]));x[ix[order[:b.side_n[side]]]]=1
   cs.append(x)
  xs.append(cs)
 return np.array(xs),np.array(receptors)
def values(seed,life):
 rng=np.random.default_rng(np.random.SeedSequence([seed,life,20261022]));out=[]
 for _ in range(2):
  lengths=rng.permutation(np.tile(np.arange(9),2));out.append([rng.integers(0,256,int(L),dtype=np.uint8).tolist() for L in lengths])
 return out

def read_one(e,x):
 v=e.clone();v.begin();b=v.bank;out=[]
 for a in x:
  actual,_=query_batch(b.fast,b.slow,b.adapt,a,DT,b.Q,b.reader_Q,b.MM,b.k0,b.ta,b.tg,b.tf,b.ts)
  neutral,_=neutral_query(b.adapt,a,DT,b.Q,b.reader_Q,b.MM,b.k0,b.ta);out.append(actual-neutral)
 return np.array(out)
def probe(e,x):
 global queries
 before=digest(e);out=[]
 for a in x:check();out.append(read_one(e,a));queries+=1
 assert digest(e)==before;return np.array(out)
def checkpoint(e,name):
 a={k:getattr(e,k) for k in ('q','q_at','last_at','ring','ring_pos','signs')};a.update(fast=e.bank.fast,slow=e.bank.slow,adapt=e.bank.adapt,time=np.array([e.time,e.events,e.write_l1,e.slow_write_l1,e.confirmations]))
 np.savez_compressed(P/'checkpoints'/name,**a)
def teach(e,x,vals,n,enabled=True,key_ids=None,observer=None):
 global records,events
 for pas in range(14):
  for i in np.random.default_rng(7000+pas+n//252).permutation(18):
   check();e.begin();a=(1/16)*(1/16+(15/16)*min(n/1024,1))
   for j,t in enumerate(vals[i]+[256]):
    m=e.margins(x[i,j])[0];e.tick(x[i,j],residual(m,t,a) if enabled else None);events+=1
   e.end(key_ids[i] if key_ids is not None else bytes([i]),write=enabled);n+=1;records+=1
  if observer is not None and pas+1 in (1,4):observer(pas+1)
 return n

def life(seed,life_id,std,arm,training=False):
 check(f'life {seed}/{life_id} STD={std} {arm}',admission=True)
 e=Engine(5177,seed,False);ks=keys(seed);x,receptors=encode(e,ks,std);old,new=values(seed,life_id)
 tag=f'{seed}_{life_id}_{int(std)}_{arm}';asset=P/'assets'/f'{seed}_{int(std)}_addresses.npz'
 if asset.exists():
  with np.load(asset) as z:assert np.array_equal(z['x'],x.astype(np.uint8)) and np.array_equal(z['receptors'],receptors)
 else:np.savez_compressed(asset,keys=np.array([list(k) for k in ks],dtype=np.uint8),x=x.astype(np.uint8),receptors=receptors,inclusion_frequency=x.mean((0,1)))
 observed=old
 if arm=='SH':
  observed=list(old)
  for L in range(9):
   ids=[i for i,v in enumerate(old) if len(v)==L]
   for j,i in enumerate(ids):observed[i]=old[ids[(j+1)%len(ids)]]
 marg=[];pools=[];labels=[]
 def pool(name,codes,vals):marg.append(probe(e,codes));pools.append(name);labels.append(vals)
 def old_observer(pas):pool(f'OLD_pass{pas}',x[:18],old)
 n=teach(e,x[:18],observed,0,arm!='N',ks[:18],None if training else old_observer)
 pool('OLD_end',x[:18],old);checkpoint(e,tag+'_OLD.npz')
 e.rest(3600);pool('OLD_rest1',x[:18],old);shadow=e.clone() if arm=='W' and not training else None
 def new_observer(pas):
  pool(f'NEW_pass{pas}',x[18:36],new);pool(f'OLD_after_NEW_pass{pas}',x[:18],old)
 n=teach(e,x[18:36],new,n,arm!='N',ks[18:36],None if training else new_observer);pool('OLD_NEW_end',x[:18],old);pool('NEW_end',x[18:36],new);checkpoint(e,tag+'_NEW.npz')
 e.rest(3600);pool('OLD_final',x[:18],old);pool('NEW_final',x[18:36],new);checkpoint(e,tag+'_FINAL.npz')
 if not training:pool('ABSENT',x[36:],[[] for _ in range(8)])
 if shadow is not None:
  teach(shadow,x[18:36],new,252,False,ks[18:36]);shadow.rest(3600)
  marg.append(probe(shadow,x[:18]));pools.append('OLD_S_NEW');labels.append(old)
  marg.append(probe(shadow,x[18:36]));pools.append('NEW_S_NEW');labels.append(new);checkpoint(shadow,tag+'_S_NEW.npz')
 data={f'D{i}':m for i,m in enumerate(marg)};np.savez_compressed(P/'traces'/f'{tag}.npz',**data)
 save(f'traces/{tag}.json',dict(seed=seed,life=life_id,STD=std,arm=arm,pools=pools,labels=labels,time=e.time,records=n,keys_hex=[k.hex() for k in ks],observed_OLD=observed))
 return marg,labels,pools
