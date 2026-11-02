"""Engineering equivalence and timing only; writes no natural-text learner state."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,pickle,resource,hashlib,gc
from pathlib import Path
import numpy as np
import optimized_native as O
N=O.N;W=N.W;H=Path(__file__).resolve().parent
START=time.process_time();CHECKS=[];COMPARES=0;MAX_ERROR=0.
def save(name,obj):
 p=H/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def equal(a,b,tag):
 global COMPARES,MAX_ERROR
 COMPARES+=1
 if isinstance(a,np.ndarray):
  ok=np.array_equal(a,b)
  if a.size:MAX_ERROR=max(MAX_ERROR,float(np.max(np.abs(a.astype(float)-b.astype(float)))))
 else:ok=a==b
 if not ok:raise AssertionError(tag)
def state(a,b,tag):
 for k in ('time','n','records'):equal(getattr(a,k),getattr(b,k),tag+':'+k)
 for k in ('theta','freq'):equal(getattr(a,k),getattr(b,k),tag+':'+k)
 if a.fe is not None:equal(pickle.dumps(a.fe,protocol=5),pickle.dumps(b.fe,protocol=5),tag+':FE')
 for j,(sa,sb) in enumerate(zip(a.bank,b.bank)):
  for k in ('fast','slow','adapt','elapsed','event_count','presentation_count'):equal(np.asarray(getattr(sa.fly.m,k)),np.asarray(getattr(sb.fly.m,k)),f'{tag}:{j}:{k}')
  for k in ('brain_t','teach_seen','pending_t'):equal(getattr(sa,k),getattr(sb,k),f'{tag}:{j}:{k}')
  equal(sa.fly.stats,sb.fly.stats,f'{tag}:{j}:stats')
def capture_failure(base,opt,error):
 (H/'local').mkdir(exist_ok=True)
 for name,m in [('literal',base),('optimized',opt)]:
  if m is not None:
   with (H/f'local/FAILED_{name}.pkl').open('wb') as f:pickle.dump(m,f,protocol=5)
 save('FAILURE.json',dict(error=repr(error),CPU=time.process_time()-START,comparisons=COMPARES,max_error=MAX_ERROR,partial_state_saved=True))
def main():
 global BASE,OPT
 # Label-independent fixed source asset; same H in literal and optimized models.
 wh=N.rng('V176|optimization|H').normal(0,1/np.sqrt(184),(5177,512)).astype(np.float32);asset=(wh,np.zeros(512,np.float32))
 for value in asset:value.flags.writeable=False
 # Deterministic tie rules and zero-positive eligibility, independent of native labels.
 g=N.rng('V176|optimization|topk');side=np.arange(5177)%2;sn={0:92,1:92}
 for raw in (np.zeros(5177),np.ones(5177),g.integers(-2,3,5177).astype(float),g.normal(size=5177)):
  literal=np.zeros(5177)
  for s in (0,1):
   ix=np.flatnonzero((side==s)&(raw>0));order=np.lexsort((ix,-raw[ix]));literal[ix[order[:min(len(ix),sn[s])]]]=1.
  equal(literal,O.topk_exact(raw,side,sn),'topk_ties')
 cases=[(b'',0,True),(b'\x00',255,True),(bytes(range(128)),17,True),(bytes(range(128,256)),256%256,False),(bytes((i*37)%256 for i in range(35)),83,True),(bytes((i*37+3)%256 for i in range(71)),41,False),(b'ababa',72,True),(b'ababa',72,True)]
 modes=[('RAW',None),('H',asset)]
 if len(sys.argv)>1 and sys.argv[1]=='continue':
  # Resume qualification from saved engineering states; do not repeat passed RAW histories.
  with (H/'local/FAILED_literal.pkl').open('rb') as f:BASE=pickle.load(f)
  with (H/'local/optimized_RAW_selfcontained.pkl').open('rb') as f:OPT=pickle.load(f)
  state(BASE,OPT,'saved_RAW_restore')
  reference=BASE.clone();resumed=OPT.clone();row=dict(id='restore-resume-RAW',cohort='engineering',prompt='00ff',answer='11')
  reference.train_row(row);resumed.train_row(row);state(reference,resumed,'saved_RAW_restore_continue')
  CHECKS.append(dict(arm='RAW',records=8,own_TF_pairs=8,completed_before_fixture_revision=True))
  modes=[('H',asset)]
 for asset_name,aa in modes:
  BASE=N.Model(aa);OPT=O.Model(aa)
  for i,(prompt,answer,write) in enumerate(cases):
   if i==4:BASE.n=OPT.n=1023
   row=dict(id=f'qualification:{asset_name}:{i}',cohort='engineering',prompt=prompt.hex(),answer=bytes([answer]).hex())
   BASE.train_row(row,write=write);OPT.train_row(row,write=write);state(BASE,OPT,row['id'])
   equal(W.pair(BASE,row),W.pair(OPT,row),row['id']+':own_TF');state(BASE,OPT,row['id']+':read_only')
   # Explicit future event plus generated-prefix path, independently of the correct target.
   b=BASE.clone();o=OPT.clone();b._event((answer+91)%256);o._event((answer+91)%256)
   equal(b.values(),o.values(),row['id']+':off_path');state(b,o,row['id']+':off_path_state')
  BASE.rest(128*N.F.DELTA);OPT.rest(128*N.F.DELTA);state(BASE,OPT,asset_name+':rest')
  # Self-contained engineering serialization avoids the historical helper's
  # process-global asset-ID registry. Rebind packed views on restore.
  path=H/f'local/optimized_{asset_name}_selfcontained.pkl';path.parent.mkdir(exist_ok=True)
  with path.open('wb') as f:pickle.dump(OPT,f,protocol=5)
  with path.open('rb') as f:restored=pickle.load(f)
  state(OPT,restored,asset_name+':restore')
  # Continuing after restore must update the same packed state, not detached copies.
  row=dict(id='restore-continuation',cohort='engineering',prompt='00ff',answer='11')
  reference=BASE.clone();resumed=restored.clone();reference.train_row(row);resumed.train_row(row);state(reference,resumed,asset_name+':restore_continue')
  CHECKS.append(dict(arm=asset_name,records=len(cases),own_TF_pairs=len(cases),cache_hits=OPT.cache_hits,cache_misses=OPT.cache_misses))
  del BASE,OPT,b,o,restored;BASE=OPT=None;gc.collect()
 save('QUALIFICATION.json',dict(status='PASS_BITWISE',comparisons=COMPARES,max_absolute_state_error=MAX_ERROR,checks=CHECKS,current_process_counts=dict(W.COUNTS),off_path_pairs=16,CPU=time.process_time()-START,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='E0 engineering equivalence, no scientific learning claim',assertions_saved_on_failure=True,prior_RAW_qualification_preserved_in_failure_snapshots=True))
 print('PASS_BITWISE',COMPARES,MAX_ERROR)
def benchmark(optimized_only=False):
 # Only run after equivalence; measured baseline and optimized same rows/birth/sequence.
 assert json.loads((H/'QUALIFICATION.json').read_text())['status']=='PASS_BITWISE'
 wh=N.rng('V176|optimization|H').normal(0,1/np.sqrt(184),(5177,512)).astype(np.float32);asset=(wh,np.zeros(512,np.float32));results=[]
 for tag,cls in ([('optimized',O.Model)] if optimized_only else [('literal',N.Model),('optimized',O.Model)]):
  for arm,aa in [('RAW',None),('H',asset)]:
   m=cls(aa)
   # Warm compilation outside timed sections; remains charged engineering work.
   row=dict(id=f'bench:{tag}:{arm}:warm',cohort='engineering',prompt='78',answer='79');m.train_row(row)
   for length in ((35,71,128) if optimized_only else (0,35,71,128)):
    row=dict(id=f'bench:{tag}:{arm}:{length}',cohort='engineering',prompt=bytes((i*37+11)%256 for i in range(length)).hex(),answer='78')
    t=time.process_time();m.train_row(row);train=time.process_time()-t
    t=time.process_time();[W.pair(m,row) for _ in range(2)];probe=(time.process_time()-t)/2
    results.append(dict(backend=tag,arm=arm,prompt_length=length,train_record_CPU=train,own_TF_pair_CPU=probe))
    save('BENCHMARK_PROGRESS.json',results)
   del m;gc.collect()
 save('BENCHMARK_V2.json' if optimized_only else 'BENCHMARK.json',dict(status='COMPLETE',results=results,CPU=time.process_time()-START,counts=dict(W.COUNTS),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,single_numerical_thread=True))
 print(json.dumps(results,indent=2))
BASE=OPT=None
if __name__=='__main__':
 try:
  if len(sys.argv)>1 and sys.argv[1] in ('benchmark','benchmark_v2'):benchmark(sys.argv[1]=='benchmark_v2')
  else:main()
 except BaseException as e:capture_failure(BASE,OPT,e);raise
