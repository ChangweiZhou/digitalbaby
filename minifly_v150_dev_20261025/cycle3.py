import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,pickle,json,hashlib,time,resource,subprocess,copy
from pathlib import Path
import numpy as np
from core import Output
from util import H,R,save,load,event
if len(sys.argv)>1 and sys.argv[1]=='resume_child':
 m,tc,ty,rng=pickle.loads((H/'RESUME_INPUT.pkl').read_bytes())
 for _ in range(8):
  i=int(rng.integers(len(tc)));m.observe(int(tc[i]),int(ty[i]));event(event='TECHNICAL_TARGET',branch='child',n=m.n)
 (H/'RESUME_OUTPUT.pkl').write_bytes(pickle.dumps((m,rng)));sys.exit(0)
start=time.process_time();manifest=load('MANIFEST.json');expected={z['id'] for z in manifest['roster']};rows=load('COMPACT.json')
def valid(rr):
 try:
  assert len(rr)==66 and {z['id'] for z in rr}==expected
  for z in rr:
   assert z['targets']==4096 and z['validation_targets']==1024 and np.isfinite(z['validation_BPS']) and np.isfinite(z['expected_BPS'])
   ident=z['id'];assert next(x for x in manifest['roster'] if x['id']==ident)['arm']==z['arm']
  return True
 except (AssertionError,KeyError,TypeError):return False
assert valid(rows);bad=[rows[:-1],rows+[rows[0]]]
b=copy.deepcopy(rows);b[0]['validation_BPS']=float('nan');bad.append(b)
b=copy.deepcopy(rows);b[0]['targets']=4095;bad.append(b)
b=copy.deepcopy(rows);b[0]['arm']='UNKNOWN';bad.append(b);assert not any(valid(x) for x in bad)
for p,sha in load('SOURCE_LOCK.json').items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==sha,p
maxerr=0.;inputhash={}
for z in rows:
 ident=z['id'];path=H/(ident+'.pkl');inputhash[path.name]=hashlib.sha256(path.read_bytes()).hexdigest();m=pickle.loads(path.read_bytes());assert m.n==4096 and m.arm==z['arm'] and m.eta==z['eta'];before=m.digest();rr=load(ident+'.json');data=np.load(H/f"DATA_{z['fixture']}_{z['seed']}.npz")
 val=0.;raw=rr['validation_rows'];assert len(raw)==1024
 for i,(c,y) in enumerate(zip(data['validation_ctx'],data['validation_y'])):
  p,lp=m.read(int(c));assert raw[i][0:2]==[int(c),int(y)] and raw[i][3]==int(np.argmax(p));maxerr=max(maxerr,abs(float(lp[y])-raw[i][2]));val-=float(lp[y])
 computed=val/1024/np.log(2);assert abs(computed-z['validation_BPS'])<1e-12
 expectedloss=float(np.mean([-q@m.read(i)[1]/np.log(2) for i,q in enumerate(data['q'])]));assert abs(expectedloss-z['expected_BPS'])<1e-12
 assert m.digest()==before and np.max(np.linalg.norm(m.C,axis=1))<=4+1e-12 and np.max(abs(m.C.mean(1)))<1e-12
# Separately paid, source-locked warmup/resume fixture. No substantive model is continued.
C=json.loads((R/'minifly_cls_dev_20261017/OUTPUT_CODEBOOK.json').read_text())['words_by_class'];C=np.array([[1. if w&(1<<j) else -1. for j in range(16)] for w in C]);data=np.load(H/'DATA_four_context_full_support_11.npz');tc,ty=data['train_ctx'],data['train_y'];rng=np.random.Generator(np.random.PCG64(15020261025));m=Output(C,4,'C',.1)
for _ in range(16):
 i=int(rng.integers(len(tc)));m.observe(int(tc[i]),int(ty[i]));event(event='TECHNICAL_TARGET',branch='prefix',n=m.n)
(H/'RESUME_INPUT.pkl').write_bytes(pickle.dumps((m,tc,ty,rng)))
for _ in range(8):
 i=int(rng.integers(len(tc)));m.observe(int(tc[i]),int(ty[i]));event(event='TECHNICAL_TARGET',branch='parent',n=m.n)
ret=subprocess.run([sys.executable,str(H/'cycle3.py'),'resume_child'],stdout=subprocess.PIPE,stderr=subprocess.PIPE);assert ret.returncode==0,ret.stderr.decode();other,rg=pickle.loads((H/'RESUME_OUTPUT.pkl').read_bytes());assert other.digest()==m.digest() and rg.bit_generator.state==rng.bit_generator.state and m.n==24
for name,sha in inputhash.items():assert hashlib.sha256((H/name).read_bytes()).hexdigest()==sha
execution=load('EXECUTION_SUMMARY.json');native_files=list(H.glob('NATIVE_*.json'));native_rows=[]
if execution['native_full_histories']:
 for text in ['bryant','buster']:
  for arm in ['A','B','C']:native_rows.append(load(f'NATIVE_{text}_{arm}.json'))
 assert len(native_rows)==6 and sum(z['targets'] for z in native_rows)==12504 and sum(z['records'] for z in native_rows)==240
 assert sum(z['calibration']['targets']+z['development_evaluation']['targets'] for z in native_rows)==18777
 assert sum(len(z['generations']) for z in native_rows)==96
else:assert not native_files
resources=resource.getrusage(resource.RUSAGE_CHILDREN)
summary={'version':'V150','status':'THREE_CYCLES_COMPLETE_'+execution['status'],'roster_exact':True,'source_hashes_match':True,'compact_and_saved_state_scores_match':True,'max_raw_log_probability_error':maxerr,'five_corrupt_artifacts_rejected':True,'cross_process_resume_equal':True,'warmup_resumed_at_n16':True,'toy_fits':66,'toy_target_events':270336,'write_free_validation_probes':67584,'technical_target_events':32,'partial_native_fixture_writes':load('CYCLE1_NATIVE.json')['partial_native_writes'],'native_full_line_records':sum(z['records'] for z in native_rows),'native_observed_target_events':sum(z['targets'] for z in native_rows),'all_observed_target_events_including_technical':270336+32+sum(z['targets'] for z in native_rows),'native_runs_complete':bool(native_rows),'holdback_or_Alice_opened':False,'addition_efficacy_established':False,'new_core_adopted':False,'cycle3_CPU':time.process_time(),'cycle3_resume_child_CPU':resources.ru_utime+resources.ru_stime,'input_checkpoint_hashes':inputhash}
save('FINAL_AUDIT.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='input_checkpoint_hashes'},indent=2))
