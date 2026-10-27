"""New frozen-checkpoint queries; no native teaching or historical trajectory replay."""
import os, sys, time, resource, json, hashlib, pickle, copy
from pathlib import Path
from collections import Counter, deque
START_CPU=time.process_time(); START_WALL=time.monotonic()
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
import numpy as np
H=Path(__file__).resolve().parent; ROOT=H.parent; OLD=ROOT/'minifly_v171_dev_20261026'
sys.path.insert(0,str(OLD))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
 p=H/p;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)
def budget(*args,**kw):
 assert time.process_time()-START_CPU<1800,'CPU_CAP'
 assert time.monotonic()-START_WALL<3600,'WALL_CAP'
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1073741824,'RSS_CAP'
def stats(a):
 a=np.asarray(a);return {'min':float(a.min()),'median':float(np.median(a)),'mean':float(a.mean()),'max':float(a.max())}
def cosine(a,b):
 d=float(np.linalg.norm(a)*np.linalg.norm(b));return float(a@b/d) if d>1e-30 else None
def main():
 assert not (H/'CREDIT_STARTED.json').exists(),'No automatic retries'
 save('CREDIT_STARTED.json',{'status':'STARTED','new_training_records':0})
 lock=json.loads((OLD/'CALIBRATION_SOURCE_LOCK.json').read_text())
 assert all(sha(ROOT/p)==v for p,v in lock.items()),'PARENT_SOURCE_CHANGED'
 import support171 as s
 s.budget=budget;s.COUNTS.clear();s.TIMES.clear();s.H=H
 import native171 as n
 from controller171 import Controller,PARAMS
 from numeric171 import forward_influence,forward_gradient,parameter_indices
 source=dict(lock)
 for mod in list(sys.modules.values()):
  path=getattr(mod,'__file__',None)
  if path and Path(path).resolve().is_relative_to(ROOT) and Path(path).suffix=='.py':source[str(Path(path).resolve().relative_to(ROOT))]=sha(path)
 source[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
 save('CREDIT_SOURCE_LOCK.json',source)
 data=json.loads((OLD/'local/TRAIN_DATA.json').read_text()); published=json.loads((OLD/'CALIBRATION_RESULTS.json').read_text())
 inputs={str((OLD/'local/TRAIN_DATA.json').relative_to(ROOT)):sha(OLD/'local/TRAIN_DATA.json')}
 def grad(c,y,nn,mode):
  q=c.clone();q.grads={k:np.zeros_like(getattr(q,k)) for k in PARAMS};q.touched=set()
  if mode=='EXACT1':q.frames=deque(list(q.frames)[-1:]);q.credit(y,nn,mode='EXACT3')
  else:q.credit(y,nn,mode=mode)
  return q
 # EXACT1 is exact conditional credit with all earlier augmented states detached.
 c=Controller(202610269790,'EXACT_RECORD',n.C);v=np.random.default_rng(17201).normal(0,2,(4,16));terms=[];prefix=None
 for i,b in enumerate((17,255,4,13)):
  if i==3:prefix=(c.h.copy(),c.r.copy())
  terms=forward_influence(c,b,v[i],terms,1)
 forward=forward_gradient(c,sum(terms),183,17);q=grad(c,183,17,'EXACT1');reverse=q.gradvector()
 rel=float(np.linalg.norm(forward-reverse)/max(1e-14,np.linalg.norm(reverse)))
 ids,_=parameter_indices(c);sel=[int(ix.ravel()[np.argmax(abs(reverse[ix.ravel()]))]) for ix in ids.values()];vec=c.vector();fd=[]
 for j in sel:
  losses=[]
  for sign in (1,-1):
   cc=c.clone();vv=vec.copy();vv[j]+=sign*1e-5;cc.setvector(vv);cc.reset();cc.h=prefix[0].copy();cc.r=prefix[1].copy();cc.step(13,v[-1]);losses.append(-__import__('controller171').warm(17)*cc.read()[1][183])
  fd.append((losses[0]-losses[1])/2e-5)
 fdrel=float(np.linalg.norm(np.array(fd)-reverse[sel])/max(1e-14,np.linalg.norm(reverse[sel])))
 save('EXACT1_NUMERICAL.json',{'forward_reverse_relative_error':rel,'finite_difference_relative_error':fdrel,'finite_difference_parameters':len(sel),'synthetic_loss_evaluations':12,'earlier_state_detached':True})
 assert rel<1e-6 and fdrel<1e-6
 details=[];sites=[];auth=[];checkpoint_hashes={};n_own=0;n_tf=0;applied=[]
 for world in (202610269711,202610269712):
  rows=data[str(world)]['OLD']+data[str(world)]['NEW'];wr=published['worlds'][str(world)]
  for stage,cp in [('OLD16','OLD16'),('FINAL','final')]:
   ck=OLD/'local'/f'cal_{world}_{cp}.pkl';checkpoint_hashes[str(ck.relative_to(ROOT))]=sha(ck)
   with ck.open('rb') as f:m,ctrls=pickle.load(f)
   before=sha(ck);state_before=hashlib.sha256(pickle.dumps((m,ctrls),protocol=5)).hexdigest()
   maps={'NATIVE':None,**{q.role:q for q in ctrls}};expected={}
   for role in maps:
    recs=[wr['panels']['OLD16'][role]] if stage=='OLD16' else [wr['panels']['final_OLD'][role],wr['panels']['NEW256'][role]]
    expected[role]={}
    for rec in recs:
     path=OLD/rec['path'];assert sha(path)==rec['sha256'];inputs[str(path.relative_to(ROOT))]=sha(path)
     expected[role].update({r['id']:r for r in json.loads(path.read_text())['rows']})
   for row in rows:
    prompt=bytes.fromhex(row['prompt']);answer=bytes.fromhex(row['answer']);ys=list(answer)+[256]
    for role,controller in maps.items():
     view=m.clone();cc=None if controller is None else controller.clone();own=n.own(view,cc,prompt);n_own+=1
     ex=expected[role][row['id']];assert own['choices']==ex['choices'],(world,stage,role,row['id'])
     auth.append({'world':world,'stage':stage,'role':role,'id':row['id'],'own_choices_match':True})
    base=maps['EXACT_RECORD'].clone();base.reset();cut=base.clone();cut.R.fill(0)
    # Native reads are queried once and shared only across the evaluator controllers.
    view=m.clone();view._control(True);trace=[]
    for b in prompt:
     view.step(b);vv=n.raw(view,'r1_frozen_prompt');base.step(b,vv);cut.step(b,vv)
    view._control(False);vv=n.raw(view,'r1_frozen_boundary');base.refresh(vv);cut.refresh(vv);n_tf+=1
    accum={key:{mode:{k:np.zeros_like(getattr(base,k)) for k in PARAMS} for mode in ('EXACT1','EXACT3','EXACT_RECORD')} for key in ('ON','R_ZERO')}
    touched={key:{mode:set() for mode in ('EXACT1','EXACT3','EXACT_RECORD')} for key in accum}
    for j,y in enumerate(ys):
     for tag,cc in [('ON',base),('R_ZERO',cut)]:
      gs={mode:grad(cc,y,m.n+j,mode) for mode in ('EXACT1','EXACT3','EXACT_RECORD')};vectors={k:g.gradvector() for k,g in gs.items()};g1=vectors['EXACT1'];g3=vectors['EXACT3'];gf=vectors['EXACT_RECORD'];den=max(1e-14,float(np.linalg.norm(gf)))
      blocks={}
      for k in PARAMS:
       a=gs['EXACT1'].grads[k];b=gs['EXACT3'].grads[k];d=gs['EXACT_RECORD'].grads[k]
       blocks[k]={'one_L2':float(np.linalg.norm(a)),'three_L2':float(np.linalg.norm(b)),'full_L2':float(np.linalg.norm(d)),'three_minus_one_L2':float(np.linalg.norm(b-a)),'full_minus_three_L2':float(np.linalg.norm(d-b))}
       for mode in gs:accum[tag][mode][k]+=gs[mode].grads[k]
      for mode in gs:touched[tag][mode]|=gs[mode].touched
      details.append({'world':world,'stage':stage,'id':row['id'],'site':j,'feedback':tag,'full_L2':den,'three_minus_one_over_full':float(np.linalg.norm(g3-g1)/den),'full_minus_three_over_full':float(np.linalg.norm(gf-g3)/den),'one_full_cosine':cosine(g1,gf),'three_full_cosine':cosine(g3,gf),'blocks':blocks})
     # Authenticate native and trained-controller teacher-prefix probabilities too.
     lp=base.read()[1];old=expected['EXACT_RECORD'][row['id']]['teacher_forced'][j]
     assert abs(float(lp[y])-old['lp'])<1e-9,(world,stage,row['id'],j,'TF probability mismatch')
     sites.append({'world':world,'stage':stage,'id':row['id'],'phase':'OLD' if row in data[str(world)]['OLD'] else 'NEW','site':j,'y':y,'v':vv.tolist()})
     if y!=256:
      view.step(y);vv=n.raw(view,'r1_frozen_target_prefix');base.step(y,vv);cut.step(y,vv)
    for tag,cc in [('ON',base),('R_ZERO',cut)]:
     steps={}
     for mode in accum[tag]:
      q=cc.clone();q.grads=accum[tag][mode];q.touched=touched[tag][mode];old=q.vector();op=q.apply();steps[mode]=q.vector()-old
     den=max(1e-14,float(np.linalg.norm(steps['EXACT_RECORD'])))
     applied.append({'world':world,'stage':stage,'id':row['id'],'feedback':tag,'full_step_L2':den,'three_minus_one_over_full_step':float(np.linalg.norm(steps['EXACT3']-steps['EXACT1'])/den),'full_minus_three_over_full_step':float(np.linalg.norm(steps['EXACT_RECORD']-steps['EXACT3'])/den),'one_full_step_cosine':cosine(steps['EXACT1'],steps['EXACT_RECORD'])})
    budget()
   assert before==sha(ck) and state_before==hashlib.sha256(pickle.dumps((m,ctrls),protocol=5)).hexdigest(),'CHECKPOINT_MUTATED'
   save('ACTIVE.json',{'world':world,'stage':stage,'own_queries':n_own,'TF_records':n_tf,'CPU':time.process_time()-START_CPU})
  del m,ctrls
 save('local/FROZEN_PREFIX_READS.json',sites);save('local/CREDIT_DETAILS.json',details);save('local/APPLIED_STEPS.json',applied);save('local/PANEL_AUTHENTICATION.json',auth)
 summary={}
 for tag in ('ON','R_ZERO'):
  rr=[x for x in details if x['feedback']==tag];aa=[x for x in applied if x['feedback']==tag]
  summary[tag]={'target_sites':len(rr),'three_minus_one_over_full_gradient':stats([x['three_minus_one_over_full'] for x in rr]),'full_minus_three_over_full_gradient':stats([x['full_minus_three_over_full'] for x in rr]),'three_minus_one_over_full_applied_step':stats([x['three_minus_one_over_full_step'] for x in aa]),'full_minus_three_over_full_applied_step':stats([x['full_minus_three_over_full_step'] for x in aa])}
 budget();assert n_own==960 and n_tf==320 and len(sites)<=1000
 assert all(sha(ROOT/p)==v for p,v in source.items())
 save('CREDIT_RESULTS.json',{'status':'COMPLETE','scope':'New non-updating frozen-checkpoint query diagnostic, not a new fitted EXACT1 history','summary':summary,'own_query_authentications':n_own,'teacher_prefix_records':n_tf,'native_target_prefix_sites':len(sites),'gradient_target_reuses':len(details)*3,'discarded_clone_parameter_steps':len(applied)*3,'new_training_records':0,'historical_native_replay':False,'input_sha256':inputs,'checkpoint_sha256_now':checkpoint_hashes,'checkpoint_original_hash_receipts_present':False,'authentication':'All native/EXACT3/FULL own choices match published panels; FULL TF target probabilities match','parent_unchanged':True,'counts':dict(s.COUNTS),'CPU_seconds':time.process_time()-START_CPU,'wall_seconds':time.monotonic()-START_WALL,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
 print(json.dumps({'status':'COMPLETE','summary':summary,'CPU_seconds':time.process_time()-START_CPU}))
if __name__=='__main__':
 import subprocess
 awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 try:main()
 except Exception as e:
  import traceback
  text=traceback.format_exc().replace(str(ROOT),'$REPO').replace(str(Path.home()),'$HOME');save('CREDIT_FAILURE.json',{'error':str(e),'traceback_redacted':text,'CPU_seconds':time.process_time()-START_CPU});raise
 finally:
  awake.terminate();awake.wait(timeout=5)
