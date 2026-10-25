import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,pickle,json,hashlib,time,subprocess,copy,resource
from pathlib import Path
import numpy as np
from kernel import Model,Trace,teach,generate,lb1
import v147_data as data
H=Path(__file__).resolve().parent;R=H.parent;START=time.process_time()
def save(n,v):(H/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
if len(sys.argv)>1 and sys.argv[1]=='child':
 m,tr,rng,w,passes=pickle.loads((H/'resume_input.pkl').read_bytes());pairs=data.oriented(w['train'])
 for _ in range(passes):
  for i in rng.permutation(len(pairs)):p=pairs[i];teach(m,tr,data.prompt(w,p),data.answer(w,p))
 (H/'resume_output.pkl').write_bytes(pickle.dumps((m,rng)));sys.exit(0)
a=json.loads((H/'CALIBRATION.json').read_text());rows=[json.loads((H/f'V147-D{d}_{arm}_{al}.json').read_text()) for d in [1,2] for arm in ['FIXED','LOCAL'] for al in [.003,.03]]
expected={f'V147-D{d}_{arm}_{al}' for d in [1,2] for arm in ['FIXED','LOCAL'] for al in [.003,.03]}
def valid(rows):
 try:
  assert len(rows)==8 and {x['id'] for x in rows}==expected
  for z in rows:
   w=data.world(z['world']);pairs=data.oriented(w['train']);assert z['records']==256*len(pairs) and z['targets']==2*z['records'];assert [c['passes'] for c in z['curve']]==[1,4,16,64,256]
   for c in z['curve']:
    q=c['taught'];assert sorted(tuple(rr['pair']) for rr in q['rows'])==sorted(pairs);g={}
    for rr in q['rows']:
     assert rr['exact']==(rr['status']=='EOS' and bytes.fromhex(rr['output_hex'])==data.answer(w,rr['pair']));g.setdefault(tuple(sorted(rr['pair'])),[]).append(rr['exact'])
    assert np.isfinite(q['complete']) and abs(q['complete']-np.mean([np.mean(v) for v in g.values()]))<1e-12
  return True
 except (AssertionError,ValueError,KeyError,TypeError):return False
assert valid(rows);bad=[rows[:-1],rows+[rows[0]]]
b=copy.deepcopy(rows);b[0]['curve'][-1]['taught']['complete']=float('nan');bad.append(b)
b=copy.deepcopy(rows);b[0]['records']+=1;bad.append(b)
b=copy.deepcopy(rows);b[0]['curve'][-1]['taught']['rows'][0]['exact']=not b[0]['curve'][-1]['taught']['rows'][0]['exact'];bad.append(b);assert not any(valid(x) for x in bad)
for p,h in json.loads((H/'SOURCE_LOCK.json').read_text()).items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h
# Saved-state diagnostics, not trajectory replay or held evaluation.
stability=[]
for z in rows:
 m,tr,rg,pas=pickle.loads((H/(z['id']+'.pkl')).read_bytes());birth=Model(data.seed(z['world'],'birth'),m.meta,m.drive,m.side,m.counts,m.alpha,0,features=z['arm']!='FIXED');ww=data.world(z['world']);same=[];jac=[];loss=[];collisions={}
 for pair in data.oriented(ww['train']):
  pp=data.prompt(ww,pair);yy=data.answer(ww,pair)[0]
  for prefix,target in [(b'',yy),(bytes([yy]),256)]:
   vec=tr.context(pp,prefix);old=set(birth.code(vec)[1]);new=set(m.code(vec)[1]);collisions.setdefault(tuple(sorted(new)),set()).add(target);same.append(old==new);jac.append(len(old&new)/len(old|new));logit=m.margins(vec);t=np.array([(target>>j)&1 for j in range(9)]);loss.append(np.mean(np.logaddexp(0,logit)-t*logit))
 stability.append(dict(id=z['id'],same_code_fraction=float(np.mean(same)),mean_KC_jaccard=float(np.mean(jac)),teacher_bit_BCE=float(np.mean(loss)),conflicting_target_code_groups=sum(len(v)>1 for v in collisions.values())))
save('SAVED_STATE_DIAGNOSTICS.json',stability)
if a['eligible']:
 alpha=max(a['eligible'],key=lambda al:(np.mean([z['curve'][-1]['taught']['complete'] for z in rows if z['alpha']==al]),-sum(z['cpu'] for z in rows if z['alpha']==al),-al));arms=['FIXED','LOCAL','SHUFFLED','NO-WRITE'];passes=16;prefix=8;suffix=8
else:alpha=.03;arms=['LOCAL'];passes=2;prefix=1;suffix=1
w=data.world('V147-TECH1');pairs=data.oriented(w['train']);md=lb1.fastmem.Compartment().fly.m;counts={s:int(np.ceil(md.active_fraction*np.sum(md.kc_side==s))) for s in [0,1]};records=0;targets=0
for arm in arms:
 m=Model(data.seed(w['name'],'birth'),md.pn_type_index,md.B,md.kc_side,counts,alpha,0,features=arm!='FIXED',writes=arm!='NO-WRITE');tr=Trace();rng=np.random.Generator(np.random.PCG64(data.seed(w['name'],'order')));shuf=data.shuffle(w)
 for pas in range(1,passes+1):
  for i in rng.permutation(len(pairs)):
   p=pairs[i];y=shuf[tuple(sorted(p))] if arm=='SHUFFLED' else data.answer(w,p);teach(m,tr,data.prompt(w,p),y);records+=1;targets+=2
  if arm=='LOCAL' and pas==prefix:(H/'resume_input.pkl').write_bytes(pickle.dumps((m,tr,rng,w,suffix)))
 if arm=='LOCAL':
  before=time.process_time();ret=subprocess.run([sys.executable,str(H/'cycle3.py'),'child'],stdout=subprocess.PIPE,stderr=subprocess.PIPE);assert ret.returncode==0,ret.stderr.decode();restored,rg=pickle.loads((H/'resume_output.pkl').read_bytes());assert m.digest()==restored.digest() and rng.bit_generator.state==rg.bit_generator.state;records+=suffix*len(pairs);targets+=2*suffix*len(pairs)
summary=dict(status='THREE_CYCLES_COMPLETE_ACQUISITION_NOT_READY' if not a['eligible'] else 'THREE_CYCLES_COMPLETE_QUALIFICATION_ONLY',calibration=a,rehearsal_four_arms=bool(a['eligible']),selected_alpha=alpha if a['eligible'] else None,cross_process_resume_equal=True,mutations_rejected=5,source_hashes_match=True,held_scores_opened=False,assessment_worlds_generated=False,ordinary_addition_goal_unmet=True,calibration_records=a['records'],calibration_targets=a['targets'],cycle3_record_applications=records,cycle3_targets=targets,cycle1_synthetic_target_events=4,total_record_applications=a['records']+records,total_observed_targets=a['targets']+targets+4,formal_assessment_jobs=0,new_core_adopted=False,cycle3_self_cpu=time.process_time()-START)
summary['cycle3_process_cpu_including_imports']=time.process_time(); rr=resource.getrusage(resource.RUSAGE_CHILDREN); summary['cycle3_child_cpu']=rr.ru_utime+rr.ru_stime
save('FINAL_AUDIT.json',summary)
print(json.dumps(summary,indent=2))
