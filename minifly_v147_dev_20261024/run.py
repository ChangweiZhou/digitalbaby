import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import time,sys,json,hashlib,pickle,copy,subprocess
from pathlib import Path
import numpy as np
from kernel import Model,Trace,teach,generate,lb1,Local
import v147_data as data
H=Path(__file__).resolve().parent;R=H.parent;START=time.process_time();md=lb1.fastmem.Compartment().fly.m;counts={s:int(np.ceil(md.active_fraction*np.sum(md.kc_side==s))) for s in [0,1]}
def save(n,v):(H/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def event(**v):
 with (H/'LEDGER.jsonl').open('a') as f:f.write(json.dumps(dict(cpu=time.process_time(),**v))+'\n');f.flush();os.fsync(f.fileno())
def born(w,alpha,arm):return Model(data.seed(w,'birth'),md.pn_type_index,md.B,md.kc_side,counts,alpha,0,features=arm!='FIXED',writes=arm!='NO-WRITE')
def score(m,tr,w,pp):
 before=m.digest();rows=[];groups={}
 for p in data.oriented(pp):
  out,s=generate(m,tr,data.prompt(w,p));hit=s=='EOS' and out==data.answer(w,p);rows.append(dict(pair=p,output_hex=out.hex(),status=s,exact=hit));groups.setdefault(tuple(sorted(p)),[]).append(hit)
 assert before==m.digest()
 return dict(complete=float(np.mean([np.mean(v) for v in groups.values()])),rows=rows)
# Cycle 1 audit.
w=data.world('V147-D1');assert len(w['train'])==207 and len(w['held'])==69 and not set(w['train'])&set(w['held']);assert all(sum(sum(p)%23==c for p in w['held'])==3 for c in range(23))
tr=Trace()
for x in range(256):assert np.array_equal(tr.context(bytes([x]))[:88],tr.base.context(bytes([x])))
m=born('V147-TECHUNIT',.03,'LOCAL');parent=Local(data.seed('V147-TECHUNIT','birth'),md.pn_type_index,md.B,md.kc_side,counts,.03,0);p=tr.context(b'\x00\xff');m.observe(p,17);parent.observe(p,17)
assert all(np.array_equal(getattr(m,k),getattr(parent,k)) for k in ['W','b','theta'])
f=born('V147-TECHUNIT',.03,'FIXED');old=f.W.copy();f.observe(p,17);assert np.array_equal(old,f.W)
n=born('V147-TECHUNIT',.03,'NO-WRITE');hh=n.arrays_hash();n.observe(p,17);assert hh==n.arrays_hash()
q=data.shuffle(w);assert sorted(q.values())==sorted(data.answer(w,p) for p in w['train'])
save('CYCLE1.json',dict(byte_path_matches_256=True,parent_local_update_identical=True,frozen_features_invariant=True,no_write_invariant=True,strict_split_and_shuffle_valid=True,synthetic_target_events=4))
manifest=dict(phase='CALIBRATION',worlds=['V147-D1','V147-D2'],arms=['FIXED','LOCAL'],alphas=[.003,.03],passes=256,fit_count=8,held_scores_opened=False,assessment_worlds_opened=False)
save('MANIFEST.json',manifest)
files={Path(v.__file__).resolve() for v in list(sys.modules.values()) if getattr(v,'__file__',None) and str(v.__file__).endswith('.py') and Path(v.__file__).resolve().is_relative_to(R)};files|=set(H.glob('*.py'));files|={R/'minifly_a_v3/package/FULL151_CANONICAL_B.npz',H/'MANIFEST.json'}
save('SOURCE_LOCK.json',{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
rows=[]
for wn in manifest['worlds']:
 w=data.world(wn);save(wn+'_WORLD.json',w);pairs=data.oriented(w['train'])
 for alpha in manifest['alphas']:
  for arm in manifest['arms']:
   ident=f'{wn}_{arm}_{alpha}';event(event='START',id=ident);m=born(wn,alpha,arm);tr=Trace();rng=np.random.Generator(np.random.PCG64(data.seed(wn,'order')));curve=[];begin=time.process_time()
   for pas in range(1,257):
    for i in rng.permutation(len(pairs)):
     p=pairs[i];teach(m,tr,data.prompt(w,p),data.answer(w,p))
    if pas in [1,4,16,64,256]:curve.append(dict(passes=pas,taught=score(m,tr,w,w['train'])));save('ACTIVE.json',dict(id=ident,passes=pas,cpu=time.process_time()))
    if pas%16==0:event(event='PROGRESS',id=ident,passes=pas,records=m.records,targets=m.targets)
    if time.process_time()>21000:raise RuntimeError('CPU admission reserve exhausted')
   result=dict(id=ident,world=wn,arm=arm,alpha=alpha,records=m.records,targets=m.targets,cpu=time.process_time()-begin,curve=curve);save(ident+'.json',result)
   with (H/(ident+'.pkl')).open('wb') as f:pickle.dump((m,tr,rng,256),f)
   rows.append(result);event(event='COMMIT',id=ident,records=m.records,targets=m.targets);save('COMPACT.json',[dict(id=z['id'],taught=z['curve'][-1]['taught']['complete'],cpu=z['cpu']) for z in rows])
eligible=[al for al in manifest['alphas'] if any(all(next(z for z in rows if z['world']==wn and z['alpha']==al and z['arm']==arm)['curve'][-1]['taught']['complete']>=.95 for wn in manifest['worlds']) for arm in manifest['arms'])]
save('CALIBRATION.json',dict(status='ACQUISITION_READY' if eligible else 'ACQUISITION_NOT_READY',eligible=eligible,records=sum(z['records'] for z in rows),targets=sum(z['targets'] for z in rows),fits=len(rows),held_scores_opened=False))
