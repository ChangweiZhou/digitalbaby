import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,time,pickle,json,hashlib,subprocess
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from core import Output,probabilities
from fixtures import make,digest
from util import H,R,save,load,event,cpu,budget
import cycle1,native

FIXTURES=['uniform257','skewed_full_support','four_context_full_support'];SEEDS=[11,12,13];RATES=[.03,.1]
C=native.fastmem.CODE.copy()
roster=[]
for f in FIXTURES:
 for s in SEEDS:
  for eta in RATES:
   for arm in (['A','B','C','C0','BIAS'] if f=='four_context_full_support' else ['A','B','C']):roster.append({'id':f'{f}_s{s}_{eta}_{arm}','fixture':f,'seed':s,'eta':eta,'arm':arm})
assert len(roster)==66
save('MANIFEST.json',{'version':'V150','roster':roster,'targets_per_fit':4096,'validation_per_fit':1024,'substantive_toy_targets':270336,'native_roster_conditional':[f'{t}_{a}' for t in ['bryant','buster'] for a in ['A','B','C']],'native_targets_conditional':12504,'toy_extra_target_allowance':256,'native_partial_fixture_allowance':64,'CPU_limit':7200,'Alice_or_addition_holdbacks':False,'empirical_frequency_rule':'add-one over all257 targets, from train only; matches pinned unigram support convention'})
# Runtime and asset closure before the first learning event.
files={Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(m.__file__).endswith('.py') and Path(m.__file__).resolve().is_relative_to(R)};files|=set(H.glob('*.py'));files|=set((H/'vendor').rglob('*'));files={p for p in files if p.is_file() and '__pycache__' not in p.parts};files|={H/'MANIFEST.json',R/'minifly_a_v3/package/FULL151_CANONICAL_B.npz',R/'minifly_cls_dev_20261017/OUTPUT_CODEBOOK.json'}
save('SOURCE_LOCK.json',{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)});save('RUNTIME.json',{'numpy':np.__version__,'python':sys.version,'threads':1})
cycle1.checks(C);native_qualified=cycle1.native_checks();event(event='CYCLE1_COMPLETE',native_qualified=native_qualified)
worlds={};oracle=[]
for f in FIXTURES:
 for s in SEEDS:
  w=make(f,s);worlds[f,s]=w;np.savez(H/f'DATA_{f}_{s}.npz',q=w['q'],context_bytes=w['context_bytes'],train_ctx=w['train'][0],train_y=w['train'][1],validation_ctx=w['validation'][0],validation_y=w['validation'][1]);event(event='DATA_LOCK',fixture=f,seed=s,sha256=digest(w))
  for ctx,q in enumerate(w['q']):
   def obj(v):
    p,lp=probabilities(C,v);g=C.T@(p-q)/4;g-=g.mean();return float(-q@lp),g
   opt=minimize(obj,np.zeros(16),jac=True,method='L-BFGS-B',options={'maxiter':1000,'gtol':1e-10,'ftol':1e-14});oracle.append({'fixture':f,'seed':s,'context':ctx,'BPS':float(opt.fun/np.log(2)),'max_gradient':float(np.max(abs(opt.jac))),'success':bool(opt.success),'iterations':int(opt.nit),'status':'best_numerical_convex_reference_not_certified_bound'})
save('CONVEX_REFERENCES.json',oracle)
rows=[];phasebegin=cpu()
for rr in roster:
 budget(120);event(event='TOY_FIT_START',**rr);w=worlds[rr['fixture'],rr['seed']];m=Output(C,len(w['q']),rr['arm'],rr['eta']);t=time.process_time();loss=0.
 for n,(ctx,y) in enumerate(zip(*w['train'])):
  loss+=m.observe(int(ctx),int(y))
  if (n+1)%1024==0:
   event(event='TOY_PROGRESS',id=rr['id'],targets=n+1);save('ACTIVE.json',{'stage':'B','id':rr['id'],'targets':n+1,'CPU':cpu()})
 before=m.digest();preds=[];val_loss=0.
 for ctx,y in zip(*w['validation']):
  p,lp=m.read(int(ctx));val_loss-=float(lp[y]);preds.append([int(ctx),int(y),float(lp[y]),int(np.argmax(p))])
 assert before==m.digest()
 expected=[]
 for i,q in enumerate(w['q']):expected.append(float(-q@m.read(i)[1]/np.log(2)))
 counts=np.bincount(w['train'][1],minlength=257);freq=(counts+1)/(counts.sum()+257);freq_bps=float(np.mean([-q@np.log2(freq) for q in w['q']]))
 result=rr|{'targets':m.n,'validation_targets':len(preds),'validation_BPS':val_loss/len(preds)/np.log(2),'expected_BPS':float(np.mean(expected)),'expected_BPS_by_context':expected,'frequency_expected_BPS':freq_bps,'entropy_BPS':float(np.mean([-q@np.log2(q) for q in w['q']])),'training_prequential_BPS':loss/m.n/np.log(2),'data_hash':digest(w),'code_drift':float(np.linalg.norm(m.C-C)),'validation_rows':preds,'CPU':time.process_time()-t}
 (H/(rr['id']+'.pkl')).write_bytes(pickle.dumps(m));save(rr['id']+'.json',result);rows.append(result);event(event='TOY_FIT_COMMIT',id=rr['id'],targets=m.n)
 save('COMPACT.json',[{k:z[k] for k in z if k!='validation_rows'} for z in rows])
 if cpu()-phasebegin>1200:raise RuntimeError('Stage B admission budget exceeded; no roster extension')
main=[z for z in rows if z['arm'] in ['A','B','C']];means={str(eta):float(np.mean([z['validation_BPS'] for z in main if z['eta']==eta])) for eta in RATES};eta=min(RATES,key=lambda e:(means[str(e)],e))
def get(f,s,a):return next(z for z in rows if z['fixture']==f and z['seed']==s and z['arm']==a and z['eta']==eta)
B=[]
for f in FIXTURES[1:]:
 aa=np.mean([get(f,s,'A')['expected_BPS'] for s in SEEDS]);bb=np.mean([get(f,s,'B')['expected_BPS'] for s in SEEDS]);ref=np.mean([z['BPS'] for z in oracle if z['fixture']==f]);B.append({'fixture':f,'A':float(aa),'B':float(bb),'reference':float(ref),'B_better_and_closer':bool(bb<aa-1e-8 and abs(bb-ref)<abs(aa-ref)-1e-8)})
conditional=[]
for s in SEEDS:
 cc=get(FIXTURES[2],s,'C');null=get(FIXTURES[2],s,'C0');conditional.append({'seed':s,'C':cc['expected_BPS'],'context_ignored':null['expected_BPS'],'frequency':cc['frequency_expected_BPS'],'passed':bool(cc['expected_BPS']<min(null['expected_BPS'],cc['frequency_expected_BPS'])-1e-8)})
gate=all(z['B_better_and_closer'] for z in B) and all(z['passed'] for z in conditional)
save('GATE.json',{'selected_eta':eta,'validation_means':means,'B_improvement':B,'C_conditional':conditional,'output_learning_ready':gate,'native_sign_qualified':native_qualified,'native_admitted':gate and native_qualified,'status':'READY' if gate else 'OUTPUT_LEARNING_NOT_READY'})
if gate and native_qualified:native.run(eta)
save('EXECUTION_SUMMARY.json',{'version':'V150','status':'NATIVE_DEVELOPMENT_COMPLETED' if gate and native_qualified else ('OUTPUT_LEARNING_NOT_READY' if not gate else 'INTEGRATION_NOT_QUALIFIED'),'toy_fits':len(rows),'toy_targets':sum(z['targets'] for z in rows),'validation_probes':sum(z['validation_targets'] for z in rows),'native_full_histories':6 if gate and native_qualified else 0,'native_targets':12504 if gate and native_qualified else 0,'partial_native_fixtures':64,'extra_synthetic_target_events':0,'CPU_including_waited_children':cpu(),'source_model_adoption':False,'addition_holdbacks_opened':False})
print(json.dumps(load('GATE.json'),indent=2),flush=True)
