"""Artifact-only independent reduction. No learner/native imports or fitting."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
from pathlib import Path
import json,hashlib
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def metric(z,vals):
 non=[i for i,v in enumerate(vals) if v];ok=[];pl=[];st=[];out=[]
 for q,v in zip(z,vals):
  bs=[];end=False
  for j in range(9):
   if q[j,8]>0:end=True;break
   if j==8:break
   bs.append(sum(1<<b for b in range(8) if q[j,b]>0))
  out.append(dict(bytes=bs,EOS=end));ok.append(end and bs==v)
  pl.append(all(sum(1<<b for b in range(8) if q[j,b]>0)==t for j,t in enumerate(v)));st.append(bool(np.all(q[:len(v),8]<=0) and q[len(v),8]>0))
 ix=[i for i in non if pl[i]]
 return dict(exact=float(np.mean(ok)),nonempty_exact=float(np.mean([ok[i] for i in non])) if non else None,payload_exact=float(np.mean([pl[i] for i in non])) if non else None,conditional_n=len(ix),conditional_stop=float(np.mean([st[i] for i in ix])) if ix else None,outputs=out)
def main():
 lock=json.loads((P/'PARENT_CLOSURE.json').read_text())
 for n,h in lock.items():assert sha(R/n)==h,n
 for n,h in json.loads((P/'SOURCE_LOCK.json').read_text()).items():assert sha(P/n)==h,n
 for n,h in json.loads((P/'RUNTIME_MODULE_CLOSURE.json').read_text()).items():assert sha(Path(n))==h,n
 freeze=json.loads((P/'HEAD_FREEZE.json').read_text())
 for n,h in freeze.items():assert sha(P/'heads'/n)==h
 rows=[];files=0;records=0
 for f in sorted((P/'traces').glob('*_outputs.json')):
  meta=json.loads(f.read_text());files+=1;records+=504
  if 'OLD_S_NEW' in meta['pools']:records+=252
  with np.load(f.with_name(f.name.replace('_outputs.json','.npz'))) as raw:
   for row in meta['results']:
    D=raw[f"D{meta['pools'].index(row['pool'])}"];vals=meta['labels'][meta['pools'].index(row['pool'])]
    if row['lesion']=='ZERO':D=np.zeros_like(D)
    elif row['lesion']=='FOREIGN':D=np.roll(D,1,axis=0)
    z=D.mean(-1).copy()
    if row['head']!='DIRECT':
     scale=np.load(P/'heads'/f"scale_{int(meta['STD'])}.npy");q=D/scale[None,None,:,None];q=np.repeat(q.mean(-1,keepdims=True),2,axis=-1);X=np.zeros((len(q),9,171));X[:,:,:162]=q.reshape(len(q),1,162);X[:,:,162:]=np.eye(9)
     with np.load(P/'heads'/f"{int(meta['STD'])}_{row['head']}.npz") as p:z[:,:,8:9]=np.tanh(X@p['w1']+p['b1'])@p['w2']+p['b2']+q.mean(-1)[:,:,8:9]
    m=metric(z,vals);assert m==row['metrics'],(f,row['pool'],row['head'])
    rows.append(dict(seed=meta['seed'],life=meta['life'],STD=meta['STD'],arm=meta['arm'],pool=row['pool'],head=row['head'],lesion=row['lesion'],**{k:v for k,v in m.items() if k!='outputs'}))
 assert files==384 and records==225792,(files,records)
 def mean(rr,k):
  vs=[r[k] for r in rr];return None if not vs or any(v is None for v in vs) else float(np.mean(vs))
 A=[]
 for seed in range(202610239601,202610239633):
  for std in (False,True):
   effects=[]
   for pool in ('OLD_end','NEW_final','OLD_S_NEW'):
    rr=[r for r in rows if r['seed']==seed and r['STD']==std and r['arm']=='W' and r['pool']==pool and r['lesion']=='NONE']
    d=[r for r in rr if r['head']=='DIRECT'];h=[r for r in rr if r['head']!='DIRECT']
    a,b=mean(h,'conditional_stop'),mean(d,'conditional_stop')
    effects.append(dict(pool=pool,head_conditional=a,direct_conditional=b,conditional_gain=None if a is None or b is None else a-b,complete_gain=mean(h,'nonempty_exact')-mean(d,'nonempty_exact'),eligible=sum(r['conditional_n'] for r in d)))
   gain=mean(effects,'conditional_gain');cg=mean(effects,'complete_gain');A.append(dict(pair=seed,STD=std,checkpoints=effects,mean_conditional_gain=gain,mean_complete_gain=cg,component_positive=gain is not None and gain>0 and cg>0))
 from scipy.stats import t
 results=[]
 for std in (False,True):
  rr=[r for r in A if r['STD']==std];v=[r['mean_conditional_gain'] for r in rr];defined=all(x is not None for x in v)
  mu=float(np.mean(v)) if defined else None;se=float(np.std(v,ddof=1)/np.sqrt(32)) if defined else None;interval=[mu-float(t.ppf(.975,31))*se,mu+float(t.ppf(.975,31))*se] if defined else None
  results.append(dict(STD=std,confirmatory=std,conditional_gain=mu,paired_95_interval=interval,positive_pairs=sum(x['component_positive'] for x in rr),all_denominators_defined=defined,passes=bool(defined and interval[0]>0 and all(x['component_positive'] for x in rr)),mean_complete_gain=mean(rr,'mean_complete_gain')))
 summary=dict(status='OFFICIAL_STOP_ONLY_EXECUTION_COMPLETE',records=records,events=records*5,assessment_pairs=32,head_fits=0,primary=next(x for x in results if x['STD']),storage_results=results,pair_results=A,final_rows=[r for r in rows if r['pool'] in ('OLD_final','NEW_final','OLD_S_NEW')])
 (P/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
 (P/'FINAL_AUDIT.json').write_text(json.dumps(dict(status='ACCEPTED_ARTIFACT_REDUCTION',source_birth_and_runtime_hashes_verified=True,heads_unchanged=True,independently_reconstructed_saved_inference=True,records=records,native_rerun=False,head_fitting=False,arithmetic_pointer_unchanged=True,independent_process_not_independent_reviewer=True),indent=2)+'\n')
 print(json.dumps(results,indent=2))
if __name__=='__main__':main()
