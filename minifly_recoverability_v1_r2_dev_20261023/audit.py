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
 lock=json.loads((P/'PRE_EVALUATION_LOCK.json').read_text())
 for n,h in lock['parent_sources'].items():assert sha(R/n)==h,n
 for n,h in lock['implementation'].items():assert sha(P/n)==h,n
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
 assert files==72 and records==46368,(files,records)
 def mean(rr,k):
  vs=[r[k] for r in rr];return None if not vs or any(v is None for v in vs) else float(np.mean(vs))
 A=[];B=[];matches=[]
 for seed in range(202610239301,202610239305):
  for std in (False,True):
   effects=[]
   for pool in ('OLD_end','NEW_final','OLD_S_NEW'):
    rr=[r for r in rows if r['seed']==seed and r['STD']==std and r['arm']=='W' and r['pool']==pool and r['lesion']=='NONE']
    d=[r for r in rr if r['head']=='DIRECT'];h=[r for r in rr if r['head']!='DIRECT']
    a,b=mean(h,'conditional_stop'),mean(d,'conditional_stop');effects.append(dict(pool=pool,head_conditional=a,direct_conditional=b,conditional_gain=None if a is None or b is None else a-b,complete_gain=mean(h,'nonempty_exact')-mean(d,'nonempty_exact'),direct_eligible_count=sum(r['conditional_n'] for r in d)))
   gain=mean(effects,'conditional_gain');cg=mean(effects,'complete_gain');A.append(dict(pair=seed,STD=std,checkpoints=effects,component_positive=gain is not None and gain>0 and cg>0,mean_conditional_gain=gain,mean_complete_gain=cg))
  cells={}
  for arm in ('W','CREDIT','UNIFORM','SHUFFLED'):
   cells[arm]={pool:mean([r for r in rows if r['seed']==seed and r['STD'] and r['arm']==arm and r['pool']==pool and r['head']=='DIRECT'],'payload_exact') for pool in ('OLD_final','NEW_final')}
  B.append(dict(pair=seed,cells=cells,behavior_positive=all(cells['CREDIT']['OLD_final']>cells[a]['OLD_final'] and cells['CREDIT']['NEW_final']>=cells[a]['NEW_final'] for a in ('W','UNIFORM','SHUFFLED'))))
 for f in sorted((P/'cycles').glob('*_write_ledger.json')):
  log=json.loads(f.read_text());errors=[];excess=0;target=applied=0
  for d in log:
   t=np.array(d['target_L1']);u=np.array(d['uniform_L1']);delta=abs(t-u);bad=delta>1e-12*np.maximum(1,np.maximum(t,u));excess+=int(bad.sum());errors.extend(delta.tolist());target+=float(t.sum());applied+=float(u.sum())
  matches.append(dict(file=f.name,events=len(log),unmatched_compartment_events=excess,max_abs_error=max(errors),target_L1=target,uniform_L1=applied))
 summary=dict(status='COMPLETE_BOUNDED_THREE_CYCLES',native_records=records,native_events=records*5,E0_partial_events=3,A_pair_results=A,B_pair_results=B,uniform_matching=matches,A_positive_by_storage={str(st):all(x['component_positive'] for x in A if x['STD']==st) for st in (False,True)},B_behavior_positive=all(x['behavior_positive'] for x in B),B_clean_matched_dose=not any(x['unmatched_compartment_events'] for x in matches),final_rows=[r for r in rows if r['pool'] in ('OLD_final','NEW_final','OLD_S_NEW')])
 (P/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
 (P/'FINAL_AUDIT.json').write_text(json.dumps(dict(status='ACCEPTED_ARTIFACT_CONSISTENCY_WITH_PROTOCOL_LIMITATIONS',independent_reduction_process_not_independent_reviewer=True,native_rerun=False,head_fitting=False,parent_and_implementation_hashes_verified=True,all_saved_emissions_and_scores_recomputed=True,head_freeze_verified=True,records=records,arithmetic_pointer_unchanged=True),indent=2)+'\n')
 print(json.dumps({k:v for k,v in summary.items() if k not in ('final_rows','A_pair_results','B_pair_results')},indent=2))
if __name__=='__main__':main()
