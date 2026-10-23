"""Artifact-only reduction. Imports no learner, optimizer or native runtime."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib,ast,contextlib
import numpy as np
from artifact_inference import logits_for
P=Path(__file__).resolve().parent;R=P.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def output(z):
 result=[]
 for a in z:
  bs=[];eos=False
  for j,q in enumerate(a):
   if q[8]>0:eos=True;break
   if j==8:break
   bs.append(int(sum(1<<k for k in range(8) if q[k]>0)))
  result.append(dict(bytes=bs,EOS=eos,truncation=not eos))
 return result

def main():
 auth=json.loads((P/'AUTHORIZATION.json').read_text());assert sha(R/'PROJECT_DOCUMENTS/CURRENT_ARITHMETIC_EXPERIMENT_DESIGN.json')==auth['arithmetic_pointer_sha256']
 for n,h in auth['native_sources'].items():assert sha(R/n)==h
 freeze=json.loads((P/'HEAD_FREEZE.json').read_text());assert all(sha(P/'heads'/n)==h for n,h in freeze['files'].items())
 fitted=[]
 for f in sorted((P/'heads').glob('*.json')):
  d=json.loads(f.read_text());assert d['updates']<=2000 and d['terms']<=2000000;fitted.append(dict(name=f.stem,updates=d['updates'],terms=d['terms'],training_loss=d['training_loss']))
 assert len(fitted)==12
 for f in (P/'heads').glob('*.npz'):
  with np.load(f) as z:assert sum(z[k].size for k in z.files)==5801
 # Candidate module cannot access evaluator data, disk, keys or native objects.
 tree=ast.parse((P/'head.py').read_text());imports=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.Import):imports.extend(n.name for n in node.names)
 assert imports==['numpy']
 records=0;forks=0;reduced=[];counterfactual=[];table_checks=0
 for f in sorted((P/'traces').glob('*.json')):
  if f.stem.endswith('_decoded'):continue
  meta=json.loads(f.read_text());records+=504
  if 'OLD_S_NEW' in meta['pools']:records+=252;forks+=1
  if meta['seed']<202610229200:continue
  dec=json.loads(f.with_name(f.stem+'_decoded.json').read_text());meta_labels=dict(zip(meta['pools'],meta['labels']))
  derived=f.with_name(f.stem+'_decoded.npz')
  with np.load(f.with_suffix('.npz'),allow_pickle=False) as raw,(np.load(derived,allow_pickle=False) if derived.exists() else contextlib.nullcontext(None)) as z:
   for row in dec['results']:
    pool=row['pool'];vals=meta_labels[pool]
    logits=logits_for(P,meta,row,raw)
    if z is not None and row['readout']!='DIRECT':assert np.allclose(logits,z[row['array']],atol=1e-12,rtol=1e-12)
    out=output(logits);assert out==row['metrics']['outputs']
    if pool=='ABSENT':
     assert float(np.mean([o['EOS'] and not o['bytes'] for o in out]))==row['metrics']['empty_emission'];continue
    correct=[v==o['bytes'] and o['EOS'] for v,o in zip(vals,out)];non=[i for i,v in enumerate(vals) if len(v)>0]
    assert float(np.mean(correct))==row['metrics']['exact'];assert float(np.mean([correct[i] for i in non]))==row['metrics']['nonempty_exact']
    bit_n=bit_d=0
    for i,v in enumerate(vals):
     for j,t in enumerate(v):bit_n+=int(np.sum((logits[i,j,:8]>0)==((int(t)>>np.arange(8))&1)));bit_d+=8
    assert bit_n/bit_d==row['metrics']['bits']
    for L,value in row['metrics']['per_length'].items():assert float(np.mean([correct[i] for i,v in enumerate(vals) if len(v)==int(L)]))==value
    if pool in ('OLD_final','NEW_final','OLD_S_NEW','NEW_S_NEW'):
     reduced.append(dict(seed=meta['seed'],life=meta['life'],STD=meta['STD'],arm=meta['arm'],cohort=pool,readout=row['readout'],head_seed=row.get('head_seed'),lesion=row['lesion'],exact=float(np.mean(correct)),nonempty_exact=float(np.mean([correct[i] for i in non]))))
   # Evaluator-only exact table, not a candidate. Known W records must be solvable.
   if meta['arm']=='W':
    directory={k:v for k,v in zip(meta['keys_hex'][:18],meta['observed_OLD'])}
    directory.update({k:v for k,v in zip(meta['keys_hex'][18:36],meta_labels['NEW_end'])})
    for pool,kk in [('OLD_final',meta['keys_hex'][:18]),('NEW_final',meta['keys_hex'][18:36])]:
     got=[directory[k] for k in kk];assert got==meta_labels[pool]
     assert all(len(v)<=8 and all(0<=b<256 for b in v) for v in got);table_checks+=len(got)
 # Physical trace manifest gives the independent episode/branch count.
 assert records==54432 and forks==24
 assert sum(int(json.loads((P/'cycles'/f'CYCLE{c}_RESULT.json').read_text()).get('native_records',0)) for c in (2,3))==records
 # Counterfactual metadata and head-weight freeze certify no head updates between lives.
 for f in sorted((P/'assets').glob('*_counterfactual.json')):counterfactual.append(dict(pair=f.stem,**json.loads(f.read_text())))
 assert len(counterfactual)==4
 pairs=[]
 for seed in range(202610229301,202610229305):
  for std in (False,True):
   for view in ('MEAN','SEP'):
    for cohort in ('OLD_final','NEW_final'):
     metrics={}
     for arm,lesion in [('W','NONE'),('N','NONE'),('SH','NONE'),('W','ZERO'),('W','FOREIGN')]:
      rr=[r for r in reduced if r['seed']==seed and r['STD']==std and r['readout']==view and r['cohort']==cohort and r['arm']==arm and r['lesion']==lesion]
      assert len(rr)==6;metrics[f'{arm}_{lesion}']=float(np.mean([r['nonempty_exact'] for r in rr]))
     direct=[r for r in reduced if r['seed']==seed and r['STD']==std and r['readout']=='DIRECT' and r['cohort']==cohort and r['arm']=='W'];assert len(direct)==2
     metrics['DIRECT']=float(np.mean([r['nonempty_exact'] for r in direct]));metrics['passes_direction']=all(metrics['W_NONE']>v for k,v in metrics.items() if k!='W_NONE')
     pairs.append(dict(pair=seed,STD=std,readout=view,cohort=cohort,**metrics))
 overview=[]
 for std in (False,True):
  for view in ('MEAN','SEP'):
   d={'STD':std,'readout':view}
   for cohort in ('OLD_final','NEW_final'):
    rr=[r for r in pairs if r['STD']==std and r['readout']==view and r['cohort']==cohort];d[cohort]=float(np.mean([r['W_NONE'] for r in rr]))
   d['joint_positive_pilot']=all(r['passes_direction'] for r in pairs if r['STD']==std and r['readout']==view);overview.append(d)
 summary=dict(status='THREE_CYCLES_FULL_BOUNDED_PILOT_COMPLETE',native_records=records,native_main_events=records*5,extra_E0_events=4,cloned_OLD_information_equivalents=6048,head_fits=fitted,overview=overview,pair_rows=pairs,counterfactual_pairs=counterfactual,all_final_rows=reduced,table_known_value_checks=table_checks,arithmetic_pointer_unchanged=True,trained_host=False)
 (P/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
 (P/'FINAL_AUDIT.json').write_text(json.dumps(dict(integrity='ACCEPTED_ARTIFACT_REDUCTION',independent_process_not_independent_reviewer=True,native_learning_rerun=False,head_fitting_rerun=False,counts_and_emissions_recomputed=True,bit_and_per_length_metrics_recomputed=True,frozen_inference_independently_reconstructed=True,head_freeze_verified=True,original_arithmetic_and_native_sources_unchanged=True,physical_records=records,logical_record_equivalents=60480,paired_direction_rule_recomputed=True,automatic_adoption=False),indent=2)+'\n')
 print(json.dumps(overview,indent=2))
if __name__=='__main__':main()
