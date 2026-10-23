"""Artifact-only independent reduction process; no model imports or reruns."""
from pathlib import Path
import json,hashlib,numpy as np
P=Path(__file__).resolve().parent
summary=[];counts=[]
for c in (1,2,3):
 result=json.loads((P/'cycles'/f'CYCLE{c}_RESULT.json').read_text());z=np.load(P/'cycles'/f'CYCLE{c}_arrays.npz',allow_pickle=False)
 plan=json.loads((P/'cycles'/f'CYCLE{c}_PLAN.json').read_text());assert hashlib.sha256((P/'cycles'/f'CYCLE{c}_SOURCE.py').read_bytes()).hexdigest()==plan['runner_sha']
 terms=z['event_terms'];assert terms.shape[1]==40;assert int(np.sum(terms[:,1]==0))==result['records'];assert np.isfinite(terms).all()
 assert np.allclose(terms[:,22:31],terms[:,3,None]*np.tanh(terms[:,4:13]/2)/2*(np.arange(9)[None,:]>=np.where(terms[:,2,None]==256,8,0)),atol=1e-15)
 coef=terms[:,31:40];target=terms[:,13:22];pred=terms[:,22:31]
 assert np.all(np.all(np.isclose(coef,0,atol=1e-15),axis=1)|np.all(np.isclose(coef,target,atol=1e-15),axis=1)|np.all(np.isclose(coef,target+pred,atol=1e-15),axis=1))
 for row in result['rows']:
  m=z[row['array']];vals=row['values'];outputs=[]
  for a in m:
   out=[];eos=False
   for j,t in enumerate(a):
    if t[8]>0:eos=True;break
    if j==8:break
    out.append(int(sum(1<<b for b in range(8) if t[b]>0)))
   outputs.append({'bytes':out,'EOS':eos,'truncation':not eos})
  assert outputs==row['metrics']['outputs']
  exact=np.mean([v==a['bytes'] and a['EOS'] for v,a in zip(vals,outputs)]);assert exact==row['metrics']['exact']
  # ABSENT keys do not have an empty-string target: never count their exact field as success.
  if '_ABSENT' in row['label']:continue
  if any(row['label'].endswith(t) for t in ('_final_OLD','_final_NEW')):
   summary.append({'cycle':c,'label':row['label'],'view':row['view'],'exact':float(exact),'bits':row['metrics']['bit_accuracy'],'calibrated_exact':row.get('calibrated',{}).get('exact')})
 counts.append({'cycle':c,'records':result['records'],'native_main_events':len(terms),'synthetic_events':result['synthetic_events'],'CPU_seconds':result['CPU_seconds']})
assert sum(x['records'] for x in counts)==13228
arith=P.parent/'PROJECT_DOCUMENTS/CURRENT_ARITHMETIC_EXPERIMENT_DESIGN.json';assert hashlib.sha256(arith.read_bytes()).hexdigest()==(P/'ARITHMETIC_POINTER_SHA.txt').read_text()
out={'status':'THREE_BOUNDED_TECHNICAL_CYCLES_COMPLETE','counts':counts,'total_records':13228,'science_jobs':0,'arithmetic_unchanged':True,'rows':summary}
(P/'SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n')
(P/'FINAL_AUDIT.json').write_text(json.dumps({'integrity':'ACCEPTED_ARTIFACT_REDUCTION','independent_process_not_independent_reviewer':True,'source_snapshots_verified':True,'outputs_and_record_counts_recomputed':True,'prediction_terms_verified':True,'full_campaign_executed':False,'arithmetic_pointer_unchanged':True,'automatic_adoption':False},indent=2)+'\n')
print(json.dumps(counts))
