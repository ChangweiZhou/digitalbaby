"""Read saved readiness output, qualify reducer, correct per-interface controls; no training."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib,ast,time,copy
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H.parent;START=time.process_time()
sys.path[:0]=[str(R/'minifly_learned_bank_v1_20261024'),str(R/'minifly_v144_dev_20261024'),str(R/'minifly_v144_r2_exercise_20261024')]
import lb1,task
from r2_core import controls
FM=lb1.fastmem
# Compile only the immutable trace class; never execute the training runner.
node=next(n for n in ast.parse((H/'run.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Trace');env=dict(np=np,FM=FM,lb1=lb1);exec(compile(ast.Module(body=[node],type_ignores=[]),'trace_definition','exec'),env);Trace=env['Trace']
expected={f'{w}_{a}_{al}_{rh}' for w in ['V146-D1','V146-D2'] for a in ['SYMBOL','ORACLE'] for al in [.003,.03] for rh in [0,.01,.03]}
rows=[json.loads((H/(name+'.json')).read_text()) for name in sorted(expected)]
def valid(rows):
 try:
  assert len(rows)==24 and {x['id'] for x in rows}==expected
  for z in rows:
   w=task.world(z['world']);pairs=task.oriented(w['training_order']);assert z['records']==1024*len(pairs);assert z['targets']==1024*sum(len(task.answer(p))+1 for p in pairs)
   assert [c['pas'] for c in z['curve']]==[16,64,256,512,1024]
   for c in z['curve']:
    for label,pp in [('held',w['held']),('taught',w['training_order'])]:
     q=c[label];assert sorted(tuple(x['pair']) for x in q['rows'])==sorted(task.oriented(pp));groups={}
     for rr in q['rows']:
      assert rr['exact']==(rr['status']=='EOS' and bytes.fromhex(rr['output_hex'])==task.answer(rr['pair']));groups.setdefault(tuple(sorted(rr['pair'])),[]).append(rr['exact'])
     assert np.isfinite(q['accuracy']) and abs(q['accuracy']-np.mean([np.mean(v) for v in groups.values()]))<1e-12
  return True
 except (AssertionError,KeyError,TypeError,ValueError):return False
assert valid(rows);mut=[rows[:-1],rows+[rows[0]]]
b=copy.deepcopy(rows);b[0]['curve'][-1]['held']['accuracy']=float('nan');mut.append(b)
b=copy.deepcopy(rows);b[0]['curve'][-1]['held']['rows'][0]['exact']=not b[0]['curve'][-1]['held']['rows'][0]['exact'];mut.append(b)
b=copy.deepcopy(rows);b[0]['records']+=1;mut.append(b);assert not any(valid(x) for x in mut)
for name,h in json.loads((H/'SOURCE_LOCK.json').read_text()).items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h
scores={}
for wn in ['V146-D1','V146-D2']:
 for arm in ['SYMBOL','ORACLE']:
  tab=np.zeros((256,32))
  if arm=='ORACLE':
   for k,b in enumerate(b'0123456789'):
    v=np.exp(-(np.arange(32)-31*k/9)**2/2);tab[b]=v/np.linalg.norm(v)
   for b in set(range(256))-set(b'0123456789'):tab[b]=np.ones(32)/np.sqrt(32)
  ctrl=controls(task.world(wn),40,Trace(tab));scores[wn+'_'+arm]={k:v['orbit_complete'] for k,v in ctrl.items()}
ready=[]
for al in [.003,.03]:
 for rh in [.01,.03]:
  cells=[z for z in rows if z['arm']=='ORACLE' and z['alpha']==al and z['rho']==rh];ready.append(dict(alpha=al,rho=rh,ready=all(z['curve'][-1]['taught']['accuracy']>=.9 and z['curve'][-1]['held']['accuracy']>=max(scores[z['world']+'_ORACLE'].values())+.2 for z in cells)))
summary=json.loads((H/'SUMMARY.json').read_text());summary.update(dict(status='REFERENCE_PIPELINE_READY' if any(x['ready'] for x in ready) else 'REFERENCE_PIPELINE_NOT_READY',actual_interface_controls=scores,corrected_readiness=ready,exact_roster_valid=True,mutation_fixtures_rejected=5,audit_cpu_including_imports=time.process_time()-START,control_correction='Runner used identity-only shortcut scores; final audit substitutes actual augmented-trace controls without replaying training.'))
(H/'FINAL_AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
