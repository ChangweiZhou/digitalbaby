"""Cycle3 repair and artifact audit; no replay of C2 histories."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,pickle,copy,hashlib
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H.parent
sys.path[:0]=[str(R/'minifly_v144_r2_exercise_20261024'),str(R/'minifly_v144_dev_20261024')]
from r2_core import trace_from_parent,controls,orbit_rates
from reference import Reference,ByteTrace,seed
import task
START=time.process_time();base=trace_from_parent();FM=sys.modules['lb1'].fastmem
class FE(FM.FEPerm):
 def __init__(self,tab):super().__init__();self.tab=tab
 def feed(self,b,t):self.advance(t);self.p=self.p[FM.PERM]+self.tab[int(b)]
class Trace(ByteTrace):
 def __init__(self,tab):self.tab=tab.copy();super().__init__(self.tab,FM.PERM,FM.DELTA,self.make)
 def make(self):return FE(self.tab)
w=task.world('V145-EXERCISE-C3-RESUME-REPAIR');pairs=task.oriented(w['training_order']);m=Reference(Trace(base.table),'V145-resume',128,.03,.001,'BP');r=np.random.default_rng(812)
def one(m,r):
 for j in r.permutation(len(pairs)):p=pairs[j];m.train(task.prompt(p),task.answer(p))
one(m,r);blob=pickle.dumps((m,r));one(m,r);a,b=pickle.loads(blob);one(a,b);assert m.state_hash()==a.state_hash()
# Saved-output audit with exact roster and orbit reduction; reject tampering.
rows=json.loads((H/'CYCLE2.json').read_text());w2=task.world('V145-EXERCISE-C2')
expected={('SYMBOL','BP'),('ORACLE-LINE','BP'),('ORACLE-CIRCLE','BP')}|{(a,b) for a in ['ORDER','PERM','BAG','NO-WRITE','ORDER-BP'] for b in ['BP','DFA']}
def valid(rows):
 try:
  assert len(rows)==len(expected) and {(r['arm'],r['mode']) for r in rows}==expected
  for r in rows:
   passes=256 if r['arm'] in ['SYMBOL','ORACLE-LINE','ORACLE-CIRCLE'] else 64
   assert r['passes']==passes and r['records']==len(task.oriented(w2['training_order']))*passes
   for label,pp in [('held',w2['held']),('taught',w2['training_order'])]:
    z=r['scores'][label];assert sorted(tuple(x['pair']) for x in z['rows'])==sorted(task.oriented(pp))
    for x in z['rows']:assert x['exact']==(x['status']=='EOS' and bytes.fromhex(x['output_hex'])==task.answer(x['pair']))
    assert np.isfinite(z['orbit_exact']) and abs(z['orbit_exact']-orbit_rates(z['rows'])['complete'])<1e-12
  return True
 except (AssertionError,KeyError,ValueError,TypeError):return False
assert valid(rows);bad=copy.deepcopy(rows);bad[0]['scores']['held']['rows'][0]['exact']=not bad[0]['scores']['held']['rows'][0]['exact']
mut=[rows[:-1],rows+[rows[0]],bad];bad=copy.deepcopy(rows);bad[0]['scores']['held']['orbit_exact']=float('nan');mut.append(bad);assert not any(valid(x) for x in mut)
locks=json.loads((H/'SOURCE_LOCK.json').read_text());assert all(hashlib.sha256((R/p).read_bytes()).hexdigest()==v for p,v in locks.items())
c=controls(w2,40,Trace(base.table));(H/'CONTROLS.json').write_text(json.dumps(c,indent=2)+'\n')
w3=task.world('V145-EXERCISE-C3');n2=len(task.oriented(w2['training_order']));n3=len(task.oriented(w3['training_order']));nr=len(pairs)
records=sum(x['records'] for x in rows)+5*2048+128+4*n3+3*nr+4
targets=sum(x['targets'] for x in rows)+5*2048*10+128*10+4*sum(len(task.answer(p))+1 for p in task.oriented(w3['training_order']))+3*sum(len(task.answer(p))+1 for p in pairs)+4
summary=dict(status='THREE_BOUNDED_CYCLES_COMPLETE_WITH_LIMITATIONS',records=records,observed_targets=targets,main_worlds_opened=False,formal_jobs=0,new_core_adopted=False,checkpoint_failure_preserved=True,resume_live_vs_restored_equal=True,artifact_mutations_rejected=4,exact_C2_roster_accepted=True,source_locks_match=True,C3_repair_records=3*nr,repair_process_cpu=time.process_time()-START,prior_CPU_lower_bound=json.loads((H/'PROGRESS.json').read_text())['cpu'],full_cost_measured=False,controls={k:v['orbit_complete'] for k,v in c.items()},results=[dict(arm=x['arm'],mode=x['mode'],passes=x['passes'],held=x['scores']['held']['orbit_exact'],taught=x['scores']['taught']['orbit_exact']) for x in rows])
(H/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
