"""Cycle 1: numerical, event-order and whole-artifact qualification. No addition assessment."""
import os,sys,time,pickle,traceback
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
import numpy as np
import support171 as s
s.STAGE='cycle2_technical_revision'
def main():
 assert not (s.H/'CYCLE2_TECHNICAL_STARTED.json').exists(),'No automatic replay of qualification'
 s.save('CYCLE2_TECHNICAL_STARTED.json',dict(CPU=s.cpu(),wall=s.wall()));s.save('CYCLE2_TECHNICAL_SOURCE_LOCK.json',s.source_lock());s.log('CYCLE2_TECHNICAL_START')
 import native171 as n
 from controller171 import Controller
 import numeric171
 # Code/source pin precedes every experimental birth; fixtures use no arithmetic targets.
 checks=numeric171.check(n.C);s.save('cycles/cycle2/NUMERICAL_CHECKS.json',checks)
 world=202610269790;roles=['ON3','ON1','MIS3','OFF3','CAP-OFF3','EXACT3','EXACT_RECORD'];ctrls=[Controller(world,k,n.C) for k in roles]
 native=n.birth();reference=n.birth();tapes={k:s.Tape('cycle1_'+k) for k in ['NATIVE']+roles};observations=[(bytes([0,255,10,61]),bytes([241])),(bytes([61,13,200,0,255]),bytes([0,255])),(bytes([17,4,0,10,200,61]),bytes([73]))]
 clocks=[]
 for i,(p,a) in enumerate(observations):
  s.records(p,a,1,technical=True);legacy=s.COUNTS['output_distributions'];ref_sites=reference.train(p,a);s.COUNTS['distributions']+=s.COUNTS['output_distributions']-legacy;s.COUNTS['physical_native_records']+=1;s.COUNTS['physical_native_targets']+=len(a)+1
  out=n.train_record(native,ctrls,p,a,tapes,dict(id=f'technical:{i}',phase='technical'),technical=True)
  n.equal(native,reference);assert all(x['argmax']==y['argmax'] and x['lp']==y['lp'] for x,y in zip(out['native'],ref_sites))
  clocks.append(dict(record=i,n=native.n,time=native.time));s.log('TECHNICAL_RECORD_COMMIT',index=i,update_checksums=out['updates'])
 for t in tapes.values():t.flush()
 checks['native_teacher_state_bit_equal']=True;checks['native_teacher_clocks']=clocks
 # One long-context preflight, charged as four observed logical records.
 bench=n.birth();bc=[Controller(world,k,n.C) for k in ['ON3','EXACT3','EXACT_RECORD']];t0=time.process_time();n.train_record(bench,bc,bytes([0,3,255,17,61,10,7]),bytes([0,200]),technical=True);bench_cpu=time.process_time()-t0
 before=s.digest(native);controllers={'NATIVE':None,**{c.role:c for c in ctrls}};rows=[dict(id='tech:query',prompt=observations[0][0].hex(),answer=observations[0][1].hex())]
 panels=n.panel(native,controllers,rows,'cycle1_roundtrip_before');assert s.digest(native)==before
 rec=s.checkpoint((native,ctrls),'cycle1_state.pkl')
 with (s.H/rec['path']).open('rb') as f:restored,rc=pickle.load(f)
 n.equal(native,restored)
 for a,b in zip(ctrls,rc):n.equal(a,b)
 after=n.panel(restored,{'NATIVE':None,**{c.role:c for c in rc}},rows,'cycle1_roundtrip_after')
 assert all(panels[k]['correct']==after[k]['correct'] and panels[k]['TF_negative_log_likelihood']==after[k]['TF_negative_log_likelihood'] for k in panels)
 assert not np.shares_memory(ctrls[0].U,ctrls[1].U)
 for k in ['fast','slow','adapt']:assert not np.shares_memory(getattr(native.bank[0].fly.m,k),getattr(restored.bank[0].fly.m,k))
 checks.update(whole_event_roundtrip=True,operative_restoration=True,mutable_independence=True,persistent_readonly=True,charged_benchmark_seconds=bench_cpu,technical_records=s.COUNTS['technical_records'])
 assert s.COUNTS['technical_records']==31
 # The complete declared fields survived NPZ serialization; no missing margin repair by rerun.
 artifacts=[]
 for p in (s.H/'local').glob('cycle1_*.npz'):
  with np.load(p,allow_pickle=False) as z:
   md=__import__('json').loads(z['metadata'].tobytes());assert len(md)==len(z['p']) and all('margin' in x and 'rival' in x for x in md);assert {'p','p0','v','h','z','a','u','g','direct_error','signal_norms'}<=set(z.files)
  artifacts.append(dict(path=str(p.relative_to(s.H)),sha256=s.sha(p)))
 checks['whole_event_schema']=True;checks['array_artifacts']=artifacts
 # Loaded project Python closure must be present in the source lock.
 lock=__import__('json').loads((s.H/'CYCLE2_TECHNICAL_SOURCE_LOCK.json').read_text());missing=[]
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f and Path(f).resolve().is_relative_to(s.ROOT) and Path(f).suffix=='.py':
   rel=str(Path(f).resolve().relative_to(s.ROOT))
   if rel not in lock:missing.append(rel)
 assert not missing,missing
 s.save('cycles/cycle2/QUALIFICATION.json',dict(status='PASS',checks=checks,source_closure=len(lock),learned_arithmetic_claim=False));s.finish('CYCLE2_TECHNICAL_QUALIFIED_AWAITING_CALIBRATION');s.log('CYCLE2_TECHNICAL_COMPLETE')
if __name__=='__main__':
 try:main()
 except Exception as e:
  s.save('cycles/cycle2/FAILURE.json',dict(stage='cycle1',type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),counts=dict(s.COUNTS),CPU=s.cpu(),wall=s.wall()));s.finish('CYCLE2_TECHNICAL_FAILED');raise
