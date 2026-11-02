"""Read-only smoke checks and prospective source/data lock, before scored histories."""
import util as U
import learner as M
import evaluate as E
import sys,time
T=M.torch;np=M.np
assert U.load('INTERFACE_SMOKE.json')['status']=='PASS'
E.native() # Import closure only; reuse saved successful smoke checks without rerunning them.
files=set(U.H.glob('*.py'))|{U.H/x for x in ['AUTHORIZATION.json','EXECUTION_SPEC.json','DATA_AUDIT.json','EXPORT.json','QUALIFICATION.json','RESOURCE_ADMISSION.json','SOURCE_RECOVERY.json','INTERFACE_SMOKE.json']}
files|=set((U.H/'local/source').rglob('*.py'))|set((U.H/'local/source').rglob('*.json'))
for mod in list(sys.modules.values()):
 p=getattr(mod,'__file__',None)
 if p:
  p=U.Path(p).resolve()
  if p.is_file() and p.suffix=='.py' and p.is_relative_to(U.ROOT):files.add(p)
for s in U.load('EXPORT.json')['splits']:files.add(U.H/'local'/f'{s}.npz')
files|={U.H/'local/records.pkl'}|set((U.H/'local').glob('*subsets.npz'))
lock={str(p.relative_to(U.ROOT)):U.sha(p) for p in sorted(files)}
U.save('SOURCE_LOCK.json',dict(epoch=time.time(),source_design_commit='0ec991e8818c7554d9f1bcdb4fc42fd73081dfc1',stage='before any V179 scored learner training',files=lock,learned_state_reuse=False))
print('SOURCE_LOCK_READY',len(lock),flush=True)
