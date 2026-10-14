import sys,json,gzip,ast,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from auditor import audit
from worlds import make_world
p=next((root/'cycles/cycle2/receipts').glob('*.json.gz'))
with gzip.open(p,'rt') as f:r=json.load(f)
s=r['source_identity'];checks=[]
audit(r,s);checks.append('complete receipt independently accepted including resources')
rec=r['addition']['N_ADD']['records'][0]
cases=[(rec,'allowed',True),(rec,'all_bytes',not rec['all_bytes']),(rec,'permissions',[True]*len(rec['permissions'])),(rec,'counters_before',[1,1,1]),(r,'technical',False),(r,'record_applications',1),(r['addition']['W']['checkpoints'][0],'parent_after','tampered'),(r['resource'],'production_cpu',0.)]
for obj,key,bad in cases:
 old=obj[key];obj[key]=bad
 try:
  try:audit(r,s)
  except (AssertionError,ValueError,KeyError):checks.append('rejected '+key)
  else:raise AssertionError('tamper accepted '+key)
 finally:obj[key]=old
plan=make_world(r['label'])
for op,task in plan['tasks'].items():
 for q in task['taught']+task['held']:
  text=bytes.fromhex(q['prompt_hex']).decode();a,b=map(int,text[:-1].split(op));expected=a+b if op=='+' else a*b
  assert bytes.fromhex(q['answer_hex'])==str(expected).encode()
 for i,q in enumerate(task['records']):assert len(bytes.fromhex(q['answer_hex']))==len(bytes.fromhex(task['shuffled_answers'][i]))
checks.append('independent arithmetic label and length-preserving shuffle checks')
assert r['scratch_state_parity']['accepted'] and r['scratch_state_parity']['counting']==r['scratch_state_parity']['after_disabled_addition']
checks.append('scratch learning and optimizer state parity')
for file in ('core.py','interface.py'):
 tree=ast.parse((root/file).read_text())
 imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
 assert not any(x in ('worlds','runner','auditor') for x in imports)
checks.append('learner has no evaluator imports')
(root/'cycles/cycle2/INDEPENDENT_CHECKS.json').write_text(json.dumps({'accepted':True,'checks':checks,'science_jobs':0},indent=2)+'\n')
print(len(checks),'checks passed')
