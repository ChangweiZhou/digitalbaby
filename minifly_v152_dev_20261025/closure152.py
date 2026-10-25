"""Saved-response mutation checks and saved-state causal audit; zero model inference."""
import os,sys,time,pickle,json,copy,hashlib,resource
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H))
import numpy as np
import audit152
from support152 import rng

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,o):(H/n).write_text(json.dumps(o,indent=2)+'\n')
rows=[];state_files=0
for d in sorted((H/'groups').iterdir()):
 a=load(d/'ACQUISITION.json');states=a['states'];models={}
 for key,entry in states.items():
  p=H/entry['path'];assert sha(p)==entry['sha256']
  with p.open('rb') as f:models[key]=pickle.load(f)
  state_files+=1
 old,sh,no=models['OLD_W'],models['OLD_SH'],models['N_OLD'];params=old.PARAMS
 assert all(np.array_equal(getattr(old,k),getattr(no,k)) for k in params)
 gen=rng(a['world'],'weights');u=gen.normal(0,1/16,(64,256));r=gen.normal(0,1/np.sqrt(128),(16,128))
 assert np.array_equal(sh.U,u) and np.array_equal(sh.R,r) and not np.any(sh.b) and not np.any(sh.q) and not np.any(sh.theta)
 data=load(d/'DATA.json');old_n=8*sum(len(bytes.fromhex(x['answer']))+1 for x in data['OLD']);new_n=8*sum(len(bytes.fromhex(x['answer']))+1 for x in data['NEW'])
 assert old.n==sh.n==old_n and all(models[k].n==old_n+new_n for k in ['W','N_OLD','SH_OLD'])
 train=a['training_records']
 for phase,branch in [('OLD','W'),('OLD','SH_OLD'),('NEW','W'),('NEW','N_OLD'),('NEW','SH_OLD')]:
  for p in range(1,9):
   actual=[x['id'] for x in train if x['phase']==phase and x['branch']==branch and x['pass_number']==p]
   expected=[data[phase][i]['id'] for i in data['orders'][phase][p-1]];assert actual==expected
 for cell in a['curves']+a['immediate']:
  assert all(0<=z['argmax']<=256 for row in cell['rows'] for z in row['teacher_forced'])
 rows.append(dict(group=d.name,checkpoint_hashes=True,OLD_inhibited_parameters_equal_newborn=True,NEW_inhibited_parameters_equal_learned_OLD=True,all_branch_target_counters_match=True,all_records_match_frozen_pass_order=True))
base=load(next((H/'groups').glob('*/ACQUISITION.json')))['curves'][0];audit152.check_eval(base)
cases={}
for mutation in ['wrong_correct_flag','wrong_EOS_flag','illegal_class','nonfinite_loss','false_CAP']:
 x=copy.deepcopy(base)
 if mutation=='wrong_correct_flag':x['rows'][0]['correct']=not x['rows'][0]['correct']
 elif mutation=='wrong_EOS_flag':x['rows'][0]['EOS']=not x['rows'][0]['EOS']
 elif mutation=='illegal_class':x['rows'][0]['choices'][0]=257
 elif mutation=='nonfinite_loss':x['rows'][0]['teacher_forced'][0]['lp']=float('nan')
 else:x['rows'][0]['choices']=[1];x['rows'][0]['EOS']=False
 try:audit152.check_eval(x)
 except (AssertionError,ValueError):cases[mutation]='REJECTED'
 else:raise AssertionError('Mutation was not rejected: '+mutation)
save('STATE_AUDIT.json',dict(status='ACCEPTED',state_files=state_files,groups=rows,no_new_forward_or_training_passes=True))
save('cycles/CYCLE3_MUTATION_TESTS.json',dict(status='PASS',synthetic_saved_response_fixtures=cases,new_training_records=0,new_model_queries=0))
print(json.dumps({'saved_states_audited':state_files,'negative_fixtures_rejected':len(cases),'CPU':time.process_time(),'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))
