"""Read-only attribution on final corrected checkpoints; no extra training."""
from pathlib import Path
import sys,json,pickle,gzip,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v4_common import *
from data_v4 import dataset,score
from geometry import BITS

def run():
 cfg=spec();rows=[];start=time.process_time()
 for world in cfg['G1_worlds']:
  folder=HERE/f'cycles/cycle3/jobs/{world}_B3';checkpoint=folder/'FINAL_LOCAL_CHECKPOINT.pkl.gz';before=sha(checkpoint)
  with gzip.open(checkpoint,'rb') as f:m=pickle.load(f)
  h=m.digest();train,held=dataset(world,64,16);recent=score(m,train[-16:]);assert m.digest()==h
  r=np.c_[BITS*2-1,np.ones(256)];prob=1/(1+np.exp(-np.clip(r@m.boundary.w,-4,4)))
  q=m.engine.confirm*np.exp(-(m.engine.time-m.engine.confirm_at)/86400)
  channel=[]
  for c in range(9):
   mask=m.engine.last_record[c]>=0;effective=abs(q[c,mask]);allocation=.5/(1+np.exp(-(effective-4)))
   channel.append(dict(output_channel=c,active_confirmed_synapses=int(mask.sum()),effective_q_max=float(effective.max()) if len(effective) else 0,
     fraction_over_threshold=float(np.mean(effective>=4)) if len(effective) else 0,mean_soft_allocation=float(allocation.mean()) if len(effective) else 0))
  receipt=json.loads((folder/'RECEIPT.json').read_text());teacher=[]
  for name,a,b in (('first16',0,16),('last16',48,64),('replay',64,72)):
   tr=receipt['traces'][a:b];teacher.append(dict(phase=name,preupdate_teacher_forced_byte_accuracy=sum(t['preupdate_correct'] for t in tr)/sum(t['target_events'] for t in tr),mean_residual_square=float(np.mean([t['loss'] for t in tr]))))
  assert sha(checkpoint)==before and m.digest()==h
  rows.append(dict(world=world,recent_taught_postrest_replay=recent,boundary_probability_min=float(prob.min()),boundary_probability_max=float(prob.max()),boundary_weight_norm=float(np.linalg.norm(m.boundary.w)),
    warmup_a=m.a,confirmation_channels=channel,teacher_forced_diagnostics=teacher,checkpoint_hash_unchanged=True,model_unchanged=True))
 save(Path('results/DIAGNOSIS.json'),dict(scope='Posthoc read-only recent-taught recall, boundary probabilities and consolidation statistics; not a new candidate or training run',native_training_record_applications=0,read_only_generated_queries=32,CPU_seconds=time.process_time()-start,worlds=rows))
 print(json.dumps([dict(world=r['world'],recent_exact=r['recent_taught_postrest_replay']['exact'],boundary_probability_range=[r['boundary_probability_min'],r['boundary_probability_max']],a=r['warmup_a']) for r in rows]),flush=True)
if __name__=='__main__':run()
