"""Paired B lesion from actual A state; ordinary local learning resumes for C."""
import util as U
import learner as M
import plasticity as P
import instrument as I
import evaluate as E
import pickle,time,gc,json
T=M.torch;np=M.np
RESULTS=[];LEDGER=[];FORKS=[];FREEZE=[];CURRENT=None

def restore(name):return M.restore(T.load(U.H/'local'/name,weights_only=False))
def train(net,opt,data,w,arm,phase,passes,freeze=False,reference_hidden=None):
 global CURRENT
 n=data['n'];ends=np.cumsum(n);y=T.from_numpy(data['y']);records=targets=0;t=time.process_time()
 for ep in range(1,passes+1):
  for i in range(0,len(y),256):
   j=min(i+256,len(y));I.measured_step(net,opt,data['idx'][i:j],y[i:j],dict(world=w,branch=arm,phase=phase,pass_index=ep,target_start=i,target_end=j),freeze=freeze)
   count=int(np.searchsorted(ends,j,side='right')-np.searchsorted(ends,i,side='right'));records+=count;targets+=j-i;c=U.load('COUNTS.json');c['records']+=count;c['targets']+=j-i;c['optimizer_steps']+=1;U.save('COUNTS.json',c);CURRENT=(net,opt,w,arm,phase,ep,j);U.save('ACTIVE.json',dict(world=w,arm=arm,phase=phase,pass_index=ep,target_site=j,counts=c,CPU=time.process_time()))
   if c['records']>3100 or c['targets']>200000:raise RuntimeError('EXPOSURE_CAP')
   if i%4096==0:U.budget()
  hidden=P.hidden_digest(net,opt)
  if freeze:
   assert hidden==reference_hidden,'FROZEN_HIDDEN_STATE_CHANGED';FREEZE.append(dict(world=w,arm=arm,pass_index=ep,hidden_hash=hidden,identical_to_A=True));U.save('FREEZE_AUDIT.json',FREEZE)
  yield ep
 LEDGER.append(dict(world=w,branch=arm,phase=phase,passes=passes,freeze_hidden=freeze,record_presentations=records,target_presentations=targets,CPU_including_panels=time.process_time()-t));U.save('TRAINING_LEDGER.json',LEDGER);CURRENT=None

def main():
 assert U.load('QUALIFICATION.json')['status']=='PASS' and U.load('RESOURCE_ADMISSION.json')['status']=='PASS';assert all(U.sha(U.ROOT/p)==h for p,h in U.load('SOURCE_LOCK.json')['files'].items());tech=U.load('TECHNICAL_COUNTS.json');U.save('COUNTS.json',dict(records=tech['records'],targets=tech['targets'],optimizer_steps=0,technical_records=tech['records'],technical_targets=tech['targets']));rec=pickle.loads((U.H/'local/records.pkl').read_bytes());U.save('STATUS.json',dict(status='PAIRED_PLASTICITY_DIAGNOSTIC_RUNNING',official_run=False))
 for w in ('W1','W2'):
  data={name:M.cached(w+'_'+name) for name in rec[w]};net,opt=restore(w+'_A_START.pt');Ahidden=P.hidden_digest(net,opt);E.checkpoint(net,opt,w,'A_START',data,rec[w],RESULTS);del net,opt;gc.collect()
  for arm in ('NORMAL','CLAMP'):
   net,opt=restore(w+'_A_START.pt');FORKS.append(dict(world=w,branch=arm,phase='B',state_hash=M.state_hash(net,opt,T.get_rng_state()),weights_hash=M.state_hash(net),hidden_hash=P.hidden_digest(net,opt),dropout_hash=U.digest(T.get_rng_state().numpy().tobytes())))
   for ep in train(net,opt,data['B_train'],w,arm,'B',3,arm=='CLAMP',Ahidden):E.checkpoint(net,opt,w,arm+'_B'+str(ep),data,rec[w],RESULTS)
   del net,opt;gc.collect()
   for feedback in ('EDITED','NOEDIT'):
    net,opt=restore(w+'_'+arm+'_B3.pt');FORKS.append(dict(world=w,branch=arm,phase=feedback,state_hash=M.state_hash(net,opt,T.get_rng_state()),weights_hash=M.state_hash(net),hidden_hash=P.hidden_digest(net,opt),dropout_hash=U.digest(T.get_rng_state().numpy().tobytes())))
    for ep in train(net,opt,data[feedback+'_train'],w,arm+'_'+feedback,'C',2,False):E.checkpoint(net,opt,w,arm+'_'+feedback+'_C'+str(ep),data,rec[w],RESULTS)
    del net,opt;gc.collect()
  U.save('FORK_AUDIT.json',FORKS)
 U.save('RESULTS.json',dict(status='EXECUTION_COMPLETE',checkpoints=RESULTS));U.save('STATUS.json',dict(status='EXECUTION_COMPLETE_AWAITING_REDUCTION',counts=U.load('COUNTS.json'),CPU=time.process_time()));U.finish_process('paired_B_lesion_C_resumption');print('PAIRED_LIFETIME_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  if CURRENT:
   net,opt,w,arm,phase,ep,j=CURRENT;T.save(dict(**M.snap(net,opt),world=w,arm=arm,phase=phase,epoch=ep,target_position=j),U.H/'local/FAILED_PARTIAL.pt')
  U.save('FAILURE.json',dict(error=repr(e),CPU=time.process_time(),counts=U.load('COUNTS.json') if (U.H/'COUNTS.json').exists() else None,partial_saved=CURRENT is not None));raise
