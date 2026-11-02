"""Fixed lifetime roster. Never stops on a poor efficacy score."""
import util as U
import learner as M
import evaluate as E
import pickle,gc,time,copy,json,os
T=M.torch;np=M.np
RESULTS=[];FORKS=[];LEDGER=[];CURRENT=None

def train(net,opt,data,w,branch,phase,passes):
 global CURRENT
 n=data['n'];ends=np.cumsum(n);y=T.from_numpy(data['y']);records=0;targets=0;begin=time.process_time()
 for ep in range(1,passes+1):
  for i in range(0,len(y),256):
   j=min(i+256,len(y));M.step(net,opt,data['idx'][i:j],y[i:j]);count=int(np.searchsorted(ends,j,side='right')-np.searchsorted(ends,i,side='right'));records+=count;targets+=j-i;counts=U.load('COUNTS.json');counts['records']+=count;counts['targets']+=j-i;counts['optimizer_steps']+=1;U.save('COUNTS.json',counts)
   CURRENT=(net,opt,w,branch,phase,ep,j);U.save('ACTIVE.json',dict(world=w,branch=branch,phase=phase,pass_index=ep,target_site=j,counts=counts,CPU=time.process_time()))
   if counts['records']>7600 or counts['targets']>510000:raise RuntimeError('EXPOSURE_CAP')
   if i%4096==0:U.budget()
  yield ep
 LEDGER.append(dict(world=w,branch=branch,phase=phase,record_presentations=records,target_presentations=targets,CPU=time.process_time()-begin));U.save('TRAINING_LEDGER.json',LEDGER);CURRENT=None

def restore(path,reset=False):return M.restore(T.load(U.H/'local'/path,weights_only=False),reset)
def main():
 assert U.load('QUALIFICATION.json')['status']=='PASS' and U.load('RESOURCE_ADMISSION.json')['status']=='PASS'
 assert all(U.sha(U.ROOT/p)==h for p,h in U.load('SOURCE_LOCK.json')['files'].items())
 tech=U.load('TECHNICAL_COUNTS.json');U.save('COUNTS.json',dict(records=tech['records'],targets=tech['targets'],optimizer_steps=0,technical_records=tech['records'],technical_targets=tech['targets']))
 rec=pickle.loads((U.H/'local/records.pkl').read_bytes());U.save('STATUS.json',dict(status='CYCLE3_FULL_LIFETIME_RUNNING',all_branches_required_regardless_of_scores=True))
 for w in ('W1','W2'):
  data={name:M.cached(w+'_'+name) for name in rec[w]};net=M.new();opt=M.optimizer(net)
  T.manual_seed(int.from_bytes(__import__('hashlib').sha256(f'V179|{w}|dropout'.encode()).digest()[:8],'big')&((1<<63)-1))
  E.checkpoint(net,opt,w,'BIRTH',data,rec[w],RESULTS)
  for ep in train(net,opt,data['A_train'],w,'SHARED','A',3):
   if ep==3:E.checkpoint(net,opt,w,'AFTER_A',data,rec[w],RESULTS)
  del net,opt;gc.collect()
  for arm in ('KEEP','RESET_OPT','B_COLD'):
   name=f'{w}_'+('BIRTH.pt' if arm=='B_COLD' else 'AFTER_A.pt');net,opt=restore(name,arm=='RESET_OPT');forkhash=M.state_hash(net,opt,T.get_rng_state());FORKS.append(dict(world=w,arm=arm,parent=name,weights_hash=M.state_hash(net),state_hash=forkhash,optimizer_entries=len(opt.state),dropout_hash=U.digest(T.get_rng_state().numpy().tobytes())))
   for ep in train(net,opt,data['B_train'],w,arm,'B',3):E.checkpoint(net,opt,w,arm+'_B'+str(ep),data,rec[w],RESULTS,save_state=ep==3)
   del net,opt;gc.collect()
  # The C state is the actual KEEP endpoint, not a chosen best checkpoint.
  for arm in ('EDITED','NOEDIT'):
   net,opt=restore(f'{w}_KEEP_B3.pt');FORKS.append(dict(world=w,arm=arm,parent=f'{w}_KEEP_B3.pt',weights_hash=M.state_hash(net),state_hash=M.state_hash(net,opt,T.get_rng_state()),dropout_hash=U.digest(T.get_rng_state().numpy().tobytes())))
   if U.load('DATA_AUDIT.json')['edits'][w]['admitted']:
    for ep in train(net,opt,data[arm+'_train'],w,arm,'C',2):E.checkpoint(net,opt,w,arm+'_C'+str(ep),data,rec[w],RESULTS,save_state=ep==2)
   del net,opt;gc.collect()
  U.save('FORK_AUDIT.json',FORKS)
 U.save('RESULTS.json',dict(status='EXECUTION_COMPLETE',checkpoints=RESULTS));U.save('STATUS.json',dict(status='EXECUTION_COMPLETE_AWAITING_REDUCTION',counts=U.load('COUNTS.json'),CPU=time.process_time()));U.finish_process('lifetime_all_branches');print('LIFETIME_EXECUTION_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  if CURRENT:
   net,opt,w,b,p,ep,j=CURRENT;T.save(dict(**M.snap(net,opt),world=w,branch=b,phase=p,epoch=ep,target_position=j),U.H/'local/FAILED_PARTIAL.pt')
  U.save('FAILURE.json',dict(error=repr(e),CPU=time.process_time(),counts=U.load('COUNTS.json') if (U.H/'COUNTS.json').exists() else None,partial_saved=CURRENT is not None));raise
