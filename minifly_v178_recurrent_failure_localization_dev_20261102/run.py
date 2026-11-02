"""Fixed A→B→C ladder, first-failure extension or fresh endpoint guard."""
import copy,json,time,os
import common as C
from model import Model
from diagnostics import diagnose
torch=C.torch;np=C.np;L=C.L
CACHE={};ACTIVE=None
def batch(split,lines):
 key=(id(split),tuple(lines))
 if key not in CACHE:
  idx,mask,pos=L.V.line_batches(split,lines);ctr,valid=L.targets(split,L.Synth(),pos);CACHE[key]=(idx,ctr,valid)
 return CACHE[key]
def evaluate(net,split):
 sums=[];counts=[]
 with torch.no_grad():
  for i in range(0,len(split['n']),64):
   ids=list(range(i,min(i+64,len(split['n']))));idx,ctr,valid=batch(split,ids);p=torch.softmax(net.out(net.run(idx)).double(),-1)
   for j in range(len(ids)):
    yy=ctr[j][valid[j]].numpy();bits=-np.log2(p[j][valid[j]].numpy()[np.arange(len(yy)),yy]);sums.append(float(bits.sum()));counts.append(len(yy))
 return dict(bits=sum(sums)/sum(counts),line_sums=sums,counts=counts)
def fitted(stage,sp,key,resume=None,end=10,open_test=True):
 global ACTIVE
 start=C.cpu();net=Model(stage,key.startswith('GUARD'));opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);g=torch.Generator().manual_seed(13);curve=[];begin=0;best=float('inf');best_ep=0;bad=0;would_stop=None
 if resume:
  state=torch.load(resume,weights_only=False);net.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);g.set_state(state['order_generator']);begin=state['epoch'];curve=state['curve'];best=min(r['calibration']['bits'] for r in curve);best_ep=1+int(np.argmin([r['calibration']['bits'] for r in curve]))
 ACTIVE=dict(net=net,opt=opt,g=g,key=key,curve=curve,epoch=begin)
 for ep in range(begin,end):
  net.train();order=torch.randperm(100,generator=g).tolist();online_sum=0.;targets=0
  for i in range(0,100,16):
   ids=order[i:i+16];idx,ctr,valid=batch(sp['train'],ids);opt.zero_grad();logits=net.out(net.run(idx));loss=torch.nn.functional.cross_entropy(logits[valid],ctr[valid]);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();net.enforce(opt);C.charge(len(ids));online_sum+=float(loss.detach())*int(valid.sum());targets+=int(valid.sum())
  net.eval();tr=evaluate(net,sp['train']);ca=evaluate(net,sp['calib']);curve.append(dict(epoch=ep+1,online_train_bits=online_sum/targets/np.log(2),train=tr,calibration=ca,preclip_norm=float(norm)))
  if ca['bits']<best:
   best=ca['bits'];best_ep=ep+1;bad=0;torch.save(net.state_dict(),C.H/f'local/{key}_best_calibration.pt')
  else:bad+=1
  if bad>=3 and would_stop is None:would_stop=ep+1
  ACTIVE.update(epoch=ep+1,curve=curve);C.save('ACTIVE.json',dict(stage=key,epoch=ep+1,training_bits=tr['bits'],calibration_bits=ca['bits'],CPU=C.cpu(),counts=json.loads((C.H/'COUNTS.json').read_text())))
 path=C.H/f'local/{key}_epoch{end}.pt';torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),order_generator=g.get_state(),epoch=end,curve=curve),path)
 test=evaluate(net,sp['test']) if open_test else None
 out=dict(stage=stage,key=key,epochs=end,record_applications=(end-begin)*100,scored_targets=(end-begin)*11200,curve=curve,final_train=curve[-1]['train'],final_calibration=curve[-1]['calibration'],test=test,best_calibration_epoch=best_ep,best_calibration_bits=best,original_patience_would_stop=would_stop,primary_is_final_checkpoint=True,checkpoint=str(path.relative_to(C.H)),checkpoint_sha256=C.sha(path),training_and_evaluation_CPU=C.cpu()-start)
 C.save(key+'_RESULT.json',out);ACTIVE=None;return net,out,path
def main():
 assert json.loads((C.H/'RESOURCE_ADMISSION.json').read_text())['status']=='PASS';locks=json.loads((C.H/'SOURCE_LOCK.json').read_text())['files'];assert all(C.sha(C.ROOT/p)==d for p,d in locks.items())
 sp=C.splits();receipts=[];verdict=None
 for stage in ('A','B','C'):
  # At a calibration failure, test remains unopened until final diagnosis.
  net,r,path=fitted(stage,sp,stage,open_test=False);receipts.append(r)
  if stage=='A':
   r['test']=evaluate(net,sp['test']);r['functional_reproduction']=r['final_calibration']['bits']<=1.35 and r['test']['bits']<=1.35;r['strict_numeric_reproduction']=r['functional_reproduction'] and abs(r['test']['bits']-1.1516320951135754)<=.05
   C.save('A_RESULT.json',r);receipts[-1]=r;diagnose(net,sp,'A')
   if not r['functional_reproduction']:verdict='REFERENCE_NOT_REPRODUCED';break
  else:
   if r['final_calibration']['bits']>1.5:
    verdict='SPARSE_PACKAGE_FAIL_10' if stage=='B' else 'GATED_PACKAGE_FAIL_10';C.save('FIRST_FAILURE.json',dict(stage=stage,matched_dose_verdict=verdict,calibration_bits=r['final_calibration']['bits'],extension_predeclared=True,downstream_forbidden=True));C.budget()
    extended,er,ep=fitted(stage,sp,stage+'_EXTENSION',resume=path,end=20,open_test=True);er['budget_diagnostic']='BUDGET_SENSITIVE' if er['final_calibration']['bits']<=1.5 else 'UNRESOLVED_OR_ARCHITECTURE_LIMITED_AT_20';C.save(stage+'_EXTENSION_RESULT.json',er);receipts.append(er);diagnose(extended,sp,stage+'_EXTENSION');break
   r['test']=evaluate(net,sp['test']);C.save(stage+'_RESULT.json',r);receipts[-1]=r;diagnose(net,sp,stage)
   if r['test']['bits']>1.65:verdict='NO_REPLICATED_REFERENCE_FEASIBILITY';C.save('GENERALIZATION_WARNING.json',dict(stage=stage,held_bits=r['test']['bits'],downstream_forbidden=True));break
 if verdict is None:
  guard=C.splits(True);rr=[]
  for stage in ('A','C'):
   net,r,path=fitted(stage,guard,'GUARD_'+stage);diagnose(net,guard,'GUARD_'+stage);rr.append(r);receipts.append(r)
  ok=all(r['final_calibration']['bits']<=1.5 and r['test']['bits']<=1.6 for r in rr) and rr[1]['test']['bits']-rr[0]['test']['bits']<=.35;verdict='REFERENCE_BRIDGE_FEASIBLE_DEV' if ok else 'NO_REPLICATED_REFERENCE_FEASIBILITY'
 C.save('RESULTS.json',dict(verdict=verdict,stages=receipts,core_adopted=False));C.save('STATUS.json',dict(status='EXECUTION_COMPLETE_AWAITING_REDUCTION',verdict=verdict,CPU=C.cpu(),peak_RSS_bytes=C.resource.getrusage(C.resource.RUSAGE_SELF).ru_maxrss))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  path=C.H/'local/FAILED_partial.pt'
  if ACTIVE is not None:torch.save(dict(model=ACTIVE['net'].state_dict(),optimizer=ACTIVE['opt'].state_dict(),order_generator=ACTIVE['g'].get_state(),epoch=ACTIVE['epoch'],curve=ACTIVE['curve'],key=ACTIVE['key']),path)
  C.save('FAILURE.json',dict(error=repr(e),partial_saved=ACTIVE is not None,path=str(path.relative_to(C.H)),counts=json.loads((C.H/'COUNTS.json').read_text())));raise
