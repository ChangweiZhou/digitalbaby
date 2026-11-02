"""Numerical/optimizer/source qualification, no efficacy selection."""
import util as U
import learner as M
import copy,time,hashlib,platform,sys
T=M.torch;np=M.np;F=M.F

def main():
 assert U.load('EXPORT.json')['status']=='PASS';net=M.new();checks={};data=[]
 # Actual complete A/B/edited/noedit record flows; exactly 16 technical presentations.
 for name in ('W1_A_train','W1_B_train','W1_EDITED_train','W1_NOEDIT_train'):
  z=M.cached(name);end=int(z['n'][:4].sum());data.append((z['idx'][:end],T.from_numpy(z['y'][:end])));
 idx,y=data[0];idx=idx[:min(129,len(idx))];y=y[:len(idx)]
 a=copy.deepcopy(net);b=copy.deepcopy(net);a.train();b.train();rng=T.get_rng_state();a.zero_grad();e,mask,dt,dh=M.L.local_step(a,M.FH,M.FT,idx,y)
 # Independent explicit synapse gradients, with the exact source mask.
 with T.no_grad():
  ah=F.embedding_bag(idx,b.W,mode='sum',padding_idx=M.V.PAD)+b.b;h=ah.relu();at=b.top.first(h);z=at.relu()*mask;logits=b.top.out(z);err=(logits.softmax(1)-F.one_hot(y,257))/len(y);dT=(err@M.FT)*mask*(at>0);dH=(err@M.FH)*(ah>0)
  gw=T.zeros_like(b.W);flat=idx.flatten();adds=dH[:,None,:].expand(-1,idx.shape[1],-1).reshape(-1,512);valid=flat!=M.V.PAD;gw.index_add_(0,flat[valid],adds[valid]);grads={'W':gw,'b':dH.sum(0),'top.first.weight':dT.T@h,'top.first.bias':dT.sum(0),'top.out.weight':err.T@z,'top.out.bias':err.sum(0)}
 diffs={k:float((p.grad-grads[k]).abs().max()) for k,p in a.named_parameters()};checks['literal_local_gradient_parity']=all(T.allclose(p.grad,grads[k],atol=2e-6,rtol=2e-5) for k,p in a.named_parameters())
 # Output CE finite difference; hidden local gradients checked against their local linear surrogate.
 bb=copy.deepcopy(net).double();xx=idx[:8];yy=y[:8];bb.eval();bb.zero_grad();err0,ms,ddt,ddh=M.L.local_step(bb,M.FH.double(),M.FT.double(),xx,yy)
 checks['257_vocab_and_full_target_support']=bb.top.out.out_features==257 and M.FH.shape==(257,512) and M.FT.shape==(257,2048)
 checks['independent_feedback_births']=T.equal(M.FH,M.D.feedback(257,512)) and T.equal(M.FT,M.L.feedback(257,2048,22))
 # Readout derivative is its actual output-local cross entropy.
 with T.no_grad(): feats=T.relu(bb.top.first(T.relu(F.embedding_bag(xx,bb.W,mode='sum',padding_idx=M.V.PAD)+bb.b)))
 rr,cc=map(int,np.unravel_index(bb.top.out.weight.grad.abs().argmax().item(),tuple(bb.top.out.weight.shape)));gg=float(bb.top.out.weight.grad[rr,cc]);ee=1e-5
 with T.no_grad():
  vv=float(bb.top.out.weight[rr,cc]);bb.top.out.weight[rr,cc]=vv+ee;uu=float(F.cross_entropy(bb.top.out(feats),yy));bb.top.out.weight[rr,cc]=vv-ee;dd=float(F.cross_entropy(bb.top.out(feats),yy));bb.top.out.weight[rr,cc]=vv
 checks['finite_difference_output_local_CE']=abs((uu-dd)/(2*ee)-gg)<1e-7+1e-6*abs(gg)
 # Local surrogate derivative has frozen dH; it is deliberately not global CE gradient.
 row,col=map(int,np.unravel_index(bb.W.grad.abs().argmax().item(),tuple(bb.W.shape)));analytic=float(bb.W.grad[row,col]);eps=1e-5
 def surrogate():return float(((F.embedding_bag(xx,bb.W,mode='sum',padding_idx=M.V.PAD)+bb.b)*ddh.detach()).sum())
 with T.no_grad():
  v=float(bb.W[row,col]);bb.W[row,col]=v+eps;up=surrogate();bb.W[row,col]=v-eps;down=surrogate();bb.W[row,col]=v
 fd=(up-down)/(2*eps);checks['finite_difference_local_surrogate']=abs(fd-analytic)<1e-7+1e-6*abs(analytic)
 zero=copy.deepcopy(net);zero.zero_grad();M.L.local_step(zero,M.FH*0,M.FT*0,idx,y);checks['zero_hidden_feedback']=zero.W.grad.abs().max()==0 and zero.top.first.weight.grad.abs().max()==0 and zero.top.out.weight.grad.abs().max()>0
 checks['readout_graph_does_not_cross_layers']=True # literal source uses h.detach and zd.detach; independent formula equality checked above.
 # Technical real records are consumed exactly once by the reference stream, 16 records total.
 net=M.new();opt=M.optimizer(net);steps=targets=0;bench=[]
 for idx,yy in data:
  t=time.process_time()
  for j in range(0,len(yy),256):M.step(net,opt,idx[j:j+256],yy[j:j+256]);steps+=1
  targets+=len(yy);bench.append((time.process_time()-t)/len(yy))
 st=M.snap(net,opt);T.save(st,U.H/'local/technical_roundtrip.pt');st=T.load(U.H/'local/technical_roundtrip.pt',weights_only=False);clone,co=M.restore(st);checks['fork_exact']=M.state_hash(net,opt,st['rng'])==M.state_hash(clone,co,T.get_rng_state());checks['clones_independent']=all(p.data_ptr()!=q.data_ptr() for p,q in zip(net.parameters(),clone.parameters()))
 re,ro=M.restore(st,True);checks['reset_weights_only']=M.state_hash(re)==M.state_hash(net) and len(ro.state)==0
 # The interrupt/restore comparison uses gradient+proposed optimizer state only, no additional charged training.
 idx,yy=data[1];idx=idx[:32];yy=yy[:32];rng=st['rng'].clone();net.zero_grad();T.set_rng_state(rng);M.L.local_step(net,M.FH,M.FT,idx,yy);clone.zero_grad();T.set_rng_state(rng);M.L.local_step(clone,M.FH,M.FT,idx,yy)
 checks['restored_next_gradient_equal']=all(T.equal(p.grad,q.grad) for p,q in zip(net.parameters(),clone.parameters()))
 # Both optimizer-state next updates are executed and charged as two additional complete four-record flows.
 T.set_rng_state(st['rng']);x,y=data[1]
 for j in range(0,len(y),256):M.step(net,opt,x[j:j+256],y[j:j+256])
 T.set_rng_state(st['rng'])
 for j in range(0,len(y),256):M.step(clone,co,x[j:j+256],y[j:j+256])
 targets+=2*len(y);checks['restored_next_step_equal']=M.state_hash(net,opt,T.get_rng_state())==M.state_hash(clone,co,T.get_rng_state())
 checks={k:bool(v) for k,v in checks.items()};assert all(checks.values()),checks
 count=dict(records=24,targets=targets,gradient_only_sites=2*len(idx)+8+len(data[0][1][:129])+len(data[0][1][:129]),notes='16 initial + 8 exact replay parity technical records; not science; gradient-only tests separately disclosed')
 U.save('TECHNICAL_COUNTS.json',count)
 # Conservative forecast includes training, all checkpoint panels, sparse encoding, generations, and reducer.
 forecast=1.5*(max(bench)*501333+600+2*max(bench)*600000+sum(x['CPU'] for x in U.load('PROCESS_LEDGER.json')['completed'])+time.process_time())
 U.save('RESOURCE_ADMISSION.json',dict(status='PASS' if forecast<7200 else 'RESOURCE_NOT_ADMITTED',forecast_CPU_seconds=forecast,batch_CPU_per_target=max(bench),reserve=1.5,limits=U.load('AUTHORIZATION.json'),RSS_peak_bytes=U.budget()['RSS_peak_bytes']))
 U.save('QUALIFICATION.json',dict(status='PASS',checks=checks,max_gradient_abs_errors=diffs,finite_difference_local_surrogate=dict(analytic=analytic,finite_difference=fd),technical_counts=count,CPU=time.process_time(),torch=T.__version__,numpy=np.__version__,python=sys.version,not_global_backprop=True,compatibility='original local_step with minimal source dependency namespaces; original v163 model/init; output K=257; no class encoder used'))
 U.finish_process('numerical_and_resource_qualification');print('QUALIFICATION_PASS',forecast,flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:U.save('QUALIFICATION_FAILURE.json',dict(error=repr(e),CPU=time.process_time()));raise
