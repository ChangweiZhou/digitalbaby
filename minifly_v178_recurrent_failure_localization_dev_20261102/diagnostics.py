"""Registered frozen-state diagnostics and train-only linear probe."""
import math,time
import common as C
from scipy.optimize import minimize
torch=C.torch;np=C.np
def diagnose(net,sp,key):
 start=C.cpu();states={};ys={};net.eval()
 with torch.no_grad():
  for part,d in sp.items():
   idx,mask,pos=C.L.V.line_batches(d,list(range(len(d['n']))));target,valid=C.L.targets(d,C.L.Synth(),pos);states[part]=net.run(idx)[valid].numpy().astype(np.float64);ys[part]=target[valid].numpy()
 mu=states['train'].mean(0);sd=states['train'].std(0);sd[sd<1e-8]=1.;xx={k:(v-mu)/sd for k,v in states.items()};X=np.column_stack((xx['train'],np.ones(len(xx['train']))));y=ys['train'];dim=X.shape[1]
 def loss(theta):
  w=theta.reshape(dim,8);z=X@w;z-=z.max(1,keepdims=True);lp=z-np.log(np.exp(z).sum(1,keepdims=True));err=np.exp(lp);err[np.arange(len(y)),y]-=1;g=X.T@err/len(y);g[:-1]+=.01*w[:-1];return float(-lp[np.arange(len(y)),y].mean()+.005*np.sum(w[:-1]**2)),g.ravel()
 fit=minimize(loss,np.zeros(dim*8),jac=True,method='L-BFGS-B',options=dict(maxiter=200,ftol=1e-10,gtol=1e-7));w=fit.x.reshape(dim,8);scores={}
 for part in xx:
  z=np.column_stack((xx[part],np.ones(len(xx[part]))))@w;z-=z.max(1,keepdims=True);lp=z-np.log(np.exp(z).sum(1,keepdims=True));scores[part]=dict(bits=float(-lp[np.arange(len(z)),ys[part]].mean()/np.log(2)),accuracy=float((lp.argmax(1)==ys[part]).mean()))
 # Current trained dynamics, first100 training lines, no labels used.
 d=sp['train'];idx,_,_=C.L.V.line_batches(d,list(range(100)));eye=None
 with torch.no_grad():
  inp=torch.nn.functional.embedding_bag(idx.reshape(12000,1),net.Wrk,mode='sum',padding_idx=C.L.V.PAD).view(100,120,256);h=torch.zeros(100,256);v=torch.randn(100,256,generator=torch.Generator().manual_seed(0));v/=v.norm(dim=1,keepdim=True);tot=0.;count=0;sat=[];gate_means=[];jac=[];D=[]
  if net.stage=='C':qin=torch.nn.functional.embedding_bag(idx.reshape(12000,1),net.Vin,mode='sum',padding_idx=C.L.V.PAD).view(100,120,256)
  W=net.Wrr if net.stage=='A' else net.Wrr*net.mask
  for t in range(120):
   old=h;a=torch.tanh(inp[:,t]+h@W.T+net.b)
   if net.stage=='C':
    g=torch.sigmoid(qin[:,t]+old@(net.Srr*net.mask).T+net.c);dv=g*(1-a*a);dq=(a-old)*g*(1-g);h=(1-g)*old+g*a;vn=(1-g)*v+dv*(v@W.T)+dq*(v@(net.Srr*net.mask).T);j=torch.diag(1-g[0])+dv[0,:,None]*W+dq[0,:,None]*(net.Srr*net.mask);dd=dv[0,:,None]*net.Wrk[:8].T+dq[0,:,None]*net.Vin[:8].T;gate_means.append(float(g.mean()))
   else:h=a;dv=1-h*h;vn=dv*(v@W.T);j=dv[0,:,None]*W;dd=dv[0,:,None]*net.Wrk[:8].T
   n=vn.norm(dim=1);v=vn/n[:,None]
   if t>0:tot+=float(torch.log(n).sum());count+=100
   sat.append(float((a.abs()>.99).float().mean()))
   if t<=16:jac.append(j.double());D.append(dd.double())
  prod=torch.eye(256,dtype=torch.float64);sens=D[8]
  for t in range(9,17):prod=jac[t]@prod;sens=jac[t]@sens
  singular=torch.linalg.svdvals(prod);input_singular=torch.linalg.svdvals(sens)
 # Lesions have distribution-shift caveats; no fitting or class repair.
 original={k:v.clone() for k,v in net.state_dict().items()};lesions={}
 for name in (['recurrence_off'] if net.stage=='A' else ['recurrence_off','cross_edges_off']):
  with torch.no_grad():
   if name=='recurrence_off':net.Wrr.zero_()
   else:
    local=torch.arange(256)//8;net.Wrr[local[:,None]!=local[None,:]]=0
   if net.stage=='C':
    if name=='recurrence_off':net.Srr.zero_();net.c.fill_(80.)
    else:net.Srr[local[:,None]!=local[None,:]]=0
  sums,counts=C.L.evaluate(net,sp['calib'],C.L.Synth());lesions[name]=float(sums.sum()/counts.sum());net.load_state_dict(original)
 info=dict(probe=dict(regularization=.01,normalization='training mean/std; constant columns scale1',objective='mean cross entropy + .01/2 weight squared, unregularized bias',optimizer='L-BFGS-B max200',converged=bool(fit.success),iterations=int(fit.nit),objective_evaluations=int(fit.nfev),scores=scores,unique_fit_positions=len(y),not_model_output=True),dynamics=dict(train_only_Lyapunov=tot/count,eight_step_product_singular_max=float(singular.max()),eight_step_product_singular_median=float(singular.median()),input_sensitivity_singular_values=input_singular.tolist(),input_sensitivity_scope='Eight actually used input-coordinate directions at t8, propagated through J9..J16; first training line',mean_tanh_saturation=float(np.mean(sat)),mean_gate=float(np.mean(gate_means)) if gate_means else None,state_variance=float(states['train'].var())),calibration_lesions=lesions,lesions_shift_state_distribution=True,extra_probe_record_exposures=100*int(fit.nfev),extra_probe_target_applications=11200*int(fit.nfev),extra_state_readonly_records=170,CPU=C.cpu()-start)
 C.save(key+'_DIAGNOSTICS.json',info);return info
