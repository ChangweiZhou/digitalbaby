"""Exact same-graph BPTT instrument. Never imported by BS/BR learners."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json
import numpy as np
import torch
import core as C
torch.set_num_threads(1)
ACTIVE=None
def gradients(p,o,e,x,y):
 pp=torch.tensor(p,requires_grad=True);oo=torch.tensor(o,requires_grad=True);ee=torch.from_numpy(e);src=torch.from_numpy(C.SRC);h=torch.zeros((len(x),128),dtype=torch.float64);loss=0.
 for t in range(x.shape[1]):
  u=ee[x[:,t]];nb=h[:,src];v=u@pp[:,:16].T+(nb*pp[None,:,16:32]).sum(2)+pp[:,32];q=u@pp[:,33:49].T+(nb*pp[None,:,49:65]).sum(2)+pp[:,65];a=torch.tanh(v);g=torch.sigmoid(q);h=(1-g)*h+g*a;z=h@oo[:,:128].T+oo[:,128];loss=loss+torch.nn.functional.cross_entropy(z,torch.from_numpy(y[:,t]),reduction='sum')
 loss.backward();return pp.grad.numpy(),oo.grad.numpy(),float(loss.detach())
def qualify():
 w=202611027701;p,o,e,b=C.init(w,0);x,y,_=C.data(w,'train',8)
 pd=p.copy();pd[:,24:32]=0;pd[:,57:65]=0
 gd=gradients(pd,o,e,x[:1,:3],y[:1,:3]);gc=gradients(p,o,e,x[:1,:3],y[:1,:3]);np.savez(C.H/'local/gradient_reference.npz',p=p,o=o,e=e,b=b,x=x,y=y,pd=pd,gd=gd[0],gdo=gd[1],gc=gc[0],gco=gc[1])
 # Independent autograd Jacobian vs analytic local recurrence, plus finite difference.
 old=torch.tensor(C.rng(w,'technical','jacobian').normal(0,.1,128),requires_grad=True);pt=torch.tensor(p,requires_grad=True);u=torch.from_numpy(e[x[0,0]]);src=torch.from_numpy(C.SRC)
 def transition(h,par):
  v=par[:,:16]@u+(par[:,16:32]*h[src]).sum(1)+par[:,32];q=par[:,33:49]@u+(par[:,49:65]*h[src]).sum(1)+par[:,65];a=torch.tanh(v);g=torch.sigmoid(q);return (1-g)*h+g*a
 J=torch.autograd.functional.jacobian(lambda h:transition(h,pt),old).detach().numpy();h=old.detach().numpy();_,_,a,g=C.step(p,o,e,h[None,:],np.array([x[0,0]]));dv=g[0]*(1-a[0]**2);dq=(a[0]-h)*g[0]*(1-g[0]);analytic=np.diag(1-g[0])
 for j in range(128):analytic[j,C.SRC[j]]+=dv[j]*p[j,16:32]+dq[j]*p[j,49:65]
 assert np.allclose(J,analytic,rtol=1e-7,atol=1e-9)
 eps=1e-6;plus=old.detach().clone();minus=plus.clone();plus[5]+=eps;minus[5]-=eps;fd=((transition(plus,pt)-transition(minus,pt))/(2*eps)).detach().numpy();assert np.allclose(fd,J[:,5],rtol=1e-7,atol=1e-9)
 start=time.process_time();gg=gradients(p,o,e,x,y);bp_batch=time.process_time()-start
 opt=C.Adam(p,o);before=p.copy();d=gg[0]/y.size;d/=np.maximum(1,np.linalg.norm(d,axis=1,keepdims=True));metrics=opt.update(gg[:2],0,y.size,.001);expected=before*(1-.001*metrics['warmup']*1e-4)-.001*metrics['warmup']*d/(np.abs(d)+1e-8);assert np.allclose(p,expected,atol=1e-12,rtol=1e-12)
 np.savez(C.H/'local/optimizer_roundtrip.npz',p=p,o=o,m0=opt.m[0],m1=opt.m[1],v0=opt.v[0],v1=opt.v[1]);r=np.load(C.H/'local/optimizer_roundtrip.npz');assert np.array_equal(p,r['p'])
 C.save('BP_QUALIFICATION.json',dict(status='PASS',jacobian_autograd=True,finite_difference=True,applied_Adam_warmup=True,checkpoint=True,bp_batch_CPU=bp_batch,technical_training_records=8,technical_targets=int(y.size),CPU=time.process_time()))
def train(w,lr,bias,epochs,role,key):
 global ACTIVE
 p,o,e,b=C.init(w,bias);x,y,_=C.data(w,'train',32 if role=='pilot' else 64);vx,vy,_=C.data(w,'val',16);opt=C.Adam(p,o);n=0;curve=[];start=time.process_time()
 for ep in range(epochs):
  order=C.rng(w,'order',f'epoch-{ep}').permutation(len(x));total=0.
  for k in range(0,len(x),8):
   ix=order[k:k+8];gp,go,loss=gradients(p,o,e,x[ix],y[ix]);m=opt.update((gp,go),n,y[ix].size,lr);n+=y[ix].size;total+=loss;ACTIVE=dict(p=p,o=o,e=e,b=b,n=n,m0=opt.m[0],m1=opt.m[1],v0=opt.v[0],v1=opt.v[1]);C.charge(len(ix),int(y[ix].size))
  val=C.evaluate(p,o,e,vx,vy);curve.append(dict(epoch=ep+1,train_bits=total/y.size/np.log(2),val=val,optimizer=m));C.check();C.save('ACTIVE.json',dict(stage=role,key=key,epoch=ep+1,targets=n,CPU=time.process_time(),validation_bits=val['lag_bits']))
 path=C.H/f'local/{key}.npz';np.savez(path,p=p,o=o,e=e,b=b,m0=opt.m[0],m1=opt.m[1],v0=opt.v[0],v1=opt.v[1],n=n,steps=opt.t)
 out=dict(key=key,world=w,arm='BP',records=len(x)*epochs,targets=n,lr=lr,bias=bias,curve=curve,final_val=curve[-1]['val'],training_CPU=time.process_time()-start,checkpoint=str(path.relative_to(C.H)))
 if role=='main':
  hx,hy,_=C.data(w,'held',64);t=time.process_time();out['held']=C.evaluate(p,o,e,hx,hy,True);out['inference_CPU']=time.process_time()-t
 C.save(key+'.json',out);return out
def pcheck():
 p,o,e,b=C.init(202611027701,0);par=torch.tensor(p,requires_grad=True);old=torch.tensor(C.rng(202611027701,'technical','P').normal(0,.1,128));u=torch.from_numpy(e[3]);src=torch.from_numpy(C.SRC)
 def f(par):
  a=torch.tanh(par[:,:16]@u+(par[:,16:32]*old[src]).sum(1)+par[:,32]);g=torch.sigmoid(par[:,33:49]@u+(par[:,49:65]*old[src]).sum(1)+par[:,65]);return (1-g)*old+g*a
 jac=torch.autograd.functional.jacobian(lambda pp:f(pp)[:8],par).numpy();h=old.numpy();_,_,a,g=C.step(p,o,e,h[None,:],np.array([3]));expected=np.zeros_like(jac)
 for j in range(8):
  phi=np.concatenate((e[3],h[C.SRC[j]],np.ones(1)));expected[j,j,:33]=g[0,j]*(1-a[0,j]**2)*phi;expected[j,j,33:]=(a[0,j]-h[j])*g[0,j]*(1-g[0,j])*phi
 assert np.allclose(jac,expected,rtol=1e-7,atol=1e-9)
 pp=par.detach().clone();pm=pp.clone();pp[3,49]+=1e-6;pm[3,49]-=1e-6;assert np.allclose(((f(pp)-f(pm))/(2e-6)).numpy()[:8],jac[:,3,49],rtol=1e-7,atol=1e-9)
 C.save('P_QUALIFICATION.json',dict(status='PASS',autograd=True,finite_difference=True,training_records=0))
def pilot():
 results=[train(202611027701,lr,bias,5,'pilot',f'pilot_{lr}_{bias}') for lr in (.001,.003) for bias in (0,-1.5)]
 selected=min(results,key=lambda v:(v['final_val']['lag_bits'],v['lr'],abs(v['bias'])));C.save('PILOT_LOCK.json',dict(lr=selected['lr'],bias=selected['bias'],selected=selected['key'],results=results,main_scores_used=False))
def main():
 lock=json.loads((C.H/'PILOT_LOCK.json').read_text());rr=[train(w,lock['lr'],lock['bias'],10,'main',f'BP_{w}') for w in (202611027711,202611027712)];C.save('BP_RESULTS.json',dict(results=rr))
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:
  if ACTIVE is not None:np.savez(C.H/'local/FAILED_BP_partial.npz',**ACTIVE)
  C.save('FAILURE.json',dict(error=repr(e),stage=sys.argv[1],partial_saved=ACTIVE is not None));raise
