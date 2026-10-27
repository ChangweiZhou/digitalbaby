"""Independent forward-sensitivity and finite-difference instruments, never learners."""
from collections import deque
import numpy as np
import support171 as s
from controller171 import Controller,PARAMS,warm,center
def parameter_indices(c):
 start=0;idx={}
 for k in PARAMS:
  a=getattr(c,k);idx[k]=np.arange(start,start+a.size).reshape(a.shape);start+=a.size
 return idx,start
def forward_influence(c,byte,v,terms,keep):
 """Augmented (h,r) Jacobian; independent dense numerical reference only."""
 hp=c.h.copy();rp=c.r.copy();c.step(byte,v);f=c.latch;H=c.H;W,G=c.dense();d=.25*(1-f['z']**2);gamma=.5*(1-np.tanh(f['a'])**2)*f['u'];L=gamma[:,None]*G
 A=.75*np.eye(H)+d[:,None]*W;B=d[:,None]*c.R;J=np.block([[A,B],[L@A,L@B]])
 ids,P=parameter_indices(c);D=np.zeros((H+16,P))
 for j in range(H):
  D[j,ids['U'][j,byte]]=d[j];D[j,ids['W'][j]]=d[j]*hp[c.ids[j]];D[j,ids['R'][j]]=d[j]*rp;D[j,ids['b'][j]]=d[j]
 D[H:]=L@D[:H]
 for j in range(16):D[H+j,ids['G'][j]]+=gamma[j]*c.h[c.gids[j]];D[H+j,ids['c'][j]]+=gamma[j]
 terms=[J@E for E in terms]+[D];return terms[-keep:]
def forward_gradient(c,E,y,n):
 p,_=c.read();e=p.copy();e[y]-=1;dv=center(c.C.T@e/4);f=c.latch;da=.5*(1-np.tanh(f['a'])**2)*f['v']*dv;W,G=c.dense();out=E[:c.H].T@(G.T@da);idx,_=parameter_indices(c)
 for j in range(16):out[idx['G'][j]]+=da[j]*c.h[c.gids[j]];out[idx['c'][j]]+=da[j]
 return warm(n)*out
def rel(a,b):return float(np.linalg.norm(a-b)/max(1e-14,np.linalg.norm(b)))
def check(C):
 world=202610269790;xs=[0,255,61,10,13];vs=s.rng(world,'numeric_native_values').normal(0,2,(5,16));y=183;n=17
 f3=Controller(world,'EXACT_RECORD',C);f3.frames=deque();terms=[]
 full=f3.clone();allterms=[];prefix=None
 for t,(x,v) in enumerate(zip(xs,vs)):
  if t==2:prefix=(f3.h.copy(),f3.r.copy())
  terms=forward_influence(f3,x,v,terms,3);allterms=forward_influence(full,x,v,allterms,100)
 gf3=forward_gradient(f3,sum(terms),y,n);gfall=forward_gradient(full,sum(allterms),y,n)
 f3.credit(y,n,mode='EXACT3');full.credit(y,n,mode='EXACT_RECORD');gr3=f3.gradvector();grall=full.gradvector()
 checks=dict(forward_three_vs_reverse=dict(relative=rel(gf3,gr3),absolute=float(np.max(abs(gf3-gr3)))),forward_full_vs_reverse=dict(relative=rel(gfall,grall),absolute=float(np.max(abs(gfall-grall)))))
 assert checks['forward_three_vs_reverse']['relative']<1e-6 and checks['forward_three_vs_reverse']['absolute']<1e-7,checks
 assert checks['forward_full_vs_reverse']['relative']<1e-6 and checks['forward_full_vs_reverse']['absolute']<1e-7,checks
 v0=f3.vector();ids,_=parameter_indices(f3);selected=[]
 for k in PARAMS:
  candidates=ids[k].ravel();order=np.argsort(abs(gr3[candidates]))[-4:];selected.extend(candidates[order].tolist())
 def loss(vec):
  q=f3.clone();q.setvector(vec);q.reset();q.h=prefix[0].copy();q.r=prefix[1].copy()
  for x,v in zip(xs[2:],vs[2:]):q.step(x,v)
  _,lp=q.read();s.COUNTS['synthetic_loss_evaluations']+=1;s.COUNTS['synthetic_target_reuses']+=1;return -warm(n)*lp[y]
 fd=[]
 for j in selected:
  vp=v0.copy();vm=v0.copy();vp[j]+=1e-5;vm[j]-=1e-5;fd.append((loss(vp)-loss(vm))/2e-5)
 analytic=gr3[selected];checks['finite_difference']=dict(parameters=len(selected),relative=rel(np.array(fd),analytic),absolute=float(np.max(abs(np.array(fd)-analytic))))
 assert checks['finite_difference']['relative']<1e-6 and checks['finite_difference']['absolute']<1e-7,checks['finite_difference']
 # Current-only local rule has no earlier G/R contribution. Mismatched-input rule preserves feedback count.
 one=Controller(world,'ON1',C);local=Controller(world,'ON3',C);mis=Controller(world,'MIS3',C)
 for x,v in zip(xs,vs):
  for q in [one,local,mis]:q.step(x,v)
 one.credit(y,n);local.credit(y,n);mis.credit(y,n)
 checks['age0_only_differs_from_three']=float(np.linalg.norm(one.gradvector()-local.gradvector()))
 checks['MIS_delayed_alignment_differs']=float(np.linalg.norm(mis.gradvector()-local.gradvector()))
 assert checks['age0_only_differs_from_three']>0 and checks['MIS_delayed_alignment_differs']>0
 # A one-transition record makes all three local rules identical.
 qs=[Controller(world,k,C) for k in ['ON1','ON3','MIS3']]
 for q in qs:q.step(xs[0],vs[0]);q.credit(y,n)
 assert np.array_equal(qs[0].gradvector(),qs[1].gradvector()) and np.array_equal(qs[0].gradvector(),qs[2].gradvector())
 checks['age0_identity']=True
 # End-of-prompt latch refresh is an exogenous native read, not a recurrent step.
 q=Controller(world,'EXACT_RECORD',C);q.step(xs[0],vs[0]);h=q.h.copy();q.refresh(vs[1]);assert np.array_equal(q.h,h) and q.steps==1;checks['boundary_refresh_no_transition']=True
 # Every flat output class and distinct EOS exists; zero native output is uniform.
 q.refresh(np.zeros(16));p,lp=q.read();assert len(p)==257 and np.ptp(p)==0 and np.allclose(p,1/257,rtol=0,atol=1e-16);checks['all_classes_and_zero_native']=True
 checks['conditional_transition_bound']=.75+.25*(4/8+16/32*.5*8/8)
 s.COUNTS['synthetic_target_reuses']+=4;s.COUNTS['synthetic_loss_evaluations']+=4;s.budget()
 return checks
