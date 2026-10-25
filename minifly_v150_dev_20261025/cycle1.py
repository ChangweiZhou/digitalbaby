import hashlib,json,pickle,sys,time
from pathlib import Path
import numpy as np
from core import Output,gradients,probabilities,project,warmup
from fixtures import make
from util import H,R,save,event

def checks(C):
 start=time.process_time();v=np.linspace(-.4,.5,16);c=.5*C.copy();y=256;gv,gc,_,_=gradients(c,v,y);eps=1e-6;err=[]
 f=lambda cc,vv:-probabilities(cc,vv)[1][y]
 for j in range(16):
  vp=v.copy();vm=v.copy();vp[j]+=eps;vm[j]-=eps;num=(f(c,vp)-f(c,vm))/(2*eps);err.append(abs(num-gv[j]))
 for i,j in [(0,0),(31,7),(256,15),(128,3)]:
  cp=c.copy();cm=c.copy();cp[i,j]+=eps;cm[i,j]-=eps;num=(f(cp,v)-f(cm,v))/(2*eps);err.append(abs(num-gc[i,j]))
 assert max(err)<1e-6;assert np.linalg.matrix_rank(C)==15
 p,_=probabilities(c,v);p2,_=probabilities(c,v+19);assert np.max(abs(p-p2))<1e-14 and abs(p.sum()-1)<1e-14
 assert warmup(0)==1/16 and warmup(1024)==1 and warmup(512)==17/32
 proj=project(c*20);assert np.max(abs(proj.mean(1)))<1e-14 and np.max(np.linalg.norm(proj,axis=1))<=4+1e-12
 # All payloads and EOS are reachable from this mathematical readout, no mask.
 for token in range(257):assert int(np.argmax(probabilities(C,4*C[token])[0]))==token
 m=Output(C,1,'C',.1);hh=m.digest();m.read(0);assert hh==m.digest();cl=m.clone();cl.C[0,0]+=.1;assert hh==m.digest() and cl.digest()!=hh
 raw=pickle.dumps(m);m2=pickle.loads(raw);assert m2.digest()==m.digest()
 save('CYCLE1_MATH.json',{'finite_difference_max_error':max(err),'rank':15,'all257_symbols_reachable':True,'warmup_formula_checked':True,'projection_checked':True,'clone_and_pickle_equal':True,'read_only':True,'target_events':0,'cpu':time.process_time()-start})

def native_checks():
 start=time.process_time();sys.path.insert(0,str(R/'minifly_cls_dev_20261017'));import fastmem
 from native import state_hash
 sys.path.insert(0,str(H/'vendor'));import corpus
 prompts=[corpus.split(corpus.records(corpus.load(name)),2,1,2)['train'][0][:8] for name in ['bryant','buster']]
 mem=fastmem.FastMemory(homeo=False);out=[];events=0
 # Only the source-pinned training partition; no historical state or held data.
 for prompt in prompts:
  pre=mem.clone();pre._prompt(prompt);before=pre.values();x=pre.encode(pre.fe.read())[0];assert x.sum()>0
  for j in range(16):
   for sign in [-1,1]:
    m=pre.clone();dt=m._tick();xx=m._code()
    for k,s in enumerate(m.bank):
     if k==j:
      s.pending_t=m.time;s.pending_x=xx;s.teach_signed(0.,sign*1e-3,m.time,write=True)
     else:s.fly.event(dt,xx,0.,False)
     s.brain_t=m.time
    n=pre.clone();n._event()
    a=m.values();b=n.values();delta=a-b;out.append({'prompt_hex':prompt.hex(),'j':j,'sign':sign,'self':float(delta[j]),'other_max':float(np.max(abs(np.delete(delta,j))))});events+=1
    event(event='PARTIAL_NATIVE_FIXTURE',number=events)
 original=state_hash(mem);cl=mem.clone();cl.bank[0].fly.m.fast[0]+=.1;assert original==state_hash(mem)
 qualified=all(z['self']*z['sign']< -1e-12 for z in out)
 result={'status':'NATIVE_SIGN_QUALIFIED' if qualified else 'INTEGRATION_NOT_QUALIFIED','fixtures':out,'partial_native_writes':events,'matched_passive_control_events':events,'prompt_input_bytes':sum(map(len,prompts)),'qualification_budget_interpretation':'64 signed fixture increments; 64 matched passive timing events, no observed targets','native_clone_independent':True,'cpu':time.process_time()-start}
 save('CYCLE1_NATIVE.json',result);return qualified
