"""Conditional circuit-local candidates; full BP gate required first."""
import sys,time,json
import numpy as np
import core as C
from local_rule import gradients
ACTIVE=None
def main():
 global ACTIVE
 assert json.loads((C.H/'BP_GATE.json').read_text())['passed']
 lock=json.loads((C.H/'PILOT_LOCK.json').read_text());results=[]
 for w in (202611027711,202611027712):
  for mode,arm in [(0,'BS'),(1,'BR')]:
   p,o,e,b=C.init(w,lock['bias']);x,y,_=C.data(w,'train',64);vx,vy,_=C.data(w,'val',16);opt=C.Adam(p,o);n=0;curve=[];start=time.process_time()
   for ep in range(10):
    order=C.rng(w,'order',f'epoch-{ep}').permutation(64);total=0.
    for k in range(0,64,8):
     ix=order[k:k+8];gp,go,loss=gradients(p,o,e,b,C.SRC,x[ix],y[ix],mode);m=opt.update((gp,go),n,y[ix].size,lock['lr']);n+=y[ix].size;total+=loss;ACTIVE=dict(p=p,o=o,e=e,b=b,n=n,m0=opt.m[0],m1=opt.m[1],v0=opt.v[0],v1=opt.v[1]);C.charge(8,int(y[ix].size))
    val=C.evaluate(p,o,e,vx,vy);curve.append(dict(epoch=ep+1,train_bits=total/y.size/np.log(2),val=val,optimizer=m));C.save('ACTIVE.json',dict(stage='C3',world=w,arm=arm,epoch=ep+1,targets=n,CPU=time.process_time()))
   cpu=time.process_time()-start;hx,hy,_=C.data(w,'held',64);t=time.process_time();held=C.evaluate(p,o,e,hx,hy,True);infer=time.process_time()-t
   path=C.H/f'local/{arm}_{w}.npz';np.savez(path,p=p,o=o,e=e,b=b,m0=opt.m[0],m1=opt.m[1],v0=opt.v[0],v1=opt.v[1],n=n,steps=opt.t)
   item=dict(key=f'{arm}_{w}',world=w,arm=arm,records=640,targets=n,lr=lock['lr'],bias=lock['bias'],curve=curve,held=held,training_CPU=cpu,inference_CPU=infer,checkpoint=str(path.relative_to(C.H)));C.save(item['key']+'.json',item);results.append(item);C.save('LOCAL_RESULTS.json',dict(results=results));ACTIVE=None
if __name__=='__main__':
 try:main()
 except BaseException as e:
  if ACTIVE is not None:np.savez(C.H/'local/FAILED_LOCAL_partial.npz',**ACTIVE)
  C.save('FAILURE.json',dict(error=repr(e),stage='C3',partial_saved=ACTIVE is not None));raise
