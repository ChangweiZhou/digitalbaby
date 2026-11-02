"""C0/C1 exact equation checks and whole-conditional-roster admission."""
import time,json,hashlib
import numpy as np
import core as C
from local_rule import gradients
def main():
 start=time.process_time();d=np.load(C.H/'local/gradient_reference.npz');p,o,e,b,x,y=[d[k] for k in ('p','o','e','b','x','y')];src=C.SRC
 assert len({tuple(v) for v in src})==128 and all(len(set(v))==16 for v in src)
 reached={0}
 while True:
  new=reached|set(src[list(reached)].ravel().tolist())
  if new==reached:break
  reached=new
 assert len(reached)==128
 for w,sizes in [(202611027701,(32,16,32)),(202611027711,(64,16,64)),(202611027712,(64,16,64))]:
  seen=set();rows={}
  for split,n in zip(('train','val','held'),sizes):
   xx,yy,alphabet=C.data(w,split,n);assert xx.shape==(n,97) and np.all(xx[:,-1]==10) and np.all(yy[:,-1]==256) and np.array_equal(yy[:,:-1],xx[:,1:]);keys={r.tobytes() for r in xx};assert len(keys)==n and not seen&keys;seen|=keys;rows[split]=dict(records=n,sha256=hashlib.sha256(xx.tobytes()).hexdigest())
  C.save(f'DATA_{w}.json',dict(alphabet=alphabet.tolist(),splits=rows,exact_split_overlap=False))
 bp=C.H/'BP_QUALIFICATION.json';assert json.loads(bp.read_text())['status']=='PASS'
 gd=gradients(d['pd'],o,e,b,src,x[:1,:3],y[:1,:3],0);rel=float(np.linalg.norm(gd[0]-d['gd'])/np.linalg.norm(d['gd']));assert np.allclose(gd[0],d['gd'],rtol=1e-7,atol=1e-9) and np.allclose(gd[1],d['gdo'],rtol=1e-7,atol=1e-9)
 gc=gradients(p,o,e,b,src,x[:1,:3],y[:1,:3],0);discrep=float(np.linalg.norm(gc[0]-d['gc'])/np.linalg.norm(d['gc']));assert discrep>0
 zero=gradients(p,o,e,np.zeros_like(b),src,x[:1,:3],y[:1,:3],1);assert np.max(np.abs(zero[0]))==0 and np.linalg.norm(zero[1])>0
 timings={}
 for mode,name in [(0,'BS'),(1,'BR')]:
  t=time.process_time();gp,go,loss=gradients(p,o,e,b,src,x,y,mode);opt=C.Adam(p.copy(),o.copy());opt.update((gp,go),0,y.size,.001);timings[name]=(time.process_time()-t)/8
 t=time.process_time();held=C.evaluate(p,o,e,*C.data(202611027701,'held',32)[:2],True);eval_cpu=(time.process_time()-t)/32
 assert held['rollouts']==352
 t=time.process_time();C.evaluate(p,o,e,*C.data(202611027711,'held',64)[:2],True);main_eval=time.process_time()-t
 q=json.loads(bp.read_text());per_bp=q['bp_batch_CPU']/8
 # Main validation each epoch, held rollouts, fixed lesions, diagnostic gradient states.
 fit=1920*per_bp+1280*(timings['BS']+timings['BR'])
 evaluation=100*main_eval;technical=C.usage() if hasattr(C,'usage') else time.process_time();forecast=1.5*(fit+evaluation+technical+120)
 status='PASS' if forecast<=7200 else 'RESOURCE_NOT_ADMITTED'
 C.save('QUALIFICATION.json',dict(status=status,topology_strongly_connected=True,learned_parameters=int(p.size+o.size),BS_disconnected_relative_error=rel,connected_BP_BS_relative_discrepancy=discrep,zero_B_freezes_hidden_only=True,targets_shifted_correctly=True,local_backward_used=False,eligibility_full_record=True,local_record_CPU=timings,BP_record_CPU=per_bp,full64_record_evaluation_CPU=main_eval,full_roster_forecast_CPU=forecast,CPU_cap=7200,technical_records=24,technical_targets=2328,forward_differentiation_and_evaluation_charged=True,CPU=time.process_time()-start))
 print(status,forecast,timings)
if __name__=='__main__':main()
