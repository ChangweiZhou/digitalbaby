"""Unchanged native writer with read-only controller hooks. No task semantics."""
import sys,time,copy,pickle
from collections import deque
import numpy as np
import support171 as s
sys.path.insert(0,str(s.ROOT/'minifly_v154_dev_20261025'))
import native154 as n
n.parent.COUNTS=s.COUNTS;n.parent.TIMES=s.TIMES
C=n.C;EOS=256
def birth():
 m=n.Native('N_2B','P_SLOW');assert not m.homeo and m.fresh_adapt and not m.q.any();s.COUNTS['model_births']+=1;return m
def raw(m,kind='read'):
 t=time.process_time();v=m.values();s.TIMES['native_vector_read_CPU']+=time.process_time()-t;s.COUNTS['native_vector_reads']+=1;s.COUNTS['native_compartment_reads']+=16;s.COUNTS['native_reads_'+kind]+=1;s.budget();return v
def baseprob(v):
 from controller171 import probs
 return probs(v,C)
def prepare(m,ctrls,prompt,condition=None,carrier=None):
 if len(prompt)>64:raise ValueError('INPUT_CAP')
 m._control(True)
 for c in ctrls:c.reset()
 def value():
  v=raw(m,'prompt')
  if condition=='native_zero':v=np.zeros(16)
  if condition=='constant_native_carrier':v=carrier.copy()
  return v
 for x in prompt:
  m.step(x)
  if ctrls:
   v=value()
   for c in ctrls:
    if condition=='feedback_cut':c.r.fill(0)
    c.step(x,v)
 m._control(False);v=value()
 for c in ctrls:c.refresh(v)
 return v
def teach_event(m,y,v,p0,write=True):
 if write:
  gv,_=n.parent.value_error(p0,y);coef=.2*n.parent.warmup(m.n)*gv
  assert np.isfinite(coef).all() and np.max(abs(coef))<=1
  m._tick();x=m._code()
  for j,b in enumerate(m.bank):
   b.pending_t=m.time;b.pending_x=x;b.teach_signed(0.,float(coef[j]),m.time,write=True);b.brain_t=m.time
  s.COUNTS['native_signed_API_calls']+=16
  if y!=EOS:m.fe.feed(y,m.time)
 else:n.fastmem.FastMemory._event(m,y,None)
 m.n+=1;s.COUNTS['physical_native_targets']+=1
def train_record(m,ctrls,prompt,answer,tapes=None,metadata=None,write=True,technical=False):
 metadata=metadata or {};s.records(prompt,answer,1+len(ctrls),technical);v=prepare(m,ctrls,prompt);arr=[];target_n=m.n
 for k,y in enumerate(list(answer)+[EOS]):
  p0,lp0=baseprob(v);wN=n.parent.warmup(m.n);base=p0.copy();base[y]=-np.inf;rv=int(base.argmax())
  site=dict(y=int(y),argmax=int(p0.argmax()),lp=float(lp0[y]),rival=rv,margin=float(lp0[y]-lp0[rv]),warmup=wN,ages=[])
  if tapes:
   tapes['NATIVE'].add(dict(metadata,site=k,n=m.n,native_time=m.time,**site),p=p0,p0=p0,v=v,h=np.empty(0),z=np.empty(0),a=np.empty(0),u=np.empty(0),g=np.ones(16),direct_error=np.empty(0),signal_norms=np.zeros(66))
  for c in ctrls:
   info,arrays=c.credit(y,m.n)
   if tapes:tapes[c.role].add(dict(metadata,site=k,n=m.n,native_time=m.time,**info),p0=p0,**arrays)
  teach_event(m,y,v,p0,write)
  if y!=EOS:
   v=raw(m,'feedback_arrival')
   for c in ctrls:c.step(y,v)
  arr.append(site)
 updates={c.role:c.apply(write) for c in ctrls}
 for c in ctrls:c.reset()
 m.records+=1;s.COUNTS['physical_native_records']+=1
 return dict(native=arr,updates=updates,n_start=target_n,n_end=m.n,native_time=m.time)
def own(m,c,prompt,condition=None,carrier=None):
 v=prepare(m,[] if c is None else [c],prompt,condition,carrier);out=[];choices=[];eos=False
 for step in range(4):
  p,lp=baseprob(v) if c is None else c.read();y=int(p.argmax());choices.append(y)
  if y==EOS:eos=True;break
  out.append(y)
  if step<3:
   m.step(y);v=raw(m,'own_continuation')
   if condition=='native_zero':v=np.zeros(16)
   if condition=='constant_native_carrier':v=carrier.copy()
   if c is not None:
    if condition=='feedback_cut':c.r.fill(0)
    c.step(y,v)
 return dict(output_hex=bytes(out).hex(),EOS=eos,CAP=not eos,choices=choices)
def pair(m,c,prompt,answer,condition=None,carrier=None):
 s.COUNTS['paired_queries']+=1;s.budget();view=m.clone();cc=None if c is None else c.clone();o=own(view,cc,prompt,condition,carrier)
 view=m.clone();cc=None if c is None else c.clone();v=prepare(view,[] if cc is None else [cc],prompt,condition,carrier);rows=[]
 for y in list(answer)+[EOS]:
  p,lp=baseprob(v) if cc is None else cc.read();masked=p.copy();masked[y]=-np.inf;rv=int(masked.argmax());rows.append(dict(y=y,argmax=int(p.argmax()),lp=float(lp[y]),rival=rv,margin=float(lp[y]-lp[rv])))
  if y!=EOS:
   view.step(y);v=raw(view,'TF_continuation')
   if condition=='native_zero':v=np.zeros(16)
   if condition=='constant_native_carrier':v=carrier.copy()
   if cc is not None:
    if condition=='feedback_cut':cc.r.fill(0)
    cc.step(y,v)
 o['correct']=o['EOS'] and o['output_hex']==answer.hex();o['teacher_forced']=rows;return o
def panel(m,controllers,rows,tag):
 before=s.digest(m);answer={}
 for role,c in controllers.items():
  out=[]
  for row in rows:out.append(dict(id=row['id'],**pair(m,c,bytes.fromhex(row['prompt']),bytes.fromhex(row['answer']))))
  p=s.H/'local'/f'{tag}_{role}.json';s.save(str(p.relative_to(s.H)),dict(tag=tag,role=role,rows=out));answer[role]=dict(correct=sum(x['correct'] for x in out),records=len(out),correct_ids=[x['id'] for x in out if x['correct']],path=str(p.relative_to(s.H)),sha256=s.sha(p),empty_EOS=sum(x['EOS'] and not x['output_hex'] for x in out),CAP=sum(x['CAP'] for x in out),TF_targets=sum(len(x['teacher_forced']) for x in out),TF_correct=sum(v['y']==v['argmax'] for x in out for v in x['teacher_forced']),TF_negative_log_likelihood=sum(-v['lp'] for x in out for v in x['teacher_forced']))
 assert s.digest(m)==before
 s.log('PANEL_COMMIT',tag=tag,results=answer);return answer
def equal(a,b,path='root'):
 """Operative value comparison, not pickle memoization identity."""
 import scipy.sparse as sp
 if isinstance(a,np.ndarray):assert isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True),path
 elif sp.issparse(a):assert sp.issparse(b) and a.shape==b.shape and (a!=b).nnz==0,path
 elif isinstance(a,dict):
  assert set(a)==set(b),path
  for k in a:equal(a[k],b[k],path+'.'+str(k))
 elif isinstance(a,(list,tuple,deque)):
  assert len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'.'+str(i))
 elif hasattr(a,'__dict__') and not callable(a):equal(a.__dict__,b.__dict__,path+'.dict')
 else:assert a==b,path
