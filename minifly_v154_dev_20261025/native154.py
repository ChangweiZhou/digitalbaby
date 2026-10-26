"""Pinned V152 native adapter, with only the declared receptor-trace factorial."""
import sys,math,time
import numpy as np
import support154 as s
sys.path.insert(0,str(s.ROOT/'minifly_v152_dev_20261025'))
import models152 as parent
parent.COUNTS=s.COUNTS;parent.TIMES=s.TIMES
fastmem=parent.fastmem;C=parent.C;EOS=256;DELTA=fastmem.DELTA
FRONTS={'I_FAST':(False,10.),'P_FAST':(True,10.),'I_SLOW':(False,-DELTA/math.log(.95)),'P_SLOW':(True,-DELTA/math.log(.95))}
class Trace(fastmem.FEPerm):
 def __init__(self,name):
  super().__init__();self.permute,self.tau=FRONTS[name];self.variant=name
 def feed(self,b,t):
  self.advance(t)
  if self.permute:self.p=self.p[fastmem.PERM]
  self.p=self.p+fastmem.BC.R_TABLE[int(b)]
class Native(parent.Native):
 def __init__(self,teacher,front):super().__init__(teacher);self.variant=front
 def _new_front(self):f=Trace(self.variant);f.t=self.time;return f

def prepared(model,prompt):
 q=model.clone();q._prompt(prompt);s.COUNTS['query_prompt_bytes_executed']+=len(prompt);return q

def pair(model,prompt,answer,kind='native',condition=None,decompose=False):
 """Share only the exact immutable prompt preparation; continuations are independent."""
 s.query(kind);t=time.process_time()
 if isinstance(model,Native):pre=prepared(model,prompt)
 else:pre=model.clone();parent.prepare(pre,prompt)
 if condition=='NO_Q':pre.q.fill(0)
 elif condition=='Q_ONLY':
  q=parent.Bias();q.q=pre.q.copy();pre=q
 own=pre.clone();tf=pre.clone();choices=[];output=[];ended=False
 for _ in range(4):
  p,lp=own.read();y=int(p.argmax());choices.append(y)
  if y==EOS:ended=True;break
  output.append(y);own.step(y)
 sites=[]
 for y in list(answer)+[EOS]:
  p,lp=tf.read();row=dict(y=y,lp=float(lp[y]),argmax=int(p.argmax()))
  if decompose and isinstance(tf,parent.Recurrent):
   v=tf.R@np.r_[tf.h,tf.r];ctx=C@((v-v.mean())/4);row.update(context_vs_EOS=float(ctx[y]-ctx[EOS]),bias_vs_EOS=float(tf.q[y]-tf.q[EOS]),full_vs_EOS=float(ctx[y]+tf.q[y]-ctx[EOS]-tf.q[EOS]))
  sites.append(row)
  if y!=EOS:tf.step(y)
 out=dict(expected_hex=answer.hex(),output_hex=bytes(output).hex(),EOS=ended,choices=choices,correct=ended and bytes(output)==answer,teacher_forced=sites)
 s.TIMES['paired_evaluation_CPU']+=time.process_time()-t;return out

def taps(view):
 f=view.fe.clone();f.advance(view.time+DELTA);rho=f.read();x=view.encode(rho)[0]
 fly=view.bank[0].fly.clone();fly.rest(DELTA);obs=fastmem.NATIVE.common.observed_activity(fly.m,x[None,:])[0]
 return {'RHO':rho,'X':x,'OBS':obs}
def metrics(rows):
 n=sum(len(x['teacher_forced']) for x in rows);return dict(records=len(rows),correct=sum(x['correct'] for x in rows),accuracy=sum(x['correct'] for x in rows)/len(rows),BPS=-sum(z['lp'] for x in rows for z in x['teacher_forced'])/n/math.log(2),targets=n,empty_EOS=sum(x['EOS'] and x['output_hex']=='' for x in rows),CAP=sum(not x['EOS'] for x in rows),first_payload_correct=sum(x['teacher_forced'][0]['argmax']==x['teacher_forced'][0]['y'] for x in rows),true_EOS_correct=sum(x['teacher_forced'][-1]['argmax']==EOS for x in rows))
def evaluate(model,records,kind='native',condition=None,decompose=False):
 rows=[]
 for r in records:
  p,a=s.unpack(r);rows.append(dict(id=r['id'],**pair(model,p,a,kind,condition,decompose)))
 return dict(metrics=metrics(rows),rows=rows)
