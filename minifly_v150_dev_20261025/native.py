"""Pinned native adapter: only common warmup and declared output teaching/code changes."""
import sys,copy,pickle,hashlib,time,json
from pathlib import Path
import numpy as np
from util import H,R,save,event,budget,cpu
sys.path.insert(0,str(R/'minifly_cls_dev_20261017'))
import fastmem
from core import probabilities,gradients,project,warmup

def state_hash(m):return hashlib.sha256(pickle.dumps(m,protocol=5)).hexdigest()
class Native(fastmem.FastMemory):
 def __init__(self,arm,eta):
  super().__init__(homeo=False);self.arm=arm;self.eta=eta;self.C=fastmem.CODE.copy();self.target_n=0;self.prequential_nats=0.;self.coef_l2=0.
 def clone(self):
  out=super().clone();out.C=self.C.copy();out.theta=self.theta.copy();out.freq=self.freq.copy();return out
 def _event(self,token=None,target=None):
  if target is None:return super()._event(token,None)
  v=self.values();gv,gc,r,loss=gradients(self.C,v,target);e=(np.tanh((v-v.mean())/2)-self.C[target])/2 if self.arm=='A' else gv;coef=self.eta*warmup(self.target_n)*e
  assert np.isfinite(coef).all() and np.max(abs(coef))<=1
  dt=self._tick();x=self._code()
  for j,s in enumerate(self.bank):
   s.pending_t=self.time;s.pending_x=x;s.teach_signed(0.,float(coef[j]),self.time,write=True);s.brain_t=self.time
  if self.arm=='C':self.C=project(self.C-self.eta/16*warmup(self.target_n)*gc)
  if token is not None and token!=256:self.fe.feed(token,self.time)
  self.target_n+=1;self.prequential_nats+=loss;self.coef_l2+=float(coef@coef)
 def read(self):return probabilities(self.C,self.values())

def evaluate(m,records):
 before=state_hash(m);rows=[];total=0.;num=0
 for rec in records:
  q=m.clone();q._prompt(b'');out=[]
  for y in list(rec)+[256]:
   p,lp=q.read();out.append([y,float(lp[y]),int(np.argmax(p))]);total-=float(lp[y]);num+=1
   if y!=256:q._event(y)
  rows.append(out)
 assert state_hash(m)==before;return {'BPS':total/num/np.log(2),'targets':num,'rows':rows}
def generate(m,prefix,cap=64):
 before=state_hash(m);q=m.clone();q._prompt(prefix);out=[];status='CAP'
 for _ in range(cap):
  p,_=q.read();y=int(np.argmax(p))
  if y==256:status='EOS';break
  out.append(y);q._event(y)
 assert state_hash(m)==before;return {'output_hex':bytes(out).hex(),'status':status,'decisions':len(out)+(status=='EOS')}
def run(eta):
 sys.path.insert(0,str(H/'vendor'));import corpus,ladder
 result=[];begin=cpu()
 for name in ['bryant','buster']:
  sp=corpus.split(corpus.records(corpus.load(name)),2,1,2)
  refs={}
  for cls in [ladder.Unigram,ladder.Bigram]:
   ref=cls().fit(sp['train']);loss=n=0
   for rec in sp['held']:
    ctx=[]
    for y in list(rec)+[256]:loss-=np.log(ref.prob(ctx,y));n+=1;ctx.append(y)
   refs[ref.name]=float(loss/n/np.log(2))
  for arm in ['A','B','C']:
   budget(300);ident=f'{name}_{arm}';event(event='NATIVE_FIT_START',id=ident);m=Native(arm,eta);t=time.process_time()
   for i,rec in enumerate(sp['train']):
    m.train(b'',rec);save('ACTIVE.json',{'stage':'C','id':ident,'records':m.records,'target_events':m.target_n,'CPU':cpu()});budget(200)
   (H/f'NATIVE_{ident}.pkl').write_bytes(pickle.dumps(m));train_seconds=time.process_time()-t
   cal=evaluate(m,sp['calib']);he=evaluate(m,sp['held']);own=[]
   for rec in sp['held'][:16]:own.append({'prefix_hex':rec[:8].hex(),'expected_hex':rec[8:].hex(),**generate(m,rec[:8])})
   row={'id':ident,'arm':arm,'eta':eta,'records':m.records,'targets':m.target_n,'native_compartment_calls':m.target_n*16,'prequential_BPS':m.prequential_nats/m.target_n/np.log(2),'calibration':cal,'development_evaluation':he,'generations':own,'references':refs,'code_drift':float(np.linalg.norm(m.C-fastmem.CODE)),'training_CPU':train_seconds,'CPU':time.process_time()-t};save(f'NATIVE_{ident}.json',row);result.append(row);event(event='NATIVE_FIT_COMMIT',id=ident,targets=m.target_n)
 save('NATIVE_COMPACT.json',[{k:z[k] for k in ['id','targets','records','prequential_BPS','code_drift','CPU']}|{'development_BPS':z['development_evaluation']['BPS'],'references':z['references']} for z in result]);return result
