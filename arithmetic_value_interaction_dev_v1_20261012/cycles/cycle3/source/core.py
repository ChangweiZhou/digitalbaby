"""VALUE-INTERACT engineered online learner; arbitrary byte I/O, no task parser.
Targets enter only after prediction. Numeric auxiliary uses observed reply at EOS.
Shared and episodic banks have independent objectives, joint inference only.
"""
from dataclasses import asdict
import hashlib,json
from functools import wraps
from pathlib import Path
import sys
import numpy as np
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT/'vendor'))
from generic_byte_io import ByteCore,CoreConfig,Prediction,EOS
from vendor.legacy_core import MechanismCore as LegacyCore
CONFIGS=('BYTE-ADD','VALUE-ADD','BYTE-PROD','VALUE-PROD','LEGACY-B')
CANDIDATES=CONFIGS[:-1];BASELINE='LEGACY-B'
SHAPES={'A':(64,96),'B':(16,96),'C':(16,96),'E':(64,16),'b':(64,),
        'bB':(16,),'bC':(16,),'w_value':(64,),'b_value':(1,),
        'D':(64,166),'bD':(64,),'R':(96,257),'V':(64,257),'c':(257,)}

def strict_numpy(fun):
 @wraps(fun)
 def call(*a,**kw):
  with np.errstate(over="raise",invalid="raise",divide="raise"):
   return fun(*a,**kw)
 return call

def softmax(z):
 if not np.isfinite(z).all():raise ArithmeticError("nonfinite logits")
 p=np.exp(z-z.max());return p/p.sum()

def build_core(name,seed):
 return LegacyCore('LINEAR-FIXED-FULL',seed) if name==BASELINE else InteractionCore(name,seed)

class InteractionCore(ByteCore):
 def __init__(self,name,seed):
  if name not in CANDIDATES or type(seed) is not int or seed<0:raise ValueError('birth')
  super().__init__(CoreConfig(address_seed=seed))
  self.name,self.seed=name,seed
  self.numeric_enabled=name.startswith('VALUE');self.product=name.endswith('PROD')
  seeds=np.random.SeedSequence(seed).spawn(6)
  self.embedding=np.random.default_rng(seeds[0]).choice((-1.,1.),(256,8))/np.sqrt(8)
  self.theta={k:np.zeros(shape,dtype=np.float64) for k,shape in SHAPES.items()}
  for key,sub in zip(('A','B','C','E','D'),seeds[1:]):
   self.theta[key][:]=np.random.default_rng(sub).normal(0,1/np.sqrt(SHAPES[key][1]),SHAPES[key])
  self.M={k:np.zeros_like(a) for k,a in self.theta.items()};self.step=0
  self.aux_events=0;self.last_event_audit=None
  self._question=bytearray();self._question_total=self._reply_offset=0
  self._previous=None;self._in_reply=False
  self._numeric_reset()
 _features=LegacyCore._features
 def _numeric_reset(self):
  self._num={'valid':True,'count':0,'negative':False,'magnitude':0,'zero':False}
 def begin_sequence(self):
  super().begin_sequence();self._question.clear();self._question_total=self._reply_offset=0
  self._previous=None;self._in_reply=False;self._numeric_reset()
 def start_reply(self):
  if self._in_reply:raise RuntimeError('duplicate reply')
  super().start_reply();self._in_reply=True;self._reply_offset=0;self._previous=None;self._numeric_reset()
 def _append(self,event):
  super()._append(event)
  if 0<=event<256:
   if self._in_reply:self._previous=event;self._reply_offset+=1
   else:
    self._question_total+=1;self._question.append(event)
    if len(self._question)>4096:del self._question[:-4096]
 def _numeric_byte(self,event):
  n=self._num
  if not n['valid']:return
  i=n['count'];n['count']=min(i+1,8)
  if i==0 and event==45:n['negative']=True;return
  if not 48<=event<=57:n['valid']=False;return
  digit=event-48
  if n['zero'] or (n['negative'] and i==1 and digit==0):n['valid']=False;return
  if i==0 and digit==0:n['zero']=True
  n['magnitude']=n['magnitude']*10+digit
  if n['magnitude']>1000000:n['valid']=False
 def observed_numeric_target(self):
  n=self._num
  if not n['valid'] or n['count']==0 or (n['negative'] and n['count']==1):return None
  return -n['magnitude'] if n['negative'] else n['magnitude']
 def _h0(self):
  # Prompt-only feature view with identical live-window availability to _features.
  # Reply offset changes availability, but never supplies reply content as a feature.
  h=np.zeros(96)
  after=self._reply_offset+1 if self._in_reply else 0
  available=max(0,min(len(self._question),4096-after))
  departed=len(self._question)-available;origin=self._question_total-len(self._question)
  for local,byte in enumerate(self._question[:max(0,8-origin)]):
   i=origin+local
   if local<departed:continue
   h[8*i:8*i+8]=self.embedding[byte]
   if 48<=byte<=57:h[64+i]=(byte-48)/9.
  tail=bytes(self._question[max(0,8-origin,departed):])
  if tail:
   raw=hashlib.shake_256(f'TAIL/{self.seed}/'.encode()+tail).digest(8)
   h[88:96]=np.frombuffer(raw,np.uint8).astype(float)/127.5-1
  norm=np.linalg.norm(h);return h/norm if norm else h
 @strict_numpy
 def _forward(self,address=None,shared_only=False):
  q=self.theta;h=self._features();x=np.sqrt(96)*self._h0()
  p=np.tanh(q['B']@x+q['bB']);r=np.tanh(q['C']@x+q['bC'])
  s=p*r if self.product else (p+r)/np.sqrt(2)
  rho=np.sqrt(np.mean(s*s)+1e-6);sb=s/rho
  f=np.tanh(q['A']@x+q['E']@sb+q['b'])
  v=float(16*(q['w_value']@f+q['b_value'][0]))
  phi=np.array([v/16,np.sqrt(v*v+1e-8)/16,np.sin(2*np.pi*v/10),np.cos(2*np.pi*v/10),
                np.sin(2*np.pi*v/100),np.cos(2*np.pi*v/100)])
  d=np.r_[h,f/8,phi/np.sqrt(6)]
  t=np.tanh(q['D']@d+q['bD']);zs=h@q['R']+t@q['V']+q['c']
  if shared_only:zm=np.zeros(257)
  else:
   if address is None:address=self._address()
   zm=self._weights[address].sum(axis=0,dtype=np.float64)
  state={'h':h,'x':x,'p':p,'r':r,'s':s,'rho':rho,'sb':sb,'f':f,'v':v,'d':d,'t':t,'zs':zs,'zm':zm}
  return state,softmax(zs if shared_only else zs+.25*zm)
 def value_prediction(self):
  self._require_open();state,_=self._forward(shared_only=True);return state['v']
 def prediction(self):
  self._require_open();_,p=self._forward();return Prediction(p,int(p.argmax()))
 def shared_prediction(self):
  self._require_open();_,p=self._forward(shared_only=True);return Prediction(p,int(p.argmax()))
 def _prediction_at(self,address):
  _,p=self._forward(address);return Prediction(p,int(p.argmax()))
 @strict_numpy
 def _loss_grad(self,state,event,y=None):
  q=self.theta;a=state;pb=softmax(a['zs']);e=pb.copy();e[event]-=1
  loss=float(-np.log(max(pb[event],np.finfo(float).tiny)))
  g={k:np.zeros_like(t) for k,t in q.items()}
  g['R']=np.outer(a['h'],e);g['V']=np.outer(a['t'],e);g['c']=e
  dt=(q['V']@e)*(1-a['t']**2);g['D']=np.outer(dt,a['d']);g['bD']=dt
  dd=q['D'].T@dt;df=dd[96:160]/8;dp=dd[160:]/np.sqrt(6);v=a['v']
  dv=float(dp@np.array([1/16,v/np.sqrt(v*v+1e-8)/16,
   2*np.pi/10*np.cos(2*np.pi*v/10),-2*np.pi/10*np.sin(2*np.pi*v/10),
   2*np.pi/100*np.cos(2*np.pi*v/100),-2*np.pi/100*np.sin(2*np.pi*v/100)]))
  if y is not None:
   loss+=.5*((v-y)/16)**2;dv+=(v-y)/256
  g['w_value']=16*dv*a['f'];g['b_value'][0]=16*dv;df+=16*dv*q['w_value']
  da=df*(1-a['f']**2);g['A']=np.outer(da,a['x']);g['E']=np.outer(da,a['sb']);g['b']=da
  dsb=q['E'].T@da;ds=dsb/a['rho']-a['s']*np.mean(dsb*a['s'])/a['rho']**3
  if self.product:dp,dr=ds*a['r'],ds*a['p']
  else:dp=ds/np.sqrt(2);dr=ds/np.sqrt(2)
  dp*=1-a['p']**2;dr*=1-a['r']**2
  g['B']=np.outer(dp,a['x']);g['C']=np.outer(dr,a['x']);g['bB']=dp;g['bC']=dr
  if not np.isfinite(loss) or not all(np.isfinite(a).all() for a in g.values()):raise ArithmeticError('nonfinite loss/gradient')
  return loss,g
 @strict_numpy
 def _shared_step(self,g):
  norm=np.sqrt(sum(float(np.sum(a*a)) for a in g.values()));factor=min(1.,1/norm) if norm else 1.
  self.step+=1;correction=1-.99**self.step
  for k in self.theta:
   grad=g[k]*factor;self.M[k]*=.99;self.M[k]+=.01*grad*grad
   self.theta[k]-=.001*grad/(np.sqrt(self.M[k]/correction)+1e-8)
   if not np.isfinite(self.theta[k]).all() or not np.isfinite(self.M[k]).all():raise ArithmeticError('nonfinite optimizer state')
 def _observe_event(self,event,learn):
  self._require_open()
  if learn and not self._in_reply:raise RuntimeError('learning requires observed reply')
  address=self._address();state,prob=self._forward(address);prior=Prediction(prob,int(prob.argmax()))
  y=self.observed_numeric_target() if self._in_reply and event==EOS else None
  use=self.numeric_enabled and learn and y is not None
  self.last_event_audit={'source':'actual_observation','phase':'reply' if self._in_reply else 'prompt',
   'event':event,'aux_target':y if use else None,'numeric_observed_target':y,
   'model_predicted_value':state['v'],'auxiliary_loss':.5*((state['v']-y)/16)**2 if use else None,
   'aux_used':bool(use),'step_before':self.step,'step_after':self.step+int(learn),
   'decoder_target_input':False,'gradient_snapshot':'preupdate_preappend'}
  if learn:
   _,g=self._loss_grad(state,event,y if use else None)
   # Both independent gradient directions exist before either bank changes.
   mem= -softmax(state['zm']);mem[event]+=1
   self._shared_step(g);self._weights[address]+=(4/16*mem).astype(np.float32)
   self.learning_events+=1;self.aux_events+=int(use)
  if self._in_reply and event!=EOS:self._numeric_byte(event)
  self._append(event);return prior
 def arrays(self):
  return dict(self.theta,**{'M_'+k:v for k,v in self.M.items()},W=self._weights)
 def persistent_components(self):
  a={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in self.arrays().items()}
  a['optimizer_step']=hashlib.sha256(str(self.step).encode()).hexdigest()
  a['aux_events']=hashlib.sha256(str(self.aux_events).encode()).hexdigest();return a
 def learned_digest(self):
  a=dict(self.persistent_components(),learning_events=self.learning_events)
  return hashlib.sha256(json.dumps(a,sort_keys=True).encode()).hexdigest()
 def birth_digest(self):
  return hashlib.sha256(self.embedding.tobytes()+b''.join(a.tobytes() for a in self.theta.values())).hexdigest()
 def storage_accounting(self):
  return {'fixed_array_bytes':self.embedding.nbytes,'learned_array_bytes':sum(a.nbytes for a in self.arrays().values()),
   'working_context_bytes':self._context.nbytes+len(self._question),
   'allocated_array_bytes':sum(a.nbytes for a in self.arrays().values())+self.embedding.nbytes+self._context.nbytes,
   'hash_allocated':True,'shared_parameter_count':sum(a.size for a in self.theta.values()),
   'output_categories':257,'payload_categories':256,'context_bound':4096,'generation_cap':16,
   'production_work':'O(window + 64*96 + 2*16*96 + 64*16 + 64*166 + 160*257 + 16*257 + 62466)',
   'audit_digest_work_charged_separately':True}
 def transient_digest(self):
  m=self._metadata();m.update(question=self._question.hex(),question_total=self._question_total,
   offset=self._reply_offset,previous=self._previous,in_reply=self._in_reply,numeric=self._num)
  return hashlib.sha256(self._context.tobytes()+json.dumps(m,sort_keys=True).encode()).hexdigest()
 def save_checkpoint(self,path):
  m={'schema':'VALUE_INTERACT_CHECKPOINT_V1','name':self.name,'seed':self.seed,'base':self._metadata(),
   'question_total':self._question_total,'offset':self._reply_offset,'previous':self._previous,
   'in_reply':self._in_reply,'numeric':self._num,'step':self.step,'aux_events':self.aux_events,
   'last_event_audit':self.last_event_audit}
  with Path(path).open('wb') as out:
   np.savez_compressed(out,metadata=np.frombuffer(json.dumps(m).encode(),np.uint8),context=self._context,
    question=np.frombuffer(bytes(self._question),np.uint8),embedding=self.embedding,**self.arrays())
 @classmethod
 def load_checkpoint(cls,path):
  with np.load(path,allow_pickle=False) as d:
   m=json.loads(d['metadata'].tobytes())
   if m['schema']!='VALUE_INTERACT_CHECKPOINT_V1':raise ValueError('schema')
   c=cls(m['name'],m['seed']);expected=set(c.arrays())|{'metadata','context','question','embedding'}
   if set(d.files)!=expected:raise ValueError('array roster')
   for k,a in c.arrays().items():
    if d[k].shape!=a.shape or d[k].dtype!=a.dtype or not np.isfinite(d[k]).all():raise ValueError('array '+k)
    if k.startswith('M_') and np.any(d[k]<0):raise ValueError('negative moment')
    a[:]=d[k]
   if not np.array_equal(c.embedding,d['embedding']):raise ValueError('embedding')
   base=m['base']
   if base['config']!=asdict(c.config):raise ValueError('config')
   for k in ('context_len','next','learning_events','observed_payload_bytes','generated_payload_bytes','context_overflow_events'):
    if type(base[k]) is not int or base[k]<0:raise ValueError('integer')
   if not 0<=base['context_len']<=4096 or not 0<=base['next']<4096:raise ValueError('ring')
   if base['context_len']<4096 and base['next']!=base['context_len']:raise ValueError('ring position')
   if type(base['open']) is not bool or type(base['generated_end']) is not bool or (base['open'] and base['generated_end']):raise ValueError('boundary')
   if type(base['elapsed_seconds']) not in (float,int) or not np.isfinite(base['elapsed_seconds']) or base['elapsed_seconds']<0:raise ValueError('clock')
   if d['context'].shape!=(4096,) or d['context'].dtype!=np.uint16 or np.any(d['context']>258):raise ValueError('context')
   if d['question'].dtype!=np.uint8 or d['question'].ndim!=1 or len(d['question'])>4096:raise ValueError('question')
   for k in ('question_total','offset','step','aux_events'):
    if type(m[k]) is not int or m[k]<0:raise ValueError('state count')
   if m['step']!=base['learning_events'] or m['aux_events']>m['step']:raise ValueError('optimizer count')
   if type(m['in_reply']) is not bool or (m['previous'] is not None and (type(m['previous']) is not int or not 0<=m['previous']<256)):raise ValueError('reply state')
   if m['question_total']<len(d['question']):raise ValueError('question origin')
   n=m['numeric']
   if set(n)!={'valid','count','negative','magnitude','zero'} or any(type(n[k]) is not bool for k in ('valid','negative','zero')):raise ValueError('parser')
   if type(n['count']) is not int or not 0<=n['count']<=8 or type(n['magnitude']) is not int or not 0<=n['magnitude']<=10000009:raise ValueError('parser bound')
   c._context[:]=d['context'];c._question=bytearray(d['question'].tobytes())
   for k,a in [('context_len','_context_len'),('next','_next'),('open','_open'),('generated_end','_generated_end')]:setattr(c,a,base[k])
   for k in ('learning_events','observed_payload_bytes','generated_payload_bytes','context_overflow_events','elapsed_seconds'):setattr(c,k,base[k])
   for k,a in [('question_total','_question_total'),('offset','_reply_offset'),('previous','_previous'),('in_reply','_in_reply'),('numeric','_num'),('step','step'),('aux_events','aux_events'),('last_event_audit','last_event_audit')]:setattr(c,a,m[k])
   return c
