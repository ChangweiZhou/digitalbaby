"""Native nine-channel byte learner; no arithmetic code or independent head."""
from v4_common import np
from geometry import Geometry,SegmentState,Boundary,BITS
from native_engine import Engine
import copy,hashlib,json,time
DELTA=30/14

def target_bits(token):return np.r_[BITS[token] if token<256 else np.zeros(8),float(token==256)]*2-1

def token_from_values(z):
 if z[8]>0:return 256
 return int(np.sum((z[:8]>0).astype(int)*(1<<np.arange(8))))

class Learner:
 def __init__(self,n,seed,segmented=True,gated=True,topographic=True,learning=True):
  self.engine=Engine(n,seed,gated);self.geometry=Geometry(n,seed+1,topographic,segmented)
  self.boundary=Boundary(seed+2);self.segmented=segmented;self.learning=learning;self.seed=seed
  self.a=1/16;self.record_applications=0;self.external_records=0;self.unique=set();self.replay=[]
  self.feedback_bytes=0;self.replay_applications=0;self.boundary_log=[];self.warmup=[];self.previous_codes=None;self.stable=0
  # Generic fixed probes contain no arithmetic delimiter or semantic target.
  self.probes=[bytes([0,137,255,33]),bytes([128,0,18]),bytes([7,203,11,91,4])]
  self.maximum_overflow=0;self.production_cpu=0.
 def digest(self):
  h=hashlib.sha256()
  for a in (self.engine.bank.fast,self.engine.bank.slow,self.engine.bank.adapt,self.engine.confirm,self.engine.confirm_at,self.engine.last_record,self.geometry.theta,self.geometry.frequency,self.boundary.w):h.update(a.tobytes())
  h.update(json.dumps((self.engine.time,self.record_applications,self.external_records,self.a,self.boundary.baseline,self.boundary.rng.bit_generator.state,sorted(self.unique),self.replay_applications,self.feedback_bytes),sort_keys=True).encode())
  return h.hexdigest()
 def context(self,prompt,explore=False):
  if len(prompt)>256:raise ValueError('INPUT_LIMIT')
  state=SegmentState()
  for b in prompt:
   state.feed(b,self.boundary.decide(b,explore) if self.segmented else False)
  return state
 def generate(self,prompt,cap=16,greedy_boundary=False):
  if len(prompt)>256:raise ValueError('INPUT_LIMIT')
  state=SegmentState();out=bytearray();policy=copy.deepcopy(self.boundary);policy.last=[]
  engine=copy.copy(self.engine);engine.bank=self.engine.bank.clone();engine.bank.adapt.fill(1.)
  for b in prompt:
   x,_=self.geometry.encode(state);engine.event(x)
   state.feed(b,policy.decide(b,not greedy_boundary) if self.segmented else False)
  for step in range(cap+1):
   x,_=self.geometry.encode(state);z=engine.values(x,DELTA);token=token_from_values(z)
   if token==256:return dict(payload=bytes(out).hex(),ended=True,truncated=False)
   if step==cap:return dict(payload=bytes(out).hex(),ended=False,truncated=True)
   engine.event(x);out.append(token);state.feed(token,policy.decide(token,not greedy_boundary) if self.segmented else False)
  raise AssertionError('unreachable')
 def _probe(self):
  codes=[set(np.flatnonzero(self.geometry.encode(self.context(p))[0]).tolist()) for p in self.probes]
  drift=None if self.previous_codes is None else float(np.mean([1-len(a&b)/max(1,len(a|b)) for a,b in zip(codes,self.previous_codes)]))
  if drift is not None and drift<=.1:self.stable+=1
  else:self.stable=0
  if self.stable>=2:self.a=min(1,self.a+1/16);self.stable=0
  self.previous_codes=codes;self.warmup.append(dict(external_records=self.external_records,drift=drift,a=self.a))
 def train(self,prompt,answer,replay=False):
  t=time.process_time();learn=self.learning
  identity=int.from_bytes(hashlib.blake2b(len(prompt).to_bytes(4,'little')+prompt+answer,digest_size=8).digest(),'little')&((1<<63)-1)
  self.boundary.last=[];self.engine.bank.adapt.fill(1.)
  state=SegmentState()
  for b in prompt:
   x,_=self.geometry.encode(state);self.engine.event(x)
   state.feed(b,self.boundary.decide(b,True) if self.segmented else False)
  losses=[];correct=0;total=0;closures_before=state.closures
  for token in (*answer,256):
   x,raw=self.geometry.encode(state);z=self.engine.values(x,DELTA)
   residual=(np.tanh(z/2)-target_bits(token))/2
   losses.append(float(np.mean(residual*residual)));correct+=int(token_from_values(z)==token);total+=1
   if learn:
    if len(self.unique)<256:self.geometry.update(x,raw,self.a)
    self.engine.event(x,-residual*self.a,record_id=identity)
   else:self.engine.event(x,record_id=identity)
   if token<256:state.feed(token,self.boundary.decide(token,True) if self.segmented else False)
  if learn and self.segmented:self.boundary.update(float(np.mean(losses)),self.a)
  self.boundary.last=[]
  self.feedback_bytes+=len(answer)+1;self.record_applications+=1
  if replay:self.replay_applications+=1
  else:
   self.external_records+=1
   if len(self.unique)<256:self.unique.add(identity)
   self.replay.append((bytes(prompt),bytes(answer)));self.replay=self.replay[-32:]
   if learn and self.external_records%16==0:self._probe()
  self.maximum_overflow=max(self.maximum_overflow,state.overflow)
  self.production_cpu+=time.process_time()-t
  return dict(record_applications=self.record_applications,external=self.external_records,replayed=replay,target_events=total,
   preupdate_correct=correct,loss=float(np.mean(losses)),a=self.a,closures=state.closures,overflow=state.overflow,
   boundary_weight_norm=float(np.linalg.norm(self.boundary.w)),record_identity=identity)
 def rest(self,seconds=86400,replay_count=8):
  t=time.process_time();self.engine.rest(seconds)
  if self.learning:self.engine.bank.slow*=.999
  self.production_cpu+=time.process_time()-t
  selected=list(self.replay[-replay_count:]) if replay_count>0 else []
  for p,a in selected:self.train(p,a,replay=True)
 def stats(self):return dict(**self.engine.stats(),record_applications=self.record_applications,external_records=self.external_records,replay_applications=self.replay_applications,observed_target_events=self.feedback_bytes,a=self.a,warmup=self.warmup,boundary_weights=self.boundary.w.tolist(),maximum_overflow=self.maximum_overflow,production_CPU_seconds=self.production_cpu)
