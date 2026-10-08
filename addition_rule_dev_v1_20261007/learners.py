"""Raw-byte learners; neither accepts a truth, pair ID, phase or noise flag."""
import copy, hashlib, json, math
import numpy as np
import dependencies, settings
from representation import ByteRoles,SharedEncoder

def softmax(values):
    logits=np.asarray(values)/settings.SCALE
    if not np.isfinite(logits).all(): raise FloatingPointError('nonfinite values')
    lp=logits-np.logaddexp.reduce(logits)
    return np.exp(lp)

class Base:
    def _start(self):
        self.sensor=ByteRoles();self.encoder=SharedEncoder();self.t=0.;self.bytes_seen=0;self.plastic=True;self.pending=None
    def feed(self,byte,t):
        t=settings.valid_time(t,self.t)
        if type(byte) is not int or byte not in b'012345678+=\n ':raise ValueError('undeclared raw byte')
        previous=self.sensor.line
        dose=[0.]*(1 if isinstance(self,Residual) else 9);update=False;observed=None
        # A pending prompt is consumed only by an actually arriving digit or newline.
        if self.pending is not None and byte!=32:
            if byte in settings.ALPHABET:
                observed=byte;dose=self._teach(byte,t);update=self.plastic
            elif byte!=10: raise ValueError('expected answer digit or missing-answer newline')
            self.pending=None
        trigger=self.sensor.feed(byte)
        prediction=None
        if trigger:
            p,values,extra=self._prediction(self.sensor.operands(),t)
            self.pending=dict(t=t,operands=self.sensor.operands(),p=p.copy(),**extra)
            prediction=dict(emitted=int(settings.ALPHABET[int(np.argmax(p))]),probabilities=p.tolist(),values=np.asarray(values).tolist())
        self.t=t;self.bytes_seen+=1
        return dict(byte=byte,t=t,pre_line=previous.hex(),post_line=self.sensor.line.hex(),prediction=prediction,
            observed_target=observed,plastic=self.plastic,answer_update=update,write_l1=dose)
    def clone(self):
        c=copy.copy(self);c.sensor=copy.copy(self.sensor);c.encoder=self.encoder.clone();c.pending=copy.deepcopy(self.pending)
        self._clone_content(c);return c
    def rest(self,seconds):
        if self.pending is not None or isinstance(seconds,bool) or not math.isfinite(seconds) or seconds<0:
            raise ValueError('invalid rest or pending answer')
        self._idle(float(seconds));self.t+=float(seconds)
    def state_digest(self):
        h=hashlib.sha256()
        for a in self.content_arrays():h.update(np.ascontiguousarray(a).tobytes())
        h.update(json.dumps([self.sensor.line.hex(),self.t,self.bytes_seen,self.plastic,self.content_scalars()],sort_keys=True).encode())
        if self.pending is not None:
            for key in sorted(self.pending):
                x=self.pending[key];h.update(key.encode());h.update(x.tobytes() if isinstance(x,np.ndarray) else json.dumps(x).encode())
        return h.hexdigest()
    def mutable_bytes(self): return sum(x.nbytes for x in self.content_arrays())+len(self.sensor.line)

class Native(Base):
    def __init__(self):
        self._start();self.stores=[dependencies.inherited.Compartment() for _ in settings.ALPHABET]
        self.common=dependencies.inherited.stores.bb.bc.native().common
        for i,s in enumerate(self.stores):
            for other in self.stores[i+1:]:
                for name in ('fast','slow','adapt'):
                    if np.shares_memory(getattr(s.fly.m,name),getattr(other.fly.m,name)):raise AssertionError('native birth aliases')
    def _prediction(self,operands,t):
        code=self.encoder.code(operands);values=[]
        for s in self.stores:
            fly=s.fly.clone();fly.rest(t-s.brain_t)
            dx=self.common.observed_activity(fly.m,np.atleast_2d(code))
            values.append(float((fly.m.expression(dx)-fly.reader.predict(dx)).mean(1)[0]))
        return softmax(values),values,dict(code=code)
    def _teach(self,byte,t):
        signs=self.pending['p'].copy();signs[settings.ALPHABET.index(byte)]-=1.
        doses=[]
        for i,s in enumerate(self.stores):
            s.pending_x=self.pending['code'].copy();s.pending_t=t
            doses.append(s.teach_signed(0.,float(signs[i]),t,write=self.plastic))
        return doses
    def _idle(self,seconds):
        for s in self.stores:
            dt=self.t+seconds-s.brain_t;s.fly.rest(dt);s.brain_t=self.t+seconds
    def _clone_content(self,c):
        c.stores=[]
        for s in self.stores:
            other=copy.copy(s);other.fly=s.fly.clone();other.pending_x=None;other.pending_t=None;c.stores.append(other)
    def content_arrays(self):return [getattr(s.fly.m,k) for s in self.stores for k in ('fast','slow','adapt')]
    def content_scalars(self):return [[s.brain_t,s.elapsed_base,s.teach_seen,float(s.fly.m.elapsed),int(s.fly.m.event_count),int(s.fly.m.presentation_count)] for s in self.stores]
    def erase(self):
        if self.pending is not None:raise ValueError('erase at completed observation boundary')
        for s in self.stores:s.fly.m.fast.fill(0.);s.fly.m.slow.fill(0.)

class Residual(Base):
    def __init__(self):
        self._start();self.fast=np.zeros((self.encoder.n,1));self.slow=np.zeros_like(self.fast)
        self.last=np.zeros(self.encoder.n);self.visits=np.zeros(self.encoder.n,np.uint32)
    def _view(self,ids,t):
        dt=t-self.last[ids]
        if np.any(dt<0):raise ValueError('row clock reversed')
        return self.fast[ids]*np.exp(-dt[:,None]/settings.FAST_TAU),self.slow[ids]*np.exp(-dt[:,None]/settings.SLOW_TAU)
    def _prediction(self,operands,t):
        code=self.encoder.code(operands);ids=np.flatnonzero((code>0)&self.encoder.mask)
        if len(ids)==0:raise ValueError('empty read/write support')
        z=np.ones(len(ids))/math.sqrt(len(ids));f,s=self._view(ids,t);value=float((z@(f+s))[0])
        # Fixed ordinal rendering; no arithmetic operation or learned side head.
        values=-.5*settings.SCALE*(np.arange(9,dtype=float)-value)**2
        return softmax(values),values,dict(ids=ids,z=z,code=code,value=value)
    def _teach(self,byte,t):
        ids,z,p=self.pending['ids'],self.pending['z'],self.pending['p']
        f,s=self._view(ids,t);before_f=f.copy();before_s=s.copy()
        if self.plastic:
            rate=settings.ORDINAL_ETA/np.sqrt(1.+self.visits[ids].astype(float))
            error=float(np.clip((byte-48)-self.pending['value'],-1.,1.))
            delta=rate[:,None]*z[:,None]*error
            f=np.clip(f+(1.-settings.SLOW_SHARE)*delta,-settings.ROW_BOUND,settings.ROW_BOUND)
            s=np.clip(s+settings.SLOW_SHARE*delta,-settings.ROW_BOUND,settings.ROW_BOUND)
            self.visits[ids]=np.minimum(self.visits[ids]+1,settings.VISIT_CAP)
        self.fast[ids]=f;self.slow[ids]=s;self.last[ids]=t
        return (np.abs(f-before_f).sum(0)+np.abs(s-before_s).sum(0)).tolist()
    def _idle(self,seconds):pass
    def _clone_content(self,c):
        for name in ('fast','slow','last','visits'):setattr(c,name,getattr(self,name).copy())
    def content_arrays(self):return [self.fast,self.slow,self.last,self.visits]
    def content_scalars(self):return []
    def erase(self):
        if self.pending is not None:raise ValueError('erase at completed observation boundary')
        self.fast.fill(0.);self.slow.fill(0.)

def build(arm):
    if arm=='NATIVE':return Native()
    if arm=='RESIDUAL':return Residual()
    raise ValueError('unknown arm')
