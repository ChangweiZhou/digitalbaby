"""One scalar content core; public input is byte, time and a permitted-write switch."""
import copy,hashlib,json,math
import numpy as np
import dependencies,settings
from representation import ByteRoles,SharedEncoder

class Learner:
    def __init__(self,arm):
        if arm not in settings.ARMS:raise ValueError('arm')
        self.arm=arm;self.sensor=ByteRoles();self.encoder=SharedEncoder()
        self.t=0.;self.bytes_seen=0;self.plastic=True;self.pending=None
        self.fast=np.zeros(self.encoder.n);self.slow=np.zeros(self.encoder.n)
        self.last=np.zeros(self.encoder.n);self.visits=np.zeros(self.encoder.n,np.uint32)
        self.summary=np.zeros(self.encoder.n) if arm=='U042-WG1' else None
        self.run=np.zeros(self.encoder.n,np.int8) if arm=='U021-SF1' else None
    def arrays(self):
        result={'fast':self.fast,'slow':self.slow,'last':self.last,'visits':self.visits}
        if self.summary is not None:result['summary']=self.summary
        if self.run is not None:result['run']=self.run
        return result
    def state_digest(self):
        h=hashlib.sha256()
        for key,a in self.arrays().items():h.update(key.encode());h.update(a.tobytes())
        h.update(json.dumps([self.arm,self.t,self.bytes_seen,self.plastic,self.sensor.line.hex()],sort_keys=True).encode())
        if self.pending is not None:
            for key,val in self.pending.items():
                h.update(key.encode());h.update(val.tobytes() if isinstance(val,np.ndarray) else json.dumps(val).encode())
        return h.hexdigest()
    def clone(self):
        other=copy.copy(self);other.sensor=copy.copy(self.sensor);other.encoder=self.encoder.clone()
        other.pending=copy.deepcopy(self.pending)
        for key,val in self.arrays().items():setattr(other,key,val.copy())
        return other
    def rest(self,seconds):
        if self.pending is not None or isinstance(seconds,bool) or not math.isfinite(seconds) or seconds<0:raise ValueError('rest')
        self.t+=seconds
    def erase(self):
        if self.pending is not None:raise ValueError('erase before complete record')
        for key,val in self.arrays().items():
            if key!='last':val.fill(0)
    def _prediction(self):
        code=self.encoder.code(self.sensor.operands());ids=np.flatnonzero((code>0)&self.encoder.mask)
        if not len(ids):raise ValueError('empty code')
        z=np.ones(len(ids))/math.sqrt(len(ids));v=float(z@(self.fast[ids]+self.slow[ids]))
        logits=-.5*(np.arange(9)-v)**2;p=np.exp(logits-np.logaddexp.reduce(logits))
        self.pending={'ids':ids,'z':z,'value':v}
        return dict(emitted=48+int(np.argmax(p)),probabilities=p.tolist(),numeric_value=v)
    def _teach(self,byte,t):
        ids,z=self.pending['ids'],self.pending['z']
        detail=dict(content_l1=0.,auxiliary_l1=0.,attenuation_l1=0.,transfer_l1=0.,commits=0,clipped=0,gate_counts=[0,0,0],raw_residual=byte-48-self.pending['value'])
        if self.plastic:
            f=self.fast[ids].copy();s=self.slow[ids].copy();v=self.visits[ids].copy()
            r=detail['raw_residual'];delta=(1/np.sqrt(1+v.astype(float)))*z*float(np.clip(r,-1,1))
            if self.arm=='STABLE':
                nf=f+.2*delta;ns=s+.8*delta
                detail['clipped']=int(np.count_nonzero((np.abs(nf)>2)|(np.abs(ns)>2)))
                nf=np.clip(nf,-2,2);ns=np.clip(ns,-2,2)
            elif self.arm=='U042-WG1':
                a=self.summary[ids].copy()
                if abs(r)<=.1:g=np.zeros(len(ids))
                else:g=np.where((v<8)|(abs(r)<=np.maximum(1,2*a)),1.,.25)
                nf=f+.2*g*delta;ns=s+.8*g*delta
                detail['clipped']=int(np.count_nonzero((np.abs(nf)>2)|(np.abs(ns)>2)))
                nf=np.clip(nf,-2,2);ns=np.clip(ns,-2,2)
                na=(7*a+min(abs(r),8))/8;self.summary[ids]=na
                detail['auxiliary_l1']=float(np.abs(na-a).sum())
                detail['gate_counts']=[int(np.count_nonzero(g==x)) for x in (0,.25,1)]
            else:
                c=self.run[ids].copy();unclipped=.5*f+delta
                detail['clipped']=int(np.count_nonzero(np.abs(unclipped)>2))
                nf=np.clip(unclipped,-2,2);ns=s.copy()
                q=int(np.sign(r)) if abs(r)>.1 else 0
                if not q:nc=np.zeros(len(ids),np.int8)
                else:nc=np.where(np.sign(c)==q,q*np.minimum(np.abs(c)+1,2),q).astype(np.int8)
                eligible=(np.abs(nc)==2)&(nc*nf>0)
                proposed=s[eligible]+nf[eligible]
                detail['clipped']+=int(np.count_nonzero(np.abs(proposed)>2))
                ns[eligible]=np.clip(proposed,-2,2);moved=ns[eligible]-s[eligible]
                nf[eligible]-=moved;nc[eligible]=0;self.run[ids]=nc
                detail.update(auxiliary_l1=float(np.abs(nc.astype(float)-c).sum()),attenuation_l1=float(np.abs(.5*f).sum()),transfer_l1=float(np.abs(moved).sum()),commits=int(eligible.sum()))
            self.fast[ids]=nf;self.slow[ids]=ns
            self.visits[ids]=np.minimum(v+1,65535)
            detail['content_l1']=float(np.abs(nf-f).sum()+np.abs(ns-s).sum())
            detail['auxiliary_l1']+=float(np.abs(self.visits[ids].astype(float)-v).sum())
        self.last[ids]=t
        return detail
    def feed(self,byte,t):
        t=settings.valid_time(t,self.t)
        if type(byte) is not int or byte not in b'012345678+=\n ':raise ValueError('byte')
        before=self.sensor.line;observed=None;detail=None
        if self.pending is not None and byte!=32:
            if byte in settings.ALPHABET:observed=byte;detail=self._teach(byte,t)
            elif byte!=10:raise ValueError('expected observed answer or newline')
            self.pending=None
        triggered=self.sensor.feed(byte);pred=self._prediction() if triggered else None
        self.t=t;self.bytes_seen+=1
        return dict(byte=byte,t=t,pre_line=before.hex(),post_line=self.sensor.line.hex(),prediction=pred,observed_target=observed,update_allowed=bool(observed is not None and self.plastic),plastic=self.plastic,update=detail)
    def snapshot(self):return {k:v.tolist() for k,v in self.arrays().items()}
    def payload_bytes(self):return sum(v.nbytes for v in self.arrays().values())
