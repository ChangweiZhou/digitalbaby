"""One native answer path; raw observed feedback only; no arithmetic imports."""
from v51_common import np
from engine import Engine,DT,decode,residual
from representation import Sequence,Geometry
import hashlib,pickle,time

class Learner:
    def __init__(self,codes,n,seed,a_max,W,gated=True,local=True,segmented=True):
        if not codes.identity_frozen or not codes.context_frozen:raise ValueError('representation not frozen')
        self.codes=codes;self.engine=Engine(n,seed,gated)
        self.geometry=Geometry(n,seed+17,self.engine.bank.kc_side,self.engine.bank.active_fraction,local)
        self.segmented=segmented;self.a_max=a_max;self.W=W;self.external=0;self.applications=0
        self.payload_bytes=0;self.target_events=0;self.replayed=0;self.buffer=[];self.production_cpu=0.
    def digest(self):
        state=dict(self.__dict__);state.pop('production_cpu',None)
        return hashlib.sha256(pickle.dumps(state,protocol=5)).hexdigest()
    def strength(self):return self.a_max*(1/16+15/16*min(self.external/self.W,1))
    def _prompt(self,p,e):
        if len(p)>256:raise ValueError('INPUT_LIMIT')
        state=Sequence(self.codes,self.segmented);e.begin()
        for b in p:
            state.feed(b);x,_=self.geometry.encode(state);e.tick(x)
        state.begin_reply();return state
    def train(self,p,a,write=True,replay=False):
        if not self.geometry.frozen:raise ValueError('thresholds not frozen')
        start=time.process_time();state=self._prompt(p,self.engine);loss=[];correct=0
        strength=self.strength()
        for token in (*a,256):
            x,_=self.geometry.encode(state);m,_,_=self.engine.margins(x)
            s=residual(m,token,strength);correct+=int(decode(m)==token)
            mask=np.array([False]*8+[True]) if token==256 else np.ones(9,dtype=bool)
            loss.append(float(np.mean((s[mask]/max(strength,1e-15))**2)))
            self.engine.tick(x,s if write else None,replay=replay)
            if token<256:state.feed_output(token)
        self.engine.end(p,write=write,replay=replay)
        self.applications+=1;self.payload_bytes+=len(p)+len(a);self.target_events+=len(a)+1
        if replay:self.replayed+=1
        else:
            self.external+=1;self.buffer.append((bytes(p),bytes(a),bool(write)));self.buffer=self.buffer[-128:]
        self.production_cpu+=time.process_time()-start
        return dict(loss=float(np.mean(loss)),correct_teacher_prefix=correct,target_events=len(a)+1,strength=strength)
    def generate(self,p,cap=16):
        if len(p)>256:return dict(payload='',ended=False,truncated=False,error='INPUT_LIMIT')
        e=self.engine.clone();state=self._prompt(p,e);out=bytearray()
        for step in range(cap+1):
            x,_=self.geometry.encode(state);m,_,_=e.margins(x);token=decode(m)
            if token==256:
                e.tick(x);return dict(payload=bytes(out).hex(),ended=True,truncated=False)
            if step==cap:return dict(payload=bytes(out).hex(),ended=False,truncated=True)
            e.tick(x);out.append(token);state.feed_output(token)
        raise AssertionError('unreachable')
    def rest(self,seconds=3600,replays=32):
        self.engine.rest(seconds)
        for p,a,allowed in list(self.buffer[:min(replays,len(self.buffer))]):self.train(p,a,write=allowed,replay=True)

