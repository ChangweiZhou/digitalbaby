"""Task-agnostic, raw-byte native learner. No evaluator/data dependency."""
from r2_common import np
from r2_engine import Engine,DT,decode,residual
from r2_geometry import Sequence,Geometry
import copy,hashlib,pickle,time

class Frames:
    """Single causal frame source for calibration, learning and own generation."""
    def __init__(self,codes):self.state=Sequence(codes)
    def prompt(self,p):
        if len(p)>256:raise ValueError('INPUT_LIMIT')
        for b in p:
            self.state.feed(b);yield self.state
        self.state.begin_reply()
    def prediction(self):return self.state
    def observed(self,b):
        if b<256:self.state.feed_output(b)

def query_view(engine):
    # Forward values/adaptation are private. Readonly q/ring arrays are never touched:
    # all query events have coeff=None, and end/confirm is never called.
    e=copy.copy(engine);e.bank=engine.bank.clone();e.signs=engine.signs.copy();return e

class Learner:
    def __init__(self,codes,seed,gated=True,geometry=None,d0_events=0):
        if not (codes.identity_frozen and codes.context_frozen):raise ValueError('representation unfrozen')
        self.codes=codes;self.engine=Engine(65536,seed,gated)
        self.geometry=geometry or Geometry(65536,seed+17,self.engine.bank.kc_side,self.engine.bank.active_fraction)
        self.engine.time=d0_events*DT;self.engine.events=d0_events
        self.external=0;self.payload_bytes=0;self.target_events=0;self.production_cpu=0.
    def digest(self):
        # Includes all persistent state; run around whole probe batches, not each item.
        return hashlib.sha256(pickle.dumps((self.codes,self.geometry,self.engine,self.external,self.payload_bytes,self.target_events,self.production_cpu),protocol=5)).hexdigest()
    def strength(self):return (1/16)*(1/16+15/16*min(self.external/1024,1))
    def prompt(self,p,e):
        f=Frames(self.codes);e.begin()
        for state in f.prompt(p):e.tick(self.geometry.encode(state)[0])
        return f
    def train(self,p,a,write=True):
        if not self.geometry.frozen:raise ValueError('thresholds unfrozen')
        t=time.process_time();f=self.prompt(p,self.engine);correct=0;bitcorrect=0;bits=0;loss=0.;strength=self.strength()
        for token in (*a,256):
            x,_=self.geometry.encode(f.prediction());m,_,_=self.engine.margins(x)
            if not np.isfinite(m).all():raise FloatingPointError('nonfinite read')
            s=residual(m,token,strength);correct+=int(decode(m)==token)
            mask=np.array([False]*8+[True]) if token==256 else np.ones(9,dtype=bool)
            from r2_engine import targets
            bitcorrect+=int(np.sum((np.where(m[mask]>0,1,-1))==targets(token)[mask]));bits+=int(mask.sum())
            loss+=float(np.mean((s[mask]/strength)**2))
            self.engine.tick(x,s if write else None);f.observed(token)
        self.engine.end(p,write=write);self.external+=1;self.payload_bytes+=len(p)+len(a);self.target_events+=len(a)+1
        elapsed=time.process_time()-t;self.production_cpu+=elapsed
        return dict(loss=loss/(len(a)+1),correct_teacher_prefix=correct,target_events=len(a)+1,
                    bit_correct=bitcorrect,bit_count=bits,strength=strength,write=bool(write),cpu=elapsed)
    def generate(self,p,cap=16):
        e=query_view(self.engine);f=self.prompt(p,e);out=bytearray()
        for step in range(cap+1):
            x,_=self.geometry.encode(f.prediction());m,_,_=e.margins(x);token=decode(m)
            if token==256:
                e.tick(x);return dict(payload=out.hex(),ended=True,truncated=False,events=e.events-self.engine.events)
            if step==cap:return dict(payload=out.hex(),ended=False,truncated=True,events=e.events-self.engine.events)
            e.tick(x);out.append(token);f.observed(token)
        raise AssertionError('unreachable')
    def rest(self,seconds):self.engine.rest(seconds)

def calibrate(codes,geometry,records,check=lambda:None):
    events=0;frames=[];closures=0
    for p,a in records:
        check();f=Frames(codes)
        for state in f.prompt(p):geometry.calibrate(state);events+=1
        for token in (*a,256):
            state=f.prediction();x,raw=geometry.encode(state)
            if len(frames)<128:frames.append((x.copy(),raw.copy()))
            geometry.calibrate(state);events+=1;f.observed(token)
        closures+=f.state.closures
    geometry.frozen=True
    return dict(events=events,records=len(records),closures=closures),frames
