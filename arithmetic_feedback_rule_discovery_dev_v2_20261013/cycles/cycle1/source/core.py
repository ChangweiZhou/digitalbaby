"""Generic joint byte learner. No task imports, parser, numeric features or masks."""
from dataclasses import dataclass, asdict
import copy, hashlib, pickle
import numpy as np

EOS, BEGIN, REPLY = 256, 257, 258

@dataclass(frozen=True)
class Config:
    reader: str = 'GRU'
    objective: str = 'CE'
    learned: bool = True
    width: int = 64
    embedding: int = 16
    window: int = 32
    rate: float = .001
    beta: float = .99
    epsilon: float = 1e-8
    decay: float = 0.
    gradient_mode: str = 'final_window'

    def __post_init__(self):
        if self.reader not in ('GRU','LAG32') or self.objective not in ('CE','SQ'):
            raise ValueError('generic reader/objective')
        if self.gradient_mode not in ('final_window','per_target'):
            raise ValueError('gradient horizon rule')
        if min(self.width,self.embedding,self.window) < 1 or not 0 < self.rate:
            raise ValueError('positive generic dimensions/rate')
        if not 0 < self.beta < 1 or self.epsilon <= 0 or not 0 <= self.rate*self.decay < 1:
            raise ValueError('RMS/decay')

class Core:
    def __init__(self, seed, config=Config()):
        self.config=config
        rng=np.random.default_rng(seed); d=config.width; m=config.embedding
        self.theta={'E':rng.choice((-1.,1.),(259,m))/np.sqrt(m)}
        if config.reader=='GRU':
            for gate in ('z','r','g'):
                self.theta['W'+gate]=rng.normal(0,1/np.sqrt(m),(d,m))
                q,r=np.linalg.qr(rng.normal(size=(d,d)))
                self.theta['U'+gate]=q*np.sign(np.diag(r))[None,:]
                self.theta['b'+gate]=np.zeros(d)
        else:
            self.theta['A']=rng.normal(0,1/np.sqrt(m),(d,config.window*m))
            self.theta['b']=np.zeros(d)
        self.V=np.zeros((d+1,257))
        self.mom={k:np.zeros_like(v) for k,v in self.parameters().items()
                  if config.learned or k=='V'}
        self.steps=0;self.learning_events=0;self.encoder_events=0
        self.h=np.zeros(d);self.ids=[];self.tape=[];self.open=False
        self.permitted=0;self.grad={};self.loss_sum=0.;self.streaming=False
        self.evictions=0;self.aborted=0;self.last_update=None

    def parameters(self): return dict(self.theta,V=self.V)
    def zeros(self): return {k:np.zeros_like(v) for k,v in self.parameters().items() if k in self.mom}

    @staticmethod
    def sigmoid(x): return np.exp(-np.logaddexp(0.,-x))

    def transition(self,h,e):
        t=self.theta
        z=self.sigmoid(t['Wz']@e+t['Uz']@h+t['bz'])
        r=self.sigmoid(t['Wr']@e+t['Ur']@h+t['br'])
        rh=r*h;g=np.tanh(t['Wg']@e+t['Ug']@rh+t['bg'])
        return (1-z)*h+z*g,(h,e,z,r,g,rh)

    def backward(self,cache,dh,accum):
        h,e,z,r,g,rh=cache
        dz=dh*(g-h)*z*(1-z);dg=dh*z*(1-g*g)
        drh=self.theta['Ug'].T@dg;dr=drh*h*r*(1-r)
        prev=dh*(1-z)+drh*r;de=np.zeros_like(e)
        for gate,a,x in (('z',dz,h),('r',dr,h),('g',dg,rh)):
            accum['W'+gate]+=np.outer(a,e);accum['U'+gate]+=np.outer(a,x)
            accum['b'+gate]+=a;de+=self.theta['W'+gate].T@a
            if gate!='g':prev+=self.theta['U'+gate].T@a
        return prev,de

    def _pending_backward(self,accum):
        dh=np.zeros(self.config.width)
        for slot in reversed(self.tape):
            dh+=slot['dh']
            dh,de=self.backward(slot['cache'],dh,accum)
            accum['E'][slot['token']]+=de

    def _target_backward(self,dh):
        for slot in reversed(self.tape):
            dh,de=self.backward(slot['cache'],dh,self.grad)
            self.grad['E'][slot['token']]+=de

    def _append(self,token):
        self.ids=(self.ids+[token])[-self.config.window:]
        if self.config.reader!='GRU': return
        new,cache=self.transition(self.h,self.theta['E'][token]);self.h=new
        if not self.config.learned:return
        if len(self.tape)==self.config.window:
            if self.config.gradient_mode=='per_target' and not self.streaming:
                # Complete the first <=window targets before evicting their ancestry.
                self._pending_backward(self.grad)
                for slot in self.tape:slot['dh'].fill(0.)
                self.streaming=True
            self.tape.pop(0);self.evictions+=1
        self.tape.append({'token':token,'cache':cache,'dh':np.zeros(self.config.width)})

    def begin(self):
        if self.permitted:self.aborted+=1
        self.h=np.zeros(self.config.width);self.ids=[];self.tape=[]
        self.permitted=0;self.loss_sum=0.;self.grad=self.zeros();self.streaming=False
        self.open=True;self._append(BEGIN)

    def start_reply(self):
        if not self.open:raise RuntimeError('BEGIN required')
        self._append(REPLY)

    def features(self,h=None,ids=None):
        ids=self.ids if ids is None else ids
        if self.config.reader=='GRU':
            hh=self.h if h is None else h;detail=None
        else:
            m=self.config.embedding;c=np.zeros(self.config.window*m)
            first=self.config.window-len(ids)
            for i,t in enumerate(ids):c[(first+i)*m:(first+i+1)*m]=self.theta['E'][t]
            hh=np.tanh(self.theta['A']@c+self.theta['b']);detail=(hh,c,tuple(ids),first)
        v=np.r_[hh,1.];u=v/np.linalg.norm(v)
        return u,(v,detail)

    def predict(self):
        if not self.open:raise RuntimeError('BEGIN required')
        u,_=self.features();z=u@self.V
        p=np.exp(z-z.max());p/=p.sum()
        return int(np.argmax(z)),p,z

    def observe(self,target,learn=False):
        if not self.open:raise RuntimeError('BEGIN required')
        if type(learn) is not bool:raise TypeError('permission must be bool')
        if isinstance(target,(bool,np.bool_)) or not isinstance(target,(int,np.integer)) or not 0<=target<=EOS:
            raise ValueError('actual byte/EOS')
        target=int(target);u,(v,detail)=self.features();z=u@self.V
        p=np.exp(z-z.max());p/=p.sum()
        if self.config.objective=='CE':
            error=p.copy();error[target]-=1.
            loss=float(np.logaddexp.reduce(z)-z[target])
        else:
            error=z.copy();error[target]-=1.;loss=float(.5*(error@error))
        event={'target':target,'permitted':learn,'choice':int(np.argmax(z)),
               'loss':loss,'readout_updates':0,'encoder_steps':0,'optimizer_steps':0}
        if learn:
            self.permitted+=1;self.loss_sum+=loss
            self.grad['V']+=np.outer(u,error)
            if self.config.learned:
                gu=self.V@error;gv=(gu-u*(u@gu))/np.linalg.norm(v);dh=gv[:-1]
                if self.config.reader=='GRU':
                    if self.streaming:self._target_backward(dh)
                    else:self.tape[-1]['dh']+=dh
                else:
                    hh,c,ids,first=detail;dl=dh*(1-hh*hh)
                    self.grad['A']+=np.outer(dl,c);self.grad['b']+=dl
                    dc=self.theta['A'].T@dl;m=self.config.embedding
                    for i,t in enumerate(ids):self.grad['E'][t]+=dc[(first+i)*m:(first+i+1)*m]
        if target==EOS:
            if learn and self.permitted:
                self._flush();event.update(readout_updates=1,optimizer_steps=1,encoder_steps=int(self.config.learned))
            self.open=False
        else:self._append(target)
        return event

    def pending_gradient(self):
        grad={k:v.copy() for k,v in self.grad.items()}
        if self.config.learned and self.config.reader=='GRU':self._pending_backward(grad)
        return {k:v/max(1,self.permitted) for k,v in grad.items()}

    def _flush(self):
        params=self.parameters()
        if any(not params[k].flags.writeable or not self.mom[k].flags.writeable for k in self.mom):
            raise ValueError('persistent arrays are read-only')
        grad=self.pending_gradient()
        norm=float(np.sqrt(sum(float((g*g).sum()) for g in grad.values())))
        clip=min(1.,1/max(norm,1e-30));step=self.steps+1
        update_norms={};component_norms={k:float(np.linalg.norm(g)) for k,g in grad.items()}
        for k,g in grad.items():
            g=g*clip;self.mom[k]*=self.config.beta;self.mom[k]+=(1-self.config.beta)*g*g
            delta=self.config.rate*g/(np.sqrt(self.mom[k]/(1-self.config.beta**step))+self.config.epsilon)
            params[k]*=(1-self.config.rate*self.config.decay);params[k]-=delta
            update_norms[k]=float(np.linalg.norm(delta))
        self.steps=step;self.learning_events+=1;self.encoder_events+=int(self.config.learned)
        self.last_update={'targets':self.permitted,'mean_loss':self.loss_sum/self.permitted,
                          'unclipped_gradient_norm':norm,'clip_factor':clip,
                          'component_gradient_norms':component_norms,'component_step_norms':update_norms,
                          'decay_factor':1-self.config.rate*self.config.decay}
        self.permitted=0;self.grad=self.zeros();self.tape=[];self.streaming=False
        self.assert_finite()

    def generate(self,cap=16):
        if type(cap) is not int or cap<0:raise ValueError('nonnegative payload cap')
        if not self.open:raise RuntimeError('BEGIN required')
        h=self.h.copy();ids=self.ids.copy();answer=[]
        for _ in range(cap+1):
            u,_=self.features(h,ids);token=int(np.argmax(u@self.V))
            if token==EOS:return bytes(answer),'EOS'
            if len(answer)==cap:return bytes(answer),'TRUNCATED'
            answer.append(token);ids=(ids+[token])[-self.config.window:]
            if self.config.reader=='GRU':h,_=self.transition(h,self.theta['E'][token])
        raise AssertionError('generation bound')

    def rest(self,seconds):
        if not np.isfinite(seconds) or seconds<0:raise ValueError('finite nonnegative delay')

    def clone(self):return copy.deepcopy(self)
    def assert_finite(self):
        if not all(np.isfinite(v).all() for v in list(self.parameters().values())+list(self.mom.values())+[self.h]):
            raise FloatingPointError('nonfinite core')
    def learning_digest(self):
        h=hashlib.sha256()
        for group in (self.parameters(),self.mom):
            for k,v in sorted(group.items()):h.update(k.encode());h.update(v.tobytes())
        h.update(str((self.steps,self.learning_events,self.encoder_events)).encode());return h.hexdigest()
    def bytes_state(self):
        return {'parameters':sum(v.nbytes for v in self.parameters().values()),
                'moments':sum(v.nbytes for v in self.mom.values()),
                'pending_gradients':sum(v.nbytes for v in self.grad.values()),
                'bounded_transient_pickle_bytes':len(pickle.dumps((self.h,self.ids,self.tape),protocol=5))}
    def save(self,path):
        with open(path,'wb') as f:pickle.dump(self,f,protocol=5)
    @staticmethod
    def load(path):
        with open(path,'rb') as f:return pickle.load(f)
