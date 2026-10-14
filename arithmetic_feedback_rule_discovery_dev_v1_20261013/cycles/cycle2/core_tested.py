"""Generic byte-feedback recurrent learner. No evaluator import or arithmetic features."""
import copy
import hashlib
import pickle
from dataclasses import dataclass, asdict
import numpy as np

EOS, BEGIN, REPLY = 256, 257, 258

@dataclass(frozen=True)
class Config:
    depth: int = 1
    full: bool = True
    learned: bool = True
    width: int = 64
    embedding: int = 16
    window: int = 32
    rate: float = .001

    def __post_init__(self):
        if self.depth not in (1,4) or self.window < 1 or self.width < 1 or self.embedding < 1:
            raise ValueError("invalid recurrent configuration")
        if not np.isfinite(self.rate) or self.rate <= 0: raise ValueError("invalid rate")

class Core:
    def __init__(self, seed, config=Config()):
        self.config = config
        rng = np.random.default_rng(seed)
        d, m = config.width, config.embedding
        self.theta = {'E': rng.choice((-1., 1.), (259, m))/np.sqrt(m)}
        for gate in ('z','r','g'):
            self.theta['W'+gate] = rng.normal(0, 1/np.sqrt(m), (d,m))
            q, r = np.linalg.qr(rng.normal(size=(d,d)))
            self.theta['U'+gate] = q * np.sign(np.diag(r))[None,:]
            self.theta['b'+gate] = np.zeros(d)
        self.mom = {k: np.zeros_like(v) for k,v in self.theta.items()} if config.learned else {}
        self.V = np.zeros((d+1,257))
        self.P = np.eye(d+1)*10 if config.full else np.ones(d+1)*10
        self.h = np.zeros(d)
        self.tape = []
        self.permitted = 0
        self.steps = 0
        self.learning_events = 0
        self.aborted = 0
        self.evictions = 0
        self.dropped_targets = 0
        self.last_update = None
        self.trace = []
        self.open = False

    @staticmethod
    def _sigmoid(x):
        # Stable, generic saturation protection; not a vocabulary constraint.
        return np.exp(-np.logaddexp(0., -x))

    def transition(self, h, e):
        t = self.theta
        z = self._sigmoid(t['Wz']@e + t['Uz']@h + t['bz'])
        r = self._sigmoid(t['Wr']@e + t['Ur']@h + t['br'])
        rh = r*h
        g = np.tanh(t['Wg']@e + t['Ug']@rh + t['bg'])
        out = (1-z)*h + z*g
        return out, (h.copy(), e.copy(), z, r, g, rh)

    def backward_transition(self, cache, grad, accum):
        h,e,z,r,g,rh = cache
        dz = grad*(g-h)*z*(1-z)
        dg = grad*z*(1-g*g)
        drh = self.theta['Ug'].T@dg
        dr = drh*h*r*(1-r)
        dh = grad*(1-z) + drh*r
        de = np.zeros_like(e)
        for gate, a, prev in (('z',dz,h),('r',dr,h),('g',dg,rh)):
            accum['W'+gate] += np.outer(a,e)
            accum['U'+gate] += np.outer(a,prev)
            accum['b'+gate] += a
            de += self.theta['W'+gate].T@a
            if gate != 'g':
                dh += self.theta['U'+gate].T@a
        return dh, de

    def zeros(self):
        return {k: np.zeros_like(v) for k,v in self.theta.items()}

    def features(self):
        h = self.h.copy()
        caches = []
        for _ in range(self.config.depth-1):
            h, cache = self.transition(h, np.zeros(self.config.embedding))
            caches.append(cache)
        v = np.r_[h,1.]
        u = v / np.linalg.norm(v)
        return u, caches

    def predict(self):
        if not self.open: raise RuntimeError("BEGIN required")
        u, _ = self.features()
        logits = u@self.V
        p = np.exp(logits-np.max(logits))
        p /= p.sum()
        return int(np.argmax(logits)), p, logits

    def _append(self, token, record=True):
        self.h, cache = self.transition(self.h, self.theta['E'][token])
        if record and self.config.learned:
            self.tape.append({'token':token,'cache':cache,'dh':np.zeros(self.config.width),
                              'direct':None,'targets':0})
            if len(self.tape) > self.config.window:
                old = self.tape.pop(0)
                self.evictions += 1
                self.dropped_targets += old['targets']

    def begin(self):
        self.open = True
        if self.permitted:
            self.aborted += 1
        self.h = np.zeros(self.config.width)
        self.tape = []
        self.permitted = 0
        self.trace = []
        self._append(BEGIN)

    def start_reply(self):
        if not self.open: raise RuntimeError("BEGIN required")
        self._append(REPLY)

    def observe(self, target, learn=False):
        if not self.open: raise RuntimeError("BEGIN required")
        if type(learn) is not bool: raise TypeError("learn must be boolean")
        if isinstance(target, (bool,np.bool_)) or not isinstance(target, (int,np.integer)) or not 0 <= target <= EOS:
            raise ValueError('observed event must be a payload byte or actual EOS')
        u, caches = self.features()
        z = u@self.V
        choice = int(np.argmax(z))
        error = -z.copy()
        error[target] += 1
        loss = float(.5*(error@error))
        event = {'target':int(target),'permitted':bool(learn),'choice':choice,'loss':loss,
                 'readout_updates':0,'encoder_steps':0}
        if learn:
            if self.config.learned:
                if not self.tape:
                    raise RuntimeError('public BEGIN required before feedback')
                # Pre-update readout, held fixed in independent gradient reference.
                gu = -(self.V@error)
                v = np.r_[self._processed_h(caches),1.]
                gv = (gu-u*(u@gu))/np.linalg.norm(v)
                direct = self.zeros()
                gh = gv[:-1]
                for cache in reversed(caches):
                    gh,_ = self.backward_transition(cache,gh,direct)
                slot = self.tape[-1]
                slot['dh'] += gh
                if slot['direct'] is None:
                    slot['direct'] = direct
                else:
                    for k in direct: slot['direct'][k] += direct[k]
                slot['targets'] += 1
                self.permitted += 1
            q = self.P@u if self.config.full else self.P*u
            denominator = float(1+u@q)
            if not np.isfinite(denominator) or denominator <= 0:
                raise FloatingPointError('nonpositive confidence denominator')
            self.V += np.outer(q,error)/denominator
            if self.config.full:
                self.P -= np.outer(q,q)/denominator
                self.P[:] = (self.P+self.P.T)*.5
            else:
                self.P -= q*q/denominator
            self.learning_events += 1
            event.update(readout_updates=1, denominator=denominator)
        # Actual sensory observation advances the base prefix, not processed temporary h.
        self._append(int(target))
        if target == EOS and learn and self.config.learned:
            self._flush()
            event['encoder_steps'] = 1
        self.trace.append(event)
        self.trace = self.trace[-self.config.window:]
        return event

    def _processed_h(self,caches):
        if not caches: return self.h
        h,e,z,r,g,rh = caches[-1]
        return (1-z)*h+z*g

    def pending_gradient(self):
        accum = self.zeros()
        dh = np.zeros(self.config.width)
        for slot in reversed(self.tape):
            if slot['direct'] is not None:
                for k in accum: accum[k] += slot['direct'][k]
            dh += slot['dh']
            dh,de = self.backward_transition(slot['cache'],dh,accum)
            accum['E'][slot['token']] += de
        divisor = max(1,self.permitted)
        return {k:v/divisor for k,v in accum.items()}

    def _flush(self):
        grad = self.pending_gradient()
        norm = float(np.sqrt(sum(float((g*g).sum()) for g in grad.values())))
        factor = min(1.,1/max(norm,1e-30))
        self.steps += 1
        squared_step = 0.
        for k,g in grad.items():
            g = g*factor
            self.mom[k] *= .99
            self.mom[k] += .01*g*g
            update = self.config.rate*g/(np.sqrt(self.mom[k]/(1-.99**self.steps))+1e-8)
            self.theta[k] -= update
            squared_step += float((update*update).sum())
        self.last_update = {'unclipped_gradient_norm':norm,'clip_factor':factor,
                            'encoder_update_norm':squared_step**.5,'targets':self.permitted,
                            'retained_targets':sum(s['targets'] for s in self.tape)}
        self.tape = []
        self.permitted = 0
        self.assert_finite()

    def generate(self, cap=16):
        # Inference on a temporary working state; all learning state stays untouched.
        if type(cap) is not int or cap < 0: raise ValueError("nonnegative integer cap")
        if not self.open: raise RuntimeError("BEGIN required")
        h = self.h.copy()
        answer = []
        try:
            for _ in range(cap):
                choice,_,_ = self.predict()
                if choice == EOS:
                    return bytes(answer), 'EOS'
                answer.append(choice)
                self._append(choice,record=False)
            return bytes(answer),'TRUNCATED'
        finally:
            self.h = h

    def rest(self, seconds):
        if seconds < 0: raise ValueError('negative simulated delay')
        # Deliberate stable persistent state, no learned-state decay.

    def clone(self):
        return copy.deepcopy(self)

    def learning_digest(self):
        h = hashlib.sha256()
        for group in (self.theta,self.mom,{'V':self.V,'P':self.P}):
            for k,v in sorted(group.items()):
                h.update(k.encode()); h.update(v.tobytes())
        h.update(str((self.steps,self.learning_events)).encode())
        return h.hexdigest()

    def state_digest(self):
        digest = hashlib.sha256()
        def visit(x):
            if isinstance(x,np.ndarray):
                digest.update(str((str(x.dtype),x.shape)).encode()); digest.update(x.tobytes()); return
            if isinstance(x,Config): visit(asdict(x)); return
            if isinstance(x,dict):
                digest.update(b'dict[')
                for k,v in sorted(x.items()): visit(k); visit(v)
                digest.update(b']'); return
            if isinstance(x,(list,tuple)):
                digest.update(b'sequence[')
                for v in x: visit(v)
                digest.update(b']'); return
            raw=repr(x).encode(); digest.update(str(len(raw)).encode()+b':'+raw)
        visit(self.__dict__)
        return digest.hexdigest()

    def save(self,path):
        with open(path,'wb') as f: pickle.dump(self,f,protocol=5)

    @staticmethod
    def load(path):
        # Trusted local checkpoints only; do not load arbitrary external pickles.
        with open(path,'rb') as f: return pickle.load(f)

    def assert_finite(self):
        if not all(np.isfinite(v).all() for v in list(self.theta.values())+list(self.mom.values())+[self.V,self.P,self.h]):
            raise FloatingPointError('nonfinite state')

    def bytes_state(self):
        return {'encoder':sum(v.nbytes for v in self.theta.values()),
                'moments':sum(v.nbytes for v in self.mom.values()),
                'readout':self.V.nbytes,'confidence':self.P.nbytes,
                'tape_pickle_bytes':len(pickle.dumps(self.tape,protocol=5))}
