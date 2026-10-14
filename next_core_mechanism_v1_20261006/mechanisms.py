"""Value-address and cue-only graph mechanisms. No fixture or task imports."""
import copy,hashlib,math
from functools import lru_cache
import numpy as np
from scipy.sparse import csr_matrix
import bootstrap
import stores
from spec import DOMAIN,DOSES,EPSILONS
from io_utils import digest

def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def ro(a):a.flags.writeable=False;return a
def rank(s):return hashlib.sha256(s.encode('utf-8')).digest()
def graph_digest(b):
    return digest({'shape':list(b.shape),'data':sha(b.data),'indices':sha(b.indices),'indptr':sha(b.indptr)})

@lru_cache(maxsize=1)
def projections():
    out=[]
    for name in ('A','B'):
        a=np.zeros((44,88),dtype='<f8')
        for r in range(44):
            ids=sorted(range(88),key=lambda j:(rank(f'{DOMAIN}|REL2|{name}|{r}|{j}|select'),j))[:5]
            for j in ids:a[r,j]=(-.2 if rank(f'{DOMAIN}|REL2|{name}|{r}|{j}|sign')[-1]&1 else .2)
        out.append(ro(a))
    cycle=sorted(range(88),key=lambda j:(rank(f'{DOMAIN}|REL2|perm|{j}'),j))
    p=np.empty(88,dtype='<i4')
    for i,j in enumerate(cycle):p[j]=cycle[(i+1)%88]
    return out[0],out[1],ro(p)

def relation_q(h,first=False):
    a,b,_=projections();z=(a@h+b@h)/2 if first else (a@h)*(b@h)
    q=np.empty(88);q[::2]=np.maximum(z,0);q[1::2]=np.maximum(-z,0)
    return stores.bb.bc._norm(q)

class Counter:
    """The source's SHA256/rejection draw algorithm, using an explicit fixed key."""
    def __init__(self,j):
        self.key=rank(f'{DOMAIN}|S3rand|store|{j}');self.counter=0
    def randbelow(self,n):
        if not 0<n<=2**64:raise ValueError('counter bound')
        cutoff=(2**64//n)*n
        while True:
            if self.counter>=2**64:raise ValueError('counter exhausted')
            v=int.from_bytes(hashlib.sha256(self.key+self.counter.to_bytes(8,'big')).digest()[:8],'big')
            self.counter+=1
            if v<cutoff:return v%n

class DiagnosticUnavailable(RuntimeError):pass

class Support:
    def __init__(self,b):
        self.shape=b.shape;p,k=b.shape;c=b.tocsc();coo=b.tocoo()
        positions={(int(a),int(z)):i for i,(a,z) in enumerate(zip(coo.row,coo.col))}
        ptr=[0];pn=[];on=[];epos=[];w=[];ties=[]
        self.tie_kc=[]
        for z in range(k):
            orig=set(map(int,c.indices[c.indptr[z]:c.indptr[z+1]]))
            pool=(sorted((j for j in range(p) if j not in orig),
                   key=lambda j:(rank(f'{DOMAIN}|S3|pool|{z}|{j}'),j))[:16] if orig else [])
            for j in sorted(orig|set(pool)):
                pn.append(j);active=j in orig;on.append(active)
                pos=positions[(j,z)] if active else -1;epos.append(pos);w.append(float(b.data[pos]) if active else 0.)
                ties.append(int.from_bytes(rank(f'{DOMAIN}|S3|tie_partner|{z}|{j}')[:8],'big'))
            ptr.append(len(pn));self.tie_kc.append(int.from_bytes(rank(f'{DOMAIN}|S3|tie_kc|{z}')[:8],'big'))
        self.ptr=ro(np.array(ptr,dtype='<i8'));self.pn=ro(np.array(pn,dtype='<i4'))
        self.kc=ro(np.repeat(np.arange(k,dtype='<i4'),np.diff(self.ptr)))
        self.on=ro(np.array(on,dtype=bool));self.epos=ro(np.array(epos,dtype='<i8'))
        self.w=ro(np.array(w,dtype='<f8'));self.tie=ro(np.array(ties,dtype='<u8'))
        self.tie_kc=ro(np.array(self.tie_kc,dtype='<u8'))
        self.hash=digest({name:sha(getattr(self,name)) for name in ('ptr','pn','kc','on','epos','w','tie','tie_kc')})

_SUPPORT=None
def support(b):
    global _SUPPORT
    if _SUPPORT is None:_SUPPORT=Support(b)
    return _SUPPORT

def graph_from_slots(s,on,epos,w):
    idx=np.flatnonzero(on);idx=idx[np.lexsort((epos[idx],s.pn[idx]))]
    p,k=s.shape;ptr=np.zeros(p+1,dtype=np.int32)
    np.cumsum(np.bincount(s.pn[idx],minlength=p),out=ptr[1:])
    b=csr_matrix((w[idx].copy(),s.kc[idx].copy(),ptr),shape=s.shape)
    for a in (b.data,b.indices,b.indptr):ro(a)
    return b

class MechanismBrain(stores.NatBrain):
    def init_mechanism(self,arm,j):
        self.mechanism=arm;self.channel=j;self.hooks=0;self.hook_t=None
        self.cue_h=None;self.cue_x=None;self.last_h=None;self.last_q=None;self.last_x=None
        self.graph_hash=graph_digest(self.fly.m.B)
        self.birth_budget=np.asarray(self.fly.m.B.sum(axis=0)).ravel()
        self.birth_degree=np.bincount(self.fly.m.B.indices,minlength=self.n_native_kc)
        self.original_alpha=float(self.fly.alpha_scale)
        if arm in DOSES:self.fly.alpha_scale=self.original_alpha*DOSES[arm]
        self.sup=None;self.counter=None
        if arm in ('S3_CUE','S3_RAND'):
            self.sup=support(self.fly.m.B);s=self.sup
            self.on=s.on.copy();self.epos=s.epos.copy();self.sw=s.w.copy()
            self.U=np.zeros(self.n_native_kc);self.A=np.zeros(self.fly.m.B.shape[0]);self.nev=0.
            self.counter=Counter(j)

    def rel_features(self,t,fe=None):
        fe=self.fe.clone() if fe is None else fe
        fe.advance(t);h=fe.read();q=relation_q(h,self.mechanism=='FIRST10')
        x=np.asarray(self.model.encode_sparse(self.fly.m,q),dtype=float)
        return h,q,x

    def _features(self,t,*,fe=None,fly_m=None):
        # Preserve the source predictor features; replace only association_x.
        x,p,rows=super()._features(t,fe=fe,fly_m=fly_m)
        if self.mechanism in DOSES:
            f=self.fe if fe is None else fe
            h=f.read();q=relation_q(h,self.mechanism=='FIRST10')
            x=np.asarray(self.model.encode_sparse(self.fly.m if fly_m is None else fly_m,q),dtype=float)
            self.last_h,self.last_q,self.last_x=h.copy(),q.copy(),x.copy()
        return x,p,rows

    def association_value(self,t):
        if self.mechanism not in DOSES:return super().association_value(t)
        t=float(t)
        if not math.isfinite(t) or t<self.brain_t-1e-9 or (self.pending_t is not None and t<self.pending_t-1e-9):raise ValueError('read clock')
        _,q,x=self.rel_features(t)
        n=self.fly.clone()
        if self.pending_t is not None:
            n.rest(self.pending_t-self.brain_t);n.rest(t-self.pending_t)
        else:n.rest(t-self.brain_t)
        dx=self.common.observed_activity(n.m,np.atleast_2d(x))
        return float((n.m.expression(dx)-self.fly.reader.predict(dx)).mean(1)[0])

    def cache_cue(self,t):
        f=self.fe.clone();f.advance(t);self.cue_h=f.read().copy()
        self.cue_x=np.asarray(self.model.encode_sparse(self.fly.m,self.cue_h),dtype=float)

    def graph_state_digest(self):
        d={'B':self.graph_hash,'hooks':self.hooks,'hook_t':self.hook_t}
        if self.sup is not None:
            d.update({k:sha(getattr(self,k)) for k in ('on','epos','sw','U','A')})
            d.update(nev=self.nev,counter=self.counter.counter)
        return digest(d)

    def adapt_cue(self,t,yoked=None):
        if self.mechanism not in EPSILONS and self.sup is None:return None
        if self.cue_h is None or self.cue_x is None:raise ValueError('missing pre-feedback adaptation cache')
        old=self.fly.m.B;old_hash=self.graph_hash;self.hooks+=1
        a=self.cue_h[np.asarray(self.fly.m.pn_type_index,dtype=int)];x=self.cue_x
        info={'hooks':self.hooks,'before':old_hash,'moves':0,'support_preserved':True}
        if self.mechanism in EPSILONS:
            cols=old.indices;rows=np.repeat(np.arange(old.shape[0]),np.diff(old.indptr))
            mean=np.divide(self.birth_budget,self.birth_degree,out=np.ones_like(self.birth_budget),where=self.birth_degree>0)
            v=old.data/mean[cols]
            w=mean[cols]*np.maximum(v+EPSILONS[self.mechanism]*a[rows]*x[cols],1e-9)
            sums=np.bincount(cols,weights=w,minlength=self.n_native_kc)
            w*=self.birth_budget[cols]/sums[cols]
            new=csr_matrix((w,old.indices.copy(),old.indptr.copy()),shape=old.shape)
            delta=new.data-old.data;info.update(l1=float(abs(delta).sum()),l2=float(np.linalg.norm(delta)),max=float(abs(delta).max()))
        else:
            d=math.exp(-(0. if self.hook_t is None else t-self.hook_t)/86400.)
            self.U=self.U*d+(x>0);self.A=self.A*d+a;self.nev=self.nev*d+1
            if self.hooks%24==0:
                s=self.sup;dev=np.log((self.U/max(self.nev,1e-12)+.001)/(.05*2**(-.5)))
                eligible=np.flatnonzero(np.isin(self.fly.m.kc_side,(0,1))&(self.birth_degree>0))
                if self.mechanism=='S3_CUE':
                    order=sorted((int(k) for k in eligible if abs(dev[k])>math.log(2)),key=lambda k:(-abs(dev[k]),int(s.tie_kc[k]),k))[:64]
                    target=64
                else:
                    if yoked is None:raise ValueError('missing S3 count yoke')
                    target=int(yoked);order=list(map(int,eligible))
                moved=[]
                while order and len(moved)<target:
                    k=order.pop(0 if self.mechanism=='S3_CUE' else self.counter.randbelow(len(order)))
                    ids=np.arange(s.ptr[k],s.ptr[k+1]);active=ids[self.on[ids]];free=ids[~self.on[ids]]
                    if not len(active) or not len(free):continue
                    if self.mechanism=='S3_CUE':
                        # At most 64 KC attempts, as in the locked port.
                        low=dev[k]<0
                        oldslot=min(active,key=lambda i:((self.A[s.pn[i]] if low else -self.A[s.pn[i]]),int(s.tie[i]),int(i)))
                        newslot=min(free,key=lambda i:((-self.A[s.pn[i]] if low else self.A[s.pn[i]]),int(s.tie[i]),int(i)))
                        gain=float(self.A[s.pn[newslot]]-self.A[s.pn[oldslot]])
                        if (low and gain<=0) or (not low and gain>=0):continue
                    else:
                        oldslot=int(active[self.counter.randbelow(len(active))]);newslot=int(free[self.counter.randbelow(len(free))])
                    self.on[oldslot]=False;self.on[newslot]=True
                    self.sw[newslot]=self.sw[oldslot];self.sw[oldslot]=0.
                    self.epos[newslot]=self.epos[oldslot];self.epos[oldslot]=-1
                    moved.append([k,int(s.pn[oldslot]),int(s.pn[newslot])])
                if self.mechanism=='S3_RAND' and len(moved)!=target:raise DiagnosticUnavailable('legal random swap pool exhausted')
                info.update(moves=len(moved),swaps=moved)
            new=graph_from_slots(self.sup,self.on,self.epos,self.sw);info['support_preserved']=False
        self.hook_t=t
        for ar in (new.data,new.indices,new.indptr):ro(ar)
        self.fly.m.B=new;self.graph_hash=graph_digest(new)
        degree=np.bincount(new.indices,minlength=self.n_native_kc)
        budget=np.asarray(new.sum(axis=0)).ravel()
        if not np.array_equal(degree,self.birth_degree) or not np.allclose(budget,self.birth_budget,rtol=1e-12,atol=1e-12):raise AssertionError('graph degree/budget')
        if self.mechanism in EPSILONS and (not np.array_equal(old.indices,new.indices) or not np.array_equal(old.indptr,new.indptr)):raise AssertionError('P support changed')
        info['after']=self.graph_hash;info['state']=self.graph_state_digest()
        self.cue_h=self.cue_x=None
        return info

    def clone(self):
        out=copy.copy(self);out.fly=self.fly.clone();out.fe=self.fe.clone()
        out.w=self.w.copy();out.bias=self.bias.copy()
        for key in ('pending_x','cue_h','cue_x','last_h','last_q','last_x','on','epos','sw','U','A'):
            v=getattr(self,key,None)
            if v is not None:setattr(out,key,v.copy())
        out.counter=copy.copy(self.counter)
        for name in ('teach_logged','teach_signed'):
            out.__dict__.pop(name,None)
        if out.state_digest()!=self.state_digest():raise AssertionError('mechanism clone identity')
        return out

    def snapshot(self):
        d=super().snapshot();b=self.fly.m.B
        m={'arm':self.mechanism,'channel':self.channel,'hooks':self.hooks,'hook_t':self.hook_t,
           'alpha_scale':float(self.fly.alpha_scale),'graph_hash':self.graph_hash,
           'B_data':b.data.copy(),'B_indices':b.indices.copy(),'B_indptr':b.indptr.copy(),
           'counter':None if self.counter is None else self.counter.counter,'nev':getattr(self,'nev',None)}
        for k in ('cue_h','cue_x','last_h','last_q','last_x','on','epos','sw','U','A'):m[k]=getattr(self,k,None)
        d['mechanism_state']=m;return d

    def restore(self,d):
        s=d['mechanism_state']
        if s['arm']!=self.mechanism or s['channel']!=self.channel:raise ValueError('mechanism identity')
        super().restore(d)
        for k in ('hooks','hook_t','graph_hash'):setattr(self,k,s[k])
        b=csr_matrix((s['B_data'].copy(),s['B_indices'].copy(),s['B_indptr'].copy()),shape=self.fly.m.B.shape)
        for a in (b.data,b.indices,b.indptr):ro(a)
        if graph_digest(b)!=s['graph_hash']:raise ValueError('graph checkpoint')
        self.fly.m.B=b
        if s['alpha_scale']!=self.original_alpha*DOSES.get(self.mechanism,1.):raise ValueError('alpha dose checkpoint')
        self.fly.alpha_scale=s['alpha_scale']
        for k in ('cue_h','cue_x','last_h','last_q','last_x','on','epos','sw','U','A'):
            if s[k] is not None:setattr(self,k,s[k].copy())
        if self.counter is not None:self.counter.counter=s['counter'];self.nev=s['nev']

    def state_digest(self):
        # Parent hashes only its own fields; append every operative mechanism field.
        base=stores.F151.snapshot(self)
        keys=('w','bias','fly_fast','fly_slow','fly_adapt','fe_p','pending_x')
        d=stores.bb._digest_arrays(tuple(base[k] for k in keys),{k:v for k,v in base.items() if k not in keys})
        extra={'graph_state':self.graph_state_digest(),'mechanism':self.mechanism,'channel':self.channel,
               'alpha_scale':float(self.fly.alpha_scale)}
        for k in ('cue_h','cue_x','last_h','last_q','last_x'):
            a=getattr(self,k,None);extra[k]=None if a is None else sha(a)
        return digest([d,extra])

def convert(m,arm,j):
    m.__class__=MechanismBrain;m.init_mechanism(arm,j);return m
