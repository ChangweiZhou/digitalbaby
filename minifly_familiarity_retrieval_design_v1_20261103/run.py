"""Fixed local hashed-byte retrieval development experiment."""
import hashlib,json,math,random,resource,sys,time
from pathlib import Path
import numpy as np

START=time.monotonic()
def guard():
    assert time.process_time()<900 and time.monotonic()-START<1800
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2**30
def ids(data,n,count):
    return [int.from_bytes(hashlib.blake2b(data+bytes([i]),digest_size=8).digest(),'big')%n for i in range(count)]
def soft(x):
    z=np.exp(x-x.max());return z/z.sum()
def bag(h):
    return {h[i:i+k] for k in (2,3) for i in range(len(h)-k+1)}
class Model:
    def __init__(self):
        self.p=np.zeros((2048,257),np.float32)
        self.m=np.zeros((4096,257),np.float32)
        self.g=np.zeros((3,2048+5),np.float32)
        self.s=np.zeros(4096,np.uint16)
    def read(self,h):
        h=h[-24:]
        p=sorted(set(ids(b'p'+h[-k:],2048,2)[0] for k in (0,1,2,3,4,8) if k))
        grams=bag(h)
        m=sorted({i for gram in grams for i in ids(b'm'+gram,4096,4)}) or [0]
        cue=sorted({i for k in (1,2,4,8) for i in ids(b'g'+h[-k:],2048,2)})+[2048]
        features=np.zeros((3,4))
        for j,k in enumerate((8,24)):
            support=np.array([ids(b's'+gram,4096,1)[0] for gram in sorted(bag(h[-k:]))],dtype=int)
            for arm,index in ((1,(support+1777)%4096),(2,support)):
                if len(index):
                    v=self.s[index]
                    features[arm,j*2:j*2+2]=[(v>0).mean(),np.minimum(v,8).mean()/8]
        pl=self.p[p].mean(0);ml=self.m[m].mean(0)
        pp,mp=soft(pl),soft(ml)
        q=1/(1+np.exp(-np.clip(self.g[:,cue].sum(1)+(self.g[:,2049:]*features).sum(1),-30,30)))
        return p,m,cue,grams,features,q,pp,mp,soft(pl+ml)
    def observe(self,h,y,mark):
        p,m,cue,grams,f,q,pp,mp,_=self.read(h)
        e=-pp;e[y]+=1;self.p[p]+=e
        if mark:
            e=-mp;e[y]+=1;self.m[m]+=4*e
            if y!=256:
                for gram in sorted(grams):
                    k=ids(b's'+gram,4096,1)[0]
                    self.s[k]=min(65535,int(self.s[k])+1)
        error=.1*(int(mark)-q)
        self.g[:,cue]+=error[:,None]/len(cue)
        self.g[:,2049:]+=error[:,None]*f
    def distributions(self,h):
        *_,q,p,m,total=self.read(h)
        return [p,total,*[(1-v)*p+v*m for v in q],m]
    def teach(self,r,mark):
        for i,y in enumerate([*r,256]):self.observe(r[:i],y,i>=mark)

def prefix(k,w):
    return (b'code '+k+b' is ',k+b' -> ',b'remember '+k+b': ',b'what is '+k+b'? ')[w]

def evaluate(model,keys,values):
    scores=np.zeros((6,2),int);loss=np.zeros((6,2)); counts=[0,0]
    for key,y in zip(keys,values):
        for col,w in enumerate((2,3)):
            h=prefix(key,w)
            first=model.distributions(h)
            for a,p in enumerate(first):
                out=bytearray()
                for _ in range(8):
                    token=int(model.distributions(h+out)[a].argmax()) if out else int(p.argmax())
                    if token==256:
                        scores[a,col]+=out==bytes([y]);break
                    out.append(token)
            record=h+bytes([y])
            for i,target in enumerate(record):
                loss[:,0]-=np.log2([max(float(p[target]),1e-30) for p in model.distributions(record[:i])])
                counts[0]+=1
        for lead,tail in ((b'today ',b' was seen.'),(b'notes ',b' list it.'),
                          (b'a line ',b' continues.'),(b'we saw ',b' nearby.'),
                          (b'some ',b' occurs.'),(b'entry ',b' is ordinary.'),
                          (b'here ',b' appears.'),(b'next ',b' was printed.')):
            prose=lead+key+tail
            for i,y in enumerate(prose):
                loss[:,1]-=np.log2([max(float(p[y]),1e-30) for p in model.distributions(prose[:i])])
                counts[1]+=1
    return dict(exact=scores.tolist(),loss=loss.tolist(),targets=counts)
def world(i,small=False):
    seed=int.from_bytes(hashlib.sha256(f'MINIFLY|FAM-RETRIEVAL-V1|WORLD|{i}'.encode()).digest()[:8],'big')
    rng=random.Random(seed)
    n=8 if small else 128
    keys=[bytes(rng.choices(b'abcdefghijklmnopqrstuvwxyz',k=6)) for _ in range(n)]
    assert len(set(keys))==n
    values=[ord('A')+j%16 for j in range(n)];rng.shuffle(values)
    model=Model()
    def phase(indices,labels):
        order=list(indices)
        for rep in range(4):
            rng.shuffle(order)
            for offset,j in enumerate(order):
                h=prefix(keys[j],rep%2);model.teach(h+bytes([labels[j]]),len(h))
                if offset%4==3:model.teach(b'in prose '+keys[j]+b' was mentioned.',999)
        guard()
    phase(range(n//4),values)
    a=evaluate(model,keys[:n//4],values[:n//4])
    phase(range(n//4,n),values)
    b=evaluate(model,keys[:n//4],values[:n//4])
    # In-memory checkpoint covers the only four mutable arrays.
    state=[getattr(model,k).copy() for k in ('p','m','g','s')]
    edited=values.copy()
    for j in range(min(8,n//4)):edited[j]=ord('A')+(values[j]-ord('A')+1)%16
    fork={}
    fork_seed=rng.getstate()
    for name,labels in [('EDIT',edited),('NOEDIT',values)]:
        for k,v in zip(('p','m','g','s'),state):setattr(model,k,v.copy())
        rng.setstate(fork_seed)
        phase(range(min(8,n//4)),labels)
        fork[name]=evaluate(model,keys[:n//4],edited[:n//4])
    return {'A':a,'B':b,'fork':fork}
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='math':
        model=Model()
        p=model.distributions(b'abc')
        assert all(abs(float(x.sum())-1)<1e-6 for x in p)
        model.observe(b'abc',33,False)
        assert not model.m.any() and not model.s.any()
        model.observe(b'abc',33,True)
        assert model.m.any() and model.s.any()
        result={'checks':'normalization and post-prediction marker isolation'}
    elif mode=='fixture':result=world(0,True)
    elif mode=='dev':
        result={}
        for i in range(4):
            result[str(i)]=world(i)
            Path('committed_'+str(i)+'.json').write_text(json.dumps(result[str(i)],allow_nan=False)+'\n')
            print('committed world',i,flush=True)
        guard()
    else:raise ValueError(mode)
    Path(mode+'.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
