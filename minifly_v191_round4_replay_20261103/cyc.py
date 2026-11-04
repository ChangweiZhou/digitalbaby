import numpy as np, json, sys
def sr(n,seed,shuffle=False,steps=20000,a=0.1,g=0.8):
    r=np.random.default_rng(seed); order=np.arange(n)
    if shuffle: order=r.permutation(n)   # stream follows a DIFFERENT cycle -> uninformative about task cycle
    nxt={order[i]:order[(i+1)%n] for i in range(n)}
    M=np.zeros((n,n)); x=r.integers(n)
    for t in range(steps):
        if r.random()<0.15: x=r.integers(n)  # new sequence start
        y=nxt[x]; M[x]+=a*(np.eye(n)[x]+g*M[y]-M[x]); x=y   # local TD(0) successor representation
    return M
def feats(kind,M,n,seed,KC=2000,k=100):
    E=np.eye(n)
    r=np.random.default_rng(seed+1); P=r.normal(size=(KC,2*n)); P/=np.linalg.norm(P,axis=1,keepdims=True)
    def f(x,y):
        if kind=='lookup': return np.outer(E[x],E[y]).ravel()
        if kind=='rank': return np.concatenate([E[x],-E[y]])
        A,B=M[x]/np.linalg.norm(M[x]),M[y]/np.linalg.norm(M[y])
        if kind=='concat': return np.concatenate([A,B])
        if kind=='conj': return np.outer(A,B).ravel()
        if kind=='kc':
            u=P@np.concatenate([A,B]); z=np.zeros(KC); ix=np.argpartition(-u,k)[:k]; z[ix]=1; return z  # fixed sparse expansion (nonlinear mixing)
    return f
def task(n,rel):
    pairs=[];lab=[]
    for x in range(n):
        for y in range(n):
            if x==y: continue
            if rel=='cyclic':
                d=(y-x)%n
                if d==n//2: continue
                pairs.append((x,y)); lab.append(1. if d<n//2 else -1.)
            else: pairs.append((x,y)); lab.append(1. if x<y else -1.)
    return pairs,np.array(lab)
def run(rel,kind,seed,n=12,frac=0.5,epochs=50,shuffle=False):
    M=sr(n,seed,shuffle) ; f=feats(kind,M,n,seed)
    pairs,lab=task(n,rel); r=np.random.default_rng(seed+2); idx=r.permutation(len(pairs)); ntr=int(frac*len(pairs))
    tr,te=idx[:ntr],idx[ntr:]
    X=np.array([f(*p) for p in pairs]); w=np.zeros(X.shape[1])
    for ep in range(epochs):                      # online normalized LMS (local, no backprop)
        for i in r.permutation(tr):
            x=X[i]; w+=0.5*(lab[i]-w@x)*x/(x@x+1)
    acc=lambda S: float(np.mean(np.sign(X[S]@w)==lab[S]))
    return acc(tr),acc(te)
if __name__=='__main__':
    out={}
    for rel in ('linear','cyclic'):
        for kind,sh in (('lookup',0),('rank',0),('concat',0),('conj',0),('kc',0),('conj',1),('kc',1)):
            R=[run(rel,kind,s,shuffle=bool(sh)) for s in range(10)]
            out[f'{rel}/{kind}{"_shuffledSR" if sh else ""}']=dict(train=round(np.mean([a for a,b in R]),3),heldout=round(np.mean([b for a,b in R]),3),heldout_min=round(min(b for a,b in R),3))
    print(json.dumps(out,indent=1))
