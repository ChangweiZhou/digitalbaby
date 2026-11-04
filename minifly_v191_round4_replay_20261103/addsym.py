import numpy as np, json
from cyc import sr
def run(seed,kind,split,n=10,g=0.8,shuffle=False,epochs=200):
    M=sr(n,seed,shuffle,g=g); E=np.eye(n)
    nz=lambda v: v/np.linalg.norm(v)
    def f(x,y):
        if kind=='conj': return np.outer(nz(M[x]),nz(M[y])).ravel()
        if kind=='conj_c':   # centered SR rows (remove DC) before conjunction
            a=M[x]-M[x].mean(); b=M[y]-M[y].mean(); return np.outer(a/np.linalg.norm(a),b/np.linalg.norm(b)).ravel()
        if kind=='lookup': return np.outer(E[x],E[y]).ravel()
    pairs=[(x,y) for x in range(n) for y in range(n)]
    r=np.random.default_rng(seed)
    if split=='random': te=set(r.permutation(len(pairs))[:len(pairs)//2].tolist())
    elif split=='sum3': te={i for i,(x,y) in enumerate(pairs) if (x+y)%n==3}
    else: te={i for i,(x,y) in enumerate(pairs) if y==7}
    tr=[i for i in range(len(pairs)) if i not in te]; te=sorted(te)
    X=np.array([f(*p) for p in pairs]); Y=np.array([(x+y)%n for x,y in pairs]); W=np.zeros((X.shape[1],n))
    for ep in range(epochs):
        for i in r.permutation(tr):
            x=X[i]; e=-x@W; e[Y[i]]+=1; W+=0.5*np.outer(x,e)/(x@x+1)
    acc=lambda S: float(np.mean(np.argmax(X[S]@W,1)==Y[S]))
    return acc(tr),acc(te)
out={}
for split in ('random','sum3','operand7'):
    for kind,sh in (('lookup',0),('conj',0),('conj_c',0),('conj_c',1)):
        v=[run(s,kind,split,shuffle=bool(sh)) for s in range(6)]
        out[f'{split}/{kind}{"_shuf" if sh else ""}']=(round(np.mean([a for a,b in v]),3),round(np.mean([b for a,b in v]),3))
print(json.dumps(out,indent=0))
