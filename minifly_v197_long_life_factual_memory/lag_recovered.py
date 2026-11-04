import numpy as np
CB=np.unpackbits(np.arange(256,dtype=np.uint8)[:,None],axis=1).astype(float)*2-1
HD=256
rs=np.random.default_rng(12345); HIDX=rs.integers(0,HD,size=(1<<16,)); HSGN=rs.choice([-1.,1.],size=(1<<16,))
def lag(w):
    h=bytes(w)[-12:]; z=np.zeros(96)
    if h: z[-8*len(h):]=CB[np.frombuffer(h,np.uint8)].ravel()
    return z
def ngram(w, W=12):
    h=bytes(w)[-W:]; z=np.zeros(HD)
    for i in range(len(h)-1):
        g=(h[i]*257+h[i+1])&0xFFFF; z[HIDX[g]]+=HSGN[g]
    for i in range(len(h)-2):
        g=(h[i]*65599+h[i+1]*257+h[i+2]*7919+1)&0xFFFF; z[HIDX[g]]+=HSGN[g]
    return z
def front(kind,w):
    if kind=='lag': z=lag(w)
    elif kind=='ngram': z=ngram(w)
    else:
        a=lag(w); b=ngram(w); na=np.linalg.norm(a); nb=np.linalg.norm(b)
        z=np.concatenate([a/(na+1e-12), b/(nb+1e-12)])
    n=np.linalg.norm(z); return z/n if n>1e-12 else z
class Code:
    def __init__(s,seed,D,N,K,c=12):
        r=np.random.default_rng(seed+3604096); s.ct=np.array([r.choice(D,c,replace=False) for _ in range(N)])
        w=r.normal(size=(N,c)); s.w=w/np.linalg.norm(w,axis=1,keepdims=True); s.K=K
    def enc(s,z):
        u=np.tanh(np.einsum('ij,ij->i',s.w,z[s.ct])); ix=np.argpartition(-np.abs(u),s.K-1)[:s.K]
        v=u[ix]; n=np.linalg.norm(v); return ix,(v/n if n>1e-12 else v)
