import sys, json, importlib.util, hashlib, numpy as np
spec=importlib.util.spec_from_file_location('v39','/tmp/v39/elm_fly_capacity_v39.py'); m=importlib.util.module_from_spec(spec); sys.modules['v39']=m; spec.loader.exec_module(m)
V38=m.V38
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
def run(seed,load,kind,N,K):
    st=m.make_stream(seed,keys_per_wave=load//4)
    D={'lag':96,'ngram':HD,'both':96+HD}[kind]
    C=Code(seed,D,N,K); W=np.zeros((N,256),np.float32)
    cache={}
    def enc(w):
        b=bytes(w)[-12:]
        if b not in cache: cache[b]=C.enc(front(kind,b))
        return cache[b]
    def scores(w):
        ix,v=enc(w); return v@W[ix]
    hist=bytearray(); tot=0; L=0.
    keys=st['keys']; lab=st['labels']; un=st['untouched']
    P=m.V38.PREFIX
    fam={'trained':lambda k:[P+b'@'+k+b':',P+b'#'+k+b'='],
         'heldout_render':lambda k:[P+b'?'+k+b'>'],
         'shift1':lambda k:[P+b'@'+k+b' :'],
         'shift2':lambda k:[P+b'@'+k+b'  :'],
         'wrapper':lambda k:[b'<<'+b'@'+k+b':', b'xyz@'+k+b':'],
         'prefix_shift':lambda k:[b' '+k+b':' ,b'@@'+k+b'::']}
    def probe():
        out={}
        for f,g in fam.items():
            ok=n=0
            for i in un:
                for c in g(keys[i]): ok+=int(np.argmax(scores(c))==lab[i]); n+=1
            out[f]=round(ok/n,3)
        return out
    res={}
    nrow=len(st['rows']); per=nrow//4
    cps={cp['record_count']:cp['name'] for cp in st['checkpoints'] if cp['type']=='wave'}
    for ri,row in enumerate(st['rows'],1):
        for y in row['raw']:
            ix,v=enc(hist); used=v@W[ix]
            p=m.softmax(used) if hasattr(m,'softmax') else None
            L+=m.loss(used,y); tot+=1
            e=-used; e[y]+=1.; vv=float(v@v)
            if vv>1e-15: W[ix]+=(m.RATE*np.outer(v/vv,e)).astype(np.float32)
            hist.append(y); del hist[:-12]
        if ri in cps and cps[ri] in ('after_A','after_D'): res[cps[ri]]=probe()
    # overlap diagnostics
    def jac(a,b): A=set(enc(a)[0]);B=set(enc(b)[0]); return len(A&B)/len(A|B)
    ov={f:round(float(np.mean([jac(P+b'@'+keys[i]+b':',fam[f](keys[i])[0]) for i in un])),3) for f in fam if f!='trained'}
    ov['diffkey']=round(float(np.mean([jac(P+b'@'+keys[un[j]]+b':',P+b'@'+keys[un[j-1]]+b':') for j in range(1,len(un))])),3)
    return {'seed':seed,'load':load,'kind':kind,'N':N,'K':K,'whole_bpb':round(L/tot,3),**res,'jaccard_vs_trained':ov}
if __name__=='__main__':
    a=sys.argv; print(json.dumps(run(int(a[1]),int(a[2]),a[3],int(a[4]),int(a[5]))),flush=True)
