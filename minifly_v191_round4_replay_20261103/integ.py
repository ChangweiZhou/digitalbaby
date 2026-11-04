import sys, json, numpy as np
sys.path.insert(0,'/tmp/rep'); import importlib.util
spec=importlib.util.spec_from_file_location('rep','/tmp/rep/rep.py'); R=importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
WORDS=b'the a of and to in it was he she they we on at by for with as is said then'.split()
TR=[lambda k,v:b'the code of '+k+b' is '+v, lambda k,v:k+b' -> '+v]          # trained templates
HO={'new_template':lambda k,v:b'remember '+k+b': '+v, 'question':lambda k,v:b'what is '+k+b'? '+v,
    'shifted':lambda k,v:b'the code of '+k+b'  is '+v}
def make(seed,load,reps=16,waves=4):
    r=np.random.default_rng(seed); A=b'abcdefghijklmnopqrstuvwxyz'
    keys=[];
    while len(keys)<load:
        k=bytes(r.choice(list(A),5).tolist())
        if k not in keys: keys.append(k)
    vals=[bytes([int(r.choice(list(b'0123456789')))]) for _ in keys]
    stream=bytearray(); teach=[]
    kpw=load//waves
    for w in range(waves):
        for rep in range(reps):
            for i in r.permutation(range(w*kpw,(w+1)*kpw)):
                filler=b' '.join(r.choice(WORDS,r.integers(2,6)))+b'. '
                stream+=filler; teach+=[0]*len(filler)
                rec=TR[rep%2](keys[i],vals[i]); stream+=rec+b'. '; teach+=[0]*(len(rec)-1)+[1]+[0,0]
    return bytes(stream),teach,keys,vals,kpw
CNT=np.zeros(1<<16)
def idf_front(b,update=False,W=16):
    z=np.zeros(R.HD); h=bytes(b)[-W:]
    gs=[(h[i]*257+h[i+1])&0xFFFF for i in range(len(h)-1)]+[(h[i]*65599+h[i+1]*257+h[i+2]*7919+1)&0xFFFF for i in range(len(h)-2)]
    for g in gs:
        z[R.HIDX[g]]+=R.HSGN[g]/np.sqrt(1+CNT[g]/50.)
    if update:
        for g in gs[-2:]: CNT[g]+=1
    a=R.lag(h[-12:]); na=np.linalg.norm(a); nb=np.linalg.norm(z)
    v=np.concatenate([a/(na+1e-12), z/(nb+1e-12)]); n=np.linalg.norm(v); return v/n if n>1e-12 else v
BAGR=np.random.default_rng(4242)
def bag_code(h,N,per=4,W=16):
    h=bytes(h)[-W:]
    gs=[(h[i]*257+h[i+1]) for i in range(len(h)-1)]+[(h[i]*65599+h[i+1]*257+h[i+2]*7919+1) for i in range(len(h)-2)]
    ix=np.unique(np.concatenate([ (np.array([g*2654435761+j*40503 for j in range(per)])%N) for g in gs])) if gs else np.zeros(0,int)
    v=np.ones(len(ix))/np.sqrt(max(1,len(ix))); return ix.astype(int),v
LAM=float(__import__('os').environ.get('LAM','1'))
def run(seed,load,mkind,N,K,gate):
    stream,teach,keys,vals,kpw=make(seed,load)
    Cb=R.Code(seed,96,4096,205); Wb=np.zeros((4096,256),np.float32)
    D={'bag':1,'lag':96,'both':96+R.HD,'idf':96+R.HD,'ngram':R.HD}[mkind]; Cm=None if mkind=='bag' else R.Code(seed+7,D,N,K); Wm=np.zeros((N,256),np.float32)
    cb={};cm={}; rng=np.random.default_rng(seed+9); rate=sum(teach)/len(teach)
    def eb(h):
        b=bytes(h)[-12:]
        return Cb.enc(R.front('lag',b))
    def em(h):
        if mkind=='idf': return Cm.enc(idf_front(h))
        if mkind=='bag': return bag_code(h,N)
        b=bytes(h)[-12:]
        return Cm.enc(R.front(mkind,b))
    L=0.;n=0; G=np.zeros(N)
    def SMx(x):
        z=np.exp((x-x.max())/0.1); return z/z.sum()
    TH=float(__import__('os').environ.get('TH','0.9')); HE=float(__import__('os').environ.get('HE','2'))
    def mix(sb,sm,im,vm):
        q=float(np.clip(vm@G[im],0,1)); Pm=SMx(sm); Pb=SMx(sb)
        if LAM==-3:
            Hb=-(Pb*np.log2(Pb+1e-12)).sum()
            if Pm.max()>TH and Hb>HE: return Pm,q   # familiar memory + uncertain predictor -> retrieve
        return q*Pm+(1-q)*Pb, q
    for t in range(1,len(stream)):
        h=stream[max(0,t-16):t]; y=stream[t]
        if mkind=='idf': idf_front(h,update=True)
        ib,vb=eb(h); im,vm=em(h); sb=vb@Wb[ib]; sm=vm@Wm[im]; s=(sm if sm.max()>0.5 else sb) if LAM<0 else sb+LAM*sm
        if LAM in (-2,-3):
            P,q=mix(sb,sm,im,vm); L+=-np.log2(max(P[y],1e-12)); n+=1
            G[im]+=0.6*(teach[t]-q)*vm/max(vm@vm,1e-12)
        else:
            L+=R.m.loss(s,y); n+=1
        e=-sb; e[y]+=1; Wb[ib]+=(0.6*np.outer(vb/(vb@vb),e)).astype(np.float32)
        g = teach[t] if gate=='teach' else (rng.random()<rate if gate=='rand' else 1)
        if g:
            e=-sm; e[y]+=1; Wm[im]+=(0.6*np.outer(vm/max(vm@vm,1e-12),e)).astype(np.float32)
    def ans(prefix):
        ib,vb=eb(prefix[-12:]); im,vm=em(prefix[-16:]); sb=vb@Wb[ib]; sm=vm@Wm[im]
        if LAM in (-2,-3): return int(np.argmax(mix(sb,sm,im,vm)[0]))
        return int(np.argmax((sm if sm.max()>0.5 else sb) if LAM<0 else sb+LAM*sm))
    def ansm(prefix):
        im,vm=em(prefix[-16:]); return int(np.argmax(vm@Wm[im]))
    out={}
    old=range(kpw)  # wave A
    fams={'trained':TR,**{k:[v] for k,v in HO.items()}}
    for f,ts in fams.items():
        out[f]=round(float(np.mean([ans(T(keys[i],b''))==vals[i][0] for i in old for T in ts])),3)
        out[f+'_memonly']=round(float(np.mean([ansm(T(keys[i],b''))==vals[i][0] for i in old for T in ts])),3)
    return {'seed':seed,'load':load,'mem':mkind,'gate':gate,'whole_bpb':round(L/n,3),'waveA_after_D':out}
a=sys.argv; print(json.dumps(run(int(a[1]),int(a[2]),a[3],int(a[4]),int(a[5]),a[6])),flush=True)
