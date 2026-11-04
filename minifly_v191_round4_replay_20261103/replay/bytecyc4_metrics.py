import numpy as np, json, sys
MULTI='multi' in sys.argv
GAIN='gain' in sys.argv
L=b'abcdefghijkl'; n=12
def make(seed,split,shuffle,two=False):
    r=np.random.default_rng(seed)
    order=list(range(n)) if not shuffle else list(r.permutation(n))
    pairs=[(x,y) for x in range(n) for y in range(n) if x!=y and (y-x)%n!=6]
    ans=lambda x,y: b'>' if (y-x)%n<6 else b'<'
    if split=='distance': test=[p for p in pairs if (p[1]-p[0])%n in (2,10)]
    elif split=='item': test=[p for p in pairs if 5 in p]
    else:
        idx=r.permutation(len(pairs)); test=[pairs[i] for i in idx[:len(pairs)//2]]
    train=[p for p in pairs if p not in test]
    stream=bytearray(); teach=[]
    for blk in range(400):
        for _ in range(6):   # background runs along the (possibly shuffled) cycle
            s=r.integers(n); ln=r.integers(4,11)
            run=bytes(L[order[(order.index(s)+i)%n]] for i in range(ln)) if shuffle else bytes(L[(s+i)%n] for i in range(ln))
            stream+=run+b' '; teach+=[0]*(ln+1)
        for _ in range(2):   # taught task records
            x,y=train[r.integers(len(train))]
            if two and blk>=200: rec=b'par '+bytes([L[x],L[y]])+b'='+(b's' if (x-y)%2==0 else b'd')+b'\n'
            else: rec=b'cmp '+bytes([L[x],L[y]])+b'='+ans(x,y)+b'\n'
            stream+=rec; teach+=[0]*7+[1,0]
    return bytes(stream),teach,train,test,ans
class KC:
    def __init__(s,D,N,K,seed):
        r=np.random.default_rng(seed); s.P=r.normal(size=(N,D)); s.P/=np.linalg.norm(s.P,axis=1,keepdims=True); s.K=K
    def __call__(s,z):
        u=s.P@z; ix=np.argpartition(-u,s.K)[:s.K]; return ix
def run(seed,split,feat,shuffle=False,two=False,N=2000,K=100,g=0.8,a=0.05):
    stream,teach,train,test,ans=make(seed,split,shuffle,two); REL=np.full((2,2),0.5); SEEN=set()
    M=np.zeros((256,256)); M2=np.zeros((256,256)); E=np.eye(256)
    kc=KC(512,N,K,seed+1); kc2=KC(512,N,K,seed+2); W=np.zeros((4*N,256)); B=np.ones((256,256))  # bigram counts for EOS/base
    def code(h):
        x,y=h[-3],h[-2]; op=h[-7]
        if feat=='sr':
            nz=lambda v: v/(np.linalg.norm(v)+1e-9)
            A=nz(M[x]); Bv=nz(M[y])
            if MULTI:
                i1=kc(np.concatenate([A,Bv])); i2=kc2(np.concatenate([nz(M2[x]),nz(M2[y])]))+N
                return np.concatenate([i1,i2])+(2*N if op==ord('p') else 0)
        else:
            A=E[x]; Bv=E[y]
        return kc(np.concatenate([A,Bv]))+ (N if op==ord('p') else 0)
    for t in range(1,len(stream)):
        p,c=stream[t-1],stream[t]
        if t>=2:
            q=stream[t-2]; M[q]+=a*(E[q]+g*M[p]-M[q]); M2[q]+=a*(E[q]-g*M2[p]-M2[q])  # +gamma and -gamma maps
        B[p,c]+=1
        if teach[t] and t>=3:          # environment marks answer byte -> association write
            ix=code(stream[:t])
            if MULTI and GAIN:
                op=1 if stream[t-7]==ord('p') else 0; key=(op,stream[t-3],stream[t-2])
                for pi,part in enumerate((ix[:K],ix[K:])):   # each population learns on its own error
                    s=W[part].sum(0); 
                    if key not in SEEN: REL[op,pi]+=0.1*(float(np.argmax(s)==c)-REL[op,pi])
                    e=-s; e[c]+=1; W[part]+=0.6*e/K
                SEEN.add(key)
            else:
                s=W[ix].sum(0); e=-s; e[c]+=1; W[ix]+=0.6*e/len(ix)
    def answer(x,y,opb=b'cmp '):
        h=opb+bytes([L[x],L[y]])+b'='
        ix=code(h)
        if MULTI and GAIN:
            op=1 if opb[0]==ord('p') else 0; w=REL[op]**4+1e-9; w=w/w.sum()
            sc=w[0]*W[ix[:K]].sum(0)+w[1]*W[ix[K:]].sum(0)
        else: sc=W[ix].sum(0)
        out=int(np.argmax(sc)); eos=int(np.argmax(B[out]))
        return out,eos
    def score(S):
        ok=okeos=0
        for x,y in S:
            o,e=answer(x,y); ok+=o==ans(x,y)[0]; okeos+=(o==ans(x,y)[0] and e==ord('\n'))
        return ok/len(S), okeos/len(S)
    if two:
        par=lambda S: np.mean([answer(x,y,b'par ')[0]==(ord('s') if (x-y)%2==0 else ord('d')) for x,y in S])
        if GAIN:
            def parp(S,pi):
                r=0
                for x,y in S:
                    ix=code(b'par '+bytes([L[x],L[y]])+b'='); part=ix[:K] if pi==0 else ix[K:]
                    r+=int(np.argmax(W[part].sum(0)))==(ord('s') if (x-y)%2==0 else ord('d'))
                return r/len(S)
            print('REL',REL.round(2).tolist(),'par_pop1',parp(test,0),'par_pop2',parp(test,1),file=sys.stderr)
        results=[]
        for opb in (b'cmp ', b'par '):
            outcomes=[]
            for x,y in test:
                expected=ans(x,y)[0] if opb==b'cmp ' else (ord('s') if (x-y)%2==0 else ord('d'))
                out,stop=answer(x,y,opb)
                outcomes.append([x,y,int(expected),int(out),int(stop)])
            results.append(outcomes)
        print("PAIR_METRICS",json.dumps({"seed":seed,"comparison":results[0],"parity":results[1]}),file=sys.stderr)
        return score(train),score(test),par(test)
    return score(train),score(test)
if len(sys.argv)>1:
    R=[run(s,'distance','sr',two=True) for s in range(6)]
    print(json.dumps({'cmp_train_after_par':round(np.mean([r[0][0] for r in R]),3),'cmp_heldout_after_par':round(np.mean([r[1][0] for r in R]),3),'par_heldout':round(np.mean([r[2] for r in R]),3)}))
    sys.exit()
out={}
for split in ('random','distance','item'):
    for feat,sh in (('sr',0),('onehot',0),('sr',1)):
        R=[run(s,split,feat,bool(sh)) for s in range(6)]
        out[f'{split}/{feat}{"_shuffledcycle" if sh else ""}']=dict(train=round(np.mean([r[0][0] for r in R]),3),heldout=round(np.mean([r[1][0] for r in R]),3),heldout_with_EOS=round(np.mean([r[1][1] for r in R]),3),heldout_min=round(min(r[1][0] for r in R),3))
print(json.dumps(out,indent=0))
