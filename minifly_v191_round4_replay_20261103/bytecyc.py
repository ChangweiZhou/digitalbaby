import numpy as np, json, sys
L=b'abcdefghijkl'; n=12
def make(seed,split,shuffle):
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
            x,y=train[r.integers(len(train))]; rec=b'cmp '+bytes([L[x],L[y]])+b'='+ans(x,y)+b'\n'
            stream+=rec; teach+=[0]*7+[1,0]
    return bytes(stream),teach,train,test,ans
class KC:
    def __init__(s,D,N,K,seed):
        r=np.random.default_rng(seed); s.P=r.normal(size=(N,D)); s.P/=np.linalg.norm(s.P,axis=1,keepdims=True); s.K=K
    def __call__(s,z):
        u=s.P@z; ix=np.argpartition(-u,s.K)[:s.K]; return ix
def run(seed,split,feat,shuffle=False,N=2000,K=100,g=0.8,a=0.05):
    stream,teach,train,test,ans=make(seed,split,shuffle)
    M=np.zeros((256,256)); E=np.eye(256)
    kc=KC(512,N,K,seed+1); W=np.zeros((N,256)); B=np.ones((256,256))  # bigram counts for EOS/base
    def code(h):
        x,y=h[-3],h[-2]
        if feat=='sr':
            A=M[x]/ (np.linalg.norm(M[x])+1e-9); Bv=M[y]/(np.linalg.norm(M[y])+1e-9)
        else: A=E[x]; Bv=E[y]
        return kc(np.concatenate([A,Bv]))
    for t in range(1,len(stream)):
        p,c=stream[t-1],stream[t]
        if t>=2: M[stream[t-2]]+=a*(E[stream[t-2]]+g*M[p]-M[stream[t-2]])  # local TD(0) successor map
        B[p,c]+=1
        if teach[t] and t>=3:          # environment marks answer byte -> association write
            ix=code(stream[:t]); s=W[ix].sum(0); e=-s; e[c]+=1; W[ix]+=0.6*e/K
    def answer(x,y):
        h=b'cmp '+bytes([L[x],L[y]])+b'='
        ix=code(h); out=int(np.argmax(W[ix].sum(0))); eos=int(np.argmax(B[out]))
        return out,eos
    def score(S):
        ok=okeos=0
        for x,y in S:
            o,e=answer(x,y); ok+=o==ans(x,y)[0]; okeos+=(o==ans(x,y)[0] and e==ord('\n'))
        return ok/len(S), okeos/len(S)
    return score(train),score(test)
out={}
for split in ('random','distance','item'):
    for feat,sh in (('sr',0),('onehot',0),('sr',1)):
        R=[run(s,split,feat,bool(sh)) for s in range(6)]
        out[f'{split}/{feat}{"_shuffledcycle" if sh else ""}']=dict(train=round(np.mean([r[0][0] for r in R]),3),heldout=round(np.mean([r[1][0] for r in R]),3),heldout_with_EOS=round(np.mean([r[1][1] for r in R]),3),heldout_min=round(min(r[1][0] for r in R),3))
print(json.dumps(out,indent=0))
