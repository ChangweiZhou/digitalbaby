# Root-cause step B: capacity / preference. One-hot symbol inputs (the information content of the FE-PERM trace),
# one hidden ReLU layer of width H, 19-class sum output. BP = exact-gradient DIAGNOSTIC only; DFA = local rule
# (fixed random feedback for the hidden layer, delta rule for the readout). True labels vs orbit-consistent shuffled labels.
import sys, numpy as np
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[1]/'minifly_learned_bank_v1_20261024')); import data
W=['RC-D%d'%i for i in range(1,9)]
def oh(a,b): return np.r_[np.eye(10)[a],np.eye(10)[b]]
def shuf_labels(wd,seed):
    r=np.random.default_rng(seed); orb=wd['train']; lab=[a+b for a,b in orb]
    for _ in range(1000):
        p=r.permutation(len(orb)); new=[lab[i] for i in p]
        if np.mean([x==y for x,y in zip(new,lab)])<0.15: break
    m={o:new[i] for i,o in enumerate(orb)}; return {(a,b):m[(min(a,b),max(a,b))] for a,b in wd['train_pairs']}
def fit(Xtr,ytr,Xte,yte,H,mode,seed,steps=6000,lr=0.05):
    r=np.random.default_rng(seed); K=19
    W1=r.normal(0,1/np.sqrt(20),(20,H)); b1=np.zeros(H); W2=r.normal(0,1/np.sqrt(H),(H,K)); b2=np.zeros(K); G=r.normal(0,1/np.sqrt(K),(K,H))
    Y=np.eye(K)[ytr]; mom=[0,0,0,0]; hist={}
    for t in range(1,steps+1):
        A=Xtr@W1+b1; Hh=np.maximum(A,0); Z=Hh@W2+b2; Z-=Z.max(1,keepdims=True); P=np.exp(Z); P/=P.sum(1,keepdims=True); E=(P-Y)/len(Xtr)
        dW2=Hh.T@E; db2=E.sum(0)
        back = E@W2.T if mode=='BP' else E@G
        D=back*(A>0); dW1=Xtr.T@D; db1=D.sum(0)
        for i,(p,g) in enumerate(((W1,dW1),(b1,db1),(W2,dW2),(b2,db2))):
            mom[i]=0.9*mom[i]+g; p-=lr*mom[i]
        if t in (300,1000,6000):
            f=lambda X: np.argmax(np.maximum(X@W1+b1,0)@W2+b2,1)
            hist[t]=(np.mean(f(Xtr)==ytr), np.mean(f(Xte)==yte))
    return hist
for mode in ('BP','DFA'):
    for H in (1,2,3,4,8,32,256):
        res={'true':[], 'shuf':[]}
        for w in W:
            wd=data.make_world(w); tr,he=wd['train_pairs'],wd['held_pairs']
            Xtr=np.array([oh(*p) for p in tr]); Xte=np.array([oh(*p) for p in he]); yte=np.array([a+b for a,b in he])
            sh=shuf_labels(wd,data.seed_of('rc-shuf:'+w))
            for s in range(3):
                res['true'].append(fit(Xtr,np.array([a+b for a,b in tr]),Xte,yte,H,mode,s))
                res['shuf'].append(fit(Xtr,np.array([sh[p] for p in tr]),Xte,yte,H,mode,s))
        g=lambda key,t,i: np.mean([h[t][i] for h in res[key]])
        print(f"{mode} H={H:<4d} TRUE taught@300/1000/6000 {g('true',300,0):.2f}/{g('true',1000,0):.2f}/{g('true',6000,0):.2f}  held@6000 {g('true',6000,1):.2f} | "
              f"SHUF taught@300/1000/6000 {g('shuf',300,0):.2f}/{g('shuf',1000,0):.2f}/{g('shuf',6000,0):.2f}", flush=True)
