# Root-cause step C: is the rule unreachable (search) rather than unrepresentable or dispreferred?
# H=1 network h=relu(u_a+v_b+c) -> 19-way softmax. Compare: random init (many restarts) vs init near the rule
# (u_a = alpha*a + noise), and init with only the ORDER known (u_a = alpha*rank(a) of a random monotone warp).
import sys, numpy as np
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[1]/'minifly_learned_bank_v1_20261024')); import data
W=['RC-D%d'%i for i in range(1,9)]
def oh(a,b): return np.r_[np.eye(10)[a],np.eye(10)[b]]
def fit(Xtr,ytr,Xte,yte,mode,init,seed,H=1,steps=8000,lr=0.05):
    r=np.random.default_rng(seed); K=19
    W1=r.normal(0,1/np.sqrt(20),(20,H)); b1=np.zeros(H)
    if init=='near_rule':   W1[:10,0]=0.3*np.arange(10)+r.normal(0,0.3,10); W1[10:,0]=0.3*np.arange(10)+r.normal(0,0.3,10); b1[:]=0.1
    if init=='order_only':  warp=np.sort(r.uniform(0,3,10)); W1[:10,0]=warp+r.normal(0,0.05,10); W1[10:,0]=np.sort(r.uniform(0,3,10)); b1[:]=0.1
    W2=r.normal(0,1/np.sqrt(H),(H,K)); b2=np.zeros(K); G=r.normal(0,1/np.sqrt(K),(K,H)); Y=np.eye(K)[ytr]; mom=[0,0,0,0]
    for t in range(steps):
        A=Xtr@W1+b1; Hh=np.maximum(A,0); Z=Hh@W2+b2; Z-=Z.max(1,keepdims=True); P=np.exp(Z); P/=P.sum(1,keepdims=True); E=(P-Y)/len(Xtr)
        back=E@W2.T if mode=='BP' else E@G; D=back*(A>0)
        for i,(p,g) in enumerate(((W1,Xtr.T@D),(b1,D.sum(0)),(W2,Hh.T@E),(b2,E.sum(0)))): mom[i]=0.9*mom[i]+g; p-=lr*mom[i]
    f=lambda X: np.argmax(np.maximum(X@W1+b1,0)@W2+b2,1)
    return np.mean(f(Xtr)==ytr), np.mean(f(Xte)==yte)
for mode in ('BP','DFA'):
    for init in ('random','order_only','near_rule'):
        rows=[]
        for w in W:
            wd=data.make_world(w); tr,he=wd['train_pairs'],wd['held_pairs']
            Xtr=np.array([oh(*p) for p in tr]); ytr=np.array([a+b for a,b in tr]); Xte=np.array([oh(*p) for p in he]); yte=np.array([a+b for a,b in he])
            rs=[fit(Xtr,ytr,Xte,yte,mode,init,s) for s in range(10 if init=='random' else 3)]
            rows.append((np.mean([x[0] for x in rs]),np.mean([x[1] for x in rs]),max(x[1] for x in rs), np.mean([x[0]>=0.99 for x in rs])))
        print(f"{mode} init={init:<10s} taught {np.mean([r[0] for r in rows]):.2f}  held {np.mean([r[1] for r in rows]):.2f}  (best restart held {np.mean([r[2] for r in rows]):.2f}; runs reaching taught 1.0: {np.mean([r[3] for r in rows]):.2f})",flush=True)
