import sys, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]/'minifly_learned_bank_v1_20261024'))
import data, lb1
ALL=[(a,b) for a in range(10) for b in range(10)]
FE=np.array([lb1.trace_for(data.prompt(a,b)) for a,b in ALL]); FE=(FE-FE.mean(0))/FE.std(0).clip(1e-9); idx={p:i for i,p in enumerate(ALL)}
codes={
 'FE-PERM trace (actual input)': lambda a,b: FE[idx[(a,b)]],
 'one-hot (arbitrary symbols)': lambda a,b: np.r_[np.eye(10)[a],np.eye(10)[b]],
 'ordinal scalar (diagnostic prior)': lambda a,b: np.array([a/9.0,b/9.0]),
 'circular (diagnostic prior)': lambda a,b: np.array([np.cos(2*np.pi*a/10),np.sin(2*np.pi*a/10),np.cos(2*np.pi*b/10),np.sin(2*np.pi*b/10)]),
}
tasks={'PAR':lambda a,b:(a+b)%2,'UNITS':lambda a,b:(a+b)%10,'SUM':lambda a,b:a+b}
def train(Xtr,ytr,Xte,K,H=256,steps=4000,lr=0.01,wd=1e-4,seed=0):
    r=np.random.default_rng(seed); d=Xtr.shape[1]
    P={'W1':r.normal(0,1/np.sqrt(d),(d,H)),'b1':np.zeros(H),'W2':r.normal(0,1/np.sqrt(H),(H,K)),'b2':np.zeros(K)}
    M={k:np.zeros_like(v) for k,v in P.items()}; V={k:np.zeros_like(v) for k,v in P.items()}
    Y=np.eye(K)[ytr]
    for t in range(1,steps+1):
        A=Xtr@P['W1']+P['b1']; Hh=np.maximum(A,0); Z=Hh@P['W2']+P['b2']; Z-=Z.max(1,keepdims=True)
        Pr=np.exp(Z); Pr/=Pr.sum(1,keepdims=True); G=(Pr-Y)/len(Xtr)
        g={'W2':Hh.T@G+wd*P['W2'],'b2':G.sum(0)}; GH=(G@P['W2'].T)*(A>0); g['W1']=Xtr.T@GH+wd*P['W1']; g['b1']=GH.sum(0)
        for k in P:
            M[k]=0.9*M[k]+0.1*g[k]; V[k]=0.999*V[k]+0.001*g[k]**2
            P[k]-=lr*(M[k]/(1-0.9**t))/(np.sqrt(V[k]/(1-0.999**t))+1e-8)
    f=lambda X: np.argmax(np.maximum(X@P['W1']+P['b1'],0)@P['W2']+P['b2'],1)
    return f
W=['D1','D2','D3','M1','M2','M3','M4']
for t,fn in tasks.items():
    K=max(fn(*p) for p in ALL)+1
    print("task",t)
    for cname,c in codes.items():
        tr_acc=[];he_acc=[]
        for w in W:
            wd_=data.make_world(w); tr,he=wd_['train_pairs'],wd_['held_pairs']
            Xtr=np.array([c(*p) for p in tr]); ytr=np.array([fn(*p) for p in tr]); Xte=np.array([c(*p) for p in he]); yte=np.array([fn(*p) for p in he])
            f=train(Xtr,ytr,Xte,K)
            tr_acc.append(np.mean(f(Xtr)==ytr)); he_acc.append(np.mean(f(Xte)==yte))
        print("   %-36s taught %.2f  held %.2f  (per world: %s)"%(cname,np.mean(tr_acc),np.mean(he_acc)," ".join("%.2f"%x for x in he_acc)))
