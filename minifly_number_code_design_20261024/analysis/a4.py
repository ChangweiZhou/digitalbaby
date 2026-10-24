# Grokking regime check: one-hot (arbitrary symbol) inputs, units digit and full sum, AdamW with strong decoupled weight decay, long training.
import sys, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]/'minifly_learned_bank_v1_20261024'))
import data
ALL=[(a,b) for a in range(10) for b in range(10)]
def oh(a,b): return np.r_[np.eye(10)[a],np.eye(10)[b]]
def run(tr,he,fn,K,wd,steps=20000,lr=3e-3,H=256,seed=0):
    r=np.random.default_rng(seed); Xtr=np.array([oh(*p) for p in tr]); ytr=np.array([fn(*p) for p in tr]); Xte=np.array([oh(*p) for p in he]); yte=np.array([fn(*p) for p in he])
    P={'W1':r.normal(0,1/np.sqrt(20),(20,H)),'b1':np.zeros(H),'W2':r.normal(0,1/np.sqrt(H),(H,K)),'b2':np.zeros(K)}
    M={k:0*v for k,v in P.items()}; V={k:0*v for k,v in P.items()}; Y=np.eye(K)[ytr]; hist=[]
    for t in range(1,steps+1):
        A=Xtr@P['W1']+P['b1']; Hh=np.maximum(A,0); Z=Hh@P['W2']+P['b2']; Z-=Z.max(1,keepdims=True); Pr=np.exp(Z); Pr/=Pr.sum(1,keepdims=True); G=(Pr-Y)/len(Xtr)
        g={'W2':Hh.T@G,'b2':G.sum(0)}; GH=(G@P['W2'].T)*(A>0); g['W1']=Xtr.T@GH; g['b1']=GH.sum(0)
        for k in P:
            M[k]=0.9*M[k]+0.1*g[k]; V[k]=0.98*V[k]+0.02*g[k]**2
            P[k]-=lr*((M[k]/(1-0.9**t))/(np.sqrt(V[k]/(1-0.98**t))+1e-8)+wd*P[k])
        if t in (2000,20000):
            f=lambda X: np.argmax(np.maximum(X@P['W1']+P['b1'],0)@P['W2']+P['b2'],1)
            hist.append((np.mean(f(Xtr)==ytr),np.mean(f(Xte)==yte)))
    return hist
W=['D1','D2','D3','M1','M2','M3','M4']
for tname,fn,K in (('UNITS',lambda a,b:(a+b)%10,10),('SUM',lambda a,b:a+b,19)):
    for wd in (0.0,0.1,0.3,1.0):
        res=[run(data.make_world(w)['train_pairs'],data.make_world(w)['held_pairs'],fn,K,wd) for w in W]
        print("%-5s wd=%.1f  step2k taught %.2f held %.2f | step20k taught %.2f held %.2f  (held per world @20k: %s)"%(tname,wd,
              np.mean([r[0][0] for r in res]),np.mean([r[0][1] for r in res]),np.mean([r[1][0] for r in res]),np.mean([r[1][1] for r in res])," ".join("%.2f"%r[1][1] for r in res)),flush=True)
