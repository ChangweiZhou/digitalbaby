# Background analysis for the V146 memo (section 1): exact-gradient evaluator MLP, NOT part of the V146 learner (which uses local rules only).
# Does grokking appear as the symbol set grows? mod-p addition, one-hot inputs, 1-hidden-layer ReLU MLP, AdamW (wd=1), full batch.
import numpy as np, sys, time
def run(p, frac, steps=20000, H=256, lr=1e-3, wd=1.0, seed=0):
    r=np.random.default_rng(seed); pairs=[(a,b) for a in range(p) for b in range(p)]; perm=r.permutation(len(pairs))
    ntr=int(frac*len(pairs)); tr=[pairs[i] for i in perm[:ntr]]; te=[pairs[i] for i in perm[ntr:]]
    X=lambda S: np.concatenate([np.eye(p)[[a for a,_ in S]],np.eye(p)[[b for _,b in S]]],1)
    Xtr,Xte=X(tr),X(te); ytr=np.array([(a+b)%p for a,b in tr]); yte=np.array([(a+b)%p for a,b in te]); Y=np.eye(p)[ytr]
    P={'W1':r.normal(0,1/np.sqrt(2*p),(2*p,H)),'b1':np.zeros(H),'W2':r.normal(0,1/np.sqrt(H),(H,p)),'b2':np.zeros(p)}
    M={k:0*v for k,v in P.items()}; V={k:0*v for k,v in P.items()}; out=[]
    for t in range(1,steps+1):
        A=Xtr@P['W1']+P['b1']; Hh=np.maximum(A,0); Z=Hh@P['W2']+P['b2']; Z-=Z.max(1,keepdims=True); Pr=np.exp(Z); Pr/=Pr.sum(1,keepdims=True); G=(Pr-Y)/len(Xtr)
        g={'W2':Hh.T@G,'b2':G.sum(0)}; GH=(G@P['W2'].T)*(A>0); g['W1']=Xtr.T@GH; g['b1']=GH.sum(0)
        for k in P:
            M[k]=0.9*M[k]+0.1*g[k]; V[k]=0.98*V[k]+0.02*g[k]**2
            P[k]-=lr*((M[k]/(1-0.9**t))/(np.sqrt(V[k]/(1-0.98**t))+1e-8)+wd*P[k])
        if t in (1000,5000,20000):
            f=lambda X: np.argmax(np.maximum(X@P['W1']+P['b1'],0)@P['W2']+P['b2'],1)
            out.append((t,round(float(np.mean(f(Xtr)==ytr)),2),round(float(np.mean(f(Xte)==yte)),2)))
    return ntr,out
for p in (10,23,47):
    for frac in (0.5,0.75):
        t0=time.time(); ntr,out=run(p,frac)
        print("p=%d frac=%.2f train pairs=%d  (step, taught, held): %s  [%.0fs]"%(p,frac,ntr,out,time.time()-t0),flush=True)
