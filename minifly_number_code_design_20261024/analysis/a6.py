import sys, numpy as np
HERE=__import__('pathlib').Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parents[1]/'minifly_learned_bank_v1_20261024')); sys.path.insert(0, str(HERE))
import data
from a5helpers import SRs
src=open(HERE/'a3.py').read()
exec(src[src.index('def train'):src.index("W=['D1'")])
ALL=[(a,b) for a in range(10) for b in range(10)]
codes={'SR learned from counting':lambda a,b:np.r_[SRs[a],SRs[b]],'one-hot':lambda a,b:np.r_[np.eye(10)[a],np.eye(10)[b]]}
tasks={'PAR':lambda a,b:(a+b)%2,'UNITS':lambda a,b:(a+b)%10,'SUM':lambda a,b:a+b}
W=['D1','D2','D3','M1','M2','M3','M4']
for t,fn in tasks.items():
    K=max(fn(*p) for p in ALL)+1
    for cname,c in codes.items():
        he_acc=[]
        for w in W:
            wd_=data.make_world(w); tr,he=wd_['train_pairs'],wd_['held_pairs']
            Xtr=np.array([c(*p) for p in tr]); mu,sd=Xtr.mean(0),Xtr.std(0).clip(1e-6); Xtr=(Xtr-mu)/sd
            Xte=(np.array([c(*p) for p in he])-mu)/sd
            f=train(Xtr,np.array([fn(*p) for p in tr]),Xte,K)
            he_acc.append(np.mean(f(Xte)==np.array([fn(*p) for p in he])))
        print("BP %-5s %-26s held %.2f  (%s)"%(t,cname,np.mean(he_acc)," ".join("%.2f"%x for x in he_acc)),flush=True)
