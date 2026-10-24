# Kernel regime (frozen random features + ridge readout) with different digit codes; SR codes learned from counting streams.
import sys, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]/'minifly_learned_bank_v1_20261024'))
import data, lb1
from scipy.stats import spearmanr
ALL=[(a,b) for a in range(10) for b in range(10)]
FE=np.array([lb1.trace_for(data.prompt(a,b)) for a,b in ALL]); FE=(FE-FE.mean(0))/FE.std(0).clip(1e-9); idx={p:i for i,p in enumerate(ALL)}
# successor representation learned by TD(0) from raw counting streams "0123456789" (forward), one-hot states, local rule
def learn_sr(gamma=0.8, passes=500, lr=0.1, seed=0):
    M=np.eye(10)
    for _ in range(passes):
        seq=list(range(10))
        for s,s2 in zip(seq[:-1],seq[1:]):
            M[s]+=lr*(np.eye(10)[s]+gamma*M[s2]-M[s])
        M[9]+=lr*(np.eye(10)[9]-M[9])          # end of stream
    return M
SR=learn_sr()
SRs=(SR+SR.T)/2                                  # forward + backward exposure (symmetric context)
d=[np.linalg.norm(SRs[a]-SRs[b]) for a in range(10) for b in range(a+1,10)]; g=[b-a for a in range(10) for b in range(a+1,10)]
print("SR codes from counting streams: Spearman(code distance, |a-a'|) = %.3f"%spearmanr(d,g)[0])
codes={'FE-PERM trace':lambda a,b:FE[idx[(a,b)]],'one-hot':lambda a,b:np.r_[np.eye(10)[a],np.eye(10)[b]],
       'ordinal scalar':lambda a,b:np.array([a/9,b/9]),'thermometer':lambda a,b:np.r_[(np.arange(10)<a+1)*1.,(np.arange(10)<b+1)*1.],
       'SR learned from counting':lambda a,b:np.r_[SRs[a],SRs[b]]}
tasks={'PAR':lambda a,b:(a+b)%2,'CARRY':lambda a,b:int(a+b>=10),'UNITS':lambda a,b:(a+b)%10,'SUM':lambda a,b:a+b}
W=['D1','D2','D3','M1','M2','M3','M4']
rng=np.random.default_rng(1)
for cname,c in codes.items():
    out=[]
    for t,fn in tasks.items():
        classes=sorted({fn(*p) for p in ALL}); accs=[]
        for w in W:
            wd=data.make_world(w); tr,he=wd['train_pairs'],wd['held_pairs']
            Xtr=np.array([c(*p) for p in tr]); Xte=np.array([c(*p) for p in he])
            mu,sd=Xtr.mean(0),Xtr.std(0).clip(1e-6); Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
            R=rng.normal(0,1/np.sqrt(Xtr.shape[1]),(Xtr.shape[1],4096)); bb=rng.normal(0,1,4096)
            F=lambda X: np.maximum(X@R+bb,0)
            Ftr,Fte=F(Xtr),F(Xte); Y=np.array([[float(fn(*p)==k) for k in classes] for p in tr])
            A=np.linalg.solve(Ftr@Ftr.T+1e-2*np.eye(len(Ftr)),Y)       # kernel ridge (dual)
            pred=np.argmax(Fte@Ftr.T@A,1); accs.append(np.mean([classes[i]==fn(*p) for i,p in zip(pred,he)]))
        out.append("%s %.2f"%(t,np.mean(accs)))
    print("frozen random features, %-26s held: %s"%(cname,"  ".join(out)))
