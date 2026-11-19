"""Part 4, second check: do the inputs of the drivers of END (10) and '=' (61) tell that '=' came 0, 1, 2 or 3 bytes back?
Ridge (1e-2 n) on the driver's own inputs (standardised over the fitting part, first three quarters), fitted to the indicator
"byte t-L is '='"; area under the ROC curve on the last quarter.      python3 why_eq.py <seed>"""
import sys, json, numpy as np
sys.path.insert(0,"/home/user/tissue_deps"); sys.path.insert(0,"/home/user/tissue_check")
import tissue_step1 as ts, my_tissue as mt
seed=int(sys.argv[1]); NB=65536; B=ts.birth(seed); _,members,_=mt.long_axons(B,seed); b,_=ts.stream(seed,NB); b=b.astype(np.int64)
isfb=np.asarray(B["kind"])==ts.FBK; drivers=list(B["driver"])
near=set(drivers)
for i in drivers: near.update(int(m) for m in members[i] if m<ts.N)
near=np.array(sorted(i for i in near if not isfb[i])); col={c:k for k,c in enumerate(near)}
X=np.fromfile(f"out/OE_L8_{seed}.rec",dtype=np.float32).reshape(-1,ts.N)[:NB][:,near].astype(np.float64)
M=NB-1; nf=(3*M)//4; sd=X[:nf].std(0); X=np.where(sd>0,X/np.where(sd>0,sd,1),0.0)
res={}
for v in (10,61):
    own=sorted({col[c] for c in [drivers[v]]+[int(m) for m in members[drivers[v]] if m<ts.N] if c in col}); Xv=X[:M,own]
    for L in (0,1,2,3):
        t=np.zeros(M); t[L:]=(b[:M-L]==61)
        mu=Xv[:nf].mean(0); Xc=Xv[:nf]-mu; G=Xc.T@Xc; G[np.diag_indices_from(G)]+=1e-2*nf
        s=(Xv[nf:]-mu)@np.linalg.solve(G,Xc.T@(t[:nf]-t[:nf].mean())); tt=t[nf:]; pos=s[tt==1]; neg=s[tt==0]
        res[f"driver of {'END' if v==10 else '='}, '=' {L} back"]=round(float(np.searchsorted(np.sort(neg),pos).sum()/(len(pos)*len(neg))),3)
json.dump(res,open(f"out/why_eq_L8_{seed}.json","w"),indent=1); print(seed,res)
