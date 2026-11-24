import numpy as np
from scipy.stats import spearmanr, pearsonr
exec(open("analysis/reg.py").read().split("R1=[r for r")[0])
R2=[r for r in P if r["round"]==2]; R3=[r for r in P if r["round"]==3]
old=R2+R3[:94]; new=R3[94:]
print("train (seen in earlier analysis):",len(old),"  new points:",len(new), [r["name"] for r in new])
def predict(train,test,y,keep,lam):
    X,names=design(train+test,keep); n=len(train); Y=np.array([r[y] for r in train]); Yt=np.array([r[y] for r in test])
    mu=X[:n].mean(0); ym=Y.mean(); A=X[:n]-mu
    out={}
    for nm,l in [("OLS",1e-8),("ridge",lam)]:
        b=np.linalg.lstsq(A.T@A+l*np.eye(X.shape[1]),A.T@(Y-ym),rcond=None)[0]; pr=ym+(X[n:]-mu)@b
        out[nm]=(1-((Yt-pr)**2).sum()/((Yt-Yt.mean())**2).sum(), 1-((Yt-pr)**2).sum()/((Yt-ym)**2).sum(), spearmanr(pr,Yt)[0], pearsonr(pr,Yt)[0])
    return out
for y in ["slope","s2","m4","cl"]:
    for keep,kn in [(lambda k:k not in MECH,"structure only"),(lambda k:True,"structure+mech")]:
        tr=old if kn=="structure only" else R3[:94]
        o=predict(tr,new,y,keep,lam=30.0)
        for m,(r2,r2m,sp,pe) in o.items(): print(f" y={y:5s} {kn:15s} train n={len(tr)} {m:5s}: holdout R2 {r2:+.2f} (vs train-mean {r2m:+.2f})  spearman {sp:+.2f}  pearson {pe:+.2f}")
