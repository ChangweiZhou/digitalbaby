import sys, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]/'minifly_learned_bank_v1_20261024'))
import data, lb1
ALL=[(a,b) for a in range(10) for b in range(10)]
X=np.array([lb1.trace_for(data.prompt(a,b)) for a,b in ALL])
Xc=X-X.mean(0)
s=np.linalg.svd(Xc,compute_uv=False)
print("FE trace: centred rank (sv>1e-9*max):", int((s>1e-9*s[0]).sum()), "top sv ratio 19/20:", s[18]/s[0], s[19]/s[0] if len(s)>19 else None)
# is the trace exactly affine in one-hot(a), one-hot(b)?
OH=np.array([np.r_[np.eye(10)[a],np.eye(10)[b],1] for a,b in ALL])
coef,res,rk,_=np.linalg.lstsq(OH,X,rcond=None)
print("affine fit residual (rel):", np.linalg.norm(OH@coef-X)/np.linalg.norm(Xc))
# class matrices
units=np.array([[(a+b)%10 for b in range(10)] for a in range(10)])
par=np.array([[(a+b)%2 for b in range(10)] for a in range(10)])
carry=np.array([[int(a+b>=10) for b in range(10)] for a in range(10)])
cmp_=np.array([[int(a>b) for b in range(10)] for a in range(10)])
def rk(M): return np.linalg.matrix_rank(M.astype(float))
print("rank parity (+-1):", rk(2*par-1), " rank carry:", rk(carry), " rank cmp:", rk(cmp_))
print("rank of each units-digit class indicator:", [rk(units==c) for c in range(10)])
print("rank of each full-sum class indicator:", [rk(np.add.outer(range(10),range(10))==s) for s in range(19)])
# additive (row+col) model: fraction of variance of each target explained by best additive fit on the full table
def addvar(M):
    M=M.astype(float); m=M.mean(); r=M.mean(1,keepdims=True)-m; c=M.mean(0,keepdims=True)-m
    A=m+r+c; return 1-((M-A)**2).sum()/((M-m)**2).sum()
print("variance explained by additive model: parity %.3f carry %.3f cmp %.3f"%(addvar(par),addvar(carry),addvar(cmp_)),
      " units classes mean %.3f"%np.mean([addvar(units==c) for c in range(10)]))
# Fourier spectra over Z10 of ASCII bits of units digit d (bits 0-3)
for j in range(4):
    g=np.array([((48+d)>>j)&1 for d in range(10)],float); g-=g.mean()
    F=np.abs(np.fft.fft(g))**2; F=F/F.sum() if F.sum()>0 else F
    print("bit%d of units digit: power by frequency k=0..5:"%j, np.round(F[:6]+np.r_[0,F[1:5][::-1][::-1]*0,0],3) if False else np.round([F[0]]+[F[k]+(F[10-k] if k!=5 else 0) for k in range(1,6)],3))
