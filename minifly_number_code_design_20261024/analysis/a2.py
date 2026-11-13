import sys, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]/'minifly_learned_bank_v1_20261024'))
import data, lb1
from scipy.stats import spearmanr
ALL=[(a,b) for a in range(10) for b in range(10)]
X=np.array([lb1.trace_for(data.prompt(a,b)) for a,b in ALL]); idx={p:i for i,p in enumerate(ALL)}
# ordinal structure in the digit codes? distance between prompts differing only in first digit vs |a-a'|
d=[];g=[]
for b in range(10):
    for a in range(10):
        for a2 in range(a+1,10):
            d.append(np.linalg.norm(X[idx[(a,b)]]-X[idx[(a2,b)]])); g.append(abs(a-a2))
print("Spearman(code distance, |a-a'|) first operand: %.3f"%spearmanr(d,g)[0])
W=['D1','D2','D3','M1','M2','M3','M4']
tasks={'PAR':lambda a,b:(a+b)%2,'CARRY':lambda a,b:int(a+b>=10),'CMP':lambda a,b:int(a>b),'COPY':lambda a,b:a,
       'UNITS':lambda a,b:(a+b)%10,'SUM':lambda a,b:a+b}
def onehot(a,b): return np.r_[np.eye(10)[a],np.eye(10)[b]]
print("\nAdditive (row+column) ridge smoother on one-hot codes = what frozen random features compute; held accuracy (plain), per world:")
for t,f in tasks.items():
    accs=[]
    for w in W:
        wd=data.make_world(w); tr,he=wd['train_pairs'],wd['held_pairs']
        if t=='CMP':
            perm=np.random.default_rng(data.seed_of("cmp:"+w)).permutation(100); he=[ALL[i] for i in perm[:26]]; tr=[ALL[i] for i in perm[26:]]
        classes=sorted({f(*p) for p in ALL})
        A=np.array([np.r_[onehot(*p),1] for p in tr]); Y=np.array([[float(f(*p)==c) for c in classes] for p in tr])
        coef=np.linalg.solve(A.T@A+1e-3*np.eye(A.shape[1]),A.T@Y)
        pred=[classes[int(np.argmax(np.r_[onehot(*p),1]@coef))] for p in he]
        accs.append(np.mean([pr==f(*p) for pr,p in zip(pred,he)]))
    print("  %-6s"%t, " ".join("%.2f"%x for x in accs), " mean %.2f"%np.mean(accs))
# sum coverage: held orbits whose sum value occurs among training orbits
print("\nHeld orbits whose exact sum occurs in some training orbit (what a learner keyed on a+b could answer):")
for w in W:
    wd=data.make_world(w); trs={a+b for a,b in wd['train']}
    print("  %s %d/%d"%(w,sum((a+b) in trs for a,b in wd['held']),len(wd['held'])))
