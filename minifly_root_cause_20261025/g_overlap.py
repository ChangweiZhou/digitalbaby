# Supporting measurement: overlap of canonical PN->KC codes (V143 FROZEN learned-bank path) by answer relation.
import sys, numpy as np
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[1]/'minifly_learned_bank_v1_20261024')); import data, lb1
m=lb1.Model('FROZEN', data.seed_of('model:RC-D1'))
ALL=[(a,b) for a in range(10) for b in range(10)]
X=np.array([m.codes_for(data.prompt(a,b), 0)[1] for a,b in ALL]); inter=X@X.T; n=X.sum(1); J=inter/(n[:,None]+n[None,:]-inter)
sh=[];no=[];same=[]
for i,(a,b) in enumerate(ALL):
    for j,(c,d) in enumerate(ALL):
        if j<=i: continue
        (same if a+b==c+d else sh if (a==c or b==d) else no).append(J[i,j])
print('mean Jaccard: sharing an operand %.3f | same answer %.3f | other %.3f'%(np.mean(sh),np.mean(same),np.mean(no)))
