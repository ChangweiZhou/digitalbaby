# Root-cause step A: do the 40 taught orbits determine the 15 held answers, and under which assumption?
import sys, numpy as np
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[1]/'minifly_learned_bank_v1_20261024')); import data
W=['RC-D%d'%i for i in range(1,9)]
for w in W:
    wd=data.make_world(w); tr=wd['train_pairs']; he=wd['held']
    # (1) additive numeric model s(a,b)=u_a+v_b: held cell determined iff its design row is in the span of taught rows
    A=np.array([np.r_[np.eye(10)[a],np.eye(10)[b]] for a,b in tr]); r=np.linalg.matrix_rank(A)
    det=sum(np.linalg.matrix_rank(np.vstack([A,np.r_[np.eye(10)[a],np.eye(10)[b]]]))==r for a,b in he)
    # (2) identity-only evidence: held orbit's own sum value appears among taught orbits (what a sum-keyed lookup could reuse)
    sums={a+b for a,b in wd['train']}; cov=sum((a+b) in sums for a,b in he)
    # (3) rectangle witness: exists c,d with (a,d),(c,b),(c,d) all taught -> one-step additive inference
    T=set(tr); rect=sum(any((a,d) in T and (c,b) in T and (c,d) in T for c in range(10) for d in range(10)) for a,b in he)
    print(f"{w}: additive model determines {det}/15 held orbits (design rank {r}/19); one-step rectangle witness {rect}/15; held sum value seen in training {cov}/15")
