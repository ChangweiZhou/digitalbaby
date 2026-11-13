# Step F: relational rule through a ReLU layer (V143-style learned layer) on the raw vs normalised FE-PERM trace.
import sys, numpy as np
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[1]/'minifly_learned_bank_v1_20261024')); import data, lb1
exec(open('e_frontend.py').read().split("OH=np.array")[0])
src=open('d_ablate.py').read(); pre=src[:src.index("for kind in")]
for raw in (False,True):
    ns={'__file__':__file__}; exec(pre,ns)
    X=np.array([trace(data.prompt(a,b),raw) for a,b in ALL]); ns['FE'][:]=(X-X.mean(0))/X.std(0).clip(1e-9)
    Rep=ns['Rep']
    def init(self,kind,r,_o=Rep.__init__):
        if kind=='FE-relu32': self.kind=kind; self.Wm=r.normal(0,1/np.sqrt(88),(88,32)); self.c=np.zeros(32)
        else: _o(self,kind,r)
    def state(self,P,_o=Rep.state):
        if self.kind=='FE-relu32': return np.maximum(ns['FE'][[ns['IDX'][p] for p in P]]@self.Wm+self.c,0)
        return _o(self,P)
    def update(self,P,i,j,g,lr,_o=Rep.update):
        if self.kind!='FE-relu32': return _o(self,P,i,j,g,lr)
        Xp=ns['FE'][[ns['IDX'][p] for p in P]]; pre_=Xp@self.Wm+self.c
        gi=g*(pre_[i]>0); gj=-g*(pre_[j]>0)
        self.Wm-=lr*(Xp[i].T@gi+Xp[j].T@gj)/len(i); self.c-=lr*(gi.sum(0)+gj.sum(0))/len(i)
    Rep.__init__=init; Rep.state=state; Rep.update=update
    T=[];S=[]
    for w in ns['W']:
        wd=data.make_world(w); tr,he=wd['train_pairs'],wd['held_pairs']; yte=np.array([x+y for x,y in he]); sh=ns['shuf_labels'](wd,data.seed_of('rc-shuf:'+w))
        for s_ in range(2):
            T.append(ns['run']('FE-relu32',tr,np.array([x+y for x,y in tr]),he,yte,s_)); S.append(ns['run']('FE-relu32',tr,np.array([sh[p] for p in tr]),he,yte,s_))
    print(("raw p    " if raw else "read()   "),"relational rule, ReLU-32 layer: TRUE taught %.2f held %.2f | SHUF taught %.2f"%(np.mean([x[0] for x in T]),np.mean([x[1] for x in T]),np.mean([x[0] for x in S])),flush=True)
