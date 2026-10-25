# Step C': clean 1-D additive bottleneck s=u_a+v_b (20 learned scalars), fixed RBF tuning bank over standardized s
# (width sigma), softmax readout (delta rule). Credit for u,v: BP-diagnostic (exact) or a local same-answer pull/push rule.
# Inits: random vs near the rule. Labels: true vs orbit-consistent shuffled.
import sys, numpy as np
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[1]/'minifly_learned_bank_v1_20261024')); import data
W=['RC-D%d'%i for i in range(1,9)]
def shuf_labels(wd,seed):
    r=np.random.default_rng(seed); orb=wd['train']; lab=[a+b for a,b in orb]
    for _ in range(1000):
        p=r.permutation(len(orb)); new=[lab[i] for i in p]
        if np.mean([x==y for x,y in zip(new,lab)])<0.15: break
    m={o:new[i] for i,o in enumerate(orb)}; return {(a,b):m[(min(a,b),max(a,b))] for a,b in wd['train_pairs']}
C=np.linspace(-2.5,2.5,81)
def feats(s,mu,sd,sig): z=(s-mu)/sd; return np.exp(-(z[:,None]-C[None,:])**2/(2*sig**2))
def run(tr,ytr,he,yte,credit,init,sig,seed,steps=6000,lr_u=0.02,lr_w=0.5):
    r=np.random.default_rng(seed); u=r.normal(0,1,10); v=r.normal(0,1,10)
    if init=='near': u=np.arange(10)+r.normal(0,0.3,10); v=np.arange(10)+r.normal(0,0.3,10)
    Wr=np.zeros((len(C),19)); a=np.array([p[0] for p in tr]); b=np.array([p[1] for p in tr]); Y=np.eye(19)[ytr]
    for t in range(steps):
        s=u[a]+v[b]; mu,sd=s.mean(),s.std()+1e-9; F=feats(s,mu,sd,sig); Z=F@Wr; Z-=Z.max(1,keepdims=True); P=np.exp(Z); P/=P.sum(1,keepdims=True); E=P-Y
        Wr-=lr_w*F.T@E/len(tr)
        if credit=='BP':
            dF=-(F*((s-mu)/sd)[:,None]-F*C[None,:])/(sig**2)/sd     # d feats / d s (holding mu,sd fixed)
            ds=np.sum((E@Wr.T)*dF,1)/len(tr)
        else:  # local same-answer rule: pull s of same-label pairs together, push different-label neighbours apart
            i,j=r.integers(0,len(tr),2*len(tr)).reshape(2,-1); same=(ytr[i]==ytr[j]); diff=s[i]-s[j]
            ds=np.zeros(len(tr)); g=np.where(same,diff,-np.sign(diff)*np.exp(-np.abs(diff)/sd)*0.5)
            np.add.at(ds,i,g); np.add.at(ds,j,-g); ds/=len(tr)
        gu=np.zeros(10); gv=np.zeros(10); np.add.at(gu,a,ds); np.add.at(gv,b,ds); u-=lr_u*gu*10; v-=lr_u*gv*10
    s=u[a]+v[b]; mu,sd=s.mean(),s.std()+1e-9
    pred=lambda aa,bb: np.argmax(feats(u[aa]+v[bb],mu,sd,sig)@Wr,1)
    ah=np.array([p[0] for p in he]); bh=np.array([p[1] for p in he])
    return np.mean(pred(a,b)==ytr), np.mean(pred(ah,bh)==yte)
for credit in ('BP','LOCAL'):
  for sig in (0.03,0.1,0.3):
    for init in ('random','near'):
      out={'true':[],'shuf':[]}
      for w in W:
        wd=data.make_world(w); tr,he=wd['train_pairs'],wd['held_pairs']; yte=np.array([x+y for x,y in he]); sh=shuf_labels(wd,data.seed_of('rc-shuf:'+w))
        for s_ in range(2):
          out['true'].append(run(tr,np.array([x+y for x,y in tr]),he,yte,credit,init,sig,s_))
          out['shuf'].append(run(tr,np.array([sh[p] for p in tr]),he,yte,credit,init,sig,s_))
      m=lambda k,i: np.mean([x[i] for x in out[k]])
      print(f"{credit:5s} sigma={sig:<4} init={init:6s}  TRUE taught {m('true',0):.2f} held {m('true',1):.2f} | SHUF taught {m('shuf',0):.2f}",flush=True)
