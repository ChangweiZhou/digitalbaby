"""Fixed-gain and analytic fixture checks. No fitted model or closed-loop learner."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
GOALS=np.array([[.30,.50],[.70,.50]])
SEED=61317


def trajectory(theta,moves):
    h=float(theta);xy=np.zeros(2)
    for yaw,advance in moves:
        h+=yaw
        xy+=advance*np.array([np.cos(h),np.sin(h)])
    return h,xy


def arena():
    rng=np.random.default_rng(SEED)
    starts=np.column_stack([rng.uniform(.43,.57,32),np.repeat([.16,.84],16)])
    heading=rng.uniform(-np.pi,np.pi,32)
    visits=np.column_stack([np.linspace(.24,.76,53),np.full(53,.5)])
    assert np.min(np.linalg.norm(starts[:,None]-visits[None],axis=2))>.25
    def run(s,h,g):
        xy=s.copy()
        for i in range(200):
            target=GOALS[g]-xy
            angular=(np.arctan2(target[1],target[0])-h+np.pi)%(2*np.pi)-np.pi
            h+=np.clip(angular,-.25,.25)
            if abs(angular)<.5:xy+=.018*np.array([np.cos(h),np.sin(h)])
            if np.any((xy<0)|(xy>1)):return -1,i+1
            zones=np.flatnonzero(np.linalg.norm(GOALS-xy,axis=1)<=.06)
            if len(zones):return int(zones[0]),i+1
        return -1,200
    a=[run(p,h,0) for p,h in zip(starts,heading)]
    b=[run(p,h,1) for p,h in zip(starts,heading)]
    both=sum(x[0]==0 and y[0]==1 for x,y in zip(a,b))
    assert both==32
    # Same visits and opportunities, two opposite scalar teaching histories.
    visit_ids=np.tile([0,1,1,0],8)
    rewards=(visit_ids==0).astype(int)
    sham=rng.permutation(rewards)
    assert rewards.sum()==(1-rewards).sum()==sham.sum()
    return {"analytic_paired_first_entries":both,"n":32,"max_steps":max(t for _,t in a+b),
            "fixed_zone_paired":0,"training_visit_separation":float(np.min(np.linalg.norm(starts[:,None]-visits[None],axis=2))),
            "reward_opportunities":len(rewards),"each_history_rewards":int(rewards.sum()),
            "engineering_only":True}


def rewire(u,v,sign,roles):
    rng=np.random.default_rng(SEED)
    groups=defaultdict(list)
    for i,(a,b,s) in enumerate(zip(u,v,sign)):
        groups[(roles[a],roles[b],s)].append(i)
    q=v.copy(); present=set(zip(u.tolist(),v.tolist()));accepted=0
    for ids in groups.values():
        if len(ids)<2:continue
        for i,j in rng.choice(ids,size=(min(2*len(ids),6000),2),replace=True):
            a,b,c,d=int(u[i]),int(q[i]),int(u[j]),int(q[j])
            if a==c or b==d or a==d or c==b or (a,d) in present or (c,b) in present:continue
            present.remove((a,b));present.remove((c,d))
            present.add((a,d));present.add((c,b))
            q[i],q[j]=d,b;accepted+=1
    for s in (-1,1):
        assert np.array_equal(np.bincount(v[sign==s],minlength=len(roles)),
                              np.bincount(q[sign==s],minlength=len(roles)))
    assert np.array_equal(roles[v],roles[q])
    assert accepted>0
    return q,accepted


def stress(u,v,w,roles):
    start=time.process_time()
    mass=np.bincount(v,weights=abs(w),minlength=len(roles))
    weights=1.2*w/np.maximum(mass[v],1e-12)
    h=np.zeros(len(roles))
    cue=np.flatnonzero(np.isin(roles,["ER","PEN","PFN"]))
    rng=np.random.default_rng(SEED)
    maximum=0
    for step in range(120):
        drive=np.zeros(len(roles))
        if step<40:drive[cue]=rng.uniform(0,.6,len(cue))
        h=.8*h+.2*np.tanh(drive+np.bincount(v,weights=weights*h[u],minlength=len(h)))
        assert np.isfinite(h).all()
        maximum=max(maximum,np.max(abs(h)))
    return {"cpu_seconds":time.process_time()-start,"max_abs_state":float(maximum),
            "late_EPG_norm":float(np.linalg.norm(h[roles=="EPG"]))}


def cycle2(g):
    a=g["usable"]
    u,v,s,w=[g[key][a] for key in ("u","v","sign","w")]
    roles=g["roles"]
    assert len(set(zip(u.tolist(),v.tolist())))==len(u)
    q,accepted=rewire(u,v,s,roles)
    h,p=trajectory(0,[(np.pi/2,0),(0,1)])
    _,r=trajectory(0,[(0,1),(np.pi/2,0)])
    assert np.allclose(p,[0,1]) and np.allclose(r,[1,0])
    return {"arena":arena(),"known_angle_path_order_canary":True,
            "rewired_swaps":accepted,"changed_edges":int((v!=q).sum()),
            "role_pairs_and_signed_degrees_preserved":True,
            "numerical_rollout":{"native":stress(u,v,s*w,roles),"rewired":stress(u,q,s*w,roles)},
            "anatomical_heading_and_displacement":"blocked before functional scoring by missing compartment mapping and boundary adapters"}


class Local:
    def __init__(self,g):
        self.n=len(g["roles"])
        u,v=g["u"],g["v"]
        r=g["roles"]
        a=(r[u]=="PN")&(r[v]=="KC")&g["usable"]
        self.pu,self.pv,self.pw=u[a],v[a],g["w"][a]
        self.norm=np.bincount(self.pv,weights=self.pw,minlength=self.n).clip(1e-9)
        mask=g["plastic"]&g["usable"]
        self.u,self.v,self.base=u[mask],v[mask],g["w"][mask]
        addressed=set(v[(r[u]=="DAN")&(r[v]=="KC")])
        self.eligible=np.isin(self.u,list(addressed))
        self.post_norm=np.bincount(self.v,weights=self.base,minlength=self.n).clip(1e-9)
        self.w=self.base.copy()
        self.e=np.zeros(len(self.w))
        self.ema=np.zeros(self.n)
        self.rbar=0.
        self.pns=np.flatnonzero(r=="PN")
        self.mbons=np.flatnonzero(r=="MBON")

    def observe(self,i):
        rng=np.random.default_rng(SEED+i)
        z=np.zeros(self.n)
        z[rng.choice(self.pns,32,replace=False)]=1
        x=np.bincount(self.pv,weights=self.pw*z[self.pu],minlength=self.n)/self.norm
        k=np.flatnonzero(x)
        k=k[np.argsort(-x[k],kind="stable")[:256]]
        xx=np.zeros(self.n);xx[k]=1
        y=np.bincount(self.v,weights=self.w*xx[self.u],minlength=self.n)/self.post_norm
        return xx,y

    def step(self,i,r,writes):
        x,y=self.observe(i)
        self.e=.75*self.e+x[self.u]*(y[self.v]-self.ema[self.v])
        self.ema=.9*self.ema+.1*y
        d=.8*(r-self.rbar)*self.e*self.eligible
        self.rbar=.9*self.rbar+.1*r
        if writes:self.w=np.clip(self.w+d,.5*self.base,1.5*self.base)
        return y[self.mbons]

    def digest(self):
        return sha256(b"".join(x.tobytes() for x in (self.w,self.e,self.ema,np.array(self.rbar)))).hexdigest()

    def save(self,path):
        np.savez(path,w=self.w,e=self.e,ema=self.ema,rbar=self.rbar)

    def load(self,path):
        with np.load(path) as z:
            self.w,self.e,self.ema,self.rbar=(z[k].copy() for k in ("w","e","ema","rbar"))


def cycle3(g,scratch):
    results={}
    for mode in ("paired","shifted","no_write"):
        m=Local(g)
        w0=m.w.copy()
        before=m.digest()
        for i in range(64):
            cue=i%2
            reward=int(cue==(0 if i<32 else 1))
            if mode=="shifted":reward=1-reward
            m.step(cue,reward,mode!="no_write")
            if i in (31,63):
                result=results.setdefault(mode,[])
                result.append({"trial":i+1,"changed_edges":int(np.count_nonzero(m.w!=w0)),
                               "cue_outputs":[m.observe(k)[1][m.mbons].tolist() for k in (0,1)]})
        assert np.all(m.w>0) and not np.any((m.w!=w0)&~m.eligible)
        assert (np.array_equal(m.w,w0))==(mode=="no_write")
        scored=m.digest()
        m.observe(0)
        assert m.digest()==scored and m.digest()!=before
        if mode=="paired":
            path=scratch/"local.npz"
            m.save(path)
            m.step(0,1,True)
            restart={"expected":m.digest(),"checkpoint_sha256":sha256(path.read_bytes()).hexdigest()}
    return {"modes":results,"eligible_DAN_addressed_KC_MBON_edges":int(m.eligible.sum()),
            "restart":restart,"scope":"anatomical-edge odor write fixture; no learned arena policy"}


def main():
    mode=sys.argv[1]
    g=np.load(ROOT/"EXTRACT.npz",allow_pickle=False)
    p=ROOT/("CYCLE2.json" if mode=="2" else "CYCLE3.json")
    if mode=="2":result=cycle2(g)
    else:
        scratch=Path(sys.argv[2])
        if mode=="3":result=cycle3(g,scratch)
        else:
            result=json.loads(p.read_text());m=Local(g);path=scratch/"local.npz"
            assert sha256(path.read_bytes()).hexdigest()==result["restart"]["checkpoint_sha256"]
            m.load(path);m.step(0,1,True)
            result["restart"]["actual"]=m.digest()
            assert m.digest()==result["restart"]["expected"]
    result["status"]="RESTART_PENDING" if mode=="3" else "ENGINEERING_PASS"
    p.write_text(json.dumps(result,indent=2)+"\n")
    print(mode,result["status"],flush=True)


if __name__=="__main__":main()
