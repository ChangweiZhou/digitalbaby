"""Restricted engineering comparators. No model sees task IDs or clean labels."""
from common75 import *

def feature(A,B,mask=None,bias=True):
    x=(A-B).copy()
    if mask is not None:x*=mask
    x/=max(float(np.linalg.norm(x)),1.)
    if bias:x=np.r_[x,1.]
    x/=max(float(np.linalg.norm(x)),1e-12)
    return x

class Bridge:
    def __init__(self,m,activity,expression,update):
        self.activity=activity;self.expression=expression;self.update=update
        self.s=np.zeros(len(m.slow));self.mask=np.any(m.T[:,2:]!=0,axis=1)
        self.Q=m.Q[:,2:];self.q=self.Q.mean(1);self.t=m.T[:,2:].mean(1)
        self.k0=m.k0;self.MM=m.MM;self.Qgamma=m.Q[:,:2];self.tau=m.kernel['slow_tau']
        self.zero_updates=0;self.updates=0;self.clips=0.;self.analytic_correct=[];self.linear_correct=[]

    def predict(self,dx,reader):
        self.g=(dx[0]-dx[1])*self.q*self.mask
        linear=float(self.g@self.s)
        if self.expression=='LINEAR':
            self.j=self.g;self.score=linear;self.clip=0.;self.analytic=linear
        else:
            gamma=np.clip(self.k0[3]*(dx@self.Qgamma)+35.2,0,71.66)-35.2
            baseline=self.k0[5]*(dx@self.Q)+gamma@self.MM
            raw=baseline+(dx*self.s)@self.Q
            alpha=np.clip(raw,-11.2,19.96);obs=reader.predict(dx)
            self.score=float((alpha-obs).mean(1)@[1.,-1.])
            self.analytic=float((alpha-np.clip(baseline,-11.2,19.96)).mean(1)@[1.,-1.])
            gate=(raw>-11.2)&(raw<19.96)
            self.j=(dx[0]*(self.Q*gate[0]).mean(1)-dx[1]*(self.Q*gate[1]).mean(1))*self.mask
            self.clip=float((~gate).mean())
        self.v=(dx[0]-dx[1])*self.t
        self.linear=linear
        return self.score

    def learn(self,label):
        r=2*int(label)-1
        if self.update=='GRADIENT':v=self.j;den=float(v@v);err=r-self.score
        else:v=self.v;den=float(self.g@v);err=r-self.score if self.update=='ROUTE_ERROR' else r
        assert den>=-1e-14
        change=CFG['delta_eta']*err*v/(1e-12+max(0.,den))
        self.s+=change
        self.zero_updates+=int(not np.any(change));self.updates+=1;self.clips+=self.clip
        self.s*=np.exp(-330/self.tau)
        if not np.isfinite(self.s).all():raise FloatingPointError('Nonfinite bridge state')
        assert not np.any(self.s[~self.mask])

def task_stream(seed,task):
    # A separate deterministic generator per task; identical stream shared by all arms.
    rng=np.random.default_rng(seed+({'ANOMALY':110000,'REVERSAL':220000}[task]))
    base=rng.integers(0,2,64);out=[]
    for t in range(CFG['trials']):
        idx=int(rng.integers(64));clean=int(base[idx])
        if task=='REVERSAL':clean^=(t//256)%2
        label=clean
        if task=='ANOMALY' and rng.random()<.1:label=1-label
        out.append((idx,label,clean))
    return out

def run(spec,folder):
    seed=spec['seed'];pn,_=bank(seed);template=Fly();codes=np.array([template.encode(p) for p in pn[:128]])
    mask=np.any(template.T[:,2:]!=0,axis=1);reader=FrozenReader(template.Q,READER)
    anchors={'NLMS_FULL':(None,True),'NLMS_SUPPORT':(mask,True),'NLMS_SUPPORT_NO_BIAS':(mask,False)}
    X={name:np.array([feature(codes[2*i],codes[2*i+1],mk,b) for i in range(64)]) for name,(mk,b) in anchors.items()}
    allrows=[];byepoch=[]
    for task in CFG['bridge_tasks']:
        parent=Fly();adapt=Fly();adapt.mode='no_learning'
        learners={f'{a}__{e}__{u}':Bridge(template,a,e,u) for a in CFG['activity_modes'] for e in CFG['expression_modes'] for u in CFG['update_modes']}
        weights={n:np.zeros(x.shape[1]) for n,x in X.items()}
        scores={n:[] for n in list(weights)+list(learners)+['MINIFLY_REF']};diags={n:[] for n in learners}
        for t,(idx,label,clean) in enumerate(task_stream(seed,task)):
            cc=codes[2*idx:2*idx+2];dx=observed_activity(adapt,cc)
            predictions={}
            for n,w in weights.items():
                z=X[n][idx];score=float(w@z);predictions[n]=(score,float(np.clip((score+1)/2,0,1)))
                w+=.5*((2*label-1)-score)*z
            for n,m in learners.items():
                score=m.predict(cc if m.activity=='STATIC' else dx,reader)
                predictions[n]=(score,None)
                if t>=CFG['warmup']:diags[n].append((float((m.analytic>=0)==bool(clean)),float((m.linear>=0)==bool(clean)),m.clip))
                m.learn(label)
            _,rd,_=probe(parent,cc,np.array([1]),reader)
            predictions['MINIFLY_REF']=(float(rd[0]),None)
            if t>=CFG['warmup']:
                for n,(score,prob) in predictions.items():
                    scores[n].append((t//256,float((score>=0)==bool(clean)),float(abs(score)<1),float((prob-clean)**2) if prob is not None else np.nan))
            sch=list(schedule(parent,*pn[2*idx:2*idx+2],label,1));train(parent,sch);train(adapt,sch)
            assert np.max(abs(parent.adapt-adapt.adapt))<1e-12
        for n,vals in scores.items():
            ar=np.array(vals);row=dict(seed=seed,task=task,arm=n,accuracy=float(ar[:,1].mean()),below_1Hz=float(ar[:,2].mean()),
                probability_MSE=float(ar[:,3].mean()) if n in weights else np.nan,n_scored=len(ar))
            if n in learners:
                m=learners[n];d=np.array(diags[n]);row.update(mutable_bytes=m.s.nbytes+(adapt.adapt.nbytes if m.activity=='ADAPTED' else 0)+24,
                    max_abs_state=float(abs(m.s).max()),state_L2=float(np.linalg.norm(m.s)),zero_update_fraction=m.zero_updates/m.updates,
                    clipped_fraction=float(d[:,2].mean()),analytic_accuracy=float(d[:,0].mean()),linear_accuracy=float(d[:,1].mean()))
            elif n in weights:row.update(mutable_bytes=weights[n].nbytes,max_abs_state=float(abs(weights[n]).max()),state_L2=float(np.linalg.norm(weights[n])))
            else:row.update(mutable_bytes=parent.mutable_bytes(),max_abs_state=float(abs(parent.slow).max()),state_L2=float(np.linalg.norm(parent.slow)))
            allrows.append(row)
            for epoch in [1,2,3]:byepoch.append(dict(seed=seed,task=task,arm=n,epoch=epoch,accuracy=float(ar[ar[:,0]==epoch,1].mean())))
    pd.DataFrame(allrows).to_csv(folder/'bridge.csv',index=False);pd.DataFrame(byepoch).to_csv(folder/'epochs.csv',index=False)
    atomic_json(folder/'support.json',dict(n_KC=len(mask),alpha_support=int(mask.sum()),alpha_Q_support=int((template.q if hasattr(template,'q') else template.Q[:,2:].sum(1)>0).sum()),
        intersection=int((mask&(template.Q[:,2:].sum(1)>0)).sum()),fixed_parent_bytes=template.fixed_numeric_bytes(),reader_bytes=reader.extra_fixed_numeric_bytes(),
        scope='T support and pooled route weights; global error and normalization remain diagnostic external computations.'))
    return dict(rows=len(allrows),seed=seed)
