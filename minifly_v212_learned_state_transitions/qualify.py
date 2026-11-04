"""Development qualification for V212's local score-function learner."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parent
N=6
PAIRS=np.array([(a,b) for a in range(N) for b in range(N) if a!=b])
A,B=PAIRS.T
INCOMING=np.eye(N)[B].T/5
FAMILIES=('coupling','copy')
ARMS=('full','frozen','shuffled')

def sha(raw):return hashlib.sha256(raw).hexdigest()

def episodes(family,seed,batch):
    rng=np.random.default_rng(seed)
    roles=np.array([rng.permutation(N)[:3] for _ in range(batch)])
    carrier,obj,other=roles.T
    x=rng.integers(4,size=batch)
    y=(x+rng.integers(1,4,size=batch))%4
    z=(y+rng.integers(1,4,size=batch))%4
    attached=rng.integers(2,size=batch).astype(bool)
    releases=rng.integers(2,size=batch).astype(bool)
    initial=np.where(rng.integers(2,size=batch),x,(x+1)%4)
    # Only the simulator computes consequences. Inputs use opaque codes.
    def evt(code,a,b,entity=False):
        return np.broadcast_to(code,(batch,)),a,b,np.full(batch,entity)
    first=np.where(attached,1,3)
    second=np.where(releases,2,3)
    loc1=np.where(attached,y if family=='coupling' else x,initial)
    loc2=np.where(attached & ~releases,z,loc1) if family=='coupling' else loc1
    return [
        ('input',evt(4,carrier,x)),('input',evt(4,obj,initial)),('input',evt(4,other,z)),
        ('input',evt(first,carrier,obj,True)),('input',evt(0,carrier,y)),
        ('probe',(obj,loc1,'changed')),('input',evt(3,other,obj,True)),
        ('probe',(carrier,y,'stable')),('input',evt(second,carrier,obj,True)),
        ('input',evt(0,carrier,z)),('probe',(obj,loc2,'changed')),
        ('probe',(other,z,'stable'))]

def sigmoid(z):return 1/(1+np.exp(-np.clip(z,-30,30)))

class Model:
    def __init__(self,seed):
        rng=np.random.default_rng(seed)
        self.entity=rng.normal(0,.08,(25,8))
        self.pair=rng.normal(0,.08,(34,4))
        self.out=rng.normal(0,.08,(9,4))
        self.baseline=np.full(4,np.log(4.))
        self.step=0

    def simulate(self,tape,seed,arm=None,assimilate=False):
        rng=np.random.default_rng(seed)
        batch=len(tape[0][1][0])
        n=np.arange(batch)
        h=np.zeros((batch,N,8))
        g=np.zeros((batch,len(A),4))
        et=np.zeros((batch,*self.entity.shape)); pt=np.zeros((batch,*self.pair.shape))
        de=np.zeros_like(self.entity); dp=np.zeros_like(self.pair); dout=np.zeros_like(self.out)
        result=[]
        def transition(code,left,right,right_entity):
            nonlocal h,g,et,pt
            cat=np.eye(5)[code]
            pos=np.eye(4)[np.minimum(right,3)]*(~right_entity[:,None])
            mask=right_entity[:,None]
            left_ind=left[:,None]==np.arange(N)
            right_ind=(right[:,None]==np.arange(N))&mask
            def spread(v,count):return np.broadcast_to(v[:,None,:],(batch,count,v.shape[-1]))
            def flags(*ls):return np.stack(ls,axis=-1)
            ex=np.concatenate((h,np.einsum('ij,bjk->bik',INCOMING,g),spread(cat,N),spread(pos,N),
                               flags(left_ind,right_ind),np.ones((batch,N,1)),
                               spread((code==4)[:,None],N)),axis=-1)
            px=np.concatenate((g,h[:,A],h[:,B],spread(cat,len(A)),spread(pos,len(A)),
                               flags(left_ind[:,A],left_ind[:,B],right_ind[:,A],right_ind[:,B]),
                               np.ones((batch,len(A),1))),axis=-1)
            pe=sigmoid(ex@self.entity); pp=sigmoid(px@self.pair)
            h=(rng.random(pe.shape)<pe).astype(float)
            g=(rng.random(pp.shape)<pp).astype(float)
            if arm is not None:
                et+=np.einsum('bni,bnj->bij',ex,h-pe)
                pt+=np.einsum('bni,bnj->bij',px,g-pp)
        for kind,values in tape:
            if kind=='input':
                transition(*values)
                continue
            entity,target,panel=values
            vec=np.column_stack((h[n,entity],np.ones(batch)))
            logits=vec@self.out
            prob=np.exp(logits-logits.max(axis=1,keepdims=True))
            prob/=prob.sum(axis=1,keepdims=True)
            loss=-np.log(prob[n,target])
            idx=len(result)
            result.append((panel,prob.copy(),target.copy(),loss.copy()))
            if arm is not None:
                err=np.eye(4)[target]-prob
                dout+=vec.T@err/batch
                centered=loss-self.baseline[idx]
                if arm=='shuffled':
                    centered=centered[np.random.default_rng(seed+30000+idx).permutation(batch)]
                de-=np.einsum('b,bij->ij',centered,et)/batch
                dp-=np.einsum('b,bij->ij',centered,pt)/batch
            if assimilate:
                transition(np.full(batch,4),entity,target,np.zeros(batch,bool))
        if arm is not None:
            if arm!='frozen':
                self.entity+=.015*de
                self.pair+=.015*dp
            self.out+=.08*dout
            self.baseline=.9*self.baseline+.1*np.array([r[3].mean() for r in result])
            self.step+=1
        return result

    def digest(self):
        return sha(b''.join(getattr(self,k).tobytes() for k in ('entity','pair','out','baseline'))+
                   str(self.step).encode())

def train(m,family,arm,end):
    for step in range(m.step,end):
        tape=episodes(family,5000+step,32)
        m.simulate(tape,7000+step,arm,assimilate=True)

def score(m,family):
    tape=episodes(family,901,64)
    result={}
    initial=tape[1][1][2]
    forecasts=[m.simulate(tape,10000+i) for i in range(32)]
    for name in ('changed','stable'):
        acc=[]; bits=[]; changed_only=[]; persistence=[]
        for j,entry in enumerate(forecasts[0]):
            panel,_,target,_=entry
            if panel!=name:continue
            p=np.mean([r[j][1] for r in forecasts],axis=0)
            hits=np.argmax(p,axis=1)==target
            acc.extend(hits)
            if panel=='changed':
                changed_only.extend(hits[target!=initial])
                persistence.extend(target==initial)
            bits.extend(-np.log2(p[np.arange(len(target)),target]))
        result[name]=dict(accuracy=float(np.mean(acc)),bits=float(np.mean(bits)))
        if name=='changed':
            result[name].update(changed_only=float(np.mean(changed_only)),changed_n=len(changed_only),
                                last_observation=float(np.mean(persistence)))
    return result

def recover(path):
    if sha(path.read_bytes())!=path.with_suffix('.sha').read_text():
        raise ValueError('Checkpoint digest mismatch')
    m=Model(101)
    with np.load(path,allow_pickle=False) as data:
        for k in ('entity','pair','out','baseline'):setattr(m,k,data[k])
        m.step=int(data['step'])
    return m

if __name__=='__main__':
    if len(sys.argv)>1:
        m=recover(Path(sys.argv[1]))
        train(m,'coupling','full',12)
        print(json.dumps([m.digest(),score(m,'coupling')]))
        sys.exit()
    rng=np.random.default_rng(101)
    w=.3
    p=sigmoid(w)
    h=rng.random(250000)<p
    # Loss for either stochastic state in a tiny delayed-credit fixture.
    sampled=float(np.mean(np.where(h,.2,1.7)*(h-p)))
    def expected(x):return .2*sigmoid(x)+1.7*(1-sigmoid(x))
    fd=float((expected(w+1e-5)-expected(w-1e-5))/2e-5)
    assert abs(sampled-fd)<.01
    rows=[]
    for family in FAMILIES:
        for arm in ARMS:
            m=Model(101)
            before=m.entity.copy(); before_pair=m.pair.copy()
            pre=score(m,family)
            train(m,family,arm,120)
            assert np.array_equal(before,m.entity)==(arm=='frozen')
            assert np.array_equal(before_pair,m.pair)==(arm=='frozen')
            state=m.digest()
            post=score(m,family)
            assert state==m.digest()
            rows.append(dict(family=family,arm=arm,pre=pre,post=post,state=state))
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'trained.npz'
        m=Model(101)
        train(m,'coupling','full',6)
        np.savez(path,**{k:getattr(m,k) for k in ('entity','pair','out','baseline','step')})
        path.with_suffix('.sha').write_text(sha(path.read_bytes()))
        train(m,'coupling','full',12)
        cmd=[sys.executable,'-B',__file__,str(path)]
        assert json.loads(subprocess.check_output(cmd))==[m.digest(),score(m,'coupling')]
        path.write_bytes(path.read_bytes()+b'bad')
        bad=subprocess.run(cmd,capture_output=True,text=True)
        assert bad.returncode and 'Checkpoint digest mismatch' in bad.stderr
    out=dict(version='V212',status='DEVELOPMENT_ONLY',source=sha(Path(__file__).read_bytes()),
             finite_difference=fd,sampled=sampled,rows=rows,restart='exact state and evaluation match')
    (ROOT/'ENGINEERING.json').write_text(json.dumps(out,indent=2)+'\n')
    for row in rows:print(row['family'],row['arm'],row['post'])
