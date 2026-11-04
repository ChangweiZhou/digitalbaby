import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ARMS = ('frozen','direct','slow','low_td')
def sha(b): return hashlib.sha256(b).hexdigest()
def parse(raw):
    assert len(raw)==7 and raw[:4]==b'par ' and raw[-1:]==b'='
    return raw[4]-97,raw[5]-97

def run(job, restart=False):
    seed,arm=job
    rng=np.random.default_rng(seed)
    old=rng.permutation(12)
    new=old.copy()
    new[:2]=old[1::-1]
    initial=np.zeros((2,12,12))
    eye=np.eye(12)
    def learn(fast,deployed,run,tau,rate):
        for m,stride in enumerate((1,2)):
            for x,y in zip(run[:14],run[stride:]):
                row=fast[m,x]
                row += rate*(eye[x]+(.65 if m==0 else .85)*fast[m,y]-row)
                deployed[m,x] += tau*(row-deployed[m,x])
    shadow=initial.copy()
    for start in rng.integers(12,size=1200):
        learn(initial,shadow,old[(start+np.arange(16))%12],1,.12)
    assert np.allclose(initial,shadow)
    fast=initial.copy()
    deployed=initial.copy()
    projection=rng.normal(size=(2000,48))
    weights=np.zeros((2000,256))
    eos=np.zeros((256,256),dtype=np.int64)
    polarity=int(rng.integers(2))
    def target(pair,order):
        x,y=pair
        pos=np.argsort(order)
        return 60+2*(int((pos[x]-pos[y])%2==0)^polarity)
    pairs=[(x,y) for x in range(12) for y in range(12) if x!=y]
    test=[p for p in pairs if (sum(p)+seed)%3==0]
    train=[p for p in pairs if p not in test]
    assert not set(train)&{p[::-1] for p in test}
    changed=[p for p in test if target(p,old)!=target(p,new)]
    stable=[p for p in test if p not in changed]
    assert changed and stable and any(target(p,old)!=target(p,new) for p in train)
    def predict(pair):
        x,y=parse(b'par '+bytes([pair[0]+97,pair[1]+97])+b'=')
        z=np.tanh(projection@deployed[:,[x,y],:].ravel())
        ids=np.argsort(-np.abs(z),kind='stable')[:100]
        v=z[ids]/np.linalg.norm(z[ids])
        return ids,v,v@weights[ids]
    def score(panel,order):
        return sum(int((a:=int(np.argmax(predict(p)[2])))==target(p,order)
                       and np.argmax(eos[a])==10) for p in panel)/len(panel)
    result={}
    for phase,count in [('clean',2048),('noise',512),('change',512)]:
        order=new if phase=='change' else old
        for event in range(count):
            if phase!='clean':
                seq=order[(int(rng.integers(12))+np.arange(16))%12].copy()
                if phase=='noise':
                    bad=rng.random(16)<.10
                    seq[bad]=(seq[bad]+rng.integers(1,12,size=sum(bad)))%12
                if arm!='frozen':
                    learn(fast,deployed,seq,.05 if arm=='slow' else 1,.006 if arm=='low_td' else .12)
            if phase=='change' and event in (0,32,128):
                result[f'change_before_feedback_{event}']=[score(stable,order),score(changed,order)]
            pair=train[int(rng.integers(len(train)))]
            ids,v,scores=predict(pair)
            y=target(pair,order)
            scores[y]-=1
            weights[ids]-=.3*np.outer(v,scores)
            eos[y,10]+=1
        before=sha(weights.tobytes()+deployed.tobytes())
        result[phase]=[score(stable,order),score(changed,order)]
        assert before==sha(weights.tobytes()+deployed.tobytes())
        if restart and phase=='noise':
            arrays=(fast,deployed,weights,eos)
            packets=[a.tobytes() for a in arrays]
            hashes=[sha(b) for b in packets]
            assert sha(packets[0]+b'corrupt')!=hashes[0]
            fast,deployed,weights,eos=[np.frombuffer(b,dtype=a.dtype).copy().reshape(a.shape)
                                       for a,b in zip(arrays,packets)]
            state=json.loads(json.dumps(rng.bit_generator.state))
            rng=np.random.default_rng()
            rng.bit_generator.state=state
    if arm=='frozen': assert np.array_equal(deployed,initial)
    if arm in ('direct','low_td'): assert np.allclose(fast,deployed)
    return dict(seed=seed,arm=arm,panels=[len(stable),len(changed)],scores=result,
                final=sha(fast.tobytes()+deployed.tobytes()+weights.tobytes()+eos.tobytes()))

if __name__=='__main__':
    jobs=[(seed,arm) for seed in (301,302) for arm in ARMS]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(run,jobs))
    assert rows[2]==run(jobs[2],True)==run(jobs[2])
    report=dict(status='ENGINEERING_PARTIAL',source_sha256=sha(Path(__file__).read_bytes()),rows=rows)
    Path(__file__).with_name('ENGINEERING.json').write_text(json.dumps(report,indent=2)+'\n')
