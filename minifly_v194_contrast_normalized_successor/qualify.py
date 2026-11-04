import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ARMS = ('raw','normalized','centered','joint')
def sha(b): return hashlib.sha256(b).hexdigest()
def parse(raw):
    assert len(raw)==7 and raw[:4]==b'par ' and raw[-1:]==b'='
    return raw[4]-97,raw[5]-97

def run(job, checkpoint=None, resume=False, continued_noise=False):
    seed,arm=job
    rng=np.random.default_rng(seed)
    old=rng.permutation(12)
    new=old.copy()
    new[:2]=old[1::-1]
    initial=np.zeros((2,12,12))
    eye=np.eye(12)
    sums=np.zeros((2,12))
    def learn(fast,deployed,run,tau,rate):
        for m,stride in enumerate((1,2)):
            for x,y in zip(run[:14],run[stride:]):
                row=fast[m,x]
                row += rate*(eye[x]+(.65 if m==0 else .85)*fast[m,y]-row)
                delta=tau*(row-deployed[m,x])
                deployed[m,x] += delta
                sums[m] += delta
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
    eligible=[p for p in pairs if p[0]<p[1]]
    test=[]
    for flips in (False,True):
        for label in (60,62):
            pool=[p for p in eligible if (target(p,old)!=target(p,new))==flips and target(p,old)==label]
            rng.shuffle(pool)
            for p in pool[:4]:
                test.extend((p,p[::-1]))
    train=[p for p in pairs if p not in test]
    assert not set(train)&{p[::-1] for p in test}
    changed=[p for p in test if target(p,old)!=target(p,new)]
    stable=[p for p in test if p not in changed]
    for panel in (stable,changed):
        assert len(panel)==16 and all(sum(target(p,o)==60 for p in panel)==8 for o in (old,new))
    assert any(target(p,old)!=target(p,new) for p in train)
    from collections import deque
    history=deque(maxlen=128)
    for start in rng.integers(12,size=128):
        history.append(old[(start+np.arange(16))%12])
    feedback=deque(maxlen=128)
    def graph_score(panel,order):
        # Infer two groups by the minimum observed two-step cut, with byte-ID tie break.
        import itertools
        counts=np.zeros((12,12),int)
        for seq in history:
            np.add.at(counts,(seq[:-2],seq[2:]),1)
        best=None
        for subset in itertools.combinations(range(1,12),5):
            group=np.zeros(12,bool)
            group[[0,*subset]]=True
            cost=counts[group[:,None]!=group[None,:]].sum()
            if best is None or cost<best[0]:
                best=(cost,group)
        g=best[1]
        votes=[int(g[x]==g[y])^((label-60)//2) for (x,y),label in feedback]
        mapping=int(sum(votes)>len(votes)/2)
        return sum(60+2*(int(g[x]==g[y])^mapping)==target((x,y),order)
                   for x,y in panel)/len(panel)
    def predict(pair):
        x,y=parse(b'par '+bytes([pair[0]+97,pair[1]+97])+b'=')
        features=deployed[:,[x,y],:]
        if arm in ('normalized','centered'):
            features=features-sums[:,None,:]/12
            if arm=='normalized':
                features=features/(np.linalg.norm(features,axis=-1,keepdims=True)+1e-8)
        assert np.isfinite(features).all()
        z=np.tanh(projection@features.ravel())
        ids=np.argsort(-np.abs(z),kind='stable')[:100]
        norm=np.linalg.norm(z[ids])
        v=z[ids]/norm if norm else np.zeros(len(ids))
        return ids,v,v@weights[ids]
    def score(panel,order):
        return sum(int((a:=int(np.argmax(predict(p)[2])))==target(p,order)
                       and np.argmax(eos[a])==10) for p in panel)/len(panel)
    result={}
    graph={}
    phases=[('clean',2048),('noise',512),('change',512)]
    if resume:
        raw=checkpoint.read_bytes()
        if sha(raw)!=checkpoint.with_suffix('.sha').read_text():
            raise ValueError('Checkpoint digest mismatch')
        with np.load(checkpoint,allow_pickle=False) as data:
            meta=json.loads(str(data['meta']))
            if meta['job']!=list(job) or meta['continued_noise']!=continued_noise or meta['source']!=sha(Path(__file__).read_bytes()):
                raise ValueError('Checkpoint identity mismatch')
            fast,deployed,weights,eos,sums=[data[n].copy() for n in ('fast','deployed','weights','eos','sums')]
            history=deque(data['history'],maxlen=128)
            feedback=deque([((int(x),int(y)),int(z)) for x,y,z in data['feedback']],maxlen=128)
            rng.bit_generator.state=meta['rng']
            result,graph=meta['result'],meta['graph']
        phases=phases[-1:]
    def measure(key,order):
        np.testing.assert_allclose(sums,deployed.sum(axis=1),atol=1e-11,rtol=0)
        before=sha(weights.tobytes()+deployed.tobytes())
        result[key]=[score(stable,order),score(changed,order)]
        graph[key]=[graph_score(stable,order),graph_score(changed,order)]
        assert before==sha(weights.tobytes()+deployed.tobytes())
    for phase,count in phases:
        if phase=='change':
            fork_maps=sha(deployed.tobytes())
            fork_readout=sha(weights.tobytes())
            fork_eos=sha(eos.tobytes())
            fork_rng=sha(json.dumps(rng.bit_generator.state).encode())
            labels=updates=background=0
            query_trace=[]
        order=new if phase=='change' else old
        for event in range(count):
            if phase=='change' and event in (0,8,32,128):
                measure(f'change_{event}',order)
            if phase!='clean':
                seq=order[(int(rng.integers(12))+np.arange(16))%12].copy()
                if phase=='noise' or (phase=='change' and continued_noise):
                    bad=rng.random(16)<.10
                    seq[bad]=(seq[bad]+rng.integers(1,12,size=sum(bad)))%12
                history.append(seq)
                if phase=='change':
                    background += len(seq)
                learn(fast,deployed,seq,1,.12)
                if phase=='change':
                    updates += 28
            pair=train[int(rng.integers(len(train)))]
            ids,v,scores=predict(pair)
            if phase=='change':
                query_trace.append(pair)
                if arm!='joint':
                    continue
                labels += 1
            y=target(pair,order)
            scores[y]-=1
            weights[ids]-=.3*np.outer(v,scores)
            if phase!='change':
                eos[y,10]+=1
            feedback.append((pair,y))
        measure(phase,order)
        if checkpoint is not None and not resume and phase=='noise':
            meta=dict(job=job,continued_noise=continued_noise,source=sha(Path(__file__).read_bytes()),rng=rng.bit_generator.state,result=result,graph=graph)
            np.savez(checkpoint,fast=fast,deployed=deployed,weights=weights,eos=eos,sums=sums,
                     history=np.array(history),feedback=np.array([(x,y,z) for (x,y),z in feedback]),meta=json.dumps(meta))
            checkpoint.with_suffix('.sha').write_text(sha(checkpoint.read_bytes()))
            return None
    assert sha(eos.tobytes())==fork_eos
    if arm!='joint':
        assert labels==0 and sha(weights.tobytes())==fork_readout
    else:
        assert labels==512
    assert updates==14336 and sha(deployed.tobytes())!=fork_maps
    assert background==8192
    return dict(seed=seed,arm=arm,continued_noise=continued_noise,scores=result,graph=graph,
                background=background,updates=updates,labels=labels,
                fork=[fork_maps,fork_readout,fork_eos,fork_rng],
                final_maps=sha(deployed.tobytes()),
                queries=sha(json.dumps(query_trace).encode()),
                stream=sha(np.array(history).tobytes()),
                final=sha(fast.tobytes()+deployed.tobytes()+weights.tobytes()+eos.tobytes()+sums.tobytes()))

def execute(job):
    seed,arm,noisy=job
    return run((seed,arm),continued_noise=noisy)

if __name__=='__main__':
    import sys, subprocess, tempfile
    if len(sys.argv)>1:
        print(json.dumps(run((301,'normalized'),Path(sys.argv[1]),True)))
        sys.exit()
    roster=[(seed,arm,noisy) for seed in (301,302) for noisy in (False,True) for arm in ARMS]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(execute,roster))
    for i in range(0,len(rows),4):
        group=rows[i:i+4]
        assert all(r['fork'][0]==group[0]['fork'][0] and r['fork'][2:]==group[0]['fork'][2:] and r['final_maps']==group[0]['final_maps'] and r['queries']==group[0]['queries']
                   and r['stream']==group[0]['stream'] for r in group)
    import importlib.util
    prior=Path(__file__).resolve().parents[1]/'minifly-v193'/'qualify.py'
    spec=importlib.util.spec_from_file_location('v193_qualification',prior)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for row in rows:
        if row['arm']=='raw':
            ref=module.execute((row['seed'],'operator',row['continued_noise']))
            assert all(row[key]==ref[key] for key in ('scores','graph','fork','updates','labels','queries','stream'))
    with tempfile.TemporaryDirectory() as temp:
        checkpoint=Path(temp)/'state.npz'
        assert run((301,'normalized'),checkpoint) is None
        cmd=[sys.executable,'-B',__file__,str(checkpoint)]
        resumed=json.loads(subprocess.check_output(cmd))
        assert resumed==execute((301,'normalized',False))==rows[1]
        checkpoint.write_bytes(checkpoint.read_bytes()+b'bad')
        error=subprocess.run(cmd,capture_output=True,text=True)
        assert error.returncode and 'Checkpoint digest mismatch' in error.stderr
    report=dict(status='ENGINEERING_PASS',source_sha256=sha(Path(__file__).read_bytes()),
                cycle1='Incremental centroids match batch sums; raw arm matches V193; maps match across arms; exposure clamps passed',
                cycle2='16 full engineering lifetimes across clean and continued-noise conditions',
                cycle3='Four-worker to serial and fresh-process resume equality for normalized fixture, corruption rejected',
                rows=rows)
    Path(__file__).with_name('ENGINEERING.json').write_text(json.dumps(report,indent=2)+'\n')
