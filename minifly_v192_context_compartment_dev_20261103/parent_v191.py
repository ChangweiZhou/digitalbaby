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

def run(job, checkpoint=None, resume=False):
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
        z=np.tanh(projection@deployed[:,[x,y],:].ravel())
        ids=np.argsort(-np.abs(z),kind='stable')[:100]
        v=z[ids]/np.linalg.norm(z[ids])
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
            if meta['job']!=list(job) or meta['source']!=sha(Path(__file__).read_bytes()):
                raise ValueError('Checkpoint identity mismatch')
            fast,deployed,weights,eos=[data[n].copy() for n in ('fast','deployed','weights','eos')]
            history=deque(data['history'],maxlen=128)
            feedback=deque([((int(x),int(y)),int(z)) for x,y,z in data['feedback']],maxlen=128)
            rng.bit_generator.state=meta['rng']
            result,graph=meta['result'],meta['graph']
        phases=phases[-1:]
    def measure(key,order):
        before=sha(weights.tobytes()+deployed.tobytes())
        result[key]=[score(stable,order),score(changed,order)]
        graph[key]=[graph_score(stable,order),graph_score(changed,order)]
        assert before==sha(weights.tobytes()+deployed.tobytes())
    for phase,count in phases:
        order=new if phase=='change' else old
        for event in range(count):
            if phase=='change' and event in (0,32,128):
                measure(f'change_{event}',order)
            if phase!='clean':
                seq=order[(int(rng.integers(12))+np.arange(16))%12].copy()
                if phase=='noise':
                    bad=rng.random(16)<.10
                    seq[bad]=(seq[bad]+rng.integers(1,12,size=sum(bad)))%12
                history.append(seq)
                if arm!='frozen':
                    learn(fast,deployed,seq,.05 if arm=='slow' else 1,.006 if arm=='low_td' else .12)
            pair=train[int(rng.integers(len(train)))]
            ids,v,scores=predict(pair)
            y=target(pair,order)
            scores[y]-=1
            weights[ids]-=.3*np.outer(v,scores)
            eos[y,10]+=1
            feedback.append((pair,y))
        measure(phase,order)
        if checkpoint is not None and not resume and phase=='noise':
            meta=dict(job=job,source=sha(Path(__file__).read_bytes()),rng=rng.bit_generator.state,result=result,graph=graph)
            np.savez(checkpoint,fast=fast,deployed=deployed,weights=weights,eos=eos,
                     history=np.array(history),feedback=np.array([(x,y,z) for (x,y),z in feedback]),meta=json.dumps(meta))
            checkpoint.with_suffix('.sha').write_text(sha(checkpoint.read_bytes()))
            return None
    if arm=='frozen': assert np.array_equal(deployed,initial)
    if arm in ('direct','low_td'): assert np.allclose(fast,deployed)
    return dict(seed=seed,arm=arm,panels=[len(stable),len(changed)],scores=result,graph=graph,
                final=sha(fast.tobytes()+deployed.tobytes()+weights.tobytes()+eos.tobytes()))


if __name__=='__main__':
    import sys, subprocess, tempfile
    from statistics import mean,stdev
    from scipy.stats import t
    root=Path(__file__).resolve().parent
    if len(sys.argv)>1:
        print(json.dumps(run((301,'slow'),Path(sys.argv[1]),True)))
        sys.exit()
    with tempfile.TemporaryDirectory() as temp:
        ck=Path(temp)/'state.npz'
        assert run((301,'slow'),ck) is None
        cmd=[sys.executable,'-B',__file__,str(ck)]
        assert json.loads(subprocess.check_output(cmd))==run((301,'slow'))
        ck.write_bytes(ck.read_bytes()+b'bad')
        rejected=subprocess.run(cmd,capture_output=True,text=True)
        assert rejected.returncode and 'Checkpoint digest mismatch' in rejected.stderr
    seeds=list(range(20261103601,20261103613))
    keys=('change_0','change_32','change_128','change')
    plan=dict(seeds=seeds,arms=ARMS,source=sha(Path(__file__).read_bytes()),
              clean_floor=.85,noise_floor=.85,adaptation_floor=.85,
              contrast_margin=.05,adaptation_noninferiority=.05,checkpoints=keys)
    (root/'OFFICIAL_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(run,[(s,a) for s in seeds for a in ARMS]))
    by={a:[r['scores'] for r in rows if r['arm']==a] for a in ARMS}
    means={a:{k:[mean(r[k][i] for r in rr) for i in (0,1)] for k in rr[0]} for a,rr in by.items()}
    def stable(r):return mean(r[k][0] for k in keys)
    def adapt(r):return r['change'][1]
    def paired(a,b,f):
        d=[f(x)-f(y) for x,y in zip(by[a],by[b])]
        m=mean(d)
        h=float(t.ppf(.975,11))*stdev(d)/12**.5
        return [m,m-h,m+h]
    contrasts=dict(stability=paired('slow','direct',stable),adaptation=paired('slow','frozen',adapt),
                   adaptation_vs_direct=paired('slow','direct',adapt),stability_vs_low_td=paired('slow','low_td',stable))
    slow=means['slow']
    gate=(mean(slow['clean'])>=.85 and mean(slow['noise'])>=.85 and slow['change'][1]>=.85
          and all(contrasts[k][0]>=.05 and contrasts[k][1]>0 for k in ('stability','adaptation'))
          and contrasts['adaptation_vs_direct'][0]>=-.05)
    out=dict(gate=gate,means=means,contrasts=contrasts,rows=rows)
    (root/'OFFICIAL_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(gate=gate,means=means,contrasts=contrasts),indent=2))
