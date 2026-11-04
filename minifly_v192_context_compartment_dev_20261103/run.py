"""V192 paired development campaign. All learned updates are local."""
import argparse
import hashlib
import io
import json
import math
import os
import random
import resource
import time
import zipfile
from itertools import combinations
from pathlib import Path
import numpy as np

ARMS=('DIRECT','CONTEXT','SHAM','ORACLE')
EOS=256
START=time.monotonic()
CLOCKS=(0,32,128,512)
def sha(b):return hashlib.sha256(b).hexdigest()
def seed(text):return int(sha(text.encode())[:16],16)
def save(p,data):
    temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    os.replace(temp,p)
def counters(out):
    return {'CPU_seconds':time.process_time(),'wall_seconds':time.monotonic()-START,
            'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'output_bytes':sum(p.stat().st_size for p in out.glob('*') if p.is_file())}
def check(out):
    values=counters(out)
    for k,cap in zip(values,(7200,10800,2**30,2**28)):
        if values[k]>cap:raise RuntimeError(f'{k}: {values[k]} > {cap}')
    return values

class World:
    def __init__(self,i,fixture=False):
        self.i=i
        self.seed=seed(f'MINIFLY|V192|WORLD|{i}')
        rng=np.random.default_rng(self.seed)
        self.old=rng.permutation(12)
        self.new=self.old.copy()
        self.new[:2]=self.old[1::-1]
        self.polarity=int(rng.integers(2))
        self.pos=[np.argsort(x) for x in (self.old,self.new)]
        self.panels=[[],[]]
        for panel in (0,1):
            for label in (60,62):
                pool=[(a,b) for a,b in combinations(range(12),2)
                      if int(self.label((a,b),0)!=self.label((a,b),1))==panel and self.label((a,b),0)==label]
                rng.shuffle(pool)
                for a,b in pool[:4]:self.panels[panel].extend([(a,b),(b,a)])
            assert len(self.panels[panel])==16
            for phase in (0,1):assert sum(self.label(p,phase)==60 for p in self.panels[panel])==8
        held=set(sum(self.panels,[]))
        self.train=[(a,b) for a in range(12) for b in range(12) if a!=b and (a,b) not in held]
        assert all((b,a) not in held for a,b in self.train)
        assert any(self.label(p,0)!=self.label(p,1) for p in self.train)
        self.warm=[self.cycle(self.old,int(k)) for k in rng.integers(12,size=8 if fixture else 1200)]
        self.a=[self.train[int(k)] for k in rng.integers(len(self.train),size=16 if fixture else 2048)]
        self.episodes={}
        for stage in ('B','EDIT','NOEDIT'):
            # Same draw stream for the two C forks, with structural order as the only observation difference.
            r=np.random.default_rng(seed(str(self.seed)+('B' if stage=='B' else 'C')))
            items=[]
            for _ in range(16 if fixture else 512):
                start=int(r.integers(12))
                structural=self.cycle(self.new if stage=='EDIT' else self.old,start)
                # Same symbol multiset as the stable-cycle window, with order disturbed.
                distractor=self.cycle(self.old,start).copy()
                r.shuffle(distractor)
                tracks=[(structural,True),(distractor,False)]
                if int(r.integers(2)):tracks.reverse()
                pair=self.train[int(r.integers(len(self.train)))]
                items.append((tracks,pair))
            self.episodes[stage]=items
    @staticmethod
    def cycle(order,start):return order[(start+np.arange(16))%12].copy()
    def label(self,p,phase):
        pos=self.pos[phase]
        return 60+2*(int((pos[p[0]]-pos[p[1]])%2==0)^self.polarity)
    def digest(self):
        content={'old':self.old.tolist(),'new':self.new.tolist(),'polarity':self.polarity,
            'warm':[r.tolist() for r in self.warm],'a':self.a,'panels':self.panels,
            'episodes':{k:[([(r.tolist(),b) for r,b in tracks],p) for tracks,p in stage]
                        for k,stage in self.episodes.items()}}
        return sha(json.dumps(content,sort_keys=True).encode())

def parse(raw):
    assert len(raw)>=7 and raw[:4]==b'par ' and raw[6]==61
    x,y=raw[4]-97,raw[5]-97
    assert 0<=x<12 and 0<=y<12
    return x,y,raw[7:]

class Model:
    def __init__(self,seed_value):
        rng=np.random.default_rng(seed_value)
        self.project=rng.normal(size=(2000,48)).astype(np.float32)
        # The post-output byte enters the same head through a fixed raw-byte suffix feature.
        self.suffix=rng.normal(size=(2000,256)).astype(np.float32)
        self.maps=np.zeros((2,2,12,12),np.float32)
        self.weights=np.zeros((2000,257),np.float32)
        self.graph=np.zeros((12,12),np.int64)
        self.oracle_graph=np.zeros_like(self.graph)
    def observe(self,record,assign,provenance):
        for m,stride in enumerate((1,2)):
            for i,(x,y) in enumerate(zip(record[:14],record[stride:])):
                bank=int(not assign[m,i])
                row=self.maps[bank,m,x]
                delta=(.65 if m==0 else .85)*self.maps[bank,m,y]-row
                delta[x]+=1
                row+=.12*delta
                if m==1:
                    self.graph[x,y]+=1
                    if provenance:self.oracle_graph[x,y]+=1
    def prediction(self,raw):
        x,y,generated=parse(raw)
        z=self.project@self.maps[0][:,[x,y],:].ravel()
        if generated:z=z+3*self.suffix[:,generated[-1]]
        z=np.tanh(z)
        ix=np.argsort(-np.abs(z),kind='stable')[:100]
        v=z[ix]
        v=v/max(float(np.linalg.norm(v)),1e-12)
        return ix,v,v@self.weights[ix]
    def teach(self,prompt,value):
        for context,target in ((prompt,value),(prompt+bytes([value]),EOS)):
            ix,v,score=self.prediction(context)
            error=-score
            error[target]+=1
            self.weights[ix]+=.3*np.outer(v,error)
    def decode(self,prompt):
        payload=bytearray()
        for _ in range(8):
            pred=int(self.prediction(prompt+payload)[2].argmax())
            if pred==EOS:return bytes(payload),True
            payload.append(pred)
        return bytes(payload),False
    def state(self):
        stream=io.BytesIO()
        np.savez(stream,project=self.project,suffix=self.suffix,maps=self.maps,
                 weights=self.weights,graph=self.graph,oracle_graph=self.oracle_graph)
        return stream.getvalue()
    @classmethod
    def restore(cls,data):
        model=object.__new__(cls)
        with np.load(io.BytesIO(data),allow_pickle=False) as archive:
            for k in ('project','suffix','maps','weights','graph','oracle_graph'):setattr(model,k,archive[k])
        return model

def query(pair):return b'par '+bytes([pair[0]+97,pair[1]+97])+b'='

def grams(prefix):
    # Byte IDs plus a fixed polynomial binning. No generator labels.
    for n in (2,3):
        for i in range(len(prefix)-n+1):
            value=n
            for x in prefix[i:i+n]:value=value*31+int(x)+97
            yield value%4096

def context_table(world):
    counts=np.zeros(4096,np.uint16)
    for row in world.warm:
        for j in grams(row):counts[j]=min(65535,int(counts[j])+1)
    return np.log((counts.astype(float)+1)/(int(counts.sum())+4096)*4096)

def raw_scores(table,record):
    result=np.zeros((2,14))
    for m,stride in enumerate((1,2)):
        for i in range(14):
            bins=list(grams(record[max(0,i+stride-8):i+stride]))
            result[m,i]=sum(table[b] for b in bins)
    return result

def threshold():
    w=World('CALIBRATION')
    table=context_table(w)
    scores=np.array([raw_scores(table,record) for tracks,_ in w.episodes['B'] for record,_ in tracks])
    return float(np.quantile(scores,.5))

def masks(w,table,cut,stage):
    rows=w.episodes[stage]
    shape=(len(rows),2,2,14)
    ctx=np.array([[raw_scores(table,record)>=cut for record,_ in tracks] for tracks,_ in rows])
    oracle=np.array([[np.full((2,14),source) for _,source in tracks] for tracks,_ in rows])
    sham=ctx.copy()
    r=np.random.default_rng(seed(f'{w.seed}|{stage.replace("NOEDIT","EDIT")}|sham'))
    for j in range(0,len(rows),32):
        for stride in range(2):
            view=sham[j:j+32,:,stride,:]
            perm=r.permutation(view.ravel()).reshape(view.shape)
            assert view.sum()==perm.sum()
            view[:]=perm
    assert ctx.shape==shape and np.array_equal(ctx.sum(axis=(0,1,3)),sham.sum(axis=(0,1,3)))
    return {'DIRECT':np.ones(shape,bool),'CONTEXT':ctx,'SHAM':sham,'ORACLE':oracle}

def measure(model,w,phase):
    before=sha(model.state())
    panels=[]
    for pairs in w.panels:
        n=byte=exact=early=truncated=extra=obsolete=0
        for pair in pairs:
            y=w.label(pair,phase)
            response,eos=model.decode(query(pair))
            n+=1
            byte+=bool(response) and response[0]==y
            exact+=eos and response==bytes([y])
            early+=eos and not response
            truncated+=not eos
            extra+=len(response)>1
            obsolete+=bool(phase) and y!=w.label(pair,0) and eos and response==bytes([w.label(pair,0)])
        panels.append(dict(n=n,byte=byte,exact=exact,early=early,truncated=truncated,extra=extra,obsolete=obsolete))
    assert before==sha(model.state())
    return panels

def graph_scores(w,phase,graph,history):
    best=None
    for members in combinations(range(1,12),5):
        group=np.zeros(12,bool)
        group[[0,*members]]=True
        cross=int(graph[group[:,None]!=group[None,:]].sum())
        if best is None or cross<best[0]:best=cross,group
    group=best[1]
    # Label mapping uses only feedback actually observed by this checkpoint.
    votes=[int(group[x]==group[y])^((value-60)//2) for (x,y),value in history[-128:]]
    polarity=int(sum(votes)>len(votes)/2)
    return [sum(60+2*(int(group[x]==group[y])^polarity)==w.label((x,y),phase) for x,y in panel)
            for panel in w.panels]

def shortcuts(w,phase):
    labels={p:w.label(p,phase) for p in w.train}
    # Majority backoff for unseen exact/reverse pairs and empty marginals.
    def predict(p,mode):
        a,b=p
        pool=[y for (x,z),y in labels.items() if mode=='constant' or
              (mode=='left' and a==x) or (mode=='right' and b==z) or
              (mode=='exact' and (x,z)==p) or (mode=='reverse' and (x,z)==(b,a))]
        pool=pool or list(labels.values())
        return 62 if pool.count(62)>pool.count(60) else 60
    return {mode:[sum(predict(p,mode)==w.label(p,phase) for p in panel) for panel in w.panels]
            for mode in ('constant','left','right','exact','reverse')}

def acquisition(w):
    model=Model(w.seed)
    mask=np.ones((2,14),bool)
    for r in w.warm:model.observe(r,mask,True)
    for p in w.a:model.teach(query(p),w.label(p,0))
    return model

def run(out,fixture,cut):
    out.mkdir(exist_ok=False)
    worlds=[World(i,fixture) for i in range(2 if fixture else 8)]
    source=sha(Path(__file__).read_bytes())
    if not fixture:
        admission=json.loads(Path('ADMISSION.json').read_text())
        assert admission['status']=='PASS' and admission['runner_sha256']==source and admission['threshold']==cut
    write_lock={'source_sha256':source,'threshold':cut,'roster':{str(w.i):w.digest() for w in worlds},
                'fixture':fixture,'arms':ARMS}
    save(out/'LOCK.json',write_lock)
    jobs=[]
    for w in worlds:
        model=acquisition(w)
        a_state=model.state()
        assert len(a_state)<=8*2**20
        ap=out/f'{w.i}_A.npz'
        ap.write_bytes(a_state)
        a=measure(model,w,0)
        table=context_table(w)
        routes={s:masks(w,table,cut,s) for s in w.episodes}
        for arm in ARMS:
            model=Model.restore(a_state)
            report={'A':a,'TD_counts':{},'A_sha256':sha(a_state)}
            b_state=None
            feedback_history=[(p,w.label(p,0)) for p in w.a]
            for stage in ('B','EDIT','NOEDIT'):
                phase=int(stage=='EDIT')
                if stage!='B':
                    model=Model.restore(b_state)
                    history=list(b_history)
                else:history=feedback_history
                outcomes={}
                for i,(tracks,pair) in enumerate(w.episodes[stage]):
                    if i in CLOCKS:
                        outcomes[str(i)]=measure(model,w,phase)
                    for r,(record,source) in enumerate(tracks):
                        model.observe(record,routes[stage][arm][i,r],source)
                    model.teach(query(pair),w.label(pair,phase))
                    history.append((pair,w.label(pair,phase)))
                    if i%32==31:check(out)
                outcomes[str(len(w.episodes[stage]))]=measure(model,w,phase)
                report[stage]=outcomes
                report['TD_counts'][stage]=routes[stage][arm].sum(axis=(0,1,3)).tolist()
                report['TD_counts'][stage+'_eligible']=[len(w.episodes[stage])*28]*2
                report['TD_counts'][stage+'_selected_structural']=sum(
                    routes[stage][arm][i,r].sum().item()
                    for i,(tracks,_) in enumerate(w.episodes[stage]) for r,(_,tag) in enumerate(tracks) if tag)
                report[f'{stage}_final_sha256']=sha(model.state())
                if stage=='B':
                    b_state=model.state()
                    b_history=list(history)
                    assert len(b_state)<=8*2**20
                    (out/f'{w.i}_{arm}_B.npz').write_bytes(b_state)
                report[f'{stage}_graph']={'all':graph_scores(w,phase,model.graph,history),
                    'provenance':graph_scores(w,phase,model.oracle_graph,history),
                    'limitation':'label mapping uses the last 128 observed feedback labels at phase end'}
            report['shortcuts']={str(s):shortcuts(w,s) for s in (0,1)}
            file=out/f'{w.i}_{arm}_RECEIPT.json'
            save(file,report)
            jobs.append({'world':w.i,'arm':arm,'receipt':file.name,'sha256':sha(file.read_bytes())})
            print('committed',w.i,arm,flush=True)
    reduce(out,jobs,fixture)
    check(out)

def reduce(out,jobs,fixture):
    rows={}
    for job in jobs:
        path=out/job['receipt']
        assert sha(path.read_bytes())==job['sha256']
        w,arm=job['world'],job['arm']
        row=json.loads(path.read_text())
        assert sha((out/f'{w}_A.npz').read_bytes())==row['A_sha256']
        assert sha((out/f'{w}_{arm}_B.npz').read_bytes())==row['B_final_sha256']
        rows.setdefault(arm,[]).append(row)
    def rate(row,stage,checkpoint,panel,metric='exact'):
        p=row[stage][str(checkpoint)][panel]
        return p[metric]/p['n']
    def series(arm,stage,checkpoint,panel,metric='exact'):
        return [rate(r,stage,checkpoint,panel,metric) for r in rows[arm]]
    decision={'status':'FIXTURE_COMPLETE','jobs':jobs}
    if not fixture:
        assert len(jobs)==32 and all(len(v)==8 for v in rows.values())
        means={a:{s:{str(c):[[float(np.mean(series(a,s,c,p,m))) for m in ('byte','exact')] for p in (0,1)]
                     for c in CLOCKS} for s in ('EDIT','NOEDIT')} for a in ARMS}
        contrasts={}
        for x,y in [('ORACLE','DIRECT'),('CONTEXT','DIRECT'),('CONTEXT','SHAM')]:
            d=np.array(series(x,'EDIT',32,1))-series(y,'EDIT',32,1)
            margin=2.364624251*np.std(d,ddof=1)/math.sqrt(len(d))
            contrasts[x+'-'+y]=[float(d.mean()),float(d.mean()-margin),float(d.mean()+margin)]
        oracle=contrasts['ORACLE-DIRECT'][0]
        c=contrasts['CONTEXT-DIRECT']
        preservation=all(
            (lambda x,y:x>=.85 and x-y>=-.05)(float(np.mean(series('CONTEXT',s,t,p))),
                                               float(np.mean(series('DIRECT',s,t,p))))
            for s in ('EDIT','NOEDIT') for t in (32,512) for p in ((0,) if s=='EDIT' else (0,1)))
        passed=oracle>=.05 and c[1]>0 and contrasts['CONTEXT-SHAM'][1]>0 and c[0]>=.5*oracle and preservation
        decision.update(status='PASS' if passed else 'DID_NOT_PASS',means=means,contrasts=contrasts,
                        records=173440,targets=2424192,guards={'oracle_headroom':oracle>=.05,
                        'paired_gain':c[1]>0 and contrasts['CONTEXT-SHAM'][1]>0,
                        'half_oracle_gain':c[0]>=.5*oracle,'preservation':preservation})
    decision['cost']=counters(out)
    save(out/'DECISION.json',decision)
    with zipfile.ZipFile(out/'RESULT_BUNDLE.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.glob('*.json')):z.write(p,p.name)
    with zipfile.ZipFile(out/'RESULT_BUNDLE.zip') as z:assert z.testzip() is None

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--fixture',action='store_true')
    parser.add_argument('--threshold',type=float,required=True)
    args=parser.parse_args()
    run(args.output,args.fixture,args.threshold)
