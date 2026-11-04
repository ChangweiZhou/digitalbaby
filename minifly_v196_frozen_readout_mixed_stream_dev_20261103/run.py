"""V196 frozen-head, paired observation-only comparison."""
import argparse,hashlib,io,itertools,json,math,os,resource,time,zipfile
from pathlib import Path
import numpy as np
from parent import parse
ARMS=('MIX','SOURCE','SHAM','FROZEN')
STAGES=('EDIT','NOEDIT')
POINTS=(0,8,32,128,512)
START=time.monotonic()
def sha(data):return hashlib.sha256(data).hexdigest()
def rng(i,role):
    return np.random.default_rng(int(sha(f'MINIFLY|V196|WORLD|{i}|{role}'.encode())[:16],16))
def save(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def resources(out):
    d={'cpu_seconds':time.process_time(),'wall_seconds':time.monotonic()-START,
       'rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       'output_bytes':sum(p.stat().st_size for p in out.iterdir() if p.is_file())}
    assert all(d[k]<=limit for k,limit in zip(d,(7200,10800,2**30,2**28))),d
    return d

class World:
    def __init__(self,i,fixture=False):
        self.i=i;self.fixture=fixture
        gen=rng(i,'structure')
        self.old=gen.permutation(12);self.new=self.old.copy()
        self.new[:2]=self.old[1::-1]
        self.positions=[np.argsort(x) for x in (self.old,self.new)]
        self.polarity=int(rng(i,'polarity').integers(2))
        self.held=[[],[]]
        for changed in (0,1):
            for y in (60,62):
                pool=[(a,b) for a,b in itertools.combinations(range(12),2)
                      if (self.y((a,b),0)!=self.y((a,b),1))==bool(changed) and self.y((a,b),0)==y]
                gen.shuffle(pool)
                for a,b in pool[:4]:self.held[changed]+=[(a,b),(b,a)]
            assert len(self.held[changed])==16
            assert all(sum(self.y(p,s)==60 for p in self.held[changed])==8 for s in (0,1))
        forbidden=set(sum(self.held,[]))
        self.train=[(a,b) for a in range(12) for b in range(12) if a!=b and (a,b) not in forbidden]
        assert all((b,a) not in forbidden for a,b in self.train)
        self.warm=[self.seq(self.old,k) for k in rng(i,'warm').integers(12,size=8 if fixture else 1200)]
        self.a=[self.train[k] for k in rng(i,'A_labels').integers(len(self.train),size=16 if fixture else 2048)]
        self.b=[]
        noise=rng(i,'B_corruption')
        for start,p in zip(rng(i,'B_obs').integers(12,size=8 if fixture else 512),
                           rng(i,'B_labels').integers(len(self.train),size=8 if fixture else 512)):
            s=self.seq(self.old,start);bad=noise.random(16)<.1
            s[bad]=(s[bad]+noise.integers(1,12,size=bad.sum()))%12
            self.b.append((s,self.train[p]))
        n=32 if fixture else 512
        self.slots=[]
        perm=rng(i,'distractor')
        for j,(start,first) in enumerate(zip(rng(i,'C_obs').integers(12,size=n),rng(i,'order').integers(2,size=n))):
            old=self.seq(self.old,start)
            while True:
                distractor=perm.permutation(old)
                if not any(np.array_equal(distractor,self.seq(order,k)) for order in (self.old,self.new) for k in range(12)):
                    break
            self.slots.append((old,self.seq(self.new,start),distractor,int(first)))
        self.sham=np.concatenate([rng(i,f'sham-{j}').permutation([True]*16+[False]*16) for j in range(n//32)])
        assert len(self.sham)==n and all(self.sham[j:j+32].sum()==16 for j in range(0,n,32))
    @staticmethod
    def seq(order,k):return order[(k+np.arange(16))%12]
    def y(self,p,phase):
        order=self.positions[phase]
        return 60+2*(int((order[p[0]]-order[p[1]])%2==0)^self.polarity)
    def locked(self):
        return {'old':self.old.tolist(),'new':self.new.tolist(),'polarity':self.polarity,'held':self.held,
                'warm':[x.tolist() for x in self.warm],'a':self.a,
                'b':[(x.tolist(),p) for x,p in self.b],
                'slots':[(a.tolist(),b.tolist(),c.tolist(),o) for a,b,c,o in self.slots],
                'sham':self.sham.tolist()}

def td(maps,row):
    for m,stride in enumerate((1,2)):
        for x,y in zip(row[:14],row[stride:]):
            d=(.65 if m==0 else .85)*maps[m,y]-maps[m,x]
            d[x]+=1
            maps[m,x]+=.12*d

class Model:
    def __init__(self,i):
        self.maps=np.zeros((2,12,12))
        self.project=rng(i,'projection').normal(size=(2000,48))
        self.weights=np.zeros((2000,256))
        self.eos=np.zeros((256,256),np.int64)
    def predict(self,p):
        x,y=parse(b'par '+bytes([p[0]+97,p[1]+97])+b'=')
        z=np.tanh(self.project@self.maps[:,[x,y],:].ravel())
        ids=np.argsort(-np.abs(z),kind='stable')[:100]
        v=z[ids]/np.linalg.norm(z[ids])
        return ids,v,v@self.weights[ids]
    def teach(self,p,y):
        ids,v,scores=self.predict(p)
        scores[y]-=1
        self.weights[ids]-=.3*np.outer(v,scores)
        self.eos[y,10]+=1
    def head_hash(self):
        return sha(self.weights.tobytes()+self.eos.tobytes()+self.project.tobytes())
    def state(self):
        b=io.BytesIO();np.savez(b,**vars(self));return b.getvalue()
    @classmethod
    def restore(cls,blob):
        m=object.__new__(cls)
        with np.load(io.BytesIO(blob),allow_pickle=False) as f:
            assert set(f.files)=={'maps','project','weights','eos'}
            for k in f.files:setattr(m,k,f[k])
        return m

def score(m,w,phase):
    result=[]
    for pairs in w.held:
        byte=joint=0
        for p in pairs:
            a=int(m.predict(p)[2].argmax());y=w.y(p,phase)
            byte+=a==y;joint+=a==y and int(m.eos[a].argmax())==10
        result.append({'n':len(pairs),'byte':byte,'byte_newline':joint})
    return result

def graph(counts,w,phase,feedback):
    best=None
    for ids in itertools.combinations(range(1,12),5):
        membership=np.zeros(12,bool);membership[[0,*ids]]=True
        cost=int(counts[membership[:,None]!=membership[None,:]].sum())
        if best is None or cost<best[0]:best=cost,membership
    g=best[1]
    votes=[int(g[a]==g[b])^((y-60)//2) for (a,b),y in feedback]
    polarity=int(sum(votes)>len(votes)/2)
    return [sum(60+2*(int(g[a]==g[b])^polarity)==w.y((a,b),phase) for a,b in panel) for panel in w.held]
def shortcuts(w,phase,feedback):
    labels=dict(feedback)
    def predict(p,kind):
        a,b=p
        pool=[y for (x,z),y in labels.items() if kind=='frequency' or
              (kind=='left' and a==x) or (kind=='right' and b==z) or
              (kind=='exact' and (x,z)==p) or (kind=='reverse' and (x,z)==(b,a))]
        pool=pool or list(labels.values())
        return 62 if pool.count(62)>pool.count(60) else 60
    return {k:[sum(predict(p,k)==w.y(p,phase) for p in panel) for panel in w.held]
            for k in ('frequency','left','right','exact','reverse')}
def common(w):
    m=Model(w.i);counts=np.zeros((12,12),int);feedback=[]
    for row in w.warm:td(m.maps,row);np.add.at(counts,(row[:-2],row[2:]),1)
    for p in w.a:m.teach(p,w.y(p,0));feedback.append((p,w.y(p,0)))
    a=score(m,w,0)
    for row,p in w.b:
        td(m.maps,row);np.add.at(counts,(row[:-2],row[2:]),1)
        m.teach(p,w.y(p,0));feedback.append((p,w.y(p,0)))
    return m,a,counts,feedback

def continuation(blob,w,arm,stage,counts,feedback):
    m=Model.restore(blob);head=m.head_hash()
    before=m.maps.copy()
    all_graph=counts.copy();source_graph=counts.copy()
    phase=int(stage=='EDIT')
    total=0;motion=0.;rows=np.zeros((2,12),int)
    result={}
    def checkpoint(i):
        assert m.head_hash()==head
        result[str(i)]={'score':score(m,w,phase),'maps_sha256':sha(m.maps.tobytes()),
                        'head_sha256':head,'graph_all':graph(all_graph,w,phase,feedback),
                        'graph_source':graph(source_graph,w,phase,feedback)}
    for i,(old,new,distractor,order) in enumerate(w.slots):
        if i in POINTS:checkpoint(i)
        structural=new if phase else old
        records=[(structural,True),(distractor,False)]
        if order:records.reverse()
        for row,is_source in records:
            if arm!='CLEAN' or is_source:np.add.at(all_graph,(row[:-2],row[2:]),1)
            if is_source:np.add.at(source_graph,(row[:-2],row[2:]),1)
            accepted=(arm=='MIX' or (arm in ('SOURCE','CLEAN') and is_source)
                      or (arm=='SHAM' and is_source==w.sham[i]))
            if accepted:
                previous=m.maps.copy();td(m.maps,row)
                for stride in range(2):np.add.at(rows[stride],row[:14],1)
                motion+=float(np.linalg.norm(m.maps-previous));total+=28
    checkpoint(len(w.slots))
    assert total==len(w.slots)*{'MIX':56,'SOURCE':28,'SHAM':28,'FROZEN':0,'CLEAN':28}[arm]
    if arm=='FROZEN':assert np.array_equal(before,m.maps)
    return {'checkpoints':result,'updates':total,'new_relation_labels':0,
            'source_row_writes':rows.tolist(),'map_motion':motion,'shortcuts':shortcuts(w,phase,feedback)}

def reduce(out,jobs,fixture):
    results={}
    for j in jobs:
        path=out/j['file'];assert sha(path.read_bytes())==j['sha256']
        r=json.loads(path.read_text())
        assert sha((out/f"{r['world']}_B.npz").read_bytes())==r['fork_sha256']
        assert all(r[s]['new_relation_labels']==0 for s in STAGES)
        results.setdefault(r['arm'],[]).append(r)
    assert len(jobs)==(8 if fixture else 32)
    d={'status':'FIXTURE_COMPLETE','jobs':jobs}
    if not fixture:
        def rates(a,s,t,p,kind):
            return np.array([r[s]['checkpoints'][str(t)]['score'][p][kind]/16 for r in results[a]])
        means={a:{s:{str(t):[[float(rates(a,s,t,p,k).mean()) for k in ('byte','byte_newline')]
                           for p in (0,1)] for t in POINTS} for s in STAGES} for a in ARMS}
        contrasts={}
        for a,b in [('SOURCE','FROZEN'),('SOURCE','MIX'),('SOURCE','SHAM')]:
            v=rates(a,'EDIT',128,1,'byte')-rates(b,'EDIT',128,1,'byte')
            half=2.364624251*float(v.std(ddof=1))/math.sqrt(len(v))
            contrasts[a+'-'+b]=[float(v.mean()),float(v.mean()-half),float(v.mean()+half)]
        def meets(k,x):return contrasts[k][0]>=x and contrasts[k][1]>0
        guards={'reference':means['SOURCE']['EDIT']['128'][0][1]>=.90 and means['SOURCE']['EDIT']['128'][1][1]>=.85 and meets('SOURCE-FROZEN',.10),
                'interference':meets('SOURCE-MIX',.05),'selectivity':meets('SOURCE-SHAM',.05)}
        guards['preservation']=all(float(rates('SOURCE',s,128,p,'byte').mean())>=.90 and
          float((rates('SOURCE',s,128,p,'byte')-rates('MIX',s,128,p,'byte')).mean())>=-.05
          for s,p in [('EDIT',0),('NOEDIT',0),('NOEDIT',1)])
        d.update(status='PASS' if all(guards.values()) else 'DID_NOT_PASS',guards=guards,means=means,contrasts=contrasts,
                 records=99712,sites=1531264,post_fork_relation_labels=0)
    d['cost']=resources(out);save(out/'DECISION.json',d)
    with zipfile.ZipFile(out/'RESULT_BUNDLE.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.glob('*.json')):z.write(p,p.name)
    with zipfile.ZipFile(out/'RESULT_BUNDLE.zip') as z:assert z.testzip() is None

def execute(out,fixture):
    worlds=[World(i,fixture) for i in range(2 if fixture else 8)]
    source=sha(Path(__file__).read_bytes())
    if not fixture:
        a=json.loads(Path('ADMISSION.json').read_text());assert a['status']=='PASS' and a['runner_sha256']==source
        assert a['world_hashes']==[sha(json.dumps(w.locked(),sort_keys=True).encode()) for w in worlds]
    out.mkdir(exist_ok=False)
    save(out/'LOCK.json',{'source':source,'parent':sha(Path('parent.py').read_bytes()),'fixture':fixture,
      'world_hashes':[sha(json.dumps(w.locked(),sort_keys=True).encode()) for w in worlds]})
    jobs=[]
    for w in worlds:
        m,a,counts,feedback=common(w);b=score(m,w,0);blob=m.state()
        assert len(blob)<=8*2**20 and Model.restore(blob).state()==blob
        (out/f'{w.i}_B.npz').write_bytes(blob)
        for arm in ARMS:
            r={'world':w.i,'arm':arm,'A':a,'B':b,'fork_sha256':sha(blob),'head_sha256':m.head_hash()}
            for s in STAGES:
                r[s]=continuation(blob,w,arm,s,counts,feedback)
                if fixture and arm=='SOURCE':
                    clean=continuation(blob,w,'CLEAN',s,counts,feedback)
                    for k,v in clean['checkpoints'].items():
                        assert v['maps_sha256']==r[s]['checkpoints'][k]['maps_sha256']
                        assert v['score']==r[s]['checkpoints'][k]['score']
            p=out/f'{w.i}_{arm}_RECEIPT.json';save(p,r)
            jobs.append({'file':p.name,'sha256':sha(p.read_bytes())})
            resources(out);print('committed',w.i,arm,flush=True)
    reduce(out,jobs,fixture)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--fixture',action='store_true')
    args=p.parse_args();execute(args.out,args.fixture)
