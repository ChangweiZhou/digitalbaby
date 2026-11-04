"""V190-R1 cue-assisted integration, independently reconstructed from Round 4."""
import argparse,copy,hashlib,json,os,pickle,random,resource,time,zipfile
from pathlib import Path
import numpy as np
from bytecyc4 import Relations,EOS,address,probabilities

POLICIES=('PREQ','REPEAT','UNIFORM','POS','NEG')
HELD={3,6,10,13}
DOMAIN=set(range(1,16))-{8}
START=time.monotonic()
def digest(data):return hashlib.sha256(data).hexdigest()
def write(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)

class Expert:
    def __init__(self,width,memory):
        self.weights=np.zeros((width,257),np.float32)
        self.memory=memory
    def active(self,h):
        h=h[-24:]
        if self.memory:
            return sorted({address(h[i:i+k]+bytes([s]),len(self.weights))
                           for k in (2,3) for i in range(len(h)-k+1) for s in range(4)}) or [0]
        return sorted({address(bytes([k])+h[-k:],len(self.weights)) for k in (1,2,3,4,8)})
    def read(self,h):
        return probabilities(self.weights[self.active(h)].mean(0))
    def teach(self,h,y):
        rows=self.active(h)
        error=-self.read(h)
        error[y]+=1
        self.weights[rows]+=4*error

class Integrated:
    def __init__(self,symbols,tags):
        self.pred=Expert(2048,False)
        self.fact=Expert(4096,True)
        self.rel=Relations(symbols,tags)
        self.records=self.targets=0

    def train(self,prompt,value=None,tag=None):
        raw=prompt if value is None else prompt+bytes([value])
        assert len(raw)<=127
        if value is not None and tag is not None:
            self.rel.teach(tag,prompt,value)
        for i,y in enumerate([*raw,EOS]):
            self.pred.teach(raw[:i],y)
            if value is not None and tag is None and i>=len(prompt):
                self.fact.teach(raw[:i],y)
        self.records+=1
        self.targets+=len(raw)+1

    def read(self,history,route,policy):
        if route is None:return self.pred.read(history)
        if route=='fact':return self.fact.read(history)
        return self.rel.read(route,history)[2][policy]

    def decode(self,prompt,route,policy):
        result=bytearray()
        for _ in range(8):
            y=int(self.read(prompt+result,route,policy).argmax())
            if y==EOS:return bytes(result),True
            result.append(y)
        return bytes(result),False

def fact_prompt(key,form):
    return (b'code '+key+b' is ',key+b' -> ',b'remember '+key+b': ',b'what is '+key+b'? ')[form]

def prepare(world,small):
    seed=int.from_bytes(hashlib.sha256(f'MINIFLY|V190|CUED-INTEGRATION|WORLD|{world}'.encode()).digest()[:8],'big')
    rng=random.Random(seed)
    symbols=rng.sample(range(128,224),16)
    tags=rng.sample(range(1,32),2)
    labels=[rng.sample(range(224,256),2) for _ in tags]
    keys=[bytes(rng.choices(b'abcdefghijklmnopqrstuvwxyz',k=6)) for _ in range(128)]
    assert len(set(keys))==128
    values=[65+i%16 for i in range(128)]
    rng.shuffle(values)
    edited=values.copy()
    for i in range(8):edited[i]=rng.choice([b for b in range(65,81) if b!=values[i]])
    reltrain=[[],[]];reltest=[[],[]]
    for t,tag in enumerate(tags):
        for a in range(16):
            for d in sorted(DOMAIN):
                b=(a+d)%16
                record=(bytes([symbols[a],symbols[b],61]), labels[t][int(d>=8) if t==0 else d%2],tag)
                (reltest if d in HELD else reltrain)[t].append(record)
        assert len(reltrain[t])==160 and len(reltest[t])==64
    stages={}
    for name,indices,relations,v in [('A',range(32),reltrain[0],values),
                                      ('B',range(32,128),reltrain[1],values),
                                      ('EDIT',range(8),[],edited),('NOEDIT',range(8),[],values)]:
        if small:
            indices=list(indices)[:4]
            relations=relations[:4]
        ordered=[]
        for rep in range(2):
            block=[(fact_prompt(keys[i],rep),v[i],None) for i in indices]+relations
            random.Random(seed+rep+sum(name.replace('NOEDIT','EDIT').encode())).shuffle(block)
            for index,record in enumerate(block):
                ordered.append(record)
                if index%4==3:
                    key=keys[list(indices)[(index//4)%len(indices)]]
                    ordered.append((b'notes mention '+key+b' in prose.',None,None))
        stages[name]=ordered
    taught_pairs={p for rows in reltrain for p,_,_ in rows}
    for rows in reltest:
        assert all(p not in taught_pairs and bytes([p[1],p[0],p[2]]) not in taught_pairs for p,_,_ in rows)
    # Matched correction histories differ only in answer bytes.
    assert [(p,t) for p,_,t in stages['EDIT']]==[(p,t) for p,_,t in stages['NOEDIT']]
    background=[bytes(symbols+[symbols[0]])]*8
    return symbols,tags,keys,values,edited,reltrain,reltest,stages,background

def panel(model,prompts,policy):
    counts=dict(n=0,byte=0,exact=0,premature=0,truncated=0,extra=0,payload_bits=0.,eos_bits=0.)
    for prompt,value,route in prompts:
        output,eos=model.decode(prompt,route,policy)
        counts['n']+=1
        counts['byte']+=int(bool(output) and output[0]==value)
        counts['exact']+=int(eos and output==bytes([value]))
        counts['premature']+=int(eos and not output)
        counts['truncated']+=int(not eos)
        counts['extra']+=int(len(output)>1)
        counts['payload_bits']-=float(np.log2(max(model.read(prompt,route,policy)[value],1e-30)))
        counts['eos_bits']-=float(np.log2(max(model.read(prompt+bytes([value]),route,policy)[EOS],1e-30)))
    return counts

def assess(model,keys,values,edited,relation,stage,small):
    fact={}
    expected=edited if stage=='EDIT' else values
    for form in (0,2,3):
        for name,indices,labels in [('old',range(32),expected),('changed_new',range(8),edited),
                                    ('obsolete',range(8),values),('stable',range(8,128),values)]:
            if small:indices=list(indices)[:3]
            fact[f'{form}_{name}']=panel(model,[(fact_prompt(keys[i],form),labels[i],'fact') for i in indices],'PREQ')
    panels={p:[panel(model,rs[:4] if small else rs,p) for rs in relation] for p in POLICIES}
    # One query per four prose records. Common prefix/prose contribution cancels.
    prose=b'ordinary notes mention '+keys[0]+b' again.'
    common=sum(-float(np.log2(max(model.pred.read(prose[:i])[b],1e-30))) for i,b in enumerate(prose))
    mixed={}
    queries=[row for group in relation for row in (group[:4] if small else group)]
    queries.extend((fact_prompt(keys[i],2),expected[i],'fact') for i in range(len(queries)))
    for policy in POLICIES:
        loss=0.
        n=0
        for prompt,y,route in queries:
            loss+=sum(-float(np.log2(max(model.pred.read(prompt[:i])[b],1e-30))) for i,b in enumerate(prompt))
            loss-=float(np.log2(max(model.read(prompt,route,policy)[y],1e-30)))
            n+=len(prompt)+1
        mixed[policy]=(loss+4*len(queries)*common)/(n+4*len(queries)*len(prose))
    return {'facts':fact,'relations':panels,'mixed_payload_bpb':mixed,
            'reliability_correct':model.rel.correct.tolist(),'reliability_total':model.rel.total.tolist()}

def references(model,tags,train,test,small):
    result=[]
    for tag,seen,held in zip(tags,train,test):
        records={prompt:y for prompt,y,_ in seen}
        majority=max(sorted(set(records.values())),key=list(records.values()).count)
        label_set=sorted(set(records.values()))
        def majority_of(values):
            return max(label_set,key=values.count) if values else majority
        def nearest(prompt):
            closest=min(records,key=lambda r:(sum(a!=b for a,b in zip(r,prompt)),r))
            return records[closest]
        rows=held[:4] if small else held
        policies={
            'frequency':lambda p:majority,
            'exact':lambda p:records.get(p,majority),
            'reverse':lambda p:records.get(bytes([p[1],p[0],p[2]]),majority),
            'left_marginal':lambda p:majority_of([y for r,y in records.items() if r[0]==p[0]]),
            'right_marginal':lambda p:majority_of([y for r,y in records.items() if r[1]==p[1]]),
            'nearest':nearest,
        }
        result.append({'n':len(rows),'byte_only':{name:sum(fn(p)==y for p,y,_ in rows) for name,fn in policies.items()},
                       'born':panel(model,rows,'PREQ')})
    return result

def checks(out):
    limits=(time.process_time()<900,time.monotonic()-START<1800,
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2**30,
            sum(p.stat().st_size for p in out.iterdir() if p.is_file())<2**28)
    if not all(limits):raise RuntimeError('resource cap exceeded')

def execute(out,small):
    out.mkdir(exist_ok=False)
    inputs={n:digest(Path(n).read_bytes()) for n in ('integ.py','bytecyc4.py')}
    prepared=[prepare(i,small) for i in range(3)]
    roster={str(i):digest(pickle.dumps(r,protocol=5)) for i,r in enumerate(prepared)}
    write(out/'LOCK.json',{'source':inputs,'world_roster':roster,'fixture':small})
    manifests=[]
    for world,(symbols,tags,keys,values,edited,reltrain,reltest,stages,background) in enumerate(prepared):
        model=Integrated(symbols,tags)
        files={}
        def snapshot(name):
            data=pickle.dumps(model,protocol=5)
            assert len(data)<=8*2**20
            p=out/f'{world}_{name}.pkl'
            p.write_bytes(data)
            files[p.name]=digest(data)
            return data
        snapshot('birth')
        for raw in background:
            model.rel.background(raw)
            model.train(raw)
        model.rel.frozen=True
        refs=references(model,tags,reltrain,reltest,small)
        reports={}
        records,targets=0,0
        b_state=None
        for name in ('A','B','EDIT','NOEDIT'):
            if name in ('EDIT','NOEDIT'):model=pickle.loads(b_state)
            for prompt,value,tag in stages[name]:
                model.train(prompt,value,tag)
            if name=='A':
                records,targets=model.records,model.targets
            if name=='B':
                records,targets=model.records,model.targets
            if name in ('EDIT','NOEDIT'):
                records+=len(stages[name])
                targets+=sum(len(p)+1+int(v is not None) for p,v,t in stages[name])
            data=snapshot(name)
            if name=='B':b_state=data
            reports[name]=assess(model,keys,values,edited,reltest,name,small)
            assert pickle.dumps(model,protocol=5)==data
            checks(out)
        if not small:
            assert records==1168 and len(model.rel.seen)==320
            assert np.all(model.rel.total==162)
            assert all(v==2 for v in model.rel.seen.values())
        fact_reference={}
        for phase in ('B','EDIT','NOEDIT'):
            observed={}
            for prior in ('A','B')+((phase,) if phase!='B' else ()):
                for prompt,value,tag in stages[prior]:
                    if value is not None and tag is None:observed[prompt]=value
            keytable={key:observed[fact_prompt(key,0)] for key in keys if fact_prompt(key,0) in observed}
            actual=edited if phase=='EDIT' else values
            fact_reference[phase]={str(form):{'n':len(keytable),
               'canonical_key':sum(keytable.get(k)==v for k,v in zip(keys,actual)),
               'trained_prefix':sum(observed.get(fact_prompt(k,form))==v for k,v in zip(keys,actual))}
               for form in (0,2,3)}
        edges={a:b for line in background for a,b in zip(line,line[1:])}
        rule_reference=[]
        for index,rows in enumerate(reltest):
            mapping={}
            for prompt,y,_ in reltrain[index]:
                a=prompt[0];distance=0
                while a!=prompt[1]:a=edges[a];distance+=1
                mapping[int(distance>=8) if index==0 else distance%2]=y
            correct=0
            for prompt,y,_ in rows:
                a=prompt[0];distance=0
                while a!=prompt[1]:a=edges[a];distance+=1
                correct+=mapping[int(distance>=8) if index==0 else distance%2]==y
            assert correct==len(rows)
            rule_reference.append(correct)
        receipt={'world':world,'records':records,'targets':targets,'checkpoints':files,'stages':reports,
                 'reference_byte_only':refs,'fact_reference_byte_only':fact_reference,'observed_cycle_ceiling':rule_reference}
        path=out/f'{world}_RECEIPT.json'
        write(path,receipt)
        manifests.append({'file':path.name,'sha256':digest(path.read_bytes()),'records':records,'targets':targets})
        print('committed world',world,records,targets,flush=True)
    decisions=[]
    for m in manifests:
        r=json.loads((out/m['file']).read_text())
        assert digest((out/m['file']).read_bytes())==m['sha256']
        for name,h in r['checkpoints'].items():assert digest((out/name).read_bytes())==h
        s=r['stages'];b=s['B'];edit=s['EDIT'];noedit=s['NOEDIT']
        def enough(row, fraction):return row['exact']>=fraction*row['n']
        decisions.append({
          'relation_complete':all(enough(row,.95) for stage in (b,edit) for row in stage['relations']['PREQ']),
          'fact_complete':all(enough(b['facts'][f'{f}_old'],.8) for f in (2,3)),
          'correction':all(edit['facts'][f'{f}_changed_new']['exact']>noedit['facts'][f'{f}_changed_new']['exact'] for f in (2,3)),
          'stable':all(edit['facts'][f'{f}_stable']['exact']>=b['facts'][f'{f}_stable']['exact'] for f in (2,3)),
          'mixed':b['mixed_payload_bpb']['PREQ']<=b['mixed_payload_bpb']['UNIFORM'],
          'comparison':{p:[a['exact']-z['exact'] for a,z in zip(b['relations']['PREQ'],b['relations'][p])] for p in ('REPEAT','UNIFORM')}})
    eligible=all(all(v for k,v in row.items() if k!='comparison') for row in decisions)
    for p in ('REPEAT','UNIFORM'):
        eligible &= all(min(row['comparison'][p])>=0 for row in decisions)
        eligible &= sum(max(row['comparison'][p])>0 for row in decisions)>=2
    write(out/'DECISION.json',{'status':'FIXTURE_COMPLETE' if small else
                               ('BOUNDED_DEV_POSITIVE' if eligible else 'DID_NOT_PASS'),
        'worlds':decisions,'commits':manifests,'records':sum(x['records'] for x in manifests),
        'targets':sum(x['targets'] for x in manifests),'cpu_seconds':time.process_time()})
    with zipfile.ZipFile(out/'RESULT_BUNDLE.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.glob('*.json')):z.write(p,p.name)
    with zipfile.ZipFile(out/'RESULT_BUNDLE.zip') as z:assert z.testzip() is None
    checks(out)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('out',type=Path)
    p.add_argument('--fixture',action='store_true')
    args=p.parse_args()
    execute(args.out,args.fixture)
