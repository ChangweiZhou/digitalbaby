"""V195-R1: recovered Round 6 fact address, distinct learned continuation address."""
import argparse, ast, hashlib, json, math, os, random, resource, time, zipfile
from pathlib import Path
import numpy as np

EOS=256
ARMS=('BASE','REOPEN','DECAY','BOTH','SHAM')
START=time.monotonic()
PARENT=Path('../round6_record_recovery/life_long2.py')
TREE=ast.parse(PARENT.read_text())
ENV={'np':np,'os':os,'WAVES':1,'n':12}
exec(compile(ast.Module(body=[n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name in ('bag','build') or
    isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('WORDS','CYC','TR','HO') for t in n.targets)],type_ignores=[]),
    str(PARENT),'exec'),ENV)
bag=ENV['bag'];FORMS=ENV['TR']+ENV['HO']
sha=lambda b:hashlib.sha256(b).hexdigest()
def seed(s):return int(sha(s.encode())[:16],16)
def write(path,obj):
    p=Path(path);t=p.with_suffix('.tmp')
    t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def resource_check(out):
    cpu=time.process_time()
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stored=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    assert cpu<=7200 and time.monotonic()-START<=10800 and rss<=2**30 and stored<=384*2**20
    return {'cpu_seconds':cpu,'wall_seconds':time.monotonic()-START,'rss_bytes':rss,'output_bytes':stored}

class Memory:
    def __init__(self,arm,seed_value,dtype=np.float16,with_eos=True,columns=257):
        self.arm=arm;self.rng=random.Random(seed_value)
        self.w=np.zeros((65536,columns),dtype)
        self.stop=np.zeros((256,257),dtype) if with_eos else None
        self.imp=np.zeros(65536);self.count=np.zeros(65536,np.uint32)
        self.triggers=self.empty=self.zero_quantized=self.dose_outside=self.dose_sites=0
        self.dose_max=0.;self.movement=0.
        self.records=self.targets=0
    def teach(self,rows,feature,target,actual_edit=False):
        old=self.w[rows].astype(np.float32)
        score=feature@old
        winner=int(score.argmax())
        e=-score;e[target]+=1
        raw=.6*np.outer(feature,e)
        base=raw/(1+self.imp[rows,None]/5)
        proposed=base.copy()
        if self.arm!='BASE' and winner not in (target,EOS) and score[winner]-np.partition(score,-2)[-2]>=.1:
            eligible=np.flatnonzero((self.count[rows]>=2)&(old.argmax(1)==winner))
            selected=sorted(eligible,key=lambda i:(self.count[rows[i]],rows[i]))[:math.ceil(len(eligible)/4)]
            if not selected:self.empty+=1
            else:
                self.triggers+=1
                if self.arm=='SHAM':selected=self.rng.sample(range(len(rows)),len(selected))
                if self.arm in ('REOPEN','BOTH','SHAM'):proposed[selected]=raw[selected]
                if self.arm in ('DECAY','BOTH','SHAM'):
                    magnitude=np.abs(base[selected,target])
                    proposed[selected,winner]-=magnitude;proposed[selected,target]+=magnitude
                strength=float(np.linalg.norm(base));norm=float(np.linalg.norm(proposed))
                proposed=proposed*strength/norm if strength and norm else base
                realized_ref=float(np.linalg.norm((old+base).astype(self.w.dtype).astype(np.float32)-old))
                stored=(old+proposed).astype(self.w.dtype)
                realized=float(np.linalg.norm(stored.astype(np.float32)-old))
                if realized_ref<=1e-5:self.zero_quantized+=1
                else:
                    gap=abs(realized/realized_ref-1)
                    self.dose_sites+=1;self.dose_max=max(gap,self.dose_max)
                    self.dose_outside+=gap>.01
        # The float32 reference rounds the update before addition, as in the recovered source.
        stored=(old+proposed.astype(np.float32)).astype(self.w.dtype)
        assert np.isfinite(stored).all()
        self.movement+=float(np.linalg.norm(stored.astype(np.float32)-old))
        self.w[rows]=stored
        self.imp[rows]+=np.abs(feature);self.count[rows]+=1
        if self.stop is not None:
            # Continuation addresses have their own 257-class scores. There is no forced stop.
            logits=self.stop[target].astype(np.float32)
            e=-logits;e[EOS]+=1
            self.stop[target]=(logits+.6*e).astype(self.stop.dtype)
        self.records+=1
        self.targets+=1+int(self.stop is not None)
    def decode(self,rows,feature):
        first=int((feature@self.w[rows].astype(np.float32)).argmax())
        if first==EOS:return b'',True
        if self.stop is None:return bytes([first]),False
        result=bytearray([first])
        for _ in range(7):
            guess=int(self.stop[result[-1]].argmax())
            if guess==EOS:return bytes(result),True
            result.append(guess)
        return bytes(result),False
    def state_hash(self):
        h=hashlib.sha256()
        for a in (self.w,self.stop,self.imp,self.count):
            if a is not None:h.update(memoryview(a).cast('B'))
        h.update(repr(self.rng.getstate()).encode())
        return h.hexdigest()

def bridge(out):
    stream,marks,keys,values,*_=ENV['build'](1,nf=128,reps=12)
    selected=[(bag(stream[max(0,i-16):i],65536),stream[i]) for i in range(len(stream))
              if marks[i] and stream[max(0,i-7):i-3]!=b'cmp ']
    assert len(selected)==1536
    models=[Memory('BASE',1,np.float32,False,256),Memory('BASE',1,np.float32),Memory('BASE',1)]
    for (rows,feat),target in selected:
        for model in models:model.teach(rows,feat,target)
    assert np.array_equal(models[0].w,models[1].w[:,:256])
    stats=[]
    for model in models:
        first=[];complete=[]
        for form in FORMS:
            results=[model.decode(*bag(form(key),65536)) for key in keys]
            first.append(sum(bool(p) and p[:1]==values[i] for i,(p,eos) in enumerate(results)))
            complete.append(sum(eos and p==values[i] for i,(p,eos) in enumerate(results)))
        stats.append({'first_byte_per_128':first,'complete_per_128':complete})
    qualified=all(c>=109 for c in stats[2]['complete_per_128'][:2]) and all(
        c>=r-2 for c,r in zip(stats[2]['first_byte_per_128'],stats[0]['first_byte_per_128']))
    result={'status':'PASS' if qualified else 'BRIDGE_UNQUALIFIED',
      'arms':['source-float32-byte','float32-split-EOS','float16-split-EOS'],'scores':stats,
      'technical_fact_presentations':len(selected)*len(models),'technical_fact_targets':len(selected)*5,
      'source_hash':sha(PARENT.read_bytes()),'measured':resource_check(out)}
    write(out/'BRIDGE.json',result)
    print(result,flush=True)

def roster(world,fixture):
    rng=random.Random(seed(f'V195-R1|WORLD|{world}'))
    size=8 if fixture else 128
    keys=[]
    while len(keys)<8*size:
        key=bytes(rng.choices(b'mnopqrstuvwxyz',k=5))
        if key not in keys:keys.append(key)
    val=[48+i%10 for i in range(len(keys))]
    rng.shuffle(val)
    corrections={i:rng.choice([c for c in range(48,58) if c!=val[i]])
                 for w in range(7) for i in range(w*size,w*size+size//4)}
    stream=[]
    for wave in range(8):
        new=list(range(wave*size,(wave+1)*size))
        edits=[i for i in corrections if i//size==wave-1]
        items=[(i,False) for i in new]+[(i,True) for i in edits]
        phase=[]
        for rep in range(4):
            order=items.copy();rng.shuffle(order)
            for j,(i,edited) in enumerate(order):
                prefix=FORMS[rep%2](keys[i])
                phase.append((prefix,corrections[i] if edited else val[i],edited))
                if j%4==3:
                    phase.append((b'notes mention '+keys[i]+b' in prose.',None,False))
        stream.append(phase)
    return keys,val,corrections,stream

class Predictor:
    """One shared noninteracting lag head, so its trajectory is computed once per world."""
    def __init__(self):
        self.w=np.zeros((2048,257),np.float32)
        self.loss=0.;self.sites=0
    def observe(self,record,is_filler):
        for i,y in enumerate([*record,EOS]):
            h=record[max(0,i-8):i]
            ids=sorted({seed(str(k)+h[-k:].hex())%2048 for k in (1,2,4,8)})
            logits=self.w[ids].mean(0)
            if is_filler and y!=EOS:
                exp=np.exp(logits-logits.max());self.loss-=math.log2(float(exp[y]/exp.sum()));self.sites+=1
            logits=-logits;logits[y]+=1;self.w[ids]+=.6*logits

def score(model,keys,values,corrections,wave,size,cache):
    before=model.state_hash()
    panels={}
    for f in range(4):
        result={}
        for i in range((wave+1)*size):
            changed=i in corrections and i//size<wave
            label=corrections[i] if changed else values[i]
            payload,eos=model.decode(*cache[f][i])
            first=payload[:1]
            groups=['all_current',f'wave_{i//size}']
            if not changed:groups+=['untouched']+(['oldest_untouched'] if i<size else [])
            if changed:groups+=['corrected',f'correction_{i//size}']+(['latest'] if i//size==wave-1 else [])
            for g in groups:
                p=result.setdefault(g,dict(n=0,exact=0,byte=0,obsolete=0,obsolete_complete=0,early=0,truncated=0,extra=0))
                p['n']+=1;p['byte']+=first==bytes([label]);p['exact']+=eos and payload==bytes([label])
                p['obsolete']+=changed and first==bytes([values[i]])
                p['obsolete_complete']+=changed and eos and payload==bytes([values[i]])
                p['early']+=eos and not payload;p['truncated']+=not eos;p['extra']+=len(payload)>1
        panels[str(f)]=result
    assert before==model.state_hash()
    return panels

def execute(out,fixture):
    count=1 if fixture else 4
    roster_data=[roster(i,fixture) for i in range(count)]
    expected=sum(len(s) for _,_,_,stream in roster_data for s in stream)*len(ARMS)
    sites=sum(len(p)+1+int(y is not None) for _,_,_,stream in roster_data for phase in stream for p,y,_ in phase)*len(ARMS)
    if not fixture:
        a=json.loads(Path('ADMISSION.json').read_text())
        assert a['status']=='PASS' and a['runner_sha256']==sha(Path(__file__).read_bytes())
        assert expected==124800 and expected+a['carried_records']<=135000 and sites+a['carried_targets']<=8800000
    write(out/'LOCK.json',{'source':sha(Path(__file__).read_bytes()),'parent':sha(PARENT.read_bytes()),
         'fixture':fixture,'roster':[sha(repr(d).encode()) for d in roster_data],'records':expected,'targets':sites})
    jobs=[]
    for i,(keys,vals,edits,stream) in enumerate(roster_data):
        cache=[[bag(form(k),65536) for k in keys] for form in FORMS]
        pred=Predictor()
        packed=[]
        for phase in stream:
            group=[]
            for p,y,is_edit in phase:
                pred.observe(p if y is None else p+bytes([y]),y is None)
                if y is not None:group.append((*bag(p,65536),y,is_edit))
            packed.append(group)
            resource_check(out)
        for arm in ARMS:
            m=Memory(arm,seed(f'{i}|sham'))
            stages=[]
            for w,phase in enumerate(packed):
                for rows,feat,y,edit in phase:m.teach(rows,feat,y,edit)
                stages.append({'scores':score(m,keys,vals,edits,w,8 if fixture else 128,cache),
                               'sha256':m.state_hash(),'triggers':m.triggers})
                resource_check(out)
            receipt={'world':i,'arm':arm,'stages':stages,'records':m.records,'targets':m.targets,
                     'dose':{'max_relative_error':m.dose_max,'outside_1pct':m.dose_outside,'sites':m.dose_sites,
                             'zero_quantized':m.zero_quantized,'total_motion':m.movement},
                     'predictor_filler_bpb':pred.loss/pred.sites}
            p=out/f'{i}_{arm}_RECEIPT.json'
            write(p,receipt);jobs.append({'file':p.name,'sha256':sha(p.read_bytes())})
            print('committed',i,arm,flush=True)
    reduce(out,jobs,fixture,expected,sites)
def reduce(out,jobs,fixture,expected,sites):
    rows={}
    for job in jobs:
        raw=(out/job['file']).read_bytes();assert sha(raw)==job['sha256']
        d=json.loads(raw);rows[d['world'],d['arm']]=d
    worlds=[]
    for i in range(1 if fixture else 4):
        result={}
        for f in ('2','3'):
            p={a:rows[i,a]['stages'][-1]['scores'][f] for a in ARMS}
            b,c=p['BASE'],p['BOTH']
            result[f]={'obsolete_cap':c['corrected']['obsolete']<=.01*c['corrected']['n'],
                'latest_all':c['latest']['exact']==c['latest']['n'],
                'corrected_preserved':c['corrected']['exact']>=b['corrected']['exact'],
                'all_current_floor':c['all_current']['exact']>=.94*c['all_current']['n'],
                'untouched_preserved':all(c[k]['exact']>=b[k]['exact'] for k in ('untouched','oldest_untouched')),
                'obsolete_complete_by_arm':{a:v['corrected']['obsolete_complete'] for a,v in p.items()}}
        worlds.append(result)
    comparisons={a:sum(v['obsolete_complete_by_arm'][a] for w in worlds for v in w.values()) for a in ARMS}
    guarded=all(all(v for k,v in p.items() if k!='obsolete_complete_by_arm') for w in worlds for p in w.values())
    dose_pass=all(r['dose']['outside_1pct']==0 for r in rows.values())
    decision={'status':'FIXTURE_COMPLETE' if fixture else ('PASS' if guarded and dose_pass and comparisons['BOTH']<min(comparisons['BASE'],comparisons['SHAM']) else 'DID_NOT_PASS'),
        'world_guards':worlds,'obsolete_complete_pooled':comparisons,'dose_guard':dose_pass,
        'jobs':jobs,'scheduled_records':expected,'scheduled_targets':sites,
        'actual_shared_predictor_records':expected//len(ARMS),
        'actual_fact_writes':sum(r['records'] for r in rows.values()),'cost':resource_check(out)}
    write(out/'DECISION.json',decision)
    with zipfile.ZipFile(out/'RESULT_BUNDLE.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.glob('*.json')):z.write(p,p.name)
    with zipfile.ZipFile(out/'RESULT_BUNDLE.zip') as z:assert z.testzip() is None

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['bridge','fixture','science']);p.add_argument('out',type=Path)
    args=p.parse_args();args.out.mkdir(exist_ok=False)
    if args.mode=='bridge':bridge(args.out)
    else:execute(args.out,args.mode=='fixture')
