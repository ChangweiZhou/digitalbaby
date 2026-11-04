"""MiniFly mixed-stream development and paired evaluation."""
import ctypes
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parent
FACT=ROOT.parent/'minifly-v197'/'round7_recovered.py'
REL=ROOT.parent/'minifly-v193'/'qualify.py'
SEEDS=tuple(range(20101,20109))

def sha(data):
    return hashlib.sha256(data).hexdigest()

def corpus():
    data=Path(os.environ['MINIFLY_ALICE']).read_bytes()
    text=data.split(b'*** START OF',1)[1].split(b'\n',1)[1].split(b'*** END OF',1)[0].strip()
    cut=len(text)*4//5
    # Visible control markers are escaped in all prose.
    return text[:cut],text[cut:]

def fact_schedule(seed, prose):
    s=FACT.read_text()
    code=s[s.index('WORDS='):s.index('def run(')]
    start=code.index('    TXT=')
    code=code[:start]+'''    TXT=PROSE_BYTES.replace(b'\\r',b'').replace(b'\\n',b' ').replace(b'\\x1e',b'\\x1e\\x1e')
'''+code[code.index('    pos=[0]',start):]
    code=code.replace('put(b\'cmp \'+bytes([CYC[x],CYC[y]])+b\'=\'); put(b\'>\' if (y-x)%n<6 else b\'<\',True); put(b\'\\n\')','pass')
    code=code.replace('                filler(); fact(', '                BOUNDARIES.append(len(S)); filler(); fact(')
    code=code.replace("put(TXT[pos[0]:pos[0]+k]+b' ');","PROSE_RANGES.append((len(S),len(S)+len(TXT[pos[0]:pos[0]+k])+1)); put(TXT[pos[0]:pos[0]+k]+b' ');")
    env=dict(np=np,os=os,G4=True,WAVES=8,PROSE=True,PROSE_BYTES=prose,BOUNDARIES=[],PROSE_RANGES=[])
    exec(compile(code,'V197_fact_schedule','exec'),env)
    return env,env['build'](seed,nf=128)

def relation_schedule(seed):
    # Instrument the V193 operator to collect its original causal inputs.
    code=REL.read_text().split("if __name__")[0]
    code=code.replace('def run(job, checkpoint=None, resume=False, continued_noise=False):',
                      'def run(job, tape, checkpoint=None, resume=False, continued_noise=False):')
    code=code.replace('    def learn(fast,deployed,run,tau,rate):','    def learn(fast,deployed,run,tau,rate):\n        tape.append(("S",bytes(np.asarray(run+97,dtype=np.uint8))))')
    code=code.replace('        before=sha(weights.tobytes()+deployed.tobytes())',
                      '        tape.append(("M",(key,[(p,target(p,order)) for p in stable],[(p,target(p,order)) for p in changed])))\n        before=sha(weights.tobytes()+deployed.tobytes())')
    code=code.replace('        if phase==\'change\':\n            fork_maps=', '        if phase==\'change\':\n            tape.append(("X",None))\n            fork_maps=')
    code=code.replace('                    continue\n                labels += 1','                    tape.append(("Q",bytes([pair[0]+97,pair[1]+97])))\n                    continue\n                labels += 1')
    code=code.replace('            scores[y]-=1','            tape.append(("R",bytes([pair[0]+97,pair[1]+97,y])))\n            scores[y]-=1')
    code=code.replace('    weights=np.zeros((2000,256))','    tape.append(("I",projection.copy()))\n    weights=np.zeros((2000,256))')
    ns={}
    exec(compile(code,'V193_instrumented','exec'),ns)
    tape=[]
    result=ns['run']((seed,'operator'),tape)
    return tape,result

def features(data, name, initial=b''):
    import hashlib
    width=7 if name=='v92' else 10
    matrix=np.full((len(data),width),-1,np.int32)
    @lru_cache(maxsize=65536)
    def address(ctx):
        if name=='round8':
            ids=set()
            for k in range(1,6):
                g=hash(ctx[-k:])
                ids.update((g*2654435761+j*40503+k*97)%65536 for j in range(2))
            return sorted(ids)
        ids=[ctx[-1]] if ctx else []
        for depth,first,buckets in ((2,256,256),(3,768,512),(4,1792,512)):
            if len(ctx)>=depth:
                h=hashlib.blake2s(b'V9.2-context-pilot-20260925'+bytes([depth])+ctx[-depth:],digest_size=8).digest()
                ids.extend((first+int.from_bytes(h[:4],'little')%buckets,
                            first+buckets+int.from_bytes(h[4:],'little')%buckets))
        return ids
    pre=initial[-5:]
    raw=pre+data
    for i in range(len(data)):
        j=i+len(pre)
        ids=address(raw[max(0,j-5):j])
        matrix[i,:len(ids)]=ids
    return matrix

def predictor(data, test, tags, name, replay=False):
    clock=time.perf_counter()
    weights=np.zeros((2816 if name=='v92' else 65536,256),np.float32)
    bias=np.zeros(256,np.float32)
    fn=ctypes.CDLL(str(ROOT/'predictor.dylib')).advance
    fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,
                 ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
    fn.restype=None
    def feed(text, idx, train):
        output=np.empty(len(text),np.float64)
        raw=np.frombuffer(text,np.uint8)
        fn(raw.ctypes.data,idx.ctypes.data,len(raw),idx.shape[1],weights.ctypes.data,
           bias.ctypes.data,name=='v92',train,output.ctypes.data)
        return output
    rows=features(data,name,b'     ' if name=='round8' else b'')
    losses=feed(data,rows,True)
    digest=sha(weights.tobytes()+bias.tobytes())
    suffix=feed(test,features(test,name,data[-5:]),False)
    assert digest==sha(weights.tobytes()+bias.tobytes())
    if replay:
        weights.fill(0)
        bias.fill(0)
        alone=feed(data,rows,True)
        assert np.array_equal(losses,alone) and digest==sha(weights.tobytes()+bias.tobytes())
    return dict(online=float(losses.mean()),heldout=float(suffix.mean()),state=digest,
                by_category={label:float(losses[tags==i].mean()) for i,label in
                             enumerate(('prose','syntax_structure','answer'))},
                predictor_bytes=weights.nbytes+bias.nbytes,feature_scratch_bytes=rows.nbytes,
                seconds=time.perf_counter()-clock,deterministic_replay=replay)

def assemble(seed, prose):
    env,(raw,teach,keys,initial,revised,corr,_,_,wave_ends,_)=fact_schedule(seed,prose)
    tape,reference=relation_schedule(seed)
    assert len(env['BOUNDARIES'])==(1024+224)*12
    base=np.ones(len(raw),np.uint8)
    for lo,hi in env['PROSE_RANGES']:
        base[lo:hi]=0
    base[np.flatnonzero(teach)]=2
    boundaries=env['BOUNDARIES']+[len(raw)]
    chunks=[]
    t=0
    for i,(lo,hi) in enumerate(zip(boundaries,boundaries[1:])):
        target=(i+1)*len(tape)//(len(boundaries)-1)
        for kind,value in tape[t:target]:
            if kind=='S':
                wire=b'\x1eS'+value+b'\n'
            elif kind in ('Q','R'):
                wire=b'\x1epar '+value[:2]+b'='+ (value[2:]+b'\n' if kind=='R' else b'\n')
            else:
                wire=b''
            tag=np.ones(len(wire),np.uint8)
            if kind=='R': tag[-2]=2
            chunks.append((kind,wire,tag,value))
        t=target
        chunks.append(('F',raw[lo:hi],base[lo:hi],None))
        if hi in wave_ends.values():
            wave=next(w for w,end in wave_ends.items() if end==hi)
            if wave:
                chunks.append(('E',b'',np.empty(0,np.uint8),wave-1))
    tail=prose[:4096].replace(b'\x1e',b'\x1e\x1e')
    chunks.append(('P',tail,np.zeros(len(tail),np.uint8),None))
    assert t==len(tape)
    data=b''.join(c[1] for c in chunks)
    assert all(form(k) not in data for k in keys for form in env['HO'])
    return chunks,data,np.concatenate([c[2] for c in chunks]),(env,keys,initial,revised,set(corr)),reference

def replay(chunks, fact, active):
    env,keys,initial,revised,corr=fact
    fact_on='fact' in active
    rel_on='relation' in active
    mem=np.zeros((65536,256),np.float32) if fact_on else None
    imp=np.zeros(65536) if fact_on else None
    @lru_cache(maxsize=8192)
    def address(context):
        return env['bag'](context,65536)
    maps=np.zeros((2,12,12))
    deploy=maps.copy()
    eye=np.eye(12)
    w=np.zeros((2000,256))
    eos=np.zeros((256,256),np.int64)
    projection=None
    history=b''
    measures={}
    immediate={}
    switched=False
    labels=updates=0
    def predict(pair):
        z=np.tanh(projection@deploy[:,[pair[0],pair[1]],:].ravel())
        idx=np.argsort(-np.abs(z),kind='stable')[:100]
        v=z[idx]/np.linalg.norm(z[idx])
        return idx,v,v@w[idx]
    for kind,wire,tags,meta in chunks:
        if fact_on and kind=='F':
            joined=history+wire
            for j in np.flatnonzero(tags==2):
                j=int(j)
                end=len(history)+j
                idx,v=address(joined[max(0,end-16):end])
                y=wire[j]
                assert 48<=y<=57
                err=-(v@mem[idx])
                err[y]+=1
                scale=1/(1+imp[idx]/5)
                mem[idx]+=(.6*np.outer(scale*v/(v@v),err)).astype(np.float32)
                imp[idx]+=np.abs(v)
        if fact_on and kind=='E':
            ids=[i for i in sorted(corr) if i//128==meta]
            def is_correct(j,i):
                a,v=address(env['HO'][j%2](keys[i]))
                return int(np.argmax(v@mem[a])==revised[i][0])
            immediate[str(meta)]=mean(is_correct(j,i) for j,i in enumerate(ids))
        if fact_on and wire:
            history=(history+wire)[-16:]
        if not rel_on:
            continue
        if kind=='I':
            projection=meta
        elif kind=='X':
            switched=True
            fork=sha(w.tobytes()+eos.tobytes())
        elif kind=='S':
            assert wire.startswith(b'\x1eS') and len(wire)==19 and wire.endswith(b'\n')
            seq=np.frombuffer(wire[2:-1],np.uint8)-97
            for m,stride in enumerate((1,2)):
                for x,y in zip(seq[:14],seq[stride:]):
                    row=maps[m,x]
                    row+=.12*(eye[x]+(.65 if m==0 else .85)*maps[m,y]-row)
                    deploy[m,x]+=(row-deploy[m,x])
            updates+=28 if switched else 0
        elif kind in ('Q','R'):
            assert wire[:5]==b'\x1epar ' and wire[7:8]==b'='
            x,y=(c-97 for c in wire[5:7])
            idx,v,score=predict((x,y))
            if kind=='R':
                assert not switched and len(wire)==10
                label=wire[8]
                score[label]-=1
                w[idx]-=.3*np.outer(v,score)
                eos[label,10]+=1
            else:
                assert switched and len(wire)==9
        elif kind=='M':
            key,*panels=meta
            state=sha(w.tobytes()+eos.tobytes()+deploy.tobytes())
            def correct(p,target):
                answer=int(np.argmax(predict(p)[2]))
                return int(answer==target and np.argmax(eos[answer])==10)
            measures[key]=[mean(correct(p,y) for p,y in panel) for panel in panels]
            assert state==sha(w.tobytes()+eos.tobytes()+deploy.tobytes())
    result={}
    if fact_on:
        panels=[]
        for wave in range(8):
            row=dict(valid=0,corrected=0,obsolete=0,valid_n=0,corrected_n=0)
            for corrected in (False,True):
                ids=(i for i in range(wave*128,(wave+1)*128) if (i in corr)==corrected)
                for j,i in enumerate(ids):
                    ctx=env['HO'][j%2](keys[i])
                    a,v=address(ctx)
                    emitted=int(np.argmax(v@mem[a]))
                    if corrected:
                        row['corrected_n']+=1
                        row['corrected']+=int(emitted==revised[i][0])
                        row['obsolete']+=int(emitted==initial[i][0])
                    else:
                        row['valid_n']+=1
                        row['valid']+=int(emitted==initial[i][0])
            panels.append(row)
        def rate(key,n): return sum(p[key] for p in panels)/sum(p[n] for p in panels)
        assert len(immediate)==7
        result['fact']=dict(panels=panels,immediate_correction=immediate,kept=rate('valid','valid_n'),
                            oldest=panels[0]['valid']/panels[0]['valid_n'],
                            durable_correction=rate('corrected','corrected_n'),
                            obsolete=rate('obsolete','corrected_n'),
                            state=sha(mem.tobytes()+imp.tobytes()),array_bytes=mem.nbytes+imp.nbytes)
    if rel_on:
        assert fork==sha(w.tobytes()+eos.tobytes()) and updates==14336
        result['relation']=dict(scores=measures,post_change_updates=updates,post_change_labels=labels,
                                state=sha(maps.tobytes()+deploy.tobytes()+w.tobytes()+eos.tobytes()),
                                array_bytes=sum(a.nbytes for a in (maps,deploy,w,eos,projection)))
    return result

def world(seed):
    import resource
    start=time.perf_counter()
    prose,test=corpus()
    chunks,data,tags,fact,reference=assemble(seed,prose)
    integrated=replay(chunks,fact,('fact','relation'))
    for name in ('fact','relation'):
        assert integrated[name]==replay(chunks,fact,(name,))[name]
    assert integrated['relation']['state']==reference['final']
    assert integrated['relation']['scores']==reference['scores']
    pred={name:predictor(data,test,tags,name,replay=True) for name in ('v92','round8')}
    r=dict(seed=seed,integrated=integrated,predictors=pred,train_bytes=len(data),test_bytes=len(test),
           mixed_sha256=sha(data),seconds=time.perf_counter()-start,
           worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    dest=ROOT/f'world-{seed}.json'
    tmp=dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(r,indent=2)+'\n')
    tmp.replace(dest)
    print(seed,len(data),integrated['fact']['kept'],integrated['relation']['scores']['change_128'],
          {k:round(v['heldout'],3) for k,v in pred.items()},flush=True)
    return r

if __name__=='__main__':
    assert os.environ.get('PYTHONHASHSEED')=='0'
    if sys.argv[1:]==['dev']:
        world(20100)
        sys.exit()
    assert sys.argv[1:]==['official'] and (ROOT/'world-20100.json').is_file()
    plan=dict(seeds=SEEDS,fact_floors=[.90,.85,.90],relation_floors=[.90,.85],
              predictor_hash_seed=0,source={p.name:sha(p.read_bytes()) for p in
              (ROOT/'run.py',ROOT/'predictor.c',FACT,REL)},corpus=sha(Path(os.environ['MINIFLY_ALICE']).read_bytes()))
    (ROOT/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(world,SEEDS))
    facts={k:mean(r['integrated']['fact'][k] for r in rows)
           for k in ('kept','oldest','durable_correction','obsolete')}
    rel={k:[mean(r['integrated']['relation']['scores'][k][i] for r in rows)
            for i in (0,1)] for k in rows[0]['integrated']['relation']['scores']}
    predictors={k:{m:mean(r['predictors'][k][m] for r in rows) for m in ('online','heldout')}
                for k in ('v92','round8')}
    gate=all(a>=b for a,b in zip(list(facts.values())[:3],plan['fact_floors'])) and all(
        a>=b for a,b in zip(rel['change_128'],plan['relation_floors']))
    out=dict(gate=gate,facts=facts,relation=rel,predictors=predictors,rows=rows,plan=plan)
    (ROOT/'RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
