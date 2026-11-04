"""Official author-disjoint V9.2 dose study. QA uses its separate admission check."""
from concurrent.futures import ProcessPoolExecutor
import ctypes
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, stdev
import sys
import time

import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent/'minifly-integration'))
import run as reference
TRAIN=(2600,2554,28054,996,1184,135,1257,1399,600,1155,1400,766,730,2807)
TEST=(1342,345,2701,84,1260,174,35,76)
ORDERS=(24051,24052,24053)
DOSE=2**20
ARMS=('zero','one','ten','shuffled')
SMOOTH=.05

def sha(raw):return hashlib.sha256(raw).hexdigest()

def read_book(book):
    raw=(ROOT/'books'/f'{book}.txt').read_bytes()
    normalized=raw.replace(b'\r\n',b'\n').replace(b'\r',b'\n')
    split=re.split(rb'(?m)^\*\*\* START OF [^\n]*\*\*\*\s*\n',normalized,maxsplit=1)
    assert len(split)==2,book
    body=re.split(rb'(?m)^\*\*\* END OF [^\n]*\*\*\*',split[1],maxsplit=1)
    assert len(body)==2,book
    body=body[0].strip()
    def field(key): return re.search(rb'(?m)^'+key+rb': ([^\n]+)',split[0])[1].decode().strip()
    return body,dict(id=book,url=f'https://www.gutenberg.org/ebooks/{book}',
                     author=field(b'Author'),title=field(b'Title'),raw_sha256=sha(raw),
                     normalized_sha256=sha(body),bytes=len(body))

def segments(seed,books):
    order=np.random.default_rng(seed).permutation(TRAIN).tolist()
    remaining=10*DOSE
    result=[]
    for b in order:
        take=min(remaining,len(books[b]))
        result.append((b,take))
        remaining-=take
        if not remaining:break
    assert not remaining
    return result

def native():
    fn=ctypes.CDLL(str(reference.ROOT/'predictor.dylib')).advance
    fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,
                 ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
    fn.restype=None
    return fn

def blank():return np.zeros((2816,256),np.float32),np.zeros(256,np.float32)
def state(model):return sha(model[0].tobytes()+model[1].tobytes())

def process(fn,model,raw,train,history=b'',idx=None):
    data=np.frombuffer(raw,np.uint8)
    if idx is None:
        idx=reference.features(raw,'v92',history)
    loss=np.empty(len(raw),np.float64)
    fn(data.ctypes.data,idx.ctypes.data,len(raw),7,model[0].ctypes.data,model[1].ctypes.data,
       1,train,loss.ctypes.data)
    return float(loss.sum())

def stream(fn,model,raw,train):
    loss=0.
    for i in range(0,len(raw),65536):
        loss+=process(fn,model,raw[i:i+65536],train,raw[max(0,i-4):i])
    return loss

def prepare_order(seed):
    import resource
    clock=time.perf_counter()
    books={b:read_book(b)[0] for b in TRAIN}
    fn=native()
    intact,shuffled=blank(),blank()
    rng=np.random.default_rng(seed+80000)
    seen=0
    order=segments(seed,books)
    for b,n in order:
        raw=books[b][:n]
        if seen<DOSE<=seen+n:
            cut=DOSE-seen
            stream(fn,intact,raw[:cut],True)
            np.savez(ROOT/f'{seed}-one.npz',w=intact[0],bias=intact[1],bytes=DOSE)
            for i in range(cut,len(raw),65536):
                process(fn,intact,raw[i:i+65536],True,raw[max(0,i-4):i])
        else:
            stream(fn,intact,raw,True)
        shuffled_raw=rng.permutation(np.frombuffer(raw,np.uint8)).tobytes()
        stream(fn,shuffled,shuffled_raw,True)
        seen+=n
    for arm,model in (('ten',intact),('shuffled',shuffled)):
        np.savez(ROOT/f'{seed}-{arm}.npz',w=model[0],bias=model[1],bytes=seen)
    print('trained',seed,seen,flush=True)
    return dict(seed=seed,segments=order,seconds=time.perf_counter()-clock,
                rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def packed(raw):
    b=np.frombuffer(raw,np.uint8).astype(np.uint32)
    return (b[:-3]<<24)|(b[1:-2]<<16)|(b[2:-1]<<8)|b[3:]

def fit_counts():
    books={b:read_book(b)[0] for b in TRAIN}
    result=[]
    for seed in ORDERS:
        material=[books[b][:n] for b,n in segments(seed,books)]
        keys,counts=np.unique(np.concatenate([packed(b) for b in material]),return_counts=True)
        contexts,starts=np.unique(keys>>8,return_index=True)
        totals=np.add.reduceat(counts,starts)
        freq=sum(np.bincount(np.frombuffer(b,np.uint8),minlength=256) for b in material)
        result.append((keys,counts,contexts,totals,freq))
    return result

def lookup(need,keys,values):
    i=np.searchsorted(keys,need)
    i=np.minimum(i,len(keys)-1)
    return np.where(keys[i]==need,values[i],0)

def evaluate(book):
    raw,metadata=read_book(book)
    fn=native()
    models={}
    for seed in ORDERS:
        for arm in ARMS[1:]:
            with np.load(ROOT/f'{seed}-{arm}.npz',allow_pickle=False) as ck:
                models[seed,arm]=(ck['w'],ck['bias'])
    hashes={k:state(v) for k,v in models.items()}
    loss={k:0. for k in models}
    for i in range(0,len(raw),65536):
        block=raw[i:i+65536]
        idx=reference.features(block,'v92',raw[max(0,i-4):i])
        for k,model in models.items():
            loss[k]+=process(fn,model,block,False,idx=idx)
    assert hashes=={k:state(v) for k,v in models.items()}
    rows=[dict(seed=s,**{arm:loss[s,arm]/len(raw) for arm in ARMS[1:]},zero=8.) for s in ORDERS]
    print('scored',book,round(mean(r['ten'] for r in rows),3),flush=True)
    return dict(book=metadata,rows=rows)

def overlap(training,tests):
    tokenize=lambda b: re.findall(rb'[a-z]+',b.lower())
    # Look for verbatim overlap using every seventh 24-word evaluation window
    # against every training window. Flag books with substantial shared text.
    eval_sets={}
    for book,b in tests.items():
        w=tokenize(b)
        eval_sets[book]={b' '.join(w[i:i+24]) for i in range(0,len(w)-23,7)}
    requested=set.union(*eval_sets.values())
    matches=set()
    for b in training.values():
        w=tokenize(b)
        matches.update(s for i in range(len(w)-23) if (s:=b' '.join(w[i:i+24])) in requested)
    return {b:len(s&matches)/len(s) for b,s in eval_sets.items()}

def interval(d):
    avg=mean(d)
    half=float(t.ppf(.975,len(d)-1))*stdev(d)/len(d)**.5
    return [avg,avg-half,avg+half]

if __name__=='__main__':
    allbooks={b:read_book(b) for b in TRAIN+TEST}
    assert not ({allbooks[b][1]['author'] for b in TRAIN}&{allbooks[b][1]['author'] for b in TEST})
    training={b:allbooks[b][0] for b in TRAIN}
    tests={b:allbooks[b][0] for b in TEST}
    audit=overlap(training,tests)
    assert max(audit.values())<=.0025,audit
    plan=dict(version='MiniFly prose pretraining',orders=ORDERS,train=TRAIN,test=TEST,
              doses=[0,DOSE,10*DOSE],sources={p.name:sha(p.read_bytes()) for p in
                 (Path(__file__),reference.ROOT/'run.py',reference.ROOT/'predictor.c',reference.ROOT/'predictor.dylib')},
              books=[m for _,m in allbooks.values()],overlap=audit,threshold=.05,
              segments={s:segments(s,training) for s in ORDERS})
    (ROOT/'OFFICIAL_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=3) as pool:
        runtime=list(pool.map(prepare_order,ORDERS))
        rows=list(pool.map(evaluate,TEST))
    counts=fit_counts()
    for row in rows:
        raw=tests[row['book']['id']]
        p=packed(raw)
        data=np.frombuffer(raw,np.uint8)
        for item,(keys,vals,ctx,totals,freq) in zip(row['rows'],counts):
            prob=(lookup(p,keys,vals)+SMOOTH)/(lookup(p>>8,ctx,totals)+256*SMOOTH)
            base=(freq+SMOOTH)/(freq.sum()+256*SMOOTH)
            item['markov3']=float((-np.log2(prob).sum()-np.log2(base[data[:3]]).sum())/len(raw))
            item['frequency']=float(-np.log2(base[data]).mean())
    cols=(*ARMS,'markov3','frequency')
    by=[{c:mean(r[c] for r in book['rows']) for c in cols} for book in rows]
    macro={c:mean(r[c] for r in by) for c in cols}
    weighted={c:sum(r[c]*b['book']['bytes'] for r,b in zip(by,rows))/sum(b['book']['bytes'] for b in rows) for c in cols}
    comparisons={c:interval([r[c]-r['ten'] for r in by]) for c in ('one','shuffled','markov3')}
    gate=all(comparisons[c][0]>=.05 and comparisons[c][1]>0 for c in ('one','shuffled'))
    out=dict(gate=gate,macro_bpb=macro,weighted_bpb=weighted,advantages=comparisons,
             runtime=runtime,rows=rows,qa_admitted=json.loads((ROOT/'QA_ADMISSION.json').read_text())['admitted'])
    (ROOT/'OFFICIAL_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
    print(macro,comparisons,gate)
