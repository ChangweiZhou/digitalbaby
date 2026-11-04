"""Run the three resident models through an explicit byte protocol."""
import ctypes
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import resource
from statistics import mean
import sys
import time

import numpy as np

import qualify
ROOT=Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent/'minifly-integration'))
import run as prior

ESC=b'\x1e'
qualify.LENGTHS[ord('D')]=17

def literal(data):
    return data.replace(ESC,ESC*2)

def frame(kind,payload):
    return ESC+kind.encode()+payload

def sha(*arrays):
    return hashlib.sha256(b''.join(a.tobytes() for a in arrays)).hexdigest()

class Live:
    def __init__(self,bag,projection,active=True):
        self.address=lru_cache(maxsize=8192)(lambda data:bag(data,65536))
        self.context=b''
        self.parser=qualify.Parser()
        self.mem=np.zeros((65536,256),np.float32) if active else None
        self.imp=np.zeros(65536) if active else None
        self.maps=np.zeros((2,12,12))
        self.deploy=self.maps.copy()
        self.eye=np.eye(12)
        self.projection=projection
        self.rel=np.zeros((2000,256))
        self.eos=np.zeros((256,256),np.int64)
        self.w=np.zeros((2816,256),np.float32)
        self.bias=np.zeros(256,np.float32)
        self.hist=b''
        self.loss_sum=0.
        self.bytes=0
        self.updates=dict(fact=0,labels=0,td=0)
        self.out=[]
        self.active=active
        self.fn=ctypes.CDLL(str(prior.ROOT/'predictor.dylib')).advance
        self.fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,
                          ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
        self.fn.restype=None
        self.address_pred=lru_cache(maxsize=16384)(lambda ctx:prior.features(b'\x00','v92',ctx)[0])

    def predict_bytes(self,data,train):
        if not data: return np.empty(0)
        raw=np.frombuffer(data,np.uint8)
        features=np.empty((len(raw),7),np.int32)
        hist=self.hist
        text=hist+data
        for i in range(len(data)):
            j=len(hist)+i
            features[i]=self.address_pred(text[max(0,j-4):j])
        loss=np.empty(len(raw),np.float64)
        self.fn(raw.ctypes.data,features.ctypes.data,len(raw),7,self.w.ctypes.data,
                self.bias.ctypes.data,1,train,loss.ctypes.data)
        if train:
            self.hist=text[-4:]
            self.loss_sum+=float(loss.sum())
            self.bytes+=len(raw)
        return loss

    def read_relation(self,payload):
        x,y=(c-97 for c in payload[:2])
        z=np.tanh(self.projection@self.deploy[:,[x,y],:].ravel())
        ids=np.argsort(-np.abs(z),kind='stable')[:100]
        v=z[ids]/np.linalg.norm(z[ids])
        return ids,v,v@self.rel[ids]

    def receive(self,data,marker_at=None,probe=False):
        if not probe:
            self.predict_bytes(data,True)
        if not self.active:
            return
        for i,byte in enumerate(data):
            event=self.parser.feed(byte,i==marker_at)
            if event is None:
                continue
            kind,payload=event
            payload=bytes.fromhex(payload)
            if kind=='P':
                self.context=(self.context+payload)[-16:]
            elif kind=='T':
                ids,v=self.address(self.context)
                y=payload[0]
                e=-(v@self.mem[ids])
                e[y]+=1
                self.mem[ids]+=(.6*np.outer((1/(1+self.imp[ids]/5))*v/(v@v),e)).astype(np.float32)
                self.imp[ids]+=np.abs(v)
                self.context=(self.context+payload)[-16:]
                self.updates['fact']+=1
            elif kind=='D':
                if not 1<=payload[0]<=16 or any(payload[1+payload[0]:]):
                    raise ValueError('Invalid factual query length')
                ids,v=self.address(payload[1:1+payload[0]])
                self.out.append(bytes([int(np.argmax(v@self.mem[ids]))]))
            elif kind=='S':
                seq=np.frombuffer(payload,np.uint8)-97
                for m,stride in enumerate((1,2)):
                    for x,y in zip(seq[:14],seq[stride:]):
                        row=self.maps[m,x]
                        row+=.12*(self.eye[x]+(.65 if m==0 else .85)*self.maps[m,y]-row)
                        self.deploy[m,x]+=(row-self.deploy[m,x])
                self.updates['td']+=28
            elif kind in ('Q','R'):
                ids,v,scores=self.read_relation(payload)
                if kind=='Q':
                    a=int(np.argmax(scores))
                    self.out.append(bytes([a,int(np.argmax(self.eos[a]))]))
                else:
                    y=payload[-1]
                    scores[y]-=1
                    self.rel[ids]-=.3*np.outer(v,scores)
                    self.eos[y,10]+=1
                    self.updates['labels']+=1
            else:
                raise ValueError('Unsupported frame')

def world(seed):
    start=time.perf_counter()
    prose,suffix=prior.corpus()
    chunks,_,_,fact,reference=prior.assemble(seed,prose)
    env,keys,initial,revised,corr=fact
    projection=next(meta for kind,_,_,meta in chunks if kind=='I')
    live=Live(env['bag'],projection)
    wire_digest=hashlib.sha256()
    physical=[]
    scores={}
    checkpoints={}
    latencies=[]
    def submit(data,marker=None):
        wire_digest.update(data)
        physical.append(data)
        live.receive(data,marker)
        if data.startswith(ESC+b'Q'):
            assert len(live.out)==1 and len(live.out[0])==2
        else:
            assert not live.out
        live.out.clear()
    def ask(kind,data):
        count=len(live.out)
        start=time.perf_counter_ns()
        live.receive(frame(kind,data),probe=True)
        latencies.append(time.perf_counter_ns()-start)
        assert len(live.out)==count+1
        return live.out.pop()
    def measure(panels):
        state=sha(live.rel,live.eos,live.deploy)
        value=[mean(ask('Q',bytes([p[0]+97,p[1]+97]))==bytes([y,10]) for p,y in panel)
               for panel in panels]
        assert state==sha(live.rel,live.eos,live.deploy)
        return value
    active_start=time.perf_counter()
    for kind,wire,tags,meta in chunks:
        if kind in ('F','P'):
            indices=np.flatnonzero(tags==2) if kind=='F' else []
            pos=0
            for j in indices:
                j=int(j)
                submit(literal(wire[pos:j]))
                submit(frame('T',wire[j:j+1]),2)
                pos=j+1
            submit(literal(wire[pos:]))
        elif kind=='S':
            submit(frame('S',wire[2:-1]))
        elif kind=='R':
            assert 'fork' not in checkpoints
            submit(frame('R',wire[5:7]+wire[8:9]),4)
        elif kind=='Q':
            submit(frame('Q',wire[5:7]))
        elif kind=='X':
            checkpoints['fork']=sha(live.rel,live.eos)
            checkpoints['fork_updates']=live.updates.copy()
        elif kind=='M':
            name,*panels=meta
            scores[name]=measure(panels)
    live.parser.finish()
    assert live.updates['labels']==checkpoints['fork_updates']['labels']
    assert live.updates['td']-checkpoints['fork_updates']['td']==14336
    assert live.updates['fact']==14976
    assert sha(live.rel,live.eos)==checkpoints['fork']
    assert sha(live.maps,live.deploy,live.rel,live.eos)==reference['final']
    assert scores==reference['scores']
    panels=[]
    state=sha(live.mem,live.imp)
    for wave in range(8):
        counts=dict(valid=0,valid_n=0,corrected=0,corrected_n=0)
        for fixed in (False,True):
            ids=[i for i in range(wave*128,(wave+1)*128) if (i in corr)==fixed]
            for j,i in enumerate(ids):
                raw=env['HO'][j%2](keys[i])[-16:]
                key='corrected' if fixed else 'valid'
                counts[key+'_n']+=1
                counts[key]+=int(ask('D',bytes([len(raw)])+raw.ljust(16,b'\0'))==(revised if fixed else initial)[i])
        panels.append(counts)
    assert state==sha(live.mem,live.imp)
    e2e_seconds=time.perf_counter()-active_start
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    frozen=live.predict_bytes(suffix,False)
    pred_state=sha(live.w,live.bias)
    from_disk=json.loads((prior.ROOT/f'world-{seed}.json').read_text())
    if state!=from_disk['integrated']['fact']['state']:
        raise AssertionError('Factual state differs from tagged replay')
    # Compare the native streaming V9.2 weights with an independent batch pass.
    alone=Live(env['bag'],projection,active=False)
    baseline_start=time.perf_counter()
    for packet in physical:
        alone.receive(packet)
    baseline_seconds=time.perf_counter()-baseline_start
    assert pred_state==sha(alone.w,alone.bias) and alone.loss_sum==live.loss_sum
    assert np.array_equal(frozen,alone.predict_bytes(suffix,False))
    def rate(k,selected):
        return sum(p[k] for p in selected)/sum(p[k+'_n'] for p in selected)
    row=dict(seed=seed,facts=dict(kept=rate('valid',panels),oldest=rate('valid',panels[:1]),
             durable=rate('corrected',panels),earlier_durable=rate('corrected',panels[:6])),
             relation=scores,updates=live.updates,post_change_labels=0,panels=panels,
             v92=dict(online=live.loss_sum/live.bytes,heldout=float(frozen.mean()),state=pred_state),
             input_sha256=wire_digest.hexdigest(),bytes=live.bytes,e2e_seconds=e2e_seconds,
             throughput=live.bytes/e2e_seconds,predictor_only_throughput=live.bytes/baseline_seconds,peak_rss_through_live_bytes=rss,
             query_p95_ms=float(np.percentile(latencies,95))/1e6,seconds=time.perf_counter()-start)
    (ROOT/f'world-{seed}.json').write_text(json.dumps(row,indent=2)+'\n')
    print(seed,row['facts'],scores['change_128'],round(row['throughput']),flush=True)
    return row

if __name__=='__main__':
    if sys.argv[1:]==['dev']:
        world(20100)
        sys.exit()
    assert sys.argv[1:]==['official'] and (ROOT/'world-20100.json').exists() and (ROOT/'RESUME_CHECK.json').exists()
    plan=dict(seeds=prior.SEEDS,source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              parser=hashlib.sha256((ROOT/'qualify.py').read_bytes()).hexdigest(),
              fact_floors=[.90,.85,.90],relation_floors=[.90,.85],
              corpus=hashlib.sha256(Path(os.environ['MINIFLY_ALICE']).read_bytes()).hexdigest(),
              predictor_native_source=hashlib.sha256((prior.ROOT/'predictor.c').read_bytes()).hexdigest(),
              predictor_binary=hashlib.sha256((prior.ROOT/'predictor.dylib').read_bytes()).hexdigest(),
              reference_source=hashlib.sha256((prior.ROOT/'run.py').read_bytes()).hexdigest())
    (ROOT/'OFFICIAL_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(world,prior.SEEDS))
    fact={k:mean(r['facts'][k] for r in rows) for k in rows[0]['facts']}
    rel=[mean(r['relation']['change_128'][i] for r in rows) for i in range(2)]
    gate=all(a>=b for a,b in zip(fact.values(),plan['fact_floors'])) and all(a>=b for a,b in zip(rel,plan['relation_floors']))
    result=dict(gate=gate,facts=fact,relation_128=rel,v92={k:mean(r['v92'][k] for r in rows) for k in ('online','heldout')},
                throughput_mean=mean(r['throughput'] for r in rows),
                predictor_only_throughput_mean=mean(r['predictor_only_throughput'] for r in rows),rss_max=max(r['peak_rss_through_live_bytes'] for r in rows),
                query_p95_ms_mean=mean(r['query_p95_ms'] for r in rows),rows=rows,plan=plan)
    (ROOT/'OFFICIAL_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
