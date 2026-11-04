"""Held-out prose comparison of the Round 7 lag head and V9.2 predictor equations."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
V92=ROOT/'v92_predictor_source.py'

def evaluate(job):
    name,seed,raw,split=job
    if name=='hashed_v92':
        # Load the exact predictor-only constants, hash and methods. No native state.
        import hashlib
        source=V92.read_text()
        env=dict(np=np,hashlib=hashlib)
        exec(source[source.index('VERSION ='):source.index('class F151ByteBrainV92')],env)
        exec('class Predictor:\n'+source[source.index('    def _features'):source.index('    def predict(')],env)
        model=env['Predictor']()
        model.history=bytearray()
        model.w=np.zeros((env['ROWS'],256),np.float32)
        model.bias=np.zeros(256,np.float32)
        nbytes=model.w.nbytes+model.bias.nbytes
    else:
        env=dict(np=np)
        exec((ROOT/'lag_recovered.py').read_text(),env)
        code=env['Code'](seed,96,4096,205)
        encode=lru_cache(maxsize=4096)(lambda h:code.enc(env['front']('lag',h)))
        weights=np.zeros((4096,256),np.float32)
        nbytes=weights.nbytes+code.ct.nbytes+code.w.nbytes
    total=0.
    for i,y in enumerate(raw):
        if name=='hashed_v92':
            p=model._distribution()
            if i<split:
                err=p.astype(np.float32)
                err[y]-=1
                for idx,rate in model._features():
                    model.w[idx]-=np.float32(rate)*err
                model.bias-=np.float32(env['BIAS_RATE'])*err
            model.history.append(y)
            del model.history[:-4]
        else:
            idx,v=encode(raw[max(0,i-12):i])
            scores=v@weights[idx]
            logits=np.exp((scores-scores.max())/.1)
            p=logits/logits.sum()
            if i<split and v@v>1e-15:
                err=-scores
                err[y]+=1
                weights[idx]+=(.6*np.outer(v/(v@v),err)).astype(np.float32)
        if i>=split:
            total-=math.log2(float(p[y]))
    return dict(arm=name,seed=seed,bpb=total/(len(raw)-split),predictor_array_bytes=nbytes)

if __name__=='__main__':
    data=Path(sys.argv[1]).read_bytes()
    raw=data.split(b'*** START OF',1)[1].split(b'\n',1)[1].split(b'*** END OF',1)[0].strip()
    split=len(raw)*4//5
    jobs=[('hashed_v92',0,raw,split)]+[('lag',i,raw,split) for i in (19701,19702,19703,19704)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(evaluate,jobs))
    result=dict(corpus='Project Gutenberg #11',download_sha256=hashlib.sha256(data).hexdigest(),
                scoring='Round 7 lag temperature 0.1; chronological frozen-weight suffix; causal observed-byte history',
                train_bytes=split,test_bytes=len(raw)-split,rows=rows)
    (ROOT/'PROSE_NATIVE_TEMPERATURE_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
