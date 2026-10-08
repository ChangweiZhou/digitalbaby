"""Non-pickle B checkpoint with current source and pending-address validation."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import environment,config
from candidate import LocalContentCore
NAMES=('environment.py','config.py','candidate.py','bench.py','checks.py','checkpoint_content.py')
def source_identity():
    h=hashlib.sha256(environment.native_checkpoint.source_identity().encode())
    for name in NAMES:
        h.update(name.encode());h.update((environment.ROOT/name).read_bytes())
    return h.hexdigest()
def payload(m):
    meta=dict(schema='LOCAL_CONTENT_CHECKPOINT_V2',source=source_identity(),history=m.history.hex(),t=m.t,bytes_seen=m.bytes_seen,
        plastic=m.plastic,cached=None,expected_digest=m.state_digest())
    a={n:getattr(m,n).copy() for n in ('fast','slow','last','visits')}
    if m.cached is not None:
        meta['cached']=dict(t=m.cached['t'],history=m.cached['history'].hex())
        for n in ('ids','z','p','code'):a['cached_'+n]=m.cached[n].copy()
    a['metadata']=np.frombuffer(json.dumps(meta,sort_keys=True,allow_nan=False).encode(),dtype=np.uint8)
    return a
def save(m,path):
    path=Path(path)
    if path.exists() or path.with_suffix('.sha256').exists():raise ValueError('no checkpoint overwrite')
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**payload(m));f.flush();os.fsync(f.fileno())
    sha=hashlib.sha256(temp.read_bytes()).hexdigest();os.replace(temp,path)
    path.with_suffix('.sha256').write_text(sha+'\n')
def load(path):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=path.with_suffix('.sha256').read_text().strip():raise ValueError('byte integrity')
    with np.load(path,allow_pickle=False) as a:
        meta=json.loads(a['metadata'].tobytes())
        if meta['schema']!='LOCAL_CONTENT_CHECKPOINT_V2' or meta['source']!=source_identity():raise ValueError('source/schema')
        if type(meta['plastic']) is not bool or type(meta['bytes_seen']) is not int or meta['bytes_seen']<0:raise ValueError('policy')
        m=LocalContentCore(plastic=meta['plastic']);m.t=float(meta['t']);m.bytes_seen=meta['bytes_seen'];m.history=bytes.fromhex(meta['history'])
        if not np.isfinite(m.t) or m.t<0 or len(m.history)>4 or any(b not in config.ALPHABET for b in m.history):raise ValueError('context/time')
        for n in ('fast','slow','last','visits'):
            x=a[n];dest=getattr(m,n)
            if x.shape!=dest.shape or x.dtype!=dest.dtype or not np.isfinite(x).all():raise ValueError('array schema')
            dest[:]=x
        if np.any(np.abs(m.fast)>2) or np.any(np.abs(m.slow)>2) or np.any(m.visits>65535) or np.any(m.last<0) or np.any(m.last>m.t):raise ValueError('bounded state/time')
        if meta['cached'] is not None:
            cached=meta['cached']
            if cached['history']!=m.history.hex():raise ValueError('pending address')
            prediction=m.predict(float(cached['t']))
            for n in ('ids','z','p','code'):
                if not np.array_equal(a['cached_'+n],m.cached[n]):raise ValueError('pending state inconsistent with content/address')
        if m.state_digest()!=meta['expected_digest']:raise ValueError('restored digest mismatch')
        return m
