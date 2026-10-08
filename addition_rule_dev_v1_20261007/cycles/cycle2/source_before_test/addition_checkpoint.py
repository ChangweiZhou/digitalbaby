"""Portable non-pickle DEV checkpoints, including a pending emitted answer."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import settings
from learners import build

def identity():
    from runner import source_lock
    return source_lock()
def save(model,arm,path):
    path=Path(path)
    if path.exists():raise ValueError('no checkpoint overwrite')
    arrays={f'a{i}':a.copy() for i,a in enumerate(model.content_arrays())}
    pending=None
    if model.pending is not None:
        pending={k:v for k,v in model.pending.items() if not isinstance(v,np.ndarray)}
        for k,v in model.pending.items():
            if isinstance(v,np.ndarray):arrays['pending_'+k]=v.copy()
    meta=dict(schema='ADDITION_DEV_CHECKPOINT',source=identity(),arm=arm,t=model.t,bytes_seen=model.bytes_seen,plastic=model.plastic,
        line=model.sensor.line.hex(),scalars=model.content_scalars(),pending=pending,expected_digest=model.state_digest())
    arrays['metadata']=np.frombuffer(json.dumps(meta,sort_keys=True,allow_nan=False).encode(),np.uint8)
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.pending')
    with tmp.open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    digest=hashlib.sha256(tmp.read_bytes()).hexdigest();os.replace(tmp,path);path.with_suffix('.sha256').write_text(digest+'\n')

def load(path):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=path.with_suffix('.sha256').read_text().strip():raise ValueError('checkpoint byte integrity')
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(z['metadata'].tobytes())
        if meta['schema']!='ADDITION_DEV_CHECKPOINT' or meta['source']!=identity():raise ValueError('checkpoint source/schema')
        if type(meta['plastic']) is not bool or type(meta['bytes_seen']) is not int or meta['bytes_seen']<0:raise ValueError('policy/counter')
        model=build(meta['arm']);model.t=settings.valid_time(meta['t'],0.);model.bytes_seen=meta['bytes_seen'];model.plastic=meta['plastic']
        line=bytes.fromhex(meta['line']);model.sensor.line=b''
        for byte in line:model.sensor.feed(byte)
        arrays=model.content_arrays()
        for i,dest in enumerate(arrays):
            a=z['a'+str(i)]
            if a.shape!=dest.shape or a.dtype!=dest.dtype or not np.isfinite(a).all():raise ValueError('array schema')
            dest[:]=a
        if meta['arm']=='NATIVE':
            if len(meta['scalars'])!=9:raise ValueError('store count')
            for s,v in zip(model.stores,meta['scalars']):
                s.brain_t,s.elapsed_base,s.teach_seen=v[:3];s.fly.m.elapsed=np.float64(v[3]);s.fly.m.event_count=np.uint64(v[4]);s.fly.m.presentation_count=np.uint64(v[5])
                if not 0<=s.brain_t<=model.t or abs(v[3]-v[1]-v[0])>1e-6:raise ValueError('native clock')
        else:
            if np.any(model.last<0) or np.any(model.last>model.t) or np.any(model.visits>settings.VISIT_CAP):raise ValueError('row clock/count')
            if np.any(np.abs(model.fast)>settings.ROW_BOUND) or np.any(np.abs(model.slow)>settings.ROW_BOUND):raise ValueError('content bounds')
        if meta['pending'] is not None:
            c=meta['pending']
            if c['t']!=model.t or tuple(c['operands'])!=model.sensor.operands():raise ValueError('pending context/time')
            p,v,extra=model._prediction(model.sensor.operands(),model.t)
            model.pending=dict(t=model.t,operands=model.sensor.operands(),p=p.copy(),**extra)
            for k,a in model.pending.items():
                if isinstance(a,np.ndarray) and not np.array_equal(a,z['pending_'+k]):raise ValueError('pending prediction/content mismatch')
        elif len(line)==4 and line.endswith(b'='):raise ValueError('missing pending answer')
        if model.state_digest()!=meta['expected_digest']:raise ValueError('restored state mismatch')
    return model
