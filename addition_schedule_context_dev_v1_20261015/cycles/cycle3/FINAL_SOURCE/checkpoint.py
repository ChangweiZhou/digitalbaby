"""Complete local snapshots with content-deduplicated arrays and no mutable aliasing."""
import base64
import gzip
import hashlib
import io
import json
import pickle
import random
from pathlib import Path
import bootstrap
import models
import numpy as np
import torch
from artifacts import write,read,digest
from identity import source_id

SCHEMA='ADDITION_ORDER_COMPLETE_STATE_V2'


class StorePickler(pickle.Pickler):
    def __init__(self,stream,assets):
        super().__init__(stream,protocol=5);self.assets=assets;self.tokens={};self.objects=[]
    def persistent_id(self,obj):
        if not isinstance(obj,np.ndarray):return None
        if obj.dtype.hasobject:raise ValueError('object arrays are not checkpoint assets')
        contiguous=np.ascontiguousarray(obj)
        key=hashlib.sha256(str((obj.dtype.str,obj.shape)).encode()+contiguous.tobytes()).hexdigest()
        path=self.assets/(key+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            pending=path.with_suffix('.pending')
            with pending.open('wb') as f:np.savez_compressed(f,array=obj)
            pending.replace(path)
        if id(obj) not in self.tokens:
            self.tokens[id(obj)]=len(self.tokens);self.objects.append(obj)
        token=self.tokens[id(obj)]
        return ('NDARRAY_V1',key,token,bool(obj.flags.writeable))


class StoreUnpickler(pickle.Unpickler):
    def __init__(self,stream,assets):super().__init__(stream);self.assets=assets;self.tokens={}
    def persistent_load(self,ref):
        kind,key,token,writable=ref
        if kind!='NDARRAY_V1':raise ValueError('unknown checkpoint asset')
        if token not in self.tokens:
            with np.load(self.assets/(key+'.npz'),allow_pickle=False) as data:array=data['array'].copy()
            observed=hashlib.sha256(str((array.dtype.str,array.shape)).encode()+np.ascontiguousarray(array).tobytes()).hexdigest()
            if observed!=key:raise ValueError('checkpoint asset changed')
            array.flags.writeable=writable;self.tokens[token]=array
        return self.tokens[token]


def save(model,path,*,cursor,next_id,manifest_id,job_state=None):
    path=Path(path);assets=path.parent/'assets';buffer=io.BytesIO()
    kind='KEEP' if isinstance(model,models.Keep) else model.context_variant
    payload=dict(model=model.state() if kind=='KEEP' else model,
        python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state().numpy().copy(),
        cursor=cursor,next_id=next_id,manifest_id=manifest_id,source_id=source_id(),job_state=job_state)
    StorePickler(buffer,assets).dump(payload);raw=buffer.getvalue()
    write(path,dict(schema=SCHEMA,kind=kind,manifest_id=manifest_id,cursor=cursor,next_id=next_id,source_id=source_id(),
          digest=model.digest(),asset_directory='assets',payload_sha256=hashlib.sha256(raw).hexdigest(),
          payload_base64=base64.b64encode(raw).decode(),trusted_local_pickle=True))


def load(path,*,expected_kind,manifest_id,restore_rng=True):
    path=Path(path);record=read(path)
    if record['schema']!=SCHEMA or record['kind']!=expected_kind or record['manifest_id']!=manifest_id or record.get('source_id')!=source_id():
        raise ValueError('checkpoint schema, operative model or manifest mismatch')
    raw=base64.b64decode(record['payload_base64'])
    if hashlib.sha256(raw).hexdigest()!=record['payload_sha256']:raise ValueError('checkpoint payload changed')
    state=StoreUnpickler(io.BytesIO(raw),path.parent/record['asset_directory']).load()
    if state['manifest_id']!=manifest_id or state['cursor']!=record['cursor'] or state['next_id']!=record['next_id'] or state['source_id']!=record['source_id']:
        raise ValueError('checkpoint cursor binding mismatch')
    if expected_kind=='KEEP':model=models.Keep(0);model.set_state(state['model'])
    else:
        model=state['model']
        if not isinstance(model,models.Native) or model.context_variant!=expected_kind:raise ValueError('native class mismatch')
        if model.fronts and (not hasattr(model.fronts[1],'q') or not hasattr(model.fronts[1],'prefix')):
            raise ValueError('native prompt/prefix state missing')
        mutable=[a for bank in model.banks for s in bank for a in (s.fly.m.fast,s.fly.m.slow,s.fly.m.adapt)]
        for i,a in enumerate(mutable):
            if any(np.shares_memory(a,b) for b in mutable[i+1:]):raise ValueError('restored mutable native alias')
    if model.digest()!=record['digest']:raise ValueError('restored operative state mismatch')
    if restore_rng:
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(torch.as_tensor(state['torch_rng']))
    return model,dict(cursor=state['cursor'],next_id=state['next_id'],manifest_id=state['manifest_id'],job_state=state['job_state'])
