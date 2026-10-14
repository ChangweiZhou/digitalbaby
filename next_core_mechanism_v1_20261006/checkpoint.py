"""Lossless array/JSON snapshot; no pickle, all operative mechanism fields retained."""
import io,json,hashlib,zipfile
import numpy as np
from core import Core
from io_utils import atomic_bytes,digest,canonical
def save(core,path,cursor,identity):
    if core.cached is not None or core.private_prediction is not None:raise ValueError('checkpoint pending outcome')
    arrays={}
    def pack(x):
        if isinstance(x,np.ndarray):
            k=f'a{len(arrays)}';arrays[k]=x;return {'__array__':k}
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,dict):return {str(k):pack(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [pack(v) for v in x]
        return x
    states=[]
    for m in core.models:
        s=m.snapshot()
        if hasattr(m.fe,'visible'):s['visible_hex']=m.fe.visible.hex()
        states.append(pack(s))
    adapter={k:getattr(core,k) for k in ('prediction_time','cue_count','awaiting_newline','last_time','records','last_write','yoked_counts')}
    meta={'schema':'NEXT_CORE_CHECKPOINT_V1','identity':identity,'arm':core.arm,'cursor':cursor,
          'states':states,'adapter':adapter,'digest':core.state_digest(),
          'arrays':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}}
    meta['seal']=digest(meta);arrays['metadata_json']=np.frombuffer(canonical(meta),dtype=np.uint8)
    out=io.BytesIO();np.savez_compressed(out,**arrays);atomic_bytes(path,out.getvalue())
def load(path,arm,identity):
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=len(set(z.namelist())) or sum(i.file_size for i in z.infolist())>256*1024**2:raise ValueError('checkpoint container')
    with np.load(path,allow_pickle=False) as z:
        m=json.loads(z['metadata_json'].tobytes());seal=m.pop('seal')
        if digest(m)!=seal or m['identity']!=identity or m['arm']!=arm or m['schema']!='NEXT_CORE_CHECKPOINT_V1':raise ValueError('checkpoint identity/seal')
        if set(z.files)!=set(m['arrays'])|{'metadata_json'}:raise ValueError('checkpoint arrays roster')
        for k,v in m['arrays'].items():
            a=z[k]
            if a.dtype.hasobject or not np.isfinite(a).all() or hashlib.sha256(a.tobytes()).hexdigest()!=v:raise ValueError('checkpoint array corruption')
        def unpack(x):
            if isinstance(x,dict):
                if set(x)=={'__array__'}:return z[x['__array__']].copy()
                return {k:unpack(v) for k,v in x.items()}
            if isinstance(x,list):return [unpack(v) for v in x]
            return x
        c=Core(arm)
        if len(m['states'])!=len(c.models):raise ValueError('checkpoint store count')
        for brain,state in zip(c.models,m['states']):
            state=unpack(state);visible=state.pop('visible_hex',None);brain.restore(state)
            if hasattr(brain.fe,'visible'):
                if visible is None:raise ValueError('missing content context')
                brain.fe.visible=bytes.fromhex(visible)
            elif visible is not None:raise ValueError('unexpected content context')
    for k,v in m['adapter'].items():setattr(c,k,v)
    if c.state_digest()!=m['digest']:raise ValueError('restored operative state mismatch')
    cur=m['cursor']
    if cur['next_record']!=c.records or c.cue_count or c.awaiting_newline or c.cached is not None:raise ValueError('cursor/core mismatch')
    from assays import make_world
    w=make_world(cur['world'],cur['assay']);n=cur['next_record'];bi=cur['branch_index'];limit=cur['limit'];names=cur['branch_names']
    if (cur['fixture_sha256']!=w['sha256'] or type(n) is not int or type(bi) is not int or type(limit) is not int
        or not 0<=n<=limit<=len(w['events']) or limit<=0 or not 0<=bi<len(names)
        or names not in (w['branches'],['W']) or set(cur['branches'])!=set(names[:bi+1])):raise ValueError('cursor fixture/branch roster')
    for i,branch in enumerate(names[:bi+1]):
        rs=cur['branches'][branch]['records'];want=n if i==bi else limit
        if len(rs)!=want or [r['index'] for r in rs]!=list(range(want)):raise ValueError('cursor record history')
    return c,cur
