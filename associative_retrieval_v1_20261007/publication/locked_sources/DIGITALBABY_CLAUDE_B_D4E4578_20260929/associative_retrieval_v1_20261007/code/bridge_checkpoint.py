import hashlib
import io
import json
import zipfile
import runtime
import numpy as np
from bridge_core import RetrievalCore
from association import Associations
from io_utils import atomic_bytes, canonical, digest

ADAPTER = ('prediction_time','cue_count','awaiting_newline','last_time','records','last_write','yoked_counts')

def save(c, path, cursor, identity):
    if c.cached is not None or c.private_prediction is not None or c.association_pending is not None or c.cue_count or c.awaiting_newline:
        raise ValueError('checkpoint only after whole-record boundary')
    arrays = {}
    def pack(v):
        if isinstance(v,np.ndarray):
            key=f'a{len(arrays)}'; arrays[key]=v; return {'array':key}
        if isinstance(v,np.generic): return v.item()
        if isinstance(v,dict): return {str(k):pack(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)): return [pack(x) for x in v]
        return v
    states=[]
    for brain in c.models:
        s=brain.snapshot()
        if hasattr(brain.fe,'visible'): s['visible_hex']=brain.fe.visible.hex()
        states.append(pack(s))
    meta={'schema':'ASSOCIATIVE_CHECKPOINT_V1','identity':identity,'states':states,
          'association':pack(c.associations.snapshot()),'adapter':{k:getattr(c,k) for k in ADAPTER},
          'cursor':cursor,'digest':c.state_digest(),
          'arrays':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}}
    meta['seal']=digest(meta); arrays['metadata_json']=np.frombuffer(canonical(meta),np.uint8)
    f=io.BytesIO(); np.savez_compressed(f,**arrays); atomic_bytes(path,f.getvalue())

def load(path, identity):
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=len(set(z.namelist())) or sum(i.file_size for i in z.infolist())>256*1024**2:
            raise ValueError('checkpoint container')
    with np.load(path,allow_pickle=False) as z:
        m=json.loads(z['metadata_json'].tobytes()); seal=m.pop('seal')
        if digest(m)!=seal or m['schema']!='ASSOCIATIVE_CHECKPOINT_V1' or m['identity']!=identity:
            raise ValueError('checkpoint seal/identity')
        if set(z.files)!=set(m['arrays'])|{'metadata_json'}: raise ValueError('checkpoint array roster')
        for k,h in m['arrays'].items():
            a=z[k]
            if a.dtype.hasobject or not np.isfinite(a).all() or hashlib.sha256(a.tobytes()).hexdigest()!=h:
                raise ValueError('checkpoint array corruption')
        def unpack(v):
            if isinstance(v,dict):
                if set(v)=={'array'}: return z[v['array']].copy()
                return {k:unpack(x) for k,x in v.items()}
            if isinstance(v,list): return [unpack(x) for x in v]
            return v
        c=RetrievalCore()
        if len(m['states'])!=8 or set(m['adapter'])!=set(ADAPTER): raise ValueError('checkpoint store/adapter roster')
        for brain,state in zip(c.models,m['states']):
            s=unpack(state); visible=s.pop('visible_hex',None); brain.restore(s)
            if hasattr(brain.fe,'visible'):
                if visible is None: raise ValueError('missing visible context')
                brain.fe.visible=bytes.fromhex(visible)
            elif visible is not None: raise ValueError('unexpected visible context')
        c.associations=Associations.restore(unpack(m['association']))
    for k,v in m['adapter'].items(): setattr(c,k,v)
    if c.state_digest()!=m['digest']: raise ValueError('operative snapshot mismatch')
    cur=m['cursor']; n=cur['next_record']
    if type(n) is not int or n!=c.records or c.associations.observations!=n or c.cue_count or c.awaiting_newline:
        raise ValueError('checkpoint record cursor mismatch')
    if len(cur['records'])!=n or [r['index'] for r in cur['records']]!=list(range(n)):
        raise ValueError('checkpoint record history mismatch')
    from assays import make_world
    w=make_world(cur['world'],cur['assay'])
    if cur['fixture_sha256']!=w['sha256'] or cur['branch'] not in w['branches'] or not 0<=n<=cur['limit']<=len(w['events']):
        raise ValueError('checkpoint fixture/branch mismatch')
    return c,cur

