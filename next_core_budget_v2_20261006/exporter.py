"""Portable JSON/array W snapshot; no history records or executable pickle."""
import io,json,zipfile,hashlib
import numpy as np
from runtime import Core,atomic_bytes,canonical,digest,SCIENTIFIC_ID
ADAPTER=('prediction_time','cue_count','awaiting_newline','last_time','records','last_write','yoked_counts')
def save_model(c,path,binding):
    if c.cached is not None or c.private_prediction is not None or c.cue_count or c.awaiting_newline:raise ValueError('model export unsafe boundary')
    arrays={}
    def pack(v):
        if isinstance(v,np.ndarray):
            k=f'a{len(arrays)}';arrays[k]=v.copy();return {'__array__':k}
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,dict):return {str(k):pack(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return [pack(x) for x in v]
        return v
    states=[]
    for m in c.models:
        s=m.snapshot()
        if hasattr(m.fe,'visible'):s['visible_hex']=m.fe.visible.hex()
        states.append(pack(s))
    meta={'schema':'FINAL_W_CORE_V2','scientific_source_identity':SCIENTIFIC_ID,'arm':c.arm,'binding':binding,
          'states':states,'adapter':pack({k:getattr(c,k) for k in ADAPTER}),'state_digest':c.state_digest(),
          'arrays':{k:{'sha256':hashlib.sha256(v.tobytes()).hexdigest(),'shape':list(v.shape),'dtype':v.dtype.str} for k,v in arrays.items()}}
    meta['seal']=digest(meta);arrays['metadata_json']=np.frombuffer(canonical(meta),dtype=np.uint8)
    out=io.BytesIO();np.savez_compressed(out,**arrays);atomic_bytes(path,out.getvalue())
    return {'file':str(path.name),'sha256':hashlib.sha256(out.getvalue()).hexdigest(),'state_digest':meta['state_digest'],'bytes':len(out.getvalue()),'binding':binding}
def load_model(path,binding):
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=len(set(z.namelist())) or sum(f.file_size for f in z.infolist())>256*1024**2:raise ValueError('model container')
    with np.load(path,allow_pickle=False) as z:
        m=json.loads(z['metadata_json'].tobytes());seal=m.pop('seal')
        if digest(m)!=seal or m['schema']!='FINAL_W_CORE_V2' or m['binding']!=binding or m['scientific_source_identity']!=SCIENTIFIC_ID:raise ValueError('model seal/binding')
        if set(z.files)!=set(m['arrays'])|{'metadata_json'}:raise ValueError('model arrays roster')
        for k,s in m['arrays'].items():
            a=z[k]
            if a.dtype.hasobject or not np.isfinite(a).all() or list(a.shape)!=s['shape'] or a.dtype.str!=s['dtype'] or hashlib.sha256(a.tobytes()).hexdigest()!=s['sha256']:raise ValueError('model array corruption')
        def unpack(v):
            if isinstance(v,dict):
                if set(v)=={'__array__'}:return z[v['__array__']].copy()
                return {k:unpack(x) for k,x in v.items()}
            if isinstance(v,list):return [unpack(x) for x in v]
            return v
        c=Core(m['arm'])
        if len(m['states'])!=len(c.models):raise ValueError('model stores')
        for brain,s in zip(c.models,m['states']):
            s=unpack(s);visible=s.pop('visible_hex',None);brain.restore(s)
            if hasattr(brain.fe,'visible'):
                if visible is None:raise ValueError('missing content state')
                brain.fe.visible=bytes.fromhex(visible)
            elif visible is not None:raise ValueError('unexpected content state')
        adapter=unpack(m['adapter'])
    if set(adapter)!=set(ADAPTER):raise ValueError('operative adapter roster')
    for k,v in adapter.items():setattr(c,k,v)
    if c.state_digest()!=m['state_digest'] or c.records!=binding['records'] or c.last_time!=binding['time']:raise ValueError('model operative digest/cursor')
    return c
