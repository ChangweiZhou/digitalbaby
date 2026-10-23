# SPDX-License-Identifier: GPL-3.0-or-later
"""Evaluator-only interventions and disposable probes; targets never enter learner."""
import contextlib,types,hashlib
from pathlib import Path
import numpy as np
from survivor_fixture import DT,ALPHABET,digest
CONTROLS=('W','N_old_relation','N_new')
ARMS=('R_center','ERROR')
BRANCHES=tuple(a+'__'+b for a in ARMS for b in CONTROLS)
@contextlib.contextmanager
def suppress_value_writes(system,enabled):
    changed=[]
    if enabled:
        for m in system.shared+system.private:
            for name in ('teach_logged','teach_signed'):
                method=getattr(type(m),name)
                def nonplastic(self,*args,_method=method,**kwargs):
                    kwargs['write']=False;return _method(self,*args,**kwargs)
                setattr(m,name,types.MethodType(nonplastic,m));changed.append((m,name))
    try:yield
    finally:
        for m,name in changed:delattr(m,name)
def cue(system,ch,at):
    for i,b in enumerate(bytes.fromhex(ch)):system.feed(b,at+i*DT)
    return system.predict(at+12*DT)
def clock_vector(system):
    return [{'brain_t':m.brain_t,'elapsed':float(m.fly.m.elapsed),'elapsed_base':m.elapsed_base,
             'fe_t':float(m.fe.t),'pending_t':m.pending_t,'last_byte_t':m.last_byte_t,
             'bytes_seen':m.bytes_seen,'teach_seen':m.teach_seen} for m in system.shared+system.private]
def state_identity(system):
    return digest({'stores':system.digests(),'cached':None if system.cached is None else system.cached.tolist(),
                   'private_prediction':None if system.private_prediction is None else system.private_prediction.tolist(),
                   'prediction_time':system.prediction_time,'cue_count':system.cue_count,
                   'audit':system.audit,'clocks':clock_vector(system)})
def normalized(v):
    from scipy.sparse import issparse
    if isinstance(v,np.ndarray):return {'shape':list(v.shape),'dtype':v.dtype.str,'sha256':hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()}
    if issparse(v):return {'class':type(v).__name__,'shape':list(v.shape),'data':normalized(v.data),'indices':normalized(v.indices),'indptr':normalized(v.indptr)}
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,Path):return {'file_basename':v.name}  # model_sha and locked source bind bytes; avoid machine-path dependence
    if isinstance(v,bytes):return {'bytes_hex':v.hex()}
    if isinstance(v,types.ModuleType):return {'module':v.__name__}
    if isinstance(v,dict):return {str(k):normalized(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [normalized(x) for x in v]
    if v is None or isinstance(v,(str,int,float,bool)):return v
    if hasattr(v,'__dict__'):return {'class':type(v).__name__,'fields':normalized(v.__dict__)}
    raise TypeError(type(v))
def operative_state_identity(system):
    stores=[]
    for m in system.shared+system.private:
        fields=dict(m.__dict__)
        # Native fly.stats is evaluator instrumentation; include it separately in wrapper audit if desired.
        fields['fly']={'class':type(m.fly).__name__,'fields':{k:v for k,v in m.fly.__dict__.items() if k!='stats'}}
        stores.append({'class':type(m).__name__,'fields':normalized(fields)})
    return digest({'stores':stores,'arm':system.arm,'scales':system.scales,'wrapper':state_identity(system)})
def fixed_identity(system):
    stores=[]
    changing={'fast','slow','adapt','elapsed','event_count','presentation_count'}
    for m in system.shared+system.private:
        fly={k:v for k,v in m.fly.__dict__.items() if k not in ('m','stats')}
        fly['m']={k:v for k,v in m.fly.m.__dict__.items() if k not in changing}
        stores.append({'fly':normalized(fly),'fe_class':type(m.fe).__name__,'fe_tau':m.fe.tau,
                       'pool':normalized(m.pool_of_kc),'pair_hash':normalized(m.pair_hash),
                       'frozen_predictor_w':normalized(m.w),'frozen_predictor_bias':normalized(m.bias),
                       'seed':m.seed,'temporal':m.temporal,'order_via_native':m.order_via_native,'lr':m.lr,'bias_lr':m.bias_lr})
    return digest({'stores':stores,'arm':system.arm,'scales':system.scales})
def scores(pred,scales):
    s=np.array(pred['shared'])/scales[0];p=np.array(pred['private'])/scales[1]
    out={}
    for name,u in [('combined',s+p),('shared',s),('private',p)]:
        out[name]={'scores':u.tolist(),'emitted':int(ALPHABET[int(np.argmax(u))]),
                   'ties':np.flatnonzero(u==u.max()).tolist()}
    assert out['combined']['emitted']==pred['emitted']
    return out
def probe(system,world,at,names):
    before=operative_state_identity(system);fixed=fixed_identity(system);clocks=clock_vector(system);out=[]
    for name in names:
        for i,ch in enumerate(world['sets'][name]['cues']):
            assert operative_state_identity(system)==before and fixed_identity(system)==fixed
            clone=system.clone();pred=cue(clone,ch,at);policies=scores(pred,system.scales)
            assert operative_state_identity(system)==before and fixed_identity(system)==fixed
            target=world['sets'][name]['outcomes'][i]
            for p in policies.values():p['correct']=int(p['emitted']==target)
            out.append({'set':name,'item':i,'cue_hex':ch,'target':target,'at':at,
                        'first_pre_feedback':True,'continuing_state_before':before,'continuing_state_after':operative_state_identity(system),'fixed_identity':fixed,**pred,'policies':policies})
    assert operative_state_identity(system)==before and fixed_identity(system)==fixed
    return {'state_before':before,'state_after':operative_state_identity(system),'fixed_identity':fixed,'store_digests':system.digests(),'clocks':clocks,'rows':out}

def shared_identity(system):
    """Full operative shared state, excluding measured instrumentation only."""
    stores=[]
    for m in system.shared:
        fields=dict(m.__dict__)
        fields['fly']={'class':type(m.fly).__name__,'fields':{k:v for k,v in m.fly.__dict__.items() if k!='stats'}}
        stores.append({'class':type(m).__name__,'fields':normalized(fields)})
    return digest(stores)

def shared_invariant(systems,tag):
    rows=[]
    for branch in CONTROLS:
        left=systems['R_center__'+branch];right=systems['ERROR__'+branch]
        a=shared_identity(left);b=shared_identity(right)
        assert a==b,('paired shared trajectory diverged',tag,branch)
        assert clock_vector(left)[:4]==clock_vector(right)[:4]
        rows.append({'at_boundary':tag,'control':branch,'R_center':a,'ERROR':b,'identical':True})
    return rows
