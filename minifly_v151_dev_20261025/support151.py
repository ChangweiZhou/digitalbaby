"""V151 evaluator accounting. No candidate architecture or native equation changes."""
import json, os, time, resource, hashlib, pickle
from pathlib import Path
from collections import Counter
import numpy as np
H = Path(__file__).resolve().parent
R = H.parent
D = R / 'minifly_v151_design_20261025'
START = time.monotonic()
COUNTS = Counter()
TIMES = Counter()
CAPS = {'branch_events':2556, 'target_applications':128,
        'nonzero_compartment_increments':4224, 'signed_API_calls':40320,
        'native_vector_readouts':10388}

def convert(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {str(k):convert(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [convert(v) for v in x]
    return x

def save(name, data):
    path = H/name
    text = json.dumps(convert(data), indent=2, allow_nan=False)+'\n'
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)

def load(path): return json.loads(Path(path).read_text())
def sha(path):
    t = time.process_time(); raw = Path(path).read_bytes()
    COUNTS['hash_file_bytes'] += len(raw)
    result = hashlib.sha256(raw).hexdigest(); TIMES['hash_IO_CPU'] += time.process_time()-t
    return result

def ledger(event, **data):
    path = H/os.environ.get('V151_LEDGER','LEDGER.jsonl')
    row = dict(event=event, CPU=cpu(), wall=time.monotonic()-START, **convert(data))
    with path.open('a') as f:
        f.write(json.dumps(row, allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())

def cpu():
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time()+r.ru_utime+r.ru_stime

def budget(reserve=0):
    prior = float(os.environ.get('V151_PRIOR_CPU','0'))
    elapsed = time.time()-float(os.environ.get('V151_WALL_START',str(time.time())))
    if cpu()+prior+reserve > 1800: raise RuntimeError('PAUSED_CPU_BUDGET')
    if elapsed+reserve > 2700: raise RuntimeError('PAUSED_WALL_BUDGET')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > 1024**3: raise RuntimeError('PAUSED_MEMORY_TARGET')
    for name,limit in CAPS.items():
        if COUNTS[name]>limit: raise AssertionError(f'Exposure/readout cap exceeded: {name}')

def clone(m):
    t=time.process_time(); out=m.clone(); TIMES['top_clone_CPU']+=time.process_time()-t
    COUNTS['top_model_clones']+=1; COUNTS['native_fly_clones_known']+=16
    # Minimum mutable arrays copied; inherited clone may copy other temporary arrays.
    COUNTS['minimum_mutable_clone_bytes']+=sum(s.fly.m.fast.nbytes+s.fly.m.slow.nbytes+s.fly.m.adapt.nbytes for s in m.bank)
    return out

def digest(m):
    t=time.process_time(); raw=pickle.dumps(m,protocol=5)
    COUNTS['state_hash_bytes']+=len(raw)
    out=hashlib.sha256(raw).hexdigest(); TIMES['state_hash_CPU']+=time.process_time()-t
    return out

def prompt(m, prefix):
    t=time.process_time(); m._control(True)
    for b in prefix: m._event(b)
    m._control(False)
    COUNTS['sensory_prefix_bytes']+=len(prefix)
    COUNTS['passive_prefix_byte_events']+=len(prefix)
    COUNTS['record_control_events']+=2
    COUNTS['native_inner_event_evaluations']+=16*len(prefix)
    COUNTS['native_rest_operations_known']+=32
    TIMES['prompt_CPU']+=time.process_time()-t

def values(m):
    t=time.process_time(); v=m.values()
    COUNTS['native_vector_readouts']+=1; COUNTS['native_fly_clones_known']+=16
    COUNTS['native_rest_operations_known']+=16
    COUNTS['minimum_mutable_clone_bytes']+=sum(s.fly.m.fast.nbytes+s.fly.m.slow.nbytes+s.fly.m.adapt.nbytes for s in m.bank)
    TIMES['readout_CPU']+=time.process_time()-t
    assert v.shape==(16,) and np.isfinite(v).all()
    budget(); return v

def prepared(anchor,prefix):
    out=clone(anchor); prompt(out,prefix); return out

def centered(v): return v-v.mean(axis=-1,keepdims=True)

def intervene(pre, a, *, observed=False, delta_C=None, passive=False, ident=''):
    budget(); m=clone(pre); a=np.asarray(a,dtype=float)
    assert a.shape==(16,) and np.isfinite(a).all() and np.max(abs(a))<=1
    assert not passive or (not observed and delta_C is None and not np.any(a))
    t=time.process_time(); dt=m._tick(); x=m._code()
    nz=int(np.count_nonzero(a)); applied=0.
    for j,s in enumerate(m.bank):
        if passive: s.fly.event(dt,x,0.,False)
        else:
            s.pending_t=m.time; s.pending_x=x
            applied+=s.teach_signed(0.,float(a[j]),m.time,write=True)
        s.brain_t=m.time
    if observed: m.target_n+=1
    if delta_C is not None: m.C=pre.C+delta_C; COUNTS['code_updates']+=1
    COUNTS['branch_events']+=1
    COUNTS['target_applications']+=int(observed)
    COUNTS['nonzero_compartment_increments']+=nz
    COUNTS['signed_API_calls']+=0 if passive else 16
    COUNTS['native_internal_plastic_evaluations']+=2*nz
    COUNTS['native_inner_event_evaluations']+=16+2*nz
    COUNTS['native_fly_clones_known']+=2*nz
    COUNTS['minimum_mutable_clone_bytes']+=sum(2*(s.fly.m.fast.nbytes+s.fly.m.slow.nbytes+s.fly.m.adapt.nbytes) for j,s in enumerate(m.bank) if a[j]!=0)
    TIMES['intervention_CPU']+=time.process_time()-t
    assert m.fe is not None and m.target_n==pre.target_n+int(observed)
    assert np.array_equal(m.theta,pre.theta) and np.array_equal(m.freq,pre.freq)
    assert m.time==pre.time+dt
    ledger('BRANCH',id=ident,observed_target=bool(observed),nonzero_compartments=nz,
           signed_API_calls=0 if passive else 16,active_KCs=int(x.sum()),coefficient_L2_squared=float(a@a),native_applied_alpha_L1=applied)
    budget(); return m

def query_vectors(m, site, records, immediate_only=False, with_keys=False):
    def key(mm):
        f=mm.fe.clone(); f.advance(mm.time+30/14)
        return mm.encode(f.read())[0]
    out=[values(m)]; keys=[key(m)] if with_keys else []
    if not immediate_only:
        entries=[site['source'],site['peers']['same_target'],site['peers']['different_target']]
        for e in entries:
            q=clone(m); prompt(q,records[e['record_index']][:e['target_position']]); out.append(values(q))
            if with_keys:keys.append(key(q))
    return (np.array(out),keys) if with_keys else np.array(out)

def functional_error(a,b):
    err=0.
    for x,y in zip(a.bank,b.bank):
        for key in ['fast','slow','adapt']:
            err=max(err,float(np.max(abs(getattr(x.fly.m,key)-getattr(y.fly.m,key)))))
    err=max(err,float(np.max(abs(a.fe.p-b.fe.p))),abs(a.time-b.time),float(np.max(abs(a.C-b.C))))
    return err
