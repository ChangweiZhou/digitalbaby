"""Cycle 1 source audit and bounded native/eligibility/interface fixtures."""
import bootstrap
import ast
import copy
import hashlib
import inspect
import json
import time
from pathlib import Path

import numpy as np
from core import Learner, CODE, UP, DELTA, ramp, MEAN_RATE, MEAN_SQUARE, hash_arrays
from artifacts import HERE, ROOT, atomic_json, sha, source_identity


def main():
    t0=time.process_time()
    output=HERE/'cycles/cycle1'
    lock=json.loads((ROOT/'minifly_a_v3/SOURCE_LOCK.json').read_text())['files']
    checked={}
    for path in (HERE/'upstream/native_package').rglob('*'):
        if path.is_file() and path.suffix in ('.py','.json','.npz'):
            relative='package/'+str(path.relative_to(HERE/'upstream/native_package'))
            if relative in lock:
                assert sha(path)==lock[relative], relative
                checked[str(path.relative_to(ROOT))]=sha(path)
    for path in (ROOT/'minifly_a_v3/src').glob('*.py'):
        relative='src/'+path.name
        if relative in lock:
            assert sha(path)==lock[relative],relative
            checked[str(path.relative_to(ROOT))]=sha(path)
    assert len(checked)>30
    atomic_json(output/'NATIVE_ASSET_AUDIT.json',dict(accepted=True,files=checked,canonical_birth_verified_by='portable_birth.canonical_fresh_native'))
    print('source assets',len(checked),flush=True)
    ref=UP.FastMemory()
    model=Learner(202610208801,source_parent=True)
    max_state=max_value=max_threshold=0.
    rng=np.random.default_rng(202610208811)
    fixtures=[(bytes(rng.integers(0,256,length,dtype=np.uint8)),bytes(rng.integers(0,256,1,dtype=np.uint8)))
              for length in (0,1,2,4,8,16,32,64)]
    for prompt,answer in fixtures:
        ref._prompt(prompt);model._prompt(prompt)
        for target in (*answer,256):
            frame=model._frame()
            error=float(abs(frame['z']-ref.values()).max());max_value=max(max_value,error)
            assert error<1e-9,('native values',error)
            ref._event(target,target);model._event(target,target,True,frame)
            for key in ('fast','slow','adapt'):
                e=float(abs(np.stack([getattr(s.fly.m,key) for s in ref.bank])-getattr(model.native,key)).max())
                max_state=max(max_state,e);assert e<1e-9,(key,e)
            e=float(abs(model.theta-ref.theta).max());max_threshold=max(max_threshold,e);assert e<1e-10
            assert np.allclose(model.freq,ref.freq,atol=1e-12,rtol=0)
        ref.records+=1;model.records+=1
    for seconds in (3600.,82800.):
        ref.rest(seconds);model.rest(seconds)
        assert model.time==ref.time
        for prompt,_ in fixtures[:4]:
            ref._prompt(prompt);model._prompt(prompt)
            e=float(abs(ref.values()-model._frame()['z']).max());max_value=max(max_value,e)
            assert e<1e-9,e
    print('native parity',max_state,max_value,flush=True)
    m=Learner(202610208802,plastic=True,gradual=True)
    p={k:v.copy() for k,v in m.params.items()}
    old=rng.normal(0,.1,64);r=np.asarray(bootstrap.stores.bb.bc.R_TABLE[137])
    v=np.tanh(p['W']@r+p['U']@old+p['b']);g=1/(1+np.exp(-(p['G']@r+p['H']@old+p['c'])))
    direct={k:(g*(1-v*v) if k in ('W','U','b') else g*(1-g)*(v-old)) for k in p}
    errors={}
    for name in p:
        idx=(7,12) if p[name].ndim==2 else (7,)
        pre=(r[12] if name in ('W','G') else old[12]) if len(idx)==2 else 1.
        def response(delta):
            q={k:value.copy() for k,value in p.items()};q[name][idx]+=delta
            vv=np.tanh(q['W']@r+q['U']@old+q['b'])
            gg=1/(1+np.exp(-(q['G']@r+q['H']@old+q['c'])))
            return ((1-gg)*old+gg*vv)[7]
        fd=(response(1e-6)-response(-1e-6))/2e-6
        error=abs(fd-direct[name][7]*pre);errors[name]=float(error);assert error<1e-8,(name,error)
    # Local traces, incoming target timing, and writes-disabled history.
    initial=m.parameter_hash();m.train(b'\x00\xff\x89',b'\x80',True)
    assert m.parameter_hash()!=initial and m.parameter_steps==1
    assert all(np.diag(m.params[k]).tolist()==[0.]*64 for k in ('U','H'))
    assert all(t['post_version']==t['pre_version']+1 and t['posttime']==t['pretime']+DELTA for t in m.last_trace)
    before=m.digest();got=m.generate(b'\x00\xff\x89');assert m.digest()==before
    # Forced byte/EOS decoder fixtures verify stopping flags independently of learned accuracy.
    fake=m.clone();fake._frame=lambda:dict(z=CODE[0]);out=fake.generate(b'',cap=0)
    assert out['payload']==b'' and out['truncated'] and not out['ended_with_EOS']
    fake._frame=lambda:dict(z=CODE[256]);out=fake.generate(b'',cap=0)
    assert out['payload']==b'' and out['ended_with_EOS'] and not out['truncated']
    caught=False
    try:m.generate(bytes(257))
    except ValueError as exc:caught=str(exc)=='INPUT_LIMIT'
    assert caught
    base_params=m.parameter_hash();theta=m.theta.copy();freq=m.freq.copy();fast=m.native.fast.copy();slow=m.native.slow.copy();t=m.time
    m.train(b'\x01\x02',b'\xfe',False)
    dt=m.time-t
    assert m.parameter_hash()==base_params and np.array_equal(theta,m.theta) and np.array_equal(freq,m.freq)
    assert np.allclose(m.native.fast[...,0],fast[...,0]*np.exp(-dt/m.native.tg),atol=1e-10)
    assert np.allclose(m.native.fast[...,1],fast[...,1]*np.exp(-dt/m.native.tf),atol=1e-10)
    assert np.allclose(m.native.slow,slow*np.exp(-dt/m.native.ts),atol=1e-10)
    # All payload identities are accepted; this is not an arithmetic capability assay.
    for b in range(256):m._context(b)
    from data import metadata,records
    for world in (202610209901,202610209902):
        meta=metadata(world);assert meta['disjoint'] and meta['held_never_taught_in_either_order']
        for phase in ('OLD','NEW'):
            rr=list(records(world,'W',phase));sh=list(records(world,'SH_OLD',phase))
            size=60 if phase=='OLD' else 20
            for j in range(0,len(rr),size):
                assert sorted(x['answer'] for x in rr[j:j+size])==sorted(x['answer'] for x in sh[j:j+size])
    rates=np.array([ramp(n) for n in range(1,7681)])
    assert abs(rates.sum()-MEAN_RATE*7680)<1e-9
    assert abs((rates*rates).sum()-MEAN_SQUARE*7680)<1e-9
    # A learner must have no data/evaluation or generic autograd/answer-head imports.
    for filename in ('core.py','native.py'):
        tree=ast.parse((HERE/filename).read_text())
        imported=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imported.extend(n.name for n in node.names)
            if isinstance(node,ast.ImportFrom):imported.append(node.module or '')
        assert not set(imported)&{'data','evaluate','torch','jax','tensorflow','slowcore','cls'}
    report=dict(schema='MINIFLY_V3_CYCLE_1',accepted=True,source_asset_files=len(checked),
                native_max_state_error=max_state,native_max_value_error=max_value,native_max_homeostasis_error=max_threshold,
                direct_local_finite_difference_errors=errors,readonly_query=True,all_learning_off=True,
                frame_timing=True,zero_recurrent_diagonals=True,all_payload_bytes=True,EOS_truncation_distinct=True,
                forbidden_learner_imports_absent=True,scheduled_dose_means=[MEAN_RATE,MEAN_SQUARE],
                record_applications=18,preliminary_record_applications=2,partial_prompt_exposures=18,
                scientific_records=0,cpu_s=time.process_time()-t0,source_identity=source_identity())
    atomic_json(output/'AUDIT.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='source_identity'}),flush=True)


if __name__=='__main__':main()
