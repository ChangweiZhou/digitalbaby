"""Independent event audit and tests; no science admission, no fitted probes."""
import copy, gzip, json, math, hashlib
from pathlib import Path
import numpy as np
import environment, config
from candidate import LocalContentCore
import bench

# This explicit truth table is independent of bench.allowed.
WRITE={'old':{'W':True,'N_OLD':False,'N_REV':True,'N_ALL':False},
       'new':{'W':True,'N_OLD':True,'N_REV':True,'N_ALL':False},
       'revised':{'W':True,'N_OLD':True,'N_REV':False,'N_ALL':False}}

def verify_rows(raw,rows,h,t,expected):
    if len(raw)!=len(rows) or not raw:raise AssertionError('dropped event')
    for b,row in zip(raw,rows):
        t+=config.DT
        assert row['byte']==b and row['context']==h.hex() and abs(row['t']-t)<1e-7,'byte/context/clock'
        p=np.asarray(row['probabilities'])
        assert p.shape==(4,) and np.isfinite(p).all() and (p>0).all() and abs(p.sum()-1.)<1e-12
        assert row['plastic'] is expected,'causal branch disabled incorrectly'
        assert row['emitted']==config.ALPHABET[int(np.argmax(p))],'model output mismatch'
        assert np.allclose(row['signs'],p-np.array([int(b==x) for x in config.ALPHABET]),rtol=0,atol=1e-12)
        dose=np.asarray(row['native_write_l1']);assert dose.shape==(4,) and np.isfinite(dose).all() and (dose>=0).all()
        assert expected or not np.any(dose),'disabled write actually applied'
        assert abs(row['loss_bits']+math.log2(float(p[config.ALPHABET.index(b)])))<1e-12
        h=(h+bytes([b]))[-4:]
    return h,t

def independent_metrics(rows,mapping):
    r=[x for x in rows if x['context'] in mapping]
    assert r and all(x['byte']==mapping[x['context']] for x in r),'wrong focal successor'
    return dict(focal_accuracy=sum(x['byte']==x['emitted'] for x in r)/len(r),
        focal_bpb=sum(x['loss_bits'] for x in r)/len(r),
        all_byte_accuracy=sum(x['byte']==x['emitted'] for x in rows)/len(rows),
        all_byte_bpb=sum(x['loss_bits'] for x in rows)/len(rows),focal_count=len(r),all_byte_count=len(rows))

def audit(r):
    config.require_dev(r['world']);assert r['evidence']=='DEV_ONLY_E0_E1' and r['actual_training_lives']==8
    assert set(r['arms'])=={'A','B'}
    assert len(r['inputs']['old'])==16 and len(r['inputs']['new'])==8 and len(r['inputs']['revised'])==8
    assert not(set(r['inputs']['old'])&set(r['inputs']['new']))
    checked=0
    for arm,a in r['arms'].items():
        assert set(a['birth'])==set(config.BRANCHES)
        for branch,birth in a['birth'].items():
            assert len(birth)==(4 if arm=='A' else 1)
            assert all(x['canonical_B_sha256']==environment.native_core.stores.EXPECTED_B for x in birth)
        for branch in config.BRANCHES:
            h=b'';t=0.;bytes_seen=0
            for stage,group in a['probes'].items():
                if stage.startswith('day'):t+=config.DAY
                else:
                    raw=bytes.fromhex(r['raw'][stage]);rows=a['training'][stage][branch]
                    h,t=verify_rows(raw,rows,h,t,WRITE[stage][branch]);checked+=len(raw);bytes_seen+=len(raw)
                pr=group[branch]
                assert pr['initial_history']==h.hex() and abs(pr['initial_time']-t)<1e-7
                raw=bytes.fromhex(pr['raw_hex']);verify_rows(raw,pr['rows'],h,t,False);checked+=len(raw)
                target='old' if stage in ('old','day1','new','day2') else 'current_old'
                assert pr['metrics']==independent_metrics(pr['rows'],r['inputs'][target])
                for label in ('unchanged','revised'):
                    if stage in ('revised','day3'):assert pr[label+'_metrics']==independent_metrics(pr['rows'],r['inputs'][label])
                if 'new_probe' in pr:
                    npb=pr['new_probe'];assert npb['initial_history']==h.hex() and abs(npb['initial_time']-t)<1e-7
                    verify_rows(bytes.fromhex(npb['raw_hex']),npb['rows'],h,t,False);checked+=len(npb['rows'])
                    assert npb['metrics']==independent_metrics(npb['rows'],r['inputs']['new'])
                state=a['states'][f'{stage}/{branch}'];assert abs(state['clock']-t)<1e-7 and state['bytes_seen']==bytes_seen
        for stage,group in a['probes'].items():
            reference=group['W'];pr=group['ERASE']
            assert pr['erase'] and all(np.allclose(x['probabilities'],.25,rtol=0,atol=1e-12) for x in pr['rows'])
            assert pr['initial_history']==reference['initial_history'] and pr['initial_time']==reference['initial_time']
            verify_rows(bytes.fromhex(pr['raw_hex']),pr['rows'],bytes.fromhex(pr['initial_history']),pr['initial_time'],False)
            checked+=len(pr['rows'])
        if not r['quick']:
            # Until the revision phase, W and N_REV must have identical state and every output.
            for stage in ('old','day1','new','day2'):
                assert a['states'][f'{stage}/W']['digest']==a['states'][f'{stage}/N_REV']['digest'],'premature revision intervention'
                assert a['probes'][stage]['W']['rows']==a['probes'][stage]['N_REV']['rows']
    for stage in r['arms']['A']['probes']:
        for branch in config.BRANCHES:
            assert r['arms']['A']['probes'][stage][branch]['raw_hex']==r['arms']['B']['probes'][stage][branch]['raw_hex']
            assert r['arms']['A']['probes'][stage][branch]['initial_time']==r['arms']['B']['probes'][stage][branch]['initial_time']
    return dict(verdict='PASS',checked_rows=checked,science_worlds=0)

def rejects(fn):
    try:fn()
    except (AssertionError,ValueError,KeyError):return
    raise AssertionError('negative control accepted')

def basic_tests():
    checks=[]
    # Same address, independently born native stores, and B only accesses native writable/readable rows.
    a=environment.native_core.ObservationCore();b=LocalContentCore()
    for history in (b'',b'3012',b'3333',b'3101'):
        a.history=b.history=history;a.predict(config.DT);b.predict(config.DT)
        assert all(np.array_equal(x,b.cached['code']) for x in a.cached['codes'])
        a.cached=b.cached=None
    assert len({s.birth_record['fly_id'] for s in a.stores})==4
    for i,s in enumerate(a.stores):
        for q in a.stores[i+1:]:
            for name in ('fast','slow','adapt'):assert not np.shares_memory(getattr(s.fly.m,name),getattr(q.fly.m,name))
    checks.append('identical pre-arrival KC addresses and four independent A births')
    b=LocalContentCore();b.predict(config.DT);c=copy.deepcopy(b.cached);old=b.fast.copy();ids=c['ids'];z=c['z']
    target=48;error=np.eye(4)[0]-c['p'];expected=config.ETA*z[:,None]*error[None,:]
    row=b.observe(target,config.DT)
    assert np.allclose(b.fast[ids],np.clip((1-config.SLOW_SHARE)*expected,-2,2),rtol=0,atol=1e-15)
    assert np.allclose(b.slow[ids],np.clip(config.SLOW_SHARE*expected,-2,2),rtol=0,atol=1e-15)
    other=np.ones(b.encoder.n,bool);other[ids]=False;assert np.array_equal(b.fast[other],old[other])
    checks.append('independent local outer-product arithmetic; inactive rows untouched')
    # Independent dense time evolution vs lazy row advancement, including no-write/rest/zero dt.
    eager_f=np.zeros_like(b.fast);eager_s=np.zeros_like(b.slow);b=LocalContentCore();clock=0.
    for i,byte in enumerate(b'301230123123'):
        if i==5:b.rest(86400.)
        t=b.t+config.DT;dt=t-clock
        eager_f*=math.exp(-dt/config.FAST_TAU);eager_s*=math.exp(-dt/config.SLOW_TAU)
        result=b.predict(t);c=b.cached;ids,z=c['ids'],c['z']
        values=z@(eager_f[ids]+eager_s[ids]);logits=values/config.SCALE;p=np.exp(logits-np.logaddexp.reduce(logits))
        assert np.allclose(p,result['probabilities'],rtol=0,atol=1e-14)
        b.plastic=i not in (4,6)
        if b.plastic:
            error=np.eye(4)[config.ALPHABET.index(byte)]-p
            delta=config.ETA*z[:,None]*error[None,:]
            eager_f[ids]=np.clip(eager_f[ids]+.2*delta,-2,2);eager_s[ids]=np.clip(eager_s[ids]+.8*delta,-2,2)
        b.observe(byte,t);clock=t
        f,s=b._view(np.arange(b.encoder.n),t)
        assert np.allclose(f,eager_f,rtol=0,atol=1e-14) and np.allclose(s,eager_s,rtol=0,atol=1e-14)
    checks.append('lazy row decay agrees with independent eager evolution including disabled writes')
    # Native禁写 compared to the actual native event, not receipt flags.
    a=environment.native_core.ObservationCore(plastic=False);a.predict(config.DT)
    references=[]
    for store,code in zip(a.stores,a.cached['codes']):
        ref=store.fly.clone();ref.event(config.DT,code,0.,False);references.append(ref)
    a.observe(48,config.DT)
    for s,ref in zip(a.stores,references):
        for name in ('fast','slow','adapt'):assert np.array_equal(getattr(s.fly.m,name),getattr(ref.m,name))
    checks.append('A disabled writes equal independent native no-plastic arrays')
    for model in (a,b):
        rejects(lambda:model.observe(48,model.t+config.DT));rejects(lambda:model.predict(float('nan')))
        model.predict(model.t+config.DT);rejects(lambda:model.observe(49,model.t+2*config.DT));model.cached=None
    rejects(lambda:bench.run(823001,True))
    checks.append('future/out-of-order observations and all science admission rejected')
    return dict(verdict='PASS',checks=checks,eligible_rows=int(b.encoder.mask.sum()),kc_count=b.encoder.n,B_mutable_bytes=b.mutable_bytes(),science_worlds=0)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--receipt',type=Path)
    args=p.parse_args();result=basic_tests()
    if args.receipt:
        with gzip.open(args.receipt,'rt') as f:r=json.load(f)
        result['receipt_audit']=audit(r)
    args.out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
