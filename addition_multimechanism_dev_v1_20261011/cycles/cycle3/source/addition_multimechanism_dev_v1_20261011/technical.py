"""Public algebra/causality fixtures. No gain-based recipe selection."""
import copy, importlib.util, json, math
from pathlib import Path
import numpy as np
import config
import cores
from cores import Core,ByteLearner,features,A
from reference import Reference,input_vector
from runner import consume,atomic_json,audit
from operations import save_checkpoint,load_checkpoint

def reject(fn):
    try:fn()
    except (AssertionError,ValueError,KeyError,TypeError):return True
    raise AssertionError('tamper unexpectedly accepted')
def reference_from(model):
    ref=Reference(model.core.arm);ref.s=copy.deepcopy(model.core.state());return ref
def baselines():
    spec=importlib.util.spec_from_file_location('baseline_parent_check',config.PRIOR/'cores.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    for arm in config.BASELINES:
        a,b=Core(arm),old.Core(arm)
        for i in range(240):
            ids=features(65+i%5,70+(i*3)%5);target=(i//5)%9;allowed=i%11!=0
            assert a.value(ids)==b.value(ids)
            out=a.update(ids,target,allowed,a.value(ids));out.pop('mechanism')
            assert out==b.update(ids,target,allowed,b.value(ids))
            assert a.digest()==b.digest()
    return dict(bit_identical_parent_baselines=list(config.BASELINES),transitions_per_arm=240)

def kci_cases():
    rng=np.random.default_rng(773);z=rng.normal(size=(31,31));p=z@z.T+np.eye(31)
    ids=features(65,66);x=input_vector(ids);w=np.full(31,4*config.KAPPA-.002)
    exercised=[]
    assert (A.sum(axis=0)[:30]==2).all() and not np.diag(A).any()
    assert not A[:15,15:].any() and not A[15:,:15].any() and not A[30,:].any()
    for saturation in (False,True):
        models={arm:Core(arm) for arm in ('COV-SUPPORT','KCI','KCI-GLOBAL-NORM')}
        for m in models.values():m.p=p.copy();m.w=w.copy() if saturation else rng.normal(size=31)*.1
        # Compare identical prestate, not unrelated random initial weights.
        for m in models.values():m.w=models['COV-SUPPORT'].w.copy()
        before=models['KCI'].w.copy();q=p@x;v=models['KCI'].value(ids)
        out={arm:m.update(ids,8,True,v) for arm,m in models.items()}
        assert np.array_equal(models['KCI'].p,models['COV-SUPPORT'].p)
        assert models['KCI'].value(ids)==models['COV-SUPPORT'].value(ids)
        event=out['KCI']['mechanism'];g=np.array(event['g'])
        assert np.array_equal(g[list(ids)],np.ones(3)) and (g<=1).all() and (g>0).all()
        assert np.any((g<1)&(np.abs(q)>1e-6)), 'inactive correction gate was inert'
        assert abs(event['proposed_norm']-out['KCI-GLOBAL-NORM']['mechanism']['proposed_norm'])<1e-12
        np.linalg.cholesky(models['KCI'].p)
        exercised.append('saturation' if saturation else 'correlated_P')
    # Severing adjacency exercises the actual production path, not only a formula.
    saved=cores.A
    try:
        cores.A=np.zeros_like(A)
        a,b=Core('KCI'),Core('COV-SUPPORT');a.p=p.copy();b.p=p.copy()
        a.update(ids,3,True,0.);b.update(ids,3,True,0.)
        assert np.array_equal(a.w,b.w) and np.array_equal(a.p,b.p)
    finally:cores.A=saved
    saved_lambda=config.KCI_LAMBDA
    try:
        config.KCI_LAMBDA=0.
        a,b=Core('KCI'),Core('COV-SUPPORT');a.p=p.copy();b.p=p.copy()
        a.update(ids,3,True,0.);b.update(ids,3,True,0.)
        assert np.array_equal(a.w,b.w) and np.array_equal(a.p,b.p)
    finally:config.KCI_LAMBDA=saved_lambda
    perm=list(range(15));np.random.default_rng(55).shuffle(perm)
    shuffled=np.zeros_like(A)
    for offset in (0,15):
        for src in range(15):
            for dst in np.flatnonzero(A[:,offset+src]):shuffled[offset+perm[dst-offset],offset+perm[src]]=1.
    assert np.array_equal(A.sum(axis=0),shuffled.sum(axis=0)) and not np.diag(shuffled).any()
    assert not np.array_equal(A,shuffled)
    zero=Core('KCI');zero.p=np.zeros((31,31));out=zero.update(ids,0,True,0.)
    assert out['mechanism']['ratio']==1 and not zero.w.any()
    equal=Core('KCI');equal.update(ids,0,True,0.);assert not equal.w.any()
    return dict(cases=exercised+['zero_adjacency_baseline','zero_lambda_baseline','zero_residual','zero_q','degree_preserved_shuffle'],graph_edges=int(A.sum()))

def retro_cases():
    ids=features(65,66);m=Core('RETRO');fixed=Core('RETRO-FIXED');ref=Reference('RETRO')
    all_rows=[];changes=0;restores=0;mode_losses=0
    # Forced majority, loss, reversal and FIFO freezing; no hidden clean answer.
    seq=[8,2,2,2,2,7,7,7,7,7,7,7,7]+[i%9 for i in range(45)]
    for y in seq:
        state_before=copy.deepcopy(m.state());gamma_ref=ref.step(ids,y,True,ref.predict(ids))
        out=m.update(ids,y,True,m.value(ids));fixed.update(ids,y,True,fixed.value(ids))
        assert out==gamma_ref and m.digest()==ref.digest()
        all_rows.append(dict(ids=list(ids),target=y,weight=out['gate']['gamma']))
        for changed in out['mechanism']['changes']:
            row=m.ledger[changed['index']];all_rows[row['arrival_index']]['weight']=row['effective']
            changes+=1;restores+=int(changed['delta']>0)
        mode_losses+=int(out['mechanism']['mode'] is None)
        h=np.eye(31)*.1;b=np.zeros(31)
        for row in all_rows:
            v=input_vector(row['ids']);h+=row['weight']*np.outer(v,v);b+=row['weight']*row['target']*v
        assert np.allclose(m.h,h,rtol=0,atol=1e-11) and np.allclose(m.b,b,rtol=0,atol=1e-11)
        assert m.ledger[-1]['effective']==m.ledger[-1]['arrival'],'current record revised'
        assert len(out['mechanism']['changes'])<=7
        ref.invariants()
    assert changes>0 and restores>0 and mode_losses>0 and len(m.ledger)==32
    assert not np.array_equal(m.h,fixed.h)
    # No-mode and unanimous streams are exactly fixed-writer paths in w/P/H/b.
    a,b=Core('RETRO'),Core('RETRO-FIXED')
    for i in range(12):
        for model in (a,b):model.update(ids,2,True,model.value(ids))
        for key in ('w','p','h','b'):assert np.array_equal(getattr(a,key),getattr(b,key))
    return dict(nonzero_revisions=changes,restorations=restores,majority_losses=mode_losses,
                evictions=len(seq)-32,reconstruction_includes_evicted=True)

def role_cases():
    a=Core('ROLE');b=Core('ROLE-FIXED');perm=Core('ROLE-PERMUTED')
    ab=features(65,66);ba=features(66,65)
    first=a.update(ab,2,True,a.value(ab));assert first['mechanism']['eta']==0 and len(a.queue)==1
    second=a.update(ba,2,True,a.value(ba));assert second['mechanism']['paired'] and len(a.queue)==0
    assert a.agree==1 and abs(second['mechanism']['eta']-.01)<1e-15
    assert second['mechanism']['coupled']==[[5,5],[6,6]]
    a.update(ab,8,False,a.value(ab));assert a.agree==1 and not a.queue
    # A zero-residual self observation can mark roles but never creates a pair vote.
    cc=features(67,67);a.update(cc,0,True,a.value(cc));assert a.agree==1 and not a.queue
    for i in range(40):
        ids=features(65,66+i%4);a.update(ids,3,True,a.value(ids))
    assert len(a.queue)<=32
    # The true relation and permuted null have equal eligibility/counter schedules.
    t=Core('ROLE')
    for i in range(30):
        ids=ab if i%2==0 else ba;y=2
        x=t.update(ids,y,True,t.value(ids))['mechanism']
        z=perm.update(ids,y,True,perm.value(ids))['mechanism']
        assert (x['agree'],x['disagree'],x['eta'])==(z['agree'],z['disagree'],z['eta'])
        assert [v[0] for v in x['coupled']]==[v[0] for v in z['coupled']]
    return dict(reverse_consumed_once=True,birth_gate_zero=True,local_current_symbols_only=True,queue_bound=32)

def cover_cases():
    a=Core('COVER');s=Core('COV-SUPPORT')
    # Repeated maximum-leverage selection is deliberately observable.
    seq=[features(65,66),features(67,68)]+[features(65,66)]*10
    chosen=[]
    for i,ids in enumerate(seq):
        y=2 if i%3 else 8
        o=a.update(ids,y,True,a.value(ids));s.update(ids,y,True,s.value(ids))
        chosen.append(o['mechanism']['selected_arrival'])
    assert a.extra_updates==len(seq)-1
    assert len(a.buffer)==len(seq) and np.array_equal(a.p,s.p) is False
    # Residual-dependent gamma diverges across lifetimes; only same-prestate P parity is valid.
    c=Core('COVER');d=Core('COV-SUPPORT');c.buffer=copy.deepcopy(a.buffer[:4])
    c.w=a.w.copy();d.w=a.w.copy();c.p=a.p.copy();d.p=a.p.copy()
    ids=features(65,66);v=c.value(ids)
    c.update(ids,2,True,v);d.update(ids,2,True,v);assert np.array_equal(c.p,d.p)
    assert c.cache==d.cache and c.history==d.history,'replay added votes'
    return dict(one_extra_weight_only_step=True,pre_insertion=True,repeated_winners=chosen,
                unique_selected=len({i for i in chosen if i is not None}))

def suite(folder,cycle):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    basics=baselines();operator=dict(KCI=kci_cases(),RETRO=retro_cases(),ROLE=role_cases(),COVER=cover_cases())
    final={};negative=0
    for arm in config.ARMS:
        model=ByteLearner(arm);ref=Reference(arm);birth=model.core.digest()
        for i in range(64):
            raw=bytes((b'ABCDE'[i%5],43,b'ABCDE'[(2*i+1)%5],61,48+i%5,10))
            if i==24:save_checkpoint(model,folder/(arm+'.json.gz'))
            consume(model,ref,raw,True)
        ref.invariants();final[arm]=dict(final_core_seal=model.core.digest(),final_ticks=model.ticks,final_bytes_seen=model.bytes_seen)
        start=model.core.digest();rng=model.core.state()['rng']
        for raw in (b'A+B=8\n',b'A+B=\n'):
            for byte in raw:model.feed(byte,model.ticks+1,False)
        assert model.core.digest()==start and model.core.state()['rng']==rng
        child=model.clone();child.core.erase();assert child.core.digest()==birth and model.core.digest()==start
        state=copy.deepcopy(model.core.state());first=next(iter(state));del state[first]
        negative+=int(reject(lambda:model.core.restore(state)))
        # Query values cannot trigger updates or alter support/replay/reweight/coupling.
        for byte in b'B+A=\n':child.feed(byte,child.ticks+1,False)
        assert child.core.digest()==birth
        cold=load_checkpoint(folder/(arm+'.json.gz'))
        for i in range(24,64):
            for byte in bytes((b'ABCDE'[i%5],43,b'ABCDE'[(2*i+1)%5],61,48+i%5,10)):
                cold.feed(byte,cold.ticks+1,True)
        assert cold.core.digest()==start and cold.ticks==final[arm]['final_ticks'] and cold.bytes_seen==final[arm]['final_bytes_seen']
    for bad in ((999,'PATH_LOOP'),(1304701,'STAR_LOOP'),(True,'PATH_LOOP')):
        negative+=int(reject(lambda args=bad:config.admit(*args)))
    result=dict(verdict='PASS_TECHNICAL_ONLY',cycle=cycle,baselines=basics,operators=operator,
                arms=final,negative_cases=negative,science_worlds=0)
    atomic_json(result,folder/'RESULT.json');return result

def tamper_suite(receipt,folder=None):
    def target_event(r):
        return next(e['update'] for row in r['branches']['W']['old_records'] for e in row['events']
                    if e['update'] and e['update']['performed'])
    def corrupt_gate(r):target_event(r)['gate']['gamma']+=.01
    def corrupt_auxiliary(r):
        state=r['branches']['W']['forks']['320']['final_checkpoint']['state']
        if 'extra' in state:state['extra']['events']+=1
        else:state['updates']+=1
    def corrupt_mechanism(r):
        event=target_event(r)
        if event['mechanism'] is None:event['mechanism']={'invented':True}
        else:
            v=event['mechanism'];arm=r['arm']
            if arm.startswith('KCI'):v['g'][0]=1.123
            elif arm.startswith('ROLE'):v['eta']=.123
            elif arm.startswith('COVER'):v['chosen']=999
            elif arm.startswith('RETRO'):v['mode']=99
    tests={
      'forbidden_old_writes':lambda r:r['branches']['N_OLD']['old_records'][0].update(allowed=True),
      'forbidden_new_writes':lambda r:r['extra_forks']['N_NEW']['new_records'][0].update(allowed=True),
      'source':lambda r:r['source'].update(forged='bad'),
      'counts':lambda r:r.update(training_records=2559),
      'output':lambda r:r['branches']['W']['old_records'][0]['events'][3]['prediction'].update(emitted=57),
      'state':lambda r:r['branches']['W']['birth']['state']['w']['array'].__setitem__(0,1.),
      'scope':lambda r:r['extra_forks'].pop('N_NEW'),
      'science_namespace':lambda r:r.update(world=999999),
      'support_gamma':corrupt_gate,
      'complete_auxiliary_state':corrupt_auxiliary,
      'mechanism_event':corrupt_mechanism,
      'teacher_in_query':lambda r:r['branches']['W']['forks']['320']['final_new_held']['rows'][0]['responses'][0]['events'][4].update(update={'performed':True}),
      'nonfinite_CPU':lambda r:r.update(inclusive_cpu_seconds=float('inf')),
      'negative_RSS':lambda r:r.update(peak_rss_bytes=-1),
      'nonfinite_record_CPU':lambda r:r['branches']['W']['old_records'][0].update(production_cpu_seconds=float('inf')),
      'fixed_storage':lambda r:r['branches']['W']['birth'].update(fixed_asset_bytes=-1),
      'payload':lambda r:r['branches']['W']['birth'].update(payload_bytes=1),
      'scalar_ops':lambda r:r['branches']['W']['birth'].update(scalar_ops=123),
      'traffic':lambda r:r['branches']['W']['birth'].update(estimated_algorithmic_bytes_moved=123),
    }
    rejected=[]
    for name,alter in tests.items():
        r=copy.deepcopy(receipt);alter(r);reject(lambda:audit(r));rejected.append(name)
    if folder is not None:
        from runner import commit
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
        forged=copy.deepcopy(receipt);forged['source']['fake']='fake'
        reject(lambda:commit(forged,folder/'forged_source.json.gz'));rejected.append('actual_commit_source')
        forged=copy.deepcopy(receipt);forged['training_records']=0
        reject(lambda:commit(forged,folder/'failed_audit.json.gz'));rejected.append('actual_commit_failed_audit')
        assert (folder/'failed_audit.json.gz.FAILED.json').exists()
        assert (folder/'failed_audit.json.gz.provisional').exists()
        assert not (folder/'failed_audit.json.gz.claim').exists()
        reject(lambda:commit(receipt,folder/'failed_audit.json.gz'));rejected.append('failed_commit_retry_blocked')
    return dict(verdict='PASS',rejected=rejected,science_worlds=0)
