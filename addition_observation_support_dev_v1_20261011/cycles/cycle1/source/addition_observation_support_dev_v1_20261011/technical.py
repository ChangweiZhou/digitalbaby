"""Finite technical tests; no science selection or parameter fitting."""
import copy, importlib.util, math, sys
import numpy as np
import config
from cores import ByteLearner,Core,features,emitted
from reference import Reference
from runner import consume,audit
from operations import save_checkpoint,load_checkpoint
from decision import synthetic_suite,select,confirm,first_crossing

def reject(call):
    try:call()
    except (AssertionError,ValueError,KeyError,TypeError):return True
    raise AssertionError('negative fixture accepted')
def restore_reference(model):
    ref=Reference(model.core.arm)
    for k,v in model.core.state().items():
        if k in ref.a:ref.a[k]=copy.deepcopy(v)
        elif k=='rng':ref.rng.setstate(v)
        else:setattr(ref,k,copy.deepcopy(v))
    return ref

def baseline_parity():
    spec=importlib.util.spec_from_file_location('accepted_feedback_core',config.PRIOR/'cores.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    actual,prior=Core('COV-R'),old.Core('COV-R')
    for i in range(240):
        ids=features(b'ABCDE'[i%5],b'FGHIJ'[(i*3)%5]);y=(i*7)%9;allowed=i%11!=0
        a,b=actual.value(ids),prior.value(ids)
        assert a==b and emitted(a)==old.emitted(b)
        actual.update(ids,y,allowed,a);prior.update(ids,y,allowed,b)
        assert np.array_equal(actual.w,prior.w) and np.array_equal(actual.p,prior.p)
        assert actual.updates==prior.updates and actual.rng.getstate()==prior.rng.getstate()
    return {'accepted_source_numeric_parity_transitions':240,'baseline_equations_changed':False}

def support_boundaries():
    passed=[]
    for history,target,eligible,factor in [([],3,False,1.),([2],3,False,1.),([2,2],3,True,.25),
          ([2,2],2,True,1.),([2,3],2,False,1.),([2,2,3,4],5,False,1.),([2,2,3],4,True,.25)]:
        m=Core('COV-SUPPORT');ids=features(65,66)
        if history:m.cache=[{'key':[65,66],'answers':list(history)}]
        ref=Reference('COV-SUPPORT');ref.cache=copy.deepcopy(m.cache)
        out=m.update(ids,target,True,m.value(ids));expected=ref.step(ids,target,True,ref.predict(ids))
        assert out==expected and m.digest()==ref.digest()
        assert out['gate']['eligible'] is eligible and out['gate']['factor']==factor
        assert m.cache[0]['answers']==(history+[target])[-8:]
        passed.append('past_'+str(history)+'_target_'+str(target))
    # Opposite orientations are not pooled; current answer cannot create its own support.
    m=Core('COV-SUPPORT')
    ids=features(65,66)
    for y in (2,2):m.update(ids,y,True,m.value(ids))
    out=m.update(features(66,65),3,True,m.value(features(66,65)))
    assert out['gate']['factor']==1. and out['gate']['prior_count']==0
    # Evidence for an initial wrong majority remains revisable; the window retains conflicts.
    m=Core('COV-SUPPORT');m.cache=[{'key':[65,66],'answers':[0,0]}]
    factors=[]
    for y in [4]*9:factors.append(m.update(ids,y,True,m.value(ids))['gate']['factor'])
    assert factors[0]==.25 and factors[-1]==1. and m.cache[0]['answers']==[4]*8
    # FIFO capacity, non-LRU reads and no-write bookkeeping.
    m=Core('COV-SUPPORT');pairs=[features(a,b) for a in config.OPERANDS for b in config.OPERANDS][:33]
    for i,p in enumerate(pairs[:32]):m.update(p,i%9,True,m.value(p))
    first=copy.deepcopy(m.cache[0]['key']);m.update(pairs[0],1,True,m.value(pairs[0]))
    assert m.cache[0]['key']==first
    m.update(pairs[32],2,True,m.value(pairs[32]));assert len(m.cache)==32 and first not in [r['key'] for r in m.cache]
    seal=m.digest();m.update(pairs[0],8,False,m.value(pairs[0]));assert m.digest()==seal
    # Exact numerical fractional-information update, independent explicit matrix algebra.
    m=Core('COV-SUPPORT');m.cache=[{'key':[65,66],'answers':[0,0]}]
    w=m.w.copy();p=m.p.copy();x=np.zeros(31);x[list(ids)]=1.;q=p@x;gamma=.25/8.;d=1+gamma*float(x@q)
    m.update(ids,8,True,0.)
    assert np.allclose(m.w,np.clip(w+gamma*8*q/d,-4*config.KAPPA,4*config.KAPPA),rtol=0,atol=1e-15)
    assert np.allclose(m.p,p-gamma*np.outer(q,q)/d,rtol=0,atol=1e-15)
    np.linalg.cholesky(m.p)
    # RATE cannot use current conflict to decide its multiplier. Same past/RNG => same gate.
    a=Core('COV-RATE');a.cache=[{'key':[65,66],'answers':[2,2]}];a.history=[0,1,0,1]
    b=a.clone();o1=a.update(ids,2,True,0.);o2=b.update(ids,3,True,0.)
    assert o1['gate']['factor']==o2['gate']['factor'] and o1['gate']['random']==o2['gate']['random']
    assert o1['gate']['conflict'] is False and o2['gate']['conflict'] is True
    # Both gates maintain exactly the same auxiliary statistics on one actual stream.
    a,b=Core('COV-SUPPORT'),Core('COV-RATE')
    for i in range(140):
        p=features(65+i%5,65+(i*2+1)%5);y=(i//5)%9
        a.update(p,y,True,a.value(p));b.update(p,y,True,b.value(p))
        assert a.cache==b.cache and a.history==b.history and a.rng.getstate()==b.rng.getstate()
    # Published earliest-opportunity prediction on a public, geometry-only fixture.
    slots=[(a,b) for u,v in config.CONTRACT['confirmation_edges'] for a,b in ((u,v),(v,u))]
    m=Core('COV-SUPPORT');flags=[]
    for a,b in slots*2:
        p=features(70+a,70+b);flags.append(m.update(p,4,True,m.value(p))['gate']['factor'])
    assert flags==[1.]*20
    p=features(70+slots[0][0],70+slots[0][1]);assert m.update(p,5,True,m.value(p))['gate']['factor']==.25
    return {'verdict':'PASS','boundary_cases':passed,'same_gamma_weight_information':True,
      'current_answer_not_supporting_itself':True,'wrong_majority_window_revision':True,
      'bounded_fifo_non_LRU':True,'random_gate_current_conflict_independence':True,
      'paired_auxiliary_state_parity':True,'first_20_no_NEW_attenuation':True,'geometry_fixture_not_science':True}

def synthetic_decisions():
    def item(j=.2):
        return {'W':{'J':j,'old_score':.7,'old_final':.95,'new_final':j},
                'new_causal_N':.5,'new_causal_shuffle':.5,'old_causal_N':.5,'old_causal_shuffle':.5,
                'production_cpu_seconds':1.,'payload_bytes':15000,'peak_rss_bytes':50*1024**2}
    rows=[]
    for f in ('PATH_LOOP','STAR_LOOP'):
        for n in ('STANDARD','BLURRY'):
            for i in range(3):rows.append(dict(world=len(rows),family=f,noise=n,**{a:item() for a in config.ARMS}))
    assert select(rows)['selected'] is None
    for r in rows:r['COV-RATE']=item(.6)
    assert select(rows)['selected'] is None
    for r in rows:r['COV-SUPPORT']=item(.4)
    assert select(rows)['selected']=='COV-SUPPORT'
    duplicate=copy.deepcopy(rows);duplicate[1]['world']=duplicate[0]['world'];reject(lambda:select(duplicate))
    records=[]
    for i in range(128):
        ordinary=i>=96;noise='BLURRY' if i<64 or 96<=i<112 else 'STANDARD'
        r=dict(world=i,family='TRIANGLE_TAIL',ordinary=ordinary,noise=noise,shortcut_J={'mode':.1})
        r['COV-R']=item(.2);r['COV-SUPPORT']=item(.6);r['COV-RATE']=item(.6);records.append(r)
    assert confirm(records,'COV-SUPPORT')['verdict'].startswith('ADOPT_')
    reject(lambda:confirm(records,'COV-RATE'))
    cases=[]
    for k in ('new_causal_N','new_causal_shuffle','old_causal_N','old_causal_shuffle'):
        bad=copy.deepcopy(records)
        for r in bad:r['COV-SUPPORT'][k]=0.
        assert confirm(bad,'COV-SUPPORT')['verdict']=='RETAIN_COV_R';cases.append(k)
    for kind in ('cost','guard','gain','shortcut'):
        bad=copy.deepcopy(records)
        for r in bad:
            if kind=='cost':r['COV-SUPPORT']['production_cpu_seconds']=2.
            if kind=='guard':r['COV-SUPPORT']['W']['old_final']=.1
            if kind=='gain':r['COV-SUPPORT']['W']['J']=.21
            if kind=='shortcut':r['shortcut_J']={'mode':.9}
        assert confirm(bad,'COV-SUPPORT')['verdict']=='RETAIN_COV_R';cases.append(kind)
    return {'verdict':'PASS_SYNTHETIC_ONLY','fixed_sole_candidate':True,'negative_cases':cases,'science_selection':None}

def suite(folder,cycle):
    folder.mkdir(parents=True,exist_ok=False);out={};baseline=baseline_parity();boundaries=support_boundaries()
    for arm in config.ARMS:
        model,ref=ByteLearner(arm),Reference(arm);birth=model.core.digest()
        frozen=copy.deepcopy(model.core.state())
        for raw in (b'A+B=8\n',b'A+B=\n'):consume(model,ref,raw,False)
        assert model.core.digest()==birth
        raws=[bytes((65+i%5,43,65+(i*2+1)%5,61,48+i%9,10)) for i in range(64)]
        for i,raw in enumerate(raws):
            consume(model,ref,raw,True)
            if i==23:save_checkpoint(model,folder/(arm+'.json.gz'))
        ref.invariants();assert not frozen['cache']
        restored=load_checkpoint(folder/(arm+'.json.gz'));shadow=restore_reference(restored)
        for raw in raws[24:]:consume(restored,shadow,raw,True)
        assert restored.core.digest()==model.core.digest() and restored.ticks==model.ticks
        clone=model.clone();assert not np.shares_memory(clone.core.w,model.core.w) and not np.shares_memory(clone.core.p,model.core.p)
        consume(clone,ref.clone(),b'E+A=8\n',True);assert model.core.digest()==restored.core.digest()
        unchanged=model.core.digest();consume(model,ref,b'E+A=8\n',False);assert model.core.digest()==unchanged
        model.core.erase();assert model.core.digest()==birth
        out[arm]={'resume_parity':True,'no_write_all_state':True,'independent_equations':True,
          'same_fractional_information':True,'payload_bytes':restored.core.payload_bytes(),
          'external_updates':restored.core.updates,'extra_updates':restored.core.extra_updates,
          'final_core_seal':restored.core.digest(),'final_ticks':restored.ticks,'final_bytes_seen':restored.bytes_seen}
    assert first_crossing([.2,.8,.9],[20,40,80],.75,320)['total_exposure']==360
    from reference import answer
    assert all(emitted(v)==answer(v) for v in [-.1,0.,.5,.5-1e-12,.5+1e-12,7.5,8.,9.])
    negatives=[lambda:config.admit(999999,'PATH_LOOP'),lambda:config.admit(1004201,'TRIANGLE_TAIL'),
      lambda:config.admit(1004201,'PATH_LOOP','BLURRY',False),lambda:ByteLearner('COV-R').feed(43,1,True),
      lambda:Core('COV-R').update(features(65,66),9,True,0.)]
    rejected=sum(reject(x) for x in negatives)
    partial=ByteLearner('COV-R');partial.feed(65,1,True)
    rejected+=int(reject(lambda:save_checkpoint(partial,folder/'partial.json.gz')))
    return {'verdict':'PASS','cycle':cycle,'arms':out,'baseline':baseline,'support':boundaries,
      'statistics':synthetic_suite(),'decision_gates':synthetic_decisions(),'negative_cases':rejected,'science_worlds':0}

def tamper_suite(receipt):
    mutations={
      'old_permission':lambda r:r['branches']['N_OLD']['old_records'][0].__setitem__('allowed',True),
      'new_permission':lambda r:r['extra_forks']['N_NEW']['new_records'][0].__setitem__('allowed',True),
      'bad_prediction':lambda r:r['branches']['W']['old_records'][0]['events'][3]['prediction'].__setitem__('emitted',57),
      'covariance_in_N_NEW':lambda r:r['extra_forks']['N_NEW']['new_checkpoints']['20']['checkpoint']['state']['p']['array'].__setitem__(0,[99]*31),
      'query_write':lambda r:r['extra_forks']['N_NEW']['new_checkpoints']['40']['delayed_new_held']['rows'][0]['responses'][0]['events'][0].__setitem__('update',{'performed':True}),
      'omitted_new_control':lambda r:r['extra_forks'].pop('SHUFFLED_NEW'),
      'missing_checkpoint':lambda r:r['branches']['W']['forks']['320']['new_checkpoints'].pop('20'),
      'changed_new_target':lambda r:r['master']['new'][0].__setitem__('target',999),
      'source':lambda r:r['source'].__setitem__('unlocked','bad'),
      'state_seal':lambda r:r['branches']['W']['old_records'][0].__setitem__('core_seal','bad'),
      'payload':lambda r:r['extra_forks']['N_NEW']['final_checkpoint'].__setitem__('payload_bytes',0),
      'resource_count':lambda r:r['extra_forks']['N_NEW']['final_checkpoint'].__setitem__('scalar_ops',0),
      'record_count':lambda r:r.__setitem__('training_records',2400),
      'new_clock':lambda r:r['extra_forks']['N_NEW']['final_checkpoint'].__setitem__('ticks',0),
      'duplicate_record':lambda r:r['branches']['W']['old_records'].append(r['branches']['W']['old_records'][0]),
      'bad_score':lambda r:r['extra_forks']['SHUFFLED_NEW']['final_new_held'].__setitem__('accuracy',1.01),
    }
    records=receipt['branches']['W']['old_records']
    idx=next(i for i,r in enumerate(records) if len(r['events'])==6)
    for field,value in [('factor',.123),('prior_count',999),('mode',99),('gamma',99.),('probability',2.),('random',-1.)]:
        mutations['gate_'+field]=lambda r,f=field,v=value:r['branches']['W']['old_records'][idx]['events'][4]['update']['gate'].__setitem__(f,v)
    for field in ('cache','history','rng'):
        mutations['N_NEW_'+field]=lambda r,f=field:r['extra_forks']['N_NEW']['new_checkpoints']['20']['checkpoint']['state'].__setitem__(f,{'cache':[{'key':[65,65],'answers':[8]}],'history':[1],'rng':[]}[f])
    for name,change in mutations.items():
        candidate=copy.deepcopy(receipt);change(candidate);reject(lambda:audit(candidate))
    return {'verdict':'PASS','rejected':list(mutations),'science_worlds':0}
