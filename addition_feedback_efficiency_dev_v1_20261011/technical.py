"""Meaningful local invariants and adversarial whole-receipt audits."""
import copy
import importlib.util
import json
import math
import sys
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
    ref=Reference(model.core.arm);state=model.core.state()
    for k,v in state.items():
        if k in ref.a:ref.a[k]=copy.deepcopy(v)
        elif k=='rng':ref.rng.setstate(v)
        else:setattr(ref,k,copy.deepcopy(v))
    return ref

def baseline_parity():
    # Import the already published implementation without changing parent files.
    sys.path.insert(0,str(config.PRIOR))
    spec=importlib.util.spec_from_file_location('accepted_sample_gen_core',config.PRIOR/'cores.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sys.path.remove(str(config.PRIOR))
    count=0
    for arm in ('COV-R','DIAG-R'):
        actual,prior=Core(arm),module.Core(arm)
        for i in range(240):
            ids=features(b'ABCDE'[i%5],b'FGHIJ'[(i*3)%5]);y=(i*7)%9;allowed=i%11!=0
            a,b=actual.value(ids),prior.value(ids)
            assert a==b and emitted(a)==module.emitted(b)
            actual.update(ids,y,allowed,a);prior.update(ids,y,allowed,b)
            assert np.array_equal(actual.w,prior.w) and np.array_equal(actual.p,prior.p)
            assert actual.updates==prior.updates and actual.rng.getstate()==prior.rng.getstate()
            count+=1
    return {'accepted_source_numeric_parity_transitions':count,'baseline_equations_changed':False}

def synthetic_decisions():
    def item(j=.2):
        return {'W':{'J':j,'old_score':.7,'old_final':.95,'new_final':j},
                'new_causal_N':.5,'new_causal_shuffle':.5,'old_causal_N':.5,'old_causal_shuffle':.5,
                'production_cpu_seconds':1.,'payload_bytes':15000,'peak_rss_bytes':50*1024**2}
    dev=[]
    for f in ('PATH_LOOP','STAR_LOOP'):
        for n in ('STANDARD','BLURRY'):
            for i in range(3):dev.append(dict(world=len(dev),family=f,noise=n,**{a:item() for a in config.ARMS}))
    assert select(dev)['selected'] is None
    duplicate=copy.deepcopy(dev);duplicate[1]['world']=duplicate[0]['world'];reject(lambda:select(duplicate))
    malformed=copy.deepcopy(dev);malformed[0]['COV-R']['W']['J']=float('nan');reject(lambda:select(malformed))
    for row in dev:row['COV-EXTRA']=item(.4)
    assert select(dev)['selected']=='COV-EXTRA'  # control winners are not discarded
    bad=copy.deepcopy(dev)
    for row in bad:row['COV-EXTRA']['W']['old_final']=.1
    assert select(bad)['selected'] is None
    records=[]
    for i in range(128):
        ordinary=i>=96;noise='BLURRY' if i<64 or 96<=i<112 else 'STANDARD'
        row=dict(world=i,family='TRIANGLE_TAIL',ordinary=ordinary,noise=noise,shortcut_J={'mode':.1})
        row['COV-R']=item(.2);row['COV-I']=item(.6);records.append(row)
    accepted=confirm(records,'COV-I');assert accepted['verdict'].startswith('ADOPT_')
    cases={}
    for name in ('new_causal_N','new_causal_shuffle','old_causal_N','old_causal_shuffle'):
        changed=copy.deepcopy(records)
        for row in changed:row['COV-I'][name]=0.
        assert confirm(changed,'COV-I')['verdict']=='RETAIN_COV_R';cases[name]=True
    for kind in ('resource','guard','shortcut'):
        changed=copy.deepcopy(records)
        for row in changed:
            if kind=='resource':row['COV-I']['production_cpu_seconds']=2.
            elif kind=='guard':row['COV-I']['W']['old_score']=.1
            else:row['shortcut_J']={'oracle_like_invalid_success':.9}
        assert confirm(changed,'COV-I')['verdict']=='RETAIN_COV_R';cases[kind]=True
    return {'verdict':'PASS_SYNTHETIC_ONLY','negative_adoption_cases':cases,'control_winner_kept':True,'scientific_selection':None}

def suite(folder,cycle):
    folder.mkdir(parents=True,exist_ok=False);out={};baseline=baseline_parity()
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
        clone=model.clone()
        assert not np.shares_memory(clone.core.w,model.core.w) and not np.shares_memory(clone.core.p,model.core.p)
        consume(clone,ref.clone(),b'E+A=8\n',True);assert model.core.digest()==restored.core.digest()
        unchanged=model.core.digest();consume(model,ref,b'E+A=8\n',False);assert model.core.digest()==unchanged
        p=model.core.p.copy();ops=model.core.extra_updates
        if arm in ('COV-REPLAY','COV-EXTRA'):
            model.core.correction(features(65,66),8,model.core.value(features(65,66)),False)
            assert np.array_equal(p,model.core.p)
        model.core.erase();assert model.core.digest()==birth
        out[arm]={'resume_parity':True,'no_write_all_state':True,'independent_equations':True,
                  'extra_information_frozen':True,'payload_bytes':restored.core.payload_bytes(),
                  'external_updates':restored.core.updates,'extra_updates':restored.core.extra_updates,
                  'final_core_seal':restored.core.digest(),'final_ticks':restored.ticks,'final_bytes_seen':restored.bytes_seen}
    reuse=Core('COV-REPLAY');repeat=Core('COV-EXTRA')
    for i in range(80):
        ids=features(65+i%5,65+(i*2)%5)
        reuse.update(ids,i%9,True,reuse.value(ids));repeat.update(ids,i%9,True,repeat.value(ids))
        assert reuse.extra_updates==repeat.extra_updates and reuse.rng.getstate()==repeat.rng.getstate()
    assert len(reuse.cache)==len(repeat.cache)==32
    assert first_crossing([.2,.8,.9],[20,40,80],.75,320)['total_exposure']==360
    assert first_crossing([.2,.5,.7],[20,40,80],.9,320)['right_censored'] is True
    boundaries=[-.1,0.,.5,.5-1e-12,.5+1e-12,7.5,8.,9.]
    from reference import answer
    assert all(emitted(v)==answer(v) for v in boundaries)
    negatives=[lambda:config.admit(999999,'PATH_LOOP'),lambda:config.admit(941001,'TRIANGLE_TAIL'),
               lambda:config.admit(941001,'PATH_LOOP','BLURRY',False),
               lambda:ByteLearner('COV-R').feed(43,1,True),
               lambda:Core('COV-R').update(features(65,66),9,True,0.)]
    rejected=sum(reject(x) for x in negatives)
    partial=ByteLearner('COV-R');partial.feed(65,1,True)
    rejected+=int(reject(lambda:save_checkpoint(partial,folder/'partial.json.gz')))
    return {'verdict':'PASS','cycle':cycle,'arms':out,'baseline':baseline,
            'statistics':synthetic_suite(),'decision_gates':synthetic_decisions(),
            'negative_cases':rejected,'science_worlds':0}

def tamper_suite(receipt):
    mutations={
      'old_permission':lambda r:r['branches']['N_OLD']['old_records'][0].__setitem__('allowed',True),
      'new_permission':lambda r:r['extra_forks']['N_NEW']['new_records'][0].__setitem__('allowed',True),
      'bad_prediction':lambda r:r['branches']['W']['old_records'][0]['events'][3]['prediction'].__setitem__('emitted',57),
      'covariance_in_N_NEW':lambda r:r['extra_forks']['N_NEW']['new_checkpoints']['20']['checkpoint']['state']['p']['array'].__setitem__(0,[99]*31 if r['arm'][:4]!='DIAG' else 99),
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
    if receipt['arm'] in ('COV-REPLAY','COV-EXTRA'):
        # Find an actual extra update; a missing/forbidden first record is not substituted.
        idx=next(i for i,r in enumerate(receipt['branches']['W']['old_records'])
                 if len(r['events'])==6 and r['events'][4]['update']['extra_performed'])
        mutations['replay_source']=lambda r:r['branches']['W']['old_records'][idx]['events'][4]['update'].__setitem__('extra_target',99)
    for name,change in mutations.items():
        candidate=copy.deepcopy(receipt);change(candidate);reject(lambda:audit(candidate))
    return {'verdict':'PASS','rejected':list(mutations),'science_worlds':0}
