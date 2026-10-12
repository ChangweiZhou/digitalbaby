"""Public fixtures and technical-only full-job qualification; not efficacy data."""
import copy,importlib.util,math,sys
import numpy as np
import config
from cores import Core,ByteLearner,features,emitted
from reference import Reference
from runner import consume,audit
from operations import save_checkpoint,load_checkpoint
from decision import synthetic_suite,select,confirm,first_crossing

def reject(call):
    try:call()
    except (AssertionError,ValueError,KeyError,TypeError):return True
    raise AssertionError('negative fixture accepted')
def restore_reference(model):
    ref=Reference(model.core.arm,getattr(model.core,'world_id',0))
    for k,v in model.core.state().items():
        if k in ref.a:ref.a[k]=copy.deepcopy(v)
        elif k in ('rng','direction_rng'):getattr(ref,k).setstate(v)
        else:setattr(ref,k,copy.deepcopy(v))
    return ref
def baselines():
    spec=importlib.util.spec_from_file_location('adopted_support_parent',config.PRIOR/'cores.py')
    parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
    for arm in config.ARMS[:2]:
        a,b=Core(arm),parent.Core(arm)
        for i in range(240):
            ids=features(65+i%5,70+(i*3)%5);target=(i//5)%9;allowed=i%11!=0
            av,bv=a.value(ids),b.value(ids);assert av==bv
            ao,bo=a.update(ids,target,allowed,av),b.update(ids,target,allowed,bv)
            ao.pop('restoration');assert ao==bo and a.digest()==b.digest()
            assert (a.scalar_ops,a.traffic_bytes)==(b.scalar_ops,b.traffic_bytes)
    return {'bit_identical_parent_baselines':list(config.ARMS[:2]),'transitions_per_arm':240}
def boundaries():
    witnesses=[]
    # A deliberately transparent two-key stream supplies repeated actual observations.
    models={a:Core(a,12345) for a in config.RESTORE_ARMS};refs={a:Reference(a,12345) for a in models}
    schedule={a:[] for a in models};restoration_vectors=0
    for i in range(140):
        ids=features(65,66+i%2);target=2+i%2
        for arm,m in models.items():
            ref=refs[arm];o=m.update(ids,target,True,m.value(ids));r=ref.step(ids,target,True,ref.predict(ids))
            assert o==r and m.digest()==ref.digest()
            event=o['restoration'];schedule[arm].append((event['qualified_added'],event['qualified_count'],event['performed'],event['restoration_count'],event['delta']))
            if event['performed']:
                assert abs(event['trace_added']-.25)<2e-12
                if event['direction'] is not None:assert abs(sum(v*v for v in event['direction'])-1)<1e-12
                restoration_vectors+=1
            ref.invariants()
        states=[m.state() for m in models.values()]
        for key in ('cache','history','rng','buffer','qualified','restorations','direction_rng','updates'):
            assert all(s[key]==states[0][key] for s in states)
    assert len({str(v) for v in schedule.values()})==1 and restoration_vectors>0
    witnesses.append('strict_matching_schedule_trace_and_rng')
    # Direct zero-gradient fixture forces the fixed fallback, not a data-dependent rescue.
    for arm in config.RESTORE_ARMS:
        m=Core(arm,4);m.qualified=7;m.buffer=[{'ids':[5,21,30],'target':0}]*4+[{'ids':[5,22,30],'target':0}]*3
        before=m.p.copy();event=m.restore_direction((5,22,30),0,dict(eligible=True,conflict=False))
        assert event['performed'] and event['fallback'] and abs(np.trace(m.p-before)-.25)<1e-12
        assert np.linalg.eigvalsh(m.p).min()>0
        if arm=='SUPPORT-DIR':
            z=np.zeros(31);z[[5,22,30]]=1/math.sqrt(3);assert np.allclose(m.p-before,.25*np.outer(z,z),rtol=0,atol=1e-15)
        m.restorations=64;m.qualified=15
        before=m.p.copy();event=m.restore_direction((5,22,30),0,dict(eligible=True,conflict=False))
        assert not event['performed'] and np.array_equal(m.p,before)
    witnesses+=['zero_gradient_fallback','lifetime_cap_64','PSD_rank_one_and_isotropic']
    # Current answer has no past support; single-key windows cannot trigger restoration.
    m=Core('SUPPORT-DIR')
    for i in range(40):
        ids=features(65,66);o=m.update(ids,2,True,m.value(ids))
        if i<2:assert not o['gate']['eligible'] and not o['restoration']['qualified_added']
        assert not o['restoration']['performed']
    assert m.qualified==38 and m.restorations==0
    # Orientation cannot borrow support; conflicts never qualify, but stay in original SUPPORT cache.
    ids=features(66,65);o=m.update(ids,2,True,m.value(ids));assert not o['gate']['eligible']
    ids=features(65,66);c=m.qualified;o=m.update(ids,8,True,m.value(ids))
    assert o['gate']['factor']==.25 and m.qualified==c and m.cache[0]['answers'][-1]==8
    seal=m.digest();rng=m.direction_rng.getstate();m.update(ids,2,False,m.value(ids))
    assert m.digest()==seal and m.direction_rng.getstate()==rng
    witnesses+=['current_answer_not_own_support','two_distinct_keys_required','oriented_keys','conflict_kept_not_qualified','all_state_forbidden']
    # Formal geometry rendered only as public, unlabelled coverage fixture; no truth learning run.
    slots=[(a,b) for u,v in config.CONTRACT['confirmation_edges'] for a,b in ((u,v),(v,u))]
    m=Core('SUPPORT-DIR')
    for a,b in slots*2:
        ids=features(70+a,70+b);o=m.update(ids,4,True,m.value(ids));assert not o['restoration']['qualified_added']
    assert m.qualified==m.restorations==0
    witnesses.append('public_formal_first20_boundary')
    return {'verdict':'PASS','checks':witnesses,'matching_140_record_trace_fixture':True,'restoration_events_checked':restoration_vectors}
def synthetic_decisions():
    def item(j=.2):
        return {'W':{'J':j,'old_score':.7,'old_final':.95,'new_final':j},'new_causal_N':.5,
          'new_causal_shuffle':.5,'old_causal_N':.5,'old_causal_shuffle':.5,
          'production_cpu_seconds':1.,'payload_bytes':15000,'peak_rss_bytes':50*1024**2}
    rows=[]
    for f in ('PATH_LOOP','STAR_LOOP'):
        for n in ('STANDARD','BLURRY'):
            for i in range(3):rows.append(dict(world=len(rows),family=f,noise=n,**{a:item() for a in config.ARMS}))
    assert select(rows)['selected'] is None
    for r in rows:r['SUPPORT-RAND']=item(.8);r['SUPPORT-ISO']=item(.8)
    assert select(rows)['selected'] is None
    for r in rows:r['SUPPORT-DIR']=item(.6)
    assert select(rows)['selected']=='SUPPORT-DIR'
    bad=copy.deepcopy(rows)
    for r in bad:r['COV-SUPPORT']=item(.6)
    assert select(bad)['selected'] is None
    records=[]
    for i in range(128):
        ordinary=i>=96;noise='BLURRY' if i<64 or 96<=i<112 else 'STANDARD'
        r=dict(world=i,family='TRIANGLE_TAIL',ordinary=ordinary,noise=noise,shortcut_J={'mode':.1},**{a:item() for a in config.ARMS})
        r['SUPPORT-DIR']=item(.6);records.append(r)
    assert confirm(records,'SUPPORT-DIR')['verdict'].startswith('ADOPT_')
    for arm in ('SUPPORT-RAND','SUPPORT-ISO'):reject(lambda:confirm(records,arm))
    tested=[]
    for baseline in ('COV-R','COV-SUPPORT'):
        bad=copy.deepcopy(records)
        for r in bad:r[baseline]=item(.6)
        assert confirm(bad,'SUPPORT-DIR')['verdict']=='RETAIN_USER_CORE';tested.append(baseline+'_must_be_beaten')
    for field in ('new_causal_N','new_causal_shuffle','old_causal_N','old_causal_shuffle'):
        bad=copy.deepcopy(records)
        for r in bad:r['SUPPORT-DIR'][field]=0.
        assert confirm(bad,'SUPPORT-DIR')['verdict']=='RETAIN_USER_CORE';tested.append(field)
    for mode in ('cost','retention','shortcut'):
        bad=copy.deepcopy(records)
        for r in bad:
            if mode=='cost':r['SUPPORT-DIR']['production_cpu_seconds']=2.
            if mode=='retention':r['SUPPORT-DIR']['W']['old_final']=.1
            if mode=='shortcut':r['shortcut_J']={'mode':.9}
        assert confirm(bad,'SUPPORT-DIR')['verdict']=='RETAIN_USER_CORE';tested.append(mode)
    return {'verdict':'PASS_SYNTHETIC_ONLY','checks':tested,'only_DIR_selectable':True,'science_worlds':0}
def suite(folder,cycle):
    folder.mkdir(parents=True,exist_ok=False);out={}
    parent=baselines();boundary=boundaries()
    for arm in config.ARMS:
        model,ref=ByteLearner(arm,9876),Reference(arm,9876);birth=model.core.digest()
        for raw in (b'A+B=8\n',b'A+B=\n'):consume(model,ref,raw,False)
        assert model.core.digest()==birth
        raws=[bytes((65+i%5,43,65+(i*2+1)%5,61,48+i%5,10)) for i in range(64)]
        for i,raw in enumerate(raws):
            consume(model,ref,raw,True)
            if i==23:save_checkpoint(model,folder/(arm+'.json.gz'))
        ref.invariants()
        restored=load_checkpoint(folder/(arm+'.json.gz'));shadow=restore_reference(restored)
        for raw in raws[24:]:consume(restored,shadow,raw,True)
        assert restored.core.digest()==model.core.digest()
        clone=model.clone();assert not np.shares_memory(clone.core.w,model.core.w) and not np.shares_memory(clone.core.p,model.core.p)
        consume(clone,ref.clone(),b'E+A=8\n',True);assert model.core.digest()==restored.core.digest()
        seal=model.core.digest();ops=(model.core.scalar_ops,model.core.traffic_bytes)
        consume(model,ref,b'A+B=\n',True);consume(model,ref,b'E+A=8\n',False)
        assert model.core.digest()==seal and (model.core.scalar_ops,model.core.traffic_bytes)==ops
        a=model.clone();b=model.clone();b.core.cache=[{'key':[65,66],'answers':[8]*8}]
        def query(x):return [x.feed(v,x.ticks+1,False) for v in b'A+B=\n'][3]['prediction']
        assert query(a)==query(b)
        out[arm]={'final_core_seal':restored.core.digest(),'final_ticks':restored.ticks,
          'final_bytes_seen':restored.bytes_seen,'payload_bytes':restored.core.payload_bytes(),'resume_parity':True,'all_state_no_write':True}
        model.core.erase();assert model.core.digest()==birth
        restored.core.erase();assert restored.core.digest()==birth
    if cycle>=2:
        for arm in config.RESTORE_ARMS:
            original=ByteLearner(arm,9876);shadow=Reference(arm,9876)
            for i in range(48):consume(original,shadow,bytes((65,43,66+i%2,61,50+i%2,10)),True)
            assert original.core.restorations>0
            path=folder/(arm+'_active.json.gz');save_checkpoint(original,path)
            cold=load_checkpoint(path);assert cold.core.digest()==original.core.digest()
            cold.core.erase();assert cold.core.digest()==ByteLearner(arm,9876).core.digest()
            before=original.core.digest();clone=original.clone()
            clone.core.buffer[0]['target']=8;clone.core.direction_rng.getrandbits(1)
            assert original.core.digest()==before
            assert Core(arm,9876).direction_rng.getstate()!=Core(arm,9877).direction_rng.getstate()
            reject(lambda:Core(arm).restore({'w':original.core.w}))
        partial=ByteLearner('COV-R');partial.feed(65,1,True)
        reject(lambda:save_checkpoint(partial,folder/'partial.json.gz'))
    if cycle>=3:
        for arm in config.RESTORE_ARMS:
            m=ByteLearner(arm,5432)
            m.core.cache=[{'key':[65,66],'answers':[2,2]}]
            m.core.qualified=7
            m.core.buffer=[{'ids':list(features(65,66)),'target':2}]*4+[{'ids':list(features(65,67)),'target':2}]*3
            ref=restore_reference(m);before=m.core.digest()
            consume(m,ref,b'A+B=\n',True);consume(m,ref,b'A+B=2\n',False)
            assert m.core.digest()==before and m.core.restorations==0
            consume(m,ref,b'A+B=2\n',True)
            assert m.core.restorations==1 and m.core.qualified==8
    c=config.CONTRACT;assert c['max_jobs']==700 and c['max_external_records']==700*2560
    import worlds
    worlds.geometry.check_mask([worlds.geometry.orbit(*edge) for edge in c['confirmation_edges']])
    assert first_crossing([.2,.8,.9],[20,40,80],.75,320)['total_exposure']==360
    negatives=[lambda:config.admit(999999,'PATH_LOOP'),lambda:config.admit(1203401,'TRIANGLE_TAIL'),
      lambda:config.admit(1203401,'PATH_LOOP','BLURRY',False),lambda:ByteLearner('COV-R').feed(43,1,True),
      lambda:Core('COV-R').update(features(65,66),9,True,0.)]
    rejected=sum(reject(f) for f in negatives)
    return {'verdict':'PASS','cycle':cycle,'arms':out,'baselines':parent,'restoration':boundary,
      'statistics':synthetic_suite(),'decision_gates':synthetic_decisions(),'negative_cases':rejected,'science_worlds':0}

def tamper_suite(receipt):
    mutations={
      'OLD_permission':lambda r:r['branches']['N_OLD']['old_records'][0].__setitem__('allowed',True),
      'NEW_permission':lambda r:r['extra_forks']['N_NEW']['new_records'][0].__setitem__('allowed',True),
      'NEW_P':lambda r:r['extra_forks']['N_NEW']['new_checkpoints']['20']['checkpoint']['state']['p']['array'].__setitem__(0,[99]*31),
      'output':lambda r:r['branches']['W']['old_records'][0]['events'][3]['prediction'].__setitem__('emitted',57),
      'omit_branch':lambda r:r['extra_forks'].pop('SHUFFLED_NEW'),
      'source':lambda r:r['source'].__setitem__('unlocked','bad'),
      'seal':lambda r:r['branches']['W']['old_records'][0].__setitem__('core_seal','bad'),
      'counts':lambda r:r.__setitem__('training_records',2400),
      'payload':lambda r:r['extra_forks']['N_NEW']['final_checkpoint'].__setitem__('payload_bytes',0),
      'resource':lambda r:r['extra_forks']['N_NEW']['final_checkpoint'].__setitem__('scalar_ops',0)}
    if receipt['arm'] in config.RESTORE_ARMS:
        for field in ('buffer','qualified','restorations','direction_rng','world_id','fallbacks'):
            mutations['NEW_'+field]=lambda r,f=field:r['extra_forks']['N_NEW']['new_checkpoints']['20']['checkpoint']['state'].__setitem__(f,[] if f in ('buffer','direction_rng') else -123)
        records=receipt['branches']['W']['old_records']
        index=next(i for i,r in enumerate(records) if len(r['events'])==6 and r['events'][4]['update']['restoration']['performed'])
        for field,value in [('performed',False),('delta',1.),('qualified_count',999),('trace_added',99.),('direction',[0.]*31),('fallback','invalid')]:
            mutations['restoration_'+field]=lambda r,f=field,v=value:r['branches']['W']['old_records'][index]['events'][4]['update']['restoration'].__setitem__(f,v)
    for name,change in mutations.items():
        x=copy.deepcopy(receipt);change(x);reject(lambda:audit(x))
    return {'verdict':'PASS','rejected':list(mutations),'science_worlds':0}
