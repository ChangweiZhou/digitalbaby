"""Hand/local and integration integrity fixtures; no scientific promotion."""
import copy,importlib.util,json,math,sys
from pathlib import Path
import numpy as np
import settings,data,dependencies
from learners import Learner
from audit import audit,numeric_update
from runner import consume,save
def require(ok,msg):
    if not ok:raise AssertionError(msg)
def local_suite():
    checks=[]
    for arm in settings.ARMS:
        a=Learner(arm);b=Learner(arm)
        require({k:v for k,v in a.encoder.birth.items() if k!='fly_id'}=={k:v for k,v in b.encoder.birth.items() if k!='fly_id'},'birth canonical values match')
        require(all(not np.shares_memory(a.arrays()[k],b.arrays()[k]) for k in a.arrays()),'independent mutable birth')
        for pair in data.ALL:require(np.array_equal(a.encoder.code(pair),b.encoder.code(pair)),'fixed code identity')
        a.plastic=b.plastic=False
        consume(a,b'1+2=0\n',0);consume(b,b'1+2=8\n',0)
        require(a.state_digest()==b.state_digest(),'forbidden target inert')
        require(not any(x.any() for k,x in a.arrays().items() if k!='last'),'forbidden auxiliaries inert')
        m=Learner(arm);consume(m,b'1+2=\n',0);require(not m.visits.any() and not m.fast.any() and not m.slow.any(),'missing no update')
        c=m.clone();require(all(not np.shares_memory(v,c.arrays()[k]) for k,v in m.arrays().items()),'clone independent auxiliaries')
        prior=m.state_digest();consume(c,b'1+2=3\n',round(c.t/settings.DT));require(m.state_digest()==prior,'clone parent immutable')
        arrays={k:v.copy() for k,v in c.arrays().items()};c.rest(settings.DAY)
        require(all(np.array_equal(v,c.arrays()[k]) for k,v in arrays.items()),'rest no content decay')
        try:c.feed(48,c.t,truth=3)
        except TypeError:pass
        else:raise AssertionError('oracle metadata accepted')
        checks += [arm+'_birth_clone',''+arm+'_disabled_auxiliary_inert',arm+'_missing_feedback',arm+'_rest_inert',arm+'_metadata_rejected']
    # Local cases span gate categories, saturated commitment, sign disagreement and zero residual.
    cases=[('U042-WG1',0.,0.,0,.2,0,8,0.),('U042-WG1',.1,.2,10,.2,0,8,0.),('U042-WG1',.1,.2,10,.2,0,1,0.),('U042-WG1',.1,.2,10,.2,0,1,1.),('U021-SF1',.4,1.9,4,0.,1,8,0.),('U021-SF1',-.4,-1.9,4,0.,-1,0,8.),('U021-SF1',1.8,0.,4,0.,-1,0,8.),('U021-SF1',.4,.5,4,0.,1,1,1.)]
    for arm,f,s,v,a,c,y,value in cases:
        m=Learner(arm);m.fast[0]=f;m.slow[0]=s;m.visits[0]=v
        if m.summary is not None:m.summary[0]=a
        if m.run is not None:m.run[0]=c
        m.pending={'ids':np.array([0]),'z':np.array([1.]),'value':value}
        before={k:x.copy() for k,x in m.arrays().items()};expected=numeric_update(before,arm,np.array([0]),48+y,value,1.,True)
        got=m._teach(48+y,1.)
        require(all(np.allclose(before[k],x,rtol=0,atol=1e-14) for k,x in m.arrays().items()),'local equation mismatch')
        require(all(np.allclose(got[k],expected[k],rtol=0,atol=1e-14) for k in got),'local ledger mismatch')
    checks.append('eight_explicit_local_gate_and_commitment_cases')
    # Hand arithmetic, not just operative/reducer agreement.
    m=Learner('U021-SF1');m.fast[0]=.4;m.slow[0]=1.9;m.visits[0]=4;m.run[0]=1;m.pending={'ids':np.array([0]),'z':np.array([1.]),'value':0.}
    m._teach(56,1.);want=.2+1/math.sqrt(5)+1.9
    require(abs(m.fast[0]+m.slow[0]-want)<1e-14 and m.slow[0]==2. and m.run[0]==0,'hand saturated conserved transfer')
    require(abs(want-2.3)>.1,'not time-only transfer')
    checks.append('hand_calculated_U021_distinct_from_time_relocation')
    # Untouched frozen STABLE source parity on new synthetic bytes, not old science replay.
    p=dependencies.ROOT.parent/'addition_value_retention_confirm_v1_20261011'
    spec=importlib.util.spec_from_file_location('frozen_temporal',(p/'temporal.py'));tm=importlib.util.module_from_spec(spec);spec.loader.exec_module(tm);sys.modules['temporal']=tm
    spec=importlib.util.spec_from_file_location('frozen_stable',(p/'learners.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    a=old.Residual('STABLE');b=Learner('STABLE');tick=0
    for raw in (b'1+2=3\n',b'4+4=8\n',b'2+0=0\n',b'1+0=\n'):
        for byte in raw:
            tick+=1;t=tick*settings.DT;ra=a.feed(int(byte),t);rb=b.feed(int(byte),t)
            if ra['prediction']:
                require(np.allclose(ra['prediction']['probabilities'],rb['prediction']['probabilities'],atol=1e-14,rtol=0),'baseline output parity')
            for key in ('fast','slow','last','visits'):require(np.allclose(getattr(a,key).ravel(),getattr(b,key),atol=1e-14,rtol=0),'baseline content parity')
    checks.append('frozen_STABLE_synthetic_numeric_parity_no_science_replay')
    # Master prefixes and legal namespace.
    master=data.generate(834001)
    for phase,blocks in master['blocks'].items():
        require(all(len(block)==10*len(data.OLD if phase=='OLD' else data.NEW) for block in blocks),'block cardinality')
        require(blocks[:1]==blocks[:2][:1] and blocks[:2]==blocks[:4][:2],'nested blocks')
    for bad in (True,834101,834201,833001,-1):
        try:settings.require_world(bad)
        except ValueError:pass
        else:raise AssertionError('unauthorized ID accepted')
    checks+=['master_block_nesting','DEV_confirmation_admission_rejected']
    from decision import lower_bound
    require(lower_bound([.05]*64)['lower']<0,'zero variance false certainty')
    checks.append('conservative_zero_variance')
    return dict(verdict='PASS',checks=checks,science_worlds=0,world_trajectories=0)
def tamper_suite(receipt):
    # Every mutation preserves the rest of the artifact. The reducer must diagnose it.
    rejected=[]
    cp='OLD_END';branch='W';phase='OLD'
    def answer(r):
        row=next(x for rr in r['branches'][branch]['training'][phase] for x in rr['trace'] if x['observed_target'] is not None);row['byte']=48+(row['byte']-47)%9
    def forbidden(r):r['branches']['N_OLD']['states'][cp]['arrays']['visits'][0]=1
    def auxiliary(r):
        key='summary' if r['arm']=='U042-WG1' else 'run' if r['arm']=='U021-SF1' else 'slow';r['branches']['N_OLD']['states'][cp]['arrays'][key][0]=1
    def gate(r):
        row=next(x for rr in r['branches'][branch]['training'][phase] for x in rr['trace'] if x['update'] is not None);row['update']['gate_counts'][0]+=1
    def commitment(r):
        row=next(x for rr in r['branches'][branch]['training'][phase] for x in rr['trace'] if x['update'] is not None);row['update']['transfer_l1']+=.1
    def padding(r):
        row=next(rr for rr in r['branches'][branch]['training'][phase] if rr['padding_ticks']);row['padding_ticks']=0
    def double_day(r):r['branches'][branch]['states']['DAY1']['ticks']+=settings.DAY_TICKS
    def held(r):r['master']['blocks']['OLD'][0][0]['pair']=[1,1]
    def dropped(r):r['branches'][branch]['training'][phase].pop()
    def readout(r):r['branches'][branch]['probes']['FINAL']['rows'][0]['numeric_value']+=1
    def query(r):r['branches'][branch]['probes']['FINAL']['rows'][0]['trace'][0]['update_allowed']=True
    def timing(r):
        row=next(rr for rr in r['branches'][branch]['training'][phase] if len(rr['trace'])==6);row['trace'][3],row['trace'][4]=row['trace'][4],row['trace'][3]
    def source(r):r['source']['files']['representation.py']='invalid'
    def counts(r):r['training_histories']=4
    cases={'altered_delivered_answer':answer,'forbidden_visits':forbidden,'forbidden_auxiliary':auxiliary,'wrong_gate_ledger':gate,'forged_commitment':commitment,'missing_padding':padding,'double_day_clock':double_day,'held_exposure':held,'dropped_training_row':dropped,'false_readout':readout,'query_mutation':query,'feedback_before_prediction':timing,'changed_projection_source':source,'false_counts':counts}
    for name,mutate in cases.items():
        r=copy.deepcopy(receipt);mutate(r)
        try:audit(r)
        except AssertionError as exc:rejected.append(dict(name=name,error=str(exc)))
        else:raise AssertionError('tamper accepted '+name)
    return dict(verdict='PASS',rejected=rejected,science_worlds=0)
