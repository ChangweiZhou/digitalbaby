"""Independent receipt and branch auditor. No core/interface/runner imports."""
import math
from worlds import make_world,CHECKPOINTS
from vendor.task_source import prompt,answer,recover_family
from analysis import rate,rates,shortcut

def check_record(rec,allowed,name,all_bytes=False,previous=None):
    assert rec['allowed'] is allowed and rec['all_bytes'] is all_bytes
    p=bytes.fromhex(rec['prompt_hex']);a=bytes.fromhex(rec['answer_hex'])
    assert rec['targets']==list(p)+list(a)+[256]
    expected=[allowed and all_bytes]*len(p)+[allowed]*(len(a)+1)
    assert rec['permissions']==expected
    assert len(rec['choices'])==len(expected) and all(type(x) is int and 0<=x<=256 for x in rec['choices'])
    before=rec['counters_before'];after=rec['counters_after']
    if previous is not None:assert list(before)==list(previous)
    delta=[y-x for x,y in zip(before,after)]
    joint=name not in ('RAW-RLS','LEGACY-B')
    head=int(allowed) if joint else int(allowed)*(len(a)+1)
    learned='FROZEN' not in name and name!='LEGACY-B'
    encoder=int(allowed and learned)
    optimizer=encoder if name=='RAW-RLS' else int(allowed) if joint else 0
    assert delta==[head,encoder,optimizer]
    assert [rec['readout_updates'],rec['encoder_steps'],rec['optimizer_steps']]==delta
    if joint and allowed:
        update=rec['update'];assert update['targets']==sum(expected)
        assert 0<update['clip_factor']<=1 and update['unclipped_gradient_norm']>=0
        assert math.isfinite(update['mean_loss'])
    else:assert rec['update'] is None
    assert rec['cpu_seconds']>=0
    return after

def check_predictions(rows,expected):
    assert [(r['prompt_hex'],r['answer_hex'],r.get('operation')) for r in rows]==[(r['prompt_hex'],r['answer_hex'],r.get('operation')) for r in expected]
    for r in rows:
        payload=bytes.fromhex(r['payload_hex'])
        assert r['status'] in ('EOS','TRUNCATED') and len(payload)<=16
        if r['status']=='TRUNCATED':assert len(payload)==16
        assert 0<=r['first_choice']<=256

def audit(receipt,expected_source):
    plan=make_world(receipt['world'],True)
    assert receipt['schema']=='RAW_RULE_V2_TECHNICAL' and receipt['technical'] is True
    assert receipt['source_identity']==expected_source and receipt['stream_digest']==plan['stream_digest']
    name=receipt['config'];assert name in ('RAW-RLS','G-SQ','G-CE','L-SQ','L-CE','G-FROZEN-CE','L-FROZEN-CE','LEGACY-B')
    assert {recover_family(r) for r in plan['training_pool']}.isdisjoint({recover_family(r) for r in plan['held_pool']})
    held=[{'prompt_hex':prompt(r).encode().hex(),'answer_hex':answer(r).encode().hex(),'operation':r[0]} for r in plan['held_pool']]
    total=0;summary={};assert set(receipt['histories'])=={'W','N_OLD','SHUFFLE_OLD'}
    for history,h in receipt['histories'].items():
        assert len(h['records'])==512 and len(h['checkpoints'])==6
        if history=='N_OLD':assert h['initial']==h['final']
        prev=None
        for i,(rec,expected) in enumerate(zip(h['records'],plan['records'])):
            target=expected['answer'] if history!='SHUFFLE_OLD' else plan['shuffled_answers'][str(i)]
            assert rec['prompt_hex']==expected['prompt'].encode().hex() and rec['answer_hex']==target.encode().hex()
            prev=check_record(rec,history!='N_OLD',name,previous=prev)
        check_predictions(h['initial_held'],held);summary[history]=[]
        for stop,cp in zip(CHECKPOINTS,h['checkpoints']):
            assert cp['budget']==stop and cp['root_records']==4*stop
            assert cp['parent_before']==cp['parent_after']==cp['fork_initial']
            if history=='N_OLD':assert cp['parent_before']==h['initial']
            prefix=h['records'][:4*stop]
            assert len(cp['cohort_records'])==32
            prev=None
            for rec,q in zip(cp['cohort_records'],plan['cohort_records']):
                assert (rec['prompt_hex'],rec['answer_hex'])==(q['prompt_hex'],q['answer_hex'])
                prev=check_record(rec,True,name,previous=prev)
            for key in ('held_pre','held_post','held_retained'):check_predictions(cp[key],held)
            assert [(r['payload_hex'],r['status']) for r in cp['held_post']]==[(r['payload_hex'],r['status']) for r in cp['held_retained']]
            taught={r['prompt']:r['answer'] for r in plan['records'] if r['checkpoint']<=stop}
            tq=[{'prompt_hex':p.encode().hex(),'answer_hex':a.encode().hex(),'operation':next(r['operation'] for r in plan['records'] if r['prompt']==p)} for p,a in taught.items()]
            for key in ('taught_pre','taught_retained'):check_predictions(cp[key],tq)
            for key in ('cohort_pre','cohort_post'):check_predictions(cp[key],plan['cohort_pairs'])
            assert abs(cp['scenario_cpu']-cp['root_cpu']-cp['fork_cpu'])<1e-8
            B={op:shortcut([r for r,e in zip(prefix,plan['records']) if e['operation']==op],
                 [q for q in held if q['operation']==op]) for op in ('+','-','*','/')}
            summary[history].append({'n':stop,'cpu':cp['scenario_cpu'],'taught':rate(cp['taught_pre']),
                'taught_retained':rate(cp['taught_retained']),'held_pre':rates(cp['held_pre']),
                'held_retained':rates(cp['held_retained']),'B':B,
                'cohort_pre':rate(cp['cohort_pre']),'cohort_post':rate(cp['cohort_post'])})
        total+=len(h['records'])+sum(len(x['cohort_records']) for x in h['checkpoints'])
    assert total==receipt['record_applications']==2112
    return {'accepted':True,'technical':True,'world':receipt['world'],'config':name,'record_applications':total,
            'summary':summary,'source_identity':expected_source}
