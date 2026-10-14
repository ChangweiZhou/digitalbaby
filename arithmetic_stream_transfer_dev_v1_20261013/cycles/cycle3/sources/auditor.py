"""Independent B write/exposure/count reducer; imports no core/interface/runner."""
import math
from worlds import make_world,PASSES
from analysis import rate,shortcut

def record(rec,q,allowed,all_bytes,previous,decay):
    assert rec['prompt_hex']==q['prompt_hex'] and rec['answer_hex']==q['answer_hex']
    assert rec['allowed'] is allowed and rec['all_bytes'] is all_bytes
    p=bytes.fromhex(q['prompt_hex']);a=bytes.fromhex(q['answer_hex'])
    expected=[allowed and all_bytes]*len(p)+[allowed]*(len(a)+1)
    assert rec['targets']==list(p)+list(a)+[256] and rec['permissions']==expected
    assert list(rec['counters_before'])==list(previous)
    delta=[a-b for a,b in zip(rec['counters_after'],rec['counters_before'])]
    assert delta==[int(allowed)]*3
    assert [rec['readout_updates'],rec['encoder_steps'],rec['optimizer_steps']]==delta
    assert len(rec['choices'])==len(expected) and all(type(c) is int and 0<=c<=256 for c in rec['choices'])
    if allowed:
        u=rec['update'];assert u['targets']==sum(expected) and abs(u['decay_factor']-(1-.001*decay))<1e-14
        assert math.isfinite(u['mean_loss']) and 0<u['clip_factor']<=1
    else:assert rec['update'] is None
    assert rec['cpu_seconds']>=0
    return rec['counters_after']
def predictions(rows,expected):
    assert [(r['prompt_hex'],r['answer_hex']) for r in rows]==[(r['prompt_hex'],r['answer_hex']) for r in expected]
    for r in rows:
        assert r['status'] in ('EOS','TRUNCATED') and len(bytes.fromhex(r['payload_hex']))<=16
        if r['status']=='TRUNCATED':assert len(bytes.fromhex(r['payload_hex']))==16

def audit(receipt,source):
    plan=make_world(receipt['label']);name=receipt['config'];all_bytes=name.startswith('ALL');decay=.01 if name.endswith('-D') else 0.
    assert name in ('ALL-0','ALL-D','ANSWER-0','ANSWER-D')
    assert receipt['technical'] is True and receipt['schema']=='STREAM_TRANSFER_TECHNICAL_V1'
    assert receipt['source_identity']==source and receipt['stream_digest']==plan['stream_digest']
    assert len(receipt['counting']['records'])==128
    prev=[0,0,0]
    for rec,q in zip(receipt['counting']['records'],plan['counting']):prev=record(rec,q,True,all_bytes,prev,decay)
    count_final=prev;count_digest=receipt['counting']['final']
    for op,task in plan['tasks'].items():
        assert len(task['records'])==17920 and len(task['train_orbits'])==35 and len(task['held_orbits'])==20
        train={tuple(x) for x in task['train_orbits']};held={tuple(x) for x in task['held_orbits']};assert train.isdisjoint(held)
        assert {q['prompt_hex'] for q in task['taught']}.isdisjoint({q['prompt_hex'] for q in task['held']})
        assert {q['prompt_hex'] for q in plan['counting']}.isdisjoint({q['prompt_hex'] for q in task['held']})
    assert set(receipt['addition'])=={'W','N_ADD','SH_ADD'}
    assert set(receipt['multiplication'])=={'W','N_ADD','SH_ADD','N_MUL','SH_MUL'}
    total=128;summary={'addition':{},'multiplication':{}}
    for phase,op in [('addition','+'),('multiplication','*')]:
        task=plan['tasks'][op]
        for history,h in receipt[phase].items():
            assert len(h['records'])==17920 and len(h['checkpoints'])==5
            if phase=='addition':expected_initial=count_digest;initial_counters=count_final
            else:
                parent=history if history in ('N_ADD','SH_ADD') else 'W'
                expected_initial=receipt['addition'][parent]['final'];initial_counters=receipt['addition'][parent]['final_counters']
            assert h['initial']==expected_initial
            allowed=history!=('N_ADD' if phase=='addition' else 'N_MUL')
            shuffle=history==('SH_ADD' if phase=='addition' else 'SH_MUL')
            prev=initial_counters
            for i,(rec,q0) in enumerate(zip(h['records'],task['records'])):
                q=dict(q0)
                if shuffle:q['answer_hex']=task['shuffled_answers'][i]
                prev=record(rec,q,allowed,all_bytes,prev,decay)
            assert list(h['final_counters'])==list(prev)
            if not allowed:assert h['initial']==h['final']
            summary[phase][history]=[]
            for pass_n,cp in zip(PASSES,h['checkpoints']):
                assert cp['pass']==pass_n and cp['root_records']==70*pass_n
                assert cp['parent_before']==cp['parent_after']==cp['fork_initial']
                fork_previous=h['records'][cp['root_records']-1]['counters_after']
                assert len(cp['cohort_records'])==32
                for rec,q in zip(cp['cohort_records'],plan['cohort_records']):fork_previous=record(rec,q,True,all_bytes,fork_previous,decay)
                for key in ('held_pre','held_post','held_retained'):predictions(cp[key],task['held'])
                assert [(r['payload_hex'],r['status']) for r in cp['held_post']]==[(r['payload_hex'],r['status']) for r in cp['held_retained']]
                predictions(cp['taught_pre'],task['taught'])
                for key in ('cohort_pre','cohort_post'):predictions(cp[key],plan['cohort_pairs'])
                if phase=='multiplication':predictions(cp['addition_retained'],plan['tasks']['+']['held'])
                assert abs(cp['scenario_cpu']-cp['prefix_cpu']-cp['root_cpu']-cp['fork_cpu'])<1e-8
                B=shortcut(h['records'][:cp['root_records']],task['held'])
                summary[phase][history].append({'pass':pass_n,'cpu':cp['scenario_cpu'],'taught':rate(cp['taught_pre']),
                  'held_pre':rate(cp['held_pre']),'held_retained':rate(cp['held_retained']),'B':float(B),
                  'cohort_pre':rate(cp['cohort_pre']),'cohort_post':rate(cp['cohort_post']),
                  'addition_retained':rate(cp.get('addition_retained',cp['held_retained']))})
            total+=len(h['records'])+sum(len(c['cohort_records']) for c in h['checkpoints'])
    assert total==receipt['record_applications']==144768
    if 'resource' in receipt:
        r=receipt['resource'];assert all(math.isfinite(v) and v>=-1e-8 for v in r.values())
        production=receipt['counting']['cpu']+sum(h['cpu']+sum(c['fork_cpu'] for c in h['checkpoints']) for phase in ('addition','multiplication') for h in receipt[phase].values())
        assert abs(production-r['production_cpu'])<1e-8
        assert abs(r['total_job_cpu']-production-sum(r[k] for k in ('probe_cpu','snapshot_cpu','audit_cpu','other_cpu')))<1e-8
    return {'accepted':True,'technical':True,'label':receipt['label'],'config':name,'record_applications':total,
            'summary':summary,'source_identity':source}
