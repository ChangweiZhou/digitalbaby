"""Independent row/counter/checkpoint reducers. No operative model imports."""
import math
import data
import config


def audit_addition_rows(source,schedule,history,rows):
    expected=data.ordered(source,schedule,history)
    assert len(rows)<=len(expected)
    for r,e in zip(rows,expected):
        for k in ('id','kind','source_index','prompt','answer','allowed'):assert r[k]==e[k],(k,r,e)
        assert type(r['allowed']) is bool
        before,after=r['versions_before'],r['versions_after']
        assert r['actor_steps']==int(e['allowed']) and math.isfinite(r['loss'])
        assert after['actor_steps']-before['actor_steps']==int(e['allowed'])
        if not e['allowed']:assert before==after and r['forbidden_delta']==0
        else:
            assert all(after['parameters'][k]==v+1 for k,v in before['parameters'].items())
            assert all(after['moments'][k]==v+2 for k,v in before['moments'].items())
    return dict(accepted=True,record_applications=len(rows),complete=len(rows)==len(expected))


def checked_own(rows,expected):
    assert len(rows)==len(expected)
    for row,(prompt,answer) in zip(rows,expected):
        assert row['prompt']==prompt.hex() and row['expected']==answer.hex()
        emitted=bytes.fromhex(row['emitted'])
        assert row['status'] in ('EOS','TRUNCATED') and len(emitted)<=16
        assert row['correct']==(emitted==answer and row['status']=='EOS')
        if row['status']=='TRUNCATED':assert len(emitted)==16


def audit_native_record(row,item,allowed):
    assert row['id']==item['id'] and row['cue']==item['cue'].hex() and row['answer']==item['answer'].hex()
    assert row['allowed'] is allowed and row['forbidden_delta']<=1e-8
    assert row['raw_alpha_l1']>=0 and math.isfinite(row['raw_alpha_l1'])
    if not allowed:assert row['raw_alpha_l1']==0
    assert [r['target'] for r in row['trace']]==[*item['answer'],256]
    for r in row['trace']:
        assert len(r['logits'])==257 and all(math.isfinite(v) for v in r['logits'])
        assert r['predicted']==max(range(257),key=lambda i:r['logits'][i])
    return dict(accepted=True,record_applications=1)


def audit_addition_job(receipt):
    m=receipt['manifest'];technical=m['technical'];source=data.make_addition(m['world'],technical)
    assert m['kind']=='addition' and receipt['technical'] is technical
    assert set(receipt['histories'])==set(config.HISTORIES)
    total=0
    for history,h in receipt['histories'].items():
        r=audit_addition_rows(source,m['schedule'],history,h['rows']);assert r['complete']
        plan=data.checkpoint_plan(source,m['schedule']);length=len(h['rows'])
        expected={(t,'OBSERVATION') for t in plan}
        expected|={(t,'AFTER_REST') for t in ((length,) if technical else (16384,17408))}
        assert {(q['T'],q['phase']) for q in h['curve']}==expected
        assert len(h['curve'])==len(expected)
        for q in h['curve']:
            t=q['T'];prefix=h['rows'][:t];seen={r['prompt'] for r in prefix}
            assert q['old_count']==sum(r['kind']=='OLD' for r in prefix)
            assert q['new_count']==sum(r['kind']=='NEW' for r in prefix)
            if q['phase']=='OBSERVATION':assert q['tags']==plan[t]
            for name,pairs in source['groups'].items():
                checked_own(q['scores'][name],[data.example(p) for p in pairs])
                for row in q['scores'][name]:assert row['actually_exposed']==(row['prompt'] in seen)
            # Recompute all shortcut predictions independently from the actual observed stream.
            from collections import Counter,defaultdict
            counts=Counter(bytes.fromhex(r['answer']) for r in prefix);fallback=min(counts,key=lambda x:(-counts[x],x)) if counts else b''
            table={r['prompt']:bytes.fromhex(r['answer']) for r in prefix};position=[defaultdict(Counter) for _ in range(4)]
            for r in prefix:
                for i,b in enumerate(bytes.fromhex(r['prompt'])):position[i][b][bytes.fromhex(r['answer'])]+=1
            observed={k:[] for k in ('FREQUENCY','RAW_KEY','CANONICAL_KEY','BYTE0','BYTE1','BYTE2','BYTE3')}
            for pair in source['groups']['held']:
                p,y=data.example(pair);z=table.get(p.hex(),fallback)
                values=dict(FREQUENCY=fallback,RAW_KEY=z,CANONICAL_KEY=z)
                for i,b in enumerate(p):
                    c=position[i][b];values['BYTE'+str(i)]=min(c,key=lambda x:(-c[x],x)) if c else fallback
                for k,v in values.items():observed[k].append(dict(emitted=v.hex(),correct=v==y))
            assert observed==q['shortcuts']['scores']
            assert q['shortcuts']['envelope']==max(sum(r['correct'] for r in rows)/len(rows) for rows in observed.values())
        resource=h['resource'];answers=sum(len(bytes.fromhex(r['answer'])) for r in h['rows'])
        assert resource['records']==length and resource['observed_answer_bytes']==answers
        assert resource['observed_EOS_targets']==length and resource['vocabulary_rows']==answers+length
        assert resource['actor_steps']==sum(r['allowed'] for r in h['rows'])
        total+=length
    assert receipt['record_applications']==m['expected_records']==total
    return dict(accepted=True,technical=technical,record_applications=total,correlated_histories=3,held_feedback_records=0)


def audit_native_job(receipt):
    m=receipt['manifest'];technical=m['technical'];fixture=data.native_fixture(m['world'])
    n=4 if technical else 768
    import random
    stream=data.shuffled_cycles(fixture,n,random.Random(data.seed(m['world'],'native-order')))
    assert set(receipt['histories'])=={'W','NO_WRITE'}
    for history,h in receipt['histories'].items():
        assert len(h['rows'])==n
        for row,item in zip(h['rows'],stream):audit_native_record(row,item,history=='W')
        expected={(t,'OBSERVATION') for t in ((0,n) if technical else (0,16,64,256,768))}|{(n,'AFTER_REST')}
        assert {(q['T'],q['phase']) for q in h['curve']}==expected and len(h['curve'])==len(expected)
        for q in h['curve']:
            assert [r['id'] for r in q['rows']]==[r['id'] for r in fixture]
            for r,item in zip(q['rows'],fixture):
                emitted=bytes.fromhex(r['emitted']);cap=2 if technical else 16
                assert r['expected']==item['answer'].hex() and r['stratum']==item['stratum']
                assert r['status'] in ('EOS','TRUNCATED') and len(emitted)<=cap
                assert r['correct']==(emitted==item['answer'] and r['status']=='EOS')
                assert r['first_byte_correct']==(bool(emitted) and emitted[:1]==item['answer'][:1])
                assert r['premature_EOS']==(r['status']=='EOS' and len(emitted)<len(item['answer']))
                if r['status']=='TRUNCATED':assert len(emitted)==cap
                tokens=[]
                for i,t in enumerate(r['trace']):
                    assert t['prefix']==bytes(tokens).hex()
                    assert len(t['logits'])==257 and all(math.isfinite(x) for x in t['logits'])
                    assert t['predicted']==max(range(257),key=lambda j:t['logits'][j])
                    if t['predicted']==256:assert i==len(r['trace'])-1
                    elif len(tokens)<cap:tokens.append(t['predicted'])
                assert bytes(tokens)==emitted and (r['trace'][-1]['predicted']==256)==(r['status']=='EOS')
                assert len(r['trace'])==len(emitted)+1
                assert r['conditional_is_autonomous_score'] is False
                assert [t['target'] for t in r['conditional']]==[*item['answer'],256]
                for t in r['conditional']:
                    assert len(t['logits'])==257 and all(math.isfinite(v) for v in t['logits'])
                    assert t['predicted']==max(range(257),key=lambda j:t['logits'][j])
    assert receipt['record_applications']==m['expected_records']==2*n
    return dict(accepted=True,technical=technical,record_applications=2*n,correlated_histories=2)
