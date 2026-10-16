"""Independent row/counter/checkpoint reducers. No operative model imports."""
import math
import data
import config


def audit_addition_rows(source,schedule,history,rows):
    expected=data.ordered(source,schedule,history)
    assert len(rows)<=len(expected)
    for r,e in zip(rows,expected):
        for k in ('id','kind','source_index','prompt','answer','allowed'):assert r[k]==e[k],(k,r,e)
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
