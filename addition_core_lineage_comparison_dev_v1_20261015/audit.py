"""Independent evaluator audit. Does not import the learner or trust write flags alone."""
import collections
import math
import data


def audit_job(receipt):
    assert receipt['science'] is False
    world,version=receipt['world'],receipt['version']
    source=data.prepare(world,receipt['old_n'],receipt['new_n'])
    assert set(receipt['histories'])=={'W','N_OLD','SH_OLD'}
    count=0
    for history,h in receipt['histories'].items():
        expected=[('OLD',*x) for x in (source['shuffled'] if history=='SH_OLD' else source['old'])]
        expected += [('NEW',*x) for x in source['new']]
        assert len(h['rows'])==len(expected)
        for row,(stage,prompt,answer) in zip(h['rows'],expected):
            assert (row['stage'],row['prompt'],row['answer'])==(stage,prompt.hex(),answer.hex())
            allowed = history!='N_OLD' or stage!='OLD'
            assert row['allowed'] is allowed
            assert row['forbidden_delta']<=1e-9
            assert math.isfinite(row['raw_alpha_l1']) and row['raw_alpha_l1']>=0
            if not allowed:
                assert row['actor_steps']==0 and row['raw_alpha_l1']==0
            if version.endswith('-CE'):
                assert row['actor_steps']==int(allowed)
                assert math.isfinite(row['loss'])
        assert h['resource']['records']==len(expected)
        total_answers=sum(len(answer) for _,_,answer in expected)
        resource=h['resource']
        if version.endswith('-CE'):
            assert resource['observed_answer_bytes']==total_answers
            assert resource['observed_EOS_targets']==len(expected)
            assert resource['actor_steps']==sum(row['allowed'] for row in h['rows'])
        else:
            assert resource['answer_bytes']==total_answers and resource['EOS_targets']==len(expected)
            assert resource['input_bytes']==sum(len(prompt) for _,prompt,_ in expected)
            assert resource['controls']==2*len(expected)
        expected_births=32 if version=='RC151-C16' else 16 if '151' in version else 0
        assert len(h['births'])==expected_births
        assert all(b['canonical_B_sha256']=='32a3726cb25a8255c1aef5696ff40aa303138c5d061a598004d2087d4da5f964' for b in h['births'])
        if expected_births:
            assert len({b['fly_id'] for b in h['births']})==expected_births
        for metric in h['metrics']:
            for name,pairs in source['groups'].items():
                rows=metric['scores'][name]
                assert len(rows)==len(pairs)
                for row,pair in zip(rows,pairs):
                    prompt,answer=data.render(pair)
                    assert (row['prompt'],row['expected'])==(prompt.hex(),answer.hex())
                    assert row['status'] in ('EOS','TRUNCATED')
                    assert len(bytes.fromhex(row['emitted']))<=16
                    assert row['correct']==(row['emitted']==answer.hex() and row['status']=='EOS')
        assert [(m['stage'],m['n']) for m in h['metrics']]==(
            [('OLD',n) for n in source['checkpoints']]+[('OLD_REST',receipt['old_n']),('NEW_REST',receipt['new_n'])])
        count+=len(expected)
    assert count==receipt['records']==3*(receipt['old_n']+receipt['new_n'])
    return dict(accepted=True,science=False,world=world,version=version,records=count,histories=3,
                scope='Stream/branch/counter/output audit plus runtime forbidden-state checks; not proof against arbitrary malicious instrumentation.')
