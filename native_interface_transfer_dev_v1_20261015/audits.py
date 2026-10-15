"""Independent data/output/counter reducers; never imports operative models."""
import math
import config
import tasks


def checked_scores(scores,universe,predicted,target=None,margin=None):
    assert len(scores)==len(universe) and all(math.isfinite(x) for x in scores)
    assert predicted==universe[max(range(len(scores)),key=lambda i:scores[i])]
    if margin is not None:
        i=universe.index(target)
        expected=scores[i]-max(v for j,v in enumerate(scores) if j!=i)
        assert abs(margin-expected)<=1e-10


def audit_native(result):
    stage=result['stage'];assert stage in config.TRACK_A_STAGES and result['science'] is False
    fixture=tasks.native_fixture();items={r['id']:r for r in fixture}
    order=tasks.ordered_stream(fixture,config.TRACK_A_DOSES[-1],config.TRACK_A_SEED)
    assert set(result['histories'])=={'W','NO_WRITE'}
    for history,h in result['histories'].items():
        assert len(h['rows'])==len(order)
        assert len(h['births'])==(8 if stage=='A1-LEGACY-C4' else 32)
        assert len({r['fly_id'] for r in h['births']})==len(h['births'])
        assert all(r['canonical_B_sha256']=='32a3726cb25a8255c1aef5696ff40aa303138c5d061a598004d2087d4da5f964' for r in h['births'])
        for index,(r,item) in enumerate(zip(h['rows'],order),1):
            allowed=history=='W'
            assert r['index']==index and r['id']==item['id'] and r['allowed'] is allowed
            assert math.isfinite(r['raw_alpha_l1']) and r['raw_alpha_l1']>=0
            assert r['forbidden_delta']<=1e-8
            if not allowed:assert r['raw_alpha_l1']==0
            if stage=='A3-SEQUENCE-C16':
                targets=[*item['answer'],256]
                assert [t['target'] for t in r['trace']]==targets
                for t in r['trace']:checked_scores(t['logits'],list(range(257)),t['predicted'])
            else:
                assert r['target']==item['answer'][0]
                assert r['output_universe']==(list(b'0123') if stage=='A1-LEGACY-C4' else list(range(257)))
                checked_scores(r['logits'],r['output_universe'],r['predicted'])
        expected=[(n,'IMMEDIATE') for n in config.TRACK_A_DOSES]+[(config.TRACK_A_DOSES[-1],'AFTER_24H')]
        assert [(p['dose'],p['phase']) for p in h['probes']]==expected
        for p in h['probes']:
            assert [r['id'] for r in p['scores']['rows']]==[r['id'] for r in fixture]
            for row in p['scores']['rows']:
                item=items[row['id']];assert row['stratum']==item['stratum']
                if stage=='A3-SEQUENCE-C16':
                    emitted=bytes.fromhex(row['emitted']);expected=item['answer']
                    assert row['expected']==expected.hex()
                    assert row['status'] in ('EOS','TRUNCATED') and len(emitted)<=16
                    assert row['correct']==(emitted==expected and row['status']=='EOS')
                    assert row['conditional_is_autonomous_score'] is False
                    assert [r['target'] for r in row['teacher_conditioned']]==[*expected,256]
                    for t in row['teacher_conditioned']:
                        checked_scores(t['logits'],list(range(257)),t['predicted'],t['target'],t['target_margin'])
                    tokens=[]
                    assert 1<=len(row['trace'])<=17
                    for step,t in enumerate(row['trace']):
                        assert t['prefix']==bytes(tokens).hex()
                        checked_scores(t['logits'],list(range(257)),t['predicted'])
                        if t['predicted']==256:assert step==len(row['trace'])-1
                        if t['predicted']!=256 and len(tokens)<16:tokens.append(t['predicted'])
                    assert bytes(tokens)==emitted
                    assert (row['trace'][-1]['predicted']==256)==(row['status']=='EOS')
                    assert row['premature_EOS']==(row['status']=='EOS' and len(emitted)<len(expected))
                    if row['status']=='TRUNCATED':assert len(emitted)==16 and len(row['trace'])==17
                else:
                    target=item['answer'][0]
                    assert row['expected_target']==target and row['autonomous_generation'] is False
                    checked_scores(row['logits'],row['output_universe'],row['emitted_target'],target,row['target_margin'])
                    assert row['correct']==(row['emitted_target']==target)
    W=result['histories']['W']['probes'][-2]['scores']['rows']
    N=result['histories']['NO_WRITE']['probes'][-2]['scores']['rows']
    clean=[r for r in W if r['stratum']=='clean'];no_clean=[r for r in N if r['stratum']=='clean']
    acquired=all(r['correct'] for r in clean)
    if stage!='A3-SEQUENCE-C16':acquired=acquired and all(r['target_margin']>0 for r in clean)
    causal=sum(r['correct'] for r in clean)>sum(r['correct'] for r in no_clean)
    suffix=all(r['correct'] for r in W if r['stratum']=='suffix_collision')
    assert result['gate']['qualified']==(acquired and causal and (suffix if stage=='A3-SEQUENCE-C16' else True))
    return dict(accepted=True,science=False,stage=stage,record_applications=2*len(order),histories=2,
                scope='Independent stream, output and recorded native no-write residual checks; behavior qualification is a separate field.')


def audit_addition(result):
    assert result['science'] is False and result['world'] in config.TRACK_B_WORLDS
    assert result['version'] in config.TRACK_B_MODELS
    assert result['seed']==tasks.model_seed(result['world'])
    source=tasks.addition_streams(result['world'])
    assert set(result['histories'])==set(config.TRACK_B_HISTORY)
    total=0
    assert len({h['initial_digest'] for h in result['histories'].values()})==1
    for history,h in result['histories'].items():
        expected=[('OLD',i,*r) for i,r in enumerate(source['shuffled'] if history=='SH_OLD' else source['old'],1)]
        expected += [('NEW',i,*r) for i,r in enumerate(source['new'],1)]
        assert len(h['rows'])==len(expected)
        for row,(phase,index,prompt,answer) in zip(h['rows'],expected):
            assert (row['phase'],row['index'],row['prompt'],row['answer'])==(phase,index,prompt.hex(),answer.hex())
            allowed=phase!='OLD' or history!='N_OLD'
            assert row['allowed'] is allowed and row['actor_steps']==int(allowed)
            assert row['forbidden_delta']==0 and math.isfinite(row['loss'])
        expected_curve=[('OLD',n) for n in config.TRACK_B_OLD_DOSES]+[('OLD_REST',16384)]+[('NEW',n) for n in config.TRACK_B_NEW_DOSES]+[('NEW_REST',1024)]
        assert [(q['phase'],q['dose']) for q in h['curve']]==expected_curve
        for q in h['curve']:
            for name,pairs in source['groups'].items():
                assert len(q['scores'][name])==len(pairs)
                for row,pair in zip(q['scores'][name],pairs):
                    prompt,answer=tasks.render(pair)
                    assert (row['prompt'],row['expected'])==(prompt.hex(),answer.hex())
                    assert row['status'] in ('EOS','TRUNCATED') and len(bytes.fromhex(row['emitted']))<=16
                    assert row['correct']==(row['emitted']==answer.hex() and row['status']=='EOS')
        resource=h['resource'];answers=sum(len(y) for _,_,_,y in expected)
        if history=='N_OLD':assert h['initial_digest']==h['OLD_learning_end_digest']
        else:assert h['initial_digest']!=h['OLD_learning_end_digest']
        assert resource['records']==len(expected)
        assert resource['observed_answer_bytes']==answers and resource['observed_EOS_targets']==len(expected)
        assert resource['vocabulary_rows']==answers+len(expected)
        assert resource['actor_steps']==sum(r['allowed'] for r in h['rows'])
        assert h['parameter_count']==(28440 if result['version']=='KEEP-C16-CE' else 28800)
        total+=len(expected)
    assert result['record_applications']==total==52224
    return dict(accepted=True,science=False,world=result['world'],version=result['version'],
                correlated_histories=3,record_applications=total,held_feedback_records=0)
