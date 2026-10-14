"""Independent receipt arithmetic/causality checks; no learner import."""
import math, ast, random, collections

def independent_answer(prompt,seed):
    pi=list(range(256));random.Random(seed+918273).shuffle(pi);inv={v:k for k,v in enumerate(pi)}
    text=bytes(inv[b] for b in bytes.fromhex(prompt)).decode()
    assert text.endswith('=')
    def evaluate(t):
        if isinstance(t,ast.Constant) and type(t.value) is int:return t.value
        if isinstance(t,ast.UnaryOp) and isinstance(t.op,ast.USub):return -evaluate(t.operand)
        assert isinstance(t,ast.BinOp)
        a,b=evaluate(t.left),evaluate(t.right)
        if isinstance(t.op,ast.Add):return a+b
        if isinstance(t.op,ast.Sub):return a-b
        if isinstance(t.op,ast.Mult):return a*b
        assert isinstance(t.op,ast.Div) and b!=0 and a%b==0
        return a//b
    value=evaluate(ast.parse(text[:-1],mode='eval').body)
    return bytes(pi[b] for b in str(value).encode()).hex()

def independent_shortcuts(records,x):
    def mode(items):
        c=collections.Counter(items)
        return min(c,key=lambda a:(-c[a],a)) if c else None
    raw=bytes.fromhex(x['prompt']); global_mode=mode([r['reply'] for r in records])
    lm=mode([r['reply'] for r in records if len(bytes.fromhex(r['prompt']))==len(raw)]) or global_mode
    return {'exact':mode([r['reply'] for r in records if r['prompt']==x['prompt']]),
            'canonical':mode([r['reply'] for r in records if r['group']==x['group']]),
            'global':global_mode,'length':lm,
            'suffix':mode([r['reply'] for r in records if bytes.fromhex(r['prompt'])[-2:]==raw[-2:]]) or global_mode,
            'bag':mode([r['reply'] for r in records if sorted(bytes.fromhex(r['prompt']))==sorted(raw)]) or lm}

def audit(r,expected_source=None):
    assert r['schema']=='KERNEL_V3_TECHNICAL_JOB' and r['technical'] is True and r['completed'] is True
    assert r['executed_external_records']==3*r['binary_n']+4*r['mixed_n']
    assert set(r['branches'])=={'W','N_BINARY','SH_BINARY','N_MIXED'}
    if expected_source is not None:assert r['source']==expected_source
    assert all(len(v)==64 for v in r['source'].values())
    assert sum(len(s['records']) for b in r['branches'].values() for s in b['segments'])==r['executed_external_records']
    checks=[]
    for name,b in r['branches'].items():
        for seg in b['segments']:
            assert len(seg['records'])==(r['binary_n'] if seg['label']=='binary' else r['mixed_n'])
            allowed=not((name=='N_BINARY' and seg['label']=='binary') or (name=='N_MIXED' and seg['label']=='mixed'))
            assert seg['allowed'] is allowed
            if not allowed:assert seg['before']==seg['after']
            if seg['label']=='binary':assert seg['before']==r['initial_digest']
            before,after=seg['counts_before'],seg['counts_after']
            targets=0;payload=0
            for i,x in enumerate(seg['records']):
                assert x['index']==i and x['allowed'] is allowed
                if not(name=='SH_BINARY' and seg['label']=='binary'):assert x['reply']==independent_answer(x['prompt'],r['world'])
                tt=[*bytes.fromhex(x['reply']),256]
                assert [p['target'] for p in x['predictions']]==tt
                assert all(0<=p['choice']<=256 and math.isfinite(p['loss']) for p in x['predictions'])
                targets+=len(tt);payload+=len(bytes.fromhex(x['prompt']))+len(bytes.fromhex(x['reply']))
            assert after['external_records']-before['external_records']==len(seg['records'])
            assert after['external_payload_bytes']-before['external_payload_bytes']==payload
            assert after['actual_target_applications']-before['actual_target_applications']==(targets if allowed else 0)
            if not allowed:
                for k in ('optimizer_steps','replay_records','replay_target_applications'):assert after[k]==before[k]
        for probe in b.get('before_mixed',[])+b['final']:
            assert probe['before']==probe['after']
            for x in probe['rows']:
                assert x['expected']==independent_answer(x['prompt'],r['world'])
                assert x['exact'] is (x['actual']==x['expected'] and x['status']=='EOS')
                assert x['status'] in ('EOS','TRUNCATED')
                assert len(bytes.fromhex(x['actual']))<=16
                if 'shortcuts' in x:
                    old=r['branches']['W']['segments'][0]['records'] if name=='N_MIXED' else b['segments'][0]['records'] if name!='N_BINARY' else []
                    if name=='SH_BINARY':old=b['segments'][0]['records']
                    feedback=list(old)
                    if probe in b['final'] and name!='N_MIXED':feedback+=b['segments'][-1]['records']
                    assert x['shortcuts']==independent_shortcuts(feedback,x)
        assert b['state_bytes']['pending_records']==0 and b['state_bytes']['raw_replay_payload']<=4096
        if b['last_update']:assert b['last_update']['applied_RMS']<=.00100001
        checks.append(name+' write/exposure/output/resource audit')
    # Taught and held probes cannot reuse a literal reinforced W prompt/group.
    taught={x['prompt'] for seg in r['branches']['W']['segments'] for x in seg['records']}
    groups={x['group'] for seg in r['branches']['W']['segments'] for x in seg['records'] if 'group' in x}
    for probe in r['branches']['W']['final']:
        if probe['label'] in ('old_held','new_held'):
            assert all(x['prompt'] not in taught and x['group'] not in groups for x in probe['rows'])
    assert r['total_process_cpu']>0 and r['total_wall']>0
    return {'accepted':True,'scope':'technical receipt checks only','checks':checks+['held literal exclusion','exact ownEOS arithmetic','bounded displacement'],'scientific_jobs':0,'external_records':r['executed_external_records']}
