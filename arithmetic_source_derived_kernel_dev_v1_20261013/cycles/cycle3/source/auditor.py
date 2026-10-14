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

def independent_group(prompt,seed):
    pi=list(range(256));random.Random(seed+918273).shuffle(pi);inv={v:k for k,v in enumerate(pi)}
    text=bytes(inv[b] for b in bytes.fromhex(prompt)).decode()[:-1]
    names={ast.Add:'+',ast.Sub:'-',ast.Mult:'*',ast.Div:'/'}
    def value(t):
        if isinstance(t,ast.Constant):return t.value
        if isinstance(t,ast.UnaryOp):return -value(t.operand)
        a,b=value(t.left),value(t.right);op=names[type(t.op)]
        return a+b if op=='+' else a-b if op=='-' else a*b if op=='*' else a//b
    def canon(t):
        if not isinstance(t,ast.BinOp):return ('leaf',value(t))
        op=names[type(t.op)];kids=[]
        def add(x):
            if op in ('+','*') and isinstance(x,ast.BinOp) and names[type(x.op)]==op:add(x.left);add(x.right)
            else:kids.append(canon(x))
        add(t.left);add(t.right)
        if op in ('+','*'):kids.sort(key=repr)
        return (op,*kids)
    t=ast.parse(text,mode='eval').body
    if not isinstance(t.left,ast.BinOp) and not isinstance(t.right,ast.BinOp):
        a,b,y=value(t.left),value(t.right),value(t);op=names[type(t.op)]
        if op in ('+','-'):
            s=sorted((abs(a),abs(b),abs(y)));return repr(('SUM',s[0],s[1]))
        pair=sorted((abs(a),abs(b)) if op=='*' else (abs(b),abs(y)));return repr(('PRODUCT',*pair))
    return repr(canon(t))



def audit(r, expected_source=None):
    from functools import lru_cache
    assert r['schema']=='SOURCE_CELL_TECHNICAL_JOB' and r['technical'] is True and r['completed'] is True
    assert r['world'] in (202610138902,202610138903)
    assert r['arm'] in ('B16','B24','B69','E6','E6_FIX')
    assert r['executed_external_records']==3*r['binary_n']+4*r['mixed_n']
    assert r['permitted_optimizer_steps']==2*r['binary_n']+3*r['mixed_n']
    assert r['correlated_final_histories']==4
    assert set(r['branches'])=={'W','N_BINARY','SH_BINARY','N_MIXED'}
    if expected_source is not None: assert r['source']==expected_source
    # Shared source generator defines membership/schedule, not truth: labels,
    # canonical identity and operation are checked independently through AST.
    from worlds import make_world, schedule, shuffle_targets, probes
    world=make_world(r['world'])
    assert r['data_digest']==world['data_digest']
    old_plan=schedule(world,'binary',r['binary_n'])
    new_plan=schedule(world,'mixed',r['mixed_n'])
    shuffled_plan=shuffle_targets(old_plan,r['world']+19)
    held_plan={'old_held':probes(world['binary_held'],None),'new_held':probes(world['mixed_held'],None)}
    def operation(prompt):
        pi=list(range(256));random.Random(r['world']+918273).shuffle(pi);inv={v:k for k,v in enumerate(pi)}
        t=ast.parse(bytes(inv[b] for b in bytes.fromhex(prompt)).decode()[:-1],mode='eval').body
        names={ast.Add:'+',ast.Sub:'-',ast.Mult:'*',ast.Div:'/'}
        root=names[type(t.op)]
        inner=t.left if isinstance(t.left,ast.BinOp) else t.right if isinstance(t.right,ast.BinOp) else None
        return names[type(inner.op)]+root if inner is not None else root
    operation=lru_cache(None)(operation)
    answer=lru_cache(None)(lambda x:independent_answer(x,r['world']))
    group=lru_cache(None)(lambda x:independent_group(x,r['world']))
    total=updates=targets=queries=0
    old_w=r['branches']['W']['segments'][0]['records']
    all_taught={x['prompt'] for seg in r['branches']['W']['segments'] for x in seg['records']}
    all_groups={x['group'] for seg in r['branches']['W']['segments'] for x in seg['records']}
    def counters(records):
        tables={k:collections.defaultdict(collections.Counter) for k in ('exact','canonical','length','suffix','bag')}; glob=collections.Counter()
        for row in records:
            p=bytes.fromhex(row['prompt']); keys=(row['prompt'],row['group'],len(p),p[-2:],tuple(sorted(p)))
            glob[row['reply']]+=1
            for name,key in zip(tables,keys):tables[name][key][row['reply']]+=1
        return tables,glob
    def shortcuts(tables,glob,row):
        mode=lambda c:min(c,key=lambda key:(-c[key],key)) if c else None
        p=bytes.fromhex(row['prompt']); keys=(row['prompt'],row['group'],len(p),p[-2:],tuple(sorted(p)))
        out={name:mode(tables[name].get(key,{})) for name,key in zip(tables,keys)};out['global']=mode(glob)
        out['length']=out['length'] or out['global'];out['suffix']=out['suffix'] or out['global'];out['bag']=out['bag'] or out['length']
        return out
    def check_probe(probe,feedback,held=False):
        nonlocal queries
        assert probe['before']==probe['after']
        tables,glob=counters(feedback)
        if held:
            plan=held_plan[probe['label']]
            actual=[(x['prompt'],x['group'],x['operation']) for x in probe['rows']]
            wanted=[(x['prompt'],x['group'],x['operation']) for x in plan]
            assert actual==wanted, 'changed or duplicate selected held groups'
        for row in probe['rows']:
            assert row['expected']==answer(row['prompt'])
            assert row['group']==group(row['prompt'])
            assert row['operation']==operation(row['prompt'])
            assert row['status'] in ('EOS','TRUNCATED')
            assert len(bytes.fromhex(row['actual']))<=16
            assert row['exact'] is (row['actual']==row['expected'] and row['status']=='EOS')
            assert row['shortcuts']==shortcuts(tables,glob,row)
            if held:assert row['prompt'] not in all_taught and row['group'] not in all_groups
            queries+=1
    for name,branch in r['branches'].items():
        old=old_w if name=='N_MIXED' else branch['segments'][0]['records'] if name!='N_BINARY' else []
        for seg in branch['segments']:
            allowed=not ((name=='N_BINARY' and seg['label']=='binary') or(name=='N_MIXED' and seg['label']=='mixed'))
            assert seg['allowed'] is allowed
            expected_n=r['binary_n'] if seg['label']=='binary' else r['mixed_n']
            assert len(seg['records'])==expected_n
            expected_stream=new_plan if seg['label']=='mixed' else shuffled_plan if name=='SH_BINARY' else old_plan
            assert [(x['prompt'],x['reply'],x['group'],x['operation']) for x in seg['records']]==[(x['prompt'],x['reply'],x['group'],x['operation']) for x in expected_stream]
            if not allowed:assert seg['before']==seg['after']
            if seg['label']=='binary':assert seg['before']==r['initial_digest']
            step_count=target_count=payload=0
            for i,row in enumerate(seg['records']):
                assert row['index']==i and row['allowed'] is allowed and row['optimizer_steps']==int(allowed)
                assert row['group']==group(row['prompt'])
                assert row['operation']==operation(row['prompt'])
                if not(name=='SH_BINARY' and seg['label']=='binary'):assert row['reply']==answer(row['prompt'])
                ts=[*bytes.fromhex(row['reply']),256]
                assert [p['target'] for p in row['predictions']]==ts
                assert all(0<=p['choice']<=256 and math.isfinite(p['loss']) for p in row['predictions'])
                step_count+=row['optimizer_steps'];target_count+=len(ts);payload+=len(bytes.fromhex(row['prompt']))+len(ts)-1
            a,b=seg['counts_before'],seg['counts_after']
            assert b['external_records']-a['external_records']==expected_n
            assert b['external_payload_bytes']-a['external_payload_bytes']==payload
            assert b['optimizer_steps']-a['optimizer_steps']==step_count
            assert b['actual_target_applications']-a['actual_target_applications']==(target_count if allowed else 0)
            total+=expected_n;updates+=step_count;targets+=target_count if allowed else 0
        for phase in ('binary','mixed'):
            if phase=='binary' and name=='N_MIXED':assert branch['binary_curve_parent']=='W';continue
            curve=branch[phase+'_curve'];nmax=r[phase+'_n']
            points=[0]+[n for n in (256,512,1024,2048,4096,8192) if n<=nmax] if phase=='binary' else [n for n in (64,128,256,512,1024,2048) if n<=nmax]
            base=0 if phase=='binary' else r['binary_n']
            assert [e['external_W_coordinates']-base for e in curve]==points
            costs=[]
            for n,entry in zip(points,curve):
                feedback=old[:n] if phase=='binary' else old+(branch['segments'][-1]['records'][:n] if name!='N_MIXED' else[])
                assert entry['resource']['external_records']==base+n
                assert math.isfinite(entry['production_cpu']) and entry['production_cpu']>0
                assert abs(entry['production_cpu']-(entry['resource']['training_cpu']+entry['resource']['generation_cpu']))<1e-9
                assert [p['label'] for p in entry['probes']]==['old_held','new_held']
                assert [len(p['rows']) for p in entry['probes']]==[128,256]
                for probe in entry['probes']:check_probe(probe,feedback,True)
                costs.append(entry['production_cpu'])
            assert costs==sorted(costs)
        feedback=old+(branch['segments'][-1]['records'] if name!='N_MIXED' else[])
        for probe in branch['taught_final']:check_probe(probe,feedback)
        assert branch['state_bytes']['pending_gradient_bytes']==0 and branch['state_bytes']['raw_history_buffer_bytes']==0
    for name in ('N_BINARY','SH_BINARY'):
        b=r['branches'][name]['segments'][0]['records']
        assert [x['prompt'] for x in b]==[x['prompt'] for x in old_w]
        if name=='N_BINARY':assert [x['reply'] for x in b]==[x['reply'] for x in old_w]
        else:
            for length in set(len(bytes.fromhex(x['reply'])) for x in old_w):
                assert collections.Counter(x['reply'] for x in b if len(bytes.fromhex(x['reply']))==length)==collections.Counter(x['reply'] for x in old_w if len(bytes.fromhex(x['reply']))==length)
    assert r['branches']['N_MIXED']['segments'][0]['before']==r['branches']['W']['segments'][0]['after']
    assert total==r['executed_external_records'] and updates==r['permitted_optimizer_steps']
    streams=[b['segments'][-1]['records'] for b in r['branches'].values()]
    assert all([(x['prompt'],x['reply']) for x in s]==[(x['prompt'],x['reply']) for x in streams[0]] for s in streams)
    assert r['total_process_cpu']>0 and r['total_wall']>0
    return dict(accepted=True,scope='independent technical-record audit;no science claim',world=r['world'],arm=r['arm'],external_records=total,optimizer_steps=updates,permitted_targets=targets,independently_checked_query_rows=queries,scientific_jobs=0)
