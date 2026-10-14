"""Receipt-only independent truth, exposure, feedback and count checks."""
import ast,collections,hashlib,importlib.util,json,math,random
import numpy as np
from functools import lru_cache
from pathlib import Path
from protocol import ROOT,D,seed,show,plans,probes
path=ROOT.parent/'arithmetic_source_derived_kernel_dev_v1_20261013/auditor.py'
spec=importlib.util.spec_from_file_location('old_truth_auditor',path)
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)

def audit(r,source=None):
    assert r['schema']=='MODEL_FEEDBACK_JOB_V1' and r['completed'] is True
    assert r['model'] in ('R0','KEEP')
    assert r['model_seed']==seed(r['world'],0,'model')
    if source is not None:assert r['source']==source
    assert set(r['prefixes'])=={'TRUE','NO_WARM','SH_WARM'}
    assert set(r['branches'])=={'FADING','DENSE','NO_WARM','SH_WARM','FREEZE_SCALAR'}
    w,warm,shuffled,tails=plans(r['world'],r['binary'],r['mixed'],r['stages']);assert w['data_digest']==r['data_digest']
    truth=lru_cache(None)(lambda p:A.independent_answer(p,r['world']))
    group=lru_cache(None)(lambda p:A.independent_group(p,r['world']))
    pi=list(range(256));random.Random(r['world']+918273).shuffle(pi);inv={b:i for i,b in enumerate(pi)}
    @lru_cache(None)
    def operation(p):
        t=ast.parse(bytes(inv[b] for b in bytes.fromhex(p)).decode()[:-1],mode='eval').body
        names={ast.Add:'+',ast.Sub:'-',ast.Mult:'*',ast.Div:'/'};root=names[type(t.op)]
        x=t.left if isinstance(t.left,ast.BinOp) else t.right if isinstance(t.right,ast.BinOp) else None
        return root if x is None else names[type(x.op)]+root
    # Countermodel construction is independent of the producer. It uses only
    # visible demonstrations or an actual generated answer verified by r=1.
    def counters(records):
        tables={k:collections.defaultdict(collections.Counter) for k in ('exact','canonical','length','suffix','bag')};glob=collections.Counter()
        for row in records:
            a=row['observed'] if row['observed'] is not None else row['actual'] if row['reward']==1 else None
            if a is None:continue
            p=bytes.fromhex(row['prompt']);keys=(row['prompt'],row['group'],len(p),p[-2:],tuple(sorted(p)))
            glob[a]+=1
            for k,x in zip(tables,keys):tables[k][x][a]+=1
        return tables,glob
    def prediction(tables,glob,row):
        mode=lambda c:min(c,key=lambda a:(-c[a],a)) if c else None
        p=bytes.fromhex(row['prompt']);keys=(row['prompt'],row['group'],len(p),p[-2:],tuple(sorted(p)))
        a={k:mode(tables[k].get(x,{})) for k,x in zip(tables,keys)};a['global']=mode(glob)
        a['length']=a['length'] or a['global'];a['suffix']=a['suffix'] or a['global'];a['bag']=a['bag'] or a['length'];return a
    trained={x['group'] for x in warm+sum(list(tails.values()),[])}
    total=0;steps=0;received=0;queries=0
    def check_scores(entry,feedback,taught=False):
        nonlocal queries
        assert entry['before']==entry['after']
        tables,glob=counters(feedback)
        for label in ('binary','mixed'):
            exposed={x['prompt']:dict(prompt=x['prompt'],reply=x['expected'],operation=x['operation'],group=x['group']) for x in feedback}
            wanted=probes([x for x in exposed.values() if len(x['operation'])==(1 if label=='binary' else 2)],None) if taught else probes(w[label+'_held'],None)
            assert [(x['prompt'],x['group'],x['operation']) for x in entry['scores'][label]]==[(x['prompt'],x['group'],x['operation']) for x in wanted]
            for row in entry['scores'][label]:
                assert row['reply']==truth(row['prompt']) and row['group']==group(row['prompt'])
                assert row['operation']==operation(row['prompt'])
                assert row['status'] in ('EOS','TRUNCATED') and len(bytes.fromhex(row['actual']))<=16
                assert row['exact'] is (row['actual']==row['reply'] and row['status']=='EOS')
                assert row['shortcuts']==prediction(tables,glob,row)
                if not taught:assert row['group'] not in trained
                queries+=1
    def check_segment(seg,plan,phase,kind,allowed,offset,feedback):
        nonlocal total,steps,received
        assert seg['phase']==phase and seg['kind']==kind and seg['allowed'] is allowed and seg['offset']==offset
        if not allowed:assert seg['before']==seg['after']
        assert len(seg['records'])==len(plan)
        checkpoints={c['n']:c for c in seg['curve']}
        for i,(row,wanted) in enumerate(zip(seg['records'],plan)):
            assert row['index']==offset+i and row['allowed'] is allowed
            assert row['learning_mode']==('DENSE' if kind=='DENSE' and phase!='WARM' else phase)
            assert row['prompt']==wanted['prompt'] and row['group']==group(row['prompt'])==wanted['group']
            assert row['operation']==operation(row['prompt'])==wanted['operation']
            assert row['expected']==truth(row['prompt'])
            expected_obs=wanted['reply'] if kind=='DENSE' or show(r['world'],offset+i,phase) else None
            assert row['observed']==expected_obs
            assert type(row['reward']) is int and row['reward']==int(row['actual']==row['expected'] and row['status']=='EOS')
            assert math.isfinite(row['baseline']) and 0<=row['baseline']<=1
            assert row['status'] in ('EOS','TRUNCATED') and 1<=len(row['actions'])<=17
            assert all(type(b) is int and 0<=b<=256 for b in row['actions'])
            if row['status']=='EOS':
                assert row['actions'][-1]==256 and row['actions'][:-1]==list(bytes.fromhex(row['actual']))
            else:assert len(bytes.fromhex(row['actual']))==16 and row['actions'][:16]==list(bytes.fromhex(row['actual'])) and row['actions'][-1]!=256
            update=row['update'];assert update['actor_steps']==update['critic_steps']==int(allowed)
            assert (update['ce'] is not None)==(allowed and expected_obs is not None)
            assert (update['pg'] is not None)==(allowed and phase!='WARM')
            assert (update['critic'] is not None)==allowed
            for key in ('ce','pg','critic'):
                if update[key] is not None:assert math.isfinite(update[key])
            if phase=='SCALAR' and kind!='DENSE':assert row['observed'] is None and update['ce'] is None
            feedback.append(row);total+=1;steps+=int(allowed);received+=len(bytes.fromhex(expected_obs)) if expected_obs is not None else 0
            if offset+i+1 in checkpoints:check_scores(checkpoints[offset+i+1],feedback)
        a,b=seg['resources_before'],seg['resources_after']
        assert b['records']-a['records']==len(plan)
        assert b['actor_steps']-a['actor_steps']==b['critic_steps']-a['critic_steps']==int(allowed)*len(plan)
        assert b['scalar_feedback']-a['scalar_feedback']==len(plan)
    prefixes={}
    fw=r['binary']+r['mixed']
    for name,plan,allowed in [('TRUE',warm,True),('NO_WARM',warm,False),('SH_WARM',shuffled,True)]:
        x=r['prefixes'][name];assert x['segment']['before']==r['initial']
        fb=[];check_scores(x['curve'][0],fb)
        check_segment(x['segment'],plan,'WARM',name,allowed,0,fb)
        assert x['curve'][1:]==x['segment']['curve'] and x['state']==x['segment']['after']
        check_scores(x['taught'],fb,True);prefixes[name]=fb
        if name=='NO_WARM':
            assert x['state']==r['initial']
            assert all([[q['actual'],q['status']] for label in ('binary','mixed') for q in c['scores'][label]]==[[q['actual'],q['status']] for label in ('binary','mixed') for q in x['curve'][0]['scores'][label]] for c in x['curve'])
    for name,b in r['branches'].items():
        if name=='FREEZE_SCALAR':
            parent=r['branches']['FADING'];fb=list(prefixes['TRUE'])+sum([s['records'] for s in parent['segments'][:2]],[])
            expected_curve=parent['curve'][:-2];offset=fw+sum(r['stages'][:2])
            assert b['fork_state']==parent['segments'][-1]['before']==b['final']
            check_segment(b['segments'][0],tails['SCALAR'],'SCALAR',name,False,offset,fb)
            assert b['curve']==expected_curve+b['segments'][0]['curve']
            fork_scores=expected_curve[-1]['scores']
            assert all({k:[(x['actual'],x['status']) for x in rows] for k,rows in c['scores'].items()}=={k:[(x['actual'],x['status']) for x in rows] for k,rows in fork_scores.items()} for c in b['segments'][0]['curve'])
        else:
            parent=b['prefix'];fb=list(prefixes[parent]);offset=fw;curves=list(r['prefixes'][parent]['curve'])
            assert b['segments'][0]['before']==r['prefixes'][parent]['state']
            for phase,seg in zip(('HALF','SPARSE','SCALAR'),b['segments']):
                check_segment(seg,tails[phase],phase,name,True,offset,fb);offset+=len(tails[phase]);curves.extend(seg['curve'])
            assert b['curve']==curves and b['final']==b['segments'][-1]['after']
        assert all(c['n']<d['n'] and c['cpu']<=d['cpu'] for c,d in zip(b['curve'],b['curve'][1:]))
        check_scores(b['taught'],fb,True)
    assert total==r['executed_records']==3*fw+4*sum(r['stages'][:2])+5*r['stages'][-1]
    assert r['correlated_final_histories']==5
    assert len(r['archives'])==13
    assert all((ROOT/p).is_file() for p in r['archives'].values())
    expected_states={'BIRTH.npz':r['initial']}
    expected_states.update({k+'_WARM.npz':v['state'] for k,v in r['prefixes'].items()})
    expected_states.update({k+'_FINAL.npz':v['final'] for k,v in r['branches'].items()})
    expected_states.update({k+'_SCALAR_START.npz':v['segments'][-1]['before'] for k,v in r['branches'].items() if k!='FREEZE_SCALAR'})
    assert set(r['archives'])==set(expected_states)
    for name,p in r['archives'].items():
        with np.load(ROOT/p,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata']));assert meta['model']==r['model']
            h=hashlib.sha256()
            for label,prefix in [('parameter','p_'),('moment','m_')]:
                names=sorted(k[len(prefix):] for k in data.files if k.startswith(prefix))
                for k in names:
                    a=data[prefix+k];assert a.dtype==np.float64 and np.isfinite(a).all()
                    h.update((label+k).encode());h.update(a.tobytes())
            h.update(json.dumps([meta['actor_steps'],meta['critic_steps'],meta['last']],sort_keys=True,allow_nan=False).encode())
            assert h.hexdigest()==meta['digest']==expected_states[name]
            assert ('p_C' in data.files)==(r['model']=='KEEP')
    return dict(accepted=True,world=r['world'],model=r['model'],records=total,actor_updates=steps,critic_updates=steps,
                answer_payload_bytes=received,checked_query_rows=queries,learning_histories=5,checked_archives=13,scope='independent truth/feedback/exposure/causality, archive and count audit, not learned algorithm proof')
