"""Independent read-only permission, exposure and receipt auditor.

No training is replayed. Births are reconstructed, and every record's complete
component chain and every response score are checked, including final states.
"""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from core import CONFIGS, build_core
from worlds import OPS, digest, make_world


def recover(row):
    op, a, b, y = row
    if op == '+':
        assert a+b == y
        return 'SUM', tuple(sorted((a,b)))
    if op == '-':
        assert a-b == y
        return 'SUM', tuple(sorted((b,y) if y >= 0 else (a,-y)))
    if op == '*':
        assert a*b == y
        return 'PRODUCT', tuple(sorted((a,b)))
    assert op == '/' and b > 0 and a == b*y
    return 'PRODUCT', tuple(sorted((b,y)))


def audit_answers(answers, pool):
    assert len(answers) == len(pool), 'probe roster'
    for x, row in zip(answers, pool):
        op, a, b, y = row
        assert x['prompt_hex'] == f'{a}{op}{b}='.encode().hex()
        assert x['target_hex'] == str(y).encode().hex() and x['operation'] == op
        payload = bytes.fromhex(x['payload_hex'])
        assert len(payload) <= 16 and x['status'] in ('ENDED','TRUNCATED')
        assert x['selected_events'] == len(payload)+1
        assert type(x['correct']) is bool
        assert x['correct'] == (payload == str(y).encode() and x['status'] == 'ENDED')


def score_answers(answers):
    return {op: sum(x['correct'] for x in answers if x['operation']==op)/
                 sum(x['operation']==op for x in answers) for op in OPS}


def audit(receipt, check_sources=False):
    assert receipt['schema'] == 'VALUE_INTERACT_JOB_V1'
    assert receipt['config'] in CONFIGS and type(receipt['technical']) is bool
    assert digest({k:v for k,v in receipt.items() if k!='receipt_digest'}) == receipt['receipt_digest']
    plan = receipt['plan']
    assert make_world(receipt['world'],receipt['technical'],tuple(plan['checkpoints'])) == plan
    training, held = plan['training_pool'], plan['held_pool']
    assert {recover(r) for r in training}.isdisjoint({recover(r) for r in held}), 'family leakage'
    assert {str(r[:3]) for r in training}.isdisjoint(str(r[:3]) for r in held)
    if check_sources:
        from runner import source_lock
        assert receipt['source_lock'] == source_lock(), 'source coverage/identity mismatch'
    assert set(receipt['histories']) == {'W','N_ALL','SHUFFLE'}
    if check_sources:assert 'resource' in receipt, 'missing resource accounting'
    if 'resource' in receipt:
        m=receipt['resource']
        for key in ('pre_commit_process_cpu_seconds','probe_cpu_seconds','training_core_cpu_seconds'):
            assert type(m[key]) in (int,float) and math.isfinite(m[key]) and m[key]>=0
        assert m['training_core_cpu_seconds']==sum(h['training_core_cpu'] for h in receipt['histories'].values())
        assert m['pre_commit_process_cpu_seconds']>=m['probe_cpu_seconds']+m['training_core_cpu_seconds']
        assert m['observed_reply_events']==sum(len(bytes.fromhex(row['target_hex']))+1
            for h in receipt['histories'].values() for row in h['ledger'])

    initial = build_core(receipt['config'],receipt['world'])
    birth, geometry = initial.learned_digest(), initial.birth_digest()
    initial_parts = initial.persistent_components()
    rows_total = 0
    for name,h in receipt['histories'].items():
        assert h['birth']==birth and h['birth_fixed']==geometry, 'birth mismatch'
        assert len(h['ledger'])==len(plan['records'])
        audit_answers(h['initial'],held)
        learned=0; auxiliary=0; prior=initial_parts; counts=[0]*4
        for row,rec in zip(h['ledger'],plan['records']):
            assert row['index']==rec['index'] and row['operation']==rec['operation']
            assert row['prompt_hex']==rec['prompt'].encode().hex()
            target=plan['shuffled_answers'][str(rec['index'])] if name=='SHUFFLE' else rec['answer']
            assert row['target_hex']==target.encode().hex(), 'target provenance'
            assert row['before']==prior, 'state chain'
            assert set(row['after'])==set(initial_parts), 'component roster'
            prior=row['after']
            expected=[(b,'prompt') for b in rec['prompt'].encode()]
            expected += [(b,'reply') for b in target.encode()]+[(256,'reply')]
            assert len(row['events'])==len(expected)
            for x,(event,phase) in zip(row['events'],expected):
                permit=int(name!='N_ALL' and phase=='reply')
                assert x['event']==event and x['phase']==phase
                assert type(x['actual_update_count']) is int and x['actual_update_count']==permit, 'branch write'
                assert type(x['prior_choice']) is int and 0<=x['prior_choice']<=256
                if receipt['config']!='LEGACY-B':
                    info=x['core_audit']
                    assert info['source']=='actual_observation' and info['phase']==phase
                    assert info['event']==event and info['decoder_target_input'] is False
                    assert info['gradient_snapshot']=='preupdate_preappend'
                    assert info['step_before']==learned and info['step_after']==learned+permit
                    numeric=int(target) if phase=='reply' and event==256 else None
                    used=receipt['config'].startswith('VALUE') and bool(permit) and numeric is not None
                    assert type(info['aux_used']) is bool and info['aux_used']==used, 'aux permission'
                    assert info['numeric_observed_target']==numeric, 'observed-only numeric target'
                    assert info['aux_target']==(numeric if used else None), 'aux target provenance'
                    auxiliary+=int(used)
                else:assert x['core_audit'] is None
                learned+=permit;rows_total+=1
            assert row['learning_events']==learned
            if name=='N_ALL':
                assert row['before']==row['after'], 'hidden forbidden update'
            if receipt['config']=='LEGACY-B':
                assert type(row['bank']) is int and 0<=row['bank']<4
                if name!='N_ALL': counts[row['bank']]+=1
                assert row['bank_counts']==counts
                assert row['after']['centers']==initial_parts['centers']
            else:
                assert row['bank'] is None and row['bank_counts'] is None
                assert row['optimizer_step']==learned and row['aux_events']==auxiliary
                assert row['after']['optimizer_step']==hashlib.sha256(str(learned).encode()).hexdigest()
                assert row['after']['aux_events']==hashlib.sha256(str(auxiliary).encode()).hexdigest()
        assert h['learning_events']==learned
        obj=dict(prior,learning_events=learned)
        assert h['final_digest']==hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest(), 'final state chain'
        assert len(h['checkpoints'])==len(plan['checkpoints'])
        assert len(h['blocks'])==4*len(plan['checkpoints'])
        old_cpu=0
        for cp,n in zip(h['checkpoints'],plan['checkpoints']):
            assert cp['samples_per_operation']==n and cp['cpu']>old_cpu
            old_cpu=cp['cpu']
            prefix=plan['records'][:4*n]
            seen={r['prompt'] for r in prefix}
            taught=[r for r in training if f'{r[1]}{r[0]}{r[2]}=' in seen]
            audit_answers(cp['held'],held);audit_answers(cp['taught'],taught)
            assert cp['scores']==score_answers(cp['held'])
            for op in OPS:
                assert cp['unique_seen_prompts'][op]==len({r['prompt'] for r in prefix if r['operation']==op})
                labels=[bytes.fromhex(x['target_hex']) for x in h['ledger'][:4*n] if x['operation']==op]
                assert cp['observed_answer_bytes'][op]==sorted(set(b''.join(labels)))
                assert cp['countermodels']['exact'][op]==0
                if name!='SHUFFLE':assert cp['countermodels']['inverse_family_lookup'][op]==0
            assert cp['countermodels']==reference_countermodels(plan,4*n,name), 'countermodel calculation'
        last_cpu=0
        for block,decl in zip(h['blocks'],plan['blocks']):
            assert (block['operation'],block['checkpoint'])==(decl['operation'],decl['checkpoint'])
            audit_answers(block['held'],held)
            prefix=plan['records'][:decl['record_indices'][-1]+1]
            seen={r['prompt'] for r in prefix}
            block_taught=[r for r in training if f'{r[1]}{r[0]}{r[2]}=' in seen]
            audit_answers(block['taught'],block_taught)
            assert block['scores']==score_answers(block['held'])
            assert block['cpu']>last_cpu;last_cpu=block['cpu']
        audit_answers(h['final'],held)
        assert h['simulated_seconds']==86400 and h['training_core_cpu']==old_cpu
    for b in plan['blocks']:
        ids=b['record_indices']
        assert Counter(plan['records'][i]['answer'] for i in ids)==Counter(plan['shuffled_answers'][str(i)] for i in ids)
    assert receipt['records_applied']==3*len(plan['records'])
    actual_bytes=sum(len(bytes.fromhex(x['prompt_hex']))+len(bytes.fromhex(x['target_hex']))
                     for h in receipt['histories'].values() for x in h['ledger'])
    assert receipt['observed_payload_bytes']==actual_bytes
    assert receipt['scientific_jobs']==int(not receipt['technical'])
    return {'accepted':True,'technical':receipt['technical'],'world':receipt['world'],
            'config':receipt['config'],'records_applied':receipt['records_applied'],
            'observed_event_rows':rows_total,'science_jobs':int(not receipt['technical'])}


def reference_countermodels(plan,prefix,history):
    # Separate implementation; never call the production countermodel builder.
    groups={name:{} for name in ('exact','modal','left','right','inverse_family_lookup')}
    def add(name,key,value):
        bucket=groups[name].setdefault(key,Counter());bucket[str(value)]+=1
    for rec in plan['records'][:prefix]:
        op,a,b,_=rec['row'];label=plan['shuffled_answers'][str(rec['index'])] if history=='SHUFFLE' else rec['answer']
        y=int(label)
        for name,key in [('exact',(op,a,b)),('modal',(op,)),('left',(op,a)),('right',(op,b))]:add(name,key,label)
        if op in ('+','-'):
            if op=='+':left,right,total=a,b,y
            elif y<0:left,right,total=a,-y,b
            else:left,right,total=b,y,a
            facts=[('+',left,right,total),('+',right,left,total),('-',total,left,right),
                   ('-',total,right,left),('-',left,total,-right),('-',right,total,-left)]
        else:
            left,right,product=(a,b,y) if op=='*' else (b,y,a)
            facts=[('*',left,right,product),('*',right,left,product)]
            if left: facts.append(('/',product,left,right))
            if right:facts.append(('/',product,right,left))
        for o,l,r,t in set(facts):add('inverse_family_lookup',(o,l,r),t)
    result={}
    for name,buckets in groups.items():
        choices={k:sorted(counts,key=lambda y:(-counts[y],y))[0] for k,counts in buckets.items()}
        result[name]={}
        for op in OPS:
            hits=[]
            for o,a,b,y in plan['held_pool']:
                if o!=op:continue
                key=(op,a,b) if name in ('exact','inverse_family_lookup') else (op,) if name=='modal' else (op,a) if name=='left' else (op,b)
                hits.append(choices.get(key)==str(y))
            result[name][op]=sum(hits)/len(hits)
    return result


if __name__=='__main__':
    with gzip.open(sys.argv[1],'rt') as f:r=json.load(f)
    print(json.dumps(audit(r,check_sources='--check-sources' in sys.argv)))
