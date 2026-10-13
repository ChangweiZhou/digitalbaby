"""Read-only receipt auditor: branch permission is reconstructed, never trusted."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
from worlds import OPS, digest, make_world


def recover(row):
    op, a, b, y = row
    if op == '+':
        assert a+b == y
        return ('SUM', tuple(sorted((a,b))))
    if op == '-':
        assert a-b == y
        return ('SUM', tuple(sorted((b,y) if y >= 0 else (a,-y))))
    if op == '*':
        assert a*b == y
        return ('PRODUCT', tuple(sorted((a,b))))
    assert op == '/' and b != 0 and a == b*y
    return ('PRODUCT', tuple(sorted((b,y))))


def audit(receipt, check_sources=False):
    assert receipt['schema'] == 'ARITHMETIC_TRANSFER_JOB_V2'
    obj = {k:v for k,v in receipt.items() if k != 'receipt_digest'}
    assert digest(obj) == receipt['receipt_digest'], 'receipt checksum'
    plan = receipt['plan']
    reconstructed = make_world(receipt['world'], receipt['technical'], tuple(plan['checkpoints']))
    assert reconstructed == plan, 'world/stream generation mismatch'
    training, held = plan['training_pool'], plan['held_pool']
    assert {recover(r) for r in training}.isdisjoint({recover(r) for r in held}), 'inverse family leak'
    assert {str(r[:3]) for r in training}.isdisjoint(str(r[:3]) for r in held), 'exact cue leak'
    if check_sources:
        root = Path(__file__).resolve().parent.parent
        for rel, sha in receipt['source_lock'].items():
            assert hashlib.sha256((root/rel).read_bytes()).hexdigest() == sha, f'stale source {rel}'
    assert set(receipt['histories']) == {'W','N_ALL','SHUFFLE'}
    births = {v['birth'] for v in receipt['histories'].values()}
    assert len(births) == 1, 'history birth mismatch'
    assert len({v['birth_fixed'] for v in receipt['histories'].values()}) == 1, 'fixed birth mismatch'
    total_events = 0
    for name, history in receipt['histories'].items():
        assert len(history['ledger']) == len(plan['records'])
        learned = 0
        counts = [0]*4
        prior = None
        for row, rec in zip(history['ledger'], plan['records']):
            assert row['index'] == rec['index'] and row['operation'] == rec['operation']
            assert row['prompt_hex'] == rec['prompt'].encode().hex()
            expected = plan['shuffled_answers'][str(rec['index'])] if name == 'SHUFFLE' else rec['answer']
            assert row['target_hex'] == expected.encode().hex(), 'target provenance'
            events = [(b,'prompt') for b in rec['prompt'].encode()]
            events += [(b,'reply') for b in expected.encode()] + [(256,'reply')]
            assert len(events) == len(row['events'])
            if prior is not None:
                assert row['before'] == prior, 'state-chain continuity'
            prior = row['after']
            for actual, (byte, phase) in zip(row['events'], events):
                assert actual['event'] == byte and actual['phase'] == phase
                expected_update = int(name != 'N_ALL' and phase == 'reply')
                assert actual['actual_update_count'] == expected_update, 'causal branch write violation'
                assert 0 <= actual['prior_choice'] <= 256
                learned += expected_update
                total_events += 1
            assert row['learning_events'] == learned
            if name == 'N_ALL':
                assert row['before'] == row['after'], 'disabled hidden persistent update'
            if receipt['config'] != 'HASH':
                if name != 'N_ALL':
                    counts[row['bank']] += 1
                assert row['bank_counts'] == counts, 'router learning count'
                if '-FIXED-' in receipt['config']:
                    assert row['before']['centers'] == row['after']['centers'], 'fixed router modified'
        assert history['learning_events'] == learned
        assert len(history['checkpoints']) == len(plan['checkpoints'])
        assert len(history['blocks']) == 4*len(plan['checkpoints'])
        for checkpoint, n in zip(history['checkpoints'], plan['checkpoints']):
            assert checkpoint['samples_per_operation'] == n
            assert checkpoint['cpu'] > 0
            observed_prompts={r['prompt'] for r in plan['records'][:4*n]}
            taught_pool=[r for r in training if f'{r[1]}{r[0]}{r[2]}=' in observed_prompts]
            for field, pool in [('held',held),('taught',taught_pool)]:
                audit_answers(checkpoint[field], pool)
            for op in OPS:
                group = [r for r in checkpoint['held'] if r['operation']==op]
                assert checkpoint['scores'][op] == sum(r['correct'] for r in group)/len(group)
                actual_prompts={r['prompt'] for r in plan['records'][:4*n] if r['operation']==op}
                assert checkpoint['unique_seen_prompts'][op]==len(actual_prompts)
                labels=[bytes.fromhex(r['target_hex']) for r in history['ledger'][:4*n] if r['operation']==op]
                assert checkpoint['observed_answer_bytes'][op]==sorted(set(b''.join(labels)))
                assert checkpoint['countermodels']['exact'][op]==0
                assert checkpoint['countermodels']['inverse_family_lookup'][op]==0
        for block, declared in zip(history['blocks'],plan['blocks']):
            assert block['operation']==declared['operation'] and block['checkpoint']==declared['checkpoint']
            audit_answers(block['held'],held)
        audit_answers(history['final'],held)
        assert history['simulated_seconds']==86400
    for block in plan['blocks']:
        idx = block['record_indices']
        assert Counter(plan['records'][i]['answer'] for i in idx) == Counter(
            plan['shuffled_answers'][str(i)] for i in idx), 'shuffle marginal mismatch'
    assert receipt['records_applied']==3*len(plan['records'])
    return {'accepted': True, 'technical':receipt['technical'], 'world':receipt['world'],
            'config':receipt['config'], 'records_applied':receipt['records_applied'],
            'observed_event_rows':total_events, 'science_jobs':int(not receipt['technical'])}


def audit_answers(answers,pool):
    assert len(answers)==len(pool), 'missing or extra probe'
    for item,row in zip(answers,pool):
        op,a,b,y=row
        assert item['prompt_hex']==f'{a}{op}{b}='.encode().hex()
        assert item['target_hex']==str(y).encode().hex()
        assert item['operation']==op
        payload=bytes.fromhex(item['payload_hex'])
        assert len(payload)<=16
        assert item['status'] in ('ENDED','TRUNCATED')
        assert item['selected_events']==len(payload)+1
        assert item['correct']==(payload==str(y).encode() and item['status']=='ENDED')


if __name__=='__main__':
    with gzip.open(sys.argv[1],'rt') as f:
        r=json.load(f)
    print(json.dumps(audit(r,check_sources='--check-sources' in sys.argv)))
