"""Independent branch, outcome, update-algebra and first-output receipt audit."""
import gzip
import hashlib
import json
import math
from v2_fixture import make_world, DT

ALPHABET = b'0123'
SCALES = (1.4911274663291492, 1.3452365735750882)


def expect(branch, stage):
    # Literal table, independent from the runner permission function.
    return {'W': {'old': True, 'new': True, 'revision': True},
            'N_old': {'old': False, 'new': True, 'revision': True},
            'N_new': {'old': True, 'new': False, 'revision': True},
            'N_revision': {'old': True, 'new': True, 'revision': False}}[branch][stage]


def check_prediction(p):
    if any(len(p[k]) != 4 for k in ('shared', 'private', 'combined')): raise ValueError('output bank roster')
    u = [p['shared'][i] / SCALES[0] + p['private'][i] / SCALES[1] for i in range(4)]
    if not all(math.isfinite(x) for x in u) or any(abs(x-y) > 1e-12 for x,y in zip(u,p['combined'])):
        raise ValueError('output values')
    if p['emitted'] != ALPHABET[max(range(4), key=lambda i: u[i])]: raise ValueError('output is not model argmax')
    return u


def read(path):
    with gzip.open(path, 'rt') as f: doc = json.load(f)
    saved = doc.pop('receipt_sha256')
    if saved != hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest():
        raise ValueError('receipt seal mismatch')
    return doc


def audit_receipt(doc, identity=None, complete=True):
    if doc['schema'] != 'PERSISTENT_V2_RECEIPT' or (identity is not None and doc['identity'] != identity):
        raise ValueError('source/schema mismatch')
    arm = doc['arm']
    if arm not in ('V1', 'ERROR', 'REPLACE', 'BOTH'): raise ValueError('arm')
    world = make_world(doc['world'], doc['assay'])
    if doc['fixture_sha256'] != world['sha256']: raise ValueError('fixture hash mismatch')
    if complete and set(doc['branches']) != set(world['branches']): raise ValueError('missing branch')
    from stores import EXPECTED_B
    for branch, history in doc['branches'].items():
        births = history['births']
        if len(births) != 8 or len({b['fly_id'] for b in births}) != 8 or any(b['canonical_B_sha256'] != EXPECTED_B for b in births):
            raise ValueError('independent canonical births')
        if complete and len(history['records']) != len(world['events']): raise ValueError('record roster')
        for i, row in enumerate(history['records']):
            e = world['events'][i]; flag = expect(branch,e['stage']); slot = e['at'] + 12*DT
            if (row['index'],row['stage'],row['item'],row['learn']) != (i,e['stage'],e['item'],flag):
                raise ValueError('branch/stage write clamp')
            if row['prediction_precedes_outcome'] is not True or row['predicted_at'] != slot or row['observed_at'] != slot:
                raise ValueError('prediction/feedback order')
            check_prediction(row['prediction'])
            w = row['write']; z = [v / SCALES[1] for v in row['prediction']['private']]
            peak = max(z); exp = [math.exp(v-peak) for v in z]; pi = [v/sum(exp) for v in exp]
            signs = [(pi[j] if arm in ('ERROR','BOTH') else .25) - float(b==e['outcome']) for j,b in enumerate(ALPHABET)]
            if (w['outcome'],w['time'],w['learn'],w['c']) != (e['outcome'],slot,flag,0.): raise ValueError('arrived teacher/clamp')
            if any(abs(a-b)>1e-12 for a,b in zip(pi,w['private_pre_outcome_probabilities'])) or len(w['s']) != 4 or any(abs(a-b)>1e-12 for a,b in zip(signs,w['s'])):
                raise ValueError('private prediction error coefficients')
            calls = row['actual_calls']
            if len(calls)!=8 or [c['store'] for c in calls]!=list(range(8)): raise ValueError('actual calls roster')
            for j,c in enumerate(calls):
                coef = [float(ALPHABET[j]!=e['outcome'])] if j<4 else [0., signs[j-4]]
                api = 'teach_logged' if j<4 else 'teach_signed'
                if c['write'] is not flag or c['api']!=api or len(c['coefficients'])!=len(coef) or any(abs(a-b)>1e-12 for a,b in zip(c['coefficients'],coef)):
                    raise ValueError('actual call permission/coefficient')
                if not math.isfinite(c['applied_l1']) or c['applied_l1']<0: raise ValueError('write amount')
                if not flag and (c['applied_l1']!=0 or not c['no_write_reference_equal']): raise ValueError('no-write transition')
            if w['shared_l1']+w['private_l1'] != [c['applied_l1'] for c in calls]: raise ValueError('call/ledger mismatch')
            if len(w['replacements'])!=4: raise ValueError('replacement roster')
            for rep in w['replacements']:
                if rep['enabled'] is not (flag and arm in ('REPLACE','BOTH')): raise ValueError('replacement permission')
                n = rep['changed']
                if len(set(rep['ids']))!=n or any(len(rep[k])!=n for k in ('ids','old','delta','after','evidence')):
                    raise ValueError('replacement coordinates')
                if not rep['outside_conflict_unchanged'] or rep['certificate_error']>1e-12: raise ValueError('replacement support')
                for old,delta,after,ev in zip(rep['old'],rep['delta'],rep['after'],rep['evidence']):
                    if old*delta>=0 or ev!=3 or abs(after-(.5*old+delta))>1e-12: raise ValueError('replacement algebra/evidence')
                if abs(rep['removed_l1']-.5*sum(abs(v) for v in rep['old']))>1e-10: raise ValueError('replacement dose')
                if not rep['enabled'] and (n or rep['removed_l1']): raise ValueError('disabled replacement wrote')
            if not flag and w['evidence_before']!=w['evidence_after']: raise ValueError('disabled conflict evidence update')
            if not math.isfinite(row['flush_error']) or row['flush_error']>=1e-8: raise ValueError('clock')
        names = [p['name'] for p in history['probes']]
        expected_names = [n for ns in world['boundaries'].values() for n in ns]
        if complete and names!=expected_names: raise ValueError('probe checkpoints')
        for p in history['probes']:
            if p['at']!=world['clocks'][p['name']] or p['continuing_state_before']!=p['continuing_state_after']:
                raise ValueError('probe time/state mutation')
            expected_roster=[]
            if doc['assay']=='lifetime':
                for stage in ('old','new','revision'):
                    if p['name'].startswith('old_') and stage!='old': continue
                    expected_roster += [(stage,i) for i in range(len(world['sets'][stage]))]
                if [(r['stage'],r['item']) for r in p['rows']]!=expected_roster: raise ValueError('fact probe roster')
                for row in p['rows']:
                    check_prediction(row['prediction']); target=world['sets'][row['stage']][row['item']]['outcome']
                    if row['correct']!=int(row['prediction']['emitted']==target): raise ValueError('fact score')
            else:
                for stage in ('old','heldout','new'):
                    if p['name'].startswith('old_') and stage=='new': continue
                    expected_roster += [(stage,i,o) for i in range(len(world['sets'][stage])//2) for o in (0,1)]
                if [(r['stage'],r['pair'],r['order']) for r in p['rows']]!=expected_roster: raise ValueError('relation probe roster')
                for row in p['rows']:
                    group=world['sets'][row['stage']][2*row['pair']:2*row['pair']+2]
                    if row['order']: group=group[::-1]
                    if [row['first_hex'],row['second_hex']]!=[r['cue_hex'] for r in group]: raise ValueError('option identities')
                    us=[check_prediction(v) for v in row['predictions']]; values=[u[1]-u[0] for u in us]
                    target=ord('L') if row['order']==0 else ord('R')
                    emitted=ord('L') if values[0]>=values[1] else ord('R')
                    if any(abs(a-b)>1e-12 for a,b in zip(values,row['values'])) or (row['emitted'],row['target'],row['correct'])!=(emitted,target,int(emitted==target)):
                        raise ValueError('internal choice/score')
                    if row['response_at']!=p['at']+25*DT: raise ValueError('choice timing')
    if not math.isfinite(doc['worker_s']) or doc['worker_s']<=0: raise ValueError('worker resource accounting')
    return {'verdict':'PASS','branches':len(doc['branches'])}
