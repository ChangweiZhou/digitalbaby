"""Independent receipt arithmetic and explicit branch/call auditor; never learns."""
import gzip
import hashlib
import json
import math
from pathlib import Path
from compact_fixture import make_world, DT, RECORD_SECONDS, BRANCHES
from compact_bridge import ALPHABET, SCALES


def argmax(v):
    if not v or not all(math.isfinite(x) for x in v): raise ValueError('bad output vector')
    return ALPHABET[max(range(len(v)), key=lambda i: v[i])]


def expected_write(branch, stage):
    # Deliberately does not call fixture.permitted(), nor use concatenated names.
    table = {'W': {'old': True, 'new': True, 'revision': True},
             'N_old': {'old': False, 'new': True, 'revision': True},
             'N_new': {'old': True, 'new': False, 'revision': True},
             'N_revision': {'old': True, 'new': True, 'revision': False}}
    return table[branch][stage]


def check_prediction(p, arm):
    private, shared, combined = p['private'], p['shared'], p['combined']
    if len(private) != 4 or len(shared) != (4 if arm == 'R_center_8' else 0) or len(combined) != 4:
        raise ValueError('prediction bank roster')
    reference = [private[i] / SCALES[1] + (shared[i] / SCALES[0] if shared else 0.) for i in range(4)]
    if any(abs(a - b) > 1e-12 for a, b in zip(reference, combined)) or p['emitted'] != argmax(combined):
        raise ValueError('wrong operative output')


def audit_receipt(doc, identity=None, complete=True):
    if identity is not None and doc['identity'] != identity: raise ValueError('wrong source')
    arm = doc['arm']; n = 8 if arm == 'R_center_8' else 4
    if arm not in ('R_center_8', 'CONTENT_4'): raise ValueError('wrong arm')
    world = make_world(doc['world'])
    if doc['fixture_sha256'] != world['sha256']: raise ValueError('wrong fixture')
    if len(doc['births']) != n or len({b['fly_id'] for b in doc['births']}) != n:
        raise ValueError('birth roster')
    from stores import EXPECTED_B
    if any(b['canonical_B_sha256'] != EXPECTED_B for b in doc['births']): raise ValueError('birth B')
    if complete and set(doc['branches']) != set(BRANCHES): raise ValueError('branch roster')
    counted = 0
    for branch, bdoc in doc['branches'].items():
        births = bdoc['births']
        if (len(births) != n or len({b['fly_id'] for b in births}) != n
                or any(b['canonical_B_sha256'] != EXPECTED_B for b in births)):
            raise ValueError('branch birth roster/canonical mismatch')
        rows = bdoc['records']
        if complete and len(rows) != 864: raise ValueError('record count')
        for i, row in enumerate(rows):
            event = world['events'][i]; flag = expected_write(branch, event['stage'])
            if row['index'] != i or row['stage'] != event['stage'] or row['item'] != event['item'] or row['learn'] is not flag:
                raise ValueError('branch/stage/index clamp mismatch')
            slot = event['at'] + 12 * DT
            if row['predicted_at'] != slot or row['observed_at'] != slot or row['prediction_precedes_outcome'] is not True:
                raise ValueError('feedback leakage/order')
            check_prediction(row['prediction'], arm)
            w = row['write']
            signs = [.25 - float(a == event['outcome']) for a in ALPHABET]
            if w['learn'] is not flag or w['outcome'] != event['outcome'] or w['s'] != signs or w['c'] != 0. or w['time'] != slot:
                raise ValueError('teacher/permission mismatch')
            calls = row['actual_calls']
            if len(calls) != n or [c['store'] for c in calls] != list(range(n)):
                raise ValueError('actual store call roster')
            offset = n - 4
            for j, call in enumerate(calls):
                private = j >= offset
                coef = [0., signs[j - offset]] if private else [float(ALPHABET[j] != event['outcome'])]
                if call['write'] is not flag or call['coefficients'] != coef or call['api'] != ('teach_signed' if private else 'teach_logged'):
                    raise ValueError('actual write call differs from contract')
                if not math.isfinite(call['applied_l1']) or call['applied_l1'] < 0: raise ValueError('nonfinite write')
                if not flag and (call['applied_l1'] != 0. or call['no_write_reference_equal'] is not True):
                    raise ValueError('no-write state/amount violated')
            applied = w['shared_l1'] + w['private_l1']
            if len(applied) != n or applied != [c['applied_l1'] for c in calls]: raise ValueError('write ledger disagreement')
            if not math.isfinite(row['flush_error']) or row['flush_error'] >= 1e-8: raise ValueError('clock residual')
            counted += n
        for pdoc in bdoc['probes']:
            if pdoc['at'] != world['clocks'][pdoc['name']]: raise ValueError('probe clock')
            expected = [('old', i) for i in range(32)] if pdoc['name'].startswith('old_') else (
                [('old', i) for i in range(32)] + [('new', i) for i in range(32)] + [('revision', i) for i in range(8)])
            if [(r['stage'], r['item']) for r in pdoc['rows']] != expected: raise ValueError('probe roster')
            for row in pdoc['rows']:
                check_prediction(row['prediction'], arm)
                target = world['sets'][row['stage']][row['item']]['outcome']
                if row['correct'] != int(row['prediction']['emitted'] == target): raise ValueError('probe score')
        if complete and [p['name'] for p in bdoc['probes']] != ['old_end', 'old_day', 'new_end', 'new_day', 'revision_end', 'final']:
            raise ValueError('checkpoint roster')
    return {'verdict': 'PASS', 'actual_store_calls': counted, 'branches': len(doc['branches'])}


def read(path):
    with gzip.open(path, 'rt') as f: doc = json.load(f)
    saved = doc.pop('receipt_sha256')
    actual = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if actual != saved: raise ValueError('receipt seal mismatch')
    return doc


def paired_private_equal(a, b):
    if a['world'] != b['world'] or a['fixture_sha256'] != b['fixture_sha256']: raise ValueError('unpaired world')
    for branch in BRANCHES:
        pa, pb = a['branches'][branch]['probes'], b['branches'][branch]['probes']
        for x, y in zip(pa, pb):
            if x['name'] != y['name'] or x['private_digest'] != y['private_digest']:
                raise ValueError('physical deletion changed private memory')
            if any(r['prediction']['private'] != s['prediction']['private'] for r, s in zip(x['rows'], y['rows'])):
                raise ValueError('private readout parity failed')
    return True
