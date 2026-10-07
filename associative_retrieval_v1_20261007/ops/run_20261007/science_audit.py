"""Strict science-envelope audit plus independent frozen address/write replay."""
import copy
import math
import bootstrap_ops
from assays import make_world, RECORD_SECONDS
from independent_audit import audit as address_audit
from contracts import require_scope, CONDITIONS, RSS_CAP

def audit_science(d, lock, *, branch_only=False):
    qualification = d.get('qualification') is True
    require_scope(d['world'], d['assay'], d['stage'], qualification)
    if d.get('schema') != 'LINK_SCIENCE_V1' or d.get('development') is not False:
        raise AssertionError('explicit science envelope required')
    if d['source_identity'] != lock['native_identity'] or d['operations_identity'] != lock['operations_identity']:
        raise AssertionError('source lock mismatch')
    if not qualification and d.get('complete_branch') is not True:
        raise AssertionError('partial science is not a committed life')
    w = make_world(d['world'], d['assay'])
    limit = d['limit']
    expected_branches = [d['branch']] if branch_only else w['branches']
    if qualification:
        expected_branches = [d['branch']]
    elif limit != len(w['events']):
        raise AssertionError('science record limit differs from frozen complete life')
    if set(d['branches']) != set(expected_branches):
        raise AssertionError('exact branch roster mismatch')
    if d['fixture_sha256'] != w['sha256']:
        raise AssertionError('fixture hash mismatch')
    # The inherited auditor is development-enveloped. Only a validated in-memory
    # view changes that label; on-disk science remains explicitly science.
    view = copy.copy(d)
    view['development'] = True
    view['complete'] = not branch_only and not qualification
    result = address_audit(view, lock['native_identity'])
    for branch, bd in d['branches'].items():
        births = bd['births']
        if any(b['canonical_B_sha256'] != births[0]['canonical_B_sha256'] for b in births):
            raise AssertionError('inconsistent canonical birth weights')
        if births[0]['canonical_B_sha256'] != '32a3726cb25a8255c1aef5696ff40aa303138c5d061a598004d2087d4da5f964':
            raise AssertionError('wrong canonical birth object')
        if len(bd['records']) != limit:
            raise AssertionError('missing whole records')
        expected = []
        if qualification:
            expected.append(('short_end', limit, bd['final_time']))
        else:
            for i in range(1, limit + 1):
                expected.extend((name, i, w['clocks'][name]) for name in w['boundaries'].get(str(i), []))
        if [(p['name'], p['after_record'], p['at']) for p in bd['probes']] != expected:
            raise AssertionError('checkpoint clock/roster mismatch')
        for p in bd['probes']:
            expected_rows = []
            for group in ('old', 'heldout', 'new', 'revision'):
                if group not in w['sets'] or (p['name'].startswith('old_') and group in ('new', 'revision')):
                    continue
                rows = w['sets'][group]
                if d['assay'] == 'reuse':
                    for i in range(len(rows)//2):
                        for order in (0, 1):
                            expected_rows.append((group, i, order, ord('L') if order == 0 else ord('R')))
                    actual = [(r['stage'], r['pair'], r['order'], r['target']) for r in p['rows'] if r['stage'] == group]
                else:
                    actual = [(r['stage'], r['item'], r['target']) for r in p['rows'] if r['stage'] == group]
                    expected_rows.extend((group, i, r['outcome']) for i, r in enumerate(rows))
            if d['assay'] == 'reuse':
                actual = [(r['stage'], r['pair'], r['order'], r['target']) for r in p['rows']]
            else:
                actual = [(r['stage'], r['item'], r['target']) for r in p['rows']]
            if actual != expected_rows:
                raise AssertionError('probe targets or complete row roster differs from fixture')
            if not math.isfinite(p['physical_probe_cpu_s']) or p['physical_probe_cpu_s'] <= 0:
                raise AssertionError('invalid query cost')
        final_clock = (w['clocks']['final'] if not qualification
                       else w['events'][limit-1]['at'] + RECORD_SECONDS)
        if bd['final_time'] != final_clock:
            raise AssertionError('native learner final clock')
        for r in bd['records']:
            if r['prediction_precedes_outcome'] is not True:
                raise AssertionError('answer exposure before prediction')
            for name in CONDITIONS:
                if name not in r['prediction']['policies']:
                    raise AssertionError('missing readout condition')
            for k in ('parent_online_cpu_s', 'LINK_online_cpu_s', 'PERM_online_cpu_s',
                      'physical_online_cpu_s', 'external_audit_cpu_s'):
                if not math.isfinite(r[k]) or r[k] < 0:
                    raise AssertionError('invalid charged event CPU')
    if any(not math.isfinite(d[k]) or d[k] <= 0 for k in ('cpu_s', 'worker_s', 'peak_rss_bytes')):
        raise AssertionError('invalid physical resources')
    if d['peak_rss_bytes'] > RSS_CAP:
        raise AssertionError('worker exceeded fixed 1 GiB RSS cap')
    return {**result, 'world': d['world'], 'assay': d['assay'],
            'scope': 'qualification' if qualification else 'science',
            'full_job': not branch_only and not qualification}
