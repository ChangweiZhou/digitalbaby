"""Independent receipt/array/permission audit. No learner imports or model calls."""
import ast
import collections
import hashlib
import json
import math
import random
from functools import lru_cache
import numpy as np
from protocol import ROOT, D, plans, probes, seed, show
from independent_truth import independent_answer, independent_group


def audit(r, source=None):
    assert r['schema'] == 'GATED_FEEDBACK_TECHNICAL_JOB_V2' and r['technical'] is True and r['completed'] is True
    assert r['model'] in ('KEEP', 'G62') and r['model_seed'] == seed(r['world'], 0, 'model')
    admitted = [x for key, x in D['technical'].items() if isinstance(x, dict) and r['world'] in x['worlds']]
    assert len(admitted) == 1
    config = admitted[0]
    assert (r['binary'], r['mixed'], r['stages']) == (config['binary'], config['mixed'], config['stages'])
    if source is not None:
        assert r['source'] == source
    assert set(r['prefixes']) == {'TRUE', 'N_WARM', 'SH_WARM'}
    assert set(r['branches']) == set(D['histories'])
    assert sum(r['parameters'].values()) == D['models'][r['model']]['parameters']
    assert r['learning_bytes'] == D['models'][r['model']]['parameter_and_RMS_bytes']
    w, warm, shuffled, tails = plans(r['world'], r['binary'], r['mixed'], r['stages'])
    assert w['data_digest'] == r['data_digest']
    truth = lru_cache(None)(lambda p: independent_answer(p, r['world']))
    group = lru_cache(None)(lambda p: independent_group(p, r['world']))
    pi = list(range(256))
    random.Random(r['world'] + 918273).shuffle(pi)
    inverse = {b: i for i, b in enumerate(pi)}
    @lru_cache(None)
    def operation(prompt):
        t = ast.parse(bytes(inverse[b] for b in bytes.fromhex(prompt)).decode()[:-1], mode='eval').body
        names = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}
        inner = t.left if isinstance(t.left, ast.BinOp) else t.right if isinstance(t.right, ast.BinOp) else None
        return names[type(t.op)] if inner is None else names[type(inner.op)] + names[type(t.op)]
    trained = {x['group'] for x in warm + sum(list(tails.values()), [])}
    total = actor_updates = critic_updates = answer_bytes = checked_queries = candidate_evaluations = 0

    def prediction_tables(records):
        tables = {k: collections.defaultdict(collections.Counter) for k in ('exact', 'canonical', 'length', 'suffix', 'bag')}
        glob = collections.Counter()
        for row in records:
            a = row['observed'] if row['observed'] is not None else row['actual'] if row['reward'] == 1 else None
            if a is None:
                continue
            p = bytes.fromhex(row['prompt'])
            keys = (row['prompt'], row['group'], len(p), p[-2:], tuple(sorted(p)))
            glob[a] += 1
            for k, key in zip(tables, keys):
                tables[k][key][a] += 1
        return tables, glob

    def predict(tables, glob, row):
        def mode(counter):
            return min(counter, key=lambda a: (-counter[a], a)) if counter else None
        p = bytes.fromhex(row['prompt'])
        keys = (row['prompt'], row['group'], len(p), p[-2:], tuple(sorted(p)))
        a = {k: mode(tables[k].get(key, {})) for k, key in zip(tables, keys)}
        a['global'] = mode(glob)
        a['length'] = a['length'] or a['global']
        a['suffix'] = a['suffix'] or a['global']
        a['bag'] = a['bag'] or a['length']
        return a

    def scores(entry, feedback, taught=False):
        nonlocal checked_queries
        assert entry['before'] == entry['after']
        tables, glob = prediction_tables(feedback)
        exposed = {x['prompt']: dict(prompt=x['prompt'], reply=x['expected'], operation=x['operation'], group=x['group']) for x in feedback}
        for label in ('binary', 'mixed'):
            wanted = probes([x for x in exposed.values() if len(x['operation']) == (1 if label == 'binary' else 2)], None) if taught else probes(w[label + '_held'], None)
            assert [(x['prompt'], x['group'], x['operation']) for x in entry['scores'][label]] == [(x['prompt'], x['group'], x['operation']) for x in wanted]
            for row in entry['scores'][label]:
                assert row['reply'] == truth(row['prompt']) and row['group'] == group(row['prompt'])
                assert row['operation'] == operation(row['prompt'])
                assert row['status'] in ('EOS', 'TRUNCATED') and len(bytes.fromhex(row['actual'])) <= 16
                assert row['exact'] is (row['actual'] == row['reply'] and row['status'] == 'EOS')
                assert row['shortcuts'] == predict(tables, glob, row)
                if not taught:
                    assert row['group'] not in trained
                checked_queries += 1

    def segment(seg, plan, phase, kind, recipe, allowed, offset, feedback):
        nonlocal total, actor_updates, critic_updates, answer_bytes, candidate_evaluations
        assert (seg['phase'], seg['kind'], seg['recipe'], seg['offset']) == (phase, kind, recipe, offset)
        assert seg['allowed'] is allowed and len(seg['records']) == len(plan)
        if not allowed:
            assert seg['before'] == seg['after']
        checkpoints = {x['n']: x for x in seg['curve']}
        wanted_checks = ({offset + len(plan) * i // 4 for i in range(1, 5)} if phase == 'WARM'
                         else {offset + len(plan) // 2, offset + len(plan)} if phase == 'SCALAR'
                         else {offset + len(plan)})
        assert set(checkpoints) == wanted_checks and len(checkpoints) == len(seg['curve'])
        assert seg['curve'][-1]['before'] == seg['curve'][-1]['after'] == seg['after']
        updates = candidates = seen = 0
        for i, (row, wanted) in enumerate(zip(seg['records'], plan)):
            assert row['index'] == offset + i and row['allowed'] is allowed
            assert (row['phase'], row['recipe']) == (phase, recipe)
            assert row['prompt'] == wanted['prompt'] and row['group'] == group(row['prompt']) == wanted['group']
            assert row['operation'] == operation(row['prompt']) == wanted['operation']
            assert row['expected'] == truth(row['prompt'])
            obs = wanted['reply'] if recipe == 'CE_ONLY' or show(r['world'], offset + i, phase) else None
            assert row['observed'] == obs
            assert type(row['reward']) is int and row['reward'] == int(row['actual'] == row['expected'] and row['status'] == 'EOS')
            assert math.isfinite(row['baseline']) and 0 <= row['baseline'] <= 1
            assert row['status'] in ('EOS', 'TRUNCATED') and 1 <= len(row['actions']) <= 17
            assert all(type(x) is int and 0 <= x <= 256 for x in row['actions'])
            if row['status'] == 'EOS':
                assert row['actions'][-1] == 256 and row['actions'][:-1] == list(bytes.fromhex(row['actual']))
            else:
                assert len(bytes.fromhex(row['actual'])) == 16 and len(row['actions']) == 17
                assert row['actions'][:16] == list(bytes.fromhex(row['actual'])) and row['actions'][-1] != 256
            u = row['update']
            assert u['critic_steps'] == int(allowed)
            assert (u['ce'] is not None) == (allowed and obs is not None)
            assert (u['pg'] is not None) == (allowed and phase != 'WARM' and recipe != 'CE_ONLY')
            assert (u['critic'] is not None) == allowed
            for key in ('ce', 'pg', 'critic'):
                assert u[key] is None or math.isfinite(u[key])
            is_safe = allowed and recipe == 'SAFE' and phase != 'WARM'
            assert u['safe'] is is_safe
            if is_safe:
                trials = u['candidates']
                assert 1 <= len(trials) <= 8 and type(u['accepted']) is bool
                assert math.isclose(u['old_actor_loss'], (u['ce'] or 0.) + (u['pg'] or 0.), abs_tol=1e-10, rel_tol=1e-12)
                for j, trial in enumerate(trials):
                    assert trial['alpha'] == 2. ** -j and type(trial['accepted']) is bool
                    kl, value = trial['kl'], trial['loss']
                    valid = (kl is not None and value is not None and math.isfinite(kl) and math.isfinite(value)
                             and -1e-12 <= kl <= .01 + 1e-12
                             and value <= u['old_actor_loss'] + 1e-12 * (1 + abs(u['old_actor_loss'])))
                    assert trial['accepted'] is valid
                    assert trial['accepted'] is (u['accepted'] and j == len(trials) - 1)
                assert u['actor_steps'] == int(u['accepted'])
                assert u['alpha'] == (trials[-1]['alpha'] if u['accepted'] else None)
                if not u['accepted']:
                    assert len(trials) == 8
                candidates += len(trials)
            else:
                assert u['actor_steps'] == int(allowed) and u['candidates'] == []
                assert u['accepted'] is (True if allowed else None)
                assert u['alpha'] == (1. if allowed else None)
            if phase == 'SCALAR' and recipe != 'CE_ONLY':
                assert row['observed'] is None and u['ce'] is None
            total += 1
            updates += u['actor_steps']
            actor_updates += u['actor_steps']
            critic_updates += u['critic_steps']
            seen += len(bytes.fromhex(obs)) if obs is not None else 0
            feedback.append(row)
            if offset + i + 1 in checkpoints:
                scores(checkpoints[offset + i + 1], feedback)
        a, b = seg['resources_before'], seg['resources_after']
        assert b['records'] - a['records'] == b['scalar_feedback'] - a['scalar_feedback'] == len(plan)
        assert b['actor_steps'] - a['actor_steps'] == updates
        assert b['critic_steps'] - a['critic_steps'] == int(allowed) * len(plan)
        assert b['candidate_evaluations'] - a['candidate_evaluations'] == candidates
        assert b['observed_answer_bytes'] - a['observed_answer_bytes'] == seen
        answer_bytes += seen
        candidate_evaluations += candidates

    prefixes, fw = {}, r['binary'] + r['mixed']
    for name, plan, allowed in (('TRUE', warm, True), ('N_WARM', warm, False), ('SH_WARM', shuffled, True)):
        x = r['prefixes'][name]
        assert x['segment']['before'] == r['initial']
        assert x['curve'][0]['n'] == 0 and x['curve'][0]['before'] == r['initial']
        fb = []
        scores(x['curve'][0], fb)
        segment(x['segment'], plan, 'WARM', name, 'OLD', allowed, 0, fb)
        assert x['curve'][1:] == x['segment']['curve'] and x['state'] == x['segment']['after']
        scores(x['taught'], fb, True)
        assert x['taught']['before'] == x['state']
        prefixes[name] = fb
        if not allowed:
            assert x['state'] == r['initial']
            base = x['curve'][0]['scores']
            for c in x['curve']:
                assert {k: [(a['actual'], a['status']) for a in rows] for k, rows in c['scores'].items()} == {k: [(a['actual'], a['status']) for a in rows] for k, rows in base.items()}
    roster = {'CE_ONLY': ('TRUE', 'CE_ONLY'), 'OLD': ('TRUE', 'OLD'), 'SAFE': ('TRUE', 'SAFE'),
              'N_WARM_SAFE': ('N_WARM', 'SAFE'), 'SH_WARM_SAFE': ('SH_WARM', 'SAFE')}
    for name, branch in r['branches'].items():
        if name.startswith('FREEZE_'):
            parent = name[len('FREEZE_'):]
            original = r['branches'][parent]
            fb = list(prefixes['TRUE']) + sum((s['records'] for s in original['segments'][:2]), [])
            curve = original['curve'][:-2]
            assert branch['curve_parent'] == parent and branch['recipe'] == parent
            assert branch['fork_state'] == original['segments'][-1]['before'] == branch['final']
            segment(branch['segments'][0], tails['SCALAR'], 'SCALAR', name, parent,
                    False, fw + sum(r['stages'][:2]), fb)
            assert branch['curve'] == curve + branch['segments'][0]['curve']
            base = curve[-1]['scores']
            for c in branch['segments'][0]['curve']:
                assert {k: [(a['actual'], a['status']) for a in rows] for k, rows in c['scores'].items()} == {k: [(a['actual'], a['status']) for a in rows] for k, rows in base.items()}
        else:
            parent, recipe = roster[name]
            assert branch['prefix'] == parent and branch['recipe'] == recipe
            fb, curve, offset = list(prefixes[parent]), list(r['prefixes'][parent]['curve']), fw
            before = r['prefixes'][parent]['state']
            for phase, seg in zip(('HALF', 'SPARSE', 'SCALAR'), branch['segments']):
                assert seg['before'] == before
                segment(seg, tails[phase], phase, name, recipe, True, offset, fb)
                before = seg['after']
                offset += len(tails[phase])
                curve.extend(seg['curve'])
            assert branch['curve'] == curve and branch['final'] == before
        assert all(a['n'] < b['n'] and a['cpu'] <= b['cpu'] for a, b in zip(branch['curve'], branch['curve'][1:]))
        scores(branch['taught'], fb, True)
        assert branch['taught']['before'] == branch['final']
    assert total == r['executed_records'] == 3 * fw + 5 * sum(r['stages'][:2]) + 7 * r['stages'][-1]
    assert r['correlated_final_histories'] == 7 and len(r['archives']) == 16
    assert all(math.isfinite(r[k]) and r[k] >= 0 for k in ('wall', 'total_cpu', 'peak_rss_bytes'))
    expected = {'BIRTH.npz': r['initial']}
    expected.update({k + '_WARM.npz': x['state'] for k, x in r['prefixes'].items()})
    expected.update({k + '_SCALAR_START.npz': r['branches'][k]['segments'][-1]['before'] for k in roster})
    expected.update({k + '_FINAL.npz': x['final'] for k, x in r['branches'].items()})
    assert set(r['archives']) == set(expected)
    for name, relative in r['archives'].items():
        with np.load(ROOT / relative, allow_pickle=False) as arrays:
            meta = json.loads(str(arrays['metadata']))
            assert meta['model'] == r['model']
            h = hashlib.sha256()
            for label, prefix in (('parameter', 'p_'), ('moment', 'm_')):
                keys = sorted(k[len(prefix):] for k in arrays.files if k.startswith(prefix))
                assert set(keys) == set(r['parameters'])
                for k in keys:
                    a = arrays[prefix + k]
                    assert a.dtype == np.float64 and np.isfinite(a).all() and a.size == r['parameters'][k]
                    if prefix == 'm_':
                        assert (a >= 0).all()
                    h.update((label + k).encode())
                    h.update(a.tobytes())
            h.update(json.dumps([meta['actor_steps'], meta['critic_steps'], meta['last']], sort_keys=True, allow_nan=False).encode())
            assert h.hexdigest() == meta['digest'] == expected[name]
            assert ('p_C' in arrays.files) is (r['model'] == 'KEEP')
    return dict(accepted=True, world=r['world'], model=r['model'], records=total,
                actor_updates=actor_updates, critic_updates=critic_updates, answer_payload_bytes=answer_bytes,
                checked_query_rows=checked_queries, checked_archives=16, related_histories=7,
                candidate_evaluations=candidate_evaluations,
                scope='Independent labels/groups/permissions/feedback/trace-constraint ledger, state arrays and counts; not a learned-rule proof')
