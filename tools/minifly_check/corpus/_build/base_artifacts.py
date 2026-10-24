"""Executor-side artifacts of the corpus base: OUTPUTS rows, receipts, FIXTURES, ADMISSION, SOURCE_LOCK,
ATTEMPTS and CYCLES rows, REVIEW files and (committed, hand-written) gate reports."""
import json
import os

import base_items as BI
from base_files import D
from mrk_util import EMPTY_SHA256, dump_json, hexs, sha256_bytes, sha256_json

HERE = os.path.dirname(os.path.abspath(__file__))
RULES_PATH = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'RULES.json')
CHANNELS = ['pair_memory', 'readout']


def att_id(utc, label):
    return 'ATT-%s-%s' % (utc, sha256_bytes(label.encode('utf-8'))[:6])


def output_rows(world, role, arms, cells, attempt_id, ncs, psha, bh, row_patch=None):
    items = BI.pool_items()
    rows = []
    for arm in arms:
        evaluator = arm.startswith('CEILING_') or arm.startswith('FLOOR_')
        for pool, ckpt in cells:
            for (iid, a, b) in items[pool]:
                em = BI.emitted_for(arm, world, pool, ckpt, iid, a, b, bh)
                stream = BI.stream_of(em)
                cut = stream.find(BI.EOS)
                row = {'world': world, 'world_role': role, 'arm': arm, 'pool': pool,
                       'checkpoint': ckpt, 'item_id': iid, 'prompt_hex': hexs(BI.prompt(a, b)),
                       'expected_hex': hexs(BI.truth(a, b)), 'stream_hex': hexs(stream),
                       'emitted_hex': hexs(stream if cut < 0 else stream[:cut]), 'own_eos': cut >= 0,
                       'answer_path': 'evaluator' if evaluator else 'native',
                       'native_closure_sha': None if evaluator else ncs,
                       'params_sha256': None if evaluator else psha, 'attempt_id': attempt_id}
                if row_patch is not None:
                    row = row_patch(row)
                if row is not None:
                    rows.append(row)
    return rows


def _dg(*parts):
    return sha256_bytes('|'.join(parts).encode('utf-8'))


def receipt(world, role, arm, attempt_id, ncs, psha, lock_sha):
    birth = {c: _dg('birth', world, c) for c in CHANNELS}
    w_pf = {c: _dg('post_first', world, 'W', c) for c in CHANNELS}
    if arm == 'W':
        pf, end, writes = dict(w_pf), {c: _dg('end', world, arm, c) for c in CHANNELS}, {
            'pair_memory': 150, 'readout': 160}
        dose = {'teaching_presented_fraction': 1.0}
    elif arm == 'NPT':
        pf = {'pair_memory': birth['pair_memory'], 'readout': w_pf['readout']}
        end = {'pair_memory': birth['pair_memory'], 'readout': _dg('end', world, arm, 'readout')}
        writes = {'pair_memory': 0, 'readout': 160}
        dose = {'teaching_withheld_fraction': 1.0}
    elif arm == 'SHUF':
        pf, end = dict(w_pf), {c: _dg('end', world, arm, c) for c in CHANNELS}
        writes = {'pair_memory': 150, 'readout': 160}
        dose = {'label_permuted_fraction': 0.5333}
    elif arm == 'FROZEN':
        pf, end, writes = dict(birth), dict(birth), {'pair_memory': 0, 'readout': 0}
        dose = {'frozen_fraction': 1.0}
    elif arm == 'ORACLE':
        pf = {'pair_memory': w_pf['pair_memory'], 'readout': _dg('post_first', world, arm, 'readout')}
        end = {c: _dg('end', world, arm, c) for c in CHANNELS}
        writes = {'pair_memory': 150, 'readout': 160}
        dose = {'oracle_feature_fraction': 1.0}
    else:
        raise KeyError(arm)
    return {'schema': 'minifly.receipt.v1', 'world': world, 'world_role': role, 'arm': arm,
            'phase': 'teach+test', 'records': 160,
            'taught_presentations': 0 if arm == 'NPT' else 150,
            'writes_by_channel': writes, 'write_counter_site': 'learner.py:Learner.write_pair',
            'channel_digests': {'birth': birth, 'post_first': pf, 'end': end},
            'state_digest_end': _dg('state_end', world, arm), 'realized_dose': dose,
            'native_closure_sha': ncs, 'params_sha256': psha, 'endpoint_lock_sha256': lock_sha,
            'attempt_id': attempt_id}


def fixtures_rows(params, attempt_id):
    p = dict(params)
    return [
        {'port': 'pair_write', 'sign': '+', 'target_margin_delta_vs_passive': 0.031,
         'zero_coef_digest_equals_passive': True, 'zero_replay_applications': 0, 'params': p,
         'attempt_id': attempt_id},
        {'port': 'pair_write', 'sign': '-', 'target_margin_delta_vs_passive': 0.027,
         'zero_coef_digest_equals_passive': True, 'zero_replay_applications': 0, 'params': p,
         'attempt_id': attempt_id},
        {'gate_id': 'stop_gate', 'kind': 'negative', 'params': p, 'result': False,
         'attempt_id': attempt_id},
        {'gate_id': 'stop_gate', 'kind': 'positive', 'params': p, 'result': True,
         'attempt_id': attempt_id},
    ]


def admission(lock_sha, sl_sha, admit_sha):
    return {'schema': 'minifly.admission.v1', 'endpoint_lock_sha256': lock_sha,
            'source_lock_sha256': sl_sha, 'producer': {'script': 'admit.py', 'sha256': admit_sha},
            'worlds': [BI.NATIVE_TECH_WORLD],
            'testbed_outputs': 'results/admission/OUTPUTS.jsonl.gz',
            'receipts_glob': 'results/receipts/%s/*.json' % BI.NATIVE_TECH_WORLD,
            'fixtures': 'results/admission/FIXTURES.jsonl',
            'input_contrast': {'metric_definition': 'L2 distance of post-read pair codes: same prompt '
                                                    'different noise vs different prompt same noise',
                               'analytic_snr': 1.7, 'same_input_diff_noise': 0.12,
                               'diff_input_same_noise': 0.41, 'ratio': 3.42,
                               'prompt_id_loo_acc': 0.31, 'chance': 0.0667},
            'structure_meters': {'shared_operand_similarity': 0.44,
                                 'nonshared_operand_similarity': 0.21, 'modal_output_fraction': 0.18,
                                 'code_overlap_birth': 0.02, 'code_overlap_end': 0.07},
            'parent_admission_sha': None, 'created_utc': '20261101T114500Z'}


def source_lock(lock_sha, files, built_utc='20261101T105500Z', native=('learner.py',),
                runner=('run.py',), admit='admit.py', params_path='params.json', delivery=D):
    pre = delivery + '/'
    closure = {}
    for rel in list(native) + list(runner) + [admit]:
        closure[pre + rel] = sha256_bytes(files[pre + rel])
    nat = {pre + r: closure[pre + r] for r in native}
    run = {pre + r: closure[pre + r] for r in list(runner) + list(native)}
    return {'schema': 'minifly.source_lock.v1', 'delivery': delivery, 'endpoint_lock_sha256': lock_sha,
            'built_utc': built_utc, 'checker_version': '1.0.0', 'closure_sha256': closure,
            'native_closure_sha256': nat, 'dynamic_imports': [],
            'env': {'python': '3.9.18', 'packages': {}}, 'native_closure_sha': sha256_json(nat),
            'runner_closure_sha': sha256_json(run),
            'params_sha256': sha256_bytes(files[pre + params_path])}


def attempt(aid, stage, argv, start, end, git_head, lock_sha, sl, produced, gate_report,
            cpu=120.0, delivery=D):
    return {'schema': 'minifly.attempt.v1', 'attempt_id': aid, 'stage': stage, 'argv': list(argv),
            'cwd': delivery, 'utc_start': start, 'utc_end': end, 'git_head': git_head,
            'closure_clean': True, 'closure_clean_after': True, 'endpoint_lock_sha256': lock_sha,
            'source_lock_sha256': sl['sha'] if sl else None,
            'native_closure_sha': sl['doc']['native_closure_sha'] if sl else None,
            'params_sha256': sl['doc']['params_sha256'] if sl else None, 'exit_code': 0,
            'wall_s': round(cpu * 1.05, 1), 'cpu_children_user_s': cpu, 'cpu_children_sys_s': 1.5,
            'max_rss_kb': 51200, 'produced_paths': [{'path': p, 'sha256': s} for p, s in produced],
            'removed_paths': [], 'gate_report': gate_report}


def cycle(k, kind, decision, reason, utc, sl, attempts, gate, outcomes, designs_k):
    return {'schema': 'minifly.cycle.v1', 'k': k, 'kind': kind, 'decision': decision, 'reason': reason,
            'utc': utc, 'computed': {
                'native_closure_sha': sl['doc']['native_closure_sha'] if sl else None,
                'params_sha256': sl['doc']['params_sha256'] if sl else None,
                'source_lock_sha256': sl['sha'] if sl else None, 'attempts': list(attempts),
                'latest_gate': gate, 'outcomes_readable': outcomes, 'designs_tried_k': designs_k}}


# ------------------------------------------------------------------------------------------------
# Gate reports. The fixture trees carry no tools/minifly_check, so the committed reports name
# checker_sha256 = sha256 of an empty kit listing (SCHEMAS 19.4 formula over zero files).

def _rules():
    with open(RULES_PATH, 'rb') as fh:
        return json.loads(fh.read().decode('utf-8'))


CTX = {'tier': 2, 'sealed': False}
NA_IN_BASE = {'LOCK-09', 'AMD-02', 'AMD-03', 'ADM-REUSE', 'AUT-04', 'DEBT-01'}


def bundle_ids(stage, sealed=None, tier=None):
    sealed = CTX['sealed'] if sealed is None else sealed
    tier = CTX['tier'] if tier is None else tier
    rules = _rules()
    b = rules['bundles'][stage]
    ids = bundle_ids(b['extends'], sealed, tier) if b.get('extends') else []
    for e in b['checks']:
        if isinstance(e, str):
            ids.append(e)
        elif 'group' in e:
            if e.get('when') != 't2' or tier == 2:
                ids.extend(e['primary'])
        else:
            when = e.get('when')
            if (when == 't2' and tier == 2) or (when in ('sealed_confirm', 'oc_intended', 'sealed_or_adoption') and sealed):
                ids.append(e['id'])
    out = []
    for i in ids:
        if i not in out:
            out.append(i)
    return out


def gate_report(stage, dry, utc, git_head, lock_sha, sl_sha, invoked_by, family_id,
                sealed=None, delivery=D):
    ids = bundle_ids(stage, sealed)
    results = []
    for i in ids:
        st = 'NOT_APPLICABLE' if i in NA_IN_BASE else 'PASS'
        results.append({'id': i, 'status': st, 'detail': '', 'evidence': []})
    counts = {}
    for r in results:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    return {'schema': 'minifly.gate_report.v1', 'stage': stage, 'dry': dry, 'delivery': delivery,
            'utc': utc, 'checker_sha256': EMPTY_SHA256, 'checker_version': '1.0.0',
            'git_mode': 'timeline', 'git_head': git_head, 'endpoint_lock_sha256': lock_sha,
            'source_lock_sha256': sl_sha, 'tier': CTX['tier'],
            'tier_reasons': ['declared: %d' % CTX['tier']],
            'family_id': family_id, 'family_members': [delivery],
            'advisory_basis': {'enforce_all': stage in ('official', 'publish', 'pointer'),
                               'first_source_lock_checker_version': '1.0.0'},
            'invoked_by': invoked_by, 'runtime_s': 0.5, 'counts': counts, 'results': results,
            'exit_code': 0, 'verdict': 'PASS'}


def gate_ref(path, data, stage, dry, utc, sl_sha):
    return {'path': path, 'stage': stage, 'dry': dry, 'verdict': 'PASS', 'utc': utc,
            'source_lock_sha256': sl_sha, 'sha256': sha256_bytes(data)}


# ------------------------------------------------------------------------------------------------
# Review (other family). Scripts are tiny stubs; outputs are small JSON files.

REVIEW_SCRIPTS = {
    'rerun_admission': 'review/rerun_admission.py',
    'reject_input': 'review/reject_input.py',
    'trace_number': 'review/trace_number.py',
    'shortcut_attempt': 'review/shortcut_attempt.py',
}


def review_script(kind):
    return ('"""Reviewer script (%s) for the corpus base; committed, never run by the checker."""\n'
            'import json\nimport sys\n\n\ndef main():\n    sys.stdout.write(json.dumps({"kind": "%s"}))\n'
            '    return 0\n\n\nif __name__ == "__main__":\n    sys.exit(main())\n' % (kind, kind)).encode('utf-8')


def review_output(kind, delivery=D):
    body = {'rerun_admission': {'world': BI.NATIVE_TECH_WORLD, 'arm': 'W', 'digest_match': 'yes',
                                'compared': 'results/receipts/w_tech_1/W.json'},
            'reject_input': {'input': 'OUTPUTS row with emitted_hex not derived from stream_hex',
                             'gate': 'MET-01', 'outcome': 'rejected'},
            'trace_number': {'number': 'admission input_contrast.ratio 3.42',
                             'producer': 'admit.py:measure_input_contrast', 'outcome': 'traced'},
            'shortcut_attempt': {'policy': 'DIGIT_NO_CARRY on HELD', 'held_rate': 0.5,
                                 'outcome': 'below the locked comparator; does not pass'}}[kind]
    body['delivery'] = delivery
    body['reviewer'] = 'claude:s-corpus-review'
    return dump_json(body)


def review_doc(lock_sha, adm_sha, files, prefix, acks=None, delivery=D, canary_report=None):
    checks = []
    for n, kind in enumerate(['rerun_admission', 'reject_input', 'trace_number', 'shortcut_attempt']):
        script = REVIEW_SCRIPTS[kind]
        out = 'review/out_%s.json' % kind
        checks.append({'id': 'X%d' % (n + 1), 'kind': kind, 'script': script,
                       'script_sha256': sha256_bytes(files[prefix + script]),
                       'cmd': 'python3 -I %s' % script, 'output': out,
                       'output_sha256': sha256_bytes(files[prefix + out]), 'cpu_cap_s': 60,
                       'result': {'rerun_admission': 'match', 'reject_input': 'rejected',
                                  'trace_number': 'traced', 'shortcut_attempt': 'not_passed'}[kind],
                       'compared_to': ('results/receipts/w_tech_1/W.json'
                                       if kind == 'rerun_admission' else None),
                       'match': True if kind == 'rerun_admission' else None})
    return {'schema': 'minifly.review.v1', 'delivery': delivery, 'reviewer_agent': 'claude',
            'reviewer_session': 's-corpus-review', 'reviewer_branch': 'claude/minifly-corpus-review',
            'author_agent': 'codex', 'reviewed_endpoint_lock_sha256': lock_sha,
            'admission_sha256': adm_sha, 'utc': '20261101T145000Z', 'executed_checks': checks,
            'canary': {'seed': '5eed01', 'n': 15, 'detected': 15,
                       'report': canary_report or 'checks/CANARY_20261101T143000Z.json'},
            'cheapest_shortcut_attempt': {'description': 'DIGIT_NO_CARRY substitution for W on the '
                                                         'technical world',
                                          'script': 'review/shortcut_attempt.py',
                                          'passed_locked_rule': False},
            'fm_judgments': {'FM-02': 'write_pair is the only mutation site; NPT writes 0 there.',
                             'FM-03': 'NPT cuts pair_memory only; readout equals W at post_first.',
                             'FM-12': 'No held orbit occurs in any taught stream.',
                             'FM-23': 'No parent; no priors added; answer_path native.'},
            'findings': [], 'source_lock_acks': list(acks or []), 'seal': None, 'postrun': None,
            'decision': 'ADMIT'}


def canary_report(delivery=D, seed='5eed01', n=15, detected=15, utc='20261101T143000Z', lock_sha=None,
                  sl_sha=None, git_head=None, family_tier=None):
    """A committed `MFC canary` report in the shape the canary command writes (minifly.canary_report.v1):
    counts, the lock shas it ran on, CAN-01..03 results, exit_code and verdict (REV-05 reads all of them)."""
    ops = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M09', 'M10', 'M11', 'M12', 'M13',
           'M14', 'M15', 'M17']
    results = [{'id': c, 'status': 'PASS', 'detail': '', 'evidence': []} for c in ('CAN-01', 'CAN-02', 'CAN-03')]
    return dump_json({'schema': 'minifly.canary_report.v1', 'delivery': delivery, 'seed': seed,
                      'utc': utc, 'n_requested': n, 'n': n, 'n_mutants': n, 'detected': detected,
                      'checker_sha256': EMPTY_SHA256, 'checker_version': '1.0.0', 'git_mode': 'timeline',
                      'git_head': git_head, 'endpoint_lock_sha256': lock_sha, 'source_lock_sha256': sl_sha,
                      'tier': CTX['tier'] if family_tier is None else family_tier,
                      'decisive_role': 'official', 'applicable_fraction': 1.0, 'operators': ops,
                      'mutants': [{'operator': o, 'applicable': True, 'caught': True} for o in ops],
                      'results': results, 'exit_code': 0, 'verdict': 'PASS'})