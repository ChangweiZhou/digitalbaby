"""Fixture table, part 2: C21-C30 and the special clean fixtures C00c (CANNOT_TEST), C00d (sealed
confirmation) and C00e (T1 dev only)."""
import hashlib
import hmac
import json

import base_artifacts as BA
import base_files as BF
import base_items as BI
import mutate as M
import scores_doc as SD
from base_timeline import FAMILY_ID, build_base
from fixtures_c import _set
from mrk_util import canonical_json, dump_compact, dump_json, dump_jsonl, gzip_bytes, sha256_bytes

D = BF.D
P = D + '/'


def b_c21():
    four = ['w_off_1', 'w_off_2', 'w_off_3', 'w_off_4']
    wrong = {'w_dev_1': ([0, 1, 2], [0, 1, 2, 3, 4, 5]), 'w_dev_2': ([0, 1], [0, 1, 2, 3, 4, 5, 6])}

    def lock(lk):
        lk['worlds']['official'] = {'mode': 'listed', 'ids': list(four), 'n': 4}
        lk['worlds']['official_role_intended'] = 'SEALED_CONFIRM'

    def claim(b):
        return M.claim_doc(b, worlds_role='SEALED_CONFIRM', n_worlds=4, primary_gate='PASS',
                           level_earned='E3')
    return build_base({'bi': {'OFF_WORLDS': four}, 'lock_patch': lock,
                       'behaviour': {'w_held_wrong': dict(BI.W_HELD_WRONG, **wrong)},
                       'claim_doc': claim})


def b_c22():
    first = 'PASS | G1 memory largely met on 2 DEV worlds'
    body = 'Native confirmation of pair reuse: the taught units carry over to held sums.\n'
    return b_c00e(report_first_line=first, report_body=body)


def b_c23():
    prior = 'minifly_corpus_prior_dev_v1_20261020'
    fast = (b'"""Corpus stub fast path (never executed)."""\n\n\ndef step(loss):\n'
            b'    loss.backward()\n    return loss\n')
    learner = BF.LEARNER_PY.encode()
    pl = {'schema': 'minifly.endpoint_lock.v1', 'delivery': prior, 'protocol_id': prior + '-p1',
          'author_agent': 'codex', 'lineage': {'parent': None, 'changes_declared': [], 'priors_added': []},
          'files': {'native': ['learner.py', 'fastpath.py'], 'runner': [], 'admit': None}}
    nat = {prior + '/learner.py': sha256_bytes(learner), prior + '/fastpath.py': sha256_bytes(fast)}
    ps = {'schema': 'minifly.source_lock.v1', 'delivery': prior, 'endpoint_lock_sha256': '0' * 64,
          'built_utc': '20261020T100000Z', 'checker_version': '1.0.0', 'closure_sha256': dict(nat),
          'native_closure_sha256': dict(nat), 'dynamic_imports': [],
          'env': {'python': '3.9.18', 'packages': {}}, 'native_closure_sha': '0' * 64,
          'runner_closure_sha': '0' * 64, 'params_sha256': '0' * 64}
    other = {prior + '/ENDPOINT_LOCK.json': dump_json(pl), prior + '/SOURCE_LOCK.json': dump_json(ps),
             prior + '/learner.py': learner, prior + '/fastpath.py': fast}

    def lock(lk):
        lk['files']['native'] = ['learner.py', 'fastpath.py']
    return build_base({'lock_patch': lock, 'extra_files': {'kb': other},
                       'extra_native': {P + 'fastpath.py': fast}})


def b_c24():
    def after_report(b):
        ptr = {'schema': 'minifly.pointer.v1', 'track': 'addition_e3', 'delivery': D,
               'claim': D + '/CLAIM.json', 'claim_sha256': '0' * 64, 'level_earned': 'E3',
               'primary_gate': 'PASS', 'gate_type': 'objective', 'worlds_role': 'MAIN',
               'objective_metric': M.claim_doc(b)['objective_metric'],
               'quoted_findings': ['F-UF-15'],
               'summary': 'F-UF-15: the STOP channel adds +22.10pp own-EOS exact on OLD.',
               'status': {'confirmation_passes': True},
               'source_note': 'numbers from /Users/zhou/minifly/results/official/SUMMARY.json',
               'narrowing_rows': [], 'updated_utc': '20261103T130000Z', 'previous_snapshot': None,
               'user_decision': None}
        b.commit('pointer', {'PROJECT_DOCUMENTS/CURRENT_ADDITION_E3_POINTER.json': dump_json(ptr)},
                 utc='20261103T130000Z')
    return build_base({'after': {'report': [after_report]}})


def b_c25():
    def lock(lk):
        lk['predictions'][0]['basis_findings'] = ['F-20261001-codex-9']
    return build_base({'lock_patch': lock, 'stop_after': 'lock'})


def b_c26():
    def claim(b):
        doc = M.claim_doc(b)
        doc['predictions_scored'] = [p for p in doc['predictions_scored'] if p['id'] != 'P3']
        return doc

    def incidents(b):
        rows = [{'id': 'INC-C26-1', 'kind': 'report', 'fm_class': 'FM-26', 'delivery': D,
                 'symptom': 'old_acq prediction dropped from the scored list', 'evidence': D + '/CLAIM.json',
                 'found_by_agent': 'codex', 'found_utc': '20261103T121000Z', 'caught_at_discovery': False,
                 'checker_sha_at_discovery': None, 'heldout_eligible': False, 'check_id': None,
                 'fixture_id': None, 'status': 'open'},
                {'id': 'INC-C26-1', 'kind': 'closure', 'fm_class': 'FM-26', 'delivery': D,
                 'symptom': 'closed by a lesson in REPORT.md', 'evidence': D + '/REPORT.md:3',
                 'found_by_agent': 'codex', 'found_utc': '20261103T122000Z', 'caught_at_discovery': False,
                 'checker_sha_at_discovery': None, 'heldout_eligible': False, 'check_id': 'CLM-10',
                 'fixture_id': None, 'status': 'closed'}]
        b.commit('incident', {BF.KB + 'INCIDENTS.jsonl': dump_jsonl(rows)}, utc='20261103T122000Z')
    return build_base({'claim_doc': claim, 'after': {'report': [incidents]}})


def b_c27():
    peer = 'minifly_corpus_peer_dev_v1_20261020'
    copied = BA.review_output('trace_number')

    def review(doc, version):
        doc['reviewer_agent'] = 'codex'
        doc['reviewer_session'] = 'codex-same-family'
        doc['reviewer_branch'] = 'codex/minifly-corpus-base'
        doc['executed_checks'] = [c for c in doc['executed_checks'] if c['kind'] == 'trace_number']
        doc['canary'] = {'seed': '5eed01', 'n': 15, 'detected': 15,
                         'report': 'checks/CANARY_20261101T143000Z.json'}
    return build_base({'review_patch': review, 'who': {'review': 'codex', 'ack': 'codex'},
                       'extra_files': {'kb': {peer + '/review/out_trace_number.json': copied}}})


def b_c28():
    def auth(a):
        a['caps']['max_records'] = 54432
        a['options_shown'][0]['label'] = 'A: open held E3 on single-digit addition (Recommended)'
        a['conditionals'] = [{'predicate': None, 'evaluated_by': 'codex', 'result': True,
                              'utc': '20261101T084500Z',
                              'note': 'proceed to official if dev looks good (judged by the executor)'}]
    return build_base({'auth_patch': auth, 'stop_after': 'lock'})


def b_c29():
    tl, b = build_base({'report_body': 'Full rows: /Users/zhou/minifly/%s/results/official/ '
                                       '(not committed).\n' % D})
    M.make_untracked(tl, P + 'results/official/OUTPUTS.jsonl.gz')
    return tl, b


def b_c30():
    def findings(b):
        rows = list(BF.FINDINGS_BASE)
        f1 = {'id': 'F-20261103-codex-1', 'kind': 'finding', 'refers_to': None,
              'claim': 'Pair-rectangle units compose to held single-digit sums.', 'level': 'E3',
              'conditions': {'substrate': 'corpus stub', 'input_code': 'raw bytes', 'priors_supplied': 'none',
                             'feedback': 'gradual_warmup', 'dose': 'full', 'comparator': 'NPT',
                             'best_shortcut': 'DIGIT_NO_CARRY', 'worlds': '8 MAIN',
                             'metric_definition': 'exact_own_eos'},
              'effect': {'value': 0.289, 'ci': [0.267, 0.31], 'unit': 'abs'},
              'source': D + '/REPORT.md:1@' + b.tl.head_sha()[:8], 'status': 'proposed', 'delivery': D,
              'author_agent': 'codex', 'utc': '20261103T121000Z', 'caveat': None}
        f2 = {'id': 'F-20261103-codex-2', 'kind': 'activation', 'refers_to': 'F-20261103-codex-1',
              'claim': 'activated', 'level': 'E3', 'conditions': f1['conditions'], 'effect': f1['effect'],
              'source': f1['source'], 'status': 'active', 'delivery': D, 'author_agent': 'codex',
              'utc': '20261103T121500Z', 'caveat': None}
        b.commit('activate', {BF.KB + 'FINDINGS.jsonl': dump_jsonl(rows + [f1, f2])},
                 utc='20261103T121500Z')
    return build_base({'after': {'report': [findings]}})


# ---------------------------------------------------------------------------------------------
# Special clean fixtures.

C00C_HEADLINE = ('CANNOT_TEST_CLAIM_AS_SPECIFIED | earned NONE (attempted E3) | held exact own-EOS '
                 'not_measured | 0 PILOT worlds | best of k=0')
C00E_HEADLINE = ('NOT_OPENED | earned NONE (attempted E3) | held exact own-EOS 84.4% [S2+ determined, '
                 'n=32] vs best shortcut DIGIT_NO_CARRY 50.0% | G 28.1pp | OLD final 100.0% (drop 0.0pp) '
                 '| 2 DEV worlds | best of k=1 | dev shadow FAIL')


def b_c00c():
    tl, b = build_base({'behaviour': {'ceiling_mode': 'fail'}, 'stop_after': 'testbed',
                        'cycle1': ('revise', 'Every ceiling fails on the locked split: amend the '
                                             'testbed (testbed_invalid).')})
    old = b.lock_sha
    lock = json.loads(b.files[P + 'ENDPOINT_LOCK.json'].decode())
    lock['protocol_id'] = D + '-p2'
    lock['testbed']['ceiling_grid'].append({'id': 'EXACT_BP_LONG', 'learner': 'EXACT_BP',
                                            'budget_multiplier': 4})
    lb = dump_json(lock)
    b.lock_sha = sha256_bytes(lb)
    amd = {'schema': 'minifly.amendment.v1', 'id': 'AMD-1', 'utc': '20261101T103000Z',
           'old_lock_sha256': old, 'new_lock_sha256': b.lock_sha, 'new_protocol_id': D + '-p2',
           'fields_changed': ['testbed.ceiling_grid'],
           'results_seen': ['results/admission/OUTPUTS.jsonl.gz', 'results/admission/TESTBED.json'],
           'reason': 'testbed_invalid', 'evidence': 'results/admission/TESTBED.json',
           'objective_distance_delta': {}, 'original_endpoint_kept_as_coprimary': False,
           'user_decision': None, 'author_agent': 'codex'}
    b.commit('amend', {P + 'ENDPOINT_LOCK.json': lb, P + 'AMENDMENTS.jsonl': dump_jsonl([amd])},
             utc='20261101T103000Z')
    aid = BA.att_id('20261101T104000Z', 'testbed2')
    arms = ['CEILING_' + c for c in BI.CEILINGS + ['EXACT_BP_LONG']] + ['FLOOR_' + c for c in BI.FLOORS]
    rows = []
    for w in BI.TECH_WORLDS:
        rows += BA.output_rows(w, 'technical', arms, BI.TESTBED_CELLS, aid, None, None, b.bh)
    b.testbed_rows = rows
    out = gzip_bytes(dump_jsonl(rows))
    gp = 'checks/GATE_lock_20261101T104000Z.json'
    g = dump_compact(BA.gate_report('lock', False, '20261101T104000Z', b.tl.head_sha(), b.lock_sha,
                                    None, 'launch', FAMILY_ID))
    b.attempts.append(BA.attempt(aid, 'testbed', ['python3', '-I', 'admit.py', '--testbed'],
                                 '20261101T104000Z', '20261101T105500Z', b.tl.head_sha(), b.lock_sha,
                                 None, [('results/admission/OUTPUTS.jsonl.gz', sha256_bytes(out))], gp,
                                 cpu=330.0))
    b.cycles.append(BA.cycle(2, 'testbed', 'stop', 'Every ceiling also fails on the amended testbed: '
                             'CANNOT_TEST_CLAIM_AS_SPECIFIED.', '20261101T105800Z', None, [aid], None,
                             False, 0))
    b.commit('testbed2', {P + 'results/admission/OUTPUTS.jsonl.gz': out, P + gp: g,
                          P + 'runlog/ATTEMPTS.jsonl': dump_jsonl(b.attempts),
                          P + 'runlog/CYCLES.jsonl': dump_jsonl(b.cycles),
                          P + 'results/admission/TESTBED.json': dump_json(b.testbed_summary())},
             utc='20261101T110000Z')
    rep = ('%s\n\nThe cycle-0 testbed could not reach the ceiling on the locked split, also after the '
           'testbed_invalid amendment AMD-1. The claim cannot be tested as specified; no learner code '
           'was written.\n' % C00C_HEADLINE)
    claim = c00c_claim(b)
    cb = dump_json(claim)
    from base_timeline import ledger_append
    f = {P + 'REPORT.md': rep.encode('utf-8'), P + 'CLAIM.json': cb}
    f.update(ledger_append(b.files, cb, '20261101T115500Z'))
    b.commit('report', f, utc='20261101T120000Z')
    M.fix_branches(tl)
    return tl, b


def c00c_claim(b):
    nm = 'not_measured'
    return {'schema': 'minifly.claim.v1', 'delivery': D, 'protocol_id': D + '-p2', 'level_attempted': 'E3',
            'level_earned': 'NONE', 'reuse_specificity': 'not_tested',
            'primary_gate': 'CANNOT_TEST_CLAIM_AS_SPECIFIED', 'dev_shadow_verdict': None,
            'invalid_reason': None, 'stop_reason': 'CANNOT_TEST_CLAIM_AS_SPECIFIED: every ceiling fails the '
                                                   'locked rule (ADM-TB-CEIL), also after AMD-1',
            'gate_type': 'objective', 'labels': [], 'worlds_role': 'PILOT', 'n_worlds': 0,
            'objective_metric': {'held_exact_own_eos': nm, 'held_by_stratum': {}, 'determined_fraction': 1.0,
                                 'old_end': nm, 'old_final': nm, 'old_drop': nm, 'new_final': nm,
                                 'best_shortcut': {'name': nm, 'value': nm},
                                 'best_other_arm': {'name': nm, 'value': nm}, 'headroom': nm, 'G': nm,
                                 'G_per_sample': nm, 'G_per_cpu_h': nm},
            'designs_tried_k': 0, 'amendments': ['AMD-1'], 'coprimary_results': [], 'posthoc_files': [],
            'lineage_parent': None, 'family_id': FAMILY_ID, 'native_path_only': True,
            'compute': {'records_total_incl_failed': 0, 'cpu_h_total_incl_failed': 0.18,
                        'family_cpu_h': 0.18},
            'provenance': {'interrupted': False, 'resume_attempts': [], 'resumed_units': [], 'replay_units': []},
            'verdict_source': {'decide_version': '1.0.0', 'verdict_sha256': None,
                               'endpoint_lock_sha256': b.lock_sha, 'source_lock_sha256': None},
            'predictions_scored': [{'id': p, 'observed': 'unresolved', 'hit': 'unresolved',
                                    'interval_score': 'unresolved', 'brier': 'unresolved'}
                                   for p in ('P1', 'P2', 'P3')],
            'findings_proposed': [],
            'findings_none_reason': 'The testbed could not test the claim; nothing durable beyond the stop.',
            'review': None, 'review_debt_id': None,
            'authorization': {'path': 'AUTHORIZATION.json',
                              'recommendation_shown': 'A, because it opens held E3 at low cost.'},
            'confirm_attempt': None, 'notes': '', 'checker_sha256': None, 'headline': C00C_HEADLINE}


def b_c00e(report_first_line=None, report_body=None, **extra):
    def lock(lk):
        lk['tier_declared'] = 1
        lk['worlds']['official'] = {'mode': 'listed', 'ids': [], 'n': 0}
        lk['worlds']['official_role_intended'] = 'PILOT'

    def auth(a):
        a['verbatim'] = ('Go ahead with %s at tier 1: implementation, technical and dev cycles only, up '
                         'to 20000 records, 4 CPU-hours and 3 cycles; no official run. Push to your own '
                         'branch.' % D)
        a['stages']['official'] = False
    def claim(b):
        return M.claim_doc(b, rows=b.dev_rows, primary_gate='NOT_OPENED', level_earned='NONE',
                           dev_shadow_verdict='FAIL', worlds_role='DEV', n_worlds=2, review=None,
                           headline=C00E_HEADLINE)
    opts = {'bi': {'OFF_WORLDS': []}, 'lock_patch': lock, 'auth_patch': auth, 'tier': 1,
            'skip': ['review', 'ack', 'rehearsal', 'official'], 'headline': C00E_HEADLINE,
            'claim_doc': claim}
    if report_first_line:
        opts['report_first_line'] = report_first_line
    if report_body:
        opts['report_body'] = report_body
    for k, v in extra.items():
        if k == 'after':
            opts.setdefault('after', {}).update(v)
        else:
            opts[k] = v
    return build_base(opts)


# C00d: complete seal (SPEC Section 10), confirm wording allowed.
NONCE = hashlib.sha256(b'mrk1-corpus-C00d-seal-nonce').hexdigest()


def derive_worlds(l1, l2, go_verbatim, n=8):
    """key = sha256(nonce || GO verbatim utf-8); seed_i = int(HMAC-SHA256(key, L1|L2|confirm_worlds|i)
    first 8 bytes, big-endian); world id = 'w_seal_<16 hex of seed_i>' (corpus convention)."""
    key = hashlib.sha256(NONCE.encode('ascii') + go_verbatim.encode('utf-8')).digest()
    out = []
    for i in range(n):
        msg = ('%s|%s|confirm_worlds|%d' % (l1, l2, i)).encode('ascii')
        seed = int.from_bytes(hmac.new(key, msg, hashlib.sha256).digest()[:8], 'big')
        out.append('w_seal_%016x' % seed)
    return out


REPLAY_DIGEST_FIELDS = ('world', 'world_role', 'arm', 'pool', 'checkpoint', 'item_id', 'prompt_hex', 'expected_hex',
                        'stream_hex', 'emitted_hex', 'own_eos', 'answer_path', 'native_closure_sha', 'params_sha256')


def replay_digest(rows, world, arm):
    """Digest of the official rows of one (world, arm) (SPEC REV-08 'the digests of the official rows'):
    sha256 over the canonical JSON of each row restricted to the OUTPUTS schema fields except attempt_id
    (which differs between attempts), sorted by (pool, checkpoint, item_id), each followed by a newline
    (SCHEMAS.md section 13)."""
    keyed = []
    for r in rows:
        if r['world'] != world or r['arm'] != arm:
            continue
        sub = {k: r.get(k) for k in REPLAY_DIGEST_FIELDS}
        keyed.append(((str(sub['pool']), str(sub['checkpoint']), str(sub['item_id'])), canonical_json(sub)))
    return sha256_bytes(b''.join(b + b'\n' for _k, b in sorted(keyed)))


def b_c00d():
    st = {}

    def seal(b):
        l1, l2 = b.lock_sha, b.sl['sha']
        sc = {'schema': 'minifly.seal_commit.v1', 'endpoint_lock_sha256': l1, 'source_lock_sha256': l2,
              'nonce_sha256': sha256_bytes(NONCE.encode('ascii')), 'reviewer_agent': 'claude',
              'reviewer_session': 's-corpus-review', 'utc': '20261102T130000Z'}
        b.commit('seal_commit', {P + 'review/SEAL_COMMIT.json': dump_json(sc)}, who='claude',
                 utc='20261102T130000Z')
        go = ('GO for the sealed confirmation of %s with ENDPOINT_LOCK %s and SOURCE_LOCK %s.'
              % (D, l1, l2))
        auth = json.loads(b.files[P + 'AUTHORIZATION.json'].decode())
        auth['stages']['official'] = {'granted': True, 'verbatim': go, 'utc': '20261102T140000Z',
                                      'lock_shas_shown': [l1, l2]}
        b.commit('go', {P + 'AUTHORIZATION.json': dump_json(auth)}, utc='20261102T140000Z')
        rv = {'schema': 'minifly.seal_reveal.v1', 'nonce': NONCE, 'utc': '20261102T150000Z'}
        b.commit('reveal', {P + 'review/SEAL_REVEAL.json': dump_json(rv)}, who='claude',
                 utc='20261102T150000Z')
        row = {'family_id': FAMILY_ID, 'delivery': D, 'attempt_index': 0, 'alpha_allocated': 0.05,
               'seal_reveal_commit': b.tl.head_sha(), 'utc': '20261102T151000Z'}
        b.commit('confirm_row', {BF.KB + 'CONFIRM_LEDGER.jsonl': dump_jsonl([row])}, who='claude',
                 utc='20261102T151000Z')
        worlds = derive_worlds(l1, l2, go)
        st['worlds'] = worlds
        b.o['official_worlds'] = worlds
        for i, w in enumerate(worlds):
            src = 'w_off_%d' % (i + 1)
            b.bh['w_held_wrong'][w] = b.bh['w_held_wrong'][src]
            if src in b.bh['w_old_drop']:
                b.bh['w_old_drop'][w] = b.bh['w_old_drop'][src]
        taught = {}
        for w in worlds:
            for arm in BI.ARMS:
                taught[P + 'manifests/TAUGHT/%s/%s.jsonl' % (w, arm)] = dump_jsonl(BF.taught_rows(w, arm))
        b.o.setdefault('extra_files', {})['official'] = taught

    def postrun(b):
        pairs = [(st['worlds'][0], 'W'), (st['worlds'][3], 'NPT'), (st['worlds'][5], 'W'),
                 (st['worlds'][7], 'SHUF')]
        rep = [{'world': w, 'arm': a, 'expected_digest': replay_digest(b.official_rows, w, a),
                'observed_digest': replay_digest(b.official_rows, w, a)} for w, a in pairs]
        doc = json.loads(b.files[P + 'review/REVIEW.json'].decode())
        doc['seal'] = {'nonce_sha256': sha256_bytes(NONCE.encode('ascii')), 'revealed_nonce': NONCE,
                       'reveal_utc': '20261102T150000Z'}
        doc['postrun'] = {'replay': rep, 'utc': '20261103T100000Z'}
        b.commit('postrun', {P + 'review/REVIEW.json': dump_json(doc)}, who='claude',
                 utc='20261103T100000Z')

    def lock(lk):
        lk['worlds']['official'] = {'mode': 'sealed', 'ids': [], 'n': 8}
        lk['worlds']['official_role_intended'] = 'SEALED_CONFIRM'
    def claim(b):
        return M.claim_doc(b, worlds_role='SEALED_CONFIRM',
                           confirm_attempt={'family_id': FAMILY_ID, 'index': 0, 'alpha': 0.05})
    # SPEC 12.2 C00d specifies 'OC 0.9'. With the base dev effect (W misses 2 and 3 of 16 held items in
    # w_dev_1/w_dev_2, mean dev d 0.281) the SPEC 9.8 simulation gives P(pass | target) = 0.79 < oc_min_power
    # 0.8, so in this fixture W misses one item fewer in each dev world (mean dev d 0.344, OC about 0.94).
    # Official worlds keep the base behaviour, so every official number is unchanged.
    dev_wrong = {'w_dev_1': ([1], []), 'w_dev_2': ([5], [6])}
    tl, b = build_base({'bi': {'OFF_WORLDS': []}, 'lock_patch': lock, 'sealed': True, 'claim_doc': claim,
                        'behaviour': {'w_held_wrong': dict(BI.default_behaviour()['w_held_wrong'], **dev_wrong)},
                        'after': {'rehearsal': [seal], 'official': [postrun]},
                        'headline_role': 'SEALED_CONFIRM',
                        'report_body': 'Sealed confirmation on 8 seal-derived worlds (SEAL_COMMIT, '
                                       'owner GO, SEAL_REVEAL, CONFIRM_LEDGER attempt 0, post-run '
                                       'replay of 4 histories). Numbers are hand-computed in '
                                       'tools/minifly_check/corpus/_build/base/EXPECTED_VALUES.md.\n'})
    return tl, b


__all__ = ['SD']
