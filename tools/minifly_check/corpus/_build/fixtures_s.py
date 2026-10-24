"""Supplementary fixtures S01-S25 (beyond the SPEC 12.2 table): one dirty fixture for each check id that
the table leaves without a firing fixture, so `selftest --mutate` can see every implemented check
notice an always-PASS replacement (SPEC 7.7, 20.2). Same base, one documented mutation each."""
import copy
import hashlib
import json

import base_files as BF
import fixtures_d as FD
import fixtures_g as G
import mutate as M
from base_timeline import FAMILY_ID, build_base
from fixtures_c import _set
from mrk_util import dump_json, dump_jsonl, sha256_bytes

D = BF.D
P = D + '/'
KIT = 'tools/minifly_check/'


def lock_only(**opts):
    o = dict(opts)
    o['stop_after'] = 'lock'
    return build_base(o)


def b_s01():
    tl, b = build_base({'extra_files': {'kb': {KIT + 'minifly_check.py': b'"""kit stub"""\n'}},
                        'stop_after': 'lock'})
    tl.worktree[KIT + 'minifly_check.py'] = b'"""kit stub, locally modified"""\n'
    tl.dirty.append(KIT + 'minifly_check.py')
    return tl, b


def b_s02():
    return lock_only(lock_patch=_set('schema', 'minifly.endpoint_lock.v0'))


def b_s03():
    def lock(lk):
        lk['predictions'][1]['point'] = '__FILL__'
    return lock_only(lock_patch=lock)


def b_s04():
    def lock(lk):
        lk['arms'][1]['zero_channels'] = []
    return lock_only(lock_patch=lock)


def b_s05():
    return lock_only(lock_patch=_set('primary.margin_abs', 0.03))


def b_s06():
    return lock_only(lock_patch=_set('constraints_sha256', '0' * 64))


def b_s07():
    def auth(a):
        del a['recommendation_shown']
        del a['session']
    return lock_only(auth_patch=auth)


def b_s08():
    def amend(b):
        old = b.lock_sha
        lock = json.loads(b.files[P + 'ENDPOINT_LOCK.json'].decode())
        lock['protocol_id'] = D + '-p2'
        lock['testbed']['floor_grid'] = lock['testbed']['floor_grid'][:3]
        lb = dump_json(lock)
        row = {'schema': 'minifly.amendment.v1', 'id': 'AMD-1', 'utc': '20261101T103000Z',
               'old_lock_sha256': old, 'new_lock_sha256': sha256_bytes(lb), 'new_protocol_id': D + '-p2',
               'fields_changed': ['testbed.floor_grid'],
               'results_seen': ['results/admission/OUTPUTS.jsonl.gz', 'results/admission/TESTBED.json'],
               'reason': 'testbed_invalid', 'evidence': 'results/admission/TESTBED.json',
               'objective_distance_delta': {}, 'original_endpoint_kept_as_coprimary': False,
               'user_decision': None, 'author_agent': 'codex'}
        b.commit('amend', {P + 'ENDPOINT_LOCK.json': lb, P + 'AMENDMENTS.jsonl': dump_jsonl([row])},
                 utc='20261101T103000Z')
    return build_base({'after': {'testbed': [amend]}, 'stop_after': 'testbed'})


def b_s09():
    return lock_only(lock_patch=_set('guards', []))


def b_s10():
    return lock_only(lock_patch=_set('primary.strata_counted', ['S0', 'S1', 'S2', 'S3']))


def b_s11():
    return build_base({'lock_patch': _set('primary.aggregator', {'type': 'k_of_n', 'k': 0})})


def b_s12():
    def rows(row):
        if row['world'] == 'w_dev_1' and row['arm'] == 'W' and row['item_id'] == 'h05':
            return None
        return row
    return build_base({'row_patch': {'dev': rows}})


def b_s13():
    def lock(lk):
        lk['primary']['strata_counted'] = ['S2', 'S3', 'S4']
        lk['units']['witness'] = 'column_closure'
    return build_base({'lock_patch': lock})


def b_s14():
    def adm(a):
        a['worlds'] = ['w_tech_1', 'w_dev_1']
    return build_base({'admission_patch': adm})


def b_s15():
    def auth(a):
        a['caps']['max_cycles'] = 2
        a['verbatim'] = a['verbatim'].replace('3 cycles', '2 cycles')
    dry = P + 'checks/DRY_official_20261102T115000Z.json'
    return build_base({'auth_patch': auth, 'drop_files': {'rehearsal': [dry]}})


def b_s16():
    def rerun(b):
        f, rows, aid = b.stage_run('official2', 'official', 'official', b.off_worlds(), '20261103T093000Z',
                                   '20261103T093000Z', '20261103T095000Z', 480.0)
        b.official_rows = rows
        b.commit('official2', f, utc='20261103T100000Z')
    return build_base({'after': {'official': [rerun]}})


def b_s17():
    def claim(b):
        doc = M.claim_doc(b, level_earned='E5', findings_none_reason='')
        return doc
    body = 'The inherited native core from minifly_b is the same fly, now composing held sums.\n'
    return build_base({'claim_doc': claim, 'ledger_row': False, 'report_body': body})


def b_s18():
    def reclaim(b):
        doc = M.claim_doc(b, primary_gate='FAIL', level_earned='NONE')
        b.commit('reclaim', {P + 'CLAIM.json': dump_json(doc)}, utc='20261104T090000Z')
    return build_base({'after': {'report': [reclaim]}})


def b_s19():
    def review(doc, version):
        for c in doc['executed_checks']:
            if c['kind'] == 'rerun_admission':
                c['match'] = False
                c['result'] = 'digest_mismatch'
        doc['cheapest_shortcut_attempt']['passed_locked_rule'] = True
        doc['canary']['detected'] = 14
        doc['findings'] = [{'id': 'R1', 'fm_ids': ['FM-02'], 'claim': 'NPT readout write site differs '
                            'from W', 'severity': 'MAJOR', 'executed': True, 'status': 'OPEN',
                            'reconciled_by_author': None, 'countersigned_by_reviewer': False,
                            'ruling_user_decision': None}]
    return build_base({'review_patch': review})


def b_s20():
    def edit(b):
        rows = copy.deepcopy(BF.FINDINGS_BASE)
        rows[0]['claim'] = 'STOP channel adds +22.10pp own-EOS exact on OLD (restated).'
        inc = [{'id': 'INC-S20-1', 'kind': 'note', 'fm_class': 'FM-26', 'delivery': D,
                'symptom': 'free-text note', 'evidence': D + '/REPORT.md:1', 'found_by_agent': 'codex',
                'found_utc': '20261103T121000Z', 'caught_at_discovery': False,
                'checker_sha_at_discovery': None, 'heldout_eligible': False, 'check_id': None,
                'fixture_id': None, 'status': 'open'}]
        b.commit('ledger_edit', {BF.KB + 'FINDINGS.jsonl': dump_jsonl(rows),
                                 BF.KB + 'INCIDENTS.jsonl': dump_jsonl(inc)}, utc='20261103T121000Z')
    return build_base({'after': {'report': [edit]}})


def b_s21():
    def ptr(b):
        claim_bytes = b.files[P + 'CLAIM.json']
        claim = json.loads(claim_bytes.decode())
        p = {'schema': 'minifly.pointer.v1', 'track': 'addition_e3', 'delivery': D, 'claim': D + '/CLAIM.json',
             'claim_sha256': sha256_bytes(claim_bytes), 'level_earned': 'E4',
             'primary_gate': claim['primary_gate'], 'gate_type': claim['gate_type'],
             'worlds_role': claim['worlds_role'], 'objective_metric': claim['objective_metric'],
             'narrowing_rows': [], 'updated_utc': '20261103T130000Z', 'previous_snapshot': None,
             'user_decision': None}
        debt = [{'id': 'DEBT-20261103-1', 'delivery': D, 'utc': '20261103T125000Z',
                 'reason': 'codex unavailable for the post-run replay', 'recorded_by': 'claude:s-corpus',
                 'repaid_by': None}]
        b.commit('pointer', {'PROJECT_DOCUMENTS/CURRENT_ADDITION_E3_POINTER.json': dump_json(p),
                             BF.KB + 'REVIEW_DEBT.jsonl': dump_jsonl(debt)}, utc='20261103T130000Z')
    return build_base({'after': {'report': [ptr]}})


def b_s22():
    """C00d with a broken seal: GO before SEAL_COMMIT, a revealed nonce that does not match the
    commitment, listed (not derived) official worlds, the CONFIRM_LEDGER row after the official outputs,
    alpha_allocated 0.025 < lock alpha 0.05, and a post-run replay digest that differs."""
    bad_nonce = hashlib.sha256(b'mrk1-corpus-S22-other-nonce').hexdigest()

    def seal(b):
        l1, l2 = b.lock_sha, b.sl['sha']
        go = 'GO for the sealed confirmation of %s with ENDPOINT_LOCK %s and SOURCE_LOCK %s.' % (D, l1, l2)
        auth = json.loads(b.files[P + 'AUTHORIZATION.json'].decode())
        auth['stages']['official'] = {'granted': True, 'verbatim': go, 'utc': '20261102T125000Z',
                                      'lock_shas_shown': [l1, l2]}
        b.commit('go', {P + 'AUTHORIZATION.json': dump_json(auth)}, utc='20261102T125000Z')
        sc = {'schema': 'minifly.seal_commit.v1', 'endpoint_lock_sha256': l1, 'source_lock_sha256': l2,
              'nonce_sha256': sha256_bytes(FD.NONCE.encode('ascii')), 'reviewer_agent': 'claude',
              'reviewer_session': 's-corpus-review', 'utc': '20261102T130000Z'}
        b.commit('seal_commit', {P + 'review/SEAL_COMMIT.json': dump_json(sc)}, who='claude',
                 utc='20261102T130000Z')
        rv = {'schema': 'minifly.seal_reveal.v1', 'nonce': bad_nonce, 'utc': '20261102T150000Z'}
        b.commit('reveal', {P + 'review/SEAL_REVEAL.json': dump_json(rv)}, who='claude',
                 utc='20261102T150000Z')

    def after_official(b):
        row = {'family_id': FAMILY_ID, 'delivery': D, 'attempt_index': 0, 'alpha_allocated': 0.025,
               'seal_reveal_commit': b.tl.sha('reveal'), 'utc': '20261103T091000Z'}
        b.commit('confirm_row', {BF.KB + 'CONFIRM_LEDGER.jsonl': dump_jsonl([row])}, who='claude',
                 utc='20261103T091000Z')
        pairs = [('w_off_1', 'W'), ('w_off_4', 'NPT'), ('w_off_6', 'W'), ('w_off_8', 'SHUF')]
        rep = [{'world': w, 'arm': a, 'expected_digest': FD.replay_digest(b.official_rows, w, a),
                'observed_digest': FD.replay_digest(b.official_rows, w, a)} for w, a in pairs]
        rep[2]['observed_digest'] = '0' * 64
        doc = json.loads(b.files[P + 'review/REVIEW.json'].decode())
        doc['seal'] = {'nonce_sha256': sha256_bytes(FD.NONCE.encode('ascii')), 'revealed_nonce': bad_nonce,
                       'reveal_utc': '20261102T150000Z'}
        doc['postrun'] = {'replay': rep, 'utc': '20261103T100000Z'}
        b.commit('postrun', {P + 'review/REVIEW.json': dump_json(doc)}, who='claude',
                 utc='20261103T100000Z')

    def claim(b):
        return M.claim_doc(b, worlds_role='SEALED_CONFIRM',
                           confirm_attempt={'family_id': FAMILY_ID, 'index': 0, 'alpha': 0.05})
    return build_base({'lock_patch': _set('worlds.official_role_intended', 'SEALED_CONFIRM'), 'sealed': True,
                       'claim_doc': claim, 'after': {'rehearsal': [seal], 'official': [after_official]},
                       'headline_role': 'SEALED_CONFIRM'})


def b_s23():
    tl = G._tl_base()
    rows = copy.deepcopy(BF.FINDINGS_BASE)
    rows[1]['caveat'] = 'edited in place'
    G._add(tl, 'g_ledger_edit', '20261105T100000Z', 'codex', {BF.KB + 'FINDINGS.jsonl': dump_jsonl(rows)})
    return tl, None


def b_s24():
    tl = G._tl_base()
    long_desc = 'Corpus miniature of the governance skill. ' * 30
    md = G.skill_md('minifly-governance').decode('utf-8').replace(
        'description: Corpus miniature of the minifly-governance skill (governance fixtures only).',
        'description: ' + long_desc.strip())
    G._add(tl, 'g_skill_long', '20261105T100000Z', 'codex',
           {'.claude/skills/minifly-governance/SKILL.md': md.encode('utf-8')})
    return tl, None


def b_s25():
    tl = G._tl_base()
    x03 = {'X03': {'commands': [['gate', '--stage', 'lock', G.GD]], 'must_fire': ['LOCK-11'],
                   'must_not_fire': [], 'must_warn': [], 'expected_exit': [1], 'expect': {},
                   'source_incident': None, 'fm_ids': ['FM-25']}}
    G._add(tl, 'g_tighten', '20261105T100000Z', 'codex',
           {KIT + 'EXPECTED.json': dump_json(G.mini_expected(extra=x03)),
            KIT + 'corpus/X03/FIXTURE.json': dump_json({'id': 'X03', 'title': 'prediction missing',
                                                        'fm_ids': ['FM-25'], 'source_incident': None,
                                                        'kind': 'dirty'})})
    return tl, None
