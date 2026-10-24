"""Unit tests for review.py (WP4): REV-01..08, DEBT-01..02, `MFC review`. TimelineGit fixtures; stdlib only."""
import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for p in (str(KIT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import common  # noqa: E402
import review  # noqa: E402
from test_admitcheck import (C1, C2, C3, C4, C5, CLAUDE, CLAUDE_BR, CODEX, CODEX_BR, NAME, WP4Fixture,  # noqa: E402
                             jsonl, output_rows, status, write)

CANARY_REL = 'checks/CANARY_20261101T030000Z.json'
SCRIPTS = {'X1': ('rerun_admission', 'review/rerun.py', 'review/out_rerun.json'),
           'X2': ('reject_input', 'review/reject.py', 'review/out_reject.json'),
           'X3': ('trace_number', 'review/trace.py', 'review/out_trace.json'),
           'X4': ('shortcut_attempt', 'review/shortcut.py', 'review/out_shortcut.json')}


class ReviewFixture(WP4Fixture):
    def add_review(self, **over):
        checks = []
        for cid, (kind, script, out) in sorted(SCRIPTS.items()):
            self.w(script, ('# reviewer script %s\nprint("%s")\n' % (cid, kind)).encode())
            self.w(out, {'check': cid, 'kind': kind, 'observed': 'value for %s' % cid})
            e = {'id': cid, 'kind': kind, 'script': script, 'script_sha256': common.sha256_file(self.d / script),
                 'cmd': 'python3 -I %s' % script, 'output': out, 'output_sha256': common.sha256_file(self.d / out),
                 'cpu_cap_s': 600, 'result': 'ok', 'compared_to': None, 'match': None}
            if kind in ('rerun_admission', 'trace_number'):
                e.update({'compared_to': 'ADMISSION.json', 'match': True})
            checks.append(e)
        self.w(CANARY_REL, {'schema': review.SCHEMA_CANARY_REPORT, 'delivery': NAME, 'seed': 'c0ffee', 'n_mutants': 20,
                            'detected': 20, 'exit_code': 0, 'verdict': 'PASS', 'git_mode': 'timeline',
                            'checker_sha256': 'x' * 64, 'endpoint_lock_sha256': self.lock_sha(),
                            'results': [{'id': c, 'status': 'PASS'} for c in ('CAN-01', 'CAN-02', 'CAN-03')]})
        rv = {'schema': review.SCHEMA_REVIEW, 'delivery': NAME, 'reviewer_agent': 'claude:s-review',
              'reviewer_session': 's-review', 'reviewer_branch': CLAUDE_BR[0], 'author_agent': 'codex',
              'reviewed_endpoint_lock_sha256': self.lock_sha(),
              'admission_sha256': common.sha256_file(self.d / 'ADMISSION.json'), 'utc': '20261101T030000Z',
              'executed_checks': checks, 'canary': {'seed': 'c0ffee', 'n': 20, 'detected': 20, 'report': CANARY_REL},
              'cheapest_shortcut_attempt': {'description': 'DIGIT_NO_CARRY on held', 'script': 'review/shortcut.py',
                                            'passed_locked_rule': False},
              'fm_judgments': {'FM-02': 'write counter sits in Learner.write_pair'}, 'findings': [],
              'source_lock_acks': [], 'seal': None, 'postrun': None, 'decision': 'ADMIT'}
        rv.update(over)
        self.w(review.REVIEW_REL, rv)
        return rv

    def review_paths(self):
        out = [NAME + '/' + p for (_k, s, o) in SCRIPTS.values() for p in (s, o)]
        return sorted(out + [NAME + '/' + review.REVIEW_REL, NAME + '/' + CANARY_REL])

    def commit_review(self, trailers=None, branches=None, after=C2, sha=C4, extra=None):
        commits = [{'sha': sha, 'trailers': trailers or CLAUDE, 'branches': branches or CLAUDE_BR,
                    'paths': self.review_paths(), 'after': after}]
        commits += list(extra or [])
        self.extra_commits = commits
        self.timeline()


class ReviewCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='mfc_wp4r_')
        self.f = ReviewFixture(self.tmpdir).build()
        self.rv = self.f.add_review()
        self.f.commit_review()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def ctx(self, **kw):
        kw.setdefault('tier', 2)
        return self.f.ctx(**kw)

    def assertStatus(self, rid, want, ctx=None, contains=None):
        st, det = status(ctx or self.ctx(), rid)
        self.assertEqual(st, want, '%s: %s %s' % (rid, st, det))
        if contains:
            self.assertIn(contains, det)
        return det

    def edit_review(self, recommit=True, **over):
        rv = json.loads((self.f.d / review.REVIEW_REL).read_text())
        rv.update(over)
        self.f.w(review.REVIEW_REL, rv)
        if recommit:
            self.f.commit_review()
        return rv


class CleanReview(ReviewCase):
    def test_rev_01_to_07_pass(self):
        ctx = self.ctx()
        for rid in review.REV_PRIMARY:
            st, det = status(ctx, rid)
            self.assertEqual(st, common.PASS, '%s: %s' % (rid, det))
        self.assertEqual(status(ctx, 'REV-08')[0], common.NOT_APPLICABLE)
        self.assertEqual(status(ctx, 'DEBT-02')[0], common.PASS)

    def test_registered_ids_match_rules(self):
        rules = common.load_rules()
        mine = sorted(r['id'] for r in rules['rules'] if r['module'] == 'review')
        self.assertEqual(mine, sorted(list(review.REV_PRIMARY) + ['REV-08', 'DEBT-01', 'DEBT-02']))


class IdentityChecks(ReviewCase):
    def test_same_family_review_fails(self):
        # FM-27 / C27: a same-family review on the author's branch
        self.edit_review(reviewer_agent='codex')
        self.f.commit_review(trailers=CODEX, branches=CODEX_BR)
        det = self.assertStatus('REV-01', common.FAIL, contains='same-family')
        self.assertIn('not the reviewer family claude', det)

    def test_reviewer_trailer_outside_namespace_fails(self):
        self.f.commit_review(branches=CODEX_BR)
        self.assertStatus('REV-01', common.FAIL, contains='origin/claude/*')

    def test_missing_trailer_fails(self):
        self.f.commit_review(trailers={'Reviewed-by': ['someone']})
        self.assertStatus('REV-01', common.FAIL, contains='no Acting-Agent trailer')

    def test_lock_trailer_disagreeing_with_author_fails(self):
        tl = json.loads((self.f.fx / 'timeline.json').read_text())
        tl['commits'][0]['trailers'] = {'Acting-Agent': ['claude:s1']}
        write(self.f.fx / 'timeline.json', tl)
        self.assertStatus('REV-01', common.FAIL, contains='lock.author_agent says codex')

    def test_author_may_only_reconcile(self):
        rel = NAME + '/' + review.REVIEW_REL
        hist = self.f.fx / 'history' / C4 / rel
        write(hist, (self.f.d / review.REVIEW_REL).read_bytes())
        # the author (codex) flips a reviewer-owned field in a later commit
        self.edit_review(recommit=False, decision='ADMIT', fm_judgments={'FM-02': 'edited by author'})
        self.f.commit_review(extra=[{'sha': C5, 'trailers': CODEX, 'branches': CODEX_BR, 'paths': [rel]}])
        self.assertStatus('REV-01', common.FAIL, contains='reviewer-owned fields')

    def test_author_reconciliation_is_allowed(self):
        rel = NAME + '/' + review.REVIEW_REL
        self.edit_review(recommit=False, findings=[{'id': 'F1', 'fm_ids': ['FM-02'], 'claim': 'x', 'severity': 'MINOR',
                                                    'executed': 'X1', 'status': 'CONFIRMED'}])
        write(self.f.fx / 'history' / C4 / rel, (self.f.d / review.REVIEW_REL).read_bytes())
        rv = json.loads((self.f.d / review.REVIEW_REL).read_text())
        rv['findings'][0]['reconciled_by_author'] = {'utc': '20261101T060000Z', 'note': 'fixed', 'evidence': 'c5'}
        self.f.w(review.REVIEW_REL, rv)
        self.f.commit_review(extra=[{'sha': C5, 'trailers': CODEX, 'branches': CODEX_BR, 'paths': [rel]}])
        self.assertStatus('REV-01', common.PASS)

    def test_uncommitted_reviewer_field_edit_fails(self):
        rel = NAME + '/' + review.REVIEW_REL
        write(self.f.fx / 'history' / C4 / rel, (self.f.d / review.REVIEW_REL).read_bytes())
        self.edit_review(recommit=False, decision='BLOCK')
        self.assertStatus('REV-01', common.FAIL, contains='worktree')

    def test_review_not_committed_is_missing(self):
        self.f.extra_commits = []
        self.f.timeline()
        self.assertStatus('REV-01', common.MISSING)

    def test_copied_review_of_another_delivery_fails(self):
        self.edit_review(delivery='minifly_other_dev_v1_20261001')
        self.assertStatus('REV-01', common.FAIL, contains='another delivery')


class ExecutedChecks(ReviewCase):
    def test_missing_kind_and_no_checks(self):
        rv = self.edit_review(executed_checks=[e for e in self.rv['executed_checks'] if e['kind'] != 'reject_input'])
        self.assertStatus('REV-02', common.FAIL, contains='reject_input')
        self.edit_review(executed_checks=[])
        self.assertStatus('REV-02', common.FAIL, contains='UNVERIFIED')
        del rv

    def test_oracle_upstream_required_with_stage_chain(self):
        lock = self.f.read('ENDPOINT_LOCK.json')
        lock['stage_chain'] = True
        self.f.set_lock(lock)
        self.assertStatus('REV-02', common.FAIL, contains='oracle_upstream')

    def test_sha_mismatch_and_uncommitted(self):
        rel = NAME + '/review/out_trace.json'
        write(self.f.fx / 'history' / C4 / rel, (self.f.d / 'review/out_trace.json').read_bytes())
        self.f.w('review/out_trace.json', {'tampered': True})
        det = self.assertStatus('REV-02', common.FAIL, contains='does not match')
        self.assertIn('not committed', det)
        checks = json.loads(json.dumps(self.rv['executed_checks']))
        checks[2]['output_sha256'] = common.sha256_file(self.f.d / 'review/out_trace.json')
        self.edit_review(recommit=False, executed_checks=checks)
        write(self.f.fx / 'history' / C4 / (NAME + '/' + review.REVIEW_REL), (self.f.d / review.REVIEW_REL).read_bytes())
        det = self.assertStatus('REV-02', common.FAIL, contains='not committed as is')
        self.assertNotIn('does not match', det)

    def test_cpu_cap_and_match(self):
        checks = json.loads(json.dumps(self.rv['executed_checks']))
        checks[0]['cpu_cap_s'] = None
        checks[0]['match'] = None   # rerun_admission compares digests (SPEC 6.4): match must be a boolean
        self.edit_review(executed_checks=checks)
        det = self.assertStatus('REV-02', common.FAIL, contains='cpu_cap_s')
        self.assertIn('match', det)

    def test_trace_number_match_optional(self):
        # SPEC 6.4: trace_number traces a number to its producing line; REV-02 requires no compared_to/match
        checks = json.loads(json.dumps(self.rv['executed_checks']))
        k = [i for i, c in enumerate(checks) if c['kind'] == 'trace_number'][0]
        checks[k]['match'] = None
        checks[k]['compared_to'] = None
        self.edit_review(executed_checks=checks)
        self.assertStatus('REV-02', common.PASS)
        checks[k]['match'] = 'yes'
        self.edit_review(executed_checks=checks)
        self.assertStatus('REV-02', common.FAIL, contains='neither null nor a boolean')

    def test_script_outside_review_dir_fails(self):
        checks = json.loads(json.dumps(self.rv['executed_checks']))
        checks[1]['script'] = 'run.py'
        self.edit_review(executed_checks=checks)
        self.assertStatus('REV-02', common.FAIL, contains='not under the delivery review/')

    def test_rerun_mismatch_needs_a_finding(self):
        checks = json.loads(json.dumps(self.rv['executed_checks']))
        checks[0]['match'] = False
        self.edit_review(executed_checks=checks)
        self.assertStatus('REV-03', common.FAIL, contains='no finding')
        self.edit_review(findings=[{'id': 'F1', 'fm_ids': ['FM-01'], 'claim': 'rerun X1 digests differ',
                                    'severity': 'MAJOR', 'executed': 'X1', 'status': 'CONFIRMED',
                                    'reconciled_by_author': {'utc': '20261101T060000Z', 'note': 'seed bug fixed'}}])
        self.assertStatus('REV-03', common.PASS)

    def test_shortcut_passing_forces_block(self):
        self.edit_review(cheapest_shortcut_attempt={'description': 'FIXED_1', 'script': 'review/shortcut.py',
                                                    'passed_locked_rule': True})
        self.assertStatus('REV-04', common.FAIL, contains='must be BLOCK')
        self.edit_review(decision='BLOCK')
        self.assertStatus('REV-04', common.PASS)
        self.assertStatus('REV-06', common.FAIL, contains='not ADMIT')

    def test_canary_survivors_and_report(self):
        self.edit_review(canary={'seed': 'c0ffee', 'n': 20, 'detected': 19, 'report': CANARY_REL})
        det = self.assertStatus('REV-05', common.FAIL, contains='19 of 20')
        self.assertIn('REVIEW says', det)
        self.edit_review(canary={'seed': 'c0ffee', 'n': 20, 'detected': 20, 'report': 'checks/NOPE.json'})
        self.assertStatus('REV-05', common.FAIL, contains='CANARY_')

    def test_canary_report_must_follow_the_lock(self):
        tl = json.loads((self.f.fx / 'timeline.json').read_text())
        rep = NAME + '/' + CANARY_REL
        for c in tl['commits']:
            if rep in c['paths']:
                c['paths'].remove(rep)
        tl['commits'][0]['paths'].append(rep)
        write(self.f.fx / 'timeline.json', tl)
        self.assertStatus('REV-05', common.FAIL, contains='does not follow the ENDPOINT_LOCK commit')

    def test_canary_report_on_another_lock_fails(self):
        rep = json.loads((self.f.d / CANARY_REL).read_text())
        rep['endpoint_lock_sha256'] = 'a' * 64
        self.f.w(CANARY_REL, rep)
        self.assertStatus('REV-05', common.FAIL, contains='another ENDPOINT_LOCK')


class FindingsAndTiming(ReviewCase):
    def test_refuted_blocker_needs_countersign(self):
        f = {'id': 'F1', 'fm_ids': ['FM-02'], 'claim': 'NPT writes pair_memory', 'severity': 'BLOCKER',
             'executed': 'X1', 'status': 'REFUTED',
             'reconciled_by_author': {'utc': '20261101T060000Z', 'note': 'counter is at write_pair', 'evidence': 'X1'}}
        self.edit_review(findings=[f])
        self.assertStatus('REV-06', common.FAIL, contains='REFUTED BLOCKER')
        self.edit_review(findings=[dict(f, countersigned_by_reviewer=True)])
        self.assertStatus('REV-06', common.PASS)
        self.f.wr('PROJECT_DOCUMENTS/KB/USER_DECISIONS.jsonl', jsonl([{'id': 'UD-20261101-1', 'verbatim': 'ruling',
                                                                       'scope': ['review_ruling:x']}]))
        self.edit_review(findings=[dict(f, ruling_user_decision='UD-20261101-1')])
        self.assertStatus('REV-06', common.PASS)
        self.edit_review(findings=[dict(f, ruling_user_decision='UD-20991231-9')])
        self.assertStatus('REV-06', common.FAIL)

    def test_open_major_and_unverified_blocker(self):
        self.edit_review(findings=[{'id': 'F2', 'severity': 'MAJOR', 'executed': 'X3', 'status': 'OPEN'}])
        self.assertStatus('REV-06', common.FAIL, contains='is open')
        self.edit_review(findings=[{'id': 'F3', 'severity': 'BLOCKER', 'executed': False, 'status': 'CONFIRMED',
                                    'reconciled_by_author': {'utc': '20261101T060000Z', 'note': 'fixed'}}])
        self.assertStatus('REV-06', common.FAIL, contains='UNVERIFIED BLOCKER')
        self.edit_review(findings=[{'id': 'F4', 'severity': 'MAJOR', 'executed': True, 'status': 'CONFIRMED'}])
        self.assertStatus('REV-06', common.FAIL, contains='not reconciled')
        self.edit_review(findings=[{'id': 'F5', 'severity': 'MINOR', 'executed': True,
                                    'status': 'CONFIRMED_WITH_QUANTITATIVE_CORRECTION'}])
        self.assertStatus('REV-06', common.PASS)
        self.edit_review(findings=[{'id': 'F6', 'severity': 'MINOR', 'status': 'MOSTLY_FINE'}])
        self.assertStatus('REV-06', common.FAIL, contains='not a known status')

    def test_review_after_first_dev_output_fails(self):
        self.f.commit_review(after=C3)
        self.assertStatus('REV-06', common.FAIL, contains='not a proper ancestor')


class Padding(ReviewCase):
    def test_copied_output_from_another_delivery_fails(self):
        # FM-27 / C27: byte-identical executed-check outputs across deliveries
        other = 'minifly_other_dev_v1_20261001/review/old_trace.json'
        self.f.wr(other, (self.f.d / 'review/out_trace.json').read_bytes())
        self.assertStatus('REV-07', common.FAIL, contains=other)

    def test_identical_outputs_within_the_review_fail(self):
        self.f.w('review/out_reject.json', (self.f.d / 'review/out_trace.json').read_bytes())
        checks = json.loads(json.dumps(self.rv['executed_checks']))
        for e in checks:
            e['output_sha256'] = common.sha256_file(self.f.d / e['output'])
        self.edit_review(executed_checks=checks)
        self.assertStatus('REV-07', common.FAIL, contains='byte-identical')

    def test_output_outside_scan_types_is_missing(self):
        self.f.w('review/out_trace.txt', b'some text output\n')
        checks = json.loads(json.dumps(self.rv['executed_checks']))
        checks[2]['output'] = 'review/out_trace.txt'
        checks[2]['output_sha256'] = common.sha256_file(self.f.d / 'review/out_trace.txt')
        self.edit_review(executed_checks=checks)
        self.assertStatus('REV-07', common.MISSING, contains='SPEC 20.4')

    def test_unrelated_identical_file_outside_review_dirs_is_ignored(self):
        self.f.wr('minifly_other_dev_v1_20261001/notes/copy.json', (self.f.d / 'review/out_trace.json').read_bytes())
        self.assertStatus('REV-07', common.PASS)


class Replay(ReviewCase):
    OFF = ['wo1', 'wo2', 'wo3']

    def setUp(self):
        super().setUp()
        lock = self.f.read('ENDPOINT_LOCK.json')
        lock['worlds']['official'] = {'mode': 'listed', 'ids': list(self.OFF), 'n': 3}
        lock['worlds']['official_role_intended'] = 'SEALED_CONFIRM'
        self.f.set_lock(lock)
        rows = [r for r in output_rows(self.OFF, 'official', ('W', 'NPT')) if r['pool'] == 'HELD']
        self.rows = rows
        self.f.w('results/official/OUTPUTS.jsonl', jsonl(rows))
        self.f.timeline()

    def digest(self, w, a, rows=None):
        return review.row_digest([r for r in (rows or self.rows) if r['world'] == w and r['arm'] == a])

    def test_replay_matches(self):
        pairs = [('wo1', 'W'), ('wo2', 'NPT'), ('wo3', 'W'), ('wo1', 'NPT')]
        rep = [{'world': w, 'arm': a, 'expected_digest': self.digest(w, a), 'observed_digest': self.digest(w, a)}
               for w, a in pairs]
        self.edit_review(postrun={'replay': rep, 'utc': '20261105T000000Z'})
        self.assertStatus('REV-08', common.PASS, ctx=self.ctx(stage='publish'))
        # attempt_id is provenance and does not enter the digest
        alt = [dict(r, attempt_id='ATT-REPLAY') for r in self.rows]
        self.assertEqual(self.digest('wo1', 'W'), self.digest('wo1', 'W', alt))

    def test_replay_mismatch_fails(self):
        bad = [dict(r) for r in self.rows]
        bad[0]['stream_hex'] = '30303030'
        pairs = [('wo1', 'W'), ('wo2', 'NPT'), ('wo3', 'W'), ('wo1', 'NPT')]
        rep = [{'world': w, 'arm': a, 'expected_digest': self.digest(w, a), 'observed_digest': self.digest(w, a, bad)}
               for w, a in pairs]
        self.edit_review(postrun={'replay': rep, 'utc': '20261105T000000Z'})
        self.assertStatus('REV-08', common.FAIL, contains='does not reproduce')
        self.edit_review(postrun={'replay': rep[:2], 'utc': '20261105T000000Z'})
        self.assertStatus('REV-08', common.FAIL, contains='at least 4')

    def test_replay_plan_seeded(self):
        ctx = self.ctx()
        p1 = review.replay_plan(ctx, 'c0ffee', 4)
        p2 = review.replay_plan(self.ctx(), 'c0ffee', 4)
        self.assertEqual(p1['pairs'], p2['pairs'])
        self.assertEqual(p1['n'], 4)
        self.assertEqual(p1['pairs_available'], 6)
        plans = set(tuple((p['world'], p['arm']) for p in review.replay_plan(self.ctx(), s, 2)['pairs'])
                    for s in ('00', '01', '02', '03', 'aa', 'bb', 'cc', 'dd'))
        self.assertGreater(len(plans), 1)
        with self.assertRaises(common.InvalidInput):
            review.replay_plan(ctx, 'xyz', 4)
        # postrun.seed, when recorded, must reproduce the replayed pairs
        rep = [{'world': p['world'], 'arm': p['arm'], 'expected_digest': p['expected_digest'],
                'observed_digest': p['expected_digest']} for p in p1['pairs']]
        self.edit_review(postrun={'replay': rep, 'utc': '20261105T000000Z', 'seed': 'c0ffee'})
        self.assertStatus('REV-08', common.PASS, ctx=self.ctx(stage='publish'))
        self.edit_review(postrun={'replay': rep, 'utc': '20261105T000000Z', 'seed': 'beef'})
        st, det = status(self.ctx(stage='publish'), 'REV-08')
        if st == common.FAIL:
            self.assertIn('replay plan', det)

    def test_cmd_review_replay_plan(self):
        import minifly_check
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = minifly_check.main(['--git', 'timeline:%s' % self.f.fx, '--repo-root', str(self.f.root), '--json',
                                     'review', 'replay-plan', NAME, '--seed', 'c0ffee', '--n', '3'])
        self.assertEqual(rc, 0)
        plan = json.loads(out.getvalue())
        self.assertEqual(plan['n'], 3)
        self.assertEqual(plan['pairs'], review.replay_plan(self.ctx(), 'c0ffee', 3)['pairs'])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = minifly_check.main(['--git', 'timeline:%s' % self.f.fx, '--repo-root', str(self.f.root),
                                     'review', 'replay-plan', NAME])
        self.assertEqual(rc, 3)


class Debt(ReviewCase):
    def set_debt(self, rows):
        self.f.wr('PROJECT_DOCUMENTS/KB/REVIEW_DEBT.jsonl', jsonl(rows))

    def debt_row(self, i, delivery=NAME, repaid=None):
        return {'id': 'DEBT-20261102-%d' % i, 'delivery': delivery, 'utc': '20261102T000000Z',
                'reason': 'other family unavailable', 'recorded_by': 'codex', 'repaid_by': repaid}

    def test_debt_01(self):
        self.assertStatus('DEBT-01', common.FAIL, contains='no open REVIEW_DEBT')
        self.set_debt([self.debt_row(1)])
        self.assertStatus('DEBT-01', common.PASS)
        self.set_debt([self.debt_row(1), self.debt_row(2, 'minifly_a_dev_v1_20261001'),
                       self.debt_row(3, 'minifly_b_dev_v1_20261001')])
        self.assertStatus('DEBT-01', common.FAIL, contains='cap review_debt_max_open 3')
        self.set_debt([self.debt_row(1), self.debt_row(1, repaid='review/REVIEW.json')])
        self.assertStatus('DEBT-01', common.FAIL, contains='no open REVIEW_DEBT')

    def test_debt_does_not_cover_main_official(self):
        self.set_debt([self.debt_row(1)])
        self.assertStatus('DEBT-01', common.FAIL, contains='dev and pilot only', ctx=self.ctx(stage='official'))
        self.assertStatus('DEBT-01', common.PASS, ctx=self.ctx(stage='official', dry=True))

    def test_review_or_debt_group(self):
        self.edit_review(reviewer_agent='codex')
        self.f.commit_review(trailers=CODEX, branches=CODEX_BR)
        res = {r.id: r for r in review.review_results(self.ctx())}
        self.assertEqual(res['REV-01'].status, common.FAIL)
        self.assertEqual(res['DEBT-01'].status, common.FAIL)
        self.set_debt([self.debt_row(1)])
        res = {r.id: r for r in review.review_results(self.ctx())}
        self.assertEqual(res['REV-01'].status, common.WARN)
        self.assertIn('covered by DEBT-01', res['REV-01'].detail)
        self.assertEqual(res['DEBT-01'].status, common.PASS)

    def test_debt_02(self):
        self.set_debt([self.debt_row(1)])
        self.assertStatus('DEBT-02', common.PASS, ctx=self.ctx(stage='publish'))
        self.assertStatus('DEBT-02', common.FAIL, contains='pointer stage', ctx=self.ctx(stage='pointer'))
        lock = self.f.read('ENDPOINT_LOCK.json')
        lock['worlds']['official_role_intended'] = 'SEALED_CONFIRM'
        self.f.set_lock(lock)
        self.assertStatus('DEBT-02', common.FAIL, contains='SEALED_CONFIRM', ctx=self.ctx(stage='publish'))
        lock['worlds']['official_role_intended'] = 'MAIN'
        self.f.set_lock(lock)
        self.f.wr('PROJECT_DOCUMENTS/KB/FINDINGS.jsonl', jsonl([
            {'id': 'F-20261105-codex-1', 'kind': 'finding', 'delivery': NAME, 'status': 'proposed'},
            {'id': 'F-20261106-claude-1', 'kind': 'activation', 'refers_to': 'F-20261105-codex-1'}]))
        self.assertStatus('DEBT-02', common.FAIL, contains='activated', ctx=self.ctx(stage='publish'))

    def test_repaid_debt_needs_a_cross_family_review(self):
        self.set_debt([self.debt_row(1), self.debt_row(1, repaid='claude:s-review')])
        self.assertStatus('DEBT-02', common.PASS, ctx=self.ctx(stage='pointer'))
        self.edit_review(reviewer_agent='codex:subagent')
        self.assertStatus('DEBT-02', common.FAIL, ctx=self.ctx(stage='pointer'))

    def test_cmd_review_check(self):
        import minifly_check
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = minifly_check.main(['--git', 'timeline:%s' % self.f.fx, '--repo-root', str(self.f.root), '--json',
                                     'review', 'check', NAME])
        rep = json.loads(out.getvalue())
        sts = {r['id']: r['status'] for r in rep['results']}
        self.assertEqual(rc, 0, sts)
        self.assertEqual(sts['REV-01'], 'PASS')
        self.assertEqual(sts['DEBT-01'], 'NOT_APPLICABLE')


if __name__ == '__main__':
    unittest.main()
