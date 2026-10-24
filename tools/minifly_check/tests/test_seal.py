"""Unit tests for seal.py (WP2): seal commit-nonce/reveal/derive/verify, SEAL-01..05, CONF-01.

Synthetic TimelineGit fixtures in temp dirs; nonce files live in the temp dir, outside the fixture repo.
"""
import contextlib
import hashlib
import hmac
import io
import json
import os
import sys
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for p in (str(KIT), str(TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import common  # noqa: E402
import seal  # noqa: E402
from common import FAIL, NOT_APPLICABLE, PASS, WARN  # noqa: E402
from test_lockcheck import (CODEX, DNAME, TmpCase, jsonl, outputs_rows, run, sha,  # noqa: E402
                            standard_history)

SEALED = {'worlds.official': {'mode': 'sealed', 'ids': [], 'n': 3},
          'worlds.official_role_intended': 'SEALED_CONFIRM'}
REVIEWER = ['origin/claude/seal-x', 'origin/codex/work']
SEAL_CHECKS = ('SEAL-01', 'SEAL-02', 'SEAL-03', 'SEAL-04', 'SEAL-05', 'CONF-01')


def reference_worlds(nonce, verbatim, l1, l2, n):
    """Independent restatement of SPEC 10 step 5 (pins the derivation)."""
    key = hashlib.sha256(nonce.encode('utf-8') + verbatim.encode('utf-8')).digest()
    out = []
    for i in range(n):
        msg = (l1 + '|' + l2 + '|confirm_worlds|' + str(i)).encode('utf-8')
        out.append('w_seal_%016x' % int.from_bytes(hmac.new(key, msg, hashlib.sha256).digest()[:8], 'big'))
    return out


class SealCase(TmpCase):
    def setUp(self):
        super().setUp()
        self.sl = standard_history(self.fx, lock_over=dict(SEALED))
        d = self.fx.repo / DNAME
        self.l1 = sha((d / 'ENDPOINT_LOCK.json').read_bytes())
        self.l2 = sha((d / 'SOURCE_LOCK.json').read_bytes())
        self.nonce_file = self.tmp / 'private' / 'nonce.txt'
        self.nonce_file.parent.mkdir()

    def seal_cmd(self, action, **kw):
        extra = kw.pop('extra', [])
        args = self.fx.args(action=action, dir=DNAME, nonce_file=kw.pop('nonce_file', str(self.nonce_file)),
                            extra=extra, **kw)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = seal.cmd_seal(args)
        return rc, out.getvalue(), err.getvalue()

    def go(self, verbatim=None, shown=None):
        verbatim = verbatim if verbatim is not None else (
            'GO for the official run. ENDPOINT_LOCK %s, SOURCE_LOCK %s. Zhou.' % (self.l1, self.l2))
        auth = {'schema': 'minifly.authorization.v1', 'verbatim': 'implementation ok', 'utc': '20261101T000000Z',
                'stages': {'official': {'granted': True, 'verbatim': verbatim, 'utc': '20261104T000000Z',
                                        'lock_shas_shown': shown if shown is not None else [self.l1, self.l2]}}}
        self.fx.write(DNAME + '/AUTHORIZATION.json', auth)
        return verbatim

    def full_seal(self):
        rc, out, err = self.seal_cmd('commit-nonce', extra=['--agent', 'claude', '--session', 's9'])
        self.assertEqual(rc, 0, err)
        self.fx.commit(branches=REVIEWER, trailers={'Acting-Agent': ['claude:s9']})
        verb = self.go()
        self.fx.commit(branches=CODEX)
        rc, out, err = self.seal_cmd('reveal')
        self.assertEqual(rc, 0, err)
        self.fx.commit(branches=REVIEWER, trailers={'Acting-Agent': ['claude:s9']})
        return verb

    def ctx(self, **kw):
        kw.setdefault('stage', 'official')
        kw.setdefault('tier', 2)
        return self.fx.ctx(**kw)

    def assertSeal(self, ctx, want):
        bad = []
        for cid in SEAL_CHECKS:
            for r in run(ctx, cid):
                if r.status != want:
                    bad.append((cid, r.status, r.detail))
        self.assertEqual(bad, [])


class TestDerivation(unittest.TestCase):
    def test_reference_vector(self):
        nonce = '0f' * 32
        got = seal.derive_world_id_list(nonce, 'go ' + 'a' * 64, 'a' * 64, 'b' * 64, 4)
        self.assertEqual(got, reference_worlds(nonce, 'go ' + 'a' * 64, 'a' * 64, 'b' * 64, 4))
        self.assertEqual(len(set(got)), 4)
        self.assertNotEqual(seal.derive_seeds(nonce, 'x', 'a' * 64, 'b' * 64, 'udi', 1),
                            seal.derive_seeds(nonce, 'x', 'a' * 64, 'b' * 64, 'replay', 1))
        self.assertEqual(seal.nonce_sha256(nonce), hashlib.sha256(nonce.encode()).hexdigest())

    def test_alpha_schedule(self):
        f, spec = seal.alpha_schedule({'confirm_alpha_spending': 'geometric_0.05_2^-j'})
        self.assertEqual([f(0), f(1), f(2)], [0.05, 0.025, 0.0125])
        f, _ = seal.alpha_schedule({})
        self.assertEqual(f(3), 0.05 / 8)
        with self.assertRaises(common.InvalidInput):
            seal.alpha_schedule({'confirm_alpha_spending': 'bonferroni'})

    def test_default_nonce_path_outside_repo(self):
        p = seal.default_nonce_path(DNAME, 'abcdef0123' + '0' * 54)
        self.assertEqual(p.name, '.minifly_nonce_%s_abcdef01' % DNAME)
        self.assertEqual(p.parent, Path.home())


class TestSealRoundTrip(SealCase):
    def test_round_trip_and_official_rows(self):
        verb = self.full_seal()
        sc = self.fx.read_json(DNAME + '/review/SEAL_COMMIT.json')
        nonce = self.nonce_file.read_text().strip()
        self.assertEqual(sc['nonce_sha256'], hashlib.sha256(nonce.encode()).hexdigest())
        self.assertEqual(sc['endpoint_lock_sha256'], self.l1)
        self.assertEqual(sc['reviewer_agent'], 'claude')
        self.assertEqual(oct(self.nonce_file.stat().st_mode & 0o777), '0o600')
        self.assertNotIn(nonce, json.dumps(sc))
        sr = self.fx.read_json(DNAME + '/review/SEAL_REVEAL.json')
        self.assertEqual(sr['nonce'], nonce)
        rows = [json.loads(x) for x in self.fx.read('PROJECT_DOCUMENTS/KB/CONFIRM_LEDGER.jsonl').decode().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['attempt_index'], 0)
        self.assertEqual(rows[0]['alpha_allocated'], 0.05)
        self.assertEqual(rows[0]['delivery'], DNAME)
        rc, out, err = self.seal_cmd('derive')
        self.assertEqual(rc, 0, err)
        derived = json.loads(out)
        want = reference_worlds(nonce, verb, self.l1, self.l2, 3)
        self.assertEqual(derived['confirm_worlds'], want)
        rc, out, err = self.seal_cmd('verify')
        self.assertEqual(rc, 0, out + err)
        self.assertSeal(self.ctx(), PASS)
        # official outputs on exactly the derived worlds, committed after the reveal
        for w in want:
            self.fx.write(DNAME + '/results/receipts/%s/W.json' % w,
                          {'world': w, 'world_role': 'official', 'arm': 'W',
                           'native_closure_sha': self.sl['native_closure_sha']})
        self.fx.write(DNAME + '/results/official/OUTPUTS.jsonl',
                      jsonl(sum([outputs_rows(self.sl, world=w, role='official') for w in want], [])))
        self.fx.commit()
        self.assertSeal(self.ctx(), PASS)
        # AMD-04 sees seal-derived ids: one leaking into another delivery fails
        self.fx.write('minifly_leak_dev_v1_20261020/notes.json', {'world': want[1]})
        res = run(self.ctx(), 'AMD-04')
        self.assertEqual(res[0].status, FAIL, res[0].detail)
        self.assertIn(want[1], res[0].detail)

    def test_derived_world_mismatch_fails(self):
        self.full_seal()
        self.fx.write(DNAME + '/results/official/OUTPUTS.jsonl',
                      jsonl(outputs_rows(self.sl, world='w_hand_picked', role='official')))
        self.fx.commit()
        res = run(self.ctx(), 'SEAL-04')
        self.assertEqual(res[0].status, FAIL, res[0].detail)
        self.assertIn('non-derived world(s) w_hand_picked', res[0].detail)

    def test_listed_ids_and_argv_must_match_derived(self):
        self.full_seal()
        ctx = self.ctx(options={'pending_argv': ['python3', 'run.py', '--worlds', '111,222']})
        res = run(ctx, 'SEAL-04')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('argv names non-derived', res[0].detail)
        lock = self.fx.read_json(DNAME + '/ENDPOINT_LOCK.json')
        lock['worlds']['official']['mode'] = 'listed'
        self.fx.write(DNAME + '/ENDPOINT_LOCK.json', lock)
        res = run(self.ctx(), 'SEAL-04')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('must be sealed', res[0].detail)

    def test_conf01_alpha_spent_by_family(self):
        ctx = self.ctx()
        fam = common.family_of(ctx)[0]
        self.fx.write('PROJECT_DOCUMENTS/KB/CONFIRM_LEDGER.jsonl', jsonl([
            {'family_id': fam, 'delivery': 'minifly_prev_dev_v1_20261001', 'attempt_index': 0, 'alpha_allocated': 0.05,
             'seal_reveal_commit': None, 'utc': '20261001T000000Z'}]))
        self.fx.commit()
        dry = run(self.ctx(dry=True), 'CONF-01')
        self.assertEqual(dry[0].status, FAIL, dry[0].detail)  # 0.05 > prospective 0.025, even before the seal
        self.full_seal()
        rows = [json.loads(x) for x in self.fx.read('PROJECT_DOCUMENTS/KB/CONFIRM_LEDGER.jsonl').decode().splitlines()]
        self.assertEqual([r['attempt_index'] for r in rows], [0, 1])
        self.assertEqual(rows[1]['alpha_allocated'], 0.025)
        res = run(self.ctx(), 'CONF-01')
        self.assertEqual(res[0].status, FAIL, res[0].detail)
        self.assertIn('exceeds alpha_allocated 0.025', res[0].detail)

    def test_conf01_requires_alpha_aggregator(self):
        lock = self.fx.read_json(DNAME + '/ENDPOINT_LOCK.json')
        lock['primary']['aggregator'] = {'type': 'k_of_n', 'k': 6}
        self.fx.write(DNAME + '/ENDPOINT_LOCK.json', lock)
        res = run(self.ctx(), 'CONF-01')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('alpha-bearing aggregator', res[0].detail)


class TestSealOrdering(SealCase):
    def test_reveal_before_go_fails(self):
        rc, _o, err = self.seal_cmd('commit-nonce', extra=['--agent', 'claude'])
        self.assertEqual(rc, 0, err)
        self.fx.commit(branches=REVIEWER)
        nonce = self.nonce_file.read_text().strip()
        self.fx.write(DNAME + '/review/SEAL_REVEAL.json',
                      {'schema': seal.SCHEMA_SEAL_REVEAL, 'nonce': nonce, 'utc': '20261104T000000Z'})
        self.fx.commit(branches=REVIEWER)
        self.go()
        self.fx.commit(branches=CODEX)
        res = run(self.ctx(), 'SEAL-02')
        self.assertEqual(res[0].status, FAIL, res[0].detail)
        self.assertIn('reveal before GO', res[0].detail)

    def test_reveal_command_refuses_before_committed_go(self):
        rc, _o, err = self.seal_cmd('commit-nonce', extra=['--agent', 'claude'])
        self.assertEqual(rc, 0, err)
        # round 1 (SPEC 20.15): the reviewer's seal commit carries its Acting-Agent trailer, as a real one does
        self.fx.commit(branches=REVIEWER, trailers={'Acting-Agent': ['claude:s9']})
        rc, _o, err = self.seal_cmd('reveal')
        self.assertEqual(rc, common.EXIT_FAIL)
        self.assertIn('never reveal before GO', err)
        self.go()  # written but not committed after SEAL_COMMIT
        rc, _o, err = self.seal_cmd('reveal')
        self.assertEqual(rc, common.EXIT_FAIL)
        self.assertIn('reveal before GO', err)
        self.assertFalse((self.fx.repo / DNAME / 'review' / 'SEAL_REVEAL.json').exists())

    def test_reveal_commit_must_act_as_the_reviewer(self):
        # SPEC 20.15 (round 1): the reveal is a reviewer step; the author committing SEAL_REVEAL fails SEAL-02
        rc, out, err = self.seal_cmd('commit-nonce', extra=['--agent', 'claude', '--session', 's9'])
        self.assertEqual(rc, 0, err)
        self.fx.commit(branches=REVIEWER, trailers={'Acting-Agent': ['claude:s9']})
        self.go()
        self.fx.commit(branches=CODEX)
        rc, out, err = self.seal_cmd('reveal')
        self.assertEqual(rc, 0, err)
        self.fx.commit(branches=CODEX, trailers={'Acting-Agent': ['codex']})
        res = run(self.ctx(), 'SEAL-02')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('SEAL_REVEAL commit', res[0].detail)
        self.assertIn('not the reviewer family claude', res[0].detail)

    def test_seal_commit_outside_reviewer_namespace_fails(self):
        rc, _o, err = self.seal_cmd('commit-nonce', extra=['--agent', 'claude'])
        self.assertEqual(rc, 0, err)
        # a reviewer trailer on a commit that only the author namespace contains (forged identity, SPEC 20.15)
        self.fx.commit(branches=CODEX, trailers={'Acting-Agent': ['claude:s9']})
        self.go()
        self.fx.commit(branches=CODEX)
        rc, _o, err = self.seal_cmd('reveal')
        self.assertEqual(rc, common.EXIT_FAIL)
        self.assertIn('origin/claude/*', err)
        res = run(self.ctx(), 'SEAL-02')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('origin/claude/*', res[0].detail)

    def test_go_must_name_lock_shas_and_nonce_must_match(self):
        self.full_seal()
        self.go(verbatim='go', shown=[])
        self.fx.commit(branches=CODEX)
        res = run(self.ctx(), 'SEAL-03')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('does not name the ENDPOINT_LOCK sha', res[0].detail)
        self.go()
        sr = self.fx.read_json(DNAME + '/review/SEAL_REVEAL.json')
        sr['nonce'] = '1' * 64
        self.fx.write(DNAME + '/review/SEAL_REVEAL.json', sr)
        res = run(self.ctx(), 'SEAL-03')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('does not match SEAL_COMMIT.nonce_sha256', res[0].detail)

    def test_reveal_after_official_outputs_and_ledger_order(self):
        rc, _o, err = self.seal_cmd('commit-nonce', extra=['--agent', 'claude'])
        self.assertEqual(rc, 0, err)
        self.fx.commit(branches=REVIEWER, trailers={'Acting-Agent': ['claude:s9']})
        self.go()
        self.fx.commit(branches=CODEX)
        self.fx.write(DNAME + '/results/official/OUTPUTS.jsonl', jsonl(outputs_rows(self.sl, world='x', role='official')))
        self.fx.commit()
        rc, _o, err = self.seal_cmd('reveal')
        self.assertEqual(rc, common.EXIT_FAIL)
        self.assertIn('official outputs already exist', err)
        res = run(self.ctx(), 'SEAL-02')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('without a committed SEAL_REVEAL', res[0].detail)
        res = run(self.ctx(), 'SEAL-05')
        self.assertEqual(res[0].status, FAIL)


class TestSealRefusalsAndApplicability(SealCase):
    def test_not_applicable_unless_sealed_confirm(self):
        lock = self.fx.read_json(DNAME + '/ENDPOINT_LOCK.json')
        lock['worlds']['official_role_intended'] = 'MAIN'
        self.fx.write(DNAME + '/ENDPOINT_LOCK.json', lock)
        self.assertSeal(self.ctx(), NOT_APPLICABLE)
        rc, _o, err = self.seal_cmd('commit-nonce', extra=['--agent', 'claude'])
        self.assertEqual(rc, common.EXIT_FAIL)
        self.fx.write(DNAME + '/CLAIM.json', {'worlds_role': 'SEALED_CONFIRM'})
        res = run(self.ctx(), 'SEAL-01')
        self.assertEqual(res[0].status, FAIL)  # a claimed SEALED_CONFIRM without a seal

    def test_missing_seal_fails_and_dry_defers(self):
        res = run(self.ctx(), 'SEAL-01')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('SEAL_COMMIT.json absent', res[0].detail)
        for cid in SEAL_CHECKS:
            r = run(self.ctx(dry=True), cid)[0]
            self.assertEqual(r.status, WARN, (cid, r.detail))
            self.assertIn('deferred', r.detail)

    def test_nonce_path_inside_repo_and_same_family_refused(self):
        rc, _o, err = self.seal_cmd('commit-nonce', extra=['--agent', 'claude'],
                                    nonce_file=str(self.fx.repo / 'nonce.txt'))
        self.assertEqual(rc, common.EXIT_FAIL)
        self.assertIn('inside the repository', err)
        self.assertFalse((self.fx.repo / 'nonce.txt').exists())
        rc, _o, err = self.seal_cmd('commit-nonce', extra=['--agent', 'codex'])
        self.assertEqual(rc, common.EXIT_FAIL)
        self.assertIn('single-agent mode cannot seal', err)
        rc, _o, err = self.seal_cmd('commit-nonce', extra=['--bogus'])
        self.assertEqual(rc, common.EXIT_ERROR)
        old = os.environ.pop('MINIFLY_AGENT', None)
        try:
            rc, _o, err = self.seal_cmd('commit-nonce')
            self.assertEqual(rc, common.EXIT_MISSING)
        finally:
            if old is not None:
                os.environ['MINIFLY_AGENT'] = old

    def test_commit_nonce_needs_reviewer_attestation_of_both_shas(self):
        rv = self.fx.read_json(DNAME + '/review/REVIEW.json')
        rv['source_lock_acks'] = []
        self.fx.write(DNAME + '/review/REVIEW.json', rv)
        rc, _o, err = self.seal_cmd('commit-nonce', extra=['--agent', 'claude'])
        self.assertEqual(rc, common.EXIT_FAIL)
        self.assertIn('must first attest', err)
        self.assertFalse(self.nonce_file.exists())

    def test_existing_nonce_file_never_overwritten(self):
        self.nonce_file.write_text('keep\n')
        rc, _o, err = self.seal_cmd('commit-nonce', extra=['--agent', 'claude'])
        self.assertEqual(rc, common.EXIT_FAIL)
        self.assertEqual(self.nonce_file.read_text(), 'keep\n')

    def test_lock_change_after_seal_commit(self):
        self.full_seal()
        lock = self.fx.read_json(DNAME + '/ENDPOINT_LOCK.json')
        lock['primary']['margin_abs'] = 0.06
        self.fx.write(DNAME + '/ENDPOINT_LOCK.json', lock)
        res = run(self.ctx(), 'SEAL-03')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('is not the current ENDPOINT_LOCK', res[0].detail)


if __name__ == '__main__':
    unittest.main()
