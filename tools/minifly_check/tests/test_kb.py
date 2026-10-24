"""Unit tests for kb.py (WP5): ledger schemas and append, the FINDINGS fold and R-levels, the E3 cadence
counter, kb status/score/backtest and KB-01..05. Synthetic TimelineGit fixtures in temp dirs only."""
import argparse
import contextlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
KIT = HERE.parent
for p in (str(KIT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import common  # noqa: E402
import kb as kbmod  # noqa: E402
from common import FAIL, MISSING, NOT_APPLICABLE, PASS  # noqa: E402
from test_claimcheck import (CONSTRAINTS, KB, LEDGERS, NAME, NARROWED, SEED_FINDING, Fixture, TmpCase,  # noqa: E402
                             build, jsonl, quiet, run, status)

FIND = {'id': 'F-20261105-codex-1', 'kind': 'finding', 'refers_to': None, 'claim': 'W beats NPT on held S3',
        'level': 'E3', 'conditions': {'best_shortcut': 'CANON'}, 'effect': {'value': 0.4, 'ci': [0.3, 0.5]},
        'source': NAME + '/native.py:1@abc', 'status': 'proposed', 'delivery': NAME, 'author_agent': 'codex',
        'utc': '20261105T000000Z', 'caveat': None}


def act(fid='F-20261106-codex-1', author='codex', target='F-20261105-codex-1', **over):
    r = {'id': fid, 'kind': 'activation', 'refers_to': target, 'delivery': NAME, 'author_agent': author,
         'utc': '20261106T000000Z', 'evidence': 'review/REDERIVE.md', 'status': 'active'}
    r.update(over)
    return r


def ledger_row(delivery, utc, **over):
    r = {'delivery': delivery, 'utc': utc, 'legacy': False, 'task_family': 'addition_single_digit',
         'level_earned': 'E1', 'primary_gate': 'FAIL', 'gate_type': 'component', 'worlds_role': 'DEV',
         'opened_held_E3': False, 'e3_basis': [], 'objective_metric': {'held': None, 'old_final': None, 'G': None},
         'family_id': 'FAM-1a2b3c4d', 'claim_sha256': 'c' * 64}
    r.update(over)
    return r


E3_ROW = dict(primary_gate='PASS', worlds_role='MAIN', opened_held_E3=True, level_earned='E3', gate_type='objective',
              e3_basis=['pool HELD', 'strata S2-S4', 'admission PASS', 'worlds_role MAIN', 'verdict PASS'])


def kb_only(tmp, rows=None, constraints=None):
    """A repo holding only PROJECT_DOCUMENTS/KB (all seven ledgers)."""
    fx = Fixture(Path(tmp) / 'kbfx')
    fx.put(KB + '/CONSTRAINTS.json', constraints or CONSTRAINTS)
    for led in LEDGERS:
        fx.put('%s/%s.jsonl' % (KB, led), jsonl((rows or {}).get(led, [])))
    fx.commit('k1')
    fx.write()
    return fx


def snapshot(fx, root, label='s1', trailers=None):
    """Copy fx.repo into a new fixture committed as one commit."""
    f2 = Fixture(root)
    for q in fx.repo.rglob('*'):
        if q.is_file():
            f2.files[q.relative_to(fx.repo).as_posix()] = q.read_bytes()
    f2.commit(label, trailers=trailers or {'Acting-Agent': ['codex']})
    return f2


def add_findings(fx, rows, label, trailers):
    rel = KB + '/FINDINGS.jsonl'
    fx.files[rel] = fx.files[rel] + jsonl(rows)
    fx.commit(label, [rel], trailers=trailers, utc='20261106T000000Z')


class TestSchemas(TmpCase):
    def test_seed_like_rows_valid(self):
        for r in [SEED_FINDING] + NARROWED + [FIND]:
            self.assertEqual(kbmod.row_problems('FINDINGS', r), [], r['id'])
        self.assertEqual(kbmod.row_problems('DELIVERY_LEDGER', ledger_row('d_dev_v1_20261101', '20261101T000000Z')), [])
        legacy = {'delivery': 'old', 'utc': '20261024T000000Z', 'legacy': True, 'level_earned': 'legacy',
                  'primary_gate': 'legacy', 'gate_type': None, 'worlds_role': None, 'opened_held_E3': 'unknown',
                  'e3_basis': [], 'objective_metric': {'held': None, 'old_final': None, 'G': None}, 'family_id': None,
                  'claim_sha256': None}
        self.assertEqual(kbmod.row_problems('DELIVERY_LEDGER', legacy), [])
        self.assertTrue(kbmod.row_problems('DELIVERY_LEDGER', dict(legacy, opened_held_E3=True)))

    def test_bad_rows(self):
        self.assertTrue(kbmod.row_problems('FINDINGS', dict(FIND, kind='rumour')))
        self.assertTrue(kbmod.row_problems('FINDINGS', dict(FIND, status='great')))
        self.assertTrue(kbmod.row_problems('FINDINGS', dict(FIND, id='F-1')))
        self.assertTrue(kbmod.row_problems('USER_DECISIONS', {'id': 'UD-20261101-1', 'utc': '20261101T000000Z',
                                                              'verbatim': '', 'scope': []}))
        self.assertTrue(kbmod.row_problems('DELIVERY_LEDGER', ledger_row('d', '20261101T000000Z', worlds_role='MAINISH')))

    def test_kb01_check(self):
        fx = kb_only(self.tmp, {'FINDINGS': [SEED_FINDING, FIND]})
        ctx = fx.ctx(delivery=None)
        self.assertEqual(status(ctx, 'KB-01'), [PASS])
        fx = kb_only(self.tmp / 'b', {'FINDINGS': [SEED_FINDING, dict(FIND, kind='rumour')]})
        self.assertEqual(status(fx.ctx(delivery=None), 'KB-01'), [FAIL])
        fx = kb_only(self.tmp / 'c', {'FINDINGS': [dict(NARROWED[1], refers_to='F-UF-77')]})
        self.assertEqual(status(fx.ctx(delivery=None), 'KB-01'), [FAIL])
        self.assertEqual(status(fx.ctx(delivery=None), 'KB-01'), [FAIL])  # FAIL outranks the MISSING below
        fx = kb_only(self.tmp / 'd', {'FINDINGS': [SEED_FINDING]})
        (fx.repo / KB / 'REVIEW_DEBT.jsonl').unlink()
        self.assertEqual(status(fx.ctx(delivery=None), 'KB-01'), [MISSING])

    def test_duplicate_ids_fail(self):
        fx = kb_only(self.tmp, {'FINDINGS': [FIND, dict(FIND, claim='different')]})
        self.assertEqual(status(fx.ctx(delivery=None), 'KB-02'), [FAIL])


class TestAppendOnly(TmpCase):
    def test_kb02_history(self):
        fx = Fixture(self.tmp / 'fx')
        fx.put(KB + '/CONSTRAINTS.json', CONSTRAINTS)
        for led in LEDGERS:
            fx.put('%s/%s.jsonl' % (KB, led), b'')
        fx.put(KB + '/FINDINGS.jsonl', jsonl([SEED_FINDING, FIND]))
        fx.commit('k1')
        fx.put(KB + '/FINDINGS.jsonl', jsonl([SEED_FINDING, FIND, NARROWED[0]]))
        fx.commit('k2', [KB + '/FINDINGS.jsonl'])
        fx.write()
        self.assertEqual(status(fx.ctx(delivery=None), 'KB-02'), [PASS])
        fx.put(KB + '/FINDINGS.jsonl', jsonl([SEED_FINDING, NARROWED[0]]))  # FIND deleted in a commit
        fx.commit('k3', [KB + '/FINDINGS.jsonl'])
        fx.write()
        res = run(fx.ctx(delivery=None), 'KB-02')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('removed or edited 1 line', res[0].detail)

    def test_kb02_worktree_and_shallow(self):
        fx = kb_only(self.tmp, {'FINDINGS': [SEED_FINDING, FIND]})
        (fx.repo / KB / 'FINDINGS.jsonl').write_bytes(jsonl([SEED_FINDING, dict(FIND, claim='edited')]))
        self.assertEqual(status(fx.ctx(delivery=None), 'KB-02'), [FAIL])
        fx = kb_only(self.tmp / 'b', {'FINDINGS': [SEED_FINDING]})
        fx.write(shallow=True)
        self.assertEqual(status(fx.ctx(delivery=None), 'KB-02'), [MISSING])


class TestActivation(TmpCase):
    def _published(self, extra_findings=()):
        """The C00a-like delivery after `claim` (T2, MAIN, DELIVERY_LEDGER-linked), committed by codex."""
        fx = build(self.tmp / 'base', kb_rows={'FINDINGS': [SEED_FINDING] + NARROWED + [FIND] + list(extra_findings)})
        cp = fx.repo / NAME / 'CLAIM.json'
        with quiet(), mock.patch('claimcheck._check_status', return_value=('PASS', 'stub')):
            import claimcheck
            claimcheck.cmd_claim(fx.args(dir=str(fx.repo / NAME)))
        self.assertTrue(cp.is_file())
        return snapshot(fx, self.tmp / 'pub')

    def test_self_activation_fails(self):
        f2 = self._published()
        add_findings(f2, [act(author='codex')], 'a1', {'Acting-Agent': ['codex']})
        f2.write()
        res = run(f2.ctx(), 'KB-03')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('same family codex', res[0].detail)

    def test_self_declared_active_fails(self):
        fx = build(self.tmp, kb_rows={'FINDINGS': [SEED_FINDING, dict(FIND, status='active')]})
        res = run(fx.ctx(), 'KB-03')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('self-activation', res[0].detail)
        st = kbmod.finding_state(fx.ctx().kb, FIND['id'])
        self.assertEqual(st['status'], 'provisional')

    def test_other_family_activation(self):
        f2 = self._published()
        add_findings(f2, [act(fid='F-20261106-claude-1', author='claude:s1')], 'a1', {'Acting-Agent': ['claude:s1']})
        f2.write()
        ctx = f2.ctx()
        self.assertEqual(status(ctx, 'KB-03'), [PASS])
        res = run(ctx, 'KB-04')
        self.assertEqual([r.status for r in res], [PASS], res[0].detail)
        st = kbmod.finding_state(ctx.kb, FIND['id'], ctx.git, ctx.rules)
        self.assertEqual((st['status'], st['r_level']), ('active', 'R1'))

    def test_activation_trailer_mismatch_and_uncommitted(self):
        f2 = self._published()
        add_findings(f2, [act(fid='F-20261106-claude-1', author='claude:s1')], 'a1', {'Acting-Agent': ['codex']})
        f2.write()
        res = run(f2.ctx(), 'KB-03')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('Acting-Agent codex', res[0].detail)
        f3 = self._published()
        f3.write()
        p = f3.repo / KB / 'FINDINGS.jsonl'
        p.write_bytes(p.read_bytes() + jsonl([act(fid='F-20261106-claude-1', author='claude:s1')]))
        self.assertEqual(status(f3.ctx(), 'KB-03'), [MISSING])

    def test_kb04_open_debt_and_dev_source(self):
        f2 = self._published()
        add_findings(f2, [act(fid='F-20261106-claude-1', author='claude:s1')], 'a1', {'Acting-Agent': ['claude:s1']})
        rel = KB + '/REVIEW_DEBT.jsonl'
        f2.files[rel] = jsonl([{'id': 'DEBT-20261102-1', 'delivery': NAME, 'utc': '20261102T000000Z',
                                'reason': 'codex unavailable', 'recorded_by': 'claude:s', 'repaid_by': None}])
        f2.commit('d1', [rel])
        f2.write()
        res = run(f2.ctx(), 'KB-04')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('open review debt', res[0].detail)
        # activation of a seed finding: never a T2 MAIN source
        f3 = self._published()
        add_findings(f3, [act(fid='F-20261106-codex-1', author='codex', target='F-UF-01', delivery=NAME)], 'a1',
                     {'Acting-Agent': ['codex']})
        f3.write()
        res = run(f3.ctx(), 'KB-04')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('not a v1 delivery', res[0].detail)

    def test_no_activation_rows(self):
        fx = build(self.tmp)
        self.assertEqual(status(fx.ctx(), 'KB-03'), [NOT_APPLICABLE])
        self.assertEqual(status(fx.ctx(), 'KB-04'), [NOT_APPLICABLE])


class TestIncidents(TmpCase):
    REPORT = {'id': 'I-20261101-1', 'kind': 'report', 'fm_class': 'FM-26', 'delivery': NAME, 'symptom': 's',
              'evidence': 'x.md:1', 'found_by_agent': 'claude', 'found_utc': '20261101T000000Z',
              'caught_at_discovery': False, 'checker_sha_at_discovery': None, 'heldout_eligible': False,
              'check_id': None, 'fixture_id': None, 'status': 'open'}

    def closure(self, **over):
        r = {'id': 'I-20261101-1', 'kind': 'closure', 'fm_class': 'FM-26', 'delivery': NAME, 'status': 'closed',
             'check_id': 'CLM-10', 'fixture_id': None, 'utc': '20261102T000000Z'}
        r.update(over)
        return r

    def test_closure_without_fixture_fails(self):
        fx = kb_only(self.tmp, {'INCIDENTS': [self.REPORT, self.closure()]})
        res = run(fx.ctx(delivery=None), 'KB-05')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('no fixture_id', res[0].detail)

    def test_closure_with_fixture(self):
        fx = kb_only(self.tmp, {'INCIDENTS': [self.REPORT, self.closure(fixture_id='C26')]})
        corpus = self.tmp / 'corpus'
        (corpus / 'C26').mkdir(parents=True)
        with mock.patch.object(kbmod, 'kit_expected', return_value={'C26': {'must_fire': ['CLM-10', 'KB-05']}}), \
                mock.patch.object(kbmod, 'corpus_dir', return_value=corpus):
            self.assertEqual(status(fx.ctx(delivery=None), 'KB-05'), [PASS])
            self.assertEqual(kbmod.open_incidents(fx.ctx(delivery=None).kb), [])
        with mock.patch.object(kbmod, 'kit_expected', return_value={'C26': {'must_fire': ['KB-05']}}), \
                mock.patch.object(kbmod, 'corpus_dir', return_value=corpus):
            res = run(fx.ctx(delivery=None), 'KB-05')
            self.assertEqual(res[0].status, FAIL)
            self.assertIn('must_fire does not include CLM-10', res[0].detail)
            self.assertEqual(len(kbmod.open_incidents(fx.ctx(delivery=None).kb)), 1)
        with mock.patch.object(kbmod, 'kit_expected', side_effect=common.MissingInput('no EXPECTED.json')):
            self.assertEqual(status(fx.ctx(delivery=None), 'KB-05'), [MISSING])
        fx = kb_only(self.tmp / 'b', {'INCIDENTS': [self.REPORT, self.closure(check_id='NOPE-01', fixture_id='C26')]})
        self.assertEqual(status(fx.ctx(delivery=None), 'KB-05'), [FAIL])

    def test_no_closures(self):
        fx = kb_only(self.tmp, {'INCIDENTS': [self.REPORT]})
        self.assertEqual(status(fx.ctx(delivery=None), 'KB-05'), [NOT_APPLICABLE])


class TestCadence(TmpCase):
    def cadence(self, rows, constraints=None):
        fx = kb_only(self.tmp, {'DELIVERY_LEDGER': rows}, constraints)
        return kbmod.deliveries_since_last_e3(fx.ctx(delivery=None).kb)

    def test_reset_and_count(self):
        rows = [ledger_row('a_dev_v1_20261101', '20261101T000000Z', **E3_ROW),
                ledger_row('b_dev_v1_20261102', '20261102T000000Z'),
                ledger_row('c_dev_v1_20261103', '20261103T000000Z')]
        self.assertEqual(self.cadence(rows), 2)
        self.assertEqual(self.cadence(rows[:1]), 0)

    def test_cannot_test_does_not_reset(self):
        rows = [ledger_row('a_dev_v1_20261101', '20261101T000000Z'),
                ledger_row('b_dev_v1_20261102', '20261102T000000Z',
                           **dict(E3_ROW, primary_gate='CANNOT_TEST_CLAIM_AS_SPECIFIED', level_earned='NONE')),
                ledger_row('c_dev_v1_20261103', '20261103T000000Z')]
        self.assertEqual(self.cadence(rows), 3)
        self.assertFalse(kbmod.opened_held_e3_valid(rows[1])[0])

    def test_basis_must_show_the_opening(self):
        no_adm = dict(E3_ROW, e3_basis=['pool HELD', 'strata S2-S4', 'ADM-EXPO PASS'])
        s0 = dict(E3_ROW, e3_basis=['pool HELD', 'strata S0-S4', 'admission PASS'])
        dev = dict(E3_ROW, worlds_role='DEV')
        for bad in (no_adm, s0, dev):
            rows = [ledger_row('a_dev_v1_20261101', '20261101T000000Z', **bad)]
            self.assertEqual(self.cadence(rows), 1, bad)

    def test_legacy_rows_and_reclaims(self):
        legacy = {'delivery': 'old', 'utc': '20261103T000000Z', 'legacy': True, 'level_earned': 'legacy',
                  'primary_gate': 'legacy', 'gate_type': None, 'worlds_role': None, 'opened_held_E3': 'unknown',
                  'e3_basis': [], 'objective_metric': {'held': None, 'old_final': None, 'G': None}, 'family_id': None,
                  'claim_sha256': None}
        rows = [ledger_row('a_dev_v1_20261101', '20261101T000000Z', **E3_ROW), legacy]
        self.assertEqual(self.cadence(rows), 0)
        self.assertEqual(self.cadence(rows, dict(CONSTRAINTS, e3_cadence_counts_legacy=True)), 1)
        # a delivery that claims twice counts once, at its first claim; its latest row decides
        rows = [ledger_row('b_dev_v1_20261101', '20261101T000000Z'),
                ledger_row('a_dev_v1_20261102', '20261102T000000Z', **E3_ROW),
                ledger_row('b_dev_v1_20261101', '20261104T000000Z')]
        self.assertEqual(self.cadence(rows), 0)


class TestFold(TmpCase):
    def test_order_independent_status_and_caveats(self):
        rows = [SEED_FINDING] + NARROWED
        for seed in range(5):
            shuffled = list(rows)
            random.Random(seed).shuffle(shuffled)
            fx = kb_only(self.tmp / str(seed), {'FINDINGS': shuffled})
            st = kbmod.finding_state(fx.ctx(delivery=None).kb, 'F-UF-15')
            self.assertEqual(st['status'], 'narrowed')
            self.assertEqual(st['narrowing_rows'], ['F-20261024-claude-1'])
            self.assertEqual(st['caveats'], ['FIXED_1 10.645% vs STOP 11.263% on NEW'])
        retr = {'id': 'F-20261107-claude-1', 'kind': 'retraction', 'refers_to': 'F-UF-15', 'claim': 'retracted',
                'caveat': 'wrong', 'delivery': 'seed', 'author_agent': 'claude', 'utc': '20261107T000000Z'}
        fx = kb_only(self.tmp / 'r', {'FINDINGS': [retr] + rows})
        self.assertEqual(kbmod.finding_status(fx.ctx(delivery=None).kb, 'F-UF-15'), 'retracted')
        self.assertIsNone(kbmod.finding_status(fx.ctx(delivery=None).kb, 'F-UF-99'))

    def test_r_levels_from_transfers(self):
        t_ok = {'id': 'F-20261108-checker-1', 'kind': 'transfer', 'refers_to': 'F-UF-01', 'prediction_id': 'P1',
                'delivery': 'later_dev_v1_20261108', 'outcome': 'ok', 'task_family': 'addition_single_digit',
                'author_agent': 'checker', 'utc': '20261108T000000Z'}
        rec = {'id': 'F-20261108-codex-1', 'kind': 'reconciliation', 'refers_to': 'F-UF-01', 'status': 'CONFIRMED',
               'evidence': 'review/RECON.md', 'delivery': 'seed', 'author_agent': 'codex', 'utc': '20261108T000000Z',
               'holds_under': 'rederived_metric'}
        rows = {'FINDINGS': [SEED_FINDING, rec],
                'DELIVERY_LEDGER': []}
        fx = kb_only(self.tmp / 'a', rows)
        self.assertEqual(kbmod.finding_state(fx.ctx(delivery=None).kb, 'F-UF-01')['r_level'], 'R2')
        rows['FINDINGS'] = [SEED_FINDING, rec, t_ok]
        fx = kb_only(self.tmp / 'b', rows)
        st = kbmod.finding_state(fx.ctx(delivery=None).kb, 'F-UF-01')
        self.assertEqual((st['r_level'], st['transferable']), ('R3', True))
        rows['FINDINGS'] = [dict(SEED_FINDING, task_family='addition_multi_digit'), rec, t_ok]
        fx = kb_only(self.tmp / 'c', rows)
        self.assertEqual(kbmod.finding_state(fx.ctx(delivery=None).kb, 'F-UF-01')['r_level'], 'R4')


def claim_with(p, gate, guard=None):
    ps = [{'id': 'P1', 'target': 'primary', 'observed': 0.5, 'hit': True, 'interval_score': 0.2, 'brier': None,
           'status': 'scored', 'p_pass': p, 'outcome': 1 if gate == 'PASS' else 0}]
    if guard is not None:
        ps.append({'id': 'P2', 'target': 'guard:old_drop', 'observed': 0.02, 'hit': guard, 'interval_score': 0.1,
                   'brier': None, 'status': 'scored', 'p_pass': 0.9, 'outcome': 1})
    return {'schema': 'minifly.claim.v1', 'primary_gate': gate, 'predictions_scored': ps}


class TestScore(TmpCase):
    def three(self):
        fx = Fixture(self.tmp / 'fx')
        fx.put(KB + '/CONSTRAINTS.json', CONSTRAINTS)
        rows = []
        for i, (d, p, gate, guard) in enumerate((('a_dev_v1_20261101', 0.8, 'PASS', True),
                                                 ('b_dev_v1_20261102', 0.3, 'FAIL', False),
                                                 ('c_dev_v1_20261103', 0.6, 'PASS', True))):
            data = fx.put(d + '/CLAIM.json', claim_with(p, gate, guard))
            rows.append(ledger_row(d, '2026110%dT000000Z' % (i + 1), primary_gate=gate,
                                   claim_sha256=common.sha256_bytes(data)))
        for led in LEDGERS:
            fx.put('%s/%s.jsonl' % (KB, led), b'')
        fx.put(KB + '/DELIVERY_LEDGER.jsonl', jsonl(rows))
        fx.put(KB + '/FINDINGS.jsonl', jsonl([SEED_FINDING, {
            'id': 'F-20261104-checker-1', 'kind': 'transfer', 'refers_to': 'F-UF-01', 'prediction_id': 'P1',
            'delivery': 'b_dev_v1_20261102', 'outcome': 'failed', 'author_agent': 'checker', 'utc': '20261104T000000Z'}]))
        fx.commit('k1')
        fx.write()
        return fx

    def test_known_brier_and_skill_for_three_deliveries(self):
        fx = self.three()
        sc = kbmod.kb_score(fx.ctx(delivery=None).kb)
        f = sc['families']['addition_single_digit']
        # Laplace (2+1)/(3+2) = 0.6; Brier = (0.04+0.09+0.16)/3; base Brier = (0.16+0.36+0.16)/3
        self.assertEqual((f['n_scored_deliveries'], f['n_pass']), (3, 2))
        self.assertAlmostEqual(f['laplace_base_rate'], 0.6, places=12)
        self.assertAlmostEqual(f['brier'], 0.29 / 3, places=12)
        self.assertAlmostEqual(f['brier_base'], 0.68 / 3, places=12)
        self.assertAlmostEqual(f['brier_skill'], 1 - 0.29 / 0.68, places=12)
        self.assertAlmostEqual(f['hit_rate'], 5 / 6.0, places=12)
        self.assertEqual(f['label'], 'INSUFFICIENT_N')
        self.assertEqual(sc['finding_transfers'], {'F-UF-01': {'ok': 0, 'failed': 1}})

    def test_spec_shaped_entries_use_the_lock_prediction(self):
        # SPEC 8 predictions_scored entries {id, observed, hit, interval_score, brier}: target and p_pass come
        # from the delivery's lock (K01-like)
        fx = Fixture(self.tmp / 'fx2')
        fx.put(KB + '/CONSTRAINTS.json', CONSTRAINTS)
        rows = []
        for i, (d, p, gate) in enumerate((('a_dev_v1_20261101', 0.7, 'PASS'), ('b_dev_v1_20261102', 0.4, 'FAIL'),
                                          ('c_dev_v1_20261103', 0.9, 'FAIL'))):
            fx.put(d + '/ENDPOINT_LOCK.json', {'predictions': [{'id': 'P1', 'target': 'primary', 'p_pass': p,
                                                                 'interval90': [0.4, 0.8]}]})
            data = fx.put(d + '/CLAIM.json', {'schema': 'minifly.claim.v1', 'primary_gate': gate, 'predictions_scored': [
                {'id': 'P1', 'observed': 0.7 if gate == 'PASS' else 0.3, 'hit': gate == 'PASS', 'interval_score': 0.4,
                 'brier': round((p - (1.0 if gate == 'PASS' else 0.0)) ** 2, 6)}]})
            rows.append(ledger_row(d, '2026110%dT000000Z' % (i + 1), primary_gate=gate,
                                   claim_sha256=common.sha256_bytes(data)))
        for led in LEDGERS:
            fx.put('%s/%s.jsonl' % (KB, led), b'')
        fx.put(KB + '/DELIVERY_LEDGER.jsonl', jsonl(rows))
        fx.put(KB + '/FINDINGS.jsonl', jsonl([SEED_FINDING]))
        fx.commit('k1')
        fx.write()
        sc = kbmod.kb_score(fx.ctx(delivery=None).kb)
        f = sc['families']['addition_single_digit']
        self.assertEqual(f['n_primary_predictions'], 3)
        self.assertAlmostEqual(f['brier'], (0.09 + 0.16 + 0.81) / 3, places=12)
        self.assertEqual(sc['brier_mean'], 0.353333)
        self.assertIn('INSUFFICIENT_N', sc['skill'])
        self.assertAlmostEqual(f['hit_rate'], 1 / 3.0, places=12)

    def test_kb_score_cli_and_mismatched_claim(self):
        fx = self.three()
        (fx.repo / 'c_dev_v1_20261103' / 'CLAIM.json').write_text('{"tampered": true}')
        out = fx.repo.parent / 'score.json'
        with quiet():
            rc = kbmod.cmd_kb(fx.args(action='score', as_of=None, out=str(out)))
        self.assertEqual(rc, 0)
        sc = json.loads(out.read_text())
        self.assertEqual(sc['families']['addition_single_digit']['n_scored_deliveries'], 2)
        self.assertEqual(sc['excluded'][0]['delivery'], 'c_dev_v1_20261103')

    def _n_pass(self, n, root):
        fx = Fixture(root)
        fx.put(KB + '/CONSTRAINTS.json', CONSTRAINTS)
        rows = []
        for i in range(n):
            d = 'd%d_dev_v1_202611%02d' % (i, i + 1)
            data = fx.put(d + '/CLAIM.json', claim_with(0.9, 'PASS'))
            rows.append(ledger_row(d, '202611%02dT000000Z' % (i + 1), primary_gate='PASS',
                                   claim_sha256=common.sha256_bytes(data)))
        for led in LEDGERS:
            fx.put('%s/%s.jsonl' % (KB, led), b'')
        fx.put(KB + '/DELIVERY_LEDGER.jsonl', jsonl(rows))
        fx.commit('k1')
        fx.write()
        return kbmod.kb_score(fx.ctx(delivery=None).kb)['families']['addition_single_digit']

    def test_labels(self):
        f = self._n_pass(6, self.tmp / 'six')
        self.assertAlmostEqual(f['laplace_base_rate'], 7 / 8.0)
        self.assertEqual(f['label'], 'OK')
        self.assertAlmostEqual(f['brier_skill'], 1 - 0.01 / 0.015625, places=12)
        f = self._n_pass(9, self.tmp / 'nine')
        self.assertAlmostEqual(f['laplace_base_rate'], 10 / 11.0)
        self.assertEqual(f['label'], 'DEGENERATE_BASE')


class TestBacktest(TmpCase):
    def test_backtest_score_known_values(self):
        fx = kb_only(self.tmp)
        contract = {'d1': 0.5, 'd2': 0.5, 'd3': 0.5, 'd4': 0.5}
        kbf = {'d1': 0.9, 'd2': 0.2, 'd3': 0.7, 'd4': 0.4}
        outcomes = {'d1': 1, 'd2': 0, 'd3': 1, 'd4': 0}
        res = kbmod.backtest_score(fx.ctx(delivery=None).kb, contract, kbf, outcomes, n_boot=500, seed=1)
        self.assertEqual(res['label'], 'LEAK_RISK')
        self.assertAlmostEqual(res['laplace_base_rate'], 0.5)
        self.assertAlmostEqual(res['brier_contract_only'], 0.25)
        self.assertAlmostEqual(res['brier_kb'], (0.01 + 0.04 + 0.09 + 0.16) / 4)
        self.assertAlmostEqual(res['kb_value'], 0.25 - 0.075)
        self.assertAlmostEqual(res['skill_kb'], 1 - 0.075 / 0.25)
        lo, hi = res['kb_value_ci95']
        self.assertTrue(lo <= res['kb_value'] <= hi)
        again = kbmod.backtest_score(fx.ctx(delivery=None).kb, contract, kbf, outcomes, n_boot=500, seed=1)
        self.assertEqual(again['kb_value_ci95'], res['kb_value_ci95'])

    def test_backtest_cli_files(self):
        fx = kb_only(self.tmp)
        c = self.tmp / 'contract.json'
        k = self.tmp / 'kbf.jsonl'
        o = self.tmp / 'outcomes.json'
        c.write_text(json.dumps({'forecasts': [{'delivery': 'd1', 'p_pass': 0.5}, {'delivery': 'd2', 'p_pass': 0.5}]}))
        k.write_text('{"delivery": "d1", "p_pass": 0.8}\n{"delivery": "d2", "p_pass": 0.4}\n')
        o.write_text(json.dumps({'d1': 'PASS', 'd2': 'FAIL'}))
        out = self.tmp / 'BACKTEST.json'
        args = fx.args(action='backtest-score', as_of='2026-10-15T00:00Z', out=str(out),
                       extra=['--contract', str(c), '--kb', str(k), '--outcomes', str(o), '--boot', '100'])
        with quiet():
            self.assertEqual(kbmod.cmd_kb(args), 0)
        res = json.loads(out.read_text())
        self.assertEqual(res['label'], 'LEAK_RISK')
        self.assertAlmostEqual(res['brier_kb'], (0.04 + 0.16) / 2)
        with quiet(), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(kbmod.cmd_kb(fx.args(action='backtest-score', as_of=None, out=None, extra=['--zzz'])),
                             common.EXIT_ERROR)

    def test_backtest_snapshot_timeline(self):
        fx = Fixture(self.tmp / 'fx')
        fx.put(KB + '/CONSTRAINTS.json', CONSTRAINTS)
        fx.put('a.md', 'v1\n')
        fx.commit('t1', utc='20261010T000000Z')
        fx.put('a.md', 'v2\n')
        fx.put('b.md', 'later\n')
        fx.commit('t2', ['a.md', 'b.md'], utc='20261020T000000Z')
        fx.write()
        out = self.tmp / 'snap'
        meta = kbmod.backtest_snapshot(fx.ctx(delivery=None), kbmod._parse_as_of('2026-10-15T00:00Z'), out)
        self.assertEqual(meta['commit_utc'], '20261010T000000Z')
        self.assertEqual((out / 'a.md').read_text(), 'v1\n')
        self.assertFalse((out / 'b.md').exists())
        self.assertEqual(meta['label'], 'LEAK_RISK')
        with self.assertRaises(common.InvalidInput):
            kbmod.backtest_snapshot(fx.ctx(delivery=None), '20261015T000000Z', fx.repo / 'inside')


@unittest.skipUnless(shutil.which('git'), 'git not installed')
class TestSnapshotRealGit(TmpCase):
    def git(self, repo, *args, date=None):
        env = dict(os.environ)
        env.update({'GIT_AUTHOR_NAME': 'T', 'GIT_AUTHOR_EMAIL': 't@example.invalid', 'GIT_COMMITTER_NAME': 'T',
                    'GIT_COMMITTER_EMAIL': 't@example.invalid', 'GIT_CONFIG_NOSYSTEM': '1'})
        if date:
            env.update({'GIT_AUTHOR_DATE': date, 'GIT_COMMITTER_DATE': date})
        subprocess.run(['git', '-c', 'commit.gpgsign=false', '-c', 'init.defaultBranch=main'] + list(args),
                       cwd=str(repo), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    def test_snapshot_last_commit_before_as_of(self):
        repo = self.tmp / 'repo'
        repo.mkdir()
        self.git(repo, 'init', '-q')
        (repo / 'a.md').write_text('v1\n')
        (repo / 'sub').mkdir()
        (repo / 'sub' / 'b.json').write_text('{}\n')
        self.git(repo, 'add', '-A')
        self.git(repo, 'commit', '-q', '-m', 'one', date='2026-10-10T00:00:00Z')
        (repo / 'a.md').write_text('v2\n')
        self.git(repo, 'commit', '-q', '-am', 'two', date='2026-10-20T00:00:00Z')
        out = self.tmp / 'snap'
        with mock.patch.object(kbmod, 'ARCHIVE_BATCH_PATHS', 1):
            with quiet():
                rc = kbmod.cmd_kb(argparse.Namespace(repo_root=str(repo), git=None, json=False, extra=[],
                                                     action='backtest-snapshot', as_of='2026-10-15T00:00Z',
                                                     out=str(out)))
        self.assertEqual(rc, 0)
        self.assertEqual((out / 'a.md').read_text(), 'v1\n')
        self.assertEqual((out / 'sub' / 'b.json').read_text(), '{}\n')
        meta = json.loads((out / '_BACKTEST_SNAPSHOT.json').read_text())
        self.assertEqual((meta['commit_utc'], meta['label']), ('20261010T000000Z', 'LEAK_RISK'))


class TestDispatcher(TmpCase):
    def test_kb_status_and_ledger_append(self):
        import minifly_check
        fx = kb_only(self.tmp, {'FINDINGS': [SEED_FINDING]})
        base = ['--repo-root', str(fx.repo), '--git', 'timeline:%s' % fx.root]
        with quiet():
            self.assertEqual(minifly_check.main(base + ['kb', 'status']), 0)
            self.assertEqual(minifly_check.main(base + ['kb', 'score', '--json']), 0)
        row = self.tmp / 'row.json'
        row.write_text(json.dumps({k: v for k, v in FIND.items() if k != 'id'}))
        with quiet():
            self.assertEqual(minifly_check.main(base + ['ledger', 'append', '--ledger', 'FINDINGS', '--row', str(row)]), 0)
        self.assertIn('F-20261105-codex-1', (fx.repo / KB / 'FINDINGS.jsonl').read_text())


class TestLedgerAppend(TmpCase):
    def append(self, fx, ledger, row):
        p = fx.repo.parent / 'row.json'
        p.write_text(json.dumps(row))
        err = io.StringIO()
        with quiet(), contextlib.redirect_stderr(err):
            rc = kbmod.cmd_ledger(fx.args(action='append', ledger=ledger, row=str(p)))
        return rc, err.getvalue()

    def test_assigns_ids_and_appends_one_line(self):
        fx = kb_only(self.tmp, {'FINDINGS': [SEED_FINDING]})
        row = {k: v for k, v in FIND.items() if k != 'id'}
        row['utc'] = '20261105T000000Z'
        rc, err = self.append(fx, 'FINDINGS', row)
        self.assertEqual(rc, 0, err)
        rc, err = self.append(fx, 'FINDINGS', dict(row, claim='second'))
        self.assertEqual(rc, 0, err)
        lines = (fx.repo / KB / 'FINDINGS.jsonl').read_text().splitlines()
        self.assertEqual([json.loads(x)['id'] for x in lines[1:]], ['F-20261105-codex-1', 'F-20261105-codex-2'])
        rc, err = self.append(fx, 'USER_DECISIONS', {'utc': '20261105T090000Z', 'verbatim': 'Go with caps 2 CPU-h.',
                                                     'session': 's', 'channel': 'chat', 'scope': ['constraint:x'],
                                                     'numbers_restated': ['2'], 'recorded_by': 'claude:s'})
        self.assertEqual(rc, 0, err)
        ud = json.loads((fx.repo / KB / 'USER_DECISIONS.jsonl').read_text())
        self.assertEqual(ud['id'], 'UD-20261105-1')

    def test_refusals(self):
        fx = kb_only(self.tmp, {'FINDINGS': [SEED_FINDING, FIND]})
        row = {k: v for k, v in FIND.items() if k != 'id'}
        rc, err = self.append(fx, 'FINDINGS', dict(row, status='active'))
        self.assertEqual(rc, common.EXIT_FAIL)
        self.assertIn('status proposed', err)
        rc, err = self.append(fx, 'FINDINGS', dict(row, source='somewhere'))
        self.assertEqual(rc, common.EXIT_FAIL)
        rc, err = self.append(fx, 'FINDINGS', {'kind': 'activation', 'refers_to': FIND['id'], 'delivery': NAME,
                                               'author_agent': 'codex', 'evidence': 'e.md'})
        self.assertEqual(rc, common.EXIT_FAIL)
        self.assertIn('same family', err)
        rc, err = self.append(fx, 'INCIDENTS', {'kind': 'closure', 'refers_to': 'I-1', 'fm_class': 'FM-26',
                                                'delivery': NAME, 'status': 'closed', 'check_id': 'CLM-10',
                                                'fixture_id': None})
        self.assertEqual(rc, common.EXIT_FAIL)
        rc, err = self.append(fx, 'DELIVERY_LEDGER', {'delivery': 'x'})
        self.assertEqual(rc, common.EXIT_FAIL)
        self.assertEqual((fx.repo / KB / 'FINDINGS.jsonl').read_bytes().count(b'\n'), 2)

    def test_review_debt_repayment(self):
        fx = kb_only(self.tmp)
        debt = {'delivery': NAME, 'utc': '20261102T000000Z', 'reason': 'codex unavailable', 'recorded_by': 'claude:s',
                'repaid_by': None}
        rc, err = self.append(fx, 'REVIEW_DEBT', debt)
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(kbmod.open_debts(fx.ctx(delivery=None).kb, NAME)), 1)
        rc, err = self.append(fx, 'REVIEW_DEBT', dict(debt, id='DEBT-20261102-1', repaid_by='review/REVIEW.json',
                                                      utc='20261103T000000Z'))
        self.assertEqual(rc, 0, err)
        self.assertEqual(kbmod.open_debts(fx.ctx(delivery=None).kb, NAME), [])
        rc, err = self.append(fx, 'REVIEW_DEBT', dict(debt, id='DEBT-20261109-9', repaid_by='x'))
        self.assertEqual(rc, common.EXIT_FAIL)


class TestStatus(TmpCase):
    def test_status_json(self):
        debt = {'id': 'DEBT-20261102-1', 'delivery': NAME, 'utc': '20261102T000000Z', 'reason': 'r',
                'recorded_by': 'claude:s', 'repaid_by': None}
        inc = dict(TestIncidents.REPORT)
        sealed = ledger_row('s_dev_v1_20261103', '20261103T000000Z', **dict(E3_ROW, worlds_role='SEALED_CONFIRM'))
        sealed['objective_metric'] = {'held': 0.6, 'old_final': 0.9, 'G': 0.4}
        fx = kb_only(self.tmp, {'FINDINGS': [SEED_FINDING] + NARROWED, 'REVIEW_DEBT': [debt], 'INCIDENTS': [inc],
                                'DELIVERY_LEDGER': [sealed, ledger_row('t_dev_v1_20261104', '20261104T000000Z')]})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = kbmod.cmd_kb(fx.args(action='status', as_of=None, out=None, json=True))
        self.assertEqual(rc, 0)
        st = json.loads(out.getvalue())
        self.assertEqual(st['cadence']['deliveries_since_last_e3'], 1)
        self.assertEqual([r['id'] for r in st['open_incidents']], ['I-20261101-1'])
        self.assertEqual([r['id'] for r in st['narrowed_findings']], ['F-UF-15'])
        self.assertEqual([r['id'] for r in st['open_debts']], ['DEBT-20261102-1'])
        self.assertEqual(st['sealed_G_series'], [{'delivery': 's_dev_v1_20261103', 'utc': '20261103T000000Z', 'G': 0.4}])
        with quiet():
            self.assertEqual(kbmod.cmd_kb(fx.args(action='status', as_of=None, out=None)), 0)


if __name__ == '__main__':
    unittest.main()
