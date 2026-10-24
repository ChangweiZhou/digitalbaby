"""Unit tests for claimcheck.py (WP5): claim generation and the SPEC 9.9 headline, CLM-*, ERR-01, WLD-*,
PTR-* and lint. Synthetic TimelineGit fixtures in temp dirs only; SCORES/VERDICT are hand-written files."""
import argparse
import contextlib
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

KIT = Path(__file__).resolve().parents[1]
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

import common  # noqa: E402
import claimcheck  # noqa: E402
import kb as kbmod  # noqa: E402
from common import FAIL, MISSING, NOT_APPLICABLE, PASS, WARN  # noqa: E402

NAME = 'minifly_c00a_dev_v1_20261101'
OFF = ['w_off_%d' % i for i in range(1, 9)]
KB = 'PROJECT_DOCUMENTS/KB'
LEDGERS = ('FINDINGS', 'INCIDENTS', 'USER_DECISIONS', 'DELIVERY_LEDGER', 'CONFIRM_LEDGER', 'REVIEW_DEBT',
           'CANARY_COMMITMENTS')
C00A_HEADLINE = ('PASS | earned E3 (attempted E3) | held exact own-EOS 62.5% [S2+ determined, n=80] vs best shortcut '
                 'CANON 12.5% | G 41.3pp | OLD final 91.0% (drop 2.0pp) | 8 MAIN worlds | best of k=2')

CONSTRAINTS = {
    'schema': 'minifly.constraints.v1',
    'tier_thresholds': {'t0_max_cpu_h': 1, 't1_max_cpu_h': 4, 't1_max_records': 20000, 'family_t1_max_cpu_h': 12},
    'threshold_floors': {'margin_abs': 0.05, 'min_worlds_nonpilot': 8, 'min_items_per_stratum': 8},
    'default_grant_caps': {'max_cpu_h': 2, 'max_records': 5000, 'max_cycles': 3},
    'e3_cadence_max': 2, 'e3_cadence_counts_legacy': False, 'review_debt_max_open': 3,
    'verification_subset_max_mb': 50,
}

SEED_FINDING = {'id': 'F-UF-01', 'kind': 'finding', 'refers_to': None, 'claim': 'seed claim', 'level': 'E1',
                'conditions': {'best_shortcut': 'FIXED_1'}, 'effect': {'value': 0.1}, 'source': 'x.md:1@abc',
                'status': 'provisional', 'delivery': 'seed', 'author_agent': 'claude', 'utc': '20261024T000000Z',
                'caveat': None}
NARROWED = [
    {'id': 'F-UF-15', 'kind': 'finding', 'refers_to': None, 'claim': 'STOP beats DIRECT', 'level': 'E1',
     'conditions': {'best_shortcut': None}, 'effect': {'value': 0.22}, 'source': 'y.md:9@abc',
     'status': 'provisional', 'delivery': 'seed', 'author_agent': 'claude', 'utc': '20261024T000000Z', 'caveat': None},
    {'id': 'F-20261024-claude-1', 'kind': 'narrowing', 'refers_to': 'F-UF-15', 'claim': 'Narrowing of F-UF-15',
     'caveat': 'FIXED_1 10.645% vs STOP 11.263% on NEW', 'status': 'narrowed', 'delivery': 'seed',
     'author_agent': 'claude', 'utc': '20261024T000000Z', 'source': 'z.md:7@abc', 'level': 'E1',
     'conditions': {}, 'effect': {}},
]


def sha(label):
    return hashlib.sha1(str(label).encode()).hexdigest()


def jdump(obj):
    return (json.dumps(obj, indent=1, sort_keys=True) + '\n').encode('utf-8')


def jsonl(rows):
    return ''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows).encode('utf-8')


class Fixture:
    """A TimelineGit fixture: repo/ (final tree), history/<sha>/<path> for every committed version."""

    def __init__(self, root):
        self.root = Path(root)
        self.files = {}
        self.commits = []
        self.history = {}

    def put(self, rel, data):
        if isinstance(data, (dict, list)):
            data = jdump(data)
        if isinstance(data, str):
            data = data.encode('utf-8')
        self.files[rel] = data
        return data

    def drop(self, rel):
        self.files.pop(rel, None)

    def commit(self, label, rels=None, trailers=None, branches=(), utc='20261101T000000Z'):
        s = sha(label)
        rels = sorted(self.files) if rels is None else list(rels)
        for rel in rels:
            if rel in self.files:
                self.history[(s, rel)] = self.files[rel]
        self.commits.append({'sha': s, 'utc': utc, 'author': 'Pencilbard', 'trailers': trailers or {},
                             'paths': rels, 'branches': list(branches)})
        return s

    def write(self, head=None, **extra):
        if self.root.exists():
            shutil.rmtree(str(self.root))
        self.root.mkdir(parents=True)
        tl = {'head': head or self.commits[-1]['sha'], 'commits': self.commits}
        tl.update(extra)
        (self.root / 'timeline.json').write_text(json.dumps(tl, indent=1))
        for rel, data in self.files.items():
            p = self.root / 'repo' / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        (self.root / 'repo').mkdir(exist_ok=True)
        for (s, rel), data in self.history.items():
            p = self.root / 'history' / s / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        return self

    @property
    def repo(self):
        return self.root / 'repo'

    def args(self, **kw):
        ns = argparse.Namespace(repo_root=str(self.repo), git='timeline:%s' % self.root, json=False, extra=[])
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns

    def ctx(self, delivery=NAME, stage='publish', **options):
        d = str(self.repo / delivery) if delivery else None
        return common.make_ctx(self.args(), delivery=d, stage=stage, options=options)


def base_lock(name=NAME, **over):
    lock = {
        'schema': common.SCHEMA_ENDPOINT_LOCK, 'delivery': name, 'protocol_id': name + '-p1',
        'created_utc': '20261101T000000Z', 'author_agent': 'codex', 'tier_declared': 2, 'intent': 'diagnostic',
        'claim_level_target': 'E3',
        'task': {'id': 'add1', 'family': 'addition_single_digit',
                 'format': {'kind': 'addition', 'sep_hex': '2b', 'terminator_hex': '3d', 'eos_hex': '0a',
                            'gen_cap_bytes': 4, 'normalize_leading_zeros': True}},
        'objective_distance': {'task_family': 'addition_single_digit', 'output': 'own_bytes_own_eos',
                               'held': 'unseen_composition', 'native_only_path': True, 'feedback': 'gradual_warmup',
                               'old_protection_measured': True},
        'units': {'witness': 'pair_rectangle', 'commutative': True},
        'split': {'taught_manifest': 'manifests/TAUGHT', 'held_manifest': 'manifests/HELD.jsonl'},
        'arms': [{'id': 'W', 'role': 'candidate'}, {'id': 'NPT', 'role': 'no_prior_teaching'},
                 {'id': 'SHUF', 'role': 'shuf'}, {'id': 'ORACLE', 'role': 'oracle_plumbing'}],
        'primary': {'metric': 'exact_own_eos', 'unconditional': True, 'pool': 'HELD', 'checkpoint': 'final',
                    'strata_counted': ['S2', 'S3', 'S4'], 'candidate_arm': 'W', 'comparator_arms': ['NPT'],
                    'chance': 0.01, 'margin_abs': 0.05, 'min_worlds': 8},
        'gates': [{'id': 'stop_gate', 'description': 'stop plumbing'}],
        'guards': [{'id': 'old_drop', 'kind': 'old_drop', 'pool': 'OLD', 'max_drop': 0.1},
                   {'id': 'old_acq', 'kind': 'old_acq', 'pool': 'OLD', 'margin': 0.05}],
        'worlds': {'technical': ['w_tech_1'], 'dev': ['w_dev_1'], 'rehearsal': ['w_reh_1'],
                   'official': {'mode': 'listed', 'ids': list(OFF), 'n': 8}, 'official_role_intended': 'MAIN'},
        'budget': {'max_records': 1000, 'max_cpu_h': 1},
        'files': {'admit': 'admit.py', 'native': ['native.py'], 'runner': ['run.py'], 'params': 'params.json',
                  'evaluator': [], 'generators': [], 'shared': [], 'assets': []},
        'lineage': {'parent': None, 'changes_declared': [], 'priors_added': []},
        'predictions': [
            {'id': 'P1', 'target': 'primary', 'point': 0.6, 'interval90': [0.5, 0.7], 'p_pass': 0.7,
             'basis_findings': ['F-UF-01'], 'base_rate_forecast': {'value': 0.5, 'source': 'laplace'}},
            {'id': 'P2', 'target': 'guard:old_drop', 'point': 0.03, 'interval90': [0.0, 0.05], 'p_pass': 0.9,
             'basis_findings': ['none'], 'base_rate_forecast': {'value': 0.8, 'source': 'laplace'}},
            {'id': 'P3', 'target': 'guard:old_acq', 'point': 0.25, 'interval90': [0.1, 0.2], 'p_pass': 0.8,
             'basis_findings': ['none'], 'base_rate_forecast': {'value': 0.6, 'source': 'laplace'}},
        ],
        'coprimary': [],
    }
    lock.update(over)
    return lock


def scores(lock_sha, src_sha, role='official', **over):
    worlds = list(OFF) if role == 'official' else ['w_dev_1']
    s = {'schema': 'minifly.scores.v1', 'role': role, 'endpoint_lock_sha256': lock_sha, 'source_lock_sha256': src_sha,
         'decide_version': '1.0.0', 'evaluated_worlds': worlds, 'held_exact_own_eos': 0.625,
         'held_by_stratum': {'S2': {'rate': 0.7, 'n': 30}, 'S3': {'rate': 0.55, 'n': 40}, 'S4': {'rate': 0.6, 'n': 30}},
         'determinacy': {'determined': 80, 'total': 100}, 'determined_fraction': 0.8,
         'old': {'end': 0.93, 'final': 0.91, 'drop': 0.02, 'acq': 0.31}, 'new_final': 0.88,
         'best_shortcut': {'name': 'CANON', 'value': 0.125}, 'best_other_arm': {'name': 'NPT', 'value': 0.2},
         'comparators': {'NPT': 0.2}, 'weak_comparator': [], 'headroom': 0.5, 'G': 0.413, 'G_per_sample': 0.0215,
         'G_per_cpu_h': 0.138, 'non_native_rows': 0, 'udi': None,
         'decision': {'verdict': 'PASS' if role == 'official' else 'DEV_NOT_EVIDENCE',
                      'shadow_verdict': None if role == 'official' else 'PASS', 'primary': {'mean_cand': 0.625}}}
    s.update(over)
    return s


def verdict(lock_sha, src_sha, **over):
    v = {'schema': 'minifly.verdict.v1', 'verdict': 'PASS', 'invalid_reason': None, 'shadow_verdict': None,
         'primary': {'mean_cand': 0.625, 'mean_comp': 0.2, 'per_world': [0.42] * 8, 'lcb': 0.38,
                     'method': 'paired_lcb'},
         'guards': {'old_drop': 'PASS', 'old_acq': 'PASS'},
         'guard_detail': {'old_drop': {'kind': 'old_drop', 'mean_drop': 0.02},
                          'old_acq': {'kind': 'old_acq', 'mean_acq': 0.31}},
         'world_roles': {w: 'official' for w in OFF}, 'decide_version': '1.0.0', 'endpoint_lock_sha256': lock_sha,
         'source_lock_sha256': src_sha, 'cells_sha256': '8' * 64}
    v.update(over)
    return v


def gate_report(stage='official', head=None, checker=None, dry=False, delivery=NAME, mode='timeline'):
    return {'schema': common.SCHEMA_GATE_REPORT, 'stage': stage, 'dry': dry, 'delivery': delivery,
            'utc': '20261102T000000Z', 'checker_sha256': checker or common.checker_sha256(), 'git_mode': mode,
            'git_head': head or sha('c1'), 'results': [{'id': 'SELF-01', 'status': 'PASS', 'detail': '',
                                                         'evidence': []}], 'exit_code': 0, 'verdict': 'PASS'}


def build(tmp, official=True, lock=None, kb_rows=None, extra_files=None, report=None, attempts=None):
    """A C00a-like delivery: E3 target, 8 official MAIN worlds, hand-written SCORES/VERDICT, k=2."""
    fx = Fixture(Path(tmp) / 'fx')
    d = NAME + '/'
    fx.put(KB + '/CONSTRAINTS.json', CONSTRAINTS)
    rows = {'FINDINGS': [SEED_FINDING] + NARROWED}
    rows.update(kb_rows or {})
    for led in LEDGERS:
        fx.put('%s/%s.jsonl' % (KB, led), jsonl(rows.get(led, [])))
    lock = lock or base_lock()
    lb = fx.put(d + 'ENDPOINT_LOCK.json', lock)
    sl = {'schema': common.SCHEMA_SOURCE_LOCK, 'delivery': NAME, 'endpoint_lock_sha256': common.sha256_bytes(lb),
          'checker_version': '1.0.0', 'closure_sha256': {d + 'native.py': 'a' * 64, d + 'run.py': 'b' * 64},
          'native_closure_sha': 'n2', 'params_sha256': 'p2'}
    sb = fx.put(d + 'SOURCE_LOCK.json', sl)
    ls, ss = common.sha256_bytes(lb), common.sha256_bytes(sb)
    fx.put(d + 'native.py', 'x = 1\n')
    fx.put(d + 'run.py', 'y = 1\n')
    fx.put(d + 'params.json', {'lr': 0.1})
    fx.put(d + 'manifests/HELD.jsonl', jsonl([{'item_id': 'h1', 'prompt_hex': '332b343d', 'expected_hex': '37'},
                                             {'item_id': 'h2', 'prompt_hex': '352b333d', 'expected_hex': '38'}]))
    att = [{'schema': 'minifly.attempt.v1', 'attempt_id': 'ATT-1', 'stage': 'dev', 'argv': ['python3', 'run.py', '--worlds', 'w_dev_1'],
            'git_head': sha('c0'), 'cpu_children_user_s': 100.0, 'cpu_children_sys_s': 0.0, 'native_closure_sha': 'n1',
            'params_sha256': 'p1', 'produced_paths': [{'path': 'results/dev/OUTPUTS.jsonl', 'sha256': 'x' * 64}]}]
    if official:
        att.append({'schema': 'minifly.attempt.v1', 'attempt_id': 'ATT-2', 'stage': 'official',
                    'argv': ['python3', 'run.py', '--worlds', ','.join(OFF)], 'git_head': sha('c0'),
                    'cpu_children_user_s': 200.0, 'cpu_children_sys_s': 0.0, 'native_closure_sha': 'n2',
                    'params_sha256': 'p2', 'produced_paths': [{'path': 'results/official/OUTPUTS.jsonl',
                                                               'sha256': 'y' * 64}]})
    fx.put(d + 'runlog/ATTEMPTS.jsonl', jsonl(attempts if attempts is not None else att))
    fx.put(d + 'results/dev/SCORES.json', scores(ls, ss, role='dev'))
    if official:
        fx.put(d + 'results/official/SCORES.json', scores(ls, ss))
        fx.put(d + 'results/official/VERDICT.json', verdict(ls, ss))
    for rel, data in (extra_files or {}).items():
        fx.put(rel, data)
    outcome = [r for r in fx.files if r.startswith(d + 'results/') or r.startswith(d + 'runlog/')]
    fx.commit('c0', [r for r in fx.files if r not in outcome], trailers={'Acting-Agent': ['codex']},
              branches=['origin/codex/x'])
    fx.commit('c1', outcome, trailers={'Acting-Agent': ['codex']}, branches=['origin/codex/x'],
              utc='20261102T000000Z')
    fx.put(d + 'checks/GATE_official_20261102T000000Z.json', gate_report())
    if report is not None:
        fx.put(d + 'REPORT.md', report)
    fx.write()
    return fx


def run(ctx, rid):
    rule = common.rule_index(common.load_rules())[rid]
    return common.run_check(ctx, rule)


def status(ctx, rid):
    return [r.status for r in run(ctx, rid)]


def adm_expo_pass(ctx, rid):
    return ('PASS', 'stub: ADM-EXPO (WP4) passes in this unit test') if rid == 'ADM-EXPO' else ('CHECKER_ERROR', 'stub')


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


class TmpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        common._RULES_CACHE.clear()

    def tearDown(self):
        self._tmp.cleanup()

    def claim(self, fx, delivery=NAME, stub=True):
        args = fx.args(dir=str(fx.repo / delivery))
        with quiet():
            if stub:
                with mock.patch.object(claimcheck, '_check_status', side_effect=adm_expo_pass):
                    rc = claimcheck.cmd_claim(args)
            else:
                rc = claimcheck.cmd_claim(args)
        return rc, json.loads((fx.repo / delivery / 'CLAIM.json').read_text())


# --------------------------------------------------------------------------------------

class TestClaimGeneration(TmpCase):
    def test_c00a_headline_and_fields(self):
        fx = build(self.tmp)
        rc, c = self.claim(fx)
        self.assertEqual(rc, 0)
        self.assertEqual(c['headline'], C00A_HEADLINE)
        self.assertEqual((c['primary_gate'], c['level_earned'], c['worlds_role'], c['n_worlds'], c['gate_type']),
                         ('PASS', 'E3', 'MAIN', 8, 'objective'))
        self.assertEqual(c['designs_tried_k'], 2)
        self.assertEqual(c['reuse_specificity'], 'not_tested')
        self.assertEqual(c['dev_shadow_verdict'], 'PASS')
        self.assertEqual(c['labels'], [])
        self.assertEqual(c['posthoc_files'], [])
        self.assertEqual(c['objective_metric']['best_shortcut'], {'name': 'CANON', 'value': 0.125})
        self.assertEqual(c['objective_metric']['held_by_stratum'], {'S2': 0.7, 'S3': 0.55, 'S4': 0.6})
        self.assertEqual(c['compute']['cpu_h_total_incl_failed'], round(300 / 3600.0, 6))
        self.assertTrue(c['native_path_only'])
        self.assertEqual(c['checker_sha256'], common.checker_sha256())
        self.assertEqual(claimcheck.claim_problems(c), [])
        # SPEC 20.8: no resume; the synthetic official attempt records no exit_code, which counts as interrupted
        # (only an exit_code of 0 without an interrupted/recovered flag is a normal end)
        self.assertEqual(c['provenance'], {'interrupted': True, 'resume_attempts': [], 'resumed_units': [],
                                           'replay_units': []})
        bad = dict(c)
        bad.pop('provenance')
        self.assertIn('missing field provenance', claimcheck.claim_problems(bad))
        bad['provenance'] = {'interrupted': 'no', 'resume_attempts': [], 'resumed_units': [], 'replay_units': []}
        self.assertTrue(any(p.startswith('provenance must be') for p in claimcheck.claim_problems(bad)))
        # REPORT.md was created with the headline as its first line
        self.assertEqual((fx.repo / NAME / 'REPORT.md').read_text().splitlines()[0], C00A_HEADLINE)

    def test_predictions_scored(self):
        fx = build(self.tmp)
        _rc, c = self.claim(fx)
        ps = {p['id']: p for p in c['predictions_scored']}
        self.assertEqual(ps['P1']['observed'], 0.625)
        self.assertIs(ps['P1']['hit'], True)
        self.assertAlmostEqual(ps['P1']['interval_score'], 0.2, places=9)
        self.assertAlmostEqual(ps['P1']['brier'], 0.09, places=9)
        self.assertAlmostEqual(ps['P2']['interval_score'], 0.05, places=9)
        self.assertAlmostEqual(ps['P2']['brier'], 0.01, places=9)
        # P3: observed 0.31 above interval [0.1, 0.2]: Winkler = 0.1 + (2 / 0.1) * 0.11 = 2.3
        self.assertIs(ps['P3']['hit'], False)
        self.assertAlmostEqual(ps['P3']['interval_score'], 2.3, places=9)
        self.assertAlmostEqual(ps['P3']['brier'], 0.04, places=9)
        self.assertEqual(claimcheck.winkler(0.2, 0.4, 0.1), 0.2 + 20 * 0.1)

    def test_ledger_row_and_transfer(self):
        fx = build(self.tmp)
        _rc, c = self.claim(fx)
        kbp = fx.repo / KB
        rows = [json.loads(x) for x in (kbp / 'DELIVERY_LEDGER.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r['claim_sha256'], common.sha256_file(fx.repo / NAME / 'CLAIM.json'))
        self.assertIs(r['opened_held_E3'], True)  # HELD, S2-S4, PASS gate report, MAIN, PASS
        self.assertIn('admission PASS', r['e3_basis'])
        self.assertTrue(kbmod.opened_held_e3_valid(r)[0])
        tr = [json.loads(x) for x in (kbp / 'FINDINGS.jsonl').read_text().splitlines() if '"transfer"' in x]
        self.assertEqual(len(tr), 1)
        self.assertEqual((tr[0]['refers_to'], tr[0]['prediction_id'], tr[0]['outcome']), ('F-UF-01', 'P1', 'ok'))
        # re-running claim appends a new ledger row but never a duplicate transfer
        _rc, _c = self.claim(fx)
        rows = (kbp / 'DELIVERY_LEDGER.jsonl').read_text().splitlines()
        self.assertEqual(len(rows), 2)
        self.assertEqual(sum('"transfer"' in x for x in (kbp / 'FINDINGS.jsonl').read_text().splitlines()), 1)

    def test_publish_checks_pass_on_generated_claim(self):
        fx = build(self.tmp)
        self.claim(fx)
        self.assertEqual(status(fx.ctx(), 'CLM-11'), [FAIL])  # zero findings without a reason
        cp = fx.repo / NAME / 'CLAIM.json'
        c = json.loads(cp.read_text())
        c['findings_none_reason'] = 'Nothing durable beyond the headline.'
        cp.write_text(json.dumps(c))
        self.claim(fx)  # re-run claim after editing: the ledger row links the new bytes
        ctx = fx.ctx()
        with mock.patch.object(claimcheck, '_check_status', side_effect=adm_expo_pass):
            for rid in ('CLM-01', 'CLM-02', 'CLM-03', 'CLM-05', 'CLM-06', 'CLM-09', 'CLM-10', 'CLM-11', 'CLM-12',
                        'CLM-13', 'CLM-14'):
                res = run(ctx, rid)
                self.assertEqual([r.status for r in res], [PASS], '%s: %s' % (rid, res[0].detail))
        self.assertEqual(status(ctx, 'CLM-08'), [PASS])
        self.assertEqual(status(ctx, 'CLM-04'), [NOT_APPLICABLE])
        self.assertEqual(status(ctx, 'CLM-07'), [NOT_APPLICABLE])
        self.assertEqual(status(ctx, 'ERR-01'), [NOT_APPLICABLE])

    def test_clm02_detects_edits_and_preserves_editable_fields(self):
        fx = build(self.tmp)
        self.claim(fx)
        cp = fx.repo / NAME / 'CLAIM.json'
        c = json.loads(cp.read_text())
        c['findings_proposed'] = [{'claim': 'W beats NPT', 'conditions': {'best_shortcut': 'CANON'},
                                   'source': NAME + '/native.py:1'}]
        c['notes'] = 'free prose'
        cp.write_text(json.dumps(c, indent=2))
        with mock.patch.object(claimcheck, '_check_status', side_effect=adm_expo_pass):
            self.assertEqual(status(fx.ctx(), 'CLM-02'), [PASS])
            self.assertEqual(status(fx.ctx(), 'CLM-11'), [PASS])
            self.assertEqual(status(fx.ctx(), 'CLM-13'), [FAIL])  # edited after claim: re-run claim
            rc, c2 = self.claim(fx)
            self.assertEqual(c2['findings_proposed'], c['findings_proposed'])
            self.assertEqual(c2['notes'], 'free prose')
            self.assertEqual(status(fx.ctx(), 'CLM-13'), [PASS])
            c2['gate_type'] = 'objective'
            c2['worlds_role'] = 'SEALED_CONFIRM'
            cp.write_text(json.dumps(c2, indent=2))
            res = run(fx.ctx(), 'CLM-02')
            self.assertEqual(res[0].status, FAIL)
            self.assertIn('worlds_role', res[0].detail)
            self.assertEqual(status(fx.ctx(), 'CLM-09'), [FAIL])
            c2['confirmed'] = True
            cp.write_text(json.dumps(c2, indent=2))
            self.assertEqual(status(fx.ctx(), 'CLM-01'), [FAIL])

    def test_dev_only_not_opened_with_dev_shadow(self):
        fx = build(self.tmp, official=False)
        rc, c = self.claim(fx, stub=False)
        self.assertEqual(rc, 0)
        self.assertEqual((c['primary_gate'], c['level_earned'], c['worlds_role']), ('NOT_OPENED', 'NONE', 'DEV'))
        self.assertTrue(c['headline'].startswith('NOT_OPENED | earned NONE (attempted E3) | held exact own-EOS 62.5%'))
        self.assertTrue(c['headline'].endswith('| 1 DEV worlds | best of k=1 | dev shadow PASS'))
        self.assertTrue(all(p['status'] == 'unresolved' and p['brier'] == 'unresolved' for p in c['predictions_scored']))

    def test_nothing_measured(self):
        fx = build(self.tmp, official=False)
        (fx.repo / NAME / 'results' / 'dev' / 'SCORES.json').unlink()
        rc, c = self.claim(fx, stub=False)
        self.assertIn('| held exact own-EOS not_measured |', c['headline'])
        self.assertEqual(c['objective_metric']['held_exact_own_eos'], 'not_measured')

    def test_nothing_measured_headline_form(self):
        # SPEC 9.9: the held segment reads 'held exact own-EOS not_measured'; G and the shortcut comparison are
        # defined on held cells and OLD was not measured either, so none of them is printed; 0 worlds -> PILOT
        fx = build(self.tmp, official=False)
        (fx.repo / NAME / 'results' / 'dev' / 'SCORES.json').unlink()
        _rc, c = self.claim(fx, stub=False)
        self.assertIn('| held exact own-EOS not_measured | 0 PILOT worlds | best of k=', c['headline'])
        self.assertNotIn('G not_measured', c['headline'])
        self.assertNotIn('OLD final', c['headline'])
        self.assertEqual((c['worlds_role'], c['n_worlds']), ('PILOT', 0))

    def test_strata_text(self):
        w = {'units': {'witness': 'pair_rectangle'}}
        self.assertEqual(claimcheck._strata_text(dict(w, primary={'strata_counted': ['S2', 'S3']})), 'S2+ determined')
        self.assertEqual(claimcheck._strata_text(dict(w, primary={'strata_counted': ['S2', 'S3', 'S4']})),
                         'S2+ determined')
        self.assertEqual(claimcheck._strata_text(dict(w, primary={'strata_counted': ['S1', 'S2']})), 'S1,S2 determined')

    def test_hand_written_first_line_is_left_for_clm05(self):
        fx = build(self.tmp, report='PASS | G1 memory largely met on 8 worlds\n\nProse.\n')
        rc, c = self.claim(fx)
        self.assertEqual(rc, 0)
        text = (fx.repo / NAME / 'REPORT.md').read_text()
        self.assertTrue(text.startswith('PASS | G1 memory largely met'))
        self.assertEqual(status(fx.ctx(), 'CLM-05'), [FAIL])

    def test_cannot_test_stop_reason(self):
        fx = build(self.tmp, official=False,
                   extra_files={NAME + '/results/admission/TESTBED.json': {'result': 'CANNOT_TEST_CLAIM_AS_SPECIFIED'}})
        (fx.repo / NAME / 'results' / 'dev' / 'SCORES.json').unlink()
        rc, c = self.claim(fx, stub=False)
        self.assertEqual((c['primary_gate'], c['stop_reason']), ('CANNOT_TEST_CLAIM_AS_SPECIFIED', 'CANNOT_TEST'))

    def test_dev_touched_official_world_is_dev(self):
        att = [{'attempt_id': 'ATT-1', 'stage': 'dev', 'argv': ['run.py', '--world=w_off_3'], 'git_head': sha('c1'),
                'cpu_children_user_s': 1.0, 'cpu_children_sys_s': 0.0},
               {'attempt_id': 'ATT-2', 'stage': 'official', 'argv': ['run.py'], 'git_head': sha('c1'),
                'cpu_children_user_s': 1.0, 'cpu_children_sys_s': 0.0, 'produced_paths': []}]
        fx = build(self.tmp, attempts=att)
        rc, c = self.claim(fx)
        self.assertEqual((c['worlds_role'], c['primary_gate'], c['dev_shadow_verdict']), ('DEV', 'NOT_OPENED', 'PASS'))
        self.assertEqual(status(fx.ctx(), 'WLD-01'), [FAIL])

    def test_dev_touched_unparseable_receipt_fails_closed(self):
        """Verifier round 4: an unparseable receipt used to be skipped by dev_touched, so it could not mark its
        world dev-touched; it now fails closed (CLM-09 and WLD-01 FAIL, `claim` refuses)."""
        att = [{'attempt_id': 'ATT-2', 'stage': 'official', 'argv': ['run.py'], 'git_head': sha('c1'),
                'cpu_children_user_s': 1.0, 'cpu_children_sys_s': 0.0, 'produced_paths': []}]
        fx = build(self.tmp, attempts=att)
        rc, c = self.claim(fx)
        self.assertEqual((rc, c['worlds_role']), (0, 'MAIN'))
        self.assertEqual(status(fx.ctx(), 'CLM-09'), [PASS])
        rp = fx.repo / NAME / 'results' / 'receipts' / 'w_off_3' / 'W.json'
        rp.parent.mkdir(parents=True, exist_ok=True)
        for bad in ('{"world": "w_off_3", "world_role": "dev",', '["w_off_3", "dev"]'):
            rp.write_text(bad)
            for rid in ('CLM-09', 'WLD-01'):
                res = run(fx.ctx(), rid)
                self.assertEqual(res[0].status, FAIL, (rid, bad))
                self.assertIn('w_off_3/W.json', res[0].detail)
            with self.assertRaises(common.InvalidInput):
                claimcheck.dev_touched(fx.ctx(), OFF)
        # a well-formed dev-role receipt on an official world makes the world DEV (as before)
        rp.write_text(json.dumps({'world': 'w_off_3', 'world_role': 'dev', 'arm': 'W'}))
        self.assertIn('w_off_3', claimcheck.dev_touched(fx.ctx(), OFF))

    def test_pilot_labels_and_suffixes(self):
        lock = base_lock(lineage={'parent': None, 'changes_declared': ['read', 'init'], 'priors_added': ['digit_order']},
                         coprimary=[{'protocol_id': NAME + '-p0'}])
        lock['worlds']['official_role_intended'] = 'PILOT'
        fx = build(self.tmp, lock=lock)
        sp = fx.repo / NAME / 'results' / 'dev' / 'SCORES.json'
        s = json.loads(sp.read_text())
        s['weak_comparator'] = ['NPT']
        sp.write_text(json.dumps(s))
        rc, c = self.claim(fx)
        self.assertEqual(c['worlds_role'], 'PILOT')
        self.assertEqual(c['labels'], ['integrated_intervention', 'weak_comparator', 'priors:digit_order'])
        self.assertTrue(c['headline'].endswith('| 8 PILOT worlds | best of k=2 | amended: co-primary NOT_SCORED | '
                                               'priors: digit_order | WEAK_COMPARATOR'), c['headline'])

    def test_shortcut_at_least_candidate_suffix(self):
        fx = build(self.tmp)
        for rel in ('results/official/SCORES.json',):
            p = fx.repo / NAME / rel
            s = json.loads(p.read_text())
            s['best_shortcut'] = {'name': 'FIXED_1', 'value': 0.7}
            s['headroom'] = -0.075
            p.write_text(json.dumps(s))
        rc, c = self.claim(fx)
        self.assertIn('vs best shortcut FIXED_1 70.0%', c['headline'])
        self.assertIn('| FIXED_1 >= candidate', c['headline'])
        self.assertEqual(c['level_earned'], 'E2')  # E3 needs headroom >= margin_abs

    def test_stale_verdict_refused(self):
        fx = build(self.tmp)
        lp = fx.repo / NAME / 'ENDPOINT_LOCK.json'
        lp.write_text(lp.read_text().replace('"chance": 0.01', '"chance": 0.02'))
        with quiet():
            with self.assertRaises(common.InvalidInput):
                claimcheck.cmd_claim(fx.args(dir=str(fx.repo / NAME)))

    def test_legacy_refused(self):
        fx = build(self.tmp)
        legacy = fx.repo / 'old_experiment'
        legacy.mkdir()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(claimcheck.cmd_claim(fx.args(dir=str(legacy))), common.EXIT_MISSING)


class TestUdiPlanned(TmpCase):
    """While UDI-01 is planned (SPEC 20.2), claim never certifies a UDI result (INTEGRATION_LOG B3)."""

    def _sealed_udi_fixture(self):
        lock = base_lock(intent='promotion', udi={'variants_manifest': 'manifests/UDI_VARIANTS.json', 'alpha': 0.05})
        lock['worlds']['official_role_intended'] = 'SEALED_CONFIRM'
        fx = build(self.tmp, lock=lock,
                   extra_files={NAME + '/review/REVIEW.json': {'decision': 'ADMIT', 'findings': []}})
        sp = fx.repo / NAME / 'results' / 'official' / 'SCORES.json'
        s = json.loads(sp.read_text())
        s['udi'] = {'valid': True, 'lcb': 0.21, 'value': 0.29}
        sp.write_text(json.dumps(s))
        return fx

    @staticmethod
    def _stub_all_pass_but_udi(real):
        def stub(ctx, rid):
            if rid == 'UDI-01':
                return real(ctx, rid)
            return ('PASS', 'stub: %s passes in this unit test' % rid)
        return stub

    def _claim_with(self, fx, stub):
        with quiet():
            with mock.patch.object(claimcheck, '_check_status', side_effect=stub):
                rc = claimcheck.cmd_claim(fx.args(dir=str(fx.repo / NAME)))
        return rc, json.loads((fx.repo / NAME / 'CLAIM.json').read_text())

    def test_planned_udi_gives_not_tested_and_no_promotion(self):
        fx = self._sealed_udi_fixture()
        real = claimcheck._check_status
        stub = self._stub_all_pass_but_udi(real)
        rc, c = self._claim_with(fx, stub)
        self.assertEqual(rc, 0)
        self.assertEqual((c['worlds_role'], c['level_earned']), ('SEALED_CONFIRM', 'E3'))
        self.assertEqual(c['reuse_specificity'], 'not_tested')
        with mock.patch.object(claimcheck, '_check_status', side_effect=stub):
            res = run(fx.ctx(), 'CLM-03')
        self.assertEqual([r.status for r in res], [PASS])
        self.assertNotIn('promotion earned', res[0].detail)
        self.assertIn('E3 promotion not certifiable: UDI-01 planned', res[0].detail)
        ok, why = claimcheck.udi_certifiable(fx.ctx())
        self.assertFalse(ok)
        self.assertIn('planned', why)

    def test_hand_written_udi_pass_fails_clm03_while_planned(self):
        fx = self._sealed_udi_fixture()
        stub = self._stub_all_pass_but_udi(claimcheck._check_status)
        self._claim_with(fx, stub)
        cp = fx.repo / NAME / 'CLAIM.json'
        c = json.loads(cp.read_text())
        c['reuse_specificity'] = 'udi_pass'
        cp.write_text(json.dumps(c))
        with mock.patch.object(claimcheck, '_check_status', side_effect=stub):
            self.assertEqual(status(fx.ctx(), 'CLM-03'), [FAIL])

    def test_enforced_udi_pass_is_claimed_only_when_udi01_passes(self):
        rules = json.loads((KIT / 'RULES.json').read_text())
        for r in rules['rules']:
            if r['id'] == 'UDI-01':
                r['status'] = 'enforced'
        self.addCleanup(common._RULES_CACHE.clear)
        common._RULES_CACHE[str(common.KIT_DIR / 'RULES.json')] = rules
        fx = self._sealed_udi_fixture()
        rc, c = self._claim_with(fx, lambda ctx, rid: ('PASS', 'stub'))
        self.assertEqual(c['reuse_specificity'], 'udi_pass')
        with mock.patch.object(claimcheck, '_check_status', side_effect=lambda ctx, rid: ('PASS', 'stub')):
            res = run(fx.ctx(), 'CLM-03')
        self.assertIn('E3 promotion earned', res[0].detail)
        fails_udi = lambda ctx, rid: ('MISSING', 'stub: seal-derived D draw unverifiable') if rid == 'UDI-01' else ('PASS', 's')
        rc, c = self._claim_with(fx, fails_udi)
        self.assertEqual(c['reuse_specificity'], 'not_tested')


class TestTextChecks(TmpCase):
    def test_clm06_largely_met_in_headline_sentence(self):
        fx = build(self.tmp)
        self.claim(fx)
        rp = fx.repo / NAME / 'REPORT.md'
        rp.write_text(C00A_HEADLINE + ' largely met\n\nThe curve looks promising.\n')
        res = run(fx.ctx(), 'CLM-06')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('headline', res[0].detail)
        self.assertEqual(status(fx.ctx(), 'CLM-05'), [FAIL])

    def test_clm06_gate_sentence_blocks_and_prose_warns(self):
        fx = build(self.tmp)
        self.claim(fx)
        rp = fx.repo / NAME / 'REPORT.md'
        rp.write_text(C00A_HEADLINE + '\n\nThe curve looks promising.\n')
        res = run(fx.ctx(), 'CLM-06')
        self.assertEqual(res[0].status, WARN)
        rp.write_text(C00A_HEADLINE + '\n\nThe ADM-TB-CEIL check\nwas nearly met.\n')
        self.assertEqual(status(fx.ctx(), 'CLM-06'), [FAIL])
        rp.write_text(C00A_HEADLINE + '\n\nThe primary gate is close to passing.\n')
        self.assertEqual(status(fx.ctx(), 'CLM-06'), [FAIL])
        rp.write_text(C00A_HEADLINE + '\n\nThe old_drop guard was broadly fine.\n')
        self.assertEqual(status(fx.ctx(), 'CLM-06'), [FAIL])

    def test_clm04_native_confirmation_on_dev(self):
        fx = build(self.tmp, official=False)
        self.claim(fx, stub=False)
        rp = fx.repo / NAME / 'REPORT.md'
        rp.write_text(rp.read_text() + '\nNative confirmation of reuse.\n')
        res = run(fx.ctx(), 'CLM-04')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('DEV', res[0].detail)
        # CONFIRM_LEDGER is a kit identifier, not confirm wording
        rp.write_text(rp.read_text().replace('Native confirmation of reuse.', 'See CONFIRM_LEDGER.'))
        self.assertEqual(status(fx.ctx(), 'CLM-04'), [NOT_APPLICABLE])
        # prefixed forms of the same word are confirm wording too (verifier round 3: 'reconfirms' passed)
        base = rp.read_text()
        for word in ('This reconfirms reuse.', 'Reuse stays unconfirmed.', 'Dev disconfirms the old result.'):
            rp.write_text(base + '\n' + word + '\n')
            self.assertEqual(status(fx.ctx(), 'CLM-04'), [FAIL], word)

    def test_clm04_identifier_tokens_and_format_chars(self):
        """Verifier round 4 (blocking): confirm wording inside snake_case/CamelCase identifiers, in JSON keys and
        values, and split by zero-width characters is confirm wording (SPEC 7.4 CLM-04, 20.5)."""
        fx = build(self.tmp, official=False)
        self.claim(fx, stub=False)
        d = fx.repo / NAME
        rp = d / 'REPORT.md'
        base = rp.read_text()
        self.assertEqual(status(fx.ctx(), 'CLM-04'), [NOT_APPLICABLE])
        variants = [
            ('SUMMARY.json', json.dumps({'verdict': 'GAIN_CONFIRMED'})),
            ('SUMMARY.json', json.dumps({'axis_confirmed': True})),
            ('SUMMARY.json', json.dumps({'native_confirmation': 'yes'})),
            ('REPORT.md', base + '\nVerdict: SCIENTIFIC_CONFIRMATION of carry reuse.\n'),
            ('STATUS.md', 'formal_confirmation: passed on held E3\n'),
            ('STATUS.md', 'status: dev_confirmed\n'),
            ('REPORT.md', base + '\nNativeConfirmation achieved on dev.\n'),
            ('REPORT.md', base + '\nReuse con​firmed.\n'),
            ('REPORT.md', base + '\nReuse conf­irmed﻿.\n'),
            ('REPORT.md', base + '\nReuse ｃｏｎｆｉｒｍｅｄ.\n'),
            ('REPORT.md', base + '\nReuse cоnfirmed.\n'),
            ('STATUS.md', 'NOT_CONFIRMED on dev\n'),
            ('REPORT.md', base + '\nSee Sealed_Confirm.\n'),
        ]
        for rel, text in variants:
            (d / rel).write_text(text, encoding='utf-8')
            res = run(fx.ctx(), 'CLM-04')
            self.assertEqual(res[0].status, FAIL, (rel, text[-60:]))
            self.assertIn('DEV', res[0].detail)
            if rel == 'REPORT.md':
                rp.write_text(base)
            else:
                (d / rel).unlink()
        # kit enum and schema names are whole tokens only
        for text in ('The SEALED_CONFIRM role needs a seal.', 'Ledger: PROJECT_DOCUMENTS/KB/CONFIRM_LEDGER.jsonl.',
                     'confirm_attempt is null; confirm_alpha_spending is geometric.', 'when: sealed_confirm'):
            rp.write_text(base + '\n' + text + '\n')
            self.assertEqual(status(fx.ctx(), 'CLM-04'), [NOT_APPLICABLE], text)
        self.assertEqual(claimcheck.confirm_tokens('SEALED_CONFIRMED and xSEALED_CONFIRM'),
                         [(1, 'SEALED_CONFIRMED'), (1, 'xSEALED_CONFIRM')])

    def test_clm07_lineage_wording(self):
        fx = build(self.tmp)
        self.claim(fx)
        rp = fx.repo / NAME / 'REPORT.md'
        rp.write_text(rp.read_text() + '\nThe inherited core keeps its memory.\n')
        ctx = fx.ctx()
        ctx.tier = 2
        res = run(ctx, 'CLM-07')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('lineage.parent', res[0].detail)

    def test_clm12_absolute_paths_and_links(self):
        fx = build(self.tmp)
        self.claim(fx)
        rp = fx.repo / NAME / 'REPORT.md'
        rp.write_text(rp.read_text() + '\nData in /Users/zhou/runs and [notes](missing/NOTES.md); see '
                      '[ok](ENDPOINT_LOCK.json) and https://example.com/home/x.\n')
        res = run(fx.ctx(), 'CLM-12')
        self.assertEqual(res[0].status, FAIL)
        self.assertEqual(len(res[0].evidence), 2, res[0].evidence)

    def test_clm16_prior_numbers(self):
        fx = build(self.tmp)
        self.claim(fx)
        rp = fx.repo / NAME / 'REPORT.md'
        rp.write_text(rp.read_text() + '\nThe previous delivery reached 22.1% on HELD.\n')
        res = run(fx.ctx(), 'CLM-16')
        self.assertEqual(res[0].status, FAIL)  # advisory: the bundle maps it to WARN
        rp.write_text(rp.read_text().replace('on HELD.', 'on HELD (F-UF-01).'))
        self.assertEqual(status(fx.ctx(), 'CLM-16'), [PASS])

    def test_delivery_gate_ids_and_role_names(self):
        fx = build(self.tmp)
        memo = fx.repo / 'MEMO.md'
        memo.write_text('Memo\n\n* **G1 memory - largely met** at 84-88%.\n\nThe SEALED_CONFIRM role needs a seal.\n')
        ctx = fx.ctx(delivery=None, stage='lint')
        ctx.options['lint_files'] = [str(memo)]
        self.assertEqual(status(ctx, 'CLM-06'), [FAIL])
        self.assertEqual(claimcheck._confirm_hits([claimcheck._doc('m', memo.read_text(), True)]), [])
        self.assertEqual(len(claimcheck._confirm_hits([claimcheck._doc('m', 'Confirmed on dev.', True)])), 1)

    def test_dispatcher_lint(self):
        import minifly_check
        fx = build(self.tmp)
        memo = fx.repo / 'MEMO.md'
        memo.write_text('A clean memo\n\nNothing soft here.\n')
        with quiet():
            rc = minifly_check.main(['--repo-root', str(fx.repo), '--git', 'timeline:%s' % fx.root, 'lint', str(memo)])
        self.assertEqual(rc, 0)

    def test_text_units(self):
        u = claimcheck.text_units('Title line\n\nA sentence. Another one\ncontinues here.\n| a | b |\n', True)
        self.assertEqual(u[0], (1, 'Title line', True))
        self.assertIn((3, 'Another one continues here.', False), u)
        self.assertIn((5, '| a | b |', False), u)


class TestPredictionsAndHistory(TmpCase):
    def test_clm10_deleted_prediction(self):
        old = base_lock()
        old['predictions'] = old['predictions'] + [
            {'id': 'P4', 'target': 'primary', 'p_pass': 0.5, 'basis_findings': ['none'],
             'base_rate_forecast': {'value': 0.5, 'source': 'x'}}]
        fx = Fixture(self.tmp / 'fx')
        tmpfx = build(self.tmp / 'a')
        fx.files = dict(tmpfx.files)
        fx.history = {}
        cur = fx.files[NAME + '/ENDPOINT_LOCK.json']
        fx.files[NAME + '/ENDPOINT_LOCK.json'] = jdump(old)
        fx.commit('c0', [NAME + '/ENDPOINT_LOCK.json'], utc='20261031T000000Z')
        fx.files[NAME + '/ENDPOINT_LOCK.json'] = cur
        fx.commit('c1')
        fx.write()
        self.claim(fx)
        res = run(fx.ctx(), 'CLM-10')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('P4', res[0].detail)

    def test_clm10_scored_entry_removed(self):
        fx = build(self.tmp)
        self.claim(fx)
        cp = fx.repo / NAME / 'CLAIM.json'
        c = json.loads(cp.read_text())
        c['predictions_scored'] = c['predictions_scored'][:2]
        cp.write_text(json.dumps(c))
        res = run(fx.ctx(), 'CLM-10')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('P3', res[0].detail)

    def _claim_history(self, gates, errata=None):
        fx = build(self.tmp / 'base')
        self.claim(fx)
        claim = json.loads((fx.repo / NAME / 'CLAIM.json').read_text())
        f2 = Fixture(self.tmp / 'fx2')
        f2.files = {}
        for p in (fx.repo).rglob('*'):
            if p.is_file():
                f2.files[p.relative_to(fx.repo).as_posix()] = p.read_bytes()
        cp = NAME + '/CLAIM.json'
        f2.commit('c1', [r for r in f2.files if r != cp])
        for i, g in enumerate(gates):
            c = dict(claim)
            c['primary_gate'] = g
            f2.put(cp, c)
            f2.commit('claim%d' % i, [cp], utc='2026110%dT000000Z' % (i + 2))
        if errata is not None:
            f2.put(NAME + '/ERRATA.jsonl', jsonl(errata))
        f2.write()
        return f2

    def test_err01_post_hoc_upgrade(self):
        fx = self._claim_history(['FAIL', 'PASS'], errata=[
            {'id': 'ERR-1', 'utc': '20261110T090000Z', 'old_primary_gate': 'FAIL', 'new_primary_gate': 'PASS',
             'classification': 'POST_HOC', 'diff_summary': 'reanalysis', 'fixture_id': None, 'reviewer_signoff': None}])
        res = run(fx.ctx(), 'ERR-01')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('POST_HOC never upgrades', res[0].detail)

    def test_err01_change_without_errata_and_valid_downgrade(self):
        fx = self._claim_history(['PASS', 'FAIL'])
        self.assertEqual(status(fx.ctx(), 'ERR-01'), [FAIL])
        fx = self._claim_history(['PASS', 'FAIL'], errata=[
            {'id': 'ERR-1', 'utc': '20261110T090000Z', 'old_primary_gate': 'PASS', 'new_primary_gate': 'FAIL',
             'classification': 'POST_HOC', 'diff_summary': 'carry bug found', 'fixture_id': None,
             'reviewer_signoff': None}])
        self.assertEqual(status(fx.ctx(), 'ERR-01'), [PASS])

    def test_err01_bug_fix_upgrade_needs_fixture_and_signoff(self):
        fx = self._claim_history(['FAIL', 'PASS'], errata=[
            {'id': 'ERR-1', 'utc': '20261110T090000Z', 'old_primary_gate': 'FAIL', 'new_primary_gate': 'PASS',
             'classification': 'BUG_FIX', 'diff_summary': 'MET-02 bug', 'fixture_id': 'C01', 'reviewer_signoff': ''}])
        with mock.patch.object(kbmod, 'kit_expected', return_value={}):
            res = run(fx.ctx(), 'ERR-01')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('fixture', res[0].detail)
        self.assertIn('sign-off', res[0].detail)

    def test_clm14_checker_sha(self):
        fx = build(self.tmp)
        self.assertEqual(status(fx.ctx(), 'CLM-14'), [PASS])  # timeline fallback: the running checker
        (fx.repo / NAME / 'checks' / 'GATE_lock_20261101T000000Z.json').write_text(
            json.dumps(gate_report(stage='lock', checker='0' * 64)))
        res = run(fx.ctx(), 'CLM-14')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('GATE_lock', res[0].detail)
        (fx.repo / NAME / 'checks' / 'GATE_lock_20261101T000000Z.json').write_text(
            json.dumps(dict(gate_report(stage='lock'), verdict='FAIL')))
        self.assertEqual(status(fx.ctx(), 'CLM-14'), [FAIL])

    def test_clm14_empty_kit_digest_in_timeline_mode(self):
        # a corpus fixture commit without a kit tree: its committed checker is the empty kit (sha256 of b'')
        fx = build(self.tmp)
        (fx.repo / NAME / 'checks' / 'GATE_lock_20261101T000000Z.json').write_text(
            json.dumps(gate_report(stage='lock', checker=common.EMPTY_KIT_SHA256)))
        self.assertEqual(common.EMPTY_KIT_SHA256, hashlib.sha256(b'').hexdigest())
        self.assertEqual(status(fx.ctx(), 'CLM-14'), [PASS])

    def test_clm15_without_official_run(self):
        fx = build(self.tmp, official=False)
        res = run(fx.ctx(), 'CLM-15')
        self.assertEqual(res[0].status, PASS, res[0].detail)
        self.assertTrue(res[0].detail.startswith('no official run'), res[0].detail)

    def test_clm15_reports_are_never_deleted(self):
        # verifier round 3: deleting committed gate, dry or canary reports left publish at exit 0
        lock_rep = NAME + '/checks/GATE_lock_20261101T000000Z.json'
        fx = build(self.tmp, official=False, extra_files={lock_rep: gate_report(stage='lock')})
        self.assertEqual(status(fx.ctx(), 'CLM-15'), [PASS])
        (fx.repo / lock_rep).unlink()
        res = run(fx.ctx(), 'CLM-15')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('committed report checks/GATE_lock_20261101T000000Z.json was deleted', res[0].detail)
        # an ATTEMPTS gate_report and a CYCLES latest_gate must still exist with the recorded bytes
        rep = gate_report(stage='roster')
        cyc = [{'k': 1, 'kind': 'technical', 'decision': 'continue', 'computed': {'attempts': [], 'latest_gate': {
            'path': 'checks/GATE_roster_20261101T000000Z.json', 'sha256': common.sha256_bytes(jdump(rep))}}}]
        fx = build(self.tmp, official=False, extra_files={
            NAME + '/checks/GATE_roster_20261101T000000Z.json': rep, NAME + '/runlog/CYCLES.jsonl': jsonl(cyc)})
        self.assertEqual(status(fx.ctx(), 'CLM-15'), [PASS])
        (fx.repo / NAME / 'checks/GATE_roster_20261101T000000Z.json').write_text(json.dumps(dict(rep, verdict='PASS ')))
        self.assertIn('no longer holds the bytes', run(fx.ctx(), 'CLM-15')[0].detail)

    def test_clm08_posthoc_verdict_input(self):
        fx = build(self.tmp)
        self.claim(fx)
        extra = fx.repo / NAME / 'results' / 'official' / 'EXTRA_ANALYSIS.json'
        extra.write_text('{}')
        (fx.repo / NAME / 'review').mkdir()
        (fx.repo / NAME / 'review' / 'post.py').write_text('z = 1\n')
        ctx = fx.ctx()
        posthoc, _off = claimcheck.posthoc_info(ctx)
        self.assertEqual(posthoc, ['results/official/EXTRA_ANALYSIS.json', 'review/post.py'])
        res = run(ctx, 'CLM-08')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('VERDICT inputs', res[0].detail)

    def test_clm15_subset(self):
        fx = build(self.tmp)
        self.claim(fx)
        ctx = fx.ctx()
        res = run(ctx, 'CLM-15')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('no official OUTPUTS', res[0].detail)

    def test_clm15_full_subset_passes(self):
        files = {}
        rows = []
        for w in OFF:
            for pool in ('HELD', 'OLD'):
                rows.append({'world': w, 'world_role': 'official', 'arm': 'W', 'pool': pool, 'checkpoint': 'final',
                             'item_id': 'i1', 'native_closure_sha': 'n2', 'params_sha256': 'p2'})
            for arm in ('W', 'NPT', 'SHUF', 'ORACLE'):
                files['%s/results/receipts/%s/%s.json' % (NAME, w, arm)] = {
                    'world': w, 'world_role': 'official', 'arm': arm, 'records': 10, 'native_closure_sha': 'n2',
                    'params_sha256': 'p2'}
                files['%s/manifests/TAUGHT/%s/%s.jsonl' % (NAME, w, arm)] = jsonl([{'world': w, 'arm': arm}])
        files[NAME + '/results/official/OUTPUTS.jsonl'] = jsonl(rows)
        fx = build(self.tmp, extra_files=files)
        res = run(fx.ctx(), 'CLM-15')
        self.assertEqual(res[0].status, PASS, res[0].detail)
        self.assertIn('written in this corpus run', res[0].detail)
        (fx.repo / NAME / 'results' / 'receipts' / OFF[0] / 'NPT.json').unlink()
        res = run(fx.ctx(), 'CLM-15')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('NPT.json absent', res[0].detail)

    def test_wld01_world_elsewhere_in_repo(self):
        fx = build(self.tmp, extra_files={'other_dev_v1_20261001/manifests/W.json': {'world': 'w_off_5'}})
        res = run(fx.ctx(), 'WLD-01')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('w_off_5', res[0].detail)
        fx = build(self.tmp / 'b')
        self.assertEqual(status(fx.ctx(), 'WLD-01'), [PASS])

    def test_wld02_overlap_in_headline(self):
        other = 'other_dev_v1_20261001/manifests/TAUGHT/w1/W.jsonl'
        fx = build(self.tmp, extra_files={other: jsonl([{'prompt_hex': '342b333d', 'answer_hex': '37'}])})
        res = run(fx.ctx(), 'WLD-02')
        self.assertEqual(res[0].status, FAIL)  # advisory rule: the bundle reports it as WARN
        _rc, c = self.claim(fx)
        self.assertTrue(c['headline'].endswith('| held overlap 1/2 with other deliveries'), c['headline'])


class TestPointers(TmpCase):
    def _pointer(self, fx, **over):
        cp = fx.repo / NAME / 'CLAIM.json'
        claim = json.loads(cp.read_text())
        ptr = {'schema': claimcheck.SCHEMA_POINTER, 'track': 'addition_e3', 'delivery': NAME,
               'claim': NAME + '/CLAIM.json', 'claim_sha256': common.sha256_file(cp),
               'narrowing_rows': [], 'updated_utc': '20261105T120000Z', 'previous_snapshot': None,
               'user_decision': None}
        for k in claimcheck.POINTER_COPY_FIELDS:
            ptr[k] = claim[k]
        ptr.update(over)
        p = fx.repo / 'PROJECT_DOCUMENTS' / 'CURRENT_ADDITION_E3_POINTER.json'
        p.write_text(json.dumps(ptr, indent=1))
        return p

    def test_confirmation_passes_and_users_path(self):
        fx = build(self.tmp)
        self.claim(fx)
        p = self._pointer(fx, confirmation_passes=True, objective_metric={'note': '/Users/zhou/x'})
        ctx = fx.ctx(pointer=str(p))
        self.assertEqual(status(ctx, 'PTR-03'), [FAIL])
        self.assertEqual(status(ctx, 'PTR-05'), [FAIL])
        self.assertEqual(status(ctx, 'PTR-02'), [FAIL])  # objective_metric is not a copy of the CLAIM
        self.assertEqual(status(ctx, 'PTR-01'), [FAIL])  # unknown field, no authorization

    def test_lint_pointer_file(self):
        fx = build(self.tmp)
        self.claim(fx)
        p = self._pointer(fx, confirmation_passes=True)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = claimcheck.cmd_lint(fx.args(files=[str(p)], json=True))
        rep = json.loads(out.getvalue())
        self.assertEqual(rc, common.EXIT_FAIL)
        st = {r['id']: r['status'] for r in rep['results']}
        self.assertEqual(st['PTR-03'], FAIL)
        self.assertEqual(st['PTR-05'], PASS)
        self.assertEqual(rep['pointer_files'], ['PROJECT_DOCUMENTS/CURRENT_ADDITION_E3_POINTER.json'])

    def test_lint_text_file(self):
        fx = build(self.tmp)
        memo = fx.repo / 'MEMO.md'
        memo.write_text('Promising route\n\nThe gate CLM-06 was largely met at /tmp/x.\n')
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = claimcheck.cmd_lint(fx.args(files=[str(memo)], json=True))
        st = {r['id']: r['status'] for r in json.loads(out.getvalue())['results']}
        self.assertEqual(rc, common.EXIT_FAIL)
        self.assertEqual((st['CLM-06'], st['CLM-12'], st['PTR-01']), (FAIL, FAIL, NOT_APPLICABLE))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(claimcheck.cmd_lint(fx.args(files=['nope.md'])), common.EXIT_MISSING)

    def test_narrowed_finding_quoted_without_narrowing_row(self):
        fx = build(self.tmp)
        self.claim(fx)
        p = self._pointer(fx, track='addition_e3 (see F-UF-15: STOP +22.10pp)')
        ctx = fx.ctx(pointer=str(p))
        res = run(ctx, 'PTR-04')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('F-20261024-claude-1', res[0].detail)
        p = self._pointer(fx, track='addition_e3 (see F-UF-15: STOP +22.10pp)', narrowing_rows=['F-20261024-claude-1'])
        self.assertEqual(status(fx.ctx(pointer=str(p)), 'PTR-04'), [PASS])
        p = self._pointer(fx, track='see F-UF-99')
        self.assertEqual(status(fx.ctx(pointer=str(p)), 'PTR-04'), [FAIL])

    def test_valid_pointer_with_adoption_ud_and_snapshot(self):
        ud = {'id': 'UD-20261105-1', 'utc': '20261105T100000Z', 'verbatim': 'Adopt it.', 'session': 's',
              'channel': 'chat', 'scope': ['adoption:%s' % NAME], 'numbers_restated': [], 'recorded_by': 'claude:s'}
        fx = build(self.tmp, kb_rows={'USER_DECISIONS': [ud]})
        self.claim(fx)
        p = self._pointer(fx, user_decision='UD-20261105-1')
        ctx = fx.ctx(pointer=str(p))
        for rid in ('PTR-01', 'PTR-02', 'PTR-03', 'PTR-04', 'PTR-05'):
            res = run(ctx, rid)
            self.assertEqual([r.status for r in res], [PASS], '%s: %s' % (rid, res[0].detail))
        self.assertEqual(status(ctx, 'PTR-07'), [NOT_APPLICABLE])
        self.assertEqual(status(ctx, 'PTR-06'), [PASS])

    def test_snapshot_required_when_pointer_changes(self):
        ud = {'id': 'UD-20261105-1', 'utc': '20261105T100000Z', 'verbatim': 'Adopt it.', 'session': 's',
              'channel': 'chat', 'scope': ['adoption:%s' % NAME], 'numbers_restated': [], 'recorded_by': 'claude:s'}
        fx = build(self.tmp, kb_rows={'USER_DECISIONS': [ud]})
        self.claim(fx)
        p = self._pointer(fx, user_decision='UD-20261105-1')
        prel = p.relative_to(fx.repo).as_posix()
        f2 = Fixture(self.tmp / 'fx2')
        for q in fx.repo.rglob('*'):
            if q.is_file():
                f2.files[q.relative_to(fx.repo).as_posix()] = q.read_bytes()
        f2.commit('c1')
        f2.write()
        ptr = json.loads(p.read_text())
        ptr['updated_utc'] = '20261106T000000Z'
        (f2.repo / prel).write_text(json.dumps(ptr))
        res = run(f2.ctx(pointer=str(f2.repo / prel)), 'PTR-01')
        self.assertEqual(res[0].status, FAIL)
        self.assertIn('previous_snapshot missing', res[0].detail)
        snap = 'PROJECT_DOCUMENTS/ADDITION_E3_POINTER_HISTORY/20261105T120000Z.json'
        (f2.repo / snap).parent.mkdir(parents=True)
        (f2.repo / snap).write_bytes(f2.files[prel])
        ptr['previous_snapshot'] = snap
        (f2.repo / prel).write_text(json.dumps(ptr))
        self.assertEqual(status(f2.ctx(pointer=str(f2.repo / prel)), 'PTR-01'), [PASS])

    def test_legacy_pointer_target(self):
        fx = build(self.tmp, extra_files={'minifly_old_dev_20261001/REPORT.md': 'old\n'})
        p = fx.repo / 'PROJECT_DOCUMENTS' / 'CURRENT_OLD.json'
        p.write_text(json.dumps({'schema': claimcheck.SCHEMA_POINTER, 'track': 'old', 'delivery': 'minifly_old_dev_20261001',
                                 'claim': None, 'claim_sha256': None, 'level_earned': 'E1', 'primary_gate': 'PASS',
                                 'gate_type': 'component', 'worlds_role': 'MAIN', 'objective_metric': {},
                                 'narrowing_rows': [], 'updated_utc': '20261105T120000Z', 'previous_snapshot': None,
                                 'user_decision': None}))
        ctx = fx.ctx(delivery=None, stage='lint', pointer=str(p))
        self.assertEqual(status(ctx, 'PTR-02'), [NOT_APPLICABLE])
        self.assertEqual(status(ctx, 'PTR-07'), [FAIL])


if __name__ == '__main__':
    unittest.main()
