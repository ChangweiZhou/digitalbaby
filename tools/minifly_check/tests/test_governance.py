"""Unit tests for governance.py (WP6): change classes, GOV-* checks, selftest engine, held-out canaries.

Every test builds small synthetic TimelineGit fixtures or batches in temp dirs. Checkers that must run as a
subprocess (GOV-MONO observed comparison, heldout score) are tiny fake kits written by the test itself."""
import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

import common  # noqa: E402
import governance as gov  # noqa: E402
from common import FAIL, MISSING, NOT_APPLICABLE, PASS, WARN, Result  # noqa: E402

KR = 'tools/minifly_check'
DESC = 'Use when testing the governance checker. ' * 3


def skill_md(name, body_lines=5, desc=DESC, extra_fm=''):
    body = '\n'.join('line %d' % i for i in range(body_lines))
    return '---\nname: %s\ndescription: "%s"\n%s---\n%s\n' % (name, desc.strip(), extra_fm, body)


def base_files():
    files = {
        KR + '/RULES.json': json.dumps({'schema': 'minifly.rules.v1', 'rules': [
            {'id': 'A-01', 'module': 'm', 'stages': ['lock'], 'tier_min': 0, 'introduced_in': '1.0.0',
             'status': 'enforced', 'fm_ids': ['FM-01'], 'applies_when': 'always'}],
            'bundles': {'lock': {'checks': ['A-01']}}}, indent=1),
        KR + '/EXPECTED.json': json.dumps({'C01': {'commands': [['gate', '--stage', 'lock', 'd']],
                                                    'must_fire': ['A-01'], 'must_not_fire': [], 'must_warn': [],
                                                    'expected_exit': [1], 'expect': {}}}, indent=1),
        KR + '/PROTECTED.json': (KIT / 'PROTECTED.json').read_text(),
        'PROJECT_DOCUMENTS/KB/CONSTRAINTS.json': json.dumps({'schema': 'c', 'threshold_floors': {
            'margin_abs': 0.05, 'old_max_drop': 0.1, 'shuf_band': [0.35, 0.65]},
            'tier_thresholds': {'t0_max_cpu_h': 1, 't1_max_cpu_h': 4, 't1_max_records': 20000,
                                'family_t1_max_cpu_h': 12},
            'native_forbidden': {'calls': ['.backward(']}}, indent=1, sort_keys=True),
        'PROJECT_DOCUMENTS/KB/USER_DECISIONS.jsonl': '',
        'PROJECT_DOCUMENTS/KB/FINDINGS.jsonl': '{"id": "F-1", "kind": "finding"}\n',
        'PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md': 'skills: ' + ' '.join(gov.SKILL_NAMES) + '\n',
    }
    for n in gov.SKILL_NAMES:
        files['.claude/skills/%s/SKILL.md' % n] = skill_md(n) + 'You MUST run the gate.\nNEVER skip it.\n'
    return files


class Timeline:
    """Builds a TimelineGit fixture: each commit is (sha, trailers, {path: text|None}, branches)."""

    def __init__(self, root):
        self.root = Path(root)
        self.commits = []
        self.state = {}

    def commit(self, sha, changes, trailers=None, branches=None, author='Pencilbard'):
        for p, v in changes.items():
            if v is None:
                self.state.pop(p, None)
            else:
                self.state[p] = v
            if v is not None:
                hp = self.root / 'history' / sha / p
                hp.parent.mkdir(parents=True, exist_ok=True)
                hp.write_text(v, encoding='utf-8')
        self.commits.append({'sha': sha, 'utc': '2026110%dT000000Z' % (len(self.commits) % 9 + 1), 'author': author,
                             'trailers': trailers or {}, 'paths': sorted(changes), 'branches': branches or []})
        return sha

    def write(self, dirty=None):
        repo = self.root / 'repo'
        if repo.exists():
            shutil.rmtree(str(repo))
        repo.mkdir(parents=True)
        for p, v in self.state.items():
            fp = repo / p
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(v, encoding='utf-8')
        tl = {'head': self.commits[-1]['sha'], 'commits': self.commits}
        if dirty:
            tl['dirty'] = dirty
        (self.root / 'timeline.json').write_text(json.dumps(tl, indent=1), encoding='utf-8')
        return self.root


_CTXS = []


def gov_ctx(fx, base, head):
    g = common.TimelineGit(fx)
    ctx = common.Ctx(repo_root=g.repo, delivery=None, git=g, kb=common.KB(g.repo), rules=common.load_rules(),
                     stage='governance', options={'base': base, 'head': head})
    _CTXS.append(ctx)
    return ctx


def run_one(ctx, rid):
    rule = common.rule_index(ctx.rules)[rid]
    res = common.run_check(ctx, rule)
    assert len(res) == 1, res
    return res[0]


CLAUDE = {'Acting-Agent': ['claude:s1']}
CODEX = {'Acting-Agent': ['codex']}


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='test_gov_'))

    def tearDown(self):
        while _CTXS:
            gov.cleanup(_CTXS.pop())
        shutil.rmtree(str(self.tmp), ignore_errors=True)

    def fixture(self, name='fx'):
        tl = Timeline(self.tmp / name)
        tl.commit('b0', base_files(), trailers=CLAUDE)
        return tl


class TestClassification(TmpCase):
    def changes(self, tl, extra_head):
        tl.commit('h1', extra_head, trailers=CLAUDE)
        fx = tl.write()
        g = common.TimelineGit(fx)
        paths = [p for p in g.changed_paths('b0', 'h1')]
        return gov.classify_paths(g, 'b0', 'h1', paths)

    def test_threshold_loosen_constraints(self):
        tl = self.fixture()
        c = json.loads(base_files()['PROJECT_DOCUMENTS/KB/CONSTRAINTS.json'])
        c['threshold_floors']['margin_abs'] = 0.03
        ch = self.changes(tl, {'PROJECT_DOCUMENTS/KB/CONSTRAINTS.json': json.dumps(c, indent=1, sort_keys=True)})
        self.assertEqual(len(ch), 1)
        self.assertEqual(ch[0]['class'], 'constraint')
        self.assertEqual(ch[0]['direction'], 'loosen')
        self.assertEqual(ch[0]['key'], 'threshold_floors.margin_abs')
        self.assertEqual(ch[0]['numbers'], [0.03, 0.05])

    def test_direction_map(self):
        d = gov.constraint_direction
        self.assertEqual(d('threshold_floors.margin_abs', 0.05, 0.07), 'tighten')
        self.assertEqual(d('threshold_floors.old_max_drop', 0.10, 0.05), 'tighten')
        self.assertEqual(d('threshold_floors.old_max_drop', 0.10, 0.20), 'loosen')
        self.assertEqual(d('threshold_floors.alpha_max', 0.05, 0.10), 'loosen')
        self.assertEqual(d('threshold_floors.per_world_floor', -0.05, -0.03), 'tighten')
        self.assertEqual(d('threshold_floors.shuf_band', [0.35, 0.65], [0.4, 0.6]), 'tighten')
        self.assertEqual(d('threshold_floors.shuf_band', [0.35, 0.65], [0.3, 0.6]), 'loosen')
        self.assertEqual(d('tier_thresholds.t1_max_cpu_h', 4, 8), 'loosen')
        self.assertEqual(d('e3_cadence_max', 2, 1), 'tighten')
        self.assertEqual(d('review_debt_max_open', 3, 5), 'loosen')
        self.assertEqual(d('default_grant_caps.max_cycles', 3, 5), 'loosen')
        self.assertEqual(d('default_grant_caps.official', False, True), 'loosen')
        self.assertEqual(d('native_forbidden.calls', ['.backward('], ['.backward(', 'x(']), 'tighten')
        self.assertEqual(d('native_forbidden.calls', ['.backward(', 'x('], ['x(']), 'loosen')
        self.assertEqual(d('evaluator_only_exemptions', [], ['a']), 'loosen')
        self.assertEqual(d('warmup.ramp', 'gradual', 'step'), 'unknown')
        self.assertEqual(d('status', 'DRAFT', 'FROZEN'), 'neutral')

    def test_rules_check_added_is_tighten_and_tier_raise_loosen(self):
        tl = self.fixture()
        r = json.loads(base_files()[KR + '/RULES.json'])
        r['rules'].append({'id': 'A-02', 'module': 'm', 'stages': ['lock'], 'tier_min': 0, 'introduced_in': '1.0.0',
                           'status': 'enforced', 'fm_ids': [], 'applies_when': 'always'})
        r['rules'][0]['tier_min'] = 2
        r['bundles']['lock']['checks'].append('A-02')
        ch = self.changes(tl, {KR + '/RULES.json': json.dumps(r, indent=1)})
        by = {(c['class'], c['what'].split(' ')[0] if 'tier_min' in c['what'] else c['what']) for c in ch}
        self.assertIn(('tighten', 'new check id A-02'), by)
        self.assertIn(('loosen', 'A-01'), by)
        self.assertTrue(any(c['class'] == 'tighten' and 'added to bundle lock' in c['what'] for c in ch))
        tier = [c for c in ch if 'tier_min' in c['what']][0]
        self.assertEqual(tier['numbers'], [0.0, 2.0])

    def test_rules_removed_status_and_fallback(self):
        old = json.loads(base_files()[KR + '/RULES.json'])
        new = json.loads(json.dumps(old))
        new['rules'][0]['status'] = 'advisory'
        new['bundles']['lock']['checks'] = [{'group': 'g', 'primary': ['A-01'], 'fallback': ['B-01']}]
        ch = gov.classify_rules('R', json.dumps(old).encode(), json.dumps(new).encode())
        self.assertTrue(any(c['class'] == 'loosen' and 'status enforced -> advisory' in c['what'] for c in ch))
        self.assertTrue(any(c['class'] == 'loosen' and 'B-01 added to bundle lock as fallback' in c['what'] for c in ch))
        self.assertTrue(any(c['class'] == 'loosen' and 'entry in bundle lock changed' in c['what'] for c in ch))
        gone = gov.classify_rules('R', json.dumps(old).encode(), json.dumps({'rules': [], 'bundles': {}}).encode())
        self.assertTrue(any(c['class'] == 'loosen' and 'removed check id A-01' in c['what'] for c in gone))
        self.assertTrue(any(c['class'] == 'loosen' and 'bundle lock removed' in c['what'] for c in gone))

    def test_expected_id_removed_is_loosen(self):
        tl = self.fixture()
        e = json.loads(base_files()[KR + '/EXPECTED.json'])
        e['C01']['must_fire'] = []
        ch = self.changes(tl, {KR + '/EXPECTED.json': json.dumps(e, indent=1)})
        self.assertEqual([(c['class'], c['what']) for c in ch], [('loosen', 'must_fire id A-01 removed from C01')])
        self.assertIn('C01', ch[0]['targets'])

    def test_expected_new_fixture_and_extended_commands(self):
        old = json.loads(base_files()[KR + '/EXPECTED.json'])
        new = json.loads(json.dumps(old))
        new['C02'] = dict(old['C01'])
        new['C01']['commands'] = old['C01']['commands'] + [['gate', '--stage', 'roster', 'd']]
        new['C01']['expected_exit'] = [1, 1]
        ch = gov.classify_expected('E', json.dumps(old).encode(), json.dumps(new).encode())
        self.assertEqual(sorted((c['class'], c['what']) for c in ch),
                         [('tighten', 'C01 commands extended'), ('tighten', 'new fixture C02')])
        new['C01']['expected_exit'] = [0, 1]
        ch = gov.classify_expected('E', json.dumps(old).encode(), json.dumps(new).encode())
        self.assertTrue(any(c['class'] == 'loosen' and 'commands/expected_exit changed' in c['what'] for c in ch))

    def test_skill_never_line_removed_is_loosen(self):
        tl = self.fixture()
        n = gov.SKILL_NAMES[4]
        ch = self.changes(tl, {'.claude/skills/%s/SKILL.md' % n: skill_md(n) + 'You MUST run the gate.\n'})
        self.assertEqual(len(ch), 1)
        self.assertEqual(ch[0]['class'], 'loosen')
        self.assertIn('NEVER skip it.', ch[0]['what'])
        self.assertIn(n, ch[0]['targets'])

    def test_skill_text_edit_without_rule_line_is_neutral(self):
        tl = self.fixture()
        n = gov.SKILL_NAMES[0]
        text = skill_md(n) + 'You MUST run the gate.\nNEVER skip it.\nAn added sentence.\n'
        ch = self.changes(tl, {'.claude/skills/%s/SKILL.md' % n: text})
        self.assertEqual([c['class'] for c in ch], ['neutral'])

    def test_gitattributes_any_removed_rule_is_loosen(self):
        ch = gov.classify_text('.gitattributes', 'gitattributes', [], ['PROJECT_DOCUMENTS/KB/*.jsonl merge=union'],
                               True, True)
        self.assertEqual(ch[0]['class'], 'loosen')

    def test_code_edit_is_pending_and_corpus_fixture_lifecycle(self):
        tl = self.fixture()
        tl.commit('b1', {KR + '/x.py': 'A = 1\n', KR + '/corpus/C01/timeline.json': '{}',
                         KR + '/corpus/C02/timeline.json': '{}'}, trailers=CLAUDE)
        tl.commit('h1', {KR + '/x.py': 'A = 2\n', KR + '/corpus/C01/timeline.json': '{"a": 1}',
                         KR + '/corpus/C02/timeline.json': None, KR + '/corpus/C03/timeline.json': '{}'},
                  trailers=CLAUDE)
        fx = tl.write()
        g = common.TimelineGit(fx)
        ch = gov.classify_paths(g, 'b1', 'h1', g.changed_paths('b1', 'h1'))
        got = sorted((c['class'], c['what']) for c in ch)
        self.assertIn(('code_pending', 'check code edit'), got)
        self.assertIn(('code_pending', 'fixture C01 modified (1 files)'), got)
        self.assertIn(('loosen', 'fixture C02 removed'), got)
        self.assertIn(('tighten', 'new fixture C03'), got)

    def test_user_decision_rows_added(self):
        tl = self.fixture()
        ch = self.changes(tl, {'PROJECT_DOCUMENTS/KB/USER_DECISIONS.jsonl': '{"id": "UD-20261101-1", "scope": []}\n'})
        self.assertEqual(ch[0]['class'], 'owner_decision')
        self.assertEqual(ch[0]['ud_ids'], ['UD-20261101-1'])


class TestGovChecks(TmpCase):
    def test_gov_num_with_and_without_number(self):
        for verbatim, want in (('Lower the margin floor from 0.05 to 0.03.', PASS),
                               ('Lower the margin floor from 0.05 a bit.', FAIL)):
            tl = Timeline(self.tmp / ('fx_%s' % want))
            tl.commit('b0', base_files(), trailers=CLAUDE)
            c = json.loads(base_files()['PROJECT_DOCUMENTS/KB/CONSTRAINTS.json'])
            c['threshold_floors']['margin_abs'] = 0.03
            ud = {'id': 'UD-20261101-1', 'utc': '20261101T000000Z', 'verbatim': verbatim, 'session': 's',
                  'channel': 'chat', 'scope': ['constraint:threshold_floors'], 'numbers_restated': [],
                  'recorded_by': 'claude:s1'}
            tl.commit('h1', {'PROJECT_DOCUMENTS/KB/CONSTRAINTS.json': json.dumps(c, indent=1, sort_keys=True),
                             'PROJECT_DOCUMENTS/KB/USER_DECISIONS.jsonl': json.dumps(ud) + '\n'},
                      trailers={'Acting-Agent': ['claude:s1'], 'User-Decision': ['UD-20261101-1']})
            ctx = gov_ctx(tl.write(), 'b0', 'h1')
            r = run_one(ctx, 'GOV-NUM')
            self.assertEqual(r.status, want, r.detail)
            if want == FAIL:
                self.assertIn('0.03', r.detail)
            self.assertEqual(run_one(ctx, 'GOV-UD').status, PASS)
            self.assertEqual(run_one(ctx, 'GOV-CLASS').status, PASS)
            self.assertEqual(run_one(ctx, 'GOV-REV').status, NOT_APPLICABLE)

    def test_gov_ud_scope_must_cover(self):
        tl = self.fixture()
        c = json.loads(base_files()['PROJECT_DOCUMENTS/KB/CONSTRAINTS.json'])
        c['threshold_floors']['margin_abs'] = 0.03
        ud = {'id': 'UD-20261101-2', 'verbatim': '0.05 to 0.03', 'scope': ['constraint:tier_thresholds']}
        tl.commit('h1', {'PROJECT_DOCUMENTS/KB/CONSTRAINTS.json': json.dumps(c, indent=1, sort_keys=True),
                         'PROJECT_DOCUMENTS/KB/USER_DECISIONS.jsonl': json.dumps(ud) + '\n'},
                  trailers={'Acting-Agent': ['claude:s1'], 'User-Decision': ['UD-20261101-2']})
        r = run_one(gov_ctx(tl.write(), 'b0', 'h1'), 'GOV-UD')
        self.assertEqual(r.status, FAIL)
        self.assertIn('constraint:threshold_floors.margin_abs', r.detail)

    def test_loosen_without_ud_fires_class_and_ud(self):
        tl = self.fixture()
        n = gov.SKILL_NAMES[4]
        tl.commit('h1', {'.claude/skills/%s/SKILL.md' % n: skill_md(n) + 'You MUST run the gate.\n'}, trailers=CLAUDE)
        ctx = gov_ctx(tl.write(), 'b0', 'h1')
        self.assertEqual(run_one(ctx, 'GOV-CLASS').status, FAIL)
        self.assertEqual(run_one(ctx, 'GOV-UD').status, FAIL)
        self.assertEqual(run_one(ctx, 'GOV-REV').status, FAIL)  # loosen also needs the review
        self.assertEqual(run_one(ctx, 'GOV-NUM').status, NOT_APPLICABLE)
        self.assertEqual(run_one(ctx, 'GOV-SKILL').status, PASS)
        self.assertEqual(run_one(ctx, 'GOV-MONO').status, NOT_APPLICABLE)

    def test_gov_sep_mixed_range(self):
        tl = self.fixture()
        r = json.loads(base_files()[KR + '/RULES.json'])
        r['rules'][0]['tier_min'] = 1
        tl.commit('h1', {KR + '/RULES.json': json.dumps(r, indent=1),
                         'lab_dev_v1_20261101/results/dev/OUTPUTS.jsonl': '{}\n'}, trailers=CLAUDE)
        ctx = gov_ctx(tl.write(), 'b0', 'h1')
        res = run_one(ctx, 'GOV-SEP')
        self.assertEqual(res.status, FAIL, res.detail)
        self.assertIn('lab_dev_v1_20261101', res.detail)
        self.assertEqual(run_one(ctx, 'GOV-UD').status, FAIL)

    def test_gov_sep_protected_only_and_legacy_dirs(self):
        self.assertIsNone(gov.delivery_dir_of('README.md'))
        self.assertIsNone(gov.delivery_dir_of('PROJECT_DOCUMENTS/KB/FINDINGS.jsonl'))
        self.assertEqual(gov.delivery_dir_of('minifly_b/results/x.json'), 'minifly_b')
        self.assertEqual(gov.delivery_dir_of('studies/rc-20261003/a.md'), 'studies/rc-20261003')
        self.assertEqual(gov.delivery_dir_of('PROJECT_DOCUMENTS/V51_AUDIT_20261020/x.json'),
                         'PROJECT_DOCUMENTS/V51_AUDIT_20261020')
        self.assertIsNone(gov.delivery_dir_of(KR + '/corpus/C01/repo/lab_dev_v1_20261101/ENDPOINT_LOCK.json'))
        tl = self.fixture()
        tl.commit('h1', {KR + '/README.md': 'docs\n', 'README.md': 'root readme\n'}, trailers=CLAUDE)
        self.assertEqual(run_one(gov_ctx(tl.write(), 'b0', 'h1'), 'GOV-SEP').status, PASS)

    def review_record(self, reviewer='codex', base='b0', decision='APPROVE'):
        return json.dumps({'schema': 'minifly.governance_review.v1', 'id': 'G1', 'reviewer_agent': reviewer,
                           'base': base, 'decision': decision, 'utc': '20261101T000000Z'})

    def test_gov_rev_other_family_review(self):
        tl = self.fixture()
        n = gov.SKILL_NAMES[0]
        text = skill_md(n) + 'You MUST run the gate.\nNEVER skip it.\nMore text.\n'
        tl.commit('h1', {'.claude/skills/%s/SKILL.md' % n: text}, trailers=CLAUDE, branches=['origin/claude/x'])
        tl.commit('h2', {gov.GOV_REVIEW_DIR + '/REVIEW_G1.json': self.review_record()},
                  trailers={'Acting-Agent': ['codex']}, branches=['origin/codex/review-g1'])
        tl.commit('h3', {'PROJECT_DOCUMENTS/KB/FINDINGS.jsonl': '{"id": "F-1", "kind": "finding"}\n'},
                  trailers={'Acting-Agent': ['claude:s1'],
                            'Governance-Review': [gov.GOV_REVIEW_DIR + '/REVIEW_G1.json']})
        ctx = gov_ctx(tl.write(), 'b0', 'h3')
        r = run_one(ctx, 'GOV-REV')
        self.assertEqual(r.status, PASS, r.detail)
        self.assertEqual(run_one(ctx, 'GOV-CLASS').status, PASS)
        self.assertEqual(run_one(ctx, 'GOV-UD').status, NOT_APPLICABLE)
        self.assertEqual(run_one(ctx, 'GOV-APPEND').status, NOT_APPLICABLE)

    def test_gov_rev_same_family_or_unbound_fails(self):
        cases = [('claude', 'b0', ['origin/claude/r'], {'Acting-Agent': ['claude:s2']}, 'equals an author family'),
                 ('codex', 'zzzzzzz', ['origin/codex/r'], {'Acting-Agent': ['codex']}, 'does not bind'),
                 ('codex', 'b0', ['origin/claude/r'], {'Acting-Agent': ['codex']}, 'no codex-namespace'),
                 ('codex', 'b0', ['origin/codex/r'], {'Acting-Agent': ['claude:s1']}, 'not the reviewer family')]
        for i, (rev, base, br, tr, msg) in enumerate(cases):
            tl = Timeline(self.tmp / ('r%d' % i))
            tl.commit('b0', base_files(), trailers=CLAUDE)
            n = gov.SKILL_NAMES[0]
            tl.commit('h1', {'.claude/skills/%s/SKILL.md' % n: skill_md(n) + 'You MUST run the gate.\nNEVER skip it.\n'
                             'x\n'}, trailers=CLAUDE)
            tl.commit('h2', {gov.GOV_REVIEW_DIR + '/REVIEW_G1.json': self.review_record(rev, base)}, trailers=tr,
                      branches=br)
            tl.commit('h3', {'README.md': 'x\n'}, trailers={'Acting-Agent': ['claude:s1'], 'Governance-Review': ['G1']})
            r = run_one(gov_ctx(tl.write(), 'b0', 'h3'), 'GOV-REV')
            self.assertEqual(r.status, FAIL, (msg, r.detail))
            self.assertIn(msg, r.detail)

    def test_gov_rev_record_without_range_binds_when_committed_in_range(self):
        for i, (intro_in_range, want) in enumerate(((True, PASS), (False, FAIL))):
            tl = Timeline(self.tmp / ('inr%d' % i))
            tl.commit('b0', base_files(), trailers=CLAUDE)
            rec = json.dumps({'schema': 'minifly.governance_review.v1', 'id': 'G1', 'reviewer_agent': 'codex',
                              'decision': 'APPROVE', 'utc': '20261101T000000Z'})
            if not intro_in_range:  # committed before the range starts: an old record re-cited
                tl.commit('b1', {gov.GOV_REVIEW_DIR + '/REVIEW_G1.json': rec}, trailers={'Acting-Agent': ['codex']},
                          branches=['origin/codex/r'])
            n = gov.SKILL_NAMES[0]
            tl.commit('h1', {'.claude/skills/%s/SKILL.md' % n: skill_md(n) + 'You MUST run the gate.\nNEVER skip it.\n'
                             'x\n'}, trailers=CLAUDE)
            if intro_in_range:
                tl.commit('h2', {gov.GOV_REVIEW_DIR + '/REVIEW_G1.json': rec}, trailers={'Acting-Agent': ['codex']},
                          branches=['origin/codex/r'])
            tl.commit('h3', {'README.md': 'x\n'}, trailers={'Acting-Agent': ['claude:s1'], 'Governance-Review': ['G1']})
            base = 'b0' if intro_in_range else 'b1'
            r = run_one(gov_ctx(tl.write(), base, 'h3'), 'GOV-REV')
            self.assertEqual(r.status, want, r.detail)
            if want == FAIL:
                self.assertIn('does not bind', r.detail)

    def test_gov_append(self):
        tl = self.fixture()
        tl.commit('h1', {'PROJECT_DOCUMENTS/KB/FINDINGS.jsonl': '{"id": "F-2", "kind": "finding"}\n'}, trailers=CLAUDE)
        r = run_one(gov_ctx(tl.write(), 'b0', 'h1'), 'GOV-APPEND')
        self.assertEqual(r.status, FAIL, r.detail)
        tl2 = Timeline(self.tmp / 'ok')
        tl2.commit('b0', base_files(), trailers=CLAUDE)
        tl2.commit('h1', {'PROJECT_DOCUMENTS/KB/FINDINGS.jsonl': '{"id": "F-1", "kind": "finding"}\n'
                          '{"id": "F-2", "kind": "finding"}\n'}, trailers=CLAUDE)
        self.assertEqual(run_one(gov_ctx(tl2.write(), 'b0', 'h1'), 'GOV-APPEND').status, PASS)

    def test_gov_skill_limits(self):
        n = gov.SKILL_NAMES[1]
        self.assertEqual(gov.skill_problems(n, skill_md(n)), [])
        self.assertTrue(any('1025 characters' in p for p in gov.skill_problems(n, skill_md(n, desc='x' * 1025))))
        self.assertTrue(any('under 400' in p for p in gov.skill_problems(n, skill_md(n, body_lines=400))))
        self.assertEqual(gov.skill_problems(n, skill_md(n, body_lines=399)), [])
        self.assertTrue(any('does not equal' in p for p in gov.skill_problems(n, skill_md('other'))))
        self.assertTrue(any('no YAML' in p for p in gov.skill_problems(n, '# no frontmatter\n')))
        folded = '---\nname: %s\ndescription: >\n  Use when a\n  b happens.\n---\nbody\n' % n
        fm, body, probs = gov.parse_frontmatter(folded)
        self.assertEqual(fm['description'], 'Use when a b happens.')
        self.assertEqual(body, ['body'])
        tl = self.fixture()
        tl.commit('h1', {'PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md': 'only minifly-governance\n'}, trailers=CLAUDE)
        r = run_one(gov_ctx(tl.write(), 'b0', 'h1'), 'GOV-SKILL')
        self.assertEqual(r.status, FAIL)
        self.assertIn('does not list', r.detail)
        tl3 = Timeline(self.tmp / 'agents')
        tl3.commit('b0', base_files(), trailers=CLAUDE)
        tl3.commit('h1', {'AGENTS.md': 'local pointer without skills\n'}, trailers=CLAUDE)
        r = run_one(gov_ctx(tl3.write(), 'b0', 'h1'), 'GOV-SKILL')
        self.assertEqual(r.status, FAIL)
        self.assertIn('AGENTS.md does not list', r.detail)

    def test_gov_self_timeline(self):
        tl = self.fixture()
        tl.commit('h1', {'README.md': 'x\n'}, trailers=CLAUDE)
        self.assertEqual(run_one(gov_ctx(tl.write(), 'b0', 'h1'), 'GOV-SELF').status, PASS)
        fx = tl.write(dirty=[KR + '/x.py'])
        self.assertEqual(run_one(gov_ctx(fx, 'b0', 'h1'), 'GOV-SELF').status, FAIL)

    def test_declared_monotone_and_missing_base(self):
        tl = self.fixture()
        e = json.loads(base_files()[KR + '/EXPECTED.json'])
        e['C01']['must_fire'] = []
        tl.commit('h1', {KR + '/EXPECTED.json': json.dumps(e)}, trailers=CLAUDE)
        ctx = gov_ctx(tl.write(), 'b0', 'h1')
        r = run_one(ctx, 'GOV-MONO')
        self.assertEqual(r.status, FAIL)
        self.assertIn('must_fire lost A-01', r.detail)
        bad = gov_ctx(tl.write(), 'nope', 'h1')
        self.assertEqual(run_one(bad, 'GOV-SEP').status, MISSING)
        rev = gov_ctx(tl.write(), 'h1', 'b0')
        self.assertEqual(run_one(rev, 'GOV-SEP').status, FAIL)


FAKE_CHECKER = textwrap.dedent('''\
    import json, os, sys
    BEHAVIOR = %s
    argv = sys.argv[1:]
    repo = argv[argv.index('--repo-root') + 1]
    cmd = [a for a in argv if a not in ('--json',)]
    cmd = cmd[cmd.index('--git') + 2:] if '--git' in cmd else cmd
    marker = os.path.join(repo, 'MODE')
    mode = open(marker).read().strip() if os.path.exists(marker) else 'clean'
    fired = BEHAVIOR.get(mode, [])
    res = [{"id": i, "status": "FAIL"} for i in fired] or [{"id": "A-01", "status": "PASS"}]
    print(json.dumps({"results": res, "exit_code": 1 if fired else 0}))
    sys.exit(1 if fired else 0)
    ''')


def fake_kit_files(behavior):
    return {KR + '/minifly_check.py': FAKE_CHECKER % repr(behavior), KR + '/common.py': '# fake\n',
            KR + '/RULES.json': json.dumps({'rules': [{'id': 'A-01', 'fm_ids': ['FM-01']},
                                                       {'id': 'B-01', 'fm_ids': ['FM-02']}], 'bundles': {}})}


def nested_fixture_files(fid, mode):
    pre = '%s/corpus/%s/' % (KR, fid)
    return {pre + 'timeline.json': json.dumps({'head': 'n1', 'commits': [{'sha': 'n1', 'paths': ['MODE']}]}),
            pre + 'repo/MODE': mode + '\n', pre + 'FIXTURE.json': json.dumps({'id': fid, 'kind':
                                                                               'clean' if mode == 'clean' else 'dirty'})}


class TestCmdGovernance(TmpCase):
    def test_dispatcher_json_report_and_exit(self):
        import minifly_check
        tl = self.fixture()
        n = gov.SKILL_NAMES[4]
        tl.commit('h1', {'.claude/skills/%s/SKILL.md' % n: skill_md(n) + 'You MUST run the gate.\n'}, trailers=CLAUDE)
        fx = tl.write()
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = minifly_check.main(['--repo-root', str(fx / 'repo'), '--git', 'timeline:%s' % fx, '--json',
                                     'governance', '--base', 'b0', '--head', 'h1'])
        rep = json.loads(out.getvalue())
        self.assertEqual(rc, 1)
        self.assertEqual(rep['schema'], gov.GOV_REPORT_SCHEMA)
        st = {r['id']: r['status'] for r in rep['results']}
        self.assertEqual(st['GOV-CLASS'], FAIL)
        self.assertEqual(st['GOV-UD'], FAIL)
        self.assertEqual(st['GOV-SELF'], PASS)
        self.assertEqual([c['class'] for c in rep['changes']], ['loosen'])
        self.assertEqual(rep['verdict'], 'FAIL')


class TestObservedMonotone(TmpCase):
    def build(self, base_behavior, head_behavior):
        tl = Timeline(self.tmp / 'obs')
        files = base_files()
        files.update(fake_kit_files(base_behavior))
        files.update(nested_fixture_files('C00a', 'clean'))
        files.update(nested_fixture_files('C01', 'dirty'))
        files[KR + '/EXPECTED.json'] = json.dumps({f: {'commands': [['gate', '--stage', 'lock', 'd']],
                                                        'expected_exit': [0 if f == 'C00a' else 1],
                                                        'must_fire': [] if f == 'C00a' else ['A-01']}
                                                    for f in ('C00a', 'C01')})
        tl.commit('b0', files, trailers=CLAUDE)
        tl.commit('h1', {KR + '/minifly_check.py': FAKE_CHECKER % repr(head_behavior)}, trailers=CLAUDE)
        return tl.write()

    def test_monotone_code_edit(self):
        fx = self.build({'dirty': ['A-01']}, {'dirty': ['A-01', 'B-01']})
        ctx = gov_ctx(fx, 'b0', 'h1')
        r = run_one(ctx, 'GOV-MONO')
        self.assertEqual(r.status, PASS, r.detail)
        a = gov.analyze(ctx)
        self.assertEqual([c['class'] for c in a['changes']], ['tighten_or_neutral'])

    def test_non_monotone_code_edit_is_loosen(self):
        for head, msg in (({'dirty': []}, 'no longer fires A-01'), ({'dirty': ['A-01'], 'clean': ['B-01']},
                                                                     'clean base fixture C00a is not clean')):
            fx = self.build({'dirty': ['A-01']}, head)
            ctx = gov_ctx(fx, 'b0', 'h1')
            r = run_one(ctx, 'GOV-MONO')
            self.assertEqual(r.status, FAIL)
            self.assertIn(msg, r.detail)
            self.assertEqual(gov.analyze(ctx)['changes'][0]['class'], 'loosen')
            self.assertEqual(run_one(ctx, 'GOV-CLASS').status, FAIL)
            shutil.rmtree(str(fx))

    def test_unrunnable_kit_overlay_without_corpus(self):
        tl = self.fixture()
        tl.commit('h1', {KR + '/newcheck.py': 'X = 1\n'}, trailers=CLAUDE)
        ctx = gov_ctx(tl.write(), 'b0', 'h1')
        r = run_one(ctx, 'GOV-MONO')
        self.assertEqual(r.status, FAIL)
        self.assertIn('no corpus fixtures', r.detail)


class TestGovCoi(TmpCase):
    def make(self, trailers_after, ud_scope=None):
        tl = Timeline(self.tmp / 'coi')
        files = base_files()
        d = 'lab_dev_v1_20261101'
        files[d + '/ENDPOINT_LOCK.json'] = json.dumps({'schema': common.SCHEMA_ENDPOINT_LOCK, 'author_agent': 'codex'})
        if ud_scope:
            files['PROJECT_DOCUMENTS/KB/USER_DECISIONS.jsonl'] = json.dumps({'id': 'UD-1', 'scope': [ud_scope]}) + '\n'
        tl.commit('c1', files, trailers=CODEX)
        tl.commit('c2', {KR + '/lockcheck.py': '# edit\n'}, trailers=trailers_after)
        fx = tl.write()
        g = common.TimelineGit(fx)
        return common.Ctx(repo_root=g.repo, delivery=g.repo / d, git=g, kb=common.KB(g.repo), rules=common.load_rules(),
                          stage='official', options={})

    def test_coi(self):
        self.assertEqual(run_one(self.make(CLAUDE), 'GOV-COI').status, PASS)
        r = run_one(self.make(CODEX), 'GOV-COI')
        self.assertEqual(r.status, FAIL)
        self.assertIn('c2', r.detail)
        self.assertEqual(run_one(self.make({}), 'GOV-COI').status, FAIL)
        self.assertEqual(run_one(self.make(CODEX, 'coi:lab_dev_v1_20261101'), 'GOV-COI').status, PASS)
        self.assertEqual(run_one(self.make(CODEX, 'amendment:lab_dev_v1_20261101'), 'GOV-COI').status, FAIL)


class TestHelpers(unittest.TestCase):
    def test_json_pointer_and_expect(self):
        obj = {'a': {'b': [1, {'c~d': 'x/y'}]}, 'v': 'FAIL', 'n': 0.30000000000000004}
        self.assertEqual(gov.json_pointer(obj, '/a/b/1/c~0d'), 'x/y')
        with self.assertRaises(KeyError):
            gov.json_pointer(obj, '/a/b/2')
        self.assertTrue(gov.expect_matches('best of k=3 FIXED_1', {'contains': 'FIXED_1'}))
        self.assertFalse(gov.expect_matches('x', {'contains': 'FIXED_1'}))
        self.assertTrue(gov.expect_matches(obj['n'], 0.3))
        self.assertFalse(gov.expect_matches(True, 1))
        self.assertTrue(gov.expect_matches(0.41, {'approx': 0.4, 'tol': 0.02}))
        self.assertTrue(gov.expect_matches('FAIL', {'in': ['FAIL', 'INVALID']}))

    def test_readme_planned_ids(self):
        text = '# x\n\n## Not yet enforced\n\nintro\n\n- `LOCK-07`: reason here\n- `ADM-TB-CEIL`: r\n\n## Next\n- `X-01`: n\n'
        ids, probs = gov.readme_planned_ids(text)
        self.assertEqual(ids, {'LOCK-07', 'ADM-TB-CEIL'})
        self.assertEqual(probs, [])
        ids, probs = gov.readme_planned_ids('## Not yet enforced\n- (none at delivery)\n- bad bullet\n')
        self.assertEqual(ids, set())
        self.assertEqual(len(probs), 1)
        self.assertTrue(gov.readme_planned_ids('no section')[1])

    def test_real_readme_matches_real_rules(self):
        rules = common.load_rules()
        ids, probs = gov.readme_planned_ids((KIT / 'README.md').read_text())
        planned = {r['id'] for r in rules['rules'] if r.get('status') == 'planned'}
        self.assertEqual(probs, [])
        self.assertEqual(ids, planned)


# --------------------------------------------------------------------------------------
# selftest engine with a fake in-process invoker
# --------------------------------------------------------------------------------------

def mk_rules(ids, planned=()):
    rules = []
    for i in list(ids) + list(planned):
        rules.append({'id': i, 'module': 'test_governance', 'stages': ['lock'], 'tier_min': 0,
                      'introduced_in': '1.0.0', 'status': 'planned' if i in planned else 'enforced', 'fm_ids': [],
                      'applies_when': 'always'})
    return {'rules': rules, 'bundles': {'lock': {'checks': list(ids)}}}


def readme_for(planned):
    lines = ['## Not yet enforced', ''] + ['- `%s`: not implemented' % p for p in planned] + ['- (none)'] * (not planned)
    return '\n'.join(lines) + '\n'


@common.check('TG-01')
def _tg_01(ctx):
    return Result('TG-01', FAIL if ctx == 'dirty' else PASS, 'synthetic')


@common.check('TG-02')
def _tg_02(ctx):
    return Result('TG-02', PASS, 'synthetic; never discriminates')


SCRIPT_KEY = 'SCRIPT.json'


def fake_invoker(log=None):
    """Interprets <repo>/SCRIPT.json: {mode, checks: [ids], write: {rel: obj}, warn: [ids]}; 'canary' commands
    write a CANARY report with CAN-01..03 PASS."""
    def invoke(argv, cwd, copy_root):
        repo = Path(argv[argv.index('--repo-root') + 1])
        cmd = argv[argv.index('--json') + 1:]
        if log is not None:
            log.append(cmd)
        spec = json.loads((repo / SCRIPT_KEY).read_text())
        if cmd[0] == 'canary':
            rep = {'results': [{'id': c, 'status': PASS} for c in ('CAN-01', 'CAN-02', 'CAN-03')]}
            (repo / cmd[1] / 'checks').mkdir(parents=True, exist_ok=True)
            (repo / cmd[1] / 'checks' / 'CANARY_x.json').write_text(json.dumps(rep))
            return 0, '', '', []
        results = []
        for rid in spec.get('checks', []):
            r = common.CHECKS[rid](spec.get('mode'))
            results.append({'id': r.id, 'status': r.status})
        for rid in spec.get('warn', []):
            results.append({'id': rid, 'status': WARN})
        for rel, obj in (spec.get('write') or {}).items():
            p = repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(obj))
        rc = 1 if any(r['status'] == FAIL for r in results) else 0
        d = repo / 'lab_dev_v1_20261101' / 'checks'
        d.mkdir(parents=True, exist_ok=True)
        n = len(list(d.iterdir()))
        (d / ('GATE_lock_%d.json' % n)).write_text(json.dumps({'schema': common.SCHEMA_GATE_REPORT,
                                                              'results': results}))
        return rc, json.dumps({'results': results}), '', []
    return invoke


class TestSelftestEngine(TmpCase):
    def corpus(self, fixtures, expected):
        cdir = self.tmp / 'corpus'
        for fid, (spec, kind) in fixtures.items():
            fx = cdir / fid
            (fx / 'repo' / 'lab_dev_v1_20261101').mkdir(parents=True)
            (fx / 'repo' / 'lab_dev_v1_20261101' / 'ENDPOINT_LOCK.json').write_text(
                json.dumps({'schema': common.SCHEMA_ENDPOINT_LOCK}))
            (fx / 'repo' / SCRIPT_KEY).write_text(json.dumps(spec))
            (fx / 'timeline.json').write_text(json.dumps({'commits': [{'sha': 'a'}]}))
            (fx / 'FIXTURE.json').write_text(json.dumps({'id': fid, 'kind': kind}))
        ep = self.tmp / 'EXPECTED.json'
        ep.write_text(json.dumps(expected))
        return cdir, ep

    def std(self):
        fixtures = {'C00a': ({'mode': 'clean', 'checks': ['TG-01', 'TG-02'],
                              'write': {'lab_dev_v1_20261101/results/official/VERDICT.json': {'verdict': 'PASS'}}},
                             'clean'),
                    'C01': ({'mode': 'dirty', 'checks': ['TG-01', 'TG-02'], 'warn': ['TG-02']}, 'dirty')}
        cmd = [['gate', '--stage', 'lock', 'lab_dev_v1_20261101']]
        expected = {'C00a': {'commands': cmd, 'expected_exit': [0], 'must_fire': [], 'must_not_fire': [],
                             'must_warn': [], 'expect': {'/verdict': 'PASS'}},
                    'C01': {'commands': cmd, 'expected_exit': [1], 'must_fire': ['TG-01'], 'must_not_fire': ['TG-02'],
                            'must_warn': ['TG-02'], 'expect': {}}}
        return fixtures, expected

    def cfg(self, cdir, ep, cache=None, planned=(), readme=None, **kw):
        rules = mk_rules([i for i in ('TG-01', 'TG-02') if i not in planned], planned)
        return gov.SelftestConfig(kit_dir=self.tmp, corpus_dir=cdir, expected_path=ep, rules=rules,
                                  readme_text=readme if readme is not None else readme_for(planned),
                                  invoke=kw.pop('invoke', fake_invoker()), checks={'TG-01': _tg_01, 'TG-02': _tg_02},
                                  import_modules=False, cache_writer=cache, **kw)

    def test_green_corpus_writes_cache(self):
        fixtures, expected = self.std()
        cdir, ep = self.corpus(fixtures, expected)
        written = []
        rc, rep = gov.run_selftest(self.cfg(cdir, ep, cache=lambda sha, s: written.append(sha) or 'cache'))
        self.assertEqual(rc, 0, json.dumps(rep, default=str)[:2000])
        self.assertEqual(rep['fixtures']['C01']['fired'], ['TG-01'])
        self.assertTrue(rep['canary']['ok'])
        self.assertEqual(written, [rep['checker_sha256']])

    def test_failures_are_reported_and_no_cache(self):
        fixtures, expected = self.std()
        expected['C01']['must_fire'] = ['TG-02']
        expected['C01']['expected_exit'] = [0]
        expected['C00a']['expect'] = {'/verdict': 'FAIL', '/headline': {'contains': 'x'}}
        cdir, ep = self.corpus(fixtures, expected)
        written = []
        rc, rep = gov.run_selftest(self.cfg(cdir, ep, cache=lambda sha, s: written.append(sha)))
        self.assertEqual(rc, 1)
        self.assertEqual(written, [])
        p01 = ' '.join(rep['fixtures']['C01']['problems'])
        self.assertIn('must_fire not fired: TG-02', p01)
        self.assertIn('exited 1, expected 0', p01)
        p00 = ' '.join(rep['fixtures']['C00a']['problems'])
        self.assertIn("has 'PASS', want 'FAIL'", p00)
        self.assertIn('no artifact produced', p00)

    def test_clean_fixture_must_not_fire_and_missing_entries(self):
        fixtures, expected = self.std()
        fixtures['C00a'][0]['mode'] = 'dirty'
        fixtures['C02'] = ({'mode': 'clean', 'checks': []}, 'dirty')
        del expected['C01']
        expected['C03'] = expected['C00a']
        cdir, ep = self.corpus(fixtures, expected)
        rc, rep = gov.run_selftest(self.cfg(cdir, ep))
        self.assertEqual(rc, 1)
        self.assertIn('clean fixture fired: TG-01', ' '.join(rep['fixtures']['C00a']['problems']))
        self.assertIn('without an EXPECTED entry', ' '.join(rep['fixtures']['C02']['problems']))
        self.assertIn('without a fixture dir', ' '.join(rep['fixtures']['C03']['problems']))

    def test_preflight_rules_registry_readme(self):
        fixtures, expected = self.std()
        cdir, ep = self.corpus(fixtures, expected)
        rc, rep = gov.run_selftest(self.cfg(cdir, ep, planned=('TG-09',), readme=readme_for([])))
        self.assertEqual(rc, 1)
        self.assertTrue(any('omits planned id(s): TG-09' in p for p in rep['preflight_problems']))
        probs = gov.preflight(mk_rules(['TG-01', 'TG-03']), readme_for([]), {'TG-01': _tg_01, 'TG-02': _tg_02}, False)
        self.assertTrue(any('TG-03' in p and 'neither registered nor planned' in p for p in probs))
        self.assertTrue(any('registered check TG-02 is not in RULES.json' in p for p in probs))
        rules = mk_rules(['TG-01'], planned=('TG-02',))
        rules['bundles']['lock']['checks'].append('TG-02')
        probs = gov.preflight(rules, readme_for(['TG-02']), {'TG-01': _tg_01}, False)
        self.assertTrue(any('planned id TG-02' in p for p in probs))

    def test_only_fixture_runs_subset_and_never_writes_cache(self):
        fixtures, expected = self.std()
        cdir, ep = self.corpus(fixtures, expected)
        written, calls = [], []
        rc, rep = gov.run_selftest(self.cfg(cdir, ep, cache=lambda s, x: written.append(s),
                                            invoke=fake_invoker(calls)), only='C01')
        self.assertEqual(rc, 0)
        self.assertEqual(sorted(rep['fixtures']), ['C01'])
        self.assertEqual(written, [])
        self.assertNotIn('canary', rep)
        with self.assertRaises(common.InvalidInput):
            gov.run_selftest(self.cfg(cdir, ep), only='NOPE')

    def test_mutate_reports_coverage(self):
        fixtures, expected = self.std()
        cdir, ep = self.corpus(fixtures, expected)
        rc, rep = gov.run_selftest(self.cfg(cdir, ep), mutate=True)
        m = rep['mutate_report']['checks']
        self.assertEqual(m['TG-01']['status'], 'ok')
        self.assertEqual(m['TG-01']['exercised_by'], 'C01')
        self.assertEqual(m['TG-01']['clean_coverage_by'], 'C00a')
        # TG-02 always passes, so an always-PASS replacement changes no dirty fixture's outcome
        self.assertEqual(m['TG-02']['status'], 'not exercised')
        self.assertEqual(m['TG-02']['clean_coverage_by'], 'C00a')
        self.assertEqual(rc, 1)
        self.assertIs(common.CHECKS['TG-01'], _tg_01)  # registry restored
        self.assertIs(common.CHECKS['TG-02'], _tg_02)

    def test_mutate_green_and_no_clean_coverage(self):
        fixtures, expected = self.std()
        cdir, ep = self.corpus(fixtures, expected)
        rc, rep = gov.run_selftest(self.cfg(cdir, ep, planned=('TG-02',)), mutate=True)
        self.assertEqual(sorted(rep['mutate_report']['checks']), ['TG-01'])
        self.assertEqual(rc, 0, rep['mutate_report'])
        fixtures['C00a'][0]['checks'] = ['TG-02']
        shutil.rmtree(str(cdir))
        cdir, ep = self.corpus(fixtures, expected)
        rc, rep = gov.run_selftest(self.cfg(cdir, ep, planned=('TG-02',)), mutate=True)
        self.assertEqual(rep['mutate_report']['checks']['TG-01']['status'], 'no clean coverage')
        self.assertEqual(rc, 1)

    def test_expected_exit_3_is_not_a_problem(self):
        oc = gov.RunOutcome()
        oc.exits = [3]
        oc.statuses = {'SELF-01': {FAIL}}
        oc.exit3_errors = [(0, 'command gate exit 3: SELF-01 FAIL: no gate report is written')]
        entry = {'commands': [['gate', '--stage', 'lock', 'd']], 'expected_exit': [3], 'must_fire': ['SELF-01']}
        self.assertEqual(gov.evaluate_fixture('S01', entry, oc, False), [])
        entry['expected_exit'] = [1]
        probs = gov.evaluate_fixture('S01', entry, oc, False)
        self.assertTrue(any('exit 3: SELF-01 FAIL' in p for p in probs), probs)

    def test_stdout_object_is_an_artifact_when_nothing_is_written(self):
        fixtures, expected = self.std()
        cdir, ep = self.corpus(fixtures, expected)

        def inv(argv, cwd, copy_root):
            return 0, json.dumps({'schema': 'minifly.kb_score.v1', 'brier_mean': 0.25}), '', []
        oc = gov.run_fixture(cdir / 'C01', [['kb', 'score']], inv)
        self.assertEqual(gov._check_expect('/brier_mean', 0.25, oc), [])
        self.assertTrue(gov._check_expect('/brier_mean', 0.5, oc))

    def test_canary_step_uses_the_c00a_canary_command(self):
        fixtures, expected = self.std()
        expected['C00a']['commands'] = expected['C00a']['commands'] + [['canary', 'lab_dev_v1_20261101', '--seed',
                                                                         gov.CANARY_SEED]]
        expected['C00a']['expected_exit'] = [0, 0]
        cdir, ep = self.corpus(fixtures, expected)
        calls = []
        rc, rep = gov.run_selftest(self.cfg(cdir, ep, invoke=fake_invoker(calls)))
        self.assertEqual(rc, 0, json.dumps(rep, default=str)[:2000])
        self.assertEqual(sum(1 for c in calls if c and c[0] == 'canary'), 1)  # not re-run on a fresh copy
        self.assertTrue(rep['canary']['ok'])
        self.assertIn('command 2 of the C00a run', rep['canary']['source'])

    def test_expected_validation(self):
        probs = gov.validate_expected({'X': {'commands': [['gate', '--repo-root', '/x']], 'expected_exit': [0, 1],
                                             'must_fire': ['NOPE']}}, {'TG-01'})
        text = ' '.join(probs['X'])
        self.assertIn('--repo-root', text)
        self.assertIn('2 entries for 1 commands', text)
        self.assertIn('NOPE', text)

    def test_inprocess_invoker_runs_dispatcher(self):
        inv = gov._inprocess_invoker()
        rc, out, err, cap = inv(['--help'], self.tmp, self.tmp)
        self.assertEqual(rc, 0)
        self.assertIn('selftest', out)
        rc, out, err, cap = inv(['no-such-command'], self.tmp, self.tmp)
        self.assertEqual(rc, 3)


# --------------------------------------------------------------------------------------
# held-out canaries
# --------------------------------------------------------------------------------------

def make_batch(root, n=10, n_clean=3):
    root = Path(root)
    key = {}
    for i in range(n):
        fid = 'H%02d' % i
        fx = root / fid
        (fx / 'repo').mkdir(parents=True)
        (fx / 'repo' / 'MODE').write_text('clean\n' if i < n_clean else 'dirty\n')
        (fx / 'timeline.json').write_text(json.dumps({'head': 'n1', 'commits': [{'sha': 'n1', 'paths': ['MODE']}]}))
        key[fid] = [] if i < n_clean else (['FM-01'] if i % 2 else ['B-01'])
    (root / 'KEY.json').write_text(json.dumps(key))
    return root


class TestHeldout(TmpCase):
    def test_seal_open_round_trip_and_tamper(self):
        batch = make_batch(self.tmp / 'batch')
        info = gov.validate_batch(batch)
        self.assertEqual((info['n'], info['n_clean'], info['fm_coverage']), (10, 3, ['FM-01']))
        key = os.urandom(32)
        blob, manifest = gov.seal_batch('HB-1', batch, key)
        self.assertNotIn(b'KEY.json', blob)
        m2, files = gov.open_batch('HB-1', blob, key)
        self.assertEqual(m2, manifest)
        self.assertEqual(files['KEY.json'], (batch / 'KEY.json').read_bytes())
        self.assertEqual(files['H05/repo/MODE'], b'dirty\n')
        tampered = bytearray(blob)
        tampered[len(gov.HB_MAGIC) + gov.HB_NONCE + 5] ^= 1
        with self.assertRaises(common.InvalidInput):
            gov.open_batch('HB-1', bytes(tampered), key)
        with self.assertRaises(common.InvalidInput):
            gov.open_batch('HB-1', blob, os.urandom(32))
        with self.assertRaises(common.InvalidInput):
            gov.open_batch('HB-2', blob, key)
        with self.assertRaises(common.InvalidInput):
            gov.seal_batch('HB-1', batch, b'short')

    def test_batch_rules(self):
        with self.assertRaises(common.InvalidInput):
            gov.validate_batch(make_batch(self.tmp / 'small', n=4, n_clean=1))
        self.assertEqual(gov.validate_batch(make_batch(self.tmp / 'one', n=1, n_clean=0))['n'], 1)
        b = make_batch(self.tmp / 'mismatch')
        (b / 'KEY.json').write_text(json.dumps({'H00': []}))
        with self.assertRaises(common.InvalidInput):
            gov.validate_batch(b)

    def heldout_args(self, repo, action, **kw):
        ns = argparse.Namespace(command='heldout', action=action, batch='HB-1', in_dir=None, key_file=None,
                                checker_sha=None, out=None, repo_root=str(repo), git=None, json=False, extra=[])
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns

    def quiet(self, fn, *a):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = fn(*a)
        return rc, out.getvalue(), err.getvalue()

    def test_cmd_seal_and_reveal(self):
        repo = self.tmp / 'repo'
        (repo / 'PROJECT_DOCUMENTS' / 'KB').mkdir(parents=True)
        batch = make_batch(self.tmp / 'batch')
        kf = self.tmp / 'owner.key'
        kf.write_bytes(os.urandom(32))
        rc, out, err = self.quiet(gov.cmd_heldout, self.heldout_args(repo, 'seal', in_dir=str(batch), key_file=str(kf),
                                                                     extra=['--author', 'owner']))
        self.assertEqual(rc, 0, err)
        row = json.loads(out.strip().splitlines()[-1])
        self.assertEqual((row['n'], row['n_clean'], row['author_identity']), (10, 3, 'owner'))
        sealed = gov._default_sealed_path(repo, 'HB-1')
        self.assertTrue(sealed.is_file())
        self.assertEqual(row['bundle_sha256'], common.sha256_file(sealed))
        rc2, _, _ = self.quiet(gov.cmd_heldout, self.heldout_args(repo, 'seal', in_dir=str(batch), key_file=str(kf),
                                                                  extra=['--author', 'owner']))
        self.assertEqual(rc2, 1)  # sealed batches are immutable
        (repo / 'PROJECT_DOCUMENTS' / 'KB' / 'CANARY_COMMITMENTS.jsonl').write_text(json.dumps(row) + '\n')
        dest = self.tmp / 'revealed'
        rc, out, err = self.quiet(gov.cmd_heldout, self.heldout_args(repo, 'reveal', key_file=str(kf), out=str(dest)))
        self.assertEqual(rc, 2, out + err)  # HELD-01/03 need the checker under test: MISSING
        self.assertIn('PASS            HELD-02', out)
        self.assertEqual((dest / 'H04' / 'repo' / 'MODE').read_text(), 'dirty\n')
        # tampered bundle: HELD-02 FAIL and nothing revealed
        data = bytearray(sealed.read_bytes())
        data[-40] ^= 0xFF
        sealed.write_bytes(bytes(data))
        dest2 = self.tmp / 'revealed2'
        rc, out, err = self.quiet(gov.cmd_heldout, self.heldout_args(repo, 'reveal', key_file=str(kf), out=str(dest2)))
        self.assertEqual(rc, 1)
        self.assertIn('FAIL            HELD-02', out)
        self.assertFalse(dest2.exists())

    def test_key_and_plaintext_outside_repo(self):
        repo = self.tmp / 'repo'
        repo.mkdir()
        kf = repo / 'k'
        kf.write_bytes(os.urandom(32))
        rc, _, _ = self.quiet(gov.cmd_heldout, self.heldout_args(repo, 'seal', key_file=str(kf)))
        self.assertEqual(rc, 1)

    def test_held_01_03_and_score_timeline(self):
        tl = Timeline(self.tmp / 'hfx')
        files = base_files()
        files.update(fake_kit_files({'dirty': ['A-01']}))
        tl.commit('k1', files, trailers=CLAUDE)
        batch = make_batch(self.tmp / 'batch')
        kf = self.tmp / 'owner.key'
        kf.write_bytes(os.urandom(32))
        blob, _ = gov.seal_batch('HB-1', batch, kf.read_bytes())
        row = {'batch_id': 'HB-1', 'author_identity': 'owner', 'utc': '20261101T000000Z',
               'bundle_sha256': common.sha256_bytes(blob), 'key_sha256': common.sha256_file(kf), 'n': 10, 'n_clean': 3,
               'fm_coverage': ['FM-01'], 'checker_sha_under_test': None, 'revealed_utc': None}
        tl.commit('r1', {gov.CANARY_REL: json.dumps(row) + '\n'}, trailers={})
        tl.commit('k2', {KR + '/minifly_check.py': FAKE_CHECKER % repr({'dirty': ['A-01', 'B-01']})}, trailers=CLAUDE)
        fx = tl.write()
        sealed = gov._default_sealed_path(fx / 'repo', 'HB-1')
        sealed.parent.mkdir(parents=True, exist_ok=True)
        sealed.write_bytes(blob)
        g = common.TimelineGit(fx)
        ctx = common.Ctx(repo_root=g.repo, delivery=None, git=g, kb=common.KB(g.repo), rules=common.load_rules(),
                         stage='heldout', options={'heldout': {'batch_id': 'HB-1', 'row': row,
                                                               'bundle_path': str(sealed), 'key_path': str(kf),
                                                               'checker_commit': 'k2'}})
        self.assertEqual(run_one(ctx, 'HELD-01').status, PASS)
        self.assertEqual(run_one(ctx, 'HELD-02').status, PASS)
        self.assertEqual(run_one(ctx, 'HELD-03').status, PASS)
        ctx.options['heldout']['checker_commit'] = 'k1'
        ctx._cache.clear()
        self.assertEqual(run_one(ctx, 'HELD-01').status, FAIL)  # the checker merged before the commitment
        ctx.options['heldout']['checker_commit'] = 'k2'
        ctx.options['heldout']['row'] = dict(row, author_identity='claude:s1')
        self.assertEqual(run_one(ctx, 'HELD-03').status, FAIL)
        # end-to-end score with the timeline checker at k2 (a fake kit written by this test)
        out = self.tmp / 'HELDOUT.json'
        args = argparse.Namespace(command='heldout', action='score', batch='HB-1', in_dir=None, key_file=str(kf),
                                  checker_sha='k2', out=str(out), repo_root=str(fx / 'repo'), git='timeline:%s' % fx,
                                  json=False, extra=[])
        rc, so, se = self.quiet(gov.cmd_heldout, args)
        self.assertEqual(rc, 0, so + se)
        rep = json.loads(out.read_text())
        self.assertTrue(rep['heldout_valid'])
        sc = rep['score']
        self.assertEqual(sc['n_clean'], 3)
        self.assertEqual(sc['false_alarm_rate_clean'], 0.0)
        self.assertEqual(sc['recall_fixtures'], 1.0)
        self.assertEqual(sc['recall_per_class']['FM-01']['recall'], 1.0)


class TestRefuseOut(TmpCase):
    def test_refuse_out(self):
        repo = self.tmp / 'repo'
        (repo / 'lab_dev_v1_20261101').mkdir(parents=True)
        (repo / 'PROJECT_DOCUMENTS' / 'KB' / 'RETRO').mkdir(parents=True)
        self.assertIsNotNone(gov._refuse_out(repo, repo / 'lab_dev_v1_20261101' / 'out'))
        self.assertIsNotNone(gov._refuse_out(repo, repo / 'minifly_b' / 'x'))
        self.assertIsNotNone(gov._refuse_out(repo, repo / KR / 'x'))
        self.assertIsNotNone(gov._refuse_out(repo, repo))
        self.assertIsNone(gov._refuse_out(repo, repo / 'PROJECT_DOCUMENTS' / 'KB' / 'RETRO'))
        self.assertIsNone(gov._refuse_out(repo, self.tmp / 'scratch'))


if __name__ == '__main__':
    unittest.main()
