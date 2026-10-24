"""Unit tests for common.py (WP1). Synthetic data in temp dirs only; stdlib only."""
import gzip
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
import zipfile
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

import common  # noqa: E402
from common import (ADVISORY_FAIL, CHECKER_ERROR, FAIL, MISSING, NOT_APPLICABLE, PASS, WARN,  # noqa: E402
                    CheckerError, InvalidInput, MissingInput, Result)

HAVE_GIT = shutil.which('git') is not None

SPEC_IDS = (
    ['SELF-01', 'SCHEMA-01', 'TIER-01', 'RUN-01', 'RUN-02', 'CYC-01', 'CYC-02', 'CYC-03']
    + ['RUN-03', 'RUN-04']  # SPEC 20.7, 20.8 (hardening round 4)
    + ['CLM-17', 'CYC-04']  # SPEC 20.13, 20.16 (self-improvement round 1, introduced_in 1.0.1)
    + ['LOCK-%02d' % i for i in range(1, 14)] + ['AMD-%02d' % i for i in range(1, 6)] + ['WIT-01']
    + ['SEAL-%02d' % i for i in range(1, 6)] + ['CONF-01']
    + ['MET-01', 'MET-02', 'LAD-01', 'LAD-02', 'LAD-03', 'LAD-04', 'STR-01', 'DET-01', 'INERT-01', 'UDI-01',
       'OC-01', 'DEC-01'] + ['RED-%02d' % i for i in range(1, 12)]
    + ['ADM-01', 'ADM-TB-CEIL', 'ADM-TB-FLOOR', 'ADM-CTRL-ZERO', 'ADM-CTRL-ISO', 'ADM-CTRL-DOSE', 'ADM-DIR',
       'ADM-FIX', 'ADM-GATE', 'ADM-INPUT', 'ADM-ACQ', 'ADM-SHUF', 'ADM-EXPO', 'ADM-BUDGET', 'ADM-REUSE']
    + ['AUT-%02d' % i for i in range(1, 5)] + ['REV-%02d' % i for i in range(1, 9)] + ['DEBT-01', 'DEBT-02']
    + ['CAN-01', 'CAN-02', 'CAN-03'] + ['CLM-%02d' % i for i in range(1, 17)] + ['ERR-01', 'WLD-01', 'WLD-02']
    + ['PTR-%02d' % i for i in range(1, 8)] + ['KB-%02d' % i for i in range(1, 6)]
    + ['GOV-SELF', 'GOV-SEP', 'GOV-CLASS', 'GOV-REV', 'GOV-UD', 'GOV-NUM', 'GOV-MONO', 'GOV-APPEND', 'GOV-SKILL',
       'GOV-COI', 'HELD-01', 'HELD-02', 'HELD-03'])

CONSTRAINTS = {
    'schema': 'minifly.constraints.v1',
    'tier_thresholds': {'t0_max_cpu_h': 1, 't1_max_cpu_h': 4, 't1_max_records': 20000, 'family_t1_max_cpu_h': 12},
    'default_grant_caps': {'max_cpu_h': 2, 'max_records': 5000, 'max_cycles': 3},
}


def write(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (dict, list)):
        data = json.dumps(data, indent=1)
    if isinstance(data, str):
        data = data.encode('utf-8')
    p.write_bytes(data)
    return p


def jsonl(rows):
    return ''.join(json.dumps(r) + '\n' for r in rows)


def make_timeline(root, commits, repo_files=None, history=None, head=None, **extra):
    root = Path(root)
    tl = {'head': head or commits[-1]['sha'], 'commits': commits}
    tl.update(extra)
    write(root / 'timeline.json', tl)
    (root / 'repo').mkdir(parents=True, exist_ok=True)
    for rel, data in (repo_files or {}).items():
        write(root / 'repo' / rel, data)
    for (sha, rel), data in (history or {}).items():
        write(root / 'history' / sha / rel, data)
    return root


def commit(sha, paths, branches=(), trailers=None, utc='20261101T000000Z', author='t'):
    return {'sha': sha, 'utc': utc, 'author': author, 'trailers': trailers or {}, 'paths': list(paths),
            'branches': list(branches)}


def lock_obj(name, **over):
    lock = {'schema': common.SCHEMA_ENDPOINT_LOCK, 'delivery': name, 'tier_declared': 1, 'intent': 'diagnostic',
            'budget': {'max_records': 1000, 'max_cpu_h': 1},
            'worlds': {'technical': ['wt'], 'dev': ['wd'], 'rehearsal': ['wr'],
                       'official': {'mode': 'listed', 'ids': [], 'n': 0}, 'official_role_intended': 'PILOT'},
            'files': {'admit': 'admit.py', 'native': ['native.py'], 'runner': ['run.py'], 'params': 'params.json'}}
    lock.update(over)
    return lock


def git(repo, *args, env=None):
    e = dict(os.environ)
    e.update({'GIT_AUTHOR_NAME': 'T', 'GIT_AUTHOR_EMAIL': 't@example.invalid', 'GIT_COMMITTER_NAME': 'T',
              'GIT_COMMITTER_EMAIL': 't@example.invalid', 'GIT_CONFIG_NOSYSTEM': '1'})
    e.update(env or {})
    return subprocess.run(['git', '-c', 'commit.gpgsign=false', '-c', 'init.defaultBranch=main'] + list(args),
                          cwd=str(repo), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=e)


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='mfc_test_'))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------
class TestTimelineGit(TmpCase):
    def setUp(self):
        super().setUp()
        self.c1, self.c2, self.c3 = 'c1' * 20, 'c2' * 20, 'c3' * 20
        make_timeline(self.tmp, [
            commit(self.c1, ['a.txt', 'b.txt', 'd/x.json'], branches=['origin/codex/x'],
                   trailers={'Acting-Agent': 'codex'}),
            commit(self.c2, ['a.txt']),
            commit(self.c3, ['b.txt', 'c.txt'], branches=['origin/claude/y']),
        ], repo_files={'a.txt': 'a2\n', 'c.txt': 'c3\n', 'd/x.json': '{"x": 1}'},
            history={(self.c1, 'a.txt'): 'a1\n', (self.c1, 'b.txt'): 'b1\n'})
        self.g = common.TimelineGit(self.tmp)

    def test_head_log_and_trailers(self):
        self.assertEqual(self.g.head(), self.c3)
        self.assertEqual([c.sha for c in self.g.log()], [self.c1, self.c2, self.c3])
        self.assertEqual([c.sha for c in self.g.log(['a.txt'])], [self.c1, self.c2])
        self.assertEqual([c.sha for c in self.g.log(['d'])], [self.c1])
        self.assertEqual(self.g.log()[0].trailers, {'Acting-Agent': ['codex']})
        self.assertEqual(self.g.log()[0].trailer_values('acting-agent'), ['codex'])

    def test_is_ancestor_linear(self):
        self.assertTrue(self.g.is_ancestor(self.c1, self.c3))
        self.assertTrue(self.g.is_ancestor(self.c2, self.c2))
        self.assertFalse(self.g.is_ancestor(self.c3, self.c1))
        self.assertTrue(self.g.is_ancestor(self.c1[:6], self.c2[:6]))  # unique prefixes
        with self.assertRaises(MissingInput):
            self.g.is_ancestor('deadbeef', self.c1)

    def test_file_at_history_overrides(self):
        self.assertEqual(self.g.file_at(self.c1, 'a.txt'), b'a1\n')          # history override
        self.assertEqual(self.g.file_at(self.c2, 'a.txt'), b'a2\n')          # final toucher -> repo/
        self.assertEqual(self.g.file_at(self.c3, 'a.txt'), b'a2\n')          # unchanged since c2
        self.assertEqual(self.g.file_at(self.c2, 'b.txt'), b'b1\n')          # toucher c1 with history
        self.assertIsNone(self.g.file_at(self.c3, 'b.txt'))                  # deleted at c3
        self.assertIsNone(self.g.file_at(self.c1, 'c.txt'))                  # not yet added
        self.assertEqual(self.g.file_at(self.c3, 'c.txt'), b'c3\n')
        self.assertEqual(self.g.file_at(self.c2, 'd/x.json'), b'{"x": 1}')
        with self.assertRaises(MissingInput):
            self.g.file_at('nope', 'a.txt')
        with self.assertRaises(InvalidInput):
            self.g.file_at(self.c1, '../etc/passwd')

    def test_first_commit_queries(self):
        self.assertEqual(self.g.first_commit_adding('*.txt').sha, self.c1)
        self.assertEqual(self.g.first_commit_adding('c.txt').sha, self.c3)
        self.assertEqual(self.g.first_commit_adding('d/**').sha, self.c1)
        self.assertIsNone(self.g.first_commit_adding('zzz/**'))
        self.assertEqual(self.g.first_commit_where('a.txt', lambda b: b'a2' in b).sha, self.c2)
        self.assertIsNone(self.g.first_commit_where('a.txt', lambda b: b'zz' in b))

    def test_branches_diff_changed(self):
        self.assertEqual(self.g.remote_branches_containing(self.c1), ['origin/codex/x'])
        self.assertEqual(self.g.remote_branches_containing(self.c2), [])
        self.assertEqual(self.g.diff_lines(self.c1, self.c2, 'a.txt'), (['a2'], ['a1']))
        self.assertEqual(self.g.changed_paths(self.c1, self.c3), ['a.txt', 'b.txt', 'c.txt'])
        self.assertEqual(self.g.list_files(self.c3), ['a.txt', 'c.txt', 'd/x.json'])
        self.assertEqual(self.g.list_files(self.c1, 'd'), ['d/x.json'])

    def test_worktree_clean_and_dirty(self):
        self.assertTrue(self.g.worktree_clean(['a.txt', 'c.txt', 'd']))
        write(self.tmp / 'repo' / 'u.txt', 'untracked')
        self.assertEqual(self.g.dirty_paths(['']), ['u.txt'])
        self.assertTrue(self.g.worktree_clean(['a.txt']))
        # history for the final toucher that differs from repo/ = modified worktree
        write(self.tmp / 'history' / self.c2 / 'a.txt', 'a2-committed\n')
        g2 = common.TimelineGit(self.tmp)
        self.assertIn('a.txt', g2.dirty_paths(['a.txt']))
        self.assertEqual(g2.file_at(self.c2, 'a.txt'), b'a2-committed\n')

    def test_freeze_keeps_committed_content_fixed(self):
        # before freezing, an in-place worktree edit of a final-toucher path silently rewrites history too
        write(self.tmp / 'repo' / 'a.txt', 'edited by a command\n')
        self.assertEqual(common.TimelineGit(self.tmp).file_at(self.c2, 'a.txt'), b'edited by a command\n')
        write(self.tmp / 'repo' / 'a.txt', 'a2\n')
        n = common.TimelineGit.freeze(self.tmp)
        self.assertEqual(n, 3)                                  # a.txt at c2, c.txt at c3, d/x.json at c1
        self.assertTrue((self.tmp / 'history' / self.c2 / 'a.txt').is_file())
        self.assertFalse((self.tmp / 'history' / self.c3 / 'b.txt').exists())   # deleted at c3: nothing to freeze
        write(self.tmp / 'repo' / 'a.txt', 'edited by a command\n')
        (self.tmp / 'repo' / 'c.txt').unlink()
        g = common.TimelineGit(self.tmp)
        self.assertEqual(g.file_at(self.c3, 'a.txt'), b'a2\n')
        self.assertEqual(g.file_at(self.c3, 'c.txt'), b'c3\n')
        self.assertEqual(g.dirty_paths(['']), ['a.txt', 'c.txt'])
        self.assertEqual(common.TimelineGit.freeze(self.tmp), 0)  # idempotent

    def test_declared_dirty_and_shallow(self):
        tl = json.loads((self.tmp / 'timeline.json').read_text())
        tl['dirty'] = ['tools/minifly_check/scorer.py']
        tl['shallow'] = True
        write(self.tmp / 'timeline.json', tl)
        g = common.TimelineGit(self.tmp)
        self.assertEqual(g.dirty_paths(['tools/minifly_check']), ['tools/minifly_check/scorer.py'])
        self.assertTrue(g.is_shallow())
        with self.assertRaises(MissingInput):
            g.log()
        with self.assertRaises(MissingInput):
            g.is_ancestor(self.c1, self.c2)


# ------------------------------------------------------------------------------------------------
@unittest.skipUnless(HAVE_GIT, 'git not installed')
class TestCommitFamily(TmpCase):
    """SPEC 20.15 (round 1, INC-HB1-C3): the acting family of a commit comes from its Acting-Agent trailer AND the
    origin/<family>/* namespace that contains it; one function serves every identity check."""

    def make(self, trailers, branches):
        sha = 'f' * 40
        make_timeline(self.tmp, [commit(sha, ['a.txt'], branches=branches, trailers=trailers)],
                      repo_files={'a.txt': 'a\n'})
        g = common.TimelineGit(self.tmp)
        return g, g.commit(sha)

    def fam(self, trailers, branches):
        g, c = self.make(trailers, branches)
        return common.commit_family(g, c)

    def test_trailer_and_namespace_must_agree(self):
        f, st, why = self.fam({'Acting-Agent': 'claude:s1'}, ['origin/claude/review'])
        self.assertEqual((f, st), ('claude', common.IDENTITY_OK))
        # shared history: the reviewer branch and the merged author branch both contain it; the trailer decides
        f, st, why = self.fam({'Acting-Agent': 'claude:s1'}, ['origin/claude/review', 'origin/codex/main'])
        self.assertEqual((f, st), ('claude', common.IDENTITY_OK))
        self.assertIn('shared history', why)
        # a forged reviewer trailer pushed to the author's own namespace (the C3 class)
        f, st, why = self.fam({'Acting-Agent': 'claude'}, ['origin/codex/kb-sync'])
        self.assertEqual((f, st), (None, common.IDENTITY_MISMATCH))
        self.assertIn('no origin/claude/* branch', why)
        # a claude-named branch of another remote is no namespace
        f, st, _ = self.fam({'Acting-Agent': 'claude'}, ['upstream/claude/x', 'origin/codex/y'])
        self.assertEqual(st, common.IDENTITY_MISMATCH)
        f, st, _ = self.fam({'Acting-Agent': 'claude'}, ['upstream/claude/x'])
        self.assertEqual(st, common.IDENTITY_UNPUSHED)
        # no trailer, a trailer naming no family, a trailer naming both: mismatch whatever the branches
        for tr in ({}, {'Acting-Agent': 'gpt'}, {'Acting-Agent': ['claude:s1', 'codex']}):
            self.assertEqual(self.fam(tr, ['origin/claude/r'])[1], common.IDENTITY_MISMATCH, tr)
        self.assertEqual(common.commit_family(None, None)[1], common.IDENTITY_NO_COMMIT)
        g, c = self.make({'Acting-Agent': 'codex'}, ['origin/codex/w'])
        self.assertEqual(common.commit_is_family(g, c, 'codex')[0], True)
        self.assertEqual(common.commit_is_family(g, c, 'claude')[0], False)

    def test_namespace_and_family_parsing(self):
        self.assertEqual(common.branch_namespace('origin/claude/a/b'), 'claude')
        self.assertEqual(common.branch_namespace('remotes/origin/codex/x'), 'codex')
        self.assertIsNone(common.branch_namespace('origin/claudette/x'))
        self.assertIsNone(common.branch_namespace('origin/main'))
        self.assertEqual(common.agent_family('Claude:s-1'), 'claude')
        self.assertEqual(common.agent_family('codex-cli'), 'codex')
        self.assertIsNone(common.agent_family('claudette'))


class TestRealGit(TmpCase):
    def setUp(self):
        super().setUp()
        self.repo = self.tmp / 'r'
        self.repo.mkdir()
        git(self.repo, 'init', '-q')
        write(self.repo / 'a.txt', 'one\n')
        write(self.repo / 'kit' / 'k.py', 'x = 1\n')
        git(self.repo, 'add', '-A')
        git(self.repo, 'commit', '-q', '-m', 'first\n\nActing-Agent: claude:test\nOther-Key: v')
        self.c1 = git(self.repo, 'rev-parse', 'HEAD').stdout.decode().strip()
        write(self.repo / 'a.txt', 'one\ntwo\n')
        write(self.repo / 'b.json', '{}')
        git(self.repo, 'add', '-A')
        git(self.repo, 'commit', '-q', '-m', 'second')
        self.c2 = git(self.repo, 'rev-parse', 'HEAD').stdout.decode().strip()
        self.g = common.RealGit(self.repo)

    def test_history_queries(self):
        self.assertEqual(self.g.head(), self.c2)
        self.assertFalse(self.g.is_shallow())
        log = self.g.log()
        self.assertEqual([c.sha for c in log], [self.c1, self.c2])
        self.assertEqual(log[0].trailers.get('Acting-Agent'), ['claude:test'])
        self.assertEqual(sorted(log[0].paths), ['a.txt', 'kit/k.py'])
        self.assertRegex(log[0].utc, r'^\d{8}T\d{6}Z$')
        self.assertTrue(self.g.is_ancestor(self.c1, self.c2))
        self.assertFalse(self.g.is_ancestor(self.c2, self.c1))
        self.assertEqual(self.g.first_commit_adding('*.json').sha, self.c2)
        self.assertEqual(self.g.first_commit_where('a.txt', lambda b: b'two' in b).sha, self.c2)
        self.assertEqual(self.g.file_at(self.c1, 'a.txt'), b'one\n')
        self.assertIsNone(self.g.file_at(self.c1, 'b.json'))
        with self.assertRaises(MissingInput):
            self.g.file_at('0' * 40, 'a.txt')
        self.assertEqual(self.g.diff_lines(self.c1, self.c2, 'a.txt'), (['two'], []))
        self.assertEqual(self.g.changed_paths(self.c1, self.c2), ['a.txt', 'b.json'])
        self.assertEqual(self.g.list_files(self.c2, 'kit'), ['kit/k.py'])
        self.assertEqual(self.g.files_at(self.c2, ['a.txt', 'nope.txt']), {'a.txt': b'one\ntwo\n', 'nope.txt': None})
        self.assertEqual(self.g.remote_branches_containing(self.c2), [])

    def test_dirty_paths(self):
        self.assertTrue(self.g.worktree_clean(['a.txt', 'kit']))
        write(self.repo / 'kit' / 'k.py', 'x = 2\n')
        write(self.repo / 'kit' / 'new.py', '')
        write(self.repo / 'kit' / '__pycache__' / 'k.cpython-313.pyc', 'junk')
        self.assertEqual(self.g.dirty_paths(['kit']), ['kit/k.py', 'kit/new.py'])
        self.assertEqual(self.g.dirty_paths(['']), ['kit/k.py', 'kit/new.py'])
        self.assertTrue(self.g.worktree_clean(['a.txt']))

    def test_checker_sha_at_matches_disk(self):
        kit = self.repo / 'kit'
        self.assertEqual(common.checker_sha256_at(self.g, self.c2, 'kit'), common.checker_sha256(kit, refresh=True))
        self.assertIsNone(common.checker_sha256_at(self.g, self.c2, 'nokit'))

    def test_read_only_guard(self):
        with self.assertRaises(CheckerError):
            common.run_git(self.repo, ['commit', '-m', 'x'])
        with self.assertRaises(CheckerError):
            common.run_git(self.repo, ['merge-base', 'a', 'b'])
        with self.assertRaises(CheckerError):
            common.run_git(self.repo, ['archive', 'HEAD'])

    def test_shallow_is_missing(self):
        clone = self.tmp / 'shallow'
        subprocess.run(['git', 'clone', '-q', '--depth', '1', 'file://' + str(self.repo), str(clone)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        g = common.RealGit(clone)
        self.assertTrue(g.is_shallow())
        with self.assertRaises(MissingInput):
            g.log()
        with self.assertRaises(MissingInput):
            g.first_commit_adding('*')


# ------------------------------------------------------------------------------------------------
class TestCanonicalJson(unittest.TestCase):
    def test_canonical_form(self):
        self.assertEqual(common.canonical_json({'b': 1, 'a': [1, 2, {'c': 'é'}]}),
                         b'{"a":[1,2,{"c":"\xc3\xa9"}],"b":1}')
        self.assertEqual(common.sha256_json({'a': 1}), common.sha256_bytes(b'{"a":1}'))

    def test_nan_rejected(self):
        for bad in (float('nan'), float('inf'), float('-inf')):
            with self.assertRaises(InvalidInput):
                common.canonical_json({'x': bad})
        with self.assertRaises(InvalidInput):
            common.canonical_json({'x': {1, 2}})
        for text in ('{"x": NaN}', '{"x": Infinity}', '[-Infinity]'):
            with self.assertRaises(InvalidInput):
                common.loads_strict(text)
        with self.assertRaises(InvalidInput):
            common.loads_strict('{bad')

    def test_utc_helpers(self):
        self.assertRegex(common.utc_now(), r'^\d{8}T\d{6}Z$')
        self.assertEqual(common.normalize_utc('2026-10-15T00:00Z'), '20261015T000000Z')
        self.assertEqual(common.normalize_utc('2026-10-15T01:02:03Z'), '20261015T010203Z')
        self.assertIsNone(common.normalize_utc('yesterday'))

    def test_glob(self):
        self.assertTrue(common.glob_match('results/dev/OUTPUTS.jsonl', 'results/**/OUTPUTS.jsonl'))
        self.assertTrue(common.glob_match('results/OUTPUTS.jsonl', 'results/**/OUTPUTS.jsonl'))
        self.assertFalse(common.glob_match('results/dev/x/OUTPUTS.jsonl', 'results/*/OUTPUTS.jsonl'))
        self.assertTrue(common.glob_match('.claude/skills/minifly-x/SKILL.md', '.claude/skills/minifly-*/**'))
        self.assertTrue(common.glob_match('d/results/a', 'd/results'))
        self.assertFalse(common.glob_match('d/results2/a', 'd/results'))

    def test_numbers_in_text(self):
        self.assertTrue(common.number_in_text(54432, 'up to 54,432 records'))
        self.assertTrue(common.number_in_text(2.5, 'at most 2.5 CPU-h'))
        self.assertFalse(common.number_in_text(3, 'three cycles'))
        self.assertFalse(common.number_in_text(1, 'v1 only'))


# ------------------------------------------------------------------------------------------------
class DeliveryCase(TmpCase):
    """A repo root with KB CONSTRAINTS, a v1 delivery and a trivial timeline."""
    NAME = 'alpha_dev_v1_20261101'

    def setUp(self):
        super().setUp()
        self.fx = self.tmp / 'fx'
        self.root = self.fx / 'repo'
        write(self.root / 'PROJECT_DOCUMENTS' / 'KB' / 'CONSTRAINTS.json', CONSTRAINTS)
        self.d = self.root / self.NAME
        write(self.d / 'ENDPOINT_LOCK.json', lock_obj(self.NAME))
        self.commits = [commit('s1' * 20, ['PROJECT_DOCUMENTS/KB/CONSTRAINTS.json', self.NAME + '/ENDPOINT_LOCK.json'])]
        make_timeline(self.fx, self.commits)
        self.rules = {'rules': [], 'bundles': {}}

    def retimeline(self, commits=None, history=None):
        make_timeline(self.fx, commits or self.commits, history=history)

    def ctx(self, stage=None, rules=None, delivery=None, **opts):
        g = common.TimelineGit(self.fx)
        return common.Ctx(repo_root=self.root, delivery=delivery or self.d, git=g, kb=common.KB(self.root),
                          rules=rules if rules is not None else self.rules, stage=stage, options=dict(opts))

    def set_lock(self, **over):
        write(self.d / 'ENDPOINT_LOCK.json', lock_obj(self.NAME, **over))


class TestRows(DeliveryCase):
    def test_jsonl_gz_and_zip_members(self):
        write(self.d / 'results/dev/OUTPUTS.jsonl', jsonl([{'i': 1}, {'i': 2}]) + '\n')
        gz = gzip.compress(jsonl([{'i': 3}, {'i': 4}]).encode())
        write(self.d / 'results/official/OUTPUTS.jsonl.gz', gz)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as zf:
            zf.writestr('rehearsal/OUTPUTS.jsonl', jsonl([{'i': 5}]))
        write(self.d / 'results/RESULT_BUNDLE_rehearsal.zip', buf.getvalue())
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as zf:
            zf.writestr('results/admission/OUTPUTS.jsonl.gz', gzip.compress(jsonl([{'i': 6}]).encode()))
            zf.writestr('notes.txt', 'ignored')
        write(self.d / 'RESULT_BUNDLE.zip', buf.getvalue())
        write(self.d / 'results/dev/SCORES.json', {'not': 'rows'})
        ctx = self.ctx()
        self.assertEqual(sorted(r['i'] for r in ctx.iter_rows('results/**/OUTPUTS.jsonl')), [1, 2, 3, 4, 5, 6])
        self.assertEqual([r['i'] for r in ctx.iter_rows('results/official/OUTPUTS.jsonl')], [3, 4])
        self.assertEqual([r['i'] for r in ctx.iter_rows('results/rehearsal/OUTPUTS.jsonl')], [5])
        self.assertEqual([r['i'] for r in ctx.iter_rows('results/admission/OUTPUTS.jsonl')], [6])
        self.assertEqual(len(list(ctx.iter_rows('results/**'))), 6)  # non-jsonl files are never row sources
        labels = sorted(lbl for lbl, _ln, _r in ctx.iter_rows_with_source('results/**/OUTPUTS.jsonl'))
        self.assertIn('RESULT_BUNDLE.zip!results/admission/OUTPUTS.jsonl.gz', labels)

    def test_duplicate_source_rejected(self):
        write(self.d / 'results/dev/OUTPUTS.jsonl', jsonl([{'i': 1}]))
        write(self.d / 'results/dev/OUTPUTS.jsonl.gz', gzip.compress(jsonl([{'i': 1}]).encode()))
        with self.assertRaises(InvalidInput):
            list(self.ctx().iter_rows('results/dev/OUTPUTS.jsonl'))

    def test_bad_rows_rejected(self):
        write(self.d / 'results/dev/OUTPUTS.jsonl', '{"x": NaN}\n')
        with self.assertRaises(InvalidInput):
            list(self.ctx().iter_rows('results/dev/OUTPUTS.jsonl'))
        write(self.d / 'results/dev/OUTPUTS.jsonl', '[1, 2]\n')
        with self.assertRaises(InvalidInput):
            list(self.ctx().iter_rows('results/dev/OUTPUTS.jsonl'))

    def test_lock_cache_and_missing(self):
        ctx = self.ctx()
        self.assertEqual(ctx.lock['delivery'], self.NAME)
        self.assertEqual(ctx.delivery_rel, self.NAME)
        with self.assertRaises(MissingInput):
            ctx.source_lock
        self.assertIsNone(ctx.source_lock_sha256)
        self.assertEqual(ctx.endpoint_lock_sha256, common.sha256_file(self.d / 'ENDPOINT_LOCK.json'))


# ------------------------------------------------------------------------------------------------
class TestExitCode(unittest.TestCase):
    def R(self, *statuses, rid='X-01'):
        return [Result(rid, s) for s in statuses]

    def test_mapping(self):
        ec = common.exit_code
        self.assertEqual(ec([]), 3)
        self.assertEqual(ec(self.R(PASS)), 0)
        self.assertEqual(ec(self.R(PASS, WARN, ADVISORY_FAIL, NOT_APPLICABLE)), 0)
        self.assertEqual(ec(self.R(PASS, MISSING)), 2)
        self.assertEqual(ec(self.R(MISSING, FAIL)), 1)
        self.assertEqual(ec(self.R(FAIL, CHECKER_ERROR)), 3)
        self.assertEqual(ec(self.R('BOGUS')), 3)
        self.assertEqual(ec(self.R(FAIL, rid='SELF-01')), 3)
        self.assertEqual(ec(self.R(MISSING, rid='SELF-01')), 3)
        self.assertEqual(ec(self.R(PASS, rid='SELF-01')), 0)


# ------------------------------------------------------------------------------------------------
class TestTierAndFamily(DeliveryCase):
    def source_lock(self, d, hashes, version='1.0.0'):
        write(d / 'SOURCE_LOCK.json', {'schema': common.SCHEMA_SOURCE_LOCK, 'checker_version': version,
                                       'closure_sha256': {'%s/f%d.py' % (d.name, i): h for i, h in enumerate(hashes)}})

    def attempts(self, d, cpu_s, stage='dev'):
        write(d / 'runlog/ATTEMPTS.jsonl', jsonl([{'attempt_id': 'A%d' % i, 'stage': stage, 'cpu_children_user_s': c,
                                                   'cpu_children_sys_s': 0.0} for i, c in enumerate(cpu_s)]))

    def tier(self, stage=None, **opts):
        return common.compute_tier(self.ctx(stage=stage, **opts))

    def test_declared_and_budget(self):
        self.assertEqual(self.tier()[0], 1)  # records > 0 -> learner execution -> T1
        self.set_lock(budget={'max_records': 0, 'max_cpu_h': 0.5}, tier_declared=0)
        self.assertEqual(self.tier()[0], 0)
        self.set_lock(budget={'max_records': 1000, 'max_cpu_h': 5})
        t, reasons = self.tier()
        self.assertEqual(t, 2)
        self.assertTrue(any('lock.budget' in r for r in reasons))
        self.set_lock(budget={'max_records': 20001, 'max_cpu_h': 1})
        self.assertEqual(self.tier()[0], 2)
        self.set_lock(tier_declared='T2')
        self.assertEqual(self.tier()[0], 2)
        self.set_lock(tier_declared='high')
        with self.assertRaises(InvalidInput):
            self.tier()

    def test_unknown_budget_never_lowers_the_tier(self):
        self.set_lock(budget={'max_records': '__FILL__', 'max_cpu_h': 1}, tier_declared=0)
        t, reasons = self.tier()
        self.assertEqual(t, 2)
        self.assertTrue(any('not numeric' in r for r in reasons), reasons)
        self.set_lock(budget=None, tier_declared=0)
        self.assertEqual(self.tier()[0], 2)

    def test_measured_attempts_and_receipts(self):
        self.set_lock(budget={'max_records': 0, 'max_cpu_h': 0}, tier_declared=0)
        self.attempts(self.d, [10.0])
        self.assertEqual(self.tier()[0], 1)
        self.attempts(self.d, [3600.0 * 3, 3600.0 * 1.5])
        self.assertEqual(self.tier()[0], 2)
        self.attempts(self.d, [1.0])
        write(self.d / 'results/receipts/wd/W.json', {'records': 20001})
        self.assertEqual(self.tier()[0], 2)
        write(self.d / 'results/receipts/wd/W.json', {'records': 'many'})
        with self.assertRaises(InvalidInput):
            self.tier()

    def test_family_cumulative_cpu(self):
        hashes = ['h%d' % i for i in range(4)]
        self.source_lock(self.d, hashes)
        beta = self.root / 'beta_dev_v1_20261102'
        write(beta / 'ENDPOINT_LOCK.json', lock_obj(beta.name))
        self.source_lock(beta, hashes[:3] + ['zz'])  # Jaccard 3/5 = 0.6
        self.attempts(beta, [3600.0 * 13])
        legacy = self.root / 'gamma_dev_20261103'  # not a v1 name: never a family member
        write(legacy / 'ENDPOINT_LOCK.json', lock_obj(legacy.name))
        self.source_lock(legacy, hashes)
        ctx = self.ctx()
        fam_id, members = common.family_of(ctx)
        self.assertEqual(members, ['alpha_dev_v1_20261101', 'beta_dev_v1_20261102'])
        self.assertEqual(fam_id, 'FAM-' + common.sha256_bytes(b'alpha_dev_v1_20261101')[:8])
        t, reasons = common.compute_tier(ctx)
        self.assertEqual(t, 2)
        self.assertTrue(any(r.startswith('family') and '-> 2' in r for r in reasons), reasons)
        nb = dict(common.closure_jaccard_neighbors(self.ctx()))
        self.assertAlmostEqual(nb['beta_dev_v1_20261102'], 0.6)

    def test_family_components_chain(self):
        a = ['a1', 'a2', 'a3', 'a4']
        self.source_lock(self.d, a)
        b = self.root / 'beta_dev_v1_20261102'
        c = self.root / 'cee_dev_v1_20261103'
        for d in (b, c):
            write(d / 'ENDPOINT_LOCK.json', lock_obj(d.name))
        self.source_lock(b, ['a1', 'a2', 'a3', 'b1'])          # J(a,b) = 3/5
        self.source_lock(c, ['a3', 'b1', 'c1'])                 # J(b,c) = 2/5? -> raise overlap
        self.source_lock(c, ['a2', 'a3', 'b1'])                 # J(b,c) = 3/4, J(a,c) = 2/5
        ctx = self.ctx()
        fam_id, members = common.family_of(ctx)
        self.assertEqual(len(members), 3)
        self.assertEqual(fam_id, 'FAM-' + common.sha256_bytes(b'alpha_dev_v1_20261101')[:8])
        d = self.root / 'delta_dev_v1_20261104'
        write(d / 'ENDPOINT_LOCK.json', lock_obj(d.name))
        self.source_lock(d, ['x', 'y'])
        self.assertEqual(common.family_of(self.ctx(delivery=d))[1], ['delta_dev_v1_20261104'])

    def test_broken_member_source_lock_fails_closed(self):
        self.source_lock(self.d, ['h1'])
        b = self.root / 'beta_dev_v1_20261102'
        write(b / 'ENDPOINT_LOCK.json', lock_obj(b.name))
        write(b / 'SOURCE_LOCK.json', '{broken')
        with self.assertRaises(InvalidInput):
            common.family_of(self.ctx())

    def test_stage_floor_and_t2_triggers(self):
        self.set_lock(budget={'max_records': 0, 'max_cpu_h': 0}, tier_declared=0)
        self.assertEqual(self.tier()[0], 0)
        self.assertEqual(self.tier(stage='official')[0], 2)
        self.assertEqual(self.tier(stage='pointer')[0], 2)
        self.assertEqual(self.tier(stage='lock', activation=True)[0], 2)
        self.set_lock(budget={'max_records': 0, 'max_cpu_h': 0}, tier_declared=0,
                      worlds={'official': {'mode': 'listed', 'ids': ['wo1'], 'n': 1}, 'official_role_intended': 'MAIN'})
        self.assertEqual(self.tier(stage='lock')[0], 2)
        self.set_lock(budget={'max_records': 0, 'max_cpu_h': 0}, tier_declared=0, intent='promotion')
        self.assertEqual(self.tier()[0], 2)
        self.set_lock(budget={'max_records': 0, 'max_cpu_h': 0}, tier_declared=0)
        write(self.d / 'AMENDMENTS.jsonl', jsonl([{'fields_changed': ['primary.margin_abs'], 'results_seen': ['x']}]))
        self.assertEqual(self.tier()[0], 2)

    def test_thresholds_required(self):
        write(self.root / 'PROJECT_DOCUMENTS' / 'KB' / 'CONSTRAINTS.json', {'schema': 'x'})
        with self.assertRaises(MissingInput):
            self.tier()


# ------------------------------------------------------------------------------------------------
@common.check('TST-PASS')
def _t_pass(ctx):
    return Result('TST-PASS', PASS, 'ok')


@common.check('TST-FAIL')
def _t_fail(ctx):
    return Result('TST-FAIL', FAIL, 'bad')


@common.check('TST-MISS')
def _t_miss(ctx):
    raise MissingInput('no input')


@common.check('TST-INV')
def _t_inv(ctx):
    raise InvalidInput('malformed')


@common.check('TST-EXC')
def _t_exc(ctx):
    raise ValueError('boom')


@common.check('TST-NA')
def _t_na(ctx):
    return Result('TST-NA', NOT_APPLICABLE, 'n/a')


@common.check('TST-EMPTY')
def _t_empty(ctx):
    return []


@common.check('TST-FOREIGN')
def _t_foreign(ctx):
    return Result('OTHER-01', PASS)


@common.check('TST-DEBT')
def _t_debt(ctx):
    return Result('TST-DEBT', PASS if ctx.options.get('debt_ok') else FAIL, 'debt row')


MOD = _t_pass.__module__.split('.')[-1]


def rule(rid, status='enforced', introduced='1.0.0', tier_min=0, aw='always', module=None, **kw):
    r = {'id': rid, 'module': module or MOD, 'stages': ['lock', 'publish'], 'tier_min': tier_min,
         'introduced_in': introduced, 'status': status, 'fm_ids': [], 'applies_when': aw}
    r.update(kw)
    return r


class TestRunBundle(DeliveryCase):
    def run_b(self, rules_list, checks, stage='lock', extra_bundles=None, **opts):
        bundles = {stage: {'checks': checks}}
        bundles.update(extra_bundles or {})
        rules = {'rules': rules_list, 'bundles': bundles}
        ctx = self.ctx(stage=stage, rules=rules, **opts)
        res = common.run_bundle(ctx, stage)
        return [(r.id, r.status) for r in res], res, ctx

    def test_status_mapping(self):
        ids = ['TST-PASS', 'TST-FAIL', 'TST-MISS', 'TST-INV', 'TST-EXC', 'TST-EMPTY', 'TST-FOREIGN']
        got, res, _ = self.run_b([rule(i) for i in ids], ids)
        self.assertEqual(got, [('TST-PASS', PASS), ('TST-FAIL', FAIL), ('TST-MISS', MISSING), ('TST-INV', FAIL),
                               ('TST-EXC', FAIL), ('TST-EMPTY', FAIL), ('TST-FOREIGN', FAIL)])
        self.assertIn('ValueError', res[4].detail)
        self.assertIn('runtime_s', res[0].data)

    def test_missing_function_module_and_mismatch(self):
        rules = [rule('TST-NOFN'), rule('TST-NOMOD', module='no_such_module_xyz'), rule('TST-PASS', module='common')]
        got, _, _ = self.run_b(rules, ['TST-NOFN', 'TST-NOMOD', 'TST-PASS'])
        self.assertEqual([s for _i, s in got], [CHECKER_ERROR] * 3)
        self.assertEqual(common.exit_code([Result(i, s) for i, s in got]), 3)

    def test_planned_and_unregistered_ids_raise(self):
        with self.assertRaises(CheckerError):
            self.run_b([rule('TST-PASS', status='planned')], ['TST-PASS'])
        with self.assertRaises(CheckerError):
            self.run_b([rule('TST-PASS')], ['TST-PASS', 'TST-UNKNOWN'])
        with self.assertRaises(CheckerError):
            self.run_b([rule('TST-PASS')], ['TST-PASS', 'TST-PASS'])
        with self.assertRaises(CheckerError):
            self.run_b([rule('TST-PASS')], ['TST-PASS'], extra_bundles={'a': {'extends': 'b', 'checks': []},
                                                                          'b': {'extends': 'a', 'checks': []}},
                       stage='a')

    def test_not_applicable_requires_declaration(self):
        got, _, _ = self.run_b([rule('TST-NA')], ['TST-NA'])
        self.assertEqual(got, [('TST-NA', FAIL)])
        got, _, _ = self.run_b([rule('TST-NA', aw='always (even here)')], ['TST-NA'])
        self.assertEqual(got, [('TST-NA', FAIL)])
        got, _, _ = self.run_b([rule('TST-NA', aw='only when a widget exists')], ['TST-NA'])
        self.assertEqual(got, [('TST-NA', NOT_APPLICABLE)])

    def test_advisory_status_is_warn(self):
        got, res, _ = self.run_b([rule('TST-FAIL', status='advisory'), rule('TST-MISS', status='advisory')],
                                 ['TST-FAIL', 'TST-MISS'])
        self.assertEqual(got, [('TST-FAIL', WARN), ('TST-MISS', WARN)])
        self.assertIn('was FAIL', res[0].detail)

    def _commit_source_lock(self, version):
        rel = self.NAME + '/SOURCE_LOCK.json'
        write(self.d / 'SOURCE_LOCK.json', {'schema': common.SCHEMA_SOURCE_LOCK, 'checker_version': version})
        self.retimeline(self.commits + [commit('s2' * 20, [rel])])

    def test_introduced_in_advisory_window(self):
        rules = [rule('TST-FAIL', introduced='1.1.0'), rule('TST-MISS', introduced='1.1.0'), rule('TST-INV')]
        ids = ['TST-FAIL', 'TST-MISS', 'TST-INV']
        self._commit_source_lock('1.0.0')
        got, res, ctx = self.run_b(rules, ids, stage='lock')
        self.assertEqual(got, [('TST-FAIL', ADVISORY_FAIL), ('TST-MISS', ADVISORY_FAIL), ('TST-INV', FAIL)])
        self.assertEqual(ctx.options['advisory_basis']['first_source_lock_checker_version'], '1.0.0')
        # T1 publish keeps the advisory window
        got, _, _ = self.run_b(rules, ids, stage='publish')
        self.assertEqual(got[0], ('TST-FAIL', ADVISORY_FAIL))
        # T2 official/publish/pointer always enforce
        self.set_lock(tier_declared=2)
        for st in ('publish', 'official', 'pointer'):
            got, _, _ = self.run_b(rules, ids, stage=st)
            self.assertEqual(got[0], ('TST-FAIL', FAIL), st)
        # T2 lock stage still has the window
        got, _, _ = self.run_b(rules, ids, stage='lock')
        self.assertEqual(got[0], ('TST-FAIL', ADVISORY_FAIL))

    def test_advisory_window_needs_a_genuine_first_version(self):
        rules = [rule('TST-FAIL', introduced='1.1.0')]
        got, _, _ = self.run_b(rules, ['TST-FAIL'])  # no committed SOURCE_LOCK
        self.assertEqual(got, [('TST-FAIL', FAIL)])
        self._commit_source_lock('0.0.1')  # forged pre-release version: enforce everything
        got, _, _ = self.run_b(rules, ['TST-FAIL'])
        self.assertEqual(got, [('TST-FAIL', FAIL)])
        self._commit_source_lock('9.9.9')  # newer than this checker: enforce everything
        got, _, _ = self.run_b(rules, ['TST-FAIL'])
        self.assertEqual(got, [('TST-FAIL', FAIL)])

    def test_kit_integrity_never_downgraded(self):
        self._commit_source_lock('1.0.0')
        got, _, _ = self.run_b([rule('SCHEMA-01', module='common', introduced='1.1.0', status='advisory')],
                               ['SCHEMA-01'], delivery=None)
        self.assertEqual(got, [('SCHEMA-01', PASS)])
        legacy = self.root / 'legacy_thing'
        write(legacy / 'ENDPOINT_LOCK.json', {'schema': 'old'})
        rules = {'rules': [rule('SCHEMA-01', module='common', status='advisory'), rule('TST-PASS')],
                 'bundles': {'lock': {'checks': ['SCHEMA-01', 'TST-PASS']}}}
        ctx = self.ctx(stage='lock', rules=rules, delivery=legacy)
        res = common.run_bundle(ctx, 'lock')
        self.assertEqual([(r.id, r.status) for r in res], [('SCHEMA-01', MISSING)])  # stops; no downgrade
        self.assertIn('legacy delivery: use retro', res[0].detail)

    def test_warn_below_tier_tier_min_and_conditions(self):
        rules = [rule('TST-FAIL', warn_below_tier=2), rule('TST-MISS', tier_min=2), rule('TST-INV', aw='tier 2'),
                 rule('TST-EXC', aw='sealed')]
        checks = ['TST-FAIL', 'TST-MISS', {'id': 'TST-INV', 'when': 't2'}, {'id': 'TST-EXC', 'when': 'sealed_confirm'}]
        got, _, _ = self.run_b(rules, checks)
        self.assertEqual(got, [('TST-FAIL', WARN), ('TST-MISS', NOT_APPLICABLE), ('TST-INV', NOT_APPLICABLE),
                               ('TST-EXC', NOT_APPLICABLE)])
        self.set_lock(tier_declared=2, worlds={'official_role_intended': 'SEALED_CONFIRM',
                                               'official': {'mode': 'sealed', 'ids': [], 'n': 8}})
        got, _, _ = self.run_b(rules, checks)
        self.assertEqual(got, [('TST-FAIL', FAIL), ('TST-MISS', MISSING), ('TST-INV', FAIL), ('TST-EXC', FAIL)])
        with self.assertRaises(CheckerError):
            self.run_b(rules, [{'id': 'TST-FAIL', 'when': 'whenever'}])

    def test_alternative_group(self):
        rules = [rule('TST-FAIL', aw='tier 2'), rule('TST-MISS', aw='tier 2'), rule('TST-DEBT', aw='no valid review'),
                 rule('TST-PASS', aw='tier 2')]
        grp = [{'group': 'review_or_debt', 'when': 't2', 'primary': ['TST-FAIL', 'TST-MISS'], 'fallback': ['TST-DEBT']}]
        got, _, _ = self.run_b(rules, grp)  # tier 1: whole group not applicable
        self.assertEqual([s for _i, s in got], [NOT_APPLICABLE] * 3)
        self.set_lock(tier_declared=2)
        got, _, _ = self.run_b(rules, grp)
        self.assertEqual(got, [('TST-FAIL', FAIL), ('TST-MISS', MISSING), ('TST-DEBT', FAIL)])
        got, res, _ = self.run_b(rules, grp, debt_ok=True)
        self.assertEqual(got, [('TST-FAIL', WARN), ('TST-MISS', WARN), ('TST-DEBT', PASS)])
        self.assertIn('covered by TST-DEBT', res[0].detail)
        grp_ok = [{'group': 'g', 'when': 't2', 'primary': ['TST-PASS'], 'fallback': ['TST-DEBT']}]
        got, _, _ = self.run_b(rules, grp_ok)
        self.assertEqual(got, [('TST-PASS', PASS), ('TST-DEBT', NOT_APPLICABLE)])

    def test_tier_failure_resolves_at_tier_2(self):
        write(self.root / 'PROJECT_DOCUMENTS' / 'KB' / 'CONSTRAINTS.json', {'schema': 'x'})
        rules = [rule('TST-PASS', tier_min=2), rule('TIER-01', module='common')]
        got, _, ctx = self.run_b(rules, ['TST-PASS', 'TIER-01'])
        self.assertEqual(got, [('TST-PASS', PASS), ('TIER-01', MISSING)])
        self.assertEqual(ctx.tier, 2)


class TestRealRules(unittest.TestCase):
    def setUp(self):
        self.rules = common.load_rules()
        self.idx = common.rule_index(self.rules)

    def test_every_spec_id_once(self):
        self.assertEqual(sorted(self.idx), sorted(SPEC_IDS))
        self.assertEqual(len(self.rules['rules']), len(SPEC_IDS))

    def test_fields_and_statuses(self):
        self.assertEqual(common.validate_rules(self.rules), [])
        adv = sorted(i for i, r in self.idx.items() if r['status'] == 'advisory')
        self.assertEqual(adv, ['CLM-16', 'LAD-03', 'LAD-04', 'PTR-06', 'WLD-02'])
        # SPEC 20.2: ids without corpus coverage or a faithful definition are 'planned' (README 'Not yet enforced')
        self.assertEqual(sorted(i for i, r in self.idx.items() if r['status'] == 'planned'), ['UDI-01'])
        # SPEC 20.10: checks added after the held-out pilot carry introduced_in 1.0.1; checks it tightened keep their
        # introduced_in and list 1.0.1 in changed_in
        new_ids = ['CLM-17', 'CYC-04']
        tightened = ['ADM-EXPO', 'CLM-04', 'DEBT-02', 'DEC-01', 'GOV-COI', 'GOV-REV', 'HELD-03', 'KB-03', 'RED-02',
                     'RED-03', 'REV-01', 'RUN-04', 'SEAL-02', 'WIT-01',
                     # round-1 repair: LOCK-02 (role candidate) and the checks whose behaviour the exposure strata changed
                     'LOCK-02', 'STR-01', 'DET-01', 'ADM-TB-CEIL', 'ADM-TB-FLOOR', 'UDI-01']
        self.assertEqual(common.CHECKER_VERSION, '1.0.1')
        self.assertEqual(self.rules['checker_version'], common.CHECKER_VERSION)
        for i, r in self.idx.items():
            self.assertEqual(r['introduced_in'], '1.0.1' if i in new_ids else '1.0.0', i)
            self.assertEqual(r.get('changed_in'), ['1.0.1'] if i in tightened else None, i)
            self.assertIn(r['status'], ('enforced', 'advisory', 'planned'), i)
            for f in ('id', 'module', 'stages', 'tier_min', 'introduced_in', 'status', 'fm_ids', 'applies_when'):
                self.assertIn(f, r, i)

    def test_bundles_follow_section_7_5(self):
        b = lambda s: common.bundle_ids(self.rules, s)  # noqa: E731
        lock = (['SELF-01', 'SCHEMA-01', 'TIER-01'] + ['LOCK-%02d' % i for i in (1, 2, 3, 4, 8, 9, 10, 11, 12, 13)]
                + ['AMD-%02d' % i for i in range(1, 6)] + ['AUT-%02d' % i for i in range(1, 5)])
        self.assertEqual(b('lock'), lock)
        self.assertEqual(b('technical'), lock + ['LOCK-05', 'LOCK-06', 'LOCK-07', 'CYC-01'])
        roster = set(b('roster'))
        for i in ['ADM-TB-CEIL', 'RED-11', 'DEC-01', 'CYC-02', 'RUN-01', 'WIT-01', 'REV-07', 'DEBT-01']:
            self.assertIn(i, roster)
        official = set(b('official'))
        for i in ['MET-01', 'LAD-04', 'INERT-01', 'CYC-03', 'AMD-04', 'RUN-02', 'GOV-COI', 'WLD-01', 'OC-01', 'SEAL-05', 'CONF-01']:
            self.assertIn(i, official)
        self.assertTrue(roster < official)
        publish = set(b('publish'))
        for i in ['SELF-01', 'CLM-16', 'ERR-01', 'KB-05', 'WLD-02', 'REV-08', 'DEBT-02', 'RUN-01', 'GOV-COI']:
            self.assertIn(i, publish)
        # SPEC 20.7/20.8: the runtime closure from roster on; resume legality and its replay at official and publish
        self.assertIn('RUN-03', roster)
        for i in ('RUN-02', 'RUN-03', 'RUN-04'):
            self.assertIn(i, official)
            self.assertIn(i, publish)
        self.assertNotIn('TIER-01', publish)
        # SPEC 20.13/20.16 (round 1): CLM-17 at publish and in the T0 lint; CYC-04 (append-only runlog) at publish
        self.assertIn('CLM-17', publish)
        self.assertIn('CLM-17', b('lint'))
        self.assertIn('CYC-04', publish)
        self.assertNotIn('CYC-04', official)
        self.assertNotIn('UDI-01', publish)  # planned: excluded from every bundle (SPEC 20.2)
        for st in self.rules['bundles']:
            self.assertNotIn('UDI-01', b(st), st)
        self.assertEqual(sorted(b('pointer')), sorted(['SELF-01', 'CLM-02', 'DEBT-02'] + ['PTR-%02d' % i for i in range(1, 8)]))

    def test_changed_in_validation_and_rule_newer_than(self):
        rules = json.loads(json.dumps(self.rules))
        r = next(x for x in rules['rules'] if x['id'] == 'ADM-EXPO')
        for bad in ('1.0.1', [], ['1.0.0'], ['x'], ['9.9.9'], ['1.0.1', '1.0.1']):
            r['changed_in'] = bad
            self.assertTrue(common.validate_rules(rules), bad)
        r['changed_in'] = ['1.0.1']
        self.assertEqual(common.validate_rules(rules), [])
        new = next(x for x in rules['rules'] if x['id'] == 'CLM-17')
        new['introduced_in'] = '9.0.0'                     # later than this checker: a registry error
        self.assertTrue(any('later than CHECKER_VERSION' in p for p in common.validate_rules(rules)))
        self.assertEqual(common.rule_newer_than(r, '1.0.0'), '1.0.1')
        self.assertIsNone(common.rule_newer_than(r, '1.0.1'))
        self.assertEqual(common.rule_versions(r), ['1.0.0', '1.0.1'])

    def test_t2_entries_are_conditional(self):
        entries = common.expand_bundle(self.rules, 'roster')
        self.assertFalse([e for e in entries if e.get('id') == 'WIT-01'])
        grp = [e for e in entries if 'group' in e][0]
        # SPEC 3.3 'REV, WIT: obtain the review, or record REVIEW_DEBT (dev and pilot only)'; SPEC 5 single-agent
        # T2 roster with a debt row: WIT-01 is in the review_or_debt group, at T2 only
        self.assertEqual(grp.get('when'), 't2')
        self.assertEqual(grp['primary'], ['REV-%02d' % i for i in range(1, 8)] + ['WIT-01'])
        self.assertEqual(grp['fallback'], ['DEBT-01'])
        self.assertEqual(self.idx['RUN-01'].get('warn_below_tier'), 2)
        self.assertEqual(self.idx['RUN-03'].get('warn_below_tier'), 2)  # T2 enforced, T1 WARN like RUN-01

    def test_wp1_checks_registered_by_declared_module(self):
        import launch  # noqa: F401
        for i in ['SELF-01', 'SCHEMA-01', 'TIER-01', 'RUN-01', 'RUN-02', 'RUN-03', 'RUN-04', 'CYC-01', 'CYC-02',
                  'CYC-03']:
            fn, err = common.resolve_check(self.idx[i])
            self.assertIsNotNone(fn, err)

    def test_protected_json(self):
        prot = json.loads((KIT / 'PROTECTED.json').read_text())
        self.assertEqual(set(prot['paths']), {'.claude/skills/minifly-*/**', 'PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md',
                                              'AGENTS.md', '.gitattributes', 'tools/minifly_check/**',
                                              'PROJECT_DOCUMENTS/KB/CONSTRAINTS.json',
                                              'PROJECT_DOCUMENTS/KB/USER_DECISIONS.jsonl'})
        self.assertEqual(prot['append_only'], ['PROJECT_DOCUMENTS/KB/*.jsonl'])
        self.assertTrue(any(common.glob_match('tools/minifly_check/INTEGRATION_LOG.md', p) for p in prot['paths']))

    def test_readme_not_yet_enforced_matches_planned(self):
        text = (KIT / 'README.md').read_text()
        m = re.search(r'^## Not yet enforced\s*$(.*?)(?=^## |\Z)', text, re.M | re.S)
        self.assertIsNotNone(m)
        listed = set(re.findall(r'^- `([A-Z]+(?:-[A-Z]+)*-\d+|[A-Z]+-[A-Z]+)`', m.group(1), re.M))
        planned = {i for i, r in self.idx.items() if r['status'] == 'planned'}
        self.assertEqual(listed, planned)

    def test_templates(self):
        t = KIT / 'templates'
        names = ['ENDPOINT_LOCK.template.json', 'AUTHORIZATION.template.json', 'REVIEW.template.json',
                 'ADMISSION.example.json', 'RECEIPT.example.json', 'OUTPUTS.example.jsonl', 'TAUGHT.example.jsonl',
                 'CLAIM.example.json', 'POINTER.example.json', 'admit_skeleton.py']
        for n in names:
            self.assertTrue((t / n).is_file(), n)
        for n in ('ENDPOINT_LOCK.template.json', 'AUTHORIZATION.template.json', 'REVIEW.template.json'):
            obj = common.load_json_file(t / n)
            self.assertIn('__FILL__', json.dumps(obj), n)
        lock = common.load_json_file(t / 'ENDPOINT_LOCK.template.json')
        self.assertEqual(lock['schema'], common.SCHEMA_ENDPOINT_LOCK)
        for n in ('OUTPUTS.example.jsonl', 'TAUGHT.example.jsonl'):
            self.assertTrue(common.load_jsonl_file(t / n))
        for row in common.load_jsonl_file(t / 'OUTPUTS.example.jsonl'):
            stream = bytes.fromhex(row['stream_hex'])
            pos = stream.find(b'\n')
            self.assertEqual(row['emitted_hex'], (stream if pos < 0 else stream[:pos]).hex())
            self.assertEqual(row['own_eos'], pos >= 0)
        claim = common.load_json_file(t / 'CLAIM.example.json')
        ptr = common.load_json_file(t / 'POINTER.example.json')
        for k in ('level_earned', 'primary_gate', 'gate_type', 'worlds_role', 'objective_metric'):
            self.assertEqual(ptr[k], claim[k])

    def test_check_decorator_duplicate(self):
        with self.assertRaises(CheckerError):
            @common.check('TST-PASS')
            def other(ctx):
                return Result('TST-PASS', PASS)
        common.check('TST-PASS')(_t_pass)  # re-registering the same definition is fine


# ------------------------------------------------------------------------------------------------
class TestKB(TmpCase):
    def setUp(self):
        super().setUp()
        self.kbdir = self.tmp / 'PROJECT_DOCUMENTS' / 'KB'
        write(self.kbdir / 'CONSTRAINTS.json', CONSTRAINTS)

    def kb(self):
        return common.KB(self.tmp)

    def test_union_merge_dedupe_and_conflicts(self):
        write(self.kbdir / 'FINDINGS.jsonl', '{"id": "F-1", "kind": "finding", "status": "proposed"}\n'
                                             '{"kind":"finding","id":"F-1","status":"proposed"}\n\n'
                                             '{"id": "F-2", "kind": "activation", "refers_to": "F-1"}\n')
        self.assertEqual(len(self.kb().rows('FINDINGS')), 2)
        write(self.kbdir / 'FINDINGS.jsonl', '{"id": "F-1", "status": "proposed"}\n{"id": "F-1", "status": "active"}\n')
        with self.assertRaises(InvalidInput):
            self.kb().rows('FINDINGS.jsonl')
        with self.assertRaises(MissingInput):
            self.kb().rows('INCIDENTS')

    def test_lifecycle_ledgers_allow_update_rows(self):
        write(self.kbdir / 'REVIEW_DEBT.jsonl', jsonl([{'id': 'D-1', 'repaid_by': None, 'utc': 'a'},
                                                       {'id': 'D-1', 'repaid_by': 'REV', 'utc': 'b'}]))
        self.assertEqual(len(self.kb().rows('REVIEW_DEBT')), 2)
        write(self.kbdir / 'REVIEW_DEBT.jsonl', jsonl([{'id': 'D-1', 'repaid_by': None, 'utc': 'a'},
                                                       {'id': 'D-1', 'repaid_by': None, 'utc': 'b'}]))
        with self.assertRaises(InvalidInput):
            self.kb().rows('REVIEW_DEBT')
        write(self.kbdir / 'INCIDENTS.jsonl', jsonl([{'id': 'I-1', 'kind': 'report'}, {'id': 'I-1', 'kind': 'closure'}]))
        self.assertEqual(len(self.kb().rows('INCIDENTS')), 2)
        write(self.kbdir / 'DELIVERY_LEDGER.jsonl', jsonl([{'delivery': 'x', 'n': 1}, {'delivery': 'x', 'n': 2}]))
        self.assertEqual(len(self.kb().rows('DELIVERY_LEDGER')), 2)

    def test_user_decision_and_finding_status(self):
        write(self.kbdir / 'USER_DECISIONS.jsonl', jsonl([{'id': 'UD-20261101-1', 'verbatim': 'yes'}]))
        self.assertEqual(self.kb().user_decision('UD-20261101-1')['verbatim'], 'yes')
        self.assertIsNone(self.kb().user_decision('UD-0'))
        rows = [{'id': 'F-N', 'kind': 'narrowing', 'refers_to': 'F-1'},
                {'id': 'F-1', 'kind': 'finding', 'status': 'proposed'},
                {'id': 'F-A', 'kind': 'activation', 'refers_to': 'F-1'},
                {'id': 'F-2', 'kind': 'finding', 'status': 'provisional'},
                {'id': 'F-3', 'kind': 'finding', 'status': 'proposed'},
                {'id': 'F-3A', 'kind': 'activation', 'refers_to': 'F-3'}]
        write(self.kbdir / 'FINDINGS.jsonl', jsonl(rows))
        kb = self.kb()
        self.assertEqual(kb.finding_status('F-1'), 'narrowed')   # order-independent: narrowing sticks
        self.assertEqual(kb.finding_status('F-2'), 'provisional')
        self.assertEqual(kb.finding_status('F-3'), 'active')
        self.assertIsNone(kb.finding_status('F-404'))
        self.assertEqual(kb.constraints()['tier_thresholds']['t1_max_cpu_h'], 4)
        self.assertEqual(kb.constraints_sha256(), common.sha256_file(self.kbdir / 'CONSTRAINTS.json'))


# ------------------------------------------------------------------------------------------------
class TestGateReportAndSchema(DeliveryCase):
    def test_v1_detection(self):
        self.assertTrue(common.is_v1_delivery(self.d))
        bad_name = self.root / 'alpha_v1_20261101'
        write(bad_name / 'ENDPOINT_LOCK.json', lock_obj(bad_name.name))
        self.assertFalse(common.is_v1_delivery(bad_name))
        old = self.root / 'old_dev_v1_20261101'
        write(old / 'ENDPOINT_LOCK.json', {'schema': 'minifly.endpoint_lock.v0'})
        self.assertFalse(common.is_v1_delivery(old))
        self.assertFalse(common.is_v1_delivery(self.root / 'nothing_dev_v1_20261101'))
        res = common.CHECKS['SCHEMA-01'](self.ctx(delivery=old))
        self.assertEqual(res.status, MISSING)
        self.assertIn('legacy delivery: use retro', res.detail)
        self.assertEqual(common.CHECKS['SCHEMA-01'](self.ctx()).status, PASS)

    def test_write_and_list_gate_reports(self):
        ctx = self.ctx(stage='lock')
        ctx.tier = 1
        res = [Result('TST-PASS', PASS, 'ok'), Result('TST-FAIL', WARN, 'w')]
        p1 = common.write_gate_report(ctx, 'roster', res)
        self.assertRegex(p1.name, r'^GATE_roster_\d{8}T\d{6}Z\.json$')
        rep = json.loads(p1.read_text())
        self.assertEqual(list(rep)[-1], 'verdict')
        self.assertEqual(rep['verdict'], 'PASS')
        self.assertEqual(rep['git_mode'], 'timeline')
        for k in ('checker_sha256', 'checker_version', 'git_head', 'endpoint_lock_sha256', 'source_lock_sha256',
                  'tier', 'tier_reasons', 'family_id', 'results'):
            self.assertIn(k, rep)
        p2 = common.write_gate_report(ctx, 'roster', res)
        self.assertNotEqual(p1, p2)
        ctx.dry = True
        p3 = common.write_gate_report(ctx, 'official', [Result('TST-FAIL', FAIL, 'x')])
        self.assertTrue(p3.name.startswith('DRY_official_'))
        # a forged report whose verdict disagrees with its rows, and one naming another delivery
        forged = dict(rep, verdict='PASS', exit_code=0, results=[{'id': 'X', 'status': 'FAIL'}])
        write(self.d / 'checks' / 'GATE_roster_20990101T000000Z.json', forged)
        write(self.d / 'checks' / 'GATE_roster_20990101T000001Z.json', dict(rep, delivery='other_dev_v1_20261101'))
        reps = common.list_gate_reports(self.ctx())
        self.assertEqual(sorted(r['_path'] for r in reps), sorted('checks/' + p.name for p in (p1, p2, p3)))
        legacy = self.root / 'legacy'
        legacy.mkdir()
        with self.assertRaises(CheckerError):
            common.write_gate_report(self.ctx(delivery=legacy), 'lock', res)
        n = len(list((self.d / 'checks').iterdir()))
        with self.assertRaises(CheckerError):  # an unverified checker never produces a gate report
            common.write_gate_report(ctx, 'lock', [Result('SELF-01', FAIL, 'dirty'), Result('TST-PASS', PASS)])
        self.assertEqual(len(list((self.d / 'checks').iterdir())), n)

    def test_self_01_timeline_mode(self):
        r = common.CHECKS['SELF-01'](self.ctx())
        self.assertEqual(r.status, PASS)
        self.assertEqual(r.data.get('mode'), 'timeline')
        tl = json.loads((self.fx / 'timeline.json').read_text())
        tl['dirty'] = ['tools/minifly_check/decide.py']
        write(self.fx / 'timeline.json', tl)
        r = common.CHECKS['SELF-01'](self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertEqual(common.exit_code([r]), 3)

    def test_selftest_cache_roundtrip(self):
        sha = 'f' * 64
        p = common.write_selftest_cache(sha, {'fixtures': 0})
        try:
            self.assertEqual(p.name, 'minifly_check_selftest_%s.ok' % sha)
            self.assertEqual(Path(p).parent, Path(tempfile.gettempdir()))
            self.assertTrue(common.selftest_cache_valid(sha)[0])
            self.assertFalse(common.selftest_cache_valid('e' * 64)[0])
            p.write_text(json.dumps({'checker_sha256': 'e' * 64}))
            self.assertFalse(common.selftest_cache_valid(sha)[0])
        finally:
            p.unlink()

    def test_effective_cap(self):
        ctx = self.ctx()
        self.assertEqual(common.effective_cap(ctx, 'max_cycles')[0], 3)
        write(self.d / 'AUTHORIZATION.json', {'verbatim': 'You may run up to 5 cycles.', 'caps': {'max_cycles': 5}})
        self.assertEqual(common.effective_cap(self.ctx(), 'max_cycles')[0], 5)
        write(self.d / 'AUTHORIZATION.json', {'verbatim': 'Go ahead.', 'caps': {'max_cycles': 5}})
        self.assertEqual(common.effective_cap(self.ctx(), 'max_cycles')[0], 3)
        write(self.d / 'AUTHORIZATION.json', {'verbatim': 'Go ahead.', 'caps': {'max_cycles': 2}})
        self.assertEqual(common.effective_cap(self.ctx(), 'max_cycles')[0], 2)

    def test_designs_tried_k(self):
        rows = [{'arm': 'W', 'native_closure_sha': 'n1', 'params_sha256': 'p1'},
                {'arm': 'W', 'native_closure_sha': 'n1', 'params_sha256': 'p2'},
                {'arm': 'CEILING_X', 'native_closure_sha': None, 'params_sha256': None}]
        write(self.d / 'results/dev/OUTPUTS.jsonl', jsonl(rows))
        write(self.d / 'results/receipts/wd/W.json', {'arm': 'W', 'native_closure_sha': 'n2', 'params_sha256': 'p2'})
        write(self.d / 'runlog/ATTEMPTS.jsonl', jsonl([{'stage': 'dev', 'produced_paths': [{'path': 'x', 'sha256': 'y'}],
                                                        'native_closure_sha': 'n1', 'params_sha256': 'p1'},
                                                       {'stage': 'testbed', 'produced_paths': [{'path': 'x', 'sha256': 'y'}],
                                                        'native_closure_sha': 'n9', 'params_sha256': 'p9'}]))
        self.assertEqual(common.designs_tried_k(self.ctx())[0], 3)


class TestRepoScan(TmpCase):
    def test_bounds_and_excludes(self):
        write(self.tmp / 'a' / 'x.json', '{}')
        write(self.tmp / 'a' / 'big.jsonl', 'x' * 2048)
        write(self.tmp / 'a' / 'skip.txt', 'no')
        write(self.tmp / '.git' / 'y.json', '{}')
        write(self.tmp / 'tools' / 'minifly_check' / 'corpus' / 'C1' / 'z.json', '{}')
        write(self.tmp / 'a' / '__pycache__' / 'w.py', '')
        scan = common.iter_repo_files(self.tmp, max_bytes=1024)
        self.assertEqual([scan.rel(p) for p in scan], ['a/x.json'])
        self.assertEqual(scan.skipped_large, ['a/big.jsonl'])
        self.assertIn('skipped', scan.scope())
        with self.assertRaises(MissingInput):
            common.iter_repo_files(self.tmp / 'nope')


# ------------------------------------------------------------------------------------------------
class TestDispatcher(DeliveryCase):
    def mfc(self, *argv):
        return subprocess.run([sys.executable, '-I', str(KIT / 'minifly_check.py')] + list(argv),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(self.tmp))

    def test_help_lists_every_subcommand_fast(self):
        t0 = time.monotonic()
        p = self.mfc('--help')
        dt = time.monotonic() - t0
        self.assertEqual(p.returncode, 0)
        out = p.stdout.decode()
        for sub in ('init', 'lock', 'gate', 'launch', 'cycle', 'score', 'oc', 'seal', 'canary', 'review', 'claim',
                    'lint', 'ledger', 'kb', 'selftest', 'governance', 'heldout', 'retro'):
            self.assertRegex(out, r'\n\s+%s\s' % re.escape(sub))
        self.assertLess(dt, 1.0)

    def test_unknown_subcommand_and_usage_errors_exit_3(self):
        self.assertEqual(self.mfc('frobnicate').returncode, 3)
        self.assertEqual(self.mfc('gate', '--stage', 'nonsense', 'x').returncode, 3)
        self.assertEqual(self.mfc('gate', 'x', '--bogus').returncode, 3)

    def test_missing_module_or_bad_handler_exit_3(self):
        import minifly_check
        saved = dict(minifly_check.HANDLERS)
        fake = types.ModuleType('mfc_fake_handlers')
        fake.cmd_none = lambda args: None
        fake.cmd_missing = lambda args: (_ for _ in ()).throw(MissingInput('nothing here'))
        fake.cmd_invalid = lambda args: (_ for _ in ()).throw(InvalidInput('malformed'))
        fake.cmd_boom = lambda args: 1 / 0
        sys.modules['mfc_fake_handlers'] = fake
        err = io.StringIO()
        try:
            sys.stderr, saved_err = err, sys.stderr
            minifly_check.HANDLERS['oc'] = ('no_such_module_xyz', 'cmd_oc')
            self.assertEqual(minifly_check.main(['oc', 'x']), 3)
            minifly_check.HANDLERS['oc'] = ('mfc_fake_handlers', 'cmd_absent')
            self.assertEqual(minifly_check.main(['oc', 'x']), 3)
            for fn, rc in (('cmd_none', 3), ('cmd_missing', 2), ('cmd_invalid', 1), ('cmd_boom', 3)):
                minifly_check.HANDLERS['oc'] = ('mfc_fake_handlers', fn)
                self.assertEqual(minifly_check.main(['oc', 'x']), rc, fn)
            self.assertIn('CHECKER_ERROR', err.getvalue())
        finally:
            sys.stderr = saved_err
            minifly_check.HANDLERS.clear()
            minifly_check.HANDLERS.update(saved)
            sys.modules.pop('mfc_fake_handlers', None)

    def test_globals_after_subcommand_and_launch_split(self):
        import minifly_check
        a, b = minifly_check._split_launch(['--repo-root', 'r', 'launch', '--stage', 'dev', 'd', '--', 'py', '--', 'x'])
        self.assertEqual(a, ['--repo-root', 'r', 'launch', '--stage', 'dev', 'd'])
        self.assertEqual(b, ['py', '--', 'x'])
        self.assertEqual(minifly_check._split_launch(['gate', '--stage', 'lock', 'launch']), (['gate', '--stage', 'lock', 'launch'], None))
        p = minifly_check.build_parser()
        ns, extra = p.parse_known_args(['gate', '--stage', 'lock', 'd', '--git', 'timeline:/x', '--json'])
        self.assertEqual((ns.git, ns.json, ns.repo_root, extra), ('timeline:/x', True, None, []))

    def test_timeline_mode_only_for_fixture_roots(self):
        root, g = common.git_view_from_spec('timeline:' + str(self.fx), None)
        self.assertEqual(root, (self.fx / 'repo').resolve())
        with self.assertRaises(CheckerError):
            common.git_view_from_spec('timeline:' + str(self.fx), str(self.tmp))
        with self.assertRaises(CheckerError):
            common.git_view_from_spec('svn:' + str(self.fx), None)
        p = self.mfc('--git', 'timeline:' + str(self.fx), '--repo-root', str(self.tmp), 'gate', '--stage', 'lock', self.NAME)
        self.assertEqual(p.returncode, 3)

    def test_gate_refuses_legacy_dir(self):
        legacy = self.root / 'legacy_thing'
        write(legacy / 'README.md', 'old')
        p = self.mfc('--git', 'timeline:' + str(self.fx), 'gate', '--stage', 'lock', 'legacy_thing')
        self.assertEqual(p.returncode, 2, p.stderr.decode())
        self.assertIn('legacy delivery: use retro', p.stderr.decode())
        self.assertFalse((legacy / 'checks').exists())
        # --json reports the refusing check (SCHEMA-01 MISSING) and still writes nothing
        p = self.mfc('--git', 'timeline:' + str(self.fx), '--json', 'gate', '--stage', 'lock', 'legacy_thing')
        self.assertEqual(p.returncode, 2)
        rep = json.loads(p.stdout.decode())
        self.assertEqual([(r['id'], r['status']) for r in rep['results']], [('SCHEMA-01', 'MISSING')])
        self.assertFalse((legacy / 'checks').exists())

    def test_admission_worlds(self):
        lock = lock_obj(self.NAME)
        lock['worlds'] = {'technical': ['wt1', 'wt2']}
        write(self.d / 'ENDPOINT_LOCK.json', lock)
        ctx = self.ctx()
        self.assertEqual(common.admission_worlds(ctx), (['wt1', 'wt2'], 'lock.worlds.technical (no usable ADMISSION.worlds)'))
        write(self.d / 'ADMISSION.json', {'worlds': ['wt1']})
        self.assertEqual(common.admission_worlds(self.ctx()), (['wt1'], 'ADMISSION.worlds'))


if __name__ == '__main__':
    unittest.main()
