"""Unit tests for the corpus builder and the committed corpus (stdlib unittest; no checker imports).

    python3 -I -B -m unittest discover -s tools/minifly_check/corpus/_build -p 'test_*.py'
"""
import ast
import gzip
import hashlib
import json
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import build_corpus  # noqa: E402
import refcalc as RC  # noqa: E402
import scores_doc as SD  # noqa: E402
import table  # noqa: E402

CORPUS = os.path.dirname(HERE)
KIT = os.path.dirname(CORPUS)
D = 'minifly_corpus_base_dev_v1_20261101'
SPEC_IDS = (['C00a', 'C00b', 'C00c', 'C00d', 'C00e', 'C00f', 'C01', 'C02a', 'C02b', 'C03', 'C04a', 'C04b',
             'C05a', 'C05b', 'C05c', 'C06a', 'C06b', 'C06c', 'C07', 'C08a', 'C08b', 'C09a', 'C09b', 'C10',
             'C11a', 'C11b', 'C12', 'C13a', 'C13b', 'C13c', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19',
             'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30']
            + ['G0%d' % i for i in range(1, 6)] + ['K01'])
SUPPLEMENTARY = ['S%02d' % i for i in range(1, 104)]  # S01-S25 (WP7), S26-S45 (round 1), S46-S50 (round 2), S51-S55 (round 3), S56-S64 (round 4), S65-S80 (self-improvement round 1), S81-S88 (round-1 repair), S89-S97 (round-1 repair part 3), S98-S103 (round-1 repair part 4)
LITERAL_NAME = re.compile(r'(_ok|_passed|_passes|overlap|_verified|_independent)$|^(no_|all_)')


def load(path):
    with open(path, 'rb') as fh:
        return json.loads(fh.read().decode('utf-8'))


def rows(path):
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rb') as fh:
        return [json.loads(x) for x in fh.read().decode('utf-8').splitlines() if x.strip()]


def rules_ids():
    return set(r['id'] for r in load(os.path.join(KIT, 'RULES.json'))['rules'])


class TestTableAndExpected(unittest.TestCase):
    def setUp(self):
        self.exp = load(os.path.join(KIT, 'EXPECTED.json'))

    def test_every_spec_row_has_a_fixture(self):
        self.assertEqual(sorted(SPEC_IDS + SUPPLEMENTARY), sorted(self.exp))
        for fid in SPEC_IDS + SUPPLEMENTARY:
            self.assertTrue(os.path.isfile(os.path.join(CORPUS, fid, 'FIXTURE.json')), fid)

    def test_every_failure_class_has_a_fixture(self):
        fm = set(f for v in self.exp.values() for f in v['fm_ids'])
        self.assertEqual(set('FM-%02d' % i for i in range(1, 31)) - fm, set())

    def test_check_ids_are_spec_ids(self):
        ids = rules_ids()
        for fid, v in self.exp.items():
            for key in ('must_fire', 'must_not_fire', 'must_warn'):
                unknown = set(v[key]) - ids
                self.assertEqual(unknown, set(), '%s %s' % (fid, key))

    def test_expected_shape(self):
        keys = {'commands', 'must_fire', 'must_not_fire', 'must_warn', 'expected_exit', 'expect',
                'source_incident', 'fm_ids'}
        for fid, v in self.exp.items():
            self.assertEqual(set(v), keys, fid)
            self.assertEqual(len(v['commands']), len(v['expected_exit']), fid)
            self.assertFalse(set(v['must_fire']) & set(v['must_not_fire']), fid)

    def test_clean_fixtures_expect_exit_zero_and_fire_nothing(self):
        for e in table.T:
            v = self.exp[e['id']]
            meta = load(os.path.join(CORPUS, e['id'], 'FIXTURE.json'))
            self.assertEqual(meta['kind'], e['kind'])
            if e['kind'] == 'clean':
                self.assertEqual(set(v['expected_exit']), {0}, e['id'])
                self.assertEqual(v['must_fire'], [], e['id'])
            else:
                self.assertTrue(v['must_fire'] or v['must_warn'] or v['expect'], e['id'])
                # a dirty fixture blocks somewhere (exit 1 FAIL, 2 MISSING, 3 CHECKER_ERROR), asserts a dirty
                # verdict through expect, or carries an advisory defect that must WARN (never blocks)
                self.assertTrue(any(x != 0 for x in v['expected_exit']) or v['expect'] or
                                (v['must_warn'] and not v['must_fire']), e['id'])

    def test_source_incidents_exist_in_kb(self):
        inc = set()
        with open(os.path.join(os.path.dirname(os.path.dirname(KIT)), 'PROJECT_DOCUMENTS', 'KB',
                               'INCIDENTS.jsonl'), 'rb') as fh:
            for line in fh.read().decode('utf-8').splitlines():
                if line.strip():
                    inc.add(json.loads(line)['id'])
        for fid, v in self.exp.items():
            if v['source_incident'] is not None:
                self.assertIn(v['source_incident'], inc, fid)


    def test_kb_incident_closures_name_a_firing_fixture(self):
        """KB-05 consistency with the real KB: every closure row's check fires in its fixture, except a row retired by
        a later closure of the same incident that names it in `supersedes`, carries a note and fires itself (SPEC
        20.20; the ledger is append-only)."""
        path = os.path.join(os.path.dirname(os.path.dirname(KIT)), 'PROJECT_DOCUMENTS', 'KB', 'INCIDENTS.jsonl')
        with open(path, 'rb') as fh:
            closures = [json.loads(x) for x in fh.read().decode('utf-8').splitlines()
                        if x.strip() and json.loads(x)['kind'] == 'closure']
        self.assertTrue(closures)

        def fires(r):
            return r.get('fixture_id') in self.exp and r.get('check_id') in self.exp[r['fixture_id']]['must_fire']

        def target(r):
            return r.get('refers_to') or r.get('id')
        by_id = dict((r['id'], r) for r in closures)
        retired = set()
        for r in closures:
            sup = r.get('supersedes')
            for t in ([sup] if isinstance(sup, str) else list(sup or [])):
                self.assertIn(t, by_id, r['id'])
                self.assertEqual(target(by_id[t]), target(r), r['id'])
                self.assertTrue(r.get('note'), r['id'])
                self.assertTrue(fires(r), r['id'])
                retired.add(t)
        for row in closures:
            if row['id'] in retired:
                continue
            self.assertIn(row['check_id'], self.exp[row['fixture_id']]['must_fire'], row['id'])


class TestBaseHandValues(unittest.TestCase):
    """The hand-computed numbers of EXPECTED_VALUES.md, re-derived from the committed C00a rows."""

    @classmethod
    def setUpClass(cls):
        cls.repo = os.path.join(CORPUS, 'C00a', 'repo', D)
        cls.off = rows(os.path.join(cls.repo, 'results', 'official', 'OUTPUTS.jsonl.gz'))

    def test_verdict_and_headline(self):
        v = SD.verdict(self.off)
        self.assertEqual(v['verdict'], 'PASS')
        self.assertAlmostEqual(v['mean_cand'], 109 / 128.0)
        self.assertAlmostEqual(v['mean_d'], 0.2890625)
        self.assertAlmostEqual(v['lcb'], 0.2673955, places=6)
        self.assertAlmostEqual(v['drop_ucb'], 0.0700092, places=6)
        with open(os.path.join(self.repo, 'REPORT.md'), 'rb') as fh:
            first = fh.read().decode('utf-8').splitlines()[0]
        self.assertEqual(first, table.BASE_HEADLINE)
        self.assertEqual(SD.headline(self.off), table.BASE_HEADLINE)

    def test_best_shortcut_unique_every_world(self):
        for x in SD.per_world(self.off):
            self.assertEqual(x['best_names'], ['DIGIT_NO_CARRY'], x['world'])
            self.assertEqual(x['best'], 0.5)

    def test_strata_and_determinacy(self):
        for r in self.off:
            if r['pool'] == 'HELD':
                a, b = RC.parse(RC.dec(r['prompt_hex']))
                self.assertIn(SD.stratum(a, b), ('S2', 'S3'))
                self.assertTrue(SD.determined(a, b))

    def test_rows_are_self_consistent(self):
        """MET-01/MET-02 analogue: emitted/own_eos derive from stream_hex; expected is the true sum."""
        for r in self.off:
            s = bytes.fromhex(r['stream_hex'])
            self.assertEqual(len(s), 4)
            i = s.find(b'\n')
            self.assertEqual(r['emitted_hex'], (s if i < 0 else s[:i]).hex())
            self.assertEqual(r['own_eos'], i >= 0)
            a, b = RC.parse(RC.dec(r['prompt_hex']))
            self.assertEqual(r['expected_hex'], str(a + b).encode().hex())

    def test_results_covered_by_attempts(self):
        """RUN-01 analogue on the clean base."""
        att = rows(os.path.join(self.repo, 'runlog', 'ATTEMPTS.jsonl'))
        produced = {(p['path'], p['sha256']) for a in att for p in a['produced_paths']}
        exempt = {'results/admission/TESTBED.json', 'results/dev/SCORES.json'}
        for root, _d, files in os.walk(os.path.join(self.repo, 'results')):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, self.repo).replace(os.sep, '/')
                if rel in exempt:
                    continue
                with open(full, 'rb') as fh:
                    sha = hashlib.sha256(fh.read()).hexdigest()
                self.assertIn((rel, sha), produced, rel)

    def test_lock_sha_propagates(self):
        with open(os.path.join(self.repo, 'ENDPOINT_LOCK.json'), 'rb') as fh:
            lock_sha = hashlib.sha256(fh.read()).hexdigest()
        sl = load(os.path.join(self.repo, 'SOURCE_LOCK.json'))
        self.assertEqual(sl['endpoint_lock_sha256'], lock_sha)
        self.assertEqual(load(os.path.join(self.repo, 'review', 'REVIEW.json'))
                         ['reviewed_endpoint_lock_sha256'], lock_sha)
        rec = load(os.path.join(self.repo, 'results', 'receipts', 'w_off_1', 'NPT.json'))
        self.assertEqual(rec['endpoint_lock_sha256'], lock_sha)
        self.assertEqual(rec['native_closure_sha'], sl['native_closure_sha'])
        self.assertEqual(rec['writes_by_channel']['pair_memory'], 0)
        self.assertEqual(rec['channel_digests']['post_first']['pair_memory'],
                         rec['channel_digests']['birth']['pair_memory'])


class TestCleanFixtureHygiene(unittest.TestCase):
    def test_no_literal_integrity_assignments_or_local_paths_in_clean_fixtures(self):
        for e in table.T:
            if e['kind'] != 'clean':
                continue
            for root, _d, files in os.walk(os.path.join(CORPUS, e['id'], 'repo')):
                if os.sep + os.path.join('tools', 'minifly_check', 'corpus') + os.sep in root + os.sep:
                    continue  # nested fixtures of a governance fixture carry their own (dirty) kind
                for f in files:
                    full = os.path.join(root, f)
                    if f.endswith('.gz'):
                        continue
                    with open(full, 'rb') as fh:
                        text = fh.read().decode('utf-8')
                    for bad in ('/Users/', '/home/', '/tmp/', 'C:\\'):
                        self.assertNotIn(bad, text, '%s %s' % (e['id'], full))
                    if f.endswith('.py'):
                        for node in ast.walk(ast.parse(text)):
                            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                                    and node.value.value in (True, False, 0, 0.0):
                                for t in node.targets:
                                    name = getattr(t, 'id', getattr(t, 'attr', ''))
                                    self.assertIsNone(LITERAL_NAME.search(name), '%s %s' % (full, name))

    def test_native_files_free_of_forbidden_patterns(self):
        repo = os.path.join(CORPUS, 'C00a', 'repo', D)
        for f in ('learner.py', 'run.py'):
            with open(os.path.join(repo, f), 'rb') as fh:
                text = fh.read().decode('utf-8')
            for bad in ('.backward(', 'transformer', 'attention', 'elm', 'requires_grad=True', 'int(',
                        'carry', 'digit'):
                self.assertNotIn(bad, text, '%s %s' % (f, bad))


class TestDeterminism(unittest.TestCase):
    def test_rebuild_is_byte_identical_and_valid(self):
        self.assertEqual(build_corpus.main(['--check']), 0)

    def test_fixture_sizes_bounded(self):
        for e in table.T:
            self.assertLess(build_corpus.dir_size(os.path.join(CORPUS, e['id'])), 512 * 1024, e['id'])


if __name__ == '__main__':
    unittest.main()
