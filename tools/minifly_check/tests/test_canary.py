"""Unit tests for canary.py (WP4): MFC canary and CAN-01..03 on the clean synthetic delivery. Stdlib only."""
import contextlib
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

KIT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for p in (str(KIT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import canary  # noqa: E402
import common  # noqa: E402
import minifly_check  # noqa: E402
from test_admitcheck import NAME, WP4Fixture, fixture_rows_default, jsonl  # noqa: E402


def tree_digest(root):
    out = {}
    for p in sorted(Path(root).rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


class CanaryCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='mfc_wp4c_')
        self.f = WP4Fixture(self.tmpdir).build()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def mfc(self, *argv, git=True):
        pre = ['--git', 'timeline:%s' % self.f.fx, '--repo-root', str(self.f.root)] if git else []
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = minifly_check.main(pre + list(argv))
        return rc, out.getvalue(), err.getvalue()

    def claim(self):
        rc, out, err = self.mfc('claim', NAME)
        self.assertEqual(rc, 0, out + err)
        self.f.timeline()

    def reports(self):
        return sorted((self.f.d / 'checks').glob('CANARY_*.json')) if (self.f.d / 'checks').is_dir() else []

    def run_canary(self, *extra):
        rc, out, err = self.mfc('--json', 'canary', NAME, *extra)
        rep = json.loads(out) if out.strip().startswith('{') else None
        return rc, rep, out, err


class CleanCanary(CanaryCase):
    def test_all_applicable_mutants_caught_and_only_the_report_is_written(self):
        self.claim()
        before = tree_digest(self.f.root)
        rc, rep, out, err = self.run_canary('--seed', 'c0ffee')
        self.assertEqual(rc, 0, err + out[:2000])
        st = {r['id']: r['status'] for r in rep['results']}
        self.assertEqual(st, {'CAN-01': 'PASS', 'CAN-02': 'PASS', 'CAN-03': 'PASS'})
        self.assertEqual(rep['n_mutants'], 20)
        self.assertEqual(rep['detected'], 20)
        self.assertTrue(all(m['caught'] for m in rep['mutants']))
        ops = {o['id']: o for o in rep['operators']}
        self.assertEqual(sorted(ops), ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M09', 'M10', 'M11', 'M12',
                                       'M13', 'M14', 'M15', 'M17'])
        self.assertTrue(all(o['applicable'] for o in ops.values()), ops)
        self.assertEqual(set(m['op'] for m in rep['mutants']), set(ops))
        self.assertTrue(all(r['equal'] for r in rep['reproduction']))
        after = tree_digest(self.f.root)
        new = sorted(set(after) - set(before))
        self.assertEqual(len(new), 1, new)
        self.assertRegex(new[0], r'^%s/checks/CANARY_\d{8}T\d{6}Z(_\d+)?\.json$' % NAME)
        self.assertEqual({k: v for k, v in after.items() if k in before}, before)  # nothing else changed
        saved = json.loads((self.f.root / new[0]).read_text())
        self.assertEqual(saved['schema'], canary.SCHEMA_CANARY_REPORT)
        self.assertEqual(saved['seed'], 'c0ffee')
        self.assertEqual(saved['verdict'], 'PASS')
        self.assertEqual(saved['endpoint_lock_sha256'], self.f.lock_sha())
        self.assertNotIn(self.tmpdir, json.dumps(saved))  # no absolute temp paths in the report

    def test_operators_that_find_no_target_are_not_applicable(self):
        rc, rep, out, err = self.run_canary('--seed', 'c0ffee', '--n', '5')
        self.assertEqual(rc, 0, err + out[:2000])
        ops = {o['id']: o for o in rep['operators']}
        for oid in ('M09', 'M10', 'M14'):  # no REPORT.md, CLAIM.json or pointer
            self.assertFalse(ops[oid]['applicable'])
            self.assertTrue(ops[oid]['reason'])
        self.assertEqual(rep['n_mutants'], 12)  # every applicable operator at least once even with --n 5
        self.assertEqual(rep['detected'], 12)

    def test_seed_determines_the_mutants(self):
        rc1, rep1, _o, _e = self.run_canary('--seed', 'c0ffee', '--n', '24')
        rc2, rep2, _o, _e = self.run_canary('--seed', 'c0ffee', '--n', '24')
        rc3, rep3, _o, _e = self.run_canary('--seed', '0badf00d', '--n', '24')
        self.assertEqual((rc1, rc2, rc3), (0, 0, 0))

        def sig(rep):
            return [(m['op'], json.dumps(m['target']), m.get('description')) for m in rep['mutants']]
        self.assertEqual(sig(rep1), sig(rep2))
        self.assertNotEqual(sig(rep1), sig(rep3))
        self.assertEqual(len(self.reports()), 3)


class CanaryFailures(CanaryCase):
    def test_baseline_failure_makes_a_survivor(self):
        rows = fixture_rows_default()
        rows[0]['target_margin_delta_vs_passive'] = -0.5
        rows[1]['target_margin_delta_vs_passive'] = -0.5
        self.f.w('results/admission/FIXTURES.jsonl', jsonl(rows))
        rc, rep, out, err = self.run_canary('--seed', 'c0ffee')
        self.assertEqual(rc, 1, err)
        st = {r['id']: r['status'] for r in rep['results']}
        self.assertEqual(st['CAN-02'], 'FAIL')
        self.assertEqual(st['CAN-01'], 'PASS')
        surv = [m for m in rep['mutants'] if not m['caught']]
        self.assertTrue(surv and all(m['op'] == 'M02' for m in surv), surv)
        self.assertLess(rep['detected'], rep['n_mutants'])

    def test_unfaithful_copy_fails_can01(self):
        real = canary.make_pristine

        def lossy(ctx, root):
            extra = real(ctx, root)
            (root / NAME / 'results/admission/FIXTURES.jsonl').unlink()
            return extra
        with mock.patch.object(canary, 'make_pristine', lossy):
            rc, rep, out, err = self.run_canary('--seed', 'c0ffee')
        st = {r['id']: r['status'] for r in rep['results']}
        self.assertEqual(st['CAN-01'], 'FAIL', rep['results'])
        self.assertIn('does not reproduce', [r for r in rep['results'] if r['id'] == 'CAN-01'][0]['detail'])
        self.assertEqual(rc, 1)

    def test_can03_threshold(self):
        ctx = self.f.ctx(stage='canary')
        ops = [{'id': o.oid, 'applicable': i < 7, 'reason': 'x'} for i, o in enumerate(canary.OPERATORS)]
        ctx._cache['wp4_canary'] = {'seed': 'c0ffee', 'operators': ops, 'n_operators': 15, 'n_applicable': 7,
                                    'applicable_fraction': 7 / 15.0, 'errors': [], 'mutants': [], 'n_mutants': 0,
                                    'detected': 0, 'reproduction': []}
        idx = common.rule_index(common.load_rules())
        self.assertEqual(common.run_check(ctx, idx['CAN-03'])[0].status, common.FAIL)
        self.assertEqual(common.run_check(ctx, idx['CAN-02'])[0].status, common.NOT_APPLICABLE)
        self.assertEqual(common.run_check(ctx, idx['CAN-01'])[0].status, common.MISSING)
        ctx._cache['wp4_canary'].update({'n_applicable': 8, 'applicable_fraction': 8 / 15.0})
        self.assertEqual(common.run_check(ctx, idx['CAN-03'])[0].status, common.PASS)

    def test_can_checks_outside_the_canary_command_are_missing(self):
        ctx = self.f.ctx(stage='roster')
        idx = common.rule_index(common.load_rules())
        for cid in ('CAN-01', 'CAN-02', 'CAN-03'):
            self.assertEqual(common.run_check(ctx, idx[cid])[0].status, common.MISSING)

    def test_usage_errors(self):
        rc, _o, err = self.mfc('canary', NAME, '--seed', 'not-hex')
        self.assertEqual(rc, 3)
        rc, _o, err = self.mfc('canary', NAME, '--seed', 'c0ffee', '--n', '0')
        self.assertEqual(rc, 3)
        rc, _o, err = self.mfc('canary', NAME, '--seed', 'c0ffee', '--out', str(self.f.root / 'keep'))
        self.assertEqual(rc, 3)
        self.assertIn('outside the repository', err)
        self.assertEqual(self.reports(), [])
        legacy = self.f.root / 'legacy_thing'
        legacy.mkdir()
        rc, _o, err = self.mfc('canary', 'legacy_thing', '--seed', 'c0ffee')
        self.assertEqual(rc, 2)
        self.assertIn('legacy', err)

    def test_out_keeps_copies_outside_the_repo(self):
        keep = Path(self.tmpdir) / 'keep'
        rc, rep, out, err = self.run_canary('--seed', 'c0ffee', '--n', '1', '--out', str(keep))
        self.assertEqual(rc, 0, err)
        runs = list(keep.iterdir())
        self.assertEqual(len(runs), 1)
        names = sorted(p.name for p in runs[0].iterdir())
        self.assertIn('pristine', names)
        self.assertTrue(any(n.startswith('m000_') for n in names))
        self.assertTrue((runs[0] / 'pristine' / NAME / 'ENDPOINT_LOCK.json').is_file())


class Mechanics(CanaryCase):
    def test_history_only_git(self):
        g = canary.HistoryOnlyGit(common.TimelineGit(self.f.fx))
        self.assertEqual(g.head(), common.TimelineGit(self.f.fx).head())
        self.assertTrue(g.log())
        with self.assertRaises(common.MissingInput):
            g.dirty_paths(['x'])
        with self.assertRaises(common.MissingInput):
            g.worktree_clean(['x'])

    def test_rewrite_rows_keeps_gz_and_zip_formats(self):
        rows = list(self.f.ctx().iter_rows('results/dev/OUTPUTS.jsonl'))
        (self.f.d / 'results/dev/OUTPUTS.jsonl').unlink()
        (self.f.d / 'results/dev/OUTPUTS.jsonl.gz').write_bytes(gzip.compress(jsonl(rows).encode()))
        ctx = self.f.ctx()
        self.assertTrue(canary.rewrite_rows(ctx, 'results/dev/OUTPUTS.jsonl', lambda rs: rs[:3]))
        self.assertEqual(len(list(self.f.ctx().iter_rows('results/dev/OUTPUTS.jsonl'))), 3)
        self.assertTrue((self.f.d / 'results/dev/OUTPUTS.jsonl.gz').is_file())
        (self.f.d / 'results/dev/OUTPUTS.jsonl.gz').unlink()
        with zipfile.ZipFile(str(self.f.d / 'RESULT_BUNDLE_dev.zip'), 'w') as z:
            z.writestr('results/dev/OUTPUTS.jsonl', jsonl(rows))
            z.writestr('results/dev/notes.md', 'kept')
        ctx = self.f.ctx()
        self.assertTrue(canary.rewrite_rows(ctx, 'results/dev/OUTPUTS.jsonl', lambda rs: rs[:5]))
        self.assertEqual(len(list(self.f.ctx().iter_rows('results/dev/OUTPUTS.jsonl'))), 5)
        with zipfile.ZipFile(str(self.f.d / 'RESULT_BUNDLE_dev.zip')) as z:
            self.assertEqual(z.read('results/dev/notes.md'), b'kept')

    def test_replace_bytes_breaks_hard_links(self):
        a = Path(self.tmpdir) / 'a.txt'
        b = Path(self.tmpdir) / 'b.txt'
        a.write_bytes(b'original')
        os.link(str(a), str(b))
        canary.replace_bytes(b, b'mutated')
        self.assertEqual(a.read_bytes(), b'original')
        self.assertEqual(b.read_bytes(), b'mutated')

    def test_registered_ids_match_rules(self):
        rules = common.load_rules()
        mine = sorted(r['id'] for r in rules['rules'] if r['module'] == 'canary')
        self.assertEqual(mine, ['CAN-01', 'CAN-02', 'CAN-03'])
        self.assertEqual(len(canary.OPERATORS), 15)
        self.assertEqual([o.oid for o in canary.OPERATORS], ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M09',
                                                              'M10', 'M11', 'M12', 'M13', 'M14', 'M15', 'M17'])
        for o in canary.OPERATORS:
            for rid in o.targets:
                self.assertIn(rid, common.rule_index(rules))


if __name__ == '__main__':
    unittest.main()
