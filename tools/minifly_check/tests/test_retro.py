"""Unit tests for retro.py (WP6): read-only adapters on synthetic legacy deliveries, --out refusal,
acceptance bookkeeping. Nothing historical is imported or executed; Python files are only parsed."""
import argparse
import ast
import contextlib
import gzip
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

import common  # noqa: E402
import retro  # noqa: E402

SUMMARIZE = '''\
import json
def main(rows):
    verdict = {}
    for w, arms in rows.items():
        o = arms.get("ORACLE")
        verdict[w] = {"gate_oracle": None if o is None else o >= 0.9}
        verdict[w]["gates_ok"] = verdict[w]["gate_oracle"] is not False
    n_ok = sum(v["gates_ok"] for v in verdict.values())
    overall = "supported" if n_ok == len(verdict) else "not supported"
    open("results/SUMMARY.json", "w").write(json.dumps({"overall": overall}))
'''
AUDIT = '''\
def run():
    complete = True
    audit = dict(status="PASS", exposure_overlap=0, all_rows_decoded_from_native_margins=True,
                 no_training_or_revalidation_performed=True, formal_confirmation=False)
    gate = {"held_ok": True}
    all_ok = True
    for x in range(3):
        all_ok = all_ok and x > 0
    verdict = "FAVORABLE" if complete else "INCOMPLETE"
    save("results/SUMMARY.json", {"verdict": verdict})
'''
CLEAN_PY = '''\
import json
def summarize(rows):
    ok = [r for r in rows if r["exact"] >= 0.5]
    passed = len(ok) >= 8 and len(ok) == len(rows)
    return {"passed": passed}
'''


def write(root, rel, text):
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        p.write_bytes(text)
    else:
        p.write_text(text, encoding='utf-8')
    return p


class RetroCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='test_retro_'))
        self.repo = self.tmp / 'repo'
        self.repo.mkdir()
        retro._CHECKER_SHA[0] = 'test'

    def tearDown(self):
        shutil.rmtree(str(self.tmp), ignore_errors=True)

    def delivery(self, name):
        src = retro.WorktreeSource(self.repo, with_history=False)
        return retro.Delivery(src, name)

    def legacy(self):
        d = 'legacy_dev_v1_20261001'
        write(self.repo, d + '/summarize.py', SUMMARIZE)
        write(self.repo, d + '/audit_results.py', AUDIT)
        write(self.repo, d + '/run.py', 'import helper\nimport json\n')
        write(self.repo, d + '/helper.py', 'X = 1\n')
        write(self.repo, d + '/SOURCE_LOCK.json', json.dumps({'run.py': 'a' * 64, 'summarize.py': 'b' * 64}))
        write(self.repo, d + '/AUDIT_SOURCE_LOCK.json', json.dumps({'audit_results.py': 'c' * 64}))
        write(self.repo, d + '/REPORT.md', '# Report\n\n* G1 memory - largely met (84% vs 95%).\n'
                                           '## Native confirmation on D2\nNot a confirmation run.\n')
        write(self.repo, d + '/cycles/CYCLE1.md', 'cycle notes /Users/someone/x\n')
        return d


class TestAdapters(RetroCase):
    def test_literal_and_failopen_flagged(self):
        d = self.delivery(self.legacy())
        lit = retro.adapter_literal(d)[0]
        self.assertEqual(lit['status'], 'FLAG')
        for name in ('exposure_overlap', 'all_rows_decoded_from_native_margins', 'no_training_or_revalidation_performed',
                     'held_ok'):
            self.assertIn(name, lit['reason'])
        self.assertNotIn('formal_confirmation', lit['reason'])
        self.assertNotIn('all_ok=', lit['reason'])  # initializer later computed: not a literal integrity flag
        fo = retro.adapter_failopen(d)[0]
        self.assertEqual(fo['status'], 'FLAG')
        self.assertIn("'is not False'", fo['reason'])
        self.assertIn('n_ok == len(...)', fo['reason'])
        self.assertIn('summarize.py', ' '.join(fo['evidence_paths']))

    def test_clean_code_passes(self):
        write(self.repo, 'clean_dev_v1_20261001/score.py', CLEAN_PY)
        d = self.delivery('clean_dev_v1_20261001')
        self.assertEqual(retro.adapter_literal(d)[0]['status'], 'PASS')
        self.assertEqual(retro.adapter_failopen(d)[0]['status'], 'PASS')
        self.assertEqual(retro.adapter_closure(d)[0]['status'], 'NOT_APPLICABLE')

    def test_closure_uses_main_lock_and_flags_unlocked_import_and_verdict_script(self):
        d = self.delivery(self.legacy())
        r = retro.adapter_closure(d)[0]
        self.assertEqual(r['status'], 'FLAG', r['reason'])
        self.assertIn('helper.py', r['reason'])
        self.assertIn('audit_results.py', r['reason'])  # AUDIT_SOURCE_LOCK does not extend the main lock

    def test_closure_follows_sys_path_insert(self):
        write(self.repo, 'lb_v1_20261024/lb1.py', 'import sys\nfrom pathlib import Path\nHERE = Path(__file__).parent\n'
                                                  'sys.path.insert(0, str(HERE.parent / "cls_dev_20261017"))\n'
                                                  'import fastmem\n')
        write(self.repo, 'lb_v1_20261024/LOCK.json', json.dumps({'code_sha256': {
            'lb1.py': 'a' * 64, '../cls_dev_20261017/fastmem.py': 'b' * 64}}))
        write(self.repo, 'cls_dev_20261017/fastmem.py', 'import bootstrap\nimport numpy\n')
        write(self.repo, 'cls_dev_20261017/bootstrap.py', 'X = 1\n')
        r = retro.adapter_closure(self.delivery('lb_v1_20261024'))[0]
        self.assertEqual(r['status'], 'FLAG', r['reason'])
        self.assertIn('cls_dev_20261017/bootstrap.py', r['reason'])
        self.assertNotIn('fastmem.py (', r['reason'])
        write(self.repo, 'lb_v1_20261024/LOCK.json', json.dumps({'code_sha256': {
            'lb1.py': 'a' * 64, '../cls_dev_20261017/fastmem.py': 'b' * 64, '../cls_dev_20261017/bootstrap.py': 'c' * 64}}))
        r = retro.adapter_closure(self.delivery('lb_v1_20261024'))[0]
        self.assertEqual(r['status'], 'PASS', r['reason'])

    def test_claim_soft_words_and_dev_confirm(self):
        d = self.delivery(self.legacy())
        r = retro.adapter_claim(d)[0]
        self.assertEqual(r['status'], 'FLAG')
        self.assertIn('largely met', r['reason'])
        self.assertIn('Native confirmation on D2', r['reason'])
        self.assertNotIn('Not a confirmation run', r['reason'])
        write(self.repo, 'quiet_v1_20261001/REPORT.md', '# Report\nThe primary gate is FAIL at 0.41.\n')
        self.assertEqual(retro.adapter_claim(self.delivery('quiet_v1_20261001'))[0]['status'], 'PASS')

    def test_claim_is_sentence_level(self):
        write(self.repo, 'ops_v1_20261001/REPORT.md',
              'A queued or partially staged upload is not verified. The execution gate was authorized.\n'
              'Immutable quarantine of partial attempts and an independent retry gate.\n'
              '## 10. Development, sealed confirmation and selection\n'
              'G4 keep: met in part on 3 of 4 worlds.\n')
        r = retro.adapter_claim(self.delivery('ops_v1_20261001'))[0]
        self.assertEqual(r['status'], 'FLAG')
        self.assertIn('G4 keep', r['reason'])
        self.assertNotIn('partially staged', r['reason'])
        self.assertNotIn('partial attempts', r['reason'])
        self.assertNotIn('sealed confirmation', r['reason'])

    def test_pointer_bool_keys(self):
        write(self.repo, 'PROJECT_DOCUMENTS/CURRENT_X.json', json.dumps({'a': {'confirmation_passes': True},
                                                                         'note': 'confirmation pending'}))
        src = retro.WorktreeSource(self.repo, with_history=False)
        r = retro.adapter_pointer(retro.Delivery(src, 'PROJECT_DOCUMENTS/CURRENT_X.json', kind='file'))[0]
        self.assertEqual(r['status'], 'FLAG')
        self.assertIn('/a/confirmation_passes', r['reason'])

    def test_ctrl_gzip_receipts(self):
        same = {'arm': 'S2', 'end_state_digests': {'W': ['a', 'b'], 'N_old_rel': ['a', 'b'], 'N_old_fact': ['c', 'd']}}
        diff = {'arm': 'S2', 'end_state_digests': {'W': ['a', 'b'], 'N_old_rel': ['x', 'b'], 'N_old_fact': ['c', 'd']}}
        write(self.repo, 'pkg_b/results/science/S2/1.json.gz', gzip.compress(json.dumps(same).encode()))
        write(self.repo, 'pkg_b/results/science/S2/2.json.gz', gzip.compress(json.dumps(same).encode()))
        r = retro.adapter_ctrl(self.delivery('pkg_b'))[0]
        self.assertEqual(r['status'], 'FLAG')
        self.assertIn('N_old_rel identical to W in 2/2 receipts', r['reason'])
        write(self.repo, 'pkg_c/results/science/S2/1.json.gz', gzip.compress(json.dumps(diff).encode()))
        self.assertEqual(retro.adapter_ctrl(self.delivery('pkg_c'))[0]['status'], 'PASS')

    def test_inert_identical_summaries(self):
        s = {'arm': 'LEARN', 'held': 0.5, 'old': 0.25, 'new': 0.75, 'wall_s': 3}
        write(self.repo, 'lb/results/c2/PAR_LEARN_D2.json', json.dumps(s))
        write(self.repo, 'lb/results/c2/PAR_FROZEN_D2.json', json.dumps(dict(s, arm='FROZEN', wall_s=9)))
        write(self.repo, 'lb/results/c2/PAR_SHUF_D2.json', json.dumps(dict(s, arm='SHUF', held=0.1)))
        write(self.repo, 'lb/results/c2/SUMMARY.json', json.dumps(s))
        write(self.repo, 'lb/results/c2/CONFIRM_SUMMARY.json', json.dumps(s))
        r = retro.adapter_inert(self.delivery('lb'))[0]
        self.assertEqual(r['status'], 'FLAG')
        self.assertIn('PAR_FROZEN_D2.json', r['reason'])
        self.assertNotIn('PAR_SHUF_D2.json', r['reason'])
        self.assertNotIn('CONFIRM_SUMMARY', r['reason'])  # a copy, not two arms

    def test_auth_and_paths(self):
        d = self.delivery(self.legacy())
        self.assertEqual(retro.adapter_auth(d)[0]['status'], 'FLAG')
        write(self.repo, self.legacy() + '/ops/AUTHORIZATION.json', '{}')
        self.assertEqual(retro.adapter_auth(self.delivery(self.legacy()))[0]['status'], 'PASS')
        p = retro.adapter_paths(self.delivery(self.legacy()))[0]
        self.assertEqual(p['status'], 'FLAG')
        self.assertEqual(p['_users'], 1)
        write(self.repo, 'nocycle_20261001/README.md', 'x\n')
        self.assertEqual(retro.adapter_auth(self.delivery('nocycle_20261001'))[0]['status'], 'NOT_APPLICABLE')

    def test_catalog_narrowed_finding_and_free_text_levels(self):
        kb = 'PROJECT_DOCUMENTS/KB/FINDINGS.jsonl'
        rows = [{'id': 'F-UF-15', 'kind': 'finding', 'claim': 'STOP-ONLY +22.10pp [20.47,23.72] vs DIRECT',
                 'status': 'provisional', 'effect': {'text': '+22.10pp'}},
                {'id': 'F-N-1', 'kind': 'narrowing', 'refers_to': 'F-UF-15', 'status': 'narrowed',
                 'caveat': 'FIXED_1 10.645% vs STOP 11.263% on NEW'}]
        write(self.repo, kb, ''.join(json.dumps(r) + '\n' for r in rows))
        c = 'PROJECT_DOCUMENTS/CENSUS_20261023'
        write(self.repo, c + '/CATALOG.md', '| P036 | stop | gain +22.10pp, positive 32/32 |\n'
                                            '| P037 | stop | +22.10pp but FIXED_1 explains it |\n')
        write(self.repo, c + '/EXPERIMENT_CATALOG.json', json.dumps({'rows': [{'evidence_level': 'Confirmed E1 component'},
                                                                              {'evidence_level': 'E3'}]}))
        rs = retro.adapter_catalog(self.delivery(c), self.repo)
        self.assertEqual([r['status'] for r in rs], ['FLAG', 'FLAG'])
        self.assertIn('CATALOG.md:1', rs[0]['reason'])
        self.assertNotIn('CATALOG.md:2', rs[0]['reason'])
        self.assertIn('FIXED_1', rs[0]['reason'])
        self.assertIn('1 distinct free-text evidence_level', rs[1]['reason'])

    def test_locktime_rules(self):
        h = retro.History([
            {'sha': 'r' * 40, 'ct': 0, 'parents': [], 'added': ['old/x.md']},
            {'sha': 'a' * 40, 'ct': 100, 'parents': ['r'], 'added': ['d1_dev_v1_20261001/results/S.json']},
            {'sha': 'b' * 40, 'ct': 200, 'parents': ['a'], 'added': ['d1_dev_v1_20261001/LOCK.json',
                                                                    'd2_dev_v1_20261001/DESIGN.md',
                                                                    'd2_dev_v1_20261001/results/SUMMARY.json']},
            {'sha': 'c' * 40, 'ct': 300, 'parents': ['b'], 'added': ['d3_x/SPEC.json']},
            {'sha': 'e' * 40, 'ct': 400, 'parents': ['c'], 'added': ['d3_x/results/R.json', 'p_a/x', 'p_b/y', 'p_c/z']},
        ], set())

        class FakeSrc(retro.Source):
            label = 'fake'

            def __init__(self, files):
                self._f = files

            def files(self, prefix):
                return [p for p in self._f if p.startswith(prefix + '/')]

            def exists(self, path):
                return path in self._f

            def read(self, path):
                return b'' if path in self._f else None

            def history(self):
                return h
        files = [p for c in h.commits for p in c['added']]
        src = FakeSrc(files)
        r1 = retro.adapter_locktime(retro.Delivery(src, 'd1_dev_v1_20261001'), [])[0]
        self.assertEqual(r1['status'], 'FLAG')
        self.assertIn('before the first lock artifact', r1['reason'])
        r2 = retro.adapter_locktime(retro.Delivery(src, 'd2_dev_v1_20261001'), [])[0]
        self.assertEqual(r2['status'], 'FLAG')
        self.assertIn('share commit bbbbbbbb', r2['reason'])
        r3 = retro.adapter_locktime(retro.Delivery(src, 'd3_x'), [])[0]
        self.assertEqual(r3['status'], 'NOT_APPLICABLE')
        self.assertIn(retro.BULK, r3['reason'])
        self.assertEqual(retro.lineage_stem('minifly_recoverability_design_v1_r2_20261023'),
                         retro.lineage_stem('minifly_recoverability_v1_r2_dev_20261023'))


class FakeSrc(retro.Source):
    """Files with contents and a fixed History; nothing is read from disk."""
    label = 'fake'

    def __init__(self, files, hist):
        self._f = dict(files)
        self._h = hist

    def files(self, prefix):
        return sorted(p for p in self._f if p.startswith(prefix + '/'))

    def exists(self, path):
        return path in self._f

    def read(self, path):
        v = self._f.get(path)
        return v.encode('utf-8') if isinstance(v, str) else v

    def history(self):
        return self._h


def commit(c, ct, parent, added):
    return {'sha': c * 40, 'ct': ct, 'parents': [parent * 40] if parent else [], 'added': list(added)}


class TestTightenedAdapters(RetroCase):
    """Patterns tightened against genuine false positives on real history (RETRO_20261024); each test keeps the
    true-positive shape of the FLAG list firing next to the false-positive shape that must stay quiet."""

    def test_claim_confirm_wording_needs_a_bound_dev_world(self):
        quiet = ('After confirmation, retire the concrete instances from future development.\n'
                 'This authorizes the frozen SAMPLE-GEN V2 DEV and conditional confirmation, analysis and audit.\n'
                 'All 492 accepted DEV/CONFIRM receipts are inside this archive.\n'
                 '## 10. Finite DEV and confirmation roster\n'
                 'Synthetic confirmation pass/fail cases, development no-advance/tie gates.\n')
        write(self.repo, 'design_dev_v1_20261011/cycles/cycle1/EXPERIMENT.md', quiet)
        r = retro.adapter_claim(self.delivery('design_dev_v1_20261011'))[0]
        self.assertEqual(r['status'], 'PASS', r['reason'])
        for i, ln in enumerate(('## 5. Native confirmation (T-PAR, D2, 150 passes, eta0 = 0.03)',
                                'It is recorded as a second development confirmation, not as evidence.',
                                'The gain was confirmed on the dev worlds D1-D4.',
                                '## Native confirmation on D2')):
            d = 'lb_dev_v1_%d' % i
            write(self.repo, d + '/cycles/CYCLE2.md', ln + '\n')
            r = retro.adapter_claim(self.delivery(d))[0]
            self.assertEqual(r['status'], 'FLAG', ln)
            self.assertIn('confirm wording on dev/tuning worlds', r['reason'])
        # a cycle heading that names a confirmation and no dev world at all still counts
        write(self.repo, 'cyc_dev_v1_1/cycles/CYCLE3.md', '## Confirmation result\n')
        self.assertEqual(retro.adapter_claim(self.delivery('cyc_dev_v1_1'))[0]['status'], 'FLAG')

    def test_closure_follows_hashed_closure_manifest_only_when_bytes_match(self):
        import hashlib
        d = 'stop_v1_20261023'
        write(self.repo, d + '/run.py', 'import sys\nsys.path.insert(0, "../dep_v1")\nimport helper\n')
        write(self.repo, 'dep_v1/helper.py', 'X = 1\n')
        manifest = json.dumps({'dep_v1/helper.py': 'e' * 64})
        write(self.repo, d + '/PARENT_CLOSURE.json', manifest)
        good = hashlib.sha256(manifest.encode()).hexdigest()
        write(self.repo, d + '/SOURCE_LOCK.json', json.dumps({'run.py': 'a' * 64, 'PARENT_CLOSURE.json': good}))
        r = retro.adapter_closure(self.delivery(d))[0]
        self.assertEqual(r['status'], 'PASS', r['reason'])
        write(self.repo, d + '/SOURCE_LOCK.json', json.dumps({'run.py': 'a' * 64, 'PARENT_CLOSURE.json': 'f' * 64}))
        r = retro.adapter_closure(self.delivery(d))[0]
        self.assertEqual(r['status'], 'FLAG', r['reason'])  # manifest bytes no longer match: not followed
        self.assertIn('dep_v1/helper.py', r['reason'])

    def test_claim_partial_is_soft_only_next_to_a_verdict(self):
        write(self.repo, 'rec_dev_v1_20261022/REPORT.md',
              '| 1 | Isolate native equations | PASS: four partial synthetic native events; parity within 1e-12 |\n')
        self.assertEqual(retro.adapter_claim(self.delivery('rec_dev_v1_20261022'))[0]['status'], 'PASS')
        for i, ln in enumerate(('* G2 retention: partially met on 3 of 4 worlds.', 'The primary gate: PASS (partial).',
                                '* **G1 memory - largely met** (84% vs 95%).')):
            write(self.repo, 'soft_dev_v1_%d/REPORT.md' % i, ln + '\n')
            r = retro.adapter_claim(self.delivery('soft_dev_v1_%d' % i))[0]
            self.assertEqual(r['status'], 'FLAG', ln)
            self.assertIn('soft wording on gate lines', r['reason'])

    def test_catalog_number_quote_needs_number_boundaries(self):
        kb = 'PROJECT_DOCUMENTS/KB/FINDINGS.jsonl'
        rows = [{'id': 'F-UF-15', 'kind': 'finding', 'claim': 'STOP-ONLY +22.10pp [20.47,23.72] vs DIRECT',
                 'status': 'provisional', 'effect': {'text': '+22.10pp'}},
                {'id': 'F-N-1', 'kind': 'narrowing', 'refers_to': 'F-UF-15', 'status': 'narrowed',
                 'caveat': 'FIXED_1 10.645% vs STOP 11.263% on NEW'}]
        write(self.repo, kb, ''.join(json.dumps(r) + '\n' for r in rows))
        c = 'hlm_20261023'
        write(self.repo, c + '/CATALOG.md', '| EDU-05 | [Frontiers](https://x.org/10.3389/fpsyg.2022.1032003/pdf) |\n')
        rs = retro.adapter_catalog(self.delivery(c), self.repo)
        self.assertEqual([r['status'] for r in rs], ['PASS'], rs)
        write(self.repo, c + '/CATALOG.md', '| P036 | stop | gain +22.10pp, positive 32/32 |\n')
        self.assertEqual([r['status'] for r in retro.adapter_catalog(self.delivery(c), self.repo)], ['FLAG'])

    def test_literal_skips_tests_negative_fixtures_and_computed_counters(self):
        d = 'lit_dev_v1_20261001'
        write(self.repo, d + '/tests/test_policy.py', 'state = {"private_readback_verified": True}\n')
        write(self.repo, d + '/source/test_retention.py', 'X = {"github_tree_verified": True}\n')
        write(self.repo, d + '/selfcheck.py', 'import copy\nentry = make()\nbad = copy.deepcopy(entry)\n'
                                              'bad["github_tree_verified"] = True\nreject(bad)\n'
                                              'forged = dict(entry, all_rows_verified=True)\n')
        write(self.repo, d + '/model.py', 'def empty_stats():\n    return dict(active_events=0, no_write_events=0)\n'
                                          'def step(s, plastic):\n    s["no_write_events"] += int(not plastic)\n')
        r = retro.adapter_literal(self.delivery(d))[0]
        self.assertEqual(r['status'], 'PASS', r['reason'])
        self.assertIn('2 test modules skipped', r['reason'])
        # mock.patch alone is not a test module; an unverified FINAL_AUDIT literal still fires next to it
        write(self.repo, d + '/finalize.py', 'from unittest.mock import patch\n'
                                             'save({"verdict": "ACCEPTED", "all_cycles_completed": True}, "FINAL_AUDIT.json")\n')
        r = retro.adapter_literal(self.delivery(d))[0]
        self.assertEqual(r['status'], 'FLAG')
        self.assertIn('finalize.py:2 all_cycles_completed', r['reason'])
        self.assertNotIn('no_write_events', r['reason'])
        self.assertNotIn('github_tree_verified', r['reason'])
        self.assertNotIn('all_rows_verified', r['reason'])
        self.assertEqual(retro.adapter_literal(self.delivery(self.legacy()))[0]['status'], 'FLAG')  # V6 shape kept

    def test_failopen_if_raise_guard_is_fail_closed(self):
        write(self.repo, 'fc_dev_v1_20261001/audit.py',
              'def audit(d, lock):\n    if d.get("development") is not False:\n        raise AssertionError("science only")\n'
              '    if lock["science_authorized"] is not False or d.get("x") is None:\n        raise ValueError("bad")\n')
        r = retro.adapter_failopen(self.delivery('fc_dev_v1_20261001'))[0]
        self.assertEqual(r['status'], 'PASS', r['reason'])
        # not a raise guard (the comparison becomes the pass), and a negated guard (raises only on an explicit False)
        write(self.repo, 'fc_dev_v1_20261001/reduce.py',
              'def gates(v):\n    if v.get("oracle") is not False:\n        ok = True\n'
              '    if not (v.get("copy") is not False):\n        raise ValueError("copy gate failed")\n    return ok\n')
        r = retro.adapter_failopen(self.delivery('fc_dev_v1_20261001'))[0]
        self.assertEqual(r['status'], 'FLAG')
        self.assertIn('reduce.py:2', r['reason'])
        self.assertIn('reduce.py:4', r['reason'])
        self.assertNotIn('audit.py', r['reason'])
        self.assertEqual(retro.adapter_failopen(self.delivery(self.legacy()))[0]['status'], 'FLAG')  # LEARNED-BANK shape

    def test_inert_compares_arms_not_worlds_or_hashes(self):
        cert = {'accepted': True, 'records_applied': 1536, 'observed_event_rows': 9789, 'science_jobs': 1}
        for w in ('202610129001', '202610129032'):
            write(self.repo, 'cert_dev_v1/results/certificates/%s_COMP-FIXED-DIAG.json' % w, json.dumps(cert))
        for h in ('7d18564a2d124e60', 'ab3ebab98c8541f1'):
            write(self.repo, 'cert_dev_v1/results/ops/PARALLEL_AUTHORIZATION-%s.json' % h, json.dumps(cert))
        r = retro.adapter_inert(self.delivery('cert_dev_v1'))[0]
        self.assertEqual(r['status'], 'PASS', r['reason'])
        write(self.repo, 'cert_dev_v1/results/c2/PAR_LEARN_D2.json', json.dumps({'held': 0.5, 'old': 0.2, 'new': 0.7}))
        write(self.repo, 'cert_dev_v1/results/c2/PAR_FROZEN_D2.json', json.dumps({'held': 0.5, 'old': 0.2, 'new': 0.7}))
        r = retro.adapter_inert(self.delivery('cert_dev_v1'))[0]
        self.assertEqual(r['status'], 'FLAG')
        self.assertIn('PAR_FROZEN_D2.json', r['reason'])

    def test_locktime_lock_bound_cycle_receipts_and_post_hoc_lock(self):
        D = 'studies/rc-20261003'
        lock = json.dumps({'acceptances': {'receipts/cycle1/smoke.json': 'a' * 64,
                                           'receipts/cycle3/pilot/manifest.json': 'b' * 64}})
        h = retro.History([
            commit('1', 0, None, ['old/x.md']),
            commit('2', 100, '1', [D + '/receipts/cycle1/smoke.json', D + '/protocol/DESIGN_CYCLE1.md']),
            commit('3', 200, '2', [D + '/protocol/OFFICIAL_LOCK.json', D + '/receipts/cycle3/pilot/manifest.json',
                                   D + '/receipts/cycle3/pilot/audit-000.json', D + '/receipts/science/1/audit.json']),
            commit('4', 300, '3', [D + '/results/report/ROUND1_REPORT.md']),
        ], set())
        files = {p: '{}' for c in h.commits for p in c['added']}
        files[D + '/protocol/OFFICIAL_LOCK.json'] = lock
        r = retro.adapter_locktime(retro.Delivery(FakeSrc(files, h), D), [])[0]
        self.assertEqual(r['status'], 'FLAG', r['reason'])
        self.assertIn('share commit 33333333', r['reason'])
        self.assertIn('receipts/science/1/audit.json', r['reason'])
        self.assertIn('hashed by the lock as accepted inputs', r['reason'])
        self.assertNotIn('before the first lock artifact', r['reason'])
        # the official results arrive one commit after the lock: PASS
        files2 = dict(files)
        del files2[D + '/receipts/science/1/audit.json']
        h2 = retro.History([dict(c, added=[p for p in c['added'] if p in files2]) for c in h.commits], set())
        r = retro.adapter_locktime(retro.Delivery(FakeSrc(files2, h2), D), [])[0]
        self.assertEqual(r['status'], 'PASS', r['reason'])
        # a post-hoc lock that hashes non-cycle results gains nothing: those results stay governed
        E = 'posthoc_dev_v1_20261001'
        h3 = retro.History([commit('1', 0, None, ['x.md']),
                            commit('5', 100, '1', [E + '/results/SUMMARY.json']),
                            commit('6', 200, '5', [E + '/SOURCE_LOCK.json'])], set())
        f3 = {E + '/results/SUMMARY.json': '{}', E + '/SOURCE_LOCK.json': json.dumps({'results/SUMMARY.json': 'c' * 64})}
        r = retro.adapter_locktime(retro.Delivery(FakeSrc(f3, h3), E), [])[0]
        self.assertEqual(r['status'], 'FLAG')
        self.assertIn('before the first lock artifact', r['reason'])

    def test_locktime_shallow_boundary_is_not_applicable(self):
        E = 'sg_dev_v2_20261011'
        h = retro.History([commit('9', 0, None, [E + '/results/SELECTION_LOCK.json', E + '/results/CONFIRM_SUMMARY.json',
                                                 E + '/SOURCE_LOCK.json'])], {'9' * 40})
        f = {E + '/results/SELECTION_LOCK.json': '{"locked_epoch": 1}', E + '/results/CONFIRM_SUMMARY.json': '{}',
             E + '/SOURCE_LOCK.json': '{}'}
        r = retro.adapter_locktime(retro.Delivery(FakeSrc(f, h), E), [])[0]
        self.assertEqual(r['status'], 'NOT_APPLICABLE')
        self.assertIn('shallow clone: commit 99999999 is a grafted boundary', r['reason'])
        # the quiet-aspect run uses SELECTION_LOCK as the lock and reports the in-artifact hash chain, still NA
        sel = f[E + '/results/SELECTION_LOCK.json'].encode()
        f[E + '/results/CONFIRM_ADMISSION.json'] = json.dumps({'selection_sha256': retro.hashlib.sha256(sel).hexdigest()})
        h.commits[0]['added'].append(E + '/results/CONFIRM_ADMISSION.json')
        h.first_add[E + '/results/CONFIRM_ADMISSION.json'] = 0
        q = dict(retro.MUST_STAY_QUIET[1], delivery=E)
        d = retro.Delivery(FakeSrc(f, h), E)
        rows = retro.quiet_aspect_rows(q, d, [d], None)
        self.assertEqual([(x['check_id'], x['status']) for x in rows], [('R-LOCKTIME', 'NOT_APPLICABLE')])
        self.assertIn('shallow clone', rows[0]['reason'])
        self.assertIn('results/CONFIRM_ADMISSION.json names sha256', rows[0]['reason'])
        self.assertIn('self-reported, not commit order', rows[0]['reason'])

    def test_aspect_rows_restrict_path_only_adapters_to_the_aspect(self):
        D = 'studies/rcenter-survivor-rounds-20261003'
        write(self.repo, D + '/operations/transport.py', 'state = {"private_readback_verified": True}\n')
        write(self.repo, D + '/source/runtime/analyze.py', 'def t(x):\n    return {"held_ok": x > 0}\n')
        write(self.repo, D + '/results/report/ROUND1_REPORT.md', '# Report\nThe combined useful-completion gate fails.\n')
        d = self.delivery(D)
        self.assertEqual(retro.adapter_literal(d)[0]['status'], 'FLAG')  # delivery level: operations code
        rows = retro.quiet_aspect_rows(retro.MUST_STAY_QUIET[0], d, [d], None)
        st = {r['check_id']: r['status'] for r in rows}
        self.assertEqual(set(st), set(retro.MUST_STAY_QUIET[0]['adapters']))
        self.assertEqual(st['R-LITERAL'], 'PASS')
        self.assertEqual(st['R-CLAIM'], 'PASS')
        write(self.repo, D + '/source/runtime/analyze.py', 'RESULT = {"exposure_overlap": 0}\n')
        rows = retro.quiet_aspect_rows(retro.MUST_STAY_QUIET[0], self.delivery(D), [d], None)
        self.assertEqual({r['check_id']: r['status'] for r in rows}['R-LITERAL'], 'FLAG')


class TestQuietAcceptance(RetroCase):
    RC = 'studies/rcenter-survivor-rounds-20261003'
    SG = 'addition_sample_generalization_dev_v2_20261011'
    V51 = 'PROJECT_DOCUMENTS/V51_CLAUDE_AUDIT_RECONCILIATION_20261020'

    def flag_list(self):
        rows = []
        for it in retro.MUST_FLAG:
            dl = it.get('delivery', 'x_%s' % it['id'])
            ev = [it['commit'] + '0' * 32] if 'commit' in it else []
            rows.append(retro.row(dl, it['adapter'], 'FLAG', 'hit %s' % it.get('match', ''), ev))
        return rows

    def test_scoping_is_explicit_and_every_adapter_accounted_for(self):
        for q in retro.MUST_STAY_QUIET:
            self.assertTrue(q['adapters'])
            self.assertEqual(set(q['adapters']) | set(q['excluded']), set(retro.ADAPTERS), q['delivery'])
            self.assertFalse(set(q['adapters']) & set(q['excluded']))
            for a in retro.COUNT_ADAPTERS:
                self.assertIn(a, q['excluded'])
            for why in list(q['adapters'].values()) + list(q['excluded'].values()):
                self.assertTrue(why.strip())
        self.assertEqual(list(retro.MUST_STAY_QUIET[1]['adapters']), ['R-LOCKTIME'])

    def test_scoped_unscoped_and_adjudicated_rates(self):
        rows = self.flag_list() + [
            retro.row(self.RC, 'R-LOCKTIME', 'FLAG', 'lock and results share commit 34246217 (x)'),
            retro.row(self.RC, 'R-LITERAL', 'FLAG', 'integrity booleans written as literals: x', [
                self.RC + '/operations/transport.py']),
            retro.row(self.RC, 'R-CLAIM', 'PASS', 'ok'),
            retro.row(self.RC, 'R-FAILOPEN', 'PASS', 'ok'),
            retro.row(self.SG, 'R-LOCKTIME', 'NOT_APPLICABLE', 'shallow clone: commit 66f304ca is a grafted boundary'),
            retro.row(self.SG, 'R-PATHS', 'FLAG', '25 files with absolute local paths'),
            retro.row(self.SG, 'R-CLAIM', 'FLAG', 'confirm wording on dev/tuning worlds: new, unreviewed'),
            retro.row(self.V51, 'R-LITERAL', 'PASS', 'ok'), retro.row(self.V51, 'R-FAILOPEN', 'PASS', 'ok')]
        aspect = {self.RC: [retro.row(self.RC, a, 'PASS', 'ok') for a in ('R-FAILOPEN', 'R-CLAIM', 'R-LITERAL')] +
                  [retro.row(self.RC, 'R-CTRL', 'NOT_APPLICABLE', 'no receipts')],
                  self.SG: [retro.row(self.SG, 'R-LOCKTIME', 'NOT_APPLICABLE', 'shallow clone: grafted boundary')],
                  self.V51: [retro.row(self.V51, 'R-LITERAL', 'PASS', 'ok'), retro.row(self.V51, 'R-FAILOPEN', 'PASS', 'ok')]}
        acc = retro.acceptance(rows, True, aspect)
        self.assertEqual(acc['recall'], 1.0)
        self.assertEqual((acc['quiet_flags'], acc['quiet_applied']), (0, 5))
        q = {x['delivery']: x for x in acc['must_stay_quiet']}
        self.assertEqual(q[self.SG]['status'], 'not measurable (NOT_APPLICABLE)')
        self.assertIn('shallow clone', q[self.SG]['not_applicable'][0])
        self.assertEqual(q[self.RC]['status'], 'quiet')
        # unscoped: 8 applied, 4 FLAGs; LOCKTIME and PATHS are reviewed true findings, LITERAL a reviewed false
        # positive, and the new SAMPLE-GEN R-CLAIM FLAG is unadjudicated, so it counts as false
        self.assertEqual((acc['quiet_flags_unscoped'], acc['quiet_applied_unscoped']), (4, 8))
        self.assertEqual(acc['quiet_false_flags_unscoped_adjudicated'], 2)
        verdicts = {(f['delivery'], f['check_id']): f['verdict'] for f in acc['quiet_unscoped_flags']}
        self.assertEqual(verdicts[(self.RC, 'R-LOCKTIME')], 'true_finding')
        self.assertEqual(verdicts[(self.RC, 'R-LITERAL')], 'false_positive')
        self.assertTrue(verdicts[(self.SG, 'R-CLAIM')].startswith('unadjudicated'))
        self.assertTrue(acc['accepted_scoped'])
        self.assertFalse(acc['accepted_unscoped_adjudicated'])  # 2/8 = 25% > 15%
        self.assertFalse(acc['accepted'])
        # an adjudication only applies while its evidence still matches the row
        rows2 = [r for r in rows if r['check_id'] != 'R-CLAIM' or r['delivery'] != self.SG]
        acc2 = retro.acceptance(rows2, True, aspect)
        self.assertEqual(acc2['quiet_false_flags_unscoped_adjudicated'], 1)
        self.assertTrue(acc2['accepted'])
        rows3 = [dict(r, reason='lock and results share commit deadbeef') if r['check_id'] == 'R-LOCKTIME' and
                 r['delivery'] == self.RC else r for r in rows2]
        self.assertEqual(retro.acceptance(rows3, True, aspect)['quiet_false_flags_unscoped_adjudicated'], 2)
        # a FLAG on the aspect itself is always false, whatever the adjudication table says
        aspect_bad = dict(aspect, **{self.RC: [retro.row(self.RC, 'R-FAILOPEN', 'FLAG', 'is not False')]})
        acc4 = retro.acceptance(rows2, True, aspect_bad)
        self.assertEqual((acc4['quiet_flags'], acc4['quiet_applied']), (1, 3))
        self.assertFalse(acc4['accepted'])
        self.assertFalse(retro.acceptance(rows2, False, aspect)['accepted'])  # only retro --all is acceptance


class TestCommand(RetroCase):
    def args(self, **kw):
        ns = argparse.Namespace(command='retro', dir=None, all=False, out=None, repo_root=str(self.repo), git=None,
                                json=False, extra=[])
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns

    def run_cmd(self, ns):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = retro.cmd_retro(ns)
        return rc, out.getvalue(), err.getvalue()

    def test_out_inside_delivery_refused(self):
        d = self.legacy()
        for bad in (self.repo / d / 'retro_out', self.repo / 'minifly_b' / 'x', self.repo / 'tools' / 'minifly_check' / 'o'):
            rc, out, err = self.run_cmd(self.args(dir=d, out=str(bad)))
            self.assertNotEqual(rc, 0)
            self.assertIn('refuses --out', err)
            self.assertFalse(bad.exists())

    def test_single_dir_run_writes_only_out(self):
        d = self.legacy()
        before = sorted(p.relative_to(self.repo).as_posix() for p in self.repo.rglob('*'))
        out = self.tmp / 'scratch' / 'retro'
        rc, so, se = self.run_cmd(self.args(dir=d, out=str(out)))
        self.assertEqual(rc, 0, se)
        after = sorted(p.relative_to(self.repo).as_posix() for p in self.repo.rglob('*'))
        self.assertEqual(before, after)
        files = sorted(p.name for p in out.iterdir())
        self.assertEqual(len(files), 2)
        full = json.loads((out / [f for f in files if f.endswith('.json')][0]).read_text())
        by = {(r['check_id'], r['status']) for r in full['rows']}
        self.assertIn(('R-LITERAL', 'FLAG'), by)
        self.assertIn(('R-FAILOPEN', 'FLAG'), by)
        self.assertIn(('R-LOCKTIME', 'NOT_APPLICABLE'), by)  # no git history in this temp dir
        for r in full['rows']:
            self.assertEqual(set(r), {'checker_sha256', 'delivery', 'check_id', 'status', 'reason', 'evidence_paths'})
            self.assertIn(r['status'], ('PASS', 'FLAG', 'NOT_APPLICABLE'))
        self.assertFalse(full['acceptance']['accepted'])
        self.assertIn('single-delivery', full['acceptance']['scope_note'])

    def test_retro_never_imports_delivery_code(self):
        d = 'trap_dev_v1_20261001'
        write(self.repo, d + '/boom.py', 'raise SystemExit("executed")\nimport os\nos.remove("/nonexistent")\n')
        write(self.repo, d + '/SOURCE_LOCK.json', json.dumps({'boom.py': 'a' * 64}))
        before = set(sys.modules)
        rc, so, se = self.run_cmd(self.args(dir=d, out=str(self.tmp / 'o')))
        self.assertEqual(rc, 0, se)
        self.assertNotIn('boom', set(sys.modules) - before)

    def test_acceptance_bookkeeping(self):
        rows = [retro.row('minifly_b', 'R-CTRL', 'FLAG', 'x'),
                retro.row('minifly_cls_dev_20261017', 'R-CLAIM', 'PASS', 'no soft words'),
                retro.row('studies/rcenter-survivor-rounds-20261003', 'R-CLAIM', 'FLAG', 'false flag'),
                retro.row('studies/rcenter-survivor-rounds-20261003', 'R-FAILOPEN', 'PASS', 'ok'),
                retro.row('studies/rcenter-survivor-rounds-20261003', 'R-LITERAL', 'FLAG', 'outside the statistics aspect'),
                retro.row('studies/rcenter-survivor-rounds-20261003', 'R-PATHS', 'FLAG', 'a count, never scored quiet'),
                retro.row('x', 'R-LOCKTIME', 'NOT_APPLICABLE', retro.BULK + ': commit 3065bb1a adds 3', ['3065bb1a00'])]
        acc = retro.acceptance(rows, True)
        st = {i['id']: i['status'] for i in acc['must_flag']}
        self.assertEqual(st['CTRL-minifly_b'], 'flagged')
        self.assertEqual(st['CLAIM-cls'], 'missed')
        self.assertEqual(st['LOCKTIME-3065bb1a'], 'not artifact-permitting')
        self.assertEqual(st['LOCKTIME-43d04b5e'], 'missed')
        self.assertEqual(acc['quiet_applied'], 2)
        self.assertEqual(acc['quiet_flags'], 1)
        self.assertEqual((acc['quiet_flags_unscoped'], acc['quiet_applied_unscoped']), (3, 4))
        self.assertFalse(acc['accepted'])


if __name__ == '__main__':
    unittest.main()
