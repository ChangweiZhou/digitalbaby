"""Unit tests for decide.py (WP3): Student-t, Hoeffding, the declarative rule of SPEC 9.5, custom reducers,
OC (SPEC 9.8) and DEC-01. Synthetic cells only (reducercheck.Synth); hand-computed expectations."""
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402
import decide  # noqa: E402
import reducercheck  # noqa: E402
from test_scorer import NAME, DeliveryCase, base_lock, jsonl, write  # noqa: E402

# Tabulated one-sided Student-t quantiles t(1 - alpha, df).
T_TABLE = {(1, 0.05): 6.3138, (1, 0.025): 12.7062, (5, 0.05): 2.0150, (5, 0.025): 2.5706,
           (7, 0.05): 1.8946, (7, 0.025): 2.3646, (30, 0.05): 1.6973, (30, 0.025): 2.0423}


def lock_for(**primary):
    lock = base_lock()
    lock['primary'].update(primary)
    return lock


class TestStats(unittest.TestCase):
    def test_t_quantiles_against_tables(self):
        for (df, alpha), want in sorted(T_TABLE.items()):
            got = decide.t_ppf(1 - alpha, df)
            self.assertAlmostEqual(got, want, delta=5e-4, msg='df=%d alpha=%s' % (df, alpha))
            self.assertAlmostEqual(decide.t_cdf(got, df), 1 - alpha, delta=1e-9)
        self.assertAlmostEqual(decide.t_ppf(0.05, 7), -decide.t_ppf(0.95, 7), places=9)
        # df = 1 is Cauchy: CDF(t) = 1/2 + atan(t)/pi
        self.assertAlmostEqual(decide.t_cdf(2.0, 1), 0.5 + math.atan(2.0) / math.pi, places=10)

    def test_betainc(self):
        self.assertAlmostEqual(decide.betainc(1.0, 1.0, 0.3), 0.3, places=12)
        self.assertAlmostEqual(decide.betainc(2.5, 3.5, 0.4) + decide.betainc(3.5, 2.5, 0.6), 1.0, places=12)
        self.assertAlmostEqual(decide.betainc(2.0, 1.0, 0.5), 0.25, places=12)  # I_x(2,1) = x^2

    def test_hoeffding_when_sd_zero(self):
        lcb, method, q, sd = decide.lower_bound([0.4] * 8, 0.05)
        self.assertEqual(method, 'hoeffding')
        self.assertAlmostEqual(lcb, 0.4 - 2 * math.sqrt(math.log(20) / 16), places=12)
        lcb, method, q, sd = decide.lower_bound([0.45, 0.4, 0.35, 0.4, 0.45, 0.4, 0.35, 0.4], 0.05)
        self.assertEqual(method, 't')
        self.assertAlmostEqual(lcb, 0.4 - 1.8946 * math.sqrt(0.01 / 7) / math.sqrt(8), places=4)
        ub, m, _q, _sd = decide.upper_bound([0.0] * 8, 0.05)
        self.assertEqual((ub, m), (0.0, 't_sd0'))
        ub, m, _q, _sd = decide.upper_bound([0.02], 0.05)
        self.assertEqual(m, 'hoeffding')


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.lock = lock_for()
        self.s = reducercheck.Synth(self.lock)

    def test_clean_positive_passes(self):
        v = self.s.evaluate(self.s.clean())
        self.assertEqual(v['verdict'], 'PASS', v['reasons'])
        self.assertEqual(v['primary']['lcb_method'], 't')
        self.assertEqual(v['decide_version'], decide.DECIDE_VERSION)
        self.assertEqual(len(v['cells_sha256']), 64)

    def test_hoeffding_identical_worlds_fail(self):
        """Candidate 18/20 in every world vs chance 0.1: d = 0.8 everywhere, sd = 0 -> Hoeffding
        LCB = 0.8 - 2 sqrt(ln 20 / 16) = 0.8 - 0.8654 = -0.0654 < 0.05 -> FAIL."""
        v = self.s.evaluate(self.s.cells(self.s.flat(18)))
        self.assertEqual(v['verdict'], 'FAIL')
        self.assertEqual(v['primary']['lcb_method'], 'hoeffding')
        self.assertAlmostEqual(v['primary']['lcb'], 0.8 - 2 * math.sqrt(math.log(20) / 16), places=9)

    def test_precondition_order(self):
        c = self.s.clean()
        pre = self.s.pre()
        v = decide.evaluate(c, self.lock, dict(pre, inert_alarms=[{'kind': 'x'}], adm_tb='FAIL', adm_ctrl='FAIL'))
        self.assertEqual((v['verdict'], v['invalid_reason']), ('INVALID', 'INTEGRITY_ALARM'))
        v = decide.evaluate(c, self.lock, dict(pre, adm_tb='FAIL', adm_ctrl='FAIL'))
        self.assertEqual(v['verdict'], 'CANNOT_TEST_CLAIM_AS_SPECIFIED')
        v = decide.evaluate(c, self.lock, dict(pre, adm_ctrl='FAIL'))
        self.assertEqual((v['verdict'], v['invalid_reason']), ('INVALID', 'TREATMENT_NOT_DELIVERED'))
        v = decide.evaluate(c, self.lock, dict(pre, adm_ctrl='MISSING'))
        self.assertEqual((v['verdict'], v['shadow_verdict']), ('INCOMPLETE', 'PASS'))
        # a missing cell: INCOMPLETE with a shadow verdict on the complete paired worlds (7 < min_worlds 8: FAIL)
        c2 = self.s.clean()
        c2['cells'] = [g for g in c2['cells'] if not (g['arm'] == 'NPT' and g['world'] == self.s.worlds[0])]
        v = decide.evaluate(c2, self.lock, pre)
        self.assertEqual(v['verdict'], 'INCOMPLETE')
        self.assertEqual(v['shadow_verdict'], 'FAIL')
        self.assertEqual(len(v['shadow_worlds']), 7)
        # missing beats non-finite; non-finite alone gives INVALID NONFINITE
        c3 = self.s.clean()
        c3['cells'][0]['n_exact'] = float('nan')
        v = decide.evaluate(c3, self.lock, pre)
        self.assertEqual((v['verdict'], v['invalid_reason']), ('INVALID', 'NONFINITE'))
        c2['cells'][0]['n_exact'] = None
        self.assertEqual(decide.evaluate(c2, self.lock, pre)['verdict'], 'INCOMPLETE')
        # an unknown precondition is never a PASS
        v = decide.evaluate(c, self.lock, {'expected_worlds': self.s.worlds, 'n_items_per_pool': pre['n_items_per_pool']})
        self.assertEqual(v['verdict'], 'INCOMPLETE')

    def test_dev_worlds_and_world_count(self):
        v = self.s.evaluate(self.s.clean(role='dev'))
        self.assertEqual((v['verdict'], v['shadow_verdict']), ('DEV_NOT_EVIDENCE', 'PASS'))
        v = self.s.evaluate(self.s.clean())
        self.assertEqual(v['verdict'], 'PASS')
        pre = dict(self.s.pre(), expected_worlds=self.s.worlds + ['wo_extra'])
        self.assertEqual(decide.evaluate(self.s.clean(), self.lock, pre)['verdict'], 'INCOMPLETE')
        pre = dict(self.s.pre(), expected_n_worlds=9)
        self.assertEqual(decide.evaluate(self.s.clean(), self.lock, pre)['verdict'], 'INCOMPLETE')

    def test_aggregators_k_of_n_and_all_worlds(self):
        cand = {w: (18 if i < 5 else 2) for i, w in enumerate(self.s.worlds)}  # 5 worlds clear the margin, 3 do not
        lock = lock_for(aggregator={'type': 'k_of_n', 'k': 5}, per_world_floor=-1.0, absolute_floor=0.0)
        s = reducercheck.Synth(lock)
        v = s.evaluate(s.cells(cand))
        self.assertEqual(v['verdict'], 'PASS', v['reasons'])
        self.assertEqual(v['primary']['count_above'], 5)
        lock = lock_for(aggregator={'type': 'k_of_n', 'k': 6}, per_world_floor=-1.0, absolute_floor=0.0)
        s = reducercheck.Synth(lock)
        self.assertEqual(s.evaluate(s.cells(cand))['verdict'], 'FAIL')
        lock = lock_for(aggregator={'type': 'all_worlds'})
        s = reducercheck.Synth(lock)
        self.assertEqual(s.evaluate(s.cells(cand))['verdict'], 'FAIL')
        self.assertEqual(s.evaluate(s.cells(s.flat(18)))['verdict'], 'PASS')  # no LCB: identical worlds pass
        lock = lock_for(aggregator={'type': 'majority'})
        v = decide.evaluate(self.s.clean(), lock, self.s.pre())
        self.assertEqual(v['verdict'], 'FAIL')
        self.assertIn('aggregator', ' '.join(v['reasons']))

    def test_comparator_is_max_of_chance_arms_and_ladder(self):
        s = self.s
        lvl = s.level(0.9)
        v = s.evaluate(s.cells(lvl, comps={'NPT': s.level(0.88)}))
        self.assertEqual(v['verdict'], 'FAIL')
        self.assertTrue(all(x['comparator'] == 'NPT' for x in v['primary']['per_world_detail']))
        v = s.evaluate(s.cells(lvl, ladder={'NEAREST': s.level(0.88)}))
        self.assertEqual(v['verdict'], 'FAIL')
        self.assertTrue(all(x['comparator'] == 'shortcut:NEAREST' for x in v['primary']['per_world_detail']))
        # members outside the pool subset are ignored: EXACT is not a member for TAUGHT
        lock = lock_for(pool='TAUGHT')
        lock['primary']['n_items_per_pool']['TAUGHT'] = 10
        s2 = reducercheck.Synth(lock)
        c = s2.clean()
        for key in c['ladder']:
            if '|TAUGHT|' in key:
                c['ladder'][key]['EXACT'] = 1.0
        self.assertEqual(s2.evaluate(c)['verdict'], 'PASS')

    def test_implied_comparators_close_declaration_minimization(self):
        """SPEC 20.6 (verifier probe 2c): every declared no_prior_teaching, shuf, frozen or lesion arm joins the
        decision comparator even when comparator_arms leaves it out; oracle_plumbing, positive_control and
        evaluator CEILING_/FLOOR_ arms never do, and the testbed view adds none. Round 1 (SPEC 20.14): an arm of role
        'other' joins too (relabel loophole); this test pinned the old exclusion and is tightened."""
        lock = base_lock()
        lock['arms'] += [{'id': 'FROZEN', 'role': 'frozen', 'zero_channels': ['pair_memory']},
                         {'id': 'LES', 'role': 'lesion', 'zero_channels': ['readout']},
                         {'id': 'ORACLE', 'role': 'oracle_plumbing', 'zero_channels': []},
                         {'id': 'POS', 'role': 'positive_control', 'zero_channels': []},
                         {'id': 'CEILING_X', 'role': 'frozen', 'zero_channels': []},
                         {'id': 'OTHER', 'role': 'other', 'zero_channels': []}]
        p = decide.parse_params(lock)[0]
        self.assertEqual(p['declared_comparator_arms'], ['NPT'])
        self.assertEqual(p['implied_comparator_arms'], ['SHUF', 'FROZEN', 'LES', 'OTHER'])
        self.assertEqual(p['comparator_arms'], ['NPT', 'SHUF', 'FROZEN', 'LES', 'OTHER'])
        self.assertEqual(decide.decision_comparator_arms(lock), ['NPT', 'SHUF', 'FROZEN', 'LES', 'OTHER'])
        self.assertEqual(decide.parse_params(lock, implied_comparators=False)[0]['comparator_arms'], ['NPT'])
        # a declared comparator is listed once, in its declared position
        lock2 = base_lock()
        lock2['primary']['comparator_arms'] = ['SHUF', 'NPT']
        self.assertEqual(decide.parse_params(lock2)[0]['comparator_arms'], ['SHUF', 'NPT'])
        s = reducercheck.Synth(lock)
        lvl = s.level(0.9)
        # clean positive still passes with every implied control at chance
        v = s.evaluate(s.clean())
        self.assertEqual(v['verdict'], 'PASS', v['reasons'])
        self.assertEqual(v['primary']['implied_comparator_arms'], ['SHUF', 'FROZEN', 'LES', 'OTHER'])
        # probe 2c: an undeclared FROZEN arm equal to the candidate never lets it pass; nor an arm relabelled 'other'
        for arm in ('FROZEN', 'LES', 'SHUF', 'OTHER'):
            v = s.evaluate(s.cells(lvl, comps={arm: dict(lvl)}))
            self.assertEqual(v['verdict'], 'FAIL', (arm, v['reasons']))
            self.assertTrue(all(x['comparator'] == arm for x in v['primary']['per_world_detail']), arm)
        # round-1 repair (verifier probe E2): a second arm of role 'candidate' joins the comparator as well
        lock3 = base_lock()
        lock3['arms'] += [{'id': 'W2', 'role': 'candidate', 'zero_channels': []}]
        self.assertEqual(decide.parse_params(lock3)[0]['implied_comparator_arms'], ['SHUF', 'W2'])
        s3 = reducercheck.Synth(lock3)
        lvl3 = s3.level(0.9)
        self.assertEqual(s3.evaluate(s3.clean())['verdict'], 'PASS')
        v = s3.evaluate(s3.cells(lvl3, comps={'W2': dict(lvl3)}))
        self.assertEqual(v['verdict'], 'FAIL', v['reasons'])
        self.assertTrue(all(x['comparator'] == 'W2' for x in v['primary']['per_world_detail']))
        # strong ORACLE / positive-control arms are not comparators (no cells are even required)
        c = s.clean()
        c['cells'] = [g for g in c['cells'] if g['arm'] not in ('ORACLE', 'POS')]
        self.assertEqual(s.evaluate(c)['verdict'], 'PASS')
        # a missing implied comparator is a missing cell: INCOMPLETE
        c = s.clean()
        c['cells'] = [g for g in c['cells'] if not (g['arm'] == 'FROZEN' and g['world'] == s.worlds[0])]
        v = s.evaluate(c)
        self.assertEqual(v['verdict'], 'INCOMPLETE')
        self.assertTrue(any('FROZEN' in r and '(comparator)' in r for r in v['reasons']), v['reasons'])

    def test_floors_gates_and_guards(self):
        s = self.s
        lock = lock_for(absolute_floor=0.95)
        s2 = reducercheck.Synth(lock)
        v = s2.evaluate(s2.clean())
        self.assertEqual(v['verdict'], 'FAIL')
        self.assertFalse(v['floors']['absolute_floor']['pass'])
        lock = base_lock(gates=[{'id': 'stop_gate', 'description': 'x'}])
        s3 = reducercheck.Synth(lock)
        self.assertEqual(s3.evaluate(s3.clean())['verdict'], 'PASS')
        v = decide.evaluate(s3.clean(), lock, dict(s3.pre(), gates={}))
        self.assertEqual(v['verdict'], 'FAIL')
        self.assertEqual(v['gates'], {'stop_gate': 'MISSING'})
        # old_drop under paired_lcb uses the upper bound: drops 0/2/0/2... items of 10 -> mean 0.1, UCB > 0.1
        end = s.level(0.9, s.n_old)
        final = {w: k - (2 if i % 2 else 0) for i, (w, k) in enumerate(sorted(end.items()))}
        v = s.evaluate(s.cells(s.level(0.9), old_end=end, old_final=final))
        self.assertEqual(v['guards']['old_drop'], 'FAIL')
        self.assertGreater(v['guard_detail']['old_drop']['bound'], 0.1)
        self.assertAlmostEqual(v['guard_detail']['old_drop']['mean_drop'], 0.1)
        # identical zero drops: the t upper bound is the mean (0), the guard passes
        v = s.evaluate(s.clean())
        self.assertEqual(v['guard_detail']['old_drop']['bound_method'], 't_sd0')
        # an OLD pool without the required guards fails
        lock = base_lock(guards=[])
        s4 = reducercheck.Synth(lock)
        v = s4.evaluate(s4.clean())
        self.assertEqual(v['verdict'], 'FAIL')
        self.assertEqual(v['guards'], {'old_drop': 'MISSING_IN_LOCK', 'old_acq': 'MISSING_IN_LOCK'})

    def test_lock_problems_fail_and_nonfinite_lock_invalid(self):
        lock = lock_for(chance=None)
        v = decide.evaluate(self.s.clean(), lock, self.s.pre())
        self.assertEqual(v['verdict'], 'FAIL')
        lock = lock_for(margin_abs=float('nan'))
        v = decide.evaluate(self.s.clean(), lock, self.s.pre())
        self.assertEqual((v['verdict'], v['invalid_reason']), ('INVALID', 'NONFINITE'))
        lock = lock_for(metric='accuracy')
        self.assertEqual(decide.evaluate(self.s.clean(), lock, self.s.pre())['verdict'], 'FAIL')
        lock = lock_for(strata_counted=[])
        self.assertEqual(decide.evaluate(self.s.clean(), lock, self.s.pre())['verdict'], 'FAIL')

    def test_testbed_mode(self):
        lv = decide.testbed_lock(self.lock, 'CEILING_X')
        self.assertEqual(lv['primary']['candidate_arm'], 'CEILING_X')
        self.assertEqual(lv['primary']['comparator_arms'], [])
        self.assertEqual(lv['primary']['aggregator'], {'type': 'all_worlds'})
        self.assertEqual(self.lock['primary']['candidate_arm'], 'W')  # the lock itself is untouched
        s = reducercheck.Synth(lv)
        c = s.cells(s.flat(9, ['wt1']), worlds=['wt1'], role='technical')
        # cycle 0 runs before learner code: no NPT/SHUF cells exist and the testbed view needs none (SPEC 20.6)
        c['cells'] = [g for g in c['cells'] if g['pool'] != 'OLD' and g['arm'] == 'CEILING_X']
        v = decide.evaluate(c, lv, {'mode': 'testbed', 'expected_worlds': ['wt1'],
                                    'n_items_per_pool': s.pre()['n_items_per_pool']})
        self.assertEqual((v['verdict'], v['shadow_verdict']), ('DEV_NOT_EVIDENCE', 'PASS'))
        self.assertEqual(v['guards'], {'old_drop': 'NOT_MEASURED', 'old_acq': 'NOT_MEASURED'})
        self.assertEqual(decide.verdict_or_shadow(v), 'PASS')

    def test_cells_sha_is_nan_safe_and_stable(self):
        c = self.s.clean()
        a = decide.cells_sha256(c)
        self.assertEqual(a, decide.cells_sha256(json.loads(json.dumps(c))))
        c['cells'][0]['n_exact'] = float('nan')
        self.assertEqual(len(decide.cells_sha256(c)), 64)


class TestCustomReducer(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='mfc_red_'))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def reducer(self, body):
        p = self.tmp / 'reducer.py'
        p.write_text('import json, sys, time\ncells = json.loads(sys.stdin.read())\n' + body + '\n')
        return p

    def test_subprocess_outputs(self):
        lock = lock_for(rule_type='custom')
        s = reducercheck.Synth(lock, str(self.reducer('print("PASS")')))
        v = s.evaluate(s.clean())
        self.assertEqual(v['verdict'], 'PASS', v['reasons'])
        self.assertEqual(v['custom_reducer']['verdict'], 'PASS')
        # a reducer PASS on a candidate below comparator + margin is overruled by the program conditions
        cc = s.chance_count(s.n)
        v = s.evaluate(s.cells(s.flat(cc)))
        self.assertEqual(v['verdict'], 'FAIL')
        s = reducercheck.Synth(lock, str(self.reducer('print("MAYBE")')))
        v = s.evaluate(s.clean())
        self.assertEqual(v['verdict'], 'FAIL')
        self.assertIn('non-enum', ' '.join(v['reasons']))
        s = reducercheck.Synth(lock, str(self.reducer('print(json.dumps({"verdict": "INCOMPLETE"}))')))
        self.assertEqual(s.evaluate(s.clean())['verdict'], 'INCOMPLETE')
        s = reducercheck.Synth(lock, str(self.tmp / 'absent.py'))
        self.assertEqual(s.evaluate(s.clean())['verdict'], 'FAIL')
        rv, why = decide.run_custom_reducer(self.reducer('time.sleep(5)\nprint("PASS")'), {'cells': []}, timeout=1.0)
        self.assertIsNone(rv)
        self.assertIn('timed out', why)
        rv, why = decide.run_custom_reducer(self.reducer('raise SystemExit(3)'), {'cells': []})
        self.assertIsNone(rv)


class TestOC(unittest.TestCase):
    def exact_power(self, n, p_cand, p_comp, need):
        """P(X_cand - X_comp >= need) for independent binomials."""
        pc = [math.comb(n, k) * p_cand ** k * (1 - p_cand) ** (n - k) for k in range(n + 1)]
        pp = [math.comb(n, k) * p_comp ** k * (1 - p_comp) ** (n - k) for k in range(n + 1)]
        return sum(pc[x] * pp[y] for x in range(n + 1) for y in range(n + 1) if x - y >= need)

    def test_oc_recovers_binomial_power(self):
        """One world, all_worlds rule, 40 items, p_comp 0.3, target delta 0.25, margin 0.06, chance 0:
        pass iff (X_cand - X_comp) / 40 > 0.06 iff X_cand - X_comp >= 3."""
        lock = lock_for(aggregator={'type': 'all_worlds'}, margin_abs=0.06, chance=0.0, absolute_floor=0.0,
                        per_world_floor=-1.0, min_worlds=1)
        lock['worlds']['official'] = {'mode': 'listed', 'ids': ['wo1'], 'n': 1}
        res = decide.oc(lock, {'per_world_d': [0.25], 'mean_comp': 0.3, 'n_items': 40, 'between_world_sd': 0.0},
                        sims=2000, seed=0)
        self.assertAlmostEqual(res['p_pass_target'], self.exact_power(40, 0.55, 0.3, 3), delta=0.03)
        self.assertAlmostEqual(res['p_pass_null'], self.exact_power(40, 0.3, 0.3, 3), delta=0.03)
        again = decide.oc(lock, {'per_world_d': [0.25], 'mean_comp': 0.3, 'n_items': 40, 'between_world_sd': 0.0},
                          sims=2000, seed=0)
        self.assertEqual(res, again)
        with self.assertRaises(common.MissingInput):
            decide.oc(lock, {'per_world_d': [0.25], 'mean_comp': 0.3, 'n_items': 40})  # no SD source

    def test_binomial_sampler_mean(self):
        import random
        rng = random.Random(1)
        xs = [decide._binom(rng, 600, 0.7) for _ in range(400)]
        self.assertAlmostEqual(sum(xs) / 400.0, 420.0, delta=3.0)
        self.assertEqual(decide._binom(rng, 10, 0.0), 0)
        self.assertEqual(decide._binom(rng, 10, 1.0), 10)


class TestDec01AndOc01(DeliveryCase):
    def test_declarative_without_verdict_passes(self):
        r = decide.check_dec_01(self.ctx('roster'))
        self.assertEqual(r.status, common.PASS)
        self.assertIn('declarative', r.detail)
        lock = base_lock()
        lock['primary']['rule_type'] = 'votes'
        self.set_lock(lock)
        self.assertEqual(decide.check_dec_01(self.ctx('roster')).status, common.FAIL)

    def test_verdict_without_rows_fails_and_rows_without_verdict_missing(self):
        write(self.d / 'results/official/VERDICT.json', {'verdict': 'PASS'})
        self.assertEqual(decide.check_dec_01(self.ctx('publish')).status, common.FAIL)
        (self.d / 'results/official/VERDICT.json').unlink()
        self.full()
        self.assertEqual(decide.check_dec_01(self.ctx('publish')).status, common.MISSING)

    def test_pre_run_stages_do_not_require_the_verdict(self):
        # a finished delivery gated again at a pre-run stage: VERDICT.json is due only from publish (score writes it)
        self.full()
        for stage in ('roster', 'official'):
            r = decide.check_dec_01(self.ctx(stage))
            self.assertEqual(r.status, common.PASS, '%s: %s' % (stage, r.detail))
            self.assertIn('required at publish', r.detail)
        self.assertEqual(decide.check_dec_01(self.ctx('publish')).status, common.MISSING)

    def custom_setup(self, reducer_body, ud=True, reviewer='codex', author='codex:helper'):
        lock = base_lock()
        lock['primary']['rule_type'] = 'custom'
        lock['primary']['custom_reducer'] = {'path': 'evaluator/reducer.py', 'author_agent': author}
        rp = write(self.d / 'evaluator/reducer.py', 'import json, sys\ncells = json.loads(sys.stdin.read())\n'
                   + reducer_body + '\n')
        lock['endpoint_files_sha256'] = {'evaluator/reducer.py': common.sha256_file(rp)}
        if ud:
            lock['user_decisions'] = ['UD-20261101-1']
            write(self.root / 'PROJECT_DOCUMENTS/KB/USER_DECISIONS.jsonl', jsonl([
                {'id': 'UD-20261101-1', 'utc': '20261101T000000Z', 'verbatim': 'custom rule ok',
                 'scope': ['custom_rule:%s' % NAME]}]))
        self.set_lock(lock)
        write(self.d / 'review/REVIEW.json', {'reviewer_agent': reviewer, 'property_fixture_seed': 'c0ffee'})

    MONOTONE = ('cand = [c for c in cells["cells"] if c["arm"] == "W" and c["pool"] == "HELD"]\n'
                'rate = sum(c["n_exact"] for c in cand) / float(sum(c["n"] for c in cand))\n'
                'print("PASS" if rate > 0.5 else "FAIL")')
    # PASS iff the first world's candidate count is even: raising every world by an odd number of items flips it
    NON_MONOTONE = ('cand = [c for c in cells["cells"] if c["arm"] == "W" and c["pool"] == "HELD"]\n'
                    'w0 = sorted(set(c["world"] for c in cand))[0]\n'
                    'k0 = sum(c["n_exact"] for c in cand if c["world"] == w0)\n'
                    'print("PASS" if k0 % 2 == 0 else "FAIL")')

    def test_custom_rule_requirements(self):
        self.custom_setup(self.MONOTONE)
        r = decide.check_dec_01(self.ctx('roster'))
        self.assertEqual(r.status, common.PASS, r.detail)
        self.custom_setup(self.MONOTONE, ud=False)
        r = decide.check_dec_01(self.ctx('roster'))
        self.assertEqual(r.status, common.FAIL)
        self.assertIn('USER_DECISIONS', r.detail)
        self.custom_setup(self.MONOTONE, reviewer='claude:x', author='claude')
        r = decide.check_dec_01(self.ctx('roster'))
        self.assertEqual(r.status, common.FAIL)
        self.assertIn('executor', r.detail)

    def test_custom_rule_property_fixture_failure(self):
        self.custom_setup(self.NON_MONOTONE)
        r = decide.check_dec_01(self.ctx('roster'))
        self.assertEqual(r.status, common.FAIL, r.detail)
        self.assertIn('property fixtures failed', r.detail)

    def test_oc01_from_dev_rows(self):
        self.full(official=False)
        ctx = self.ctx('official')
        r = decide.check_oc_01(ctx)
        # dev: d = 0.45 and 0.4 (wd1, wd2), 20 counted items, comparator 0.5; 8 official worlds; the t-LCB with
        # sd = 0.0354 clears 0.05 almost always
        self.assertEqual(r.status, common.PASS, r.detail)
        self.assertGreater(r.data['oc']['p_pass_target'], 0.8)
        self.assertLess(r.data['oc']['p_pass_null'], 0.2)


if __name__ == '__main__':
    unittest.main()
