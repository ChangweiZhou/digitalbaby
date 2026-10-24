"""Unit tests for reducercheck.py (WP3): RED-01..11 on the lock's own configuration, property fixtures, and the
registration of every WP3 check id under its RULES.json module."""
import sys
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
from test_scorer import DeliveryCase, base_lock  # noqa: E402

RED_IDS = ['RED-%02d' % i for i in range(1, 12)]
WP3_IDS = ['MET-01', 'MET-02', 'LAD-01', 'LAD-02', 'LAD-03', 'LAD-04', 'STR-01', 'DET-01', 'INERT-01', 'UDI-01',
           'OC-01', 'DEC-01'] + RED_IDS


def run(ctx, rid):
    fn = common.CHECKS[rid]
    out = fn(ctx)
    return out if isinstance(out, common.Result) else out[0]


class TestRegistration(unittest.TestCase):
    def test_every_wp3_id_resolves_to_its_rules_module(self):
        idx = common.rule_index(common.load_rules())
        for rid in WP3_IDS:
            fn, err = common.resolve_check(idx[rid])
            self.assertIsNotNone(fn, '%s: %s' % (rid, err))


class TestBinomialQuantiles(unittest.TestCase):
    def test_hand_computed(self):
        """Binomial(10, 0.9): CDF(7)=0.0702, CDF(8)=0.2639, CDF(9)=0.6513; quantiles (i+0.5)/8 =
        0.0625 0.1875 0.3125 0.4375 0.5625 0.6875 0.8125 0.9375 -> 7 8 9 9 9 10 10 10."""
        self.assertEqual(reducercheck.binom_quantiles(10, 0.9, 8), [7, 8, 9, 9, 9, 10, 10, 10])
        self.assertEqual(reducercheck.binom_quantiles(10, 0.0, 3), [0, 0, 0])
        self.assertEqual(reducercheck.binom_quantiles(10, 1.0, 2), [10, 10])


class TestRedOnLocks(DeliveryCase):
    def statuses(self, stage='roster'):
        ctx = self.ctx(stage)
        return {rid: run(ctx, rid) for rid in RED_IDS}

    def test_all_red_pass_on_a_clean_lock(self):
        res = self.statuses()
        for rid, r in res.items():
            self.assertEqual(r.status, common.PASS, '%s: %s' % (rid, r.detail))

    def test_red_02_03_cover_implied_comparators(self):
        """SPEC 20.6: RED-02 tries the candidate equal to every decision-comparator arm and RED-03 drops each of
        them, including the declared shuf arm that comparator_arms leaves out."""
        ctx = self.ctx('roster')
        self.assertEqual(ctx.lock['primary']['comparator_arms'], ['NPT'])
        r2, r3 = run(ctx, 'RED-02'), run(ctx, 'RED-03')
        self.assertEqual((r2.status, r3.status), (common.PASS, common.PASS), (r2.detail, r3.detail))
        self.assertIn('candidate == arm SHUF', r2.detail)
        self.assertIn('comparator SHUF missing', r3.detail)

    def test_all_red_pass_for_other_aggregators(self):
        for agg in ({'type': 'k_of_n', 'k': 6}, {'type': 'all_worlds'}):
            lock = base_lock()
            lock['primary']['aggregator'] = agg
            self.set_lock(lock)
            for rid, r in self.statuses().items():
                self.assertEqual(r.status, common.PASS, '%s %s: %s' % (agg, rid, r.detail))

    def test_red01_unreachable_rules(self):
        for over in ({'margin_abs': 0.9}, {'absolute_floor': 0.95}, {'chance': 0.9}):
            lock = base_lock()
            lock['primary'].update(over)
            self.set_lock(lock)
            r = run(self.ctx('roster'), 'RED-01')
            self.assertEqual(r.status, common.FAIL, over)
            self.assertIn('unreachable', r.detail)
        # too few planned official worlds for min_worlds (C05c-like: the rule cannot be reached as planned)
        lock = base_lock()
        lock['worlds']['official'] = {'mode': 'listed', 'ids': ['wo1', 'wo2', 'wo3'], 'n': 3}
        self.set_lock(lock)
        self.assertEqual(run(self.ctx('roster'), 'RED-01').status, common.FAIL)

    def test_red08_fails_when_s0_counts(self):
        lock = base_lock()
        lock['primary']['strata_counted'] = ['S0', 'S2', 'S3']
        self.set_lock(lock)
        r = run(self.ctx('roster'), 'RED-08')
        self.assertEqual(r.status, common.FAIL)
        self.assertIn('the lock counts', r.detail)

    def test_not_applicable_conditions(self):
        lock = base_lock(task={'id': 'recall', 'family': 'non_arithmetic',
                               'format': {'kind': 'other', 'eos_hex': '0a', 'gen_cap_bytes': 4}}, units=None,
                         guards=[], split={'taught_manifest': 'manifests/TAUGHT'})
        lock['primary']['n_items_per_pool'] = {'HELD': 20}
        self.set_lock(lock)
        res = self.statuses()
        self.assertEqual(res['RED-08'].status, common.NOT_APPLICABLE)
        self.assertEqual(res['RED-09'].status, common.NOT_APPLICABLE)
        self.assertEqual(res['RED-11'].status, common.NOT_APPLICABLE)
        for rid in ('RED-01', 'RED-02', 'RED-03', 'RED-04', 'RED-05', 'RED-06', 'RED-07', 'RED-10'):
            self.assertEqual(res[rid].status, common.PASS, '%s: %s' % (rid, res[rid].detail))
        # the RULES entries declare these conditions, so run_check keeps NOT_APPLICABLE
        idx = common.rule_index(common.load_rules())
        for rid in ('RED-08', 'RED-09', 'RED-11'):
            self.assertEqual(common.run_check(self.ctx('roster'), idx[rid])[0].status, common.NOT_APPLICABLE)

    def test_red09_red11_need_the_lock_guard_to_reject(self):
        # S09: an OLD pool without guards; decide rejects everything, which does not show that a guard works
        lock = base_lock()
        lock['guards'] = []
        self.set_lock(lock)
        res = self.statuses()
        for rid in ('RED-09', 'RED-11'):
            self.assertEqual(res[rid].status, common.FAIL, '%s: %s' % (rid, res[rid].detail))
            self.assertIn('no %s guard' % ('old_drop' if rid == 'RED-09' else 'old_acq'), res[rid].detail)
        # only the old_drop guard: RED-09 can pass on its guard, RED-11 cannot
        lock = base_lock()
        lock['guards'] = [g for g in lock['guards'] if g['kind'] == 'old_drop']
        self.set_lock(lock)
        res = self.statuses()
        self.assertEqual(res['RED-11'].status, common.FAIL, res['RED-11'].detail)
        self.assertIn('guard', res['RED-09'].detail)

    def test_invalid_lock_fails_every_red(self):
        lock = base_lock()
        lock['primary']['aggregator'] = {'type': 'paired_lcb'}  # alpha missing
        self.set_lock(lock)
        idx = common.rule_index(common.load_rules())
        for rid in RED_IDS:
            r = common.run_check(self.ctx('roster'), idx[rid])[0]
            self.assertEqual(r.status, common.FAIL, rid)

    def test_red_cases_against_a_permissive_decide(self):
        """If decide ignored the ladder, RED-10 and RED-02 must catch it (the qualification is not vacuous)."""
        orig = decide.ladder_best
        try:
            decide.ladder_best = lambda cells, w, pool, ck: ('CONST', 0.0)
            res = self.statuses()
            self.assertEqual(res['RED-10'].status, common.FAIL)
            self.assertEqual(res['RED-02'].status, common.FAIL)
        finally:
            decide.ladder_best = orig


class TestPropertyFixtures(unittest.TestCase):
    def test_deterministic_and_monotone_under_declarative_decide(self):
        lock = base_lock()
        a = reducercheck.property_fixtures(lock, 'c0ffee', n=50)
        b = reducercheck.property_fixtures(lock, 'c0ffee', n=50)
        c = reducercheck.property_fixtures(lock, 'beef', n=50)
        self.assertEqual(len(a), 50)
        self.assertEqual(decide.cells_sha256([f['low'] for f in a]), decide.cells_sha256([f['low'] for f in b]))
        self.assertNotEqual(decide.cells_sha256([f['low'] for f in a]), decide.cells_sha256([f['low'] for f in c]))
        verdicts = []
        s = reducercheck.Synth(lock)
        for fx in a:
            lo = s.evaluate(fx['low'])['verdict']
            hi = s.evaluate(fx['high'])['verdict']
            verdicts.append(lo)
            self.assertIn(lo, decide.VERDICTS)
            if lo == 'PASS':
                self.assertEqual(hi, 'PASS', fx['id'])
        # effects are drawn near the margin: both outcomes occur
        self.assertIn('PASS', verdicts)
        self.assertIn('FAIL', verdicts)


if __name__ == '__main__':
    unittest.main()
