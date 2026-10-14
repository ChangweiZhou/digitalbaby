"""First-cycle hostile test: query scope, legality and exact old readout parity."""
import json
import math
from pathlib import Path
import random
import unittest
import numpy as np
from authority_policy import Observation, SCALES, authority, emit


def initial_per_option_rule(observations):
    values = []
    for o in observations:
        s = np.asarray(o.shared) / SCALES[0]
        p = np.asarray(o.private) / SCALES[1]
        sw, pw = int(np.argmax(s)), int(np.argmax(p))
        sm = float(np.sort(s)[-1]-np.sort(s)[-2]); pm = float(np.sort(p)[-1]-np.sort(p)[-2])
        a = .5 if sw != pw and sm > pm and sm > 0 else 1.
        u = s + a * p
        values.append(float(u[1] - u[0]))
    return int(values[0] < values[1])


class Cycle1(unittest.TestCase):
    def test_rejected_initial_rule_reads_irrelevant_binary_channels(self):
        rng = random.Random(20261007)
        witness = None
        for _ in range(10000):
            a = [Observation(tuple(rng.uniform(-3, 3) for _ in range(4)),
                             tuple(rng.uniform(-3, 3) for _ in range(4))) for _ in range(2)]
            b = [Observation(o.shared[:2] + (100.0, -100.0),
                             o.private[:2] + (-100.0, 100.0)) for o in a]
            if initial_per_option_rule(a) != initial_per_option_rule(b):
                witness = a, b
                break
        self.assertIsNotNone(witness)
        a, b = witness
        for policy in ('ERROR', 'Q_HALF', 'Q_EARNED'):
            self.assertEqual(emit(a, policy)['emitted'], emit(b, policy)['emitted'])

    def test_strict_gate_and_ties(self):
        self.assertEqual(authority((0, 1.5), (2, 0))['alpha'], 0.5)
        self.assertEqual(authority((0, 1), (1, 0))['alpha'], 1.0)
        self.assertEqual(authority((0, 1), (2, 0))['alpha'], 1.0)
        self.assertEqual(authority((0, 2), (1, 0))['alpha'], 1.0)
        self.assertEqual(authority((1, 1), (1, 0))['alpha'], 1.0)
        self.assertEqual(authority((0, 2), (0, 1))['alpha'], 1.0)

    def test_no_label_or_world_api(self):
        o = Observation((0., 1., -3., -4.), (1., 0., -3., -4.))
        for kw in ('target', 'world', 'task', 'stage', 'was_taught', 'row', 'outcome'):
            with self.assertRaises(TypeError):
                emit([o], 'Q_EARNED', **{kw: 1})

    def test_nonfinite_and_wrong_roster_rejected(self):
        for bad in (math.nan, math.inf, -math.inf, True):
            with self.assertRaises(ValueError):
                Observation((bad, 0., 0., 0.), (0., 0., 0., 0.))
        with self.assertRaises(ValueError):
            emit([], 'ERROR')
        with self.assertRaises(TypeError):
            emit([{'target': 0}], 'ERROR')

    def test_exact_frozen_baselines(self):
        rng = random.Random(91)
        for n in (1, 2):
            for _ in range(200):
                obs = [Observation(tuple(rng.gauss(0, 1) for _ in range(4)),
                                   tuple(rng.gauss(0, 1) for _ in range(4))) for _ in range(n)]
                for name, alpha in [('ERROR', 1.), ('Q_HALF', .5)]:
                    us = [np.asarray(o.shared)/SCALES[0] + alpha*np.asarray(o.private)/SCALES[1] for o in obs]
                    values = [u[1]-u[0] for u in us] if n == 2 else us[0]
                    expected = b'LR'[int(np.argmax(values))] if n == 2 else b'0123'[int(np.argmax(values))]
                    result = emit(obs, name)
                    self.assertEqual(expected, result['emitted'])
                    self.assertTrue(np.array_equal(result['combined'], values))

    def test_pair_swap_unique_decision(self):
        a = Observation((0., 2., -3., -4.), (0., .5, -3., -4.))
        b = Observation((2., 0., -3., -4.), (.5, 0., -3., -4.))
        for policy in ('ERROR', 'Q_HALF', 'Q_EARNED'):
            left, right = emit([a, b], policy), emit([b, a], policy)
            self.assertEqual(left['emitted'], ord('L'))
            self.assertEqual(right['emitted'], ord('R'))
            self.assertEqual(left['alpha'], right['alpha'])

    def test_bank_offset_invariance_non_tied(self):
        obs = [Observation((0., 2., -3., -4.), (2., 0., -3., -4.))]
        shifted = [Observation(tuple(x+10 for x in obs[0].shared), tuple(x-11 for x in obs[0].private))]
        self.assertEqual(emit(obs, 'Q_EARNED')['emitted'], emit(shifted, 'Q_EARNED')['emitted'])


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Cycle1)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    out = {'cycle': 1, 'tests_run': result.testsRun, 'failures': len(result.failures),
           'errors': len(result.errors), 'verdict': 'PASS' if result.wasSuccessful() else 'FAIL',
           'revision': 'query-level gate; original per-option witness preserved; mechanically corrected margin window; exact source arithmetic retained',
           'new_science_trajectories': 0}
    dest = Path(__file__).parent/'cycles/cycle1'; dest.mkdir(parents=True, exist_ok=True)
    (dest/'TEST_RESULT.json').write_text(json.dumps(out, indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
