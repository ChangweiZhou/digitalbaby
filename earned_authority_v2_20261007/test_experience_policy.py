import copy
import json
import unittest
from experience_policy import (ExperienceAuthority, Observation, sensory_key,
                               scores_emit, SCALES, MAX_COUNTER)


def obs(shared, private):
    return Observation(tuple(float(x) * SCALES[0] for x in shared),
                       tuple(float(x) * SCALES[1] for x in private))


CONFLICT = obs((0, 1.5, -100, -100), (2, 0, 0, 0))
NEUTRAL = obs((0, 0, 0, 0), (0, 0, 0, 0))


class Tests(unittest.TestCase):
    def test_teacher_cannot_precede_prediction(self):
        with self.assertRaises(RuntimeError):
            ExperienceAuthority().observe(49, learn=True)

    def test_prediction_is_before_feedback(self):
        c = ExperienceAuthority()
        before = c.predict(b'abcd', CONFLICT)
        self.assertEqual(before['EXPERIENCE']['emitted'], 48)
        self.assertFalse(before['EXPERIENCE']['supported'])
        c.observe(49, learn=True)
        self.assertEqual(c.query((b'efgh',), (CONFLICT,))['EXPERIENCE']['emitted'], 49)
        self.assertEqual(before['EXPERIENCE']['emitted'], 48)

    def test_no_write_disables_all_observer_learning(self):
        c = ExperienceAuthority()
        d = c.digest()
        c.predict(b'abcd', CONFLICT)
        c.observe(49, learn=False)
        self.assertEqual(c.digest(), d)
        self.assertEqual(c.cache, [])

    def test_duplicate_prediction_or_teacher_fails(self):
        c = ExperienceAuthority()
        c.predict(b'abcd', CONFLICT)
        with self.assertRaises(RuntimeError):
            c.predict(b'efgh', CONFLICT)
        c.observe(49, learn=True)
        with self.assertRaises(RuntimeError):
            c.observe(49, learn=True)

    def test_query_has_no_label_metadata_api(self):
        c = ExperienceAuthority()
        for k in ('target', 'stage', 'world', 'was_taught', 'correct'):
            with self.assertRaises(TypeError):
                c.query((b'abcd',), (CONFLICT,), **{k: 49})

    def test_queries_never_update_or_cache(self):
        c = ExperienceAuthority()
        d = c.digest()
        for _ in range(10):
            c.query((b'abcd', b'efgh'), (CONFLICT, NEUTRAL))
        self.assertEqual(c.digest(), d)

    def test_both_strata_and_equal_prior(self):
        c = ExperienceAuthority()
        c.predict(b'abcd', CONFLICT)
        c.observe(49, learn=True)
        c.predict(b'abcd', CONFLICT)
        c.observe(48, learn=True)
        self.assertEqual(c.stats, [[1, 1, 0], [1, 0, 1]])
        self.assertEqual(c.query((b'abcd',), (CONFLICT,))['EXPERIENCE']['alpha'], 1)
        self.assertEqual(c.query((b'efgh',), (CONFLICT,))['EXPERIENCE']['alpha'], .5)
        self.assertEqual(c.query((b'efgh',), (CONFLICT,))['GLOBAL_CALIBRATION']['alpha'], 1)

    def test_two_options_one_common_gate_mixed_support(self):
        c = ExperienceAuthority()
        c.predict(b'abcd', CONFLICT)
        c.observe(49, learn=True)
        c.predict(b'abcd', CONFLICT)
        c.observe(48, learn=True)
        row = c.query((b'abcd', b'efgh'), (CONFLICT, NEUTRAL))['EXPERIENCE']
        self.assertFalse(row['supported'])
        self.assertEqual(row['alpha'], .5)
        self.assertEqual(row['combined'], scores_emit((CONFLICT, NEUTRAL), .5)['combined'])

    def test_capture_key_before_outcome(self):
        c = ExperienceAuthority()
        c.predict(sensory_key(b'1+2='), CONFLICT)
        c.observe(ord('3'), learn=True)
        self.assertEqual(c.cache, [b'1+2='])
        self.assertNotIn(b'+2=3', c.cache)

    def test_inherited_key_aliases_and_short_context(self):
        self.assertEqual(sensory_key(b'1 +2='), sensory_key(b'Q1+2='))
        self.assertEqual(sensory_key(b'12+3='), sensory_key(b'22+3='))
        self.assertEqual(sensory_key(b'aaaa\n1+2='), b'1+2=')
        for cue in (b'AB', b'1+2=\n', b'AB\nCD'):
            with self.assertRaises(ValueError):
                sensory_key(cue)

    def test_fifo_fixed_capacity_and_no_refresh(self):
        c = ExperienceAuthority()
        keys = [i.to_bytes(4, 'big') for i in range(65)]
        for key in keys[:64]:
            c.predict(key, NEUTRAL)
            c.observe(48, learn=True)
        c.predict(keys[0], NEUTRAL)
        c.observe(48, learn=True)
        c.predict(keys[64], NEUTRAL)
        r = c.observe(48, learn=True)
        self.assertEqual(r['evicted'], keys[0].hex())
        self.assertEqual(c.cache, keys[1:])
        self.assertEqual(len(c.cache), 64)

    def test_pending_snapshot_resume_and_independent_clone(self):
        c = ExperienceAuthority()
        c.predict(b'abcd', CONFLICT)
        d = ExperienceAuthority.restore(json.loads(json.dumps(c.snapshot())))
        self.assertEqual(c.digest(), d.digest())
        c.observe(49, learn=True)
        d.observe(49, learn=True)
        self.assertEqual(c.digest(), d.digest())
        child = c.clone()
        child.predict(b'efgh', CONFLICT)
        child.observe(48, learn=True)
        self.assertNotEqual(c.digest(), child.digest())

    def test_record_boundary_resumed_equals_uninterrupted(self):
        a, b = ExperienceAuthority(), ExperienceAuthority()
        stream = [(i.to_bytes(4, 'big'), 49 if i % 2 else 48, i % 5 != 0) for i in range(100)]
        for i, (key, target, flag) in enumerate(stream):
            self.assertEqual(a.predict(key, CONFLICT), b.predict(key, CONFLICT))
            a.observe(target, learn=flag)
            b.observe(target, learn=flag)
            if i == 30:
                b = ExperienceAuthority.restore(json.loads(json.dumps(b.snapshot())))
        self.assertEqual(a.digest(), b.digest())

    def test_output_permutation_equivariance_without_ties(self):
        perm = (2, 0, 3, 1)
        a, b = ExperienceAuthority(), ExperienceAuthority()
        swapped = Observation(tuple(CONFLICT.shared[i] for i in perm), tuple(CONFLICT.private[i] for i in perm))
        a.predict(b'abcd', CONFLICT)
        b.predict(b'abcd', swapped)
        a.observe(49, learn=True)
        b.observe(48 + perm.index(1), learn=True)
        self.assertEqual(a.stats, b.stats)
        ra = a.query((b'efgh',), (CONFLICT,))
        rb = b.query((b'efgh',), (swapped,))
        for p in ra:
            self.assertEqual(rb[p]['emitted'] - 48, perm.index(ra[p]['emitted'] - 48))

    def test_no_nonfinite_or_hidden_output(self):
        with self.assertRaises(ValueError):
            Observation((0, 0, 0, float('nan')), (0, 0, 0, 0))
        c = ExperienceAuthority()
        c.predict(b'abcd', CONFLICT)
        for teacher, flag in ((ord('9'), True), (49, 'False'), (True, True)):
            with self.assertRaises(ValueError):
                c.observe(teacher, learn=flag)

    def test_no_confidence_from_single_observation(self):
        c = ExperienceAuthority()
        c.predict(b'abcd', CONFLICT)
        c.observe(49, learn=True)
        r = c.query((b'efgh',), (CONFLICT,))
        self.assertEqual(r['EXPERIENCE']['alpha'], .5)
        self.assertNotIn('confidence', r['EXPERIENCE'])

    def test_finite_input_arithmetic_overflow_rejected(self):
        huge = Observation((1.7e308,) * 4, (1.7e308,) * 4)
        with self.assertRaises(ValueError):
            scores_emit((huge,), 1.0)

    def test_counter_exhaustion_stops(self):
        c = ExperienceAuthority()
        c.stats[0] = [MAX_COUNTER, 0, 0]
        c.predict(b'abcd', CONFLICT)
        with self.assertRaises(OverflowError):
            c.observe(49, learn=True)

    def test_snapshot_tamper_cases_rejected(self):
        c = ExperienceAuthority()
        c.predict(b'abcd', CONFLICT)
        d = c.snapshot()
        cases = []
        for key, value in (('schema', 'bad'), ('cache', ['00000000'] * 65),
                           ('cache', ['00']), ('cache', ['00000000'] * 2),
                           ('stats', [[1, 1, 1], [0, 0, 0]]),
                           ('stats', [[-1, 0, 0], [0, 0, 0]]),
                           ('stats', [[True, 0, 0], [0, 0, 0]])):
            t = copy.deepcopy(d)
            t[key] = value
            cases.append(t)
        for key, value in (('stratum', 1), ('stratum', True), ('error', 57), ('key', '00')):
            t = copy.deepcopy(d)
            t['pending'][key] = value
            cases.append(t)
        t = copy.deepcopy(d)
        t['oracle'] = 49
        cases.append(t)
        for t in cases:
            with self.assertRaises(ValueError):
                ExperienceAuthority.restore(t)


if __name__ == '__main__':
    unittest.main(verbosity=2)
