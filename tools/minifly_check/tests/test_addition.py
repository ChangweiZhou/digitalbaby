"""Unit tests for addition.py (WP3): truth, canonical orbits, strata S0-S4, witnesses. Hand-computed cases."""
import sys
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

import addition  # noqa: E402
import common  # noqa: E402

FMT = {'kind': 'addition', 'sep_hex': '2b', 'terminator_hex': '3d', 'eos_hex': '0a', 'gen_cap_bytes': 4,
       'normalize_leading_zeros': True}


def ph(text):
    return text.encode('ascii').hex()


class TestTruth(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(addition.parse('332b343d', FMT), ('3', '4'))
        self.assertEqual(addition.parse(ph('007+3='), FMT), ('007', '3'))
        for bad in (ph('3+4'), ph('3+4+5='), ph('3+x='), ph('+4='), ph('3+='), 'zz', None, ph('3 +4=')):
            self.assertIsNone(addition.parse(bad, FMT), bad)
        self.assertIsNone(addition.parse('332b343d', dict(FMT, kind='other')))
        other_sep = dict(FMT, sep_hex='2c', terminator_hex='3a')  # '3,4:'
        self.assertEqual(addition.parse(ph('3,4:'), other_sep), ('3', '4'))
        with self.assertRaises(common.InvalidInput):
            addition.parse('332b343d', dict(FMT, sep_hex='31'))  # a digit separator is ambiguous

    def test_expected(self):
        self.assertEqual(addition.expected('332b343d', FMT), '37')
        self.assertEqual(addition.expected(ph('99+1='), FMT), ph('100'))
        self.assertEqual(addition.expected(ph('007+3='), FMT), ph('10'))
        self.assertEqual(addition.expected(ph('0+0='), FMT), ph('0'))
        self.assertIsNone(addition.expected(ph('3-4='), FMT))
        with self.assertRaises(common.InvalidInput):
            addition.expected('332b343d', dict(FMT, normalize_leading_zeros=False))

    def test_canonical_and_carry_and_columns(self):
        self.assertEqual(addition.canonical('7', '03'), ('3', '7'))
        self.assertEqual(addition.canonical('10', '9'), ('9', '10'))
        self.assertEqual(addition.canonical('0', '000'), ('0', '0'))
        self.assertFalse(addition.has_carry('12', '34'))
        self.assertTrue(addition.has_carry('15', '7'))
        self.assertTrue(addition.has_carry('5', '5'))
        self.assertEqual(addition.columns('15', '7'), ((5, 7, 0), (1, None, 1)))
        self.assertEqual(addition.columns('95', '5'), ((5, 5, 0), (9, None, 1), (None, None, 1)))
        self.assertEqual(addition.columns('012', '3'), ((2, 3, 0), (1, None, 0)))


class TestStrata(unittest.TestCase):
    def test_strata_definitions(self):
        T = addition.Taught([('1', '5'), ('4', '2'), ('3', '2'), ('5', '1'), ('2', '6'), ('2', '2')])
        # S0: an operand is 0 (V6-like zero-operand items), never counted
        self.assertEqual(addition.stratum(('0', '5'), T), 'S0')
        self.assertEqual(addition.stratum(('4', '0'), T), 'S0')
        self.assertEqual(addition.stratum(('00', '7'), T), 'S0')
        # S1: the canonical orbit is taught (commuted twin of the taught 3+2)
        self.assertEqual(addition.stratum(('2', '3'), T), 'S1')
        self.assertEqual(addition.stratum(('3', '2'), T), 'S1')
        # S2: 1 is a taught first operand (1+5), 2 a taught second operand (4+2), 1+2 has no carry
        self.assertEqual(addition.stratum(('1', '2'), T), 'S2')
        # S4: 7 never appears as a first operand
        self.assertEqual(addition.stratum(('7', '2'), T), 'S4')
        # S4: operands seen in position (5 in 5+1, 6 in 2+6) but 11 is longer than every taught answer (max 8)
        self.assertEqual(addition.stratum(('5', '6'), T), 'S4')
        # S3: with a two-digit taught answer, 5+6 has seen operands and a carry
        T2 = addition.Taught(list(T.pairs) + [('9', '9')])
        self.assertEqual(addition.stratum(('5', '6'), T2), 'S3')
        self.assertIsNone(addition.stratum({'prompt_hex': 'zz'}, T))
        self.assertEqual(addition.stratum({'prompt_hex': ph('1+2=')}, T), 'S2')

    def test_from_rows_witness_filter(self):
        rows = [{'prompt_hex': ph('1+2='), 'answer_hex': ph('3'), 'feedback_bearing': True, 'presentations': 4},
                {'prompt_hex': ph('3+4='), 'answer_hex': ph('7'), 'feedback_bearing': False, 'presentations': 4},
                {'prompt_hex': ph('3+2='), 'answer_hex': ph('5'), 'feedback_bearing': True, 'presentations': 0}]
        T = addition.Taught.from_rows(rows, FMT)
        self.assertEqual(T.pairs, {('1', '2'), ('3', '4'), ('3', '2')})
        self.assertEqual(T.wpairs, {('1', '2')})
        self.assertEqual(addition.stratum(('4', '3'), T), 'S1')  # presentation-only rows still make S1
        self.assertFalse(addition.determined(('1', '4'), T, 'pair_rectangle'))  # (3,4),(3,2) are not units
        with self.assertRaises(common.InvalidInput):
            addition.Taught.from_rows([{'prompt_hex': ph('1-2='), 'feedback_bearing': True}], FMT)


class TestPairRectangle(unittest.TestCase):
    def test_all_twenty_held_items_determined(self):
        """Taught = every (x, y) in 1..9 x 1..9 with x >= 6 or y >= 6. Held = (a, b), a in 1..4, b in 1..5 (20 items):
        (a, 6), (6, b) and (6, 6) are all taught, so every held item is determined (rectangle a'=6, b'=6)."""
        taught = [(str(x), str(y)) for x in range(1, 10) for y in range(1, 10) if x >= 6 or y >= 6]
        T = addition.Taught(taught)
        held = [(str(a), str(b)) for a in range(1, 5) for b in range(1, 6)]
        self.assertEqual(len(held), 20)
        self.assertTrue(all(addition.determined(h, T, 'pair_rectangle') for h in held))
        self.assertEqual(addition.support(('1', '1'), T, 'pair_rectangle'),
                         frozenset({('1', '6'), ('6', '1'), ('6', '6')}))
        # no held item is S0 or S1; all are interpolation without carry
        self.assertEqual(set(addition.stratum(h, T) for h in held), {'S2'})

    def test_negative_and_completion(self):
        T = addition.Taught([('1', '2'), ('3', '4')])
        self.assertFalse(addition.determined(('1', '4'), T, 'pair_rectangle'))  # (3, 2) missing
        self.assertIsNone(addition.support(('1', '4'), T, 'pair_rectangle'))
        T = addition.Taught([('1', '2'), ('3', '4'), ('3', '2')])
        self.assertTrue(addition.determined(('1', '4'), T, 'pair_rectangle'))
        self.assertEqual(addition.support(('1', '4'), T, 'pair_rectangle'),
                         frozenset({('1', '2'), ('3', '4'), ('3', '2')}))
        # a' must differ from a and b' from b: the item itself plus its row and column does not count
        T = addition.Taught([('1', '4'), ('1', '2'), ('3', '4')])
        self.assertFalse(addition.determined(('1', '4'), T, 'pair_rectangle'))

    def test_commutative(self):
        T = addition.Taught([('2', '1'), ('4', '3'), ('3', '2')])
        self.assertFalse(addition.determined(('1', '4'), T, 'pair_rectangle', commutative=False))
        self.assertTrue(addition.determined(('1', '4'), T, 'pair_rectangle', commutative=True))
        self.assertEqual(addition.support(('1', '4'), T, 'pair_rectangle', commutative=True),
                         frozenset({('2', '1'), ('4', '3'), ('3', '2')}))


class TestColumnClosure(unittest.TestCase):
    def test_column_closure_examples(self):
        # 21+13 -> columns (1,3,0), (2,1,0); 34+42 -> (4,2,0), (3,4,0)
        T = addition.Taught([('21', '13'), ('34', '42')])
        self.assertTrue(addition.determined(('24', '12'), T, 'column_closure'))      # (4,2,0), (2,1,0)
        self.assertEqual(addition.support(('24', '12'), T, 'column_closure'),
                         frozenset({('21', '13'), ('34', '42')}))
        self.assertFalse(addition.determined(('31', '44'), T, 'column_closure'))     # (1,4,0) never taught
        # carry: 18+25 -> (8,5,0), (1,2,1); held 28+15 needs (8,5,0) and (2,1,1)
        T = addition.Taught([('18', '25')])
        self.assertFalse(addition.determined(('28', '15'), T, 'column_closure'))
        T = addition.Taught([('18', '25'), ('25', '16')])                             # 25+16: (5,6,0), (2,1,1)
        self.assertTrue(addition.determined(('28', '15'), T, 'column_closure'))
        self.assertEqual(addition.support(('28', '15'), T, 'column_closure'), frozenset({('18', '25'), ('25', '16')}))
        # a missing digit is its own symbol and orientation is as taught: 5+12 needs (5,2,0) and (None,1,0)
        T = addition.Taught([('15', '2')])                                            # (5,2,0), (1,None,0)
        self.assertFalse(addition.determined(('5', '12'), T, 'column_closure'))
        T = addition.Taught([('15', '2'), ('3', '14')])                               # + (3,4,0), (None,1,0)
        self.assertTrue(addition.determined(('5', '12'), T, 'column_closure'))
        # carry-out column (None, None, 1): 95+5 needs a taught carry-out
        T = addition.Taught([('15', '5'), ('9', '1')])                                # (5,5,0),(1,None,1) / (9,1,0),(None,None,1)
        self.assertFalse(addition.determined(('95', '5'), T, 'column_closure'))       # (9,None,1) missing
        T = addition.Taught([('15', '5'), ('91', '9')])                               # 91+9: (1,9,0),(9,None,1),(None,None,1)
        self.assertTrue(addition.determined(('95', '5'), T, 'column_closure'))

    def test_unknown_witness(self):
        with self.assertRaises(common.InvalidInput):
            addition.determined(('1', '2'), addition.Taught([]), 'magic')


if __name__ == '__main__':
    unittest.main()
