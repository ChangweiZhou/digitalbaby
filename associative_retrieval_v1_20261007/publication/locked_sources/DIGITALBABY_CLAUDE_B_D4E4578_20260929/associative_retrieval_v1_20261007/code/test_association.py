import copy
import unittest
import numpy as np
from association import Associations, code_ids, validate_ids, CAPACITY

class TestAssociations(unittest.TestCase):
    def test_empty(self):
        a = Associations(5177)
        self.assertEqual(a.query(np.array([1, 2], np.int32)), [])

    def test_self_and_weights(self):
        a = Associations(5177)
        a.observe(np.array([1, 2, 3]), np.array([10, 20, 30]))
        a.observe(np.array([1, 2, 4]), np.array([11, 21, 31]))
        r = a.query(np.array([1, 2, 3]))
        self.assertEqual(r[0]['private_ids'], [10, 20, 30])
        self.assertAlmostEqual(sum(x['weight'] for x in r), 1)
        self.assertEqual(r[0]['score'], 1)

    def test_fifo_and_repeat(self):
        a = Associations(5177)
        for i in range(CAPACITY):
            a.observe(np.array([i+1]), np.array([1000+i]))
        a.observe(np.array([100]), np.array([1000]))
        self.assertEqual(a.insertions, CAPACITY)
        a.observe(np.array([500]), np.array([2000]))
        self.assertEqual(a.private[0, 0], 2000)
        self.assertEqual(a.count, CAPACITY)
        self.assertEqual(a.cursor, 1)
        self.assertEqual(a.mutable_bytes(), 131840)

    def test_tie_oldest(self):
        a = Associations(5177)
        for i in range(6):
            a.observe(np.array([1, 2]), np.array([100+i]))
        self.assertEqual([r['slot'] for r in a.query(np.array([1, 2]))], [0, 1, 2, 3])

    def test_clone_restore(self):
        a = Associations(5177)
        a.observe(np.array([1, 2]), np.array([10, 20]))
        b = Associations.restore(a.snapshot())
        self.assertEqual(a.digest(), b.digest())
        b.observe(np.array([3, 4]), np.array([30, 40]))
        self.assertNotEqual(a.digest(), b.digest())
        c = a.clone(); c.shared[0, 0] = 0
        self.assertNotEqual(a.shared[0, 0], c.shared[0, 0])

    def test_invalid_addresses(self):
        for x in ([2, 1], [1, 1], [-1], [5177], [], [1.1], list(range(257))):
            with self.assertRaises(ValueError):
                validate_ids(np.asarray(x), 5177)
        for x in (np.full(5177, .5), np.full(5177, np.nan), np.ones(3), np.zeros(5177)):
            with self.assertRaises(ValueError):
                code_ids(x, 5177)

    def test_query_readonly(self):
        a = Associations(5177)
        a.observe(np.array([1]), np.array([2]))
        before = a.digest()
        a.query(np.array([1]))
        self.assertEqual(a.digest(), before)

    def test_no_label_api(self):
        a = Associations(5177)
        with self.assertRaises(TypeError):
            a.observe(np.array([1]), np.array([2]), answer=49)

    def test_snapshot_tamper(self):
        a = Associations(5177); a.observe(np.array([1, 2]), np.array([10, 20]))
        changes = [('cursor', 5), ('count', 65), ('insertions', 0), ('observations', -1)]
        for k, v in changes:
            d = a.snapshot(); d[k] = v
            with self.assertRaises(ValueError): Associations.restore(d)
        d = a.snapshot(); d['shared'][0, 2] = 44
        with self.assertRaises(ValueError): Associations.restore(d)

if __name__ == '__main__': unittest.main(verbosity=2)

