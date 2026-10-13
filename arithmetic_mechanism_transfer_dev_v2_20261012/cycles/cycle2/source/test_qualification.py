"""E0/technical invariants. Scores do not tune or select mechanisms."""
import copy
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from core import MechanismCore, CONFIGS, EOS, unit
from worlds import make_world, OPS, TECH_MIN
from runner import probe
from auditor import audit, recover
from analysis import paired_interval, j_curve, cpu_pair


class Qualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thread_cap=threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.thread_cap.restore_original_limits()

    def test_overflow_does_not_relabel_recent_bytes_as_first_positions(self):
        c=MechanismCore('COMP-LOCAL-FULL',TECH_MIN)
        c.begin_sequence();c.observe_bytes(b'12345678'+b'Z'*4100,learn=False);c.start_reply()
        h=c._features()
        np.testing.assert_array_equal(h[:72],np.zeros(72))
        self.assertGreater(c.context_overflow_events,0)
        self.assertLessEqual(len(c._question),4096)

    def test_hash_exact_reference_parity(self):
        from generic_byte_io import ByteCore,CoreConfig
        c=MechanismCore('HASH',TECH_MIN);base=ByteCore(CoreConfig(address_seed=TECH_MIN))
        for item in (c,base):
            item.begin_sequence();item.observe_bytes(b'opaque\x00\n=',learn=False);item.start_reply()
            item.observe_bytes(b'\xffabc');item.observe_end()
            item.begin_sequence();item.observe_bytes(b'opaque\x00\n=',learn=False);item.start_reply()
        np.testing.assert_array_equal(c._weights,base._weights)
        self.assertEqual(c.generate(16),base.generate(16))

    def test_birth_and_geometry_matched(self):
        cores=[MechanismCore(n,TECH_MIN) for n in CONFIGS[:-1]]
        self.assertEqual(len({c.birth_digest() for c in cores}),1)
        for c in cores:
            self.assertEqual(c.V.shape,(4,96,257))
            self.assertFalse(np.shares_memory(c._weights,cores[-1]._weights) if c!=cores[-1] else False)
            c.begin_sequence();c.observe_bytes(b'12*3=',learn=False);c.start_reply()
            h=c._features();u=c._encode(h)
            self.assertAlmostEqual(np.linalg.norm(u),1.)
            if c.name.startswith('COMP'):
                self.assertLessEqual(np.count_nonzero(u),8)
                self.assertTrue(np.all(u>=0))
            else:
                np.testing.assert_allclose(u,unit((c.A[:,:96]-c.A[:,96:])@h))
        self.assertGreater(np.linalg.norm(cores[0].centers[0]-cores[0].centers[1]),0)

    def test_independent_one_event_formula(self):
        for name in CONFIGS[:-1]:
            c=MechanismCore(name,TECH_MIN)
            c.begin_sequence();c.observe_bytes(b'7-12=',learn=False);c.start_reply()
            x=c._address();u=c._encode(c._features());b=c._bank
            P=c.P[b].copy();V=c.V[b].copy();W=c._weights[x].copy()
            z=.25*W.astype(float).sum(axis=0)+u@V
            q=P@u if name.endswith('FULL') else P*u
            denom=1.+16*(.25**2)+float(u@q)
            e=np.eye(257)[45]-z
            before=c.prediction()
            actual=c.observe(45)
            self.assertEqual(before.event,actual.event)
            np.testing.assert_array_equal(before.probabilities,actual.probabilities)
            np.testing.assert_allclose(c.V[b],V+np.outer(q,e)/denom)
            np.testing.assert_array_equal(c._weights[x],W+(.25*e/denom).astype(np.float32))
            expected=P-np.outer(q,q)/denom if name.endswith('FULL') else P-q*q/denom
            np.testing.assert_allclose(c.P[b],expected,atol=1e-14)

    def test_no_write_and_generation_all_learning_components(self):
        for name in CONFIGS:
            c=MechanismCore(name,TECH_MIN)
            before=c.learned_digest()
            c.begin_sequence();c.observe_bytes(bytes(range(256))+b'\x00\n=',learn=False)
            c.start_reply();c.observe_bytes(b'-49',learn=False);c.observe_end(learn=False)
            self.assertEqual(c.learned_digest(),before)
            c.begin_sequence();c.observe_bytes(b'question',learn=False);c.start_reply()
            g=c.generate(16)
            self.assertEqual(g.status,'TRUNCATED')
            self.assertEqual(c.learned_digest(),before)
            self.assertEqual(c.learning_events,0)

    def test_targets_and_generic_output_not_masked(self):
        c=MechanismCore('COMP-LOCAL-FULL',TECH_MIN)
        for target in range(257):
            c.begin_sequence();c.observe_bytes(b'any\x00\n*=',learn=False);c.start_reply()
            prior=c.learning_events
            c.observe_end() if target==EOS else c.observe(target)
            self.assertEqual(c.learning_events,prior+1)
        # Controlled readout fixture, not a trained-acquisition claim.
        for target in (0,10,32,45,48,255,EOS):
            c=MechanismCore('LINEAR-FIXED-DIAG',TECH_MIN)
            c.begin_sequence();c.start_reply()
            address=c._address()
            c._weights[address,target]=10
            self.assertEqual(c.prediction().event,target)

    def test_local_route_updates_after_observed_eos_only(self):
        for name in ('COMP-LOCAL-FULL','COMP-FIXED-FULL'):
            c=MechanismCore(name,TECH_MIN)
            c.begin_sequence();c.observe_bytes(b'6*7=',learn=False);c.start_reply()
            before=c.centers.copy();bank=c._bank;h=c._route_h.copy()
            c.observe_bytes(b'42')
            np.testing.assert_array_equal(before,c.centers)
            self.assertEqual(c._bank,bank)
            c.observe_end()
            self.assertEqual(c.counts[bank],1)
            if '-LOCAL-' in name:
                np.testing.assert_allclose(c.centers[bank],unit(.95*before[bank]+.05*h))
            else:
                np.testing.assert_array_equal(before,c.centers)

    def test_world_family_leak_and_permutation(self):
        for seed in range(TECH_MIN,TECH_MIN+8):
            w=make_world(seed)
            tr={recover(r) for r in w['training_pool']}
            ho={recover(r) for r in w['held_pool']}
            self.assertTrue(tr.isdisjoint(ho))
            self.assertEqual(len(ho),24)
            self.assertEqual(len(w['records']),512)
            self.assertEqual(len(w['blocks']),24)
            self.assertEqual({op:sum(r['operation']==op for r in w['records']) for op in OPS},
                             {op:128 for op in OPS})
        with self.assertRaises(ValueError): make_world(202610129001,technical=True)
        with self.assertRaises(ValueError): make_world(TECH_MIN,technical=False)

    def test_cold_checkpoint_and_probe_isolation(self):
        for name in CONFIGS:
            c=MechanismCore(name,TECH_MIN)
            c.begin_sequence();c.observe_bytes(b'first\x00\xff',learn=False);c.start_reply()
            c.observe_bytes(b'-12')
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'core.npz';c.save_checkpoint(path)
                code=('from core import MechanismCore;import json;'
                      'c=MechanismCore.load_checkpoint(__import__("sys").argv[1]);'
                      'print(json.dumps({"learned":c.learned_digest(),"p":c.prediction().event}))')
                env=__import__('os').environ.copy();env['OPENBLAS_NUM_THREADS']='1'
                out=subprocess.check_output([sys.executable,'-c',code,str(path)],
                    cwd=Path(__file__).parent,env=env,text=True)
                data=json.loads(out)
                self.assertEqual(data['learned'],c.learned_digest())
                self.assertEqual(data['p'],c.prediction().event)
                restored=MechanismCore.load_checkpoint(path)
                for core in (c,restored):core.observe_end()
                self.assertEqual(c.learned_digest(),restored.learned_digest())
            before=c.learned_digest()
            probe(c,[('+',1,2,3),('-',1,7,-6),('*',7,7,49),('/',49,7,7)])
            self.assertEqual(c.learned_digest(),before)

    def test_statistical_contract_no_zero_width_claim(self):
        self.assertEqual(j_curve([.1]*6),.1)
        self.assertEqual(paired_interval([0.]*48)['status'],'ZERO_VARIANCE_UNRESOLVED')
        self.assertEqual(paired_interval([None]*48)['status'],'UNAVAILABLE')
        result=paired_interval([.1,.2,.3,.4]*12)
        self.assertEqual(result['family_size'],116)
        self.assertLess(result['lower'],result['mean'])


if __name__=='__main__':
    with threadpool_limits(limits=1):
        unittest.main()
