"""Technical numerical/interface tests; no scientific selection or E3 claim."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from core import AdapterCore, build_core, CONFIGS, HEAD_RATE, ADAPTER_RATE, STEP_CAP, EOS
from runner import probe, working_state
from worlds import TECH_MIN, make_world


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cap=threadpool_limits(limits=1)
    @classmethod
    def tearDownClass(cls):cls.cap.restore_original_limits()

    def reply(self,name='LEARN-SHARED'):
        c=build_core(name,TECH_MIN);c.begin_sequence();c.observe_bytes(b'7-12=',learn=False);c.start_reply();return c

    def test_matched_birth_and_sensory_input(self):
        cs=[self.reply(n) for n in CONFIGS]
        self.assertEqual(len({c.birth_digest() for c in cs[:-1]}),1)
        for c in cs[:-1]:
            np.testing.assert_array_equal(c._features(),cs[-1]._features())
            self.assertFalse(np.shares_memory(c.V,cs[-1].V))
        self.assertFalse(np.shares_memory(cs[0].A1,cs[1].A1))

    def test_shared_has_no_hash_allocation_or_address_access(self):
        for n in ('LEARN-SHARED','FROZEN-SHARED'):
            c=self.reply(n);self.assertFalse(hasattr(c,'_weights'))
            c.prediction();c.observe(45);c.observe_end();self.assertFalse(c.storage_accounting()['hash_allocated'])
            with self.assertRaises(RuntimeError):c._address()

    def test_full_ce_gradient_finite_difference(self):
        for n in ('LEARN-SHARED','LEARN-HASH'):
            c=self.reply(n);rng=np.random.default_rng(10)
            c.V[:]=rng.normal(0,.02,c.V.shape);c.b1[:]=rng.normal(0,.01,96);c.b2[:]=rng.normal(0,.01,96)
            state,p=c._forward();dirs=c._directions(state,p,255)
            for key in ('A1','A2','b1','b2','V','c'):
                a=getattr(c,key)
                for flat in (0,a.size//3,a.size//2,a.size-1):
                    idx=np.unravel_index(flat,a.shape);old=a[idx];step=1e-6
                    a[idx]=old+step;lp=-np.log(c._forward()[1][255])
                    a[idx]=old-step;lm=-np.log(c._forward()[1][255]);a[idx]=old
                    numeric=-(lp-lm)/(2*step)
                    self.assertAlmostEqual(numeric,float(dirs[key][idx]),delta=2e-7)

    def test_preupdate_formula_and_first_zero_internal_gradient(self):
        for n in CONFIGS[:-1]:
            c=self.reply(n)
            before={k:a.copy() for k,a in c.arrays().items()}
            oldp=c.prediction();event=c.observe(45)
            np.testing.assert_array_equal(event.probabilities,oldp.probabilities)
            for k in ('A1','A2','b1','b2'):np.testing.assert_array_equal(getattr(c,k),before[k])
            state,p=c._forward();directions=c._directions(state,p,EOS)
            before={k:a.copy() for k,a in c.arrays().items()};address=c._address() if c.hash_enabled else None
            e=-p.copy();e[EOS]+=1
            c.observe_end()
            np.testing.assert_allclose(c.V,before['V']+HEAD_RATE*directions['V'],rtol=0,atol=1e-15)
            np.testing.assert_allclose(c.c,before['c']+HEAD_RATE*e,rtol=0,atol=1e-15)
            for layer in (1,2):
                da=ADAPTER_RATE*directions[f'A{layer}'];db=ADAPTER_RATE*directions[f'b{layer}']
                norm=np.sqrt(np.sum(da*da)+db@db);scale=min(1.,STEP_CAP/norm) if norm else 1.
                for key,delta in [(f'A{layer}',da),(f'b{layer}',db)]:
                    expected=before[key]+scale*delta if c.internal_learning else before[key]
                    np.testing.assert_array_equal(getattr(c,key),expected)
            if c.hash_enabled:
                np.testing.assert_array_equal(c._weights[address],before['W'][address]+(.6*.25*e).astype(np.float32))

    def test_step_cap_and_nonzero_internal_update(self):
        c=self.reply();rng=np.random.default_rng(12);c.V[:]=rng.normal(0,100,c.V.shape)
        old={k:getattr(c,k).copy() for k in ('A1','A2','b1','b2')}
        c.observe(255)
        for i in (1,2):
            norm=np.sqrt(np.sum((c.arrays()[f'A{i}']-old[f'A{i}'])**2)+np.sum((c.arrays()[f'b{i}']-old[f'b{i}'])**2))
            self.assertGreater(norm,0);self.assertLessEqual(norm,.10000000001)

    def test_no_write_and_generated_guesses_preserve_all_state(self):
        for n in CONFIGS:
            c=build_core(n,TECH_MIN);before=c.learned_digest()
            c.begin_sequence();c.observe_bytes(bytes(range(256))+b'\x00\n=',learn=False);c.start_reply()
            c.observe_bytes(b'-49',learn=False);c.observe_end(learn=False)
            self.assertEqual(before,c.learned_digest())
            c.begin_sequence();c.observe_bytes(b'anything',learn=False);c.start_reply();c.generate(16)
            self.assertEqual(before,c.learned_digest());self.assertEqual(c.learning_events,0)

    def test_every_category_can_be_observed_and_selected(self):
        c=self.reply()
        for target in range(257):
            c.begin_sequence();c.observe_bytes(b'\x00\n=',learn=False);c.start_reply()
            old=c.learning_events
            c.observe_end() if target==EOS else c.observe(target)
            self.assertEqual(c.learning_events,old+1)
        for target in range(257):
            c.begin_sequence();c.start_reply();c.V.fill(0);c.c.fill(0);c.c[target]=20
            self.assertEqual(c.prediction().event,target)

    def test_own_negative_multibyte_and_binary_sequence_output(self):
        # Directly authored readout fixture checks generation, not acquisition.
        for payload in (b'-49',b'\x00\n\xff',b''):
            c=self.reply('FROZEN-SHARED');states=[]
            for event in list(payload)+[EOS]:
                states.append(c._forward()[0][4]);c._append(event)
            targets=np.zeros((len(states),257))
            for i,event in enumerate(list(payload)+[EOS]):targets[i,event]=20
            c.V[:]=np.linalg.lstsq(np.array(states),targets,rcond=None)[0]
            c.begin_sequence();c.observe_bytes(b'7-12=',learn=False);c.start_reply()
            before=c.learned_digest();result=c.generate(16)
            self.assertEqual(result.payload,payload);self.assertEqual(result.status,'ENDED')
            self.assertEqual(before,c.learned_digest())
        c=self.reply();c.c[255]=20;result=c.generate(2)
        self.assertEqual(result.payload,b'\xff\xff');self.assertEqual(result.status,'TRUNCATED')

    def test_payload_and_learning_validation(self):
        c=self.reply()
        for byte in (True,-1,256,'x'):
            with self.assertRaises((TypeError,ValueError)):c.observe(byte)
        with self.assertRaises(TypeError):c.observe(0,learn=1)
        c.begin_sequence()
        with self.assertRaises(RuntimeError):c.observe(0,learn=True)

    def test_chunk_stream_and_window_overflow(self):
        a,b=[self.reply('LEARN-SHARED') for _ in range(2)]
        data=bytes(range(256))+b'\x00\n=';a.observe_bytes(data)
        for chunk in (data[:3],data[3:200],data[200:]):b.observe_bytes(chunk)
        self.assertEqual(a.learned_digest(),b.learned_digest());self.assertEqual(a.transient_digest(),b.transient_digest())
        c=build_core('FROZEN-SHARED',TECH_MIN);c.begin_sequence();c.observe_bytes(b'12345678'+b'Z'*4100,learn=False);c.start_reply()
        np.testing.assert_array_equal(c._features()[:72],np.zeros(72));self.assertGreater(c.context_overflow_events,0)

    def test_cold_checkpoint_and_readonly_probe(self):
        for n in CONFIGS[:-1]:
            c=self.reply(n);c.observe_bytes(b'-12')
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'state.npz';c.save_checkpoint(path)
                out=subprocess.check_output([sys.executable,'-c',
                    'from core import AdapterCore;import sys,json;c=AdapterCore.load_checkpoint(sys.argv[1]);print(json.dumps([c.learned_digest(),c.transient_digest(),c.prediction().event]))',str(path)],cwd=Path(__file__).parent,text=True)
                self.assertEqual(json.loads(out),[c.learned_digest(),c.transient_digest(),c.prediction().event])
                restored=AdapterCore.load_checkpoint(path)
                for x in (c,restored):x.observe_end()
                self.assertEqual(c.learned_digest(),restored.learned_digest())
            before=c.learned_digest();trans=c.transient_digest()
            probe(c,[('+',1,2,3),('-',1,7,-6),('*',7,7,49),('/',49,7,7)])
            self.assertEqual(before,c.learned_digest());self.assertEqual(trans,c.transient_digest())

    def test_legacy_code_exact_and_context_protocol(self):
        import hashlib
        root=Path(__file__).parent.parent
        for local,source in [('vendor/legacy_core.py','arithmetic_mechanism_transfer_dev_v2_20261012/core.py'),
                             ('vendor/generic_byte_io/core.py','generic_byte_io_v1_20261012/generic_byte_io/core.py')]:
            self.assertEqual(hashlib.sha256((Path(__file__).parent/local).read_bytes()).hexdigest(),hashlib.sha256((root/source).read_bytes()).hexdigest())
        c=build_core('LEGACY-B',TECH_MIN)
        c.begin_sequence();c.observe_bytes(b'\x00\n=',learn=False)
        self.assertEqual(c._question,bytearray(b'\x00\n='))

    def test_world_namespaces_and_counts(self):
        for world in range(TECH_MIN,TECH_MIN+4):
            p=make_world(world);self.assertEqual(len(p['records']),512)
            self.assertEqual(p,json.loads(json.dumps(p)))
        with self.assertRaises(ValueError):make_world(202610129201,True)
        with self.assertRaises(ValueError):make_world(TECH_MIN,False)


if __name__=='__main__':unittest.main()
