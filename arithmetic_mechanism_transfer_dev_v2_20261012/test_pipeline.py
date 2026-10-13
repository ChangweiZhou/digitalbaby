"""Adversarial audit and synthetic finish tests; no new learning here."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from analysis import contrasts, reduce, cpu_pair, curve, j_curve, CANDIDATES
from auditor import audit
from core import CONFIGS
from runner import run_job
from worlds import digest, TECH_MIN

OUT=Path(__file__).parent


class AuditPipeline(unittest.TestCase):
    def receipt(self):
        path=OUT/'cycles/cycle3/dispatch/receipts/202610128003_COMP-LOCAL-FULL.json.gz'
        if not path.exists():
            path=OUT/'cycles/cycle2/receipts/COMP-LOCAL-FULL.json.gz'
        with gzip.open(path,'rt') as f:
            return json.load(f)

    def reject(self,edit):
        r=self.receipt();edit(r)
        r['receipt_digest']=digest({k:v for k,v in r.items() if k!='receipt_digest'})
        with self.assertRaises((AssertionError,ValueError,KeyError)):
            audit(r)

    def test_untampered_fixture_accepted(self):
        self.assertTrue(audit(self.receipt())['accepted'])

    def test_world_canonical_disk_roundtrip(self):
        from worlds import make_world
        w=make_world(TECH_MIN)
        self.assertEqual(w,json.loads(json.dumps(w)))

    def test_deadlines_roster_and_atomic_status(self):
        from operations import manifest,may_admit,atomic_json
        roster=manifest()
        self.assertEqual(len(roster),432)
        self.assertEqual(len({r['job_id'] for r in roster}),432)
        self.assertTrue(may_admit(60,300))
        self.assertFalse(may_admit(8*3600,300))
        self.assertFalse(may_admit(7*3600,4000))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'STATUS.json'
            atomic_json(path,{'status':'TEST_ONLY'})
            self.assertEqual(json.loads(path.read_text())['status'],'TEST_ONLY')
            self.assertEqual(len(list(Path(d).glob('*.tmp'))),0)

    def test_tamper_write_branch(self):
        def edit(r):
            r['histories']['N_ALL']['ledger'][0]['events'][-1]['actual_update_count']=1
        self.reject(edit)

    def test_tamper_hidden_no_write_statistics(self):
        self.reject(lambda r:r['histories']['N_ALL']['ledger'][0]['after'].update(P='forged'))

    def test_tamper_no_write_router_count(self):
        self.reject(lambda r:r['histories']['N_ALL']['ledger'][0].update(bank_counts=[1,0,0,0]))

    def test_tamper_target(self):
        self.reject(lambda r:r['histories']['W']['ledger'][0].update(target_hex='3030'))

    def test_tamper_birth(self):
        self.reject(lambda r:r['histories']['W'].update(birth='not_same_birth'))

    def test_tamper_exact_split(self):
        self.reject(lambda r:r['plan']['held_pool'].append(r['plan']['training_pool'][0]))

    def test_tamper_missing_event(self):
        self.reject(lambda r:r['histories']['W']['ledger'][0]['events'].pop())

    def test_tamper_character_coverage(self):
        self.reject(lambda r:r['histories']['W']['checkpoints'][0]['observed_answer_bytes'].update({'+':[]}))

    def test_tamper_generated_accuracy(self):
        def edit(r):
            x=r['histories']['W']['checkpoints'][0]['held'][0]
            x['correct']=not x['correct']
        self.reject(edit)

    def test_tamper_truncation_as_eos(self):
        def edit(r):
            x=r['histories']['N_ALL']['final'][0]
            x['correct']=True
        self.reject(edit)

    def test_tamper_record_accounting(self):
        self.reject(lambda r:r.update(records_applied=r['records_applied']+1))

    def test_function_science_authorization_not_only_cli(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'forbidden.json.gz'
            with self.assertRaises(PermissionError):
                run_job(202610129001,'HASH',path,technical=False)
            self.assertFalse(path.exists())

    def test_duplicate_trajectory_rejected_before_world_build(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'exists.json.gz';path.write_bytes(b'preserved')
            with self.assertRaises(FileExistsError):
                run_job(TECH_MIN,'HASH',path)
            self.assertEqual(path.read_bytes(),b'preserved')

    def test_full_116_contrasts_and_unavailable_cpu(self):
        files=list((OUT/'cycles/cycle2/receipts').glob('*.json.gz'))
        if len(files)!=9:
            self.skipTest('full technical receipts not yet available')
        world={}
        for f in files:
            with gzip.open(f,'rt') as handle:r=json.load(handle)
            world[r['config']]=r
        # Repeated fixture only exercises the reducer; not 48 scientific worlds.
        summary=reduce([world]*48,technical=True)
        self.assertEqual(len(summary['contrasts']),116)
        self.assertEqual(summary['verdict'],'TECHNICAL_REDUCER_ONLY')
        self.assertIsNone(summary['selected'])
        from core import BASELINE
        a=copy.deepcopy(world[BASELINE]);b=copy.deepcopy(world[BASELINE])
        for i,row in enumerate(a['histories']['W']['checkpoints']):row['cpu']=100.+i
        for i,row in enumerate(b['histories']['W']['checkpoints']):row['cpu']=1.+i
        self.assertIsNone(cpu_pair(a,b))

    def test_cpu_integral_independent_reference(self):
        def mock(values,cpu):
            return {'histories':{'W':{'checkpoints':[{'scores':dict.fromkeys('+-*/',v),'cpu':c}
                for v,c in zip(values,cpu)]}}}
        a=mock([.7]*6,[1,2,4,8,16,32]);b=mock([.2]*6,[2,4,8,16,32,64])
        self.assertAlmostEqual(cpu_pair(a,b),.5)


if __name__=='__main__':
    unittest.main()
