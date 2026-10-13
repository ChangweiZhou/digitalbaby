"""Adversarial technical tests; accepted fixtures are never scientific samples."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from core import AdapterCore, CONFIGS, build_core
from worlds import TECH_MIN, digest
from auditor import audit, reference_countermodels
from runner import run_job, countermodels
from operations import manifest, may_admit, atomic_json, committed_index
from analysis import reduce, contrasts, cpu_pair, j_curve, paired_interval, direct_interference

OUT=Path(__file__).parent


class PipelineTests(unittest.TestCase):
    def folder(self):
        cold=OUT/'cycles/cycle3/dispatch/receipts'
        return cold if len(list(cold.glob('*.json.gz')))==5 else OUT/'cycles/cycle2/receipts'

    def receipt(self):
        with gzip.open(self.folder()/'202610129110_LEARN-HASH.json.gz','rt') as f:return json.load(f)

    def reject(self,edit):
        r=self.receipt();edit(r);r['receipt_digest']=digest({k:v for k,v in r.items() if k!='receipt_digest'})
        with self.assertRaises((AssertionError,ValueError,KeyError)):audit(r)

    def test_fixture_accepted(self):self.assertTrue(audit(self.receipt(),True)['accepted'])
    def test_forbidden_write(self):self.reject(lambda r:r['histories']['N_ALL']['ledger'][0]['events'][-1].update(actual_update_count=1))
    def test_forbidden_hidden_update(self):self.reject(lambda r:r['histories']['N_ALL']['ledger'][0]['after'].update(A2='forged'))
    def test_wrong_birth_all_histories(self):
        def edit(r):
            for h in r['histories'].values():h['birth']='consistent_but_wrong'
        self.reject(edit)
    def test_wrong_observed_target(self):self.reject(lambda r:r['histories']['SHUFFLE']['ledger'][0].update(target_hex='3939'))
    def test_wrong_final_state_chain(self):self.reject(lambda r:r['histories']['W'].update(final_digest='forged'))
    def test_missing_event(self):self.reject(lambda r:r['histories']['W']['ledger'][0]['events'].pop())
    def test_scoring_forgery(self):
        def edit(r):
            x=r['histories']['W']['final'][0];x['correct']=not x['correct']
        self.reject(edit)
    def test_source_coverage_forgery(self):
        r=self.receipt();r['source_lock']={};r['receipt_digest']=digest({k:v for k,v in r.items() if k!='receipt_digest'})
        with self.assertRaises(AssertionError):audit(r,True)
    def test_block_taught_denominator_forgery(self):self.reject(lambda r:r['histories']['W']['blocks'][0]['taught'].pop())
    def test_countermodel_forgery(self):self.reject(lambda r:r['histories']['SHUFFLE']['checkpoints'][-1]['countermodels']['left'].update({'+':-1}))
    def test_actual_label_countermodels(self):
        p=self.receipt()['plan']
        for h in ('W','SHUFFLE'):
            for prefix in (16,32,512):self.assertEqual(countermodels(p,prefix,h),reference_countermodels(p,prefix,h))
        # An intentionally false observed sum gives an inverse table match;
        # this must not silently be hard-coded zero.
        p={'records':[{'row':['+',2,3,5],'index':0,'answer':'5'}],
           'shuffled_answers':{'0':'8'},'held_pool':[['-',8,2,3],['+',2,3,5],['*',2,3,6],['/',6,2,3]]}
        self.assertEqual(countermodels(p,1,'SHUFFLE')['inverse_family_lookup']['-'],1.)

    def test_science_admission_and_immutable_claim(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'never.json.gz'
            with self.assertRaises(PermissionError):run_job(202610129201,'LEARN-HASH',path,False)
            self.assertFalse(path.exists())
            from runner import source_lock
            auth=Path(d)/'auth.json';auth.write_text(json.dumps({'science_authorized':True,'source_lock':source_lock()}))
            with patch('runner.qualification_ready',return_value=False):
                with self.assertRaises(PermissionError):run_job(202610129201,'LEARN-HASH',path,False,authorization=auth)
            self.assertFalse(path.exists())
            path.with_suffix('.claim').write_text('do not rerun')
            with self.assertRaises(FileExistsError):run_job(TECH_MIN,'LEARN-HASH',path,True)
            self.assertFalse(path.exists())

    def test_checkpoint_rejects_malformed_metadata_and_frozen_changes(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'x.npz';c=build_core('FROZEN-SHARED',TECH_MIN);c.begin_sequence();c.start_reply();c.save_checkpoint(path)
            with np.load(path,allow_pickle=False) as data:original={k:data[k].copy() for k in data.files}
            for key,value in [('learning_events',True),('elapsed_seconds',float('nan')),('next',4000)]:
                arrays=copy.deepcopy(original);m=json.loads(arrays['metadata'].tobytes());m['base'][key]=value
                arrays['metadata']=np.frombuffer(json.dumps(m).encode(),np.uint8)
                with path.open('wb') as f:np.savez_compressed(f,**arrays)
                with self.assertRaises(ValueError):AdapterCore.load_checkpoint(path)
            arrays=copy.deepcopy(original);arrays['A1'][0,0]+=.1
            with path.open('wb') as f:np.savez_compressed(f,**arrays)
            with self.assertRaises(ValueError):AdapterCore.load_checkpoint(path)

    def test_fixed_roster_counts_and_deadlines(self):
        roster=manifest();self.assertEqual(len(roster),240);self.assertEqual(len({r['job_id'] for r in roster}),240)
        self.assertTrue(may_admit(10,60));self.assertFalse(may_admit(8*3600,60));self.assertFalse(may_admit(7*3600,4000))
        self.assertEqual(240*3*512,368640)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'STATUS.json';atomic_json(p,{'status':'TEST_ONLY'})
            self.assertEqual(json.loads(p.read_text())['status'],'TEST_ONLY')
            self.assertFalse(list(Path(d).glob('*.tmp')))

    def test_41_exact_contrasts_and_unresolved_zero_variance(self):
        world={}
        for c in CONFIGS:
            with gzip.open(self.folder()/f'202610129110_{c}.json.gz','rt') as f:world[c]=json.load(f)
        # Duplicate references exercise reduction only, never generate n=48 science.
        summary=reduce([world,world],True)
        spec=json.loads((OUT/'design/DESIGN.json').read_text())
        self.assertEqual(set(summary['contrasts']),set(spec['statistics']['contrasts']))
        self.assertEqual(len(summary['contrasts']),41);self.assertIsNone(summary['selected'])
        self.assertEqual(summary['verdict'],'TECHNICAL_PIPELINE_ONLY')
        self.assertEqual(paired_interval([0.]*48)['status'],'ZERO_VARIANCE_UNRESOLVED')
        self.assertEqual(paired_interval([None]*48)['status'],'UNAVAILABLE')

    def test_cpu_interval_reference_and_no_overlap(self):
        def mock(v,c):return {'histories':{'W':{'checkpoints':[{'scores':dict.fromkeys('+-*/',v),'cpu':x} for x in c]}}}
        a=mock(.7,[1,2,4,8,16,32]);b=mock(.2,[2,4,8,16,32,64])
        self.assertAlmostEqual(cpu_pair(a,b),.5);self.assertIsNone(cpu_pair(a,mock(.2,[100,101,102,103,104,105])))
        self.assertAlmostEqual(j_curve([.1]*6),.1)

    def test_direct_same_operation_interference(self):
        r=copy.deepcopy(self.receipt())
        # Make all non-target operations huge changes: they must not affect + loss.
        for cp in r['histories']['W']['checkpoints']:cp['scores']={'+':.6,'-':1.,'*':1.,'/':1.}
        for block in r['histories']['W']['blocks']:block['scores']={'+':.8,'-':0.,'*':0.,'/':0.}
        d=direct_interference(r,'+')
        self.assertAlmostEqual(d['mean'],-.2)
        self.assertEqual(len(d['interference_points'])+len(d['last_block_no_intervening_points']),6)

    def test_certificate_required_for_committed_counts(self):
        folder=self.folder();expected={(202610129110,c) for c in CONFIGS}
        self.assertEqual(committed_index(folder,expected,True,True)['committed_jobs'],5)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'receipts';p.mkdir();r=self.receipt()
            with gzip.open(p/'uncertified.json.gz','wt') as f:json.dump(r,f)
            with self.assertRaises(FileNotFoundError):committed_index(p,expected,True)


if __name__=='__main__':unittest.main()
