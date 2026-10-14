"""Pipeline/source/receipt tamper tests use saved technical data, never train science."""
import ast,copy,gzip,json,unittest
from pathlib import Path
from auditor import audit
from analysis import mean_test,common_cpu,shortcut
from interface import birth,teach,digest,probe

class Tests(unittest.TestCase):
    def receipt(self):
        p=Path('cycles/cycle2/receipts/202610139802_G-CE.json.gz')
        with gzip.open(p,'rt') as f:r=json.load(f)
        return r
    def test_independent_receipt_rejections(self):
        original=self.receipt();source=original['source_identity']
        self.assertTrue(audit(original,source)['accepted'])
        cases=[]
        r=copy.deepcopy(original);r['histories']['N_OLD']['records'][0]['permissions'][-1]=True;cases.append(r)
        r=copy.deepcopy(original);r['histories']['N_OLD']['records'][0]['all_bytes']=True;cases.append(r)
        r=copy.deepcopy(original);r['histories']['W']['records'][0]['counters_before'][0]=7;cases.append(r)
        r=copy.deepcopy(original);r['histories']['W']['checkpoints'][0]['cohort_records'][0]['counters_before'][0]+=1;cases.append(r)
        r=copy.deepcopy(original);r['histories']['W']['checkpoints'][0]['parent_after']='tampered';cases.append(r)
        r=copy.deepcopy(original);r['histories']['W']['checkpoints'][0]['held_retained'][0]['answer_hex']='00';cases.append(r)
        r=copy.deepcopy(original);r['record_applications']-=1;cases.append(r)
        r=copy.deepcopy(original);r['technical']=False;cases.append(r)
        r=copy.deepcopy(original);r['source_identity']['core.py']='tampered';cases.append(r)
        r=copy.deepcopy(original);r['resource']['production_cpu']+=1;cases.append(r)
        for r in cases:
            with self.assertRaises(AssertionError):audit(r,source)

    def test_frozen_and_whole_record_permission(self):
        for name in ('G-CE','L-CE','G-FROZEN-CE','L-FROZEN-CE','RAW-RLS','LEGACY-B'):
            c=birth(name,71);initial=digest(c)
            teach(c,b'\x00\x0a\x3d\xff',b'\xff\x00',False)
            self.assertEqual(initial,digest(c))
            probe(c,[{'prompt_hex':'000a3dff','answer_hex':'ff00'}])
            self.assertEqual(initial,digest(c))

    def test_core_import_boundary(self):
        tree=ast.parse(Path('core.py').read_text())
        imports=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imports.extend(n.name.split('.')[0] for n in node.names)
            if isinstance(node,ast.ImportFrom):imports.append(node.module.split('.')[0])
        self.assertEqual(set(imports),{'dataclasses','copy','hashlib','pickle','numpy'})

    def test_constant_mean_and_actual_initial_compute_score(self):
        self.assertAlmostEqual(mean_test([.1]*16),__import__('math').exp(-.08))
        self.assertEqual(mean_test([0]*16),1)
        # actual initial score .3 persists until first observed checkpoint.
        a=[(0,.3),(2,.8),(4,1.)];b=[(0,.1),(1,.5),(3,.7)]
        ga,gb=common_cpu(a,b)
        self.assertAlmostEqual(ga,(2*.3+1*.8)/3)
        self.assertAlmostEqual(gb,(1*.1+2*.5)/3)
        self.assertEqual(shortcut([], [{'prompt_hex':'00','answer_hex':'01'}]),0)

if __name__=='__main__':unittest.main(verbosity=2)
