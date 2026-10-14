import ast,copy,gzip,json,pickle,unittest
from pathlib import Path
import numpy as np
from core import Core,Config,EOS
from interface import teach,probe,ReadOnly,working,restore,birth,digest
from auditor import audit
from worlds import make_world
from metrics import J,G,superiority,holm

class Tests(unittest.TestCase):
 def test_no_evaluator_imports_or_numerical_features(self):
  source=Path('core.py').read_text(); tree=ast.parse(source)
  imports=[]
  for node in ast.walk(tree):
   if isinstance(node,ast.Import):imports += [x.name for x in node.names]
   if isinstance(node,ast.ImportFrom):imports.append(node.module)
  self.assertEqual(set(imports),{'copy','hashlib','pickle','dataclasses','numpy'})
  for forbidden in ('decode(', 'ord(', 'task_source','operand','operation','float(byte','48 <=', 'answer_length'):
   self.assertNotIn(forbidden,source)

 def test_all_births_matched_independent_and_reference_pinned(self):
  names=('R1-DIAG','R1-FULL','R4-DIAG','R4-FULL','F1-FULL','F4-FULL')
  cores=[birth(n,202610130102) for n in names]
  for c in cores[1:]:
   for k in cores[0].theta:np.testing.assert_array_equal(c.theta[k],cores[0].theta[k])
  cores[0].theta['E'][0,0]+=1
  self.assertNotEqual(cores[0].theta['E'][0,0],cores[1].theta['E'][0,0])
  src=Path('../arithmetic_value_interaction_dev_v1_20261012/vendor/legacy_core.py').read_bytes()
  self.assertEqual(src,Path('vendor/legacy_core.py').read_bytes())
  for c in cores:
   self.assertEqual(sum(a.size for a in c.theta.values()),19696)
   self.assertEqual(c.V.size,16705)

 def test_branch_permissions_and_parent_fork(self):
  for name in ('R1-FULL','R4-DIAG','F1-FULL','LEGACY-B'):
   c=birth(name,123);initial=digest(c)
   teach(c,b'\x00\xff=',b'\x89',False)
   self.assertEqual(initial,digest(c))
   parent=copy.deepcopy(c); child=copy.deepcopy(c)
   teach(child,b'\x00\xff=',b'\x89',True)
   self.assertNotEqual(digest(child),initial);self.assertEqual(digest(c),digest(parent))
   state=working(child);before=digest(child)
   rows=probe(child,[{'prompt_hex':'00ff3d','answer_hex':'89'}])
   self.assertEqual(before,digest(child))
   if isinstance(child,Core):self.assertEqual(child.state_digest(),copy.deepcopy(child).state_digest())

 def test_control_guard_rejects_actual_bad_writer(self):
  c=Core(2);c.begin();c.start_reply()
  with ReadOnly(c):
   with self.assertRaises(ValueError):c.observe(77,True)
  self.assertEqual(c.learning_events,0)

 def test_probes_preserve_pending_gradients(self):
  c=Core(2,Config(depth=4));c.begin();c.observe(77,True)
  initial=c.state_digest();probe(c,[{'prompt_hex':'3d','answer_hex':'00'}])
  self.assertEqual(initial,c.state_digest())

 def test_metrics_common_clock_and_holm(self):
  def row(cpu,a):return {'cpu':cpu,'held_retained':dict.fromkeys('+-*/',a)}
  a=[row(1,.2),row(2,.8),row(4,1.)];b=[row(2,.3),row(3,.4),row(6,.5)]
  ga,gb=G(a,b);self.assertAlmostEqual(ga,.45);self.assertAlmostEqual(gb,.175)
  self.assertEqual(superiority([1]*8),2**-8);self.assertEqual(superiority([0]*8),1.)
  self.assertEqual(superiority([-1]*8),1.)
  self.assertEqual(holm([.001,.02,.5]),[True,True,False])
  self.assertEqual(holm([.01]*12),[False]*12)

 def test_tampered_full_receipts_rejected(self):
  with gzip.open('cycles/cycle2/receipts/202610130101_R1-FULL.json.gz','rt') as f:r=json.load(f)
  self.assertTrue(audit(r)['accepted'])
  changes=[
   lambda x:x['histories']['N_OLD']['root_records'][0].update(allowed=True),
   lambda x:x['histories']['N_OLD']['root_records'][0].update(encoder_steps=1),
   lambda x:x['histories']['W']['checkpoints'][0]['cohort_records'][0].update(allowed=False),
   lambda x:x['histories']['W']['checkpoints'][0].update(parent_after='corrupted'),
   lambda x:x.update(record_applications=2113),
   lambda x:x['histories']['W']['checkpoints'][0]['held_retained'].pop(),
   lambda x:x['histories']['W']['root_records'][0].update(answer_hex='00'),
   lambda x:x['histories']['W']['checkpoints'][0].update(fork_cpu=-1),
   lambda x:x['histories']['W']['checkpoints'][0]['held_retained'][0].update(choice=999),
  ]
  for mutate in changes:
   x=copy.deepcopy(r);mutate(x)
   with self.assertRaises(AssertionError):audit(x)
  with self.assertRaises(AssertionError):audit(r,expected_source={'wrong':'hash'})

 def test_technical_admission_only(self):
  with self.assertRaises(PermissionError):make_world(202610130201,False)
  with self.assertRaises(ValueError):make_world(202610130201,True)
  a=make_world(202610130102);b=make_world(202610130102)
  self.assertEqual(a,b);self.assertEqual(len(a['cohort_records']),32)
  self.assertEqual(len(a['records']),512)
  keys={r['prompt_hex'] for r in a['cohort_pairs']}
  self.assertEqual(len(keys),8)
  self.assertTrue(keys.isdisjoint({r['prompt'].encode().hex() for r in a['records']}))

 def test_long_stream_bounded_and_actual_eos_after_max_payload(self):
  c=Core(4,Config(width=4,embedding=2));c.begin()
  for i in range(100):c.observe(i%256,True)
  self.assertEqual(len(c.tape),32);self.assertEqual(len(c.trace),32)
  self.assertGreater(c.dropped_targets,0)
  state=c.state_digest(); c.generate(0);self.assertEqual(state,c.state_digest())
  c.observe(EOS,True);self.assertEqual(c.last_update['targets'],101)
  self.assertLessEqual(c.last_update['retained_targets'],32)

if __name__=='__main__':unittest.main(verbosity=2)
