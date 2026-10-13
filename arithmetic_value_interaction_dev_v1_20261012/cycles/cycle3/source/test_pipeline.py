"""Independent audit tampering, fixed-statistics and admission tests; no science."""
import copy,gzip,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from core import InteractionCore,CONFIGS,CANDIDATES,build_core,EOS
from worlds import TECH_MIN,SCI_MIN,digest,make_world
from runner import run_job,source_lock,probe
from auditor import audit
from operations import manifest,may_admit,committed_index
from analysis import contrasts,paired_interval,cpu_scores,cpu_pair,j_curve,reduce
OUT=Path(__file__).parent
class PipelineTests(unittest.TestCase):
 def receipt(self):
  for folder,world in [('cycles/cycle3/dispatch/receipts',202610129621),('cycles/cycle2/dispatch/receipts',202610129610),('cycles/cycle2/initial/receipts',202610129610)]:
   p=OUT/folder/f'{world}_VALUE-PROD.json.gz'
   if p.exists():
    with gzip.open(p,'rt') as f:return json.load(f)
  raise RuntimeError('full technical receipt missing')
 def reject(self,edit):
  r=self.receipt();edit(r);r['receipt_digest']=digest({k:v for k,v in r.items() if k!='receipt_digest'})
  with self.assertRaises((AssertionError,ValueError,KeyError)):audit(r)
 def test_receipt_actual_full_record_audit(self):self.assertTrue(audit(self.receipt())['accepted'])
 def test_forbidden_N_parameter_write(self):self.reject(lambda r:r['histories']['N_ALL']['ledger'][0]['after'].update(A='forged'))
 def test_forbidden_N_optimizer_write(self):self.reject(lambda r:r['histories']['N_ALL']['ledger'][0]['after'].update(M_A='forged'))
 def test_forbidden_N_step_write(self):self.reject(lambda r:r['histories']['N_ALL']['ledger'][0].update(optimizer_step=1))
 def test_wrong_SHUFFLE_aux_value(self):
  def edit(r):
   i=next(i for i,x in enumerate(r['plan']['records']) if x['answer']!=r['plan']['shuffled_answers'][str(i)])
   row=r['histories']['SHUFFLE']['ledger'][i];row['events'][-1]['core_audit']['aux_target']=int(r['plan']['records'][i]['answer'])
  self.reject(edit)
 def test_early_aux_value(self):self.reject(lambda r:r['histories']['W']['ledger'][0]['events'][-2]['core_audit'].update(aux_used=True,aux_target=3))
 def test_true_value_decoder_input(self):self.reject(lambda r:r['histories']['W']['ledger'][0]['events'][-1]['core_audit'].update(decoder_target_input=True))
 def test_generated_aux_provenance(self):self.reject(lambda r:r['histories']['W']['ledger'][0]['events'][-1]['core_audit'].update(source='generation'))
 def test_prior_snapshot_tamper(self):self.reject(lambda r:r['histories']['W']['ledger'][0]['events'][-1]['core_audit'].update(gradient_snapshot='postappend'))
 def test_missing_event(self):self.reject(lambda r:r['histories']['W']['ledger'][0]['events'].pop())
 def test_score_forgery(self):
  def edit(r):r['histories']['W']['final'][0]['correct']=not r['histories']['W']['final'][0]['correct']
  self.reject(edit)
 def test_wrong_source(self):
  r=self.receipt();r['source_lock']={};r['receipt_digest']=digest({k:v for k,v in r.items() if k!='receipt_digest'})
  with self.assertRaises(AssertionError):audit(r,True)
 def test_source_isolation(self):
  import ast
  tree=ast.parse((OUT/'core.py').read_text());imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
  self.assertFalse(any(m and ('worlds' in m or 'task_source' in m or 'runner' in m) for m in imports))
 def test_82_contrasts_exact(self):
  r=self.receipt();w={c:copy.deepcopy(r) for c in CONFIGS}
  cs=contrasts([w]);spec=json.loads((OUT/'design/DESIGN.json').read_text())
  self.assertEqual(set(cs),set(spec['statistics']['contrasts']));self.assertEqual(len(cs),82)
  s=reduce([w,w],True);self.assertIsNone(s['selected']);self.assertEqual(s['verdict'],'TECHNICAL_PIPELINE_ONLY')
 def test_zero_variance_full_family_missing(self):
  ordinary=paired_interval([0.]*48);interaction=paired_interval([0.]*48,True)
  self.assertLess(ordinary['lower'],0);self.assertGreater(ordinary['upper'],0)
  self.assertAlmostEqual(interaction['upper'],2*ordinary['upper'])
  self.assertEqual(paired_interval([0.]*47)['status'],'UNAVAILABLE')
  self.assertEqual(paired_interval([None]+[0.]*47)['status'],'UNAVAILABLE')
 def test_common_cpu_domain_and_normalization(self):
  def mock(v,x):return {'histories':{'W':{'checkpoints':[{'scores':dict.fromkeys('+-*/',v),'cpu':a} for a in x]}}}
  a=mock(.9,[1,2,4,8,16,32]);b=mock(.5,[2,4,8,16,32,64]);c=mock(.4,[4,8,16,32,64,128]);d=mock(.1,[8,16,32,64,128,256])
  np.testing.assert_allclose(cpu_scores([a,b,c,d]),[.9,.5,.4,.1])
  self.assertAlmostEqual(cpu_pair(a,b),.4)
  self.assertEqual(cpu_scores([a,b,c,mock(.1,[100,101,102,103,104,105])]),[None]*4)
  self.assertEqual(cpu_scores([mock(.1,[0,2,4,8,16,32])]),[None])
  self.assertAlmostEqual(j_curve([.7]*6),.7)
 def test_namespace_and_family_leakage(self):
  p=make_world(TECH_MIN,True)
  self.assertEqual(len(p['records']),512);self.assertTrue(any(r[3]<0 for r in p['held_pool']))
  self.assertTrue(any(r[3]>9 for r in p['held_pool']))
  for world in (202610129201,202610129748):
   with self.assertRaises(ValueError):make_world(world,True)
  self.assertEqual(len(manifest()),240);self.assertEqual(len({x['job_id'] for x in manifest()}),240)
  self.assertEqual(240*1536,368640)
  self.assertFalse(may_admit(8*3600,60));self.assertTrue(may_admit(60,60))
 def test_science_permission_and_immutable_claim(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x.json.gz'
   with self.assertRaises(PermissionError):run_job(SCI_MIN,'BYTE-ADD',p,False)
   auth=Path(d)/'auth.json';auth.write_text(json.dumps({'science_authorized':True,'source_lock':source_lock()}))
   with patch('runner.qualification_ready',return_value=False):
    with self.assertRaises(PermissionError):run_job(SCI_MIN,'BYTE-ADD',p,False,authorization=auth)
   p.with_suffix('.claim').write_text('immutable')
   with self.assertRaises(FileExistsError):run_job(TECH_MIN,'BYTE-ADD',p,True)
 def test_probe_restores_parser_and_working_state(self):
  c=InteractionCore('VALUE-ADD',TECH_MIN);c.begin_sequence();c.observe_bytes(b'ctx',False);c.start_reply();c.observe_bytes(b'-7',True)
  before=(c.transient_digest(),c.learned_digest(),copy.deepcopy(c.last_event_audit))
  probe(c,[['+',1,2,3],['*',3,4,12]])
  self.assertEqual(before,(c.transient_digest(),c.learned_digest(),c.last_event_audit))
 def test_preupdate_joint_gradient_is_actual(self):
  c=InteractionCore('VALUE-PROD',TECH_MIN);c.begin_sequence();c.observe_bytes(b'ctx',False);c.start_reply();c.observe_bytes(b'12',True)
  state,_=c._forward();_,g=c._loss_grad(state,EOS,12);norm=np.sqrt(sum(np.sum(a*a) for a in g.values()));factor=min(1,1/norm)
  old={k:a.copy() for k,a in c.theta.items()};mom={k:a.copy() for k,a in c.M.items()};step=c.step+1
  c.observe_end(True)
  for k in old:
   grad=g[k]*factor;m=.99*mom[k]+.01*grad*grad
   np.testing.assert_allclose(c.M[k],m,atol=1e-15)
   np.testing.assert_allclose(c.theta[k],old[k]-.001*grad/(np.sqrt(m/(1-.99**step))+1e-8),atol=1e-14)
 def test_exposure_forgery(self):self.reject(lambda r:r['histories']['W']['checkpoints'][0]['exposure']['+'].update(records=-1))
 def test_auxiliary_loss_forgery(self):self.reject(lambda r:r['histories']['W']['ledger'][0]['events'][-1]['core_audit'].update(auxiliary_loss=-1))
if __name__=='__main__':unittest.main()
