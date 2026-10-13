import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from core import InteractionCore,CANDIDATES,EOS,SHAPES,softmax
class CoreTests(unittest.TestCase):
 def setup_prompt(self,c,text=b'ctx-A',answer=b''):
  c.begin_sequence();c.observe_bytes(text,False);c.start_reply();c.observe_bytes(answer,False)
 def test_gradients_every_array_and_path(self):
  rng=np.random.default_rng(99)
  for name in CANDIDATES:
   c=InteractionCore(name,202610129600);self.setup_prompt(c,b'7*6=',b'4')
   for key in ('w_value','R','V','c'):c.theta[key][:]=rng.normal(0,.1,c.theta[key].shape)
   c.theta['b_value'][:]=.7
   for y in (None,49):
    state,_=c._forward();_,grads=c._loss_grad(state,EOS,y)
    for key,a in c.theta.items():
     for _ in range(3):
      ix=tuple(rng.integers(n) for n in a.shape);old=a[ix];eps=1e-5
      a[ix]=old+eps;sp,_=c._forward();lp=c._loss_grad(sp,EOS,y)[0]
      a[ix]=old-eps;sm,_=c._forward();lm=c._loss_grad(sm,EOS,y)[0]
      a[ix]=old;num=(lp-lm)/(2*eps)
      self.assertAlmostEqual(num,float(grads[key][ix]),delta=2e-5,msg=f'{name}/{key}/{y}')
 def test_births_sizes_independent(self):
  cs=[InteractionCore(n,202610129600) for n in CANDIDATES]
  self.assertEqual(len({c.birth_digest() for c in cs}),1)
  for c in cs:
   self.assertEqual(sum(a.size for a in c.theta.values()),62466)
   self.assertEqual(c.storage_accounting()['learned_array_bytes'],5210144)
  for a,b in zip(cs,cs[1:]):
   for k in a.arrays():self.assertFalse(np.shares_memory(a.arrays()[k],b.arrays()[k]))
 def test_parser_and_no_write(self):
  valid={b'0':0,b'12':12,b'-7':-7,b'1000000':1000000,b'-1000000':-1000000}
  for text,target in list(valid.items())+[(s,None) for s in (b'',b'-',b'+1',b'-0',b'00',b'01',b' 1',b'1\n',b'1000001',b'\xff')]:
   c=InteractionCore('VALUE-PROD',202610129600);before=c.learned_digest();self.setup_prompt(c,answer=text)
   self.assertEqual(c.observed_numeric_target(),target);c.observe_end(False);self.assertEqual(c.learned_digest(),before)
 def test_observed_only_aux_and_preupdate(self):
  c=InteractionCore('VALUE-PROD',202610129600);self.setup_prompt(c)
  for byte in b'-7':
   before=c.prediction();ret=c.observe(byte,True);np.testing.assert_array_equal(before.probabilities,ret.probabilities)
   self.assertFalse(c.last_event_audit['aux_used'])
  before=c.prediction();c.observe_end(True);self.assertEqual(c.last_event_audit['aux_target'],-7)
  self.assertEqual(c.aux_events,1);self.assertEqual(c.step,3)
  self.setup_prompt(c);state=c.learned_digest();num=copy.deepcopy(c._num);c.generate(16)
  self.assertEqual(c.learned_digest(),state);self.assertEqual(c._num,num)
 def test_independent_gradients_and_initial_zero(self):
  c=InteractionCore('BYTE-PROD',202610129600);self.setup_prompt(c)
  state,_=c._forward();_,g=c._loss_grad(state,65)
  for k in ('A','B','C','E','b','bB','bC','w_value','b_value','D','bD'):self.assertFalse(g[k].any(),k)
  address=c._address();mem=softmax(state['zm']);mem[65]-=1
  c.observe(65,True);np.testing.assert_allclose(c._weights[address],np.tile(-mem*.25,(16,1)),rtol=1e-6,atol=1e-9)
 def test_optimizer_exact(self):
  c=InteractionCore('BYTE-ADD',202610129600);g={k:np.ones_like(a) for k,a in c.theta.items()}
  original={k:a.copy() for k,a in c.theta.items()};c._shared_step(g)
  q=1/np.sqrt(62466)
  for k in g:
   np.testing.assert_allclose(c.M[k],.01*q*q)
   np.testing.assert_allclose(c.theta[k],original[k]-.001*q/(q+1e-8),atol=1e-14)
 def test_binary_chunking_checkpoint_overflow(self):
  a=InteractionCore('VALUE-ADD',202610129600);b=InteractionCore('VALUE-ADD',202610129600)
  for c in (a,b):c.begin_sequence()
  payload=bytes(range(256))*17
  a.observe_bytes(payload,False)
  for k in range(0,len(payload),127):b.observe_bytes(payload[k:k+127],False)
  for c in (a,b):c.start_reply();c.observe_bytes(b'\x00\n\xff',True)
  self.assertEqual(a.transient_digest(),b.transient_digest());self.assertEqual(a.learned_digest(),b.learned_digest())
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'c.npz';a.save_checkpoint(p);z=InteractionCore.load_checkpoint(p)
   self.assertEqual(a.transient_digest(),z.transient_digest());self.assertEqual(a.learned_digest(),z.learned_digest())
   a.observe_end(True);z.observe_end(True);self.assertEqual(a.learned_digest(),z.learned_digest())
  self.assertFalse(a._h0()[:72].any())
 def test_no_true_target_at_inference(self):
  c=InteractionCore('VALUE-ADD',202610129600);self.setup_prompt(c)
  p=c.prediction().probabilities.copy();c._num.update(magnitude=123,count=3)
  np.testing.assert_array_equal(c.prediction().probabilities,p)
 def test_shared_fixture_never_accesses_hash(self):
  c=InteractionCore('VALUE-PROD',202610129600);self.setup_prompt(c)
  def forbidden():raise AssertionError('shared diagnostic touched hash')
  c._address=forbidden;c.shared_prediction()
 def test_checkpoint_rejects_optimizer_and_parser_tampering(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'c.npz';c=InteractionCore('VALUE-ADD',202610129600);self.setup_prompt(c,answer=b'-7');c.save_checkpoint(p)
   with np.load(p,allow_pickle=False) as z:original={k:z[k].copy() for k in z.files}
   for edit in ('moment','step','parser','elapsed'):
    arrays=copy.deepcopy(original);m=json.loads(arrays['metadata'].tobytes())
    if edit=='moment':arrays['M_A'][0,0]=-1
    elif edit=='step':m['step']=True
    elif edit=='parser':m['numeric']['magnitude']=10000010
    else:m['base']['elapsed_seconds']=float('nan')
    arrays['metadata']=np.frombuffer(json.dumps(m).encode(),np.uint8)
    with p.open('wb') as f:np.savez_compressed(f,**arrays)
    with self.assertRaises(ValueError):InteractionCore.load_checkpoint(p)
if __name__=='__main__':unittest.main()
