"""Bounded technical-only tests, never counted as addition training."""
from core import *
import ops, itertools

def run():
 ops.begin();results={};full=0;events=0
 try:
  # Finite differences of local log probabilities, conditional on sampled inputs.
  c=Controller(6101);mu,p,g,v,u=c.step(19);i=3;j=19;eps=1e-6
  def lp(wv,wg):
   m=wv@u;z=wg@u;return float(np.sum(g*(-np.logaddexp(0,-z))+(1-g)*(-np.logaddexp(0,z))-.5*g*(v-m)**2/SIGMA**2))
  errs=[]
  for name,E in [('Wv',c.Ev),('Wg',c.Eg)]:
   a=getattr(c,name);a[i,j]+=eps;plus=lp(c.Wv,c.Wg);a[i,j]-=2*eps;minus=lp(c.Wv,c.Wg);a[i,j]+=eps;errs.append(abs((plus-minus)/(2*eps)-E[i,j]))
  assert max(errs)<1e-7;results['local_logprob_fd_max_error']=max(errs)
  # Two causal actions: future reward credits both accumulated past scores.
  def expected(z):
   p=1/(1+np.exp(-z));value=0.;grad=0.
   for a,b in itertools.product([0,1],repeat=2):
    pr=(p if a else 1-p)*(p if b else 1-p);r=float(a!=b);value+=pr*r;grad+=pr*(r+.3)*(a+b-2*p)
   return value,grad
  z=-.7;value,grad=expected(z);fd=(expected(z+eps)[0]-expected(z-eps)[0])/(2*eps);assert abs(fd-grad)<1e-9
  results['enumerable_two_action_gradient_error']=abs(fd-grad)
  # Conditional Gaussian score identity using antithetic quadrature samples.
  rng=np.random.default_rng(6102);z=rng.normal(size=200000);z=np.r_[z,-z];mu=.2;v=mu+.1*z;estimate=float(np.mean(v*v*z/.1));assert abs(estimate-2*mu)<.005;results['gaussian_score_expected_gradient']=estimate
  f=Fly(202610236000);assert f.e.bank.side_n=={0:92,1:92};assert len(f.e.bank.kc_side)==5177
  # Every byte is legal, arbitrary controls are separate, all native output codecs roundtrip.
  for t in range(257):assert decode(targets(t))==t
  for t in range(259):f.c.step(t,False)
  results['all_256_payloads_and_EOS_codec']=True
  c1=Controller(34);c2=Controller(34)
  for t in [256,5,2]:c1.step(t)
  for t in [256,2,5]:c2.step(t)
  assert not np.array_equal(c1.h,c2.h);results['order_sensitivity']=float(np.linalg.norm(c1.h-c2.h))
  # Check reward before target transition and frozen baseline across a record.
  c=Controller(7);c.step(256);E=c.Ev.copy();c.credit(np.ones(9),13);assert np.array_equal(c.Gv,(c.rewards[0]-c.b_start)*E);assert c.b==0 and c.b_start==0
  w=c.Wv.copy();c.step(13);c.credit(-np.ones(9),256);assert np.array_equal(w,c.Wv) and c.b_start==0;c.finish();assert not np.array_equal(w,c.Wv)
  results['prefeedback_credit_and_record_frozen_weights_baseline']=True
  # Signed write/cross influence on actual training-only recurrent frames, fixed mild dose.
  f.begin();frames=[]
  for t in [17,250,0,61,3,14]:f.c.step(t);frames.append(f.code())
  base=f.e.clone();m0=base.margins(frames[0])[0];p=base.clone();p.tick(frames[0],residual(m0,173,1/256));events+=1
  m1=p.margins(frames[0])[0];effect=float(np.dot(targets(173),m1-m0));assert effect>0
  results['actual_frame_signed_self_effect']=effect;results['actual_frame_cross_margin_L2']=float(np.linalg.norm(p.margins(frames[1])[0]-base.margins(frames[1])[0]));results['actual_frame_active_counts']=[int(x.sum()) for x in frames]
  f.checkpoint(P/'checkpoints/qualification_restore.npz');dg=f.digest();a=f.query(b'\x00\xff+\x0a=',101);assert f.digest()==dg
  f.record(b'\x00\xff=',b'\x08\xfe',tape=99);full+=1;events+=6
  f.restore(P/'checkpoints/qualification_restore.npz');assert f.digest()==dg;assert f.query(b'\x00\xff+\x0a=',101)==a;assert f.digest()==dg
  results['snapshot_restore_RNG_and_query_readonly']=True
  f.record(b'\x80=',b'\x00',native=False,controller=False,tape=103);full+=1;events+=4
  assert f.e.write_l1==0 and f.e.slow_write_l1==0 and f.c.movement==0 and f.c.b==0
  results['NO_OLD_both_paths_off']=True
  # A saturating EOS fixture tests the generic stop and cap code without target data.
  f.e.margins=lambda x:(np.ones(9),None,None);out=f.generation(1);assert out[0]==[] and out[1] and not out[2]
  f.e.margins=lambda x:(-np.ones(9),None,None);out=f.generation(1,2);assert out[0]==[0,0] and not out[1] and out[2]
  results['own_EOS_and_explicit_truncation_fixture']=True
  # Manifest-only arithmetic validity, exclusion, orientation and countermodel audit.
  import v6_data as data,collections
  for seed in data.WORLDS:
   d=data.build(seed);sets=[{data.key(r['prompt']) for r in d[k]} for k in ('held','old','new')];assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
   assert [len(s) for s in sets]==[64,128,32]
   assert collections.Counter(tuple(r['answer']) for r in d['old'])==collections.Counter(tuple(r['answer']) for r in d['shuffled'])
   assert all(len(a['answer'])==len(b['answer']) for a,b in zip(d['old'],d['shuffled']))
   for r in d['old']+d['new']+d['held']:assert bytes(r['answer'])==str(sum(r['operands'])).encode()
   assert len(data.records(d,'old'))==4096 and len(data.records(d,'new'))==256
   cm=data.countermodels(data.records(d,'old'),data.orientations(d['held']));assert cm['raw_lookup']==cm['canonical_lookup']==0
  results['manifest_exclusion_shuffling_analytic_label_and_lookup_controls']=True
  results['status']='PASS'
 finally:
  results.update(technical_full_records=full,technical_full_record_events=events-1,technical_partial_native_events=1,technical_generation_events_in_resource_only=True)
  ops.save('cycles/02_QUALIFICATION.json',results);ops.end()
if __name__=='__main__':run()
