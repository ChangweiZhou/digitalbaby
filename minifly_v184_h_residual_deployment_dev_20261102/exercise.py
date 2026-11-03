"""Three bounded technical cycles. No scientific fitting or alpha/cap search."""
import bootstrap as B
from learner import Learner,ControlNotQualified
from ledger import Ledger,H
from data import prepare
import protocol as Q
import evaluate as E
import copy,io,json,time
T=B.T;np=B.np

def save(name,value):(H/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def fixture(s,records):return T.from_numpy(np.concatenate([s.codes(r) for r in records])),T.tensor([v for r in records for v in [*r,256]])
def main():
 l=Ledger();l.charge('prior label-blind 3-arm7-site sanity before ledger creation; fully charged',3,forward_sites=21)
 s=B.sensor.Sensor();assert s.digest=='7e0b78ae71cd9f508e9069018ec2343a88ab8bfd9263067506ee6eb3fae4962b';sp,ids,manifest=prepare();save('MANIFEST.json',manifest)
 x,y=fixture(s,[b'sample']);tags=['same-source-record']*len(y)
 # CYCLE1: source parity, source arrays and explicit control-validity witness.
 l.charge('NORMAL,parent,HRES(alpha0) same-mask forward parity',3,forward_sites=21);n=Learner('NORMAL');parent=B.P.Learner();zero=Learner('HRES',alpha=0);a,ca=n.forward(x,True);b,cb=parent.forward(x,True);z,cz=zero.forward(x,True);assert T.equal(a,b) and T.equal(z,b)
 assert all(T.equal(p,q) for p,q in zip(ca,cb)) and T.equal(n.rng.get_state(),parent.rng.get_state()) and T.equal(zero.rng.get_state(),parent.rng.get_state())
 assert sum(p.numel() for p in n.p.values())==37313 and a.shape==(7,257);fan=np.bincount(n.indices.ravel(),minlength=5177);assert (fan.min(),fan.max())==(3,4)
 l.charge('harmless empty-prefix singleton norm case',1,forward_sites=1);empty=Learner('NORM_SHAM');empty.forward(x[:1]);assert empty.last['zero_denominator_sites']==empty.last['harmless_both_zero_sites']==1 and empty.last['material_norm_mismatch_sites']==0
 witnesses={}
 for name in ('zero_T_nonzero_H','gain_clip_51_not_cap_retune'):
  t=Learner('NORM_SHAM');t.p['T'].zero_();t.p['Tb'].zero_()
  if name.startswith('gain'):t.p['T'].copy_(T.eye(64)*.01)
  l.charge(name+' expected validity rejection',1,forward_sites=7)
  try:t.forward(x);raise AssertionError('invalid control silently accepted')
  except ControlNotQualified:witnesses[name]=dict(t.last)
 l.charge('NUL,255,newline byte legality without learner feedback',1,forward_sites=4);assert s.codes(b'\x00\xff\n').shape==(4,5177)
 save('CYCLE1.json',{'status':'PASS_WITH_ENFORCED_RUNTIME_CONTROL_DOMAIN','source_parity_NORMAL_and_alpha0_bitwise':True,'parameters':37313,'canonical_sensor_digest':s.digest,'same_dropout_RNG':True,'matched_control_empty_zero_is_harmless':True,'invalid_control_witnesses_rejected':witnesses,'policy':'actual nonnegligible mismatch in a future trajectory stops as CONTROL_NOT_QUALIFIED; no cap change','M1_only_all_source_and_C_ID_hashes_match':True,'technical_counts':l.write()})
 # CYCLE2: actual source-local updates and one-head output finite difference.
 n=Learner('NORMAL');p=B.P.Learner();l.charge('NORMAL/source exact next update incl moments',2,targets=14,updates=2,forward_sites=14);n.step(x,y,tags);p.step(x,y,tags,'NORMAL')
 for d in ('p','m','v'):assert all(T.equal(getattr(n,d)[k],getattr(p,d)[k]) for k in n.p)
 assert T.equal(n.rng.get_state(),p.rng.get_state())
 movements={}
 for arm in ('HRES','NORM_SHAM'):
  t=Learner(arm);l.charge(arm+' genuine observed-byte/EOS update',1,targets=7,updates=1,forward_sites=7);movements[arm]=t.step(x,y,tags)
 t=Learner('HRES');capture={};orig=t.forward
 def hook(*args,**kw):
  capture['calls']=capture.get('calls',0)+1;out=orig(*args,**kw);capture['logits']=out[0];capture['cache']=out[1];return out
 t.forward=hook;l.charge('HRES manual local gradients, exactly one mask draw',1,targets=7,forward_sites=7);g,_,_=t.gradients(x,y,tags);assert capture['calls']==1;logit=capture['logits'];xx,ah,h,at,mask,feat=capture['cache'];e=logit.softmax(1);e[T.arange(len(y)),y]-=1;e/=len(y)
 eh=(e@t.fh)*(ah>0);et=(e@t.ft)*(at>0)*mask
 assert T.equal(g['H'],T.einsum('nj,njr->jr',eh,xx)) and T.equal(g['Hb'],eh.sum(0));assert T.equal(g['T'],et.T@h) and T.equal(g['Tb'],et.sum(0))
 # Independent float64 output objective; no derivative is fed into H/T.
 l.charge('independent output-local CE reference',1,targets=7,forward_sites=7);w=t.p['O'].double().clone().requires_grad_();bias=t.p['Ob'].double().clone().requires_grad_();f=feat.detach().double();loss=T.nn.functional.cross_entropy(f@w.T+bias,y);go,gb=T.autograd.grad(loss,(w,bias));assert T.allclose(g['O'].double(),go,rtol=1e-5,atol=1e-7) and T.allclose(g['Ob'].double(),gb,rtol=1e-5,atol=1e-7)
 l.charge('output-weight central finite difference, two actual losses',2,targets=14,forward_sites=14);wp=w.detach().clone();wm=wp.clone();wp[0,0]+=1e-5;wm[0,0]-=1e-5;fd=float((T.nn.functional.cross_entropy(f@wp.T+bias.detach(),y)-T.nn.functional.cross_entropy(f@wm.T+bias.detach(),y))/2e-5);assert abs(fd-float(go[0,0]))<1e-6
 assert all(not q.requires_grad for q in t.p.values())
 save('CYCLE2.json',{'status':'PASS','NORMAL_bitwise_parent_weights_moments_RNG':True,'H_T_literal_fixed_feedback_no_skip_backprop':True,'O_local_CE_reference':True,'readout_finite_difference_error':abs(fd-float(go[0,0])),'one_actual_dropout_mask':True,'post_Adam_H_T_O_movements':movements,'technical_counts':l.write()})
 # CYCLE3: exact continuing state, actual tails, alignment and statistic rules.
 t=Learner('HRES');l.charge('moment-bearing start state',1,targets=7,updates=1,forward_sites=7);t.step(x,y,tags);st=copy.deepcopy(t.state());buf=io.BytesIO();T.save(st,buf);assert len(buf.getvalue())<=1048576
 forks=[]
 for name in ('NOEDIT','NOEDIT_duplicate','EDIT'):
  q=Learner('HRES');q.restore(T.load(io.BytesIO(buf.getvalue()),weights_only=True));assert B.statehash(q)==B.statehash(t);xr,yr=fixture(s,[b'samplx']) if name=='EDIT' else (x,y);l.charge(name+' charged continuation',1,targets=7,updates=1,forward_sites=7);q.step(xr,yr,tags);forks.append(q)
 assert B.statehash(forks[0])==B.statehash(forks[1]);assert forks[0].clock==forks[2].clock and T.equal(forks[0].rng.get_state(),forks[2].rng.get_state());assert any(not T.equal(forks[0].p[k],forks[2].p[k]) for k in t.p)
 cases=[]
 for records,arm in [([b'a'*127,b'b'*65],'HRES'),([b'c'*51],'NORM_SHAM'),([b'a'*127,b'b'*127],'HRES'),([b''],'NORM_SHAM')]:
  xx,yy=fixture(s,records);q=Learner(arm);l.charge(f'{len(yy)}-site tail/full actual whole-record fixture',len(records),targets=len(yy),updates=1,forward_sites=len(yy));start=time.process_time();info=q.step(xx,yy,['fixture']*len(yy));cases.append({'sites':len(yy),'CPU_seconds':time.process_time()-start,'readout_norm':info['readout_norm']})
 l.charge('deterministic label-blind alignment mapping numerical fixture',1);mapping=Q.derangement([4,4,1],['a','b','c']);assert mapping==Q.derangement([4,4,1],['a','b','c']);assert all(r!=o for (r,pos),o in mapping.items());assert len(mapping)==9
 l.charge('paired-record weighted bootstrap numerical canary',1);a=np.ones(64);b=np.ones(64)*2;counts=np.arange(1,65);ci=Q.paired_bootstrap(a,b,counts);assert ci['pass_upper_lt_zero'];assert not Q.paired_bootstrap(b,a,counts)['pass_upper_lt_zero'];assert ci['candidate_minus_control_pooled_bits']==float((a-b).sum()/counts.sum())
 l.charge('log-domain mean forecast underflow numerical fixture',1,targets=2);logq=E.forecast_from_summaries({'record_class_log_sums':np.full((2,257),-10000.),'record_class_site_counts':np.array([1,1])});assert np.isfinite(logq).all() and E.forecast_loss(logq,{'y':np.array([0,256])})>10000
 save('CYCLE3.json',{'status':'PASS','full_checkpoint_bytes':len(buf.getvalue()),'serialized_equal_continuation':True,'moment_bearing_fork_weights_moments_clock_RNG_equal_at_start':True,'all_tails':cases,'label_blind_hash_derangement_nonself_and_deterministic':True,'alignment_fixture_eligible_sites':len(mapping),'paired_record_bootstrap_weighted_denominators':ci,'underflow_log_mean_forecast_no_clipping':True,'main_science':'NOT_RUN','technical_counts':l.write()})
 print('THREE_CYCLES_PASS',l.write(),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  import traceback
  save('FAILURE.json',{'error':repr(e),'traceback':traceback.format_exc(),'main_science':'NOT_RUN'});raise
