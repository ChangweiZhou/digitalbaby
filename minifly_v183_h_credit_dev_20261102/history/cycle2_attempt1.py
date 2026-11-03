"""Bounded numeric and optimizer tests; every fixture is charged before use."""
import bootstrap as B
from learner import Learner,base_gradients,paired,hcredit,proposals,hnorm
from ledger import H,Ledger
import json,copy,io,time,resource,sys
T=B.T;np=B.np

def save(n,v):(H/n).write_text(json.dumps(v,indent=2)+'\n')
def continuing_ledger():
 old=json.loads((H/'EXPOSURE_LEDGER.json').read_text());l=Ledger.__new__(Ledger);l.path=H/'EXPOSURE_LEDGER.json';l.start=time.process_time()-(old['charged_CPU_seconds']-45);l.wall=time.time()-(old['wall_seconds_including_300_setup_allowance']-300);l.rows=old['rows'];return l

def main():
 ledger=continuing_ledger();s=B.sensor.Sensor();x=T.from_numpy(s.codes(b'012345'));y=T.tensor([*b'012345',256]);ids=['fixture']*7;net=Learner();parent=B.P.Learner();a=copy.deepcopy(net.state())
 ledger.add('NORMAL next update candidate/source bit parity',n=7,paths=1,base_records=2,updates=2,loss_records=2)
 parent.step(x,y,ids,'NORMAL');net.step(x,y,ids,'NORMAL')
 for d in ('p','m','v'):
  assert all(T.equal(getattr(parent,d)[k],getattr(net,d)[k]) for k in net.p),(d,'source parity')
 assert T.equal(parent.rng.get_state(),net.rng.get_state())
 net.restore(a);ledger.add('one base path and per-site one-pair loss, frozen mask/weights',n=7,paths=3,base_records=1)
 g,cache=base_gradients(net,x,y);noise=T.randn((7,64),dtype=T.float64,generator=T.Generator().manual_seed(183));after_base=B.statehash(net);q,sigma,losses=paired(net,cache,y,noise);assert B.statehash(net)==after_base
 assert noise.shape==(7,64) and not T.equal(noise[0],noise[1]) and sigma==.01*max(float(cache[1].double().square().mean().sqrt()),1e-3)
 # An independent NumPy downstream implementation checks natural-log sign,
 # per-site assignment, frozen sigma and actual inverted dropout.
 ledger.add('independent NumPy paired loss reference',n=7,paths=2,base_records=1)
 wt,bt,wo,bo=[net.p[k].numpy().astype(float) for k in ('T','Tb','O','Ob')];drop=cache[4].numpy().astype(float);ah=cache[1].numpy().astype(float);yy=y.numpy()
 def nloss(a):
  log=np.maximum(0,np.maximum(0,a)@wt.T+bt)*drop;log=log@wo.T+bo;log-=log.max(1,keepdims=True);return np.log(np.exp(log).sum(1))-log[np.arange(len(y)),yy]
 ref=(nloss(ah+sigma*noise.numpy())-nloss(ah-sigma*noise.numpy()))/(2*sigma);qerr=float(np.max(np.abs(ref-q.numpy())));assert qerr<1e-9
 gh=hcredit(cache[0],q,noise);delta=q[:,None]*noise/7;assert T.allclose(gh['H'].double(),T.einsum('nj,njr->jr',delta,cache[0].double()),rtol=1e-6,atol=1e-7)
 assert T.allclose(gh['Hb'].double(),delta.sum(0),rtol=1e-6,atol=1e-7)
 assert float((delta*(cache[1]>0)-delta).abs().max())>1e-8 # no extra ReLU derivative
 # Independent exact derivatives are instruments, never assigned to learner.
 ledger.add('exact read-only instrument and finite differences at fixed sigma',n=7,paths=5,base_records=2,loss_records=5)
 exact_a=cache[1].detach().double().requires_grad_();true=T.autograd.grad(losses(exact_a).mean(),exact_a)[0];max_fd=0.
 for row,col in [(0,0),(3,25)]:
  original=float(exact_a[row,col]);plus=exact_a.detach().clone();minus=plus.clone();plus[row,col]=original+1e-5;minus[row,col]=original-1e-5;fd=float((losses(plus).mean()-losses(minus).mean())/2e-5);max_fd=max(max_fd,abs(fd-float(true[row,col])))
 assert max_fd<1e-6 and all(not p.requires_grad for p in net.p.values())
 # T/O rules and their committed moments match the NORMAL source at common state.
 common={}
 for arm in ('HPC','SHUFFLED'):
  t=Learner();t.restore(a);ledger.add(arm+' same-state real update',n=7,paths=3,base_records=1,updates=1);info=t.step(x,y,ids,arm)
  for k in ('T','Tb','O','Ob'):
   assert T.equal(t.p[k],parent.p[k]) and T.equal(t.m[k],parent.m[k]) and T.equal(t.v[k],parent.v[k]),(arm,k)
  if arm=='SHUFFLED':assert abs(info['applied_proposed_H_energy']-info['HPC_counterfactual_H_energy'])<=max(1e-18,info['HPC_counterfactual_H_energy']*2e-7)
  common[arm]=info
 # Serializing both independent generators preserves next update; this is two
 # deliberately charged fixture updates, not a free duplicate reference run.
 b=Learner();b.restore(a);buf=io.BytesIO();T.save(b.state(),buf);assert len(buf.getvalue())<=1048576;c=Learner();c.restore(T.load(io.BytesIO(buf.getvalue()),weights_only=True));assert B.statehash(b)==B.statehash(c)
 ledger.add('serialized HPC continuation both executions',n=7,paths=6,base_records=2,updates=2,loss_records=6)
 b.step(x,y,ids,'HPC');c.step(x,y,ids,'HPC');assert B.statehash(b)==B.statehash(c)
 # Singleton and actual-size full/tail batches: synthetic fixed feature rows,
 # all target supervision and repeated references charged.
 cases=[]
 for n,arm in [(1,'SHUFFLED'),(194,'HPC'),(256,'HPC')]:
  rows=x[T.arange(n)%7];labels=y[T.arange(n)%7];t=Learner();t.restore(a);ledger.add(f'n={n} {arm} fixture',n=n,paths=3,base_records=max(1,(n+128)//129),updates=1);start=time.process_time();info=t.step(rows,labels,['fixture']*n,arm);cases.append({**info,'CPU_seconds':time.process_time()-start})
  if n==1:assert info['shuffle_shift']==0
 # Zero-credit cannot freeze the whole learner: H moments/decay continue;
 # nonzero output error still writes Ob. No new downstream evaluations here.
 ledger.add('zero-H-credit ordinary moment/decay and invalid sham arithmetic',n=0,paths=1,base_records=1,loss_records=0)
 z=Learner();gg={k:T.zeros_like(v) for k,v in z.p.items()};gg['Ob'].fill_(1);m,v,d=proposals(z,gg);assert d['H'].abs().sum()>0 and d['Ob'].abs().sum()>0
 assert float(hnorm({k:T.zeros_like(z.p[k]) for k in ('H','Hb')}))==0
 save('CYCLE2.json',{'status':'PASS','NORMAL_bitwise_source_update_and_moments':True,'T_O_bitwise_source_parity':True,'independent_paired_loss_max_error':qerr,'fixed_sigma_exact_reference_finite_difference_max_error':max_fd,'extra_H_ReLU_factor_absent':True,'per_site_noise_and_loss_not_batch_mean':True,'loss_path_preserves_weights_and_mask_RNG':True,'no_gradient_crosses_learned_layers_in_candidate':True,'shuffled_post_Adam_bias_decay_strength_matched':True,'common_state_proposals':common,'n1_and_full_batch_tail_cases':cases,'checkpoint_bytes_with_two_extra_generators':len(buf.getvalue()),'serialized_next_update_equality':True,'all_scientific_training':'NOT_RUN','counts':ledger.write()})
 print('CYCLE2_PASS',ledger.write(),flush=True)
if __name__=='__main__':main()
