"""Cycle-one formula, IO, control and mutation fixtures within the technical reserve."""
import numpy as np
from scipy import sparse
import support156 as s
import native156 as n
import feasibility156 as f
import audit156 as audit

def run():
 checks=[];X=sparse.eye(2,format='csr');y=np.asarray([17,256]);w=np.zeros((257,2));w[y,[0,1]]=4;b=np.zeros(257)
 good=f.verify(X,y,w,b,'FQ');assert good['valid'];checks.append('sound_positive_fixture')
 changed=w.copy();changed[17,0]=-4;assert not f.verify(X,y,changed,b,'FQ')['valid'];checks.append('mutated_witness_rejected')
 xx=sparse.csr_matrix([[1.,0.],[1.,0.]]);bad=f.duplicate_certificate(xx,np.asarray([17,256]));assert f.check_duplicate(xx,np.asarray([17,256]),bad)
 wrong={**bad,'lambda_values':[1,-1]};assert not f.check_duplicate(xx,np.asarray([17,256]),wrong);checks.append('exact_negative_certificate_and_mutation')
 ambiguous=f.verify(X,y,np.zeros((257,2)),b,'FQ');assert not ambiguous['valid'];checks.append('zero_or_approximate_margin_not_accepted')
 assert audit.independent_witness(X,y,w,b,'FQ',n.C)['valid'] and not audit.independent_witness(X,y,changed,b,'FQ',n.C)['valid'];checks.append('independent_verifier_mutation')
 neg=audit.negative_fixtures()
 # Raw-key scorer fixture over all payload byte values; no native training/records.
 for byte in range(256):assert audit.response([byte,256],bytes([byte]))[0]
 checks.append('all_256_payloads_distinct_from_EOS_scorer')
 model=n.n.Native('N_2B','I_FAST');prompt=bytes([0,255,61,43,10]);row=dict(id='technical',prompt=prompt.hex(),answer=bytes([255,0]).hex())
 before=s.digest(model)
 for i in range(4):
  observed=model.clone();idle=model.clone();prior_n=model.n;prior_t=model.time
  s.record(row,'technical',True);sites=model.train(prompt,bytes([255,0]),True)
  s.record(row,'technical',False);sites0=observed.train(prompt,bytes([255,0]),False)
  s.COUNTS['technical_training_distributions']+=len(sites)+len(sites0)
  n.n.fastmem.FastMemory.rest(idle,model.time-prior_t)
  assert model.n==observed.n==prior_n+3 and idle.n==prior_n
  assert abs(idle.time-model.time)<1e-10 and abs(observed.time-model.time)<1e-10
  assert abs(sites[0]['warmup']-n.n.parent.warmup(prior_n))<1e-15 and not model.q.any()
  # Replicated no-write path preserves the matched non-plastic trajectory.
  for actual,passive in zip(model.bank,observed.bank):assert np.array_equal(actual.fly.m.adapt,passive.fly.m.adapt)
  view=model.clone();view._prompt(prompt);ref=view.read()[1];z,lp,cl=n.read(view);assert np.max(abs(ref-lp))<1e-12
  s.COUNTS['technical_reference_read_distributions']+=1;s.COUNTS['evaluator_distributions']+=1
  dig=s.digest(model);out=n.pair(model,row,'technical');assert s.digest(model)==dig
  own=out['choices'];yy=out['y']
  for j in range(min(len(own),len(yy))):
   if own[:j]==yy[:j].tolist():assert own[j]==int(out['z'][j].argmax())
  saved=s.checkpoint(model,f'TECHNICAL_{i}.pkl');assert saved['bytes']>0
 checks.extend(['gradual_warmup_not_reset','q_stays_zero','signed_teach_local_native_path','matched_passive_observations_and_clocks','idle_seconds_not_steps','read_and_clone_immutability','own_teacher_prefix_agreement','read_equation_parity','checkpoint_logger_path'])
 assert s.digest(model)!=before
 s.save('cycles/cycle1/QUALIFICATION.json',dict(status='PASS',checks=checks,negative_fixtures=neg,technical_records=8,technical_targets=24,technical_query_pairs=4,counters=s.snapshot_counts()))
 return dict(status='PASS',checks=checks)
