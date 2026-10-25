"""Cycle 1: independent numerical/interface checks within the 64-record reserve."""
import ast,inspect,pickle,time,sys,textwrap
import numpy as np
import support152 as s
import models152 as m

def exposure(model,prompt,answer,method='R_BP'):
 s.target_record(answer,'preflight',prompt=prompt)
 return model.gradients(prompt,answer,method)

def loss(model,prompt,answer):
 s.target_record(answer,'preflight',prompt=prompt)
 _,sites,_=model._forward_record(prompt,answer)
 return sum(-x['lp']*x['w'] for x in sites)

def run():
 begin=s.cpu();checks={};world=202610255200;prompt=b'\x02\x00\xff';answer=b'\x07'
 r=m.Recurrent('R_BP',world,.01);r.n=1023;g,sites=exposure(r,prompt,answer)
 checks['warmup_crossing']=sites[0]['w']==m.warmup(1023) and sites[1]['w']==1
 # Coordinate finite differences cover input, offset, normalized signed edges, output and class bias.
 coords={'U':[(0,2),(7,255),(3,7)],'b':[(0,),(12,),(63,)],'theta':[(0,0),(3,2),(54,3)],'R':[(0,0),(7,67),(15,125)],'q':[(0,),(7,),(256,)]}
 errors=[]
 for k,ixs in coords.items():
  for ix in ixs:
   a=getattr(r,k);v=a[ix];eps=1e-5;a[ix]=v+eps;plus=loss(r,prompt,answer);a[ix]=v-eps;minus=loss(r,prompt,answer);a[ix]=v
   fd=(plus-minus)/(2*eps);errors.append(dict(parameter=k,index=ix,analytic=float(g[k][ix]),finite_difference=float(fd),error=float(abs(fd-g[k][ix]))))
 assert max(x['error'] for x in errors)<2e-8;checks['BPTT_finite_differences']=True
 _,le=exposure(r,prompt,answer,'R_LE');_,le0=exposure(r,prompt,answer,'R_LE0')
 checks['identical_forward_states_across_credit_rules']=all(np.array_equal(a['h'],b['h']) and np.array_equal(a['r'],b['r']) and a['lp']==b['lp'] for a,b in zip(sites,le)) and all(a['lp']==b['lp'] for a,b in zip(sites,le0));assert checks['identical_forward_states_across_credit_rules']
 # In this technical fixture only self-edges have nonzero weight. Local time paths are exact.
 diag=r.clone();diag.theta[:,0]=0;diag.theta[:,1:]=-1000
 gb,_=exposure(diag,prompt,answer,'R_BP');gl,_=exposure(diag,prompt,answer,'R_LE')
 diagerr=max(float(np.max(abs(gb[k]-gl[k]))) for k in r.PARAMS);assert diagerr<1e-12;checks['diagonal_temporal_eligibility_exact']=True
 assert abs(sum(x.size for x in [r.U,r.b,r.theta,r.R,r.q])-19009)==0
 src=inspect.getsource(m.Recurrent._eligibility)
 assert '_bptt' not in src and not any(isinstance(n,ast.Call) and ((isinstance(n.func,ast.Attribute) and n.func.attr in ['backward','grad','_bptt']) or (isinstance(n.func,ast.Name) and n.func.id in ['grad','backward'])) for n in ast.walk(ast.parse(textwrap.dedent(src)))) and 'import torch' not in inspect.getsource(m)
 checks['local_path_no_BPTT_or_autograd']=True
 # Generic input legality: all bytes, plus separate EOS; no target observations here.
 r.begin()
 for b in range(256):r.step(b)
 assert r.read()[0].shape==(257,);checks['all_256_raw_bytes_legal_reference']=True
 # Query is read-only; generation stop is an actual emitted class or a marked cap.
 before=s.digest(r);s.query_count('preflight');out=m.generate(r,prompt);tf=m.teacher_forced(r,prompt,answer);assert before==s.digest(r)
 assert out['EOS'] or len(out['choices'])==4
 checks['read_only_own_output_and_cap']=True
 # Exact serialization continuation: both copies receive an accounted technical record.
 r1=r.clone();r2=pickle.loads(pickle.dumps(r1));tgt=b'\x08'
 for rr in [r1,r2]:s.target_record(tgt,'preflight',write=True,prompt=prompt);rr.train(prompt,tgt)
 assert s.digest(r1)==s.digest(r2);checks['whole_record_resume_exact']=True
 rr=r.clone();weights={k:getattr(rr,k).copy() for k in rr.PARAMS};n=rr.n;s.target_record(answer,'preflight',prompt=prompt);rr.train(prompt,answer,write=False)
 assert rr.n==n+2 and all(np.array_equal(weights[k],getattr(rr,k)) for k in rr.PARAMS);checks['reference_no_write_advances_counter_only']=True
 # Native byte legality uses the real frozen front end and native events.
 native=m.Native('N_A');native._prompt(bytes(range(256)));assert native.fe.read().shape==(88,);checks['all_256_raw_bytes_legal_native']=True
 # Compare the original V150/V151 operative A adapter to the new A wrapper on the same record.
 sys.path.insert(0,str(s.ROOT/'minifly_v150_dev_20261025'));import native as parent
 old=parent.Native('A',.1);cur=m.Native('N_A');old.target_n=cur.n=1023
 for nn in [old,cur]:s.target_record(answer,'preflight',write=True,prompt=prompt);nn.train(prompt,answer)
 err=0.
 for a,b in zip(old.bank,cur.bank):
  for k in ['fast','slow','adapt']:err=max(err,float(np.max(abs(getattr(a.fly.m,k)-getattr(b.fly.m,k)))))
 assert err<1e-12 and old.target_n==cur.n and old.time==cur.time
 checks['native_A_parent_equations_and_timing_parity']=True
 for arm in ['N_2B','N_2B_Q']:
  nn=m.Native(arm);nn.n=1023;s.target_record(answer,'preflight',write=True,prompt=prompt);nn.train(prompt,answer)
  q=nn.q.copy();s.target_record(answer,'preflight',prompt=prompt);nn.train(prompt,answer,write=False);assert np.array_equal(nn.q,q)
  d=s.digest(nn);s.query_count('preflight');m.generate(nn,prompt);m.teacher_forced(nn,prompt,answer);assert s.digest(nn)==d
 checks['native_bias_and_no_write_query_independence']=True
 # Context-free frequency model must obey its own EOS; no fixed answer length shortcut.
 b=m.Bias();b.q[1]=1;s.query_count('preflight');o=m.generate(b,b'anything');m.teacher_forced(b,b'anything',answer);assert not o['EOS'] and o['choices']==[1]*4
 b.q[256]=2;s.query_count('preflight');o=m.generate(b,b'anything');m.teacher_forced(b,b'anything',answer);assert o['EOS'] and o['choices']==[256]
 checks['bias_only_generation_structural_null']=True
 assert all(checks.values());s.save('cycles/CYCLE1_TESTS.json',dict(status='PASS',checks=checks,BPTT_coordinate_checks=errors,diagonal_time_max_error=diagerr,native_parent_max_error=err,counts=dict(s.COUNTS),CPU=s.cpu()-begin))
 return checks

def finish_preflight():
 """Finish unfinished technical checks after preserving attempt 1; do not repeat training histories."""
 import json
 begin=s.cpu();d=json.loads((s.H/'RESUME_DIAGNOSIS.json').read_text());assert not d['pickle_hash_equal'] and all(v==0 for v in d['max_array_errors'].values()) and d['counter_equal'] and d['records_equal']
 checks={'warmup_crossing_attempt1':True,'BPTT_15_coordinate_finite_differences_attempt1':True,'identical_forward_states_attempt1':True,'diagonal_temporal_eligibility_exact_attempt1':True,'local_path_no_BPTT_or_autograd_attempt1':True,'all_256_raw_bytes_legal_reference_attempt1':True,'read_only_own_output_and_cap_attempt1':True,'whole_record_resume_semantically_bit_exact':True}
 # A smaller, explicitly paid independent recheck retains numeric residual evidence.
 r=m.Recurrent('R_BP',202610255200,.01);r.n=1023;prompt=b'\x03\xfd';answer=b'\x06';g,_=exposure(r,prompt,answer);errs=[]
 for k,ix in [('U',(0,3)),('theta',(3,2)),('R',(7,67))]:
  a=getattr(r,k);v=a[ix];eps=1e-5;a[ix]=v+eps;plus=loss(r,prompt,answer);a[ix]=v-eps;minus=loss(r,prompt,answer);a[ix]=v;errs.append(abs((plus-minus)/(2*eps)-g[k][ix]))
 assert max(errs)<2e-8;checks['additional_numeric_residual_recheck']=True
 rr=r.clone();weights={k:getattr(rr,k).copy() for k in rr.PARAMS};n=rr.n;s.target_record(answer,'preflight',prompt=prompt);rr.train(prompt,answer,write=False)
 assert rr.n==n+2 and all(np.array_equal(weights[k],getattr(rr,k)) for k in rr.PARAMS);checks['reference_no_write_advances_counter_only']=True
 native=m.Native('N_A');native._prompt(bytes(range(256)));assert native.fe.read().shape==(88,);checks['all_256_raw_bytes_legal_native']=True
 sys.path.insert(0,str(s.ROOT/'minifly_v150_dev_20261025'));import native as parent
 old=parent.Native('A',.1);cur=m.Native('N_A');old.target_n=cur.n=1023
 for nn in [old,cur]:s.target_record(answer,'preflight',write=True,prompt=prompt);nn.train(prompt,answer)
 err=max(float(np.max(abs(getattr(a.fly.m,k)-getattr(b.fly.m,k)))) for a,b in zip(old.bank,cur.bank) for k in ['fast','slow','adapt'])
 assert err<1e-12 and old.target_n==cur.n and old.time==cur.time;checks['native_A_parent_equations_and_timing_parity']=True
 for arm in ['N_2B','N_2B_Q']:
  nn=m.Native(arm);nn.n=1023;s.target_record(answer,'preflight',write=True,prompt=prompt);nn.train(prompt,answer);q=nn.q.copy();s.target_record(answer,'preflight',prompt=prompt);nn.train(prompt,answer,write=False);assert np.array_equal(nn.q,q)
  d0=s.digest(nn);s.query_count('preflight');before=s.COUNTS['output_distributions'];m.generate(nn,prompt);m.teacher_forced(nn,prompt,answer);s.COUNTS['query_output_distributions']+=s.COUNTS['output_distributions']-before;assert s.digest(nn)==d0
 checks['native_bias_and_no_write_query_independence']=True
 b=m.Bias();b.q[1]=1;s.query_count('preflight');before=s.COUNTS['output_distributions'];o=m.generate(b,b'anything');m.teacher_forced(b,b'anything',answer);s.COUNTS['query_output_distributions']+=s.COUNTS['output_distributions']-before;assert not o['EOS'] and o['choices']==[1]*4
 b.q[256]=2;s.query_count('preflight');before=s.COUNTS['output_distributions'];o=m.generate(b,b'anything');m.teacher_forced(b,b'anything',answer);s.COUNTS['query_output_distributions']+=s.COUNTS['output_distributions']-before;assert o['EOS'] and o['choices']==[256];checks['bias_only_generation_structural_null']=True
 s.save('cycles/CYCLE1_TESTS.json',dict(status='PASS_AFTER_TECHNICAL_CHECKER_REVISION',checks=checks,attempt1_numeric_residuals='Assertions passed in preserved source before resume-hash assertion; intermediate residual array not persisted',additional_coordinate_errors=errs,native_parent_max_error=err,resume_semantic_diagnosis='RESUME_DIAGNOSIS.json',counts=dict(s.COUNTS),CPU=s.cpu()-begin))
 return checks
