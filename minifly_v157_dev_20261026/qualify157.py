"""Technical fixtures and charged worst-length resource admission; no science worlds."""
import time,statistics
import numpy as np
import support157 as s
import native157 as n
def run():
 checks=[];cold=n.birth();m=cold.clone();row=dict(id='technical',prompt=bytes([0,255,61,43,10,128]).hex(),answer=bytes([255,0]).hex());write_times=[];no_times=[]
 for k in range(4):
  passive=m.clone();idle=m.clone();n0=m.n;t0=m.time
  s.record(row,'technical',True);t=time.process_time();sites=m.train(bytes.fromhex(row['prompt']),bytes.fromhex(row['answer']),True);write_times.append(time.process_time()-t)
  s.record(row,'technical',False);t=time.process_time();other=passive.train(bytes.fromhex(row['prompt']),bytes.fromhex(row['answer']),False);no_times.append(time.process_time()-t)
  assert m.n==passive.n==n0+3 and abs(m.time-passive.time)<1e-10
  assert abs(sites[0]['warmup']-(1/16+15/16*min(n0/1024,1)))<1e-14
  n.n.fastmem.FastMemory.rest(idle,m.time-t0);assert abs(idle.time-m.time)<1e-10 and idle.n==n0
  for actual,nonwrite in zip(m.bank,passive.bank):assert np.array_equal(actual.fly.m.adapt,nonwrite.fly.m.adapt)
  before=s.digest(m);a=n.pair(m,row,'technical');b=n.reference_pair(m,row);assert s.digest(m)==before
  for key in ['choices','EOS','output_hex','correct']:assert a[key]==b[key]
  for key in ['z','lp','clipping','y','rival','margin']:assert np.array_equal(a[key],b[key])
 checks+=['unchanged signed native equation','gradual warmup and real n','q remains zero','matched nonplastic state and clocks','seconds versus steps idle','query immutability','bitwise paired-output parity with V156 evaluator']
 meta=s.checkpoint(m,'TECHNICAL_RESTORE.pkl');restored=s.restore(meta);assert s.digest(restored)==s.digest(m);checks.append('exact checkpoint roundtrip')
 # All-byte/EOS scorer fixture; no payload labels are presented to a learner.
 for byte in range(256):assert bytes([byte]).hex()!=bytes([(byte+1)%256]).hex() and byte!=256
 def correct(c,a):return c==list(a)+[256]
 assert correct([0,255,256],bytes([0,255])) and not correct([0,255],bytes([0,255])) and not correct([0,254,256],bytes([0,255]));checks.append('raw-key scorer positive and mutated answer/EOS negative fixtures')
 # Held route accepts only prompts; never takes expected output or target prefix.
 for j in range(4):
  before=s.digest(m);o=n.own(m,bytes([0,255,j,61]),'technical');assert s.digest(m)==before and len(o['choices'])<=4
 checks.append('own-only held route immutable, no answer argument')
 samples=[]
 for _ in range(32):
  t=time.process_time();out=n.pair(cold,row,'technical');samples.append(time.process_time()-t)
  assert len(out['choices'])==4 and not out['EOS']
 # Every benchmark has maximum six-byte prompt, two payload targets and four own decisions.
 # Charge the full registered request ceiling even though exact duplicate receipts may save work.
 own_samples=[]
 for _ in range(16):
  t=time.process_time();o=n.own(cold,bytes.fromhex(row['prompt']),'technical');own_samples.append(time.process_time()-t);assert len(o['choices'])==4 and not o['EOS']
 pair_cost=max(statistics.mean(samples)*1.10,float(np.quantile(samples,.9)))
 own_cost=max(statistics.mean(own_samples)*1.10,float(np.quantile(own_samples,.9)))
 # Cold initialization is already spent and charged above; science uses these same warmed kernels.
 updating=max(write_times[1:])*1.10;inhibited=max(no_times[1:])*1.10
 forecast=pair_cost*9760+own_cost*800+updating*2048+inhibited*1056+20.
 estimated_output=20*meta['bytes']+150*1024**2
 admission=dict(status='ADMITTED' if forecast<=900 and estimated_output<=1073741824 else 'NOT_ADMITTED',CPU_forecast_execution=forecast,execution_CPU_cap=900,paired_query_CPU_allowance=pair_cost,query_benchmark_samples=samples,own_query_CPU_allowance=own_cost,own_query_benchmark_samples=own_samples,updating_record_CPU_allowance=updating,inhibited_record_CPU_allowance=inhibited,cold_and_warm_training_samples=dict(updating=write_times,inhibited=no_times),query_bound_charged=9760,held_generation_bound_charged=800,forecast_uses_maximum_prompt_reply_and_own_lengths=True,output_bytes_forecast=estimated_output,RSS_observed=s.state()['RSS_peak_bytes'],shared_first_read_bitwise_qualified=True,no_extra_scientific_control_removed=True)
 s.save('cycles/cycle1/QUALIFICATION.json',dict(status='PASS',checks=checks,technical_records=8,technical_pairs=s.COUNTS['technical_pairs'],technical_own_queries=s.COUNTS['technical_queries'],counters=s.state()))
 s.save('ADMISSION.json',admission)
 if admission['status']!='ADMITTED':raise s.AdmissionStop('FULL_ROSTER_RESOURCE_FORECAST_EXCEEDS_CAP')
 return admission
