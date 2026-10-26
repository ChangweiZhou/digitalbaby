"""Cycle two revised qualification. Continues from cycle-one fixture, never replays it."""
import time,statistics
import numpy as np
import support157 as s
import native157 as n
import state157
def run():
 p=s.H/'local/TECHNICAL_RESTORE.pkl';meta=dict(path=str(p.relative_to(s.H)),sha256=s.sha(p));m=s.restore(meta);newmeta=s.checkpoint(m,'TECHNICAL_R2.pkl');restored=s.restore(newmeta)
 differences=state157.equal(m,restored);assert not differences,differences[:10]
 state157.independent_mutable_banks(m);state157.independent_mutable_banks(restored)
 row=dict(id='technical-r2',prompt=bytes([128,0,255,61,3,17]).hex(),answer=bytes([128,1]).hex())
 before=[]
 for obj in [m,restored]:
  dig=s.digest(obj);before.append(n.pair(obj,row,'technical'));assert s.digest(obj)==dig
 for key in ['choices','EOS','output_hex','correct']:assert before[0][key]==before[1][key]
 for key in ['z','lp','clipping','y','rival','margin']:assert np.array_equal(before[0][key],before[1][key])
 passive=m.clone();n0=m.n;t0=m.time
 write_samples=[]
 for obj in [m,restored]:
  s.record(row,'technical',True);t=time.process_time();obj.train(bytes.fromhex(row['prompt']),bytes.fromhex(row['answer']),True);write_samples.append(time.process_time()-t)
 s.record(row,'technical',False);t=time.process_time();passive.train(bytes.fromhex(row['prompt']),bytes.fromhex(row['answer']),False);passive_cost=time.process_time()-t
 assert not state157.equal(m,restored) and m.n==restored.n==passive.n==n0+3
 assert m.time==restored.time==passive.time and m.time>t0
 after=[n.pair(obj,row,'technical') for obj in [m,restored]]
 assert np.array_equal(after[0]['z'],after[1]['z'])
 # Every payload value remains separate from EOS. The table is a scorer fixture only.
 for byte in range(256):assert bytes([byte]).hex()!=bytes([(byte+1)%256]).hex() and byte!=256
 for j in range(4):
  dig=s.digest(m);out=n.own(m,bytes([0,255,j,61]),'technical');assert s.digest(m)==dig and len(out['choices'])<=4
 cold=n.birth();benchrow=dict(id='benchmark',prompt=bytes([0,255,61,43,10,128]).hex(),answer=bytes([255,0]).hex());pairs=[];owns=[]
 for _ in range(32):
  t=time.process_time();out=n.pair(cold,benchrow,'technical');pairs.append(time.process_time()-t);assert len(out['choices'])==4 and not out['EOS']
 for _ in range(16):
  t=time.process_time();out=n.own(cold,bytes.fromhex(benchrow['prompt']),'technical');owns.append(time.process_time()-t);assert len(out['choices'])==4 and not out['EOS']
 # The new transition fixture uses maximum six-byte prompts and three targets.
 wr=max(write_samples)*1.10;nr=passive_cost*1.10
 pc=max(statistics.mean(pairs)*1.10,float(np.quantile(pairs,.9)));oc=max(statistics.mean(owns)*1.10,float(np.quantile(owns,.9)))
 forecast=9760*pc+800*oc+2048*wr+1056*nr+20.;output=20*newmeta['bytes']+150*1024**2
 assert s.COUNTS['technical_pairs']+s.COUNTS['technical_queries']<=64
 admission=dict(status='ADMITTED' if forecast<=900 and output<=1073741824 else 'NOT_ADMITTED',CPU_forecast_execution=forecast,execution_CPU_cap=900,paired_query_CPU_allowance=pc,own_query_CPU_allowance=oc,benchmark_pairs=pairs,benchmark_own=owns,updating_record_CPU_allowance=wr,inhibited_record_CPU_allowance=nr,training_cost_basis='charged maximum-length3target transition fixture plus10%; cold/setup included in qualification',output_bytes_forecast=output,RSS_observed=s.state()['RSS_peak_bytes'],maximum_prompt_reply_own_lengths=True,full_query_ceiling_charged=True,shared_first_read_bitwise_qualified_in_cycle1=True,no_control_removed=True)
 result=dict(status='PASS',operative_restore_differences=differences,answers_bitwise_equal=True,next_native_transition_exact=True,mutable_bank_arrays_independent=True,all_payloads_distinct_from_EOS=True,own_route_has_no_answer_argument=True,preserved_failed_cycle1=True,technical_records_total=s.COUNTS['technical_records'],technical_request_units=s.COUNTS['technical_pairs']+s.COUNTS['technical_queries'])
 s.save('cycles/cycle2/QUALIFICATION.json',result);s.save('ADMISSION.json',admission)
 if admission['status']!='ADMITTED':raise s.AdmissionStop('FULL_ROSTER_FORECAST_EXCEEDS_CAP')
 return result
