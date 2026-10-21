import ast,hashlib,json,time,resource,signal
from pathlib import Path
import numpy as np
ROOT=Path('/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929')
OUT=Path('/tmp/v51_claude_audit_context_20261020')
start=time.process_time();results=[]
resource.setrlimit(resource.RLIMIT_CPU,(30,32))
def stop(sig,frm):
 (OUT/'RESULTS.json').write_text(json.dumps({'status':'CPU_CAP','results':results,'CPU_seconds':time.process_time()-start},indent=2));raise SystemExit(0)
signal.signal(signal.SIGXCPU,stop)
def loaded(path,name):
 tree=ast.parse(path.read_text())
 tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom) and n.module in ('v51_common','numba'))]
 env={'np':np,'njit':lambda **kw:lambda f:f,'__name__':name}
 exec(compile(tree,str(path),'exec'),env)
 return env
rep=loaded(ROOT/'minifly_v5_1_dev_20261020/representation.py','counterfactual_representation')
dat=loaded(ROOT/'minifly_v5_1_dev_20261020/data.py','counterfactual_data')
Codes=rep['Codes'];BITS=rep['BITS']
for seed in (202610205101,202610205102):
 records=dat['stream'](seed,6000,'train');selected=[];total=0
 for p,a in records:
  if total+len(p)+len(a)>50000:break
  selected.append((p,a));total+=len(p)+len(a)
 alphabet=sorted(set(b''.join(p+a for p,a in selected)))
 for multiplier in (1,2):
  c=Codes(seed)
  # Test-only oracle byte identity, no native writes or efficacy experiment.
  for u,b in enumerate(alphabet):c.w[u]=BITS[b]
  c.identity_frozen=True
  assert len(set(c.table()[alphabet]))==len(alphabet)
  c.class_proto*=multiplier
  initial_sums=c.class_proto.sum(1).tolist()
  unit_ids=[c.winner(b) for b in alphabet]
  t0=time.process_time();c.hear_context(selected)
  profiles=c.profiles();occupied=sorted(set(c.classes[unit_ids].tolist()))
  grouping={str(k):[chr(b) for b in alphabet if int(c.classes[c.winner(b)])==k] for k in occupied}
  row={'seed':seed,'prototype_initial_sum':multiplier,'heard_payload_bytes':total,'complete_records':len(selected),'occupied_classes':occupied,'grouping':grouping,'initial_proto_sums':initial_sums,'final_proto_sums':c.class_proto.sum(1).tolist(),'seen_profile_sums':profiles[unit_ids].sum(1).tolist(),'CPU_seconds':time.process_time()-t0}
  results.append(row);(OUT/'PROGRESS.json').write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
 # Quantify profile distances independently of class assignments.
 digits=[c.winner(b) for b in alphabet if 48<=b<=57];ops=[c.winner(b) for b in alphabet if b in (43,61)]
 dd=[float(np.linalg.norm(profiles[u]-profiles[v])) for i,u in enumerate(digits) for v in digits[i+1:]]
 do=[float(np.linalg.norm(profiles[u]-profiles[v])) for u in digits for v in ops]
 results.append({'seed':seed,'mean_digit_digit_profile_L2':float(np.mean(dd)),'mean_digit_operator_profile_L2':float(np.mean(do)),'metric':'Euclidean distance; oracle identity control, not a learned semantic classifier'})
result={'status':'COMPLETE','scope':'Synthetic perfect identity counterfactual; unchanged M3 vs initialization normalization only. No native value writes, no arithmetic efficacy, no new candidate implemented.','source_sha256':hashlib.sha256((ROOT/'minifly_v5_1_dev_20261020/representation.py').read_bytes()).hexdigest(),'results':results,'CPU_seconds':time.process_time()-start}
(OUT/'RESULTS.json').write_text(json.dumps(result,indent=2));print('DONE',result['CPU_seconds'])
