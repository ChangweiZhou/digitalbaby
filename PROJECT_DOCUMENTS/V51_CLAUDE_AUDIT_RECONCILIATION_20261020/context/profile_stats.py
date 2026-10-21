import ast,json,hashlib,time
from pathlib import Path
import numpy as np
R=Path('/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929')
P=Path('/tmp/v51_claude_audit_context_20261020')
t=time.process_time();tree=ast.parse((R/'minifly_v5_1_dev_20261020/data.py').read_text());tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom) and n.module=='v51_common')];env={'np':np};exec(compile(tree,'evaluator_data','exec'),env)
rows=[]
for seed in (202610205101,202610205102):
 records=env['stream'](seed,6000,'train');used=[];n=0
 for p,a in records:
  if n+len(p+a)>50000:break
  used.append(p+a);n+=len(p+a)
 alphabet=sorted(set(b''.join(used)));ix={b:i for i,b in enumerate(alphabet)};counts=np.zeros((12,12))
 for s in used:
  for a,b in zip(s[:-1],s[1:]):counts[ix[a],ix[b]]+=1
 profile=np.c_[counts.T/np.maximum(1,counts.T.sum(1))[:,None],counts/np.maximum(1,counts.sum(1))[:,None]]
 digits=[ix[b] for b in alphabet if 48<=b<=57];ops=[ix[b] for b in alphabet if b in (43,61)]
 dd=[float(np.sum((profile[a]-profile[b])**2)) for i,a in enumerate(digits) for b in digits[i+1:]];do=[float(np.sum((profile[a]-profile[b])**2)) for a in digits for b in ops]
 # Evaluator-only farthest-first Lloyd fit, no class labels provided.
 centers=[0]
 while len(centers)<3:
  distances=((profile[:,None,:]-profile[np.array(centers)][None,:,:])**2).sum(2).min(1)
  centers.append(int(np.argmax(distances)))
 mu=profile[centers].copy()
 for _ in range(30):
  labels=np.argmin(((profile[:,None,:]-mu[None,:,:])**2).sum(2),axis=1)
  new=np.array([profile[labels==j].mean(0) if np.any(labels==j) else mu[j] for j in range(3)])
  if np.allclose(new,mu):break
  mu=new
 rows.append({'seed':seed,'payload_bytes':n,'mean_digit_digit_squared_L2':float(np.mean(dd)),'mean_digit_operator_squared_L2':float(np.mean(do)),'ratio':float(np.mean(do)/np.mean(dd)),'evaluator_only_three_cluster_groups':[[chr(alphabet[i]) for i in range(12) if labels[i]==j] for j in range(3)]})
out={'scope':'Offline profile statistics and evaluator-only three-cluster fit; no M3/native learning','rows':rows,'CPU_seconds':time.process_time()-t};(P/'PROFILE_STATS.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
