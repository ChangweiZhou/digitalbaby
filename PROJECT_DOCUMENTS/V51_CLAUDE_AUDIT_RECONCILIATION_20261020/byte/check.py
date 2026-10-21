from pathlib import Path
import numpy as np,json,hashlib
ROOT=Path('/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929')
B=(((np.arange(256)[:,None]>>np.arange(8))&1)*2-1).astype(float)
A=[43,*range(48,58),61]
def groups(mapping):
 d={}
 for byte,u in mapping.items():d.setdefault(int(u),[]).append(int(byte))
 return sorted([sorted(v) for v in d.values()])
out=[]
for seed in [202610205101,202610205102]:
 w=np.random.default_rng(seed).uniform(-1,1,(64,8))
 distances=((B[A,None,:]-w[None,:,:])**2).sum(2)
 win=distances.argmin(1);m={str(b):int(u) for b,u in zip(A,win)}
 used=set(win.tolist());unused=[u for u in range(64) if u not in used]
 second=np.sort(distances,axis=1)[:,1]-distances.min(1)
 dead=distances[:,unused].min(1)-distances.min(1)
 rows=[]
 for eta in [.001,.005]:
  for gain in [0.,.01]:
   p=ROOT/f'minifly_v5_1_dev_20261020/cycles/cycle3/M2_{seed}_{eta}_{gain}.json'
   d=json.loads(p.read_text());final=d['trace'][-1]
   f=final['winner_map'];changed=[{'byte':int(b),'initial_unit':m[b],'final_unit':f[b]} for b in m if m[b]!=f[b]]
   trace=[{'bytes':t['bytes'],'reconstruction':t['type_weighted_reconstruction'],'aliases':t['alias_count'],'groups':groups(t['winner_map'])} for t in d['trace']]
   rows.append({'eta':eta,'gain':gain,'same_equivalence_groups':groups(m)==groups(f),'same_exact_units':m==f,'changed':changed,'final_groups':groups(f),'trace':trace})
 out.append({'seed':seed,'initial_map':m,'initial_groups':groups(m),'used_units':sorted(used),'distance_gap_to_second_min_median_max':[float(np.min(second)),float(np.median(second)),float(np.max(second))],'distance_gap_to_nearest_unused_min_median_max':[float(np.min(dead)),float(np.median(dead)),float(np.max(dead))],'per_byte_gaps':[{'byte':b,'second_gap':float(s),'unused_gap':float(t)} for b,s,t in zip(A,second,dead)],'settings':rows})
Path('/tmp/v51_claude_audit_byte/RESULT.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
