"""Untaught native true-prefix taps and their exact ambiguity audit."""
import numpy as np
from scipy import sparse
import support154 as s
import native154 as n

def collect(data,front):
 key=f"{front}_{data['world']}";mem=n.Native('N_2B',front);mem.bank=mem.bank[:1];arrays={k:[] for k in ['RHO','X','OBS']};ys=[];record_ids=[];positions=[];raw_groups={k:{} for k in arrays}
 for partition in ['OLD','NEW','validation']:
  for row in data[partition]:
   prompt,answer=s.unpack(row);s.record(prompt,answer,'feature');mem._prompt(prompt)
   for pos,y in enumerate(list(answer)+[256]):
    taps=n.taps(mem)
    for kind in arrays:
     v=taps[kind];raw=sparse.csr_matrix(v);raw_groups[kind].setdefault((raw.indices.tobytes(),raw.data.tobytes()),[]).append(len(ys));arrays[kind].append(sparse.csr_matrix(v/max(1.,np.linalg.norm(v))))
    ys.append(y);record_ids.append(row['id']);positions.append(pos)
    mem._event(y,None)
 result={k:sparse.vstack(v).tocsr() for k,v in arrays.items()};path=s.H/'local'/f'FEATURE_{key}.npz'
 np.savez_compressed(path,y=np.array(ys),record_ids=np.array(record_ids),positions=np.array(positions))
 for kind,x in result.items():sparse.save_npz(s.H/'local'/f'FEATURE_{key}_{kind}.npz',x)
 collisions={}
 for kind,x in result.items():
  groups={}
  for i in range(len(ys)):
   r=x.getrow(i);keyrow=(r.indices.tobytes(),r.data.tobytes());groups.setdefault(keyrow,[]).append(i)
  bad=[v for v in raw_groups[kind].values() if len({ys[i] for i in v})>1];normalized_bad=[v for v in groups.values() if len({ys[i] for i in v})>1];collisions[kind]=dict(rows=x.shape[0],dimensions=x.shape[1],nonzeros=x.nnz,exact_feature_groups=len(raw_groups[kind]),normalized_conflicting_target_groups=len(normalized_bad),conflicting_target_groups=len(bad),conflicting_rows=sum(len(v) for v in bad),examples=[dict(ids=[record_ids[i] for i in v],positions=[positions[i] for i in v],targets=[ys[i] for i in v]) for v in bad[:8]])
 s.save(f'FEATURE_{key}.json',dict(status='COMPLETE',front=front,world=data['world'],records=160,targets=len(ys),taps=collisions,cache_sha256=s.sha(path)));s.log('FEATURE_COMMIT',front=front,world=data['world'],targets=len(ys));return result,np.array(ys),record_ids

def stream_indices(data,ids):
 by={r:[i for i,z in enumerate(ids) if z==r] for r in ids};out=[];stage=[];weights=[];counter=0
 for phase in ['OLD','NEW']:
  for p in range(8):
   for ix in data['orders'][phase][p]:
    for i in by[data[phase][ix]['id']]:out.append(i);stage.append(phase);weights.append(n.parent.warmup(counter));counter+=1
 return np.array(out),np.array(weights),np.array(stage)
