"""Read fixed anatomical routes and saved supports, with zero teaching calls."""
from rc_common import *
from v51_common import fresh_bank
b=fresh_bank(5177,0);ds=np.load(P/'assets/donors.npy');rows=[]
for f in (P/'assets').glob('*_1_addresses.npz'):
 N=int(f.name.split('_')[0]);sides=b.kc_side if N==5177 else b.kc_side[ds];Q=b.Q if N==5177 else np.load(P/'assets/expanded_Q.npy');T=b.T if N==5177 else b.T[ds]
 with np.load(f) as z:x=z['x'].astype(float)
 for slot in range(9):
  for side in (0,1):
   mask=sides==side;a=x[:,slot,mask];freq=a.mean(0);gram=a@a.T;w=(abs(Q).sum(1)*abs(T).sum(1))[mask];wg=(a*w)@a.T
   oldnew=gram[:18,18:36];within=gram[:18,:18];ix=~np.eye(18,dtype=bool)
   den=a.sum(1)[:,None]+a.sum(1)[None,:]-gram
   rows.append(dict(asset=f.name,N=N,slot=slot,side=side,active_counts=a.sum(1).tolist(),inclusion_square_sum=float(freq@freq),effective_N=float(freq.sum()**2/(freq@freq)),dead_on_these_queries=int((freq==0).sum()),old_new_overlap=float(oldnew.mean()),old_old_overlap=float(within[ix].mean()),old_new_jaccard=float((gram/np.maximum(den,1))[:18,18:36].mean()),weighted_old_new_overlap=float(wg[:18,18:36].mean()),identical_old_new_codes=int((oldnew==a[:18].sum(1)[:,None]).sum()) if np.all(a.sum(1)==a[0].sum()) else None))
save('SUPPORT_MEASUREMENTS.json',rows)

from scipy.sparse import csr_matrix,load_npz
M=csr_matrix((np.ones(len(b.pn_type_index)),(b.pn_type_index,np.arange(len(b.pn_type_index)))),shape=(88,len(b.pn_type_index)))
duplicates={}
for N,B in ((5177,b.B),(20708,load_npz(P/'assets/expanded_B.npz'))):
 effective=(M@B).tocsc();seen=set();dup=0;dead=0
 for j in range(N):
  v=effective[:,j];signature=(tuple(v.indices.tolist()),tuple(np.round(v.data,12).tolist()))
  dead+=int(v.nnz==0);dup+=int(signature in seen);seen.add(signature)
 duplicates[N]=dict(duplicate_columns_at_12_decimals=dup,dead_columns=dead,unique_effective_columns=len(seen))
save('EFFECTIVE_INPUT_DIVERSITY.json',duplicates)

print(len(rows),'support rows; no native events')
