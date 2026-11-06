"""Enumerate observed one-relay MBON to CX paths without claiming a function."""
from collections import defaultdict
import json
import numpy as np
import pyarrow.parquet as pq
from anatomy import ROOT,META,EDGE

g=np.load(ROOT/"EXTRACT.npz",allow_pickle=False)
r,p=g["roles"],g["parent"]
m=pq.read_table(META,columns=["root_id","cell_type","hemibrain_type","top_nt"]).to_pydict()
roles=np.full(len(m["root_id"]),"",dtype="<U8")
roles[p]=r
mbon=set(p[r=="MBON"].tolist())
cx=set(p[~np.isin(r,["MBON","PN","KC","DAN","APL"])].tolist())
incoming=defaultdict(set);outgoing=defaultdict(set)
for batch in pq.ParquetFile(EDGE).iter_batches(batch_size=400000,columns=[
        "Presynaptic_Index","Postsynaptic_Index","Excitatory x Connectivity"]):
    u,v,w=[np.asarray(x) for x in batch.columns]
    a=np.isin(u,list(mbon))&(roles[v]=="")&(w!=0)
    b=np.isin(v,list(cx))&(roles[u]=="")&(w!=0)
    for source,relay in zip(u[a],v[a]):incoming[int(relay)].add(int(source))
    for relay,target in zip(u[b],v[b]):outgoing[int(relay)].add(int(target))
connected=incoming.keys()&outgoing.keys()
best=sorted(connected,key=lambda k:(-len(incoming[k])*len(outgoing[k]),k))[:12]
rows=[dict(root_id=m["root_id"][k],label=m["cell_type"][k] or m["hemibrain_type"][k],
           transmitter=m["top_nt"][k],MBON_sources=len(incoming[k]),CX_targets=len(outgoing[k]),
           CX_roles=sorted(set(roles[list(outgoing[k])].tolist()))) for k in best]
out=dict(omitted_one_relay_candidates=len(connected),
         tested_MBON_sources_with_any_relay=len(set().union(*(incoming[k] for k in connected))),
         examples_by_two_hop_combinations=rows,claim="directed nonzero graph paths; behavioral roles unverified")
(ROOT/"BOUNDARY_AUDIT.json").write_text(json.dumps(out,indent=2)+"\n")
print(out["omitted_one_relay_candidates"],out["tested_MBON_sources_with_any_relay"])
