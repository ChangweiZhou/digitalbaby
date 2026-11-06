"""Read-only v630 extraction for engineering qualification."""
from collections import Counter
import csv
from hashlib import sha256
from pathlib import Path
import re
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parent
ARCHIVE=Path("/Users/pencilbard/Downloads/second_try_copy/minifly")
META=ARCHIVE/"data/flywire630/metadata.parquet"
EDGE=ARCHIVE/"sources/shiu/2023_03_23_connectivity_630_final.parquet"
ROSTERS=[ARCHIVE/x for x in ("iteration5/data/input_neurons.csv",
        "iteration5/data/coding_units.csv","iteration5/results/interface_neurons.csv")]


def hash_file(path):
    h=sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()


def extract():
    cols=["parent_index","root_id","side","cell_class","cell_type","hemibrain_type","top_nt"]
    m=pq.read_table(META,columns=cols).to_pydict()
    n=len(m["root_id"])
    assert m["parent_index"]==list(range(n)) and n==127400
    memberships={}
    for role,path in zip(("PN","KC","MB"),ROSTERS):
        with path.open() as f:
            for row in csv.DictReader(f):
                memberships[row["root_id"]]=(role if role!="MB" else
                                             "DAN" if row["role"].startswith("D") else "MBON")
    roles=[];labels=[]
    for root,cl,t,h in zip(m["root_id"],m["cell_class"],m["cell_type"],m["hemibrain_type"]):
        names=(t or "",h or "")
        role=memberships.get(root,"")
        if not role and any(re.match(r"^APL(?:_|$)",x) for x in names):role="APL"
        if cl=="CX":
            for family,pattern in (
                ("EPG",r"^EPG$"),("PEN",r"^PEN_"),("PEG",r"^PEG$"),("D7",r"^Delta7$"),
                ("ER",r"^ER[1-5]"),("PFN",r"^PFN"),("PFR",r"^PFR"),
                ("hDelta",r"^hDelta"),("FC2",r"^FC2"),("PFL",r"^PFL[123]$")):
                if any(re.search(pattern,x) for x in names):
                    assert not role
                    role=family
        roles.append(role)
        labels.append(t or h or "")
    keep=np.flatnonzero(roles)
    lookup=np.full(n,-1,dtype=np.int32)
    lookup[keep]=np.arange(len(keep))
    r=np.array(roles)[keep]
    nts=np.array([x or "" for x in m["top_nt"]])[keep]
    us=[];vs=[];ws=[]; dropped=0.; full_in=0.
    for b in pq.ParquetFile(EDGE).iter_batches(batch_size=300000,columns=[
            "Presynaptic_Index","Postsynaptic_Index","Excitatory x Connectivity"]):
        a,z,w=[np.asarray(x) for x in b.columns]
        into=lookup[z]>=0
        mask=(lookup[a]>=0)&into&(w!=0)
        dropped+=np.abs(w[into&(lookup[a]<0)]).sum()
        full_in+=np.abs(w[into]).sum()
        us.append(lookup[a[mask]]);vs.append(lookup[z[mask]]);ws.append(w[mask])
    u,v,w=map(np.concatenate,(us,vs,ws))
    expected=np.where(nts[u]=="acetylcholine",1,np.where(nts[u]=="gaba",-1,0))
    # KC class override follows the archived Iteration 5 treatment.
    expected=np.where(r[u]=="KC",1,expected)
    direct=(r[u]=="KC")&(r[v]=="MBON")
    mod=(r[u]=="DAN")
    usable=(expected!=0)&((expected==np.sign(w))|(r[u]=="KC"))&~mod
    assert np.all(usable[direct])
    signs=expected.astype(np.int8)
    observed=Counter(zip(r[u].tolist(),r[v].tolist()))
    admitted=Counter(zip(r[u[usable]].tolist(),r[v[usable]].tolist()))
    np.savez_compressed(ROOT/"EXTRACT.npz",u=u,v=v,w=np.abs(w),signed_raw=w,roles=r,
                        parent=keep,ids=np.asarray(m["root_id"])[keep],labels=np.asarray(labels)[keep],
                        sides=np.asarray([x or "" for x in m["side"]])[keep],
                        sign=signs,usable=usable,plastic=direct)
    return {"files":{str(p.relative_to(ARCHIVE)):hash_file(p) for p in (META,EDGE,*ROSTERS)},
            "nodes":len(keep),"roles":dict(Counter(r.tolist())),
            "edges":len(u),"admitted_edges":int(usable.sum()),
            "known_sign_conflicts_excluding_KC":int(((expected!=0)&(expected!=np.sign(w))&(r[u]!="KC")).sum()),
            "KC_sign_overrides":int(((r[u]=="KC")&(np.sign(w)!=1)).sum()),
            "DAN_modulatory_records":int(mod.sum()),"incoming_external_abs_mass_fraction":float(dropped/full_in),
            "routes":{f"{a}->{b}":n for (a,b),n in sorted(observed.items())},
            "admitted_routes":{f"{a}->{b}":n for (a,b),n in sorted(admitted.items())},
            "shared_gain_groups":len(admitted),"fitted_scalars":0,"candidate_KC_MBON_edges":int(direct.sum()),
            "direct_MBON_CX_records":sum(n for (a,b),n in observed.items() if a=="MBON" and b not in ("PN","KC","APL","DAN","MBON")),
            "compartment_sector_map":"missing from this historical edge table",
            "FB_plastic_mask":"unregistered; no writes allowed"}
