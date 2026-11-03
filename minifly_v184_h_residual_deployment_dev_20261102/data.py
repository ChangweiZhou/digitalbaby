"""Only M1, literal authenticated records and original correction IDs."""
import bootstrap as B
import json

def prepare():
 root=B.ROOT;old=root/'minifly_v181_mini_local_plasticity_dev_20261102/campaign_20261102';audit=json.loads((old/'DATA_AUDIT.json').read_text())['worlds']['M1'];sp={}
 for role,fn,n in [('A','austen-persuasion.txt',160),('B','austen-sense.txt',48)]:
  raw=(root/'minifly_v180_coherence_plasticity_dev_20261102/local/corpora'/fn).read_bytes();assert B.hashlib.sha256(raw).hexdigest()==audit[role]['raw_sha256'];rs=B.D.C.records(raw);cut=next((i for i,r in enumerate(rs) if r.startswith(b'End of Project Gutenberg')),len(rs));rs=rs[:cut]
  sp[role+'_train']=rs[:n];sp[role+'_calib']=rs[n:n+32];sp[role+'_held']=rs[-64:];assert not set(sp[role+'_train'])&set(sp[role+'_held'])
 c=audit['correction'];ids=c['selected_train_ids'];rule=(bytes.fromhex(c['context_hex']),c['old'],c['new']);sp['NOEDIT_train']=[sp['A_train'][i] for i in ids];sp['EDIT_train']=[B.D.corrected(r,rule) for r in sp['NOEDIT_train']]
 for k in ('calib','held'):sp['EDIT_'+k]=[B.D.corrected(r,rule) for r in sp['A_'+k]]
 manifest=json.loads((root/'minifly_v183_h_credit_dev_20261102/campaign/MANIFEST.json').read_text())
 for name,rs in sp.items():assert [B.hashlib.sha256(r).hexdigest() for r in rs]==manifest[name]['record_sha256'],name
 assert sum(len(r)+1 for rs in sp.values() for r in rs)==33307
 return sp,ids,manifest
