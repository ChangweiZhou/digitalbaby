"""Train-only compact correction selection, fixed splits, immutable sensory parity."""
from sensor import Sensor,ROOT,sha
import numpy as np,collections,json,importlib.util
from pathlib import Path
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('pinned_corpus',ROOT/'minifly_v182_shallow_local_dev_20261102/assets/corpus.py');C=importlib.util.module_from_spec(sp);sp.loader.exec_module(C)
def corrected(r,rule):
 a=bytearray(r)
 for t in range(2,len(r)):
  if r[t-2:t]==rule[0] and r[t]==rule[1]:a[t]=rule[2]
 return bytes(a)
def prepare(sensor):
 spec=json.loads((ROOT/'minifly_v181_mini_local_plasticity_design_20261102/SPEC.json').read_text());out={};audit={}
 for world in spec['worlds']:
  w=world['id'];splits={};a={}
  for role,nt in [('A',160),('B',48)]:
   raw=(ROOT/'minifly_v182_shallow_local_dev_20261102/assets/corpora'/world[role]).read_bytes();assert sha(raw)==world[role+'_sha256'];rs=C.records(raw);cut=next((i for i,r in enumerate(rs) if r.startswith(b'End of Project Gutenberg')),len(rs));rs=rs[:cut];train=rs[:nt];cal=rs[nt:nt+32];held=rs[-64:]
   assert not set(train)&set(held);assert len(train)==nt and len(cal)==32 and len(held)==64
   splits[role+'_train']=train;splits[role+'_calib']=cal;splits[role+'_held']=held
   a[role]={'raw_sha256':sha(raw),'records':[len(train),len(cal),len(held)],'target_applications_per_pass':sum(len(r)+1 for r in train),'train_held_exact_overlap':0}
  train=splits['A_train'];counts=collections.defaultdict(collections.Counter);rid=collections.defaultdict(set);letters=collections.Counter(b for r in train for b in r if 65<=b<=90 or 97<=b<=122)
  for i,r in enumerate(train):
   for t in range(2,len(r)):counts[r[t-2:t]][r[t]]+=1;rid[(r[t-2:t],r[t])].add(i)
  eligible=[]
  for k,c in counts.items():
   old=min(c,key=lambda b:(-c[b],b))
   if c[old]>=12 and c[old]/sum(c.values())>=.6 and (65<=old<=90 or 97<=old<=122):eligible.append((k,old,len(rid[(k,old)]),sum(c.values())))
  assert eligible,'NO_ELIGIBLE_CONTEXT'
  k,old,nrec,freq=min(eligible,key=lambda v:(-v[2],-v[3],sha(b'V181|'+w.encode()+b'|edit|'+v[0])));top=sorted(letters,key=lambda b:(-letters[b],b))[:16];new=min([b for b in top if b!=old],key=lambda b:sha(b'V181|'+w.encode()+b'|replacement|'+k+bytes([b])));rule=(k,old,new)
  ids=sorted([i for i,r in enumerate(train) if corrected(r,rule)!=r],key=lambda i:sha(f'V181|{w}|revision-record|{i}'.encode()))[:24]
  splits['NOEDIT_train']=[train[i] for i in ids];splits['EDIT_train']=[corrected(r,rule) for r in splits['NOEDIT_train']]
  changed=same=altered=0
  for r in splits['A_held']:
   rr=corrected(r,rule);c0=sensor.codes(r);c1=sensor.codes(rr);y=np.array([*r,256]);yy=np.array([*rr,256]);ch=y!=yy;eq=(c0==c1).all(1);changed+=int(ch.sum());same+=int((~ch&eq).sum());altered+=int((~ch&~eq).sum())
  a['correction']={'context_hex':k.hex(),'old':old,'new':new,'train_records_with_change':nrec,'context_frequency':freq,'selected_train_ids':ids,'changed_held':changed,'identical_KC_stable_held':same,'altered_KC_stable_held':altered,'qualified':len(ids)==24 and changed>=8 and same>=64}
  out[w]=splits;audit[w]=a
 return out,audit
