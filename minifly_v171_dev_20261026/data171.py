"""External generator only. Candidate code never receives orbit IDs or rules."""
import hashlib
import support171 as s
import native171 as n
def make():
 out={};held={}
 for world in s.SPEC['data']['main_worlds']+s.SPEC['data']['calibration_worlds']:
  d=n.n.s.dataset(world);new=sorted(d['NEW'],key=lambda x:(hashlib.sha256(f"V169|{world}|NEW|{x['id']}".encode()).digest(),x['id']))[:16]
  seen=set();raw=set()
  for group in [d['OLD'],new,d['assessment']]:
   for row in group:
    key=tuple(row['orbit']);assert key not in seen and row['prompt'] not in raw;seen.add(key);raw.add(row['prompt'])
  out[str(world)]=dict(world=world,OLD=d['OLD'],NEW=new,orders=d['orders']['OLD'])
  held[str(world)]=d['assessment']
 s.save('local/TRAIN_DATA.json',out);s.save('local/ASSESSMENT_SEALED.json',held)
 receipt=dict(training_sha256=s.sha(s.H/'local/TRAIN_DATA.json'),assessment_sha256=s.sha(s.H/'local/ASSESSMENT_SEALED.json'),worlds=list(out),assessment_labels_used_for_learning=False,selected_NEW=16,OLD=64,assessment_orbits=50,salt='V169',version_alias='V171',labels_generated_external_only=True)
 s.save('DATA_LOCK.json',receipt);return out
