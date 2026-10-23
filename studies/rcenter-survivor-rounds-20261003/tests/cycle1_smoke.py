# SPDX-License-Identifier: GPL-3.0-or-later
import sys,time,json,resource,hashlib,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source/runtime'))
from survivor_fixture import make_world,latin_squares,DT,RECORD_SECONDS
from engine import Learner,SCALES
from assay import cue,state_identity,scores,suppress_value_writes,clock_vector
start=time.monotonic();w=make_world(310000)
assert len(latin_squares())==576 and len(w['events'])==384
assert len(w['sets']['old']['cues'])==12 and len(w['sets']['heldout']['cues'])==4
held=set(w['sets']['heldout']['cues']);assert all(e['cue_hex'] not in held for e in w['events'])
assert w['identifiability']['heldout_unique']
l=Learner('R_center',SCALES);n=l.clone()
records=[]
for e in w['events'][:2]:
 for name,m in [('W',l),('N_old_relation',n)]:
  p=cue(m,e['cue_hex'],e['at']);sc=scores(p,SCALES)
  with suppress_value_writes(m,name!='W'):m.observe_outcome(e['outcome'],e['at']+12*DT)
  m.feed(10,e['at']+13*DT);error=m.flush(e['at']+RECORD_SECONDS)
  assert error<1e-6 and m.cached is None
  assert m.audit[-1]['c']==0 and sorted(m.audit[-1]['s'])==[-.75,.25,.25,.25]
  if name!='W':assert m.audit[-1]['shared_l1']==[0.]*4 and m.audit[-1]['private_l1']==[0.]*4
  records.append({'branch':name,'prediction':p,'policies':sc,'state':state_identity(m),'clock':clock_vector(m),'error':error})
assert l.digests()!=n.digests()
d={'schema':'RC-SURVIVOR-CYCLE1-TEST-v1','passed':True,'fixture':w['sha256'],'identifiability':w['identifiability'],
   'checks':['576_latin_squares','unique_completion','heldout_never_trained','384_records','fixed_signed_teacher','no_write_actuator','clock_flush','all_readouts'],
   'records':records,'resources':{'wall_s':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}}
p=ROOT/'receipts/cycle1/smoke.json';p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,sort_keys=True,indent=2)+'\n')
with q.open('rb') as fh:os.fsync(fh.fileno())
try:os.link(q,p)
finally:q.unlink()
print(json.dumps({'passed':True,'receipt':str(p),'resources':d['resources']}))
