# SPDX-License-Identifier: GPL-3.0-or-later
"""Exactly two records, two rules, three controls: twelve outcome events."""
import os,sys,time,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source/runtime'))
import numpy as np
from engine import Learner,SCALES
from survivor_fixture import make_world,DT,RECORD_SECONDS
from assay import ARMS,BRANCHES,CONTROLS,cue,suppress_value_writes,shared_invariant,state_identity,operative_state_identity
from integrity import ROOT,hashes,require_technical,require_runtime,read
from durable import create
start=time.monotonic();require_technical();source=hashes();world=make_world(320000)
base=Learner('R_center',SCALES);systems={key:base.clone() for key in BRANCHES}
for key,system in systems.items():system.arm=key.split('__')[0]
for i,left in enumerate(systems.values()):
 for right in list(systems.values())[i+1:]:
  for a,b in zip(left.shared+left.private,right.shared+right.private):
   for attr in ('fast','slow','adapt'):assert not np.shares_memory(getattr(a.fly.m,attr),getattr(b.fly.m,attr))
rows=[];invariants=shared_invariant(systems,'birth')
for index,event in enumerate(world['events'][:2]):
 for key,system in systems.items():
  prediction=cue(system,event['cue_hex'],event['at']);before=system.private_prediction.copy()
  z=np.asarray(prediction['private'])/SCALES[1];z-=z.max();pi=np.exp(z);pi/=pi.sum();assert np.array_equal(before,pi)
  blocked=key.endswith('__N_old_relation')
  with suppress_value_writes(system,blocked):system.observe_outcome(event['outcome'],event['at']+12*DT)
  log=system.audit[-1];target=np.asarray([float(b==event['outcome']) for b in b'0123'])
  assert np.array_equal(log['s'],pi-target if key.startswith('ERROR__') else .25-target)
  assert log['private_pre_outcome_probabilities']==before.tolist() and system.private_prediction is None
  if blocked:assert log['shared_l1']==log['private_l1']==[0.]*4
  system.feed(10,event['at']+13*DT);assert system.flush(event['at']+RECORD_SECONDS)<1e-6
  rows.append({'index':index,'branch':key,'prediction':prediction,'write':log,'store_digests':system.digests()})
 invariants+=shared_invariant(systems,'event-'+str(index))
# Six disposable cue-only probes, no outcome: one heldout per continuing branch.
probe_rows=[]
for key,system in systems.items():
 before=operative_state_identity(system);clone=system.clone();pred=cue(clone,world['sets']['heldout']['cues'][0],330.)
 assert operative_state_identity(system)==before
 assert clone.cached is not None and clone.private_prediction is not None
 probe_rows.append({'branch':key,'before':before,'after':operative_state_identity(system),'prediction':pred})
assert source==hashes();active=read(ROOT/'operations/ACTIVE_JOB.json')
result={'schema':'ERROR-COMPLETION-TECHNICAL-v1','passed':True,'key':'cycle1/test','world':320000,
 'attempt':active['attempt'],'reservation_id':active['reservation_id'],'source':source,'runtime':require_runtime(),
 'native_teaching_events':12,'disposable_probe_predictions':6,'fixture_sha256':world['sha256'],
 'records':rows,'shared_invariants':invariants,'probe_rows':probe_rows,
 'resources':{'wall_s':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}}
create(Path(os.environ['SURVIVOR_DEST']),result)
print('cycle1 passed: twelve teaching events, six unreinforced probes',flush=True)
