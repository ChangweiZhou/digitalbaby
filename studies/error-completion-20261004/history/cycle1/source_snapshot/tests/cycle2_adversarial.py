# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixed four-record adversarial quota; no extra valid teaching on probe clones."""
import os,sys,time,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source/runtime'))
import numpy as np
from engine import Learner,SCALES
from survivor_fixture import make_world,DT,RECORD_SECONDS
from assay import BRANCHES,cue,suppress_value_writes,shared_invariant,operative_state_identity,shared_identity
from integrity import ROOT,hashes,require_technical,require_runtime,read
from durable import create

def rejects(action):
 try:action()
 except (ValueError,AssertionError):return
 raise AssertionError('invalid operation accepted')
start=time.monotonic();require_technical();source=hashes();world=make_world(320101)
base=Learner('R_center',SCALES);systems={key:base.clone() for key in BRANCHES}
for key,system in systems.items():system.arm=key.split('__')[0]
rows=[];invariants=shared_invariant(systems,'birth');events=[world['events'][i] for i in (0,1,192,193)]
for index,event in enumerate(events):
 for key,system in systems.items():
  rejects(lambda:system.observe_outcome(event['outcome'],event['at']))
  pred=cue(system,event['cue_hex'],event['at']);pi=system.private_prediction.copy()
  rejects(lambda:system.predict(event['at']+12*DT));rejects(lambda:system.observe_outcome(event['outcome'],event['at']+12*DT+1))
  rejects(lambda:system.observe_outcome(255,event['at']+12*DT))
  # Returned arrays/emission are caller-owned; adversarial mutation must not alter cached teacher inputs.
  pred['emitted']=999;pred['private'][:]=[99.]*4;pred['combined'][:]=[99.]*4
  assert np.array_equal(system.private_prediction,pi)
  clone=system.clone();clone.private_prediction[0]+=1.;assert np.array_equal(system.private_prediction,pi)
  control=key.split('__')[1];blocked=(control=='N_old_relation' and event['stage']=='old') or (control=='N_new' and event['stage']=='new')
  with suppress_value_writes(system,blocked):system.observe_outcome(event['outcome'],event['at']+12*DT)
  target=np.asarray([float(b==event['outcome']) for b in b'0123']);log=system.audit[-1]
  assert log['s']==(pi-target if key.startswith('ERROR__') else .25-target).tolist()
  if blocked:assert log['shared_l1']==log['private_l1']==[0.]*4
  system.feed(10,event['at']+13*DT);assert system.flush(event['at']+RECORD_SECONDS)<1e-6
  rows.append({'index':index,'event':event,'branch':key,'write':log,'digests':system.digests()})
 invariants+=shared_invariant(systems,'event-'+str(index))
# Mutating a private operative state on a disposable clone leaves the shared hash and parent unchanged.
for key,system in systems.items():
 before=operative_state_identity(system);shared=shared_identity(system);c=system.clone();c.private[0].fly.m.slow[0]+=1.
 assert shared_identity(c)==shared and operative_state_identity(system)==before
 c.shared[0].fly.m.slow[0]+=1.;assert shared_identity(c)!=shared and operative_state_identity(system)==before
assert source==hashes();active=read(ROOT/'operations/ACTIVE_JOB.json')
result={'schema':'ERROR-COMPLETION-TECHNICAL-v1','passed':True,'key':'cycle2/test','world':320101,
 'attempt':active['attempt'],'reservation_id':active['reservation_id'],'source':source,'runtime':require_runtime(),
 'native_teaching_events':24,'extra_valid_teaching_on_clones':0,'fixture_sha256':world['sha256'],
 'records':rows,'shared_invariants':invariants,
 'resources':{'wall_s':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}}
create(Path(os.environ['SURVIVOR_DEST']),result)
print('cycle2 passed: twenty-four teaching events and fixed adversarial checks',flush=True)
