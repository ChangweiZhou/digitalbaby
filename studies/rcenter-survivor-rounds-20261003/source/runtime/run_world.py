# SPDX-License-Identifier: GPL-3.0-or-later
"""One fixed world with immutable bounded JSON receipt parts."""
import gc,json,resource,time
from pathlib import Path
import numpy as np
from engine import Learner,SCALES
from survivor_fixture import make_world,DT,RECORD_SECONDS,OFFICIAL_WORLDS,PILOT_WORLD
from assay import BRANCHES,cue,probe,clock_vector,state_identity,suppress_value_writes,fixed_identity,operative_state_identity
from integrity import ROOT,hashes,digest_map,require_technical,require_science,runtime_versions,sha
from durable import create,canonical,fsync_dir
MAX_PART_BYTES=512*1024

def validate_clock(m,t):
 for s in m.shared+m.private:
  assert abs(s.brain_t-t)<1e-6
  assert abs(float(s.fly.m.elapsed)-s.elapsed_base-t)<1e-6
  assert s.pending_t is None and s.pending_x is None and abs(float(s.fe.t)-t)<1e-6
  assert s.last_byte_t is None or s.last_byte_t<=t+1e-6
 return clock_vector(m)
def state_budget(m):
 return [{'learned_shapes':{'fast':list(s.fly.m.fast.shape),'slow':list(s.fly.m.slow.shape),'adapt':list(s.fly.m.adapt.shape),'w':list(s.w.shape),'bias':list(s.bias.shape)},
          'core_mutable_bytes':int(s.fly.m.mutable_bytes()+s.w.nbytes+s.bias.nbytes+s.fe.p.nbytes),
          'fixed_numeric_bytes':int(s.fly.m.fixed_numeric_bytes()+s.fly.reader.extra_fixed_numeric_bytes()),
          'content_window_bound':4 if hasattr(s.fe,'visible') else 0} for s in m.shared+m.private]

def execute(world_id,kind,dest):
 if kind=='science':
  assert world_id in OFFICIAL_WORLDS;acceptance=require_science()
 else:
  assert world_id==PILOT_WORLD;acceptance=require_technical()
 dest=Path(dest);assert not dest.exists(),'attempt output must be fresh; never overwrite'
 start=time.monotonic();source=hashes();base=Learner('R_center',SCALES);birth=base.digests();budget=state_budget(base)
 anchor=fixed_identity(base)
 world=make_world(world_id);systems={b:base.clone() for b in BRANCHES};assert len({state_identity(x) for x in systems.values()})==1
 first=[];probes=[];clock_error=0.;clock_history=[];interventions=[];boundaries={b:{} for b in BRANCHES}
 for index,event in enumerate(world['events']):
  for name,system in systems.items():
   pred=cue(system,event['cue_hex'],event['at']);first.append({'index':index,'branch':name,**pred})
   blocked=(name=='N_old_relation' and event['stage']=='old') or (name=='N_new' and event['stage']=='new')
   interventions.append({'index':index,'branch':name,'write':not blocked})
   with suppress_value_writes(system,blocked):system.observe_outcome(event['outcome'],event['at']+12*DT)
   system.feed(10,event['at']+13*DT);clock_error=max(clock_error,system.flush(event['at']+RECORD_SECONDS))
   cv=validate_clock(system,event['at']+RECORD_SECONDS)
   clock_history.append({'index':index,'branch':name,'clocks':cv})
  # Nonplastic adaptation/clocks must be identical for corresponding bank/stores.
  for store in range(8):
   ms=[(s.shared+s.private)[store] for s in systems.values()]
   assert all(np.array_equal(ms[0].fly.m.adapt,m.fly.m.adapt) for m in ms[1:])
  if index==191:
   for name,system in systems.items():
    boundaries[name]['old_end']=validate_clock(system,world['old_end'])
    probes.append({'branch':name,'when':'old_end',**probe(system,world,world['old_end'],('old','heldout'))})
    clock_error=max(clock_error,system.flush(world['new_start']));boundaries[name]['new_start']=validate_clock(system,world['new_start'])
  if index==383:
   for name,system in systems.items():
    boundaries[name]['new_end']=validate_clock(system,world['new_end'])
    probes.append({'branch':name,'when':'new_end',**probe(system,world,world['new_end'],('old','heldout','new'))})
    clock_error=max(clock_error,system.flush(world['final']));boundaries[name]['final']=validate_clock(system,world['final'])
    probes.append({'branch':name,'when':'final',**probe(system,world,world['final'],('old','heldout','new'))})
 assert clock_error<1e-6 and all(state_budget(s)==budget and fixed_identity(s)==anchor for s in systems.values())
 assert source==hashes(),'source changed during world'
 parts={}
 def part(name,data):
  blob=canonical(data);assert len(blob)<=MAX_PART_BYTES,('oversized part',name,len(blob));parts[name]={'sha256':create(dest/name,data),'bytes':len(blob)}
 part('header.json',{'schema':'RC-SURVIVOR-WORLD-v1','world':world_id,'kind':kind,'arm':'R_center','fixture_sha256':world['sha256'],
                     'source':source,'source_digest':digest_map(source),'acceptance_sha256':acceptance,'runtime':runtime_versions(),
                     'scales':list(SCALES),'birth_contract_sha256':sha(ROOT/'source/runtime/BIRTH_CONTRACT.json'),'birth_digests':birth,'births':[{k:v for k,v in b.items() if k!='fly_id'} for b in base.births],
                     'state_budget':budget,'fixed_identity':anchor,'tie_policy':'lowest ASCII via np.argmax','endpoint':'taught-dependent held-out table completion'})
 part('fixture.json',world)
 part('boundaries.json',boundaries)
 for start_i in range(0,384,64):
  stop=start_i+64
  part(f'predictions-{start_i:03d}.json',[r for r in first if start_i<=r['index']<stop])
  part(f'clocks-{start_i:03d}.json',[r for r in clock_history if start_i<=r['index']<stop])
  part(f'interventions-{start_i:03d}.json',[r for r in interventions if start_i<=r['index']<stop])
  for branch,system in systems.items():part(f'audit-{branch}-{start_i:03d}.json',system.audit[start_i:stop])
 for j,p in enumerate(probes):part(f'probe-{j:02d}.json',p)
 resources={'wall_s':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
 part('end.json',{'end_state':{b:s.digests() for b,s in systems.items()},'end_wrapper':{b:operative_state_identity(s) for b,s in systems.items()},
                   'end_clocks':{b:clock_vector(s) for b,s in systems.items()},'clock_error':clock_error,'resources':resources})
 manifest={'schema':'RC-SURVIVOR-RECEIPT-v1','complete':True,'world':world_id,'kind':kind,'parts':parts,
           'source_digest':digest_map(source),'fixture_sha256':world['sha256'],'resources':resources}
 create(dest/'manifest.json',manifest);fsync_dir(dest)
 return manifest
