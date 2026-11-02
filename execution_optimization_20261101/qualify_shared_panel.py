"""Bounded exact common-prefix evaluation qualification; no new learner fitting."""
import json,pickle,time,resource
from pathlib import Path
import optimized_native as O
H=Path(__file__).resolve().parent;W=O.N.W;t=time.process_time();results=[]
for arm in ('RAW','H'):
 with (H/f'local/optimized_{arm}_selfcontained.pkl').open('rb') as f:m=pickle.load(f)
 for length in (0,35,71):
  row=dict(id=f'shared-panel:{arm}:{length}',cohort='engineering',prompt=bytes((i*37+11)%256 for i in range(length)).hex(),answer='78')
  before=W.digest(m);a=W.pair(m,row);b=O.pair_shared(m,row)
  assert a==b and W.digest(m)==before
  start=time.process_time();O.pair_shared(m,row);seconds=time.process_time()-start
  results.append(dict(arm=arm,prompt_length=length,bitwise_endpoint_equal=True,source_state_unchanged=True,own_TF_pair_CPU=seconds,own_decisions_cap=4))
out=dict(status='PASS_BITWISE',results=results,own_TF_pairs=18,CPU=time.process_time()-t,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scientific_records=0)
(H/'SHARED_PANEL_QUALIFICATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
