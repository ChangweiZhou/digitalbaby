import json,time,hashlib,platform,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parent
s=json.loads((R/'minifly_writer_interference_design_v1_20261027/SPEC.json').read_text())
s.update(status='AUTHORIZED_THREE_CYCLES',implemented=True,execution_authorized=True)
s['execution_details']=dict(generator_salt='WCFI-V1',hash_selection='sha256 WCFI-V1|world|purpose|id',solver='endpoints+midpoint, then bounded bisection, maximum20 evaluations',zero_floor='max(1e-12,100*zero_write_residual)',timing_admission_multiplier=1.5,calibration_prompts='same opaque generator restricted first32;balanced16labels',own_decisions=4)
for n,x in [('EXECUTION_SPEC.json',s),('AUTHORIZATION.json',dict(user_request='Do the 3-cycle exercise on WRITER-INTERFERENCE V1: design, audit, test run, revise.',epoch=time.time(),GitHub_push=False,automatic_retry=False))]:
 p=H/n
 if p.exists():raise RuntimeError('existing execution, do not reset')
 p.write_text(json.dumps(x,indent=2)+'\n')
