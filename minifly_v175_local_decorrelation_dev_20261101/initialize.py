import json,time,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parent
s=json.loads((R/'minifly_v175_local_decorrelation_design_20261101/SPEC.json').read_text())
s.update(execution_authorized=True,implementation_present=True,status='AUTHORIZED_THREE_CYCLES')
s['execution_clarifications']={'source_commit':'52e73caae1259ecd4aa7c95e7ad4cd02d1a9881c','design_commit':'e5f90d1923c79b32fe723feb27b27cbae2b46105','wall_seconds':14400,'output_bytes_max':1073741824,'rng':'SHA256(V175|world|stream) first16bytes big endian PCG64','topology_rng':'SHA256(V175|topology|side) first16bytes big endian PCG64','gamma_prompts':'64 independent uniform five-byte prompts, technical world gamma-prompts stream; no labels','floor':'modal payload per output position, independent of prompt identity; EOS after one payload','ceiling':'evaluator-only exact initial first-answer KC-code lookup; no learner uses table','shadow_events':'payload target at first NEW-pass record ordinals 1,3,5,7,9,11,13,15 per arm/world','damage':'mean zero-minus-written teacher-prefix target margin across payload/EOS of8 fixed OLD sentinels','REST128':'128*DELTA native seconds','S_and_J':'first-answer prospective prompt codes; OLD8/OLD16 frozen encoder snapshots','undefined_retention':'UNRESOLVED, never substituted zero','no_automatic_retry':True,'training_record_cap':5000,'own_TF_pair_cap':2700,'native_vector_read_cap':100000,'shadow_target_cap':160}
for n,x in [('EXECUTION_SPEC.json',s),('AUTHORIZATION.json',{'user':'Can you touch base and do the 3 cycle exercise for the new design proposed?','identity':s['identity'],'epoch':time.time(),'scope':'C0,C1,C2 bounded development, no confirmation or adoption','push_raw_data':False})]:
 p=H/n
 if p.exists():raise RuntimeError('Do not reset an existing session')
 p.write_text(json.dumps(x,indent=2)+'\n')
