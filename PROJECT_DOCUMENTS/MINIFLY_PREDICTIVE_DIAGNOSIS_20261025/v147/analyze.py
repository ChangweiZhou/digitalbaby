"""Reduce existing taught generation records only. No learner import or new generation."""
from pathlib import Path
import hashlib,json,time,sys
H=Path(__file__).resolve().parent
R=H.parents[2]
D=R/'minifly_v147_dev_20261024'
start=time.process_time();sources={}
def read(name):
 p=D/name;b=p.read_bytes();sources[str(p.relative_to(R))]=hashlib.sha256(b).hexdigest();return json.loads(b)
compact=read('COMPACT.json');diag={x['id']:x for x in read('SAVED_STATE_DIAGNOSTICS.json')};out=[]
for row in compact:
 z=read(row['id']+'.json');w=read(z['world']+'_WORLD.json');curves=[]
 taught=set(map(tuple,w['train']))
 for c in z['curve']:
  groups={}
  for q in c['taught']['rows']:
   pair=tuple(q['pair']);assert tuple(sorted(pair)) in taught
   expected=w['sigma'][sum(pair)%23];payload=bytes.fromhex(q['output_hex']);first=bool(payload) and payload[0]==expected
   v={'complete':q['status']=='EOS' and payload==bytes([expected]),'first_payload_correct':first,'EOS_after_one_payload':q['status']=='EOS' and len(payload)==1,'correct_first_but_incomplete':first and not q['exact'],'invalid':q['status']=='INVALID','truncated':q['status']=='TRUNCATED'}
   assert v['complete']==q['exact'];groups.setdefault(tuple(sorted(pair)),[]).append(v)
  assert len(groups)==207
  m={k:sum(sum(x[k] for x in vs)/len(vs) for vs in groups.values())/len(groups) for k in v}
  assert abs(m['complete']-c['taught']['complete'])<1e-12
  m['complete_given_correct_first']=m['complete']/m['first_payload_correct'] if m['first_payload_correct'] else None
  if m['complete_given_correct_first'] is not None:
   assert abs(m['complete']-m['first_payload_correct']*m['complete_given_correct_first'])<1e-12
  curves.append({'passes':c['passes'],**m})
 out.append({'id':z['id'],'world':z['world'],'arm':z['arm'],'alpha':z['alpha'],'records':z['records'],'curves':curves,'saved_state_diagnostics':diag[z['id']]})
paired=[]
for world in ['V147-D1','V147-D2']:
 f=next(x for x in out if x['world']==world and x['arm']=='FIXED' and x['alpha']==.03);l=next(x for x in out if x['world']==world and x['arm']=='LOCAL' and x['alpha']==.03)
 paired.append({'world':world,'LOCAL_minus_FIXED_complete_pp':100*(l['curves'][-1]['complete']-f['curves'][-1]['complete']),'LOCAL_minus_FIXED_teacher_bit_BCE':l['saved_state_diagnostics']['teacher_bit_BCE']-f['saved_state_diagnostics']['teacher_bit_BCE']})
res={'status':'RETROSPECTIVE_TAUGHT_OUTPUT_REDUCTION','evidence':'Existing E1 modular calibration, no new E3 evaluation','new_training_records':0,'new_target_events':0,'new_generated_outputs':0,'held_inputs_scored':0,'assessment_worlds_generated':False,'fits':out,'paired_high_rate_effects':paired,'source_sha256':sources,'CPU_seconds':time.process_time()-start,'previous_scalar_only_reduction_CPU_seconds':0.030125,'limitations':['Only two development worlds; no confirmation or population generalization estimate','Stable, distinct codes do not prove readout separability or convergence','Mean teacher bit loss is not complete autonomous output accuracy','All outcomes already reported; no independent prediction validation']}
(H/'RESULTS.json').write_text(json.dumps(res,indent=2)+'\n')
lines=['# V147 saved-output reanalysis','','Existing taught records only. No new generation, training or held scoring.','','| Fit, alpha 0.03 | Correct first payload | Complete byte/EOS | EOS after one byte |','| --- | ---: | ---: | ---: |']
for x in out:
 if x['alpha']==.03:
  y=x['curves'][-1];lines.append(f"| {x['world']} {x['arm']} | {100*y['first_payload_correct']:.2f}% | {100*y['complete']:.2f}% | {100*y['EOS_after_one_payload']:.2f}% |")
lines+=['','For this single-payload-byte task, complete / correct-first is the conditional probability of immediate exact EOS given a correct first byte on the SAME own-generated stream. It is valid here; the known-length native payload score must not be substituted into it.']
for x in out:
 if x['alpha']==.03 and x['arm']=='LOCAL':
  lines.append(f"{x['world']} LOCAL: conditional immediate correct EOS = {100*x['curves'][-1]['complete_given_correct_first']:.2f}%.")
lines+=['','LOCAL has lower mean teacher bit BCE but worse complete generation in both high-rate worlds. Its representation changes; FIXED is stable and still misses readiness. Neither static code uniqueness nor mean bit loss is a sufficient success proxy.','', 'This does not identify the dominant mechanism and does not test held generalization. All holdback boundaries remain intact.']
(H/'FINDINGS.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'paired':paired,'CPU':res['CPU_seconds']},indent=2))
