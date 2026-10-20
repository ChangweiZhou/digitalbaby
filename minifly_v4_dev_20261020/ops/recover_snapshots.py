from pathlib import Path
import sys,json,hashlib,shutil,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v4_common import HERE,ROOT,sha,save
initial='''{
 "schema":"MINIFLY_V4_THREE_CYCLE_TECHNICAL",
 "claude_commit":"190a26cb430079f2096fd6ffe7da1c635f8611cf",
 "technical_only":true,"formal_campaign_authorized":false,"automatic_adoption":false,
 "task":"Addition only, ordinary byte strings, exact own output and own EOS",
 "worlds":[202610204401,202610204402],
 "sizes":[65536,262144],"segment_slots":3,"segment_limit":16,"input_limit":256,"output_limit":16,
 "fan_in":12,"active_fraction":0.02,"widths":[1,2,4,8],
 "q_tau":86400,"q0":4,"q_slope":1,"fraction_min":0.0,"fraction_max":0.5,
 "boundary_eta":0.001,"boundary_logit_cap":4,"boundary_e_clip":8,
 "a0":0.0625,"drift_checkpoint":16,"drift_threshold":0.1,"stable_checkpoints_required":2,"ramp_increment":0.0625,
 "homeostasis_window":256,"homeostasis_eta":0.005,
 "G0_train_records":128,"G0_test_records":64,"G0_ridge":0.01,
 "G0_required_correct_byte_fraction":0.95,
 "G1_records":64,"G1_probe_records":16,"replay_limit":8,"replay_capacity":32,
 "rest_seconds":86400,"slow_downscale":0.999,
 "stop":"If coverage is below G0, no full P3/G2 campaign; complete bounded interface/learning-sanity tests and report prerequisites unmet.",
 "arithmetic_features":false,"attention":false,"transformer":false,"ELM":false,"backprop":false
}
'''
current=json.loads((HERE/'SPEC.json').read_text());candidates=[initial,(HERE/'SPEC.json').read_text()]
for remove in (('G1_worlds','superseded_G1_worlds','signed_write_convention','positive_control_record_applications'),('signed_write_convention','positive_control_record_applications')):
 d=copy.deepcopy(current)
 for key in remove:d.pop(key,None)
 if 'G1_worlds' in d:d['G1_worlds']=[202610204451,202610204452]
 candidates.append(json.dumps(d,indent=2)+'\n')
design=(HERE/'design/EXPERIMENT.md').read_text();design_candidates=[design]
for marker in ('\n## Cycle-2 audit revisions before cycle 3','\nThe third-cycle replay-zero check','\nA subsequent source-to-port audit','\nThe new directional fixture itself'):
 if marker in design:
  prefix=design.split(marker)[0]
  design_candidates.extend(prefix.rstrip('\n')+'\n'*k for k in range(5))

stages=['cycle1','cycle2','cycle3_preflight_serialization_failure','cycle3_replay_zero_failure','cycle3_invalid_teaching_sign','cycle3_directional_fixture_failure','cycle3']
report=[]
for stage in stages:
 folder=HERE/'cycles'/stage
 if stage=='cycle1':expected=json.loads((folder/'AUDIT.json').read_text())['source_after']
 else:expected=json.loads((folder/'SOURCE_LOCK.json').read_text())
 out=folder/'SOURCE_EXECUTED';out.mkdir(exist_ok=True);missing=[];reconstructed=[]
 for path,want in expected.items():
  f=Path(path)
  if not str(f).startswith(HERE.name+'/'):continue
  if f.name=='SPEC.json' or str(f).endswith('design/EXPERIMENT.md'):
   options=candidates if f.name=='SPEC.json' else design_candidates
   matches=[s for s in options if hashlib.sha256(s.encode()).hexdigest()==want]
   if matches:
    dest=out/('SPEC.json' if f.name=='SPEC.json' else 'EXPERIMENT_EXECUTED.md');dest.write_text(matches[0]);reconstructed.append(dest.name)
   else:missing.append(path)
  elif f.suffix=='.py':
   choices=[out/f.name,HERE/'cycles/cycle2/SOURCE_EXECUTED'/f.name,ROOT/f]
   found=[p for p in choices if p.exists() and sha(p)==want]
   if found and found[0]!=out/f.name:shutil.copy2(found[0],out/f.name)
   if not found:missing.append(path)
 result=dict(stage=stage,exact_hash_verified=not missing,reconstructed_metadata_from_preserved_execution_inputs=reconstructed,missing=missing)
 report.append(result)
save(Path('SNAPSHOT_AUDIT.json'),dict(stages=report,all_exact=all(x['exact_hash_verified'] for x in report)))
print(json.dumps(report))
