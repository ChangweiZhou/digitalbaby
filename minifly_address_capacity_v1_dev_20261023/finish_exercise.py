"""Complete artifact reductions and package without replaying native learning."""
from pathlib import Path
import json,subprocess,sys,time,hashlib,zipfile,resource
P=Path(__file__).resolve().parent;R=P.parent
start=time.monotonic();cpu=time.process_time()
for script in ('reduce.py','audit_artifacts.py','measure_supports.py'):
 r=subprocess.run([sys.executable,'-B',str(P/script)],capture_output=True,text=True)
 (P/'cycles'/f'{script}.log').write_text(r.stdout+r.stderr);r.check_returncode()
s=json.loads((P/'PILOT_SUMMARY.json').read_text());c=json.loads((P/'CONSTRUCTION.json').read_text());cycle=json.loads((P/'cycles/CYCLE3.json').read_text())
boundaries=[]
for line in (P/'cycles/cycle3.log').read_text().splitlines():
 try:v=json.loads(line)
 except ValueError:continue
 if v['message'].startswith(('life ','pair ')):boundaries.append(v)
costs={'CANON':0.,'EXPAND':0.};index=0
for a,b in zip(boundaries,boundaries[1:]):
 if a['message'].startswith('life '):
  size='CANON' if index%6<3 else 'EXPAND';costs[size]+=b['CPU_seconds']-a['CPU_seconds'];index+=1
(P/'COST_BY_SIZE.json').write_text(json.dumps({'full_life_and_readout_CPU_seconds':costs,'note':'Boundary timings include encoding, teaching, probing, checkpoint serialization and readout. Shared construction, qualification and final artifact processing are separate.'},indent=2)+'\n')
rows=[]
for metric in ('payload','complete'):
 for cohort in ('OLD_final','NEW_final'):
  rr=[x[f'{cohort}_{metric}'] for x in s['comparisons']];means=[sum(x[k] for x in rr)/8 for k in ('CANON','EXPAND','delta')]
  rows.append('| '+cohort+' | '+metric+' | '+' | '.join(f'{100*v:.3f}%' for v in means)+' |')
report=f'''# ADDRESS CAPACITY V1 three cycle exercise

All three bounded stages finished. Storage screen: **{s['storage_screen']}**. Complete-output screen: **{s['integration_screen']}**. This is E1 taught random-association evidence, not unseen addition or a qualified external memory.

## Cycle 1 saved response controls

All 32 official STD W pairs, two value lives and all saved checkpoints were evaluated without native training or head fitting. Final NEW complete recall was 11.263% STOP, 10.645% fixed endpoint 1, 10.645% position-only, and 4.850% foreign EOS input. Final OLD was 0.326% STOP versus 0.977% fixed endpoint 1. Most of the benefit over zero-scoring DIRECT is generic stopping calibration. Real memory input contributes a smaller benefit on NEW; do not confuse the original 22.10-point conditional gain over DIRECT with gain over this stronger fixed reference. All fixed lengths 0–8 remain in the raw rows; no winner was selected for the candidate. Undefined conditional subsets remain explicitly counted and cannot be averaged into a confirmation claim. The official historical confirmation is unchanged.

Revision: foreign input changes EOS only; real payload signs and eligibility counts are identical across every control.

## Cycle 2 fixed activity construction and fixtures

The extension has 20,708 KCs and 110,288 input edges versus 5,177 and 27,572. Calibration recruits exactly 92 cells per side in both. Effective recruitment diversity is {c['effective_N0']:.2f} versus {c['effective_N']:.2f}. Mean signed self-write fixture gain is 0.93884 versus 1.00320, while mean absolute cross-write response is 0.04623 versus 0.01981. That roughly 6.9% self-gain difference rules out describing the experiment as population size alone. The added wiring and unlabelled Q normalization are part of this engineering intervention.

288 partial calibration fixture events tested the self/cross-write matrix; two additional full one-byte synthetic records (four events) verified full checkpoint restoration before fresh values. Read-only probes and all 257 codec symbols passed. No graph selection or gain retuning followed the measurements.

Revision: the old helper was replaced with explicit B construction, fixed side activity limits and N-shaped bookkeeping; restore qualification was added before values. SOURCE_LOCK_AMENDMENT.json preserves the early source lock and identifies the authoritative pre-value imported-runtime closure after that revision.

## Cycle 3 fresh pilot

Eight pairs, two value lives, CANON/EXPAND, W/N/SH and matched passive NEW branches completed **56,448 new record applications / 282,240 main events**. There are 96 base histories and 32 correlated continuations, not 128 independent worlds. The assessment unit is the paired world. Gradual warmup continued across OLD and NEW exactly as designed. No head fits or arithmetic changes were made.

| Cohort | Outcome | CANON | EXPAND | Difference |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(rows)}

Complete scores average all three frozen STD heads. Payload uses native signs and evaluator length only for scoring. Raw pair differences, no-write/shuffled branches, acquisition checkpoints, absent keys, passive continuations and saved output tensors remain available. The conservative storage screen requires every pair to improve OLD payload without NEW regression; an average improvement alone cannot pass it.

## Cost and audit limits

Native qualification plus roster resources: {json.dumps(cycle['resources'])}. Per-size full-life and readout CPU seconds: {json.dumps(costs)}. Saved stopping analysis used 13.23 CPU seconds separately. Environment setup initially found incompatible NumPy/Numba; a cached NumPy 1.24.3 / SciPy 1.11.1 / Numba 0.57.1 runtime was used. The unused download was terminated. Dependency setup is operational overhead, not training exposure.

The separate artifact audit reconstructs payload exactness from saved native tensors and complete exactness from emitted bytes, and verifies pre-value runtime/assets and arithmetic pointer hashes. It is a separate process, not an independent human audit. No training is replayed.

The development implementation records aggregate native write magnitudes and saved state, but does not provide every proposed per-event L2, clipping or signed fast/slow diagnostic. PRE_VALUE_ASSET_LOCK also precedes assessment address files, which are generated deterministically and verified on reuse. This is a bounded development result with disclosed instrumentation limits, not full formal-campaign qualification. No mechanistic claim rests on the missing measurements.

No automatic confirmation, host integration, size sweep, EOS reweighting, warmup increase or core adoption follows. See REVISED_DESIGN.md for the next decision.
'''
(P/'REPORT.md').write_text(report)
rev=f'''# ADDRESS CAPACITY V1 revision after three cycles

The fixed-reference stopping result narrows the earlier explanation. Keep STOP as the confirmed component relative to DIRECT, but use FIXED_1 and position-only controls in any practical readout comparison. Preserve all fixed endpoints without choosing one from assessment scores.

Storage screen passed: {s['storage_screen']}. Integration screen passed: {s['integration_screen']}. Follow the prospectively specified pair-level decision rather than replacing it with mean accuracy. Do not repeat these worlds, retune the graph/gains, or extend n.

The expanded native population is an engineering contrast with changed input connectivity and a measured 6.9% self-write-gain difference. It is not a pure N experiment or a compute-efficiency result. Report OLD and NEW together and charge the fourfold cell/edge state.

Before a formal follow-up, finish the omitted per-event plastic L2 and signed fast/slow/clipping instrumentation and add an independently reconstructed stopping-logit audit, rather than only metric reduction. This is an instrumentation requirement, not permission to change the completed candidate or its gate. Native payload and complete output remain separate outcomes. Neither result establishes E3 arithmetic transfer.

EOS balancing and mature-life feedback remain separate possible designs. Gradual warmup, the arithmetic track, no-attention/no-Transformer/no-ELM and no supplied arithmetic-rule constraints remain unchanged. No follow-up is launched.
'''
(P/'REVISED_DESIGN.md').write_text(rev)
(P/'cycles/CYCLE3_AUDIT.md').write_text('Cycle 3 completed the fixed roster. Artifact reducer independently checked native payload and emitted exactness; compare FINAL_AUDIT.json. Instrumentation omissions and gain confounding are retained in REPORT.md. No efficacy-based changes were made to training.\n')
# Account separately for completion processes and retain all checkpoints in the package inventory.
u=resource.getrusage(resource.RUSAGE_CHILDREN)
(P/'COMPLETION_RESOURCES.json').write_text(json.dumps(dict(controller_CPU_seconds=time.process_time()-cpu,children_CPU_seconds=u.ru_utime+u.ru_stime,wall_seconds=time.monotonic()-start),indent=2)+'\n')
files=[f for f in P.rglob('*') if f.is_file() and not {'scratch','__pycache__'}.intersection(f.relative_to(P).parts) and not f.name.startswith('RESULT_BUNDLE') and f.name not in ('MANIFEST.json','PACKAGE.json')]
manifest={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files};(P/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');files+=[P/'MANIFEST.json']
parts=[];current=[];size=0
for f in files:
 if current and size+f.stat().st_size>32*1024**2:parts.append(current);current=[];size=0
 current.append(f);size+=f.stat().st_size
if current:parts.append(current)
receipts=[]
for i,items in enumerate(parts,1):
 path=P/f'RESULT_BUNDLE_PART_{i:02d}.zip'
 with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
  for f in items:z.write(f,str(f.relative_to(P)))
 with zipfile.ZipFile(path) as z:assert z.testzip() is None
 receipts.append(dict(file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(P/'PACKAGE.json').write_text(json.dumps(dict(status='COMPLETE',parts=receipts,includes_checkpoints=True,standalone=False,parent_repository_required=True),indent=2)+'\n')
ptr=R/'PROJECT_DOCUMENTS/CURRENT_EXTERNAL_MEMORY_EXPERIMENT_DESIGN.json';d=json.loads(ptr.read_text());d['next_design'].update(status='THREE_CYCLE_DEVELOPMENT_COMPLETE',implemented=True,report=str(P/'REPORT.md'),revised_design=str(P/'REVISED_DESIGN.md'),audit=str(P/'FINAL_AUDIT.json'),package=str(P/'PACKAGE.json'),native_main_records=56448,head_fits=0,new_core_adopted=False);ptr.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(status='COMPLETE',storage_screen=s['storage_screen'],integration_screen=s['integration_screen'],package_parts=len(receipts))))
