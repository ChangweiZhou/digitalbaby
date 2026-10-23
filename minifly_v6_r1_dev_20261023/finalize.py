"""Postprocessing only. Main scientific source and training are never changed."""
import ops,os,sys,subprocess,time,json,hashlib,zipfile,ctypes,traceback,shutil
from pathlib import Path
P=ops.P;R=P.parent

def stage(name):
 log=P/'results'/f'{Path(name).stem}_artifact.log'
 with log.open('w') as out:
  proc=subprocess.Popen([sys.executable,'-B',str(P/name)],stdout=out,stderr=subprocess.STDOUT,cwd=R)
  while proc.poll() is None:
   buf=ctypes.create_string_buffer(1024)
   if ops.LIB.proc_pid_rusage(proc.pid,0,ctypes.byref(buf))==0:
    rss=int(ctypes.cast(ctypes.byref(buf,16),ctypes.POINTER(ctypes.c_uint64))[6])
    for pid in (os.getpid(),ops.awake.pid):
     own=ctypes.create_string_buffer(1024)
     if ops.LIB.proc_pid_rusage(pid,0,ctypes.byref(own))==0:rss+=int(ctypes.cast(ctypes.byref(own,16),ctypes.POINTER(ctypes.c_uint64))[6])
    ops.peak=max(ops.peak,rss)
   ops.check(postprocessing=name);time.sleep(.2)
  if proc.returncode:raise RuntimeError(f'{name} failed: see {log}')
 ops.check('ARTIFACT_STAGE_COMPLETE',stage=name)
def pct(x):return f'{100*x:.3f}%'

def revision(S):
 ad=json.loads((P/'results/ADDRESS_DIAGNOSTICS.json').read_text());md=json.loads((P/'results/MARGINAL_DIAGNOSTICS.json').read_text());sd=json.loads((P/'results/SAVED_OUTPUT_DIAGNOSTICS.json').read_text());lines=['# V6-R1 revision after three actual development cycles','','The exercise is complete. The bounded addition verdict is **'+S['verdict']+'**. No confirmation, adoption or rescue training is launched. The arithmetic goal, gradual schedules, native-only payload/EOS path and architecture exclusions are preserved.','','## What was learned from the run','','All full histories and native outputs are preserved in the package. E0 interface/causal-write checks passed within the tested bounded domain. Exact taught association and the strict held-out causal screen are distinct; lower next-byte loss does not satisfy either by itself. The favorable development gate did not pass when whole-answer exactness tied the controls.','','The recurring output pattern is localized using saved outputs and observed bit priors, not another decoder fit:','','| World | Arm | Mode predicted from observed bit priors | Final outputs identical to repeated mode + truncation |','|---|---|---|---:|']
 for x in md['rows']:lines.append(f'| {x["seed"]} | {x["arm"]} | 0x{x["bit_mode_hex"]} | {pct(x["saved_final_outputs_matching_unconditional_bit_mode_fraction"])} |')
 lines+=['','This supports a marginal-byte explanation for the apparent loss improvement. The countermodel ignores order and native physiology, so agreement is evidence about failure behavior, not a proof of the internal algorithm. It is posthoc and never replaces the locked endpoint.','','Even the generous, offline target-length prefix diagnostic is separate from own-output success:','','| World | Arm | OLD payload-prefix upper bound | NEW payload-prefix upper bound | Held payload-prefix upper bound |','|---|---|---:|---:|---:|']
 for h in S['history_statistics']:
  x={r['pool']:r for r in sd['rows'] if r['seed']==h['seed'] and r['arm']==h['arm']};lines.append(f'| {h["seed"]} | {h["arm"]} | '+ ' | '.join(pct(x[k]['oracle_target_length_payload_prefix_upper_bound']) for k in ('old','new','held'))+' |')
 lines+=['','True length is used only in this offline upper bound. Margins after a wrong generated prefix cannot establish teacher-forced EOS recognition. Do not call the observed failure an EOS-only defect.','','## Sensory signal versus random state','','The controller/address diagnostic uses eight OLD one-digit-pair prompts per world, equal-length inputs, three fixed tapes, no labels, no held prompts, no native events and no learning. Its scope is E0 localization:','','| World | State | Same prompt / different tapes, h distance | Different prompts / same tape, h distance | Ratio | Same-prompt different-tape KC Jaccard | Different-prompt same-tape KC Jaccard |','|---|---|---:|---:|---:|---:|---:|']
 for x in ad['rows']:lines.append(f'| {x["seed"]} | {x["stage"]} | {x["same_prompt_different_tapes_h_distance"]:.6f} | {x["different_prompt_same_tape_h_distance"]:.6f} | {x["paired_prompt_difference_over_tape_difference"]:.6f} | {x["same_prompt_different_tapes_KC_Jaccard"]:.5f} | {x["different_prompt_same_tape_KC_Jaccard"]:.5f} |')
 lines+=['','The birth formula supplies only one active symbol column with SD0.05/sqrt348≈0.00268, while each open content proposal has noise SD0.1. The measured tape-versus-prompt contrast should therefore be examined before claiming learned abstraction or changing native capacity. A code can be mathematically order-sensitive yet vary much more with random innovations than with byte identity.','','Working hypothesis, not a causal proof: weak cue signal produces unstable conditional addresses; native writes learn pooled bit frequencies; those near-unconditional values give little useful reward contrast to the representation learner. Correct local score derivatives do not by themselves break this joint-learning bottleneck. This explains why stable old addresses could support recitation while a new stochastic front end can fail even taught exact answers, without concluding that native learning or recurrence is impossible. Native capacity/interference, credit variance and coupled readout conditioning remain alternative contributors.','','## Concrete revision','','1. Keep the executed main kernel immutable. A separate `core_boundary_revision.py` enforces input/generation caps and rejects invalid payloads before mutation. Its zero-native-event tests pass; it changes no valid bounded arithmetic computation and does not repair the negative performance.','2. Preserve R1 as the comparator. `NEXT_DESIGN.md` proposes one generic active-fan-in initialization correction, with native values, stochastic rule, sigma, warmup, observed data and endpoint retained. All byte columns stay exchangeable and learned. It does not supply arithmetic roles, rules or a separate answer network.','3. Compare that revision against original R1, fixed-controller, no-prior-teaching and shuffled-label controls on fresh locked engineering worlds. Keep acquisition and held-out retention/sample/compute curves together. Do not launch confirmation of R1, increase n after failure, add a decoder, replay or population expansion to rescue this run.','','Block rescaling also changes the native operating distribution and gate statistics; a favorable result would establish an integrated initialization intervention, not a pure mechanistic proof that noise alone caused the original failure. Further attribution would need a prospective control.','','## Audit limitations retained, not erased','','- Cycle 1 failed on an evaluator import-name collision after two synthetic records. Source and error are preserved; cycle 2 fixed module isolation and passed the bounded suite. No main trajectory was retried.','- The input-overflow check was missing from the frozen query API, although every actual addition prompt was within its cap. The separate wrapper is the tested fix.','- The original source lock covered identical vendor asset copies rather than some actual upstream birth-asset locations. Their hashes match; seven dormant configuration differences are document-path metadata. `ASSET_LOCATION_AUDIT.json` records the omission and equivalence, and each actual birth validates canonical anatomy.','- The independent artifact reducer was written during execution after baseline probes. Main scoring and decision rules were frozen beforehand. State, saved-output and frequency localization are supplemental; their posthoc timing is explicit.','- FROZEN also leaves its unused reward baseline fixed; because its weights never update, that baseline cannot affect actions. This is an operational deviation in ignored diagnostic state, not a causal treatment difference.','- Technical private-generation calls were timed but not individually counted: two runs have a conservative bound of 300 private native event applications (2×[2×(5+17)+2×3×17+1+3]). There are exactly20 observed full-record events and four separate partial fixture events. Even charging the full bound gives at most324 technical native event applications, below512. Main observed and private generation counts are exact.','- Hardware memory traffic and native clipping frequency were not instrumented. Dense scans, actual CPU, arrays, projection edges, output margins and sampled RSS are reported. Early first-suite import CPU was not timestamped; an explicit overhead reserve is charged in the final package.','','The full bounded result is a failed addition-learning candidate under these conditions, not an arithmetic impossibility result and not an exclusion of every mechanism in the 254-entry register. No new task is substituted for addition.']
 (P/'REVISION.md').write_text('\n'.join(lines)+'\n')

def main():
 ops.begin();success=False
 try:
  assert json.loads((P/'results/SESSION_RESULT.json').read_text())['status']=='MAIN_COMPLETE_AWAITING_AUDIT'
  for name in ('audit_results.py','audit_checkpoints.py','test_boundary_revision.py','diagnose_saved.py','diagnose_marginals.py','diagnose_addresses.py'):stage(name)
  S=json.loads((P/'results/SUMMARY.json').read_text());A=json.loads((P/'results/FINAL_AUDIT.json').read_text());revision(S)
  stage('report.py')
  with (P/'REPORT.md').open('a') as f:f.write('\n## Additional review findings\n\nSee `REVISION.md` for the measured sensory/tape contrast and saved-output/bit-prior diagnostics. These are posthoc localization, not new primary endpoints. The original query API missed the input cap; a separate zero-event boundary revision passes, while the frozen arithmetic path remains unchanged. The source-lock location omission, unused FROZEN baseline difference and technical private-query count bound are explicitly retained there. The bounded results remain valid within the manifest; they do not amount to full protocol conformance for an unrestricted API.\n')
  A.update(status='PASS_WITH_DISCLOSED_LIMITATIONS',data_integrity_status='PASS',checkpoint_audit=json.loads((P/'results/CHECKPOINT_AUDIT.json').read_text()),boundary_revision_test=json.loads((P/'cycles/03_BOUNDARY_REVISION_TEST.json').read_text()),technical_private_native_events_upper_bound=300,all_technical_native_events_upper_bound=324,protocol_conformance='See REVISION.md: bounded numerical pilot valid; API cap, asset path lock, supplemental audit timing and technical private-event telemetry omissions disclosed',records_or_training_trajectories_replayed=0)
  ops.save('results/FINAL_AUDIT.json',A)
  for path,h in json.loads((P/'UNCHANGED_TRACK_LOCK.json').read_text()).items():assert ops.sha(R/path)==h
  # Verify supplemental audited code as well as the original scientific lock.
  for path,h in json.loads((P/'AUDIT_SOURCE_LOCK.json').read_text())['files'].items():assert ops.sha(P/path)==h,path
  ops.save('cycles/03_RESULT.json',dict(status='COMPLETE_NEGATIVE_ENGINEERING',verdict=S['verdict'],main_records=S['main_records'],observed_native_events=S['main_observed_events'],histories=S['histories'],primary_gate_passed=False,formal_confirmation=False,new_core_adopted=False,revision='REVISION.md',proposed_next_design='NEXT_DESIGN.md',api_only_revision='core_boundary_revision.py'))
  (P/'cycles/03_REVIEW.md').write_text('# Cycle 3 — result and revision\n\nThe full frozen addition roster completed once. Saved native margins, counters, source hashes and 72 checkpoint arrays were audited independently. The fixed held-out screen did not pass. See REPORT.md and REVISION.md for absolute scores, frequency-mode and address diagnostics, and all protocol limitations. The API cap defect is fixed only in a separate wrapper, with zero-native-event tests; no performance is backfilled or training retried. NEXT_DESIGN.md remains a proposal, not a launched rescue.\n')
  # Carry provenance memos as documents; the original delivery remains untouched.
  for name in ('EVIDENCE.json','CLAUDE_SOURCE.json','CLAUDE_V6_ORIGINAL.md','REVIEW.md'):
   shutil.copy2(R/'minifly_v6_revised_design_20261023'/name,P/'design'/name)
  ptr=R/'PROJECT_DOCUMENTS/CURRENT_PROCEDURAL_EXPERIMENT_DESIGN.json';v=json.loads(ptr.read_text());v.update(status='THREE_CYCLE_DEVELOPMENT_COMPLETE_NEGATIVE_AUDITED',technical_qualification='bounded path passed; API boundary defect separately revised; see report',development_complete=True,formal_confirmation=False,new_core_adopted=False,main_record_applications=S['main_records'],main_observed_native_events=S['main_observed_events'],report=str(P/'REPORT.md'),final_audit=str(P/'results/FINAL_AUDIT.json'),revision=str(P/'REVISION.md'),next_design_proposed_not_launched=str(P/'NEXT_DESIGN.md'),package=str(P/'results/PACKAGE.json'));ptr.write_text(json.dumps(v,indent=2)+'\n')
  # Leave caches/live logs outside the immutable scientific bundle.
  (P/'.gitignore').write_text('scratch/\n__pycache__/\nresults/RESULT_BUNDLE.zip\n')
  ops.save('results/FINAL_RESOURCE_SNAPSHOT.json',dict(**ops.stats(),untracked_setup_orchestration_CPU_reserve=60,note='Snapshot before archive compression; PACKAGE.json reports resources through archive verification. CPU reserve conservatively charges first-suite import and orchestration metadata, not uncounted training.'))
  exclude={'results/RESULT_BUNDLE.zip','results/PACKAGE.json','results/RESOURCE_LEDGER.json','results/STATUS.json','results/finalize.log','MANIFEST.json'}
  files=[q for q in sorted(P.rglob('*')) if q.is_file() and 'scratch' not in q.parts and '__pycache__' not in q.parts and str(q.relative_to(P)) not in exclude and not q.name.endswith('.pending') and '.zip.part-' not in q.name]
  manifest={str(q.relative_to(P)):dict(bytes=q.stat().st_size,sha256=ops.sha(q)) for q in files};ops.save('MANIFEST.json',dict(files=manifest,excluded_live_files=sorted(exclude),dependencies='SOURCE_LOCK.json and ASSET_LOCATION_AUDIT.json; parent repository required for execution'))
  files.append(P/'MANIFEST.json');target=P/'results/RESULT_BUNDLE.zip'
  with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
   for i,q in enumerate(files):
    if i%10==0:ops.check(packaging_file=i)
    z.write(q,arcname=str(Path(P.name)/q.relative_to(P)))
  with zipfile.ZipFile(target) as z:
   for name in z.namelist():
    rel=str(Path(name).relative_to(P.name))
    if rel=='MANIFEST.json':continue
    assert hashlib.sha256(z.read(name)).hexdigest()==manifest[rel]['sha256'],name
  parts=[]
  if target.stat().st_size>90*1024**2:
   with target.open('rb') as f:
    i=1
    while True:
     blob=f.read(80*1024**2)
     if not blob:break
     q=target.with_name(target.name+f'.part-{i:03d}');q.write_bytes(blob);parts.append(dict(file=q.name,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()));i+=1
  resources=ops.stats();assert resources['CPU_seconds']+60<7200
  ops.save('results/PACKAGE.json',dict(status='COMPLETE',bundle=target.name,bundle_bytes=target.stat().st_size,bundle_sha256=ops.sha(target),verified_members=len(files),manifest='MANIFEST.json',parts=parts,part_reassembly='Concatenate numbered .part files in order; SHA256 must match bundle_sha256.',resources_through_archive_verification=resources,conservative_CPU_with_60_second_overhead_reserve=resources['CPU_seconds']+60,scientific_verdict=S['verdict'],all_three_cycles_complete=True,new_core_adopted=False))
  ops.save('results/STATUS.json',dict(status='COMPLETE_NEGATIVE_ENGINEERING',verdict=S['verdict'],main_records=S['main_records'],main_observed_native_events=S['main_observed_events'],accepted_data_audit=True,protocol_limitations='REVISION.md',package='PACKAGE.json',**resources));success=True
 except BaseException as e:
  ops.save('results/POSTPROCESS_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
 finally:ops.end()
if __name__=='__main__':main()
