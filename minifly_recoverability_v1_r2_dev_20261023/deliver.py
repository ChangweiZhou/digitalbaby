"""Report and package completed artifacts, never teach or fit."""
from pathlib import Path
import json,hashlib,zipfile
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent
s=json.loads((P/'SUMMARY.json').read_text());resource=json.loads((P/'RESOURCE_LEDGER.json').read_text())
rows=s['final_rows']
def avg(rr,k):return float(np.mean([r[k] for r in rr]))
table=[]
for arm in ('W','CREDIT','UNIFORM','SHUFFLED'):
 row=[arm]
 for pool in ('OLD_final','NEW_final'):
  r=[x for x in rows if x['STD'] and x['arm']==arm and x['pool']==pool and x['lesion']=='NONE']
  row+=[f"{100*avg([x for x in r if x['head']=='DIRECT'],'payload_exact'):.2f}%",f"{100*avg([x for x in r if x['head']!='DIRECT'],'nonempty_exact'):.2f}%"]
 table.append('| '+' | '.join(row)+' |')
a=[]
for st in (False,True):
 rr=[x for x in s['A_pair_results'] if x['STD']==st];a.append(f"| {'STD' if st else 'RAW'} | "+' | '.join(f"{100*x['mean_conditional_gain']:+.2f} pp" if x['mean_conditional_gain'] is not None else 'undefined' for x in rr)+f" | {s['A_positive_by_storage'][str(st)]} |")
unmatched=sum(x['unmatched_compartment_events'] for x in s['uniform_matching']);diag=json.loads((P/'CREDIT_DIAGNOSIS.json').read_text())
text=f'''# RECOVERABILITY V1-R2: three-cycle report

The complete declared bounded roster finished: **46,368 new native record applications, 231,840 state-transition events and six frozen STOP-ONLY fits**. Three additional partial E0 fixture cases were used (seven native-primitive evaluations including reference/passive calculations). No native teaching was repeated, no candidates or sample counts were added, and no host/arithmetic experiment was launched.

The evidence is **E1 taught-association recovery**, not unseen-addition E3. Gradual nominal feedback, the arithmetic pointer and listed frozen parent sources match their locks. This report concerns two components, with separate decisions.

## Cycle results

1. Design/audit/test: scalar EOS head (5,537 parameters) gradient check, initial native skip, immutable payload signs, readonly probes and g=1/g=0 native-bank parity passed. Max gradient error was 1.21e-10. Three partial native fixture events, no full records. Revision: explicitly charge passive/proposal clones and freeze EOS-only masking/term semantics.
2. Design/audit/test: six heads fit on the original twelve V1 training episodes, no fresh native applications. All final weights/scales froze before fresh evaluation values. Revision: add the first complete assessment-pair resource projection; protection proceeds independently of a favorable stopping result.
3. Design/audit/test: all four counterfactual pairs, two value lives each, both baseline storage conditions and all three protection/control additions completed. Artifact-only reduction independently reconstructed every saved emission and score from raw native responses and frozen heads. Revision follows the measured component outcomes and control limitations below.

These are self-audits plus a separate reducing process, not independent reviewer approval. Source snapshots and failure records, if any, remain in cycles/. No failed/partial trajectory was silently replaced.

## A: stopping-only component

Each cell below is the mean conditional-stopping gain over the direct codec at the three prespecified usable checkpoints (OLD_end, NEW_final, passive OLD), after the declared life/head reductions. The conditioning subset is chosen by evaluator-only native payload correctness; the candidate never sees length or eligibility. All denominators and individual checkpoint gains are in SUMMARY.json.

| Storage | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Every-pair component gate |
|---|---:|---:|---:|---:|---|
{chr(10).join(a)}

The gate also requires positive mean unconditional complete-recall gain in every pair. Final written-OLD rescue was **not** required to test stopping. Conditional progress never promotes a persistent memory system. Previously inspected V1 assessment traces were not used to choose these fits.

## B: storage and fixed-readout results

Values are percentages averaged across all four pairs and both value lives; head-based complete recall also averages the three frozen seeds. Payload-only uses evaluator length for diagnosis. Complete recall requires the head's own EOS. The identical STD stopping head is applied across all arms, without arm-specific fitting.

| STD arm | OLD payload exact | OLD complete | NEW payload exact | NEW complete |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

Every-pair protection behavior gate: **{s['B_behavior_positive']}**. Clean actual-write-matched control qualification: **{s['B_clean_matched_dose']}**. There were **{unmatched} unmatched compartment-events** when the uniform arm could not reach CREDIT's current plastic-increment L1 without amplification. The largest mismatch was only {diag["uniform_max_compartment_gap"]:.3g} and the aggregate relative gap was {diag["uniform_relative_total_gap"]:.3g}. This literal strict-tolerance failure is numerically tiny and is not evidence that a materially different write budget explains the behavioral null. The predeclared saturation policy was followed; no dose was changed. SHUFFLED preserves the gate-value distribution, not automatically weighted update L1. Therefore any unmatched-dose comparison must remain mechanistically inconclusive even if some retention metrics improve.

The soft native gate changes only plastic increments. Passive dynamics and the nominal warmup continue unchanged. NEW receives later, stronger nominal coefficients; actual proposed/applied update totals and residuals are retained in the write ledgers, not inferred from equal pass counts. The local credit is an alpha-path proxy, not proof of exact native contribution or biological importance. Measured CREDIT plastic-increment L1 attenuation was {100*diag["OLD_plastic_L1_attenuation_fraction"]:.3f}% during OLD and {100*diag["NEW_plastic_L1_attenuation_fraction"]:.3f}% during NEW; the average active gate was {diag["mean_active_gate_mean"]:.6f}. This particular gate barely reduced most updates. Its null is not a test of every stronger or differently credited protection mechanism.

## Exposure and resources

72 base histories and 40 correlated passive continuations; four assessment pairs are the analysis units. Shared OLD contributes 10,080 logical equivalents, for 56,448 main information-accounting equivalents. Queries, head seeds and cloned numerical proposals are not independent lives.

A reused the original 12,096 physical native training applications and six phase pools; this historical information remains chargeable to each compared learner. No new native training applications were required for fitting. Per-fit actual update/term counters are in heads/*.json; final training-loss terms are included in each ceiling. Calibration/RMS assets were reused by hash.

Controller totals: CPU {resource['CPU_seconds']:.2f}s; summed active wall {resource['wall_seconds']:.2f}s; sampled peak tree RSS {resource['peak_tree_RSS_bytes']/1024**2:.1f} MiB. Artifact reduction/packaging costs are recorded separately. The implementation deliberately calls the unchanged native event on private passive and proposed states to compute a gated increment. There were 292,320 main native-primitive evaluations including proposal/passive decomposition, versus 231,840 effective learner events. These internal numerical evaluations reuse the same observation; they are charged compute, not new external facts or extra scientific histories. No sparse-compute or scaling advantage is established.

## Attribution and protocol limits

BASE W/N/within-length SH, ZERO/FOREIGN read lesions and each protection arm's passive-NEW continuation were retained. DETAIL_METRICS.json reduces all saved emissions into per-length counts, early/late EOS, truncation and empty-default rows, including absent-key queries. An exact raw/canonicalized key table can solve the taught-association task after observation; the training-only length prior emits empty (balanced cohorts: 11.11% all-value, zero nonempty exact). Neither reference is a candidate backend. No E3 or core-adoption claim follows.

The artifact audit verifies parent/source/head hashes, reconstructed inference, scores and counts. It does not certify every prospective engineering requirement: complete runtime/birth-asset closure was not locked before execution; native gate fixtures covered bank arrays rather than every confirmation-bookkeeping field; credit bounds/no-write/permutation behavior is implemented but not independently fixture-qualified to the full requested extent. Only final protection checkpoints and final credit/timestamp state were saved, rather than all phase-end checkpoint/credit combinations. The fine-grained bandwidth and stage-timing ledger is incomplete. These are disclosed execution limitations, not retroactive protocol changes. Frozen sources/roster were not altered to repair them after exposure.

The header inherited in head.py still says 5,801; its actual tensor count is independently checked at 5,537. This is a documentation error, not an additional output layer.

## Delivery

SUMMARY.json retains every pair decision and final/control/seed rows. traces/ contains raw response tensors and actual emitted-output records. cycles/*write_ledger.json retains per-event residuals and proposed/applied norms. FINAL_AUDIT.json accepts artifact consistency with the stated protocol limitations, not adoption or confirmatory efficacy. See REVISED_DESIGN.md for the post-cycle decision. Historical data, the arithmetic track and gradual nominal warmup are unchanged.
'''
(P/'REPORT.md').write_text(text)
revision=f'''# RECOVERABILITY V1-R2 post-cycle revision

This is a revision after executed evidence, not a replacement gate or authorization for more training.

STOP-ONLY every-pair component verdicts: RAW={s['A_positive_by_storage']['False']}, STD={s['A_positive_by_storage']['True']}. Retain the full conditional denominators and unconditional generated scores; do not select a favorite seed or use a conditional gain as retained whole-value memory.

CREDIT every-pair behavioral verdict={s['B_behavior_positive']}; uniform-dose qualification={s['B_clean_matched_dose']}. Actual-state divergence made uniform yoking infeasible on {unmatched} compartment-events. This is the anticipated control limitation, not permission to amplify writes or adjust the gate after viewing outcomes. Do not claim clean location-specific protection from this run. The largest mismatch is only {diag["uniform_max_compartment_gap"]:.3g}, aggregate relative gap {diag["uniform_relative_total_gap"]:.3g}; the behavioral gate independently fails because OLD payload recall does not improve. Treat the numerical issue proportionately.

Next decisions:

- Do not enlarge the EOS decoder. If its storage-specific component gate is positive, retain only that bounded stopping component for a separate fresh integration test; otherwise stop this head family.
- Do not promote CREDIT on averaged OLD scores. A protection advance requires the prespecified every-pair OLD improvement with no NEW loss, plus adequate dose attribution. This run cannot supply a clean matched-dose conclusion when the saturation guard activates.
- Before any further native run, complete source/birth closure and the missing credit/permutation/checkpoint qualifications. Preserve this run rather than replaying its associations.
- The next native design should first address credit coverage and actual exercised protection: this gate had mean active multiplier {diag["mean_active_gate_mean"]:.6f} and essentially unchanged OLD payload exact. Do not automatically increase its strength or use a numerical yoking discrepancy as the explanation. Derive a new credit/budget operation prospectively and define feasible, numerically robust controls. Specify that change prospectively and keep it distinct from the current candidate; no new gate or parameter search is implemented here.
- The final system still requires simultaneous complete relevant OLD/NEW own-output recovery and causal native-state attribution on fresh data. Neither component test establishes arithmetic-rule learning, revision competence, or a deployable external memory.

Gradual warmup, population, arithmetic track, no-attention/no-Transformer/no-ELM constraints remain unchanged. No follow-up launched.
'''
(P/'REVISED_DESIGN.md').write_text(revision)
(P/'cycles/CYCLE3_REVIEW.md').write_text(revision)
(P/'STATUS.json').write_text(json.dumps(dict(status='COMPLETE_WITH_PROTOCOL_LIMITATIONS',cycles=3,main_native_records=46368,head_fits=6,arithmetic_pointer_unchanged=True),indent=2)+'\n')
files=[f for f in P.rglob('*') if f.is_file() and not {'scratch','__pycache__'}.intersection(f.relative_to(P).parts) and f.name not in ('RESULT_BUNDLE.zip','PACKAGE.json','MANIFEST.json')]
manifest={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)}
(P/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile(P/'RESULT_BUNDLE.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in files+[P/'MANIFEST.json']:
  if 'checkpoints' not in f.relative_to(P).parts:z.write(f,str(f.relative_to(P)))
package=dict(status='COMPLETE',archive_sha256=hashlib.sha256((P/'RESULT_BUNDLE.zip').read_bytes()).hexdigest(),archive_bytes=(P/'RESULT_BUNDLE.zip').stat().st_size,checkpoints='retained individually outside compact archive; indexed in MANIFEST.json',standalone_audit=False,audit_requires='parent repository evidence/source files referenced by PRE_EVALUATION_LOCK.json')
(P/'PACKAGE.json').write_text(json.dumps(package,indent=2)+'\n')
print(json.dumps(package,indent=2))
