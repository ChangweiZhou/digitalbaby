"""Documentation and supplementary saved-panel reductions only; no model execution."""
import util as U
import sys,json,time,zipfile
sys.path.insert(0,str(U.ROOT/'minifly_v178_recurrent_failure_localization_dev_20261102/local/compat'))
import numpy as np
S=U.load('SUMMARY.json');R=U.load('RESULTS.json');rows=R['checkpoints'];q={(r['world'],r['checkpoint']):r for r in rows};paired={};degrade={};gen=[]
for w in ('W1','W2'):
 sub=np.load(U.H/f'local/{w}_subsets.npz');ch=sub['changed'];old=sub['old'];pp=[]
 for arm in ('EDITED','NOEDIT'):
  r=q[w,arm+'_C2']['scores']['EDITED_held'];z=np.load(U.H/r['panel']);y=z['y'];lp=z['log2p'];pp.append((lp[np.arange(len(y)),y],lp[np.arange(len(y)),old],z['n']))
 n=pp[0][2];ends=np.cumsum(np.r_[0,n]);loss=pp[0][0]-pp[1][0];margin=(pp[0][0]-pp[0][1])-(pp[1][0]-pp[1][1]);arr=[]
 for i in range(len(n)):
  mask=ch[ends[i]:ends[i+1]];arr.append(dict(record=i,changed_targets=int(mask.sum()),new_loss_advantage_sum=float(loss[ends[i]:ends[i+1]][mask].sum()),extra_margin_sum=float(margin[ends[i]:ends[i+1]][mask].sum())))
 paired[w]=arr;degrade[w]={arm:q[w,arm+'_B3']['scores']['B_held']['bits']-q[w,arm+'_B1']['scores']['B_held']['bits'] for arm in ('KEEP','RESET_OPT','B_COLD')}
for r in rows:
 for name,g in r['generation'].items():gen.append(dict(world=r['world'],checkpoint=r['checkpoint'],corpus=name,**g))
U.save('SUPPLEMENTARY_SAVED_ANALYSIS.json',dict(revision_record_pairs=paired,B_pass3_minus_pass1_held_bits=degrade,all_B_arms_worsened=all(v>0 for d in degrade.values() for v in d.values()),generation_complete_exact_total=sum(g['complete_exact'] for g in gen),generation_record_queries=sum(g['records'] for g in gen),generation_emitted_targets=sum(g['output_targets'] for g in gen),posthoc_interpretation_only=True,no_gates_changed=True))
U.save('THREE_CYCLES.json',dict(status='THREE_CYCLES_COMPLETE_LIFETIME_DEV_GAPS',cycles=[dict(cycle=1,design='V179 source and authenticated data/adaptation contract',audit='full-vocabulary source, empty-input column, raw-edition hashes, train-only edits and held coverage',test='literal sensory parity, full-byte and future-target-independence fixtures',revision='fixed metadata serialization, explicit empty column and namespaced dependency adapter; original model algebra unchanged'),dict(cycle=2,design='local learning and continuous state contract',audit='original-source gradient parity, finite differences of local surrogate/readout, zero feedback and optimizer-fork invariants',test='24 full technical records; save/restore and exact next-step equality; read-only output/generation smoke',revision='source/data/execution lock and full-program resource admission before scoring histories'),dict(cycle=3,design='fixed two-world full lifetime roster',audit='saved full257 panels, dose ledger, raw manifest, source/checkpoint hashes and fork contrasts',test='all7532 scheduled main records; all B and C arms executed despite weak early performance',revision='retain mixed evidence; diagnose repeated-pass deterioration before adding a mechanism; no extra training')],replications='two opened development worlds; no independent confirmation',main_trajectory_retries=0))
# Close process accounting with explicit conservative allowances, not invented precision.
l=U.load('PROCESS_LEDGER.json');l['completed'].append(dict(name='source_lock_finalization_and_small_orchestration',CPU=10,CPU_is_conservative_allowance=True,training_records=0));U.save('PROCESS_LEDGER.json',l)
main=next(x for x in l['completed'] if x['name']=='lifetime_all_branches');cpu=sum(x['CPU'] for x in l['completed']);peak=max(x.get('RSS_peak',0) for x in l['completed']);S.update(CPU_accounted_before_documentation=cpu,CPU_includes_conservative_allowances=True,peak_RSS_bytes=peak,wall_seconds_to_report=time.time()-U.load('AUTHORIZATION.json')['epoch'],generation_complete_exact=sum(g['complete_exact'] for g in gen),generation_record_queries=sum(g['records'] for g in gen));U.save('SUMMARY.json',S)
report='''# V179 continuous-life local learner: three-cycle result

**Verdict: LIFETIME_DEV_GAPS. The exercise completed, but the full advancement criteria did not pass.** The local learner exhibits bounded prior-experience reuse and feedback-specific revision on both opened development worlds. Acquisition and retention are inconsistent, and complete autonomous continuations remain unsuccessful. No new core is adopted.

## Registered outcomes

| Capability | W1 Thursday→Edgeworth | W2 Emma→Ball |
|---|---|---|
'''
for name in ['acquisition','reuse','retention','revision']:report+=f"| {name} | {'PASS' if S['worlds']['W1'][name]['pass'] else 'FAIL'} | {'PASS' if S['worlds']['W2'][name]['pass'] else 'FAIL'} |\n"
report+='''
These are development screens, not tests of statistical significance. Both text pairs were previously opened. Positions, repeated passages and forked histories are correlated, not thousands of independent worlds.

## What was learned, retained and revised

| Quantity (bits per actual target; lower loss is better) | W1 | W2 |
|---|---:|---:|
'''
quant=[('A held at birth',lambda v:v['acquisition']['birth_A']),('A held after A',lambda v:v['acquisition']['after_A']),('A train-fitted unigram',lambda v:v['acquisition']['unigram']),('B-COLD minus RESET-OPT after one B pass',lambda v:v['reuse']['1']['cold_minus_reset']),('B-COLD minus RESET-OPT after three B passes',lambda v:v['reuse']['3']['cold_minus_reset']),('Stable A loss change after B',lambda v:v['retention']['A_stable_after_B_minus_after_A']),('Stable A loss change after edited C (relative to after A)',lambda v:v['retention']['A_stable_after_C_minus_after_A']),('B loss change after edited C',lambda v:v['retention']['B_after_C_minus_after_B']),('Correction margin advantage vs NOEDIT',lambda v:v['revision']['extra_margin_gain']),('Corrected-target loss advantage vs NOEDIT',lambda v:v['revision']['loss_advantage'])]
for label,f in quant:report+=f"| {label} | {f(S['worlds']['W1']):.6f} | {f(S['worlds']['W2']):.6f} |\n"
report+='''
Acquisition requires ≥0.20-bit improvement from birth **and** ≥0.05 bits beyond the A unigram. W1 learns substantially from birth but finishes 0.128381 bits worse than the unigram. W2 beats its unigram by 0.211299 bits. Reuse requires ≥0.05 bits at B passes 1 and 3 for RESET-OPT versus B-COLD; both worlds pass. This supports operational transfer of A-trained parameters, not abstract rule discovery. KEEP's continuing optimizer improves practical B loss over cold as well; the relative KEEP/reset effect varies by checkpoint/world.

Retention gates are ≤0.10 stable-A degradation after B, ≤0.15 after C relative to after A, and ≤0.10 B degradation after C. W2 fails after B (+0.177351) and for B after C (+0.127670). W1's stable-A means improve; this is not a case where every new phase causes forgetting. Mean improvement does not mean every site improved: see per-site degradation counts and paired-record summaries.

Revision requires ≥0.20-bit extra new/old log-odds gain and ≥0.10-bit corrected-target loss benefit over NOEDIT. Both pass by substantial margins on 416 and 233 corrected held sites. EDITED and NOEDIT start from identical KEEP parameters, Adam state and dropout RNG; they observe the same 64 original record IDs, equal target counts and two presentations, with only legitimate raw-byte corrections differing. These train-derived edits define a synthetic environment, not factual/semantic understanding. They do not give the learner privileged keys, rules, task labels or old/new identities.

## Learning deterioration is a separate finding

All six B branches have worse held loss after three passes than after one:

| Branch: B3−B1 held loss | W1 | W2 |
'''
for arm in ('KEEP','RESET_OPT','B_COLD'):report+=f"| {arm} | {degrade['W1'][arm]:.6f} | {degrade['W2'][arm]:.6f} |\n"
report+='''
This is descriptive evidence of an unfavorable dose/optimization trajectory under this fixed chronological schedule. The saved data do not distinguish training underfit, overfit, recency bias, writer magnitude or local-credit geometry causally. Do not diagnose it solely as forgetting, prove a representation impossibility, or choose a validation-best pass after seeing these scores. The primary checkpoints remain the fixed final passes.

The chosen calibrated count controls predict B at 2.916540/3.018339 bits, versus KEEP at 4.533722/4.441904. The gap is 1.617182/1.423566 bits. The countermodels are evaluator-side, fitted using their own disclosed training/calibration budget, and never supply the learner's answers.

## Own bytes/EOS and evidence boundary

Every neural score comes from the same unmasked own 257-class head: payload 0–255 and genuine EOS256. Full next-byte panels use observed prefixes. Separately, all A/B held records receive three observed prefix bytes and then the model's own argmax continuation for up to32 bytes/EOS; truncation is explicitly reported. Complete continuations score **0 exact over all registered generation queries**. Most do not emit EOS before the cap; in W2 only a few pre-correction continuations do. No forced EOS or decoder repair was used. Predictive loss gains therefore cannot be presented as complete autonomous answers.

This is E0-qualified access plus bounded predictive-lifetime development evidence. It does not establish a complete-response E1 battery, E2 relevance/retention, E3 never-taught addition, E4 broad cognition or biological validity. No attention, Transformer, arithmetic representation, BPTT or evaluator-fitted learner head was introduced. The eventual unseen-addition objective remains unmet.

The descriptive exposure audit finds no identical train/held records, but some prefixes and labels overlap. Exact-prefix, canonical-prefix and suffix-16 tables with unigram fallback are reported, not silently excluded. Learned conditional frequencies remain an explanation for reuse/revision. A two-byte environmental edit can be learned statistically; passing it is not evidence of reasoning. The STABLE subset requires unchanged expected bytes and preceding16 observed bytes; the longer decaying sensory trace is not asserted identical. Retention is scored on the original still-valid held contexts.

## Fidelity, three cycles and limits

Cycle1 recovered pinned V167/V169/V163 source, authenticated all corpora, froze train-only edits and audited sensory timing. It preserved V167's input empty-state column5177 and pad5178; every original A train/calibration code matches the V176 cache after that explicit representation adapter. B and other panels are newly exported with the authenticated literal sensory function; no historical B binary-cache comparison is claimed. Full-byte fixtures compare literal native and exported codes, and changing a future byte leaves earlier codes unchanged.

Cycle2 used the literal V167 local_step, fixed feedback seeds21/22 and original V163 births. Independent synapse gradients, local-surrogate/readout finite differences, zero hidden-feedback, independent forks and save→restore→next-update equality passed. The local hidden teaching signal is random output-error broadcast; gradients do not pass through learned neighboring layers. AdamW is retained as an explicit optimizer, not claimed to be an animal synapse.

Cycle3 executed every registered A/B/C history on both worlds without efficacy-based gating, resets of the primary trajectory, best-checkpoint rescue, additional passes or retries. A is physically learned once per world; KEEP, RESET-OPT and B-COLD fork as declared. C forks the actual final KEEP checkpoint. Source/data, starting RNG, checkpoint hashes and complete full-class saved-output panels support a separate-process saved-only reduction. This is an internal artifact audit, **not an external independent reconstruction**. Raw panels/checkpoints remain local; publication manifests make that limit explicit.

Source/setup issues corrected during engineering are preserved in history and the ledger: NumPy metadata serialization, a dependency-order check, a generic module-name collision and loading existing native dependencies into the Torch environment. None involved a main scientific optimizer update or a scientific retry. A malformed historical commit string in SPEC is resolved using the full pinned commit in DESIGN.md, disclosed in SOURCE_RECOVERY.json. No historical source is modified.

V179 explicitly freezes **no added warmup**, unlike the broader project's warmup preference. This is the requested literal baseline adaptation; it does not adopt a new curriculum or settle whether gradual warmup helps. V179 changes the old A-specific OTHER interface to257, chronological batching and optimizer continuity; it is not a numerical replication of historical V169. Those adaptations are confounded when comparing across versions, so historical 3.443-bit performance cannot be attributed to a single changed mechanism.

## Exposure, cost and release

'''
report+=f"Main learning consumed **7,532 full-record presentations / {S['main_targets']:,} target applications**. Technical optimizer tests used24 records /1,401 targets. Total: **7,556 records /464,678 supervised target applications**, within7,600/510,000. These include repeated exposure; each descendant inherits the A learning cost. The1832 recorded science optimizer updates exclude the separately charged technical steps. Gradient-only qualification sites, read-only target panels, own generation, sensory export and count-model fits are separately disclosed and charged CPU. The count controls fit {U.load('COUNT_CONTROLS.json')['fit_record_presentations']:,} record presentations /{U.load('COUNT_CONTROLS.json')['fit_target_presentations']:,} targets; the posthoc table audit has its own ledger.\n\n"
report+=f"The full sequential scientific worker used **{main['CPU']:.2f} CPU seconds** and {main['wall']:.2f} wall seconds. Across export, technical checks, count controls, saved-only reduction and conservative engineering allowances, accounting before final documentation is **{cpu:.2f} CPU seconds**; peak process RSS is {peak/1e6:.1f} MB. Final receipt adds packaging/documentation cost. Pure immutable sensory caches avoid repeated encoding; no learned activations are cached. The dense H/T learner still performs work over its matrices; this is not a sparse-computation scaling claim. No global environment install, GPU migration or model-equation optimization was needed.\n\n"
report+='''The revised document is a prospective diagnostic plan, not permission to rerun V179 or invent a successor automatically. Existing source locks, final losses and failure gates remain unchanged. Package contains compact code, summaries and manifests; raw data, checkpoints, dependencies and full raw output panels are excluded.
'''
(U.H/'REPORT.md').write_text(report)
(U.H/'REVISED_DESIGN.md').write_text('''# V179-R1 — lifetime-result revision

**State:** documentation revision following the completed V179-CLLR-D1 three-cycle exercise. No new version, successor training, additional passes, architecture change or adoption is authorized here. D1's source locks, dose, gates and results remain immutable. Verdict: LIFETIME_DEV_GAPS.

## What changes in the working diagnosis

The operative local learner can reuse A-trained parameters on B and shift held predictions following changed feedback in both opened text worlds. Therefore “local learning cannot adapt” is too strong. It cannot yet acquire useful structure consistently beyond unigram or retain all still-valid predictions across the same lifetime. Its complete own continuations remain unsuccessful. Count models retain a large predictive advantage.

The most reproducible newly observed issue is worsening B held loss from pass1 to pass3 in KEEP, RESET-OPT and B-COLD on both worlds. Because cold also worsens, prior-memory interference cannot explain the entire problem. Because KEEP and RESET both worsen, merely restarting Adam moments is not a sufficient remedy. This localizes an issue to the present learning/data/update regime without proving which component causes it.

## Next diagnostic, before another mechanism

First prospectively instrument the **same** fixed257 local learner, immutable FE code and chronological raw-byte stream. Add same-checkpoint train and held losses, output entropy/EOS/unused-class mass, and local layer update/activation/moment summaries at every prescribed pass. Save pass1/2 as well as final states and exact label/site denominators. Never restore a best pass for the continuing life. This distinguishes improved training with worsening held prediction from degradation on both; the current saved held panels cannot decide that distinction.

The diagnostic must retain a complete A→B→edited-A trajectory with KEEP, RESET-OPT, B-COLD and NOEDIT, and report acquisition/retention/revision/reuse together. Weak acquisition must not suppress later tests. Fresh future worlds, a complete exposure ledger and prospectively fixed cost remain necessary for stronger claims. This instrumented follow-up is not launched by this revision.

If the diagnostic shows excess overwrite/recency from correlated updates, the next single local intervention should target update estimation or writer magnitude, with a matched exposure/optimizer/compute control. If training loss itself remains poor, study local credit/head learning before proposing consolidation. If next-byte prediction improves but own generation/EOS remains poor, isolate that deployment/output mismatch rather than substituting a fitted decoder. Choose the intervention from those measured alternatives; do not append an arbitrary topology, new bank or mechanism list as a serial prerequisite.

## Controls that must survive

- Fixed257 own byte/EOS head, no OTHER, forced EOS, supplied arithmetic structure, attention, Transformer, pretrained answerer or learned external readout.
- Original source-local update algebra unless a later explicitly named single intervention changes it. Preserve all optimizer/RNG state in the primary life.
- No free A pretraining, hidden replay, changed sample count, chosen authors, best-checkpoint reset or more passes after a negative score.
- Acquisition must beat the prespecified frequency reference, not only random birth. Revision must beat the same-state same-dose old-feedback control. Retention must distinguish still-valid from intentionally obsolete labels.
- Teacher-forced/predictive gains are not complete own-byte/EOS competence. Conditional-frequency learning remains a countermodel; the later never-taught addition requirement is unchanged.

## Warmup and comparison boundary

D1 deliberately retained the source's no-added-warmup schedule. A gradual-warmup treatment remains a possible later, separately controlled update-schedule hypothesis, not a retrofit to these results. Fixed257, chronological batching and optimizer continuity differ jointly from V169; do not attribute the cross-version gap to one of them without a matched control. Any further design must justify its unique intervention and budget before training, without automatically allocating V180.
''')
# Save artifacts and a compact package; historical raw results are never re-executed.
U.finish_process('documentation_and_supplementary_saved_analysis')
final=U.load('FINAL_AUDIT.json');final['supplementary_saved_analysis']='SUPPLEMENTARY_SAVED_ANALYSIS.json';final['all_six_B_curves_worsen']=True;final['record_level_revision_effects_saved']=True;U.save('FINAL_AUDIT.json',final)
files=[p for p in U.H.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json') and p.name not in ('PACKAGE.json','ACTIVE.json')]
archive=U.H/'RESULT_BUNDLE.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(files):z.write(p,p.name)
U.save('PACKAGE.json',dict(status='COMPLETE',archive=archive.name,sha256=U.sha(archive),files={p.name:U.sha(p) for p in sorted(files)},excluded=['raw texts/encoded data','raw output panels','checkpoints','local dependencies','historical source copies'],external_raw_reconstruction=False))
print('REPORT_AND_REVISED_DESIGN_COMPLETE')
