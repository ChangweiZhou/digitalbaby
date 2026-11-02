"""Documentation from saved causal comparisons; no learner execution."""
import util as U
import json,time,zipfile
S=U.load('SUMMARY.json');R=U.load('RESULTS.json');q={(r['world'],r['checkpoint']):r for r in R['checkpoints']};pairs=U.load('PAIRED_RECORD_EFFECTS.json');report='''# V179-PLASTICITY-1 — paired hidden-plasticity diagnostic

**Completed bounded causal-package diagnostic. Official run not launched; no new model adopted.**

'''
report+=f"Prespecified interpretation: **{S['interpretation']}**. This compares ordinary continuing local H/T/O learning with a temporary H/T learning clamp during B, from the exact same A-trained state. Both branches then resume ordinary hidden learning for C.\n\n"
report+='''## Primary comparison: B held prediction

Bits per actual byte/EOS target; lower is better. Effect is CLAMP−NORMAL, so a negative value favors the clamp. Final pass3 is primary; pass1/2 are reported without choosing a preferred stopping point.

| World | B pass | NORMAL train | CLAMP train | NORMAL held | CLAMP held | CLAMP−NORMAL held |
|---|---:|---:|---:|---:|---:|---:|
'''
for w,d in S['worlds'].items():
 for ep in ('1','2','3'):
  v=d['B'][ep];report+=f"| {w} | {ep} | {v['normal_train']:.6f} | {v['clamp_train']:.6f} | {v['normal_held']:.6f} | {v['clamp_held']:.6f} | {v['clamp_minus_normal_held']:+.6f} |\n"
report+='''
| World | Branch | Δtrain B1→B3 | Δheld B1→B3 | Pattern |
|---|---|---:|---:|---|
'''
for w,d in S['worlds'].items():
 for arm in ('NORMAL','CLAMP'):
  v=d['B'][arm+'_trajectory'];report+=f"| {w} | {arm} | {v['delta_train_B3_minus_B1']:+.6f} | {v['delta_held_B3_minus_B1']:+.6f} | {v['category']} |\n"
report+='''
Numerical tie tolerance1e−4bits is not a statistical significance threshold. The saved paired-record effects retain the distribution of differences. Positions, repetitions, forks and records within a text are not independent worlds; both literary-text pairs were previously opened.

## Continuation after the intervention

H/T learning is switched back on for both C branches. The clamp branch resumes its actual A-preserved hidden moments/steps; the normal branch carries its actual B-adapted hidden moments. Neither receives a reset, synthetic moment aging or best-checkpoint reload. Each B endpoint is forked into equal-dose EDITED and NOEDIT feedback, with identical initial parameters/moments/RNG for that pair.

| World | B history | New/old margin advantage EDITED−NOEDIT | Changed-target loss benefit NOEDIT−EDITED | Revision screen | Stable A change after C vs after A | B loss change after C vs after B | Retention screen |
|---|---|---:|---:|---|---:|---:|---|
'''
for w,d in S['worlds'].items():
 for arm in ('NORMAL','CLAMP'):
  v=d['C'][arm];report+=f"| {w} | {arm} | {v['edited_minus_noedit_margin_advantage']:.6f} | {v['noedit_minus_edited_loss_advantage']:.6f} | {'PASS' if v['revision_screen'] else 'FAIL'} | {v['stable_A_after_C_minus_A']:+.6f} | {v['B_after_C_minus_B']:+.6f} | {'PASS' if v['retention_screen'] else 'FAIL'} |\n"
report+='''
These are the original V179 revision and retention screens: margin advantage≥0.20bits and corrected-target loss benefit≥0.10; stable A degradation≤0.10 after B and≤0.15 after C, B degradation≤0.10 after C. They are development screens, not confirmation tests. We do not re-label an inherited A acquisition/reuse result as a new test of those capabilities. Source A learning and historical B-COLD/RESET results remain separately scoped.

| World | History | B held at B3 | B held after edited C2 | A held after edited C2 |
|---|---|---:|---:|---:|
'''
for w,d in S['worlds'].items():
 for arm in ('NORMAL','CLAMP'):
  v=d['C'][arm];report+=f"| {w} | {arm} | {d['B']['3'][arm.lower()+'_held']:.6f} | {v['final_B_loss']:.6f} | {v['final_A_loss']:.6f} |\n"
report+='''
## What this establishes and leaves unresolved

'''
if S['interpretation']=='CLAMP_HELD_BETTER_BOTH_OPENED_WORLDS':
 report+='''Preserving the A-trained hidden package during B improves final held prediction in both opened worlds. Under this fixed dose/history, continued hidden plasticity is a contributor to the weaker final B prediction. The result localizes a useful causal handle; it does not establish that the learner cannot acquire taught information or that all feature learning is harmful. The full B curves show whether an early benefit reverses later. Any held improvement must be considered alongside the C continuation results above.

'''
elif S['interpretation']=='NORMAL_HELD_BETTER_BOTH_OPENED_WORLDS':
 report+='''Continuing hidden learning gives better final held prediction than preserving A's hidden state in both worlds. Under this comparison, the hidden-plasticity package provides a net predictive benefit. The earlier train/held divergence remains real, but stopping hidden learning does not repair it. This directs attention to other or interacting causes without proving that the readout or data order is uniquely responsible.

'''
else:report+='''The package effect is mixed or numerically tied across worlds. The comparison does not identify a consistent intervention to adopt. Preserve all signs and continuation tradeoffs instead of selecting a favorable world or allocating another architecture automatically.

'''
report+='''The clamp jointly preserves hidden weights, biases, Adam moments and per-parameter optimizer age and removes hidden writing. NORMAL and CLAMP can consequently produce different own-output gradients and write amounts. The test does **not** isolate representation drift from effective plastic capacity, writer magnitude, moment history, H versus T, or generalization effects of the output layer. It also does not prove that freezing features is a suitable final architecture. Hidden representations are A-trained features, not a new random-feature learner.

The C comparison is a downstream consequence of that B history, including its different optimizer state. A C failure cannot be attributed solely to weight unfreezing. An improved B endpoint cannot be promoted while suppressing a continuation failure. A future single intervention must target the measured history-dependent tradeoff with a matched comparator; it is not authorized by this report.

## Actual writing and integrity

| World | B arm | H sum squared updates | T sum squared updates | O sum squared updates |
|---|---|---:|---:|---:|
'''
for w,d in S['actual_B_update_energy'].items():
 for a,v in d.items():report+=f"| {w} | {a} | {v['H']['sum_update_squared']:.6f} | {v['T']['sum_update_squared']:.6f} | {v['O']['sum_update_squared']:.6f} |\n"
# Supplementary read-only analysis of the registered update traces, not a new intervention.
trace=[json.loads(line) for line in (U.H/'UPDATE_TRACE.jsonl').read_text().splitlines()];ct={}
for w in ('W1','W2'):
 ct[w]={}
 for arm in ('NORMAL','CLAMP'):
  rr=[x for x in trace if x['context']['world']==w and x['context']['branch']==arm+'_EDITED' and x['context']['phase']=='C']
  ct[w][arm]={k:sum(x['layers'][k]['update_sq'] for x in rr) for k in ('H','T','O')}
 ct[w]['T_clamp_to_normal_ratio']=ct[w]['CLAMP']['T']/ct[w]['NORMAL']['T']
forks=U.load('FORK_AUDIT.json');rng_checks={w:{ph:len({r['dropout_hash'] for r in forks if r['world']==w and r['phase']==ph})==1 for ph in ('B','EDITED','NOEDIT')} for w in ('W1','W2')};assert all(v for d in rng_checks.values() for v in d.values())
U.save('SUPPLEMENTARY_UPDATE_ANALYSIS.json',dict(C_edited_squared_update_sums=ct,cross_arm_dropout_fork_hashes_match=rng_checks,posthoc_descriptive=True,causality_not_isolated=True))
report+='\nAfter hidden learning resumes, the edited-C T-layer sums of squared parameter updates are larger for CLAMP histories: '
report+=', '.join(f"{w}: {ct[w]['CLAMP']['T']:.4f} vs {ct[w]['NORMAL']['T']:.4f} ({ct[w]['T_clamp_to_normal_ratio']:.3f}×)" for w in ('W1','W2'))
report+='. This saved-trace association supports examining the resumption dynamics, but does not isolate moment age, parameter position or feature/readout co-adaptation. The dropout RNG hashes also match across B-history arms at every corresponding fork.\n'
report+='''
These are sums of squared parameter movements, not biological energy or net displacement. H/T movements are exactly zero in every CLAMP B update; all six B-boundary digests verify that weights, moments and steps equal the common A state. Discarded hidden teaching gradients are still computed and logged. Actual hidden writing resumes during C. Output O learns throughout, using the same local CE rule and full257 alphabet. No evaluator-fitted decoder is used.

Technical source/lesion qualification passed before main scores. Source NORMAL and explicit NORMAL updates match exactly; the clamp preserves decay/moments/steps while O changes; a checkpoint roundtrip preserves the clamp state; H/T updates resume; read-only scoring preserves state/RNG. NORMAL reproduces all16 corresponding A/B/C parameter/optimizer/RNG hashes and held scores. The two B forks start identical, and every C feedback pair starts identical. The protocol, sources, parent checkpoints, reducer and data manifests were locked before main training. No source, dose, gate, seed, edit mapping or optimizer policy was tuned after observing outcomes.

## Own output, data and evidence level

'''
gen=[g for r in R['checkpoints'] for g in r['generation'].values()];report+=f"Complete own continuations: **{sum(g['complete_exact'] for g in gen)}/{sum(g['records'] for g in gen)}** queries over all registered checkpoints. Three observed bytes are followed by at most32 model-generated bytes/EOS. Own EOS, truncation and exactness are saved separately; predictive loss is not a substitute for complete responses.\n\n"
report+='''Both arms observe the identical raw B stream and teacher information, in identical chronological batches and matched dropout streams at the fork. Sensory inputs encode observed prefixes only. H/T supervision is computed in both arms; the intervention discards its gradients before the optimizer. Correction is generated from the original train-only mappings; edited raw records are re-encoded, no keys/old-new identities or phase IDs enter neural input. The freeze is an experimenter-controlled intervention, not a deployed task-phase controller.

Retain the original exposure/recitation audit and count references. Exact held records are not duplicates, but overlapping prefixes and conditional-frequency learning remain countermodels. No additional counting/probe model is fitted here. Class masks are evaluator statistics only. Same raw257 head for payload/EOS, no forced EOS, arithmetic structure, attention, Transformer, ELM, BPTT or pretrained answerer. Original no-added-warmup is kept for causal comparability; it is not adopted as a preferred curriculum.

Evidence is E0-qualified access and a bounded predictive-learning causal-package diagnostic. No E2/E3/E4, never-taught arithmetic, biological mechanism, broad architecture success or adoption claim. This does not launch the official experiment or a new version. The eventual goal remains retained reusable learning and own-byte/EOS unseen-addition generalization per information and compute.

## Exposure and compute

'''
ledger=U.load('PROCESS_LEDGER.json');cpu=sum(x['CPU'] for x in ledger['completed']);peak=max(x.get('RSS_peak',0) for x in ledger['completed']);counts=S['counts'];source=U.ROOT/'minifly_v179_continuous_local_life_dev_20261102';diag=U.ROOT/'minifly_v179_lifetime_diagnostic_20261102';previous=[json.loads((p/'COUNTS.json').read_text()) for p in (source,diag)];prior=sum(x['records'] for x in previous);priort=sum(x['targets'] for x in previous)
report+=f"New main learning: **2,980 full-record presentations /187,606 target applications**. Technical tests: **8 /381**. New total: **{counts['records']:,} records /{counts['targets']:,} targets**, within3,100/200,000. Main optimizer updates: **{counts['optimizer_steps']}**. Both source A states inherit **4,086 A training presentations /246,108 targets**, learned before this diagnostic and charged as prior experience; no physical A retraining or free-pretraining efficiency claim. Shared descendants inherit that history; its logical cost is not erased by cloning.\n\n"
report+=f"Across the three V179 runs, physical recorded learning totals are **{prior+counts['records']:,} record presentations /{priort+counts['targets']:,} targets**. These include repetitions and multiple control lives, not unique samples. Current accounted CPU before documentation: **{cpu:.2f}s**; peak process RSS **{peak/1e6:.1f}MB**. Read-only panels, own generation, discarded hidden-gradient computation, source hashing and saved-only reduction all count toward CPU. Existing authenticated sensory assets and identical checkpoints are referenced rather than copied; new control states are retained. Serialization reuse occurs only after the corresponding normal learning was physically executed and hash-matched.\n\n"
report+='''The separate saved-only reducer checks scores/labels/denominators, fork equality, all-update coverage, exact frozen/restored writing, sources and state/panel hashes. Per-site metrics and checkpoints stay local and allow reconstruction; no external raw reconstruction is claimed. No model was retrained merely for reporting. Compact package excludes raw texts, encoded data, raw output panels, weights, local dependencies and per-update raw traces.
'''
(U.H/'REPORT.md').write_text(report)
recommend='''# V179 plasticity-location decision memo

No new experiment, automatic confirmation, version allocation or architecture adoption is authorized here. Preserve the source-locked original and both diagnostics.

Read the paired B effects and C continuation tradeoffs together. If hidden preservation improves B prediction but correction continuation loses B, this is evidence for a history-dependent plasticity-regulation problem, not a complete repair by freezing. A future proposal must distinguish actual weight change from optimizer-history/write-strength effects before calling its mechanism “representation protection.” If normal hidden learning wins or signs are mixed, do not suppress it based on the original overfitting observation alone.

The same own-output learner can fit taught data and revise predictions. Predictive generalization and stable continued adaptation remain the relevant bottlenecks. Complete byte/EOS behavior remains a separate unmet behavioral endpoint. Additional memory banks, a new sparse ring, a fitted external decoder, larger scale or more repetitions would not be justified by this diagnostic alone.

The next decision is a scoped review of the measured causal package and its tradeoff. Select at most one generic local-learning intervention tied to that measurement, preserve the four lifetime outcomes, and lock matched information/compute controls before training. Do not pick a preferred pass from these held curves or convert opened-text consistency into fresh confirmation. The eventual unseen-addition objective remains unchanged.
'''
(U.H/'DECISION_MEMO.md').write_text(recommend)
U.save('DELIVERY.json',dict(status='COMPLETE',identity='V179-PLASTICITY-1',interpretation=S['interpretation'],normal_reference_checkpoints=S['normal_reproduced_checkpoints'],counts=counts,accounted_CPU_before_report=cpu,peak_RSS=peak,official_run=False,new_core_adopted=False,new_version_allocated=False,causal_scope='hidden-plasticity package conditional on A state; weight/moment/write-strength not separated'))
U.finish_process('saved_report_packaging');files=[p for p in U.H.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json') and p.name not in ('PACKAGE.json','ACTIVE.json')];archive=U.H/'RESULT_BUNDLE.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(files):z.write(p,p.name)
U.save('PACKAGE.json',dict(status='COMPLETE',archive=archive.name,sha256=U.sha(archive),files={p.name:U.sha(p) for p in sorted(files)},excluded=['raw data','raw panels','checkpoint weights','raw per-update traces','local dependencies'],external_reconstruction=False));print('REPORT_PACKAGE_COMPLETE')
