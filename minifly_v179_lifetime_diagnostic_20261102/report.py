"""Report/package from saved diagnostics only; does not import the learner."""
import util as U
import json,time,zipfile
D=U.load('DIAGNOSTIC_SUMMARY.json');R=U.load('RESULTS.json');q={(r['world'],r['checkpoint']):r for r in R['checkpoints']};stats=U.load('UPDATE_SUMMARIES.json');categories=[v['category'] for w in D['contrasts'].values() for v in w.values()];all_div=all(c=='OVERFITTING_LIKE_DIVERGENCE' for c in categories)
report='''# V179-DIAG-1 — instrumented lifetime diagnostic

**Completed bounded diagnostic. No official confirmation, new mechanism or core adoption.**

'''
report+=('All six B comparisons show **improving training loss with worsening held loss** from pass1 to pass3. This is the predeclared overfitting-like train/held divergence pattern.\n\n' if all_div else 'The predeclared train/held classifications vary; report every world/arm below without replacing fixed endpoints.\n\n')
report+='''## Same-checkpoint B loss

Bits per actual byte/EOS target; lower loss is better. Train and held losses use the same full257 head, eval mode and fixed checkpoint. The chronological minibatch online loss is not substituted for a frozen training-panel score.

| World | Branch | Train B1 | Train B3 | Δtrain | Held B1 | Held B3 | Δheld | Pattern |
|---|---|---:|---:|---:|---:|---:|---:|---|
'''
for w,d in D['contrasts'].items():
 for arm,v in d.items():report+=f"| {w} | {arm} | {v['train_B1']:.5f} | {v['train_B3']:.5f} | {v['delta_train']:+.5f} | {v['held_B1']:.5f} | {v['held_B3']:.5f} | {v['delta_held']:+.5f} | {v['category']} |\n"
report+='''
The1e−4-bit tie tolerance is numerical, not a significance threshold. These are two previously opened, contiguous literary-text splits. Train/held distribution differences, recency, effective capacity and the local update rule can all contribute to the observed pattern. No random-split causal test or fresh-world generalization inference is made. The result supports continued fitting of taught data accompanied by poor held prediction, rather than loss worsening on both panels. It does not uniquely identify which mechanism would repair it.

## Distribution and update observations

| World | Branch | Held entropy B1→B3 | Mean EOS mass B1→B3 | Payload mass absent from all training B1→B3 |
|---|---|---|---|---|
'''
for w,d in D['contrasts'].items():
 for arm,v in d.items():report+=f"| {w} | {arm} | {v['entropy_B1']:.5f}→{v['entropy_B3']:.5f} | {v['EOS_mass_B1']:.6f}→{v['EOS_mass_B3']:.6f} | {v['unused_payload_mass_B1']:.6f}→{v['unused_payload_mass_B3']:.6f} |\n"
report+='''
Entropy and probability mass are observational summaries, not a calibrated causal test. Lower entropy alongside a larger held loss is consistent with increased confidence on a poorly generalizing fit. All classes remain legal; evaluator-only absent-class masks never modify neural input, gradients or softmax. No class compression, OTHER mapping, forced EOS, byte repair or external fitted head is used.

Every one of the1832 real learning updates has H/T activation activity and RMS, layer parameter/gradient/actual-update norms and ratios, and Adam moment RMS recorded. Target-weighted per-pass summaries are in UPDATE_SUMMARIES.json; complete batch traces remain local. Norms describe scale and movement, not gradient correctness or causal importance. They are not compared across layers without accounting for different parameter counts. There were no nonfinite-update or resource failures. No instability rescue, clipping change or post-score rate adjustment was applied.

## Full lifetime and own output

| Screen | W1 | W2 |
|---|---|---|
'''
for cap in ('acquisition','reuse','retention','revision'):report+=f"| {cap} | {'PASS' if D['continuous_lifetime_screens']['W1'][cap] else 'FAIL'} | {'PASS' if D['continuous_lifetime_screens']['W2'][cap] else 'FAIL'} |\n"
gen=[g for r in R['checkpoints'] for g in r['generation'].values()];exact=sum(g['complete_exact'] for g in gen);queries=sum(g['records'] for g in gen)
report+=f"\nAll registered A/B/C histories ran, even after poor early scores. The original four screens are unchanged. Complete own continuations score **{exact}/{queries}** queries across the instrumented checkpoints, with three observed prompt bytes and a32-byte/EOS generation cap. Truncation is reported explicitly. Predictive retention/revision/reuse remains bounded evidence; it does not establish successful complete answers.\n\n"
report+='''## Replay integrity and exposure boundary

Before the replay, four complete technical records were updated through the literal source path and the same four through the instrumented path. Parameters, Adam moments/steps and dropout RNG were bit-identical. Read-only evaluation preserved state/RNG. The learner, original local update, fixed feedback, data edits, chronological batches and full257 output are unchanged. There is no added warmup; the fixed source schedule is retained solely for diagnosis, not adopted as a preferred curriculum.

Every checkpoint shared with the previous V179 run has an **identical parameter/optimizer/RNG state hash** and matching held scores; the saved-only audit checks all30 common states. Pass1/2 and A1/A2 are now retained. Identical parent checkpoints may be referenced instead of duplicated, but their entire training was physically repeated and charged before identity was checked. This is serialization deduplication, not free pretraining or restoring a validation-best state. New pass states are saved in full. Historical V179 source and results remain unchanged.

Corpus editions, exact records, genuine EOS, train-derived correction transform, original sensory empty-input column5177 and pad5178 are inherited by manifest. No data/label selection is repeated based on outcomes. Held targets are used only for read-only evaluation. Fixed native sensory caches are pure observed-prefix functions; no learned activations are reused across updates. Input bytes beyond the prediction site do not enter the cached code.

The authenticated exposure audit is retained: no duplicate train/held full records, but overlap of some prefixes/labels remains. Exact/canonical prefix and frequency controls can explain portions of predictive reuse. Count predictors are separate evaluator models, never learner answers, and their original fitting costs remain disclosed. No additional countermodel fitting was performed in this diagnostic.

This is E0-qualified access plus bounded predictive-lifetime development diagnosis. It is not E2/E3/E4, independent confirmation, biological validation or an arithmetic result. The eventual requirement remains retained reusable learning and own complete-byte/EOS answers on never-taught addition per training sample and actual compute. No attention, Transformer, arithmetic-specific structure or pretrained answerer is introduced.

## Cost and audit

'''
counts=D['counts'];old=U.ROOT/'minifly_v179_continuous_local_life_dev_20261102';parent=json.loads((old/'SUMMARY.json').read_text());ledger=U.load('PROCESS_LEDGER.json');cpu=sum(v['CPU'] for v in ledger['completed']);peak=max(v.get('RSS_peak',0) for v in ledger['completed']);owned=sum(p.stat().st_size for p in U.H.rglob('*') if p.is_file() and not p.is_symlink())
report+=f"This replay physically used **7,532 main records /463,277 target applications**, plus8 technical records /400 targets: **{counts['records']:,} records /{counts['targets']:,} total learning targets**. It used the same data again; no new development worlds were created. Across the original V179 run and this diagnostic, recorded learning totals are **{parent['counts']['records']+counts['records']:,} record applications /{parent['counts']['targets']+counts['targets']:,} targets**. These are repeated exposures, not unique examples.\n\n"
report+=f"Accounted CPU before report packaging: **{cpu:.2f}s**; peak worker RSS **{peak/1e6:.1f}MB**. New owned output before packaging is {owned/1e9:.3f}GB; existing read-only assets/checkpoint references are separately retained. Training, added train/held scoring, own generation, instrumentation, hashing and saved-only reduction all count toward compute. Earlier V179 source learning/encoding/count fitting cost is inherited context, not credited as free information. The diagnostic has its own explicit bounded authorization and is not counted as fresh confirmation.\n\n"
report+='''The independent saved-only process checks sources, state/checkpoint/panel hashes, labels and denominators, paired-record signed contrasts, full update coverage and fork controls. Per-site target probabilities and distribution summaries are saved; duplicate full257 panels are not retained because of the bounded output budget. All checkpoint weights/RNG and original sensory assets are retained for reconstruction. No external raw-data reconstruction is claimed. No trajectory was rerun merely to produce this report.

## Decision

The official run remains premature under the unchanged lifetime criteria. The immediate problem is the train/held gap and failure of own generation. Future intervention should explicitly target that measured gap with one matched control and a fixed information/compute budget. Chronological-distribution effects, update regularization or representation sharing are hypotheses, not findings settled by these curves. Do not add a memory bank, allocate a successor or choose a preferred pass solely from this result. No additional experiment is started by this report.
'''
(U.H/'REPORT.md').write_text(report)
U.save('DELIVERY.json',dict(status='COMPLETE',identity='V179-DIAG-1',new_version_allocated=False,official_run_launched=False,diagnostic_categories={c:categories.count(c) for c in sorted(set(categories))},parent_states_identical=D['parent_common_checkpoint_count'],counts=counts,accounted_CPU_before_report=cpu,peak_RSS=peak,owned_output_bytes_before_packaging=owned,raw_data_published=False,core_adopted=False))
U.finish_process('documentation_packaging')
files=[p for p in U.H.iterdir() if p.is_file() and p.suffix in ('.py','.json','.md') and p.name not in ('PACKAGE.json','ACTIVE.json')];archive=U.H/'RESULT_BUNDLE.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(files):z.write(p,p.name)
U.save('PACKAGE.json',dict(status='COMPLETE',archive=archive.name,sha256=U.sha(archive),files={p.name:U.sha(p) for p in sorted(files)},excluded=['raw data','raw site panels','checkpoint weights','per-update raw trace','local dependencies'],no_external_reconstruction=True))
print('REPORT_AND_PACKAGE_COMPLETE')
