"""Assemble existing analysis outputs only; no learner import, fitting, or updates."""
from pathlib import Path
import json,hashlib,datetime
H=Path(__file__).resolve().parent
read=lambda p:json.loads((H/p).read_text())
l=read('law/RESULTS.json');c=read('capacity/RESULTS.json');r=read('recoverability/RESULTS.json');v=read('v147/RESULTS.json')
fixed_error=max(x['max_equation_absolute_error'] for x in l['fixtures'] if '_FIXED_' in x['fit'])
b=c['prediction_results']['baseline'];e=c['prediction_results']['geometry_added']
pr=read('PREDICTIONS.json')
adj={'schema':'PREDICTION_ADJUDICATION','register_sha256':hashlib.sha256((H/'PREDICTIONS.json').read_bytes()).hexdigest(),'historical_outcomes_already_known':True,'independent_prospective_validation':False,'entries':[{'id':'L1','status':'VERIFIED_E0_ONLY','fixed_max_absolute_error':fixed_error,'all_identity_max_absolute_error':l['max_equation_absolute_error'],'unseen_addition_efficacy':False},{'id':'R1','status':'RECOMPUTED_KNOWN_HISTORICAL_CONTRAST','STD_DIRECT_final_W_minus_S_NEW_OLD_payload_pp':-53.90625},{'id':'C1','status':'NO_INCREMENTAL_PREDICTIVE_GAIN_IN_THIS_SMALL_RETROSPECTIVE_CHECK','baseline_RMSE':b['RMSE'],'geometry_RMSE':e['RMSE'],'improved_worlds':4,'worlds':8,'pre_fit_schema_refinement':'Preliminary OLD_end name refined to OLD_rest1 in capacity/ANALYSIS_SPEC.json before its first fit; this is the post-rest pre-NEW checkpoint, not an endpoint-derived predictor','broader_geometry_hypothesis_disproven':False},{'id':'F1','status':'UNTESTED_FUTURE_HYPOTHESIS_NOT_A_RUN_LOCK'},{'id':'F2','status':'UNTESTED_FUTURE_HYPOTHESIS_NOT_A_RUN_LOCK'}]}
(H/'PREDICTION_ADJUDICATION.json').write_text(json.dumps(adj,indent=2)+'\n')
report=f'''# MiniFly bounded predictive diagnosis — 25 October 2026

**Completed: exact special-case verification and three bounded artifact reanalyses.** The proposed unified theory remains unvalidated as a predictive account. No new architecture or scientific training trajectory was run, and no unseen-addition improvement is established.

The objective stays complete own-byte/EOS unseen-addition generalization per total experience and compute, with gradual warmup, local rules and no arithmetic-specific candidate structure. The historical 53-delivery catalogue is reused; this dossier is not V148.

## Main findings

| Question | Result | Permitted inference |
| --- | --- | --- |
| Does the FIXED single-update formula match actual code? | Maximum margin error **{fixed_error:.3g}**; all FIXED/LOCAL identities below {l['max_equation_absolute_error']:.3g} | Exact source-specific E0 law, not learned arithmetic |
| Does NEW writing harm protected OLD responses in Recoverability? | STD DIRECT final OLD known-length payload W−S_NEW **−53.91 pp**; RAW **−19.53 pp** | Net write-dependent accessibility loss in that protocol |
| Does pinned pre-value overlap improve later-retention prediction? | Whole-world RMSE **{100*b['RMSE']:.3f} → {100*e['RMSE']:.3f} pp**; MAE **{100*b['MAE']:.3f} → {100*e['MAE']:.3f} pp** | No incremental gain in the small declared check; do not rescue it by searching another feature |
| What fails in V147 taught generation? | High-rate LOCAL correct first payload **44.69% / 45.41%**; exact EOS conditional on correct first byte **99.46% / 98.94%** | Most errors occur before stopping; held assessment remains unopened |

## 1. Source-specific mathematics, verified rather than assumed

For normalized binary KC code phi and actual V147 naming weight theta, m=2 theta dot phi. A single target update gives

`Delta m_j(q) = alpha s(n) [t_j−tanh(m_j(s)/2)] K(q,s)`

when codes stay fixed. We tested 8 FIXED and 4 LOCAL isolated target events on disposable clones of six preserved calibration checkpoints. Each starts from the same saved state; there is no continuation/replay of a training history. Historical source, checkpoint and original in-memory model/cache hashes stayed unchanged.

LOCAL requires the additional exact term `2 theta'_j dot (phi'_q−phi_q)`. Some of the four LOCAL fixtures changed a sampled query's TopK code, producing terms up to 0.352 versus direct naming terms up to 0.026. This illustrates how a static kernel can omit an update component. It does not estimate how often drift occurs, whether it helps on average, or why acquisition failed. Because the post-update code enters that term, it is a decomposition, not an independently successful drift prediction. All sampled states were beyond warmup; ramp correctness was derived from code, not requalified experimentally.

See [equations and limits](EQUATIONS.md), [law findings](law/FINDINGS.md) and [fixture ledger](law/FIXTURE_LEDGER.json).

## 2. Recoverability: signed loss and protocol boundaries

The independent scalar reducer crosschecked 448 compact payload rows and 16 saved phase summaries against detailed data, retaining four grouped worlds, correlated value lives and heads. In STD DIRECT, OLD known-length payload was **54.69%** after OLD teaching, **0%** final with NEW writes, and **53.91%** final with matched NEW writes disabled. Complete OLD output was already zero, so a complete-score-only analysis hides the storage/accessibility deterioration.

On the same native responses, the STD MEAN joint head at NEW final increased complete score by **3.91 pp** while reducing known-length payload exact by **41.15 pp** versus DIRECT. This is another reason to keep readout success and memory content separate. These are different readout procedures; their marginal scores cannot be multiplied to infer an own-generation conditional EOS probability.

The compact files do not supply per-query native margins or each suppressed-write branch. W−S_NEW is a net intervention effect, not a reconstructed event-wise operator or proof of physical erasure. With only four worlds, common time/write dose and floor-valued final complete scores, a richer predictive fit is not identifiable or useful here. No fit was attempted.

See [Recoverability findings](recoverability/FINDINGS.md).

## 3. Capacity: a plausible geometric predictor failed the declared check

The pre-value weighted OLD/NEW support overlap fell from **0.07918** to **0.02285** under expansion. The historical E1 gains remain: final OLD payload **1.56% → 35.55%**, NEW payload **52.73% → 100%**. However, expansion also changes connectivity, Q normalization and self gain, and costs **3.154×** historical full-life/readout CPU. This is not a pure neuron-count or compute-efficiency result.

We pinned one additional feature, weighted OLD/NEW overlap, without searching alternatives. Baseline predictors are the engineering-arm indicator and OLD payload immediately before NEW, after the first rest. The enhanced model adds only that overlap. Eight folds each withhold an entire paired world; both sizes, all heads and lives remain in the same fold. Scaling uses training folds only. These tiny statistical OLS fits analyze saved scalars; they are not candidate neural or ELM learners.

The enhanced model worsened MAE/RMSE and improved squared error in only **4/8** worlds. This is exploratory retrospective evidence at small n, with known aggregate outcomes. It weakens the claim that this one scalar provides extra prediction beyond initial acquisition and engineering condition. It does not prove that geometry is irrelevant.

A useful additional caution: expansion had higher final OLD recovery but a larger absolute passive-minus-write loss (**64.45 pp**, versus **53.91 pp**), because it began with stronger acquisition. Thus neither lower cross gain nor absolute accuracy loss alone orders the systems' retention quality.

DIRECT payload uses evaluator-known length on fixed key/slot responses. STOP complete decoding uses the same saved slot tensor after EOS calibration; emitted prefixes do not change the native query. This is not V147's own-prefix autoregressive protocol. No cross-protocol ratio was used as `P(EOS|payload)`.

See the [pre-fit specification](capacity/ANALYSIS_SPEC.json), [findings](capacity/FINDINGS.md) and every fold in [results](capacity/RESULTS.json).

## 4. V147: output decomposition, no holdback leakage

The reducer independently checked every saved taught checkpoint prediction against its taught orbit and expected byte. On the same orbit-weighted own-generation protocol, the complete-score identity holds. In this one-byte assay, conditional exact stopping given a correct first byte is already high in the high-rate LOCAL fits. An EOS-only change cannot repair the low payload acquisition.

LOCAL has lower mean teacher bit BCE yet loses **22.46 / 32.61 pp** of taught complete accuracy against FIXED in the two high-rate worlds. FIXED has stable codes and no exact conflicting-target groups, yet misses readiness. These observations reject mean bit loss and code uniqueness as sufficient readiness proxies; they do not identify a universal cause. No held relation was scored and no H world generated.

See [V147 reanalysis](v147/FINDINGS.md).

## What is now tested, and what is still unknown

- Verified: an exact one-update local naming law, E0, with explicit architecture and fixed-context limits.
- Recomputed: scoped native write-interference and decoder contrasts, E1, with signed rather than absolute effects.
- Tested unsuccessfully: one pinned geometric scalar's incremental retrospective predictive value on eight paired historical worlds.
- Not shown: native event-wise causal matrices, a universal persistence law, new generalization, ordinary-addition improvement or independent prospective predictive validation.

Final-query influence is already an outcome. A model that uses it or final held labels to explain those same outcomes has not forecast them. The dossier therefore avoids a high-dimensional pooled fit across native, surrogate and engineered lineages. It retains simple current-score baselines and reports unknown quantities rather than generating missing trajectories.

## Predictions and next decision

[PREDICTIONS.json](PREDICTIONS.json) preserves the initial register; [PREDICTION_ADJUDICATION.json](PREDICTION_ADJUDICATION.json) records outcomes and the pre-fit OLD_end→OLD_rest1 naming refinement. Historical expectations are labeled known, and the failed feature is not replaced. Raw-trace and bounded relational-update predictions remain tentative and untested. They are not a fully specified experiment lock or launch authorization.

The theory currently earns the status **useful measurement framework with one exact special-case law; no demonstrated independent forecasting advantage**. The next separately authorized comparison should measure early target-oriented effects before later outcomes, retain a simple baseline and keep one learner/output path fixed. Raw trace × relational teaching remains a candidate discriminating experiment. Do not add a new architecture, task or auxiliary decoder merely to make the theory appear successful.

## Cost, artifact and scope ledger

- **New scientific training histories: 0. External record applications: 0. Replayed histories: 0.**
- **Disposable E0 target-update events: 12**, 8 FIXED + 4 LOCAL, all counted and discarded; no historical state changed.
- Capacity used 48 tiny statistical OLS fits across three deterministic report-analysis passes; a prior extraction error occurred before fitting and is preserved. These are not 48 candidate learners or new science lives.
- Measured analysis CPU is recorded per component. Tool/authoring/import/review overhead and the early failed extraction are not all measured; no all-in compute claim is made.
- Git preserves 783 Recoverability and 812 Capacity paths omitted by sparse checkout. Compact blobs were read in memory. **Raw arrays, historical checkpoints and ZIP parts restored: 0.** Existing local V147 calibration checkpoints were read for fixtures only.
- Frozen code, gates, warmup, scientific history, active experiment pointer and V147 holdbacks remain unchanged. Nothing was pushed to GitHub.

The [availability inventory](ARTIFACT_AVAILABILITY.json), per-analysis source hashes, [identifiability report](IDENTIFIABILITY.md), scripts and compact outputs form the dossier. It does not require another global evidence-registry project.
'''
(H/'REPORT.md').write_text(report)
status={'schema':'BOUNDED_PREDICTIVE_DIAGNOSIS','date':'2026-10-25','status':'COMPLETED_FRAMEWORK_NOT_INDEPENDENTLY_VALIDATED','source_commit':c['source_commit'],'analysis_only':True,'new_architecture':False,'scientific_training_histories':0,'external_record_applications':0,'trajectory_replays':0,'fixture_target_events':12,'statistical_OLS_fits':c['total_statistical_OLS_fits_across_script_runs'],'raw_artifact_restoration_bytes':0,'held_scores_opened':False,'assessment_worlds_generated':False,'new_core_adopted':False,'github_pushed':False,'goal':'Complete own-byte/EOS unseen-addition generalization per total sample and compute; gradual warmup, local rules, no arithmetic-specific candidate structure','report':'REPORT.md','prospective_validation_completed':False}
(H/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
print('Built bounded diagnosis report, prediction adjudication and status.')
