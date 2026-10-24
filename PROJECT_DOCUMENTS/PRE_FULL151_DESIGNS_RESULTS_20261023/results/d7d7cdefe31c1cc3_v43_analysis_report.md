# V43: completed experiment and analysis

**All 64 scientific trajectories and all 48 diagnostic folds are complete. The planned local-specialist qualification is not established because the underlying memory failed its acquisition-validity gate.** There is nevertheless a useful, consistent descriptive result: blending fast and slow predictions reduces loss while retaining most of the slow memory's delayed recall. The E2-seeded local rule adds no observed benefit over the unseeded local rule on these endpoints.

## What was implemented and run

The recovered V43 design was refined into a frozen protocol before scientific execution. Sixteen fresh seeds, 43101–43116, each received total identity loads 64, 128, 256 and 512. Eight online read policies shared **one physical learner per seed/load**. Offline shuffling and oracles added comparisons without new learning histories. Capacities remained 512 traces and 384 consolidated slots; each acquisition rendering retained the inherited eight presentations.

The design kept learning, admission, promotion, revision and eviction unchanged. Only read authority changed. Mature exact anchors retained slow authority; probationary exact traces became silent in the proposed policies; supported similarity reads received fixed, global or object-local weights. Evidence used frozen pre-outcome predictions, with explicit generation creation, eviction and revision-reset rules. Read-only probes never updated either learner or evidence. The protocol also fixed the shuffle strata, singleton treatment, statistical unit and gate denominators.

The run processed **131,520 complete 14-byte records: 1,841,280 physical byte presentations, including 131,520 designated target presentations**. The sum of distinct complete records within the 64 histories was 32,640; this is not a count of globally distinct byte strings. Policy scoring does not multiply sample size. Four separate engineering replays compared 76,720 bytes and 1,920 terminal queries against the inherited implementation. Diagnostic fitting added zero learner updates.

The primary deliverable is `elm_v43_fast_slow_authority_complete.py`, a self-contained run/resume/report entrypoint. It embeds the exact scientific core plus the documented diagnostic reader correction. Physical checkpoints are atomic and checksummed, default to every eight records, retain a previous generation, and reject changed identities or corruption. Diagnostic folds are independently resumable. The completed rerun reused **64/64 scientific cells and 48/48 diagnostic folds, with zero new learning and zero refits**.

## Validity comes before qualification

Q0 required native SLOTS to answer at least 90% of **all intended learned aliases** at every acquisition checkpoint. Correct surviving slots alone could not satisfy the denominator, and the fast fallback could not certify slow-memory acquisition.

| Identity load | Cells passing every acquisition checkpoint | Final new-cohort native accuracy | Final new-cohort mature-slot coverage | Old held-out recall, SILENT-HARD |
|---:|---:|---:|---:|---:|
| 64 | 16/16 | 100.00% | 96.88% | 99.22% |
| 128 | 15/16 | 94.53% | 17.19% | 99.61% |
| 256 | 1/16 | 66.89% | 4.59% | 99.02% |
| 512 | 0/16 | 44.14% | 0.44% | 74.61% |

Each percentage is an equal-seed mean over all 16 cells at that load. Mature coverage is the proportion of the intended new-cohort population represented by a mature exact slot; it is a separate descriptive measure, not a substituted gate.

The one invalid load-128 cell was seed 43116: final native acquisition was 57/64 = 89.06%. Load 256 presents a broader problem: 15/16 cells failed, and mean current-cohort native accuracy had fallen to 64.79% after C and 66.89% after D. All load-512 cells failed. Every cell at loads 128, 256 and 512 ended with all 384 slots occupied. Mean cumulative slot-capacity evictions were 258.31, 978.19 and 802.44 respectively.

The contrast between strong old recall and weak acquisition of the newest cohort matters for persistent learning. This architecture can retain old answers while failing to consolidate enough new knowledge under the fixed budget. Successful old-memory probes alone therefore do not establish continued learning. The observed capacity pressure and low new-cohort mature coverage identify where to investigate; these summaries do not isolate a unique causal bottleneck among admission, matching and eviction.

**Q0 failed; Q1–Q10 remain not evaluable under the locked qualification rules.** The generated report's generic decision string is `incomplete_or_not_evaluable`; its execution status is separately `complete`. No trajectories are missing. The original decision function returns that generic string when downstream gates are unavailable.

There was ample numerical arbitration headroom: primary-load minimum G counts were 1,019 and 1,588 per seed, and mean hard-minus-event-oracle headroom was 1.82860 and 3.45261 bits/G at loads 128 and 256. Their gate status is unavailable because the required physical-validity condition failed, rather than because observations or headroom were absent.

## What the complete fixed roster actually shows

The following are **descriptive means over all 16 seeds per load**, including failed-acquisition cells. They are not estimates from a selected valid subset, and they do not rescue qualification. Lower bits/byte is better; higher recall is better.

| Read policy | Load128 bits/byte | Load128 old held-out % | Load256 bits/byte | Load256 old held-out % |
|---|---:|---:|---:|---:|
| SUPER | 2.3365 | 68.7500 | 2.5722 | 28.3203 |
| READ-HYBRID | 2.4404 | 99.6094 | 2.7258 | 99.0234 |
| SILENT-HARD | 2.3829 | 99.6094 | 2.6958 | 99.0234 |
| ANCHOR-ONLY | 2.3246 | 68.7500 | 2.5589 | 28.3203 |
| FIXED-50 | 2.2909 | 99.6094 | 2.5420 | 99.0234 |
| GLOBAL-BAYES | 2.3247 | 68.7500 | 2.5589 | 28.3203 |
| LOCAL-BAYES-0 | 2.2910 | 99.2188 | 2.5312 | 98.6328 |
| LOCAL-BAYES-E2 | 2.3012 | 99.2188 | 2.5392 | 98.6328 |

Three findings stand out.

1. **A simple blend already combines useful strengths.** At load 256, FIXED-50 reduced whole-stream loss from SUPER's 2.57220 to 2.54196 bits/byte while raising old held-out recall from 28.32% to 99.02%. Its delayed held-out accuracy exactly matched SILENT-HARD in every one of the 64 cells. These are paired read comparisons on identical learned states, so changes in consolidation cannot explain their differences.
2. **Object-local evidence is useful descriptively, but E2 seeding is not earned here.** LOCAL-BAYES-0 had lower whole-stream loss than LOCAL-BAYES-E2 in every cell at all four loads, and their delayed held-out accuracies were identical cell by cell. E2 seeding increased mean loss by 0.00417, 0.01029, 0.00799 and 0.00290 bits/byte at loads 64, 128, 256 and 512. At load 256, unseeded local evidence improved on FIXED-50 by 0.01074 bits/byte, at a 0.39 percentage-point recall cost. At load 128, their prediction losses were almost tied and FIXED-50 retained slightly better recall. These are 16 seeds observed at four loads, not 64 independent replications.
3. **One global evidence total does not preserve the useful memory advantage.** GLOBAL-BAYES's old held-out recall matched SUPER at every displayed load, while the fixed and local mixtures retained substantially more. That makes GLOBAL-BAYES an unsuitable replacement on these measured endpoints. It does not prove that every possible global controller must fail.

LOCAL-BAYES-E2 itself improved over hard similarity reads and separated from the retrospective shuffle:

| Load | LOCAL-E2 minus SILENT-HARD, whole bits/byte | LOCAL-E2 minus SHUFFLED, permutable bits/G | Mean seed-level oracle capture |
|---:|---:|---:|---:|
| 64 | -0.04565 | -0.20626 | 65.48% |
| 128 | -0.08163 | -0.37170 | 61.94% |
| 256 | -0.15661 | -0.79655 | 80.17% |
| 512 | -0.10703 | -0.69907 | 82.58% |

At load 256, every seed had at least 99.937% permutable coverage. The shuffle preserved the locked within-stratum weight multiset and total weight exactly; the independent audit verified its assignment and losses. These descriptive differences support investigating how evidence is assigned to slots/events. Q6 and Q8 still cannot be declared passed under the frozen primary-load validity requirement.

The capture denominator is hard-similarity loss minus event-oracle loss on the same G events. A convex mixture cannot beat this eventwise oracle; capture is at most one with positive headroom, up to rounding. The pre-science `PROTOCOL_ERRATUM.md` corrects an erroneous interpretation sentence while leaving the formulas and thresholds unchanged.

The full eight-policy, four-load summaries are in `ALL_CELLS_SUMMARY.csv/.json` and `V43_all_cells.png`. The original `LOAD_CURVES.csv` and `V43_results.png` summarize only observed-valid cells. At loads 128/256 those subsets contain 15/1 seeds and must not be interpreted as complete confirmatory panels.

## What the richer diagnostics establish—and what they do not

Three fixed models completed leave-one-whole-seed-out validation across all four loads: 48 folds on **85,769 online G events**, without event subsampling. Training-fold preprocessing excluded the held-out seed, including all its loads. Features were available before the outcome; seed, load, identity and evaluator target labels were excluded. The models never changed read policies, learning or qualification.

| Diagnostic | ROC AUC | Binary log loss, nats | Exploratory mixture bits/G | Pooled oracle capture |
|---|---:|---:|---:|---:|
| GRADIENT-BOOSTING | 0.9770 | 0.1868 | 2.8263 | 92.64% |
| LOGISTIC | 0.9666 | 0.2255 | 2.8445 | 92.05% |
| SPLINE-LOGISTIC | 0.9728 | 0.2038 | 2.8317 | 92.47% |

These are event-pooled descriptive metrics, with seed-blocked training and separate seed/load rows supplied in the raw reports. They do not use events as independent statistical replications. They also have a different aggregation population from the per-load, mean-seed capture table above.

Slow predictions won the binary per-event comparison on 55.46% of G events, yet their pooled loss was 5.67510 bits/G versus 3.64936 for fast predictions. The frequency of wins therefore hides the size of costly misses. The high held-out-seed AUCs show that recorded pre-outcome information helps distinguish the two experts on this online population. A predicted probability of winning the binary comparison is not automatically a calibrated Bayes mixture weight.

Fifteen of the sixteen **SPLINE-LOGISTIC** folds reached the fixed 300-iteration limit and emitted convergence warnings. Their scores are reported as produced and should be treated cautiously. Logistic and gradient boosting had no recorded convergence warnings. No iteration cap, model or hyperparameter was changed after seeing these results.

The exposure audit places an important limit on interpretation:

| Load | Online G events | G events at identity targets | Old held-out target probes taking G |
|---:|---:|---:|---:|
| 64 | 7,086 | 0 | 124/128 |
| 128 | 17,619 | 6 | 254/256 |
| 256 | 27,705 | 63 | 506/512 |
| 512 | 33,359 | 285 | 736/1,024 |

Only 354/85,769 online G events—0.41%—were designated identity targets. Most evidence and diagnostic training concerned ordinary bytes, while delayed recall queried identity targets. This is an observed population difference. Strong online-G diagnostic performance does **not** validate a learned arbiter on the held-out recall population. Target tags were used only for this evaluator-side description, never as policy features. G-by-stage counts cannot be recovered from the frozen authority table because it does not contain stage labels.

The planned paired t intervals and Holm comparisons remain in the raw reports with their actual valid-seed denominators. Qualification requires the complete valid primary panels. Seed-blocked load F-tests, interactions, bootstrap scaling comparisons and L50 could not be evaluated as planned: only load 64 had a complete 16-seed valid panel. Fitting a selected survivor subset or relaxing Q0 after seeing the data would answer a different question. Diagnostic classification supplies additional evidence without overriding that limitation.

## My assessment and next step

V43 advances the project by separating a read-authority improvement from a continued-learning failure. The complete fixed-roster measurements show that fast/slow blending can improve prediction and preserve old recall on the same learned state. They also show that this state increasingly fails to acquire the newest cohort at the fixed capacity. Neither result alone is a persistent learner.

I would carry **FIXED-50 as the simple reference and LOCAL-BAYES-0 as the local-evidence candidate**, keeping LOCAL-E2 as a recorded comparison rather than the preferred mechanism. The observed E2 initialization consistently costs prediction loss without improving the tested delayed-recall endpoint. That is a reason to simplify the next hypothesis, not to claim a completed confirmatory selection.

The next bounded experiment should first establish acquisition and consolidation validity under the intended capacity, with a small screen and explicit all-identity denominators. It should then test whether a causal arbiter handles both ordinary prediction and genuine target-generalization reads under the same budget. The current diagnostic classifier is a hypothesis generator for such a rule, not an already validated replacement. No V44 run or post-result retuning was performed here.

This remains a synthetic mechanism experiment toward the project's non-Transformer persistent-learning goal. It does not establish natural-language learning, biological equivalence, or a reduced neuron count. The inherited fast layer still uses 4,096 units and an 8 MiB store; making trace reads silent does not remove their storage. The slow semantic peak stayed below 2.084 MiB, while the maximum recorded scientific worker RSS was about 252.6 MiB, including result serialization; these are different accounting measures.

## Reproducibility, corrections and deliverables

The frozen scientific core has SHA256 `fb8ebb5355ea03e79d6049e5498db60772b18aa2bd51a54beed0689266a77bd8`. The completion entrypoint is `456daa818b500a6d9ad128e52b43451626a691f674c7141d018cc605071d4b2d`; its corrected diagnostic module is `98f88ef5b5f17efe0982f0876faae599a76d4e85fff22fde84e507c9bfd133f9`. Scientific PLAN and checkpoint identities retain the original core hash; `ANALYSIS_IMPLEMENTATION.json` separately records the actual postprocessor.

The original writer encodes an undefined integer revision age as `-1`. Its diagnostic reader incorrectly rejected that missing marker. A separate, verified correction restores it to missing **only for never-revised slots**, preserving the specified training-fold imputation. Contradictory markers still fail. No physical learning, scoring formula, feature definition, model setting or gate changed. The original process returned status 1 after all scientific cells and the core report were written; the completion entrypoint reused those results and completed diagnostics. `DIAGNOSTIC_ENCODING_FIX.md/.json` records the exact diff and tests.

The independent final audit passed across all 64 cells and 1,841,280 saved byte events. It checked identities, engineering receipts, result hashes, routing and firewall receipts, ledger reconstruction, probability-mixture losses, shuffle controls and oracles. Its limits are explicit: it does not independently reconstruct unlogged route-F pre-promotion vectors or full top-1 predictions from scalar true-byte probabilities. The outer scientific console log retained only an initial fragment; structured per-cell completion files and invocation receipts provide the full completion evidence.

Execution used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8 and scikit-learn 1.8.0. Diagnostic caches bind the input and implementation identity, including Python/scikit-learn versions; reuse in an incompatible environment is deliberately rejected. Fresh runs use the same single-file command. The original design, adopted protocol, correction documents, original and completion sources, verification code, raw results, plots and this analysis are included in the handoff. `ANALYSIS_DIGEST.json`, reproduced by `v43_build/analysis_digest.py`, supplies the descriptive arithmetic behind this narrative.

The antecedent review distinguishes Codex's V41/V42 measurements from their Claude-derived source dependencies. No separately identified Claude V43 results archive was available, so this report does not label a source component as an independent Claude V43 replication.
