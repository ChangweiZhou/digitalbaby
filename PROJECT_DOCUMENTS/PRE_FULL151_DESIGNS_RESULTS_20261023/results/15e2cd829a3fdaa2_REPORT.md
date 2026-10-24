# V66 / Iteration 11 — Localizing persistent-memory interference

**13 September 2026. Completed diagnostic iteration. No learning-model promotion.**

V66 identifies a reproducible failure mechanism **inside the current MiniFly model**: later writes reinforce or damage an old association according to their direction on shared memory coordinates. Aggregate write magnitude alone predicts little. This finding holds in the full 5,177-KC parent, so KC deletion cannot explain it. It does not establish that the model has recovered the corresponding biological mechanism or solved persistent learning.

The strongest result is a prospective intervention on independent inputs. Keeping the later cue pair fixed, requiring comparable total, compartment and teaching magnitudes, and swapping which cue receives the consequence changes the next-day effect on an old memory from **−0.443 Hz to +0.501 Hz**. The paired difference is **+0.944 Hz**, with a **95% stream-bootstrap interval of +0.878 to +1.013 Hz**. All inference treats an independently generated stream/checkpoint as the sampling unit.

The accepted learner remains iteration5/bi with the frozen iteration8 RBF64 observer. There is no new learning rule, reader fit, neuron reduction or physiology fit in this iteration. The historical naming is **V65 = Iteration 10; V66 = Iteration 11**.

## What was refined before running

The supplied design was treated as a proposal. The interrupted task had already produced a useful retrospective feasibility audit on old V65 inputs. Those outputs were preserved and used only for design refinement; they are not counted as new confirmation.

The frozen kernel makes an average across three memory compartments misleading at the primary one-day horizon. After 86,400 seconds it retains 99.14% of slow state, about 1.74e−11 of alpha fast state and 9.62e−21 of gamma fast state. V66 therefore uses **slow-footprint direction** as its primary alignment variable, while retaining the other compartments in diagnostic tables.

Two apparent forms of strong evidence were also separated from independent prediction. First, a clipping-aware output reconstruction using the realized write vector mostly reconstructs the model's own equations. Second, with matched adaptation, the actual-memory and fixed-reader *changes* are algebraically equal because their baseline terms cancel. Neither is counted as an additional successful mechanism test. Endpoint support interventions test expression locality; they do not exclude globally pooled feedback from generating a write.

The protocol fixed 16 ordinary streams, 16 independent intervention checkpoints, 256 candidate pairs per intervention checkpoint, 24 selected pairs per focal, all calipers and all analysis blocks before new outcomes were measured. No candidate-pool expansion or caliper relaxation was needed for the primary role intervention. The magnitude control has incomplete coverage and is reported accordingly. See [PROTOCOL.md](PROTOCOL.md) and [the execution lock](checks/lock.json).

## Model, inputs and estimand

The parent contains 5,177 effective KCs, 302 selected PNs represented through 88 input channels, the unchanged top-5%-per-side encoder, anatomical/effective Q/T/feedback routes, the original plasticity kernel and the fixed reader. Mutable state remains **165,688 bytes**. Its state variables are effective plastic traces and adaptation, not individual physical synapses or a whole spiking fly brain.

All **11,392 new cue vectors** are globally unique eight-of-88 patterns and disjoint from the **9,848 unique saved prior cues**. The exclusion chain includes development, evaluation, capacity, regression and later iteration input banks. There are no natural odor population recordings in this new campaign. Each association has six bouts, with A for 30 s, rest for 135 s, B for 30 s, rest for 135 s: 1,980 s in total. Consequence orientation is counterbalanced.

The four initial focal memories define a checkpoint. A focal is eligible when both the frozen reader and the actual-memory criterion are satisfied there. No ineligible focal is redrawn: Stage A has 63/64 eligible focals, Stage B/C has 62/64. Every focal remains in local raw outputs and the eligibility table.

For each later association, W learns normally; N starts from the same state and receives exactly the same cues, consequences and timing with that association's plastic updates disabled. Their adaptation and counters agree. The effect E is the signed old-memory margin in W minus N, evaluated immediately or after a day on clones. The main trajectory continues only through W. A focal footprint is the difference between the full four-association checkpoint and a history omitting only that focal's writes. It includes any mediated changes to subsequent focal learning; it is not an isolated causal synapse label.

## A. Ordinary learning streams: direction carries the information

Sixteen fresh streams proceed from four focal associations to 96 total associations. The atlas contains 11,776 focal/event/horizon rows, including 5,796 eligible delayed rows.

| Diagnostic | Mean within-stream Spearman rho | 95% interval |
|---|---:|---:|
| Slow-footprint signed cosine | 0.906 | 0.875–0.932 |
| Signed cue-code cosine | 0.771 | 0.748–0.795 |
| Unsigned cue-support overlap | 0.017 | −0.003–0.039 |
| Whole-write magnitude | 0.005 | −0.020–0.025 |
| Existing slow-state magnitude | −0.005 | −0.039–0.027 |

Unsigned overlap alone misses the central distinction: overlapping representations can reinforce as well as interfere. Among delayed eligible events, 1,708 damage the old margin by more than 0.1 Hz, 1,823 reinforce it by more than 0.1 Hz, and 2,265 fall within that descriptive deadband. Average change is only +0.0145 Hz, masking the opposing effects.

Fixed ridge models were evaluated by leaving out one complete stream. Standardization and imputation use training streams only; alpha is fixed at 1. The outcome is delayed E, not classification accuracy of the fly model.

| Predictor block | Held-out R² | MAE, Hz | Sign accuracy outside ±0.1 Hz |
|---|---:|---:|---:|
| Training-mean baseline | −0.001 | 0.256 | 51.6% |
| Cue-only geometry and consequence orientation | 0.550 | 0.181 | 89.8% |
| Local realized-write features | 0.783 | 0.115 | 97.0% |
| Aggregate magnitudes/state covariates | 0.002 | 0.258 | 52.3% |
| Local + aggregate features | 0.784 | 0.116 | 97.0% |

The LOCAL interval for R² is 0.728–0.823; GLOBAL is −0.0046–0.0081. Adding aggregate features provides little incremental predictive value in this declared linear analysis. This is not a proof against every nonlinear global hypothesis. Cue features require no realized later write, but still use the evaluator's specified old and new cue/consequence information. LOCAL uses realized writes and causal footprints. Neither block is an autonomous controller available to the deployed learner without further work.

![Directional atlas](figures/01_directional_atlas.png)

![Held-out diagnostic prediction](figures/02_heldout_prediction.png)

The ordinary end-of-stream retention result is also informative. Of 63 initially eligible focal memories, **48/63** satisfy both criteria after 92 later learning associations plus one day, versus **60/63** under matched later sensory exposure without writes. The three failures already present under delayed no-write conditions remain visible; V66 does not claim that all forgetting is caused by later writes. These are focal-retention counts, not a new measurement of all 96 associations' capacity.

## B. Independent, outcome-blind role selection

At each of 16 independent four-focal checkpoints, all 256 fresh candidate cue pairs were simulated at both consequence orientations. Selection used write-vector direction and norms only, never the resulting focal-memory score. Qualifying pairs had opposite signs of slow-footprint cosine and role-0/role-1 ratios within [0.8,1.25] for total write norm, all three compartment norms and teaching-payload RMS. The same cue pair preserves encoder support and adaptation across role arms.

All 64 focals supplied 24 qualifying pairs. Of 1,536 selected focal/candidate pairs, **1,488** belong to the 62 eligible focals. The selected teaching RMS ratios range from 0.912 to 1.096; all other calipers pass. Selection lists were saved before candidate outcomes were probed, then independently reconstructed from feature-only files.

| Horizon | OPPOSE mean E | REINFORCE mean E | Paired difference, 95% interval |
|---|---:|---:|---:|
| Immediate | −1.207 Hz | +1.280 Hz | +2.487 [2.282, 2.692] Hz |
| One day — primary | −0.443 Hz | +0.501 Hz | **+0.944 [0.878, 1.013] Hz** |

The delayed difference is positive for 1,422/1,488 selected focal/candidate pairs. These pairs are dependent within streams; the confidence interval uses 16 stream means after averaging candidates and eligible focals. This is a prospective intervention conditioned on simulated write features, not a forecast of write formation without simulation.

The secondary sequence challenge replays 24 selected pairs in a fixed candidate order under opposite role schedules. Initial matching is not assumed to persist as the states diverge; realized alignment and norms are recorded at every step.

| After 24 selected associations + one day | Old memories meeting both criteria |
|---|---:|
| OPPOSE | **0/62** |
| REINFORCE | **62/62** |
| Matched NO-WRITE | 60/62 |
| LOW ABSOLUTE INITIAL ALIGNMENT | 58/62 |

The mean stream-weighted causal effects are −11.589 Hz for OPPOSE and +11.994 Hz for REINFORCE. This is deliberately selected stress, not a 100% ordinary-capacity claim. Low initial alignment is not exactly neutral: its delayed effect is −0.187 Hz [−0.243, −0.135], and alignment can change during later learning.

![Prospective interventions](figures/03_prospective_interventions.png)

The exploratory magnitude control obtained 73 high/low pairs across 14/16 checkpoints; 72 pairs were eligible. It required near-zero slow cosine, similar cosine, the same consequence role, matched active counts, teaching magnitude and naive output norm, but at least a 1.5-fold write-norm difference. The delayed high-minus-low effect is **−0.012 Hz [−0.052, +0.028]**. This does not show a systematic magnitude penalty under the feasible controls. Coverage is incomplete and local coordinate distributions remain unmatched, so it cannot establish the absence of a global causal contribution.

## C. Durable reversal remains a real limitation

Each Stage-B focal was separately reversed from the same four-association checkpoint. Doses of 0, 6 and 12 bouts were padded to a common 3,960-second endpoint before immediate and next-day evaluation. Padding is rest; therefore dose comparisons vary sensory exposure as intended. Each dose also has its own sensory/time-matched no-write counterfactual.

| Reversal dose | Both criteria at common endpoint, among eligible old memories | Both criteria one day later |
|---|---:|---:|
| 0 bouts | 0/62 | 0/62 |
| 6 bouts | **43/62** | **4/62** |
| 12 bouts | **60/62** | **52/62** |

The primary durable count includes only a memory that was demonstrably retained before reversal. Six bouts yield an average delayed write effect of +2.440 Hz toward the new contingency, yet its resulting new-contingency actual margin remains −1.062 Hz. Twelve bouts yield +6.676 Hz of delayed write effect and a resulting margin of +3.174 Hz. More reversal dose helps substantially but still leaves **10/62** initially eligible memories without joint delayed success.

Per-bout delayed effects on the old margin become more negative, from about −0.304 Hz at bout 1 to −0.772 Hz at bout 12, accompanied by increasingly opposite slow-footprint direction. These are local matched-bout effects at their respective states; summing them is not asserted to equal the whole reversal contrast.

This protocol differs from the historical load-12, serial-probe revision benchmark. Counts must not be substituted into that benchmark or advertised as a repair. Plot points and intervals weight streams equally; table counts pool the 62 focals, so their percentages differ slightly.

![Durable reversal](figures/04_durable_revision.png)

## What the localization does and does not establish

**Supported:** in this fixed effective model, the direction of later writes on shared coordinates is a useful and causally manipulable determinant of old-memory expression. The effect survives independent inputs and same-cue, magnitude-matched role controls. A scalar description of accumulated write amount loses most of the relevant information.

**Still unresolved:** why the teaching/feedback system produces those conflicting writes, how to protect old memories using only information actually available to the learner, and whether this mechanism depends on real wiring compared with matched random wiring. V66 does not add an anatomical null, a cross-specimen replication, a biological task-capacity test or a new physiology validation. The paired consequence intervention changes teaching signals through the existing circuit; it does not independently manipulate cosine while holding every other directional feature fixed.

Cosine is also not a complete mechanistic description. **58 delayed damaging events have positive slow-footprint cosine.** Old-memory expression additionally weights coordinates through current cue activity, output routing and clipping. The clipping-aware gradient has delayed MAE 0.0043 Hz, versus 0.0499 Hz without clipping; exact endpoint reconstruction reaches floating-point accuracy. This accuracy follows from the equations and is labelled an expression audit, not learned predictive generalization.

Replacing only outside-focal-support endpoint state has exactly zero effect, and replacing only inside-support state reconstructs the full effect exactly. That is an architectural property of the reader path. It does not exclude whole-population feedback affecting the *formation* of writes. The checkpoint footprints and future outcome scores used by the evaluator must never be silently supplied to a proposed biological learner as task-indexed protection masks.

![Controls and expression limits](figures/05_controls_and_limits.png)

The recent Huang raw-data/source audit remains a guardrail, with unresolved raw-to-summary transformations and published-mean discrepancies documented in the parent audit. It provides no measured odor-to-88-channel mapping or individual KC synaptic state for this campaign. V66 neither uses those records for candidate selection nor silently treats the historical fit as a new validation against them.

## Verification, cost and next decision

Stage 0 reproduced the original full parent with **zero state and rate error**. The saved V65 fixture differs by at most 4.44e−16. Save/restore, continued dynamics, adaptation matching and branch isolation pass. All 32 run receipts and numerical checks pass; 19 final acceptance checks include input separation, caliper reconstruction without outcomes, 60 unchanged transitive V65 assets, and an unchanged V65 release ZIP. Independent replay from saved intervention checkpoints differs from recorded effects by at most **1.11e−16**.

The main campaign used two single-threaded worker processes, **289.9 summed worker seconds**, with peak reported RSS of **234.8 MB for one worker**. This is not an assertion that total machine memory use was 234.8 MB. Raw per-stream trajectories and checkpoints stay in the project and are excluded from the compact supplement. No new source dataset was downloaded.

The next bounded question should concern **write formation**, while keeping this V66 diagnostic as a regression panel: separate cue-overlap contributions from state-dependent feedback contributions using matched teacher-signal interventions, then consider one repair that can use local runtime information. A protection rule that reads the evaluator's old cue identities or omit-history footprints would not count as distilling a persistent core. Any proposed repair must preserve acquisition and legitimate reversal as well as old-memory retention, and must face the reconciled physiology constraints. This report does not launch V67.

The delivered [acceptance checks](checks/acceptance_checks.csv), [final metrics](results/summary.json), [frozen configuration](config.json), [diagnostic models](models/README.md) and [reproduction instructions](README.md) make this conclusion reviewable. **V66 succeeds as causal diagnosis; the persistent learning core remains unfinished.**
