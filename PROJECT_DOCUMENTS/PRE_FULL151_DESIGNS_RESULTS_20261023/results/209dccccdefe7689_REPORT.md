# MiniFly iteration 9 — association load, specific interference, and durable revision

13 September 2026. **Preferred learner: iteration5/bi. Fixed observer: iteration8/current_RBF64. No model promotion or KC reduction.**

The new measurements expose two practical limits. The parent retains almost every association when each is trained in isolation and rested for the same age, but loses distinctions when many associations share its learning state. Extra reversal trials improve revision, yet immediate success often disappears the next day. Equal allocation improves some moderate-load and delayed-reversal outcomes, but remains physiologically rejected and becomes worse than the parent at the largest snapshot load.

This iteration therefore supplies an empirical preservation target for distillation: **the association-load curve, specificity under later writes, and delayed revision at a stated exposure cost.** It does not establish a minimum KC count or recover an additional biological learning mechanism.

## What was actually tested

The [protocol](PROTOCOL.md) and [configuration](config.json) were frozen before new outcomes. No learning parameter, anatomical route, sparsification setting, readout coefficient or decision threshold changed. The same FlyWire v630 interface contains 5,177 KCs, 302 selected PNs and 88 input channels. The parent remains 165,688 mutable bytes; the observer adds zero mutable bytes and retains its existing 170,008 fixed numeric bytes. Equal allocation changes only the already declared slow-write fraction and is an engineering comparator.

Eight new streams each supply 192 A-punished/B-unpunished discriminations and eight untrained control pairs. One association is one cue pair, trained for six bouts over 1,980 seconds; it is not a 24-cue bank. Loads are nested prefixes of 1, 4, 12, 24, 48, 96 and 192 associations. Each cue activates eight of the same 88 PN-type channels. A separate twelve-bank reversal panel adds another 288 new patterns. All **3,488 new patterns** are unique and excluded from saved earlier inputs. The twelve iteration-6 reserved banks and twelve iteration-8 test banks are retrospective regression panels, reported separately.

Capacity outcomes are measured five minutes and 24 hours after loading. Primary observations are independent diagnostic branches from a common snapshot. A separately labelled serial audit executes actual A/B probes, allowing them to change the learner. Its probes do not enter another load's training stream. An equation-derived, same-activity baseline supplies an offline actual-memory diagnostic; it never enters the fixed reader. The experiment reuses anatomical parent assets. It neither rereads the large BANC synapse table nor introduces biological replication or measured odor-to-PN activity.

## Association load has a measurable cost

The table shows **24-hour snapshot choices**, pooled over eight streams. Each cell counts every loaded pair, not only the first few memories.

| Associations per stream | Parent correct | Equal allocation correct | Parent isolated, age matched |
|---:|---:|---:|---:|
| 1 | 8/8 — 100% | 8/8 — 100% | 100% |
| 4 | 32/32 — 100% | 32/32 — 100% | 100% |
| 12 | 92/96 — 95.8% | 95/96 — 99.0% | 99.0% |
| 24 | 170/192 — 88.5% | 181/192 — 94.3% | 99.0% |
| 48 | 327/384 — 85.2% | 346/384 — 90.1% | 99.0% |
| 96 | 562/768 — 73.2% | 557/768 — 72.5% | 98.7% |
| 192 | 1009/1536 — 65.7% | 862/1536 — 56.1% | 98.7% |

Equal allocation's isolated controls remain 100% throughout. Loading 192 associations consumes **105.6 model hours**, so omitting memory-age controls would have confounded load with more than four days of elapsed time. The isolated controls receive the same delay since each pair completed training. They do not match sensory state during acquisition; the within-state experiment below addresses that limitation for the first four memories.

The frozen descriptive criterion was 87.5% mean correct, carried from the earlier 7/8 standard. The parent's first tested snapshot load below this criterion is **48**, versus **96** for equal allocation. These are observed schedule boundaries, not exact capacities or neuron-count bounds. At load 48, the parent's 95% stream-bootstrap interval is **81.0–88.5%**, spanning the criterion; at load 96 it is **70.2–76.0%**. Equal allocation at load 48 is **88.3–92.2%**, and at load 96 **69.8–75.8%**. Intervals resample eight whole streams, not individual pairs. Nested loads and multiple descriptive comparisons do not provide independent biological replications.

The fixed reader is not the dominant source of this decline. At load 192, actual-memory success is **66.0%** for the parent and **56.1%** for equal allocation, close to reader choices. Across all capacity no-learning conditions, there are **0/12,960 pair decisions** and **0/25,920 single-cue decisions** above threshold; pooled baseline RMSE is 0.160 Hz. These counts include repeated conditions and are engineering checks, not independent trial estimates.

Actual serial probes change the results: at load 192 and 24 hours the parent gives **937/1536 = 61.0%**, and equal allocation **982/1536 = 63.9%**. The observer is fixed, but reading the memory presents cues to a plastic learner. Snapshot capacity and capacity under a particular readout schedule must remain separate. Both serial 24-hour curves still cross the descriptive criterion first at 48 and 96 respectively.

![Association load, serial readout, untrained cues and causal controls](figures/association_load.png)

Figure: all panels use the 24-hour delay. Shading in A is the stream-bootstrap interval; the dashed horizontal line marks the descriptive 87.5% criterion. Panel D follows the same first four jointly acquired memories from a shared checkpoint.

Final tables: [load curves](results/capacity_summary.csv), [age quartiles](results/capacity_age_quartiles.csv), and [descriptive boundaries](results/capacity_boundaries.csv).

## Later writes corrupt distinctions without uniformly weakening memory

After four pairs have been jointly trained, three branches receive all later events with identical durations: normal learning, REST, or SENSORY with all plastic writes disabled. The sensory branch preserves the exposure/adaptation history. At load 192 and after a further day:

| First four memories, across eight streams | Parent | Equal allocation |
|---|---:|---:|
| Matched REST | 32/32 correct | 32/32 correct |
| Same sensory exposures, no later writes | 32/32 | 32/32 |
| Full later learning | 24/32 | 23/32 |

This directly implicates later writes in the lost choices for this fixed cohort. It is more specific than blaming the passage of time or sensory adaptation. It is not a whole-brain or biological causal experiment.

The signed mean alone is misleading. For the parent, mean A-minus-B memory changes from **−3.09 Hz** in the no-write control to **−6.33 Hz** with later learning: the aggregate aversive contrast becomes stronger while eight of 32 correct distinctions are lost. The mean absolute write-induced contrast change is **6.99 Hz**. Some memories are reinforced and others are distorted or reverse sign. At the largest load, success is similar across age quartiles (parent 64.6–66.9%; equal allocation 55.2–58.1%), so a simple oldest-first decay account misses this pattern.

Clipping contributes differently in the two allocations. At load 192, **70.5%** of equal-allocation alpha3 probe outputs are clipped, compared with **2.5%** for the parent. Removing only the final alpha3 clip in an offline expression diagnostic raises equal allocation's success from 56.1% to **66.5%**, still far below its isolated controls. The corresponding parent preclip diagnostic is **64.3%**, so final alpha3 clipping alone does not explain its failure. This diagnostic retains gamma clipping and the learned dynamics; it is not an accepted unclipped model.

The observer's no-learning specificity also does not guarantee specificity after learning. At load 192, it makes a preference on **60/64 untrained cue pairs** for the parent and **59/64** for equal allocation. Single-cue decision rates are 95.3% and 97.7%. These are learning-induced generalization responses on distinct but potentially overlapping engineered cues, not baseline-estimation false alarms. They do not prove biologically inappropriate generalization. They do show that high trained-pair accuracy alone would not establish arbitrary-association specificity or novelty rejection.

Evidence: [causal signed decomposition](results/focal_interference_decomposition.csv) and [choice losses and absolute distortion](results/focal_choice_and_distortion.csv). In this panel a uniform retention multiplier is an inadequate summary of memory quality. A later distillation should preserve the distribution of specific write effects, including unwanted generalization.

## More exposure helps reversal, but immediate revision is an incomplete endpoint

No frozen pre-reversal checkpoint collection was present in iteration6. We reconstructed **144 checkpoints** from exact saved inputs: 36 banks × two allocations × two readout histories. All twelve original reserved-bank six-bout outcomes reproduce for both allocations; iteration-8 snapshot scores and serial revision counts also reproduce. The new checkpoints and per-pair outputs remain available in the project but are excluded from the compact delivery.

Each checkpoint branches to 0, 6, 9 or 12 reversal bouts. Shock amplitude and individual presentation duration remain fixed. For delayed comparisons, shorter-dose branches receive REST padding to the common 12-bout endpoint, then a further 24-hour rest. Thus added training is separated from mere added elapsed time. Both allocations receive the same doses. The serial-history results below include actual acquisition/interference readout exposures and target-first serial endpoint probes.

| Twelve untouched new banks | 6 bouts | 9 bouts | 12 bouts |
|---|---:|---:|---:|
| Parent: immediate target choice reversed | 12/12 | 12/12 | 12/12 |
| Equal allocation: immediate reversed | 10/12 | 10/12 | 11/12 |
| Parent: reversed at common endpoint +24 h | 3/12 | 8/12 | 10/12 |
| Equal allocation: reversed at common endpoint +24 h | 7/12 | 8/12 | 10/12 |
| Parent: other eleven choices at +24 h | 121/132 | 122/132 | 120/132 |
| Equal allocation: other eleven choices at +24 h | 130/132 | 129/132 | 129/132 |

The parent six-bout delayed actual-memory diagnostic passes **2/12**, versus the observer's 3/12; this one threshold discrepancy is not an extra stored memory. At twelve bouts both allocations pass the delayed actual-memory and reader criteria on 10/12 new banks.

Doubling from six to twelve bouts adds **180 shock-presentation seconds and 1,980 training seconds** per bank. The parent delayed-reader improvement is +58.3 percentage points, stream-bootstrap interval **+25.0 to +83.3**; equal allocation improves +25.0 points, interval **0 to +50.0**. The latter effect is less precisely resolved with twelve banks. Collateral effects at the equal-time delayed endpoint are small in this sample, not zero: versus zero-dose controls, twelve bouts lose two old choices and gain one for the parent, and lose one for equal allocation. These bounds do not establish the physiological acceptability of extra exposure.

![Reversal dose, next-day persistence and collateral memories](figures/revision_dose.png)

There is another metric issue. Before any reversal, only **7/12** new serial parent cases still express the old correct choice; two already express the proposed reversed choice and three abstain. Equal allocation retains the old choice in **11/12**, with one already reversed. Among cases that really retain the old choice, the parent's six-bout immediate **7/7** switches become **0/7** the next day; twelve bouts preserve **6/7** switches the next day. Equal allocation's corresponding delayed counts are **6/11** and **9/11**. The headline twelve-bank counts remain comparable with earlier iterations, while the [eligibility audit](results/revision_eligibility_audit.csv) prevents spontaneous loss of the old choice from being counted as evidence of successful overwriting.

The old difficult cases do not support a blanket dose rescue:

- **601002, equal allocation, original snapshot schedule:** six bouts leave −1.621 Hz; nine reach +1.291 Hz; twelve reach +4.173 Hz. After the common endpoint plus a day, nine return to −5.206 Hz. Twelve yield only +0.864 Hz of actual memory, below the 1-Hz criterion, even though the approximate reader crosses its threshold. With serial history, twelve bouts are −0.150 Hz the next day.
- **601004:** even twelve bouts remain wrong-sign, **−2.713 Hz** under the snapshot schedule and **−3.170 Hz** with serial history. Immediate alpha3 outputs are not clipped in this case, and the preclip contrast is the same. A final-output clipping fix alone cannot rescue it.

On the entire old iteration-6 reserved snapshot panel, equal allocation improves from **10/12** to **11/12**, not 12/12, at both nine and twelve bouts. Across the new panel, it also stops at 11/12 immediate and 10/12 delayed at double dose. Extra exposure is useful, but it neither removes every failure nor makes the fixed-dose tradeoff fictitious. A binary “dose versus mechanism” interpretation is too strong: dose acts through the existing dynamics, and their durable consequences must be measured.

Final tables: [revision outcomes](results/revision_summary.csv), [paired dose effects](results/revision_dose_effects.csv), [collateral choices](results/revision_collateral.csv), [old failure cases](results/two_legacy_failure_cases.csv), and [actual teaching-payload summaries](results/revision_teaching_summary.csv).

## Disposition of the external suggestions

Claude's handoff was found in the Downloads copy and copied into project sources for audit. The supplied revision table agrees with local frozen outputs within 3.6×10⁻¹⁵ Hz. The quoted stored ×3.31 and displacement ×1.46 ratios pool **13 banks**, including the legacy bank. On the twelve reserved banks alone they are **×3.09 and ×1.42**. The second quantity is net output displacement, not a direct teaching-drive measurement; its scaling cannot determine required extra exposure under clipping and feedback. See the [ratio audit](results/feedback_ratio_audit.csv).

The proposed “627 directions” reader compression uses a different object. Its script covers **5,177 × 97 KC→MBON graph rows** with a greedy cosine criterion. The actual observer projects through **5,177 × 4** weights, with linear rank four. A cosine cover is not a matrix basis, does not remove the cost of assigning/weighting all KCs, and does not establish an eightfold reader compression. That experiment was skipped because its premise did not transfer. No 647-KC reduction is recommended. Even perfect performance at the largest tested load would only provide a lower bound on task success, not justify removing neurons without a reduced-model comparison.

The supplied DoOR table is retained unused. Receptor responses mapped to glomerular labels do not supply a measured same-specimen PN transform. The existing incomplete LH test remains a scoped negative result, not a whole-LH exclusion. ChatGPT's 90/100 progress score and proposed evidence-consistency allocation were not adopted: neither follows from these data. No intermediate timescale was introduced.

Equal allocation remains rejected by the original physiological checks: conditioning RMSE **4.127 → 5.896 Hz**, spacing RMSE **4.396 → 14.838 Hz**. These are carried unchanged from earlier results; the new generated stimuli and four-day load schedule do not constitute biological fits. The parent's slow lifetime remains weakly identified at an earlier fitting boundary. This stress test is an extrapolation of that model, not evidence that a fly has this capacity or lifetime.

## What this changes for the persistent core

The current demonstrator has a usable fixed observer, but its learner does not preserve arbitrary distinctions under increasing writes, and a successful immediate revision need not persist. The next useful mechanistic comparison should explain the **specific distribution of cross-association write effects and the delayed survival of actual old-to-new switches**, using the existing encoding, teaching and output routes. A global mean-strength improvement, a spectral dimension count, or a longer trace alone would miss measured failures.

The direct distillation benchmark is now concrete: match the full parent's load/age curves, no-write controls, generalization responses and dose-dependent durable revision before claiming that a smaller KC population preserves its learning function. These tests do not select a particular compression algorithm. No new learning rule is warranted from this iteration alone; iteration5/bi remains the parent and equal allocation remains a comparator.

All **36 final implementation checks pass**, including exact checkpoint restoration, unchanged reader hash, retrospective outcomes, no-learning zero memory, disjoint inputs, fixed byte capacity, actual serial-step accounting, and equal elapsed times across dose controls. The pre-run fixture checks also passed. The load stage produced 52,480 pair records in 385.5 seconds with peak process RSS 249 MB; the reversal stage produced 20,736 records in 112.2 seconds with peak RSS 241 MB. They initially ran concurrently, so their durations are not a summed wall-clock campaign time. No download or additional user help was needed.

The compact supplement contains this report, final tables, four figure files, experiment/configuration code and acceptance records. Unchanged models/data, raw per-pair outputs, 144 checkpoints, earlier archives and caches are excluded and referenced by manifest hashes. See [reproduction instructions](README.md) and [implementation checks](results/implementation_checks.json). All four earlier compact archives remain byte-identical.
