**V42 completed: read authority on four fixed V41 histories**

My assessment is that SLOT-ONLY is a useful candidate for the next experiment, with a clear tradeoff. Suppressing exact probationary-trace reads reduces log loss while preserving the measured terminal recall endpoints. It does not dominate the alternatives: next-byte accuracy falls slightly, and retaining similarity-based slot reads still costs ordinary-byte prediction loss.

The work is complete: the recovered recommendation was refined into a fixed five-policy protocol; the interruption-friendly single-file runner was implemented and verified; all four audits ran; and the results, numerical tables, source and checks are included. V42 reuses the four V41 learning histories. It adds **zero training presentations and zero learner updates**. The complete latest chat specification was unavailable, so the policy panel and evaluation details are explicitly documented refinements.

The five policies switch which saved prediction is read. All learning, admission and memory contents remain fixed. READ-HYBRID permits both exact trace and supported slot reads. SLOT-ONLY suppresses exact trace reads; EXACT-ONLY suppresses slot similarity reads; ANCHOR-ONLY suppresses both. SUPER uses the distributed predictor throughout.

**What the results show**

| Policy | Whole loss, bits/byte ↓ | Next-byte accuracy | Old seen recall | Old held-out recall | Revised held-out after D |
|---|---:|---:|---:|---:|---:|
| READ-HYBRID | 2.4396 | 73.571% | 126/128 | 63/64 | 32/32 |
| SLOT-ONLY | 2.3821 | 73.421% | 126/128 | 63/64 | 32/32 |
| EXACT-ONLY | 2.3778 | 73.903% | 126/128 | 45/64 | 29/32 |
| ANCHOR-ONLY | 2.3202 | 73.754% | 126/128 | 45/64 | 29/32 |
| SUPER | 2.3331 | 73.747% | 98/128 | 45/64 | 29/32 |

Counts pool the four histories for transparency; statistical comparisons use the four seeds as their units. Old recall concerns untouched cohort-A identities after the later cohorts; each identity has two seen renderings and one held-out rendering. Revised held-out retention is a new read-only query on the saved final state, not a new training run.

SLOT-ONLY improves whole-stream loss by **0.05759 bits/byte** against READ-HYBRID; the paired Student 95% interval is [−0.06847, −0.04672]. Its remaining loss gap to SUPER is **+0.04898** [+0.00174, +0.09623]. The exact four-seed sign-reversal reference gives p=0.125 for both contrasts. These are exploratory, pointwise comparisons on reused histories, with only four independent seed units.

Lower log loss does not mean more correct next-byte guesses. SLOT-ONLY makes **92 fewer correct predictions across 61,376 historical byte events**, a decline of 0.150 percentage points from READ-HYBRID. Its paired accuracy interval spans zero. The log-loss and top-1 metrics measure different aspects of prediction, and both should remain visible.

The loss decomposition is precise. Exact traces contribute +0.057337 ordinary-byte and +0.000256 target-byte loss, expressed per whole-stream byte, relative to SUPER. Removing that authority accounts for the improvement. Supported slot similarity contributes another +0.061881 ordinary-byte loss. Exact consolidated anchors contribute a net benefit of about 0.01290 bits/byte. ANCHOR-ONLY therefore has the lowest average loss in this panel, but loses the old held-out advantage: 45/64 correct versus SLOT-ONLY's 63/64. Similarity-based retrieval carries both the observed transfer benefit and a substantial ordinary-byte cost.

**How much retention evidence this adds**

All five policies already achieve 512/512 on E1 held-out accuracy, including SUPER. That endpoint is saturated here. For the same 64 untouched old identities, being correct at E1 and every delayed held-out checkpoint after B, C and D is 63/64 for READ-HYBRID and SLOT-ONLY, versus 43/64 for SUPER, EXACT-ONLY and ANCHOR-ONLY. This conjunction uses the same identities across checkpoints; it is not a conjunction of seen and held-out recall.

Several equalities are structural. Held-out contexts never occur in training, so they cannot use an exact trace; suppressing exact-trace reads cannot change these probes. The old-seen final probes comprise 124 exact-slot reads and four no-candidate reads, with no exact traces. The final revision-retention probes also contain no exact traces: 64 exact-slot seen queries and 32 supported-slot held-out queries. These endpoints support retained consolidated memory, but do not test whether trace suppression preserves an unconsolidated fact.

The final current-seen panel does exercise the intervention: 61 of its 256 queries use exact traces. READ-HYBRID and SLOT-ONLY both remain 256/256 correct; average loss improves from 0.119465 to 0.058660 bits/query. That is useful narrow evidence. It does not replace probing early exact recall before consolidation or after varied intervening experience. Final revised-seen retention is 64/64 for both policies versus 60/64 for SUPER; revised-held-out retention is 32/32 versus 29/32. These small terminal samples should not be treated as fresh confirmation.

**Recommended next decision**

Carry SLOT-ONLY forward as a candidate alongside READ-HYBRID, with ANCHOR-ONLY and SUPER retaining their control roles. The next useful experiment is a small prospective test on fresh short histories, with shared learning enabled and the same fixed read masks. Measure delayed exact recall after a first exposure while facts are still probationary, plus difficult related contexts where slot transfer can help or hurt. Record pre-observation prediction scores so more selective arbitration can later be audited without another expensive learner replay. Keep the write rule unchanged. A new learned gate or a larger policy grid is premature.

A policy must never use target labels, identity classes or the evaluator's target/ordinary distinction to choose authority. Avoid a broad threshold sweep on these four histories. The present results justify a focused question about when a memory should answer; they do not establish general persistent learning, better sample efficiency, lower memory use, or a reduced neuron count. Disabling reads leaves the traces and the inherited 4,096-unit predictor in place.

**Execution and reproducibility**

The completed run evaluated 609 units per history, 2,436 total. A second invocation verified and reused all four completed records with **zero new evaluation units**. Reference preparation made 672 read-only model calls and evaluated 1,344 scalar predictions; original terminal outputs matched exactly and learner state hashes remained unchanged. The audit process used approximately 144.7 MiB peak resident memory. This is evaluation resource use, not a memory saving in the underlying learner.

Verification covers synthetic policy selection, common-stratum reconstruction, held-out context absence, same-identity joins, pooled attribution denominators, real SIGINT and SIGKILL resume, completed-run reuse, identity and corruption rejection, concurrent writers, and archive integrity. The runner keeps atomic checksummed checkpoints and a previous generation, runs sequentially, and enables macOS caffeinate. The same command resumes an interrupted evaluation:

```bash
python3 elm_v42_read_authority.py --install-deps
```

The standalone file embeds the validated observations. It reproduces the audit and analysis, not the original V41 learning. The results ZIP contains per-seed completed records, reference inputs, protocol, numeric tables, plots and provenance. The complete handoff adds development modules, the build helper, preparation and independent audit reports, and execution logs. The original V41 model checkpoints are only required to repeat the earlier reference extraction.

The detailed machine-generated report follows. Its estimates are unchanged from the fixed-source run.

---

# V42: fixed-history read-authority audit

All scheduled seed audits are complete.

Completed histories: **4/4**. Recorded seeds: 41101, 41102, 41103, 41104. **New training presentations: zero.**

V42 evaluates five read policies on the same SLOTS writing histories saved by V41. The four existing V41 seeds are reused; this is a post hoc intervention on retrieval and is not four new independent replications. The policy masks are fixed before the new audit is run. No threshold is selected or gate relaxed from these outcomes.

The original histories contain 15,344 byte presentations per seed, or 61,376 in total. These are reused observations, not newly presented training data. Reference preparation scans raw contexts and makes read-only terminal queries; it never calls the learner update. The standalone runner transforms embedded sufficient statistics and scalar probe outcomes, so it also performs zero learner updates and does not replay V41 training.

The recoverable chat recommendation was to separate probationary exact-trace authority from corroborated slot authority, retaining the firewall between fallback reading and writing. The operational policy masks, paired contrasts, context checks and stopping rules are documented as refinements in the accompanying protocol; they are not presented as verbatim recovery of an unseen numerical specification.

| Policy | Exact consolidated anchor | Exact probationary trace | Supported slot similarity |
|---|---:|---:|---:|
| READ-HYBRID | Yes | Yes | Yes |
| SLOT-ONLY | Yes | No | Yes |
| EXACT-ONLY | Yes | Yes | No |
| ANCHOR-ONLY | Yes | No | No |
| SUPER | No | No | No |

When a removed authority would have answered, the policy uses the saved SUPER prediction. The first four policies retain the saved READ-HYBRID nonauthoritative fallback, including its base fallback if the superposed output norm is effectively zero; SUPER always uses its own saved prediction. No read mask changes allocation, confirmation, demotion, output training or subsequent writing.

## Prediction loss and retention

The table shows equal-weight means over completed seeds. Loss is bits per observed byte; lower is better. Recall values are endpoint accuracies with their original finite denominators.

| Policy | Whole bpb ↓ | Target bpb ↓ | Non-target bpb ↓ | E1 held-out | Old held-out | Old seen | Current seen |
|---|---:|---:|---:|---:|---:|---:|---:|
| READ-HYBRID | 2.4396 | 1.7797 | 2.4904 | 100.00% | 98.44% | 98.44% | 100.00% |
| SLOT-ONLY | 2.3821 | 1.7761 | 2.4287 | 100.00% | 98.44% | 98.44% | 100.00% |
| EXACT-ONLY | 2.3778 | 1.7797 | 2.4238 | 100.00% | 70.31% | 98.44% | 100.00% |
| ANCHOR-ONLY | 2.3202 | 1.7761 | 2.3620 | 100.00% | 70.31% | 98.44% | 100.00% |
| SUPER | 2.3331 | 1.9586 | 2.3619 | 100.00% | 70.31% | 76.56% | 100.00% |

Primary paired differences use the same seed on both sides. Negative loss differences favor the first policy. The brackets are pointwise Student 95% intervals, conditional on the retained seed histories.

- SLOT-ONLY minus READ-HYBRID, whole loss: **-0.05759 [-0.06847, -0.04672] (n=4)** bits/byte.
- SLOT-ONLY minus SUPER, whole loss: **+0.04898 [+0.00174, +0.09623] (n=4)** bits/byte.

PER_SEED.csv preserves all policy/metric seed values. PAIRED_CONTRASTS.csv preserves every matched seed difference, intervals and both inferential reference calculations. A lower overall loss does not imply a uniform improvement: old seen, current seen, old held-out and revision endpoints must be inspected separately.

## Where the loss difference arises

Attribution is evaluated on frozen predictions before the actual byte is observed. Category membership comes from the inherited retrieval path; target versus non-target labels are used only for reporting. A policy never receives evaluator labels. The table pools event totals across completed seeds. Its final column divides by all scored stream bytes, including histories with zero events in that category. Category contributions within a scope therefore add to that scope’s contribution to whole-stream loss.

| Contrast | Scope | Authority category | Events | Δ bits/category event | Δ bits/whole byte |
|---|---|---|---:|---:|---:|
| READ-HYBRID minus SUPER | nontarget | exact_slot | 660 | 0.01263 | 0.000136 |
| READ-HYBRID minus SUPER | nontarget | exact_trace | 7,637 | 0.46080 | 0.057337 |
| READ-HYBRID minus SUPER | nontarget | no_candidate | 31,083 | 0.00000 | 0.000000 |
| READ-HYBRID minus SUPER | nontarget | supported_slot | 4,428 | 0.85772 | 0.061881 |
| READ-HYBRID minus SUPER | nontarget | tentative_trace | 13,184 | 0.00000 | 0.000000 |
| READ-HYBRID minus SUPER | target | exact_slot | 2,545 | -0.31429 | -0.013032 |
| READ-HYBRID minus SUPER | target | exact_trace | 98 | 0.16035 | 0.000256 |
| READ-HYBRID minus SUPER | target | no_candidate | 916 | 0.00000 | 0.000000 |
| READ-HYBRID minus SUPER | target | tentative_trace | 825 | 0.00000 | 0.000000 |
| SLOT-ONLY minus READ-HYBRID | nontarget | exact_slot | 660 | 0.00000 | 0.000000 |
| SLOT-ONLY minus READ-HYBRID | nontarget | exact_trace | 7,637 | -0.46080 | -0.057337 |
| SLOT-ONLY minus READ-HYBRID | nontarget | no_candidate | 31,083 | 0.00000 | 0.000000 |
| SLOT-ONLY minus READ-HYBRID | nontarget | supported_slot | 4,428 | 0.00000 | 0.000000 |
| SLOT-ONLY minus READ-HYBRID | nontarget | tentative_trace | 13,184 | 0.00000 | 0.000000 |
| SLOT-ONLY minus READ-HYBRID | target | exact_slot | 2,545 | 0.00000 | 0.000000 |
| SLOT-ONLY minus READ-HYBRID | target | exact_trace | 98 | -0.16035 | -0.000256 |
| SLOT-ONLY minus READ-HYBRID | target | no_candidate | 916 | 0.00000 | 0.000000 |
| SLOT-ONLY minus READ-HYBRID | target | tentative_trace | 825 | 0.00000 | 0.000000 |

The exact-trace and supported-similarity categories are disjoint for every scored query. Consequently READ-HYBRID − SLOT-ONLY − EXACT-ONLY + ANCHOR-ONLY is identically zero for loss and correctness at each event, up to arithmetic roundoff. This is an algebraic consequence of the masks. It supplies a bookkeeping check, not evidence that the mechanisms are biologically or statistically independent. It also does not describe how they would interact if their writing histories were allowed to change.

## Context evidence and scope of the conclusion

Historical held-out probes use rendering 2. The evaluator checks that their complete contexts were never among the preceding training contexts. Once that check passes, exact-trace removal cannot affect them: SLOT-ONLY and READ-HYBRID necessarily retain identical E1 and old held-out results. EXACT-ONLY and ANCHOR-ONLY likewise coincide there because both disable the remaining similarity route. Preservation of these endpoints is a structural implication of this fixture, not additional evidence of durable learning.

Seen-context endpoints are evaluated only where an appropriate recorded historical prediction or read-only final-state probe is available. A terminal-state check does not validate an unrecorded earlier seen-context endpoint. No such historical inference is substituted. The result records retain endpoint counts, context-proof outcomes and final read-only audit provenance; their authority should be assessed separately.

Every finalized seed record retains the supplied evaluator audits and provenance without rewriting them. ANALYSIS.json collects those records for inspection. Intermediate reports reflect completed records only; checkpoints can contain additional progress which has not yet qualified as a completed audit.

## Interpretation and next decision

On these fixed histories, removing exact-trace read authority reduces average whole-stream loss by 0.05759 bits/byte. SLOT-ONLY’s residual difference from SUPER is +0.04898 bits/byte. The category decomposition identifies which inherited reads account for that change. The seen-context results determine whether the improvement sacrifices exact recall that the held-out fixture cannot detect.

SLOT-ONLY still has a positive mean whole-stream loss gap to SUPER. Any case for retaining it therefore rests on the measured retention or recall tradeoff, rather than claiming best average prediction loss.

The appropriate carry-forward claim is conditional: this mask is a candidate read policy for the unchanged memory history. A future end-to-end learner would still need a prospective test on new streams with all learning enabled. V42 itself creates no new evidence about learning efficiency, capacity under further training, arbitrary natural byte streams or neuron reduction.

## Statistical interpretation

Seeds are the replication unit. Bytes, identities, repeated presentations and probe checkpoints are dependent within a seed and are not counted as independent observations. With all four existing seeds, Student intervals have three degrees of freedom. Their Gaussian paired-difference assumption cannot be diagnosed reliably from four points. The exact two-sided sign-reversal reference calculation enumerates all sixteen sign patterns and has minimum attainable p=0.125 for a nonzero four-seed contrast. A parametric Student p-value can be smaller under its assumptions; this does not evade the small independent sample size. Sign reversal additionally requires the relevant sign symmetry/exchangeability interpretation, rather than following automatically from paired data.

All comparisons are exploratory and pointwise; no family-wide confidence claim or post hoc winner qualification is made. Deterministic policy equalities are documented as equalities. For zero sample variance and a nonzero paired mean, a Student p-value is left blank rather than reporting a misleading numerical zero. Category estimates are exact decompositions of these recorded losses, not independently replicated causal mediators.

## Reproducibility files

The single-file runner embeds the validated reference data; it requires no historical model files and does not regenerate V41 learning histories. It regenerates these reports from checksummed completed records and resumes interrupted evaluation. PER_SEED.csv, SUMMARY.csv, PAIRED_CONTRASTS.csv, ATTRIBUTION.csv and POOLED_ATTRIBUTION.csv contain numeric tables. ANALYSIS.json contains completion status, preserved seed audits and provenance. The accompanying protocol specifies the fixed roster and operational refinements; execution logs and verification records describe source matching, historical checks and read-only behavior.
