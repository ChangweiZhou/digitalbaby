# MiniFly V9.6 — source-defined byte-value interventions

## Decision

The constructive comparison is complete, but neither candidate is established as a repair of persistent write interference. The historical **B write reassignment** transfers poorly to this byte workload. The **four-bank comparator** improves absolute old and some new delayed choices at extra state cost; almost all of its old-choice advantage is already present before the four new cohorts. It does **not** measurably reduce the old-choice penalty caused by subsequent writes. Its within-model analog write penalty is larger than the parent's, although the much larger analog scale makes this a qualified comparison. Retain V9.5 as the reference rather than promoting either candidate to the accepted persistent core.

This is a result about the declared synthetic byte-association workload, not a fly-physiology claim or a general impossibility result.

## What was run

Three frozen arms saw the same bytes, rewards, times, and four overlapping new association cohorts in 80 untouched paired worlds, indices 701–780. `REF` is the V9.5 learner. `B_TRANSFER` applies the historical A8/B local write reassignment with its PN-calibrated 5,177-element vector unchanged. `FB4_TRANSFER` combines B with four fast/slow alpha stores addressed by a frozen eight-centre router acting on the current KC code; the router never sees evaluator labels or future probes. Every arm includes all-write and matched no-new-write histories. There was no performance-based parameter selection on the 80 worlds.

The engineering world 401 reproduced archived V9.5 probes exactly; a one-bank wrapper reproduced B probes and states exactly. Four banks were separately checked to decay during passive time, with zero numerical error in that regression. The source/configuration lock and preflight preceded prospective execution.

## Old and new memory

Choice is the fraction of the three within-set aversive-versus-neutral decisions correct, averaged over worlds. Brackets are paired 95% *t* intervals across 80 worlds, in percentage points (pp). These are descriptive, unadjusted comparisons of a single frozen model seed.

| Arm | Old before new cohorts | Old after four cohorts, D4 | Old D4 versus REF | Own write-minus-no-write at D4 |
|---|---:|---:|---:|---:|
| REF | 74.17% | 65.00% | — | −9.17 pp [−15.30, −3.04] |
| B_TRANSFER | 74.58% | 63.75% | −1.25 pp [−5.74, +3.24] | −10.42 pp [−15.77, −5.06] |
| FB4_TRANSFER | 81.67% | 72.92% | +7.92 pp [+1.11, +14.72] | −8.75 pp [−12.79, −4.71] |

The decisive difference-in-differences is **FB4 versus REF in the own write penalty: +0.42 pp [−6.10, +6.93]**. FB4's initial old-choice advantage is +7.50 pp [+2.78, +12.22]; its final advantage is +7.92 pp. Thus the experiment supports stronger absolute performance from birth, not lower write-induced old-choice harm. On old pairs that both models knew before new writing, REF lost 33/170 and FB4 lost 27/170; the paired world-level difference is −2.50 pp [−8.13, +3.13] of all old pairs, also inconclusive.

| Arm | New cohort 1 at E1 → D4 | Cohort 2 at E2 → D4 | Cohort 3 at E3 → D4 | Cohort 4 at E4 → D4 |
|---|---:|---:|---:|---:|
| REF | 90.42% → 68.75% | 87.08% → 67.92% | 87.50% → 77.50% | 86.67% → 72.08% |
| B_TRANSFER | 90.00% → 64.58% | 87.50% → 62.08% | 90.00% → 70.83% | 86.67% → 72.50% |
| FB4_TRANSFER | 92.50% → 72.08% | 90.83% → 70.42% | 90.42% → 80.00% | 89.17% → 82.08% |

B's delayed cohorts 2 and 3 fall below REF by −5.83 pp [−10.05, −1.61] and −6.67 pp [−11.45, −1.89], respectively. FB4's delayed cohort 4 is +10.00 pp [+4.80, +15.20] above REF, with +7.50 pp [+1.15, +13.85] less E4-to-D4 decline. For cohorts 1–3, the delayed FB4-versus-REF intervals span zero, and none shows a clear retention-change advantage. Immediate new acquisition remains high in all arms, so the old-choice pattern is not an artifact of simply refusing to learn new cohorts.

The analog margin reveals a separate caution. FB4's old D4 mean is 0.42184 Hz versus REF's 0.03284 Hz, but FB4's distribution is wide (median 0.07437 Hz; range −0.759 to +4.937 Hz). Its old write-minus-no-write mean is −0.17741 Hz versus REF's −0.01897 Hz; the paired change is −0.15843 Hz [−0.28821, −0.02865]. The enlarged margin is already present before new learning (FB4 0.60583 Hz versus REF 0.05241 Hz). The numbers therefore cannot be read as a clean retention gain; the banks also alter the output scale and starting competence.

## State and addressing costs

| Arm | Persistent deployable bytes | Increase versus REF | Extra allocated scratch |
|---|---:|---:|---:|
| REF | 3,050,272 | — | 0 |
| B_TRANSFER | 3,091,688 | +41,416 (+1.36%) | 0 |
| FB4_TRANSFER | 3,671,576 | +621,304 (+20.37%) | 82,832 |

The FB4 increase consists of 331,328 bytes of added mutable bank state, 331,392 bytes of fixed router centres, and the 41,416-byte frozen B vector, on top of REF. Its total counted allocation is 3,754,408 bytes. Four-bank write occupancy is strongly unequal (3,779 / 17,874 / 6,128 / 1,019 events); delayed cue reads show the same pattern (315 / 1,489 / 511 / 85), and 874/1,200 queried pair opponents share a bank. This is a fixed but poorly balanced address rule. The experiment does not isolate the effect of capacity, routing, B, or altered gain.

## Evidence and limits

The read-only post-run auditor verified all frozen source hashes and 80 receipt hashes; regenerated symbol sets and byte schedules; checked arm, branch, target, and count rosters; and rederived **19,440** probe scores from their three pair margins. It reproduced the key old/new endpoint means and confirmed that the independent byte predictor remained identical across all three arms in each world. The passive-time test verified that all four bank states decay. The original V9.5 audit is already complete and was not reopened.

The receipts and source assets stay in the project; the compact bundle contains this report, the frozen model/configuration code, summary results, and audit/acceptance records. This 80-world sample cannot establish 1-pp noninferiority, the reported intervals are not multiplicity adjusted, and only one fixed learner/router seed was tested. A separate study would be required to disentangle bank capacity from routing and output scaling. A11 PN-cue characterization remains a distinct workstream.
