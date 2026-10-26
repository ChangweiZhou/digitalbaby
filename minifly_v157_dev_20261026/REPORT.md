# V157 A1 — three-cycle report

**Result: the fixed-dose scheduling contrast does not establish improved unseen addition. Primary complete-answer endpoints are auditable; one secondary margin artifact requirement was not fulfilled.**

The experiment kept N_2B/P_SLOW, native Full151 stores, q=0, the 257-class C16 port, gradual warmup and signed local teacher unchanged. No arithmetic parser, arithmetic feature, new learned head, attention, Transformer or ELM was introduced. B was not resumed and architecture C was not implemented.

## Main finding

Clustering eight observations per cue produces some complete taught answers, so the native system is not universally incapable of memorizing a NEW fact. But its advantage at the final equal-dose boundary is only 0 or 1 answer out of 16 across these comparisons. Earlier OLD value teaching makes final NEW acquisition worse than the matched OLD-write-inhibited condition in both worlds. That is a paired, descriptive effect at this dose, not a family-wide negative-transfer theorem.

| World | OLD prior | NEW interleaved | NEW clustered | Clustered minus interleaved |
|---|---|---:|---:|---:|
| 202610265701 | W | 3/16 | 4/16 | +1 |
| 202610265701 | SH_OLD | 6/16 | 6/16 | +0 |
| 202610265702 | W | 1/16 | 1/16 | +0 |
| 202610265702 | SH_OLD | 3/16 | 4/16 | +1 |

Both schedules finish with 128 NEW presentations, eight per cue and identical final feedback/time budgets. Intermediate checkpoints have different cue coverage and target/byte/time totals; their curves are not equal-information comparisons. SH_OLD observes the same OLD feedback with those value writes inhibited, and **does write during NEW**.

## Acquisition versus survival

| World / prior / schedule | Correct immediately after teaching, out of 128 | Distinct NEW ever correct after its own teaching | Correct after last own teaching | Final NEW |
|---|---:|---:|---:|---:|
| 5701 / W / INTERLEAVED | 13 | 3/16 | 3/16 | 3/16 |
| 5701 / W / CLUSTERED | 13 | 5/16 | 5/16 | 4/16 |
| 5701 / SH_OLD / INTERLEAVED | 68 | 10/16 | 8/16 | 6/16 |
| 5701 / SH_OLD / CLUSTERED | 48 | 11/16 | 11/16 | 6/16 |
| 5702 / W / INTERLEAVED | 9 | 2/16 | 2/16 | 1/16 |
| 5702 / W / CLUSTERED | 13 | 2/16 | 2/16 | 1/16 |
| 5702 / SH_OLD / INTERLEAVED | 31 | 7/16 | 3/16 | 3/16 |
| 5702 / SH_OLD / CLUSTERED | 34 | 10/16 | 10/16 | 4/16 |

For example, SH_OLD/CLUSTERED reached 11/16 and 10/16 at the cues’ individual last-teaching times, but ended at 6/16 and 4/16. Those individual times are not simultaneous model checkpoints. The losses include elapsed time and intervening observations/teaching; they do not alone identify all losses as competing-write damage. The sampled no-write/idle forks identify local effects only.

The control probes recorded four complete-answer losses attributable to the sampled current write across their 80-record panels. Other sampled writes helped, and sampled averages are not a stationary learning kernel. No full-panel pre-record probe exists, so there is no full-panel I/O/W telescoping claim here.

## Retention updates the earlier diagnosis

World 5701 W began with 0/64 correct OLD answers, so initially-acquired-subset retention is unmeasurable. World 5702 W began with **5/64** and ended with **0/64** under both schedules, also after delay. These are five shared acquired OLD items followed in two correlated branches, not ten independent forgotten facts.

Thus V156’s conditional 15/15 final retention result must not be generalized to this longer, different history. The V156 result stays valid in its own states/worlds. V157 demonstrates loss under an actual subsequent history, but its sparse controls do not uniquely attribute every lost OLD answer to time, observations or writes. Retention is E1-R, not E2.

## Never-taught addition

| World | Schedule | W held | SH_OLD held | W minus SH_OLD | Best declared countermodel |
|---|---|---:|---:|---:|---:|
| 202610265701 | INTERLEAVED | 1/50 | 1/50 | +0 | 2/50 |
| 202610265701 | CLUSTERED | 1/50 | 1/50 | +0 | 2/50 |
| 202610265702 | INTERLEAVED | 0/50 | 0/50 | +0 | 2/50 |
| 202610265702 | CLUSTERED | 0/50 | 0/50 | +0 | 2/50 |

Final and delayed held counts are identical. Every branch scores 1/50 in world 5701 and 0/50 in world 5702. W−SH_OLD is zero everywhere, and native W is below a declared exposure countermodel in both worlds. There is no established unseen-addition gain. The prior control isolates OLD value teaching conditional on NEW teaching; it does not isolate every possible contribution of NEW learning.

All eight taught histories and both worlds’ taught audits were frozen globally before held generations. The 800 readonly held decisions are repeated endpoint/branch observations of **100 distinct held orbits**, not 800 independent relations. Only prompts entered the held policy; answers were joined afterward by the external reducer. No held feedback or expected prefix entered the learner.

Raw-key and canonical-orbit coverage are zero on held questions. Their fallback can still guess: zero coverage is distinct from zero accuracy. The best countermodel was prompt-length frequency (2/50) in world 5701; the frequency-based family reached 2/50 in world 5702. Counters used only observed main-history feedback, including inhibited OLD observations, and never supplied native answers.

## The three cycles, failures and revisions

1. **Initial design/audit/technical test.** Native read/control checks passed, but the checkpoint check wrongly required equality of pickle byte streams across restore. It stopped before science after eight synthetic records. The assertion and failed attempt are preserved.
2. **Revised qualification and admission.** Compared all operative values, generated decisions and the next native transition; all passed. Continued from the existing technical checkpoint using three new synthetic records, rather than replaying its history. The measured full-roster forecast was 934.76 CPU seconds, so the original 900-second execution allocation did not admit science.
3. **Explicit resource revision and fixed A1 test.** Reallocated 60 CPU seconds from audit to execution: 120 qualification / 960 execution / 120 audit, with the **same 1,200-second total cap** and unchanged scientific roster. A duplicate `stage` keyword then stopped status logging after the first 64 OLD records. An AST-checked status-only fix resumed the actual preserved pass boundary, with counters/clocks/budgets carried through. No birth replacement, dataset regeneration or training replay occurred. Both worlds and all endpoints then completed.

The original rejection, controller failure, source versions and partial checkpoint are preserved; this is not a claim of three clean, failure-free cycles. Native learning equations, dataset identities and outcome criteria were not changed.

## Audit limitation: immediate margin persistence

The code computed all required teacher-prefix margins at every immediate post-write query, but its immediate-summary serializer omitted those arrays. Complete own answers are retained for all **1,024** NEW presentations. Only the **32** boundary-current margin snapshots are independently recoverable; **992** other immediate snapshots were not retained.

This is a real secondary artifact deviation, not a successful full-protocol audit. The final audit accepts the primary complete-answer/exposure contrasts **with this declared limitation**. It does not accept an every-presentation margin mechanism claim. No histories or model queries were repeated to repair the missing arrays. The revised design requires a whole-event artifact-schema fixture before any future run.

## Cost and evidence boundary

- **3,115 observed records / 8,783 reply-EOS targets**: 3,104 scientific records + 11 technical. Scientific components: 1,024 OLD updating, 1,024 OLD inhibited, 1,024 NEW updating and 32 discarded NEW inhibited probes.
- Four physical OLD preparations and eight correlated NEW continuations; cloned prefixes are not re-counted as separately trained lives. The original 64-record prefix is counted exactly once.
- **9,664** scientific own/TF pairs, 32 immediate-boundary result reuses, 44 technical pairs and 20 technical own queries; these technical requests fit the 64-unit reserve. Held own-only generations: **800**.
- Evaluator distributions: **45,601 physically computed / 55,305 logical**; 9,704 identical immutable first reads were reused. Bitwise equivalence to the V156 evaluator was qualified; both continuations remain independent after that shared read.
- Cumulative worker CPU **707.95 s**, plus saved-artifact reduction **0.33 s**, below 1,200. Cumulative wall **1667.19 s** includes the between-cycle operations interval. Science execution subcost **696.44 s** is below the revised 960-second subcap.
- Peak execution-process RSS **442.53 MiB**; recorded cycle-two peak was about 407.56 MiB. Raw local output at final audit **575.89 MiB**, below 1 GiB.
- At least 79.59 GB of mutable-clone traffic and 330,228 global side sorts were instrumented. These are inherited evaluator/core costs, not evidence of sparse production computation; internal clone traffic is incompletely counted. Shell preparation/narrative-writing CPU and tiny process-shutdown gaps are not separately metered, so no end-to-end speed gain is claimed.

The primary taught results are E1; the held E3 assay produced no causal gain beyond controls. This is not an E2 presentation-change experiment, not formal confirmation and not adoption. Eight repetitions do not count as eight independent examples. No sample/compute-efficiency improvement on unseen addition is established.

## Artifacts and next action

- [Revised design](REVISED_DESIGN.md)
- [Results and countermodels](RESULTS.json)
- [Acquisition curves](ACQUISITION_CURVES.json)
- [Final audit and limitations](FINAL_AUDIT.json)
- [Actual counts](COUNTS.json)

The tested scheduling branch closes at this dose; do not automatically increase repetition, rates or protection machinery. Architecture protocol C remains independently open and requires an exact forward/local-rule specification before implementation. V155/B is not resumed by these results.

Checkpoints, data, per-prefix arrays and raw responses remain local. The compact bundle contains source, aggregate results and audits without raw data. Nothing has been pushed, no follow-up was launched, and the historical arithmetic pointer and V156 results remain unchanged.
