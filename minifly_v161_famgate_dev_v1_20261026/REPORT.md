V161 FAMGATE V1: X2 FAIL as specified (seed 1: 0.773 on unseen members, pass >= 0.80; seed 0 passes); X3 not run. DEV_NOT_EVIDENCE (synthetic streams).

# V161 report: a familiarity gate on the specific pathway (26 October 2026)

**Question:** if the specific KC pathway (S) steps aside where it has no evidence, does the gated combination with the shared predictive pathway (G) generalise to unseen members while keeping memory of seen ones?

**Answer:**
* The gate does its job.
  * On its own, it separates novel contexts from familiar ones (AUC 0.92–0.93).
  * Combined, it makes the system follow G on unseen members and S on seen ones.
* X2 still fails on seed 1: 0.773 on unseen members against a required 0.80. Seed 0 passes with 0.817 and 0.985.
* The remaining limit is G's own accuracy, about 0.80.
* A post-hoc diagnostic traces that limit to G's slowly fading feedback trace.
* By the design, V161 stops, and X3 (text) was not run.

* **Design:** `minifly_v161_famgate_design_20261026/DESIGN.md`.
* **Grant:** `AUTHORIZATION.json` (default caps).
* **Ledgers:** `runlog/ATTEMPTS.jsonl` (3 attempts), `runlog/CYCLES.jsonl` (testbed *continue*, technical *stop*), `runlog/AMENDMENTS.jsonl` (1, diagnostic).

## The gate (design §1)
* For each output class c and KC j, the evidence count is the number of training positions with target c at which j was active. It is a count at the KC→output synapse, as in the fly's novelty detector.
* Familiarity f_t is the share of the active KCs with evidence for S's most probable class.
* S is switched off where f_t < τ. τ is the 5th percentile of f on calibration positions, which uses no test data; it came out at 0.967 on both seeds.

## C0 testbed (before learner code)
| | Seed 0 | Seed 1 |
|---|---:|---:|
| Gate turns S off: unseen / seen target positions | 74% / 3% | 79% / 2% |
| AUC of novelty, unseen against seen | 0.923 | 0.933 |
| S alone, unseen / seen (floor ≤ 0.65) | 0.634 / 0.996 | 0.616 / 1.000 |
| ORACLE-PoE without gate (V160), unseen / seen | 0.945 / 1.000 | 0.934 / 1.000 |
| **ORACLE-GATED**, unseen / seen (ceiling ≥ 0.90 / ≥ 0.95) | **1.000 / 1.000** | **0.975 / 0.996** |

The verdict was CONTINUE. With a perfect shared expert, the gate removes most of the loss that S's overconfidence caused in V160.

## C1 technical
* **Checks: 6 of 6 pass.**
  * familiarity against an independent per-row computation;
  * τ from calibration only;
  * determinism;
  * gate fully open reproduces V160's PoE exactly (0.7404 / 0.7025);
  * G-SHUF alone reproduces V160 (0.4979 / 0.5000);
  * validity: G-SHUF alone ≤ 0.65.
* **X2 failed on seed 1.**

  | Arm | Seed 0 (unseen / seen) | Seed 1 (unseen / seen) | Bits at targets, unseen (seed 0 / 1) |
  |---|---:|---:|---:|
  | S alone | 0.634 / 0.996 | 0.616 / 1.000 | 1.36 / 1.57 |
  | G alone | 0.817 / 0.801 | 0.802 / 0.777 | – |
  | PoE without gate (V160) | 0.740 / 0.996 | 0.702 / 1.000 | 2.22 / 2.48 |
  | **GATED** | **0.817 / 0.985** | **0.773 / 0.992** | 4.26 / 4.47 |
  | GATED-SHUF | 0.519 / 0.977 | 0.479 / 0.988 | – |

  Pass required GATED ≥ 0.80 on unseen and ≥ 0.95 on seen members, with GATED-SHUF ≤ 0.65.

## Post-hoc diagnostic (not gating; written after the failure)
Faster fades of G's feedback trace:

| Fade | G alone, unseen (seed 0 / 1) | GATED, unseen (seed 0 / 1) | GATED, seen (seed 0 / 1) |
|---|---:|---:|---:|
| 0.807 (as designed) | 0.817 / 0.802 | 0.817 / 0.773 | 0.985 / 0.992 |
| 0.5 | 0.826 / 0.839 | 0.826 / 0.810 | 0.989 / 0.996 |
| 0.3 | 0.843 / 0.855 | 0.843 / 0.818 | 0.992 / 1.000 |

* With the designed fade, each segment's class leaks into the next segment's target position. A faster fade removes most of that.
* **Caveat:** the fade was chosen after seeing these two seeds, so a confirmation needs **fresh streams**.

## What it shows
1. **A local familiarity signal solves the arbitration problem on this task.** The gate tracks the shared pathway where the memory lacks evidence, and the memory where it has it. The shuffled control stays at chance.
2. **The bottleneck has moved to the shared pathway's precision.** That is a property of its trace (fade), not of the arbitration.
3. **Probability quality is still poor where S is switched off** (4.3–4.5 bits at unseen targets, against 1.4–1.6 for S). Removing S throws away what it knows reliably, such as *which kind* of symbol comes next, and G alone spreads probability over every symbol.

   For text, where loss is measured in bits, the gate should *temper* S (lower its weight, or let G choose among S's top candidates) rather than remove it. X3's design already chooses the weights on calibration, but this needs a stated form.

## Accounting
* 0.211 CPU-h (C0 0.061, C1 0.150 including the diagnostic).
* 2,000 learner records, 1 of 3 test cycles, designs tried k = 2.
