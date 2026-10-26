V162 TEXT V1: (b) FAIL (no carry-over gain on text); (a) CANNOT_TEST (S floor 0.659 > 0.65 on one fresh seed). DEV_NOT_EVIDENCE.

# V162 report: freezing the local fixes and testing on text (26 October 2026)

**Question:** with the local fixes fixed in advance (feedback-trace fade 0.3, a soft familiarity gate), does the new architecture help where language needs it? The architecture is the KC memory S plus a shared predictive pathway G with familiarity-gated arbitration.

**Answer: no, not at byte level on this text.**
* On *Edgeworth* carry-over (10 KB of readout data), calibration rejected every combination in favour of the memory alone, so the gain is zero.
* The synthetic confirmation could not test its claim, because the memory alone was just above its floor on one fresh seed. The non-gating numbers fall short of 0.80 anyway.

* **Design:** `minifly_v162_text_design_20261026/DESIGN.md`.
* **Grant:** `AUTHORIZATION.json` (default caps).
* **Ledgers:** `runlog/ATTEMPTS.jsonl` (3 attempts), `runlog/CYCLES.jsonl` (testbed *continue*, technical *continue*, dev *stop*).

## (a) Synthetic confirmation, fresh seeds 2 and 3
* **C0 testbed: CANNOT_TEST.** S alone scored 0.659 on unseen members of seed 2 (floor 0.65) and 0.589 on seed 3.
  * The ceiling was met: ORACLE-GATED scored 1.000 / 1.000 on both seeds.
  * The gate again separated novel from familiar contexts (AUC 0.92 and 0.94).
* **C1 technical: 4 of 4 checks pass.**
  * the code reproduces V161's GATED exactly with V161's settings;
  * the soft gate equals its formula;
  * an open gate reduces to the plain product;
  * the predictive layer is deterministic.
* **Fresh-seed numbers (non-gating):**

  | Arm | Seed 2: unseen / seen / bits at unseen targets | Seed 3: unseen / seen / bits at unseen targets |
  |---|---:|---:|
  | S alone | 0.659 / 1.000 / 1.27 | 0.589 / 0.996 / 1.47 |
  | G alone (fade 0.3) | 0.856 / 0.872 / 4.17 | 0.775 / 0.790 / 4.42 |
  | GATED (soft, a_low 0.25) | 0.774 / 1.000 / 3.22 | 0.771 / 0.996 / 3.54 |
  | GATED-SHUF | 0.648 / 1.000 / 3.23 | 0.605 / 0.996 / 3.19 |

## (b) Text carry-over: *Thursday* → *Edgeworth* (C2 dev)
**Setup:**
* The predictive layer learned on *Thursday*'s 40 KB taught text and was then frozen.
* On *Edgeworth*: 10,287 taught, 5,146 calibration and 10,266 held-out positions.

**Results (bits/symbol):**

| | Calibration | Held-out |
|---|---:|---:|
| S alone (R1 on the KC code, L2 30) | 3.171 | **3.320** |
| G alone (trace readout) | 4.171 | 4.216 |
| G-SHUF alone | 4.395 | 4.484 |
| Best gated combination (β 0.25, τ at the 1st percentile, a_low 0.5) | 3.228 | – (S chosen) |
| Best combination with G-SHUF | 3.213 | – (S chosen) |
| Gate always open, β 0.25 / 0.5 / 1 | 3.233 / 3.374 / 3.806 | – |

**Chosen on calibration:** S alone, for both GATED and GATED-SHUF. The gain is 0.000, so (b) fails.

**Note on G's fit:** the L2 chosen for G was 0.01, the edge of its grid.

## What it shows
1. **On byte-level text, the shared pathway carries little that the memory lacks.** Its predictions do hold real information: G alone is 0.27 bits better than its shuffled control. But every way of adding it to the memory makes calibration worse, and the shuffled version is no worse than the real one. Whatever "shared future" structure a next-receptor-pattern prediction captures at byte level is already available to the KC memory.
2. **The synthetic success does not transfer.** The toy was built so that sharing was the only route to the answer. Byte-level text is not like that at 10 KB of readout data.
3. **By the reading fixed in advance (design §4), the architectural question reopens.** The next design should change the shared pathway itself, not tune the toy. Candidates:
   * a richer target, such as the next KC code or several steps ahead;
   * feeding the shared signal into the KC layer, to form conjunctions with the current input;
   * a coarser level of prediction (word-scale units).

## Accounting
* 0.390 CPU-h over 3 attempts (C0 0.079, C1 0.223, C2 0.087).
* 3,410 learner records, 2 of 3 test cycles, designs tried k = 2.
