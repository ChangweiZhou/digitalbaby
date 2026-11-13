V172 NATIVE-E1 V1: writer effect PASS (the error-corrective delta writer raises complete taught recall from 10/80 to 29–39/80); homeostasis effect FAIL (+6 items, below the 8-item threshold); the lineage failure is reproduced (N2B 10/80); CLS level not reached at the end (DELTA-H 39/80), because learning NEW erased about half of the recalled OLD items. DEV_NOT_EVIDENCE.

# V172 report: why the native memory fails taught recall (27 October 2026)

**Question:** the native memory of the current arithmetic lineage (Codex V154/V171) recalls 3–8 of 64 taught facts. The CLS-round configuration of the same native bank recalled 84–88%. Which of the two configuration differences explains the gap: the writer, or one-sided KC homeostasis?

**Answer: mainly the writer.**
* N2B, the lineage's teaching rule, reproduces the failure: 10/80 complete taught answers.
* Replacing it with the CLS error-corrective delta writer, keeping the owner's gradual warmup, gives 29/80 without homeostasis and 39/80 with it.
* Homeostasis adds 6 items on average: +10 with the delta writer, +2 with N2B. That is below the pre-stated threshold.
* With the delta writer, the memory acquires the taught facts: 50 of 64 OLD items at pass 16, against 75% at pass 16 in the CLS round. **The remaining loss is interference.** 128 NEW records erase 27 of those 50.

**Files:**
* Design: `minifly_v172_native_e1_design_20261027/DESIGN.md`, committed before any code (5cc3927a).
* Grant: `AUTHORIZATION.json` (default caps).
* Ledgers in `runlog/`: 3 attempts, none failed. Cycles: testbed *continue*, technical *continue*, dev *stop*. No amendments.
* Vendored reference: Codex's `models152.py` and `support152.py` from commit ec38e8be.

## Setup
* **Native bank:** `fastmem.FastMemory`, unchanged. FE-PERM at 0.95 per byte (= Codex's P_SLOW), fresh adaptation, C16 port, q = 0.
* **Warmup:** w(n) = 1/16 + (15/16)·min(n/1024, 1) in every arm.
* **Task:** addition `a+b=` with operands 0–19 (V154's generator, salt V172), on the fresh dev world 202610277211.
  * OLD: 64 orbits, 16 passes.
  * NEW: 16 orbits, 8 passes.
  * 1,152 records per arm.
* **Scoring:** complete own answers, meaning payload plus own EOS, within 4 decisions.

| Arm | Writer coefficient | Homeostasis |
|---|---|---|
| N2B (Codex's parent) | 0.2·w·center(Cᵀ(p − e_y)/4) | off |
| N2B-H | same | on |
| DELTA | w·(tanh((v − v̄)/2) − C_y)/2 | off |
| DELTA-H | same | on |

## C0 testbed (no learner records)
On an untaught bank, the 80 taught items of the technical and dev worlds were screened:

| | Technical world | Dev world |
|---|---:|---:|
| Answer sites; distinct KC codes | 228; 228 | 232; 232 |
| Codes shared by sites with different targets | 0 | 0 |
| Prompt Jaccard overlap: mean / 95th percentile / max | 0.33 / 0.59 / 0.87 | 0.34 / 0.60 / 0.91 |
| Prompt pairs above 0.8 overlap | 11 of 3,160 | 11 of 3,160 |
| **CEILING** (evaluator ridge readout of the KC code) | **80/80** | **80/80** |
| **FLOOR** (most frequent byte per answer position) | 0/80 | 2/80 |

The code can carry the whole taught set, so a native failure is a failure of storage.

## C1 technical: 7 of 7 checks pass (21 records)
* N2B reproduces Codex's `models152.Native('N_2B')` exactly: values differ by 0.0 after 3 records.
* DELTA-H with warmup forced to 1 reproduces the CLS `FastMemory(homeo=True, teach='delta')` exactly.
* A panel leaves the model unchanged, and training is deterministic.
* The warmup counter counts answer and EOS targets.
* Homeostasis behaves as specified.
* Cost: 0.19 CPU-s per record.

## C2 dev (dev world; 4,608 records)
**Complete taught answers:**

| Arm | OLD after pass 8 (/64) | OLD after pass 16 (/64) | Final OLD (/64) | Final NEW (/16) | **Final (/80)** | After rest (/80) |
|---|---:|---:|---:|---:|---:|---:|
| N2B | 5 | 7 | 7 | 3 | **10** | 10 |
| N2B-H | 6 | 12 | 8 | 4 | **12** | 12 |
| DELTA | 26 | 37 | 17 | 12 | **29** | 29 |
| DELTA-H | 33 | **50** | 24 | 15 | **39** | 40 |

**Pre-registered readings** (paired by item, item-bootstrap SD):

| Reading | Value | Rule | Result |
|---|---:|---|---|
| Writer effect: mean over homeostasis of (DELTA − N2B) | +23.0 items (SD 4.1) | ≥ 16 and ≥ 3 SD | **PASS** |
| Homeostasis effect: mean over writers of (on − off) | +6.0 items (SD 2.0) | ≥ 8 and ≥ 3 SD | **FAIL** |
| Interaction (not gated) | +8.0 (SD 4.1) | – | homeostasis helps DELTA, not N2B |
| N2B ≤ 16/80 (reference reproduced) | 10 | – | **yes** |
| DELTA-H ≥ 64/80 (CLS level) | 39 | – | **no** |

**Retention of OLD items correct after pass 16, through NEW learning:**

| Arm | Retained |
|---|---|
| N2B | 3 of 7 |
| N2B-H | 6 of 12 |
| DELTA | 15 of 37 |
| DELTA-H | 23 of 50 |

**Diagnostics (not gated):**
* **Write strength.** The mean per-site maximum |coefficient| at full warmup is about 0.05 for N2B and about 0.4 for DELTA, about 8× weaker. The design cannot separate magnitude from form.
* **Training accuracy.** Immediate site accuracy during training at OLD pass 16 is 0.59 (N2B) and 0.87 (DELTA-H). During NEW pass 8 it is 0.59 and 0.96.
* **Teacher-forced bits per target.** At OLD pass 16: N2B 3.66, DELTA-H 1.44. At the end: 3.57 and 1.89.
* **Where the errors are.** EOS is predicted correctly on 97–100% of sites in every arm. Payload accuracy is what differs: 0.41 (N2B) against 0.68 (DELTA-H) at the end, and 0.89 for DELTA-H after OLD pass 16. There is no truncation (CAP 0); empty answers occur 0–4 times.
* **Homeostasis** raised thresholds on 650 of 5,177 KCs.
* **Rest:** 128 native rest steps change no more than one answer in any arm.

## What it shows
1. **The lineage's E1 failure is mainly its teaching rule.**
   * On a code that can carry all 80 items, the N2B writer stores few of them.
   * The CLS delta writer, with the owner's warmup, acquires 37–50 of 64 OLD items in 16 passes.
   * The likely cause is write strength: N2B's coefficient is about 8× smaller. Its form is not separately tested.
   * **Consequence for Codex's line:** V171's output-gate reader was trained on a parent whose writer stores little. The parent's writer, not the reader, was the first-order limit.
2. **Homeostasis is a secondary aid.** It helps the strong writer (+13 OLD at pass 16, +10 at the end) but not the weak one. Its pre-stated effect fails.
3. **The new binding problem is interference from later learning.**
   * Eight passes over 16 NEW items erase about half of the OLD items recalled at pass 16, in both delta arms. NEW items themselves are learned well (12–15 of 16).
   * Rest causes no loss, so the loss is interference, not decay. That matches the CLS-round diagnosis.
   * The prompts share bytes ('+', '=', digits) and overlap by 0.33 on average. Corrective writes for NEW items therefore move values on KCs that OLD items also use.
4. **E1 for taught recall is partly restored, not reached.** DELTA-H ends at 39/80, against the CLS round's 84–88%. The CLS round used 24 passes without warmup on single-digit facts, and its G1 had no later NEW phase.

## Limits
* **Scale:** one dev world, one native birth per arm. DEV_NOT_EVIDENCE. The rehearsal on the reserved fresh world (202610277221) was not run; it needs about 2,304 records beyond the default cap.
* **Confounds:** the writer contrast confounds coefficient magnitude with coefficient form. The dose is fixed: OLD 16 passes, NEW 8. Neither curve had saturated (DELTA-H OLD went from 33 to 50 between passes 8 and 16).
* **Scope:** answer-site teaching on taught items only. No held assessment was read; nothing here bears on E3.

## Next steps suggested by the result (not launched)
1. **Rehearsal of N2B against DELTA-H** on the fresh world, to confirm the writer effect. Needs a cap of at least 7,000 records.
2. **Interference:** the same DELTA-H memory with NEW interleaved with OLD replay, against blocked NEW (this round), at matched NEW exposure. This is the generic remedy that protected old skill in the CLS round (G4).
3. **Codex's line:** adopt the delta writer, with warmup, as the native parent before any further reader or controller test. This is a pointer and adoption decision for the owner.

## Predictions, scored
| Prediction | p | Outcome |
|---|---:|---|
| Ceiling passes and floor fails | 0.85 | yes |
| Writer effect passes | 0.6 | yes |
| Homeostasis effect passes | 0.45 | no |
| N2B ≤ 16/80 | 0.85 | yes |
| DELTA-H ≥ 64/80 | 0.25 | no |

## Accounting
* **CPU:** 0.238 CPU-h (testbed 0.004, technical 0.009, dev 0.225).
* **Records:** 4,629 learner records (C1 21, C2 4,608).
* **Cycles:** 2 of 3 test cycles; designs tried k = 3.
* Default caps; no official run, no adoption, no pointer update.
