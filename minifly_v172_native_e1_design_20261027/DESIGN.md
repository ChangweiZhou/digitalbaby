V172 DESIGN (owner go-ahead; committed before any V172 code)

# V172 design: why the native memory fails taught recall (E1), with a writer × homeostasis factorial (27 October 2026)

**Owner's words, verbatim:**
1. "Can you launch the skill and do some experiments based on above recommendations?"
2. "I meant the previous recursive-self-improvement skill."

The recommendations are in `minifly_next_steps_20261027/NOTE.md` §2, item 2: make native E1 the immediate target.

**Caps:** the defaults, since no number was written: 2 CPU-h, 5,000 learner records, 3 test cycles. No official run, no adoption, no pointer update.

## 0. Question and evidence
**Question:** the native mushroom-body memory of the current arithmetic lineage fails E1. Two candidate causes are tested: the teaching rule (writer) and one-sided KC homeostasis.

**Evidence of the failure.** The lineage stores and recalls few taught facts:

| Source | Writer | Homeostasis | Front | Complete taught recall |
|---|---|---|---|---|
| Codex V154 | N_2B and N_A | off | four fronts | 0–8 of 64 OLD |
| Codex V171 (parent) | N_2B | off | P_SLOW | 3–4 of 64 OLD; 1–5 of 16 NEW |
| CLS round, G1 (Claude, 17 October; 80 single-digit facts, 24 passes) | delta | on | FE-PERM | **84–88%** |

* In the CLS round, the same native bank recalled 63–64% without homeostasis, and only 1–3% with plain, non-error-corrective signed teaching.
* "Lineage native cores on the same facts" recalled 0–8%.

**What differs between them (source read, not inferred):**
* V171's parent is `models152.Native('N_2B')` on `fastmem.FastMemory(homeo=False)`, with the P_SLOW front. P_SLOW is FastMemory's own FE-PERM at 0.95 per byte.
* The two writers:
  * N_2B's coefficient is 0.2·w(n)·center(Cᵀ(p − e_y)/4), with |coef| ≤ 0.2·w(n). It is error-corrective in softmax space, but weak.
  * The CLS delta writer is (tanh((v − v̄)/2) − C_y)/2 per compartment, up to 1.
* Homeostasis is off in the lineage and on in the CLS round.
* The task differs: operands 0–19 with two-digit answers here, single digits in the CLS round.

**Factors not tested, with reasons:**
* **Front end:** V154 already found the fast fade with permutation no better than the slow one for taught recall (P_FAST 4–6 of 64 against P_SLOW 7–8). The front stays at P_SLOW.
* **Class-bias port:** the CLS round reached 84–88% without it.
* **Even fan-out:** it changes the canonical newborn wiring, which the native birth check forbids. It needs a separate decision.

## 1. Fixed components
* **Native bank:** 16 compartments, each a canonical newborn native fly. Each compartment's value is one bit of the fixed 257-class C16 code. Uses `minifly_cls_dev_20261017/fastmem.py`, unchanged, with:
  * front FE-PERM at 0.95 per byte (= Codex's P_SLOW);
  * fresh adaptation at each record start;
  * output bias q = 0.
* **Read:** p = softmax(C(v − v̄)/4).
* **Own generation:** greedy argmax for at most 4 decisions. The answer stops only on the model's own end-of-line (EOS); a fourth non-EOS decision is recorded as CAP. An answer is correct only if the payload and the EOS are both exact.
* **Gradual warmup in every arm** (owner requirement): w(n) = 1/16 + (15/16)·min(n/1024, 1), where n counts all arriving answer and EOS targets.
* **Teaching:** the signed native write at each answer site, after the prediction, as in V154/V171. The arms differ only in the coefficient and in homeostasis.
* **Scope:** this is a diagnostic of native storage with answer-site teaching. Under the owner's principle of 27 October, such objectives are admissible as diagnostics; the every-position objective is a later step. No arithmetic structure, attention or back-propagation.

## 2. Arms (2 × 2)
| Arm | Coefficient for compartment j | Homeostasis |
|---|---|---|
| N2B (the lineage parent) | 0.2·w(n)·center(Cᵀ(p − e_y)/4)_j | off |
| N2B-H | as N2B | on |
| DELTA | w(n)·(tanh((v_j − v̄)/2) − C_yj)/2 | off |
| DELTA-H (CLS full, plus warmup) | as DELTA | on |

Homeostasis uses the CLS parameters (rate 0.05, c 5, β 0.01), updated at teaching events only.

## 3. Data and dose
* **Generator:** V154's `dataset()`, reimplemented under the salt `V172`.
  * The 210 unordered operand orbits of 0–19, each in one random orientation, with prompts `a+b=` and the decimal sum as answer.
  * OLD = the first 64 orbits. NEW = 16 orbits from the next 64, chosen by ascending SHA-256 of `V172|world|NEW|id`.
  * The validation and assessment rows are never read.
* **Worlds:** technical 202610277201; dev 202610277211; rehearsal 202610277221, reserved. None appears in either branch.
* **Dose per history:**
  * OLD: 16 passes over 64 items, using 8 fixed permutations twice, so 1,024 records;
  * then NEW: 8 passes over 16 items, each pass a fixed permutation, so 128 records;
  * total 1,152 records.
* **Panels** (no writes; on clones):
  * OLD after pass 8 and pass 16;
  * all 80 items at the end;
  * all 80 after 128 native rest steps.

## 4. Cycles and rules
**C0 testbed** (no learner records), on the technical and dev worlds' taught items, with an untaught native bank:
* **Code collisions:** the binary KC code at each answer site along the teacher-forced path.
  * Count site pairs with identical codes but different targets. No memory can recall both.
  * Report the distribution of pairwise Jaccard overlap at the end of the prompt.
* **CEILING** (evaluator only, never on the answer path): a ridge least-squares map from the KC code to the target's C16 code, fitted on all taught sites and decoded as at recall. Complete answers along the teacher-forced path. Passes if at least 72 of 80 are correct.
* **FLOOR:** the most frequent taught byte at each answer position, ignoring the prompt. Fails as required if at most 8 of 80 are correct.
* **Continue** if the ceiling passes and the floor fails; otherwise the verdict is CANNOT_TEST_CLAIM_AS_SPECIFIED.

**C1 technical** (technical world):
* N2B equals Codex's `models152.Native('N_2B')` (vendored from commit ec38e8be): native values identical after 3 records.
* DELTA-H with warmup forced to 1 equals `fastmem.FastMemory(homeo=True, teach='delta').train`: values identical after 3 records.
* Evaluation leaves the model unchanged (state digest).
* Training is deterministic.
* The warmup counter counts targets.
* Homeostasis thresholds stay at 0 when it is off and rise when it is on.
* Cost per record is measured.

**C2 dev** (dev world; all four arms at full dose). Primary endpoint: complete taught recall over all 80 items at the end. Readings fixed in advance, paired by item, with SD from an item bootstrap:
* **Writer effect:** mean over the homeostasis levels of (DELTA arm − N2B arm). Passes at ≥ 16 items and ≥ 3 SDs.
* **Homeostasis effect:** mean over the writers of (on − off). Passes at ≥ 8 items and ≥ 3 SDs.
* **Reference reproduced:** N2B at most 16 of 80. If not, the lineage failure is not reproduced under this protocol, and that is reported.
* **CLS level reached:** DELTA-H at least 64 of 80 (descriptive).
* **Reported, not gated:**
  * OLD retention: of the OLD items correct after pass 16, the share still correct at the end;
  * recall after rest;
  * per-target teacher-forced accuracy, separately for payload bytes and EOS;
  * CAP and empty-EOS counts;
  * coefficient magnitudes for each writer.

**C3 rehearsal:** not within the default record cap. If the dev result warrants it, N2B and DELTA-H on the reserved fresh world need 2,304 more records, so the owner must write a cap of at least 7,000 records.

## 5. Readings fixed in advance
* **Writer effect passes:** the lineage's E1 failure is largely its teaching rule. The next native work uses the error-corrective writer with warmup.
* **Homeostasis effect passes:** collisions among frequent KCs limit storage. Homeostasis becomes a default component.
* **Both fail, with the ceiling passing:** the code can carry the taught set, but neither change makes the native writer store it at this dose. Storage is then limited elsewhere, for example by write strength under warmup, by interference across the 80 items, or by the two-digit task. Report and stop.
* **DELTA-H at CLS level:** E1 for taught addition facts is restored in the native lineage. The output-gate reader of V171 then becomes testable on a parent that passes E1.

## 6. Accounting and predictions
| Item | Learner records |
|---|---:|
| C1 checks | about 50 |
| C2: 4 arms × 1,152 | 4,608 |
| **Total** | **about 4,660** (within 5,000) |

**Estimated CPU:** about 0.35 h (0.2 s per record, plus the panels).

**Predictions:**
* the ceiling passes and the floor fails: 0.85;
* the writer effect passes: 0.6;
* the homeostasis effect passes: 0.45;
* N2B at most 16 of 80: 0.85;
* DELTA-H at least 64 of 80: 0.25.
