V174 REPLAY V1: R1 PASS (replaying observed records during NEW learning keeps 15–17 more of 80 taught items than observing them without writing); R2 FAIL (replay roughly halts the loss, net +2 to +7 items, but does not make NEW learning a net gain of ≥ 8); R3 FAIL (homeostasis does not improve retention). DEV_NOT_EVIDENCE.

# V174 report: replay against write-caused forgetting in the native memory (28 October 2026)

**Question:** at the write strength that acquires taught facts (V172's DELTA), learning NEW erases OLD; Codex's WCFI-V1 showed the loss is caused by the writes themselves. Do either of two generic mechanisms make NEW learning a net gain?
1. Interleaved replay of genuinely observed records.
2. One-sided KC homeostasis.

**Answer:**
* **Replay largely stops the forgetting, in both worlds.** OLD items still correct at the end, out of those correct before NEW learning: 32/41, 35/47 and 21/33 with replay, against 19, 19 and 7 without. NEW items are still learned: 9–12 of 16 with replay, against 8–14 without.
* **Replay does not yet make NEW learning a clear net gain.** The net change over all 80 items is +2, +2 and +7 with replay, against −13, −13 and −10 without.
* **Homeostasis adds nothing to retention:** 19 against 19 OLD items kept.

**Files:**
* Design: `minifly_v174_replay_design_20261028/DESIGN.md`, committed before code (6b7e0898).
* Grant: `AUTHORIZATION.json` (default caps).
* Ledgers in `runlog/`:
  * 5 attempts, one failed;
  * cycles: testbed *revise*, testbed *continue*, technical *continue*, dev *stop*;
  * amendments 1 and 2, both measurement defects in my own checks, both before any C2 outcome.

## Design in brief
* **Memory and task:** V172's native memory (DELTA writer with the owner's warmup), on addition `a+b=` with operands 0–19 (salt V174).
* **OLD phase:** 16 passes over 64 items, 1,024 records. There is one history per homeostasis level.
* **NEW phase,** forked from that checkpoint: 8 passes over 16 NEW items. Each NEW record is followed by one slot record, drawn uniformly (fixed seed) from all earlier presentations: 256 records per fork. The draw is label-free; in practice 117–119 of the 128 slots drew OLD records, covering 58–63 distinct items.
* **REPLAY** writes the slot record. **BLOCKED** presents the same record write-inhibited: same events, prediction, clock, homeostasis update and warmup counter, with the signed teach at write=False.
* So the forks differ only in whether replayed records are written.
* **Worlds:** W1 202610287411 (homeostasis off, H0, and on, H1); W2 202610287412 (H0 only).

## C0 and C1
* **C0 testbed:** the evaluator ceiling recalls 80/80 in both worlds, and the floor scores 0/80 and 4/80. There are no code collisions with conflicting targets. testbed-001 was flagged only because my label-free check was a text search defeated by the word "phase" in a docstring (amendment 1). testbed-002 replaced it with a structural check.
* **C1 technical (5 of 5 after amendment 2):**
  * H0 equals V172's DELTA.
  * A write-inhibited record equals the same record run as non-teaching events, and differs from a written one, with equal clock and counter.
  * The first technical attempt assumed that inhibited records leave the native stores unchanged. They do not: the native fly decays its fast and slow stores in every event.
  * Slot draws are identical and content-free in both forks.
  * Panels leave the model unchanged, and training is deterministic.

## C2 dev (4,608 records)
| History | OLD correct before NEW (/64) | Fork | Final OLD (/64) | Final NEW (/16) | **Final (/80)** | OLD kept | **Net change** |
|---|---:|---|---:|---:|---:|---:|---:|
| W1, H0 | 41 | BLOCKED | 20 | 8 | 28 | 19/41 | **−13** |
| | | REPLAY | 34 | 9 | 43 | 32/41 | **+2** |
| W1, H1 | 47 | BLOCKED | 20 | 14 | 34 | 19/47 | **−13** |
| | | REPLAY | 38 | 11 | 49 | 35/47 | **+2** |
| W2, H0 | 33 | BLOCKED | 10 | 13 | 23 | 7/33 | **−10** |
| | | REPLAY | 28 | 12 | 40 | 21/33 | **+7** |

**Pre-registered readings:**

| Reading | Value | Rule | Result |
|---|---|---|---|
| R1: REPLAY − BLOCKED, final over 80 | W1 +15.0 (SD 3.3, mean of H0 and H1); W2 +17 (SD 4.2) | W1 ≥ 8 and ≥ 3 SD; W2 > 0 | **PASS** |
| R2: net change with REPLAY | W1 +2 (both levels); W2 +7; BLOCKED −13, −13, −10 | REPLAY ≥ +8 in W1 and > 0 in W2, with BLOCKED ≤ 0 | **FAIL** |
| R3: homeostasis, OLD kept under BLOCKED (W1) | 19 against 19 | ≥ 6 | **FAIL** |

**Diagnostics:**
* **Answer digits.** Final teacher-forced payload accuracy is 0.70–0.79 with replay and 0.54–0.63 without. EOS is correct at every site, and there are no truncations or empty answers.
* **Homeostasis** raised thresholds on 558 KCs and gave 6 more OLD items before the NEW phase (47 against 41). That matches V172's acquisition benefit, but it kept no more OLD items afterwards.

## What it shows
1. **Replay of genuinely observed records is an effective generic protection.**
   * In both worlds it keeps most OLD items that would otherwise be erased: 32–35 of 41–47 in W1 and 21 of 33 in W2, against 19 and 7 without.
   * NEW learning is barely affected.
   * The control isolates the replay writes themselves: the BLOCKED fork had the same events, clock, slot records and target arrivals.
   * This reproduces in the native lineage's arithmetic task the CLS round's finding (G4).
2. **It roughly stops the forgetting, but does not yet make NEW learning a net gain.**
   * Each replayed write also costs some OLD items, and one replay per NEW record barely balances the interference.
   * Whether more replay, a different replay ratio, or replay combined with a local protection rule turns the balance positive is untested.
3. **Homeostasis is an acquisition aid, not a retention aid.** It helps storage (+6 here, +13 in V172) but leaves the forgetting caused by later writes unchanged.

## Limits
* **Scale:** two dev worlds, one native birth per history. DEV_NOT_EVIDENCE. The rehearsal on the reserved world 202610287421 was not run; it needs records beyond the default cap.
* **Unequal writes:** REPLAY performs 128 more writes than BLOCKED. That is intended, since it is the treatment, but it means the comparison is not matched on total write count. A matched-write comparison against a local protection rule (Codex's announced candidate) is the next contrast.
* **Fixed settings:** one replay ratio (1:1), the uniform draw over the presentation history, and a fixed dose (8 NEW passes).

## Predictions, scored
| Prediction | p | Outcome |
|---|---:|---|
| R1 passes | 0.6 | yes |
| R2 passes | 0.4 | no |
| R3 passes | 0.2 | no |
| C0 ceiling passes | 0.9 | yes |

## Accounting
* **CPU:** about 0.24 CPU-h (testbed 0.009, technical 0.013, dev 0.214).
* **Records:** 4,759 learner records (technical 151, dev 4,608).
* **Cycles:** 2 of 3 test cycles. Default caps; no official run, no adoption, no pointer update.
