V174 DESIGN (owner go-ahead; committed before any V174 code)

# V174 design: does replay of observed records, or KC homeostasis, remove the write-caused forgetting of the native memory? (28 October 2026)

**Owner's words, verbatim:** "Activate the recursive improvement skill. We can do both A and B here." Option B of `minifly_next_steps_20261028/NOTE.md` §3. A runs as V173.

**Caps:** the defaults, since no number was written: 2 CPU-h, 5,000 learner records, 3 test cycles. No official run, no adoption, no pointer update. The note estimated 6,000–8,000 records for B; this design is the form that fits 5,000.

## 0. Question and evidence
**Question:** at the write strength that acquires taught facts (V172's DELTA, gain 1), learning NEW erases OLD. Do either of two generic mechanisms make NEW learning a net gain?
1. Interleaved replay of genuinely observed records.
2. One-sided KC homeostasis.

**Evidence:**
* **V172 (one world, arithmetic):** DELTA acquired 37 of 64 OLD items and kept 15 of them through 128 NEW records. DELTA-H acquired 50 and kept 23.
* **Codex WCFI-V1 (two worlds, opaque associations):** at high write strength, NEW 14–16 of 16 was learned while OLD fell from 50 to 25–35. A write-inhibited observation control equals an exact-time rest control, so the loss is caused by writing. About one OLD item is lost per NEW item gained.
* **CLS round (G4):** replay into both systems protected an old skill: 70–85% kept, against 4–6% without replay.

## 1. Fixed components: V172, unchanged
* **Native memory:** `fastmem.FastMemory`; FE-PERM at 0.95 per byte; fresh adaptation; C16 port; q = 0.
* **Writer:** DELTA, coefficient w(n)·(tanh((v − v̄)/2) − C_y)/2.
* **Warmup:** w(n) = 1/16 + (15/16)·min(n/1024, 1). n counts every arriving answer and EOS target, including write-inhibited ones (the WCFI convention).
* **Task:** V172's addition generator, `a+b=` with operands 0–19, under salt V174.
* **Scoring:** complete own answers, payload plus own EOS within 4 decisions.

## 2. Design: one common OLD history per condition, then NEW-phase forks
**OLD phase:** 16 passes over 64 OLD items (8 fixed orders, twice), 1,024 records. There is one history per homeostasis level, off (H0) and on (H1).

**NEW phase, forked from the OLD checkpoint.** 8 passes. Each pass presents the 16 NEW items in a fixed order, and **each NEW record is followed by one slot**, so 256 records per fork. The slot record is drawn uniformly, with a fixed seed, from all record presentations before it in that history. That draw:
* is label-free: no OLD/NEW tag, item ID or phase is used;
* in practice mostly selects OLD records, because they dominate the history.

The two forks differ only in what happens in the slot:
* **REPLAY:** the slot record is taught normally, with writes enabled.
* **BLOCKED:** the same slot record is presented as a write-inhibited observation. Same prompt events, prediction, clock, target arrival and warmup counter; signed teach with `write=False`.

BLOCKED and REPLAY therefore match NEW exposure, slot records, events, clock and target arrivals. They differ only in whether replayed records are written. Homeostasis updates, where on, run identically in both.

**Worlds:**
* technical 202610287401;
* dev W1 202610287411 and W2 202610287412;
* rehearsal 202610287421, reserved.

All are unused in either branch.

**Conditions:**
* **W1:** H0 and H1 OLD histories, each with BLOCKED and REPLAY forks: 2 × 1,024 + 4 × 256 = 3,072 records.
* **W2:** H0 only, with BLOCKED and REPLAY forks: 1,024 + 2 × 256 = 1,536 records. This replicates the replay effect.

**Panels** (on clones; no writes):
* OLD after pass 16 (the common pre-state);
* all 80 items at the end of each fork.

## 3. Cycles and rules
**C0 testbed** (no learner records):
* V172's evaluator screen on the two dev worlds' taught items, with an untaught bank: the ridge ceiling must reach ≥ 72 of 80, and the position-majority floor must be ≤ 8 of 80.
* A label-free check of the slot draw: it uses only the presentation history and a fixed seed.

**C1 technical** (technical world):
* The H0 learner equals V172's DELTA (values identical after 3 records).
* A write-inhibited slot leaves every compartment's fast and slow values unchanged, while advancing the clock and the warmup counter exactly as a written slot does.
* BLOCKED and REPLAY slot draws are identical.
* Panels leave the model unchanged.
* Training is deterministic.
* Cost per record is measured.

**C2 dev:** all conditions above. The primary endpoint is complete taught recall over all 80 items at the end of each fork. Readings fixed in advance, paired by item, with SD from an item bootstrap:
* **R1, replay effect:** REPLAY − BLOCKED on final recall over all 80. In W1 it is the mean over H0 and H1; in W2, H0 only.
  * Passes if it is ≥ 8 items and ≥ 3 SD in W1, and positive in W2.
* **R2, replay makes NEW learning a net gain.**
  * Net gain = final recall over 80 − OLD recall at the pre-state. The pre-state stands in for no NEW writes: WCFI found the write-inhibited and rest controls equal to the pre-state within 1 item.
  * Passes if REPLAY's net gain is ≥ +8 items in W1 (H0 and H1 mean) and positive in W2, while BLOCKED's net gain is ≤ 0. If BLOCKED's net gain is positive, it is reported, and R2 then requires REPLAY − BLOCKED net gain ≥ 8 items.
* **R3, homeostasis on retention (W1):** H1 − H0 in retained OLD items (correct at the pre-state and at the end), under BLOCKED. Passes at ≥ 6 items.
* **Reported, not gated:**
  * OLD and NEW counts separately;
  * the retention of initially correct OLD items;
  * per-target teacher-forced accuracy for payload and EOS;
  * which records the slots drew (OLD and NEW counts, unique items).

**C3 rehearsal:** not within the default caps. The reserved world needs about 1,536 more records.

## 4. Readings fixed in advance
* **R1 and R2 pass:** replay of genuinely observed records turns NEW learning into a net gain at acquisition-level write strength. The CLS mechanism transfers to the native lineage's arithmetic task. The next step is a budget-matched comparison against Codex's local protection candidate.
* **R1 passes, R2 fails:** replay reduces forgetting, but not enough to make NEW learning a net gain at this ratio of one replay per NEW record.
* **R1 fails:** replay at this ratio does not counter the interference. Interference is then likely structural: shared KCs among prompts that share bytes. Remedies would have to change the code, for example by sparser or decorrelated keys, not the schedule.
* **R3:** homeostasis is or is not a retention aid, separately from its acquisition benefit in V172.

## 5. Accounting and predictions
| Item | Learner records |
|---|---:|
| C1 | about 40 |
| W1 | 3,072 |
| W2 | 1,536 |
| **Total** | **about 4,650** (within 5,000) |

**Estimated CPU:** about 0.3 h (0.2 s per record, plus the panels).

**Predictions:**
* R1 passes: 0.6;
* R2 passes: 0.4;
* R3 passes: 0.2;
* the C0 ceiling passes: 0.9.
