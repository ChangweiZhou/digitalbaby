V153 WHERE-THE-BITS-GO V1: DEV_NOT_EVIDENCE (test cycles complete; official run not granted)

# V153 report after the three test cycles (26 October 2026)

**Status:** C0 testbed, C1 technical, C2 dev and C3 rehearsal are complete. The main run on *Persuasion* has **not** been run; it needs the owner's grant and a cross-agent review. Every number below is a development result.
* **Design:** `minifly_v153_bits_design_v1_20261025/DESIGN.md`.
* **Grants:** `AUTHORIZATION.json`, with three owner grants recorded verbatim.
* **Ledgers:** `runlog/ATTEMPTS.jsonl` (7 attempts) and `runlog/CYCLES.jsonl` (testbed plus 3 test cycles; 2 designs tried).

## Rehearsal (fresh text: *The Man Who Was Thursday*, 40 KB taught / 5 KB calibration / 10 KB held-out)
Counting ladder: unigram 4.445, bigram 3.538, PPM-2 2.986, PPM-3 2.667.

**Block A: the front-end factorial** (best linear readout R1 of the binary KC code; held-out bits/symbol)

| Fade per byte | Scrambling on | Scrambling off |
|---|---:|---:|
| 0.807 (fast) | **2.963** | 3.321 (FE0) |
| 0.95 (slow) | 3.833 (V148's FE-PERM) | 4.113 |

* The fast fade is better by 0.79–0.87.
* Scrambling is *also* better, by 0.28–0.36. Line-bootstrap SDs are 0.014–0.019.
* With scrambling on and the fast fade, the frozen, untaught KC state carries PPM-2-level context: 2.963 against 2.986.

**Block B: the readout chain on FE0's observed KC state** (what the native stores receive)

| Arm | Held-out bits/symbol |
|---|---:|
| R1, batch linear readout | 3.271 |
| ONLINE, local delta rule, after 1 / 3 passes | 3.537 / 3.424 |
| C16, best 16-value native port | 5.653 |
| C16Q, the port plus a class bias | 3.528 |
| NATIVE, unchanged MiniFly, one pass | 6.436 |
| R2, KC vote | 4.102 |

**Descriptive decomposition** (line-bootstrap SD in brackets):
* online-learning cost 0.153 (0.008);
* port cost 2.382 (0.021), of which a class bias recovers 2.125;
* native-structure cost (NATIVE − C16) 0.783 (0.024).

**Predictions** (scored by `summarize.py` with the design's rules):

| Prediction | Result |
|---|---|
| P1: the fade dominates the scrambling | **Weakened.** Scrambling helps 0.28–0.36, in the opposite direction to my expectation that it hardly matters |
| P2: the online rule nearly reaches the ceiling | **Partly.** The gap after 3 passes is 0.153, just above 0.15; after 1 pass 0.266 |
| P3: the port is a large cost | **As predicted:** 2.38 |
| P4: native learning adds cost beyond the port | **As predicted:** 0.78 (on dev, Buster at 10 KB, it was 0.41: weakened) |

**Decision rule:** the next round targets **the port**, the largest cost; the best front-end cell is scrambling on with the fast fade.

## What this shows (development, not evidence)
1. **Context reaches MiniFly's code when the trace fades fast and keeps order.** Each of the two front-end factors matters on its own. The best cell reaches PPM-2 level with no learning in the code at all. On the dev text, at 10 KB, it was 3.178 against PPM-2 3.129.
2. **The 16-bit port's cost is mostly a missing frequency term.**
   * With equal-weight random code words, the port cannot express how often symbols occur.
   * Adding a 257-class bias brings it to 0.26 bits from the batch ceiling.
   * The remaining port limit is small compared with the 0.78-bit native-structure cost.
3. **A local online delta rule gets within about 0.15 bits of the batch ceiling in three passes.** Learning a readout locally is not the bottleneck at this data size.
4. **MiniFly's own learning adds 0.78 bits beyond the best 16-value port.** The candidate causes are per-bit teaching, a single pass, fast/slow decay and interference. This round does not separate them.

## Accounting
* **CPU** (launch-measured child CPU; owner caps 2 → 2.2 → 3 CPU-h): 1.97 CPU-h in total.
  * testbed 0.015;
  * technical 0.18;
  * dev 0.41;
  * rehearsal 1.36.
* **Learner records:** 4,660 of 5,000.
* **Test cycles:** 3 of 3.
* **The rehearsal was suspended once** (SIGSTOP), because a `ps` CPU reading projected that the cap would be exceeded. The owner raised the cap and it was resumed. The launch measurement (1.36 CPU-h for the rehearsal) and that `ps` reading disagree; this is unresolved. The ledger counts the launch measurement.
* **Failed attempts:** none.
  * Two technical checks first failed because of defects in the checks themselves: a one-byte test line, and an exact comparison of a 2.2e-16 rounding difference. They were corrected and the checks re-run within the same cycle.
  * One dev smoke attempt (1 KB) verified the per-arm grid code.
* **Before any run:** the C0 split was moved to full size (design §4), and the before-top-k reading was deferred for cost.

## Next
* **Official main on *Persuasion*.** It needs the owner's grant (stage official) and a cross-agent review by Codex (T2).
* **Next round** (per the decision rule; to be designed):
  * a native readout with a class bias, plus the scrambled fast-fade front end;
  * then separate the 0.78-bit native-structure cost.

## Owner decision (26 October 2026)
* The recommendation shown was to skip the V153 main run and carry its baselines into the next round's single official run.
* Owner's reply, verbatim: "Okay, let us push it."
* The V153 main run is therefore not run. Stage official stays not granted.
* The next round, a native readout with a class bias on the scrambled fast-fade front end, includes the V153 baselines.
