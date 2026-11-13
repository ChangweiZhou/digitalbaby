# V149 INPUT-CEILING V1: how much context reaches MiniFly's KC code?

25 October 2026. **Design document.** It is refined in three cycles (implement and audit; dev test run; rehearsal and lock) before one locked main run, as V148 was. Every number about earlier work cites its source. Every prediction has three sentences: what is expected, why, and what would weaken it.

## 1. Why this round
* **V148** (`minifly_v148_pretrain_probe_dev_20261025/REPORT.md`): MiniFly predicts raw English text at 7.10 bits/byte held-out, below a unigram counter (4.63).
* **S0 explore** (`minifly_v148_explore_s0_20261025/EXPLORE.md`):
  * The fixed random 16-bit output code alone caps prediction at 7.03 bits/byte with no context, and MiniFly sits on that cap.
  * A graded 16-number code learned from the text would reach the bigram level (3.73).
  * The output is therefore *a* blocker. But because it caps everything, we cannot yet see whether context reaches the memory at all.
* **Beating the counting predictors can only come from contexts that share predictions** through overlap of their KC codes. That is the input side.
* **Hence the question.** If the KC state carries little context, fixing the output only reaches unigram, and the context side must be fixed first. If it carries more than bigram-level context, the output side comes first.

This round runs **no native learning**. It reads the native front-end and KC state of an untaught memory, and measures with readouts placed outside MiniFly how well that state predicts the next byte. Those readouts are labelled ceiling learners and are never part of MiniFly's answer path.

## 2. What is read (the states)
The memory is `FastMemory()` with the V148 settings (`minifly_cls_dev_20261017/fastmem.py`). No record is ever taught.
* Each text line is one record, as in V148.
* At every position the state is taken exactly where V148's probe reads it before predicting the next symbol (`probe.py` `line_logits` → `values()`, `fastmem.py:194-205`):
  1. the front end is advanced one step;
  2. the binary per-side top-k KC code `x` is computed (`encode`, `fastmem.py:132-141`);
  3. the observed KC activity `dx = observed_activity(m, x)` is computed after the compartment's adaptation; this is what the native stores read.
* Then the byte is fed passively, as in V148.

| Arm | State | Purpose |
|---|---|---|
| **PERM-obs** (primary) | observed KC activity `dx`, FE-PERM front end | what MiniFly's stores actually receive |
| PERM-x | binary KC code `x`, FE-PERM | the code before adaptation |
| FE0-x | binary KC code, FE0 front end (no permutation, decays 0.81 per byte; `fastmem.py:9-10`) | how much the permutation (order keeping) adds |
| PERM-last | binary KC code of a fresh FE-PERM fed only the last byte | context depth 1, for comparison with the full trace |

**Refinement after reading the code.** Design 9a in V148 proposed the raw trace in place of the max-normalised one. Here, with an untaught memory, the homeostatic thresholds are zero and `encode` keeps the top k KCs by rank. Dividing the trace by its maximum cannot change which KCs win, so the raw-trace arm would be identical to PERM-x and is not run. Cycle 1 checks this on real states, rather than assuming it.

## 3. Readouts (evaluator-only, outside MiniFly)
1. **R1, ceiling.** Multinomial logistic regression on the state, plus a bias. L2 strength is chosen on the calibration block from a fixed grid.
2. **R2, KC vote.** Every active KC keeps counts of the next symbol in the taught text. The prediction is the activity-weighted mean of the active KCs' next-symbol distributions, mixed with the unigram. The mixing weight is chosen on calibration. This is the kind of statistic a local store could hold.

**Shared conventions:**
* Both readouts predict over the symbols seen in the taught text, plus one OTHER class. OTHER's probability is spread evenly over the unseen symbols, so every symbol has non-zero probability.
* The primary measure is held-out bits per symbol: 256 bytes plus end-of-line, the same alphabet as the V148 ladder. Top-1 accuracy is reported as well.

## 4. Readout checks (cycle 0 of the kit: ceiling must pass, floor must fail)
R1 is fitted on three evaluator-side codes on the same split:

| Code | Check |
|---|---|
| one-hot of the last symbol, with a line-start flag | must land within 0.15 bits of the bigram rung |
| one-hot of the last two symbols, plus the one-hot of the last symbol | must land within 0.20 bits of PPM-2 |
| constant code (bias only) | must not beat unigram by more than 0.02 bits |

If a check fails, the readout cannot measure what this round asks. The verdict is then CANNOT_TEST_AS_SPECIFIED and no arm is interpreted.

## 5. Data and split
* **Corpus:** NLTK's copy of the Gutenberg texts (archive SHA-256 `2d3c3ab5…d611`). The record and split rules are the same as V148's (`minifly_v148_pretrain_probe_dev_20261025/corpus.py`):
  * lines over 128 bytes are cut;
  * blank lines are skipped;
  * train pool: the first 40 KB of records;
  * calibration: the next 5 KB;
  * held-out: the last 10 KB.
* **Texts:**
  * Cycle 2 (dev): *Buster Bear*.
  * Cycle 3 (rehearsal): Bryant's *Stories to Tell to Children*. It has not been used to tune this round.
  * Main: *The Wisdom of Father Brown* (G. K. Chesterton, 1914; `chesterton-brown.txt`), new to this project, pinned by SHA-256 in cycle 1. (Correction made in cycle 1, before any run: the first version of this design named *The Innocence of Father Brown*, but NLTK's `chesterton-brown.txt` is *The Wisdom of Father Brown*. The file and its role are unchanged.) *Alice* is not used; it stays reserved for comparing later components on V148's split (V148 design section 9).
* **Counting ladder:** the V148 `ladder.py` rungs on the same split (uniform, unigram, bigram, PPM-C orders 2–5).

## 6. Decision rule (fixed now; scored once on the main text)
Let C = R1 held-out bits/byte on **PERM-obs**, and B = bigram held-out bits/byte, both at 40 KB.
* **C ≥ B − 0.10 → CONTEXT_FIRST.** Within 0.10 bits of bigram, or worse: the KC state carries at most last-byte-level context, so the context side is fixed first.
* **C < B − 0.10 → OUTPUT_FIRST.** The state carries more than bigram-level context, so the output side is fixed first.
* The margin of 0.10 bits is about three times the sampling error of a 10 KB held-out mean; cycle 2 measures the actual spread.

## 7. Predictions (three sentences each; written before any code)
1. **Primary level.**
   * *Expect:* C lies between unigram and B − 0.10, i.e. CONTEXT_FIRST.
   * *Because:* F-UF-11 found that the front-end code barely encodes byte order. F-UF-13 found that an external readout of KC codes generalised to only 5–9% of held-out bytes (older v4 substrate). The FE-PERM trace sums decaying byte patterns into 88 channels.
   * *Weakened if:* C beats bigram by more than 0.10 bits.
2. **History beyond the last byte.**
   * *Expect:* R1 on PERM-last lands within 0.15 bits of bigram, and R1 on PERM-x is not better than PERM-last by more than 0.10 bits.
   * *Because:* a single byte's code identifies that byte (F-UF-11: clean byte queries 12/12), while the decaying trace mixes it with earlier bytes.
   * *Weakened if:* PERM-x beats PERM-last by more than 0.10 bits, which would mean the trace adds usable history.
3. **A local readout.**
   * *Expect:* R2 is at least 0.30 bits worse than R1 on PERM-obs.
   * *Because:* averaging single-KC next-symbol distributions cannot express conjunctions of KCs.
   * *Weakened if:* R2 comes within 0.30 bits of R1.

## 8. Three cycles before the lock
1. **Implement and audit** (dev text: Buster, 1–2 KB). Build:
   * the corpus extension, including the new text pin;
   * state extraction (`states.py`);
   * the readouts and readout checks (`readouts.py`);
   * the runner (`run.py`).

   Unit checks:
   * states are identical to what `values()` computes (`x` and `dx` on sample positions);
   * a single-compartment bank gives the same states as the 16-compartment bank;
   * a record's states do not depend on records read before it;
   * max-normalised and raw traces give identical codes when the thresholds are zero;
   * readout probabilities sum to 1;
   * the R1 and R2 toy values match hand computation;
   * calibration and held-out never enter readout fitting;
   * the runtime closure can be pinned.

   Also measure cost.
2. **Dev test run** on Buster: the full measurement, plus the spread of held-out bits under bootstrap over lines. Fix the L2 grid, the R2 grid and the iteration limits here; nothing is tuned later.
3. **Rehearsal and lock.** The exact protocol on Bryant, then `LOCK.json`:
   * runtime closure from `sys.modules`;
   * data and native asset hashes;
   * settings;
   * the decision rule and the prediction scoring.

   The main run on *Father Brown* then runs once.

**Reporting rules (as in V148):**
* Failed attempts are counted.
* Dev and rehearsal numbers are development results.
* The report states measured values and ladder positions, and claims nothing beyond them.

## 9. Cost and scope
* Estimated well under 2 CPU-hours in total, with no native learning. Cycle 1 measures it.
* Not in scope:
  * any change to MiniFly;
  * the output-side fix;
  * choosing among context-side mechanisms (temporal context model, Oja plasticity, biased sampling, KC clock).

  Each of those is a later, separate step, chosen according to the decision rule.
