# V153 WHERE-THE-BITS-GO V1: front-end timescale and the readout chain on frozen native states

25 October 2026. **Design, written before any V153 code or outcome.** This round follows the research-loop skill (`.claude/skills/minifly-research-loop` at origin/claude/vibrant-cerf-sje726 66925b04).

**Stages:**
* S0 explore: already done, see the evidence below.
* S1 design, with authorization and a lock committed before any result.
* C0 testbed screen.
* C1 technical, C2 dev and C3 rehearsal.
* S6 official only if Mr. Zhou grants it, after a cross-agent review.

**The kit's schema-v1 gates are not run.** Adopting the kit (UD-0001) is still the owner's open decision, and its checker is shaped around arithmetic (`KB/RETRO/DOGFOOD_LEARNED_BANK_20261024.md`), so this is a legacy-scope delivery. Its core rules are emulated by `launch.py`:
* one launch per phase;
* the lock re-verified at every launch;
* append-only `runlog/ATTEMPTS.jsonl` and `runlog/CYCLES.jsonl`;
* the runtime closure taken from `sys.modules`;
* the cycle and CPU caps enforced.

## 1. Question and evidence (S0)
Between what MiniFly's frozen KC state *contains* and what MiniFly *learns to output*, where are the bits lost?

**Evidence:**
* **V149** (`minifly_v149_input_ceiling_dev_20261025/REPORT.md`, main text *The Wisdom of Father Brown*): a best-possible logistic readout (R1) of untaught KC states scores:
  * V148 front end (FE-PERM, binary code): 3.757 bits/byte;
  * FE0: 3.379;
  * last byte alone: 3.554;
  * against bigram 3.553 and PPM-2 3.052.

  A KC vote (R2) on FE0 scores 4.137.
* **V148:** native MiniFly trained on the FE-PERM front end reaches 7.10 bits/byte (Alice).
* **S0 explore** (`minifly_v148_explore_s0_20261025/EXPLORE.md`): the fixed C16 output code caps prediction at 7.03 with no context.
* **Codex V151:** native writes execute a local error linearly. Codex V150 found output-rule changes did not help at a tiny native dose.
* **Comparison memo** (owner-adopted, this session): V149 confounded two front-end factors, and the readout chain has never been measured end to end on the same states.

**Cheapest analytic falsifier, done before code.** Share of the trace's energy carried by the most recent byte, `(1 − d²)` for per-byte fade d:
* 9.75% at d = 0.95;
* 34.9% at d = 0.807.

The permutation only relabels channels by lag, so it does not change that share. This motivates P1 below.

## 2. Block A: front-end factorial (state information)
* Scrambling (the fixed 88-channel permutation applied at every byte) ∈ {on, off} × per-byte fade ∈ {0.95, 0.807}.
* (on, 0.95) is V148's FE-PERM; (off, 0.807) is FE0 (`BC.FE0`, τ = 10 s). The other two cells are new, and identical except for the one factor.
* Each cell is read as the binary per-side top-k KC code of an untaught memory, exactly as in V149 (`values()` path; homeostasis thresholds zero).
* **Readout:** R1 only.
* The before-top-k reading is deferred. A dense 5,177-unit regression costs about 25× a sparse one and does not fit the default 2 CPU-h; it is listed as a follow-up.

## 3. Block B: the readout chain on one state
* **State:** FE0's observed KC activity `dx`, i.e. the KC code after the compartment's adaptation, which is exactly what MiniFly's stores receive.
* **The same 40 KB train / 5 KB calibration / 10 KB held-out split for every arm.**

| Arm | What it is | Role |
|---|---|---|
| R1 | Batch L2 multinomial logistic readout (V149 code); L2 chosen on calibration | ceiling of a linear readout |
| ONLINE | The same linear readout, trained **online with the local delta rule** `ΔW = −η s(n)(p − y) xᵀ`, `Δb = −η s(n)(p − y)`, with gradual warmup `s(n) = 1/16 + (15/16)·min(n/1024, 1)`; η chosen on calibration in C2 and then fixed; 3 passes, scored after passes 1 and 3 | what a local output rule achieves on the same samples |
| C16 | Batch-optimised **native port**: 16 values linear in the state (`v = W x + b`), centred, read through the fixed C16 code (`logits = C·(v − mean v)/4`) over all 257 symbols; L2 chosen on calibration | best any 16-compartment linear-value memory with this port could do |
| C16Q | C16 plus a learned 257-class bias q | how much a class bias recovers |
| NATIVE | Unchanged `FastMemory(front="fe0", homeo=False)`, V148 teaching (per-bit delta, temperature 2), one pass over the train pool, scored with V148's probe (temperature calibrated on calibration). Homeostasis is off so that its KC codes equal the frozen codes the other arms read | what MiniFly actually learns |
| R2 | V149's KC vote | local statistic, for reference |

**Descriptive decomposition** (one chosen path, not a causal decomposition):
* online-learning cost = ONLINE − R1;
* port cost = C16 − R1;
* bias recovery = C16 − C16Q;
* native-structure cost = NATIVE − C16.

**Classes:**
* R1, ONLINE and R2 use V149's convention: symbols seen in training plus OTHER, with OTHER shared evenly by unseen symbols.
* C16, C16Q and NATIVE use all 257 symbols, as the native port does.
* Both give proper distributions over all 257 symbols.

## 4. C0 testbed (before new learner code): a ceiling must pass and a floor must fail
On *Buster Bear* at **full size (40 / 5 / 10 KB)**, using evaluator-side codes, as in V149.

*Correction before any run:* the first version said "on the C2 dev split", which is the reduced 10 / 2 / 4 KB. V149's tiny run showed these checks are size-sensitive: the one-hot check failed at 4 KB and passed at 40 KB. C0 uses no states, so full size costs seconds.

The checks:
* R1 on the one-hot last symbol must land within 0.15 of bigram (ceiling);
* R1 on a constant code must not beat unigram by more than 0.02 (floor).

Failing either gives CANNOT_TEST_CLAIM_AS_SPECIFIED, and the round stops.

## 5. Predictions (three sentences each, with my probability; registered before code)
1. **The fade dominates the scrambling** (p = 0.6).
   * *Expect:* at the same fade, scrambling on and off differ by at most 0.10 in R1; at the same scrambling, fade 0.807 beats 0.95 by at least 0.20.
   * *Because:* the most recent byte's energy share is about 10% at 0.95 against about 35% at 0.807, whatever the scrambling.
   * *Weakened if:* the scrambling effect exceeds 0.10 at either fade.
2. **A local online rule nearly reaches the batch ceiling** (p = 0.5).
   * *Expect:* ONLINE after 3 passes is within 0.15 of R1, and after 1 pass within 0.40.
   * *Because:* the objective is convex, the features are sparse (about 181 of 5,177 active), there are about 41,000 samples, and the delta rule is the exact stochastic gradient.
   * *Weakened if:* the 3-pass gap exceeds 0.30.
3. **The 16-bit port is a large cost on real states** (p = 0.8).
   * *Expect:* C16 at least 1.0 bits worse than R1.
   * *Because:* the S0 analysis capped the C16 port at 7.03 with no context and 6.10 with the previous byte, against bigram 3.71.
   * *Weakened if:* C16 comes within 1.0 of R1.
4. **Native learning adds a further cost beyond the port** (p = 0.6).
   * *Expect:* NATIVE at least 0.5 bits worse than C16.
   * *Because:* per-bit teaching settles away from the port's optimum (S0: 6.34 against 6.10 with the previous byte), a single pass stores little (V148: early text erased), and fast/slow decay and adaptation act between writes.
   * *Weakened if:* NATIVE comes within 0.5 of C16.

## 6. Decision rule (fixed now)
The next round targets **the largest of the three measured costs**: online-learning (ONLINE after 3 passes − R1), port (C16 − R1), or native structure (NATIVE − C16).
* If the top two are within 0.20 of each other, the port goes first, because its fix (a learned graded code, S0 falsifier 4) is the best specified.
* Block A separately fixes which front-end cell the next round uses: the best R1 cell.

## 7. Data, cycles and caps
* **Corpus:** NLTK's copy of the Gutenberg texts (archive SHA-256 `2d3c3ab5…d611`), with V148's record and split rules (lines, 128-byte cap, contiguous train / calibration / held-out blocks).
* **C2 dev:** *Buster Bear* at reduced size (10 KB taught, 2 KB calibration, 4 KB held-out), to fix η and the L2 grids.
* **C3 rehearsal:** the full protocol (40 / 5 / 10 KB) on a fresh text, *The Man Who Was Thursday* (`chesterton-thursday.txt`), never used before.
* **S6 main, if granted:** a further fresh text, *Persuasion* (`austen-persuasion.txt`). V149's main text is not reused; official texts must be fresh.
* **Authorization** (`AUTHORIZATION.json`, the owner's words verbatim): implementation, technical and dev cycles. Default caps: 2 CPU-h, 5,000 records, 3 test cycles. No official run, no adoption, no pointer update.
  * Records count line presentations to learners: NATIVE writes and ONLINE updates. Batch evaluator fits count no learner records.
  * Estimated cost: C0 + C1 about 0.15 h; C2 about 0.4 h; C3 about 1.2 h; total about 1.8 CPU-h and about 4,200 records. Measured in C1; if the measurement exceeds the caps, the round stops and asks.

## 8. Not in scope
* Learning the KC code (input-side plasticity).
* Any change to native equations.
* Output-port replacement.
* Retention.

These come after this round, chosen by §6.
