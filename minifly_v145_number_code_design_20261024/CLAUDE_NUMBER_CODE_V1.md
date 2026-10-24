# NUMBER-CODE V1: can a digit code learned from counting make held-out addition transfer?

24 October 2026. Design memo for the next round, written before any code for it. Nothing in it has been run except the failure analysis in section 1, whose scripts and logs are in `analysis/`. It is refined in three cycles (refine, audit, test run, exercise) before a lock, as in LEARNED-BANK V1.

## 1. Why we are stuck: the failure analysis

Every unseen-addition attempt so far fails on held-out operand pairs, and in V144 so does an exact-gradient reference. LEARNED-BANK V1's mod-10 task failed too, while parity worked. The analysis below explains all of these with one cause. It uses evaluator-side checks on the LEARNED-BANK splits: 7 worlds, 40 training and 15 held orbits each.

**The learner sees two arbitrary symbols.**
* The FE-PERM trace of `a+b=` spans exactly a 19-dimensional space, which is 10 + 10 − 1 (`a1.log`).
* The distance between two digits' codes is unrelated to the distance between the digits: Spearman 0.09 (`a2.log`). To the learner, 3 is no closer to 4 than to 9.

**The input similarity points away from the answer.**
* Addition is a Latin square: each sum occurs at most once per row and once per column.
* So two different prompts with the same answer never share an operand, and prompts that share an operand never share the answer.
* Whatever the input makes look similar has a different answer.

**Frozen random features (the kernel regime) behave like a row-plus-column smoother.**
* The prediction for (a, b) is built from the training answers in row a and in column b. For a Latin-square task, the true answer of a held pair is absent from both, so it is never predicted.

  | Task | Smoother's prediction, 7 worlds (`a2.log`) | Measured with frozen features (LEARNED-BANK cycle 2) |
  |---|---:|---:|
  | Units digit, full sum | 0.00 in every world | units digit 0.00; V144 complete answers 0.067 (1 of 15 orbits) |
  | Parity | 0.23 (below chance) | 0.22–0.24 |
  | Carry | 0.83 | 0.84–0.89 |
  | Compare | 0.87 | 0.73–0.77 |
  | Copy | 1.00 | — |

* This also explains the negative held transfer gap measured at birth in LEARNED-BANK.

**Feature learning rescues only parity.**
* As a 10×10 table, parity is rank 1 (`a1.log`): one learned bit per digit, which the training pairs determine. That is why exact gradient, DFA and the native LEARN arm all succeed.
* Every units-digit class is a permutation matrix (rank 10), and the full-sum classes are anti-diagonals (rank up to 10).
* Of the units digit's four ASCII bits, only bit 0 (parity) is a single Fourier mode. Bits 1–3 spread over frequencies 1–4.
* An exact-gradient MLP on the actual trace learns every taught pair and transfers almost nothing (`a3.log`):

  | Training regime | Held units digit | Held full sum |
  |---|---:|---:|
  | Standard training | 0.02 | 0.00 |
  | AdamW with strong decoupled weight decay, 20,000 steps (grokking regime) | 0.13 | 0.00 |

  (Grokking regime: `a4.log`.)

**The isolating test: same learner and split, only the digit code changes** (`a3.log`, `a6.log`):

| Digit code | Held parity | Held units digit | Held full sum |
|---|---:|---:|---:|
| FE-PERM trace (the actual input) | 1.00 | 0.02 | 0.00 |
| Number line, a/9 (diagnostic prior) | 0.41 | 0.50 | 0.64 |
| Circle, angle 36°·a (diagnostic prior) | 0.77 | 0.64 | 0.31 |
| Ordinal codes learned from counting (successor representation; Spearman 0.99) | 1.00 | 0.01 | 0.00 |

* The code decides which functions are simple. Symbol codes make parity trivial and the sum impossible; a number line does the reverse.
* Ordinal similarity is not enough. What helps is a code in which "+1" is the *same transformation* for every digit (a homomorphism: a translation on a line, a rotation on a circle). Then a+b becomes a single direction a learner can find.
* Frozen random features fail even on good codes (0.00–0.01, `a5.log`). Their similarity is isotropic, so it prefers axis neighbours (a±1, b), which have different sums, over (a+1, b−1), which has the same one. A learned projection is still needed.

**Reading.** The stuck state is not caused by the native store, the local rule, stopping, or training length. The task asks the fly to discover the number concept from about 40 facts, in a code where that concept is invisible. Humans never do this: the count list and number line come first, and unseen small sums are obtained by counting on. Experiments that only vary mechanism, length or capacity on symbolic codes are predicted to fail.

**Limits.**
* These are evaluator-side, one-hidden-layer networks on the LEARNED-BANK splits; deeper models and other optimisers were not tried.
* Grokking was not observed at this size within 20,000 steps.
* Even with ideal codes, the vanilla MLP reaches only 0.50–0.64, so the downstream learner is a second, smaller bottleneck.

## 2. The question of this round

> Can a generic, local, predictive transition learner, trained only on raw-byte counting streams (no addition), learn digit codes in which "+1" is a fixed transformation? And when those learned codes replace the symbolic digit codes, does held-out addition transfer for a downstream learner that transfers nothing on the current code?

This round is evaluator-side and deliberately small. It tests the representation step only. The native port, recurrence (counting on), multi-digit sums and own-EOS generation are later rounds (section 9).

## 3. What is kept, changed and excluded

* **Kept:**
  * raw-byte input;
  * the FE-PERM event front end (timing, permute-then-add, decay), which supplies position;
  * the LEARNED-BANK orbit splits and world-naming scheme;
  * the controls culture: FROZEN, SHUF, a no-teaching arm and ORACLE plumbing;
  * the lock, fresh main worlds and a fail-closed summarizer.
* **Changed:** for digit bytes only, the fixed random receptor patterns (R_TABLE rows) are replaced by learned patterns, `r(byte) = U z(byte)`. Here `z(byte)` is a learned low-dimensional code and `U` a fixed random 88×d projection. Everything downstream of the receptor patterns is unchanged, so no parser and no operand slots are introduced.
* **Excluded (the round's constraints):**
  * arithmetic-specific features, number or carry encodings supplied to the learner;
  * attention, Transformers and ELM;
  * backprop on the candidate path (exact gradient is an evaluator reference only);
  * blind scaling.
  * The code learner never sees an addition prompt or an addition answer.

## 4. Code learner (the candidate mechanism)

**Experience.**
* Raw-byte counting runs, `0123456789`, separated by a boundary byte (space). Each run starts at a random digit and runs to 9: `3456789`, `0123456789`, `789`.
* There is no wrap from 9 to 0, so the stream teaches a successor without imposing a circle, and a line is the natural solution.
* A declared variant adds a cyclic stream (`...7890123...`) to test whether a circle emerges.
* Counting is the data the round adds. It is legitimate experience, not supplied structure, and it mirrors human development (count list before addition). The owner confirms this curriculum before implementation (section 10).

**Model.**
* A d-dimensional state (d ∈ {2, 4, 8}, provisional).
* A byte code `z(x)` for every byte.
* One shared affine transition `T`, `c`, reused at every step: the prediction of the next state is `T z(x_t) + c`.
* A decoder `D` reads the next byte's identity from the predicted state, so codes cannot collapse to zero.

**Learning (all local).**
* `D`: delta rule on the next-byte prediction error.
* `T` and `c`: delta rule on the state prediction error, `z(x_{t+1}) − (T z(x_t) + c)`.
* `z`: a three-factor update in which presynaptic input, postsynaptic gate and broadcast prediction error, the rule family that worked in LEARNED-BANK, move each byte's code to reduce both errors. A small norm constraint prevents collapse.
* An exact-gradient version of the same model is the evaluator reference. It separates "the model class can represent a homomorphic code" from "the local rule finds it".

**Homomorphism meters,** measured on the learned code with no addition involved:

| Meter | Definition |
|---|---|
| Transition error | mean ‖z(a+1) − (T z(a) + c)‖ / mean ‖z(a+1) − z(a)‖ over a = 0..8 |
| Composition error | mean ‖z(a+k) − T^k-composed z(a)‖, relative, for k = 2..5 |
| Spacing uniformity | coefficient of variation of the step lengths ‖z(a+1) − z(a)‖ |
| Order | Spearman correlation between code distance and \|a − a′\| |

## 5. Downstream test (addition, evaluator-side)

**Input.** The FE-PERM trace of `a+b=`, computed with the learned digit patterns. Non-digit bytes keep their original patterns.

**Tasks.**
* T-SUM: the full sum, 19 classes.
* T-UNITS: the units digit, 10 classes.
* T-PAR is kept as a positive sanity task, since every arm should pass it.
* Answers are classes in this round; complete byte output with own EOS moves to the native round.

**Downstream learners.**
* BP-REF: a one-hidden-layer MLP trained by exact gradient. It is a reference and answers whether the code is sufficient.
* DFA: the same network trained by LEARNED-BANK's three-factor rule. It is the candidate path.
* FROZEN-DOWN: fixed random features with a trained readout (kernel regime).

**Arms (code given to the front end).**

| Arm | Digit patterns | Role |
|---|---|---|
| SYMBOL | original R_TABLE rows | the current state (floor) |
| LEARNED | learned from counting streams (section 4) | candidate |
| SHUF-COUNT | learned from counting streams whose digit order is a fixed random permutation (e.g. 3→7→0→…) | control: a code that is just as structured but has the wrong successor, so it must not help addition |
| NO-COUNT | the same learner initialised but not trained | control: the effect of the code-learning machinery without experience |
| ORACLE-LINE / ORACLE-CIRCLE | ideal number-line or circle codes | evaluator-only ceilings, never a candidate |

## 6. Cycle 0, before any code-learner code is written

The testbed screen is the lesson of LEARNED-BANK and the kit review.

* **Ceiling.** ORACLE codes, swept over a grid of downstream settings (hidden size {64, 128, 256, 512} × learning rate × weight decay), must clear SYMBOL by at least 0.20 on held T-SUM *and* T-UNITS in at least 2 dev worlds.
* **Floor.** SYMBOL and FROZEN-DOWN must stay below the ceiling's best on both tasks, taking the best over their own grids.
* If the ceiling fails, the downstream learner is the bottleneck: redesign it, for example with a readout that can exploit a learned projection, before any representation learning. The preliminary numbers above (ceiling 0.50–0.64 against floor 0.00–0.02) suggest it passes, but the formal screen uses the grid and dev worlds only.

## 7. Measurements and decision rule (to be frozen at the cycle-3 lock)

**Primary metric.** Held accuracy on T-SUM, counted per orbit (15 per world), for BP-REF on the LEARNED arm, at the final checkpoint.

**Per main world, LEARNED must satisfy all of:**
* LEARNED − SYMBOL ≥ 0.20;
* LEARNED − SHUF-COUNT ≥ 0.20;
* LEARNED − NO-COUNT ≥ 0.20;
* the LEARNED code passes the homomorphism meters (thresholds fixed at the lock from dev worlds, before any main world exists).

**Gates.** ORACLE-LINE clears SYMBOL by ≥ 0.20, and T-PAR ≥ 0.90 for every arm, in every world.

**Verdict.** Over 4 fresh main worlds:
* **supported**: 4 of 4 pass;
* **partial**: 3 of 4 pass;
* **not supported**: otherwise.

The summarizer is fail-closed: it requires the exact set of locked worlds and arms, evaluates every gate, and invalidates the verdict on any source or closure mismatch.

**Also reported, not gated.**
* The same comparisons for DFA (the candidate learning path) and FROZEN-DOWN.
* T-UNITS.
* Per-orbit outcomes, split by one-byte and two-byte answers.
* The homomorphism meters for every arm.
* Taught accuracy.
* Full accounting: records, CPU, and every failed or diagnostic attempt.

**How outcomes will be read.**

| Outcome | Reading | Next |
|---|---|---|
| LEARNED passes with BP-REF and DFA | A code learned from counting makes addition transferable for the local rule too | Native round (section 9) |
| LEARNED passes with BP-REF but not DFA | The representation is sufficient; local credit is not | Study local credit on this code before the native round |
| The LEARNED code is homomorphic but transfer is weak | The downstream learner limits it (the ORACLE ceiling bounds this) | Improve the readout; the code is not at fault |
| The LEARNED code is not homomorphic | The local transition learner fails (compare the exact-gradient code reference) | Fix the code learner; no downstream work |
| SHUF-COUNT helps as much as LEARNED | Structure, not numeric order, drives transfer | The theory is wrong; stop and rethink |

## 8. Worlds, cycles and budget

* **Worlds.**
  * Development worlds D1–D3 for the cycles, tuning and cycle 0.
  * One fresh dev world, D4, for the dress rehearsal.
  * Four fresh main worlds, M5–M8, touched only by the locked protocol. M1–M4 are already used.
  * Split rules as in LEARNED-BANK.
* **Cycles.**
  1. Implement the code learner, the meters and the arms; unit checks (homomorphism meters on synthetic codes, front-end equality for unchanged bytes, probes leave state unchanged); a tiny test run.
  2. Dev-world runs; tune d, rates and stream length on D1–D3 only; measure cost.
  3. Fixes; dress rehearsal of the exact protocol on D4; lock (code hashes computed from the runtime closure, settings, worlds, rule).
* **Budget.** Evaluator models of this size run in seconds to minutes. The whole round, including cycle 0 and all failed attempts, should stay under 2 CPU-hours. The owner confirms a numeric cap before implementation.

## 9. Roadmap: later rounds, not part of this one

1. **Native port.** Learn the digit code with the native learned layer from counting streams, and rerun a LEARNED-BANK-style comparison on complete addition with native byte output and own EOS. Report both the component (learned bank) and the whole system.
2. **Counting on.** A recurrent state that applies the learned "+1" b times. It needs learned state over steps, which the native line lacks so far.
3. **Facts and procedure.** Store single-digit facts (memorising them is legitimate), then test generalisation where it belongs: unseen multi-digit sums built from known column facts with a learned carry procedure, held out at the level of column relations.

## 10. Owner decisions needed before implementation

1. Counting streams as added training data (section 4).
2. A numeric budget cap for the round (proposal: 2 CPU-hours).
3. Whether answers as classes are acceptable for this evaluator round, with complete byte output and own EOS deferred to the native round.
