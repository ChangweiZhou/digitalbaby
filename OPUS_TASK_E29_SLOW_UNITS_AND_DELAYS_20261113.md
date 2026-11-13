# E29, slow units and transmission delays in the GRU. Revision 2, after a hostile audit (13 November 2026, 15:09 EST by the clock)

> **Status.** **Part A is released** to one Opus 5.5 worker: a read-only measurement on saved weights. **Part B is withdrawn as drafted and held** until the span job's block A is read and a new fixture is briefed and audited. Revision 1 (commit 873f6492) was read by a fresh hostile auditor of the same model family, a second reader and not an independent one. It confirmed eleven findings; the coordinator reran the evidence for the three that change the design and adopted all eleven (table at the end). No claim of a learning-core improvement is made, and no E-level is claimed for a measurement.

## Where this comes from

Mr. Xu, 13 November: a neuron's decision at one moment is shaped by several earlier moments, and not by a fixed number of them. Two things do this, and both may matter: **integration** (a slow neuron's state is already a blend of recent steps, with nothing stored outside it) and **delay** (transmission takes time, so neurons hear others at different past moments).

## What the audit showed before any run

1. **The plain GRU already carries a byte 40 steps.** The span job's block B is finished (branch `claude/target-test-sizing-stage-1-drmnlq`, `runs/span/B_M_{1104,1105,1106}.json`): with distances 4, 12, 24 and 40 mixed in one stream, the plain GRU from random weights passes at every distance at 100%, in all three seeds, after about 1,400 to 1,500 repeats. Stage 1's "nothing at 24" at a fixed distance is therefore a plateau of optimisation, not a limit on what the network can carry. Block A (fixed distances at equal dose) is not finished.
2. **The trained GRUs already have units that hold.** In the auditor's preliminary look at E26's births of seed 73129, a quarter of the units keep their state at 0.99 or more through the distractor letters, and every unit is fully rewritten somewhere in the item. The gate is used as a switch: hold through what does not matter, rewrite at what does. An average of the gate over positions hides this, and revision 1's measure and predictions were built on that average.
3. **On a copy at one fixed distance, a delay is a shortcut.** A unit whose delay is one less than the distance hands the key straight to the output. The auditor's 600-chunk test at distance 12 (rerun by the coordinator): plain reaches 6.6%, the delay arm 100%, and with every delay set to 7 it is back at plain's level. The delay arm would have measured whether the draw happened to contain the right lag.

## Part A (released). How the trained networks use their gates

Read-only; no training. New file `MOD97_LOCAL_RULE_20261106/e29_gates.py`; records in `exp45/e29/`.

**Files,** each checked against `weights_sha256` in its receipt: the untrained G1 per seed; E26's six births `exp45/e26/e26_dev_{S,B}_<seed>.pt`; E26's six lives `e26_life_{S,B}_<seed>.pt` (branch `claude/adoring-cori-b07aog`, commit ce409d64); the six crossover lives of E26-X. Seeds 73129, 93683, 116089.

**Items:** the sealed development rows add89 at k = 12 and at k = 20, 300 items each, loaded as the scorer loads them (`E.panel(t, E.split(t, .5, seed)[1], seed, ks=(0,4,8,12,20), n=300)` and `X.batch_tensor`). For the lives, also the untaught plus-minus composition panel the life scorer uses. Run `arch26.forward` with exact arithmetic and record z for every unit at every position, with each position's role (task code, operand digit, operator, distractor letter, `=`, answer).

**Per unit:**
- *hold share on letters:* the share of distractor-letter positions with z above 0.9;
- *lowest z off letters:* the minimum of z over the other positions;
- *retention across the letters:* the product of z over the run of distractor letters, averaged over items;
- the mean of z, for comparison with revision 1 only.

**A unit is a gated holder** if its hold share on letters is at least 0.9 and its lowest z off letters is at most 0.1. Report per file: the share of gated holders; the quartiles of retention across the letters; the share of units whose retention is at least one half; and the same split by which position rewrites the unit (the role of the position where its z is lowest).

**Check before any trained file is read:** on a hand-built GRU in which unit 0 has z = 1 on letters and z = 0 on digits and unit 1 has z = 0.5 everywhere, the measure returns a gated holder with retention 1 for unit 0 and not for unit 1.

**Predictions** (copy unchanged into `E29_PREDICTIONS.md` before any trained file is read). The auditor and the coordinator have seen the two births of seed 73129. **Those two files are reported and not judged.** Judgment is on what nobody has seen: the births of 93683 and 116089, and every life.

- **P29.1** In both births of both unseen seeds, at least 15% of units are gated holders. About 65%.
- **P29.2** The sweep-born network has a smaller share of gated holders than the backprop-born one, in both unseen seeds. About 40%.
- **P29.3** Among the four life arms, the two updated by backprop during the life have a larger share of units with retention at least one half at k = 20 than the two updated by the sweep, in at least two of three seeds. About 50%.

**Readings, fixed now.**
- P29.1 holding: training at this task turns a fast network into one with holding units, with no change at birth. The question for Part B is then whether birth can supply what training already finds, sooner or where training stalls.
- P29.2 or P29.3 holding is a difference between networks and no more. It does not show that holding is why the sweep composes less; that needs an intervention.
- A share is not read against another unless they differ by at least 5 points of units.

**Cost:** forward passes only; under one CPU-hour. Own branch; nothing to the default branch.

## Part B (held). What a new brief must contain

Not released. It is rewritten after the span job's block A reports, and audited again. Fixed now, from the audit:

1. **The fixture has no single lag.** Mixed distances, chosen from the span job's results so that the plain GRU fails or is slow at equal dose. If plain passes everything that fits the budget, Part B on a copy fixture is dropped and the question moves to E26's births and lives.
2. **The reading is not "carries a byte further".** Plain carries 40 already. A variant can show that it escapes a plateau, or learns in fewer repeats.
3. **Plain is rerun in the new implementation.** A written-out step does not match `nn.GRU` bit for bit (0 of 13 tensors after 10 chunks; largest difference 1.3e-5). Agreement to 1e-4 after 10 chunks and the same class.
4. **Speed arms:** the spread as a fixed buffer; the same spread as an initial update-gate bias, learned and exempt from decay; plain with its gate biases exempt from decay; uniform speed at two timescales. Without these a gain cannot be told from the decay pulling plain toward one half.
5. **Delay arms:** the spread; every delayed unit at one value that is no distance in the stream minus one; the spread with its assignment shuffled. The ring's contents before a delay has elapsed, which units are delayed, the generator and the rounding are specified, and a check shows the gradient crosses a delay and stops at the cut.
6. **Ranges are not taken from the fixture's own numbers.**

## Source check, read-only, alongside Part A

Open and cite, marking each claim confirmed, corrected or not found: work on initialising recurrent gates to a range of timescales (the coordinator recalls Tallec and Ollivier, 2018, from memory and has not opened it); heterogeneous time constants in recurrent or spiking networks; fixed or learned transmission delays in recurrent networks. For each: what was changed, tested on what and against what, seeds, and whether credit was exact. One page, `PROJECT_DOCUMENTS/LIT_CHECK_20261113/SOURCE_CHECK_TIMESCALES_AND_DELAYS.md`. It decides nothing.

## The audit and what was done with it

| # | Finding | Coordinator's check | Done |
|---|---|---|---|
| 1 | The delay arm measures whether a tap sits at the distance | `b_learn.py` rerun: plain 6.6%, delay 100% | Part B withdrawn; controls fixed above |
| 2 | Plain already passes at 40 under mixed distances | span records read: 100% at 4, 12, 24, 40, three seeds | stated; reading rewritten |
| 3 | Bit-equality with `nn.GRU` is impossible | auditor's output | tolerance; plain rerun |
| 4 | Mean z is not a timescale; P29.2 sat on its threshold | `a_meas.py` rerun | Part A rebuilt on holding through letters; seen files not judged |
| 5 | Fixed buffer against plain is confounded with decay | reasoning and born file's gates | two arms added to Part B |
| 6 | One distance cannot carry "not a fixed number" | — | uniform at two timescales; ranges not from the fixture |
| 7 | An admission item failed as written | auditor's output | goes with Part B |
| 8 | The delay was under-specified | read against the draft | specification fixed above |
| 9 | Citations: line range; grep finds the brief; G3 ignored | — | E20-B's G3 (fixed-rate banks with feedback, `exp45/e20b/e20_dev_G3_*`) did have slow units at birth in a loop and scored 20.2 / 1.1 / 1.0; revision 1 was wrong to say nothing had |
| 10 | Cost had no margin | auditor's timings | Part B resized when rewritten |
| 11 | P29.1 and the check were near-vacuous | untrained maximum timescale 2.54 | both replaced |
