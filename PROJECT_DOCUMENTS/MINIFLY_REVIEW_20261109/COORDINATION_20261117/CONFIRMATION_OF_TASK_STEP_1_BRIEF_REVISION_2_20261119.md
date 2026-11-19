# Confirmation of the brief of step 1, revision 2 (19 November 2026, written from 00:44 UTC by the clock of this machine)

Checked file: `TASK_STEP_1_THE_TISSUE_READS_THE_STREAM_20261118.md`, revision 2 (from 00:35 UTC). It was read with `check_step1_brief.py` as revised, with `THE_TISSUE_DEFINED_REVISION_5_20261119.md` ("revision 5"), with the roadmap's step 1 as changed at 00:31 UTC, and against the check of revision 1 (`CHECK_OF_TASK_STEP_1_BRIEF_20261119.md`, "the check"). The checker wrote neither the brief nor revision 5. The only programs run were `python3 -I` with the standard library, in all about 40 seconds of CPU: the check script; revision 5's worked example, part 1, by the check script's own functions; and the classifier of section 7.3 applied to two codes of the stream.

## Verdict

**Fit to go to the coordinator after the nine sentence changes of section 6 are entered.** Each is given as old text and new text. No further round is needed. Without changes 1 to 4, the worker would program a definition (revision 4 with errata) that differs from revision 5 in its order of opening, in its stored quantities and in its worked example. Without change 7, stage A's first outcome could be reached by a tissue that carries nothing about the identity of the present byte.

## 0. Terms

- *Position within the item*: for a byte of the stream, the number of bytes since the last end mark (byte 10) before it; it is 0 for the byte that follows an end mark.
- *Code of the stream*: a one-hot code of some function of the stream at each position, scored by the classifier of section 7.3 of the brief, as the brief's countermodel C0 is.
- All other terms are those of the brief and of revision 5.

## 1. Task (1): the findings of the check, one by one

| Finding of the check | State | Place in revision 2 |
|---|---|---|
| 1. Formula 11's sending-side test | Settled | E10 adopted (section 5.2; operation 7 of section 5.4); the 512-byte tissue of stage A records the loss (section 11.1, item 4; expectation X0) |
| 2. The rule could be met by the present byte alone | Settled | Section 7.6, condition 3; section 3, item 3 |
| 3. Baseline, distances, population | Settled | B = max(S(W0), S_fixed(W3)) (section 7.6); distance by the mode over blocks (section 1); a sample of 128 cells (section 7.3); all cells recorded (section 7.1, item 2) |
| 4. Amplitude-blind classifier | Settled | Sections 1, 5.8 item 11, 7.3 and check 9 (c) |
| 5. Coincidence rule | Settled | Section 5.5; it agrees with sentence 53 of revision 5 (section 2 below) |
| 6. Reversal of kinds | Settled | Section 5.2 (E3 not applied); section 5.3, item 7; check 1 (c) |
| 7. Lags 4 and 8 | Settled by the reviewer's decision | Sections 2, 7.6, 7.7; roadmap step 1 as changed |
| 8. The fallback | Partly | E10 is adopted, so the cause is addressed. However, opening 2 is still synaptic scaling, whereas revision 5 makes a* = 1/32 the second opening for step 1 (section 12.3; change 4) |
| 9. E4 against E5 | Settled by the reviewer's decision | Section 5.2, E5 row; revision 5, sentence 87 |
| 10. Growth under the threshold control | Partly | Registered (X3) and named in stage A's outcome 2 (section 11.3). Opening 2 does not follow revision 5's order (change 4) |
| 11. Registered expectations | Settled | Section 17: X0, X1, X2, X15 |
| 12. Cost | Partly | The cost of making is added (section 9). However, check 10's forecast multiplies by 2 a rate that is measured after most growth has taken place (section 3 below; change 8) |
| 13. Smaller points (six) | Settled | Section 5.4 operation 7; A.6b; section 1; check 5 with A.10; sections 2 and 5.8 |
| 14. The contract's design block | Settled | Section 3, items 2, 4 and 5; check 9 (d) |

## 2. Task (2): the brief's tissue against revision 5

The two texts were compared formula by formula, number by number, and operation by operation.

**Agreements.**
- Formulas 1 to 9 agree, as do formula 11 with E10 and formula 12, including the rule that a cell with no candidate keeps its step of last making.
- The numbers of section 16 agree, and so does the time constant of 4,096 steps.
- The feedback cells agree: they have their line only, θ = 0, and formulas 7, 9 and 12 do not apply to them.
- The kinds agree: sentence 89, with E3 not applied, and E5 not applied.
- The order of operations agrees. Revision 5, sentence 120, item 7, reads "gives up connections and then makes one, in the order of cell index". The brief gives up all connections first and then makes them in index order. The two orders give the same result, because a cell's giving up concerns only its own incoming connections.

**The coincidence rule.** The brief's section 5.5 and revision 5's sentence 53 are the same formula:
- Excitatory senders: η · e · x(k − 1) · (x(k − 1) − ϑ(k)).
- The threshold: ϑ(k + 1) = (1 − 1/4096) ϑ(k) + (1/4096) x(k)²/a*, with ϑ(0) = a*.
- Inhibitory senders: η · e · (x(k − 1) − a*).
- The rule acts at inner cells only.

There is one small inconsistency, inside revision 5 itself. Its sentence 123 says that all thresholds start at 0, whereas sentence 53 starts ϑ at a*. The brief follows sentence 53 and says so ("starting values 0 (except ϑ)"). The worker should follow the brief.

**Differences, and which text the worker follows.**
1. *The definition cited.* The brief cites revision 4 with an errata table (section 0) and uses revision 4's sentence numbers throughout; revision 5 renumbers them, for example the order of operations from 118 to 120 and the kinds from 87 to 89. The worker follows revision 5. Changes 1 and 2 cite it and give the rule for reading the sentence numbers.
2. *The running average ū.* Revision 5 withdraws ū (sentence 56). The brief still updates it in every tissue (operation 6), although it is used only by the 512-byte tissue. The worker follows revision 5 in the tissues with E10 (change 3). Keeping ū there would not change the result, but it would be a quantity that the definition does not have.
3. *The worked example.* Revision 5, section 13, part 1, has the feedback cell F with its line only and v = 0. Its table is not the brief's C.2, where F starts at v = 0.02. The checker reproduced revision 5's table to four decimals at every row given, using the check script's own functions with F's v set to 0. It also reproduced the final strengths: O u 0.7504, f 0.2498, F 0.2502; E h 0.2554, c 0.4973; I E 0.5003, q 0.2510, g 0.2495. Check 1 does not test this table, and it does not test part 2 (the whole tissue over the first 160 steps), which is the only published check of a whole-tissue implementation under E10. The worker follows revision 5, and check 1 tests both parts (change 5).
4. *The order of opening for step 1.* Revision 5 (section 15) has C15's second form, then C29 with a* = 1/32, then C17 and C18. The brief's opening 2 is synaptic scaling, with the reason "the first further form of C29, 'exempt output-end cells', is passed over"; that reason is revision 4's list. The worker follows revision 5 (change 4).
5. *The kind of a magnitude feedback cell.* Revision 5, sentence 113, makes it excitatory. The brief, section 5.8 item 3, makes it inhibitory. By revision 5's own sentence 89, a line that is larger when the answer is nearer plays the part of "yes", and under formula 7 a "yes" needs an inhibitory cell; on this point the brief's reasoning is the sounder one. The cell is unused at steps 1 and 2, so neither text binds the worker now. The sentence should leave the matter to the brief that uses the cell and put it to the reviewer (change 6).

## 3. Task (3): stage A

**Is it enough to decide between its outcomes?** Yes, in its structure:
- The three outcomes cover every case.
- The tissues are the right ones: both rules with E10, the fixed tissue, and the 512-byte tissue for finding 1.
- The quantities are read at the right places.

**Can a tissue pass "stays informative" without carrying anything about the byte?**
- *The part "neither silent nor constant".* It can be met without any information about the byte. Under E10, cells below their target add a connection every four bytes, so a cell's activity can drift within the window and its standard deviation then exceeds 0.01, whatever the byte.
- *The part S(d, 0, A2) ≥ 0.0717.* It can be met by a tissue that carries only the stream's structure and not the identity of the present byte. The checker scored, in A2 of seed 73129, three codes that do not contain the present byte:

| Code | Score for the present byte |
|---|---|
| Byte one back | 0.0617 |
| Position within the item | 0.0882 |
| Byte one back and position within the item, as one-hot of the pair | 0.0946 |

A population that tracks only the position within the item, for example through activity that follows each end mark, reaches 0.088 and passes the level 0.0717 at distances 2 and 4. The level should be the best of these codes plus the brief's margin, 0.0946 + 0.02 = 0.1146, and the check script should recompute it (change 7).

**Is the stop-and-ask level of 2.5 CPU-hours reasonable?** Yes. The checker's own estimate of stage A was under 2 CPU-hours, and the brief's section 9 gives 0.7 to 2.1. One fault remains in the forecast.
- *The cause.* Check 10 times bytes 256 to 1,279 and multiplies by 2 for growth. Under E10 a cell that stays below its target fills its neighbourhood in about 355 bytes (F.5b). By the timed bytes, cells may therefore already hold most of their connections.
- *The effect.* If they do, a measured rate near 0.2 s a byte, multiplied by 2, gives about 3.8 CPU-hours for the two plastic tissues, and the worker stops and asks although the true cost is near 2.
- *The fix.* The factor should depend on the number of connections reached (change 8).

An unnecessary question costs only a delay, so this is the least serious of the changes.

## 4. Task (4): the check script

`python3 -I check_step1_brief.py` was run at 00:42 UTC and took 24.0 seconds. Its output is identical, character for character, to the block of the brief's section 19.

## 5. Task (5): what is new and wrong

1. **Stage A's level of 0.0717 can be passed without the identity of the present byte** (section 3; change 7).
2. **The brief does not cite revision 5, and its check 1 tests neither part of revision 5's worked example** (section 2, differences 1 and 3; changes 1, 2 and 5).
3. **Opening 2 is not revision 5's second opening** (change 4). One cost consequence also follows:
   - The comparison for a* = 1/32 must be a fixed tissue with the same target, because the fixed tissue keeps formula 9.
   - This adds three fixed tissues, about 2.4 CPU-hours.
   - Three plastic tissues cost 11 to 46 CPU-hours, by section 9's range from 12 connections a cell to every neighbourhood full. Opening 2's limit of 20 CPU-hours may therefore be too small, and the coordinator should confirm it at the release of stage B.
4. **A consequence of the coincidence rule is not registered.** The rule weakens an inhibitory connection into an inner cell by η · e · a* = e/64 at every step at which the cell is silent, and by a smaller amount whenever its activity is below a*.
   - Inner cells are expected to stay below their target for long stretches (revision 5, sentence 60).
   - Their inhibitory connections are therefore weakened towards 1/128 and given up under E10: in about 150 steps when the sender's trace is 0.05, and in about 730 steps when it is 0.01.
   - The coincidence tissue's inner cells are thus expected to lose most of their inhibitory inputs within tens to hundreds of bytes. This is the homeostatic sign working as intended, but it is a large change of structure that stage A should register and report (change 9).
   - In the same tissue, ϑ starts at a* above the activity of nearly every inner cell, so the excitatory rule weakens active excitatory inputs at first, for a few hundred bytes, before ϑ falls below the activity. This needs no change, but the report should expect it.
5. **Revision 5 contradicts itself in two places**: on ϑ(0) (sentences 53 and 123), and on the kind of a magnitude feedback cell (sentences 89 and 113). Neither affects step 1. They are for the reviewer, not for the brief.

## 6. The sentence changes, ready to enter

Each old text occurs exactly once in revision 2 of the brief (verified with `grep -c`, except change 9, which adds a row). The new text replaces it.

**Change 1 (section 0, the definition cited).**
Old: ``- The tissue is `$C/THE_TISSUE_DEFINED_REVISION_4_20261118.md` ("the definition"). Sentences are cited as "sentence n" and formulas as "formula n" of the definition. The program is this definition, with the errata of section 5.2 and the decisions of section 5.8, and nothing else.``
New: ``- The tissue is `$C/THE_TISSUE_DEFINED_REVISION_5_20261119.md` ("the definition"), which folds in errata E1 to E10. Formulas are cited by their numbers, which revisions 4 and 5 share. Where this brief cites "sentence n", n is the number in revision 4, and the worker reads the sentence of revision 5 that carries the bracket "(r4 n)". Where the two revisions differ, revision 5 governs; the decisions of section 5.8 and the measurement of sections 7, 11 and 12 are this brief's. The program is this definition, with the decisions of section 5.8, and nothing else.``

**Change 2 (section 0, E10).**
Old: `Erratum E10 is the reviewer's, from finding 1 of the check; revision 5 of the definition will carry it.`
New: `Erratum E10 is the reviewer's, from finding 1 of the check. Revision 5 of the definition carries E1 to E10 (its section 17); the table of section 5.2 records how each is applied here.`

**Change 3 (section 5.4, operation 6, the withdrawn average ū).**
Old: `ā_i(k + 1) = (127/128) ā_i(k) + (1/128) x_i(k) and ū_ij(k + 1) = (127/128) ū_ij(k) + (1/128) z_j(k) (formula 10).`
New: `ā_i(k + 1) = (127/128) ā_i(k) + (1/128) x_i(k) (formula 10). In the 512-byte tissue of stage A only, also ū_ij(k + 1) = (127/128) ū_ij(k) + (1/128) z_j(k), which revision 5 withdraws (its sentence 56) and which is kept there for formula 11 as written; no other tissue holds ū.`

**Change 4 (section 12.3, opening 2).**
Old: `| 2 | C29 | Synaptic scaling in place of the threshold control: θ stays 0 for every cell (formula 9 is not applied), and after operation 5 of every step every strength from a cell into cell i (not a feedback cell) is multiplied by 1 + (1/128)(1/8 − x_i(k)), then formula 8. The first further form of C29, "exempt output-end cells", is passed over because step 1 reads no output-end cell | Run if round 1 does not reach step 1, on the arm that met more of the 36 conditions (12 per seed) in round 1, the first-form arm on a tie. Its comparison is round 1's fixed tissue (scaling is a change of strength) |`
New: `| 2 | C29 | The target a* = 1/32 in place of 1/8 wherever a* appears (formulas 9 and 12, and section 5.5), the first further form of C29 in revision 5 (its section 15, step 1) | Run if round 1 does not reach step 1, on the arm that met more of the 36 conditions (12 per seed) in round 1, the first-form arm on a tie. Its comparison is a fixed tissue with a* = 1/32 on each seed (three more fixed tissues, about 2.4 CPU-hours); the coordinator confirms the limit of opening 2 at the release of stage B |`

**Change 5 (section 8, check 1).**
Old: `(c) those of section 19, C.3, as the coincidence tissue makes it (E4; F excitatory, since E3 is not applied; the coincidence rule of section 5.5 at the inner cell I, ϑ_I starting at a*). Variant (a)`
New: `(c) those of section 19, C.3, as the coincidence tissue makes it (E4; F excitatory, since E3 is not applied; the coincidence rule of section 5.5 at the inner cell I, ϑ_I starting at a*); (d) every row of the table of section 13, part 1, of revision 5 and its final strengths (O: u 0.7504, f 0.2498, F 0.2502; E: h 0.2554, c 0.4973; I: E 0.5003, q 0.2510, g 0.2495), with F as revision 5 writes it (line only, v = 0, θ = 0); (e) for the whole first-form tissue of seed 73129 with E10 over its first 1,024 steps, the values of section 13, part 2, of revision 5 for cells 497 and 333 at steps 0, 32, 64, 96, 127, 128, 129 and 160, and the counts of its sentences 150 to 152 and of its section 12 (no connection given up; 1,984 connections made at step 128; 13,875 made by step 1,024; smallest starting strength between cells 0.0455). Variant (a)`

**Change 6 (section 5.8, item 3).**
Old: `3. **The kind of a magnitude feedback cell (item 3).** The magnitude line carries 1 − d/2, larger when the answer is nearer; it plays the part of "yes". A magnitude feedback cell therefore takes the kind of a yes-cell: inhibitory with formula 7. Unused at steps 1 and 2 (decision D3).`
New: `3. **The kind of a magnitude feedback cell (item 3).** Unused at steps 1 and 2 (decision D3). Revision 5 (its sentence 113) makes it excitatory; by the reasoning of its sentence 89, a line that is larger when the answer is nearer plays the part of "yes" and would need an inhibitory cell under formula 7. The brief of the step that first uses it settles this with the reviewer.`

**Change 7 (section 11.3, the level of "stays informative"; three places, and check 9).**
- Old: `S(d, 0, A2) ≥ 0.0717 (chance 1/46 plus 0.05).`
  New: `S(d, 0, A2) ≥ 0.1146: the score for the present byte of a one-hot code of the pair (byte one back, position of the byte since the last byte 10 before it), which holds the stream's structure but not the present byte (0.0946 in A2 of seed 73129; check 9 (e)), plus the margin 0.02.`
- Old: `shares at least 1/2 but S(d, 0, A2) below 0.0717`
  New: `shares at least 1/2 but S(d, 0, A2) below 0.1146`
- Old (section 5.7): `the level 0.0717 (chance plus 0.05) of the outcomes of stage A`
  New: `the level 0.1146 (the score of the code of the pair (byte one back, position within the item), plus 0.02) of the outcomes of stage A`
- Old (check 9): `it gives S within 0.0217 ± 0.01 at every lag. (d) It computes C<L(L) for L = 1, 2, 4, 8 in the same windows and writes the values into the check's receipt.`
  New: `it gives S within 0.0217 ± 0.01 at every lag. (d) It computes C<L(L) for L = 1, 2, 4, 8 in the same windows and writes the values into the check's receipt. (e) In A2 of seed 73129 it scores the present byte from a one-hot code of the pair (byte one back, number of bytes since the last byte 10 before it, 0 for the byte that follows it) and gives 0.0946 to four decimals; the check script prints the same value.`

**Change 8 (section 8, check 10).**
Old: `times 2 for the two plastic tissues (connections grow under E10; section 9)`
New: `times min(2, 93.8 / c) for the two plastic tissues, where c is the mean number of connections of a cell at the end of the timed bytes (under E10 connections grow at most to 93.8 a cell; section 9)`

**Change 9 (section 17, a new expectation after X16).**
Old: `| X16 | — | Checks 1 to 10 pass at the first attempt | Pass | 0.5 |`
New: `| X16 | — | Checks 1 to 10 pass at the first attempt | Pass | 0.5 |` followed on the next line by `| X17 | A; coincidence | Connections from inhibitory cells into inner cells at byte 16,383, against birth | Fewer than half remain: below its target an inner cell's inhibitory connections are weakened at every step (by η · e · a* when it is silent) and given up below 1/128 under E10 | 0.8 |`

Changes 7 and 8 also need the check script to print the level of change 7 (part E). That is a few lines; the value given here was computed by the checker with the classifier of section 7.3 (four stretches of 1,024 positions, gap 16, standardised one-hot features, nearest centroid, balanced accuracy).
