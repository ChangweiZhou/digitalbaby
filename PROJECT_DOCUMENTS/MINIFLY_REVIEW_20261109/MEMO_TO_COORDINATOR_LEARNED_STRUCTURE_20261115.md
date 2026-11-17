# Memo to the coordinator: what the saved GRUs have built, what a life takes away, and three jobs to brief (15 November 2026)

From the outside reviewer, at Mr. Xu's request: "Write an appropriate memo for the coordinator to implement."

**Revision 2, the same evening.** E33 addendum 1 and E34 stage 1 have reported on `claude/vibrant-curie-ivmrt1` (ba6f5d52). I recomputed both tables from the committed weights; section 14 gives the check and what it changes. Jobs A, B and C are unchanged. One item is added to section 1 (item 5), one bullet to section 4 and three rows to section 12.

**Revision 3, later the same evening.** Three things are new. (a) The audit of the E36, E37 and E38 briefs is in `AUDIT_E36_E37_E38_BRIEFS_20261115.md` beside this memo; two of its required changes correct marks I proposed here, and sections 5 and 7 now carry the corrected marks. (b) Table 3.3 is restated with whole answers, as the harness scores them; the earlier table counted the two digits only. (c) Part II (sections 15 to 18) sets out the programme from here, after Mr. Xu's statements of this evening: a continual learner judged over a long life. Items 6 and 7 of section 1 are the new requests.

**Revision 4.** Section 17.10 is added: how short and long runs are combined (rate laws, trunks and branches, two probes, agreement between lengths, time in the learner's own units). It adds requirements to the protocol brief of job L0 and two items to L2; nothing else changes.

**Revision 5.** The audit of the E39 brief is in `AUDIT_E39_SMALL_WORLD_PROTOCOL_20261115.md`, with the second implementation in `e39_world.py`. It changes three things in this memo's Part II; section 17.11 lists them. Where 17.11 and an earlier section differ, 17.11 holds.

**Revision 6.** E36 and E37 have reported, E38 and the long E34 births in part, and E35-B's first stage. Section 19 gives my recomputation, what the results say, what they change in Part II, and one new job (item 8 of section 1). Section 3.6's explanation by weight decay is refuted by E38 and is marked so.

**Revision 7.** Section 20 is added after Mr. Xu pointed out that the plan was returning to the retention schemes of the earlier period. It withdraws item 2 of section 19.3 and sets a rule for adding any mechanism.

**Revision 8.** Section 21 is added after Mr. Xu's statements on scale: the quantity to compare is the total computation needed to reach a threshold, and which axis to scale is to be found by experiment. It asks for one new full-scale job (item 9 of section 1) and adds width as an axis in the small world.

**Revision 9.** The audit of the E41 brief is in `AUDIT_E41_COMPUTATION_TO_THRESHOLD_20261116.md`. It corrects two things in section 21.4 that were mine, the doses and the treatment of a run that does not reach a threshold; section 21.4 now carries the corrected doses.

**Revision 10.** Section 22 is added after Mr. Xu's statements of 16 November: each cell is connected to a limited number of cells inside a neighbourhood fixed in advance, the connections may change during learning, and each cell has its own pre-set period. It asks for one new full-scale job (item 10 of section 1). The arithmetic behind it is in `e42_design_checks.py` beside this memo.

**Revision 11.** The audit of the E42 brief is in `AUDIT_E42_CELLS_BRIEF_20261116.md`, with a numpy reader of the cell layer in `e42_cells_reader.py`. It corrects one thing in section 22.3 that was mine, the start of the variant `source`; the table there now carries the corrected form. It also defers `P8-mean` and replaces my cost estimate (sections 22.4 and 22.9).

**Revision 12.** Section 23 is added: a read-only measurement of what the first 64 updates change when a network that has lived learns a new family, on E40's files of seed 73129. It asks for four more files (item 11 of section 1).

**Revision 13.** Section 23 is repeated on seeds 93683 and 116089 with the four files the E40 worker committed (6922a3a4). The four readings hold on all three seeds under the sweep and under backpropagation; the tables of 23.1 now give the three seeds. Item 11 of section 1 is done.

**Revision 14.** Section 24 is added after Mr. Xu's objection that long-term memory exists without rehearsal. It restricts what section 20 established, gives a measurement of what survives a life without rehearsal, and asks for one small job (item 12 of section 1).

**Revision 15.** The audit of the E43 brief is in `AUDIT_E43_UNRELATED_LEARNING_BRIEF_20261116.md`. Its main finding bears on section 24 itself: under the standing weight decay, a family that receives no gradient is removed by the decay alone within about 2,000 updates. Section 24.6 gives the figures and what they change; the cost in 24.5 is corrected.

**Revision 16.** E43, E39's job L1a, E35-A's last stage and E41's first seed have reported. Section 24.7 reads E43 and asks for three additions to it (item 13 of section 1). Section 25 reads L1a, withdraws my explanation of the failed calibration, and proposes what replaces the small world's task set (item 14).

**Revision 17.** E42's first seed is read and checked with my reader (`E42_SEED_73129_VERIFICATION_20261116.md`). Section 22.12 gives the reading, one defect of my own replacement rule that it exposes, and what follows. Nothing is asked for until the other two seeds report.

**Revision 18.** The audits of the E46 and E44 briefs are in `AUDIT_E46_WEIGHT_DECAY_BRIEF_20261116.md` and `AUDIT_E44_LONG_LIFE_WORLD_BRIEF_20261116.md`. Section 26 is added after Mr. Xu's statement that locality must hold at the input and the output as well: it gives the arithmetic and the requirements for the job that follows E42 (item 15 of section 1).

**Revision 19.** E43's additions have reported in part (reminders complete; plain steps complete under backpropagation). Section 24.8 reads them with a measurement of my own on the weight files (`e43_where_the_loss_sits.py`): the first family is overwritten throughout the network, and neither the readout alone nor the optimizer's equal steps explain it. Section 27 asks for one small job that follows from it, E48, and item 17 for two additions to E43.

**Revision 20.** A change of name only. The coordinator has given the number E47 to the job of section 26 (locality at the input and the output; brief at 7247682a). The job of section 27, which I had called E47 in revision 19, is E48 from here on. Item 15 is done: the specification exists and is being audited.

**Revision 21 (17 November).** Part III (sections 28 to 30) is added at Mr. Xu's instruction to discuss next steps and to coordinate without him. Section 28 checks every result that arrived on 17 November (E41 and E42 on three seeds, E46, the E47 trial, E48, E49, E43's second addendum, E44's calibration) and gives the audits of the roster note for the first long lives and of the E50 brief. One reading of mine in 22.12 is withdrawn: periods do not bring the limited network level with the dense one. Section 29 is a measurement on E43's networks: every cell is in use on every item, and one family fills the state of the cells. Section 30 gives the decisions and one new job, E51. Items 18 to 23 of section 1 are the requests.

Everything measured here is read-only: forward passes on saved weights with the numpy reader, and a read of receipts. Nothing was trained. The files are the six E26 births and nine lives (E26 arms S and B, E31 arm K) on seeds 73129, 93683 and 116089. The lives were not designed for these questions, so sections 3.1 to 3.5 are readings, not outcomes of a registered test.

Shell names used below: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106; R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`. The scripts are on `claude/vibrant-cerf-sje726` at `b983f74a`; file arguments are paths under `$M/exp45/` on the default branch.

## 1. What is asked of you

1. **Brief three jobs**, in this order. Each brief comes to me before any worker starts, as usual.

   | Job | Question | Runs | CPU (my estimate) | Section |
   |---|---|---|---|---|
   | A | After a life, how many updates bring the unrehearsed birth family back? | 9 reminders, 6 references, 1,024 updates each | 0.6 h | 5 |
   | C | Does a learner that has lived need fewer items for a new family than a fresh one? | 6 short births | 0.9 h | 6 |
   | B | What erases the unrehearsed family, and does an anchor protect it? | 10 lives of two blocks, one seed | 4.7 h | 7 |

2. **Let E35 run through stage 1b as released, and brief nothing after it** (section 8).
3. **From now on, score every earlier family at every stage of a life**, with add and sub split by wrap (section 9).
4. **Enter in the record** Mr. Xu's three statements of today (section 2) and the fact that the E26 and E31 lives never rehearsed the birth family and lost it (sections 3.3 and 3.4). I searched the review from 9.111 onward and found no discussion of it.

5. **Release E34 stage 2 as its brief says, with two additions** (section 14): one backprop birth with the same gain at tau 0.5 on seed 73129, and one reported fact, the row at which each arm first learns in range.

6. **Apply the audit's six required changes to E36, E37 and E38 as written and release them** (`AUDIT_E36_E37_E38_BRIEFS_20261115.md`). No re-audit is needed after them.
7. **Brief the long-life programme of Part II**: one protocol brief (job L0, section 17.7), audited once, and the two long births of E34 (section 18), which need no new code.

8. **Brief one more full-scale job, E37 at a tenth of the pairs** (section 19.4): lived and fresh learners on a new family with a tenth of its pairs taught. About 4 CPU-hours by my estimate. It comes to me before any worker starts.

9. **Brief one more full-scale job, E41: computation to threshold by width and by taught fraction** (section 21.3). About 28 CPU-hours by my estimate, to be divided among the three workers that E36, E37 and E38 have freed. It comes to me before any worker starts.

10. **Brief one more full-scale job, E42: cells with a limited number of local connections and their own periods** (section 22). Forty-eight births in two stages after a stage of checks, 50 to 70 CPU-hours by my estimate. It comes to me before any worker starts. It does not wait for the small world and displaces nothing released.

11. *(Done, revision 13.)* **Ask the E40 worker for four more files** (section 23.4): the row-64 weights of the lived S and B runs of seeds 93683 and 116089, whose hashes are in the receipts. About 18 MB. They let me repeat section 23 on three seeds.

12. **Brief one small job, E43: is a family lost when the learner then learns something unrelated?** (section 24.5). About 2 CPU-hours by my estimate. It comes to me before any worker starts.

13. **Three additions to E43** (section 24.7): push the weight files its report lists, which are not on the remote; a reminder test from the networks at update 4,096; one arm under plain gradient steps. About 2 CPU-hours.

14. **Brief the replacement of the small world's task set** (section 25): families in the full-scale format, a learner born on one of them, and a calibration first. It comes to me before any worker starts.

15. *(Done; it is the coordinator's E47.)* **Write the specification of the job with a local interface** (section 26), as runnable code, for audit. It runs after E42's other two seeds and E43's additions.

16. **Brief one small job, E48: a new family learned only in connections gated by its own code** (section 27; called E47 in revision 19). About 1.5 CPU-hours. It comes to me before any worker starts.

17. **Two further additions to E43** (section 24.8, end): the `unrelated` arm with the first family rehearsed, as the conventional reference; and the `unrelated` arm started from E42's `F10` and `P8` networks of seed 73129. About 2.5 CPU-hours together.

18. **Enter the nine changes of the audit of the roster note for the first long lives and let the lives start** (28.8; `AUDIT_E44_L2_ROSTER_NOTE_20261117.md`, section 8). No second audit of the note. I read the changed runner and its check receipts when they are committed; that need not hold the trial.

19. **Do not release E47's roster, and do not continue E49** (28.4, 28.5, 30.2).

20. **Enter the nine changes of the audit of the E50 brief and release its stage 1** (30.2; `AUDIT_E50_SPARSE_INPUT_CODE_BRIEF_20261117.md`, section 9). No second audit.

21. **Brief one job, E51: a new task learned in cells held in reserve, with what the first family uses left unchanged** (30.3). Twelve runs on one seed first, about 2.5 CPU-hours. It comes to me before any worker starts. `e51_design_checks.py` holds the construction and the rule in numpy, for your specification to be compared with.

22. **Housekeeping.** Merge E41's three-seed report (2722d1c1) and E46 (b76f5979) into the default branch, since the roster note cites them. Have E43's report mark the two `P8` weight files of its second addendum as incomplete (28.6).

23. **Enter in the record** the corrected reading of E42 on three seeds (28.3), that the sweep does not learn with a limited input (28.5), and the measurement of section 29.

E35 continues as released. E33 is complete and needs nothing further.

## 2. Mr. Xu's statements of 15 November

Verbatim, from the session with the reviewer.

- **On the goal.** "Note our goal is persistent learning, not conventional training+validation+test data set setting. So the main problem is given fixed memory / computation capacity, how does local learning beat naive back-propagation in generalization tasks."
- **On the line.** "I repeatedly said as long as we do not do full backpropagation, anything is okay."
- **On structure.** "So now given the circuit structure itself is shaped by learning in our setting, the literature is only of limited help. Our neural network structure would change with time."

My reading of the third, which the measurements below support: the thing to study is the structure the learner builds for itself and what later learning does to it. A hand-wired circuit is at most a bound on that.

## 3. Findings

### 3.1 Every trained network holds a circular code of the answer, the same under both rules

Take the hidden state at `=` for every operand pair of a task, no letters, and average it per answer value. Call the 89 (or 97) averages the answer code.

| Networks | Task family | Own exact answers | Two strongest Fourier frequencies of the answer code | Their share of the code |
|---|---|---|---|---|
| 6 births (S and B) | add, sub mod 89 | 93–100 | 1 and 9, in all 12 readings | 0.85–0.95 |
| 9 lives, at the end | add, sub mod 89 (birth family, not rehearsed) | 1–47 | 1 and 9, in all 18 | 0.83–0.98 |
| 9 lives, at the end | add, sub mod 97 (taught in the life) | 88–99 | 1 and 10, in all 18 | 0.85–0.97 |
| Untrained, 2 draws | add mod 89 | — | — | no answer code: 0.000 of the state's variance lies between answer values |

- Frequency 1 is a circle over the size of the answer. Frequency 9 (mod 89) or 10 (mod 97) is a circle over its units digit: 89/10 and 97/10 cycles.
- The geometry is circular, not ordered: distances between codes are explained by the circular difference of the answers about as well as by the plain difference (largest gap 0.06). For max mod 89 the plain difference explains far more (0.77–0.92 against 0.56–0.64, seed 73129), as an ordered code should.
- The code has the same form for both moduli. That is nearer to inventory 乙 of E35 than to inventory 甲's table for one modulus.

```
python3 -I $R/learned_structure.py add89,sub89,add97,sub97 e26/e26_dev_S_73129.pt e26/e26_life_S_73129.pt e31/e31_life_K_73129.pt e26/e26_dev_B_73129.pt e26/e26_life_B_73129.pt
python3 -I $R/learned_structure.py max89 e26/e26_dev_S_73129.pt e26/e26_dev_B_73129.pt e26/e26_life_S_73129.pt e26/e26_life_B_73129.pt e31/e31_life_K_73129.pt
python3 -I $R/learned_structure.py add89,add97 untrained
```

### 3.2 A life rebuilds the weights and keeps the form of the code

Seed 73129. "Graft" means the life-end network with one group of weights put back to its birth value.

| | add 89 | sub 89 | add 97 | sub 97 | Change of the group over the life (norm of difference / norm at birth) |
|---|---|---|---|---|---|
| Life S, as saved | 23.6 | 40.3 | 95.7 | 90.1 | |
| with the birth's input layers | 9.4 | 16.8 | 2.5 | 3.8 | 0.27 |
| with the birth's recurrent weights | 4.4 | 1.6 | 1.0 | 7.2 | 1.17 |
| with the birth's read-out | 12.8 | 21.3 | 53.0 | 43.6 | 0.96 |
| Life B, as saved | 19.6 | 46.9 | 95.6 | 93.5 | |
| with the birth's recurrent weights | 35.9 | 30.3 | 7.9 | 15.9 | 1.12 |

- The recurrent weights move by more than their own size at birth. No single group put back restores the old family.
- So what persists through a life is the form of the code (3.1), not the weights that carry it.

```
python3 -I $R/graft_birth_into_life.py add89,sub89,add97,sub97 e26/e26_dev_S_73129.pt e26/e26_life_S_73129.pt e31/e31_life_K_73129.pt
python3 -I $R/graft_birth_into_life.py add89,sub89,add97,sub97 e26/e26_dev_B_73129.pt e26/e26_life_B_73129.pt
```

### 3.3 What a life removes is the old family's modulus

The E26 and E31 lives draw from families F1 (mod 97), F2 (compositions) and F3 (mod 73). The birth family (mod 89) is never in the stream (`git show $D:$M/exp26_candidate.py | sed -n '214p;242,244p'`).

Whole answers (two digits and the end symbol, as `exp16_integrated.score` counts them) at the end of each life on the birth family, all pairs, no letters. "Wrap" pairs are those with a + b ≥ 89, or a − b < 0. The last column is read from the two digits alone. Revision 2 of this memo counted the two digits throughout; six cells moved, the largest from 17.6 to 3.9 (K 116089, sub).

| Life | add: no wrap | add: wrap | sub: no wrap | sub: wrap | Wrap pairs answered with a life modulus |
|---|---|---|---|---|---|
| S 73129 | 45.5 | 1.3 | 79.3 | 0.4 | add 13% mod 97; sub 12% mod 73 |
| S 93683 | 54.0 | 0.3 | 12.1 | 0.4 | add 32% mod 97 |
| S 116089 | 25.8 | 1.3 | 2.5 | 0.1 | add 13% mod 73 |
| B 73129 | 37.8 | 1.0 | 92.8 | 0.0 | **sub 89% mod 97**; add 20% mod 97 |
| B 93683 | 25.8 | 0.1 | 39.8 | 0.1 | sub 32% mod 73 |
| B 116089 | 49.3 | 0.1 | 91.6 | 0.1 | **sub 83% mod 97**; add 26% mod 73 |
| K 73129 | 56.4 | 0.7 | 73.8 | 0.1 | **sub 59% mod 73** |
| K 93683 | 43.0 | 0.6 | 72.9 | 0.2 | add 30% mod 97 |
| K 116089 | 62.9 | 0.6 | 3.9 | 0.3 | add 41% mod 73 |
| Six births | 94.0–99.8 | 96.4–99.5 | 95.2–99.6 | 91.5–99.6 | 0.0–0.1% |

- Pairs that need the wrap at 89 are right 0.0–1.3% of the time in all 18 cells. The network still answers with two digits (98–100%).
- Pairs that need no wrap are right 2.5–93%. The shared arithmetic survives in part.
- In several lives the old prompts are answered with a modulus the life taught.
- Sweep, Kolen–Pollack and backprop lives show the same pattern.

```
python3 -I $R/e36_e38_brief_checks.py H e26/e26_life_S_73129.pt e26/e26_life_B_73129.pt e31/e31_life_K_73129.pt      # whole answers
python3 -I $R/life_answers_on_birth_tasks.py e26/e26_dev_S_73129.pt e26/e26_life_S_73129.pt e26/e26_life_B_73129.pt e31/e31_life_K_73129.pt   # two digits; the last column
```

### 3.4 The loss happens in the first block (receipts)

Held-out add and sub of the birth family at 12 letters, mean of the two, from the life receipts' own columns:

| Arm | At the start | After block 1 | At the end |
|---|---|---|---|
| S | 98 · 98 · 100 | 49 · 7 · 34 | 33 · 18 · 14 |
| B | 98 · 99 · 94 | 44 · 1 · 33 | 37 · 27 · 36 |
| K | 98 · 98 · 100 | 46 · 5 · 28 | 31 · 36 · 30 |

Max mod 89 swings instead of falling (seed 93683, arm S: 98, 19, 54, 84, 91, 74, 28).

### 3.5 Later families reach high held-out scores from a quarter of the pairs in few items (receipts)

Same score, by items of that family seen so far:

| Family | Pairs taught | Items seen | Held-out add/sub at 12 letters |
|---|---|---|---|
| Birth, mod 89, from nothing | half | 262,000 | 63–84 (six births) |
| | | 524,000 | 88–95 |
| Life F1, mod 97 | a quarter | 393,000 | 91–98 (nine lives) |
| Life F3, mod 73 | a quarter | 131,000 | 77–94 |
| | | 262,000 | 80–95 |

This is not yet a saving: no fresh network has been taught a quarter of the pairs of one family. Job C supplies that. The same pattern was seen once before on the window network, where developed births beat backprop from nothing by 20–40 points at a tenth of the pairs (9.4, 9.6).

```
python3 -I $R/acquisition_from_receipts.py
```

### 3.6 A likely cause in our own settings (arithmetic, not a measurement)

AdamW at learning rate 0.001 with decay 0.25 multiplies every weight outside the embedding table by 0.99975 per update (`git show $D:$M/exp16_integrated.py | sed -n '176,187p'`). A component of the weights that the current stream does not push on is left with 21% of its size after one block of 6,144 updates and 0.01% after a six-block life. The embedding rows of the old code bytes have no decay and no gradient, so they stay; everything they feed is shared and decays or is rewritten. Job B separates decay from rewriting.

**Refuted (revision 6).** E38 shows the same loss with no decay at all: the wrap pairs fall from 97% to 18% within 64 updates and to 3% within 256, with decay toward zero, without decay, and with either anchor (section 19.2). The new family's own updates erase the old modulus, about fifty times faster than decay could.

## 4. Reading

- **The rule has again made no difference.** Both rules build the same code, lose the same thing in the same block, and acquire later families alike. Every job below runs both and reports the paired difference.
- **Retention in this learner is a small problem with a large symptom.** What is lost is one parameter per family, the modulus. The symptom is that the whole family scores 1–47%.
- **Where a claim against naive backprop could come from.** Naive backprop keeps nothing of an unrehearsed family's modulus (0.0–1.0% on wrap pairs in the three backprop lives). Rehearsal keeps it and takes a share of every batch (9.21, 9.64). A learner that keeps it with no rehearsal, at the price of one extra number per weight, would do at equal updates what naive backprop does not. Job B tests the simplest such device. The same device could be added to backprop, so the claim would be against naive backprop only, which is how Mr. Xu put the question.
- **A second place, since E34.** With every byte a target, naive backprop ends at 82.0, 82.8 and 85.4 at 40 letters. The sweep with E34's confidence gain at tau 0.5 ends at 79.8 on the one seed run, with a final row of 99.3 (section 14). If stage 2 puts it above backprop on the new seeds, that is a claim against naive backprop at no extra memory. The backprop birth with the same gain says whether it belongs to the rule or to the objective.
- **The form the claim should take:** at equal stored numbers and equal updates, with every earlier family scored at every stage, and with both the memory and the rehearsal share of each arm stated.

## 5. Job A: the reminder test

**Question.** After a life that never rehearsed the birth family, how many updates on that family bring it back, against learning it from nothing by the same rule?

- **Start (9 files).** `e26/e26_life_S_<seed>.pt`, `e26/e26_life_B_<seed>.pt`, and `e31/e31_life_K_<seed>.pt` with its `e31/e31_life_K_<seed>_Bhh.pt`; three seeds.
- **Update.** Each by the rule of its own life. Arm K continues from the saved feedback matrix with its after-step update; `exp31_switch.py` builds that matrix from the seed, so loading it is new code. A fresh optimizer from `make_opt`, since the lives saved no optimizer state that I can find.
- **Data.** The birth's own stream: the four mod-89 operations, the taught half of the pairs of that seed, 0 to 12 letters, batches of 64. Use a stream seed other than the birth's, the same for every arm of a seed.
- **Dose and rows.** 1,024 updates. Save weights at 0, 16, 32, 64, 128, 256, 512 and 1,024. At 256 updates each taught pair has been shown about once (4,096 items per operation over 3,960 taught pairs).
- **Reference (6 runs).** A fresh network with E26's initial values for the seed, taught the same batches by the sweep and by backprop, same rows. Arm K is read against the sweep reference.
- **Scores at every row.**
  - the birth family's held-out pairs at 12 letters, on the panels the life receipts use, per operation;
  - add and sub on all pairs without letters, split by wrap;
  - the life's families at 12 letters and `compose.within`, to show what the reminder costs them.
- **Marks (corrected in revision 3; see the audit, 4.1).** "Restored" at a row: mean of held-out add and sub at 12 letters at least 85 at that row. "Wrap score" at a row: the mean of the add and sub wrap-pair scores on all pairs without letters. Per arm, in two of three seeds; the rows are tried in order and the first that holds is the reading. My first version read rows 2 and 3 on the held-out mean, which a reminder already meets at row 0 (it starts at 13 to 37, because everything but the wrap is kept):

  | # | Pattern | Reading |
  |---|---|---|
  | 1 | Restored at row 256 | The family's rule was still in the network. A life costs the old family about one showing of each taught pair. |
  | 2 | At row 1,024 the wrap score is at least 30 points above the reference's | Partial savings. |
  | 3 | At row 1,024 the wrap score is less than 15 points from the reference's | No savings: what the life removed is relearned no faster than a fresh network learns it. |
  | 4 | Anything else | Mixed; the curves are reported. |

- **Checks.** Row 0 has the life-end file's tensor hash and reproduces the last stage's birth-family scores in the life receipt. Batch digests are equal across the arms of a seed. For arm K, the loaded matrix has the saved file's hash and E31's identity check holds on the first steps. One dry run of 64 updates with CPU time.
- **What it cannot show.** The optimizer state is fresh in both reminder and reference. The reminder is given alone, not inside a life's stream.

## 6. Job C: a from-scratch reference for a life family

**Question.** Is 3.5 a saving? Does a learner that has lived reach a held-out score on a new family from fewer items than a fresh learner under the same rule?

- **Runs (6).** A fresh network with E26's initial values, sweep and backprop, three seeds, taught family F3 alone at a quarter of the pairs: `X16.family_entries("F3", seed)` through `X16.entry_stream(entries, seed + 105)`. That is the life's own F3 stream (`seed + 100 + si` with `si = 5`), so the fresh network sees the same items in the same order as the life did.
- **Dose and rows.** 4,096 updates; weights at 1,024, 2,048 and 4,096. Row 2,048 is 131,072 items, against 130,000–131,000 F3 items in the lives after block 5.
- **Score.** The life receipts' F3 panels at 12 letters; mean of held-out add and sub.
- **Read.** Life minus fresh at equal items of the family: the life's score after block 5 against row 2,048, and after block 6 against row 4,096; S life against sweep fresh, B life against backprop fresh, K life against sweep fresh. "Saving": at least 15 points in two of three seeds. "None": within 15. "Cost": at least 15 the other way.
- **Optional second half.** The same for F1 (`seed + 101`, 6,144 updates, read against the lives after block 1). About 1.4 CPU-hours more.
- **Cross-check, free.** E35's arm `G` at f = 0.25 is a fresh GRU on a quarter of the pairs of the mod-89 family; its curve at 4,096 updates belongs in the same table.
- **What it cannot show.** The lived learner had also seen about 1.8 million items of other families in its life and 1.6 million in its birth, so the saving is in items of the new family, not in total experience. The fresh network gives every update to F3; the life gave a third. That favours the fresh network per update and is neutral per item.

## 7. Job B: what erases the unrehearsed family, and does an anchor protect it?

**Question.** Is the loss in the first block made by decay toward zero, by the new family's own updates, or by both? Does pulling each weight toward its birth value keep the old modulus while the new family is still acquired?

- **Start.** The E26 births, arms S and B. Stage 1 on seed 73129.
- **Life.** E26's first two blocks only (F1.1 and F1.2, 12,288 updates), same entries and stream seeds as E26, so that the first arm below repeats E26.
- **Arms, per rule (10 lives).**

  | Arm | One difference from E26's life | Extra memory | Rehearsal share |
  |---|---|---|---|
  | `zero` | none: decay 0.25 toward zero | none | 0 |
  | `none` | no decay during the life | none | 0 |
  | `anchor` | decay 0.25 toward the birth value of each weight | one number per decayed weight (1.12 million; 4.5 MB) | 0 |
  | `anchor10` | the same at strength 2.5 | the same | 0 |
  | `rehearse` | decay as `zero`; the birth family's taught entries join the joint stream from block 1 as one more family | none | half of every batch |

- **The anchor, as code (my proposal; the specification is yours).** Set the optimizer's `weight_decay` to 0 for the group that had 0.25, and before each `opt.step()`:

  ```
  for p, p0 in zip(decayed_params, birth_values):
      p.sub_(LR * KAPPA * (p - p0))          # KAPPA = 0.25 (anchor) or 2.5 (anchor10); LR = 0.001
  ```

  torch's AdamW shrinks before the Adam step, so the pull goes before the step too. Each weight uses only its own two values, so the device is inside Mr. Xu's line.
- **Rows.** Weights at 0, 64, 128, 256, 512, 1,024, then every 1,024 to 12,288. The early rows separate the two causes: decay acts over about 4,000 updates, rewriting can act within hundreds.
- **Scores at every row.** As Job A's first two bullets for the birth family; held-out add and sub of F1 at 12 letters.
- **Marks (my proposal), after block 2.**
  - "Keeps the birth family": held-out add/sub mean at 12 letters at least 80, and wrap pairs at least 70.
  - "Loses it" (corrected in revision 3; see the audit, 4.3): wrap pairs, mean of add and sub, at most 20. My first version, the held-out mean at most 50, sat exactly on the case of interest: a life that loses the wrap and nothing else scores 48 to 50.5 on seed 73129's panel.
  - "Acquires the new family": F1 held-out add/sub mean at 12 letters at least 85 (E26: 92–97).
- **Readings, per rule.**

  | Result | Reading |
  |---|---|
  | `zero` does not repeat E26's stage 1 and 2 scores and batch digests | The runner is wrong. Stop. |
  | `none` keeps | Decay toward zero is what erased the family. |
  | `none` loses | The new family's own updates erase it; decay is not needed for the loss. |
  | `anchor` or `anchor10` keeps and acquires | A candidate: retention with no rehearsal for one stored number per weight. Stage 2. |
  | An anchor keeps and does not acquire | Too stiff at that strength; report both strengths side by side. |
  | Neither anchor keeps | A uniform pull is not enough. The next device would weight the pull by each weight's use for the old family, or give families separate resources. Not briefed now. |
  | `rehearse` | The reference: what rehearsal keeps, and what it costs F1 per update and per item. |

- **Checks.** `zero` repeats E26's stage 1 and 2 (digests, losses, scores). `anchor` with `birth_values` set to zero agrees with `zero` after 64 updates (largest absolute difference at most 1e-5 of the tensor's largest entry). In `rehearse`, the count of birth-family items per block is recorded. Two independent implementations of the anchor step, since it is new code.
- **Stage 2 (released by you after stage 1 is read).** Arms that keep and acquire, with `zero`, on the other two seeds; then one full six-block life per rule, to see composition and length under the anchor. Once the small world of Part II exists, further strengths and the moving anchor are searched there and only the winner returns to full scale.
- **What it cannot show.** A fixed anchor suits a two-block screen only. Over a long life the anchor has to move, for example a slow copy of each weight; that is designed only if the screen passes. One seed in stage 1.

## 8. E35: scope

- Stages 0, 1a and 1b run as released, stop rule included. E35 now serves two purposes: the bound on what a perfect, never-forgotten circuit gives, and (arm `G`) a from-scratch reference at a quarter of the pairs.
- Nothing after stage 1b is briefed. The stage I had sketched, an admissible rule learning the interface into a hand-wired module, answers a question the programme no longer has: the learner makes its own circuit (3.1).
- Taught fractions below 0.10 are worth adding only if row 1 occurs. If they are added, score held-out pairs in two groups (both operands seen in that position in that task, or not): at f = 0.03, 1–10 of the 89 values never appear in a given position of a task; at 0.05, 0–2 (`python3 -I $R/e35_low_fraction_coverage.py`).

## 9. Standing readings

For every report that contains a life or a sequence of families:

1. Each earlier family's held-out score at every stage, and for add and sub the split by wrap on all pairs without letters. `life_answers_on_birth_tasks.py` is the independent numpy implementation; the worker's reader is the torch one.
2. Beside any online score, a blind predictor that answers from recent history without reading the operands, and a second score for each family after the stream has left it. Both come from the second literature report (section 10).
3. For each arm, the stored numbers beyond the weights and the rehearsal share of each batch.

## 10. What the two literature reports add

Mr. Xu commissioned two literature reports today. They are not in the repository. What bears on this memo:

- No rule that avoids end-to-end backpropagation beats a tuned backprop baseline on held-out compositional or length tests in a recurrent network. The most careful recurrent comparison (Marschall et al., 2020) is negative.
- The large out-of-range effects in the literature come from fixed structure, and every interface into it was hand-coded or backpropagated. Local rules have learned such an interface only where the fixed module's own state was the teacher, and never with a test on untaught combinations.
- Under matched compute, plain replay is the baseline to beat in online continual learning. All of that evidence is vision with backprop; no budget-matched comparison exists on sequence streams or with a learner that avoids backprop.

The reports' sources were read through a text extractor and I opened none of them. Nothing in sections 3 to 9 rests on them except item 2 of section 9.

## 11. Withdrawn from my notes of this afternoon

If Mr. Xu relayed them:

- the two-candidate plan for a later stage of E35 (the sweep through the module; a teacher-driven rule with a fixed read-out);
- the stream experiment with the module arm against the GRU with and without a replay store. Job B is that experiment in its first form, with the learner's own structure in place of the module.

## 12. My expectations, on record before any run

| | Expectation | Chance |
|---|---|---|
| A | Restored by row 256 in two of three seeds, sweep lives | 35% |
| A | Restored by row 1,024 in two of three seeds, sweep lives | 65% |
| A | Wrap score at least 30 above the reference's at row 1,024 (restated in revision 3) | 70% |
| C | A saving of at least 15 at 131,000 items, under each rule | 85% |
| C | A saving of at least 15 at 262,000 items | 60% |
| B | `none` keeps the birth family | 20% |
| B | `anchor` keeps it | 30% |
| B | `anchor10` keeps it | 60% |
| B | Some anchor keeps and acquires | 40% |
| any | A reading that differs between sweep and backprop | 15% |
| E34 | T050 learns in range and extrapolates on both new seeds | 40% |
| E34 | T050 does so on at least one new seed | 75% |
| E34 | Backprop with the gain at tau 0.5 is below 70 at 12 letters at row 8,192 (plain backprop: 89.3) | 60% |

## 13. Limits

- Three seeds, one pass over saved files, lives designed for another purpose.
- 3.1 describes the form of the code and is blind to whether it holds the right answer; 3.3 covers that.
- 3.2 is one seed.
- 3.3 scores all pairs with no letters; 3.4 is the receipts' held-out panel at 12 letters. The two agree in direction and are not the same number. Since revision 3, 3.3 counts whole answers; its last column and the script `life_answers_on_birth_tasks.py` count the two digits.
- A linear read-out of the answer from the state at `=` was too weak an instrument (10–37% where the network itself is 96–99% right), so nothing here uses it.
- 3.6 is arithmetic. No run has yet separated decay from rewriting.
- CPU figures are scaled from E26's receipts (557 CPU-seconds per 4,096 birth updates; about 840 per life block with scoring) and are mine.

## 14. E33 addendum 1 and E34 stage 1, read after the first version of this memo

Both are on `claude/vibrant-curie-ivmrt1` at ba6f5d52 and not yet on the default branch.

### 14.1 The check

My numpy reader reproduces every endpoint the two reports give, from the committed final files and rows 16,384 and 20,480 (seven births, 21 files). Rows before 16,384 are not committed and are taken from the worker's tables.

| Birth | Endpoint at 12 / 20 / 40 letters, mine | The report's |
|---|---|---|
| E34 T050, 73129 | 83.67 / 82.56 / 79.78 | 83.66 / 82.56 / 79.78 |
| E34 T100, 73129 | 95.33 / 90.11 / 57.56 | the same |
| E34 T200, 73129 | 86.11 / 31.56 / 2.00 | 86.11 / 31.55 / 2.00 |
| E33 A+D, 93683 | 86.00 / 76.67 / 29.11 | the same |
| E33 A+OS, 93683 | 97.44 / 97.11 / 94.89 | the same |
| E33 A+D, 116089 | 87.89 / 75.67 / 31.56 | the same |
| E33 A+OS, 116089 | 93.11 / 91.78 / 86.89 | the same |

```
python3 -I $R/e33a1_e34_check.py
```

### 14.2 What they settle

- **E33.** On three seeds, a sweep birth with answer and distractor targets does not extrapolate to 40 letters (23.8, 29.1, 31.6) and one with answer, operand and task-code targets does (94.2, 94.9, 86.9). The question E33 asked is answered.
- **E34.** On seed 73129 the confidence gain at tau 0.5 learns in range and extrapolates with every byte a target and no answer mask. Tau 1 is intermediate and tau 2 fails. Outcome row 1; a candidate on one seed.
- **Amount, not only kind.** E33 left open whether distractor targets hurt by their kind or by their weight. E34's three arms give a dose: a distractor position's weight relative to an answer position's ends at 0.001, 0.033 and 0.19, and accuracy at 40 letters at 79.8, 57.6 and 2.0. One seed.

### 14.3 What they change

1. **Nothing in jobs A, B and C.** They stay on answer targets and E26's files, one change at a time.
2. **E34 stage 2 runs as briefed** (T050 on seeds 93683 and 116089), with two additions.
   - **One backprop birth with the same gain at tau 0.5, seed 73129** (`local = False`, otherwise arm T050; about 0.8 CPU-hours). The first literature report calls this control essential, and I had made it conditional on a sweep arm passing the backprop reference. The final row now does (99.3 against 84.7) and the endpoint does not (79.8 against 82.0), so run it rather than argue which applies.
   - **Report the row at which each arm first reaches 70 at 12 letters**, beside plain backprop and the plain sweep. On seed 73129:

     | Arm | 4,096 | 8,192 | 12,288 | 16,384 | 20,480 | 24,576 |
     |---|---|---|---|---|---|---|
     | T050 | 2.3 | 6.3 | 17.0 | 78.3 | 73.3 | 99.3 |
     | T100 | 17.7 | 85.0 | 91.7 | 95.0 | 91.7 | 99.3 |
     | Backprop, every byte, no gain | 73.7 | 89.3 | 95.0 | 91.3 | 93.3 | 93.7 |
     | Sweep, every byte, no gain | 15.7 | 56.7 | 61.3 | 66.3 | 37.0 | 75.3 |

3. **The price of tau 0.5 is acquisition.** T050 needs about 16,000 updates to learn in range. Plain backprop is at 74 by 4,096 updates and 89 by 8,192, and tau 1 is at 85 by 8,192. An answer the network cannot yet predict has high entropy and so almost no weight; the gain holds back exactly what is new. In a life, where items needed for a new family are the currency (Job C), every new family would start that way. So the gain does not go into a life until stage 2 has reported and this cost has been measured there; the brief for a mask-free life has to carry a row-by-row acquisition score for each new family, not only an endpoint.
4. **The endpoint hides a swing.** T050's last three rows at 40 letters are 78.0, 62.0 and 99.3. Stage 2's sentence should give the three rows beside the mean for each seed.

### 14.4 What I did not check

- Rows before 16,384, the keeper counts, the weight-ratio tables and the checks in `E34_CHECKS.json`.
- Whether a temperature between 0.5 and 1, or one that falls during the birth, keeps the extrapolation without the delay. That is a new arm and I do not propose it before stage 2 reports.

---

# Part II. The programme from here: a continual learner judged over a long life

## 15. The audit of E36, E37 and E38, and two errors of mine

The audit is `AUDIT_E36_E37_E38_BRIEFS_20261115.md` in this folder; its checks are `e36_e38_brief_checks.py` (33dcd187).

- Every factual claim in the three briefs that could be tested holds, including the batch identity of a life's first block, which was rebuilt and hashed against the receipts of all nine lives.
- Six changes are required, all worded to be applied as written; none needs a re-audit.
- **Two of the six correct marks that I proposed in this memo**, not anything the coordinator did. Both failed for one reason, which section 3.3 should have told me: a life that loses the wrap and nothing else still scores about 50 on the held-out panel. So a reminder starts at 13 to 37 and my "30 above the reference" could be met with nothing relearned, and my "loses it at 50" sat on the case of interest. Both marks now read the wrap pairs.
- One required change is a real defect in the code path: the K runner on the default branch imports the other worker's `arch31`, whose `forward` has a different signature.

## 16. Mr. Xu's further statements of 15 November

Verbatim, with my translation.

- **On the nature of the problem.** "从哲学角度上来说，最朴素的反向传播只对每个样本进行同样的学习，通过误差进行逐级修正，网络结构是恒定的，虽然可能在RL或其他阶段只是激发网络的一部分。meta learning可以学习最优的某个网络结构，但一般也认为训练的objective是确定的才能做到。但在持续学习情况下，样本之间存在内在结构，学习本身会帮助调整网络结构，不断去自动优化学习样本之间的关系。作为代价静态的backpropagation无法大规模使用。因此我们回到一个全局元学习层次的优化问题。"
  - *Translation:* naive backpropagation applies the same learning to every sample and corrects by error, layer by layer, in a constant network, even if in reinforcement learning or at other stages only part of the network is active. Meta-learning can find a good structure, but is generally taken to need a fixed training objective. In continual learning the samples have structure among themselves, and learning itself helps adjust the network's structure, so that the learning of those relations keeps being optimised. The price is that static backpropagation cannot be used at scale. So we are back at an optimisation problem at the level of global meta-learning.
- **On direction and pace.** "你可以自己决定实验的快慢，不用等我决定。因为此前目标出现错位，我们先讨论下一步如何做。我们坚持做持续学习。"
  - *Translation:* you (the reviewer) may set the pace of the experiments yourself, without waiting for my decision. Because the goal was misaligned before, we first discuss how to proceed. We stay with continual learning.
- **On what the measure must be able to show.** "但是这个还是不能回答问题，因为显然可能出现我们的学习者一开始比传统的学习者更慢，但是在持续学习过程中最后做到“后来居上”的问题。只要我们的学习者还在学习，而传统的学习者卡住了，就不能排除这一点。一天上千个是个好办法，但无法处理这种情况。"
  - *Translation:* this still cannot answer the question, because our learner may be slower than the conventional learner at first and overtake it later in the course of continual learning. As long as ours is still learning and the conventional one is stuck, that cannot be excluded. A thousand lives a day is a good method, but it cannot handle this case.

The third statement was made against my first proposal (one score averaged over a life; many short lives). That proposal is withdrawn. Section 17 is what replaces it.

## 17. The programme

### 17.1 The question

In one stream read once, at fixed memory and fixed computation per item, without full backpropagation: is there a learner whose performance late in a long life exceeds that of naive backpropagation, and of backpropagation with replay, at the same budget? A slow start does not count against it.

What we measured until now could not answer this. It was accuracy on fixed panels after a fixed dose, with backpropagation as the ceiling, so the best outcome was parity. A score averaged over a short life cannot answer it either, for the reason Mr. Xu gives.

### 17.2 The world (requirements; the specification is yours)

1. **One stream per life, read once.** Items in the present format. Answer bytes are the targets, as in E26. Every-byte targets enter later as a candidate, after E34 has reported.
2. **Tasks.** A task is an operation and a modulus with a code of its own. The generator supplies more distinct tasks than the longest life meets for the first time, from one parametric space, so that tasks share structure. Codes are arbitrary; a code that names the modulus is a later variant.
3. **Visits.** The stream is a sequence of visits. A visit presents a small group of tasks for a fixed number of updates. New groups keep arriving through the whole life; old groups return after short, medium and long absences. The schedule comes from the world's seed and is the same for every arm.
4. **Taught and asked items.** Each task's operand pairs are split once. Taught pairs appear with their answer and the learner is updated on them. Asked pairs appear without their answer, at a fixed rate inside their task's visits; the learner's answer is scored and nothing is learned from them. An asked pair is never taught.
5. **Size.** The same kind of network, narrower; smaller moduli; fewer letters. Small enough that a life of 1,000 visits costs about 6 CPU-hours at most. I estimate that the present scale would need more than ten days for such a life; neither figure is measured.

### 17.3 What is recorded for every life

- **Score by age.** Accuracy on asked items, whole answers, in consecutive bins of the stream, with a blind predictor beside it.
- **Cost of the n-th new task.** The taught items of that task shown before its asked items first reach 80% over a running window; or "not reached", with the level at the end of the visit.
- **Kept at return.** Accuracy on the first asked items of a task when it returns, against the length of its absence, and for add and sub split by wrap; then the taught items needed to reach 80% again.
- **The ledger.** Stored numbers by kind (weights, optimizer state, anchors, buffers); updates; stream items consumed per update.

### 17.4 What is compared

- **Late level:** mean score over the last quarter of the life.
- **Trend:** the slope of score against the logarithm of age over the second half, and the slope of the logarithm of cost against the logarithm of n.
- Everything is paired by world, with intervals across worlds. No fixed thresholds in points are set before the calibration job shows the scale of the numbers; my two errors in section 15 came from setting such thresholds without looking.
- **"Stuck"** means that over the second half of the longest life run, neither the score's slope nor the cost's has an interval on the improving side of zero. **"Still learning"** means that at least one has.
- **"Overtakes"** means behind on late level at one length, ahead at a longer one, and still ahead at the next.

### 17.5 The horizon rule

- Lives are run at lengths that grow by a factor of four: 64, 256, 1,024 visits, and further where needed.
- A comparison of two learners is closed only when the sign of the late-level difference is the same at two successive lengths and the trend difference does not point the other way. Otherwise that pair goes to the next length.
- Every statement carries its horizon: "up to 1,024 visits". No finite run excludes overtaking beyond its horizon; what can be stated is the trend at it.
- When settings are chosen, a setting goes on to the next length if it leads on late level **or** on trend. This is what keeps a slow starter from being dropped.

### 17.6 Arms and budget

- All arms have the same network, the same world and the same number of updates, and every update consumes the same number of stream items.
- `naive`: backpropagation on the taught items of each batch.
- `replay`: backpropagation with a buffer of raw past items. A share of each batch is drawn from the buffer in place of new taught items, which are skipped. The buffer holds as much memory as one copy of the weights, which is what the anchor of E38 stores; its share of the batch is tuned.
- `sweep` and `sweep+replay`: the same under the sweep. Sections 3 and 4 found no difference between the rules at this level, so these two serve as the check that the small world agrees.
- Each arm's own learning rate and decay, and for replay its share and size, are chosen on tuning worlds. Comparisons are made on other worlds.
- **Candidates come after job L2.** They are the sweep with the three controls of section 4: which weights may move (the anchor of E38, then a slow copy of each weight, then a step size per weight); when to rehearse (short reminders, if E36 shows savings); which samples teach (the confidence gain, after E34). At first each control is a fixed number per life. The step in which a control depends on the learner's own state is the one Mr. Xu's first statement in section 16 describes, and it is last because it needs the others as its baseline.

### 17.7 Jobs, in order

| Job | What | Size | Audit |
|---|---|---|---|
| L0 | The protocol brief: world generator, runner, scorer, ledger, the recorded quantities of 17.3 | new code | once, by me |
| L1 | Calibration: 8 worlds, lives of 16 visits, `naive` and `sweep` | about 4 CPU-hours | none |
| L2 | Does the conventional learner get stuck? `naive` and `replay` at 64, 256 and 1,024 visits; `sweep` and `sweep+replay` at 64 and 256; `naive` at a second width; the probes and the agreement between lengths of 17.10 | cap of 48 CPU-hours | none |
| L3 | Candidates, chosen by 17.5 | set after L2 | none |
| L4 | The winning setting on the E26 families at full scale | as E38 | as now |

- **L0, second implementation.** I write the world generator and the scorer in numpy from the brief's text alone. Before any roster, the two implementations agree on the digests of the first batches of three worlds and on the scores of saved weights.
- **L1, gate.** Under `naive` and under `sweep`, the first group's asked items are at least 90% right by the end of its first visit in six of eight worlds, and the measured cost puts a 1,000-visit life within about 6 CPU-hours. If either fails, change the size and repeat. Reported and not gated: the wrap pairs at a task's first return, and the cost of tasks 2 to 16 beside the first.
- **L2, roster.** Eight worlds at the two shorter lengths and four at the longest. If the forecast passes the cap, drop worlds before lengths. No candidate takes part. The reading is 17.4's: stuck or still learning, for each arm.
- **Why L2 comes before any candidate.** Mr. Xu's condition for overtaking is that the conventional learner is stuck while ours still learns. Whether the conventional learner gets stuck in our world is not known. Our longest lives so far have six to nine stages. In the one study the second literature report cites on this, backpropagation fell from 89% to 77% over 2,000 tasks, and it took that length to see it.

### 17.8 Pace

Mr. Xu has left the pace to me. Inside the small world it is this:

- One audit, of L0. After it, a roster inside the protocol needs a one-page note (arms, worlds, lengths, predictions on file) and no audit.
- Any change to the generator, runner or scorer is audited, and so is any claim at full scale.
- At least eight worlds per setting up to 256 visits; paired differences with intervals, in place of marks on three seeds.
- I spot-check a sample of saved lives with the numpy scorer.

Full-scale work (E36 to E38, E34, L4) keeps the present discipline.

### 17.9 What can be concluded

- A result in the small world is a claim only after L4.
- If no candidate passes `replay` at any length and the trends do not differ, the result is that replay is the answer at this scale and that our learner equals it. That is reportable and is not the advance Mr. Xu asks for.
- If the conventional learner is not stuck at 1,024 visits and no candidate's trend is better, the statement is "no overtaking up to 1,024 visits and no trend toward it", not "no overtaking".

### 17.10 How short and long runs are combined (added in revision 4)

Short runs are many and varied; long runs are few and are the real measure. This is the long-delay problem of reinforcement learning, and the programme uses its standard answers: predict the long outcome from quantities measured early, learn the laws and compose them, and do not restart what can be continued.

1. **Short runs measure rate laws; long runs test whether the laws compose.** Three laws describe a learner at any age: what remains of a task after an absence of a given length; what a new task costs; what a reminder restores. E36, E37 and E38 measure these at full scale. Before a long life is run, its curves are predicted from the laws measured in short runs, the prediction is filed, and it is scored afterwards. Where the prediction fails is where a long-horizon effect lives.
2. **Trunks and branches.** A few long lives (trunks) save their complete state at ages that grow by a factor of two: weights, optimizer state, buffers and the position of the stream. From any saved state a short run (a branch) continues with one thing changed and the same future stream. Two branches from one state differ only by that change, so they are compared item by item. This gives many varied short experiments at old ages for the cost of one long life. The harness already has a qualified resume for the older runners (9.112 H); the new runner needs it from the start.
3. **Two probes, taken as branches at every saved age.**
   - *The learning probe:* a fixed set of tasks that no world's stream ever uses is taught for a short dose; the taught items needed to reach 80% are the learner's ability to learn at that age, on the same tasks at every age.
   - *The keeping probe:* asked items of tasks that have been absent for known lengths.
   - These are the candidates for leading indicators. L2 shows whether their values at 64 and 256 visits predict the late level at 1,024.
4. **Measure how far short runs can be trusted.** L2 reports the agreement between lengths: of the arms and settings run, how often the order at 64 visits is the order at 1,024, on late level, on trend and on each probe. Selection at short length (17.5) then uses whichever quantity agrees best. If none agrees, short runs cannot choose among learners in this world, and the budget moves to trunks and branches.
5. **Time in the learner's own units.** The brief lists the learner's own time constants and states visit and absence lengths as multiples of them. One is known: decay shrinks an unsupported weight with a constant of 1 / (learning rate × decay), 4,000 updates at the present settings. L2 runs `naive` at two widths, to test whether the age at which anything changes scales with capacity. If it does, a narrower network reaches old age sooner and long horizons become cheaper; if it does not, width is only a cost setting.
6. **What cannot be shortened.** A branch does not show an effect that needs a setting to have been in force since birth, or the avoidance of a state that cannot be undone. Every candidate that survives selection therefore gets full-length lives of its own; branches and probes decide which candidates those are.

### 17.11 What the audit of E39 changes in this Part (added in revision 5)

The audit is `AUDIT_E39_SMALL_WORLD_PROTOCOL_20261115.md`. Its second implementation of the generator reproduces the coordinator's three digests. It requires seven changes to the brief, the first of which is a real defect: the specification file has the start and end symbols the other way round from the harness.

Three of its findings change what sections 17.3 to 17.7 say:

1. **The trend that is compared is the first-visit level, not the cost.** Sections 17.3 and 17.4 name the cost of the n-th new task and its slope. In the brief's schedule a task met late has fewer visits left (groups first met in the four quarters of a 1,024-visit life get 9.3, 4.8, 3.5 and 2.3 visits), so among late tasks only the cheap ones reach the criterion. That pushes the cost's slope toward the improving side for any learner, which is the result L2 looks for. Every task has one whole first visit, so the level it reaches there is free of that bias. The cost and its slope are still reported and are not used in any comparison.
2. **The calibration gate is replaced, and calibration lives have 64 visits.** The coordinator's objection to my gate was right: "90% by the end of the first visit" asks a 256-update birth of a network with nothing behind it, and L2 does not need it. The gate is now that the first group reaches the criterion within the life, that the median first-visit level of later tasks is between 0.10 and 0.90, and the cost bound. The two bounds say that the instrument is off its floor and its ceiling; they were set before any data, as my two wrong marks were, and L1 will show whether they are sensible. Lives of 16 visits cannot show the second condition, since they meet one to five groups.
3. **The horizon is bounded by tasks as well as by compute.** With 250 usable groups and a fifth of visits new, a life runs out of new tasks at about 1,250 visits. A length beyond 1,024 visits needs a larger task space, which is a change to the generator and one more audit of it.

Two further points for the record:

- A life of V visits is the exact start of any longer life in the same world and arm. L2 therefore runs each arm once at its longest length and reads the shorter lengths from the same file; section 17.7's roster of three lengths costs one.
- The probe tasks of 17.10 are reserved per world (its last 20 groups), not across worlds, because that leaves every schedule and digest unchanged.

## 18. E34: two long births, with the present code

E34's passing arm is the one case on file of the pattern Mr. Xu describes: behind for 12,000 updates (17% against 95% at 12 letters), ahead at the last row (99.3 against 84.7 at 40 letters), on one seed, with the last three rows still swinging (section 14.3). The test of that pattern is a longer run, not more runs.

- **Runs.** Arm T050 and plain backpropagation with every byte a target (E28's B-all), seed 73129, 98,304 updates each, four times the present dose. Rows every 4,096 updates. About 6.6 CPU-hours by my estimate.
- **Check.** At row 24,576 each run has the tensor hash of its existing final file, since the stream and the update are unchanged.
- **Reading, at 40 letters, over the 13 rows from 49,152 to 98,304.** "Overtakes and stays": T050's mean is at least 10 points above backpropagation's and it is above in at least 10 of the 13 rows. "Does not": its mean is not above. Otherwise mixed. The rows at 12 and 20 letters are reported beside them.
- **Then.** If "overtakes and stays", the same two runs on seeds 93683 and 116089. The backpropagation birth with the gain (addendum 1) is run at this length only if its present-length endpoint differs from plain backpropagation's by 15 points or more at 40 letters.
- **What it cannot show.** This is a stationary stream, so it bears on the pattern and not on continual learning as such. One seed.

## 19. E36, E37, E38 so far, E35-B and the long E34 births (revision 6)

Read on the workers' branches at 22:16 UTC on 15 November: `claude/vibrant-allen-8hziln` (E36), `claude/magical-pascal-sflem4` (E37), `claude/peaceful-hopper-8ofq4i` (E38, in progress), `claude/busy-darwin-dlrwtm` (E35-B), `claude/vibrant-curie-ivmrt1` (E34, in progress). None is on the default branch yet.

### 19.1 The check

From the committed weights, with my reader: `python3 -I $R/e36_e37_check.py`.

| Figure | Mine | The report's |
|---|---|---|
| E36, wrap score at row 1,024, nine reminders | S 93.2, 88.1, 96.0; B 97.4, 99.2, 95.0; K 94.6, 99.0, 91.9 | the same |
| E36, the same for six fresh references | S 1.1, 1.3, 1.1; B 1.1, 1.4, 2.5 | the same |
| E37, fresh F3 at row 4,096 | S 36.5, 47.1, 53.5; B 64.2, 59.6, 73.5 | the same |
| E37, fresh F1 at row 6,144 | S 67.0, 81.4, 67.6; B 80.4, 83.5, 78.9 | the same |

E38's rows are not committed and E35-B's weights were not recomputed; their figures below are the workers'.

### 19.2 What they say

- **E36: what a life removes comes back in about a hundred updates.** In all nine reminders the birth family's held-out score is back above 85 at row 64 or 128, and the wrap pairs go from under 1% to 79–93% by row 128. A fresh network taught the same batches is at about 1% after 1,024 updates. Table row 1 for all three rules.
- **E36, the other direction.** Sixteen updates on the birth family alone take 40 to 60 points off the life's families, and the compositions go to zero within 256. The modulus in force follows whatever the stream last taught.
- **E37: a learner that has lived needs far fewer items for a new family.** At 131,000 items of family F3 the lived learners are at 77–94 and fresh ones at 7–19, under all three rules; at 262,000 the difference is 17 to 47 points. For F1, learned straight after birth, the difference is 17 to 28 under the sweep and 8 to 16 under backpropagation. Section 3.5 is a saving.
- **E38 so far (seed 73129; sweep arms and the backpropagation `zero`).** The wrap pairs fall from 97% to 18% by row 64, 3% by 256 and 0% by 1,024 in `zero`, `none`, `anchor` and `anchor10` alike, while the new family reaches 90–97. So decay is not needed for the loss, and a uniform pull toward the birth weights does not prevent it at either strength. `rehearse` is at row 64 with the wrap pairs at 65%. The backpropagation arms and the rest of `rehearse` are still running.
- **E35-B, first stage (seed 73129).** With parsing supplied the modulus-free circuit generalises from a tenth of the pairs (99% against 9–34% scrambled). From the byte stream a GRU under backpropagation does not come to use it, forced or not: held-out add and sub 19.9 with the module, 18.0 forced, 19.6 for the GRU alone, which has memorised (98% on taught pairs). Outcome row 3; the stop rule waits for the forced arm on two more seeds.
- **E34, long births, to row 32,768 of 98,304.** Both runs have the short runs' hashes at row 24,576. At 40 letters the passing arm reads 99, 95, 88 at rows 24,576 to 32,768 and plain backpropagation 85, 89, 90. Plain backpropagation is still improving past its old endpoint. Nothing is read until row 49,152.

### 19.3 What this changes

1. **A family is two things with very different lifetimes.** The shared arithmetic lasts and accumulates: it is why a reminder takes a hundred updates and why a new family takes a fraction of the items. The family's own modulus is rewritten by a few dozen updates on anything else. Sweep, Kolen–Pollack feedback and backpropagation agree on both.
2. *(Withdrawn in revision 7; see section 20.)* **Among Part II's controls, rehearsal moves ahead of anchors.** A uniform anchor failed at full scale on one seed under the sweep. What E36 measured, that restoring costs about a hundred updates, makes short, well-timed reminders the first candidate. A pull weighted by each weight's use for the old family remains a candidate; a uniform one does not.
3. **The rate laws of section 17.10 now have first values at full scale:** loss within 64 to 256 updates of another family; restoration within 64 to 128 updates; a new family at 77–94 after 131,000 items for a lived learner against 7–19 for a fresh one. L1 should report the same three quantities in the small world, so that the two scales can be set side by side.
4. **Learned structure has now done what the innate circuit did not.** The comparison is not like for like (a quarter of the pairs after a life, against a tenth at birth), which is what the job below is for.

### 19.4 One new job: E37 at a tenth of the pairs

**Question.** A fresh GRU taught a tenth of the pairs memorises: 98% on taught pairs and about 20% on held-out add and sub after a full birth (E35's arm `G`). Does a learner with a life behind it generalise from the same tenth?

- **Start.** The nine life-end files of E36 (S, B, K), and fresh networks under the sweep and under backpropagation; three seeds. Fifteen runs.
- **Family.** One the lives never met: add, sub, max and min at a new modulus (83 is free), with code bytes no family uses. A tenth of the pairs taught.
- **Stream.** `X16.entry_stream` on that family's entries, with a generator seed no other stream has used; the same batches for every run of a seed.
- **Dose and rows.** 8,192 updates; weights at 512, 1,024, 2,048, 4,096 and 8,192.
- **Scores at every row.** Held-out add and sub at 12 letters; the same on taught pairs, to see memorising; all pairs without letters split by wrap; the earlier families at 12 letters, reported and not read.
- **Reading.** As E37: a saving if lived minus fresh is at least 15 points at equal items in two of three seeds, per rule. Beside it, as a fact and not a mark: each run's held-out score at the row where its taught-pair score first passes 90.
- **Checks.** As E36: row 0 has the life-end file's hash; batch digests equal across the runs of a seed; the update path against E26's dry births; the K step as E36 wrote it.
- **What it cannot show.** One new family; lives made for another purpose; a fresh optimizer in every run.

My expectations, on record: fresh runs at or below 30 on held-out add and sub at every row, 85%; lived sweep runs at or above 60 at row 8,192 in two of three seeds, 50%; a saving under each rule, 75%.

## 20. On returning to retention schemes (revision 7)

Mr. Xu, 15 November: "Earlier many minifly period attempts tried to use slow learner, long term storage, and other fancy schemes, but these yield at best mixed results. So we are revisiting the issue to some extent."

He is right, and item 2 of section 19.3 was a step in that direction. The record on such schemes:

| Scheme | Result | Where |
|---|---|---|
| Connectome learner with fast, slow and adaptation state (V66–V78) | 62.6% against 93.3% for a simple reference; nothing promoted | review, section 2 |
| Eight-store persistent core | taught items 97.1%; effect on held-out items −8.2 points | section 2 |
| Complementary-learning-systems replay (v3–v6) | best held-out 13%; 0% in v6-R1 | section 2 |
| Fast and slow two-store | no gain over a single store | 9.12 |
| Sleep replay; sleep order and scaling | +3 to +6 points; no effect | 9.12, 9.17 |
| Plain replay from a bounded store of raw bytes, on a learner that generalises | 0.81 to 0.98 of the joint-training reference | 9.38, 9.42 |
| Uniform pull toward earlier weights (E38's anchors) | no protection at either strength | section 19.2 |

Three things follow.

1. **The earlier schemes were tested on learners that memorised.** A store or a slow path keeps what the learner has; those learners had nothing that generalised (held-out 0 to 13%). The one audit of that period that named the problem (26 September) found that the goal had shrunk to retention.
2. **On a learner that generalises, the plain store has been enough and nothing more elaborate has added to it.** E38's anchors are one more negative of the same kind.
3. **The new result is not about retention.** It is E37: a learner with a life behind it reaches 77–94% on a new family where a fresh one reaches 7–19%. Whether that ability keeps growing over a long life is what job L2 measures, and it is the quantity the earlier period never had a learner to measure.

So:

- **Retention is not a research question in Part II for now.** Where a store is used it is the plain bounded store of raw items, already qualified in this project. No reminder schedule, slow copy, weighted anchor or second store is a candidate until L2 shows a deficit that the plain store does not cover.
- **A rule for any added mechanism.** It is tested only if (a) the deficit it addresses has been measured on a learner that generalises, (b) the plain baseline is in the same experiment at equal budget, and (c) its expected effect is stated beforehand from the measured rate laws.
- **The headline of every report stays generalisation and its trend with experience**; what is kept at a task's return is reported beside it as a constraint.

## 21. Computation to threshold, and scale as an axis (revision 8)

### 21.1 Mr. Xu's statements

Verbatim, with my translation.

- "scaling的帮助如何呢？同样的任务，如果scaling有帮助，也可以认为在固定算力下我们能做的更多。"
  - *Translation:* how much does scaling help? On the same task, if scaling helps, we can also say that at fixed computation we can do more.
- "对，如果网络计算规模变大，或者数据变多，但是为了达到目标(threshold)需要的训练次数本身变少，结果一样甚至更好，平均下来算力一样甚至更少。那这是一个可以讨论的目标。至于scale which axis是可以通过实验讨论的。"
  - *Translation:* yes: if the network's computation grows, or the data grow, but the number of training steps needed to reach the target (a threshold) falls, with the same or a better result and the same or less computation on average, that is a goal worth discussing. Which axis to scale can be settled by experiment.

### 21.2 The quantity

**Computation to threshold.** For a run, the total computation spent up to the first scored row at which its accuracy reaches a threshold.

- *Accuracy* is on operand pairs never taught, whole answers. It is given twice: on the held-out panel at 12 letters, and on held-out pairs that need the reduction, no letters. The second is the one free of what a network can answer without having learned the task's modulus (section 19.2 and the E40 audit).
- *Thresholds* are 50, 80 and 90%, all three reported, so that no conclusion rests on one number.
- *Computation* is counted in two ways: the number of weights times the number of items trained on, which does not depend on the host; and CPU-seconds on the qualified host, one thread.
- A run that does not reach a threshold within its dose is recorded as "not reached", with its highest accuracy.

A learner or a setting is better, in Mr. Xu's sense, if it reaches the same threshold with the same or less computation, whatever it spent that computation on.

### 21.3 What the record has, and what it lacks

- On the window network of early November, width helped: at width 512 a network from random weights reached 91–94% from a quarter of the pairs, as developed networks did, and collapsed at width 128 (9.7); compositions rose from 37–38% to 52–68% between widths 256 and 512 (9.18).
- For the present GRU there is no measurement at any width but 256. The sizing roster of 11 November was never run.
- So it is not known whether the saving of E37 (a learner that has lived, against a fresh one) shrinks as the network widens. If a wider fresh network generalises from fewer pairs, experience and width are two ways to the same end, and the comparison between them is one of computation to threshold.

### 21.4 Job E41: the same task, three widths, three taught fractions

**Question.** On the birth family, how does computation to threshold change with the network's width and with the share of pairs taught, under each rule?

- **Task.** The birth family (add, sub, max, min mod 89), E26's stream and panels, 0 to 12 letters.
- **Grid.** Width at one half, one and two times the present sizes (every hidden size scaled together; the embedding left at 8). Taught fraction 0.10, 0.25 and 0.50. Sweep and backpropagation. Three seeds. Fifty-four runs, all from random weights.
- **Settings.** Learning rate, decay and batch as E26 at every width.
- **Dose and rows (corrected in revision 9).** 8,192 updates at the widest network, 12,288 at the present width and 40,960 at the narrowest. Scored every 256 updates. My first doses (8,192, and 16,384 at the narrowest) were too short in two ways: a threshold that must hold for three scored rows could not be recorded after 7,680 at the present width, and a narrow run that failed to reach a threshold would have left the comparison of total computation undecided. A run that does not reach a threshold is known only to need more than its dose; the audit gives the rule for when a comparison can still be decided.
- **What is tabled.** For each cell: the row at which each threshold is first reached on each of the two accuracies, the computation to that row in both units, the highest accuracy, and the accuracy on taught pairs at that row.
- **Reading.** None is fixed beforehand; this is a first measurement. The report states, for each rule and fraction, whether computation to the 80% threshold falls, stays within a factor of 1.5, or rises from each width to the next, with the range over seeds.
- **Checks.** At the present width and fraction 0.50 the run has the hash of E26's dry birth at 2,048 updates, under both rules. Batch digests are equal across widths and rules for a seed and fraction. Each width's parameter count is recorded.
- **Cost.** With the corrected doses, about 30 CPU-hours of updates and 5 of scoring by the audit's estimate, a little over half of it at the widest network. One seed per worker divides it evenly.
- **What it cannot show.** Settings tuned at one width are used at all three, so a width may be handicapped by them. Networks with a life behind them exist at one width only, so the crossing of width with experience is left to the small world.

My expectations, on record: at fraction 0.50 the widest network reaches 80% in fewer updates than the present one, 70%; it does so with less total computation, 25%; at fraction 0.10 the widest network reaches 80% on held-out reduction pairs within its dose, 20%; the reading differs between the rules in some cell, 25%.

### 21.5 In the small world

- Width becomes an axis of job L2 for every arm, at three widths, in place of `naive` at one extra width (section 17.10, item 5).
- Every table of L1 and L2 gives computation in the two units of 21.2 beside updates, and the first-visit cost of the n-th task is also given as computation to threshold.
- Comparisons between learners are made at equal total computation with each at its best width, as well as at equal width.
- The crossing that full scale cannot make is made here: computation to threshold for the n-th new task, by width. It says which buys more, experience or width.

### 21.6 What this replaces

In discussion I had proposed adding fresh networks at other widths to E40. Job E41 replaces that: it uses the same task at every point of the grid, which is what "the same task" in Mr. Xu's statement requires, and E40 stays as released.


## 22. Job E42: cells with a limited number of local connections and their own periods (revision 10)

### 22.1 Mr. Xu's statements of 16 November

Verbatim.

- "If a cell A and B has no hyper-connection at all - not in the past, not now, then the weights logically cannot connect A with a cell B that remembers distant history. In other words A does not know this part of the history even if the weight is high. So to some extent the apparent overfitting of weights and hyper-connections during the learning process is the same thing. Can we utilize this?"
- "In theory due to the local learning hypothesis, the neuron S_i can only interact with a pre-specified (perhaps high) number of neurons. So in our setting depending on the update frequency, S_i can add 100 connections in T=10, and add zero at T=11, drop 50 at T=12 etc. Nowhere did we dictate that S_i have to adjust by one connection at a time."
- "Let us presume for simplicity that each cell can connect at most N=10 neurons (after all addition is simple), but we can adjust N and the cell can learn the connection strengths".
- In answer to three questions of mine: "1) both is fine, but ideally the connections can adjust. however a neuron is only allowed to adjust within a pre-defined neighborhood, say via some pre-specified embedding coordinates to mimic biological situation. This is to impose locality. 2) The cell's previous value would be available, and each cell has its own pre-set delays or info processing frequencies. During the period it absorbs info from all other cells that emits to it. So cell at T=0 and T=5 may get inputs from the connections. How to process this will not be pre-determined and will be part of local learning, and you can try some variants. 3) Yes, the encoder and decoder parts can stay."

My three questions were: (1) whether a cell's connections are fixed in advance or change during learning; (2) what counts against the limit, and whether a cell's own previous value is available to it; (3) whether the limit applies only to the cells that replace the GRU, with the input and output parts of the network left as they are.

**For the record of P5.** The clarification of 7 November says that "local" does not mean a fixed small neighbourhood and that which neurons a synapse takes its signals from can be adjusted dynamically. Today's statement keeps the dynamic adjustment and adds that it happens inside a neighbourhood defined in advance. Please enter it beside the clarification.

### 22.2 What the record has

- **Units with pre-set time constants have been tried once, without connections among them.** The two-bank network (`arch17`) had fixed retention rates of 0.5 to 0.999 and no learned recurrence. It did not compose from memory: untaught plus-minus pairs 1.5 / 0.5 / 7.5, against 73.6 / 78.8 / 75.1 for the GRU; one added feedback matrix gave 20.2 / 1.1 / 1.0 (review 9.101).
- **The variants asked for on 13 November were never run.** E29 part B (slow units and delays at birth) was withdrawn after its audit and not rewritten (9.118). This job takes its place.
- **The trained GRU makes its own holding units:** 41 to 46% of units after a birth, none before (9.120 C).
- **I found no experiment that limits the number of connections of a unit, places units in a space, or lets connections change.** I searched the review for the words sparse, prune, rewire and neighbour; the hits are the mushroom-body store and the fly circuits of October, not this learner.

The present job differs from the two-bank network in three ways: the cells are connected to each other with learned strengths, the gates read the state, and a period is an interval between updates, not a rate of decay. Because the earlier design failed, the periods enter as one separate step (22.4), so that their effect is read apart from that of limiting the connections.

### 22.3 The learner

Everything outside the recurrent layer is the present network, unchanged: the byte table, `l1`, the input map into the layer, and the readout `l2`, `lm`, `l3`. The layer keeps 256 units, now called cells, and the GRU's own equations. Three things change: which cells a cell reads, when a cell is updated, and how it combines what arrived since its last update.

```
positions      cell i (0..255) stands at row i // 16, column i % 16 of a 16 x 16 grid with wrap-around
neighbourhood  B_i: the 48 cells j != i whose row and whose column each differ from i's by at most 3, with wrap-around
connections    A_i: N cells of B_i; the limit is used in full (a connection given up is always replaced); cell i also reads its
               own previous value, which is not counted in N
period, phase  p_i from the configuration's set, f_i in 0 .. p_i - 1; cell i is updated at positions t = f_i, f_i + p_i, f_i + 2 p_i, ...
               (t counts from the start symbol at position 0); at every other position its value does not change

at every position t, as arch26.forward does, with the recurrent matrix masked:
    gi(t) = Wi z1(t) + bi                       the input map, unchanged: every cell reads the whole current byte
    gh(t) = (Wh * mask) h(t-1) + bh             mask[i, j] is true for j in A_i and for j = i, the same for the three gates
at a position where cell i is updated, with G and H made from gi and gh over its period (table below):
    r = s(G_r + H_r),  z = s(G_z + H_z),  n = tanh(G_n + r * H_n),  h_i(t) = (1 - z) n + z h_i(t-1)
elsewhere h_i(t) = h_i(t-1);  h(-1) = 0 for every item
readout, unchanged:  z2 = tanh(l2 z1 + lm h),  logits = l3 z2
```

**How a cell combines what arrived during its period.** For a cell of period p updated at position t, the steps of its period are t, t-1, ..., t-p+1. Positions before the start symbol contribute nothing. Four variants; a cell of period 1 is the same cell in all four.

| Variant | G | H | Added parameters |
|---|---|---|---|
| `last` | gi(t) | gh(t) | none |
| `mean` | the mean of gi over the period | the mean of gh over the period | none |
| `step` (the default) | sum over k of a_i[k] gi(t-k) | sum over k of a_i[k] gh(t-k) | a_i: one weight for each step of the period, starting at 1/p |
| `source` | as `step` | sum over k of a_i[k] times [(Wh[k] * mask) h(t-k-1) + bh]: a separate strength for every connection at every step of the period | Wh[k] for k = 0 .. p_i - 1, each starting as a copy of the masked, scaled Wh; a_i as in `step` |

`mean`, `step` and `source` compute the same function at the start. `source` is the variant in which a cell can read each of its sources at its own distances in time, up to its period.

*Corrected in revision 11.* I first wrote that each Wh[k] starts at the masked Wh divided by the period. That put the factor 1/p inside the weights, where the sign rule's single scale cannot see it: under the sweep the error sent back would have been 0.40 of the right size for cells of period 1 and 3.2 times it for cells of period 8 (`e42_brief_checks.py`, part F). With the factor carried by a_i, as in `step`, the ratio is 1 for every period.

**Start.** `arch26.new_net(seed)` unchanged; then periods, phases and connections are drawn; then the recurrent weights outside the mask are set to zero and those inside are multiplied by sqrt(256 / (N + 1)), so that a cell's recurrent input has the variance it has in the present network (4.82 at N = 10; exactly 1 when every cell is connected to every other).

**Draws.** None uses torch's generator, so the initial weights and the data are those of E26.
- Periods: `order = list(range(256)); Random(f"e42|periods|{seed}").shuffle(order)`; the first cells of `order` take period 1, the next period 2, and so on, in the counts of 22.4.
- Phases: `rp = Random(f"e42|phases|{seed}")`; for i in index order, `f_i = rp.randrange(p_i)`.
- Connections: `rc = Random(f"e42|connections|{seed}")`; for i in index order, `A_i = sorted(rc.sample(sorted(B_i), N))`.

**The rules.** Arm B: exact gradients. Arm S: every linear map sends its error back through the signs of its weights times one scale per stacked matrix, as now. For a masked matrix the scale is the mean magnitude of the weights inside the mask, so that the error sent back through an existing connection has the size of an existing weight. In `source` the matrices Wh[k] are stacked into one for this purpose. When every entry is inside the mask, `exp3_state.lin` is called unchanged. The gates' nonlinearities, the products, the weights a_i and the holding of a value between updates are differentiated exactly in both arms, as the gates are now. Credit runs through the whole item.

**Changing connections** (the configurations marked "changing"). Every 256 updates, after the update and before scoring:
1. The strength of a connection is the sum of the magnitudes of its weights, over the three gates and, in `source`, over the steps.
2. A connection of cell i is given up if it has existed for at least 512 updates and its strength is below theta times the mean strength of cell i's N connections. A cell may give up none, or several.
3. Each connection given up is replaced by one to a cell of B_i that is not connected to i at that moment and was not given up at this event. Cells in index order, connections given up in increasing order of source: `rr.choice(sorted(candidates))`, with `rr = Random(f"e42|replace|{seed}")` created once per run.
4. The new connection's weights are zero, so the network's output is unchanged by the addition. The weights of a connection given up are set to zero. The optimizer's two running averages are set to zero for every entry that changed.

A cell that satisfies the locality hypothesis has no information about a cell it is not connected to, so the new partner is drawn at random and is kept or given up later by its strength. theta is set once in stage 0 (22.6, check 10).

### 22.4 Configurations

Stage 1, under the sweep and under backpropagation, three seeds: 24 runs.

| Name | Connections | N | Neighbourhood | Periods | What the step changes |
|---|---|---|---|---|---|
| `G` | every cell to every other | 255 | all | all 1 | nothing: the present GRU, run by the new code |
| `F10` | fixed | 10 | 48 | all 1 | the limit and the neighbourhood |
| `A10` | changing | 10 | 48 | all 1 | connections may change |
| `P8` | changing | 10 | 48 | 1, 2, 4, 8; variant `step` | periods |

Stage 2, under the sweep only, three seeds: 24 runs. Each differs from its reference in one thing.

| Name | Differs from | In |
|---|---|---|
| `N5`, `N20` | `A10` | N = 5; N = 20 |
| `W10` | `A10` | the neighbourhood is all 255 other cells: the limit without the locality |
| `P4`, `P16` | `P8` | periods 1, 2, 4; periods 1, 2, 4, 8, 16 |
| `P8-last`, `P8-source` | `P8` | the variant |

`P8-mean` is deferred (revision 11): it is `step` with its weights held fixed, the least informative of the variants, and the cost is higher than I estimated. Stage 2 is then 21 runs.

Cells by period: half of the cells have period 1 in every set with periods, so the number of cells that follow every byte is the same across them.

| Set | Period 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 1, 2, 4 | 128 | 64 | 64 | | |
| 1, 2, 4, 8 | 128 | 64 | 32 | 32 | |
| 1, 2, 4, 8, 16 | 128 | 64 | 32 | 16 | 16 |

Parameters: 1,124,113 for `G`. With N = 10 the recurrent matrix keeps 8,448 of its 196,608 weights and the network has 935,953 (932,113 at N = 5, 943,633 at N = 20), plus 384, 512 or 640 weights a_i in the three sets (`e42_design_checks.py`, part C).

### 22.5 Data, dose, scores

- **Data and settings:** E26's birth exactly, as E41's present width at taught fraction 0.50: the four operations mod 89, 0 to 12 letters, answer bytes as targets, batch 64, `exp26_candidate.make_opt` with the new parameters in the decayed group, learning rate 0.001, decay 0.25.
- **Dose:** 24,576 updates, E26's own, for every run.
- **Scored every 256 updates:**
  1. E41's three accuracies with E41's panels and its sample of 1,000, unchanged: **A** (held-out add and sub at 12 letters), **R** (held-out pairs that need the reduction, no letters) and the taught panel.
  2. **The length test**, E30's: `exp11_cell.panel(t, exp11_cell.split(t, .5, seed)[1], seed, ks=(12, 40), n=300)` with `t = ("add", 89, b"p")`; accuracy at 12 and at 40 letters. On file for E26's final weights at 40 letters: 96.0, 97.0, 99.67 under the sweep and 97.67, 98.67, 92.33 under backpropagation (`exp45/e30/E30_UNSEEN.json`).
  3. The number of connections replaced since the previous row.
- **Per run:** updates to the 50, 80 and 90 thresholds on A and on R in E41's two versions, the three-row version carrying the description; computation in weight-items, with the count of weights inside the mask; CPU-seconds, reported but not used, since a masked dense matrix costs as much as a full one.
- **At the first batch and at the end:** the cosine between the sweep's gradient and the exact gradient for the recurrent matrix, on one batch. It says whether the sign rule is a better or a worse approximation on a matrix with few entries.
- **At the end, in the weights file:** the mask, periods, phases, the update at which each connection was made, and all added parameters. From them the report gives the share of connections never replaced, and a table of the share of connections by the reader's period and the source's period, at the start and at the end.
- **Committed:** the full weights for seed 73129, sixteen files. For the other two seeds a small file per run with the mask, periods, phases, the update at which each connection was made, the recurrent weights and the added parameters; the structure tables need no more.

### 22.6 Checks before the runs

Stage 0 is done by the worker of seed 73129; the other workers take the runner at that commit and repeat checks 1 and 2.

1. **`G` is E26.** The thirteen tensors of the present network are bit-equal to `arch26.new_net(seed)`'s at the start, and after 2,048 updates their `tensors_sha256`, computed over those thirteen only, is that of E26's dry births: 79618ab2… under the sweep and adf933d7… under backpropagation.
2. **Same data in every configuration.** The batch digest at row 4,096 is E26's: bbd71ab7…, 65b1c7e0…, 1df203fb… for the three seeds.
3. **The limit.** At every scored row: every cell has exactly N connections, all inside its neighbourhood, none to itself; every weight outside the mask is exactly zero.
4. **Time.** On random weights: changing the byte at position t changes no cell's value at an earlier position; a cell's value changes only at its own update positions; a cell with phase f is zero before position f.
5. **The variants agree where they must.** On the first batch `mean`, `step` and `source` give the same logits to 1e-6.
6. **Gradients.** Under backpropagation the parameter gradients equal those of a reference written cell by cell, without masks or buffers, to 1e-5 of each tensor's largest entry, for `P8`, `P8-source` and `F10`. Under the sweep they differ. Gradients outside the mask are zero.
7. **Replacement.** At an event, adding the new connections changes no logit; the same command run twice gives the same `tensors_sha256`, with changing connections.
8. **The reviewer's reader.** For seed 73129 the worker commits dry runs of 2,048 updates of `F10`, `P8` and `P8-source` with their weights files and their answers on the length-test panel. I write a numpy reader from this section and reproduce the answers. This does not hold stage 1.
9. **Cost.** CPU-seconds per update and per scored row for each configuration, on an idle host, and a forecast.
10. **theta.** Three dry runs of `A10` under the sweep, 4,096 updates, with theta = 0.1, 0.25 and 0.5. The value used everywhere is the one whose mean share of connections replaced per event, over the events at updates 768 to 4,096, is nearest 5%. If all three are below 1% or above 30%, stop and report. The choice uses no accuracy.

### 22.7 How the result is stated

Fixed beforehand, as in E41. For each comparison, rule and seed:

- **Updates to the 80 threshold**, on A and on R: E41's table of four cases (*falls* below 2/3, *rises* above 1.5, otherwise *stays*, with its rules for a run that does not reach the threshold), read here as fewer, more or about the same. The same in weight-items.
- **The length test:** the mean accuracy at 40 letters over the last three scored rows; *lower* or *higher* if the difference is 15 points or more, otherwise *about the same*.
- A cell of the table carries the description that two of its three seeds share; otherwise *mixed*. All three values are given.

Comparisons: `F10`, `A10` and `P8` each against `G`; `A10` against `F10`; `P8` against `A10`; every stage 2 configuration against its reference; and, in stage 1, the sweep against backpropagation within each configuration.

**One stop.** After seed 73129's stage 1 runs: if none of `F10`, `A10` and `P8` reaches 50 on A within the dose under either rule, stop and report. Otherwise stage 2 runs without a further release.

### 22.8 What arithmetic says beforehand, and my expectations

A cell with a long period can carry the first operand to the `=` sign without learning to keep anything: it is updated once while the operand's two digits are inside its period and is not updated again until after the `=`. Whether that is possible depends on the period, the phase and the number of letters (`e42_design_checks.py`, part D):

| Period | Share of (phase, number of letters) combinations in training for which it is possible | Numbers of letters for which no phase allows it | At 40 letters |
|---|---|---|---|
| 2 | 0 | all | never |
| 4 | 0.06 | 2 to 12 | never |
| 8 | 0.44 | 10, 11, 12 | never |
| 16 | 0.67 | none | never |

So with periods up to 4 the operand must be kept across updates in nearly every training item, as in the GRU; with periods up to 16 it need never be; and at 40 letters it must be kept across updates in every case. The argument predicts that longer periods make training easier and the length test harder. `P4`, `P8` and `P16` test it.

| | Expectation | Chance |
|---|---|---|
| X1 | Check 1 passes under both rules | 85% |
| X2 | `A10` reaches 80 on A within the dose, two of three seeds, under the sweep | 65% |
| X3 | `A10` needs more updates than `G` to reach 80 on A (*rises*), two of three seeds, under the sweep | 60% |
| X4 | `A10` reaches 80 on A in no more updates than `F10`, two of three seeds, under the sweep | 55% |
| X5 | `P8` against `A10`, updates to 80 on A under the sweep: fewer 20%, about the same 40%, more 40% | |
| X6 | `P16` is lower than `A10` at 40 letters, two of three seeds | 55% |
| X7 | `P4` is lower than `A10` at 40 letters, two of three seeds | 25% |
| X8 | Some stage 1 configuration's description against `G` differs between the sweep and backpropagation | 40% |

### 22.9 Cost and order

E26's birth took 3,276 CPU-seconds with its own scoring, 0.133 per update. The new layer does the same matrix work plus buffers; `source` does more. My estimate was 50 to 70 CPU-hours for the 48 runs and about 2 for stage 0. The audit replaces it: the coordinator's own timings give about 150 CPU-hours for the specification file as it stands, and about 75 to 80 if the loop over a period's steps is written to cost what its arithmetic costs, which the worker is asked to do and to check against the specification. Check 9's forecast decides. One worker per seed. The job does not wait for the small world and does not displace E39, E40 or E41.

### 22.10 What it cannot show

- **A saving in computation.** The input map and the readout stay dense and hold most of the weights, so limiting the recurrent connections removes 17% of the network. The job measures updates to threshold and generalisation. Whether many more cells can be afforded at the same computation needs the input and the readout limited too, which is a later job.
- **Settings.** Learning rate and decay were chosen for the dense GRU.
- **The form of a cell.** The GRU's equations are kept. A cell whose way of combining its inputs is itself learned is the next step if this one works (22.11).
- **One task family, answer targets, one grid.** The arrangement of cells is arbitrary; `W10` says whether the neighbourhood matters at all.
- **Distances longer than an item.** The state is reset at every item, so no period or connection can reach further back.

### 22.11 Choices that are mine, and what comes next

Mr. Xu left these to me; each can be overturned.

1. **The limit N counts connections between cells only.** Every cell still reads the whole current byte through the input map, as every GRU unit does. Everything a cell knows about the past then comes through its N connections and its own previous value.
2. **The GRU's equations are kept in this job.** What is learned is the strengths, which cells are connected, and how what arrived during a period is weighted. A cell that holds a graded value for a variable time needs the form `(1 - z) n + z h` in some guise, so the equations cost less generality than it may seem.
3. **A grid of 16 by 16 with wrap-around and a neighbourhood of 48.** The simplest arrangement with no edges.
4. **Half the cells at period 1**, the rest divided as in 22.4.
5. **The scale of the kept weights at the start**, and **the scale of the sign rule on a masked matrix** (22.3).
6. **Replacement by a strength threshold with a target rate of 5% per event** (22.6, check 10), and **the limit used in full**: Mr. Xu's limit is "at most N"; here a cell always has N.

Next, depending on the result: a cell whose update is a learned sum of its inputs and of products of pairs of them, with no gate equations given; limited input and readout connections with more cells at the same computation; and the cell layer as an arm of the small world once L1 has reported.

### 22.12 The first seed read (revision 17)

Seed 73129, fifteen runs, on `claude/gallant-dijkstra-i9zgsr` at 7fbdbdc2. The check is in `E42_SEED_73129_VERIFICATION_20261116.md` (script `e42_seed73129_verify.py`): my reader reproduces the worker's recorded answers of the three dry runs byte for byte and the final scores of all fifteen runs with no item different; the thresholds below are recomputed from the receipts, by the checking agent and again by me. One seed: nothing here is a result of E42 yet.

| Run | Updates to 80 on A | to 90 on A | At 12 letters | At 40 letters | Weight-items to 80 on A |
|---|---|---|---|---|---|
| `G`, sweep | 5,632 | 12,800 | 98.4 | 96.4 | 4.05e11 |
| `G`, backpropagation | 5,120 | 10,240 | 96.9 | 94.6 | 3.68e11 |
| `F10`, sweep | 13,312 | 20,480 | 90.1 | 89.1 | 7.97e11 |
| `F10`, backpropagation | 8,704 | 15,360 | 92.9 | 92.6 | 5.21e11 |
| `A10`, sweep | 13,824 | 22,528 | 92.7 | 91.8 | 8.28e11 |
| `A10`, backpropagation | 9,472 | 20,992 | 92.8 | 92.2 | 5.67e11 |
| `P8`, sweep | 7,680 | 10,496 | 96.3 | 96.0 | 4.60e11 |
| `P8`, backpropagation | 6,400 | 8,192 | 97.0 | 96.6 | 3.84e11 |
| `N5`, sweep | 17,152 | not reached | 88.2 | 86.9 | 1.02e12 |
| `N20`, sweep | 14,336 | not reached | 89.4 | 87.9 | 8.66e11 |
| `W10`, sweep | 17,408 | not reached | 88.6 | 88.0 | 1.04e12 |
| `P4`, sweep | 8,192 | 10,496 | 95.7 | 94.4 | 4.91e11 |
| `P16`, sweep | 7,936 | 11,776 | 94.6 | 82.8 | 4.76e11 |
| `P8-last`, sweep | 11,520 | 20,224 | 92.0 | 91.3 | 6.90e11 |
| `P8-source`, sweep | 7,680 | 11,008 | 94.2 | 93.7 | 4.67e11 |

The letter columns are means of the last three scored rows.

**What the seed shows, by the brief's descriptions.**

1. **Ten local connections a cell are enough to learn the task and to answer at 40 letters,** at 1.7 times the dense network's updates under backpropagation and 2.4 times under the sweep (`F10` against `G`: *more*).
2. **Periods take most of that cost away.** `P8` against `A10` is *fewer* under the sweep (7,680 against 13,824) and `P8` against `G` is *about the same* on every measure under both rules. `P8` reaches 90 before the dense network does, under both rules, and equals it at 40 letters. I gave "fewer" 20%. *(Revision 21: on three seeds `P8` is not level with `G`; it needs 1.25 to 1.75 times its updates, and it reaches 90 first on one seed of three. What holds is the comparison with `A10` under the sweep and the equality at 40 letters. See 28.3.)*
3. **Taking in what arrives during the period is the part that matters:** `P8-last` needs half as many updates again as `P8` (11,520 against 7,680). *(Revision 21: slower on all three seeds, by 1.50, 1.09 and 1.31 times.)* A separate strength for every step adds nothing (`P8-source`).
4. **Period 16 costs at 40 letters in the direction the arithmetic of 22.8 gave,** 82.8 against 96.0 for `P8`, in every one of the last six rows, but under the mark of 15 points. Periods 4 and 8 do not.
5. **The number of connections and the neighbourhood change little** (`N5`, `N20`, `W10` against `A10`: *about the same*).
6. **The rules differ only where the cells are sparse and have no periods:** `F10` and `A10` need about one and a half times the updates under the sweep that they need under backpropagation; `G` and `P8` need the same. The cosine between the sweep's gradient and the exact one for the recurrent matrix ends at 0.60 to 0.64 in `F10` and `A10`, against 0.74 in `G` and 0.67 in `P8` (the receipts' figures).

**A defect of my replacement rule.** `A10` against `F10` is *about the same*, but the comparison does not test changing connections. A new connection starts at zero and is judged after 512 updates against half the mean strength of connections that have been trained from the start; it reaches about a tenth of that mean in 256 updates. So nearly every new connection is given up at its first judgement: 35 to 41% of the connections at the end were made at the last two events and carry no weight, 6 to 13% are replaced connections that lasted, and a cell works with about six of its ten (3.2 of 5 in `N5`, 12.4 of 20 in `N20`). The share replaced per event rises from 9% in the first quarter of a run to 18 to 20% in the last, where the calibration of check 10 read only the first 4,096 updates. The rule and the calibration window were mine. Readings 2 to 4 are between runs that share the defect and are not touched by it; reading 5 is, since the three runs differ in connections that mostly carry no weight.

**What follows, after the other two seeds.**

- If they agree, the finding of E42 is reading 2: a network with a twenty-fifth of the recurrent connections and cells of several pre-set periods does what the dense network does on this task. It is not an advantage, and E42 cannot show one (22.10).
- Changing connections need one corrected run before anything is said of them: a new connection judged against its own age, or kept for at least 2,048 updates.
- The test this structure was argued for is retention (section 24.4, item 1). E43's question put to `F10` and `P8` is the next job there, once E43's three additions say where the loss sits.

## 23. What 64 updates change when a network that has lived learns a new family (revision 12)

A read-only measurement of mine on E40's files, seed 73129: the life-end weights (row 0 of E40's lived runs), and rows 64 and 8,192 of the three lived runs. The row-64 files were committed by the E40 worker at my request (45b6fd79 on `claude/vibrant-allen-8hziln`). Nothing was trained; every figure is a forward pass of the numpy reader on saved or recombined weights. Script: `e40_row64_changes.py` beside this memo, parts A to E.

Probes, fixed once: for add and for sub of the new family (mod 83, a tenth taught), 1,500 never-taught pairs that need the reduction and 1,000 that do not, no letters; for the two earlier families of the life, 600 pairs each of add and sub mod 97 and mod 73. Whole answers; each figure is the mean of add and sub.

### 23.1 Figures

Never-taught pairs of the new family that need the reduction, percent right. Seeds 73129, 93683, 116089 in that order; K has row-64 weights for seed 73129 only.

| Network | Lived under the sweep | Under backpropagation | Under K (73129) |
|---|---|---|---|
| Life end | 6.2, 2.3, 0.1 | 4.2, 0.4, 11.6 | 8.8 |
| After 64 updates | 75.6, 69.8, 75.5 | 84.1, 92.3, 95.7 | 80.3 |
| After 8,192 updates | 76.8, 82.5, 82.9 | 84.8, 94.1, 89.1 | 79.4 |
| Life end, with only the input map and the recurrent matrix taken from row 64 | 57.1, 63.0, 64.6 | 68.0, 85.2, 87.5 | 70.9 |
| Life end, with only the recurrent matrix from row 64 | 33.8, 39.8, 44.0 | 44.1, 70.5, 71.5 | 61.0 |
| Life end, with only the input map from row 64 | 16.5, 13.4, 7.0 | 9.4, 29.3, 44.0 | 27.1 |
| Life end, with only the readout (`lm`, `l2`, `l3`) from row 64 | 7.3, 5.3, 0.8 | 7.2, 2.1, 13.7 | 12.0 |
| Life end, with only the byte table and `l1` from row 64 | 8.3, 4.2, 1.0 | 5.1, 7.3, 17.4 | 12.8 |
| Row 64, with the recurrent matrix put back to the life's end | 24.0, 16.4, 14.3 | 24.3, 28.8, 62.6 | 38.8 |
| Life end plus the largest 1% of the changes | 7.9, 4.0, 0.8 | 3.8, 0.8, 12.4 | 10.6 |
| Life end plus the largest 5% | 15.6, 11.0, 2.1 | 4.7, 14.9, 17.0 | 17.8 |
| Life end plus the largest 20% | 38.4, 27.7, 32.3 | 29.7, 65.0, 72.3 | 52.7 |
| Life end plus the largest 50% | 66.2, 68.1, 72.8 | 76.9, 92.9, 95.2 | 78.1 |
| Every matrix's change replaced by its best approximation of rank 4 | 41.0, 25.1, 24.0 | 53.1, 49.4, 80.0 | 60.5 |
| Of rank 8 | 62.3, 47.0, 45.4 | 74.4, 81.2, 90.1 | 75.1 |
| Of rank 16 | 71.0, 61.8, 67.0 | 81.2, 90.5, 95.4 | 79.2 |

The earlier family mod 97, percent right:

| | Lived under the sweep | Under backpropagation |
|---|---|---|
| Life end | 92.8, 90.2, 95.2 | 94.5, 98.8, 94.0 |
| After 64 updates | 77.0, 54.0, 47.5 | 64.6, 56.4, 57.2 |
| After 64 updates, with the recurrent matrix put back | 89.6, 83.7, 93.3 | 89.8, 95.9, 93.0 |

How far the weights moved in 64 updates (seed 73129): each weight matrix by 4 to 12% of its size at the life's end (`l2` by 31%), the byte table by 0.3%. The rows of the four new code bytes moved by 0.02 to 0.06 against sizes of 2.2 to 3.3; the rows of the earlier families' codes did not move at all.

### 23.2 Readings

1. **What is learned in 64 updates lies in the recurrent layer.** The changes to its input map and its recurrent matrix, placed alone on the life-end network, give 57 to 65 of the 70 to 76 points under the sweep and 68 to 88 of the 84 to 96 under backpropagation. The changes to the readout give nothing by themselves, and neither do those to the byte table and `l1`.
2. **The new codes' rows of the byte table are used as they stood.** They were never trained before E40 and moved by about one percent of their size. The network reads four random vectors it had never seen and adjusts what follows them.
3. **The change is spread over many weights and is low in rank.** The largest 5% of the changed weights add at most 15 points; half of them are needed for most of the effect. But an approximation of rank 16 of each matrix's change recovers 62 to 71 of the 70 to 76 points under the sweep and 81 to 95 of the 84 to 96 under backpropagation. At rank 8 the sweep's change recovers less of its total (45 to 62) than backpropagation's (74 to 90): the sweep's change has somewhat more directions. I note it and do not read it further.
4. **The gain and the loss are in the same matrix.** Putting the recurrent matrix back removes most of the new family and restores the earlier one to 84 to 96%, in all six networks.

### 23.3 What it does not show

- **More than three seeds, or K beyond one.** The readings hold in all six networks of the sweep and backpropagation and in the one of K.
- **That few weights could not do it.** A change spread over many weights is what this optimizer produces: it moves every weight whose gradient keeps its sign by about the same amount. Whether a learner allowed few connections reaches the same result is E42's question. The low rank says only that the change needed has few directions.
- **Anything after row 64 but before row 8,192,** and anything about pairs with letters.

### 23.4 The request, met

The row-64 weights of the lived S and B runs of seeds 93683 and 116089 were committed by the E40 worker at 6922a3a4, and the tables above include them.

## 24. Long-term memory: Mr. Xu's objection, and what section 20 does and does not establish (revision 14)

### 24.1 The objection

Mr. Xu, 16 November: "按照你的三项发现，爱因斯坦势必需要每隔一段时间重新复习世界地理，才能回忆起德国的生活。现实生活中长期记忆是客观存在的，我们遗漏了什么？"

*Translation:* by your three findings, Einstein would have to review world geography at intervals in order to recall his life in Germany. In real life long-term memory exists as a matter of fact; what have we left out?

The objection is right, and section 20 is restricted below.

### 24.2 What survives a life without rehearsal (my measurement)

`kept_after_life.py` beside this memo; read-only. The first family (mod 89) at the end of the nine lives, which never rehearse it: whole answers on 1,500 pairs of each operation, no letters, percent. Seeds 73129, 93683, 116089 in that order. At birth every figure is between 88 and 100.

| | Lived under the sweep | Under backpropagation | Under K |
|---|---|---|---|
| Add, pairs that need the reduction | 0.7, 0.1, 1.4 | 1.7, 0.0, 0.1 | 1.0, 0.6, 0.7 |
| Add, pairs that do not | 44.7, 55.5, 25.9 | 38.1, 26.9, 49.4 | 57.9, 41.8, 62.9 |
| Sub, pairs that need the reduction | 0.4, 0.1, 0.1 | 0.1, 0.0, 0.0 | 0.0, 0.1, 0.4 |
| Sub, pairs that do not | 79.4, 11.9, 1.7 | 91.6, 39.6, 92.5 | 73.8, 73.0, 3.7 |
| Max | 98.5, 3.2, 25.7 | 98.3, 14.1, 0.0 | 98.7, 51.9, 0.0 |
| Min | 0.0, 96.4, 95.1 | 33.5, 65.5, 92.7 | 0.0, 92.3, 99.1 |

1. **What needs the modulus is lost in every network:** 0 to 1.7%.
2. **Some knowledge survives a whole life of 36,864 updates without one rehearsal.** Max or min is at 92% or more in 8 of the 18 cases.
3. **Survival is not reliable.** The other 10 cases are at 66% or less, 8 of them below 35%. Which of the two operations survives goes with the seed, not with the rule: max in seed 73129, min in the other two, under all three rules.

So the learner is not without long-term retention, but nothing in it guarantees retention, and what later learning contradicts is always lost.

### 24.3 What section 20 established, and what it did not

- **Established:** for families that contradict one another (the same operands and operator, another modulus), what is lost returns within about 100 updates, and items of the old family mixed into the stream keep it. For that case no mechanism beyond a store of items is needed.
- **Not established, and stated too widely by me:** that retention is not a research question. Three reasons.
  1. **Fixed memory.** A store of items cannot hold examples of everything a long life has taught. Under Mr. Xu's premise of fixed memory, retention without rehearsal is necessary, not optional.
  2. **Only the most adverse case has been measured.** Every life here consists of families that contradict the earlier ones. Mr. Xu's example is the other case: later learning that neither repeats nor contradicts the memory. It has never been measured.
  3. **Only one kind of memory is present.** The tasks are rules computed by one shared circuit. Nothing here is a fact stored once and recalled by a cue.

### 24.4 What the learner lacks beside a memory that lasts (argued, not measured)

1. **Separation.** In this learner every update changes nearly every weight, and the gain of a new family and the loss of an old one are in the same matrix (section 23). A memory held in connections that later learning does not touch could not be lost to it. This is the principle of Mr. Xu's statement of 16 November on cells with no connection, and the reason E42's networks should later be given a retention test.
2. **A second store.** A fast store of particular events in separate codes, with replay produced by the learner itself. The replay store of E38 is the experimenter's device, and it holds raw items.
3. **Consolidation at the single connection.** A connection that has been stable for long becoming harder to change. E38 tested only a pull of equal strength on every weight, which failed.

I recall the account of memory that combines the second and third (a fast store beside a slow learner, and replay during rest) from the literature and did not check it in this session. No mechanism is proposed here.

### 24.5 Job E43: is a family lost when the learner then learns something unrelated?

**Question.** After a birth, does training on a task that shares no input with the first family remove that family, as training on a contradicting family does?

- **Starts.** The six E26 births.
- **Arms.** `conflict`: the second family alone (mod 97), as E38's arm `none`. `unrelated`: a task on bytes that no family uses, with its own codes, with answers of two bytes and the end symbol so that the harness is unchanged; for example, returning two symbols shown before a run of fillers. The specification is yours.
- **Dose and rows.** 4,096 updates; scored at 0, 64, 256, 1,024 and 4,096.
- **Scores.** The first family: add and sub split by the reduction, max and min. The new task itself, to show that it is being learned.
- **Reading, fixed beforehand.** *Kept* if under `unrelated` the first family's reduction pairs are at 80% or more of their level at row 0 at row 4,096, in two of three seeds; *lost* if at 20% or less; otherwise *partly kept*. `conflict` is the reference and is expected at zero by row 1,024.
- **What each outcome means.** *Kept*: the fast loss measured so far is loss by contradiction, and the learner holds what is not contradicted. *Lost*: updating every weight removes even unrelated knowledge, and separation (24.4, item 1) becomes the first thing to test.
- **Cost.** Corrected in revision 15: two arms are twelve runs, and the brief's four arms are twenty-four, about 3.5 to 4 CPU-hours with scoring by the audit's estimate. My "twenty-four runs, about 2 CPU-hours" was a slip.

My expectation, on record: under `unrelated` the reduction pairs fall below half their starting level by row 1,024 in two of three seeds, 60%.

### 24.6 What weight decay alone does (found by the audit of E43; recomputed by me)

Every life so far runs with a decay of 0.25 at a learning rate of 0.001: each update multiplies every weight outside the byte table by 0.99975, a time constant of 4,000 updates. The audit computed what that does with no learning at all, on the six births: every parameter but the byte table multiplied by 0.99975^t, nothing else changed, and the first family scored (`e43_brief_checks.py`, part C; I reran it). The reduction pairs of add and sub, as a share of their level at birth, seeds 73129, 93683, 116089:

| Updates | Factor on the weights | Sweep births | Backpropagation births |
|---|---|---|---|
| 256 | 0.938 | 95.2, 97.8, 98.1 | 97.3, 96.7, 96.8 |
| 1,024 | 0.774 | 17.8, 35.3, 26.1 | 39.3, 49.0, 46.0 |
| 2,048 | 0.599 | 0.4, 1.3, 0.3 | 0.4, 3.1, 1.3 |
| 4,096 | 0.359 | 0, 0, 0 | 0, 0, 0 |

Max and min go the same way, a little later: 4 to 35% at 2,048 and zero at 4,096.

What it changes.

1. **Under the standing decay nothing is kept passively.** A family whose weights no gradient re-creates is gone after about 2,000 updates, whatever else is learned and whether or not anything contradicts it. This is a second cause of loss beside the one E38 measured: E38's loss by a contradicting family takes 64 to 256 updates and happens without decay too.
2. **What survives a life in 24.2 is being re-created, not stored.** A life is 36,864 updates, nine time constants. The max or min that stands at 98% at its end is held there by the life's own max and min tasks, which use the same machinery under other codes. It is rehearsal by shared use, and the table of 24.2 should be read so.
3. **For Mr. Xu's objection:** this learner, as it is set, has a forgetting time of a few thousand updates built into its optimizer. The decay was chosen for births, where it is commonly held to favour rules over tables; whether the births here need it is not on file as far as I have found. A learner that is to keep what it does not use cannot have every weight decay at one rate.
4. **For E43:** the reading stays on the arm without decay, as the coordinator wrote it. The arm under decay is read against the table above, not against the other arm: it asks whether unrelated learning holds anything up against a decay that would otherwise remove everything.

### 24.7 E43 read (revision 16)

Branch `claude/peaceful-hopper-8ofq4i` at ac56179d. I recomputed the figures below from the 24 receipts; I could not use the weights (see the first request).

- **The registered reading is *lost*, under both rules.** After 4,096 updates on the unrelated task without decay, the first family's reduction pairs are right in 0 of 3,916 for add and for sub, in all six networks. The new task is learned completely (300 of 300 on taught and on held-out pairs).
- **The loss is complete after 64 updates,** before the new task scores anything: 0 to 6 of 3,916 in five networks, 254 and 131 in the sixth. Pairs without the reduction, max and min go with it. Under `conflict` at the same point the pairs without the reduction, max and min are still at 49 to 98%.
- **The answers become the end symbol alone** (the worker's observation on seed 73129: 497 and 511 of 600 prompts at update 64).
- **The weights do not move much more than under a contradicting family:** the recurrent matrix by 17 to 20% of its size in 64 updates against 13 to 16%. The state at the `=` sign changes by 0.39 to 0.86 of its size against 0.21 to 0.30.

My expectation of 24.5 (below half by 1,024 updates, 60%) was right in direction and far too mild.

**What it does not settle.** Three explanations remain, and they call for different remedies.

1. *The knowledge is still in the network and is not expressed.* The unrelated task never rewards a digit, so the shared readout is driven away from digits for every input.
2. *The optimizer's first steps.* A fresh AdamW moves every weight by about the learning rate at each early step whatever the size of its gradient, and the new task starts from a loss of 18.
3. *The function is overwritten,* and would be under any update that learns the new task in this network.

**Three additions asked for.**

1. **The weight files.** The report says that update 4,096 of every run and updates 64 and 1,024 of seed 73129's `unrelated` runs are committed. They are not on the remote; the branch ignores `*.pt`. With them I can place the loss as in section 23: the first family's score with only the readout put back, and with answers chosen among digits only.
2. **A reminder,** as E36's: from the six `unrelated` networks at update 4,096, 1,024 updates on the first family, scored at 0, 16, 32, 64, 128, 256 and 1,024, beside E36's fresh references. If the family is back within about 100 updates, explanation 1 or 2 holds and what was learned was not destroyed.
3. **One arm under plain gradient steps:** `unrelated` again with the update in proportion to the gradient (no momentum, no normalisation by the gradient's size), the learning rate being the smallest of 0.01, 0.03 and 0.1 at which seed 73129 meets the brief's condition under backpropagation. If the first family survives there, the equal-sized step on every weight is the cause, which is section 24.4's first item in its simplest form.

### 24.8 E43's additions read (revision 19)

Branch `claude/peaceful-hopper-8ofq4i` at 722cb621. Figures from the receipts, read by me; the plain runs under the sweep are being continued to 16,384 updates and are not final.

**The reminder.** From the six `unrelated` networks at update 4,096, training on the first family as E36 did. Reduction pairs of add and sub, percent:

| Updates of reminder | Sweep (73129, 93683, 116089) | Backpropagation |
|---|---|---|
| 64 | 1.2, 1.0, 0.9 | 1.4, 0.0, 0.0 |
| 256 | 5.8, 1.1, 1.2 | 1.7, 1.0, 1.5 |
| 1,024 | 91.3, 10.7, 69.3 | 91.3, 14.4, 42.0 |

After a whole life E36 found the family back at 85% or more within 64 to 128 updates, and a fresh network at 1.1 to 2.5% after 1,024. So after unrelated learning the family is not back in a hundred updates in any network; something is saved, since two of six are above 90% at 1,024 where a fresh network is near zero, but the family was not merely unexpressed.

**Plain gradient steps** (no momentum, no normalisation, learning rate 0.03 by the brief's rule). Under backpropagation the first family's reduction pairs are at 0.4, 2.8 and 2.1% after 64 updates and at zero from 256 on, and the new task is learned by update 2,048 to 4,096. Under the sweep they are at 16.7, 45.2 and 19.7% after 64 updates, 6.6, 25.7 and 6.4% after 256 and at most 4.9% after 1,024, before the new task has scored. So the equal-sized steps of AdamW are not the cause.

**Where the loss sits** (`e43_where_the_loss_sits.py`, my measurement on the weight files of seed 73129; 1,500 reduction pairs each of add and sub).

| After 64 unrelated updates | Sweep | Backpropagation |
|---|---|---|
| Whole answers | 0.1 | 0.0 |
| Both digits right when chosen among the ten digit outputs only | 3.9 | 2.0 |
| The readout put back to the birth's | 3.7 | 0.7 |
| The input map and the recurrent matrix put back | 8.1 (digits only 84.5) | 27.6 (digits only 47.4) |
| The birth with only its recurrent matrix changed | 34.8 | 30.6 |
| The birth with only its input map changed | 68.5 | 59.0 |
| The birth with only its readout changed | 5.3 (digits only 91.5) | 78.4 (digits only 92.1) |

After 1,024 and 4,096 updates the birth with only its recurrent matrix changed is at about 1%.

**Reading.** Of the three explanations of 24.7, the third holds. The first family's function is overwritten throughout the network within 64 updates, most of all in the recurrent matrix. The readout is also driven away from digits, under the sweep enough to remove the answers by itself, but putting the readout back restores nothing. The remedy to test is therefore separation: a change that cannot reach the connections the earlier family uses. Section 27 gives its simplest test.

**Two further additions to E43** (item 17).

1. *The conventional reference.* The `unrelated` arm with items of the first family mixed into the stream in E38's share, with and without decay, on the six births. If rehearsal keeps the family at no cost here as it did in E38, it is the reference any other remedy is measured against.
2. *E42's networks.* The `unrelated` arm from `F10` and `P8` of seed 73129 under each rule, with E42's forward. Every cell there still reads the whole byte and the readout is shared, so I expect *lost* (90%); the run replaces the expectation by a figure.

## 25. The small world after job L1a (revision 16)

### 25.1 What L1a found

Branch `claude/gallant-edison-p0qa68` at ac85c093; the worker's figures, not recomputed. Seventy-two runs: twelve linear tasks, each alone, at 64, 96 and 192 units, under both rules, 16,384 updates each.

- **No run reaches the criterion** of 0.8 on asked pairs, at any size, under either rule.
- **Every run fits its taught pairs:** the training loss ends at 0.033 or less.
- **The level on asked pairs goes with the number of taught pairs:** 0.00 to 0.02 for the four tasks with 32 to 72 pairs, 0.00 to 0.36 for those with 162 to 264, 0.02 to 0.69 for those with 544 to 684.

### 25.2 My explanation is withdrawn

I attributed the failed calibration to the missing birth (section 6.3 of the discussion memorandum; my message of 16 November). L1a refutes it for these tasks: a task that is not learned alone at three sizes would not be learned in a birth either. The coordinator had built the life with a birth on my recommendation, and it was rightly not run.

### 25.3 What the three experiments say together (argued)

A network from random weights finds the rule only when a task offers enough taught pairs; otherwise it stores them.

| Source | Taught pairs per operation | A network from random weights |
|---|---|---|
| E41, a half or a quarter of the pairs at modulus 89 | 3,960; 1,980 | finds the rule in 3,000 to 17,000 updates |
| E41 and E35, a tenth | 792 | does not within 8,000 to 25,000 updates; at half width under backpropagation it reaches 48% after 40,960 and is rising (one seed) |
| L1a, half of the pairs at moduli 8 to 37 | 32 to 684 | does not within 16,384 updates |
| E40, a tenth at modulus 83, a network that has lived | 689 | finds it in about 64 updates |

So the small world's tasks are below what a learner without experience can learn, by construction: its largest task has fewer taught pairs than a tenth of a full-scale family. Neither the audit of the first brief nor mine of revision 3 compared those two numbers.

### 25.4 What I propose in its place

Keep the instrument (schedule, event files, scorer, the measures of 17.3 and the horizon rule) and replace the task set and the learner.

- **Tasks:** families in the full-scale format, one modulus each, at moduli not yet used, taught from a tenth of their pairs, as E40's.
- **Learner:** a network born on the first family from half of its pairs. E41's first seed says the half-width network reaches the threshold with the least computation (0.035 to 0.043 CPU-seconds per update); if the other two seeds agree it is the learner, otherwise the present width.
- **Cost:** at 256 updates a visit, a life of 1,024 visits is 262,144 updates: about 3 CPU-hours at half width before scoring, 9 at the present width.
- **Calibration first (the new L1):** born networks on eight new families in turn, visits of 64, 128 and 256 updates; the gate as before, a median first-visit level between 0.10 and 0.90 on reduction pairs. E40 warns of the other failure here: a learner with three families behind it was at 70 to 84% after 64 updates, so the ceiling is the risk, and the visit length or the taught share is what the calibration sets.
- **What it gives up:** the eighteen operations with coefficients, and lives of more than about 240 tasks.

The generator is new and is audited before any run.

## 26. Locality at the input and the output (revision 18)

### 26.1 Mr. Xu's statements

- 16 November: "It seems the byte level translation / output is the fixed cost head that in the long term can be ignored. What is the problem?"
- 16 November: "No, biological cells obviously cannot see the whole input, so locality is essential and our current set up does not scale. Also it makes no sense for a cell to see one million tokens in a future NLP task."

So locality is a requirement of the design, not an option under test. E42 met it between cells and left it unmet in two places.

### 26.2 The arithmetic (mine; multiply-adds per item and position, forward)

| Part of the present network | Now | How it grows |
|---|---|---|
| The byte table, `l1`, and the readout's own layers `l2` and `l3` | 397,824 | not with the number of cells |
| The input map into the cells and `lm` out of them: every cell reads all 512 input values and every readout unit reads every cell | 2,048 a cell | with the number of cells |
| Connections between cells at N = 10 | 33 a cell | with the number of cells |

In E42's wiring a cell costs 2,081 and limiting its connections to other cells saves 17% at most. With the input read by a fixed number of cells and the readout reading a fixed number, a cell costs 33, or 22 with E42's periods. At the present network's 1.12 million multiply-adds and 1.12 million weights, such a network could have about 16,000 to 20,000 cells in place of 256.

### 26.3 Requirements for the job (the specification is yours)

1. **The interface is a boundary.** A fixed number of cells read the current byte and a fixed number are read by the readout; no other cell touches either. The byte stays the unit at both ends, so the interface does not grow with the task.
2. **Every cell reads at most N cells of its neighbourhood** and its own previous value, as in E42, with fixed connections (E42's replacement rule is not used; 22.12).
3. **Time.** A local connection passes a value one cell a step, and the answer is due at the step of the `=` sign. The cells that read the byte and the cells the readout reads are therefore the same cells or neighbours. A variant that allows a stated number of steps between the `=` sign and the answer may be added, for every learner alike.
4. **Periods by place, as one variant.** E42 scattered the periods. Here one arm gives period 1 to the cells at the interface and longer periods to cells further from it, so that neighbours have neighbouring periods; the other arm scatters them as E42 did. With equal numbers of cells at periods 1, 2, 4, and so on, the computation a step stays below twice that of the fastest group however many groups there are, while the longest period doubles with each group.
5. **Sizes.** At least three numbers of cells (for example 256, 1,024 and 4,096) at one interface, and the comparison is made at equal computation a update and at equal number of weights, beside E42's `G` and `P8`.
6. **Arithmetic by index, not by masked matrix,** since a matrix of all pairs of cells cannot be held at these sizes. At 256 cells the new forward and both rules' gradients must equal E42's to its tolerances.
7. **Measures.** E42's (updates and weight-items to threshold on A and R, the length test), the multiply-adds a update, and one diagnostic for whether the cells away from the interface are used: the scores with those cells held at zero.
8. **A full-length run on one seed before the roster,** with its diagnostics read, as the working rules now require.

### 26.4 What I expect, and what it cannot show

On one family, a few hundred cells are enough, so I do not expect more cells to lower the updates to threshold; the job shows whether a network wired in this way learns at all and what a cell costs. Where more cells should pay is in holding many families and in keeping them apart, which is the long-life world of E44 and the question of section 24. The error has further to travel to cells away from the interface, and E42's first seed already shows the sweep's gradient less well aligned in a sparse matrix (0.60 to 0.64 against 0.74), so the sweep may fall behind backpropagation here by more than it did in E42.

## 27. Job E48: a new family learned only in connections gated by its own code (revision 19; renamed in revision 20)

### 27.1 Why

Three measurements point the same way. What 64 updates learn of a new family is a change of low rank in the recurrent layer (section 23). The loss of the earlier family is in the same matrices (section 23). Under unrelated learning the function is overwritten throughout the network, with plain steps as with AdamW (24.8). The programme's question is whether changes can be confined to the connections in use. This job puts it in its simplest form: the change for a new family is made only in connections that take part when that family's code is present. The earlier families are then untouched by construction, and what is measured is the price in learning.

### 27.2 Requirements (the specification is yours)

- **Starts, family, stream, panels and scores: E40's,** on the networks that lived under the sweep and under backpropagation, three seeds, so that E40's own runs are the reference.
- **What may change.** The four new code rows of the byte table, and for each of the two matrices of the recurrent layer (the input map, 768 by 512, and the recurrent matrix, 768 by 256) one added term of rank r, the product of two matrices, applied only on items of the new family. One term a family, shared by its four operations. It starts at zero. Every other parameter is frozen.
- **The gate is supplied by the harness** from the item's code byte. This is a first stage in the sense of E35's: an upper bound. Whether the learner can produce the gate itself is a later question, and the report says so.
- **Ranks** 4, 16 and 64. **Rules:** each network under the rule it lived under.
- **Dose** 1,024 updates; scored at 0, 16, 32, 64, 128, 256, 512 and 1,024.
- **Checks.** Row 0 equals E40's row 0 on every score. With the term off, every earlier family's answers are the same item for item at every row, and the frozen tensors keep their hash: the keeping is by construction and is checked, not reported as a result. The batch digest is E40's.
- **Reading, fixed beforehand,** for each rule and rank, on the new family's reduction score at row 256 against E40's run of the same seed at row 256: *no price* if within 10 points in two of three seeds; *a price* if 30 or more below; otherwise *some price*. The same at rows 64 and 1,024 is tabled.
- **Tabled beside it:** the count of numbers changed for the family (2,304 times r, and 32 for the code rows: 9,216, 36,864 and 147,456), against the network's 1,124,113 and against the number of items a replay store of that size would hold.
- **Cost.** Eighteen runs of 1,024 updates: about 1.5 CPU-hours with scoring.

### 27.3 What it cannot show

- That a learner can find the gate, or decide when a new term is needed.
- Fixed memory: the terms grow by one a family, and the comparison with a replay store of equal size is the fair one; it belongs to the long-life world.
- Unrelated tasks, whose answers need output units of their own.

### 27.4 My expectation, on record

Rank 16 gives *no price* under backpropagation, 60%, and under the sweep, 50%. Rank 4 gives *some price* or *a price* under both, 60%.

# Part III. The results of 17 November and what follows from them

## 28. The results of 17 November, checked (revision 21)

Mr. Xu's instruction of 17 November: "Most works are done or waiting for your audit. Discuss next steps. As I am outside, coordinate yourself." This Part is the answer. Section 28 gives what was checked, section 29 a measurement of my own, section 30 the decisions and the plan.

Every result that arrived was checked from its receipts and weight files with the numpy reader, by an agent of mine whose main figures I then ran again myself. Each document below ends with its commands. Nothing was trained.

| Job | Document beside this memo | What was reproduced |
|---|---|---|
| E41 on three seeds; E46 | `E41_E46_VERIFICATION_20261117.md` | 54 and 18 receipts; every threshold; the last-row counts of all 42 and 18 weight files |
| E42 on three seeds | `E42_THREE_SEEDS_VERIFICATION_20261117.md` | 45 runs: every final count; 720 of 720 thresholds and means; 105 of 105 descriptions |
| E47, the trial | `E47_TRIAL_VERIFICATION_20261117.md` | 4 runs: all 7,200 recorded answers byte for byte; every threshold |
| E48 | `E48_RESULT_VERIFICATION_20261117.md` | 18 runs (section 28.7) |
| E49; E43's second addendum | `E49_E43ADD2_VERIFICATION_20261117.md` | 8 and 16 runs: every last-row count |
| E44's calibration and the roster note | `AUDIT_E44_L2_ROSTER_NOTE_20261117.md` | section 28.8 |
| The E50 brief | `AUDIT_E50_SPARSE_INPUT_CODE_BRIEF_20261117.md` | section 30.2 |

No tabled figure of any worker's report was found wrong. What changes below is what the figures are taken to show.

### 28.1 E41 on three seeds: computation to threshold by width

Updates until three scored rows in a row are at or above 80% on score A (held-out add and sub at 12 letters); median of the three seeds and range; "n.r." is not reached within the dose.

| Rule, taught fraction | Half width (128 cells) | Present width (256 cells) | Double width (512 cells) |
|---|---|---|---|
| Sweep, 0.50 | 10,496 (10,240 to 12,288) | 5,376 (4,864 to 5,632) | 2,816 (2,816 to 3,328) |
| Back-propagation, 0.50 | 7,680 (7,424 to 11,520) | 4,352 (4,096 to 5,120) | 3,072 (2,816 to 3,328) |
| Sweep, 0.25 | 17,408 (14,080 to 24,832) | 8,704; 9,984; n.r. | 6,400 (4,096 to 6,912) |
| Back-propagation, 0.25 | 7,680 (6,144 to 10,240) | 4,352 (3,584 to 4,864) | 4,608 (3,584 to 4,608) |
| Either rule, 0.10 | n.r. on every seed | n.r. | n.r. |

Four statements hold on three seeds.

1. **The sweep against back-propagation.** With half of the pairs taught the sweep needs 1.10 times back-propagation's updates to 80% on A (median; range 0.92 to 1.38) and 1.19 times on the reduction score R (1.00 to 2.29). The ratio is nearest one at the double width (0.92 to 1.15) and largest at the half width (1.07 to 2.29). With a quarter of the pairs taught the sweep reaches 80% on R in 4 of 9 runs.
2. **Width.** Updates to threshold fall as the network widens, and computation to threshold rises at every step (by 1.1 to 2.5 times from half to present, by 1.5 to 4.0 from present to double). The half width is the cheapest in 24 of 24 cases. It is also the narrowest width that was run, so the cheapest width is not located.
3. **A tenth of the pairs** is not learned at any width under either rule. At the half width under back-propagation, score A is still rising at 40,960 updates (48, 68 and 58%).
4. **The learners of the long lives.** E44's rule for choosing a width gives the half width on all three seeds. The six half-width births at fraction 0.50 are on the default branch and score 92.8 to 96.2 on A and 90.9 to 94.7 on R.

One consequence for the long lives: the half width is where the sweep is furthest behind back-propagation, so a comparison of the two rules over a life at that width starts from a handicap that the present width does not show.

### 28.2 E46: learning the first family without weight decay

The sentence of the worker's report is reproduced: at the present width, with half of the pairs taught, the first family is learned without decay as fast as with it under both rules (updates to 80% on R at decay 0 against decay 0.25: sweep 0.96, 1.05 and 0.88 times; back-propagation 1.28, 1.50 and 0.80 times), and the length test at 40 letters is within 3 points. Three qualifications belong beside the sentence. Under the sweep the final level of R is lower without decay on all three seeds (by 3.5, 1.6 and 5.2 points). Without decay the size of the recurrent matrix has not settled at the end of the run (70 to 72 against 26 to 29, and still growing by about 1 a thousand updates). And there is no birth without decay at the half width, which is the width of the long lives. The lives without decay should therefore record the sizes of the weights at every scored row.

### 28.3 E42 on three seeds: what holds and what does not

Updates to 80% on A, seeds 73129, 93683, 116089.

| Configuration | Sweep | Back-propagation |
|---|---|---|
| `G`, the dense network | 5,632, 4,864, 5,376 | 5,120, 4,096, 4,352 |
| `F10`, ten fixed connections a cell | 13,312, 13,056, 12,544 | 8,704, 6,912, 9,216 |
| `A10`, ten connections, replaced during learning | 13,824, 14,336, 13,824 | 9,472, 7,680, 7,936 |
| `P8`, as `A10` with periods 1, 2, 4, 8 | 7,680, 8,448, 7,424 | 6,400, 7,168, 7,168 |

- **Holds on three seeds.** Limiting a cell to ten connections raises the updates to threshold by 2.3 to 2.7 times under the sweep and 1.7 to 2.1 times under back-propagation. Restricting the connections to a neighbourhood adds nothing consistent to that (`W10` against `A10`: 1.26, 0.84, 1.02). Replacing connections during learning is about the same as keeping them fixed (1.04 to 1.10), and the defect of my replacement rule (22.12) repeats on every seed. Five connections are slower than twenty by 1.10 to 1.20 times.
- **Holds under the sweep only.** Periods remove 62 to 76% of the extra updates that the limit costs (`P8` against `A10`: 0.54 to 0.59). Under back-propagation they remove 71, 14 and 21%.
- **Does not hold.** In 22.12 I read the first seed as showing `P8` level with the dense network. On three seeds `P8` needs 1.36, 1.74 and 1.38 times the dense network's updates under the sweep and 1.25, 1.75 and 1.65 times under back-propagation. That sentence of 22.12 is withdrawn.
- **Holds, and was not read before.** With periods the sweep is level with back-propagation (1.04 to 1.20 times its updates), as in the dense network (1.10 to 1.24), where without periods the limited network puts the sweep behind (1.36 to 1.89).
- **Computation.** Counted in multiply-adds to 80% on A against the dense network under back-propagation, no configuration is below one: `P8` under back-propagation 1.04 to 1.46, `P8` under the sweep 1.25 to 1.72, `F10` under the sweep 2.2 to 2.7. The limit saves 17% of the multiply-adds a position (section 26) and costs more than that in updates.
- **Length.** `P16` is low at 40 letters on two of three seeds (82.8 and 67.3 against 94.1).

### 28.4 E47, the trial: input and output at a boundary

The four runs (1,024 cells of which 256 read the byte and are read by the readout; seed 73129) learn the task: 80% on A after 10,496 updates under back-propagation in both arms, and after 13,568 (`one`) and 15,360 (`place`) under the sweep. Two findings decide what follows.

- **No saving.** Computation to 80% on A is 1.7 times the dense network's under back-propagation and 2.3 to 2.5 times under the sweep, and 1.05 to 1.29 times that of E42's `F10`, which has a quarter of the cells.
- **Cells beyond the first ring take no part.** With the 540 cells of the second and third rings held at their mean, or given the states of another item, no score changes by more than 0.7 points. Their states hardly differ between items (standard deviation at the `=` sign at most 0.018 in the second ring and 0.002 in the third, against 0.37 to 0.46 at the interface). This is so before any training: on the starting weights the differences between items fall six- to tenfold from one ring to the next. The first ring does take part.

So the trial shows that a network with the interface at a boundary learns, and it shows that this wiring does not use cells away from the boundary. That the weak passage of differences at the start is the cause of the later non-use is an explanation and has not been tested; it would be refuted if a start at which the differences reach the third ring still left those cells unused after training.

### 28.5 E49: an interface in which each cell reads part of the input values

On 256 cells, seed 73129, with E42's `F10` as the reference (levels are means of the last three scored rows, A / R).

| Setting (input values a cell reads / cells a readout unit reads; multiply-adds a position against `F10`) | Back-propagation | Sweep |
|---|---|---|
| `F10` (512 / 256; 1.00) | 93 / 94; 80% on R at 9,472 | 91 / 89; at 15,104 |
| `in16` (16 / 256; 0.59) | 85 / 83; at 20,480 | 1.9 / 1.4 |
| `I64` (64 / 32; 0.51) | 74 / 65; not reached, rising | 1.3 / 0.9 |
| `out8` (512 / 8; 0.86) | 74 / 71; not reached, rising | 10.6 / 7.8 |
| `I16` (16 / 8; 0.45) | 24 / 17; not reached | 1.2 / 0.9 |

- Under back-propagation a limited interface is learned slowly: the one setting that reaches 80% on R does so at 1.28 times `F10`'s computation, so nothing is saved.
- **Under the sweep a limited input is not learned at all.** The training loss has not fallen since update 4,096 in `I64`, `I16` and `in16`. The sweep's gradient at the start is as well aligned with the exact one as in `F10` (cosine 0.73 to 0.77 at the input map), so the alignment at the start does not explain it. What is seen at the end is another state of the network: the byte's code has shrunk from 0.41 to 0.05 to 0.07, the cells' states at the `=` sign differ between items by 0.03 to 0.04 (0.40 in `F10`), and `l3` has grown 9 to 21 times. The cause is not established.
- A continuation would at best show 80% on R under back-propagation at about 3.4 times `F10`'s updates. It is not worth its cost.

### 28.6 E43's second addendum: rehearsal, and E42's networks

- **Rehearsal keeps the first family** while the unrelated task is learned, under both rules, with and without decay: the reduction score at update 4,096 is 95 to 105% of its start in all twelve runs. The price is in computation and in speed. Half of the items of every update are rehearsal (one old item for each new one), and at update 256 the unrelated task is behind the unrehearsed run in all twelve comparisons.
- **E42's networks lose the family as the dense one does**: `F10` and `P8` are at or below 1% on reduction pairs by update 64. Limiting the connections between cells does not separate what is learned.
- The two `P8` weight files at update 4,096 lack the coefficients over the steps of a period, so those two networks cannot be rebuilt. No reported figure depends on them. The report's list of committed weights should mark the two files as incomplete.

### 28.7 E48: a new family learned in connections that are present only with its code

The 18 runs are reproduced from the life-end weights and the term files. The figure is the reading fixed in the brief: the mean of the reduction score on never-taught pairs at rows 128, 256 and 512, E48 minus E40's run of the same seed and rule (seeds 73129, 93683, 116089).

| Rule | Rank 4 | Rank 16 | Rank 64 |
|---|---|---|---|
| Sweep | -15.8, -31.2, -27.4 (*some price*) | -2.4, -8.7, -8.2 (*no price*) | +1.2, +5.0, -1.0 (*no price*) |
| Back-propagation | -4.8, -8.5, +0.2 (*no price*) | +3.9, -1.9, +4.5 (*no price*) | +9.3, +1.4, +6.3 (*no price*) |

- **The earlier families are kept item for item,** as the construction requires. About 37,500 earlier items a network (every earlier task at five lengths, the compositions, every held-out reduction pair) are answered as by the life-end network, in all six networks. With the gate forced open on those items, the same term lowers the life's reduction pairs under modulus 97 from 87 to 98% down to 1 to 52%. The separation is the gate's and nothing else's.
- **The new family is learned later, and to a higher level.** At row 64 every run is below E40, by 6 to 59 points. At row 1,024 the mean reduction score at ranks 16 and 64 is 91% under the sweep and 97% under back-propagation, against E40's 86 and 90%, and it is above the highest value that E40 reaches in 8,192 updates in all six networks. This second comparison was not fixed beforehand, and E48 differs from E40 in one more thing: the lived weights do not decay.
- **E40's loss was not needed.** In E40, with every weight free, the life's four panels fell by 38 to 43 points within 256 updates of learning the new family. E48 learns the same family with those panels untouched.
- **The gate also withholds.** Forced open on items of the birth family, the term raises that family's `max` or `min` in some networks (for example from 4 to 52% and from 14 to 94%). What a family learns behind its gate is not available to another family.
- **Memory.** A term of rank 16 is 36,896 numbers; thirty such terms equal the network.

The brief's reading used the words "as fast as". They were my requirement, and the result does not support them for the first 64 updates. The reading by the three-row mean stands as tabled.

### 28.8 E44: the calibration, and the roster note for the first long lives

**The calibration is reproduced.** With every one of its 917,504 answers marked again against my own stream, the gate's medians equal the report's. Under the conventional learner at 256 updates a visit the median level on reduction pairs at a family's first visit is 0.30 to 0.64 in the eight worlds, all inside the gate; at 128 updates four of the eight worlds pass and at 64 none. 256 is the shortest length that passes.

Two things the calibration shows beside the gate.

- **Nothing is kept from one visit to the next** (review 9.156; I recomputed the second lap). One visit after a family's own, its share right on the fixed list is 0.007 to 0.076 of what it was. At a return the first update scores 0.011; the last quarter of a return visit reaches 0.58 against 0.49 at first visits.
- **What the wrong answers are** (my count on the calibration's lists). One visit later, 15% of the conventional learner's answers on a family's reduction items, and 21% of the sweep's, are the right answer under the modulus of the family taught last, where chance is about 1%. The network answers every family with the modulus it learned last. This is section 29's finding seen from another side: the modulus is held in connections that every family uses.

**The roster note** (release 1: four arms under back-propagation, `naive`, `replay` and both without decay; the learners are E41's half-width births; 32 lives; 59 CPU-hours; world 1's two long lives are the trial). The check is in `AUDIT_E44_L2_ROSTER_NOTE_20261117.md`.

- **The change to the runner is sound, with conditions.** It is not code yet: the runner on the default branch is the one the calibration ran. As the note describes it, the two new arguments select which rows of the list are written, and cannot change what a life learns. A second change is needed that the note does not name, because the present runner refuses the `replay` arm.
- **Nine changes are required,** given as text to apply in the audit's section 8. Those that bear on what the lives can show: the quantities the note says it records are not the ones the scorer computes, and "kept" is not defined for a life with returns (definitions D1 to D14 are proposed); the trend of the score at 256 visits partly reads the schedule and should be read on returns only (the defective definition was mine, section 17.4); the wrong answers are to be classified as above; `replay` with half of each update taken from the store teaches 6,144 new items a visit, the dose at which the gate failed in four of eight worlds; and decay alone cannot be read in a life with returns.
- **In order as written.** The store is 1,269,828 bytes, the size of the network, and holds 55,209 items, against 80,000 to 108,000 distinct taught pairs in a life of 1,024 visits, so the store cannot hold everything it is offered. Such a life meets 41 to 52 families and has 972 to 983 returns. The closing rule is that of section 17.5. The six births are committed.
- **After the nine changes the lives may start, and the note needs no second audit.** I read the changed runner and its check receipts when they are committed. That reading need not hold the trial, because a defect in the code of the list cannot change the weights or the event file of a life.

## 29. Cells in use and directions in use: why the first family is overwritten, and what would leave it intact (revision 21)

### 29.1 The question, and two definitions

The programme's question speaks of changes "confined to the connections in use". E43 showed that the first family is overwritten within 64 updates of unrelated learning, and its second addendum that limited connections between cells do not prevent it. This section measures on E43's saved networks how much of the network an item uses, and whether the change made by the unrelated task can be kept apart from what the first family uses. The scripts are `cells_in_use.py` and `directions_in_use.py`; the files are the six E26 births, E43's `unrelated` networks at updates 64 (seed 73129 only) and 4,096, and its rehearsed networks at 4,096. Forward passes only.

*A cell in use.* For a cell at one position of an item, with update gate z, candidate n and previous state h (so that the new state is (1 - z) n + z h), every change made at that position to the cell's incoming connections is a product that contains the factor (1 - z)(1 - n²) + z(1 - z)|h - n|, under back-propagation and under the sweep alike. I call it the change factor. Where it is zero the cell's incoming connections cannot change, whatever the error. Every change to a connection that leaves a cell is a product that contains the cell's state.

*Directions in use.* A matrix of connections receives input vectors. The directions in use by a set of items are those spanned by the input vectors that the items present to the matrix, with a constant appended for the bias. A change of the matrix that is zero on those directions leaves the matrix's output on those items as it was; if this holds at every matrix, the whole forward pass on those items is as it was.

### 29.2 Figures

**In the born network, before any unrelated learning, every cell is in use on unrelated items.** The unrelated task's bytes (128 to 225) occur in no item of the first family.

| Seed, rule | Change factor on unrelated items: median over the 256 cells; cells below 0.05 | State size on unrelated items: median; cells below 0.05 | Correlation over cells between the change factor on family items and on unrelated items |
|---|---|---|---|
| 73129, sweep | 0.39; 0 | 0.45; 0 | +0.73 |
| 73129, back-propagation | 0.37; 0 | 0.53; 0 | +0.76 |
| 93683, sweep | 0.33; 0 | 0.40; 0 | +0.80 |
| 93683, back-propagation | 0.28; 0 | 0.49; 0 | +0.72 |
| 116089, sweep | 0.36; 0 | 0.45; 0 | +0.81 |
| 116089, back-propagation | 0.24; 2 | 0.58; 0 | +0.78 |

**The change is spread over all cells, and the loss comes before the gain.** After 64 unrelated updates (seed 73129) each cell's incoming connections have changed by 12 to 26% of their size (5th to 95th percentile of cells, both rules), and half of the squared change lies in 90 to 99 of the 256 cells, where an even spread would give 128. At that point the first family is at 0.1% and the unrelated task at 0.0% under both rules: the family is destroyed before anything of the new task has been learned.

**With rehearsal the weights move as far, in another direction, and the family is intact along the whole way.** d1 is the change made by 4,096 unrelated updates without rehearsal, d2 with it.

| Seed, rule | Size of d1, of d2, against the size of the birth's weights | Cosine between d1 and d2 | First family on the straight line from the birth to the unrehearsed network, at 0.1, 0.2, 0.3 of the way | Lowest value of the first family on the straight line from the birth to the rehearsed network |
|---|---|---|---|---|
| 73129, sweep | 0.66, 0.71 | +0.28 | 92.8, 71.9, 35.1 | 98.3 |
| 73129, back-propagation | 0.57, 0.65 | +0.39 | 86.4, 53.3, 23.6 | 98.8 |
| 93683, sweep | 0.59, 0.65 | +0.37 | 97.8, 89.7, 52.2 | 99.4 |
| 93683, back-propagation | 0.53, 0.61 | +0.47 | 92.0, 67.4, 24.8 | 98.8 |
| 116089, sweep | 0.62, 0.70 | +0.30 | 97.8, 81.0, 31.5 | 97.6 |
| 116089, back-propagation | 0.54, 0.62 | +0.43 | 90.7, 56.0, 9.3 | 97.9 |

On the line to the unrehearsed network the unrelated task is at 0.0 to 3.7% half-way and at 61 to 90% at 0.7 of the way, so no point of that line holds both. On the line to the rehearsed network the unrelated task rises from 0 at the half-way point to 60 to 89% at three quarters and 97 to 100% at nine tenths, with the family at 97.6% or above throughout.

**Directions in use by the first family's taught items** (four operations, 0 to 12 letters, 2,080 items; the number of directions that hold the stated share of the trace of the inputs' second-moment matrix; ranges over the six births).

| Matrix (input dimensions with the bias) | 99.9% | 99.99% | Not zero |
|---|---|---|---|
| Input map (513) | 9 | 12 to 15 | 46, the number of bytes in use |
| Recurrent matrix (257) | 239 to 244 | 255 to 256 | 257 |
| `lm`, at the answer positions (257) | 243 to 248 | 255 to 256 | 257 |
| `l2`, at the answer positions (513) | 9 | 9 | 11 |
| `l3`, at the answer positions (513) | 249 to 361 | 361 to 452 | 513 |

**The code of a byte has nine directions, and every byte shares them.** The code of a byte is its row of the byte table, 8 numbers, passed through `l1`. Of the squared size of the code of a byte of the unrelated task, none of which has occurred before, 99.9% lies inside nine directions of the codes of the family's bytes, and 99.98 to 99.99% inside the span of the 46 bytes in use (six births; a random direction would have 9%).

**A change projected away from those directions.** For each matrix the change d1 (or d2) is multiplied on the input side by the projector on the complement of the family's directions; `l1` and the byte table's rows of bytes in use are put back to the birth's. With the directions that hold 99.99% of the trace removed, the first family is as at birth to within 0.2 points in every case, on reduction pairs and on held-out pairs at 12 letters alike (for example 95.9 and 97.5 against the birth's 95.9 and 97.5); removing as many directions chosen at random leaves it destroyed or damaged (4 to 83% on reduction pairs). The unrelated task is at 0.0% in every projected network.

**Where the change lies.** Of the squared change of the input map made by the unrelated task, 51 to 61% under the sweep and 80 to 85% under back-propagation lies in the nine leading directions of the family's inputs, where a change in random directions would have 1.8%; the figures with rehearsal are the same (54 to 66% and 82 to 84%). (The optimizer scales its steps entry by entry, so a change need not lie wholly in the directions of the inputs that caused it.)

### 29.3 Readings

1. **Nothing is left unused, so nothing can be left unchanged.** In the present cells every item drives every cell, including an item made of bytes the network has never met. A rule that changes only the connections in use therefore changes all of them. This is what E43 and its addendum found from outside, and it is why limiting the connections between cells did not help: the limit removes connections, it does not make any cell idle.
2. **One family fills the state of the cells.** At the recurrent matrix the family's own items use 255 or 256 of the 257 directions. A rule that leaves the family's directions untouched would be exact (the projected networks show it, also on pairs never taught) and would leave no room at all for anything else at that matrix.
3. **The input code gives a new byte no direction of its own.** At the input map the family's bytes use 46 of 513 directions, which looks like room. It is not: the code of any byte, met or not, lies in the same nine directions, because it is the image of 8 numbers. A byte that has never occurred therefore drives the input map along directions that the family's bytes already use, and a rule that leaves those directions untouched leaves the new byte nothing. With a code in which a value stands for one byte, as in E50, a new byte would drive only connections that no earlier byte drives. This is an argument for E50's code from the side of keeping; it was drafted for the side of computation.
4. **Rehearsal does not work by holding the weights near their old values**, nor by keeping each matrix's action on old inputs unchanged. Its change is as large as the unrehearsed one and lies as much in the family's input directions. What it finds is a distant set of weights with the same function for the family, reached along a line on which that function never fails. A mechanism that pulls every connection back toward its old value is therefore not a substitute for rehearsal: on the line to the unrehearsed network the family is already below 90% at a fifth of the way.
5. **The family is lost before the new task is learned.** The loss is not the price of the new solution. It is made by the first few dozen updates, in which the connections of every cell are moved by an eighth to a quarter of their size for a task of which nothing has yet been learned.

The rule "change a matrix only along input directions that earlier tasks do not use" is in the literature: Zeng, Chen, Cui and Yu (arXiv 1810.01256) for feed-forward networks; Saha, Garg and Roy (ICLR 2021); and for recurrent networks Duncker, Driscoll, Shenoy, Sahani and Sussillo (NeurIPS 2020), who report that it removes interference between tasks learned in sequence and that the networks reuse structure across similar tasks. I have read their abstracts, not the papers. What the measurement adds for our network is the count of reading 2: after one family there is no room left where the rule would need it.

### 29.4 What follows (argued; the test is the job of section 30.3)

Keeping what has been learned without rehearsing it needs two things together: a part of the network that earlier learning has left unused, and a rule that confines new learning to that part. The record now has each of the two separately.

- E48 supplies the unused part from outside: a term that is present only when the new family's code is present. In the terms of this section the gate multiplies the number of input directions by the number of codes, which is why it finds room where a plain change finds none.
- E41 shows that the first family does not need the whole network: a birth with 128 cells reaches the threshold, at less computation than one with 256.

So a network of the present size can be born using half of its cells, with the other half held in reserve, at no loss. `e51_design_checks.py` checks the arithmetic of this on E41's six half-width births, without training. The half-width birth is placed inside a network of the present size; the 128 added cells have update gates that start shut, so that they are silent on every item of the first family. The embedded network answers the family as the half-width birth does (largest difference of an output 0.004 with the gates' bias at 10). The family then uses 129 of the 257 directions at the recurrent matrix, and 128 are free. A random change of every connection that the rule of section 30.3 allows to change, a sixth to a quarter of the whole network's size, leaves the family's scores as they were on all six births; the same change made without the rule halves them (39 to 58%). One limit follows from reading 3. In the present code the reserve cannot be reached through directions that are free, because a new byte has none; it can be reached only through connections that are free, which are the reserve cells' own.

## 30. Decisions, and the plan from here (revision 21)

### 30.1 Where the adopted question stands

The question (review 9.148): under fixed memory and continued learning, can a learner whose changes are confined to the connections in use keep what it has learned while learning new things, at no greater computation to threshold than back-propagation?

**Computation, for first learning.** In the dense network the sweep needs 1.1 times back-propagation's updates when half of the pairs are taught (28.1). Every structure tried so far costs more computation to threshold than the dense network under back-propagation: a limit of ten connections a cell 1.4 to 2.7 times; the same with periods 1.0 to 1.7 times; the interface at a boundary 1.7 to 2.5 times; an interface in which a cell reads part of the input about 1.6 times at best under back-propagation (1.3 times `F10`'s), and under the sweep it is not learned. What lowers the computation is a smaller network: the half width reaches the threshold at about half the computation of the present width under both rules. On this task, then, sparsity and locality have been costs. Their case rests on networks with many more cells, and the task does not need more cells: E47's cells away from the boundary are not used. More cells can pay only where there is more to hold, which is many families. The computation line and the keeping line therefore meet in one experiment, in which cells that the first family does not need are the room for what comes later.

**Keeping.** Without rehearsal nothing is kept: not through 64 updates of an unrelated task (E43), not from one visit to the next in the long-life world (28.8). Section 29 gives the reason: every cell is in use on every item. With rehearsal the family is kept, at twice the items an update (28.6). With a part of the network that only the new family uses, supplied from outside, the earlier families are kept exactly and the new one is learned (28.7).

**Fixed memory and continued learning.** The instrument is calibrated and audited. No life has been run.

### 30.2 Decisions on what waited for me

| Item | Decision | What it needs |
|---|---|---|
| The runner change and the roster note of the first long lives | Sound with conditions; nine changes as text (audit, section 8); the lives start after them without a second audit | You enter the changes; the worker makes the two runner changes, runs the checks and starts the trial |
| E47's roster (4,096 cells; scattered periods) | Not released | Nothing. A larger sheet would add cells of the kind the trial shows unused, at 37 CPU-hours. The line rests until a start is found at which differences between items reach the far cells; that can be studied on starting weights without training |
| E49, continuation | None | Nothing |
| The E50 brief | Run, changed: nine changes (audit, section 9). The settings `C64` and `C16` are E49's masks, which failed on the learned code, so their failure here would say nothing. Stage 1 is the fixed code on a dense interface (`C-dense`), both rules; stage 2 a limited input with the dense readout, where stage 1 reaches 80% on R; stage 3 a control in which every cell reads one value of every used byte, only if stage 2 fails | You enter the changes; stage 1 can start at once |
| E48 | Closed as read in 28.7 | Nothing |
| E41, E42, E46 | Closed as read in 28.1 to 28.3 | The three-seed report of E41 (2722d1c1) and E46 (b76f5979) are cited by the roster note and are not merged into the default branch |
| E43's second addendum | Closed as read in 28.6 | The report's list of committed weights marks the two `P8` files as incomplete |

On E50, one finding of the audit matters beyond the brief. Under the sweep with a limited input, the code of a byte shrinks until two bytes that must be told apart are 3 to 6 apart, where they are 14 to 16 apart under back-propagation and 14 at the start. E50's code is fixed and cannot shrink. Stage 2 under the sweep therefore tests one explanation of E49's failure: if the sweep learns with a limited input once the code is fixed, the shrinking code was the cause; if it still does not learn, it was not.

### 30.3 Job E51: a new task learned in cells held in reserve, with what the first family uses left unchanged

**Why.** Section 29.4. E48 kept the earlier families by a part supplied from outside. This job gives the network such a part of its own, cells that the first family does not use, and a rule that confines new learning to it, and asks two things: is the first family kept, and what does the new task cost to learn? The arithmetic of the construction is checked in `e51_design_checks.py` (three seeds, both rules, no training).

**The rule of section 20** for any added mechanism is met as follows. (a) The deficit is measured on a learner that generalises: E43. (b) The plain baseline is in the same experiment at equal updates: the arm `rehearse`. (c) The expected effect is stated beforehand, below.

**The learner.** E41's half-width birth with half of the pairs taught (`e41_<rule>_h256_f50_<seed>.pt`, on the default branch), placed inside a network of the present size (feed-forward width 512, 256 cells) as the function `embed` of my script does it. The old weights keep their places. Weights from the 256 added feed-forward units into old units are zero. Every other added weight is drawn at the scale of a fresh network, those from the added cells into old cells included, since without them no error would reach the added cells. The 128 added cells are the reserve.

**Directions in use.** After the embedding, 2,080 taught items of the first family (40 for each operation and each number of letters from 0 to 12) give, for each of the five matrices (input map, recurrent matrix, `lm`, `l2`, `l3`), the second-moment matrix of the inputs it receives, a constant appended; the directions in use are its eigenvectors whose eigenvalue exceeds one millionth of the largest. My script finds on the six births: 129 of 257 at the recurrent matrix and at `lm`, 36 to 41 of 513 at the input map, 11 at `l2`, 329 to 437 at `l3`. They are computed once, before the new task, and stored (0.23 to 0.28 of the network's parameters; 0.08 without `l3`'s).

**The rule.** "Projected" means: the gradient of the matrix is multiplied on the input side by the projector on the complement of the directions in use before it enters the optimizer, and the optimizer's step is multiplied by it again before it is applied. The second multiplication is what makes the keeping exact.

- Rows of the input map and of the recurrent matrix that belong to old cells, and every row of `lm`, `l2` and `l3`: projected.
- `l1`, and the byte table's rows of the 46 bytes in use: not changed. The byte table's other rows: free.
- Candidate and reset rows of the reserve cells: free.
- Update-gate rows of the reserve cells: free. On the first family's items the gates are held shut by the condition of the arm `keyed`.
- No weight decay in any arm (decay would have to be confined as well, which is a later question). AdamW otherwise as in E43.

**Arms,** each from the same embedded network.

| Arm | What may change | What it shows |
|---|---|---|
| `open` | Everything, as in E43. The reserve cells' update gates start shut (a bias of 10) and are free | The loss without any rule, and the speed of the new task without any limit |
| `rehearse` | As `open`, with one item of the first family for each new item, as in E43's second addendum | The plain baseline |
| `keyed` | The rule. The reserve cells' update gates are held shut on every item whose first byte is one of the bytes in use; on every other item they are those of a fresh network, and free | Whether the first family is kept, and whether the new task can be learned in the reserve |

In `keyed` the condition is computed from the item (has its first byte occurred before?) and is not supplied for a family. It is still a device of the harness, and the report says so.

**An arm that is left out, and why.** The arm one would want beside `keyed` has no device. In it the reserve cells' gates start shut, their update-gate rows are projected like the old cells' rows, so that they stay shut on the first family's items whatever is learned, and only the optimizer can open them, through directions that the first family does not use. In the present code this arm cannot work, by reading 3 of 29.3: a new byte has no direction of its own (0.1% of its code), so nothing can open the gates. It becomes possible with E50's code, in which a new byte has eleven values of its own. It is a job for after E50's stage 1, on a birth with that code, and its question is the one that matters for a learner without any device: does the optimizer bring unused cells into use for a new input?

**Tasks.** `unrelated`: E43's task, stream and panels, 4,096 updates. `family`: the first block of E26's life (the family of modulus 97, with E26's taught fraction, stream and scores), its first 4,096 updates. For a born network of the present size the record has the conventional outcome of that block: the new family is learned and the birth family's reduction is lost (sections 3.3 and 3.4). Scored at updates 0, 16, 32, 64, 128, 256, 512, 1,024, 2,048 and 4,096. Each network under the rule of its birth.

**Checks before the runs.**
1. The embedded network answers the first family's panels as the half-width birth does, item for item (my script: largest difference of an output 0.004).
2. The numbers of directions in use equal those my script prints for the birth.
3. A random step that obeys the rule changes no output on the first family's panels by more than 0.05, and the same step without the projection changes the scores (my script: 39 to 58% left).
4. In `keyed`, at every scored row, the first family's answers are those of row 0 on at least 99.9% of items. This is by construction, and a failure is an error of implementation.
5. The batch digests of the two tasks equal E43's and E26's.
6. A dry run with the cost a update of each arm.

**Recorded at every scored row.** For the first family: the reduction score, the four operations on taught pairs, the held-out panel at 12 letters, and what the wrong answers are, in the classes of E43's addendum. For the new task: E43's two panels, or E26's scores of the new family (never-taught reduction pairs, and taught pairs). For the reserve: the number of reserve cells whose update gate is open by more than 0.1 on average over the new task's items, and the largest state of a reserve cell on the first family's items. The sizes of the weights. The weight files at updates 64 and 4,096, listed in the report from `git ls-tree`.

**Reading, fixed beforehand,** for each rule, task and arm. *Kept*: the first family's reduction score at update 4,096 is within 2 points of row 0. *Learned*: the new task's score on its never-taught items reaches 80% within the dose; its cost is the first scored row at 80% against that of `open`.

**Trial first.** Seed 73129: twelve runs. I estimate 2.5 CPU-hours with scoring, from E46's 0.11 CPU-seconds an update at this size and a fifth more for the projection. The other two seeds are released after the trial is read.

**What I expect, on record.**

| | Statement | My chance |
|---|---|---|
| P51.1 | `open` loses the first family on `unrelated`: below 5% on reduction pairs by update 64, both rules. Refuted if the reserve takes the new task and the family survives | 90% |
| P51.2 | `open` loses the first family's reduction pairs on `family` (below half of row 0 at update 4,096), as E26's lives did | 80% |
| P51.3 | `keyed` keeps the first family on both tasks, both rules. By construction | 97% |
| P51.4 | `keyed` learns `unrelated` to 80% within the dose | 60% under back-propagation, 45% under the sweep |
| P51.5 | `keyed` learns `family` to 80% on never-taught reduction pairs within the dose | 50% and 35% |
| P51.6 | Where `keyed` learns `unrelated`, its first row at 80% is at most four times that of `open` | 60% |
| P51.7 | `rehearse` keeps the first family within 5 points on both tasks, both rules | 85% |

If P51.4 fails under both rules, cells in reserve that read the old cells but may not change them are not enough to learn a new task, and the candidate for the long lives is E48's term, chosen by the item's first byte. If P51.4 and P51.5 hold, the reserve learner is the candidate, and the question that remains for it is that of the arm left out.

**What it cannot show.** That a learner can do without the device of `keyed`. How a reserve is shared among many tasks: here the first new task may take all of it. A life. Whether a network can be born with a reserve by itself: here the reserve is made by taking a narrower birth. A comparison with a store at equal memory, which belongs to the long-life world.

### 30.4 Order and pace

Three lines of work, which is the limit of the working rules.

1. **The first long lives** (release 1 of L2): the trial, then the roster.
2. **E51**, the trial seed. The brief comes to me before any worker starts.
3. **E50**, stage 1.

Release 2 of the long lives (the sweep, and a learner that keeps without a store) is specified when the trials of lines 1 and 2 have been read. Its candidate is the reserve learner if E51 shows that it keeps and learns, and E48's term chosen by the first byte otherwise. E50's stage 1 has a second use beside its own question: it gives the births on which the arm left out of E51 can be run.

Resting: E47 at larger sizes; a fourth, narrower width for E41, which would locate the cheapest width (six births, about 2 CPU-hours); a corrected rule for replacing connections (22.12). After release 1, step 5 of the adopted plan: confirmations on sealed tasks of E40's result, E48's and E43's.

### 30.5 What is not proposed

A pull of every connection toward its earlier value is not proposed. E38's anchors gave no protection (section 20), and section 29 shows why: the family is lost a fifth of the way along the straight line to the unrelated task's solution, and the solution that rehearsal finds is as far from the birth as the unrehearsed one. Schemes that weight the pull by an estimate of each connection's importance are in the literature; they are a different mechanism from the one above, and by the rule of section 20 they are not candidates until a deficit remains that the store and the reserve do not cover.

### 30.6 On coordinating while Mr. Xu is away

Everything in this Part was done inside the reviewer's role: checks from receipts and weights, two audits, one measurement, one proposal. I have written to no worker session and started no run. The decisions of 30.2 reach the workers through you, as before.
