# Memo to the coordinator: what the saved GRUs have built, what a life takes away, and three jobs to brief (15 November 2026)

From the outside reviewer, at Mr. Xu's request: "Write an appropriate memo for the coordinator to implement."

**Revision 2, the same evening.** E33 addendum 1 and E34 stage 1 have reported on `claude/vibrant-curie-ivmrt1` (ba6f5d52). I recomputed both tables from the committed weights; section 14 gives the check and what it changes. Jobs A, B and C are unchanged. One item is added to section 1 (item 5), one bullet to section 4 and three rows to section 12.

**Revision 3, later the same evening.** Three things are new. (a) The audit of the E36, E37 and E38 briefs is in `AUDIT_E36_E37_E38_BRIEFS_20261115.md` beside this memo; two of its required changes correct marks I proposed here, and sections 5 and 7 now carry the corrected marks. (b) Table 3.3 is restated with whole answers, as the harness scores them; the earlier table counted the two digits only. (c) Part II (sections 15 to 18) sets out the programme from here, after Mr. Xu's statements of this evening: a continual learner judged over a long life. Items 6 and 7 of section 1 are the new requests.

**Revision 4.** Section 17.10 is added: how short and long runs are combined (rate laws, trunks and branches, two probes, agreement between lengths, time in the learner's own units). It adds requirements to the protocol brief of job L0 and two items to L2; nothing else changes.

**Revision 5.** The audit of the E39 brief is in `AUDIT_E39_SMALL_WORLD_PROTOCOL_20261115.md`, with the second implementation in `e39_world.py`. It changes three things in this memo's Part II; section 17.11 lists them. Where 17.11 and an earlier section differ, 17.11 holds.

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

