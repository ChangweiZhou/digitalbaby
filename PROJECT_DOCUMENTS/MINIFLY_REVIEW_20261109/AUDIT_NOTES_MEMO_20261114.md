# Audit notes of 13 and 14 November 2026 that had not been pushed

Written by the outside review session on 14 November 2026. It collects, in one file, every note this session gave Mr. Xu after the note of 11 November (`REVIEW_AND_NEXT_STEPS_NOTE_20261111.md`, commit `78c099fa`), so that none of it lives only in a chat window. Nothing in it was trained here: every figure comes from receipts and saved weights read with the standard library or numpy.

## How to read this file

- **Section 1** is a digest written for this memo: where the work stands, what is open, and which of my own earlier claims turned out wrong.
- **Notes 1 to 11** are the notes as they were given, in order. Each has its time and what prompted it.
- **The last section** lists the scripts behind the figures and how to run them.

What was changed in the notes:

- Closing offers and questions to Mr. Xu were removed.
- Two commands in note 9 that pointed at a scratch folder now point at the same scripts in this folder.
- Headings were lowered so the notes nest under this file's sections.
- Where a later note overturned or settled an earlier claim, a line marked *[Added 14 November: ...]* says so. Nothing else was edited, so the notes still say "you" where they addressed Mr. Xu.

Times are the session's, in UTC. The repository's own clock is Eastern time, five hours behind: note 9, at 04:10 UTC on 14 November, was given at 23:10 on 13 November by that clock.

## 1. Digest

### Where the work stands

- **Birth.** The sweep (the backward pass with every weight matrix replaced by its signs times one mean magnitude) brings the 256-unit GRU to birth level with backprop under answer targets: 98.0 / 98.2 / 99.8 against 98.5 / 98.7 / 93.7.
- **Life.** Under the sweep the network composes untaught operator pairs from memory at 37–54%, against 63–89% under backprop. The registered marks are missed, and the crossover run shows the gap comes from the update used during the life, not from the birth.
- **Every-byte targets.** On the GRU the sweep does not collapse; it ends at 71.7–74.8 against 93.5–97.7 (E28, all twelve births). The collapse seen in E24 belonged to the two-bank network.
- **Past the trained range.** This is the largest gap on file. Born under every-byte targets, the sweep answers mod-89 addition at 71–80% with 12 letters between the operands and 2% with 40; backprop holds 80–85% at 40. Births train on 0 to 12 letters.
- **Mechanism, as far as it is known.**
  - Registered (E30): the sweep leaves the update gates undecided. Three of four predictions held.
  - Post hoc, unregistered: the every-byte sweep birth has 50–62 units at update 4,096 that keep half their state across 12 letters, and loses nearly all of them by update 12,288 while its accuracy at 12 letters rises. Their mean pre-activation at letter steps falls from about 4.0–5.0 at update 4,096 to about 2.7 at update 8,192. The gate rows' weight norm rises over the same rows, so this is not weight decay.
  - Post hoc, unregistered: in lives, the sweep learns the same procedure as backprop and executes the second step less precisely (31–46% second-step failures against 7–17%; 75–91% of mistakes have exactly one digit right).
- **What missed.** My hypothesis that the sign path's missing magnitudes explain both failures (E27). The coordinator's reading that the sweep's networks hold as much as backprop's (E29 part A), which the receipts and then E30 overturned.

### Open items

1. **E32 brief, revision 3** (`bc6a0eaf`). Not cleared. Six text changes are listed at the end of note 11; with them made as written it can go to workers without another read.
2. **E31.** Stage 1 agreed between the two workers on four arms. Stage 1b (the bounded push and the sign-start feedback) and stage 2 are with the workers. Two things are unresolved from my side:
   - At 03:51 UTC on 14 November Mr. Xu sent this session an instruction to "do stage 1b, and if the two of you agree, go straight to stage 2. Do not run arm Ks at full length." I judged it was meant for the two E31 workers and did not act on it; this session reviews and does not train.
   - That instruction conflicts with the brief's addendum 3, which releases Ks to worker B. Worker B should be told if it has not been.
3. **Repository.**
   - The merge `004b6b9f` put 856 files of unrelated October work (17 MB) on the default branch.
   - The resume gate (E21b) leaves five options the loop reads outside its command shape: the replay capacity, `E11_REPLAY`, `E11_TRACE`, `E11_GATES`, `E11_WD`.
   - No receipt made on the owner's machine records its torch version; the record gives 2.14.0 in one place and 2.9.1 in another.
4. **Unregistered measurements that are being built on.** Keeping units, second-step errors and error kinds are all post hoc. E32 registers keeping units as a measure; the other two are not registered anywhere.
5. **A rival explanation not yet measured.** The outside literature report raised step length along the true gradient (curvature) as an alternative to gate precision. One cheap measurement would test it.
6. **Target test.** On hold. Its alphabets clash with the roles the birth gives those bytes, and at its proposed size the stream gives 281 updates where a fixed carry of 16–20 bytes took 750–3,750.
7. **Process without Codex.** The working substitutes are a second read of every brief before a worker starts, each finding carrying the command that shows it, and two workers writing each new piece independently. This session is the same model family as the coordinator and workers: a second reader, not an independent one.

### My own errors, for the record

- **The magnitude hypothesis (note 1, item 4).** Tested as E27; its registered predictions missed.
- **The byte-clash explanation for the slow born start (note 4, item 3).** The rerun with clean bytes showed it accounts for one birth's slowness only.
- **A conjunctive-program idea for the composition gap.** My own error analysis refuted it: wrong-operator and wrong-operand mistakes are 0–4%.
- **The owner's machine's torch version.** My note of 11 November repeated the record's 2.14.0; the gate's refusal shows 2.9.1.
- **Letter-specific lean of the gates (note 9, point 2).** True of an untrained network, and it was the reason to move the push's switch-on. At update 4,096 two thirds of the units lean one way.
- **K's window (note 10, point 5).** I asked for the window 4,096 to 12,288 to be named; the feedback matrix is also off before 4,096, and the arm cannot place the need inside that window by itself.
- **A first-order model of the hinge (note 11, point 3).** Applied to the floor alone it predicts the wrong sign. I used it for the hinge's side only and said so.

## 2. Note 1: next steps without Codex

**When:** 13 November, 16:19 UTC.

**What prompted it:** Asked to discuss next steps after access to Codex, the outside auditor until then, was lost. E26, E26-X and E21b were still unmerged on worker branches.

### Where things stand

The default branch stops at the morning of 12 November. The E26 lives, the crossover run (E26-X) and the E21b requalification sit unmerged on worker branches. I recomputed E24, E26 and E26-X from their receipts and every cell matches the workers' tables.

- **Birth:** the sweep brings the GRU to birth level with backprop (98.0 / 98.2 / 99.8 against 98.5 / 98.7 / 93.7).
- **Life:** the sweep now composes from memory, but under the registered marks (85 on new instances of taught pairs, 40 on untaught plus-minus pairs). No candidate pair exists by the registered reading.
- **Crossover:** the gap comes from the update used during the life, not from the birth.

| Birth → life update | New instances of taught pairs | Untaught plus-minus pairs | Both marks |
|---|---|---|---|
| sweep → sweep | 83.5 / 83.8 / 81.0 | 54.2 / 44.5 / 37.2 | 0 of 3 |
| backprop → sweep | 85.5 / 81.7 / 83.0 | 39.8 / 45.8 / 53.9 | 0 of 3 |
| sweep → backprop | 93.2 / 97.3 / 93.0 | 72.3 / 82.8 / 73.9 | 3 of 3 |
| backprop → backprop | 84.7 / 93.7 / 87.2 | 63.2 / 89.2 / 66.3 | 2 of 3 |

- **Every-byte targets (E24):** on the two-bank network the sweep's birth collapses (0.8 / 1.0 / 1.2) and backprop's does not (91.5 / 89.3 / 91.2). Giving the answer half the loss restores it (81.5 / 79.8 / 83.3).
- **Target-test sizing:** a GRU trained from random weights copies a byte 12 positions back and learns nothing at 24 or more.

### What losing Codex costs

Between 10 and 12 November Codex stopped four runs that would have been wrong as briefed: E20-A, E22, E26 (the wrong update loop) and the sizing sweep (12 of 18 settings scored against the wrong modulus). The coordinator confirmed each by running it. That is roughly one run-invalidating defect per brief.

What can stand in:

- **A second reader:** I can read every brief before a worker starts and every result before it enters the record. I am the same model family as the coordinator and workers, so this is a second reader, not an independent one.
- **Mechanical checks for what Codex caught most:**
  - every claim about code in a brief carries the command that showed it;
  - every number comes from a script over receipts;
  - each pass mark has one responder that must reach it and one that must not;
  - one real update step per arm is checked against an independent computation.
- **Smaller steps:** prefer runs that flip one switch on qualified code, as E26-X did, over new training loops.

The coordinator's section on the overnight results (review 9.110) also says the coordinating session now runs on the same model as the workers, which makes shared blind spots more likely.

### Next steps I would suggest

1. **Land the unmerged work first.** Merge the three branches, with the coordinator's reading. Also commit two things that are not in the repository:
   - the memo E26-X was authorised on (`MINIFLY_E26_RESULT_AND_NEXT_STEPS_20261112.md`, on the owner's Mac);
   - the end-of-life weights of the six E26 lives, which live only on the worker's container.
2. **Keep the marks.** The close on 17 November can say: born level with backprop, composes at 37–54% against 63–89%, marks missed, the gap is in the later update.
3. **Build on one pattern in the receipts.** At equal skill on taught pairs (75–88), lives updated by the sweep score 43 on untaught pairs on average and lives updated by backprop score 64. So more training is unlikely to close the gap; the sweep's solution is less compositional.
4. **Test one hypothesis for both failures, without training.** Weight signs carry no magnitudes. That could explain why unpredictable-byte error floods back (the every-byte collapse) and why credit does not favour circuits that already do the work (weaker composition). Two measurements on saved weights would check it: the share of filler error reaching each layer under exact and sign paths, and alignment on composition items against single operations.

   *[Added 14 November: this was run as E27. Its registered predictions missed, this hypothesis included; see notes 2 and 4.]*

5. **If it holds, try learned feedback weights,** which are already ruled inside the line. Run them as lives from the existing sweep-born births, with bounds on file (37–54 below, 72–83 above), and as one every-byte birth. The source check comes first.
6. **Do not shorten the target test's distances yet.** E20-B's GRU held operands across 40 letters, so the 12-byte span looks like a product of training from scratch at one fixed delay. Rerun the copy fixture with mixed delays, a gate biased to keep, and E26's born weights.

### What my check found beyond the tables

- 9.110 reports the birth-curve differences as averages over seeds; they are sums, so the true averages are a third as large. The reading is unaffected.
- The three crossed lives finished by a new resumable runner equal the first runner's interrupted receipts at every stage those reached (5, 5 and 6 of 7).

Not checked: E21b's code (I read its report, which lists the refusal tests Codex asked for) and the sizing code. I ran no learner.

## 3. Note 2: interim report on the five deliveries

**When:** 14 November, 00:44 UTC.

**What prompted it:** Asked to look at what the Opus 5.5 workers had shipped, and to be sceptical after the previous round. Three parallel checks were still running when this was sent.

Interim report: E27, E28 and the repository are checked; the span roster, the gate measurements (E29 part A) and the literature source checks are still being audited in the background, and I'll send the full review when they return.

### E27 (measurements on saved weights)

- **The numbers stand.** I recomputed every figure in reviews 9.117 and 9.119 from the measurement files and all match.
- **The registered predictions missed, including my own hypothesis.** Amplification of non-answer error does not single out the collapsed birth, and alignment on untaught pairs is no worse in networks that lived under the sweep.
- **The coordinator's summary is stronger than the worker's report.** The report calls the first half "mixed", as the pre-registered rule requires. Under weight signs the distractor letters' error is amplified 1.9–2.8× on the sweep-born file and about 2–3× in every trained GRU. Review 9.119 says "not supported in either half". The accurate statement is that the mechanism is present but does not explain the collapse.

### E28 (GRU born under every-byte targets)

- **The brief carries a false premise that passed both the hostile audit and the coordinator's check.** It says E24's sweep "cut credit in time and used the trace routine". Lines 91–94 of `exp16_integrated.py` force credit through time and switch the trace routine off under that setting, and the E27 worker's executed check shows the same gradient.
- **Consequence:** the sweep arms of E24 and E28 differ only in the network. The brief's "survival" reading (no attribution among three differences) rests on a difference that does not exist.
- **The runner itself checks out.** Its two host-check births equal E26's in all 13 tensors, by my own comparison.
- **First births, not yet read by the coordinator:**

| Arm | Seed 73129 | Seed 93683 |
|---|---|---|
| Sweep, every byte | 74.8 | 74.7 |
| Backprop, every byte | 93.5 | 97.7 |
| Sweep, balanced | 86.8 | interrupted |
| Backprop, balanced | 92.3 | interrupted |

  So there is no collapse on the GRU. By the brief's own rule this is neither survival nor collapse. The third seed was interrupted by a container restart.

### Repository

- **Unrelated history merged.** The merge of the E24 worker's branch (`004b6b9f`) put 856 files of old October work (17 MB, mostly old `minifly_v1xx` experiment folders) on the default branch. Review 9.110 had deliberately kept that history out.
- **Resume gate (E21b) merged with a known finding open.** Codex's last memo said the gate's command shape omits the replay capacity. My read adds four more options the loop reads that are outside the shape: `E11_REPLAY`, `E11_TRACE`, `E11_GATES`, `E11_WD`. The worker's sentence that the shape holds every option that changes the loop is not true.
- **Crossover (E26-X) stands.** I ran the check Codex's memo asked for and the pilot skipped: lives sharing a birth start from identical scores in all three seeds.
- **Owner's machine.** The record gives its torch version as 2.14.0 in one place and 2.9.1 in another, and no receipt made there records it.

## 4. Note 3: the gate measurements (E29 part A), interim

**When:** 14 November, 00:53 UTC.

**What prompted it:** The second interim report from the same audit.

The gate-measurement audit is back and I've reproduced its key evidence; only the literature check is still running.

- **The delivery's numbers are all correct**, but review 9.120 C's reading ("the sweep's networks hold as much", so holding is not the explanation of the weaker composition) does not survive a closer look.
- **Evidence in the receipts themselves:** at the end of life, going from 12 to 40 distractor letters on mod-97 addition, lives updated by the sweep lose 7.5 to 11.5 points and lives updated by backprop lose 0.2 to 2.2, in all three seeds and from either birth. The loss follows the life update, as the composition gap does.
- **Inside composition items** (a post hoc measurement, not registered): units that keep their state across the second operator, the third operand and `=` number 6 to 21 in sweep-updated lives against 19 to 60 in backprop-updated lives.
  - The forward pass used for this reproduces every receipt count exactly.
  - Over the twelve lives the count tracks composition accuracy (rank correlation 0.84), but mostly because it separates the two updates. Among sweep-updated lives alone it does not track accuracy (0.12).
- **Why the delivery missed it:** the "rewritten somewhere" half of its holder definition pools over 300 items, so nearly every unit passes; and the lives were measured on mod-89 addition, which they answer only 2 to 40% of the time, not on their own operations.

This is a lead and not a result: thresholds were chosen after seeing the data, and nothing shows cause. It needs a registered measurement on unseen files (E28's births would do) before anything is built on it.

## 5. Note 4: all five deliveries checked

**When:** 14 November, 01:02 UTC.

**What prompted it:** The full report of the audit. The coordinator adopted its findings in review 9.122 on this evidence, without rerunning it.

All five deliveries are checked. This round the Opus 5.5 workers' numbers hold up: every table I or my checks rebuilt from receipts and weight files matched. The errors are in the layer around them: the briefs, the "hostile audit" step and the coordinator's readings. Four of those errors change a conclusion.

| Delivery | Worker's numbers | Reading that does not hold |
|---|---|---|
| E27, measurements on saved weights | correct | Review 9.119 says "not supported in either half"; the worker's own report says the first half is mixed |
| E28, GRU under every-byte targets | runner correct; 6 of 12 births done | The brief says E24 used a different sweep; it did not |
| Span roster, 48 runs | correct | "A born start is not a head start" |
| E29 part A, gate measurements | correct | "The sweep's networks hold as much" |
| Literature source checks | memo's figures correct | The audit's verdict on learned feedback in recurrent networks |

### The four that change a conclusion

1. **E28's brief misdescribes E24.** It says E24's sweep cut credit in time. The harness (`exp16_integrated.py`, lines 91–94) forces credit through time under that setting, and the E27 worker's executed check confirms it. So the two sweep arms differ only in the network.
   - E28's first births: the sweep ends at 74.8 and 74.7, backprop at 93.5 and 97.7 (two seeds; the third was interrupted).
   - Read with the corrected premise: the collapse belonged to the two-bank network, and on the GRU the sweep pays about 20 points.

2. **The sweep's networks do not hold as well.** At the end of life, going from 12 to 40 distractor letters on mod-97 addition, sweep-updated lives lose 7.5–11.5 points and backprop-updated lives 0.2–2.2. This holds in all three seeds from either birth, and I recomputed it from the receipts.
   - Inside composition items, units that keep their state across the second half of the prompt number 6–21 against 19–60.
   - That second measure is post hoc and unregistered, and among sweep-updated lives alone the count does not track accuracy. It is a lead, not a result.

3. **The born-start reading is contradicted by the roster's own table.** At distance 24 the born start passes in 3 of 3 seeds against 2 of 3 and learns sooner in every seed. A permuted copy of the born weights beats the born weights in all six pairs.
   - The fixture's keys are the capital letters the birth was trained to ignore, and two of its filler bytes are the birth's operators.
   - The target stream uses the same alphabets, so a born learner would meet the same clash there.

     *[Added 14 November: the rerun with clean bytes showed this byte-clash explanation was mostly wrong; see note 8.]*

4. **The literature audit's verdict is unreliable.** It found three recurrent studies and called none favourable; at least five exist.
   - The one it cites as "no gain over random feedback" is a Kolen–Pollack rule. The paper's main text says "similar results". My checker reads the supplementary figure as showing it ahead in three of four comparisons, by up to 7.6 points; I could not confirm those bar values from the text.
   - The audit missed the one clear null result (Marschall et al.).
   - Its forecast for E29 part B rests on one dataset of five.

     *[Added 14 November: the worker's own source check later agreed with mine on the disputed study; see note 8.]*

### Smaller defects

- **E27 wording.** The registered predictions missed, mine included. But letter error is amplified 2–3× under weight signs on every trained GRU, so the mechanism exists; it does not explain the collapse.
- **Span roster.** Two runs lost a learned copy, not one. The target stream at its proposed size gives 281 updates, where a fixed carry of 16–20 bytes took 750–3,750.
- **E29's measure.** Its "rewritten somewhere" test pools over 300 items, so nearly every unit passes. The lives were measured on mod-89 addition, which they answer 2–40% of the time.
- **The repository items from my interim report stand:** 856 unrelated October files merged, the resume gate's shape missing five loop options, and the owner's machine's torch version unrecorded.

### What I would do

1. File errata for 9.115, 9.119, 9.120 B and C, and 9.121, and correct E28's readings before its last births are read.
2. Register the holding measurement before anyone opens E28's saved rows, then run it on those unseen files.
3. Rerun the born span blocks with bytes the birth never used, and change the target stream's alphabets before any born learner meets it.
4. Require every audit finding to carry the command that shows it. The E24 claim was accepted on reading receipts, which cannot show it.

### How far to trust this

- I verified items 1 to 3 myself from source, receipts and weights.
- Item 4 comes from a delegated check that opened the papers; I spot-checked seven quotations against the downloaded text and found five verbatim.
- I ran no learner.

## 6. Note 5: the 70-entry mechanism list

**When:** 14 November, 01:19 UTC.

**What prompted it:** Asked how much of the mechanism catalogue of 8 November is still relevant, and whether there is a better plan than one agent per mechanism.

The list is the 8 November catalogue: 70 legacy entries, which its own reconciliation reduces to 20 fly mechanisms (F01–F20). Only 23 of the 70 point at one of the 20; the other 47 are excluded as non-fly, archived, or sent to other surveys. I read all 70 titles and the 20 entries' fields, not the 785-line evidence document.

### How much is still relevant

Very little for the question the main line is stuck on, and about a third of the 20 for the target test.

- **Nothing on the list addresses credit into hidden or recurrent weights.** It catalogues rules for one plastic layer, ways to manage a store, and fixed circuits. Its own entry F05 says a reward plus eligibility "does not by itself identify which hidden representation should be changed to generalize". The composition gap is exactly that problem.
- **Already absorbed.** Working state, gating, accumulation and multiplication (F10, F14, F15, F18) are what the GRU learns by itself: 41–46% of its units become holders at birth. Innate ring arithmetic (F14, F16) was run as E10 and composes at 100% because the operation is wired in.
- **Already tested at length.** The associative-store rules (F05–F08, about ten of the 70) are what the 220 pre-November versions tried. They gave a memory, not a rule.
- **Relevant once the target test starts:**
  - F05 with F08: fast writing gated by prediction error, for binding a symbol to a stated value;
  - F11: an opposing memory instead of overwriting, for revision;
  - F06 and F09: a learned teaching gate, as a source of "which bytes matter" without an answer mask;
  - F03: learned inhibition of background, for unpredictable filler;
  - F13: re-anchoring new symbols to an existing internal code, for a born learner meeting new alphabets.
- **Not relevant:** F19 (needs actions), F20 (hard-codes solutions), and the cross-species entries.

### Why not one agent per mechanism

- That is the pre-November pattern: learner, task and gate change together, so a negative result localises nothing.
- Each entry's "minimal test" compares the mechanism with a scrambled copy of itself on a task built for it. That shows the component works, not that it moves our endpoint.
- Twenty parallel briefs with no outside auditor means twenty unaudited designs, and briefs are where this round's errors were.

### What I would do instead

1. **Start from measured deficits, not mechanisms.** Each now has a floor and a ceiling on file:

| Deficit | Now | Reference | On the list for it |
|---|---|---|---|
| Composition in a life under the sweep | 37–54 | 72–83 (same births, backprop) | nothing |
| Holding from 12 to 40 letters | loses 8–11 points | loses 0–2 | F14, as wiring only |
| Every-byte targets on the GRU | 75 | 94–98; 87 with the mask-balanced loss (one seed) | F06, F09, F03 |
| Retention without replay | about 1% on hard pairs | same under backprop | F11, consolidation family |
| Fast binding and revision | not measured | not measured | F05 with F08, F11, F13 |

2. **Bound the gain before building anything.** For the composition gap, run lives from the existing sweep-born births in which one parameter group, or one kind of item, gets the exact gradient and everything else stays under the sweep. That is one switch on qualified code, about 4 CPU-hours per variant.
   - If exactness on the gate pathway alone closes most of the gap, the remedy is learned feedback on that pathway, which is already ruled inside the line.
   - If no single group does, mechanism-by-mechanism trials would have failed anyway.
3. **One bench, mechanisms as switches.** Same births, same data, same scorer written beforehand; no new harnesses. Judge at the first composition checkpoint, where the gap already shows, and continue only the leaders.
4. **Screen together, not one at a time,** if more than four switches survive step 2. Birth scores of one seed already swing by up to 14 points between adjacent checkpoints, so three seeds against three detects only huge effects. A 12-run screening design over three seeds reads each effect from 18 lives against 18.
5. **Have two workers write each new switch independently,** and require a dry run to agree tensor for tensor before any full run. E24's two programs giving byte-identical births was the strongest check this project has had, and it was accidental. This is the nearest substitute for the lost outside audit.
6. **Give the list's store-side entries their turn on the target test,** under backprop first, against a GRU-alone floor and an oracle that reads the stated values. That comes after the stream is sized and its alphabets no longer clash with the birth.

One decision is yours. The list's real proposal is the fly's architecture: fixed circuits compute, and life only learns bindings. If step 2 shows the sweep cannot be brought near backprop on composition, that is the fallback, and it changes what the learner is asked to discover.

*[Added 14 November: Mr. Xu answered the same evening that throwing away the fly's architecture is acceptable and that starting from evidence is agreed.]*

A cheap first step needs no training: force the holding units' gates fully open over the letters in the sweep-lived networks and see whether long-gap accuracy returns. *[Added 14 November: not run.]*

## 7. Note 6: what to do next, three measurements, and a literature prompt

**When:** 14 November, 01:35 UTC.

**What prompted it:** Asked what to do next, and to supply a prompt if a new literature search was needed.

Iterating on the current work is enough to choose the next experiment, and one narrow literature round is worth running beside it. I ran three read-only measurements on the saved weights, and they turn "the sweep composes less" into something specific. They are post hoc and unregistered, but the forward pass behind them reproduces every receipt count.

### What the saved weights show

- **Same kind of mistake, made more often.** On untaught pairs both updates get the first step right at similar rates (69–81% of items for the sweep, 72–96% for backprop). Given a correct first step, the second step then fails in 31–46% of items under the sweep against 7–17% under backprop.
- **The mistakes are near misses.** Under both updates, 75–91% of them have exactly one digit of the answer right. Using the wrong operator or the wrong operand accounts for 0–4%. So the sweep learns the same procedure and executes it less precisely.
- **The sweep leaves the gates undecided.** On composition items, 73–79% of update-gate values lie between 0.1 and 0.9 in sweep-lived networks, against 43–53% in backprop-lived ones.
  - This flips within one life in both directions: a backprop-born network that lives under the sweep goes from 43–52% to 76–79%, and the reverse case from 62–69% to 43–49%.
  - Weight scales are about the same, so it is not a matter of size.

Together with the receipts (sweep-lived networks lose 8–11 points from 12 to 40 letters, backprop-lived 0–2), the picture is a softer, less digital memory.

### What I would do next

1. **Register the gate measure, then read it on unseen files.** E28's births have not been looked at this way. It takes minutes and turns the lead into a result or kills it.
2. **Run one experiment of lives from the existing sweep-born births,** one switch each, with the floor (37–54) and ceiling (72–83) already on file. About 4 CPU-hours per variant:
   - exact credit for the gate rows only, as a bound: does decisive gating alone close the gap?
   - a local term that pushes each gate away from one half. It needs no feedback at all, but it is a new kind of term, so it needs your yes.
   - learned feedback on the recurrent matrices, already ruled inside the line.
3. **Have two workers write each switch independently** and require a dry run to agree before any full run.
4. **Finish E28 and close on the 17th** with the cause narrowed to imprecise holding, not a missing mechanism.
5. **Hold the target test** until step 2 reports; sizing a stream for a learner with imprecise memory would measure the wrong thing.

### Literature prompt

Run this in parallel. It is written without names or project details.

```
I need a source-checked literature review on one specific question. Open the
primary papers, including figures, tables and supplements; do not rely on
abstracts or secondary summaries. For every claim give the URL, the exact
sentence or table/figure it rests on, the network type (rate or spiking,
gated or not), the task, the number of seeds, and whether it is peer reviewed.
"Not found" is an acceptable answer; say what you searched.

Setting. A 256-unit GRU learns byte-level modular arithmetic with distractors
and two-step compositions answered from memory. Two training rules are
compared on identical data: (1) backpropagation through time; (2) the same
backward pass with every weight matrix replaced by its signs times one mean
magnitude per matrix (sign symmetry applied through time).

Observation. Both rules reach about 98% on single operations. On untaught
operator pairs rule 2 scores 37-54% against 63-89%. Its errors are near
misses of the same kind as rule 1's, three to four times as frequent. Its
update gates stay graded: 73-79% of gate values lie between 0.1 and 0.9,
against 43-53%, and this flips within one training phase when the rule is
swapped. One-step gradient alignment (cosine about 0.6) does not differ.

Questions.
1. Has anyone reported that approximate-gradient rules (sign symmetry,
   feedback alignment, DFA, e-prop, RFLO, KeRNL, target propagation) produce
   less saturated gates, less precise memory, or worse systematic
   generalization in gated recurrent networks than exact gradients do?
2. Which mechanisms make gates or memory units decisive using only
   information local to the unit: bistable cells, gate regularizers,
   saturation-promoting initializations, hard or stochastic gates? For each,
   what was shown on tasks needing memory over 10+ steps?
3. Which studies learn the backward weights of a recurrent network by a
   local rule (Kolen-Pollack, weight mirrors, learned sensitivity weights)
   and compare with fixed random or sign feedback on the same task? Report
   every arm's number, read from the figure if necessary.
4. Are there recurrent units for which exact credit through time is cheap
   and local (element-wise or diagonal recurrence with learned rates) that
   have been shown to compose learned operations on untaught combinations?
5. Negative results for any of the above.

Output. One table with a row per study; then at most five candidate remedies
ranked by evidence, each with the cheapest test in the setting above; then a
list of claims you could not verify from a primary source.
```

*[Added 14 November: step 1 was registered and run as E30, and three of its four predictions held. The local gate push was ruled inside the line (sixth working ruling). The three scripts are `error_kinds.py`, `second_step.py` and `gate_weights.py` in this folder. See note 8.]*

## 8. Note 7: status summary, in Chinese

**When:** 14 November, 01:52 UTC.

**What prompted it:** Asked in Chinese where the project stood. Reproduced as given. In short: the arithmetic stage is closing, the cause is narrowed to imprecise memory and not a missing mechanism, and the remedy experiments were about to start. Its "6 of 12 births" for E28 was true at that hour; all twelve were complete by note 8.

截至 11 月 14 日，项目处在"算术阶段收尾、原因基本定位、补救实验待启动"的状态。

### 已经确立的结果

- **规则和网络**：现行规则是"符号反馈的反向扫描"（用权重的符号代替权重做沿时间的反向传播），网络是 256 单元的 GRU。
- **出生阶段**：这条规则能把 GRU 训练到与反向传播持平（单步运算 98–100%）。
- **生命阶段**：它第一次做到了凭记忆组合未教过的运算对，但只有 37–54%，反向传播是 63–89%，没达到预先登记的标准，所以按登记的判读，目前还没有"候选组合"。
- **差距的来源**：交叉实验表明差距来自生命阶段用的更新规则，与出生方式无关。
- **逐字节目标**：在 GRU 上规则不会崩溃（约 75，对比反向传播 94–98），崩溃是旧的双库网络特有的。这个实验（E28）只完成了 12 个出生中的 6 个。
- **无重放的遗忘**：两种更新都会在难题对上掉到约 1%，这是数据流的性质，不是规则的问题。

### 我今天的新发现（事后分析，尚未登记）

- 规则学到的是同一套步骤，只是执行得不精确：第一步正确率相近，第二步出错率是 31–46%，反向传播是 7–17%，错误基本都是差一位数字的近似错。
- 规则让门控停在中间值：73–79% 的门值落在 0.1–0.9 之间，反向传播是 43–53%，而且在一个生命阶段内换规则就会翻转。
- 结论是记忆"偏软"、不够数字化，而不是缺少某种机制。

### 流程方面

- Codex 已不可用，目前没有外部审计。
- 这一轮 Opus 5.5 工作者交付的数字都可靠；错误出在任务书、"敌意审计"环节和协调者的判读上，其中四处改变了结论，勘误还没写。
- 70 条机制清单对当前的主线问题基本无关，只有约三分之一在目标测试阶段才用得上。

### 待办

1. 把门控度量登记后，在没看过的 E28 文件上验证。
2. 用现有的出生权重跑一组生命实验，每组只改一个开关：只给门控行精确梯度（上界）、给门控加一个局部的"远离 0.5"项（需要您同意）、对循环矩阵用学习的反馈权重。
3. 并行跑一轮文献检索，提示词已给出。
4. 完成 E28，17 日按原计划收尾算术阶段。
5. 目标测试暂缓，等第 2 步出结果；它的字母表与出生时的角色冲突、数据量不足这两个问题可以先改。

## 9. Note 8: E28 complete, E30, the clean-byte span run, E31 stage 1, and the literature report

**When:** 14 November, 03:45 UTC.

**What prompted it:** Mr. Xu pasted the outside literature report and said all the work seemed done. The report itself is his and is not reproduced here.

Not quite all done: the experiment that tests the remedies (E31) has not run a single full life. Four of the five jobs are complete and check out, and the Gemini report is usable with two cautions.

### What is complete, and what I verified

- **E28 (GRU under every-byte targets), all twelve births.** I recomputed them from the receipts.

| Arm | Seeds 73129 / 93683 / 116089 |
|---|---|
| Sweep, every byte | 74.8 / 74.7 / 71.7 |
| Backprop, every byte | 93.5 / 97.7 / 94.3 |
| Sweep, balanced | 86.8 / 89.7 / 88.7 |
| Backprop, balanced | 92.3 / 98.8 / 96.7 |

  The sweep pays 19–23 points and does not collapse. P28.1 and P28.4 hold; P28.2 and P28.3 miss.

- **E30 (gate measure, registered, then read on unseen files).** Three of four predictions hold, so the undecided-gates lead is now a result. I reproduced its numbers with my own forward pass and extended them to the seeds it could not reach.
  - **The largest effect so far:** sweep births trained on every byte answer 2–16% at 40 letters, against 80–95% for backprop. Sweep births trained on answer bytes hold 96–100% at 40.
  - So without an answer mask the sweep learns the task at the trained gap and nothing beyond it. That is the condition the target test imposes.
- **Span with clean bytes, 27 runs.** My byte-clash explanation was mostly wrong. It accounts for one birth's slowness; with clean bytes the born start is still twice as slow at distance 4 and still loses to its permuted copy in 3 of 3. Relabelling the bytes alone changed a random start's outcome in 2 of 6 runs, so single fixed-distance runs near the edge cannot be read.
- **Source checks.** The worker's check and my earlier one agree on the disputed study: the Kolen–Pollack variant beats random feedback in three of four comparisons, by up to 7.6 points.

### E31 is stopped at stage 1

The two workers wrote it independently, as planned. The no-switch arm, the exact-recurrent arm, the learned-feedback arm and the zero-strength gate push agree bit for bit. Two arms differ, and both workers stopped as instructed:

- **Exact gate rows:** same rule, different order of two float operations. This is rounding only.
- **Gate push:** one worker averages over padding positions and one does not. The brief did not say.

The coordinator has not read any of this; the default branch last moved before the workers finished.

Before releasing the full lives I would settle five things:

1. **Accept the rounding difference** on the gate-rows arm.
2. **Exclude padding** from the gate push; padding is not data.
3. **Change the gate push's strength.** In the dry run it drove the undecided share from 65% to 7% in 1,536 updates, far past backprop's 43–53%. As written it would test "over-saturation hurts", not "decisive gates help". Use a cost that stops at a limit, at three strengths, as the Gemini report's first remedy describes.
4. **Note that the learned-feedback arm starts from random feedback.** At the dry dose it is worse than the floor (79 against 86 on single operations). Either add a start from the weight signs, which the rulings allow, or read its early stages separately.
5. **Add an every-byte birth with accuracy at 40 letters as the endpoint** for the gate push and the learned feedback. That is where the effect is largest, and a birth costs under one CPU-hour.

### The Gemini report

- **It supports novelty:** nobody appears to have measured gate values or composition under approximate gradients. Two of its findings match ours independently: alignment does not predict learning, and approximate rules lose on longer memory.
- **It adds one rival explanation worth a cheap measurement:** worse generalization from sharper minima, controlled by step length along the true gradient and not by its direction.
- **Its remedy ranking fits the evidence:** a bounded saturation cost first, then the refine gate, then learned feedback.
- **Cautions:** its quotes went through a summarising extractor, and it could not read the disputed figure that our two checks did read. Spot-check anything before citing it.

*[Added 14 November: the coordinator's addendum 2 to the E31 brief took points 1 to 4. The rounding difference was accepted, padding was excluded, the push was withdrawn and replaced by one that stops at a limit at three strengths (P2a, P2b, P2c), and a start from the weight signs was added as arm Ks. Point 5 became E32, audited in notes 9 to 11.]*

## 10. Note 9: audit of the E32 brief, revision 1

**When:** 14 November, 04:10 UTC.

**What prompted it:** Asked to audit `OPUS_TASK_E32_EVERY_BYTE_BIRTH_SWITCHES_20261114.md` at `aaa070e4` before any worker started, each finding with the command that shows it, on five named points.

**Disposition:** Hold and revise before any worker starts.

I audited the brief at `aaa070e4`; it is unchanged at the tip `d4ceb739`. The fixes are about a page of text and need no new experiment. Every command below runs from the repository root after:

```
D=origin/codex/minifly-consolidated-20261023; A=origin/claude/sleepy-knuth-q96dd1; B=origin/claude/charming-clarke-u9g850; M=MOD97_LOCAL_RULE_20261106
```

### 1. Do the arms differ from the floor in the backward pass only?

H does; K and P2 do not, and the push's mask is not pinned.

- **H is clean.** Only the error sent back changes; the weight and bias gradients are the floor's.
  `git show $A:$M/arch31.py | sed -n '46,53p'`
- **K adds state.** It carries a 768×256 matrix, a random draw and an update after each step. The draw uses its own generator, so the data order is safe.
  `git show $A:$M/exp31_switch.py | sed -n '56,58p'`
- **P2 is an added loss term.** The brief calls it "one change to the backward pass", which it is not.
- **The mask is ambiguous.** The brief says "the task's mask of real positions". E28's mask is `y != BOS`; E31's ruled mask is `x[:, 1:] != BOS`. They differ at position 0 in every item and at one more position in every padded item. This is the same kind of gap that split workers A and B on arm P.
  `git show $D:$M/exp28_everybyte.py | sed -n '13p'; git show $B:$M/exp31_switch.py | sed -n '103p'`

**Fix:** write the mask as code. Add two checks for every arm: the first batch's task loss equals the floor's bit for bit, and `data_digest` equals E28's at every row.

### 2. Does setting the push's strength on the first batch make sense?

No, for three reasons.

- **Every gate is inside the band at the start.** Five untrained networks drawn from the same default distributions give U = 100.00%, with z between 0.19 and 0.80. The term therefore acts on every gate, in whatever direction the initial noise leans.
- **The initial lean is letter-specific.** Only 3 to 10 of 256 units lean the same way on at least 90% of letter positions. My reasoning (not a measurement) is that the push would drive the rest to hold on some letters and overwrite on others, which works against holding across letters.

  *[Added 14 November: this holds for an untrained network. At update 4,096, where revision 2 moved the switch-on, 175 to 178 of 256 units lean one way; see note 10.]*

  `python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/untrained_gates.py`
- **What the push is meant to stop happens later.** The floor's U on add-89 is 65% at row 4,096 and climbs to 92% by the end, in both seeds.
  `git show $D:$M/exp45/e30/E30_UNSEEN.json | python3 -c "import json,sys; [print(r['arm'],r['seed'],r['row'],r['U_add89_k12'],r['acc12'],r['acc40']) for r in json.load(sys.stdin)['rows']]"`
- **The ratio describes update 0 only.** The loss starts near ln 257 = 5.55 and averages 2.44 over the first 512 updates, then 2.07 at the end. The brief records λ and both norms once.
  `git show $D:$M/exp45/e28/e28_dev_S-all_73129.json | python3 -c "import json,sys; print([round(b['train_loss'],2) for b in json.load(sys.stdin)['role_curve']])"`

**Fix:** switch the push on after update 4,096 and set λ on that batch. Each P2 birth's row-4096 file must then equal E28's `e28_dev_S-all_<seed>_row4096.pt`, a free check on all three seeds. Record both norms at every row, and write the three λ values into the predictions file before any full birth.

### 3. Is accuracy at 40 letters, read with `e30_gates.py`, the right endpoint?

The quoted floor and ceiling are correct (same command as the E30 record above). For seed 116089 my reader gives floor 71.0 at 12 letters and 2.0 at 40, and ceiling 93.7 and 80.3. The endpoint has four problems.

- **40 letters alone hides partial gains.** Births train on 0 to 12 letters. At 20 letters the receipts already separate the arms, so an arm that moves the cliff from under 20 to 30 would read as "none does".

  | Birth | At 12 letters | At 20 letters | At 40 letters |
  |---|---|---|---|
  | S-all (floor) | 71–80 | 16.7 / 24.7 / 41.3 | 2.0–2.3 |
  | S-bal | 87–91 | 65.7 / 83.0 / 60.3 | 2.0–16.3 |
  | B-all (ceiling) | 94–96 | 94.0 / 93.3 / 91.7 | 80–85 |

  `for a in S-all B-all S-bal; do for s in 73129 93683 116089; do git show $D:$M/exp45/e28/e28_dev_${a}_$s.json | python3 -c "import json,sys; d=json.load(sys.stdin); x=d['curve'][-1]['sealed']['add89']; print('$a $s kmax',d['kmax'],{k: round(100*v['exact']/v['n'],1) for k,v in x.items()})"; done; done`
- **A single final row is noisy.** The floor's own accuracy at 12 letters goes 66.3, 37.0, 75.3 over its last three rows (seed 73129, in the E30 record). The 15-point no-read margin is under half that swing.
- **`e30_gates.py` unchanged cannot read E32 files.** It looks for `e28_dev_*.json`, labels rows "E28", and loads weights strictly, so a `Bhh` in the file would fail.
  `git show $D:$M/e30_gates.py | sed -n '93p;99p;114p'`
- **U's band is too wide for a 40-letter endpoint.** The table below is post hoc and unregistered, from my numpy reader on the 18 saved births.

  | Birth | Accuracy at 40 | Letter gates above 0.98 | Units keeping half their state across 12 / 40 letters |
  |---|---|---|---|
  | S-all | 2.0–2.3 | 0.01–0.03% | 0–1 / 0 |
  | S-bal | 2.0–16.3 | 0.1–1.1% | 8–14 / 0–1 |
  | B-all | 80–85 | 20–25% | 67–85 / 46–64 |
  | B-bal | 87–95 | 33–35% | 98–103 / 81–90 |
  | S-ans | 96–99.7 | 35–41% | 109–130 / 85–101 |
  | B-ans | 92–99 | 37–46% | 104–129 / 92–113 |

  `python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/retention.py`

  - A gate at 0.9 keeps 0.9^40 = 1.5% of its state, so the bounded push, which stops at 0.9, cannot by itself make a unit that holds 40 letters.
  - S-bal is the example on file: 11–14% of its letter gates sit between 0.9 and 0.98, and nothing is held at 40.
  - The floor has no keeping unit even at 12 letters, where it scores 71–80%. It carries the operand some other way, and that way is what fails past the trained range.

**Fix:**
- Keep 40 letters as primary and register 20 letters from the birth's own receipt as secondary; it costs nothing.
- Read both on the mean of the last three rows.
- Measure with a small new script that imports `U_of` and `H_of` and first reproduces the E30 record's floor and ceiling exactly.
- Record the count of keeping units at every row.

### 4. Is taking `arch31.py` from worker A's branch sound?

Not as written.

- **`arch31.py` holds none of the code in question.** The push, its strength setting and the feedback update are all in `exp31_switch.py`, inside the life runner.
  `for f in arch31.py exp31_switch.py; do echo "$f $(git show $A:$M/$f | grep -c 'relu\|\*\* 2\|self.lam\|DECAY\|Bhh = ')"; done` prints 0 and 16.
- **Worker A has no P2 code yet; worker B has.**
  `git show $A:$M/exp31_switch.py | grep -c P2_RATIO; git show $B:$M/exp31_switch.py | grep -c P2_RATIO`
- **So the stated reason fails.** The E32 worker would write the push, the strength and the feedback update anew, once, for a different loop (every-byte loss, random start).

**Fix:** write the term for the birth loop and check it on one batch against an independently written form, as E28's check 4 did. Then have a second worker reproduce one dry birth's hash.

### 5. Do the predictions and readings leave outcomes unread?

Yes.

- **P32.4 misses.** There is no reading. It should say the push did not take and the P2 arms say nothing about gates.
- **H and K both do, or K does and H does not.** Neither has a reading.
- **"Does" is too low a mark for the claims.** It means 32% at 40 letters against a ceiling of 80–85, yet the readings say "would do it" and "are enough". Add a second mark, for example within 15 points of the same seed's ceiling.
- **Gains of 15 to 30 points** are neither "does" nor "not read".
- **Harm at 12 letters,** or a trade of 12 for 40, is unread.
- **Reading 1 contradicts the update rule.** The difference between `Bhh` and the forward matrix shrinks by exactly (1 − 0.00025) per update: 36% left at row 4,096, 13% at 8,192, 0.2% at the end. K becomes H by construction, so "did not get there in a birth's length" cannot be the reading; "exact credit was needed early" can.
  `python3 -c "print([round((1-0.00025)**n,4) for n in (4096,8192,12288,24576)])"`
- **Reading 3 over-reads.** "Undecided gates go with the loss but do not cause it" does not follow, because P2 cannot make keeping gates (table in point 3).
- **"None does" over-reads.** It rests on one bound for one map. Either add E31's arm G as a second bound or narrow the sentence.
- **"Every P2 arm" in P32.4** lumps three doses into one prediction.

### Smaller points

- **Weights.** Fifteen births with six rows each is about 405 MB if every row is committed; the brief should say which rows.
- **E31 dependency.** Worker A has not pushed stage 1b, so E32 cannot start by its own rule. E31's addendum 3 releases Ks to worker B, which conflicts with your instruction not to run Ks at full length; B should be told if it has not been.

### Not checked

- I ran no learner; torch is absent here.
- The untrained-gate figures use the same distributions as the real network but not the same draws.
- I did not measure gradient norms at the start, so I cannot say which way the nominal ratios drift.
- The seed 116089 figures are from my reader, not `e30_gates.py`.

*[Added 14 November: both scripts were pushed later the same night, in commit `aeb47f07`.]*

## 11. Note 10: audit of the E32 brief, revision 2

**When:** 14 November, 04:26 UTC.

**What prompted it:** Asked to audit revision 2 at `00854280` on six named points and to push the two scripts.

**Disposition:** One more small revision before any worker starts.

Revision 2 (`00854280`) fixes most of what revision 1 got wrong. What remains is a new fact from the saved rows that changes the framing, one mark that cannot be computed, three defects in the push's code block, and several gaps in the readings and checks.

The scripts are pushed: branch `claude/vibrant-cerf-sje726`, commit `aeb47f07`, folder `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/`. They are `np_gru.py` (the reader), `untrained_gates.py`, `retention.py` and a new `e32_r2_checks.py`. Each runs with `python3 -I` and numpy, reading weights from the default branch.

```
D=origin/codex/minifly-consolidated-20261023; A=origin/claude/sleepy-knuth-q96dd1; M=MOD97_LOCAL_RULE_20261106; R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109
```

### 1. Are the revision 1 findings fixed?

| Finding | Status | What remains |
|---|---|---|
| K and the push are not "a change to the backward pass" | fixed | none |
| Mask not pinned | pinned | one stray position per padded item (point 2) |
| Strength set on an untrained network | fixed | "where the climb starts" is not known (point 3) |
| 40 letters alone | fixed | the 20-letter panel call is not pinned (point 4) |
| One final row is noisy | fixed for arms and floor | the ceiling's rows are not on file (point 5) |
| `e30_gates.py` cannot read E32 files | fixed | none |
| Limit of 0.9 | raised to 0.99 | the push fades long before 0.99 (point 3) |
| `arch31.py` holds none of the code | fixed | none |
| Outcomes without readings | mostly fixed | five gaps (point 5) |
| One bound for one map | G added | its difference from H is misnamed (point 5) |
| 405 MB of rows | fixed | none |
| Dependence on E31 | fixed | none |

The table's rerun counts reproduce: 0 and 16, 0 and 7.

### 2. The push's code block

The formula for `z` matches the real forward pass (`sigmoid(i_z + h_z)`, rows 256–511), and detaching both inputs does confine the gradient to those rows. Three things are wrong or open.

- **`hp` is not defined, and `arch26.forward` does not return it.** It returns `dict(z1=z1, h=hs)`, the states after each step. A worker who passes those as `hp` computes every gate one step late, and none of the listed checks would catch it, because the hand form would use the same input.
  `git show $D:$M/arch26.py | sed -n '50,75p'`
  **Fix:** define `hp[:, t]` as the state step t read (zeros at t = 0), and add a check that the term's `z` equals the forward pass's own `z` on one batch.
- **`lin` takes three arguments; the block passes two.** Pin the third.
  `git show $D:$M/exp3_state.py | sed -n '107,108p'`
- **The mask pushes one position per padded item that has no target.** There the input is the item's end byte and the target is padding: about 6% of masked positions, on an input the task never reads.
  **Fix:** `msk = (x != X.BOS) & (y != X.BOS)`.

```
python3 -I -c "
BOS,EOS=257,256
rows=[[112,49,50,43,65,66,51,52,61,52,54,EOS],[112,49,50,43,51,52,61,52,54,EOS]]
L=max(map(len,rows))+1; data=[[BOS]*L for _ in rows]
for i,r in enumerate(rows): data[i][1:len(r)+1]=r
for d in data:
    x,y=d[:-1],d[1:]; print([int(v!=BOS) for v in x],[int(v!=BOS) for v in y])"
```

### 3. Can the push make 40-letter holders, and is the risk fatal?

**The floor already has keeping units at update 4,096 and then loses them.** This is new, post hoc and unregistered, from the rows on file by the brief's own definition.

| Floor row | Keeping units at 12 letters | at 40 letters | Letter gates above 0.983 | Accuracy at 12 |
|---|---|---|---|---|
| 4,096 | 50 / 59 / 62 | 22 / 0 / 48 | 10% / 11% / 19% | 16 / 26 / 17 |
| 8,192 | 22 / 20 / 30 | 0 / 0 / 0 | 1.2% / 1.0% / 0.5% | 57 / 42 / 48 |
| 12,288 | 5 / 5 / 2 | 0 | under 0.1% | 61 / 54 / 51 |
| 24,576 | 0 / 0 / 1 | 0 | 0 | 75 / 80 / 71 |

`python3 -I $R/e32_r2_checks.py B`

- **The question changes.** It is which change lets the sweep keep these units, not which change gives them. The Why and the phrase "made by a local push" should say so.
- **P32.5 is a retention test.** Its bar of 20 is below the starting count.
- **The push is weakest where protection is needed.** Its force on a gate is 2·(0.49 − |z − 0.5|)·z·(1 − z). Relative to its peak that is 1/15 at 0.9, 1/50 at 0.944 (what 12 letters need), 1/155 at 0.966 (20 letters) and 1/1047 at 0.983 (40 letters).
  `python3 -c "f=lambda z:2*(0.49-abs(z-0.5))*z*(1-z); print([round(f(0.5)/f(z)) for z in (0.9,0.944,0.966,0.983)])"`
- **My expectation, not a measurement:** it may hold 12-letter keepers and is unlikely to hold 40-letter ones, which agrees with P32.7 at 15%. I cannot tell from files whether a force that small beats what erodes them.
- **Two ways forward.** Either keep the term and record, at every row, the share of letter gates above 0.944 and 0.983 so a stall is visible; or use a term whose force does not fade before the limit, such as a hinge on the pre-activation.

**The stated risk does not make the arm unreadable.** At update 4,096, 175–178 of 256 units have their gate on one side of one half at 90% or more of letter positions; 78–81 are split. If every gate completed its present lean, 98–127 units would keep across 12 letters and 78–87 across 40.

`python3 -I $R/e32_r2_checks.py C`

- **Some "keepers" would carry nothing.** 24–42 of the units that lean towards keeping at letters also lean that way before the gap, so they would be pushed to keep everywhere and never load the operand. In the saved births nearly every keeper also writes before the gap (67 of 67, 85 of 85, 79 of 82 in the ceiling).
  **Fix:** report "keeps and writes" beside "keeps".
- **"The push starts where that climb starts" is not known.** No row before 4,096 exists, and seed 116089 is at 51% there, not 65%.

### 4. Is a keeping unit computable, and does it agree with my counts?

Yes on both, with three pins.

- **Agreement.** The brief's rule and my median form agree in 51 of 54 cells and differ by one in three. The ceiling has 67 / 85 / 82 at 12 letters, as quoted. The floor has 0 / 0 / 1, so "no unit" is wrong for seed 116089.
  `python3 -I $R/e32_r2_checks.py A`
- **Pin the positions:** the steps whose input byte is one of the k letters.
- **Pin the panel calls.** E30's call is `ks=(12, 40)`, and the letters at 40 depend on that tuple. Adding 20 to the same call changes the 40-letter panel and breaks the "reproduce E30 exactly" step. Give 20 a call of its own, `ks=(20,)`, and take the floor's 20-letter figure from the same call, not from the receipts.
- **Pin "ends with":** the final row, or the mean of the last three.
- `U_of` and `H_of` do not return gate values; `e29_gates.gates(m, x)` does.

### 5. Marks and readings

- **"Closes" cannot be computed as defined.** The ceiling's row files were never committed, only its three final files, so its last-three-row mean at 40 letters does not exist.
  `git ls-tree -r --name-only $D $M/exp45/e28/ | grep 'B-all.*\.pt$'`
  **Fix:** use the ceiling's final row (its 12- and 20-letter rows swing by about 5 points in the receipts), or rerun the three ceiling births with rows saved.
- **"Partial" leaves a hole.** Gains of 35, 20 and 0 points are neither "moves" nor "partial", so the bounds table reads them as "does not" and concludes the recurrent map "is not where this loss arises".
  **Fix:** partial is at least 15 points in two of three seeds and not "moves"; a partial result gets no "not where" reading.
- **The bounds table reads 40 letters only.** H moving at 20 but not 40 would get the same "not where" reading. Add the 20-letter row, as the push table has.
- **G and H differ in two ways, and the brief names one.** G adds the input map's gate rows and lacks the recurrent map's candidate rows (512–767). If H moves and G does not, the candidate rows are the difference.
  `git show $A:$M/arch31.py | sed -n '7,9p'`
- **The push table's last row claims more than the arm shows.** It tests keepers at 12 letters and concludes about 20 and 40. S-bal is the case on file: 8–14 keepers at 12, 0–7 at 20, 0–1 at 40.
  **Fix:** count keepers at 20 as well, read each distance against the count at that distance, and use the keeps-and-writes count.
- **Two seeds of three with 20 keepers** has no row; P32.5 says all three, the accuracy marks say two.
- **K's window.** The floor loses its keepers between 4,096 and 12,288, while `Bhh` still differs from the forward matrix by 36% down to 5% of its start. The reading "in the first few thousand updates" should name that window.

  *[Added 14 November: my wording here was too narrow; `Bhh` is also off before update 4,096. See note 11, point 6.]*

### 6. Are checks 5 and 6 achievable?

**Check 5's row-4,096 equality is achievable.** E28's receipts carry `tensors_sha256` for every row, and nothing in a P3 birth touches updates 1 to 4,096.

`git show $D:$M/exp45/e28/e28_dev_S-all_73129.json | python3 -c "import json,sys; print([(r['step'], r['tensors_sha256'][:12]) for r in json.load(sys.stdin)['curve']])"`

- **The `lam = 0` check is empty as written.** A dry birth of 2,048 updates never reaches the switch-on.
  **Fix:** run `lam = 0` to 8,192 updates for seed 73129 and require E28's row-8,192 hash.

**Check 6 needs four changes.**

- **Hashes will not settle P3.** `lam` comes from two gradient norms, so two independent implementations will differ in the last bits and the births diverge. For P3 the second worker's dry births compare nothing.
- **Compare P3 without training instead.** Both workers compute `lam` and both norms for all nine seed-and-strength pairs from E28's row-4,096 files and the batch of update 4,097. All nine go into the predictions file; the brief says "the three values", but `lam` is per seed.
- **"Within 1e-6" is undefined** in checks 3, 5 and 6. E28's check 4 measured 3e-7 to 5e-7 between two forms and settled on 1e-5 of the tensor's largest entry; use that.
- **Smaller points.** G's form is pinned for the recurrent map only; pin it for the input map too, since G differed by rounding in E31. "Three additions" should be four (G, H, K, P3). Dry births of 2,048 updates are enough for G, H and K.

### Not checked

- I ran no learner; torch is absent here.
- I did not measure gradient norms, so the force argument in point 3 is arithmetic plus the floor's rows.
- The row-4,096 projection is static and ignores how the state changes as gates harden.
- All counts are from my numpy reader, not `e30_gates.py`.

## 12. Note 11: audit of the E32 brief, revision 3

**When:** 14 November, 05:17 UTC.

**What prompted it:** Asked to audit revision 3 at `bc6a0eaf` on seven named points and to say plainly whether it can go to workers.

**Disposition:** Not yet; six text changes away.

Revision 3 (`bc6a0eaf`) fixes all sixteen findings in its table. It cannot go to workers as it stands, but what remains is six text changes and no new experiment. The script behind this audit is pushed: `claude/vibrant-cerf-sje726`, commit `f5e876d0`, `e32_r3_checks.py`.

```
D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106; R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109
```

### 1. Revision 2 findings

All sixteen rows are fixed. Five leave a residue, covered below: the gate comparison (point 2), the strength setting (3), the ceiling command (5), the keeper bar (4) and K's window (6).

### 2. The push's code block

The block is right; one check built on it may fail spuriously.

- **`hp`, the mask, `C` and the 12% claim are correct.**
  - The mask keeps 947 of the 1,008 non-padding steps on the batch of update 4,097 (seed 73129). The 61 dropped are end-byte steps with no target.
    `python3 -I $R/e32_r3_checks.py D`
  - `C` is ln 99 = 4.59512. The pull is 38.5%, 27.2% and 11.7% of its peak at gates of 0.944, 0.966 and 0.983.
    `python3 -c "import math; C=math.log(99); print(round(C,5), [round((C-math.log(z/(1-z)))/C,3) for z in (0.944,0.966,0.983)])"`
- **"Bit for bit" in check 6 may fail for correct code.** The forward pass makes one recurrent product per step; the block makes one over all steps. I cannot test here whether those give the same last bits, and E31's term, which used the stacked form, never compared its gate with the forward pass's.
  `git show $D:$M/arch26.py | sed -n '58,62p'`
  **Fix:** say "agrees" at the brief's tolerance and report whether it is identical. A one-step slip gives differences near 1, so the check keeps its purpose.

### 3. Can the hinge hold the keepers, and are the strengths sensible?

It can, by a first-order estimate; no strength is plainly too weak or too strong; the way the strength is set is the problem.

**What the floor does to its keepers** (mean pre-activation at letter steps, seeds 73129 / 93683 / 116089):

| Row | Pre-activation | Gate it gives |
|---|---|---|
| 4,096 | 4.08 / 4.02 / 4.97 | 0.983 / 0.982 / 0.993 |
| 8,192 | 2.74 / 2.65 / 2.83 | 0.939 / 0.934 / 0.944 |
| 12,288 | 2.02 / 1.93 / 2.00 | 0.883 / 0.874 / 0.881 |
| 24,576 | 1.43 / 1.16 / 1.29 | 0.806 / 0.762 / 0.784 |

- The keepers start at or just past the hinge's limit of 4.595, so its pull on them starts small and grows as they slip.
- The gate rows' weight norm rises from 28.7 to 36–39 over the same rows, so the loss is not weight decay.

**What the hinge adds**, per 1,000 updates, to those keepers' pre-activation at the row-4,096 weights:

| | 73129 | 93683 | 116089 |
|---|---|---|---|
| Floor takes away (observed, rows 4,096 to 8,192) | −0.32 | −0.32 | −0.50 |
| Hinge adds at 0.25 | +8.6 | +15.5 | +24.2 |
| at 0.05 | +1.8 | +3.5 | +6.0 |
| at 0.01 | +0.4 | +0.7 | +1.2 |

`python3 -I $R/e32_r3_checks.py GH`

- **0.01 is marginal:** 1.3 to 2.4 times what the floor takes.
- **0.25 is a freeze:** 27 to 48 times, and by the same estimate the median gate reaches the limit in 55 to 193 updates. P4a would fix the gate pattern as it stood just after update 4,096, when accuracy at 12 letters was 16–26%. That is a legitimate top dose, but it carries the highest risk of harm.
- **Trust this to an order of magnitude only.** The model assumes AdamW at its stationary step over 128 batches at frozen weights. Applied to the floor alone it predicts +2 to +10 where the floor shows −0.3 to −0.5, so I use it for the hinge's side only.

**The strength is set on one batch, and one batch is noisy.** Over the 128 batches after update 4,096 the task-gradient norm on the gate rows ranges 3 to 5 times (for example 0.035 to 0.189 in seed 73129), while the hinge's norm moves by about 6%. Batch 4,097 happens to give effective ratios of 0.12, 0.33 and 0.40 of the mean. One label covers a 3.3-fold spread across seeds, against 5-fold steps between doses.

**Fix:** set `lam` on the mean task norm over batches 4,097 to 4,224 at the frozen row-4,096 weights, which still needs no training. Or fix three values for all seeds; 0.25 of the mean gives about 0.021 / 0.017 / 0.017.

### 4. "Keeps and writes" and "retains keepers"

Both are computable and separate the saved births cleanly. Counts at the floor's row 4,096, keeps-and-writes with keepers in brackets:

| Letters | 73129 | 93683 | 116089 | Bar |
|---|---|---|---|---|
| 12 | 40 (50) | 46 (59) | 46 (62) | 25 / 30 / 31 |
| 20 | 33 (39) | 31 (34) | 43 (55) | same |
| 40 | 18 (22) | 0 (0) | 36 (48) | same |

`python3 -I $R/e32_r3_checks.py I`

- **The mark separates what it should.** The ceiling's final files pass at every distance (67 / 85 / 79 at 12, 46 / 56 / 63 at 40). The floor's end (0–1) and S-bal (8–14) fail.
- **At 40 letters "retains" is the wrong word.** The bar is above what exists at update 4,096 in two seeds (18 and 0 against 25 and 30). A push that held every keeper would still be marked "no", and the reading would say it "did not hold" them.
  **Fix:** call the mark "has keepers" and reword the "no" row to "does not have keepers at this distance".
- **The bar is half the keeper count but is applied to the keeps-and-writes count.** At 12 letters it asks for 62–67% of what exists; at 20 letters, 72–97%. That is fine if meant; say so.
- **"Reproduce the floor table exactly" may miss by one.** My reader multiplies in double precision, and a unit on the 0.5 line can fall either way. Say "within one unit" and keep the bars as the stated numbers.

### 5. The ceiling rerun

It is achievable with E28's runner unchanged, but the brief does not say so.

- The runner has `--out`, refuses to overwrite, and sets one thread.
  `git show $D:$M/exp28_everybyte.py | sed -n '7p;190p;273,276p'`
- E28's receipts record one thread, Python 3.13.18 and torch 2.15.0+cpu, and carry a hash for all six rows, not only the last.
  `git show $D:$M/exp45/e28/e28_dev_B-all_73129.json | python3 -c "import json,sys; print([(r['step'], r['tensors_sha256'][:10]) for r in json.load(sys.stdin)['curve']])"`
- "One new runner" and "receipts say `experiment="E32"`" would lead a worker to write a backprop path into the new runner.

**Fix:** name the command, `python3 exp28_everybyte.py dev B-all <seed> --out exp45/e32/ceiling`; say these three receipts stay in E28's format; compare all six row hashes, so a mismatch shows after nine minutes.

### 6. Marks and readings

The nine H-by-K cells, both G cases, partial, harm, trade and "closes" are all covered. Four things remain.

- **The push table cannot be read at 12 letters.** "Moves" there asks for 89.6 / 97.2 / 104.0, which is impossible in one seed and above the ceiling in another. An arm 14 points above the floor and near the ceiling would be read as "keepers do not give accuracy at this distance".
  **Fix:** read the push table at 20 and 40 only; 12 letters is covered by harm.
- **Accuracy without keepers, and keepers without accuracy, have no sentence for G, H and K.** For the push, "no keepers, accuracy moves" is read as "says nothing", though it would count against the keeper account. One sentence covers both.
- **K's window is over-stated.** `Bhh` is also off before update 4,096, by 100% falling to 36%. The arm shows exact credit was needed before `Bhh` converged; K's own keeper counts by row say when. My wording in the last audit was too narrow.
- **"Keepers that write, held to the end, do not give accuracy"** should say "these keepers, held from update 4,096".

### 7. Checks 6 and 7

Both are achievable.

- **`lam = 0` to 8,192:** the row-8,192 hash is on file. Say the term must be computed and added with `lam = 0.0`, not skipped.
- **Nine values without training:** I did it without the harness. The batch stream rebuilt in pure Python gives the receipt's digest at 4,096 batches for all three seeds. As a third reference:

  | Seed | Task norm (sweep) | Hinge norm | `lam` at 0.25 / 0.05 / 0.01 |
  |---|---|---|---|
  | 73129 | 0.036740 | 0.890970 | 0.01031 / 0.00206 / 0.00041 |
  | 93683 | 0.077113 | 0.836491 | 0.02305 / 0.00461 / 0.00092 |
  | 116089 | 0.093320 | 0.849241 | 0.02747 / 0.00549 / 0.00110 |

  `python3 -I $R/e32_r3_checks.py DEF`

  My backward pass matches finite differences to 2e-5 on the exact path; the sweep's path is not independently checked. Expect the workers to match me to about 1e-3, not 1e-5.
- **Two pins are missing.**
  - The task norm must be "as the sweep computes it"; with exact weights it is 0.082 / 0.094 / 0.115.
  - It must be one norm over the two matrices' rows and both biases together.
- **Each P4 receipt's `lam`** should equal the registered value.

### Verdict

Not as it stands. The smallest set of changes:

1. Check 6: "agrees" in place of "bit for bit" for the gate comparison.
2. Strength: set on the mean task norm over batches 4,097 to 4,224 (or three fixed values); "as the sweep computes it"; one joint norm; the receipt's `lam` equals the registered one.
3. Ceiling: name the E28 runner command with `--out`, and compare all six row hashes.
4. Push table read at 20 and 40 letters only.
5. "Retains" becomes "has keepers", with the "no" row reworded; state what the bar is half of.
6. Readings: K's window, "these keepers", one sentence for accuracy without keepers and the reverse; the reader reproduces the floor table within one unit.

If these are made as written, I do not need to see the brief again before the worker messages go out.

### Not checked

- I ran no learner; torch is absent here.
- The gate-comparison concern in point 2 is untested.
- The strength estimate in point 3 is a model that fails its own check on the floor.
- All counts are from my numpy reader.

## 13. Scripts behind the figures

All are in this folder (`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/`). None trains. Each reads receipts and weights with `git show` from a fetched branch, so run them from inside a clone with `python3 -I <script>`; the numpy ones need numpy and nothing else.

| Script | What it backs | Notes that use it |
|---|---|---|
| `compare_pt_tensors.py` | a torch-free reader for saved weight files | all |
| `np_gru.py` | the panels and the GRU's forward pass in numpy; reproduces every receipt count it was checked against | 3 to 11 |
| `e26_recompute.py` | E26 births and lives and the crossover run, from receipts | 1, 2 |
| `error_kinds.py` | what kind of mistake each life makes on composition items | 6 |
| `second_step.py` | second-step failures given a correct first step | 6 |
| `gate_weights.py` | gate values and gate-weight scales by update | 6 |
| `e30_check.py` | E30's undecided share and holding loss on all twelve E28 births | 8 |
| `untrained_gates.py` | gate values of untrained networks drawn from the default distributions | 9 |
| `retention.py` | how much state each unit keeps across the letters, on the eighteen saved births | 9 |
| `e32_r2_checks.py` | keeping units by the brief's definition, the floor row by row, the gate state at update 4,096 | 10 |
| `e32_r3_checks.py` | the batch stream rebuilt and digest-checked, a numpy backward pass for the gate rows, the nine push strengths, the keepers followed through the rows, the marks | 11 |

Two defaults to know:

- `e26_recompute.py` and `e30_check.py` read worker branches by default (`origin/claude/adoring-cori-b07aog` and `origin/claude/charming-cori-ljzwz1`). Fetch those by explicit refspec, because a tag shares one worker branch's name: `git fetch origin "+refs/heads/<branch>:refs/remotes/origin/<branch>"`.
- The others read `origin/codex/minifly-consolidated-20261023`.

Not pushed, because they read scratch copies of the deliveries' own files: the five step scripts of the E29 audit and the recomputation of the span roster. Their findings are in notes 3 and 4.

## 14. Reference figures the notes mention only in passing

**E30's two measures on all twelve E28 births** (`e30_check.py`; note 8 says they were extended to the seeds E30 could not reach). U is the share of update-gate values between 0.1 and 0.9 on the plus-minus composition panel; accuracy is on mod-89 addition.

| Birth | Seed | U | Accuracy at 12 letters | at 40 letters |
|---|---|---|---|---|
| Sweep, every byte | 73129 | 68.37 | 75.3 | 2.0 |
| | 93683 | 72.50 | 79.7 | 2.3 |
| | 116089 | 78.09 | 71.0 | 2.0 |
| Backprop, every byte | 73129 | 50.25 | 93.7 | 84.7 |
| | 93683 | 56.42 | 96.3 | 81.7 |
| | 116089 | 58.48 | 93.7 | 80.3 |
| Sweep, balanced | 73129 | 88.03 | 91.3 | 6.0 |
| | 93683 | 77.36 | 90.7 | 16.3 |
| | 116089 | 87.66 | 87.0 | 2.0 |
| Backprop, balanced | 73129 | 48.35 | 94.7 | 87.0 |
| | 93683 | 55.39 | 98.7 | 95.0 |
| | 116089 | 53.34 | 95.0 | 92.0 |

**The floor's means over its last three rows, and what "moves" (30 points above) asks for** (`e32_r3_checks.py I`; note 11 quotes the 12-letter column only). The 20-letter figures use a panel call of their own, as revision 3 of the E32 brief pins it.

| Seed | Accuracy at 12 / 20 / 40 letters | "Moves" asks for |
|---|---|---|
| 73129 | 59.6 / 8.1 / 1.0 | 89.6 / 38.1 / 31.0 |
| 93683 | 67.2 / 16.6 / 1.6 | 97.2 / 46.6 / 31.6 |
| 116089 | 74.0 / 34.1 / 1.3 | 104.0 / 64.1 / 31.3 |

## 15. What this memo does not contain

- The outside literature report of 14 November. It is Mr. Xu's; note 8 gives my reading of it.
- The briefs and reviews themselves, which are on the default branch.
- Anything trained by this session. It trained nothing.
