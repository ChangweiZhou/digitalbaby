# Audit of the tissue (E63 and the work of 19 November) at the level of its programs

**For** Mr. Xu and Mr. Zhou; for the reviewer and the coordinator. **Written by** the outside session that wrote the advice memo and the audit of 18 November beside this file, at Mr. Xu's request of 19 November: the design based on cells did not work; audit it at the level of the actual implementation and say what I think.

**Status:** proposed. It decides nothing and authorises no run. No learner was trained. The figures marked *computed here* come from `audit_tissue.py` beside this file (output: `audit_tissue_output.txt`), which reads the programs with `git show`, builds the reviewer's `tissue.c` in a temporary folder and runs it. Every reading is mine and is marked. Mr. Xu's statements are quoted from the reviewer's record `WHAT_A_NEURON_HAS_20261118.md` and are not extended by me.

## 1. The answer in brief

1. **What was run under the name of cells is three different things.** (a) The tissue of the definition: 2,048 cells on a square, 256 input lines on one edge, 256 output lines on the opposite edge (E63, stage A). (b) The same tissue with its rules replaced during the night ("the tissue as run"), read from outside by a separate program, "the output end". (c) From 14:30 UTC on 19 November, small programs with no tissue. I read all three as programs, built the tissue here, and reproduce the reviewer's own reference figures exactly (3.24 connections a cell, 1,240 made, 10,474 withdrawn).
2. **In no tissue did an error ever change a connection.** In both tissue programs the feedback is switched off in the text of the program: the yes and no lines are set to zero at every step, and the feedback cells give zero. The output end reads a recording of the tissue's life after that life is over and "does not act back". In the tissue as run no strength changes by experience at all. The loop that Mr. Xu states as the aim, that the system is told when it is wrong and corrects itself, was therefore not run. **The result does not show that cells fail at it. It shows what a tissue without it does.**
3. **Step 1 took the whole effort for reasons that can be computed from the definition's numbers.** The input and the output lie 9 to 11 connections apart; a connection passes one eighth, so the largest output falls 8-fold a connection (*computed here*: 0.126, 0.016, 0.0025, 0.0004); outputs below 0.01 are recorded as zero and a byte is emitted only above 1/2. Step 1 could not be met unless something multiplied the signal at every connection, and the definition holds nothing that does. Nor could learning raise the measure for the present byte: with no change at all, the present byte already explains 1.000 of every cell's output. What could be gained was amplitude, reach and slower cells, and that is what the night's search found: a tissue that carries and does not learn.
4. **What that tissue hands to a reader is the bytes side by side.** A balanced probe of all 100 pairs of digits (*computed here*): of the variance of the cells' outputs, 1% to 3% belongs to two digits jointly; the rest is the present digit, the digit before it, and 29% to 61% that is neither. A reader that adds such inputs can learn the form of an item and not a sum.
5. **The score could not have shown the task, and the lives were too short for it.** On the stream of step 1, a guesser that knows the form of an item and no arithmetic gets 25.9% of next bytes right; one that also does the arithmetic gets 36.6% (*computed here*). Every figure of the output end on a tissue that night (5% to 20%) lies below the first; on the exact last three bytes with no tissue the best rule reached 26.5%, the level of the form. A life of 16,384 bytes shows 6.5% of one pass over the taught pairs, and the longest life one pass; the record's reference network needed about 26 passes.
6. **In the small systems that answered sums never shown, the program does the adding.** In all three the sum of the two digits is a line of Python, and a "cell" is an index made from that sum. What is learned from yes or no is a table from 19 totals to answer bytes.
7. **What I think** (section 6): the cells were given everything except the one thing asked of them. I would build the loop first, at the smallest size, with the error formed inside the cells; give "prediction" its content (a cell predicts what arrives from what other cells send, which makes the mismatch a learning signal that needs no reader outside); and drop the geometry that made transport the problem.

## 2. Scope, independence, and what was executed

**REVIEW_SCOPE.** The reviewer's branch `claude/vibrant-cerf-sje726` at 21246a44: `WHAT_A_NEURON_HAS_20261118.md` (in full), `ROADMAP_TO_END_TO_END_PERSISTENT_LEARNING_20261118.md`, `THE_TISSUE_DEFINED_REVISION_5_20261119.md` (sections 3, 4, 6 and 7), `THE_TISSUE_AS_RUN_20261119.md` (in full), `STATUS_OF_STEP_1_20261119.md` (sections 1, 2a, 3 to 4 and rows 135 to 153 in full; rows 1 to 60 in part), and the programs of `reviewer_tissue_20261119/`: `tissue.c`, `tis.py`, `drive.py`, `outq.py`, `deps/TISSUE_STEP1/tissue_step1.py` (E63's program), `deps/TISSUE_EVENT/readings.py`, `yesno/add.py`, `addtext.py`, the heads of `sysx.py` and `mid.py`. The workers' addition programs: `ADDITION/addsys.py` (43f6487c), `output_reach4_20261119/addcheck.py` (b33cc712), `TISSUE_CHECK_20261119/addsys.py` (033d9b52). The default branch at a8838141: the coordinator's decisions 38 to 74. Not read: rows 61 to 134 of the status except as section 2a summarises them; the 432 result files; the event-driven kernel; the workers' reports beyond what the status quotes.

**INDEPENDENCE.** Executed, by a separate session with its own script. The tissue itself is run with the reviewer's program, so a fault of that program would be shared; it has been written again from its description by a worker and found equal at every byte (status, rows 39 and 47). I do not claim a cross-family review in the sense of the project's skill; no checker gate exists and none is claimed.

**CHECKS_EXECUTED.** `python3 PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/audit_tissue.py first` from the repository's root (about 20 minutes on four cores; without `first`, which leaves out the two first forms of the definition, about ten). Its four parts are quoted below. Two controls of the script itself: the tissue "N0 as run" ends at 3.24 connections a cell with 1,240 made and 10,474 withdrawn, the reviewer's own reference for the build that rounds every operation separately (`THE_TISSUE_AS_RUN`, section 5); and the tissue whose strengths never change gives the mean outputs 0.0046, 0.00088, 0.00022 and 0.00005 at 1 to 4 connections, the reviewer's row 2.

## 3. What was built, as programs

| | Program | What it is | What learns in it |
|---|---|---|---|
| a | `tissue_step1.py` (E63, stage A) | The definition, revision 5: a cell adds what arrives over its connections (strength 1/8 each at birth, a quarter of the cells inhibitory), passes it through a threshold, and gives an output between 0 and 1; 32 steps a byte; the line of the present byte is held at 1 | Strengths, by one of two rules that use no error (section 4.3); connections made and withdrawn |
| b | `tissue.c` with about thirty options | The same step, written for speed; the options are the changes tried in the night. "The tissue as run": totals of strength conserved, a gain for each cell, connections withdrawn by order in time, made when a cell's long mean is low, a time constant for each cell, long axons | No strength by experience; connections made and withdrawn; one gain a cell |
| c | `outq.py`, `yesno/yq.py` | "The output end": one unit for each byte value, a weighted sum of the recorded outputs of cells, changed by the right byte or by yes or no (rule Q) | The weights of these units, outside the tissue |
| d | `add.py`, `addtext.py`, the workers' `addsys.py`, `addcheck.py` | Small systems on lines such as `7+8=15`: features computed from the last bytes, read by one answering unit under rule Q | The answering unit's weights |

## 4. Findings

### 4.1 No error reaches a cell in any tissue that was run (*the programs' text*, part 4 of the script)

- `tissue_step1.py`, line 331: `z[YES_LINE] = 0.0; z[NO_LINE] = 0.0`, at every step. The definition says so itself (sentence 88: "At step 1 their lines are never active, because there is no correction yet").
- `tissue.c`, line 98: `if (t->kind[i] == FBK) { x[i] = 0.0; continue; }`. The 64 feedback cells give zero in every run of the night.
- `outq.py`, head: "The tissue lives its whole life once (no output end inside it; the output end reads and does not act back)". The life is recorded in an array and the output units learn from the array afterwards.
- `tissue.c`, line 119: with the option of the tissue as run (`prune_only`), the rule keeps two sums for each connection and `continue`s; no strength is changed. Strengths move only when a cell rescales all of them to a fixed total and when a connection is made or withdrawn.

This follows the roadmap's staging: step 1 has no correction, step 2 brings it. But the work did not leave step 1, and a second consequence is not discussed in the status. In the definition the feedback acts through the first form's rule: a cell that receives from a feedback cell has a mismatch, and formula 7 changes its strengths (sentences 90 to 93). The tissue as run no longer has formula 7 or any rule in which a mismatch changes a strength. **The changes that let step 1 be met removed the means by which step 2 was to work.** A feedback cell switched on in the tissue as run could change activity, and through activity which connections are made or withdrawn; no strength would change with the sign of the answer.

In the small programs of the last two hours an error was given to cells in a few first attempts, each one rule tried once: as a further stimulus to cells that copy what is present, and as a gate on how fast they copy (`sysx.py`, status row 139: no gain); and as a change of a cell's two sensitivities up after yes and down after no (a worker's variant, row 144: failed). `mid.py`, in which a middle cell changes in proportion to yes or no less its running share, was set aside by the reviewer as not Mr. Xu's picture, because the program hands the answer to every cell. None of them relays a mismatch to a cell over connections.

### 4.2 The numbers of the birth decide step 1 before any rule acts (*computed here*, parts 1 and 3)

| Quantity | Value | Consequence |
|---|---|---|
| Connections between the input lines and the 256 cells that drive the output lines | 9 to 11 (median 10) on three seeds | A byte must cross ten cells in series |
| Largest output, by connections from the input, wiring unchanged | 0.126, 0.016, 0.0025, 0.0004 at 1 to 4; 4e-7 at the output side | 8-fold a connection: a cell passes on one eighth of one sender |
| Recording floor; emission level | 0.01; 1/2 | Every cell from 3 connections on is recorded as silent (240 of 240, 220 of 220), and with the wiring unchanged no byte can ever be emitted |
| Cells whose threshold is above zero | 1 of 2,048 | The threshold can only lower activity and has nothing to lower; above zero the cells are linear |
| Target of a cell's mean output against how often a byte value is present | 12.5% against 6.2% at most (a digit 3.7%, a letter 1.4%) | A cell that answers to one value is below target for ever and makes a connection every 128 steps |
| Connections from input lines at birth | 137 to 154 from the 46 lines in use; 611 to 696 (81% to 84%) from lines that never carry anything | 5% of the cells (102 to 115) receive a byte at all; a byte value has a median of 3 receiving cells, and on two of three seeds one or two values have none |

The reviewer found most of these during the night (the fall of amplitude, row 2; the one-sided threshold; the unreceived lines, row 136), and the coordinator had foreseen two before the run (decision 51: that every cell may come to read its whole neighbourhood, and that the unchanging tissue is "expected silent" at four connections). The definition states the third itself (sentence 60: "a cell driven only during one byte value cannot reach a* = 1/8 ... so such a cell keeps growing"). What I add is that they are arithmetic on the definition's numbers, one page of it, and that together they fix the outcome of stage A under the definition's rules: only something that multiplies the signal about 8-fold at every connection could change it, and the definition's one control, the threshold, can only lower activity.

### 4.3 The measure of step 1 is at its ceiling when nothing changes, and the two first forms could only lower it (*computed here*, part 3; the coordinator's decision 73)

In the wiring with no change of strength the present byte explains 1.000 of the variance of every varying cell's output, at every distance. Step 1 asks that the byte "can be told from their recorded outputs ... better than a tissue whose strengths never change". Nothing that changes strengths without a teacher can add to what is already complete; it can only keep it or lose it. For the bytes one and two back the unchanging wiring holds nothing, because a cell's potential halves at each of the 32 steps of a byte; what later supplied them was a time constant for each cell, drawn at birth, and no change of strength. What the unchanging tissue lacks is amplitude and slowness, and neither is learning.

The two first forms lose it, as decision 73 verified. Run again here with the reviewer's program: the first form ends at 91.7 connections a cell, having made 1.10 million and withdrawn 0.93 million connections in 16,384 bytes; the coincidence form at 25.6 a cell, having made 1.91 million and withdrawn 1.87 million. In both, the bytes explain none of the cells' output on bytes held out from the fit (0.03 and below zero), and cells far from the input are active on their own, in the coincidence form at about the target rate from two connections on. My reading of the cause, which is the definition's own sentence 43: the first form takes as a cell's prediction "the running average of the cell's own arrival" and changes strengths so as to "cancel the foretold part of the arrival". A cell that makes its input equal to its own average is removing what distinguishes one byte from another. A running average can foretell only that what arrives next resembles what arrived lately; nothing in it lets a cell foretell a change, and a rule that shrinks the mismatch against such a prediction shrinks the changes themselves (section 6, point 2).

### 4.4 The tissue as run carries bytes and combines almost nothing (*computed here*, part 3)

Seed 73129, a life of 16,384 bytes, then a probe: every pair of digits 24 times as the bytes `+`, d1, d2, the output read at d2. The layout is balanced, so the variance of each cell's output divides exactly into the part of the present digit, the part of the digit before it, the part of the two jointly beyond the sum of their parts, and the rest (earlier bytes and the tissue's own state). Pooled over cells:

| Tissue | Present digit | Digit before | The two jointly | The rest | The same near the output lines |
|---|---|---|---|---|---|
| Wiring unchanged | 1.000 | 0.000 | 0.0000 | 0.000 | 0.882; 0.117; 0.0004; 0.000 |
| N0 as run | 0.617 | 0.041 | 0.024 | 0.305 | 0.268; 0.135; 0.020; 0.554 |
| N256 as run (time constants) | 0.204 | 0.150 | 0.010 | 0.610 | 0.011; 0.018; 0.000; 0.933 |
| L8-256 as run (long axons as well) | 0.443 | 0.226 | 0.028 | 0.291 | 0.410; 0.212; 0.031; 0.333 |

Three readings, mine.

1. The joint part is 1% to 3%. Cell by cell, among the cells whose output reaches 0.01, the median joint share is 1.3%, 0.0% and 0.5% in the three tissues as run and the largest 27%, 18% and 9%; one cell in the three tissues together has more than a quarter of its variance there. (In the unchanged wiring 48 cells do, at outputs of the order of a billionth; of its 113 cells that reach 0.01, none has any.) A cell that answers to two bytes together, which the roadmap names as "the means of generalising" (capability h), hardly exists in these tissues. The reviewer met the same fact from the other side in row 142 ("a cell that adds up evidence from each recent byte separately cannot answer to two numbers together"); the probe places it in the tissue and not only in the reader.
2. A third to nearly two thirds of what the cells do is neither of the last two digits. Part of it is earlier bytes; part is the tissue's own state (thresholds above zero in 791 to 1,642 of 2,048 cells, outputs at the ceiling 1 at nearly every distance, 218 to 627 gains at their bound). To a reader this part is noise.
3. Time constants buy the digit before at the price of the present one, and near the output lines of N256 the two digits together explain 3% of the output. The reviewer's rows 26, 42 to 44 say the same.

On the stream itself the picture agrees: the present byte explains 0.58, 0.20 and 0.56 of the variance in the three tissues as run, the last eight bytes with their effects added 0.71, 0.46 and 0.76.

### 4.5 What the score of that night could and could not show (*computed here*, part 2)

| Guesser of the next byte, on the stream of step 1 | Share right |
|---|---|
| Table of the last 1, 2, 3 bytes (the project's yardsticks; last quarter of 262,144 bytes) | 10.4%, 12.4%, 27.4% |
| Knows the form of an item (where the task byte, the numbers, the letters, the equals sign, the answer and the end mark stand) and no arithmetic | 25.9% |
| Knows the form and does the arithmetic | 36.6% |
| The output end on any tissue of the night, told the right byte (status, section 2a and rows 135, 136) | 5% to 20% |
| The output end on the exact last three bytes, no tissue (status, section 2a) | 26.5% told the right byte; 24% to 26% from yes or no |

The answer's two bytes are 12.5% of the stream; the whole arithmetic is worth 10.7 points of this score, and all of the night's figures lie below the level at which it begins to count. The reviewer saw this at row 141 ("every earlier score of tonight was the share of all next bytes right, which rises on the form of the items alone"). Two things I add. First, the level of the form alone is a number, 25.9%, and it was not reached. Second, the lives could not have shown arithmetic under any learner: a life of 16,384 bytes holds 1,029 items, 6.5% of one pass over the 15,840 taught pairs; 262,144 bytes hold one pass; the dense network under back-propagation needed 409,600 items, about 26 passes, to reach 80 on never-taught pairs (receipt `e42_G_B_73129.json`).

### 4.6 In the systems that answered sums never shown, the program adds (*the programs' text*, part 4)

- `ADDITION/addsys.py`, line 82: `m = int(np.floor(w1 * hist[d1] + w2 * hist[d2]))`, and line 84 makes one column for each value of `m` with the present byte: `col(('m', d1, d2, G, m) + tuple(hist[:G]))`.
- `addcheck.py`, line 40: `m1 = val[hist[-2]] + val[hist[-4]]; m2 = val[hist[-3]] + val[hist[-5]]`; line 44: `f1 = m1 * NA + AI[p]`.
- `TISSUE_CHECK_20261119/addsys.py`, line 82: `tot = s.A + s.B; cnt = (tot >= np.arange(19)).astype(float)`.

In each, the two digits are taken from positions that the author chose, their values are added by the program, and a "cell" is a column whose index is that sum and the present byte. The answering unit then learns which byte follows which sum: a table of 19 rows. A pair never shown is answered because its sum was shown through other pairs; in the two programs that reach 100% the only misses are the pairs whose sum no shown pair has (9+9; 0+0). The status says that the amount code is "laid down at birth, not learned"; at the level of the program more than the code is laid down: the adding is. Where the pools of random cells had to find the adding cells themselves, the status reports 20% to 70% on unseen sums with the exact tuning still given, and that "no keep rule finds the adding cells" (rows 147, 148, 151).

These runs do establish something useful: what a learned inner code would have to look like for sums to carry over (digits in order on one scale, a gate by place in the line), and that the control with the digits in random order fails.

### 4.7 How far the figures can be relied on (the status's own statements)

The reviewer records these itself, and I repeat them because they bear on any reading of the night: two lives that differ in the last bit of one sum become different lives, and "a difference of 0.04 between two settings on one seed is therefore within what rounding alone gives" (`THE_TISSUE_AS_RUN`, section 5); "the three seeds used by all lines tonight are the best three of six" (section 2a, 13:30); the ranking of tissues by the readings tool was withdrawn when it did not agree with what the output end learns (section 2a, item 5). The status has 153 rows from five lines in less than a day, and the tissue as run has about twenty numbers set by hand, counting those of the birth (`THE_TISSUE_AS_RUN`, sections 1 and 4), several of them there to mend a consequence of an earlier one: a gain because a connection passes one eighth, long axons because the ends are ten connections apart, slow copies because a cell forgets within a byte.

## 5. Why it did not work, in one chain (my reading)

1. The roadmap puts first a step with no correction: the tissue must carry bytes before it is taught anything.
2. The definition places the two ends on opposite edges, ten connections apart, with numbers under which a signal falls 8-fold a connection. Step 1 becomes the problem of carrying a byte across ten cells.
3. For the present byte the measure of step 1 is already complete in the wiring with no change, so a rule without a teacher can only lose on it, and the first two do. The search turns from learning to amplitude, wiring and slower cells, and ends with a tissue in which no strength changes by experience.
4. A reader is attached outside. It learns about two bytes' worth of the form of an item, on a score that the arithmetic hardly enters, in lives a twenty-fifth to a four-hundredth of what the reference needs.
5. After Mr. Xu's rulings of 14:24 to 15:12 the work moves to small programs. Those that answer unseen sums compute the sum themselves.
6. At no point does a cell change a connection because an answer was wrong.

## 6. What I think

These are my views. They propose; the choices are Mr. Xu's and Mr. Zhou's.

1. **Build the loop first, at the smallest size.** The thing not yet built is the one Mr. Xu states as the aim: told when wrong, the cells correct themselves. I would make it the first object and not the second step: single-digit addition as text (his choice of 15:09 to 15:12), a few dozen cells, the right answer shown after the attempt (the form of teaching he accepted) or yes or no. One measure, fixed before any run: whole answers right on sums never shown, beside a table of pairs (which gives zero), and beside the same small network trained by back-propagation as the comparator.
2. **Give "prediction" its content.** In his words a cell predicts "the new information that arrives from other cells" (section 8 of the record; the English is the reviewer's). The reading that follows is mine and not his: the cell makes the prediction from what it has, its own state and what other cells are sending it now, over connections that learn, and compares it in the cell with what then arrives. He has accepted cells with separate parts, which is what this needs: one part receives what arrives, the other receives from other cells and predicts it; the predicting connections change by the mismatch times the sender's recent activity. Both quantities are at the connection. At the cells that receive the bytes this is the prediction of the next byte, the mismatch is the system's error, and it is formed inside the cell with no reader outside. Such rules are in the literature and learn small tasks (section 8). The definition's first form took the other reading, an average of the cell's own past, and section 4.3 shows where that leads.
3. **Relay the mismatch, and measure how far it must go.** Cells whose output a predicting cell used receive that cell's mismatch over return connections, through dedicated cells, as he described; how many relays are needed is then a measurement, as in the roadmap's step 3. The record already holds the encouraging fact: the sweep, a crude relayed error, reaches 80 on never-taught pairs of this family.
4. **Drop the geometry that made transport the problem.** The two opposite edges, the 256 lines and the neighbourhood of 1/8 were the definition's choices, and he has withdrawn the lines. With an input part and an output part of many cells each, one to three connections apart, none of the machinery of section 4.7 is needed. I would add a mechanism only when a failure of the end-to-end measure asks for it.
5. **Keep a cell that can answer to two things together.** In the definition's tissue 2,047 of 2,048 thresholds were zero. Such a cell adds what arrives and cannot require two inputs together, whatever it is taught. A cell with a working threshold can, and so can the gated cell, which he has allowed to stay.
6. **Let the code of a digit be learned by the error, and look for it.** The addition runs show what the code must be like. The test of a learning rule is whether that order appears in the strengths from the digit lines without being laid down. That is a reading of saved weights, not a new mechanism.
7. **Before a run, the page of arithmetic.** What a connection passes on, how often a line is active against a target, how many cells receive a line, how many items a life holds against what the comparator needs. Each of the faults of section 4.2 and 4.5 is visible on such a page, and the project's own rule is to run the cheapest contradicting check first.
8. **What is worth keeping from the night.** Rule Q, which learns from yes or no at 0.88 to 0.92 of the rule told the right byte and was reproduced by two programs; output cells that come to speak with one voice through an inhibition learned from their joint errors (rows 137, 140); the local rule that equalises an adding cell's two strengths (row 147); the stream and the reserve of never-taught pairs, which the coordinator verified; and the event-driven design of the program.

One connection to the audit of 18 November. I proposed there, for E61, the form in which each cell keeps a trace at its own connections and uses only the error that reaches it, with no pass backward in time. E61 was withdrawn; the coordinator's decision 49 gives as the reason that it relays in reverse order in time over an item, which the route does not allow. The form with a trace at each connection needs no such pass: the trace runs forward with the stream. It is the same proposal as points 2 and 3, seen from the network of the record.

## 7. Findings by weight, limits and disposition

**FIRST_FALSE_ASSERTION.** None found among the statements that I executed; the status corrects its own in its section 6. One statement I would not let stand without its condition: "a small network gets right sums it was never shown, provided cells carry a digit as an amount" (row 143). It holds for systems in which the program forms the sum (section 4.6).

**SEVERITY.**

| | Finding | Weight | Ground |
|---|---|---|---|
| 1 | No error changes a connection in any tissue run; the tissue as run has no rule through which the feedback of step 2 could act | BLOCKER for any conclusion about cells that learn | 4.1 |
| 2 | Step 1's outcome is fixed by the numbers of the birth; its measure is complete with no learning | MAJOR | 4.2, 4.3 |
| 3 | The score of the output end lies below the level of the form alone; the lives are too short for the arithmetic | MAJOR | 4.5 |
| 4 | The tissue as run offers almost no cell that answers to two bytes together | MAJOR | 4.4 |
| 5 | In the systems that answer unseen sums the program adds | MAJOR for how row 143 is read | 4.6 |
| 6 | 81% to 84% of the input connections go to lines never used | MINOR | 4.2 |
| 7 | One seed, lives that diverge on a last bit, three favourable seeds | NOTE; the status says so | 4.7 |

**GAPS.** One seed (73129) for every tissue run here, and lives of 16,384 bytes; the probe is one sequence. The tissue is the reviewer's program. Rows 61 to 134 of the status were read through its summary. The workers' reports were read as the status quotes them. I did not run the output end or the addition programs; for those I rely on their text.

**CAUSAL_LIMITS.** The probe measures variance in recorded outputs; a small joint share does not prove that no reader could use it, only that it is small beside the rest. The chain of section 5 is a reading of the record and the programs, not an experiment. Nothing here shows that the proposals of section 6 would work; they are untested in this project, and the literature I name is cited from memory.

**DISPOSITION.** The line as built: INCOMPLETE as a test of cells that learn, since the learning was not in it; its negative result is to be read as "a tissue with no correction, read from outside, does not learn the family", and no further. The next object, in my view: the loop of section 6, points 1 to 3, at the smallest size.

## 8. Literature to source-check

Cited from memory; none was opened for this document.

- R. Urbanczik and W. Senn, "Learning by the dendritic prediction of somatic spiking", *Neuron* 81 (2014): a cell with two parts, one predicting what the other receives; the rule is local.
- R. P. N. Rao and D. H. Ballard, "Predictive coding in the visual cortex", *Nature Neuroscience* 2 (1999); J. C. R. Whittington and R. Bogacz, "An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity", *Neural Computation* 29 (2017): mismatches relayed between cells over several passes, with local changes of strength.
- J. Sacramento, R. P. Costa, Y. Bengio and W. Senn, "Dendritic cortical microcircuits approximate the backpropagation algorithm", NeurIPS 2018.
- T. P. Lillicrap, D. Cownden, D. B. Tweed and C. J. Akerman, "Random synaptic feedback weights support error backpropagation for deep learning", *Nature Communications* 7 (2016): an error relayed over separate return connections.
- G. Bellec and others, "A solution to the learning dilemma for recurrent networks of spiking neurons", *Nature Communications* 11 (2020): a trace at each connection, run forward in time, and an error delivered to the cell.
- T. P. Vogels and others, "Inhibitory plasticity balances excitation and inhibition in sensory pathways and memory networks", *Science* 334 (2011), and E. L. Bienenstock, L. N. Cooper and P. W. Munro (1982): the two rules of the coincidence form, which regulate activity and do not by themselves teach a task.
