# Step 2 of the roadmap: the two ends correct themselves (brief, revision 1; written from 01:03 UTC on 19 November 2026 by the clock of this machine)

This is task 2.1 of `$C/TASK_GRAPH_20261119.md`. It is written while stage A of step 1 (E63) runs and **is not released before the coordinator has read stage A's report** (decision 56). It is also not released before stage B of step 1 has been run and step 1 has been reached (section 5.1). The coordinator gives it its experiment number. The brief carries everything the worker needs; where it seems not to, the worker stops that part and asks (section 14). Every number in it that can be computed without running a tissue is computed by `check_step2_brief.py` beside this file (Python standard library only, `python3 -I`, about 5 seconds); its output is section 19. Numbers that can be known only from stage A of step 1 are written ⟦A: …⟧, and numbers known only from stage B of step 1 are written ⟦B: …⟧ (section 18). They are filled in at release without any other change to the text.

## 0. Sources, shell names and host

- `C = PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117` on the reviewer's branch `V = origin/claude/vibrant-cerf-sje726`.
- The governing words are Mr. Xu's, in `$C/WHAT_A_NEURON_HAS_20261118.md`, sections 7 to 17; for this step, sections 11 (the system is told when it was wrong and corrects itself), 13 (cells receive the feedback of the error; yes or no is the form for the next byte), 14 (richer feedback, not used here), 15 (the input end receives the feedback of the output end's information), 16 (dedicated cells receive the feedback and pass it on; how is learned) and 17 (a connection may run through other cells).
- The roadmap is `$C/ROADMAP_TO_END_TO_END_PERSISTENT_LEARNING_20261118.md` (draft 3, with the change of 00:31 UTC on 19 November). Step 2 and the conditions of its section 5 bind this brief.
- The tissue is `$C/THE_TISSUE_DEFINED_REVISION_5_20261119.md` ("the definition"); sentences and formulas are cited by revision 5's numbers. The program is this definition, the step-1 program it continues, and the decisions of section 4.6, and nothing else.
- The brief of step 1 is `$C/TASK_STEP_1_THE_TISSUE_READS_THE_STREAM_20261118.md`, revision 3 ("the step-1 brief"). Its program (`TISSUE_STEP1/tissue_step1.py` on the worker's branch), its stream, its birth and its saved states are continued here.
- The risks H2 to H4 are those of `$C/FIFTH_READING_OF_THE_TISSUE_DEFINED_20261118.md`.
- The coordinator's decisions 48 to 57 (19 November); decision 54 asks this brief to name the comparator, its stream and the units of the comparison; decision 56 says that the first form of step 2 does not depend on stage A's outcome.
- The design block is that of `minifly_b/package/DOCS/MINIFLY_OBJECTIVE_CONTRACT.md` ("the contract").
- The host of the worker is Linux on x86-64 with four cores and Python 3.13; the program uses Python 3.13 and numpy only.

## 1. Terms

All terms of the definition (its section 0) and of the step-1 brief (its section 1) are used in their sense. In addition:

- *Life position* t: the t-th byte of the one stream of a tissue's life, counted from 0. Step 1 read positions 0 to 262,143; step 2 reads positions 262,144 to 393,215 (131,072 bytes).
- *Attempt for position t*: the emission read at the reading step of position t − 1 (sentence 75). It is the tissue's first and only attempt at the byte that arrives at position t, made before that byte arrives; no emission counts as wrong (sentence 77).
- *Told*: a tissue whose yes and no lines are set by the world as sentences 86 and 120 say, from life position 262,144 on.
- *Never told*: a tissue whose feedback lines (and, in form (b), emission lines) stay 0 at every step, as at step 1.
- *Form (a)* and *form (b)*: the two forms of choice C52 (sentence 97). In form (a) the 64 feedback cells receive the yes or the no line. In form (b), in addition, 64 *emission feedback cells* each receive 8 of the 256 *emission lines*; the line of the emitted byte is active for one step at φ = 0.
- *Class of a position*: the role, in its item, of the byte that arrives at that position: TASK (the task byte), A1 and A2 (the digits of the first operand), SIGN (`+ - > <`), LET (a letter), B1 and B2 (the digits of the second operand), EQ (`=`), R1 and R2 (the digits of the answer), END (the end mark, byte 10). The class is the experimenter's reading of the stream; no cell has it.
- *Classes knowable from the last few bytes*, K: SIGN, EQ and END (section 3.3).
- *Stretch*: 16,384 consecutive step-2 positions; P1 is life positions 262,144 to 278,527, …, P8 is 376,832 to 393,215. *Late* means P8.
- *Share right* R(c, P): over the positions of class c in stretch P, the share whose attempt equals the arriving byte.
- *Frequency emitter*: a rule that emits, at every position, the byte most frequent in positions 0 to 262,143 (the end mark on all three seeds; section 19, B.1).
- *Table of the last n bytes*: a rule that emits, at position t, the byte that most often followed the bytes at t − n, …, t − 1 in positions 0 to 262,143 (ties to the lower byte value; an unseen context gives the frequency emitter's byte).
- *Comparator*: the network of section 9, trained by back-propagation, the control known to learn this stream (roadmap, section 5, condition 5).

## 2. The question, and what reaching step 2 buys

**Question.** When the world tells the tissue of step 1, continuing in the same life on the same stream, at every position whether the byte it gave out was the byte that came (yes or no, through the feedback cells, sentences 85 to 95), does the tissue come to give out, at the first attempt, the byte that comes wherever that byte can be known from the last few bytes, above what the frequencies of bytes give, while a tissue that is never told does not; and does it matter whether the input end receives only the yes or no (form (a)) or also what the output end gave (form (b))?

**What reaching it buys.** From here on the tissue is end to end: bytes go in at one end and come out at the other, and both ends are corrected by the loop that Mr. Xu names as persistent learning (section 11: "告诉系统它错了让它自己去纠正"). Its level at the knowable positions is the floor against which step 3 (relay inward) is measured (roadmap, step 2). It is also the first test of whether yes or no passed on alike, through dedicated cells, can correct the cells that emit (sentences 94, 95; risk H2).

**Level.** E0 to E1. The bytes of the classes K are reinforced by yes or no at every position, and the same transitions are probed again and again, each probe a first attempt before its own feedback. This is taught association (E1) at most; a table of the last three bytes passes the rule (section 3.3), so no claim above E1 is possible, and none is made.

## 3. The design and report block of the contract

1. **Claim and level.** "Under the stated conditions, the [form (a) | form (b)] tissue, told yes or no at every position, gives out at its first attempt the byte that comes at the positions where it is fixed by the last three bytes (the sign after the first operand, the `=` after the second operand, the end mark after the answer), in at least half of those positions of each kind late in step 2, above the frequency emitter and above a tissue of the same life that is never told." Level E1 for these transitions; E0 for the rest.
2. **Exposure record.** The tissue has read, without feedback, life positions 0 to 262,143 (step 1: the taught half of the first family, 15,840 pairs, about 16,360 items). At step 2 it reads positions 262,144 to 393,215 (about 8,160 more items of the same taught pairs, in the record's order; section 19, A.1, E.2) and is told yes or no after every attempt. The never-taught half, 15,844 pairs, never appears. Every scored attempt precedes its own feedback; no position is scored after its feedback. The answer bytes R1 and R2 are scored and reported but not part of the rule.
3. **Countermodels.** Computed on the stream (section 19, part B), fitted on positions 0 to 262,143 (what the tissue has read) and scored on the step-2 positions:

   | Rule | SIGN | EQ | END | TASK | A1 | A2 | LET | B1 | B2 | R1 | R2 |
   |---|---|---|---|---|---|---|---|---|---|---|---|
   | Frequency emitter (the end mark) | 0.000 | 0.000 | 1.000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
   | Table of the last 1 byte | 0.000 | 0.497 to 0.602 | 0.410 to 0.503 | about 0.25 | about 0.12 | 0 | about 0.04 | 0 | 0 | about 0.12 | 0 |
   | Table of the last 2 bytes | 0.000 | 0.503 to 0.541 | 0.488 to 0.517 | about 0.25 | about 0.12 | about 0.11 | about 0.04 | about 0.005 | about 0.10 | about 0.12 | about 0.11 |
   | Table of the last 3 bytes | 1.000 | 0.995 to 0.998 | 1.000 | about 0.25 | about 0.11 | about 0.11 | about 0.03 | about 0.04 | about 0.10 | 0.335 to 0.344 | 0.326 to 0.332 |

   (Ranges over the three seeds.) The classes in which the table of the last three bytes scores at least 0.9 are SIGN, EQ and END on every seed; these are the classes K. The second answer byte given the first (R2) is not knowable from the last three bytes (about 0.33): it belongs to later steps. An exact table of the last three bytes passes the rule of section 7.3, so the result is a taught association (E1) and no more. A constant emitter, whatever byte it emits, reaches 0.5 in at most one of the three classes K (section 19, B.1), and the frequency emitter reaches it only in END; so the rule cannot be passed by emitting the most frequent byte, and it can be passed without any answer byte being right.
4. **State and controls.** All state persists from birth through step 1 into step 2; nothing is reset (roadmap, conditions 1 and 2). The control is the never-told tissue of the same seed and form, which lives the same life with silent feedback lines (condition 3; section 5.3 on how it is started). The positive control known to solve the stream is the comparator (section 9), run only on the coordinator's release. Feedback budgets: the told tissues receive one yes or no per position; the never-told receive none; the comparator receives the arriving byte itself as its target, which is more than yes or no, and the report says so.
5. **Rule of success, cost and stopping.** Section 7.3; sections 10 and 11. The cheap preflight is checks 1 to 10 (section 8) and the first stretch P1 of form (a) on seed 73129, read before the rest (section 11.2).

## 4. The tissues

### 4.1 What step 2 continues

Step 2 continues, in the same life, a tissue of step 1 from its saved state at the end of step 1: the file `TISSUE_STEP1/runs/B_<arm>_<seed>/final_state.npz` of the step-1 brief (section 13.1), written after the reading step of life position 262,143, with the states of its three generators and the stream position 262,144. `<arm>` is the arm that reached step 1 (section 5.1). If step 1 was reached only by opening 2 of the step-1 brief (a* = 1/32), the tissue continued is that opening's tissue and its final state, as the step-1 report names it, and a* = 1/32 holds at step 2 as well. The never-told tissue and the told tissue of form (a) of a seed both start from this state (section 5.3).

### 4.2 The arms

| Tissue | Birth | Step-1 phase | Step 2 | Seeds |
|---|---|---|---|---|
| Form (a), told | Step 1's birth | Step 1's stage B tissue, continued | Yes and no lines set by the world | 73129, 93683, 116089 |
| Form (a), never told | Step 1's birth | The same | All feedback lines 0 | the same three |
| Form (b), told | Step 1's birth with 64 cells made emission feedback cells (section 4.4) | Its own, 262,144 bytes, all feedback and emission lines 0, the rules of the continued arm | Yes and no lines, and the emission lines, set by the world | 73129 |
| Form (b), never told | As form (b), told | The same step-1 phase | All feedback and emission lines 0 | 73129 |

Form (b) needs its 64 emission feedback cells from birth (cells added by an agent that knows the task has changed do not exist; `WHAT_A_NEURON_HAS`, item 4.8; sentence 88). It therefore cannot continue a tissue of step 1: its tissue is born anew from the same seed with those cells (section 4.4) and lives its own step-1 phase on the same 262,144 bytes, with the rules of the arm that reached step 1 and with every feedback and emission line silent, before its step 2. Its step-1 phase is not scored against step 1's rule; its emissions and connection counts are recorded (section 6). Form (b) is run on one seed because its step-1 phase doubles its cost; the coordinator may extend it to the other two seeds at release.

**The second tissue of the roadmap** ("a second tissue in which only the output end receives it") **is left out.** The definition cannot state it without a new rule: which cells take a feedback cell as a source is decided by the neighbourhoods and the random draws of formulas 12 and of the birth (sentences 6, 28, 58), and sentence 90 says that "nothing lays down in advance which cells of either end they reach". A tissue in which only output-end cells can reach feedback cells would need feedback cells removed from the neighbourhoods of all other cells, a new rule fixed at birth, and so a new birth and a new step-1 phase. The input end's share is read instead from the counts of encoder cells and output-end cells that take a feedback cell as a source (section 6, item 4), and from the difference between forms (a) and (b).

### 4.3 The rules at step 2

Everything is as at step 1 (step-1 brief, sections 5.3 to 5.6, with revision 5), except operation 1 of sentence 120:

1. If φ = 0 and the tissue is told: the world compares the byte that arrives at this position with the emission read at the reading step of the previous position. If they are equal, the yes line is set to 1; otherwise (including when no byte was emitted) the no line is set to 1 (sentences 77, 86). The line is 1 for this one step and 0 at all other steps. In form (b), the emission line of the emitted byte is also set to 1 for this one step if a byte was emitted. At life position 262,144 the world compares the emission read at the reading step of position 262,143 (the last of step 1) with the byte at 262,144: the telling begins at the first position of step 2.
2. A never-told tissue sets no feedback or emission line, at any step.

The feedback cells (64 in form (a); 64 plus 64 in form (b)) have their lines only, at fixed strength 1 (feedback lines) or 1/8 (emission lines), no ordinary sources, no strength change, no control of activity and no growth; their θ stays 0 (sentences 85, 97). Their outgoing connections are ordinary connections of the cells that took them as sources and change by formula 7, 11 and 12 like any other (sentence 90). No-cells are excitatory, yes-cells inhibitory (sentence 89); emission feedback cells have the kind drawn for them at birth (sentence 97).

### 4.4 The birth of form (b), exactly

The birth is that of the step-1 brief (section 5.3) for the same seed, drawn by `Random(seed)` in the same order and with the same results, with one addition after item 6 (the drivers) and before item 9 (the starting sources):

- A second generator `re = random.Random(seed + 700000)` draws the emission feedback cells: `EM = re.sample(eligible, 64)`, where `eligible` is the sorted list of cells that are neither feedback cells nor drivers (1,728 cells on each seed). Then, for each cell of `EM` in that order, `sorted(re.sample(range(256), 8))` are its emission lines, each at strength 1/8, never changed and never given up.
- The starting sources of every cell are drawn by `rb` exactly as at step 1, including those of the cells of `EM`; the sources drawn for a cell of `EM` are then discarded, so that every other cell's sources are the same as at step 1 and a cell of `EM` has its emission lines only. A cell of `EM` keeps the kind drawn for it in item 7 (sentence 97: "kinds as for other cells") and stays in the neighbourhoods of other cells.

The digests of `[EM, lines]` and the first cells are in section 19, D.1. The emission lines of the 64 cells cover 220 to 233 of the 256 byte values. All 128 feedback cells are part of the 2,048 (sentence 68).

### 4.5 The kind of feedback cells under the coincidence form

Sentence 89 leaves to this brief whether inner cells that receive from feedback cells under the coincidence form need the reversed kinds. A cell's kind is fixed for life (sentence 16), and the feedback cells were born at step 1 with the kinds of sentence 89, so a tissue continued from step 1 cannot have them reversed. If the coincidence arm is the one continued (section 5.1), its feedback cells keep the kinds of sentence 89; the reversal is a further form of C56 that needs a new birth and its own step-1 phase, and is not run under this brief.

### 4.6 Decisions of this brief

1. Step 2 reads 131,072 bytes, life positions 262,144 to 393,215, about 8,160 items: as many positions per class (about 8,160 for each class but LET) as there are taught pairs per half of each operation's split, twice; long against every time constant of the tissue (P_s = 128 steps is four bytes; ϑ's 4,096 steps is 128 bytes).
2. The telling begins at the first position of step 2 (section 4.3, item 1).
3. If both arms reached step 1, the first-form arm is continued: it is the definition's first form, and the coincidence form does not use the cell's prediction and would first have to be reconciled with sentence 38. The coincidence arm is continued only on the coordinator's word.
4. Form (b) on seed 73129 only (section 4.2).
5. The second tissue of the roadmap is left out (section 4.2).
6. Feedback kinds under the coincidence form are those of birth (section 4.5).
7. The never-told and told tissues of form (a) start from the same saved state (section 5.3).

## 5. Conditions of release, and how the tissues are started

### 5.1 Conditions of release

This brief may be released only when (i) the coordinator has read stage A's report of step 1 (decision 56), and (ii) stage B of step 1 has been run and step 1 has been reached by at least one arm on all three seeds (step-1 brief, section 7.6), possibly through its opening 2. If stage B is not run, or step 1 is not reached, this brief is not released; it is not adapted to start from a tissue that has not reached step 1. The arm continued, and the values marked ⟦B: …⟧, are entered at release.

### 5.2 Which state

For each seed, `final_state.npz` of the step-1 tissue named in section 4.1, with its SHA-256 recorded in the receipt of every step-2 tissue that starts from it (⟦B: three SHA-256 values⟧).

### 5.3 How the never-told and the told tissue of a seed are started

Roadmap condition 3 says that a tissue is never copied or reset to make a control. Both tissues are tissues born from the seed that have lived the same step-1 phase on the same stream; the program is deterministic (step-1 brief, checks 2, 3 and 8), so each has, at life position 262,144, the state that every tissue of that seed and arm has after 262,144 bytes, which is the saved state. The program loads that state instead of repeating the arithmetic of step 1 twice. This is not a copy made to build a control: no state is changed or chosen by the experimenter, and either tissue could be recomputed from birth with the same result. The coordinator confirms this reading of condition 3 at release; if the coordinator reads condition 3 otherwise, each step-2 tissue is recomputed from birth, which adds 262,144 bytes to each (section 10).

## 6. What is recorded

For every step-2 tissue (and, where stated, for the step-1 phase of form (b)):

1. At every position: the attempt (the emission, int16, −1 for none), the arriving byte, and the feedback given (yes, no, or none). For form (b), the emission line set.
2. Per stretch: the counts of yes and no given; the share of positions with a byte emitted; R(c, P) for every class; the five commonest emitted bytes.
3. Per stretch, for the output-end cells: the mean activity at the reading step; the mean activity at the reading step of the cell whose byte arrives next (the "right cell"); the number of output-end cells whose activity exceeded 1/2 at least once.
4. At birth (or at the start of step 2), at the end of every stretch, and at the end: the numbers of output-end cells and of encoder cells that take a yes-cell, a no-cell and an emission feedback cell as a source (E2); the number of connections from each kind of feedback cell, with their mean strength; the connection counts of the step-1 brief (its section 7.1, item 4).
5. For the step-1 phase of form (b): the emissions and the counts of item 4 at its end.
6. At the end: the full state, as at step 1 (step-1 brief, section 13.3), so that step 3 can continue the tissue.

No cell's output is recorded beyond these; nothing acts on the tissue for a measurement.

## 7. The measure and the rule of success

### 7.1 The measure

For each told and never-told tissue, each stretch P1 to P8 and each class c: R(c, P), the share right at the first attempt (no emission counts as wrong), with its standard error (√(R(1 − R)/n), n the number of positions of the class in the stretch, about 1,020 for each class but LET; section 19, A.3 and B.4). Beside it: the frequency emitter, the tables of the last 1, 2 and 3 bytes (section 3.3, computed on each stretch by the analysis program as by the check script), and the comparator's share at the same positions (section 9), when run.

### 7.2 What can be passed and what cannot

- A tissue that emits the frequency emitter's byte (the end mark) everywhere scores 1.000 in END and 0 in SIGN and EQ, and fails conditions 1 and 3 below.
- A tissue that never emits scores 0 everywhere.
- A tissue that has learned the three transitions (task byte, digit, digit → sign; letter or sign, digit, digit → `=`; `=`, digit, digit → end mark) passes, whether or not any answer byte is right.
- The table of the last three bytes passes (section 3.3), which is why the claim stays at E1.

### 7.3 The rule of success (fixed before any run)

For a told tissue of one seed, step 2 is **reached** when, in the late stretch P8:

1. For each class c in K = {SIGN, EQ, END}: R(c, P8) ≥ 0.5.
2. For each class c in K: R(c, P8) ≥ R_never(c, P8) + 0.2, where R_never is the never-told tissue of the same seed and form.
3. The mean of R(c, P8) over the three classes of K is at least the frequency emitter's mean over them plus 0.2, that is at least 1/3 + 0.2 = 0.5333.

**Step 2 is reached in form (a)** when it is reached on all three seeds; **in form (b)** when it is reached on seed 73129 (and on any further seed the coordinator adds). **Step 2 is reached** when it is reached in form (a). Form (b) is read beside it; if form (b) reaches it and form (a) does not, the report says so and the coordinator decides which form step 3 continues.

The level 0.5 is half of what the table of the last three bytes gives; the margin 0.2 over the never-told tissue is more than ten standard errors of a share near 0.5 over 1,014 positions (0.0157; section 19, B.4). The numbers 0.5 and 0.2 are this brief's.

### 7.4 Readings beside the rule

For each told tissue and class, in these words: "the tissue gives the [class] byte at the first attempt in [R] of the positions late in step 2, against [table of the last 3], [never told], [frequency emitter]". For R1 and R2: "the answer bytes are right in [R] of the positions, against [table of the last 3]; they are not part of this step". The count of yes and no given in each stretch, and the share of positions with a byte emitted. Whether the curve of R(c, P) over P1 to P8 rises, stays or falls.

## 8. Checks before any run

Each check writes `TISSUE_STEP2/qual/check<n>.json` with an entry `pass`. If a check fails, the worker stops, reports the first difference and asks.

1. **The worked example with a no.** The program's functions, applied to revision 5's section 13, part 1 (F a no-cell with its line only, v = 0, θ = 0; the no line active at step 0), reproduce every row of its table to four decimals and its final strengths (O: u 0.7504, f 0.2498, F 0.2502; E: h 0.2554, c 0.4973; I: E 0.5003, q 0.2510, g 0.2495), as section 19, C.1 recomputes them.
2. **What a no and a yes move (sentence 93).** In the same example, O's strength from u changes by −0.0032 at step 1, −0.0013 at step 2 and −0.0005 at step 3 after a no (section 19, C.1). With F a yes-cell (inhibitory), the yes line active at step 0 and every other value unchanged, the rows and final strengths are those of section 19, C.2, and the change at steps 1, 2 and 3 is +0.0032, +0.0009 and +0.0003: a no weakens the excitatory input of the context that led to the emission, and a yes strengthens it. A cell with x = 0 at the previous step does not change (asserted on O with its previous activity set to 0).
3. **The never-told tissue is the step-1 tissue continued.** From the saved state of seed 73129 (section 5.2), the never-told tissue of form (a) run by the step-2 program for 2,048 bytes, and the step-1 program (`tissue_step1.py`, unchanged) continued from the same state for 2,048 bytes, give bit-equal states (every array, the three generators' states) and equal emissions.
4. **The feedback lines.** In a told tissue of form (a), seed 73129, over its first 2,048 bytes: at every step exactly one of the yes and no lines is 1 at φ = 0 and both are 0 at every other step; the yes line is 1 exactly when the emission read at the previous reading step equals the arriving byte; no emission gives the no line; at life position 262,144 the comparison uses the emission of position 262,143. In form (b), the emission line set is that of the emitted byte, and none when no byte was emitted.
5. **The birth of form (b).** For each seed, the birth of section 4.4 gives the digests and first cells of section 19, D.1; every array other than the 64 cells' sources, θ rules and lines is bit-equal to the step-1 birth of the same seed (step-1 brief, check 2).
6. **The stream goes on.** For each seed, the SHA-256 of life positions 262,144 to 393,215 equals section 19, A.1, and the bytes at 262,144 to 262,175 are those printed there; positions 0 to 262,143 still give step 1's digest.
7. **The scorer.** The analysis program, applied to synthetic attempts on seed 73129's step-2 positions, gives: for the frequency emitter and the tables of the last 1, 2 and 3 bytes, the values of section 19, B.1 to three decimals; for an emitter that never emits, 0 in every class; and the rule of section 7.3, taking each of these in turn as the told tissue and the never-emitter as the never-told tissue, is met by the table of the last 3 bytes only.
8. **Saving and resuming.** As the step-1 brief's check 8, for a told tissue of form (a) on seed 73129: 4,096 bytes uninterrupted, against `--ckpt-bytes 1024 --die-after-byte 1700` and `--resume`; bit-equal final states and equal records.
9. **No reset.** The program contains no assignment that sets a cell's or a connection's state other than by the formulas; at life position 262,144 the state loaded is the saved state bit for bit (SHA-256 of every array recorded).
10. **Cost.** CPU seconds per byte of a told tissue of form (a) and of the never-told tissue on seed 73129, over their first 1,024 bytes after 256 not counted, with the mean number of connections. The forecast of section 10 is made from these. If it exceeds the ceiling, the worker asks before starting.

## 9. The comparator (decision 54): a request to the coordinator

**What is asked.** On each of the three seeds, a small recurrent network trained by back-propagation on the same byte stream, life positions 0 to 393,215, read one item at a time, predicting the next byte:

- One layer of 128 gated recurrent units (GRU), a learned embedding of 32 numbers for each of the 256 byte values, and a readout of 256 classes; 103,424 weights (section 19, E.3).
- The state is carried from one item to the next and never reset; the gradient is cut at the first byte of each item (back-propagation through the item only).
- One update after each item's end mark, on the mean cross-entropy of the next-byte predictions of that item's bytes; Adam with rate 0.001 and betas 0.9 and 0.999, no weight decay; initialisation by the library's default from the seed.
- Its attempt for position t is its most probable byte before the update that uses position t's item; so, like the tissue's, every scored attempt precedes the use of its byte.

**Units of the comparison.** Items read (about 16,360 in step 1's positions and 8,160 in step 2's; section 19, E.2) and multiply-adds: 94,208 a byte forward, 282,624 a byte with the backward pass, about 1.1 · 10¹¹ over the 393,216 bytes (section 19, E.1, E.2). The tissue's operations a byte are given beside them (section 10; ⟦A: operations a byte at the end of stage A⟧).

**Use in the reading.** For each class and stretch, the comparator's share right is the ceiling of what learning by back-propagation reaches on this stream with this much exposure; and the number of items it needs to reach 0.5 in each class of K. The tissue's result is set beside both, in items read and in multiply-adds. The comparator receives the arriving byte as its target, which is more than yes or no; the report says so. It is **no part of the tissue's program**, is run on its own branch, and only on the coordinator's release (roadmap, condition 5).

## 10. Size, cost and budget

**Size.** As at step 1: 2,048 cells, 32 steps a byte; the connections a cell holds at the end of step 1 ⟦B: mean connections a cell at life position 262,143, per seed⟧. Step 2 adds only the setting of one or two lines at φ = 0 and the scoring, which are negligible against the tissue's arithmetic.

**Bytes.** Form (a): six step-2 tissues of 131,072 bytes, 786,432 bytes. Form (b): its step-1 phase of 262,144 bytes and two step-2 tissues of 131,072, 524,288 bytes. In all 1,310,720 bytes (section 19, F.1).

**Cost, with its basis.** At the rates of the step-1 brief (its section 9, 49.1, 88.4 and 209.9 ms a byte at 12, 32 and 93.8 connections a cell with makings): 17.9, 32.2 and 76.4 CPU-hours in all (section 19, F.2). The rate is replaced at release by stage A's measured rate at the end of its stream, ⟦A: CPU seconds a byte of the first-form tissue over bytes 12,288 to 16,383⟧, and by stage B's ⟦B: CPU seconds a byte over the last 8,192 bytes of the continued arm⟧ where it exists; the forecast is 1,310,720 bytes times the larger rate times 1.25 = ⟦B: forecast⟧.

**Budget.** At most 80 CPU-hours (the highest figure above, rounded up), or the ceiling the coordinator sets at release from the forecast. Three processes at a time; about 25 to 35 hours of wall time at the upper rate.

**Files.** Emissions and feedback per position, int16 and int8: about 0.4 MB a step-2 tissue (section 19, F.4 gives 0.26 MB for the emissions alone); summaries under 1 MB; final states 1.3 to 8.7 MB a tissue (step-1 brief, section 9). Under 100 MB in all. No records of cells' outputs are kept.

## 11. Stopping rule, order of opening, and the report when the step is not reached

### 11.1 Stopping

- Any check fails: stop and ask.
- The forecast exceeds the ceiling: ask before starting; at the end of P2 of each step-2 tissue, re-forecast from its own rate; if the sum exceeds the ceiling, stop at the next saved state and ask.
- A tissue is never stopped early because of its measures.

### 11.2 Order of the runs

1. Checks 1 to 10.
2. Form (a), told and never told, seed 73129, to the end of P1 (16,384 bytes); the worker sends the message of section 15 with R(c, P1) for both, the counts of yes and no, and the share of positions with a byte emitted, and goes on without waiting.
3. The rest of form (a) on seed 73129, the other two seeds, and form (b)'s step-1 phase, three processes at a time; then form (b)'s two step-2 tissues.

### 11.3 Order of opening for step 2 (definition, section 15), one change at a time

None is run under this brief; each needs its own exact form where marked and the coordinator's release. Each opening named "new birth" changes the tissue from birth and so needs its own step-1 phase of 262,144 bytes before its step 2.

| Order | Choice | The one change | Note |
|---|---|---|---|
| 1 | C56 | 128 feedback cells (64 yes, 64 no) | New birth; the fifth reading counts 44 to 81 output-end cells taking a yes-cell at birth, against 20 to 32 |
| 2 | C41 | Receiving factor m_i · ε_i in formula 7 | The definition does not define ε_i; needs an exact form from the reviewer |
| 3 | C33 | Competition by inhibitory cells in place of the fixed conversion | Needs an exact form from the reviewer |
| 4 | C40 | Feedback lines directly to output-end cells, in addition to the feedback cells | Needs the reviewer to say which output-end cells receive which line; a yes or no given alike to all is the form withdrawn by section 16 |
| 5 | C15 | η/16 at output-end cells | Exact; no new birth |
| 6 | C29 | Output-end cells exempt from formula 9 (θ stays 0 at them) | Exact; no new birth |
| 7 | C16 | Traces with τ = 64 | Exact; no new birth |
| 8 | C56 | Feedback cells with ordinary sources; some encoder cells receiving the line directly; kinds at random | New birth |
| 9 | C37 | Per-line feedback (sentence 111) | Needs a change of decision D3 by Mr. Xu; not opened without it |
| 10 | — | The order for step 1 | |

### 11.4 What is reported when the step is not reached

The sentence "Step 2 was not reached within its budget", with the forms and seeds run; for every told tissue R(c, P) for every class and stretch, against the never-told tissue, the frequency emitter, the tables and, if run, the comparator; the conditions of section 7.3 that failed; the counts of yes and no per stretch; the share of positions with a byte emitted; the numbers of output-end cells taking yes-cells and no-cells as sources at the start and at the end of step 2 (E2); and the outcome of the registered expectations. Mr. Xu decides whether the route continues, changes or stops there (roadmap, section 5, condition 8).

## 12. The program, saving and resuming, receipts and files

- `TISSUE_STEP2/tissue_step2.py`: imports the step-1 program `TISSUE_STEP1/tissue_step1.py` and does not change it; adds the world's comparison and the setting of the feedback and emission lines (section 4.3), the birth of form (b) (section 4.4), and the records of section 6. `python3 tissue_step2.py run <a|b> <told|never> <seed> [--from-state PATH] [--phase step1|step2] [--bytes N] [--out DIR] [--ckpt-bytes N] [--resume] [--die-after-byte N]` and `python3 tissue_step2.py check <1..10>`. It is committed before the first check is run.
- `TISSUE_STEP2/step2_read.py`: the measure of section 7 and the tables of section 3.3 on each stretch; `step2_tables.py`: the report's tables from the receipts.
- `TISSUE_STEP2/runs/<form>_<told|never>_<seed>/`: `receipt.json`, `attempts.npz`, `summary.json`, `final_state.npz`, `log.txt`; for form (b)'s step-1 phase, `runs/b_step1_<seed>/`.
- Receipts record, beyond the step-1 brief's entries (its section 13.2): the SHA-256 of the state started from; the form; told or never told; the life positions read.
- Saving and resuming as in the step-1 brief (section 13.3), a saved state every 16,384 bytes and at the end of each stretch.
- Committed: the programs, receipts, attempts, summaries, final states, logs, check receipts and the report `TISSUE_STEP2/STEP2_REPORT.md`.

## 13. What the report must state

1. The claim at the level earned (section 3, item 1), or the sentence of section 11.4.
2. Which step-1 tissue was continued, from which state (its SHA-256), and the conditions of release met.
3. The numbers fixed by us and the form of every choice used: those of the step-1 brief (its section 5.7), with the arm continued; form (a) or (b); 131,072 bytes; the levels 0.5, 0.2 and 0.5333; the stretches of 16,384 positions.
4. For every told and never-told tissue: R(c, P) with standard errors for every class and stretch; the counts of yes and no; the share of positions emitted; the measures of section 6, items 3 and 4; the readings of section 7.4.
5. The countermodels of section 3.3 on every stretch; the comparator's shares and items to 0.5, if run, with its units.
6. Form (b) beside form (a), on seed 73129: the same measures and the difference.
7. That the output end is reached through feedback cells only (sentence 91), that the yes or no passed on alike cannot point the right way for an active cell that did not emit (sentence 94), and the numbers of output-end cells reached from yes-cells and no-cells at the start and end of step 2 (E2).
8. The outcome of each registered expectation of section 16.
9. CPU time per tissue against the forecast; the reading of condition 3 that the coordinator confirmed (section 5.3).
10. The list of committed files and the commit the work started from.

## 14. Rules for the worker

As in the step-1 brief, section 14, with `TISSUE_STEP2/` in place of `TISSUE_STEP1/`. In addition: the step-1 program and the step-1 saved states are read, never changed; the worker does not start the comparator, which is a separate release.

## 15. Reporting

A short message with `send_message` of the claude-code-remote server to session `session_01YBMaMvtGtQvDFcgppgThrZ` at: the checks passed (or which failed), with the forecast; the end of P1 of form (a) on seed 73129 (section 11.2); all form (a) tissues finished, with the conditions of section 7.3 per seed; form (b) finished; the report committed. Each message gives the branch, the commit and the main figures in at most ten lines.

## 16. Registered expectations (before any run)

The fifth reading (H2): at birth only 20 to 32 of the 256 output-end cells take a yes-cell and 15 to 39 a no-cell; a yes or no moves the emitting cell the right way, and other active output-end cells the same way, which is wrong for the cell of the right byte; at a single position the feedback's change is of the size of the change formula 7 makes in the rest of the window, so the feedback acts only in expectation over many positions; learning begins mainly by elimination. The recomputed example agrees: after a no, O's strength from u falls by 0.0050 over steps 1 to 3 but ends the window 0.0004 higher than it began; after a yes, it ends 0.0080 higher (section 19, C.1, C.2). Revision 5 adds that output-end cells are excitatory and their emission needs an activity above 1/2 at the reading step, against a target of 1/8 for their mean activity (sentences 55, 75). H3 is removed by E4 (feedback cells have their line only); H4 by the restriction of E3 (sentence 89). The expectations below are the brief writer's; each is scored as held or not held.

| No. | Tissue | Measure | Expectation | Chance it holds |
|---|---|---|---|---|
| Y1 | All step-2 tissues | Share of positions with a byte emitted in P8 | Below 0.1: an output-end cell must exceed 1/2 at the reading step while its threshold holds its mean near 1/8; ⟦A: the share of positions emitted in stage A's last stretch⟧ is the starting point | 0.6 |
| Y2 | Form (a), told | Number of yes given in P8 | Below 5 per cent of positions (few emissions; most of them wrong) | 0.7 |
| Y3 | Form (a), told | R(c, P8) against R_never(c, P8) at the three classes K | Higher by at least 0.02 in at least one class on at least two seeds (the feedback acts, in expectation) | 0.4 |
| Y4 | Form (a), told | Step 2 reached on all three seeds | Reached | 0.05 |
| Y5 | Form (b), told | Step 2 reached on seed 73129 | Reached | 0.05 |
| Y6 | Form (a), never told | R(c, P8) at K | Below 0.05 in every class | 0.8 |
| Y7 | Form (a), told | Mean activity at the reading step of the output-end cells that take a no-cell, P8, against the never-told tissue | Lower (elimination by the no) | 0.7 |
| Y8 | Form (a), told | Output-end cells taking a yes-cell at the end of step 2, against its start | Not fewer (E10 gives up a connection only below 1/128, and a connection from a yes-cell that is silent is not changed) | 0.8 |
| Y9 | Form (a), told | R(R1, P8) and R(R2, P8) | Below the table of the last 3 bytes | 0.95 |
| Y10 | — | Checks 1 to 10 pass at the first attempt | Pass | 0.5 |

## 17. What step 2 cannot show

- Anything about answers, keeping or generalisation (steps 4 to 7); the answer bytes are scored only to be reported.
- Relay inward from the ends (step 3): cells more than one connection from a feedback cell receive the feedback only mixed with context (sentence 103).
- Which reading of section 15 Mr. Xu intends: both are run; the result does not decide it.
- Whether the input end alone can be corrected, since the second tissue of the roadmap is not run (section 4.2).
- Anything beyond three seeds (form (b): one), one stream and the arm continued.

## 18. What depends on stage A and stage B of step 1

| Mark | Where | Filled from |
|---|---|---|
| ⟦A: CPU seconds a byte of the first-form tissue over bytes 12,288 to 16,383⟧ | Section 10 | Stage A's report, section 11.4 of the step-1 brief |
| ⟦A: operations a byte at the end of stage A⟧ | Section 9 | Stage A's mean connections a cell at byte 16,383, by the step-1 brief's section 9 count |
| ⟦A: the share of positions emitted in stage A's last stretch⟧ | Section 16, Y1 | Stage A's report, section 11.2, item 5 |
| ⟦B: mean connections a cell at life position 262,143, per seed⟧ | Section 10 | Stage B's receipts |
| ⟦B: CPU seconds a byte over the last 8,192 bytes of the continued arm⟧ and ⟦B: forecast⟧ | Section 10 | Stage B's receipts |
| ⟦B: three SHA-256 values⟧ | Section 5.2 | The final states of stage B |
| The arm continued | Sections 4.1, 5.1 | Stage B's report |

Nothing else in the brief depends on stage A or B; the first form of step 2 does not depend on stage A's outcome (decision 56). If stage A's outcome is 2 (cells at two and four connections silent or constant), stage B is not run, and this brief is not released (section 5.1).

## 19. Output of `check_step2_brief.py`

`python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/check_step2_brief.py`, run at 01:05 UTC on 19 November 2026 (6 seconds):

```
==============================================================================================================
A. THE STREAM OF STEPS 1 AND 2
A.1 seed 73129: SHA-256 of positions 0 to 262143 7772341c01ab0954 (step 1, as in the brief of step 1); of positions 262144 to 393215 7ea85f8bf128b75a807402b51299716dc92402260c2bd80e53fd9c8928808266
    first 32 bytes of step 2: b'74<HZUEOH88=74\np49+GJYSJDOTC23=7'; class of position 262144: A1
A.1 seed 93683: SHA-256 of positions 0 to 262143 a6363a43f56c1cdc (step 1, as in the brief of step 1); of positions 262144 to 393215 70ca5578ce500e33c4a2a7954a96b93754bddc87ac3c4e8c1c819d6089bb61ce
    first 32 bytes of step 2: b'PYF18=37\nv46>X08=46\nv67>PR64=67\n'; class of position 262144: LET
A.1 seed 116089: SHA-256 of positions 0 to 262143 0473b978c0a8db0d (step 1, as in the brief of step 1); of positions 262144 to 393215 a3ef9af083f568fd65beb9d97764f75b5689b52a68b675a3e614a51c3c4f87de
    first 32 bytes of step 2: b'QFO19=61\nq67-UXH05=62\nw87<SAJDDP'; class of position 262144: LET
A.2 seed 73129, step-2 positions by class of the byte that arrives (count, share): TASK 8161 (0.0623), A1 8162 (0.0623), A2 8162 (0.0623), SIGN 8162 (0.0623), LET 49459 (0.3773), B1 8161 (0.0623), B2 8161 (0.0623), EQ 8161 (0.0623), R1 8161 (0.0623), R2 8161 (0.0623), END 8161 (0.0623)
A.3 seed 73129, the last stretch (positions 376,832 to 393,215), by class: TASK 1022, A1 1022, A2 1022, SIGN 1022, LET 6169, B1 1021, B2 1021, EQ 1021, R1 1021, R2 1021, END 1022
==============================================================================================================
B. TABLES OF THE LAST n BYTES, FITTED ON POSITIONS 0 TO 262,143, SCORED ON THE STEP-2 POSITIONS (share right, by class)
B.1 seed 73129: the most frequent byte of positions 0 to 262143 is b'\n' (share 0.0624)
    class                                  TASK     A1     A2   SIGN    LET     B1     B2     EQ     R1     R2    END
    frequency emitter (step 2)            0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  1.000
    frequency emitter (last stretch)      0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  1.000
    table of the last 1 (step 2)          0.250  0.112  0.000  0.000  0.039  0.000  0.000  0.497  0.117  0.000  0.499
    table of the last 1 (last stretch)    0.263  0.114  0.000  0.000  0.036  0.000  0.000  0.492  0.136  0.000  0.518
    table of the last 2 (step 2)          0.248  0.112  0.111  0.000  0.036  0.004  0.100  0.503  0.122  0.108  0.517
    table of the last 2 (last stretch)    0.261  0.114  0.120  0.000  0.037  0.005  0.110  0.530  0.130  0.100  0.533
    table of the last 3 (step 2)          0.253  0.111  0.111  1.000  0.027  0.039  0.097  0.995  0.335  0.326  1.000
    table of the last 3 (last stretch)    0.248  0.116  0.120  1.000  0.029  0.035  0.087  0.994  0.304  0.303  1.000
    classes where the table of the last 3 bytes scores at least 0.9 on the step-2 positions: ['SIGN', 'EQ', 'END']
    table of the last 3, step 2, all positions: 0.2758
    a constant emitter (one byte value at every position) reaches 0.5 in at most 1 of the 3 classes in the last stretch
    the rule of section 7.3 in the last stretch, against a never-emitting never-told tissue (frequency emitter's mean over K 0.3333, so condition 3 asks 0.5333): frequency emitter: not met; table of the last 1: not met; table of the last 2: not met; table of the last 3: met
    class SIGN: byte values that arrive there: b'<' 2061, b'+' 2043, b'>' 2037, b'-' 2021
    class EQ: byte values that arrive there: b'=' 8161
    class END: byte values that arrive there: b'\n' 8161
B.1 seed 93683: the most frequent byte of positions 0 to 262143 is b'\n' (share 0.0624)
    class                                  TASK     A1     A2   SIGN    LET     B1     B2     EQ     R1     R2    END
    frequency emitter (step 2)            0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  1.000
    frequency emitter (last stretch)      0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  1.000
    table of the last 1 (step 2)          0.247  0.118  0.000  0.000  0.039  0.000  0.000  0.506  0.118  0.000  0.503
    table of the last 1 (last stretch)    0.223  0.118  0.000  0.000  0.042  0.000  0.000  0.512  0.133  0.000  0.523
    table of the last 2 (step 2)          0.243  0.118  0.114  0.000  0.037  0.005  0.098  0.541  0.117  0.107  0.488
    table of the last 2 (last stretch)    0.242  0.118  0.131  0.000  0.037  0.006  0.102  0.542  0.121  0.107  0.519
    table of the last 3 (step 2)          0.248  0.113  0.114  1.000  0.026  0.044  0.094  0.998  0.340  0.331  1.000
    table of the last 3 (last stretch)    0.248  0.114  0.131  1.000  0.025  0.041  0.082  1.000  0.332  0.339  1.000
    classes where the table of the last 3 bytes scores at least 0.9 on the step-2 positions: ['SIGN', 'EQ', 'END']
    table of the last 3, step 2, all positions: 0.2776
    a constant emitter (one byte value at every position) reaches 0.5 in at most 1 of the 3 classes in the last stretch
    the rule of section 7.3 in the last stretch, against a never-emitting never-told tissue (frequency emitter's mean over K 0.3333, so condition 3 asks 0.5333): frequency emitter: not met; table of the last 1: not met; table of the last 2: not met; table of the last 3: met
B.1 seed 116089: the most frequent byte of positions 0 to 262143 is b'\n' (share 0.0625)
    class                                  TASK     A1     A2   SIGN    LET     B1     B2     EQ     R1     R2    END
    frequency emitter (step 2)            0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  1.000
    frequency emitter (last stretch)      0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  1.000
    table of the last 1 (step 2)          0.244  0.119  0.000  0.000  0.038  0.000  0.000  0.602  0.113  0.000  0.410
    table of the last 1 (last stretch)    0.244  0.125  0.000  0.000  0.036  0.000  0.000  0.570  0.121  0.000  0.415
    table of the last 2 (step 2)          0.252  0.119  0.114  0.000  0.037  0.004  0.101  0.511  0.122  0.109  0.513
    table of the last 2 (last stretch)    0.268  0.125  0.109  0.000  0.040  0.005  0.093  0.504  0.121  0.106  0.524
    table of the last 3 (step 2)          0.249  0.111  0.114  1.000  0.027  0.042  0.096  0.998  0.344  0.332  1.000
    table of the last 3 (last stretch)    0.242  0.103  0.109  1.000  0.029  0.059  0.089  1.000  0.336  0.322  1.000
    classes where the table of the last 3 bytes scores at least 0.9 on the step-2 positions: ['SIGN', 'EQ', 'END']
    table of the last 3, step 2, all positions: 0.2771
    a constant emitter (one byte value at every position) reaches 0.5 in at most 1 of the 3 classes in the last stretch
    the rule of section 7.3 in the last stretch, against a never-emitting never-told tissue (frequency emitter's mean over K 0.3333, so condition 3 asks 0.5333): frequency emitter: not met; table of the last 1: not met; table of the last 2: not met; table of the last 3: met
B.2 the classes that can be known from the last few bytes are the same on the three seeds: True (['SIGN', 'EQ', 'END'])
B.3 positions of those classes in the last stretch: [3065, 3078, 3042] (seeds in order); of each class about [1022, 1021, 1022] (seed 73129)
B.4 standard error of a share 0.5 over the smallest such class (1014 positions): 0.0157
B.4 standard error of a share 0.3 over the smallest such class (1014 positions): 0.0144
B.4 standard error of a share 0.1 over the smallest such class (1014 positions): 0.0094
==============================================================================================================
C. REVISION 5, SECTION 13, PART 1, WITH A NO (AS PUBLISHED) AND WITH A YES
C.1 no: step | F x | E A, m, x | I A, m, x | O A, m, x
     0 | 0.5000 0.1250 0.0750 0.0625 0.0462 -0.0737 0.0231 0.3125 0.0000 0.4062
     1 | 0.2500 0.2500 0.1976 0.1467 0.0625 -0.0552 0.0186 0.4375 0.1250 0.3697
     2 | 0.1250 0.1874 0.1288 0.1571 0.1047 -0.0113 0.0379 0.3730 0.0566 0.3183
     3 | 0.0625 0.1559 0.0934 0.1464 0.1099 -0.0057 0.0504 0.3409 0.0228 0.2761
     8 | 0.0020 0.1252 0.0518 0.1081 0.0868 -0.0263 0.0449 0.3105 -0.0081 0.2042
    16 | 0.0000 0.1238 0.0391 0.1051 0.0840 -0.0227 0.0443 0.0302 -0.2864 0.0555
    24 | 0.0000 0.1235 0.0300 0.1060 0.0845 -0.0172 0.0496 0.0314 -0.2213 0.0000
    31 | 0.0000 0.1233 0.0239 0.1069 0.0850 -0.0133 0.0541 0.0314 -0.1772 0.0000
    change of O's strength from u at steps 1, 2 and 3: -0.0032, -0.0013, -0.0005; over the window +0.0004; final strengths {"F": {}, "E": {"h": 0.2554, "c": 0.4973, "F": 0.2496}, "I": {"E": 0.5003, "q": 0.251, "g": 0.2495}, "O": {"u": 0.7504, "f": 0.2498, "F": 0.2502}}
C.2 yes: step | F x | E A, m, x | I A, m, x | O A, m, x
     0 | 0.5000 0.1250 0.0750 0.0625 0.0462 -0.0737 0.0231 0.3125 0.0000 0.4062
     1 | 0.2500 -0.0000 -0.0524 0.0217 0.0625 -0.0552 0.0186 0.1875 -0.1250 0.2447
     2 | 0.1250 0.0625 0.0118 0.0332 0.0422 -0.0738 0.0066 0.2520 -0.0566 0.1963
     3 | 0.0625 0.0938 0.0427 0.0548 0.0479 -0.0658 0.0040 0.2838 -0.0230 0.1879
     8 | 0.0020 0.1238 0.0636 0.1045 0.0820 -0.0248 0.0347 0.3142 0.0078 0.2055
    16 | 0.0000 0.1243 0.0497 0.1091 0.0859 -0.0159 0.0477 0.0321 -0.2761 0.0619
    24 | 0.0000 0.1239 0.0382 0.1097 0.0863 -0.0120 0.0529 0.0332 -0.2133 0.0000
    31 | 0.0000 0.1237 0.0304 0.1103 0.0866 -0.0093 0.0571 0.0332 -0.1708 0.0000
    change of O's strength from u at steps 1, 2 and 3: +0.0032, +0.0009, +0.0003; over the window +0.0080; final strengths {"F": {}, "E": {"h": 0.2543, "c": 0.4978, "F": 0.2504}, "I": {"E": 0.5002, "q": 0.2508, "g": 0.2496}, "O": {"u": 0.758, "f": 0.246, "F": 0.2498}}
==============================================================================================================
D. THE BIRTH OF FORM (b): 64 EMISSION FEEDBACK CELLS, DRAWN BY Random(seed + 700000) AFTER THE BIRTH OF STEP 1
D.1 seed 73129: eligible cells 1728; digest of [emission cells, their lines] e063a0b5db615db4; first five cells [1302, 795, 1374, 1237, 347] with lines [17, 63, 101, 117, 174, 226, 228, 243]; x of the cells from 0.002 to 0.831; byte values with at least one emission line 223 of 256
D.1 seed 93683: eligible cells 1728; digest of [emission cells, their lines] f208fad23aa24a00; first five cells [881, 675, 1076, 1518, 1142] with lines [14, 37, 63, 73, 158, 200, 201, 238]; x of the cells from 0.014 to 0.855; byte values with at least one emission line 220 of 256
D.1 seed 116089: eligible cells 1728; digest of [emission cells, their lines] cbfbb16642e2bc60; first five cells [835, 201, 599, 698, 1030] with lines [3, 37, 39, 44, 68, 76, 246, 251]; x of the cells from 0.017 to 0.854; byte values with at least one emission line 233 of 256
==============================================================================================================
E. THE COMPARATOR REQUESTED (a GRU of 128 cells, byte embedding of 32, readout to 256; one update per item)
E.1 multiply-adds a byte: forward 94,208; with the backward pass (taken as twice the forward) 282,624
E.2 seed 73129: items ended in positions 0 to 262143: 16,361; in the step-2 positions: 8,161; multiply-adds of training over all 393,216 bytes: 1.111e+11
E.3 its weights: 103,424
==============================================================================================================
F. COST OF STEP 2 AT THE RATES OF THE BRIEF OF STEP 1 (to be replaced by stage A's measured rate)
F.1 bytes: form (a), told and never told, three seeds: 786,432; form (b), its step-1 phase and told and never told, seed 73129: 524,288; in all 1,310,720
F.2 at 49.1 ms a byte (step-1 brief, section 19, G.1: 12, 32 and 93.8 connections a cell with makings): form (a) 10.7 CPU-hours, form (b) 7.2, in all 17.9
F.2 at 88.4 ms a byte (step-1 brief, section 19, G.1: 12, 32 and 93.8 connections a cell with makings): form (a) 19.3 CPU-hours, form (b) 12.9, in all 32.2
F.2 at 209.9 ms a byte (step-1 brief, section 19, G.1: 12, 32 and 93.8 connections a cell with makings): form (a) 45.9 CPU-hours, form (b) 30.6, in all 76.4
F.3 opening 1 (128 feedback cells; a new birth with its own step-1 phase, told and never told) on one seed: 524,288 bytes, 7.2 to 30.6 CPU-hours
F.4 records: emissions (int16) 0.26 MB a tissue of step 2; the step-2 positions of the eight step-2 tissues 2.1 MB
```

## 20. References

- Matsumoto and Hikosaka (2007), Nature 447: 1111-1115.
- Schultz, Dayan and Montague (1997), Science 275: 1593-1599.
- As cited in the definition. Page numbers are from memory and should be checked before citation.

## History

Revision 1 (19 November 2026, from 01:03 UTC): written from roadmap draft 3, revision 5 of the definition, the fifth reading, the step-1 brief (revision 3) and the coordinator's decisions 48 to 57; for a checker, then for the coordinator after stage A of step 1 has been read.
