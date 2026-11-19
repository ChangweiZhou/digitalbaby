# Third reading of "The tissue defined, revision 2" (18 November 2026, written from 23:06 UTC by the clock of this machine)

## Summary

**Verdict: does not pass.** It can pass after the changes listed below. Most of them are rewordings. Seven are decisions about the first form that cannot be left to the programmer.

1. Mr. Xu's rulings of 23:03 and 23:04 UTC (sections 13 and 14 of `WHAT_A_NEURON_HAS_20261118.md`) withdraw sentence 90 and the narrow reading in sentence 92. They also contradict sentences 20, 83, 103, 126 and 135 as general statements. Section 14's example of richer feedback has no place to enter: the term "signal" is defined only for one byte position and only for a mismatch.
2. Sentences 96, 99 and 104 are marked as following by necessity, but they do not follow. Sentence 96 ignores its own alternative. Sentence 99 depends on the sign of the learning rule (C15), which is never fixed, and the living counterpart cited in sentence 55 has the opposite sign. Sentence 104 conflicts with sentence 95 and with the first form of C10.
3. Under the first-form wiring (64 cells drawn at random out of 512, 16 sources each), every cell that is not at the output end lies one connection from the output end. No cell lies four connections from the input lines. The groups of cells that roadmap steps 1 and 3 measure at those distances are therefore empty, and this can be seen on paper.
4. The first form contradicts itself on return connections. C33 makes output-end cells send to inner cells, and by the definition's own term such a connection is a return connection; C43 says there are none.
5. The first form of the feedback rule (C41) can only weaken strengths. Because a byte is emitted only when exactly one of 256 lines exceeds its level, nearly every position is wrong early in life. The resulting weakening does not tell the right line from the wrong ones, and the activity control (C29) undoes it. On paper, step 2 can begin only through rare positions at which the emission was right.
6. A programmer would still have to make about thirty decisions (section 5). These include the activation function, the formula of the prediction, the sign of C15, the timing of the feedback and the composition of the neighbourhood.
7. The list of fixed numbers leaves out at least eight numbers. The roadmap (section 4, item 8) and Mr. Xu's ruling in section 12 ask that a cell set its own number of connections from step 1. C6 fixes that number at 16, although in the first form a cell could already set it.
8. Marks: 103 sentences pass, 11 fail and 21 are doubtful.

## 1. Terms used in this reading

- *Feedback* is whatever the world gives the tissue about its emission. It is the definition's "signal", widened as section 13 requires. Its *yes-or-no form* is one quantity for each byte position: 1 if the emission differed from the arriving byte and 0 if it did not.
- *Window* means the s steps between the arrival of one byte and the arrival of the next. Steps within a window are numbered 0 to s − 1, and the byte arrives at step 0.
- *Path sign* is the product of the signs (excitatory +1, inhibitory −1) of the cells along a path of connections.
- *Visible on paper* means that the outcome follows from the stated forms and numbers without running anything.

All other terms are the definition's own.

## 2. Findings in order of seriousness

### F1. Sections 13 and 14 have no place in the definition (sentences 18, 20, 21, 53, 83, 89, 90, 92, 103, 126, 129, 130, 134, 135; choices C37 to C41; term "signal")

Mr. Xu: "细胞应该可以得到误差的回馈 … 应该可以容纳其他形式的回答" ("cells should be able to receive the feedback of the error … other forms of answer should be able to be accommodated"). For predicting the next byte the yes-or-no form is accepted. Section 14 gives the right answer, together with the absolute difference between the given answer and the right one, as an example of additional feedback for addition.

What the definition says against this:

- The term "signal" is "what the world gives the tissue when its emission did not match". A "yes" (a match) is not covered, so the yes-or-no form is not stated completely.
- Sentence 90 excludes telling a silent line that it should have fired. Section 13 withdraws that exclusion.
- Sentence 92, choice C40 ("the wider reading … is not used"), the paragraph "Entry at the output end" in section 2.2, and item 3 of section 5 keep "not assumed". This wording is withdrawn.
- Sentences 20, 83, 126 and 135 say that bytes and the right answer reach the tissue only through the input end, and that the feedback never says which byte was right. Two things already contradict this. When the yes-or-no value is 0, the cell of the one active line learns that its byte was right. And section 14 admits the right answer as feedback.
- Sentence 103 says that a cell at distance 1 "does not have the signal itself" and "never has the byte that was right". Under section 13 the first clause is a property of the first form only. The second clause is false even in the first form, because encoder cells lie at distance 1 (finding F3) and they read the right byte.
- Sentence 18 [S] says that a cell other than an encoder cell receives input only from other cells. Sentence 21 gives output-end cells the feedback, which no cell sends. The definition has to say how the two fit together.

**Test case (section 14).**

- *The right answer as bytes.* The definition offers only the input end, where the answer already arrives (sentence 82). For a cell to use the right answer as feedback, the answer must reach the output-end cell of the right byte's line at each position of the answer, as "this line should have been active". Only that cell can read the byte as concerning itself, because a cell knows its line only through being connected to it. This is exactly the form that sentence 90 excludes.
- *The absolute difference.* This is one number for each answer, not for each byte. The world can compute it, because the experimenter knows where answers are (sentence 81). It would be given after the last byte of the answer, as one graded quantity, to output-end cells alike or to any cells.
- *What has to be added:*
  - a term "feedback" with named components;
  - in C37, the forms (d) "a quantity for each output line at each answer position: 1 if the line should have been active" and (e) "a graded quantity for each answer: the absolute difference";
  - in C39, a time for each answer (after its last byte);
  - in C40, entry for each line;
  - in C41, a rule for each line: strengthen the inputs of the cell whose line should have been active, in proportion to the traces of those inputs. The living counterparts are the instructive climbing fibre and the plateau potential of Bittner and colleagues (2017).
  - A trace long enough to span an answer. A three-byte answer with s = 4 needs at least 12 + e steps, which is more than the first value of 8 in C16.
  - For modulo-89 arithmetic, a definition of the difference: circular or plain.
  - Two parts of a cell (C1) are the natural place for line-specific feedback, so that it does not mix with the forward arrivals (see F4).

**Proposed wording.**

- Term: "*Feedback* is what the world gives the tissue about its emission at each position, or about an answer. In the first form it is yes or no for each position (C37 a). Richer forms are listed under C37 and none is excluded (section 13)."
- Sentence 90 [S, section 13]: "Cells can receive the feedback of the error. For the prediction of the next byte the first form is yes or no for each position. Telling each output line whether it should have been active, the right answer, and a magnitude are further forms of C37, to be used where a brief names them (section 14)."
- Sentence 18: add "…and, for a cell that receives feedback, from the world through feedback lines (section 13). A feedback line, like an input line, is a fixed channel and not a cell. Its living counterpart is the climbing fibre or a modulatory projection."
- Sentences 20, 83, 126, 135: "In the first form…". In 126, delete "and never which byte was right" and write: "In the first form the feedback says whether the emission was right; when it was right, this identifies the byte."
- Sentence 103: "In the first form, a cell at distance 1 that is not an encoder cell does not receive the feedback directly. Encoder cells read every arriving byte, including the right one."
- Sentences 92 and C40: "Where the feedback enters is choice C40. First form: every output-end cell alike. Further forms: the cell of each line (forms c and d); any cell (section 13)."

### F2. The first-form wiring empties the groups that steps 1 and 3 measure (C7, C30, sentence 16; roadmap steps 1 and 3)

Take C30's first form literally: 64 cells drawn for each cell from among all the others. Add 16 sources drawn from those 64 (C6) and encoder cells that draw from the lines only (C8). The arithmetic is then this.

- One output-end cell takes a given cell as a source with probability 16/511.
- Over 256 output-end cells, the probability that a given cell is the source of none of them is about e^(−8), roughly 0.0003.

A count over three random draws (run with `python3 -I`, standard library) confirms it.

| | Cells counted, three draws |
|---|---|
| Distance 0 from the output end | 256 |
| Distance 1 from the output end | 256 |
| Distance 2 or more from the output end | none |
| One connection from the input lines | 128 |
| Two connections from the input lines | 377 to 381 |
| Three connections from the input lines | 3 to 7 |
| Four connections from the input lines | none |

Step 1 reads cells "one, two and four connections from the input lines". Step 3 reads cells "one, two and four connections from the output end". Under this wiring the groups at four, and at two from the output end, are empty. The same arithmetic also makes "local" (sentence 16, section 9) nearly empty of content: one cell's neighbourhood is one eighth of the tissue, and two steps reach almost every cell.

**Proposed wording for C30's first form.** "Cells are placed on a ring (or a line) in the order: encoder cells, inner cells, output-end cells. A cell's neighbourhood is the 64 cells nearest to it on that ring. Sources are drawn from the neighbourhood." Distances then grow with the number of cells. The brief of step 1 must state the distances that exist for the sizes it runs.

### F3. Sentence 96 does not follow (marked [F])

The sentence offers an alternative ("Otherwise, the conversion … must not respond to that change") and then concludes that more than one step per byte is required, which ignores that alternative. The condition is also not sufficient. The state of a cell is kept over 8 steps (C11), which is longer than the 4 steps per byte (C24). So a change of activity caused at step 1 is still present at the reading step 3, both directly and through recurrent connections. The separation in time that the sentence relies on is therefore not achieved by the first-form numbers.

Other ways of separating emission from relay are also open: feedback that enters at a second part that does not drive the axon, or separate cells that relay.

**Proposed wording, marked [C], as a condition on C24, C11 and C42 together.** "When the feedback is relayed by a change of activity, that change also reaches the output lines. To keep it out of the next emission, either (i) the change has died away before the reading step, which requires s steps per byte to exceed the decline period of the state, or (ii) the conversion ignores it, or (iii) the relay leaves the output-end cells that drive lines unchanged. Which way is used belongs to C42."

### F4. Sentences 99 and 104 do not follow, and they conflict with sentences 55 and 95

**Sentence 99.**

- The sentence assumes that C15 strengthens the active inputs when the arrival is above the prediction (a positive sign). The first form of C15 ("rate × mismatch × trace") never states the sign.
- The living counterpart named in sentence 55 (cerebellum-like structures) weakens the active inputs. Under that sign, a rise in a return arrival lowers A's drive in the context that was wrong. So a rise in activity when wrong would be the corrective sign, and the conclusion of sentence 99 reverses.
- The conclusion also holds only for one path: A excitatory, and B's activity adding to the wrong emission. A quarter of the cells are inhibitory (C3), and what decides is the path sign, which no cell knows.
- Feedback-alignment models (Lillicrap and colleagues, 2016) show that the forward strengths can come into line with fixed return signs. So no single sign of the signal is necessary.

**Sentence 104.** The sentence says that, in a cell with one part, a relayed change acts "only through its mismatch" and changes the cell's output only under conditions. But a relayed change is part of the arrival, and under the first form of C10 activity is computed from the arrival. The change therefore alters the output directly, which is how sentence 95 says that relay works. "Cannot tell" is also too strong: a cell with one part still has synapses that differ, and it can respond to the time at which a change arrives.

**Linked conflict.** The first form of C44 contains "with C10 opened to its second form", while C10's first form is the first form. The first forms of C10 and C44 therefore conflict.

**Proposed wording.**

- Sentence 99, marked [C], or [F] stated as a condition: "Whether the relayed change must lower or raise the activity of the cells it enters depends on the sign of C15 and on the path sign from the receiving cell to the emission. C15 must fix its sign. With the sign that weakens the inputs active when the arrival is above the prediction (sentence 55), a rise is corrective along excitatory paths."
- Sentence 104: "With one part, a relayed change enters the cell's arrival. It changes the cell's activity at once (C10, first form) and its strengths through the mismatch (C15). The cell cannot assign it to a source except by synapse or by timing. With two parts it can be kept apart (sentence 8)."
- C44's first form: "None until step 3. At step 3: …".

### F5. The first form contradicts itself on return connections (C33, C43, term "return connection")

A return connection is "a connection from a cell to a cell at a greater distance from the output end". C33's first form makes output-end cells (distance 0) send to inner cells (distance 1 or more). Those are return connections. C43's first form says "None".

In addition, distance is a property of the whole graph. It changes whenever connections are made or given up (C17, C18), so whether a connection counts as a return connection changes during life. No cell knows its distance.

**Proposed wording.** "C43, first form: no connections other than those of C33. The connections from output-end cells to inner cells are return connections in the sense of the terms, and they carry ordinary output. Distance is computed by the experimenter for reading only; no rule of a cell uses it."

### F6. On paper, step 2 starts only from rare right positions (C29, C33, C34, C38, C41)

The first form of C41 lowers strengths only: the feedback value is 0 or 1, and the rule has no term for being right.

Early in life the emission is rarely right, for three reasons:

- An emission requires that exactly one of the 256 lines exceed 1/2 (C33).
- No emission counts as wrong (C38).
- Only one byte value is right at each position.

So the yes-or-no value is 1 at almost every position. Every output-end cell receives it alike (C40). Each such cell weakens every input whose trace is nonzero, in proportion to its own activity trace. This includes the cell of the right byte whenever it was active, including when two lines were above the level.

The weakening is therefore almost the same for right and wrong lines. Meanwhile C29 drives every output-end cell toward a mean activity of 1/8, including the cells for byte values that never occur in the stream, and so it undoes the weakening. A right line gains an advantage only at positions where it was the single emission and was right. This is not impossible, but the expected learning signal is in proportion to a probability that starts near zero.

C38's second form ("nothing where nothing can be expected") and capability l also cannot be judged by the comparison of sentence 87. A byte arrives at every position, so "no byte" never matches. Capability l therefore needs the experimenter's reading of the stream (sentence 81).

**Proposed first form of C41.** "Each strength changes by rate × (running average of the feedback − feedback) × trace of the input × trace of the cell's own activity." This is the present second form. Being right more often than usual then strengthens, and the change has the living counterpart of the dip and burst of dopamine. For C29 at the output end, either state that the activity control acts on the mean activity below the emission level, or exempt output-end cells. Add to C38: "Whether a position is one at which nothing can be expected is the experimenter's reading (sentence 81)."

### F7. A missing formula and a missing sign make the first form unwritable (C10, C11, C13, C15)

- No formula gives activity from arrival and state (C10, C11). The maximum activity, the transfer function, and the variable that declines over 8 steps are all unstated.
- No formula gives the prediction "from the cell's own state" (C13). If the prediction is a running average of the arrival, it has no strengths, and nothing in C15 makes it improve, which sentence 45 requires.
- The sign of C15 is not given, and it is not said whether inputs from inhibitory cells follow the same rule. With the positive sign, C15 together with strengths that form the arrival increases the mismatch; it does not reduce it.
- Sentence 21 lists the feedback as an arrival, which would enter the total arrival and its prediction (C12). C41 uses it as a separate factor.

**Proposed wording.** State in the first form, as [C] and in one line each:

- the state v(k) = (1 − 1/8)·v(k−1) + arrival(k);
- activity = min(1, max(0, v − θ));
- prediction p(k) = (1 − 1/8)·p(k−1) + (1/8)·arrival(k−1);
- mismatch = arrival − p;
- Δw = −(1/32)·mismatch·trace (the sign of sentence 55), clipped to [0, 1], for excitatory and inhibitory inputs alike;
- "The feedback is not an arrival; it enters only as the factor of C41."

These are examples. Whatever is chosen must be written down.

### F8. The trace arithmetic does not reach step 3 with the first values (sentences 93, 100; C16, C24, C39)

- The emission is read at step 3 of a window. The byte arrives at step 0 of the next window. The feedback is given "at the first step after the byte arrives", which may mean step 0 or step 1, so e = 1 or e = 2.
- For relay to distance d the trace needs 2d + e steps:

| d | e = 1 | e = 2 |
|---|---|---|
| 3 | 7 | 8 |
| 4 | 9 | 10 |

  At d = 4 this exceeds C16's first value of 8 steps, and the next value on the grid is 16.
- C16's entry "at least 2d + e steps for relay" cannot be applied cell by cell, because no cell knows d. It has to be one length for all cells, at least 2·d_max + e.
- The arithmetic of sentence 100 itself is right as a lower bound, and so is sentence 101's 50^2 = 2,500.
- The count in section 4 is right: 20 values, less 5, gives 15.

**Proposed wording.** C39: "the feedback is given at step 1 of the window in which the byte arrives (e = 2)". C16: "one trace length for all cells; 8 at steps 1 and 2; at step 3, at least 2·d_max + e, which is 16 for d_max = 4".

### F9. Silent encoder cells stay silent for life (C7, C17, C21, C32)

An encoder cell draws 16 of the 256 lines. Let A be the number of byte values that occur in the stream. The share of encoder cells that draw no line in use is C(256 − A, 16)/C(256, 16):

| A | Share of encoder cells with no line in use |
|---|---|
| 16 | 0.34 |
| 20 | 0.26 |
| 32 | 0.11 |

Such a cell has arrival 0, prediction 0 and mismatch 0, so its strengths never change. Its strengths therefore never fall below 1/8 of their mean, it gives up no connection (C17), and it stays silent for life.

Step 1 asks that cells one connection from the input lines be "neither silent nor constant". Unless the brief reads this as a share, the criterion fails on paper.

**Proposed wording for C17's first form.** "…or when the input has been silent for the period", so that a connection to a line that never fires is given up. The new source is drawn among the lines (C18 for encoder cells).

### F10. The number of connections is fixed against the roadmap and Mr. Xu's ruling in section 12 (C6)

The roadmap (section 4, item 8) says: "The number of a cell's connections is the cell's own from step 1." C6's first form fixes 16.

A cell can already set this number from quantities it has in the first form. One way is to make a new connection whenever its running activity stays below its target (C29) for the period, and to give up connections by C17 independently. The number then follows from the cell's own activity. One fixed number (16) becomes a starting value.

**Proposed wording for C6's first form.** "Starting number 16; a connection is given up by C17 and made by C18 when the cell's running activity has been below its target for the period of C29; the two are not tied."

### F11. Marks that do not hold

- **Sentence 1 [S].** "Nothing in it learns apart from those cells" is the reviewer's inference from "没有专门的学习者" ("no dedicated learner"). It should be [F] with that reason.
- **Sentence 12 [F].** Dale's principle concerns the identity of the transmitter, not the sign of its effect. The sign depends on the receptor: metabotropic glutamate receptors can inhibit, and GABA depolarises early in development. As a property of a model, a fixed sign is an idealisation. Proposed: [C], "an idealisation of Dale's principle".
- **Sentence 31 [F].** Electrical synapses (gap junctions) conduct in both directions and are textbook. Proposed: add "Electrical synapses, which conduct both ways, exist in living tissue and are left out by choice (C-new)."
- **Sentences 8, 10, 34, 82, 112, 113 [S].** These rest on sections 7 and 9, which report Mr. Xu in the reviewer's wording without the original. Sentences 34 and 112 put that wording in quotation marks as if it were his. Proposed: say "reported in the reviewer's wording" in each, as sentences 8 and 16 do.
- **Sentence 87 [F].** That the comparison is made outside the tissue is Mr. Xu's permission ("你可以给", "you can give"), not a necessity. The reasons given are also weak: after the byte arrives, encoder cells do hold it. Proposed mark: [S] for the permission, [C] for doing it outside.
- **Sentence 88 [F].** "连续" can mean continuing in time or continuous in value. The second reading supports form (b). Proposed: name both readings and take the first as a choice.
- **Sentence 114 [F].** "The tissue defined here contains no such group" is a choice, not a consequence.
- **Sentences 123, 129, 130, 134 [F].** Each rests on "nothing outside the cells acts on the tissue" or on "the only quantity … is the signal". After sections 13 and 14 these need "in the first form", and the list of what the world may give.
- **Sentences 21, 53.** The feedback is listed for output-end cells only. Add "or any cell that receives feedback (C40)".
- **Sentences 55, 100, 101.** These are doubtful for the reasons in F4 and F8. In C43 and sentence 101, "return connections per cell" does not say whether incoming or outgoing connections are meant. The relay count r^d counts outgoing ones, which no first-form rule controls.

### F12. The order of opening across choices is missing (roadmap step 0)

Step 0 requires "the order in which choices are opened when a step fails". The definition gives an order within each choice, but not across choices. Step 1 refers on failure to section 4 of the earlier check, which proposes first forms but does not give an order across choices. Proposed: a short list for each step, for example "Step 1 fails: C30, then C17, then C29, then C11".

## 3. Task (1): sentence by sentence

**Counts.** 135 sentences, of which 25 are marked [S], 66 [F] and 44 [C]. These match the definition's own count, and the table has 44 choices.

| Verdict | Count | Sentences |
|---|---|---|
| Fail | 11 | 18, 20, 83, 90, 92, 96, 99, 103, 104, 126, 135 |
| Doubtful | 21 | 1, 8, 10, 12, 21, 31, 34, 53, 55, 82, 87, 88, 100, 101, 112, 113, 114, 123, 129, 130, 134 |
| Pass | 103 | all others |

Reasons for the failing sentences:

- 18, 20, 83, 90, 92, 103, 126, 135: F1. Sentence 103 also fails on the encoder cells at distance 1 (F2).
- 96: F3.
- 99 and 104: F4.

Reasons for the doubtful sentences are given in F11, F4 and F8.

**Biology.**

- No sentence describes something that a living cell certainly cannot do.
- The reader is unsure whether a cell can compute "the mean of its strengths" (C17). Synaptic scaling shows that a cell senses its total drive, not a mean of its strengths.
- Unsettled: the two-part prediction (C13); error relay in cortex (C42, C44); whether climbing-fibre and dopamine signals are error signals in every condition.
- Sentence 59 holds: glutamate released near a dendrite induces new spines in young cortex.

## 4. Task (2): the feedback against Mr. Xu's words

- **More than his words:**
  - sentence 90 (withdrawn by section 13);
  - sentences 83, 126 and 135 as general statements;
  - sentence 96's "requires";
  - sentence 99's "must";
  - sentence 104's "cannot" and "only".
- **Less than his words:**
  - "整个系统" (the whole system), taken narrowly;
  - "连续" (continuous) read only as continuing in time;
  - the "yes" half of yes or no missing from the term;
  - no place for richer feedback (section 14).
- **C40 against the rule that no signal reaches the whole tissue.** C40 is consistent with it.
  - Section 9 rules on a chemical signal released by a cell. The world's feedback is not such a signal.
  - Section 13 rules that cells may receive the feedback.
  - "No signal of any kind reaches every cell" (section 9, last sentence) is the reviewer's generalisation, and section 13 overtakes it as far as feedback is concerned.
  - C40 reaches 256 of 512 cells, which is not every cell.
  - Proposed: the definition should say this explicitly, and should restrict sentence 114 to signals released by cells.
- **Does the definition keep a place for richer feedback without a rewrite?** No. C37 has a graded form (b) and a form for each line (c). But (c) is cut back by sentence 90, the term covers only a mismatch at one position, and timing (C39) and use (C41) exist only for one position. F1 lists what must be added.
- **Is the yes-or-no first form stated completely?** No. The open points are:
  - the value at a right position must be part of the term;
  - the step at which the feedback is given (e);
  - whether the feedback is an arrival or a factor;
  - the sign convention linking it to C15.

## 5. Task (3): one byte by hand in the first form

Byte t arrives at step 0 of its window. Steps are 0 to 3. All cells compute at once from the outputs of the previous step.

- **Step 0.**
  - The line of value b_t becomes active. The definition does not say whether the line stays active for all four steps or for one.
  - The world compares b_t with the emission read at step 3 of the previous window and forms the yes-or-no value.
  - Encoder cells: each sums its 16 line strengths. Only cells with a line equal to b_t have an arrival.
  - Inner and output-end cells: arrivals from the previous window's outputs only.
  - Every cell forms its mismatch (no formula, F7) and applies C15 (no sign, F7).
- **Step 1.**
  - Encoder outputs reach inner and output-end cells. By F2, nearly all of these are one connection from the encoder.
  - The feedback is given to all 256 output-end cells (C39, if "first step after" means step 1).
  - Each output-end cell applies C41 with traces that now also contain step-0 activity. Each also applies C15. The order of the two changes, and whether the feedback is part of the arrival, are not stated (F7).
  - Output-end cells send to inner cells. These are return connections that C43 says do not exist (F5).
- **Step 2.**
  - Output-end cells receive information about b_t through paths of two connections.
  - C29 moves thresholds. C17 counts periods (whether consecutive or averaged is not stated).
- **Step 3.**
  - The conversion counts the lines above 1/2. Exactly one gives the emission for position t + 1; none or several gives no emission, which counts as wrong (C38).
  - No cell has seen b_{t+1}, so nothing leaks from the future. Step 2 is not trivially passed.
- **Delays.**
  - The emission sees b_t only through paths of three connections or fewer.
  - The information about b_t is held in the state, which declines over 8 steps (two bytes). It reaches eight bytes back only through recurrence.
  - With e = 2, the traces of C41 cover the emission.
- **Places where the definition is silent or in conflict:** F2, F5, F7 and F8, and the timing of the line.
- **Quantities that no cell has:**
  - d in C16 (F8);
  - distance in C43 (F5);
  - "the cell's mean strength" in C17 (the reader is unsure).
- **Step 1.** Not trivially passed. It is impossible as worded for cells four connections from the input lines (F2), and it fails for silent encoder cells (F9).
- **Step 2.** Not impossible, but it can start only from rare right positions (F6).

## 6. Task (4): decisions the programmer would still have to make

1. Transfer function, maximum activity, and how the state and the arrival combine (C10, C11).
2. Formula of the prediction (C13).
3. Sign of C15, and whether it applies to inhibitory inputs and to output-end cells besides C41.
4. Order within a step: activity, then changes, or the reverse.
5. Whether a line is active for one step of the window or for all of them, and its value.
6. The step at which the feedback is given (e = 1 or 2).
7. Whether the feedback is an arrival or a factor.
8. Order of the C15 and C41 changes on the same strength.
9. The form of the trace: exponential with a period of 8, or linear to zero.
10. Which kinds of cells may lie in a neighbourhood, and whether neighbourhoods are symmetric.
11. Which kinds of cells output-end cells send to (C33 names inner cells only).
12. Whether inhibitory cells are spread evenly over the three kinds, and whether output-end cells may be inhibitory.
13. The encoder cells' "neighbourhood" of lines (C7, C30).
14. Drawing with or without replacement; connections of a cell to itself; double connections.
15. C17: whether "for a period" means consecutive steps or an average, and whether "mean strength" includes the connection in question.
16. The starting strength of a new connection.
17. The new source for an encoder cell (C18 names "a cell").
18. C29: the rate at which the threshold moves, its starting value, and whether the target is a mean of the activity or a share of time above a level.
19. The maximum activity behind "1/2 of the maximum" (C33).
20. A reading at the moment the next byte arrives: whether "the last step before" is step 3 of the window or step 0 of the next.
21. Whether distance is computed once or during life.
22. Whether return connections are counted within the 16 of C6, and whether they are incoming or outgoing.
23. C41's activity trace: the same length as the input trace or not.
24. Whether C29 applies to output-end cells.
25. The value of the feedback at positions where an emission of "no byte" was right. No such position can occur under sentence 87 (F6).
26. The random seed and the generator.
27. The value at which a threshold starts, and whether the state may fall below zero.
28. For section 14: the definition of the absolute difference modulo 89, and the time at which it is given.

## 7. Task (5): the numbers fixed by us

The list in section 4 of the definition is not complete. Missing:

- the maximum activity (C33 refers to it);
- the activity of an active input line;
- the decline period of the prediction;
- the rate at which the threshold moves, and its starting value (C29);
- the starting strength of a new connection;
- the step of the window at which the feedback is given (e);
- the length of the activity trace in C41;
- the shape of the transfer function, if it has a gain or slope.

The number of output-end cells is listed, but the 128 encoder and 128 inner cells could also be read as a ratio of kinds; that is a matter of wording only.

Numbers that a cell could set itself in the first form already, from quantities it holds:

- the number of incoming connections, 16 (F10);
- the starting strength of a new connection (the cell's own mean strength);
- the period of C17, which can be identified with the cell's own averaging period of C29, so that two numbers become one.

The level of 1/8 in C17 is already relative to the cell. The other numbers (rate, retention, trace length, target activity, emission level, neighbourhood, shares, steps per byte) need a further rule before a cell can set them, and the definition rightly marks them "possibly" or "no".

## References

Bittner and colleagues (2017) and Lillicrap and colleagues (2016) are as cited in the definition. The remaining facts are textbook neuroscience: gap junctions, receptor-dependent sign, synaptic scaling and new spine growth.
