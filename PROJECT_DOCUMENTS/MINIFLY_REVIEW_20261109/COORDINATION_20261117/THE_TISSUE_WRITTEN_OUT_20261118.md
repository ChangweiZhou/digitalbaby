# The tissue written out from the owner's statement (18 November 2026, 22:25 UTC)

**Note by the reviewer (18 November 2026, about 22:35 UTC), not part of the agent's text.** This document was written by an agent from Mr. Xu's statement alone and has not yet been checked by a second reader. Two things have changed since it was written. First, the correction that the chemical signal is local came from Mr. Xu himself and is now section 9 of `WHAT_A_NEURON_HAS_20261118.md`; the words "relayed by the coordinator" below mean the reviewer. Second, Mr. Xu ruled at about 22:25 UTC (section 10 of the same file) that there is an input end and an output end and that both learn; gap 3 of section 3 below ("the tissue acts on nothing outside itself") is answered by that ruling, and the description is to be extended by the output end. The document is step 0 of `ROADMAP_TO_END_TO_END_PERSISTENT_LEARNING_20261118.md`.

This document describes a tissue of nerve cells using only the owner's (Mr. Xu's) statement of 18 November 2026, 22:15 UTC, and his further rulings. It marks every place where the statement leaves something open. It contains no program and no result. Its sources are:

- the statement in section 8 of `WHAT_A_NEURON_HAS_20261118.md` (Chinese original and translation); quotations below use the translation;
- the rulings in section 7 of the same file (the gated cell, separate parts of a cell, the right answer shown as input after the learner's own attempt, and a chemical signal released by a cell);
- the rulings given with the task: the fewer numbers fixed by the experimenters the better; a number that must be fixed for now is varied on a grid of powers of two (2, 4, 8, ..., 128); long-term memory is to form in the course of learning and not by an act of a program; the input is a stream of bytes in which nothing tells a cell where an item begins or ends;
- a correction relayed by the coordinator at about 22:21 UTC on 18 November 2026, which is not yet in the owner's file: the chemical signal is local. A cell may release a chemical signal into a region near it; the cells of that region receive it alike; it does not reach the whole tissue. A cell acts and adjusts only within its own neighbourhood. Where this correction differs from section 7 of the owner's file (which speaks of "large numbers of other cells"), this document follows the correction.

Sections 1 to 6 of `WHAT_A_NEURON_HAS_20261118.md` are a reviewer's earlier draft. They are used here only to name possibilities, and nothing in them is marked [S].

**Marks.** [S]: the sentence is in the statement or the rulings; the words it rests on are quoted. [F]: the sentence follows from the statement by necessity; the reason is given. [C]: the statement leaves the matter open and the experimenters would have to choose; the choice is numbered C1, C2, ... and listed again in section 2.

**Terms.** A *cell outside the encoder* is any nerve cell of the tissue that is not a cell of the byte-level encoder. An *encoder cell* is a cell of the byte-level encoder. The *arrival* at a cell is what reaches that cell from other cells at one moment. A *prediction* is a quantity that a cell holds before an arrival and that stands for what the cell will receive. The *mismatch* is the difference between an arrival and the prediction of it. A *step* is one exchange of outputs among cells. A *strength* is a number attached to a connection that scales the effect of the sending cell's output on the receiving cell; whether strengths exist at all is choice C5. The *neighbourhood* of a cell is the set of cells within which, by the relayed correction, the cell acts and adjusts.

## 1. The tissue

### 1.1 What a cell is and holds

1. [S] The tissue consists of nerve cells, and nothing in it learns apart from those cells: "There is no separate 'learner'."
2. [S] There are two kinds of cell, encoder cells and cells outside the encoder: "Apart from the byte-level encoder, all that any nerve cell can do is ...".
3. [S] A cell outside the encoder does three things and no more: it receives input from other cells, it sends an output, and it adjusts its connections ("all that any nerve cell can do is to receive input from other cells, to send, from that input and its present state, an output to the cells it is connected to, and to keep adjusting which cells it is connected to").
4. [S] A cell holds a state: the statement speaks of "its present state" and of "the state it has stored".
5. [F] The stored state is the only record of the past that a cell has, because "it can read only the state it has stored and the new information that reaches it through other cells".
6. [S] The state is small compared with the stream: "a million bytes cannot be kept as a window".
7. [S] A cell may have separate parts on which inputs from different sources arrive and are treated differently (section 7, point 1: "separate parts of a cell that treat inputs from different sources differently may be used").
8. [C] Whether a cell has one part or several, and how many, is choice C1; the ruling permits separate parts but does not require them.
9. [S] A cell may be gated (section 7, point 1: "The gated cell may stay").
10. [C] Whether a cell is gated, and what its gates are computed from, is choice C2; the ruling permits a gate but does not require one, and a gate computed from anything other than the cell's own arrivals and state would go beyond sentence 3.
11. [C] Whether each cell has a fixed sign of effect on the cells it sends to (excitatory or inhibitory) is choice C3; neither the statement nor the rulings say so.
12. [C] Whether a cell's output may be negative is choice C4; neither the statement nor the rulings say so.
13. [C] Whether a cell keeps its own level of activity within a range by a change of its own threshold, computed from its own recent activity, is choice C29; the statement does not mention it, but a cell could do it from its own state alone.
14. [S] A cell acts and adjusts only within its own neighbourhood (relayed correction: "a cell acts and adjusts only within its own neighbourhood").
15. [C] What a neighbourhood is (an arrangement of cells in a space and a distance, or a fixed set of cells drawn for each cell), and its size, is choice C30; the correction names a neighbourhood but not its form.

### 1.2 What arrives at a cell and from where

16. [S] A cell outside the encoder receives input only from other cells ("receive input from other cells").
17. [S] A cell does not in general read the bytes that have passed ("A cell cannot in general read the bytes that have passed").
18. [F] The bytes therefore enter the tissue only through encoder cells, and a cell outside the encoder receives from the bytes only what encoder cells, or cells that receive from them, send; the encoder is the only exception the statement names.
19. [F] What arrives at a cell is the outputs of the cells connected to it (and, by sentence 52, the chemical signals released in its region); it does not receive the states, connections, strengths or inputs of other cells except so far as their outputs carry them, because the statement gives a cell no other way to receive anything.
20. [S] What arrives differs from cell to cell: "So the new information differs from cell to cell".
21. [S] What arrives may result from earlier bytes: "the new information that arrives from other cells, which may be the result of processing bytes read earlier".
22. [F] A cell's sources lie within its neighbourhood, by sentence 14.
23. [C] Whether a connection has a graded strength or is simply present or absent is choice C5; the statement names only "which cells it is connected to", which in its strict reading is a set of cells without strengths.
24. [C] How many incoming connections a cell has, and whether that number is held fixed, is choice C6.
25. [C] Which cells a cell is connected to at the start of its life is choice C7.
26. [C] Whether encoder cells also receive input from cells outside the encoder is choice C8; the statement says only that encoder cells read bytes.

### 1.3 What a cell sends and to whom

27. [S] A cell sends an output, computed "from that input and its present state", "to the cells it is connected to".
28. [C] Whether a cell sends the same output to all the cells it is connected to is choice C9; the translation says "an output", but the Chinese does not mark number, so this is not taken as settled.
29. [F] A cell sends nothing backward to the cells from which it receives, other than its ordinary output when one of them is also among the cells it sends to; the statement gives a cell no second kind of electrical output.
30. [F] A loop in which a cell A sends to a cell B and B sends to A is not excluded, and in such a loop whatever B's output carries reaches A as an ordinary input.
31. [C] What a cell's output carries (its activity computed from arrival and state alone, its mismatch, or an activity into which the mismatch enters) is choice C10; the statement says only that the output is computed from input and present state, and the prediction and the mismatch are both parts or functions of these.
32. [S] Beside its output, a cell can release a chemical signal (section 7, point 2, as corrected: a chemical signal released by a cell into a nearby region).

### 1.4 What the state must contain for a cell to predict what arrives

33. [S] What a cell does is to be put as predicting what arrives: "It is to be put as predicting the new information that arrives from other cells".
34. [S] It is not to be put as expecting the next byte: "what a cell does cannot be put as 'expecting the next byte'".
35. [F] A prediction of an arrival must exist before that arrival, so it must be formed from what the cell has before the arrival, namely its stored state and earlier arrivals; otherwise it would be a copy of the arrival and not a prediction.
36. [F] For the prediction to be compared with the arrival, both must be present in the same cell at the same moment; hence the state must hold either the prediction itself or quantities from which the cell can form it at that moment.
37. [F] In this stream some bytes depend on bytes several positions earlier (the two answer digits depend on both pairs of digits, which precede them by the operation byte, the letters and the sign =); an arrival that reflects such a byte can be predicted only if information from the earlier bytes is present, at the moment of prediction, in the cell's state or in its present arrivals, because no cell reads past bytes (sentence 17).
38. [F] The tissue as a whole must therefore keep information about earlier bytes, over at least the distance from the first digits to the answer, in the states of cells and in the outputs they are sending; the statement offers no other place for it.
39. [C] How long a cell's state keeps information (for example, the rate at which its activity or a trace of its inputs declines) is choice C11.

### 1.5 What predicting requires of a cell, quantity by quantity

40. [F] The cell must hold the arrival, since without it there is nothing to predict; the arrival is present at the part of the cell where the predicted inputs arrive.
41. [F] The cell must hold a prediction (sentences 35 and 36).
42. [F] The cell must form some comparison of the two, since otherwise nothing in the cell differs between a right and a wrong prediction; the comparison is the mismatch.
43. [F] If the prediction is to improve by the cell's own adjustment, the cell must also hold the inputs, or traces of the inputs, from which the prediction was formed, because the adjustment can only change how those inputs form the prediction.
44. [F] The mismatch is a quantity of the cell that formed it; no other cell receives it unless it enters the cell's output (sentence 19).
45. [C] The number of predictions a cell holds (one for each incoming connection, one for each part, or one for the cell's total arrival) is choice C12.
46. [C] Which inputs are predicted and which serve to predict them is choice C13: all inputs may be predicted from the cell's own state, or, with separate parts (sentence 7), the inputs arriving at one part may be predicted from the inputs arriving at another part.
47. [C] Where each quantity lies follows from choice C13: with one part, the arrival, prediction and mismatch lie in the same place; with two parts, the prediction is formed in the part that receives the predicting inputs and is compared with the arrival in the part that receives the predicted inputs.
48. [C] The form of the comparison (a signed difference, its magnitude, or only whether the difference exceeds a level) is choice C14.

### 1.6 What adjusting which cells it is connected to consists of

49. [S] A cell keeps adjusting its connections: "to keep adjusting which cells it is connected to".
50. [S] Learning is local: "Learning is therefore local."
51. [F] A cell can adjust its connections only from quantities it holds: its present arrivals, the traces of earlier arrivals in its state, its prediction, its mismatch, its own output and activity, and the chemical signal it receives; it holds nothing else (sentences 5 and 19).
52. [F] A cell receives the chemical signal of another cell only if it lies in the region into which that cell releases it (relayed correction).
53. [C] If strengths exist (C5), how a strength changes is choice C15; the simplest form that uses the mismatch changes the strength of a predicting input by the product of the mismatch and that input's earlier value as kept in the cell's state, and a form without the mismatch changes it by the coincidence of that input with the cell's own activity.
54. [F] When the input that formed a prediction arrived several steps before the mismatch it led to, the cell can relate the two only if a trace of that input is kept in its state until the mismatch is formed.
55. [C] How long such a trace lasts is choice C16.
56. [C] When a cell gives up a connection is choice C17: when its strength has stayed below a level for a period, or, without strengths, when the input on it has not reduced the mismatch over a period, as measured by the product of its trace and the mismatch.
57. [F] A cell holds nothing about a cell to which it is not connected, so the choice of a new source cannot rest on anything that the candidate has sent.
58. [F] A new source must lie in the cell's neighbourhood (sentence 14).
59. [C] How and when a cell makes a new connection is choice C18: for example, to a cell drawn at random from its neighbourhood each time it gives one up, or when its mismatch has stayed high over a period.
60. [C] Whether a cell adjusts its incoming connections, its outgoing connections, or both is choice C19; the words "which cells it is connected to" do not say, and a cell holds quantities (sentence 51) only about its incoming connections, so the simplest form adjusts incoming ones.
61. [C] The rate at which strengths change is part of choice C15.

### 1.7 The encoder's cells and what each reads

62. [S] Encoder cells read bytes; they are the one exception to sentence 17 ("Apart from the byte-level encoder").
63. [S] What an encoder cell reads may be a collection of small passages of the input with gaps between them, and different cells read different ones ("what one of its cells reads may be only a collection of small passages of the input with gaps between them, and different cells read different ones").
64. [F] Since nothing marks where an item begins, a passage can be placed only relative to the present byte (for example, the present byte and the byte three positions back) or by a timing of the cell's own; it cannot be placed relative to the beginning of an item.
65. [F] A passage that reaches back to earlier bytes requires those bytes to be held for that span within the encoder; this falls within the encoder's exception only for a short span, since "a million bytes cannot be kept as a window".
66. [C] The length of a passage, the length of a gap, the number of passages read by one cell, and so the total span, are choice C20.
67. [C] How an encoder cell responds to the bytes in its passages (to one particular value at one particular position, to any of a set of values, or to single bits of the byte) is choice C21.
68. [C] How encoder cells come to differ from one another (by a random draw of positions and values for each cell, or by a systematic cover of positions and values) is part of choice C20.
69. [C] Whether encoder cells adjust what they read and whether they also predict is choice C22; the statement says that the limit on reading "holds for the byte-level encoder" but does not say whether encoder cells otherwise behave as cells outside the encoder.
70. [C] The numbers of encoder cells and of cells outside the encoder are choice C23.

### 1.8 Time

71. [S] The input is one byte after another, one item after another, and nothing tells a cell where an item begins or ends (the description of the input given with the rulings).
72. [C] How many steps take place between two bytes is choice C24.
73. [C] Whether all cells compute their outputs at the same step or in some order is part of choice C24.
74. [F] A cell outside the encoder whose shortest path from the encoder is d connections long receives the effect of a byte no earlier than d steps after the encoder reads it; when few steps pass between bytes, what reaches such a cell while one byte is read is the result of earlier bytes, as the statement says it may be.
75. [F] Nothing sets the states of cells back at the beginning or end of an item, because nothing marks an item and the cells are the only agents of change in the tissue (sentences 1 and 71).
76. [F] The tissue meets one stream, one byte after another, and there is one tissue; there are no copies of it receiving several items at once.
77. [C] The cells' states and connections at the start of the tissue's life are choice C25 (with C7); this one start is not a reset, since it happens once.

### 1.9 The chemical signal

78. [S] A cell can release a chemical signal into a region near it, and the cells of that region receive it alike (relayed correction).
79. [S] How each receiving cell reads the signal is that cell's own matter ("how each receiving cell reads that signal is another matter").
80. [F] There is no signal that reaches every cell of the tissue (relayed correction: "it does not reach the whole tissue").
81. [F] A cell can release the signal only on an occasion it can detect from its own quantities (sentence 51), for example from its own mismatch or its own activity; an occasion defined by the item, the answer or the state of the whole tissue is not available to it.
82. [C] Which cells can release the signal (every cell, or some cells) and on what occasion (for example, when its own mismatch exceeds its own running average of the mismatch by a factor) is choice C26.
83. [C] The size of the region reached by one release is choice C27; when several cells release into overlapping regions and a receiving cell reads the total, the total is a sum over those cells, which is formed by the medium in living tissue but must be computed by the program in a simulation, and a region of small size keeps that sum local.
84. [F] Because the signal is the same for all cells of the region, it cannot tell a receiving cell which of its connections to change; any effect on one connection rather than another must come from the receiving cell's own quantities.
85. [C] What a receiving cell does with the signal (scale the rate at which its strengths change, permit the making or giving up of connections, or permit a change to become lasting) and how its reading differs from that of other cells is choice C28.

### 1.10 Long-term memory

86. [S] Long-term memory is to form in the course of learning and not by an act of a program (ruling given with the task).
87. [F] No step of the program marks a change as lasting, freezes connections, or copies the tissue at a moment chosen by the experimenters, since each of these would be an act of a program.
88. [C] How a change becomes lasting is choice C31: for example, each strength has a component that changes quickly and one that follows it slowly; or a change becomes lasting only while the cell receives the chemical signal; or a connection that has lasted longer is given up less readily.

### 1.11 Numbers

89. [S] The fewer numbers fixed by the experimenters the better, and a number that must be fixed for now is varied on a grid of powers of two, 2, 4, 8, ..., 128 (ruling given with the task).
90. [C] For quantities smaller than one (rates and fractions), this document reads the grid as the reciprocals 1/2, 1/4, ..., 1/128; for the numbers of cells, the grid as stated ends at 128, which may be too few, and whether it is to be extended by further powers of two is a choice the ruling does not settle (it is noted under C23).

### 1.12 What does not exist in the tissue

91. [F] No cell is given the right answer; the answer reaches the tissue only as the answer digits in the byte stream, read by encoder cells like any other byte (section 7, point 1: "the right answer shown as input after the learner's own attempt is the form of teaching"; sentence 18).
92. [F] No part of the tissue is its output and no cell reads every cell, because there is no learner apart from the cells, every cell does only what sentence 3 says, and a cell receives only from cells in its neighbourhood.
93. [F] No error is passed backward along the connections, because a cell sends only its output, forward, to the cells it is connected to (sentence 29).
94. [F] There is no pass over an item after it has ended, because no cell reads past bytes and nothing marks the end of an item.
95. [F] No quantity is computed over a whole population by the program and given to the cells, because learning is local and the only signal that reaches several cells alike is a cell's chemical release into its region.
96. [F] No instruction is computed separately for each cell and delivered to it, and no cell uses another cell's strengths (section 7, point 2, and sentence 19).
97. [F] There is no reset of cells' states at each item (sentence 75).
98. [F] There are no items presented at once in copies and no change made from their average (sentence 76).
99. [F] There is no window of past bytes beyond the short span of the encoder's passages (sentences 6 and 65).
100. [F] There is no goal of the tissue as a whole that is distinct from what each cell does, because the statement describes nothing that acts on the tissue as a whole.

## 2. The choices

Grid G means 2, 4, 8, 16, 32, 64, 128; grid 1/G means 1/2, 1/4, ..., 1/128. "Requirements" refers to the owner's requirements: locality (L), few fixed numbers (N), prediction of what arrives (P), long-term memory formed in learning (M), no item marks (T), the encoder's passages (E), the chemical signal is local (S). The last column says whether a cell could in the end set the value itself from its own quantities.

| No. | Choice (sentences) | Forms, simplest first | Numbers and grid | Requirements | Set by the cell itself? |
|---|---|---|---|---|---|
| C1 | Parts of a cell (8) | one part; two parts, one for inputs that predict and one for inputs that are predicted; a third part for the chemical signal | number of parts (1, 2, 3; too small for the grid) | P, L | No for the number of parts; the assignment of a source to a part could follow from where a new connection is made |
| C2 | Gate (10) | no gate; one gate on the output computed from the cell's own arrivals and state; a gate on how much of the state is kept | number of gates (0 to 3) | P (keeping information over an item) | The gate's strengths, yes; the number of gates, no |
| C3 | Fixed sign of a cell (11) | no fixed sign; each cell excitatory or inhibitory | share of inhibitory cells, 1/G | L, N | No |
| C4 | Negative outputs (12) | outputs of either sign; outputs not below zero | none | P | No |
| C5 | Graded strengths (23) | connections present or absent only; graded strengths | starting strength, 1/G (only with graded strengths) | N, L | The strength, yes, once it exists |
| C6 | Number of incoming connections (24) | fixed number per cell; a number that changes with making and giving up | connections per cell, G | N, L | Yes, if making and giving up are not tied to each other |
| C7 | Connections at the start (25) | drawn at random from the neighbourhood; drawn at random with a bias toward encoder cells | as C6 and C30 | L | Not at the start; afterwards by C17 and C18 |
| C8 | Encoder cells receive from other cells (26) | no; yes, on a separate part | none | E | No |
| C9 | Same output to every target (28) | one output to all targets; different outputs to different targets | none | L | No |
| C10 | What the output carries (31) | activity from arrival and state alone; the mismatch; activity into which the mismatch enters | weight of the mismatch in the output, 1/G | P, and gap 2 of section 3 | Possibly, if that weight is itself adjusted by the cell |
| C11 | How long a cell's state keeps information (39) | one decline rate for all cells; a rate drawn for each cell; a rate adjusted by the cell | retention in steps, G | P, T | Possibly, from whether its mismatch falls when the rate changes |
| C12 | Number of predictions in a cell (45) | one for the total arrival; one for each part; one for each connection | none | P | No |
| C13 | What predicts what (46, 47) | all inputs predicted from the cell's own state; inputs at one part predicted from inputs at another part | none beyond C1 | P, L | No |
| C14 | Form of the comparison (48) | signed difference; its magnitude; whether it exceeds a level | level, 1/G (third form only) | P | The level, yes, from its running average |
| C15 | How a strength changes (53, 61) | product of mismatch and trace of the predicting input; coincidence of input and own activity | rate of change, 1/G | L, P, N | The rate, possibly, from the running size of its mismatch |
| C16 | Length of an input trace (55) | the last step only; a trace declining over several steps | trace length in steps, G | P, T | Possibly |
| C17 | When a connection is given up (56) | strength below a level for a period; no reduction of the mismatch over a period | level, 1/G; period in steps, G | L, N | Yes, if the level is taken relative to the cell's other connections |
| C18 | When and to whom a connection is made (59) | to a random cell of the neighbourhood each time one is given up; to a random cell of the neighbourhood when the mismatch stays high | starting strength (with C5); rate of new connections, 1/G | L | When, yes; to whom, only at random (sentence 57) |
| C19 | Incoming or outgoing connections adjusted (60) | incoming only; outgoing only; both | none | L | No |
| C20 | The encoder's passages and how cells differ (66, 68) | each cell reads passages at positions drawn at random relative to the present byte; a systematic cover of positions | passage length, G; gap length, G; passages per cell, G; total span bounded by these | E, T, N | Not if the encoder is fixed (C22) |
| C21 | How an encoder cell responds to a byte (67) | one value at one position; any of a set of values; single bits | size of the value set, G | E | Not if the encoder is fixed |
| C22 | Whether encoder cells adjust and predict (69) | fixed and not predicting; adjusting what they read; adjusting and predicting | none | E, L, P | No |
| C23 | Numbers of cells (70, 90) | a fixed number of encoder cells and of other cells | encoder cells and other cells, G, possibly extended by further powers of two | N | No |
| C24 | Steps between two bytes and their order (72, 73) | one step, all cells at once; several steps, all cells at once; cells in a fixed or random order | steps per byte, G (one step as the simplest case) | T, P | No |
| C25 | States at the start (77) | all states zero; states drawn at random | spread of the starting states, 1/G | T | No |
| C26 | Who releases the chemical signal and when (82) | every cell, when its mismatch exceeds its running average by a factor; some cells only | factor, G; averaging period in steps, G | S, L, M | The averaging, yes; whether a cell can release at all, no |
| C27 | Size of the region the signal reaches (83) | the cell's own neighbourhood; a smaller or larger fixed region | region size in cells or in distance, G | S, L | No |
| C28 | What a receiving cell does with the signal (85) | scales its rate of change of strengths; permits the making or giving up of connections; permits a change to become lasting | sensitivity of each cell, drawn once or 1/G | S, M | The sensitivity, possibly |
| C29 | A cell's control of its own level of activity (13) | none; a threshold that follows the cell's own recent activity toward a target | target activity, 1/G; period in steps, G | L, P | Yes, from its own activity |
| C30 | Form and size of the neighbourhood (15) | a fixed set of cells drawn for each cell; cells placed in a space with a distance | neighbourhood size in cells, G | L, S | No |
| C31 | How a change becomes lasting (88) | a fast and a slow component of each strength; lasting only while the chemical signal is received; longer-lived connections given up less readily | slow period in steps or bytes, G (possibly extended); ratio of fast to slow rate, G | M, N | Partly: by the second form the cell's own reading of the signal decides |

## 3. Gaps that no choice fills without adding to the statement

1. **Nothing prevents the tissue from making prediction trivial.** If every cell outside the encoder predicts what arrives from other cells, and those cells are under the same rule, every cell is served best when the outputs it receives are constant or silent, and a cell that adjusts which cells it is connected to will tend to keep the sources that are easiest to predict. The statement contains nothing that requires a cell's output to carry information about the bytes. A choice such as C10 or C29 changes how fast this happens but does not by itself supply such a requirement.

2. **Nothing in the statement makes a cell's prediction matter to any other cell.** The statement gives each cell its own task, predicting its own arrivals, and says that its output is computed from its input and state, but it does not say why another cell should be served by that output or by whether the prediction was right. Choice C10 can put the mismatch into the output, but the statement does not say that any cell's task depends on it, so the tissue may be a collection of cells that each fit their own arrivals and do not divide work among them.

3. **The arithmetic task has no place in the statement apart from being more bytes.** The answer digits are, for the tissue, bytes like the task byte, the letters and the sign =, and many of those other bytes are easier to predict. The statement gives no reason why the tissue should come to predict the answer digits rather than the easier regularities. Section 7 also speaks of the right answer shown "after the learner's own attempt", but the tissue as written out acts on nothing outside itself, so it makes no attempt, and nothing in the statement says what an attempt would be.

4. **A new connection can be made only blindly.** A cell holds no quantity about a cell it is not connected to (sentence 57), so the statement's "adjusting which cells it is connected to" can only be trial at random within the neighbourhood followed by giving up; the statement gives no means by which a cell could find the cells worth connecting to. Whether trial at random suffices for information about the first digits to reach the cells that must predict the answer is a question the statement does not answer.

## 4. How an experimenter could read whether the tissue has learned anything

The operations available are those that have a meaning for a living learner: recording what cells do (their outputs, and, as a physiologist records the potential of a part of a cell, their internal quantities such as the mismatch); showing the tissue a stream and recording its response; and taking cells away. Whatever the experimenter sums or compares, he does so on the records, and nothing computed from the records is returned to the tissue.

**The proposed comparison.** The tissue's surprise at a byte is the sum, over recorded cells and over the steps that follow that byte, of the magnitude of their mismatch. The proposal is to compare the surprise at the answer digits when an item is followed by its right answer and when it is followed by a wrong one. This requires the following of the tissue as written out.

1. The mismatch must exist as a quantity in each cell (sentence 42), and the experimenter must be able to record it; with C10 it may also be visible in the outputs.
2. The wrong answers must differ from the right ones only in the answer digits, and each wrong digit must occur at that place in the stream as often as the right digits do; otherwise a difference in surprise reflects how often digits occur and not arithmetic.
3. The items tested must include items whose pairs of digits never occurred with that answer in the stream before, or the difference may show only that particular items were kept.
4. Since a cell never stops adjusting, every wrong answer shown is also taught; stopping all change during the test is an act of a program with no counterpart in the living tissue, so test items must be few compared with the stream, and the change they cause must be accepted and recorded.
5. Since the tissue cannot be copied and is never reset (sentences 75 and 76), the right and the wrong answer to an item are shown at different times to a tissue whose history differs, so the comparison must be made over many items, with the order of right and wrong balanced.
6. Encoder cells read the answer byte directly; if they predict (C22), their mismatch reflects how often each byte occurs, so the surprise of encoder cells and that of cells outside the encoder must be recorded and reported separately.

**What it could fail to show.**

- A lower surprise for right answers can come from a partial regularity rather than the whole rule; for example, in addition the units digit of the answer depends only on the units digits of the two numbers, so the comparison must be broken down by which digit and by whether the item needs a carry.
- After a wrong first answer digit, the second digit is surprising whether or not the tissue has learned anything about the second digit, so the two answer digits must be compared separately and the second only after a right first digit.
- A difference shows that the tissue's expectations distinguish right from wrong answers; it does not show that the tissue could produce an answer, since it acts on nothing (gap 3).
- The absence of a difference does not show that nothing was learned: the information may be present in cells whose mismatch is not recorded, or the surprise at the answer may be small beside the surprise at the letters and other bytes that cannot be predicted.
- A difference measured soon after teaching may rest on recent changes only; to read long-term memory (sentence 86), the comparison must be repeated after a long stretch of the stream in which those items do not occur.

**Two further readings.** First, recording the outputs of cells outside the encoder just before the answer digits and asking whether those outputs differ with the right answer shows whether information about the answer is present in the tissue; it is the experimenter's analysis of records, not a part of the tissue, and it does not show that any cell uses that information. Second, taking cells away (for example, the cells whose outputs differ most with the right answer, or the cells of one region) and repeating the comparison of surprise shows which cells the difference depends on; if the difference remains after cells chosen at random of the same number are taken away but disappears after the chosen cells are taken away, those cells carry it.
