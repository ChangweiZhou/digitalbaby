# The tissue defined, revision 4 (18 November 2026, written from 23:36 UTC by the clock of this machine; Mr. Xu's rulings of 23:37 UTC, section 16, taken in from 23:39 UTC)

This file replaces revision 3 (`THE_TISSUE_DEFINED_REVISION_3_20261118.md`), which is not changed. It settles the findings of the fourth reading (`FOURTH_READING_OF_THE_TISSUE_DEFINED_20261118.md`), follows roadmap draft 3 (both ends are corrected at step 2, and relay runs inward from both ends at step 3), and takes in sections 15 and 16 of `WHAT_A_NEURON_HAS_20261118.md`. The briefs of steps 1 and 2 are to be written from it. The only programs run were `python3 -I` with the standard library, for the counts in section 12 and the arithmetic in sections 8 and 13.

## Summary

The first form changes from revision 3 in eight places:

1. **Timing (G1).** There are 32 steps per byte, τ_v = 2, and τ = 32. At the reading step an output-end cell has 0.97 to 1.00 of the effect of the byte just read, against at most 0.03 in revision 3.
2. **One rule for a strength, set by both ends of the connection (section 16; G2).** The sending side supplies the trace of its own output at the connection. The receiving side supplies its mismatch times its own activity at the previous step. The same rule acts at every connection between cells, including those into output-end cells. Revision 3's separate feedback rule (formula 14) is withdrawn, and so there are no longer two rules at one cell that undo each other.
3. **Feedback cells (section 16).** Only some cells receive the feedback: 64 feedback cells with random coordinates. They pass it on to whichever cells connect to them, and those connections are made, kept and given up by the ordinary rules. Nothing lays down which cells of the ends they reach. The output end is reached in the same way, by our choice.
4. **Coding of yes and no (G3).** Yes and no arrive on separate lines, so yes, no and silence differ. Feedback cells that receive "no" are excitatory and those that receive "yes" are inhibitory. Given the sign of the rule, this makes the change at a receiving cell that was active point the corrective way (sentence 91).
5. **Which cells move (G4).** A cell changes only in proportion to its own activity just after the emission. A yes or no passed on alike cannot point the right way for an active cell that did not cause the emission (sentence 92). The per-line and per-answer forms are not excluded.
6. **Section 15 (G6).** Both readings are forms of C52 and both are run at step 2. In form (a) the feedback cells receive yes or no. In form (b) a further 64 feedback cells receive the emitted byte.
7. **Start of life (G7).** The first new connection can be made only P_s steps after the start.
8. **Numbers.** P is split into P_s and P_r. C35 is merged into C52. C55 (the recording step) and C56 (feedback cells) are new.

Counts: 146 sentences: [S] 28, [F] 58, [C] 60. There are 52 choices.

## 0. Marks and terms

**Marks.**
- [S]: Mr. Xu's words, quoted from sections 7 to 16 of `WHAT_A_NEURON_HAS_20261118.md`. "Reported" means the section gives his words only in the reviewer's wording.
- [F]: follows by necessity, for the reason given.
- [C]: our choice, tabled in section 14. A ruling whose words are not quoted is marked [C], and the sentence says so.

The bracket after each mark gives the sentence's number in revision 3 ("r3 n"), or "new".

**Terms.**
- *Cell*: a nerve cell of the tissue.
- *Coordinate*: the point given to a cell at random at the start of its life.
- *Input face* and *output face*: the sides x = 0 and x = 1 of the space.
- *Input line*: a fixed channel, not a cell, that is active (value 1) while the present byte has one particular value.
- *Output line*: a fixed channel through which one output-end cell drives the emission of one byte value.
- *Feedback line*: a fixed channel through which the world gives feedback to one feedback cell.
- *Feedback cell*: a cell that receives a feedback line (sentence 85).
- *Encoder cell*: a cell, not a feedback cell, with at least one input line in its neighbourhood.
- *Output-end cell*: a cell that drives an output line.
- *Inner cell*: any other cell.
- *Step*: one exchange of outputs among all cells.
- *Window*: the s steps from one byte's arrival to the next. Steps within it are numbered φ = 0, …, s − 1, and the byte arrives at φ = 0.
- *Reading step*: φ = s − 1.
- *Emission*: the byte given out for a position, or no byte.
- *Attempt*: the emission where, by the experimenter's reading, an answer is due.
- *Feedback*: what the world gives about an emission or an answer. It includes yes and no.
- *Distance from the input lines* or *from the output end*: the number of connections on the shortest path.
- *Return connection*: a connection to a cell with a smaller coordinate x.
- *Sending side* and *receiving side* of a connection: the parts of the sending and receiving cells at that connection (in a living synapse, the presynaptic terminal and the postsynaptic spine).
- *Part* of a cell: a region at which inputs from some sources arrive and are combined.
- *Living counterpart*: a structure or process in living tissue that does what the sentence describes.

## 1. Cells and space

1. [F] (r3 1) Nothing in the tissue learns apart from its nerve cells. The reason is that "没有专门的'学习者'" ("there is no dedicated learner", section 8), and the statement lets nothing else act.
2. [S] (r3 2) "Apart from the byte-level encoder, all that any nerve cell can do is ..." (section 8). There is an output end and an input end, and "these two also have to learn, through learning, how to take in and how to give out" (section 10).
3. [F] (r3 3) Each cell is an encoder cell, an inner cell, an output-end cell, or a feedback cell (sentence 85), and all are nerve cells.
4. [S] (r3 4) "为什么一开始不能给每个细胞一个随机生成的坐标呢？" ("why can each cell not be given a randomly generated coordinate at the start?", section 15).
5. [F] (r3 4) Each cell is therefore given a coordinate at random at the start, and near and far are distances between coordinates; section 15 records the question as his ruling. No arrangement of cells is laid down. Placing the lines on two opposite faces lays down one axis, and that is our choice (C53).
6. [C] (r3 5; C30, C53, C54) The first form of the space and lines:
    - The space is the unit square 0 ≤ x, y ≤ 1, without joined edges. The 2,048 cells, feedback cells included, are placed uniformly at random in it.
    - The 256 input lines are at random points on x = 0, and the 256 output lines at random points on x = 1.
    - Each output line, in byte order, is driven by the nearest cell that is neither a feedback cell nor already driving a line. Ties are broken by draw order.
    - A neighbourhood is every cell, and for a cell that is not a feedback cell every input line, within distance 1/8, excluding the cell itself.
7. [F] (r3 6, corrected) A cell at x lies at least about 8x connections from the input lines, and at least about 8(x_out − x) connections from the output end, where x_out ≥ 0.83 is the x of the nearest output-end cell. Distances are computed by the experimenter for reading only.
8. [S] (r3 7) A cell receives input from other cells, sends from that input and its present state an output to the cells it is connected to, and keeps adjusting which cells it is connected to (section 8).
9. [S] (r3 8) A cell has "its present state" and "the state it has stored" (section 8).
10. [F] (r3 9) What a cell holds is its only record of the past (section 8: "it can read only the state it has stored and the new information that reaches it through other cells").
11. [S] (r3 10) "A million bytes cannot be kept as a window" (section 8).
12. [S] (r3 11) Separate parts of a cell may be used (section 7, point 1, reported).
13. [C] (r3 12; C1) In the first form a cell has one part.
14. [S] (r3 13) "The gated cell may stay" (section 7, point 1, reported). A living counterpart of three gates is unsettled.
15. [C] (r3 14; C2) In the first form a cell has no gate.
16. [C] (r3 15, 16; C3) Each cell is excitatory or inhibitory for life, as an idealisation of Dale's principle. One in four cells, at random, is inhibitory, except that output-end cells are excitatory and feedback cells take the kind given in sentence 87.
17. [F] (r3 17) A cell's output is never below zero.
18. [S] (r3 18) A cell acts and adjusts only within its own neighbourhood (section 9, reported).
19. [C] (r3 19; C48) Electrical synapses are left out.

## 2. What reaches a cell, and what it sends

20. [S] (r3 20) A cell receives "input from other cells" (section 8).
21. [S] (r3 21) "细胞应该可以得到误差的回馈" ("cells should be able to receive the feedback of the error", section 13).
22. [F] (r3 22, corrected) Only these can reach a cell:
    - the outputs of cells connected to it;
    - for an encoder cell, input lines;
    - for a feedback cell, its feedback line or lines;
    - the chemical signals of its region.
23. [S] (r3 23) A cell does not in general read the bytes that have passed (section 8, "未必能够读到已有的字节").
24. [F] (r3 24) In the first form, bytes enter through input lines, and feedback enters only through feedback cells.
25. [S] (r3 25, 26) What arrives differs from cell to cell, and it may result from earlier bytes (section 8).
26. [F] (r3 27) A cell's sources lie within its neighbourhood.
27. [C] (r3 28; C5) A strength is graded between 0 and 1.
28. [C] (r3 29; C7, C8) At the start each cell draws 8 sources at random from its neighbourhood: without replacement, never itself, and never twice. An encoder cell draws from lines and cells together. A feedback cell draws only from cells.
29. [S] (r3 30) A cell sends an output "to the cells it is connected to" (section 8).
30. [F] (r3 31) A cell sends the same output to all its targets, including its output line.
31. [F] (r3 32, 33) A cell sends no electrical output backward along a chemical synapse. Loops are not excluded.
32. [F] (r3 34) Return connections exist in the first form, because neighbourhoods extend in all directions.
33. [C] (r3 35; C43) There are no return connections other than those the neighbourhood gives.
34. [C] (r3 36; C10) A cell's output is its activity (formula 3).
35. [S] (r3 37) A cell can release a chemical signal into a region near it (section 9, reported).

## 3. Prediction

36. [S] (r3 38) A cell's work is "预测从其他细胞得来的新来的信息" ("predicting the information newly arriving from other cells", section 8).
37. [S] (r3 39) It "cannot simply be taken as" expecting the next byte (section 8, "不能简单认为是").
38. [F] (r3 40, 41) A prediction is formed before its arrival and is compared with the arrival in the same cell at the same step.
39. [F] (r3 42, 43) The task's stream requires the tissue to keep information about earlier bytes in the states and outputs of cells.
40. [F] (r3 44, 45) A cell must hold its arrival, a prediction, the mismatch, and traces of its inputs, if its mismatch is to fall by adjustment.
41. [C] (r3 46; C12 to C14) The prediction is the running average of the cell's own arrival, and the mismatch is the signed difference. What is learned is held in the strengths. This is cancellation of the foretold part of the arrival: it has a living counterpart in cerebellum-like structures and is unsettled in cortex.

## 4. Formulas of the first form

All are [C] except sentence 42. For a cell i:
- j ranges over its sources;
- σ_j = +1 for an excitatory cell or a line, and −1 for an inhibitory cell;
- z_j(k) is a line's value at step k, or a cell's output x_j(k − 1).

42. [S] (new) "细胞之间连接的强度是连接的两端共同决定的，是局部的量" ("the strength of a connection between cells is determined jointly by the two ends of the connection; it is a local quantity", section 16).
43. [C] (r3 48; formula 1, arrival) A_i(k) = Σ_j σ_j · w_ij(k) · z_j(k).
44. [C] (r3 49; formula 2, state; C11) v_i(k) = (1 − 1/τ_v) · v_i(k − 1) + (1/τ_v) · A_i(k), with τ_v = 2.
45. [C] (r3 50; formula 3, activity; C10) x_i(k) = min(1, max(0, v_i(k) − θ_i(k))).
46. [C] (r3 51; formula 4, prediction; C13) p_i(k + 1) = (1 − 1/τ) · p_i(k) + (1/τ) · A_i(k), with τ = 32.
47. [C] (r3 52; formula 5, mismatch; C14) m_i(k) = A_i(k) − p_i(k).
48. [C] (r3 53; formula 6, sending-side trace; C16) e_ij(k + 1) = (1 − 1/τ) · e_ij(k) + (1/τ) · z_j(k). This is held at the sending side of the connection, as the recent output of cell j arriving at that connection.
49. [C] (r3 55, corrected; formula 7, the strength of a connection between cells; C15, C41) Δw_ij = −η · σ_j · [m_i(k) · x_i(k − 1)] · e_ij(k), with η = 1/8.
    - The sending side contributes e_ij(k), its trace at the connection, and its sign σ_j.
    - The receiving side contributes m_i(k) · x_i(k − 1), its mismatch times its own activity at the previous step.
    - The rule acts on every connection between cells, into every kind of cell.
    - Connections from input lines and feedback lines are not changed by it (sentence 63).
    - The living counterparts are presynaptic activity and the requirement of postsynaptic depolarisation for plasticity, both textbook; the cancelling sign is that of cerebellum-like structures.
50. [F] (new) Formula 7 is a function of quantities present at the connection only: its strength, the sending side's trace and sign, and the receiving side's mismatch and activity. It meets sentence 42.
51. [C] (r3 56; formula 8, bounds) After all the changes of a step, each strength is set to min(1, max(0, w)).
52. [C] (r3 57; formula 9, activity control; C29) θ_i(k + 1) = max(0, θ_i(k) + η_θ · (x_i(k) − a*)), with η_θ = 1/128, a* = 1/8 and θ starting at 0, for all cells.
53. [C] (r3 58; formula 10, running averages; C17) ā_i(k + 1) = (1 − 1/P_s) · ā_i(k) + (1/P_s) · x_i(k), and ū_ij(k + 1) = (1 − 1/P_s) · ū_ij(k) + (1/P_s) · z_j(k), with P_s = 128 steps, both starting at 0. ū_ij is held at the sending side of the connection.
54. [C] (r3 59; formula 11, giving up; C17, C19) A connection at least P_s steps old is given up when its strength is below 1/128, or when its sending side's running average ū_ij is below 1/128. Either end's quantity at the connection can therefore end it. A feedback cell's feedback line is never given up.
55. [C] (r3 60, corrected; formula 12, making; C18, C19, C6) A cell makes one new incoming connection when all three of these hold:
    - θ_i = 0;
    - ā_i < a*;
    - at least P_s steps have passed since its last making, or since the start of life.

    The source is drawn at random from its neighbourhood members that are not already its sources. The strength starts at 1/8 and the traces at 0. The receiving side starts a connection, and the strength is then set by both ends (formula 7).
56. [F] (r3 61) A cell's number of connections is its own, as asked in section 12 ("有没有办法让每个细胞自己决定要进行多少条连接"). Only the starting number, 8, is fixed by us.
57. [F] (r3 62, 147, corrected) The first makings occur at step P_s = 128. A cell whose activity at θ = 0 stays below a* adds one connection every P_s steps. A cell whose drive allows it holds its mean activity near a* by its threshold.

The order of operations is sentence 118.

## 5. The input end

58. [S] (r3 63, 64) Encoder cells read bytes ("Apart from the byte-level encoder"). What one reads "may be only a collection of small passages of the input with gaps between them, and different cells read different ones" (section 8).
59. [F] (r3 65) The input end consists of the encoder cells.
60. [C] (r3 66; C32, C21) There are 256 input lines, one for each byte value. The line of the present byte is active for all s steps of its window.
61. [F] (r3 67) An encoder cell's passages are formed by its traces, its state and its connections.
62. [C] (r3 68; C20) There are no separate numbers for passages.
63. [C] (r3 70, corrected) Connections from input lines and from feedback lines change only by being given up and made. Input-line connections start at 1/8, and feedback-line connections are fixed at 1.
64. [C] (r3 71; C22) Encoder cells predict their arrivals.
65. [C] (r3 72; C23) There are 2,048 cells: 256 output-end cells, 64 feedback cells (128 in form (b) of C52), and 236 to 262 encoder cells in the draws of section 12.
66. [F] (r3 144, corrected) Over all encoder cells, 38 to 42 per cent of the starting draws are lines (fourth reading).

## 6. The output end

67. [F] (r3 73) The tissue gives out bytes (section 10; the example in section 11 is "预测next byte出现错误").
68. [F] (r3 74) The conversion of output-line activity into a byte is fixed and does not learn.
69. [C] (r3 76; C34) The tissue emits at every position. The emission is read at the reading step.
70. [F] (r3 77) No cell has the coming byte at the reading step. Where an answer is due is known only to the experimenter.
71. [S] (r3 78) The right answer is shown as input after the learner's own attempt (section 7, point 1, reported).
72. [C] (r3 79; C33) The emitted byte is that of the most active output-end cell at the reading step, if its activity exceeds 1/2. Otherwise no byte is emitted. Ties go to the lowest value.
73. [F] (r3 80) This conversion is the one comparison over many cells in the first form. It reaches cells only through the emission and the feedback. Whether a conversion that is not learned agrees with section 10 is left open. Its learned alternative is the further form of C33 (competition by inhibitory cells).
74. [C] (r3 82; C38) A position with no byte emitted receives "no", and for scoring it is wrong. Positions where nothing can be expected are identified by the experimenter, are not scored, and still receive feedback.
75. [F] (r3 84) Whether the emitted byte is read back is form (b) of C52. C35 is merged into C52.

## 7. Feedback

76. [S] (r3 83) "所以你可以给整个系统连续的信号，如果输出端的信息和预期不符合（例如预测next byte出现错误），告诉系统它错了让它自己去纠正" (section 11).
77. [S] (r3 84) Cells should be able to receive the feedback of the error. Yes or no is acceptable, and it is the form for the prediction of the next byte; other forms are to be accommodated (section 13, "细胞应该可以得到误差的回馈 … 应该可以容纳其他形式的回答").
78. [S] (r3 85) "我们可以把正确答案加上差多少的绝对值作为一个额外的反馈信息给它" (section 14).
79. [S] (r3 86) Information may be relayed from neuron to neuron, and a single backward pass over the whole network is not allowed (section 11, reported).
80. [S] (new) "我们可以假设输入端可以得到输出端信息的反馈。在NLP任务中这是next byte prediction的结果，但在其他任务中可以是其他的" (section 15).
81. [S] (new) "我们的输入端不是所有的细胞都收到yes/no，只有某些收到。可以认为有专门的细胞收到反馈信息之后再传送给输入端，具体如何传送需要学习" (section 16): not all cells of the input end receive the yes or no, only some; dedicated cells receive the feedback and pass it on to the input end, and how it is passed on is to be learned.
82. [C] (r3 88; C49) The following remain excluded:
    - the right answer for a position given to a cell before that position's emission is read;
    - a cell's change computed outside the cell;
    - a backward pass through forward connections;
    - a chemical signal reaching the whole tissue;
    - any reset.
83. [S] (r3 89) The world may make the comparison and give its result ("你可以给", section 11).
84. [C] (r3 90, 91; C37) The world compares the emission read at the reading step with the byte arriving at the next φ = 0, at every position for the whole life. "连续" is read as continuing in time; the reading as continuous in value is a further form.

### 7.1 Feedback cells and the coding of yes and no

85. [C] (new; C56) There are 64 feedback cells. They are cells like the others, with coordinates drawn at random like every cell's. Each has 8 ordinary sources and follows formulas 1 to 12. Besides these, each receives one feedback line at fixed strength 1, which is never given up. A feedback cell does not draw input lines and does not drive an output line.
86. [C] (r3 92, corrected; C37, C39; settles finding G3) For each position there are two feedback lines: a yes line, active (value 1) for one step at φ = 0 of the window in which the byte arrives if the emission equalled that byte; and a no line, active then otherwise. At all other steps both are 0, and at the first position of life neither is active. Half of the feedback cells, chosen at random, receive the yes line and the other half the no line. A feedback cell therefore tells yes, no and silence apart by which line it receives and whether that line is active, without knowing φ.
87. [C] (new; C56) Feedback cells that receive the no line are excitatory, and those that receive the yes line are inhibitory. The reason is sentence 91. Kinds drawn as for other cells are a further form. The living counterpart: worse outcomes excite cells of the lateral habenula, and better outcomes excite dopamine cells (Matsumoto and Hikosaka, 2007; Schultz, Dayan and Montague, 1997). The particular pairing of kinds is not a finding about those cells.
88. [F] (new, from sentence 81) Feedback reaches other cells only over the ordinary connections of cells that have drawn a feedback cell as a source. These connections are made, kept, given up and changed by formulas 7, 11 and 12, which is how the passing on is learned. Nothing lays down in advance which cells of either end they reach.
89. [C] (new; C40) The output end is reached in the same way as the input end, through feedback cells only. Section 16 speaks only of the input end, so this is our choice. Feedback lines ending directly at output-end cells are a further form.
90. [F] (new) What a cell that receives from a feedback cell has from which to change: at φ = 1 (the feedback cell's output reaches it one step after the feedback line is active) its arrival rises, if the source is a no-cell, or falls, if it is a yes-cell. The change enters its mismatch, and by formula 7 each of its connections then changes in proportion to the sending side's trace and to its own activity at the previous step. The connection from the feedback cell itself changes by the same rule.

### 7.2 What moves, for a yes and for a no (settles findings G2 and G4 as restated by section 16)

91. [F] (new) For a cell that receives from a feedback cell and was active at φ = 0:
    - **No:** the no-cell's excitatory output raises the cell's arrival above its prediction. Formula 7 then weakens the excitatory inputs, and strengthens the inhibitory inputs, whose traces are high, which are those of the context that led to the emission. The cell is less driven in that context.
    - **Yes:** the yes-cell's inhibitory output lowers the arrival below the prediction, and the same inputs move the other way. The cell is more driven in that context.
    - **A cell with x = 0 at the previous step does not change.**
    - At the output end this is corrective for the cell that emitted.
92. [F] (new) A yes or no passed on alike to the active cells that receive it cannot point the right way for an active cell that did not cause the emission. A no weakens such a cell's context even if its byte was the right one, and a yes strengthens a context in which its byte was wrong. The factor x_i(k − 1) confines this to cells that were active just after the reading step. The per-line form (sentence 109) points the right way for each output-end cell. It is not excluded.
93. [F] (new) The two rules that the fourth reading found undoing each other at output-end cells (G2) are, under sentence 42, one strength set by its two ends (formula 7). The feedback is no longer a separate factor: it is part of the receiving cell's arrival and mismatch. Over many positions the rule moves a connection until the cell's context drive foretells the feedback arrival that follows. Concretely, an output-end cell becomes less driven in contexts that are followed by a "no" from a no-cell connected to it, and more driven in those followed by a "yes" from a yes-cell. Whether enough output-end cells come to receive from feedback cells is read at step 2 (section 12 gives the starting counts).

### 7.3 The two readings of section 15 (settles G6)

94. [F] (new) "输出端信息的反馈" admits two readings: the result of the comparison (yes or no), or the output end's information itself (the emitted byte). Section 11 uses "输出端的信息" for what the output end gives.
95. [C] (r3 172, replaced; C52) Both readings are forms of C52, and both are run at step 2, each in its own tissue, beside the tissue named by roadmap draft 3. No choice is made between them by argument.
    - **Form (a):** the feedback cells of sentences 85 to 87 only.
    - **Form (b):** in addition, 64 further feedback cells, with coordinates at random, kinds as for other cells, and 8 ordinary sources each. Each also draws 8 of 256 emission lines at random, at strength 1/8, changed only by giving up and making. The line of the emitted byte is active (value 1) for one step at φ = 0 of the window in which the next byte arrives. No line is active if no byte was emitted, or at the first position.
96. [F] (new) Under sentence 88, where feedback cells pass the feedback is not laid down in either form. The two forms differ only in what the dedicated cells receive:
    - **Form (a):** a cell of the input end that receives from a feedback cell has a rise or a fall in its arrival at φ = 1 that says whether the emission was right, and its own traces of the context.
    - **Form (b):** such a cell may also have, at φ = 1, an arrival that depends on which byte was emitted. That arrival comes at the step after the arriving byte appears on the input lines. The comparison of the two, if any, is made by the cells' own predictions and mismatches.
97. [F] (r3 174, corrected) Timing at a receiving cell: the feedback cell's output arrives at φ = 1. The factor x_i(k − 1) is then the cell's activity at φ = 0, which, with τ_v = 2, is half carried over from the reading step. The traces at φ = 1 keep (31/32)^32 ≈ 0.36 of the context from the start of the window.
98. [F] (r3 175, corrected) Feedback cells lie anywhere in the square, so the input end is not the last part reached. Relay at step 3 runs from wherever feedback cells are connected.
99. [F] (r3 176) Feedback comes from the world to feedback cells and does not pass back through forward connections, so it is not a backward pass.
100. [C] (r3 177; C52) In other tasks the brief names the output end's information, which enters through feedback cells in the same way.

### 7.4 Elsewhere in the first form

101. [F] (r3 108, corrected) A cell more than one connection from every feedback cell receives the effect of feedback only as changed outputs of the cells between, mixed with context.
102. [C] (r3 103; C42) No further relay by a change of activity is added in the first form. If one is added, it must be kept out of the emission (by τ_v = 2 and s = 32, a change at φ = 0 has fallen to (1/2)^31 by the reading step).
103. [F] (r3 104) With the sign of formula 7, a relayed rise lowers a cell's drive in that context along paths whose signs multiply to +1, and raises it along paths whose signs multiply to −1. No cell knows its path signs, and the balance is read at step 3.
104. [F] (r3 106, 107) A change reaches a cell d connections from a feedback cell at least d steps later. The interval its traces must span is at least 2d + 1 steps, which is 9 for d = 4, where (31/32)^9 ≈ 0.75 remains. With r outgoing connections per cell, d relays reach at most r^d cells.
105. [F] (r3 109) With one part, a relayed change enters the arrival. With two parts it can be kept apart.
106. [C] (r3 110, 154; C44, C1) How a cell combines relayed correction with prediction is not opened before step 3. The two-part form assigns return connections and connections from feedback cells to the second part.
107. [C] (r3 105; C40) The further receivers of feedback lines are output-end cells directly (sentence 89) and more feedback cells (C56).

### 7.5 Richer feedback (further forms of C37 and C39)

108. [F] (new) Each richer form below enters, in the first place, through feedback cells, as section 16 requires for the input end.
109. [C] (r3 97, coding settled) In the per-line form, each output line has two lines: "should have been active", active for the line of the arriving byte, and "was wrong", active for the emitted line if it differs. They are active at φ = 0 for one step. In the first variant they end at feedback cells, one for each line; in the further variant they end at the output-end cell of that line (sentence 107).
110. [C] (r3 98) In the per-answer form, the right answer is given by the lines of sentence 109 at each position of the answer, after the emission for that position.
111. [C] (r3 99, 100, coding settled) The magnitude goes on one magnitude line to dedicated feedback cells, at φ = 0 after the answer's last byte, for one step.
     - The line carries 1 − d/2, which lies between 1/2 and 1 when present and is 0 when absent.
     - d = |a − b| / D, where D is the largest difference the brief's answers allow. For arithmetic modulo 89, d is the circular difference divided by 44.
     - d = 1 if no readable number was given.
112. [C] (new) When two kinds of feedback reach a cell at one step, they enter its arrival together, and formula 7 acts on the sum.
113. [F] (r3 101, recomputed) Traces:
     - **For each position:** context traces at φ = 1 keep 0.36 from the window's start.
     - **For a three-byte answer:** the feedback reaches back 3 · 32 + 2 = 98 steps, where (31/32)^98 ≈ 0.04 remains. A brief using per-answer feedback opens C16 to 64 or 128.

## 8. Time and order of operations

114. [C] (r3 111; C45) The input is one byte after another, with nothing marking an item. This ruling is not quoted (section 12 says so), so it is marked as our choice.
115. [C] (r3 112, corrected; C24, C11) s = 32, τ_v = 2, and all cells compute at once from the previous step's outputs.
116. [F] (new; settles G1) A byte's effect at a cell n connections in is delayed by about n · τ_v steps. Output-end cells lie 8 to 11 connections from the input lines. For a chain of n cells with unit gain and no threshold, and a line held at 1 from φ = 0, the share of the eventual response reached at φ = 31 is:

| n | 8 | 9 | 10 | 11 |
|---|---|---|---|---|
| τ_v = 2, s = 32 (first form) | 0.999 | 0.997 | 0.990 | 0.975 |
| τ_v = 4, s = 16 (revision 3) | 0.027 | 0.008 | 0.002 | 0.000 |

     Thresholds and strengths lower the size of the effect, not this share. The emission can therefore depend on the byte just read.
117. [C] (new; C55) For step 1, outputs are recorded at the reading step. The brief may add other steps.
118. [C] (r3 113, updated) The order of operations at step k, with φ = k mod s:
     1. If φ = 0, the world sets the input lines and compares the new byte with the emission read at the previous step. It makes the yes line or the no line active, and in form (b) the emitted byte's emission line. At the first position none of these is active.
     2. Every cell computes A (formula 1).
     3. Every cell computes m (formula 5).
     4. Every cell computes v and x (formulas 2 and 3).
     5. Every cell computes the changes of formula 7 from the traces held before the step and from x(k − 1), and applies formula 8.
     6. Every cell updates its traces, p, θ, ā and ū (formulas 4, 6, 9 and 10).
     7. Every cell gives up and makes connections (formulas 11 and 12).
     8. If φ = s − 1, the emission is read (sentence 72).
119. [F] (r3 114, 115) Within one byte:
     - at φ = 0 the byte reaches encoder cells and the feedback lines reach feedback cells;
     - at φ = 1 feedback cells' outputs reach their targets;
     - a cell n connections in receives the byte's effect from φ = n − 1, and most of it by about φ = 2n;
     - at φ = 31 the emission reflects the byte just read.
120. [F] (r3 116, 117) Nothing outside the cells resets their states, and there is one tissue meeting one stream.
121. [C] (r3 118; C25) At the start all states, traces, predictions, thresholds and running averages are 0, once.

## 9. Chemical signal, lasting change, reactivation and numbers

122. [S] (r3 119, 120) A cell's chemical signal is local, and how each receiving cell reads it is its own matter (sections 9 and 7, reported).
123. [F] (r3 121) A chemical signal released by a cell never reaches the whole tissue.
124. [C] (r3 122, 124; C26 to C28) In the first form no cell releases a chemical signal.
125. [F] (r3 123, 125) A cell can release a chemical signal only on an occasion it detects itself. A signal received alike by a region cannot choose which connection changes.
126. [C] (r3 126; C46) Long-term memory forms in learning and not by an act of a program. This ruling is on record but not quoted, so it is marked as our choice.
127. [F] (r3 127) No program marks a change as lasting, freezes connections, or copies the tissue.
128. [C] (r3 128; C31) Each strength has one component.
129. [F] (r3 129) No program passes over an item after it has ended. The tissue's own reactivation is neither excluded nor imposed.
130. [S] (r3 130) "我们预先给定的参数越少，长期来看越好", and, where needed for now, "从2, 4, 8, 16,32,64,128等等取一些数字" (section 12).
131. [C] (r3 131; C47) Fractions are read on the grid 1/2 to 1/128. Numbers of cells go beyond 128 by powers of two.

## 10. What does not exist

132. [C] (r3 132) In the first form no cell is given the right answer. Feedback cells receive yes or no and, in form (b), the emitted byte. The per-line and per-answer forms give the right answer after the emission (sentence 82).
133. [F] (r3 133) No cell reads every cell.
134. [F] (r3 134) No error passes backward through forward connections.
135. [F] (r3 135) The quantities given to cells from outside are input lines and feedback lines, and only feedback cells receive the latter.
136. [F] (r3 136) No change of strength is computed outside the connection's two ends.
137. [F] (r3 137 to 139) There is no reset, no copy, no averaging over items, and no window beyond what states and traces keep.
138. [F] (r3 140) The only criterion over the whole tissue is the world's comparison, reaching cells only through feedback cells.
139. [F] (r3 150) The first form uses only quantities that a living cell holds or that are present at its connections.
140. [C] (r3 145; C51) The random draws use one generator with a recorded seed.
141. [C] (new) The numbers fixed by us are those of section 16.
142. [C] (new) The orders of opening are those of section 15.
143. [F] (new) Each finding and undecided item of the fourth reading is settled as section 17 shows.
144. [C] (r3 19, 70; C19) Who adjusts a connection: its strength is set by both ends (formula 7); either end's quantity can end it (formula 11); the receiving side starts it (formula 12). The receiving side alone adjusting incoming connections (revision 3's C19) is withdrawn by sentence 42.
145. [F] (new) In the first form, a feedback cell's output has a no-cell's or a yes-cell's sign, fixed by sentence 87, at every connection it makes.
146. [C] (new; C56) The number of feedback cells, 64, is on the grid. Section 12 shows its effect on how many end cells are reached at the start.

**Counts.** 146 sentences: [S] 28, [F] 58, [C] 60.

## 11. (No section; the numbers 12 to 18 are kept for reference from the readings.)

## 12. Counts at the start of life (first form, `python3 -I`)

**Distances, three draws with 2,048 cells and radius 1/8.** These were counted before feedback cells were set apart. Setting 64 of the 2,048 apart as feedback cells changes these counts by a few per cent.

| Distance | From the input lines | From the output end |
|---|---|---|
| 1 | 237 to 247 | 175 to 196 |
| 2 | 163 to 188 | 172 to 189 |
| 3 | 175 to 204 | 192 to 207 |
| 4 | 177 to 237 | 194 to 215 |
| 5 to 9 | 191 to 225 each | 141 to 238 each |
| 10 | 138 to 219 | 21 to 60 |
| 11 | 3 to 36 | 1 to 7 |

- Neighbourhoods hold 89 to 91 cells on average.
- Output-end cells lie at x ≥ 0.83, and 8 to 11 connections from the input lines.
- The shortest path from the input lines to the output end is 8 to 9 connections.

**End cells reached from feedback cells at the start.** Three draws for each number of feedback cells, counting connections from a feedback cell to the cell.

| Feedback cells | Output-end cells at 1 connection | Output-end cells at 2 | Output-end cells within 3 | Encoder cells at 1 | Encoder cells within 3 |
|---|---|---|---|---|---|
| 16 | 12 to 26 | 66 to 106 | 237 to 250 of 256 | 2 to 4 | 82 to 113 |
| 32 | 26 to 42 | 104 to 174 | 244 to 256 | 10 to 24 | 149 to 218 |
| **64 (first form)** | **43 to 63** | **160 to 173** | **all 256** | **25 to 38** | **211 to 249** |
| 128 | 100 to 114 | 142 to 148 | all 256 | 52 to 69 | 232 to 248 |

## 13. Worked example: one byte through the first form (s = 32; form (a))

The starting values are invented for the example. The previous emission was "7", and byte "3" arrives after "5". The feedback is therefore a "no", and the no line is active at step 0.

- **Feedback cell F** receives the no line (strength 1) and is excitatory. Its one ordinary source s1 has output 0.1, w 0.25 and e 0.10. Its v is 0.02, p 0.02, θ 0.05, and its output at the previous step was 0.
- **Encoder cell E.** Sources:
  - line "3": w 0.125, e 0.02;
  - line "5": w 0.125, e 0.40;
  - inhibitory h, output 0.25: w 0.25, e 0.25;
  - excitatory c, output 0.125: w 0.5, e 0.125;
  - F: w 0.25, e 0.

  Its v is 0.04, p 0.05, θ 0.02, and its previous activity was 0.03.
- **Inner cell I.** Sources:
  - E: w 0.5, e 0.06;
  - excitatory q, output 0.25: w 0.25, e 0.25;
  - inhibitory g, output 0.125: w 0.25, e 0.125.

  Its v is 0.10, p 0.12, θ 0.05, and its previous activity was 0.06.
- **Output-end cell O,** line "7", emitted with activity 0.60 at the previous reading step. Sources:
  - excitatory u, output 0.5 at steps 0 to 15 and 0.125 at steps 16 to 31: w 0.75, e 0.50;
  - inhibitory f, output 0.25: w 0.25, e 0.25;
  - F: w 0.25, e 0.

  Its v is 0.70, p 0.3125 and θ 0.10.

**By hand.**

*Step 0, F:*
- A = 1 · 1 + 0.25 · 0.10 = 1.025.
- v = 0.5 · 0.02 + 0.5 · 1.025 = 0.5225, and x = 0.4725.
- Its line does not change. It changes nothing by formula 7, because its previous activity was 0.

*Step 0, E:*
- A = 0.125 − 0.0625 + 0.0625 + 0 = 0.125, and m = 0.075.
- v = 0.0825, and x = 0.0625.
- Formula 7, with the receiving factor m · x(−1) = 0.075 · 0.03 = 0.00225:
  - h: −(1/8) · (−1) · 0.00225 · 0.25 = +0.0000703;
  - c: −(1/8) · 0.00225 · 0.125 = −0.0000352.

*Step 0, O:*
- A = 0.375 − 0.0625 + 0 = 0.3125, and m = 0, so no change.
- v = 0.5063, and x = 0.4063.

*Step 1, O* (F's output 0.4725 arrives):
- A = 0.375 − 0.0625 + 0.25 · 0.4725 = 0.4306, and m = 0.4306 − 0.3125 = 0.1181.
- Formula 7, with receiving factor 0.1181 · 0.4063 = 0.0480:
  - u: −(1/8) · 0.0480 · 0.5 = −0.0030;
  - f: +0.0015.

  The context that led to the wrong "7" is weakened.

*Step 1, E* (F's output arrives):
- A = 0.2431, and m = 0.1908.
- h: +0.000373, and c: −0.000186.

**Selected steps (computed with formulas 1 to 12 in the order of sentence 118).**

| Step | F: x | E: A | E: m | E: x | I: A | I: m | I: x | O: A | O: m | O: x |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.4725 | 0.1250 | 0.0750 | 0.0625 | 0.0462 | −0.0737 | 0.0231 | 0.3125 | 0.0000 | 0.4062 |
| 1 | 0.2210 | 0.2431 | 0.1908 | 0.1433 | 0.0625 | −0.0552 | 0.0186 | 0.4306 | 0.1181 | 0.3662 |
| 2 | 0.0959 | 0.1801 | 0.1218 | 0.1518 | 0.1030 | −0.0130 | 0.0370 | 0.3659 | 0.0497 | 0.3131 |
| 3 | 0.0340 | 0.1487 | 0.0866 | 0.1402 | 0.1072 | −0.0083 | 0.0486 | 0.3339 | 0.0161 | 0.2700 |
| 8 | 0.0000 | 0.1243 | 0.0521 | 0.1061 | 0.0851 | −0.0275 | 0.0427 | 0.3099 | −0.0076 | 0.2026 |
| 16 | 0.0000 | 0.1239 | 0.0401 | 0.1055 | 0.0842 | −0.0221 | 0.0446 | 0.0304 | −0.2855 | 0.0561 |
| 24 | 0.0000 | 0.1236 | 0.0308 | 0.1064 | 0.0847 | −0.0167 | 0.0499 | 0.0315 | −0.2205 | 0.0000 |
| 31 | 0.0000 | 0.1234 | 0.0245 | 0.1072 | 0.0851 | −0.0129 | 0.0544 | 0.0315 | −0.1766 | 0.0000 |

What happens over the window:
- The "no" weakens O's connection from u by 0.0030 at step 1 and 0.0011 at step 2.
- From step 8, O's arrival falls below its prediction, and u is strengthened until O's activity reaches 0 at step 17, after which formula 7 no longer moves O's strengths.
- O ends the window with u 0.75099, f 0.24949 and F 0.25014.
- E's lines stay at 0.125, and I's strengths end at E 0.5003, q 0.2510 and g 0.2495.
- O is below 1/2 at φ = 31.

## 14. The choices

The first form is the simplest that a living cell or tissue can do and that the roadmap's section 5 allows. Further forms are in the order to be opened. In the last column, "yes" means the cell sets the value itself in the first form.

| No. | Choice | First form | Further forms, in order | Numbers | Cell sets |
|---|---|---|---|---|---|
| C1 | Parts | One | Two (106) | — | No |
| C2 | Gate | None | Retention; input and output (unsettled) | — | Possibly |
| C3 | Kinds | 1/4 inhibitory at random; output-end cells excitatory; feedback cells by sentence 87 | 1/8; 1/2 | 1/4 | No |
| C5 | Strength | Graded, 0 to 1 | Counted in synapses | — | Yes (both ends) |
| C6 | Number of connections | Starts at 8, then the cell's own | Start 4 or 16 | 8 | Yes |
| C7 | Starting sources | 8 at random within the neighbourhood | Bias toward smaller x | — | No |
| C8 | Encoder cells receive from cells | Yes | No | — | No |
| C10 | Output | min(1, max(0, v − θ)) | Mismatch in the output, signed by two kinds of cell | — | No |
| C11 | τ_v | 2 | 1; 4 with s = 64 | 2 | Possibly |
| C12 | Predictions | One | One per part | — | No |
| C13 | Prediction | Running average, τ = 32 | One part predicting the other | τ | No |
| C14 | Comparison | Signed difference | Magnitude; above a level | — | No |
| C15 | Strength rule | Formula 7 at all connections between cells | Coincidence rule with a sliding threshold; rate η/16 at output-end cells | 1/8 | Possibly |
| C16 | Traces | Exponential, τ = 32 | 64 or 128 for per-answer feedback | 32 | Possibly |
| C17 | Giving up | Age ≥ P_s and (w or ū < 1/128) | Product of trace and mismatch near 0 | 1/128; P_s | Partly |
| C18 | Making | θ = 0, ā < a*, P_s since the last making or the start; random source; w = 1/8 | Toward coincident activity through a chemical signal | 1/8 | Yes (when) |
| C19 | Who adjusts | Strength by both ends; either end ends it; the receiving side starts it (144) | The sending side also starts connections | — | Yes |
| C20 | Passages | Formed by traces | Drawn positions | — | Yes |
| C21 | Response to a byte | One line for each value | Bit lines | — | No |
| C22 | Encoder cells predict | Yes | No | — | No |
| C23 | Cells | 2,048 | 4,096; 8,192 | 2,048 | No |
| C24 | Steps per byte | 32, all at once | 64; 16 with τ_v = 1; random order | 32 | No |
| C25 | Start | All 0 | Random states | — | No |
| C26 | Chemical release | None | On a feedback line; above running average; passed on at most once | — | Possibly |
| C27 | Region | Not used | Neighbourhood; falling with distance | — | No |
| C28 | Chemical effect | Not used | Scales the rate; permits making or giving up; permits lasting change | — | Possibly |
| C29 | Activity control | Formula 9 at all cells | Exempt output-end cells; synaptic scaling | 1/128; 1/8 | Yes (θ) |
| C30 | Space | Two dimensions, unit square, radius 1/8 | One dimension; radius 1/4 with s = 16; three dimensions; longer along x; k nearest | 1/8 | No |
| C31 | Lasting change | One component | Fast and slow; gated by a chemical signal; age-protected | — | Partly |
| C32 | Input lines | 256, active for the whole window | Bit lines | — | No |
| C33 | Conversion | Most active output-end cell above 1/2; ties lowest | Competition by inhibitory cells, any single line above 1/2, no byte if several | 1/2 | No |
| C34 | When emitted | Every position, at φ = s − 1 | When a line exceeds the level | — | No |
| C37 | Feedback content | Yes and no lines for each position | Per line (109); per answer (110, 111); graded | — | No |
| C38 | No emission | "No"; wrong for scoring | — | — | No |
| C39 | Timing | φ = 0, one step | After the answer's last byte | — | No |
| C40 | Route to the ends | Through feedback cells only, at both ends | Lines directly to output-end cells; more feedback cells | — | No |
| C41 | Receiving factor in formula 7 | m_i · x_i(k − 1) | m_i · ε_i; m_i alone | — | Yes |
| C42 | Relay by activity | None added | Kept out of the emission (102) | — | No |
| C43 | Return connections | As the neighbourhood gives | Set share from larger x, 2 to 128 | — | Yes |
| C44 | Relay and prediction | Not opened before step 3 | Two parts; chemical factor | — | No |
| C45 | Stream | Bytes only | — | — | No |
| C46 | Long-term memory | Only by the cells' own changes | — | — | No |
| C47 | Grid | 2 to 128 and reciprocals; cells beyond | — | — | No |
| C48 | Electrical synapses | None | Between neighbours of one kind | — | No |
| C49 | Exclusions | Sentence 82 | Mr. Xu may restate | — | No |
| C50 | Difference modulo 89 | Circular / 44 | Plain / D | — | No |
| C51 | Seed | One recorded seed per tissue | — | — | No |
| C52 | Section 15 | Forms (a) and (b), both run at step 2 | For other tasks, what the brief names | 64 more in (b) | No |
| C53 | Lines | Random points on x = 0 and x = 1 | Within strips 1/8 wide | — | No |
| C54 | Driver of an output line | Nearest eligible cell, ties by draw order | Several lines per cell | — | No |
| C55 | Recording step for step 1 | Reading step | As the brief names | — | No |
| C56 | Feedback cells | 64, random coordinates, half yes and half no; no-cells excitatory and yes-cells inhibitory; fixed line at strength 1 | 128; kinds at random; several lines per cell | 64 | No |

C4, C9 and C36 were retired earlier, and C35 is merged into C52.

## 15. Order of opening across choices

**Step 1 fails** (cells at 1, 2 or 4 connections from the input lines are silent or constant, or the bytes back cannot be told from their outputs):
1. C29.
2. C17 and C18.
3. C15, second form.
4. C11 with C24.
5. C30.
6. C16.
7. C3.
8. C6.
9. C1 with C13.
10. C2.

**Step 2 fails** (the ends do not come to give the coming byte above the frequencies of bytes):
1. C56: 128 feedback cells.
2. C41: m_i · ε_i.
3. C33: competition by inhibition.
4. C40: lines directly to output-end cells.
5. C15: η/16 at output-end cells.
6. C29: exempt output-end cells.
7. C16.
8. C56: kinds at random.
9. C37 per-line form. This needs a change of decision D3 and is not opened without it.
10. The order for step 1.

**Step 3 fails** (cells at 1, 2 and 4 connections from either end, or from feedback cells, do not correct themselves, or the tissue does not exceed the control tissue without relay):
1. C43.
2. C56 (more feedback cells).
3. C42.
4. C16.
5. C44 with C1.
6. C24 with C11.
7. C26 to C28.
8. C10.

## 16. Numbers fixed by us in the first form

**Numbers varied on the grid (16).**

| No. | Number | Value | Choice | Set by the cell? |
|---|---|---|---|---|
| 1 | Cells | 2,048 | C23 | No |
| 2 | Feedback cells | 64 | C56 | No |
| 3 | Radius | 1/8 | C30 | No |
| 4 | Share of inhibitory cells | 1/4 | C3 | No |
| 5 | Starting number of connections | 8 | C6 | **Yes after the start** |
| 6 | Starting strength | 1/8 | C5, C18 | No |
| 7 | τ_v | 2 steps | C11 | Possibly later |
| 8 | τ | 32 steps | C13, C16 | Possibly later |
| 9 | η | 1/8 | C15 | Possibly later |
| 10 | η_θ | 1/128 | C29 | No |
| 11 | a* | 1/8 | C29 | Possibly later |
| 12 | P_s | 128 steps | C17, C18 | No |
| 13 | Giving-up level | 1/128 | C17 | No |
| 14 | Steps per byte | 32 | C24 | No |
| 15 | Emission level | 1/2 | C33 | No |
| 16 | Feedback cells in form (b) | 64 more | C52 | No |

P_r of the earlier draft is no longer needed, since the running average of the feedback is withdrawn with formula 14.

**Values fixed by decision, convention or unit.**
- 256 input lines, 256 output lines and 256 emission lines (decision D1).
- Two yes and no lines.
- Two dimensions of unit extent, and faces at x = 0 and x = 1.
- Feedback-line strength 1.
- Maximum strength, maximum activity, slope and line value, each 1.
- Half yes and half no.
- Magnitude coding 1 − d/2.
- The denominators 44 and D.
- Starting values 0.
- The seed.

**Quantities set by the cells.** The number of connections. The strengths, set jointly by the two ends. θ, ā and ū.

## 17. Settlement of the fourth reading

| Item | Settled by |
|---|---|
| G1 Emission blind to the last byte | s = 32 and τ_v = 2; shares 0.975 to 0.999 (116); timing recomputed (97, 119); example recomputed (section 13) |
| G2 Two rules undoing each other | By section 16, one strength set by both ends (42, 49, 50). Formula 14 is withdrawn and the feedback enters as an arrival from feedback cells (93) |
| G3 "No" equals silence | Separate yes and no lines to feedback cells (86); magnitude coding without a zero when present (111); per-line "was wrong" lines (109) |
| G4 A shared factor | What moves for yes and for no (91); a shared factor cannot point right for an active cell that did not emit (92); receiving factor x_i(k − 1) (49); per-line form not excluded (109) |
| G5 Lines changed by formula 14 | Lines change by no rule (63); formula 14 withdrawn |
| G6 Two readings of section 15 | Both forms of C52, both run at step 2 (94 to 96) |
| G7 Growth at the start | "Or since the start of life" (55); first makings at step 128 (57) |
| G8 Conversion over the whole face | Kept, named as the one comparison over many cells, with the learned competition as the further form; agreement with section 10 left open (73) |
| G9 Marks and facts | Sentence 4 split (4, 5); bound with x_out (7); shortest path 8 to 9 (section 12); share of lines (66); clause on cells that cannot reach a* (57); sentence 96 replaced (91, 92); sentence 150 (139); sentences 106 and 108 (101, 104); roadmap draft 2 references removed |
| F6 | Settled by G2 to G4 |
| F12 | Section 15 |
| Undecided 1: r = 0 | Separate lines (86) |
| Undecided 2: lines and formula 14 | Not changed; formula 14 withdrawn (63) |
| Undecided 3: making at the start | Sentence 55 |
| Undecided 4: first position | No line active (86, 95, 118) |
| Undecided 5: two kinds at one step | Summed in the arrival (112) |
| Undecided 6: q = −1 | "Was wrong" line (109) |
| Undecided 7: largest difference | D from the brief; d = 1 for no number (111) |
| Undecided 8: ties | Draw order (6) |
| Undecided 9: recording step | Reading step (117) |
| Richer-feedback gaps (zero and negative coding; two kinds at once; normalisation) | Sentences 109, 111 and 112 |
| P as two numbers | P_s stays; P_r is no longer needed (section 16) |
| Section 16 (1), strength by both ends | Sentences 42, 49, 50 and 144; C19 restated |
| Section 16 (2), feedback cells | Sentences 81, 85 to 93; C56; starting counts in section 12 |

## 18. What the definition still does not determine

1. Whether the first form learns: steps 1 to 3. Two risks are named. The cancelling rule may flatten what inner cells carry, and too few end cells may come to receive from feedback cells (section 12 gives the starting counts).
2. On paper, a yes or no passed on alike has the right sign only for the cell that emitted (sentence 92).
3. Which reading of section 15 Mr. Xu intends. Both forms are run at step 2.
4. Whether the output end is reached through feedback cells as the input end is. Section 16 does not say, and the first form takes the same route by our choice (sentence 89).
5. Whether a conversion that is not learned agrees with section 10.
6. How relay combines with prediction (step 3), and how change becomes lasting (step 4).
7. The wording of the rulings behind C45 and C46.

## References

- Bao, Chan and Merzenich (2001), Nature 412: 79-83.
- Bittner and colleagues (2017), Science 357: 1033-1036.
- Lillicrap, Cownden, Tweed and Akerman (2016), Nature Communications 7: 13276.
- Matsumoto and Hikosaka (2007), Nature 447: 1111-1115.
- Schultz, Dayan and Montague (1997), Science 275: 1593-1599.

Textbook facts (Dale's principle, homeostatic plasticity, the requirement of postsynaptic depolarisation for plasticity, cerebellum-like cancellation, electrical synapses, spine growth and elimination) are not referenced. Page numbers are from memory and should be checked before citation.
