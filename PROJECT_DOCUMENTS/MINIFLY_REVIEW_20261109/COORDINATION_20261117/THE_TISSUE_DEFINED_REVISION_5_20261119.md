# The tissue defined, revision 5 (19 November 2026, written from 00:34 UTC by the clock of this machine)

**Change of 01:18 UTC, 19 November 2026, entered directly (Mr. Xu, `WHAT_A_NEURON_HAS_20261118.md`, section 19).** The rule for withdrawing a connection (formula 11, choice C17) is changed in its first form for every run after stage A of E63. A connection is withdrawn when what arrives over it has been of no use to the receiving cell for a period: the receiving cell keeps, for each incoming connection from a cell, a running average over 128 steps of the product of that input's trace (less its own running average) and the cell's own activity (less its running average); when the magnitude of that average stays below 1/128 of the largest such magnitude among the cell's connections for 128 steps, the connection is withdrawn. Withdrawal by strength below 1/128 stays as well. Making a connection (formula 12) is unchanged: a cell below its target tries a cell of its neighbourhood at random. The number of a cell's connections is then the balance of trying and withdrawing. The two numbers are on the grid and are listed with the numbers fixed by us. Why: under revision 5 as written, connections are made and almost never withdrawn, so that every cell would come to read its whole neighbourhood; stage A of E63 runs the rule as written and its connection counts are the record of that.

**Note by the reviewer (00:46 UTC, 19 November 2026), not part of the agent's text.** Two points from `CONFIRMATION_OF_TASK_STEP_1_BRIEF_REVISION_2_20261119.md`: sentence 123 has been given one added sentence (the sliding threshold of the coincidence form starts at a*, as sentence 53 says); and the kind of the magnitude feedback cell (sentence 113 and the table of choices) is disputed between this definition and the brief of step 1, is used at neither step 1 nor step 2, and is to be settled in the brief of the step that first uses it.

This file replaces revision 4 (`THE_TISSUE_DEFINED_REVISION_4_20261118.md`), which is not changed. It folds in errata E1 to E10, so no table of errata is needed. E1 to E9 come from the fifth reading (`FIFTH_READING_OF_THE_TISSUE_DEFINED_20261118.md`). E10 and findings 5, 6 and 10 come from the check of the brief of step 1 (`CHECK_OF_TASK_STEP_1_BRIEF_20261119.md`). The file also enters section 17 of `WHAT_A_NEURON_HAS_20261118.md` and takes the changed step 1 of the roadmap. It is task 0.3 of `TASK_GRAPH_20261119.md`. The only programs run were `python3 -I` with the standard library, for sections 12 and 13. They used the stream and birth of the check script `check_step1_brief.py`, copied without change, with seed 73129.

## Summary

Changes from revision 4:
1. **E10.** A connection is given up only when its strength is below 1/128. The test on the sending side's running average, which removed about 98 in 100 starting connections at step 128, becomes a further form, and ū is withdrawn.
2. **E4.** A feedback cell has its feedback line only. It has no ordinary sources, no strength change, no control of its activity and no growth.
3. **E5 not applied.** Feedback reaches the input end only through dedicated feedback cells. This is our choice, on Mr. Xu's words in section 16 (sentence 87).
4. **Coincidence rule written out.** The second form of the strength rule (C15) is now a formula, with a slow sliding threshold for excitatory inputs and the homeostatic sign for inhibitory inputs (check, finding 5). Only inner cells take it; encoder and output-end cells keep the cancelling rule (finding 6).
5. **E3 restricted.** The kinds of feedback cells are not reversed for output-end cells (finding 6).
6. **Section 17 entered.** A connection may run through other cells. This is an [S] sentence and choice C57, with two forms and neither excluded.
7. **E1, E2, E6 to E9 folded in.** Changes are in sentences 50, 93, 97 and 101, in section 12, and in the order of opening: C15 is first for step 1, and a* = 1/32 is added (finding 10).
8. **Worked example extended.** A second part follows two cells over the first 160 steps.
9. **Check of E10** (`python3 -I`, seed 73129, first form with E4 and E10). All 1,856 starting connections of the 232 cells at one connection from the input lines survive the first 1,024 steps, and so do all 1,512 of the 189 cells at two connections. No connection anywhere in the tissue is given up in that time. The smallest strength reached by any starting connection between cells is 0.0455, against the level of 1/128 ≈ 0.0078. Every cell other than the feedback cells makes one connection at step 128, and 13,875 connections are made by step 1,024, about 7 for each cell (finding 10).

Counts: 152 sentences: [S] 28, [F] 62, [C] 62. There are 53 choices.

## 0. Marks and terms

**Marks.**
- [S]: Mr. Xu's words, quoted from sections 7 to 17 of `WHAT_A_NEURON_HAS_20261118.md`. "Reported" means the section gives his words only in the reviewer's wording.
- [F]: follows by necessity, for the reason given.
- [C]: our choice, tabled in section 14. A ruling whose words are not quoted is marked [C], and the sentence says so.

The bracket after each mark gives the sentence's number in revision 4 ("r4 n"), or "new".

**Terms.**
- *Cell*: a nerve cell of the tissue.
- *Coordinate*: the point given to a cell at random at the start of its life.
- *Input face* and *output face*: the sides x = 0 and x = 1 of the space.
- *Input line*: a fixed channel, not a cell, active (value 1) while the present byte has one particular value.
- *Output line*: a fixed channel through which one output-end cell drives the emission of one byte value.
- *Feedback line*: a fixed channel through which the world gives feedback to one feedback cell.
- *Feedback cell*: a cell that receives a feedback line (sentence 85). A *no-cell* receives the no line, and a *yes-cell* the yes line.
- *Encoder cell*: a cell, not a feedback cell, with at least one input line in its neighbourhood.
- *Output-end cell*: a cell that drives an output line.
- *Inner cell*: any other cell.
- *Step*: one exchange of outputs among all cells.
- *Window*: the s steps from one byte's arrival to the next. Steps within it are numbered φ = 0, …, s − 1, and the byte arrives at φ = 0.
- *Reading step*: φ = s − 1.
- *Emission*: the byte given out for a position, or no byte. *Attempt*: the emission where, by the experimenter's reading, an answer is due.
- *Feedback*: what the world gives about an emission or an answer. It includes yes and no.
- *Distance from the input lines* or *from the output end*: the number of connections on the shortest path.
- *Return connection*: a connection to a cell with a smaller coordinate x.
- *Sending side* and *receiving side* of a connection: the parts of the sending and receiving cells at that connection (in a living synapse, the presynaptic terminal and the postsynaptic spine).
- *Part* of a cell: a region at which inputs from some sources arrive and are combined.
- *Birth*: the tissue at step 0. *Starting connections*: the connections it has then.
- *Living counterpart*: a structure or process in living tissue that does what the sentence describes.

## 1. Cells and space

1. [F] (r4 1) Nothing in the tissue learns apart from its nerve cells, because "没有专门的'学习者'" ("there is no dedicated learner", section 8) and the statement lets nothing else act.
2. [S] (r4 2) "Apart from the byte-level encoder, all that any nerve cell can do is ..." (section 8). There is an output end and an input end, and "these two also have to learn, through learning, how to take in and how to give out" (section 10).
3. [F] (r4 3) Each cell is an encoder cell, an inner cell, an output-end cell, or a feedback cell, and all are nerve cells.
4. [S] (r4 4) "为什么一开始不能给每个细胞一个随机生成的坐标呢？" (section 15).
5. [F] (r4 5) Each cell is therefore given a coordinate at random at the start, and near and far are distances between coordinates; section 15 records the question as his ruling. Placing the lines on two opposite faces lays down one axis, and that is our choice (C53).
6. [C] (r4 6; C30, C53, C54) The first form of the space and lines:
    - The unit square 0 ≤ x, y ≤ 1, with 2,048 cells placed uniformly at random; 64 of them, chosen at random, are feedback cells.
    - The 256 input lines are at random points on x = 0, and the 256 output lines at random points on x = 1.
    - Each output line, in byte order, is driven by the nearest cell that is neither a feedback cell nor already driving a line. Ties are broken by draw order.
    - A neighbourhood is every cell, and for a cell that is not a feedback cell every input line, within distance 1/8, excluding the cell itself.
7. [F] (r4 7) A cell at x lies at least about 8x connections from the input lines, and at least about 8(x_out − x) connections from the output end, where x_out ≥ 0.83 is the x of the nearest output-end cell. Distances are computed by the experimenter for reading only.
8. [S] (r4 8) A cell receives input from other cells, sends from that input and its present state an output to the cells it is connected to, and keeps adjusting which cells it is connected to (section 8).
9. [S] (r4 9) A cell has "its present state" and "the state it has stored" (section 8).
10. [F] (r4 10) What a cell holds is its only record of the past (section 8).
11. [S] (r4 11) "A million bytes cannot be kept as a window" (section 8).
12. [S] (r4 12) Separate parts of a cell may be used (section 7, point 1, reported).
13. [C] (r4 13; C1) A cell has one part.
14. [S] (r4 14) "The gated cell may stay" (section 7, point 1, reported). A living counterpart of three gates is unsettled.
15. [C] (r4 15; C2) A cell has no gate.
16. [C] (r4 16; C3) Each cell is excitatory or inhibitory for life, an idealisation of Dale's principle. One in four, at random, is inhibitory, except that output-end cells are excitatory and feedback cells take the kind given in sentence 89.
17. [F] (r4 17) A cell's output is never below zero.
18. [S] (r4 18) A cell acts and adjusts only within its own neighbourhood (section 9, reported).
19. [C] (r4 19; C48) Electrical synapses are left out.

## 2. What reaches a cell, and what it sends

20. [S] (r4 20) A cell receives "input from other cells" (section 8).
21. [S] (r4 21) "细胞应该可以得到误差的回馈" (section 13).
22. [F] (r4 22) Only these can reach a cell:
    - the outputs of cells connected to it;
    - for an encoder cell, input lines;
    - for a feedback cell, its feedback line, and in the first form nothing else (sentence 85);
    - chemical signals of its region.
23. [S] (r4 23) A cell does not in general read the bytes that have passed (section 8, "未必能够读到已有的字节").
24. [F] (r4 24) Bytes enter through input lines, and feedback enters only through feedback cells.
25. [S] (r4 25) What arrives differs from cell to cell, and it may result from earlier bytes (section 8).
26. [F] (r4 26) A cell's sources lie within its neighbourhood.
27. [C] (r4 27; C5) A strength is graded between 0 and 1.
28. [C] (r4 28; C7, C8) At birth each cell other than a feedback cell draws 8 sources at random from its neighbourhood: without replacement, never itself, and never twice. An encoder cell draws from lines and cells together.
29. [S] (r4 29) A cell sends an output "to the cells it is connected to" (section 8).
30. [F] (r4 30) A cell sends the same output to all its targets, including its output line.
31. [F] (r4 31) A cell sends no electrical output backward along a chemical synapse. Loops are not excluded.
32. [F] (r4 32) Return connections exist, because neighbourhoods extend in all directions.
33. [C] (r4 33; C43) There are no return connections other than those the neighbourhood gives.
34. [C] (r4 34; C10) A cell's output is its activity (formula 3).
35. [S] (r4 35) A cell can release a chemical signal into a region near it (section 9, reported).
36. [S] (new) "我们不排除细胞A和B之间的连接可能经过细胞C，D， E等等。具体强度在这种情况下如何确定 … 可以进一步讨论" (section 17). In translation: the connection between cells A and B may run through cells C, D, E and so on, and how strengths are determined in that case is open for discussion.
37. [C] (new; C57) How a connection that runs through other cells is set is choice C57, with two forms, neither excluded:
    - **First form:** each direct connection is set by its own two ends (formula 7). The influence of one cell on another is the sum of its effects over all paths between them. Cells on a path bear on it only through their own connections.
    - **Second form:** a third cell bears on a connection that is not its own through a local chemical signal released in that connection's region, and opposed signals add.

    Section 17 leaves the matter open, and so does this definition.

## 3. Prediction

38. [S] (r4 36) A cell's work is "预测从其他细胞得来的新来的信息" (section 8).
39. [S] (r4 37) It "cannot simply be taken as" expecting the next byte (section 8, "不能简单认为是").
40. [F] (r4 38) A prediction is formed before its arrival and is compared with the arrival in the same cell at the same step.
41. [F] (r4 39) The task's stream requires the tissue to keep information about earlier bytes in the states and outputs of cells.
42. [F] (r4 40) A cell must hold its arrival, a prediction, the mismatch, and traces of its inputs, if its mismatch is to fall by adjustment.
43. [C] (r4 41; C12 to C14) The prediction is the running average of the cell's own arrival, and the mismatch is the signed difference. What is learned is held in the strengths, which come to cancel the foretold part of the arrival. This has a living counterpart in cerebellum-like structures, where unchanging afferents carry the signal, and is unsettled in cortex.

## 4. Formulas of the first form

All are [C] except sentences 44 and 52. For a cell i:
- j ranges over its sources;
- σ_j = +1 for an excitatory cell or a line, and −1 for an inhibitory cell;
- z_j(k) is a line's value at step k, or a cell's output x_j(k − 1).

44. [S] (r4 42) "细胞之间连接的强度是连接的两端共同决定的，是局部的量" (section 16).
45. [C] (r4 43; formula 1) A_i(k) = Σ_j σ_j · w_ij(k) · z_j(k).
46. [C] (r4 44; formula 2; C11) v_i(k) = (1 − 1/τ_v) · v_i(k − 1) + (1/τ_v) · A_i(k), with τ_v = 2.
47. [C] (r4 45; formula 3; C10) x_i(k) = min(1, max(0, v_i(k) − θ_i(k))).
48. [C] (r4 46; formula 4; C13) p_i(k + 1) = (1 − 1/τ) · p_i(k) + (1/τ) · A_i(k), with τ = 32.
49. [C] (r4 47; formula 5; C14) m_i(k) = A_i(k) − p_i(k).
50. [C] (r4 48; formula 6; C16) e_ij(k + 1) = (1 − 1/τ) · e_ij(k) + (1/τ) · z_j(k), held at the sending side of the connection.
51. [C] (r4 49; formula 7, the strength of a connection between cells; C15) Δw_ij = −η · σ_j · [m_i(k) · x_i(k − 1)] · e_ij(k), with η = 1/8.
    - The sending side contributes e_ij(k) and σ_j.
    - The receiving side contributes m_i(k) · x_i(k − 1).
    - The rule acts on every connection between cells into encoder, inner and output-end cells.
    - It does not act on connections from lines or into feedback cells.
52. [F] (r4 50, reworded; E6) Formula 7 is a function of quantities present at the connection. The receiving cell's mismatch and activity are taken to reach the synapse by the spread of potential into the dendrite, including back-propagating action potentials. Whether a signed mismatch of this kind is present at a living synapse is unsettled. On that understanding the rule meets sentence 44.
53. [C] (new; C15, second form; check findings 5 and 6) The coincidence form of C15 is as follows:
    - For a connection from an excitatory cell: Δw_ij = η · e_ij(k) · x_i(k − 1) · (x_i(k − 1) − ϑ_i(k)), with the sliding threshold ϑ_i(k + 1) = (1 − 1/4096) · ϑ_i(k) + (1/4096) · x_i(k)² / a*, and ϑ_i(0) = a*.
    - For a connection from an inhibitory cell: Δw_ij = η · e_ij(k) · (x_i(k − 1) − a*). This strengthens inhibition when the receiving cell is above its target and weakens it when below (Vogels and colleagues, 2011).
    - Only inner cells take this form. Encoder and output-end cells keep formula 7, so the kinds of feedback cells stay as in sentence 89 for the output end.
    - The threshold's time constant, 4,096 steps (128 bytes), is a number of this form on the grid.
    - Under this form inner cells do not use their prediction, so a result in its favour must be reconciled with sentence 38.
    - The living counterpart is the sliding threshold of Bienenstock, Cooper and Munro (1982) and homeostatic inhibitory plasticity.
54. [C] (r4 51; formula 8) After all the changes of a step, each strength is set to min(1, max(0, w)).
55. [C] (r4 52; formula 9; C29) θ_i(k + 1) = max(0, θ_i(k) + η_θ · (x_i(k) − a*)), with η_θ = 1/128, a* = 1/8 and θ starting at 0. It applies to all cells other than feedback cells, whose θ stays 0.
56. [C] (r4 53, corrected; formula 10; C17) ā_i(k + 1) = (1 − 1/P_s) · ā_i(k) + (1/P_s) · x_i(k), with P_s = 128 steps, starting at 0. Revision 4's sending-side average ū is withdrawn.
57. [C] (r4 54, corrected by E10; formula 11; C17, C19) A connection at least P_s steps old is given up when its strength is below 1/128. Both ends decide this through the strength they set together (sentence 44). A feedback cell's line is never given up. The test on the sending side's running average is a further form of C17.
58. [C] (r4 55; formula 12; C18, C6) A cell other than a feedback cell makes one new incoming connection at step k when all three of these hold:
    - θ_i = 0;
    - ā_i < a*;
    - k − k_last ≥ P_s, where k_last is the step of its last making, or 0 at birth.

    The source is drawn at random from its neighbourhood members that are not already its sources. Its strength starts at 1/8 and its trace at 0. If there is no candidate, nothing is made and k_last is not changed.
59. [F] (r4 56) A cell's number of connections is its own, as section 12 asks ("有没有办法让每个细胞自己决定要进行多少条连接"). Only the starting number, 8, is fixed by us.
60. [F] (r4 57, corrected) Every cell starts with θ = 0 and ā = 0 < a*, so every cell other than a feedback cell makes a connection at step 128. In the check of section 12 this was 1,984 cells. A cell whose activity stays below a* goes on adding one connection every P_s steps. A cell driven only during one byte value cannot reach a* = 1/8, because each byte value occurs at 1.4 to 6.2 per cent of positions, so such a cell keeps growing (check, finding 10). Whether growth keeps cells informative is read at step 1. The smaller target a* = 1/32 is the first further form of C29.

The order of operations is sentence 120.

## 5. The input end

61. [S] (r4 58) Encoder cells read bytes ("Apart from the byte-level encoder"). What one reads "may be only a collection of small passages of the input with gaps between them, and different cells read different ones" (section 8).
62. [F] (r4 59) The input end consists of the encoder cells.
63. [C] (r4 60; C32, C21) There are 256 input lines, and the line of the present byte is active for all s steps of its window.
64. [F] (r4 61) An encoder cell's passages are formed by its traces, its state and its connections.
65. [C] (r4 62; C20) There are no separate numbers for passages.
66. [C] (r4 63) Connections from input lines change only by being given up and made. They start at 1/8. Feedback lines are fixed at 1.
67. [C] (r4 64; C22) Encoder cells predict their arrivals.
68. [C] (r4 65; C23) There are 2,048 cells: 256 output-end cells, 64 feedback cells (128 in form (b) of C52), and 236 to 262 encoder cells in the draws of section 12.
69. [F] (r4 66) About 38 to 42 per cent of encoder cells' starting draws are lines.

## 6. The output end

70. [F] (r4 67) The tissue gives out bytes (section 10; the example in section 11 is "预测next byte出现错误").
71. [F] (r4 68) The conversion of output-line activity into a byte is fixed and does not learn.
72. [C] (r4 69; C34) The tissue emits at every position. The emission is read at the reading step.
73. [F] (r4 70) No cell has the coming byte at the reading step.
74. [S] (r4 71) The right answer is shown as input after the learner's own attempt (section 7, point 1, reported).
75. [C] (r4 72; C33) The emitted byte is that of the most active output-end cell at the reading step, if its activity exceeds 1/2. Otherwise no byte is emitted. Ties go to the lowest value.
76. [F] (r4 73) This is the one comparison over many cells in the first form, and it reaches cells only through the emission and the feedback. Whether a conversion that is not learned agrees with section 10 is left open (C33's further form is the learned alternative).
77. [C] (r4 74; C38) A position with no byte emitted receives "no", and for scoring it is wrong. Positions where nothing can be expected are not scored and still receive feedback.
78. [F] (r4 75) Whether the emitted byte is read back is form (b) of C52.

## 7. Feedback

79. [S] (r4 76) "所以你可以给整个系统连续的信号，如果输出端的信息和预期不符合（例如预测next byte出现错误），告诉系统它错了让它自己去纠正" (section 11).
80. [S] (r4 77) Cells should be able to receive the feedback of the error. Yes or no is acceptable for the prediction of the next byte, and other forms are to be accommodated (section 13, "细胞应该可以得到误差的回馈 … 应该可以容纳其他形式的回答").
81. [S] (r4 78) "我们可以把正确答案加上差多少的绝对值作为一个额外的反馈信息给它" (section 14).
82. [S] (r4 79) Information may be relayed from neuron to neuron, and a single backward pass over the whole network is not allowed (section 11, reported).
83. [S] (r4 80) "我们可以假设输入端可以得到输出端信息的反馈。在NLP任务中这是next byte prediction的结果，但在其他任务中可以是其他的" (section 15).
84. [S] (r4 81) "我们的输入端不是所有的细胞都收到yes/no，只有某些收到。可以认为有专门的细胞收到反馈信息之后再传送给输入端，具体如何传送需要学习" (section 16).
85. [C] (r4 85, corrected by E4; C56) There are 64 feedback cells with coordinates drawn at random like every cell's. In the first form each has its feedback line only, at fixed strength 1. It has no ordinary sources. Formulas 7, 9 and 12 do not apply to it, and its θ stays 0. It does not draw input lines and does not drive an output line. Feedback cells with ordinary sources are a further form.
86. [C] (r4 86; C37, C39) For each position there are two feedback lines:
    - the yes line is active (value 1) for one step at φ = 0 of the window in which the byte arrives, if the emission equalled that byte;
    - the no line is active then otherwise.

    At other steps both are 0, and at the first position neither is active. Half of the feedback cells, chosen at random, receive the yes line and half the no line. Yes, no and silence therefore differ at a feedback cell without knowledge of φ.
87. [C] (new; E5 not applied) "Only some cells of the input end receive the yes or no" (sentence 84) is met by the next sentence of section 16: dedicated cells receive the feedback and pass it on. No encoder cell receives a feedback line. This reading is our choice. The reading in which some encoder cells receive the line directly is a further form of C56.
88. [C] (new; E9) Feedback cells are present from birth in every tissue of every step. At step 1 their lines are never active, because there is no correction yet.
89. [C] (r4 87, with E3 restricted; C56) No-cells are excitatory and yes-cells inhibitory, a pairing tied to the cancelling sign of formula 7 at the receiving cell. Output-end cells keep formula 7 in every form of C15 (sentence 53), so this pairing is kept for the output end. Whether inner cells that receive from feedback cells under the coincidence form need the reversed kinds is left to the brief of step 2. The living counterpart: worse outcomes excite cells of the lateral habenula, and better ones excite dopamine cells. The pairing of kinds is not a finding about those cells.
90. [F] (r4 88) Feedback reaches other cells only over ordinary connections of cells that have drawn a feedback cell as a source. These connections are made, kept, given up and changed by formulas 7, 11 and 12, and that is how the passing on is learned. Nothing lays down in advance which cells of either end they reach.
91. [C] (r4 89; C40) The output end is reached in the same way, through feedback cells only. Section 16 speaks of the input end only, so this is our choice.
92. [F] (r4 90) What a cell that receives from a feedback cell has: at φ = 1 its arrival rises, from a no-cell, or falls, from a yes-cell. The change enters its mismatch, and by formula 7 each of its connections changes in proportion to the sending side's trace and to its own activity at the previous step.
93. [F] (r4 91, with E7) At the steps at which the feedback cell's output arrives (φ = 1 to about 4), for a receiving cell that was active:
    - **No:** the arrival rises above the prediction, and formula 7 weakens the excitatory inputs, and strengthens the inhibitory inputs, whose traces are high, which are those of the context that led to the emission.
    - **Yes:** the reverse.
    - A cell with x = 0 at the previous step does not change.

    The net change over the window also includes the change formula 7 makes at the other steps, which in the worked example is of the same size.
94. [F] (r4 92) A yes or no passed on alike cannot point the right way for an active cell that did not cause the emission. The per-line form (sentence 111) points the right way for each output-end cell and is not excluded.
95. [F] (r4 93, with E2) Over many positions, and in expectation, an output-end cell that takes a no-cell becomes less driven in contexts followed by a "no", and one that takes a yes-cell more driven in contexts followed by a "yes". At a single position the feedback's change is of the size of the change formula 7 makes in the rest of one window. At birth 20 to 32 output-end cells take a yes-cell and 15 to 39 a no-cell (fifth reading). Learning therefore begins mainly by elimination.

### 7.1 The two readings of section 15

96. [F] (r4 94) "输出端信息的反馈" admits two readings: yes or no, or the emitted byte.
97. [C] (r4 95; C52) Both are forms of C52, and both are run at step 2 beside the tissue named by roadmap draft 3.
    - **Form (a):** the feedback cells of sentences 85 to 89 only.
    - **Form (b):** in addition, 64 further feedback cells with coordinates at random and kinds as for other cells. Each has 8 of the 256 emission lines, drawn at random, at strength 1/8, and nothing else. Formulas 7, 9 and 12 do not apply to them. The line of the emitted byte is active for one step at φ = 0. No line is active if no byte was emitted, or at the first position.
98. [F] (r4 96) Where feedback cells pass the feedback is not laid down in either form. In form (a) a receiving cell has a rise or a fall at φ = 1. In form (b) it may also have an arrival that depends on the emitted byte, one step after the arriving byte appears on the input lines.
99. [F] (r4 97) At a receiving cell, the factor x_i(k − 1) at φ = 1 is its activity at φ = 0, half carried over from the reading step. Traces keep (31/32)^32 ≈ 0.36 of the context from the start of the window.
100. [F] (r4 98) Feedback cells lie anywhere in the square, so the input end is not the last part reached.
101. [F] (r4 99) Feedback comes from the world to feedback cells and is not a backward pass.
102. [C] (r4 100; C52) In other tasks, the brief names the output end's information, which enters through feedback cells in the same way.

### 7.2 Elsewhere in the first form

103. [F] (r4 101) A cell more than one connection from every feedback cell receives feedback's effect only as changed outputs of the cells between, mixed with context.
104. [C] (r4 102; C42) No relay by a change of activity is added. If one is added, it must be kept out of the emission (a change at φ = 0 has fallen to (1/2)^31 by the reading step).
105. [F] (r4 103) With the sign of formula 7, a relayed rise lowers a cell's drive in that context along paths whose signs multiply to +1, and raises it along paths whose signs multiply to −1. The balance is read at step 3.
106. [F] (r4 104) A change reaches a cell d connections from a feedback cell at least d steps later. Its traces must span at least 2d + 1 steps, which is 9 for d = 4, where (31/32)^9 ≈ 0.75 remains. With r outgoing connections per cell, d relays reach at most r^d cells.
107. [F] (r4 105) With one part, a relayed change enters the arrival. With two parts it can be kept apart.
108. [C] (r4 106; C44, C1) How a cell combines relay and prediction is not opened before step 3. The two-part form assigns return connections and connections from feedback cells to the second part.
109. [C] (r4 107; C40) The further routes are feedback lines directly to output-end cells, and more feedback cells.

### 7.3 Richer feedback

110. [F] (r4 108) Each richer form enters, in the first place, through feedback cells.
111. [C] (r4 109) In the per-line form, each output line has a "should have been active" line, active for the line of the arriving byte, and a "was wrong" line, active for the emitted line if it differs. They are active at φ = 0 for one step. They end at feedback cells, one for each line, or, in the further variant, at the output-end cell of that line.
112. [C] (r4 110) In the per-answer form, the right answer is given by the lines of sentence 111 at each position of the answer, after the emission for that position.
113. [C] (r4 111) The magnitude goes on one line to dedicated feedback cells, at φ = 0 after the answer's last byte, for one step.
     - The line carries 1 − d/2.
     - d = |a − b| / D, with D the largest difference the brief's answers allow. For arithmetic modulo 89, d is the circular difference divided by 44.
     - d = 1 if no readable number was given.
     - The kind of a magnitude feedback cell is excitatory.
114. [C] (r4 112) Kinds of feedback reaching a cell at one step enter its arrival together.
115. [F] (r4 113) A three-byte answer reaches back 98 steps, where (31/32)^98 ≈ 0.04 remains. Briefs using per-answer feedback open C16.

## 8. Time and order of operations

116. [C] (r4 114; C45) The input is one byte after another, with nothing marking an item. This ruling is not quoted, so it is marked as our choice.
117. [C] (r4 115; C24, C11) s = 32, τ_v = 2, and all cells compute at once from the previous step's outputs.
118. [F] (r4 116) A byte's effect at a cell n connections in is delayed by about n · τ_v steps. For a chain of n cells with unit gain, the share reached at φ = 31 is 0.999, 0.996, 0.990 and 0.975 for n = 8, 9, 10 and 11, against at most 0.027 with τ_v = 4 and s = 16. Output-end cells lie 8 to 11 connections in, so the emission can depend on the byte just read.
119. [C] (r4 117; C55) For step 1, outputs are recorded at the reading step.
120. [C] (r4 118) The order of operations at step k, with φ = k mod s:
     1. If φ = 0, the world sets the input lines and compares the new byte with the emission read at the previous step. It makes the yes line or the no line active, and in form (b) the emitted byte's emission line. At the first position, and at every position of step 1, none of these is active.
     2. Every cell computes A (formula 1).
     3. Every cell computes m (formula 5).
     4. Every cell computes v and x (formulas 2 and 3).
     5. Every cell other than a feedback cell computes the changes of formula 7 (or of sentence 53) from the traces held before the step and from x(k − 1), and applies formula 8.
     6. Every cell updates its traces, p, θ and ā (formulas 4, 6, 9 and 10).
     7. Every cell other than a feedback cell gives up connections (formula 11) and then makes one (formula 12), in the order of cell index.
     8. If φ = s − 1, the emission is read (sentence 75).
121. [F] (r4 119) Within one byte, the following hold:
     - at φ = 0 the byte reaches encoder cells and the feedback lines reach feedback cells;
     - at φ = 1 feedback cells' outputs reach their targets;
     - a cell n connections in receives most of the byte's effect by about φ = 2n;
     - at φ = 31 the emission reflects the byte just read.
122. [F] (r4 120) Nothing outside the cells resets their states, and there is one tissue meeting one stream.
123. [C] (r4 121; C25) At birth all states, traces, predictions, thresholds and running averages are 0, once. The one exception is the sliding threshold ϑ of the coincidence form (sentence 53), which starts at a*. (Sentence added by the reviewer at 00:46 UTC, 19 November 2026, to remove a conflict with sentence 53 that the confirmation of the brief of step 1 found.)
124. [F] (new) Bytes four and eight back lie 128 and 256 steps back, where traces keep (31/32)^128 ≈ 0.017 and (31/32)^256 ≈ 0.0003 and the state keeps nothing. Holding them needs sustained recurrent activity, which the roadmap asks for at step 3. Step 1 reads them and reports them.

## 9. Chemical signal, lasting change, reactivation and numbers

125. [S] (r4 122) A cell's chemical signal is local, and how each receiving cell reads it is its own matter (sections 9 and 7, reported).
126. [F] (r4 123) A chemical signal released by a cell never reaches the whole tissue.
127. [C] (r4 124; C26 to C28) In the first form no cell releases a chemical signal.
128. [F] (r4 125) A chemical signal received alike by a region cannot choose which connection changes. That is why the second form of C57 makes opposed signals add rather than select.
129. [C] (r4 126; C46) Long-term memory forms in learning and not by an act of a program. This ruling is on record but not quoted, so it is marked as our choice.
130. [F] (r4 127) No program marks a change as lasting, freezes connections, or copies the tissue.
131. [C] (r4 128; C31) Each strength has one component.
132. [F] (r4 129) No program passes over an item after it has ended. The tissue's own reactivation is neither excluded nor imposed.
133. [S] (r4 130) "我们预先给定的参数越少，长期来看越好", and "从2, 4, 8, 16,32,64,128等等取一些数字" (section 12).
134. [C] (r4 131; C47) Fractions are read on the grid 1/2 to 1/128. Numbers of cells go beyond 128 by powers of two.

## 10. What does not exist

135. [C] (r4 132) No cell is given the right answer in the first form. The per-line and per-answer forms give it after the emission.
136. [F] (r4 133) No cell reads every cell.
137. [F] (r4 134) No error passes backward through forward connections.
138. [F] (r4 135) The quantities given to cells from outside are input lines and feedback lines, and only feedback cells receive the latter.
139. [F] (r4 136) No change of strength is computed outside the connection's two ends.
140. [F] (r4 137) There is no reset, no copy, no averaging over items, and no window beyond what states and traces keep.
141. [F] (r4 138) The only criterion over the whole tissue is the world's comparison.
142. [F] (r4 139, reworded; E6) The first form uses only quantities that a living cell holds, or that are taken to reach its connections (sentence 52).
143. [C] (r4 140; C51) One generator with a recorded seed.
144. [C] (r4 141) The numbers fixed by us are those of section 16.
145. [C] (r4 142) The orders of opening are those of section 15.
146. [F] (r4 143) Every erratum E1 to E10 is folded into the sentences named in section 17.
147. [C] (r4 144; C19) A strength is set by both ends (formula 7). It ends when it falls below 1/128 (formula 11). The receiving side starts a connection (formula 12).
148. [F] (r4 145) A feedback cell's sign is the same at every connection it makes.
149. [C] (r4 146; C56) There are 64 feedback cells, a number on the grid.
150. [F] (new) In the first 160 steps of the first form, formula 11 gives up none of the 15,872 starting connections, because none is P_s steps old before step 128 and none falls below 1/128 by step 160 (section 13, part 2).
151. [F] (new) In the same 160 steps, formula 12 acts once at every one of the 1,984 cells other than feedback cells, at step 128. It adds 1,984 connections and removes none of the birth's.
152. [F] (new) In the same 160 steps, formula 7 changes 6,188 of the 15,118 starting connections between cells; 12 of them by more than 0.0001. The 754 starting connections from lines are changed by no rule.

**Counts.** 152 sentences: [S] 28, [F] 62, [C] 62.

## 12. Counts (first form, `python3 -I`)

**Distances at birth, three draws with 2,048 cells and radius 1/8.**

| Distance | From the input lines | From the output end |
|---|---|---|
| 1 | 237 to 247 | 175 to 196 |
| 2 | 163 to 188 | 172 to 189 |
| 3 | 175 to 204 | 192 to 207 |
| 4 | 177 to 237 | 194 to 215 |
| 5 to 9 | 191 to 225 each | 141 to 238 each |
| 10 | 138 to 219 | 21 to 60 |
| 11 | 3 to 36 | 1 to 7 |

Further facts:
- Output-end cells lie at x ≥ 0.83, 8 to 11 connections from the input lines, and the shortest path between the faces is 8 to 9.
- With the birth of the check script (seed 73129), the counts are 232 cells at distance 1, 189 at 2 and 193 at 4.

**End cells reached from feedback cells at birth.**
- 64 feedback cells: 44 to 66 output-end cells take a feedback cell as a source (20 to 32 a yes-cell, 15 to 39 a no-cell), and 22 to 46 encoder cells do, about 10 to 20 per cent.
- 128 feedback cells: 44 to 81 output-end cells take a yes-cell and 30 to 77 a no-cell.

**Check of E10, seed 73129, first form with E4 and E10, steps 0 to 1,023.**

| Cells | Starting connections | Surviving at step 1,024 | Smallest surviving starting strength | Connections per cell at step 1,024 |
|---|---|---|---|---|
| Distance 1 (232) | 1,856 | 1,856 | 0.1178 | 15 |
| Distance 2 (189) | 1,512 | 1,512 | 0.1032 | 15 |
| All cells | 15,872 | 15,872 | 0.0455 (between cells) | — |

No connection is given up in the 1,024 steps. 13,875 connections are made, one for each cell other than a feedback cell at each of steps 128, 256, …, 896, with 13 cells missing one. With revision 4's formula 11, the check of the brief found that 15,617 to 15,646 of the 15,872 starting connections were certainly given up at step 128.

## 13. Worked example

### Part 1: one byte (s = 32; form (a))

The starting values are invented for the example. The previous emission was "7", and byte "3" arrives after "5", so the no line is active at step 0.

- **Feedback cell F:** a no-cell, excitatory, with its line only, and v = 0, θ = 0.
- **Encoder cell E.** Sources:
  - line "3": w 0.125, e 0.02;
  - line "5": w 0.125, e 0.40;
  - inhibitory h, output 0.25: w 0.25, e 0.25;
  - excitatory c, output 0.125: w 0.5, e 0.125;
  - F: w 0.25, e 0.

  Its v is 0.04, p 0.05, θ 0.02, and its previous activity 0.03.
- **Inner cell I.** Sources:
  - E: w 0.5, e 0.06;
  - excitatory q, output 0.25: w 0.25, e 0.25;
  - inhibitory g, output 0.125: w 0.25, e 0.125.

  Its v is 0.10, p 0.12, θ 0.05, and its previous activity 0.06.
- **Output-end cell O,** line "7", emitted with activity 0.60. Sources:
  - excitatory u, output 0.5 at steps 0 to 15 and 0.125 after: w 0.75, e 0.50;
  - inhibitory f, output 0.25: w 0.25, e 0.25;
  - F: w 0.25, e 0.

  Its v is 0.70, p 0.3125 and θ 0.10.

**By hand.**
- *Step 0, F:* A = 1, v = 0.5 and x = 0.5.
- *Step 0, E:*
  - A = 0.125, m = 0.075, v = 0.0825 and x = 0.0625.
  - The receiving factor is 0.075 · 0.03 = 0.00225.
  - h changes by +(1/8) · 0.00225 · 0.25 = +0.0000703, and c by −0.0000352.
- *Step 0, O:* A = 0.3125 and m = 0, so there is no change; x = 0.4063.
- *Step 1, O* (F's output 0.5 arrives):
  - A = 0.375 − 0.0625 + 0.125 = 0.4375, and m = 0.125.
  - The receiving factor is 0.125 · 0.4063 = 0.0508.
  - u changes by −(1/8) · 0.0508 · 0.5 = −0.0032, and f by +0.0016.

**Selected steps (formulas 1 to 12 in the order of sentence 120).**

| Step | F: x | E: A | E: m | E: x | I: A | I: m | I: x | O: A | O: m | O: x |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.5000 | 0.1250 | 0.0750 | 0.0625 | 0.0462 | −0.0737 | 0.0231 | 0.3125 | 0.0000 | 0.4062 |
| 1 | 0.2500 | 0.2500 | 0.1976 | 0.1467 | 0.0625 | −0.0552 | 0.0186 | 0.4375 | 0.1250 | 0.3697 |
| 2 | 0.1250 | 0.1874 | 0.1288 | 0.1571 | 0.1047 | −0.0113 | 0.0379 | 0.3730 | 0.0566 | 0.3183 |
| 3 | 0.0625 | 0.1559 | 0.0934 | 0.1464 | 0.1099 | −0.0057 | 0.0504 | 0.3409 | 0.0228 | 0.2761 |
| 8 | 0.0020 | 0.1252 | 0.0518 | 0.1081 | 0.0868 | −0.0263 | 0.0449 | 0.3105 | −0.0081 | 0.2042 |
| 16 | 0.0000 | 0.1238 | 0.0391 | 0.1051 | 0.0840 | −0.0227 | 0.0443 | 0.0302 | −0.2864 | 0.0555 |
| 24 | 0.0000 | 0.1235 | 0.0300 | 0.1060 | 0.0845 | −0.0172 | 0.0496 | 0.0314 | −0.2213 | 0.0000 |
| 31 | 0.0000 | 0.1233 | 0.0239 | 0.1069 | 0.0850 | −0.0133 | 0.0541 | 0.0314 | −0.1772 | 0.0000 |

What happens over the window:
- The "no" weakens u by 0.0032 at step 1 and by 0.0013 at step 2.
- Later in the window u is strengthened again, while O's arrival is below its prediction and O is still active.
- At the end, O's strengths are u 0.7504, f 0.2498 and F 0.2502; E's are h 0.2554 and c 0.4973, with the lines at 0.125; I's are E 0.5003, q 0.2510 and g 0.2495.

### Part 2: the first 160 steps of life (seed 73129, the whole tissue, first form with E4 and E10)

The stream begins "w58<V". The two cells followed are:
- **cell 497**, at distance 1, which has the line of "w";
- **cell 333**, at distance 2, which takes cell 497 as a source.

Strengths are given to four decimals.

| Step | Cell 497: connections (of the birth) | Cell 497: strengths | Cell 497: activity | Cell 333: connections (of the birth) | Cell 333: strengths | Cell 333: activity |
|---|---|---|---|---|---|---|
| 0 | 8 (8) | all 0.1250 | 0.0625 | 8 (8) | all 0.1250 | 0.0000 |
| 32 | 8 (8) | all 0.1250 | 0.0625 | 8 (8) | all 0.1250 | 0.0193 |
| 64 | 8 (8) | all 0.1250 | 0.0000 | 8 (8) | all 0.1250 | 0.0000 |
| 96 | 8 (8) | all 0.1250 | 0.0000 | 8 (8) | all 0.1250 | 0.0000 |
| 127 | 8 (8) | all 0.1250 | 0.0021 | 8 (8) | all 0.1250 | 0.0003 |
| 128 | 9 (8) | all 0.1250 | 0.0021 | 9 (8) | all 0.1250 | 0.0003 |
| 129 | 9 (8) | all 0.1250 | 0.0021 | 9 (8) | all 0.1250 | 0.0041 |
| 160 | 9 (8) | all 0.1250 | 0.0000 | 9 (8) | two at 0.1249, the rest 0.1250 | 0.0322 |

The thresholds of both cells are 0 throughout. The rules that remove or add connections act in these 160 steps as sentences 150 and 151 state:
- formula 11 gives up none of the 15,872 starting connections;
- formula 12 adds one connection to each of the 1,984 cells other than feedback cells, at step 128, and removes none.

The strength rule, formula 7, changes 6,188 of the 15,118 starting connections between cells, but only 12 of them by more than 0.0001 (sentence 152). By step 1,024 both cells have 15 connections, all eight of their starting connections intact.

## 14. The choices

The first form is the simplest that a living cell or tissue can do and that the roadmap's section 5 allows. Further forms are in the order to be opened. In the last column, "yes" means the cell sets the value itself in the first form.

| No. | Choice | First form | Further forms, in order | Numbers | Cell sets |
|---|---|---|---|---|---|
| C1 | Parts | One | Two (108) | — | No |
| C2 | Gate | None | Retention; input and output (unsettled) | — | Possibly |
| C3 | Kinds | 1/4 inhibitory; output-end cells excitatory; feedback cells by 89 | 1/8; 1/2 | 1/4 | No |
| C5 | Strength | Graded, 0 to 1 | Counted in synapses | — | Yes (both ends) |
| C6 | Number of connections | Starts at 8, then the cell's own | Start 4 or 16 | 8 | Yes |
| C7 | Starting sources | 8 at random within the neighbourhood | Bias toward smaller x | — | No |
| C8 | Encoder cells receive from cells | Yes | No | — | No |
| C10 | Output | min(1, max(0, v − θ)) | Mismatch in the output | — | No |
| C11 | τ_v | 2 | 1; 4 with s = 64 | 2 | Possibly |
| C12 | Predictions | One | One per part | — | No |
| C13 | Prediction | Running average, τ = 32 | One part predicting the other | τ | No |
| C14 | Comparison | Signed difference | Magnitude; above a level | — | No |
| C15 | Strength rule | Formula 7 | Sentence 53 at inner cells; rate η/16 at output-end cells | 1/8; 4,096 (second form) | Possibly |
| C16 | Traces | Exponential, τ = 32 | 64 or 128 for per-answer feedback | 32 | Possibly |
| C17 | Giving up | Age ≥ P_s and w < 1/128 | Also when the sending side's running average is below 1/128 | 1/128; P_s | Yes (by the strength) |
| C18 | Making | θ = 0, ā < a*, P_s since the last making or birth; random source; w = 1/8 | Toward coincident activity through a chemical signal | 1/8 | Yes (when) |
| C19 | Who adjusts | Strength by both ends; the receiving side starts it (147) | The sending side also starts connections | — | Yes |
| C20 | Passages | Formed by traces | Drawn positions | — | Yes |
| C21 | Response to a byte | One line for each value | Bit lines | — | No |
| C22 | Encoder cells predict | Yes | No | — | No |
| C23 | Cells | 2,048 | 4,096; 8,192 | 2,048 | No |
| C24 | Steps per byte | 32, all at once | 64; 16 with τ_v = 1; random order | 32 | No |
| C25 | Start | All 0 | Random states | — | No |
| C26 | Chemical release | None | On a feedback line; above running average; passed on at most once | — | Possibly |
| C27 | Region | Not used | Neighbourhood; falling with distance | — | No |
| C28 | Chemical effect | Not used | Scales the rate; permits making or giving up; permits lasting change | — | Possibly |
| C29 | Activity control | Formula 9 at all cells but feedback cells, a* = 1/8 | a* = 1/32; exempt output-end cells; synaptic scaling | 1/128; 1/8 | Yes (θ) |
| C30 | Space | Two dimensions, unit square, radius 1/8 | One dimension; radius 1/4 with s = 16; three dimensions; longer along x; k nearest | 1/8 | No |
| C31 | Lasting change | One component | Fast and slow; gated by a chemical signal; age-protected | — | Partly |
| C32 | Input lines | 256, active for the whole window | Bit lines | — | No |
| C33 | Conversion | Most active output-end cell above 1/2 | Competition by inhibitory cells | 1/2 | No |
| C34 | When emitted | Every position, at φ = s − 1 | When a line exceeds the level | — | No |
| C37 | Feedback content | Yes and no lines for each position | Per line (111); per answer (112, 113); graded | — | No |
| C38 | No emission | "No"; wrong for scoring | — | — | No |
| C39 | Timing | φ = 0, one step | After the answer's last byte | — | No |
| C40 | Route to the ends | Through feedback cells, at both ends | Lines directly to output-end cells; more feedback cells | — | No |
| C41 | Receiving factor in formula 7 | m_i · x_i(k − 1) | m_i · ε_i; m_i alone | — | Yes |
| C42 | Relay by activity | None added | Kept out of the emission (104) | — | No |
| C43 | Return connections | As the neighbourhood gives | Set share from larger x | — | Yes |
| C44 | Relay and prediction | Not opened before step 3 | Two parts; chemical factor | — | No |
| C45 | Stream | Bytes only | — | — | No |
| C46 | Long-term memory | Only by the cells' own changes | — | — | No |
| C47 | Grid | 2 to 128 and reciprocals; cells beyond | — | — | No |
| C48 | Electrical synapses | None | Between neighbours of one kind | — | No |
| C49 | Exclusions | Sentence 82 of revision 4, unchanged: the answer before the emission; a change computed outside the cell; a backward pass; a chemical signal reaching the whole tissue; any reset | Mr. Xu may restate | — | No |
| C50 | Difference modulo 89 | Circular / 44 | Plain / D | — | No |
| C51 | Seed | One recorded seed per tissue | — | — | No |
| C52 | Section 15 | Forms (a) and (b), both run at step 2 | For other tasks, what the brief names | 64 more in (b) | No |
| C53 | Lines | Random points on x = 0 and x = 1 | Within strips 1/8 wide | — | No |
| C54 | Driver of an output line | Nearest eligible cell, ties by draw order | Several lines per cell | — | No |
| C55 | Recording step for step 1 | Reading step | As the brief names | — | No |
| C56 | Feedback cells | 64, random coordinates, line only, half yes and half no; no-cells excitatory and yes-cells inhibitory | 128; with ordinary sources; some encoder cells receiving the line directly (87); kinds at random | 64 | No |
| C57 | Connection through other cells (section 17) | Each direct connection set by its own ends; influence summed over paths | A third cell's local chemical signal bears on a connection in its region, opposed signals adding | — | No |

C4, C9 and C36 were retired earlier, and C35 is merged into C52.

## 15. Order of opening across choices

**Step 1 fails** (cells at 1, 2 or 4 connections from the input lines are silent or constant, or the present byte and the bytes one and two back cannot be told from their outputs better than at birth, than a tissue whose strengths never change, and than the present byte alone; roadmap draft 3):
1. C15, second form (sentence 53).
2. C29: a* = 1/32.
3. C17 and C18.
4. C11 with C24.
5. C30.
6. C16.
7. C3.
8. C6.
9. C1 with C13.
10. C2.

The brief of step 1 runs C15's second form beside the first form from the start (E1).

**Step 2 fails:**
1. C56: 128 feedback cells.
2. C41: m_i · ε_i.
3. C33.
4. C40: lines directly to output-end cells.
5. C15: η/16 at output-end cells.
6. C29: exempt output-end cells.
7. C16.
8. C56: with ordinary sources; some encoder cells receiving the line directly; kinds at random.
9. C37 per-line form. This needs a change of decision D3.
10. The order for step 1.

The brief of step 2 reports the number of output-end cells reached from yes-cells and from no-cells at birth and at the end of life (E2).

**Step 3 fails:**
1. C43.
2. C56 (more feedback cells).
3. C42.
4. C57, second form.
5. C16.
6. C44 with C1.
7. C24 with C11.
8. C26 to C28.
9. C10.

## 16. Numbers fixed by us in the first form

**Numbers varied on the grid (16).**

| No. | Number | Value | Choice | Set by the cell? |
|---|---|---|---|---|
| 1 | Cells | 2,048 | C23 | No |
| 2 | Feedback cells | 64 | C56 | No |
| 3 | Radius | 1/8 | C30 | No |
| 4 | Share of inhibitory cells | 1/4 | C3 | No |
| 5 | Starting number of connections | 8 | C6 | **Yes after birth** |
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

**Numbers of further forms named in this file.** The time constant of the sliding threshold, 4,096 steps (sentence 53), and a* = 1/32 (C29).

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

**Quantities set by the cells.** The number of connections. The strengths, set by both ends. θ and ā.

## 17. Where each erratum and finding is folded in

| Item | Place |
|---|---|
| E1 (C15 first for step 1; both rules from the start) | Section 15, step 1; sentence 53 |
| E2 (expectation wording; report reach) | Sentence 95; section 15, step 2 |
| E3 (reversal of kinds) | Restricted by check finding 6: sentences 53 and 89 |
| E4 (feedback cells with their line only) | Sentences 22, 28, 55, 85, 120 |
| E5 (encoder cells receiving the line directly) | Not applied; our choice, sentence 87; a further form of C56 |
| E6 (what reaches the synapse; the sending side's share) | Sentences 52, 142; section 18, item 4 |
| E7 (net change over the window) | Sentence 93 |
| E8 (encoder cells reached at birth) | Section 12 |
| E9 (feedback cells at step 1) | Sentence 88; sentence 120, item 1 |
| E10 (giving up by strength only) | Sentence 57; ū withdrawn (sentence 56); checked in section 12 and section 13, part 2; sentences 150 to 152 |
| Check finding 5 (coincidence formula) | Sentence 53 |
| Check finding 6 (which cells keep which rule) | Sentences 53 and 89 |
| Check finding 10 (growth) | Sentence 60; C29 further form a* = 1/32, second in the opening order for step 1 |
| Check finding 13, item 1 (no candidate) | Sentence 58 |
| Section 17 | Sentences 36 and 37; C57 |
| Roadmap step 1, changed rule | Section 15, step 1; sentence 124 |

## 18. What the definition still does not determine

1. Whether the first form learns (steps 1 to 3). Three risks are named: the cancelling rule may flatten what inner cells carry; cells under their target keep growing (sentence 60); and few end cells receive from feedback cells at birth (sentence 95).
2. Whether a yes or no passed on alike suffices at step 2 (sentence 94).
3. Which reading of section 15 Mr. Xu intends. Both forms are run at step 2.
4. Whether Mr. Xu's "共同决定" ("determined jointly") gives the sending side more than its output trace (E6). How strengths are set when a connection runs through other cells, which section 17 leaves open (C57).
5. Whether a conversion that is not learned agrees with section 10.
6. How relay combines with prediction (step 3), and how change becomes lasting (step 4).
7. The wording of the rulings behind C45 and C46.

## References

- Bienenstock, Cooper and Munro (1982), Journal of Neuroscience 2: 32-48.
- Matsumoto and Hikosaka (2007), Nature 447: 1111-1115.
- Schultz, Dayan and Montague (1997), Science 275: 1593-1599.
- Vogels, Sprekeler, Zenke, Clopath and Gerstner (2011), Science 334: 1569-1573.

Textbook facts are not referenced. Page numbers are from memory and should be checked before citation.
