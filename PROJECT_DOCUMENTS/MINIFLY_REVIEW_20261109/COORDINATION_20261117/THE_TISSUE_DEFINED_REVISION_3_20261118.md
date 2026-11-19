# The tissue defined, revision 3 (18 November 2026, written from 23:15 UTC by the clock of this machine; the rulings of section 15, of 23:18 UTC, taken in at 23:25 UTC)

This file replaces revision 2 (`THE_TISSUE_DEFINED_REVISION_2_20261118.md`), which is not changed. It settles the twelve findings and the 28 open decisions of the third reading (`THIRD_READING_OF_THE_TISSUE_DEFINED_20261118.md`). It takes in Mr. Xu's rulings of 23:03, 23:04 and 23:18 UTC and his earlier ruling on the number of connections (sections 13, 14, 15 and 12 of `WHAT_A_NEURON_HAS_20261118.md`), and the changed decisions D2 and D3 of the roadmap (`ROADMAP_TO_END_TO_END_PERSISTENT_LEARNING_20261118.md`, section 9). Nothing has been run apart from the counting and arithmetic stated in sections 2.2 and 8, done with `python3 -I` and the standard library.

## Summary

1. **Feedback replaces "signal".** Feedback is defined to cover yes as well as no. It has three places to enter: one quantity for each byte position (yes or no, the first form); one quantity for each output line; and quantities for each answer (the right answer as bytes, and a magnitude such as the absolute difference). Each comes with its timing, the cells that receive it, and the traces it needs. Revision 2's sentence 90 and the words "not assumed" are withdrawn.
2. **Each cell is given a random coordinate** (Mr. Xu, section 15). The 2,048 cells lie in a unit square, with the input lines placed at random on one side and the output lines on the opposite side. A cell's neighbourhood is every cell within distance 1/8. In three random draws, every distance from 1 to 10 from the input lines and from the output end holds between 138 and 247 cells (section 2.2).
3. **The input end also receives feedback of the output end's information** (section 15). In the first form this is the same yes or no, given at the same step, so the input end is reached by correction at once rather than last (sentences 171 to 177).
4. **A cell sets its own number of connections.** It starts with 8. It gives up a connection that is weak or silent, and it makes a new one when its threshold is at zero and its activity is still below its target.
5. **Every formula is written down** (section 3), together with the order of operations within one step and within one byte (section 7). The sign of the strength rule is fixed: excitatory inputs that were active weaken when the arrival exceeds the prediction. Revision 2's sentences 96, 99 and 104 are re-marked or corrected.
6. **The output end can start from a tissue that is almost always wrong.** Its rule uses the feedback minus the cell's own running average of it, so a rare right emission strengthens the inputs that produced it. The emitted byte is that of the most active output line, if that line exceeds 1/2.
7. **Fixes for the third reading's arithmetic.** Traces have a time constant of 16 steps, enough for relay over four connections, and there are 16 steps per byte. Connections from input lines that stay silent are given up. Where no byte is emitted, the position counts as "no" for feedback and as wrong for scoring.
8. **The rest:** one order of opening across all choices for each of roadmap steps 1 to 3; a list of the 14 numbers that can be varied, plus fixed units and structural values; a worked example of one byte; and a table that settles each finding and each of the 28 decisions.
9. **Counts:** 177 numbered sentences, of which 29 are [S], 66 are [F] and 82 are [C]. There are 51 choices.

## 0. Sources, marks and terms

**Marks.**
- [S]: Mr. Xu's words, quoted from sections 7 to 14 of `WHAT_A_NEURON_HAS_20261118.md`. Where a section gives his words only in the reviewer's wording, without the original, the sentence says "reported in the reviewer's wording".
- [F]: follows by necessity. The reason is given. The reason may be a textbook fact about living nerve cells, used because the cells are nerve cells.
- [C]: our choice, numbered and listed in section 4. Where a ruling exists but its words are not quoted in sections 7 to 14, the sentence is marked [C] and says so.

Each sentence notes its number in revision 2 ("r2 n"), or "new".

**Terms.**
- *Cell*: a nerve cell of the tissue.
- *Coordinate*: the point of the space (sentence 4) given to a cell at random at the start of its life. *Distance in space* between two cells is the ordinary (Euclidean) distance between their coordinates.
- *Input face* and *output face*: the two opposite sides of the space, x = 0 and x = 1, on which the input lines and the output lines are placed (sentence 5).
- *Encoder cell*: a cell whose neighbourhood contains at least one input line. *Output-end cell*: a cell that drives an output line (sentence 5). *Inner cell*: any other cell.
- *Input line*: a fixed channel, not a cell, that is active while the present byte has one particular value. It corresponds to a receptor of a sense organ.
- *Output line*: a fixed channel, not a cell, whose activity, through the fixed conversion of sentence 79, decides whether the tissue emits one particular byte value. It corresponds to the nerve to a muscle.
- *Feedback line*: a fixed channel, not a cell, through which the world gives feedback to the cells that receive it. Its living counterparts are the climbing fibre of the cerebellum and a modulatory projection.
- *Step*: one exchange of outputs among all cells, numbered k = 0, 1, 2, and so on.
- *Window*: the s steps from the arrival of one byte to the arrival of the next. Steps within a window are numbered φ = 0 to s − 1, and the byte arrives at φ = 0.
- *Arrival* of a cell at a step: the sum of its inputs at that step, each multiplied by its strength and its sign (formula 1).
- *State*: everything a cell holds. This includes its connections, their strengths, its traces, its prediction, its threshold and its running averages.
- *Prediction* p: the quantity a cell holds before a step that stands for its arrival at that step.
- *Mismatch* m: the arrival minus the prediction.
- *Trace*: a quantity that follows a past activity and declines by a factor (1 − 1/τ) at each step.
- *Strength* w of a connection: a number between 0 and 1 by which the sending cell's output, or the line's activity, is multiplied at the receiving cell.
- *Threshold* θ: the level the cell's state must exceed before the cell is active.
- *Running average*: a trace with a long time constant P.
- *Emission*: the byte the tissue gives out for a position, or the absence of a byte.
- *Attempt*: the emission at a position where, by the experimenter's reading of the stream, an answer is due.
- *Feedback*: what the world gives about the tissue's emission at a position, or about an answer. It includes "yes" (the emission was right) as well as "no".
- *Distance from the input lines* of a cell: the number of connections on the shortest path from an input line to the cell (an encoder cell connected to a line has distance 1).
- *Distance from the output end* of a cell: the number of connections on the shortest path from the cell to an output-end cell (output-end cells have distance 0).
- *Return connection*: a connection from a cell to a cell whose coordinate x (the axis from the input face to the output face) is smaller, that is, from a cell nearer the output face to a cell farther from it. It is fixed by the coordinates and does not change during life.
- *Part* of a cell: a region of the cell (in a living cell, a dendritic branch or a set of branches) where inputs from some sources arrive and are combined.
- *Living counterpart*: a structure or process in living nerve tissue that does what the sentence describes.

## 1. The cells and their arrangement

### 1.1 What a cell is and where it lies

1. [F] (r2 1, re-marked) Nothing in the tissue learns apart from its nerve cells. The reason is that Mr. Xu says "没有专门的'学习者'" ("there is no dedicated learner", section 8), and the cells are the only parts of the tissue that the statement lets act.
2. [S] (r2 2) "Apart from the byte-level encoder, all that any nerve cell can do is ..." (section 8), and "there is an output end and an input end; these two also have to learn, through learning, how to take in and how to give out" (section 10).
3. [F] (r2 3) Each cell is therefore an encoder cell, an inner cell or an output-end cell, and all are nerve cells, because the cells of both ends learn (sentence 2) and nothing learns apart from nerve cells (sentence 1).
4. [S] (new) "为什么一开始不能给每个细胞一个随机生成的坐标呢？" ("why can each cell not be given a randomly generated coordinate at the start?", section 15). Each cell is given a coordinate drawn at random at the start. Near and far are distances between coordinates, and no ordered arrangement of cells is laid down.
5. [C] (new; C30, C53, C54) The first form of the space and of the placing of lines is as follows:
    - **Space.** The space is two-dimensional, the unit square 0 ≤ x ≤ 1, 0 ≤ y ≤ 1, without joined edges. Each of the 2,048 cells is given a coordinate drawn uniformly at random in it.
    - **Input lines.** The 256 input lines are placed at points (0, y) on the input face, with y drawn at random.
    - **Output lines.** The 256 output lines are placed at points (1, y) on the output face, with y drawn at random.
    - **Output-end cells.** Each output line, taken in the order of its byte value, is driven by the cell nearest to it that does not already drive a line; that cell is an output-end cell.
    - **Neighbourhood.** A cell's neighbourhood is every cell, and for a cell near the input face every input line, within distance ρ = 1/8 of its coordinate, excluding the cell itself.

    The further forms of the space, simplest first, are listed under C30: one dimension, two, three, and an extent longer along x.
6. [F] (new) A connection joins only cells within distance 1/8 in space, and the faces are a distance 1 apart. A cell at coordinate x therefore lies at least about 8x connections from the input lines and about 8(1 − x) from the output end, whatever connections it has. Distances of 1, 2 and 4 from both ends exist in every draw (counts in section 2.2). The distances are computed by the experimenter for reading only, and no rule of a cell uses them.
7. [S] (r2 4) A cell receives input from other cells, sends from that input and its present state an output to the cells it is connected to, and keeps adjusting which cells it is connected to (section 8: "只是", "only"). Sentences 37 and 92 state what sections 9 and 13 add to these acts: a local chemical signal, and the feedback of the error.
8. [S] (r2 5) A cell holds a state: section 8 speaks of "its present state" and of "the state it has stored".
9. [F] (r2 6) The state, as defined in section 0, is the only record of the past that a cell has, because "it can read only the state it has stored and the new information that reaches it through other cells" (section 8).
10. [S] (r2 7) The state is small compared with the stream: "a million bytes cannot be kept as a window" (section 8).
11. [S] (r2 8, re-marked) A cell may have separate parts that treat inputs from different sources differently. This is section 7, point 1, reported in the reviewer's wording.
12. [C] (r2 9; C1) In the first form a cell has one part.
13. [S] (r2 10, re-marked) A cell may be gated. This is section 7, point 1, reported in the reviewer's wording. A living counterpart of three gates, each a weighted sum of the cell's inputs, is unsettled.
14. [C] (r2 11; C2) In the first form a cell has no gate.
15. [C] (r2 12, re-marked; C3) Each cell is either excitatory or inhibitory for its whole life. Its strengths change but their sign does not. This is an idealisation of Dale's principle. In living tissue a cell releases the same transmitters at all its synapses, but the sign of their effect depends on the receptors, and the inhibitory transmitter GABA excites early in development.
16. [C] (new; C3) Among the cells that are not output-end cells, one in four, chosen at random, is inhibitory. All output-end cells are excitatory, as the living cells that drive muscles are.
17. [F] (r2 14) A cell's output is never below zero, because the output of a living nerve cell is its firing or its release of transmitter. A decrease can be signalled only as a fall below the cell's ongoing activity.
18. [S] (r2 16) A cell acts and adjusts only within its own neighbourhood. This is section 9, reporting in the reviewer's wording Mr. Xu's statement of 16 November ("a neuron acts and adjusts only within its own neighbourhood"). Sentence 5 gives "neighbourhood" its meaning in this tissue.
19. [C] (new; C48) Electrical synapses (gap junctions), which in living tissue pass the membrane potential in both directions, are left out of the first form.

### 1.2 What arrives at a cell and what it sends

20. [S] (r2 18, corrected) A cell receives input from other cells (section 8: "receive input from other cells").
21. [S] (r2 18, corrected) "细胞应该可以得到误差的回馈" ("cells should be able to receive the feedback of the error", section 13).
22. [F] (r2 21, corrected) The only things that can reach a cell are:
    - the outputs of the cells connected to it;
    - for an encoder cell, the activity of the input lines connected to it;
    - the chemical signals released in its region (sentence 37);
    - for a cell that receives feedback (sentence 105), the feedback through feedback lines.

    A cell receives other cells' states, strengths or inputs only as far as their outputs carry them.
23. [S] (r2 19) A cell does not in general read the bytes that have passed (section 8: "未必能够读到已有的字节", "is not necessarily able to read the bytes already passed").
24. [F] (r2 20, corrected) In the first form, bytes enter the tissue only through the input lines, read by encoder cells. A further form of the feedback (sentence 98) gives the right answer, after the attempt, to output-end cells.
25. [S] (r2 22) What arrives differs from cell to cell (section 8).
26. [S] (r2 23) What arrives may result from earlier bytes (section 8).
27. [F] (r2 24) A cell's sources lie within its neighbourhood (sentence 18).
28. [C] (r2 25; C5) A connection has a graded strength between 0 and 1. The Chinese "不断调整自己连接那些细胞" can be read either as "which cells it connects to" or as "its connections to those cells". A living connection is graded. Whether a single synapse has only a few discrete states is unsettled.
29. [C] (r2 27, 28; C7, C8) At the start of life each cell draws 8 sources at random from its neighbourhood: without replacement, never itself, and never the same source twice. For an encoder cell, the draw is from the input lines and the cells of its neighbourhood together, so encoder cells may also receive from cells.
30. [S] (r2 29) A cell sends an output, computed "from that input and its present state", "to the cells it is connected to" (section 8).
31. [F] (r2 30) A cell sends the same output to every cell it is connected to, and an output-end cell sends that same output to its output line, because a living nerve cell has one axon and its output enters every branch.
32. [F] (r2 31, reworded) A cell sends no electrical output backward along a chemical synapse. What reaches the cells that send to it, from it, is its ordinary output, if they are among its targets, and its chemical signal, if they lie in its region. Electrical synapses, which do conduct both ways, are left out by choice (sentence 19).
33. [F] (r2 32) A loop in which A sends to B and B sends to A is not excluded.
34. [F] (new; settles finding F5) A neighbourhood extends equally in all directions, so a cell may take a source whose coordinate x is larger than its own. Return connections therefore exist in the first form. In particular, cells near the output face may take output-end cells as sources.
35. [C] (r2 98, corrected; C43) The first form has no return connections other than those the neighbourhood gives in sentence 34. They are drawn, kept and given up like any other connection, and they count among the incoming connections of the cell that receives them.
36. [C] (r2 33; C10) In the first form a cell's output is its activity, computed from its arrival and state by formulas 2 and 3.
37. [S] (r2 34, re-marked) Beside its output, a cell can release a chemical signal into a region near it. This is section 9, reported in the reviewer's wording.

## 2. Prediction, and the counts of distances

### 2.1 Prediction

38. [S] (r2 35) What a cell does is to be put as predicting what arrives (section 8: "预测从其他细胞得来的新来的信息", "predicting the information newly arriving from other cells").
39. [S] (r2 36) It cannot simply be taken as expecting the next byte (section 8: "不能简单认为是'预期下一个到来的字节'").
40. [F] (r2 37) A prediction of an arrival is formed before that arrival, from what the cell has before it, because "新来的" means "newly arriving".
41. [F] (r2 38) Prediction and arrival must both be in the same cell at the same step for the two to be compared.
42. [F] (r2 39) In the stream of the task some bytes depend on bytes several positions earlier. An arrival that reflects them can be predicted only if information from the earlier bytes is present in the cell's state or its present arrival (sentence 23). This follows from the task's stream, not from the statement.
43. [F] (r2 40) The tissue must therefore keep information about earlier bytes in the states of cells and in the outputs they are sending.
44. [F] (r2 42, 43, 44) The cell must hold the arrival, a prediction, and their comparison, the mismatch. If the mismatch is to fall by the cell's own adjustment, the cell must also keep traces of the inputs active before the arrival.
45. [F] (r2 46) The mismatch is a quantity of the cell that formed it. Another cell receives a function of it only through the cell's output or through a chemical release that depends on it.
46. [C] (r2 47, 48, 50; C12, C13, C14) In the first form:
    - the cell holds one prediction, for its total arrival;
    - the prediction is the running average of its own past arrivals (formula 4);
    - the comparison is the signed difference (formula 5).

    What the cell learns is held in its strengths. By the rule of formula 6, its strengths come to cancel the part of its arrival that the inputs active just before it foretell. Where the arrival can be foretold, it comes to equal the prediction, and what remains in the arrival, and so in the output, is what the context did not foretell. A cell of this kind has a living counterpart in cerebellum-like structures. Whether cortical cells work this way is unsettled.
47. [F] (r2 49) With one part, arrival, prediction and mismatch lie at the same place in the cell.

### 2.2 Counts of cells at each distance in the first form

The distances were counted with `python3 -I` for three random draws (seeds 1, 2 and 3) of the first form. Each draw used 2,048 coordinates, 256 input and 256 output lines placed as in sentence 5, a neighbourhood radius of 1/8, and 8 starting sources for each cell (sentence 29). The count is breadth-first over the connections. The counts change during life as connections are made and given up. The lower bounds of sentence 6 do not change.

| Distance | From the input lines (three draws) | From the output end (three draws) |
|---|---|---|
| 0 | none | 256 |
| 1 | 237 to 247 | 175 to 196 |
| 2 | 163 to 188 | 172 to 189 |
| 3 | 175 to 204 | 192 to 207 |
| 4 | 177 to 237 | 194 to 215 |
| 5 | 191 to 209 | 175 to 210 |
| 6 | 192 to 218 | 199 to 238 |
| 7 | 192 to 213 | 173 to 237 |
| 8 | 193 to 216 | 185 to 202 |
| 9 | 200 to 225 | 141 to 183 |
| 10 | 138 to 219 | 21 to 60 |
| 11 | 3 to 36 | 1 to 7 |
| No path | none | 0 to 3 |

Further facts from the same draws:
- A neighbourhood holds 89 to 91 cells on average, and never fewer than 27.
- The number of cells with input lines in reach, which are the encoder cells, is 245 to 262.
- The output-end cells lie at x of 0.83 or more.
- The shortest path from the input lines to the output end is about ten to eleven connections.

In one dimension, the same 2,048 cells would need a radius of about 0.022 to give neighbourhoods of the same size, and the path between the faces would then be about 45 connections. A radius giving a path of about ten would give neighbourhoods of about 400 cells. This is why the first form takes two dimensions. A radius of 1/4 in two dimensions shortens the path to about five connections but gives neighbourhoods about four times larger; it is a further form of C30.

## 3. The rule of a cell: every formula of the first form

All of the following are [C], choices C10, C11, C13, C14, C15, C16, C17, C18, C29 and C6. The numbers are those of section 6. For a cell i, a source j is either a cell or, for an encoder cell, an input line. Write σ_j for the sign of source j (+1 for an excitatory cell or a line, −1 for an inhibitory cell). Write z_j(k) for what source j presents at step k: for a line, its activity L_j(k); for a cell, its output x_j(k − 1) at the previous step.

48. [C] (new; formula 1, arrival) A_i(k) = Σ_j σ_j · w_ij(k) · z_j(k).
49. [C] (new; formula 2, state; C11) v_i(k) = (1 − 1/τ_v) · v_i(k − 1) + (1/τ_v) · A_i(k), with τ_v = 4 steps. The state v may fall below zero, as a living membrane potential falls below rest under inhibition.
50. [C] (new; formula 3, activity, the transfer function; C10) x_i(k) = min(1, max(0, v_i(k) − θ_i(k))). The maximum activity is 1 and the slope is 1.
51. [C] (new; formula 4, prediction; C13) p_i(k + 1) = (1 − 1/τ) · p_i(k) + (1/τ) · A_i(k), with τ = 16 steps. The prediction for step k is p_i(k), held before step k.
52. [C] (new; formula 5, mismatch; C14) m_i(k) = A_i(k) − p_i(k).
53. [C] (new; formula 6, trace of an input, kept at the synapse; C16) e_ij(k + 1) = (1 − 1/τ) · e_ij(k) + (1/τ) · z_j(k), with τ = 16 steps. The trace used at step k is e_ij(k), which holds the inputs up to step k − 1.
54. [C] (new; formula 7, trace of the cell's own activity; C16) ε_i(k + 1) = (1 − 1/τ) · ε_i(k) + (1/τ) · x_i(k), with the same τ.
55. [C] (r2 55, sign fixed; formula 8, strength; C15) For every connection from a cell (not from an input line, sentence 70):
    - Δw_ij(k) = −η · σ_j · m_i(k) · e_ij(k), with η = 1/32.
    - When the arrival exceeds the prediction, excitatory inputs that were active weaken and inhibitory inputs that were active strengthen. When the arrival falls short, the reverse happens.
    - The rule applies alike to encoder, inner and output-end cells.
    - This is the sign of the living counterpart: synapses that weaken when sending and receiving activity coincide in the electrosensory lobe of mormyrid fish, and inhibitory synapses that strengthen when the receiving cell is too active in cortex. That cortical excitatory synapses follow this sign generally is unsettled.
56. [C] (new; formula 9, bounds) After all changes of a step are added (formulas 8 and 14), each strength is set to min(1, max(0, w)). A living synapse cannot reverse its sign and cannot grow without limit.
57. [C] (r2 15; formula 10, control of the cell's own activity; C29) θ_i(k + 1) = max(0, θ_i(k) + η_θ · (x_i(k) − a*)), with η_θ = 1/128, a* = 1/8 and θ_i(0) = 0. The target a* is a target for the mean of the activity. This rule applies to all cells, including output-end cells. Its living counterpart is homeostatic change of a cell's excitability, which is textbook.
58. [C] (new; formula 11, running averages; C17) ā_i(k + 1) = (1 − 1/P) · ā_i(k) + (1/P) · x_i(k), and, for each connection, ū_ij(k + 1) = (1 − 1/P) · ū_ij(k) + (1/P) · z_j(k), with P = 128 steps. ū_ij starts at 0 when the connection is made.
59. [C] (r2 58, corrected; formula 12, giving up; C17) A connection that has existed for at least P steps is given up when its strength is below 1/128, or when its input's running average ū_ij is below 1/128. The second condition gives up connections to input lines or cells that are almost never active (settles finding F9). No rule uses a mean of the cell's strengths.
60. [C] (r2 61, corrected; formula 13, making; C18, C6) A cell makes one new connection when its threshold is 0, its running activity ā_i is below a*, and at least P steps have passed since it last made one. The new source is drawn at random, with equal chances, from the members of its neighbourhood (cells and, for an encoder cell, lines) that are not already its sources. The new connection starts with strength 1/8, and its traces start at 0.
61. [F] (new) The number of a cell's connections is therefore set by the cell from its own quantities. It grows while the cell is too little active even with its threshold at zero, and it falls as connections become weak or silent. The starting number 8 is the only number of connections that we fix. This meets Mr. Xu's question "有没有办法让每个细胞自己决定要进行多少条连接" ("is there a way to let each cell decide for itself how many connections to make", section 12). The living counterparts are the growth of new spines when a cell is deprived of input, and the elimination of inactive synapses. Both are textbook in outline; their rules in detail are unsettled.
62. [F] (new) A cell that is silent at the start lowers its threshold by a*/128 per step. From a threshold of 0.1 it reaches 0 in about 102 steps, and from then on it makes a connection every P steps until it is active. No cell stays silent for life unless no member of its neighbourhood is ever active.

## 4. The input end

63. [S] (r2 63) Encoder cells read bytes; they are the one exception to sentence 23 (section 8: "Apart from the byte-level encoder").
64. [S] (r2 64) What an encoder cell reads may be a collection of small passages of the input with gaps between them, and different cells read different ones (section 8).
65. [F] (r2 65) The input end consists of the encoder cells, because bytes enter only through them in the first form (sentence 24).
66. [C] (r2 68; C32) There are 256 input lines, one for each byte value. The line of the present byte has activity 1 for all s steps of the window, and every other line has 0. This is decision D1 of the roadmap.
67. [F] (r2 69) An encoder cell can therefore read passages with gaps only through the traces of its lines (formula 6), the decline of its state (formula 2), and its connections from other cells. The span of its passages is set by τ and τ_v and by which lines and cells it is connected to.
68. [C] (r2 70, 71; C20, C21) In the first form there are no separate numbers for passages: an encoder cell's passages are formed as sentence 67 states.
69. [F] (r2 72) Encoder cells adjust what they take in by their own local learning, because the input end learns (sentence 2): they give up silent lines and draw new ones (formulas 12 and 13).
70. [C] (new; C15) Connections from input lines change only by being given up and made, not by formula 8. Their strength stays at its starting value of 1/8. A rule that cancels predictable input would otherwise weaken a line in proportion to its own sustained activity. The living counterpart is the primary afferent synapse of cerebellum-like structures, which is not plastic in this way.
71. [C] (r2 73; C22) Encoder cells predict their arrivals by formulas 4 and 5, and formula 8 applies to their connections from cells.
72. [C] (r2 74; C23) The first number of cells is 2,048, of which 256 are output-end cells. How many are encoder cells follows from the coordinates; it is 245 to 262 in the draws of section 2.2.

## 5. The output end

73. [F] (r2 75) What the tissue gives out is bytes, because the output end gives out (sentence 2), the tissue's world is a stream of bytes, and section 11 gives as its example "预测next byte出现错误" ("the next byte was predicted wrongly").
74. [F] (r2 76) The conversion of output-line activity into an emitted byte is fixed and does not learn, because nothing learns apart from the cells (sentence 1).
75. [F] (r2 79) An output-end cell is a cell like any other (sentences 7 and 48 to 62). The activity it sends to its output line is the activity it sends to every cell connected to it (sentence 31).
76. [C] (r2 80; C34) The tissue emits at every byte position. The emission for the coming position is read at step φ = s − 1 of the window, before the next byte arrives at φ = 0.
77. [F] (r2 81) No cell has the coming byte when the emission is read, because it arrives only at the next step (sentence 66). Where an answer is due is known to the experimenter and to no cell (sentence 133).
78. [S] (r2 82, re-marked) The right answer is shown as input after the learner's own attempt. This is section 7, point 1, reported in the reviewer's wording.
79. [C] (r2 77, corrected; C33) The emitted byte is the byte value y of the output-end cell with the greatest activity at φ = s − 1, if that activity exceeds 1/2. If no output-end cell exceeds 1/2, no byte is emitted. If two or more share the greatest activity, the lowest y is taken.
80. [F] (r2 78) The conversion of sentence 79 compares all output lines. It is admissible only as the fixed conversion of sentence 74, and it gives nothing back to the cells except through the emission and the feedback. Its living counterpart is competition among actions formed by inhibition among cells, which the conversion stands in for.
81. [C] (r2 84; C35) In the first form the tissue does not read its own emission back through its input end.
82. [C] (new; C38) A position at which no byte was emitted receives "no" as feedback, because a byte arrives at every position and no byte differs from it. For scoring, an attempt with no byte is wrong. Positions at which nothing can be expected are identified by the experimenter's reading of the stream, are not scored, and still receive feedback.

## 6. Feedback

### 6.1 Sentences

83. [S] (r2 85) "所以你可以给整个系统连续的信号，如果输出端的信息和预期不符合（例如预测next byte出现错误），告诉系统它错了让它自己去纠正。这本来就是我们persistent learning的目标" (section 11): the whole system can be given a continuous signal; if the information at the output end does not match what was expected, the system is told that it was wrong and corrects itself; this has been the aim of persistent learning all along.
84. [S] (new) "细胞应该可以得到误差的回馈，只是'yes/no'的形式也可以接受，但是应该可以容纳其他形式的回答。具体到byte level input reading，预测下一个字节的任务，yes/no的形式是可以接受的。但我们不排除未来会有更复杂的任务和更复杂的误差回馈存在" (section 13): cells should be able to receive the feedback of the error; yes or no is acceptable, and is acceptable for the prediction of the next byte; other forms should be accommodated, and more complex feedback is not excluded later.
85. [S] (new) "我们可以把正确答案加上差多少的绝对值作为一个额外的反馈信息给它" (section 14): the right answer, together with the absolute value of how far off it was, can be given as additional feedback.
86. [S] (r2 86) Information may be relayed from neuron to neuron so that a few passes cover the network, and a single backward pass over the whole network is not allowed. This is section 11, "What follows", reporting Mr. Xu's earlier words in the reviewer's wording. The figures "one cell to fifty, fifty to fifty times fifty" are in the roadmap, section 1, and are not quoted in sections 7 to 14.
87. [F] (new) Revision 2's sentence 90 (no form richer than yes or no for lines that were silent) and the words "not assumed" in its sentence 92 and choice C40 are withdrawn, because section 13 rules that cells can receive feedback and that other forms are to be accommodated.
88. [C] (new; C49) What remains excluded is fixed by our reading of sections 11, 13 and 14 together:
    - giving any cell the right answer for a position before the tissue's emission for that position has been read;
    - computing outside a cell the change of its strengths;
    - a single backward pass through the forward connections;
    - a chemical signal that reaches the whole tissue;
    - any reset.

    Feedback after the emission, including the right answer (section 14), is not excluded.
89. [S] (r2 87, re-marked) The world may make the comparison and give the result: "你可以给" ("you can give", section 11).
90. [C] (r2 87; C37) In the first form the comparison is made outside the tissue, between the emission read at φ = s − 1 of one window and the byte that arrives at φ = 0 of the next.
91. [C] (r2 88, re-marked; C37) "连续" can mean continuing in time or continuous in value. The first form takes it as continuing in time: feedback is given at every position for the whole life. The reading continuous in value is form (b) of C37.
92. [C] (new; C37, the first form) The feedback for a position is one quantity: r = 1 ("yes") if the emitted byte equals the arriving byte, and r = 0 ("no") otherwise, including when no byte was emitted (sentence 82). This is the form section 13 accepts for the prediction of the next byte, and roadmap step 2 uses it (decision D3).
93. [C] (new; C39) The feedback for a position is present at step φ = 0 of the window in which the byte arrives, for that one step. The interval from the reading of the emission (φ = s − 1 of the previous window) to the feedback is e = 1 step.
94. [C] (new; C40) In the first form, the feedback for a position reaches every output-end cell alike, through one feedback line. This is the roadmap's step 2 ("the signal acts on the cells at the output end only").
95. [C] (new; C41, formula 14) An output-end cell holds a running average of the feedback it receives: r̄_i ← (1 − 1/P) · r̄_i + (1/P) · r, updated at each position, with P = 128 and r̄_i starting at 0. At the step at which feedback is present, each of its connections from a cell changes by Δw_ij = η · σ_j · (r − r̄_i) · e_ij(k) · ε_i(k), with η = 1/32. This change is added to that of formula 8 before the bounds of formula 9 are applied. The feedback is not an arrival: it does not enter A, p or m, and it does not change the cell's activity at that step.
96. [F] (new; settles finding F6) Early in life r̄ is near 0. A "no" then changes strengths very little, and a rare "yes" changes by nearly η · e · ε the inputs of the cell that was most active, which is the cell whose line was emitted. So the output end can begin to learn from a tissue that is almost always wrong. When right emissions become common, r̄ rises, and a "no" weakens the active inputs of the cells that were active.
97. [C] (new; C37, C39, C40, C41; the further form for each output line) The feedback for a position may also be given to each output line, through a feedback line for each output-end cell, at φ = 0 for one step. In this form q_y = 1 for the line of the arriving byte; q_y = −1 for the line emitted, if it differs from the arriving byte; and q_y = 0 otherwise. The cell of line y then changes its connections from cells by Δw_ij = η · σ_j · q_y · e_ij(k). Section 13 does not exclude this form, and roadmap step 2 does not use it (decision D3). Its living counterparts are the climbing fibre, one for each Purkinje cell, and the plateau potential of hippocampal CA1 cells, which changes the inputs active within seconds of it (Bittner and colleagues, 2017).
98. [C] (new; C37, C39, C40; the further form for each answer: the right answer as bytes) Where the experimenter's reading of the stream marks an answer, the right answer may be given as feedback in the form of sentence 97, at each position of the answer, after the emission for that position. The right answer also arrives as input (sentence 78).
99. [C] (new; C37, C39, C40, C41; the further form for each answer: a magnitude) For an answer that is a number, a magnitude may be given after the last byte of the answer, at φ = 0 of the next window, for one step, to every output-end cell alike. The magnitude is the absolute difference between the number given and the right number, divided by the largest possible difference so that it lies between 0 and 1. If the attempt contains no readable number, the magnitude is 1. A cell uses it in the place of (r − r̄) in formula 14, as (r̄_d − d), where d is the magnitude and r̄_d is the cell's running average of it.
100. [C] (new; C50) For arithmetic modulo 89, the difference is first taken as the distance around the circle, min(|a − b|, 89 − |a − b|), divided by 44. The plain difference is the further form.
101. [F] (new) The traces needed for each place follow from formulas 6 and 7:
    - **Yes or no for a position:** the feedback comes e = 1 step after the emission is read, and the activity that produced the emission comes 0 to s − 1 steps before that. With τ = 16, an activity 8 steps before the feedback keeps (15/16)^8 ≈ 0.60 of its weight.
    - **Feedback for each line:** the same.
    - **Feedback for an answer of n bytes:** it reaches back n · s + 1 steps. For a three-byte answer with s = 16 that is 49 steps, where (15/16)^49 ≈ 0.04 remains. A brief that uses this form opens C16 to a longer trace.

    With s = 16 (sentence 112), the activity at the start of the window keeps (15/16)^16 ≈ 0.36 of its weight when the feedback arrives.
102. [F] (new) In the first form only output-end cells and encoder cells receive feedback directly (sentences 94 and 172). Other cells are affected only through the changed outputs of these cells: the changes at the output end reach the cells near the output face by return connections (sentence 34), the changes at the input end reach cells farther in by forward connections, and each connection costs at least one step.
103. [C] (r2 96, re-marked; C42) In the first form the feedback does not change any cell's activity directly (sentence 95), so the emission and any relay do not mix. If a further form of C42 relays feedback by a change of activity, that change also reaches the output lines. It must then be kept out of the next emission in one of three ways: it has died away before φ = s − 1, which requires s to exceed the decline time of the state; or the conversion ignores it; or the relay leaves the output-end cells that drive lines unchanged. Which way is used belongs to C42.
104. [F] (r2 99, corrected) Let a relayed change reach a cell A as a rise in an arrival from a source that the emission depends on. With the sign of formula 8, that rise weakens A's active excitatory inputs, so it lowers A's drive in the context that was wrong along a path whose signs multiply to +1, and raises it along a path whose signs multiply to −1. No cell knows the signs along its paths. Whether a relay by a rise, or by a fall, corrects the emission on balance is therefore not decided by the statement or by formula 8, and is read at roadmap step 3. Models in which forward strengths come into line with fixed return signs exist (Lillicrap and colleagues, 2016). They are simulations, not findings in tissue.
105. [C] (new; C40, further forms) The further forms of C40, which the roadmap reads at step 3 and section 13 allows, are:
    - feedback to the cells within distance 1/8 in space of an output-end cell as well;
    - to cells at every position in space;
    - to any cell, through feedback lines that end in its neighbourhood.

    Such feedback reaches 256 or more cells but comes from the world, not from a cell's chemical release. It is therefore not excluded by sentence 115, which concerns signals released by cells.
106. [F] (new) The delay of relay over connections: a change at the output end reaches a cell at distance d at least d steps later. That cell's contributing activity came at least d steps before the emission, so the interval to be spanned by its traces is at least 2d + e steps. For d = 4 and e = 1 that is 9 steps, where (15/16)^9 ≈ 0.56 of the weight remains. The trace of the first form therefore suffices for relay over four connections (settles finding F8).
107. [F] (r2 101, corrected) A cell with r outgoing connections reaches at most r cells in one relay and at most r^d in d relays. With r = 50, two relays reach at most 2,500 cells. No first-form rule sets the number of outgoing connections. It is the number of other cells that have drawn the cell as a source, and in the first form it averages the number of incoming connections.
108. [F] (r2 103, corrected) What a cell has, at each distance, from which to correct itself in the first form:
    - **Output-end cells (distance 0):** the feedback, their traces, prediction and mismatch.
    - **Cells one connection from the output end:** changes in their arrivals from output-end cells over return connections, at least one step after the feedback, with their own traces, prediction and mismatch.
    - **Encoder cells:** the feedback of the output end's information (sentence 172). They also read every arriving byte, including the right one, at their input lines.
    - **Other cells:** only the changes that have passed through the cells between them and either end. A cell's wait is set by its distance to the nearer of the two ends (sentence 175).

    In every case the cell computes its own change (formulas 8 and 14).
109. [F] (r2 104, corrected) With one part, a relayed change enters the cell's arrival. It changes the cell's activity at once (formulas 2 and 3) and its strengths through the mismatch (formula 8). The cell can tell it apart from other arrivals only by the synapse through which it arrives and by when it arrives. With two parts it can be kept apart (sentence 11).
110. [C] (r2 105; C44) The way a cell combines a relayed correction with its prediction is not opened before roadmap step 3. In the first form, sentence 109 describes what happens. The further forms are two parts, in which the mismatch at the part receiving return connections changes the strengths of the other part, and a local chemical signal of output-end cells as a factor on the rate (C28).

### 6.2 Living counterparts of section 6.1

- **Yes or no for each position (sentences 92 to 96).** Dopamine cells of the midbrain fire more when an outcome is better than predicted and less when it is worse (Schultz, Dayan and Montague, 1997). Dopamine is released over a region and decides, within about 0.3 to 2 seconds, whether coincident activity at a corticostriatal synapse strengthens it (Yagishita and colleagues, 2014). Formula 14, with r − r̄, corresponds to this. In songbirds, dopamine cells signal to one region of the song system whether a syllable was better or worse than usual (Gadagkar and colleagues, 2016).
- **Feedback for each line (sentence 97).** One climbing fibre reaches one Purkinje cell. When it fires shortly after the parallel-fibre inputs, those inputs weaken; this is textbook. Whether the climbing fibre carries an error in every condition is unsettled.
- **Right answer and magnitude (sentences 98 to 100).** An instructive input that tells a cell what it should have done, after it acted, is found in the climbing fibre and in the plateau potential of CA1 (Bittner and colleagues, 2017). A graded magnitude of error corresponds to graded dopamine responses. Whether one living system carries both at once is not known to this reader.
- **Relay inward (sentences 102 to 110).** That an error is relayed over several connections in cortex, each cell correcting itself, is unsettled. Return connections in cortex end largely on distal dendrites, which supports the two-part form of C44 as a hypothesis.

## 7. Time and the order of operations

111. [C] (r2 106; C45) The input is one byte after another, and nothing marks an item other than the bytes themselves. The ruling is Mr. Xu's by the record of 17 and 18 November (section 12 says so), but its words are not quoted, so this is marked as our choice.
112. [C] (r2 107; C24) There are s = 16 steps per byte, and all cells compute at once from the outputs of the previous step. Sixteen steps exceed the path of about ten connections from the input lines to the output end (section 2.2), so the emission can reflect the byte just read.
113. [C] (new; order within one step k, with φ = k mod s) The operations of one step are carried out in this order:
     1. If φ = 0, the world sets the input lines to the new byte, compares it with the emission read at the previous step, forms the feedback, and makes it present, for this step, at the output-end cells and at the encoder cells (sentence 172).
     2. Every cell computes its arrival A_i(k) from z_j(k) (formula 1).
     3. Every cell computes m_i(k) with its prediction p_i(k) (formula 5).
     4. Every cell computes v_i(k) and x_i(k) (formulas 2 and 3).
     5. Every cell computes the changes of formula 8 and, where feedback is present, of formula 14, from the traces e_ij(k) and ε_i(k) held before this step. It adds them and applies the bounds of formula 9.
     6. Every cell updates its traces, prediction, threshold and running averages (formulas 4, 6, 7, 10, 11 and 14).
     7. Every cell gives up and makes connections (formulas 12 and 13).
     8. If φ = s − 1, the emission for the coming position is read (sentence 79).
114. [F] (new) Within one byte, therefore:
     - at φ = 0 the byte reaches encoder cells, and the feedback for the previous emission reaches output-end and encoder cells;
     - at φ = d − 1 the byte's effect has reached at most the cells d connections from the input lines;
     - the emission at φ = 15 reflects the byte through paths of at most sixteen connections, and the shortest such path is about ten connections (section 2.2).
115. [F] (r2 108) A cell whose shortest path from the input lines is d connections long receives the effect of a byte no earlier than step φ = d − 1 of its window.
116. [F] (r2 109) Nothing outside the cells sets their states back at the beginning or end of an item, because nothing marks an item (sentence 111) and the cells are the only agents in the tissue (sentence 1). A cell may come, by its own adjustment, to respond differently when particular bytes arrive.
117. [F] (r2 110) There is one tissue meeting one stream, with no copies receiving items at once.
118. [C] (r2 111; C25) At the start of life all states, traces, predictions, thresholds and running averages are 0. This start happens once and is not a reset.

## 8. A worked example: one byte through the first form

The numbers were computed with `python3 -I` from formulas 1 to 14. They were checked by hand at step 0, as shown below. The starting values describe a moment in mid-life and are invented for the example. The previous byte was "5" and the arriving byte is "3". The emission read at the end of the previous window was "7", which is wrong. Three cells are followed.

- **Encoder cell E**, excitatory, near the input face. It has four sources:
  - line "3": strength 0.125, trace 0.02;
  - line "5": strength 0.125, trace 0.40;
  - inhibitory cell h, output 0.25: strength 0.25, trace 0.25;
  - excitatory cell c, output 0.125: strength 0.5, trace 0.125.

  Its v = 0.04, p = 0.05, θ = 0.02 and ε = 0.06, and its running average of the feedback is r̄ = 1/64.
- **Inner cell I**, excitatory, two connections from the input lines. It has three sources:
  - E: strength 0.5, trace 0.06;
  - excitatory cell q, output 0.25: strength 0.25, trace 0.25;
  - inhibitory cell g, output 0.125: strength 0.25, trace 0.125.

  Its v = 0.10, p = 0.12, θ = 0.05 and ε = 0.08. E's output at the last step of the previous window was 0.03.
- **Output-end cell O**, the cell of line "7". It has two sources:
  - excitatory cell u, one connection from the output end, output 0.5 at steps 0 to 7 and 0.125 at steps 8 to 15: strength 0.75, trace 0.50;
  - inhibitory cell f, output 0.25: strength 0.25, trace 0.25.

  Its v = 0.60, p = 0.40, θ = 0.10 and ε = 0.45 (it was active at the emission), and r̄ = 1/64.

**Step 0 by hand.**

The feedback is r = 0 ("no"), because "7" was emitted and "3" arrived. It reaches the output-end cells and the encoder cells (sentence 172).

*Cell E:*
- A = 0.125 · 1 − 0.25 · 0.25 + 0.5 · 0.125 = 0.125, so m = 0.125 − 0.05 = 0.075.
- v = 0.75 · 0.04 + 0.25 · 0.125 = 0.06125, and x = 0.06125 − 0.02 = 0.04125.
- Formula 8 gives Δw for h = −(1/32) · (−1) · 0.075 · 0.25 = +0.00059, and for c = −(1/32) · 0.075 · 0.125 = −0.00029. The lines do not change by formula 8 (sentence 70).
- Formula 14, applied to all of E's connections (sentence 173), gives:
  - for line "5": (1/32) · (0 − 1/64) · 0.40 · 0.06 = −0.0000117;
  - for line "3": −0.0000006;
  - for h: +0.0000073;
  - for c: −0.0000037.

  The line that was active before the wrong emission ("5") is weakened most.
- The next θ = 0.02 + (1/128) · (0.04125 − 0.125) = 0.0193.

*Cell I:*
- A = 0.5 · 0.03 + 0.25 · 0.25 − 0.25 · 0.125 = 0.0462, so m = −0.0737.
- v = 0.0866, and x = 0.0366.
- Δw for E = +0.00014, for q = +0.00058, and for g = −0.00029. The arrival was below the prediction, so excitatory inputs strengthen and inhibitory ones weaken.

*Cell O:*
- A = 0.75 · 0.5 − 0.25 · 0.25 = 0.3125, so m = −0.0875.
- v = 0.75 · 0.60 + 0.25 · 0.3125 = 0.5281, and x = 0.4281.
- For u, formula 8 gives −(1/32) · 0.0875 · 0.5 · (−1) = +0.00137. Formula 14 gives (1/32) · (0 − 1/64) · 0.5 · 0.45 = −0.00011. The sum is +0.00126.
- For f, formula 8 gives −0.00068 and formula 14 gives +0.00005.
- The next r̄ = (127/128) · (1/64) = 0.0155.

**The remaining steps (computed).**

| Step | E: A | E: m | E: x | I: A | I: m | I: x | O: A | O: m | O: x |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.1250 | 0.0750 | 0.0412 | 0.0462 | −0.0737 | 0.0366 | 0.3125 | −0.0875 | 0.4281 |
| 1 | 0.1248 | 0.0701 | 0.0578 | 0.0521 | −0.0633 | 0.0286 | 0.3133 | −0.0812 | 0.3720 |
| 2 | 0.1246 | 0.0656 | 0.0702 | 0.0605 | −0.0509 | 0.0250 | 0.3141 | −0.0754 | 0.3300 |
| 3 | 0.1245 | 0.0613 | 0.0795 | 0.0668 | −0.0414 | 0.0241 | 0.3148 | −0.0699 | 0.2986 |
| 4 | 0.1243 | 0.0573 | 0.0865 | 0.0716 | −0.0341 | 0.0248 | 0.3155 | −0.0649 | 0.2750 |
| 5 | 0.1242 | 0.0536 | 0.0917 | 0.0752 | −0.0284 | 0.0264 | 0.3161 | −0.0602 | 0.2573 |
| 6 | 0.1241 | 0.0501 | 0.0956 | 0.0779 | −0.0239 | 0.0285 | 0.3167 | −0.0558 | 0.2440 |
| 7 | 0.1239 | 0.0469 | 0.0985 | 0.0799 | −0.0204 | 0.0308 | 0.3173 | −0.0518 | 0.2340 |
| 8 | 0.1238 | 0.0438 | 0.1008 | 0.0814 | −0.0176 | 0.0330 | 0.0334 | −0.3325 | 0.1554 |
| 9 | 0.1237 | 0.0410 | 0.1024 | 0.0826 | −0.0153 | 0.0351 | 0.0347 | −0.3104 | 0.0972 |
| 10 | 0.1236 | 0.0383 | 0.1037 | 0.0834 | −0.0135 | 0.0371 | 0.0358 | −0.2898 | 0.0542 |
| 11 | 0.1235 | 0.0358 | 0.1046 | 0.0841 | −0.0120 | 0.0389 | 0.0369 | −0.2706 | 0.0226 |
| 12 | 0.1234 | 0.0335 | 0.1054 | 0.0846 | −0.0107 | 0.0405 | 0.0379 | −0.2527 | 0.0000 |
| 13 | 0.1234 | 0.0313 | 0.1059 | 0.0850 | −0.0097 | 0.0420 | 0.0388 | −0.2360 | 0.0000 |
| 14 | 0.1233 | 0.0293 | 0.1064 | 0.0853 | −0.0088 | 0.0434 | 0.0396 | −0.2204 | 0.0000 |
| 15 | 0.1232 | 0.0274 | 0.1067 | 0.0856 | −0.0080 | 0.0446 | 0.0404 | −0.2059 | 0.0000 |

At φ = 15, O's activity is 0, below 1/2, so line "7" is not emitted again from O. The emission is the line of whichever output-end cell is most active above 1/2, or no byte. Over the window:
- E's arrival stays near its value at step 0, while its activity rises toward the arrival as its state follows (τ_v = 4) and its threshold falls.
- I responds to E's rise one step later.
- O's input from u falls at step 8. Its arrival then lies far below its prediction, so formula 8 strengthens u and weakens f. The "no" at step 0 weakened u by 0.00011 (formula 14).

The example shows that, in the first form, the feedback acts at one step and is small compared with formula 8 while r̄ is small (sentence 96).

## 9. Chemical signal, lasting change and reactivation

119. [S] (r2 112, re-marked) A cell's chemical signal reaches a region near it. This is section 9 ("the signal is local"), reported in the reviewer's wording.
120. [S] (r2 113, re-marked) How each receiving cell reads it is its own matter. This is section 7, point 2, reported in the reviewer's wording.
121. [F] (r2 114, corrected) A chemical signal released by a cell never reaches the whole tissue (sentence 119). This concerns signals released by cells only, not the feedback of section 6.
122. [C] (r2 114; C26) The tissue contains no group of cells that releases a chemical signal over all of it, such as the noradrenergic cells of the locus coeruleus in a whole brain.
123. [F] (r2 115) A cell can release its chemical signal only on an occasion it detects from its own quantities, which for a cell that receives feedback include the feedback.
124. [C] (r2 116, 117, 119; C26, C27, C28) In the first form no cell releases a chemical signal. The further forms are listed under C26 to C28.
125. [F] (r2 118) A chemical signal received alike by a region cannot tell a receiving cell which connection to change; that comes from the cell's own quantities.
126. [C] (r2 120; C46) Long-term memory forms in the course of learning and not by an act of a program. The ruling is Mr. Xu's by the record (section 12 says so), with rehearsal acceptable, but its words are not quoted, so this is marked as our choice.
127. [F] (r2 121) No step of the program marks a change as lasting, freezes connections, or copies the tissue (sentence 126).
128. [C] (r2 122; C31) In the first form each strength has one component, which stays as changed.
129. [F] (r2 123, corrected) In the first form no program passes over an item, or over stored bytes, after the item has ended. The world gives feedback only as section 6 states. The tissue's own reactivation of recent activity is neither excluded nor imposed.

## 10. Numbers

130. [S] (r2 124, re-marked) "我们预先给定的参数越少，长期来看越好" ("the fewer parameters we fix in advance, the better in the long run"), and, where needed for now, "从2, 4, 8, 16,32,64,128等等取一些数字" ("take some numbers from 2, 4, 8, 16, 32, 64, 128 and so on", section 12).
131. [C] (r2 125; C47) For quantities smaller than one, the grid is read as 1/2, ..., 1/128. Numbers of cells go beyond 128 by further powers of two ("等等", "and so on", section 12).

## 11. What does not exist in the tissue

132. [C] (r2 126, corrected) In the first form no cell is given the right answer. The answer reaches the tissue as bytes at the input end, and the feedback says yes or no. When the feedback says yes, the cell of the emitted line can learn from it that its byte was right. Under the further forms of sentences 97 and 98, the right answer is given as feedback after the emission. This is permitted by sections 13 and 14 within the limits of sentence 88.
133. [F] (r2 127) No cell reads every cell, because a cell receives only from its neighbourhood (sentence 27).
134. [F] (r2 128) No error passes backward through the connections that carry the forward signal, and there is no single backward pass (sentence 86). A connection carries only its sending cell's output, in one direction.
135. [F] (r2 129, corrected) The quantities computed outside the cells and given to cells are the feedback of section 6 (in the first form, r alone), the activity of input lines, and nothing else. No other quantity over a population of cells is computed and given to cells, because learning is local (section 8).
136. [F] (r2 130, corrected) No change of a cell's strengths is computed outside that cell. The feedback states only whether the emission, or a line, or an answer was right, or by how much it was off. Each cell computes its own change (formulas 8 and 14).
137. [F] (r2 131) There is no reset of cells' states at each item (sentence 116).
138. [F] (r2 132) There are no copies of the tissue and no averaging over items (sentence 117).
139. [F] (r2 133) There is no window of past bytes beyond what traces and states keep (sentences 10 and 67).
140. [F] (r2 134, corrected) The criterion applied to the tissue as a whole is the world's comparison of its emissions and answers with the stream (section 6). It reaches the cells only as feedback, and no cell is given another goal.
141. [F] (r2 135, corrected) No sentence of this definition gives a cell the right answer before the emission for that position, or a change computed for it outside it (sentence 88).

## 12. Further sentences required by the findings

142. [F] (new; settling finding F2) The ordered ring proposed by the third reader is not taken, because Mr. Xu's ruling of section 15 gives each cell a random coordinate instead (sentence 4). With that ruling, the distances of 1, 2 and 4 from both ends that steps 1 and 3 read exist in every draw (section 2.2).
143. [C] (new; C43, further forms) Roadmap step 3 reads how the result depends on the number of return connections (2 to 128). The further form of C43 sets, for each cell, the share of its draws that is taken from cells with a larger coordinate x. In the first form that share is whatever the neighbourhood gives (about a half).
144. [F] (new) An encoder cell near the input face has in its neighbourhood the input lines within 1/8 of it (about 2 · 1/8 · 256, or 64 lines, for a cell at the face itself) and about 45 to 90 cells. So about half to two thirds of its starting draws are lines. A cell whose lines are not in use gives them up after P steps (sentence 59) and draws again (sentence 60).
145. [C] (new; C51) The random draws use one generator with a recorded seed. A tissue with another seed is another tissue, for use as a control (roadmap section 5, condition 3).
146. [F] (new) Formula 8 applies to connections into output-end cells as well as formula 14. Both changes are summed at the step at which feedback is present (sentence 113, item 5).
147. [F] (new) Since x is at most 1, every strength at most 1, and a cell's number of connections is its own, the arrival is bounded by the number of the cell's excitatory sources. The threshold, which rises when activity exceeds a* (formula 10), keeps the mean activity near a*.
148. [C] (new; C29) Formula 10 applies to output-end cells in the first form. Exempting them, or applying the control only below the emission level of 1/2, is a further form.
149. [C] (new; C24, C11) The decline time of the state (τ_v = 4) is shorter than the window (s = 16), so a change caused at φ = 0 has fallen to (3/4)^15 ≈ 0.013 of its size by φ = 15. This matters only for the further forms of C42 (sentence 103).
150. [F] (new) The first form uses only quantities a living cell holds: its arrival, state, threshold, the traces at its synapses, running averages of its own activity and of its inputs, and the feedback that reaches it.
151. [C] (new; settles finding F12) The orders of opening across choices for roadmap steps 1 to 3 are those of section 5.
152. [C] (new) The numbers fixed by us in the first form are those of section 6.
153. [F] (new) Each of the 28 decisions listed by the third reader is settled in a sentence or a formula, or listed as a choice with its first form, as section 7 shows.
154. [C] (new; C1, C44) The two-part form, when opened, assigns to the second part the return connections (from sources with a larger coordinate x) and the feedback lines, and to the first part all other connections. A cell does not compute where a source lies; the part at which a connection arrives is fixed when it is made, as living cortical cells receive feedback from higher areas on their distal dendrites.
155. [F] (new) Under sentence 154 the assignment of a source to a part is fixed by where the source lies, not computed by the cell.
156. [C] (new; C41, further forms) The further forms of formula 14 are: the yes or no factor r − r̄ also applied to cells within 1/8 in space of an output-end cell (C40); and a factor of separate size for "yes" and "no".
157. [C] (new; C17, further form) The further form of formula 12 gives up a connection whose product of input trace and mismatch has averaged near zero over P steps.
158. [C] (new; C18, further form) The further form of formula 13 draws the new source toward a member of the neighbourhood whose recent activity coincides with the cell's own, read through the local chemical signal (C26).
159. [C] (new; C6) The starting number of 8 incoming connections is a number fixed by us. The cell's number of connections after the start is its own (sentence 61).
160. [C] (new; C16) One trace time constant τ = 16 serves inputs, own activity, and the prediction. Separate values are a further form.
161. [C] (new; C11) The state's decline time τ_v = 4 is fixed by us. A cell setting it from its own mismatch is a further form.
162. [C] (new; C15) The rate η = 1/32 serves formulas 8 and 14. Separate rates, or a rate set by the cell from the running size of its own mismatch, are further forms.
163. [C] (new; C17) The giving-up level 1/128 serves both the strength and the input's running average.
164. [C] (new; C17, C29) One period P = 128 steps serves the running averages, the minimum age of a connection, the interval between makings, and r̄ (counted in positions).
165. [C] (new; C33) The emission level 1/2 is fixed by us.
166. [C] (new; C37) The feedback values 1 for yes and 0 for no are a convention, not a number to vary.
167. [C] (new; C23) The number of cells, 2,048, is fixed by us. It is a power of two on the grid extended for numbers of cells, and the smallest such number that gives about 1,800 cells besides the 256 output-end cells. A larger number is a further form.
168. [C] (new; C30) The neighbourhood radius 1/8 is fixed by us. It gives about 90 cells in a neighbourhood, of the order of the fifty in Mr. Xu's example as reported in the roadmap.
169. [C] (new; C30) The space is two-dimensional and of unit extent in x and y. One dimension gives either a path of about 45 connections or neighbourhoods of about 400 cells, as section 2.2 shows.
170. [C] (new; C3) The share of inhibitory cells, 1/4, is fixed by us.

## 13. The feedback at the input end (section 15)

171. [S] (new) "我们可以假设输入端可以得到输出端信息的反馈。在NLP任务中这是next byte prediction的结果，但在其他任务中可以是其他的" (section 15): we can assume that the input end can receive the feedback of the output end's information; in language tasks this is the result of the prediction of the next byte, but in other tasks it can be something else.
172. [C] (new; C52) In the first form the input end receives the same feedback as the output end:
    - r for each position, through one feedback line;
    - reaching every encoder cell alike;
    - at step φ = 0 of the window in which the byte arrives, for one step.

    "输出端信息的反馈" can be read in two ways: as the result of the comparison (yes or no), or as the output end's information itself (the emitted byte). The first form takes the first reading. The further forms of C52 are:
    - the emitted byte itself, given to the input end through separate feedback lines, one for each byte value, at φ = 0 (this is also the second form of C35);
    - for other tasks, whatever the brief of the task names as the output end's information.
173. [C] (new; C52, C41) An encoder cell uses this feedback by formula 14, with its own running average r̄_i, on all its connections, including those from input lines. Formula 8 still does not apply to line connections (sentence 70). Through the feedback, the input end learns how to take in from the result of the tissue's own prediction, as section 10 requires.
174. [F] (new) The feedback received at the input end differs from that received at the output end in three ways:
     - **Timing relative to the cell's own contribution.** An output-end cell's activity drove the emission directly, between 1 and s steps before the feedback. An encoder cell's activity reached the emission only along a path of about ten connections (section 2.2), so the activity that contributed to it lies 11 or more steps before the feedback, together with up to s more steps within the window. With τ = 16, activity 11 steps back keeps (15/16)^11 ≈ 0.49 of its weight, and activity 27 steps back keeps (15/16)^27 ≈ 0.18.
     - **Coincidence with the new byte.** At the same step φ = 0, the encoder cell also receives the new byte on its lines. The changes of formula 14 use the traces held before the step (sentence 113, item 5), so they concern the earlier byte and not the new one.
     - **No line of its own.** An encoder cell drives no output line, so the feedback cannot say whether its own line was right. Its correction is of what it takes in, not of what it gives out.
175. [F] (new) The input end is therefore not the last part of the tissue that correction reaches. It is reached at the same step as the output end. A cell that receives no feedback directly is reached, by relay, from whichever end is nearer in connections. With a path of about ten connections between the ends, no cell lies more than about five connections from one of them. The roadmap's overlap 2 ("the input end is farthest and is reached last") and its ordering of steps 2 and 3 on that ground no longer follow. The cells at four connections from the output end and those at four connections from the input lines both lie within the reach that step 3 reads.
176. [F] (new) Because the feedback at both ends is computed by the world, and not passed from the output end to the input end through the cells, it is not a backward pass through the forward connections (sentence 134). It is two entries of the same feedback.
177. [C] (new; C52) What the input end receives in tasks other than the prediction of the next byte is left to the brief of each task, as section 15 leaves it. The entry is the same feedback line, the same step φ = 0, and the use of formula 14 with the quantity the brief names in place of r.

A living counterpart of feedback at the input end: in rats, pairing a tone with stimulation of the dopamine cells of the ventral tegmental area enlarges the representation of that tone in primary auditory cortex (Bao, Chan and Merzenich, 2001). Whether sensory relay cells of the thalamus receive such feedback is not known to this reader.

**Counts.** 177 sentences: [S] 29, [F] 66, [C] 82.

## 4. The choices

The first form is the simplest form a living cell or tissue can do that the roadmap's section 5 allows. The further forms are listed in the order to be opened. G is the grid 2 to 128 (with further powers of two for cells), and 1/G is 1/2 to 1/128. "Cell" in the last column means the cell sets the number itself from its own quantities: "yes" in the first form, "possibly" in a further form, "no" otherwise.

C4 and C9 are no longer choices (sentences 17 and 31). C36 is withdrawn (revision 2). C48 to C54 are new.

| No. | Choice (sentences) | First form | Further forms, in order | Numbers (first value) | Cell |
|---|---|---|---|---|---|
| C1 | Parts (12, 154) | One part | Two parts as in sentence 154 | None | No |
| C2 | Gate (14) | None | A gate on retention from own arrival and state; input and output gates (unsettled) | None | Possibly |
| C3 | Inhibitory cells (15, 16, 170) | 1/4 at random among cells other than output-end cells; output-end cells excitatory | 1/8; 1/2 | 1/4 | No |
| C5 | Strength (28) | Graded, 0 to 1 | Counted in synapses | Maximum 1 (unit) | Yes, the strength |
| C6 | Number of connections (60, 61, 159) | Starts at 8; then the cell's own, by formulas 12 and 13 | Starting number 4, 16 | 8 | Yes |
| C7 | Connections at start (29) | 8 drawn at random within the neighbourhood, without replacement, no self | Bias toward sources with a smaller coordinate x | As C6 | No |
| C8 | Encoder cells receive from cells (29) | Yes, as their neighbourhood allows | No | None | No |
| C10 | Output and transfer (36, 50) | x = min(1, max(0, v − θ)) | Mismatch entering the output, sign carried by two kinds of cell | Slope 1, maximum 1 (units) | No |
| C11 | Decline of the state (49, 149, 161) | τ_v = 4 steps | 2, 8; set by the cell | 4 | Possibly |
| C12 | Number of predictions (46) | One, for the total arrival | One for each part | None | No |
| C13 | Prediction (46, 51) | Running average of own arrival, τ = 16 | Prediction of one part's arrival from the other part | τ, as C16 | No |
| C14 | Comparison (52) | Signed difference | Magnitude; above a level | None | No |
| C15 | Strength rule (55, 70, 162) | Δw = −η σ m e, η = 1/32, for connections from cells; line connections only structural | Coincidence of the input's trace with own activity, with a sliding threshold (long-term potentiation as in Bienenstock, Cooper and Munro) | 1/32 | Possibly |
| C16 | Traces (53, 54, 160) | Exponential, τ = 16, for inputs, own activity and prediction | 32 where answers of several bytes need it (sentence 101); separate constants | 16 | Possibly |
| C17 | Giving up (59, 157, 163) | Age ≥ P and (w < 1/128 or ū < 1/128) | Product of trace and mismatch near zero over P | 1/128; P | Partly: the level is fixed, the averages are the cell's |
| C18 | Making (60, 158) | When θ = 0, ā < a*, and P steps since the last making; source at random within the neighbourhood; strength 1/8 | Toward coincident activity through the chemical signal | 1/8; P | Yes, when; no, to whom |
| C19 | Incoming or outgoing (60) | Incoming only | Both | None | No |
| C20 | Passages (68) | Formed by traces and connections; no numbers | Positions drawn at random | None | Yes |
| C21 | Response to a byte (66) | One line for each value | Lines for bits | None | No |
| C22 | Encoder cells predict (71) | Yes | No | None | No |
| C23 | Numbers of cells (72, 167) | 2,048 cells, of which 256 are output-end cells | 4,096; 8,192 | 2,048 | No |
| C24 | Steps per byte (112) | s = 16, all cells at once | 32; 8 with radius 1/4; random order | 16 | No |
| C25 | Start (118) | All quantities 0 | Random states | None | No |
| C26 | Chemical release (122, 124) | None | Output-end cells release on feedback; cells release on activity above their running average; release passed on at most once | Factor, G (later) | Possibly |
| C27 | Region (124) | Not used in the first form | The releasing cell's neighbourhood; concentration falling with distance | As C30 | No |
| C28 | Effect of the chemical signal (124) | Not used in the first form | Scales the rate; permits making or giving up; permits lasting change | 1/G (later) | Possibly |
| C29 | Activity control (57, 148) | Formula 10 for all cells, η_θ = 1/128, a* = 1/8, θ from 0 | Exempt output-end cells, or control below 1/2 only; multiplication of all strengths (synaptic scaling) | 1/128; 1/8 | Yes, θ |
| C30 | Space and neighbourhood (5, 168, 169) | Two dimensions; unit square without joined edges; coordinates uniform at random; all cells and lines within distance 1/8 | Simplest first among the further forms: one dimension; radius 1/4 (with s = 8); three dimensions; an extent longer along x; a neighbourhood of the k nearest cells | 1/8 | No |
| C31 | Lasting change (128) | One component | Fast and slow components; lasting while a chemical signal is present; older connections given up less readily (step 4) | G (later) | Partly |
| C32 | Input lines (66) | 256, active for the whole window, value 1 | Bit lines | 256 (D1) | No |
| C33 | Output conversion (79) | Byte of the most active output-end cell, if above 1/2; ties to the lowest value | Exactly one line above the level | 1/2 | No |
| C34 | When emitted (76) | At every position, read at φ = s − 1 | Whenever a line exceeds the level | None | No |
| C35 | Reading its own emission (81) | No | Through the input lines | None | No |
| C37 | What the feedback is (90 to 92, 97 to 99) | r = 1 for yes, 0 for no, for each position | For each output line (97); the right answer (98); a magnitude (99); graded for each position | None | No |
| C38 | No emission (82) | "No" for feedback; wrong for scoring | Not scored where nothing can be expected (already in the first form by the experimenter's reading) | None | No |
| C39 | Timing (93, 99) | At φ = 0 of the byte's window, for one step (e = 1) | Answer feedback at φ = 0 after the answer's last byte | None | No |
| C40 | Who receives it (94, 105) | Every output-end cell alike | The cell of each line; cells within 1/8 of an output-end cell; cells at every position; any cell | None | No |
| C41 | Use at a receiving cell (95, 156, 173) | Formula 14 with r − r̄, η as C15, P as C17, at output-end and encoder cells | Also at cells near output-end cells; separate factors for yes and no; line form of sentence 97 | As C15 and C17 | Yes, r̄ |
| C42 | Relay by activity (103) | None: feedback is a factor only | A change of activity of the receiving cells, kept out of the emission as sentence 103 requires | Size, 1/G (later) | No |
| C43 | Return connections (35, 143) | Those the neighbourhood gives | A set share of draws from sources with a larger coordinate x, 2 to 128 | None | Yes, by C17 and C18 |
| C44 | Combining relay and prediction (110) | Not opened before step 3; sentence 109 describes the first form | Two parts; chemical factor | None | No |
| C45 | The stream (111) | Bytes only; no item marks | None | None | No |
| C46 | Long-term memory (126) | Only by the cells' own changes | None | None | No |
| C47 | Grid (131) | 2 to 128; reciprocals; cells beyond 128 | None | None | No |
| C48 | Electrical synapses (19) | None | Between neighbouring cells of the same kind | None | No |
| C49 | What remains excluded (88) | As sentence 88 | Mr. Xu may restate it | None | No |
| C50 | Difference modulo 89 (100) | Circular, divided by 44 | Plain | None | No |
| C51 | Generator and seed (145) | One recorded seed for each tissue | None | None | No |
| C52 | Feedback at the input end (172, 173, 177) | The same r, at φ = 0, to every encoder cell alike, used by formula 14 on all its connections | The emitted byte itself through feedback lines for each byte value; for other tasks, what the brief names | None | Yes, r̄ |
| C53 | Placing of the lines (5) | Input lines at random points on x = 0; output lines at random points on x = 1 | Lines placed at random points within a strip of width 1/8 at each face | None | No |
| C54 | Which cell drives an output line (5) | The nearest cell not already driving a line, lines taken in order of byte value | The nearest cell, allowing one cell to drive several lines | None | No |

## 5. The order of opening across choices

The order is from what is nearest the failure to what is farthest. Within each choice, its further forms are opened in the order of the table.

**Step 1 fails** (cells at 1, 2 or 4 connections from the input lines are silent or constant, or the bytes cannot be told from their outputs):

1. C29 (target, then rate).
2. C17 and C18 (giving up and making).
3. C15: the cancelling rule may flatten arrivals; open the second form.
4. C11 and C24 (decline of the state and steps per byte).
5. C30 (half-width).
6. C16 (trace length).
7. C3 (inhibitory share).
8. C6 (starting number).
9. C1 with C13 (two parts).
10. C2 (gate).

**Step 2 fails** (the output end does not come to give the coming byte above the frequencies of bytes):

1. C33 (emission level).
2. C41 (rate; separate factors for yes and no).
3. C29 at the output end (exempt).
4. C16 (trace length).
5. C52 (the input end's feedback: the emitted byte itself).
6. C40 (cells near output-end cells also receive feedback).
7. C37 form for each output line. This requires a change of decision D3 for step 2 and is not opened without it.
8. Then the order for step 1, since step 2 rests on it.

**Step 3 fails** (cells one, two and four connections from the output end do not correct themselves, or the result does not exceed the control tissue without relay):

1. C43 (share of return connections).
2. C40 (feedback to cells at more positions, as section 13 and decision D2 allow).
3. C42 (relay by activity, with sentence 103).
4. C16 (to 32).
5. C44 (two parts, sentence 154).
6. C24 with C11.
7. C26 to C28 (chemical relay).
8. C10 (mismatch in the output).

## 6. Every number fixed by us in the first form

Numbers that can be varied on the grid (14):

| No. | Number | Value | Choice | Set by the cell? |
|---|---|---|---|---|
| 1 | Number of cells | 2,048 | C23 | No |
| 2 | Radius of the neighbourhood | 1/8 | C30 | No |
| 3 | Share of inhibitory cells (other than output-end cells) | 1/4 | C3 | No |
| 4 | Starting number of incoming connections | 8 | C6 | **Yes after the start**: the cell's number is its own (formulas 12, 13) |
| 5 | Starting strength, initial and new | 1/8 | C5, C18 | No |
| 6 | Decline time of the state, τ_v | 4 steps | C11 | No (possibly later) |
| 7 | Time constant of traces and prediction, τ | 16 steps | C13, C16 | No (possibly later) |
| 8 | Rate of strength change, η | 1/32 | C15, C41 | No (possibly later) |
| 9 | Rate of the threshold, η_θ | 1/128 | C29 | No |
| 10 | Target activity, a* | 1/8 | C29 | No (possibly later) |
| 11 | Period of running averages, P | 128 steps or positions | C17, C18, C41 | No |
| 12 | Giving-up level | 1/128 | C17 | No |
| 13 | Steps per byte, s | 16 | C24 | No |
| 14 | Emission level | 1/2 | C33 | No |

Values fixed by a decision, or as units, and not varied:
- 256 input lines and 256 output lines (decision D1);
- the space: two dimensions, unit extent (C30);
- the faces on which lines are placed, x = 0 and x = 1 (C53);
- maximum strength 1;
- maximum activity 1 and slope 1;
- line activity 1;
- feedback values 1 and 0;
- duration of the feedback, one step, at φ = 0 (e = 1);
- starting values 0 (state, prediction, threshold, traces, averages);
- the denominator 44 of C50 (half of 89, rounded down);
- the seed (recorded, C51).

Quantities that each cell sets itself in the first form: its number of connections; its strengths; its threshold; its running averages ā, ū and r̄.

## 7. How each finding and each decision of the third reading is settled

**Findings.**

| Finding | Settled by |
|---|---|
| F1 Feedback has no place | Term "feedback" with yes and no; places for each position, each line and each answer (sentences 92 to 101); sentence 90 and "not assumed" withdrawn (87); sentences 20, 24, 108, 132 and 141 corrected; sentences 21 and 22 add feedback lines |
| F2 Empty distance groups | Random coordinates by Mr. Xu's ruling of section 15 (4); lines on opposite faces; neighbourhood by distance 1/8 (5); counts in section 2.2 show cells at every distance from 1 to 10 from both ends. The ring proposal is not taken because section 15 overrides it (142) |
| F3 Sentence 96 | Re-marked as a choice within C42 (103), and not needed in the first form, where feedback is a factor only |
| F4 Sentences 99 and 104 | 99 corrected to a conditional statement with the sign of formula 8 fixed and the path sign named (104); 104 corrected (109); C44's first form no longer conflicts with C10 (110) |
| F5 Return connections | Defined geometrically; they exist in the first form as the neighbourhood gives them (34, 35); distance is for reading only (6) |
| F6 Output end cannot start | Formula 14 with r − r̄ (95, 96); conversion by the most active line above 1/2 (79); C29's exemption kept as a further form, because the first form already lets the output end begin from rare right emissions (148) |
| F7 Missing formulas and sign | Formulas 1 to 14 (sections 3 and 6) and order of operations (113); feedback is a factor, not an arrival (95) |
| F8 Trace arithmetic | τ = 16 and e = 1; relay over four connections needs 9 steps (106); answers of three bytes need 49 and open C16 (101) |
| F9 Silent encoder cells | Connections to lines that are almost never active are given up (59), and new draws follow (60, 144) |
| F10 Number of connections fixed | The cell's own from the start (60, 61); 8 is only the starting number |
| F11 Marks | Sentence 1 → [F]; 12 → [C] (15); 31 adds electrical synapses (32, 19); 8, 10, 34, 82, 112 and 113 say "reported in the reviewer's wording" (11, 13, 37, 78, 119, 120); 87 split into [S] and [C] (89, 90); 88 → [C] with both readings (91); 114 split (121, 122); 123, 129, 130 and 134 corrected (129, 135, 136, 140); 21 and 53 include any cell receiving feedback (22, 150); 55, 100 and 101 corrected (55, 106, 107). All are accepted |
| F12 Order across choices | Section 5 |

**The 28 decisions.**

| No. | Decision | Settled by |
|---|---|---|
| 1 | Transfer function, maximum activity, state and arrival | Formulas 2 and 3 |
| 2 | Formula of the prediction | Formula 4 |
| 3 | Sign of C15; inhibitory inputs; output-end cells | Formula 8, sentences 55 and 146 |
| 4 | Order within a step | Sentence 113 |
| 5 | Line active for all steps, value 1 | Sentence 66 |
| 6 | Step of the feedback | Sentence 93 (e = 1) |
| 7 | Feedback as arrival or factor | Factor (sentence 95) |
| 8 | Order of C15 and C41 changes | Summed, then bounded (sentence 113, item 5) |
| 9 | Form of the trace | Exponential, τ = 16 (formulas 6 and 7) |
| 10 | Composition and symmetry of the neighbourhood | Sentence 5: all kinds; symmetric |
| 11 | To whom output-end cells send | Any cell that draws them (sentence 34) |
| 12 | Spread of inhibitory cells | Sentence 16 |
| 13 | Encoder cells' neighbourhood of lines | The input lines within distance 1/8 (sentence 5) |
| 14 | Replacement, self and double connections | Sentence 29 |
| 15 | Period and mean in C17 | Age and running averages; no mean strength (sentence 59) |
| 16 | Starting strength of a new connection | 1/8 (sentence 60) |
| 17 | New source for an encoder cell | Lines and cells of its neighbourhood (sentence 60) |
| 18 | C29 rate, start, target | Formula 10 |
| 19 | Maximum activity | 1 (formula 3) |
| 20 | Step of the reading | φ = s − 1 (sentence 76) |
| 21 | When distance is computed | By the experimenter, for reading only (sentence 6) |
| 22 | Return connections counted, and direction | Incoming, among the cell's own (sentence 35) |
| 23 | Length of C41's activity trace | Same τ (sentence 160) |
| 24 | C29 at output-end cells | Applies; exemption is a further form (sentence 148) |
| 25 | Positions where no byte was right | None exist; no emission is "no" (sentence 82) |
| 26 | Seed and generator | Sentence 145 (C51) |
| 27 | Starting threshold; state below zero | θ from 0; v may be negative (sentences 49, 57) |
| 28 | Difference modulo 89, and timing | Sentences 99 and 100 |

## 8. What the definition still does not determine

1. **Whether the first form learns.** That is for roadmap steps 1 to 3. The definition names one specific risk: the cancelling rule of formula 8 may flatten the arrivals of inner cells so that their outputs carry little. The order of opening for step 1 places its second form third.
2. **The limits in sentence 88.** They are our reading of sections 11, 13 and 14 together, and Mr. Xu may restate them. In particular, whether the right answer may also be given to cells other than output-end cells is permitted by section 13 and is a choice (C40), not a question his words leave open.
3. **How a cell combines a relayed correction with its prediction.** This is unsettled in neuroscience and is opened at step 3 (C44).
4. **How a change becomes lasting.** This is chosen at step 4 (C31).
5. **The roadmap's step 2 against section 15.** Step 2 says that the feedback acts on the output end only, while section 15 gives the input end feedback too, and this definition follows section 15 (C52). Step 2 can be run with C52 in its first form, or with feedback at the input end withheld as a second tissue for comparison; which of these the brief uses is for the reviewer to decide. The roadmap's overlap 2, which places the input end last, no longer follows (sentence 175).
6. **The wording of the rulings behind C45 and C46.** These rulings are on record but not quoted. Their marks become [S] when they are.

## References

- Bao, Chan and Merzenich (2001), Nature 412: 79-83.
- Bittner and colleagues (2017), Science 357: 1033-1036.
- Bienenstock, Cooper and Munro (1982), Journal of Neuroscience 2: 32-48.
- Gadagkar and colleagues (2016), Science 354: 1278-1282.
- Lillicrap, Cownden, Tweed and Akerman (2016), Nature Communications 7: 13276.
- Schultz, Dayan and Montague (1997), Science 275: 1593-1599.
- Yagishita and colleagues (2014), Science 345: 1616-1620.

The remaining facts are textbook: Dale's principle and its exceptions, the climbing fibre and long-term depression, homeostatic plasticity, electrical synapses, the growth and elimination of synapses, and cerebellum-like cancellation of predictable input. Page numbers are from this reader's memory and should be checked before the list is cited elsewhere.
