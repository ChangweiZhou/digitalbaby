# The tissue defined, revision 2 (18 November 2026, written from 22:55 UTC by the clock of this machine)

**Note by the reviewer (23:03 UTC, 18 November 2026), not part of the agent's text.** By Mr. Xu's ruling of 23:03 UTC (`WHAT_A_NEURON_HAS_20261118.md`, section 13) cells can receive the feedback of the error; for the present task the feedback is yes or no; richer forms are to be accommodated and none is excluded in principle. Sentence 90 (the exclusion of telling a silent output line) and the wording that direct receipt by cells is not assumed are therefore to be changed in the next revision; the first form (yes or no, choice C37) stands.

This file replaces `THE_TISSUE_WRITTEN_OUT_20261118.md` as the definition of the tissue (step 0 of `ROADMAP_TO_END_TO_END_PERSISTENT_LEARNING_20261118.md`, draft 2). It was written by the second reader of that description. It takes in the check (`CHECK_OF_THE_TISSUE_WRITTEN_OUT_20261118.md`), corrected by Mr. Xu's statement of 22:46 UTC (section 11 of `WHAT_A_NEURON_HAS_20261118.md`). Neither earlier file has been changed. Nothing has been run.

**What changed from the check.** The check's sentences 116 and 119 to 121 said that nothing tells the tissue whether its output was right, and from this the check derived a further gap, numbered 5. Those sentences rested on a wording in the check's brief ("no cell is given an answer or an error computed for it") that went beyond Mr. Xu's words. Section 11 says that the system is told when it was wrong and corrects itself. The further gap is therefore withdrawn. Choice C36 of the check (a path laid down at the start from each shown byte to its output line) is withdrawn with it, because the signal of section 11 reaches the output end directly.

**Sources and marks.** [S]: Mr. Xu's words, quoted from sections 7 to 11 of `WHAT_A_NEURON_HAS_20261118.md`. Where a section reports his words in the reviewer's wording rather than quoting them, the sentence says so. [F]: follows by necessity, either from [S] sentences and earlier sentences or from a textbook fact about living nerve cells, which is applied because the cells are nerve cells (section 8). The reason is given. [C]: a choice made by us (the experimenters). It is numbered and listed in section 3. Each sentence shows in brackets the number it had in the description ("former n") or in the check ("check n"), or "new". Three conditions that the description marked [S] (the stream without item marks, long-term memory formed in learning, and the grid of numbers) rest on rulings not quoted in sections 7 to 11. They are marked [C] here, as choices C45 to C47, until his words are quoted.

**Terms.**
- A *cell* is a nerve cell of the tissue.
- An *encoder cell* is a cell at the input end. An *output-end cell* is a cell at the output end. An *inner cell* is a cell at neither end.
- An *input line* is a fixed channel whose activity is set by something particular in the present byte. It does not learn and it is not a cell. It corresponds to a receptor cell of a sense organ.
- An *output line* is a fixed channel whose activity determines something particular in the byte the tissue emits. It corresponds to the nerve to a muscle.
- A *step* is one exchange of outputs among cells.
- The *arrival* at a cell is what reaches it at one step. The *prediction* is a quantity the cell holds before an arrival that stands for that arrival. The *mismatch* is the difference between the arrival and its prediction.
- A *synapse* is one point of contact through which one cell acts on another. A *connection* from cell A to cell B is the set of synapses from A onto B. A connection's *strength* is the number that scales the effect of A's output on B.
- A *trace* is a quantity kept in a cell's state that records an earlier input or the cell's own earlier activity and declines over a number of steps.
- A *part* of a cell is a region of the cell (in a living cell, a dendritic branch or set of branches) at which inputs from some sources arrive and are combined before they reach the rest of the cell.
- The *neighbourhood* of a cell is the set of cells that its axon and dendrites can reach.
- The *emission* is the byte the tissue gives out at a position, or the absence of a byte. The *attempt* is the emission at a position where, by the experimenter's reading of the stream, an answer is due.
- The *signal* is what the world gives the tissue when its emission did not match the byte that then arrived (section 11).
- The *distance* of a cell from the output end is the number of connections on the shortest path from that cell to an output-end cell.
- A *return connection* is a connection from a cell to a cell at a greater distance from the output end.
- A *living counterpart* is a structure or process found in living nerve tissue that does what the sentence describes.

## 1. The tissue

### 1.1 What a cell is and holds

1. [S] (former 1) There is no separate learner; the tissue consists of nerve cells, and nothing in it learns apart from those cells (section 8: "There is no separate 'learner'").
2. [S] (former 2, corrected) "Apart from the byte-level encoder, all that any nerve cell can do is ..." (section 8), and "there is an output end and an input end; these two also have to learn, through learning, how to take in and how to give out" (section 10).
3. [F] (former 2, corrected) Each cell of the tissue is therefore an encoder cell, an output-end cell or an inner cell, and all of them are nerve cells. The reason is that the cells of both ends learn (sentence 2) and nothing learns apart from nerve cells (sentence 1). In the translation of section 8, the words "one of its cells" are the translator's. That the encoder consists of cells rests on section 10.
4. [S] (former 3, reworded) A cell receives input from other cells, sends from that input and its present state an output to the cells it is connected to, and keeps adjusting which cells it is connected to (section 8: "只是", "only"). Sentences 34 and 85 state what sections 9 and 11 add to these three acts: a cell may release a local chemical signal, and the system is told when it was wrong.
5. [S] (former 4) A cell holds a state: section 8 speaks of "its present state" and of "the state it has stored".
6. [F] (former 5, reworded) In this definition a cell's state includes its connections, their strengths, and the short-term state of its synapses. So understood, the state is the only record of the past that a cell has, because "it can read only the state it has stored and the new information that reaches it through other cells" (section 8).
7. [S] (former 6) The state is small compared with the stream: "a million bytes cannot be kept as a window" (section 8).
8. [S] (former 7) A cell may have separate parts on which inputs from different sources arrive and are treated differently (section 7, point 1, recording Mr. Xu's acceptance: "separate parts of a cell that treat inputs from different sources differently may be used").
9. [C] (former 8) Whether a cell has one part or two is choice C1.
10. [S] (former 9, reworded) A cell may be gated (section 7, point 1: "The gated cell may stay"). Whether a living cell has a counterpart of three gates, each a weighted sum of the cell's inputs, is unsettled. Parts of such a gate have living counterparts: multiplicative interaction of inputs on a dendrite, shunting inhibition, and persistent firing sustained by currents inside the cell.
11. [C] (former 10) Whether a cell is gated, and from what, is choice C2. A gate may be computed only from the cell's own arrivals and state.
12. [F] (former 11, corrected) Each cell is either excitatory or inhibitory for its whole life. Its strengths change with learning, but their sign does not. The reason is that the cells are nerve cells (sentence 1), and a living nerve cell acts on all its targets with the same sign (Dale's principle). The known exceptions do not make the sign learnable.
13. [C] (former 11) The share of inhibitory cells is choice C3.
14. [F] (former 12, corrected) A cell's output is never below zero, because the output of a living nerve cell is its firing or its release of transmitter. A cell can signal a decrease only as a fall below its own ongoing rate.
15. [C] (former 13) Whether, and how, a cell keeps its own level of activity within a range from its own recent activity is choice C29.
16. [S] (former 14, reworded) A cell acts and adjusts only within its own neighbourhood (section 9, reporting Mr. Xu's statement of 16 November: "a neuron acts and adjusts only within its own neighbourhood"). The neighbourhood is the set of cells that the cell's processes reach, as defined above, and not the set of cells whose bodies lie near its body. A living cortical cell may send its axon several millimetres or more, and it forms synapses only where its axon passes close to other cells' dendrites.
17. [C] (former 15) The form and size of the neighbourhood are choice C30.

### 1.2 What arrives at a cell and from where

18. [S] (former 16) A cell other than an encoder cell receives input only from other cells (section 8: "receive input from other cells").
19. [S] (former 17) A cell does not in general read the bytes that have passed (section 8; the Chinese "未必能够读到已有的字节" means "is not necessarily able to read the bytes already passed").
20. [F] (former 18) Bytes therefore enter the tissue only through encoder cells (sentences 18 and 19).
21. [F] (former 19, updated) The only things that can arrive at a cell are these:
    - the outputs of the cells connected to it;
    - the chemical signals released in its region (sentence 34);
    - for an output-end cell, the signal (sentence 85);
    - for an encoder cell, the activity of input lines.

    A cell receives other cells' states, connections, strengths or inputs only as far as their outputs carry them, because the statement gives a cell no other way to receive anything.
22. [S] (former 20) What arrives differs from cell to cell (section 8: "the new information differs from cell to cell").
23. [S] (former 21) What arrives may result from earlier bytes (section 8: "which may be the result of processing bytes read earlier").
24. [F] (former 22) A cell's sources lie within its neighbourhood (sentence 16).
25. [C] (former 23, reworded) Whether a connection has a graded strength is choice C5. The Chinese "不断调整自己连接那些细胞" can be read as "keep adjusting which cells it connects to" or as "keep adjusting its connections to those cells", so it does not settle the matter. A living connection is graded (number of synapses, probability of release, number of receptors). Whether a single synapse has only a few discrete states is unsettled.
26. [C] (former 24) The number of a cell's incoming connections, and whether it is held fixed, is choice C6.
27. [C] (former 25) Which cells a cell is connected to at the start of its life is choice C7.
28. [C] (former 26) Whether encoder cells also receive from other cells is choice C8.

### 1.3 What a cell sends and to whom

29. [S] (former 27) A cell sends an output, computed "from that input and its present state", "to the cells it is connected to" (section 8).
30. [F] (former 28, corrected) A cell sends the same output to every cell it is connected to and to the output line it drives, if any. The reason is that a living nerve cell has one axon, and its output enters every branch. Differences between targets lie in the receiving synapses, not in the output.
31. [F] (former 29, reworded) A cell sends no electrical output backward. What reaches the cells that send to it, from it, is its ordinary output, if they are among its targets, and its chemical signal, if they lie in its region. The reason is that sentence 4 and sentence 34 give it no other means. In living tissue, chemical signals released by a receiving cell do act on the sending terminals (endocannabinoids, nitric oxide). How specific to single synapses these signals are is unsettled.
32. [F] (former 30) A loop in which A sends to B and B sends to A is not excluded. In such a loop, whatever B's output carries reaches A as an ordinary input.
33. [C] (former 31) What a cell's output carries is choice C10. The forms are: activity computed from arrival and state alone; or activity into which the mismatch enters. A signed mismatch can be carried only by two kinds of cell (one for arrivals above the prediction, one for arrivals below), or as variation around a rate above zero (sentence 14).
34. [S] (former 32) Beside its output, a cell can release a chemical signal into a region near it (section 9: "a cell may be able to release a chemical substance, a signal, into a region near it; the signal is local").

### 1.4 Prediction

35. [S] (former 33) What a cell does is to be put as predicting what arrives (section 8: "预测从其他细胞得来的新来的信息", "predicting the information newly arriving from other cells").
36. [S] (former 34, reworded) It cannot simply be taken as expecting the next byte (section 8: "不能简单认为是'预期下一个到来的字节'"). What is denied is that a cell's task is simply the expectation of bytes. For an encoder cell, predicting what arrives includes predicting parts of the next byte.
37. [F] (former 35, reworded) A prediction of an arrival is formed before that arrival, from what the cell has before it: its state, and its earlier arrivals at any of its parts. The reason is that "新来的" means "newly arriving", so the prediction precedes what it predicts.
38. [F] (former 36) For the prediction to be compared with the arrival, both must be in the same cell at the same step. The state must therefore hold the prediction, or quantities from which the cell forms it at that step.
39. [F] (former 37) In the stream of the task, some bytes depend on bytes several positions earlier: the answer digits depend on both numbers. An arrival that reflects such a byte can be predicted only if information from the earlier bytes is present, at the moment of prediction, in the cell's state or in its present arrivals (sentence 19). This follows from the task's stream, not from the statement.
40. [F] (former 38, reworded) The tissue must therefore keep information about earlier bytes, over at least the distance from the first digits to the answer, in the states of cells (including the short-term state of their synapses, sentence 6) and in the outputs they are sending.
41. [C] (former 39) How long a cell's state keeps information is choice C11.

### 1.5 What predicting requires of a cell

42. [F] (former 40) The cell must hold the arrival.
43. [F] (former 41) The cell must hold a prediction (sentences 37 and 38).
44. [F] (former 42) The cell must form a comparison of the two, since otherwise nothing in the cell differs between a right and a wrong prediction. That comparison is the mismatch.
45. [F] (former 43) If the prediction is to improve by the cell's own adjustment, the cell must keep the inputs, or traces of the inputs, from which the prediction was formed.
46. [F] (former 44, reworded) The mismatch is a quantity of the cell that formed it. Another cell receives a function of it only if it enters the cell's output (C10), or if the cell's release of its chemical signal depends on it (C26).
47. [C] (former 45) The number of predictions a cell holds is choice C12.
48. [C] (former 46, reworded) What predicts what is choice C13. In one form, all arrivals are predicted from the cell's own state. In the other, arrivals at one part predict the arrivals that will come next at another part. The second form is unsettled as biology. In living pyramidal cells, inputs to the distal and the proximal dendrite interact and their coincidence produces bursts of firing. That this interaction compares a prediction with an arrival is a hypothesis.
49. [F] (former 47, mark corrected) Where each quantity lies follows from C13. With one part, arrival, prediction and mismatch lie at the same place. With two parts, the prediction is formed at the part that receives the predicting arrivals and compared at the part that receives the predicted arrivals.
50. [C] (former 48) The form of the comparison is choice C14.

### 1.6 Adjusting connections

51. [S] (former 49) A cell keeps adjusting its connections (section 8: "to keep adjusting which cells it is connected to").
52. [S] (former 50) Learning is local (section 8: "Learning is therefore local").
53. [F] (former 51, updated) A cell can adjust its connections only from quantities it holds:
    - its arrivals and the traces of earlier arrivals;
    - its prediction and its mismatch;
    - its own activity and traces of it;
    - the chemical signal of its region;
    - for an output-end cell, the signal.

    It holds nothing else (sentences 6 and 21).
54. [F] (former 52) A cell receives another cell's chemical signal only if it lies in the region into which that cell releases it (sentence 34).
55. [C] (former 53 and 61, reworded) How a strength changes, and at what rate, is choice C15. Every form keeps a strength between zero and a maximum, because a living synapse cannot reverse its sign (sentence 12) and cannot grow without limit. Both forms in the table have living counterparts. The coincidence form is long-term potentiation. The form driven by the mismatch exists in cerebellum-like structures: there, synapses weaken when sending and receiving activity coincide, so that the cell's response comes to cancel predictable input.
56. [F] (former 54) When the input that formed a prediction or an emission arrived several steps before the quantity that is to change it, the cell can relate the two only if a trace of that input is kept until then.
57. [C] (former 55) The length of traces is choice C16.
58. [C] (former 56) When a connection is given up is choice C17.
59. [F] (former 57, corrected) A cell receives nothing through a connection from a cell it is not connected to. It can receive, through the chemical signal of its region, a quantity that depends on the activity of nearby cells to which it is not connected (sentences 34 and 54), so the choice of a new source may rest on that quantity. In living tissue, glutamate released near a dendrite causes a new spine to grow toward the release site, and active cells release growth factors that promote nearby synapses.
60. [F] (former 58) A new source lies in the cell's neighbourhood (sentence 16).
61. [C] (former 59) How and when a cell makes a new connection is choice C18.
62. [C] (former 60, reason corrected) Whether a cell adjusts its incoming connections, its outgoing ones, or both is choice C19. Both sides of a living synapse hold local quantities, and both spines and axonal boutons are formed and removed. The first form adjusts incoming connections only because that form is the simplest.

### 1.7 The input end

63. [S] (former 62) Encoder cells read bytes; they are the one exception to sentence 19 (section 8: "Apart from the byte-level encoder").
64. [S] (former 63) What an encoder cell reads may be a collection of small passages of the input with gaps between them, and different cells read different ones (section 8: "它可能读到的只是输入信息中间有间隔的某些小段落的一个总集，不同的读到的不一样").
65. [F] (check 103) The input end consists of the encoder cells, because bytes enter only through them (sentence 20).
66. [F] (former 64) Since nothing marks where an item begins (C45), a passage can be placed only relative to the present byte or by the cell's own timing.
67. [F] (former 65) A passage that reaches back to earlier bytes requires that those bytes, or their effects, be held for that span within the input end. By sentence 7 this span is short.
68. [C] (check 105) Whether there are input lines between the bytes and the encoder cells is choice C32.
69. [F] (check 106) If input lines carry only the present byte, an encoder cell can read passages with gaps only by traces of the lines' earlier activity or by delays on its connections. The span of its passages is then bounded by its traces (C11, C16).
70. [C] (former 66 and 68) The passages of encoder cells, and how encoder cells come to differ, are choice C20.
71. [C] (former 67) How an encoder cell responds to a byte is choice C21.
72. [F] (former 69, corrected; check 104) Encoder cells adjust their connections by their own local learning (sentences 2 and 52).
73. [C] (former 69) Whether encoder cells also predict their arrivals is choice C22. Section 10 says that the ends learn "through learning", not that their learning is the same as that of inner cells.
74. [C] (former 70) The numbers of cells are choice C23.

### 1.8 The output end

75. [F] (check 107) What the tissue gives out is bytes. The output end gives out (sentence 2), the tissue's world is a stream of bytes, and section 11 gives as its example of a mismatch at the output end "预测next byte出现错误" ("the next byte was predicted wrongly").
76. [F] (check 108) The conversion of the activity of output lines into an emitted byte is fixed and does not learn, because nothing learns apart from the cells (sentence 1).
77. [C] (check 109 and 112) How the activity of output lines becomes a byte, and whether output-end cells also send to other cells, is choice C33.
78. [F] (check 110) A conversion that selects the most active of all output lines is a comparison over all of them. It is admissible only as the fixed conversion of sentence 76, and its result reaches the tissue only as the emission and through the signal.
79. [F] (check 111) An output-end cell is a cell like any other: it receives from cells, sends its output and adjusts its connections (sentence 4). The activity it sends on its output line is the same activity it sends to every other cell it contacts (sentence 30).
80. [C] (check 113) When the tissue emits is choice C34.
81. [F] (check 114, changed) The emission for a position is formed before the byte of that position arrives. Where an answer is due is known to the experimenter and to no cell, since nothing marks an item (C45). The attempt is therefore defined by the experimenter's reading of the stream.
82. [S] (check 115) The right answer is shown as input after the learner's own attempt (section 7, point 1: "the right answer shown as input after the learner's own attempt is the form of teaching").
83. [F] (check 116, changed) The right answer reaches the tissue only as bytes read at the input end (sentence 20). It is not delivered to output-end cells or to output lines. From the comparison of the emission with the arriving byte, output-end cells receive only the signal (sentences 85 to 92).
84. [C] (check 118) Whether the tissue also reads its own emission back through its input end is choice C35.

## 2. The signal

### 2.1 Sentences

85. [S] (new) "所以你可以给整个系统连续的信号，如果输出端的信息和预期不符合（例如预测next byte出现错误），告诉系统它错了让它自己去纠正。这本来就是我们persistent learning的目标" (section 11). In translation: the whole system can be given a continuous signal: if the information at the output end does not match what was expected (for example, the next byte was predicted wrongly), the system is told that it was wrong and corrects itself; this has been the aim of persistent learning all along.
86. [S] (new) Information may be relayed from neuron to neuron so that a few passes cover the network; a single backward pass over the whole network is not allowed; and a single nerve cell cannot see the answer. Section 11 ("What follows") reports these as Mr. Xu's earlier words, in the reviewer's wording. The figures "one cell to fifty, fifty to fifty times fifty" are given in section 1 of the roadmap and are not quoted in sections 7 to 11.
87. [F] (new) The world compares the emission at a position (a byte, or no byte) with the byte that then arrives at that position. This comparison is made outside the tissue, by those who give the signal ("你可以给", "you can give"). The reason is that no cell holds the arriving byte before it arrives (sentence 19) and no cell reads every output line (sentence 127).
88. [F] (new) The comparison can be made only when the byte arrives, after the emission. The signal therefore follows each emission, and it is given at every position for the whole life, since it is "continuous" (连续的).
89. [C] (new) What the system is told is choice C37. The forms, in the order of the roadmap's decision D3, are:
    - (a) one quantity for each byte position, saying that the emission was wrong;
    - (b) one quantity for each byte position, saying how wrong it was;
    - (c) each output line that was active is told whether it was wrong.

    Section 11 says "告诉系统它错了" ("tell the system that it was wrong"), which form (a) states most directly.
90. [F] (new) Telling every output line whether its activity matched, including the lines that were silent, would tell the cell on the line of the arriving byte that it should have been active. That hands that cell the answer, which sentence 86 excludes ("a single nerve cell cannot see the answer"). Form (c) of C37 is therefore restricted to lines that were active.
91. [C] (new) Whether a position at which nothing was emitted counts as wrong is choice C38.
92. [C] (new) When the signal is given and for how many steps it lasts is choice C39. Where it enters is choice C40. The narrower reading is that it enters at the output-end cells (decision D2). The wider reading is that every cell senses it directly. Section 11 says "整个系统" ("the whole system"), which may name the recipient as a whole. The wider reading is not assumed.
93. [F] (new) An output-end cell that receives the signal has, from which to correct itself:
    - the signal;
    - traces of its own activity at the emission, and of its inputs at that time;
    - its prediction and mismatch;
    - the chemical signal of its region.

    For the correction to reach the strengths that produced its activity at the emission, its traces must last at least from the emission until the signal arrives (sentence 56).
94. [C] (new) How an output-end cell uses the signal is choice C41.
95. [F] (new) A cell has one output (sentence 30). A cell can therefore relay the signal over its connections only by a change of its own activity, and that change reaches every target of the cell, including its output line.
96. [F] (new) When the signal is relayed over connections, an output-end cell's emission and its relay share one output. They must therefore be separated in time: the emission is read at steps at which the change is absent. Otherwise, the conversion of sentence 76 must not respond to that change. With one step per byte, the signal for one position and the emission for the next fall on the same step, so relay over connections requires more than one step per byte (C24).
97. [C] (new) Whether, and how, the signal changes the activity of the cells it enters, so that it is relayed over their connections, is choice C42.
98. [C] (new) Whether there are return connections, from which cells they come, and how many each cell has, is choice C43. Return connections are ordinary connections: they carry the sending cell's ordinary output and are adjusted like any other connection.
99. [F] (new) Consider an excitatory cell A that drives a cell B nearer the output end and receives a return connection from B. Suppose that A corrects itself by strengthening the inputs that were active when its arrival rose above its prediction. If being wrong raises B's activity, A's correction raises A's drive in the context that led to the wrong byte. With such a rule, therefore, the signal must lower the activity of the cells it enters, or the receiving cell must read the relayed change with the opposite sign (through an inhibitory cell, or by treating it as a factor of opposite sign).
100. [F] (new) A change relayed over connections reaches a cell at distance d from the output end no earlier than d steps after the signal reaches the output end. That cell's own activity contributed to the emission at least d steps before the emission. Let e be the number of steps from the emission to the signal. The interval between the cell's contributing activity and the relayed change reaching it is then at least 2d + e steps, and the cell's traces must last that long (sentence 56). With s steps per byte, this is (2d + e)/s bytes.
101. [F] (new) If each cell has r return connections, d relays reach at most r to the power d cells (with r = 50, two relays reach at most 2,500 cells). The change that reaches a cell at distance d is the sum of the changes in its return arrivals, each already mixed with the relaying cells' other activity.
102. [F] (new) An output-end cell may release its chemical signal on receiving the signal, since that is an occasion it detects from its own quantities (sentence 115). The release reaches the cells of its region, connected or not, and no farther (sentence 34). If a cell's release could be caused by another cell's release without limit, one occasion would reach every cell through a chain of releases. That would make the chemical signal reach the whole tissue in effect, which section 9 excludes. Release caused by a received chemical signal is therefore either absent or limited in the number of times it is passed on (C26).
103. [F] (new) What a cell has, at each distance from the output end, from which to correct itself:
    - **Distance 0:** sentence 93.
    - **Distance 1:** changes in its arrivals over return connections from output-end cells, at least one step after the signal; the chemical signal of output-end cells whose region contains it; its traces, prediction and mismatch. It does not have the signal itself, and it never has the byte that was right.
    - **Distance 2 and more:** only the relayed changes in its arrivals and the chemical signal of its own region, with its traces, prediction and mismatch.

    At every distance, the cell computes its own change from these quantities (sentence 53).
104. [F] (new) With one part, a cell cannot tell a relayed change from its other arrivals. The relayed change then acts on the cell only through its mismatch. It changes the cell's output only if the strengths that form the prediction are also those that form the activity, or if the mismatch enters the output (C10). With two parts, the return arrivals can arrive at one part and the forward arrivals at the other, and the cell can treat them differently (sentence 8).
105. [C] (new) How a cell combines the correction relayed to it with the prediction of its own arrivals is choice C44.

### 2.2 Living counterparts of section 2.1

**What is compared and when (sentences 87 and 88).** An animal's nervous system compares what happens with what it expected. In some systems this comparison is carried by identified cells. The cells of the inferior olive send one climbing fibre to each Purkinje cell of the cerebellum, and that fibre fires when a movement goes wrong or an unexpected stimulus arrives. Dopamine cells of the midbrain fire more when an outcome is better than predicted and less when it is worse (Schultz, Dayan and Montague, 1997). In these animals the comparison is made by cells inside the organism, from their own inputs. In this definition the world makes it, as Mr. Xu's words allow ("你可以给"). The order in time matches the animal case: the signal follows the act it concerns, by tens to hundreds of milliseconds.

**The two forms (C37).** Form (a), one quantity for each byte, corresponds to dopamine. Dopamine is one quantity at a time, released over a region, and its dip on a worse outcome tells the region that the outcome was worse, not what would have been right. In songbirds, dopamine cells signal to one region of the song system whether a sung syllable was better or worse than usual (Gadagkar and colleagues, 2016). Form (c), a message to each active output line, corresponds to the climbing fibre. One climbing fibre reaches one Purkinje cell, and its effect depends on which parallel-fibre inputs were active just before. Whether the climbing fibre signal is an error signal under all conditions is unsettled.

**Entry at the output end (C40).** In both counterparts the signal reaches a defined set of cells, not every cell of the brain: a climbing fibre reaches its own Purkinje cells, and a dopamine cell's axon reaches one region. This supports the narrower reading. The wider reading has no clear counterpart in a tissue without a widely projecting system (check, sentence 80).

**Use at the output end (C41).** When a climbing fibre fires shortly after the parallel-fibre inputs to a Purkinje cell, those inputs weaken (long-term depression). This is textbook. It is a change computed at the synapse from the signal, the trace of the input and the cell's own state, which is form (a) of C41. Dopamine arriving within about 0.3 to 2 seconds after coincident activity at a corticostriatal synapse decides whether the synapse is strengthened (Yagishita and colleagues, 2014). This gives traces of the length sentence 93 requires and a counterpart of form (b).

**Relay over connections (C42, C43, sentences 95 to 101).** After a climbing fibre input, a Purkinje cell stops its ordinary firing for a period (a pause), so its targets receive a fall of activity. This is a counterpart of form (b) of C42 and of the sign required by sentence 99. Reciprocal connections between pairs of cortical cells are more common than chance would give (Song and colleagues, 2005), which is a counterpart of form (b) of C43. That an error is relayed in cortex over several connections, each cell correcting itself, is unsettled. Rules of this kind learn small tasks in simulation, for example with fixed random return connections (Lillicrap and colleagues, 2016) and with cells of two parts (Sacramento and colleagues, 2018; Payeur and colleagues, 2021). These are models, not findings in tissue.

**Delay (sentences 96 and 100).** In living tissue one synaptic delay is about a millisecond and a stimulus lasts tens to hundreds of milliseconds, so cells exchange signals many times during one stimulus. More than one step per byte is therefore the living case. Traces of a second or more are established at living synapses (Yagishita and colleagues, 2014; Bittner and colleagues, 2017).

**Chemical relay (sentence 102).** Dopamine and other modulators are released over a region from the axons of a few cells. Neurons also release peptides from their dendrites when they fire strongly (Ludwig and Leng, 2006). That such release is relayed from region to region without a dedicated projecting system is not known to this reader.

**Combination with prediction (C44, sentence 104).** In the hippocampus, input from the entorhinal cortex to the distal dendrite of a CA1 cell can trigger a long depolarisation (a plateau potential). That plateau changes the strengths of inputs at the proximal dendrite that were active within some seconds before or after it (Bittner and colleagues, 2017). This is a living counterpart of form (b) of C44: an input at one part changes the strengths of the inputs at another part. Whether such an input carries a correction is unsettled. In cortex, return connections from higher areas end largely on the distal dendrites of pyramidal cells. That they carry predictions or corrections is a hypothesis.

## 1 (continued). Time, chemical signal, lasting change, numbers, and what does not exist

### 1.9 Time

106. [C] (former 71) The input is one byte after another and one item after another, and nothing marks where an item begins or ends other than the bytes themselves. This is choice C45, a condition of the project. It will be marked [S] when the ruling is quoted.
107. [C] (former 72 and 73) How many steps take place between two bytes, and in what order cells compute, is choice C24.
108. [F] (former 74) A cell whose shortest path from the encoder is d connections long receives the effect of a byte no earlier than d steps after the encoder reads it.
109. [F] (former 75, reworded) Nothing outside the cells sets their states back at the beginning or end of an item, because nothing marks an item (C45) and the cells are the only agents in the tissue (sentence 1). A cell may come, by its own adjustment, to clear its state when particular bytes arrive.
110. [F] (former 76) The tissue meets one stream, one byte after another, and there is one tissue. There are no copies of it receiving several items at once.
111. [C] (former 77) The cells' states at the start of the tissue's life are choice C25. This start happens once and is not a reset.

### 1.10 The chemical signal

112. [S] (former 78, reworded) A cell's chemical signal reaches a region near it (section 9: "the signal is local"). This definition takes the signal as received alike by the cells of that region. That is an idealisation: in living tissue the concentration falls with distance from the release site.
113. [S] (former 79) "How each receiving cell reads that signal is another matter" (section 7, point 2).
114. [F] (former 80, mark corrected) A chemical signal released by a cell never reaches the whole tissue (sentence 112). In a whole brain, a few small groups of cells (for example the noradrenergic cells of the locus coeruleus) reach almost the entire forebrain. The tissue defined here contains no such group.
115. [F] (former 81, updated) A cell can release its chemical signal only on an occasion it detects from its own quantities (sentence 53). For an output-end cell, these include the signal.
116. [C] (former 82) Which cells release, and on what occasion, is choice C26.
117. [C] (former 83) The size of the region is choice C27.
118. [F] (former 84, reworded) Under the idealisation of sentence 112, the chemical signal cannot tell a receiving cell which of its connections to change. Any effect on one connection rather than another comes from the receiving cell's own quantities. If C27 takes the concentration as falling with distance, the signal acts more on the synapses nearer the release site.
119. [C] (former 85) What a receiving cell does with the chemical signal is choice C28.

### 1.11 Lasting change and the tissue's own reactivation

120. [C] (former 86) Long-term memory forms in the course of learning and not by an act of a program. This is choice C46, a condition of the project, to be marked [S] when quoted.
121. [F] (former 87) No step of the program marks a change as lasting, freezes connections, or copies the tissue (C46).
122. [C] (former 88) How a change becomes lasting is choice C31.
123. [F] (former 94, corrected) No program passes over an item, or over stored bytes, after the item has ended, because no cell reads past bytes (sentence 19) and nothing outside the cells acts on the tissue (sentence 1). The tissue's own reactivation of recent activity is neither excluded nor imposed. Living hippocampus reactivates sequences of recent activity during rest, in forward and reverse order.

### 1.12 Numbers

124. [C] (former 89) A number that must be fixed is varied on the grid 2, 4, 8, ..., 128. Fewer fixed numbers are better. Numbers of cells may go beyond 128 by further powers of two. This is choice C47, a condition of the project, to be marked [S] when quoted.
125. [C] (former 90) For quantities smaller than one, the grid is read as 1/2, 1/4, ..., 1/128 (part of C47).

### 1.13 What does not exist in the tissue

126. [F] (former 91, updated) No cell is handed the right answer. The answer reaches the tissue only as bytes at the input end (sentence 83). The signal says that the emission, or an active line, was wrong, and never which byte was right (sentences 89 and 90).
127. [F] (former 92, corrected) No cell reads every cell, because a cell receives only from cells in its neighbourhood (sentence 24). The output end is a set of cells like the others.
128. [F] (former 93, updated) No error passes backward through the connections that carry the forward signal, and there is no single backward pass over the network (sentence 86). A connection carries only its sending cell's output, in one direction. What travels on a return connection is that cell's ordinary output.
129. [F] (former 95, updated) The only quantity computed outside the cells and given to them is the signal. It is computed from the emission and the arriving byte, and it is given where it enters (C40). No other quantity over a population of cells is computed and given to cells, because learning is local (sentence 52).
130. [F] (former 96, updated) No correction is computed for a cell outside that cell. The signal states only that the system, or an active output line, was wrong. Each cell computes its own change from its own quantities (sentence 53). No cell uses another cell's strengths.
131. [F] (former 97) There is no reset of cells' states at each item (sentence 109).
132. [F] (former 98) There are no items presented at once in copies, and no change made from their average (sentence 110).
133. [F] (former 99) There is no window of past bytes beyond the short span of the input end (sentences 7 and 67).
134. [F] (former 100, corrected) The only criterion applied to the tissue as a whole is the world's: whether the emission matched the arriving byte (sentence 87). It reaches the cells only as the signal where it enters, and from there by relay. No cell is given any other goal.
135. [F] (check 121, changed) No sentence of this definition hands a cell the right answer or a correction computed for it (sentences 126 and 130).

**Counts.** [S]: 25 sentences. [F]: 66 sentences. [C]: 44 sentences. Total 135. The table in section 3 lists 44 choices.

## 3. The choices

The first form is the simplest form a living cell or tissue can do that the roadmap's section 5 allows. The further forms are listed in the order in which they are to be opened. G means the grid 2, 4, ..., 128 (extended for numbers of cells), and 1/G means 1/2, ..., 1/128. The last column says whether a cell could set the number itself from its own quantities.

Changes from the description's table:
- C4 (negative outputs) and C9 (different outputs to different targets) are no longer choices (sentences 14 and 30).
- C3 keeps only the share of inhibitory cells (sentence 12).
- C22 keeps only whether encoder cells predict (sentence 72).
- C36 of the check is withdrawn.
- C37 to C47 are new.

| No. | Choice (sentences) | First form | Further forms, in the order to open | Numbers (first value) | Set by the cell? |
|---|---|---|---|---|---|
| C1 | Parts of a cell (9) | One part | Two parts: forward arrivals at one, return arrivals at the other (opened at roadmap step 3) | None | The assignment of a new source to a part, possibly; the number of parts, no |
| C2 | Gate (11) | No gate | A gate on how much of the state is kept, computed from the cell's own arrivals and state; gates on input and output (unsettled as biology) | None | Gate strengths, yes |
| C3 | Share of inhibitory cells (13) | 1/4 of cells inhibitory, sign fixed for life | 1/8; 1/2 | 1/4 (1/G) | No |
| C5 | Strength of a connection (25) | Graded, between zero and a maximum of 1 | A count of synapses each present or absent | Starting strength 1/8 of the maximum (1/G) | The strength, yes |
| C6 | Number of incoming connections (26) | Fixed per cell; one new connection made each time one is given up | A number that changes with making and giving up | 16 (G) | Yes, if making and giving up are untied |
| C7 | Connections at the start (27) | Drawn at random within the neighbourhood (encoder cells: from input lines) | Drawn with a bias toward the input end | As C6 and C30 | Not at the start |
| C8 | Encoder cells receive from other cells (28) | No | Yes, on a second part | None | No |
| C10 | What the output carries (33) | Activity from arrival and state alone | Activity into which the mismatch enters, with sign carried by two kinds of cell (opened at step 3 together with C44) | Weight of the mismatch in the output, 1/G (second form only) | Possibly |
| C11 | Retention of a cell's state (41) | One decline rate for all cells | A rate drawn for each cell; a rate the cell adjusts | 8 steps (G) | Possibly, from its mismatch |
| C12 | Number of predictions (47) | One, for the total arrival | One for each part | None | No |
| C13 | What predicts what (48) | All arrivals predicted from the cell's own state | Arrivals at one part predict the next arrivals at the other part | None | No |
| C14 | Form of the comparison (50) | Signed difference, held as a quantity inside the cell | Magnitude; whether it exceeds a level | Level, 1/G (third form only) | The level, yes |
| C15 | How a strength changes (55) | Rate times the cell's mismatch times the trace of the input, within the bounds of sentence 55 | Coincidence of the input's trace with the cell's own activity (long-term potentiation) | Rate 1/32 (1/G) | Possibly, from the running size of its mismatch |
| C16 | Length of traces (57) | A trace declining over several steps | Separate traces for strengthening and weakening | 8 steps (G); at least 2d + e steps for relay (sentence 100) | Possibly |
| C17 | When a connection is given up (58) | Strength below a level, relative to the cell's mean strength, for a period | No reduction of the mismatch over a period, measured by the product of the input's trace and the mismatch | Level 1/8 (1/G); period 128 steps (G) | Yes, relative level |
| C18 | When and to whom a new connection is made (61) | Each time one is given up, to a cell drawn at random within the neighbourhood | Toward a site where the local chemical signal of nearby active cells coincides with the cell's own activity (sentence 59); when the mismatch stays high | None beyond C5 | When, yes; to whom, only at random or by the local signal |
| C19 | Incoming or outgoing connections adjusted (62) | Incoming only | Both | None | No |
| C20 | The encoder cells' passages (70) | Formed by traces of input lines (sentence 69), with no separate numbers | Passages at positions drawn at random relative to the present byte | Passage length, gap length, passages per cell, G (second form only) | Yes in the first form, through the cell's own connections and traces |
| C21 | How an encoder cell responds to a byte (71) | Through the lines it is connected to, each line one byte value (C32) | Lines for single bits | None | Through its connections |
| C22 | Whether encoder cells predict (73) | Yes, by the same rule as other cells | No; they adjust by coincidence alone | None | No |
| C23 | Numbers of cells (74) | 128 encoder cells; 128 inner cells; 256 output-end cells (one for each output line) | Larger numbers of encoder and inner cells by powers of two | 128, 128 (G); 256 fixed by C33 | No |
| C24 | Steps per byte, and order (107) | 4 steps per byte, all cells at once | 8, 16; 1 step per byte only where no relay is used (sentence 96); cells in random order | 4 (G) | No |
| C25 | States at the start (111) | All states at rest (zero) | States drawn at random | Spread, 1/G (second form only) | No |
| C26 | Release of the chemical signal (116) | No release | An output-end cell releases on receiving the signal; every cell releases when its activity exceeds its running average by a factor; release caused by another's release, passed on at most a fixed number of times (sentence 102) | Factor, G; averaging period, G; times passed on, 1 (later forms only) | The averaging, yes |
| C27 | Region reached by a release (117) | The releasing cell's neighbourhood, received alike | Concentration falling with distance | Region in cells, as C30 | No |
| C28 | What a receiving cell does with the chemical signal (119) | Scales the rate of its strength changes | Permits making or giving up of connections; permits a change to become lasting | Sensitivity of each cell, 1/G (later forms) | Possibly |
| C29 | Control of a cell's own activity (15) | A threshold that follows the cell's own recent activity toward a target | Multiplication of all its excitatory strengths by a common factor from its own activity (synaptic scaling); a threshold dividing strengthening from weakening that moves with recent activity | Target 1/8 (1/G); period 128 steps (G) | Yes, from its own activity |
| C30 | Neighbourhood (17) | A fixed set of cells drawn for each cell, from among the cells its processes reach | Cells placed in a space, with connection probability falling with distance | 64 cells (G) | No |
| C31 | How a change becomes lasting (122) | A single component of each strength, which stays as changed | A fast and a slow component; lasting only while the chemical signal is received; longer-lived connections given up less readily (opened at roadmap step 4) | Slow period, G; ratio, G (later forms) | Partly |
| C32 | Input lines (68) | Fixed lines, one for each of the 256 byte values, active when the present byte has that value (decision D1) | Eight lines, one for each bit | 256 (fixed by the form) | No |
| C33 | Output lines and conversion (77) | One line for each of the 256 byte values, each driven by one output-end cell; a byte emitted when exactly one line exceeds a level; output-end cells also send to inner cells | Eight lines, one for each bit; selection of the most active line (sentence 78) | Level 1/2 of the maximum activity (1/G) | No |
| C34 | When the tissue emits (80) | At every byte position, read at the last step before the next byte arrives | Whenever a line exceeds its level | None | No |
| C35 | Reading its own emission (84) | No | Through the same input lines, before the arriving byte; through separate lines | None | No |
| C37 | What the system is told (89) | (a) One quantity for each position, 1 if the emission was wrong and 0 otherwise | (c) Each active output line told whether it was wrong (restricted by sentence 90), run beside (a) at roadmap step 2 by decision D3; (b) a graded quantity | None | No |
| C38 | No emission (91) | Counts as wrong | Counts as nothing where nothing can be expected (capability l of the roadmap) | None | No |
| C39 | When the signal is given and how long it lasts (92) | At the first step after the byte arrives, for one step | For all steps until the next byte | Duration in steps, G (second form only) | No |
| C40 | Where the signal enters (92) | Every output-end cell alike (with C37 a); the cell of the active line (with C37 c) | The wider reading, every cell, is not used | None | No |
| C41 | How an output-end cell uses the signal (94) | Each strength is lowered by the rate of C15 times the signal times the trace of its input times the trace of the cell's own activity | The cell uses the signal minus its own running average of the signal, so that being right more often than usual strengthens; the signal enters as an arrival at a second part (C44) | Same rate as C15; averaging period, G (second form) | The average, yes |
| C42 | Relay of the signal over connections (97) | None: the signal acts at the output end only (roadmap step 2) | The signal lowers, for one step, the activity of the cells it enters (sentence 99); it raises it, read with opposite sign by the receiver | Size of the change, 1/G (later forms) | No |
| C43 | Return connections (98) | None (with C42 first form) | For each connection from A to a cell B nearer the output end, a connection from B to A made at the start; return connections drawn at random from cells farther from the output end within the neighbourhood | Return connections per cell, G (Mr. Xu's example of fifty lies between 32 and 64) | Yes, by C17 and C18 like any connection |
| C44 | Combining relayed correction with prediction (105) | One part: the relayed change enters only through the cell's mismatch (sentence 104), with C10 opened to its second form | Two parts: the mismatch at the return part changes the strengths of the forward part in proportion to their traces; the local chemical signal of output-end cells as a factor on the rate (C28) | Weight of the second part, 1/G (second form) | Possibly |
| C45 | The stream (106) | One byte after another; items separated only by bytes | None | None | No |
| C46 | Long-term memory (120) | Forms only by the cells' own changes | None | None | No |
| C47 | The grid (124, 125) | 2 to 128, reciprocals for fractions, cells beyond 128 by powers of two | None | None | No |

## 4. Every number the first form fixes

The following numbers are fixed by us in the first form. Each lies on the grid, or is fixed by a decision of the roadmap.

1. Input lines: 256, one for each byte value (C32; decision D1).
2. Output lines and output-end cells: 256 (C33; decision D1).
3. Encoder cells: 128 (C23).
4. Inner cells: 128 (C23).
5. Share of inhibitory cells: 1/4 (C3).
6. Maximum strength: 1, the unit of strength (C5). Starting strength: 1/8 of the maximum (C5).
7. Incoming connections per cell: 16 (C6).
8. Neighbourhood: 64 cells (C30).
9. Steps per byte: 4 (C24).
10. Retention of a cell's state: 8 steps (C11).
11. Trace length: 8 steps (C16).
12. Rate of strength change: 1/32 (C15). The same rate serves the signal at the output end (C41).
13. Level for giving up a connection: 1/8 of the cell's mean strength (C17). Period for giving up a connection: 128 steps (C17).
14. Target activity: 1/8 (C29). Period of activity control: 128 steps (C29).
15. Level at which an output line emits: 1/2 of the maximum activity (C33).
16. Duration of the signal: one step (C39). The value of the signal is 1 when the emission was wrong and 0 otherwise (C37).

Counted as numbers that could be varied, the first form fixes 15: the twenty values listed above less the five fixed by a decision or as a unit (input lines, output lines, maximum strength, duration of the signal, value of the signal). Of these 15, the cell could in later forms set six itself: the level and period for giving up a connection, the target and period of activity control, the rate, and the retention.

## 5. What the definition still does not determine

1. **Whether the first form learns at all.** That is the question of steps 1 to 3 of the roadmap. In particular, the first form of C42 and C43 confines correction to the output end. Relay inward is defined (sentences 95 to 104) but is first switched on at step 3.
2. **The form of the signal.** Mr. Xu's words "告诉系统它错了" ("tell the system that it was wrong") are met by form (a) of C37. Whether it is to say how wrong (form b), or which active line was wrong (form c), is our decision D3, not his words, and he may settle it.
3. **The reading of "整个系统" ("the whole system").** His words admit the narrower reading (the signal enters at the output end) and the wider reading (every cell senses it). The definition takes the narrower one as our decision D2. It does not treat this as unanswered by Mr. Xu; it is a point on which his confirmation would replace a choice.
4. **How one cell combines its prediction of its own arrivals with a relayed correction (C44, sentence 104).** Mr. Xu's words give both sources. How they combine in a living cell is unsettled in neuroscience, and it is the roadmap's conflict 10.
5. **How a change becomes lasting (C31, C46).** This is to be chosen at roadmap step 4.
6. **Three conditions (C45, C46, C47)** rest on rulings that are not quoted in sections 7 to 11. Their wording must be quoted to be marked [S].
7. **What living tissue does that the definition leaves out:** the concentration of a chemical signal falling with distance; continuous rather than stepped time; the tissue's own reactivation during rest. Each is admitted as a later form of a choice (C27, C24, sentence 123) and is not assumed in the first form.

## References

- Bittner and colleagues (2017), Science 357: 1033-1036.
- Gadagkar and colleagues (2016), Science 354: 1278-1282.
- Lillicrap, Cownden, Tweed and Akerman (2016), Nature Communications 7: 13276.
- Ludwig and Leng (2006), Nature Reviews Neuroscience 7: 126-136.
- Payeur, Guerguiev, Zenke, Richards and Naud (2021), Nature Neuroscience 24: 1010-1019.
- Sacramento, Ponte Costa, Bengio and Senn (2018), Advances in Neural Information Processing Systems 31.
- Schultz, Dayan and Montague (1997), Science 275: 1593-1599.
- Song, Sjöström, Reigl, Nelson and Chklovskii (2005), PLoS Biology 3: e68.
- Yagishita and colleagues (2014), Science 345: 1616-1620.

The remaining facts are textbook neuroscience: Dale's principle, the climbing fibre and long-term depression in the cerebellum, the pause after a complex spike, hippocampal replay, and synaptic delay. The references of the check apply where the same facts are used. The page numbers are given from this reader's memory and should be checked before the list is cited elsewhere.
