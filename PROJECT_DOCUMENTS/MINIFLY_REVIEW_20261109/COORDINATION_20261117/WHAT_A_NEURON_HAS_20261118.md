# What a neuron has: the page from which the list of experiments is to be derived (18 November 2026, draft 1, for Mr. Xu's correction)

By the outside reviewer. Mr. Xu has asked several times for a list of experiments made from the biological principle. The three lists the reviewer delivered (17 November; 18 November afternoon; 18 November evening) each started from the existing network and used his statements to test it. None started from the neuron. This page is the missing start. It states, positively, what a living neuron and a living synapse have, and what does not exist in living tissue. It contains no experiment. The list of experiments is derived from it only after Mr. Xu has corrected it.

The statements are textbook neuroscience as the reviewer knows it, not Mr. Xu's words, unless marked. Where the science is unsettled, the statement is marked *unsettled*. Where the reviewer is unsure, it says so.

## 1. A cell

1. **One output.** A neuron's output is its own activity (its firing over time). The same output goes to every cell it contacts. It cannot send one signal to one target and another to a second target, and it cannot send a second signal backward along the same connection. Its output is not negative: a cell fires more or less, or not at all.
2. **Inputs over its own synapses.** It receives the outputs of a limited number of other cells, a small share of all cells. It receives nothing else from them: not their weights, not their inputs, not their errors.
3. **A fixed kind.** A cell is excitatory or inhibitory. The sign of its effect on others belongs to the cell and does not change with learning; only the strength of a synapse changes. (Dale's principle; it has known exceptions, which do not make the sign learnable.)
4. **Its own state.** Its present level of excitation, its recent activity over seconds, and slower changes over minutes to days. *Unsettled, and possibly important:* a cell of the cortex has separate parts (dendritic branches) on which inputs from different sources arrive, and it may treat them differently.
5. **Time is continuous.** A signal crosses a synapse in milliseconds. While one stimulus lasts, cells exchange signals many times. Nothing tells a cell where an item begins or ends, nothing sets its state to zero, and it meets one thing after another, never many at once.

## 2. A synapse

A synapse changes its strength from what is present at the synapse:

1. the activity that arrives from the sending cell;
2. the state and activity of the receiving cell;
3. its own recent past: a trace of 1 and 2 that lasts seconds;
4. a diffuse signal released over a wide region and received by many cells alike (for example on reward, punishment or surprise). It is one signal for all, not a separate instruction for each cell;
5. slower processes of its own: a change that is at first easily undone and later becomes lasting, depending on the synapse's history; and synapses are formed and removed, so that the number of a cell's connections is itself changed by activity.

## 3. What the world gives the learner

1. Input, over lines that each respond to something particular.
2. The consequences of what it does, later in time.
3. Where it is taught, the right answer arrives as further input after its own attempt (it sees or hears the answer), as activity on input lines. It does not arrive as a computed error for each cell.
4. *The organism's own repetition:* during rest, recent activity is replayed by the brain itself. This is the biological form of rehearsal; it is the learner's act, not a store kept by an outside program.

## 4. What does not exist

Each of these is in the present network or its training.

1. An error passed backward through the connections that carry the forward signal, whether exactly or by the signs of the weights; any use by one cell of another cell's weights.
2. A quantity computed over a whole population of cells or over all the weights of a map (a scale for a matrix; a normalisation over all outputs computed exactly).
3. A pass over an item after it has ended, in reverse order.
4. Many items at once in separate copies of the network, and a change of a weight from their average.
5. A reset of every cell's state by an outside agent; a marked beginning and end of an item.
6. A count of updates shared by all weights.
7. A synapse whose effect changes sign in learning; a cell with a negative output.
8. Cells added, weights frozen or projected, or directions computed, by an agent that knows when the task changes.
9. On the side of measurement: a quantity that requires recording every cell over hundreds of items and imposing an average on the tissue. Removing or silencing cells exists as an experimenter's act and has a meaning for a living learner (can it still do the task without them); a clamp at a mean over items has none.

## 5. What the reviewer is unsure of, and puts to Mr. Xu

1. Whether the gated cell of the present network (three internal gates, each a weighted sum of the cell's inputs) has a counterpart, or whether the cell should be a simpler unit.
2. Whether statement 1.4 on separate parts of a cell is to be used. Several proposals for local learning rest on it.
3. Whether the right answer shown as input after the attempt (3.3) is the form of teaching he intends.

## 6. How the list will be derived from this page

Not by changing the present network one part at a time. The first question is: a population of cells as in section 1, with synapses as in section 2, taught as in section 3 and with nothing of section 4: what is the simplest form of the present task that it learns at all, and what does it keep of it? The present network enters only as the reference that the record already holds.

## 7. Mr. Xu's corrections of draft 1 (18 November 2026, 22:03 UTC), which override the text above

1. On the three points of section 5: all three are accepted. The gated cell may stay; separate parts of a cell that treat inputs from different sources differently may be used; the right answer shown as input after the learner's own attempt is the form of teaching.
2. **A cell can send a uniform chemical signal to large numbers of other cells; how each receiving cell reads that signal is another matter.** Statement 1.1 (one output, nothing else) is therefore too narrow, and so are statement 2.4 and item 4.2. Corrected: beside its electrical output to the cells it contacts, a cell can release a chemical signal that reaches many cells alike, and each receiving cell reads it in its own way, so that one signal can act differently in different cells. What does not exist is an instruction computed separately for each cell and delivered to it, and the use by one cell of another cell's weights; a signal that is the same for many cells is not excluded.
3. The verification runs and re-runs are the coordinator's work and not the reviewer's; the reviewer reviews and arranges the tasks of other agents.

## 8. Mr. Xu's statement of 18 November 2026, 22:15 UTC, which governs everything above (his words, then the reviewer's translation)

The reviewer had asked whether what the learner does can be put as "expecting the next byte from the bytes already read". Mr. Xu's answer:

> 不是，我们没有专门的"学习者"。除了byte-level encoder之外，所有的神经细胞能够做到的只是从其他输入细胞得到输入，然后根据输入和现有状态给输出连接的部分送出输出，并不断调整自己连接那些细胞。因此学习是局部的。它未必能够读到已有的字节（不可能储存一百万个字节作为信息窗口），只能读到过去储存的状态和通过其他细胞得到的新的信息。这对byte level encoder也适用，它可能读到的只是输入信息中间有间隔的某些小段落的一个总集，不同的读到的不一样。因此对不同的细胞，它们得到的新的信息也不同，不能简单认为是"预期下一个到来的字节"。这应该被认为是预测从其他细胞得来的新来的信息，这可能来自旧的字节读取后的处理结果。

Translation (the reviewer's). No: there is no separate "learner". Apart from the byte-level encoder, all that any nerve cell can do is to receive input from other cells, to send, from that input and its present state, an output to the cells it is connected to, and to keep adjusting which cells it is connected to. Learning is therefore local. A cell cannot in general read the bytes that have passed (a million bytes cannot be kept as a window); it can read only the state it has stored and the new information that reaches it through other cells. The same holds for the byte-level encoder: what one of its cells reads may be only a collection of small passages of the input with gaps between them, and different cells read different ones. So the new information differs from cell to cell, and what a cell does cannot be put as "expecting the next byte". It is to be put as predicting the new information that arrives from other cells, which may be the result of processing bytes read earlier.

What follows at once, in the reviewer's words: there is no cell that is the output and no cell that receives an answer; there is no objective of the whole; every cell has the same task, which is its own; and a cell's adjusting of its connections is part of that task from the first step, not a later refinement.

## 9. A further correction by Mr. Xu (22:18 UTC): the chemical signal is local

Section 7, point 2, as the reviewer wrote it ("reaches many cells alike") and the reviewer's use of it ("a signal to all cells") are wrong. Mr. Xu: a cell may be able to release a chemical substance, a signal, into a region near it; the signal is local. This had been discussed before (his statement of 16 November that a neuron acts and adjusts only within its own neighbourhood), and the reviewer had forgotten it. Corrected: a cell's chemical signal reaches the cells of a region near the releasing cell, alike; it never reaches the whole tissue; how each receiving cell reads it is that cell's own matter. No signal of any kind reaches every cell.

## 9a. Mr. Xu's statement of the goal (18 November 2026, about 22:22 UTC), on the reviewer's draft 2 of a list

> 但我们要求做到的是有泛化能力的end to end persistent learning，你列出来这些清单有什么帮助呢？我还是无法理解。

Translation (the reviewer's). But what we require is end-to-end persistent learning that is able to generalise; how do the lists you have drawn up help with that? I still cannot understand.

## 10. Mr. Xu's ruling of about 22:25 UTC: the two ends, and what was asked for

The reviewer had written that "end to end" implies a byte-level output end beside the encoder, and had marked this as its own inference. Mr. Xu:

> 我让你做的是为了达到这个目标，如何将它变成一步一步的路线图，因为各个目标之间势必有依赖性，有重叠。对有一个输出端和输入端不是问题，这两个也需要通过学习来学会如何输入/输出。为什么项目到了现在你还是有这么多迷惑的地方？我们这个项目一开始就是这个方向，从来没有变过！

Translation (the reviewer's). What I asked of you is how, in order to reach this goal, to turn it into a step-by-step roadmap, because the goals necessarily depend on one another and overlap. That there is an output end and an input end is not a problem; these two also have to learn, through learning, how to take in and how to give out. Why, at this point in the project, are you still confused about so many things? This project has had this direction from the beginning; it has never changed.

What follows: there is an input end and an output end; the cells at both ends learn as the other cells do; the tissue therefore acts, by emitting bytes. The direction is the one the project's objective contract of 26 September 2026 states. The roadmap asked for is `ROADMAP_TO_END_TO_END_PERSISTENT_LEARNING_20261118.md`.

## 11. Mr. Xu's statement of 22:46 UTC: the system is told when it is wrong and corrects itself

The reviewer had reported, as the largest open point of its roadmap, that nothing tells a cell at the output end whether what the tissue gave out was right. Mr. Xu:

> 所以你可以给整个系统连续的信号，如果输出端的信息和预期不符合（例如预测next byte出现错误），告诉系统它错了让它自己去纠正。这本来就是我们persistent learning的目标，为何你到现在才提起？我感觉非常怪异。

Translation (the reviewer's). So you can give the whole system a continuous signal: if the information at the output end does not match what was expected (for example, the next byte was predicted wrongly), tell the system that it was wrong and let it correct itself. This has been the aim of our persistent learning all along; why do you bring it up only now? I find this very strange.

What follows. The point was not open. The system receives a continuous signal of being wrong, formed from the mismatch at its output end, and corrects itself; that loop is what persistent learning means in this project. It agrees with his earlier words: information may be relayed from neuron to neuron so that a few passes cover the network, while a single backward pass over the whole network is not allowed; and a single nerve cell cannot see the answer. The reviewer's sentences that go beyond his words and are withdrawn: "no error sent back" and "no objective of the whole" (`ORDER_FROM_BIOLOGY_20261118.md`, draft 2), "there is no cell that is the output" (section 8 above, last paragraph), and the wording "no cell is given an answer or an error computed for it" in the briefs of the three readers of 22:31 UTC, in so far as it excludes a signal of being wrong given to the system. What stays excluded: handing a cell the answer, and computing a cell's correction outside the cell.

## 12. An earlier ruling by Mr. Xu, quoted here because the definition of the tissue rests on it (18 November 2026, afternoon)

> 1）有没有办法让每个细胞自己决定要进行多少条连接，将这个也变成学习的一部分？我们预先给定的参数越少，长期来看越好。2）如果暂时不行，可以采取log_2的形式，从2, 4, 8, 16,32,64,128等等取一些数字来看效果如何。

Translation (the reviewer's). (1) Is there a way to let each cell decide for itself how many connections to make, so that this too becomes part of learning? The fewer parameters we fix in advance, the better in the long run. (2) If that is not possible for the moment, a log₂ form can be used: take some numbers from 2, 4, 8, 16, 32, 64, 128 and so on, and see what the effect is.

The requirement that long-term memory form by itself in the course of learning, with some rehearsal acceptable (called requirement A in the record), and the description of the stream as bytes with nothing that marks an item, are Mr. Xu's by the record of 17 and 18 November, but their words are not quoted here; the definition marks them as choices until they are.
