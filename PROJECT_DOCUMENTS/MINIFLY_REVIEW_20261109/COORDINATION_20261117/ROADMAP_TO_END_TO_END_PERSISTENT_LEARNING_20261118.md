# Roadmap to end-to-end persistent learning with generalisation (18 November 2026, draft 2, for Mr. Xu)

By the outside reviewer. Draft 1 is replaced. It had faults found by three readers, and it left out the centre of the matter, which Mr. Xu stated at 22:46 UTC: the system is told, continuously, when what its output end gives does not match what was expected, and it corrects itself. This file starts nothing; releases and runs are the coordinator's.

## 1. The goal and the principle

The project's objective contract (`minifly_b/package/DOCS/MINIFLY_OBJECTIVE_CONTRACT.md`, effective 26 September 2026):

> Build a bounded, biologically informed learner that uses information acquired from earlier byte-level experience when it becomes useful later, despite delay and intervening learning. The eventual test is reuse of learned information on a relevant, previously unreinforced input or task, with the answer selected before feedback.

Mr. Xu's words, all recorded in `WHAT_A_NEURON_HAS_20261118.md` (the section is given):

1. A cell receives input from other cells, sends an output from that input and its own state, and keeps adjusting which cells it is connected to. Learning is local. A cell has its stored state and what newly arrives through other cells, and nothing else (section 8).
2. A chemical signal released by a cell reaches a region near it and never the whole tissue (section 9).
3. The goal is end-to-end persistent learning that generalises (section 9a).
4. There is an input end and an output end, and both learn how to take in and how to give out (section 10).
5. **The system is given a continuous signal: when the information at the output end does not match what was expected, for example when the next byte was predicted wrongly, the system is told that it was wrong and corrects itself. This is what persistent learning is** (section 11).
6. Earlier (16 to 18 November): a single backward pass over the whole network is not allowed, but information can be relayed from neuron to neuron, one cell to fifty, fifty to fifty times fifty, so that a few passes cover the network; and a single nerve cell cannot see the answer.

These fit together as follows. No cell is handed the answer, and nothing outside a cell computes that cell's correction. The system is told when its output was wrong. Inside the system that message travels as everything travels: it arrives where the mismatch arises, at the output end; it is relayed from cell to cell over connections, and by chemical signals that act locally; and each cell corrects itself from what reaches it, together with its own input and state.

**The reviewer's error, stated once.** From "a single nerve cell cannot see the answer" and "there is no separate learner" the reviewer removed every signal of right and wrong from the system, wrote "no error sent back" into its own list, gave that stricter wording to three readers, and then reported the absence of such a signal as the largest open point of the route. It was never open. Draft 1's capability "the output end is learned by hearing itself" and its overlap 6 are withdrawn.

## 2. What persistent learning is in this system

At every byte the tissue gives out what it expects. The byte arrives. The system is told when it was wrong, and the cells correct themselves. This never stops and nothing is reset. The three properties of the goal are aspects of this one loop:

- **End to end:** bytes at both ends, and both ends corrected by the same loop.
- **Persistent:** the loop runs for the whole life; what was corrected earlier must still hold after a delay and after other corrections.
- **Generalising:** the attempt always comes before the signal, so an instance that was never taught is answered before feedback; whether that first attempt is right is the test.

What grows along the route is therefore not the number of properties but two things: how far inward from the output end the correction reaches, and how hard the stream is.

## 3. What the goal needs, and what each need depends on

| | Capability | Depends on | Serves |
|---|---|---|---|
| a | Cells away from the input lines stay informative: not silent, not constant | the input lines are driven by the bytes | everything |
| b | The output end corrects itself from the signal: its cells come to give the coming byte from what the cells they listen to carry | a | end to end |
| c | The correction is relayed inward: cells one, two, four connections from the output end receive what is relayed and correct themselves | b; return connections | everything past b |
| d | The input end is reached by the relay and corrects what it takes in | c | end to end; generalisation |
| e | A byte's information is held in cells' states across later bytes | a; improved by c | delay |
| f | A taught answer is given at the first attempt once taught | b, c, e | E1 |
| g | A taught answer is kept under later correction | f | persistence; requirement A |
| h | Cells that respond to two parts of an input together, with the code of each part shared between instances | c, d, e | the means of generalising |
| i | The answer depends on the task byte | h | first family |
| j | A never-taught combination of taught parts is answered rightly at the first attempt | f, h | generalisation (E3) |
| k | The same at numbers of bytes between the parts that were never reinforced | e, j | generalisation (E2) |
| l | Giving nothing where nothing can be expected, and the end mark where an answer ends | b | end to end |
| m | g, j and k together over a life of several families that return | g, j, k | the goal |
| n | Cost beside the comparator (B1, B2) | read from f onward | the condition |

## 4. Overlaps and conflicts, and what they mean for the order

1. **One loop gives all three properties** (section 2). None of them is a late stage; from step 2 on, every step is end to end, on an unbroken stream, and scored on first attempts.
2. **Correction spreads from the output end inward.** The cells at the output end are nearest to the signal and can correct themselves first; the input end is farthest and is reached last. The order of steps 2 and 3 follows from this.
3. **The steps are phases of one life.** A tissue at step k has lived through the streams of the earlier steps, and their material returns. Every later step is thereby also a test of keeping.
4. **Keeping and generalising use the same connections.** Shared codes of parts are what lets a never-taught instance be answered, and they let a new family reuse what exists instead of overwriting it; but a later correction to a shared cell changes every instance at once. The two are read together from step 5, and no form of the cell is accepted on one alone.
5. **Generalisation and B2 rest on one measurement:** the share of never-taught instances right at the first attempt, against the number of examples taught. B2 is the comparison of that curve with the comparator's on the same stream in the same units.
6. **The input end and generalisation overlap.** Whether the parts of an input reach later cells separately is decided by what the input end comes to carry. A failure at step 5 is first looked for there.
7. **Holding a byte, binding two parts, and passing over the bytes between them are three capabilities**, introduced at steps 3, 5 and 6b.
8. **The number of a cell's connections is the cell's own from step 1.** With fifty return connections a cell, two relays reach 2,500 cells; with two, they reach four. The reach of the correction, and with it step 3, depends on this number.
9. **Conflict: giving up unused connections.** It is how useful paths are found, and it is also how what an absent family needs is lost. The rule of steps 1 to 3 is provisional until step 4.
10. **Conflict: two sources of adjustment in one cell.** A cell predicts what arrives (statement 1) and is corrected by what is relayed (statement 5). They may agree or pull apart; how one cell combines them is a choice of step 0, opened at step 3.
11. **Conflict: a cell's regulation of its own activity** changes strengths that keeping needs.
12. **Every attempt is followed by correction.** A never-taught instance can be scored once in a life.

## 5. Conditions that hold at every step

1. The tissue is the one defined at step 0. No cell is handed the answer; no correction is computed for a cell from outside it; no backward pass through the forward connections; nothing is reset; nothing is averaged over items.
2. One stream. Items follow one another and are separated only by bytes. The tissue gives out at every position, before the byte arrives, and is told when it was wrong. It is scored on the bytes it gives where an answer is due.
3. Controls are other tissues from the same start: one whose stream lacks the teaching under test, and one in which the relay under test is absent. A tissue is never copied or reset to make a control.
4. Never-taught instances are set aside before the stream is made. Each is asked once in a life. The reserve is divided beforehand: asked early, asked after a delay, asked after other learning; and by which of its parts were taught in other instances.
5. Each new stream has a positive control known to solve it: the comparator (a network trained by back-propagation) on the same stream, counted in the same units, which are items read and arithmetic operations. Its release is the coordinator's.
6. The brief of each step carries the contract's design block: claim and level; record of what was shown with feedback; what a table of taught instances and the frequencies of parts would score (a table that treats a pair and its reverse as one key where the relation allows); controls; the rule of success, the budget and the rule for stopping. Steps 5 to 7 are designed by the project's method for E3 designs.
7. Smallest size first. From the first form, one number is varied at a time on the grid 2, 4, ..., 128; numbers of cells go beyond 128 by further powers of two when a step requires it.
8. A step not reached within its budget is reported as not reached, with what was found. It is not passed over. Mr. Xu decides whether the route continues, changes or stops there.
9. The numbers fixed by us are listed at every step and handed to the cell as soon as a cell can set them.

## 6. The steps

**Step 0. The tissue defined, with its signal.** The description (`THE_TISSUE_WRITTEN_OUT_20261118.md`) is revised from its check (`CHECK_OF_THE_TISSUE_WRITTEN_OUT_20261118.md`: 8 sentences fail, 21 are doubtful) and from statement 5: the two ends; the signal, where it enters and how it is relayed; return connections; one first form, each choice in the simplest form a living cell can do that the conditions above allow; and the order in which choices are opened when a step fails.
*Reached when* a third reading finds no sentence that a living cell cannot do, and every sentence is marked as Mr. Xu's words, as following from them, or as our choice.

**Step 1. The tissue reads the stream and stays informative (E0).** New: capability a. Stream: the items of the first family one after another. No correction yet.
*Reached when*, late in a long stream, cells one, two and four connections from the input lines are neither silent nor constant, and the present byte and the bytes one, two, four and eight positions back can be told from their recorded outputs better than at the start of life.
*What it buys:* the material every later correction works on. *On failure:* the order of opening in section 4 of the check.

**Step 2. The output end corrects itself (E0 to E1).** New: capabilities b and l. The signal acts on the cells at the output end only; the rest of the tissue is as in step 1. The feedback is yes or no (decision D3).
*Reached when* the byte the tissue gives is right, at the first attempt, wherever the coming byte can be known from the last few bytes, above what the frequencies of bytes give; in a control tissue that is never told, it is not.
*What it buys:* the tissue is end to end from here on, and its level is the floor against which step 3 is measured.

**Step 3. The correction is relayed inward (E1).** New: capabilities c, d, e and f. Cells one, two and four connections from the output end receive what is relayed and correct themselves; at the largest reach the input end is included. Stream: places that the output end alone cannot get right: a byte shown, then one, two, four or eight other bytes, then due again; and a small fixed table of taught answers.
*Reached when* these are answered rightly at the first attempt, and are still answered after a long stretch of other bytes, while the control tissue without relay stays at the level of step 2. Read: how the result depends on the number of relays allowed and on the number of return connections of a cell (2 to 128). The comparison with the comparator (section 5, condition 5) begins here.
*What it buys:* this is the relay Mr. Xu described, shown or not shown in the tissue; every later step needs it. *On failure:* the form by which a cell combines its own prediction with what is relayed (conflict 10); cells with two parts.

**Step 4. Kept while something else is learned (E1 retention; requirement A).** New: capability g. A second table follows the first, and the first returns. No store and no act of a program; the tissue's own activity in quiet stretches of the stream is allowed.
*Reached when* the first table is still answered and the second was really learned, by floors fixed in the brief (the constraints file's floors are a draft that awaits the owner's decision).
*What it buys:* the forms by which a change becomes lasting are first chosen here, and the rule for giving up connections is revised (conflict 9). They are read again at steps 5 and 7.

**Step 5. The first generalisation (E3, smallest size).** New: capabilities h, i and j. A part of the first family in which a never-taught instance is a new combination of parts taught in other instances; the candidates are its two operations whose answer is one of the two numbers (the larger, the smaller), each with its task byte. Reserve, controls and countermodels as in section 5. Never-taught instances are asked early, after a delay, and after other learning; step 4 is read again in the same life.
*Reached when* never-taught instances are answered above the countermodels and above the control tissue at all three times, with keeping intact. This is the contract's test at its smallest size.
*On failure:* first the input end (overlap 6), then the reach of the relay.

**Step 6a. The rest of the first family (E3).** Addition and subtraction modulo 89, where the places of a number interact. *Reached when* never-taught pairs, including those that need the modulus, are answered at the level the comparator reaches on the same stream.

**Step 6b. Presentations never reinforced (E2).** New: capability k. A varying number of letters between the parts, including numbers never taught. *Reached when* the result of step 5 or 6a holds at those numbers. Steps 6a and 6b do not depend on each other.

**Step 7. A life.** New: capability m. The long-life world on record: several families one after another, returning. *Reached when*, in one life, earlier families are kept without a store, and never-taught instances of an earlier family are answered after later families have been learned, above the control tissue. This is the goal by the contract.

**Step 8. Cost, size and breadth (B1, B2, E4).** The tissue's curves beside the comparator's along a ladder of sizes and stream lengths; the numbers still fixed by us handed to the cells one at a time, with step 7 read again; then families and streams of other kinds.

## 7. Where the route is most likely to stop

1. **Step 3.** Rules in which an error is relayed over separate return connections and each cell corrects itself locally exist in the literature and learn small tasks in simulation; they are generally weaker than back-propagation on large ones (the reviewer's confidence in this general statement is moderate). On record, the sweep, which relays the output's error backward through the forward connections using their signs, reaches 80 in 100 on never-taught pairs of the first family; that shows a crude relayed correction can suffice for this family, but under conditions the route does not allow (reverse order in time, the sign of the forward weight, batches, resets, a dense readout).
2. **Steps 4 and 7.** Keeping without a store has failed in every line of the record.
3. **Step 5.** Generalisation under a relayed local correction, without the conditions listed under 1, has not been shown on record.

## 8. The record on this route

An agent has placed eighteen lines of work on the steps (`THE_EARLIER_RECORD_PLACED_ON_THE_ROADMAP_20261118.md`; not re-checked by the reviewer). It was briefed with the reviewer's stricter wording, under which an error told at the output end counted as a violation. Its rows in which output units were taught by an error at the output end are therefore to be read again under statement 5 before anything is concluded from them. What stands independently of that: the network of the review period learns by a backward pass with a dense readout, resets and batches, and is the comparator; without a store it keeps almost nothing of an earlier family; and the record holds reusable streams, scorers and second implementations of the two worlds. E53 and E62 finish, being paid for, and are filed with the comparator. The question of E61 (how many relays the correction needs) is step 3's question; its brief as written relays in reverse order in time and is not used.

## 9. Decisions made in this roadmap that are not Mr. Xu's words

They are listed so that he can strike any of them.

- D1. Input lines and output lines are fixed, one for each byte value, and say nothing about digits; the cells on them learn.
- D2. (Changed at 23:03 UTC by Mr. Xu's ruling, section 13 of `WHAT_A_NEURON_HAS_20261118.md`.) Cells can receive the feedback of the error. How far it must be passed on from cell to cell before a cell that needs it has it is read at step 3; no restriction is placed on a cell receiving it.
- D3. (Changed at 23:03 UTC by the same ruling.) For the present task, the prediction of the next byte, the feedback is yes or no. The definition of the tissue keeps a place for richer feedback, which more complex tasks may bring later; no form is excluded in principle, and where a step has an arithmetic answer (steps 5 to 7), the feedback may also carry the right answer and the absolute difference between the answer given and the right one, as Mr. Xu's example of 23:04 UTC states (section 14 of the same file). Which of these a step uses is fixed in its brief, and the result is reported with the form of feedback named.
- (By Mr. Xu's rulings of 23:18 UTC, section 15 of the same file; these are his words and not decisions of the reviewer.) Each cell has a coordinate drawn at random at the start, and a cell's neighbourhood is the cells near it by that coordinate. The input end receives the feedback of the output end's information, which in the present task is the result of the prediction of the next byte.
- D4. Controls are second tissues. Recording from cells is the measure at step 1 only, where there is no behaviour yet.
- D5. The steps are phases of one life.
- D6. Step 5 begins with the two operations whose answer is one of the two numbers.
- D7. Each new stream has the comparator as positive control.

## 10. What the reviewer arranges next

1. The revision of the tissue's definition from its check and from statement 5, by an agent, and a third reading of it.
2. A second reading of the earlier record's rows under statement 5.
3. When step 0 is reached: the briefs of steps 1 and 2, written by an agent, checked by another, and put to the coordinator.

## History

Change of 23:18 UTC: Mr. Xu's rulings on random coordinates and on the feedback of the output end's information to the input end entered in section 9; the dependency "the input end is reached last" (overlap 2, capability d) is to be reconsidered in draft 3, since the input end now receives that feedback directly.

Change of 23:04 UTC: decision D3 takes in Mr. Xu's example of richer feedback for addition (the right answer and the absolute difference).

Change of 23:03 UTC to draft 2: decisions D2 and D3 and step 2, after Mr. Xu's ruling that cells can receive the feedback of the error, that yes or no is the form for the present task, and that richer forms are not excluded later.

Draft 1 (18 November 2026, about 22:30 UTC). Read by three agents: `INDEPENDENT_READING_OF_THE_ROADMAP_20261118.md` (fourteen findings), `CHECK_OF_THE_TISSUE_WRITTEN_OUT_20261118.md`, `THE_EARLIER_RECORD_PLACED_ON_THE_ROADMAP_20261118.md`. Draft 2 (22:49 UTC), after Mr. Xu's statement of 22:46 UTC on the continuous signal: the signal and its relay are made the centre; the output end is corrected first and the input end last; controls are second tissues; the reserve of never-taught instances is divided beforehand; the steps are phases of one life; a rule for stopping; the places where the route may stop; and the decisions that are not Mr. Xu's words are listed.
