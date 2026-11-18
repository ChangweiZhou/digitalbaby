# Roadmap to end-to-end persistent learning with generalisation (18 November 2026, draft 1, for Mr. Xu)

By the outside reviewer. This file replaces `ORDER_FROM_BIOLOGY_20261118.md` (draft 2), which was a list of measurements and not a route to the goal. It orders what the goal needs by what depends on what. It starts nothing; releases and runs are the coordinator's.

## 1. The goal, which has not changed since the project began

The project's objective contract (`minifly_b/package/DOCS/MINIFLY_OBJECTIVE_CONTRACT.md`, effective 26 September 2026) states it:

> Build a bounded, biologically informed learner that uses information acquired from earlier byte-level experience when it becomes useful later, despite delay and intervening learning. The eventual test is reuse of learned information on a relevant, previously unreinforced input or task, with the answer selected before feedback.

Mr. Xu's words of 18 November state the same goal: end-to-end persistent learning with generalisation, by cells that each only receive input from other cells, send an output from that input and their own state, and keep adjusting which cells they are connected to (`WHAT_A_NEURON_HAS_20261118.md`, section 8). His ruling of 22:25 UTC adds that there is an input end and an output end and that both learn how to take in and how to give out by the same learning (the same file, section 10).

The goal has three properties, and one condition on how they are reached.

| Property | What it means here |
|---|---|
| End to end | Bytes go in and bytes come out. The answer and its end mark are produced by the tissue itself. Everything between, and both ends, is learned by the cells' own local adjustment. |
| Persistent | One life on one unbroken stream. Learning is never switched off and nothing is reset. What was learned is still used after a delay and after other things have been learned, and this long-term memory forms in the course of learning (requirement A; the tissue's own repetition is accepted). |
| Generalising | An input that was never taught is answered rightly at the first attempt, before the right answer is shown, and this is so because of what was taught earlier. |
| Condition | Only what a living cell can do: local learning everywhere, a chemical signal that is local, no backward pass, no quantity taken over the whole tissue. As few numbers fixed by us as possible (requirements B1, B2 and C are read along the way; see section 6). |

The contract's levels of evidence are used below as the project defined them: E0 (access: the interface works), E1 (taught association: acquisition, retention, interference), E2 (the same relation under a presentation never reinforced), E3 (a relation taught by examples, answered on instances never reinforced, with a control without the prior teaching), E4 (breadth across families or streams).

## 2. What the goal needs, and what each need depends on

| | Capability | Depends on | Property it serves |
|---|---|---|---|
| a | The cells stay informative while each predicts what arrives: they do not fall silent or constant | nothing | all three |
| b | The input end is learned: the cells on the input lines come to carry what distinguishes the bytes and their order | a | end to end; generalisation (see overlap 5) |
| c | The output end is learned: cells drive the output lines so that the tissue itself emits bytes | a, b | end to end |
| d | Information reaches cells that are not on the input lines, over several exchanges, and is held in the cells' state across later bytes | a, b | end to end; the delay part of persistence |
| e | A taught answer is given by the tissue, in bytes, before the right answer is shown | b, c, d | end to end with learned content (E1) |
| f | A taught answer is kept while other things are learned | e | persistence (E1 retention; requirement A) |
| g | An answer that depends on two separate bytes together | d, e | needed before generalisation can be asked at all |
| h | A never-taught instance is answered rightly at the first attempt | g, b | generalisation (E3) |
| i | The same under presentations never reinforced (a varying number of bytes between the parts) | d, h | generalisation (E2 within E3) |
| j | f, h and i together over a life of several families that return | f, h, i | the goal |
| k | Cost beside the reference: slower at first and ahead later (B1); the same result from far fewer examples (B2) | read from e onward | the condition |

## 3. Overlaps, and what they mean for the order

1. **End to end is not a late stage.** It is reached once, at the smallest behaviour (step 2), and is never given up afterwards: from step 2 on, everything is read from the bytes the tissue itself emits.
2. **Persistence has two parts.** Operating on one unbroken stream with no reset is a condition of every step from the first. Keeping what was learned is a result; it can first be read when there is a second thing to learn (step 4), and it is read again at every later step.
3. **Keeping and generalising act on the same connections in opposite directions.** Generalising needs the instances of one relation to share connections; keeping needs a later family not to overwrite them. A form of the cell's adjustment chosen for one alone can destroy the other. They are therefore read on the same tissue in the same life from step 5 on, and no form is accepted on one of them alone.
4. **Generalisation and requirement B2 are one measurement.** The share of never-taught instances answered rightly, as a function of how many examples were taught, is both.
5. **The input end and generalisation overlap.** Whether the parts of an input (for instance the tens digit and the units digit) are separately available to later cells is decided by what the cells at the input end come to carry. A failure at step 5 may therefore be a failure of capability b and is first looked for there.
6. **The output end cannot be learned before the input end.** The tissue's attempt is followed by the right answer, which arrives as input. What teaches the output end therefore passes through the input end and through the relay of capability d.
7. **Relay, delay and changed presentation are one capability at three sizes.** Holding a byte in the cells' state across later bytes (step 2), bringing two bytes together (step 5), and doing so across a varying number of bytes between them (step 6).
8. **The number of a cell's connections belongs to the cell from the first step.** It is part of what a cell does by Mr. Xu's statement, and it bears on keeping (fewer shared connections) and on relay (reach). It is not a later refinement.

## 4. Conditions that hold at every step

1. The tissue is the one defined at step 0, and nothing else acts on it. No error is sent back, no cell is given an answer, nothing is averaged over items, and no state is reset.
2. One stream, one byte after another. The tissue emits at every position; it is scored where an answer is due, on its first attempt, before the right answer arrives. Nothing in the stream marks where an item begins or ends other than the bytes themselves.
3. A step begins at the smallest size that can show it. Steps 1 to 5 need few cells and short streams, so that many forms can be tried at small cost; cost grows only from step 6.
4. One tissue throughout. When the form of the cell is changed at a later step, the earlier steps are read again with the changed form before the later result is accepted.
5. A step that is not reached is not passed over. Work returns to the choices of step 0 that the failure points to.
6. At every step the numbers fixed by us are listed, each on the grid 2, 4, ..., 128, and handed to the cell wherever a cell can set them.

## 5. The steps

**Step 0. The tissue defined.** A draft exists (`THE_TISSUE_WRITTEN_OUT_20261118.md`: 100 sentences, 31 open choices, 4 gaps). To be done: (i) the two ends are added by Mr. Xu's ruling: input lines that each respond to something particular in the bytes, output lines that each produce something particular in the bytes, and cells on both that are cells like the others and learn; (ii) the draft's gaps are answered from what living cells do: a cell regulates its own level of activity, so cells do not fall silent (gap 1); a cell passes on what is new to it, which is Mr. Xu's own wording ("new information"), so one cell's prediction matters to the next (gap 2); the tissue emits bytes and the right answer follows as input, so there is an attempt (gap 3); connections are tried at random within the neighbourhood and kept or given up by use, and whether that suffices is answered at step 2 (gap 4); (iii) one first form is fixed, the simplest of each choice, with the order in which choices are opened when a step fails; (iv) a second reader checks every sentence with two questions: is it in Mr. Xu's words, and can a living cell do it.
*Reached when* every sentence passes both questions and one first form is fixed.

**Step 1. The tissue reads the stream and stays informative (E0).** New: the cells' prediction and adjustment running on bytes; the input end forming (capabilities a, b).
*Reached when*, over a long stream, the cells do not fall silent or constant, each cell's own mismatch falls, and the present byte and the bytes just before it can be told from the recorded outputs of cells at increasing numbers of connections from the input lines.
*Serves* all three properties: nothing can be emitted, kept or generalised from cells that carry nothing. *On failure:* the choices on a cell's regulation of its own activity and on what its output carries.

**Step 2. The loop is closed on the simplest behaviour.** New: the output end learns (capability c); information is relayed and held (capability d). The stream shows a byte and later shows it again; the tissue is to emit it before it is shown the second time. The number of bytes in between rises (0, 1, 2, 4, 8), and so does the number of connections between the input lines and the output lines.
*Reached when* the emitted byte is right far above chance, lastingly, at the first attempt.
*Serves* end to end, which is kept from here on, and the delay part of persistence. *On failure:* the choices on the output end, on the number of exchanges between two bytes, and on how connections are made and given up.

**Step 3. A taught association, end to end (E1).** New: the answer is not a copy; it is learned from being shown (capability e). A small fixed table: one byte, the sign `=`, the tissue's attempt, then the right byte.
*Reached when* the taught answers are given far above chance and are still given after a long stretch of other bytes. From this step on the columns of section 6 are filled.
*On failure:* the form by which a strength changes and the length of a cell's traces.

**Step 4. A taught association kept while another is learned (E1 retention; requirement A).** New: a second table is taught after the first, and the first returns (capability f). No store and no act of a program; the tissue's own activity between visits is allowed.
*Reached when* the first table is still answered after the second has been learned, and the second was really learned; both, by the floors the project's constraints file gives for loss and for acquisition.
*Serves* persistence. The forms by which a change becomes lasting, and the use of the local chemical signal, are settled here. *On failure:* those choices; the route does not go on.

**Step 5. The first generalisation, at the smallest size (E3).** New: the answer depends on two bytes together (capability g), and some pairs are never taught (capability h); one-digit addition is the smallest case of the first family. Never-taught pairs are asked within the stream and scored on the first attempt. Controls as the contract requires: the same tissue and stream without the earlier teaching of the relation; and what a table of the taught pairs, and the frequencies of the parts, would score. Step 4 is read again in the same life (overlap 3).
*Reached when* never-taught pairs are answered above those controls and the keeping of step 4 still holds in the same tissue. This is the goal at its smallest size.
*On failure:* first the input end (overlap 5), then the choices on what predicts what within a cell.

**Step 6. The first family complete, with changed presentation (E2 within E3).** New: two-digit numbers, two answer bytes and the tissue's own end mark, and a varying number of letters between the parts, including numbers of letters never taught (capability i).
*Reached when* the never-taught pairs of the record's first family are answered at the level the record uses (80 in 100), at every number of letters.

**Step 7. A life.** New: several families one after another, returning, with never-taught instances inside the stream: the long-life world that the record already has, with its verified reference lives (capability j).
*Reached when*, in one life, the earlier families are kept without a store (which the reference does not do) and never-taught instances are answered. This is the goal on the project's own test.

**Step 8. Cost, size and breadth (B1, B2, E4).** Along a ladder of sizes and stream lengths, the tissue's curves are set beside the reference's: where the tissue overtakes (B1), how many examples it needs for the same share right on never-taught instances (B2). The numbers still fixed by us are handed to the cells one at a time, with step 7 read again each time. Then families and streams of other kinds.

## 6. What is read at every step from step 3 on

Examples needed to reach the step; computation spent; the share right on what was taught earlier, after delay and after later learning; the numbers fixed by us. The reference's figures for the same stream, where the record holds them, stand beside these. No new run of the reference is asked for.

## 7. Where the record stands on this route

The network of the review period learns by a backward pass (the exact gradient, or the sweep), has a dense readout, is reset at every item and averages over 64 items. It is the comparator, which is how the project's draft constraints already name exact back-propagation, and it is not a stage of this route. What it contributes is the stream worlds (the first family; the long-life world), the scoring, the reference curves, and facts about the comparator (for instance, that it does not keep earlier families without a store). The experiments that changed one part of that network at a time (limited connections, masks at the interface, the fixed input code, the sweep) did not reach any step above, because in all of them learning used a backward pass. E53 and E62 finish, being paid for, and are filed with the comparator.

The mechanisms of the period before the review (to September 2026) learned without a backward pass, and the contract records that they reached E1 and not E3. The reviewer has not re-read them and states nothing about them here; an agent is arranged to place them on the steps above, so that what was already reached is not done again.

## 8. What the reviewer arranges now

1. The second reader for step 0, with Mr. Xu's ruling on the two ends.
2. An independent reading of this roadmap against the contract and Mr. Xu's statement: is each step something a living tissue can do, is each dependency right, is anything missing.
3. The placing of the earlier record on the steps.

When step 0 is reached, the brief for step 1 is written and put to the coordinator.

## History

Draft 1 (18 November 2026, about 22:35 UTC), after Mr. Xu's message of about 22:25 UTC: what was asked for is a step-by-step route to the goal with its dependencies and overlaps; the two ends exist and both learn.
