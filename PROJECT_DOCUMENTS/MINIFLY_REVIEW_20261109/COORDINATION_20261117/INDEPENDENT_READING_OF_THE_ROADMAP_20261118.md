# An independent reading of the roadmap to end-to-end persistent learning (18 November 2026, written from 22:36 UTC)

Reader: an agent that took no part in writing the roadmap. Document read: `ROADMAP_TO_END_TO_END_PERSISTENT_LEARNING_20261118.md`, draft 1 (called "the roadmap" below), against the project's objective contract (`minifly_b/package/DOCS/MINIFLY_OBJECTIVE_CONTRACT.md`, called "the contract"), sections 7 to 10 of `WHAT_A_NEURON_HAS_20261118.md` (Mr. Xu's statement and rulings), sections 3 and 4 of `THE_TISSUE_WRITTEN_OUT_20261118.md` (called "the tissue document"), the draft constraints file `PROJECT_DOCUMENTS/KB/CONSTRAINTS.json`, and, for facts about the earlier record, `INVENTORY_STRUCTURE_AND_WORLD_20261117.md` and `LOG.md`. Nothing was run and no other file was changed.

## Summary

1. Step 7 does not amount to the project's goal under the contract. No step scores a never-taught instance of an earlier family after a delay and after other learning, which is the contract's central test, and the causal control is worded in a way a single unreset tissue cannot carry out. The contract's design block (exposure audit, a table that treats a+b and b+a as one key, a positive control, equal budgets, stopping rules, costs) is mostly absent (finding 1).
2. The route has no stated way for the cells at the output end to learn from the right answer, which arrives later and as input. The obvious ways to close this gap (a fixed wiring from each input line to the matching output line; a program that picks the strongest output line) would give a cell the answer or take a quantity over many cells. This decision must be made at step 0 and put to Mr. Xu (finding 2).
3. Step 0 presents choices as answers. The answer to gap 2 is attributed to Mr. Xu's words although it is a design choice. That choice makes the whole tissue a predictive-coding network (one in which cells pass on their prediction error), and in such a network an error can travel back through loops. Whether that counts as a backward pass is not asked. Gap 1 is not closed. Taking "the simplest form of each choice" contradicts the roadmap's own requirements (finding 3).
4. The capability table lacks binding of two bytes into one response, dependence on the task code, a code of digits in which the relation is shared across pairs (the actual means of generalisation), the tissue's delay against the stream's timing, a "no byte" emission and end mark, and a period without input for rehearsal. Several stated dependencies are wrong (finding 4).
5. The steps are still mainly checks. The roadmap names no mechanism by which a local tissue could generalise addition, says what four of eight steps contribute, and leaves open whether the steps are phases of one life or separate tissues. Running them as phases of one life would make the route itself a test of persistence (finding 5).
6. Further findings concern overlaps and conflicts (pruning by disuse against keeping; every probe teaches); measurement in an unbroken stream; streams too loosely defined to brief; an oversized step 6; the comparison with back-propagation; stopping rules; and wording.

## Terms used in this reading

A *tissue* is the set of model nerve cells described in step 0 of the roadmap. A *probe* is an item in the stream at which the tissue's first emitted answer is scored before the right answer appears. A *never-taught instance* is an item whose answer has never been shown in the tissue's stream. A *family* is a set of items sharing one relation (for example addition modulo 89). A *visit* is a stretch of the stream in which one family's items occur. The *reference* is the network of the review period trained by back-propagation, which the roadmap treats as the comparator. *Back-propagation* is learning by an error passed backward through the network's connections. A *store* is a buffer of past items kept by a program outside the learner and replayed to it. The *design block* is the five-part "required design and report block" of the contract. Other terms are defined where they first appear.

## Findings, in order of seriousness

### Finding 1. Neither step 5 nor step 7 meets the contract's test for the goal, and the controls cannot be carried out as worded

*Place.* Step 5 ("Controls as the contract requires ... This is the goal at its smallest size"); step 7 ("Reached when, in one life, the earlier families are kept without a store ... and never-taught instances are answered. This is the goal on the project's own test"); section 6; section 1, row "Generalising".

*Reason.* The contract (section "Evidence levels", paragraph after the table) allows a claim about reusable information only with "an E3 result with delayed and intervening-learning probes, plus a causal state control". E3 is the level at which a relation taught by examples is answered on never-reinforced instances. Its objective is information "acquired from earlier byte-level experience" used "when it becomes useful later, despite delay and intervening learning". Step 7 asks for two separate facts in the same life: earlier families are kept (an E1 fact, E1 being the level of taught associations), and some never-taught instances are answered. It never asks for the conjunction. That conjunction is a never-taught instance of an *earlier* family, asked for the first time *after* another family has been learned, answered rightly because of the earlier teaching. Step 5 likewise reads step 4's keeping (E1) again, but its E3 probes are not delayed and have no intervening learning. Calling step 5 "the goal at its smallest size" therefore overstates it.

The causal control is worded as "the same tissue and stream without the earlier teaching of the relation" (step 5). By the roadmap's own section 4, condition 1, and by sentences 75 and 76 of the tissue document, a tissue is never reset or copied, so the same tissue cannot also live without the teaching. The control must be a second tissue started in the same way (the same starting draw, choices C7 and C25 of the tissue document). It must receive a stream with "matched sensory input and timing" (contract, design block item 4) in which the relation is absent. For example, the same items with the answer bytes drawn from a shuffled mapping, so that byte frequencies are equal. The roadmap does not say what replaces the teaching bytes, which is the substance of the control.

The design block items absent from the roadmap are as follows.
- Item 1, claim and level: no step distinguishes a diagnostic from a promotion test, and step 6's "E2 within E3" is not a level of the contract (see finding 8).
- Item 2, exposure audit: in an unbroken stream with no reset, every probe whose answer then appears becomes taught (tissue document, section 4, point 4). The audit must therefore be a per-instance record of the stream position at which each item was first answered by the stream. The roadmap does not mention this.
- Item 3, recitation countermodel: step 5 names "a table of the taught pairs, and the frequencies of the parts". It omits the canonicalised-key table that the contract names. For addition, a table that stores the unordered pair answers b+a from a taught a+b, so never-taught pairs whose reversed pair was taught must be scored in a separate group, and only the others can support E3.
- Item 4, controls: there is no positive control known to solve the task on the same stream. The draft constraints file's `required_arms_by_level` also asks for a shuffled arm at E1, E2 and E3 and an `oracle_plumbing` arm at E3, neither of which the roadmap mentions. There is no statement of equal information and feedback budgets between the tissue and the reference (see finding 10).
- Item 5, promotion rule and cost: there is no prespecified threshold beyond "far above chance" and "above those controls", no failure or stopping criterion, no resource accounting, and no cheap check before a costly run.

*Proposed change.* Replace the reached-when of step 7 with the following wording: "In one life, never-taught instances of an earlier family, reserved from all teaching, are asked for the first time after at least one other family has been visited and after a stated delay. They are answered rightly at the first attempt above the margin of the constraints file, and above a second tissue with the same start whose stream carries the same bytes with the relation removed. In the same life the taught instances of the earlier families are kept above the floor for loss, and the later families are acquired above the floor for acquisition." Add the same delayed probe, at small size, to step 5, and rename step 5 "the first test of reuse at small size". Add to section 4 a condition that every step from step 3 on states the five items of the design block before it is run. Define the control as "a second tissue with the same starting draw" in place of "the same tissue".

### Finding 2. The output end has no stated way to learn, and the obvious ways of supplying one give a cell the answer or take a quantity over many cells

*Place.* Section 2, row c ("cells drive the output lines so that the tissue itself emits bytes"); step 0 (i) ("output lines that each produce something particular in the bytes"); step 2; section 3, overlap 6.

*Reason.* Every cell, including the cells on the output lines, has one task: to predict what arrives at it from other cells (Mr. Xu, section 8). An output cell's emission is not one of its arrivals. The right answer arrives later, at the input end. For the output end to learn, the answer's effect must reach the output cells, through ordinary connections from cells driven by the input end, as an arrival they can relate to a trace of what they emitted earlier. The roadmap states this in overlap 6, but the table gives c no dependency on d (relay) and names no means. Three decisions are hidden here.
1. *How the activity of many output lines becomes one byte.* If a program chooses the most active output line, that is a quantity taken over a population of cells by the program. A competition among the output cells through their own inhibitory connections would be an operation of the tissue, but it must be stated.
2. *Whether each input line is wired by the experimenters to the output line of the same byte.* With such a fixed pairing, the output cell for byte v receives the right answer directly when v arrives. That is a label given to a cell, and it contradicts the statement that a single nerve cell cannot see the answer. Step 2, which teaches the copying of a byte, makes this wiring especially tempting, and it would pass step 2 trivially.
3. *Whether the tissue hears its own emissions*, that is, whether emitted bytes enter the input stream.

There is also a consequence the roadmap should state. If output cells learn by predicting the arrivals that the answer causes, the emitted byte is in effect the tissue's prediction of the coming byte, read at answer positions. That is close to the reading "expecting the next byte" that Mr. Xu rejected as a description of what a cell does (section 8). It may be acceptable for the output end alone, but it should be put to him and not discovered later.

*Proposed change.* Add a capability between b and c: "c0. The answer's effect, arriving through the input end and the relay, reaches the output cells and can be related within each output cell to a trace of its own earlier emission." c depends on c0, a, b and d. Add to step 0 three explicit choices to put to Mr. Xu: the conversion of output-line activity into one byte (by the cells' own competition, not by a program), the starting connections between input-end cells and output cells (random within neighbourhoods, never a byte-to-byte pairing), and whether emissions are heard. Add to step 2's reached-when: "with starting connections between input-end and output cells drawn at random, and with the conversion to a byte done by the output cells themselves".

### Finding 3. Step 0 presents design choices as answers to the gaps; one of them decides the backward-pass question without asking; and "the simplest form of each choice" contradicts the roadmap

*Place.* Step 0 (ii) and (iii).

*Reason.*
- *Gap 2.* The answer "a cell passes on what is new to it, which is Mr. Xu's own wording ('new information')" attributes a design choice to Mr. Xu. In his statement, "new information" (新来的信息) is what newly arrives at a cell. It does not say that a cell's output carries its mismatch, the difference between an arrival and its prediction. That is choice C10 of the tissue document, second or third form. The choice makes the tissue a predictive-coding network, a network in which cells transmit prediction errors. Loops of connections are not excluded (tissue document, sentence 30). In such a network with loops, a mismatch passes through ordinary outputs to the cells upstream. It is known with fair confidence that predictive-coding networks of this kind can approximate the gradients of back-propagation (Whittington and Bogacz, 2017; Millidge and colleagues, 2020–2022, with conditions on settling time). Whether this falls under Mr. Xu's ban on a backward pass is therefore a ruling only he can give, and the roadmap passes over it.
- *Gap 1.* "A cell regulates its own level of activity, so cells do not fall silent" prevents silence only. The tissue document (section 3, gap 1) says that such regulation (C29) "changes how fast this happens but does not by itself supply" a requirement that outputs carry information about the bytes. A cell can hold its average activity at a target and still be constant across bytes, or follow a rhythm of its own. Step 1's reached-when tests for this, but step 0 claims the gap is answered.
- *Gap 3.* Only its second half (that there is an attempt) is answered. Its first half is left open: why the tissue should come to predict the answer digits rather than the easier regularities.
- *Gap 4* is deferred to step 2. Finding 11 explains why this is a likely place to stop.
- *Step 0 (iii).* "One first form is fixed, the simplest of each choice" conflicts with the roadmap itself. The simplest form of C22 (encoder cells fixed and not adjusting) contradicts capability b ("the input end is learned") and requirement C (learning is local also at the input). The simplest form of C10 (activity alone) contradicts the answer to gap 2. The simplest form of C5 (connections present or absent, no strengths) leaves choice C15 (how a strength changes) empty. The simplest form of C24 (one step between two bytes) limits how far information travels per byte and so interacts with step 2's rising path length.
- *Reached-when of step 0.* "Every sentence passes both questions", one of which is "is it in Mr. Xu's words". This cannot be met by the [C] sentences (the open choices), which by construction are not in his words.

*Proposed change.* In step 0 (ii), write the gap answers as numbered choices with the alternatives the tissue document lists, marked "put to Mr. Xu". Add the question: "When a cell's mismatch reaches upstream cells through a loop of ordinary connections, is that a backward pass?" Rewrite (iii) as "one first form is fixed. For each choice it is the simplest form compatible with capabilities a to j and requirements A and C, and each departure from the simplest form is stated with its reason". Rewrite the reached-when as "every [S] and [F] sentence is checked against Mr. Xu's words, every [C] sentence is checked for whether a living cell can do it, and the open questions are answered by Mr. Xu".

### Finding 4. Capabilities the goal needs are missing from the table of section 2, and several stated dependencies are wrong

*Place.* Section 2.

*Reason and missing capabilities.*
1. *Binding.* An answer that depends on two bytes together (row g) needs cells whose response depends on the combination and not on either byte alone. For addition this is not a sum of separate effects of the two digits. The table lists g as depending on d and e only. Holding a byte (d) and learning an association (e) do not supply such combination cells.
2. *A code in which the relation is shared across pairs.* Answering a never-taught pair (row h) requires that the cells that answer taught pairs do so through structure shared with the untaught pair, for example a code of the digits in which their order is present. A cell for each taught pair, which binding alone yields, answers taught pairs and generalises to nothing. Nothing in the bytes '0' to '9' gives their order; it must be learned from the stream. This is the capability on which the goal depends most, and it is not named.
3. *Dependence on a context byte.* In the record's first family and in the long-life world, the same pair of numbers has different answers under different task codes (the inventory, section 1.3: add, subtract, maximum and minimum at many moduli, with a two-byte task code). Step 7 needs answers to depend on the task code as well as on the two numbers, and no earlier step builds or tests this.
4. *Timing.* With s steps of exchange between two bytes (choice C24) and a path of length L connections from the input lines to the output lines, an emission reflects bytes about L/s positions earlier (tissue document, sentence 74). Whether the tissue's emission can be ready at the position where the answer is due is a capability (or a constraint on C24 and on the stream). It is not listed.
5. *Emission of no byte, and the end mark.* The tissue "emits at every position" (section 4, condition 2), but there is no capability of emitting nothing, or a defined filler, where no answer is due, nor of emitting the end mark that the constraints file requires (`own_eos_required`). The end mark appears only in step 6.
6. *Rehearsal without a store.* Requirement A accepts the tissue's own rehearsal, and step 4 allows "the tissue's own activity between visits". In an unbroken stream of bytes there is no "between". Either the world has stretches without input (a decision about the world that must be stated), or rehearsal happens while bytes arrive.

*Wrong or missing dependencies.*
- Capability a (cells stay informative) is said to depend on nothing. Informative means informative about the bytes, so a and b depend on each other and must be shown together, as step 1 in fact does.
- Capability c should depend on d (overlap 6 says so) and on the new c0 of finding 2.
- Capability g should depend on b (overlap 5 says so).
- Capability h depends on the shared code (item 2 above), not on g alone.
- Capability j depends on the context dependence (item 3).
- Capability k is "read from e onward", but the reference has no figures on the streams of steps 3 to 5 (finding 10).

*Proposed change.* Add rows for binding, the shared code, the context byte, timing, the null emission with the end mark, and rehearsal, and correct the four dependencies above. State in the text that the shared code is the capability most likely to decide the route.

### Finding 5. The route still decomposes the goal into checks without saying what each contributes to the goal, and it does not say whether the steps are one life

*Place.* Section 5, the "Serves" lines; section 4, condition 4; the whole of section 5.

*Reason.* The last list was rejected because it did not show how its parts lead to the goal. Steps 3, 5, 6, 7 and 8 have no "Serves" line. Those of steps 1 and 2 ("all three properties"; "end to end") are generic. The roadmap says what each step checks but not what changes in the tissue's form when the step is reached, nor why reaching it makes the next step more likely. Steps 5 and 6 say "the never-taught pairs are answered". Their failure paths ("first the input end, then the choices on what predicts what within a cell") name places to look, not a mechanism to build. No hypothesis is stated of how a tissue of locally learning cells could come to generalise addition. Examples of such hypotheses: an ordered code of digits that forms from predicting the stream; combination cells formed by competition among cells; a code that the relation shares across operands. Without a hypothesis, steps 5 to 7 are measurements of whether generalisation happened, which is the form already rejected.

Second, section 4, condition 4, fixes "one tissue throughout" as one *form*. It does not say whether steps 1 to 7 are phases of a single life on a single stream (copy, then a table, then a second table, then addition) or separate tissues started afresh. If the steps are phases of one life, the route is itself a test of persistence. Each later step tests reuse of what the earlier ones built, and the goal is approached by the same tissue that will be scored on it. If they are separate tissues, the route is a sequence of checks.

*Proposed change.* Give every step a line "Contributes to the goal" that states (i) what the tissue's form or state has, once the step is reached, that it did not have before, and (ii) which later step uses it. Add a section "How a local tissue could generalise" naming one or two hypotheses, each tied to the step that tests it and the failure that would refute it. State that, from step 2 on, each step is a phase of one continuing life unless a step's own design requires a fresh tissue, and that each later phase reads the earlier phases' items again.

### Finding 6. Some overlaps are wrong or one-sided, and several conflicts between the goals are missing

*Place.* Section 3.

*On the stated overlaps.*
- *Overlap 3* treats keeping and generalising as opposed. They are opposed when a later family maps the same inputs differently, as in the long-life world. They cooperate when codes are shared: a code of digits learned for one family makes later families faster to learn and is kept because it is used. The contract's goal is that cooperation (earlier learning reused later). The overlap should say both.
- *Overlap 4* is not right as stated. B2 (requirement that a similar result be reached with far fewer examples than back-propagation) compares two curves, the tissue's and the reference's, counted in the same unit on the same stream. The share right on never-taught instances against examples taught is one of the two curves, not the comparison. Moreover, in one unreset life each never-taught instance can be scored once (finding 7), so a curve along one life needs disjoint reserved sets at each reading point.
- *Overlap 7* treats holding (step 2), bringing two bytes together (step 5) and holding across varying gaps (step 6) as one capability at three sizes. Holding is a matter of the state's duration. Bringing together needs combination cells (finding 4, item 1). Varying gaps need a response that does not depend on the number of letters in between and is not disturbed by them. These are three different capabilities and can fail separately.
- *Overlap 8* asserts that fewer shared connections help keeping. That is a hypothesis and should be marked as one.

*Missing conflicts.*
1. *Pruning by disuse against keeping.* Step 0 (ii) answers gap 4 with "connections ... kept or given up by use". During a long absence of a family, the connections that carry it are not used and do not reduce the cell's mismatch, so by choice C17 they are given up. This is the most direct threat to requirement A in the roadmap, and it is created by step 0's own answer. The third form of choice C31 (longer-lived connections given up less readily) is one remedy and should be named.
2. *Self-regulation of activity against keeping.* A threshold that tracks a cell's recent activity (C29) changes how stored strengths are expressed after a long absence. Thus the answer to gap 1 can erode old answers. I am moderately confident that this tension between homeostatic regulation and stored memories is recognised in the modelling literature.
3. *Measuring against learning.* Every probe followed by its answer teaches, and a wrong answer shown teaches the wrong answer (tissue document, section 4, point 4). The measurement and the learning cannot be separated.
4. *Each cell's own prediction task against the answer task* (gap 3, first half). The task gives no priority to the answer bytes.
5. *Numbers handed to the cells.* Section 4, condition 6, and overlap 8 hand numbers to the cells "from the first step", while step 8 hands them over "one at a time" at the end. These contradict each other.

*Proposed change.* Correct overlaps 3, 4 and 7 as above and mark overlap 8 as a hypothesis. Add the five missing conflicts as overlaps 9 to 13. Settle condition 6 against step 8: either numbers are handed to the cells from step 0 where a cell can set them, or they are handed over at step 8, and the roadmap must say which.

### Finding 7. Measurement in an unbroken, never-reset stream is not worked out, and the number of usable probes is small

*Place.* Section 4, condition 2; steps 5, 6 and 7.

*Reason.* If the right answer follows each probe, a never-taught instance gives one E3 score per life and is taught thereafter. If the right answer does not follow, the stream contains unanswered items, which is a property of the world that must be stated. In step 5 (one-digit addition, 100 ordered pairs) the reserve of never-taught pairs is small. Some reserve must be kept untouched until after a delay and after intervening learning (finding 1). The canonicalised-key group (reverse pair taught) must be scored apart. The draft constraints file asks for at least 8 items per group (`min_items_per_stratum`) and at least 8 worlds outside pilots (`min_worlds_nonpilot`). These figures fix a minimum number of tissues and a minimum size of the reserve. The roadmap does not do this arithmetic.

*Proposed change.* Add to section 4: "Each never-taught instance is scored at its first occurrence only. After that it counts as taught. The reserve of never-taught instances is divided in advance into parts asked early, asked after a delay, and asked after intervening learning, and into groups by whether the reversed pair was taught. The number of instances in each part and the number of tissues are stated before the step is run and meet the constraints file's minimums." For step 5, state the reserve explicitly (for example, of the 100 ordered pairs, which are held out and how many in each group).

### Finding 8. Step 5 is not a case of the first family, and step 6 adds too many things at once and mixes two levels of evidence

*Place.* Steps 5 and 6.

*Reason.*
- The record's first family is arithmetic modulo 89 with a task code (the inventory, sections 1.2 and 1.3). One-digit addition without a modulus and without a code is a different, smaller task, not "the smallest case of the first family". It also has answers from 0 to 18, so one- and two-byte answers occur and an end mark is needed already in step 5, although the end mark is introduced only in step 6. Addition modulo 10 keeps one answer byte and is closer to the family.
- Step 6 adds at once: two-byte operands, the modulus, the task code, two answer bytes, the end mark, intervening letters, and numbers of letters never taught. A failure cannot be attributed to any one of them.
- The reached-when "80 in 100 at every number of letters" is a level the reference reached by back-propagation on the first family with half the pairs taught and long training. The reference's own level at the record's long length test (40 letters) is not stated in the roadmap and may be lower. A target at every number of letters should be set relative to the reference at the same number of letters, or stated as the project's own choice.
- "E2 within E3" is not a level of the contract. The contract defines E2 as a taught relation under a never-reinforced presentation and says that E2 success must not stand in for E3. It also orders a small E2 test before the E3 test ("Current interpretation and next decision"). Scoring only never-taught pairs at never-taught numbers of letters mixes the two.

*Proposed change.* State step 5's stream exactly (for example: digit, '+', digit, '=', the tissue's attempt, then the right digit of the sum modulo 10). Split step 6 into (6a) two-digit operands with a fixed small number of letters; (6b) two answer bytes and the end mark; (6c) the task code with two operations; and (6d) numbers of letters never taught. In 6d, report taught pairs (E2) and never-taught pairs (E3) separately. Move a cheap E2 test (the step 3 table under numbers of letters between cue and '=' never used in teaching) to just after step 3, as the contract orders.

### Finding 9. Several streams are not defined well enough for a brief to be written from them

*Place.* Steps 1 to 4.

*Reason.*
- Step 1 does not say which stream (random bytes, the first family's stream, or something else). This matters because the input end learns the statistics of whatever stream it sees.
- Step 2 ("the stream shows a byte and later shows it again") does not say what marks the position at which the copy is due. Without a cue byte, the tissue cannot know when to emit; with a cue byte, the cue must be named. Step 2 also introduces two capabilities at once (c and d), so its failure cannot be localised. It should start with zero bytes in between, which tests c alone, and only then raise the gap.
- Step 3 does not say what occupies the "tissue's attempt" position in the input stream (nothing, a filler byte, or the tissue's own emission), what the "long stretch of other bytes" consists of, or how many items the "small fixed table" has.
- Step 4 does not say whether the second table uses the same cue bytes with other answers (revision, which the contract lists under E1) or other cue bytes. The two give very different interference.
- Throughout, the thresholds "far above chance", "lastingly", "long stream", "long stretch", "few cells" and "short streams" each hide a number that a brief must state.
- Step 1's "can be told from the recorded outputs" needs a stated method of reading the records and a stated accuracy. A decoder fitted by the experimenters on recordings is a permissible passive reading, but it must never be used as the tissue's answer (contract, item 4: an engineered predictor does not count for the native branch).

*Proposed change.* For each of steps 1 to 4, add a line "Stream" that gives the byte format of one item, the alphabet, what fills the attempt position, the order and frequency of items, and the length, and a line "Numbers" that replaces each vague threshold by a value or by a rule that refers to the constraints file. Split step 2 into 2a (zero gap) and 2b (gaps 1, 2, 4, 8).

### Finding 10. The comparison with back-propagation (B1, B2) is not defined, and "no new run of the reference" conflicts with it

*Place.* Section 2, row k; section 6; step 8; section 7.

*Reason.* The reference learned from batches of 64 items, was reset at every item, and was scored on panels outside the stream (the inventory, sections 1.2 and 1.3). The tissue learns from one unbroken stream. "Examples needed" and "computation spent" have no common unit until one is defined. Candidates are bytes of stream read, presentations of a taught item, and arithmetic operations. The contract (design block, item 4) asks for equal information and feedback budgets between compared learners. The reference has no figures for the streams of steps 3 to 5. Even for steps 6 and 7 it was not run on an unbroken stream. Section 6's "no new run of the reference is asked for" therefore leaves B1 and B2 unreadable before step 8. It also leaves the contract's positive control (a learner known to solve the task on the same stream) missing at every step.

Two related claims should be qualified. Step 7's "which the reference does not do" rests on the record's finding that, without a store and *with weight decay*, about 0.8 falls to about 0.02 after one other visit (LOG, 16:50). Step 7's "verified reference lives" are lives in the batched, reset format.

*Proposed change.* Define in section 6 one unit of examples (presentations of items of the scored family) and one unit of computation (operations per byte of stream for the tissue; operations per item for the reference). Allow one run of the reference per step on the step's own stream, as positive control and as the B1 and B2 curve. If a run of the reference on an unbroken stream is judged out of scope, say that B1 and B2 are compared only on the record's format and that this comparison is indirect. Qualify step 7 with "with weight decay 0.25".

### Finding 11. The places where the route is most likely to stop are not named, and there is no rule for stopping

*Place.* Section 4, condition 5 ("A step that is not reached is not passed over"); step 4 ("the route does not go on"); steps 2 and 5.

*Reason.* Condition 5 sends work back to step 0 without a limit. With 31 open choices and a grid of seven values each, the space cannot be searched, and the roadmap gives no rule for when to stop and report. In my reading, the most likely stopping places are these.
1. *Step 2, closing the loop.* The output end must learn from an answer that arrives later through the input end, with connections found by random trial within neighbourhoods (gap 4). I am moderately confident that random trial with pruning by use, without any guiding signal, forms useful paths of several connections slowly, if at all, at the sizes of step 2. Living tissue guides the growth of connections by local chemical signals. That is within Mr. Xu's rules (choices C26 and C28: a local signal that permits the making of connections) and should be named now as the first remedy.
2. *Step 5, generalisation by local learning.* I am confident that networks trained by back-propagation can learn modular addition from a fraction of the pairs, often only after long training past the point of fitting the taught pairs (the result reported as "grokking", Power and colleagues, 2022). I am confident that the most common biological account of task learning by local rules uses a third factor, a reward or neuromodulatory signal that reaches many cells. Here that is excluded, since no signal reaches the whole tissue. I know of no published result in which a network with purely local learning, no broadcast signal and no error sent backward learns an arithmetic relation and answers held-out pairs. I am moderately confident that none is well established. One known way to give a single cell a target without a backward pass is a cell with two parts, one receiving the predicted inputs and one receiving the predicting inputs, with the strengths on one part changed to reduce the mismatch with the other (for example Urbanczik and Senn, 2014; I am fairly confident of this reference). That is allowed by Mr. Xu's ruling on separate parts of a cell and could be named as the first form for the output end.
3. *Steps 4 and 7, keeping without a store.* The record shows the reference loses nearly everything after one other visit without a store. Synapses with several components at different time scales extend memory lifetime in models of random-pattern storage (Benna and Fusi, 2016; fairly confident). This bears on the first form of C31 but says nothing about whether codes that generalise are kept.

*Proposed change.* Add a section "Where the route may stop" that names these three places, the first remedy for each that is within Mr. Xu's rules, and a stopping rule for each step: a stated number of tissue forms and a stated number of CPU-hours. After these, the step is reported as not reached, together with the specific obstacle found, and put to Mr. Xu before any further search. State that a well-documented negative at step 2 or step 5 is a result for the project.

### Finding 12. The arrangement of fixed numbers cannot be searched as written, and the claim that early steps are cheap is unsupported

*Place.* Section 4, conditions 3 and 6; step 8.

*Reason.* Condition 6 places every fixed number on the grid 2 to 128. With tens of such numbers a full grid is impossible. The roadmap does not say how the grid is used: one number at a time around the first form, or a sample. The tissue document (section 1.11) notes that 128 may be too few for numbers of cells and that the grid for rates is read as reciprocals. These readings are not Mr. Xu's and must be put to him. Condition 3 says steps 1 to 5 need few cells and short streams. Steps 4 and 5, however, need long stretches for keeping and delayed probes, and possibly more than 128 cells.

*Proposed change.* State the rule: each fixed number is varied one at a time over the grid around the first form, and the number of tissues this needs at each step is given. Put the reciprocal reading and the extension beyond 128 to Mr. Xu. Replace "few cells and short streams" by figures in each step's line "Numbers".

### Finding 13. The input lines could supply arithmetic that the constraints forbid

*Place.* Step 0 (i) ("input lines that each respond to something particular in the bytes"); section 3, overlap 5.

*Reason.* The draft constraints file forbids supplied arithmetic heuristics (`supplied_arithmetic_heuristics`: "int() over decoded operand bytes", "carry or digit tables"). An input line that responds in proportion to a digit's value, or to whether a byte is a digit, would supply exactly the ordered code of digits that finding 4, item 2 says the tissue must learn. Generalisation would then rest on the experimenters' coding. Mr. Xu's statement describes encoder cells as reading passages of bytes with gaps (section 8), not as detecting features of meaning.

*Proposed change.* Add to step 0 (i): "An input line responds to the identity or bits of bytes at positions relative to the present byte, never to a numerical value or to a class of bytes chosen by the experimenters."

### Finding 14. Wording that is imprecise or hides a decision

*Place and proposed change, item by item.*
- Section 1: "Mr. Xu's words of 18 November state the same goal: end-to-end persistent learning with generalisation ... (section 8)". Section 8 describes what a cell does; the phrase of the goal is not there. Cite where the phrase comes from, or write "the goal Mr. Xu named".
- Section 1, row "Persistent": "the tissue's own repetition is accepted". Requirement A says "some rehearsal is acceptable". Use his word and say what limits "some".
- Step 4: "the floors the project's constraints file gives". The file's status is `DRAFT_PENDING_OWNER_DECISION`, and its floors (`old_max_drop` 0.1, `old_acq_margin` 0.05) were set for the batched reference. Say "the draft floors, pending the owner's decision".
- Step 4: "the forms ... are settled here". Overlap 3 and condition 4 say that no form is accepted on one property alone. Write "first chosen here, and read again at steps 5 to 7".
- Section 7: "E2 within E3" (step 6 heading) and "E0" for step 1. Step 1 includes learning of the input end, which the contract's E0 (access without a learned-value test) does not quite describe. Say "E0, with learning of the input end read from recordings".
- History: the draft is dated "about 22:35 UTC". The reader's clock read 22:32:22 UTC when the file already existed. Read the clock when writing a time.
- Section 7: "the period before the review ... reached E1 and not E3". Its mechanisms should be placed on the steps before step 3 is briefed, as section 8, item 3 arranges, so that step 3 does not repeat them.

## What is sound

1. The goal in section 1 is quoted faithfully from the contract, and the three properties with their condition match Mr. Xu's statement and rulings.
2. Reading everything from step 2 on from the bytes the tissue emits, scored at the first attempt before the right answer appears, is right and should be kept.
3. The conditions of section 4 (no error sent back, no answer given to a cell, no averaging over items, no reset, no marks of item boundaries, the earlier steps read again when the form changes) agree with Mr. Xu's rulings and with sentences 91 to 100 of the tissue document.
4. Overlap 3's rule that keeping and generalising are read on the same tissue in the same life, and that no form is accepted on one of them alone, is right, even though its reasoning is one-sided (finding 6).
5. Overlap 6 (the output end is taught through the input end) is right and should be carried into the table (finding 2).
6. Overlap 8 (a cell's number of connections belongs to the cell from the first step) follows Mr. Xu's statement.
7. Placing the reference as comparator and not as a stage of the route, and not counting the experiments that changed one part of a back-propagation network as steps, answers the earlier rejections.
8. Beginning each step at the smallest size that can show it, and naming the choices to reopen on failure, is the right discipline. It needs stopping rules (finding 11).
9. Step 5 already names the table of taught pairs and the frequency of the parts as countermodels, and a control without the earlier teaching. These need only the additions of finding 1.
