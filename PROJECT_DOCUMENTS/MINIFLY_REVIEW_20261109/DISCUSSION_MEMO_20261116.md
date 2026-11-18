# Memorandum on the discussions of 15 and 16 November 2026

From the outside reviewer, at Mr. Xu's request of 16 November 2026, 03:30 UTC: "Write a memo on everything we discussed so far in standard academic language and push it."

## 1. Scope and sources

This memorandum summarises the discussions between Mr. Xu and the reviewer from 15 November 2026, 20:53 UTC, to 16 November 2026, 03:30 UTC. It continues `DISCUSSION_MEMO_20261115.md`, which records the preceding period verbatim (14 November, 14:26 UTC, to 15 November, 20:45 UTC), and `AUDIT_NOTES_MEMO_20261114.md` before it. Section 3 restates the conclusions of the earlier period that the present one relies on, so that this document can be read alone.

The memorandum is a synthesis and not a transcript. Mr. Xu's own statements are reproduced verbatim in Appendix A, in order and with their times. Messages that Mr. Xu relayed from the coordinating session or from workers are listed there in abridged form.

Three kinds of statement are distinguished throughout:

- **measured**: a figure read from a receipt or computed from saved weights, with a note on whether the reviewer recomputed it;
- **argued**: a conclusion reached by reasoning or arithmetic and not tested by experiment;
- **proposed**: a design or recommendation.

The reviewer trained nothing. All of the reviewer's computations are forward passes of a numpy reader on saved weights, or arithmetic. The designs, audits and scripts referred to are in `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/` on the branch `claude/vibrant-cerf-sje726`; the principal document is `MEMO_TO_COORDINATOR_LEARNED_STRUCTURE_20261115.md` (below, "the memo to the coordinator"), now at revision 13.

## 2. Terms

- **The network.** A recurrent network with 1,124,113 parameters: a table of byte vectors, one input layer, a layer of 256 gated recurrent units (GRU), and a two-layer readout. It reads one byte at a time and predicts the next.
- **The present rule.** The learning rule under study. Errors are sent backwards through each linear map by the signs of its weights and one scale per matrix, in place of the weights themselves. The project's documents call it "the sweep". **Back-propagation** is the conventional rule, in which errors are sent back through the weights. A third rule, **K**, learns separate feedback weights.
- **Task family.** Four arithmetic operations on two-digit operands under one modulus (addition, subtraction, maximum, minimum). An item states a task code, the first operand, the operator, a number of irrelevant letters, the second operand and an equals sign; the answer is two digits and an end symbol.
- **Taught and untaught pairs.** Only a share of the operand pairs of a family is shown in training. Accuracy on untaught pairs measures generalisation. **Reduction pairs** are those whose answer requires the modulus (the sum is at least the modulus, or the difference is negative); a network can answer the others without having learned the modulus.
- **Birth and life.** A birth is training from random weights on the first family (modulus 89). A life is subsequent training on further families presented in sequence (moduli 97 and 73, and compositions). A **lived** network has completed a life; a **fresh** network has random initial weights.
- **Update.** One step of the optimiser on a batch of 64 items.
- **Computation to threshold.** The total computation spent up to the first evaluation at which accuracy on untaught pairs reaches a stated threshold.
- **The small world.** A reduced environment (540 tasks from 18 operations and 30 moduli, networks of 64 to 192 units) intended to make very long lives affordable.

## 3. The position at the start of the period

The following had been established or decided before 15 November, 20:53 UTC.

1. **Objective.** Mr. Xu's objective is persistent learning under fixed memory and computation, and the question is how local learning can exceed conventional back-propagation in generalisation. Any method is admissible provided it is not full back-propagation.
2. **What the trained networks contain (measured by the reviewer).** Every trained network represents the answer of a modular addition by a circular code with two periodic components. A life changes the recurrent weights by more than their original size and removes the first family.
3. **Programme.** After Mr. Xu's objection that a learner which is slower at first may overtake later, the programme was reorganised around long lives: a small world, trend measures, and a rule that a comparison is closed only when its ordering is stable at two successive lengths of life.
4. **Jobs.** E36 (relearning after a life), E37 (a lived against a fresh learner on a later family) and E38 (the cause of forgetting) had been audited and released; E39 (the small world) had been specified.

## 4. Chronology of the period

Times are UTC.

| Time | Subject | Outcome |
|---|---|---|
| 15 Nov, 20:53 to 21:00 | Terms of use of the repository | The Creative Commons statement was replaced by `LICENSE.md` on the default branch (commit `035f8755`): no commercial use, and any other use by others only with the owner's prior written permission |
| 21:05 | Audit of the small-world brief (E39) | Eight required changes, among them a start symbol and an end symbol exchanged in the generator |
| 21:14 | A third long training run for E34 | Approved by the reviewer |
| 22:15 to 22:32 | Reading of E36, E37 and partial E38; retention schemes | Section 5 below; retention mechanisms withdrawn as a research question (section 6.1) |
| 22:35 to 22:56 | Audit of E40; E38 complete | Two required changes; E40 released |
| 23:30 to 23:45 | Scale; computation to threshold | The quantity adopted as an objective (section 6.2); E41 specified and audited |
| 16 Nov, 00:10 to 01:36 | Learned structure | Section 7 |
| 01:36 to 01:55 | Design of E42 | Section 8 |
| 01:55 | New results | E40, E39, E35 read (section 5) |
| 02:02 to 02:26 | Audits of the E42 brief and of revision 3 of the small-world brief | Seven and six required changes (section 9) |
| 02:32 to 02:39 | State of the project; further tasks | No new large job; two small additions |
| 03:01 to 03:25 | What 64 updates change | Section 5.7 |
| 03:25 | Overall assessment | Section 10 |

## 5. Empirical results discussed

All results are on development data and have not been confirmed on tasks sealed in advance. Unless stated otherwise, figures are for seeds 73129, 93683 and 116089 in that order.

### 5.1 E36: relearning after a life

After a life, the first family, which was never rehearsed, is answered at chance on reduction pairs. When it is presented again, accuracy returns to 85% or more within 64 to 128 updates under all three rules. After 1,024 updates, accuracy on reduction pairs is 93.2, 88.1, 96.0 under the present rule and 97.4, 99.2, 95.0 under back-propagation, against 1.1 to 2.5 for fresh networks given the same updates. The reviewer recomputed these figures.

### 5.2 E37: a lived against a fresh learner on a later family

After 131,000 items of the third family, lived networks score 77 to 94% and fresh networks 7 to 19%. On the second family after 393,216 items the advantage of the lived network is 17 to 28 points under the present rule and 8 to 16 under back-propagation. The reviewer recomputed these figures.

### 5.3 E38: the cause of forgetting

When the stream leaves a family, accuracy on its reduction pairs falls from 97% to 18% within 64 updates and to zero by 1,024. This occurs with and without weight decay and under a uniform penalty towards the birth weights at two strengths, under both rules. Mixing items of the old family into the stream keeps it (98.3 and 96.8% under the present rule, 96.0 and 93.9% under back-propagation) and does not prevent acquisition of the new family (92.4 and 92.9%). These are the worker's figures. They refute the reviewer's earlier hypothesis that weight decay caused the loss.

### 5.4 E40: a new family taught from a tenth of its pairs

A new family (modulus 83) was taught from one tenth of its pairs to lived and to fresh networks for 8,192 updates. Accuracy on untaught reduction pairs:

| Network | Present rule | Back-propagation | K |
|---|---|---|---|
| Lived | 77.0, 82.1, 83.7 | 85.0, 94.5, 89.3 | 79.2, 89.5, 83.8 |
| Fresh | 6.0, 7.6, 6.4 | 7.7, 11.8, 8.3 | (not run) |

Lived networks exceed 90% on taught pairs after 64 updates in eight of nine runs, when each taught pair has been shown about 1.5 times on average. Fresh networks memorise the taught pairs and do not generalise. While the new family is learned alone, the family with modulus 97 falls by 55 to 80 points. The reviewer recomputed the table from the fifteen weight files at update 8,192; every figure agrees with the worker's report.

### 5.5 E35: modules supplied by hand

Two hand-built arithmetic modules were tested at one tenth of the pairs. Given the operands in separated form, both generalise (for the ring module, 97.3, 86.8 and 92.1%). Learning from the byte stream, neither helps. In E35-B the network forced to use the module scores 18.0, 32.6 and 19.1% on untaught pairs against 19.6, 32.1 and 27.8% without it; the registered stopping rule applied and that line is closed. In E35-A, on one seed, the four configurations score 19.6, 21.1, 19.4 and 6.6%; its last stage (two further seeds) was released and is running. These are the workers' figures.

### 5.6 E34: long training with a confidence-weighted error; E39: the small world; E41

- **E34 (one seed, 98,304 updates).** Over the second half of training, accuracy with 40 irrelevant letters averages 97.0% for the present rule with the error at each position weighted by the network's own confidence, 92.6% for plain back-propagation, and about 98.8% for back-propagation with the same weighting. The weighting therefore improves both rules, and the improvement cannot be attributed to the present rule. Worker's figures.
- **E39, instrument.** All nine checks pass, and the worker's generator reproduces the reviewer's independent implementation.
- **E39, calibration.** The calibration failed at all three settings. Operations that take a maximum or minimum are learned within a first visit. Linear operations, which are nine tenths of the tasks, remain near chance: 1, 20 and 30 of 192 reach 80% at the three settings. Under back-propagation the mean score rises from 10.8% in the first sixteen visits to 27.9% in the last sixteen at the largest setting. The present rule scores lower than back-propagation in every world at every setting (about 12% against 18 to 20%). Worker's figures.
- **E41.** In progress; no reading.

### 5.7 What the first 64 updates change (measured by the reviewer)

The reviewer compared the weights of lived networks at the end of the life with their weights after 64 updates on the new family of E40, on three seeds for the present rule and back-propagation and one seed for K (memo to the coordinator, section 23; script `e40_row64_changes.py`). The measure is accuracy on untaught reduction pairs of the new family.

| Network | Present rule | Back-propagation |
|---|---|---|
| End of life | 6.2, 2.3, 0.1 | 4.2, 0.4, 11.6 |
| After 64 updates | 75.6, 69.8, 75.5 | 84.1, 92.3, 95.7 |
| End of life with only the recurrent layer's changes added | 57.1, 63.0, 64.6 | 68.0, 85.2, 87.5 |
| With only the readout's changes added | 7.3, 5.3, 0.8 | 7.2, 2.1, 13.7 |
| With only the largest 5% of the changes | 15.6, 11.0, 2.1 | 4.7, 14.9, 17.0 |
| With only the largest 50% | 66.2, 68.1, 72.8 | 76.9, 92.9, 95.2 |
| Each matrix's change reduced to rank 16 | 71.0, 61.8, 67.0 | 81.2, 90.5, 95.4 |

Four conclusions follow.

1. What is learned in 64 updates is located in the recurrent layer, in its input weights and its recurrent weights.
2. The vectors of the four new task codes, which had never been trained, are used almost unchanged.
3. The change is distributed over many weights but is of low rank: half of the changed weights are needed for most of the effect, whereas sixteen directions per matrix recover nearly all of it.
4. The gain of the new family and the loss of the earlier one occur in the same matrix: restoring the recurrent weights removes most of the new family and returns the family with modulus 97 from 48–77% to 84–96%.

The distributed character of the change is a property of the optimiser, which moves every weight whose gradient keeps its sign by a similar amount. It does not show that a change confined to few weights would be insufficient.

## 6. Decisions on the programme

### 6.1 Retention mechanisms

Mr. Xu observed that earlier work in the project had tried slow learners, long-term stores and similar mechanisms with mixed results, and that the plan was returning to them. The reviewer agreed and withdrew a recommendation to study rehearsal schedules. The position adopted is that retention is not a research question at present. E36 and E38 show that what is lost is recovered within about 100 updates and that plain rehearsal from a bounded store suffices. Any further mechanism requires a stated failure that plain rehearsal cannot remove (memo to the coordinator, section 20).

### 6.2 Computation to threshold, and scale

Mr. Xu proposed that if a larger network or more data reduces the number of training steps needed to reach a threshold, so that the total computation is equal or lower for an equal or better result, this is a legitimate objective, and that the axis to be scaled should be determined by experiment. The quantity was adopted (section 2). E41 measures it at three widths and three shares of taught pairs under both rules. The reviewer's doses for E41 were too short and were corrected by its audit.

### 6.3 The small world after the failed calibration

The reviewer attributed the failure to the absence of an initial learning phase: in the small world a life begins from random weights, whereas at full scale every life follows a birth. This is argued and not tested. The reviewer recommended, and the coordinator specified, two further jobs: a measurement of how many updates the small network needs to learn one linear task alone, and a life preceded by a birth on its first tasks. The specification was audited (section 9) and released as revisions 4 and 5, which include a third arm (the present rule's life after a birth under back-propagation) and a third network size of 192 units at the reviewer's request. The measurement is running.

### 6.4 Other decisions

- A conditional full-scale experiment on a sequence of families was withdrawn, because E40 shows that a lived network learns a new modulus in about 64 updates.
- No new large job was assigned on 16 November: the workers were occupied, and the next designs depend on results not yet available.

## 7. The discussion of learned structure (16 November, 00:10 to 01:36 UTC)

This section records an argument developed in successive exchanges. None of it has been tested.

### 7.1 Quantities other than the number of units

Mr. Xu asked which quantities besides the number of units could be scaled, given that connections among cells and connections that integrate over different time scales ought themselves to be learned. The reviewer listed: the number of connections per unit; the order of interaction (additive or multiplicative); the range of time scales; the number of internal steps per byte; the distance over which credit is assigned; the size of the input code; the number of items per update; and prior experience.

### 7.2 Whether the gating structure can be learned

Mr. Xu asked whether the structure of the GRU could itself be learned. The reviewer first answered in terms of generalised units and architecture search, which was not the question. Mr. Xu clarified that the structure should be acquired in the same way as the higher-order connections are, neither assigned nor found by a separate search. The corrected answer was as follows. A gate is a multiplicative connection, in which the value of one unit scales the influence of a second on a third. It is therefore a connection among three units, which is what Mr. Xu calls a hyper-connection, and the GRU is one fixed pattern of such connections. A single mechanism, learnable multiplicative connections, can therefore cover both.

### 7.3 The first-order assumption

Mr. Xu pointed out that the discussion still assumed S(T) = h(S(T−1), g(T)), whereas a state may depend on several earlier moments, different cells may be organised differently, and the complete gating form may be unnecessary. The reviewer separated three assumptions contained in the first-order form, each of which can be relaxed for each unit independently: which earlier moments a unit reads; how it combines its inputs; and when it is updated.

### 7.4 The term "recurrent", and information from a later moment

Mr. Xu objected to the reviewer's continued use of the word "recurrent", gave the example of a state depending on moments 4, 14, 29 and 100 steps earlier, and added that a slow cell might receive information from a faster cell at a nominally later moment. The reviewer withdrew the word and made two statements.

- Any system with fixed memory satisfies a weaker property by definition: its entire stored content at one step is determined by its entire stored content at the previous step and the new input. This places no restriction on which earlier moments a unit reads.
- Apparent dependence on a later moment requires no additional mechanism. A unit that changes once in several steps has a value that describes an interval and is determined when the interval ends. A network of connections with delays can be computed in order of time if and only if the delays around every cycle of connections sum to a positive number. The only real restriction is that an answer cannot depend on input arriving after it is due.

### 7.5 Whether all of this can be learned

The reviewer argued that it can, since every structural choice can be represented as a weight: which earlier moments a unit reads, as one weight per source unit and delay; multiplicative interaction, as products of weighted sums; the time of updating and the retention of a value, through a unit's access to its own earlier values. Four limits were stated: a delay longer than the stored history cannot be learned; the number of weights grows with the number of delays; the time step, the total memory, the learning rule and the moment at which an answer is due must be fixed; and selection of an earlier moment by its content is attention, which the constraints of October excluded.

The reviewer also noted what the first-order assumption has been providing. In training, the first operand lies 5 to 17 steps before the equals sign; in the test with 40 letters it lies 45 steps before. A network that reads only the previous step must retain the operand at every step, and such a mechanism can extend to 45 steps. A network free to learn delays could learn one delay for each training distance and would then fail the test.

### 7.6 Absence of a connection

Mr. Xu observed that if two cells have no connection, no weight can give one access to what the other stores, so that over-fitting by weights and over-fitting by connections are the same phenomenon, and asked whether this could be used. The reviewer agreed and stated the observation exactly: where no path of connections leads from one unit to another, the second is independent of everything the first has stored, for every setting of the weights; a small weight gives no such guarantee. The reviewer's proposal that all connections be present from the start removes this guarantee, and was reversed: the network should begin with few connections and acquire others. Three uses were identified: a preference for connections useful on more items; exact measurement, since connections can be counted; and lower cost, since absent connections are neither stored nor computed.

### 7.7 Two corrections

- The reviewer had stated that a procedure adding one connection at a time favours solutions that need a single connection. Mr. Xu replied that nothing in the setting requires connections to change one at a time: under the hypothesis of locality a neuron interacts with a number of neurons fixed in advance, and may add or drop many connections at any update. The reviewer accepted this. The difficulty had come from judging each absent connection separately, not from the number added, and it disappears when connections are added in groups and their weights learned together. The selection then takes place when connections are removed.
- The reviewer then stated that the fixed limit on a cell's connections decides between a solution that treats each case separately and a general one. This was corrected in the next exchange: cases can be divided among several cells, so what bounds the case-by-case solution is the total number of connections in the network.

### 7.8 The simplified setting

Mr. Xu proposed, for simplicity, that each cell be connected to at most N = 10 neurons, with N adjustable and the strengths learned. In answer to three questions from the reviewer, Mr. Xu stated: that connections may be fixed or, preferably, adjustable, but only within a neighbourhood defined in advance by coordinates assigned to the cells, to impose locality; that a cell's own previous value is available, that each cell has its own pre-set delay or frequency of processing and absorbs what the cells connected to it emit during its period, and that the way it processes this is to be learned, with variants to be tried; and that the input and output parts of the network may remain as they are.

## 8. Experiment E42

The design is section 22 of the memo to the coordinator. Its elements are these.

- **Cells.** The 256 units are placed on a 16 by 16 grid whose opposite edges are joined. A cell may be connected to N cells among the 48 within three rows and columns of it, and reads its own previous value in addition.
- **Connections.** Fixed, or changing: every 256 updates a connection whose strength has fallen below a threshold is replaced by a connection to another cell of the neighbourhood drawn at random.
- **Periods.** Each cell has a pre-set period of 1, 2, 4, 8 or 16 steps, is updated once per period and keeps its value in between. Four variants are defined for combining what arrives during a period.
- **Comparisons.** Four configurations under both rules (the present network run by the new code; fixed connections; changing connections; changing connections with periods), then single changes of N, of the neighbourhood, of the set of periods and of the variant.
- **Measures.** Updates and computation to a threshold on untaught pairs; accuracy with 40 letters; the number of connections replaced and the final pattern of connections.

Two considerations shaped the design. First, units with pre-set time constants had been tried once in the project, without connections among them, and had failed on untaught combinations (1.5, 0.5 and 7.5% against 73.6, 78.8 and 75.1% for the GRU); periods are therefore introduced as a separate step. Second, arithmetic done in advance shows that a cell with a long period can carry the first operand to the equals sign without learning to retain anything, in 6%, 44% and 67% of training cases for periods 4, 8 and 16, and in no case with 40 letters. The design therefore predicts that longer periods ease training and impair the test with 40 letters.

The experiment cannot show a saving in computation, because the input and output parts hold most of the weights.

The coordinator's brief was audited (section 9) and released as revision 2. A worker has begun. Mr. Xu asked that E42 be left out of the status reports and the assessment of 16 November.

## 9. Audits in the period

Each audit was carried out by a separate agent; the reviewer rechecked its main findings before reporting.

| Brief | Required changes | Errors of the reviewer's own design found |
|---|---|---|
| E39, small world (revision 1) | Eight; start and end symbols exchanged in the generator; the gate replaced | — |
| E40 (revision 1) | Two; the verdict moved to the gain on reduction pairs | The reviewer's mark of "15 points above fresh" could not separate learning from what a lived network answers before training |
| E41 (revision 1) | Five | Doses too short; no rule for runs that do not reach a threshold |
| E42 (revision 1) | Seven | The start of one variant divided the recurrent weights by the period, which made the present rule send back errors of the wrong size (0.40 of the right size for period 1, 3.2 times for period 8) |
| E39 (revision 3) | Six; the 32-item criterion is met by chance at a true level of 0.55 to 0.7 | — |

All five briefs were released after the changes without a second audit.

## 10. Assessment as of 16 November

This is the reviewer's judgement, given at Mr. Xu's request with E42 set aside.

1. **No result shows the present rule to be superior to back-propagation.** At full scale the two agree in direction in every measurement and the present rule is slightly lower. In the short lives of the small world it is clearly lower.
2. **The experiments so far could hardly have shown superiority.** The present rule approximates back-propagation and both learners use the same network. This is argued, not measured.
3. **The evidence supports a weaker conclusion:** the present rule is sufficient for every phenomenon of continual learning measured so far (acquisition, forgetting, rapid recovery, transfer, generalisation from few examples), at a small cost.
4. **The principal source of efficiency and of generalisation is prior experience.** A lived network learns a new family in about 64 updates from a tenth of its pairs. A fresh network needs about 4,000 to 8,000 updates when half the pairs are taught, and does not generalise from a tenth. Measured by computation to threshold, this saving is far larger than any difference between rules observed.
5. **Three matters remain open:** whether the present rule overtakes over a long life, for which the only indirect evidence is unfavourable and weak; whether it depends more strongly on the width of the network; and confirmation of the findings on tasks sealed in advance.

## 11. Recommendations and pending work

1. Describe the present state of the project as "a local rule is sufficient, and experience is the main source of efficiency", not as "a local rule is superior".
2. Treat the long-life comparison in the small world, with an initial learning phase, as the decisive test of the original question, with the criterion written beforehand: the present rule's score or its trend in the later part of life must exceed that of back-propagation.
3. Confirm E40 with a second new modulus and new seeds, with predictions registered in advance (about 4 CPU-hours).
4. Close E35, and record E34's confidence weighting as an improvement of the training signal that is independent of the rule.

Pending at the time of writing: E41 (one seed in progress); the single-task measurement in the small world; the last stage of E35-A; E42.

## 12. Errors of the reviewer in the period

1. The hypothesis that weight decay causes forgetting, refuted by E38.
2. A recommendation that returned to retention mechanisms, withdrawn.
3. The doses of E41, and the missing rule for runs that do not reach a threshold.
4. The question on learning the gating structure, misread twice.
5. The word "recurrent", which carried the assumption under discussion.
6. The proposal that all connections be present from the start, which gave up the guarantee that absence provides.
7. The attribution of a difficulty to adding one connection at a time.
8. The statement that the limit per cell decides between solutions.
9. The start of the variant `source` in E42, and an estimate of E42's cost that was too low.
10. The audit of the first small-world brief did not ask whether the small network could learn a task within one visit.
11. On three occasions Mr. Xu found the reviewer's writing incomprehensible and asked for standard academic language.

## Appendix A. Mr. Xu's statements in the period, in order

Times are UTC. Statements are verbatim. Chinese statements are followed by the reviewer's translation. Messages relayed from the coordinating session or from a worker are described in one sentence and are not reproduced.

1. **15 Nov, 20:53** "Push all the discussions as a long memo in the repo. BTW check the repo's declaration page, make sure it is strictly non-commercial and others use it need Mr. Zhou's permission."
2. **15 Nov, 20:56** "What is the problem with CC license?"
3. **15 Nov, 20:58** "Draft a license for Mr. Zhou as he does not want it to be used for any commercial purposes / external use for pedagogical purposes etc needs to get his permission."
4. **15 Nov, 21:00** "1) Years are 2026-2126. 2) Changwei Zhou. 3) Contact by [the address given in LICENSE.md]. 4) Law being US law for now. Can you modify the main branch accordingly?"
5. **15 Nov, 21:05** "Do you have new tasks or we are okay for now?"
6. **15 Nov, 21:14** "A new question" followed by a relayed message, not reproduced. The relayed question was whether a third long training run of E34 should be queued.
7. **15 Nov, 22:15** "A lot of jobs have shipped. Take a look."
8. **15 Nov, 22:23** "Use the standard academic language. Re-write the above. "Forgetting here is shallow" is incomprehensible."
9. **15 Nov, 22:32** "Earlier many minifly period attempts tried to use slow learner, long term storage, and other fancy schemes, but these yield at best mixed results. So we are revisiting the issue to some extent. What is your thought?"
10. **15 Nov, 22:35** "Send an agent to do the audit etc:" followed by a relayed message, not reproduced. The relayed message reported E39's release, the merging of E36, E37 and E35-B, and asked for an audit of the E40 brief.
11. **15 Nov, 22:42** (relayed) The coordinator's acknowledgement of memo revision 7: its own sentence withdrawn, retention recorded as not a research question, E38's second stage not to be released.
12. **15 Nov, 22:56** "E38 is done. Do you have new jobs to assign? Do we have other pending jobs?"
13. **15 Nov, 23:14** "现在总的来说是什么情况呢？"
   - *Translation:* What is the situation overall now?
14. **15 Nov, 23:30** "scaling的帮助如何呢？同样的任务，如果scaling有帮助，也可以认为在固定算力下我们能做的更多。"
   - *Translation:* How much does scaling help? On the same task, if scaling helps, we can also say that at fixed computation we can do more.
15. **15 Nov, 23:34** "对，如果网络计算规模变大，或者数据变多，但是为了达到目标(threshold)需要的训练次数本身变少，结果一样甚至更好，平均下来算力一样甚至更少。那这是一个可以讨论的目标。至于scale which axis是可以通过实验讨论的。"
   - *Translation:* Yes: if the network's computation grows, or the data grow, but the number of training steps needed to reach the target (a threshold) itself falls, with the same or a better result, then on average the computation is the same or less. That is a goal worth discussing. Which axis to scale can be settled by experiment.
16. **15 Nov, 23:45** (relayed) The coordinator's report that the E41 brief was written and pushed (213c38f4), with six points for the audit.
17. **16 Nov, 00:10** "In theory we should have many hyper-connections between the cells that we can scale as well as connections between cells absorbing info in different frequencies and so on. But the issue is in our set up we want these to be learned during the learning process. Other than the total number of neurons, are there are other parameters we can scale？"
18. **16 Nov, 00:31** "Question: Is it possible to make the current GRU structure itself also a learned factor?"
19. **16 Nov, 00:34** "Use standard academic language, the above is incomprehensible again"
20. **16 Nov, 00:37** "But this is not what I meant. I meant to self-learn this much as the hyper-connections are learned, not assigned or searched in the learning process."
21. **16 Nov, 00:37** "So in the learning experience, this together with the hyper-connections will be learned together"
22. **16 Nov, 00:43** "Let us think about this carefully. We are still assuming implicitly that S(T)=h(S(T-1), g(T)) with S(T-1) being the state in T-1, and g(T) being some updates that happens only in time T. h then combines the info together, whether diagonally or otherwise. But in reality it may be that S(T) is dependent upon into at T-5 and g(T), for example. We had discussed this earlier. In short the gating structure may be arbitrary and the complete form not even needed depending on the task, and different cells could have different learning structures."
23. **16 Nov, 00:48** "But I am confused why you still claim this is "recurrent". We cannot rule out messy situation that S(T)=h((S(T-4), T(-14), T(-29), T(-100)),g(T)) for example. And in reality as the cell proceed from T to T+4 for example, S(T+4) could receive info from the future higher frequency cell at T+5 via some residual connections. Since we cannot figure this out before hand, can this all be learned? Spend some more time thinking about this"
24. **16 Nov, 01:05** "But here is the issue. If a cell A and B has no hyper-connection at all - not in the past, not now, then the weights logically cannot connect A with a cell B that remembers distant history. In other words A does not know this part of the history even if the weight is high. So to some extent the apparent overfitting of weights and hyper-connections during the learning process is the same thing. Can we utilize this?"
25. **16 Nov, 01:11** "The writing quality has degraded again. Use standard academic language."
26. **16 Nov, 01:18** "I am confused. In theory due to the local learning hypothesis, the neuron S_i can only interact with a pre-specified (perhaps high) number of neurons. So in our setting depending on the update frequency, S_i can add 100 connections in T=10, and add zero at T=11, drop 50 at T=12 etc. Nowhere did we dictate that S_i have to adjust by one connection at a time."
27. **16 Nov, 01:25** "Let us presume for simplicity that each cell can connect at most N=10 neurons (after all addition is simple), but we can adjust N and the cell can learn the connection strengths, do you have more questions before we proceed to design the experiment?" [The word "addition" is Mr. Xu's own correction, sent one minute later; the message first read "additional".]
28. **16 Nov, 01:36** "1) both is fine, but ideally the connections can adjust. however a neuron is only allowed to adjust within a pre-defined neighborhood, say via some pre-specified embedding coordinates to mimic biological situation. This is to impose locality. 2) The cell's previous value would be available, and each cell has its own pre-set delays or info processing frequencies. During the period it absorbs info from all other cells that emits to it. So cell at T=0 and T=5 may get inputs from the connections. How to process this will not be pre-determined and will be part of local learning, and you can try some variants. 3) Yes, the encoder and decoder parts can stay."
29. **16 Nov, 01:55** "先不考虑E42，目前的实验结果有更新吗？"
   - *Translation:* Leaving E42 aside for now, are there updates to the experimental results?
30. **16 Nov, 02:02** (relayed) The coordinator's request to audit the E42 brief and its specification file at 95f2d80f, with six points.
31. **16 Nov, 02:11** (relayed) The coordinator's report that E38, E39, E40 and E35-B were merged and that revision 3 of the small-world brief (a3275395) adds a single-task measurement and a life with a birth, with five points for the audit.
32. **16 Nov, 02:25** "E35 seems finished too?"
33. **16 Nov, 02:26** "Is E39 audit done BTW?"
34. **16 Nov, 02:32** "不考虑E42，项目目前的状态是？"
   - *Translation:* Leaving E42 aside, what is the present state of the project?
35. **16 Nov, 02:39** "你有其他任务吗？还是先等现在的任务跑完？"
   - *Translation:* Do you have other tasks? Or should we first wait for the present ones to finish?
36. **16 Nov, 03:01** (relayed) The E40 worker's report that five weight files at update 64 for seed 73129 were committed (45b6fd79).
37. **16 Nov, 03:17** (relayed) The E40 worker's report that four further weight files at update 64 for seeds 93683 and 116089 were committed (6922a3a4).
38. **16 Nov, 03:25** "综合目前的信息，忽略E42，你的看法是？"
   - *Translation:* Taking the present information together and ignoring E42, what is your view?
39. **16 Nov, 03:30** "Write a memo on everything we discussed so far in standard academic language and push it."

## Appendix B. The reviewer's commits in the period

On `claude/vibrant-cerf-sje726`, oldest first. Three commits on other branches belong to the period as well: `035f8755` on the default branch (the terms of use), and the E40 worker's `45b6fd79` and `6922a3a4` (weight files committed at the reviewer's request).

| Commit | Subject |
|---|---|
| `a3bdc854` | README: put the terms of use at its head and point to LICENSE.md |
| `3a426acc` | LICENSE.md: fuller draft of the terms of use for the owner's review |
| `8880de30` | LICENSE.md: the owner's details filled in; draft marks removed |
| `b9d6cb02` | Add the second implementation of the E39 small world, written from the brief's text alone |
| `4fb9ed40` | e39_world.py: work in progress on the E39 audit's second implementation |
| `b622775f` | e39_world.py: the coordinator's generator rules re-implemented; the scorer with corrected definitions |
| `de157e63` | Audit of the E39 brief (small-world protocol, jobs L0 and L1) |
| `ec8ea8e8` | Memo revision 5: what the audit of E39 changes in Part II; a note on R8 in the audit |
| `e41d41f3` | Memo revision 6: E36, E37, E38 so far, E35-B and the long births read; one new job |
| `2759b8b3` | Memo revision 7: on returning to retention schemes |
| `65f3411c` | Add the read-only checks behind the audit of the E40 brief (audit in progress) |
| `301f0366` | e40_brief_checks.py: work in progress on the E40 audit |
| `24ba1e9e` | Audit of the E40 brief (a new family from a tenth of the pairs) |
| `089409fe` | Audit of E40: note that the coordinator has since recorded memo revision 7 |
| `557e3749` | Memo revision 8: computation to threshold, and scale as an axis |
| `3b515f00` | Checks behind the audit of the E41 brief (computation to threshold by width and taught fraction) |
| `861bc99d` | Audit of the E41 brief at 213c38f4: five required changes, then release without a re-audit |
| `43a89440` | Memo revision 9: E41's doses corrected to agree with its audit |
| `f68f2b7d` | Memo revision 10: job E42, cells with a limited number of local connections and their own periods |
| `d1979d71` | Work in progress: the reader and the checks for the audit of the E42 brief at 95f2d80f |
| `634a606a` | Audit of the E42 brief and its specification file at 95f2d80f: seven required changes, then release without a re-audit |
| `7378e653` | Memo revision 11: the start of the variant `source` corrected after the audit of the E42 brief |
| `ae61164e` | Work in progress: e39_world.py extended for the audit of revision 3 of the small-world brief |
| `d0047917` | Work in progress: the checks for the audit of revision 3 of the small-world brief |
| `7b59ae8b` | Audit of revision 3 of the E39 brief (one linear task alone, and the life with a birth): six required changes |
| `9f86761b` | Memo revision 12: what the first 64 updates change when a lived network learns a new family |
| `3845b4ed` | Memo revision 13: what 64 updates change, repeated on three seeds |
