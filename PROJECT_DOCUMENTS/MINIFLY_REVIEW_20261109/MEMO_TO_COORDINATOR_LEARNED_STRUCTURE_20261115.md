# Memo to the coordinator: what the saved GRUs have built, what a life takes away, and three jobs to brief (15 November 2026)

From the outside reviewer, at Mr. Xu's request: "Write an appropriate memo for the coordinator to implement."

**Revision 2, the same evening.** E33 addendum 1 and E34 stage 1 have reported on `claude/vibrant-curie-ivmrt1` (ba6f5d52). I recomputed both tables from the committed weights; section 14 gives the check and what it changes. Jobs A, B and C are unchanged. One item is added to section 1 (item 5), one bullet to section 4 and three rows to section 12.

**Revision 3, later the same evening.** Three things are new. (a) The audit of the E36, E37 and E38 briefs is in `AUDIT_E36_E37_E38_BRIEFS_20261115.md` beside this memo; two of its required changes correct marks I proposed here, and sections 5 and 7 now carry the corrected marks. (b) Table 3.3 is restated with whole answers, as the harness scores them; the earlier table counted the two digits only. (c) Part II (sections 15 to 18) sets out the programme from here, after Mr. Xu's statements of this evening: a continual learner judged over a long life. Items 6 and 7 of section 1 are the new requests.

**Revision 4.** Section 17.10 is added: how short and long runs are combined (rate laws, trunks and branches, two probes, agreement between lengths, time in the learner's own units). It adds requirements to the protocol brief of job L0 and two items to L2; nothing else changes.

**Revision 5.** The audit of the E39 brief is in `AUDIT_E39_SMALL_WORLD_PROTOCOL_20261115.md`, with the second implementation in `e39_world.py`. It changes three things in this memo's Part II; section 17.11 lists them. Where 17.11 and an earlier section differ, 17.11 holds.

**Revision 6.** E36 and E37 have reported, E38 and the long E34 births in part, and E35-B's first stage. Section 19 gives my recomputation, what the results say, what they change in Part II, and one new job (item 8 of section 1). Section 3.6's explanation by weight decay is refuted by E38 and is marked so.

**Revision 7.** Section 20 is added after Mr. Xu pointed out that the plan was returning to the retention schemes of the earlier period. It withdraws item 2 of section 19.3 and sets a rule for adding any mechanism.

**Revision 8.** Section 21 is added after Mr. Xu's statements on scale: the quantity to compare is the total computation needed to reach a threshold, and which axis to scale is to be found by experiment. It asks for one new full-scale job (item 9 of section 1) and adds width as an axis in the small world.

**Revision 9.** The audit of the E41 brief is in `AUDIT_E41_COMPUTATION_TO_THRESHOLD_20261116.md`. It corrects two things in section 21.4 that were mine, the doses and the treatment of a run that does not reach a threshold; section 21.4 now carries the corrected doses.

**Revision 10.** Section 22 is added after Mr. Xu's statements of 16 November: each cell is connected to a limited number of cells inside a neighbourhood fixed in advance, the connections may change during learning, and each cell has its own pre-set period. It asks for one new full-scale job (item 10 of section 1). The arithmetic behind it is in `e42_design_checks.py` beside this memo.

**Revision 11.** The audit of the E42 brief is in `AUDIT_E42_CELLS_BRIEF_20261116.md`, with a numpy reader of the cell layer in `e42_cells_reader.py`. It corrects one thing in section 22.3 that was mine, the start of the variant `source`; the table there now carries the corrected form. It also defers `P8-mean` and replaces my cost estimate (sections 22.4 and 22.9).

**Revision 12.** Section 23 is added: a read-only measurement of what the first 64 updates change when a network that has lived learns a new family, on E40's files of seed 73129. It asks for four more files (item 11 of section 1).

**Revision 13.** Section 23 is repeated on seeds 93683 and 116089 with the four files the E40 worker committed (6922a3a4). The four readings hold on all three seeds under the sweep and under backpropagation; the tables of 23.1 now give the three seeds. Item 11 of section 1 is done.

**Revision 14.** Section 24 is added after Mr. Xu's objection that long-term memory exists without rehearsal. It restricts what section 20 established, gives a measurement of what survives a life without rehearsal, and asks for one small job (item 12 of section 1).

**Revision 15.** The audit of the E43 brief is in `AUDIT_E43_UNRELATED_LEARNING_BRIEF_20261116.md`. Its main finding bears on section 24 itself: under the standing weight decay, a family that receives no gradient is removed by the decay alone within about 2,000 updates. Section 24.6 gives the figures and what they change; the cost in 24.5 is corrected.

**Revision 16.** E43, E39's job L1a, E35-A's last stage and E41's first seed have reported. Section 24.7 reads E43 and asks for three additions to it (item 13 of section 1). Section 25 reads L1a, withdraws my explanation of the failed calibration, and proposes what replaces the small world's task set (item 14).

**Revision 17.** E42's first seed is read and checked with my reader (`E42_SEED_73129_VERIFICATION_20261116.md`). Section 22.12 gives the reading, one defect of my own replacement rule that it exposes, and what follows. Nothing is asked for until the other two seeds report.

**Revision 18.** The audits of the E46 and E44 briefs are in `AUDIT_E46_WEIGHT_DECAY_BRIEF_20261116.md` and `AUDIT_E44_LONG_LIFE_WORLD_BRIEF_20261116.md`. Section 26 is added after Mr. Xu's statement that locality must hold at the input and the output as well: it gives the arithmetic and the requirements for the job that follows E42 (item 15 of section 1).

**Revision 19.** E43's additions have reported in part (reminders complete; plain steps complete under backpropagation). Section 24.8 reads them with a measurement of my own on the weight files (`e43_where_the_loss_sits.py`): the first family is overwritten throughout the network, and neither the readout alone nor the optimizer's equal steps explain it. Section 27 asks for one small job that follows from it, E48, and item 17 for two additions to E43.

**Revision 20.** A change of name only. The coordinator has given the number E47 to the job of section 26 (locality at the input and the output; brief at 7247682a). The job of section 27, which I had called E47 in revision 19, is E48 from here on. Item 15 is done: the specification exists and is being audited.

**Revision 21 (17 November).** Part III (sections 28 to 30) is added at Mr. Xu's instruction to discuss next steps and to coordinate without him. Section 28 checks every result that arrived on 17 November (E41 and E42 on three seeds, E46, the E47 trial, E48, E49, E43's second addendum, E44's calibration) and gives the audits of the roster note for the first long lives and of the E50 brief. One reading of mine in 22.12 is withdrawn: periods do not bring the limited network level with the dense one. Section 29 is a measurement on E43's networks: every cell is in use on every item, and one family fills the state of the cells. Section 30 gives the decisions and one new job, E51. Items 18 to 23 of section 1 are the requests.

**Revision 22 (17 November, evening).** Section 31 is added. It corrects four statements of revision 21: the sweep's lag behind back-propagation, which depends on the width and was given as one figure; the design of E51, which is superseded by revision 2 of its brief; the number of lines of work; and section 30.6, since the reviewer has coordinated the workers from 20:22 UTC on Mr. Xu's instruction. It records where the learning rate and the weight decay of every run came from, adds one measurement (cells in use in the limited and the local networks), and points to the ordered plan and the log in `COORDINATION_20261117/`. Item 24 of section 1 is new.

**Revision 23 (17 November, late evening).** Section 32 is added: the first result of the evening (E50, stage 1, verified), four measurements and readings made from the record and from saved weights (the passage of a difference at the starting weights of the local networks; E40's saving counted in the learner's whole experience; E38's two failed checks under the sweep; the gate of the long-life world applied to the sweep), an incident on the workers' host, and the briefs written since revision 22. Item 25 of section 1 is new.

**Revision 24 (18 November, 01:14 UTC).** Section 33 is added: two results verified from receipts and weights (E51's trial, with three statements that limit it; E50's stage 2), what has been reported and is not yet verified, the briefs written and checked since revision 23, two corrections of the reviewer's own statements, the restarts of the workers' machines, and, in section 33.7, the handover of the coordination to you, which Mr. Xu announced at 00:44 UTC on 18 November. Items 26 and 27 of section 1 are new. Sections 33.5 and 33.8 were extended after the handover, at 01:38 UTC: a correction of the verification of E51's trial, the completed check of the brief of E55, and the verification of E51's Part B. Section 33.9 (the trial of the first long lives, verified) was added at 02:03 UTC.

**Revision 25 (18 November, 14:15 UTC).** Sections 33.10 to 33.13 were added in the night after revision 24 and are named in your decisions. New since your decisions 13 to 17: sections 33.14 to 33.17, four results verified from receipts and weights (E58, E59, the sweep's long lives, E52), each with what it means for Mr. Xu's requirements and with the corrections the worker has made to its report; and section 33.18, which reports that the reviewer read your decisions 13 to 17 ten hours late, what it forwarded at 14:15 UTC, and the state of the lines. Items 28 to 31 of section 1 are new. Added after revision 25, on the same day: section 33.19 (E54 verified on one seed), section 33.20 (the brief of E60 and a correction on messages) and section 33.21 (the outside advice memo, taken suggestion by suggestion, with the goal as Mr. Xu restated it to the outside session).

Everything measured here is read-only: forward passes on saved weights with the numpy reader, and a read of receipts. Nothing was trained. The files are the six E26 births and nine lives (E26 arms S and B, E31 arm K) on seeds 73129, 93683 and 116089. The lives were not designed for these questions, so sections 3.1 to 3.5 are readings, not outcomes of a registered test.

Shell names used below: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106; R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`. The scripts are on `claude/vibrant-cerf-sje726` at `b983f74a`; file arguments are paths under `$M/exp45/` on the default branch.

## 1. What is asked of you

1. **Brief three jobs**, in this order. Each brief comes to me before any worker starts, as usual.

   | Job | Question | Runs | CPU (my estimate) | Section |
   |---|---|---|---|---|
   | A | After a life, how many updates bring the unrehearsed birth family back? | 9 reminders, 6 references, 1,024 updates each | 0.6 h | 5 |
   | C | Does a learner that has lived need fewer items for a new family than a fresh one? | 6 short births | 0.9 h | 6 |
   | B | What erases the unrehearsed family, and does an anchor protect it? | 10 lives of two blocks, one seed | 4.7 h | 7 |

2. **Let E35 run through stage 1b as released, and brief nothing after it** (section 8).
3. **From now on, score every earlier family at every stage of a life**, with add and sub split by wrap (section 9).
4. **Enter in the record** Mr. Xu's three statements of today (section 2) and the fact that the E26 and E31 lives never rehearsed the birth family and lost it (sections 3.3 and 3.4). I searched the review from 9.111 onward and found no discussion of it.

5. **Release E34 stage 2 as its brief says, with two additions** (section 14): one backprop birth with the same gain at tau 0.5 on seed 73129, and one reported fact, the row at which each arm first learns in range.

6. **Apply the audit's six required changes to E36, E37 and E38 as written and release them** (`AUDIT_E36_E37_E38_BRIEFS_20261115.md`). No re-audit is needed after them.
7. **Brief the long-life programme of Part II**: one protocol brief (job L0, section 17.7), audited once, and the two long births of E34 (section 18), which need no new code.

8. **Brief one more full-scale job, E37 at a tenth of the pairs** (section 19.4): lived and fresh learners on a new family with a tenth of its pairs taught. About 4 CPU-hours by my estimate. It comes to me before any worker starts.

9. **Brief one more full-scale job, E41: computation to threshold by width and by taught fraction** (section 21.3). About 28 CPU-hours by my estimate, to be divided among the three workers that E36, E37 and E38 have freed. It comes to me before any worker starts.

10. **Brief one more full-scale job, E42: cells with a limited number of local connections and their own periods** (section 22). Forty-eight births in two stages after a stage of checks, 50 to 70 CPU-hours by my estimate. It comes to me before any worker starts. It does not wait for the small world and displaces nothing released.

11. *(Done, revision 13.)* **Ask the E40 worker for four more files** (section 23.4): the row-64 weights of the lived S and B runs of seeds 93683 and 116089, whose hashes are in the receipts. About 18 MB. They let me repeat section 23 on three seeds.

12. **Brief one small job, E43: is a family lost when the learner then learns something unrelated?** (section 24.5). About 2 CPU-hours by my estimate. It comes to me before any worker starts.

13. **Three additions to E43** (section 24.7): push the weight files its report lists, which are not on the remote; a reminder test from the networks at update 4,096; one arm under plain gradient steps. About 2 CPU-hours.

14. **Brief the replacement of the small world's task set** (section 25): families in the full-scale format, a learner born on one of them, and a calibration first. It comes to me before any worker starts.

15. *(Done; it is the coordinator's E47.)* **Write the specification of the job with a local interface** (section 26), as runnable code, for audit. It runs after E42's other two seeds and E43's additions.

16. **Brief one small job, E48: a new family learned only in connections gated by its own code** (section 27; called E47 in revision 19). About 1.5 CPU-hours. It comes to me before any worker starts.

17. **Two further additions to E43** (section 24.8, end): the `unrelated` arm with the first family rehearsed, as the conventional reference; and the `unrelated` arm started from E42's `F10` and `P8` networks of seed 73129. About 2.5 CPU-hours together.

18. **Enter the nine changes of the audit of the roster note for the first long lives and let the lives start** (28.8; `AUDIT_E44_L2_ROSTER_NOTE_20261117.md`, section 8). No second audit of the note. I read the changed runner and its check receipts when they are committed; that need not hold the trial.

19. **Do not release E47's roster, and do not continue E49** (28.4, 28.5, 30.2).

20. **Enter the nine changes of the audit of the E50 brief and release its stage 1** (30.2; `AUDIT_E50_SPARSE_INPUT_CODE_BRIEF_20261117.md`, section 9). No second audit.

21. **Brief one job, E51: a new task learned in cells held in reserve, with what the first family uses left unchanged** (30.3). Twelve runs on one seed first, about 2.5 CPU-hours. It comes to me before any worker starts. `e51_design_checks.py` holds the construction and the rule in numpy, for your specification to be compared with.

22. **Housekeeping.** Merge E41's three-seed report (2722d1c1) and E46 (b76f5979) into the default branch, since the roster note cites them. Have E43's report mark the two `P8` weight files of its second addendum as incomplete (28.6).

23. **Enter in the record** the corrected reading of E42 on three seeds (28.3), that the sweep does not learn with a limited input (28.5), and the measurement of section 29.

24. **On your return, read `COORDINATION_20261117/LOG.md` and `COORDINATION_20261117/PLAN.md`** (section 31.5). The log records what the reviewer decided and started in your place while Mr. Xu was away; the plan is the ordered list of what remains. Anything in either can be changed by you.

25. **Rule on E38's checks 2 and 3 under the sweep** (section 32.4). They failed and no ruling is on file. The reviewer's reading is that the checks were specified for a rule that is continuous in the weights, that the results under the sweep stand, and that no life needs to be run again.

26. **Take over from section 33.7.** It lists the seven worker sessions and what each is doing, every decision that the reviewer took in your place since revision 23, and what waits for your decision, in the order the reviewer recommends. From the handover on the reviewer takes no decision in your place; it goes on verifying results from receipts and weights and logging the workers' messages unless you arrange otherwise.

27. **Requirement A is not met by E51's trial** (section 33.1). The first family's answers were kept exactly while a new task was learned, but by two lines of the program that read the item. E55's first part asks whether the two tasks can be kept apart without those lines; its brief was checked and is not released, because the check's emulation shows that its construction keeps the first family's score and not its answers exactly (sections 33.5 and 33.7).

28. **Two matters that E52 raises are yours to decide** (section 33.17). By the brief's rule the other two seeds are due at the half width with both fractions and at the present width with a quarter of the pairs. The brief's condition for asking again for the long lives at the rate 0.003 is met on the one seed run; the present runner does not admit a birth made at another rate to a life, so such lives need a decision on the births. Nothing is started.

29. **Before the sweep's long lives are quoted, read the limits in section 33.16.** Each arm is a birth together with its rule, and the sweep's births answer less; the one reading that excludes zero treats worlds 1 and 4, which share their births, as two replications; every arm is still learning at 1,024 visits. On Mr. Xu's question of overtaking the verified answer is "not seen up to 1,024 visits, at one rate, between two learners that differ in their births", and what a firmer answer would need is listed there for your decision.

30. **By its default scheduling the host delivers a message to a worker only when the worker's turn ends** (sections 33.18 and 33.20). Three workers received messages seven to eleven hours late. The message tool has a priority that joins a running turn; the reviewer did not use it before 18 November, 14:47 UTC, and uses it from then on. A brief still has to carry everything a worker needs, and the reviewer now commits the text of every instruction it sends.

31. **The lines after the forwarding of 14:15 UTC:** the first long lives, E55 (E51's worker), E53 (E58's worker) and E57 (E59's worker) are running or starting; the helper's, E52's and E54's lines are free; E54's report is in verification. Your decision 15 (a note of at most one page after the arm `blind` is reported) and the verification of `blind` before `shut` is read are the reviewer's next duties on E55.

32. **Mr. Xu answered the two questions of section 33.21 on 18 November** (section 33.22). A code that the learner has learned by itself is not information given by us. A single global backward pass is prohibited; information may be relayed between neurons over their own connections, so that a few passes can cover the network. Section 33.22 gives his words, the reviewer's reading, and what follows for the running jobs and for the plan's items 20 and 23; nothing is started on it.

33. **A review at Mr. Xu's request: the experiments set against the principle** (section 33.23). On the line of the local learner as Mr. Xu defines it, the first step is taken, the second is not passed, and the steps on memory and on a life have not been begun on a network inside the definition; most of the computation of 17 and 18 November went to the dense network. The section gives the steps, where each stands, and the order the reviewer proposes. It is a proposal; the order of work is yours.

34. **Mr. Xu states that he never defined the sweep, and asks where it came from and why it is the rule under test** (section 33.24). The record's answer is that it was introduced on 6 November as a variant that the constraint as then read permitted, was proposed as the working rule by the reviewer on 10 November, and was adopted by you and confirmed by Mr. Xu in one sentence. Its standing against the principle of 33.22 is now open and is his and yours to rule; every running job uses it as the candidate.

35. **Mr. Xu gave four instructions after your decisions 30 to 37 were written; two briefs are put to you** (section 33.28). On the point that decision 31 leaves to be run, the reviewer reads his answer as placing the exact gradient outside what a neuron has; this is a reading, and the briefs keep decision 31's terms. The number of connections of a cell is to be varied on a grid of powers of two until a cell can set it itself. The chain of steps is `STEPS_FROM_THE_PRINCIPLE_20261118.md`. E61 is the brief of decision 33; its network is the setting `C46-one`, since stage 3 of E50 does not have 16 inputs a cell. E62 (the grid of the number of connections) is his instruction. Both are checked, at revision 2, and await your release; two lines are free.

36. **The first long lives are verified and closed** (section 33.25). The conventional learner is still learning at 1,024 visits and learns new families faster as it ages; a store raises what is kept and not, distinguishably, the level. Several readings of the arms without decay do not survive counting each birth once; the worker has corrected its report.

37. **E55's arm `blind` is verified on seed 73129, within decision 14's reading** (section 33.26). On items of 50 positions the first family is lost in one run and nothing bounds the reserve in any. The note of decision 15 is written (one page); it advises against a brief for the second part of item 11 as formulated and names what `shut` is to be read for.

38. **Forgetting at a return, against the length of the absence** (section 33.27, read-only). Without a store and with decay it is a step: one visit to another family takes what is answered from about 0.8 to about 0.02 under both rules. With the store about half is kept whatever the absence. Without decay back-propagation shows a graded curve over some tens of visits.

39. **E52 is verified on three seeds** (section 33.29; your decision 19). At each rule's better rate of 0.001 and 0.003 the sweep's lag is nowhere larger; at the present width with a quarter of the pairs it is gone on two seeds and smaller on the third, on margins of one to four scored rows; at the half width with a quarter of the pairs the seeds differ and no statement over them is drawn. The rate of E61 and E62 is the record's 0.001 under both rules; a pair of runs at 0.003 on that network (about 4 CPU-hours) would say whether the sweep's lag there is partly the rate, and is in neither brief.

40. **E55 is verified whole** (sections 33.26 and 33.30). In the arm `shut` the first family is partly kept in all four runs, so that decision 14's statement is met by none; the gates of the reserve stay shut on the first family and open on the new task; back-propagation learns the task on the first family's own bytes (92.9 against 34.4 with the reserve held shut) with 7 of 31,684 answers changed at the end; under the sweep neither task is learned. Requirement A is not met. The three figures of the note of decision 15 are in section 33.30; the note's advice stands. The message priority that joins a running turn did not deliver into a running turn (decision 29: observed, negative).

41. **Your decisions 38 to 47 are carried out but for E61** (section 33.31). E62 went to E54's worker at 20:21 UTC. With the relay of your decisions Mr. Xu told the reviewer to handle the matters other than E61, so that revision 3 of E61 is not written and E52's worker has been sent nothing; decision 39 is carried out when he says so. The chain of steps is at revision 3 with decisions 38, 40, 42, 43 and 44 entered.

42. **A second review found structures that every experiment shares and that were never held against the principle** (section 33.32; `REVIEW_OF_WHAT_THE_PROGRAM_FIXES_AGAINST_THE_PRINCIPLE_20261118.md`). The heaviest: the cells exchange what they hold once for each byte and the answer is due one byte after the last digit, so that relay has no time; and nine tenths of the weights of the network called sparse are a dense readout that reads every cell and the current byte directly. The outside audit, which arrived at 20:50, finds E61 not a test of relay between cells. The reviewer proposes a job that gives relay time before E61 in either form, and asks you to weigh decisions 39, 40 and 44 against it.

E35 continues as released. E33 is complete and needs nothing further.

## 2. Mr. Xu's statements of 15 November

Verbatim, from the session with the reviewer.

- **On the goal.** "Note our goal is persistent learning, not conventional training+validation+test data set setting. So the main problem is given fixed memory / computation capacity, how does local learning beat naive back-propagation in generalization tasks."
- **On the line.** "I repeatedly said as long as we do not do full backpropagation, anything is okay."
- **On structure.** "So now given the circuit structure itself is shaped by learning in our setting, the literature is only of limited help. Our neural network structure would change with time."

My reading of the third, which the measurements below support: the thing to study is the structure the learner builds for itself and what later learning does to it. A hand-wired circuit is at most a bound on that.

## 3. Findings

### 3.1 Every trained network holds a circular code of the answer, the same under both rules

Take the hidden state at `=` for every operand pair of a task, no letters, and average it per answer value. Call the 89 (or 97) averages the answer code.

| Networks | Task family | Own exact answers | Two strongest Fourier frequencies of the answer code | Their share of the code |
|---|---|---|---|---|
| 6 births (S and B) | add, sub mod 89 | 93–100 | 1 and 9, in all 12 readings | 0.85–0.95 |
| 9 lives, at the end | add, sub mod 89 (birth family, not rehearsed) | 1–47 | 1 and 9, in all 18 | 0.83–0.98 |
| 9 lives, at the end | add, sub mod 97 (taught in the life) | 88–99 | 1 and 10, in all 18 | 0.85–0.97 |
| Untrained, 2 draws | add mod 89 | — | — | no answer code: 0.000 of the state's variance lies between answer values |

- Frequency 1 is a circle over the size of the answer. Frequency 9 (mod 89) or 10 (mod 97) is a circle over its units digit: 89/10 and 97/10 cycles.
- The geometry is circular, not ordered: distances between codes are explained by the circular difference of the answers about as well as by the plain difference (largest gap 0.06). For max mod 89 the plain difference explains far more (0.77–0.92 against 0.56–0.64, seed 73129), as an ordered code should.
- The code has the same form for both moduli. That is nearer to inventory 乙 of E35 than to inventory 甲's table for one modulus.

```
python3 -I $R/learned_structure.py add89,sub89,add97,sub97 e26/e26_dev_S_73129.pt e26/e26_life_S_73129.pt e31/e31_life_K_73129.pt e26/e26_dev_B_73129.pt e26/e26_life_B_73129.pt
python3 -I $R/learned_structure.py max89 e26/e26_dev_S_73129.pt e26/e26_dev_B_73129.pt e26/e26_life_S_73129.pt e26/e26_life_B_73129.pt e31/e31_life_K_73129.pt
python3 -I $R/learned_structure.py add89,add97 untrained
```

### 3.2 A life rebuilds the weights and keeps the form of the code

Seed 73129. "Graft" means the life-end network with one group of weights put back to its birth value.

| | add 89 | sub 89 | add 97 | sub 97 | Change of the group over the life (norm of difference / norm at birth) |
|---|---|---|---|---|---|
| Life S, as saved | 23.6 | 40.3 | 95.7 | 90.1 | |
| with the birth's input layers | 9.4 | 16.8 | 2.5 | 3.8 | 0.27 |
| with the birth's recurrent weights | 4.4 | 1.6 | 1.0 | 7.2 | 1.17 |
| with the birth's read-out | 12.8 | 21.3 | 53.0 | 43.6 | 0.96 |
| Life B, as saved | 19.6 | 46.9 | 95.6 | 93.5 | |
| with the birth's recurrent weights | 35.9 | 30.3 | 7.9 | 15.9 | 1.12 |

- The recurrent weights move by more than their own size at birth. No single group put back restores the old family.
- So what persists through a life is the form of the code (3.1), not the weights that carry it.

```
python3 -I $R/graft_birth_into_life.py add89,sub89,add97,sub97 e26/e26_dev_S_73129.pt e26/e26_life_S_73129.pt e31/e31_life_K_73129.pt
python3 -I $R/graft_birth_into_life.py add89,sub89,add97,sub97 e26/e26_dev_B_73129.pt e26/e26_life_B_73129.pt
```

### 3.3 What a life removes is the old family's modulus

The E26 and E31 lives draw from families F1 (mod 97), F2 (compositions) and F3 (mod 73). The birth family (mod 89) is never in the stream (`git show $D:$M/exp26_candidate.py | sed -n '214p;242,244p'`).

Whole answers (two digits and the end symbol, as `exp16_integrated.score` counts them) at the end of each life on the birth family, all pairs, no letters. "Wrap" pairs are those with a + b ≥ 89, or a − b < 0. The last column is read from the two digits alone. Revision 2 of this memo counted the two digits throughout; six cells moved, the largest from 17.6 to 3.9 (K 116089, sub).

| Life | add: no wrap | add: wrap | sub: no wrap | sub: wrap | Wrap pairs answered with a life modulus |
|---|---|---|---|---|---|
| S 73129 | 45.5 | 1.3 | 79.3 | 0.4 | add 13% mod 97; sub 12% mod 73 |
| S 93683 | 54.0 | 0.3 | 12.1 | 0.4 | add 32% mod 97 |
| S 116089 | 25.8 | 1.3 | 2.5 | 0.1 | add 13% mod 73 |
| B 73129 | 37.8 | 1.0 | 92.8 | 0.0 | **sub 89% mod 97**; add 20% mod 97 |
| B 93683 | 25.8 | 0.1 | 39.8 | 0.1 | sub 32% mod 73 |
| B 116089 | 49.3 | 0.1 | 91.6 | 0.1 | **sub 83% mod 97**; add 26% mod 73 |
| K 73129 | 56.4 | 0.7 | 73.8 | 0.1 | **sub 59% mod 73** |
| K 93683 | 43.0 | 0.6 | 72.9 | 0.2 | add 30% mod 97 |
| K 116089 | 62.9 | 0.6 | 3.9 | 0.3 | add 41% mod 73 |
| Six births | 94.0–99.8 | 96.4–99.5 | 95.2–99.6 | 91.5–99.6 | 0.0–0.1% |

- Pairs that need the wrap at 89 are right 0.0–1.3% of the time in all 18 cells. The network still answers with two digits (98–100%).
- Pairs that need no wrap are right 2.5–93%. The shared arithmetic survives in part.
- In several lives the old prompts are answered with a modulus the life taught.
- Sweep, Kolen–Pollack and backprop lives show the same pattern.

```
python3 -I $R/e36_e38_brief_checks.py H e26/e26_life_S_73129.pt e26/e26_life_B_73129.pt e31/e31_life_K_73129.pt      # whole answers
python3 -I $R/life_answers_on_birth_tasks.py e26/e26_dev_S_73129.pt e26/e26_life_S_73129.pt e26/e26_life_B_73129.pt e31/e31_life_K_73129.pt   # two digits; the last column
```

### 3.4 The loss happens in the first block (receipts)

Held-out add and sub of the birth family at 12 letters, mean of the two, from the life receipts' own columns:

| Arm | At the start | After block 1 | At the end |
|---|---|---|---|
| S | 98 · 98 · 100 | 49 · 7 · 34 | 33 · 18 · 14 |
| B | 98 · 99 · 94 | 44 · 1 · 33 | 37 · 27 · 36 |
| K | 98 · 98 · 100 | 46 · 5 · 28 | 31 · 36 · 30 |

Max mod 89 swings instead of falling (seed 93683, arm S: 98, 19, 54, 84, 91, 74, 28).

### 3.5 Later families reach high held-out scores from a quarter of the pairs in few items (receipts)

Same score, by items of that family seen so far:

| Family | Pairs taught | Items seen | Held-out add/sub at 12 letters |
|---|---|---|---|
| Birth, mod 89, from nothing | half | 262,000 | 63–84 (six births) |
| | | 524,000 | 88–95 |
| Life F1, mod 97 | a quarter | 393,000 | 91–98 (nine lives) |
| Life F3, mod 73 | a quarter | 131,000 | 77–94 |
| | | 262,000 | 80–95 |

This is not yet a saving: no fresh network has been taught a quarter of the pairs of one family. Job C supplies that. The same pattern was seen once before on the window network, where developed births beat backprop from nothing by 20–40 points at a tenth of the pairs (9.4, 9.6).

```
python3 -I $R/acquisition_from_receipts.py
```

### 3.6 A likely cause in our own settings (arithmetic, not a measurement)

AdamW at learning rate 0.001 with decay 0.25 multiplies every weight outside the embedding table by 0.99975 per update (`git show $D:$M/exp16_integrated.py | sed -n '176,187p'`). A component of the weights that the current stream does not push on is left with 21% of its size after one block of 6,144 updates and 0.01% after a six-block life. The embedding rows of the old code bytes have no decay and no gradient, so they stay; everything they feed is shared and decays or is rewritten. Job B separates decay from rewriting.

**Refuted (revision 6).** E38 shows the same loss with no decay at all: the wrap pairs fall from 97% to 18% within 64 updates and to 3% within 256, with decay toward zero, without decay, and with either anchor (section 19.2). The new family's own updates erase the old modulus, about fifty times faster than decay could.

## 4. Reading

- **The rule has again made no difference.** Both rules build the same code, lose the same thing in the same block, and acquire later families alike. Every job below runs both and reports the paired difference.
- **Retention in this learner is a small problem with a large symptom.** What is lost is one parameter per family, the modulus. The symptom is that the whole family scores 1–47%.
- **Where a claim against naive backprop could come from.** Naive backprop keeps nothing of an unrehearsed family's modulus (0.0–1.0% on wrap pairs in the three backprop lives). Rehearsal keeps it and takes a share of every batch (9.21, 9.64). A learner that keeps it with no rehearsal, at the price of one extra number per weight, would do at equal updates what naive backprop does not. Job B tests the simplest such device. The same device could be added to backprop, so the claim would be against naive backprop only, which is how Mr. Xu put the question.
- **A second place, since E34.** With every byte a target, naive backprop ends at 82.0, 82.8 and 85.4 at 40 letters. The sweep with E34's confidence gain at tau 0.5 ends at 79.8 on the one seed run, with a final row of 99.3 (section 14). If stage 2 puts it above backprop on the new seeds, that is a claim against naive backprop at no extra memory. The backprop birth with the same gain says whether it belongs to the rule or to the objective.
- **The form the claim should take:** at equal stored numbers and equal updates, with every earlier family scored at every stage, and with both the memory and the rehearsal share of each arm stated.

## 5. Job A: the reminder test

**Question.** After a life that never rehearsed the birth family, how many updates on that family bring it back, against learning it from nothing by the same rule?

- **Start (9 files).** `e26/e26_life_S_<seed>.pt`, `e26/e26_life_B_<seed>.pt`, and `e31/e31_life_K_<seed>.pt` with its `e31/e31_life_K_<seed>_Bhh.pt`; three seeds.
- **Update.** Each by the rule of its own life. Arm K continues from the saved feedback matrix with its after-step update; `exp31_switch.py` builds that matrix from the seed, so loading it is new code. A fresh optimizer from `make_opt`, since the lives saved no optimizer state that I can find.
- **Data.** The birth's own stream: the four mod-89 operations, the taught half of the pairs of that seed, 0 to 12 letters, batches of 64. Use a stream seed other than the birth's, the same for every arm of a seed.
- **Dose and rows.** 1,024 updates. Save weights at 0, 16, 32, 64, 128, 256, 512 and 1,024. At 256 updates each taught pair has been shown about once (4,096 items per operation over 3,960 taught pairs).
- **Reference (6 runs).** A fresh network with E26's initial values for the seed, taught the same batches by the sweep and by backprop, same rows. Arm K is read against the sweep reference.
- **Scores at every row.**
  - the birth family's held-out pairs at 12 letters, on the panels the life receipts use, per operation;
  - add and sub on all pairs without letters, split by wrap;
  - the life's families at 12 letters and `compose.within`, to show what the reminder costs them.
- **Marks (corrected in revision 3; see the audit, 4.1).** "Restored" at a row: mean of held-out add and sub at 12 letters at least 85 at that row. "Wrap score" at a row: the mean of the add and sub wrap-pair scores on all pairs without letters. Per arm, in two of three seeds; the rows are tried in order and the first that holds is the reading. My first version read rows 2 and 3 on the held-out mean, which a reminder already meets at row 0 (it starts at 13 to 37, because everything but the wrap is kept):

  | # | Pattern | Reading |
  |---|---|---|
  | 1 | Restored at row 256 | The family's rule was still in the network. A life costs the old family about one showing of each taught pair. |
  | 2 | At row 1,024 the wrap score is at least 30 points above the reference's | Partial savings. |
  | 3 | At row 1,024 the wrap score is less than 15 points from the reference's | No savings: what the life removed is relearned no faster than a fresh network learns it. |
  | 4 | Anything else | Mixed; the curves are reported. |

- **Checks.** Row 0 has the life-end file's tensor hash and reproduces the last stage's birth-family scores in the life receipt. Batch digests are equal across the arms of a seed. For arm K, the loaded matrix has the saved file's hash and E31's identity check holds on the first steps. One dry run of 64 updates with CPU time.
- **What it cannot show.** The optimizer state is fresh in both reminder and reference. The reminder is given alone, not inside a life's stream.

## 6. Job C: a from-scratch reference for a life family

**Question.** Is 3.5 a saving? Does a learner that has lived reach a held-out score on a new family from fewer items than a fresh learner under the same rule?

- **Runs (6).** A fresh network with E26's initial values, sweep and backprop, three seeds, taught family F3 alone at a quarter of the pairs: `X16.family_entries("F3", seed)` through `X16.entry_stream(entries, seed + 105)`. That is the life's own F3 stream (`seed + 100 + si` with `si = 5`), so the fresh network sees the same items in the same order as the life did.
- **Dose and rows.** 4,096 updates; weights at 1,024, 2,048 and 4,096. Row 2,048 is 131,072 items, against 130,000–131,000 F3 items in the lives after block 5.
- **Score.** The life receipts' F3 panels at 12 letters; mean of held-out add and sub.
- **Read.** Life minus fresh at equal items of the family: the life's score after block 5 against row 2,048, and after block 6 against row 4,096; S life against sweep fresh, B life against backprop fresh, K life against sweep fresh. "Saving": at least 15 points in two of three seeds. "None": within 15. "Cost": at least 15 the other way.
- **Optional second half.** The same for F1 (`seed + 101`, 6,144 updates, read against the lives after block 1). About 1.4 CPU-hours more.
- **Cross-check, free.** E35's arm `G` at f = 0.25 is a fresh GRU on a quarter of the pairs of the mod-89 family; its curve at 4,096 updates belongs in the same table.
- **What it cannot show.** The lived learner had also seen about 1.8 million items of other families in its life and 1.6 million in its birth, so the saving is in items of the new family, not in total experience. The fresh network gives every update to F3; the life gave a third. That favours the fresh network per update and is neutral per item.

## 7. Job B: what erases the unrehearsed family, and does an anchor protect it?

**Question.** Is the loss in the first block made by decay toward zero, by the new family's own updates, or by both? Does pulling each weight toward its birth value keep the old modulus while the new family is still acquired?

- **Start.** The E26 births, arms S and B. Stage 1 on seed 73129.
- **Life.** E26's first two blocks only (F1.1 and F1.2, 12,288 updates), same entries and stream seeds as E26, so that the first arm below repeats E26.
- **Arms, per rule (10 lives).**

  | Arm | One difference from E26's life | Extra memory | Rehearsal share |
  |---|---|---|---|
  | `zero` | none: decay 0.25 toward zero | none | 0 |
  | `none` | no decay during the life | none | 0 |
  | `anchor` | decay 0.25 toward the birth value of each weight | one number per decayed weight (1.12 million; 4.5 MB) | 0 |
  | `anchor10` | the same at strength 2.5 | the same | 0 |
  | `rehearse` | decay as `zero`; the birth family's taught entries join the joint stream from block 1 as one more family | none | half of every batch |

- **The anchor, as code (my proposal; the specification is yours).** Set the optimizer's `weight_decay` to 0 for the group that had 0.25, and before each `opt.step()`:

  ```
  for p, p0 in zip(decayed_params, birth_values):
      p.sub_(LR * KAPPA * (p - p0))          # KAPPA = 0.25 (anchor) or 2.5 (anchor10); LR = 0.001
  ```

  torch's AdamW shrinks before the Adam step, so the pull goes before the step too. Each weight uses only its own two values, so the device is inside Mr. Xu's line.
- **Rows.** Weights at 0, 64, 128, 256, 512, 1,024, then every 1,024 to 12,288. The early rows separate the two causes: decay acts over about 4,000 updates, rewriting can act within hundreds.
- **Scores at every row.** As Job A's first two bullets for the birth family; held-out add and sub of F1 at 12 letters.
- **Marks (my proposal), after block 2.**
  - "Keeps the birth family": held-out add/sub mean at 12 letters at least 80, and wrap pairs at least 70.
  - "Loses it" (corrected in revision 3; see the audit, 4.3): wrap pairs, mean of add and sub, at most 20. My first version, the held-out mean at most 50, sat exactly on the case of interest: a life that loses the wrap and nothing else scores 48 to 50.5 on seed 73129's panel.
  - "Acquires the new family": F1 held-out add/sub mean at 12 letters at least 85 (E26: 92–97).
- **Readings, per rule.**

  | Result | Reading |
  |---|---|
  | `zero` does not repeat E26's stage 1 and 2 scores and batch digests | The runner is wrong. Stop. |
  | `none` keeps | Decay toward zero is what erased the family. |
  | `none` loses | The new family's own updates erase it; decay is not needed for the loss. |
  | `anchor` or `anchor10` keeps and acquires | A candidate: retention with no rehearsal for one stored number per weight. Stage 2. |
  | An anchor keeps and does not acquire | Too stiff at that strength; report both strengths side by side. |
  | Neither anchor keeps | A uniform pull is not enough. The next device would weight the pull by each weight's use for the old family, or give families separate resources. Not briefed now. |
  | `rehearse` | The reference: what rehearsal keeps, and what it costs F1 per update and per item. |

- **Checks.** `zero` repeats E26's stage 1 and 2 (digests, losses, scores). `anchor` with `birth_values` set to zero agrees with `zero` after 64 updates (largest absolute difference at most 1e-5 of the tensor's largest entry). In `rehearse`, the count of birth-family items per block is recorded. Two independent implementations of the anchor step, since it is new code.
- **Stage 2 (released by you after stage 1 is read).** Arms that keep and acquire, with `zero`, on the other two seeds; then one full six-block life per rule, to see composition and length under the anchor. Once the small world of Part II exists, further strengths and the moving anchor are searched there and only the winner returns to full scale.
- **What it cannot show.** A fixed anchor suits a two-block screen only. Over a long life the anchor has to move, for example a slow copy of each weight; that is designed only if the screen passes. One seed in stage 1.

## 8. E35: scope

- Stages 0, 1a and 1b run as released, stop rule included. E35 now serves two purposes: the bound on what a perfect, never-forgotten circuit gives, and (arm `G`) a from-scratch reference at a quarter of the pairs.
- Nothing after stage 1b is briefed. The stage I had sketched, an admissible rule learning the interface into a hand-wired module, answers a question the programme no longer has: the learner makes its own circuit (3.1).
- Taught fractions below 0.10 are worth adding only if row 1 occurs. If they are added, score held-out pairs in two groups (both operands seen in that position in that task, or not): at f = 0.03, 1–10 of the 89 values never appear in a given position of a task; at 0.05, 0–2 (`python3 -I $R/e35_low_fraction_coverage.py`).

## 9. Standing readings

For every report that contains a life or a sequence of families:

1. Each earlier family's held-out score at every stage, and for add and sub the split by wrap on all pairs without letters. `life_answers_on_birth_tasks.py` is the independent numpy implementation; the worker's reader is the torch one.
2. Beside any online score, a blind predictor that answers from recent history without reading the operands, and a second score for each family after the stream has left it. Both come from the second literature report (section 10).
3. For each arm, the stored numbers beyond the weights and the rehearsal share of each batch.

## 10. What the two literature reports add

Mr. Xu commissioned two literature reports today. They are not in the repository. What bears on this memo:

- No rule that avoids end-to-end backpropagation beats a tuned backprop baseline on held-out compositional or length tests in a recurrent network. The most careful recurrent comparison (Marschall et al., 2020) is negative.
- The large out-of-range effects in the literature come from fixed structure, and every interface into it was hand-coded or backpropagated. Local rules have learned such an interface only where the fixed module's own state was the teacher, and never with a test on untaught combinations.
- Under matched compute, plain replay is the baseline to beat in online continual learning. All of that evidence is vision with backprop; no budget-matched comparison exists on sequence streams or with a learner that avoids backprop.

The reports' sources were read through a text extractor and I opened none of them. Nothing in sections 3 to 9 rests on them except item 2 of section 9.

## 11. Withdrawn from my notes of this afternoon

If Mr. Xu relayed them:

- the two-candidate plan for a later stage of E35 (the sweep through the module; a teacher-driven rule with a fixed read-out);
- the stream experiment with the module arm against the GRU with and without a replay store. Job B is that experiment in its first form, with the learner's own structure in place of the module.

## 12. My expectations, on record before any run

| | Expectation | Chance |
|---|---|---|
| A | Restored by row 256 in two of three seeds, sweep lives | 35% |
| A | Restored by row 1,024 in two of three seeds, sweep lives | 65% |
| A | Wrap score at least 30 above the reference's at row 1,024 (restated in revision 3) | 70% |
| C | A saving of at least 15 at 131,000 items, under each rule | 85% |
| C | A saving of at least 15 at 262,000 items | 60% |
| B | `none` keeps the birth family | 20% |
| B | `anchor` keeps it | 30% |
| B | `anchor10` keeps it | 60% |
| B | Some anchor keeps and acquires | 40% |
| any | A reading that differs between sweep and backprop | 15% |
| E34 | T050 learns in range and extrapolates on both new seeds | 40% |
| E34 | T050 does so on at least one new seed | 75% |
| E34 | Backprop with the gain at tau 0.5 is below 70 at 12 letters at row 8,192 (plain backprop: 89.3) | 60% |

## 13. Limits

- Three seeds, one pass over saved files, lives designed for another purpose.
- 3.1 describes the form of the code and is blind to whether it holds the right answer; 3.3 covers that.
- 3.2 is one seed.
- 3.3 scores all pairs with no letters; 3.4 is the receipts' held-out panel at 12 letters. The two agree in direction and are not the same number. Since revision 3, 3.3 counts whole answers; its last column and the script `life_answers_on_birth_tasks.py` count the two digits.
- A linear read-out of the answer from the state at `=` was too weak an instrument (10–37% where the network itself is 96–99% right), so nothing here uses it.
- 3.6 is arithmetic. No run has yet separated decay from rewriting.
- CPU figures are scaled from E26's receipts (557 CPU-seconds per 4,096 birth updates; about 840 per life block with scoring) and are mine.

## 14. E33 addendum 1 and E34 stage 1, read after the first version of this memo

Both are on `claude/vibrant-curie-ivmrt1` at ba6f5d52 and not yet on the default branch.

### 14.1 The check

My numpy reader reproduces every endpoint the two reports give, from the committed final files and rows 16,384 and 20,480 (seven births, 21 files). Rows before 16,384 are not committed and are taken from the worker's tables.

| Birth | Endpoint at 12 / 20 / 40 letters, mine | The report's |
|---|---|---|
| E34 T050, 73129 | 83.67 / 82.56 / 79.78 | 83.66 / 82.56 / 79.78 |
| E34 T100, 73129 | 95.33 / 90.11 / 57.56 | the same |
| E34 T200, 73129 | 86.11 / 31.56 / 2.00 | 86.11 / 31.55 / 2.00 |
| E33 A+D, 93683 | 86.00 / 76.67 / 29.11 | the same |
| E33 A+OS, 93683 | 97.44 / 97.11 / 94.89 | the same |
| E33 A+D, 116089 | 87.89 / 75.67 / 31.56 | the same |
| E33 A+OS, 116089 | 93.11 / 91.78 / 86.89 | the same |

```
python3 -I $R/e33a1_e34_check.py
```

### 14.2 What they settle

- **E33.** On three seeds, a sweep birth with answer and distractor targets does not extrapolate to 40 letters (23.8, 29.1, 31.6) and one with answer, operand and task-code targets does (94.2, 94.9, 86.9). The question E33 asked is answered.
- **E34.** On seed 73129 the confidence gain at tau 0.5 learns in range and extrapolates with every byte a target and no answer mask. Tau 1 is intermediate and tau 2 fails. Outcome row 1; a candidate on one seed.
- **Amount, not only kind.** E33 left open whether distractor targets hurt by their kind or by their weight. E34's three arms give a dose: a distractor position's weight relative to an answer position's ends at 0.001, 0.033 and 0.19, and accuracy at 40 letters at 79.8, 57.6 and 2.0. One seed.

### 14.3 What they change

1. **Nothing in jobs A, B and C.** They stay on answer targets and E26's files, one change at a time.
2. **E34 stage 2 runs as briefed** (T050 on seeds 93683 and 116089), with two additions.
   - **One backprop birth with the same gain at tau 0.5, seed 73129** (`local = False`, otherwise arm T050; about 0.8 CPU-hours). The first literature report calls this control essential, and I had made it conditional on a sweep arm passing the backprop reference. The final row now does (99.3 against 84.7) and the endpoint does not (79.8 against 82.0), so run it rather than argue which applies.
   - **Report the row at which each arm first reaches 70 at 12 letters**, beside plain backprop and the plain sweep. On seed 73129:

     | Arm | 4,096 | 8,192 | 12,288 | 16,384 | 20,480 | 24,576 |
     |---|---|---|---|---|---|---|
     | T050 | 2.3 | 6.3 | 17.0 | 78.3 | 73.3 | 99.3 |
     | T100 | 17.7 | 85.0 | 91.7 | 95.0 | 91.7 | 99.3 |
     | Backprop, every byte, no gain | 73.7 | 89.3 | 95.0 | 91.3 | 93.3 | 93.7 |
     | Sweep, every byte, no gain | 15.7 | 56.7 | 61.3 | 66.3 | 37.0 | 75.3 |

3. **The price of tau 0.5 is acquisition.** T050 needs about 16,000 updates to learn in range. Plain backprop is at 74 by 4,096 updates and 89 by 8,192, and tau 1 is at 85 by 8,192. An answer the network cannot yet predict has high entropy and so almost no weight; the gain holds back exactly what is new. In a life, where items needed for a new family are the currency (Job C), every new family would start that way. So the gain does not go into a life until stage 2 has reported and this cost has been measured there; the brief for a mask-free life has to carry a row-by-row acquisition score for each new family, not only an endpoint.
4. **The endpoint hides a swing.** T050's last three rows at 40 letters are 78.0, 62.0 and 99.3. Stage 2's sentence should give the three rows beside the mean for each seed.

### 14.4 What I did not check

- Rows before 16,384, the keeper counts, the weight-ratio tables and the checks in `E34_CHECKS.json`.
- Whether a temperature between 0.5 and 1, or one that falls during the birth, keeps the extrapolation without the delay. That is a new arm and I do not propose it before stage 2 reports.

---

# Part II. The programme from here: a continual learner judged over a long life

## 15. The audit of E36, E37 and E38, and two errors of mine

The audit is `AUDIT_E36_E37_E38_BRIEFS_20261115.md` in this folder; its checks are `e36_e38_brief_checks.py` (33dcd187).

- Every factual claim in the three briefs that could be tested holds, including the batch identity of a life's first block, which was rebuilt and hashed against the receipts of all nine lives.
- Six changes are required, all worded to be applied as written; none needs a re-audit.
- **Two of the six correct marks that I proposed in this memo**, not anything the coordinator did. Both failed for one reason, which section 3.3 should have told me: a life that loses the wrap and nothing else still scores about 50 on the held-out panel. So a reminder starts at 13 to 37 and my "30 above the reference" could be met with nothing relearned, and my "loses it at 50" sat on the case of interest. Both marks now read the wrap pairs.
- One required change is a real defect in the code path: the K runner on the default branch imports the other worker's `arch31`, whose `forward` has a different signature.

## 16. Mr. Xu's further statements of 15 November

Verbatim, with my translation.

- **On the nature of the problem.** "从哲学角度上来说，最朴素的反向传播只对每个样本进行同样的学习，通过误差进行逐级修正，网络结构是恒定的，虽然可能在RL或其他阶段只是激发网络的一部分。meta learning可以学习最优的某个网络结构，但一般也认为训练的objective是确定的才能做到。但在持续学习情况下，样本之间存在内在结构，学习本身会帮助调整网络结构，不断去自动优化学习样本之间的关系。作为代价静态的backpropagation无法大规模使用。因此我们回到一个全局元学习层次的优化问题。"
  - *Translation:* naive backpropagation applies the same learning to every sample and corrects by error, layer by layer, in a constant network, even if in reinforcement learning or at other stages only part of the network is active. Meta-learning can find a good structure, but is generally taken to need a fixed training objective. In continual learning the samples have structure among themselves, and learning itself helps adjust the network's structure, so that the learning of those relations keeps being optimised. The price is that static backpropagation cannot be used at scale. So we are back at an optimisation problem at the level of global meta-learning.
- **On direction and pace.** "你可以自己决定实验的快慢，不用等我决定。因为此前目标出现错位，我们先讨论下一步如何做。我们坚持做持续学习。"
  - *Translation:* you (the reviewer) may set the pace of the experiments yourself, without waiting for my decision. Because the goal was misaligned before, we first discuss how to proceed. We stay with continual learning.
- **On what the measure must be able to show.** "但是这个还是不能回答问题，因为显然可能出现我们的学习者一开始比传统的学习者更慢，但是在持续学习过程中最后做到“后来居上”的问题。只要我们的学习者还在学习，而传统的学习者卡住了，就不能排除这一点。一天上千个是个好办法，但无法处理这种情况。"
  - *Translation:* this still cannot answer the question, because our learner may be slower than the conventional learner at first and overtake it later in the course of continual learning. As long as ours is still learning and the conventional one is stuck, that cannot be excluded. A thousand lives a day is a good method, but it cannot handle this case.

The third statement was made against my first proposal (one score averaged over a life; many short lives). That proposal is withdrawn. Section 17 is what replaces it.

## 17. The programme

### 17.1 The question

In one stream read once, at fixed memory and fixed computation per item, without full backpropagation: is there a learner whose performance late in a long life exceeds that of naive backpropagation, and of backpropagation with replay, at the same budget? A slow start does not count against it.

What we measured until now could not answer this. It was accuracy on fixed panels after a fixed dose, with backpropagation as the ceiling, so the best outcome was parity. A score averaged over a short life cannot answer it either, for the reason Mr. Xu gives.

### 17.2 The world (requirements; the specification is yours)

1. **One stream per life, read once.** Items in the present format. Answer bytes are the targets, as in E26. Every-byte targets enter later as a candidate, after E34 has reported.
2. **Tasks.** A task is an operation and a modulus with a code of its own. The generator supplies more distinct tasks than the longest life meets for the first time, from one parametric space, so that tasks share structure. Codes are arbitrary; a code that names the modulus is a later variant.
3. **Visits.** The stream is a sequence of visits. A visit presents a small group of tasks for a fixed number of updates. New groups keep arriving through the whole life; old groups return after short, medium and long absences. The schedule comes from the world's seed and is the same for every arm.
4. **Taught and asked items.** Each task's operand pairs are split once. Taught pairs appear with their answer and the learner is updated on them. Asked pairs appear without their answer, at a fixed rate inside their task's visits; the learner's answer is scored and nothing is learned from them. An asked pair is never taught.
5. **Size.** The same kind of network, narrower; smaller moduli; fewer letters. Small enough that a life of 1,000 visits costs about 6 CPU-hours at most. I estimate that the present scale would need more than ten days for such a life; neither figure is measured.

### 17.3 What is recorded for every life

- **Score by age.** Accuracy on asked items, whole answers, in consecutive bins of the stream, with a blind predictor beside it.
- **Cost of the n-th new task.** The taught items of that task shown before its asked items first reach 80% over a running window; or "not reached", with the level at the end of the visit.
- **Kept at return.** Accuracy on the first asked items of a task when it returns, against the length of its absence, and for add and sub split by wrap; then the taught items needed to reach 80% again.
- **The ledger.** Stored numbers by kind (weights, optimizer state, anchors, buffers); updates; stream items consumed per update.

### 17.4 What is compared

- **Late level:** mean score over the last quarter of the life.
- **Trend:** the slope of score against the logarithm of age over the second half, and the slope of the logarithm of cost against the logarithm of n.
- Everything is paired by world, with intervals across worlds. No fixed thresholds in points are set before the calibration job shows the scale of the numbers; my two errors in section 15 came from setting such thresholds without looking.
- **"Stuck"** means that over the second half of the longest life run, neither the score's slope nor the cost's has an interval on the improving side of zero. **"Still learning"** means that at least one has.
- **"Overtakes"** means behind on late level at one length, ahead at a longer one, and still ahead at the next.

### 17.5 The horizon rule

- Lives are run at lengths that grow by a factor of four: 64, 256, 1,024 visits, and further where needed.
- A comparison of two learners is closed only when the sign of the late-level difference is the same at two successive lengths and the trend difference does not point the other way. Otherwise that pair goes to the next length.
- Every statement carries its horizon: "up to 1,024 visits". No finite run excludes overtaking beyond its horizon; what can be stated is the trend at it.
- When settings are chosen, a setting goes on to the next length if it leads on late level **or** on trend. This is what keeps a slow starter from being dropped.

### 17.6 Arms and budget

- All arms have the same network, the same world and the same number of updates, and every update consumes the same number of stream items.
- `naive`: backpropagation on the taught items of each batch.
- `replay`: backpropagation with a buffer of raw past items. A share of each batch is drawn from the buffer in place of new taught items, which are skipped. The buffer holds as much memory as one copy of the weights, which is what the anchor of E38 stores; its share of the batch is tuned.
- `sweep` and `sweep+replay`: the same under the sweep. Sections 3 and 4 found no difference between the rules at this level, so these two serve as the check that the small world agrees.
- Each arm's own learning rate and decay, and for replay its share and size, are chosen on tuning worlds. Comparisons are made on other worlds.
- **Candidates come after job L2.** They are the sweep with the three controls of section 4: which weights may move (the anchor of E38, then a slow copy of each weight, then a step size per weight); when to rehearse (short reminders, if E36 shows savings); which samples teach (the confidence gain, after E34). At first each control is a fixed number per life. The step in which a control depends on the learner's own state is the one Mr. Xu's first statement in section 16 describes, and it is last because it needs the others as its baseline.

### 17.7 Jobs, in order

| Job | What | Size | Audit |
|---|---|---|---|
| L0 | The protocol brief: world generator, runner, scorer, ledger, the recorded quantities of 17.3 | new code | once, by me |
| L1 | Calibration: 8 worlds, lives of 16 visits, `naive` and `sweep` | about 4 CPU-hours | none |
| L2 | Does the conventional learner get stuck? `naive` and `replay` at 64, 256 and 1,024 visits; `sweep` and `sweep+replay` at 64 and 256; `naive` at a second width; the probes and the agreement between lengths of 17.10 | cap of 48 CPU-hours | none |
| L3 | Candidates, chosen by 17.5 | set after L2 | none |
| L4 | The winning setting on the E26 families at full scale | as E38 | as now |

- **L0, second implementation.** I write the world generator and the scorer in numpy from the brief's text alone. Before any roster, the two implementations agree on the digests of the first batches of three worlds and on the scores of saved weights.
- **L1, gate.** Under `naive` and under `sweep`, the first group's asked items are at least 90% right by the end of its first visit in six of eight worlds, and the measured cost puts a 1,000-visit life within about 6 CPU-hours. If either fails, change the size and repeat. Reported and not gated: the wrap pairs at a task's first return, and the cost of tasks 2 to 16 beside the first.
- **L2, roster.** Eight worlds at the two shorter lengths and four at the longest. If the forecast passes the cap, drop worlds before lengths. No candidate takes part. The reading is 17.4's: stuck or still learning, for each arm.
- **Why L2 comes before any candidate.** Mr. Xu's condition for overtaking is that the conventional learner is stuck while ours still learns. Whether the conventional learner gets stuck in our world is not known. Our longest lives so far have six to nine stages. In the one study the second literature report cites on this, backpropagation fell from 89% to 77% over 2,000 tasks, and it took that length to see it.

### 17.8 Pace

Mr. Xu has left the pace to me. Inside the small world it is this:

- One audit, of L0. After it, a roster inside the protocol needs a one-page note (arms, worlds, lengths, predictions on file) and no audit.
- Any change to the generator, runner or scorer is audited, and so is any claim at full scale.
- At least eight worlds per setting up to 256 visits; paired differences with intervals, in place of marks on three seeds.
- I spot-check a sample of saved lives with the numpy scorer.

Full-scale work (E36 to E38, E34, L4) keeps the present discipline.

### 17.9 What can be concluded

- A result in the small world is a claim only after L4.
- If no candidate passes `replay` at any length and the trends do not differ, the result is that replay is the answer at this scale and that our learner equals it. That is reportable and is not the advance Mr. Xu asks for.
- If the conventional learner is not stuck at 1,024 visits and no candidate's trend is better, the statement is "no overtaking up to 1,024 visits and no trend toward it", not "no overtaking".

### 17.10 How short and long runs are combined (added in revision 4)

Short runs are many and varied; long runs are few and are the real measure. This is the long-delay problem of reinforcement learning, and the programme uses its standard answers: predict the long outcome from quantities measured early, learn the laws and compose them, and do not restart what can be continued.

1. **Short runs measure rate laws; long runs test whether the laws compose.** Three laws describe a learner at any age: what remains of a task after an absence of a given length; what a new task costs; what a reminder restores. E36, E37 and E38 measure these at full scale. Before a long life is run, its curves are predicted from the laws measured in short runs, the prediction is filed, and it is scored afterwards. Where the prediction fails is where a long-horizon effect lives.
2. **Trunks and branches.** A few long lives (trunks) save their complete state at ages that grow by a factor of two: weights, optimizer state, buffers and the position of the stream. From any saved state a short run (a branch) continues with one thing changed and the same future stream. Two branches from one state differ only by that change, so they are compared item by item. This gives many varied short experiments at old ages for the cost of one long life. The harness already has a qualified resume for the older runners (9.112 H); the new runner needs it from the start.
3. **Two probes, taken as branches at every saved age.**
   - *The learning probe:* a fixed set of tasks that no world's stream ever uses is taught for a short dose; the taught items needed to reach 80% are the learner's ability to learn at that age, on the same tasks at every age.
   - *The keeping probe:* asked items of tasks that have been absent for known lengths.
   - These are the candidates for leading indicators. L2 shows whether their values at 64 and 256 visits predict the late level at 1,024.
4. **Measure how far short runs can be trusted.** L2 reports the agreement between lengths: of the arms and settings run, how often the order at 64 visits is the order at 1,024, on late level, on trend and on each probe. Selection at short length (17.5) then uses whichever quantity agrees best. If none agrees, short runs cannot choose among learners in this world, and the budget moves to trunks and branches.
5. **Time in the learner's own units.** The brief lists the learner's own time constants and states visit and absence lengths as multiples of them. One is known: decay shrinks an unsupported weight with a constant of 1 / (learning rate × decay), 4,000 updates at the present settings. L2 runs `naive` at two widths, to test whether the age at which anything changes scales with capacity. If it does, a narrower network reaches old age sooner and long horizons become cheaper; if it does not, width is only a cost setting.
6. **What cannot be shortened.** A branch does not show an effect that needs a setting to have been in force since birth, or the avoidance of a state that cannot be undone. Every candidate that survives selection therefore gets full-length lives of its own; branches and probes decide which candidates those are.

### 17.11 What the audit of E39 changes in this Part (added in revision 5)

The audit is `AUDIT_E39_SMALL_WORLD_PROTOCOL_20261115.md`. Its second implementation of the generator reproduces the coordinator's three digests. It requires seven changes to the brief, the first of which is a real defect: the specification file has the start and end symbols the other way round from the harness.

Three of its findings change what sections 17.3 to 17.7 say:

1. **The trend that is compared is the first-visit level, not the cost.** Sections 17.3 and 17.4 name the cost of the n-th new task and its slope. In the brief's schedule a task met late has fewer visits left (groups first met in the four quarters of a 1,024-visit life get 9.3, 4.8, 3.5 and 2.3 visits), so among late tasks only the cheap ones reach the criterion. That pushes the cost's slope toward the improving side for any learner, which is the result L2 looks for. Every task has one whole first visit, so the level it reaches there is free of that bias. The cost and its slope are still reported and are not used in any comparison.
2. **The calibration gate is replaced, and calibration lives have 64 visits.** The coordinator's objection to my gate was right: "90% by the end of the first visit" asks a 256-update birth of a network with nothing behind it, and L2 does not need it. The gate is now that the first group reaches the criterion within the life, that the median first-visit level of later tasks is between 0.10 and 0.90, and the cost bound. The two bounds say that the instrument is off its floor and its ceiling; they were set before any data, as my two wrong marks were, and L1 will show whether they are sensible. Lives of 16 visits cannot show the second condition, since they meet one to five groups.
3. **The horizon is bounded by tasks as well as by compute.** With 250 usable groups and a fifth of visits new, a life runs out of new tasks at about 1,250 visits. A length beyond 1,024 visits needs a larger task space, which is a change to the generator and one more audit of it.

Two further points for the record:

- A life of V visits is the exact start of any longer life in the same world and arm. L2 therefore runs each arm once at its longest length and reads the shorter lengths from the same file; section 17.7's roster of three lengths costs one.
- The probe tasks of 17.10 are reserved per world (its last 20 groups), not across worlds, because that leaves every schedule and digest unchanged.

## 18. E34: two long births, with the present code

E34's passing arm is the one case on file of the pattern Mr. Xu describes: behind for 12,000 updates (17% against 95% at 12 letters), ahead at the last row (99.3 against 84.7 at 40 letters), on one seed, with the last three rows still swinging (section 14.3). The test of that pattern is a longer run, not more runs.

- **Runs.** Arm T050 and plain backpropagation with every byte a target (E28's B-all), seed 73129, 98,304 updates each, four times the present dose. Rows every 4,096 updates. About 6.6 CPU-hours by my estimate.
- **Check.** At row 24,576 each run has the tensor hash of its existing final file, since the stream and the update are unchanged.
- **Reading, at 40 letters, over the 13 rows from 49,152 to 98,304.** "Overtakes and stays": T050's mean is at least 10 points above backpropagation's and it is above in at least 10 of the 13 rows. "Does not": its mean is not above. Otherwise mixed. The rows at 12 and 20 letters are reported beside them.
- **Then.** If "overtakes and stays", the same two runs on seeds 93683 and 116089. The backpropagation birth with the gain (addendum 1) is run at this length only if its present-length endpoint differs from plain backpropagation's by 15 points or more at 40 letters.
- **What it cannot show.** This is a stationary stream, so it bears on the pattern and not on continual learning as such. One seed.

## 19. E36, E37, E38 so far, E35-B and the long E34 births (revision 6)

Read on the workers' branches at 22:16 UTC on 15 November: `claude/vibrant-allen-8hziln` (E36), `claude/magical-pascal-sflem4` (E37), `claude/peaceful-hopper-8ofq4i` (E38, in progress), `claude/busy-darwin-dlrwtm` (E35-B), `claude/vibrant-curie-ivmrt1` (E34, in progress). None is on the default branch yet.

### 19.1 The check

From the committed weights, with my reader: `python3 -I $R/e36_e37_check.py`.

| Figure | Mine | The report's |
|---|---|---|
| E36, wrap score at row 1,024, nine reminders | S 93.2, 88.1, 96.0; B 97.4, 99.2, 95.0; K 94.6, 99.0, 91.9 | the same |
| E36, the same for six fresh references | S 1.1, 1.3, 1.1; B 1.1, 1.4, 2.5 | the same |
| E37, fresh F3 at row 4,096 | S 36.5, 47.1, 53.5; B 64.2, 59.6, 73.5 | the same |
| E37, fresh F1 at row 6,144 | S 67.0, 81.4, 67.6; B 80.4, 83.5, 78.9 | the same |

E38's rows are not committed and E35-B's weights were not recomputed; their figures below are the workers'.

### 19.2 What they say

- **E36: what a life removes comes back in about a hundred updates.** In all nine reminders the birth family's held-out score is back above 85 at row 64 or 128, and the wrap pairs go from under 1% to 79–93% by row 128. A fresh network taught the same batches is at about 1% after 1,024 updates. Table row 1 for all three rules.
- **E36, the other direction.** Sixteen updates on the birth family alone take 40 to 60 points off the life's families, and the compositions go to zero within 256. The modulus in force follows whatever the stream last taught.
- **E37: a learner that has lived needs far fewer items for a new family.** At 131,000 items of family F3 the lived learners are at 77–94 and fresh ones at 7–19, under all three rules; at 262,000 the difference is 17 to 47 points. For F1, learned straight after birth, the difference is 17 to 28 under the sweep and 8 to 16 under backpropagation. Section 3.5 is a saving.
- **E38 so far (seed 73129; sweep arms and the backpropagation `zero`).** The wrap pairs fall from 97% to 18% by row 64, 3% by 256 and 0% by 1,024 in `zero`, `none`, `anchor` and `anchor10` alike, while the new family reaches 90–97. So decay is not needed for the loss, and a uniform pull toward the birth weights does not prevent it at either strength. `rehearse` is at row 64 with the wrap pairs at 65%. The backpropagation arms and the rest of `rehearse` are still running.
- **E35-B, first stage (seed 73129).** With parsing supplied the modulus-free circuit generalises from a tenth of the pairs (99% against 9–34% scrambled). From the byte stream a GRU under backpropagation does not come to use it, forced or not: held-out add and sub 19.9 with the module, 18.0 forced, 19.6 for the GRU alone, which has memorised (98% on taught pairs). Outcome row 3; the stop rule waits for the forced arm on two more seeds.
- **E34, long births, to row 32,768 of 98,304.** Both runs have the short runs' hashes at row 24,576. At 40 letters the passing arm reads 99, 95, 88 at rows 24,576 to 32,768 and plain backpropagation 85, 89, 90. Plain backpropagation is still improving past its old endpoint. Nothing is read until row 49,152.

### 19.3 What this changes

1. **A family is two things with very different lifetimes.** The shared arithmetic lasts and accumulates: it is why a reminder takes a hundred updates and why a new family takes a fraction of the items. The family's own modulus is rewritten by a few dozen updates on anything else. Sweep, Kolen–Pollack feedback and backpropagation agree on both.
2. *(Withdrawn in revision 7; see section 20.)* **Among Part II's controls, rehearsal moves ahead of anchors.** A uniform anchor failed at full scale on one seed under the sweep. What E36 measured, that restoring costs about a hundred updates, makes short, well-timed reminders the first candidate. A pull weighted by each weight's use for the old family remains a candidate; a uniform one does not.
3. **The rate laws of section 17.10 now have first values at full scale:** loss within 64 to 256 updates of another family; restoration within 64 to 128 updates; a new family at 77–94 after 131,000 items for a lived learner against 7–19 for a fresh one. L1 should report the same three quantities in the small world, so that the two scales can be set side by side.
4. **Learned structure has now done what the innate circuit did not.** The comparison is not like for like (a quarter of the pairs after a life, against a tenth at birth), which is what the job below is for.

### 19.4 One new job: E37 at a tenth of the pairs

**Question.** A fresh GRU taught a tenth of the pairs memorises: 98% on taught pairs and about 20% on held-out add and sub after a full birth (E35's arm `G`). Does a learner with a life behind it generalise from the same tenth?

- **Start.** The nine life-end files of E36 (S, B, K), and fresh networks under the sweep and under backpropagation; three seeds. Fifteen runs.
- **Family.** One the lives never met: add, sub, max and min at a new modulus (83 is free), with code bytes no family uses. A tenth of the pairs taught.
- **Stream.** `X16.entry_stream` on that family's entries, with a generator seed no other stream has used; the same batches for every run of a seed.
- **Dose and rows.** 8,192 updates; weights at 512, 1,024, 2,048, 4,096 and 8,192.
- **Scores at every row.** Held-out add and sub at 12 letters; the same on taught pairs, to see memorising; all pairs without letters split by wrap; the earlier families at 12 letters, reported and not read.
- **Reading.** As E37: a saving if lived minus fresh is at least 15 points at equal items in two of three seeds, per rule. Beside it, as a fact and not a mark: each run's held-out score at the row where its taught-pair score first passes 90.
- **Checks.** As E36: row 0 has the life-end file's hash; batch digests equal across the runs of a seed; the update path against E26's dry births; the K step as E36 wrote it.
- **What it cannot show.** One new family; lives made for another purpose; a fresh optimizer in every run.

My expectations, on record: fresh runs at or below 30 on held-out add and sub at every row, 85%; lived sweep runs at or above 60 at row 8,192 in two of three seeds, 50%; a saving under each rule, 75%.

## 20. On returning to retention schemes (revision 7)

Mr. Xu, 15 November: "Earlier many minifly period attempts tried to use slow learner, long term storage, and other fancy schemes, but these yield at best mixed results. So we are revisiting the issue to some extent."

He is right, and item 2 of section 19.3 was a step in that direction. The record on such schemes:

| Scheme | Result | Where |
|---|---|---|
| Connectome learner with fast, slow and adaptation state (V66–V78) | 62.6% against 93.3% for a simple reference; nothing promoted | review, section 2 |
| Eight-store persistent core | taught items 97.1%; effect on held-out items −8.2 points | section 2 |
| Complementary-learning-systems replay (v3–v6) | best held-out 13%; 0% in v6-R1 | section 2 |
| Fast and slow two-store | no gain over a single store | 9.12 |
| Sleep replay; sleep order and scaling | +3 to +6 points; no effect | 9.12, 9.17 |
| Plain replay from a bounded store of raw bytes, on a learner that generalises | 0.81 to 0.98 of the joint-training reference | 9.38, 9.42 |
| Uniform pull toward earlier weights (E38's anchors) | no protection at either strength | section 19.2 |

Three things follow.

1. **The earlier schemes were tested on learners that memorised.** A store or a slow path keeps what the learner has; those learners had nothing that generalised (held-out 0 to 13%). The one audit of that period that named the problem (26 September) found that the goal had shrunk to retention.
2. **On a learner that generalises, the plain store has been enough and nothing more elaborate has added to it.** E38's anchors are one more negative of the same kind.
3. **The new result is not about retention.** It is E37: a learner with a life behind it reaches 77–94% on a new family where a fresh one reaches 7–19%. Whether that ability keeps growing over a long life is what job L2 measures, and it is the quantity the earlier period never had a learner to measure.

So:

- **Retention is not a research question in Part II for now.** Where a store is used it is the plain bounded store of raw items, already qualified in this project. No reminder schedule, slow copy, weighted anchor or second store is a candidate until L2 shows a deficit that the plain store does not cover.
- **A rule for any added mechanism.** It is tested only if (a) the deficit it addresses has been measured on a learner that generalises, (b) the plain baseline is in the same experiment at equal budget, and (c) its expected effect is stated beforehand from the measured rate laws.
- **The headline of every report stays generalisation and its trend with experience**; what is kept at a task's return is reported beside it as a constraint.

## 21. Computation to threshold, and scale as an axis (revision 8)

### 21.1 Mr. Xu's statements

Verbatim, with my translation.

- "scaling的帮助如何呢？同样的任务，如果scaling有帮助，也可以认为在固定算力下我们能做的更多。"
  - *Translation:* how much does scaling help? On the same task, if scaling helps, we can also say that at fixed computation we can do more.
- "对，如果网络计算规模变大，或者数据变多，但是为了达到目标(threshold)需要的训练次数本身变少，结果一样甚至更好，平均下来算力一样甚至更少。那这是一个可以讨论的目标。至于scale which axis是可以通过实验讨论的。"
  - *Translation:* yes: if the network's computation grows, or the data grow, but the number of training steps needed to reach the target (a threshold) falls, with the same or a better result and the same or less computation on average, that is a goal worth discussing. Which axis to scale can be settled by experiment.

### 21.2 The quantity

**Computation to threshold.** For a run, the total computation spent up to the first scored row at which its accuracy reaches a threshold.

- *Accuracy* is on operand pairs never taught, whole answers. It is given twice: on the held-out panel at 12 letters, and on held-out pairs that need the reduction, no letters. The second is the one free of what a network can answer without having learned the task's modulus (section 19.2 and the E40 audit).
- *Thresholds* are 50, 80 and 90%, all three reported, so that no conclusion rests on one number.
- *Computation* is counted in two ways: the number of weights times the number of items trained on, which does not depend on the host; and CPU-seconds on the qualified host, one thread.
- A run that does not reach a threshold within its dose is recorded as "not reached", with its highest accuracy.

A learner or a setting is better, in Mr. Xu's sense, if it reaches the same threshold with the same or less computation, whatever it spent that computation on.

### 21.3 What the record has, and what it lacks

- On the window network of early November, width helped: at width 512 a network from random weights reached 91–94% from a quarter of the pairs, as developed networks did, and collapsed at width 128 (9.7); compositions rose from 37–38% to 52–68% between widths 256 and 512 (9.18).
- For the present GRU there is no measurement at any width but 256. The sizing roster of 11 November was never run.
- So it is not known whether the saving of E37 (a learner that has lived, against a fresh one) shrinks as the network widens. If a wider fresh network generalises from fewer pairs, experience and width are two ways to the same end, and the comparison between them is one of computation to threshold.

### 21.4 Job E41: the same task, three widths, three taught fractions

**Question.** On the birth family, how does computation to threshold change with the network's width and with the share of pairs taught, under each rule?

- **Task.** The birth family (add, sub, max, min mod 89), E26's stream and panels, 0 to 12 letters.
- **Grid.** Width at one half, one and two times the present sizes (every hidden size scaled together; the embedding left at 8). Taught fraction 0.10, 0.25 and 0.50. Sweep and backpropagation. Three seeds. Fifty-four runs, all from random weights.
- **Settings.** Learning rate, decay and batch as E26 at every width.
- **Dose and rows (corrected in revision 9).** 8,192 updates at the widest network, 12,288 at the present width and 40,960 at the narrowest. Scored every 256 updates. My first doses (8,192, and 16,384 at the narrowest) were too short in two ways: a threshold that must hold for three scored rows could not be recorded after 7,680 at the present width, and a narrow run that failed to reach a threshold would have left the comparison of total computation undecided. A run that does not reach a threshold is known only to need more than its dose; the audit gives the rule for when a comparison can still be decided.
- **What is tabled.** For each cell: the row at which each threshold is first reached on each of the two accuracies, the computation to that row in both units, the highest accuracy, and the accuracy on taught pairs at that row.
- **Reading.** None is fixed beforehand; this is a first measurement. The report states, for each rule and fraction, whether computation to the 80% threshold falls, stays within a factor of 1.5, or rises from each width to the next, with the range over seeds.
- **Checks.** At the present width and fraction 0.50 the run has the hash of E26's dry birth at 2,048 updates, under both rules. Batch digests are equal across widths and rules for a seed and fraction. Each width's parameter count is recorded.
- **Cost.** With the corrected doses, about 30 CPU-hours of updates and 5 of scoring by the audit's estimate, a little over half of it at the widest network. One seed per worker divides it evenly.
- **What it cannot show.** Settings tuned at one width are used at all three, so a width may be handicapped by them. Networks with a life behind them exist at one width only, so the crossing of width with experience is left to the small world.

My expectations, on record: at fraction 0.50 the widest network reaches 80% in fewer updates than the present one, 70%; it does so with less total computation, 25%; at fraction 0.10 the widest network reaches 80% on held-out reduction pairs within its dose, 20%; the reading differs between the rules in some cell, 25%.

### 21.5 In the small world

- Width becomes an axis of job L2 for every arm, at three widths, in place of `naive` at one extra width (section 17.10, item 5).
- Every table of L1 and L2 gives computation in the two units of 21.2 beside updates, and the first-visit cost of the n-th task is also given as computation to threshold.
- Comparisons between learners are made at equal total computation with each at its best width, as well as at equal width.
- The crossing that full scale cannot make is made here: computation to threshold for the n-th new task, by width. It says which buys more, experience or width.

### 21.6 What this replaces

In discussion I had proposed adding fresh networks at other widths to E40. Job E41 replaces that: it uses the same task at every point of the grid, which is what "the same task" in Mr. Xu's statement requires, and E40 stays as released.


## 22. Job E42: cells with a limited number of local connections and their own periods (revision 10)

### 22.1 Mr. Xu's statements of 16 November

Verbatim.

- "If a cell A and B has no hyper-connection at all - not in the past, not now, then the weights logically cannot connect A with a cell B that remembers distant history. In other words A does not know this part of the history even if the weight is high. So to some extent the apparent overfitting of weights and hyper-connections during the learning process is the same thing. Can we utilize this?"
- "In theory due to the local learning hypothesis, the neuron S_i can only interact with a pre-specified (perhaps high) number of neurons. So in our setting depending on the update frequency, S_i can add 100 connections in T=10, and add zero at T=11, drop 50 at T=12 etc. Nowhere did we dictate that S_i have to adjust by one connection at a time."
- "Let us presume for simplicity that each cell can connect at most N=10 neurons (after all addition is simple), but we can adjust N and the cell can learn the connection strengths".
- In answer to three questions of mine: "1) both is fine, but ideally the connections can adjust. however a neuron is only allowed to adjust within a pre-defined neighborhood, say via some pre-specified embedding coordinates to mimic biological situation. This is to impose locality. 2) The cell's previous value would be available, and each cell has its own pre-set delays or info processing frequencies. During the period it absorbs info from all other cells that emits to it. So cell at T=0 and T=5 may get inputs from the connections. How to process this will not be pre-determined and will be part of local learning, and you can try some variants. 3) Yes, the encoder and decoder parts can stay."

My three questions were: (1) whether a cell's connections are fixed in advance or change during learning; (2) what counts against the limit, and whether a cell's own previous value is available to it; (3) whether the limit applies only to the cells that replace the GRU, with the input and output parts of the network left as they are.

**For the record of P5.** The clarification of 7 November says that "local" does not mean a fixed small neighbourhood and that which neurons a synapse takes its signals from can be adjusted dynamically. Today's statement keeps the dynamic adjustment and adds that it happens inside a neighbourhood defined in advance. Please enter it beside the clarification.

### 22.2 What the record has

- **Units with pre-set time constants have been tried once, without connections among them.** The two-bank network (`arch17`) had fixed retention rates of 0.5 to 0.999 and no learned recurrence. It did not compose from memory: untaught plus-minus pairs 1.5 / 0.5 / 7.5, against 73.6 / 78.8 / 75.1 for the GRU; one added feedback matrix gave 20.2 / 1.1 / 1.0 (review 9.101).
- **The variants asked for on 13 November were never run.** E29 part B (slow units and delays at birth) was withdrawn after its audit and not rewritten (9.118). This job takes its place.
- **The trained GRU makes its own holding units:** 41 to 46% of units after a birth, none before (9.120 C).
- **I found no experiment that limits the number of connections of a unit, places units in a space, or lets connections change.** I searched the review for the words sparse, prune, rewire and neighbour; the hits are the mushroom-body store and the fly circuits of October, not this learner.

The present job differs from the two-bank network in three ways: the cells are connected to each other with learned strengths, the gates read the state, and a period is an interval between updates, not a rate of decay. Because the earlier design failed, the periods enter as one separate step (22.4), so that their effect is read apart from that of limiting the connections.

### 22.3 The learner

Everything outside the recurrent layer is the present network, unchanged: the byte table, `l1`, the input map into the layer, and the readout `l2`, `lm`, `l3`. The layer keeps 256 units, now called cells, and the GRU's own equations. Three things change: which cells a cell reads, when a cell is updated, and how it combines what arrived since its last update.

```
positions      cell i (0..255) stands at row i // 16, column i % 16 of a 16 x 16 grid with wrap-around
neighbourhood  B_i: the 48 cells j != i whose row and whose column each differ from i's by at most 3, with wrap-around
connections    A_i: N cells of B_i; the limit is used in full (a connection given up is always replaced); cell i also reads its
               own previous value, which is not counted in N
period, phase  p_i from the configuration's set, f_i in 0 .. p_i - 1; cell i is updated at positions t = f_i, f_i + p_i, f_i + 2 p_i, ...
               (t counts from the start symbol at position 0); at every other position its value does not change

at every position t, as arch26.forward does, with the recurrent matrix masked:
    gi(t) = Wi z1(t) + bi                       the input map, unchanged: every cell reads the whole current byte
    gh(t) = (Wh * mask) h(t-1) + bh             mask[i, j] is true for j in A_i and for j = i, the same for the three gates
at a position where cell i is updated, with G and H made from gi and gh over its period (table below):
    r = s(G_r + H_r),  z = s(G_z + H_z),  n = tanh(G_n + r * H_n),  h_i(t) = (1 - z) n + z h_i(t-1)
elsewhere h_i(t) = h_i(t-1);  h(-1) = 0 for every item
readout, unchanged:  z2 = tanh(l2 z1 + lm h),  logits = l3 z2
```

**How a cell combines what arrived during its period.** For a cell of period p updated at position t, the steps of its period are t, t-1, ..., t-p+1. Positions before the start symbol contribute nothing. Four variants; a cell of period 1 is the same cell in all four.

| Variant | G | H | Added parameters |
|---|---|---|---|
| `last` | gi(t) | gh(t) | none |
| `mean` | the mean of gi over the period | the mean of gh over the period | none |
| `step` (the default) | sum over k of a_i[k] gi(t-k) | sum over k of a_i[k] gh(t-k) | a_i: one weight for each step of the period, starting at 1/p |
| `source` | as `step` | sum over k of a_i[k] times [(Wh[k] * mask) h(t-k-1) + bh]: a separate strength for every connection at every step of the period | Wh[k] for k = 0 .. p_i - 1, each starting as a copy of the masked, scaled Wh; a_i as in `step` |

`mean`, `step` and `source` compute the same function at the start. `source` is the variant in which a cell can read each of its sources at its own distances in time, up to its period.

*Corrected in revision 11.* I first wrote that each Wh[k] starts at the masked Wh divided by the period. That put the factor 1/p inside the weights, where the sign rule's single scale cannot see it: under the sweep the error sent back would have been 0.40 of the right size for cells of period 1 and 3.2 times it for cells of period 8 (`e42_brief_checks.py`, part F). With the factor carried by a_i, as in `step`, the ratio is 1 for every period.

**Start.** `arch26.new_net(seed)` unchanged; then periods, phases and connections are drawn; then the recurrent weights outside the mask are set to zero and those inside are multiplied by sqrt(256 / (N + 1)), so that a cell's recurrent input has the variance it has in the present network (4.82 at N = 10; exactly 1 when every cell is connected to every other).

**Draws.** None uses torch's generator, so the initial weights and the data are those of E26.
- Periods: `order = list(range(256)); Random(f"e42|periods|{seed}").shuffle(order)`; the first cells of `order` take period 1, the next period 2, and so on, in the counts of 22.4.
- Phases: `rp = Random(f"e42|phases|{seed}")`; for i in index order, `f_i = rp.randrange(p_i)`.
- Connections: `rc = Random(f"e42|connections|{seed}")`; for i in index order, `A_i = sorted(rc.sample(sorted(B_i), N))`.

**The rules.** Arm B: exact gradients. Arm S: every linear map sends its error back through the signs of its weights times one scale per stacked matrix, as now. For a masked matrix the scale is the mean magnitude of the weights inside the mask, so that the error sent back through an existing connection has the size of an existing weight. In `source` the matrices Wh[k] are stacked into one for this purpose. When every entry is inside the mask, `exp3_state.lin` is called unchanged. The gates' nonlinearities, the products, the weights a_i and the holding of a value between updates are differentiated exactly in both arms, as the gates are now. Credit runs through the whole item.

**Changing connections** (the configurations marked "changing"). Every 256 updates, after the update and before scoring:
1. The strength of a connection is the sum of the magnitudes of its weights, over the three gates and, in `source`, over the steps.
2. A connection of cell i is given up if it has existed for at least 512 updates and its strength is below theta times the mean strength of cell i's N connections. A cell may give up none, or several.
3. Each connection given up is replaced by one to a cell of B_i that is not connected to i at that moment and was not given up at this event. Cells in index order, connections given up in increasing order of source: `rr.choice(sorted(candidates))`, with `rr = Random(f"e42|replace|{seed}")` created once per run.
4. The new connection's weights are zero, so the network's output is unchanged by the addition. The weights of a connection given up are set to zero. The optimizer's two running averages are set to zero for every entry that changed.

A cell that satisfies the locality hypothesis has no information about a cell it is not connected to, so the new partner is drawn at random and is kept or given up later by its strength. theta is set once in stage 0 (22.6, check 10).

### 22.4 Configurations

Stage 1, under the sweep and under backpropagation, three seeds: 24 runs.

| Name | Connections | N | Neighbourhood | Periods | What the step changes |
|---|---|---|---|---|---|
| `G` | every cell to every other | 255 | all | all 1 | nothing: the present GRU, run by the new code |
| `F10` | fixed | 10 | 48 | all 1 | the limit and the neighbourhood |
| `A10` | changing | 10 | 48 | all 1 | connections may change |
| `P8` | changing | 10 | 48 | 1, 2, 4, 8; variant `step` | periods |

Stage 2, under the sweep only, three seeds: 24 runs. Each differs from its reference in one thing.

| Name | Differs from | In |
|---|---|---|
| `N5`, `N20` | `A10` | N = 5; N = 20 |
| `W10` | `A10` | the neighbourhood is all 255 other cells: the limit without the locality |
| `P4`, `P16` | `P8` | periods 1, 2, 4; periods 1, 2, 4, 8, 16 |
| `P8-last`, `P8-source` | `P8` | the variant |

`P8-mean` is deferred (revision 11): it is `step` with its weights held fixed, the least informative of the variants, and the cost is higher than I estimated. Stage 2 is then 21 runs.

Cells by period: half of the cells have period 1 in every set with periods, so the number of cells that follow every byte is the same across them.

| Set | Period 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 1, 2, 4 | 128 | 64 | 64 | | |
| 1, 2, 4, 8 | 128 | 64 | 32 | 32 | |
| 1, 2, 4, 8, 16 | 128 | 64 | 32 | 16 | 16 |

Parameters: 1,124,113 for `G`. With N = 10 the recurrent matrix keeps 8,448 of its 196,608 weights and the network has 935,953 (932,113 at N = 5, 943,633 at N = 20), plus 384, 512 or 640 weights a_i in the three sets (`e42_design_checks.py`, part C).

### 22.5 Data, dose, scores

- **Data and settings:** E26's birth exactly, as E41's present width at taught fraction 0.50: the four operations mod 89, 0 to 12 letters, answer bytes as targets, batch 64, `exp26_candidate.make_opt` with the new parameters in the decayed group, learning rate 0.001, decay 0.25.
- **Dose:** 24,576 updates, E26's own, for every run.
- **Scored every 256 updates:**
  1. E41's three accuracies with E41's panels and its sample of 1,000, unchanged: **A** (held-out add and sub at 12 letters), **R** (held-out pairs that need the reduction, no letters) and the taught panel.
  2. **The length test**, E30's: `exp11_cell.panel(t, exp11_cell.split(t, .5, seed)[1], seed, ks=(12, 40), n=300)` with `t = ("add", 89, b"p")`; accuracy at 12 and at 40 letters. On file for E26's final weights at 40 letters: 96.0, 97.0, 99.67 under the sweep and 97.67, 98.67, 92.33 under backpropagation (`exp45/e30/E30_UNSEEN.json`).
  3. The number of connections replaced since the previous row.
- **Per run:** updates to the 50, 80 and 90 thresholds on A and on R in E41's two versions, the three-row version carrying the description; computation in weight-items, with the count of weights inside the mask; CPU-seconds, reported but not used, since a masked dense matrix costs as much as a full one.
- **At the first batch and at the end:** the cosine between the sweep's gradient and the exact gradient for the recurrent matrix, on one batch. It says whether the sign rule is a better or a worse approximation on a matrix with few entries.
- **At the end, in the weights file:** the mask, periods, phases, the update at which each connection was made, and all added parameters. From them the report gives the share of connections never replaced, and a table of the share of connections by the reader's period and the source's period, at the start and at the end.
- **Committed:** the full weights for seed 73129, sixteen files. For the other two seeds a small file per run with the mask, periods, phases, the update at which each connection was made, the recurrent weights and the added parameters; the structure tables need no more.

### 22.6 Checks before the runs

Stage 0 is done by the worker of seed 73129; the other workers take the runner at that commit and repeat checks 1 and 2.

1. **`G` is E26.** The thirteen tensors of the present network are bit-equal to `arch26.new_net(seed)`'s at the start, and after 2,048 updates their `tensors_sha256`, computed over those thirteen only, is that of E26's dry births: 79618ab2… under the sweep and adf933d7… under backpropagation.
2. **Same data in every configuration.** The batch digest at row 4,096 is E26's: bbd71ab7…, 65b1c7e0…, 1df203fb… for the three seeds.
3. **The limit.** At every scored row: every cell has exactly N connections, all inside its neighbourhood, none to itself; every weight outside the mask is exactly zero.
4. **Time.** On random weights: changing the byte at position t changes no cell's value at an earlier position; a cell's value changes only at its own update positions; a cell with phase f is zero before position f.
5. **The variants agree where they must.** On the first batch `mean`, `step` and `source` give the same logits to 1e-6.
6. **Gradients.** Under backpropagation the parameter gradients equal those of a reference written cell by cell, without masks or buffers, to 1e-5 of each tensor's largest entry, for `P8`, `P8-source` and `F10`. Under the sweep they differ. Gradients outside the mask are zero.
7. **Replacement.** At an event, adding the new connections changes no logit; the same command run twice gives the same `tensors_sha256`, with changing connections.
8. **The reviewer's reader.** For seed 73129 the worker commits dry runs of 2,048 updates of `F10`, `P8` and `P8-source` with their weights files and their answers on the length-test panel. I write a numpy reader from this section and reproduce the answers. This does not hold stage 1.
9. **Cost.** CPU-seconds per update and per scored row for each configuration, on an idle host, and a forecast.
10. **theta.** Three dry runs of `A10` under the sweep, 4,096 updates, with theta = 0.1, 0.25 and 0.5. The value used everywhere is the one whose mean share of connections replaced per event, over the events at updates 768 to 4,096, is nearest 5%. If all three are below 1% or above 30%, stop and report. The choice uses no accuracy.

### 22.7 How the result is stated

Fixed beforehand, as in E41. For each comparison, rule and seed:

- **Updates to the 80 threshold**, on A and on R: E41's table of four cases (*falls* below 2/3, *rises* above 1.5, otherwise *stays*, with its rules for a run that does not reach the threshold), read here as fewer, more or about the same. The same in weight-items.
- **The length test:** the mean accuracy at 40 letters over the last three scored rows; *lower* or *higher* if the difference is 15 points or more, otherwise *about the same*.
- A cell of the table carries the description that two of its three seeds share; otherwise *mixed*. All three values are given.

Comparisons: `F10`, `A10` and `P8` each against `G`; `A10` against `F10`; `P8` against `A10`; every stage 2 configuration against its reference; and, in stage 1, the sweep against backpropagation within each configuration.

**One stop.** After seed 73129's stage 1 runs: if none of `F10`, `A10` and `P8` reaches 50 on A within the dose under either rule, stop and report. Otherwise stage 2 runs without a further release.

### 22.8 What arithmetic says beforehand, and my expectations

A cell with a long period can carry the first operand to the `=` sign without learning to keep anything: it is updated once while the operand's two digits are inside its period and is not updated again until after the `=`. Whether that is possible depends on the period, the phase and the number of letters (`e42_design_checks.py`, part D):

| Period | Share of (phase, number of letters) combinations in training for which it is possible | Numbers of letters for which no phase allows it | At 40 letters |
|---|---|---|---|
| 2 | 0 | all | never |
| 4 | 0.06 | 2 to 12 | never |
| 8 | 0.44 | 10, 11, 12 | never |
| 16 | 0.67 | none | never |

So with periods up to 4 the operand must be kept across updates in nearly every training item, as in the GRU; with periods up to 16 it need never be; and at 40 letters it must be kept across updates in every case. The argument predicts that longer periods make training easier and the length test harder. `P4`, `P8` and `P16` test it.

| | Expectation | Chance |
|---|---|---|
| X1 | Check 1 passes under both rules | 85% |
| X2 | `A10` reaches 80 on A within the dose, two of three seeds, under the sweep | 65% |
| X3 | `A10` needs more updates than `G` to reach 80 on A (*rises*), two of three seeds, under the sweep | 60% |
| X4 | `A10` reaches 80 on A in no more updates than `F10`, two of three seeds, under the sweep | 55% |
| X5 | `P8` against `A10`, updates to 80 on A under the sweep: fewer 20%, about the same 40%, more 40% | |
| X6 | `P16` is lower than `A10` at 40 letters, two of three seeds | 55% |
| X7 | `P4` is lower than `A10` at 40 letters, two of three seeds | 25% |
| X8 | Some stage 1 configuration's description against `G` differs between the sweep and backpropagation | 40% |

### 22.9 Cost and order

E26's birth took 3,276 CPU-seconds with its own scoring, 0.133 per update. The new layer does the same matrix work plus buffers; `source` does more. My estimate was 50 to 70 CPU-hours for the 48 runs and about 2 for stage 0. The audit replaces it: the coordinator's own timings give about 150 CPU-hours for the specification file as it stands, and about 75 to 80 if the loop over a period's steps is written to cost what its arithmetic costs, which the worker is asked to do and to check against the specification. Check 9's forecast decides. One worker per seed. The job does not wait for the small world and does not displace E39, E40 or E41.

### 22.10 What it cannot show

- **A saving in computation.** The input map and the readout stay dense and hold most of the weights, so limiting the recurrent connections removes 17% of the network. The job measures updates to threshold and generalisation. Whether many more cells can be afforded at the same computation needs the input and the readout limited too, which is a later job.
- **Settings.** Learning rate and decay were chosen for the dense GRU.
- **The form of a cell.** The GRU's equations are kept. A cell whose way of combining its inputs is itself learned is the next step if this one works (22.11).
- **One task family, answer targets, one grid.** The arrangement of cells is arbitrary; `W10` says whether the neighbourhood matters at all.
- **Distances longer than an item.** The state is reset at every item, so no period or connection can reach further back.

### 22.11 Choices that are mine, and what comes next

Mr. Xu left these to me; each can be overturned.

1. **The limit N counts connections between cells only.** Every cell still reads the whole current byte through the input map, as every GRU unit does. Everything a cell knows about the past then comes through its N connections and its own previous value.
2. **The GRU's equations are kept in this job.** What is learned is the strengths, which cells are connected, and how what arrived during a period is weighted. A cell that holds a graded value for a variable time needs the form `(1 - z) n + z h` in some guise, so the equations cost less generality than it may seem.
3. **A grid of 16 by 16 with wrap-around and a neighbourhood of 48.** The simplest arrangement with no edges.
4. **Half the cells at period 1**, the rest divided as in 22.4.
5. **The scale of the kept weights at the start**, and **the scale of the sign rule on a masked matrix** (22.3).
6. **Replacement by a strength threshold with a target rate of 5% per event** (22.6, check 10), and **the limit used in full**: Mr. Xu's limit is "at most N"; here a cell always has N.

Next, depending on the result: a cell whose update is a learned sum of its inputs and of products of pairs of them, with no gate equations given; limited input and readout connections with more cells at the same computation; and the cell layer as an arm of the small world once L1 has reported.

### 22.12 The first seed read (revision 17)

Seed 73129, fifteen runs, on `claude/gallant-dijkstra-i9zgsr` at 7fbdbdc2. The check is in `E42_SEED_73129_VERIFICATION_20261116.md` (script `e42_seed73129_verify.py`): my reader reproduces the worker's recorded answers of the three dry runs byte for byte and the final scores of all fifteen runs with no item different; the thresholds below are recomputed from the receipts, by the checking agent and again by me. One seed: nothing here is a result of E42 yet.

| Run | Updates to 80 on A | to 90 on A | At 12 letters | At 40 letters | Weight-items to 80 on A |
|---|---|---|---|---|---|
| `G`, sweep | 5,632 | 12,800 | 98.4 | 96.4 | 4.05e11 |
| `G`, backpropagation | 5,120 | 10,240 | 96.9 | 94.6 | 3.68e11 |
| `F10`, sweep | 13,312 | 20,480 | 90.1 | 89.1 | 7.97e11 |
| `F10`, backpropagation | 8,704 | 15,360 | 92.9 | 92.6 | 5.21e11 |
| `A10`, sweep | 13,824 | 22,528 | 92.7 | 91.8 | 8.28e11 |
| `A10`, backpropagation | 9,472 | 20,992 | 92.8 | 92.2 | 5.67e11 |
| `P8`, sweep | 7,680 | 10,496 | 96.3 | 96.0 | 4.60e11 |
| `P8`, backpropagation | 6,400 | 8,192 | 97.0 | 96.6 | 3.84e11 |
| `N5`, sweep | 17,152 | not reached | 88.2 | 86.9 | 1.02e12 |
| `N20`, sweep | 14,336 | not reached | 89.4 | 87.9 | 8.66e11 |
| `W10`, sweep | 17,408 | not reached | 88.6 | 88.0 | 1.04e12 |
| `P4`, sweep | 8,192 | 10,496 | 95.7 | 94.4 | 4.91e11 |
| `P16`, sweep | 7,936 | 11,776 | 94.6 | 82.8 | 4.76e11 |
| `P8-last`, sweep | 11,520 | 20,224 | 92.0 | 91.3 | 6.90e11 |
| `P8-source`, sweep | 7,680 | 11,008 | 94.2 | 93.7 | 4.67e11 |

The letter columns are means of the last three scored rows.

**What the seed shows, by the brief's descriptions.**

1. **Ten local connections a cell are enough to learn the task and to answer at 40 letters,** at 1.7 times the dense network's updates under backpropagation and 2.4 times under the sweep (`F10` against `G`: *more*).
2. **Periods take most of that cost away.** `P8` against `A10` is *fewer* under the sweep (7,680 against 13,824) and `P8` against `G` is *about the same* on every measure under both rules. `P8` reaches 90 before the dense network does, under both rules, and equals it at 40 letters. I gave "fewer" 20%. *(Revision 21: on three seeds `P8` is not level with `G`; it needs 1.25 to 1.75 times its updates, and it reaches 90 first on one seed of three. What holds is the comparison with `A10` under the sweep and the equality at 40 letters. See 28.3.)*
3. **Taking in what arrives during the period is the part that matters:** `P8-last` needs half as many updates again as `P8` (11,520 against 7,680). *(Revision 21: slower on all three seeds, by 1.50, 1.09 and 1.31 times.)* A separate strength for every step adds nothing (`P8-source`).
4. **Period 16 costs at 40 letters in the direction the arithmetic of 22.8 gave,** 82.8 against 96.0 for `P8`, in every one of the last six rows, but under the mark of 15 points. Periods 4 and 8 do not.
5. **The number of connections and the neighbourhood change little** (`N5`, `N20`, `W10` against `A10`: *about the same*).
6. **The rules differ only where the cells are sparse and have no periods:** `F10` and `A10` need about one and a half times the updates under the sweep that they need under backpropagation; `G` and `P8` need the same. The cosine between the sweep's gradient and the exact one for the recurrent matrix ends at 0.60 to 0.64 in `F10` and `A10`, against 0.74 in `G` and 0.67 in `P8` (the receipts' figures).

**A defect of my replacement rule.** `A10` against `F10` is *about the same*, but the comparison does not test changing connections. A new connection starts at zero and is judged after 512 updates against half the mean strength of connections that have been trained from the start; it reaches about a tenth of that mean in 256 updates. So nearly every new connection is given up at its first judgement: 35 to 41% of the connections at the end were made at the last two events and carry no weight, 6 to 13% are replaced connections that lasted, and a cell works with about six of its ten (3.2 of 5 in `N5`, 12.4 of 20 in `N20`). The share replaced per event rises from 9% in the first quarter of a run to 18 to 20% in the last, where the calibration of check 10 read only the first 4,096 updates. The rule and the calibration window were mine. Readings 2 to 4 are between runs that share the defect and are not touched by it; reading 5 is, since the three runs differ in connections that mostly carry no weight.

**What follows, after the other two seeds.**

- If they agree, the finding of E42 is reading 2: a network with a twenty-fifth of the recurrent connections and cells of several pre-set periods does what the dense network does on this task. It is not an advantage, and E42 cannot show one (22.10).
- Changing connections need one corrected run before anything is said of them: a new connection judged against its own age, or kept for at least 2,048 updates.
- The test this structure was argued for is retention (section 24.4, item 1). E43's question put to `F10` and `P8` is the next job there, once E43's three additions say where the loss sits.

## 23. What 64 updates change when a network that has lived learns a new family (revision 12)

A read-only measurement of mine on E40's files, seed 73129: the life-end weights (row 0 of E40's lived runs), and rows 64 and 8,192 of the three lived runs. The row-64 files were committed by the E40 worker at my request (45b6fd79 on `claude/vibrant-allen-8hziln`). Nothing was trained; every figure is a forward pass of the numpy reader on saved or recombined weights. Script: `e40_row64_changes.py` beside this memo, parts A to E.

Probes, fixed once: for add and for sub of the new family (mod 83, a tenth taught), 1,500 never-taught pairs that need the reduction and 1,000 that do not, no letters; for the two earlier families of the life, 600 pairs each of add and sub mod 97 and mod 73. Whole answers; each figure is the mean of add and sub.

### 23.1 Figures

Never-taught pairs of the new family that need the reduction, percent right. Seeds 73129, 93683, 116089 in that order; K has row-64 weights for seed 73129 only.

| Network | Lived under the sweep | Under backpropagation | Under K (73129) |
|---|---|---|---|
| Life end | 6.2, 2.3, 0.1 | 4.2, 0.4, 11.6 | 8.8 |
| After 64 updates | 75.6, 69.8, 75.5 | 84.1, 92.3, 95.7 | 80.3 |
| After 8,192 updates | 76.8, 82.5, 82.9 | 84.8, 94.1, 89.1 | 79.4 |
| Life end, with only the input map and the recurrent matrix taken from row 64 | 57.1, 63.0, 64.6 | 68.0, 85.2, 87.5 | 70.9 |
| Life end, with only the recurrent matrix from row 64 | 33.8, 39.8, 44.0 | 44.1, 70.5, 71.5 | 61.0 |
| Life end, with only the input map from row 64 | 16.5, 13.4, 7.0 | 9.4, 29.3, 44.0 | 27.1 |
| Life end, with only the readout (`lm`, `l2`, `l3`) from row 64 | 7.3, 5.3, 0.8 | 7.2, 2.1, 13.7 | 12.0 |
| Life end, with only the byte table and `l1` from row 64 | 8.3, 4.2, 1.0 | 5.1, 7.3, 17.4 | 12.8 |
| Row 64, with the recurrent matrix put back to the life's end | 24.0, 16.4, 14.3 | 24.3, 28.8, 62.6 | 38.8 |
| Life end plus the largest 1% of the changes | 7.9, 4.0, 0.8 | 3.8, 0.8, 12.4 | 10.6 |
| Life end plus the largest 5% | 15.6, 11.0, 2.1 | 4.7, 14.9, 17.0 | 17.8 |
| Life end plus the largest 20% | 38.4, 27.7, 32.3 | 29.7, 65.0, 72.3 | 52.7 |
| Life end plus the largest 50% | 66.2, 68.1, 72.8 | 76.9, 92.9, 95.2 | 78.1 |
| Every matrix's change replaced by its best approximation of rank 4 | 41.0, 25.1, 24.0 | 53.1, 49.4, 80.0 | 60.5 |
| Of rank 8 | 62.3, 47.0, 45.4 | 74.4, 81.2, 90.1 | 75.1 |
| Of rank 16 | 71.0, 61.8, 67.0 | 81.2, 90.5, 95.4 | 79.2 |

The earlier family mod 97, percent right:

| | Lived under the sweep | Under backpropagation |
|---|---|---|
| Life end | 92.8, 90.2, 95.2 | 94.5, 98.8, 94.0 |
| After 64 updates | 77.0, 54.0, 47.5 | 64.6, 56.4, 57.2 |
| After 64 updates, with the recurrent matrix put back | 89.6, 83.7, 93.3 | 89.8, 95.9, 93.0 |

How far the weights moved in 64 updates (seed 73129): each weight matrix by 4 to 12% of its size at the life's end (`l2` by 31%), the byte table by 0.3%. The rows of the four new code bytes moved by 0.02 to 0.06 against sizes of 2.2 to 3.3; the rows of the earlier families' codes did not move at all.

### 23.2 Readings

1. **What is learned in 64 updates lies in the recurrent layer.** The changes to its input map and its recurrent matrix, placed alone on the life-end network, give 57 to 65 of the 70 to 76 points under the sweep and 68 to 88 of the 84 to 96 under backpropagation. The changes to the readout give nothing by themselves, and neither do those to the byte table and `l1`.
2. **The new codes' rows of the byte table are used as they stood.** They were never trained before E40 and moved by about one percent of their size. The network reads four random vectors it had never seen and adjusts what follows them.
3. **The change is spread over many weights and is low in rank.** The largest 5% of the changed weights add at most 15 points; half of them are needed for most of the effect. But an approximation of rank 16 of each matrix's change recovers 62 to 71 of the 70 to 76 points under the sweep and 81 to 95 of the 84 to 96 under backpropagation. At rank 8 the sweep's change recovers less of its total (45 to 62) than backpropagation's (74 to 90): the sweep's change has somewhat more directions. I note it and do not read it further.
4. **The gain and the loss are in the same matrix.** Putting the recurrent matrix back removes most of the new family and restores the earlier one to 84 to 96%, in all six networks.

### 23.3 What it does not show

- **More than three seeds, or K beyond one.** The readings hold in all six networks of the sweep and backpropagation and in the one of K.
- **That few weights could not do it.** A change spread over many weights is what this optimizer produces: it moves every weight whose gradient keeps its sign by about the same amount. Whether a learner allowed few connections reaches the same result is E42's question. The low rank says only that the change needed has few directions.
- **Anything after row 64 but before row 8,192,** and anything about pairs with letters.

### 23.4 The request, met

The row-64 weights of the lived S and B runs of seeds 93683 and 116089 were committed by the E40 worker at 6922a3a4, and the tables above include them.

## 24. Long-term memory: Mr. Xu's objection, and what section 20 does and does not establish (revision 14)

### 24.1 The objection

Mr. Xu, 16 November: "按照你的三项发现，爱因斯坦势必需要每隔一段时间重新复习世界地理，才能回忆起德国的生活。现实生活中长期记忆是客观存在的，我们遗漏了什么？"

*Translation:* by your three findings, Einstein would have to review world geography at intervals in order to recall his life in Germany. In real life long-term memory exists as a matter of fact; what have we left out?

The objection is right, and section 20 is restricted below.

### 24.2 What survives a life without rehearsal (my measurement)

`kept_after_life.py` beside this memo; read-only. The first family (mod 89) at the end of the nine lives, which never rehearse it: whole answers on 1,500 pairs of each operation, no letters, percent. Seeds 73129, 93683, 116089 in that order. At birth every figure is between 88 and 100.

| | Lived under the sweep | Under backpropagation | Under K |
|---|---|---|---|
| Add, pairs that need the reduction | 0.7, 0.1, 1.4 | 1.7, 0.0, 0.1 | 1.0, 0.6, 0.7 |
| Add, pairs that do not | 44.7, 55.5, 25.9 | 38.1, 26.9, 49.4 | 57.9, 41.8, 62.9 |
| Sub, pairs that need the reduction | 0.4, 0.1, 0.1 | 0.1, 0.0, 0.0 | 0.0, 0.1, 0.4 |
| Sub, pairs that do not | 79.4, 11.9, 1.7 | 91.6, 39.6, 92.5 | 73.8, 73.0, 3.7 |
| Max | 98.5, 3.2, 25.7 | 98.3, 14.1, 0.0 | 98.7, 51.9, 0.0 |
| Min | 0.0, 96.4, 95.1 | 33.5, 65.5, 92.7 | 0.0, 92.3, 99.1 |

1. **What needs the modulus is lost in every network:** 0 to 1.7%.
2. **Some knowledge survives a whole life of 36,864 updates without one rehearsal.** Max or min is at 92% or more in 8 of the 18 cases.
3. **Survival is not reliable.** The other 10 cases are at 66% or less, 8 of them below 35%. Which of the two operations survives goes with the seed, not with the rule: max in seed 73129, min in the other two, under all three rules.

So the learner is not without long-term retention, but nothing in it guarantees retention, and what later learning contradicts is always lost.

### 24.3 What section 20 established, and what it did not

- **Established:** for families that contradict one another (the same operands and operator, another modulus), what is lost returns within about 100 updates, and items of the old family mixed into the stream keep it. For that case no mechanism beyond a store of items is needed.
- **Not established, and stated too widely by me:** that retention is not a research question. Three reasons.
  1. **Fixed memory.** A store of items cannot hold examples of everything a long life has taught. Under Mr. Xu's premise of fixed memory, retention without rehearsal is necessary, not optional.
  2. **Only the most adverse case has been measured.** Every life here consists of families that contradict the earlier ones. Mr. Xu's example is the other case: later learning that neither repeats nor contradicts the memory. It has never been measured.
  3. **Only one kind of memory is present.** The tasks are rules computed by one shared circuit. Nothing here is a fact stored once and recalled by a cue.

### 24.4 What the learner lacks beside a memory that lasts (argued, not measured)

1. **Separation.** In this learner every update changes nearly every weight, and the gain of a new family and the loss of an old one are in the same matrix (section 23). A memory held in connections that later learning does not touch could not be lost to it. This is the principle of Mr. Xu's statement of 16 November on cells with no connection, and the reason E42's networks should later be given a retention test.
2. **A second store.** A fast store of particular events in separate codes, with replay produced by the learner itself. The replay store of E38 is the experimenter's device, and it holds raw items.
3. **Consolidation at the single connection.** A connection that has been stable for long becoming harder to change. E38 tested only a pull of equal strength on every weight, which failed.

I recall the account of memory that combines the second and third (a fast store beside a slow learner, and replay during rest) from the literature and did not check it in this session. No mechanism is proposed here.

### 24.5 Job E43: is a family lost when the learner then learns something unrelated?

**Question.** After a birth, does training on a task that shares no input with the first family remove that family, as training on a contradicting family does?

- **Starts.** The six E26 births.
- **Arms.** `conflict`: the second family alone (mod 97), as E38's arm `none`. `unrelated`: a task on bytes that no family uses, with its own codes, with answers of two bytes and the end symbol so that the harness is unchanged; for example, returning two symbols shown before a run of fillers. The specification is yours.
- **Dose and rows.** 4,096 updates; scored at 0, 64, 256, 1,024 and 4,096.
- **Scores.** The first family: add and sub split by the reduction, max and min. The new task itself, to show that it is being learned.
- **Reading, fixed beforehand.** *Kept* if under `unrelated` the first family's reduction pairs are at 80% or more of their level at row 0 at row 4,096, in two of three seeds; *lost* if at 20% or less; otherwise *partly kept*. `conflict` is the reference and is expected at zero by row 1,024.
- **What each outcome means.** *Kept*: the fast loss measured so far is loss by contradiction, and the learner holds what is not contradicted. *Lost*: updating every weight removes even unrelated knowledge, and separation (24.4, item 1) becomes the first thing to test.
- **Cost.** Corrected in revision 15: two arms are twelve runs, and the brief's four arms are twenty-four, about 3.5 to 4 CPU-hours with scoring by the audit's estimate. My "twenty-four runs, about 2 CPU-hours" was a slip.

My expectation, on record: under `unrelated` the reduction pairs fall below half their starting level by row 1,024 in two of three seeds, 60%.

### 24.6 What weight decay alone does (found by the audit of E43; recomputed by me)

Every life so far runs with a decay of 0.25 at a learning rate of 0.001: each update multiplies every weight outside the byte table by 0.99975, a time constant of 4,000 updates. The audit computed what that does with no learning at all, on the six births: every parameter but the byte table multiplied by 0.99975^t, nothing else changed, and the first family scored (`e43_brief_checks.py`, part C; I reran it). The reduction pairs of add and sub, as a share of their level at birth, seeds 73129, 93683, 116089:

| Updates | Factor on the weights | Sweep births | Backpropagation births |
|---|---|---|---|
| 256 | 0.938 | 95.2, 97.8, 98.1 | 97.3, 96.7, 96.8 |
| 1,024 | 0.774 | 17.8, 35.3, 26.1 | 39.3, 49.0, 46.0 |
| 2,048 | 0.599 | 0.4, 1.3, 0.3 | 0.4, 3.1, 1.3 |
| 4,096 | 0.359 | 0, 0, 0 | 0, 0, 0 |

Max and min go the same way, a little later: 4 to 35% at 2,048 and zero at 4,096.

What it changes.

1. **Under the standing decay nothing is kept passively.** A family whose weights no gradient re-creates is gone after about 2,000 updates, whatever else is learned and whether or not anything contradicts it. This is a second cause of loss beside the one E38 measured: E38's loss by a contradicting family takes 64 to 256 updates and happens without decay too.
2. **What survives a life in 24.2 is being re-created, not stored.** A life is 36,864 updates, nine time constants. The max or min that stands at 98% at its end is held there by the life's own max and min tasks, which use the same machinery under other codes. It is rehearsal by shared use, and the table of 24.2 should be read so.
3. **For Mr. Xu's objection:** this learner, as it is set, has a forgetting time of a few thousand updates built into its optimizer. The decay was chosen for births, where it is commonly held to favour rules over tables; whether the births here need it is not on file as far as I have found. A learner that is to keep what it does not use cannot have every weight decay at one rate.
4. **For E43:** the reading stays on the arm without decay, as the coordinator wrote it. The arm under decay is read against the table above, not against the other arm: it asks whether unrelated learning holds anything up against a decay that would otherwise remove everything.

### 24.7 E43 read (revision 16)

Branch `claude/peaceful-hopper-8ofq4i` at ac56179d. I recomputed the figures below from the 24 receipts; I could not use the weights (see the first request).

- **The registered reading is *lost*, under both rules.** After 4,096 updates on the unrelated task without decay, the first family's reduction pairs are right in 0 of 3,916 for add and for sub, in all six networks. The new task is learned completely (300 of 300 on taught and on held-out pairs).
- **The loss is complete after 64 updates,** before the new task scores anything: 0 to 6 of 3,916 in five networks, 254 and 131 in the sixth. Pairs without the reduction, max and min go with it. Under `conflict` at the same point the pairs without the reduction, max and min are still at 49 to 98%.
- **The answers become the end symbol alone** (the worker's observation on seed 73129: 497 and 511 of 600 prompts at update 64).
- **The weights do not move much more than under a contradicting family:** the recurrent matrix by 17 to 20% of its size in 64 updates against 13 to 16%. The state at the `=` sign changes by 0.39 to 0.86 of its size against 0.21 to 0.30.

My expectation of 24.5 (below half by 1,024 updates, 60%) was right in direction and far too mild.

**What it does not settle.** Three explanations remain, and they call for different remedies.

1. *The knowledge is still in the network and is not expressed.* The unrelated task never rewards a digit, so the shared readout is driven away from digits for every input.
2. *The optimizer's first steps.* A fresh AdamW moves every weight by about the learning rate at each early step whatever the size of its gradient, and the new task starts from a loss of 18.
3. *The function is overwritten,* and would be under any update that learns the new task in this network.

**Three additions asked for.**

1. **The weight files.** The report says that update 4,096 of every run and updates 64 and 1,024 of seed 73129's `unrelated` runs are committed. They are not on the remote; the branch ignores `*.pt`. With them I can place the loss as in section 23: the first family's score with only the readout put back, and with answers chosen among digits only.
2. **A reminder,** as E36's: from the six `unrelated` networks at update 4,096, 1,024 updates on the first family, scored at 0, 16, 32, 64, 128, 256 and 1,024, beside E36's fresh references. If the family is back within about 100 updates, explanation 1 or 2 holds and what was learned was not destroyed.
3. **One arm under plain gradient steps:** `unrelated` again with the update in proportion to the gradient (no momentum, no normalisation by the gradient's size), the learning rate being the smallest of 0.01, 0.03 and 0.1 at which seed 73129 meets the brief's condition under backpropagation. If the first family survives there, the equal-sized step on every weight is the cause, which is section 24.4's first item in its simplest form.

### 24.8 E43's additions read (revision 19)

Branch `claude/peaceful-hopper-8ofq4i` at 722cb621. Figures from the receipts, read by me; the plain runs under the sweep are being continued to 16,384 updates and are not final.

**The reminder.** From the six `unrelated` networks at update 4,096, training on the first family as E36 did. Reduction pairs of add and sub, percent:

| Updates of reminder | Sweep (73129, 93683, 116089) | Backpropagation |
|---|---|---|
| 64 | 1.2, 1.0, 0.9 | 1.4, 0.0, 0.0 |
| 256 | 5.8, 1.1, 1.2 | 1.7, 1.0, 1.5 |
| 1,024 | 91.3, 10.7, 69.3 | 91.3, 14.4, 42.0 |

After a whole life E36 found the family back at 85% or more within 64 to 128 updates, and a fresh network at 1.1 to 2.5% after 1,024. So after unrelated learning the family is not back in a hundred updates in any network; something is saved, since two of six are above 90% at 1,024 where a fresh network is near zero, but the family was not merely unexpressed.

**Plain gradient steps** (no momentum, no normalisation, learning rate 0.03 by the brief's rule). Under backpropagation the first family's reduction pairs are at 0.4, 2.8 and 2.1% after 64 updates and at zero from 256 on, and the new task is learned by update 2,048 to 4,096. Under the sweep they are at 16.7, 45.2 and 19.7% after 64 updates, 6.6, 25.7 and 6.4% after 256 and at most 4.9% after 1,024, before the new task has scored. So the equal-sized steps of AdamW are not the cause.

**Where the loss sits** (`e43_where_the_loss_sits.py`, my measurement on the weight files of seed 73129; 1,500 reduction pairs each of add and sub).

| After 64 unrelated updates | Sweep | Backpropagation |
|---|---|---|
| Whole answers | 0.1 | 0.0 |
| Both digits right when chosen among the ten digit outputs only | 3.9 | 2.0 |
| The readout put back to the birth's | 3.7 | 0.7 |
| The input map and the recurrent matrix put back | 8.1 (digits only 84.5) | 27.6 (digits only 47.4) |
| The birth with only its recurrent matrix changed | 34.8 | 30.6 |
| The birth with only its input map changed | 68.5 | 59.0 |
| The birth with only its readout changed | 5.3 (digits only 91.5) | 78.4 (digits only 92.1) |

After 1,024 and 4,096 updates the birth with only its recurrent matrix changed is at about 1%.

**Reading.** Of the three explanations of 24.7, the third holds. The first family's function is overwritten throughout the network within 64 updates, most of all in the recurrent matrix. The readout is also driven away from digits, under the sweep enough to remove the answers by itself, but putting the readout back restores nothing. The remedy to test is therefore separation: a change that cannot reach the connections the earlier family uses. Section 27 gives its simplest test.

**Two further additions to E43** (item 17).

1. *The conventional reference.* The `unrelated` arm with items of the first family mixed into the stream in E38's share, with and without decay, on the six births. If rehearsal keeps the family at no cost here as it did in E38, it is the reference any other remedy is measured against.
2. *E42's networks.* The `unrelated` arm from `F10` and `P8` of seed 73129 under each rule, with E42's forward. Every cell there still reads the whole byte and the readout is shared, so I expect *lost* (90%); the run replaces the expectation by a figure.

## 25. The small world after job L1a (revision 16)

### 25.1 What L1a found

Branch `claude/gallant-edison-p0qa68` at ac85c093; the worker's figures, not recomputed. Seventy-two runs: twelve linear tasks, each alone, at 64, 96 and 192 units, under both rules, 16,384 updates each.

- **No run reaches the criterion** of 0.8 on asked pairs, at any size, under either rule.
- **Every run fits its taught pairs:** the training loss ends at 0.033 or less.
- **The level on asked pairs goes with the number of taught pairs:** 0.00 to 0.02 for the four tasks with 32 to 72 pairs, 0.00 to 0.36 for those with 162 to 264, 0.02 to 0.69 for those with 544 to 684.

### 25.2 My explanation is withdrawn

I attributed the failed calibration to the missing birth (section 6.3 of the discussion memorandum; my message of 16 November). L1a refutes it for these tasks: a task that is not learned alone at three sizes would not be learned in a birth either. The coordinator had built the life with a birth on my recommendation, and it was rightly not run.

### 25.3 What the three experiments say together (argued)

A network from random weights finds the rule only when a task offers enough taught pairs; otherwise it stores them.

| Source | Taught pairs per operation | A network from random weights |
|---|---|---|
| E41, a half or a quarter of the pairs at modulus 89 | 3,960; 1,980 | finds the rule in 3,000 to 17,000 updates |
| E41 and E35, a tenth | 792 | does not within 8,000 to 25,000 updates; at half width under backpropagation it reaches 48% after 40,960 and is rising (one seed) |
| L1a, half of the pairs at moduli 8 to 37 | 32 to 684 | does not within 16,384 updates |
| E40, a tenth at modulus 83, a network that has lived | 689 | finds it in about 64 updates |

So the small world's tasks are below what a learner without experience can learn, by construction: its largest task has fewer taught pairs than a tenth of a full-scale family. Neither the audit of the first brief nor mine of revision 3 compared those two numbers.

### 25.4 What I propose in its place

Keep the instrument (schedule, event files, scorer, the measures of 17.3 and the horizon rule) and replace the task set and the learner.

- **Tasks:** families in the full-scale format, one modulus each, at moduli not yet used, taught from a tenth of their pairs, as E40's.
- **Learner:** a network born on the first family from half of its pairs. E41's first seed says the half-width network reaches the threshold with the least computation (0.035 to 0.043 CPU-seconds per update); if the other two seeds agree it is the learner, otherwise the present width.
- **Cost:** at 256 updates a visit, a life of 1,024 visits is 262,144 updates: about 3 CPU-hours at half width before scoring, 9 at the present width.
- **Calibration first (the new L1):** born networks on eight new families in turn, visits of 64, 128 and 256 updates; the gate as before, a median first-visit level between 0.10 and 0.90 on reduction pairs. E40 warns of the other failure here: a learner with three families behind it was at 70 to 84% after 64 updates, so the ceiling is the risk, and the visit length or the taught share is what the calibration sets.
- **What it gives up:** the eighteen operations with coefficients, and lives of more than about 240 tasks.

The generator is new and is audited before any run.

## 26. Locality at the input and the output (revision 18)

### 26.1 Mr. Xu's statements

- 16 November: "It seems the byte level translation / output is the fixed cost head that in the long term can be ignored. What is the problem?"
- 16 November: "No, biological cells obviously cannot see the whole input, so locality is essential and our current set up does not scale. Also it makes no sense for a cell to see one million tokens in a future NLP task."

So locality is a requirement of the design, not an option under test. E42 met it between cells and left it unmet in two places.

### 26.2 The arithmetic (mine; multiply-adds per item and position, forward)

| Part of the present network | Now | How it grows |
|---|---|---|
| The byte table, `l1`, and the readout's own layers `l2` and `l3` | 397,824 | not with the number of cells |
| The input map into the cells and `lm` out of them: every cell reads all 512 input values and every readout unit reads every cell | 2,048 a cell | with the number of cells |
| Connections between cells at N = 10 | 33 a cell | with the number of cells |

In E42's wiring a cell costs 2,081 and limiting its connections to other cells saves 17% at most. With the input read by a fixed number of cells and the readout reading a fixed number, a cell costs 33, or 22 with E42's periods. At the present network's 1.12 million multiply-adds and 1.12 million weights, such a network could have about 16,000 to 20,000 cells in place of 256.

### 26.3 Requirements for the job (the specification is yours)

1. **The interface is a boundary.** A fixed number of cells read the current byte and a fixed number are read by the readout; no other cell touches either. The byte stays the unit at both ends, so the interface does not grow with the task.
2. **Every cell reads at most N cells of its neighbourhood** and its own previous value, as in E42, with fixed connections (E42's replacement rule is not used; 22.12).
3. **Time.** A local connection passes a value one cell a step, and the answer is due at the step of the `=` sign. The cells that read the byte and the cells the readout reads are therefore the same cells or neighbours. A variant that allows a stated number of steps between the `=` sign and the answer may be added, for every learner alike.
4. **Periods by place, as one variant.** E42 scattered the periods. Here one arm gives period 1 to the cells at the interface and longer periods to cells further from it, so that neighbours have neighbouring periods; the other arm scatters them as E42 did. With equal numbers of cells at periods 1, 2, 4, and so on, the computation a step stays below twice that of the fastest group however many groups there are, while the longest period doubles with each group.
5. **Sizes.** At least three numbers of cells (for example 256, 1,024 and 4,096) at one interface, and the comparison is made at equal computation a update and at equal number of weights, beside E42's `G` and `P8`.
6. **Arithmetic by index, not by masked matrix,** since a matrix of all pairs of cells cannot be held at these sizes. At 256 cells the new forward and both rules' gradients must equal E42's to its tolerances.
7. **Measures.** E42's (updates and weight-items to threshold on A and R, the length test), the multiply-adds a update, and one diagnostic for whether the cells away from the interface are used: the scores with those cells held at zero.
8. **A full-length run on one seed before the roster,** with its diagnostics read, as the working rules now require.

### 26.4 What I expect, and what it cannot show

On one family, a few hundred cells are enough, so I do not expect more cells to lower the updates to threshold; the job shows whether a network wired in this way learns at all and what a cell costs. Where more cells should pay is in holding many families and in keeping them apart, which is the long-life world of E44 and the question of section 24. The error has further to travel to cells away from the interface, and E42's first seed already shows the sweep's gradient less well aligned in a sparse matrix (0.60 to 0.64 against 0.74), so the sweep may fall behind backpropagation here by more than it did in E42.

## 27. Job E48: a new family learned only in connections gated by its own code (revision 19; renamed in revision 20)

### 27.1 Why

Three measurements point the same way. What 64 updates learn of a new family is a change of low rank in the recurrent layer (section 23). The loss of the earlier family is in the same matrices (section 23). Under unrelated learning the function is overwritten throughout the network, with plain steps as with AdamW (24.8). The programme's question is whether changes can be confined to the connections in use. This job puts it in its simplest form: the change for a new family is made only in connections that take part when that family's code is present. The earlier families are then untouched by construction, and what is measured is the price in learning.

### 27.2 Requirements (the specification is yours)

- **Starts, family, stream, panels and scores: E40's,** on the networks that lived under the sweep and under backpropagation, three seeds, so that E40's own runs are the reference.
- **What may change.** The four new code rows of the byte table, and for each of the two matrices of the recurrent layer (the input map, 768 by 512, and the recurrent matrix, 768 by 256) one added term of rank r, the product of two matrices, applied only on items of the new family. One term a family, shared by its four operations. It starts at zero. Every other parameter is frozen.
- **The gate is supplied by the harness** from the item's code byte. This is a first stage in the sense of E35's: an upper bound. Whether the learner can produce the gate itself is a later question, and the report says so.
- **Ranks** 4, 16 and 64. **Rules:** each network under the rule it lived under.
- **Dose** 1,024 updates; scored at 0, 16, 32, 64, 128, 256, 512 and 1,024.
- **Checks.** Row 0 equals E40's row 0 on every score. With the term off, every earlier family's answers are the same item for item at every row, and the frozen tensors keep their hash: the keeping is by construction and is checked, not reported as a result. The batch digest is E40's.
- **Reading, fixed beforehand,** for each rule and rank, on the new family's reduction score at row 256 against E40's run of the same seed at row 256: *no price* if within 10 points in two of three seeds; *a price* if 30 or more below; otherwise *some price*. The same at rows 64 and 1,024 is tabled.
- **Tabled beside it:** the count of numbers changed for the family (2,304 times r, and 32 for the code rows: 9,216, 36,864 and 147,456), against the network's 1,124,113 and against the number of items a replay store of that size would hold.
- **Cost.** Eighteen runs of 1,024 updates: about 1.5 CPU-hours with scoring.

### 27.3 What it cannot show

- That a learner can find the gate, or decide when a new term is needed.
- Fixed memory: the terms grow by one a family, and the comparison with a replay store of equal size is the fair one; it belongs to the long-life world.
- Unrelated tasks, whose answers need output units of their own.

### 27.4 My expectation, on record

Rank 16 gives *no price* under backpropagation, 60%, and under the sweep, 50%. Rank 4 gives *some price* or *a price* under both, 60%.

# Part III. The results of 17 November and what follows from them

## 28. The results of 17 November, checked (revision 21)

Mr. Xu's instruction of 17 November: "Most works are done or waiting for your audit. Discuss next steps. As I am outside, coordinate yourself." This Part is the answer. Section 28 gives what was checked, section 29 a measurement of my own, section 30 the decisions and the plan.

Every result that arrived was checked from its receipts and weight files with the numpy reader, by an agent of mine whose main figures I then ran again myself. Each document below ends with its commands. Nothing was trained.

| Job | Document beside this memo | What was reproduced |
|---|---|---|
| E41 on three seeds; E46 | `E41_E46_VERIFICATION_20261117.md` | 54 and 18 receipts; every threshold; the last-row counts of all 42 and 18 weight files |
| E42 on three seeds | `E42_THREE_SEEDS_VERIFICATION_20261117.md` | 45 runs: every final count; 720 of 720 thresholds and means; 105 of 105 descriptions |
| E47, the trial | `E47_TRIAL_VERIFICATION_20261117.md` | 4 runs: all 7,200 recorded answers byte for byte; every threshold |
| E48 | `E48_RESULT_VERIFICATION_20261117.md` | 18 runs (section 28.7) |
| E49; E43's second addendum | `E49_E43ADD2_VERIFICATION_20261117.md` | 8 and 16 runs: every last-row count |
| E44's calibration and the roster note | `AUDIT_E44_L2_ROSTER_NOTE_20261117.md` | section 28.8 |
| The E50 brief | `AUDIT_E50_SPARSE_INPUT_CODE_BRIEF_20261117.md` | section 30.2 |

No tabled figure of any worker's report was found wrong. What changes below is what the figures are taken to show.

### 28.1 E41 on three seeds: computation to threshold by width

Updates until three scored rows in a row are at or above 80% on score A (held-out add and sub at 12 letters); median of the three seeds and range; "n.r." is not reached within the dose.

| Rule, taught fraction | Half width (128 cells) | Present width (256 cells) | Double width (512 cells) |
|---|---|---|---|
| Sweep, 0.50 | 10,496 (10,240 to 12,288) | 5,376 (4,864 to 5,632) | 2,816 (2,816 to 3,328) |
| Back-propagation, 0.50 | 7,680 (7,424 to 11,520) | 4,352 (4,096 to 5,120) | 3,072 (2,816 to 3,328) |
| Sweep, 0.25 | 17,408 (14,080 to 24,832) | 8,704; 9,984; n.r. | 6,400 (4,096 to 6,912) |
| Back-propagation, 0.25 | 7,680 (6,144 to 10,240) | 4,352 (3,584 to 4,864) | 4,608 (3,584 to 4,608) |
| Either rule, 0.10 | n.r. on every seed | n.r. | n.r. |

Four statements hold on three seeds.

1. **The sweep against back-propagation.** With half of the pairs taught the sweep needs 1.10 times back-propagation's updates to 80% on A (median; range 0.92 to 1.38) and 1.19 times on the reduction score R (1.00 to 2.29). The ratio is nearest one at the double width (0.92 to 1.15) and largest at the half width (1.07 to 2.29). With a quarter of the pairs taught the sweep reaches 80% on R in 4 of 9 runs.
2. **Width.** Updates to threshold fall as the network widens, and computation to threshold rises at every step (by 1.1 to 2.5 times from half to present, by 1.5 to 4.0 from present to double). The half width is the cheapest in 24 of 24 cases. It is also the narrowest width that was run, so the cheapest width is not located.
3. **A tenth of the pairs** is not learned at any width under either rule. At the half width under back-propagation, score A is still rising at 40,960 updates (48, 68 and 58%).
4. **The learners of the long lives.** E44's rule for choosing a width gives the half width on all three seeds. The six half-width births at fraction 0.50 are on the default branch and score 92.8 to 96.2 on A and 90.9 to 94.7 on R.

One consequence for the long lives: the half width is where the sweep is furthest behind back-propagation, so a comparison of the two rules over a life at that width starts from a handicap that the present width does not show.

### 28.2 E46: learning the first family without weight decay

The sentence of the worker's report is reproduced: at the present width, with half of the pairs taught, the first family is learned without decay as fast as with it under both rules (updates to 80% on R at decay 0 against decay 0.25: sweep 0.96, 1.05 and 0.88 times; back-propagation 1.28, 1.50 and 0.80 times), and the length test at 40 letters is within 3 points. Three qualifications belong beside the sentence. Under the sweep the final level of R is lower without decay on all three seeds (by 3.5, 1.6 and 5.2 points). Without decay the size of the recurrent matrix has not settled at the end of the run (70 to 72 against 26 to 29, and still growing by about 1 a thousand updates). And there is no birth without decay at the half width, which is the width of the long lives. The lives without decay should therefore record the sizes of the weights at every scored row.

### 28.3 E42 on three seeds: what holds and what does not

Updates to 80% on A, seeds 73129, 93683, 116089.

| Configuration | Sweep | Back-propagation |
|---|---|---|
| `G`, the dense network | 5,632, 4,864, 5,376 | 5,120, 4,096, 4,352 |
| `F10`, ten fixed connections a cell | 13,312, 13,056, 12,544 | 8,704, 6,912, 9,216 |
| `A10`, ten connections, replaced during learning | 13,824, 14,336, 13,824 | 9,472, 7,680, 7,936 |
| `P8`, as `A10` with periods 1, 2, 4, 8 | 7,680, 8,448, 7,424 | 6,400, 7,168, 7,168 |

- **Holds on three seeds.** Limiting a cell to ten connections raises the updates to threshold by 2.3 to 2.7 times under the sweep and 1.7 to 2.1 times under back-propagation. Restricting the connections to a neighbourhood adds nothing consistent to that (`W10` against `A10`: 1.26, 0.84, 1.02). Replacing connections during learning is about the same as keeping them fixed (1.04 to 1.10), and the defect of my replacement rule (22.12) repeats on every seed. Five connections are slower than twenty by 1.10 to 1.20 times.
- **Holds under the sweep only.** Periods remove 62 to 76% of the extra updates that the limit costs (`P8` against `A10`: 0.54 to 0.59). Under back-propagation they remove 71, 14 and 21%.
- **Does not hold.** In 22.12 I read the first seed as showing `P8` level with the dense network. On three seeds `P8` needs 1.36, 1.74 and 1.38 times the dense network's updates under the sweep and 1.25, 1.75 and 1.65 times under back-propagation. That sentence of 22.12 is withdrawn.
- **Holds, and was not read before.** With periods the sweep is level with back-propagation (1.04 to 1.20 times its updates), as in the dense network (1.10 to 1.24), where without periods the limited network puts the sweep behind (1.36 to 1.89).
- **Computation.** Counted in multiply-adds to 80% on A against the dense network under back-propagation, no configuration is below one: `P8` under back-propagation 1.04 to 1.46, `P8` under the sweep 1.25 to 1.72, `F10` under the sweep 2.2 to 2.7. The limit saves 17% of the multiply-adds a position (section 26) and costs more than that in updates.
- **Length.** `P16` is low at 40 letters on two of three seeds (82.8 and 67.3 against 94.1).

### 28.4 E47, the trial: input and output at a boundary

The four runs (1,024 cells of which 256 read the byte and are read by the readout; seed 73129) learn the task: 80% on A after 10,496 updates under back-propagation in both arms, and after 13,568 (`one`) and 15,360 (`place`) under the sweep. Two findings decide what follows.

- **No saving.** Computation to 80% on A is 1.7 times the dense network's under back-propagation and 2.3 to 2.5 times under the sweep, and 1.05 to 1.29 times that of E42's `F10`, which has a quarter of the cells.
- **Cells beyond the first ring take no part.** With the 540 cells of the second and third rings held at their mean, or given the states of another item, no score changes by more than 0.7 points. Their states hardly differ between items (standard deviation at the `=` sign at most 0.018 in the second ring and 0.002 in the third, against 0.37 to 0.46 at the interface). This is so before any training: on the starting weights the differences between items fall six- to tenfold from one ring to the next. The first ring does take part.

So the trial shows that a network with the interface at a boundary learns, and it shows that this wiring does not use cells away from the boundary. That the weak passage of differences at the start is the cause of the later non-use is an explanation and has not been tested; it would be refuted if a start at which the differences reach the third ring still left those cells unused after training.

### 28.5 E49: an interface in which each cell reads part of the input values

On 256 cells, seed 73129, with E42's `F10` as the reference (levels are means of the last three scored rows, A / R).

| Setting (input values a cell reads / cells a readout unit reads; multiply-adds a position against `F10`) | Back-propagation | Sweep |
|---|---|---|
| `F10` (512 / 256; 1.00) | 93 / 94; 80% on R at 9,472 | 91 / 89; at 15,104 |
| `in16` (16 / 256; 0.59) | 85 / 83; at 20,480 | 1.9 / 1.4 |
| `I64` (64 / 32; 0.51) | 74 / 65; not reached, rising | 1.3 / 0.9 |
| `out8` (512 / 8; 0.86) | 74 / 71; not reached, rising | 10.6 / 7.8 |
| `I16` (16 / 8; 0.45) | 24 / 17; not reached | 1.2 / 0.9 |

- Under back-propagation a limited interface is learned slowly: the one setting that reaches 80% on R does so at 1.28 times `F10`'s computation, so nothing is saved.
- **Under the sweep a limited input is not learned at all.** The training loss has not fallen since update 4,096 in `I64`, `I16` and `in16`. The sweep's gradient at the start is as well aligned with the exact one as in `F10` (cosine 0.73 to 0.77 at the input map), so the alignment at the start does not explain it. What is seen at the end is another state of the network: the byte's code has shrunk from 0.41 to 0.05 to 0.07, the cells' states at the `=` sign differ between items by 0.03 to 0.04 (0.40 in `F10`), and `l3` has grown 9 to 21 times. The cause is not established.
- A continuation would at best show 80% on R under back-propagation at about 3.4 times `F10`'s updates. It is not worth its cost.

### 28.6 E43's second addendum: rehearsal, and E42's networks

- **Rehearsal keeps the first family** while the unrelated task is learned, under both rules, with and without decay: the reduction score at update 4,096 is 95 to 105% of its start in all twelve runs. The price is in computation and in speed. Half of the items of every update are rehearsal (one old item for each new one), and at update 256 the unrelated task is behind the unrehearsed run in all twelve comparisons.
- **E42's networks lose the family as the dense one does**: `F10` and `P8` are at or below 1% on reduction pairs by update 64. Limiting the connections between cells does not separate what is learned.
- The two `P8` weight files at update 4,096 lack the coefficients over the steps of a period, so those two networks cannot be rebuilt. No reported figure depends on them. The report's list of committed weights should mark the two files as incomplete.

### 28.7 E48: a new family learned in connections that are present only with its code

The 18 runs are reproduced from the life-end weights and the term files. The figure is the reading fixed in the brief: the mean of the reduction score on never-taught pairs at rows 128, 256 and 512, E48 minus E40's run of the same seed and rule (seeds 73129, 93683, 116089).

| Rule | Rank 4 | Rank 16 | Rank 64 |
|---|---|---|---|
| Sweep | -15.8, -31.2, -27.4 (*some price*) | -2.4, -8.7, -8.2 (*no price*) | +1.2, +5.0, -1.0 (*no price*) |
| Back-propagation | -4.8, -8.5, +0.2 (*no price*) | +3.9, -1.9, +4.5 (*no price*) | +9.3, +1.4, +6.3 (*no price*) |

- **The earlier families are kept item for item,** as the construction requires. About 37,500 earlier items a network (every earlier task at five lengths, the compositions, every held-out reduction pair) are answered as by the life-end network, in all six networks. With the gate forced open on those items, the same term lowers the life's reduction pairs under modulus 97 from 87 to 98% down to 1 to 52%. The separation is the gate's and nothing else's.
- **The new family is learned later, and to a higher level.** At row 64 every run is below E40, by 6 to 59 points. At row 1,024 the mean reduction score at ranks 16 and 64 is 91% under the sweep and 97% under back-propagation, against E40's 86 and 90%, and it is above the highest value that E40 reaches in 8,192 updates in all six networks. This second comparison was not fixed beforehand, and E48 differs from E40 in one more thing: the lived weights do not decay.
- **E40's loss was not needed.** In E40, with every weight free, the life's four panels fell by 38 to 43 points within 256 updates of learning the new family. E48 learns the same family with those panels untouched.
- **The gate also withholds.** Forced open on items of the birth family, the term raises that family's `max` or `min` in some networks (for example from 4 to 52% and from 14 to 94%). What a family learns behind its gate is not available to another family.
- **Memory.** A term of rank 16 is 36,896 numbers; thirty such terms equal the network.

The brief's reading used the words "as fast as". They were my requirement, and the result does not support them for the first 64 updates. The reading by the three-row mean stands as tabled.

### 28.8 E44: the calibration, and the roster note for the first long lives

**The calibration is reproduced.** With every one of its 917,504 answers marked again against my own stream, the gate's medians equal the report's. Under the conventional learner at 256 updates a visit the median level on reduction pairs at a family's first visit is 0.30 to 0.64 in the eight worlds, all inside the gate; at 128 updates four of the eight worlds pass and at 64 none. 256 is the shortest length that passes.

Two things the calibration shows beside the gate.

- **Nothing is kept from one visit to the next** (review 9.156; I recomputed the second lap). One visit after a family's own, its share right on the fixed list is 0.007 to 0.076 of what it was. At a return the first update scores 0.011; the last quarter of a return visit reaches 0.58 against 0.49 at first visits.
- **What the wrong answers are** (my count on the calibration's lists). One visit later, 15% of the conventional learner's answers on a family's reduction items, and 21% of the sweep's, are the right answer under the modulus of the family taught last, where chance is about 1%. The network answers every family with the modulus it learned last. This is section 29's finding seen from another side: the modulus is held in connections that every family uses.

**The roster note** (release 1: four arms under back-propagation, `naive`, `replay` and both without decay; the learners are E41's half-width births; 32 lives; 59 CPU-hours; world 1's two long lives are the trial). The check is in `AUDIT_E44_L2_ROSTER_NOTE_20261117.md`.

- **The change to the runner is sound, with conditions.** It is not code yet: the runner on the default branch is the one the calibration ran. As the note describes it, the two new arguments select which rows of the list are written, and cannot change what a life learns. A second change is needed that the note does not name, because the present runner refuses the `replay` arm.
- **Nine changes are required,** given as text to apply in the audit's section 8. Those that bear on what the lives can show: the quantities the note says it records are not the ones the scorer computes, and "kept" is not defined for a life with returns (definitions D1 to D14 are proposed); the trend of the score at 256 visits partly reads the schedule and should be read on returns only (the defective definition was mine, section 17.4); the wrong answers are to be classified as above; `replay` with half of each update taken from the store teaches 6,144 new items a visit, the dose at which the gate failed in four of eight worlds; and decay alone cannot be read in a life with returns.
- **In order as written.** The store is 1,269,828 bytes, the size of the network, and holds 55,209 items, against 80,000 to 108,000 distinct taught pairs in a life of 1,024 visits, so the store cannot hold everything it is offered. Such a life meets 41 to 52 families and has 972 to 983 returns. The closing rule is that of section 17.5. The six births are committed.
- **After the nine changes the lives may start, and the note needs no second audit.** I read the changed runner and its check receipts when they are committed. That reading need not hold the trial, because a defect in the code of the list cannot change the weights or the event file of a life.

## 29. Cells in use and directions in use: why the first family is overwritten, and what would leave it intact (revision 21)

### 29.1 The question, and two definitions

The programme's question speaks of changes "confined to the connections in use". E43 showed that the first family is overwritten within 64 updates of unrelated learning, and its second addendum that limited connections between cells do not prevent it. This section measures on E43's saved networks how much of the network an item uses, and whether the change made by the unrelated task can be kept apart from what the first family uses. The scripts are `cells_in_use.py` and `directions_in_use.py`; the files are the six E26 births, E43's `unrelated` networks at updates 64 (seed 73129 only) and 4,096, and its rehearsed networks at 4,096. Forward passes only.

*A cell in use.* For a cell at one position of an item, with update gate z, candidate n and previous state h (so that the new state is (1 - z) n + z h), every change made at that position to the cell's incoming connections is a product that contains the factor (1 - z)(1 - n²) + z(1 - z)|h - n|, under back-propagation and under the sweep alike. I call it the change factor. Where it is zero the cell's incoming connections cannot change, whatever the error. Every change to a connection that leaves a cell is a product that contains the cell's state.

*Directions in use.* A matrix of connections receives input vectors. The directions in use by a set of items are those spanned by the input vectors that the items present to the matrix, with a constant appended for the bias. A change of the matrix that is zero on those directions leaves the matrix's output on those items as it was; if this holds at every matrix, the whole forward pass on those items is as it was.

### 29.2 Figures

**In the born network, before any unrelated learning, every cell is in use on unrelated items.** The unrelated task's bytes (128 to 225) occur in no item of the first family.

| Seed, rule | Change factor on unrelated items: median over the 256 cells; cells below 0.05 | State size on unrelated items: median; cells below 0.05 | Correlation over cells between the change factor on family items and on unrelated items |
|---|---|---|---|
| 73129, sweep | 0.39; 0 | 0.45; 0 | +0.73 |
| 73129, back-propagation | 0.37; 0 | 0.53; 0 | +0.76 |
| 93683, sweep | 0.33; 0 | 0.40; 0 | +0.80 |
| 93683, back-propagation | 0.28; 0 | 0.49; 0 | +0.72 |
| 116089, sweep | 0.36; 0 | 0.45; 0 | +0.81 |
| 116089, back-propagation | 0.24; 2 | 0.58; 0 | +0.78 |

**The change is spread over all cells, and the loss comes before the gain.** After 64 unrelated updates (seed 73129) each cell's incoming connections have changed by 12 to 26% of their size (5th to 95th percentile of cells, both rules), and half of the squared change lies in 90 to 99 of the 256 cells, where an even spread would give 128. At that point the first family is at 0.1% and the unrelated task at 0.0% under both rules: the family is destroyed before anything of the new task has been learned.

**With rehearsal the weights move as far, in another direction, and the family is intact along the whole way.** d1 is the change made by 4,096 unrelated updates without rehearsal, d2 with it.

| Seed, rule | Size of d1, of d2, against the size of the birth's weights | Cosine between d1 and d2 | First family on the straight line from the birth to the unrehearsed network, at 0.1, 0.2, 0.3 of the way | Lowest value of the first family on the straight line from the birth to the rehearsed network |
|---|---|---|---|---|
| 73129, sweep | 0.66, 0.71 | +0.28 | 92.8, 71.9, 35.1 | 98.3 |
| 73129, back-propagation | 0.57, 0.65 | +0.39 | 86.4, 53.3, 23.6 | 98.8 |
| 93683, sweep | 0.59, 0.65 | +0.37 | 97.8, 89.7, 52.2 | 99.4 |
| 93683, back-propagation | 0.53, 0.61 | +0.47 | 92.0, 67.4, 24.8 | 98.8 |
| 116089, sweep | 0.62, 0.70 | +0.30 | 97.8, 81.0, 31.5 | 97.6 |
| 116089, back-propagation | 0.54, 0.62 | +0.43 | 90.7, 56.0, 9.3 | 97.9 |

On the line to the unrehearsed network the unrelated task is at 0.0 to 3.7% half-way and at 61 to 90% at 0.7 of the way, so no point of that line holds both. On the line to the rehearsed network the unrelated task rises from 0 at the half-way point to 60 to 89% at three quarters and 97 to 100% at nine tenths, with the family at 97.6% or above throughout.

**Directions in use by the first family's taught items** (four operations, 0 to 12 letters, 2,080 items; the number of directions that hold the stated share of the trace of the inputs' second-moment matrix; ranges over the six births).

| Matrix (input dimensions with the bias) | 99.9% | 99.99% | Not zero |
|---|---|---|---|
| Input map (513) | 9 | 12 to 15 | 46, the number of bytes in use |
| Recurrent matrix (257) | 239 to 244 | 255 to 256 | 257 |
| `lm`, at the answer positions (257) | 243 to 248 | 255 to 256 | 257 |
| `l2`, at the answer positions (513) | 9 | 9 | 11 |
| `l3`, at the answer positions (513) | 249 to 361 | 361 to 452 | 513 |

**The code of a byte has nine directions, and every byte shares them.** The code of a byte is its row of the byte table, 8 numbers, passed through `l1`. Of the squared size of the code of a byte of the unrelated task, none of which has occurred before, 99.9% lies inside nine directions of the codes of the family's bytes, and 99.98 to 99.99% inside the span of the 46 bytes in use (six births; a random direction would have 9%).

**A change projected away from those directions.** For each matrix the change d1 (or d2) is multiplied on the input side by the projector on the complement of the family's directions; `l1` and the byte table's rows of bytes in use are put back to the birth's. With the directions that hold 99.99% of the trace removed, the first family is as at birth to within 0.2 points in every case, on reduction pairs and on held-out pairs at 12 letters alike (for example 95.9 and 97.5 against the birth's 95.9 and 97.5); removing as many directions chosen at random leaves it destroyed or damaged (4 to 83% on reduction pairs). The unrelated task is at 0.0% in every projected network.

**Where the change lies.** Of the squared change of the input map made by the unrelated task, 51 to 61% under the sweep and 80 to 85% under back-propagation lies in the nine leading directions of the family's inputs, where a change in random directions would have 1.8%; the figures with rehearsal are the same (54 to 66% and 82 to 84%). (The optimizer scales its steps entry by entry, so a change need not lie wholly in the directions of the inputs that caused it.)

### 29.3 Readings

1. **Nothing is left unused, so nothing can be left unchanged.** In the present cells every item drives every cell, including an item made of bytes the network has never met. A rule that changes only the connections in use therefore changes all of them. This is what E43 and its addendum found from outside, and it is why limiting the connections between cells did not help: the limit removes connections, it does not make any cell idle.
2. **One family fills the state of the cells.** At the recurrent matrix the family's own items use 255 or 256 of the 257 directions. A rule that leaves the family's directions untouched would be exact (the projected networks show it, also on pairs never taught) and would leave no room at all for anything else at that matrix.
3. **The input code gives a new byte no direction of its own.** At the input map the family's bytes use 46 of 513 directions, which looks like room. It is not: the code of any byte, met or not, lies in the same nine directions, because it is the image of 8 numbers. A byte that has never occurred therefore drives the input map along directions that the family's bytes already use, and a rule that leaves those directions untouched leaves the new byte nothing. With a code in which a value stands for one byte, as in E50, a new byte would drive only connections that no earlier byte drives. This is an argument for E50's code from the side of keeping; it was drafted for the side of computation.
4. **Rehearsal does not work by holding the weights near their old values**, nor by keeping each matrix's action on old inputs unchanged. Its change is as large as the unrehearsed one and lies as much in the family's input directions. What it finds is a distant set of weights with the same function for the family, reached along a line on which that function never fails. A mechanism that pulls every connection back toward its old value is therefore not a substitute for rehearsal: on the line to the unrehearsed network the family is already below 90% at a fifth of the way.
5. **The family is lost before the new task is learned.** The loss is not the price of the new solution. It is made by the first few dozen updates, in which the connections of every cell are moved by an eighth to a quarter of their size for a task of which nothing has yet been learned.

The rule "change a matrix only along input directions that earlier tasks do not use" is in the literature: Zeng, Chen, Cui and Yu (arXiv 1810.01256) for feed-forward networks; Saha, Garg and Roy (ICLR 2021); and for recurrent networks Duncker, Driscoll, Shenoy, Sahani and Sussillo (NeurIPS 2020), who report that it removes interference between tasks learned in sequence and that the networks reuse structure across similar tasks. I have read their abstracts, not the papers. What the measurement adds for our network is the count of reading 2: after one family there is no room left where the rule would need it.

### 29.4 What follows (argued; the test is the job of section 30.3)

Keeping what has been learned without rehearsing it needs two things together: a part of the network that earlier learning has left unused, and a rule that confines new learning to that part. The record now has each of the two separately.

- E48 supplies the unused part from outside: a term that is present only when the new family's code is present. In the terms of this section the gate multiplies the number of input directions by the number of codes, which is why it finds room where a plain change finds none.
- E41 shows that the first family does not need the whole network: a birth with 128 cells reaches the threshold, at less computation than one with 256.

So a network of the present size can be born using half of its cells, with the other half held in reserve, at no loss. `e51_design_checks.py` checks the arithmetic of this on E41's six half-width births, without training. The half-width birth is placed inside a network of the present size; the 128 added cells have update gates that start shut, so that they are silent on every item of the first family. The embedded network answers the family as the half-width birth does (largest difference of an output 0.004 with the gates' bias at 10). The family then uses 129 of the 257 directions at the recurrent matrix, and 128 are free. A random change of every connection that the rule of section 30.3 allows to change, a sixth to a quarter of the whole network's size, leaves the family's scores as they were on all six births; the same change made without the rule halves them (39 to 58%). One limit follows from reading 3. In the present code the reserve cannot be reached through directions that are free, because a new byte has none; it can be reached only through connections that are free, which are the reserve cells' own.

## 30. Decisions, and the plan from here (revision 21)

### 30.1 Where the adopted question stands

The question (review 9.148): under fixed memory and continued learning, can a learner whose changes are confined to the connections in use keep what it has learned while learning new things, at no greater computation to threshold than back-propagation?

**Computation, for first learning.** In the dense network the sweep needs 1.1 times back-propagation's updates when half of the pairs are taught (28.1). Every structure tried so far costs more computation to threshold than the dense network under back-propagation: a limit of ten connections a cell 1.4 to 2.7 times; the same with periods 1.0 to 1.7 times; the interface at a boundary 1.7 to 2.5 times; an interface in which a cell reads part of the input about 1.6 times at best under back-propagation (1.3 times `F10`'s), and under the sweep it is not learned. What lowers the computation is a smaller network: the half width reaches the threshold at about half the computation of the present width under both rules. On this task, then, sparsity and locality have been costs. Their case rests on networks with many more cells, and the task does not need more cells: E47's cells away from the boundary are not used. More cells can pay only where there is more to hold, which is many families. The computation line and the keeping line therefore meet in one experiment, in which cells that the first family does not need are the room for what comes later.

**Keeping.** Without rehearsal nothing is kept: not through 64 updates of an unrelated task (E43), not from one visit to the next in the long-life world (28.8). Section 29 gives the reason: every cell is in use on every item. With rehearsal the family is kept, at twice the items an update (28.6). With a part of the network that only the new family uses, supplied from outside, the earlier families are kept exactly and the new one is learned (28.7).

**Fixed memory and continued learning.** The instrument is calibrated and audited. No life has been run.

### 30.2 Decisions on what waited for me

| Item | Decision | What it needs |
|---|---|---|
| The runner change and the roster note of the first long lives | Sound with conditions; nine changes as text (audit, section 8); the lives start after them without a second audit | You enter the changes; the worker makes the two runner changes, runs the checks and starts the trial |
| E47's roster (4,096 cells; scattered periods) | Not released | Nothing. A larger sheet would add cells of the kind the trial shows unused, at 37 CPU-hours. The line rests until a start is found at which differences between items reach the far cells; that can be studied on starting weights without training |
| E49, continuation | None | Nothing |
| The E50 brief | Run, changed: nine changes (audit, section 9). The settings `C64` and `C16` are E49's masks, which failed on the learned code, so their failure here would say nothing. Stage 1 is the fixed code on a dense interface (`C-dense`), both rules; stage 2 a limited input with the dense readout, where stage 1 reaches 80% on R; stage 3 a control in which every cell reads one value of every used byte, only if stage 2 fails | You enter the changes; stage 1 can start at once |
| E48 | Closed as read in 28.7 | Nothing |
| E41, E42, E46 | Closed as read in 28.1 to 28.3 | The three-seed report of E41 (2722d1c1) and E46 (b76f5979) are cited by the roster note and are not merged into the default branch |
| E43's second addendum | Closed as read in 28.6 | The report's list of committed weights marks the two `P8` files as incomplete |

On E50, one finding of the audit matters beyond the brief. Under the sweep with a limited input, the code of a byte shrinks until two bytes that must be told apart are 3 to 6 apart, where they are 14 to 16 apart under back-propagation and 14 at the start. E50's code is fixed and cannot shrink. Stage 2 under the sweep therefore tests one explanation of E49's failure: if the sweep learns with a limited input once the code is fixed, the shrinking code was the cause; if it still does not learn, it was not.

### 30.3 Job E51: a new task learned in cells held in reserve, with what the first family uses left unchanged

**Why.** Section 29.4. E48 kept the earlier families by a part supplied from outside. This job gives the network such a part of its own, cells that the first family does not use, and a rule that confines new learning to it, and asks two things: is the first family kept, and what does the new task cost to learn? The arithmetic of the construction is checked in `e51_design_checks.py` (three seeds, both rules, no training).

**The rule of section 20** for any added mechanism is met as follows. (a) The deficit is measured on a learner that generalises: E43. (b) The plain baseline is in the same experiment at equal updates: the arm `rehearse`. (c) The expected effect is stated beforehand, below.

**The learner.** E41's half-width birth with half of the pairs taught (`e41_<rule>_h256_f50_<seed>.pt`, on the default branch), placed inside a network of the present size (feed-forward width 512, 256 cells) as the function `embed` of my script does it. The old weights keep their places. Weights from the 256 added feed-forward units into old units are zero. Every other added weight is drawn at the scale of a fresh network, those from the added cells into old cells included, since without them no error would reach the added cells. The 128 added cells are the reserve.

**Directions in use.** After the embedding, 2,080 taught items of the first family (40 for each operation and each number of letters from 0 to 12) give, for each of the five matrices (input map, recurrent matrix, `lm`, `l2`, `l3`), the second-moment matrix of the inputs it receives, a constant appended; the directions in use are its eigenvectors whose eigenvalue exceeds one millionth of the largest. My script finds on the six births: 129 of 257 at the recurrent matrix and at `lm`, 36 to 41 of 513 at the input map, 11 at `l2`, 329 to 437 at `l3`. They are computed once, before the new task, and stored (0.23 to 0.28 of the network's parameters; 0.08 without `l3`'s).

**The rule.** "Projected" means: the gradient of the matrix is multiplied on the input side by the projector on the complement of the directions in use before it enters the optimizer, and the optimizer's step is multiplied by it again before it is applied. The second multiplication is what makes the keeping exact.

- Rows of the input map and of the recurrent matrix that belong to old cells, and every row of `lm`, `l2` and `l3`: projected.
- `l1`, and the byte table's rows of the 46 bytes in use: not changed. The byte table's other rows: free.
- Candidate and reset rows of the reserve cells: free.
- Update-gate rows of the reserve cells: free. On the first family's items the gates are held shut by the condition of the arm `keyed`.
- No weight decay in any arm (decay would have to be confined as well, which is a later question). AdamW otherwise as in E43.

**Arms,** each from the same embedded network.

| Arm | What may change | What it shows |
|---|---|---|
| `open` | Everything, as in E43. The reserve cells' update gates start shut (a bias of 10) and are free | The loss without any rule, and the speed of the new task without any limit |
| `rehearse` | As `open`, with one item of the first family for each new item, as in E43's second addendum | The plain baseline |
| `keyed` | The rule. The reserve cells' update gates are held shut on every item whose first byte is one of the bytes in use; on every other item they are those of a fresh network, and free | Whether the first family is kept, and whether the new task can be learned in the reserve |

In `keyed` the condition is computed from the item (has its first byte occurred before?) and is not supplied for a family. It is still a device of the harness, and the report says so.

**An arm that is left out, and why.** The arm one would want beside `keyed` has no device. In it the reserve cells' gates start shut, their update-gate rows are projected like the old cells' rows, so that they stay shut on the first family's items whatever is learned, and only the optimizer can open them, through directions that the first family does not use. In the present code this arm cannot work, by reading 3 of 29.3: a new byte has no direction of its own (0.1% of its code), so nothing can open the gates. It becomes possible with E50's code, in which a new byte has eleven values of its own. It is a job for after E50's stage 1, on a birth with that code, and its question is the one that matters for a learner without any device: does the optimizer bring unused cells into use for a new input?

**Tasks.** `unrelated`: E43's task, stream and panels, 4,096 updates. `family`: the first block of E26's life (the family of modulus 97, with E26's taught fraction, stream and scores), its first 4,096 updates. For a born network of the present size the record has the conventional outcome of that block: the new family is learned and the birth family's reduction is lost (sections 3.3 and 3.4). Scored at updates 0, 16, 32, 64, 128, 256, 512, 1,024, 2,048 and 4,096. Each network under the rule of its birth.

**Checks before the runs.**
1. The embedded network answers the first family's panels as the half-width birth does, item for item (my script: largest difference of an output 0.004).
2. The numbers of directions in use equal those my script prints for the birth.
3. A random step that obeys the rule changes no output on the first family's panels by more than 0.05, and the same step without the projection changes the scores (my script: 39 to 58% left).
4. In `keyed`, at every scored row, the first family's answers are those of row 0 on at least 99.9% of items. This is by construction, and a failure is an error of implementation.
5. The batch digests of the two tasks equal E43's and E26's.
6. A dry run with the cost a update of each arm.

**Recorded at every scored row.** For the first family: the reduction score, the four operations on taught pairs, the held-out panel at 12 letters, and what the wrong answers are, in the classes of E43's addendum. For the new task: E43's two panels, or E26's scores of the new family (never-taught reduction pairs, and taught pairs). For the reserve: the number of reserve cells whose update gate is open by more than 0.1 on average over the new task's items, and the largest state of a reserve cell on the first family's items. The sizes of the weights. The weight files at updates 64 and 4,096, listed in the report from `git ls-tree`.

**Reading, fixed beforehand,** for each rule, task and arm. *Kept*: the first family's reduction score at update 4,096 is within 2 points of row 0. *Learned*: the new task's score on its never-taught items reaches 80% within the dose; its cost is the first scored row at 80% against that of `open`.

**Trial first.** Seed 73129: twelve runs. I estimate 2.5 CPU-hours with scoring, from E46's 0.11 CPU-seconds an update at this size and a fifth more for the projection. The other two seeds are released after the trial is read.

**What I expect, on record.**

| | Statement | My chance |
|---|---|---|
| P51.1 | `open` loses the first family on `unrelated`: below 5% on reduction pairs by update 64, both rules. Refuted if the reserve takes the new task and the family survives | 90% |
| P51.2 | `open` loses the first family's reduction pairs on `family` (below half of row 0 at update 4,096), as E26's lives did | 80% |
| P51.3 | `keyed` keeps the first family on both tasks, both rules. By construction | 97% |
| P51.4 | `keyed` learns `unrelated` to 80% within the dose | 60% under back-propagation, 45% under the sweep |
| P51.5 | `keyed` learns `family` to 80% on never-taught reduction pairs within the dose | 50% and 35% |
| P51.6 | Where `keyed` learns `unrelated`, its first row at 80% is at most four times that of `open` | 60% |
| P51.7 | `rehearse` keeps the first family within 5 points on both tasks, both rules | 85% |

If P51.4 fails under both rules, cells in reserve that read the old cells but may not change them are not enough to learn a new task, and the candidate for the long lives is E48's term, chosen by the item's first byte. If P51.4 and P51.5 hold, the reserve learner is the candidate, and the question that remains for it is that of the arm left out.

**What it cannot show.** That a learner can do without the device of `keyed`. How a reserve is shared among many tasks: here the first new task may take all of it. A life. Whether a network can be born with a reserve by itself: here the reserve is made by taking a narrower birth. A comparison with a store at equal memory, which belongs to the long-life world.

### 30.4 Order and pace

Three lines of work, which is the limit of the working rules.

1. **The first long lives** (release 1 of L2): the trial, then the roster.
2. **E51**, the trial seed. The brief comes to me before any worker starts.
3. **E50**, stage 1.

Release 2 of the long lives (the sweep, and a learner that keeps without a store) is specified when the trials of lines 1 and 2 have been read. Its candidate is the reserve learner if E51 shows that it keeps and learns, and E48's term chosen by the first byte otherwise. E50's stage 1 has a second use beside its own question: it gives the births on which the arm left out of E51 can be run.

Resting: E47 at larger sizes; a fourth, narrower width for E41, which would locate the cheapest width (six births, about 2 CPU-hours); a corrected rule for replacing connections (22.12). After release 1, step 5 of the adopted plan: confirmations on sealed tasks of E40's result, E48's and E43's.

### 30.5 What is not proposed

A pull of every connection toward its earlier value is not proposed. E38's anchors gave no protection (section 20), and section 29 shows why: the family is lost a fifth of the way along the straight line to the unrelated task's solution, and the solution that rehearsal finds is as far from the birth as the unrehearsed one. Schemes that weight the pull by an estimate of each connection's importance are in the literature; they are a different mechanism from the one above, and by the rule of section 20 they are not candidates until a deficit remains that the store and the reserve do not cover.

### 30.6 On coordinating while Mr. Xu is away

Everything in this Part was done inside the reviewer's role: checks from receipts and weights, two audits, one measurement, one proposal. I have written to no worker session and started no run. The decisions of 30.2 reach the workers through you, as before.

## 31. Corrections, the origin of two settings, and the ordered plan (revision 22)

### 31.1 Corrections to revision 21

**1. The sweep's lag behind back-propagation (28.1, statement 1; 30.1, "Computation, for first learning").** Revision 21 said that the sweep needs 1.1 times back-propagation's updates when half of the pairs are taught. That figure is the median over three widths and three seeds, and it hides the width. For each seed (73129, 93683, 116089), the sweep's updates to 80% divided by back-propagation's, from E41's receipts:

| Width, taught fraction | On A | On R |
|---|---|---|
| Half width, 0.50 | 1.07, 1.38, 1.37 | 1.22, 1.53, 2.29 |
| Present width, 0.50 | 1.10, 1.19, 1.24 | 1.04, 1.19, 1.25 |
| Double width, 0.50 | 1.00, 1.00, 0.92 | 1.15, 1.09, 1.00 |
| Half width, 0.25 | 1.70, 2.29, 3.23 | 2.12, 3.31, 3.31 |
| Present width, 0.25 | 2.00, 2.79, sweep not reached | sweep not reached on any seed |
| Double width, 0.25 | 1.39, 1.14, 1.50 | sweep not reached, 1.33, sweep not reached |

The statement that holds is this: the sweep's lag falls as the network widens and grows as fewer pairs are taught. At the half width, which is the cheapest width and the width of the long lives, the sweep needs 1.1 to 1.4 times back-propagation's updates on A and 1.2 to 2.3 times on R with half of the pairs, and 1.7 to 3.3 times with a quarter. The sentence of 30.1 is to be read with these figures. Every one of them is at one learning rate and one weight decay (31.2).

**2. The design of E51 (30.3).** It is superseded by revision 2 of the brief, `COORDINATION_20261117/TASK_E51_A_NEW_TASK_LEARNED_IN_CELLS_HELD_IN_RESERVE_20261117.md` (aab57310). The check before release (`COORDINATION_20261117/CHECK_OF_TASK_E51_BRIEF_20261117.md`) found that the keeping of the first family, which revision 1 of the brief called exact by construction, was not exact: the taught items of the first family occupy every direction at the input of the readout's last matrix, so that no change of that matrix leaves the family's answers as they are, and the mark below which a direction counted as unused (1e-6) left changes in directions that the family uses. In revision 2 the columns of the last matrix that belong to the old units, its bias, the first matrix and the 46 rows of the byte table that the family uses are not changed, and the mark is 1e-12. The reviewer's numpy check (`COORDINATION_20261117/check_e51_brief.py`) gives a change of the family's outputs of 3.9e-3 under the rule of revision 1 and 1.5e-9 under the rule of revision 2. Revision 2 also adds an arm that applies the rule without the reserve, so that the two can be told apart, and a Part B on E43's networks that asks how much rehearsal is needed (shares of a quarter down to 1/32 of the items; rehearsal at intervals; smaller steps for the new task). Part B answers to Mr. Xu's statement of 17 November that some rehearsal is acceptable, provided that long-term memory forms in the course of learning and is not produced by our intervention.

**3. Lines of work (30.4).** Four lines are running in place of three: the first long lives with a helper session for the sweep's lives, E50, E51 and E52; E54 is next. The reason for the fourth line is given in rule 5 of `PLAN.md`.

**4. Section 30.6 is superseded.** From 20:22 UTC on 17 November the reviewer has started worker sessions, entered the audits' changes through the workers' own revisions, written three briefs and taken the decisions that an audit leaves to the coordinator, on Mr. Xu's instruction ("coordinate yourself"; "instruct the workers yourself this time"). Every such step is in `COORDINATION_20261117/LOG.md`. Nothing was pushed to the default branch and no document of the coordinator was edited.

### 31.2 Where the learning rate and the weight decay came from

The two inventories of the record (`COORDINATION_20261117/INVENTORY_DENSE_LINE_20261117.md`, `INVENTORY_STRUCTURE_AND_WORLD_20261117.md`) trace every setting to its first use.

- **The learning rate, 0.001,** was taken from the literature on tasks of this kind (the comment in `mod97_local.py`, lines 142 to 144: "as in the grokking papers") and given to both rules. No run of the dense network with this optimizer has another value.
- **The weight decay, 0.25,** was chosen for the sign rule on the two earlier learners (review 9.24 and 9.27). Back-propagation's births used 0.30 until E26, whose brief gave both arms the rule's value.

Both comparisons of the two rules that this memo relies on (E26, E41) were therefore made at a learning rate chosen for neither rule and at a decay chosen for the sign rule. In two messages to Mr. Xu and in the first version of `PLAN.md` the reviewer wrote that these settings had been chosen for back-propagation. That was wrong and is corrected there. What follows is unchanged: the settings were carried into every later experiment without being varied, and E52 varies the learning rate for both rules at two widths and two taught fractions, with a control in which the decay is changed with it. E46 has shown that learning the first family does not need the decay (28.2).

### 31.3 One measurement added: cells in use in the limited and the local networks

`CELLS_IN_USE_IN_LIMITED_NETWORKS_20261117.md` and `cells_in_use_limited.py` (item 2 of the plan; seed 73129; the reviewer ran the script again). Section 29 found that in the dense network every cell is in use on every item. The same holds in the networks of 256 cells in which every cell receives the input: in E42's `G`, `F10` and `P8` and in E49's `in16` no cell's state is independent of the item, and the items of the first family occupy 245 to 256 of the 257 directions of the cells' state. Limiting a cell's connections, or giving cells periods, leaves nothing unused.

E47's networks of 1,024 cells are different. In them 435 to 628 cells have a state that does not vary with the item, all of them outside the 256 cells that receive the input and are read, and the family's items occupy 303 to 334 of the 1,025 directions. Unrelated items do not move these cells either, and only 13 to 75 of them are read by a cell of the interface. A network with more cells than the task needs does hold cells that the first family leaves alone, but no signal reaches them. Whether new learning could use them depends on whether differences between items can be made to pass into them, which is the question of the plan's item 1 (the passage of differences between items at the start) and of the job that waits for it, E53.

### 31.4 Mr. Xu's requirements, and where the record stands on each

Mr. Xu restated the requirements on 17 November, and `PLAN.md` orders the work by them.

| | Requirement | Where the record stands | Running or next |
|---|---|---|---|
| A | Long-term memory forms by itself in the course of learning; some rehearsal is acceptable | Without rehearsal nothing is kept (28.6, 28.8, section 29). With one earlier item for each new item the family is kept. With a part of the network supplied from outside for the new family, the earlier ones are kept exactly (28.7) | E51 (a reserve of cells and a rule; how little rehearsal is enough); the first long lives |
| B1 | Our learner may be slower at first and overtake the conventional one in continued learning | No long life has been run. In first learning the sweep is slower (31.1) | The first long lives under back-propagation and, in the helper session, under the sweep |
| B2 | The end result may be about the same while our learner needs far fewer samples | On the record the sweep needs more taught pairs and not fewer: with a quarter of the pairs it reaches 80% on R in 4 of 9 runs against back-propagation's 9 of 9, and with a tenth its level of R at the end of the run is 7 to 15 against 15 to 62 (E41, three widths, three seeds; the runs differ in length between widths). No limited or periodic network has been run with fewer than half of the pairs | E52 (other learning rates); E54 (the limited and the periodic network with a quarter, 15% and a tenth of the pairs) |
| C | Local learning in the place of full connection, at the input and the output as well | Limited connections between cells are learned at 1.4 to 2.7 times the dense network's computation; an interface at a boundary at 1.7 to 2.5 times; a limited input is not learned under the sweep (28.3 to 28.5) | E50 (an input code in which a value stands for one byte; stage 1 has reported and is being verified); then E53 and the sweep's scale taken for each cell |

The plan has 29 numbered items and nine groups that wait, and it assigns each of the 107 entries of the two inventories to one of them or sets it aside with a reason (`plan_check.py` tests this). Its section 7 is kept current with the state of every item.

### 31.5 For the coordinator

- Section 4 of `LOG.md` lists each decision taken in your place, and section 5 what to read first.
- Of the requests of revision 21, items 18 and 20 have been carried out through the workers' own revisions 2 of the note and of the brief, on their branches; item 21 has been replaced by the reviewer's brief of E51; item 19 stands; items 22 and 23 are yours.
- Three briefs are the reviewer's own (E51, E52, E54). Each was checked before release by an agent of the reviewer that had not seen the reviewer's reasoning, and each check found defects that were entered. Such a check is not the independent audit that the programme's rules ask for. If you wish to audit any of them in the usual way, the workers follow a revised document or a message.

## 32. The evening of 17 November: one result, four readings of the record, and the briefs written (revision 23)

Every document named here is beside this memo on `claude/vibrant-cerf-sje726`; the briefs and the log are in `COORDINATION_20261117/`. Each measurement was made by an agent of the reviewer on saved weights or receipts, without training, and the reviewer ran each script again before reading the result.

### 32.1 E50, stage 1: a fixed code in which a value stands for one byte is learned under both rules

`E50_STAGE1_VERIFICATION_20261117.md`, `e50_stage1_verify.py`; the worker's report is on `claude/e50-fixed-input-code-017dSE`. E42's network with ten local connections a cell (`F10`) was given, in place of the learned byte table and the first layer, a fixed code in which each of the 46 bytes turns on eleven values of its own; every cell still reads all 512 values (setting `C-dense`). Seed 73129.

| | `C-dense`, back-propagation | `F10`, back-propagation | `C-dense`, sweep | `F10`, sweep |
|---|---|---|---|---|
| Updates to 80 on A | 7,680 | 8,704 | 13,056 | 13,312 |
| Updates to 80 on R | 6,912 | 9,472 | 12,800 | 15,104 |
| Level of A, of R at the end | 97.7, 97.1 | 93.2, 94.1 | 93.5, 94.1 | 91.2, 89.3 |

By the bands fixed beforehand every comparison is *about the same*. On R the network with the fixed code reaches 80 earlier than each of `F10`'s three seeds under both rules (6,912 against 7,936 to 9,728; 12,800 against 14,336 to 15,104); with one seed of `C-dense` this is not read as a gain. The verification reproduces every figure of the report from the receipts, and the final weights give the recorded answers on 600 of 600 items in each run. The learned code is therefore not needed for learning the first family: a code that a cell could read part of, value by value, does as well with every cell reading all of it. Stage 2, in which a cell reads 16 of the 512 values, is running; it is the case that E49 found not learned under the sweep with the learned code.

### 32.2 The passage of a difference between items at the starting weights of the local networks

`SIGNAL_PASSAGE_AT_THE_START_20261117.md`, `signal_passage.py` (item 1 of the plan; seed 73129). Two items that differ in one byte are run through a network at its starting weights, and the difference of the cells' states is followed from the cells that read the byte to the others.

- **E47** (1,024 cells, input and output at a boundary of 256 cells). Of the difference at the interface, the first ring of far cells holds 0.19, the second 0.02 and the third 0.003. No uniform change of the start brings the third ring to a third without the difference growing along the item; the recurrent weights times 3 give 0.14 outward and 0.15 back and are the largest multiple without growth. The reason is the wiring: only 2.4 to 4.0 of a far cell's ten connections come from the ring nearer the interface. Training raised the passage into the first two rings and not into the third.
- **E49** (a cell reads 16 of the 512 input values, learned code). Every cell reads values that differ, and the cells carry a difference as evenly as in `F10`. The start has nothing to repair, so the sweep's failure in E49 is not a matter of the start.
- **E50, `C16-in`** (a cell reads 16 values of the fixed code). The cells that read neither of the two bytes hold 0.27 of the reading cells' difference; the recurrent weights times 1.5 to 2 raise this to 0.39 to 0.51.
- In every form in which all cells are updated at every position, `F10` included, a difference of one byte shrinks at the start by a factor of 0.63 to 0.66 a position. In E47's two trained networks of the arm `one` it no longer shrinks.

Consequences. E53 (32.6) is written for E47 alone. E49's case goes to the question of the sweep's scale (item 18 of the plan). A larger start for `C16-in` is decided when E50's stage 2 is read.

### 32.3 E40's saving, counted in everything the learner was given

`E40_SAVING_IN_WHOLE_EXPERIENCE_20261117.md`, `e40_whole_experience.py` (item 26 of the plan). The record states the saving of a lived network on a new family in items of that family only. Three accounts side by side:

- **The new family alone.** A lived network generalises the new family from 2,756 taught pairs (a tenth); a fresh network needs 7,920 to 15,840 pairs of a family of this kind (a quarter to a half), and given the same tenth it reaches 6 to 12%. The lived network needs 2.9 to 5.7 times fewer pairs and 19 to 88 times fewer items. This holds under both rules.
- **With everything before.** Before the new family the lived network was given 30,576 distinct pairs and 24,000 compositions in 3.9 million items. Counted so, it has used 2.1 to 4.2 times the pairs and 5.5 to 17.2 times the items of one fresh network.
- **Over further families.** If each further family cost what E40's did, the lived learner's total would fall below that of fresh networks after 3 to 14 further families in pairs and 6 to 18 in items. No run supports this: no lived network has been taught a second new family, and the earlier families fall while the new one is learned (the family of modulus 97 from 91 to 99% to 18 to 40%).

On Mr. Xu's question of 17 November this gives a first, limited answer. The record does show a learner that needs far fewer samples, in the first account, and the saving comes from what the network learned before; it is found under back-propagation as much as under the sweep, and between the two rules the sweep shows no saving anywhere. Whether the saving outweighs the experience that produced it depends on many families being learned in turn while the earlier ones are kept, which is what the long lives measure and what E51 and E58 work on.

### 32.4 E38's two checks that failed under the sweep

`E38_FAILED_CHECKS_UNDER_THE_SWEEP_20261117.md`, `e38_checks_read.py` (item 27 of the plan). E38's checks 2 and 3 compare, after 64 updates, two runs that should be equal. Under back-propagation they pass (largest difference 2.8e-6 and 6.0e-7). Under the sweep they failed (up to 5.2e-4 and 7.8e-3); the worker reported it, and an agent of the reviewer that searched the fetched branches found no ruling (review 9.141, which reports E38, does not mention the failures). The cause is the check: the sweep passes the error through the signs of the weights, so that one weight near zero whose sign differs between two runs changes the next update by 1e-5 to 2e-3, where back-propagation gives at most 4e-7. Before such a point the two runs agree to 1.5e-6 in single precision, and in double precision to 5.9e-15 over all 64 updates. All twenty weight files of E38 give the receipts' counts. The reviewer's reading, for your ruling (item 25 of section 1): the results under the sweep stand; checks 2 and 3 remain failed as specified; and a check of this kind on a rule that uses signs should compare in double precision, or before the first sign that differs.

### 32.5 The gate of the long-life world applied to the sweep

`E44_GATE_FOR_THE_SWEEP_20261117.md`, `e44_gate_sweep.py` (item 28 of the plan). The visit length of 256 updates was chosen by the gate for the conventional learner. From the calibration's receipts, the sweep meets both conditions of the gate at 256 updates in all eight worlds (figure 0.54 to 0.74, against 0.30 to 0.64 for the conventional learner) and would meet them at 128 as well. Two things follow for the reading of the sweep's long lives. First, before any life the arm `sweep` is above `naive` by 0.161 in the gate's figure (seven of eight worlds), while the sweep continued from back-propagation's birth is 0.257 below: the difference belongs to the rule together with its birth. Second, a late level of the sweep that is ahead by such an amount therefore shows nothing that a long life adds. The helper was asked to report the level of the first 64 visits beside the late level, and the change of the difference with its interval; "ahead in late level" alone is not read as overtaking.

### 32.6 The host, and the briefs written since revision 22

**The host.** A worker's machine is stopped about five minutes after its session's turn ends, with the disk kept, and detached runs die with it. The first long lives lost 51 minutes in this way and were started again; the other sessions were warned, checked their runs, and now hold their turns open while runs proceed. Every brief written since carries this as a rule for the worker. The reviewer's own container was restarted twice; nothing committed was lost.

**Briefs.** Each is checked by an agent that has not seen the reviewer's reasoning before it is released.

| Job | Plan item | What it asks | State |
|---|---|---|---|
| E54 | 10 | The limited and the periodic network with a quarter, 15% and a tenth of the pairs taught, and the dense network at the half width with a tenth for three times the length | Revision 3 after two checks (eight and ten defects entered); the worker runs since 22:40 |
| E58 | 19 | E48's gated term against free learning, with the decay of the lived weights separated from the place of the change, run to 8,192 updates, and both at taught fractions of 0.05, 0.10 and 0.20 | Revision 1, in check |
| E59 | 15 | A task of another kind (E43's) learned by a fresh, a born and a lived network from 2,048, 512 and 256 taught pairs | Revision 1, in check |
| E53 | 9 | E47's network without weight decay, and without decay from a start with the recurrent weights times 3: do the cells beyond the first ring come into use? | Revision 1, in check |

Rule 5 of the plan now allows six lines of work, of which at most four in their first hours; its reason is given there.

## 33. The night of 17 to 18 November: two verified results, the briefs, and the handover (revision 24)

All times are UTC. "Sweep; back-propagation" is the order of every pair of figures. Everything is for seed 73129.

### 33.1 E51's trial: the first family is kept exactly while a new task is learned, by an act of the program

**What was run** (`origin/claude/e51-reserve-017dSEpW` at 2aa469c1; brief `COORDINATION_20261117/TASK_E51_A_NEW_TASK_LEARNED_IN_CELLS_HELD_IN_RESERVE_20261117.md`, revision 2). E41's birth of half the width is placed inside a network of the present size, so that 128 cells and 256 readout units are in reserve. A new task (E43's unrelated task, or the family modulo 97) is then learned for 4,096 updates without weight decay, in four arms: `open` (nothing protected; the gates of the reserve start shut by a bias), `rehearse` (as `open`, every second item from the first family), `keyed-free` (two lines of the forward pass hold the reserve at zero on every item whose first byte the first family uses) and `keyed` (those two lines, and a rule that projects every update of the old units' rows away from the directions that the first family's items present).

**Verified** (`E51_TRIAL_VERIFICATION_20261117.md`, `e51_trial_verify.py`; the reviewer ran the eight parts of the script again, 683 seconds). Every cell of the worker's tables is reproduced from the receipts (2,428 cells, 228 values), and a forward pass written for the verification reproduces the counts from the saved weights. The nine predictions held.

| Arm | Reduction score of the first family at update 4,096 (92.4; 95.3 before) | New task learned from update |
|---|---|---|
| `keyed` | 92.4; 95.3 on both tasks; none of the 32,884 scored answers changed at any scored row | `unrelated` 1,024; 512. `conflict` 1,024; 512 |
| `keyed-free` | `unrelated` 0.0; 0.0. `conflict` 1.0; 0.2 | 512; 512 and 2,048; 1,024 |
| `open` | `unrelated` 0.0; 0.0, lost by update 16. `conflict` 1.3; 0.0 | 512; 512 and 1,024; 1,024 |
| `rehearse` | `unrelated` 90.0; 92.8. `conflict` 88.3; 97.8; 4 to 10% of the answers differ from those before | (the task is learned in every run) |

**Three statements go with the result.**

1. The answers are unchanged; the outputs are not. By update 4,096 they have moved by up to 5.3e-4, because the weights are stored in single precision after each projected step, and the smallest difference between the largest and the second largest output of a first-family answer is 1.8e-4. Nothing bounds this movement over a longer run.
2. The separation is made by the program. With the two lines removed, the same weights of update 4,096 answer 0.00% of the first family's reduction pairs after `unrelated` and 0.40%; 0.00% after `conflict`. The trial therefore does not meet Mr. Xu's requirement that long-term memory form by itself (requirement A); it shows that such a separation, once it exists, is enough.
3. The new task is not learned in the cells in reserve alone. The brief's criterion of that name (a score below 20% with the reserve held shut) is met for `unrelated`, but the task is also lost when the weights from the reserve into the old units are put back to their starting values; for `conflict` the network answers 41.0%; 40.4% of the never-taught items with the cells in reserve held shut, namely most of those whose answer is the same under both moduli.

**Decision taken in your place.** The other two seeds of Part A are not run now: the keeping in `keyed` follows from the construction, and the open question is the one of statement 2. The worker has been told the three statements for its report.

### 33.2 E50, stage 2: with a fixed code and 16 values a cell, neither rule reaches 80 on R

**What was run** (`origin/claude/e50-fixed-input-code-017dSE` at c6c24d9c). E50's network `C16-in` is E49's network with a limited input (the same 16 of 512 values of the code for each cell, the same limited connections, a dense readout) with E50's fixed code in place of the learned one; 49,152 updates under each rule.

**Verified** (`E50_STAGE2_VERIFICATION_20261118.md`, `e50_stage2_verify.py`; the reviewer ran the script again, with the same output). All 88 figures of the worker's result table are confirmed; the saved weights reproduce 2,400 of 2,400 recorded answers; the code, the byte table and `l1` are bit-equal at the start, at update 24,576 and at the end; the masks hold. Two figures of the report's text differ in the third decimal.

| Rule | A, R, T at update 24,576 | A, R, T at update 49,152 | Updates to 80 |
|---|---|---|---|
| back-propagation | 67.9, 62.1, 85.3 | 83.8, 79.4, 96.3 | A 40,704; R not reached |
| the sweep | 32.7, 29.0, 54.6 | 49.5, 43.6, 74.8 | neither reached |

T is the score on taught pairs. Max and min reach 80 under both rules (5,376 and 5,632; 7,424 and 7,680 updates, back-propagation first).

**What it shows, for this seed.** A limited input alone does not prevent the sweep from learning part of the family: E49's network with the learned code stayed below 5 on every score under the sweep. E50's report does not assign the difference to the size of the code, because the fixed code differs from the learned one in its form as well, and by the criterion of E50's brief a code that cannot shrink is refuted as a sufficient condition. Under the sweep the network remains at about half of back-propagation's levels although no error has to reach the code. Measured in addition: all 256 cells are in use under both rules, and the sweep's answers to the reduction pairs do not fall onto a few values (first byte right in 72.0% of the items, second in 55.0%; back-propagation 82.9% and 90.9%). Stage 3 (one value of every used byte for each cell) is running; it was started again at 00:00 after a restart of the machine.

### 33.3 Reported by the workers and not yet verified

- **E51, Part B** (rehearsal; verified after the handover, see 33.8). B1: the first family is kept within 5 points only when every second item is rehearsed (reduction score at update 4,096 over that before: 100.5%; 98.6% at 1/2, 92.2%; 90.6% at 1/4, 89.8%; 79.9% at 1/8, 65.4%; 58.6% at 1/16, 45.8%; 35.3% at 1/32). B2 (the new task with reviews of the first family at intervals): at the end of the eighth gap 58.25%; 72.5% of the probe is still right, against 0.0 at the end of the first; the review needed becomes shorter under the sweep and not under back-propagation. B3 is running.
- **E58** (checks 1 to 5 and 8 passed; its free runs repeat E40's lived runs at all eight rows and its runs with the term repeat E48's to row 1,024). Part A runs on seed 73129. The worker corrected two figures of its message of 00:02, which it had not read from the receipt.
- **E52**: the eight runs at rate 0.001 on seed 93683 end on E41's hashes; the ten runs at 0.003 started at 23:45.
- **E54**: checks 1 to 4 and 6 passed; part 1 is expected to end about 05:00.
- **The long lives**: the helper's three lives of the sweep were at visits 715 to 720 of 1,024 at 00:37 and resumed from visit 640 after a restart. The branch of the first long lives had no commit between 23:37 and 00:45; its worker was asked at 00:45 to check its lives.

### 33.4 The briefs since revision 23

| Job | Plan item | What it asks | State |
|---|---|---|---|
| E57 | 18 | In E49's network with a limited input: the sweep with a scale for each receiving unit (at every map; at the cells only; a product form), and with the scale as before but without weight decay on `l1`; the scale in the network with limited connections and in the dense one. Nine runs, about 15 CPU-hours | Revision 3 after two checks (thirteen and six findings entered). Ready; no line |
| E53 | 9 | E47's network from a start with the recurrent weights times 3, with and without weight decay | Revision 2 after its check (fourteen findings). Ready; no line |
| E59 | 15 | E43's unrelated task from a fresh, a born and a lived network and from a born network with shuffled weights, from 2,048, 512 and 256 taught pairs | Revision 2 after its check (fourteen findings). Ready; no line |
| E55, first part | 11 | E51's arm `keyed` without the two lines of the program: the update gates of the reserve shut by a bias, and the rule extended to their rows. Eight runs of 16,384 updates, about 5 CPU-hours, with E51's runner | Revision 2 after its check (thirteen findings entered). Not released; see 33.5 and 33.7 |
| E58 | 19 | E48's gated term against free learning, with decay and the place of the change separated, at three taught fractions | Revision 2 after its check; running since 23:26 |
| E54 | 10 | The limited and the periodic network with fewer pairs taught | Revision 3 after two checks; running since 22:40 |

### 33.5 Two corrections of the reviewer's own statements

- **E50's network.** Revision 2 of the brief of E57 said that `C16-in` has a dense recurrent map and that in E50 "a code that cannot shrink lets the sweep learn the taught pairs". Both are wrong: the network has E49's limited connections, the score on taught pairs is 74.8 and never reaches 80, and E50's report declines to assign the gain to the size of the code. The second check found this; revision 3 is corrected.
- **E51's arm `open`, and the design of E55.** Revision 1 of the brief of E55 said that in that arm "the gates did not open" and that the new task was learned in the old cells. Its check (`COORDINATION_20261117/CHECK_OF_TASK_E55_BRIEF_20261118.md`) found this wrong for the unrelated task: 102; 75 of the 128 cells in reserve open by more than a half at some position of an item, although only 1; 8 are open by more than 0.1 on the average over positions, and with the reserve held shut the never-taught score is 0.7; 5.7 against 100. The sentence holds for the family modulo 97. The reviewer confirmed this from the receipts and by running the check's script. The same check emulated the brief's main arm in its own arithmetic and found that the construction does not keep the first family's answers exactly: the gates stay shut on the first family's items, but the weights from the cells in reserve into the old cells are free and multiply states of about 1e-3, so that 13 to 924 of the 31,684 answers change while the reduction score stays within 1.3 points; only the unrelated task under back-propagation is learned within 4,096 updates. The emulation is not a run of the runner and the reviewer has not repeated it. Revision 2 of the brief enters the check's thirteen findings and says this; among them the number of updates is raised to 16,384, because the gates must first be opened. The brief is not released.

- **The verification of E51's trial was called complete with six figures unfilled.** The agent that wrote it was interrupted, and the reviewer filled the running times and did not see that the last column of the table in its section 8.3, and one sentence, still held marks. The reviewer filled them on 18 November from its own run of the script: of the 340 never-taught items of the family modulo 97 whose answer differs from the first family's and whose operands are both below 89, the network with the reserve held shut gives the first family's answer in 291; 314 cases.

### 33.6 The workers' machines

Besides the stop of a machine about five minutes after a session's turn ends (section 32.6), the host has restarted machines while a turn was held open: E50's at 23:59 (two runs of stage 3 lost and started again), E54's at about 00:24 (three runs resumed from saved states) and the helper's at about 00:37 (75 to 80 visits of each life lost). The reviewer's four agents were stopped at about 00:19 by a limit of use of their model and were resumed on another model on Mr. Xu's word; their unfinished files had been committed, and nothing was lost.

### 33.7 Handover

Mr. Xu wrote at 00:44 on 18 November that you have returned and take over the coordination once the reviewer's present work is done. That work is: the two verifications recorded above, revision 3 of the brief of E57, the check and the revision of the brief of E55, and this memo. All four are done. The reviewer started no job after 00:44.

**The seven worker sessions.** Each was started by the reviewer from the default branch at 7cabb13a, pushes to its own branch only, and reports with `send_message` to the reviewer's session `session_01YBMaMvtGtQvDFcgppgThrZ`. The reviewer logs each message in `COORDINATION_20261117/LOG.md`.

| Line | Session | Branch | State at about 00:50 |
|---|---|---|---|
| The first long lives (E44 L2) | `session_017dSEpXu6oc38q4xQZ5sPv3` | `claude/e44-l2-first-long-lives` | three lives, uninterrupted since 22:07, at visits 741, 742 and 735 of 1,024 at 00:45; states pushed every 30 minutes |
| The sweep's long lives (helper) | `session_017dSEpTpaat6RxRKpJXsZWQ` | `claude/e44-l2-sweep-long-lives` | three lives, resumed from visit 640 at 00:38 |
| E50 | `session_017dSEpTbexj8Rn8nQ4eXUjq` | `claude/e50-fixed-input-code-017dSE` | stage 3, expected to end about 02:00 (at 00:47 back-propagation had passed 80 on R at about 16,000 updates; the sweep was at 55.0 after 17,408); then its final report; the line is then free |
| E51 | `session_017dSEpWQ9yfbN8dyC2VsWjy` | `claude/e51-reserve-017dSEpW` | finished: all 32 runs and 11 checks, report at 608f6b4c (a note added at 6ccaf1be); starts nothing unless asked; the line is free |
| E52 | `session_017dSEpVsT2w6DQAJDRGDa8c` | `claude/e52-learning-rates-sdga8c` | runs at rate 0.003 (three were killed by a restart at 00:44 and started again); the runs at 0.0003 follow, with a runner that resumes from saved states if its check passes; forecast 27 CPU-hours in all |
| E54 | `session_017dSEpX6GyjE567qTkELsQP` | `claude/e54-fewer-pairs-ELsQP` | part 1 of four; forecast 49 CPU-hours in all |
| E58 | `session_017dSEpVVK1y6oaJUKQasnme` | `claude/e58-gated-term-kqasnme` | Part A on seed 73129; two more seeds and Part B follow |

**Decisions taken in your place since revision 23,** each with its reason in the log (the earlier ones are in sections 31 and 32): E58 started after its check; E50's worker allowed 10.2 CPU-hours where its brief has 10, because of the lost runs; the other two seeds of E51's Part A not run now; no run of E50's `C16-in` from a start times 2; one run added to E57 (without weight decay on `l1`); item 11 of the plan begun with a first part that keeps E51's births, in place of a birth with E50's code.

**What waits for your decision,** in the order the reviewer recommends.

1. **What to do with E55's first part.** Its brief (revision 2) is ready and not released; its section "For the coordinator" gives the three things to know. The reviewer's recommendation is not to run it as it stands unless the result of the actual runner is wanted for the record (about 5 CPU-hours on E51's line), and to have one further arm specified and checked first: the cells in reserve made blind, in all their rows, to what the first family's items present, so that their states are zero on those items by construction and no gate has to be opened by learning. The stronger design, which the plan named, uses a code in which each byte has values of its own and needs a new birth. E51's worker has been told that the announced addition is withdrawn for now and that its next job comes from you.
2. **E59 on the line that E50 frees** at about 02:00 (21 CPU-hours): whether having learned the families saves samples on a task of other content (requirement B2).
3. **E53, then E57,** on the next lines that become free (requirement C).
4. **The other two seeds of E51's Part A,** together with those of E55 if its first seed is positive.
5. **Your ruling on E38's checks 2 and 3 under the sweep** (item 25) and on the limit of six lines.
6. **Whether the workers go on reporting to the reviewer's session.** If you prefer another arrangement, a message to each session is enough; their briefs name the reviewer's session.

**What the reviewer does from the handover on,** unless you arrange otherwise: it verifies each reported result from receipts and weights before the result is read (rule 4 of the plan), logs the workers' messages, answers a worker only where a run would otherwise be lost, and takes no decision in your place. Everything in `COORDINATION_20261117/` (the plan, the log, the briefs and their checks) can be changed by you.

### 33.8 After the handover: E51's Part B, verified

(`E51_PARTB_VERIFICATION_20261118.md`, `e51_partb_verify.py`; the reviewer ran the script again and obtained the agent's output line for line.) Part B asks how much rehearsal of the first family keeps it while E43's unrelated task is learned, in a network of the present size without a reserve; seed 73129. Every figure of the worker's tables (274 cells) and of its readings file (296 values) is reproduced from the receipts; the streams were built again and the batch digests agree at every scored row; the five predictions are scored by the rules fixed beforehand; two weight files, taken as a sample, reproduce their counts. Two figures of the report's prose are wrong and the worker has been told: the runs with rehearsal at intervals end after 2,640; 2,496 updates (the report has 4,616 and 4,448), and a share of 1/4 leaves the reduction score 7.5 points below its start under the sweep (the report has 7.6).

| Share of the stream given to the first family | 1/2 | 1/4 | 1/8 | 1/16 | 1/32 | none |
|---|---|---|---|---|---|---|
| Reduction score after 4,096 updates, as a percentage of that before | 100.5; 98.6 | 92.2; 90.6 | 89.8; 79.9 | 65.4; 58.6 | 45.8; 35.3 | 0.0; 0.0 |

- **The share.** Only a half keeps the first family within 5 points; the prediction that a quarter would (PB.1) failed. The unrelated task is learned at every share.
- **Rehearsal at intervals** (256 updates of the new task alone, then a review of the first family until 90% of its starting score on a probe, eight times). At the end of the first two gaps nothing of the probe is left; from the fourth cycle on, 46 to 78% of it is still right at the end of a gap (58.25; 72.5 at the eighth). The review needed falls from 80 updates to 32 under the sweep and to 48 under back-propagation.
- **Smaller steps** (rates 0.0003 and 0.0001, no rehearsal). The loss is slower in updates, but the first family is at 0 to 0.25% of the probe by the time the new task reaches 80%, at every rate.

Two sentences of the report say more than its figures. "Does not become cheaper to restore" under back-propagation is the brief's wording for a review that is not halved; every review after the first is shorter there (32 to 64 updates against 80), and the criterion is missed by one probe interval of 16 updates. "The explanation is refuted under both rules" (that rehearsal does not make the memory last) is right by the criterion fixed beforehand, but the first gap is also where the new task is first learned, so that the comparison of the first cycle with the eighth does not isolate what the earlier reviews did.

For requirement A, on one seed: in this network rehearsal keeps the first family only at half of the stream, repeated reviews make what is kept last longer across a gap, and smaller steps do not help.

### 33.9 After the handover: the trial of the first long lives, verified

(`E44_L2_TRIAL_VERIFICATION_20261118.md`, `e44_l2_trial_verify.py`; the reviewer ran the script again, 109 seconds, with no difference from the worker's figures in any part.) The trial is two lives of 1,024 visits (262,144 updates) in world 1 under back-propagation, seed 73129: one without rehearsal and one with a store that is replayed at a share of 0.5 (worker's branch at 118b6047). Every figure of the report, of its JSON, of its two CSV files and of its tables is reproduced from the event files and list files. The stream of world 1 was built again with the reviewer's reader and all 1,024 stream digests agree with both receipts. Each life was started again from the beginning at 22:07 and resumed from its state after 768 visits at 00:59; each event file is continuous across the resume, and the visits that were run before and after it give the same shares right. No stop condition holds, under either reading of the third condition, and the forecast for the roster is 48 CPU-hours.

| Quantity, world 1, 1,024 visits | Without rehearsal | With replay |
|---|---|---|
| Late level (share right over the last quarter of the life) | 0.669 (0.582 at 256 visits) | 0.730 (0.577) |
| Median level at the first visit of a family | 0.757 (0.653 over the first 256 visits) | 0.595 (0.393) |
| Trend of the first-visit level over the life | +0.106 (−0.006 at 256 visits) | +0.104 (−0.023) |
| Kept at a return after five or more absences | 0.015 | 0.472 |

Without rehearsal the commonest wrong answer at a return, in 42 to 48% of the answers, is the right answer under the modulus of the family taught in the visit before. Three sentences of the report say more than its figures and the worker has been asked to narrow them: that neither quantity of the table by number of earlier visits rises without rehearsal (the figure after the first 16 updates is higher at 16 or more earlier visits than at one, in all three classes of absence); that both rise with replay (they do, but the table does not separate the number of earlier visits from the age of the life); and that the branches after 1,024 visits are done (only their start records are on the branch).

For requirement B1, on one world and one seed: in both lives the median level at the first visit of a family is higher over the whole life than over its first 256 visits, the late level is higher at 1,024 visits than at 256, and the trend of the first-visit level, which was not above zero at 256 visits, is positive at 1,024. With one world there is no interval between worlds, so that the brief's words for the comparison of the arms (higher, lower, closed, still learning) cannot be given yet; worlds 2 and 3 are running.

### 33.10 After the handover: E50's stage 3 and its final report, verified

(`E50_STAGE3_AND_REPORT_VERIFICATION_20261118.md`, `e50_stage3_verify.py`; the reviewer ran the script again, with no error. Receipts and weights at 36aa8771; the reports as amended at 1799b2ba, between which only three report files changed.) Stage 3 is the setting `C46-one`: each cell reads 46 of the 512 values of the fixed code, one value of every used byte, so that it has an input for every byte; the connections between cells are limited as before and the readout is dense. All 140 figures of the final report's two result tables are reproduced from the receipts; the four saved weight files reproduce 2,400 of 2,400 recorded answers; the code, the byte table and `l1` are unchanged through both runs; every cell reads exactly 46 values, one of each used byte. Each completed run equals the run that the restart of the machine cut short on every scored row that both have.

| Setting (seed 73129) | Updates to 80 on R, back-propagation; the sweep | Levels of A and R at the end, back-propagation; the sweep |
|---|---|---|
| `C-dense`: every cell reads all 512 values | 6,912; 12,800 | 97.7, 97.1; 93.5, 94.1 |
| `C46-one`: 46 values a cell, one of every used byte | 15,360; 34,304 (2.22 and 2.68 times `C-dense`) | 93.8, 94.2; 85.9, 87.2 |
| `C16-in`: 16 values a cell, an input for 1 to 10 of the 20 needed bytes | not reached in 49,152; not reached | 83.8, 79.4; 49.5, 43.6 |

For requirement C, on one seed: with a limited input in which a cell still has an input for every byte, both rules learn the first family; the sweep needs 2.68 times the updates of the dense input and does not reach 80 within the 24,576 updates of the earlier experiments. When a cell has an input for only some of the bytes, neither rule reaches 80 on R within 49,152 updates. The report's sentence that "the number of bytes a cell has input for is what costs" is the form that the brief licensed and its condition is met, but the figures support less: `C46-one` itself costs 2.22 and 2.68 times `C-dense` and is far lower on the length test at 40 letters; the two limited settings differ also in the number of values a cell reads and in the starting scale of the input weights; and under back-propagation the two differ in speed near the threshold (levels of R 94.2 against 79.4). Four small figures of the final report differ in the last digit, and it lacks a dated section of amendments; the worker has been told.

### 33.11 The coordinator has taken over

The coordinator's decisions 1 to 12 are in `COORDINATION_20261117/COORDINATOR_DECISIONS_20261118.md` on the default branch (16b0e494). The reviewer forwarded its two messages to the workers of E50 and E51 and logs what follows from the decisions in `LOG.md`.

### 33.12 For the coordinator: E55 revision 4, and the clauses on saving and resuming

**E55** (decision 1). The brief is at revision 4 and waits for your release: `COORDINATION_20261117/TASK_E55_THE_FIRST_FAMILY_KEPT_WITHOUT_A_LINE_OF_THE_PROGRAM_THAT_LOOKS_AT_THE_ITEM_20261118.md`. Revision 3 specified the arm that you asked for, in which the cells in reserve receive nothing, in any of their rows, from what the first family's items present (the arm `blind`), kept the arm with gates shut by a bias beside it, and left out the third arm. Its check (`CHECK_OF_TASK_E55_BRIEF_REVISION_3_20261118.md`) found eight defects, all entered in revision 4, and emulated the arm `blind` in its own arithmetic to 16,384 updates. The emulation is not a run of the runner; the reviewer repeated its start and its first eight updates and obtained the same figures.

| Emulated run of the arm `blind` | New task, never-taught score at the end (with the cells in reserve held shut) | First family's scored answers that differ, of 32,884 | Of 1,200 held-out pairs written with 40 letters, answers that differ |
|---|---|---|---|
| unrelated, back-propagation | 100.0 (0.0), learned from update 2,048 | 0 at every scored row | 1,055 |
| unrelated, the sweep | 99.7 (0.0), learned from update 8,192 | 1 held-out answer at one row | 0 |
| the family modulo 97, back-propagation | 63.2 (28.9), not learned | 0 | 0 |
| the family modulo 97, the sweep | 66.6 (34.0), not learned | 0 | 0 |

What this indicates before any run: without a line of the program that reads the item, and without a gate that must first be opened, the first family's scored answers are kept and a task on other bytes is learned under both rules. Two limits appear. The cells in reserve, which cannot read an operand, raise the family modulo 97 by about 33 points and do not bring it to 80. And the keeping holds for the lengths that are scored and not for every length: learning makes the state zero of the reserve unstable, the remainder of the projection (about 1e-7) then grows from one position of an item to the next, and on items of 50 positions one of the four emulated runs loses the first family's answers. Revision 4 records the eigenvalue that decides this and a panel of 40 letters at every scored row, and words the result accordingly ("kept on items of up to 22 positions and not on items of 50"). The second part of item 11, a code in which each byte has values of its own, would give a cell in reserve exactly zero on the first family's items; that is the reviewer's reasoning and is not tested.

**E57 and E53** (decision 5). E57 is at revision 5 and E53 at revision 4. E57 has saving and resuming as in E52's runner, with a check in which a run killed after update 7,000 and resumed from the state of update 4,096 must equal an uninterrupted run; E53 had saving and resuming from its first revision, and its check now kills the run between its saved state and its end. The same agent checked both clauses (six findings for E57, one for E53; all entered).

**Lines** (decision 8, your request 4). At 03:52 on 18 November six lines run: the two long lives with their helper, E52, E54, E58, E59 (on E50's line) and Part B of E51 on two more seeds. None is free for E53. Reported and not yet verified by the reviewer: E58's Part A on one seed, E52's runs at two rates, E54's part 1, the helper's interim figures, E51's Part B on seed 93683; none of these is a finished job.

### 33.13 E51's Part B on three seeds, verified (the coordinator's decision 3)

(`E51_PARTB_SEEDS_VERIFICATION_20261118.md`, `e51_partb_seeds_verify.py`; the reviewer ran the script again, 55 seconds, without error. Worker's branch at 2abb0c3f.) Between the verified state of seed 73129 and this commit no file of that seed and no line of the report's earlier sections changed. All 28 receipts of seeds 93683 and 116089 are complete and record their settings; the streams were built again and the batch digests agree at every scored row; the worker's three-seed tables (312 cells) and readings (660 values) and every figure of the report's section 9 are reproduced; two weight files, taken as a sample, reproduce their counts. Six runs of seed 93683 were started twice by a fault of the queue; no file of an aborted start is among those that the tables read.

Reduction score of the first family after 4,096 updates, as a percentage of that before, with the share of the stream given to the first family (sweep; back-propagation):

| Seed | 1/2 | 1/4 | 1/8 | 1/16 | 1/32 | Least share that keeps it within 5 points |
|---|---|---|---|---|---|---|
| 73129 | 100.5; 98.6 | 92.2; 90.6 | 89.8; 79.9 | 65.4; 58.6 | 45.8; 35.3 | 1/2; 1/2 |
| 93683 | 100.3; 100.5 | 96.2; 94.9 | 94.5; 88.1 | 75.0; 85.3 | 68.4; 62.6 | 1/4; 1/4 |
| 116089 | 96.5; 105.0 | 92.7; 89.8 | 85.6; 87.9 | 71.0; 66.6 | 2.6; 44.2 | 1/2; 1/2 |

- **The share.** A half of the stream keeps the first family within 5 points on all three seeds and under both rules; 1/32 never does. A quarter keeps it on one seed, and there under back-propagation by 0.02 points (390 of 7,832 reduction pairs lost, where 391.6 is the limit), so that two more wrong answers would reverse the verdict; read as a percentage of the score before, a quarter keeps it on no seed under back-propagation. The prediction that a quarter keeps (PB.1) held on one seed of three.
- **Rehearsal at intervals.** On all six pairs of seed and rule more of the probe is right at the end of the eighth gap than at the end of the first, where nothing is left; the smallest gain is 57.5 points. The review needed is halved under the sweep on all three seeds and under back-propagation on one.
- **What it does not show.** One network, one birth for each seed and rule, one rate; no share between a half and a quarter; and the comparison of the first gap with the eighth does not separate the earlier reviews from the new task being learned by then.

For requirement A, with what the three seeds support: without an act of the program this network keeps the first family only with rehearsal, of between a quarter and a half of the stream; repeated reviews lengthen what a gap leaves, under both rules.

### 33.14 E58, verified: the gated term ends above free learning; the reason is not attributed; it reaches the same level from half the pairs under back-propagation only

(`E58_VERIFICATION_20261118.md`, `e58_verify.py`; written by an agent that had not seen the reviewer's reasoning; the reviewer ran the script again, 116.9 seconds, with the same output, and corrected three sentences of the agent's document, which are marked there. Worker's branch `claude/e58-gated-term-kqasnme` at e4c9bc0b; report `exp45/e58/E58_REPORT.md`.)

**The terms.** A lived network is one of the six networks at the end of E26's lives. The new family is add, sub, max and min under modulus 83, taught from a fraction f of its pairs of operands. *Free learning* (`free`) lets every weight change under the standing optimizer (decay 0.25); `free-wd0` has no decay; `free-anchor` decays the change toward the lived weights instead of the weights toward zero; `free-again` is `free` with the items in another order and measures the scatter of one arm. *The gated term* (`term`) freezes the lived network and learns a product of two matrices of rank 16 added to the two matrices of the recurrent layer, used only for items of the new family; the program decides this from the item's first byte. The *reduction score* is the percentage of untaught pairs of add and sub that need the reduction by the modulus and are answered right in every byte; *R-late* is its mean over rows 4,096 to 8,192; *the common pairs* are reduction pairs untaught at every fraction. A reading *above* needs 5 points on at least two seeds and no contrary margin on the third.

**What was verified.** All 72 receipts are complete and record the brief's settings; the streams were built again and their digests agree at every scored row of every run; the six runs that were started again after two restarts of the machine are whole runs and equal their interrupted receipts at every shared row. Of 483 figures and statements of the report, 475 are reproduced; the eight that are not are small (three bounds rounded the wrong way, one difference of 0.1, two phrases, and the sentence of finding 1 below) and change no reading. All 42 weight files and 48 term files match their receipts; the lived weights are unchanged in the term arms at every row; the network with the term was written again in numpy and reproduces the receipts' counts exactly on eight saved states.

**Part A, f = 0.10: R-late by arm** (seeds 73129, 93683, 116089).

| Arm | Sweep | Back-propagation |
|---|---|---|
| `free` | 79.0, 85.1, 83.5 | 83.0, 91.9, 89.1 |
| `free-again` | 74.7, 82.5, 83.55 | 83.4, 90.7, 91.8 |
| `free-wd0` | 76.7, 84.5, 83.5 | 83.5, 91.6, 93.7 |
| `free-anchor` | 84.7, 86.0, 85.8 | 85.4, 95.5, 94.6 |
| `term` | 92.0, 94.4, 87.2 | 94.2, 98.5, 99.0 |
| `term-wd0` | 91.6, 92.7, 86.9 | 94.6, 97.6, 98.75 |

- The gated term is above free learning at 8,192 updates under both rules (sweep +13.0, +9.3, +3.7; back-propagation +11.3, +6.6, +9.8), where the scatter of one arm is 4.2, 2.6, 0.0 and 0.4, 1.2, 2.6. The lead that E48 showed at 1,024 updates therefore persists.
- The lead is not attributed. The brief split it into a part that the absence of decay on the lived weights explains (`free-anchor` minus `free`: 5.7, 0.9, 2.2; 2.4, 3.6, 5.4) and the rest. The share of the first part is 0.44 and 0.10 under the sweep and 0.21, 0.55, 0.55 under back-propagation on the seeds where the lead is 5 points or more; the seeds disagree, and none of the brief's three sentences is stated. `free-anchor` and `free-wd0` are each not distinguished from `free` under either rule.
- In the term arms the earlier families are answered identically, item for item, at every row; in the three free arms the four panels of the life's families fall within 16 updates (seed 73129: from 86.6 to 60.9 under the sweep and from 90.5 to 56.6 under back-propagation).

**Part B: R-late on the common pairs** (sweep; back-propagation; three seeds each).

| f | `free` | `free-wd0` | `term` |
|---|---|---|---|
| 0.05 | 42.85, 50.48, 57.31; 55.58, 70.21, 75.27 | 75.88, 79.42, 76.39; 69.66, 89.57, 85.37 | 85.67, 83.83, 74.11; 88.56, 91.69, 93.90 |
| 0.10 | 79.32, 85.07, 83.69; 83.24, 91.76, 89.16 | 76.85, 84.60, 83.62; 83.86, 91.43, 93.72 | 92.07, 94.62, 87.16; 94.33, 98.50, 98.98 |
| 0.20 | 91.32, 92.42, 93.05; 92.46, 97.12, 95.27 | 92.44, 94.46, 93.59; 94.19, 96.68, 96.75 | 96.39, 95.63, 94.36; 97.80, 99.55, 99.82 |

- **From fewer pairs.** The brief licenses "the gated term reaches from f the level that free learning reaches from g" only if free learning gains from the additional pairs, the term at f is at most 2 points below the higher of `free` and `free-wd0` at g on two seeds and less than 5 below on the third, and the comparison is not at the ceiling. Under back-propagation this holds for (0.05, 0.10): the deficits are −4.7, +0.1, −0.2. On the second and third seed the condition is met by 1.9 and 2.2 points, which is the size of the scatter of one arm on that score (1.1 and 2.5 points); the scatter was measured at f = 0.10 only. For (0.10, 0.20) the comparison is at the ceiling, and for (0.05, 0.20) the condition fails (+5.6, +5.4, +2.9). Under the sweep no pair of fractions qualifies ((0.05, 0.10): −6.3, +1.2, +9.6), and neither does the sweep's term against back-propagation's free learning (−1.8, +7.9, +19.6).
- **At f = 0.05 free learning falls during the run** (reduction score at rows 1,024 and 8,192: 78.1 and 36.6, 70.95 and 44.2, 73.1 and 49.6 under the sweep; 71.8 and 45.5, 91.1 and 60.6, 87.3 and 78.4 under back-propagation). The term is above `free` there by 16 to 43 points under both rules, but under the sweep it is not distinguished from `free-wd0` (+9.45, +4.2, −2.5): much of that lead is the effect of the standing decay on free learning from few pairs.
- **In presented items the term is not faster.** Every run sees the same number of items (524,288); a smaller fraction means more presentations of each pair (381, 190 and 95). The term reaches 80 later than the free arm in 33 of 36 comparisons; it reaches 90 earlier, or the free arm never reaches it, in 16 of 36.

**Findings on the report** (eleven, sent to the worker for correction; none is an error in a table or a verdict). (1) "slower in presented items at every fraction" is stronger than the figures (33 of 36 at the level of 80; the reverse in 16 of 36 at 90). (2) Where the condition between fractions fails across the rules, the report states the contrary as a result; the brief allows only the deficits. (3) The limits do not set the scatter beside the margins that decide the one licensed sentence, nor say where the scatter was not measured. (4) Tables that the brief requires in the report are in committed text files instead. (5) One reading of Part B in support of the explanation was not fixed beforehand and is not marked so. (6) to (11) are matters of form and rounding.

**What it means for the requirements.** For B2 (a similar result from far fewer samples) E58 gives one licensed statement: under back-propagation, at an equal number of presented items, the gated term learns the new family from half the pairs to the level of free learning, on margins that two of three seeds do not separate from the scatter. It is not shown under the sweep, it concerns the number of distinct pairs and not the number of presentations, and it compares two ways of adding knowledge to one lived network, not our learner with a conventional one. For B1 (slower at first, ahead later) the same qualification holds: at f = 0.10 the term reaches 80 later than free learning on all six pairs of seed and rule (rows 256 to 512 against 128 to 256 under the sweep; 64 to 256 against 32 to 64 under back-propagation) and has the higher R-late on all six. For A the job says nothing: the gate is supplied by the program. The level of evidence is E2, on three seeds; there has been no independent audit.

### 33.15 E59, verified: a network that has learned before does not need fewer presentations for an unrelated task under the sweep; under back-propagation it does with 256 pairs only; no saving of pairs is shown

(`E59_VERIFICATION_20261118.md`, `e59_verify.py`; written by an agent that had not seen the reviewer's reasoning; the reviewer ran the script again, 64.7 seconds, with the same output, and found no sentence of the agent's document to correct. Worker's branch `claude/e50-fixed-input-code-017dSE`; report `exp45/e59/E59_REPORT.md` verified at 4594ffc9; the weights of all 60 runs are committed at f7314f4e.)

**The terms.** The *unrelated task* is E43's: an item is a code byte, two symbols out of 64, then 0 to 12 filler bytes and a query byte, and the answer is the two symbols; none of its bytes occurs in the items of the families of arithmetic. A *fresh* network has had no update; a *born* network has had the birth of E26 (24,576 updates on the first family); a *lived* network has had the birth and the life (61,440 updates on four families); `born-shuffled` is the born network with the values of every tensor but the byte table put in a random order, a control for the sizes of the weights. Each start learns the task from 2,048, 512 or 256 taught pairs for 8,192 updates of 64 items under each rule, on three seeds (60 runs). *Updates to 80* is the first scored row at which 240 of 300 held items are answered and stay so; one update is one presentation of a batch. By the brief's rule a start needs *fewer* presentations than another on a seed if its updates to 80 are below two thirds of the other's and at least 48 less, *more* in the reverse case, otherwise *about the same*; a comparison takes a word that two seeds give and the third does not contradict.

**What was verified.** All 60 receipts are complete, with the brief's settings, finite losses and no sign of an interruption; the streams were built again and the data digests agree at every one of 272 rows of all nine streams; 1,313 of 1,314 figures and statements of the report are reproduced (the one that is not is a count of files); all 60 weight files match their receipts; the forward computation was written again in numpy and reproduces the receipts' counts exactly on twelve networks; the shuffled networks were rebuilt from the brief's definition and match.

**Updates to 80 on the held panel** (seeds 73129, 93683, 116089; "–" where 80 is not reached in 8,128 updates).

| Start | Pairs | Sweep | Back-propagation |
|---|---|---|---|
| fresh | 2,048 | 192, 224, 208 | 192, 192, 192 |
| born | 2,048 | 400, 320, 320 | 224, 240, 208 |
| lived | 2,048 | 608, 448, 512 | 272, 272, 256 |
| born-shuffled | 2,048 | 256, 240, 240 | 192, 208, 208 |
| fresh | 512 | 256, 400, 288 | 288, 304, 272 |
| born | 512 | 512, 384, 368 | 256, 272, 224 |
| lived | 512 | 1,280, 736, 1,248 | 320, 320, 272 |
| fresh | 256 | 5,152, 6,656, 3,456 | 5,024, 3,392, 2,176 |
| born | 256 | 7,136, 4,544, 2,624 | 1,536, 1,120, 608 |
| lived | 256 | –, 7,680, – | 2,656, 3,360, 992 |

- **Under the sweep** a network that has learned before never needs fewer presentations than a fresh one. The born network needs *more* with 2,048 pairs and *about the same* with 512 and 256; the lived network needs *more* at every number of pairs and with 256 pairs does not reach 80 on two of three seeds.
- **Under back-propagation** the three starts are *about the same* with 2,048 and 512 pairs. With 256 pairs the born network needs *fewer* on all three seeds (ratios 0.31, 0.33, 0.28) and the lived network on two (0.53, 0.99, 0.46). This is the one saving the job finds; it is 1,568 to 3,488 updates, beside the 24,576 updates of the birth that preceded it. Its source is not named: the control with shuffled weights was run at 2,048 pairs only, where there is no saving.
- **No saving of pairs is shown under either rule,** because the fresh network also reaches 80 from 256 pairs on every seed; none is excluded below 256.
- **The sweep against back-propagation.** For the born and the lived network the sweep needs *more* presentations at every number of pairs; for the fresh network *about the same* with 2,048 and 512 and *more* with 256. In the case that bears on requirement B2, a born or lived network under the sweep against a fresh network under back-propagation, the word is *more* or *about the same* and never *fewer*.
- **The explanation under test** (that what a network gains from the families includes an ability, independent of their content, to hold two items across bytes that do not matter) is neither supported nor refuted by the brief's criteria: it is not supported because the born network is not *fewer* than the fresh one at 2,048 pairs under either rule; the refutation's conditions hold under the sweep and not under back-propagation.
- **Limits that the report did not state** (finding 1): three seeds; with 256 pairs the updates to 80 of one start differ between seeds by a factor of 1.9 to 3.4, and under the sweep the ratio of born to fresh is 1.39, 0.68, 0.76; two words rest on two seeds with a third that shows no difference; most curves with 256 pairs have not flattened.

**Findings on the report** (nine, sent to the worker; none changes a table or a reading). (1) The omitted limits above. (2) The list of committed files counts 179; 161 are E59's and 18 belong to other projects whose names contain "e59". (3) The report explains 283 against 282 held items at update 64 as two measures; they are two networks (E43's arm without decay and the arm with decay, which E59 reproduces), and on one network the two measures agree. (4) Check 3: the prescribed dry run forecast 29.6 CPU-hours, above the brief's limit of 28, at which the brief says to stop and ask; the worker made a second dry run (forecast 18.4; measured 18.53) and ran the 60 runs before the reviewer's answer reached it. The report should give P59.1 one outcome and say that the instruction, not only the wording, was departed from. (5) Check 5 has no receipt of its own. (6) to (9) are matters of form.

**A correction of the reviewer's own record.** The log (02:49 UTC) and the reviewer's message to the worker spoke of a formula of the brief that overstates the cost. The brief has no formula; it prescribes a dry run of 64 updates, two of whose five scored rows also score the eight panels of the earlier families, which a full run does at 2 of 273 rows. The formula is the runner's, and the dry run that the brief prescribed is what made it overstate. The defect is the brief's all the same, and it is the reviewer's.

**What it means for the requirements.** For B2 the result is negative under the sweep: earlier learning does not lower the number of presentations that a task of other content needs; the lived network needs more at every number of pairs and the born network more with 2,048 pairs (fresh, born, lived in that order on all three seeds with 2,048 pairs). Under back-propagation earlier learning lowers it only with the fewest pairs tested. The saving of pairs that E40 found for a new family of the same form (section 32.3) is not shown for a task of other content: the fresh network learns this task from the fewest pairs tested, so the question remains open below 256 pairs. On the sweep against back-propagation at an equal start, a trained network (born, lived or born-shuffled) needs more updates to 80 under the sweep in every cell of the table; the job does not address whether the sweep overtakes with continued learning (B1). For A the job says nothing. The level of evidence is E2, one task, one size of network, three seeds.

### 33.16 The sweep's long lives, verified: no overtaking is seen up to 1,024 visits; without the store the learner under the sweep ends lower; the comparison is between two learners and holds at one learning rate

(`E44_L2_SWEEP_VERIFICATION_20261118.md`, `e44_l2_sweep_verify.py`; written by an agent that had not seen the reviewer's reasoning; the reviewer ran the script again, 473 seconds, with the same output. Helper's branch `claude/e44-l2-sweep-long-lives`; report `exp45/e44/L2/sweep/E44_L2_SWEEP_REPORT.md` verified at 3879225d. The lives under back-propagation are the main worker's, whose own report is not yet written; only what this report takes from them is verified.)

**The terms.** A *visit* is 256 updates on one family of four operations under one modulus; a *life* is 1,024 visits in a *world*, which fixes the families and the order of the visits; there are four worlds on three birth seeds (worlds 1 and 4 start from the same births). In each update the network answers 16 items and then learns from 48. An *arm* is a birth together with the rule of the life: `naive` and `replay` start from the birth made by back-propagation and learn by back-propagation; `sweep` and `sweep+replay` start from the birth made by the sweep and learn by the sweep. The *store* is a buffer of 55,209 past items, the size of the network; `replay` and `sweep+replay` take half of each update from it. The *late level* is the share of answers right over the last quarter of the life. An *interval* is that of the roster note: the mean over the four worlds plus or minus 1.18 standard deviations; a difference is *lower* or *higher* when the interval excludes zero and *not distinguished* otherwise.

**What was verified.** The stream of each world was built again (16,777,216 items) and agrees with every visit's digest in all 16 receipts; every sweep life had the same stream, visit for visit, as the life under back-propagation it is compared with, and the store drew the same items under both rules. All 76,860 figures of the 32 score files are reproduced at full precision, and the tables are rebuilt character for character. Of 280 figures in the report's text, 271 agree; six are off by one in the third decimal through rounding twice, and three are wrong (two counts, and one value in the paragraph on the table by number of earlier visits). Three lives were resumed twice after restarts of the machine; the saved states have the recorded hashes, the visits computed twice give the same share right and loss, and the event files are continuous. No uninterrupted long life exists to compare hashes with.

**Late level** (at 1,024 visits; at 256 visits in parentheses).

| World | `naive` | `sweep` | `replay` | `sweep+replay` |
|---|---|---|---|---|
| 1 | 0.669 (0.582) | 0.650 (0.589) | 0.730 (0.577) | 0.784 (0.665) |
| 2 | 0.756 (0.696) | 0.628 (0.512) | 0.812 (0.755) | 0.574 (0.525) |
| 3 | 0.742 (0.692) | 0.683 (0.611) | 0.835 (0.707) | 0.789 (0.588) |
| 4 | 0.732 (0.685) | 0.672 (0.599) | 0.709 (0.659) | 0.769 (0.669) |

- **Without the store** `sweep` ends below `naive` in all four worlds: −0.066 [−0.120, −0.013] at 1,024 visits, *lower*; at 256 visits −0.086 [−0.178, +0.007], *not distinguished*.
- **With the store** no difference is distinguished: −0.042 [−0.206, +0.121] at 1,024 visits. `sweep+replay` is above `replay` in worlds 1 and 4 (+0.054, +0.061) and far below in world 2 (−0.238). The interval is 0.33 wide; it does not show that the two learners are equal.
- **Overtaking.** None is seen up to 1,024 visits. Over the first 64 visits `sweep` is 0.037 below `naive` [−0.091, +0.017]; at the end it is 0.066 below; in every world it lost ground between the two (−0.009, −0.030, −0.065, −0.013; mean −0.029 [−0.060, +0.001], *not distinguished*). The trend of the score over the second half of the life is steeper for `sweep` in three of four worlds (+0.073, +0.056, −0.059, +0.073), and the mean, +0.036 [−0.039, +0.111], is *not distinguished*. With the store nothing is distinguished (change +0.000 [−0.138, +0.139]).
- **All four arms are still learning at 1,024 visits:** the trend of learning is above zero for each (`sweep` +0.081 [+0.056, +0.106]; `naive` +0.098 [+0.083, +0.114]), and the late level is higher at 1,024 visits than at 256 in all sixteen lives.
- **What is kept.** Without the store neither learner keeps a family over an absence of 33 visits or more (0.005 to 0.020 of the answers at a return). With it `sweep+replay` keeps 0.340 to 0.509 and `replay` 0.446 to 0.599; the store raises what the sweep keeps by 0.43 [+0.34, +0.52].
- **A description that was not fixed beforehand** (`e44_l2_sweep_late_change.py`, the reviewer's arithmetic on the late levels above): between the readings at 256 and at 1,024 visits the difference `sweep` − `naive` moved toward zero in three of four worlds (−0.027, +0.056, +0.022, +0.026; mean +0.019 [−0.021, +0.060], *not distinguished*). The data therefore do not show the difference widening steadily either.

**Limits, of which the first two the report did not state** (findings 1, 2, 3, 6).
1. The two arms of a pair differ in the rule of the life and in the birth. The sweep's births answer less before the life begins (R 91.95, 92.55, 90.90 against 94.65, 94.25, 93.00). A difference between the arms cannot be assigned to the rule alone; the reviewer's own note on the calibration said so (`E44_GATE_FOR_THE_SWEEP_20261117.md`, section 7).
2. The one reading that excludes zero, *lower* without the store at 1,024 visits, treats worlds 1 and 4 as two replications although they share their births. With the two averaged as one birth the interval is −0.075 [−0.154, +0.003], *not distinguished*. That computation is not the roster note's definition; it shows how little the reading has in hand.
3. *Level*, *the same* and *not changed* in the report mean that an interval includes zero, with intervals 0.11 to 0.33 wide.
4. One learning rate, 0.001, taken from the literature for both rules and never varied in the lives. E52 (reported, in verification) bears on this.
5. Four worlds, three birth seeds, lives stopped at 1,024 visits while every arm is still learning; level E2.

**Findings on the report** (eleven, sent to the helper). Besides the limits above: three wrong figures and a sentence on the table by number of earlier visits that holds for one class of absence only; the direction of *grown* and *shrunk* undefined; the check of resumption described as covering the long lives, which it does not (it covers a life of eight visits); two matters of record and three of form.

**Corrections of the reviewer's own record.** (a) The rule for the report's section on the early level was sent to the helper by message on 17 November at 22:35 and was put in no committed file; it is now the appendix of the verification document. (b) That rule fixed the words *grown*, *not changed* and *shrunk* without saying which sign is which; for a negative difference *shrunk* then names a gap that widened. (c) *Not changed* states more than an interval that includes zero can give. The words are the reviewer's, and the helper applied them as instructed. (d) The report's phrase that the rate 0.001 "was chosen for back-propagation", which the verification lists as an error of fact, was the wording of the reviewer's instructions as first issued (3b4bdda5); the reviewer replaced it twenty minutes later (9eda3569), after the helper had started. The helper has corrected its report on all eleven points (738e2d61).

**What it means for requirement B1** (our learner may be slower at first and overtake later). On this evidence the learner under the sweep is behind from the first 64 visits on and does not overtake within 1,024 visits; without a store it ends lower in every world, by 0.02 to 0.13 of the answers. The evidence does not exclude overtaking later, because every arm is still learning and the intervals of the trends are wide; it does not exclude it at another learning rate; and it does not separate the rule from the birth. A statement that the sweep does or does not overtake would need longer lives, a comparison in which the birth does not differ with the rule, and the rate that E52 finds best for each rule. Which of these is run is the coordinator's decision.

### 33.17 E52, verified on one seed: at a quarter of the pairs part of the sweep's lag belongs to the learning rate; a lag remains everywhere when each rule runs at its best rate; several readings rest on one scored row

(`E52_VERIFICATION_20261118.md`, `e52_verify.py`; written by an agent that had not seen the reviewer's reasoning; the reviewer ran the script again, 20.7 seconds, with the same output. Worker's branch `claude/e52-learning-rates-sdga8c`; report `MOD97_LOCAL_RULE_20261106/E52_REPORT.md` verified at cb5cacfc. Every run is on seed 93683.)

**The terms.** E41's networks are trained on a fraction f of the pairs of operands, at the *half* width (256 cells) and the *present* width (512), under back-propagation (B) and the sweep (S). *R* is the share right on never-taught pairs of add and sub whose answer needs the modulus; *A* the same over all never-taught pairs of add and sub, with 12 letters. A *count* is the first scored row from which three rows in succession are at or above 80; rows are 256 updates apart. The *lag* is the sweep's count divided by back-propagation's. Every experiment so far used the rate 0.001; E52 ran 0.0003, 0.001 and 0.003, and 0.01 where 0.003 was best. By the brief's rule a count is *faster* below two thirds of the count at 0.001 and *slower* above 1.5 times it, and a lag at the rules' own best rates is *smaller* below two thirds of the lag at 0.001, *larger* above 1.5 times it, *the same* between.

**What was verified.** All 35 receipts are complete, with the brief's settings and finite losses; the eight runs at 0.001 equal E41's at every row and in their final weights, tensor for tensor; at every scored row all runs of a fraction have the same batch digest; 1,332 of 1,334 figures of the report and its tables are reproduced (the two that are not: a range of 37.3 points given as 30, and 1.80 given as 1.81); all 35 weight files match their receipts, and the forward computation written in numpy reproduces every count of the final row of three runs. Three runs at 0.0003 were resumed from saved states after a restart of the machine: their batch digests after the resumption equal those of uninterrupted runs, and the 9 to 12 rows that their logs hold both from before the restart and from after the resumption are identical.

**Counts on R (sweep; back-propagation)** ("–" where 80 is not reached within the dose).

| Width, f | 0.0003 | 0.001 | 0.003 | 0.01 | Lag at 0.001 | Lag at each rule's best rate | Reading |
|---|---|---|---|---|---|---|---|
| half, 0.50 | 32,000; 37,376 | 13,312; 8,704 | 7,424; 4,864 | –; – | 1.53 | 1.53 | the same |
| half, 0.25 | 54,272; 22,784 | 24,576; 7,424 | 11,520; 5,632 | –; 12,032 | 3.31 | 2.05 | smaller |
| present, 0.50 | 14,336; 12,800 | 4,864; 4,096 | 4,864; 2,816 | not run; – | 1.19 | 1.73 | the same |
| present, 0.25 | –; 12,288 | –; 4,352 | 5,376; 3,072 | –; 11,776 | above 2.70 | 1.75 | smaller |

- **The best rate among those tried is 0.003** in seven of eight cases of rule, width and fraction (the eighth, the sweep at the present width with half the pairs, is a tie of 0.001 and 0.003). Every run at 0.0003 and at 0.01 is slower than at 0.001, or does not reach 80.
- **The sweep gains more from the higher rate than back-propagation where a quarter of the pairs is taught:** at the half width its count on R falls to 0.47 of that at 0.001 and back-propagation's to 0.76; at the present width the sweep reaches 80 on R at 0.003 and at no other rate. There the lag is *smaller* at the rules' own best rates (3.31 to 2.05; above 2.70 to 1.75). With half the pairs it is *the same* (1.53 to 1.53; 1.19 to 1.73, the second because back-propagation gains and the sweep does not).
- **A lag remains everywhere:** the smallest at the rules' own best rates is 1.33 (on A, present width, a quarter of the pairs). It is gone in no case and larger in none.
- **The margins** (finding 1, which the report states for two readings only). The count of rows by which a single count would have to move to change a reading: *smaller* on R at the present width with a quarter, one row; *the same* at the present width with half, one row from *larger*; *smaller* on R at the half width with a quarter, two rows; "back-propagation does not gain" at the present width, one row at each fraction. The sweep's gains are firmer (6, 19 and 10 rows).
- **Against the scatter between seeds.** At 0.001 E41's three seeds give lags on R at the half width with a quarter of the pairs of 2.12, 3.31, 3.31: seeds differ at one rate by as much as the change that E52 measures on one seed.
- **The control** in which the product of rate and decay is held (decay 0.083 at the rate 0.003) is one observation outside the brief's fixed readings: there the sweep reaches 80 on R before back-propagation (5,632 against 6,912). It is not a result.

**Findings on the report** (twelve, sent to the worker; none is an error in a table or a reading). (1) The margins above. (2) P52.6, that the sweep's R is steadier at the smaller rate, is scored on rows at different numbers of updates; over the rows that end at 40,960 updates the run at 0.0003 is less steady (2.84 against 2.45). (3) The range given as 30 points is 37.3. (4) The costs are understated by what the logs and the child processes of check 4 add. (5) Three sentences say more than the figures, among them that the best rate "lies between 0.001 and 0.01", which four rates on one seed do not show. (6) to (12) are a bound rounded upward, a time, the order of a message and a restart, the scope of check 4, and matters of form.

**A correction of the reviewer's own record.** The reviewer's request that E52's worker add saving and resuming, on which three runs of the roster rest, was sent by message (17 November, 22:11) and put in no committed file; it is now the appendix of the verification document. With the rule for the helper's section on the early level (33.16) this is the second instruction that existed only in a message. From now on the reviewer commits the text of every instruction it sends that changes what a worker does, and gives the commit in the message.

**What it means for the requirements, and what is the coordinator's to decide.** For B1 the result bears on the limit that section 33.16 states, one learning rate: on this seed the sweep's lag at a quarter of the pairs is in part a matter of the rate, and at each rule's best rate among those tried the sweep still needs at least a third more updates than back-propagation (lags of 1.33 to 2.05). The brief's two conditions are met on this seed, as the worker states: the other two seeds are due at the half width with both fractions and at the present width with a quarter of the pairs; and, since the sweep gains at the half width, the question of the long lives at the rate 0.003 may be asked again. A birth made at another rate than 0.001 is not admitted to a life by the present runner, so that such lives need a decision on the births as well. Neither is started without the coordinator's word. The level of evidence is E2, on one seed.

### 33.18 The coordinator's decisions 13 to 17 were read ten hours late; what was forwarded, and the state of the lines

**The failure.** Your decisions 13 to 17 were committed on the default branch at 03:55 UTC on 18 November (39eacc33). The reviewer fetched the default branch last at 03:52 and then waited for a message relayed by Mr. Xu, which is how decisions 1 to 12 had reached it. It did not fetch the default branch again until 14:12, when Mr. Xu asked whether the work was ready to be sent to you. In that time it told Mr. Xu several times that E53 and E55 waited for your word, which was false from 03:55 on. E51's line, on which E55 was to start when Part B on seed 116089 was reported (04:16), stood idle for ten hours; the lines of E58, E59, the helper and E52 stood idle from 09:02, 09:29, 09:37 and 10:27. For the three jobs that were decided, about 20 line-hours were lost: 10.0 for E55, and, since E53 and E57 would have started when the lines of E58 and E59 fell free, 5.2 and 4.8 for them. The reviewer now fetches the default branch and reads the decisions file at every check of the lines.

**What was forwarded at 14:15 UTC** (the texts are in `COORDINATION_20261117/MESSAGES_TO_WORKERS_20261118.md`, committed at dfb7ad7b before they were sent). Decision 13: your message to E51's worker, word for word, with the statement that its condition is met. Decision 17: E53 (revision 4) to E58's worker, whose line was the first to fall free, and E57 (revision 5) to E59's worker, whose line was the next. The three briefs are the same files, blob for blob, on the reviewer's branch and on the default branch at 39eacc33. Because five lines were already free when E55 could at last be sent, "the first line that falls free after E55 has started" was read as the order in which the lines had fallen free; if you meant otherwise, the two workers can be redirected before their checks are done. The cap of decision 8 is kept: four lines run, and one finished report (E54) is in verification.

**What the reviewer owes under decisions 13 and 15:** the verification of the four runs of the arm `blind` before the arm `shut` is read, and after it a note of at most one page on how a later task that shares the first family's bytes is to be learned when the reserve receives nothing from those bytes.

**Decision 16** (the share of rehearsal on two seeds) is overtaken by section 33.13: the three seeds are verified, and the least share that keeps the family within 5 points is a half, a quarter and a half.

**The host.** By the default scheduling, which is what the reviewer used, a message to a worker is delivered only when the worker's turn ends (corrected in section 33.20: the message tool has a priority that joins a running turn). A worker that holds one turn open while its runs proceed receives nothing: E54's worker read the reviewer's messages of 02:15 and 08:06 at 13:52, E59's worker one of 02:49 at 09:29, and the helper those of 22:35 and 08:08 at 09:37. Two rules that the reviewer had sent by message alone (the reading of the early level for the helper, and saving and resuming for E52's worker) are now appendices of the two verification documents.

**The state at 14:15 UTC.** Verified and closed: E58 (33.14), E59 (33.15), the sweep's long lives (33.16), E52 on one seed (33.17). In verification: E54, whose report was committed at 13:51. Running: the first long lives (the short lives of worlds 5 to 8 and the lives without decay), and from now E55, E53 and E57. Free: the helper's, E52's and E54's lines.

### 33.19 E54, verified on one seed: neither the limited nor the periodic network generalises from fewer taught pairs than the dense network; seven of twelve readings are not decided

(`E54_VERIFICATION_20261118.md`, `e54_verify.py`; written by an agent that had not seen the reviewer's reasoning; the reviewer ran the script again, 77.0 seconds, with the same output, and found the numbers of the agent's main findings as printed. Worker's branch `claude/e54-fewer-pairs-ELsQP`; report `MOD97_LOCAL_RULE_20261106/E54_REPORT.md` verified at 87787a54. Every run is on seed 73129.)

**The terms.** *G* is the dense network of 256 cells; *F10* the limited network, in which each cell has 10 fixed connections in a neighbourhood of a grid of 16 by 16; *P8* the periodic network, a limited network whose cells update every 1, 2, 4 or 8 positions and whose connections are replaced at every 256th update. *f* is the taught fraction of the pairs of operands; *R* the share right on untaught pairs of add and sub whose answer needs the modulus. A curve *has flattened* when the mean of R over nine scored rows changed by less than 3 points over each of the last two spans of 8,192 updates. A network *generalises better* than G at a fraction only if both curves have flattened and its end level is above G's by a margin (10 or 20 points) taken from the spread of the dense network between seeds; where a curve has not flattened the reading is *not decided*. Each run has 49,152 updates.

**What was verified.** All 20 receipts are complete, with the brief's settings and finite losses; the batch digests agree across the runs of a fraction at every row; 1,289 of 1,295 figures are reproduced, and the six that are not are small (a time, three costs, one description of an earlier experiment, one rounded bound); all 20 final weight files match their receipts, the limited and periodic networks have no weight outside their connections, and the forward computation in numpy reproduces the final counts of four runs. The runner is rebuilt exactly from E42's by the committed list of differences. Six runs were resumed from saved states after two restarts of the machine; twelve rows printed before each stop and after the resumption are identical, and the batch digests after the resumption equal those of uninterrupted runs.

**End level of R at 49,152 updates** (the sweep; back-propagation), with whether the curve has flattened.

| f | G | F10 | P8 |
|---|---|---|---|
| 0.25 | 96.00 flat; 95.72 not flat | 82.77 not flat; 94.58 flat | 90.72 flat; 93.98 flat |
| 0.15 | 46.55; 72.98, neither flat | 29.65; 56.78, neither flat | 56.57; 72.92, neither flat |
| 0.10 | 15.08; 24.62, both flat | 9.35; 17.30, both flat | 19.07; 29.08, both flat |

- **No reading is "generalises better",** at any fraction under either rule, so the sentence that a network needs fewer samples is not licensed, and the brief's sentence for this outcome stands: on this seed no lowering of the needed fraction to 0.15 or to a tenth was found.
- **Of the twelve readings against G, five are decided and seven are not.** Decided, all as *not distinguished*: P8 under the sweep at a quarter (−5.3 against a margin of 10), and the four readings at a tenth, where no network generalises (differences −7.3 to +4.5). Not decided: all four at 15%, the fraction at which the dense network begins to generalise, and three at a quarter.
- **With a quarter of the pairs every run reaches 80 on R.** The limited network needs 3.65 times the dense network's updates to 80 on A under the sweep and 4.29 times under back-propagation; the periodic network needs about the same as the dense one under the sweep (1.21 on A, 1.13 on R).
- **Part 4** (the dense network at the half width with a tenth of the pairs, 122,880 updates): under back-propagation R rises from 49.9 at update 49,152 to 75.0 at the end, with a single row at 83.25, and does not reach 80 by the rule; under the sweep it ends at 13.4.
- **The limit that the report did not state** (finding 1): the readings at a tenth under back-propagation rest on two verdicts "flattened" whose larger rises are 2.78 and 2.46 points against the boundary of 3; one scored row 2.0 points higher would change the first. In part 4 the same rule called the dense curve flattened at update 78,592, after which R rose by 17.4 points. "Flattened" at a tenth is therefore not evidence that a level is final.
- **Two further limits** (finding 3): every margin comes from the spread of the dense network alone, and none is on record at 15%; and the taught pairs of the three fractions are not subsets of one another (of the 792 addition pairs taught at a tenth, 199 are taught at a quarter), so that a comparison across fractions changes which pairs are taught as well as how many.

**Findings on the report** (twelve, sent to the worker; none is an error in a table or a reading). (1) to (5): the limits above; the scope of the check of resumption, which was made on the periodic network under the sweep while the six resumed runs were G and F10; sentences that lack the seed and the spread the brief requires beside each difference; and the outcome "holds" of P54.8 without the count of readings not decided. (6) to (12): a time, a description, costs, and matters of form.

**What it means for the requirements.** For B2 (a similar result from far fewer samples) the result on this seed is negative for the two networks tested: neither a network with local connections nor one whose connections are replaced periodically learns the families from fewer taught pairs than the dense network. It does not show that they need more: at 15% nothing is decided. For C (local connections in place of full connection) the job adds that the limited network generalises from a quarter of the pairs under both rules, at three to four times the dense network's updates. The level of evidence is E2, on one seed and one width.

**Decision 22.** With E54 verified, the reviewer is to name the next item of the plan whose brief is checked. There is none: the three checked briefs (E55, E53, E57) are running. The brief of decision 21 is being written for the helper's line. For E54's line the items of the plan whose conditions are met and which have no brief are item 14 (widths: a birth narrower than the half width, a birth without decay at the half width, the limited network at the half width; it rested on item 8, now verified on one seed) and item 24 (new families for a learner that has lived; it rested on the long lives, of which the sweep's are verified and the first are still running). The reviewer writes neither brief unless you ask.

### 33.20 For the coordinator: the brief of decision 21 (E60) is ready for release; a correction on messages; the state at 14:47 UTC

**The brief.** `COORDINATION_20261117/TASK_E60_THE_SWEEPS_LONG_LIFE_FROM_BACK_PROPAGATIONS_BIRTH_20261118.md`, revision 2, with its check `CHECK_OF_TASK_E60_BRIEF_20261118.md` and `check_e60_brief.py` (the reviewer ran the script again, 9.7 seconds). It asks for three lives of the runner's existing arm `sweep-after-naive` (the birth of rule B, a life by the sweep) in worlds 1 to 3, 1,024 visits, with the settings of the eight lives of the sweep, about 12 CPU-hours on the helper's line. The runner and the scorer are not changed. Six checks precede the lives: the sources; the admission of the birth of rule B and the refusal of the sweep's birth; the calibration's life of this arm reproduced on the host; the same start as `naive`, item for item; a resumption after a kill with the lives' own settings; the cost. The reading is fixed for three pairs over the three worlds, at the first 64 visits, at 256 and at 1,024 visits: the rule with the birth held equal, the birth with the rule held equal, and both (the verified lives), with rules for the three sentences and for the word "overtaking".

**The check** by an agent that had not seen the reviewer's reasoning found seventeen defects, three of them of weight: a rule on messages that would have made the worker end its turn and so stop its own lives; a sentence on the change of the difference that was undefined when a difference changes sign; and a rule that could have assigned to the rule or to the birth a difference that is not itself distinguished. All seventeen are entered; one is entered in part, and the brief's history says why. The check confirmed every figure of the brief's record, that the arm can be run with the runner as it is, and that such a life and `naive` of the same world differ in the rule alone (the same stream at all 1,024 visits, the same settings, the same start).

**Three things about the design that are yours to weigh before release.** (1) With three worlds the interval is the mean ± 1.69 standard deviations; the verified difference between `sweep` and `naive` at 1,024 visits over these three worlds, −0.069 [−0.162, +0.024], is not distinguished by it, so the job may end without a distinguished reading. (2) The design has three of the four cells of rule and birth: a life by back-propagation from the sweep's birth cannot be run without one new arm in the runner. (3) The visit length of 256 updates was calibrated for `naive`; in the calibration the arm `sweep-after-naive` was inside the gate's band in 3 of 8 worlds, and it started 0.257 below `naive`, so the comparison is of the sweep on a birth it did not form.

**A correction on messages.** Section 33.18 and item 30 said that a message reaches a worker only when its turn ends. That is the default scheduling. The message tool has a priority that joins a running turn without stopping it; the check of the E60 brief pointed this out from the tool's own description. The reviewer had never set it, which is why three workers read messages hours late. From 14:47 UTC on 18 November the reviewer sends its messages to workers with that priority; it has not yet observed a delivery into a running turn.

**The state at 14:47 UTC.** Verified and closed: E58, E59, the sweep's long lives, E52 on one seed, E54 on one seed. In verification: the first long lives, whose report was committed at 14:43. Running: E55 (the arm `blind`), E53 (checks), E52's other two seeds, and E57 if its worker has begun (no message yet). Free: E54's line and the main worker's line; the helper's line waits for the release of E60.

### 33.21 The outside advice memo of 18 November: what the reviewer has done with it, and what is yours or the owner's (15:39 UTC)

**The document.** `PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/ADVICE_MEMO_20261118.md`, revision 3 (draft), on the branch whose name ends in `friendly-darwin-8mmhfm` at d0bc8a7d; written by an outside session at Mr. Xu's request. By its own statement it decides nothing, authorises no run, trained nothing and recomputed no figure. Mr. Xu asked the reviewer on 18 November to see which of its suggestions the reviewer can implement. The reviewer read its summary, its recommendations (sections 5, 6, 12 and 17) and its Part III; it did not read its argument (sections 3 and 4) or its reading of the older branches (section 11).

**The goal as Mr. Xu restated it to that session** (its section 14, given there as verbatim; Mr. Xu has not yet repeated it to the reviewer): (1) "a local learner with no back-propagation, and while some structural enhancements are possible, he prefers to let the model figure out where / how to store / update the memory as new information / feedback coming up. Also beating backpropagation has to be understood in epistiemological sense, namely for fixed computation power and accuracy demand, local learning can achieve at least the same generalization as back-propagation, perhaps with even less samples needed but may be with more time for digestion etc." (2) "Local in the sense that each neuron only gets info from its (sparse) connections, not the whole network, and each neuron adjusts the connection strength etc on its own." (3) "Local backward pass is fine, what prohibited is global backward pass."

**What this changes in how the running jobs are read, if Mr. Xu confirms it.** The outside memo's working reading is that a backward pass is local when a unit receives its error over its own sparse connections, so that the candidate is the sweep on the networks with limited connections and the dense network is the reference. On that reading the long lives and E60, which are run on the dense network of the half width under both rules, compare two rules; they do not compare the local learner of statement 2 with its reference. The jobs on networks with limited connections are E42, E47, E49, E50, E53, E54 and E57. The reviewer states this so that it is weighed when E60 is read; it changes no decision.

**The suggestions, one by one.**

| Suggestion of the outside memo | State |
|---|---|
| The entry points are out of date (its 5.5): `README.md` and `AGENTS.md` describe the state of 2 November | **Done by the reviewer as a draft:** `PROPOSED_ENTRY_POINT_UPDATE_20261118.md` in this folder gives the text for the head of `README.md` and one item for `AGENTS.md`. The reviewer checked the fault on the default branch at 392b3eca. You apply it; `AGENTS.md` is protected and needs the owner's authorisation |
| Find where the sweep's shortfall lies before it is read as the local learner's (its 17.1): the scale taken for a whole matrix, and feedback by signs against learned feedback, in the two cells where the sweep is furthest behind (a tenth of the pairs at length; the lived start of E59) | **Covered in part:** E57, running, tests the scale in the network with a limited input and in two others, with half the pairs. Learned feedback (E31, not replicated) and the two cells named are not covered. A job for your decision when E57 is read |
| Follow the arms without decay (its 17.2): one arm in which each cell holds the size of its own incoming weights; the same arms under the sweep | **Rests on a figure in verification:** that the conventional learner without decay and without a store begins to keep is the worker's statement; the reviewer says whether it stands when the verification of the first long lives returns. The arm needs a change of the runner; the sweep's lives without decay are the first half of the plan's item 23. Yours to decide |
| Births on 1, 4 and 16 families taught together, then the smallest change belonging to the new family alone that suffices to learn it (its 5.2; about 40 CPU-hours by its arithmetic) | **Not started.** A new job; it assumes an answer to the owner's question below. The reviewer writes and has checked the brief on your word |
| A first look on saved states: teach a probe family to the end state of a replay life by changing rows of the byte table only (its 5.2) | **Not the reviewer's:** it needs training, which the reviewer does not do. A worker's job, on your word |
| The world's headline: old families answered at a return before any teaching, as a curve against the length of the absence (its 5.4) | **The reviewer can do this read-only.** The roster's D8 gives it in three classes of absence (verified for the sweep's lives, 33.16). The finer curve, by length of absence, from the event files of the verified lives, follows the verification of the first long lives, for reasons of disk |
| A larger world, and families of both kinds of code (its 5.4 and 12.3) | Yours to decide; the present world has 52 families and one kind of code |
| Process (its 5.5): let the running jobs finish; do not work off the plan as a list; one rule for each exploratory run; eight or more seeds or worlds; the confirmation of E40 on a sealed family | The first two agree with your decisions 22 and 28. The others are yours; the sealed data are held by you (plan, item 17) |
| Controls and an objective for a rule of plasticity that prefers unused connections (its 5.3, 12.1, 12.2, 12.6) | For the time when such a job is specified; the outside memo itself puts 5.2 first |
| **A question for the owner** (its section 6): does a cue that is part of the item, its family code, count as the learner's own information? E48 and E51 treated a gate keyed on an item's first byte as an act of the program | **Put to Mr. Xu by the reviewer on 18 November.** The answer bears on how E48, E51, E55 and E58 are read against requirement A, and on whether its 5.2 is worth running |

**Two remarks on the outside memo's use of the verified results.** (1) Its table gives E54's part 4 under back-propagation as "75.0 and rising". By the brief's rule that curve has flattened (rises of +2.46 and −0.74 over the last two spans), and the verification's finding 1 is that such a verdict is not evidence of a final level; the last span does not rise. (2) Its figures of the first long lives are the worker's and are in verification, as it says.

**What the reviewer did not do:** it wrote no brief and started no job on the outside memo's proposals, and it changed no file outside its own folder.

### 33.22 Mr. Xu's answers of 18 November to the two questions of 33.21 (15:44 UTC)

**His words**, to the reviewer, in answer to the two questions (the original is Chinese; the English is the reviewer's):

1. "所有它自己学到的编码都不算是我们给它的信息。" Every code that it has learned by itself does not count as information that we have given it.
2. "对single pass global backpropagation是不可以的，但是可以通过neuron之间的relay传递信息。换句话说1->50->50^2->50^3等等，可能几个pass下来就能够覆盖整个网络。" A single-pass global back-propagation is not allowed, but information may be passed by relay between neurons. In other words 1 → 50 → 50² → 50³ and so on: after a few passes the whole network may be covered.

**The reviewer's reading, which is the reviewer's and open to correction.**

- *On the code.* The line between the learner's information and ours runs between what the learner has formed and what the program does. The rows of the byte table and any code in the cells that learning has formed are the learner's own, also when they stand for the byte that names a family. A line of the program that reads a byte of the item and switches something (E48's gate; the two lines of E51's arm `keyed`) is ours, and so is anything the program computes at a change of task (E55's directions and named rows). The readings of E48, E51, E55 and E58 against requirement A therefore stand as they are. What changes is the outside memo's proposal 5.2: a new family held in the rows of its own code bytes, reached through the ordinary forward pass, would be kept by the learner's own means. Its limit, which the outside memo states, remains: the item marks the family with a byte of its own, and a positive result would not show that a learner finds its cue in a stream that does not mark it.
- *On locality.* What is prohibited is that one pass brings to a unit information from the whole network. What is allowed is that each pass moves information one step along a unit's own connections, of which the example has about fifty, so that a few passes reach every unit. (The first text of this point ended by saying that the definition leaves open whether the exact gradient, relayed along sparse connections, is allowed, and that the reviewer had put this to Mr. Xu. That was an error of the reviewer's, corrected at 15:52 UTC; see the paragraph that follows the list.)

**The principle, and the reviewer's error** (added at 15:52 UTC). Mr. Xu answered the reviewer's remaining question in one sentence: the matter is clear from biology, and the reviewer has been confused on it repeatedly. He is right, and the definition is to be read from that origin and not parsed as a specification. A neuron has what arrives over its own connections, forward or back, and its own state; it changes its own connections from these. It does not know the weights of other neurons, and nothing delivers to it a quantity computed from the whole network. The exact gradient needs both, on any network, sparse or dense; it is therefore excluded, and the question whether it would be allowed along sparse connections does not arise. Relay is what neurons do: a signal passed on over real connections, step by step, by units that each act on what they receive. The cause of the repeated confusion is the reviewer's own framing. It carried the record's two experimental factors, the rule (back-propagation or the sweep) and the connections (dense or limited), as if they were independent choices, and read each statement of Mr. Xu's as a clause to be mapped onto those factors; it then asked him to close gaps that exist only in that framing. Requirement C in this memo and in the plan ('local learning in place of full connection, also at input and output') is the product of the same framing and is too thin. From here on the reviewer reads every job against the principle above. One consequence inside the candidate itself: the sweep takes one scale for a whole matrix, the mean size of all its weights, which no single unit has; the scale for each receiving unit, which E57 is testing, is the form that a unit can hold.

**What follows, for your weighing; no decision is changed and nothing is started.**

1. The dense network is outside the definition under either rule, because in it one pass reaches every unit. The long lives and E60 are on the dense network of the half width; they compare two rules and remain worth reading as that, and they are not a test of the local learner as now defined. The outside memo's working reading (33.21) agrees with this.
2. The networks inside the definition are those with limited connections: 10 connections a cell in a neighbourhood (E42, E54), 16 inputs a cell (E49, E50, E57), the network with an interface at a boundary (E47, E53). None has about fifty connections a unit, and in none has the number of steps that the error is relayed been varied.
3. The plan's item 20 (the error passed back over a limited number of positions instead of the whole item, under both rules) is the experiment that measures how many steps of relay are needed. It has had no priority; Mr. Xu's second statement makes it the direct test of his picture, on a network with limited connections.
4. Item 23 and E60 keep their place for the comparison of the rules; a statement on requirement B1 for the local learner needs a life on a network with limited connections, which does not exist yet.
5. The outside memo's 5.2 is worth its cost on Mr. Xu's first answer; its brief is the reviewer's to write on your word.

### 33.23 A review at Mr. Xu's request: the experiments set against the principle (15:59 UTC)

Mr. Xu, 18 November: he had asked for the experiments to be arranged step by step from the biological principle; there now seem to be problems; where do we stand? This section is the reviewer's answer. It is a proposal and decides nothing.

**The principle** (33.22). A neuron has what arrives over its own sparse connections and its own state, and changes its own connections from these; nothing delivers to it a quantity computed from the whole network; a signal is relayed step by step. Long-term memory is to form in the course of learning and not by an act of the program. The learner so defined is to reach, at equal computation and accuracy, at least the generalisation of back-propagation, perhaps from fewer samples and with more time.

**What went wrong, and whose fault.**

1. The candidate was never defined once from the principle. In practice "our learner" meant the sweep, and it was run mostly on the dense network, in which one pass reaches every unit and which is therefore outside the definition.
2. The reviewer's plan of 17 November is a list: an inventory of 107 untested entries in 29 items, ordered by requirement. It is not a chain in which a step is taken only when the step before it holds and in which an outcome decides what comes next. That is the reviewer's fault. The outside memo says the same of it (33.21).
3. The three requirements were taken apart and each was tested on the dense network: memory in a network in which every item changes every connection; overtaking between two rules on that network.
4. Parts of the sweep itself do not meet the principle and were not treated as a question until E57: one scale for a whole matrix is a quantity that no single unit has.

**The steps that the principle gives, and where each stands.**

| Step | Question | Where it stands |
|---|---|---|
| 1 | Does a network with sparse recurrent connections learn and generalise under a local rule? | **Yes, more slowly.** With 10 connections a cell and half the pairs taught, updates to 80 on A are 2.36 times the dense network's under the sweep and 1.70 times under back-propagation (E42; quoted in 33.19's verification). With a quarter of the pairs, 3.65 and 4.29 times (E54, one seed) |
| 2 | Does it learn when the input is sparse as well? | **Not passed.** With 16 inputs a cell the sweep does not learn (E49). With a fixed code it learns only when each cell has an input for every byte, and then needs 2.68 times the updates that it needs with the whole code and 2.23 times the exact gradient's in the same network (E50; corrected on 18 November after the outside audit, which found that 2.68 had been given as the lag behind back-propagation). E57, running, asks whether the scale is the cause; at update 8,192 all three scales are still low, and the worker makes no reading |
| 3 | Over how many steps must the error be relayed? | **Never run.** The plan's item 20, without priority until 33.22 |
| 4 | What changes at about fifty connections a unit? | **Never run.** The networks have 10 or 16 |
| 5 | Where is an experience written in a sparse network; are there cells not in use that a new task can take? | **One read-only measurement:** in the network with an interface at a boundary, the cells away from the boundary are not in use (plan, item 2). No second task has been learned in such a network (E56, not briefed) |
| 6 | A life on a sparse network, beside back-propagation with a store on the dense one at equal memory | **Never run** |

**What the dense network has given, as reference** (verified; these are statements about the rule and about the conventional learner): without a store almost nothing is kept at a return, and a store of the network's size keeps about half; the sweep on the dense network ends lower without a store and does not overtake up to 1,024 visits; under the sweep a network that has learned before is not faster on a task of other content (E59); the conventional learner learns new families faster as it ages (in verification; the agent's document confirms it and the reviewer's own run is not finished).

**The order the reviewer proposes.**

1. Let the running jobs finish, and add no job on the dense network. E60 and E55 are read as reference.
2. Define the candidate once: sparse connections, and only quantities that a unit has. E57 decides how the scale is taken.
3. Pass step 2. If the scale does not do it, the next candidate cause is the feedback (by signs against learned feedback; E31, not replicated).
4. Take step 3 on the sparse network: the plan's item 20.
5. Only when steps 2 to 4 hold: memory on the sparse network (cells not in use; the learner's own code, the outside memo's 5.2), and last the life of step 6.

### 33.24 Where the sweep came from, at Mr. Xu's question (16:03 UTC)

Mr. Xu, 18 November: "I never defined the sweep rule. Where did it come from? Why was this experiment done?" The reviewer answers from the record; every quotation below was read again for this section in the review for Mr. Zhou on the default branch at 392b3eca and in the reviewer's own note of 10 November.

**The record.**

1. **6 November** (review, sections 9.1 and 9.2). At Mr. Xu's request the project's own local rule of that time, V181's direct random feedback, was run beside back-propagation on modular addition in a feed-forward network: 70 to 84% of never-seen sums at its peak against 95 to 99%. The review then tried "sign-symmetric feedback, which the clarified constraint permits": each layer receives the error through the signs of its own forward weights. It reached 95 to 98% at backprop's speed, and the review wrote: "under the clarified constraint this is a local rule, and it is the one to carry forward." The constraint as read then was that anything is allowed except exact back-propagation and attention.
2. **Early November, the recurrent networks.** When the learner read an item byte by byte without a window, the same idea was applied backward over the positions of the item (the review calls it "the Phase 4 rule"). It was the only rule on file that trained such a learner (0.92 to 0.95 of the reference); a forward-only form reached 0.63 to 0.79 at about twice the cost (E19).
3. **10 November, the reviewer's proposal** (`NEXT_STEPS_NOTE_20261110.md`, section 3.1). The reviewer suggested: "A backward sweep over the item with weight-sign feedback (the Phase 4 rule): accept it as the working life rule, with the sweep length a declared budget." It wrote beside it: "The life rule is then backprop through time with weight signs in place of weights. What stays outside the line is exact weight transport," and "The rule can no longer be the project's novelty."
4. **10 November, the adoption** (review, section 9.82 and its addendum). The coordinator adopted this as a working rule "on my reading of Mr. Xu's instruction" (his instruction had been: "We received the next step note. Now implement the steps"), asked Mr. Xu whether the three working rulings were his, and recorded the answer: "Mine, they stand." A later section of the review says that the backward sweep "rests on Mr. Xu's working ruling (addendum to 9.82) and on nothing else."

**What this amounts to.** Mr. Xu did not define the rule. It was chosen because it worked under a loose reading of his constraint; the reviewer proposed it as the working rule; and it was put to him for confirmation as one of three items in one question. From that day "the sweep" stood for "our learner" in every brief, and the name hid what the reviewer had itself written of it: that it is back-propagation through time with the signs of the weights in place of the weights. The reviewer recommended it and bears that part of the fault.

**Its standing against the principle of 33.22.** Three things in it are not what a neuron has: the error returns over the forward connection with that connection's sign; one scale is taken for a whole matrix; and the pass runs backward over the positions of the item after the item has ended, which needs the past states of every unit. On the dense network it is, besides, a pass that reaches every unit. The record holds rules that are nearer to the principle and weaker so far: direct random feedback (item 1), the forward-only form (item 2), and learned feedback (E31, not replicated).

**What is open.** Whether the sweep remains the candidate, in which form and on which network, is Mr. Xu's and yours to rule. The reviewer changes no running job on this account. E57 (the scale for each receiving unit) is the one running job that tests a part of the rule itself.

### 33.25 The first long lives, verified: the conventional learner keeps learning to 1,024 visits and learns new families faster as it ages; a store adds what is kept, not level; several readings count worlds that share a birth

(`E44_L2_VERIFICATION_20261118.md`, `e44_l2_verify.py`; written by an agent that had not seen the reviewer's reasoning; the reviewer ran the script again, 593 seconds, with the same output. Main worker's branch `claude/e44-l2-first-long-lives`; report `exp45/e44/L2/E44_L2_REPORT.md` verified at 45e9ba16. This is the reference line: the dense network under back-propagation.)

**What was verified.** All 32 lives and 24 probe branches are complete with the prescribed settings; all 14,336 visit streams were regenerated and agree; 212,218 figures of the 40 score files and of the final files are reproduced with none differing, and the 164 rows of the tables are rebuilt verbatim; of 410 figures in the report's text three differ (two rounded twice, one cost). The lives interrupted by three restarts were resumed from states whose hashes match or were run again; the visits computed twice agree.

**The readings** (four worlds with three births at 1,024 visits; eight worlds with three births at 256 visits; the rate 0.001; intervals over worlds by the roster note).

- **Still learning.** In each of the four long worlds and in both arms the level that a family reaches at its first visit rises with the number of families met before (trend of learning +0.082 to +0.111 without the store, +0.091 to +0.145 with it). This reading survives counting each birth once.
- **New families are learned faster with age (the probe; decision 28).** Families that no life has met, taught for one visit from the states saved after 64, 256 and 1,024 visits: the first-visit level rises from 0.427 to 0.826 without the store (+0.399 [+0.228, +0.570]; by world +0.332, +0.617, +0.333, +0.313) and from 0.267 to 0.743 with it (+0.476 [+0.396, +0.557]). It rests on 24 branches and cannot separate the age of the network from the number of families met or from the growth of its weights. No such probe exists for the sweep's lives.
- **The store.** At a return after five or more visits the arm with the store answers 0.537 right on the first update against 0.011 without it (+0.526 [+0.434, +0.618], in every world). In late level the two are not distinguished (+0.047 [−0.011, +0.105]); the interval admits a gain of up to 0.105, so this is not a finding of equality. The store lowers the first-visit level of a new family (−0.128 [−0.201, −0.055] in the median).
- **Without the store** almost nothing is kept after any absence of two or more visits (0.000 to 0.027), and the commonest answer at a return is the right answer under the modulus of the family taught in the visit before (0.499 of 12,683 answers; chance 0.007).
- **Without weight decay** (read to 256 visits only): what is kept at a return without the store is higher but small (after 5 to 32 visits +0.075 [+0.045, +0.105], in all eight worlds); the late level is lower (−0.095 [−0.166, −0.024]) and so is the trend of learning (−0.119 [−0.161, −0.076]); the weights grow throughout (from 52.8 to about 126). The outside memo's sentence that the conventional learner "begins to keep" without decay (33.21) stands in this form: a small amount, at a stated price, to 256 visits.

**The weightiest finding** (of nine, sent to the worker). The eight worlds at 256 visits come from three births and the four long worlds from three, and the report does not say so. With the worlds of one birth averaged into one value, these readings no longer exclude zero: the lower late level of both arms without decay (−0.110 [−0.306, +0.086] for the arm without the store), the higher kept at return after 33 visits or more without decay (which rests on 18 right answers of 471 against 4), and "still learning" for the arm without the store at 256 visits. The readings that survive are the store's effect on what is kept, the lower trend of learning without decay, and both long arms still learning at 1,024 visits. The other findings: the check of resumption quoted for more than it shows; a table of the wrong answers by class of absence said to exist that does not; the limit on the table by number of earlier visits, which omits two arms (under the arm without the store the rise of the figure over the first 16 updates almost vanishes in the second half of the life: 0.118 to 0.123); required statements and limits that are missing; and matters of form.

**For the requirements.** These are statements about the conventional learner on the dense network, the reference. For B1 they set what the local learner is to be measured against over a life: a learner that is still improving at 1,024 visits and whose first-visit level on a new family about doubles between the ages of 64 and 1,024 visits. For A they confirm that without a store nothing is kept, and that a store of the network's size keeps about half and does not raise the level. The level of evidence is E2.

### 33.26 E55, the arm `blind`, verified on seed 73129: the first family's scored answers are identical with a forward pass that reads no item; on items of 50 positions this fails in one run and is unbounded in all four

(`E55_BLIND_VERIFICATION_20261118.md`, `e55_blind_verify.py`; written by an agent that had seen no opinion of the result; the reviewer ran the script again, 250 seconds, with the same output. Worker's branch `claude/e51-reserve-017dSEpW` at fb0ec89a; brief revision 4; the reading is the one the coordinator fixed in decision 14, and nothing wider. The network is the dense one: a statement about the construction, not about the local learner. The arm `shut` is not read here.)

**What was verified.** All 1,227 figures of the worker's table, reading and message are reproduced from the receipts' counts, none differing. The twelve saved networks reproduce, in numpy with the plain forward pass, every scored answer of the first family, every count of the new tasks' scores, the eigenvalues and the constraints of the arm. The runner's forward pass in this arm contains no line that reads the item (the one such line of E51's forward stands under a condition that is off, and the run asserts it).

**The result, in the brief's words.**

| Run | First family's 32,884 scored answers (items of up to 22 positions) | 300 answers at 40 letters (50 positions) that differ | New task | Never-taught at 16,384: as run; with the reserve shut |
|---|---|---|---|---|
| `unrelated`, sweep | identical at every row | 0 | learned from row 8,192 | 99.0; 0.0 |
| `unrelated`, back-propagation | identical at every row | 21 at row 2,048, 278 at 16,384 | learned from row 2,048 | 100.0; 0.0 |
| `conflict`, sweep | identical at every row | 0 | not learned | 69.4; 34.9 |
| `conflict`, back-propagation | identical at every row | 0 | not learned | 63.2; 28.9 |

A task on bytes that the first family does not use is learned, under both rules, by cells in reserve that receive nothing from the first family's items, and the first family's scored answers do not change; it is learned three doublings of the number of updates later than in E51's `keyed` under the sweep (8,192 against 1,024) and two later under back-propagation (2,048 against 512). A task on the first family's own bytes is not learned within 16,384 updates; the cells in reserve, which can then read only the operation byte and their own states, raise its score by about 34 points. Predictions P55.1, P55.9, P55.10, P55.12 and P55.13 held; P55.11 (the state of a cell in reserve below 1e-4 at every row) failed in `unrelated` under back-propagation.

**The findings that limit it** (sent to the worker for its table and final report).

1. *On items of 50 positions the first family is lost in one run.* In `unrelated` under back-propagation the right answers on the 40-letter panel fall from 274 of 300 to 21, and the state of a cell in reserve on those items reaches the limit of the hyperbolic tangent. The worker gave only the number of answers that changed.
2. *Nothing bounds the reserve on longer items.* The weights among the cells in reserve are free. The largest eigenvalue of the map by which a small state of the reserve passes from one position to the next is above 1 at row 16,384 in all four runs (1.10, 1.43, 1.06, 1.20), and the state on the 40-letter items grows in all four. "Kept exactly" is an observation on items of up to 22 positions.
3. *The counts on long items depend on the arithmetic.* What reaches the reserve on a first-family item is a rounding remainder, multiplied at each position: the same weights give 45 (receipt), 15 (double precision) or 142 (single precision in numpy) changed answers at row 4,096. The brief's rules give the same outcome in each.
4. The wording the brief fixes for `unrelated` under back-propagation is "kept on items of up to 22 positions and not on items of 50" alone; outputs move by more than the smallest lead without a changed answer on the scored panels; two comparisons the brief asks for were missing; matters of provenance.

**For requirement A.** By decision 14 the result does not meet it, and the verification agrees: the four acts of the program remain (directions computed at the change of task, rows named, the reserve added at that moment, the change of task known). Two parts of them should be named where the acts are listed: the record of the bytes in use, by which the program restores rows of the byte table after every update, and the single use of the keyed forward pass at the change of task to compute the directions. What the arm adds to E51 is narrow and exact: the line that read the item at every position is not needed for a task on other bytes, if the reserve is made blind to the first family's directions by the program; the price is a slower new task, no learning on shared bytes, and no bound on long items. The note that decision 15 asks for (how a later task on the first family's bytes is to be learned when the reserve receives nothing from those bytes) follows from this section and is owed next.

### 33.27 What is answered at a return against the length of the absence (read-only; decision 35, the outside memo's 5.4)

(`absence_curve.py`, 2 seconds; it reads the score files of the lives, whose every figure the two verifications recomputed from the event files: the first long lives at 45e9ba16, the sweep's at 738e2d61. Nothing was run. The dense network; four worlds from three births at 1,024 visits, eight worlds from three births at 256 visits; shares pooled over worlds, with the smallest and largest world in the script's output.)

*Kept* is the share of a family's items answered right at the first update of a return, before anything of the visit is learned; *first 16* is the mean share right over the first 16 updates of the return; the *absence* is the number of visits since the family's previous visit, so that 1 means that nothing came between.

| Absence (visits) | Returns | back-propagation: kept; first 16 | the same with the store | the sweep: kept; first 16 | the same with the store | back-propagation without decay, 256 visits: returns; kept; first 16 |
|---|---|---|---|---|---|---|
| 1 | 627 | 0.801; 0.801 | 0.751; 0.755 | 0.759; 0.753 | 0.692; 0.711 | 568; 0.662; 0.652 |
| 2 | 444 | 0.023; 0.187 | 0.581; 0.657 | 0.021; 0.139 | 0.545; 0.607 | 357; 0.186; 0.367 |
| 3 to 4 | 613 | 0.011; 0.132 | 0.576; 0.639 | 0.017; 0.103 | 0.526; 0.576 | 381; 0.140; 0.304 |
| 5 to 8 | 306 | 0.016; 0.117 | 0.561; 0.635 | 0.011; 0.091 | 0.500; 0.561 | 207; 0.115; 0.260 |
| 9 to 16 | 394 | 0.015; 0.112 | 0.517; 0.598 | 0.017; 0.089 | 0.406; 0.529 | 163; 0.098; 0.217 |
| 17 to 32 | 510 | 0.008; 0.105 | 0.550; 0.610 | 0.011; 0.090 | 0.467; 0.547 | 146; 0.066; 0.188 |
| 33 to 64 | 599 | 0.010; 0.100 | 0.510; 0.590 | 0.014; 0.085 | 0.442; 0.525 | 121; 0.038; 0.157 |
| 65 to 128 | 269 | 0.012; 0.111 | 0.554; 0.624 | 0.016; 0.086 | 0.456; 0.560 | |
| 129 to 256 | 113 | 0.009; 0.091 | 0.533; 0.623 | 0.014; 0.077 | 0.449; 0.551 | |
| 257 or more | 18 | 0.018; 0.115 | 0.596; 0.692 | 0.000; 0.091 | 0.491; 0.603 | |

Three statements, each for these lives and no further.

1. **Without a store and with weight decay, forgetting is a step and not a curve.** One visit to another family (256 updates) takes what is answered from about 0.8 to about 0.02, under both rules, and no longer absence takes more: the slope of kept on the logarithm of the absence, from 2 visits on, is −0.001 a doubling under both rules. What a longer absence still changes is the speed of relearning: the first 16 updates fall from 0.187 to about 0.10 under back-propagation and from 0.139 to about 0.08 under the sweep.
2. **With the store, what is kept hardly depends on the absence.** It falls from 0.58 after 2 visits to between 0.51 and 0.60 after 33 or more under back-propagation (−0.009 a doubling) and from 0.55 to between 0.44 and 0.49 under the sweep (−0.017 a doubling). A store of the network's size therefore keeps a fixed part, about half, whatever the absence up to 256 visits.
3. **Without weight decay the conventional learner shows a graded curve.** Kept falls from 0.186 after 2 visits to 0.038 after 33 to 64 (−0.034 a doubling), and the first 16 updates from 0.367 to 0.157. This is the only arm on file in which something of a family outlasts the next family's visit without a store, and it lasts some tens of visits. It is read to 256 visits only, on eight worlds from three births, and it comes with the lower level and the growing weights of section 33.25.

Limits: the classes pool worlds that share births; a return's absence and its place in the life are not separated (long absences come late); the number of items in the last class is 57. No run is proposed here. For the chain of steps the curve fixes what steps 8 and 10 are to be measured against: on the dense network with decay there is nothing to keep after one intervening visit, so that any amount kept by a local learner without a store at an absence of 2 would be new.

### 33.28 For the coordinator: Mr. Xu's instructions after decisions 30 to 37; the briefs E61 and E62; the note of decision 15; the lines (16:51 UTC)

**Decisions 30 to 37 were read at 16:33 UTC,** forty minutes after their commit (e647e7d7, 15:53), at the reviewer's check-in; nothing in them was to be forwarded to a worker.

**What Mr. Xu said to the reviewer after those decisions were written** (between about 15:55 and 16:10 UTC; his words are in `STEPS_FROM_THE_PRINCIPLE_20261118.md`, section 1, and in sections 33.22 to 33.24 as corrected).

1. On the point that decision 31 leaves to be run: he did not rule on it as a specification and told the reviewer that it is clear from biology and is not to be put to him again. The reviewer's reading, marked as a reading: a neuron has what arrives over its own connections and its own state; the exact gradient is outside this on any network. This is the reviewer's reading and not a ruling of his. The briefs below keep decision 31's terms: both rules are arms, each named, the matter is run and not ruled, and decision 31's reference (back-propagation over the whole item on the dense network; for seed 73129 E42's `G`, 80 on R after 6,400 updates) is given beside the record. Whether the report reads the exact gradient's arm as a candidate is yours to say; the runs are the same either way. (Revision 1 of both briefs wrote the reading as if it were ruled; the check of E61's brief found this, and revision 2 corrects it.)
2. He did not define the sweep (33.24); he keeps it as the working rule for now ("我们暂时保留这个，尊重事实而不是我们自己的想法").
3. He asked whether each cell can decide its own number of connections as part of learning, since the fewer numbers we fix the better; and, if that cannot be done yet, that the number be taken on a grid of powers of two (2, 4, 8, 16, 32, 64, 128) and the effect read.
4. He repeated that the list of experiments in order from the biological principle is to be made; it is `STEPS_FROM_THE_PRINCIPLE_20261118.md` (eleven steps). He adopted the reviewer's three suggestions: the number of steps of relay with the number of connections; the sparse input after E57; memory and a life on the sparse network.

**The briefs.**

- **E61** (`TASK_E61_THE_ERROR_RELAYED_OVER_A_LIMITED_NUMBER_OF_STEPS_20261118.md`) is the brief of decision 33. Against its points: (a) the network is E50's stage 3, which is the setting `C46-one` (46 values a cell, one of every used byte) and not a setting with 16 inputs a cell; `C16-in` reached 80 on R under neither rule, and `C46-one` is the one both rules learn (33.10); (b) K = 0, 1, 2, 4, 8 and 16 steps, the record's whole-item runs as the last value, under the sweep and under the exact gradient; (c) updates and CPU-seconds to 80 on R and the levels, beside the record; (d) one seed, full length, twelve runs; (e) every cell reads 10 cells, and the brief says that at 256 cells a number near fifty reaches every cell in two steps, so that Mr. Xu's example is not claimed; (f) nine predictions; (g) the child's environment as in E57's check. The paragraph on one step: in this network a step back in position and a step between cells are the same step, and because the readout is dense the limit K bounds how far back an answer's error acts, not whether an error reaches a cell.
- **E62** (`TASK_E62_THE_NUMBER_OF_CONNECTIONS_OF_A_CELL_20261118.md`) is not asked by a decision; it is Mr. Xu's instruction 3 and step 2 of the chain, put to you for release. The same network with N = 2, 4, 8, 16, 32, 64, 128 and 255 connections drawn from all cells, and N = 10 from all cells beside the record's N = 10 from the neighbourhood; both rules; eighteen runs on one seed.
- **Both are checked and at revision 2, ready for your release (17:03 UTC).** Each was checked by a fresh agent that had seen no opinion of it (`CHECK_OF_TASK_E61_BRIEF_20261118.md` with `check_e61_brief.py`, sixteen findings; `CHECK_OF_TASK_E62_BRIEF_20261118.md` with `check_e62_brief.py`, fourteen findings); the reviewer ran both scripts again with the same output and entered every finding. For E61 the check wrote the update with the error relayed over K steps by hand in double precision and found it equal to the direct definition for seven values of K under both rules (largest difference 1.1e-15). What the checks changed that bears on the reading: in E61, the scored items have fixed numbers of letters (R none, A and T twelve), so that K = 5 already reaches the first operand in every item of R and only K of 17 or more does in the items of A and T; in E62, with connections drawn from all cells N = 2 leaves 40 cells that no other cell reads and N = 4 leaves 4, so that these two runs are not read as the effect of the number of connections alone. Forecasts: E61 about 27 to 35 CPU-hours for twelve runs (limit 36); E62 about 37 CPU-hours for eighteen runs (limit 44). Each needs one line; two lines are free. Step 6 of the chain (a cell that sets its own number) is not briefed: it follows E62's curve.

**The note of decision 15** is `NOTE_FOR_DECISION_15_A_LATER_TASK_ON_THE_FIRST_FAMILYS_BYTES_20261118.md` (one page). Its position: cells that receive nothing from the first family's bytes cannot learn a task on those bytes; of the three conditions under which such a task could be learned, the one that a design could add (a gate of the cell's own, opened by what the first family does not present) is what E55's arm `shut` tests, and the protection of the old rows has on record only the program's projection and rehearsal; the reviewer advises against a brief for the second part of item 11 as formulated.

**The lines.** Free: the main worker's (its report is corrected and confirmed, 33.25) and E54's. Running: E55's arm `shut`; E53 (six runs); E57 (nine runs, the first three past update 24,576); E60 (four lives). Reported and in verification: E52's other two seeds (5cb22315). Verified since your last decisions: the first long lives (33.25), E55's arm `blind` (33.26); computed: the curve against the length of the absence (33.27).

### 33.29 E52 on three seeds, verified (decision 19): at each rule's better rate the sweep's lag is nowhere larger; at the present width with a quarter of the pairs it is gone on two seeds and smaller on the third; no statement over the seeds is licensed where they differ

(`E52_SEEDS_VERIFICATION_20261118.md`, `e52_seeds_verify.py`; written by an agent that had seen no opinion of the result; the reviewer ran the script again, 25 seconds, with the same output. Worker's branch at 5cb22315. The dense network: by decision 32 a comparison of two rules on a network outside the definition of locality, from which nothing on requirement B1 or C for the local learner is drawn.)

**What was verified.** The twelve runs at the rate 0.003 on seeds 73129 and 116089 are complete with the prescribed settings; their batch digests equal E41's at every row; none was resumed; 5.94 CPU-hours. All 2,101 figures of the worker's tables, report and message are recomputed from the receipts' counts, none differing, and every word the worker gives is the one the brief's definitions give. Two weight files, taken as a sample, reproduce their final counts.

**The lag** (the sweep's updates to 80 divided by back-propagation's) at the rate 0.001 and at each rule's better rate of 0.001 and 0.003, on R, with the brief's word and its margin in scored rows:

| Setting | 73129 | 116089 | 93683 |
|---|---|---|---|
| half width, half the pairs | 1.22 → 1.36, the same (9) | 2.29 → 2.53, the same (5) | 1.53 → 1.53, the same (7) |
| half width, a quarter | 2.12 → 2.90, the same (3) | 3.31 → 1.43, smaller (10) | 3.31 → 2.05, smaller (2) |
| present width, a quarter | above 1.53 → 0.87, gone (4) | above 2.00 → 1.00, gone (1) | above 2.70 → 1.75, smaller (1) |

- Over the 18 readings (three seeds, three settings, R and A) the lag at the better rates is nowhere larger: 9 are the same, 5 smaller, 4 gone. The sweep gains from the rate 0.003 in 8 of the 9 settings by seed; back-propagation gains at the half width with half the pairs on all three seeds, with a quarter on one, and at the present width on none.
- At the present width with a quarter of the pairs, the setting of the long lives' births, the sweep at 0.003 reaches 80 on R about as soon as back-propagation at its better rate on two seeds. Each of these readings has a lower bound at 0.001 on one side and a margin of one to four rows.
- At the case the explanation names (the half width with a quarter of the pairs) the seeds differ: smaller on two, the same on the third, where back-propagation gains as well and the lag rises. The brief fixes no rule for combining seeds, and the verification draws no single statement there.
- The brief's test does not refute the explanation (that part of the sweep's lag belongs to a rate chosen for back-propagation) on any seed; on seed 73129 this rests on one setting with a margin of five rows.

**Limits.** Two rates on the new seeds; the better of two is not a best rate; a faster rate does not end higher (in 6 of 18 pairs of runs the final R at 0.003 is below that at 0.001); 18 of the 108 readings rest on one or two rows. The worker has been sent six corrections of wording; no figure changes.

**What it bears on.** Decision 20 (no long lives at 0.003) is yours and is not reopened by this. For the two briefs now put to you (E61, E62) the rate is the record's 0.001 under both rules; if the sweep is slower there, part of that may again be the rate, which this section's result makes more likely than not on the dense network and which has not been measured on a network with limited connections. A pair of runs at 0.003 on the record's network would settle it at a cost of about 4 CPU-hours; it is not in either brief.

### 33.30 E55, the arm `shut` and the final report, verified on seed 73129: with the gates of the reserve held shut by a bias, back-propagation learns the task on the first family's own bytes while fewer than 1% of the first family's answers change; under the sweep neither new task is learned; the first family is partly kept in all four runs

(`E55_SHUT_AND_REPORT_VERIFICATION_20261118.md`, `e55_shut_verify.py`; written by an agent that had seen no opinion of the result; the reviewer ran the script again, 398 seconds, with the same output. The runs, receipts and weights at 340a0009; the report, tables and readings at the head 05d223ff, between which no file of a run or of a check changed. The dense network; the reading stays within decision 14. With this, E55 is closed for the reviewer but for the corrections below.)

**What was verified.** Of 2,102 figures and words of the worker's tables, readings, report and message, one differs (a forecast of the cost). All twelve saved networks of the arm reproduce, in numpy in double precision, every count of the receipts. The nine corrections of the arm `blind` (33.26) are entered in the final report.

**The result, in the brief's words** (rows 0 to 16,384; the first family's 31,684 scored answers at up to 22 positions, and 300 answers at 50 positions).

| Run | New task | Never-taught at 16,384: as run; with the reserve held shut | Cells of the reserve whose gate opens on the new task (of 128) | First family at 22 positions: most answers that differ at a row; at 16,384 | First family at 50 positions: right answers at row 0; lowest; at 16,384 |
|---|---|---|---|---|---|
| `unrelated`, sweep | not learned | 72.7; 0.0 | 126 | 135; 49 | 282; 273; 283 |
| `unrelated`, back-propagation | learned from row 4,096 | 99.7; 0.0 | 128 | 34; 14 | 274; 268; 272 |
| `conflict`, sweep | not learned | 19.8; 21.9 | 0 | 925; 726 | 282; 36; 65 |
| `conflict`, back-propagation | learned from row 8,192 | 92.9; 34.4 | 58 | 244; 7 | 274; 268; 276 |

- **The first family is partly kept in all four runs,** and decision 14's statement (kept while a task on other bytes is learned) is met by none of them.
- **The gates did what the construction intends.** On every item of the first family that was measured, at 22 and at 50 positions, no gate of the reserve opened (largest opening 6.5e-5 and 1.4e-4); on the new task's items learning opened them in three runs, at 57% to 87% of positions.
- **The gate is not the route by which the first family's answers change.** With the cells in reserve held shut, none of its answers differs from the start's in any of the twelve networks. The changes pass through the small state that a cell in reserve still takes behind a gate that is almost shut (about 1e-3, growing with the position), read by weights that the construction leaves free. In `conflict` under the sweep this loses most of the first family on items of 50 positions (36 to 65 of 300 right, against 282), with the gates shut throughout; the report did not say so.
- **A task on the first family's own bytes was learned, under back-propagation,** by cells whose candidate reads everything and whose gate is opened by learning: 92.9 against 34.4 with the reserve held shut, while 7 of the first family's 31,684 answers differ at the end (244 at the worst row). Under the sweep no gate opened on that task, and on the other task 126 gates opened without the task being learned within 16,384 updates.

**For the note of decision 15** (its three figures): the first family's answers are those of the table; `conflict` reaches 19.8 and 92.9 as run against 21.9 and 34.4 with the reserve held shut; the gates open on the new task's items and at no position of the first family's. The note's second condition is therefore met on this seed under back-propagation, with two qualifications: the gate's staying shut on the first family is enforced by the program (the gate rows are projected at the start and after every update) and not learned, and under the sweep the condition is not met. The note's third condition (protection of the old rows) is still supplied by the program. The note's advice stands; where it says "if the gate opens on the new task only", the case holds under back-propagation, so that the experiment it describes (gated cells present from birth, rehearsal in place of projection, on the network that E61 and E62 leave standing) is the one to brief if you want this line continued, and under the sweep it has no ground yet.

**For requirement A.** Not met (decision 14): the four acts of the program remain, with the record of the bytes in use and the one use of the keyed forward pass as parts of them, and in this arm also the bias of 10 on the gates. The report did not state the sentence; the worker has been asked to.

**Findings** (nine, of which eight ask a correction of the report; one error of a figure, one of definition): the loss at 50 positions in `conflict` under the sweep; decision 14's sentence; the forecast of the cost (2,736 seconds a run, not 1,370, so that there was no excess); one more commit in the provenance; the arm's definition, which left out the rows of `lm` and `l2`; the gate figures at every row for the runs that do not learn; the evidence for the route of the change; the title.

**The message priority.** The corrections of the arm `blind`, sent at 16:33 with the priority that joins a running turn, reached the worker only when its turn ended, after the arm `shut` had run. This is the observation that decision 29 asks for, and it is negative.

### 33.31 For the coordinator: decisions 38 to 47 carried out but for E61, which Mr. Xu has told the reviewer to leave (20:22 UTC)

**Read.** Decisions 38 to 47 (5a27108e, committed 20:06 UTC) were read whole at 20:19, on Mr. Xu's relay.

**E62 is with its worker.** Your message of decision 41 was sent to E54's worker at 20:21 UTC, its text committed first (`MESSAGES_TO_WORKERS_20261118.md`, message 12). Beside your sentence it says three things that the released brief could not: that the worker may work on its present branch after merging the default branch, if its session cannot push to a new one; decision 38's reading of the exact gradient's arm and its rule that a report names the sweep's three parts, since the brief's terms were written under decision 31; and that a message reaches the worker only when its turn ends (decision 45). The brief itself is unchanged.

**E61 is held.** With the relay of your decisions Mr. Xu instructed the reviewer to handle the matters other than E61. Revision 3 with your three amendments (the reading on A as well as on R; two runs of the whole item at the rate 0.003; the report's first paragraph on what the job does not test) is therefore not written, and nothing has been sent to E52's worker, whose line stays free for it. The reviewer does not know his reason and has not asked. Decision 39 stands as yours; it is carried out when he says so.

**Entered in the chain of steps** (revision 3): decision 44 (the chain is the order of work; no new job on the dense network), decision 38 (the reading of the exact gradient's arm; the three parts named in every report), decision 40 (the second part of step 3, on E53's network, when E53 is verified), decision 42 (an arm without decay in step 8), decision 43 (the outside memo's 5.2 as step 9; the experiment of the note of decision 15 not written until the sweep's failure to open a gate is understood or the rule's form has changed; E40 in step 11).

**For decision 45,** in briefs from now on: where a later stage depends on the reading of an earlier one, the brief names the point at which the worker sends its message and ends its turn with no process running. The brief of the second part of step 3 will have such a point after its checks, if its design needs the reading of E53.

**For decision 46.** The reviewer's disk stands at 1.2 GB; no verification has been limited by it so far. Three verifications are ahead (E57, E60, E53), of which E60's reads the largest files; the reviewer says in the log if one cannot be made in full.

**The lines.** Running: E53, E57, E60 (world 4), E62 (starting). Free: E52's (kept for E61), the main worker's and E55's (decision 46: one for the second part of step 3, one for the readings of E57 and E53).

### 33.32 A second review, at Mr. Xu's instruction: what the program fixes, set against what a neuron has; the outside audit; E61 (21:00 UTC)

**What happened.** At about 20:40 UTC Mr. Xu asked why a sparse input is not learned although the cells are connected. The reviewer found in the forward pass what it had never examined: between two bytes of an item the cells exchange what they hold once, and the answer is due one byte after the last digit. Mr. Xu's judgement was that finding such things is what the reviewer is engaged for, and he instructed that every place of this kind be examined. The result is `REVIEW_OF_WHAT_THE_PROGRAM_FIXES_AGAINST_THE_PRINCIPLE_20261118.md` (revision 1): an inventory of 32 structures that the program fixes (27 in revision 1), each with its source line, whether a neuron has it, and whether the record examined it. A fresh agent that had seen neither the list nor any document of the project made its own inventory from the source (80 entries; `INDEPENDENT_INVENTORY_OF_WHAT_THE_PROGRAM_FIXES_20261118.md`); its ten weightiest include the reviewer's first three, and the five rows it had beyond the reviewer's are entered in revision 2. It puts first that the sweep is itself one backward pass over all maps and positions, and it counted that only 520 of the 2,560 connections between cells have a connection in the reverse direction, so that four fifths of the sweep's crossings between cells go where no connection exists (the review's finding 6).

**The findings that were never examined, by weight** (the review's section 4).

1. *Relay has no time.* One exchange between cells for each byte, forward and backward, and an answer due at once. The record's reading of the sparse input (E49, E50), of the unused cells beyond the first ring (E47, and E53's design) and of E61 each took this as given. A deduction from the source, not a measurement; the test is a run with several exchanges for each byte.
2. *The network called sparse is a small sparse part inside a large dense one.* The readout is a dense network of two layers with 92.1% of the weights in use; it reads every cell and, through its largest map, the current byte directly, past the cells. What the cells compute and what the readout computes has not been separated.
3. *The learner never meets one item after another:* its state is reset by the program, and each change of a weight averages 64 items.
4. *From the outside audit, which the reviewer had missed:* the stage-3 network has no input for a byte outside the first family's 46; and the reviewer's own section 28.6 already answers the question that step 8 of the chain asks as open.

No verified figure changes; the review's section 5 says what each conclusion of the record may now be quoted for.

**The outside audit** (`origin/claude/friendly-darwin-8mmhfm` at ae1816b8) finds E61 as written not a test of relay between cells and proposes the form in which a cell keeps its own path and only the crossings between cells are limited; lets E62 stand; and asks for six changes to the chain of steps. A fresh agent of the reviewer is verifying its computation independently and drafting revision 3 of E61 in that form with your three amendments of decision 39. The draft is not released; it comes back to you.

**What the reviewer proposes to you and to Mr. Xu** (the review's section 6): first a job that gives relay time (R = 1, 2, 4, 8 exchanges for each byte, on the input with 16 values a cell and on E47's network); read-only measurements of the readout on the saved networks now; E61 held in either form until the first is read, because with several exchanges a position and a crossing are no longer one step; E62 to finish, quoted with its limit; the audit's changes for requirement A. This bears on your decisions 39, 40 and 44: decision 40 places the test of relay between cells on E53's network, where by the review's finding 1 the cells beyond the first ring could not serve the last operand at any start.

**One correction of fact,** made in the chain of steps and in section 33.23: 2.68 is the sweep with 46 values a cell against the sweep with the whole code, not its lag behind back-propagation, which is 2.23 in the same network.
