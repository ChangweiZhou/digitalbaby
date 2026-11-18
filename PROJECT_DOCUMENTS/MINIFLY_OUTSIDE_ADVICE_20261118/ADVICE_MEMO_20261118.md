# Advice memo: persistent learning as a question of where an experience is written

**Date:** 18 November 2026. **Revision 3 (draft).** Part I is revision 1, unchanged; Part II (sections 9 to 13) says what the unmerged branches add, corrects five statements of Part I, and revises the recommendations; Part III (sections 14 to 17) records the goal as Mr. Xu restated it to this session and the results verified since revision 2, which supersede section 10.1. For Mr. Zhou and Mr. Xu, and for the coordinator. Written by an outside session at Mr. Xu's request to look at the project and offer advice on the larger picture.

**Status:** proposed. This memo decides nothing, authorises no run and reports no new result. Nothing was trained and no figure was recomputed.

## 1. Scope and sources

**Read**, all at `codex/minifly-consolidated-20261023@16b0e494`:

- the review (`MINIFLY_PAST_WORK_REVIEW_FOR_MR_ZHOU_20261106.md`), sections 1 to 8 and 9.125 to 9.158, and the one-page close of 16 November;
- the reviewer's memo to the coordinator (`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/MEMO_TO_COORDINATOR_LEARNED_STRUCTURE_20261115.md`, revision 24), sections 24 and 28 to 33; its plan and log (`COORDINATION_20261117/`); the two discussion memos of 15 and 16 November; the coordinator's decisions of 18 November;
- for section 5.2 only, the sources `arch26.py`, `exp11_cell.py` (tasks and item format), `e40_new_family.py`, `e48_gated.py` and `smallworld/lw_world.py` in `MOD97_LOCAL_RULE_20261106/`.

**Not read for Part I:** the branches that are not merged into the default branch (the long lives, E52, E54, E58, E59, and the older Codex lines of 3 to 6 November). Part II says what they add or change, and corrects statements in sections 2, 3.3, 4.1, 4.2, 4.4 and 4.5.

**How statements are marked.** *Measured*: a figure in the record, as the workers reported it and the reviewer verified it. *Argued*: my reasoning, not tested. *Proposed*: a recommendation. Every measured figure is a development result on one to three seeds and is not confirmed on tasks sealed in advance. Pointers to the literature are from memory and are listed in section 7 for checking.

## 2. Summary

1. Since 6 November the experiments have varied how the error is carried back (the exact gradient, the signs of the weights, learned feedback) and have held fixed where a new experience is written: in the present network every item changes every connection. Whether a learner keeps what it has learned is decided by the second of these, and the results of the last week show it.
2. The project has now measured both ends of one scale. The mushroom-body learner kept everything and generalised nothing. The dense recurrent network generalises and keeps nothing. The requirements describe a learner between the two, and the only learners there so far were placed there by a line of the program (E48, E51), where they worked.
3. The three requirements of 17 November (keeping, overtaking, locality) are one mechanism seen from three sides and should be one experiment. At present each is tested in the configuration in which it must fail.
4. The first test I propose needs no new mechanism: births on several families taught together, and then a measurement of how small a change belonging to the new family alone suffices to learn it (section 5.2).

## 3. The reading: one trade-off, measured at both ends

### 3.1 The argument (*argued*)

In back-propagation, in the sweep and in Hebbian rules alike, the change of a connection is the product of the sending cell's activity and an error signal at the receiving cell. The change that learning an item B makes to the network's response to an item A is therefore proportional, at each layer, to the overlap between the activity that A and B evoke there.

- Where the two patterns overlap and the two items call for consistent changes, learning B improves the response to A. This is generalisation and transfer.
- Where they overlap and the changes conflict, learning B damages the response to A. This is forgetting.
- Where they do not overlap, neither occurs.

Generalisation and forgetting are thus the same quantity with opposite sign. A code decides both at once.

### 3.2 The two ends (*measured*)

| | Mushroom-body learner (September to October) | Dense recurrent network (November) |
|---|---|---|
| Code | random sparse expansion; about 5% of cells active on an input | every cell active on every item |
| What is kept | taught old facts at 97.1% (R_center core, review section 2) | a family is at 0.0% after 64 updates of a task that shares no input byte with it (E43) |
| What generalises | no gain on held-out items in any raw-byte test | 98 to 100% on untaught pairs (E26); after a life, a new family from a tenth of its pairs in about 64 updates (E40) |

### 3.3 The middle, when it is supplied (*measured*)

- **E48.** A term of rank 16 or more that is present only for the new family's code: about 37,500 earlier items a network are answered as before, and the new family is learned to a higher level than by free learning.
- **E51, arm `keyed`.** A reserve of cells and a rule that keeps updates away from what the first family uses: none of about 32,000 scored answers of the first family changes while an unrelated task and a conflicting family are learned.

In both the separation is made by a line of the program. The reviewer's sentence is exact: the trial "shows that such a separation, once it exists, is enough". The open question is how it comes to exist.

## 4. The five questions

### 4.1 Why no mechanism has exceeded back-propagation

- *Measured.* The sweep's update has a cosine of roughly 0.5 to 0.8 with the exact gradient, by layer, where it has been measured. With half of the pairs taught it needs 0.92 to 2.29 times back-propagation's updates, more at the narrower width and with fewer pairs (E41). Forgetting, relearning and transfer are alike under both rules (E36 to E43).
- *Measured, in the sweep's favour.* In the calibration of the long-life world the sweep born by its own rule is above the conventional learner by 0.161 in the gate's figure, in seven of eight worlds, and the sweep continued from back-propagation's birth is 0.257 below (plan, item 28). The difference belongs to the rule together with its birth and has no explanation on file. One read-only comparison of the two births would be worth making before the long lives are interpreted.
- *Argued.* A rule that approximates the gradient of the same loss in the same network cannot be where requirement A is met. What back-propagation does badly over a long life follows from changing every connection on every item, and the sweep does the same. An advantage has to come from which connections change and by how much. The adopted question says so already ("changes confined to the connections in use"), but in the present network every connection is in use on every item, so the phrase excludes nothing.

### 4.2 Long-term memory

- *Measured* (memo, section 29). On an unrelated item no cell of the 256 is idle. The first family occupies 255 or 256 of the 257 directions at the input of the recurrent matrix. The code of a byte that has never occurred lies 99.9% inside the nine directions that the family's bytes use. The family is lost in the first dozens of updates, before anything of the new task has been learned.
- *Argued.* There is no other place in this network in which a new thing could be put. Four properties of nervous systems are absent, and the record has a counterpart for each:

| Property | Counterpart in the record |
|---|---|
| An experience engages a small part of the network, and connections of silent cells do not change | None. Every locality experiment limited connections, not activity, and limited connections left no cell idle (memo, 31.3; E43, second addendum) |
| Capacity far exceeds any one thing learned | The first family needs half the cells (E41); E51's reserve; E47's sheet has 435 to 628 unused cells of 1,024 that nothing reaches |
| A connection in long use is hard to change, and a fresh one is easy | The literature check of 10 November named synapse-local consolidation as the candidate (review 9.74). As far as I can find it was never run. E38 tested a pull of equal strength on every weight, which is a different mechanism |
| A fast separate store whose contents are replayed among new learning | Rehearsal keeps the family (E38; E43, second addendum; E16). It does so only at half of the stream (E51, Part B). I read that price as a consequence of the first three being absent |

- Two qualifications. People do lose unused skills and recover them quickly, and the learner already shows the recovery (E36: 64 to 128 updates against thousands). The target is therefore better stated as a curve: the share of an old family answered without new teaching, against the number of updates since it was last taught. And the standing settings forget by themselves: the decay of 0.25 removes what receives no gradient within about 2,000 updates (memo, 24.6), and E46 shows that a birth does not need it.

### 4.3 Locality

- *Measured.* Limited connections between cells cost 1.4 to 2.7 times the dense network's computation to threshold; an interface at a boundary 1.7 to 2.5 times; a limited input is not learned under the sweep on the learned code (E42, E47, E49). None of them protected the first family.
- *Argued.* Three things are called locality here: limited wiring, a cell that reads part of the input, and a rule that uses only signals present at the connection. On one family that a network of 128 cells learns, the first two can only cost. Their return, if there is one, lies in keeping (an item cannot change connections it does not use) and in a cost per item that does not grow with the network. Computation to threshold on a first family measures neither. Requirement C should be judged inside the lifetime experiment.
- On the statement that a cell cannot see a million tokens. A network that reads one byte at a time already meets this in its forward pass. What sees the whole item is the learning rule, which passes the error back over every position. If locality is essential, that is the violation that remains (the eligibility traces of review 9.10 failed on the dense cell because credit runs through other cells). Limited connections are what would make a forward-only rule affordable, since the cost of carrying each connection's influence forward grows with the number of connections a cell has. I recall this from work on sparse approximations of real-time recurrent learning and have not checked it.

### 4.4 Overtaking

- *Measured.* A lived network learns a new family from 2.9 to 5.7 times fewer pairs than a fresh one, under both rules (E40; plan, item 26). Counted over everything the lived network was given before, it has used more; it would fall below the fresh learners after 3 to 14 further families if each cost what E40's did, and no run supports that yet.
- *Argued.* The saving compounds only if what was learned earlier is kept. Requirements B1 and B2 therefore depend on requirement A and are not properties of the rule.
- *Measured* (the trial of the first long lives; one world, one seed; memo, 33.9):

| Quantity at 1,024 visits | Without rehearsal | With replay |
|---|---|---|
| Late level | 0.669 | 0.730 |
| Kept at a return after five or more absences | 0.015 | 0.472 |
| Trend of the first-visit level over the life | +0.106 | +0.104 |

- *Argued.* Two things follow. The conventional learner is not stuck at this length of life. And this world hardly rewards keeping: the arms differ by 0.46 in what is kept and by 0.06 in level, because a return visit teaches the family again for 256 updates and relearning takes tens of updates. Moreover the store holds 55,209 items against 80,000 to 108,000 distinct taught pairs in a life, more than half of everything taught, so replay is strong here by construction. A learner that keeps a family in its connections could overtake replay at equal memory only where there are many more families than the store can cover.

### 4.5 The meta-level view, and the pace

- Mr. Xu's statement of 15 November is that learning itself reorganises the network and that the problem is an optimisation at the meta level. *Argued:* in the terms of section 3, what is optimised at that level is the rule of allocation: for each experience, which connections change, by how much, and what is given up. Its inputs can be local (a cell's activity, a connection's use and age, the error) and its objective is performance over the life. Back-propagation's allocation is "every connection, in proportion to its gradient". That is right when every item returns, as in shuffled data, and wrong in a stream.
- E40 is already an instance: a life has reshaped the network so that a change of rank 16 installs a new modulus.
- The pace is that of an outer loop run by hand. From E26 to E59 took about a week; each experiment had a brief of its own with one to three audits and six to fourteen required changes; a long life costs about four CPU-hours.

## 5. Recommendations

### 5.1 One line of work (*proposed*)

State the candidate once: a learner in which an item engages a small part of the network, only that part changes, connections harden with use, and unused cells can be brought into use. Compare it with back-propagation with a plain replay store at equal memory, over a life, in a world in which old knowledge must be used without being taught again. Locality is then the means, keeping the return, and the lifetime account the place where the return is read.

### 5.2 A job: diversity at birth, and the smallest change that belongs to the new family alone (*proposed; the specification is the coordinator's*)

**Question.** After births on increasing numbers of families taught together, how small a change that only the new family uses suffices to learn that family?

**Why.**

- Births are on one family and lives teach one family at a time. Within any stretch of the stream the family's code is the same on every item, so it carries no information, and nothing has ever pressed the network to make its arithmetic depend on the code. *Measured:* at a return without rehearsal, 42 to 48% of the answers are the right answer under the modulus of the family taught in the visit before (memo, 33.9); in E40 the four new code rows were used almost unchanged (memo, section 23).
- Under fixed memory the private cost of a family decides how many families fit. E48's term of rank 16 is 36,896 numbers, and thirty of them equal the network.
- If the shared circuit were general across moduli, a new family could be held in the byte-table rows of its own code bytes, 32 numbers, which no other item reads. It would then be kept by construction, with no gate, projector, reserve or store.

**Learner and format.** E26's network and item format, in which each task has one code byte of its own (`exp11_cell.DEV`, `e40_new_family.F4`). The long-life world differs: there a task's code is two bytes drawn from a shared set of 128 (`lw_world.py`), so a family has no row of its own; see 5.4.

**Births.** k = 1, 4 and 16 families taught together from random weights, half of the pairs of each. `exp11_cell.batches` already takes a list of tasks. Code bytes are taken from those that no family uses. The dose is set by a rule fixed beforehand. Three seeds, after a trial on one.

**Arms for the new family** (a modulus absent from the birth, a tenth of its pairs, 8,192 updates, E40's rows and scores):

| Arm | What may change |
|---|---|
| `free` | every weight, as in E40 |
| `rows` | the four code rows only (E48's runner without a term) |
| `rank 1`, `rank 4`, `rank 16` | the code rows and E48's gated term at that rank |

**Read.** For each k, the smallest arm whose reduction score on never-taught pairs is within 10 points of `free`. In `free`, the earlier families' scores at every row.

**Readings, fixed beforehand.**

| Outcome | What it means | What follows |
|---|---|---|
| The smallest sufficient arm falls with k, and `rows` suffices at k = 16 | A new family of a known kind can live in lines of its own | Section 5.3: make free learning prefer those lines |
| It falls with k, and `rows` does not suffice | Diversity lowers the private cost | Larger k (56 moduli are unused) before any mechanism |
| It does not fall with k | A cue cannot separate the modulus from the shared circuit | Reserve cells (E51, E55) are needed even for a family of a known kind |

**Expectations, on record.**

| | Statement | My chance |
|---|---|---|
| P1 | At k = 1, `rows` stays below 20% on reduction pairs | 85% |
| P2 | The smallest sufficient arm is smaller at k = 16 than at k = 1, in two of three seeds | 60% |
| P3 | `rows` suffices at k = 16 | 25% |
| P4 | A birth on 16 families reaches the birth's criterion within four times the single-family dose at the present width | 60% |

P3 is the less likely outcome and the most valuable one. The job is worth its cost because each of the three outcomes names a different next step.

**A cheaper first look, on saved states.** The replay arm of the long lives has been taught about fifty families among one another by its store. A probe family of that world could be taught to its end-of-life state by changing byte-table rows only. The rows are shared there, so the run says nothing about keeping, but a failure would lower P3 before any birth is run.

**What it cannot show.** Keeping in a learner all of whose weights are free (section 5.3); content of a new kind, such as E43's task; a life.

**Cost.** About 40 CPU-hours by my arithmetic from the record's 0.11 CPU-seconds an update (twelve births and 45 runs of 8,192 updates). The forecast is the worker's.

### 5.3 Plasticity that prefers unused connections (*proposed; requirements only*)

- A quantity kept for each connection, or more cheaply for each cell, that rises with use and lowers the step. It is computed from what is present at the connection, receives no signal from the harness about a change of task, and is counted in the stored numbers.
- The steps are the rule's own. AdamW sets each weight's step from its gradient history and would undo the quantity; the decay acts on everything.
- Tested on E51's protocol, beside the arms on file: `open`, `rehearse` at a half and a quarter, `keyed`.
- The criterion is the reduction score within a few points, not unchanged answers. Exactness is what requires the projector, which stores a quarter of the network's size and is not local.
- The rule of the memo's section 20 for an added mechanism is met: the deficit is measured on a learner that generalises (E43; E51, Part B; the trial's 0.015), the plain baselines are in the same experiment, and the expected effect is to be stated by whoever specifies the job.
- For content of a new kind the input code matters. E50's fixed code gives a new byte lines of its own, and the reviewer has noted that this is what would let a reserve open without a device (memo, 29.3 and 30.3). A random sparse expansion of the input is what the mushroom body computes. This is the place in the design where it does something that nothing else in the present learner does: it gives a new symbol, or a new combination of symbols, lines that earlier items do not use.

### 5.4 The world (*proposed*)

- **Headline.** Old families answered at a return before any teaching, as a curve against the length of the absence, beside the level.
- **Size.** Several hundred families, so that a store of the network's size holds about a tenth of what was taught. The small world's list had 540 tasks.
- **Codes.** A task's code is at present a pair of bytes from a shared set, so a family's identity is a combination. That is a legitimate and harder case, and it calls for a code of combinations, which is again what a sparse expansion provides. It should be a decision and not an accident of the generator.
- **Later.** Tasks in which an old family is needed inside a new one, without being taught again.

### 5.5 Process (*proposed*)

- Let the running jobs finish, and do not work off tiers 2 and 3 of the plan of 17 November as a list. The inventory of 107 untested entries is a map of what has one value in the record. The test for starting an item is the project's own (research-loop skill, S0): which of its two outcomes would change what is built next?
- One rule for each exploratory run, the candidate's own, with back-propagation as the conventional reference once for each world.
- Audited briefs for confirmations and for anything that touches sealed data. Exploration by changes of configuration on one tested runner, on eight or more seeds or worlds, read with intervals. The questions of 5.2 to 5.4 have effects of tens of points.
- One confirmation is owed and cheap: E40's saving on a sealed family (plan, item 17).
- A life of four CPU-hours rules out any automatic search over the learner's settings. Running many small lives as one batch (review 9.149) is now worth its rewrite. Alternatively the questions of allocation can be studied first on the window learner, which the record has and which needs no credit through time.
- The entry point is out of date. `AGENTS.md`, the README and `MINIFLY_VERSION_INDEX.json` describe V181 and V196 as of 2 and 3 November. A session that starts from them, as this one did, is sent to a state two weeks old. The live state is review 9.158 and the reviewer's plan. `AGENTS.md` is a protected file, so the change is the owner's to authorise.

## 6. A question for the owner

Does a cue that is part of the item, its family code, count as the learner's own information? E48 and E51 treated a gate keyed on an item's first byte as an act of the program. Section 5.2 assumes that only labels supplied by the harness are excluded, which is how I read the governance rule that the learner "must not see those labels as neural inputs, hidden gated update signals, task IDs".

## 7. Literature to source-check

From memory. The papers of Dohare, Bricken and Shen are cited in the repository's documents of October and early November and not in the present plan.

| Subject | Pointers |
|---|---|
| The trade-off, and two memory systems | McCloskey and Cohen (1989); McClelland, McNaughton and O'Reilly (1995) |
| Gating by context with stabilised synapses | Masse, Grant and Freedman (2018); Serrà et al. (2018) |
| Synapse-local consolidation | Zenke, Poole and Ganguli (2017); Benna and Fusi (2016) |
| Learning to learn without forgetting | Javed and White (2019); Beaulieu et al. (2020); Riemer et al. (2019) |
| A conventional learner that stops learning over long streams | Dohare et al. (2024) |
| Task diversity, and new tasks taken up without retraining | Raventós et al. (2023); He et al. (2024, modular arithmetic) |
| Sparse codes of the fly's kind for continual learning | Shen, Dasgupta and Navlakha (arXiv 2107.07617); Bricken et al. (2023) |

## 8. Limits of this memo

- It rests on reading. No run, no recomputation, no check of a worker's figure.
- Sections 3.1, 4 (the items marked *argued*) and 5 are untested. In particular, the claim that sparse activity and plasticity that falls with use would keep a family while preserving the transfer of E40 is the hypothesis, not a finding. The mushroom-body learner is the evidence that sparse codes can remove generalisation altogether.
- The reading of the long-life world in 4.4 rests on one world and one seed.
- The unmerged branches were not read at revision 1. Part II below covers them and corrects statements in sections 2, 3.3, 4.1, 4.2, 4.4 and 4.5. Part I is left as it was written, so that the corrections can be seen.

---

# Part II. What the unmerged branches add (revision 2, 18 November 2026)

## 9. What was read, and how

| Branch (head) | Content | How it was read |
|---|---|---|
| `claude/e44-l2-first-long-lives` (c48498f8) | The first long lives: the trial's report; worlds 2 and 3 running | By me: the trial's report |
| `claude/e44-l2-sweep-long-lives` (3e9f4f2b) | The sweep's long lives: interim tables, worlds 1 and 2 | By me: the tables |
| `claude/e52-learning-rates-sdga8c` (75da7e28) | E52: 18 runs on seed 93683 | By me: the tables |
| `claude/e54-fewer-pairs-ELsQP` (2a6af8b7) | E54: first part, seed 73129 | By me: the tables |
| `claude/e58-gated-term-kqasnme` (b45dca24) | E58, Part A: receipts of seeds 73129 and 93683; no report yet | By me: a tabulation of the 24 receipts |
| `claude/e50-fixed-input-code-017dSE` (9f38483f), `claude/e51-reserve-017dSEpW` (5fdac298) | E59, and E51's Part B on two more seeds, both in progress | Not read |
| `claude/vibrant-cerf-sje726` (3456d5f3) | The reviewer: memo section 33.10 (E50, stage 3); the check of E55, revision 3, in progress | By me |
| `codex/minifly-three-experiments-20261106` (501ca0c0), which contains `codex/minifly-v39-v40-source`, `-ordered-front-end-` and `-recurrent-worked-examples-` | V191 to V223 and the unnumbered folders of 3 to 6 November | By three reading agents working from the reports, each asked for figures with their file paths. I checked against the files the figures quoted in section 11 |
| `codex/minifly-whole-brain-learning-scope-20261106`, `codex/minifly-recursive-runtime-20261103`, `codex/minifly-governance-routing-repair-20261103`, `codex/gemini-human-neuronal-mechanisms-audit-20261025` | One to seven files each that the default branch lacks | By a reading agent |

Nine further `claude/*` branches (the target test, E21, E28, E30, E16-M4, the storage audit) hold no file that the default branch lacks, and `claude/charming-clarke-u9g850` is merged. Of the older work that is already on the default branch I read the opening sections of the reports of V158, V161, V185, V192 and V196, which bear on Part I.

Every figure of a running job below is the worker's own, is not yet verified by the reviewer, and is provisional by the worker's own statement.

## 10. The newest results, and what they change in Part I

### 10.1 The long lives (*measured, provisional*)

Lives of 1,024 visits from E41's half-width births, each rule from its own birth. `naive` and `replay` are back-propagation without and with the store; `sweep` and `sweep+replay` the same under the sweep.

| Quantity | World 1: `naive` / `sweep` | World 2: `naive` / `sweep` | World 1 with the store: `replay` / `sweep+replay` |
|---|---|---|---|
| Late level | 0.669 / 0.650 | 0.756 / 0.628 | 0.730 / 0.784 |
| Kept at a return after five or more absences | 0.015 / 0.015 | 0.009 / 0.020 | 0.472 / 0.503 |
| First 16 updates of such a return | 0.095 / 0.078 | 0.111 / 0.085 | 0.538 / 0.602 |
| Last 64 updates of such a return | 0.747 / 0.760 | 0.834 / 0.696 | 0.706 / 0.773 |
| Trend of the first-visit level | +0.106 / +0.082 | +0.111 / +0.070 | +0.104 / +0.113 |
| Score trend over the returns of the second half | −0.009 / +0.063 | +0.042 / +0.098 | +0.079 / +0.046 |

My readings (*argued*):

1. **Keeping does not depend on the rule.** Without a store nothing is kept under either rule in either world (0.009 to 0.020). The argument of 4.1 stands for requirement A.
2. **Between the rules the comparison is open, and three figures now point the sweep's way.** Without a store the late levels are not distinguished (differences of −0.019 and −0.128), while the sweep's score trend over the second half is steeper in both worlds (mean difference +0.064, 90% interval +0.012 to +0.117). With the store, in world 1, `sweep+replay` is above `replay` by 0.054 in late level and in every block of 64 visits after the first. With the calibration's figure (4.1) these are the first results in the direction of requirement B1.
3. **They are not yet an overtaking.** In world 2 the sweep's steeper trend follows a dip (0.49 at visits 320 to 383) and its last block is 0.675 against 0.777; in world 1 the two curves cross back and forth. A learner returning to a level from below has the same shape. By the horizon rule (memo, 17.5) the way to decide is to continue these same lives from their saved states to 4,096 visits, with no new arm.
4. **This world rewards keeping a little more than 4.4 said, and still little.** Under back-propagation a returning family is relearned within its visit to 0.75 without a store against 0.71 with it, the store taking half of every update (under the sweep, 0.76 and 0.77). In late level the store is worth +0.06 under back-propagation and +0.13 under the sweep (world 1).
5. **Under replay a memory lasts longer the more often its family has been met:** kept at a return rises from 0.13 after one earlier visit to 0.65 after sixteen or more under back-propagation, and from 0.20 to 0.71 under the sweep (absences of 2 to 4 visits). Without the store it stays between 0.00 and 0.04. The report notes that a family met more often has also put more items into the store.
6. **Under replay the wrong answers are confusions between families.** At a return after five or more absences, 24 to 29% of the answers are right under an earlier family's modulus and 1 to 2% under the modulus taught last; without the store the modulus taught last accounts for 41 to 52%. The network with a store holds several moduli at once and assigns the wrong one to the code. That is the binding of a code to its family, which is what 5.2 and the note on codes in 5.4 are about.
7. **The conventional learner is not stuck** at 1,024 visits under either rule, by the roster's own definition; both are classed as stuck at 256.

**Correction to section 4.1 and to my message of 18 November.** I wrote that the sweep can only tie or lose. On first learning that is what the record shows. In the long-life world it is not: the three figures of reading 2 are in its favour, on one or two worlds. What I should have written is that the rule does not bear on keeping, and that for overtaking the comparison is open.

### 10.2 E52, E54, E50 and E58 (*measured; one or two seeds*)

- **E52** (seed 93683). At a rate of 0.003 both rules are faster and no run diverges. The sweep's lag at each rule's own best rate is 1.3 to 2.1 times, against 1.2 to 3.3 times at 0.001. In the one control in which the decay is lowered in proportion (0.083), at the half width with half of the pairs, the two rules are level (5,632 and 6,912 updates to 80 on R) and both end at 97 to 98 on R, against 91 and 94 at the standing settings. The settings carried over from earlier learners therefore cost both rules several points at the width of the long lives.
- **E54** (seed 73129). With a quarter of the pairs the dense network under the sweep ends level with back-propagation at 49,152 updates (96.0 and 95.7 on R) after needing 1.6 to 2.0 times the updates to 80. Under the sweep the limited network needs 3.4 to 3.7 times the dense network's updates and is at 82.8 and rising.
- **E50, stage 3** (verified by the reviewer; memo, 33.10). A limited input in which each cell still has an input for every byte is learned at 2.2 and 2.7 times the updates of the dense input. When a cell has an input for only some bytes, neither rule reaches 80 on R within 49,152 updates. Stage 1 had shown that the fixed code read densely costs nothing.
- **E58, Part A** (my tabulation of the receipts; reduction score of the new family, and the panels of the earlier families' addition and subtraction):

| Arm | New family after 64 updates | New family at 8,192 | Earlier families, from 69 to 78 at the start, at 8,192 |
|---|---|---|---|
| Free learning (three arms: as in E40, its repeat, and without decay) | 70 to 92 | 68 to 95 | 16 to 39 |
| Free learning with a pull of every weight toward the lived weights | 71 to 92 | 87 to 98 | 38 to 48 |
| The gated term, with or without decay | 29 to 69 | 90 to 98 | unchanged |

A pull of equal strength on every weight gives a little and does not keep, as in E38. Where the change is made decides.

**What these change.** They narrow the difference between the rules further, with settings and with time. Beyond the corrections of 10.1 they change nothing in section 4. For the job of 5.2, the births should use settings that E52 supports once its other seeds are in. And E50's stage 1 offers a second form of the arm `rows`: with the fixed code read densely, a new code byte has eleven input lines of its own, which is a larger private change than a byte-table row and still needs no gate.

## 11. The older Codex branches (*measured, as their own reports give it; development level*)

Ref `codex/minifly-three-experiments-20261106@501ca0c0`, 30 folders in all. The figures quoted here I checked against the files named; the rest of each reading I did not check.

### 11.1 What this work is

- Nearly every learner in 24 of the 30 folders is at the store end of the scale of section 3: a fixed hashed or random sparse code with a delta-rule, overwrite or count update; a small transition map learned by a local rule; a table over tokens. The exceptions are small: a recurrent network of 256 binary units with 16 active, whose local rules came from an evolutionary search (Rounds 28 and 29); a hidden Markov model trained by counting (Round 31); a stochastic recurrent state of 372 weights with a local rule (V212); and V181's feed-forward learner with fixed feedback, in a scaling design that was not run. None of these 24 trains by back-propagation, and none pairs a store with a learned recurrent computation. The other six folders, of 6 November, are the exception and are taken in 11.4.
- Almost every positive result rests on structure that the harness supplied: a flag on the bytes to be written, frames that route bytes to a component, operand positions, the call to read. The reports say so themselves.
- Several folders are reruns by Codex of learners from the other agent's review series of 3 to 5 November, whose own code is not on that branch.

### 11.2 What it adds to Part I

1. **The argument of section 3.1 was already in the project, twice.**
   - The header of `minifly_v39_v40_archive/elm_fly_separation_v36.py` (September) asks "does input separation, not the write rule, decide persistence?". It records that the write rules of V20 to V35, among them learning-rate cascades, novelty gating and metaplasticity, "failed, traded retention against transfer, or was matched by its own control". It gives a measurement on the real connections from projection neurons to Kenyon cells: two different odours share 3% of their active cells and two renderings of one odour 42%, where the project's byte code of that time gave 60% and 85%. The branch holds the source and no result file.
   - `minifly_capacity_experiment/notes/MINIFLY_WHY_STUCK_20261103.md`: in a sparse superposed store interference follows the overlap of the active sets, about K/N, and the "capacity wall" of V39 to V44 moved with the number of units (512 facts: old recall 50.2% at 16,384 units and 91.6% at 65,536, four seeds, `RESULTS.md`).
2. **A step that falls with use has been run, on the store** (V197; `run.py`, `RESULTS.json`). Each row of the store keeps one number, the sum of its activations at taught writes, and a write to the row is divided by (1 + that number / 5). Recall of the oldest wave of 1,024 facts, four seeds:

   | | Without the rule | With it |
   |---|---|---|
   | Addresses of 2 and 3 bytes | 44.8% | 84.1% |
   | With 4-byte addresses added | 69.3% | 94.5% |

   The rule is worth +25.3 points with the longer addresses (95% interval 16.0 to 34.5). There is no arm with the counters shuffled and none matched for the total reduction of the step; the authors' memo of 5 November names that comparison as owed. At 4,096 facts recall is 64.6%, which is the store's size and not the rule.
3. **A cue in the item has kept one relation while a second was learned, in a learner that also generalised** (`minifly_two_step_parity/notes/MINIFLY_MEMO_REP_GEN_ROUND2_20261103.md`; six seeds). The learner is a successor map learned from the stream, a fixed sparse expansion of it (2,000 cells, 100 active) and a readout written on flagged bytes; the operator byte sends writes to a population of cells of its own. After the second relation was taught the first stood at 96.2% on taught pairs and 97.2% on distances never taught. The second relation did not transfer (2.1% on held-out distances), because the map did not hold its structure.
4. **Gates learned from answers attached themselves to the surface of the cue** (V202 to V204, `minifly_qa1_frozen_generalization`). The write gate ended as the literal prefix "the ": it fired at all 2,000 such positions and at none of 58,104 others. With the prefix changed it wrote at none of 2,400 positions and the score fell to 15.5%; a gate that decided three bytes later did the same.
5. **The learner's own signals for when to write have a mixed record.** A gate on prediction error was "no better than random writes at the same rate" (note of 3 November); writes on surprise gave 0.07 to 0.23 at the answer's first byte (Round 29). On the default branch, V161's count of familiarity at the cell-to-output connection separated new from familiar contexts with an area under the curve of 0.92 to 0.93.
6. **Experience paid where new content came on symbols of its own** (V217, `REPORT.md`; ten seeds). After 24 earlier worlds, answers on objects rose from 60.4% to 98.1% when the new world's vocabulary was disjoint (+37.7 points, 32.3 to 43.4). With vocabulary reused they went from 60.4% to 54.6% (interval −17.6 to +5.6).
7. **An automatic outer loop has been run** (Rounds 28 and 29; `minifly_round28_small_replication/REPORT.md`, `minifly_research_memos/MINIFLY_MEMO_ROUND29_20261105.md`). A search over 15 or 16 parameters of the local rules took the loss on a fresh world from 4.07 to 1.28 bits a byte. When a memory of recent byte sequences was added, the search set its weight to about 4% ("evolution turned the memory off"), and the fast weights were slightly harmful (1.283 against 1.230 without them). My reading: the objective was the loss on the next byte, and neither memory lowered it.
8. **Two speeds of one weight did not help** in the two places tried: a slowly following copy of the maps (V191: −0.7 points, interval −1.8 to +0.5) and the fast weights of Round 28.

### 11.3 Corrections that follow for Part I

- **Sections 2 and 3.3.** I wrote that the only learners between the two ends were placed there by a line of the program. The learner of item 3 is a third, in miniature and at the store end: it kept and it generalised, by a cue in the item which, as the note describes it, acts through the learner's own wiring. Its limit is the one section 3 predicts: it generalised only the relation whose structure its map already held.
- **Section 4.2, third row.** I wrote that, as far as I could find, plasticity that depends on a connection's history was never run. It was, on the store (item 2), with a large effect and an owed control; and, by the September header, as one of the write rules that failed on a code in which everything overlapped. It has not been run on a learner that generalises.
- **Section 4.5.** The outer loop was run by hand in the November line. It was automatic once before (item 7), and what that search did is the caution for 5.5.

### 11.4 The recurrent reference and the three experiments of 6 November

These six folders hold recurrent or windowed networks, several trained by back-propagation as references. One to three seeds or worlds each.

- **V221 to V223.** V221 is engineering only. V222, the same learner at 65,536 sums in one world, answered 0 of 512 held sums and 0 of 128 taught ones under both of its credit rules. V223, a recurrent network of 32 gated cells trained by two exact gradient methods, ended at 7 and 6 of 1,536 held sums, and at 1 and 2 of 384 taught sums (`OFFICIAL_REPORT.md`). Its own label is that the reference missed its floor, and its later phases were not run. The figure "7 of 1,536 held sums at 65,536 records" in section 1 of the review therefore describes a reference that did not learn the sums it was taught. It bounds nothing about what a recurrent network can learn.
- **The ordered front end.** A feed-forward learner of 37,313 weights with fixed feedback, given the fly's sensory code, a fixed ordered code, or a code with the order removed. Held quartets are 0 of 32 with every input; taught items 78 to 79 of 128 with the fly's code and 93 to 94 with the ordered one. It is the only one of the six that ran the later phases: of the items of the second phase that were right before the fork, 0 of 8 and 4 of 14 were kept after 10,000 further records.
- **Worked steps.** A recurrent network of 32 cells on a window, back-propagation within an episode, 24 histories. Taught items 234 to 250 of 256; the same graphs in a new row order 108 to 126 of 256. Worked teaching showed no consistent difference from answer-only or shuffled teaching.
- **The three remaining experiments** (three seeds each).
  - A frozen pretrained encoder with a locally trained readout learned nothing: 0 of 384 taught items.
  - Addition modulo 97 by a window network under back-propagation: 87.6 to 93.5% on the sealed held pairs; the shared path alone 86.3 to 91.6%, the path through the fly's code alone 0.9 to 1.6%. This is the one result of the six on untaught items above 13%, and it agrees with the default branch's first experiments of the same days.
  - A ring network with counting records: 75 of 75 taught, 0 of 25 held, in all 18 histories.
- **The four single documents.** The whole-brain note reports no experiment; it records that the learning line kept 302 projection neurons and 5,177 Kenyon cells and no cell of the central complex, and says the negative results do not show that this omission caused them. The meta guard is a small tool for process, which by its own README does not certify science. The note on V185 (status: proposed, not verified) records a session that ran the research skill when it was asked for the process skill. The Gemini audit is a check of a document and has no result for the learner.

What 11.4 adds:

- **Nothing on keeping.** Only the ordered front end reached its later phases, and nothing in these folders uses replay.
- **The fly's sensory code as the input of a shared computation failed again,** beside a direct input, in two more comparisons (V223's screen: 5 of 12 facts against 12 of 12; modulo 97 above). That is the finding of section 1 of the review. It concerns the code as the input of a computation, and not a sparse code as the place where new things are written, which is its use in 5.3.
- **A caution for 5.2.** The frozen pretrained encoder with a local readout is the nearest thing in the record to "learn broadly first, then learn the new thing in a small part", and it learned nothing. The job of 5.2 differs in two respects (the births are on families of the same kind, and the private change feeds the recurrent computation instead of reading from it), but P3's chance of 25% should be read with this beside it.
- **The earlier learners kept renderings, not content.** Both the ordered front end and the worked-steps network lose most of their taught score when only the rendering changes (93 to 94 of 128 down to 10 to 12; 234 to 250 of 256 down to 108 to 126). This is the review's own reading of the period.

## 12. What changes in the recommendations

1. **The code comes before the rule.** This is the September header's conclusion and it fits the present network exactly: where every item uses every connection, a rule of plasticity has nothing to prefer. Section 5.3 is therefore to be run only on a start in which some cells are unused and are silent on the old items (E51's embedding with E50's code read densely), and 5.2 stays first.
2. **Two controls are added to 5.3,** the ones V197 lacked: the same rule with the counters shuffled among connections, and a step reduced by the same total amount without regard to use.
3. **The world should hold both kinds of code** (5.4): families whose code bytes are their own, and families whose codes reuse bytes. V217 and the wrong answers under replay (10.1, reading 6) both say that the second kind is where experience stops paying. The present long-life world has only the second.
4. **A limit on 5.2, stated beforehand.** A family code is a designated byte. A positive result would not show that a learner can find its cue in a stream that does not mark it; item 4 is the record of what happened when a gate had to.
5. **If a signal of the learner's own is to replace E51's line of the program,** the record favours a count of familiarity over prediction error (item 5).
6. **The objective before the search** (5.5). An automatic search removes what its objective does not reward (item 7). The headline of 5.4, old families answered before any teaching, has to be in the objective before a search over the learner's settings is worth running.
7. **The long lives** (10.1). Continue the existing lives without a store to 4,096 visits from their saved states before anything is concluded on requirement B1.

Unchanged: the reading of section 3, the answers of 4.2 to 4.4 as corrected, the job of 5.2 and the question of section 6. On that question, the note of 3 November used such a cue under the owner's ruling of that time ("anything is allowed except backpropagation and attention") without raising it.

## 13. Limits of Part II

- The figures of the running jobs (10.1 and 10.2, except E50's stage 3) are the workers' own and have not been verified by the reviewer. The long lives are scored on two worlds without a store and on one with it.
- The older Codex branches were read by agents from the reports. I checked the figures quoted in section 11 against the files and nothing else of those readings. Several of the folders are reruns of another session's learners, and several notes cite code that is not on the branch.
- The header of V36 and the notes of 3 November are statements in a source file and in memos. The branch holds no result file for V36.
- Nothing here was run.

---

# Part III. The goal as restated, and the results verified since revision 2 (revision 3, 18 November 2026)

## 14. The goal, in Mr. Xu's words to this session (18 November, verbatim)

1. "He meant a local learner with no back-propagation, and while some structural enhancements are possible, he prefers to let the model figure out where / how to store / update the memory as new information / feedback coming up. Also beating backpropagation has to be understood in epistiemological sense, namely for fixed computation power and accuracy demand, local learning can achieve at least the same generalization as back-propagation, perhaps with even less samples needed but may be with more time for digestion etc."
2. "Local in the sense that each neuron only gets info from its (sparse) connections, not the whole network, and each neuron adjusts the connection strength etc on its own."
3. "Local backward pass is fine, what prohibited is global backward pass."

**An error of mine, corrected by statement 3.** After statements 1 and 2 I answered as if no backward pass at all were allowed. That was my wording and not Mr. Xu's, and what I built on it is withdrawn: that the backward pass over an item puts the present candidate outside the definition, and that delays in place of that pass are the first test to run.

**My working reading, which is mine and open to correction.** A backward pass in which a neuron receives its error over its own connections is local. What makes a pass global is a network in which every cell is connected to every other, so that each neuron receives from the whole network. On this reading the sweep on the networks with limited connections (E42, E47, E49, E50, E54) is the candidate, and the dense network is the reference. Nothing below depends on more than this.

## 15. Results verified since revision 2

Source: the default branch at 392b3eca (review 9.159 to 9.161) and the reviewer's memo, sections 33.13 to 33.19, at `claude/vibrant-cerf-sje726@fffb1ec6`. All are verified by the reviewer from receipts and weights unless marked. I reran nothing.

**The long lives, four worlds** (memo, 33.16). Late level at 1,024 visits:

| World | `naive` | `sweep` | `replay` | `sweep+replay` |
|---|---|---|---|---|
| 1 | 0.669 | 0.650 | 0.730 | 0.784 |
| 2 | 0.756 | 0.628 | 0.812 | 0.574 |
| 3 | 0.742 | 0.683 | 0.835 | 0.789 |
| 4 | 0.732 | 0.672 | 0.709 | 0.769 |

- Without the store the learner under the sweep ends lower in every world: −0.066 (interval −0.120 to −0.013). Worlds 1 and 4 share their births; with the two counted as one the interval includes zero.
- With the store nothing is distinguished: −0.042 (−0.206 to +0.121). The sweep is ahead in worlds 1 and 4 and behind in worlds 2 and 3.
- No overtaking is seen up to 1,024 visits, and every arm is still learning. The two arms of a pair differ in their births as well as in the rule; E60 (the sweep's life from back-propagation's birth, four worlds) is running to separate the two.

**This supersedes section 10.1, readings 2 and 3, and the correction that follows them.** That correction rested on two worlds without the store and one with it. Over four worlds none of the three figures in the sweep's favour is distinguished, and the lead with the store holds only in the two worlds that share a birth. Section 4.1's sentence, that on the record the sweep ties or loses, is what the four worlds show; "can only", as a statement of principle, stays withdrawn.

**The other verified results.**

| Job | What it shows | Seeds |
|---|---|---|
| E52 (33.17) | 0.003 is the best of four rates in seven of eight cases. At each rule's best rate the sweep still needs 1.33 to 2.05 times back-propagation's updates. The control of section 10.2 in which the two were level is one observation and "not a result" | 1 |
| E54 (33.19) | Neither the limited nor the periodic network generalises from fewer pairs than the dense one. With 15% of the pairs, at 49,152 updates: dense 46.6 under the sweep and 73.0 under back-propagation, neither curve flat. With a tenth at the half width, after 122,880 updates: back-propagation 75.0 and rising, the sweep 13.4 | 1 |
| E58 (33.14) | A lived network learns a new family from 5% of its pairs through the gated term: 74 to 86 under the sweep, 89 to 94 under back-propagation, with the earlier families unchanged. Free learning from 5% falls during the run, largely through the decay. The gate is the program's | 3 |
| E59 (33.15) | For a task of other content, under the sweep a network that has learned before never needs fewer presentations than a fresh one. Updates to 80 with 2,048 pairs: fresh 192 to 224, born 320 to 400, lived 448 to 608. Under back-propagation the three are about the same, and with 256 pairs the born network needs about a third of the fresh one's | 3 |
| E51, Part B (33.13) | Rehearsal of half the stream keeps the first family within 5 points on all three seeds; a quarter on one. Reviews at intervals lengthen what a gap leaves, on all six pairs of seed and rule | 3 |

**The first long lives** (reported by the worker in `E44_L2_REPORT.md` on `claude/e44-l2-first-long-lives@45e9ba16`; in verification, so provisional):

- With the store against without, at 1,024 visits over four worlds: late level +0.047 (−0.011 to +0.105), not distinguished; kept at a return +0.53. This is section 4.4's point on four worlds: the world's level hardly registers keeping.
- **Without weight decay the conventional learner with no store begins to keep.** At 256 visits over eight worlds, what is kept at a return rises with the number of earlier visits to the family: from 0.03 after one to 0.22 after sixteen or more at absences of 2 to 4 visits, and from 0.03 to 0.17 at absences of 5 to 32. With the decay it stays between 0.00 and 0.03.
- **The price:** without decay the late level is lower by 0.095 (−0.166 to −0.024), the trend of the first-visit level is lower by 0.119, the arm is classed as stuck, and the size of the weights rises throughout (from 53 at birth to 126 after 256 visits).
- **The learning probe:** the first-visit level of a family never met rises with the learner's age: 0.43, 0.67 and 0.83 after 64, 256 and 1,024 visits without the store.

## 16. What these say against the goal as restated

1. **"At least the same generalisation."** Where half of the pairs are taught the sweep comes within a few points of back-propagation's level, on the dense network and, at 1.4 to 2.7 times the dense network's updates, on the networks with limited connections (E42, three seeds). Where fewer are taught it falls behind, and the deficit grows as the pairs get fewer (E41; E54).
2. **"Perhaps with fewer samples, with more time for digestion."** The record runs the other way so far. At a tenth of the pairs a long run helped back-propagation and not the sweep (E54, one seed). Experience lowers the samples a new family needs under both rules (E40; the probe above; E58 at 5%), which is not a property of the rule.
3. **Experience and the sweep.** E59 is a new difficulty for persistence under this rule: a network that has lived learns a task of other content more slowly than a fresh one under the sweep, by two to three times, and not under back-propagation.
4. **"Let the model figure out where and how to store."** The arms without decay are the first sign of memory forming with no store and no act of the program, and they show what it costs. Everything else on keeping is still rehearsal or the program's doing (E51, E58; E55 is running).

These are statements about one local rule, the sweep with signs and one scale for each matrix. They do not show that a local backward pass in general falls short.

## 17. What changes in the recommendations (*proposed*)

1. **Find where the sweep's shortfall lies before reading it as the local learner's.** Two candidates are already in the record: the scale taken for a whole matrix (E57, running), and the feedback by signs, where learned feedback recovered most of a loss on three seeds and was not replicated (E31; ruled inside the line on 13 November). The two cells in which to test them are those where the sweep is furthest behind: a tenth of the pairs at length (E54, part 4) and the lived start of E59. If one of them closes both, a local backward pass equals back-propagation wherever back-propagation generalises, which is the claim in its first form.
2. **Follow the arms without decay.** The question they raise is local to a neuron: how its weights are kept from growing without the shrinking that erases what is not refreshed. One arm in which each cell holds the size of its own incoming weights, beside `naive` and `naive-wd0` at 256 visits, would say whether keeping can be had without the price. The same arms under the sweep have not been run (plan, item 23).
3. **Section 12, item 7 is withdrawn.** The lives cannot be continued to 4,096 visits in the present world, which has 52 families; the roster's report says a longer life needs a larger task space. E60 is the right job for the comparison of the rules, and the larger world of 5.4 is what a longer horizon waits for.
4. **Section 5.2 stands, with its premise now seen in the lives.** The probe shows that experience of many families makes a new one learnable within a visit, and E58 that a small private change suffices from 5% of the pairs. Neither says whether the change can be as small as the family's own code rows.
