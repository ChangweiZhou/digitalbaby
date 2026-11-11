# Review and next steps after 11 November

- **From:** the audit session (the cloud session that wrote the memo, the roadmap note and the next-steps note in this folder).
- **For:** the coordinating session and Mr. Xu.
- **Read for this note:**
  - `codex/minifly-consolidated-20261023` at `fff36103`: the state addenda to 11 November 16:38 EST, review sections 9.82 to 9.100, the E20-B report, the E24 brief (revision 3), the target-test design draft with its sections 7a to 7c, and the step 0 brief (version 5);
  - `claude/charming-clarke-u9g850` at `85c9bfdc`: E24's host-check wave;
  - `claude/target-test-step0` at `7005a6af`: the step 0 report, its sizes and its code;
  - `claude/funny-mendel-34puq7` at `454d76cc`: E21b, commit messages only.
- **Status:** suggestions. The auditor audits and the coordinating session decides. Everything here is read-only: I parsed receipts and weight files, trained nothing and imported no harness code. Section 9 lists each check and what I left unchecked. Five scripts in this folder reproduce the checks.

## 1. Summary

**Where the project stands.** It has a network that composes from memory and has never been trained by the working rule, and a rule that works only with answer-only supervision on a network that does not compose. The two have not met.

**Six suggestions.**

1. **Release E24 on same-host baselines.** Its host check stopped because the backprop original was born on the owner's machine and the rerun ran on a cloud host. The rerun's weights are, byte for byte, those of E20-B's birth of the same network and seed, which another runner made on another cloud host. So the cloud side is one result, and same-host baselines are within reach for all three seeds.
2. **Before a GRU is born under the rule, measure the sweep's alignment on E20-B's saved weights,** by layer and by steps back in time. It trains nothing. In every network the rule has worked on, the path back through time carries no weights. In a GRU it does.
3. **Size the target test for learnability before more audit rounds.** Step 0 shows that the held answers are determined by the stream. It does not show that a learner can acquire them, and at the proposed sizes I expect that none can.
4. **Ask for one ruling: learned feedback weights.**
5. **Make E24 the last experiment on the two-bank network.**
6. **Route every commission through the coordinating session.**

**For the close on 17 November:** name the candidate as a pair, a rule and a network, with the run in which that pair was born and lived. If no such run exists by then, say that no candidate exists yet.

## 2. Where the project stands

Seeds are in the order the record gives them. "Recomputed" means I rebuilt the figures from the receipts with my own script and they agree with the filed ones in every cell.

| | Finding | Figures | Filed in | My check |
|---|---|---|---|---|
| 1 | Without replay, both updates lose the hard pairs. The forgetting belongs to the stream | Plus-minus pairs at the end: 1.6 / 0.9 / 1.5 under the rule, 1.0 / 1.0 / 0.8 under backprop on the same pipeline. At their best during the life: 61 to 85 and 91 to 92 | 9.88 | recomputed |
| 2 | With every next byte as the target, backprop keeps the skill and the sweep loses it | Mod-97 mean at the end: 10.3 / 5.4 / 9.7 against 89.3 / 90.6 / 89.2. The sweep stood at 84 / 83 / 82 after the first block | 9.88 | recomputed |
| 3 | The same happens at birth, with local heads | 0.3 / 0.3 / 0.7 against 74.8 / 69.7 / 66.3 | 9.96 D, 9.97 | as filed |
| 4 | A learner born and living under one rule ends where the backprop-born one does, on single operations | Mod-97 mean at the end: 88.1 / 84.8 / 84.3 against 87.9 / 86.1 / 86.9. Largest gap 2.6 | 9.85 | recomputed |
| 5 | A windowless network can compose from memory under backprop | Untaught plus-minus pairs: GRU 73.6 / 78.8 / 75.1; LSTM 78.6 / 72.2 / 69.2; two-bank 1.5 / 0.5 / 7.5; two-bank with the added loop 20.2 / 1.1 / 1.0 | E20-B report | recomputed |
| 6 | The rule's own learner does not compose, even with the operands in view | Sign-rule window learner: 0 of 300 twins in each seed; with everything in view 121 / 83 / 236 of 600 | 9.96 A | as filed |
| 7 | Between the banks, the relay is the part that works | Cutting it costs 7 to 19 points at birth and 12 to 15 after a life | 9.97 | as filed |

Three gaps separate the working rule from the target test:

- **The answer mask.** The rule needs it (rows 2 and 3). The target test has none.
- **The network.** The rule runs on the two-bank network, which does not compose (rows 5 and 6). The networks that compose have only ever been trained by backprop through time, with the mask.
- **The stream.** Whether any learner can acquire the target stream's held answers is not known (4.3).

Row 4 is the strongest result for the rule so far. Its limits are the ones 9.85 lists: answer targets, replay, single operations, three seed pairs.

## 3. Pending jobs

| Job | Where | State | Waits for |
|---|---|---|---|
| E20-B: which windowless network composes | default branch, `fff36103`, the worker's commit | Complete. 24 runs, a report, and a review paragraph filed as its own file | The coordinator's own verification of the 24 receipts and hashes (9.98 C), and its reading. My check is in section 9 |
| E24: is the every-byte collapse the loss or the rule | `claude/charming-clarke-u9g850` | The host-check wave ran and stopped. None of the nine births has started | A decision on the baseline (4.1) |
| Target test, step 0 | `claude/target-test-step0` | Delivered. Ten checks pass for world seeds 1 to 3. The worker raises two points | Codex's audit, and a decision on the two points (4.3) |
| E21b: resume qualified on the merged harness | `claude/funny-mendel-34puq7` | In progress | Nothing |
| Retention cost curve | Not designed | Conditions adopted (9.91, item 3) | E21b, and the network decision |
| The every-byte failure with one change | Not designed | Conditions adopted (9.91, item 4) | E24 |
| Probes on the stage 2 window checkpoints | Announced in 9.97 | No result on file at `fff36103` | Mr. Xu's go |
| The coordinator's rerun of E20-B's stage 1 parity on the owner's machine | Owed since 9.96 | Not on file | Nothing |

Two remarks on E20-B, both from my recomputation:

- The three G0 lives that the container restart killed were rerun from their births. The reruns equal the interrupted receipts in scores and loss at all five stages the interrupted runs had reached.
- The GRU was selected over the LSTM by 2.5 points on the plus-minus pairs (75.8 against 73.3 over three seeds). That margin is no larger than the host effect of 4.1. A figure the report's table does not carry separates the two more clearly: at 40 distractors, where training used 0 to 12, the mod-97 family mean is 95.8 / 91.7 / 95.4 for the GRU, 87.8 / 79.6 / 79.8 for the LSTM, 35.2 / 35.1 / 38.0 for the two-bank network and 42.9 / 42.6 / 73.5 with the added loop. It is outside the registered predictions, and the mean includes max and min, which copying one operand answers half the time (erratum in 9.87). I give it as a description.

## 4. The suggestions

### 4.1 Release E24 on same-host baselines

**What happened.** The brief pins six original births and asks each host check to reproduce its original at every checkpoint. On a cloud host:

- the rule-born birth (T-ans, seed 73129) reproduced its original bit for bit;
- the backprop birth (D-ans, seed 73129) differed at every checkpoint, ending at 93.2 against 91.2.

The worker stopped, as instructed.

**Where the two originals came from.** The T-ans original is E18's rule-born birth, made on a cloud host (commit `2eafb07b`, 10 November). The D-ans original was born on the owner's machine (commit `fad2cec7`, 9 November; review 9.71 labels these births "owner's machine; levels, not digits"). Its manifest entry says it "predates hashed receipts", and its receipt records no host.

**What I add.** E20-B's G0 network is this same network, and E20-B gave it a birth for the same seeds with its own runner on another cloud container. I set the three births of seed 73129 side by side (`e24_host_compare.py`).

| Birth of the two-bank network, seed 73129, backprop, answer targets | Made on | Hard mean at the six checkpoints | Loss at the first |
|---|---|---|---|
| The pinned original, `e17_dev_bank2_nbd_73129` | the owner's machine | 20.3, 52.5, 77.5, 87.2, 90.7, 91.2 | 0.5642 |
| E24's host-check rerun, the harness itself | a cloud host, Python 3.11.18, torch 2.15.0 | 16.8, 54.5, 84.3, 87.7, 91.8, 93.2 | 0.5585 |
| E20-B's G0 birth, `e20_dev_G0_73129`, E20-B's own runner | another cloud host, Python 3.13.18, torch 2.15.0 | 16.8, 54.5, 84.3, 87.7, 91.8, 93.2 | 0.5585 |

- The rerun and the G0 birth have equal sealed scores and equal losses at all six checkpoints, and **all 24 weight tensors are byte for byte equal.** Their file hashes differ only because a saved file carries its own name.
- The original differs from both at every checkpoint. One of 24 tensors is equal, and the worker identifies it as the fixed rate vector.

**Reading.** Two cloud hosts, two programs and two Python versions give one result. The owner's machine gives another.

- The record already holds this pattern for sign-rule receipts: "bit-identical on one machine and diverge across machines from the first update: parity clauses are machine-bound" (state, E16-RP; review 9.54).
- E20-B's stage 1 record notes that the coordinator's receipts carry torch 2.14.0, against 2.15.0 in the containers.
- The worker's report reads the difference at the first checkpoint as a sign of a different configuration or code path, and not of numerical drift. I read it the other way. The first checkpoint is 4,096 updates in, and rounding differences from the first update are fully grown by then.
- What my comparison rules out is a difference between today's harness and E20-B's runner on a cloud host.
- What it does not rule out is a change in the harness's birth path between 9 November and today. One birth settles that (step 4 below).

**How large the host effect is.** The same command and seed, the final hard mean on a cloud host minus on the owner's machine: +2.0 / −1.7 / +2.5 (93.2 / 92.8 / 92.8 against 91.2 / 94.5 / 90.3). That is the size of the differences this program reads. A seed is a pair only within one host class.

**One figure in the record this changes.** Review 9.71 puts the rule-born births at 0.96 / 0.96 / 0.94 of the backprop births. That divides cloud-born births by births from the owner's machine, as its own label says. Against the cloud-born backprop births the ratios are 0.94 / 0.98 / 0.91. The middle is about the same and the spread is wider.

**What I suggest.**

1. **Release the nine births, with D-ans baselines from the same host class.** For seed 73129 the harness's own rerun is on the worker's branch (93.2).
2. **Add the two missing D-ans reruns (seeds 93683 and 116089) to the first wave.** They cost about 3,200 CPU-seconds each by the worker's measurement. I expect them to come out byte for byte equal to E20-B's G0 births, at 92.8 and 92.8. If they do, every E24 baseline comes from one program on one host, and the two programs are shown equal at full dose for all three seeds. If the two births are not wanted, E20-B's G0 births can serve as the baselines on the strength of seed 73129.
3. **Keep the T-ans baselines** (87.5 / 90.7 / 84.7). They are cloud-born and one of them has been reproduced.
4. **Run D-ans 73129 once on the owner's machine with today's harness.** If it reproduces the pinned original (weights hash `8ee5e4e6…`), the cause is the host. If it reproduces neither, the birth path has changed since 9 November and that needs finding. One birth.
5. **Two standing changes.**
   - Record the host in every birth and life receipt and in the manifest: machine, platform, Python, torch and threads. E20-B's receipts already do.
   - Write a host check as "reproduces a pinned original born on the same host class". Figures across host classes are descriptive.

### 4.2 Before a GRU is born under the rule, measure the sweep's alignment on E20-B's saved weights

**Why this is new ground.** The working rule replaces weights by their signs, scaled by the layer's mean magnitude, wherever the error crosses a weight matrix on its way back. Where it has worked, the step back in time crosses none:

| Network | State update | What the error is multiplied by, one step back in time |
|---|---|---|
| Bank (`arch17.py`) | `h_t = R·h_{t-1} + w_t·c_t` | `R`, a fixed rate. The write gate and the content never read the bank's own state |
| E11 cell | `h_t = r_t·h_{t-1} + (1 − r_t)·c_t` | `r_t`, a gate value the unit already holds. The gate reads the input only |
| GRU | `h_t = z_t·h_{t-1} + (1 − z_t)·n_t` | `z_t`, and three more terms, because `z_t`, `r_t` and `n_t` each read `h_{t-1}` through a recurrent weight matrix |

In the two-bank network the error crosses a weight matrix only when it moves down a layer, a handful of times between the output and any synapse. In a GRU it crosses the recurrent matrices again at every step back, about twenty times in an item. Each crossing replaces the weights by their signs.

The literature check found no study of sign feedback through recurrent weights, and one preprint in which unscaled sign feedback through four time steps of a spiking network fell to chance (9.84). Ours differs from it in three ways: the feedback is scaled, the units do not spike, and the sweep covers one item. So that result does not transfer. It is a reason to measure before spending births.

**What stands in the way today.** E20-B's GRU and LSTM use the library's recurrent layers, which the harness's `lin()` cannot reach into, and `arch20.forward` refuses the local path by an assertion. A sign path needs the step written by hand.

**The measurement. It trains nothing.**

1. Write the GRU step from `lin()` and load G1's saved weights into it. Two checks before anything is read:
   - its outputs equal the library layer's on the same input, to rounding;
   - with true weights on the way back, its gradients equal autograd's through the library layer.
2. Switch the way back to weight signs. On a few hundred items, record:
   - for each parameter matrix, the cosine between the sweep's update and the true gradient;
   - for the error at the state, the same cosine as a function of steps back from the answer, from 0 to about 20.
3. Do it on the three G1 births and the three G1 end-of-life files, under answer targets and under every-byte targets. Do the same for G2. An LSTM step is as easy to write and its files are saved too.
4. Run the same measurement on E20-B's G0 files. That gives the figure for a network on which the rule is known to train, and it is the comparison that matters.

**Readings, fixed before the numbers.**

- If the GRU's alignment at the longest gaps is at or above the two-bank network's, a birth under the sweep is worth its compute.
- If it falls toward zero within a few steps, three remedies are known in advance, cheapest first:
  - keep the weight-free part of the step (`z_t`) and drop the three weighted terms on the way back. The harness has a trace of this kind for the cell. The nearest result on file is E19, where a forward-only rule with weight signs reached 0.63 to 0.79 of its ceiling (9.79);
  - learned feedback weights for the recurrent matrices (4.4);
  - a finer scale than one mean per matrix.
- If G2 aligns clearly better than G1, that outweighs a 2.5-point selection margin.

**The two runs this prepares.**

1. **A reference cell that is missing whatever the rule does:** the GRU under backprop with every next byte as the target, birth and life, scored on composition from memory. E20-B taught with answer targets. The target test has no answer mask. Whether composition from memory survives without the mask is not on file for any network. Review 9.88 shows only that backprop keeps single operations on the two-bank network.
2. **The candidate run:** the GRU born and living under the sweep with answer targets, against backprop on the same pipeline, on one host. E20-B's own marks serve as absolute marks: new instances of taught pairs at 85 or above, plus-minus pairs at 40 or above.

### 4.3 Size the target test for learnability before more audit rounds

**What step 0 shows.** For three world seeds, a solver given only the bytes determines every held answer; the oracle passes; suffix, table and recency responders fail; no held pair is exposed; the control leaves its contexts undetermined. That is identifiability, and the work looks careful. I did not audit its code.

**What it does not show** is that a learner can acquire the held answers from this stream. The brief asks this itself, as question 1 for the audit: are 24 contexts and 12 cues enough for an ordinary recurrent reference to learn the values from 216 taught pairs? Nothing queued answers it. Step 1 of the design would, after the audit rounds.

**The sizes, from the worker's report.**

| | |
|---|---|
| Pairs per family | 24 contexts × 12 cues = 288: 216 taught, 48 scored on first encounter, 24 sealed |
| Hidden numbers per family | 24 context values and 12 cue values in Z_97, fixed by the stream up to one shift |
| Before the first panel | 1,760 episodes, about half of the stream's 46,564 bytes; each taught pair shown 4 times |
| Over the whole stream | 3,552 episodes; a taught pair of family 1 shown 7 times, of family 2 shown 9 times |
| What is in view at a probe | a cue letter. The context letter is ten or more bytes back. In the ordinary stream no hidden value is ever written; only outcomes are |

**What this project's own record says about learning sums.**

- With both operands in view as digits, the smallest taught share ever run was 15% of the 9,409 pairs, about 1,400 pairs. With several hundred presentations of each taught pair, backprop with weight decay and dropout then reached 82 to 91% on held pairs at its peak, sign feedback 61 to 81%, and the random-feedback rule 36 to 39% (reviews 9.2 and 9.3).
- The target stream gives 216 pairs, which is 2.3% of the value pairs, four presentations, and letters in place of digits. So it is not the same task at a smaller share. The learner has to find 36 hidden numbers and the addition that joins them, from outcomes alone.
- Review 9.3 named the regime: "The project's 80-fact tables with a random-expansion memory are the opposite regime: the table is always cheaper, so nothing ever pushes toward the rule."

**My expectation. It is a prediction, not a finding.** At the proposed sizes no learner that starts from nothing acquires the held pairs, under any update. The reference of step 1 learns the taught pairs as a table, if four presentations are enough even for that, and scores the held pairs at the level of the controls. The design would then stop at step 1, after four audit rounds on the brief and at least one more on the code, at sizes that have to change.

**What I suggest.**

1. **A sizing sweep now, beside the audit.** It is a sizing exercise and should be registered as one: no prediction judged, no verdict, nothing concluded about any rule.
   - A backprop GRU of E20-B's shape on the ordinary stream from the step 0 generator, which keeps its sizes in one place.
   - World seeds that will never be scored. Not 1, 2 or 3.
   - Vary the modulus (97, 31, 11), the presentations of each taught pair before the first panel (4, 32, 256), and whether the value lines are in the stream.
   - Read the taught pairs with the context in view, the first-encounter pairs, and the sealed twins against the suffix and recency tables.
   - By my estimate a cell at the proposed size, 46 KB of stream, costs seconds to minutes, and a cell at 256 presentations some tens of minutes. I have not timed one.
2. **Say what a learner may bring.** "Taught by nothing except the later bytes of its own stream" reads as a learner that starts from nothing. Every life in the E-series starts from a birth. The design should choose:
   - *from nothing:* the hidden structure then has to be learnable within the stream's length. That points to a much smaller modulus and many more episodes, and the test then measures the slow learning of a rule;
   - *born on a declared curriculum* that shares no pair and no value with the world: the stream then has to offer the values in a form the born skill can take up.
3. **Consider making the stated values the base and their absence the harder arm.** With the lines `X:vv c:uu` the task is to bind a letter to a stated number, keep it across episodes, and add two recalled numbers. That is E20-B's composition from memory with one more step of recall. It is also Mr. Zhou's delayed observation in the form closest to what the project has shown to work. Without the lines the learner has to solve for hidden values from sums, and nothing on file approaches that. The draft has it the other way round: the lines are "an arm and not a premise". If the sweep shows a reference acquiring held pairs with the lines and not without, I would turn the order round.
4. **Settle the line schedule after the sweep.** The worker's two points have one cause, a line after every episode:
   - in the moved-lines stream a line names the probed pair before 51 / 47 / 55 of the 96 first-encounter probes;
   - the disclosure ledger never differs between the aligned and the moved arm.

   A sparser schedule removes both. How sparse depends on whether the lines are the base or an arm.

### 4.4 One ruling: learned feedback weights

**The question.** Is a feedback pathway inside the line when its weights are separate from the forward weights and learn by a local rule, so that they come to approximate the forward weights?

**Why it needs saying.** Ruling 1 of 10 November puts exact weight transport outside the line. A learned pathway never copies a weight, but where it converges it delivers what transport would. Mr. Xu's answer that feedback may be "relayed layer by layer or by some sub-network / hyper-connection" seems to me to cover it. It changes what the rule is, so it should be a ruling and not my reading.

**What it would be for.** Two open results may both need feedback that is closer to the gradient than weight signs are:

- the every-byte collapse, if E24 lays it on the sign-feedback configuration and not on the loss;
- the sweep through a GRU's recurrent weights, if 4.2 shows alignment falling with steps back.

**What is known, and how far I checked.**

- Akrout and colleagues, "Deep Learning without Weight Transport" (NeurIPS 2019, arXiv:1904.05391), propose two such mechanisms: a weight mirror, and a modified form of Kolen and Pollack's 1994 rule. Their abstract reports that on ImageNet both outperform feedback alignment and sign symmetry and come close to backprop.
- In Kolen and Pollack's rule, as a search summary states it and as I recall it, the forward and the feedback synapse receive the same adjustment and the same decay, and so converge.
- I confirmed the paper and its abstract by search on 11 November. I could not open the paper itself from this session, so I quote no figure from it.
- My search found no use of either mechanism on recurrent weights or through time.
- Under the coordinator's rule 8, all of this needs a source check before it steers a design.

**A diagnostic that needs no ruling, and could come first.** Mix the weights used on the way back between the rule's (signs with the mean magnitude) and the true weights, with one coefficient, at the sign rule's own weight decay. Births under every-byte targets at three or four values of the coefficient would show how exact the feedback has to be before the collapse goes away.

- Its end point, true weights at decay 0.25, is the matched-decay control that E24 does not run.
- It uses exact weights as a diagnostic only, as E24's balanced arm uses the answer mask.
- It needs a switch in shared frozen code and its own check. The coordinator's scaled-against-unscaled comparison (9.84, item 4) sits in the same place and could share the switch.

### 4.5 Make E24 the last experiment on the two-bank network

- **The design has lost its claim.** The addendum to 9.82 says the fixed-rate two-bank design "has no claim to stay if E20-B finds a windowless network that composes from memory". E20-B found two.
- **E24 is still worth running as briefed.** It is audited, cheap (the worker forecasts about three hours for the nine births), and its question was observed on this network three times. I would add no arm to it.
- **Its result carries over as a pointer only:** whether to test the loss or the feedback first on the GRU. The cause of the collapse is open. The coordinator's own hypothesis missed its check (9.89).
- **After it, nothing more on this network:** no repair, no predictable-filler stream, no retention curve. Each of those is worth its compute only on the network that composes.

### 4.6 Route every commission through the coordinating session

**What the record shows for 10 and 11 November.**

- E18b, E18c and E23 were commissioned directly in another session, on a plan that marks itself as awaiting Codex (9.96 D). Their results are useful: rows 3 and 7 of section 2 come from them.
- That job changed the harness from a different starting commit. With E21's branch that made three versions (9.96). The merge had three conflicts. It would also have carried in a line that stopped every command on the window architecture at import, which none of the three branches' own checks had seen (9.98 A). The harness was one version again about seven hours after the three were noted.
- E20-B's full dose began twelve minutes after its stage 1 record, on a release the coordinating session says it did not give (9.96 C, 9.98 C).
- E22's second version ran unaudited (9.85).
- On the afternoon of 11 November the rule of 9.82, every brief to Codex before a worker starts, caught two wrong baselines in the E24 brief and a check that did not test what it claimed (9.99).

**What I suggest.** These releases are Mr. Xu's to make, and nothing here questions that. I suggest one route for them:

- a job goes to the coordinating session as one line;
- that session writes or amends the brief from the current head, Codex reads it, and the worker starts from the current head;
- an urgent job can skip Codex's reading on Mr. Xu's word, recorded as a deviation as 9.98 C does. It should not skip the current head.

The route costs an hour or two for each job. The merge of 11 November cost about seven hours.

## 5. The close on 17 November

I would aim for three things on one page.

**1. The candidate, as a pair.** A rule and a network, with the run in which that pair was born and lived, set against backprop on the same pipeline and host. As of today no such pair has been run. Section 4.2 is the shortest path to one. If the week ends without it, the close should say so in those words.

**2. The table by aid removed.** A first filling from the record:

| Aid | With it | Without it | Open |
|---|---|---|---|
| The answer mask | The rule lives within 3 points of the backprop-born learner (E22) | The rule collapses at birth and in life. Backprop holds in life | Backprop at birth is E24. No network that composes has been run without the mask |
| Operands in view | The window reference composes | A GRU and an LSTM compose from memory under backprop. The two-bank network does not | The rule on a network that composes |
| Replay, half of every batch | What is rehearsed is kept: 0.92 to 0.95 of the reference on mod-97 (Phase 4, 9.70) | The hard pairs end near 1% under either update | The cost curve |
| Backprop at birth | | Rule-born births exist and live as well (E18, E22) | Removed for single operations only |
| State reset between items | Every learner on file | | Not tested |
| Labeled items written by the harness | Every result on file | | The target stream. Step 0 only |

**3. What arithmetic leaves to the target test,** and what a learner may bring to it (4.3).

## 6. What my earlier notes got right and wrong

- **The two runs I asked for** (next-steps note, 3.3 step 1 and 3.4) were run the same evening. Both came out as the note said they might: the forgetting belongs to the stream, and the every-byte life fails under the rule while backprop holds.
- **3.2 held.** E20-B chose a windowless network and the fixed-rate banks have no claim to stay. One sentence there, that a miss by all three networks would point to the teaching, was withdrawn by the coordinator after Codex's audit (9.90). I agree with the withdrawal.
- **The reverse-replay remark in 3.1 is withdrawn.** The source check (9.84) shows that what runs backward in those studies is reactivated state, not an error signal.
- **3.6 was adopted and then passed over three times in two days** (4.6).
- **My chat review of 11 November** said the taught pairs "cover 2.3% of the mod-97 addition table". That sentence invites a comparison with the fraction sweep that does not hold: there the operands were digits in view. Section 4.3 states it properly.
- **The same review** called E24's stop "most likely the known Mac-versus-cloud difference". Section 4.1 now shows that the cloud side is one result across two programs and two hosts. The other side still needs its one rerun.

## 7. Suggested order

| When | What | Compute |
|---|---|---|
| Now | Release E24 on cloud baselines, with the two extra D-ans reruns in the first wave (4.1) | Eleven births. The worker measured about 3,200 CPU-seconds each |
| Now | D-ans 73129 once on the owner's machine (4.1) | One birth |
| Now | The ruling (4.4) and the route for commissions (4.6) | None |
| Now, beside E24 | The hand-written GRU and LSTM steps, their two checks, and the alignment measurement on E20-B's files (4.2) | No training |
| Now, beside the audit of step 0 | The sizing sweep on world seeds that are never scored (4.3) | Seconds to tens of minutes per cell |
| After E24 | Its one follow-up, on the GRU: the mixing diagnostic if the configuration is at fault, a predictable filler if the loss is (4.4, 4.5) | To be forecast |
| After the alignment measurement | The GRU under backprop with every-byte targets, birth and life. Then the GRU born and living under the sweep against backprop on the same pipeline (4.2) | E20-B measured about 2,900 CPU-seconds for a GRU birth and 4,600 for a life with the library layer. A hand-written step will cost more |
| After E21b and the network decision | The retention cost curve | To be forecast |
| 17 November | The close (5) | None |

## 8. What I would not do

- Start a GRU birth under the sweep at full dose before the alignment measurement and a reduced-dose dry run.
- Read a difference of less than about 3 points between runs on different host classes.
- Spend more audit rounds on the rules for moving disclosure lines before the sizing sweep says whether the lines are an arm or the base.
- Repair, extend or re-teach the two-bank learner after E24, or add arms to E24.
- Take the selection of the GRU over the LSTM as a finding about the two networks.

## 9. What I checked, and what I did not

Checked, all read-only, each with a script in this folder that reads the files with `git show`:

- **E24's host check** (`e24_host_compare.py`, `compare_pt_tensors.py`). The three births of seed 73129 as in the table of 4.1: equal scores and losses at six checkpoints and 24 of 24 tensors byte-equal between the cloud rerun and E20-B's G0 birth; 1 of 24 against the original. The final hard means of the other two seeds.
- **E20-B** (`e20b_recompute.py`). All 24 runs have a completion record, the registered step count, 6 curve rows for a birth or 7 stage records for a life, and a weights file whose SHA-256 equals the recorded hash. The report's table agrees in every cell. The selection rule gives G1 and G2 three seeds each and G0 and G3 none. The interrupted G0 lives equal their reruns at all five stages reached.
- **E22** (`e22_recompute.py`). The table of 9.85 and the six weights hashes.
- **The two runs of 9.88** (`zero_replay_every_byte_recompute.py`). Every figure quoted in rows 1 and 2 of section 2.
- **Code, by reading.** `arch20.py`: G1 and G2 use the library's recurrent layers, and `forward` refuses the local path. `arch17.py`: fixed rates, a write gate that reads the input only, `lin()` on every weighted crossing. The step 0 generator's size table against the worker's report.
- **The 15% figure:** reviews 9.2 and 9.3.

Not checked:

- **The step 0 code.** I read the report, the size table and the interfaces of the solver and the oracle. I did not audit the generator, the checks or their fixtures. That is Codex's audit.
- **E21b.** Commit messages only.
- **E18b, E18c, E23, E20-A and the probes of 9.93 and 9.95.** As filed.
- **E20-B's tensor hashes.** I checked file hashes. The tensor hash needs the runner's own function.
- **Why the owner's machine differs.** Not diagnosed. The test is step 4 of 4.1.
- **The alignment figures on file for the two-bank learner.** I did not recompute any.
- **Literature.** One search confirmed the paper named in 4.4 and its abstract. The paper could not be opened from this session. The statement about sign feedback through time is the coordinator's literature check (9.84), not mine.
- **CPU figures.** They are the workers' own, from the E20-B report and E24's host-check report.
- **The expectation in 4.3.** It is reasoning from the record. No learner has been run on the target stream by anyone.

## 10. Files added with this note

| File | What it does |
|---|---|
| `compare_pt_tensors.py` | Compares two saved weight files tensor by tensor without importing torch |
| `e24_host_compare.py` | The three births of seed 73129 side by side, and the host effect for all three seeds |
| `e20b_recompute.py` | E20-B's completion, hashes, table, selection and interrupted lives |
| `e22_recompute.py` | E22's table and hashes |
| `zero_replay_every_byte_recompute.py` | The two runs of review 9.88 |

Each is standard library only and is run as `python3 -I <script>` from inside a clone that has the branches fetched.
