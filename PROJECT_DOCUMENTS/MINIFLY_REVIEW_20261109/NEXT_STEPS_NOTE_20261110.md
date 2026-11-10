# Suggestions for next steps after 10 November

- **From:** the audit session (the cloud session that wrote the memo and the roadmap note in this folder).
- **For:** the coordinating session and Mr. Xu.
- **Read for this note:** `codex/minifly-consolidated-20261023` at `9152dc62`: the state addenda of 9–10 November, review sections 9.59 to 9.81, the coordinator's note, Codex's two audits, its status note and its delayed-observation memo, and the E20, E18b, E16-M4 and multi-timescale briefs.
- **Status:** suggestions. The auditor audits and the coordinating session decides. This round is a read of the record, not an audit. I recomputed no scores. I checked three things in the harness and the receipt list, and section 7 says which.

## 1. Summary

1. **Close the open rulings now.** I would accept the backward sweep over the item with weight signs, accept a bounded window, and let references use any optimizer.
2. **If the backward sweep is accepted, drop the fixed-rate bank design** and let E20-B choose the windowless network.
3. **Put retention without rehearsal on the main line.** It is the largest measured failure and the queue holds only a literature check for it. One control is missing: backprop at zero replay.
4. **Run a life with every next byte as the target, under the corrected rule.** It is Mr. Zhou's delayed-observation idea in its plainest form, and the harness can run it today.
5. **Write one target test, design only.** Codex's promotion test, Codex's delayed-observation test and my battery have the same endpoint and can be one document.
6. **Two process changes:** every brief to Codex before any worker starts, and no partial receipts on the default branch.
7. **Give arithmetic one more week, then close it.**

Codex's delayed-observation memo helps. Section 4 says how, and where I would be careful.

## 2. What the last two days change in my earlier notes

- **The propagation reading in my memo is void.** Mr. Xu's answer of 10 November (review 9.67) puts feedback relayed layer by layer, or by a separate pathway, inside the line. Cases A and G are in.
- **The births question matters much less than I said.** E18's arm B0 was born with no backprop anywhere at 0.94–0.96 of the backprop birth. The fork in section 6 of my roadmap note is mostly dissolved. The ruling the plan now turns on is the backward sweep (3.1).
- **Stage 1 of the roadmap has in effect been run.** The rule inside the line fails at retention without replay, passes the hard-pair gate in 2 of 15 seeds, and has no architecture on file that composes from memory.
- **Section 2.2 of the roadmap note** is to be read with the literature check's corrections (review 9.72).
- **Stages 2 and 3 stand.** Suggestion 3.5 below is the roadmap's battery in the form the two Codex notes now share.

## 3. The suggestions

### 3.1 Close the open rulings

Each one prunes experiments, and none needs compute.

| Open question | Suggested answer | Why |
|---|---|---|
| A backward sweep over the item with weight-sign feedback (the Phase 4 rule) | Accept it as the working life rule, with the sweep length a declared budget | It is the only rule on file that trains a windowless learner (0.92–0.95 of the reference on mod-97). It costs what backprop costs (48.7 against 49.4 ms per update). The forward-only form reached 0.63–0.79 of its ceiling at about twice the cost (E19) |
| A bounded input window | Accept it as a sensory buffer | The defect was never the window. It was a task that fitted inside it |
| Backprop for births | Any optimizer for references. A candidate is born by its own rule before any promotion claim | E18's B0 shows this skill needs no backprop birth |
| Whose rulings are final | Say it once in the record | Reviews 9.60, 9.67 and 9.70 each carry "Mr. Zhou's to decide" forward |

What accepting the sweep means should be written down with the ruling:

- The life rule is then backprop through time with weight signs in place of weights. What stays outside the line is exact weight transport.
- The rule can no longer be the project's novelty. Memory, retention and cost can.
- If the sweep is ruled out, the windowless learner has no working rule on file, and the forward-only line (E19) is the only one left.

One remark from memory, to be source-checked before anyone quotes it: rodent hippocampus replays recent sequences in reverse, which is the nearest biological analogue of a backward sweep over recent experience that I know.

### 3.2 If the sweep is accepted, drop the fixed-rate banks

- **The banks exist for a property the working rule does not use.** They are diagonal with fixed rates so that credit can be carried forward exactly, one trace per synapse (`arch17.py` header). The Phase 4 rule carries credit by a backward sweep. E19, which does use the property, is parked.
- **The same design is the one that does not compose under backprop** (1–7% on untaught add/subtract pairs, review 9.76).
- **So decide before E20-B is read, not after.** If the GRU or the LSTM meets P20.8 and P20.9, it becomes the windowless candidate. The next run is then the sign rule with the backward sweep on that network, against backprop on the same pipeline.
- **That run is new ground.** The literature check found no study of sign feedback through time, and gated cells have multiplicative paths the banks lack. It may fail, and that would be worth knowing early.
- **If all three networks miss P20.9,** the brief's own reading applies: the question is the teaching, not the network.
- **Keep the banks only if the sweep is ruled out.** They are then the only design with a forward-only rule.

### 3.3 Retention without rehearsal, on the main line

Why it belongs there:

- With no replay the hard pairs end at about 1% and mod-97 at 0.13–0.35 of the reference (review 9.64).
- In Phase 4 the rule kept 11–43 on operations it never rehearsed, where backprop on the same pipeline kept 41–54 (review 9.70).
- It is the cost concern in Mr. Xu's third answer. The queue holds one literature check for it, and the replay job still running (E16-M4) tests shares of 0.875 and 0.9375.

Three steps, smallest first:

1. **The missing control: backprop at zero replay.** E16-M3 ran the rule at share 0 and backprop on the pipeline at 0.25, 0.5 and 0.75 only. Three lives close it: arm `M` with `E16_RSHARE=0` on seeds 400009, 600013 and 300007, with E16-M3's other settings. The options exist today; a dry run confirms the command shape.
   - If backprop also ends near 1%, the forgetting belongs to the stream under any update.
   - If it ends well above the rule's 1%, the forgetting is the rule's.
   - Until then "under the sign rule the learner retains nothing" has no comparator.
2. **A cost curve.** Keep the fresh items, their order and their count fixed, and add replay as extra updates at several levels. Plot retention against total CPU for both updates. This is the owner's question in its direct form, and it removes the allocation confound of review 9.75 item 3. It needs one new option and the usual qualification.
3. **One protection mechanism,** after the multi-timescale check reports: a slow and fast weight, or a consolidation term, local to the synapse. Test it at zero replay and at one small level taken from step 2, under backprop first and the rule second.

Stop rule: if the best mechanism under backprop retains no more than plain backprop at the same cost, replay stays and the design budgets it. Separate bounded blocks of replay would then be the form to cost, in place of half of every batch.

### 3.4 A life with every next byte as the target, under the corrected rule

- **Why.** This is Mr. Zhou's delayed-observation idea in its plainest form: the stream's own next byte is the only teaching signal. The answer mask is also the aid that most separates this harness from any real stream.
- **What is on file.** Only E16-L of 8 November: the window learner, under the rule as it then was, acquired addition and subtraction partly in block 1 and lost them from block 2. The reference under the same loss was unaffected.
- **What has changed since.** The rule gained credit through time at every layer. At a trained point the cheap rule's alignment under every-byte targets measured 0.50–0.77, against 0.16–0.23 under answer bytes (the E18b brief). E18b tests births only.
- **The run.** Six lives on existing options, three seeds, after a dry run of each command shape:
  - the windowless learner with `E16_LOSS=all` under the Phase 4 rule (`E16_ARCH=bank2 E16_BOOKDECAY=off E16_TRACEUP=all`);
  - backprop on the same pipeline (`M`) under the same loss.
  
  The two differ in the update alone.
- **A limit.** `E16_TRACEUP=all` is refused for the window architecture, so the window learner cannot be rerun under the corrected rule without new code. If the windowless life works, that port is worth making.
- **If it fails while `M` holds,** the next single change is predictable distractors. That is the cause of E16-L which Codex's memo calls unresolved.
- **Order.** Before E18b and before the delayed-observation test of section 4. Both read differently depending on this result.

### 3.5 One target test, design only

Three proposals now ask for the same thing:

- Codex's promotion test (learn, revise, act; the status note of 10 November);
- Codex's delayed-observation test (the memo of 10 November, its last section);
- the battery in my roadmap note.

They differ in the stream. They agree on the endpoint: one continuing learner, scored on its first output, on pairs of held cases that share their recent bytes and differ in earlier history, both members correct. E20's twins are already a prototype of it.

I suggest one design document that merges them:

- **Teaching signal:** the stream's own later bytes, with no answer mask.
- **Endpoint:** Codex's paired score before the reveal.
- **Controls:**
  - targets shuffled within delay and frequency strata;
  - a next-byte comparator at the same target budget;
  - an oracle and a visible-suffix lookup;
  - a branch without the relevant earlier association.
  
  The last is the prior-teaching control that the objective contract requires for evidence level E3. Nothing in the E-series has run it.
- **References:** backprop on the same observations, on an ordinary recurrent network, before any candidate.
- **Later:** a revision step, and the cost ledger throughout.

Design it now, have Codex audit the brief before any worker starts, and run it after E20 and 3.4 report. The coordinator deferred the promotion test as premature. Running it is premature. Designing it is what stops the question changing each day.

### 3.6 Two process changes

- **Every brief goes to Codex before any worker starts.** The first version of E20-A was withdrawn before a worker touched it only because the audit happened to arrive first. That saved a roster. Make it the rule.
- **No partial receipts on the default branch.** They were 129 of the 204 commits between `c8cfbbbc` and `b168c922`. With several writers each one forces a rebase, and the history is hard to read. A scratch branch, or one commit at completion, would do.

### 3.7 One more week of arithmetic, then close it

- **In the week:** E20, steps 1 and 2 of 3.3, and 3.4. E16-M4 and E18b finish and are filed.
- **Then a one-page close,** organised by the aid removed: the answer mask, composition in view, the replay share, supervised labels, the state reset between items.
- **Then the stream of 3.5.** Outside baselines exist there, and the gates are not written by the agents who run them.

## 4. Codex's delayed-observation memo

It helps in three ways:

1. **It gives Mr. Zhou's idea a form that can fail.** The two controls that matter are in it: targets shuffled within strata, and a next-byte comparator at the same target budget. Without them a gain from later views cannot be told from a gain from more targets.
2. **Its endpoint is the promotion test's.** That is why the three proposals can be one design (3.5).
3. **Its order is right.** Test whether the observation is useful before changing the rule, and the credit path after.

Where I would be careful:

1. **It is written for the existing two-bank model.** That model does not compose under backprop and may be dropped (3.2). Build the assay independent of the network and qualify it on an ordinary recurrent reference first.
2. **It adds a new world and new panels.** The coordinator's rules 11 to 13 apply in full: an oracle that must pass and a suffix-only responder that must fail, before any roster.
3. **Two cheap results should come first:** the every-byte life (3.4) and E20.
4. **Its literature items are Codex's reading.** I opened none of the sources. Under the coordinator's rule 8 they need a source check before they steer a design.

One reading of my own, for what it is worth. A target at every step and every stage removes the need to relay error through depth. It does not remove credit through time for what must be remembered: a synapse that wrote an operand twelve bytes ago is still credited only when the answer arrives. E18 is consistent with that. Local heads worked with credit through time inside each bank (66–82), and the arm without it learned nothing, although that arm also could not learn its input code, so E18b has to report before this is more than a reading.

What the memo does not touch is retention (3.3). Later observations supply targets. They do not protect what was learned earlier.

## 5. Suggested order

| When | What | Compute |
|---|---|---|
| Now | The rulings (3.1), the target-test design (3.5), the process changes (3.6) | none |
| Now | Backprop at zero replay (3.3 step 1) | three lives, about 4 CPU-hours |
| Now | The every-byte life (3.4) | six lives, about 11 CPU-hours |
| In flight | E20-A, E20-B, E16-M4, E18b, the multi-timescale check | as briefed |
| After E20 and 3.4 | The network decision (3.2), the cost curve and one protection mechanism (3.3 steps 2 and 3) | to be forecast from dry runs |
| After one week | Close arithmetic (3.7) and qualify the target test on references | to be forecast |

## 6. What I would not do

- Launch the delayed-observation test, or any new world, before its assay has passed an oracle and failed a suffix-only responder.
- Commission repairs to the two-bank design before E20-B reports.
- Extend the replay-share line beyond E16-M4.
- Add seeds to any existing gate.

## 7. What I checked, and what I did not

Checked for this note:

- **No backprop life at zero replay exists.** The receipt list on the default branch has arm `M` at shares 0.25, 0.5 and 0.75 and the rule at share 0.
- **The only every-byte lives on file are E16-L's six,** from 8 November.
- **`E16_TRACEUP=all` is refused for the window architecture** (`exp16_integrated.py`, the assertion after the option is read), and `update16` applies the every-byte mask for arm `M` as well as for the rule.

Not checked:

- Any score in the review sections. The coordinating session recomputed the workers' numbers and Codex audited them; I took both as filed.
- The sources behind Codex's memo and behind the literature check.
- The state of the jobs in flight.
- The CPU figures in section 5 are scaled from the receipts of E16-M3 and E17 Phase 4, not forecast from dry runs.
