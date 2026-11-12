# Draft for audit: a sizing sweep for the target test (11 November 2026, 20:26 EST by the clock)

> **Status.** Released with changes by Codex's audit of revision 2, **for stage 1 only**; the full roster is held until the coordinator has read stage 1's record (revision 3 below). Coordinator's design, adopted from the audit session's note of 11 November (its section 4.3; review 9.101 D.3). **This is a sizing exercise. It registers no prediction for judgment, gives no verdict, and concludes nothing about any learning rule.** Its only product is a table that says at what sizes a learner can acquire anything at all from this stream.

> **Revision 2, 11 November 2026, 20:34 EST (clock), after Codex's audit of this brief** (`PROJECT_DOCUMENTS/MINIFLY_TARGET_SIZING_SWEEP_AUDIT_20261111.md`), whose disposition was to hold learner training as written. All four required revisions hold and are adopted:
> 1. *The modulus did not reach the evaluator.* The generator, the oracle and the solver each carry their own modulus, and the checker reads the generator's when it is imported. The coordinator ran it: with the generator set to 31, the oracle and the solver still compute modulo 97. Every smaller-modulus cell would have been scored against the wrong answers. One explicit configuration is now passed to every component.
> 2. *The decision rule for taught pairs was defective.* It asked for taught pairs "above the tables". A table keyed by context and cue recalls every taught pair it has seen, so nothing can be above it. Taught recall is now read against that table as its ceiling, and held behaviour is read separately against shortcuts.
> 3. *A probe inside a training chunk was not defined.* Saving the hidden state cannot undo an update that has already used the probe's reveal. The order of events at a probe is now fixed and tested.
> 4. *The cost was wrong.* The roster is 81 fits, not 18 cells, and the longest stream is 31.6 times the proposed one, not 64 times as the first draft said: only the first segment's repetitions scale.
> The job is now in two stages: implementation with a bounded qualification, then the roster on the coordinator's release. Still a draft; it goes back to Codex.

> **Revision 3, 11 November 2026, 20:46 EST (clock), after Codex's audit of revision 2** (`PROJECT_DOCUMENTS/MINIFLY_TARGET_SIZING_SWEEP_REV2_AUDIT_20261111.md`): release stage 1 with changes, hold the full roster. The four changes, all adopted:
> 1. *Twins are forked before either prompt.* Revision 2 restored a state saved at the first member's `=`, which already holds the first member's context. Both members now start from one state saved before the first byte of either. The boundary before a probe is set by the last loss target used, not the last input read: a chunk that ends with `=` as input has already been trained on the first answer digit.
> 2. *The 20-point margin applies to recitation shortcuts only.* In the streams with lines, both stated values have been disclosed before all 96 of 96 first-encounter probes (step 0's own record), so a responder that adds them scores near 100% and no learner could clear it by 20 points. That responder is reported beside the learner as a benchmark of direct use. The size decision is pinned to the acquisition phase.
> 3. *The matched extra-bytes stream is trained on,* in three fits, so that the lines' content can be told from their bytes. The roster is 84 fits.
> 4. *Stage 1 is bounded,* and the parameterized modules are copies under `sizing/`; step 0's own files are not edited.

## Why

Step 0 is done (reviews 9.101 C, 9.105, 9.106): at the proposed sizes every held answer of the target stream is determined by the bytes before it. That is identifiability. It does not show that a learner can acquire those answers, and the brief of step 0 asked this itself as its first question.

The proposed sizes are small in a way this project has not tried: 216 taught pairs per family, each shown four times before the first panel, with letters in place of digits, so that the learner has to find 36 hidden numbers and the addition that joins them from outcomes alone. The audit session expects that no learner starting from nothing acquires the held pairs at these sizes under any update. If that is right, the design would stop at its reference step after several more rounds of audit, at sizes that have to change. A sweep that costs minutes per cell says so now.

## What is varied

**One configuration for every component.** The generator, the oracle, the prefix-only solver and the checker each take the same explicit configuration (modulus, counts, repetitions), and each still does its own computation from it; none reads another's constants. The parameterized modules are copies placed under `TARGET_TEST_STEP0/sizing/`; step 0's own four files are not edited. Two conditions: at the default configuration the copies reproduce the accepted record `TARGET_STEP0_CHECK.json` for world seeds 1, 2 and 3, every hash and every value, with all ten checks passing; and the following fixtures exist for addition and for subtraction at each of 97, 31 and 11: a pair whose sum or difference wraps round the modulus is answered correctly by oracle and solver, and an oracle or solver deliberately left at another modulus is caught. A configuration that does not reach a component stops admission for the cells that use it. Every cell records the configuration it ran.

| Factor | Values |
|---|---|
| Modulus | 97 (proposed), 31, 11 |
| Presentations of each taught pair before the first panel | 4 (proposed), 32, 256 |
| Disclosure lines in the stream | none (the ordinary stream); the aligned lines (`X:vv c:uu` after each episode) |

Nine settings of arm A and eighteen of arm B (below), each on **world seeds 101, 102 and 103**, which are never scored in any later step: 81 fits. Seeds 1, 2 and 3 are not touched as worlds. One further stream is trained on at the proposed size only (modulus 97, four presentations), in arm B, on the same three worlds: the stream with as many extra ordinary bytes in the lines' places (step 0's extra-observation control), so that at one setting a gain from the lines' content can be told from a gain from more bytes and shifted cuts. **84 fits in all.**

A smaller modulus changes the task as well as its size: the answers' distribution and the digits' difficulty change with it. Accuracies at different moduli are therefore figures for different task settings and are not compared as if on one scale.

## The learner, one for every cell

A GRU of E20-B's shape (byte embedding, one layer to 512, a GRU of 256 units, the two-layer readout; the library layer), trained by backprop with AdamW at the settings of E20-B's births, **starting from random weights, with every next byte as the target and no answer mask.** No rule, no sign path, no replay store.

Two ways of feeding the stream, because they answer different questions:

- **A. Episode by episode.** Each episode is one item and the state starts from zero. The network can use only what is in the episode and what its weights have stored. In the ordinary stream that is the context letter and the cue; an answer on a taught pair then shows the weights have learned that pair, and an answer on a held pair shows they have learned the values and the addition. This arm says whether the hidden structure is learnable from outcomes at all at a given size. Disclosure lines cannot help here by construction, so arm A runs on the ordinary stream only: nine cells.
- **B. As one continuing stream.** The bytes are read once, in order, with the state carried from byte to byte and credit cut every 256 bytes. Nothing is reset. This is how the target test itself runs. It can use a line read earlier only if the state still holds it. Both streams: eighteen settings.
  - 256 bytes is a limit on credit, not on memory: the state is carried across a cut, so information can survive it, and even a short dependency can straddle one. For every use of a line the report logs the distance in bytes from the line to the probe that needs it and how many cuts lie between. Before a null on the lines is read at all, one controlled dependency that crosses a cut is shown to be learnable by this learner (a fixture stream in which a byte must be repeated a fixed distance later, across a cut).

One pass over the stream in arm B, since a continuing learner sees its stream once. In arm A the same episodes in the same order, once. No second pass in this sweep; whether more updates on the same data would help is a separate comparison.

**Pinned for every fit, and recorded:** the model seed (the world seed plus 1000), AdamW at E20-B's birth settings, the batch (arm A: 64 episodes in stream order; arm B: one stream, chunks of 256 bytes), the loss as the mean over the targets of a batch or chunk, and one update per batch or chunk.

## A probe inside the stream (arm B), exactly

A chunk of 256 bytes can hold a probe and the bytes that reveal its answer. The order of events is:

**Where the last update before a probe ends.** The learner predicts the next byte, so a chunk whose last input is `=` has the first answer digit as a target and has been trained on it. The last update before a probe is therefore the last chunk whose final *target* comes before the probe's first answer byte. Chunks are cut there if need be; the cut positions are recorded.

**A first encounter** (an episode of the stream itself):

1. Training proceeds chunk by chunk to the boundary just defined. The state, the byte cursor and the weights at that boundary are the probe's starting point.
2. The learner reads forward to `=` with those weights, carrying its state. No update happens.
3. After `=` it writes its whole answer, two digits and the end byte, feeding back its own bytes.
4. The state and the cursor are restored to their values at the boundary of step 1. Training continues from there over the true bytes, reveal included, exactly once.

**A sealed twin** (two episodes that are not in the stream, inserted at the panel's place):

1. At the panel's place the state, cursor and weights are saved, **before the first byte of either member.**
2. From that saved state the learner reads the first member's whole prompt, context included, and writes its answer.
3. The saved state is restored, and the second member is read and answered from it in the same way. Neither member sees the other.
4. The saved state is restored again and the stream continues. A twin leaves no trace.

Shown by fixture on the real network and training path: changing the bytes of the reveal and of everything after it leaves the response unchanged; a chunk ending with `=` as input is shown to change the response to that probe, so that the boundary rule can fail; the second member of a twin gives the same answer whether or not the first was presented before it; and a run with probes scored and a run with probes skipped agree afterwards in weights, optimizer state, carried state, cursor and every counter. The change point of the revision phase is given to the evaluator and the solver only and never appears in the learner's input.

## What is read, at the first panel and at the end of the stream

Two things are read and never mixed.

*Recall of what was taught.* Step 0 has no taught panel, so one is built and frozen before any fit: for each world, 96 taught pairs per family that are not changed by the revision, each rendered twice, once with filler of 10 to 14 bytes (the context out of the last ten bytes) and once with filler of 0 to 2. The reference beside it is the exact-key table (context and cue), fed the same prefix: it recalls every taught pair it has seen, and it is a ceiling for recall, not a shortcut to beat. The short-filler version says whether a miss is a failure to learn the pair or a failure to hold the context across the filler.

*Behaviour on what was never taught.* First-encounter pairs at their first occurrence, before the reveal; and sealed twins, both members right, counted over the 12 distinct twin pairs per family as well as over renderings. Beside them, two kinds of reference, kept apart:

- *recitation shortcuts*, built from the same arm's actual prefix: the ten-byte-suffix table, the exact-key table and the recency table. These are what a learner must beat;
- in the streams with lines, *the direct-use benchmark*: a responder that takes the two values stated in the lines seen so far, for the probe's family and for the era the probe is in, and applies that family's operation. With a line after every episode it has both values before every first-encounter probe and scores near the ceiling. It is not something to beat; it says what using the lines perfectly would give.

The prefix-only solver is the ceiling.

Every figure is given by world, by family and by phase, with the number of times the probed pair, its context and its cue had occurred before the probe.
- The mean next-byte loss on outcome bytes and on all other bytes, per tenth of the stream.
- CPU-seconds per cell.

In arm B the probes are scored the way step 0 fixes: the learner's state is saved before a probe and restored after it, and both members of a twin are answered from the same state.

## What the table is for

It is read by eye and decides three things about the design, none about a learner:

Each fit is put in one of five classes, fixed now:

- *held success:* in that world, **at the first panel and on the acquisition phase only** (the 32 first-encounter pairs of the acquire set and the sealed twins scored at the first panel), first-encounter accuracy at least 20 points above the best *recitation shortcut* on the same probes, and both members right on at least 20% of the distinct sealed twin pairs. The retention and revision phases are reported in full and do not enter this class;
- *taught-only success:* long-filler recall of the frozen taught panel, read at the first panel, at 50% or more, without held success;
- *negative:* the fit completed, admission passed, and neither holds;
- *missing:* not built or not admitted, with the reason;
- *capped:* stopped by the budget.

A setting counts for a decision only if its three worlds agree; otherwise it is reported as mixed.

1. **Can the proposed sizes stay?** Held success in all three worlds at modulus 97 and four presentations supports taking those sizes to the qualification of step 1. If that setting is negative, this says the tested GRU and recipe did not acquire the held answers there. An all-negative grid leaves this GRU and recipe unqualified for the task; it does not show the task is beyond every learner.
2. **What may a learner bring?** Held success from random weights at a setting shows that no birth curriculum is needed there. A negative leaves the value of a curriculum open: this sweep cannot say whether one would help. If the table leaves that open, the answer comes from a separate comparison, prospectively registered: the same learner from random weights and from a warm start, the warm start trained on worlds with their own independent values and no test pair, with the cost of that pretraining charged to it. That is not part of this job.
3. **Should the stated values be the base arm?** Ordinary against lines measures the lines as a package: their content, the extra targets they add, and the shift they cause in where the credit cuts fall. At the proposed size the three fits on the stream with matched extra bytes say how much of a gain is not content: lines against extra bytes, world by world, is the comparison for content. Before any reading, the ledger of what had been disclosed before each probe is checked for the fits compared. A sparser line schedule is a different stream and needs its own qualification.

## What this is not

- Not a reference for step 1. The learner here is not qualified against anything, and its scores are not baselines.
- Not evidence about backprop against the rule. One learner, one update.
- Not on the scored worlds.
- Not a search for sizes that make a result look good. The grid is fixed above; cells are not added after the table is seen. If the table shows the whole grid is in the wrong place, that is reported and a second grid is briefed and audited.

## Admission

- The stream builder at the proposed sizes, for world seed 1, still gives the hashes on file (`TARGET_STEP0_CHECK.json`), which shows the copy-and-set route changes nothing by itself.
- At every changed size the step 0 checks that do not depend on a learner (identification, oracle, exposure, twins, determinism) pass, or the cell is reported as not built and why. A smaller modulus makes equal outcomes more likely; twins are redrawn as step 0 requires.
- A fixture learner that answers from the oracle scores every probe through the sweep's own scoring code, and one that answers from the ten-byte suffix scores no twin.
- State save and restore around a probe leaves the continuing learner's state unchanged, shown on a fixture.

## Running it

- New files only, all in `TARGET_TEST_STEP0/sizing/`, the parameterized copies included. The arithmetic harness is not imported; the GRU is built directly.
- One host, at most three processes on four cores, one thread each.
- **Two stages.** Stage 1: the parameterized copies with default parity and the modulus fixtures; the frozen taught panels; the probe protocol with its fixtures; the cross-cut dependency fixture; and a cost profile. Then stop and return the admission record and a forecast for the 84 fits. Stage 2, the roster, on the coordinator's release.
- **Stage 1 is bounded.** Profiling uses world seed 104 only, which is in no roster; nothing run in stage 1 counts as a fit, and its learners are discarded. Four paths are profiled, since batches of 64 episodes and single-stream chunks cost differently: arm A and arm B on the ordinary stream, arm B with lines, and arm B with matched extra bytes, each at 256 presentations for the first 500 updates, plus the full generation and learner-free checks of that longest stream. Stage 1 as a whole stays under two hours of CPU; if it cannot, stop and say why.
- Cost: only the first segment's repetitions scale. The proposed stream has 3,552 episodes; at 32 repetitions it has 15,648 and at 256 it has 112,416, which is 31.6 times the proposed one. Log, for every fit, the bytes read, the target positions, the updates, and the CPU for generation, checks, training and scoring separately. If the forecast for the 84 fits passes twelve hours of CPU, stop and report it; the coordinator then cuts the grid before anything runs, not after.
- Work on your own branch. One commit when the table is complete.

## Report

One table per arm (cells by modulus and presentations, three world seeds each, the four readings with the tables' and the solver's figures beside them), the loss curves, the sizes used in every cell, the admission record, CPU per cell, and whatever was unclear. No conclusion about any learner or rule.

## Questions for the audit (answered by the audit of revision 1: keep arm A as a reset-per-episode diagnostic of what the weights learned, noting that outcomes fix the hidden values only up to a shift within a family, so a held success shows reuse beyond the shortcuts and not recovery of the values themselves; keep arm B to one pass; treat 256 bytes as a credit limit and qualify a dependency across a cut; keep any warm start out of this grid)

1. Arm A resets the state at every episode, which the target test never does. Is it still the right probe of "learnable from outcomes at all", or does it answer a question the design does not ask?
2. Arm B reads the stream once. Should a second pass be allowed in some cells, marked as such, to separate "too little data" from "too few updates"?
3. Is credit cut every 256 bytes long enough for a line to be used at the next episode of its context, which is some hundreds of bytes later? If not, arm B with lines would fail for a reason that has nothing to do with the sizes.
4. Should one cell with a born learner (pretrained on other worlds, then run on a sweep world) be in this sweep, or is that a design of its own?
