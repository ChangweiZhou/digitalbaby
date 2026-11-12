# Draft for audit: E26, a GRU born and living under the sign sweep, beside backprop on the same pipeline and host (11 November 2026, 18:41 EST by the clock)

> **Status.** Stage 1 accepted and stage 2 released by the coordinator (revision 3 below). Released with changes by Codex's audit (`PROJECT_DOCUMENTS/MINIFLY_E26_STEP0_IMPLEMENTATION_AUDIT_20261111.md`), **for stage 1 only.** Coordinator's design. Evidence level aimed at E2, development panels. Three stages with a stop after each.

> **Revision 2, 11 November 2026, 18:53 EST (clock).** Every point of the audit was checked and is adopted:
> 1. *The update is specified exactly and qualified as it will run.* The harness's own rule update (`E.update(..., "T")`) cuts credit at every step and calls the bank's trace; it must not be used. One training loop serves both arms and switches only the path back. One real optimizer step is tested in each arm.
> 2. *The scorer is qualified on the written-out network,* including the end symbol and the silenced-state diagnostic, which the E25 helper does not support.
> 3. *All six lives run.* A weak birth can improve in a life, and running only the stronger births would change the comparison. The stop before stage 3 checks that the births are technically valid; it does not choose among them.
> 4. *Two readings are kept apart:* whether the sweep's arm meets the absolute marks, and how far it is from its backprop partner.
> 5. *Kept as drafted:* decay 0.25 in both arms, the joint stream first, one scale per stacked matrix. A backprop arm at 0.30 is a follow-up, wanted only if the matched reference disappoints and a comparison with E20-B would change a decision.

> **Revision 4, 12 November 2026, 09:52 EST (clock): the six births are admitted and stage 3 is released.**
> - *What was checked.* All six births are complete, their weights match their recorded hashes, their sources are those of 360e1b8d, both arms were fed identical data, and both read back decay 0.25 (`exp45/e26/E26_BIRTHS.md`). The coordinator recomputed the six birth curves from the receipts.
> - *The gap Codex found* (`PROJECT_DOCUMENTS/MINIFLY_OVERNIGHT_E21B_E26_REVIEW_20261112.md`): the birth loop clears its list of losses every 512 updates and records a finite-loss flag only at the six scored rows. The flags therefore cover 3,072 of each birth's 24,576 updates. Confirmed in the code.
> - *The coordinator's ruling: no update of any birth had a non-finite loss.* A non-finite loss or gradient at any update would be written by AdamW into the weights and into both moment estimates, and a NaN stays a NaN under every later update, decay included, since there is no clipping or reset in this loop. Every later loss would then be non-finite. The last 512 updates of every birth have finite losses, and its final weights answer 98 to 100% of 1,200 sealed items. So nothing non-finite entered any birth at any update. This is an inference from the record, not a direct observation, and it is recorded as a ruling.
> - *The lives are covered directly.* The life loop keeps every loss of a stage (6,144 updates) and flags them all, so the gap does not recur there.
> - **Before the first life starts, commit the six birth weight files** (`exp45/e26/*.pt`, about 4.5 MB each) to your branch, overriding the `.gitignore`. Containers have restarted at least four times this week; a lost birth cannot be remade with the same arithmetic on another host. They are also what the coordinator needs to check the lives' starting points.
> - **Stage 3 is released: all six lives,** as the brief says, with the sources of 360e1b8d unchanged. Then stop and report.
> - *Future receipts* in this project record a cumulative finite-loss flag and the number of updates it covers.

> **Revision 3, 11 November 2026, 20:18 EST (clock): stage 1 accepted, stage 2 released.** The worker's stage 1 record (`exp45/e26_qual/E26_STAGE1_RECORD.md`) reports every check passing except check 1, which fails in 1 case of 18 by a logit difference of 1.14e-5 against the limit of 1e-5, and it changed nothing to make it pass. **The coordinator's ruling: the limit was badly set, and the check is accepted.**
> - I set 1e-5 as an absolute limit from a fixture on untrained networks, whose logits are below 1. On E20-B's trained files the logits reach 25 to 36, where 1e-5 is a few float32 spacings. The failing difference is 2e-6 of the logit it sits on.
> - Everything that bears on the question agrees: the gradients match the library's to 3.7e-6 relative in all 18 cases; no answer position has a different top byte; the written-out step is bit-identical to `e25_alignment`'s; and the decoder reproduces E20-B's receipts exactly in all 70 scored blocks.
> - The coordinator repeated the decisive part on another host class (`e26_coordinator_check.py`, `exp45/E26_COORDINATOR_CHECK.json`): for all three seeds the composition counts through the production scorer equal the receipts, the worker's network equals the E25 helper bit for bit under both paths, and the largest difference from the library layer is 3e-7 to 4e-7 of the largest logit.
> - **Check 1's limit is restated as relative: a logit difference of at most 1e-5 times the larger of 1 and the largest logit in the batch.** Under it all 18 cases pass. This is a change of criterion made after the result, and it is recorded as that.
> - The reading at the end of `E26_PREDICTIONS.md` (which receipt field each quantity is read from) is confirmed.
> - **Stage 2 is released:** the six births at full dose, with the sources of commit 360e1b8d unchanged, into `exp45/e26`. Then stop and report. Stage 3 is released when the six births are technically valid.

## Why this job

The project has a rule and a network that have never met.

- **The rule** (sign feedback swept backward over the item, answer targets, replay) is born and lives within a few points of backprop on single operations, on the two-bank network (E22, review 9.85). That network does not compose, under either update (E20-A, E20-B).
- **The network** that composes from memory is the GRU, selected in E20-B by a rule fixed beforehand (review 9.101). It has only ever been trained by backprop.
- **E25** (review 9.103) measured the sweep's update on E20-B's saved weights. In a trained GRU it keeps a cosine of 0.71 to 0.76 with the true error twelve steps back, as good as or better than in the two-bank network. The reading registered before that measurement was: a GRU birth under the sweep is worth its compute, at a reduced dose first. Alignment is not learning; this job is the test.

The close on 17 November is to name a candidate as a pair, a rule and a network, with the run in which that pair was born and lived. No such run exists. This is it.

## The question

Can the sign sweep bring a GRU to birth, and through a life, to where backprop brings the same GRU, on the same data in the same order?

## One difference: the update

| | Arm S, the sweep | Arm B, backprop |
|---|---|---|
| Network | E20-B's G1: byte embedding (8), one layer to 512, a GRU of 256 units, the two-layer readout; 1,124,113 parameters; the same initial values from the same seed | the same |
| Recurrent step | written out from `exp3_state.lin`, as in `e25_alignment.py` (`forward_rnn`) | the same written-out step |
| Way back | weight signs scaled by the matrix's mean magnitude at every linear map; the gates' own nonlinearities and products exact; credit through the whole item | exact |
| Targets | answer bytes | the same |
| Optimizer | AdamW, rate 0.001, betas (0.9, 0.98), **weight decay 0.25**, embedding table without decay | **the same, decay 0.25** |
| Data | E20-B's birth and life protocol, same seed, same batches in the same order | the same |
| Host | one host for both arms of a seed | |

- **Why decay 0.25 for both.** Every rule run on file uses 0.25 and every backprop birth uses 0.30. Codex's audit of E24 showed what that confound costs. Here both arms use the rule's decay, so the update is the only difference. E20-B's G1 (the library layer, decay 0.30) stays on file as the reference this network is known to reach; it is not an arm, and its host and decay differ.
- **Why the written-out step in both arms.** It equals the library layer to rounding and not bit for bit, so a library arm and a written-out arm would not see the same arithmetic. Both arms run the same forward code; only the path back differs.
- **Why E20-B's protocol.** Birth: the four mod-89 operations, 24,576 updates of 64 items, 0 to 12 distractor letters. Life: six blocks of 6,144 updates from the joint stream of the families admitted so far, scored after each block. That is the protocol under which this network is known to compose, with E20-B's marks as absolute marks. It rehearses by construction. The learner's own pipeline (a replay store filled as the stream arrives) is the next step and is not run here.

## The update, exactly

One loop, used by both arms, for births and lives:

```
logits = forward(m, x, local=<True for arm S, False for arm B>)     # credit through the whole item in both arms
loss   = cross_entropy(logits[answer_mask], y[answer_mask])
opt.zero_grad(); loss.backward(); opt.step()                          # the same AdamW in both arms
```

- `forward` is `arch26`'s written-out network. With `local=True` every linear map, the recurrent one included, sends its error back through `exp3_state.SignLinear`; with `local=False` through the weights themselves.
- **Do not call `E.update(m, opt, items, "T")`.** In `exp11_cell.py` that path runs the forward with credit cut at every step and then calls the cell's trace routine, which is written for the bank and is not defined for a GRU. `exp20_reference.py` calls `E.update(..., "D")`; replace that call in both arms with the loop above so that the two arms share every line but the flag.
- No trace routine, no scan, no local head, no store.

## Seeds and runs

Seeds 73129, 93683, 116089. Two arms. Six births and six lives.

## Stages

**Stage 1: implementation and qualification. Nothing at full dose.**

New files only: `arch26.py` (the written-out network with both paths) and `exp26_candidate.py` (birth and life for the two arms), importing from `arch20.py`, `exp20_reference.py` and `exp16_integrated.py` without editing them. Returned at the end of stage 1, in `exp45/e26_qual/E26_CHECK.json` and a record:

1. *The step.* On a fresh network and on E20-B's saved G1 files of all three seeds: the written-out forward equals `arch20`'s library forward in every logit (to 1e-5), and with the exact path every gradient equals the library's (to 1e-4 relative). A version with two gates' recurrent weights swapped must fail. The coordinator's record for a fresh network is `exp45/E25_FIXTURES_<seed>.json`.
2. *The two paths.* Same forward values under both. Under the sweep the top layer's own gradient equals the exact one and the recurrent matrix's does not. Every parameter and every active embedding row receives a finite, non-zero gradient under both (the start-symbol row exempt).
2a. *One real step in each arm.* From one saved state and one batch, take one step of the production loop in each arm. The weights afterwards equal, tensor for tensor, those obtained by giving the same AdamW the gradient computed independently (`e25_alignment.grads` with the same flag). The two arms' weights after the step differ from each other below the top layer. A loop that calls `E.update(..., "T")` must fail this check: include it as the fixture.
2b. *The scorer.* `arch26.forward` has the harness's signature, `silence` included, and is what `exp16_integrated.score` runs. Shown by: the three stub fixtures of E20-B's stage 1 (whole answer; right ending after a wrong intermediate; no end symbol); with `silence=True` the readout sees a zero state, as in `arch20`; and E20-B's saved end-of-life G1 file of each seed, loaded into the written-out network and scored by the production scorer, gives the counts in its own receipt on the composition panels and on the silenced panels. If a count differs, report which and by how much; do not adjust.
3. *The data.* For one seed at the reduced dose, the two arms are fed the same batches in the same order: a digest of every batch, birth and life, is equal between the arms.
4. *The optimizer.* Both arms record AdamW with decay 0.25 on everything except the embedding table, which has none. Read from the optimizer's groups, not from a constant.
5. *Causality, rows, reset,* as in E20-B's stage 1 (amendment 11 of that brief), for the written-out network.
6. *Binding.* A life refuses a birth of the other arm, of another seed, with a changed weights file, or made by other sources. No existing file is overwritten.
7. *Reduced-dose runs* for seed 73129, both arms: birth at 2,048 updates, life at 6 × 256. Both complete with completion records. Report update CPU-seconds per update and scoring CPU-seconds per checkpoint for each arm, and a forecast for stages 2 and 3 from them.

Then stop.

**Stage 2: the six births at full dose, on the coordinator's release.** Then stop and report the birth curves. No life starts.

**Stage 3: the six lives, on the coordinator's release.** All six run, whatever the births scored. The stop before stage 3 is for checking that each birth is technically valid: a completion record, a weights file matching its hash, finite losses throughout. A birth that is not valid is reported; its life is then not run and is counted as unmeasured, and the three-seed denominator stays.

## Endpoints

- **Birth:** hard mean (add89 and sub89 at k = 12, whole answers), at the six checkpoints.
- **Life, at the end:** new instances of taught operator pairs (`compose.within`, 600 items) and the untaught plus-minus pairs (`+-` and `-+`, 1,200 items), whole answers with the end symbol, scored by `exp16_integrated.score`. Each of the four untaught pairs separately at every checkpoint. The taught-instances panel as in E20-B.
- **Cost:** update and scoring CPU-seconds for each run.

## Predictions (copy unchanged into `E26_PREDICTIONS.md` before any full-dose run). "Holds" means in at least two of three seeds

- **P26.1** A sweep-born GRU is born at a hard mean of at least 70. Confidence about 50%.
- **P26.2** It is born at 0.85 or more of its backprop partner's hard mean. Confidence about 40%.
- **P26.3** Arm S ends its life at or above 85 on new instances of taught pairs. Confidence about 25%.
- **P26.4** Arm S ends its life at or above 40 on the untaught plus-minus pairs. Confidence about 25%.
- **P26.5** Arm B meets both marks (85 and 40) in the same two seeds. Confidence about 80%.

The marks 85 and 40 are E20-B's, fixed before E20-B ran.

## Readings, fixed now

- Two things are read, and kept apart. *Feasibility:* does arm S meet the absolute marks. *The gap:* S minus B, seed by seed.
- **A candidate pair exists** if arm S meets both marks in the same two or more seeds. Then the sweep and the GRU were born and lived together and compose from memory on untaught pairs, under answer targets and a joint stream. That is what the close may name, with those two conditions attached.
- **P26.5 misses:** the reference of this job did not reach the marks. Arm S's own result against the marks is still reported, and so are the paired differences, beside the fact that the reference fell short. E20-B's G1 differs from arm B in decay, in the recurrent layer's code and in host, so it cannot say which of the three is responsible; a backprop arm at 0.30 on this host would, and is then the follow-up.
- **P26.1 holds and P26.3 or P26.4 misses:** the sweep learns single operations in a GRU and does not carry them to composition. The scores are reported beside arm B's, and the gap is located by checkpoint and by operator pair.
- **P26.1 misses:** report the level. A birth under 70 may still have learned a good deal, and it still gets its life. Only if the three births are near the floor is it said that the sweep does not bring a GRU to birth at this dose, although its update is well aligned on backprop-trained weights. Then the remedies of E25's predictions file are considered in their order, with learned feedback weights now permitted (Mr. Xu's ruling of 11 November). The alignment measurement is repeated on the sweep-born file to see where the sweep took the network.
- Under every outcome the paired difference S minus B is reported for each seed at birth and at the end of the life.

## What this cannot say

- Answer targets only. The every-byte question is E24's and is not asked here.
- The joint stream rehearses everything. Nothing here is about retention or about a replay store.
- The registered decoder keeps the prompt and reruns the network for each output byte.
- Three seeds, one size, one host class. A development result; no claim of reuse or promotion.

## Running it

- One host for both arms of a seed, at most three processes on four cores, one thread each.
- Leave `E16_RESUME` unset: resume is not qualified, and this runner does not use the harness's life loop in any case. An interrupted run is kept as written, stopped and reported; it is restarted from the beginning only on the coordinator's word.
- Every receipt records its command, environment, host (machine, platform, Python, torch, threads) and the hashes of the source files it ran.
- Work in progress on your own branch. Nothing goes to the default branch before a stage is complete.
- Forecast stages 2 and 3 from stage 1's own measurements, update CPU and scoring CPU separately; the written-out step's cost falls on both, the sweep's extra cost on updates only. A first estimate until then: E20-B's G1 took about 2,900 CPU-seconds for a birth and 4,600 for a life with the library layer. On the owner's machine, on one birth batch, an update through the written-out step costs 1.4 times the library layer's with the exact path and 1.6 times with the sweep. That puts a birth near 4,100 to 4,700 CPU-seconds and a life near 6,500 to 7,400: about 68,000 for the twelve runs, six to seven hours on three cores.

## Report

`E26_PREDICTIONS.md`, `E26_REPORT.md`, receipts, weights, logs, and one paragraph as `E26_REVIEW_SECTION.md`. State what was not run.

## Questions for the audit (answered by the audit of 11 November: keep decay 0.25 in both arms; the joint stream first; no floor, all lives run; one scale per stacked matrix)

1. Is decay 0.25 in both arms the right matched pair, or should arm B also be run at 0.30 so that it can be set beside E20-B's G1 on the same host?
2. Is the joint stream the right first pipeline, with the learner's own pipeline as a later step, or does the close need the replay pipeline from the start?
3. The release of an arm S life is left to the coordinator's judgment of its birth. Should a floor be fixed now instead (for example a hard mean of at least 20)?
4. Should the sweep use one scale per whole stacked matrix, as written, or one per gate?
