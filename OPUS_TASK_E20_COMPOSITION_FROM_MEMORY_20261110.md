# Two cloud jobs for Claude Opus 5.5 workers: E20, composition from memory (10 November 2026, 16:00 EST)

> **Status under the process rules of 10 November (review 9.82).** Job A version 2 and Job B have not been audited by Codex. Send this brief to Codex first; a worker starts only after that audit is answered. **Commits:** one commit at completion on the default branch; partial receipts stay on your session's own branch or uncommitted, never on the default branch. This replaces any sentence below about pushing partial receipts.

> **Sizing (Mr. Xu, 10 November, 21:30 EST).** Never more processes than cores. Job A stage 2: at most three lives at a time on a four-core container, one arm per instance if three instances are available. Job B: one network per instance (G0, G1, G2, G3): its three births together, then its three lives together, about three hours; the acceptance test runs first on whichever instance takes G0.

**Repository:** `ChangweiZhou/digitalbaby`, branch `codex/minifly-consolidated-20261023`. Fetch and check out; confirm this file is present and **start from the branch head**: the harness was corrected after the 10 November source audit and every receipt now carries a credit census, strict-panel scores, exposure counts and hashes. **Folder:** `MOD97_LOCAL_RULE_20261106/`. Python 3, PyTorch on CPU, NumPy. One job per worker; say which in your first message.

**Branch rule.** Commit to `codex/minifly-consolidated-20261023` if your session can push there (pull with `--rebase` first, never force, at most one push an hour while runs are in flight); otherwise to the branch your session is given, named in your last message. No other branch.

**Rules, both jobs.** Launch runs detached and poll logs with short commands. Dry-run each command shape first (`E11_STAGE_UPDATES=4`, and `E11_DEV_UPDATES=512` for births) and archive the dry files under `exp45/archive_e20_dry/cloud_<date>/`. Commit your predictions file before any run and do not change it. Add files; edit no existing file. No reruns of a finished run, no tuning, no change of seeds: a miss is a result. Report every seed; forecast from your dry runs, then report forecast against measured; record machine, core count, torch version, commit, each process's environment, and CPU-seconds per run. Report the **whole worked answer** (`exact`) and the final answer (`final`) side by side, in absolute numbers, and say which panel each number is from.

## Why: the decision behind these jobs

The coordinating session's reading of the record, 10 November. All composition on file comes from the window learner, and a composition prompt (`cAA+BB-CC=`) is exactly ten bytes, the width of that learner's window. At `=` the whole prompt is in view; when the final answer is written, the second operator, the third operand and the intermediate result are in view. That learner has never had to carry an operand in memory to compose. The windowless two-bank learner must carry everything, and under backprop it ends at 49–59% whole answers on **new instances of taught operator pairs** and 1–7% on the untaught plus-minus pairs (the window reference: 92–93% and 85–89%). So the open problem is not a learning rule. It is that no learner on file composes from memory, and work on credit rules and replay shares for these learners cannot change that. E20 asks two questions, one per job, before any more rule variants are run.

Standing notes. Development results on sealed panels, held out by ordered pair (evidence level E2; strict-panel scores are in every receipt). Backprop is used here as a reference to find out what can be computed, not as a candidate rule. No learning-core improvement is claimed by either job.

---

## Job A (version 2, 10 November 17:00 EST): does the window learner compose when an operand must be remembered?

**Version 1 of this job is withdrawn; no worker had started it.** Codex's second audit of 10 November found, and the coordinating session confirmed by running it, that version 1 could not answer its question: its panel pooled items in which the first operand was still in view with items in which it was not, it scored only the final suffix, it compared arms with and without the score-time store, its single prediction tied existence to a size of penalty, and the harness's qualifier failed on the new prompts. Version 2 replaces the panel and the predictions.

**The assay (in the harness at the commit this version ships with).** `E16_COMPGAP=6` now does two things. In teaching, each composition gets 0 to 6 letters twice; with probability one half they go inside the prompt (`c12+PRRPM34-EHUEQ56=`), otherwise the same number of letters lead it (`SELTPc12+34-56=`). Same lengths either way; inside, the first operand usually leaves a ten-byte window, leading, the whole prompt stays in view. The registered composition panels are unchanged. Memory use is measured on separate panels, all scored on **whole worked answers with no store, for every arm alike**:
- `memory.twins`: for each held operator pair, 150 twins. The two prompts of a twin have the same operators, the same second and third operands and the same letters in the same places, and differ only in the first operand, which is outside the last ten bytes at `=`. Their intermediate and final results both differ. A predictor that sees only the last ten bytes cannot get both; the harness asserts that such a predictor scores zero. The endpoint is **both members correct**.
- `memory.lead`: the same items with the same letters moved in front: same length, every operand in view. This is the control for length.
- `memory.within_hidden` and `memory.taught_hidden`: with the first operand hidden, new operand triples of taught operator pairs, and taught triples themselves. Three different things are kept apart: taught instances, new instances of taught operator pairs, and untaught operator pairs.
- `memory_silenced.twins`: the same twins with the recurrent state zeroed. Read it as sensitivity to an out-of-distribution ablation, not as a clean test.

**Admission.** Two commands must exit 0 on your machine before any life. `python3 e20_assay_check.py` runs the memory scorer against two stand-in responders on all three seeds: one that answers every prompt correctly must score every twin, and one that answers from the last ten bytes alone must score no twin while still scoring the whole leading control (the coordinator's record is `exp45/E20_ASSAY_CHECK.json`; note that such a responder can still get one member of each twin, half the items, which is why both-correct is the endpoint and item accuracy is only descriptive). And `E16_COMPGAP=6 python3 exp16_integrated.py qualify` must exit 0 (it round-trips every memory item against its structured operands, checks disjointness from every taught triple, checks that twins hide and the control shows the first operand, and records the panel hash; the coordinator's record is `exp45/QUALIFY_E20_MEMORY_PANELS.json`). Every receipt records its memory-panel hash, its birth's admission and a completion record naming the hash of its weights; a life refuses a birth whose hash is not the expected one.

**Runs, in two stages, seeds 73129, 93683, 116089, all with `E16_COMPGAP=6`** (receipts tagged `_cm6`).

```
# stage 1: the assay's own qualification on a reference (three lives)
E16_COMPGAP=6 python3 exp16_integrated.py D <seed>                                         # window reference: backprop, joint stream
# stage 2, only if P20.3 holds (nine lives)
E16_COMPGAP=6 E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py T <seed>            # window learner: sign rule, raw reservoir
E16_COMPGAP=6 E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py M <seed>            # backprop on the learner's pipeline
E16_COMPGAP=6 E16_ARCH=bank2 E16_BOOKDECAY=off python3 exp16_integrated.py D <seed>        # windowless two-bank reference
```

If P20.3 misses, stop after stage 1 and report: the reference cannot do the in-view control at these lengths, so the assay cannot speak about memory, and that has to be diagnosed first.

**Predictions (copy into `E20A_PREDICTIONS.md`).** End of life; plus-minus twins means the `+-` and `-+` twins pooled; item accuracy is whole answers over single items.
- **P20.1 (existence)** The window reference gets both members right on at least 20% of plus-minus twins in at least two seeds.
- **P20.2 (what hiding costs)** The window reference's item accuracy on plus-minus twins is at least 15 points below its accuracy on the same items with leading letters, in all three seeds.
- **P20.3 (the control works)** The window reference reaches at least 70% item accuracy on the leading control for the plus-minus pairs in at least two seeds.
- **P20.4** The windowless reference gets both members right on at most 5% of plus-minus twins in all three seeds.
- **P20.5** The sign-rule learner's item accuracy on plus-minus twins is at or below 0.60 of the window reference's in at least two seeds.
- **P20.6** Backprop on the learner's pipeline is at or above 0.85 of the window reference's in at least two seeds.

Existence (P20.1) and cost (P20.2) are separate verdicts. The window reference is a joint-stream ceiling with its own decay and supervision; the one-factor comparison for the rule is the learner against backprop on its pipeline, both without the store.

Reading, written before the result. P20.1 holding with P20.3 says a learner on file can compose with an operand in memory, at the cost P20.2 measures, and the windowless failure belongs to the two-bank design. P20.1 missing with P20.3 holding says the window learner composes only in view. P20.4 missing would mean the two-bank network can use a hidden operand after all, and version 1's reading of it was wrong.

**Report.** Per arm and seed at every checkpoint: twins both-correct and item accuracy, the leading control, `within_hidden`, `taught_hidden`, silenced twins, each by operator pair; the registered panels; family means; the six verdicts; update CPU and scoring CPU separately (both are in the receipt). `E20A_PREDICTIONS.md`, `E20A_REPORT.md`, receipts and weights, logs, one paragraph as `E20A_REVIEW_SECTION_9_77.md`. The coordinator's dry runs of the four command shapes (seed 73129, four updates per block, seven stages each, completion records present) are in `exp45/archive_e20_dry_v2/`; at that dose every memory score is zero, as expected. Cost: about 5,000 CPU-s of updates per life on the cloud containers used so far, plus scoring, which the dry runs put at about 60–85 CPU-s per checkpoint on the owner's machine under load; stage 1 about 1.5 hours on four cores, stage 2 about 4.

---

## Job B: can any windowless network of this size compose from memory under backprop?

A positive reference. If an ordinary recurrent network learns new instances of the taught operator pairs and transfers to the untaught pairs from its state alone, the task is solvable from memory and the two-bank design lacks something specific. If none does, the teaching protocol does not support that transfer at this scale, and a bounded window stays part of the design.

**New files only** (`arch20.py`, `exp20_reference.py`), following `arch17.py`'s `install` pattern so that `E.respond`, `E.score` and `exp16_integrated.panels` / `strict_panels` / `score` are used unchanged. Three networks, each reading one byte per step with no window, each within 5% of 1,125,265 parameters (report the count):
- **G1, GRU:** byte embedding, one GRU layer, the two-layer readout of `arch17`.
- **G2, LSTM:** the same with an LSTM layer.
- **G3, two banks with a loop:** `arch17`'s `bank2` with one added matrix, so that the first bank's projection also reads the second bank's previous state (`p_t = tanh(lp z1_t + lr h2_{t-1})`). One difference from `bank2`.
- **G0, the matched reference (added 10 November 17:00 EST after the second audit):** `arch17`'s `bank2` itself, run through your loop on your machine at the same dose. G3 is compared with G0, not with the receipt on file, so that the loop is the only difference. A GRU or LSTM that succeeds shows that some network of this size can do the task; it does not say what `bank2` lacks.
- **Labels.** `within` is *new operand triples of taught operator pairs*, not taught compositions. Add a third panel of taught triples themselves (the first 50 of `triples(pair, seed, 2000)` for each taught pair) and report the three separately: taught instances, new instances of taught pairs, untaught pairs.

**Protocol, identical to the references on file.** Birth: `develop17`'s protocol (the four mod-89 operations, 24,576 updates, batches of 64, panels at k = 0, 4, 8, 12, 20). Life: `life`'s arm D, staged (six blocks of 6,144 updates from the joint stream of the families admitted so far, exactly as the `arm == "D"` branch builds it), scored after each block with `panels` and `strict_panels`. AdamW, learning rate 0.001, decay 0.3, betas (0.9, 0.98), embedding tables without decay. Seeds 73129, 93683, 116089, four networks (G0 to G3). No distractor letters in compositions (the standard prompts), so that the numbers can be set beside `e17_life_D_bank2_nbd_<seed>.json` on file (owner's machine; a historical comparison, G0 is the matched one): new instances of taught pairs 48.7 / 58.5 / 51.7 whole answer, plus-minus pairs 0.8 / 6.8 / 2.9, comparison pairs 22.2 / 23.1 / 25.4, mod-97 92.3 / 93.7 / 93.3.

**Acceptance before any run:** a test that your life loop, run with `arch17`'s `bank2` network, reproduces the first two checkpoints of `e17_life_D_bank2_nbd_73129.json` in a dry setting of your choice against `exp16_integrated.py D` with the same setting (same scores, same losses). Commit it as `exp45/E20B_CHECK.json`. If it does not reproduce, stop and report.

**Predictions (copy into `E20B_PREDICTIONS.md`).** Whole answers, end of life.
- **P20.7** G1 and G2 are born at a hard mean (add89, sub89 at k = 12) of at least 85 in at least two seeds each.
- **P20.8** At least one of G1 and G2 ends at or above 85 on new instances of taught operator pairs in at least two seeds.
- **P20.9** At least one of G1, G2 and G3 ends at or above 40 on the plus-minus pairs in at least two seeds. Held at about 50%.
- **P20.10** G3 ends at least 15 points above G0 on new instances of taught operator pairs in at least two seeds. Held at about 50%.

Reading, written before the result. P20.8 and P20.9 holding for a gated recurrent network say composition from memory is learnable here by a network of this size; they do not say why `bank2` fails. P20.10 says whether one loop between the banks, the only difference from G0, helps. All three missing P20.9 says transfer to untaught operator pairs from memory is not supported by twelve taught pairs at this size, and the next question is the teaching, not the network.

**Report.** Per network and seed: the birth curve; at every checkpoint taught instances, new instances of taught pairs, the four held pairs (whole and final answer), family means, strict-panel means; parameter counts; the acceptance test; four verdicts; CPU-seconds. `E20B_PREDICTIONS.md`, `E20B_REPORT.md`, births, receipts and weights, logs, one paragraph as `E20B_REVIEW_SECTION_9_78.md`. Cost: twelve births and twelve lives, about 100,000 CPU-s, about 7 hours on four cores.

---

## Amendments to Job B after Codex's audit of the dispatch (10 November 23:00 EST; `PROJECT_DOCUMENTS/MINIFLY_E22_DISPATCH_HOSTILE_AUDIT_20261110.md`). These govern.

1. **The premise is updated.** E20-A stage 1 showed that the window reference does compose with an operand in memory, partly (review 9.77). Job B asks whether a network with no window can.
2. **Acceptance covers the whole life.** Your runner with `arch17`'s `bank2` must reproduce `exp16_integrated.py D <seed>` run on the same host at the same reduced dose, in every score and loss at all seven stages, not the first two. Do not compare a reduced-dose run with the full-dose receipt on file.
3. **Each new network is checked before it is trained:** tensor shapes; no byte sees a later byte; state is reset between items and nothing leaks between rows of a batch; the start and end symbols are handled as in `arch17`; every parameter receives a gradient on a first batch; embeddings are in the group without decay.
4. **G3 is compared with G0 on one host.** The instance that runs G0 also runs G3, three processes at a time. G3 starts from the same initial values as G0 for every shared parameter, with the added matrix initialised last. G1 and G2 may run elsewhere and are feasibility packages: their success shows that some network of this size can do the task and says nothing about why the two-bank network does not.
5. **E20's own file names** (`e20_dev_<net>_<seed>`, `e20_life_<net>_<seed>`). Do not call `develop17` or `life` in a way that can reuse a cached birth or write over a receipt on file; G0's birth is made fresh under its E20 name.
6. **Reading narrowed.** If all three new networks miss P20.9, that bounds these three packages at this dose. It does not show that the teaching is the cause.
7. **If a run is interrupted,** keep what it wrote, stop that cell and report it. No restart without authorization.

---

## Second amendments to Job B, after Codex's audit of the amendments (10 November 2026, 21:45 EST by the clock; `PROJECT_DOCUMENTS/MINIFLY_E20B_AMENDMENTS_AUDIT_20261110.md`). These govern, together with 1 to 7 above.

Every finding of that audit about the inherited source was run by the coordinator and holds (`MOD97_LOCAL_RULE_20261106/e20b_audit_checks.py`, output in `exp45/E20B_AUDIT_CHECKS.json`). You may reuse that script's fixtures.

8. **Two stages, and a stop between them.**
   - Stage 1 is the implementation and its qualification at a reduced dose. Nothing at full dose is run in stage 1.
   - At the end of stage 1 you return: the commit of your runner, `exp45/e20b_qual/E20B_CHECK.json`, `E20B_PREDICTIONS.md` (the four predictions and the selection rule of amendment 13, copied unchanged), and the command and environment of every qualification run.
   - Stage 2, the twelve births and twelve lives, starts only on the coordinator's go, after Codex has read the stage 1 record.

9. **The comparator, exactly.** A bare `exp16_integrated.py D <seed>` is the window network with embedding decay on. The reference for parity is, in an environment with no other `E16_` or `E11_` variable set:
   - birth: `E16_ARCH=bank2 E16_BOOKDECAY=off E16_COMPGAP=0 E11_DEV_UPDATES=<d> E11_STAGE_UPDATES=<s> OMP_NUM_THREADS=1 python3 exp16_integrated.py dev 73129`
   - life: the same variables plus `E16_OUTTAG=_e20bref`, then `python3 exp16_integrated.py D 73129`.
   - `<d>` must be a multiple of 512: the birth loop works in whole blocks of 512 updates and stops with an error below one block (found in the coordinator's dry run). Choose `<d>` and `<s>` so that the reduced-dose file names the reference writes exist nowhere under `exp45/` (check with `find` first and record the check). The coordinator's own dry run used 512 and 4 and its files are in `exp45/archive_e20b_ref_dry/`; choose a larger dose than that, so that some scores are above zero. When the reference has finished, move its birth, receipt and log into `exp45/e20b_qual/reference/`.
   - Parity is shown in two parts, on one host, one thread:
     a. **Birth.** Your birth loop with `arch17`'s `bank2`, same seed and dose, gives the same curve and the same weights hash as the reference birth.
     b. **Life.** Your life loop, started from the reference birth file itself (loaded by path, its hash checked), gives at all seven stages the same counts on every panel the two runs share, the same recorded losses, the same panel hashes, the same number of optimizer steps, and at the end the same weights hash.
   - If the weights hashes differ, report the largest absolute difference per tensor and stop. Do not adjust a tolerance to pass.
   - The full-dose receipt on file (`e17_life_D_bank2_nbd_<seed>.json`) is a historical comparison only.

10. **Scoring.**
    - Compositions are scored with `exp16_integrated.score` (six bytes). The birth helper `E.score` generates at most four bytes and cannot score a worked answer; use it for births only.
    - Three fixtures in `E20B_CHECK.json`, with the responder replaced by a stub: a fully correct answer scores whole and final; a wrong intermediate with the right ending scores final only; an answer without the end symbol scores neither.
    - The panel of taught instances is kept in a dictionary of its own. Never add it to the `comp` dictionary: `shortcut_ceilings` stops on any key other than an operator pair or `within`. Record its hash, and check its exposure separately: every item of it was in the training stream's taught set, and no item of the `within` or held-pair panels was.
    - Whole answers including the end symbol are the primary endpoint. Final-answer scores are diagnostic.

11. **Architecture dispatch and state.**
    - The parent selects its architecture when it is imported and knows only its own five names. Import it with `E16_ARCH=bank2 E16_BOOKDECAY=off`, so that the windowless settings hold (the k = 40 panels, no store), then install your network. Record in every receipt the class actually installed and its parameter count.
    - `adamw16` finds the embedding table by the name prefix `book.` only. A table under any other name gets decay 0.3 without a word. Name it `book`, and in any case record the optimizer's groups (parameter names and decay per group) in the receipt.
    - Checks per network, on a batch of mixed lengths, before any training: changing later bytes leaves earlier logits unchanged; changing or reordering other rows leaves a row's outputs unchanged; every recurrent state is reset between items, the LSTM's cell state included; active embedding rows and the recurrent weights receive finite, non-zero gradients. The start-symbol row is exempt: it is a padding row and never learns, as in `arch17`.
    - G3: the added connection is a 64-by-512 matrix with no bias, as the specification above says ("one added matrix"). With it set to zero the outputs equal G0's exactly; with it enabled the first step of an item does not depend on it, and a later step reads the second bank's state of the step before.

12. **Each life is bound to its own birth.**
    - Before loading, a life verifies from the birth's complete receipt: network, seed, birth dose, optimizer settings, the hashes of the source files used, and the weights hash. The seed in the receipt, the seed in the file name and the life's seed must agree. There is no legacy admission in your runner.
    - Fixtures in `E20B_CHECK.json`: a birth presented under another seed is refused; a changed weights file is refused; an output path that already exists is refused.
    - The birth used for parity (9b) and G0's own birth for the experiment are different files with different purposes. G0's experimental birth is made fresh under its E20 name at full dose.
    - For your information: the inherited `admit_birth` had this hole (a checkpoint copied with its receipt under another seed's name was admitted). It now compares the receipt's seed with the file name.

13. **How the predictions feed the decision. Fixed now, before any result.** P20.7 to P20.10 are judged exactly as registered. They do not by themselves choose a network: P20.8 and P20.9 can hold for different networks, or for one network in different seeds. The selection rule is separate:
    - A network **qualifies** if it ends at or above 85 on new instances of taught operator pairs and at or above 40 on the plus-minus pairs, whole answers, in the same two or more seeds.
    - If G3 qualifies, G3 is the windowless candidate, whether or not a gated network also qualifies. Reason: it is one matrix away from the network on which the working rule already trains and is born (review 9.70, 9.85).
    - Otherwise, if exactly one of G1 and G2 qualifies, that one is the candidate. If both do, the one with the higher three-seed mean on the plus-minus pairs; on an exact tie, the GRU.
    - If none qualifies, there is no windowless candidate from this job, whatever the four verdicts are.
    - Report each of the four held operator pairs separately at every checkpoint, so that the pooled plus-minus figure can be read.

14. **Sentences above that no longer stand.** Amendment 6 replaces two earlier sentences of this brief: "If none does, the teaching protocol does not support that transfer at this scale, and a bounded window stays part of the design" (opening of Job B), and "the next question is the teaching, not the network" (the reading after the predictions). A miss by all three bounds these three packages at this dose and says nothing about the cause.

15. **What the results can and cannot say.** State these in the report.
    - P20.10 measures the added connection as a package: 32,768 weights (a 64-by-512 matrix with no bias) and the work they cost. It does not show that the placement of the feedback matters; that would need a control with as many extra weights elsewhere. Record G3's added parameters and its CPU beside G0's.
    - "Windowless" describes the network's input. The registered decoder keeps the prompt and the bytes produced so far and reruns the network for each output byte. Nothing here measures single-pass inference or its memory cost.
    - This is a development feasibility result at evidence level E2. It supports no claim of reuse or promotion.

16. **Running stage 2, when it is released.**
    - G0 and G3 on one host in separate waves, at most three processes at a time on four cores.
    - Every receipt records its command and environment.
    - Containers have restarted twice this week. Push work in progress to your own branch as often as you like. Nothing partial goes to the default branch; the default branch receives one commit when all twenty-four completion records and their weights hashes have been verified.
    - Amendment 7 stands: an interrupted cell is kept as written, stopped and reported.
