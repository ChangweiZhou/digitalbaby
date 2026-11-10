# Two cloud jobs for Claude Opus 5.5 workers: E20, composition from memory (10 November 2026, 16:00 EST)

**Repository:** `ChangweiZhou/digitalbaby`, branch `codex/minifly-consolidated-20261023`. Fetch and check out; confirm this file is present and **start from the branch head**: the harness was corrected after the 10 November source audit and every receipt now carries a credit census, strict-panel scores, exposure counts and hashes. **Folder:** `MOD97_LOCAL_RULE_20261106/`. Python 3, PyTorch on CPU, NumPy. One job per worker; say which in your first message.

**Branch rule.** Commit to `codex/minifly-consolidated-20261023` if your session can push there (pull with `--rebase` first, never force, at most one push an hour while runs are in flight); otherwise to the branch your session is given, named in your last message. No other branch.

**Rules, both jobs.** Launch runs detached and poll logs with short commands. Dry-run each command shape first (`E11_STAGE_UPDATES=4`, and `E11_DEV_UPDATES=512` for births) and archive the dry files under `exp45/archive_e20_dry/cloud_<date>/`. Commit your predictions file before any run and do not change it. Add files; edit no existing file. No reruns of a finished run, no tuning, no change of seeds: a miss is a result. Report every seed; forecast from your dry runs, then report forecast against measured; record machine, core count, torch version, commit, each process's environment, and CPU-seconds per run. Report the **whole worked answer** (`exact`) and the final answer (`final`) side by side, in absolute numbers, and say which panel each number is from.

## Why: the decision behind these jobs

The coordinating session's reading of the record, 10 November. All composition on file comes from the window learner, and a composition prompt (`cAA+BB-CC=`) is exactly ten bytes, the width of that learner's window. At `=` the whole prompt is in view; when the final answer is written, the second operator, the third operand and the intermediate result are in view. That learner has never had to carry an operand in memory to compose. The windowless two-bank learner must carry everything, and under backprop it ends at 49–59% whole answers on **taught** compositions and 1–7% on the untaught plus-minus pairs (the window reference: 92–93% and 85–89%). So the open problem is not a learning rule. It is that no learner on file composes from memory, and work on credit rules and replay shares for these learners cannot change that. E20 asks two questions, one per job, before any more rule variants are run.

Standing notes. Development results on sealed panels, held out by ordered pair (evidence level E2; strict-panel scores are in every receipt). Backprop is used here as a reference to find out what can be computed, not as a candidate rule. No learning-core improvement is claimed by either job.

---

## Job A: does the window learner compose when an operand must be remembered?

**Harness option (in the commit this brief ships with):** `E16_COMPGAP=<k>` puts 0 to k distractor letters after each operator of every composition prompt, in teaching and in the panels (`c46+SE10-75=`); receipts are tagged `_cg<k>`, and the same held triples are also scored without letters as `compose_nogap`. With k = 6 the first operand and operator are out of view at `=` in most items, and often the second operand too.

**Runs: twelve staged lives, seeds 73129, 93683, 116089, all with `E16_COMPGAP=6`.** Births are on file.

```
E16_COMPGAP=6 python3 exp16_integrated.py D <seed>                                         # window reference: backprop, joint stream
E16_COMPGAP=6 E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py T <seed>            # window learner: sign rule, raw reservoir
E16_COMPGAP=6 E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py M <seed>            # backprop on the learner's pipeline
E16_COMPGAP=6 E16_ARCH=bank2 E16_BOOKDECAY=off python3 exp16_integrated.py D <seed>        # windowless two-bank reference
```

Receipts `exp45/e16_life_{D,T_raw65536,M_raw65536}_cg6_<seed>.json` and `exp45/e17_life_D_bank2_nbd_cg6_<seed>.json` with their `.pt` (`git add -f`). Record that none exists before launch. Reference without letters, on file: the windowless reference `e17_life_D_bank2_nbd_<seed>.json`; for the window learner use `compose_nogap` from your own receipts (same weights, letter-free panel).

**Predictions (copy into `E20A_PREDICTIONS.md`).** "Hard pairs" is the mean of `+-` and `-+`, final answer, end of life, on the panel with letters unless stated.
- **P20.1** The window reference reaches at least 50 on the hard pairs in at least two of the three seeds, and is at least 15 points below its own letter-free score in all three. (It holds one operand across twelve letters at 93–95% in single operations; here it must hold two.)
- **P20.2** The windowless reference stays at or under 15 on the hard pairs in all three seeds.
- **P20.3** The sign-rule learner ends at or below 0.60 of its reference on the hard pairs in at least two seeds.
- **P20.4** Backprop on the learner's pipeline ends at or above 0.85 of the reference on the hard pairs in at least two seeds.

Reading, written before the result. P20.1 holding says the window learner can compose with an operand in memory, the windowless failure belongs to the two-bank design, and Job B says which ingredient. P20.1 missing with the reference near the windowless floor says no learner on file composes from memory at all and every composition number on record is composition in view. P20.3 and P20.4 say how much of whatever the reference achieves the rule keeps.

**Report.** Per arm and seed at every checkpoint: taught compositions (`within`), the four held pairs, with letters and without, whole answer and final answer; the mod-97 and mod-73 family means; the strict-panel means. Four verdicts. `E20A_PREDICTIONS.md`, `E20A_REPORT.md`, twelve receipts and weights, logs, one paragraph as `E20A_REVIEW_SECTION_9_77.md`. Cost: about 5,000 CPU-s per life on the cloud containers used so far, twelve lives about 4.5 hours on four cores.

---

## Job B: can any windowless network of this size compose from memory under backprop?

A positive reference. If an ordinary recurrent network learns the taught compositions and transfers to the untaught pairs from its state alone, the task is solvable from memory and the two-bank design lacks something specific. If none does, the teaching protocol does not support that transfer at this scale, and a bounded window stays part of the design.

**New files only** (`arch20.py`, `exp20_reference.py`), following `arch17.py`'s `install` pattern so that `E.respond`, `E.score` and `exp16_integrated.panels` / `strict_panels` / `score` are used unchanged. Three networks, each reading one byte per step with no window, each within 5% of 1,125,265 parameters (report the count):
- **G1, GRU:** byte embedding, one GRU layer, the two-layer readout of `arch17`.
- **G2, LSTM:** the same with an LSTM layer.
- **G3, two banks with a loop:** `arch17`'s `bank2` with one added matrix, so that the first bank's projection also reads the second bank's previous state (`p_t = tanh(lp z1_t + lr h2_{t-1})`). One difference from `bank2`.

**Protocol, identical to the references on file.** Birth: `develop17`'s protocol (the four mod-89 operations, 24,576 updates, batches of 64, panels at k = 0, 4, 8, 12, 20). Life: `life`'s arm D, staged (six blocks of 6,144 updates from the joint stream of the families admitted so far, exactly as the `arm == "D"` branch builds it), scored after each block with `panels` and `strict_panels`. AdamW, learning rate 0.001, decay 0.3, betas (0.9, 0.98), embedding tables without decay. Seeds 73129, 93683, 116089. No distractor letters in compositions (the standard prompts), so that the numbers compare with `e17_life_D_bank2_nbd_<seed>.json` on file: taught compositions 48.7 / 58.5 / 51.7 whole answer, plus-minus pairs 0.8 / 6.8 / 2.9, comparison pairs 22.2 / 23.1 / 25.4, mod-97 92.3 / 93.7 / 93.3.

**Acceptance before any run:** a test that your life loop, run with `arch17`'s `bank2` network, reproduces the first two checkpoints of `e17_life_D_bank2_nbd_73129.json` in a dry setting of your choice against `exp16_integrated.py D` with the same setting (same scores, same losses). Commit it as `exp45/E20B_CHECK.json`. If it does not reproduce, stop and report.

**Predictions (copy into `E20B_PREDICTIONS.md`).** Whole answers, end of life.
- **P20.5** G1 and G2 are born at a hard mean (add89, sub89 at k = 12) of at least 85 in at least two seeds each.
- **P20.6** At least one of G1 and G2 ends at or above 85 on taught compositions in at least two seeds.
- **P20.7** At least one of the three networks ends at or above 40 on the plus-minus pairs in at least two seeds. Held at about 50%.
- **P20.8** G3 ends at least 15 points above `bank2` on taught compositions in at least two seeds. Held at about 50%.

Reading, written before the result. P20.6 and P20.7 holding for a gated recurrent network say composition from memory is learnable here and what `bank2` lacks is learned recurrence; the project then has to choose between paying for credit through learned recurrence and keeping a bounded window. P20.8 says whether one loop between the banks is enough of it. All three missing P20.7 says transfer to untaught operator pairs from memory is not supported by twelve taught pairs at this size, and the next question is the teaching, not the network.

**Report.** Per network and seed: the birth curve; at every checkpoint taught compositions, the four held pairs (whole and final answer), family means, strict-panel means; parameter counts; the acceptance test; four verdicts; CPU-seconds. `E20B_PREDICTIONS.md`, `E20B_REPORT.md`, births, receipts and weights, logs, one paragraph as `E20B_REVIEW_SECTION_9_78.md`. Cost: nine births and nine lives, about 77,000 CPU-s, about 5.5 hours on four cores.
