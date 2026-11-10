# One cloud job for a Claude Opus 5.5 worker: E18b, every next byte as the target (10 November 2026, 18:15 EST)

**Repository:** `ChangweiZhou/digitalbaby`, branch `codex/minifly-consolidated-20261023`. Fetch, check out, **start from the branch head**, and confirm this file is present. **Folder:** `MOD97_LOCAL_RULE_20261106/`. Python 3, PyTorch on CPU, NumPy. A small job for one worker: six births. It has lower priority than E20; run it on an instance that would otherwise be idle.

**Branch rule.** Commit to `codex/minifly-consolidated-20261023` if your session can push there (pull with `--rebase` first, never force, at most one push an hour while runs are in flight); otherwise to the branch your session is given, named in your last message. No other branch.

## Why

Mr. Xu's original idea was that a learner in a stream receives delayed feedback at every step, in its purest form the next byte, and that this could serve as a target at every stage. E18 tested a target at every bank, but only on the **answer bytes**. Its cheapest arm, B3 (a head at each bank, no gradient between banks, weight signs, credit within the step, the exact trace on the write synapses only), learned nothing. Its saved weights show why: its operator, letter and task-code embeddings never moved, because with the loss on answer bytes alone those symbols are never the input at a supervised position.

Measured by the coordinating session before this brief, with no training (`exp45/E18B_ALIGNMENT.json`, and the census in the dry receipts):

| | Answer bytes (as E18) | Every next byte |
|---|---|---|
| Input symbols whose embedding receives any gradient under B3's rule | 11 of 45 | 45 of 45 |
| B3's rule against the exact gradient of its own objective, first bank's projection, at a trained point (three seeds) | 0.23 / 0.16 / 0.20 | 0.63 / 0.50 / 0.77 |
| The same, first layer | 0.69 / 0.65 / 0.69 | 0.83 / 0.72 / 0.75 |

So with every next byte as the target the cheap rule reaches all of its input code and points along the gradient of its own objective. What is not known is whether that objective, most of whose bytes are prompt bytes, teaches the arithmetic. That is the question.

## Runs: six births, seeds 73129, 93683, 116089

All with `E16_ARCH=bank2 E16_BOOKDECAY=off E16_HEADS=local E16_BIRTHRULE=T E16_LOSS=all`, 24,576 updates, the standard development stream. Scoring is unchanged: whole answers on the sealed development panels.

| Arm | Extra environment | Birth file `exp45/e17_dev_bank2_nbd<tag>_<seed>` | Its one-factor partner on file |
|---|---|---|---|
| B3-all | none | `_lh_bT_lossall` | B3, `_lh_bT`: hard mean 1.2 / 0.8 / 0.8, max/min mean 0.8 / 2.0 / 2.8 |
| B2-all | `E16_TRACEUP=all` | `_lh_bT_tall_lossall` | B2, `_lh_bT_tall`: hard mean 74.8 / 69.7 / 66.3 |

```
E16_ARCH=bank2 E16_BOOKDECAY=off E16_HEADS=local E16_BIRTHRULE=T E16_LOSS=all [E16_TRACEUP=all] python3 exp16_integrated.py dev <seed>
```

Each arm differs from its partner in **one thing, the loss mask**: same network, same heads, same rule, same decay (0.25), same seeds, same stream, same updates. The partners were run on a cloud container at an earlier commit; `e18_check.py` shows the gradients of that path unchanged since, and the coordinator's dry birth of B3 reproduces its archived dry birth score for score. B2-all is the control: it says what every-byte targets do to a package that already works, so that a B3-all result is not read without it.

**Admission, before any birth.** `python3 e18_check.py` exits 0. One dry birth of each arm (`E11_DEV_UPDATES=512`, archived under `exp45/archive_e18b_dry/cloud_<date>/`) whose receipt shows `loss: all`, a completion record, and a credit census in which every symbol role except the padding rows has all of its rows reached (the coordinator's: operators 4/4, digits 10/10, `=` 1/1, letters 26/26, task codes 4/4). If the census shows a role unreached, stop and report. Record that none of the six birth files exists. Commit `E18B_PREDICTIONS.md` before the first birth. Launch detached; poll logs with short commands. No reruns, no tuning, no change of seeds. The logged loss is now a mean over every byte and is not comparable with the answer-loss logs; compare scores, not losses.

## Predictions (copy into `E18B_PREDICTIONS.md`)

"Hard mean" is the mean of add89 and sub89 at twelve distractors, whole answers, final checkpoint, main head. "Max/min mean" likewise for max89 and min89. "Embedding movement" is `book_shift_by_role` in the receipt's last checkpoint.

- **P18b.1 (the input code is learned)** In B3-all the operator and distractor-letter embeddings have each moved at least 0.2 of a row norm in all three seeds. (B3: 0.00.)
- **P18b.2 (the package now learns something)** B3-all's max/min mean is at least 30 in at least two seeds. Held at about 55%.
- **P18b.3 (it stays far from the long trace)** B3-all's hard mean is at least 20 points below B2-all's in all three seeds: the layers above the write still get no credit through time for what must be held.
- **P18b.4 (every-byte targets do not break a working package)** B2-all's hard mean is within 15 points of B2's in at least two seeds. Held at about 50%: five sixths of the loss is now on prompt bytes.

Reading, written before the result. P18b.1 and P18b.2 holding say Mr. Xu's purest form does what the answer-only target could not: it gives a forward, within-step rule a path to its input code, and a package that learned nothing learns. P18b.1 holding with P18b.2 missing says the input code was not the only blockage: the layers above the write need credit through time whatever the target, and the cheap rule needs an architecture in which every synapse is a write synapse. P18b.4 missing says every-byte targets dilute the answer signal, and the comparison for B3-all is then with a weakened ceiling. Whatever the outcome, these are births on single operations: no life, no retention, no composition.

## Report

Per arm and seed: the six checkpoints of the birth curve (four operations at five k), the hard and max/min means beside the partner's, the first bank's own head, embedding movement by symbol role at each checkpoint, the credit census, the four verdicts, machine, commit, each process's environment, CPU-seconds, forecast against measured, what went wrong. Recompute your tables from the receipts and check each receipt's final step, loss mode and completion record. `E18B_PREDICTIONS.md`, `E18B_REPORT.md`, six births with weights (`git add -f` the `.pt`), logs, one paragraph as `E18B_REVIEW_SECTION_9_80.md`.

**Standing notes.** Development panels held out by ordered pair (evidence level E2). The rule uses weight-sign feedback, whose standing under the locality constraint is the owner's to decide; B2-all also uses a backward pass over the item inside each bank, B3-all does not. No learning-core improvement is claimed. Cost: the partners took 5,700 (B2) and 6,400 (B3) CPU-s per birth on a four-core cloud container; six births about 36,000 CPU-s, about 2.6 hours.
