# One cloud job for Claude Opus 5.5 workers: E22, a life with no backprop anywhere (10 November 2026, 21:30 EST)

> **Status under the process rules of 10 November (review 9.82).** Not yet audited by Codex. Send this brief to Codex first; a worker starts after that audit is answered. **Commits:** one commit at completion; nothing partial on the default branch. **Sizing (Mr. Xu, 10 November):** never more processes than cores. On a four-core container run the three lives of one half and nothing else.

**Repository:** `ChangweiZhou/digitalbaby`, default branch `codex/minifly-consolidated-20261023`; start from its head. **Folder:** `MOD97_LOCAL_RULE_20261106/`. Python 3, PyTorch on CPU, NumPy. Two halves, one per instance; say which in your first message. Each half is three lives, about two hours on four cores.

## Why

Mr. Xu's ruling of 10 November: a candidate is born by its own rule before any promotion claim (review 9.82). E18 showed that the windowless two-bank learner can be born with no backprop anywhere, by weight signs with credit through time, at 0.94–0.96 of the backprop birth (arm B0, review 9.71). Every life on file starts from a backprop birth. Nobody has run a life from a rule-born one. This job does, beside the same life from the backprop birth on the same class of machine.

## Runs: six lives, seeds 73129, 93683, 116089

Both halves: `E16_ARCH=bank2 E16_BOOKDECAY=off E16_REPLAY=raw E16_CAP=65536 E16_TRACEUP=all`, arm `T`, the staged stream, share 0.5, as E17 Phase 4.

```
# half a: born by the rule (birth e17_dev_bank2_nbd_bT_tall_<seed>.pt, from E18's arm B0)
E16_ARCH=bank2 E16_BOOKDECAY=off E16_REPLAY=raw E16_CAP=65536 E16_TRACEUP=all E16_BIRTHRULE=T python3 exp16_integrated.py T <seed>
# half b: born by backprop (birth e17_dev_bank2_nbd_<seed>.pt), tagged so that the Phase 4 receipts made on the owner's machine are not overwritten
E16_OUTTAG=_cloud E16_ARCH=bank2 E16_BOOKDECAY=off E16_REPLAY=raw E16_CAP=65536 E16_TRACEUP=all python3 exp16_integrated.py T <seed>
```

Receipts `exp45/e17_life_T_bank2_nbd_tall_bT_raw65536_<seed>.json` and `exp45/e17_life_T_bank2_nbd_tall_raw65536_cloud_<seed>.json`, with weights at completion (`git add -f`). The two halves differ in the birth alone: same network, rule, stream, reservoir, decay in life (0.25), panels and harness. The births themselves are not equal: the rule-born ones stand at 87.5 / 90.7 / 84.7 on the hard mean, the backprop ones at 91.2 / 94.5 / 90.3, so a difference in life cannot be split between how the learner was born and how well. Report both lives' scores at checkpoint zero beside everything else.

**Admission.** One dry run of your command (`E11_STAGE_UPDATES=4`, archived under `exp45/archive_e22_dry/cloud_<date>/`): seven stages, a completion record, and `birth_admission` reading `manifest` or `own receipt`; a life stops by itself if its birth's hash is not the expected one. Record that your three receipts do not exist. Commit `E22A_PREDICTIONS.md` or `E22B_PREDICTIONS.md` before launch. Launch detached, keep one foreground command waiting while the lives run, and poll logs with short commands. No reruns, no tuning.

## Predictions (copy into your predictions file; half b copies them for the record and judges none alone)

- **P22.1 (a rule-born learner lives)** The rule-born learner's mod-97 mean after block 1 is at least 80 in at least two seeds. (Backprop-born on the owner's machine: 92.9 / 92.1 / 90.7.)
- **P22.2 (and ends where the backprop-born one does)** At the end of life it is within 5 points of its backprop-born partner on mod-97 and on mod-73 in at least two seeds.
- **P22.3** Reported, not predicted: the unrehearsed development operations at every checkpoint, the scores at 20 and 40 distractors, and compositions (none is expected above 25 on the held pairs; this network does not compose).

Reading, written before the result. P22.1 and P22.2 holding say the ruling can be met in this network for single operations: a learner born and living under the working rule, with no backprop at any point, ends where the backprop-born one does. P22.1 missing says a rule-born birth is a worse place to start a life than its score suggests. P22.2 missing with P22.1 holding says it starts well and falls behind. Nothing here bears on composition or on retention without replay.

## Report

Per life: the seven checkpoints (mod-97, mod-73, development operations, held pairs; whole answers and final answers), the scores at 20 and 40 distractors at the end, the credit census, update CPU and scoring CPU. The two halves side by side once both are in; the verdicts; machine, commit, environments; forecast against measured; what went wrong. `E22_REPORT.md` (the second half to finish writes it), receipts, weights, logs, one paragraph as `E22_REVIEW_SECTION_9_85.md`.

**Standing notes.** Evidence level E2 on panels held out by ordered pair. The rule is the working life rule by Mr. Xu's ruling: a backward sweep over the item with weight signs. No learning-core improvement is claimed. Cost: such a life took about 6,700 CPU-s on the owner's machine under load; expect two hours for three on four cores, and forecast from your dry run.
