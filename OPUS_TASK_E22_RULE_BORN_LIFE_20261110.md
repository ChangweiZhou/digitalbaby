# E22 (version 2): one rule at birth and in life (10 November 2026, 23:00 EST)

> **Status.** Version 1 of this brief is withdrawn; no worker started it. Codex audited it (`PROJECT_DOCUMENTS/MINIFLY_E22_DISPATCH_HOSTILE_AUDIT_20261110.md`) and every point is accepted (review 9.87). This version has not been audited. **It is written to run on one host.** The coordinating session proposes to run it on the owner's machine on Mr. Xu's go; a cloud worker runs it only if told to, and then under the one-host rule below.

## What changed from version 1, and why

| Version 1 | The defect | Version 2 |
|---|---|---|
| "Within 5 points of its partner" as the main prediction | passes if both learners collapse together; the family mean hides addition and subtraction; "within" was not defined | acquisition, final competence and parity are separate verdicts with literal definitions; addition and subtraction are reported and gated on their own |
| One birth arm per cloud instance | host and birth history vary together; runs are not bit-identical across machines | all six lives on one host |
| A birth admitted by its hash | a hash says which file, not how it was trained | admission also checks the recorded training configuration and final step against what the life asks for (now in the harness; a rule-born file presented as a backprop birth stops the life) |
| "A life with no backprop anywhere" | the rule is a backward pass over the item with sign-modified feedback | named for what it is: a sign-feedback temporal sweep, the same in birth and in life |

## Why run it

Mr. Xu's ruling: a candidate is born by its own rule before any promotion claim (review 9.82). E18's arm B0 was born by the sign-feedback temporal sweep at 0.94–0.96 of the backprop birth. Every life on file starts from a backprop birth. This runs a life from the rule-born birth beside the same life from the backprop birth.

## Runs: six lives on one host, seeds 73129, 93683, 116089

```
# rule-born (birth e17_dev_bank2_nbd_bT_tall_<seed>.pt, E18 arm B0)
E16_OUTTAG=_e22 E16_ARCH=bank2 E16_BOOKDECAY=off E16_REPLAY=raw E16_CAP=65536 E16_TRACEUP=all E16_BIRTHRULE=T python3 exp16_integrated.py T <seed>
# backprop-born partner (birth e17_dev_bank2_nbd_<seed>.pt); repeated here so that both partners share host, harness and panel hash
E16_OUTTAG=_e22 E16_ARCH=bank2 E16_BOOKDECAY=off E16_REPLAY=raw E16_CAP=65536 E16_TRACEUP=all python3 exp16_integrated.py T <seed>
```

Both: arm `T`, staged stream, raw reservoir 65,536, block admission, share 0.5, life decay 0.25, answer loss, nothing frozen. The partners differ in the birth and in nothing else. The births differ in level as well as in history (hard mean 87.5 / 90.7 / 84.7 against 91.2 / 94.5 / 90.3), so a difference in life is a difference between two birth packages. **One host:** on the owner's machine, all six together. On a cloud container, three at a time in two waves on the same instance, or one seed pair per instance; never one birth arm per instance. `E16_LEGACY_BIRTH` is not to be used.

**The six births, pinned** (`exp45/BIRTH_MANIFEST.json`, which also records each one's training configuration and final step, 24,576):

| Seed | Rule-born, sha256 | Backprop-born, sha256 |
|---|---|---|
| 73129 | `e679f51601ee4128…` | `8ee5e4e673b70b28…` |
| 93683 | `03f1ed34257bd6e1…` | `5d37943da6dbe1d7…` |
| 116089 | `77d8e9c722dc3cf3…` | `aac1f6dfe55db03c…` |

**Admission.** A dry run of each command (`E11_STAGE_UPDATES=4`): seven stages, a completion record, `birth_admission: manifest`. Before any verdict, a validator over the six receipts checks: arm, seed, harness file hashes, birth file and hash, every setting equal between partners except `birthrule` and the birth fields, equal panel hashes between partners, stages 0 to 6 at 6,144 updates, and the recorded weight hash against the weight file. A cell that fails is not scored. If a run is interrupted, keep what it wrote, stop that cell, report it; no restart without authorization.

## Endpoints, written out

All at checkpoint 6 unless a block is named; twelve distractors; whole answers including the end symbol (`exact`); the four operations of a family weighted equally. "Hard mean" is the mean of add97 and sub97. Max and min are reported but carry a copy-one-operand shortcut of about 50%, so no verdict rests on them alone.

- **P22.1 (acquisition)** After block 1 the rule-born learner's mod-97 mean is at least 80 and its hard mean at least 70, in at least two seeds.
- **P22.2 (final competence, absolute)** At checkpoint 6 the rule-born learner's mod-97 mean is at least 75, its hard mean at least 65, and its mod-73 mean at least 75, all three in the same seed, in at least two seeds. The marks are 0.80 of the joint-stream reference's 92–94, and the backprop-born learner's 77–86 on the hard mean less the 4 to 6 points by which the rule-born births start lower and the spread between seeds.
- **P22.3 (parity, one-sided)** In at least two seeds, the same two for both families, the rule-born learner is no more than 5 points below its partner on the mod-97 mean and on the mod-73 mean. Doing better passes.
- **P22.4** Reported with no mark: the development operations (never shown in life) at every checkpoint for both partners; scores at 20 and 40 distractors; compositions.

If both partners end under the marks of P22.2, the result is common loss, whatever P22.3 says. Three seed pairs describe; they do not establish equivalence.

Reading, written before the result. P22.1 to P22.3 holding say that for single operations in this network the birth-lineage condition of the ruling can be met: one rule, birth and life, ending where the backprop-born learner ends and above an absolute mark. That is all it would say: the learner is answer-supervised, replay-supported and does not compose, and the promotion question is separate. P22.2 missing in both partners says this life loses the skill however it was born. P22.3 missing with P22.2 holding says the rule-born learner is competent and behind.

## Report

The six birth-to-life curves, add, sub, max and min separately; the verdicts with the validator's output; the sweep length used (the longest item, in bytes); the cost of each life and of each birth (the rule-born births took 4,307–4,321 CPU-s on a cloud container); update and scoring CPU apart; machine, CPU quota, peak memory. Review paragraph 9.85.

**Cost on the owner's machine:** six processes, about 6,700 CPU-s each, about an hour and a half.
