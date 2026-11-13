# E28, the GRU born under every-byte targets. Revision 2, after a hostile audit (13 November 2026, 12:01 EST by the clock)

> **Status.** Stage 1 is released to one Opus 5.5 worker. Stage 2 waits for the coordinator's release. Revision 1 (commit 02897c6e) was read by a fresh hostile auditor of the same model family, a second reader and not an independent one. It returned nine findings; the coordinator reran the evidence for seven and adopted all nine (list at the end). Evidence level aimed at E2, development panels, births only.

## Why

The target test has no answer mask: its learner predicts every next byte. E24 (review 9.110 C) showed a birth under every-byte targets collapsing under a sign rule (hard mean 0.8 / 1.0 / 1.2) and not under backprop (91.5 / 89.3 / 91.2), with a balanced loss restoring the sign rule (81.5 / 79.8 / 83.3). **E24's configuration is not E28's in three ways:** the two-bank network; a sweep with credit cut in time and the trace routine (`birthrule T, traceup all` in its receipts; `sed -n '624p' exp16_integrated.py`); and a backprop partner at decay 0.30 against 0.25. E28's sweep is sign feedback through the whole item on the GRU of E26. Whether that survives every-byte targets is not on file.

E28 is necessary for the rule to enter the target test and not sufficient: the test's distances exceed this GRU's measured span of about 12 bytes (review 9.110 E).

## The question

Born under every-byte targets, does the GRU reach under the sweep what it reaches under backprop?

## The code: a new runner, one difference in the condition

The condition differs from E26's birth in the loss mask alone. The code does not: E26's birth loop cannot record what E28 needs, so the loop is new and the host check is what ties it to E26.

- New file `MOD97_LOCAL_RULE_20261106/exp28_everybyte.py`. It imports `exp26_candidate` and `arch26` unedited for the model, the data, the optimizer groups and the scorer. **It does not call or patch `exp26_candidate.dev` or `exp26_candidate.step`**: `dev` writes `experiment="E26"`, an "answer-masked" update string and `e26_dev_` file names (`grep -n '"E26"\|e26_dev_\|answer-masked' exp26_candidate.py`), and `admit_e26` does not look at targets, so a patched birth would pass as an E26 birth.
- Every receipt carries `experiment="E28"`, a `targets` field (`answer`, `all`, `balanced`), and files are named `e28_dev_<arm>_<seed>` in `exp45/e28/`.

| Targets | Loss |
|---|---|
| `answer` (host check only) | cross entropy over the answer mask `X.batch_tensor` returns |
| `all` | cross entropy over every real position, `y != E.BOS` (padding is BOS in y; no real target is BOS) |
| `balanced` | one half the mean over answer positions plus one half the mean over the other real positions |

Everything else is E26's: the 256-unit GRU with the written-out step, the same initial values per seed, the same birth data in the same order (24,576 updates of 64 items), AdamW at 0.001, betas (0.9, 0.98), decay 0.25 in every arm (E26's B births used 0.25 too), the embedding table without decay, whole own answers on the sealed panels at the six rows.

## Arms. Seeds 73129, 93683, 116089

| Arm | Update | Targets | Status |
|---|---|---|---|
| S-ans, B-ans | sweep, backprop | answer | on file: E26's births, 98.0 / 98.2 / 99.8 and 98.5 / 98.7 / 93.7. Seed 73129 of each is rerun as the host check |
| **S-all** | sweep | every byte | new |
| **B-all** | backprop | every byte | new |
| **S-bal** | sweep | balanced | new |
| **B-bal** | backprop | balanced | new (added on audit: without it a shortfall of S-bal cannot be assigned to the rule) |

Twelve new births and two host checks.

## Host

**Precondition: Linux x86-64, torch 2.15.0+cpu, the class E26 was born on.** Print both before anything else. On any other host, stop and report; do not run the baselines on another class without the coordinator's word.

## Stage 1, then stop

1. *Host check.* The new runner at `answer` targets, seed 73129, both arms, in a fresh folder: `tensors_sha256` equals E26's receipt for that arm, every tensor equals `exp45/e26/e26_dev_{S,B}_73129.pt`, and all six rows' sealed scores and `data_digest` values equal the receipt's.
2. *Refusal.* `exp26_candidate.admit_e26` rejects an E28 birth file of every targets value. Show the assertion text.
3. *The mask.* On the first birth batch: the `all` mask covers every real byte of prompt and answer and no padding position. With `answer` targets, the weights after one step equal `exp26_candidate.step`'s bit for bit under both paths.
4. *The balanced loss* is checked against an independently written form: unreduced cross entropy with weight 0.5/n_a on answer positions and 0.5/n_o on the others, summed. Loss and every gradient tensor agree to a **relative tolerance of 1e-5 of that tensor's largest entry**, under both paths. They do not agree bit for bit (the auditor measured 3e-7 to 5e-7); do not ask them to.
5. *One real step per new arm* equals AdamW given an independently computed gradient: bit for bit for `all`, to the tolerance of item 4 for `balanced`.
6. *Dry births* of the four new arms at 2,048 updates complete, with every record below present. Report CPU per update.

## What every birth records

- A **cumulative** finite-loss flag over all 24,576 updates and the count it covers.
- **Loss by target role**, as means over each 512 updates for all 48 blocks: distractor letters; operand digits; structural bytes (operator, `=`, end); answer bytes. State how each role is identified from the item, and show on one batch that the four roles partition the real positions.
- The `data_digest` at each row. It hashes x, y and the answer mask, so it must equal E26's for the same seed and row in all twelve births; the runner asserts this.
- **The weights at every scored row.** Every row file's SHA-256 goes in the receipt.
- Host, torch, command, environment, source hashes, optimizer groups read back.

**Committing weights.** `.gitignore` ignores `*.pt` (`git check-ignore -v MOD97_LOCAL_RULE_20261106/exp45/e28/x.pt`), so each needs `git add -f`; this is how E26's life weights were left off its branch. Commit the six row files of each S-all birth, the final file of each other birth, and the two host checks: 18 + 9 + 2 = 29 files of 4,501,581 bytes, about 131 MB. The other row files stay on the container with their hashes in the receipts; say so.

## Endpoints

Final hard mean (add89 and sub89 at k = 12, whole answers); the peak hard mean over the six rows and its row; the four loss curves. Max and min means are reported and are descriptive: nothing is predicted of them.

## Predictions (copy unchanged into `E28_PREDICTIONS.md` before stage 2)

- **P28.1** B-all ends at a hard mean of at least 85 in all three seeds. Confidence about 70%.
- **P28.2** S-all ends at or below 10 in all three seeds. Confidence about 40%: E24's collapse was under a different sweep.
- **P28.3** S-bal ends at least 20 points above S-all in at least two of three seeds. Confidence about 50%.
- **P28.4** B-bal ends within 10 points of B-ans in all three seeds. Confidence about 70%.

## Readings, fixed now

"Near" means within 10 points of the same seed's answer-target birth. Differences under 15 points between two births of one seed are not read: E26's own adjacent rows swing by up to 14.

- **Rule-specific collapse:** S-all at or below half of B-all in all three seeds, with B-all at or above 60 in all three. Then every-byte targets defeat this sweep and not backprop on the GRU, at this dose, and the rule cannot enter the target test as it stands. Whether S-all never learned or learned and then died is read from its peak row (a peak at or above 50 is "learned and died") and said either way.
- **Survival:** S-all near S-ans in all three seeds. Then E24's collapse belonged to E24's configuration, with no attribution among its three differences, and the rule's obstacle to the target test is composition (E26), not the mask.
- **Shared cost:** B-all below 60 in any seed. Every-byte targets cost backprop here too; that seed is dropped from the rule-specific reading, and with two or more such seeds no rule-specific claim is made, whatever S-all does. The paired gap is still reported.
- **Anything else** (S-all between, or split across seeds): the levels and paired gaps are reported and neither survival nor collapse is claimed.
- **The balanced arms:** S-bal is read against S-all (P28.3) for the effect of the weighting, and S-bal minus B-bal against S-ans minus B-ans for whether the rule pays more than backprop under it. The balanced loss uses the answer mask: a diagnostic, not a candidate.
- **Flooding is not shown by levels.** At E26's born weights the non-answer loss carries over 99% of the every-byte gradient under both paths (auditor's measurement, rerun by the coordinator), so proportion alone does not separate the rules. A cause is read only from the role curves and from E27's measurement on the saved rows.

## What this cannot say

Births only; nothing about lives or composition under every-byte targets. One size, three seeds, one host class. Not a comparison with E24 like for like.

## Running it

One host, at most three processes on four cores. E26's births took 3,068 to 3,282 CPU-seconds; fourteen are about 12.5 CPU-hours, a little over four hours of wall time. Replace with stage 1's figure. No resume: an interrupted birth is kept as written, reported, and restarted only on the coordinator's word. Own branch; nothing to the default branch.

## The audit and what was done with it

| # | Finding | Coordinator's check | Done |
|---|---|---|---|
| 1 | E24's sweep was a different update, and its partner's decay differed | receipts read: `birthrule T, traceup all, decay 0.25` against `D, 0.3` | "Why" and the survival reading rewritten |
| 2 | Outcomes without readings; undefined "near"; no margin | read against the draft | readings rewritten with fixed margins; P28.1 now all three |
| 3 | A patched `dev` would pass as an E26 birth | `grep` on `exp26_candidate.py` lines 113, 135, 137, 176 | new runner, new names, refusal test |
| 4 | Balanced checks cannot hold bit for bit | auditor's script rerun: 3.4e-7 and 5.1e-7 | tolerance and an independent form |
| 5 | Flooding not separated; S-bal had no partner | script rerun: answer part 0.6 to 0.7% of the every-byte gradient norm under both paths | B-bal added; loss by role; reading on flooding |
| 6 | `*.pt` ignored; file count wrong | `git check-ignore`; 4,501,581 bytes per file | `git add -f` and the count stated |
| 7 | Max/min had no prediction | — | declared descriptive |
| 8 | Host check fails across classes | not runnable here (Mac torch 2.9.1) | host precondition; digests compared |
| 9 | Necessary, not sufficient, for the target test | review 9.110 E | said in "Why" |
