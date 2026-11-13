# Draft for hostile audit: E28, the GRU born under every-byte targets (13 November 2026, 11:55 EST by the clock)

> **Status.** Draft. **No worker starts from this file** until a hostile audit has read it and the coordinator has answered. Implementation and runs are an Opus 5.5 worker's; the coordinator writes and audits briefs (Mr. Xu, 13 November). The auditors are fresh sessions of the same model family: second readers, not independent ones. Evidence level aimed at E2, development panels, births only.

## Why

The target test has no answer mask: its learner predicts every next byte. E24 (review 9.110 C) showed, on the two-bank network, that a birth under every-byte targets collapses under the sign sweep (hard mean 0.8 / 1.0 / 1.2) and not under backprop (91.5 / 89.3 / 91.2), and that giving the answer bytes half the loss restores the sweep (81.5 / 79.8 / 83.3). The two-bank network has been left behind. The network that composes is the GRU of E26. **Whether the sweep on the GRU survives every-byte targets is not on file,** and it decides whether the working rule can enter the target test at all.

## The question

Born under every-byte targets, does the GRU reach under the sweep what it reaches under backprop?

## One switch on qualified code

The runner is E26's, `MOD97_LOCAL_RULE_20261106/exp26_candidate.py`, whose update is one function used by both arms (shown by `sed -n '84,91p' exp26_candidate.py`):

```
x, y, mask = X.batch_tensor(items)
logits = arch26.forward(m, x, local, True)[0]
loss = E.cross_entropy(logits[mask], y[mask])
opt.zero_grad(); loss.backward(); opt.step()
```

E28 changes the loss mask and nothing else. New files only (`exp28_everybyte.py`, importing `exp26_candidate` and `arch26` unedited):

| Targets | Loss |
|---|---|
| answer (E26, for the host check only) | `mask` as returned by `X.batch_tensor` |
| every byte | every real position: `y != E.BOS` (the form `exp16_integrated.update16` uses; shown by `grep -n "mask = y != E.BOS" exp16_integrated.py`) |
| balanced | one half the mean over answer positions plus one half the mean over all other real positions (the form of `E16_LOSSW=balanced`; shown by `grep -n "balanced" exp16_integrated.py`) |

Everything else is E26's: the GRU of 256 units with the written-out step, the same initial values per seed, the same birth data in the same order (four mod-89 operations, 0 to 12 distractor letters, 24,576 updates of 64 items), AdamW at 0.001, betas (0.9, 0.98), **decay 0.25 in every arm**, the embedding table without decay, scoring by whole own answers on the sealed panels at the six rows.

## Arms. Seeds 73129, 93683, 116089. One host

| Arm | Update | Targets | Status |
|---|---|---|---|
| S-ans, B-ans | sweep, backprop | answer | on file: E26's births, 98.0 / 98.2 / 99.8 and 98.5 / 98.7 / 93.7 (`exp45/e26/e26_dev_{S,B}_<seed>.json`, cloud host, torch 2.15.0). Rerun seed 73129 of each as the host check |
| **S-all** | sweep | every byte | new |
| **B-all** | backprop | every byte | new |
| **S-bal** | sweep | balanced | new |

Nine new births and two host checks. No lives. No balanced backprop arm: it is wanted only if B-all falls, and is then its own decision.

## Stages

**Stage 1, then stop.**
1. *Host check.* With the new runner at answer targets, seed 73129, both arms, in a fresh folder: every weight tensor equals E26's birth of that arm (`exp45/e26/e26_dev_{S,B}_73129.pt`), and the six rows' sealed scores equal its receipt's. If the host is of another class and they differ, stop: the baselines must then be rerun on your host for all three seeds, and you say so before running them.
2. *The mask.* On one birth batch: the every-byte mask covers every real byte of prompt and answer and no padding position; the balanced loss equals one half of each part computed separately, and so does its gradient, under both paths; with answer targets the loss and every gradient equal `exp26_candidate.step`'s bit for bit.
3. *One real step per new arm* equals the same AdamW given an independently computed gradient, as E26's check 2a did.
4. *Dry births* of the three new arms at 2,048 updates complete; report CPU per update.

**Stage 2, on the coordinator's release:** the nine births.

## What every birth records

- A **cumulative** finite-loss flag over all updates, and the number of updates it covers (the gap Codex found in E26's births, review 9.110).
- The loss on answer positions and on other positions separately, as means over each 512 updates, for all 48 blocks. E24's every-byte births rose and then died between 8,192 and 12,288 updates and nothing was recorded in between.
- **The weights at every scored row** (4,096 to 24,576). Commit the six row files of the three S-all births and the final file of every birth; keep the rest on the container and say so. Review 9.114 found that the weights at which E24's collapse happened were never saved.
- Host, command, environment, source hashes, optimizer groups read back.

## Endpoints

Final hard mean (add89 and sub89 at k = 12, whole answers) and max/min mean; the six-row curve; the two loss curves. Max and min are read against 50, the level of copying one operand.

## Predictions (copy unchanged into `E28_PREDICTIONS.md` before stage 2). "Holds" means in at least two of three seeds unless it says all

- **P28.1** B-all ends at a hard mean of at least 85. Confidence about 80%.
- **P28.2** S-all ends at or below 10 in all three seeds: the collapse of E24 repeats on the GRU. Confidence about 50%.
- **P28.3** S-bal ends at or above 60. Confidence about 60%.

## Readings, fixed now

- *Feasibility and gap are read apart.* Each new arm against the absolute level of its answer-target birth, and S-all minus B-all seed by seed.
- **P28.1 holds and P28.2 holds:** on the GRU as on the two-bank network, every-byte targets defeat the sweep and not backprop, in these configurations at this dose. The rule cannot enter the target test as it stands. The saved row files then let E27's measurement be made where the collapse happens.
- **P28.2 misses with S-all near its answer-target level in all three seeds:** the collapse of E24 belonged to the two-bank network, and the rule's obstacle to the target test is composition (E26), not the mask.
- **P28.2 misses with S-all in between:** the levels are reported; one seed above 10 does not settle survival.
- **P28.3:** reaching 60 shows an effect of the weighting only where S-bal is above S-all in the same seed; the paired difference is given for every seed. The balanced loss uses the answer mask and is a diagnostic, not a candidate.
- **P28.1 misses:** every-byte targets cost backprop on the GRU too; report the level and do not single out the rule.

## What this cannot say

Births only; nothing about lives or composition under every-byte targets. One size, three seeds, one host class. The decoder reruns the network for each output byte.

## Running it

One host, at most three processes on four cores. E26's births took about 3,100 to 3,300 CPU-seconds each on such a host; eleven births are about ten CPU-hours, three and a half hours of wall time. Replace this with stage 1's measurement. `exp26_candidate` refuses resume variables; an interrupted birth is kept as written and reported, and restarted only on the coordinator's word. Own branch; nothing to the default branch.

## Questions for the audit

1. Is one switch really one switch here: does anything else in a birth depend on the mask (the data digest, the census, the scorer)?
2. Should B-ans and S-ans be rerun for all three seeds on the worker's host regardless, given what E24's host check cost?
3. Is 162 MB of row files (36 files) worth committing for both every-byte arms, against 81 MB for S-all alone?
