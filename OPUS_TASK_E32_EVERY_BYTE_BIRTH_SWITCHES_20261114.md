# E32: every-byte births under the sweep with one switch each, scored at 40 letters (14 November 2026)

Evidence level E2. One worker. **Not yet audited by a second reader; no worker starts until it is audited or Mr. Xu waives that.** Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each.

## Why

E28 and E30 together give the largest gap on file. A GRU born under the sweep with every byte as a target answers mod-89 addition at 75–80% with 12 letters between the operands and 2% with 40 (seeds 73129 and 93683; `exp45/e30/E30_UNSEEN.json`). Born under backprop it holds 82–85% at 40. Born under the sweep with answer bytes only it holds 96–97% at 40. The target test gives no answer mask, so this is the condition that matters there. E32 asks which of E31's switches removes the gap at birth, where a run costs under one CPU-hour.

## Arms

Every arm is E28's S-all birth (`exp28_everybyte.py dev S-all <seed>`: every byte a target, the sweep, 24,576 updates, rows saved every 4,096) with one change to the backward pass, taken from E31. Seeds 73129, 93683, 116089.

| Arm | Change | Kind |
|---|---|---|
| floor | none | on file (E28 S-all) |
| ceiling | backprop | on file (E28 B-all) |
| **H** | exact weights for all of `weight_hh_l0` | bound |
| **K** | Kolen–Pollack feedback on the recurrent map, random start, as E31 defines it | candidate |
| **P2a, P2b, P2c** | the bounded gate push of E31 addendum 2, at ratios 0.25, 0.05, 0.01 set on the first batch | candidate |

Fifteen births, about 14 CPU-hours.

## Code

- Take `arch31.py` from worker A's branch `claude/sleepy-knuth-q96dd1` **after** E31's stage 1b agreement is pushed, so the P2 code is the cross-checked one. Record the commit and the file's hash.
- New runner `exp32_births.py`: E28's loop with `arch31.forward` in place of `arch26.forward`, and for K the feedback update after each optimizer step. Edit nothing existing. Receipts `experiment="E32"`, in `exp45/e32/`; `git add -f` the weights.
- The gate push uses the task's mask of real positions; padding is excluded.

## Checks before any full birth

1. With no switch, a dry birth at 2,048 updates has the same `tensors_sha256` as `exp45/e28_qual/dry/e28_dev_S-all_dry2048_73129.json`.
2. For K: the update identity to 1e-7 at updates 1, 10, 100, and the cosine between `Bhh` and the forward matrix recorded at every row.
3. For P2: the term's gradient is zero outside the update-gate rows, and λ and both norms are in the receipt.
4. One dry birth of each arm at 2,048 updates, with its CPU time, and a forecast of the full cost.

## Endpoint (fixed)

For each final file and each saved row, with `e30_gates.py` unchanged: accuracy on mod-89 addition at 12 and at 40 letters (300 items each), and U on the add-89 panel. The endpoint is **accuracy at 40 letters**. The hard mean at 12 letters, as E28 reports it, is given beside it. First compute the floor and ceiling for seed 116089, which E30 could not reach.

## Predictions (mine)

- **P32.1** H is at least 30 points above the floor at 40 letters in at least two of three seeds. About 50%.
- **P32.2** K is at least 30 points above the floor at 40 letters in at least two of three seeds. About 35%.
- **P32.3** At least one P2 strength is at least 30 points above the floor at 40 letters in at least two of three seeds. About 30%.
- **P32.4** Every P2 arm ends with U at least 20 points below the floor's (about 92%) in all three seeds. About 80%.

## Readings (fixed)

- H does and K does not: exact credit across the recurrent map would do it, and the learned pathway did not get there in a birth's length; look at the recorded cosine.
- A P2 arm does: decisive gates are enough without any feedback change. It is run again on fresh seeds before a claim.
- P32.4 holds and P32.3 misses: undecided gates go with the loss but do not cause it.
- None does: the every-byte loss at 40 letters is not in the recurrent map's backward pass or the gates.
- Differences under 15 points between single births are not read.

## Done

Runner, check receipts, fifteen complete births, and `E32_REPORT.md` with the table (arm, seed, accuracy at 12, at 40, U, hard mean) beside floor and ceiling, and the four outcomes. No readings beyond them.
