# E32: every-byte births under the sweep with one switch each, scored past the trained range (14 November 2026, revision 2)

Evidence level E2. Revision 1 (aaa070e4) was audited by the outside reviewer and held; no birth ran under it, and its four predictions are withdrawn unjudged. The table at the end lists what changed and the command behind each finding. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each. Two workers: a main worker, and a second who reproduces the dry births.

Shell names used below: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106`.

## Why

A GRU born under the sweep with every byte as a target answers mod-89 addition at 71–80% with 12 letters between the operands, 17–41% with 20, and 2% with 40. Born under backprop it answers 92–94% at 20 and 80–85% at 40. Births train on 0 to 12 letters, so this is failure past the trained range. The target test gives no answer mask, so it is the condition that matters there.

`for a in S-all B-all; do for s in 73129 93683 116089; do git show $D:$M/exp45/e28/e28_dev_${a}_$s.json | python3 -c "import json,sys; d=json.load(sys.stdin); x=d['curve'][-1]['sealed']['add89']; print('$a $s',{k: round(100*v['exact']/v['n'],1) for k,v in x.items()})"; done; done`

The reviewer's unregistered measurement adds that the floor has no unit that keeps half its state across even 12 letters, where backprop births have 67–85 such units. E32 asks which change gives the sweep such units, and whether accuracy past the trained range follows.

## Arms

Every arm is E28's S-all birth (`exp28_everybyte.py dev S-all <seed>`: every byte a target, the sweep, 24,576 updates) with one addition. Seeds 73129, 93683, 116089. Eighteen births, about 16 CPU-hours.

| Arm | What is added | Kind |
|---|---|---|
| floor | nothing | on file (E28 S-all) |
| ceiling | backprop | on file (E28 B-all) |
| **G** | the error crossing rows 0–511 (reset and update gates) of `weight_hh_l0` and `weight_ih_l0` uses the exact weights. Only the error sent back changes. | bound |
| **H** | the error crossing all of `weight_hh_l0` uses the exact weights. Only the error sent back changes. | bound |
| **K** | extra state: a feedback matrix `Bhh` (768 × 256) that the error crosses in place of the recurrent map's signs, with its own update after each optimizer step | candidate |
| **P3a, P3b, P3c** | an extra local loss term on the update-gate rows, switched on after update 4,096 | candidate |

G and H read forward weights across the whole backward pass and are bounds, not candidates.

### Arm K, in full

- `Bhh` is drawn once from U(−1/16, 1/16) by `torch.Generator().manual_seed(seed + 31)`, a generator of its own, so the data order is untouched.
- After each optimizer step, with `W0` and `W1` the recurrent matrix before and after: `Bhh ← (1 − 0.00025) · Bhh + (W1 − (1 − 0.00025) · W0)`.
- The difference between `Bhh` and the forward matrix shrinks by exactly 0.99975 per update: 36% of the start is left at update 4,096, 13% at 8,192, 0.2% at the end. **K becomes H by construction.** What K tests is whether inexact credit across the recurrent map in the first few thousand updates does lasting harm.

### Arms P3, in full

- Updates 1 to 4,096 are the floor's, unchanged. The floor's undecided share is 65% at update 4,096 and climbs to 92% by the end; the push starts where that climb starts.
- From update 4,097, this term is added to the loss:

```
msk  = (x != X.BOS)                      # input positions; position 0 and padding are BOS
a    = lin(z1.detach(), ih)[..., 256:512] + lin(hp.detach(), hh)[..., 256:512]
z    = sigmoid(a)
term = lam * (relu(0.49 - (z - 0.5).abs()) ** 2)[msk].sum() / (msk.sum() * 256)
```

- The inputs are detached, so the gradient reaches only rows 256–511 of the two matrices and their biases: each weight moves by its own input times a factor of its own unit.
- The push stops when a gate leaves 0.01 to 0.99, not 0.1 to 0.9. A gate at 0.9 keeps 0.9^40 = 1.5% of its state across 40 letters; a gate at 0.99 keeps 67%. E31's limit of 0.9 could not make a unit that holds 40 letters.
- `lam` is set once, on the batch of update 4,097, so that the term's gradient norm on those rows is 0.25 (P3a), 0.05 (P3b) or 0.01 (P3c) of the task loss's gradient norm on the same rows. The three values are written into `exp45/e32/E32_PREDICTIONS.md` and pushed before any full birth.
- Both norms are recorded at every saved row, because the ratio describes one update only.
- **A known risk, stated beforehand.** A gate is pushed towards whichever end it is nearer on each position, so a unit may be pushed to keep on some letters and overwrite on others. The count of keeping units at every row shows whether that happened.

## Code

- One new runner, `exp32_births.py`, with the forward pass and all three additions written for the birth loop. Edit nothing existing. E31's files may be read but are not imported: `arch31.py` on worker A's branch holds none of the push or the feedback update, which live in the life runner.
- Receipts say `experiment="E32"` and go to `exp45/e32/`.
- **Weights.** Commit (`git add -f`) every final file, and the row files of seed 73129 only. Row files of the other seeds are measured, their hashes recorded, and not committed.

## Checks before any full birth

1. **Floor reproduced.** With nothing added, a dry birth at 2,048 updates has the same `tensors_sha256` as `exp45/e28_qual/dry/e28_dev_S-all_dry2048_73129.json`.
2. **Same data and same forward pass in every arm.** On the first batch every arm's task loss equals the floor's bit for bit, and each arm's `data_digest` equals E28's at every saved row.
3. **G and H.** On one batch the error sent back across the recurrent map equals `g[:, :512] @ W[:512] + g[:, 512:] @ S[512:]` for G and `g @ W` for H, with `S = sign(W) * mean|W|`, within 1e-6. Weight and bias gradients given the same incoming error equal the floor's.
4. **K.** The update identity holds to 1e-7 at updates 1, 10 and 100. With `Bhh` set equal to the forward matrix by hand, K's gradients equal H's. Every saved row records the cosine between `Bhh` and the forward matrix.
5. **P3.** The term's gradient is zero on every tensor and row outside the update-gate rows. On one batch it equals a second form written by hand as the two-factor product, within 1e-6. With `lam = 0` a dry birth equals the floor's. **Each P3 birth's row at 4,096 has the same `tensors_sha256` as `e28_dev_S-all_<seed>_row4096.pt`.**
6. **Second worker.** A second worker writes the three additions independently, without reading the first worker's code, and runs one dry birth of each arm at 6,144 updates for seed 73129 (past the switch-on). Agreement means equal hashes, or gradients within 1e-6 on one common batch when each runs the other's function. Disagreement: both stop and report.
7. **Cost.** CPU time of each dry birth and a forecast for the eighteen.

## Reading the files

A new script `e32_read.py` imports `U_of` and `H_of` from `e30_gates.py` (which cannot read E32 files itself: it looks for `e28_dev_*.json` and loads weights strictly) and loads only the network's own tensors.

- **First**, it reproduces the floor and ceiling rows of `exp45/e30/E30_UNSEEN.json` exactly, and then computes the same for seed 116089, which E30 did not reach.
- **Accuracy** on mod-89 addition at 12, 20 and 40 letters, 300 items each, for every saved row.
- **Keeping units** at every saved row: a unit keeps across k letters if, on the add-89 panel with k letters, the product of its update gate over the k letter positions is at least 0.5 in at least half the items. Counts at k = 12 and k = 40.
- **U** on the add-89 panel, as E30 defines it.

## Endpoints and marks (fixed)

All read on the **mean of the last three saved rows** (16,384, 20,480 and 24,576), because a single row swings: the floor's accuracy at 12 letters goes 66, 37, 75 over those rows in seed 73129.

- **Primary:** accuracy at 40 letters. **Secondary:** accuracy at 20 letters.
- **Moves:** at least 30 points above the same seed's floor, in at least two of three seeds.
- **Closes:** within 15 points of the same seed's ceiling, in at least two of three seeds.
- **Partial:** 15 to 30 points above the floor in at least two of three seeds. Reported, not claimed.
- **Harm:** accuracy at 12 letters at least 15 points below the same seed's floor in at least two of three seeds.
- Differences under 15 points are not read.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P32.1 | H moves accuracy at 40 letters | 45% |
| P32.2 | G moves accuracy at 40 letters | 30% |
| P32.3 | K moves accuracy at 40 letters | 35% |
| P32.4 | K ends within 15 points of H at 40 letters in all three seeds | 50% |
| P32.5a | P3a ends with at least 20 keeping units at 12 letters in all three seeds | 60% |
| P32.5b | the same for P3b | 45% |
| P32.5c | the same for P3c | 25% |
| P32.6 | at least one P3 strength moves accuracy at 20 letters | 30% |
| P32.7 | at least one P3 strength moves accuracy at 40 letters | 15% |
| P32.8 | no arm does harm | 70% |

## Readings (fixed; every outcome has one)

**Bounds and K**

| H | K | Reading |
|---|---|---|
| moves | moves | Exact credit across the recurrent map is enough, and a learned pathway that starts random and converges recovers it. Learned feedback goes forward as a candidate, on fresh seeds. |
| moves | does not | Exact credit across the recurrent map was needed in the first few thousand updates; a pathway that only converges later does not recover it. |
| does not | moves | Not expected, since K converges to H. Reported as an anomaly and rerun before anything is said. |
| does not | does not | The recurrent map's backward pass is not where this loss arises. If G also does not move, the same holds for the gate rows of both maps. Nothing wider is concluded. |

If G moves and H does not, or the reverse, the two bounds are reported side by side and the gate rows of the input map are named as the difference.

**The push, per strength**

| Keeping units (P32.5) | Accuracy | Reading |
|---|---|---|
| fewer than 20 | any | The push did not take at this strength. This arm says nothing about gates. |
| at least 20 | moves at 40 | Units that keep, made by a local push, are enough. Run again on fresh seeds before a claim. |
| at least 20 | moves at 20 only | The push extends the range but not to 40; reported with the keeping count at 40. |
| at least 20 | neither | Keeping units alone do not give accuracy past the trained range. |

**Marks.** "Moves" without "closes" is reported as a gain that leaves most of the gap; "is enough" is said only when an arm closes. An arm that moves 40 letters and harms 12 is reported as a trade and not as a gain.

## Done

Runner, reader, check receipts, both workers' dry receipts, eighteen complete births, and `E32_REPORT.md` with one table per seed (arm; accuracy at 12, 20, 40; keeping units at 12 and 40; U; hard mean) beside floor and ceiling, and the outcome of each prediction. No readings beyond the tables above.

## What changed from revision 1, and why

Findings are the outside reviewer's; I reran each command marked "rerun" and it gave what the reviewer reported.

| Finding | Evidence | Change |
|---|---|---|
| K and the push are not "a change to the backward pass" | reading of both runners | arms described as what they are |
| The push's mask was not pinned (`y != BOS` in E28, `x[:, 1:] != BOS` in E31) | `git show $D:$M/exp28_everybyte.py \| sed -n 13p` (rerun) | mask written as code |
| Strength set on an untrained network: every gate is in the band, and the rise the push should stop comes later | reviewer's script on untrained draws (not rerun); E30 record | switch-on at 4,096, strength set there, row-4096 equality check |
| 40 letters alone hides partial gains | the receipts' 20-letter column (rerun) | 20 letters as secondary endpoint |
| One final row is noisy | E30 record | mean of the last three rows |
| `e30_gates.py` cannot read E32 files | `git show $D:$M/e30_gates.py \| sed -n '93p;99p;114p'` (rerun) | new reader that reproduces E30 first |
| A limit of 0.9 cannot make a 40-letter holder | 0.9^40 = 0.015 | limit 0.99; keeping units counted |
| `arch31.py` holds none of the code; worker A has no P2 code | `grep -c` on both branches gives 0 and 16, and 0 and 7 (rerun) | code written for the birth loop, hand-form check, second worker |
| Outcomes without readings; "does" too low a mark; K's reading contradicted its own rule | `python3 -c "print([round((1-0.00025)**n,4) for n in (4096,8192,12288,24576)])"` (rerun) | marks, full reading tables, K's reading rewritten |
| One bound for one map | — | arm G added |
| 405 MB of row files | — | rows committed for one seed only |
| Dependence on E31's stage 1b | worker A's branch still at bc2ddfa9 | removed |
