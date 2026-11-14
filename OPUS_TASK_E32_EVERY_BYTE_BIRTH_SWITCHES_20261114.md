# E32: every-byte births under the sweep with one switch each, scored past the trained range (14 November 2026, revision 4)

Evidence level E2. Revisions 1 (aaa070e4), 2 (00854280) and 3 (bc6a0eaf) were audited by the outside reviewer and held; no birth ran under any of them, and their predictions are withdrawn unjudged. Revision 4 is revision 3 with the six text changes the third audit named, listed in the last section; the reviewer stated that with those made it need not see the brief again. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each. Two workers: a main worker, and a second who checks independently.

Shell names used below:
`D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106; V=origin/claude/vibrant-cerf-sje726; R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`

## Why

A GRU born under the sweep with every byte as a target answers mod-89 addition at 71–80% with 12 letters between the operands, 20–40% with 20, and 2% with 40. Born under backprop it answers 92–94% at 20 and 80–85% at 40. Births train on 0 to 12 letters, so this is failure past the trained range, the condition the target test imposes.

The saved rows of the floor show how it happens. Call a unit a *keeper* across k letters if it keeps at least half its state over them (defined below). Seeds 73129 / 93683 / 116089:

| Floor row | Keepers at 12 letters | at 40 letters | Accuracy at 12 letters |
|---|---|---|---|
| 4,096 | 50 / 59 / 62 | 22 / 0 / 48 | 16 / 26 / 17 |
| 8,192 | 22 / 20 / 30 | 0 / 0 / 0 | 57 / 42 / 48 |
| 12,288 | 5 / 5 / 2 | 0 / 0 / 0 | 61 / 54 / 51 |
| 24,576 | 0 / 0 / 1 | 0 / 0 / 0 | 75 / 80 / 71 |

The reviewer's numpy reader, post hoc and unregistered; I reran it and got these numbers: copy `$V:$R/{np_gru,compare_pt_tensors,e32_r2_checks}.py` into a folder inside the clone and run `python3 -I <folder>/e32_r2_checks.py B`.

So the sweep has keepers early and **loses them between updates 4,096 and 12,288, while its accuracy at the trained distance rises**. Backprop births end with 67–85 keepers at 12 letters. E32 asks which change lets the sweep keep them, and whether accuracy past the trained range follows.

## Arms

Every arm is E28's S-all birth (`exp28_everybyte.py dev S-all <seed>`: every byte a target, the sweep, 24,576 updates) with one addition. Seeds 73129, 93683, 116089.

| Arm | What is added | Kind |
|---|---|---|
| floor | nothing | on file (E28 S-all, all rows) |
| ceiling | backprop | **rerun here with rows saved** (E28 B-all kept only its final files) |
| **G** | the error crossing rows 0–511 (reset and update gates) of `weight_hh_l0` and of `weight_ih_l0` uses the exact weights | bound |
| **H** | the error crossing all of `weight_hh_l0` uses the exact weights | bound |
| **K** | a feedback matrix `Bhh` that the error crosses in place of the recurrent map's signs, with its own update | candidate |
| **P4a, P4b, P4c** | a local loss term on the update-gate rows, from update 4,097 | candidate |

Twenty-one births (three ceiling reruns and eighteen new), about 19 CPU-hours. G and H read forward weights along the whole backward pass and are bounds. They differ in two ways: G adds the input map's gate rows, and G lacks the recurrent map's candidate rows (512–767).

**G and H, pinned.** With `S = sign(W) * W.abs().mean()` over the whole matrix, the error sent back is `g[:, :512] @ W[:512] + g[:, 512:] @ S[512:]` for each of G's two maps, evaluated as two products added, and `g @ W` for H. Weight and bias gradients are the floor's.

### Arm K

- `Bhh` (768 × 256) is drawn once from U(−1/16, 1/16) by `torch.Generator().manual_seed(seed + 31)`.
- After each optimizer step, with `W0`, `W1` the recurrent matrix before and after: `Bhh ← (1 − 0.00025) · Bhh + (W1 − (1 − 0.00025) · W0)`.
- The difference between `Bhh` and the forward matrix shrinks by exactly 0.99975 per update: 36% of its start is left at update 4,096, 13% at 8,192, 4.6% at 12,288. **K becomes H by construction.** `Bhh` is off by 100% at the start, falling to 36% at update 4,096 and 4.6% at 12,288. What K tests is whether inexact credit before `Bhh` has converged does lasting harm; K's own keeper counts by row say when.

### Arms P4

Updates 1 to 4,096 are the floor's, unchanged; 4,096 is the earliest row on file and keepers are present there. From update 4,097 this term is added to the loss. `x` and `y` are the batch's input and target tensors, `z1`, `ih`, `hh` as in `arch26.forward`:

```
# hp[:, t] is the state that step t reads: zeros at t = 0, else the state after step t-1.
# (arch26.forward returns the states AFTER each step; those are hp shifted by one.)
msk  = (x != X.BOS) & (y != X.BOS)        # a real input byte with a real target; no position 0, no padding
a    = lin(z1.detach(), ih, False)[..., 256:512] + lin(hp.detach(), hh, False)[..., 256:512]
C    = 4.59512                             # logit(0.99)
term = lam * (relu(C - a.abs()) ** 2)[msk].sum() / (msk.sum() * 256)
```

- The term is a hinge on the gate's pre-activation `a`, not on the gate value. Its pull on a gate falls in a straight line from the middle to the limit: at a gate of 0.983, which 40 letters need, it is still 12% of its peak. Revision 2's form, on the gate value, had fallen to 1/1047 there.
- The inputs are detached, so the gradient reaches only rows 256–511 of the two matrices and their biases.
- `lam` is set once per birth, at the frozen weights after update 4,096, from the 128 batches of updates 4,097 to 4,224 (no training is needed to compute it):
  - the task norm is the **mean over those 128 batches** of the task loss's gradient norm, **as the sweep computes it** (not with exact weights);
  - the term's norm at `lam = 1` is the mean over the same batches;
  - each is **one norm taken jointly** over rows 256–511 of `weight_ih_l0` and `weight_hh_l0` and the same entries of both biases;
  - `lam` is 0.25 (P4a), 0.05 (P4b) or 0.01 (P4c) times the task norm divided by the term's norm.
  One batch is too noisy for this: over those batches the task norm varies three- to five-fold. There are nine values, one per seed and strength, and each P4 receipt's `lam` must equal the registered one.
- **The top strength is close to a freeze.** By the reviewer's order-of-magnitude estimate, 0.25 pulls 27 to 48 times harder than what erodes the keepers and fixes the gate pattern as it stood just after update 4,096, when accuracy at 12 letters was 16–26%. It is kept as the top dose and carries the highest risk of harm. 0.01 is marginal, about 1.3 to 2.4 times.
- **Known risk.** The term pushes a gate towards whichever end it is nearer at each position. At update 4,096 about 78–81 of 256 units have gates on both sides of one half across letter positions (reviewer's count, `e32_r2_checks.py C`). Some units pushed to keep will keep everywhere and never load an operand. The measure "keeps and writes" below separates these.

## Code

One new runner, `exp32_births.py`, with the forward pass and all four additions (G, H, K, P4) written for the birth loop. Edit nothing existing; E31's files may be read, not imported. Receipts say `experiment="E32"` and go to `exp45/e32/`.

**The ceiling is not run by the new runner.** It is E28's runner, unchanged: `python3 exp28_everybyte.py dev B-all <seed> --out exp45/e32/ceiling`. Those three receipts stay in E28's format. Do not write a backprop path into `exp32_births.py`. Commit (`git add -f`) every final file and the row files of seed 73129 only; other row files are measured, their hashes recorded, and not committed.

**Tolerance, wherever "agree" is said below:** the largest absolute difference is at most 1e-5 of the tensor's largest absolute entry (E28's check 4 measured 3e-7 to 5e-7 between two forms).

## Checks before any full birth

1. **Floor reproduced.** With nothing added, a dry birth at 2,048 updates has the `tensors_sha256` of `exp45/e28_qual/dry/e28_dev_S-all_dry2048_73129.json`.
2. **Ceiling reproduced.** Each rerun ceiling birth has, at all six saved rows, the `tensors_sha256` that `exp45/e28/e28_dev_B-all_<seed>.json` records in its `curve`. Compare as each row is written, so a mismatch shows at the first row; on a mismatch, stop.
3. **Same data, same forward pass.** On the first batch every arm's task loss equals the floor's bit for bit, and each arm's `data_digest` equals E28's at every saved row.
4. **G and H.** On one batch the error sent back equals the pinned forms, and agrees with a second form written by hand.
5. **K.** The update identity holds to 1e-7 at updates 1, 10, 100. With `Bhh` set to the forward matrix by hand, K's gradients agree with H's. Every saved row records the cosine between `Bhh` and the forward matrix.
6. **P4.**
   - The `z = sigmoid(a)` computed for the term agrees with the forward pass's own update gate on one batch, at the brief's tolerance; report whether it is also identical. (The forward pass makes one recurrent product per step and the block one over all steps, so the last bits may differ. `hp` taken one step late gives differences near 1.)
   - The term's gradient is zero outside the update-gate rows, and agrees with a hand-written two-factor form.
   - With `lam = 0.0`, the term computed and added and not skipped, a birth of seed 73129 run to 8,192 updates has E28's row-8,192 hash:
     `git show $D:$M/exp45/e28/e28_dev_S-all_73129.json | python3 -c "import json,sys; print([(r['step'], r['tensors_sha256'][:12]) for r in json.load(sys.stdin)['curve']])"`
   - Every P4 birth's row at 4,096 has the hash of E28's row 4,096 for its seed.
7. **Second worker**, writing independently and without reading the first worker's code:
   - G, H, K: one dry birth each at 2,048 updates, seed 73129. Equal hashes, or else gradients that agree on one common batch when each runs the other's function.
   - P4: no training. Both workers compute `lam` and both norms for all nine seed-and-strength pairs from E28's row-4,096 files and batches 4,097 to 4,224, as defined above. The nine values must agree to 1e-5 relative. The main worker's values go into `exp45/e32/E32_PREDICTIONS.md`, pushed before any full birth.
   - A third reference, from the reviewer's numpy reader, for **batch 4,097 alone** (task norm as the sweep computes it, then the term's norm at `lam = 1`): seed 73129, 0.036740 and 0.890970; seed 93683, 0.077113 and 0.836491; seed 116089, 0.093320 and 0.849241. Both workers report their single-batch values beside these; agreement to about 1e-3 relative is expected, and a larger difference is reported before any birth.
   - Disagreement on anything: both stop and report.
8. **Cost.** CPU time of the dry births and a forecast for the twenty-one.

## Reading the files

A new script `e32_read.py`. It takes gate values from `e29_gates.gates(m, x)` and scoring from `exp16_integrated.score`, and loads only the network's own tensors.

- **Panels.** `exp11_cell.panel(t, held, seed, ks=(12, 40), n=300)` for 12 and 40 letters, exactly E30's call, and a separate call `ks=(20,)` for 20 letters. The floor's and ceiling's 20-letter figures come from that same call, not from the receipts.
- **First**, reproduce the floor and ceiling rows of `exp45/e30/E30_UNSEEN.json` exactly, and the keeper counts of the floor table in "Why" above within one unit (a unit on the 0.5 line can fall either way between a float32 and a double-precision reader). The bars below are the stated numbers whatever the reader gives.
- **Accuracy** at 12, 20 and 40 letters at every saved row.
- **Keeper across k letters:** over the steps whose input byte is one of the k letters, the product of the unit's update gate is at least 0.5, in at least half the items.
- **Keeps and writes:** a keeper that also has its update gate at or below 0.5 at some step before the first letter (after position 0), in at least half the items.
- Counts of both at k = 12, 20 and 40, and the share of letter-step gate values above 0.944 and above 0.983, at every saved row. For P4, also both gradient norms at every row.
- **U** as E30 defines it.

## Endpoints and marks (fixed)

Every figure is the **mean of the last three saved rows** (16,384, 20,480, 24,576), for arms, floor and ceiling alike.

- **Primary:** accuracy at 40 letters. **Secondary:** accuracy at 20 letters.
- **Moves** (at a distance): at least 30 points above the same seed's floor in at least two of three seeds.
- **Partial:** not "moves", and at least 15 points above the floor in at least two of three seeds.
- **Closes:** within 15 points of the same seed's ceiling in at least two of three seeds.
- **Harm:** accuracy at 12 letters at least 15 points below the same seed's floor in at least two of three seeds.
- **Has keepers** (at a distance): the keeps-and-writes count at that distance is at least 25 / 30 / 31 (seeds 73129 / 93683 / 116089), in at least two of three seeds. The bar is half the floor's count of *keepers* at 12 letters at row 4,096 (50 / 59 / 62), applied to the *keeps-and-writes* count, and is the same at every distance. That is meant: at row 4,096 the floor's keeps-and-writes counts are 40 / 46 / 46 at 12 letters, 33 / 31 / 43 at 20 and 18 / 0 / 36 at 40, so at 40 letters the bar asks for more than exists then in two seeds. The backprop births' final files clear it at every distance (67 / 85 / 79 at 12, 46 / 56 / 63 at 40).
- Differences under 15 points are not read.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P32.1 | H moves accuracy at 40 letters | 45% |
| P32.2 | G moves accuracy at 40 letters | 30% |
| P32.3 | K moves accuracy at 40 letters | 30% |
| P32.4 | K is within 15 points of H at 40 letters in at least two of three seeds | 45% |
| P32.5 | H has keepers at 12 letters | 60% |
| P32.6a | P4a has keepers at 12 letters | 55% |
| P32.6b | the same for P4b | 40% |
| P32.6c | the same for P4c | 20% |
| P32.7 | at least one P4 strength moves accuracy at 20 letters | 30% |
| P32.8 | at least one P4 strength moves accuracy at 40 letters | 20% |
| P32.9 | no arm does harm | 65% |
| P32.10 | the rerun ceiling keeps at least 50 keepers at 12 letters at every saved row from 8,192 on, in all three seeds | 60% |

## Readings (fixed)

**Bounds and K.** Read at 40 letters and again at 20 letters, each on its own.

| H | K | Reading at that distance |
|---|---|---|
| moves | moves | Exact credit across the recurrent map is enough to move it, and a learned pathway that starts random recovers it. Learned feedback goes forward on fresh seeds. |
| moves | partial or neither | Exact credit across the recurrent map was needed before `Bhh` had converged; a pathway that only converges later does not recover it. K's keeper counts by row say when. |
| partial | any | Reported as a partial gain. No statement about where the loss arises. |
| neither | moves | Not expected, since K converges to H. Reported as an anomaly and rerun. |
| neither | partial or neither | Exact credit across the recurrent map does not move accuracy at this distance. Nothing wider is concluded. |

G is reported beside H. If H moves and G does not, the recurrent map's candidate rows are named as what G lacks; if G moves and H does not, the input map's gate rows are named as what G adds.

**Keepers and accuracy, for G, H and K.** For each arm and distance the report states whether it has keepers. An arm whose accuracy moves without keepers counts against the account that keepers are what carries the operand past the trained range; an arm that has keepers without accuracy moving shows that these keepers are not enough at that distance. Either is reported in those words and nothing wider.

**The push, per strength, at 20 and at 40 letters.** Each distance is read against the keeps-and-writes count at that distance. The table is not read at 12 letters: "moves" there would need 90 to 104%, which is at or above the ceiling. Twelve letters is covered by the harm mark.

| Has keepers at that distance | Accuracy at that distance | Reading |
|---|---|---|
| no | partial or neither | The arm does not have keepers at this distance at this strength, and says nothing about them there. |
| no | moves | Accuracy moved without keepers at this distance. This counts against the account that keepers carry the operand. |
| yes | moves | Keepers held by a local push go with accuracy at this distance. Run again on fresh seeds before a claim. |
| yes | partial | Reported as a partial gain. |
| yes | neither | These keepers, held from update 4,096, do not give accuracy at this distance. |


**Wording.** "Is enough" is said only of an arm that closes. "Moves" without "closes" is a gain that leaves most of the gap. An arm that moves 40 letters and harms 12 is a trade.

## Done

Runner, reader, check receipts, both workers' check records, twenty-one complete births, and `E32_REPORT.md` with one table per seed (arm; accuracy at 12, 20, 40; keepers and keeps-and-writes at 12, 20, 40; gate shares; U) for every saved row, beside floor and ceiling, and the outcome of each prediction. No readings beyond the tables above.

## What changed from revision 3 (the third audit's six changes)

The reviewer's script is `e32_r3_checks.py` on `claude/vibrant-cerf-sje726` at f5e876d0. I made these changes as the audit wrote them and reran none of its measurements.

| # | Change | Where |
|---|---|---|
| 1 | The term's gate "agrees" with the forward pass's, not "bit for bit" | check 6 |
| 2 | Strength set on the mean task norm over batches 4,097 to 4,224, as the sweep computes it, one joint norm; the receipt's `lam` equals the registered one; `lam = 0.0` computed and added | arms P4, checks 6 and 7 |
| 3 | Ceiling run by E28's runner with `--out`, receipts in E28's format, all six row hashes compared | code, check 2 |
| 4 | Push table read at 20 and 40 letters only | readings |
| 5 | "Retains keepers" becomes "has keepers"; what the bar is half of is stated; the "no" row reworded | marks, readings |
| 6 | K's window not confined to 4,096–12,288; "these keepers, held from update 4,096"; a sentence for accuracy without keepers and the reverse; reader reproduces the floor table within one unit | arm K, readings, reader |

Also added from the audit: the reviewer's single-batch norms as a third reference, and its estimate that the top strength is close to a freeze.

## What changed from revision 2 to revision 3, and why

The reviewer's second audit. "Rerun" means I ran the command and got what the audit reported.

| Finding | Evidence | Change |
|---|---|---|
| The floor has keepers at 4,096 and loses them | `e32_r2_checks.py B` (rerun; table in "Why") | question reframed as keeping, not making; P32.5 and P32.6 are retention marks above half the starting count |
| `hp` undefined; `arch26.forward` returns states after each step | `git show $D:$M/arch26.py \| sed -n '50,75p'` (read) | `hp` defined; check that the term's gate equals the forward pass's |
| `lin` takes three arguments | `git show $D:$M/exp3_state.py \| sed -n '107,108p'` (read) | third argument pinned |
| Mask included one position per padded item with no target | reviewer's toy batch (not rerun) | `(x != BOS) & (y != BOS)` |
| Push on the gate value fades long before the limit | `python3 -c "f=lambda z:2*(0.49-abs(z-0.5))*z*(1-z); print([round(f(0.5)/f(z)) for z in (0.9,0.944,0.966,0.983)])"` gives 15, 50, 155, 1047 (rerun) | hinge on the pre-activation; gate shares above 0.944 and 0.983 recorded |
| Keepers that never write | `e32_r2_checks.py C` (not rerun) | "keeps and writes" counted and used in the marks |
| Keeper positions, panel calls and "ends with" not pinned | E30's call is `ks=(12, 40)` | pinned; 20 letters by its own call |
| "Closes" not computable: ceiling rows not on file | `git ls-tree -r --name-only $D $M/exp45/e28/ \| grep 'B-all.*\.pt$'` shows final files only (rerun) | ceiling rerun with rows, hash-checked against E28 |
| "Partial" left a hole; bounds table read 40 letters only | — | partial redefined; no "not where" reading on a partial; table read at both distances |
| G and H differ in two ways | `git show origin/claude/sleepy-knuth-q96dd1:$M/arch31.py \| sed -n '7,9p'` (rerun) | both named |
| Push table concluded about 20 and 40 from keepers at 12 | S-bal on file | each distance read against its own count |
| Two-of-three against all-three | — | two of three throughout |
| K's window | 0.99975^n | named as 4,096 to 12,288 |
| `lam = 0` check never reached the switch-on | — | run to 8,192 against E28's row hash |
| Hashes cannot settle two implementations of the push | `lam` comes from two norms | compared without training, nine values |
| "Within 1e-6" undefined | E28 check 4 | tolerance defined once |
