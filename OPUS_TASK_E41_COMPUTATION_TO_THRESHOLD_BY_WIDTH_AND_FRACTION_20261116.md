# E41: computation to threshold on the same task, at three widths and three taught fractions (16 November 2026, revision 2)

Evidence level E2; a first measurement, with no reading fixed beforehand beyond the description below. The quantity and the design are the outside reviewer's (its memo, revision 8, section 21, on `claude/vibrant-cerf-sje726` at 557e3749), written from the user's statements of 15 November. This specification and the predictions are mine. Revision 1 (213c38f4) was audited by the reviewer (`AUDIT_E41_COMPUTATION_TO_THRESHOLD_20261116.md` on `claude/vibrant-cerf-sje726` at 43a89440), which required five changes and stated that after them no re-audit is needed. Revision 2 makes them and the recommendations; the last section lists them. **Released.** Three workers, one seed each.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106`. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit. Edit nothing existing.

## The goal this serves

The user, 15 November: if the network's computation grows, or the data grow, but the number of training steps needed to reach a threshold falls, with the same or a better result and the same or less computation on average, that is a goal worth discussing; which axis to scale is for experiment.

**Computation to threshold** (the memo's section 21.2): for a run, the total computation spent up to the first scored row at which its accuracy on never-taught pairs reaches a threshold. A learner or a setting is better if it reaches the same threshold with the same or less computation, whatever it spent that computation on.

## What is on file

- For the present GRU there is no measurement at any width but one. (On the earlier window network width helped: review 9.7 and 9.18.)
- **The present sizes are two numbers, not one.** `exp11_cell` sets the feedforward width to 512 (`git show $D:$M/exp11_cell.py | sed -n '28p'`) and `arch20` sets the GRU to 256 units (`git show $D:$M/arch20.py | sed -n '32p;43,50p'`): 1,124,113 parameters, the embedding table included.
- At these sizes, held-out add and sub at 12 letters, mean of the two, by updates (my read of the receipts):

  | Taught fraction | Rule | 4,096 | 8,192 | Source |
  |---|---|---|---|---|
  | 0.50 | sweep, three seeds | 69.7, 73.3, 76.5 | 89.2, 91.8, 92.0 | `exp45/e26/e26_dev_S_<seed>.json` |
  | 0.50 | backpropagation, three seeds | 62.7, 84.3, 79.2 | 88.0, 94.8, 88.8 | `exp45/e26/e26_dev_B_<seed>.json` |
  | 0.25 | backpropagation, 73129 | 76.7 | 87.7 | E35-A's arm `G` (`claude/nice-rubin-y5de8k`) |
  | 0.10 | backpropagation, 73129 | 14.7 | 17.0 | the same; it memorises and stays between 15 and 21 to 24,576 |

  So at the present sizes the 80% threshold is passed at or before 8,192 updates at fractions 0.50 and 0.25 (one birth is already at 84.3 by 4,096), and never at 0.10. 90% is reached by 8,192 in three of the six births at f = 0.50 and by 12,288 in five (the audit's count).

## The grid: 54 runs from random weights

| Axis | Values |
|---|---|
| Width (feedforward H, GRU units U) | half (256, 128): 317,457 parameters; present (512, 256): 1,124,113; double (1,024, 512): 4,211,985 |
| Taught fraction f | 0.10, 0.25, 0.50 |
| Rule | the sweep (`local=True`); backpropagation (`local=False`) |
| Seed | 73129, 93683, 116089 |

Everything else is E26's birth: the four mod-89 operations, 0 to 12 letters, answer bytes as targets, batch 64, `exp26_candidate.step`'s update (lines 84-91), learning rate 0.001, decay 0.25, betas (0.9, 0.98), the embedding table of width 8 without decay.

- **Dose:** 8,192 updates at the double width, 12,288 at the present width, 40,960 at the half width. The half and the present width then receive nearly the same computation (1.30 and 1.38 × 10^10 weight-updates) and the double width 3.45 × 10^10.
- **Data:** `exp11_cell.batches([(t, exp11_cell.split(t, f, seed)[0]) for t in exp11_cell.DEV], seed, 12)`. At f = 0.50 this is E26's own stream. The batches depend on the seed and the fraction only, not on width or rule.

## The network at other widths, as code

A new file `exp41_scale.py` with a class that has `arch20.NetRNN("gru")`'s parameters under the same names, created in the same order, with the two sizes as arguments:

```
class NetW(nn.Module):
    def __init__(self, H, U):
        super().__init__()
        self.kind, self.units = "gru", U
        self.book = nn.Embedding(258, 8, padding_idx=257)     # BOS = 257, as arch20.py line 43
        self.l1  = nn.Linear(8, H)
        self.rnn = nn.GRU(H, U, num_layers=1, batch_first=True)
        self.lm  = nn.Linear(U, H)
        self.l2  = nn.Linear(H, H)
        self.l3  = nn.Linear(H, 257)

def new_net(seed, H, U):
    T.manual_seed(seed); return NetW(H, U)
```

`arch26.forward(m, x, local, True)` runs on it unchanged, since it takes its sizes from the module. Import `exp26_candidate` before any other harness module and build the optimizer with `exp26_candidate.make_opt(m)`, which asserts the two groups by parameter name at any width: `adamw16` exempts the embedding from decay only when `E16_BOOKDECAY` is `off`, and that variable is set by `exp20_reference` on import, not by default. Scoring uses `exp11_cell.score` after `arch26.install()`.

## Scores, every 256 updates

1. **Held-out panel at 12 letters:** `exp11_cell.panel(t, exp11_cell.split(t, f, seed)[1], seed, ks=(12,), n=300)` for each of the four tasks. **Accuracy A** is the mean of add and sub.
2. **Held-out reduction pairs, no letters:** for add and sub, the held-out pairs with a + b ≥ 89, or a − b < 0, rendered with k = 0. To bound the cost of scoring, a fixed sample of 1,000 per task, drawn once by `Random(f"e41|{seed}|{round(f * 100)}|{t[0]}").sample(sorted(pairs), 1000)`, where `t` is the task's tuple in `exp11_cell.DEV` (so `t[0]` is `"add"` or `"sub"`) and `pairs` is the list of `(a, b)` tuples of that task's held-out reduction pairs. **Accuracy R** is the mean of add and sub. It is the one free of what a network can answer without the task's modulus.
3. **Taught panel at 12 letters:** the panel call on the taught pairs with `seed + 1`. The taught accuracy tabled is the mean of add and sub, as for A.

A fresh network scores 0 on all three before any update, so no starting level has to be subtracted here; row 0 is scored to show it.

## What is recorded and tabled

For each run and each of A and R, at thresholds 50, 80 and 90:

- the first scored row at which the accuracy is at or above the threshold, or "not reached";
- the computation to that row in two units: **weight-items**, the parameter count times 64 times the row's updates; and **CPU-seconds** of updates up to that row (`time.process_time()`, scoring excluded, one thread);
- the taught accuracy at that row;
- for every run, its highest A and R and their rows.

Two versions of the row at which a threshold is reached are tabled. *Single-row:* the first scored row at or above the threshold. *Three-row:* the first row of the first three consecutive scored rows that are all at or above it. A run's last two rows cannot begin three such rows, so under the three-row version a threshold can be recorded up to the dose less 512 updates. The description below and predictions P41.2 to P41.8 use the three-row version. The single-row version is tabled beside it, and a cell in which the two versions give different descriptions is marked.

Also tabled, since a highest value grows with the number of rows scored and the half width has 160 rows against 32: the mean of A and of R over rows 7,680, 7,936 and 8,192, which every width has.

Receipts go to `exp45/e41/e41_<rule>_h<H>_f<10|25|50>_<seed>.json`, in E26's format with the scores per row; the runner refuses an existing path. Each receipt records `tensors_sha256` and `weights_sha256` of the final weights, as E26's receipts do, and the sha256 of the runner. Commit the final weights of the half and present widths, and of the double width for seed 73129 only (the 54 files together are 407 MB). Commit the script that computes the thresholds and descriptions from the receipts.

**One runner.** The worker of seed 73129 writes `exp41_scale.py` and pushes it with its checks; the other two take it at that commit and run the checks on their own hosts.

## Checks before the full runs

1. **The present width is E26.** `new_net(seed, 512, 256)` has a state dict bit-equal to `arch26.new_net(seed)`'s for the three seeds. At f = 0.50, seed 73129, 2,048 updates give the `tensors_sha256` of `exp45/e26_qual/e26_dev_S_dry2048_73129.json` (79618ab2…) under the sweep and of `e26_dev_B_dry2048_73129.json` (adf933d7…) under backpropagation. Every worker runs this check, whatever its own seed.
2. **Same data at every width and rule.** The batch digest (`exp26_candidate.Digest`) at row 4,096 is equal across the six runs of a seed and fraction. At f = 0.50 it equals `curve[0].data_digest.sha256` of `exp45/e26/e26_dev_S_<seed>.json`: bbd71ab7ca27b72e… for 73129, 65b1c7e06b17459a… for 93683, 1df203fb8f299dc0… for 116089. In check 1's two dry runs the digest at row 2,048 is 4f622d94e0156993….
3. **Parameter counts** are 317,457, 1,124,113 and 4,211,985.
4. **The forward at other widths.** At the half and the double width, on the first batch, the logits equal those of `torch.nn.GRU` run on the same parameters to 1e-6, and under `local=False` the parameter gradients equal autograd's through that reference to 1e-5 of each tensor's largest entry; under `local=True` they differ.
5. **No held-out pair is presented** (asserted on every batch), and the reduction sample is disjoint from the taught pairs.
6. **Cost.** A dry run of 256 updates and one scored row at each width, with CPU-seconds per update and per scored row, and a forecast for the worker's 18 runs. The reviewer's estimate for all 54 is about 30 CPU-hours of updates and 5 of scoring, a little over half of it at the double width. If a worker's forecast exceeds 15 CPU-hours, report before starting. Run the dry run twice at the half and the double width; the two `tensors_sha256` must be equal. Take CPU-seconds per update from these dry runs on an otherwise idle host (three processes at once share the memory cache and the double width's matrices do not fit in it), and record how many processes ran beside each full run.
7. **The scorer against the reviewer's reader.** (a) For the worker's seed and each fraction, the sha256 of `json.dumps([list(p) for p in sample])` for the add and the sub sample begins with the digits in the first table. (b) The run's own scoring code, given f = 0.50 and the file's seed, applied to the six E26 births `exp45/e26/e26_dev_<S|B>_<seed>.pt`, each loaded into `NetW(512, 256)` with `load_state_dict(T.load(file))`, gives the counts in the second table. A count that differs by one or two items is reported and the work proceeds; a larger difference is a failure.

   | Seed | f = 0.10 (add, sub) | f = 0.25 (add, sub) | f = 0.50 (add, sub) |
   |---|---|---|---|
   | 73129 | 2d9d251a354dc291, 92c29b4a374be126 | 3f31c9a15729adc3, 89a86290d78daddd | f433a55ca52ae34d, 18ac75b5013b6d47 |
   | 93683 | 7a5f67e2fc00ed81, c316f9e9ca56339e | 81d35edfe17f2b54, 19b61154e17286f4 | 368844acc03dfd7a, 83274756387907f7 |
   | 116089 | 49965d4fbf5633ee, 3efe08c20a9e5ad3 | bef6158428f9008a, 0fdd40508ad49609 | 26b2eff70e51e033, 8ccb9c3177772cbf |

   | File | A: add, sub (of 300) | A | R: add, sub (of 1,000) | R | Taught: add, sub (of 300) | Taught |
   |---|---|---|---|---|---|---|
   | `e26_dev_S_73129.pt` | 295, 293 | 98.00 | 962, 955 | 95.85 | 298, 298 | 99.33 |
   | `e26_dev_S_93683.pt` | 296, 293 | 98.17 | 988, 979 | 98.35 | 297, 298 | 99.17 |
   | `e26_dev_S_116089.pt` | 300, 299 | 99.83 | 993, 996 | 99.45 | 300, 296 | 99.33 |
   | `e26_dev_B_73129.pt` | 295, 296 | 98.50 | 976, 987 | 98.15 | 295, 297 | 98.67 |
   | `e26_dev_B_93683.pt` | 298, 294 | 98.67 | 970, 968 | 96.90 | 298, 298 | 99.33 |
   | `e26_dev_B_116089.pt` | 284, 279 | 93.83 | 955, 895 | 92.50 | 285, 288 | 95.50 |

   As a fixed point for (a): the add sample at f = 0.50, seed 73129, begins (30, 64), (84, 37), (64, 49), (46, 54).
8. **The worker's own seed, at no cost (recommended by the audit).** At the present width and f = 0.50 a run is E26's birth, so scoring E26's own panel (`ks=(0, 4, 8, 12, 20)`, n = 300) at rows 4,096 and 8,192 reproduces the `sealed` counts in `exp45/e26/e26_dev_<S|B>_<seed>.json` exactly. For seed 73129, at f = 0.25 and 0.10 the digest at row 4,096 should be a71b1ca10b24a36b… and e120924e5af546f9… (E35-A's arm `G`, not merged: a difference is reported, not a failure).

## How the result is stated (no marks; a description fixed beforehand)

The description is given for updates to threshold and for computation to threshold in weight-items, on the 80% threshold of accuracy R and of accuracy A separately, for each rule and fraction and each step from a width to the next wider one. The same is given for 50 and 90 in a second table. The CPU-second figures are given beside the weight-items.

For each seed, let x_n and x_w be the narrower and the wider run's updates to the threshold, and D_n and D_w the largest count at which each could have been recorded (the dose under the single-row version, the dose less 512 under the three-row version).

| Narrower run | Wider run | Description for that seed |
|---|---|---|
| reached | reached | *falls* if x_w / x_n is below 2/3; *rises* if above 1.5; otherwise *stays* |
| reached | not reached | *rises* if D_w / x_n is 1.5 or more; otherwise *not decided* |
| not reached | reached | *falls* if x_w / D_n is 2/3 or less; otherwise *not decided* |
| not reached | not reached | *neither reaches* |

For computation in weight-items the same table is used with every count multiplied by its width's parameter count (317,457, 1,124,113 or 4,211,985). A cell's description is the one that at least two of its three seeds share; otherwise the cell is *mixed*. The three seeds' values are given in every cell.

At f = 0.10 expect "neither reaches" in most cells; the information there is in the levels.

A doubling of both sizes multiplies the weights by about 3.7, so computation in weight-items stays level only if the updates to threshold fall by that factor.

## What it cannot show

- **Settings.** Learning rate and decay were chosen at one width and are used at all three, so a width may be handicapped by them.
- **Scoring every 256 updates** bounds how finely a threshold is located. A threshold reached before about 1,500 updates is located to no better than one part in six, so a description there can turn on one row. The three-row version records a crossing later than a smooth curve would, by an amount not measured here.
- **Experience.** Networks with a life behind them exist at one width only, so whether E37's saving shrinks as the network widens is left to the small world.
- **One task family**, from random weights, with answer bytes as targets.
- **CPU-seconds** depend on the host and on one thread; the weight-items do not, but they ignore that a wider layer may be computed more efficiently per weight.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P41.1 | Check 1 passes under both rules | 90% |
| P41.2 | At f = 0.50 the double width reaches 80 on A in fewer updates than the present width, in two of three seeds, under backpropagation | 65% |
| P41.3 | At f = 0.50 the double width reaches 80 on A with fewer weight-items than the present width, in two of three seeds, under backpropagation | 15% |
| P41.4 | At f = 0.50 the half width reaches 80 on A with fewer weight-items than the present width, in two of three seeds, under backpropagation | 45% |
| P41.5 | At f = 0.10 the double width reaches 80 on R within its dose, in at least one seed under either rule | 15% |
| P41.6 | At f = 0.25 the half width does not reach 80 on R within 16,384 updates, in two of three seeds under backpropagation | 55% |
| P41.7 | The description (falls, stays, rises) of weight-items to 80 on A differs between the sweep and backpropagation in at least one of the six width steps by fraction | 35% |
| P41.8 | At f = 0.50 and the present width, R reaches 80 at the same scored row as A or within 512 updates of it, in two of three seeds under each rule | 60% |

## Done

`exp41_scale.py`, the check receipts, the worker's 18 runs with receipts and final weights, and `E41_REPORT_<seed>.md` with the worker's tables. The three-seed tables, the description by rule and fraction, and the predictions are assembled by the worker of seed 73129 once all three branches are complete. No readings beyond the description above.

## What changed from revision 1 (the audit)

Made as the audit wrote them. I reran none of its figures; its seventh check has each worker reproduce its tables.

| | Change | Required? |
|---|---|---|
| 1 | Doses: 12,288 at the present width and 40,960 at the half width. The memo's doses, which I carried, were too short for the three-row threshold and for deciding a comparison when the half width does not reach it. | yes |
| 2 | The three-row version carries the description and the predictions; a table of four cases says what is stated when a run does not reach a threshold; the boundaries 2/3 and 1.5 belong to "stays". My "not reached counted as above the dose" gave no ratio. | yes |
| 3 | The sample's generator string fixed exactly | yes |
| 4 | Check 1 run by every worker; check 2 at row 4,096, where all three seeds have a digest on file. My check 2 could not be computed for two of the three seeds. | yes |
| 5 | Check 7: the new scoring code against the reviewer's reader | yes |
| — | `make_opt` and the import order; one runner; check 8; levels over rows 7,680 to 8,192; CPU-seconds from idle dry runs; which weights are committed; two limits; two wordings | recommended |
