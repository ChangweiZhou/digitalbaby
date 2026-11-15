# E41: computation to threshold on the same task, at three widths and three taught fractions (16 November 2026, revision 1)

Evidence level E2; a first measurement, with no reading fixed beforehand beyond the description below. The quantity and the design are the outside reviewer's (its memo, revision 8, section 21, on `claude/vibrant-cerf-sje726` at 557e3749), written from the user's statements of 15 November. This specification and the predictions are mine. **Not released: it waits for the reviewer's audit.** Three workers, one seed each.

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
  | 0.10 | backpropagation, 73129 | 14.7 | 17.0 | the same; it memorises and stays near 17 to 24,576 |

  So at the present sizes the 80% threshold is passed between 4,096 and 8,192 updates at fractions 0.50 and 0.25, and never at 0.10. The dose below covers that.

## The grid: 54 runs from random weights

| Axis | Values |
|---|---|
| Width (feedforward H, GRU units U) | half (256, 128): 317,457 parameters; present (512, 256): 1,124,113; double (1,024, 512): 4,211,985 |
| Taught fraction f | 0.10, 0.25, 0.50 |
| Rule | the sweep (`local=True`); backpropagation (`local=False`) |
| Seed | 73129, 93683, 116089 |

Everything else is E26's birth: the four mod-89 operations, 0 to 12 letters, answer bytes as targets, batch 64, `exp26_candidate.step`'s update (lines 84-91), learning rate 0.001, decay 0.25, betas (0.9, 0.98), the embedding table of width 8 without decay.

- **Dose:** 8,192 updates; 16,384 at the half width.
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

`arch26.forward(m, x, local, True)` runs on it unchanged, since it takes its sizes from the module. The optimizer is `exp16_integrated.adamw16(m, 0.001, 0.25, ())`, with its groups read back and recorded. Scoring uses `exp11_cell.score` after `arch26.install()`.

## Scores, every 256 updates

1. **Held-out panel at 12 letters:** `exp11_cell.panel(t, exp11_cell.split(t, f, seed)[1], seed, ks=(12,), n=300)` for each of the four tasks. **Accuracy A** is the mean of add and sub.
2. **Held-out reduction pairs, no letters:** for add and sub, the held-out pairs with a + b ≥ 89, or a − b < 0, rendered with k = 0. To bound the cost of scoring, a fixed sample of 1,000 per task, drawn once by `Random(f"e41|{seed}|{f}|{op}").sample(sorted(pairs), 1000)`. **Accuracy R** is the mean of add and sub. It is the one free of what a network can answer without the task's modulus.
3. **Taught panel at 12 letters:** the panel call on the taught pairs with `seed + 1`.

A fresh network scores 0 on all three before any update, so no starting level has to be subtracted here; row 0 is scored to show it.

## What is recorded and tabled

For each run and each of A and R, at thresholds 50, 80 and 90:

- the first scored row at which the accuracy is at or above the threshold, or "not reached";
- the computation to that row in two units: **weight-items**, the parameter count times 64 times the row's updates; and **CPU-seconds** of updates up to that row (`time.process_time()`, scoring excluded, one thread);
- the taught accuracy at that row;
- for every run, its highest A and R and their rows.

Because single rows swing by ten points or more (the table above), a second version is tabled beside the first: the first row from which the accuracy stays at or above the threshold for three consecutive scored rows.

Receipts go to `exp45/e41/e41_<rule>_h<H>_f<10|25|50>_<seed>.json`, in E26's format with the scores per row; the runner refuses an existing path. Commit each run's final weights (54 files; the double width is about 17 MB each) and record the sha256 of nothing else.

## Checks before the full runs

1. **The present width is E26.** `new_net(seed, 512, 256)` has a state dict bit-equal to `arch26.new_net(seed)`'s for the three seeds. At f = 0.50, seed 73129, 2,048 updates give the `tensors_sha256` of `exp45/e26_qual/e26_dev_S_dry2048_73129.json` (79618ab2…) under the sweep and of `e26_dev_B_dry2048_73129.json` (adf933d7…) under backpropagation.
2. **Same data at every width and rule.** The batch digest (`exp26_candidate.Digest`) at row 2,048 is equal across the six runs of a seed and fraction; at f = 0.50 it equals E26's at that row.
3. **Parameter counts** are 317,457, 1,124,113 and 4,211,985.
4. **The forward at other widths.** At the half and the double width, on the first batch, the logits equal those of `torch.nn.GRU` run on the same parameters to 1e-6, and under `local=False` the parameter gradients equal autograd's through that reference to 1e-5 of each tensor's largest entry; under `local=True` they differ.
5. **No held-out pair is presented** (asserted on every batch), and the reduction sample is disjoint from the taught pairs.
6. **Cost.** A dry run of 256 updates and one scored row at each width, with CPU-seconds per update and per scored row, and a forecast for the worker's 18 runs. The reviewer's estimate for all 54 is about 24 CPU-hours of updates and 4 of scoring, three quarters of it at the double width. If a worker's forecast exceeds 12 CPU-hours, report before starting.

## How the result is stated (no marks; a description fixed beforehand)

For each rule and fraction, from each width to the next, on the 80% threshold of accuracy R and of accuracy A separately:

- **updates to threshold:** falls, stays (within a factor of 1.5), or rises;
- **computation to threshold in weight-items:** falls, stays (within a factor of 1.5), or rises;
- with the three seeds' values given, and "not reached" counted as above the dose.

The same for 50 and 90 in a second table. The CPU-second figures are given beside the weight-items. A cell is described only if at least two of its three seeds agree; otherwise it is "mixed".

A doubling of both sizes multiplies the weights by about 3.7, so computation in weight-items stays level only if the updates to threshold fall by that factor.

## What it cannot show

- **Settings.** Learning rate and decay were chosen at one width and are used at all three, so a width may be handicapped by them.
- **Scoring every 256 updates** bounds how finely a threshold is located.
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
| P41.6 | At f = 0.25 the half width does not reach 80 on R within its dose, in two of three seeds under backpropagation | 55% |
| P41.7 | The description (falls, stays, rises) of weight-items to 80 on A differs between the sweep and backpropagation in at least one of the six width steps by fraction | 35% |
| P41.8 | At f = 0.50 and the present width, R reaches 80 at the same scored row as A or within 512 updates of it, in two of three seeds under each rule | 60% |

## Done

`exp41_scale.py`, the check receipts, the worker's 18 runs with receipts and final weights, and `E41_REPORT_<seed>.md` with the worker's tables. The three-seed tables, the description by rule and fraction, and the predictions are assembled by the worker of seed 73129 once all three branches are complete. No readings beyond the description above.
