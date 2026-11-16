# E42: cells with a limited number of local connections and their own periods (16 November 2026, revision 1)

Evidence level E2; a first measurement. The design is the outside reviewer's (its memo, revision 10, section 22, on `claude/vibrant-cerf-sje726` at f68f2b7d), written from the user's statements of 16 November. This specification, the specification file and the predictions are mine. **Not released: it waits for the reviewer's audit.** Three workers, one seed each. It does not wait for the small world and displaces nothing released.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106; K=PROJECT_DOCUMENTS/COORD_CHECKS_20261116`. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit. Edit nothing existing.

## The idea (the user's, 16 November)

- If two cells have no connection, in the past or now, no weight can connect one with what the other remembers of distant history. So the apparent overfitting of weights and the making of connections during learning are to some extent the same thing. Can this be used?
- Under the local-learning hypothesis a cell can interact only with a pre-specified number of others. Let each cell connect to at most N = 10 others to start with (N can be adjusted), learn the strengths, and adjust which cells it connects to, by any number at a time, **inside a neighbourhood defined in advance** by coordinates, to impose locality.
- Each cell has its own pre-set period. Its previous value is available to it; during its period it absorbs what arrives from the cells that send to it; how it processes that is not fixed in advance and variants may be tried.
- The encoder and decoder parts stay as they are.

## What changes in the network, and what does not

Everything outside the recurrent layer is the present network (feedforward width 512, 256 recurrent units, 1,124,113 parameters), unchanged: the byte table, `l1`, the input map into the layer, and the readout `l2`, `lm`, `l3`. The layer keeps its 256 units, now called cells, and the GRU's equations. Three things change: which cells a cell reads, when a cell is updated, and how it combines what arrived since its last update.

**`$K/e42_cells_spec.py` on the default branch is the specification** of the layer: its geometry and draws, the forward, the masked sign rule and the replacement of connections. Copy it to `$M/e42_cells.py` and import it; do not rewrite it. Its self-checks run without training (`python3 -I $K/e42_cells_spec.py`; my output is `$K/OUTPUT_E42_SPEC_MAC.txt`). What follows states its rules in words, so that the reviewer's numpy reader can be written from the text.

### Geometry, connections, periods

- **Positions.** Cell i (0 to 255) stands at row i // 16, column i % 16 of a 16 by 16 grid whose opposite edges are joined.
- **Neighbourhood B_i.** The 48 cells j ≠ i whose row and whose column each differ from i's by at most 3, with wrap-around.
- **Connections A_i.** N cells of B_i. A cell always has exactly N; a connection given up is always replaced. Cell i also reads its own previous value, which is not counted in N.
- **Period p_i and phase f_i.** Cell i is updated at positions t = f_i, f_i + p_i, f_i + 2 p_i, … and keeps its value elsewhere. Position 0 is the start symbol. Every item starts from zero state.
- **Draws** use `random.Random` seeded with strings and never torch's generator, so the initial weights and the data are E26's:
  - periods: `order = list(range(256)); Random(f"e42|periods|{seed}").shuffle(order)`; the first cells of `order` take period 1, the next period 2, and so on, in the counts of the table below;
  - phases: `rp = Random(f"e42|phases|{seed}")`; for i in index order, `f_i = rp.randrange(p_i)`;
  - connections: `rc = Random(f"e42|connections|{seed}")`; for i in index order, `A_i = sorted(rc.sample(sorted(B_i), N))`.

| Period set | Cells of period 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 1, 2, 4 | 128 | 64 | 64 | | |
| 1, 2, 4, 8 | 128 | 64 | 32 | 32 | |
| 1, 2, 4, 8, 16 | 128 | 64 | 32 | 16 | 16 |

### The forward

With `Wh = m.rnn.weight_hh_l0` (768 × 256: rows r, z, n for each cell; the column is the source cell) and `mask[i, j]` true for j in A_i and for j = i, the same for the three gates:

```
gi(t) = Wi z1(t) + bi                      the input map, unchanged: every cell reads the whole current byte
gh(t) = (Wh * mask) h(t-1) + bh
at a position where cell i is updated, with G and H made from gi and gh over its period (table below):
    r = s(G_r + H_r),  z = s(G_z + H_z),  n = tanh(G_n + r * H_n),  h_i(t) = (1 - z) n + z h_i(t-1)
elsewhere h_i(t) = h_i(t-1);   h(-1) = 0
readout, unchanged:  z2 = tanh(l2 z1 + lm h),  logits = l3 z2
```

For a cell of period p updated at position t, the steps of its period are t − k for k = 0 … p − 1. **A step before the start symbol (t − k < 0) contributes nothing**, in every variant.

| Variant | G | H | Added parameters |
|---|---|---|---|
| `last` | gi(t) | gh(t) | none |
| `mean` | (1/p) Σ_k gi(t − k) | (1/p) Σ_k gh(t − k) | none |
| `step` (the default) | Σ_k a_i[k] gi(t − k) | Σ_k a_i[k] gh(t − k) | a_i[k] for cells of period above 1, starting at 1/p |
| `source` | as `step` | Σ_k [ (Wh[k] * mask) h(t − k − 1) + a_i[k] bh ] | Wh[k] for k = 1 … p_i − 1, and a_i as in `step` |

**Points the memo left open, which I fixed in the file; the audit should test each.**

1. **`mean` divides by p always**, also when some steps fall before the start symbol. Otherwise `mean` and `step` would differ at the first positions.
2. **In `source` the bias enters as a_i[k] · bh at each step that exists.** With the bias added once, as the memo writes it, `source` differs from `step` at the first positions of an item, where part of a period lies before the start symbol. With this form the three variants agree at the start to 2e-7 (self-check 3).
3. **A cell of period 1 has no learned weight a.** Its coefficient is fixed at 1, so it is the same cell in all four variants. The learned weights a number 384, 512 and 640 in the three period sets, as the memo counts.
4. **In `source`, Wh[0] is the existing tensor** `rnn.weight_hh_l0`, with each cell's rows divided by its period at the start, and Wh[1 …] are one new tensor of shape (K − 1, 768, 256), zero for steps outside a cell's period. So the thirteen tensors of the present network keep their names in every configuration.

### The start

`arch26.new_net(seed)` unchanged; then periods, phases and connections are drawn; then the recurrent weights outside the mask are set to zero and those inside are multiplied by sqrt(256 / (N + 1)) (4.82 at N = 10; 1 when every cell reads every other), so that a cell's recurrent input keeps its variance. Network size: 1,124,113 parameters for `G`; 932,113, 935,953 and 943,633 at N = 5, 10 and 20 counting only recurrent weights inside the mask (self-check 2).

### The two rules

- **Backpropagation (`local=False`):** exact gradients.
- **The sweep (`local=True`):** every linear map sends its error back through the signs of its weights times one scale per stacked matrix, as now (`git show $D:$M/exp3_state.py | sed -n '91,108p'`). For the masked recurrent map the file's `MaskedSign` does the same with two differences: the scale is the mean magnitude of the weights inside the mask (in `source`, over all steps stacked), and the weight gradient is zero outside the mask. When every entry is inside the mask (`G`), `exp3_state.lin` is called unchanged.
- The gates, the products, the weights a and the holding of a value between updates are differentiated exactly under both rules, as the gates are now. Credit runs through the whole item.

At the start, the cosine between the sweep's and the exact gradient of the recurrent matrix on the first batch is 0.74 for `F10`, 0.73 for `P8` and 0.72 for `P8-source` (self-check 5, seed 73129).

### Changing connections

In configurations marked "changing", every 256 updates, after the update and before scoring (`replace` in the file):

1. A connection's strength is the sum of the magnitudes of its weights over the three gates and, in `source`, over the steps.
2. A connection of cell i is given up if it has existed for at least 512 updates and its strength is below theta times the mean strength of cell i's N connections. A cell may give up none, or several. The cell's reading of itself is never given up.
3. Each one given up is replaced by a connection to a cell of B_i that is not connected to i at that moment and was not given up by i at this event: cells in index order, connections given up in increasing order of source, `rr.choice(sorted(candidates))` with `rr = Random(f"e42|replace|{seed}")` created once per run.
4. The new connection's weights are zero, so adding it changes no output. The weights of the one given up are set to zero. The optimizer's two running averages are set to zero at both.

A cell has no information about a cell it is not connected to, so the new partner is drawn at random and is kept or given up later by its strength. theta is set once in stage 0 (check 10).

## Configurations

**Stage 1**, under the sweep and under backpropagation, three seeds: 24 runs.

| Name | Connections | N | Neighbourhood | Periods | What the step changes |
|---|---|---|---|---|---|
| `G` | every cell to every other | 255 | all | all 1 | nothing: the present GRU, run by the new code |
| `F10` | fixed | 10 | 48 | all 1 | the limit and the neighbourhood |
| `A10` | changing | 10 | 48 | all 1 | connections may change |
| `P8` | changing | 10 | 48 | 1, 2, 4, 8; variant `step` | periods |

**Stage 2**, under the sweep only, three seeds: 24 runs. Each differs from its reference in one thing.

| Name | Differs from | In |
|---|---|---|
| `N5`, `N20` | `A10` | N = 5; N = 20 |
| `W10` | `A10` | the neighbourhood is all 255 other cells: the limit without the locality |
| `P4`, `P16` | `P8` | periods 1, 2, 4; periods 1, 2, 4, 8, 16 |
| `P8-last`, `P8-mean`, `P8-source` | `P8` | the variant |

## Data, dose, optimizer

- **Data:** E26's birth exactly, which is E41's present width at taught fraction 0.50: `exp11_cell.batches([(t, exp11_cell.split(t, .5, seed)[0]) for t in exp11_cell.DEV], seed, 12)`, answer bytes as targets, batch 64.
- **Update:** `exp26_candidate.step`'s text (lines 84-91) with the file's `forward(m, layer, x, local)` in place of `arch26.forward`.
- **Optimizer:** import `exp26_candidate` first; `opt = exp26_candidate.make_opt(m)`, then the layer's added parameters (`layer.added_parameters()`) join the decayed group. Learning rate 0.001, decay 0.25.
- **Dose:** 24,576 updates for every run.
- **Scoring** uses `exp11_cell.score` after `arch26.install()`, with `exp11_cell.forward` replaced by a function of the harness's signature that calls the file's forward with the run's layer.

## Scores, every 256 updates

1. **E41's three accuracies, unchanged** (E41 brief, revision 2): **A**, held-out add and sub at 12 letters on `exp11_cell.panel(t, held, seed, ks=(12,), n=300)`; **R**, the fixed sample of 1,000 held-out reduction pairs per task without letters, drawn by `Random(f"e41|{seed}|50|{t[0]}")`; and the taught panel.
2. **The length test**, E30's: `exp11_cell.panel(t, exp11_cell.split(t, .5, seed)[1], seed, ks=(12, 40), n=300)` with `t = ("add", 89, b"p")`; accuracy at 12 and at 40 letters. On file for E26's final weights at 40 letters: 96.0, 97.0, 99.67 under the sweep and 97.67, 98.67, 92.33 under backpropagation (`exp45/e30/E30_UNSEEN.json`).
3. The number of connections replaced since the previous row.

**Per run:** updates to the 50, 80 and 90 thresholds on A and on R in E41's two versions, the three-row version carrying the description; computation in weight-items with the count of weights inside the mask; CPU-seconds, reported and not used, since a masked dense matrix costs as much as a full one.

**At the first batch and at the end:** the cosine between the sweep's gradient and the exact gradient of the recurrent matrix on one batch.

**At the end, in the weights file:** the mask, periods, phases, the update at which each connection was made, and all added parameters. The report gives the share of connections never replaced, and the share of connections by the reader's period and the source's period, at the start and at the end.

**Committed:** the full weights for seed 73129 (sixteen files). For the other two seeds, a small file per run with the mask, periods, phases, the update at which each connection was made, the recurrent weights and the added parameters. Receipts in `exp45/e42/e42_<configuration>_<S|B>_<seed>.json`, in E26's format; the runner refuses an existing path.

## Checks before the runs (stage 0)

Done by the worker of seed 73129, who writes the runner `exp42_cells.py` and pushes it; the other two take it at that commit and repeat checks 1 and 2 on their hosts.

1. **`G` is E26.** The thirteen tensors are bit-equal to `arch26.new_net(seed)`'s at the start. After 2,048 updates their `tensors_sha256` is that of E26's dry births: 79618ab2… under the sweep and adf933d7… under backpropagation (seed 73129; every worker runs this).
2. **Same data in every configuration.** The batch digest at row 4,096 is E26's: bbd71ab7…, 65b1c7e0…, 1df203fb… for the three seeds.
3. **The specification file.** `e42_cells.py` is byte-identical to `$K/e42_cells_spec.py`, and its self-checks print what `$K/OUTPUT_E42_SPEC_MAC.txt` has for checks 1 to 6 (the logit differences in check 3 below 1e-6; the cosines in check 5 to two decimals).
4. **The limit, at every scored row:** every cell has exactly N connections, all inside its neighbourhood, none to itself beyond its own reading; every weight outside the mask is exactly zero.
5. **Time.** On random weights: changing the byte at position t changes no cell's value at an earlier position; a cell's value changes only at its own update positions; a cell with phase f is zero before position f (self-check 4).
6. **Gradients against a second form.** Under backpropagation the parameter gradients equal those of a reference written cell by cell, without masks or buffers, to 1e-5 of each tensor's largest entry, for `P8`, `P8-source` and `F10`. Under the sweep they differ. Gradients outside the mask are zero under both.
7. **Replacement.** At an event, adding the new connections changes no logit; the same command run twice gives the same `tensors_sha256`, with changing connections.
8. **The reviewer's reader.** For seed 73129, commit dry runs of 2,048 updates of `F10`, `P8` and `P8-source` with their weights files and their answers on the length-test panel. The reviewer writes a numpy reader from this brief and reproduces the answers. This does not hold stage 1.
9. **Cost.** CPU-seconds per update and per scored row for each configuration, on an idle host, and a forecast for the worker's sixteen runs. On my Mac the unoptimised file takes 2.6 times `G`'s time per update for `P8` and 5.4 times for `P8-source`, which would put the 48 runs near 85 CPU-hours where the reviewer estimates 50 to 70. The worker may speed up the loop over a period's steps in the runner's own copy of `forward`, provided check 6 and the self-checks still pass on it. If a worker's forecast exceeds 30 CPU-hours, report before starting.
10. **theta.** Three dry runs of `A10` under the sweep, 4,096 updates, with theta = 0.1, 0.25 and 0.5. The value used everywhere is the one whose mean share of connections replaced per event, over the events at updates 768 to 4,096, is nearest 5%. If all three are below 1% or above 30%, stop and report. The choice uses no accuracy.

## How the result is stated (fixed beforehand, as in E41)

For each comparison, rule and seed:

- **Updates to the 80 threshold**, on A and on R: E41's table of four cases (*falls* below 2/3, *rises* above 1.5, otherwise *stays*, with its rules for a run that does not reach the threshold), read here as fewer, more or about the same. The same in weight-items.
- **The length test:** the mean accuracy at 40 letters over the last three scored rows; *lower* or *higher* if the difference is 15 points or more, otherwise *about the same*.
- A cell of the table carries the description that two of its three seeds share; otherwise *mixed*. All three values are given.

**Comparisons:** `F10`, `A10` and `P8` each against `G`; `A10` against `F10`; `P8` against `A10`; every stage 2 configuration against its reference; and, in stage 1, the sweep against backpropagation within each configuration.

**One stop.** After seed 73129's stage 1 runs: if none of `F10`, `A10` and `P8` reaches 50 on A within the dose under either rule, stop and report. Otherwise stage 2 runs without a further release.

## What the record and arithmetic say beforehand

- **Units with pre-set time constants were tried once, without connections among them.** The two-bank network had fixed retention rates and no learned recurrence, and did not compose from memory: untaught plus-minus pairs 1.5, 0.5 and 7.5%, against 73.6, 78.8 and 75.1 for the GRU (review 9.101). Here the cells are connected with learned strengths, the gates read the state, and a period is an interval between updates, not a rate of decay. Because of that result the periods enter as one separate step (`P8` against `A10`).
- **A cell with a long period can carry the first operand to the `=` without learning to keep anything**, if it is updated once while the operand's digits are inside its period and not again until after the `=`. The reviewer's count: possible in 6% of training cases at period 4, 44% at period 8, 67% at period 16, and never at 40 letters. So longer periods should make training easier and the length test harder; `P4`, `P8` and `P16` test it. I did not rerun that count.

## What it cannot show

- **A saving in computation.** The input map and the readout stay dense and hold most of the weights, so limiting the recurrent connections removes 17% of the network. The job measures updates to threshold and generalisation.
- **Settings.** Learning rate and decay were chosen for the dense GRU.
- **The form of a cell.** The GRU's equations are kept; a cell whose way of combining its inputs is itself learned is a later job.
- **One task family, answer targets, one arrangement of cells.** `W10` says whether the neighbourhood matters at all.
- **Distances longer than an item.** The state is reset at every item.
- **The sign rule on few entries.** The scale and the masked weight gradient are choices; the cosine at the start and the end says how good the approximation is, not whether another choice would do better.

## Choices left to the reviewer and to me, which the user can overturn

1. The limit N counts connections between cells only; every cell still reads the whole current byte.
2. The GRU's equations are kept in this job.
3. A cell always has exactly N connections, where the user's limit is "at most N".
4. Half the cells have period 1 in every period set.
5. The scale of the kept weights at the start, and of the sign rule on a masked matrix.
6. Replacement by a strength threshold tuned to about 5% per event.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P42.1 | Check 1 passes under both rules | 90% |
| P42.2 | `A10` reaches 80 on A within the dose, two of three seeds, under the sweep | 60% |
| P42.3 | `F10` reaches 80 on A within the dose, two of three seeds, under backpropagation | 70% |
| P42.4 | `A10` against `G`, updates to 80 on A under the sweep: *rises* (more updates), two of three seeds | 65% |
| P42.5 | `A10` against `F10`, updates to 80 on A under the sweep: not *rises*, two of three seeds | 60% |
| P42.6 | `P8` against `A10`, updates to 80 on A under the sweep: fewer 20%, about the same 35%, more 45% | |
| P42.7 | `P16` is *lower* than `A10` at 40 letters, two of three seeds | 55% |
| P42.8 | `F10` is *lower* than `G` at 40 letters under the sweep, two of three seeds | 45% |
| P42.9 | `W10` against `A10`: *about the same* on both updates to 80 on A and the length test | 55% |
| P42.10 | Some stage 1 configuration's description against `G` differs between the sweep and backpropagation | 40% |
| P42.11 | The stop applies (no limited configuration reaches 50 on A on seed 73129) | 10% |

## Done

Stage 0: the runner, the check receipts, the chosen theta. Stages 1 and 2: the worker's sixteen runs with receipts and committed files, and `E42_REPORT_<seed>.md` with the worker's tables. The three-seed tables, the descriptions for every comparison, the structure tables and the predictions are assembled by the worker of seed 73129 once all three branches are complete. No readings beyond the description above.
