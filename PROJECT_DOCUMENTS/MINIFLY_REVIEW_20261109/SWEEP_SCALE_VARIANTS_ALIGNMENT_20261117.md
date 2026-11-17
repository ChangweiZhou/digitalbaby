# How close the sweep's gradient is to the exact gradient under other choices of its scale

Seed 73129; saved weights of the experiments E42, E49 and E50. This is a read-only measurement: nothing is trained. Written on 17 November 2026 at 23:02 UTC. Every figure in this document is printed by `sweep_scale_variants.py`, which lies beside it; the run that produced the figures started at 22:55:22 UTC on 17 November 2026 and took 263 seconds.

## 1. The answer

The programme compares two learning rules on one recurrent network. Rule B is back-propagation, which gives the exact gradient of the loss. Rule S, which the programme calls the sweep, is back-propagation with one change: when the error crosses a linear map backward, it is multiplied by the transpose of `sign(W) * s` in place of the transpose of the weight matrix `W`. The number `s` is the scale. In every run so far the scale has been one number for a whole matrix, the mean of `|W|`; this is variant V0. This document measures how close the sweep's gradient is to the exact gradient under V0 and under four other choices of the scale: one scale for each gate's block of rows of the two matrices of the recurrent layer (V1), one for each row of a matrix (V2), one for each column (V3), and the product of the row's mean and the column's mean divided by the matrix's mean (V4). Closeness is the cosine between the two gradients of one parameter tensor on one batch of 64 training items, that is, their inner product divided by the product of their lengths; it is 1 when the two gradients have the same direction. The figures of this section are means over 16 batches, on the networks at the end of their training unless the start is named; section 2 defines every term and section 6 gives the tables.

In short: a scale for each gate (V1) hardly changes the direction of the sweep's gradient; the cosines of the two matrices of the recurrent layer change by between −0.04 and +0.02. A scale for each row (V2) brings the sweep's gradient of these two matrices closer to the exact gradient in the networks with limited connections (by 0.03 to 0.11) and not in the dense network (by 0.002 or less). The product form (V4), applied at the recurrent layer only, gives the largest gain there (0.01 to 0.11), whereas a column scale at the last matrix `l3` moves the gradients of the readout away from the exact ones (by 0.02 to 0.12). At the start of training no variant changes these cosines by more than 0.006. Whether any variant changes learning was not measured.

In more detail:

1. **The present rule (V0).** The cosine is 0.58 to 0.82 for the recurrent map and 0.52 to 0.79 for the input map, which are the two matrices of the recurrent layer (eight trained networks). It is 0.30 to 0.66 for `l1`, 0.37 to 0.72 for the byte table, 0.61 to 0.88 for `lm` and `l2`, and exactly 1 for `l3`, whose gradient does not depend on the scale. At the start of training it is 0.73 to 0.75 for the two matrices of the recurrent layer.
2. **One scale for each gate (V1)** changes the cosines of these two matrices by between −0.04 and +0.02. The gates' mean magnitudes differ by a factor of up to 3.6 in the input map, but this choice of scale hardly changes the direction of the gradient.
3. **One scale for each row (V2)**, applied at every map, raises the cosines of these two matrices by 0.03 to 0.11 in the six trained networks with limited connections, on each of the 16 batches, and by 0.002 or less in the dense network. It leaves `lm` and `l2` within 0.001 of V0, except in the network that did not learn under the sweep (E49's `in16`), where it raises them by 0.07 to 0.09.
4. **The column scale and the product form (V3, V4).** V4 applied at the two matrices of the recurrent layer only raises their cosines by 0.01 to 0.11 in every trained network, and it is the best of V1 to V4 there in 15 of the 16 cases (eight networks, two matrices). Applied at every map, V3 and V4 lower the cosines of `lm` and `l2` by 0.02 to 0.12 in every trained network: the column means of `l3` are means over its 257 rows, whereas after training the error at the logits lies on the 11 rows of the classes that occur as targets.
5. **The upper limit at the recurrent layer.** With the exact magnitudes at the two matrices of the recurrent layer and V0 at the other maps, the cosine of the recurrent map reaches 0.76 to 0.93, where V0 gives 0.58 to 0.82 and V4 at these two matrices gives 0.61 to 0.83. What then remains below 1 is caused by the signs at `l3` and `lm`, which no scale at the recurrent layer changes.
6. **At the start** no variant changes a cosine of the two matrices of the recurrent layer by more than 0.006, because the starting magnitudes are nearly equal across rows, columns and gates.
7. **The tests.** The exact gradient agrees with finite differences of the loss to a relative error below 1e-9 on 157 weights with a sizeable gradient (Test 1). V0 on the first batch reproduces the 33 cosines that the receipts record to between four and seven decimals in double precision, and to six or more decimals when the computation is repeated in single precision with the two norms accumulated in eight running sums; the receipts' figures are therefore shifted, upward in 25 of 33 cases, by up to 0.000016 by single-precision arithmetic (Test 2). The variant with exact magnitudes gives cosine 1 to within 2.2e-16 for every tensor (Test 3).
8. **What is not shown.** A cosine nearer to 1 is not a demonstration that a network would learn faster; no network was trained under any variant (section 7). The dense network at its start could not be measured, because no saved file holds its recurrent matrix before training. All figures are for one seed, for batches of taught items, and for weights that were trained under V0 or under back-propagation.

## 2. Terms

**The network.** The network reads one byte at each position of an item. Its parameters are named as in the weight files.

- The *byte table* (`book.weight`, 258 rows of 8 numbers) holds one row for each of the 256 byte values, for the end symbol (256) and for the start symbol (257).
- `l1` (512 by 8) and a hyperbolic tangent turn the byte's row into the *byte's code* `z1`, a vector of 512 values. In E50 the byte table and `l1` are replaced by a fixed code that is not trained.
- The *recurrent layer* is a gated recurrent unit (GRU) of 256 cells with state `h`. For each cell it has a reset gate `r`, an update gate `z` and a candidate state `n`: `r = σ(Wi_r z1 + Wh_r h')`, `z = σ(Wi_z z1 + Wh_z h')`, `n = tanh(Wi_n z1 + r * (Wh_n h'))`, `h = (1 - z) * n + z * h'`, where `σ` is the logistic function, `h'` is the state after the previous position (zero before the first) and biases are omitted from the formulae. The *input map* (`rnn.weight_ih_l0`, 768 by 512) stacks `Wi_r`, `Wi_z`, `Wi_n`; the *recurrent map* (`rnn.weight_hh_l0`, 768 by 256) stacks `Wh_r`, `Wh_z`, `Wh_n`. A *gate's block* is the 256 rows of one of the three (rows 0 to 255 for `r`, 256 to 511 for `z`, 512 to 767 for `n`). These two matrices are called the GRU maps.
- The *readout* is `z2 = tanh(l2 z1 + lm h)` followed by `l3 z2`, which gives 257 numbers (logits), one for each class of the next byte. `l2` is 512 by 512, `lm` is 512 by 256, `l3` is 257 by 512.
- A *linear map* is the multiplication by one of the six weight matrices (`l1`, input map, recurrent map, `l2`, `lm`, `l3`) followed by the addition of its bias.

**Limited connections.** In some networks a matrix is multiplied entry by entry by a fixed *mask* of zeros and ones. The entries where the mask is one are the *kept entries*; the others are zero throughout training. In E42's `F10` each cell reads 10 other cells and itself, so each row of the recurrent map keeps 11 of 256 entries. In E49's `in16` each cell, in addition, reads 16 of the 512 values of the byte's code, so each row of the input map keeps 16 of 512 entries. A network without a mask is called dense.

**Items, batches, the stream, the loss.** An *item* is a line such as `p41+ABC27=68` followed by the end symbol: a task code, two operands with an operation symbol, between 0 and 12 filler letters, and the answer. The *answer's positions* are the three positions at which the network must predict the two answer bytes and the end symbol. A *batch* is 64 items; the *stream* is the sequence of batches a run is fed, which draws its items from the taught half of all pairs of operands. The *loss* of a batch is the mean, over its 192 answer positions, of the cross-entropy between the network's prediction and the right byte. A *class that occurs as a target* is one of the 11 classes that stand at an answer position (the ten digits and the end symbol).

**The two rules.** The *error* at a point of the network is the derivative of the loss with respect to the values at that point. Back-propagation carries the error from the logits toward the input; where it crosses a linear map backward, it is multiplied by the transpose of the map's weight matrix, and the gradient of the map's own weights is the outer product of the error at the map's output and the map's input. The result is the *exact gradient*. *The sweep* differs in the crossing only: the error is multiplied by the transpose of `F = sign(W) * S`, where `S` is a matrix of non-negative *magnitudes*. Under the present rule every entry of `S` is the same number, the *scale*, equal to the mean of `|W|` over the matrix (over the kept entries where the matrix is masked, with `sign` zero outside the mask). The gradient of a map's own weights is still the outer product, and the gates' logistic functions, hyperbolic tangents and products are differentiated exactly under both rules. The error passes back through all positions of an item under both rules, so the recurrent map is crossed once for every position.

**The variants.** Each variant is a choice of the magnitude `S[i, j]` that replaces `|W[i, j]|`; all means are over kept entries.

- **V0** (the present rule): the mean of `|W|` over the whole matrix.
- **V1**: for the two GRU maps, the mean of `|W|` over the gate's block that holds row `i` (three numbers a matrix); for the other maps as V0.
- **V2**: the mean of `|W|` over row `i`, that is, over the incoming weights of the unit that receives through that row.
- **V3**: the mean of `|W|` over column `j`, that is, over the outgoing weights of the unit that sends through that column.
- **V4**: the mean of row `i` times the mean of column `j`, divided by the mean of the matrix.
- **V5**: `|W[i, j]|` itself. Then `F = W` and the sweep is back-propagation. V5 is a test of the code and the upper limit for comparison, not a candidate.

A variant is applied either *at every map* (`l1`, the two GRU maps, `l2`, `lm`, `l3`) or *at the two GRU maps only*, with V0 at the other maps. V1 is the same in both cases by its definition.

**The quantities reported.** For a parameter tensor, the *cosine* is the inner product of the sweep's gradient and the exact gradient, each taken as one vector of all the tensor's entries, divided by the product of their Euclidean norms. The *norm ratio* is the norm of the sweep's gradient divided by the norm of the exact gradient. "All parameters" means all trained tensors, weights and biases, joined into one vector; "all but l3" leaves out the weights and the bias of `l3`, whose gradient is the same under both rules. The *coefficient of variation* (CV) of a set of numbers is their standard deviation divided by their mean. `cos(F, W)` is the cosine between the matrix `F` of a variant and the weight matrix, over the kept entries.

**Start, end, receipt and other words of the programme.** The *start* is a network before its first update, the *end* the network after its 24,576 updates. A *receipt* is the file a run writes with its settings and results. A *runner* is the programme's word for the script that carries out the runs of one experiment. The *training library* is PyTorch (`torch` in the code and in the script's output). *Single precision* and *double precision* are the 32-bit and 64-bit floating-point formats (float32, float64).

**The networks measured.** The labels join the experiment's own name of the configuration with the rule under which the network was trained (S or B), or with "start".

```
networks (for a trained network, the last row of its receipt: A, the per cent of exact answers on 600 pairs that were not taught, addition and subtraction with 12 filler letters; R, the same on 2,000 untaught pairs whose answer needs the reduction modulo 89, without filler letters):
  G_S            E42 G (dense), trained under the sweep  after 24,576 updates A 98.00 R 95.85  [origin/claude/gallant-dijkstra-i9zgsr:MOD97_LOCAL_RULE_20261106/exp45/e42/e42_G_S_73129.pt]
  G_B            E42 G (dense), trained under back-propagation  after 24,576 updates A 98.50 R 98.15  [origin/claude/gallant-dijkstra-i9zgsr:MOD97_LOCAL_RULE_20261106/exp45/e42/e42_G_B_73129.pt]
  F10_start      E42 F10 at its start (the start file of E49's check 2, a run of F10 by E49's runner)  [origin/claude/peaceful-noether-vcvrmf:MOD97_LOCAL_RULE_20261106/exp45/e49_qual/check2_73129_runs/e49_F10_S_dry2048_73129_start.pt]
  F10_S          E42 F10 (10 neighbours and itself), trained under the sweep  after 24,576 updates A 92.17 R 88.45  [origin/claude/gallant-dijkstra-i9zgsr:MOD97_LOCAL_RULE_20261106/exp45/e42/e42_F10_S_73129.pt]
  F10_B          E42 F10, trained under back-propagation  after 24,576 updates A 91.83 R 93.30  [origin/claude/gallant-dijkstra-i9zgsr:MOD97_LOCAL_RULE_20261106/exp45/e42/e42_F10_B_73129.pt]
  in16_start     E49 in16 at its start  [origin/claude/peaceful-noether-vcvrmf:MOD97_LOCAL_RULE_20261106/exp45/e49/e49_in16_S_73129_start.pt]
  in16_S         E49 in16 (16 of 512 input values a cell), sweep: did not learn  after 24,576 updates A 2.17 R 1.30  [origin/claude/peaceful-noether-vcvrmf:MOD97_LOCAL_RULE_20261106/exp45/e49/e49_in16_S_73129.pt]
  in16_B         E49 in16, trained under back-propagation  after 24,576 updates A 87.67 R 83.20  [origin/claude/peaceful-noether-vcvrmf:MOD97_LOCAL_RULE_20261106/exp45/e49/e49_in16_B_73129.pt]
  C-dense_start  E50 C-dense at its start (fixed code)  [origin/claude/e50-fixed-input-code-017dSE:MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C-dense_S_73129_start.pt]
  C-dense_S      E50 C-dense, trained under the sweep  after 24,576 updates A 92.33 R 93.65  [origin/claude/e50-fixed-input-code-017dSE:MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C-dense_S_73129.pt]
  C-dense_B      E50 C-dense, trained under back-propagation  after 24,576 updates A 98.67 R 97.50  [origin/claude/e50-fixed-input-code-017dSE:MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C-dense_B_73129.pt]
```

## 3. Files

All weight files and receipts are read with `git show <ref>:<path>` into memory, one at a time; no code of another branch is executed. The paths of the weight files are printed in the list above; the receipts are the `.json` files of the same names. The code that defines the rules was read, not run:

| what | file (folder `MOD97_LOCAL_RULE_20261106`) | branch |
|---|---|---|
| the sweep's crossing of a map | `exp3_state.py`, class `SignLinear`, lines 91 to 104 | `origin/codex/minifly-consolidated-20261023` |
| the dense network | `arch26.py`, `forward`; `arch20.py`, class `NetRNN` | the same |
| the answer's positions and the loss | `exp45.py`, `batch_tensor`; `exp26_candidate.py`, `step` | the same |
| the stream | `exp11_cell.py`, `render`, `split`, `stream`, `batches` | the same |
| the masked recurrent map and its scale | `e42_cells.py`, class `MaskedSign`, `Layer.sign_scale`, `forward`; `exp42_cells.py`, `cosines` | `origin/claude/gallant-dijkstra-i9zgsr` |
| the masked input map | `e49_interface.py`, `masked`, `forward`; `exp49_interface.py`, `cosines` | `origin/claude/peaceful-noether-vcvrmf` |
| the fixed code | `e50_sparse_code.py`, `forward`; `exp50_code.py`, `cosines` | `origin/claude/e50-fixed-input-code-017dSE` |

Files of this folder that the script uses: `compare_pt_tensors.py` (reads a weight file without the training library, with an unpickler that refuses every class except those that describe a tensor); `COORDINATION_20261117/check_e54_brief.py` (the task and the stream were copied from it into the script; it uses the standard library's random generator only); and, for one comparison, the reviewer's readers `e42_cells_reader.py`, `e49_interface_reader.py` and `e50_code_reader.py`. The untested scales are listed in `COORDINATION_20261117/INVENTORY_DENSE_LINE_20261117.md` (row 2.10 and item 6 of its closing list: one scale for each gate) and in `COORDINATION_20261117/INVENTORY_STRUCTURE_AND_WORLD_20261117.md` (item 9 of its closing list: a scale from a cell's own weights).

The start of `F10` is not among E42's files. It is taken from the start file of check 2 of E49, a run of `F10` by E49's programme (`exp45/e49_qual/check2_73129_runs/e49_F10_S_dry2048_73129_start.pt` on E49's branch). This file holds the same tensors, bit for bit, as the start file of E50's `C-dense`, as it must, because both are the dense starting network with the recurrent map masked; and its recurrent cosine reproduces the start cosine in E42's receipts of `F10` (section 5.3).

## 4. Method

**The forward pass and the loss** are written in numpy in double precision from the equations of section 2, with every masked matrix multiplied by its mask. Two arrangements are used that change no value. The byte's code and the input map's output depend on the byte alone, so they are computed once for each of the 258 bytes and looked up. The readout is computed at the answer's positions only, since no other position enters the loss.

**The backward pass** is one function for both rules. It takes, for each linear map, the matrix `F` through whose transpose the error crosses the map backward. With `F` equal to the (masked) weight matrix it gives the exact gradient; with `F = sign(W) * S` it gives the sweep's gradient under the variant that `S` belongs to. The gradient of a map's own weights is the outer product of the error at its output and its input, set to zero outside the mask, as in the programme's code. The error at the readout is zero outside the answer's positions, so that part is written on those positions only, and on the input side the sums over positions are taken byte by byte first; neither changes a value.

**One departure from the derivative of the loss.** The byte table is built with the start symbol's row as a padding row: the training library gives that row no gradient, and the row is zero in every file. The script therefore sets the gradient of that row to zero under both rules, as training does. The true derivative of the loss with respect to that row is not zero (Test 1 prints it); it plays no part in any run.

**The batches.** The stream is rebuilt from the task's definition. The first batch of the stream is the batch on which the receipts record their cosines; it is used for Test 2. The tables use the 16 batches that follow it (batches 2 to 17), and give for every quantity the mean over these 16 batches and, in brackets, the least and the greatest value. The same batches are used for every network and every variant.

**Which gradient depends on which map's matrix `F`.** The gradient of `l3` uses the error at the logits, which no map has yet changed, so it is the same under every rule and variant. The gradients of `l2` and `lm` use the error after one crossing, of `l3`; they depend on the matrix `F` of `l3` and on no other. The gradients of the two GRU maps depend on `F` of `l3`, of `lm` and of the recurrent map. The gradient of `l1` depends in addition on `F` of `l2` and of the input map, and the gradient of the byte table on all six. In E50's `C-dense` the byte's code is fixed, so there is no gradient for the byte table and `l1`, and the matrices `F` of `l2` and of the input map are never used.

**Biases.** The tables give the weight tensors. The script also prints, for every network, the mean cosines of the six bias vectors under V0 to V4. Under V0 they lie within 0.08 of the cosines of the weight tensors of the same maps, with one exception (the bias of `l1` in `G_B`, 0.479 against 0.657 for the weights); they are not repeated here.

## 5. The tests and their outcomes

### 5.1 The stream and the forward pass

The receipts of E42, E49 and E50 record a digest (SHA-256) of the batches they were fed. The digest of the first 256 batches rebuilt here equals the recorded one in all four receipts read:

```
e42/e42_G_S_73129.json: recorded a921d91607c6421b over 256 batches; rebuilt a921d91607c6421b; equal True
e42/e42_F10_S_73129.json: recorded a921d91607c6421b over 256 batches; rebuilt a921d91607c6421b; equal True
e49/e49_in16_S_73129.json: recorded a921d91607c6421b over 256 batches; rebuilt a921d91607c6421b; equal True
e50/e50_C-dense_S_73129.json: recorded a921d91607c6421b over 256 batches; rebuilt a921d91607c6421b; equal True
batches used: the first (the receipts' batch) and the next 16; items a batch 64; positions a batch [22]; answer positions a batch [192]
```

The forward pass of the script was compared with the reviewer's readers, which are separate implementations in single precision written from the briefs. The largest difference of a logit at an answer position is 1.3e-05 or less in every network, on logits of magnitude up to 38, which is the size of single-precision rounding. E50's receipt records the loss on the first batch at the start, 5.596784592; the script gives 5.596785118.

```
G_S            e42_cells_reader.forward: largest difference of a logit 9.5e-06 (largest logit magnitude 28.01); loss here 0.032447388; weights outside the masks all zero True
G_B            e42_cells_reader.forward: largest difference of a logit 1.1e-05 (largest logit magnitude 31.55); loss here 0.023945867; weights outside the masks all zero True
F10_start      e42_cells_reader.forward: largest difference of a logit 4.3e-07 (largest logit magnitude 0.67); loss here 5.577162846; weights outside the masks all zero True
F10_S          e42_cells_reader.forward: largest difference of a logit 1.3e-05 (largest logit magnitude 38.37); loss here 0.049130056; weights outside the masks all zero True
F10_B          e42_cells_reader.forward: largest difference of a logit 1.3e-05 (largest logit magnitude 23.48); loss here 0.046299035; weights outside the masks all zero True
in16_start     e49_interface_reader.forward: largest difference of a logit 4.5e-07 (largest logit magnitude 0.63); loss here 5.573763931; weights outside the masks all zero True
in16_S         e49_interface_reader.forward: largest difference of a logit 1.3e-05 (largest logit magnitude 22.13); loss here 0.951090168; weights outside the masks all zero True
in16_B         e49_interface_reader.forward: largest difference of a logit 1.2e-05 (largest logit magnitude 24.79); loss here 0.105008098; weights outside the masks all zero True
C-dense_start  e50_code_reader.forward: largest difference of a logit 3.7e-07 (largest logit magnitude 0.69); loss here 5.596785118; weights outside the masks all zero True
               E50's receipt, loss on the first batch at the start: 5.596784592 (B) 5.596784592 (S); here 5.596785118
C-dense_S      e50_code_reader.forward: largest difference of a logit 1.1e-05 (largest logit magnitude 30.33); loss here 0.027161893; weights outside the masks all zero True
C-dense_B      e50_code_reader.forward: largest difference of a logit 1.2e-05 (largest logit magnitude 33.14); loss here 0.024414073; weights outside the masks all zero True
```

### 5.2 Test 1: the exact gradient against finite differences of the loss

For single weights, the derivative of the loss was computed by a five-point central difference with step 0.001 in double precision and compared with the script's exact gradient on the first batch, in five networks that cover the four kinds (dense, masked recurrent map, masked input map, fixed code) at the start and at the end. The weights were drawn at random from every tensor in two groups. Group A is drawn among all kept entries: 3 from each weight tensor and 2 from each bias. Group B is drawn among the kept entries whose exact gradient is at least a tenth of the largest in the tensor, in the same numbers. The reason for the second group is that a finite difference carries a rounding error of at least about 1.5 times 2.2e-16 times the loss divided by the step, so a relative error has no meaning for a weight whose gradient is near that size; after training many weights have gradients of that size, as the figures of group A show.

Outcome: in group B (157 weights over the five networks) the largest relative error is 7.6e-10. In group A (157 weights) the largest relative error is 1.3e-03, at a weight whose gradient is 3.3e-12 in magnitude, where the absolute error is 4.3e-15. In each of the four trained networks the largest relative error of group A is at a weight whose gradient is below 1e-9 in magnitude; in the network at its start it is 6.3e-08, at a gradient of 9.8e-06. Two weights of group A have a gradient and a finite difference that are both exactly zero. The test is passed.

```
G_S            loss 0.032447; rounding error of the difference at least about 1e-14; 66 weights over 13 tensors
               group A (33 weights among all kept entries): both exactly zero in 0; largest relative error 1.3e-03, at a weight whose gradient is 3.3e-12 in magnitude (absolute error there 4.3e-15)
               group B (33 weights with a gradient of at least a tenth of the tensor's largest; the least of them 1.4e-04): largest relative error 1.0e-10
               group B, largest relative error by tensor: book.weight 3e-12, l1.weight 9e-12, rnn.weight_ih_l0 2e-11, rnn.weight_hh_l0 7e-11, lm.weight 4e-11, l2.weight 1e-10, l3.weight 3e-12, l1.bias 9e-12, rnn.bias_ih_l0 3e-11, rnn.bias_hh_l0 3e-11, lm.bias 2e-11, l2.bias 3e-11, l3.bias 6e-12
               the start symbol's row of the byte table: derivative of the loss by finite difference -7.501e-03; gradient used in training (padding row) 0.0
F10_B          loss 0.046299; rounding error of the difference at least about 2e-14; 66 weights over 13 tensors
               group A (33 weights among all kept entries): both exactly zero in 0; largest relative error 2.2e-05, at a weight whose gradient is 2.0e-10 in magnitude (absolute error there 4.6e-15)
               group B (33 weights with a gradient of at least a tenth of the tensor's largest; the least of them 3.0e-04): largest relative error 4.1e-11
               group B, largest relative error by tensor: book.weight 5e-12, l1.weight 2e-11, rnn.weight_ih_l0 2e-11, rnn.weight_hh_l0 1e-11, lm.weight 2e-11, l2.weight 2e-11, l3.weight 3e-12, l1.bias 4e-12, rnn.bias_ih_l0 1e-11, rnn.bias_hh_l0 3e-11, lm.bias 4e-11, l2.bias 4e-11, l3.bias 1e-12
               the start symbol's row of the byte table: derivative of the loss by finite difference 4.682e-04; gradient used in training (padding row) 0.0
in16_start     loss 5.573764; rounding error of the difference at least about 2e-12; 66 weights over 13 tensors
               group A (33 weights among all kept entries): both exactly zero in 0; largest relative error 6.3e-08, at a weight whose gradient is 9.8e-06 in magnitude (absolute error there 6.1e-13)
               group B (33 weights with a gradient of at least a tenth of the tensor's largest; the least of them 7.7e-04): largest relative error 7.6e-10
               group B, largest relative error by tensor: book.weight 5e-10, l1.weight 1e-10, rnn.weight_ih_l0 6e-10, rnn.weight_hh_l0 8e-10, lm.weight 3e-10, l2.weight 1e-10, l3.weight 3e-11, l1.bias 3e-10, rnn.bias_ih_l0 6e-11, rnn.bias_hh_l0 2e-10, lm.bias 5e-11, l2.bias 4e-11, l3.bias 1e-11
               the start symbol's row of the byte table: derivative of the loss by finite difference 5.006e-05; gradient used in training (padding row) 0.0
in16_S         loss 0.951090; rounding error of the difference at least about 3e-13; 66 weights over 13 tensors
               group A (33 weights among all kept entries): both exactly zero in 0; largest relative error 9.0e-04, at a weight whose gradient is 5.8e-11 in magnitude (absolute error there 5.2e-14)
               group B (33 weights with a gradient of at least a tenth of the tensor's largest; the least of them 1.6e-04): largest relative error 5.0e-10
               group B, largest relative error by tensor: book.weight 7e-12, l1.weight 1e-11, rnn.weight_ih_l0 4e-12, rnn.weight_hh_l0 3e-11, lm.weight 2e-10, l2.weight 5e-10, l3.weight 2e-11, l1.bias 2e-11, rnn.bias_ih_l0 3e-10, rnn.bias_hh_l0 1e-11, lm.bias 1e-11, l2.bias 7e-11, l3.bias 6e-12
               the start symbol's row of the byte table: derivative of the loss by finite difference 1.153e-02; gradient used in training (padding row) 0.0
C-dense_B      loss 0.024414; rounding error of the difference at least about 8e-15; 50 weights over 10 tensors
               group A (25 weights among all kept entries): both exactly zero in 2; largest relative error 3.0e-04, at a weight whose gradient is 9.4e-13 in magnitude (absolute error there 2.8e-16)
               group B (25 weights with a gradient of at least a tenth of the tensor's largest; the least of them 3.4e-04): largest relative error 7.5e-11
               group B, largest relative error by tensor: rnn.weight_ih_l0 8e-11, rnn.weight_hh_l0 5e-12, lm.weight 7e-11, l2.weight 6e-11, l3.weight 1e-11, rnn.bias_ih_l0 3e-11, rnn.bias_hh_l0 2e-11, lm.bias 6e-12, l2.bias 5e-11, l3.bias 6e-12
```

### 5.3 Test 2: V0 on the first batch against the cosines in the receipts

The receipts record the cosine between the sweep's and the exact gradient on the first batch of the stream, at the start and at the end of each run: for the recurrent map in E42, and for the input map, `lm` and the recurrent map in E49 and E50. These were computed by the training library in single precision. The table gives, for each of the 33 recorded cosines that concern the networks measured here, the recorded figure and two figures of the script. The first is the script's figure in double precision, the one used in the rest of this document. The second repeats the whole computation in single precision and forms the cosine as the sum of the products of the two gradients after each has been divided by its norm, with the squares that make up each norm accumulated in eight running sums (the first sum takes entries 1, 9, 17 and so on of the tensor, the second entries 2, 10, 18 and so on), which is how a vectorised single-precision loop of eight accumulators adds them. E42's receipts take the cosine over the whole recurrent matrix, E49's and E50's over the kept entries; the script does the same in each case. The column "decimals" gives the largest n for which two figures differ by less than half a unit of the n-th decimal.

| network | tensor | receipt | at | recorded | here, float64 | difference | decimals | here, float32 and the norms in eight running sums | difference | decimals |
|---|---|---|---|---|---|---|---|---|---|---|
| G_S | recurrent map | e42_G_S_73129.json | end | 0.7448674 | 0.7448663 | -1.1e-06 | 5 | 0.7448676 | +1.2e-07 | 6 |
| G_B | recurrent map | e42_G_B_73129.json | end | 0.6885734 | 0.6885713 | -2.1e-06 | 5 | 0.6885734 | +0.0e+00 | 9 |
| F10_start | input map | e49_F10_S_dry2048_73129.json | start | 0.7499694 | 0.7499682 | -1.1e-06 | 5 | 0.7499693 | -6.0e-08 | 6 |
| F10_start | lm | e49_F10_S_dry2048_73129.json | start | 0.8650787 | 0.8650766 | -2.1e-06 | 5 | 0.8650787 | +6.0e-08 | 6 |
| F10_start | recurrent map | e49_F10_S_dry2048_73129.json | start | 0.7436286 | 0.7436286 | -6.6e-08 | 6 | 0.7436286 | -6.0e-08 | 6 |
| F10_start | recurrent map | e42_F10_S_73129.json | start | 0.7436285 | 0.7436286 | +5.3e-08 | 6 | 0.7436284 | -6.0e-08 | 6 |
| F10_start | recurrent map | e42_F10_B_73129.json | start | 0.7436285 | 0.7436286 | +5.3e-08 | 6 | 0.7436284 | -6.0e-08 | 6 |
| F10_S | recurrent map | e42_F10_S_73129.json | end | 0.6041967 | 0.6041963 | -3.9e-07 | 6 | 0.6041968 | +1.2e-07 | 6 |
| F10_B | recurrent map | e42_F10_B_73129.json | end | 0.6312042 | 0.6312043 | +8.2e-08 | 6 | 0.6312042 | +6.0e-08 | 6 |
| in16_start | input map | e49_in16_S_73129.json | start | 0.7418082 | 0.7418081 | -8.9e-08 | 6 | 0.7418082 | -6.0e-08 | 6 |
| in16_start | lm | e49_in16_S_73129.json | start | 0.8654440 | 0.8654432 | -8.4e-07 | 5 | 0.8654439 | -6.0e-08 | 6 |
| in16_start | recurrent map | e49_in16_S_73129.json | start | 0.7311290 | 0.7311291 | +1.5e-07 | 6 | 0.7311291 | +6.0e-08 | 6 |
| in16_start | input map | e49_in16_B_73129.json | start | 0.7418082 | 0.7418081 | -8.9e-08 | 6 | 0.7418082 | -6.0e-08 | 6 |
| in16_start | lm | e49_in16_B_73129.json | start | 0.8654440 | 0.8654432 | -8.4e-07 | 5 | 0.8654439 | -6.0e-08 | 6 |
| in16_start | recurrent map | e49_in16_B_73129.json | start | 0.7311290 | 0.7311291 | +1.5e-07 | 6 | 0.7311291 | +6.0e-08 | 6 |
| in16_S | input map | e49_in16_S_73129.json | end | 0.6192584 | 0.6192580 | -3.6e-07 | 6 | 0.6192584 | +0.0e+00 | 9 |
| in16_S | lm | e49_in16_S_73129.json | end | 0.5440302 | 0.5440294 | -9.0e-07 | 5 | 0.5440301 | -1.8e-07 | 6 |
| in16_S | recurrent map | e49_in16_S_73129.json | end | 0.6053340 | 0.6053339 | -6.4e-08 | 6 | 0.6053340 | +6.0e-08 | 6 |
| in16_B | input map | e49_in16_B_73129.json | end | 0.7272432 | 0.7272434 | +2.0e-07 | 6 | 0.7272432 | +6.0e-08 | 6 |
| in16_B | lm | e49_in16_B_73129.json | end | 0.7518172 | 0.7518159 | -1.3e-06 | 5 | 0.7518171 | -6.0e-08 | 6 |
| in16_B | recurrent map | e49_in16_B_73129.json | end | 0.7058899 | 0.7058900 | +8.1e-08 | 6 | 0.7058900 | +6.0e-08 | 6 |
| C-dense_start | input map | e50_C-dense_S_73129.json | start | 0.7566603 | 0.7566447 | -1.6e-05 | 4 | 0.7566604 | +1.2e-07 | 6 |
| C-dense_start | lm | e50_C-dense_S_73129.json | start | 0.8671099 | 0.8671082 | -1.7e-06 | 5 | 0.8671100 | +1.2e-07 | 6 |
| C-dense_start | recurrent map | e50_C-dense_S_73129.json | start | 0.7234635 | 0.7234633 | -1.3e-07 | 6 | 0.7234634 | -1.2e-07 | 6 |
| C-dense_start | input map | e50_C-dense_B_73129.json | start | 0.7566603 | 0.7566447 | -1.6e-05 | 4 | 0.7566604 | +1.2e-07 | 6 |
| C-dense_start | lm | e50_C-dense_B_73129.json | start | 0.8671099 | 0.8671082 | -1.7e-06 | 5 | 0.8671100 | +1.2e-07 | 6 |
| C-dense_start | recurrent map | e50_C-dense_B_73129.json | start | 0.7234635 | 0.7234633 | -1.3e-07 | 6 | 0.7234634 | -1.2e-07 | 6 |
| C-dense_S | input map | e50_C-dense_S_73129.json | end | 0.7205225 | 0.7205163 | -6.2e-06 | 4 | 0.7205225 | +0.0e+00 | 9 |
| C-dense_S | lm | e50_C-dense_S_73129.json | end | 0.8746144 | 0.8746128 | -1.6e-06 | 5 | 0.8746142 | -1.2e-07 | 6 |
| C-dense_S | recurrent map | e50_C-dense_S_73129.json | end | 0.6981251 | 0.6981251 | +7.7e-08 | 6 | 0.6981250 | -6.0e-08 | 6 |
| C-dense_B | input map | e50_C-dense_B_73129.json | end | 0.6500675 | 0.6500617 | -5.8e-06 | 4 | 0.6500674 | -6.0e-08 | 6 |
| C-dense_B | lm | e50_C-dense_B_73129.json | end | 0.7108536 | 0.7108524 | -1.2e-06 | 5 | 0.7108536 | +0.0e+00 | 9 |
| C-dense_B | recurrent map | e50_C-dense_B_73129.json | end | 0.6390204 | 0.6390204 | -3.1e-08 | 7 | 0.6390204 | +0.0e+00 | 9 |

```
decimals: the largest n for which the two figures differ by less than half a unit of the n-th decimal (9 at most).
33 recorded cosines compared. float64: largest difference 1.6e-05; decimals: least 4, most 7; recorded above this script's figure in 25 of 33.
float32 with the norms in eight running sums: largest difference 1.8e-07 (the spacing of float32 numbers between 0.5 and 1 is 6.0e-08); decimals: least 6, most 9.
C-dense_start, input map (393,216 entries), recorded 0.7566603; the float32 gradients of this script with the cosine in float64 0.7566447; with the norms in single precision in 1, 4, 8, 16 running sums: 0.7568470, 0.7566823, 0.7566604, 0.7566515
not reproduced: E42's receipts of G record at the start 0.7409766 (S) and 0.7409766 (B) for the recurrent map; no saved file holds G's recurrent matrix before training.
start files of the two rules hold the same tensors: E49 in16 True; E50 C-dense True
the start file of E49's check 2 (F10) and E50's C-dense start file hold the same tensors: True
```

Outcome and what was found.

1. In double precision the script reproduces the recorded cosines to between 4 and 7 decimals; the largest difference is 0.000016, for the input map of `C-dense` at its start. The differences are not symmetric: the recorded figure is above the script's in 25 of the 33 cases, and the differences grow with the number of entries of the tensor (at most 3.9e-07 for the masked maps with 8,448 or 12,288 kept entries, 8.4e-07 to 2.1e-06 for `lm` with 131,072 entries and for the dense recurrent map with 196,608, and 1.1e-06 to 1.6e-05 for the dense input map with 393,216).
2. This asymmetry was examined before anything else was reported. Repeating the script's computation in single precision does not by itself produce it: the single-precision gradients, with the cosine formed in double precision, give 0.7566447 for the input map of `C-dense` at its start, the double-precision figure. What produces it is the accumulation of the two norms in single precision. With the squares accumulated in eight running sums, the script gives 0.7566604 where the receipt has 0.7566603, and all 33 recorded cosines are reproduced to within 1.8e-07, which is three times the spacing of single-precision numbers of this size (6.0e-08). With one, four or sixteen running sums the same case gives 0.7568470, 0.7566823 and 0.7566515, so the agreement is specific to eight.
3. The conclusion is that the gradients of the script agree with those of the recorded runs to single precision, and that the cosines in the receipts are shifted, mostly upward, by up to 0.000016 because the norms of large tensors were accumulated in single precision. The training library is not installed here and its source was not read, so the form of its accumulation is inferred from this agreement and is not confirmed from the library itself. The shift is far below the third decimal, to which the tables of this document are given.
4. The start files of the two rules hold the same tensors in E49 and in E50, so one start is measured for each. The receipts hold no cosine for `l1`, `l2` or the byte table; for these tensors the sweep's gradient is supported by Test 3 and Test 1 only.

The test is passed.

### 5.4 Test 3: V5 gives cosine 1

V5 builds `F` as the signs times the exact magnitudes and passes it through the same code as the other variants. Its gradient was compared with the exact gradient for every network on all 17 batches: the largest deviation of a cosine from 1 is 2.2e-16 over all weight tensors, biases, gate blocks and all parameters together, and every norm ratio is exactly 1. Since Test 1 shows that the backward pass with `F = W` gives the derivative of the loss, this test shows that the construction of `F` from signs and magnitudes, masks included, loses nothing. The test is passed.

```
G_S            largest |1 - cosine| over 21 cosines (weights, biases, all parameters, gate blocks) and 17 batches 2.2e-16; largest |1 - norm ratio| 0.0e+00
G_B            largest |1 - cosine| over 21 cosines (weights, biases, all parameters, gate blocks) and 17 batches 2.2e-16; largest |1 - norm ratio| 0.0e+00
F10_start      largest |1 - cosine| over 21 cosines (weights, biases, all parameters, gate blocks) and 17 batches 2.2e-16; largest |1 - norm ratio| 0.0e+00
F10_S          largest |1 - cosine| over 21 cosines (weights, biases, all parameters, gate blocks) and 17 batches 2.2e-16; largest |1 - norm ratio| 0.0e+00
F10_B          largest |1 - cosine| over 21 cosines (weights, biases, all parameters, gate blocks) and 17 batches 2.2e-16; largest |1 - norm ratio| 0.0e+00
in16_start     largest |1 - cosine| over 21 cosines (weights, biases, all parameters, gate blocks) and 17 batches 2.2e-16; largest |1 - norm ratio| 0.0e+00
in16_S         largest |1 - cosine| over 21 cosines (weights, biases, all parameters, gate blocks) and 17 batches 2.2e-16; largest |1 - norm ratio| 0.0e+00
in16_B         largest |1 - cosine| over 21 cosines (weights, biases, all parameters, gate blocks) and 17 batches 2.2e-16; largest |1 - norm ratio| 0.0e+00
C-dense_start  largest |1 - cosine| over 18 cosines (weights, biases, all parameters, gate blocks) and 17 batches 2.2e-16; largest |1 - norm ratio| 0.0e+00
C-dense_S      largest |1 - cosine| over 18 cosines (weights, biases, all parameters, gate blocks) and 17 batches 2.2e-16; largest |1 - norm ratio| 0.0e+00
C-dense_B      largest |1 - cosine| over 18 cosines (weights, biases, all parameters, gate blocks) and 17 batches 2.2e-16; largest |1 - norm ratio| 0.0e+00
```

## 6. Results

### 6.1 The present rule against the best variant

For each network and tensor: the mean cosine under V0; the variant among V1 to V4 with the highest mean cosine when applied at every map, and when applied at the two GRU maps only, each with its gain over V0 and, in square brackets, the number of the 16 batches on which it is above V0; and the mean cosine with the exact magnitudes at the two GRU maps and V0 elsewhere. `l3` is left out because its cosine is 1 in every case. The best variant is chosen on the same 16 batches on which its gain is reported, so a small gain may owe something to the choice.

| network | tensor | V0 | best at every map | gain | best at the GRU maps only | gain | exact magnitudes at the GRU maps only |
|---|---|---|---|---|---|---|---|
| G_S | byte table | 0.722 | V4 0.840 | +0.117 [16] | V4 0.783 | +0.060 [16] | 0.912 |
| G_S | l1 | 0.661 | V4 0.760 | +0.100 [16] | V4 0.740 | +0.079 [16] | 0.942 |
| G_S | input map | 0.788 | V4 0.811 | +0.023 [14] | V4 0.800 | +0.012 [16] | 0.932 |
| G_S | recurrent map | 0.819 | V4 0.841 | +0.022 [15] | V4 0.830 | +0.011 [16] | 0.925 |
| G_S | lm | 0.809 | V2 0.810 | +0.000 [10] | V1 0.809 | +0.000 [0] | 0.809 |
| G_S | l2 | 0.809 | V2 0.810 | +0.001 [9] | V1 0.809 | +0.000 [0] | 0.809 |
| G_S | all parameters | 0.691 | V4 0.780 | +0.090 [16] | V4 0.706 | +0.015 [16] | 0.854 |
| G_S | all but l3 | 0.753 | V4 0.777 | +0.024 [15] | V4 0.773 | +0.021 [16] | 0.912 |
| G_B | byte table | 0.722 | V4 0.771 | +0.049 [13] | V4 0.756 | +0.034 [13] | 0.915 |
| G_B | l1 | 0.657 | V4 0.714 | +0.057 [14] | V4 0.696 | +0.038 [14] | 0.936 |
| G_B | input map | 0.751 | V3 0.811 | +0.060 [16] | V4 0.767 | +0.016 [13] | 0.914 |
| G_B | recurrent map | 0.772 | V3 0.828 | +0.056 [16] | V4 0.788 | +0.016 [14] | 0.914 |
| G_B | lm | 0.766 | V1 0.766 | +0.000 [0] | V1 0.766 | +0.000 [0] | 0.766 |
| G_B | l2 | 0.761 | V1 0.761 | +0.000 [0] | V1 0.761 | +0.000 [0] | 0.761 |
| G_B | all parameters | 0.602 | V4 0.696 | +0.095 [16] | V3 0.610 | +0.008 [16] | 0.765 |
| G_B | all but l3 | 0.710 | V3 0.749 | +0.040 [16] | V4 0.723 | +0.014 [11] | 0.887 |
| F10_start | byte table | 0.731 | V4 0.796 | +0.065 [16] | V3 0.731 | +0.000 [11] | 0.829 |
| F10_start | l1 | 0.727 | V4 0.730 | +0.003 [16] | V4 0.727 | +0.001 [16] | 0.760 |
| F10_start | input map | 0.751 | V2 0.755 | +0.005 [16] | V2 0.752 | +0.001 [16] | 0.762 |
| F10_start | recurrent map | 0.744 | V2 0.749 | +0.005 [16] | V4 0.746 | +0.001 [16] | 0.756 |
| F10_start | lm | 0.867 | V4 0.868 | +0.001 [16] | V1 0.867 | +0.000 [0] | 0.867 |
| F10_start | l2 | 0.868 | V4 0.869 | +0.001 [16] | V1 0.868 | +0.000 [0] | 0.868 |
| F10_start | all parameters | 0.927 | V4 0.927 | +0.001 [16] | V2 0.927 | +0.000 [16] | 0.928 |
| F10_start | all but l3 | 0.848 | V4 0.849 | +0.001 [16] | V2 0.848 | +0.000 [16] | 0.850 |
| F10_S | byte table | 0.677 | V2 0.794 | +0.117 [15] | V4 0.778 | +0.101 [14] | 0.844 |
| F10_S | l1 | 0.537 | V4 0.659 | +0.123 [15] | V4 0.693 | +0.156 [16] | 0.877 |
| F10_S | input map | 0.647 | V2 0.696 | +0.049 [16] | V4 0.710 | +0.064 [16] | 0.829 |
| F10_S | recurrent map | 0.685 | V2 0.730 | +0.044 [16] | V4 0.738 | +0.053 [16] | 0.837 |
| F10_S | lm | 0.837 | V2 0.837 | +0.000 [10] | V1 0.837 | +0.000 [0] | 0.837 |
| F10_S | l2 | 0.840 | V2 0.840 | +0.000 [10] | V1 0.840 | +0.000 [0] | 0.840 |
| F10_S | all parameters | 0.722 | V2 0.781 | +0.060 [16] | V3 0.736 | +0.014 [16] | 0.802 |
| F10_S | all but l3 | 0.656 | V2 0.685 | +0.028 [15] | V4 0.702 | +0.046 [16] | 0.825 |
| F10_B | byte table | 0.589 | V4 0.724 | +0.135 [16] | V4 0.683 | +0.094 [14] | 0.812 |
| F10_B | l1 | 0.545 | V4 0.644 | +0.099 [16] | V4 0.644 | +0.099 [16] | 0.857 |
| F10_B | input map | 0.585 | V4 0.683 | +0.098 [16] | V4 0.648 | +0.063 [16] | 0.793 |
| F10_B | recurrent map | 0.595 | V4 0.695 | +0.100 [16] | V4 0.657 | +0.062 [16] | 0.797 |
| F10_B | lm | 0.745 | V2 0.747 | +0.001 [12] | V1 0.745 | +0.000 [0] | 0.745 |
| F10_B | l2 | 0.749 | V2 0.750 | +0.001 [12] | V1 0.749 | +0.000 [0] | 0.749 |
| F10_B | all parameters | 0.694 | V4 0.736 | +0.043 [16] | V3 0.701 | +0.007 [16] | 0.766 |
| F10_B | all but l3 | 0.601 | V2 0.616 | +0.015 [11] | V4 0.630 | +0.029 [15] | 0.777 |
| in16_start | byte table | 0.558 | V3 0.578 | +0.020 [16] | V3 0.562 | +0.005 [10] | 0.525 |
| in16_start | l1 | 0.716 | V4 0.722 | +0.005 [16] | V4 0.720 | +0.004 [16] | 0.748 |
| in16_start | input map | 0.745 | V2 0.749 | +0.004 [16] | V4 0.746 | +0.000 [10] | 0.756 |
| in16_start | recurrent map | 0.734 | V2 0.739 | +0.006 [16] | V2 0.734 | +0.001 [13] | 0.743 |
| in16_start | lm | 0.868 | V4 0.869 | +0.001 [16] | V1 0.868 | +0.000 [0] | 0.868 |
| in16_start | l2 | 0.868 | V4 0.869 | +0.001 [16] | V1 0.868 | +0.000 [0] | 0.868 |
| in16_start | all parameters | 0.941 | V4 0.941 | +0.000 [16] | V4 0.941 | +0.000 [16] | 0.941 |
| in16_start | all but l3 | 0.864 | V4 0.865 | +0.001 [16] | V4 0.864 | +0.000 [16] | 0.864 |
| in16_S | byte table | 0.367 | V4 0.741 | +0.373 [16] | V4 0.638 | +0.271 [16] | 0.738 |
| in16_S | l1 | 0.298 | V4 0.714 | +0.416 [16] | V4 0.721 | +0.422 [16] | 0.839 |
| in16_S | input map | 0.603 | V4 0.729 | +0.126 [16] | V3 0.715 | +0.111 [16] | 0.825 |
| in16_S | recurrent map | 0.651 | V2 0.762 | +0.110 [16] | V4 0.710 | +0.059 [16] | 0.787 |
| in16_S | lm | 0.606 | V2 0.694 | +0.089 [16] | V1 0.606 | +0.000 [0] | 0.606 |
| in16_S | l2 | 0.643 | V2 0.710 | +0.068 [16] | V1 0.643 | +0.000 [0] | 0.643 |
| in16_S | all parameters | 0.612 | V4 0.700 | +0.088 [15] | V4 0.706 | +0.094 [16] | 0.782 |
| in16_S | all but l3 | 0.587 | V2 0.742 | +0.155 [16] | V4 0.689 | +0.102 [16] | 0.766 |
| in16_B | byte table | 0.442 | V2 0.405 | -0.036 [7] | V3 0.445 | +0.004 [8] | 0.738 |
| in16_B | l1 | 0.309 | V2 0.363 | +0.055 [16] | V2 0.331 | +0.022 [14] | 0.742 |
| in16_B | input map | 0.523 | V2 0.567 | +0.043 [16] | V4 0.562 | +0.038 [16] | 0.761 |
| in16_B | recurrent map | 0.582 | V2 0.631 | +0.049 [16] | V4 0.608 | +0.026 [16] | 0.756 |
| in16_B | lm | 0.700 | V2 0.700 | +0.001 [12] | V1 0.700 | +0.000 [0] | 0.700 |
| in16_B | l2 | 0.706 | V2 0.707 | +0.001 [11] | V1 0.706 | +0.000 [0] | 0.706 |
| in16_B | all parameters | 0.640 | V2 0.670 | +0.030 [16] | V4 0.642 | +0.002 [12] | 0.758 |
| in16_B | all but l3 | 0.496 | V2 0.511 | +0.015 [16] | V4 0.510 | +0.014 [15] | 0.732 |
| C-dense_start | input map | 0.752 | V2 0.758 | +0.005 [16] | V4 0.754 | +0.001 [16] | 0.761 |
| C-dense_start | recurrent map | 0.726 | V2 0.732 | +0.005 [16] | V2 0.728 | +0.001 [16] | 0.734 |
| C-dense_start | lm | 0.869 | V4 0.870 | +0.001 [16] | V1 0.869 | +0.000 [0] | 0.869 |
| C-dense_start | l2 | 0.870 | V4 0.871 | +0.001 [16] | V1 0.870 | +0.000 [0] | 0.870 |
| C-dense_start | all parameters | 0.933 | V4 0.934 | +0.000 [16] | V4 0.933 | +0.000 [16] | 0.934 |
| C-dense_start | all but l3 | 0.852 | V4 0.854 | +0.001 [16] | V4 0.853 | +0.000 [16] | 0.854 |
| C-dense_S | input map | 0.735 | V4 0.817 | +0.082 [16] | V4 0.791 | +0.056 [16] | 0.852 |
| C-dense_S | recurrent map | 0.713 | V4 0.794 | +0.081 [16] | V4 0.772 | +0.059 [16] | 0.848 |
| C-dense_S | lm | 0.880 | V2 0.881 | +0.000 [7] | V1 0.880 | +0.000 [0] | 0.880 |
| C-dense_S | l2 | 0.879 | V2 0.880 | +0.000 [8] | V1 0.879 | +0.000 [0] | 0.879 |
| C-dense_S | all parameters | 0.640 | V4 0.793 | +0.154 [16] | V4 0.668 | +0.029 [16] | 0.710 |
| C-dense_S | all but l3 | 0.736 | V4 0.768 | +0.032 [14] | V4 0.782 | +0.046 [16] | 0.838 |
| C-dense_B | input map | 0.637 | V4 0.756 | +0.119 [16] | V4 0.694 | +0.057 [16] | 0.779 |
| C-dense_B | recurrent map | 0.629 | V4 0.751 | +0.123 [16] | V4 0.691 | +0.063 [16] | 0.796 |
| C-dense_B | lm | 0.762 | V1 0.762 | +0.000 [0] | V1 0.762 | +0.000 [0] | 0.762 |
| C-dense_B | l2 | 0.758 | V1 0.758 | +0.000 [0] | V1 0.758 | +0.000 [0] | 0.758 |
| C-dense_B | all parameters | 0.528 | V4 0.699 | +0.170 [16] | V4 0.562 | +0.034 [16] | 0.626 |
| C-dense_B | all but l3 | 0.613 | V4 0.702 | +0.089 [16] | V4 0.664 | +0.051 [16] | 0.747 |

### 6.2 Every variant against V0 on the same batches

Mean difference of the cosine from V0 over the 16 batches, and in square brackets the number of batches on which the variant is above V0.

| network | tensor | V0 | V1, every map | V2, every map | V3, every map | V4, every map | V2, GRU maps | V3, GRU maps | V4, GRU maps |
|---|---|---|---|---|---|---|---|---|---|
| G_S | byte table | 0.722 | -0.007 [2] | +0.067 [14] | +0.032 [12] | +0.117 [16] | +0.015 [11] | +0.041 [16] | +0.060 [16] |
| G_S | l1 | 0.661 | -0.002 [6] | +0.029 [15] | +0.037 [13] | +0.100 [16] | +0.026 [15] | +0.053 [16] | +0.079 [16] |
| G_S | input map | 0.788 | +0.001 [9] | +0.002 [10] | +0.017 [12] | +0.023 [14] | +0.002 [10] | +0.009 [16] | +0.012 [16] |
| G_S | recurrent map | 0.819 | +0.001 [8] | +0.002 [9] | +0.015 [12] | +0.022 [15] | +0.002 [11] | +0.008 [16] | +0.011 [16] |
| G_S | lm | 0.809 | +0.000 [0] | +0.000 [10] | -0.034 [3] | -0.035 [3] | +0.000 [0] | +0.000 [0] | +0.000 [0] |
| G_S | l2 | 0.809 | +0.000 [0] | +0.001 [9] | -0.037 [3] | -0.037 [3] | +0.000 [0] | +0.000 [0] | +0.000 [0] |
| G_S | all parameters | 0.691 | -0.001 [5] | +0.045 [16] | +0.046 [15] | +0.090 [16] | +0.002 [10] | +0.013 [16] | +0.015 [16] |
| G_S | all but l3 | 0.753 | -0.002 [6] | +0.003 [9] | +0.013 [10] | +0.024 [15] | +0.003 [11] | +0.017 [16] | +0.021 [16] |
| G_B | byte table | 0.722 | +0.006 [11] | +0.022 [12] | +0.005 [9] | +0.049 [13] | +0.024 [13] | +0.005 [13] | +0.034 [13] |
| G_B | l1 | 0.657 | +0.010 [16] | +0.026 [12] | +0.013 [12] | +0.057 [14] | +0.027 [13] | +0.009 [13] | +0.038 [14] |
| G_B | input map | 0.751 | +0.006 [16] | -0.001 [7] | +0.060 [16] | +0.056 [16] | +0.001 [7] | +0.014 [15] | +0.016 [13] |
| G_B | recurrent map | 0.772 | +0.006 [16] | -0.001 [7] | +0.056 [16] | +0.052 [16] | +0.002 [6] | +0.014 [16] | +0.016 [14] |
| G_B | lm | 0.766 | +0.000 [0] | -0.001 [8] | -0.065 [1] | -0.064 [1] | +0.000 [0] | +0.000 [0] | +0.000 [0] |
| G_B | l2 | 0.761 | +0.000 [0] | -0.000 [7] | -0.073 [2] | -0.071 [2] | +0.000 [0] | +0.000 [0] | +0.000 [0] |
| G_B | all parameters | 0.602 | +0.003 [11] | +0.033 [16] | +0.059 [16] | +0.095 [16] | -0.002 [6] | +0.008 [16] | +0.007 [11] |
| G_B | all but l3 | 0.710 | +0.005 [16] | -0.003 [6] | +0.040 [16] | +0.038 [16] | +0.001 [8] | +0.013 [16] | +0.014 [11] |
| F10_start | byte table | 0.731 | -0.000 [3] | +0.047 [16] | +0.013 [16] | +0.065 [16] | -0.004 [0] | +0.000 [11] | -0.004 [0] |
| F10_start | l1 | 0.727 | -0.000 [0] | +0.002 [16] | +0.001 [14] | +0.003 [16] | +0.000 [16] | +0.000 [11] | +0.001 [16] |
| F10_start | input map | 0.751 | +0.000 [16] | +0.005 [16] | -0.003 [0] | +0.001 [16] | +0.001 [16] | -0.000 [1] | +0.001 [16] |
| F10_start | recurrent map | 0.744 | +0.000 [16] | +0.005 [16] | -0.003 [0] | +0.002 [16] | +0.001 [16] | -0.000 [4] | +0.001 [16] |
| F10_start | lm | 0.867 | +0.000 [0] | +0.000 [16] | +0.001 [16] | +0.001 [16] | +0.000 [0] | +0.000 [0] | +0.000 [0] |
| F10_start | l2 | 0.868 | +0.000 [0] | +0.000 [15] | +0.001 [16] | +0.001 [16] | +0.000 [0] | +0.000 [0] | +0.000 [0] |
| F10_start | all parameters | 0.927 | +0.000 [16] | +0.000 [16] | +0.000 [16] | +0.001 [16] | +0.000 [16] | -0.000 [1] | +0.000 [16] |
| F10_start | all but l3 | 0.848 | +0.000 [16] | +0.001 [16] | +0.000 [16] | +0.001 [16] | +0.000 [16] | -0.000 [1] | +0.000 [16] |
| F10_S | byte table | 0.677 | +0.016 [12] | +0.117 [15] | +0.003 [9] | +0.116 [15] | +0.040 [13] | +0.040 [16] | +0.101 [14] |
| F10_S | l1 | 0.537 | +0.057 [13] | +0.094 [16] | +0.003 [10] | +0.123 [15] | +0.083 [15] | +0.058 [16] | +0.156 [16] |
| F10_S | input map | 0.647 | +0.017 [16] | +0.049 [16] | -0.018 [4] | +0.028 [14] | +0.036 [16] | +0.022 [16] | +0.064 [16] |
| F10_S | recurrent map | 0.685 | +0.015 [16] | +0.044 [16] | -0.027 [3] | +0.012 [10] | +0.030 [16] | +0.019 [16] | +0.053 [16] |
| F10_S | lm | 0.837 | +0.000 [0] | +0.000 [10] | -0.118 [0] | -0.118 [0] | +0.000 [0] | +0.000 [0] | +0.000 [0] |
| F10_S | l2 | 0.840 | +0.000 [0] | +0.000 [10] | -0.116 [0] | -0.116 [0] | +0.000 [0] | +0.000 [0] | +0.000 [0] |
| F10_S | all parameters | 0.722 | -0.005 [1] | +0.060 [16] | +0.004 [10] | +0.049 [16] | -0.003 [5] | +0.014 [16] | +0.010 [15] |
| F10_S | all but l3 | 0.656 | +0.005 [10] | +0.028 [15] | -0.051 [0] | -0.013 [5] | +0.013 [13] | +0.027 [16] | +0.046 [16] |
| F10_B | byte table | 0.589 | +0.078 [13] | +0.088 [14] | +0.027 [12] | +0.135 [16] | +0.076 [14] | +0.026 [15] | +0.094 [14] |
| F10_B | l1 | 0.545 | +0.055 [15] | +0.085 [16] | +0.000 [8] | +0.099 [16] | +0.074 [16] | +0.025 [15] | +0.099 [16] |
| F10_B | input map | 0.585 | +0.012 [16] | +0.042 [16] | +0.054 [15] | +0.098 [16] | +0.025 [16] | +0.030 [16] | +0.063 [16] |
| F10_B | recurrent map | 0.595 | +0.011 [16] | +0.041 [16] | +0.059 [16] | +0.100 [16] | +0.023 [16] | +0.031 [16] | +0.062 [16] |
| F10_B | lm | 0.745 | +0.000 [0] | +0.001 [12] | -0.066 [0] | -0.065 [0] | +0.000 [0] | +0.000 [0] | +0.000 [0] |
| F10_B | l2 | 0.749 | +0.000 [0] | +0.001 [12] | -0.068 [0] | -0.067 [0] | +0.000 [0] | +0.000 [0] | +0.000 [0] |
| F10_B | all parameters | 0.694 | -0.006 [3] | +0.040 [16] | +0.009 [14] | +0.043 [16] | -0.008 [2] | +0.007 [16] | -0.000 [5] |
| F10_B | all but l3 | 0.601 | +0.009 [13] | +0.015 [11] | -0.017 [2] | -0.001 [8] | +0.010 [9] | +0.017 [16] | +0.029 [15] |
| in16_start | byte table | 0.558 | +0.000 [10] | -0.017 [4] | +0.020 [16] | +0.014 [14] | -0.026 [0] | +0.005 [10] | -0.023 [0] |
| in16_start | l1 | 0.716 | +0.000 [14] | +0.004 [16] | +0.002 [16] | +0.005 [16] | +0.002 [16] | +0.002 [16] | +0.004 [16] |
| in16_start | input map | 0.745 | +0.000 [16] | +0.004 [16] | -0.003 [0] | +0.000 [9] | +0.000 [11] | -0.000 [3] | +0.000 [10] |
| in16_start | recurrent map | 0.734 | +0.000 [16] | +0.006 [16] | -0.005 [0] | +0.001 [12] | +0.001 [13] | -0.001 [0] | +0.000 [10] |
| in16_start | lm | 0.868 | +0.000 [0] | +0.000 [16] | +0.001 [16] | +0.001 [16] | +0.000 [0] | +0.000 [0] | +0.000 [0] |
| in16_start | l2 | 0.868 | +0.000 [0] | +0.000 [15] | +0.001 [16] | +0.001 [16] | +0.000 [0] | +0.000 [0] | +0.000 [0] |
| in16_start | all parameters | 0.941 | +0.000 [13] | +0.000 [16] | +0.000 [16] | +0.000 [16] | +0.000 [16] | +0.000 [14] | +0.000 [16] |
| in16_start | all but l3 | 0.864 | +0.000 [12] | +0.000 [16] | +0.001 [16] | +0.001 [16] | +0.000 [16] | +0.000 [14] | +0.000 [16] |
| in16_S | byte table | 0.367 | -0.050 [2] | +0.329 [16] | +0.109 [16] | +0.373 [16] | +0.106 [16] | +0.098 [16] | +0.271 [16] |
| in16_S | l1 | 0.298 | -0.046 [0] | +0.326 [16] | +0.178 [16] | +0.416 [16] | +0.229 [16] | +0.257 [16] | +0.422 [16] |
| in16_S | input map | 0.603 | -0.043 [1] | +0.072 [16] | +0.037 [13] | +0.126 [16] | -0.009 [6] | +0.111 [16] | +0.110 [16] |
| in16_S | recurrent map | 0.651 | -0.018 [4] | +0.110 [16] | -0.010 [8] | +0.103 [16] | +0.002 [7] | +0.058 [16] | +0.059 [16] |
| in16_S | lm | 0.606 | +0.000 [0] | +0.089 [16] | -0.034 [4] | -0.018 [5] | +0.000 [0] | +0.000 [0] | +0.000 [0] |
| in16_S | l2 | 0.643 | +0.000 [0] | +0.068 [16] | -0.041 [1] | -0.031 [2] | +0.000 [0] | +0.000 [0] | +0.000 [0] |
| in16_S | all parameters | 0.612 | -0.016 [3] | +0.025 [8] | -0.040 [6] | +0.088 [15] | +0.025 [14] | +0.073 [15] | +0.094 [16] |
| in16_S | all but l3 | 0.587 | +0.002 [10] | +0.155 [16] | -0.043 [6] | +0.141 [16] | +0.047 [16] | +0.077 [16] | +0.102 [16] |
| in16_B | byte table | 0.442 | -0.084 [0] | -0.036 [7] | -0.072 [1] | -0.082 [3] | -0.051 [3] | +0.004 [8] | -0.059 [3] |
| in16_B | l1 | 0.309 | -0.014 [3] | +0.055 [16] | -0.036 [1] | +0.019 [12] | +0.022 [14] | -0.015 [2] | +0.013 [11] |
| in16_B | input map | 0.523 | +0.008 [16] | +0.043 [16] | -0.020 [6] | +0.031 [12] | +0.013 [15] | +0.021 [15] | +0.038 [16] |
| in16_B | recurrent map | 0.582 | +0.006 [16] | +0.049 [16] | -0.031 [5] | +0.026 [9] | +0.011 [16] | +0.013 [15] | +0.026 [16] |
| in16_B | lm | 0.700 | +0.000 [0] | +0.001 [12] | -0.097 [0] | -0.097 [0] | +0.000 [0] | +0.000 [0] | +0.000 [0] |
| in16_B | l2 | 0.706 | +0.000 [0] | +0.001 [11] | -0.095 [0] | -0.094 [0] | +0.000 [0] | +0.000 [0] | +0.000 [0] |
| in16_B | all parameters | 0.640 | -0.001 [4] | +0.030 [16] | -0.011 [1] | +0.011 [13] | -0.001 [4] | +0.001 [11] | +0.002 [12] |
| in16_B | all but l3 | 0.496 | +0.005 [14] | +0.015 [16] | -0.066 [0] | -0.045 [1] | +0.006 [14] | +0.004 [12] | +0.014 [15] |
| C-dense_start | input map | 0.752 | +0.000 [16] | +0.005 [16] | -0.003 [0] | +0.003 [16] | +0.001 [16] | -0.000 [3] | +0.001 [16] |
| C-dense_start | recurrent map | 0.726 | +0.000 [16] | +0.005 [16] | -0.004 [0] | +0.002 [16] | +0.001 [16] | -0.000 [1] | +0.001 [16] |
| C-dense_start | lm | 0.869 | +0.000 [0] | +0.000 [16] | +0.001 [16] | +0.001 [16] | +0.000 [0] | +0.000 [0] | +0.000 [0] |
| C-dense_start | l2 | 0.870 | +0.000 [0] | +0.000 [13] | +0.001 [16] | +0.001 [16] | +0.000 [0] | +0.000 [0] | +0.000 [0] |
| C-dense_start | all parameters | 0.933 | +0.000 [16] | +0.000 [16] | +0.000 [16] | +0.000 [16] | +0.000 [16] | -0.000 [3] | +0.000 [16] |
| C-dense_start | all but l3 | 0.852 | +0.000 [16] | +0.001 [16] | +0.000 [16] | +0.001 [16] | +0.000 [16] | -0.000 [3] | +0.000 [16] |
| C-dense_S | input map | 0.735 | +0.009 [16] | +0.027 [16] | +0.049 [16] | +0.082 [16] | +0.020 [16] | +0.027 [16] | +0.056 [16] |
| C-dense_S | recurrent map | 0.713 | +0.011 [16] | +0.031 [16] | +0.042 [16] | +0.081 [16] | +0.024 [16] | +0.027 [16] | +0.059 [16] |
| C-dense_S | lm | 0.880 | +0.000 [0] | +0.000 [7] | -0.100 [0] | -0.101 [0] | +0.000 [0] | +0.000 [0] | +0.000 [0] |
| C-dense_S | l2 | 0.879 | +0.000 [0] | +0.000 [8] | -0.101 [0] | -0.101 [0] | +0.000 [0] | +0.000 [0] | +0.000 [0] |
| C-dense_S | all parameters | 0.640 | +0.006 [16] | +0.136 [16] | +0.024 [15] | +0.154 [16] | +0.012 [16] | +0.010 [16] | +0.029 [16] |
| C-dense_S | all but l3 | 0.736 | +0.009 [16] | +0.022 [16] | +0.001 [9] | +0.032 [14] | +0.018 [16] | +0.020 [16] | +0.046 [16] |
| C-dense_B | input map | 0.637 | +0.010 [16] | +0.037 [16] | +0.082 [16] | +0.119 [16] | +0.031 [16] | +0.018 [16] | +0.057 [16] |
| C-dense_B | recurrent map | 0.629 | +0.011 [16] | +0.038 [16] | +0.079 [16] | +0.123 [16] | +0.034 [16] | +0.021 [16] | +0.063 [16] |
| C-dense_B | lm | 0.762 | +0.000 [0] | -0.001 [7] | -0.098 [0] | -0.100 [0] | +0.000 [0] | +0.000 [0] | +0.000 [0] |
| C-dense_B | l2 | 0.758 | +0.000 [0] | -0.000 [6] | -0.099 [0] | -0.101 [0] | +0.000 [0] | +0.000 [0] | +0.000 [0] |
| C-dense_B | all parameters | 0.528 | +0.007 [16] | +0.103 [16] | +0.057 [15] | +0.170 [16] | +0.019 [16] | +0.008 [16] | +0.034 [16] |
| C-dense_B | all but l3 | 0.613 | +0.010 [16] | +0.032 [16] | +0.049 [15] | +0.089 [16] | +0.028 [16] | +0.015 [16] | +0.051 [16] |

What the two tables show.

- **V1.** For the recurrent map the difference from V0 is between −0.018 and +0.015, for the input map between −0.043 and +0.017. It is positive on all 16 batches in six of the eight trained networks, but small; it is negative in `in16_S`.
- **V2 at every map.** For the recurrent map: +0.044 (`F10_S`), +0.041 (`F10_B`), +0.110 (`in16_S`), +0.049 (`in16_B`), +0.031 (`C-dense_S`), +0.038 (`C-dense_B`), each on all 16 batches; +0.002 (`G_S`) and −0.001 (`G_B`) in the dense network. `lm` and `l2` change by 0.001 or less, except in `in16_S` (+0.089 and +0.068).
- **V3 and V4 at every map** lower `lm` and `l2` in every trained network, by 0.034 to 0.118 (V3) and 0.018 to 0.118 (V4); on the two GRU maps V4 is above V0 in every trained network (+0.012 to +0.126) and V3 is not (−0.031 to +0.082).
- **V3 and V4 at the two GRU maps only** leave `lm` and `l2` unchanged, as they must. V4 raises the recurrent map by +0.011 to +0.063 and the input map by +0.012 to +0.110, on at least 13 of the 16 batches in every trained network.
- **The byte table and `l1`** gain more than the GRU maps under V2 and V4 in most networks (up to +0.42 in `in16_S`), but not in `in16_B`, where the byte table loses under every variant except V3 at the GRU maps.
- **At the start** (`F10_start`, `in16_start`, `C-dense_start`) no variant moves the cosine of a GRU map, of `lm` or of `l2` by more than 0.006. The byte table moves by up to 0.065; the rows of `l1` hold 8 weights each, so that their means differ even at the start (coefficient of variation 0.20, section 6.3).
- **Between batches** the cosines vary widely once a network is trained, as does the loss itself (in `G_S` between 0.0054 and 0.0519 on the 16 batches). In `G_S`, for example, the cosine of the byte table under V0 is 0.493 on the first batch and between 0.621 and 0.810 on the 16 that follow.
- **Networks trained under the sweep** have higher cosines under V0 than the same networks trained under back-propagation, in each of the four pairs (recurrent map: 0.819 against 0.772, 0.685 against 0.595, 0.651 against 0.582, 0.713 against 0.629).

### 6.3 How unequal the magnitudes are that a scale replaces

A variant can change the gradient only where the magnitudes it distinguishes differ. For every map of every network the table gives the number of kept entries, the mean magnitude, the mean of each gate's block, the coefficient of variation of the row means (and the same within each gate's block), of the column means and of the magnitudes themselves, and `cos(F, W)` for each variant. The line under each table concerns `l3`: the share of the squared error at the logits that falls on the 11 rows of the classes that occur as targets, the mean magnitude of `l3` on those rows and on the other 246, and `cos(F, W)` on those 11 rows only.

CV: coefficient of variation, the standard deviation over the mean.  cos(F, W): the cosine between the matrix a variant sends the error back through (the signs times the variant's magnitudes) and the weight matrix, over the kept entries.

**G_S** (E42 G (dense), trained under the sweep)

| map | kept entries | mean magnitude | mean by gate block r / z / n | CV of row means | the same within block r / z / n | CV of column means | CV of magnitudes | cos(F, W) V0 | V1 | V2 | V3 | V4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| l1 | 4,096 of 4,096 | 0.0347 | – | 0.73 | – | 0.11 | 1.06 | 0.685 | 0.685 | 0.849 | 0.689 | 0.853 |
| input map | 393,216 of 393,216 | 0.0421 | 0.0504 / 0.0401 / 0.0357 | 0.40 | 0.37 / 0.37 / 0.34 | 0.30 | 0.94 | 0.727 | 0.735 | 0.782 | 0.759 | 0.819 |
| recurrent map | 196,608 of 196,608 | 0.0492 | 0.0453 / 0.0546 / 0.0477 | 0.15 | 0.17 / 0.11 / 0.09 | 0.12 | 0.84 | 0.767 | 0.769 | 0.775 | 0.773 | 0.781 |
| l2 | 262,144 of 262,144 | 0.0138 | – | 0.30 | – | 0.16 | 0.85 | 0.762 | 0.762 | 0.795 | 0.772 | 0.805 |
| lm | 131,072 of 131,072 | 0.0557 | – | 0.19 | – | 0.26 | 0.82 | 0.772 | 0.772 | 0.785 | 0.798 | 0.811 |
| l3 | 131,584 of 131,584 | 0.0564 | – | 0.17 | – | 0.71 | 0.78 | 0.789 | 0.789 | 0.800 | 0.968 | 0.966 |

l3 of G_S: 11 of its 257 rows belong to classes that occur as targets; share of the squared output error on these rows, least over the 17 batches: 1.000000; mean magnitude on these rows 0.0877, on the other rows 0.0550; cos(F, W) on these rows only: V0 0.750, V1 0.750, V2 0.797, V3 0.681, V4 0.707

**G_B** (E42 G (dense), trained under back-propagation)

| map | kept entries | mean magnitude | mean by gate block r / z / n | CV of row means | the same within block r / z / n | CV of column means | CV of magnitudes | cos(F, W) V0 | V1 | V2 | V3 | V4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| l1 | 4,096 of 4,096 | 0.0436 | – | 0.25 | – | 0.18 | 0.73 | 0.809 | 0.809 | 0.835 | 0.823 | 0.850 |
| input map | 393,216 of 393,216 | 0.0431 | 0.0525 / 0.0396 / 0.0371 | 0.43 | 0.40 / 0.40 / 0.40 | 0.09 | 0.93 | 0.732 | 0.741 | 0.798 | 0.735 | 0.801 |
| recurrent map | 196,608 of 196,608 | 0.0438 | 0.0387 / 0.0492 / 0.0435 | 0.18 | 0.17 / 0.14 / 0.13 | 0.15 | 0.91 | 0.740 | 0.743 | 0.751 | 0.748 | 0.759 |
| l2 | 262,144 of 262,144 | 0.0109 | – | 0.32 | – | 0.22 | 0.86 | 0.758 | 0.758 | 0.796 | 0.776 | 0.814 |
| lm | 131,072 of 131,072 | 0.0469 | – | 0.08 | – | 0.38 | 0.99 | 0.711 | 0.711 | 0.713 | 0.761 | 0.763 |
| l3 | 131,584 of 131,584 | 0.0483 | – | 0.11 | – | 0.69 | 0.78 | 0.787 | 0.787 | 0.792 | 0.957 | 0.958 |

l3 of G_B: 11 of its 257 rows belong to classes that occur as targets; share of the squared output error on these rows, least over the 17 batches: 1.000000; mean magnitude on these rows 0.0662, on the other rows 0.0475; cos(F, W) on these rows only: V0 0.623, V1 0.623, V2 0.645, V3 0.591, V4 0.628

**F10_start** (E42 F10 at its start (the start file of E49's check 2, a run of F10 by E49's runner))

| map | kept entries | mean magnitude | mean by gate block r / z / n | CV of row means | the same within block r / z / n | CV of column means | CV of magnitudes | cos(F, W) V0 | V1 | V2 | V3 | V4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| l1 | 4,096 of 4,096 | 0.1773 | – | 0.20 | – | 0.03 | 0.57 | 0.868 | 0.868 | 0.886 | 0.869 | 0.887 |
| input map | 393,216 of 393,216 | 0.0313 | 0.0313 / 0.0313 / 0.0312 | 0.03 | 0.03 / 0.03 / 0.03 | 0.02 | 0.58 | 0.866 | 0.866 | 0.866 | 0.866 | 0.866 |
| recurrent map | 8,448 of 196,608 | 0.1509 | 0.1522 / 0.1491 / 0.1512 | 0.18 | 0.17 / 0.19 / 0.18 | 0.10 | 0.58 | 0.865 | 0.865 | 0.879 | 0.870 | 0.882 |
| l2 | 262,144 of 262,144 | 0.0221 | – | 0.03 | – | 0.03 | 0.58 | 0.866 | 0.866 | 0.866 | 0.866 | 0.866 |
| lm | 131,072 of 131,072 | 0.0312 | – | 0.04 | – | 0.02 | 0.58 | 0.865 | 0.865 | 0.866 | 0.866 | 0.866 |
| l3 | 131,584 of 131,584 | 0.0220 | – | 0.03 | – | 0.04 | 0.58 | 0.865 | 0.865 | 0.865 | 0.866 | 0.866 |

l3 of F10_start: 11 of its 257 rows belong to classes that occur as targets; share of the squared output error on these rows, least over the 17 batches: 0.996140; mean magnitude on these rows 0.0219, on the other rows 0.0220; cos(F, W) on these rows only: V0 0.864, V1 0.864, V2 0.865, V3 0.865, V4 0.865

**F10_S** (E42 F10 (10 neighbours and itself), trained under the sweep)

| map | kept entries | mean magnitude | mean by gate block r / z / n | CV of row means | the same within block r / z / n | CV of column means | CV of magnitudes | cos(F, W) V0 | V1 | V2 | V3 | V4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| l1 | 4,096 of 4,096 | 0.0409 | – | 0.87 | – | 0.12 | 1.21 | 0.638 | 0.638 | 0.843 | 0.642 | 0.847 |
| input map | 393,216 of 393,216 | 0.0832 | 0.1193 / 0.0965 / 0.0338 | 0.59 | 0.25 / 0.49 / 0.43 | 0.46 | 1.19 | 0.643 | 0.701 | 0.746 | 0.708 | 0.829 |
| recurrent map | 8,448 of 196,608 | 0.1882 | 0.1567 / 0.1880 / 0.2199 | 0.33 | 0.33 / 0.36 / 0.21 | 0.21 | 0.88 | 0.751 | 0.758 | 0.790 | 0.766 | 0.804 |
| l2 | 262,144 of 262,144 | 0.0240 | – | 0.29 | – | 0.17 | 0.84 | 0.766 | 0.766 | 0.796 | 0.776 | 0.810 |
| lm | 131,072 of 131,072 | 0.0785 | – | 0.11 | – | 0.23 | 0.81 | 0.776 | 0.776 | 0.780 | 0.795 | 0.800 |
| l3 | 131,584 of 131,584 | 0.0769 | – | 0.19 | – | 0.69 | 0.79 | 0.786 | 0.786 | 0.801 | 0.956 | 0.954 |

l3 of F10_S: 11 of its 257 rows belong to classes that occur as targets; share of the squared output error on these rows, least over the 17 batches: 1.000000; mean magnitude on these rows 0.1454, on the other rows 0.0739; cos(F, W) on these rows only: V0 0.791, V1 0.791, V2 0.797, V3 0.693, V4 0.703

**F10_B** (E42 F10, trained under back-propagation)

| map | kept entries | mean magnitude | mean by gate block r / z / n | CV of row means | the same within block r / z / n | CV of column means | CV of magnitudes | cos(F, W) V0 | V1 | V2 | V3 | V4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| l1 | 4,096 of 4,096 | 0.0453 | – | 0.28 | – | 0.14 | 0.76 | 0.795 | 0.795 | 0.826 | 0.802 | 0.833 |
| input map | 393,216 of 393,216 | 0.0790 | 0.1160 / 0.0857 / 0.0353 | 0.57 | 0.31 / 0.39 / 0.48 | 0.17 | 1.12 | 0.666 | 0.723 | 0.766 | 0.676 | 0.778 |
| recurrent map | 8,448 of 196,608 | 0.1776 | 0.1348 / 0.1922 / 0.2057 | 0.37 | 0.35 / 0.39 / 0.21 | 0.25 | 1.00 | 0.706 | 0.716 | 0.751 | 0.726 | 0.772 |
| l2 | 262,144 of 262,144 | 0.0194 | – | 0.30 | – | 0.22 | 0.90 | 0.744 | 0.744 | 0.777 | 0.762 | 0.797 |
| lm | 131,072 of 131,072 | 0.0658 | – | 0.13 | – | 0.30 | 0.96 | 0.723 | 0.723 | 0.729 | 0.755 | 0.762 |
| l3 | 131,584 of 131,584 | 0.0651 | – | 0.16 | – | 0.68 | 0.81 | 0.776 | 0.776 | 0.785 | 0.938 | 0.937 |

l3 of F10_B: 11 of its 257 rows belong to classes that occur as targets; share of the squared output error on these rows, least over the 17 batches: 1.000000; mean magnitude on these rows 0.1130, on the other rows 0.0629; cos(F, W) on these rows only: V0 0.659, V1 0.659, V2 0.660, V3 0.596, V4 0.601

**in16_start** (E49 in16 at its start)

| map | kept entries | mean magnitude | mean by gate block r / z / n | CV of row means | the same within block r / z / n | CV of column means | CV of magnitudes | cos(F, W) V0 | V1 | V2 | V3 | V4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| l1 | 4,096 of 4,096 | 0.1773 | – | 0.20 | – | 0.03 | 0.57 | 0.868 | 0.868 | 0.886 | 0.869 | 0.887 |
| input map | 12,288 of 393,216 | 0.1765 | 0.1759 / 0.1764 / 0.1771 | 0.14 | 0.14 / 0.14 / 0.13 | 0.13 | 0.58 | 0.865 | 0.865 | 0.873 | 0.872 | 0.879 |
| recurrent map | 8,448 of 196,608 | 0.1509 | 0.1522 / 0.1491 / 0.1512 | 0.18 | 0.17 / 0.19 / 0.18 | 0.10 | 0.58 | 0.865 | 0.865 | 0.879 | 0.870 | 0.882 |
| l2 | 262,144 of 262,144 | 0.0221 | – | 0.03 | – | 0.03 | 0.58 | 0.866 | 0.866 | 0.866 | 0.866 | 0.866 |
| lm | 131,072 of 131,072 | 0.0312 | – | 0.04 | – | 0.02 | 0.58 | 0.865 | 0.865 | 0.866 | 0.866 | 0.866 |
| l3 | 131,584 of 131,584 | 0.0220 | – | 0.03 | – | 0.04 | 0.58 | 0.865 | 0.865 | 0.865 | 0.866 | 0.866 |

l3 of in16_start: 11 of its 257 rows belong to classes that occur as targets; share of the squared output error on these rows, least over the 17 batches: 0.996142; mean magnitude on these rows 0.0219, on the other rows 0.0220; cos(F, W) on these rows only: V0 0.864, V1 0.864, V2 0.865, V3 0.865, V4 0.865

**in16_S** (E49 in16 (16 of 512 input values a cell), sweep: did not learn)

| map | kept entries | mean magnitude | mean by gate block r / z / n | CV of row means | the same within block r / z / n | CV of column means | CV of magnitudes | cos(F, W) V0 | V1 | V2 | V3 | V4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| l1 | 4,096 of 4,096 | 0.0272 | – | 1.01 | – | 0.17 | 1.41 | 0.578 | 0.578 | 0.821 | 0.586 | 0.834 |
| input map | 12,288 of 393,216 | 0.1020 | 0.1508 / 0.1130 / 0.0422 | 0.82 | 0.56 / 0.69 / 0.97 | 0.65 | 1.26 | 0.621 | 0.678 | 0.803 | 0.724 | 0.858 |
| recurrent map | 8,448 of 196,608 | 0.2061 | 0.1998 / 0.2423 / 0.1762 | 0.43 | 0.40 / 0.47 / 0.27 | 0.30 | 0.83 | 0.769 | 0.776 | 0.838 | 0.802 | 0.854 |
| l2 | 262,144 of 262,144 | 0.0229 | – | 0.52 | – | 0.27 | 1.15 | 0.656 | 0.656 | 0.740 | 0.679 | 0.784 |
| lm | 131,072 of 131,072 | 0.1035 | – | 0.67 | – | 0.27 | 1.08 | 0.678 | 0.678 | 0.816 | 0.702 | 0.843 |
| l3 | 131,584 of 131,584 | 0.1305 | – | 0.19 | – | 1.27 | 1.34 | 0.599 | 0.599 | 0.609 | 0.969 | 0.968 |

l3 of in16_S: 11 of its 257 rows belong to classes that occur as targets; share of the squared output error on these rows, least over the 17 batches: 1.000000; mean magnitude on these rows 0.0858, on the other rows 0.1325; cos(F, W) on these rows only: V0 0.405, V1 0.405, V2 0.649, V3 0.265, V4 0.264

**in16_B** (E49 in16, trained under back-propagation)

| map | kept entries | mean magnitude | mean by gate block r / z / n | CV of row means | the same within block r / z / n | CV of column means | CV of magnitudes | cos(F, W) V0 | V1 | V2 | V3 | V4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| l1 | 4,096 of 4,096 | 0.0785 | – | 0.40 | – | 0.09 | 0.88 | 0.752 | 0.752 | 0.811 | 0.756 | 0.819 |
| input map | 12,288 of 393,216 | 0.1981 | 0.2762 / 0.2341 / 0.0840 | 0.63 | 0.31 / 0.54 / 0.67 | 0.31 | 1.01 | 0.702 | 0.760 | 0.831 | 0.731 | 0.861 |
| recurrent map | 8,448 of 196,608 | 0.2055 | 0.1839 / 0.2205 / 0.2121 | 0.33 | 0.30 / 0.41 / 0.21 | 0.19 | 0.93 | 0.732 | 0.734 | 0.771 | 0.744 | 0.782 |
| l2 | 262,144 of 262,144 | 0.0150 | – | 0.44 | – | 0.32 | 0.98 | 0.715 | 0.715 | 0.780 | 0.751 | 0.821 |
| lm | 131,072 of 131,072 | 0.0774 | – | 0.12 | – | 0.32 | 0.94 | 0.728 | 0.728 | 0.733 | 0.765 | 0.771 |
| l3 | 131,584 of 131,584 | 0.0718 | – | 0.11 | – | 0.69 | 0.79 | 0.786 | 0.786 | 0.790 | 0.953 | 0.949 |

l3 of in16_B: 11 of its 257 rows belong to classes that occur as targets; share of the squared output error on these rows, least over the 17 batches: 1.000000; mean magnitude on these rows 0.1069, on the other rows 0.0702; cos(F, W) on these rows only: V0 0.656, V1 0.656, V2 0.657, V3 0.576, V4 0.577

**C-dense_start** (E50 C-dense at its start (fixed code))

| map | kept entries | mean magnitude | mean by gate block r / z / n | CV of row means | the same within block r / z / n | CV of column means | CV of magnitudes | cos(F, W) V0 | V1 | V2 | V3 | V4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| input map | 393,216 of 393,216 | 0.0313 | 0.0313 / 0.0313 / 0.0312 | 0.03 | 0.03 / 0.03 / 0.03 | 0.02 | 0.58 | 0.866 | 0.866 | 0.866 | 0.866 | 0.866 |
| recurrent map | 8,448 of 196,608 | 0.1509 | 0.1522 / 0.1491 / 0.1512 | 0.18 | 0.17 / 0.19 / 0.18 | 0.10 | 0.58 | 0.865 | 0.865 | 0.879 | 0.870 | 0.882 |
| l2 | 262,144 of 262,144 | 0.0221 | – | 0.03 | – | 0.03 | 0.58 | 0.866 | 0.866 | 0.866 | 0.866 | 0.866 |
| lm | 131,072 of 131,072 | 0.0312 | – | 0.04 | – | 0.02 | 0.58 | 0.865 | 0.865 | 0.866 | 0.866 | 0.866 |
| l3 | 131,584 of 131,584 | 0.0220 | – | 0.03 | – | 0.04 | 0.58 | 0.865 | 0.865 | 0.865 | 0.866 | 0.866 |

l3 of C-dense_start: 11 of its 257 rows belong to classes that occur as targets; share of the squared output error on these rows, least over the 17 batches: 0.996151; mean magnitude on these rows 0.0219, on the other rows 0.0220; cos(F, W) on these rows only: V0 0.864, V1 0.864, V2 0.865, V3 0.865, V4 0.865

**C-dense_S** (E50 C-dense, trained under the sweep)

| map | kept entries | mean magnitude | mean by gate block r / z / n | CV of row means | the same within block r / z / n | CV of column means | CV of magnitudes | cos(F, W) V0 | V1 | V2 | V3 | V4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| input map | 393,216 of 393,216 | 0.0862 | 0.0652 / 0.1365 / 0.0569 | 0.55 | 0.35 / 0.33 / 0.31 | 0.19 | 0.90 | 0.742 | 0.803 | 0.846 | 0.755 | 0.852 |
| recurrent map | 8,448 of 196,608 | 0.1517 | 0.1024 / 0.1691 / 0.1837 | 0.39 | 0.37 / 0.35 / 0.24 | 0.30 | 0.96 | 0.723 | 0.742 | 0.776 | 0.755 | 0.812 |
| l2 | 262,144 of 262,144 | 0.0055 | – | 0.58 | – | 1.87 | 3.32 | 0.289 | 0.289 | 0.334 | 0.612 | 0.699 |
| lm | 131,072 of 131,072 | 0.0698 | – | 0.09 | – | 0.23 | 0.88 | 0.751 | 0.751 | 0.754 | 0.770 | 0.773 |
| l3 | 131,584 of 131,584 | 0.0580 | – | 0.27 | – | 0.59 | 0.75 | 0.801 | 0.801 | 0.829 | 0.928 | 0.934 |

l3 of C-dense_S: 11 of its 257 rows belong to classes that occur as targets; share of the squared output error on these rows, least over the 17 batches: 1.000000; mean magnitude on these rows 0.1312, on the other rows 0.0547; cos(F, W) on these rows only: V0 0.787, V1 0.787, V2 0.788, V3 0.700, V4 0.701

**C-dense_B** (E50 C-dense, trained under back-propagation)

| map | kept entries | mean magnitude | mean by gate block r / z / n | CV of row means | the same within block r / z / n | CV of column means | CV of magnitudes | cos(F, W) V0 | V1 | V2 | V3 | V4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| input map | 393,216 of 393,216 | 0.0938 | 0.0848 / 0.1316 / 0.0649 | 0.46 | 0.38 / 0.31 / 0.35 | 0.18 | 0.86 | 0.757 | 0.790 | 0.833 | 0.769 | 0.839 |
| recurrent map | 8,448 of 196,608 | 0.1436 | 0.0965 / 0.1639 / 0.1704 | 0.42 | 0.44 / 0.37 / 0.28 | 0.26 | 1.01 | 0.704 | 0.723 | 0.764 | 0.727 | 0.787 |
| l2 | 262,144 of 262,144 | 0.0043 | – | 0.77 | – | 1.94 | 3.97 | 0.244 | 0.244 | 0.309 | 0.532 | 0.662 |
| lm | 131,072 of 131,072 | 0.0617 | – | 0.11 | – | 0.38 | 0.99 | 0.709 | 0.709 | 0.714 | 0.759 | 0.764 |
| l3 | 131,584 of 131,584 | 0.0542 | – | 0.17 | – | 0.65 | 0.81 | 0.778 | 0.778 | 0.790 | 0.927 | 0.927 |

l3 of C-dense_B: 11 of its 257 rows belong to classes that occur as targets; share of the squared output error on these rows, least over the 17 batches: 1.000000; mean magnitude on these rows 0.0978, on the other rows 0.0522; cos(F, W) on these rows only: V0 0.646, V1 0.646, V2 0.649, V3 0.579, V4 0.588

What these tables show.

- **At the start** the dense maps have nearly equal magnitudes everywhere (coefficients of variation of the row and column means of 0.02 to 0.04, gate blocks equal to three decimals). The masked maps and `l1` have more unequal row means (0.14 to 0.20), as is expected of means over the few entries that a row holds there (11, 16 or 8).
- **After training** the gate blocks differ, most in the input map of `F10` and `in16`, where the block of the candidate state has less than a third of the mean magnitude of the block of the reset gate (0.1193, 0.0965 and 0.0338 for `r`, `z` and `n` in `F10_S`; 0.1508, 0.1130 and 0.0422 in `in16_S`). The row means differ also within a block: the coefficient of variation of the row means within a block is 0.25 to 0.97 in the input map and 0.09 to 0.47 in the recurrent map of the trained networks.
- **A scale for each gate of the input map affects few gradients.** The matrix `F` of the input map is used only where the error passes from the recurrent layer back to `l1` and the byte table. The gradients of the two GRU maps depend on the matrix `F` of the recurrent map, whose blocks differ by a factor of 1.8 at most. This agrees with the small effect of V1 on the GRU maps, although the blocks of the input map differ by a factor of up to 3.6.
- **`cos(F, W)` does not predict the cosine of the gradients.** For `l3` of the trained networks the column scale raises `cos(F, W)` from between 0.60 and 0.80 to between 0.93 and 0.97, yet V3 and V4 at `l3` lower the cosines of `lm` and `l2`. After training the whole of the error at the logits (1.000000 of its square) lies on the 11 rows of the classes that occur as targets. A column mean is taken over all 257 rows, and on those 11 rows the matrix `F` of V3 is further from `l3` than the matrix of V0 in every trained network (0.681 against 0.750 in `G_S`). In `in16_S` the same 11 rows have smaller magnitudes than the other rows (0.0858 against 0.1325), `cos(F, W)` on them is low under V0 (0.405) and higher under the row scale (0.649); this is the network in which V2 raises the cosines of `lm` and `l2`.

### 6.4 The tables for each network

For each network there are four tables: the cosine with the exact gradient for each tensor with the variant at every map; the same with the variant at the two GRU maps only (V1 is repeated there, and V5 at the GRU maps is the upper limit for any scale at those maps); the norm of the sweep's gradient over the norm of the exact gradient; and the cosine for each gate's block of the two GRU maps. Each cell is the mean over batches 2 to 17 with the least and the greatest value in brackets. The first row of the first table gives V0 on the first batch, the batch of the receipts.

#### G_S

E42 G (dense), trained under the sweep. Loss on the 16 batches: mean 0.0158 (least 0.0054, greatest 0.0519); first batch 0.0324. Norm of the exact gradient: mean 0.315 (0.084 to 0.965).

G_S, cosine with the exact gradient, variant at every map:

| variant | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters | all but l3 |
|---|---|---|---|---|---|---|---|---|---|
| V0, first batch | 0.493 | 0.584 | 0.711 | 0.745 | 0.783 | 0.795 | 1.000 | 0.681 | 0.689 |
| V0 | 0.722 (0.621 to 0.810) | 0.661 (0.570 to 0.772) | 0.788 (0.681 to 0.837) | 0.819 (0.735 to 0.852) | 0.809 (0.749 to 0.855) | 0.809 (0.726 to 0.868) | 1.000 (1.000 to 1.000) | 0.691 (0.627 to 0.754) | 0.753 (0.686 to 0.795) |
| V1 | 0.715 (0.618 to 0.810) | 0.659 (0.569 to 0.771) | 0.789 (0.687 to 0.836) | 0.820 (0.739 to 0.854) | 0.809 (0.749 to 0.855) | 0.809 (0.726 to 0.868) | 1.000 (1.000 to 1.000) | 0.690 (0.622 to 0.755) | 0.751 (0.685 to 0.794) |
| V2 | 0.789 (0.643 to 0.886) | 0.690 (0.606 to 0.788) | 0.791 (0.670 to 0.837) | 0.821 (0.724 to 0.856) | 0.810 (0.749 to 0.854) | 0.810 (0.727 to 0.868) | 1.000 (1.000 to 1.000) | 0.736 (0.688 to 0.789) | 0.755 (0.673 to 0.793) |
| V3 | 0.755 (0.680 to 0.843) | 0.697 (0.584 to 0.815) | 0.805 (0.750 to 0.855) | 0.834 (0.800 to 0.874) | 0.775 (0.694 to 0.831) | 0.772 (0.711 to 0.822) | 1.000 (1.000 to 1.000) | 0.737 (0.699 to 0.787) | 0.766 (0.719 to 0.803) |
| V4 | 0.840 (0.743 to 0.899) | 0.760 (0.653 to 0.866) | 0.811 (0.755 to 0.853) | 0.841 (0.807 to 0.888) | 0.775 (0.696 to 0.830) | 0.772 (0.712 to 0.822) | 1.000 (1.000 to 1.000) | 0.780 (0.750 to 0.825) | 0.777 (0.741 to 0.809) |

G_S, cosine with the exact gradient, variant at the two GRU maps only (V0 at the other maps; V5 here is the upper limit for a scale at the GRU maps):

| variant | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters | all but l3 |
|---|---|---|---|---|---|---|---|---|---|
| V1 | 0.715 (0.618 to 0.810) | 0.659 (0.569 to 0.771) | 0.789 (0.687 to 0.836) | 0.820 (0.739 to 0.854) | 0.809 (0.749 to 0.855) | 0.809 (0.726 to 0.868) | 1.000 (1.000 to 1.000) | 0.690 (0.622 to 0.755) | 0.751 (0.685 to 0.794) |
| V2 | 0.737 (0.637 to 0.834) | 0.687 (0.587 to 0.791) | 0.790 (0.688 to 0.834) | 0.822 (0.740 to 0.854) | 0.809 (0.749 to 0.855) | 0.809 (0.726 to 0.868) | 1.000 (1.000 to 1.000) | 0.693 (0.630 to 0.755) | 0.755 (0.687 to 0.792) |
| V3 | 0.763 (0.681 to 0.844) | 0.714 (0.615 to 0.816) | 0.798 (0.695 to 0.846) | 0.828 (0.749 to 0.860) | 0.809 (0.749 to 0.855) | 0.809 (0.726 to 0.868) | 1.000 (1.000 to 1.000) | 0.704 (0.637 to 0.769) | 0.770 (0.708 to 0.808) |
| V4 | 0.783 (0.672 to 0.851) | 0.740 (0.633 to 0.840) | 0.800 (0.701 to 0.843) | 0.830 (0.755 to 0.863) | 0.809 (0.749 to 0.855) | 0.809 (0.726 to 0.868) | 1.000 (1.000 to 1.000) | 0.706 (0.641 to 0.770) | 0.773 (0.707 to 0.810) |
| V5 | 0.912 (0.854 to 0.944) | 0.942 (0.868 to 0.974) | 0.932 (0.861 to 0.959) | 0.925 (0.859 to 0.953) | 0.809 (0.749 to 0.855) | 0.809 (0.726 to 0.868) | 1.000 (1.000 to 1.000) | 0.854 (0.789 to 0.922) | 0.912 (0.846 to 0.945) |

G_S, norm of the sweep's gradient over the norm of the exact gradient:

| variant | where | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters |
|---|---|---|---|---|---|---|---|---|---|
| V0 | every map | 0.08 (0.05 to 0.15) | 0.15 (0.09 to 0.23) | 0.24 (0.18 to 0.31) | 0.26 (0.22 to 0.34) | 0.48 (0.40 to 0.63) | 0.49 (0.38 to 0.65) | 1.00 (1.00 to 1.00) | 0.41 (0.31 to 0.55) |
| V1 | every map | 0.07 (0.05 to 0.13) | 0.14 (0.08 to 0.22) | 0.24 (0.18 to 0.32) | 0.27 (0.22 to 0.34) | 0.48 (0.40 to 0.63) | 0.49 (0.38 to 0.65) | 1.00 (1.00 to 1.00) | 0.41 (0.31 to 0.55) |
| V2 | every map | 0.12 (0.08 to 0.17) | 0.18 (0.12 to 0.27) | 0.32 (0.27 to 0.42) | 0.35 (0.31 to 0.43) | 0.65 (0.56 to 0.86) | 0.67 (0.52 to 0.89) | 1.00 (1.00 to 1.00) | 0.47 (0.36 to 0.58) |
| V3 | every map | 0.11 (0.08 to 0.15) | 0.20 (0.14 to 0.27) | 0.30 (0.24 to 0.34) | 0.33 (0.27 to 0.38) | 0.61 (0.50 to 0.75) | 0.62 (0.47 to 0.76) | 1.00 (1.00 to 1.00) | 0.46 (0.34 to 0.60) |
| V4 | every map | 0.19 (0.15 to 0.25) | 0.24 (0.19 to 0.29) | 0.40 (0.28 to 0.45) | 0.45 (0.32 to 0.50) | 0.84 (0.66 to 1.02) | 0.85 (0.63 to 1.04) | 1.00 (1.00 to 1.00) | 0.53 (0.40 to 0.66) |
| V2 | GRU maps | 0.07 (0.04 to 0.11) | 0.14 (0.09 to 0.22) | 0.25 (0.18 to 0.31) | 0.27 (0.22 to 0.34) | 0.48 (0.40 to 0.63) | 0.49 (0.38 to 0.65) | 1.00 (1.00 to 1.00) | 0.41 (0.31 to 0.55) |
| V3 | GRU maps | 0.09 (0.06 to 0.16) | 0.17 (0.10 to 0.25) | 0.25 (0.18 to 0.32) | 0.27 (0.22 to 0.35) | 0.48 (0.40 to 0.63) | 0.49 (0.38 to 0.65) | 1.00 (1.00 to 1.00) | 0.42 (0.31 to 0.55) |
| V4 | GRU maps | 0.08 (0.05 to 0.12) | 0.16 (0.10 to 0.25) | 0.25 (0.18 to 0.32) | 0.27 (0.22 to 0.35) | 0.48 (0.40 to 0.63) | 0.49 (0.38 to 0.65) | 1.00 (1.00 to 1.00) | 0.42 (0.31 to 0.55) |
| V5 | GRU maps | 0.20 (0.13 to 0.27) | 0.38 (0.26 to 0.54) | 0.38 (0.28 to 0.52) | 0.38 (0.28 to 0.51) | 0.48 (0.40 to 0.63) | 0.49 (0.38 to 0.65) | 1.00 (1.00 to 1.00) | 0.49 (0.41 to 0.59) |

G_S, cosine for each gate's block of the two GRU maps:

| variant | where | input map, r | input map, z | input map, n | recurrent map, r | recurrent map, z | recurrent map, n |
|---|---|---|---|---|---|---|---|
| V0 | every map | 0.822 (0.717 to 0.881) | 0.775 (0.674 to 0.867) | 0.788 (0.672 to 0.852) | 0.841 (0.765 to 0.907) | 0.814 (0.769 to 0.867) | 0.817 (0.710 to 0.861) |
| V1 | every map | 0.824 (0.726 to 0.885) | 0.777 (0.668 to 0.870) | 0.789 (0.677 to 0.851) | 0.843 (0.773 to 0.909) | 0.816 (0.765 to 0.871) | 0.818 (0.714 to 0.859) |
| V2 | every map | 0.828 (0.703 to 0.887) | 0.783 (0.688 to 0.871) | 0.789 (0.663 to 0.849) | 0.848 (0.759 to 0.912) | 0.821 (0.774 to 0.870) | 0.817 (0.697 to 0.875) |
| V3 | every map | 0.828 (0.767 to 0.880) | 0.804 (0.722 to 0.865) | 0.804 (0.736 to 0.858) | 0.838 (0.760 to 0.896) | 0.836 (0.806 to 0.861) | 0.833 (0.784 to 0.887) |
| V4 | every map | 0.839 (0.787 to 0.892) | 0.815 (0.734 to 0.873) | 0.808 (0.742 to 0.864) | 0.852 (0.791 to 0.908) | 0.847 (0.813 to 0.865) | 0.837 (0.784 to 0.897) |
| V2 | GRU maps | 0.826 (0.728 to 0.887) | 0.778 (0.676 to 0.870) | 0.790 (0.677 to 0.849) | 0.844 (0.776 to 0.912) | 0.817 (0.771 to 0.870) | 0.819 (0.715 to 0.862) |
| V3 | GRU maps | 0.832 (0.737 to 0.891) | 0.792 (0.703 to 0.877) | 0.795 (0.684 to 0.860) | 0.850 (0.785 to 0.915) | 0.826 (0.782 to 0.879) | 0.824 (0.724 to 0.860) |
| V4 | GRU maps | 0.835 (0.747 to 0.899) | 0.794 (0.706 to 0.880) | 0.797 (0.689 to 0.856) | 0.854 (0.795 to 0.921) | 0.829 (0.785 to 0.882) | 0.826 (0.728 to 0.861) |
| V5 | GRU maps | 0.934 (0.875 to 0.968) | 0.930 (0.883 to 0.971) | 0.932 (0.852 to 0.959) | 0.932 (0.885 to 0.963) | 0.923 (0.882 to 0.964) | 0.924 (0.842 to 0.954) |

#### G_B

E42 G (dense), trained under back-propagation. Loss on the 16 batches: mean 0.0147 (least 0.0037, greatest 0.0292); first batch 0.0239. Norm of the exact gradient: mean 0.392 (0.061 to 0.828).

G_B, cosine with the exact gradient, variant at every map:

| variant | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters | all but l3 |
|---|---|---|---|---|---|---|---|---|---|
| V0, first batch | 0.471 | 0.326 | 0.674 | 0.689 | 0.673 | 0.672 | 1.000 | 0.637 | 0.609 |
| V0 | 0.722 (0.388 to 0.903) | 0.657 (0.377 to 0.810) | 0.751 (0.656 to 0.842) | 0.772 (0.658 to 0.846) | 0.766 (0.701 to 0.814) | 0.761 (0.698 to 0.830) | 1.000 (1.000 to 1.000) | 0.602 (0.516 to 0.703) | 0.710 (0.622 to 0.772) |
| V1 | 0.728 (0.410 to 0.900) | 0.668 (0.379 to 0.816) | 0.758 (0.667 to 0.843) | 0.778 (0.670 to 0.848) | 0.766 (0.701 to 0.814) | 0.761 (0.698 to 0.830) | 1.000 (1.000 to 1.000) | 0.604 (0.518 to 0.710) | 0.715 (0.629 to 0.773) |
| V2 | 0.744 (0.540 to 0.910) | 0.683 (0.376 to 0.821) | 0.750 (0.673 to 0.836) | 0.771 (0.684 to 0.840) | 0.765 (0.709 to 0.812) | 0.761 (0.704 to 0.827) | 1.000 (1.000 to 1.000) | 0.634 (0.539 to 0.733) | 0.707 (0.641 to 0.769) |
| V3 | 0.727 (0.263 to 0.892) | 0.670 (0.345 to 0.834) | 0.811 (0.734 to 0.890) | 0.828 (0.747 to 0.887) | 0.701 (0.547 to 0.814) | 0.689 (0.546 to 0.798) | 1.000 (1.000 to 1.000) | 0.661 (0.592 to 0.731) | 0.749 (0.668 to 0.820) |
| V4 | 0.771 (0.423 to 0.903) | 0.714 (0.431 to 0.855) | 0.807 (0.740 to 0.884) | 0.824 (0.767 to 0.889) | 0.702 (0.549 to 0.816) | 0.690 (0.547 to 0.798) | 1.000 (1.000 to 1.000) | 0.696 (0.615 to 0.767) | 0.748 (0.682 to 0.818) |

G_B, cosine with the exact gradient, variant at the two GRU maps only (V0 at the other maps; V5 here is the upper limit for a scale at the GRU maps):

| variant | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters | all but l3 |
|---|---|---|---|---|---|---|---|---|---|
| V1 | 0.728 (0.410 to 0.900) | 0.668 (0.379 to 0.816) | 0.758 (0.667 to 0.843) | 0.778 (0.670 to 0.848) | 0.766 (0.701 to 0.814) | 0.761 (0.698 to 0.830) | 1.000 (1.000 to 1.000) | 0.604 (0.518 to 0.710) | 0.715 (0.629 to 0.773) |
| V2 | 0.746 (0.553 to 0.906) | 0.684 (0.380 to 0.823) | 0.753 (0.673 to 0.832) | 0.774 (0.676 to 0.837) | 0.766 (0.701 to 0.814) | 0.761 (0.698 to 0.830) | 1.000 (1.000 to 1.000) | 0.600 (0.510 to 0.707) | 0.710 (0.639 to 0.760) |
| V3 | 0.727 (0.416 to 0.907) | 0.666 (0.387 to 0.820) | 0.766 (0.662 to 0.852) | 0.786 (0.664 to 0.856) | 0.766 (0.701 to 0.814) | 0.761 (0.698 to 0.830) | 1.000 (1.000 to 1.000) | 0.610 (0.535 to 0.713) | 0.723 (0.625 to 0.782) |
| V4 | 0.756 (0.568 to 0.910) | 0.696 (0.396 to 0.820) | 0.767 (0.679 to 0.841) | 0.788 (0.682 to 0.847) | 0.766 (0.701 to 0.814) | 0.761 (0.698 to 0.830) | 1.000 (1.000 to 1.000) | 0.609 (0.528 to 0.717) | 0.723 (0.643 to 0.774) |
| V5 | 0.915 (0.782 to 0.979) | 0.936 (0.816 to 0.987) | 0.914 (0.838 to 0.957) | 0.914 (0.843 to 0.955) | 0.766 (0.701 to 0.814) | 0.761 (0.698 to 0.830) | 1.000 (1.000 to 1.000) | 0.765 (0.686 to 0.911) | 0.887 (0.819 to 0.947) |

G_B, norm of the sweep's gradient over the norm of the exact gradient:

| variant | where | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters |
|---|---|---|---|---|---|---|---|---|---|
| V0 | every map | 0.10 (0.06 to 0.16) | 0.15 (0.10 to 0.22) | 0.17 (0.10 to 0.26) | 0.17 (0.10 to 0.26) | 0.47 (0.31 to 0.71) | 0.50 (0.34 to 0.71) | 1.00 (1.00 to 1.00) | 0.35 (0.28 to 0.46) |
| V1 | every map | 0.09 (0.06 to 0.15) | 0.14 (0.10 to 0.22) | 0.17 (0.10 to 0.26) | 0.17 (0.10 to 0.27) | 0.47 (0.31 to 0.71) | 0.50 (0.34 to 0.71) | 1.00 (1.00 to 1.00) | 0.35 (0.28 to 0.46) |
| V2 | every map | 0.10 (0.06 to 0.16) | 0.15 (0.09 to 0.23) | 0.21 (0.12 to 0.33) | 0.21 (0.12 to 0.33) | 0.58 (0.40 to 0.91) | 0.62 (0.44 to 0.91) | 1.00 (1.00 to 1.00) | 0.37 (0.29 to 0.48) |
| V3 | every map | 0.12 (0.08 to 0.17) | 0.18 (0.12 to 0.29) | 0.21 (0.13 to 0.27) | 0.21 (0.14 to 0.27) | 0.55 (0.44 to 0.69) | 0.57 (0.46 to 0.68) | 1.00 (1.00 to 1.00) | 0.37 (0.30 to 0.48) |
| V4 | every map | 0.12 (0.08 to 0.18) | 0.18 (0.11 to 0.26) | 0.26 (0.16 to 0.33) | 0.26 (0.17 to 0.34) | 0.68 (0.56 to 0.88) | 0.70 (0.59 to 0.87) | 1.00 (1.00 to 1.00) | 0.40 (0.31 to 0.52) |
| V2 | GRU maps | 0.08 (0.05 to 0.14) | 0.13 (0.08 to 0.19) | 0.17 (0.10 to 0.27) | 0.17 (0.10 to 0.27) | 0.47 (0.31 to 0.71) | 0.50 (0.34 to 0.71) | 1.00 (1.00 to 1.00) | 0.35 (0.28 to 0.46) |
| V3 | GRU maps | 0.10 (0.07 to 0.16) | 0.15 (0.11 to 0.22) | 0.17 (0.11 to 0.26) | 0.17 (0.11 to 0.27) | 0.47 (0.31 to 0.71) | 0.50 (0.34 to 0.71) | 1.00 (1.00 to 1.00) | 0.35 (0.28 to 0.46) |
| V4 | GRU maps | 0.09 (0.05 to 0.14) | 0.13 (0.08 to 0.20) | 0.17 (0.10 to 0.27) | 0.17 (0.11 to 0.27) | 0.47 (0.31 to 0.71) | 0.50 (0.34 to 0.71) | 1.00 (1.00 to 1.00) | 0.35 (0.28 to 0.46) |
| V5 | GRU maps | 0.20 (0.12 to 0.32) | 0.27 (0.17 to 0.42) | 0.25 (0.17 to 0.42) | 0.24 (0.16 to 0.42) | 0.47 (0.31 to 0.71) | 0.50 (0.34 to 0.71) | 1.00 (1.00 to 1.00) | 0.39 (0.31 to 0.49) |

G_B, cosine for each gate's block of the two GRU maps:

| variant | where | input map, r | input map, z | input map, n | recurrent map, r | recurrent map, z | recurrent map, n |
|---|---|---|---|---|---|---|---|
| V0 | every map | 0.745 (0.446 to 0.859) | 0.749 (0.682 to 0.824) | 0.754 (0.647 to 0.855) | 0.769 (0.528 to 0.867) | 0.766 (0.711 to 0.816) | 0.782 (0.627 to 0.864) |
| V1 | every map | 0.754 (0.483 to 0.862) | 0.754 (0.692 to 0.832) | 0.761 (0.658 to 0.856) | 0.778 (0.566 to 0.869) | 0.772 (0.721 to 0.821) | 0.787 (0.640 to 0.866) |
| V2 | every map | 0.746 (0.492 to 0.858) | 0.745 (0.693 to 0.828) | 0.755 (0.670 to 0.852) | 0.770 (0.574 to 0.863) | 0.763 (0.712 to 0.818) | 0.782 (0.658 to 0.858) |
| V3 | every map | 0.818 (0.511 to 0.938) | 0.817 (0.774 to 0.863) | 0.810 (0.709 to 0.896) | 0.833 (0.576 to 0.931) | 0.836 (0.786 to 0.877) | 0.830 (0.719 to 0.900) |
| V4 | every map | 0.816 (0.564 to 0.930) | 0.809 (0.766 to 0.866) | 0.807 (0.714 to 0.897) | 0.831 (0.625 to 0.922) | 0.828 (0.767 to 0.867) | 0.827 (0.745 to 0.896) |
| V2 | GRU maps | 0.750 (0.498 to 0.857) | 0.748 (0.694 to 0.831) | 0.756 (0.666 to 0.846) | 0.775 (0.581 to 0.860) | 0.766 (0.723 to 0.820) | 0.784 (0.648 to 0.856) |
| V3 | GRU maps | 0.772 (0.437 to 0.886) | 0.767 (0.692 to 0.849) | 0.765 (0.651 to 0.863) | 0.795 (0.516 to 0.889) | 0.784 (0.720 to 0.836) | 0.790 (0.631 to 0.870) |
| V4 | GRU maps | 0.777 (0.488 to 0.885) | 0.765 (0.704 to 0.856) | 0.767 (0.670 to 0.854) | 0.800 (0.567 to 0.898) | 0.784 (0.732 to 0.844) | 0.792 (0.652 to 0.862) |
| V5 | GRU maps | 0.925 (0.834 to 0.991) | 0.917 (0.852 to 0.970) | 0.911 (0.824 to 0.955) | 0.926 (0.841 to 0.990) | 0.915 (0.874 to 0.964) | 0.909 (0.828 to 0.953) |

#### F10_start

E42 F10 at its start (the start file of E49's check 2, a run of F10 by E49's runner). Loss on the 16 batches: mean 5.5906 (least 5.5787, greatest 5.6025); first batch 5.5772. Norm of the exact gradient: mean 2.036 (1.927 to 2.165).

F10_start, cosine with the exact gradient, variant at every map:

| variant | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters | all but l3 |
|---|---|---|---|---|---|---|---|---|---|
| V0, first batch | 0.754 | 0.723 | 0.750 | 0.744 | 0.865 | 0.867 | 1.000 | 0.926 | 0.847 |
| V0 | 0.731 (0.676 to 0.826) | 0.727 (0.719 to 0.734) | 0.751 (0.748 to 0.757) | 0.744 (0.740 to 0.749) | 0.867 (0.865 to 0.870) | 0.868 (0.863 to 0.872) | 1.000 (1.000 to 1.000) | 0.927 (0.925 to 0.929) | 0.848 (0.845 to 0.851) |
| V1 | 0.731 (0.676 to 0.826) | 0.727 (0.719 to 0.734) | 0.751 (0.748 to 0.757) | 0.745 (0.740 to 0.749) | 0.867 (0.865 to 0.870) | 0.868 (0.863 to 0.872) | 1.000 (1.000 to 1.000) | 0.927 (0.925 to 0.929) | 0.848 (0.845 to 0.851) |
| V2 | 0.778 (0.734 to 0.848) | 0.729 (0.721 to 0.736) | 0.755 (0.753 to 0.762) | 0.749 (0.745 to 0.754) | 0.868 (0.865 to 0.871) | 0.868 (0.864 to 0.872) | 1.000 (1.000 to 1.000) | 0.927 (0.926 to 0.929) | 0.848 (0.846 to 0.852) |
| V3 | 0.743 (0.693 to 0.836) | 0.728 (0.720 to 0.735) | 0.747 (0.744 to 0.754) | 0.741 (0.737 to 0.746) | 0.868 (0.866 to 0.871) | 0.869 (0.864 to 0.873) | 1.000 (1.000 to 1.000) | 0.927 (0.925 to 0.929) | 0.848 (0.846 to 0.851) |
| V4 | 0.796 (0.755 to 0.862) | 0.730 (0.722 to 0.737) | 0.752 (0.749 to 0.759) | 0.746 (0.742 to 0.751) | 0.868 (0.866 to 0.871) | 0.869 (0.865 to 0.873) | 1.000 (1.000 to 1.000) | 0.927 (0.926 to 0.929) | 0.849 (0.846 to 0.852) |

F10_start, cosine with the exact gradient, variant at the two GRU maps only (V0 at the other maps; V5 here is the upper limit for a scale at the GRU maps):

| variant | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters | all but l3 |
|---|---|---|---|---|---|---|---|---|---|
| V1 | 0.731 (0.676 to 0.826) | 0.727 (0.719 to 0.734) | 0.751 (0.748 to 0.757) | 0.745 (0.740 to 0.749) | 0.867 (0.865 to 0.870) | 0.868 (0.863 to 0.872) | 1.000 (1.000 to 1.000) | 0.927 (0.925 to 0.929) | 0.848 (0.845 to 0.851) |
| V2 | 0.726 (0.670 to 0.822) | 0.727 (0.720 to 0.734) | 0.752 (0.749 to 0.759) | 0.746 (0.741 to 0.751) | 0.867 (0.865 to 0.870) | 0.868 (0.863 to 0.872) | 1.000 (1.000 to 1.000) | 0.927 (0.925 to 0.929) | 0.848 (0.846 to 0.851) |
| V3 | 0.731 (0.677 to 0.826) | 0.727 (0.720 to 0.734) | 0.750 (0.747 to 0.757) | 0.744 (0.740 to 0.749) | 0.867 (0.865 to 0.870) | 0.868 (0.863 to 0.872) | 1.000 (1.000 to 1.000) | 0.927 (0.925 to 0.929) | 0.848 (0.845 to 0.851) |
| V4 | 0.727 (0.671 to 0.822) | 0.727 (0.720 to 0.734) | 0.752 (0.748 to 0.759) | 0.746 (0.741 to 0.751) | 0.867 (0.865 to 0.870) | 0.868 (0.863 to 0.872) | 1.000 (1.000 to 1.000) | 0.927 (0.925 to 0.929) | 0.848 (0.845 to 0.851) |
| V5 | 0.829 (0.791 to 0.905) | 0.760 (0.748 to 0.770) | 0.762 (0.759 to 0.768) | 0.756 (0.752 to 0.761) | 0.867 (0.865 to 0.870) | 0.868 (0.863 to 0.872) | 1.000 (1.000 to 1.000) | 0.928 (0.926 to 0.929) | 0.850 (0.847 to 0.852) |

F10_start, norm of the sweep's gradient over the norm of the exact gradient:

| variant | where | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters |
|---|---|---|---|---|---|---|---|---|---|
| V0 | every map | 0.59 (0.54 to 0.74) | 0.73 (0.70 to 0.74) | 0.67 (0.66 to 0.69) | 0.66 (0.65 to 0.69) | 0.85 (0.84 to 0.86) | 0.87 (0.86 to 0.88) | 1.00 (1.00 to 1.00) | 0.92 (0.92 to 0.93) |
| V1 | every map | 0.59 (0.54 to 0.74) | 0.73 (0.70 to 0.74) | 0.67 (0.66 to 0.69) | 0.66 (0.65 to 0.69) | 0.85 (0.84 to 0.86) | 0.87 (0.86 to 0.88) | 1.00 (1.00 to 1.00) | 0.92 (0.92 to 0.93) |
| V2 | every map | 0.57 (0.49 to 0.73) | 0.72 (0.70 to 0.73) | 0.66 (0.65 to 0.68) | 0.65 (0.64 to 0.68) | 0.84 (0.83 to 0.85) | 0.85 (0.84 to 0.86) | 1.00 (1.00 to 1.00) | 0.91 (0.91 to 0.92) |
| V3 | every map | 0.58 (0.53 to 0.73) | 0.73 (0.70 to 0.74) | 0.67 (0.66 to 0.69) | 0.66 (0.65 to 0.68) | 0.85 (0.84 to 0.86) | 0.87 (0.86 to 0.88) | 1.00 (1.00 to 1.00) | 0.92 (0.92 to 0.93) |
| V4 | every map | 0.56 (0.50 to 0.72) | 0.72 (0.70 to 0.73) | 0.66 (0.65 to 0.68) | 0.65 (0.64 to 0.68) | 0.84 (0.83 to 0.85) | 0.85 (0.84 to 0.86) | 1.00 (1.00 to 1.00) | 0.91 (0.91 to 0.92) |
| V2 | GRU maps | 0.58 (0.53 to 0.74) | 0.73 (0.70 to 0.74) | 0.67 (0.66 to 0.69) | 0.66 (0.65 to 0.69) | 0.85 (0.84 to 0.86) | 0.87 (0.86 to 0.88) | 1.00 (1.00 to 1.00) | 0.92 (0.92 to 0.93) |
| V3 | GRU maps | 0.59 (0.54 to 0.74) | 0.73 (0.70 to 0.74) | 0.68 (0.66 to 0.69) | 0.66 (0.65 to 0.69) | 0.85 (0.84 to 0.86) | 0.87 (0.86 to 0.88) | 1.00 (1.00 to 1.00) | 0.92 (0.92 to 0.93) |
| V4 | GRU maps | 0.59 (0.53 to 0.74) | 0.73 (0.70 to 0.74) | 0.68 (0.66 to 0.69) | 0.66 (0.65 to 0.69) | 0.85 (0.84 to 0.86) | 0.87 (0.86 to 0.88) | 1.00 (1.00 to 1.00) | 0.92 (0.92 to 0.93) |
| V5 | GRU maps | 0.68 (0.63 to 0.81) | 0.75 (0.73 to 0.77) | 0.68 (0.67 to 0.70) | 0.67 (0.65 to 0.69) | 0.85 (0.84 to 0.86) | 0.87 (0.86 to 0.88) | 1.00 (1.00 to 1.00) | 0.92 (0.92 to 0.93) |

F10_start, cosine for each gate's block of the two GRU maps:

| variant | where | input map, r | input map, z | input map, n | recurrent map, r | recurrent map, z | recurrent map, n |
|---|---|---|---|---|---|---|---|
| V0 | every map | 0.726 (0.708 to 0.751) | 0.724 (0.708 to 0.737) | 0.752 (0.749 to 0.758) | 0.718 (0.704 to 0.738) | 0.720 (0.702 to 0.734) | 0.745 (0.740 to 0.751) |
| V1 | every map | 0.727 (0.708 to 0.751) | 0.724 (0.708 to 0.737) | 0.752 (0.749 to 0.758) | 0.718 (0.704 to 0.738) | 0.720 (0.702 to 0.734) | 0.745 (0.740 to 0.751) |
| V2 | every map | 0.725 (0.706 to 0.751) | 0.731 (0.716 to 0.744) | 0.756 (0.753 to 0.763) | 0.718 (0.703 to 0.739) | 0.725 (0.707 to 0.740) | 0.750 (0.746 to 0.755) |
| V3 | every map | 0.722 (0.704 to 0.746) | 0.720 (0.703 to 0.734) | 0.748 (0.745 to 0.755) | 0.714 (0.700 to 0.733) | 0.718 (0.702 to 0.733) | 0.742 (0.738 to 0.747) |
| V4 | every map | 0.721 (0.702 to 0.747) | 0.727 (0.712 to 0.741) | 0.753 (0.750 to 0.760) | 0.715 (0.700 to 0.736) | 0.723 (0.707 to 0.739) | 0.747 (0.743 to 0.752) |
| V2 | GRU maps | 0.726 (0.706 to 0.751) | 0.725 (0.709 to 0.739) | 0.753 (0.750 to 0.760) | 0.718 (0.703 to 0.739) | 0.720 (0.702 to 0.735) | 0.747 (0.742 to 0.752) |
| V3 | GRU maps | 0.726 (0.707 to 0.750) | 0.723 (0.707 to 0.737) | 0.752 (0.748 to 0.758) | 0.718 (0.703 to 0.737) | 0.719 (0.701 to 0.734) | 0.745 (0.741 to 0.750) |
| V4 | GRU maps | 0.725 (0.706 to 0.751) | 0.725 (0.708 to 0.739) | 0.753 (0.750 to 0.760) | 0.718 (0.702 to 0.739) | 0.719 (0.702 to 0.735) | 0.747 (0.742 to 0.752) |
| V5 | GRU maps | 0.729 (0.708 to 0.752) | 0.735 (0.721 to 0.749) | 0.763 (0.760 to 0.769) | 0.719 (0.704 to 0.739) | 0.730 (0.714 to 0.745) | 0.757 (0.753 to 0.762) |

#### F10_S

E42 F10 (10 neighbours and itself), trained under the sweep. Loss on the 16 batches: mean 0.0430 (least 0.0293, greatest 0.0586); first batch 0.0491. Norm of the exact gradient: mean 0.345 (0.223 to 0.548).

F10_S, cosine with the exact gradient, variant at every map:

| variant | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters | all but l3 |
|---|---|---|---|---|---|---|---|---|---|
| V0, first batch | 0.647 | 0.579 | 0.565 | 0.604 | 0.826 | 0.837 | 1.000 | 0.810 | 0.666 |
| V0 | 0.677 (0.436 to 0.830) | 0.537 (0.418 to 0.640) | 0.647 (0.581 to 0.712) | 0.685 (0.624 to 0.757) | 0.837 (0.822 to 0.870) | 0.840 (0.822 to 0.865) | 1.000 (1.000 to 1.000) | 0.722 (0.674 to 0.772) | 0.656 (0.617 to 0.701) |
| V1 | 0.693 (0.325 to 0.818) | 0.594 (0.376 to 0.728) | 0.664 (0.605 to 0.728) | 0.700 (0.646 to 0.775) | 0.837 (0.822 to 0.870) | 0.840 (0.822 to 0.865) | 1.000 (1.000 to 1.000) | 0.716 (0.666 to 0.768) | 0.661 (0.612 to 0.688) |
| V2 | 0.794 (0.577 to 0.885) | 0.631 (0.433 to 0.756) | 0.696 (0.648 to 0.752) | 0.730 (0.680 to 0.799) | 0.837 (0.821 to 0.870) | 0.840 (0.821 to 0.865) | 1.000 (1.000 to 1.000) | 0.781 (0.746 to 0.818) | 0.685 (0.646 to 0.705) |
| V3 | 0.679 (0.359 to 0.815) | 0.539 (0.350 to 0.649) | 0.629 (0.551 to 0.701) | 0.658 (0.572 to 0.725) | 0.719 (0.698 to 0.746) | 0.724 (0.689 to 0.757) | 1.000 (1.000 to 1.000) | 0.726 (0.679 to 0.779) | 0.606 (0.542 to 0.668) |
| V4 | 0.792 (0.604 to 0.889) | 0.659 (0.492 to 0.755) | 0.675 (0.595 to 0.759) | 0.697 (0.615 to 0.768) | 0.719 (0.698 to 0.746) | 0.724 (0.689 to 0.757) | 1.000 (1.000 to 1.000) | 0.771 (0.737 to 0.810) | 0.643 (0.594 to 0.697) |

F10_S, cosine with the exact gradient, variant at the two GRU maps only (V0 at the other maps; V5 here is the upper limit for a scale at the GRU maps):

| variant | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters | all but l3 |
|---|---|---|---|---|---|---|---|---|---|
| V1 | 0.693 (0.325 to 0.818) | 0.594 (0.376 to 0.728) | 0.664 (0.605 to 0.728) | 0.700 (0.646 to 0.775) | 0.837 (0.822 to 0.870) | 0.840 (0.822 to 0.865) | 1.000 (1.000 to 1.000) | 0.716 (0.666 to 0.768) | 0.661 (0.612 to 0.688) |
| V2 | 0.716 (0.472 to 0.848) | 0.620 (0.426 to 0.745) | 0.683 (0.632 to 0.745) | 0.715 (0.665 to 0.790) | 0.837 (0.822 to 0.870) | 0.840 (0.822 to 0.865) | 1.000 (1.000 to 1.000) | 0.719 (0.669 to 0.768) | 0.669 (0.631 to 0.694) |
| V3 | 0.717 (0.511 to 0.852) | 0.595 (0.453 to 0.695) | 0.669 (0.608 to 0.734) | 0.704 (0.651 to 0.779) | 0.837 (0.822 to 0.870) | 0.840 (0.822 to 0.865) | 1.000 (1.000 to 1.000) | 0.736 (0.696 to 0.783) | 0.684 (0.640 to 0.731) |
| V4 | 0.778 (0.556 to 0.888) | 0.693 (0.507 to 0.806) | 0.710 (0.662 to 0.772) | 0.738 (0.697 to 0.813) | 0.837 (0.822 to 0.870) | 0.840 (0.822 to 0.865) | 1.000 (1.000 to 1.000) | 0.732 (0.689 to 0.777) | 0.702 (0.659 to 0.727) |
| V5 | 0.844 (0.650 to 0.921) | 0.877 (0.752 to 0.929) | 0.829 (0.784 to 0.890) | 0.837 (0.777 to 0.902) | 0.837 (0.822 to 0.870) | 0.840 (0.822 to 0.865) | 1.000 (1.000 to 1.000) | 0.802 (0.785 to 0.819) | 0.825 (0.794 to 0.868) |

F10_S, norm of the sweep's gradient over the norm of the exact gradient:

| variant | where | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters |
|---|---|---|---|---|---|---|---|---|---|
| V0 | every map | 0.11 (0.08 to 0.19) | 0.24 (0.18 to 0.32) | 0.19 (0.15 to 0.23) | 0.21 (0.17 to 0.26) | 0.40 (0.37 to 0.44) | 0.41 (0.37 to 0.45) | 1.00 (1.00 to 1.00) | 0.60 (0.52 to 0.69) |
| V1 | every map | 0.08 (0.05 to 0.13) | 0.15 (0.12 to 0.22) | 0.20 (0.16 to 0.25) | 0.22 (0.18 to 0.27) | 0.40 (0.37 to 0.44) | 0.41 (0.37 to 0.45) | 1.00 (1.00 to 1.00) | 0.59 (0.51 to 0.69) |
| V2 | every map | 0.17 (0.13 to 0.26) | 0.25 (0.20 to 0.34) | 0.37 (0.30 to 0.46) | 0.41 (0.34 to 0.50) | 0.73 (0.67 to 0.80) | 0.73 (0.67 to 0.81) | 1.00 (1.00 to 1.00) | 0.68 (0.60 to 0.74) |
| V3 | every map | 0.17 (0.10 to 0.30) | 0.34 (0.26 to 0.47) | 0.23 (0.19 to 0.29) | 0.25 (0.20 to 0.32) | 0.50 (0.46 to 0.55) | 0.50 (0.46 to 0.54) | 1.00 (1.00 to 1.00) | 0.63 (0.55 to 0.71) |
| V4 | every map | 0.26 (0.20 to 0.40) | 0.34 (0.27 to 0.42) | 0.48 (0.41 to 0.60) | 0.51 (0.41 to 0.63) | 0.91 (0.84 to 0.99) | 0.91 (0.84 to 0.98) | 1.00 (1.00 to 1.00) | 0.74 (0.66 to 0.80) |
| V2 | GRU maps | 0.07 (0.05 to 0.12) | 0.14 (0.11 to 0.19) | 0.21 (0.17 to 0.26) | 0.23 (0.19 to 0.28) | 0.40 (0.37 to 0.44) | 0.41 (0.37 to 0.45) | 1.00 (1.00 to 1.00) | 0.59 (0.51 to 0.69) |
| V3 | GRU maps | 0.14 (0.10 to 0.22) | 0.29 (0.21 to 0.38) | 0.19 (0.16 to 0.24) | 0.22 (0.18 to 0.26) | 0.40 (0.37 to 0.44) | 0.41 (0.37 to 0.45) | 1.00 (1.00 to 1.00) | 0.61 (0.53 to 0.69) |
| V4 | GRU maps | 0.08 (0.06 to 0.13) | 0.17 (0.14 to 0.22) | 0.22 (0.18 to 0.27) | 0.24 (0.20 to 0.29) | 0.40 (0.37 to 0.44) | 0.41 (0.37 to 0.45) | 1.00 (1.00 to 1.00) | 0.60 (0.51 to 0.69) |
| V5 | GRU maps | 0.15 (0.12 to 0.21) | 0.32 (0.26 to 0.41) | 0.31 (0.25 to 0.38) | 0.31 (0.26 to 0.38) | 0.40 (0.37 to 0.44) | 0.41 (0.37 to 0.45) | 1.00 (1.00 to 1.00) | 0.62 (0.56 to 0.70) |

F10_S, cosine for each gate's block of the two GRU maps:

| variant | where | input map, r | input map, z | input map, n | recurrent map, r | recurrent map, z | recurrent map, n |
|---|---|---|---|---|---|---|---|
| V0 | every map | 0.650 (0.526 to 0.734) | 0.681 (0.602 to 0.739) | 0.645 (0.580 to 0.712) | 0.652 (0.500 to 0.759) | 0.697 (0.611 to 0.770) | 0.684 (0.620 to 0.757) |
| V1 | every map | 0.667 (0.551 to 0.744) | 0.698 (0.622 to 0.751) | 0.662 (0.604 to 0.727) | 0.668 (0.519 to 0.773) | 0.713 (0.631 to 0.786) | 0.699 (0.642 to 0.775) |
| V2 | every map | 0.700 (0.582 to 0.768) | 0.726 (0.664 to 0.772) | 0.694 (0.647 to 0.751) | 0.701 (0.541 to 0.798) | 0.741 (0.669 to 0.806) | 0.729 (0.677 to 0.799) |
| V3 | every map | 0.640 (0.500 to 0.748) | 0.665 (0.564 to 0.737) | 0.627 (0.550 to 0.700) | 0.627 (0.473 to 0.739) | 0.684 (0.564 to 0.769) | 0.656 (0.574 to 0.722) |
| V4 | every map | 0.680 (0.549 to 0.757) | 0.701 (0.612 to 0.777) | 0.673 (0.587 to 0.759) | 0.663 (0.456 to 0.784) | 0.717 (0.606 to 0.793) | 0.696 (0.617 to 0.766) |
| V2 | GRU maps | 0.687 (0.568 to 0.759) | 0.714 (0.650 to 0.761) | 0.681 (0.631 to 0.745) | 0.688 (0.530 to 0.786) | 0.728 (0.657 to 0.794) | 0.714 (0.662 to 0.790) |
| V3 | GRU maps | 0.673 (0.540 to 0.760) | 0.698 (0.631 to 0.752) | 0.667 (0.606 to 0.733) | 0.675 (0.518 to 0.776) | 0.713 (0.633 to 0.787) | 0.704 (0.653 to 0.779) |
| V4 | GRU maps | 0.714 (0.589 to 0.782) | 0.735 (0.683 to 0.786) | 0.709 (0.661 to 0.772) | 0.716 (0.561 to 0.807) | 0.747 (0.685 to 0.812) | 0.738 (0.698 to 0.813) |
| V5 | GRU maps | 0.827 (0.696 to 0.909) | 0.855 (0.818 to 0.908) | 0.828 (0.780 to 0.890) | 0.825 (0.672 to 0.918) | 0.851 (0.807 to 0.906) | 0.836 (0.773 to 0.903) |

#### F10_B

E42 F10, trained under back-propagation. Loss on the 16 batches: mean 0.0467 (least 0.0286, greatest 0.0660); first batch 0.0463. Norm of the exact gradient: mean 0.405 (0.228 to 0.615).

F10_B, cosine with the exact gradient, variant at every map:

| variant | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters | all but l3 |
|---|---|---|---|---|---|---|---|---|---|
| V0, first batch | 0.447 | 0.444 | 0.626 | 0.631 | 0.715 | 0.714 | 1.000 | 0.740 | 0.602 |
| V0 | 0.589 (0.368 to 0.766) | 0.545 (0.391 to 0.679) | 0.585 (0.492 to 0.666) | 0.595 (0.468 to 0.658) | 0.745 (0.681 to 0.786) | 0.749 (0.689 to 0.786) | 1.000 (1.000 to 1.000) | 0.694 (0.597 to 0.811) | 0.601 (0.516 to 0.650) |
| V1 | 0.667 (0.495 to 0.863) | 0.600 (0.455 to 0.757) | 0.596 (0.508 to 0.675) | 0.606 (0.486 to 0.666) | 0.745 (0.681 to 0.786) | 0.749 (0.689 to 0.786) | 1.000 (1.000 to 1.000) | 0.687 (0.600 to 0.807) | 0.610 (0.548 to 0.664) |
| V2 | 0.677 (0.500 to 0.842) | 0.630 (0.472 to 0.755) | 0.626 (0.541 to 0.713) | 0.636 (0.511 to 0.694) | 0.747 (0.681 to 0.787) | 0.750 (0.689 to 0.787) | 1.000 (1.000 to 1.000) | 0.733 (0.647 to 0.843) | 0.616 (0.565 to 0.677) |
| V3 | 0.616 (0.423 to 0.813) | 0.545 (0.373 to 0.648) | 0.639 (0.480 to 0.729) | 0.654 (0.538 to 0.737) | 0.680 (0.641 to 0.730) | 0.681 (0.639 to 0.738) | 1.000 (1.000 to 1.000) | 0.702 (0.609 to 0.825) | 0.584 (0.485 to 0.652) |
| V4 | 0.724 (0.520 to 0.880) | 0.644 (0.494 to 0.753) | 0.683 (0.546 to 0.781) | 0.695 (0.572 to 0.789) | 0.681 (0.643 to 0.731) | 0.682 (0.639 to 0.739) | 1.000 (1.000 to 1.000) | 0.736 (0.650 to 0.854) | 0.600 (0.542 to 0.684) |

F10_B, cosine with the exact gradient, variant at the two GRU maps only (V0 at the other maps; V5 here is the upper limit for a scale at the GRU maps):

| variant | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters | all but l3 |
|---|---|---|---|---|---|---|---|---|---|
| V1 | 0.667 (0.495 to 0.863) | 0.600 (0.455 to 0.757) | 0.596 (0.508 to 0.675) | 0.606 (0.486 to 0.666) | 0.745 (0.681 to 0.786) | 0.749 (0.689 to 0.786) | 1.000 (1.000 to 1.000) | 0.687 (0.600 to 0.807) | 0.610 (0.548 to 0.664) |
| V2 | 0.665 (0.470 to 0.843) | 0.618 (0.442 to 0.749) | 0.609 (0.526 to 0.686) | 0.618 (0.498 to 0.678) | 0.745 (0.681 to 0.786) | 0.749 (0.689 to 0.786) | 1.000 (1.000 to 1.000) | 0.686 (0.588 to 0.807) | 0.611 (0.561 to 0.668) |
| V3 | 0.615 (0.402 to 0.772) | 0.569 (0.404 to 0.715) | 0.614 (0.535 to 0.689) | 0.625 (0.526 to 0.689) | 0.745 (0.681 to 0.786) | 0.749 (0.689 to 0.786) | 1.000 (1.000 to 1.000) | 0.701 (0.617 to 0.814) | 0.617 (0.542 to 0.662) |
| V4 | 0.683 (0.483 to 0.844) | 0.644 (0.441 to 0.782) | 0.648 (0.578 to 0.724) | 0.657 (0.563 to 0.721) | 0.745 (0.681 to 0.786) | 0.749 (0.689 to 0.786) | 1.000 (1.000 to 1.000) | 0.693 (0.620 to 0.809) | 0.630 (0.577 to 0.683) |
| V5 | 0.812 (0.641 to 0.898) | 0.857 (0.669 to 0.929) | 0.793 (0.715 to 0.874) | 0.797 (0.717 to 0.887) | 0.745 (0.681 to 0.786) | 0.749 (0.689 to 0.786) | 1.000 (1.000 to 1.000) | 0.766 (0.703 to 0.838) | 0.777 (0.701 to 0.871) |

F10_B, norm of the sweep's gradient over the norm of the exact gradient:

| variant | where | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters |
|---|---|---|---|---|---|---|---|---|---|
| V0 | every map | 0.16 (0.11 to 0.24) | 0.25 (0.18 to 0.34) | 0.17 (0.12 to 0.24) | 0.17 (0.12 to 0.22) | 0.39 (0.33 to 0.46) | 0.39 (0.34 to 0.46) | 1.00 (1.00 to 1.00) | 0.58 (0.37 to 0.74) |
| V1 | every map | 0.11 (0.08 to 0.13) | 0.17 (0.11 to 0.22) | 0.18 (0.12 to 0.25) | 0.18 (0.12 to 0.22) | 0.39 (0.33 to 0.46) | 0.39 (0.34 to 0.46) | 1.00 (1.00 to 1.00) | 0.57 (0.35 to 0.74) |
| V2 | every map | 0.16 (0.12 to 0.22) | 0.26 (0.18 to 0.34) | 0.30 (0.21 to 0.42) | 0.30 (0.22 to 0.38) | 0.68 (0.58 to 0.78) | 0.67 (0.60 to 0.78) | 1.00 (1.00 to 1.00) | 0.64 (0.45 to 0.78) |
| V3 | every map | 0.17 (0.11 to 0.24) | 0.28 (0.18 to 0.42) | 0.20 (0.15 to 0.27) | 0.20 (0.15 to 0.25) | 0.48 (0.44 to 0.52) | 0.48 (0.40 to 0.54) | 1.00 (1.00 to 1.00) | 0.60 (0.39 to 0.76) |
| V4 | every map | 0.18 (0.13 to 0.23) | 0.29 (0.20 to 0.45) | 0.36 (0.27 to 0.49) | 0.36 (0.26 to 0.45) | 0.83 (0.75 to 0.91) | 0.83 (0.70 to 0.92) | 1.00 (1.00 to 1.00) | 0.68 (0.48 to 0.83) |
| V2 | GRU maps | 0.10 (0.07 to 0.13) | 0.15 (0.10 to 0.20) | 0.18 (0.12 to 0.25) | 0.18 (0.12 to 0.23) | 0.39 (0.33 to 0.46) | 0.39 (0.34 to 0.46) | 1.00 (1.00 to 1.00) | 0.57 (0.35 to 0.74) |
| V3 | GRU maps | 0.17 (0.11 to 0.24) | 0.26 (0.18 to 0.35) | 0.17 (0.12 to 0.24) | 0.17 (0.12 to 0.22) | 0.39 (0.33 to 0.46) | 0.39 (0.34 to 0.46) | 1.00 (1.00 to 1.00) | 0.58 (0.38 to 0.74) |
| V4 | GRU maps | 0.10 (0.07 to 0.13) | 0.15 (0.10 to 0.21) | 0.18 (0.12 to 0.25) | 0.18 (0.12 to 0.23) | 0.39 (0.33 to 0.46) | 0.39 (0.34 to 0.46) | 1.00 (1.00 to 1.00) | 0.57 (0.35 to 0.74) |
| V5 | GRU maps | 0.17 (0.13 to 0.21) | 0.26 (0.18 to 0.31) | 0.24 (0.15 to 0.32) | 0.23 (0.15 to 0.32) | 0.39 (0.33 to 0.46) | 0.39 (0.34 to 0.46) | 1.00 (1.00 to 1.00) | 0.59 (0.41 to 0.74) |

F10_B, cosine for each gate's block of the two GRU maps:

| variant | where | input map, r | input map, z | input map, n | recurrent map, r | recurrent map, z | recurrent map, n |
|---|---|---|---|---|---|---|---|
| V0 | every map | 0.468 (0.298 to 0.749) | 0.624 (0.477 to 0.750) | 0.581 (0.473 to 0.677) | 0.460 (0.335 to 0.611) | 0.657 (0.583 to 0.727) | 0.588 (0.452 to 0.653) |
| V1 | every map | 0.486 (0.325 to 0.755) | 0.633 (0.498 to 0.764) | 0.593 (0.489 to 0.685) | 0.480 (0.340 to 0.628) | 0.665 (0.589 to 0.741) | 0.599 (0.471 to 0.661) |
| V2 | every map | 0.537 (0.436 to 0.754) | 0.656 (0.521 to 0.767) | 0.624 (0.523 to 0.723) | 0.537 (0.357 to 0.649) | 0.686 (0.600 to 0.750) | 0.630 (0.499 to 0.689) |
| V3 | every map | 0.569 (0.366 to 0.744) | 0.701 (0.564 to 0.815) | 0.633 (0.458 to 0.732) | 0.588 (0.424 to 0.793) | 0.731 (0.629 to 0.800) | 0.645 (0.524 to 0.731) |
| V4 | every map | 0.652 (0.493 to 0.839) | 0.736 (0.588 to 0.838) | 0.678 (0.526 to 0.774) | 0.674 (0.480 to 0.893) | 0.756 (0.663 to 0.831) | 0.688 (0.560 to 0.784) |
| V2 | GRU maps | 0.512 (0.375 to 0.774) | 0.643 (0.509 to 0.764) | 0.606 (0.507 to 0.696) | 0.511 (0.352 to 0.640) | 0.673 (0.595 to 0.745) | 0.612 (0.483 to 0.674) |
| V3 | GRU maps | 0.488 (0.320 to 0.766) | 0.652 (0.512 to 0.783) | 0.611 (0.519 to 0.697) | 0.482 (0.348 to 0.648) | 0.680 (0.599 to 0.753) | 0.620 (0.513 to 0.683) |
| V4 | GRU maps | 0.535 (0.397 to 0.797) | 0.681 (0.517 to 0.796) | 0.645 (0.569 to 0.726) | 0.535 (0.359 to 0.685) | 0.701 (0.614 to 0.772) | 0.652 (0.559 to 0.717) |
| V5 | GRU maps | 0.753 (0.592 to 0.913) | 0.814 (0.716 to 0.885) | 0.791 (0.708 to 0.873) | 0.766 (0.570 to 0.893) | 0.819 (0.763 to 0.878) | 0.794 (0.715 to 0.890) |

#### in16_start

E49 in16 at its start. Loss on the 16 batches: mean 5.5901 (least 5.5809, greatest 5.5996); first batch 5.5738. Norm of the exact gradient: mean 1.979 (1.874 to 2.099).

in16_start, cosine with the exact gradient, variant at every map:

| variant | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters | all but l3 |
|---|---|---|---|---|---|---|---|---|---|
| V0, first batch | 0.518 | 0.707 | 0.742 | 0.731 | 0.865 | 0.867 | 1.000 | 0.941 | 0.862 |
| V0 | 0.558 (0.411 to 0.649) | 0.716 (0.701 to 0.728) | 0.745 (0.740 to 0.752) | 0.734 (0.728 to 0.741) | 0.868 (0.865 to 0.871) | 0.868 (0.864 to 0.872) | 1.000 (1.000 to 1.000) | 0.941 (0.938 to 0.942) | 0.864 (0.860 to 0.867) |
| V1 | 0.558 (0.411 to 0.649) | 0.716 (0.701 to 0.728) | 0.745 (0.740 to 0.752) | 0.734 (0.728 to 0.741) | 0.868 (0.865 to 0.871) | 0.868 (0.864 to 0.872) | 1.000 (1.000 to 1.000) | 0.941 (0.938 to 0.942) | 0.864 (0.860 to 0.867) |
| V2 | 0.541 (0.343 to 0.658) | 0.720 (0.704 to 0.731) | 0.749 (0.743 to 0.756) | 0.739 (0.734 to 0.747) | 0.868 (0.865 to 0.871) | 0.868 (0.864 to 0.872) | 1.000 (1.000 to 1.000) | 0.941 (0.939 to 0.942) | 0.864 (0.861 to 0.867) |
| V3 | 0.578 (0.450 to 0.663) | 0.718 (0.703 to 0.730) | 0.742 (0.736 to 0.749) | 0.728 (0.723 to 0.736) | 0.868 (0.866 to 0.871) | 0.869 (0.865 to 0.873) | 1.000 (1.000 to 1.000) | 0.941 (0.939 to 0.943) | 0.864 (0.861 to 0.868) |
| V4 | 0.572 (0.391 to 0.681) | 0.722 (0.706 to 0.733) | 0.746 (0.739 to 0.754) | 0.734 (0.729 to 0.742) | 0.869 (0.866 to 0.871) | 0.869 (0.865 to 0.873) | 1.000 (1.000 to 1.000) | 0.941 (0.939 to 0.943) | 0.865 (0.861 to 0.868) |

in16_start, cosine with the exact gradient, variant at the two GRU maps only (V0 at the other maps; V5 here is the upper limit for a scale at the GRU maps):

| variant | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters | all but l3 |
|---|---|---|---|---|---|---|---|---|---|
| V1 | 0.558 (0.411 to 0.649) | 0.716 (0.701 to 0.728) | 0.745 (0.740 to 0.752) | 0.734 (0.728 to 0.741) | 0.868 (0.865 to 0.871) | 0.868 (0.864 to 0.872) | 1.000 (1.000 to 1.000) | 0.941 (0.938 to 0.942) | 0.864 (0.860 to 0.867) |
| V2 | 0.532 (0.378 to 0.631) | 0.718 (0.702 to 0.729) | 0.745 (0.740 to 0.753) | 0.734 (0.729 to 0.742) | 0.868 (0.865 to 0.871) | 0.868 (0.864 to 0.872) | 1.000 (1.000 to 1.000) | 0.941 (0.938 to 0.942) | 0.864 (0.860 to 0.867) |
| V3 | 0.562 (0.427 to 0.648) | 0.718 (0.703 to 0.729) | 0.745 (0.740 to 0.752) | 0.733 (0.727 to 0.740) | 0.868 (0.865 to 0.871) | 0.868 (0.864 to 0.872) | 1.000 (1.000 to 1.000) | 0.941 (0.938 to 0.942) | 0.864 (0.860 to 0.867) |
| V4 | 0.535 (0.391 to 0.629) | 0.720 (0.705 to 0.731) | 0.746 (0.740 to 0.754) | 0.734 (0.728 to 0.742) | 0.868 (0.865 to 0.871) | 0.868 (0.864 to 0.872) | 1.000 (1.000 to 1.000) | 0.941 (0.938 to 0.942) | 0.864 (0.860 to 0.867) |
| V5 | 0.525 (0.367 to 0.652) | 0.748 (0.731 to 0.760) | 0.756 (0.750 to 0.763) | 0.743 (0.737 to 0.751) | 0.868 (0.865 to 0.871) | 0.868 (0.864 to 0.872) | 1.000 (1.000 to 1.000) | 0.941 (0.939 to 0.942) | 0.864 (0.861 to 0.868) |

in16_start, norm of the sweep's gradient over the norm of the exact gradient:

| variant | where | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters |
|---|---|---|---|---|---|---|---|---|---|
| V0 | every map | 0.82 (0.66 to 1.06) | 0.76 (0.73 to 0.78) | 0.68 (0.66 to 0.71) | 0.66 (0.65 to 0.68) | 0.85 (0.84 to 0.86) | 0.87 (0.86 to 0.88) | 1.00 (1.00 to 1.00) | 0.94 (0.94 to 0.94) |
| V1 | every map | 0.82 (0.66 to 1.06) | 0.76 (0.73 to 0.78) | 0.68 (0.66 to 0.71) | 0.66 (0.65 to 0.68) | 0.85 (0.84 to 0.86) | 0.87 (0.86 to 0.88) | 1.00 (1.00 to 1.00) | 0.94 (0.94 to 0.94) |
| V2 | every map | 0.79 (0.60 to 1.08) | 0.75 (0.73 to 0.78) | 0.67 (0.65 to 0.70) | 0.65 (0.64 to 0.67) | 0.84 (0.83 to 0.85) | 0.86 (0.85 to 0.86) | 1.00 (1.00 to 1.00) | 0.93 (0.93 to 0.94) |
| V3 | every map | 0.80 (0.67 to 1.03) | 0.76 (0.73 to 0.78) | 0.68 (0.66 to 0.70) | 0.65 (0.64 to 0.68) | 0.85 (0.85 to 0.86) | 0.87 (0.86 to 0.88) | 1.00 (1.00 to 1.00) | 0.94 (0.94 to 0.94) |
| V4 | every map | 0.77 (0.60 to 1.03) | 0.76 (0.73 to 0.78) | 0.67 (0.65 to 0.69) | 0.64 (0.63 to 0.67) | 0.84 (0.83 to 0.85) | 0.86 (0.85 to 0.86) | 1.00 (1.00 to 1.00) | 0.93 (0.93 to 0.94) |
| V2 | GRU maps | 0.82 (0.66 to 1.06) | 0.76 (0.74 to 0.78) | 0.68 (0.66 to 0.71) | 0.66 (0.65 to 0.68) | 0.85 (0.84 to 0.86) | 0.87 (0.86 to 0.88) | 1.00 (1.00 to 1.00) | 0.94 (0.94 to 0.94) |
| V3 | GRU maps | 0.81 (0.65 to 1.04) | 0.76 (0.74 to 0.78) | 0.68 (0.66 to 0.71) | 0.66 (0.65 to 0.68) | 0.85 (0.84 to 0.86) | 0.87 (0.86 to 0.88) | 1.00 (1.00 to 1.00) | 0.94 (0.94 to 0.94) |
| V4 | GRU maps | 0.81 (0.65 to 1.04) | 0.77 (0.74 to 0.79) | 0.68 (0.67 to 0.71) | 0.66 (0.65 to 0.68) | 0.85 (0.84 to 0.86) | 0.87 (0.86 to 0.88) | 1.00 (1.00 to 1.00) | 0.94 (0.94 to 0.94) |
| V5 | GRU maps | 0.75 (0.63 to 0.94) | 0.77 (0.75 to 0.79) | 0.69 (0.67 to 0.71) | 0.66 (0.65 to 0.69) | 0.85 (0.84 to 0.86) | 0.87 (0.86 to 0.88) | 1.00 (1.00 to 1.00) | 0.94 (0.94 to 0.94) |

in16_start, cosine for each gate's block of the two GRU maps:

| variant | where | input map, r | input map, z | input map, n | recurrent map, r | recurrent map, z | recurrent map, n |
|---|---|---|---|---|---|---|---|
| V0 | every map | 0.764 (0.741 to 0.783) | 0.731 (0.704 to 0.745) | 0.746 (0.741 to 0.754) | 0.757 (0.737 to 0.769) | 0.721 (0.684 to 0.741) | 0.734 (0.728 to 0.741) |
| V1 | every map | 0.764 (0.741 to 0.783) | 0.731 (0.704 to 0.745) | 0.746 (0.741 to 0.754) | 0.757 (0.737 to 0.769) | 0.721 (0.684 to 0.741) | 0.734 (0.728 to 0.741) |
| V2 | every map | 0.767 (0.743 to 0.785) | 0.733 (0.706 to 0.748) | 0.750 (0.744 to 0.757) | 0.761 (0.741 to 0.773) | 0.724 (0.687 to 0.743) | 0.740 (0.734 to 0.747) |
| V3 | every map | 0.755 (0.732 to 0.774) | 0.729 (0.702 to 0.742) | 0.743 (0.737 to 0.751) | 0.747 (0.726 to 0.759) | 0.720 (0.684 to 0.739) | 0.729 (0.723 to 0.736) |
| V4 | every map | 0.758 (0.734 to 0.776) | 0.731 (0.705 to 0.745) | 0.746 (0.740 to 0.754) | 0.751 (0.730 to 0.763) | 0.723 (0.687 to 0.742) | 0.735 (0.729 to 0.742) |
| V2 | GRU maps | 0.765 (0.742 to 0.783) | 0.731 (0.704 to 0.746) | 0.746 (0.741 to 0.754) | 0.758 (0.738 to 0.770) | 0.722 (0.685 to 0.741) | 0.735 (0.729 to 0.742) |
| V3 | GRU maps | 0.763 (0.740 to 0.782) | 0.731 (0.704 to 0.745) | 0.746 (0.740 to 0.753) | 0.755 (0.736 to 0.768) | 0.722 (0.685 to 0.741) | 0.733 (0.727 to 0.740) |
| V4 | GRU maps | 0.764 (0.741 to 0.782) | 0.731 (0.704 to 0.746) | 0.746 (0.740 to 0.754) | 0.757 (0.737 to 0.769) | 0.723 (0.686 to 0.742) | 0.734 (0.728 to 0.742) |
| V5 | GRU maps | 0.776 (0.756 to 0.794) | 0.738 (0.710 to 0.753) | 0.757 (0.751 to 0.764) | 0.767 (0.748 to 0.778) | 0.729 (0.697 to 0.748) | 0.744 (0.737 to 0.751) |

#### in16_S

E49 in16 (16 of 512 input values a cell), sweep: did not learn. Loss on the 16 batches: mean 1.0361 (least 0.9380, greatest 1.1664); first batch 0.9511. Norm of the exact gradient: mean 3.768 (1.625 to 6.018).

in16_S, cosine with the exact gradient, variant at every map:

| variant | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters | all but l3 |
|---|---|---|---|---|---|---|---|---|---|
| V0, first batch | 0.275 | 0.297 | 0.619 | 0.605 | 0.544 | 0.603 | 1.000 | 0.597 | 0.606 |
| V0 | 0.367 (0.144 to 0.498) | 0.298 (0.206 to 0.345) | 0.603 (0.462 to 0.729) | 0.651 (0.472 to 0.766) | 0.606 (0.506 to 0.690) | 0.643 (0.550 to 0.718) | 1.000 (1.000 to 1.000) | 0.612 (0.442 to 0.729) | 0.587 (0.371 to 0.726) |
| V1 | 0.317 (-0.004 to 0.487) | 0.252 (0.169 to 0.316) | 0.560 (0.431 to 0.699) | 0.634 (0.473 to 0.740) | 0.606 (0.506 to 0.690) | 0.643 (0.550 to 0.718) | 1.000 (1.000 to 1.000) | 0.596 (0.458 to 0.719) | 0.588 (0.411 to 0.734) |
| V2 | 0.697 (0.540 to 0.826) | 0.624 (0.482 to 0.675) | 0.676 (0.568 to 0.776) | 0.762 (0.673 to 0.818) | 0.694 (0.621 to 0.747) | 0.710 (0.648 to 0.758) | 1.000 (1.000 to 1.000) | 0.637 (0.559 to 0.722) | 0.742 (0.591 to 0.829) |
| V3 | 0.476 (0.168 to 0.602) | 0.477 (0.309 to 0.580) | 0.640 (0.509 to 0.781) | 0.641 (0.472 to 0.782) | 0.572 (0.437 to 0.709) | 0.602 (0.515 to 0.683) | 1.000 (1.000 to 1.000) | 0.572 (0.297 to 0.743) | 0.543 (0.251 to 0.742) |
| V4 | 0.741 (0.616 to 0.823) | 0.714 (0.519 to 0.772) | 0.729 (0.630 to 0.840) | 0.754 (0.693 to 0.835) | 0.588 (0.473 to 0.710) | 0.611 (0.537 to 0.687) | 1.000 (1.000 to 1.000) | 0.700 (0.603 to 0.804) | 0.727 (0.631 to 0.841) |

in16_S, cosine with the exact gradient, variant at the two GRU maps only (V0 at the other maps; V5 here is the upper limit for a scale at the GRU maps):

| variant | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters | all but l3 |
|---|---|---|---|---|---|---|---|---|---|
| V1 | 0.317 (-0.004 to 0.487) | 0.252 (0.169 to 0.316) | 0.560 (0.431 to 0.699) | 0.634 (0.473 to 0.740) | 0.606 (0.506 to 0.690) | 0.643 (0.550 to 0.718) | 1.000 (1.000 to 1.000) | 0.596 (0.458 to 0.719) | 0.588 (0.411 to 0.734) |
| V2 | 0.473 (0.234 to 0.590) | 0.527 (0.377 to 0.593) | 0.594 (0.432 to 0.736) | 0.654 (0.510 to 0.773) | 0.606 (0.506 to 0.690) | 0.643 (0.550 to 0.718) | 1.000 (1.000 to 1.000) | 0.637 (0.488 to 0.777) | 0.633 (0.440 to 0.793) |
| V3 | 0.465 (0.252 to 0.579) | 0.555 (0.400 to 0.633) | 0.715 (0.528 to 0.822) | 0.709 (0.534 to 0.833) | 0.606 (0.506 to 0.690) | 0.643 (0.550 to 0.718) | 1.000 (1.000 to 1.000) | 0.685 (0.509 to 0.823) | 0.663 (0.452 to 0.819) |
| V4 | 0.638 (0.542 to 0.751) | 0.721 (0.576 to 0.790) | 0.714 (0.549 to 0.815) | 0.710 (0.555 to 0.838) | 0.606 (0.506 to 0.690) | 0.643 (0.550 to 0.718) | 1.000 (1.000 to 1.000) | 0.706 (0.534 to 0.844) | 0.689 (0.481 to 0.843) |
| V5 | 0.738 (0.612 to 0.835) | 0.839 (0.727 to 0.912) | 0.825 (0.642 to 0.920) | 0.787 (0.618 to 0.928) | 0.606 (0.506 to 0.690) | 0.643 (0.550 to 0.718) | 1.000 (1.000 to 1.000) | 0.782 (0.560 to 0.925) | 0.766 (0.525 to 0.923) |

in16_S, norm of the sweep's gradient over the norm of the exact gradient:

| variant | where | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters |
|---|---|---|---|---|---|---|---|---|---|
| V0 | every map | 0.18 (0.14 to 0.24) | 0.29 (0.25 to 0.37) | 0.30 (0.26 to 0.39) | 0.44 (0.36 to 0.56) | 1.66 (1.46 to 1.92) | 1.64 (1.46 to 1.84) | 1.00 (1.00 to 1.00) | 0.45 (0.37 to 0.65) |
| V1 | every map | 0.12 (0.10 to 0.17) | 0.14 (0.12 to 0.18) | 0.23 (0.19 to 0.29) | 0.35 (0.28 to 0.44) | 1.66 (1.46 to 1.92) | 1.64 (1.46 to 1.84) | 1.00 (1.00 to 1.00) | 0.35 (0.30 to 0.53) |
| V2 | every map | 0.10 (0.08 to 0.12) | 0.11 (0.09 to 0.13) | 0.13 (0.11 to 0.15) | 0.20 (0.18 to 0.23) | 0.71 (0.63 to 0.77) | 0.69 (0.62 to 0.74) | 1.00 (1.00 to 1.00) | 0.28 (0.21 to 0.44) |
| V3 | every map | 0.26 (0.20 to 0.33) | 0.60 (0.51 to 0.85) | 0.57 (0.41 to 0.92) | 0.69 (0.52 to 0.99) | 2.63 (2.16 to 3.29) | 2.48 (2.09 to 2.88) | 1.00 (1.00 to 1.00) | 0.69 (0.44 to 1.23) |
| V4 | every map | 0.31 (0.23 to 0.37) | 0.37 (0.29 to 0.49) | 0.26 (0.21 to 0.39) | 0.32 (0.26 to 0.43) | 1.09 (0.91 to 1.28) | 1.01 (0.89 to 1.14) | 1.00 (1.00 to 1.00) | 0.37 (0.26 to 0.62) |
| V2 | GRU maps | 0.13 (0.10 to 0.19) | 0.20 (0.17 to 0.25) | 0.24 (0.20 to 0.30) | 0.37 (0.29 to 0.46) | 1.66 (1.46 to 1.92) | 1.64 (1.46 to 1.84) | 1.00 (1.00 to 1.00) | 0.37 (0.31 to 0.55) |
| V3 | GRU maps | 0.22 (0.17 to 0.29) | 0.53 (0.41 to 0.65) | 0.52 (0.42 to 0.63) | 0.61 (0.50 to 0.75) | 1.66 (1.46 to 1.92) | 1.64 (1.46 to 1.84) | 1.00 (1.00 to 1.00) | 0.64 (0.48 to 0.89) |
| V4 | GRU maps | 0.21 (0.15 to 0.25) | 0.57 (0.44 to 0.69) | 0.40 (0.32 to 0.46) | 0.49 (0.39 to 0.62) | 1.66 (1.46 to 1.92) | 1.64 (1.46 to 1.84) | 1.00 (1.00 to 1.00) | 0.53 (0.43 to 0.73) |
| V5 | GRU maps | 0.23 (0.21 to 0.27) | 0.74 (0.62 to 0.84) | 0.71 (0.59 to 0.82) | 0.72 (0.58 to 0.89) | 1.66 (1.46 to 1.92) | 1.64 (1.46 to 1.84) | 1.00 (1.00 to 1.00) | 0.75 (0.62 to 0.96) |

in16_S, cosine for each gate's block of the two GRU maps:

| variant | where | input map, r | input map, z | input map, n | recurrent map, r | recurrent map, z | recurrent map, n |
|---|---|---|---|---|---|---|---|
| V0 | every map | 0.648 (0.546 to 0.755) | 0.734 (0.483 to 0.843) | 0.598 (0.452 to 0.723) | 0.627 (0.409 to 0.805) | 0.652 (0.412 to 0.768) | 0.652 (0.473 to 0.766) |
| V1 | every map | 0.603 (0.499 to 0.726) | 0.727 (0.498 to 0.860) | 0.554 (0.420 to 0.692) | 0.604 (0.395 to 0.794) | 0.621 (0.395 to 0.736) | 0.634 (0.474 to 0.740) |
| V2 | every map | 0.713 (0.649 to 0.799) | 0.790 (0.620 to 0.888) | 0.670 (0.550 to 0.771) | 0.731 (0.610 to 0.835) | 0.692 (0.551 to 0.831) | 0.762 (0.674 to 0.818) |
| V3 | every map | 0.684 (0.545 to 0.777) | 0.642 (0.248 to 0.809) | 0.641 (0.510 to 0.782) | 0.654 (0.330 to 0.826) | 0.656 (0.502 to 0.818) | 0.641 (0.471 to 0.783) |
| V4 | every map | 0.756 (0.668 to 0.846) | 0.782 (0.646 to 0.883) | 0.728 (0.616 to 0.843) | 0.766 (0.527 to 0.862) | 0.744 (0.561 to 0.838) | 0.754 (0.692 to 0.836) |
| V2 | GRU maps | 0.637 (0.548 to 0.746) | 0.745 (0.535 to 0.891) | 0.588 (0.417 to 0.728) | 0.635 (0.471 to 0.833) | 0.644 (0.443 to 0.771) | 0.654 (0.509 to 0.773) |
| V3 | GRU maps | 0.756 (0.665 to 0.827) | 0.796 (0.667 to 0.871) | 0.711 (0.520 to 0.822) | 0.707 (0.591 to 0.877) | 0.745 (0.623 to 0.828) | 0.709 (0.533 to 0.833) |
| V4 | GRU maps | 0.748 (0.665 to 0.817) | 0.790 (0.692 to 0.885) | 0.711 (0.535 to 0.814) | 0.711 (0.607 to 0.888) | 0.732 (0.576 to 0.816) | 0.710 (0.554 to 0.838) |
| V5 | GRU maps | 0.854 (0.764 to 0.919) | 0.859 (0.637 to 0.937) | 0.823 (0.642 to 0.921) | 0.780 (0.645 to 0.938) | 0.853 (0.736 to 0.949) | 0.787 (0.617 to 0.928) |

#### in16_B

E49 in16, trained under back-propagation. Loss on the 16 batches: mean 0.0829 (least 0.0600, greatest 0.1089); first batch 0.1050. Norm of the exact gradient: mean 0.635 (0.389 to 0.890).

in16_B, cosine with the exact gradient, variant at every map:

| variant | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters | all but l3 |
|---|---|---|---|---|---|---|---|---|---|
| V0, first batch | 0.628 | 0.362 | 0.727 | 0.706 | 0.752 | 0.759 | 1.000 | 0.628 | 0.601 |
| V0 | 0.442 (0.289 to 0.588) | 0.309 (0.177 to 0.415) | 0.523 (0.257 to 0.715) | 0.582 (0.366 to 0.751) | 0.700 (0.651 to 0.757) | 0.706 (0.629 to 0.776) | 1.000 (1.000 to 1.000) | 0.640 (0.559 to 0.740) | 0.496 (0.378 to 0.608) |
| V1 | 0.358 (0.195 to 0.525) | 0.294 (0.175 to 0.430) | 0.531 (0.265 to 0.723) | 0.587 (0.370 to 0.757) | 0.700 (0.651 to 0.757) | 0.706 (0.629 to 0.776) | 1.000 (1.000 to 1.000) | 0.639 (0.561 to 0.739) | 0.501 (0.381 to 0.616) |
| V2 | 0.405 (0.223 to 0.595) | 0.363 (0.221 to 0.478) | 0.567 (0.310 to 0.741) | 0.631 (0.443 to 0.766) | 0.700 (0.652 to 0.758) | 0.707 (0.628 to 0.775) | 1.000 (1.000 to 1.000) | 0.670 (0.603 to 0.761) | 0.511 (0.396 to 0.619) |
| V3 | 0.370 (0.181 to 0.518) | 0.272 (0.166 to 0.363) | 0.504 (0.314 to 0.672) | 0.550 (0.367 to 0.735) | 0.602 (0.554 to 0.677) | 0.611 (0.547 to 0.718) | 1.000 (1.000 to 1.000) | 0.629 (0.560 to 0.732) | 0.430 (0.346 to 0.545) |
| V4 | 0.360 (0.113 to 0.548) | 0.328 (0.211 to 0.442) | 0.554 (0.366 to 0.696) | 0.607 (0.447 to 0.747) | 0.603 (0.555 to 0.676) | 0.612 (0.546 to 0.717) | 1.000 (1.000 to 1.000) | 0.651 (0.596 to 0.745) | 0.451 (0.360 to 0.561) |

in16_B, cosine with the exact gradient, variant at the two GRU maps only (V0 at the other maps; V5 here is the upper limit for a scale at the GRU maps):

| variant | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters | all but l3 |
|---|---|---|---|---|---|---|---|---|---|
| V1 | 0.358 (0.195 to 0.525) | 0.294 (0.175 to 0.430) | 0.531 (0.265 to 0.723) | 0.587 (0.370 to 0.757) | 0.700 (0.651 to 0.757) | 0.706 (0.629 to 0.776) | 1.000 (1.000 to 1.000) | 0.639 (0.561 to 0.739) | 0.501 (0.381 to 0.616) |
| V2 | 0.391 (0.237 to 0.577) | 0.331 (0.192 to 0.464) | 0.537 (0.263 to 0.728) | 0.592 (0.375 to 0.758) | 0.700 (0.651 to 0.757) | 0.706 (0.629 to 0.776) | 1.000 (1.000 to 1.000) | 0.639 (0.564 to 0.740) | 0.502 (0.384 to 0.612) |
| V3 | 0.445 (0.273 to 0.618) | 0.294 (0.173 to 0.392) | 0.544 (0.286 to 0.727) | 0.595 (0.379 to 0.753) | 0.700 (0.651 to 0.757) | 0.706 (0.629 to 0.776) | 1.000 (1.000 to 1.000) | 0.641 (0.566 to 0.741) | 0.500 (0.384 to 0.609) |
| V4 | 0.383 (0.225 to 0.587) | 0.321 (0.187 to 0.448) | 0.562 (0.293 to 0.734) | 0.608 (0.392 to 0.759) | 0.700 (0.651 to 0.757) | 0.706 (0.629 to 0.776) | 1.000 (1.000 to 1.000) | 0.642 (0.574 to 0.741) | 0.510 (0.393 to 0.616) |
| V5 | 0.738 (0.501 to 0.847) | 0.742 (0.483 to 0.847) | 0.761 (0.435 to 0.897) | 0.756 (0.480 to 0.894) | 0.700 (0.651 to 0.757) | 0.706 (0.629 to 0.776) | 1.000 (1.000 to 1.000) | 0.758 (0.694 to 0.836) | 0.732 (0.486 to 0.873) |

in16_B, norm of the sweep's gradient over the norm of the exact gradient:

| variant | where | byte table | l1 | input map | recurrent map | lm | l2 | l3 | all parameters |
|---|---|---|---|---|---|---|---|---|---|
| V0 | every map | 0.11 (0.07 to 0.14) | 0.18 (0.13 to 0.22) | 0.10 (0.09 to 0.13) | 0.14 (0.12 to 0.17) | 0.43 (0.37 to 0.48) | 0.44 (0.37 to 0.48) | 1.00 (1.00 to 1.00) | 0.56 (0.45 to 0.70) |
| V1 | every map | 0.08 (0.06 to 0.10) | 0.11 (0.07 to 0.14) | 0.11 (0.09 to 0.13) | 0.14 (0.12 to 0.18) | 0.43 (0.37 to 0.48) | 0.44 (0.37 to 0.48) | 1.00 (1.00 to 1.00) | 0.56 (0.44 to 0.69) |
| V2 | every map | 0.13 (0.09 to 0.18) | 0.15 (0.10 to 0.19) | 0.16 (0.13 to 0.20) | 0.20 (0.16 to 0.26) | 0.65 (0.56 to 0.72) | 0.66 (0.57 to 0.73) | 1.00 (1.00 to 1.00) | 0.59 (0.49 to 0.72) |
| V3 | every map | 0.10 (0.08 to 0.14) | 0.21 (0.15 to 0.27) | 0.12 (0.09 to 0.14) | 0.14 (0.10 to 0.18) | 0.54 (0.45 to 0.59) | 0.53 (0.45 to 0.62) | 1.00 (1.00 to 1.00) | 0.57 (0.46 to 0.71) |
| V4 | every map | 0.13 (0.09 to 0.17) | 0.18 (0.13 to 0.28) | 0.18 (0.14 to 0.22) | 0.22 (0.17 to 0.27) | 0.81 (0.69 to 0.91) | 0.81 (0.69 to 0.96) | 1.00 (1.00 to 1.00) | 0.63 (0.52 to 0.76) |
| V2 | GRU maps | 0.08 (0.06 to 0.11) | 0.10 (0.07 to 0.13) | 0.11 (0.09 to 0.13) | 0.14 (0.11 to 0.17) | 0.43 (0.37 to 0.48) | 0.44 (0.37 to 0.48) | 1.00 (1.00 to 1.00) | 0.56 (0.44 to 0.69) |
| V3 | GRU maps | 0.10 (0.07 to 0.13) | 0.18 (0.13 to 0.22) | 0.11 (0.08 to 0.13) | 0.14 (0.11 to 0.17) | 0.43 (0.37 to 0.48) | 0.44 (0.37 to 0.48) | 1.00 (1.00 to 1.00) | 0.56 (0.45 to 0.70) |
| V4 | GRU maps | 0.08 (0.06 to 0.10) | 0.10 (0.07 to 0.13) | 0.11 (0.09 to 0.13) | 0.14 (0.11 to 0.17) | 0.43 (0.37 to 0.48) | 0.44 (0.37 to 0.48) | 1.00 (1.00 to 1.00) | 0.56 (0.44 to 0.69) |
| V5 | GRU maps | 0.18 (0.13 to 0.25) | 0.25 (0.20 to 0.35) | 0.26 (0.22 to 0.38) | 0.28 (0.22 to 0.38) | 0.43 (0.37 to 0.48) | 0.44 (0.37 to 0.48) | 1.00 (1.00 to 1.00) | 0.59 (0.52 to 0.71) |

in16_B, cosine for each gate's block of the two GRU maps:

| variant | where | input map, r | input map, z | input map, n | recurrent map, r | recurrent map, z | recurrent map, n |
|---|---|---|---|---|---|---|---|
| V0 | every map | 0.508 (0.404 to 0.727) | 0.573 (0.270 to 0.756) | 0.521 (0.256 to 0.714) | 0.577 (0.432 to 0.725) | 0.554 (0.334 to 0.729) | 0.582 (0.366 to 0.752) |
| V1 | every map | 0.517 (0.412 to 0.735) | 0.581 (0.279 to 0.766) | 0.530 (0.264 to 0.724) | 0.583 (0.439 to 0.730) | 0.562 (0.341 to 0.737) | 0.588 (0.370 to 0.759) |
| V2 | every map | 0.552 (0.439 to 0.744) | 0.620 (0.371 to 0.781) | 0.565 (0.308 to 0.739) | 0.620 (0.510 to 0.746) | 0.603 (0.439 to 0.738) | 0.632 (0.442 to 0.768) |
| V3 | every map | 0.527 (0.384 to 0.733) | 0.499 (0.127 to 0.637) | 0.504 (0.312 to 0.672) | 0.577 (0.431 to 0.779) | 0.507 (0.323 to 0.737) | 0.551 (0.368 to 0.735) |
| V4 | every map | 0.572 (0.451 to 0.748) | 0.557 (0.220 to 0.690) | 0.554 (0.363 to 0.697) | 0.625 (0.519 to 0.794) | 0.570 (0.447 to 0.738) | 0.608 (0.446 to 0.747) |
| V2 | GRU maps | 0.519 (0.402 to 0.734) | 0.584 (0.275 to 0.763) | 0.535 (0.262 to 0.727) | 0.585 (0.444 to 0.734) | 0.558 (0.345 to 0.726) | 0.593 (0.375 to 0.759) |
| V3 | GRU maps | 0.530 (0.412 to 0.750) | 0.574 (0.278 to 0.748) | 0.543 (0.286 to 0.726) | 0.591 (0.426 to 0.754) | 0.571 (0.354 to 0.781) | 0.596 (0.378 to 0.754) |
| V4 | GRU maps | 0.544 (0.410 to 0.757) | 0.588 (0.281 to 0.754) | 0.561 (0.292 to 0.734) | 0.601 (0.442 to 0.764) | 0.578 (0.371 to 0.780) | 0.608 (0.392 to 0.760) |
| V5 | GRU maps | 0.768 (0.640 to 0.909) | 0.788 (0.494 to 0.904) | 0.760 (0.433 to 0.897) | 0.764 (0.598 to 0.917) | 0.776 (0.588 to 0.943) | 0.756 (0.478 to 0.893) |

#### C-dense_start

E50 C-dense at its start (fixed code). Loss on the 16 batches: mean 5.5872 (least 5.5572, greatest 5.6022); first batch 5.5968. Norm of the exact gradient: mean 1.797 (1.739 to 1.843).

C-dense_start, cosine with the exact gradient, variant at every map:

| variant | input map | recurrent map | lm | l2 | l3 | all parameters | all but l3 |
|---|---|---|---|---|---|---|---|
| V0, first batch | 0.757 | 0.723 | 0.867 | 0.871 | 1.000 | 0.933 | 0.853 |
| V0 | 0.752 (0.745 to 0.759) | 0.726 (0.718 to 0.737) | 0.869 (0.866 to 0.872) | 0.870 (0.867 to 0.876) | 1.000 (1.000 to 1.000) | 0.933 (0.932 to 0.935) | 0.852 (0.849 to 0.858) |
| V1 | 0.752 (0.745 to 0.759) | 0.726 (0.718 to 0.737) | 0.869 (0.866 to 0.872) | 0.870 (0.867 to 0.876) | 1.000 (1.000 to 1.000) | 0.933 (0.932 to 0.935) | 0.853 (0.849 to 0.858) |
| V2 | 0.758 (0.751 to 0.764) | 0.732 (0.723 to 0.743) | 0.869 (0.866 to 0.872) | 0.870 (0.867 to 0.876) | 1.000 (1.000 to 1.000) | 0.933 (0.932 to 0.935) | 0.853 (0.850 to 0.858) |
| V3 | 0.750 (0.743 to 0.757) | 0.723 (0.715 to 0.734) | 0.869 (0.866 to 0.872) | 0.871 (0.867 to 0.877) | 1.000 (1.000 to 1.000) | 0.933 (0.932 to 0.935) | 0.853 (0.850 to 0.858) |
| V4 | 0.755 (0.749 to 0.761) | 0.728 (0.720 to 0.740) | 0.870 (0.867 to 0.873) | 0.871 (0.867 to 0.877) | 1.000 (1.000 to 1.000) | 0.934 (0.932 to 0.935) | 0.854 (0.851 to 0.859) |

C-dense_start, cosine with the exact gradient, variant at the two GRU maps only (V0 at the other maps; V5 here is the upper limit for a scale at the GRU maps):

| variant | input map | recurrent map | lm | l2 | l3 | all parameters | all but l3 |
|---|---|---|---|---|---|---|---|
| V1 | 0.752 (0.745 to 0.759) | 0.726 (0.718 to 0.737) | 0.869 (0.866 to 0.872) | 0.870 (0.867 to 0.876) | 1.000 (1.000 to 1.000) | 0.933 (0.932 to 0.935) | 0.853 (0.849 to 0.858) |
| V2 | 0.754 (0.746 to 0.760) | 0.728 (0.720 to 0.739) | 0.869 (0.866 to 0.872) | 0.870 (0.867 to 0.876) | 1.000 (1.000 to 1.000) | 0.933 (0.932 to 0.935) | 0.853 (0.849 to 0.858) |
| V3 | 0.752 (0.745 to 0.759) | 0.726 (0.718 to 0.737) | 0.869 (0.866 to 0.872) | 0.870 (0.867 to 0.876) | 1.000 (1.000 to 1.000) | 0.933 (0.932 to 0.935) | 0.852 (0.849 to 0.858) |
| V4 | 0.754 (0.747 to 0.760) | 0.728 (0.720 to 0.738) | 0.869 (0.866 to 0.872) | 0.870 (0.867 to 0.876) | 1.000 (1.000 to 1.000) | 0.933 (0.932 to 0.935) | 0.853 (0.849 to 0.858) |
| V5 | 0.761 (0.754 to 0.767) | 0.734 (0.727 to 0.745) | 0.869 (0.866 to 0.872) | 0.870 (0.867 to 0.876) | 1.000 (1.000 to 1.000) | 0.934 (0.932 to 0.935) | 0.854 (0.850 to 0.859) |

C-dense_start, norm of the sweep's gradient over the norm of the exact gradient:

| variant | where | input map | recurrent map | lm | l2 | l3 | all parameters |
|---|---|---|---|---|---|---|---|
| V0 | every map | 0.68 (0.67 to 0.71) | 0.68 (0.67 to 0.70) | 0.85 (0.84 to 0.85) | 0.86 (0.85 to 0.87) | 1.00 (1.00 to 1.00) | 0.92 (0.92 to 0.93) |
| V1 | every map | 0.68 (0.67 to 0.71) | 0.68 (0.67 to 0.70) | 0.85 (0.84 to 0.85) | 0.86 (0.85 to 0.87) | 1.00 (1.00 to 1.00) | 0.92 (0.92 to 0.93) |
| V2 | every map | 0.68 (0.67 to 0.70) | 0.67 (0.66 to 0.68) | 0.84 (0.83 to 0.84) | 0.85 (0.84 to 0.86) | 1.00 (1.00 to 1.00) | 0.92 (0.92 to 0.93) |
| V3 | every map | 0.68 (0.67 to 0.70) | 0.68 (0.67 to 0.69) | 0.85 (0.84 to 0.85) | 0.86 (0.85 to 0.87) | 1.00 (1.00 to 1.00) | 0.92 (0.92 to 0.93) |
| V4 | every map | 0.67 (0.67 to 0.70) | 0.67 (0.66 to 0.68) | 0.84 (0.83 to 0.84) | 0.85 (0.84 to 0.86) | 1.00 (1.00 to 1.00) | 0.92 (0.92 to 0.92) |
| V2 | GRU maps | 0.68 (0.67 to 0.70) | 0.68 (0.67 to 0.69) | 0.85 (0.84 to 0.85) | 0.86 (0.85 to 0.87) | 1.00 (1.00 to 1.00) | 0.92 (0.92 to 0.93) |
| V3 | GRU maps | 0.68 (0.67 to 0.71) | 0.68 (0.67 to 0.70) | 0.85 (0.84 to 0.85) | 0.86 (0.85 to 0.87) | 1.00 (1.00 to 1.00) | 0.92 (0.92 to 0.93) |
| V4 | GRU maps | 0.68 (0.67 to 0.71) | 0.68 (0.67 to 0.69) | 0.85 (0.84 to 0.85) | 0.86 (0.85 to 0.87) | 1.00 (1.00 to 1.00) | 0.92 (0.92 to 0.93) |
| V5 | GRU maps | 0.69 (0.68 to 0.71) | 0.68 (0.67 to 0.70) | 0.85 (0.84 to 0.85) | 0.86 (0.85 to 0.87) | 1.00 (1.00 to 1.00) | 0.92 (0.92 to 0.93) |

C-dense_start, cosine for each gate's block of the two GRU maps:

| variant | where | input map, r | input map, z | input map, n | recurrent map, r | recurrent map, z | recurrent map, n |
|---|---|---|---|---|---|---|---|
| V0 | every map | 0.767 (0.758 to 0.778) | 0.769 (0.759 to 0.777) | 0.752 (0.744 to 0.758) | 0.726 (0.711 to 0.738) | 0.747 (0.733 to 0.769) | 0.725 (0.716 to 0.737) |
| V1 | every map | 0.768 (0.758 to 0.778) | 0.769 (0.759 to 0.777) | 0.752 (0.744 to 0.759) | 0.727 (0.711 to 0.738) | 0.747 (0.733 to 0.769) | 0.725 (0.716 to 0.737) |
| V2 | every map | 0.773 (0.764 to 0.782) | 0.774 (0.764 to 0.782) | 0.757 (0.750 to 0.763) | 0.735 (0.720 to 0.746) | 0.750 (0.737 to 0.774) | 0.731 (0.721 to 0.743) |
| V3 | every map | 0.769 (0.760 to 0.779) | 0.763 (0.754 to 0.772) | 0.749 (0.742 to 0.756) | 0.726 (0.709 to 0.737) | 0.742 (0.727 to 0.763) | 0.722 (0.713 to 0.734) |
| V4 | every map | 0.775 (0.766 to 0.783) | 0.769 (0.759 to 0.777) | 0.755 (0.748 to 0.761) | 0.735 (0.718 to 0.745) | 0.746 (0.732 to 0.768) | 0.727 (0.718 to 0.739) |
| V2 | GRU maps | 0.770 (0.761 to 0.780) | 0.770 (0.760 to 0.778) | 0.753 (0.746 to 0.760) | 0.730 (0.715 to 0.742) | 0.747 (0.733 to 0.771) | 0.727 (0.718 to 0.738) |
| V3 | GRU maps | 0.768 (0.758 to 0.778) | 0.767 (0.758 to 0.776) | 0.752 (0.744 to 0.758) | 0.727 (0.711 to 0.739) | 0.745 (0.731 to 0.768) | 0.725 (0.716 to 0.737) |
| V4 | GRU maps | 0.771 (0.762 to 0.781) | 0.768 (0.759 to 0.777) | 0.753 (0.746 to 0.760) | 0.731 (0.715 to 0.743) | 0.746 (0.732 to 0.770) | 0.727 (0.718 to 0.738) |
| V5 | GRU maps | 0.779 (0.769 to 0.789) | 0.772 (0.762 to 0.781) | 0.761 (0.754 to 0.766) | 0.742 (0.727 to 0.754) | 0.750 (0.736 to 0.773) | 0.733 (0.725 to 0.744) |

#### C-dense_S

E50 C-dense, trained under the sweep. Loss on the 16 batches: mean 0.0267 (least 0.0142, greatest 0.0339); first batch 0.0272. Norm of the exact gradient: mean 0.401 (0.182 to 0.568).

C-dense_S, cosine with the exact gradient, variant at every map:

| variant | input map | recurrent map | lm | l2 | l3 | all parameters | all but l3 |
|---|---|---|---|---|---|---|---|
| V0, first batch | 0.721 | 0.698 | 0.875 | 0.875 | 1.000 | 0.679 | 0.745 |
| V0 | 0.735 (0.707 to 0.821) | 0.713 (0.639 to 0.776) | 0.880 (0.865 to 0.894) | 0.879 (0.862 to 0.898) | 1.000 (1.000 to 1.000) | 0.640 (0.595 to 0.674) | 0.736 (0.700 to 0.788) |
| V1 | 0.744 (0.717 to 0.830) | 0.724 (0.649 to 0.789) | 0.880 (0.865 to 0.894) | 0.879 (0.862 to 0.898) | 1.000 (1.000 to 1.000) | 0.646 (0.604 to 0.679) | 0.745 (0.710 to 0.797) |
| V2 | 0.762 (0.732 to 0.842) | 0.744 (0.666 to 0.806) | 0.881 (0.869 to 0.895) | 0.880 (0.863 to 0.899) | 1.000 (1.000 to 1.000) | 0.775 (0.743 to 0.806) | 0.758 (0.716 to 0.807) |
| V3 | 0.784 (0.745 to 0.853) | 0.755 (0.723 to 0.810) | 0.780 (0.747 to 0.805) | 0.779 (0.740 to 0.817) | 1.000 (1.000 to 1.000) | 0.663 (0.627 to 0.698) | 0.737 (0.709 to 0.799) |
| V4 | 0.817 (0.788 to 0.877) | 0.794 (0.764 to 0.846) | 0.780 (0.746 to 0.807) | 0.778 (0.740 to 0.817) | 1.000 (1.000 to 1.000) | 0.793 (0.770 to 0.841) | 0.768 (0.736 to 0.824) |

C-dense_S, cosine with the exact gradient, variant at the two GRU maps only (V0 at the other maps; V5 here is the upper limit for a scale at the GRU maps):

| variant | input map | recurrent map | lm | l2 | l3 | all parameters | all but l3 |
|---|---|---|---|---|---|---|---|
| V1 | 0.744 (0.717 to 0.830) | 0.724 (0.649 to 0.789) | 0.880 (0.865 to 0.894) | 0.879 (0.862 to 0.898) | 1.000 (1.000 to 1.000) | 0.646 (0.604 to 0.679) | 0.745 (0.710 to 0.797) |
| V2 | 0.755 (0.726 to 0.839) | 0.737 (0.670 to 0.803) | 0.880 (0.865 to 0.894) | 0.879 (0.862 to 0.898) | 1.000 (1.000 to 1.000) | 0.652 (0.612 to 0.684) | 0.754 (0.724 to 0.806) |
| V3 | 0.762 (0.732 to 0.846) | 0.740 (0.672 to 0.808) | 0.880 (0.865 to 0.894) | 0.879 (0.862 to 0.898) | 1.000 (1.000 to 1.000) | 0.650 (0.609 to 0.682) | 0.755 (0.722 to 0.808) |
| V4 | 0.791 (0.755 to 0.869) | 0.772 (0.721 to 0.841) | 0.880 (0.865 to 0.894) | 0.879 (0.862 to 0.898) | 1.000 (1.000 to 1.000) | 0.668 (0.632 to 0.699) | 0.782 (0.755 to 0.834) |
| V5 | 0.852 (0.809 to 0.902) | 0.848 (0.806 to 0.889) | 0.880 (0.865 to 0.894) | 0.879 (0.862 to 0.898) | 1.000 (1.000 to 1.000) | 0.710 (0.673 to 0.740) | 0.838 (0.810 to 0.881) |

C-dense_S, norm of the sweep's gradient over the norm of the exact gradient:

| variant | where | input map | recurrent map | lm | l2 | l3 | all parameters |
|---|---|---|---|---|---|---|---|
| V0 | every map | 0.16 (0.13 to 0.20) | 0.16 (0.13 to 0.20) | 0.33 (0.30 to 0.37) | 0.33 (0.28 to 0.37) | 1.00 (1.00 to 1.00) | 0.41 (0.35 to 0.49) |
| V1 | every map | 0.16 (0.14 to 0.21) | 0.17 (0.13 to 0.21) | 0.33 (0.30 to 0.37) | 0.33 (0.28 to 0.37) | 1.00 (1.00 to 1.00) | 0.42 (0.35 to 0.49) |
| V2 | every map | 0.38 (0.31 to 0.46) | 0.38 (0.29 to 0.44) | 0.76 (0.68 to 0.83) | 0.75 (0.65 to 0.83) | 1.00 (1.00 to 1.00) | 0.57 (0.49 to 0.65) |
| V3 | every map | 0.18 (0.15 to 0.22) | 0.18 (0.15 to 0.22) | 0.38 (0.35 to 0.43) | 0.38 (0.34 to 0.42) | 1.00 (1.00 to 1.00) | 0.43 (0.37 to 0.50) |
| V4 | every map | 0.44 (0.36 to 0.53) | 0.44 (0.37 to 0.53) | 0.87 (0.79 to 0.94) | 0.86 (0.77 to 0.93) | 1.00 (1.00 to 1.00) | 0.62 (0.57 to 0.70) |
| V2 | GRU maps | 0.17 (0.14 to 0.21) | 0.17 (0.13 to 0.21) | 0.33 (0.30 to 0.37) | 0.33 (0.28 to 0.37) | 1.00 (1.00 to 1.00) | 0.42 (0.35 to 0.50) |
| V3 | GRU maps | 0.16 (0.13 to 0.21) | 0.16 (0.13 to 0.20) | 0.33 (0.30 to 0.37) | 0.33 (0.28 to 0.37) | 1.00 (1.00 to 1.00) | 0.41 (0.35 to 0.49) |
| V4 | GRU maps | 0.17 (0.14 to 0.22) | 0.18 (0.14 to 0.22) | 0.33 (0.30 to 0.37) | 0.33 (0.28 to 0.37) | 1.00 (1.00 to 1.00) | 0.42 (0.35 to 0.50) |
| V5 | GRU maps | 0.20 (0.17 to 0.25) | 0.21 (0.17 to 0.26) | 0.33 (0.30 to 0.37) | 0.33 (0.28 to 0.37) | 1.00 (1.00 to 1.00) | 0.43 (0.36 to 0.50) |

C-dense_S, cosine for each gate's block of the two GRU maps:

| variant | where | input map, r | input map, z | input map, n | recurrent map, r | recurrent map, z | recurrent map, n |
|---|---|---|---|---|---|---|---|
| V0 | every map | 0.595 (0.239 to 0.928) | 0.734 (0.670 to 0.791) | 0.735 (0.702 to 0.830) | 0.601 (0.157 to 0.844) | 0.725 (0.663 to 0.768) | 0.712 (0.634 to 0.785) |
| V1 | every map | 0.610 (0.257 to 0.932) | 0.746 (0.686 to 0.798) | 0.744 (0.711 to 0.838) | 0.618 (0.160 to 0.851) | 0.738 (0.680 to 0.777) | 0.723 (0.645 to 0.797) |
| V2 | every map | 0.634 (0.276 to 0.933) | 0.766 (0.710 to 0.811) | 0.762 (0.731 to 0.850) | 0.645 (0.199 to 0.845) | 0.760 (0.707 to 0.797) | 0.742 (0.661 to 0.814) |
| V3 | every map | 0.634 (0.235 to 0.913) | 0.764 (0.738 to 0.802) | 0.786 (0.744 to 0.860) | 0.662 (0.171 to 0.906) | 0.755 (0.697 to 0.801) | 0.756 (0.715 to 0.817) |
| V4 | every map | 0.669 (0.117 to 0.929) | 0.799 (0.770 to 0.830) | 0.819 (0.788 to 0.882) | 0.703 (0.260 to 0.904) | 0.795 (0.750 to 0.835) | 0.794 (0.761 to 0.851) |
| V2 | GRU maps | 0.629 (0.272 to 0.933) | 0.760 (0.701 to 0.805) | 0.755 (0.722 to 0.847) | 0.637 (0.152 to 0.847) | 0.754 (0.699 to 0.787) | 0.735 (0.665 to 0.811) |
| V3 | GRU maps | 0.632 (0.234 to 0.932) | 0.754 (0.696 to 0.806) | 0.763 (0.725 to 0.854) | 0.660 (0.126 to 0.884) | 0.747 (0.684 to 0.786) | 0.740 (0.668 to 0.815) |
| V4 | GRU maps | 0.674 (0.228 to 0.937) | 0.787 (0.736 to 0.828) | 0.791 (0.755 to 0.875) | 0.704 (0.120 to 0.918) | 0.785 (0.730 to 0.814) | 0.771 (0.712 to 0.846) |
| V5 | GRU maps | 0.814 (0.470 to 0.974) | 0.870 (0.837 to 0.915) | 0.851 (0.807 to 0.904) | 0.828 (0.388 to 0.969) | 0.874 (0.828 to 0.929) | 0.845 (0.798 to 0.887) |

#### C-dense_B

E50 C-dense, trained under back-propagation. Loss on the 16 batches: mean 0.0197 (least 0.0119, greatest 0.0262); first batch 0.0244. Norm of the exact gradient: mean 0.439 (0.227 to 0.731).

C-dense_B, cosine with the exact gradient, variant at every map:

| variant | input map | recurrent map | lm | l2 | l3 | all parameters | all but l3 |
|---|---|---|---|---|---|---|---|
| V0, first batch | 0.650 | 0.639 | 0.711 | 0.707 | 1.000 | 0.589 | 0.625 |
| V0 | 0.637 (0.495 to 0.724) | 0.629 (0.548 to 0.687) | 0.762 (0.719 to 0.795) | 0.758 (0.705 to 0.787) | 1.000 (1.000 to 1.000) | 0.528 (0.405 to 0.605) | 0.613 (0.490 to 0.676) |
| V1 | 0.647 (0.510 to 0.734) | 0.640 (0.565 to 0.697) | 0.762 (0.719 to 0.795) | 0.758 (0.705 to 0.787) | 1.000 (1.000 to 1.000) | 0.536 (0.414 to 0.615) | 0.622 (0.504 to 0.687) |
| V2 | 0.674 (0.547 to 0.766) | 0.666 (0.602 to 0.728) | 0.762 (0.719 to 0.791) | 0.757 (0.703 to 0.786) | 1.000 (1.000 to 1.000) | 0.631 (0.506 to 0.684) | 0.644 (0.533 to 0.716) |
| V3 | 0.719 (0.643 to 0.829) | 0.708 (0.644 to 0.780) | 0.665 (0.602 to 0.773) | 0.659 (0.593 to 0.768) | 1.000 (1.000 to 1.000) | 0.585 (0.516 to 0.658) | 0.662 (0.600 to 0.761) |
| V4 | 0.756 (0.692 to 0.862) | 0.751 (0.683 to 0.823) | 0.663 (0.602 to 0.770) | 0.657 (0.590 to 0.764) | 1.000 (1.000 to 1.000) | 0.699 (0.644 to 0.777) | 0.702 (0.650 to 0.799) |

C-dense_B, cosine with the exact gradient, variant at the two GRU maps only (V0 at the other maps; V5 here is the upper limit for a scale at the GRU maps):

| variant | input map | recurrent map | lm | l2 | l3 | all parameters | all but l3 |
|---|---|---|---|---|---|---|---|
| V1 | 0.647 (0.510 to 0.734) | 0.640 (0.565 to 0.697) | 0.762 (0.719 to 0.795) | 0.758 (0.705 to 0.787) | 1.000 (1.000 to 1.000) | 0.536 (0.414 to 0.615) | 0.622 (0.504 to 0.687) |
| V2 | 0.668 (0.545 to 0.754) | 0.662 (0.602 to 0.716) | 0.762 (0.719 to 0.795) | 0.758 (0.705 to 0.787) | 1.000 (1.000 to 1.000) | 0.547 (0.432 to 0.627) | 0.641 (0.534 to 0.704) |
| V3 | 0.655 (0.520 to 0.742) | 0.649 (0.577 to 0.708) | 0.762 (0.719 to 0.795) | 0.758 (0.705 to 0.787) | 1.000 (1.000 to 1.000) | 0.537 (0.417 to 0.611) | 0.627 (0.511 to 0.692) |
| V4 | 0.694 (0.584 to 0.781) | 0.691 (0.633 to 0.754) | 0.762 (0.719 to 0.795) | 0.758 (0.705 to 0.787) | 1.000 (1.000 to 1.000) | 0.562 (0.456 to 0.638) | 0.664 (0.569 to 0.728) |
| V5 | 0.779 (0.645 to 0.854) | 0.796 (0.712 to 0.857) | 0.762 (0.719 to 0.795) | 0.758 (0.705 to 0.787) | 1.000 (1.000 to 1.000) | 0.626 (0.544 to 0.696) | 0.747 (0.649 to 0.805) |

C-dense_B, norm of the sweep's gradient over the norm of the exact gradient:

| variant | where | input map | recurrent map | lm | l2 | l3 | all parameters |
|---|---|---|---|---|---|---|---|
| V0 | every map | 0.13 (0.09 to 0.22) | 0.13 (0.09 to 0.22) | 0.33 (0.27 to 0.39) | 0.34 (0.28 to 0.40) | 1.00 (1.00 to 1.00) | 0.31 (0.22 to 0.41) |
| V1 | every map | 0.13 (0.09 to 0.23) | 0.13 (0.09 to 0.22) | 0.33 (0.27 to 0.39) | 0.34 (0.28 to 0.40) | 1.00 (1.00 to 1.00) | 0.31 (0.22 to 0.41) |
| V2 | every map | 0.24 (0.16 to 0.37) | 0.23 (0.17 to 0.37) | 0.59 (0.48 to 0.67) | 0.59 (0.49 to 0.69) | 1.00 (1.00 to 1.00) | 0.38 (0.28 to 0.48) |
| V3 | every map | 0.16 (0.12 to 0.23) | 0.15 (0.11 to 0.21) | 0.39 (0.32 to 0.43) | 0.39 (0.33 to 0.44) | 1.00 (1.00 to 1.00) | 0.32 (0.23 to 0.42) |
| V4 | every map | 0.30 (0.23 to 0.39) | 0.29 (0.22 to 0.36) | 0.68 (0.57 to 0.77) | 0.69 (0.59 to 0.78) | 1.00 (1.00 to 1.00) | 0.43 (0.32 to 0.53) |
| V2 | GRU maps | 0.14 (0.10 to 0.24) | 0.13 (0.09 to 0.23) | 0.33 (0.27 to 0.39) | 0.34 (0.28 to 0.40) | 1.00 (1.00 to 1.00) | 0.31 (0.22 to 0.41) |
| V3 | GRU maps | 0.13 (0.09 to 0.22) | 0.13 (0.09 to 0.22) | 0.33 (0.27 to 0.39) | 0.34 (0.28 to 0.40) | 1.00 (1.00 to 1.00) | 0.31 (0.22 to 0.41) |
| V4 | GRU maps | 0.14 (0.10 to 0.24) | 0.14 (0.10 to 0.24) | 0.33 (0.27 to 0.39) | 0.34 (0.28 to 0.40) | 1.00 (1.00 to 1.00) | 0.31 (0.22 to 0.41) |
| V5 | GRU maps | 0.17 (0.12 to 0.28) | 0.17 (0.12 to 0.28) | 0.33 (0.27 to 0.39) | 0.34 (0.28 to 0.40) | 1.00 (1.00 to 1.00) | 0.32 (0.24 to 0.42) |

C-dense_B, cosine for each gate's block of the two GRU maps:

| variant | where | input map, r | input map, z | input map, n | recurrent map, r | recurrent map, z | recurrent map, n |
|---|---|---|---|---|---|---|---|
| V0 | every map | 0.399 (-0.338 to 0.625) | 0.653 (0.509 to 0.734) | 0.636 (0.497 to 0.739) | 0.369 (-0.070 to 0.640) | 0.649 (0.535 to 0.717) | 0.627 (0.551 to 0.695) |
| V1 | every map | 0.429 (-0.308 to 0.646) | 0.664 (0.525 to 0.740) | 0.646 (0.512 to 0.745) | 0.404 (-0.037 to 0.659) | 0.661 (0.554 to 0.726) | 0.638 (0.561 to 0.705) |
| V2 | every map | 0.488 (-0.252 to 0.703) | 0.685 (0.547 to 0.762) | 0.673 (0.550 to 0.772) | 0.472 (0.011 to 0.727) | 0.685 (0.583 to 0.751) | 0.665 (0.601 to 0.734) |
| V3 | every map | 0.589 (0.370 to 0.801) | 0.729 (0.639 to 0.793) | 0.718 (0.633 to 0.837) | 0.573 (0.425 to 0.698) | 0.720 (0.646 to 0.781) | 0.707 (0.631 to 0.785) |
| V4 | every map | 0.699 (0.470 to 0.917) | 0.769 (0.701 to 0.823) | 0.754 (0.683 to 0.868) | 0.697 (0.516 to 0.842) | 0.763 (0.721 to 0.804) | 0.751 (0.678 to 0.825) |
| V2 | GRU maps | 0.491 (-0.228 to 0.694) | 0.682 (0.552 to 0.758) | 0.668 (0.548 to 0.760) | 0.473 (0.035 to 0.717) | 0.681 (0.585 to 0.743) | 0.661 (0.594 to 0.722) |
| V3 | GRU maps | 0.471 (-0.296 to 0.653) | 0.668 (0.528 to 0.743) | 0.654 (0.522 to 0.751) | 0.460 (-0.017 to 0.731) | 0.666 (0.555 to 0.728) | 0.648 (0.571 to 0.717) |
| V4 | GRU maps | 0.597 (-0.083 to 0.792) | 0.703 (0.580 to 0.772) | 0.693 (0.587 to 0.786) | 0.601 (0.182 to 0.830) | 0.706 (0.616 to 0.758) | 0.690 (0.624 to 0.760) |
| V5 | GRU maps | 0.781 (0.531 to 0.912) | 0.813 (0.733 to 0.852) | 0.776 (0.627 to 0.855) | 0.810 (0.574 to 0.939) | 0.821 (0.756 to 0.871) | 0.793 (0.705 to 0.858) |

## 7. What this does and does not show

**What it shows.** On the saved weights of seed 73129 and on 17 batches of the training stream, the tables give how far the direction of the sweep's gradient is from the direction of the exact gradient for each parameter tensor, and how that distance changes when the one scale of the present rule is replaced by a scale for each gate, each row, each column, or by the product of the row's and the column's mean.

**A cosine nearer to 1 is not a demonstration that a network would learn faster.** Four reasons can be stated from the material itself.

1. No network was trained under any variant. The cosines are properties of single gradients on weights that were trained under V0 or under back-propagation. Under another scale the weights, and with them the magnitudes that the scale is computed from, would develop differently. The pairs of networks show that the rule under which the weights were trained changes the cosine: the networks trained under the sweep have higher cosines under V0 than those trained under back-propagation (section 6.2).
2. The optimizer of these runs is AdamW (`exp26_candidate.py`), which divides the running mean of each coordinate's gradient by the square root of the running mean of its square. The step is therefore not proportional to the gradient: a factor common to the gradients of a coordinate cancels, so that the norm ratios of section 6.4 matter little for the step, and the cosine between two gradients is not the cosine between the two steps.
3. Within these files the cosine does not separate the network that learned from the one that did not. `in16_S`, which did not learn (A 2.17 after 24,576 updates), has cosines of 0.603 and 0.651 on its two GRU maps under V0; `F10_S`, which learned (A 92.17), has 0.647 and 0.685.
4. The measurement covers one seed, two points of training (the start and the end), 17 batches, and batches of taught items only, on which a trained network has little loss left.

**The cases in which a variant changes nothing.** The first seven hold by the construction of the network and of the rule; the last two are measured and are cases of almost no change.

1. The gradient of `l3` (weights and bias) is identical under both rules and under every variant: its cosine is 1 and its norm ratio 1 in every table.
2. The gradients of `l2` and `lm` depend on the matrix `F` of `l3` only. Every variant applied at the two GRU maps only leaves them exactly at V0, and so does V1, which differs from V0 at the two GRU maps only.
3. V1 at every map and V1 at the two GRU maps are the same thing.
4. Multiplying the matrix `F` of `l3` by one positive number multiplies the gradient of every other tensor by that number and so changes no cosine of a single tensor; only the cosine of all parameters together changes, because the gradient of `l3` itself keeps its length. The value of the one scale at `l3` under V0 is therefore without effect on the direction of any single tensor's gradient.
5. In E50's `C-dense` the matrices `F` of the input map and of `l2` are never used, and there is no gradient for the byte table and `l1`. A variant at the input map changes nothing there: "at the two GRU maps" means at the recurrent map alone.
6. A variant at the input map changes the gradients of `l1` and of the byte table only; a variant at `l1` changes the gradient of the byte table only.
7. Outside a mask the sign is zero under every variant, so no variant sends error through an entry that is not kept.
8. Where the magnitudes that a variant distinguishes are equal, the variant is V0. This is nearly the case for the dense maps at the start (section 6.3), where no variant changes a cosine of the GRU maps, `lm` or `l2` by more than 0.006.
9. V2 changes the two GRU maps of the dense network by 0.002 or less, and V1 changes them by 0.006 or less, although the magnitudes there are not equal.

**What could not be established.**

1. The dense network `G` at its start. No saved file holds its recurrent matrix before training: the start files of E49 and E50 hold the starting network with the recurrent map already masked. E42's receipts record a start cosine of 0.7409766 for the recurrent map of `G` (printed with Test 2), which could not be reproduced for this reason.
2. Whether any variant changes the speed or the outcome of learning. That needs runs.
3. The effect of a variant at a single one of the maps `l1`, `l2`, `lm` or `l3` taken alone, except where it follows from the dependences listed in section 4 (for `lm` and `l2`, the effect of a variant at every map is the effect of the variant at `l3`).
4. Other seeds, items that were not taught, and points of training between the start and the end.
5. The form of the single-precision accumulation in the training library, which is inferred from the agreement in Test 2 and was not read from the library.
6. The sweep's gradients of `l1`, `l2` and the byte table have no recorded figure to be compared with; they rest on Tests 1 and 3.

## 8. Commands

From the repository root, with the three branches named in section 3 fetched. Python 3.13.17 with numpy only; the training library is not needed.

```
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/sweep_scale_variants.py                     # every part, every network: 263 s
python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/sweep_scale_variants.py stream forward      # section 5.1
python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/sweep_scale_variants.py test1               # section 5.2 (49 s)
python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/sweep_scale_variants.py test2 test3         # sections 5.3 and 5.4
python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/sweep_scale_variants.py summary magnitudes tables   # section 6
python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/sweep_scale_variants.py tables G_S F10_B    # a part for chosen networks
```

A part that needs the gradients measures each network once, at about 20 seconds a network on one core, and keeps the result in memory for the parts that follow in the same call. The script writes no file. The run reported here was made with the script whose SHA-256 is `f6dead2d3d1588b2783b2a1a421706106ddd241674a81e38c3225259496c594c`; it started at 22:55:22 UTC and ended at 22:59:45 UTC on 17 November 2026. The script was not committed by its author. Eight commits that another process made on this branch while the work was in progress, from `41ac6004` (22:16 UTC) to `c0919f75` (22:56 UTC), contain states of it; the first seven hold unfinished states, and `c0919f75` holds the finished script with the SHA-256 given above.
