# E51: a new task learned in cells held in reserve; and how much rehearsal keeps the first family (17 November 2026, revision 2)

Evidence level E2. **Status.** Written by the outside reviewer, acting for the coordinator, who cannot be reached while Mr. Xu is away (his instruction of 17 November: "我人在外面你自己安排", arrange it yourself). Revision 1 was checked by an agent of the reviewer that had not seen the reviewer's reasoning (`CHECK_OF_TASK_E51_BRIEF_20261117.md` beside this file, with a numpy script behind every figure); revision 2 enters its seven findings and adds Part B. That check is not the independent audit the programme's rules ask for, and the coordinator has not seen the brief. It is item 4 of `PLAN.md` beside this file. A trial on one seed: Part A 16 runs, about 4 CPU-hours; Part B 16 runs, about 8.

Shell names: `D=origin/codex/minifly-consolidated-20261023` (the default branch; it is your checkout); `V=origin/claude/vibrant-cerf-sje726` (the reviewer's branch; this brief and `$R` are read from it with `git show`); `M=MOD97_LOCAL_RULE_20261106`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`. Host: Linux x86-64, Python 3.13, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit.

## Terms

- *The network* is `arch20`'s recurrent network: a byte table (258 by 8), a layer `l1` (8 to H) with tanh, a GRU of U cells written out step by step in `arch26.forward`, and a readout `z2 = tanh(l2 z1 + lm h)` followed by `l3`. *Present size*: H = 512, U = 256. *Half width*: H = 256, U = 128.
- *Back-propagation* (rule B) is the exact gradient. *The sweep* (rule S) passes the error backward through each linear map by the signs of its weights times one scale for the map (`exp3_state.SignLinear`); it is `local=True` in `arch26.forward` and in `exp26_candidate.step`.
- *The first family* is the four operations add, sub, max and min under modulus 89, with the code bytes `p q v w`. Its *reduction pairs* are the pairs of add and sub whose answer needs the modulus (a + b ≥ 89; a − b < 0). *A birth* is a network trained on the first family alone.
- *A unit in reserve* is a cell, or a unit of the readout's layer z2, that the first family does not use: it is zero on every item of the first family.
- *The directions in use* at a matrix of connections are the directions spanned by the input vectors that 2,080 taught items of the first family present to that matrix, a constant 1 appended for the bias. A change of the matrix that is zero on those directions leaves the matrix's output on those items as it was.

## The question

**Part A.** E43 showed that a born network loses the first family within 64 updates of an unrelated task. The reviewer's measurement on E43's networks (memo, section 29; `$R/cells_in_use.py`, `$R/directions_in_use.py`) gives the reason: every cell is in use on every item, and the first family's own items occupy all the directions of the cells' state (255 or 256 of 257 hold 99.99% of the trace; all 257 are spanned). E48 showed that when the change for a new family is made in a part of the network that only that family uses, the earlier families are kept exactly and the new one is learned; there the part was supplied from outside. Part A gives the network such a part of its own. E41's half-width birth is placed inside a network of the present size, so that 128 cells and 256 units of the readout are in reserve, and a rule confines new learning to the reserve and to directions that the first family does not use. Two questions: **is the first family kept, and what does the new task cost to learn?**

**Part B.** Mr. Xu, 17 November: some rehearsal is acceptable, and long-term memory should form by itself in the course of learning. The record has rehearsal at one share only, one item of the first family for each new item, in every update. **How little rehearsal keeps the first family; does rehearsal at intervals make it cheaper to restore and longer-lived from one time to the next; and do smaller steps on the new task leave more of it?** Part B is run on E43's own networks, without any reserve.

## What the record says (working rule 1)

- **E43** (`exp43_unrelated.py`; six E26 births of the present size). Under the arm `unrelated` without decay the first family's reduction pairs are at 0 of 3,916 at update 4,096 in all six networks; at update 64 they are at 0.1% or less in five networks and at 4.9% in the sixth, while the unrelated task is not yet learned at all. At update 4,096 the unrelated task is at 300 of 300. The arm `conflict` (the second family, modulus 97, alone) is the record for a related task.
- **E43's second addendum** (`e43_addendum2.py`). With one item of the first family for each new item, the first family's reduction score at update 4,096 is 94.8 to 105.0% of its start in all twelve runs; at update 256 the unrelated task is behind the unrehearsed run in all twelve comparisons.
- **E43's first addendum.** A reminder of the first family after 4,096 unrelated updates restores little in 64 or 256 updates and, unevenly between seeds, 11 to 91% in 1,024.
- **E41.** The half-width births with half of the pairs taught reach 92.8 to 96.2 on held-out add and sub at 12 letters and 90.9 to 94.7 on reduction pairs (six files, on the default branch).
- **E48.** A term of rank 16 or 64, present only on the new family's items, keeps every earlier family item for item and reaches a higher level on the new family than E40's free learning by update 1,024; it is slower in the first 64 updates.
- **The reviewer's arithmetic for Part A** (`$R/e51_design_checks.py`, no training, six births), in the form with gates shut by a bias; the form of this brief was not run by the reviewer. **The check's arithmetic** (the script in the check's appendix), in the form of this brief: after a random change, as large as 4 in every entry that the rule allows to change, no output on a first-family item moves by more than 1e-9, on taught and held-out pairs with 0 to 40 letters.
- **What a new byte can use.** For a born `l1` the code of a byte (its row of the byte table passed through `l1`) lies in nine directions that every byte shares. In the embedded network half of `l1`'s units are fresh, and a byte that has never occurred has about 2% of its code, a 2-norm of about 1 in 8, outside the directions in use at the input map. The old cells' rows can therefore learn a response to a new byte, about ten times more slowly for each update than a reserve cell's rows. The job does not confine the new task to the reserve; it measures where the task is learned.

# Part A

## The learner, as code

For a seed and a rule (S or B), with `sd0` the state dict of `$M/exp45/e41/e41_<rule>_h256_f50_<seed>.pt` (its file hash and tensors hash asserted against its receipt) and `m` a network of the present size whose every weight is first that of `exp41_scale.new_net(seed, 512, 256)` (the fresh network of the seed; `sw_net.SWNet(512, 256)` has the same parameters in the same order if `exp41_scale` cannot be imported beside the other modules):

```python
with torch.no_grad():
    sd = m.state_dict()
    sd["book.weight"].copy_(sd0["book.weight"])
    sd["l1.weight"][:256] = sd0["l1.weight"];  sd["l1.bias"][:256] = sd0["l1.bias"]
    for g in range(3):                                   # torch's gate order: r, z, n
        new, old = slice(g * 256, g * 256 + 128), slice(g * 128, (g + 1) * 128)
        sd["rnn.weight_ih_l0"][new, :256] = sd0["rnn.weight_ih_l0"][old]
        sd["rnn.weight_ih_l0"][new, 256:] = 0.0          # added feed-forward units do not reach old cells
        sd["rnn.weight_hh_l0"][new, :128] = sd0["rnn.weight_hh_l0"][old]   # columns 128..255 keep the fresh draws
        sd["rnn.bias_ih_l0"][new] = sd0["rnn.bias_ih_l0"][old]
        sd["rnn.bias_hh_l0"][new] = sd0["rnn.bias_hh_l0"][old]
    sd["lm.weight"][:256, :128] = sd0["lm.weight"];  sd["lm.bias"][:256] = sd0["lm.bias"]
    sd["l2.weight"][:256, :256] = sd0["l2.weight"];  sd["l2.weight"][:256, 256:] = 0.0;  sd["l2.bias"][:256] = sd0["l2.bias"]
    sd["l3.weight"][:, :256] = sd0["l3.weight"];     sd["l3.weight"][:, 256:] = 0.0;     sd["l3.bias"].copy_(sd0["l3.bias"])
```

Cells 0 to 127 are the *old cells*, cells 128 to 255 the *reserve cells*. Units 0 to 255 of `l1` and of z2 are old, 256 to 511 added. Every weight not named above keeps the fresh network's draw: the reserve cells' own rows, the weights from reserve cells into old cells and into the readout, and the added units' rows. The weights from reserve cells into old cells keep the fresh draws, and through them the reserve can act on the old cells on the items of a new task.

**The two forms of the start.** In torch's GRU the new state is `(1 - z) n + z h`, so `z = 1` keeps the state, which starts at zero.

- *The start with a bias* (arms `open` and `rehearse`): `sd["rnn.bias_ih_l0"][384:512] = 10.0` and `sd["rnn.bias_hh_l0"][384:512] = 0.0` (the update-gate rows of the reserve cells), and nothing else. The reserve cells' gates start shut on every item and are free to change.
- *The keyed start* (arms `keyed` and `keyed-free`): the biases keep the fresh draws, and the forward pass holds the reserve at zero on every item whose first byte, `x[:, 1]`, is one of the bytes in use (below). `arch26.forward`'s text is copied into the new runner with two lines added: after `z` is computed, `z = torch.where(shut, torch.ones_like(z), z)` with `shut[b, i] = in_use[x[b, 1]] and i >= 128`; and after `z2` is computed, `z2 = torch.where(off, torch.zeros_like(z2), z2)` with `off[b, t, j] = in_use[x[b, 1]] and j >= 256`. The copy also returns the list of z. **Every forward pass of a keyed run applies the two lines: the update, under both rules, and the scorer's generation.** After `arch26.install()`, set `exp11_cell.forward` to the runner's copy, and call the copy in the runner's copy of `exp26_candidate.step`. `exp43_unrelated.eq_states` and `e43_addendum2.rehearse` call `arch26.forward` directly and are not used in a keyed run. No existing source file is edited.

The embedded network is saved before any update as `e51_start_<bias|keyed>_<rule>_<seed>.pt` and committed.

## The bytes in use and the directions in use (arm `keyed` only)

The 2,080 items from which they are taken are the first family's taught half at every number of letters:

```python
rng = Random(f"e51|{seed}"); items = {k: [] for k in range(13)}
for t in E.DEV:                                           # add, sub, max, min under 89
    taught = E.split(t, .5, seed)[0]
    for k in range(13):
        items[k] += [E.render(t, a, b, k, rng) for a, b in rng.sample(taught, 40)]
```

The SHA-256 of `json.dumps` of the list of `[prompt, answer]` pairs, k = 0 first, begins `d22b5733778491b1` for seed 73129 (`54bdb3c36ba2dc6e` for 93683, `f29bee3306c1a5d1` for 116089), by the reviewer's reader and by the check's. *The bytes in use* are the start symbol and every byte of those prompts and answers but the end symbol: 46 bytes (the start symbol, `p q v w`, the ten digits, `+ - > <`, `=`, the 26 capital letters).

The items are run through the embedded network under teacher forcing, as in an update (`x = [start] + prompt + answer[:-1]`), with the keyed forward, one number of letters at a time, so that no row is padded. For each matrix the input vectors are collected in float64, a constant 1 is appended, and with `C = X.T @ X / len(X)` the directions in use are the eigenvectors of `C` whose eigenvalue exceeds 1e-12 of the largest, that is, every direction the vectors span; `Vm` holds them as columns.

| Matrix (weight, bias) | Its input vectors | Collected at | Directions (the check's count) |
|---|---|---|---|
| input map (`rnn.weight_ih_l0`, `rnn.bias_ih_l0`) | `z1 = tanh(l1(book[x]))`, 512 values | every position | 46 |
| recurrent matrix (`rnn.weight_hh_l0`, `rnn.bias_hh_l0`) | the state before the step, 256 values (zero at position 0) | every position | 129 |
| `lm` | the state after the step | the three answer positions | 129 |
| `l2` | `z1` | the three answer positions | 11 |

They are computed once, before the first update, and saved with the start file. `l3` needs none: in the keyed forward the added units of z2 are zero on the first family's items, so those items reach `l3` only through its columns 0 to 255, which are not changed.

## The rule (arm `keyed`), as code

The runner's copy of `exp26_candidate.step` with two additions, one before and one after `opt.step()`. For a matrix with weight `W`, bias `b`, directions `Vm` and a set of rows:

```python
def away(D, Vm):                         # D: [rows, inputs + 1], float64
    return D - (D @ Vm) @ Vm.T

# before opt.step(): the gradient
G = torch.cat([W.grad, b.grad[:, None]], 1).double()
G[rows] = away(G[rows], Vm);  W.grad.copy_(G[:, :-1]);  b.grad.copy_(G[:, -1])
# after opt.step(): the step itself, which the optimizer scales entry by entry
S = torch.cat([W - W_before, (b - b_before)[:, None]], 1).double()
S[rows] = away(S[rows], Vm);  W.copy_(W_before + S[:, :-1]);  b.copy_(b_before + S[:, -1])
```

| Tensors | Rows projected | Rows free |
|---|---|---|
| input map and its bias; recurrent matrix and its bias | the old cells' rows: `g * 256 + i` for g = 0, 1, 2 and i < 128 | the reserve cells' rows, all three gates |
| `lm`, `l2` and their biases | rows 0 to 255 | rows 256 to 511 |
| `l3.weight[:, :256]` and `l3.bias` | not changed at all (restored after the step) | `l3.weight[:, 256:]` is free |
| `l1.weight`, `l1.bias` | not changed at all (restored after the step) | |
| `book.weight` | the 46 rows of the bytes in use are not changed (restored after the step) | the other rows |

With the keyed forward and this rule the first family's outputs are unchanged by construction: on its items the reserve cells and the added units of z2 are zero, every old unit receives the inputs it received at the start, and every matrix acts on those inputs as it did.

Optimizer: a fresh `exp26_candidate.make_opt(m)` with the weight decay of group 0 set to 0, in every arm, as in E43's arms without decay. One property of the construction to keep in mind: under the sweep the scale of a map is the mean size of all its weights, the zero blocks included, so the sweep's signal in the embedded network is not that of the half-width birth; this is the same in every arm.

## Arms

| Arm | Start | What may change | What it shows |
|---|---|---|---|
| `open` | with a bias | everything, as in E43 | the loss without any rule, and the speed of the new task without any limit |
| `rehearse` | with a bias | everything; each item of the stream is drawn with equal chance from the new task's stream or from the first family's taught half, as `e43_addendum2.py` does it (for `conflict` the same construction with the conflict stream in the place of the unrelated one, the choice by `Random(f"e51|{seed}|joint|conflict").randrange(2)`) | the plain baseline |
| `keyed-free` | keyed | everything; nothing projected, nothing restored | what holding the reserve shut on the first family's items does by itself |
| `keyed` | keyed | the rule | whether the first family is kept, whether the new task is learned, and how much of it is lost when the reserve is shut |

The cost of the rule is `keyed` against `keyed-free`; the cost of the whole device is `keyed` against `open`. In the keyed arms the condition is computed from the item (has its first byte occurred before?) and is not supplied for a family. It is still a device of the harness, and the report says so.

## Tasks, dose, rows

- `unrelated`: E43's task, stream and panels (`e43_task.py`; the arm `unrelated` of `exp43_unrelated.py`).
- `conflict`: the second family alone (modulus 97), E43's arm `conflict`: `X16.entry_stream(X16.family_entries("F1", seed), seed + 101)`, scored as E43 scores that arm (its score 3 is two panels, add97 and sub97, of 400 held-out pairs each).

4,096 updates, batches of 64. Scored at updates 0, 16, 32, 64, 128, 256, 512, 1,024, 2,048 and 4,096. Each network under the rule of its birth. Seed 73129: two rules, four arms, two tasks, sixteen runs.

## Checks before the runs of Part A (a receipt for each)

1. **The embedding.** On E43's panel of the first family held out at 12 letters, and on the 2,000 items `Random(f"e51|{seed}|check1").sample(items, 2000)` of the 31,684 items of `exp43_unrelated.first_family_items()` in its order of tasks, the embedded network's outputs at the answer positions, under teacher forcing with the right answer, equal the half-width birth's: to 1e-5 in the keyed start, to 0.02 in the start with a bias. The scores of row 0 equal the half-width birth's on every panel in the keyed start.
2. **The directions.** The digest of the 2,080 items equals the reviewer's. The numbers of directions are 46 at the input map, 129 at the recurrent matrix, 129 at `lm` and 11 at `l2`.
3. **The rule.** (a) After 8 updates of `keyed` on the unrelated stream, no output at an answer position on the panels of check 1 has changed by more than 1e-4, and every tensor or row that the table calls not changed is bit-equal to its start. The same on 8 updates of the conflict stream, whose items contain bytes in use; and with the restoring of the byte table switched off, at least one of its 46 rows differs after those 8 updates. (b) The same 8 updates of `keyed-free` change those outputs by more than 0.1. (c) On a batch of the first family every reserve cell's state and every added unit of z2 is exactly zero at every position in the keyed forward; on a batch of the new task they are not.
4. **The scorer.** In a keyed run the generated answers of the first family at row 0 equal those of the half-width birth item for item, which shows that the two added lines act in generation.
5. **The data.** The batch digests of the two streams at update 4,096 equal those in E43's receipts of the same seed (`unrelated`, `conflict`); for `rehearse` on `unrelated`, the digest of E43's second addendum.
6. **`open` is E43 in another network.** The runner of `open`, started from E26's birth of the seed as it is, with no embedding and no change of any bias, reproduces E43's `unrelated` receipt of that seed and rule at row 64 (the first family's counts and the tensors hash).
7. **Cost.** A dry run of 64 updates of every arm with CPU-seconds an update and a scored row, and a forecast for the sixteen runs.

## What is recorded at every scored row of Part A

- **The first family:** E43's scores 1 and 2 (all pairs without letters, add and sub split by reduction, max and min on all pairs; the held-out panel at 12 letters). The runner calls `exp11_cell.respond` itself on the items of scores 1 and 2 and keeps the generated bytes of row 0; at every row it records the share of answers equal to row 0's. What the wrong answers on reduction pairs are, in these classes, tried in this order on each wrong answer: (1) the end symbol alone; (2) an answer that contains a byte of 128 or above; (3) two digits and the end symbol that are the right answer under modulus 97; (4) two digits and the end symbol that differ from the right answer by exactly 1 or exactly 10; (5) any other two digits and the end symbol; (6) anything else. The generated bytes of the reduction pairs are written to a file at rows 0, 64 and 4,096.
- **The new task:** E43's score 3 for its stream (for `conflict`, the 800 items of the two panels together), and the same score a second time with the reserve cells held shut on every item (`shut[b, i] = i >= 128` for every b), in every arm.
- **The reserve:** the number of reserve cells whose update gate is open by more than 0.1 (the mean of `1 - z` over the positions and items of the new task's panel); the largest size of a reserve cell's state on the first family's panel.
- **The weights:** E43's score 4 (movement of each group from the start), the same separately for the old cells' rows and for the reserve cells' rows of the input map and of the recurrent matrix, and the norm of each tensor.

Committed: receipts, logs, the start files with their directions, and the weights at updates 64 and 4,096 of every run. Weight files are ignored by `.gitignore`; they are added with `git add -f`, each after its hash has been compared with its receipt. The report lists the committed files from `git ls-tree`.

## How Part A's result is stated (fixed beforehand)

For each rule, task and arm:

- *Kept*: the first family's reduction score at update 4,096 is within 2 points of row 0. Otherwise the report gives what is left and what the wrong answers are.
- *Learned*: the new task's score on its never-taught items reaches 80% at some scored row and stays there at every later row.
- *Learned in the reserve*: *learned*, and at update 4,096 the score with the reserve shut is below 20%. Otherwise the report gives both scores.
- *The cost of the rule*: the first row at which `keyed` is *learned* against the first such row of `keyed-free`. *The cost of the device*: the same against `open`.

No reading beyond these, and no statement about more than one seed.

## Predictions for Part A (the reviewer's, on file before any run)

| | Statement | Chance |
|---|---|---|
| P51.1 | `open` loses the first family on `unrelated`: below 5% on reduction pairs by update 64, both rules. Refuted if the reserve takes the new task and the family survives | 90% |
| P51.2 | `open` loses the first family's reduction pairs on `conflict` (below half of row 0 at update 4,096), both rules | 80% |
| P51.3 | `keyed` keeps the first family on both tasks, both rules. By construction: a failure is an error of implementation | 97% |
| P51.4 | `keyed` learns `unrelated` | 60% under back-propagation, 45% under the sweep |
| P51.5 | `keyed` learns `conflict` | 50% and 35% |
| P51.6 | Where `keyed` learns `unrelated`, the cost of the rule is at most four | 60% |
| P51.7 | `rehearse` keeps the first family within 5 points on both tasks, both rules | 85% |
| P51.8 | `keyed-free` does not keep the first family on `unrelated` under either rule | 85% |
| P51.9 | Where `keyed` learns `unrelated`, it is *learned in the reserve* | 75% |

If P51.4 fails under both rules and `keyed-free` learns the task, the rule is what prevents it; if `keyed-free` fails too, the embedding is. Which part of the rule binds is not shown. If P51.4 and P51.5 hold, the reserve learner is the candidate for the long lives.

# Part B: how much rehearsal, at what times, and with what steps

Part B uses E43's own setting and runner: the E26 birth of the seed at the present size (`exp43_unrelated.run`'s start), the unrelated stream, no decay, no embedding and no reserve. It starts when Part A's runs are finished and its report is sent; you do not wait for an answer. Seed 73129, both rules.

**A light probe**, used in all three sets beside E43's scores: 400 reduction pairs of the first family without letters (200 of add89 and 200 of sub89, `Random(f"e51|{seed}|probe").sample` from the reduction pairs of each task in `first_family_items()`'s order) and E43's held-out panel of the unrelated task (300 items). The probe is scored every 64 updates up to update 1,024 and every 256 after that, unless a set says otherwise.

**B1. The share of rehearsal.** E43's `unrelated` stream with each item drawn from the first family's taught half with chance 1/k (E38's rehearsal stream, `X16.entry_stream([("op", t, a, b) for t in DEV for a, b in split(t, .5, seed)[0]], seed + 100)`): an item is a rehearsal item when `Random(f"e51|{seed}|joint|{k}").randrange(k) == 0`, and the two streams are consumed in order as `e43_addendum2.rehearse` does. k = 4, 8, 16, 32. 4,096 updates; E43's scores at its rows (0, 64, 256, 1,024, 2,048, 4,096). Eight runs. The references are E43's `unrelated` (no rehearsal) and the second addendum's `unrelated-rehearse` (k = 2) of the same seed and rule.

**B2. Rehearsal at intervals.** One run a rule, one optimizer for the whole run. Eight cycles. A cycle is a *gap* of 256 updates of the unrelated stream alone, then a *review*: updates of the first family's taught stream alone (E38's rehearsal stream, continued from cycle to cycle), with the probe scored every 16 updates, until the probe's first-family share is at least 90% of its value at update 0, or 1,024 review updates have passed. Recorded for each cycle: the probe at the end of the gap; the number of review updates; the probe at the end of the review. E43's scores at update 0 and at the end of the run.

**B3. Smaller steps on the new task.** E43's `unrelated` arm at learning rates 0.001, 0.0003 and 0.0001 (every parameter group's `lr` set after `make_opt`; the receipt records the run's rate and the groups read back, not `exp26_candidate.opt_record`'s constants), for 4,096, 13,824 and 40,960 updates. The probe every 16 updates up to update 256 and every 64 after that; E43's scores at update 0 and at the end. Six runs.

## Checks before the runs of Part B (a receipt for each)

8. **B1's runner is the addendum's at k = 2.** With the addendum's own choice (`Random(f"e43|{seed}|joint").randrange(2)`), the runner reproduces the second addendum's `unrelated-rehearse` receipt of seed 73129 under each rule: the first family's counts at row 64 and the tensors hash at update 4,096.
9. **B3 at 0.001 is E43's run.** The run at 0.001 has E43's `unrelated` receipt's first-family counts at rows 64 and 4,096 and its tensors hash at update 4,096.
10. **The probe.** At update 0 the probe's first-family share equals the share of the same 400 items in E43's score 1 at row 0.
11. **Cost.** A forecast for the sixteen runs from Part A's measured CPU-seconds an update.

## How Part B's result is stated (fixed beforehand)

- **B1.** For each rule and share: the first family's reduction score at update 4,096 as a share of row 0, and the lowest value of the probe over the run. *Kept* at a share if the reduction score at update 4,096 is within 5 points of row 0. *The least share that keeps* is the smallest share that is kept with every larger share also kept. Beside it the first row at which the unrelated task's held-out score is at 80%.
- **B2.** *The review shortens* if the review of cycle 8 needs at most half the updates of the review of cycle 1. *More is left* if the probe at the end of the gap of cycle 8 is at least 20 points above that of cycle 1. The table gives all eight cycles. The sentence for each rule: "with rehearsal at intervals the first family [becomes / does not become] cheaper to restore, and [more / no more] of it survives a gap".
- **B3.** For each rule and rate: the probe's first-family share at the first probe row at which the unrelated task's held-out score is at 80%, and the unrelated task's score at the last probe row at which the first-family share is still at least half of its value at update 0. *Smaller steps leave more* if the first of these is at least 20 points higher at a smaller rate than at 0.001.

## Predictions for Part B (the reviewer's, on file before any run)

| | Statement | Chance |
|---|---|---|
| PB.1 | A share of a quarter (k = 4) keeps the first family, under both rules | 70% |
| PB.2 | A share of 1/32 does not keep it, under either rule | 65% |
| PB.3 | The review shortens, under back-propagation | 60% |
| PB.4 | More is left after a gap at cycle 8, under back-propagation | 30% |
| PB.5 | Smaller steps do not leave more, under either rule | 65% |

The explanation under test in B2, with what refutes it: in the present network a memory does not become more lasting by being rehearsed; each gap removes it as the first did. It is refuted for a rule if *more is left*.

# For both parts

## What the job cannot show

- That a learner can do without the device of the keyed arms. An arm without it (reserve gates that only the optimizer opens, through directions the first family does not use) is weak in the present code, in which a new byte has about 2% of its code outside the directions in use; it waits for a birth with E50's code, in which a new byte has values of its own.
- How a reserve is shared among many tasks: here the first new task may take all of it.
- A life; more than one seed; a comparison with a store at equal memory, which belongs to the long-life world.
- Whether a network can be born with a reserve by itself: here the reserve is made by taking a narrower birth.
- In Part B: more than two tasks; a share of rehearsal that changes with time; rehearsal chosen by the learner.

## Rules for the worker

- A new runner, `exp51_reserve.py`, in `$M`, built on `exp43_unrelated.py` and `e43_addendum2.py`, which are imported and not edited; outputs in `$M/exp45/e51/` (Part B in `$M/exp45/e51/partB/`). Every existing source file stays as it is.
- At most three processes, one thread each (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`), launched detached and outside any wrapper with a time limit. torch 2.15.0+cpu from `https://download.pytorch.org/whl/cpu` in a virtual environment; versions recorded in every receipt.
- The container can restart without warning and loses what is not pushed. Commit and push receipts and logs as runs finish; a run interrupted by a restart is started again from its beginning after its partial receipt has been moved aside.
- Push only to your own branch. Never push to the default branch or to any other branch, and open no pull request. Other branches are read-only: fetch with `git fetch origin '+refs/heads/<name>:refs/remotes/origin/<name>'` and read with `git show`.
- Commit trailers as `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` says. No name of any language model in any committed file or commit message.
- The report, `E51_REPORT.md`, in standard academic English: full sentences, every term defined at first use, every figure with the receipt or the command that gives it.
- If the brief cannot be followed at some point, stop that part, say what you found, and ask.

## Reporting

Send the reviewer a short message with the tool `send_message` of the claude-code-remote server, to session `session_01YBMaMvtGtQvDFcgppgThrZ`, at each of these points: Part A's checks passed (or which failed); Part A's sixteen runs finished, with the readings; Part B's checks passed; each of B1, B2 and B3 finished, with its table; the report committed. Each message gives your branch, the commit and the main figures in at most ten lines. If you need a decision, ask in the same way and go on with whatever does not depend on the answer. The two other seeds are released by the reviewer after the trial is read.

## Done

The eleven checks; the 32 runs; `E51_REPORT.md` with the table of every run at every row, the readings of both parts, the reserve's figures, the outcome of each prediction, and the list of committed files.

## What changed from revision 1

| | Change (the check's seven findings, then the reviewer's addition) |
|---|---|
| 1 | The keeping in `keyed` is now exact by construction. Revision 1 protected only the largest directions at `l3`, where the first family uses all 513, and at the input map 36 to 39 of 46; a correct implementation would have failed its own check 3(a) and the first family's outputs would have drifted. The added units of z2 are now held at zero on the first family's items, `l3`'s old columns are not changed, and every direction the vectors span is protected |
| 2 | The statement that old cells cannot learn a new byte is withdrawn (it holds for a born `l1`, not for the embedded network). The new task's score is recorded a second time with the reserve shut, and "learned in the reserve" is a reading |
| 3 | The arm `keyed-free` is added, so that the cost of the rule can be told from the cost of the whole device |
| 4 | The classes of wrong answers are defined here; revision 1 named classes that do not exist in the record |
| 5 | Check 3(a) is repeated on the conflict stream, where the restoring of the byte table can fail |
| 6 | Nine places where a worker could not follow the brief are filled in: check 6's start, how the added lines reach the scorer and the update, the gates returned, check 1's items, the score of `conflict`, `git add -f` for weight files, the generated bytes, unpadded batches for the directions, the checkout |
| 7 | Five statements about the record corrected: why the weights from the reserve into old cells keep their draws; which form the reviewer's arithmetic tested; the criterion behind "255 or 256 of 257"; the first family at update 64 (4.9% in one network); 94.8 to 105.0% |
| – | Part B is added at Mr. Xu's statement of 17 November: shares of rehearsal below a half, rehearsal at intervals, and smaller steps on the new task, on E43's networks |
