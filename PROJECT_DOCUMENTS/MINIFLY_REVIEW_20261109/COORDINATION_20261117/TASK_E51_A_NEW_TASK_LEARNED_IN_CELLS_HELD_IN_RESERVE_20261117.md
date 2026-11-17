# E51: a new task learned in cells held in reserve, with what the first family uses left unchanged (17 November 2026, revision 1)

Evidence level E2. **Status.** Written by the outside reviewer, acting for the coordinator. Mr. Xu, through whom the programme's owner works, is away; his instruction to the reviewer on 17 November, twice: "As I am outside, coordinate yourself" and "Unfortunately I am outside, so you have to coordinate yourself." The coordinator, which normally writes a brief for the reviewer to audit, cannot be reached until he returns, so this brief is the reviewer's own. It was checked before release by an agent of the reviewer that had not seen the reviewer's reasoning; that is not the independent audit the programme's rules ask for, and the coordinator has not seen the brief. It is a trial on one seed, about 3 CPU-hours.

Shell names: `D=origin/codex/minifly-consolidated-20261023` (the default branch); `V=origin/claude/vibrant-cerf-sje726` (the reviewer's branch); `M=MOD97_LOCAL_RULE_20261106`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`. Host: Linux x86-64, Python 3.13, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit.

## Terms

- *The network* is `arch20`'s recurrent network: a byte table (258 by 8), a layer `l1` (8 to H) with tanh, a GRU of U cells written out step by step in `arch26.forward`, and a readout `tanh(l2 z1 + lm h)` followed by `l3`. *Present size*: H = 512, U = 256. *Half width*: H = 256, U = 128.
- *Back-propagation* (rule B) is the exact gradient. *The sweep* (rule S) passes the error backward through each linear map by the signs of its weights times one scale for the map (`exp3_state.SignLinear`); it is `local=True` in `arch26.forward` and in `exp26_candidate.step`.
- *The first family* is the four operations add, sub, max and min under modulus 89, with the code bytes `p q v w`. Its *reduction pairs* are the pairs of add and sub whose answer needs the modulus (a + b ≥ 89; a − b < 0). *A birth* is a network trained on the first family alone.
- *A cell in reserve* is a cell that the first family does not use: its state is zero on every item of the first family.
- *The directions in use* at a matrix of connections are the directions spanned by the input vectors that the first family's items present to that matrix, a constant 1 appended for the bias. A change of the matrix that is zero on those directions leaves the matrix's output on those items as it was.

## The question

E43 showed that a born network loses the first family within 64 updates of an unrelated task. The reviewer's measurement on E43's networks (memo, section 29; `$R/cells_in_use.py`, `$R/directions_in_use.py`) gives the reason: every cell is in use on every item, and the first family's own items occupy 255 or 256 of the 257 directions of the cells' state. E48 showed that when the change for a new family is made in a part of the network that only that family uses, the earlier families are kept exactly and the new one is learned; there the part was supplied from outside.

This job gives the network such a part of its own. E41's half-width birth is placed inside a network of the present size, so that 128 cells are in reserve, and a rule confines new learning to the reserve and to directions that the first family does not use. Two questions: **is the first family kept, and what does the new task cost to learn?**

## What the record says (working rule 1)

- **E43** (`exp43_unrelated.py`; six E26 births of the present size). Under the arm `unrelated` without decay the first family's reduction pairs are at 0 of 3,916 at update 4,096 in all six networks and nearly so at update 64, while the unrelated task reaches 300 of 300. The arm `conflict` (the second family, modulus 97, alone) is the record for a related task.
- **E43's second addendum** (`e43_addendum2.py`). With one item of the first family for each new item, the first family's reduction score at update 4,096 is 95 to 105% of its start in all twelve runs; at update 256 the unrelated task is behind the unrehearsed run in all twelve comparisons.
- **E41.** The half-width births with half of the pairs taught reach 92.8 to 96.2 on held-out add and sub at 12 letters and 90.9 to 94.7 on reduction pairs (six files, on the default branch).
- **E48.** A term of rank 16 or 64, present only on the new family's items, keeps every earlier family item for item and reaches a higher level on the new family than E40's free learning by update 1,024; it is slower in the first 64 updates.
- **The reviewer's arithmetic for this job** (`$R/e51_design_checks.py`, no training, six births). The embedded network answers the first family as the half-width birth does. The first family then uses 129 of the 257 directions at the recurrent matrix and at `lm`, 36 to 41 of 513 at the input map, 11 at `l2`, and 329 to 437 at `l3`. A random change of every connection that the rule below allows to change, a sixth to a quarter of the whole network's size, leaves the first family's scores as they were; the same change without the rule leaves 39 to 58%.
- **One limit known beforehand.** The code of a byte (its row of the byte table passed through `l1`) lies in nine directions that every byte shares: 99.9% of the code of a byte that has never occurred lies in directions that the first family's bytes use. A new byte therefore has no direction of its own at the input map, and the old cells cannot learn a new response to it. The new task has to be learned by the cells in reserve, whose own connections are free.

## The learner, as code

For a seed and a rule (S or B), with `sd0` the state dict of `exp45/e41/e41_<rule>_h256_f50_<seed>.pt` (its file hash and tensors hash asserted against its receipt) and `m` a network of the present size whose every weight is first that of `exp41_scale.new_net(seed, 512, 256)` (the fresh network of the seed; `sw_net.SWNet(512, 256)` has the same parameters in the same order if `exp41_scale` cannot be imported beside the other modules):

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

Cells 0 to 127 are the *old cells*, cells 128 to 255 the *reserve*. Feed-forward units 0 to 255 are old, 256 to 511 added. Every weight not named above keeps the fresh network's draw: the reserve cells' own rows, the weights from reserve cells into old cells and into the readout, and the added feed-forward units' rows. The weights from reserve cells into old cells are not zero, because otherwise no error would reach the reserve.

**The reserve's update gates.** In torch's GRU the new state is `(1 - z) n + z h`, so `z = 1` keeps the state, which starts at zero.

- In the arms `open` and `rehearse`: `sd["rnn.bias_ih_l0"][384:512] = 10.0` and `sd["rnn.bias_hh_l0"][384:512] = 0.0` (the update-gate rows of the reserve cells), and nothing else. The gates start shut on every item and are free to change.
- In the arm `keyed`: the biases keep the fresh draws, and the forward pass sets `z = 1` exactly for the reserve cells on every item whose first byte, `x[:, 1]`, is one of the bytes in use (below). `arch26.forward`'s text is copied into the new runner with that one line added after `z` is computed (`z = torch.where(shut, torch.ones_like(z), z)`, with `shut[b, i] = in_use[x[b, 1]] and i >= 128`). **Every forward pass of a `keyed` run applies it: the update, under both rules, and the scorer's generation.** No existing source file is edited.

The embedded network is saved before any update as `e51_start_<open|keyed>_<rule>_<seed>.pt` and committed.

## The bytes in use and the directions in use (arm `keyed` only)

The 2,080 items from which they are taken are the first family's taught half at every number of letters:

```python
rng = Random(f"e51|{seed}"); items = {k: [] for k in range(13)}
for t in E.DEV:                                           # add, sub, max, min under 89
    taught = E.split(t, .5, seed)[0]
    for k in range(13):
        items[k] += [E.render(t, a, b, k, rng) for a, b in rng.sample(taught, 40)]
```

The SHA-256 of `json.dumps` of the list of `[prompt, answer]` pairs, k = 0 first, begins `d22b5733778491b1` for seed 73129 (`54bdb3c36ba2dc6e` for 93683, `f29bee3306c1a5d1` for 116089), by the reviewer's reader. *The bytes in use* are the start symbol and every byte of those prompts and answers but the end symbol: 46 bytes (the start symbol, `p q v w`, the ten digits, `+ - > <`, `=`, the 26 capital letters).

The items are run through the embedded network under teacher forcing, as in an update (`x = [start] + prompt + answer[:-1]`), with the `keyed` forward. For each matrix the input vectors are collected in float64, a constant 1 is appended, and with `C = X.T @ X / len(X)` the directions in use are the eigenvectors of `C` whose eigenvalue exceeds one millionth of the largest; `Vm` holds them as columns.

| Matrix (weight, bias) | Its input vectors | Collected at |
|---|---|---|
| input map (`rnn.weight_ih_l0`, `rnn.bias_ih_l0`) | `z1 = tanh(l1(book[x]))`, 512 values | every position |
| recurrent matrix (`rnn.weight_hh_l0`, `rnn.bias_hh_l0`) | the state before the step, 256 values (zero at position 0) | every position |
| `lm` | the state after the step | the three answer positions |
| `l2` | `z1` | the three answer positions |
| `l3` | `z2 = tanh(l2 z1 + lm h)`, 512 values | the three answer positions |

They are computed once, before the first update, and saved with the start file.

## The rule (arm `keyed`), as code

`exp26_candidate.step`'s text with two additions, one before and one after `opt.step()`. For a matrix with weight `W`, bias `b`, directions `Vm` and a set of rows:

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

The second projection is what makes the keeping exact.

| Tensors | Rows projected | Rows free |
|---|---|---|
| input map and its bias; recurrent matrix and its bias | the old cells' rows: `g * 256 + i` for g = 0, 1, 2 and i < 128 | the reserve cells' rows, all three gates |
| `lm`, `l2`, `l3` and their biases | all | none |
| `l1.weight`, `l1.bias` | not changed at all (restored after the step) | |
| `book.weight` | the 46 rows of the bytes in use are not changed (restored after the step) | the other rows |

Optimizer: a fresh `exp26_candidate.make_opt(m)` with the weight decay of group 0 set to 0, in every arm, as in E43's arms without decay.

Two properties of the construction to keep in mind. Under the sweep the scale of a map is the mean size of all its weights, the zero blocks included, so the sweep's signal in the embedded network is not that of the half-width birth; this is the same in every arm. And in the present code the old cells cannot learn a new response to a new byte (the limit above); only the reserve can.

## Arms

| Arm | Start | What may change | What it shows |
|---|---|---|---|
| `open` | reserve gates biased shut (10) | everything, as in E43 | the loss without any rule, and the speed of the new task without any limit |
| `rehearse` | as `open` | everything; each item of the stream is drawn with equal chance from the new task's stream or from the first family's taught half, as `e43_addendum2.py` does it (for `conflict` the same construction with the conflict stream in the place of the unrelated one, the choice by `Random(f"e51|{seed}|joint|conflict").randrange(2)`) | the plain baseline |
| `keyed` | reserve gates shut by the item's first byte | the rule | whether the first family is kept, and whether the new task can be learned in the reserve |

In `keyed` the condition is computed from the item (has its first byte occurred before?) and is not supplied for a family. It is still a device of the harness, and the report says so.

## Tasks, dose, rows

- `unrelated`: E43's task, stream and panels (`e43_task.py`; the arm `unrelated` of `exp43_unrelated.py`).
- `conflict`: the second family alone (modulus 97), E43's arm `conflict`: `X16.entry_stream(X16.family_entries("F1", seed), seed + 101)`, scored as E43 scores that arm.

4,096 updates, batches of 64. Scored at updates 0, 16, 32, 64, 128, 256, 512, 1,024, 2,048 and 4,096. Each network under the rule of its birth. Seed 73129: two rules, three arms, two tasks, twelve runs.

## Checks before the runs (a receipt for each)

1. **The embedding.** On E43's panel of the first family held out at 12 letters and on 2,000 of its items without letters, the embedded network's outputs at the answer positions equal the half-width birth's: to 1e-5 in the `keyed` start, to 0.02 in the `open` start. The scores of row 0 equal the half-width birth's on every panel in `keyed`.
2. **The directions.** The digest of the 2,080 items equals the reviewer's. The numbers of directions in use are 129 at the recurrent matrix, 129 at `lm` and 11 at `l2`; those at the input map and at `l3` are recorded (the reviewer's reader, with other draws for the added weights, finds 36 to 41 and 329 to 437).
3. **The rule.** (a) After 8 updates of `keyed` on the unrelated stream, no output at an answer position on the panels of check 1 has changed by more than 0.001, and every tensor or row that the table calls not changed is bit-equal to its start. (b) The same 8 updates with the two projections switched off change those outputs by more than 0.1. (c) On a batch of the first family every reserve cell's state is exactly zero at every position in `keyed`; on a batch of the new task it is not.
4. **The scorer.** In a `keyed` run the generated answers of the first family at row 0 equal those of the half-width birth item for item, which shows that the condition is applied in generation.
5. **The data.** The batch digests of the two streams at update 4,096 equal those in E43's receipts of the same seed (`unrelated`, `conflict`); for `rehearse` on `unrelated`, the digest of E43's second addendum.
6. **`open` is E43 in another network.** `open` with the embedding replaced by E26's birth of the seed reproduces E43's `unrelated` receipt of that seed and rule at row 64 (the first family's counts and the tensors hash).
7. **Cost.** A dry run of 64 updates of every arm with CPU-seconds an update and a scored row, and a forecast for the twelve runs.

## What is recorded at every scored row

- **The first family:** E43's scores 1 and 2 (all pairs without letters, add and sub split by reduction, max and min on all pairs; the held-out panel at 12 letters), and what the wrong answers on reduction pairs are, in the classes of E43's second addendum.
- **The new task:** E43's score 3 for its stream.
- **The reserve:** the number of reserve cells whose update gate is open by more than 0.1 (the mean of `1 - z` over the positions and items of the new task's panel); the largest size of a reserve cell's state on the first family's panel.
- **The weights:** E43's score 4 (movement of each group from the start), and the norm of each tensor.
- In `keyed`: the share of the first family's answers that equal row 0's, item for item.

Committed: receipts, logs, the start files with their directions, and the weights at updates 64 and 4,096 of every run. The report lists the committed files from `git ls-tree`.

## How the result is stated (fixed beforehand)

For each rule, task and arm:

- *Kept*: the first family's reduction score at update 4,096 is within 2 points of row 0. Otherwise the report gives what is left and what the wrong answers are.
- *Learned*: the new task's score on its never-taught items reaches 80% at some scored row and stays there at every later row. Its *cost* is that first row against the first such row of `open`.

No reading beyond these, and no statement about more than one seed.

## Predictions (the reviewer's, on file before any run)

| | Statement | Chance |
|---|---|---|
| P51.1 | `open` loses the first family on `unrelated`: below 5% on reduction pairs by update 64, both rules. Refuted if the reserve takes the new task and the family survives | 90% |
| P51.2 | `open` loses the first family's reduction pairs on `conflict` (below half of row 0 at update 4,096), both rules | 80% |
| P51.3 | `keyed` keeps the first family on both tasks, both rules. By construction | 97% |
| P51.4 | `keyed` learns `unrelated` | 60% under back-propagation, 45% under the sweep |
| P51.5 | `keyed` learns `conflict` | 50% and 35% |
| P51.6 | Where `keyed` learns `unrelated`, its cost is at most four times `open`'s | 60% |
| P51.7 | `rehearse` keeps the first family within 5 points on both tasks, both rules | 85% |

If P51.4 fails under both rules, cells in reserve that read the old cells but may not change them are not enough to learn a new task. If P51.4 and P51.5 hold, the reserve learner is the candidate for the long lives.

## What it cannot show

- That a learner can do without the device of `keyed`. The arm without it (reserve gates that only the optimizer opens, through directions the first family does not use) cannot work in the present code and waits for a birth with E50's code.
- How a reserve is shared among many tasks: here the first new task may take all of it.
- A life; more than one seed; a comparison with a store at equal memory, which belongs to the long-life world.
- Whether a network can be born with a reserve by itself: here the reserve is made by taking a narrower birth.

## Rules for the worker

- A new runner, `exp51_reserve.py`, in `$M`, built on `exp43_unrelated.py` and `e43_addendum2.py`, which are imported and not edited; outputs in `$M/exp45/e51/`. Every existing source file stays as it is.
- At most three processes, one thread each (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`), launched detached and outside any wrapper with a time limit. torch 2.15.0+cpu from `https://download.pytorch.org/whl/cpu` in a virtual environment; versions recorded in every receipt.
- The container can restart without warning and loses what is not pushed. Commit and push receipts and logs as runs finish; a run interrupted by a restart is started again from its beginning.
- Push only to your own branch. Never push to the default branch or to any other branch, and open no pull request. Other branches are read-only: fetch with `git fetch origin '+refs/heads/<name>:refs/remotes/origin/<name>'` and read with `git show`.
- Commit trailers as `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` says. No name of any language model in any committed file or commit message.
- The report, `E51_REPORT.md`, in standard academic English: full sentences, every term defined at first use, every figure with the receipt or the command that gives it.
- If the brief cannot be followed at some point, stop that part, say what you found, and ask.

## Reporting

Send the reviewer a short message with the tool `send_message` of the claude-code-remote server, to session `session_01YBMaMvtGtQvDFcgppgThrZ`, at each of these points: the checks passed (or which failed); the twelve runs finished; the report committed. Each message gives your branch, the commit and the main figures in at most ten lines. If you need a decision, ask in the same way and go on with whatever does not depend on the answer. The two other seeds are released by the reviewer after the trial is read.

## Done

The seven checks; the twelve runs; `E51_REPORT.md` with the table of every run at every row, the two readings for each run, the reserve's figures, the outcome of each prediction, and the list of committed files.
