# E61: the error relayed over a limited number of steps, in the network with limited connections and a limited input that both rules learn (18 November 2026, revision 2)

Written by the outside reviewer at the coordinator's decision 33 (`COORDINATOR_DECISIONS_20261118.md` on the default branch at e647e7d7), which asks for the plan's item 20 on E50's stage-3 network. It is step 3 of the chain of steps (`STEPS_FROM_THE_PRINCIPLE_20261118.md`, revision 2, in `COORDINATION_20261117` on the reviewer's branch). A threshold is reached here at the first of three consecutive scored rows at or above it (E50's definition), not at the first row from which the score stays at or above it (the chain's); for the record's sweep the two give 34,304 and 36,352 updates to 80 on R, and only the first is used. Revision 1 was checked by an agent of the reviewer that had seen no opinion of it (`CHECK_OF_TASK_E61_BRIEF_20261118.md` beside this file, with `check_e61_brief.py`); revision 2 enters its sixteen findings. It is not released: the coordinator releases it, and a worker starts nothing before that. One seed; this is the full-length trial that decision 33 (d) asks for before any other seed.

## Shell names and host

`D=origin/codex/minifly-consolidated-20261023` (the default branch, at e647e7d7); `V=origin/claude/vibrant-cerf-sje726` (the reviewer's branch: this brief and the reviewer's documents); `M=MOD97_LOCAL_RULE_20261106`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`. The host is Linux on x86-64 with four cores, Python 3.13 and torch 2.15.0+cpu. Every source file named below is on the default branch.

## Terms

- *The network* is that of E50's stage 3, the setting `C46-one` (`exp50_code.build("C46-one", 73129)`): 256 recurrent cells of the gated kind of E26; each cell reads 10 other cells, fixed for the run and drawn from the 48 cells around it on a grid of 16 by 16, and itself; the input is a fixed code of 512 values in which each of the 46 used bytes turns on 11 values of its own, and each cell reads 46 of the values, one of every used byte; the readout is dense (every one of the 512 readout units reads all 256 cells). The byte table and `l1` take no part.
- *The sweep* (rule S, `local=True`) sends the error back through each linear map by the signs of its weights times one scale for the map; *the exact gradient* (rule B, `local=False`) sends it back through the weights. Both are arms here, each named for what it is (decision 31): the sweep relayed over K steps and the exact gradient relayed over K steps. Whether the relayed signal must be an approximation is not ruled (decision 31: it is run, not ruled), and neither arm is called the candidate alone. (The reviewer reads Mr. Xu's answer of 18 November as placing the exact gradient outside what a neuron has; memo, section 33.28. That is a reading and not his ruling; by decision 31 it bears on how the arm is read and not on the runs.) *The reference* of decision 31 is back-propagation over the whole item on the dense network; for seed 73129 it is E42's configuration `G` under the exact gradient (`$D:$M/exp45/e42/e42_G_B_73129.json`: 80 on R after 6,400 updates, on A after 5,120), and the report gives it beside the record.
- *The record* is the two runs of E50's stage 3, `$D:$M/exp45/e50/e50_C46-one_<S|B>_73129.json`, in which the error is passed back over the whole item. They are not repeated.
- *A position* is one byte of an item. An item is the start symbol, a byte that names the task, two digits (the first operand), the operation byte, 0 to 12 letters, two digits (the second operand), the sign `=`, and the answer of two digits with an end symbol. With k letters an item has 10 + k positions under teacher forcing (10 to 22). The loss is taken at the three *answer positions* only (7 + k, 8 + k and 9 + k, counted from 0).
- *The state* of a cell at a position is its output there. *The error at a state* is the derivative of the loss with respect to it, as the rule in force computes it.
- *One step of relay* (see the next section) moves the error from the states of one position to the states of the position before. *K* is the largest number of steps over which an error is relayed. *The whole item* is K of at least 21.
- A run is scored every 256 updates (a *scored row*). A threshold is *reached* at the first of three consecutive scored rows at or above it; the first single row at or above it is given in brackets. A *level* at an update is the mean of the three scored rows that end there. *A* and *R* are the shares right on never-taught pairs (A: 600 items of addition and subtraction with 12 letters; R: 2,000 items whose answer needs the reduction by the modulus, written with no letters); *T* is the share right on 600 taught items with 12 letters. These are E50's definitions and E50's scorer (`exp42_cells.panels`).

## What one step is in this network (from the source)

In `e50_sparse_code.forward` the state of cell i at position t is computed from the input at position t and from `(Wh * mask) h(t-1)`, the states at position t - 1 of the 10 cells that cell i reads and of cell i itself, and from the cell's own state through the term `z * h(t-1)`. There is no other path between cells. Backward, the error at the state of cell i at position t therefore goes to the states at position t - 1 of the cells that cell i reads (through the signs of its weights under the sweep, through the weights under the exact gradient) and to its own state at position t - 1, through its own reading of itself in the recurrent map (crossed like the others) and through the term `z * h(t-1)` (differentiated exactly under both rules). One step is thus a step back in position and a step between cells at once; the source does not allow one without the other. Two consequences are stated here and in the report. First, because the readout is dense, every cell receives an error directly at every answer position; relay is not needed for the error to reach a cell, and what K limits is how far back in the item an answer's error can change what a cell did. Second, the limit cuts the cell's own path together with the paths between cells; the form in which a cell keeps its own error for the whole item and only the crossings between cells are limited would cost about K + 1 backward passes an update and is not run here.

## The question

How many steps must the error be relayed for the network to learn the first family? The record passes it over the whole item, which the principle does not allow as one global pass and allows as relay. The distances in an item say what K can reach: from the first answer position the second operand is 1 and 2 steps back, the operation byte 3 + k, the first operand 4 + k and 5 + k, the byte that names the task 6 + k. If the number of letters is uniform on 0 to 12, an answer's error reaches both digits of the first operand in no item for K of 4 or less, in 4 of 13 items for K = 8 and in 12 of 13 for K = 16. Whether a smaller K than the distance suffices is open, because the weights are the same at every position: what a cell learns from the second operand, two steps before the answer, it also applies to the first.

The scored items do not have the distribution of the training items. The items of R are written with no letters, so that the first operand stands 4 and 5 steps before the first answer position and K = 5 reaches both of its digits in every item of R. The items of A and of T are written with 12 letters, so that the first operand stands 16 and 17 steps before the first answer position and only K of 17 or more reaches both digits; K = 16 reaches the second digit only. With K between 5 and 16, items of R's form are among the training items in which the error reaches the first operand (one training item in 13 has no letters), whereas the items of A and T are reached only through what the weights, which are the same at every position, carry over from shorter items. The report states beside every reading on R that its items have no letters, and beside every reading on A or T that theirs have 12.

## What the record says

| Network of E50's stage 3, seed 73129, whole item | Exact gradient | Sweep |
|---|---|---|
| Updates to 80 on R [single row] | 15,360 [13,824] | 34,304 [34,304] |
| Updates to 80 on A [single row] | 14,080 [12,288] | 35,584 [33,536] |
| Levels of A, R, T at update 24,576 | 88.50, 88.87, 94.89 | 63.61, 65.50, 84.61 |
| Levels of A, R, T at update 49,152 | 93.83, 94.25, 98.33 | 85.89, 87.22, 94.00 |
| Levels of the length test at 12 and at 40 letters, update 49,152 | 93.44, 54.33 | 84.33, 27.78 |
| CPU-hours of the run (updates and scoring; one other process beside it) | 1.90 | 2.18 |
| For comparison, the dense network with a learned code, whole item (E42's `G`; under the exact gradient it is decision 31's reference): updates to 80 on R | 6,400 | 6,656 |

(Verified by the reviewer: `$V:$R/E50_STAGE3_AND_REPORT_VERIFICATION_20261118.md`; the levels and the CPU-hours are read from the two receipts, the last row from E42's receipts.) No run exists in which the error is relayed over fewer steps than the whole item on any network with limited connections.

## Runs

Twelve runs of 49,152 updates on seed 73129: K = 0, 1, 2, 4, 8 and 16, each under the sweep and under the exact gradient. Order: the six runs of K = 16, 8 and 4 first, then K = 2 and 0, then K = 1. Everything other than the update is the record's: the network and its start, the stream (`E.batches(tp, seed, E.KMAX["M"])` with `tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]`, half the pairs), batches of 64, AdamW at rate 0.001 with betas 0.9 and 0.98 and weight decay 0.25 (`R42.make_opt`), the panels and the scorer.

## The runner

A new file, `$M/exp61_relay.py`; no existing source file is edited. It imports `exp50_code` as `R50`, `e50_sparse_code` as `S50`, `e42_cells` as `L42`, `e49_interface` as `I49`, `exp49_interface` as `R49`, `exp42_cells` as `R42`, `exp26_candidate` as `C26` (first, as in `exp50_code.py`), and `lin` from `exp3_state`; `E = C26.E`, `X = C26.X` and `R_ = C26.R`, as in `exp50_code.py`. The names of `exp50_code` that the copied text uses are reached as `R50.<name>`, which is not counted as a change. It is committed before the first check is run.

    python3 exp61_relay.py run <K|whole> <S|B> 73129 [--dose N] [--out DIR] [--ckpt N] [--resume] [--die-after N]
    python3 exp61_relay.py check <1|2|3|4|5|7> [--qual DIR]

1. **The run** is a copy of the text of `exp50_code.run` for the setting `C46-one` with these changes and no other: `experiment="E61"`; the new entries `K` and `relay`; with `whole` the update is `R50.step` unchanged, and with a number K it is the update of point 2; the cosines of point 4 in place of `cosine_first_batch_start` and `cosine_first_batch_end`, and the set of cosines of point 4 at update 24,576 in place of the entry `cosine_first_batch` of `first_dose`; the entries of point 3; saving and resuming (point 5); the receipt is `$M/exp45/e61/e61_K<K>_<rule>_73129.json` (`e61_whole_…` for `whole`, `_dry<N>` before the seed for a run of N updates other than 49,152), the other files with E50's endings; the variable `E61_BESIDE` in place of `E50_BESIDE` (the worker sets it at the start of a run to the number of other runs in progress); a function `provenance` of this file that is `R50.provenance`'s text with `exp61_relay.py` as the runner, so that `sources_sha256` covers `exp61_relay.py` and the files of `R50.SOURCES`. The network is built by `R50.build("C46-one", seed)`, scoring is installed by `R50.install(n)`, and every assertion of `exp50_code.run` at a scored row is kept (the masks, the fixed code, the byte table and `l1` bit-equal to their start). The report shows the difference between the two texts.

2. **The update with the error relayed over K steps.** The reference form is the following; `n` is the object that `R50.build` returns and `U` is 256.

       def one_step(n, gi, hp, local, scale):          # the step of S50.forward for any rows: gi (..., 768), hp (..., 256)
           m = n.m
           gh = L42.rec(n.layer, m, hp, m.rnn.weight_hh_l0, m.rnn.bias_hh_l0, local, scale)
           i_r, i_z, i_n = gi.chunk(3, -1); h_r, h_z, h_n = gh.chunk(3, -1)
           r = T.sigmoid(i_r + h_r); z = T.sigmoid(i_z + h_z); c = T.tanh(i_n + r * h_n)
           return (1 - z) * c + z * hp

       def relayed_grads(n, x, y, mask, local, K):
           m, layer, iface = n.m, n.layer, n.iface
           z1 = n.table[x]; B, L = x.shape
           Wi, bi, Wh, bh = m.rnn.weight_ih_l0, m.rnn.bias_ih_l0, m.rnn.weight_hh_l0, m.rnn.bias_hh_l0
           gi = I49.masked(z1, Wi, bi, iface.mask_in.repeat(3, 1).float(), local)
           scale = layer.sign_scale(m) if local else None
           with T.no_grad():                           # (a) the states, without credit
               h = T.zeros(B, U); before = []
               for t in range(L):
                   before.append(h); h = one_step(n, gi[:, t], h, local, scale)
           Hp = T.stack(before, 1).requires_grad_()    # h(t-1) for every position, as a leaf
           Hs = one_step(n, gi, Hp, local, scale)      # (b) the step of every position at once, with credit for one step
           Hd = Hs.detach().requires_grad_()           # (c) the readout, and its error at every state
           z2 = T.tanh(lin(z1, m.l2, local) + lin(Hd, m.lm, local))
           loss = E.cross_entropy(lin(z2, m.l3, local)[mask], y[mask])
           for q in m.parameters(): q.grad = None
           loss.backward()                             # gradients of l2, lm and l3; Hd.grad is the error at the states
           d = Hd.grad; total = d.clone()
           for k in range(K):                          # (d) K steps of relay
               g, = T.autograd.grad(Hs, Hp, grad_outputs=d, retain_graph=True)
               d = T.zeros_like(g); d[:, :-1] = g[:, 1:]        # the error at h(t-1) is the error at the states of position t-1
               total = total + d
           gs = T.autograd.grad(Hs, [Wi, bi, Wh, bh], grad_outputs=total)       # (e) the cells' own weights
           for q, g in zip([Wi, bi, Wh, bh], gs): q.grad = g
           return float(loss.detach())

   The update is `relayed_grads` followed by `opt.step()`. The forward values are those of `S50.forward` to rounding. With K = 0 no error leaves the position at which it arises; with K of 21 or more the gradients are those of the record's update to rounding. Under the sweep the gates' own nonlinearities and the term `z * hp` are differentiated exactly, as in the record. Two economies are allowed if checks 2, 3 and 4 pass with them. First, in (d), a second graph `Hr = one_step(n, gi.detach(), Hp, local, scale)` built with a crossing written in the new file that has the forward and the backward value `gx` of `L42.MaskedSign` and returns `None` for the weight and the bias when `ctx.needs_input_grad` says they are not needed, called with `Wh.detach()` and `bh.detach()`; the K rounds use `Hr`, and (e) uses `Hs` as before. (`ctx.needs_input_grad` is fixed when the forward pass runs, so that one graph cannot serve both.) Second, in (d), relaying only the rows of `(item, position)` at which `d` is not zero (at most three for each item in a round), by computing `one_step` again on those rows from `gi.detach()` and from the rows of `Hp.detach()` made leaves that require a gradient. The report says which form ran.

3. **What the receipt records once,** from the first 256 batches of the stream (drawn from a second call of `E.batches(tp, seed, E.KMAX["M"])`, which gives the same stream as the run's without advancing it): the number of items by number of letters, and for the run's K the share of items in which the error of the first answer position reaches the second digit of the first operand (K at least 4 + k), both of its digits (5 + k), the operation byte (3 + k) and the byte that names the task (6 + k); for the panels, the distances of the two digits of the first operand from the first answer position (R: 5 and 4; A and T: 17 and 16); and the multiply-adds of one update in the form that ran, by a count written in the runner: those of the record's update plus, for each of the K rounds, those of the crossings of (d) on the rows relayed. **Each scored row records in addition** `multiply_adds_with_relay`, that count times the updates made; the field `multiply_adds` is kept and does not depend on K. For a run with `whole`, K is 21 in points 3 and 4.

4. **Cosines during the run.** Before the first update and after every 4,096 updates the receipt appends to a list `cosines` the update and, as the mean over 16 batches of the cosine computed on each batch (the second to the seventeenth batch of the stream, drawn once from a call of `E.batches` of their own), in double precision and over all entries of the tensor, for the recurrent map and for the input map: the cosine between the run's gradient (its rule, its K) and the gradient of the same rule over the whole item (`R50.map_grads(n, x, y, mask, local)`), and the cosine between the run's gradient and the exact gradient over the whole item (`R50.map_grads(n, x, y, mask, False)`). Every gradient is set to `None` before and after a set and no step of the optimizer is taken; at the first two sets the run asserts that the tensors of the network and the optimizer's running averages are unchanged by the set.

5. **Saving and resuming,** as in `$D:$M/exp52_resume.py`. A saved state is `<receipt's stem>.ckpt`: the number of updates made, `m.state_dict()`, `opt.state_dict()`, the receipt so far and the counters of CPU time. It is written at every multiple of `--ckpt` updates below the run's number (4,096 if the option is absent; with 0 no saved state is written and the condition is not evaluated; otherwise the number is a multiple of 2,048, which the run asserts), at a scored row, after the row and its cosines are in the receipt and the receipt is written; it is written under another name and renamed with `os.replace`. With `--resume` and a saved state present the run builds the network and the optimizer as at the start, draws the first u batches again into a new digest, asserts that the digest equals the `data_digest` of the saved receipt's row of update u, that the rebuilt network's `R_.tensors_sha(m.state_dict())` equals the saved receipt's `start_tensors_sha256` and that the SHA-256 of the starting file on disk equals its `start_sha256` (the starting file is not written again), loads the two states, asserts that `R42.opt_record(m, layer, opt)` equals the receipt's `optimizer`, sets `first` to the first of the batches drawn again and keeps the saved receipt's `first_batch_loss_start` without computing it again, takes the saved receipt (cut back to the rows and the sets of cosines up to update u, if the receipt on disk is ahead of the saved state) as its receipt, does not score update u a second time, appends the update and the time to a list `resumes`, and goes on from update u + 1; the 16 batches of the cosines are drawn again from their own call; it then refuses only the final weights and the answers file. With `--resume` and no saved state it is an ordinary start. The saved state is deleted when the final receipt is written. `--die-after N` ends the process with `os._exit(9)` after update N (for check 7 only).

6. **Files committed:** the receipts, the starting weights, the weights at update 24,576 and the final weights (with `git add -f`), both answers files and the logs of every run and check. A weights file is about 5 MB, so that the twelve runs add about 180 MB; check the free disk before the runs.

## Checks

Checks 1 to 5 and 7 are made before the runs; each writes `$M/exp45/e61_qual/check<n>_73129.json` with an entry `pass`. Check 6 is made inside every run. If a check fails, the worker stops, reports the first difference and asks. *The check batches* are the first 8 batches of the stream at the start and the same 8 after the 512 updates of check 1's dry run under the rule concerned. The weights after 512 updates are made again inside each check by `R50.build("C46-one", 73129)`, `R42.make_opt` and 512 calls of `R50.step` on the stream, and the check asserts that `R_.tensors_sha(m.state_dict())` equals the `tensors_sha256` of check 1's dry run under the same rule. *Equal to rounding* means, for each of the ten trained tensors (the weights and biases of the input map, the recurrent map, `lm`, `l2` and `l3`): cosine in double precision at least 1 - 1e-6 and largest difference at most 1e-4 of the tensor's largest entry; and losses that differ by at most 1e-6.

1. **With `whole` the run is the record.** A dry run of 512 updates under each rule: `start_tensors_sha256` is the record's, and the scored rows of updates 0, 256 and 512 equal the record's in the loss, in every count and in the batch digest. If the counts and the digests are equal and the losses differ by at most 1e-6, the worker reports the difference and goes on.
2. **K = 24 is the record's update.** On the check batches, under each rule: the gradients of `relayed_grads` with K = 24 are equal to rounding to those of the record's update (`fwd`, `cross_entropy`, `backward`), and the states `Hs` differ from those of `S50.forward` by at most 1e-6.
3. **K = 0 is the form with the state detached at every position.** On the check batches, under each rule: equal to rounding to the gradients of a forward pass written in the check as `S50.forward` with `h.detach()` given to `L42.rec` and to the term `z * h` (the form of `arch26.forward` with `through_time=False`).
4. **K = 2 and K = 5 are the direct definition.** On the first 2 check batches at the start and the first 2 after the dry run, under each rule: equal to rounding to the sum, over the three answer positions j of an item, of the gradients of that position's loss, computed with a forward pass written as `S50.forward` in which, for each item, the state that enters position a - K is replaced by its detached value (`h = T.where(cut[:, t, None], h.detach(), h)` before the step of position t, with `cut[b, t]` true where t equals the j-th answer position a of item b minus K). The loss of position j is the cross-entropy summed over the items at that position and divided by `mask.sum()`, so that the three add up to the batch's loss; the check asserts that they do.
5. **Cost.** CPU-seconds an update for each K and rule (64 updates each, after 8 that are not counted) in the reference form and, if checks 2 to 4 pass with them, in the form with both economies; a scored row; a set of cosines; a forecast for the twelve runs in each form, with the checks' own cost. The runs use the cheapest form that passed checks 2 to 4. If its forecast exceeds 36 CPU-hours, the two runs of K = 1 are left out and the report says so; if it still exceeds 36, the worker asks before starting. The check passes when every measured update is finite and the forecast of the form chosen is written. (The reviewer's count gives about 35 CPU-hours for the reference form and about 27 with the second economy.)
6. **Inside every run:** the assertions of `exp50_code.run` at every scored row. A failure stops the run.
7. **A run that is killed and resumed equals an uninterrupted one.** Two dry runs of 8,192 updates with K = 4 under the sweep, each in a folder of its own: (i) with `--ckpt 0`; (ii) with `--ckpt 4096 --die-after 7000`, which ends with exit code 9, then the same command with `--resume` in place of `--die-after 7000`. Before the second call the check records that the receipt on disk has its last scored row at update 6,144, ahead of the saved state of update 4,096. The two finished receipts have the same `tensors_sha256` and `start_tensors_sha256`; they are equal in every field except those whose names begin with `cpu` or `wall`, `provenance` and `resumes`; each has three sets of cosines (updates 0, 4,096 and 8,192), and these are equal; the optimizer has made 8,192 steps in each, and `optimizer_read_back` equals `optimizer`; `cpu_updates` of the last row of (ii) is at least 1.5 times that of its row of update 4,096; (ii) has one entry in `resumes`, for update 4,096, and (i) none. (About 1.2 CPU-hours.) **The environment of the child processes:** the harness sets variables on import that a child must not inherit, or the child is refused (this fault was met in E52, E57 and E53; decision 37). The check starts each child with `env = {k: v for k, v in os.environ.items() if not k.startswith(("E11_", "E16_", "EXP45_"))}` and `cwd` set to `$M`; the same holds for any script of the worker that starts runs from Python.

An interrupted check is made again from its beginning, after its folder has been moved to `$M/exp45/e61_qual/interrupted/`.

## How the result is stated (fixed beforehand)

For each run the report gives: updates to 80 on R and on A [single row]; the levels of A, R and T and of the length test at 12 and 40 letters (`len12`, `len40`) at updates 24,576 and 49,152; beside the updates to 80 on R, the multiply-adds to that row (`multiply_adds_with_relay`) and the CPU-seconds of the updates to that row, with the number of processes beside each run, and the record's (read from the record's receipt at its row); the shares and distances of point 3; the cosines of point 4 at the start, at 24,576 and at the end. All beside the record's column of the same rule and beside decision 31's reference.

For each rule, a K is *sufficient* if its run reaches 80 on R within 49,152 updates in at most 1.25 times the record's updates (19,200 for the exact gradient, 42,880 for the sweep); *slower* if it reaches 80 on R later than that; otherwise *not reached*. (The factor 1.25 is the reviewer's choice; on one seed a difference smaller than that is not read.) The sentences:

- If a sufficient K exists for the sweep: "On seed 73129, in the network of E50's stage 3, the sweep with the error relayed over K steps reached 80 on R after u updates, against 34,304 with the whole item; with this K the error of the first answer position reaches both digits of the first operand in p of the training items (point 3) and, when K is at least 5, in every item of R, which has no letters." K is the smallest sufficient one; the same sentence is given for the exact gradient with its own smallest K.
- If no K of 16 or less is sufficient for a rule: "On seed 73129, in the network of E50's stage 3, [the rule] did not reach 80 on R within 1.25 times the record's updates when the error was relayed over 16 steps or fewer; the whole item is 21 steps, and K of 17 to 20 was not run."
- For K = 0: "With no relay (K = 0) the levels of A, R and T at update 49,152 were …", whatever they are.
- For every K at which both rules reach 80 on R within 49,152 updates: the ratio of the sweep's updates to 80 on R to the exact gradient's, beside the record's 2.23 (34,304 / 15,360).

No sentence says that a number of steps "is enough" or "is needed" without "on seed 73129, in the network of E50's stage 3". The report states that the result is of one seed and is not confirmed, that the readout is dense, and the two consequences of the section on what one step is.

## Predictions (the reviewer's, recorded before any run)

| | Statement | Chance |
|---|---|---|
| P61.1 | Checks 1 to 5 and 7 pass at the first attempt; a check made again from its beginning after an interruption of the host is the same attempt | 0.50 |
| P61.2 | None of the runs of K = 0, 1 and 2 that were made reaches 80 on R within 49,152 updates (the report says how many were made) | 0.70 |
| P61.3 | Neither run of K = 4 reaches 80 on R | 0.55 |
| P61.4 | K = 16 is sufficient for the exact gradient | 0.70 |
| P61.5 | The sweep with K = 16 reaches 80 on R within 49,152 updates | 0.60 |
| P61.6 | K = 8: the exact gradient reaches 80 on R within 49,152 updates | 0.70 |
| P61.7 | K = 8: the sweep reaches 80 on R within 49,152 updates | 0.50 |
| P61.8 | At every K at which both rules reach 80 on R, the sweep needs more updates; not scored if there is no such K | 0.85 |
| P61.9 | With K = 0 the level of T at 49,152 is below 50 under both rules | 0.65 |
| P61.10 | With K = 8 under the exact gradient the level of A at 49,152 (items with 12 letters) is at least 10 points below the level of R (items with no letters) | 0.60 |

## What it cannot show

- Anything beyond one seed, one family, half the pairs and this network; the other seeds follow only on the coordinator's word.
- The number of connections: every cell reads 10 cells here. The number of connections is E62's subject. With 256 cells, a cell that read about fifty cells drawn from all cells would reach nearly every cell in two steps (50² = 2,500 > 256), and one that read all 48 cells of its neighbourhood would reach every cell in three; Mr. Xu's example of several passes (1, 50, 50², 50³) cannot be reproduced at this width, and this brief does not claim it. In this network (10 of the 48 neighbours) an error at one cell has reached every cell after four or five steps, depending on the cell; because the readout is dense, this number does not limit which cells receive an error.
- Relay between cells apart from the step back in position, and relay as the only way an error reaches a cell: the readout is dense (see the section on what one step is). The network in which only some cells are read is E53's.
- What K does to items of other lengths than those scored: R has no letters, A and T have 12. The report gives the levels of the length test for each run beside the record's; no sentence is fixed for them.
- Whether a cell could hold what the relay needs: to relay over K steps a cell must keep its own quantities of the last K positions. This is not examined.
- Requirement A, and anything about a life.

## Rules for the worker

- Start a branch of your own from `$D` at e647e7d7; fetch the reviewer's branch by `git fetch origin '+refs/heads/claude/vibrant-cerf-sje726:refs/remotes/origin/claude/vibrant-cerf-sje726'` and no other. Install torch 2.15.0+cpu from `https://download.pytorch.org/whl/cpu` and numpy in a virtual environment under Python 3.13; every receipt records the versions. Run at most three processes at a time, each with one thread (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`), from the folder `$M`.
- This host stops a machine about five minutes after its session's turn ends; the disk is kept and detached processes are terminated. While runs proceed, do not end your turn: issue a command that waits for the processes and repeat it until they have finished; commit and push the logs and receipts between two such commands. Start every run with `--resume` from the first time on, with its output appended to a log of its own. A run interrupted before its first saved state is started again after its files have been moved to `$M/exp45/e61/interrupted/`. The report says which runs were resumed, from which update, and which were started again.
- Push only to your own branch; never to the default branch or any other; open no pull request; read other branches with `git show`.
- Every commit carries the trailers that `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` prescribes. Apart from the trailers, no name of a language model appears in a committed file or commit message.
- The report, `$M/exp45/e61/E61_REPORT.md`, is in standard academic English: full sentences, every term defined at first use, every figure with the receipt or command that gives it; the committed files are listed from `git ls-tree`. Its tables are made by a script `e61_tables.py` from the receipts.
- If the brief cannot be followed at some point, stop that part, say what you found, and ask.

## Reporting

Send the reviewer a short message with the tool `send_message` of the claude-code-remote server, to session `session_01YBMaMvtGtQvDFcgppgThrZ`, at each of these points: the checks passed (or which failed), with the forecast of check 5 and the form of the update that ran; the six first runs have written the scored row of update 24,576, with their levels of A, R and T; the six first runs finished; all runs finished, with the table; the report committed. Each message gives your branch, the commit and the main figures in at most ten lines. If you need a decision, ask in the same way and go on with whatever does not depend on the answer.

## Done

The job is done when the receipts of checks 1 to 5 and 7, the runs with their receipts, weights, answers files and logs, and `E61_REPORT.md` exist; the report holds the table of every run beside the record, the readings in the words fixed above, the cosines, the outcome of each prediction, the difference between the runner's `run` and `exp50_code.run`, and the list of committed files.

## History of the revisions

Revision 1 (18 November 2026): written at the coordinator's decision 33. Decision 33 (a) names "the network of E50's stage 3 (16 inputs a cell)"; stage 3 is the setting `C46-one`, in which a cell reads 46 values, one of every used byte, and it is the setting that both rules are verified to learn; the setting with 16 values a cell (`C16-in`) reached 80 on R under neither rule. The brief takes `C46-one`.

Revision 2 (18 November 2026): the sixteen findings of the check entered (`CHECK_OF_TASK_E61_BRIEF_20261118.md`; the reviewer ran its script again, 1.4 seconds, with the same output). The check's own numerical test, written by hand in double precision, found the update of point 2 equal to the direct definition of check 4 for K = 0, 1, 2, 5, 12, 13 and 17 under both rules (largest difference 1.1e-15), equal to the full gradient for K at least the length, and equal to the gradient with the state held constant for K = 0. The weightiest findings: the scored items have fixed numbers of letters (R none, A and T twelve), which the question, point 3 and the fixed sentence now state; the terms followed the reviewer's reading where decision 31 leaves the matter to be run, and decision 31's reference (E42's `G`) was missing; a resumed run would have overwritten the first batch's loss and taken a cosine on another batch, and check 7 would not have noticed; the sentence for a rule with no sufficient K said more than the runs show; check 7 now has a receipt ahead of the saved state and a set of cosines after the resumption; the first economy is restated so that it can be written; the measure of computation is defined (`multiply_adds_with_relay`); the length test is reported. From the check of E62's brief: the hashes asserted at a resumption. The reviewer set the chances of P61.3, P61.6 and P61.7 again and added P61.10 after finding 1, before any run.
