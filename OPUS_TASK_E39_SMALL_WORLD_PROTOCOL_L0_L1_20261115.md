# E39 (jobs L0 and L1): the small world — a protocol for long lives, and its calibration (15 November 2026, revision 1)

Evidence level: none for L0 (it is an instrument); E2 for anything L1 measures, and a result in the small world is a claim only after it is repeated at full scale. The programme is the outside reviewer's (its memo, revision 3, sections 17 and 18, on `claude/vibrant-cerf-sje726` at 823a3884), written after the user's statements of 15 November. The specification below is mine. **Not released: it waits for the reviewer's audit, the one audit this protocol gets.** One worker. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106; K=PROJECT_DOCUMENTS/COORD_CHECKS_20261115`.

## The question the protocol must be able to answer

The user, 15 November: the goal is persistent learning at fixed memory and computation, and a learner that is slower at first may overtake a conventional one later; as long as ours is still learning and the conventional one is stuck, this cannot be excluded, and many short lives cannot show it.

So the protocol records trends over a long life, not an average over a short one: score against age, the cost of learning the n-th new task, and what is kept when a task returns. Its first use (job L2, not in this brief) needs no candidate: do naive backpropagation and backpropagation with replay get stuck over a life of 1,024 visits?

Nothing measured so far can answer this: fixed panels after a fixed dose, with backpropagation as the ceiling, allow parity at best, and our longest lives have six to nine stages.

## The world, as code

**`$K/sw_world_spec.py` on the default branch is the specification** (pure Python, no torch; run it to see a world). Copy it to `$M/smallworld/sw_world.py` and import it; do not rewrite it. The reviewer writes a second implementation from the text below, and the two must agree on the digests in check 1.

- **Tasks.** A task is an operation and a modulus with a two-byte code of its own. 18 operations: the 16 linear ones `(c1·a + c2·b) mod p` with c1, c2 in {1, 2, −1, −2} (among them add and sub), and max and min. 30 moduli, 8 to 37. That gives 540 tasks. Each operation has one symbol byte, the same under every modulus; codes are arbitrary and say nothing about the modulus.
- **Items.** `code(2) a(2) symbol d_1..d_k b(2) =` then the answer: two digits and the end symbol. k is 0 to 4 letters. The longest prompt is 12 bytes.
- **Taught and asked pairs.** Each task's p² pairs are split once in half. Taught pairs come with their answer. Asked pairs are never taught.
- **Groups and visits.** Tasks are met in groups of 2. A visit is 256 updates on one group. Visit 0 is group 0. After that a visit brings the next new group with chance 0.2; otherwise it returns to a met group, drawn so that short (1 to 4 visits), medium (5 to 32) and long (33 or more) absences are about equally common.
- **One stream, read once.** Every update takes 64 items from the stream: 48 taught and 16 asked, each from a task of the visit's group chosen at random.
- **Everything follows from the world's seed**, through `random.Random` objects seeded with strings (`f"world|{seed}|tasks"` and so on, as the file writes them). The schedule and the items are the same for every arm.

What the generator gives (`$K/OUTPUT_SW_WORLD_MAC.txt`): in 1,024 visits, worlds 1, 2 and 3 meet 189, 211 and 192 new groups of the 270 available, and returns fall 273/279/283, 289/264/260 and 309/260/263 into the three absence classes, the longest absence 758 to 922 visits.

## The learner

A GRU of E26's form, narrower. Write `sw_net.py` with a class holding the same parameter names as `arch20.NetRNN("gru")` (`git show $D:$M/arch20.py | sed -n '36,50p'`): `book = Embedding(258, 8, padding_idx=256)`, `l1 = Linear(8, H)`, `rnn = GRU(H, U)`, `lm = Linear(U, H)`, `l2 = Linear(H, H)`, `l3 = Linear(H, 257)`, with `kind = "gru"` and `units = U`, so that `arch26.forward(m, x, local, True)` runs on it unchanged. **Default size H = U = 64.** Initial values: `T.manual_seed(world_seed)` then the constructor.

- **Update.** `exp26_candidate.step`'s text (lines 84-91): `x, y, mask = exp45.batch_tensor(items)` on the 48 taught items, answer-masked cross-entropy, one AdamW step. `local=False` is backpropagation; `local=True` is the sweep.
- **Optimizer.** One for the whole life: `T.optim.AdamW` with betas (0.9, 0.98), the embedding table in its own group without decay, as `exp16_integrated.adamw16` (lines 176-187). Learning rate and decay are arguments; defaults 0.001 and 0.25.
- **Asked items are answered before the update of their batch and nothing is learned from them:** greedy generation of at most 3 bytes under `T.no_grad()`, correct if all three bytes (two digits and the end symbol) are right.

## Arms the runner must offer

| Arm | Update |
|---|---|
| `naive` | backpropagation on the 48 taught items |
| `sweep` | the sweep on the 48 taught items |
| `replay` | backpropagation; a share ρ of the 48 slots is filled from a buffer of past taught items, and only the first (1 − ρ)·48 new taught items are learned from |
| `sweep+replay` | the same under the sweep |

**The buffer.** It holds raw items. Its capacity in bytes is 4 times the number of weights of the network, one copy of the weights' memory; an item costs the length of its prompt plus its answer. All 48 new taught items of every update are offered to it by reservoir sampling over items (generator `Random(f"world|{seed}|replay")`); draws for a batch are uniform over the buffer from the same generator. ρ is an argument (0.25 and 0.5 for a start). Every arm consumes the same 64 stream items per update.

Later candidates (an anchor, a slow copy, a step size per weight, reminders, the confidence gain) are added as arms by a one-page roster note; the runner's update is the only place they touch.

## What a life writes

One receipt (JSON) and one event file per life, in `$M/exp45/e39/`, named `sw_<arm>_w<world>_v<visits>[_<setting tag>]`; the runner refuses an existing path. Weights are saved at the end and at visits 64 and 256.

**The event file** has one record per asked item, in stream order: update index, visit, task id, correct (0 or 1), blind-correct (0 or 1), needs-wrap (0 or 1, from `needs_wrap`), and the count of that task's taught items learned from so far. Store it as compact arrays.

**The blind predictor** answers an asked item with the answer of the most recent taught item of the same task presented before this update, without reading the operands; it is wrong if there is none.

**The receipt** has the settings, provenance and source hashes as E26's receipts do, the schedule, a digest of all batches, and the ledger: stored numbers by kind (weights, optimizer state, buffer bytes, any anchor), updates, and stream items per update.

## The scorer: quantities per life, computed from the event file alone

`sw_score.py`, reading no weights.

| Quantity | Definition |
|---|---|
| Score by age | Share of asked items correct in each visit; the blind predictor's share beside it. |
| Cost of the n-th new task | For the n-th task met: the count of its taught items learned from before the first asked item at which the mean of its last 32 asked items is at least 0.8. "Not reached" if never; then the mean of its last 32 asked items in the life. |
| Kept at return | For each return visit and each task of the group: the mean of its first 32 asked items in that visit, with the absence in visits; for linear operations also split by needs-wrap. |
| Relearning at return | The taught items of that task learned from within the visit before its 32-item mean is at least 0.8 again; "not reached" otherwise. |
| Late level | Mean score by age over the last quarter of the visits. |
| Trend of the score | Least-squares slope of score by age against the natural logarithm of (visit + 1), over the second half of the visits. |
| Trend of the cost | Least-squares slope of log(cost) against log(n), over tasks that reached 0.8, n from 2. |

No marks in points are fixed in this brief. The reviewer's two withdrawn marks came from fixing thresholds before seeing the scale of the numbers; L1 shows the scale first.

## Checks (job L0)

1. **The generator.** `sw_world.py` is byte-identical to `$K/sw_world_spec.py`. The digests of the first 16 visits of worlds 1, 2 and 3 are 64486ccd…, cad4f54b… and bd6ea889… (`$K/OUTPUT_SW_WORLD_MAC.txt`).
2. **Asked is never taught.** For every task, the taught and asked pair sets are disjoint (the file asserts it), and the runner asserts that no item it learns from has an asked pair.
3. **Asking does not teach.** The `tensors_sha256` of the weights and of the optimizer state is unchanged across the answering of a batch's asked items.
4. **Same stream for every arm.** The batch digest of a 4-visit life is equal across the four arms.
5. **The forward and the two rules.** On the first batch, the network's logits equal those of `torch.nn.GRU` run on the same parameters to 1e-6; under `local=False` the parameter gradients equal autograd's through that reference to 1e-5 of each tensor's largest entry; under `local=True` they differ from them.
6. **Replay.** With ρ = 0, `replay` gives the same weights as `naive` after 64 updates. The buffer never exceeds its capacity in bytes. With ρ = 0.5, 24 items of each batch come from the buffer once it holds that many.
7. **The scorer.** On a hand-made event file of two tasks with known answers, every quantity in the table equals its hand-computed value. The reviewer's numpy scorer and this one agree on the event file of one 16-visit life.
8. **Cost.** CPU-seconds per visit at the default size, for `naive` and `sweep`, and a forecast for a life of 1,024 visits.

Report L0's checks before L1 starts; do not wait for me unless one fails.

## Job L1: calibration (runs after L0's checks pass; no further audit)

Worlds 1 to 8, lives of 16 visits, arms `naive` and `sweep`, default settings. Sixteen short lives.

**The reviewer's gate.** Under each arm, the first group's asked items are at least 90% right by the end of its first visit in six of eight worlds; and the measured cost puts a life of 1,024 visits within 6 CPU-hours.

**I expect the first half of that gate to fail at any affordable visit length, and say so now.** At full scale a birth needs about 8,000 updates before held-out pairs are mostly right; a first visit of 256 updates on a 64-unit network will not do that for a task it meets with nothing behind it. This does not make the protocol unusable: the cost of a task is counted across its visits, and tasks return. So L1 reports, for each arm and world, and I decide the size with the reviewer from these figures:

- the first group's asked score at the end of each of its visits within the 16;
- the cost of each new task met in the 16 visits, or "not reached" with its level;
- score by age and the blind predictor's;
- for linear tasks, wrap and no-wrap scores at each task's first return;
- CPU-seconds per visit.

**The size ladder, tried in this order if the first group is not at 90% by the end of its second visit in six of eight worlds:** visits of 512 updates; then moduli 8 to 19 with a new-group chance of 0.1 (so that the 108 groups outlast 1,024 visits); then H = U = 96. Each rung is a constant in a copy of the world file under a new name, with its digests recorded; the 6-hour bound on a 1,024-visit life holds at every rung. Stop at the first rung that passes and report all rungs tried.

## Not in this brief

- **L2**, whether the conventional learner gets stuck (`naive` and `replay` at 64, 256 and 1,024 visits; the sweep arms at 64 and 256), comes as a one-page roster note with predictions on file, after L1 fixes the size. It needs no audit; any change to the generator, runner or scorer does.
- **Candidates** come after L2.
- **Comparison rules** (late level, trend, the horizon rule of lengths growing by four, intervals across worlds) are the memo's sections 17.4 and 17.5 and are fixed in the L2 note.

## What the small world cannot show

- It is smaller in every dimension than the E26 world; a result here is a claim only when the winning setting is repeated on the E26 families at full scale.
- No finite life excludes overtaking beyond its horizon. Every statement will carry its horizon and the trend at it.
- Answer bytes are the only targets. Every-byte targets enter later as a candidate, after E34.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P39.1 | The reviewer's second implementation of the generator gives the three digests of check 1 at the first attempt | 75% |
| P39.2 | The reviewer's gate as written (first group at 90% by the end of its first visit, six of eight worlds) passes at the default size | 10% |
| P39.3 | At the default size a life of 1,024 visits is forecast within 6 CPU-hours | 75% |
| P39.4 | At the default size the first group is at 90% by the end of its second visit in six of eight worlds, under `naive` | 30% |
| P39.5 | Some rung of the ladder passes within the 6-hour bound | 70% |
| P39.6 | In L1, tasks met later in the 16 visits cost fewer taught items than the first two, in six of eight worlds under `naive` | 65% |
| P39.7 | `naive` and `sweep` differ by less than 10 points in mean score over the 16 visits, in six of eight worlds | 60% |

## Done

L0: `smallworld/` with the world file, network, runner and scorer; the check receipts. L1: sixteen lives (and any ladder rungs) with receipts and event files, and `E39_REPORT.md` with the figures listed under L1, which rung is adopted, and the outcome of each prediction. No readings beyond these.
