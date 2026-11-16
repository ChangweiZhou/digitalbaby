# E39 (jobs L0 and L1): the small world — a protocol for long lives, and its calibration (15 November 2026, revision 4 of 16 November)

Evidence level: none for L0 (it is an instrument); E2 for anything L1 measures, and a result in the small world is a claim only after it is repeated at full scale. The programme is the outside reviewer's (its memo, revision 3, sections 17 and 18, on `claude/vibrant-cerf-sje726` at 823a3884), written after the user's statements of 15 November. The specification below is mine. Revision 1 (18751457) had its one audit (`AUDIT_E39_SMALL_WORLD_PROTOCOL_20261115.md` on `claude/vibrant-cerf-sje726`), which required eight changes and stated that after them no re-audit is needed; check 1 with the audited digests stands in for one. Revision 2 makes them; the last section lists them. **Released.** **Revisions 3 and 4 add jobs L1a and L1b at the end, after L1's gate failed at every rung. Revision 3 was audited (`AUDIT_E39_REVISION3_BIRTH_PHASE_20261116.md` on `claude/vibrant-cerf-sje726` at 7b59ae8b): six required changes, then release without a re-audit. Revision 4 makes them. L1a and L1b are released. Nothing above them changes.** One worker. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106; K=PROJECT_DOCUMENTS/COORD_CHECKS_20261115`.

## The question the protocol must be able to answer

The user, 15 November: the goal is persistent learning at fixed memory and computation, and a learner that is slower at first may overtake a conventional one later; as long as ours is still learning and the conventional one is stuck, this cannot be excluded, and many short lives cannot show it.

So the protocol records trends over a long life, not an average over a short one: score against age, the cost of learning the n-th new task, and what is kept when a task returns. Its first use (job L2, not in this brief) needs no candidate: do naive backpropagation and backpropagation with replay get stuck over a life of 1,024 visits?

Nothing measured so far can answer this: fixed panels after a fixed dose, with backpropagation as the ceiling, allow parity at best, and our longest lives have six to nine stages.

## The world, as code

**`$K/sw_world_spec.py` on the default branch is the specification** (pure Python, no torch; run it to see a world). Copy it to `$M/smallworld/sw_world.py` and import it; do not rewrite it. The reviewer's second implementation (`e39_world.py` on its branch) reproduces the digests in check 1 from the rules below.

- **Tasks.** A task is an operation and a modulus with a two-byte code of its own. 18 operations: the 16 linear ones `(c1·a + c2·b) mod p` with c1, c2 in {1, 2, −1, −2} (among them add and sub), and max and min. 30 moduli, 8 to 37. That gives 540 tasks. Each operation has one symbol byte, the same under every modulus; codes are arbitrary and say nothing about the modulus.
- **Items.** `code(2) a(2) symbol d_1..d_k b(2) =` then the answer: two digits and the end symbol. k is 0 to 4 letters. The longest prompt is 12 bytes.
- **Taught and asked pairs.** Each task's p² pairs are split once in half. Taught pairs come with their answer. Asked pairs are never taught.
- **Groups and visits.** Tasks are met in groups of 2. A visit is 256 updates on one group. Visit 0 is group 0. After that a visit brings the next new group with chance 0.2; otherwise it returns to a met group, drawn so that short (1 to 4 visits), medium (5 to 32) and long (33 or more) absences are about equally common.
- **One stream, read once.** Every update takes 64 items from the stream: 48 taught and 16 asked, each from a task of the visit's group chosen at random.
- **Everything follows from the world's seed**, through `random.Random` objects seeded with strings (`f"world|{seed}|tasks"` and so on, as the file writes them). The schedule and the items are the same for every arm.

**The rules, in full.** With S the world's seed:

- *Tasks.* The pairs (operation index, modulus), operation outer and modulus inner, are shuffled by `Random(f"world|{S}|tasks")`. A task's id is its place after the shuffle. Operations are numbered: the 16 linear ones with c1 outer and c2 inner over (1, 2, −1, −2), then max, then min.
- *Codes.* The 676 two-lower-case-letter codes in order, first letter outer, are shuffled by the same generator after the tasks. Task i takes code i.
- *Symbols.* The bytes of `+-*/%&|^~!?:;<>@$#` belong to operations 0 to 17 in order. So `+` is a + b, `*` is a − b, `-` is a + 2b, `<` and `>` are linear, `$` is max and `#` is min. They do not mean what they mean in E26.
- *Split.* A task's pairs, a outer and b inner, are shuffled by `Random(f"world|{S}|split|{id}")`. The first `round(p*p/2)` are taught, which is the smaller half when p is odd.
- *Groups.* Tasks 2g and 2g + 1 form group g. Groups are met in this order.
- *Schedule.* One generator, `Random(f"world|{S}|schedule")`. At every visit after the first, one `random()` is drawn; the visit is new if it is below the new-group chance and an unmet, unreserved group remains. Otherwise `randrange(3)` draws the absence class (visits since the group's last visit: 1 to 4, 5 to 32, 33 or more); if that class is empty, the first non-empty of short, medium, long is used; the group is `sorted(class)[randrange(len(class))]`.
- *Items.* Each visit v has its own generator, `Random(f"world|{S}|visit|{v}")`. Each update draws 64 items in order, the first 48 taught: task `group[randrange(2)]`, pair `pool[randrange(len(pool))]`, then `randint(0, 4)` for k, then k letters by `choice`.
- *`needs_wrap`.* For a linear operation, the plain integer c1·a + c2·b is outside 0 to p − 1. Never for max and min.
- *Digest.* SHA-256 over every item in order, taught then asked: the bytes `[id // 256, id % 256, a, b, len(prompt)]`, the prompt, and each answer value mod 256.
- *Symbols for start and end.* `EOS = 256` ends every answer and `BOS = 257` is never a target, as `mod97_local.py` lines 34-35.
- **Probe tasks.** The last 20 groups of each world (tasks 500 to 539) are never scheduled: a visit is new only while fewer than 250 groups are met. They are that world's probe tasks. No schedule or digest of a 1,024-visit life changes by this (over 200 worlds the most groups met is 248, the reviewer's count).
- **Prefixes.** A life of V visits is the first V visits of any longer life in the same world and arm, because the schedule is drawn in sequence and each visit's items have their own generator.
- The visit length and the new-group chance are arguments of the world's constructor.

**Properties to know (the audit's).**

- Coefficient 2 at an even modulus halves the answers that occur for the four operations with both coefficients ±2. These are properties of the tasks.
- `needs_wrap` is nearly constant for half the linear operations (98 to 100% of pairs when both coefficients are negative; 44 to 49% for a + b and a − b), so the split by wrap is informative for some operations only.
- **A shortcut.** Six of the 18 operations are symmetric (c1 = c2, max, min). For those, about half of the asked pairs have their mirrored pair (b, a) among the taught pairs of the same task, with the same answer, and can be answered by recall plus commutativity. E26's split has the same property. The event file marks these items (below).
- **What a task receives.** A visit gives each task 6,144 taught items in 256 updates. Over a life a group gets about 5 visits (median 3 to 4), and about a fifth get one visit only; groups first met in each quarter of a 1,024-visit life get 9.3, 4.8, 3.5 and 2.3 visits on average. In showings of each taught pair the small world is generous (192 per visit at p = 8, 9 at p = 37, against 16 to 33 for a full-scale birth); in updates it is not (a median of 768 to 1,024 over a task's whole life, against 4,096 to 8,192).
- **The learner's own time constant.** Decay shrinks a weight the stream does not push on with a constant of 1 / (learning rate × decay): 4,000 updates at the defaults, 15.6 visits. A short absence is 0.06 to 0.26 constants, a medium one 0.32 to 2.0, a long one 2.1 or more. A roster that changes the learning rate, the decay or the visit length restates these.

What the generator gives (`$K/OUTPUT_SW_WORLD_MAC.txt`): in 1,024 visits, worlds 1, 2 and 3 meet 189, 211 and 192 new groups of the 270 available, and returns fall 273/279/283, 289/264/260 and 309/260/263 into the three absence classes, the longest absence 758 to 922 visits.

## The learner

A GRU of E26's form, narrower. Write `sw_net.py` with a class holding the same parameter names as `arch20.NetRNN("gru")` (`git show $D:$M/arch20.py | sed -n '36,50p'`): `book = Embedding(258, 8, padding_idx=257)` (BOS, as `arch20.py` line 43), `l1 = Linear(8, H)`, `rnn = GRU(H, U)`, `lm = Linear(U, H)`, `l2 = Linear(H, H)`, `l3 = Linear(H, 257)`, with `kind = "gru"` and `units = U`, so that `arch26.forward(m, x, local, True)` runs on it unchanged. **Default size H = U = 64.** Initial values: `T.manual_seed(world_seed)` then the constructor.

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

**The buffer.** It holds raw items. Its capacity in items is `(4 × P) // 15`, with P the number of parameters, embedding table included, and 15 bytes the longest item: 14,033 items at the default size (P = 52,625). In each update the batch's replay items are drawn first, by `randrange(len(buffer))` with replacement from `Random(f"world|{S}|replay")`; then the 48 new taught items are offered in batch order from the same generator. The n-th item offered is stored while the buffer is not full; after that `j = randrange(n)` is drawn and the item replaces slot j if j is below the capacity. Replay starts at the first update in which the buffer holds ρ·48 items; until then the arm learns from all 48 new items. ρ is an argument (0.25 and 0.5 for a start). Every arm consumes the same 64 stream items per update.

Later candidates (an anchor, a slow copy, a step size per weight, reminders, the confidence gain) are added as arms by a one-page roster note; the runner's update is the only place they touch.

## What a life writes

One receipt (JSON) and one event file per life, in `$M/exp45/e39/`, named `sw_<arm>_w<world>_v<visits>[_<setting tag>]`; the runner refuses an existing path. **Complete state** is saved at visits 1, 2, 4, 8, … and at the end: weights, optimizer state, the buffer and the replay generator's state, and the per-task counts the event file needs.

- `--resume <state>` continues a life exactly.
- `--branch <state> --groups g1,g2,…` runs the named groups, probe groups included, from that state in place of the schedule, with items from `Random(f"world|{S}|branch|{visit of the state}|{k}")` for its k-th visit, writing its own receipt and event file and leaving the trunk's files untouched.
- `--ask <state>` answers a given list of asked items at that state without learning.

**What is committed.** Receipts, the scorer's tables, and compressed event files and states for world 1 of each arm; hashes for the rest. A 1,024-visit event file has 4.2 million entries and the container's disk is small.

**The event file** is one compressed `.npz` with arrays of equal length, one entry per asked item in stream order: `update` (uint32), `visit` (uint16), `task` (uint16), `correct`, `blind`, `wrap`, `mirror` (uint8), `seen` (uint32: the task's stream taught items presented before this update, learned from or not, so that it is the same for every arm), and `replayed` (uint32: the task's replayed items before this update; zero without replay). `mirror` is 1 if the operation is symmetric, a ≠ b and (b, a) is a taught pair of the task; else 0.

**The blind predictor** answers an asked item with the answer of the most recent taught item of the same task presented before this update, without reading the operands; it is wrong if there is none.

**The receipt** has the settings, provenance and source hashes as E26's receipts do, the schedule, a digest of all batches, and the ledger: stored numbers by kind (weights, optimizer state, buffer bytes, any anchor), updates, and stream items per update.

## The scorer: quantities per life, computed from the event file alone

`sw_score.py`, reading no weights.

Throughout: a mean of the last 32 asked items exists only once the task has 32; tasks are numbered by first visit, then id; "taught items" is `seen`. The scorer takes the number of visits to read as an argument.

| Quantity | Definition |
|---|---|
| Score by age | Share of asked items correct in each visit; the blind predictor's share beside it. |
| First-visit level | The mean of the task's last 32 asked items in its first visit. |
| First-visit cost | Its `seen` at the first asked item of that visit at which the mean of its last 32 is at least 0.8; "not reached in the first visit" otherwise. |
| **Trend of learning (the one compared)** | Least-squares slope of first-visit level against log(n), n from 3. Every task has one whole first visit, so none is left out. |
| Cost of the n-th new task | Its `seen` before the first asked item at which the mean of its last 32 asked items is at least 0.8, over the whole life; "not reached" otherwise, with the mean of its last 32. Reported, not used in any comparison. |
| Trend of the cost | Least-squares slope of log(cost) against log(n), over tasks that reached, n from 2. Reported, not used in any comparison: a task met late has fewer visits left, so among late tasks only the cheap ones reach, which pushes this slope negative for any learner. |
| Kept at return | The share right among the task's asked items in the visit's first update (answered before any update of the visit), pooled over returns in each absence class, and split by wrap; the first-32 version beside it. |
| Relearning at return | `seen` within the visit at the first asked item at which the mean of the visit's own last 32 is at least 0.8. |
| Symmetric operations | Score on asked items of symmetric operations, split by `mirror`. |
| Late level | Mean score by age over the last quarter of the visits. |
| Trend of the score | Least-squares slope of score by age against the natural logarithm of (visit + 1), over the second half of the visits. |

No marks in points are fixed in this brief. The reviewer's two withdrawn marks came from fixing thresholds before seeing the scale of the numbers; L1 shows the scale first.

## Checks (job L0)

1. **The generator.** `sw_world.py` is byte-identical to `$K/sw_world_spec.py`. Its digests are these (the audit's, which the corrected file gives on the Mac as well, `$K/OUTPUT_SW_WORLD_MAC.txt`). If the file gives them, the generator is the one audited.

   | World | First 16 visits | First 64 visits |
   |---|---|---|
   | 1 | 05e48881… | ef5afac9… |
   | 2 | 3d7ba088… | 983ff313… |
   | 3 | ea204b19… | dfc3270f… |

   Both columns are needed: the first 16 visits of world 3 are all group 0.
2. **Asked is never taught.** For every task, the taught and asked pair sets are disjoint (the file asserts it), and the runner asserts that no item it learns from has an asked pair.
3. **Asking does not teach.** The `tensors_sha256` of the weights and of the optimizer state is unchanged across the answering of a batch's asked items.
4. **Same stream for every arm.** The batch digest of a 4-visit life is equal across the four arms.
5. **The forward and the two rules.** On the first batch, the network's logits equal those of `torch.nn.GRU` run on the same parameters to 1e-6; under `local=False` the parameter gradients equal autograd's through that reference to 1e-5 of each tensor's largest entry; under `local=True` they differ from them.
6. **Replay.** With ρ = 0, `replay` gives the same weights as `naive` after 64 updates. The buffer never exceeds its capacity in items (14,033 at the default size). With ρ = 0.5, 24 items of each batch come from the buffer once it holds that many.
7. **The scorer.** On a hand-made event file of two tasks with known answers, every quantity in the table equals its hand-computed value. The reviewer's numpy scorer and this one agree on the event file of one 16-visit life, and that event file equals the reviewer's rebuild from the generator in every column but `correct` (`e39_world.py spec events <file.npz> <world>`).
8. **Cost.** CPU-seconds per visit at the default size, for `naive` and `sweep`, and a forecast for a life of 1,024 visits.

9. **Resume.** A `naive` life and a `replay` life (ρ = 0.5) of 8 visits, stopped at the state of visit 4 and resumed, give the weights hash, the optimizer-state hash and the event file of the uninterrupted lives. A life run twice gives the same hashes.

Report L0's checks before L1 starts; do not wait for me unless one fails.

## Job L1: calibration (runs after L0's checks pass; no further audit)

Worlds 1 to 8, lives of 64 visits, arms `naive` and `sweep`, default settings. Sixteen lives, together the cost of one life of 1,024 visits.

**The gate (the reviewer's, revised in its audit).** At a rung, under `naive`, in at least six of eight worlds: (1) both tasks of the first group reach the 0.8 criterion within the life; (2) the median first-visit level of the tasks met after the first group is between 0.10 and 0.90; and the measured cost puts a life of 1,024 visits within 6 CPU-hours. The two bounds of (2) are not marks on a learner: they say that the instrument is off its floor and its ceiling.

**The ladder.** If (1) fails or the median of (2) is below 0.10: visits of 512 updates; then H = U = 96 with visits of 512. If the median of (2) is above 0.90: visits of 128 updates. Each rung's digests are recorded. Stop at the first rung that passes and report every rung tried.

**L1 reports,** for each arm and world: every task's first-visit level and first-visit cost; the first group's score at the end of each of its visits; score by age and the blind predictor's; kept at return on the visit's first update, by absence class and by wrap; the score on symmetric operations by `mirror`; CPU-seconds per visit. It also reports, to set beside the full-scale figures of E36 to E38: how fast a task is lost when the stream leaves it, how fast it is restored at return, and the first-visit level of late tasks against the first.

L1's lives at the adopted rung are the start of L2's and are not run again.

## Not in this brief

- **L2**, whether the conventional learner gets stuck (`naive` and `replay` at 64, 256 and 1,024 visits; the sweep arms at 64 and 256), comes as a one-page roster note with predictions on file, after L1 fixes the size. It needs no audit; any change to the generator, runner or scorer does. Each arm is run once at its longest length and the shorter lengths are read from the same event file. L2 also runs `naive` at a second width and reports the agreement between lengths and the two probes (memo 17.10). At least eight worlds per setting up to 256 visits, paired differences with intervals, and the reviewer spot-checks saved lives with its numpy scorer.
- **Candidates** come after L2.
- **Comparison rules** (late level, trend, the horizon rule of lengths growing by four, intervals across worlds) are the memo's sections 17.4 and 17.5 and are fixed in the L2 note.

## What the small world cannot show

- It is smaller in every dimension than the E26 world; a result here is a claim only when the winning setting is repeated on the E26 families at full scale.
- No finite life excludes overtaking beyond its horizon. Every statement will carry its horizon and the trend at it.
- **Supply of tasks.** 250 groups last about 1,250 visits at a new-group chance of 0.2. A longer life needs a larger task space, which is a change to the generator and so an audit.
- Answer bytes are the only targets. Every-byte targets enter later as a candidate, after E34.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P39.1 | The reviewer's second implementation of the generator gives the three digests of check 1 at the first attempt | 75% | 

P39.1 is judged: **false from the text alone** (the text of revision 1 left ten points open, and my file had the start and end symbols swapped), true once the file's rules were written out. P39.2 and P39.4 were on the old gate and are withdrawn unjudged; P39.8 and P39.9 replace them.

| | Prediction | Chance |
|---|---|---|
| P39.3 | At the default size a life of 1,024 visits is forecast within 6 CPU-hours | 75% |
| P39.5 | Some rung of the ladder passes within the 6-hour bound | 70% |
| P39.8 | The revised gate passes at the default size | 35% |
| P39.9 | At the default size, part (1) holds: both tasks of the first group reach 0.8 within 64 visits in six of eight worlds | 60% |
| P39.6 | In L1, the mean first-visit level of tasks met after the first group is above the first group's, in six of eight worlds under `naive` | 65% |
| P39.7 | `naive` and `sweep` differ by less than 10 points in mean score over the 64 visits, in six of eight worlds | 60% |

## Done

L0: `smallworld/` with the world file, network, runner and scorer; the check receipts. L1: sixteen lives of 64 visits (and any ladder rungs) with receipts and event files, and `E39_REPORT.md` with the figures listed under L1, which rung is adopted, and the outcome of each prediction. No readings beyond these.

## What changed from revision 1 (the audit)

Made as the audit wrote them. I reran the corrected generator: it gives all six of the audit's digests.

| | Change |
|---|---|
| R1 | **My error:** the specification file had the start and end symbols the other way round from the harness, so every answer ended in a target the network cannot emit and the first update would have failed. Corrected in the file and in `padding_idx`; new digests in check 1. I wrote the constants from memory and did not open `mod97_local.py`. |
| R2 | The generator's rules written out in the text; ten points were open. |
| R3 | The `mirror` value and its score: a recall-plus-commutativity shortcut on symmetric operations. |
| R4 | The gate replaced (the audit agrees the first-visit 90% gate could not pass, and that lives of 16 visits meet too few groups); lives of 64 visits; a new ladder. My rung "moduli 8 to 19 at a new-group chance of 0.1" is dropped: it would have run out of groups in a fifth of worlds. |
| R5 | The replay buffer made deterministic: capacity in items, order of drawing and offering, when replay starts. |
| R6 | The event file's layout; `seen` counted the same for every arm; first-visit level and cost; the trend of learning replaces the trend of the cost as the compared quantity, which was biased by construction; kept at return read on the visit's first update. |
| R7 | Probe tasks; complete state at powers of two; resume, branch and ask; check 9; prefixes; the learner's time constants. |
| — | Recommended: pace lines, the supply of tasks, what is committed, the symbols' meanings. |

## Note (15 November 2026, after release; no change to L0 or L1)

The sentence under "Arms the runner must offer" that lists later candidates (an anchor, a slow copy, a step size per weight, reminders, the confidence gain) is superseded by the reviewer's memo, revision 7, section 20, following the user's statement that earlier slow-learner and long-term-storage schemes gave at best mixed results. Retention is not a research question in this programme for now: the only store is the plain bounded buffer of the `replay` arms. A mechanism is added only if the deficit it addresses has been measured on a learner that generalises, the plain baseline is in the same experiment at equal budget, and its expected effect is stated beforehand from measured rates. The headline of every report is generalisation and its trend with experience (first-visit level and the trend of learning); what is kept at return is reported beside it as a constraint. Nothing the worker builds changes.

---

# Revisions 3 and 4 (16 November 2026): L1 read; a measurement and a birth phase (jobs L1a and L1b)

**Released as revision 4.** The recommendation is the reviewer's (16 November): before any other change, a birth phase in the small world (joint training on the first few tasks until they reach the criterion, then the life as specified), preceded by a measurement of how many updates the small network needs to learn one linear task alone. The generator does not change. The specification is mine. The same worker, whose branch `claude/gallant-edison-p0qa68` (42fe1a24) is merged.

## L0 and L1 as I read them

- **L0 passes all nine checks**; the generator gives the audited digests and the reviewer's implementation agrees with the worker's scorer.
- **L1's gate fails at all three rungs, always low**: 0, 0 and 0 worlds of eight pass both parts (64 units and 256 updates a visit; 512 updates; 96 units and 512 updates). P39.3 holds; P39.5, P39.8 and P39.9 miss.
- **The floor is the linear operations', not the instrument's.** Max and min are learned within their first visit (first-visit level 0.55, 0.76, 0.80 at the three rungs). The 16 linear operations, 192 of the 212 tasks met, stay near chance: 1, 20 and 30 of 192 reach 0.8 within the life under `naive`; 1, 4 and 2 under the sweep.
- **The first group starts from nothing.** A life's first visit is a birth of 256 or 512 updates. The one first group that reached the criterion (world 7 at 96 units) did so after 171,379 and 187,283 taught items per task, about 7,100 to 7,800 updates spent on that group.
- **Cost is not the obstacle:** 1.0 to 2.6 CPU-hours for a life of 1,024 visits, against a bound of 6.
- I recomputed none of L1's tables; they are the worker's, made with a scorer the reviewer's agrees with.

At full scale the same shape is on file: a fresh network needs thousands of updates for its first family, and a lived one takes a new modulus within 64 to 128 updates (E36, E40). So the small world as run asks every task to be a first task. The addition below gives the learner a birth, and first measures what a birth costs here.

## The criterion, in both jobs (the audit's change 1)

A task is **at the criterion** at a check, made every 64 updates, if the share right over all of its asked items in the last 64 updates is at least 0.8 (1,024 items for a task alone, about 256 a task in a birth). L1a's updates to criterion and the end of a birth both use it.

The 32-item window I wrote in revision 3 is met by chance well below 0.8: examined at every asked item of a solo run it is passed at a true level of about 0.55 to 0.6, and at a birth's checks at about 0.65 to 0.7 (the audit's table). It stays in L1a's table as a second column, for comparison with L1, and decides nothing.

## Job L1a: how many updates does the small network need for one linear task alone?

**Runs.** One task alone, from the network's initial values, 16,384 updates. Six tasks of world 1 and six of world 2, chosen by rule so that nothing is picked by eye: in each of three bands of modulus (8 to 13, 18 to 24, 30 to 37), the task of lowest id whose operation is linear with c1 ≠ c2, and the task of lowest id whose operation is linear with c1 = c2. Sizes H = U = 64 and 96. Arms `naive` and `sweep`. Default learning rate and decay. 48 runs.

The rule gives (the audit's list for world 1; the worker prints both worlds'):

| Band | World 1, c1 ≠ c2 | World 1, c1 = c2 | World 2 |
|---|---|---|---|
| 8 to 13 | task 20, −a + 2b mod 9, 40 taught pairs | task 2, a + b mod 8, 32 taught pairs | among tasks 0, 4, 6, 7, 13, 24 |
| 18 to 24 | task 5, a − b mod 23, 264 | task 1, 2a + 2b mod 19, 180 | |
| 30 to 37 | task 3, −a − 2b mod 33, 544 | task 12, −a − b mod 37, 684 | |

All runs of a world at a size start from the same initial values (`T.manual_seed(S)`), so they differ in the task only. World 2 is a second set of initial values (the audit's recommendation): with one world the choice of size would rest on four tasks and one draw of the weights.

**The solo stream, in the runner (the generator file does not change).** Each update draws 64 items in order, the first 48 taught, by the item rules of `visit_batches` with the single task in place of the group and **one generator for all the run's updates**, `Random(f"world|{S}|solo|{task id}")`: pair `pool[randrange(len(pool))]`, then `randint(0, 4)` for k, then k letters by `choice`. **No draw is made for the task**, since there is one; a runner that calls `visit_batches` with a one-task group draws `randrange(1)` and produces other items. In a solo event file the `visit` column is 0 throughout. Asked items are answered before the update, as in a life. Initial values: `T.manual_seed(S)` then the constructor, as a life's.

**Recorded**, from an event file of the life's layout: the update of the first check at which the task is at the criterion (**updates to criterion**), or "not reached"; the same for a share of 0.5; in a second column, `seen` and the update at the first asked item at which the mean of the last 32 asked items is at least 0.8; and, for every run whether it reached or not, the share right over the last 256 asked items, on wrap and no-wrap items separately, and the mean training loss over the last 64 updates, which separates a network that has memorised the taught pairs from one that has learned nothing.

**What is tabled:** per size and arm, each task's updates to criterion with its count of taught pairs.

**Cost:** from L1's measured 3.5 to 9.9 CPU-seconds per 256 or 512 updates, each run is 4 to 6 CPU-minutes; about 3.5 CPU-hours for the 48.

## Job L1b: the life with a birth

**The setting is chosen from L1a by a rule fixed now** (the audit's change 2). A size qualifies if, under `naive`, at least six of the eight tasks of the two upper bands (world 1's tasks 1, 3, 5 and 12 and world 2's four) reach the criterion within 8,192 updates. Use the smaller qualifying size, with visits of 512 updates. If neither qualifies, stop and report: the task space or the learner's settings are the next thing to change, which is a decision for the reviewer and me and not a rung.

The tasks of the band 8 to 13 (32 and 40 taught pairs in world 1) are run and tabled and are not counted. It is likely, though not shown, that a fresh network cannot learn a linear task of small modulus alone to the criterion on untaught pairs, for want of examples: in L1 2 of 38 such tasks reached it, and at full scale a fresh network given a tenth of the pairs memorises them. A median over all six, which revision 3 used, would also have let a size qualify with half its tasks beyond what a birth allows.

**The birth.** Before visit 0, the learner is trained jointly on the tasks of groups 0 and 1 (four tasks).

- **Stream, in the runner:** each update draws 64 items in order, the first 48 taught, with **one generator for the whole birth**, `Random(f"world|{S}|birth")`: task `tasks[randrange(4)]` with `tasks` the four in id order, then pair, k and letters as in a visit. Asked items are answered before the update.
- **End of the birth** (the audit's change 3): checked after every 64 updates; the birth is complete at the first check at which at least three of its four tasks are at the criterion. The share right of all four is recorded at every check and at the end. **Cap: 32,768 updates.** A birth that reaches the cap is recorded as not complete and its life still runs. (Four of the eight worlds have a birth task of modulus 13 or less, which may never be learned; with all four required those births would run to the cap whatever else happened.)
- **Three arms.** `naive`: born by backpropagation, lives by it. `sweep`: born by the sweep, lives by it. **`sweep-after-naive` (the reviewer's request):** the sweep's life starting from the `naive` arm's state at the end of its birth, optimizer state included. The project's constraint allows any optimizer before life, and the brief expects the sweep's own birth often not to complete; without this arm the sweep would then have no life that can be read.
- **Then the life as specified,** from visit 0 with the unchanged schedule and items. The schedule's visits to groups 0 and 1 are returns for the scorer.

**The scorer** takes `--birth-groups 0,1` (the audit's change 4). The four birth tasks are left out of the first-visit quantities and of the trend of learning. **The trend of learning is fitted over every task that is not a birth task, from n = 1**, with tasks numbered from the first that is not a birth task. **The birth counts as a visit before visit 0**: a birth task's first visit in the life has an absence of its visit number plus one, so visit 0 of group 0 has absence 1. Every `seen` count includes the birth's items for those tasks. The event file gains the birth's asked items with `visit` = 65,535. The reviewer's scorer implements both definitions (`score(..., birth_tasks=...)` in `e39_world.py`) with a hand-computed fixture of two birth tasks and two later ones (`e39_world.py spec scorer`), which may be used for check 13.

**The ledger** gives the birth's updates and items beside the life's; computation to any threshold includes them.

**Lives.** Worlds 1 to 8, 64 visits, three arms: 24 lives, sixteen of them with a birth of their own. With births at the cap the sixteen are at most about 4.0 CPU-hours at 64 units and 5.4 at 96 (the audit's figures from L1's measured rates); the third arm adds 1 to 1.3.

**The gate (this rung's).** Under `naive`, in at least six of eight worlds: (1) the birth is complete within the cap, by the definition above; (2) the median first-visit level of the tasks met after the birth groups is between 0.10 and 0.90; and a life of 1,024 visits with its birth is forecast within 6 CPU-hours. If it passes, this setting with its birth is the protocol's size and L2 follows by roster note.

**One further step for each way it can fail, named now so that neither ends in a new round** (the audit's recommendation). If the median of (2) is above 0.90: the same lives with visits of 256 updates. If it is below 0.10 with births complete: a birth on groups 0 to 3, complete when six of its eight tasks are at the criterion, with a cap of 65,536 updates. If births are not complete in six worlds: stop and report. After the one further step, stop and report whatever it gives.

Comparisons between the arms in L1b are descriptions; the gate is `naive`'s alone.

**Reported beside the gate:** per world, the visit at which the first later group is met (visit 6 to 29; the birth groups take 14 to 43 of the 64 visits, and the later tasks number 12 to 32 a world) and the birth tasks' levels at their last visit before it; everything L1 reports, with first-visit level by kind of operation (linear with c1 ≠ c2, linear with c1 = c2, max, min) as the worker's report already gives; the birth's updates per world and arm; and the first-visit level of linear tasks here beside L1's at the same size and visit length (0.12 at both higher rungs under `naive`).

## Checks added

10. **Solo and birth streams.** The digest is the world digest's rule (the bytes `[id // 256, id % 256, a, b, len(prompt)]`, the prompt, each answer value mod 256) over each update's taught then asked items, for the first 256 updates. The runner's digests equal the reviewer's for all six solo streams of world 1 and the births of worlds 1, 2 and 3 (`python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e39_world.py spec streams` on `claude/vibrant-cerf-sje726`):

    | Stream | SHA-256 |
    |---|---|
    | solo, world 1, task 1 | 1fd661d4df3adec6… |
    | solo, world 1, task 2 | 84af069bc95624b1… |
    | solo, world 1, task 3 | 435f5c4638f01a12… |
    | solo, world 1, task 5 | 81271cee73bc7a2e… |
    | solo, world 1, task 12 | 69fc5bf96bedb841… |
    | solo, world 1, task 20 | 3de3b1eda53ced30… |
    | birth, world 1 | 8ac9eb07e8e1beb2… |
    | birth, world 2 | db31da0c84814890… |
    | birth, world 3 | 1e012e3c01a63ec1… |
11. **The life after a birth is the life.** The stream digest of the 64 visits after a birth equals `sw_world.digest` of that world at that visit length, as in L1.
12. **Birth and resume.** A life with a birth, stopped at the state of visit 4 and resumed, gives the hashes and event file of the uninterrupted one.
13. **The scorer with birth groups.** On a hand-made event file with two birth tasks and two later tasks, every quantity equals its hand-computed value; without `--birth-groups` the scorer's output on an L1 event file is unchanged.
14. **The new quantities, by a second implementation.** The worker commits the event files of the 48 solo runs and of world 1's births; the reviewer recomputes the updates to criterion and the update at which each birth ends. This check does not hold the runs.
15. **The third arm's start.** The `sweep-after-naive` life's state at visit 0 has the weights hash and optimizer-state hash of the `naive` arm's state at the end of its birth.

## What this cannot show

- That a birth is the right remedy and not, say, another learning rate; only the birth is tried, as the reviewer recommends.
- Anything about long lives; that is L2.
- With a birth the first tasks are learned jointly and at leisure. What the trend of learning then measures starts from a learner that already has two linear operations and two moduli or more, which is the condition the full-scale results describe.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P39.10 | L1a: 96 units qualifies (six of the eight upper-band tasks at the criterion within 8,192 updates under `naive`); restated on the audit's rule | 45% |
| P39.11 | L1a: 64 units qualifies | 25% |
| P39.12 | L1a: under the sweep at 96 units, at least four of the eight upper-band tasks reach the criterion within 16,384 updates | 25% |
| P39.13 | L1a, world 1, `naive`, 96 units: the task with c1 = c2 reaches the criterion in fewer updates than the one with c1 ≠ c2 in the same band, in at least two of three bands; a band in which neither reaches does not count as fewer | 50% |
| P39.18 | L1a: no task of the band 8 to 13 reaches the criterion under `naive` at either size, in either world | 55% |
| P39.14 | L1b: the birth is complete (three of four tasks) within the cap in at least six of eight worlds under `naive` | 60% |
| P39.15 | L1b: the gate passes | 35% |
| P39.16 | L1b: the mean first-visit level of linear tasks met after the birth is at least twice L1's figure at the same size and visit length, under `naive` | 55% |
| P39.17 | L1b: the sweep's own birth is complete within the cap in at least six of eight worlds | 15% |
| P39.19 | L1b: `sweep-after-naive`'s mean first-visit level of later tasks is within 0.10 of `naive`'s, in six of eight worlds | 40% |

## Done

L1a: the 48 runs with receipts and the table. L1b: checks 10 to 15, 24 lives, and an addendum to `E39_REPORT.md` with the gate, the listed figures and the outcome of P39.10 to P39.19. No readings beyond these.

## What changed from revision 3 (the audit)

| | Change | Required? |
|---|---|---|
| 1 | One criterion in both jobs: the share right over all of a task's asked items in the last 64 updates, checked every 64 updates. My 32-item window was passed by chance at a true level of 0.55 to 0.7, differently in the two jobs. | yes |
| 2 | The size is set by "six of the eight upper-band tasks within 8,192 updates", not by a median over six; the small-modulus tasks are tabled and not counted | yes |
| 3 | A birth is complete when three of its four tasks are at the criterion | yes |
| 4 | Two scorer definitions: the birth is a visit before visit 0; the trend of learning is fitted from n = 1 over tasks that are not birth tasks | yes |
| 5 | The two streams: one generator each; what the digest hashes; all six solo streams and three births against the reviewer's digests; the `visit` column of a solo file; check 14 | yes |
| 6 | L1b's cost at 96 units is 5.4 CPU-hours, not 4 | yes |
| — | The third arm, `sweep-after-naive`, as the reviewer asked; a second world for L1a; one named further step per failure of the gate; the added report items; P39.13's wording | recommended |
