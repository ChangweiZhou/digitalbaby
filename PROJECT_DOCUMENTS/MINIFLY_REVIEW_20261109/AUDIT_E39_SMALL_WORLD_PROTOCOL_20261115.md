# Audit of the E39 brief (small-world protocol, jobs L0 and L1) at 18751457 (15 November 2026)

From the outside reviewer. Read-only: nothing was trained and no code from another branch was executed.

**Verdict.** The design is the memo's and the generator is sound: my second implementation reproduces its three digests and every schedule figure. The brief cannot go as it stands. The specification file has the start and end symbols the other way round from the harness, so the first update would fail; the brief's text is not enough to rebuild the generator; one scorer quantity is biased by construction; and the brief predates section 17.10 of the memo. On the gate the coordinator is right, and I replace it. Eight changes are required, all worded to be applied as written. After them E39 can be released without a re-audit.

Shell names: `D` and `M` as in the brief; `K=PROJECT_DOCUMENTS/COORD_CHECKS_20261115`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`; `C=$R/e39_world.py` on `claude/vibrant-cerf-sje726`. `python3 -I $C <mode> <part>` runs one part; mode `text` is my reading of the brief's text alone, `spec` the coordinator's rules written again by me, `spec-file` the same with the file's end symbol.

## 1. The second implementation

- I wrote the generator from the brief's text before opening `sw_world_spec.py` (commit b9d6cb02). It gave the right kind of world and not the file's numbers: 183, 211 and 195 new groups against 189, 211 and 192, and different digests. `python3 -I $C text stats`
- The text is silent on ten points that the digests depend on (section 3). After reading the file I wrote its rules again in my own code. That version gives all three digests of check 1 and every figure of `OUTPUT_SW_WORLD_MAC.txt`, on Python 3.13.17 against the Mac's 3.11.5. `python3 -I $C spec-file stats`

| World | New groups in 1,024 visits | Returns short / medium / long | Longest absence | Digest of the first 16 visits |
|---|---|---|---|---|
| 1 | 189 | 273 / 279 / 283 | 758 | 64486ccd… equal |
| 2 | 211 | 289 / 264 / 260 | 777 | cad4f54b… equal |
| 3 | 192 | 309 / 260 / 263 | 922 | bd6ea889… equal |

- For P39.1: from the text alone, no; with the file's rules, yes at the first attempt.

## 2. Start and end symbols are swapped in the specification file

- `sw_world_spec.py` line 8 has `BOS, EOS = 256, 257`. The harness has `EOS = 256` and `BOS = 257` (`git show $D:$M/mod97_local.py | sed -n '34,35p'`).
- Every answer the file renders therefore ends in 257. `l3` has 257 outputs, numbered 0 to 256, and `exp45.batch_tensor` puts the whole answer under the mask (`git show $D:$M/exp45.py | sed -n '158,169p'`). The first update would fail on a target out of range, and no greedy answer could ever end correctly.
- The brief's `padding_idx=256` follows the same swap. `arch20.NetRNN` has `padding_idx=BOS`, which is 257 (`git show $D:$M/arch20.py | sed -n '43p'`).
- The digest hashes each answer value mod 256, so correcting the symbol changes all three digests.

**Required, R1.**

- In `sw_world_spec.py` line 8 write `EOS, BOS = 256, 257`.
- In the brief's learner section write `padding_idx=257` (BOS, as `arch20.py` line 43).
- Regenerate `OUTPUT_SW_WORLD_MAC.txt` and replace the digests of check 1 by these, which my implementation gives with every other rule unchanged (`python3 -I $C spec stats`):

  | World | First 16 visits | First 64 visits |
  |---|---|---|
  | 1 | 05e4888194e7b2fcbe83d7aba98a9e31cb2ca0c7d327c977daa12490e3c898b3 | ef5afac92b5ef98d7e3d63eb7bf53ca10198cb0811ab2662188f553a0d6272f2 |
  | 2 | 3d7ba088a0b76b0278571d67cba55b0b1ba9d3233b10a82cfac310b6ecc2a998 | 983ff313b0fb88a4ccc6bc01d609733ed8d5f8981f9e56fe256e24ffaad44ce2 |
  | 3 | ea204b199ba7ef450189571d881dae7b87980d8528f0fbde7a1a5a751adb99e1 | dfc3270f241638e0bed82db04da8daaf3692c1bd2eb6aba2439b827c258a0efc |

  Check 1 should use both columns: the first 16 visits of world 3 are all group 0, so that digest covers two tasks only.

## 3. The text is not enough to rebuild the generator

The brief says the reviewer's implementation is written "from the text below" and must agree on the digests. The text does not state: the order and numbering of tasks; the code alphabet and how codes are assigned; which bytes are the operation symbols; the generator and order of the split and which half takes the odd pair; how groups are formed; how a return's class and group are drawn and what happens when a class is empty; the generator and order of draws for items; how k and the letters are drawn; what the digest hashes; and what `needs_wrap` is.

**Required, R2.** Add this paragraph under "The world, as code":

> **The rules, in full.** With S the world's seed:
> - *Tasks.* The pairs (operation index, modulus), operation outer and modulus inner, are shuffled by `Random(f"world|{S}|tasks")`. A task's id is its place after the shuffle. Operations are numbered: the 16 linear ones with c1 outer and c2 inner over (1, 2, −1, −2), then max, then min.
> - *Codes.* The 676 two-lower-case-letter codes in order, first letter outer, are shuffled by the same generator after the tasks. Task i takes code i.
> - *Symbols.* The bytes of `+-*/%&|^~!?:;<>@$#` belong to operations 0 to 17 in order. So `+` is a + b, `*` is a − b, `-` is a + 2b, `<` and `>` are linear, `$` is max and `#` is min.
> - *Split.* A task's pairs, a outer and b inner, are shuffled by `Random(f"world|{S}|split|{id}")`. The first `round(p*p/2)` are taught, which is the smaller half when p is odd.
> - *Groups.* Tasks 2g and 2g + 1 form group g. Groups are met in this order.
> - *Schedule.* One generator, `Random(f"world|{S}|schedule")`. At every visit after the first, one `random()` is drawn; the visit is new if it is below the new-group chance and an unmet, unreserved group remains. Otherwise `randrange(3)` draws the absence class (visits since the group's last visit: 1 to 4, 5 to 32, 33 or more); if that class is empty, the first non-empty of short, medium, long is used; the group is `sorted(class)[randrange(len(class))]`.
> - *Items.* Each visit v has its own generator, `Random(f"world|{S}|visit|{v}")`. Each update draws 64 items in order, the first 48 taught: task `group[randrange(2)]`, pair `pool[randrange(len(pool))]`, then `randint(0, 4)` for k, then k letters by `choice`.
> - *`needs_wrap`.* For a linear operation, the plain integer c1·a + c2·b is outside 0 to p − 1. Never for max and min.
> - *Digest.* SHA-256 over every item in order, taught then asked: the bytes `[id // 256, id % 256, a, b, len(prompt)]`, the prompt, and each answer value mod 256.

The file's docstring says an empty class falls back to "the nearest non-empty one"; the code takes the first non-empty of short, medium, long. Keep the code and correct the sentence.

## 4. The task space, codes and split

Sound, with one control to add. `python3 -I $C text tasks`; `python3 -I $C spec blind`

- **No two operations coincide** on all pairs at any modulus from 8 to 37, and no operation ignores an operand.
- **Coefficient 2 at an even modulus** (15 of the 30) halves the answers that occur for the four operations with both coefficients ±2, and makes 2a take p/2 values. These are properties of the tasks and not errors.
- **Format.** The largest operand and answer is 36. Codes are lower-case, letters upper-case, and no symbol is a digit, a letter or `=`. Codes are unique (the file asserts it).
- **The modulus.** The code says nothing about it and the symbol is shared across moduli, so a task's modulus has to be learned from its examples, as the memo asks.
- **`needs_wrap` is nearly constant for half the linear operations.** With both coefficients negative, 98 to 100% of pairs need a wrap; for a + b and a − b, 44 to 49%. The split by wrap is informative only for some operations.
- **A shortcut the brief does not record.** Six of the 18 operations are symmetric (c1 = c2, max, min): 180 of 540 tasks. For those, 49.8% of asked pairs have their mirrored pair (b, a) among the taught pairs of the same task, with the same answer. E26's split has the same property. A learner can answer such an item by recall plus commutativity.

**Required, R3.** Add to the event file one more value per asked item, `mirror`: 1 if the operation is symmetric, a ≠ b and (b, a) is a taught pair of the task; else 0. Add to the scorer's table: "Score on asked items of symmetric operations, split by `mirror`."

**Recommended.** Record the number of wraps, `(c1·a + c2·b) // p`, in place of the 0/1.

## 5. What a task receives, and the gate

**The arithmetic.** `python3 -I $C spec alloc`

- A visit gives each task of its group 24 taught items per update: 6,144 per visit.
- Over a life the share of new visits fixes the allocation, whatever the length:

  | Life | Groups met (8 worlds, mean) | Visits per group: mean, median | Groups with one visit only | Taught items per task: median, mean |
  |---|---|---|---|---|
  | 64 visits | 13.2 | 4.8, 4 | 22% | 24,576; 29,700 |
  | 256 | 52.8 | 4.9, 3 | 22% | 18,432; 29,800 |
  | 1,024 | 203.2 | 5.0, 4 | 20% | 24,576; 31,000 |

- **The allocation falls with the order of meeting.** In a 1,024-visit life, groups first met in each quarter get 9.3, 4.8, 3.5 and 2.3 visits on average, and 1%, 10%, 23% and 45% of them get one visit only.
- **Beside full scale.** A task of an E26 birth had 65,500 taught items (4,096 updates) at 63 to 84% and 131,000 (8,192 updates) at 88 to 95%. A task of life family F3, with a life behind it, had 32,800 at 77 to 94%.
- **In showings of each taught pair** the small world is generous: a visit shows each taught pair 12,288 / p² times, 192 at p = 8, 25 at p = 22 and 9 at p = 37, against 16 to 33 for a full-scale birth. **In updates it is not:** 256 per visit, and a median of 768 to 1,024 over a task's whole life, against 4,096 to 8,192.
- The first group is alone for 5.3 visits on average before a second arrives (median 4), and has 6 to 16 of the first 16 visits and 7 to 39 of the first 64.

**The decision.** The coordinator is right. "90% by the end of the first visit" asks a 256-update birth of a network with nothing behind it, and L2 does not need it. What L2 needs is that the first group is learned at all, that the quantity whose trend is read is off the floor and the ceiling, and the cost bound. Lives of 16 visits cannot show the second: they meet 1 to 5 groups (world 3 meets one, world 4 two), and no absence of 33 visits exists in them.

**Required, R4.** Replace "Job L1" from its first line to the end of the ladder by:

> **Job L1: calibration.** Worlds 1 to 8, lives of 64 visits, arms `naive` and `sweep`, default settings. Sixteen lives, together the cost of one life of 1,024 visits.
>
> **The gate (the reviewer's, revised in its audit).** At a rung, under `naive`, in at least six of eight worlds: (1) both tasks of the first group reach the 0.8 criterion within the life; (2) the median first-visit level of the tasks met after the first group is between 0.10 and 0.90; and the measured cost puts a life of 1,024 visits within 6 CPU-hours. The two bounds of (2) are not marks on a learner: they say that the instrument is off its floor and its ceiling.
>
> **The ladder.** If (1) fails or the median of (2) is below 0.10: visits of 512 updates; then H = U = 96 with visits of 512. If the median of (2) is above 0.90: visits of 128 updates. The visit length and the new-group chance are arguments of the world's constructor, not copies of the file; each rung's digests are recorded. Stop at the first rung that passes and report every rung tried.
>
> **L1 reports,** for each arm and world: every task's first-visit level and first-visit cost; the first group's score at the end of each of its visits; score by age and the blind predictor's; kept at return on the visit's first update, by absence class and by wrap; the score on symmetric operations by `mirror`; CPU-seconds per visit.

- I dropped the rung "moduli 8 to 19 with a new-group chance of 0.1". Its reason does not hold: at 0.1 a life of 1,024 visits meets more than 108 groups in a fifth of worlds (200 worlds: mean 102, most 133).
- P39.2 and P39.4 are on the old gate and need restating.
- A life of V visits is the first V visits of any longer life in the same world and arm (section 8), so L1's lives at the adopted rung are the start of L2's and are not run again.

## 6. The batch, replay and the ledger

- The batch is as the memo asks: 48 taught and 16 asked from one stream, updates on taught items only, asked items answered before the update with whole answers. `exp45.batch_tensor` pads to the longest item, so mixed lengths are fine.
- `arch26.forward` takes its sizes from the module (`m.units` and the weights), so H = U = 64 runs unchanged (`git show $D:$M/arch26.py | sed -n '50,75p'`).
- The replay arm is not yet deterministic as written: the buffer's capacity is in bytes while the sampling is over items of different lengths; the order of drawing and offering within an update is not given; and "the number of weights" does not say whether the embedding table counts.

**Required, R5.** Replace "The buffer" paragraph's second and third sentences by:

> Its capacity in items is `(4 × P) // 15`, with P the number of parameters, embedding table included, and 15 bytes the longest item: 14,033 items at the default size (P = 52,625). In each update the batch's replay items are drawn first, by `randrange(len(buffer))` with replacement from `Random(f"world|{S}|replay")`; then the 48 new taught items are offered in batch order from the same generator. The n-th item offered is stored while the buffer is not full; after that `j = randrange(n)` is drawn and the item replaces slot j if j is below the capacity. Replay starts at the first update in which the buffer holds ρ·48 items; until then the arm learns from all 48 new items.

## 7. The event file and the scorer

- **The cost trend is biased by construction.** It is fitted over "tasks that reached 0.8". A task met late has fewer visits left (section 5), so among late tasks only the cheap ones reach. That pushes the slope toward negative for any learner, and a negative slope is the result L2 looks for.
- **"Kept at return" is not yet what was kept.** The first 32 asked items of a task span the visit's first four updates, so up to three updates of relearning are inside it. The first update's items are answered before any update of the visit.
- **Three definitions are open:** whether a mean of "the last 32" exists before 32 items; how the n-th task is ordered within a group; and what the taught-item count means under replay, where an arm learns from fewer new items. A count that depends on the arm makes costs incomparable between arms.
- **The file's layout is not given,** so two scorers cannot be pointed at one file.

**Required, R6.** Replace the event-file paragraph by:

> **The event file** is one compressed `.npz` with arrays of equal length, one entry per asked item in stream order: `update` (uint32), `visit` (uint16), `task` (uint16), `correct`, `blind`, `wrap`, `mirror` (uint8), `seen` (uint32: the task's stream taught items presented before this update, learned from or not, so that it is the same for every arm), and `replayed` (uint32: the task's replayed items before this update; zero without replay).

and in the scorer's table:

> - *Throughout:* a mean of the last 32 asked items exists only once the task has 32; tasks are numbered by first visit, then id; "taught items" is `seen`.
> - *First-visit level* (new): the mean of the task's last 32 asked items in its first visit. *First-visit cost* (new): its `seen` at the first asked item of that visit at which the mean of its last 32 is at least 0.8; "not reached in the first visit" otherwise.
> - *Trend of learning* (new, and the one compared): least-squares slope of first-visit level against log(n), n from 3. Every task has one whole first visit, so none is left out.
> - *Cost of the n-th new task* and *Trend of the cost*: kept as described, reported, and not used in any comparison.
> - *Kept at return:* the share right among the task's asked items in the visit's first update, pooled over returns in each absence class, and split by wrap; the first-32 version beside it.
> - *Relearning at return:* `seen` within the visit at the first asked item at which the mean of the visit's own last 32 is at least 0.8.
> - The scorer takes the number of visits to read as an argument.

My numpy scorer implements these and passes a hand-made file (`python3 -I $C text scorer`). It also rebuilds every column of an event file except `correct` from the generator alone (`python3 -I $C spec events <file.npz> <world>`). Add that to check 7: the runner's 16-visit event file equals the reviewer's rebuild in every column but `correct`.

## 8. Additions for the memo's section 17.10

The brief was written against revision 3. Revision 4 (ef2c03a3) needs four things of L0. Two properties of the generator already help: each visit's items have their own generator, so the stream restarts at any visit boundary from the visit's number alone; and the schedule is drawn in sequence, so a shorter life is an exact start of a longer one.

**Required, R7.** Add:

> - **Probe tasks.** The last 20 groups of each world (tasks 500 to 539) are never scheduled: a visit is new only while fewer than 250 groups are met. They are that world's probe tasks. (No schedule or digest of a 1,024-visit life changes: over 200 worlds the most groups met is 248.)
> - **Complete state** is saved at visits 1, 2, 4, 8, … and at the end: weights, optimizer state, the buffer and the replay generator's state, and the per-task counts the event file needs. This replaces "at the end and at visits 64 and 256".
> - **`--resume <state>`** continues a life exactly. **`--branch <state> --groups g1,g2,…`** runs the named groups, probe groups included, from that state in place of the schedule, with items from `Random(f"world|{S}|branch|{visit of the state}|{k}")` for its k-th visit, writing its own receipt and event file and leaving the trunk's files untouched. **`--ask <state>`** answers a given list of asked items at that state without learning.
> - **Check 9, resume.** A `naive` life and a `replay` life (ρ = 0.5) of 8 visits, stopped at the state of visit 4 and resumed, give the weights hash, the optimizer-state hash and the event file of the uninterrupted lives. A life run twice gives the same hashes.
> - **Prefixes.** A life of V visits is the first V visits of any longer life in the same world and arm. L2 runs each arm once at its longest length and reads 64 and 256 visits from the same event file.
> - **The learner's own time constants.** Decay shrinks a weight the stream does not push on with a constant of 1 / (learning rate × decay): 4,000 updates at the defaults, 15.6 visits. A short absence is 0.06 to 0.26 constants, a medium one 0.32 to 2.0, a long one 2.1 or more. A roster that changes the learning rate, the decay or the visit length restates these.
> - **Not in this brief:** L2 also runs `naive` at a second width and reports the agreement between lengths and the two probes (memo 17.10).

## 9. Checks, pace and storage (recommended)

- **Checks.** Check 5 can fail for swapped rules, which E36's first version could not. Checks 2, 3, 4 and 6 are sound.
- **Pace.** The brief's "Not in this brief" matches memo 17.8. Add its other two lines: at least eight worlds per setting up to 256 visits with paired differences and intervals, and the reviewer's spot-check of saved lives.
- **Supply of tasks.** 250 groups last about 1,250 visits at a new-group chance of 0.2. A longer life needs a larger task space, which is a change to the generator. Say so under "What the small world cannot show".
- **Storage.** A 1,024-visit event file has 4.2 million entries, and the container's disk is nearly full (a checkout of the default branch failed here today). Say what is committed: receipts, the scorer's tables, and compressed event files and states for world 1 of each arm; hashes for the rest.
- **The output file** has two lines per world where the script as committed prints three. Regenerate it from the corrected file.
- **Symbols.** `-`, `<` and `>` do not mean what they mean in E26. R2's text says so; nothing else depends on it.

## 10. E34, addendum 2

- The addendum is the memo's section 18, and the reading is as I fixed it.
- `exp28_everybyte.dev` does run longer as it is: its digest checks are guarded at lines 231 and 260, and `e26_rows` always reads the full-dose receipts.
- **Line 181 of `exp34_entropy.py` is not the only obstacle.** `e28_reference` (lines 126 to 131) chooses its file by `E.DEVTAG`, which is set for any dose other than the standard one. At 98,304 it looks for `e28_qual/dry/e28_dev_S-all_dry98304_73129.json`, which does not exist, and fails before line 181 is reached. `git show $D:$M/exp34_entropy.py | sed -n '126,131p;181p'`

**Required, R8.** In addendum 2, after "asserts at line 181 that every saved row is in the reference", add: "and `e28_reference` (lines 126 to 131) must return the full-dose reference `exp45/e28/e28_dev_S-all_<seed>.json` when the dose is above 24,576, since `DEVTAG` is set for any non-standard dose and the dry path it builds does not exist."

**Added after the audit was written.** The worker's branch already carries this fix: `exp34_entropy.py` on `claude/vibrant-curie-ivmrt1` (commit be01b07c) takes the dry path only when the dose is below the full dose, and the full-dose reference otherwise (`git show origin/claude/vibrant-curie-ivmrt1:$M/exp34_entropy.py | sed -n '130,136p'`). R8 therefore asks only that the addendum's text say so.

## 11. Can E39 be released?

Not as it stands. After R1 to R7 as written, yes, without a re-audit; R8 is for E34's addendum. The check that stands in for a re-audit is check 1 with the digests of R1: if the corrected file gives them, the generator is the one audited here.

| | Change | Where |
|---|---|---|
| R1 | Start and end symbols; `padding_idx`; new digests | file line 8; learner section; check 1 |
| R2 | The generator's rules in the text | "The world, as code" |
| R3 | The `mirror` value and its score | event file; scorer |
| R4 | The gate, the ladder, lives of 64 visits | Job L1 |
| R5 | The buffer made deterministic | "The buffer" |
| R6 | The event file's layout; first-visit quantities; kept at return | "What a life writes"; scorer |
| R7 | Probe tasks, complete state, resume, branch, ask, time constants | world; "What a life writes"; checks |
| R8 | `e28_reference` at a longer dose | E34 addendum 2 |

## 12. What I did not check

- Anything that needs torch: that a target of 257 raises as I expect, the cost per visit, the forward and gradient checks.
- The runner, network and scorer files, which do not exist yet.
- Whether 0.10 and 0.90 are good bounds for the gate; they are wide on purpose, and L1 shows the scale.
- The coordinator's predictions, beyond the two that the new gate touches.
- E34's addendum 2 beyond the two code passages.
