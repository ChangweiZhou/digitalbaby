# E35-A: innate circuits, inventory 甲 (the four mod-89 operations wired, as E10) — reference stage with backpropagation (15 November 2026, revision 2)

Evidence level E2. **This is the reference stage of a new line: backpropagation is allowed here and nothing in this brief is a candidate rule.** The programme and the two inventories are the outside reviewer's (its memo of 15 November); the user asked for one brief per inventory. The architecture, checks, marks, outcome table and predictions are mine. Revision 1 (907c9fc4) was audited by the reviewer, who named four required changes and five recommendations and stated that with the required ones made it need not see the briefs again. Revision 2 makes all nine; the last section lists them. **Released.** One worker; a different worker does E35-B. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch so that no wrapper limit under 90 minutes applies.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106`.

## The question

Earlier work (E10, review 9.20) showed that a ring with hard-wired operations answers held-out and untaught problems exactly once its interfaces are accurate, but the interfaces were learned with the parsing supplied: the network was handed the two operands and the operator. Two things are open, and this line tests only these:

1. Can the interface from the byte stream to a fixed circuit be learned when no parsing is supplied?
2. How wide a range of tasks does a small set of fixed circuits cover? (Not in this brief.)

Two inventories are tested in parallel on one apparatus. **Inventory 甲 (E35-A)** wires the four operations for the modulus, as E10 did. **Inventory 乙 (E35-B)** wires only primitives that know nothing about the modulus, and the number code has to be learned. This brief is inventory 甲.

## Task and data (the birth protocol of E26, unchanged except the taught fraction)

The four mod-89 operations of `exp11_cell.DEV` (`git show $D:$M/exp11_cell.py | sed -n '49,56p;224,268p'`): an item is `code a1 a2 op d_1..d_k b1 b2 = c1 c2 EOS`, k uniform in 0 to 12 letters in training. Answer bytes are the only targets (`exp26_candidate.step`, lines 84-91). 24,576 updates, batch 64, AdamW as `exp26_candidate.make_opt` builds it.

**The taught fraction f is the one thing varied in the data.** `exp11_cell.split(task, f, seed)` returns (taught pairs, held-out pairs) for each task; E26 used f = 0.5. Here f is 0.10 and 0.25.

- **Held-out panel:** `exp11_cell.panel(task, held, seed, ks=(0, 4, 8, 12, 20, 40), n=300)` for each of the four tasks.
- **Taught panel:** the same call on the taught pairs with `seed + 1`, ks = (12,).
- **Exposure.** A held-out item's operand pair is never presented with that task's code, at any k. The same pair of numbers may be presented with another operation (each task has its own split). A table of taught items scores zero on the held-out panel.

## The apparatus (identical in E35-A and E35-B except the module)

The controller is E26's GRU, `arch26.new_net(seed)`, with `arch26.forward`'s own lines (`git show $D:$M/arch26.py | sed -n '50,76p'`), `local=False` throughout. Three arms:

| Arm | Network |
|---|---|
| `G` | The GRU alone. It is exactly E26's arm B. Run by the E35-A worker; the E35-B worker reproduces only its dry hash (check 1). |
| `M` | The GRU plus this inventory's fixed module |
| `X` | The GRU plus the scrambled module: same wiring, same learned parameters, the fixed operations replaced as written below |
| `F` | Forced: arm `M` with the direct read-out's logits for the ten digit bytes removed, so digits can be emitted only through the module. Run only if outcome row 3 occurs (below). |

The wiring, as code. `Hs` is the stack of GRU states after each byte, shape (B, L, 256); `z2` is as in `arch26.forward`.

```
c  = content(Hs)                                   # the module's code for "what the controller holds now"; per inventory, below
g  = T.sigmoid(wg(Hs))                             # wg: Linear(256, 2); write gates for registers A and B at every step
o  = T.softmax(wo(Hs), -1)                         # wo: Linear(256, 4); weights over the four fixed operations at every step
q  = T.sigmoid(wq(Hs))                             # wq: Linear(256, 2); how much of the module's tens and units read-out reaches the logits
A, Bq = neutral, neutral                           # per inventory, below; every item starts from the neutral code
r = []
for t in range(L):
    A  = (1 - g[:, t, 0:1]) * A  + g[:, t, 0:1] * c[:, t]      # (for inventory 乙 the gate multiplies every component of the code)
    Bq = (1 - g[:, t, 1:2]) * Bq + g[:, t, 1:2] * c[:, t]
    r.append(sum(o[:, t, j:j+1] * OP[j](A, Bq) for j in range(4)))
r = T.stack(r, 1)
d = dec(r)                                         # (B, L, 2, 257): a tens head and a units head; per inventory, below
base = l3(z2)
if forced: base = T.cat([base[..., :48], T.zeros_like(base[..., 48:58]), base[..., 58:]], -1)   # arm F: bytes '0'..'9' get nothing from l3
logits = base + q[..., 0:1] * d[:, :, 0] + q[..., 1:2] * d[:, :, 1]
```

**Write the lines from `A, Bq = neutral` to `r = T.stack(r, 1)` as one function `module_path(c, g, o)`** that the forward calls, so that check 2b can drive it with hand-made inputs.

- **No parsing is supplied.** Every gate is computed from the GRU's own state. The module never sees the bytes, the positions or the roles.
- **Arm `G` is the network with the last two terms absent**, so it is E26's arm B bit for bit.
- **New parameters** (`content`, `wg`, `wo`, `wq`, `dec`) are created after `arch26.new_net(seed)` under `T.manual_seed(seed + 35)`, in the order listed, with torch's defaults. They join the optimizer's decayed group; `exp26_candidate.make_opt` asserts its groups, so build the optimizer with `exp16_integrated.adamw16(m, LR, WD, ())` and record the groups read back.
- **Scoring** uses `exp11_cell.score` unchanged: install your forward as `exp11_cell.forward` with the signature `forward(m, x, local, through_time, keep=False, silence=False)`.
- **Ablation at test** (`module_off=True`): the two module terms are dropped from the logits; nothing else changes.
- **Arm `F` masks by byte class, not by position:** the same ten logits are zeroed at every step, so no parsing is supplied.

## Inventory 甲: the module

Codes are distributions over 89 positions. The operations are E10's (`git show $D:$M/exp10_ring.py | sed -n '36,61p'`) with `MOD = 89`:

```
IDX = T.arange(89)
def op_add(a, b): return T.einsum("bi,bki->bk", a, b[:, (IDX[:, None] - IDX[None, :]) % 89])
def op_sub(a, b): return T.einsum("bi,bki->bk", a, b[:, (IDX[None, :] - IDX[:, None]) % 89])
def op_max(a, b): F = a.cumsum(1) * b.cumsum(1); return T.cat([F[:, :1], F[:, 1:] - F[:, :-1]], 1)
def op_min(a, b): S = a.flip(1).cumsum(1).flip(1) * b.flip(1).cumsum(1).flip(1); return T.cat([S[:, :-1] - S[:, 1:], S[:, -1:]], 1)
OP = [op_add, op_sub, op_max, op_min]
content = lambda Hs: T.softmax(wc(Hs), -1)         # wc: Linear(256, 89)
neutral = T.full((B, 89), 1 / 89)
dec     = lambda r: wd(r).reshape(B, L, 2, 257)    # wd: Linear(89, 514)
```

Apply the operations to (B·L, 89) by flattening, or per step.

**Scrambled module (arm `X`).** Each operation is replaced by a fixed random table `tab_j[i, k]` in 0 to 88, four tables drawn once from `T.Generator().manual_seed(35)` by `T.randint(0, 89, (4, 89, 89), generator=...)`:

```
def table_op(tab):
    flat = tab.reshape(-1)
    def f(a, b):
        outer = (a[:, :, None] * b[:, None, :]).reshape(len(a), -1)
        return T.zeros(len(a), 89, dtype=a.dtype).index_add(1, flat, outer)
    return f
```

A random table has no group or order structure for the interfaces to learn around, and no learned parameters. My check (`PROJECT_DOCUMENTS/COORD_CHECKS_20261115/innate_ops_check.py`, output beside it): the four operations are exact on all 7,921 pairs of one-hot codes; `table_op` with the true addition table equals `op_add`; a random table agrees with addition on 1.4% of pairs.

**What is wired and what is learned.** Wired: the modulus, the four operations, and that numbers are positions on a ring of 89. Learned: which states mean which number, when to write each register, which operation, and the read-out to digit bytes. With this inventory, generalisation to held-out pairs is guaranteed by the wiring once the interfaces are right; the brief tests whether they come right from the byte stream.

## Stage 0: the upper bound with parsing supplied

E10 ran at modulus 97 with its own split. Before the stream version, confirm the bound at modulus 89 and these fractions.

A new file `exp35a_parsed.py`: E10's `RingNet` (not flat) with `MOD = 89` and the four `exp11_cell.DEV` operations, operands given as digit pairs and the operator as its byte, no letters, backpropagation (`local=False`), `T.optim.AdamW(m.parameters(), lr=.003, weight_decay=.01)` as E10's development (`exp10_ring.py` line 175), batch 64, 32,768 updates. Fractions 0.10 and 0.25 by `exp11_cell.split`; seeds 73129, 93683, 116089; arms: the ring, and the ring with the four random tables above. Score: exact two-digit answers on every held-out pair, mean over the four operations.

Record held-out accuracy every 4,096 updates. E10's own record is why the length is 32,768: its 8,192-update run did not converge, and at 32,768 its held accuracy by seed was 100, 96 to 97, and 62 to 93 (`git show $D:$M/E45_REPORT.md | sed -n '207,209p'`).

**Gate.** At f = 0.25 the ring reaches at least 90 (mean over the four operations, final row) in at least two of three seeds. If not, stop and report with the curves; the stream stage does not run.

## Stage 1a: from the byte stream, seed 73129

Up to six births: arms `G`, `M`, `X` at f = 0.10 and f = 0.25.

A new runner `exp35a_stream.py`: E26's birth loop (`exp26_candidate.dev`, lines 120-170) with the taught fraction, the arm's network and the two panels above. Do not call or patch existing `dev` or `step` functions; edit nothing existing. Receipts `experiment="E35A"`, `arm`, `fraction`, in `exp45/e35a/e35a_dev_<arm>_f<10|25>_<seed>.json`, in E26's format.

- At rows 16,384, 20,480 and 24,576: the held-out panel at every k, the taught panel, and the held-out panel at k = 12 and 40 with `module_off=True`. Save and commit (`git add -f`) the weights at these three rows.
- Per 512 updates, on the training batch, detached: the mean of each write gate and of `q` at each fine role (`exp28_everybyte.roles_of`), and the mean largest operation weight at the `=` position. These show where the controller writes and reads.
- **Order (the reviewer's recommendation).** Run arm `G` at both fractions first and report its held12 by family. Run `M` and `X` only at fractions where some family has room. Push `G`'s receipts as soon as they are complete: the E35-B worker reads them.
- Cost: arm `G` is about 3,070 CPU-seconds (E26's arm B); the module arms are expected at 1.3 to 2 times that. Six births, about 7 to 9 CPU-hours.

## Checks before any full birth

1. **Arm `G` is E26's arm B.** With f = 0.5 and E26's panels, a dry birth of `G` at 2,048 updates has the `tensors_sha256` of `exp45/e26_qual/e26_dev_B_dry2048_73129.json` (adf933d7…). This also checks the two workers' loops against each other.
2. **The fixed operations.** On all 7,921 pairs of one-hot codes the four operations give the exact answer with all mass on it; `table_op` with the true addition table equals `op_add` to 1e-6; the four random tables are the ones `manual_seed(35)` gives (record their sha256).
   - **2b. The whole path with ideal interfaces, no training.** Drive `module_path(c, g, o)` with hand-made inputs for an item of length L: `c` holds the exact code of the first operand at one step and of the second at a later step (a one-hot distribution), `g` is 1 for register A at the first of those steps and 1 for register B at the second and 0 elsewhere, and `o` is one-hot for the operation at every step. On all 7,921 pairs, each of the four operations, and 0, 12 and 40 steps between the two writes, the result at the last step has all its mass on the right answer; and with `wd` set by hand to the tens and units look-up, `dec` gives the right two digit bytes. A wiring fault would otherwise look like outcome row 3.
3. **The module arms leave the controller's start unchanged.** Before the first update, arms `M` and `X` have the same values as `G` in every parameter `G` has.
4. **Ablation is exact.** With `module_off=True`, the logits of `M` equal those of a `G` network holding the same shared parameters.
5. **No parsing leaks.** The forward of `M` takes the byte tensor only; assert that no mask, role or position tensor is an argument of it.
6. **Dry births** of `M` and `X` at f = 0.25, 2,048 updates: finite losses, CPU time, a forecast. Report before the full births; do not wait for me unless a loss is not finite or the forecast for the whole of stage 1a exceeds 10 CPU-hours.

## Endpoint and marks (fixed)

Every figure is the mean of rows 16,384, 20,480 and 24,576; each operation is tabled separately as well.

- **held12:** exact answers on held-out pairs at 12 letters. **held40:** the same at 40 letters. **taught12:** on taught pairs at 12 letters.
- **Generalises:** held12 at least 90. **Memorises:** taught12 at least 90 and held12 at most 30.
- **Two families, read separately (the reviewer's recommendation):** add and sub; max and min. They differ much in difficulty for the GRU, and a four-operation mean can hide room in one. held12, held40 and taught12 are means over the two operations of a family; room and the outcome row are decided per family and fraction.
- **Room:** at a fraction, for a family, arm `G`'s held12 is below 85.
- **Uses the module:** `M`'s held12 falls by at least 15 points with `module_off=True`.
- Differences under 15 points between single births are not read.

## Outcomes and readings for stage 1a (fixed; written before the predictions)

Read for each family at each fraction separately; the first row that applies is the reading.

| # | Pattern at a fraction | Reading |
|---|---|---|
| 0 | No room: `G` held12 at least 85 | The GRU alone already generalises at this fraction; the module cannot be shown to add anything here. Not read further. |
| 1 | Room; `M` held12 at least 15 above `G` and at least 15 above `X`; `M` uses the module | The controller learned to use the fixed circuit from the byte stream on this seed, and the structure is what helps. One seed; stage 1b runs. |
| 2 | Room; `M` at least 15 above `G`; `X` within 15 of `M` or above it | Something in the added pathway helps, but the scrambled circuit does as well: not the structure. |
| 3 | Room; `M` within 15 of `G` | The controller did not gain from the circuit on this seed. This does not yet say the interface cannot be learned: the GRU's direct read-out can memorise the taught items, the training loss then vanishes, and the module path gets no gradient. **Arm `F` runs at this fraction on this seed before anything else** (below). |
| 4 | Anything else, including `M` at least 15 above both `G` and `X` without using the module, `M` below `G` by 15 or more, or a non-finite loss | Reported as it is; no reading. |

**After row 3: arm `F`, same fraction, seed 73129** (about one birth). Its held12 for that family is read against `G` and `M`:

| `F` | Reading |
|---|---|
| At least 15 above both `G` and `M` | The interface can be learned from the stream when it must be; in `M` the obstacle was competition from the direct read-out. The stop rule does not apply. I write the next brief on how to give the module path its share. |
| Within 15 of `G` | Forced through the module, the controller still did not learn the interface on this seed. Stage 1b applies the stop rule with arm `F`. |
| Otherwise | Reported; no reading. |

If every family is row 0 at both fractions, I decide whether to rerun at f = 0.05 before anything else.

## Stage 1b (released by me after stage 1a is read; seeds 93683 and 116089)

- **Row 1 at a fraction:** arms `M`, `X` and `G` at that fraction on both seeds. The sentence, if row 1 holds on all three seeds: "With inventory 甲, a GRU trained by backpropagation learns from the byte stream, with no parsing supplied, to use the fixed circuit; held-out pairs at fraction f are answered at … against … for the GRU alone and … for the scrambled circuit, on three seeds."
- **Row 3 with arm `F` also within 15 of `G`:** arm `F` at that fraction on both seeds. **Stop rule (the reviewer's, as amended by its audit):** if `F` stays within 15 of `G` on at least two of the three seeds, the obstacle is the interface and this inventory's line stops here. A failure of `M` alone never stops the line.
- **Row 2 or 4:** I name the arms from the table before any run.

## What this brief cannot show

- **Nothing about a rule inside the line.** Every network here is trained by backpropagation. A candidate born by an allowed rule is the next step and needs its own brief.
- **Untaught compositions, few-shot acquisition of a new family, and retention** are not measured here. The birth protocol has single operations only.
- **Breadth.** Whether the same circuits serve a task family not looked at during design is the third step; the family is not chosen yet.
- **Why the controller does or does not use the circuit.** The gate diagnostics describe, they do not explain.
- **A pass-through.** The fixed module can act as a pipe: `max(A, 0)` returns A unchanged. A controller could memorise answers itself and send them through the module, which would meet "uses the module" without using the structure. Row 1 also requires 15 points over the scrambled circuit, so its reading is not affected, but "uses the module" alone is not evidence of computation in the module.
- **Thin coverage at f = 0.10.** Every number appears in both operand positions at both fractions, but at 0.10 some appear in a given position of a given task only once or twice (the reviewer's count; at least ten times at 0.25). A result at 0.10 therefore also tests transfer of the number code between tasks.
- **Inventory 甲 wires the answer's structure.** A pass says the interface can be learned from the stream, not that the circuit was discovered.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P35A.1 | Stage 0 gate passes (ring at least 90 at f = 0.25 in two of three seeds) | 80% |
| P35A.2 | Stage 0: the ring also reaches 90 at f = 0.10 in two of three seeds | 55% |
| P35A.3 | Stage 0: the random-table arm stays at or below 30 at both fractions in all seeds | 85% |
| P35A.4 | Arm `G` has room at f = 0.10 for add and sub | 85% |
| P35A.5 | Arm `G` has room at f = 0.25 for add and sub | 60% |
| P35A.6 | Row 1 for add and sub at one fraction or both | 35% |
| P35A.7 | Row 3 for add and sub at every fraction with room | 45% |
| P35A.9 | If arm `F` runs, it is at least 15 above both `G` and `M` | 60% |
| P35A.8 | Where `M` generalises at 12 letters, its held40 is at least 60 | 60% |

## Done

Stage 0: the parsed runner, its receipts, and a table (fraction × seed × arm; mean and per operation) with the gate's verdict. Stage 1a: the stream runner, check receipts (2b included), the dry births, the full births with receipts and the three saved rows each, and `E35A_REPORT.md` with the table (arm × fraction: held12, held40, taught12, held12 and held40 with the module off; mean and per operation; last-three mean and final row), the gate diagnostics at the end of each module birth, which outcome row applies for each family at each fraction, and the outcome of each prediction. No readings beyond the tables above.

## What changed from revision 1 (the reviewer's audit; its script is `e35_brief_checks.py` on `claude/vibrant-cerf-sje726` at 6858f25c)

I made these as the audit wrote them. I checked one of its facts myself: E10's development ran 32,768 updates after an 8,192-update run failed to converge (`E45_REPORT.md`, lines 207 to 209).

| | Change | Required? |
|---|---|---|
| 1 | After outcome row 3, a forced arm `F` (digit logits of the direct read-out removed) runs before any stop; the stop rule is applied to `F`, and a failure of `M` alone never stops the line | yes |
| 2 | Check 2b: the whole register, operation and result path driven with ideal interfaces on every pair, operation and three gap lengths, with no training | yes |
| 3 | Row 3 is only "`M` within 15 of `G`"; "`M` above `G` and `X` without using the module" goes to row 4 | yes |
| 4 | Stage 0 runs 32,768 updates with a curve every 4,096 | yes |
| 5 | Arm `G` first; module arms only where there is room | recommended |
| 6 | Add/sub and max/min read as separate families | recommended |
| 7 | (E35-B) stage 0 also under the stream stage's optimizer | recommended |
| 8 | (E35-B) stage 0 with a complex parameterisation and a digit-table form | recommended |
| 9 | The pass-through, and thin coverage at f = 0.10, under "cannot show" | recommended |

---

# Addendum 1 (16 November 2026): stages 0 and 1a read; stage 1b released

**As I read it** (branch `claude/nice-rubin-y5de8k`, b988f1b8, merged; the figures are the worker's, and the reviewer checked the final rows of `G`, `M` and `X` against the receipts; I recomputed none).

- **Stage 0, gate passed.** With operands supplied in separated form, the ring generalises: 92.7, 91.4 and 49.1 at f = 0.25 (two of three seeds at 90 or more), and 97.3, 86.8 and 92.1 at f = 0.10, against at most 13.6 for the random table. P35A.1, P35A.2 and P35A.3 hold.
- **Stage 1a, seed 73129.** Only add and sub at f = 0.10 leave room (the GRU alone at 19.6; 89.2 at f = 0.25, so P35A.4 holds and P35A.5 misses). There, held-out add and sub at 12 letters: `G` 19.6, `M` 21.1, `X` 19.4. **Outcome row 3.** Turning the module off changes `M` by 0.1 points: it is not used. The forced arm `F`: 6.6, within 15 of `G` (13.0 below it). All four have 95 to 98% on taught pairs. P35A.7 holds; P35A.6 and P35A.9 miss. I had given the forced arm 60% to succeed.

**Stage 1b, as the brief fixes it.** Arm `F` at f = 0.10 on seeds 93683 and 116089: two births, about 6 CPU-hours by the reviewer's estimate. Arm `G` at f = 0.10 on those seeds is already on file from the E35-B worker (`exp45/e35b/`, the same network and data under the runner whose arm `G` reproduced E26's hash): held-out add and sub 32.1 and 27.8. Use those; do not rerun them.

**Stop rule.** If `F` stays within 15 points of `G` on at least two of the three seeds, the obstacle is the interface and inventory 甲's line stops here. Nothing is briefed after stage 1b in either case (the reviewer's memo, section 8).

**Prediction (mine).** P35A.10: `F` is within 15 of `G`, or below it by more, on both new seeds, so that the stop rule applies: 85%.

**Done.** The two births with receipts and the three saved rows each, an addendum to `E35A_REPORT.md` with the per-seed table for `G` and `F` (held12, held40, taught12 for add and sub), whether the stop rule applies, and P35A.10.
