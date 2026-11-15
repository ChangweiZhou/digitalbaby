# E35-B: innate circuits, inventory 乙 (modulus-free primitives: phase addition and a comparator) — reference stage with backpropagation (15 November 2026, revision 1)

Evidence level E2. **This is the reference stage of a new line: backpropagation is allowed here and nothing in this brief is a candidate rule.** The programme and the two inventories are the outside reviewer's (its memo of 15 November); the user asked for one brief per inventory. The architecture, checks, marks, outcome table and predictions are mine. **Not released: it waits for the reviewer's audit.** One worker; a different worker does E35-A. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch so that no wrapper limit under 90 minutes applies.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106`.

## The question

Earlier work (E10, review 9.20) showed that a ring with hard-wired operations answers held-out and untaught problems exactly once its interfaces are accurate, but the interfaces were learned with the parsing supplied: the network was handed the two operands and the operator. Two things are open, and this line tests only these:

1. Can the interface from the byte stream to a fixed circuit be learned when no parsing is supplied?
2. How wide a range of tasks does a small set of fixed circuits cover? (Not in this brief.)

Two inventories are tested in parallel on one apparatus. **Inventory 甲 (E35-A)** wires the four operations for the modulus, as E10 did. **Inventory 乙 (E35-B)** wires only primitives that know nothing about the modulus, and the number code has to be learned. This brief is inventory 乙.

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
logits = l3(z2) + q[..., 0:1] * d[:, :, 0] + q[..., 1:2] * d[:, :, 1]
```

- **No parsing is supplied.** Every gate is computed from the GRU's own state. The module never sees the bytes, the positions or the roles.
- **Arm `G` is the network with the last two terms absent**, so it is E26's arm B bit for bit.
- **New parameters** (`content`, `wg`, `wo`, `wq`, `dec`) are created after `arch26.new_net(seed)` under `T.manual_seed(seed + 35)`, in the order listed, with torch's defaults. They join the optimizer's decayed group; `exp26_candidate.make_opt` asserts its groups, so build the optimizer with `exp16_integrated.adamw16(m, LR, WD, ())` and record the groups read back.
- **Scoring** uses `exp11_cell.score` unchanged: install your forward as `exp11_cell.forward` with the signature `forward(m, x, local, through_time, keep=False, silence=False)`.
- **Ablation at test** (`module_off=True`): the two module terms are dropped from the logits; nothing else changes.

## Inventory 乙: the module

Nothing in the module mentions 89. A code is K = 8 unit phasors and one scalar: a tensor `P` of shape (..., 8, 2) holding (cos, sin), and `s` of shape (...,). The two primitives are **phase addition** (a product of phasors) and **a comparator that routes a whole code**:

```
def mul(a, b):  return T.stack([a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1], a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]], -1)   # angles add
def div(a, b):  return T.stack([a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1], a[..., 1] * b[..., 0] - a[..., 0] * b[..., 1]], -1)   # angles subtract
def big(a, b, sa, sb):   g = T.sigmoid(sa - sb)[..., None, None]; return g * a + (1 - g) * b      # the code with the larger scalar passes
def small(a, b, sa, sb): g = T.sigmoid(sa - sb)[..., None, None]; return (1 - g) * a + g * b
OP = [lambda A, Bq: mul(A.P, Bq.P), lambda A, Bq: div(A.P, Bq.P), lambda A, Bq: big(A.P, Bq.P, A.s, Bq.s), lambda A, Bq: small(A.P, Bq.P, A.s, Bq.s)]

def content(Hs):
    ang = wa(Hs)                                   # wa: Linear(256, 8): eight angles
    return Code(P=T.stack([ang.cos(), ang.sin()], -1), s=ws(Hs).squeeze(-1))     # ws: Linear(256, 1)
neutral = Code(P=(1, 0) in every channel, s=0)
dec = lambda r: wd2(T.tanh(wd1(r.reshape(B, L, 16)))).reshape(B, L, 2, 257)       # wd1: Linear(16, 128); wd2: Linear(128, 514)
```

Every operation returns eight phasors (16 numbers); the scalar is used only inside the comparator. The registers interpolate `P` and `s` with the same gate; `P` is not renormalised.

**Scrambled module (arm `X`).** Each of the four operations is replaced by a fixed random bilinear map of the same shape on the phasor parts, the scalars unused: `W = T.randn(4, 16, 16, 16, generator=T.Generator().manual_seed(35))`, `out_j = T.einsum("bi,bj,ijk->bk", a.reshape(-1, 16), b.reshape(-1, 16), W[j])`. Scale each `W[j]` once by a constant so that the mean output norm over 10,000 pairs of random unit-phasor codes is `sqrt(8)`, and record the four constants.

My check (`PROJECT_DOCUMENTS/COORD_CHECKS_20261115/innate_ops_check.py`, output beside it, with four frequencies): if a number v is coded as angles `2π f_k v / 89` and scalar `10 v`, then `mul`, `div`, `big` and `small` decode to add, sub, max and min on all 7,921 pairs. So an exact interface exists. Whether learning finds one is the question.

**What is wired and what is learned.** Wired: that codes can be added and subtracted as angles, and that a comparison can choose between two codes. Learned: the whole number system (which angles and which magnitude each two-digit number gets, hence the modulus), when to write each register, which operation, and the read-out to digit bytes. Unlike inventory 甲, generalisation here is not guaranteed by the wiring: it needs the learned code to be a homomorphism, which the taught pairs alone must force.

## Stage 0: the upper bound with parsing supplied

Nothing like this module has been run here. Before the stream version, test whether the interfaces can be learned at all when the operands and operator are handed over.

A new file `exp35b_parsed.py`, shaped as E10's `RingNet` (`git show $D:$M/exp10_ring.py | sed -n '64,96p'`): `digit = Embedding(10, 16)`; `enc1 = Linear(32, 128)`, tanh; `wa = Linear(128, 8)` and `ws = Linear(128, 1)` give each operand's code; `opsel = Embedding(256, 4)`, softmax, weights the four operations above; the result goes through `Linear(16, 128)`, tanh, `Linear(128, 20)` to tens and units logits, with E10's `loss_digits`. Operands as digit pairs, the operator as its byte, no letters, backpropagation, `T.optim.AdamW(m.parameters(), lr=.003, weight_decay=.01)` as E10's development (`exp10_ring.py` line 175), batch 64, 16,384 updates. Fractions 0.10 and 0.25 by `exp11_cell.split`; seeds 73129, 93683, 116089; arms: the module, and the scrambled module. Score: exact two-digit answers on every held-out pair, mean over the four operations, and each operation separately.

**Gate.** At f = 0.25 the module reaches at least 90 in at least two of three seeds. If not, stop and report: with parsing supplied, modulus-free primitives with interfaces learned by backpropagation did not give held-out generalisation at these fractions. The stream stage does not run, and I decide whether a smaller question (addition and subtraction only, or f = 0.5) is worth a note. Minutes of CPU.

## Stage 1a: from the byte stream, seed 73129

Four births: arms `M` and `X` at f = 0.10 and f = 0.25. Arm `G` at both fractions is run by the E35-A worker; take its figures from that branch when it reports, and until then report yours without the comparison.

A new runner `exp35b_stream.py`: E26's birth loop (`exp26_candidate.dev`, lines 120-170) with the taught fraction, the arm's network and the two panels above. Do not call or patch existing `dev` or `step` functions; edit nothing existing. Receipts `experiment="E35B"`, `arm`, `fraction`, in `exp45/e35b/e35b_dev_<arm>_f<10|25>_<seed>.json`, in E26's format.

- At rows 16,384, 20,480 and 24,576: the held-out panel at every k, the taught panel, and the held-out panel at k = 12 and 40 with `module_off=True`. Save and commit (`git add -f`) the weights at these three rows.
- Per 512 updates, on the training batch, detached: the mean of each write gate and of `q` at each fine role (`exp28_everybyte.roles_of`), and the mean largest operation weight at the `=` position. These show where the controller writes and reads.
- Cost: arm `G` is about 3,070 CPU-seconds (E26's arm B); the module arms are expected at 1.3 to 2 times that. Four births, about 5 to 7 CPU-hours.

## Checks before any full birth

1. **Arm `G` is E26's arm B.** With f = 0.5 and E26's panels, a dry birth of `G` at 2,048 updates has the `tensors_sha256` of `exp45/e26_qual/e26_dev_B_dry2048_73129.json` (adf933d7…). This also checks the two workers' loops against each other.
2. **The fixed operations.** With numbers coded as angles `2π f_k v / 89` for eight frequencies of your choice and scalar `10 v`, `mul`, `div`, `big` and `small` decode (nearest exact code) to add, sub, max and min on all 7,921 pairs; `mul(a, div(b, a))` equals `b` to 1e-5 on random unit-phasor codes; the four scaled random maps are recorded by sha256 with their constants.
3. **The module arms leave the controller's start unchanged.** Before the first update, arms `M` and `X` have the same values as `G` in every parameter `G` has.
4. **Ablation is exact.** With `module_off=True`, the logits of `M` equal those of a `G` network holding the same shared parameters.
5. **No parsing leaks.** The forward of `M` takes the byte tensor only; assert that no mask, role or position tensor is an argument of it.
6. **Dry births** of `M` and `X` at f = 0.25, 2,048 updates: finite losses, CPU time, a forecast. Report before the full births; do not wait for me unless a loss is not finite or the forecast for the whole of stage 1a exceeds 10 CPU-hours.

## Endpoint and marks (fixed)

Every figure is the mean of rows 16,384, 20,480 and 24,576, and the mean over the four operations; each operation is tabled separately beside it.

- **held12:** exact answers on held-out pairs at 12 letters. **held40:** the same at 40 letters. **taught12:** on taught pairs at 12 letters.
- **Generalises:** held12 at least 90. **Memorises:** taught12 at least 90 and held12 at most 30.
- **Room:** at a fraction, arm `G`'s held12 is below 85.
- **Uses the module:** `M`'s held12 falls by at least 15 points with `module_off=True`.
- Differences under 15 points between single births are not read.

## Outcomes and readings for stage 1a (fixed; written before the predictions)

Read at each fraction separately; the first row that applies is that fraction's reading.

| # | Pattern at a fraction | Reading |
|---|---|---|
| 0 | No room: `G` held12 at least 85 | The GRU alone already generalises at this fraction; the module cannot be shown to add anything here. Not read further. |
| 1 | Room; `M` held12 at least 15 above `G` and at least 15 above `X`; `M` uses the module | The controller learned to use the fixed circuit from the byte stream on this seed, and the structure is what helps. One seed; stage 1b runs. |
| 2 | Room; `M` at least 15 above `G`; `X` within 15 of `M` or above it | Something in the added pathway helps, but the scrambled circuit does as well: not the structure. |
| 3 | Room; `M` within 15 of `G`, or `M` does not use the module | The controller did not come to use the circuit on this seed. The gate diagnostics are reported. Stage 1b applies the stop rule. |
| 4 | Anything else, including `M` below `G` by 15 or more, or a non-finite loss | Reported as it is; no reading. |

If both fractions are row 0, I decide whether to rerun at f = 0.05 before anything else.

## Stage 1b (released by me after stage 1a is read; seeds 93683 and 116089)

- **Row 1 at a fraction:** arms `M`, `X` and `G` at that fraction on both seeds. The sentence, if row 1 holds on all three seeds: "With inventory 乙, a GRU trained by backpropagation learns from the byte stream, with no parsing supplied, to use the fixed circuit; held-out pairs at fraction f are answered at … against … for the GRU alone and … for the scrambled circuit, on three seeds."
- **Row 3 at every fraction with room:** arm `M` only, at the fraction where `G` is lowest, on both seeds. **Stop rule (the reviewer's):** if `M` fails to use the module, or stays within 15 of `G`, on at least two of the three seeds, the obstacle is the interface and this inventory's line stops here.
- **Row 2 or 4:** I name the arms from the table before any run.

## What this brief cannot show

- **Nothing about a rule inside the line.** Every network here is trained by backpropagation. A candidate born by an allowed rule is the next step and needs its own brief.
- **Untaught compositions, few-shot acquisition of a new family, and retention** are not measured here. The birth protocol has single operations only.
- **Breadth.** Whether the same circuits serve a task family not looked at during design is the third step; the family is not chosen yet.
- **Why the controller does or does not use the circuit.** The gate diagnostics describe, they do not explain.
- **Two differences from inventory 甲 at once.** The code is learned and the read-out has a hidden layer. If 乙 fails where 甲 passes, stage 0 says whether the code or the stream interface was the obstacle.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P35B.1 | Stage 0 gate passes (module at least 90 at f = 0.25 in two of three seeds) | 45% |
| P35B.2 | Stage 0: add and sub reach 90 at f = 0.25 in two of three seeds, whatever max and min do | 55% |
| P35B.3 | Stage 0: the scrambled arm stays at or below 30 at both fractions in all seeds | 75% |
| P35B.4 | Stage 1a runs and gives row 1 at one fraction or both | 20% |
| P35B.5 | Stage 1a runs and gives row 3 at every fraction with room | 20% |
| P35B.6 | Where `M` generalises at 12 letters, its held40 is at least 60 | 55% |

## Done

Stage 0: the parsed runner, its receipts, and a table (fraction × seed × arm; mean and per operation) with the gate's verdict. Stage 1a: the stream runner, check receipts, the dry births, the full births with receipts and the three saved rows each, and `E35B_REPORT.md` with the table (arm × fraction: held12, held40, taught12, held12 and held40 with the module off; mean and per operation; last-three mean and final row), the gate diagnostics at the end of each module birth, which outcome row applies at each fraction, and the outcome of each prediction. No readings beyond the tables above.
