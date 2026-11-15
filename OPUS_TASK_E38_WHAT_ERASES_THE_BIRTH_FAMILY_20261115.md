# E38 (job B): what erases the unrehearsed family, and does an anchor protect it? (15 November 2026, revision 1)

Evidence level E2; stage 1 is one seed and supports no claim by itself. The question, the arms and the proposed marks are the outside reviewer's (its memo at a8b5ace2 on `claude/vibrant-cerf-sje726`, section 7); this specification, the checks and the predictions are mine. **Not released: it waits for the reviewer's audit.** One worker.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106`. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit. Edit nothing existing; do not call or patch existing `dev`, `life` or `step` functions except where a line below names one to call.

## Why

The birth family is lost in the first block of every life, under the sweep, Kolen–Pollack feedback and backpropagation alike (table in the E36 brief). Two causes are possible and no run has separated them:

- **Decay toward zero.** AdamW at rate 0.001 with decay 0.25 multiplies every weight outside the embedding table by 0.99975 per update (`git show $D:$M/exp16_integrated.py | sed -n '176,187p'`). What the stream does not push on keeps 21% of its size after one block of 6,144 updates.
- **Rewriting.** The new family's own updates overwrite what the old family used.

E38 separates them and tests the simplest protection: pull each weight toward its birth value instead of toward zero. Each weight uses only its own two values, so the device is inside the user's line.

## Lives: E26's first two blocks, seed 73129, ten lives

Start from `exp45/e26/e26_dev_S_73129.pt` (sweep arms) and `e26_dev_B_73129.pt` (backpropagation arms), admitted by `exp26_candidate.admit_e26`. The life is `exp26_candidate.life`'s loop (lines 196-279) over `X16.SCHEDULE[:2]` only (F1.1 and F1.2, 12,288 updates), with the same entries and stream seeds, so that arm `zero` repeats E26.

| Arm | The one difference from E26's life | Extra stored numbers | Share of each batch that is rehearsal |
|---|---|---|---|
| `zero` | none | none | 0 |
| `none` | no decay during the life | none | 0 |
| `anchor` | decay 0.25 toward each weight's birth value | one per decayed weight | 0 |
| `anchor10` | the same at strength 2.5 | the same | 0 |
| `rehearse` | decay as `zero`; the birth family's taught items join the stream as a second family | none | about half |

**The decay variants, as code.** `opt = exp26_candidate.make_opt(m)`; group 0 is every parameter but the embedding table and has `weight_decay = 0.25` (`exp26_candidate.expected_optimizer`, lines 67-69).

```
# none, anchor, anchor10:
opt.param_groups[0]["weight_decay"] = 0.0
decayed = list(opt.param_groups[0]["params"])
birth   = [p.detach().clone() for p in decayed]          # taken once, right after the birth file is loaded
# anchor (KAPPA = 0.25) and anchor10 (KAPPA = 2.5), inside the step, after loss.backward() and before opt.step():
with T.no_grad():
    for p, p0 in zip(decayed, birth):
        p.sub_(LR * KAPPA * (p - p0))                    # LR = 0.001
opt.step()
```

torch's AdamW shrinks the weight before the Adam step, so the pull goes before the step too. Record the count of stored birth values.

**Arm `rehearse`, as code.** In `life`'s loop, before block 1:

```
birth_entries = [("op", t, a, b) for t in exp11_cell.DEV for a, b in exp11_cell.split(t, .5, seed)[0]]
seen_streams  = [X16.entry_stream(birth_entries, seed + 100)]       # then block 1 appends F1's stream as E26 does
```

`X16.joint_stream` then draws each item from one of the two streams with equal chance. Record the count of birth-family and F1 items per block.

## Rows and scores

Score at updates 0, 64, 128, 256, 512, 1,024 and then every 1,024 to 12,288 (17 rows). Save weights at each; commit those at 6,144 and 12,288 (20 files) and record the sha256 of the rest.

1. **Birth family, held out:** `exp16_integrated.panels(seed)[0]` entries `add89`, `sub89`, `max89`, `min89` at k = 12.
2. **Birth family, all pairs, no letters, split by wrap:** as in the E36 brief, score 2.
3. **The new family:** `add97` and `sub97` at k = 12 from the same panels.
4. At rows 6,144 and 12,288 only: everything `life`'s own `evaluate()` returns, for the check below.

## Checks

1. **`zero` is E26.** For each rule, at rows 6,144 and 12,288 the batch digest, the mean loss of the last 512 updates and every score of `evaluate()` equal `stages[1]` and `stages[2]` of `exp45/e26/e26_life_<S|B>_73129.json`. If not, the runner is wrong: stop.
2. **The anchor step against torch's own decay.** With `birth` set to zeros and KAPPA = 0.25, the weights after 64 updates agree with arm `zero`'s (largest absolute difference at most 1e-5 of each tensor's largest entry).
3. **The anchor step in a second form.** A second implementation, written separately as `p.copy_(p0 + (1 - LR * KAPPA) * (p - p0))`, gives weights after 64 updates that agree with the first form to 1e-6 relative, for both strengths. This and check 2 stand in for a second worker's implementation of a three-line step.
4. **`none` has no decay:** the optimizer's groups read back with `weight_decay` 0.0 in both.
5. **`rehearse`:** no item of the held-out panels of either family is presented (as `life` asserts for its own panels); the per-block item counts are recorded.
6. One dry life of 256 updates per arm under the sweep, with CPU time and a forecast (the reviewer's estimate: 4.7 CPU-hours for the ten lives).

## Marks (fixed; the reviewer's proposal, adopted), read at row 12,288

- **Keeps the birth family:** held-out add and sub mod 89 at 12 letters, mean at least 80, and wrap pairs (mean of add and sub) at least 70.
- **Loses it:** that held-out mean at most 50. (E26 after block 2: 47.8 under the sweep and 47.5 under backpropagation on this seed.)
- **Acquires the new family:** held-out add and sub mod 97 at 12 letters, mean at least 85.
- Between the marks: intermediate, not read.

## Readings (fixed; written before the predictions), per rule

| # | Result | Reading |
|---|---|---|
| 0 | Check 1 fails | The runner is wrong. Stop. |
| 1 | `none` keeps | Decay toward zero is what erased the family on this seed. |
| 2 | `none` loses | The new family's own updates erase it; decay is not needed for the loss. |
| 3 | `anchor` or `anchor10` keeps and acquires | A candidate on one seed: retention with no rehearsal, for one stored number per weight. Stage 2. |
| 4 | An anchor keeps and does not acquire | Too stiff at that strength. Both strengths are reported side by side. |
| 5 | Neither anchor keeps | A uniform pull is not enough on this seed. |
| 6 | `rehearse` | The reference, not a reading: what rehearsal keeps, and what it costs the new family per update and per item of that family. |

Rows 1 and 2 are about `none`; rows 3 to 5 about the anchors; both sets are reported. An intermediate result is reported as intermediate. Reported beside them, as facts: the row at which each arm's held-out mod-89 mean first falls below 80, since decay acts over thousands of updates and rewriting can act within hundreds; and the paired difference between the two rules for each arm.

## Stage 2 (released by me after stage 1 is read)

Arms that keep and acquire, on seeds 93683 and 116089. Then one full six-block life per rule under the passing anchor, to see composition and length. A fixed anchor suits a two-block screen only; over a long life the anchor has to move, and that is designed only if the screen passes.

## What stage 1 cannot show

- One seed, and seed 73129 is the one whose E26 lives lost least in block 1 (49 and 44 against 1 to 34 on the others), so "loses it" sits close to the mark there.
- Whether the protection is specific to the sweep: the same device can be added to backpropagation, and both are run. A claim would be against naive backpropagation without the device.
- Why a pull works or fails.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P38.1 | Check 1 passes for both rules | 90% |
| P38.2 | `none` keeps the birth family (sweep) | 15% |
| P38.3 | `anchor` keeps it (sweep) | 25% |
| P38.4 | `anchor10` keeps it (sweep) | 55% |
| P38.5 | `anchor10` acquires the new family (sweep) | 55% |
| P38.6 | Some anchor keeps and acquires (sweep) | 35% |
| P38.7 | `rehearse` keeps and acquires (sweep) | 70% |
| P38.8 | A reading that differs between the sweep and backpropagation | 15% |

## Done

Runner, check receipts, ten lives with receipts, and `E38_REPORT.md` with the curves (arm × rule × row: the three score groups), the marks at row 12,288, which readings apply per rule, the facts listed, and the outcome of each prediction. No readings beyond the table above.
