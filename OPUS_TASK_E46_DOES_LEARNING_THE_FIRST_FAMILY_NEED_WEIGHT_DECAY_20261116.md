# E46: does learning the first family need weight decay at all? (16 November 2026, revision 1)

Evidence level E2. This is the first check of the plan the user adopted on 16 November (review 9.148): weight decay alone removes whatever the stream stops pushing on, so before any retention experiment it must be known what decay buys. The specification and predictions are mine. **Not released: it waits for the reviewer's audit.** One worker, best an E41 worker, whose runner this copies.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106`. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit. Edit nothing existing. (The number E45 is skipped: `exp45` is the harness's own module and folder.)

## Why

- Every birth and life on file uses AdamW at rate 0.001 with decay 0.25. At that setting a weight that receives no gradient shrinks by 0.99975 per update, and that alone takes a learned family's reduction pairs to 18 to 52% of their level after 1,024 updates and to zero by about 2,000, with no learning at all (review 9.145; my rerun on two births).
- So no retention without rehearsal is possible at the standing setting, whatever else is done. The simplest remedy is no decay. Whether that costs anything is not on file for a birth.
- What is on file: a second family is acquired without decay as well as with it (E38, seed 73129, new family at 96.3 and 96.6 against 92.5 and 94.8). The decay of 0.25 was chosen on 7 November for the sign rule on another network (review 9.20 to 9.21) and has been carried since.

## The comparison with the record (the working rule adopted on 16 November)

At decay 0.25, the present width and half of the pairs taught, accuracy A passes 80% between 3,886 and 6,899 updates in the seven births on file, and 90% by 12,288 in five of six (E41 audit). On seed 73129, E41's own runs (three-row version): 80 on A at 5,632 updates under the sweep and 5,120 under backpropagation; 80 on R at 6,656 and 6,400. So a dose of 12,288 locates the 80% threshold at the standing decay with room of about a factor of two, which is what a comparison of "falls, stays, rises" at 1.5 needs.

## Runs

E41's present-width cell at taught fraction 0.50, exactly, with one thing changed: the decay of the optimizer's first group.

| Decay | Runs | Source |
|---|---|---|
| 0.25 | the reference | E41's own runs `exp45/e41/e41_<S|B>_h512_f50_<seed>.json`. Seed 73129's are on the default branch. For a seed whose E41 receipts are not there, this worker runs that cell with `exp41_scale.py` unchanged; it is then an E41 run and is named as one. |
| 0.05 | 6: both rules, three seeds | new |
| 0 | 6 | new |

- **Code.** A copy of `exp41_scale.py` named `exp46_decay.py` with one added argument, the decay. After `opt = exp26_candidate.make_opt(m)`, which asserts the standing groups, set `opt.param_groups[0]["weight_decay"] = wd` and record the groups read back and the decay in a field of its own. The embedding table has no decay in any arm.
- **Dose, data, scores, rows:** E41's at the present width: 12,288 updates, scored every 256, accuracies A, R and the taught panel, the same panels and sample.
- **Added score, every 256 updates:** the length test of E30 and E42, `exp11_cell.panel(t, exp11_cell.split(t, .5, seed)[1], seed, ks=(12, 40), n=300)` with `t = ("add", 89, b"p")`, since decay may matter for extrapolation and not for accuracy in range.
- **Added record, at every scored row:** the norm of each weight group, to show what the weights do without decay.
- **Committed:** final weights of every run (`git add -f`; confirm they are on the remote), receipts in `exp45/e46/e46_<S|B>_wd<000|005>_<seed>.json`. For the reference, the length test is scored once on E41's committed final weights by this runner's code.

About 2.5 CPU-hours for the twelve new runs at E41's measured 0.111 to 0.120 CPU-seconds per update, and about 1.5 of scoring.

## Checks before the full runs

1. **With decay 0.25 the copy is E41.** A dry run of 2,048 updates at decay 0.25 on seed 73129 has the `tensors_sha256` of E26's dry births (79618ab2… under the sweep, adf933d7… under backpropagation), and at row 4,096 of a full-length check under backpropagation the scores equal the row of `e41_B_h512_f50_73129.json`.
2. **The decay is what is set.** With decay 0, on the first batch after one update, every weight outside the embedding equals its value under decay 0.25 divided by (1 − 0.00025), to 1e-6 relative. (AdamW shrinks before its step, so the two runs differ at the first update by that factor alone.)
3. **Same data in every arm:** the batch digest at row 4,096 is E26's for the seed (bbd71ab7…, 65b1c7e0…, 1df203fb…).
4. **The length test's code:** on E26's six final files it gives E30's figures at 12 and 40 letters (`exp45/e30/E30_UNSEEN.json`), as E42's check 12.
5. One dry run of 256 updates per decay, with CPU time and a forecast.

## How the result is stated (fixed beforehand, E41's forms)

For each rule and seed, decay 0 against 0.25 and decay 0.05 against 0.25:

- **Updates to the 80 threshold** on A and on R, three-row version: E41's table of four cases, read as fewer (*falls*, ratio below 2/3), more (*rises*, above 1.5) or about the same (*stays*), with its rules for a run that does not reach the threshold.
- **The level:** the mean of A, of R and of the length test at 40 letters over rows 11,776, 12,032 and 12,288; *lower* or *higher* if the difference is 15 points or more, otherwise *about the same*. For the reference's length test, the final row alone.
- A cell carries the description two of its three seeds share; otherwise *mixed*.

**The sentence this job licenses**, per rule: learning the first family **needs** decay if without it the 80 threshold on R is reached in more updates (*rises* or not reached) or the level on R or at 40 letters is *lower*; **does not need** it if all of those are *stays* or better; otherwise **mixed**, with the cells given. Nothing is said about lives or about other families.

## What it cannot show

- **A life.** Without decay the weights may grow over 36,864 updates in ways a birth of 12,288 does not show; the norms recorded are the first evidence either way.
- **Composition**, which births do not teach.
- **Other rates.** Decay acts through rate × decay; only the rate 0.001 is run.
- **That removing decay gives retention.** It removes one cause of loss. E43 shows a second, faster one.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P46.1 | Check 1 passes | 90% |
| P46.2 | Decay 0 reaches 80 on R within the dose in two of three seeds, under backpropagation | 80% |
| P46.3 | The same under the sweep | 65% |
| P46.4 | The sentence under backpropagation: does not need 45%, mixed 30%, needs 25% | |
| P46.5 | The sentence under the sweep: does not need 35%, mixed 30%, needs 35% | |
| P46.6 | At 40 letters decay 0 is *lower* than 0.25, two of three seeds, under the sweep | 35% |
| P46.7 | Decay 0.05 is *about the same* as 0.25 on every description, under each rule | 55% |
| P46.8 | Without decay the recurrent matrix's norm at row 12,288 is at least 1.5 times its norm under decay 0.25, two of three seeds, under each rule | 70% |

## Done

`exp46_decay.py`, the check receipts, the twelve runs (and any reference cell run as E41) with receipts and final weights on the remote, and `E46_REPORT.md` with the tables (rule × seed × decay: updates to 50, 80 and 90 on A and R in both versions; the three levels; the norms at the end), the descriptions, the sentence per rule and the outcome of each prediction. No readings beyond the sentence above.
