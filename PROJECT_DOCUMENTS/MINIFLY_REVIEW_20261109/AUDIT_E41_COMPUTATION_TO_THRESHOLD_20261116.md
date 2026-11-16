# Audit of the E41 brief at 213c38f4 (16 November 2026)

From the outside reviewer. Read-only: nothing was trained and no code from another branch was executed. The brief is `OPUS_TASK_E41_COMPUTATION_TO_THRESHOLD_BY_WIDTH_AND_FRACTION_20261116.md`, revision 1; it implements section 21.4 of the reviewer's memo (revision 8, at 557e3749).

**Verdict.** The three widths are what the memo meant, the network class is E26's at the present width as far as reading the source can show, the two accuracies and the sample of 1,000 are appropriate, and every factual claim I could test holds, including the brief's table and its cost estimate. Five changes are required before release. Two of them correct the memo's own design, not the coordinator's reading of it: the doses, and the rule for a run that does not reach a threshold. The other three are a check that two of the three workers cannot compute as written, a sample whose generator is not fully specified, and one added check of the new scoring code. All five are given in section 8 as text to apply. After them E41 can be released without a re-audit.

**Terms.**

- The *widths* are pairs (H, U): H is the size of the three feed-forward layers and U the number of GRU units. *Half* is (256, 128), *present* (512, 256), *double* (1,024, 512).
- The *taught fraction* f is the share of each task's operand pairs that the training stream draws from; the other pairs are *held out* and never taught.
- A *scored row* is a number of updates at which a run is scored; the brief scores every 256 updates. The *dose* is a run's total number of updates.
- A *reduction pair* is an operand pair of add or sub whose answer needs the reduction at the modulus: a + b ≥ 89, or a − b < 0.
- *Accuracy A* is the share of whole answers right on 300 held-out pairs per task at 12 letters, mean of add and sub. *Accuracy R* is the same on 1,000 held-out reduction pairs per task without letters. Whole answers are two digits and the end symbol.
- *Updates to threshold* is the first scored row at which an accuracy reaches a threshold (50, 80 or 90). *Computation to threshold* is that count in *weight-items*: parameters × 64 × updates.
- The *sweep* is the project's local rule (`local=True`): credit is sent back through the sign of each weight matrix times its mean magnitude.
- A *birth* is a run from random weights on the first task family (add, sub, max, min mod 89). E26's six births (two rules, three seeds, f = 0.50) and E35-A's arm `G` (the same network under backpropagation at f = 0.25 and 0.10, seed 73129) are the births on file at the present width.
- A *receipt* is the JSON record a run writes. A *batch digest* is a SHA-256 over every batch fed to the learner, in order. E26's *dry receipts* are its two qualification births of 2,048 updates, seed 73129.

Shell names: `D` and `M` as in the brief; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`; `C=$R/e41_brief_checks.py` on `claude/vibrant-cerf-sje726`. `python3 -I $C <letter>` runs one part (A to G).

## 1. The three widths (the coordinator's point 1)

**They are what the memo meant.** No change.

- The memo's words are "every hidden size scaled together; the embedding left at 8". The network has two hidden sizes, 512 and 256, set in two files; the brief's line references are right (`git show $D:$M/exp11_cell.py | sed -n '28p'`; `git show $D:$M/arch20.py | sed -n '32p;43,50p'`). Scaling both by the same factor, with the embedding at 8 and the 257 outputs unchanged, is the intended definition.
- **The parameter counts are right:** 317,457, 1,124,113 and 4,211,985, from the layer shapes; the middle one is the count in E26's receipts. The steps multiply the weights by 3.541 and 3.747. `python3 -I $C A`

## 2. The network class at the present width (point 2)

**Bit-identical by reading; the brief's check 1 is the test.** No change required.

- `NetW(512, 256)` creates the same six modules as `arch20.NetRNN("gru")` in the same order (`book`, `l1`, `rnn`, `lm`, `l2`, `l3`), with the same arguments, directly after `T.manual_seed(seed)`. The two attributes set first draw no random numbers. So the initial values should be equal bit for bit. I could not run it (no torch in the reviewer's container).
- `arch26.forward` takes its sizes from the module (`m.units` and the weight shapes) and contains no fixed size; `exp3_state.lin` and the sign rule are shape-free; `exp26_candidate.step` is lines 84 to 91 as cited. `git show $D:$M/arch26.py | sed -n '50,75p'`
- E26's birth loop does nothing between updates that changes the trajectory (its one extra call after the first update only reads gradients), so scoring every 256 updates instead of every 4,096 leaves the hash at 2,048 updates unchanged. `git show $D:$M/exp26_candidate.py | sed -n '141,146p'`
- The two hashes in check 1 are the ones on file (79618ab2… and adf933d7…). `python3 -I $C F`
- One caution, recommended: `exp16_integrated.adamw16` exempts the embedding from decay only when the environment variable `E16_BOOKDECAY` is `off`, and its default is `on` (`git show $D:$M/exp16_integrated.py | sed -n '145p;176,187p'`). `exp26_candidate` sets it through `exp20_reference` on import. The brief should say: import `exp26_candidate` before any harness module, and build the optimizer with `exp26_candidate.make_opt(m)`, which asserts the two groups by parameter name at any width. Check 1 would catch the error at the present width; this avoids it.

## 3. The two accuracies and the sample of 1,000 (point 3)

**Appropriate.** One required change, to the sample's generator (change 3).

- **The two accuracies are the memo's.** A is comparable with E26's receipts; R excludes the pairs a network can answer without the modulus.
- **A at f = 0.50 uses E26's 300 pairs with other letters.** `exp11_cell.panel` draws the pairs first and then the letters for each letter count in turn, so asking for 12 letters alone gives different letters from E26's panel of five letter counts. On the six births' final weights the two panels differ by at most two items per task. `python3 -I $C E`
- **The sample of 1,000 is large enough.** Held-out reduction pairs number 1,944 to 1,987 per task at f = 0.50, 2,933 to 2,975 at 0.25 and 3,517 to 3,541 at 0.10. The sampling error of R at 80% is 0.9 points (A: 1.6), less than the swing of a single row (section 5). `python3 -I $C D G`
- **Required: the generator's string is not fully specified.** `Random(f"e41|{seed}|{f}|{op}")` depends on how the fraction is printed (`0.1` or `0.10`, `0.5` or `0.50`) and on what `op` is. Two programs can then draw different samples, and the added check below needs one sample. Change 3 fixes the string.
- Recommended: say that the taught accuracy tabled is the mean of add and sub on the taught panel, as for A.

## 4. The doses (the memo's own; they bear on points 4 and 5)

**Required: two doses are too short for what the brief reads from them** (change 1). The doses are the memo's (section 21.4), so this corrects the reviewer.

- **Present width.** By a straight line between the rows on file, A passes 80% between 3,886 and 6,899 updates in the seven births on file. 90% is reached by 8,192 in three of the six births at f = 0.50, by 12,288 in five, and not before 24,576 at f = 0.25. With a dose of 8,192 the three-row version of the threshold (section 5) can record nothing after 7,680. The present width is one side of both width steps, so a threshold missed there leaves two comparisons open. **12,288 updates.** `python3 -I $C B G`
- **Half width.** If a half-width run does not reach the threshold, the comparison in weight-items can still be decided, but only if its dose was long enough: the half width must have been given at least 1.5 × 3.541 times the present width's updates to the threshold. Under the brief's 16,384 that holds only if the present width reaches the threshold by 3,085 updates, which none of the seven births on file does. Under **40,960 updates** it holds if the present width reaches the threshold by 7,712, which covers all seven births on file. The half and the present width then also receive nearly the same total computation (1.30 and 1.38 × 10^10 weight-updates; the double width 3.45 × 10^10). `python3 -I $C G`
- **Double width.** 8,192 stands: when the present width reaches a threshold and the double width does not, the weight-items comparison is decided wherever within its dose the present width reached it.
- **Cost.** The brief's estimate reproduces from E35-A's receipt (0.111 CPU-seconds per update at the present width; 24.1 CPU-hours of updates and 4.1 of scoring). With the two longer doses it is about 30 of updates and 5 of scoring, 11.8 per worker, so the limit in check 6 has to rise. `python3 -I $C G`

## 5. The second threshold rule (point 4)

**Keep it, and let it carry the description** (required, change 2).

- **The brief's reason holds.** On file a single row moves against the trend by up to ten points (backpropagation, seed 73129: 94.3, 84.2, 98.5 at rows 16,384 to 24,576). Across the 19 series scored at finer spacing in E36 and E38, a first crossing of 90 is followed by a row below 90 in 8, and of 80 in 1. `python3 -I $C B C`
- **The single-row version is early by an amount that depends on width.** Near 80% the births on file rise by 0.6 to 1.6 points per scored row of 256, less than the swing of a single row. Several rows in succession therefore lie within the swing of the threshold, and the first to exceed it is recorded. The more slowly a curve rises, the more such rows there are, so the single-row version is early by more for a network that learns slowly. Width is expected to change the slope, so this error would differ between the widths compared, and it would understate the advantage of whichever width learns faster.
- **The brief does not say which version the description and the predictions use.** That has to be fixed beforehand. The three-row version errs late and the single-row version early, so tabling both and marking the cells where they disagree shows where a description depends on the choice.
- Under the three-row version the recorded row is the first of the three, and nothing can be recorded in a run's last two rows. Change 2 states both.

## 6. How the result is stated (point 5)

**Not computable as written when a threshold is not reached; complete once that is defined** (required, change 2).

- "Not reached counted as above the dose" gives no ratio, and the doses differ by width. A run that does not reach the threshold is known only to need more than its dose. The description can then be given only when it holds for every value above the dose; otherwise that seed is "not decided". Change 2 gives the four cases as a table.
- **The boundary has to be assigned.** Rows are multiples of 256, so a ratio of exactly 1.5 occurs (2,048 against 3,072). Change 2 puts both boundaries inside "stays".
- With "not decided" and "neither reaches" added, the rule of two seeds in three stays as written.
- At f = 0.10 expect "neither reaches" in most cells: at the present width the network memorises its taught pairs (97.7%) and scores about 21 on A and 19 on R after 24,576 updates. The information there is in the levels; see section 9, item 4. `python3 -I $C E`

## 7. The checks (point 6)

**Six are not enough: check 2 needs a different row, and the new scoring code needs a check** (required, changes 4 and 5).

- **Check 2 cannot be computed by two of the three workers.** E26's digest at row 2,048 exists only in the two dry receipts, which are for seed 73129. E26's full receipts carry digests at multiples of 4,096, for all three seeds and equal between the rules. Change 4 moves the comparison to row 4,096. `python3 -I $C F`
- **Check 1's dry runs are for seed 73129 only.** They are the only test of the update at the level of a hash, so every worker has to run them, whatever its own seed. Change 4 says so.
- **Accuracy R, its sample and the taught panel are new code with one implementation,** and the description rests on R. None of the six checks would catch a sample drawn from the wrong pairs. Change 5 adds a check against the reviewer's reader: digests of the samples, and counts on the six E26 births. The reader reproduces the receipts exactly where the two can be compared (E26's sealed panel at 12 letters on six files; E35-A's taught panel on two). `python3 -I $C D E`
- Checks 3, 4 and 5 are sound. The reference in check 4 exists: `arch20._forward_rnn(m, x, False)` runs torch's own GRU on the module's parameters.

## 8. The five required changes, as text to apply

**Change 1. Doses.** Replace the line "**Dose:** 8,192 updates; 16,384 at the half width." with:

> **Dose:** 8,192 updates at the double width, 12,288 at the present width, 40,960 at the half width. The half and the present width then receive nearly the same computation (1.30 and 1.38 × 10^10 weight-updates) and the double width 3.45 × 10^10.

In check 6, replace the last two sentences with:

> The reviewer's estimate for all 54 is about 30 CPU-hours of updates and 5 of scoring, a little over half of it at the double width. If a worker's forecast exceeds 15 CPU-hours, report before starting.

In P41.6, replace "within its dose" with "within 16,384 updates", so that the prediction keeps the meaning it was written with.

**Change 2. The two versions of the threshold, and the description.** Replace the paragraph that begins "Because single rows swing" with:

> Two versions of the row at which a threshold is reached are tabled. *Single-row:* the first scored row at or above the threshold. *Three-row:* the first row of the first three consecutive scored rows that are all at or above it. A run's last two rows cannot begin three such rows, so under the three-row version a threshold can be recorded up to the dose less 512 updates. The description below and predictions P41.2 to P41.8 use the three-row version. The single-row version is tabled beside it, and a cell in which the two versions give different descriptions is marked.

In "How the result is stated", replace everything from "For each rule and fraction, from each width to the next" down to and including "otherwise it is 'mixed'." with the following; the closing sentence on the factor of 3.7 stays.

> The description is given for updates to threshold and for computation to threshold in weight-items, on the 80% threshold of accuracy R and of accuracy A separately, for each rule and fraction and each step from a width to the next wider one. The same is given for 50 and 90 in a second table. The CPU-second figures are given beside the weight-items.
>
> For each seed, let x_n and x_w be the narrower and the wider run's updates to the threshold, and D_n and D_w the largest count at which each could have been recorded (the dose under the single-row version, the dose less 512 under the three-row version).
>
> | Narrower run | Wider run | Description for that seed |
> |---|---|---|
> | reached | reached | *falls* if x_w / x_n is below 2/3; *rises* if above 1.5; otherwise *stays* |
> | reached | not reached | *rises* if D_w / x_n is 1.5 or more; otherwise *not decided* |
> | not reached | reached | *falls* if x_w / D_n is 2/3 or less; otherwise *not decided* |
> | not reached | not reached | *neither reaches* |
>
> For computation in weight-items the same table is used with every count multiplied by its width's parameter count (317,457, 1,124,113 or 4,211,985). A cell's description is the one that at least two of its three seeds share; otherwise the cell is *mixed*. The three seeds' values are given in every cell.

**Change 3. The sample's generator.** In score 2, replace `Random(f"e41|{seed}|{f}|{op}").sample(sorted(pairs), 1000)` with:

> `Random(f"e41|{seed}|{round(f * 100)}|{t[0]}").sample(sorted(pairs), 1000)`, where `t` is the task's tuple in `exp11_cell.DEV` (so `t[0]` is `"add"` or `"sub"`) and `pairs` is the list of `(a, b)` tuples of that task's held-out reduction pairs.

**Change 4. Checks 1 and 2.** Add to check 1: "Every worker runs this check, whatever its own seed." Replace check 2 with:

> 2. **Same data at every width and rule.** The batch digest (`exp26_candidate.Digest`) at row 4,096 is equal across the six runs of a seed and fraction. At f = 0.50 it equals `curve[0].data_digest.sha256` of `exp45/e26/e26_dev_S_<seed>.json`: bbd71ab7ca27b72e… for 73129, 65b1c7e06b17459a… for 93683, 1df203fb8f299dc0… for 116089. In check 1's two dry runs the digest at row 2,048 is 4f622d94e0156993….

**Change 5. A seventh check.** Add:

> 7. **The scorer against the reviewer's reader.** (a) For the worker's seed and each fraction, the sha256 of `json.dumps([list(p) for p in sample])` for the add and the sub sample begins with the digits in the first table. (b) The run's own scoring code, given f = 0.50 and the file's seed, applied to the six E26 births `exp45/e26/e26_dev_<S|B>_<seed>.pt`, each loaded into `NetW(512, 256)` with `load_state_dict(T.load(file))`, gives the counts in the second table. A count that differs by one or two items is reported and the work proceeds; a larger difference is a failure.
>
> | Seed | f = 0.10 (add, sub) | f = 0.25 (add, sub) | f = 0.50 (add, sub) |
> |---|---|---|---|
> | 73129 | 2d9d251a354dc291, 92c29b4a374be126 | 3f31c9a15729adc3, 89a86290d78daddd | f433a55ca52ae34d, 18ac75b5013b6d47 |
> | 93683 | 7a5f67e2fc00ed81, c316f9e9ca56339e | 81d35edfe17f2b54, 19b61154e17286f4 | 368844acc03dfd7a, 83274756387907f7 |
> | 116089 | 49965d4fbf5633ee, 3efe08c20a9e5ad3 | bef6158428f9008a, 0fdd40508ad49609 | 26b2eff70e51e033, 8ccb9c3177772cbf |
>
> | File | A: add, sub (of 300) | A | R: add, sub (of 1,000) | R | Taught: add, sub (of 300) | Taught |
> |---|---|---|---|---|---|---|
> | `e26_dev_S_73129.pt` | 295, 293 | 98.00 | 962, 955 | 95.85 | 298, 298 | 99.33 |
> | `e26_dev_S_93683.pt` | 296, 293 | 98.17 | 988, 979 | 98.35 | 297, 298 | 99.17 |
> | `e26_dev_S_116089.pt` | 300, 299 | 99.83 | 993, 996 | 99.45 | 300, 296 | 99.33 |
> | `e26_dev_B_73129.pt` | 295, 296 | 98.50 | 976, 987 | 98.15 | 295, 297 | 98.67 |
> | `e26_dev_B_93683.pt` | 298, 294 | 98.67 | 970, 968 | 96.90 | 298, 298 | 99.33 |
> | `e26_dev_B_116089.pt` | 284, 279 | 93.83 | 955, 895 | 92.50 | 285, 288 | 95.50 |

As a fixed point for (a): the add sample at f = 0.50, seed 73129, begins (30, 64), (84, 37), (64, 49), (46, 54). The tolerance of two items allows for the arithmetic of two programs; a wrong sample would differ by more in most cells, and part (a) tests the sample directly.

## 9. Other points (recommended; none holds the release)

1. **One runner or three.** Say whether the three workers use one `exp41_scale.py` (written by one of them and taken at a stated commit) or each writes its own. With checks 1, 2, 4 and 7 either is safe; each receipt should record the file's sha256.
2. **A check of each worker's own seed at no cost.** At the present width and f = 0.50 a run is E26's birth, so scoring E26's own panel (`ks=(0, 4, 8, 12, 20)`, n = 300) at rows 4,096 and 8,192 must reproduce the `sealed` counts in `exp45/e26/e26_dev_<S|B>_<seed>.json` exactly. The dry receipts cover seed 73129 only; this covers the other two.
3. **A further fixed point for seed 73129.** The stream at f = 0.25 and 0.10 is the same call as E35-A's arm `G`, and that arm's network is `arch26.new_net(seed)` under backpropagation (`git show origin/claude/nice-rubin-y5de8k:$M/exp35a_stream.py | sed -n '212,223p'`). The digest at row 4,096 should be a71b1ca10b24a36b… and e120924e5af546f9…, and under backpropagation at the present width the taught panel's counts at rows 4,096 and 8,192 should equal that receipt's. E35-A is not merged, so a difference is reported, not a failure.
4. **Levels beside the highest value.** The highest accuracy over a run grows with the number of rows scored, and the half width has 160 rows against 32. Table also the mean of A and of R over rows 7,680, 7,936 and 8,192, which every width has.
5. **CPU-seconds.** Three processes at once share the memory cache, and the double width's matrices do not fit in it, so CPU-seconds per update measured during the runs may be inflated by different amounts at different widths. Take the figure per update from check 6's dry runs on an otherwise idle host, and record how many processes ran beside each full run.
6. **Reproducibility at the other widths.** Run check 6's dry run twice at the half and the double width; the two `tensors_sha256` must be equal.
7. **Weights.** The 54 final files are 407 MB (303 MB of it the double width). They are useful: the reviewer's reader can recompute the last row of every run from them. If the size is unwelcome, commit the half and present widths for all seeds and the double width for seed 73129, and record `tensors_sha256` for the rest. The sentence "record the sha256 of nothing else" is unclear; I suggest "each receipt records `tensors_sha256` and `weights_sha256` of the final weights, as E26's receipts do".
8. **The tables' script.** Commit the script that computes the thresholds and descriptions from the receipts; the reviewer will recompute them independently.
9. **"What it cannot show", two additions.** A threshold reached before about 1,500 updates is located to no better than one part in six, so a description there can turn on one row. The three-row version records a crossing later than a smooth curve would, by an amount not measured here.
10. **Two wordings in "What is on file".** At f = 0.10 the figure stays between 15 and 21 to 24,576, not "near 17". At f = 0.50 one birth (backpropagation, 93683) is already at 84.3 by 4,096, so "at or before 8,192" is the accurate statement. The remark on the earlier window network is right (review 9.7 and 9.18).

## 10. Can E41 be released?

**Yes, after changes 1 to 5, without a re-audit.** They are text replacements and one added check; none alters the network, the data, the update or the scores of a run. The memo's section 21.4 still carries the old doses; this audit supersedes it on that point.

## 11. What I did not check

- **Bit-equality of `NetW(512, 256)` with `arch26.new_net`,** and that the forward and the sweep run at the other widths: read from source only. Checks 1 and 4 test them on the worker's host.
- **CPU-seconds at the half and the double width.** My figures scale the present width's by the parameter count; check 6 measures them.
- **The three-row version at a spacing of 256 on births.** No birth on file is scored more finely than every 4,096 updates; the reversal counts are from E36 and E38, which are not births.
- **Whether learning rate 0.001 and decay 0.25 suit the other widths.** The brief lists this under what it cannot show.
- **The chances attached to the predictions.**
- **E35-A's receipts** are read from an unmerged branch (`claude/nice-rubin-y5de8k` at 01e6fa1d).
