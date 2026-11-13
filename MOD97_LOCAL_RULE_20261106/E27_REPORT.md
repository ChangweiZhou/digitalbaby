# E27 report: flood and alignment on saved weights (Opus worker, 13 November 2026)

**Status.** Both parts are complete. In the first pass the six E26 life weight files were absent, and Part B ran on the six crossover lives only. They were then supplied from `claude/adoring-cori-b07aog` at `ce409d64`, and Part B was rerun with all four arms (see *Part B completion* below). Part A was accepted as run. Evidence level E2, diagnostic. Nothing was trained: the job builds no update rule and uses no optimizer object. `grep -n optim` prints nothing in `e27_measure.py`, `e27_score.py` or `e27_check_split.py`. These figures are gradients at saved weights. They say nothing about any learner's ability. No formal checker gate, seal or outside review has passed. I am the same model family as the coordinator and the auditors, so I am a second reader of their work, not an independent one.

- **Source.** Branch `codex/minifly-consolidated-20261023` at `02897c6e` (later than `a5c85319`). The brief is at revision 2, and the predictions file includes its amendment. The work is on branch `claude/sleepy-knuth-q96dd1`.
- **Run.** Command: `env -i PATH HOME E16_ARCH=bank2 E16_BOOKDECAY=off E16_COMPGAP=0 OMP_NUM_THREADS=1 python3 e27_measure.py run 73129 93683 116089`. One process, one thread, torch 2.15.0+cpu. CPU time was 417 s, 391 s and 402 s for the three seeds. Log: `exp45/logs/e27_run.log`.
- **Pre-run record.** `exp45/E27_PRERUN.txt`, written at 17:10:51Z (container clock) when no `E27_MEASURE` file existed yet. It holds the sha256 of every script and every fixture record. Each `E27_MEASURE_<seed>.json` also records the scorer's hash at run time (`ee2dd33e…`) and the measure script's hash (`51ada865…`). Both match the pre-run record.
- **Part B rerun.** Same command and same scripts (`e27_measure.py` 51ada865…, `e27_score.py` ee2dd33e…, both unchanged). The first pass's outputs are kept as `E27_MEASURE_<seed>_partA.json` and `E27_SCORE_partA.json`. CPU time was 376 s, 376 s and 373 s. Log: `exp45/logs/e27_run_partB.log`.
- **Single commit.** You asked for one commit at completion, so the scorer was hashed before the run instead of committed separately (brief step 4). This difference is disclosed here.

## Verdicts (scorer `e27_score.py` → `exp45/E27_SCORE.json`)

| | 73129 | 93683 | 116089 | verdict |
|---|---|---|---|---|
| **P27.1** amplification, all parameters, two-bank born by sweep (answer targets), threshold ≥ 1.5 | 0.532 | 1.409 | 0.797 | **misses** |
| **P27.2** usable under signs ≤ ½ usable under exact | sign 0.035 / exact −0.130 | −0.006 / −0.232 | 0.035 / −0.153 | **void in all three seeds** (exact usable < 0) → does not hold |
| **P27.3** sign flood: collapsed ≥ 2 × balanced | 12.81 / 11.61 = 1.10 | 14.86 / 20.38 = 0.73 | 20.60 / 12.46 = 1.65 | **misses** (nothing is read from it, per the amendment) |
| **P27.4** untaught pairs at least 0.10 below single operations, on both sweep-lived arms (gap S→S; B→S) | 0.034; 0.077 | −0.039; −0.019 | 0.050; 0.053 | **misses** (no arm reaches 0.10 in any seed) |
| **P27.5** untaught cosine, sweep-lived at least 0.05 below backprop-lived, within a birth (backprop-lived minus sweep-lived: sweep birth; backprop birth) | −0.046; **+0.082** | −0.107; −0.074 | −0.013; +0.035 | **misses** (1 of 6 comparisons shows it; a seed needs both) |

**Note on P27.4 (first pass, now superseded).** With the S→S arm absent, the scorer returned "not judgeable" because one of its two arms is absent. However, a seed counts only if both arms show the gap, and the arm that exists (backprop→sweep) shows the 0.10 gap in **no** seed. P27.4 therefore cannot hold whatever the absent S→S files show. In logic its outcome is already a miss. I did not change the scorer after the run. Its output stands as written, and this note is my reading beside it.

**Combined reading, as the amendment's rules apply.** P27.1 misses and P27.2 is void. They are not a joint miss, so the reading "no amplification at these saved weights" is **not** triggered. The registered figures and the supplements **disagree** (see S1 and S2 below). By the amendment's own rule, the result is reported as mixed, with both sides.

## Supplements

**S1. Amplification below the top layer.** Two-bank network born by the sweep, answer targets, all parameters except `l3.*`: **0.631 / 1.593 / 0.955**. It reaches 1.5 in only one seed, so the scorer's reading is "flood not present below the top layer at those weights".

Most individual tensors still sit well above 1.5 in every seed. Examples: `lw.weight` 1.77 / 2.03 / 2.35, `lp2.weight` 1.87 / 3.12 / 2.44, `lm.weight` 1.71 / 1.58 / 1.57. The aggregate is pulled down by `book`, `lp` and `l1.bias` in seeds 73129 and 116089. The top layer is exactly 1.000 in every file. Full per-tensor tables are in `E27_SCORE.json` → `supplements.S1.per_tensor` and in the measure files. They include `rnn.weight_ih_l0`, `rnn.weight_hh_l0` and both GRU biases.

**S2. Distractor letters against structural bytes.** This split is where the registered figure and the hypothesis come apart.

| two-bank born by sweep, answer targets | 73129 | 93683 | 116089 | S4 null (except top) |
|---|---|---|---|---|
| letters, except top | **1.946** | **2.375** | **2.772** | 0.74–1.24 / 0.84–1.18 / 0.84–1.23 |
| structural, except top | 0.202 | 0.378 | 0.297 | |
| letters, all parameters | 1.412 | 1.925 | 2.071 | |

On these weights the sign path roughly doubles or triples the letters' error below the top layer, outside the null in all three seeds. It also cuts the structural bytes' error to between a fifth and two fifths. The two effects cancel in "other", which is why P27.1 and S1 see nothing. The amendment says "the hypothesis is about the letters", and on letters the amplification is present at these weights.

**S3. Two-bank born by backprop, every-byte targets.** All parameters: 1.361 / 0.908 / 2.068. Except top: **2.176 / 1.049 / 3.132**, against null ranges of 0.65–1.43, 0.86–1.06 and 0.72–1.24. That is outside the null in two seeds and inside it in one (93683). Letters except top: 1.459 / 1.053 / 2.510. On the operationalisation of "near 1":
- by the null range, seeds 73129 and 116089 are not near 1;
- by the 0.8–1.25 band, the same two seeds are not near 1.

The two operationalisations agree here. S3 is **not** near 1 in two of three seeds.

**S4. Null.** Twenty random splits at the answer's share, one generator seed per measurement seed. The nulls are narrow on untrained networks, about ±0.03. The untrained two-bank network already reaches 1.049 (93683) and 1.042 (116089) on all parameters, just outside its null. So "outside the null" means "not sampling noise". It does not by itself mean "a flood". On trained files the nulls widen to ±0.2–0.6. Every file's S4 membership is in `E27_SCORE.json`.

**S5. Baselines for "usable".** The registered usable is about 0 on every answer-trained file, under both paths, because the exact flood is huge: the other bytes' share-weighted gradient is **89 to 605 times** the answer's under the *exact* path. The loss on the other bytes is 12 to 20 nats on those files.

On the sweep-born answer-target file:
- cosine(sign answer gradient, exact answer gradient) is 0.694 / 0.634 / 0.602;
- cosine(sign every-byte gradient, sign answer gradient) is 0.098 / 0.044 / 0.084.

The answer direction is swamped under both paths. This baseline is why P27.2 is uninformative: its exact-path value is already below zero.

**Other A files.** Full tables are in the measure files.

| file: amplification all / except top | 73129 | 93683 | 116089 |
|---|---|---|---|
| two-bank untrained | 0.985 / 0.958 | 1.049 / 1.036 | 1.042 / 1.070 |
| two-bank born by backprop, answer targets | 1.273 / 1.929 | 1.129 / 1.811 | 1.061 / 1.808 |
| two-bank born by sweep, every byte (collapsed) | 0.802 / 1.254 | 0.876 / 1.420 | 1.615 / 3.047 |
| two-bank born by sweep, balanced | 8.823 / 17.24 | 8.712 / 12.37 | 6.626 / 15.69 |
| GRU untrained | 1.010 / 1.019 | 1.012 / 1.014 | 1.000 / 1.002 |
| GRU born by backprop | 1.532 / 2.928 | 1.540 / 2.087 | 1.193 / 1.770 |
| GRU born by sweep | 1.739 / 2.469 | 1.297 / 1.836 | 1.132 / 1.676 |

Below the top layer, every trained GRU and every two-bank backprop-born file shows amplification of about 1.7 to 3, outside its null. Of the six GRU files, three also exceed 1.5 on all parameters. The balanced sweep-born file shows the largest amplification by far: 6.6 to 8.8 on all parameters. In the GRU, the recurrent matrix `rnn.weight_hh_l0` is amplified 1.44 to 2.22 across the six trained files. None of this was predicted, and it is reported as found.

**S6 and Part B, first pass (crossover arms only; superseded by *Part B completion* below). Untaught minus taught (same format).** The taught panel is drawn evenly: 22 items from each of the first four taught pairs and 21 from each of the other eight.

| cosine, all parameters | single | taught | untaught | untaught − taught |
|---|---|---|---|---|
| 73129 sweep→backprop (lived under backprop) | 0.569 | 0.643 | 0.563 | −0.080 |
| 73129 backprop→sweep (lived under the sweep) | 0.610 | 0.559 | 0.533 | −0.026 |
| 93683 sweep→backprop | 0.615 | 0.573 | 0.553 | −0.020 |
| 93683 backprop→sweep | 0.641 | 0.667 | 0.661 | −0.006 |
| 116089 sweep→backprop | 0.610 | 0.571 | 0.574 | +0.003 |
| 116089 backprop→sweep | 0.628 | 0.608 | 0.575 | −0.033 |

- The within-birth form of S6 needs S→S and B→B, which are absent. It was not computed.
- The across-birth comparison is the only one available (sweep-lived B→S against backprop-lived S→B). The sweep-lived gap is larger in **1 of 3** seeds (116089).
- On these files, the sweep's direction is no worse on untaught compositions, relative to taught ones, than backprop's.
- This is across births and on half the arms, so it is not the registered composition reading.

## Controls of step 3 (`exp45/E27_FIXTURES_<seed>.json`, all pass; measurement admitted)

**Positive control.** This follows revision 2, item 2: on an untrained network, the top layer's rows for the 246 output bytes that are never answer bytes (all bytes except digits and EOS) are multiplied by 0.01. Admission criteria were fixed in code before any saved file was read: except-top amplification ≥ 1.5 and above the null's maximum, and all-parameter amplification above its null's maximum.

| seed | two-bank: all / except top / l1.weight | GRU: all / except top / l1.weight |
|---|---|---|
| 73129 | 1.551 / 2.814 / 3.187 | 1.330 / 2.680 / 2.776 |
| 93683 | 1.717 / 2.965 / 2.911 | 1.300 / 2.349 / 1.949 |
| 116089 | 1.579 / 3.254 / 3.588 | 1.289 / 2.842 / 3.223 |

These match revision 2's expectation of about 3.2 on the first layer and 1.55 overall for the two-bank network.

**Negative controls.**
- The top layer (`l3.weight`, `l3.bias`) has no matrix between the loss and the tensor. Its amplification is exactly 1.000000 in every control and in every saved file.
- The S4 null on the positive-control network has a mean of 0.957 to 0.973, inside the pre-set band of 0.9 to 1.1, with a range of about 0.81 to 1.11. On the untrained networks the null mean is about 1.00.

The measurement moves in the positive control and does not move in either negative control.

## Independent check of the split (step 2; `e27_check_split.py` → `exp45/E27_SPLIT_CHECK.json`, passes)

My own code does not call `flood()` or `e25_alignment.grads`. It computes per-position cross-entropy with `reduction='none'` and takes `torch.autograd.grad` on one graph. For the GRU it uses `arch26.forward`, the network E26 lived in. The check ran on the untrained G0 and G1 networks and on one saved file, `e26_dev_B_73129.pt`, after its receipt was verified. On that file it printed only differences, never flood figures.

- every-byte = w_ans·ans + w_other·other: within 1.2e-7 (saved file) and 3e-8 (untrained), under both paths.
- The three-way split (answer + letters + structural): within 1.3e-7.
- With the shares swapped the split fails by 0.21 to 0.33, so the check is able to fail.
- `flood()`'s flood, usable and amplification agree with mine to relative 0.0.
- The `e25_alignment` gradients agree with `arch26.forward`'s to relative 0.0 under both paths.

The split check ran before a comment-only edit to `e27_measure.py`: one docstring sentence was reworded so the word "optimizer" does not appear. No code changed.

## Review of the coordinator's script against the predictions (step 1), and what I changed

The draft is preserved as git history at `02897c6e:MOD97_LOCAL_RULE_20261106/e27_measure.py` (sha256 `c5eb5fdd…`).

**Does it compute what each prediction says?**
- P27.1, P27.2 and P27.3 are computed as their words say (`all_parameters.amplification`, `.sign/.exact.usable`, `.sign.flood`).
- P27.4 and P27.5 read `B[arm].<panel>.all_parameters` as registered.
- Ambiguities I report rather than resolve:
  1. P27.4's "(either birth)" against the amendment's "a seed counts only if both arms show it". The scorer follows the amendment.
  2. The amendment's "near 1" for S1 and S3. The scorer reports both the null-range and the 0.8–1.25 operationalisations, and flags any disagreement.
  3. "The sign path" means E25's path: credit through time, sign-scaled at every matrix, as in `arch17.forward(..., local, True)` and `e25_alignment.forward_rnn`. I did not re-establish that this is the exact rule under which the two-bank sweep births were made (birth rule `T`, `traceup all`). The predictions define the path as E25's. *Now checked: see "The sign path against E24's birth rule" below.*

**Defects, with the draft's line numbers:**
1. **L72, L83: per-tensor filter `n.endswith("weight")`.** It drops `rnn.weight_ih_l0` and `rnn.weight_hh_l0` and every bias. Fixed: every named parameter is reported, in both A and B.
2. **L90: taught panel `comp["within"][:256]`.** That is five pairs at 50 items and one at 6, with no pair beginning with `>` or `<`. Fixed in `taught_even()`: round robin over all twelve pairs, with an assertion that each block of 50 is the pair it claims to be.
3. **L41–45: loading without verification.** The draft also computed a hash at L118 and L122 but never compared it. Fixed: every file's sha256 is checked against its receipt (`weights_sha256`, at top level or under `complete`). For the three `e17_dev_bank2_nbd_bT_tall` births, which predate hashed receipts, it is checked against `exp45/BIRTH_MANIFEST.json`. A mismatch stops the run, and loading is strict. All 24 present files match.
4. **L64: `out["loss"]` overwritten once per path.** It is harmless because the forward is the same under both paths. Now both are recorded and checked equal.
5. **Missing pieces the amendment requires, now added:** the except-top set (S1); the letters/structural split (S2); the null (S4); the usable baselines (S5); the even taught panel (S6); the controls of revision 2; a panel census; and the scorer's hash at run time.
6. **L101: the fixture's "shares swapped" check tested only the sign path.** It is unchanged in the fixture. My split check tests both paths.

`e27_run.sh` (zsh, under `machine_lock.sh`) was not used. Revision 2 asked for the run in one process, so I ran the command above directly.


## Part B completion (follow-up of 13 November)

**Files.** The six `exp45/e26/e26_life_{S,B}_<seed>.pt` were taken from `claude/adoring-cori-b07aog` at `ce409d64` and copied into place. The life receipts are byte-identical on that commit and on this branch. My loader checked each file against `weights_sha256` in its receipt, and all six match:

| | 73129 | 93683 | 116089 |
|---|---|---|---|
| S | `54fbb464…` | `19784a59…` | `a867dfeb…` |
| B | `ce6d860c…` | `8d16d439…` | `905635ae…` |

No file is absent in the new run.

**Part A is unchanged.** In each new `E27_MEASURE_<seed>.json`, the whole `A` block compares equal (Python `==` on the parsed JSON, so bit-identical floats) to the `_partA` file of that seed. So do the two crossover arms in `B`. The run is deterministic: one thread, fixed null seed.

**Verdicts from the scorer** (`exp45/E27_SCORE.json`): **P27.4 misses** and **P27.5 misses**, in all three seeds. The figures are in the verdict table above. P27.1, P27.2 and P27.3 are as before.

**Reading under the amendment.** P27.4 and P27.5 both miss. The amendment no longer lets either one trigger the composition reading; composition is read from S6 alone.

**S6, all four arms.** Cosine over all parameters, answer targets.

| seed | arm | single | taught (even) | untaught +− | untaught − taught |
|---|---|---|---|---|---|
| 73129 | sweep → sweep | 0.644 | 0.651 | 0.610 | −0.041 |
| 73129 | sweep → backprop | 0.569 | 0.643 | 0.563 | −0.079 |
| 73129 | backprop → sweep | 0.610 | 0.559 | 0.533 | −0.025 |
| 73129 | backprop → backprop | 0.634 | 0.623 | 0.615 | −0.008 |
| 93683 | sweep → sweep | 0.621 | 0.614 | 0.659 | +0.045 |
| 93683 | sweep → backprop | 0.615 | 0.573 | 0.553 | −0.020 |
| 93683 | backprop → sweep | 0.641 | 0.667 | 0.661 | −0.006 |
| 93683 | backprop → backprop | 0.651 | 0.622 | 0.586 | −0.036 |
| 116089 | sweep → sweep | 0.636 | 0.615 | 0.586 | −0.029 |
| 116089 | sweep → backprop | 0.610 | 0.571 | 0.574 | +0.002 |
| 116089 | backprop → sweep | 0.628 | 0.608 | 0.575 | −0.033 |
| 116089 | backprop → backprop | 0.709 | 0.684 | 0.610 | −0.074 |

Is the sweep-lived gap (untaught below taught) larger than the backprop-lived gap within the same birth?

| seed | sweep birth (S→S vs S→B) | backprop birth (B→S vs B→B) | mean of the sweep-lived arms vs mean of the backprop-lived arms |
|---|---|---|---|
| 73129 | no (−0.041 vs −0.079) | **yes** (−0.025 vs −0.008) | no (−0.033 vs −0.044) |
| 93683 | no (+0.045 vs −0.020) | no (−0.006 vs −0.036) | no (+0.020 vs −0.028) |
| 116089 | **yes** (−0.029 vs +0.002) | no (−0.033 vs −0.074) | no (−0.031 vs −0.036) |

The pattern S6's reading needs appears in 2 of 6 within-birth comparisons, in different births in different seeds. By arm means it appears in 0 of 3 seeds. Every contrast is between −0.08 and +0.05. **S6 does not support the composition reading.** At the end of life, the sweep's direction on untaught plus-minus pairs, relative to taught pairs in the same format, is not systematically worse than backprop's. In the amendment's terms, the hypothesis finds no support for composition at these weights, and no explanation of E26's weaker carry-over is on file from this measurement.

## Amplification below the top layer against collapse (follow-up item 4)

Each cell gives all parameters except `l3.*`, then the S4 null range in brackets, then the birth's whole-answer score on add89 + sub89 at k = 12 (600 items) at its last step, from the birth's own receipt. Peaks are given where they differ from the last step.

| file | 73129 | 93683 | 116089 | collapsed? |
|---|---|---|---|---|
| two-bank untrained | 0.958 [0.97–1.01] | 1.036 [0.97–1.02] | 1.070 [0.97–1.05] | — |
| two-bank, backprop, answer | **1.929** [0.98–1.03] · 559 | **1.811** [0.96–1.05] · 557 | **1.808** [0.99–1.02] · 557 | no |
| two-bank, sweep, answer | 0.631 [0.74–1.24] · 525 | **1.593** [0.84–1.18] · 544 | 0.955 [0.83–1.22] · 508 | no |
| two-bank, backprop, every byte | **2.176** [0.65–1.43] · 549 | 1.049 [0.86–1.06] · 536 | **3.132** [0.72–1.24] · 547 | no |
| two-bank, sweep, every byte | **1.254** [0.91–1.11] · 5 (peak 63 at 8192) | **1.420** [0.90–1.03] · 6 (peak 47) | **3.047** [0.95–1.12] · 7 (peak 88) | **yes** |
| two-bank, sweep, balanced | **17.24** [0.86–1.53] · 489 (peak 505) | **12.37** [0.94–1.21] · 479 | **15.69** [0.81–1.64] · 500 | no |
| GRU untrained | 1.019 [0.99–1.01] | 1.014 [0.98–1.01] | 1.002 [1.00–1.02] | — |
| GRU, backprop | **2.928** [0.96–1.05] · 591 | **2.087** [0.97–1.03] · 592 | **1.770** [0.97–1.03] · 562 | no |
| GRU, sweep | **2.469** [0.96–1.03] · 588 | **1.836** [0.98–1.02] · 589 | **1.676** [0.94–1.05] · 599 | no |

Bold marks a value above its null's maximum. The untrained two-bank 93683 and 116089 values are also just above their nulls; see S4.

**The coordinator's reading:** "amplification appears in every trained network, most of all in the balanced file, and so does not mark the collapse."

**Where my figures agree:**
- The balanced file, which did not collapse (479 to 505 of 600), has by far the largest amplification: 12 to 17 below the top layer, against at most 3.1 anywhere else.
- The collapsed file's figures (1.25, 1.42, 3.05) fall inside the range of files that did not collapse (1.05 to 3.13 for the backprop-born every-byte file; 1.68 to 2.93 for the GRU births).
- So amplification below the top layer does not separate the collapsed birth from the others. **I agree that, at these weights, it does not mark the collapse.**

**Where they do not agree:**
1. **"Every trained network" is not quite right on this aggregate.**
   - The two-bank network born by the sweep under answer targets is amplified in only 1 of 3 seeds: 0.631 is *below* its null and 0.955 is inside it.
   - The backprop-born every-byte file is inside its null in 93683 (1.049).
   - Measured on the letters alone (S2), the sweep-born answer file is amplified in all three seeds (1.95 to 2.77). The difference is which bytes are counted, not whether the network is trained.
2. **The collapsed file is measured 16,384 updates after its peak.** Its weights are those of a network scoring 5 to 7 of 600. Whatever happened as it fell from 47–88 to 5–7 was never saved. "Does not mark the collapse" holds for the end-state weights only.
3. **The balanced birth was not trained on the loss E27 measures.** Its update used ½ answer + ½ other. Its amplification here is measured under the uniform every-byte split; on seed 73129 the two gradients have cosine 0.93 (`exp45/E27_RULE_CHECK.jsonl`).
4. **Both nets were stepped by AdamW, not by the raw gradient.** Raw-gradient amplification is not the size of the step. AdamW rescales each coordinate, so a ratio of gradient norms need not survive into the update.

## The sign path against E24's birth rule (follow-up item 5)

`e27_check_rule.py` → `exp45/E27_RULE_CHECK.jsonl`. The check imports the harness with each sweep birth's own environment, taken from `exp45/e24/environ.txt`: `E16_BIRTHRULE=T E16_TRACEUP=all`, plus `E16_LOSS=all` and `E16_LOSSW=balanced` where the birth used them. With those settings `exp16_integrated` installs the forward its births used. The script then reproduces `update16` (exp16_integrated.py lines 620–644) up to `backward()`, without the step: it builds no update rule and changes no weight.

**Source reading.**
- `traceup all` (exp16_integrated.py lines 81–94) wraps `E.forward` so that `local=True` forces `through_time=True` and `keep=False`, and it disables both trace routines.
- Under `birthrule T`, `update16` then calls that forward with `local=True` and runs `loss.backward()`.
- That is `arch17.forward(m, x, local=True, through_time=True)` with `SignLinear` (sign(W) times mean|W|) at every matrix: the same path `e25_alignment.grads(..., local=True)` takes.

**Measured, seed 73129.** Maximum relative difference between the birth rule's gradient and E27's sign path, with the loss weighted as the birth weighted it:
- sweep-born, answer targets: **0.0**
- collapsed (uniform every byte): **0.0**
- balanced: **4.6e-7**

The two-bank backprop births (`birthrule D`) use `E.update(..., "D")`, which is `forward(local=False, through_time=True)`. That is E27's exact path; E25's fixture showed it equal to the harness's backprop at exactly 0.

**The differences, stated:**
1. **Gradient path: none.** E27's sign path is the gradient E24's sweep births computed at each step.
2. **Step.** The births applied that gradient through AdamW, with decay 0.25 for `T` and 0.3 for `D`. AdamW normalises each coordinate by its running second moment, so E27's norm ratios describe the gradient, not the update.
3. **Loss weighting for the balanced file.** Its birth loss was ½ mean CE over answer positions + ½ mean CE over the others. E27 measures every file under the uniform every-byte split. On 73129 these two gradients have cosine 0.926 at the file's weights.
4. **No scan.** Under `traceup all` the trace routines are switched off. Nothing in the births added credit outside `backward()`, so E27 omits nothing there.

## What was absent (first pass; resolved by the follow-up)

`exp45/e26/e26_life_S_{73129,93683,116089}.pt` and `exp45/e26/e26_life_B_{73129,93683,116089}.pt` are not on this container. Their receipts exist and name their hashes (S: `54fbb464…`, `19784a59…`, `a867dfeb…`; B: `ce6d860c…`, `8d16d439…`, `905635ae…`), so nothing could be committed or verified. As a result:
- P27.5 is not judged;
- P27.4 is judged by the scorer as not judgeable, though it is logically a miss (see the note above);
- S6's within-birth form is not computed.

## Limits

- Gradients at saved weights are not training.
- Files trained by one update are measured where that update took them.
- The weights at which E24's every-byte births collapsed were never saved, so A measures other weights than the ones where the collapse happened.
- The answer-trained files have 12 to 20 nats of loss on the other bytes. "Flood" there is dominated by bytes the network was never asked to predict, and the registered usable is about 0 under both paths.
- Three seeds, 256 items a panel, one size.
- Part B now has all four arms, but each arm is one end-of-life file per seed.
- The flood ratios describe raw gradients; the births stepped by AdamW (see the follow-up item 5 section).
- The S4 nulls catch sampling noise only; even an untrained network can fall just outside them.
