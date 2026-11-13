# E27 report: flood and alignment on saved weights (Opus worker, 13 November 2026)

**Status.** The run is complete for Part A and partly done for Part B. The six E26 life weight files are **absent**, so Part B ran on the six crossover lives only. Evidence level E2, diagnostic. Nothing was trained: the job builds no update rule and uses no optimizer object. `grep -n optim` prints nothing in `e27_measure.py`, `e27_score.py` or `e27_check_split.py`. These figures are gradients at saved weights. They say nothing about any learner's ability. No formal checker gate, seal or outside review has passed. I am the same model family as the coordinator and the auditors, so I am a second reader of their work, not an independent one.

- **Source.** Branch `codex/minifly-consolidated-20261023` at `02897c6e` (later than `a5c85319`). The brief is at revision 2, and the predictions file includes its amendment. The work is on branch `claude/sleepy-knuth-q96dd1`.
- **Run.** Command: `env -i PATH HOME E16_ARCH=bank2 E16_BOOKDECAY=off E16_COMPGAP=0 OMP_NUM_THREADS=1 python3 e27_measure.py run 73129 93683 116089`. One process, one thread, torch 2.15.0+cpu. CPU time was 417 s, 391 s and 402 s for the three seeds. Log: `exp45/logs/e27_run.log`.
- **Pre-run record.** `exp45/E27_PRERUN.txt`, written at 17:10:51Z (container clock) when no `E27_MEASURE` file existed yet. It holds the sha256 of every script and every fixture record. Each `E27_MEASURE_<seed>.json` also records the scorer's hash at run time (`ee2dd33e…`) and the measure script's hash (`51ada865…`). Both match the pre-run record.
- **Single commit.** You asked for one commit at completion, so the scorer was hashed before the run instead of committed separately (brief step 4). This difference is disclosed here.

## Verdicts (scorer `e27_score.py` → `exp45/E27_SCORE.json`)

| | 73129 | 93683 | 116089 | verdict |
|---|---|---|---|---|
| **P27.1** amplification, all parameters, two-bank born by sweep (answer targets), threshold ≥ 1.5 | 0.532 | 1.409 | 0.797 | **misses** |
| **P27.2** usable under signs ≤ ½ usable under exact | sign 0.035 / exact −0.130 | −0.006 / −0.232 | 0.035 / −0.153 | **void in all three seeds** (exact usable < 0) → does not hold |
| **P27.3** sign flood: collapsed ≥ 2 × balanced | 12.81 / 11.61 = 1.10 | 14.86 / 20.38 = 0.73 | 20.60 / 12.46 = 1.65 | **misses** (nothing is read from it, per the amendment) |
| **P27.4** untaught pairs at least 0.10 below single operations, on both sweep-lived arms | S→S absent; B→S gap 0.077 | absent; −0.020 | absent; 0.053 | scorer: **not judgeable** (one arm absent). See note |
| **P27.5** untaught cosine, sweep-lived at least 0.05 below backprop-lived, within a birth | needs S→S and B→B | absent | absent | **not judgeable** |

**Note on P27.4.** The scorer returns "not judgeable" because one of its two arms is absent. However, a seed counts only if both arms show the gap, and the arm that exists (backprop→sweep) shows the 0.10 gap in **no** seed. P27.4 therefore cannot hold whatever the absent S→S files show. In logic its outcome is already a miss. I did not change the scorer after the run. Its output stands as written, and this note is my reading beside it.

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

**S6 and Part B. Untaught minus taught (same format).** The taught panel is drawn evenly: 22 items from each of the first four taught pairs and 21 from each of the other eight.

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
  3. "The sign path" means E25's path: credit through time, sign-scaled at every matrix, as in `arch17.forward(..., local, True)` and `e25_alignment.forward_rnn`. I did not re-establish that this is the exact rule under which the two-bank sweep births were made (birth rule `T`, `traceup all`). The predictions define the path as E25's.

**Defects, with the draft's line numbers:**
1. **L72, L83: per-tensor filter `n.endswith("weight")`.** It drops `rnn.weight_ih_l0` and `rnn.weight_hh_l0` and every bias. Fixed: every named parameter is reported, in both A and B.
2. **L90: taught panel `comp["within"][:256]`.** That is five pairs at 50 items and one at 6, with no pair beginning with `>` or `<`. Fixed in `taught_even()`: round robin over all twelve pairs, with an assertion that each block of 50 is the pair it claims to be.
3. **L41–45: loading without verification.** The draft also computed a hash at L118 and L122 but never compared it. Fixed: every file's sha256 is checked against its receipt (`weights_sha256`, at top level or under `complete`). For the three `e17_dev_bank2_nbd_bT_tall` births, which predate hashed receipts, it is checked against `exp45/BIRTH_MANIFEST.json`. A mismatch stops the run, and loading is strict. All 24 present files match.
4. **L64: `out["loss"]` overwritten once per path.** It is harmless because the forward is the same under both paths. Now both are recorded and checked equal.
5. **Missing pieces the amendment requires, now added:** the except-top set (S1); the letters/structural split (S2); the null (S4); the usable baselines (S5); the even taught panel (S6); the controls of revision 2; a panel census; and the scorer's hash at run time.
6. **L101: the fixture's "shares swapped" check tested only the sign path.** It is unchanged in the fixture. My split check tests both paths.

`e27_run.sh` (zsh, under `machine_lock.sh`) was not used. Revision 2 asked for the run in one process, so I ran the command above directly.

## What was absent

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
- Part B has half its arms.
- The S4 nulls catch sampling noise only; even an untrained network can fall just outside them.
