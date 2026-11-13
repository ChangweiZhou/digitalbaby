# E29 Part A: how the trained GRUs use their update gates. Worker's report

13 November 2026, Opus 5.5 worker, branch `claude/wizardly-franklin-sr2l6q`. Brief: `OPUS_TASK_E29_SLOW_UNITS_AND_DELAYS_20261113.md`, revision 2, commit a5cb6259 on the default branch. **Part A only.** Part B is held and was not started. This is a read-only measurement on saved weights: nothing was trained and no existing file was edited. No E-level and no learning-core claim is made. **The verdicts below are my mechanical readings. The coordinator rules.**

## Order of work, as asked

1. **Host.** The host had Python 3.13.18 and no torch. I installed `torch==2.15.0+cpu` with `python3.13 -m pip install "torch==2.15.0+cpu" --index-url https://download.pytorch.org/whl/cpu`. pip reported "Successfully installed ... torch-2.15.0+cpu" (along with filelock 4.2.4, fsspec 2026.9.0, mpmath 1.3.0, networkx 3.7, setuptools 78.1.0 and sympy 1.14.0) and exited 0. All runs used one thread.
2. **E26 life weights.** I took the six files `exp45/e26/e26_life_{S,B}_{73129,93683,116089}.pt` from branch `claude/adoring-cori-b07aog` at commit ce409d64 using `git checkout ce409d64 -- <paths>`. They were added as new files. Each one's sha256 matched `complete.weights_sha256` in its receipt, as checked below.
3. **`e29_gates.py`** is a new file. It reruns arch26's GRU step line for line, keeping the update gate z, and asserts that every state h is bit-identical (`torch.equal`) to what `arch26.forward(m, x, local=False, through_time=True)` returns. This held on every panel of every file. The arithmetic is the weights' own float32.
4. **Hand-built check first** (`E29_HAND_CHECK.json`, passed). It ran before any trained file was opened. Unit 0 has z = 1 on letters and z = 0 elsewhere, and is a gated holder with retention 1.0. Unit 1 has z = 0.5 everywhere: it is not a holder, and its retention is 0.5^k exactly (2.44e-4 at k = 12, 9.54e-7 at k = 20). As a negative control, an inverted unit 0 is not a holder and has retention 0. The net contains exactly one holder.
5. **Predictions** were copied unchanged into `E29_PREDICTIONS.md` at 15:28 EST, then committed (1008de25) and pushed before any receipt or trained weight file was opened. The text matches the brief byte for byte, apart from the extract's trailing newline.
6. **Measurement** covered 21 files, with about 33 CPU-s in total. Every saved file's sha256 was checked against its receipt before loading (births: `weights_sha256`; lives: `complete.weights_sha256`). The untrained G1 has no file: it is `arch26.new_net(seed)`, and its tensors hash is recorded.

## Choices the brief left open (stated here so they can be overruled)

- **Position 0 (BOS, h_prev = 0) is excluded** from "lowest z off letters" and from the mean z, because z there cannot hold or rewrite anything. It is not among the brief's six roles. The variant that includes it is recorded per unit. In every file the rewrite-role counts are identical with and without it.
- **Pooling.** Hold share and the lowest z are pooled over all 300 items of a panel. The lowest z is the minimum over every non-letter position of every item.
- **Composition panel.** For the lives this is `comp["+-"] + comp["-+"]` from `exp16_integrated.panels(seed)`, 1,200 items. **It has no distractor letters** (E16_COMPGAP = 0), so hold share, retention and gated holders are undefined there and are recorded as None. Only the lowest z, its role and the mean z are recorded.
- **k is not named in P29.1 or P29.2.** Both k = 12 and k = 20 are reported. They agree in every case.
- **Life arms.** XY means birth X, life update Y. SS = E26 life S, BB = E26 life B, SB and BS = E26-X. "Updated by backprop during the life" means BB and SB; "by the sweep" means SS and BS.

## One property of the measure the coordinator should see

The condition **lowest z off letters ≤ 0.1 holds for 251–256 of 256 units in every trained file**, and for 0 units untrained. As the audit said, every unit is rewritten somewhere. In the trained files, gated holder therefore reduces almost entirely to **hold share on letters ≥ 0.9**. Units meeting that condition: births 95–118, lives 122–161.

## Results (share of 256 units, %; k = 12 / k = 20)

| file | gated holders | retention ≥ 1/2 | retention quartiles at k = 20 | mean z (k = 20) |
|---|---|---|---|---|
| untrained 73129 / 93683 / 116089 | 0 / 0 each | 0 / 0 each | ≈ 0, 0, 0 | 0.497 / 0.498 / 0.501 |
| **birth S 73129 (seen, not judged)** | 37.1 / 37.5 | 41.8 / 36.7 | 0.000, 0.154, 0.838 | 0.587 |
| **birth B 73129 (seen, not judged)** | 37.1 / 37.5 | 40.6 / 39.1 | 0.000, 0.034, 0.931 | 0.502 |
| birth S 93683 | 44.9 / 44.9 | 50.4 / 44.9 | 0.000, 0.382, 0.886 | 0.605 |
| birth B 93683 | 46.1 / 46.1 | 50.4 / 47.7 | 0.000, 0.413, 0.969 | 0.571 |
| birth S 116089 | 43.8 / 43.4 | 48.8 / 46.1 | 0.000, 0.345, 0.889 | 0.549 |
| birth B 116089 | 41.0 / 41.0 | 45.3 / 44.1 | 0.000, 0.112, 0.916 | 0.505 |
| life SS 73129 | 54.3 / 54.3 | 56.6 / 51.6 | 0.000, 0.526, 0.915 | 0.631 |
| life BB 73129 | 52.7 / 52.7 | 54.7 / 52.7 | 0.000, 0.550, 0.916 | 0.589 |
| life SB 73129 | 53.5 / 53.5 | 57.0 / 54.7 | 0.000, 0.596, 0.918 | 0.621 |
| life BS 73129 | 46.5 / 46.9 | 47.7 / 43.4 | 0.000, 0.300, 0.845 | 0.587 |
| life SS 93683 | 62.1 / 62.1 | 64.1 / 60.2 | 0.000, 0.724, 0.956 | 0.646 |
| life BB 93683 | 62.5 / 62.9 | 64.8 / 62.9 | 0.000, 0.852, 0.967 | 0.618 |
| life SB 93683 | 56.2 / 56.2 | 59.0 / 54.7 | 0.000, 0.751, 0.965 | 0.595 |
| life BS 93683 | 60.5 / 60.5 | 62.5 / 57.8 | 0.000, 0.737, 0.951 | 0.628 |
| life SS 116089 | 47.3 / 46.5 | 51.6 / 46.5 | 0.000, 0.413, 0.927 | 0.590 |
| life BB 116089 | 52.3 / 52.3 | 55.9 / 54.3 | 0.000, 0.622, 0.945 | 0.563 |
| life SB 116089 | 54.3 / 55.1 | 56.6 / 54.3 | 0.000, 0.674, 0.961 | 0.568 |
| life BS 116089 | 47.3 / 48.0 | 50.4 / 45.7 | 0.000, 0.385, 0.862 | 0.579 |

The quartiles are rounded to three places. A lower quartile of "0.000" means below 5e-5. Per-role splits, per-unit arrays and the composition panel are in `e29_<tag>.json` and `E29_GATES_SUMMARY.json`.

**Split by rewrite role (k = 20, units / of which gated holders).** Operand-digit positions rewrite the most units in every file: births 132–162, lives 93–149. Among the units rewritten at a digit, 34–47% are holders in the births and 28–77% in the lives (BS 73129 is lowest, at 26 of 93). Among the units rewritten at the answer, 52–82% are holders across births and lives. Among the units rewritten at `=`, 0–15% are holders in the births and 0–55% in the lives. The task code rewrites more than 20 units only in SS 93683 (54), BS 93683 (55), SS 116089 (21) and BS 116089 (26), all sweep-updated lives. In the untrained networks the digit is likewise the commonest rewrite position (172–191 units), with no holders.

**Composition panel (lives; no letters).** Mean z is 0.39–0.48. The lowest z falls at a digit for 136–162 units and at an operator for 55–92 units.

## Seed 73129 births: reported, not judged

Birth S has 95 gated holders at k = 12 and 96 at k = 20 (37.1 / 37.5%). Birth B has the same counts, 95 and 96, but they are different units with different retention distributions (sha256 85ed1de2… and 24f2d7e1…). The share with retention ≥ 1/2 is 41.8 / 36.7 for S and 40.6 / 39.1 for B. These two files are left out of every reading below.

## Mechanical readings of the predictions (unseen files only; 5-point rule applied)

- **P29.1** (≥ 15% gated holders in both births of both unseen seeds): 44.9, 46.1, 43.8 and 41.0 at k = 12; 44.9, 46.1, 43.4 and 41.0 at k = 20. **Holds** at both k.
- **P29.2** (sweep-born < backprop-born, in both unseen seeds): in 93683, S 44.9 vs B 46.1, a difference of 1.2 points, under 5, so it is not read as a difference. In 116089, S 43.8 vs B 41.0: the sweep-born network has more holders. The figures are the same at k = 20. **Misses.**
- **P29.3** (retention ≥ 1/2 at k = 20; backprop-updated BB and SB above sweep-updated SS and BS, in ≥ 2 of 3 seeds). **The outcome depends on how "the two ... than the two" is read, so I do not settle it.**

  | seed | BB | SB | SS | BS | each above each, by ≥ 5 | pair means, by ≥ 5 |
  |---|---|---|---|---|---|---|
  | 73129 | 52.7 | 54.7 | 51.6 | 43.4 | no (BB − SS = 1.1) | yes (53.7 vs 47.5, +6.2) |
  | 93683 | 62.9 | 54.7 | 60.2 | 57.8 | no (SB below both) | no (58.8 vs 59.0) |
  | 116089 | 54.3 | 54.3 | 46.5 | 45.7 | yes (+7.8) | yes (54.3 vs 46.1, +8.2) |

  Each above each: 1 of 3 seeds, **misses**. Pair means: 2 of 3 seeds, **holds**. The coordinator should fix the reading.

The brief's fixed readings apply: P29.2 or P29.3 holding would show a difference between networks and nothing more, and P29.1 holding would show that training creates holding units where the untrained network has none (0 of 256 in all three seeds).

## Limits

- One panel family (add89) carries the letter measures. The composition panel has no letters, so it cannot speak to holding.
- The thresholds 0.9 / 0.1 / 0.5 are the brief's. No sensitivity sweep was run.
- The minimum is pooled over 300 items, which makes the "rewritten" condition nearly vacuous in trained files (see above).
- The untrained hash is of tensors, not of a saved file, because no saved untrained file exists.
- This is a measurement by one worker. Nobody independent has reviewed it.
