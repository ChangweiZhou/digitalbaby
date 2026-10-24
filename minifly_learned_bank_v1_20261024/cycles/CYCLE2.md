# Cycle 2: development-world test runs, testbed audit, tuning, cost (24 Oct 2026)

Development worlds only (D1, D2, D3). No main world (M1 to M4) has been generated, run or looked at. Logs are in `c2_logs/`; native histories are in `../results/c2a` and `../results/c2b`.

## 1. Native test run on the design's primary task (T-MOD, D1, 30 passes, eta0 = 0.003)
| Arm | Held, learned bank | Taught, learned bank | Held, system | Held transfer gap, birth to end | Code drift (Jaccard per 10 passes) |
|---|---:|---:|---:|---:|---:|
| ORACLE | 1.000 | 1.000 | 0.885 | 0.957 (class-coded) | 1.00 |
| FROZEN | 0.077 | 1.000 | 0.000 | -0.023 to -0.023 | 1.00 |
| LEARN | 0.000 | 0.689 | 0.038 | -0.023 to -0.027 | 0.51 |
| SHUF | 0.115 | 0.797 | 0.154 | -0.023 to -0.022 | 0.48 |
| BASE | - | - | 0.038 | - | - |

* The plumbing works: with class-coded addresses (ORACLE) the native learned bank answers every held prompt, and its answer survives the 1 h rest.
* LEARN did not help on T-MOD. Its codes drifted (Jaccard 0.51 between checkpoints) and that cost taught accuracy (0.69 against 1.00 for FROZEN).

## 2. Testbed audit: T-MOD cannot test condition 1
An evaluator-side network was given the same FE-PERM traces and splits (`diag_mlp.py`; never a candidate): input → ReLU layer (H = 128) → linear readout to 8 answer bits. Held accuracy is shown for D2 / D3.

| Task | FROZEN (random features) | BP (exact gradient) | DFA (the candidate's local rule) |
|---|---:|---:|---:|
| MOD (last digit of a+b), 400 epochs | 0.00 / 0.00 | 0.00 / 0.11 | 0.00 / 0.11 |
| MOD, weight decay 0.01, 3,000 epochs | - | 0.16 / 0.19 | 0.08 / 0.04 |
| CARRY (a+b ≥ 10) | 0.84 / 0.89 | 0.80 / 0.93 | 0.80 / 0.93 |
| CMP (a > b) | 0.77 / 0.73 | 0.73 / 0.85 | 0.77 / 0.85 |
| **PAR (parity of a+b)**, 1,000 epochs | **0.24 / 0.22** | **1.00 / 1.00** | **1.00 / 1.00** |
| MOD3, 1,000 epochs | 0.16 / 0.37 | 0.52 / 0.82 | 0.52 / 0.96 |
| MOD5, 1,000 epochs | 0.12 / 0.15 | 0.84 / 0.52 | 0.44 / 0.44 |

* **T-MOD tells us nothing about local learning.** With 74 taught prompts not even exact gradient descent generalises last-digit addition from this input code. A failure there cannot be blamed on the local rule or on the native stores.
* CARRY and CMP are additively separable, so frozen random features already generalise them. They cannot show feature learning.
* **Parity of a+b (PAR) is a valid test.**
  * Frozen random features generalise below chance: a held prompt resembles taught prompts that share a digit, and those usually have the opposite parity.
  * Exact gradient and the local three-factor rule both reach 1.00.
  * Parity is an interaction (an XOR of the operands' parities), so a kernel that is not learned cannot solve it.
* DFA on PAR needs time: at 30 / 100 / 300 epochs, held accuracy is 0.36 / 0.84 / 0.96 on D2 and 0.78 / 1.00 / 1.00 on D3.

## 3. Consequences for the protocol (to be locked in cycle 3)
1. **Primary task becomes T-PAR**: prompt `a+b=`, answer byte `'0'` or `'1'` for (a+b) mod 2. The same orbit splits are used. T-MOD is dropped from the main run; the evidence above is the reason. T-COPY is kept as a secondary task (FROZEN, LEARN only), not part of the decision.
2. **SHUF for a binary task.** A derangement of binary labels flips every label, which teaches anti-parity: just as structured, and not a control. For PAR, SHUF instead permutes labels among training orbits (both orientations of an orbit keep one label), choosing the permutation that agrees with the truth on as close to half the orbits as possible. Agreement is 36 of 74, 37 of 75 and 37 of 73 prompts on D1, D2 and D3. T-MOD / T-COPY keep the derangement.
3. **Primary metric is balanced held accuracy** (mean per-answer recall). Held answer classes are unbalanced: 8 of 25 held prompts are `'1'` in D2, 18 of 27 in D3. A constant answer would therefore score up to 0.68 plain accuracy, but exactly 0.50 balanced.

## 4. Surrogate calibration and tuning (D2, D3)
`surrogate.py` keeps the same traces, layer, rule and PN→KC top-k, but replaces the native stores with a linear readout on the KC code.
* **Readout speed calibrated to native.** Readout lr 0.005 reproduces native FROZEN's taught curve on T-MOD D1: 0.70 / 0.99 / 1.00 against native 0.69 / 0.95 / 1.00 at passes 10 / 20 / 30. All later surrogate runs use 0.005.
* **60 passes are too few.** At 60 passes no eta0 in {0.001, 0.003, 0.01} helps PAR (LEARN balanced held 0.25 to 0.75, against FROZEN 0.43 / 0.47). The KC code is a high-capacity memoriser: taught prompts are memorised by about pass 20, after which the residual that drives the layer is small.
* **With longer training the effect appears.** At 300 passes, LEARN balanced held:

  | eta0 | D2 at pass 90 | D2 final | D3 |
  |---|---:|---:|---|
  | 0.03 | 0.91 | 0.97 | 1.00 by pass 60 |
  | 0.01 | — | 0.82 | 1.00 |

  At 300 passes FROZEN is 0.36 / 0.50 and SHUF (eta0 0.03) 0.21 / 0.44 on D2 / D3.
* **The native store keeps the informative residual alive** (`diag_margins.py`, native FROZEN, D2 PAR).
  * Bits that never vary between `'0'` and `'1'` saturate: median |margin| 3.2 to 5.3, mean |residual| 0.05 falling to 0.008.
  * The parity bit's margin grows slowly (median 0.34 → 1.43 over passes 5 to 20), and its residual stays at 0.41 → 0.22.
  * So the layer's third factor is dominated by the informative bit after the first few passes.
* **Settings chosen: eta0 = 0.03, lam = 1e-3, H = 128, 150 passes, probes every 25 passes.** These are the design's H and lam; the learning rate is 10 times the provisional 0.003, and the run is longer.

## 5. Native confirmation (T-PAR, D2, 150 passes, eta0 = 0.03; final = after the 1 h rest)
| Arm | Balanced held, learned bank | Held, learned bank | Held, system | Taught, learned bank | Kernel-target alignment, birth to end | Held transfer gap, birth to end | Mean code overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| FROZEN | 0.331 | 0.36 | 0.44 | 1.00 | 0.046 to 0.046 | -0.009 to -0.009 | 0.24 |
| LEARN | **0.824** | 0.76 | 0.68 | 1.00 | 0.046 to **0.411** | -0.009 to **+0.025** | 0.78 |
| SHUF | 0.357 | 0.44 | 0.36 | 1.00 | 0.046 to 0.029 | -0.009 to -0.005 | 0.77 |

LEARN's plain held accuracy at passes 50 / 75 / 100 / 125 / 150 was 0.72 / 0.80 / 0.88 / 0.72 / 0.76: above FROZEN at every checkpoint, but not monotone. Learned-bank kernel alignment rose at every checkpoint: 0.26, 0.33, 0.36, 0.39, 0.41.

On the native stores, a locally trained feature layer turns the learned bank's address kernel task-aligned, and the held-out answers follow, on one development world. This is not yet a result; the main run on fresh worlds decides.

## 6. Other findings
* **T-COPY (D1, 30 passes, eta0 0.003, native).**
  * LEARN: held 0.731 (balanced 0.717); FROZEN: 0.577 (balanced 0.625).
  * LEARN's alignment rose from 0.423 to 0.554 and its transfer gap from 0.093 to 0.127. Taught accuracy was 1.00 in both arms.
  * The learned layer also sharpens a separable kernel without costing memorisation.
* **Code collapse.**
  * In LEARN and SHUF, mean pairwise overlap of learned-bank codes rose from 0.24 to about 0.78.
  * The cause: answer bits that never vary give the same residual sign for every prompt, so the broadcast third factor moves every prompt's hidden vector the same way.
  * Memorisation was slowed but not prevented: SHUF taught 0.80 at pass 75, 1.00 at pass 150.
* **Centred third factor (variant tested, not adopted).** The error was replaced by its deviation from a running mean (tau = 75 records). Surrogate, 150 passes:

  | Rule | Overlap D2 / D3 | Balanced held D2 / D3 |
  |---|---:|---:|
  | Uncentred (design rule) | 0.69 / 0.62 | 0.94 / 1.00 |
  | Centred | 0.42 / 0.46 | 0.88 / 1.00 |

  It reduces collapse but does not improve the primary metric, so the design's rule is kept. Overlap stays recorded as a meter, and the option remains in `lb1.Layer` (off by default).
* **Readout decay** (`wd_r`, a stand-in for faster native forgetting) prevented memorisation without helping T-MOD (logs `sur_wd*.log`). Not pursued.

## 7. Audit of the cycle-2 code changes
* `data.py`: `answer('PAR')` and `shuffled_labels` added. The derangement is unchanged for MOD / COPY. Checked for D1 to D3: SHUF labels agree with the truth on 36 / 37 / 37 prompts, and both orientations of every training orbit get the same label.
* `run.py` / `surrogate.py` call `shuffled_labels`. `summarize.py` computes balanced held accuracy from the stored per-prompt outputs and applies the PAR decision rule proposed below.
* `lb1.Layer` takes an optional `centre_tau`, default 0, which gives exactly the cycle-1 rule. `checks.py` reran: 34/34 pass.

## 8. Cost (native, measured on D2)
* 0.126 CPU-s per training record (32 native compartments) and 6.4 to 7.1 CPU-s per 100-prompt probe; peak RSS 244 MB.
* A 150-pass history: 11,250 records, about 1,480 CPU-s.
* Proposed main run: 4 worlds × (PAR: BASE, FROZEN, LEARN, SHUF, ORACLE; COPY: FROZEN, LEARN). That is about 26 histories at about 25 CPU-minutes each, a little over 2.5 hours of wall time on 4 cores.

## 9. Proposed decision rule (to lock in cycle 3)
Metric: balanced held accuracy of the learned bank on T-PAR at the final checkpoint (after the 1 h rest). In each main world M1 to M4:
* LEARN ≥ FROZEN + 0.20, and LEARN ≥ SHUF + 0.20;
* LEARN's held transfer gap and its kernel-target alignment both end above their birth values;
* gates: ORACLE ≥ 0.90, and FROZEN's learned-bank codes are unchanged (sanity check).

Verdict:
* **supported** if all four worlds pass;
* **partial** if three pass;
* **not supported** otherwise.

The descriptive report adds: LEARN balanced held ≥ 0.75, the system's held accuracy, code overlap, and T-COPY.
