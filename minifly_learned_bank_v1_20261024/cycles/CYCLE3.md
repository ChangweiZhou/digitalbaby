# Cycle 3: fixes, dress rehearsal, lock (24 Oct 2026)

## Refine
* Wrote **DESIGN.md Amendment A1** (section 11) from the cycle-2 evidence:
  * primary task T-PAR;
  * binary SHUF (orbit-level label permutation at about half agreement);
  * balanced held accuracy of the learned bank as the primary metric;
  * held transfer gap and all-prompt kernel-target alignment as the kernel meters;
  * settings H 128, eta0 0.03, lam 1e-3, 150 passes;
  * decision rule with a 0.20 margin, three gates and a four-world verdict.
* `summarize.py` applies that rule, including the T-COPY measurement-chain gate (every T-COPY arm's system held accuracy ≥ 0.30).
* `run.py` refuses any main world (M*) unless every hash in `LOCK.json` matches the code on disk and the command-line settings equal the locked ones. Checked:
  * an M1 run with off-lock settings is refused before any history starts;
  * locked settings are accepted;
  * an unknown main world is refused.
* `main.sh` is the single main command. `WORLDS=D3 TAG=c3_rehearsal sh main.sh` is the identical protocol on a development world.
* `make_lock.py` writes `LOCK.json`, which records:
  * hashes of `lb1.py`, `data.py`, `run.py`, `summarize.py`, `main.sh` and the imported native `fastmem.py`;
  * settings, worlds, the run roster, the decision rule and the environment (Python 3.11, numpy 2.2.6, scipy 1.14.1, numba 0.61.2).

## Audit
* `checks.py` now has 41 checks, all passing (`CYCLE3_CHECKS.json`). The new ones cover:
  * PAR answers;
  * PAR SHUF labels cover exactly the training pairs, are orbit-consistent and agree with the truth on 36 / 37 / 37 prompts (D1 / D2 / D3);
  * MOD SHUF is unchanged from cycle 1.
* Read-through of the rule against the stored outputs. The balanced accuracies `summarize.py` prints were recomputed by hand from the per-prompt outputs for D3 LEARN, FROZEN and SHUF, and match.
* The rehearsal was run after the lock with unchanged code: hashes were verified after it and match `LOCK.json`.

## Dress rehearsal: exact main protocol on development world D3 (`../results/c3_rehearsal`, log `c3_rehearsal.log`)
| Task | Arm | Balanced held, learned bank | Held, learned bank | Held, system | Taught, learned bank | Kernel alignment, birth to end | Held transfer gap, birth to end |
|---|---|---:|---:|---:|---:|---:|---:|
| PAR | BASE | - | - | 0.407 | - | - | - |
| PAR | FROZEN | 0.361 | 0.333 | 0.481 | 1.000 | 0.048 to 0.048 | -0.002 to -0.002 |
| PAR | **LEARN** | **0.944** | 0.963 | 0.852 | 1.000 | 0.048 to **0.274** | -0.002 to **+0.044** |
| PAR | SHUF | 0.500 | 0.518 | 0.444 | 0.931 | 0.048 to 0.030 | -0.002 to 0.000 |
| PAR | ORACLE | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.989 |
| COPY | BASE | - | - | 0.963 | - | - | - |
| COPY | FROZEN | 0.633 | 0.518 | 0.815 | 1.000 | 0.429 to 0.429 | 0.109 to 0.109 |
| COPY | LEARN | 1.000 | 1.000 | 1.000 | 1.000 | 0.429 to 0.814 | 0.109 to 0.310 |

Rule applied to D3: every clause and gate passes.
* LEARN 0.944 against FROZEN 0.361 + 0.20 and SHUF 0.500 + 0.20.
* LEARN's gap and alignment both rose.
* ORACLE scored 1.00; FROZEN's codes were unchanged; T-COPY system held was 0.82 to 1.00.

D3 was used for surrogate tuning in cycle 2 but had not been run natively before this rehearsal. It is therefore a second native development confirmation, not evidence for the main question.

LEARN's PAR trajectory on D3 (learned bank):

| Checkpoint | Held | Taught | Kernel alignment |
|---|---:|---:|---:|
| Pass 25 | 0.67 | 0.85 | 0.13 |
| Pass 50 | 0.85 | 1.00 | 0.21 |
| Pass 75 | 1.00 | 0.99 | 0.24 |
| Pass 100 | 1.00 | 1.00 | 0.27 |
| Pass 150 | 0.96 | 1.00 | 0.27 |
| After 1 h rest | 0.96 | 1.00 | 0.27 |

Other trajectory features:
* Code overlap rose from 0.27 to about 0.65 by pass 25 (the common-mode collapse noted in cycle 2), then stayed flat.
* W1's norm went from 11.3 to 14.2 and settled at 13.7.

## Exercise
* The whole pipeline ran from one command (`main.sh`) to the JSON histories and then `summarize.py`, giving the verdict and `SUMMARY.json`.
* Every history has 8 checkpoints (birth, passes 25 to 150, and after the 1 h rest).
* Cost:
  * 0.125 CPU-s per record with 32 compartments, 0.058 for BASE;
  * 1,420 to 1,450 CPU-s per history (660 for BASE);
  * peak RSS 245 MB;
  * the rehearsal took about 62 minutes of wall time on 4 cores.
* Main-run forecast: 32 histories, about 11 CPU-hours, about 3 hours of wall time.

## Lock
* No change was needed after the rehearsal, so `LOCK.json` stands as written before it.
* Main worlds M1 to M4 have not been run, probed or summarised. Their splits and seeds are derived from their names by `data.make_world`.
