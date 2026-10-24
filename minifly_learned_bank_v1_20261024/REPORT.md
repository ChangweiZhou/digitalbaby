# LEARNED-BANK V1: report (24 Oct 2026)

**Question (condition 1).** Can a feature layer trained only by a local three-factor rule turn the address kernel of a native Full151 memory bank task-aligned, so that the bank answers held-out prompts that a fixed random kernel cannot?

**Answer: partial support (3 of 4 fresh worlds), by the locked rule.**
* In every main world:
  * the learned bank's kernel became task-aligned (kernel-target alignment rose 3.8 to 7.2 times);
  * LEARN beat frozen random features by 0.29 to 0.61 in balanced held accuracy.
* In three worlds LEARN also beat the shuffled-label control by the pre-registered 0.20 margin. In M1 it did not: LEARN 0.726 against SHUF 0.622.
* The post-hoc fail-closed check agrees with the locked summary.

## Protocol
* Design: `DESIGN.md` sections 1 to 10, with Amendment A1 (section 11) adopted from development-world evidence before the lock.
* Lock: `LOCK.json`, commit 1ce1eadb. Command: `sh main.sh`.
* Main worlds M1 to M4 were generated and run only after the lock. `run.py` refuses main worlds unless the code hashes and settings match the lock.
* Task T-PAR: prompt `a+b=` with single digits; answer byte `'0'` or `'1'` for (a+b) mod 2.
  * 15 held orbits per world (25 to 28 held prompts); neither orientation of a held orbit is ever taught.
  * 40 training orbits (72 to 75 prompts), interleaved for 150 passes (10,800 to 11,250 records), then 1 h of simulated rest before the final probe.
* Arms:
  * BASE: shared Full151 bank only.
  * FROZEN: learned bank fed by a fixed random layer.
  * LEARN: the same layer trained by the three-factor rule.
  * SHUF: as LEARN, but training labels permuted among orbits at about half agreement with the truth.
  * ORACLE: class-coded addresses; a plumbing check only.
* Control task T-COPY (answer = a), run with BASE, FROZEN and LEARN.
* Settings: H 128, eta0 0.03 (ramped over 1,024 records), lam 1e-3, native warmup from 1/16 to 1 over 1,024 records.
* Metric: balanced held accuracy (mean per-answer recall) of the learned bank at the final checkpoint.

## Results: T-PAR (`results/main`, `SUMMARY.txt` / `SUMMARY.json`)
All columns except the last show the learned bank's balanced held accuracy.

| World | LEARN | FROZEN | SHUF | ORACLE | LEARN − FROZEN | LEARN − SHUF | Kernel alignment (LEARN), birth → end | Held transfer gap (LEARN), birth → end | Rule |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M1 | 0.726 | 0.434 | 0.622 | 1.000 | +0.29 | **+0.10** | 0.052 → 0.195 | −0.005 → +0.019 | fail (SHUF margin) |
| M2 | 0.972 | 0.361 | 0.489 | 1.000 | +0.61 | +0.48 | 0.042 → 0.274 | −0.003 → +0.046 | pass |
| M3 | 1.000 | 0.467 | 0.642 | 1.000 | +0.53 | +0.36 | 0.049 → 0.353 | −0.002 → +0.042 | pass |
| M4 | 0.964 | 0.607 | 0.500 | 1.000 | +0.36 | +0.46 | 0.043 → 0.215 | −0.007 → +0.017 | pass |

* **Gates.** All passed in every world:
  * ORACLE scored 1.00;
  * FROZEN's learned-bank codes were byte-identical at every checkpoint;
  * every T-COPY arm's system held accuracy was at least 0.71.
* **System answer** (shared bank plus learned bank; plain held accuracy):

  | Arm | M1 | M2 | M3 | M4 |
  |---|---:|---:|---:|---:|
  | LEARN | 0.48 | 0.75 | 0.81 | 0.64 |
  | FROZEN | 0.28 | 0.29 | 0.37 | 0.43 |
  | BASE | 0.28 | 0.36 | 0.41 | 0.43 |

  The shared bank's fixed kernel answers held prompts below chance, and summing it with the learned bank dilutes the learned bank's answer.
* **Taught accuracy.** The learned bank reached 0.97 to 1.00 in LEARN and FROZEN. In SHUF it reached 0.81 to 0.99: random labels are harder to store once codes overlap.
* **LEARN trajectories** (balanced held at passes 25 / 50 / 75 / 100 / 125 / 150 / after rest):

  | World | 25 | 50 | 75 | 100 | 125 | 150 | After rest | Note |
  |---|---:|---:|---:|---:|---:|---:|---:|---|
  | M1 | 0.50 | 0.63 | 0.59 | 0.63 | 0.72 | 0.73 | 0.73 | Kernel alignment still rising at the end (0.18 → 0.20) |
  | M2 | 0.46 | 0.81 | 0.94 | 0.97 | 1.00 | 0.97 | 0.97 | |
  | M3 | 0.53 | 0.63 | 0.80 | 0.93 | 1.00 | 1.00 | 1.00 | |
  | M4 | 0.43 | 0.79 | 0.71 | 0.89 | 1.00 | 0.96 | 0.96 | |

  The answers survive the 1 h rest unchanged.

## Results: T-COPY (control)
| World | LEARN balanced held, learned bank | FROZEN balanced held, learned bank | Kernel alignment, LEARN | System held, BASE / FROZEN / LEARN |
|---|---:|---:|---:|---:|
| M1 | 1.000 | 0.682 | 0.531 → 0.837 | 0.84 / 0.80 / 0.92 |
| M2 | 1.000 | 0.577 | 0.444 → 0.815 | 0.82 / 0.93 / 0.96 |
| M3 | 1.000 | 0.452 | 0.413 → 0.813 | 0.81 / 0.81 / 0.93 |
| M4 | 0.975 | 0.298 | 0.386 → 0.826 | 0.89 / 0.71 / 1.00 |

The measurement chain works. The learned layer also sharpens a separable kernel, with no loss of taught accuracy (1.00 in every arm).

## Verdict and integrity checks
* **Locked rule** (`summarize.py`): M2, M3 and M4 pass every clause and gate; M1 fails the clause LEARN ≥ SHUF + 0.20. Overall: **partial (3 of 4)**.
* **POSTHOC fail-closed check** (`posthoc_check.py`, `results/main/POSTHOC_VERDICT.json`; added after the lock and not part of it; see `cycles/POSTHOC_REVIEW.md`):
  * found exactly M1 to M4, all 32 locked histories, locked settings, correct record counts and final post-rest checkpoints, with every gate actually evaluated;
  * recomputed balanced accuracies from the raw outputs with independent code;
  * result: **partial (3 of 4)**, with no problems, and it agrees with the locked summary.
* **Runtime closure.** 55 files are loaded at runtime: 9 tracked files match the git blobs at the lock commit, and 46 files of the untracked native package match `minifly_a_v3/SOURCE_LOCK.json`. The locked code hashes still match `LOCK.json`.
* **Interruption and determinism.**
  * The container restarted during the T-COPY phase. All 20 T-PAR histories and COPY BASE M1 had finished.
  * The 11 missing T-COPY histories were rerun with the identical locked settings; the lock check passed again. Log: `main_run_resume.log`; first part: `cycles/main_run_part1.log`.
  * One completed history (COPY BASE M1) was rerun to `results/main_recheck`. All 8 checkpoints are bit-identical to the original in per-prompt outputs, accuracies, meters and native write logs.
* **Cost.** 11.2 CPU-hours for the 32 main histories.

## Interpretation
What the run supports, within this toy:
* A local rule produces the property that random features lack. Held-out parity is an interaction that frozen random features fail (0.36 to 0.61 balanced); exact gradient descent solves it in the evaluator-side screen on development worlds. The learned bank answered it in all four worlds (0.73 to 1.00). In three worlds the margins over both controls were large.
* The mechanism is the address kernel. In LEARN both kernel meters rose in every world: alignment 3.8 to 7.2 times, and the held transfer gap by +0.024 to +0.049, turning positive. In SHUF alignment fell in every world, and its gap rose only by 0.001 to 0.009, staying near zero. The answers follow the kernel: ORACLE shows that the native store reads out any class-aligned address perfectly.

Why M1 failed the rule:
* LEARN was slowest there: 0.73 at pass 150, with its kernel alignment still rising.
* SHUF scored 0.62, and moved between 0.56 and 0.70 across checkpoints. With 15 held orbits, a control at chance has a sampling SD of roughly 0.13 if orbits are treated as independent coin flips. SHUF's 0.62 is within that range, but the rule's 0.20 margin over SHUF is fixed, and the rule is applied as written.
* The pre-registered verdict is "partial". Longer training or more held orbits might change M1; that would be a new experiment, not a re-reading of this one.

Limits:
* **A toy.** Single digits, one binary answer, 15 held orbits per world.
* **Data-hungry.** About 11,000 records (150 passes over about 74 examples). The sparse KC store memorises the taught set within about 20 passes, which starves the error signal. On the native stores LEARN was at 0.63 to 0.81 balanced by pass 50 and needed 100 to 125 passes to plateau. On the cycle-2 surrogate, 60 passes gave no gain at all.
* **The system answer is weaker than the learned bank.** The shared bank's fixed kernel anti-generalises and the plain sum of margins dilutes the learned bank. Arbitration between the banks was out of scope.
* **Code collapse.** In LEARN and SHUF, mean code overlap rose from 0.22–0.34 to 0.70–0.83, driven by the common-mode error of answer bits that never vary. It slows storage of arbitrary labels.
* **Scope.** Only condition 1 was tested. Multi-digit numbers, carries, working memory, replay and stopping are out of scope (DESIGN.md §10).

## Possible next steps (not run)
* Bank arbitration: let the system answer weight the learned bank by its confidence, rather than summing margins with a bank that anti-generalises.
* A baseline-subtracted third factor (tested on the surrogate in cycle 2) to limit code collapse, and a study of data efficiency (fewer passes).
* More held orbits or more worlds, so that the SHUF control is less noisy. Operands with more than one digit are the natural route to more orbits.

## Files
* Process notes and design:
  * `DESIGN.md` (with Amendment A1), `LOCK.json`, `main.sh`;
  * `cycles/CYCLE1.md`, `CYCLE2.md`, `CYCLE3.md`, `POSTHOC_REVIEW.md`, `NOTES_FOR_SKILLS_SESSION.md`.
* Code: `lb1.py` (model), `data.py`, `run.py`, `summarize.py` (locked rule), `posthoc_check.py`, `checks.py` (41 checks), `diag_mlp.py`, `surrogate.py`, `diag_margins.py`.
* Results:
  * `results/main` (32 histories, `SUMMARY.json`, `SUMMARY.txt`, `POSTHOC_VERDICT.json`);
  * `results/main_recheck`;
  * development runs in `results/c1_tiny`, `c2a`, `c2b`, `c3_rehearsal`.
