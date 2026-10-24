# LEARNED-BANK V1: lessons for the MiniFly research-loop skills (24 Oct 2026)

These notes answer a request from the session distilling MiniFly's research loop into reusable skills (explore, design, refine/audit, three test cycles, official run, audit/search). The question is how future iterations can improve without reward hacking or silent drift. Paths are relative to this folder.

## 1. Moments where a gate, metric or design could have been gamed or drifted
* **The primary task could not test the claim.**
  * The design's T-MOD (last digit of a+b) cannot be learned from 74 examples even by exact gradient descent (held 0.00 / 0.11). A failure would have been read as "the local rule fails".
  * The design review missed this. It was caught in cycle 2 by an evaluator-side diagnostic (`diag_mlp.py`: BP, DFA and FROZEN on the same traces and splits). See `cycles/CYCLE2.md` §2.
* **Tasks that would pass for nothing.** CARRY and CMP are additively separable: FROZEN random features already reach 0.73–0.89 held. A "pass" there shows nothing. The FROZEN arm in the diagnostic caught it.
* **The control can become a structured task.**
  * SHUF was a derangement. On binary labels a derangement inverts every label, i.e. it teaches anti-parity, which is fully structured. LEARN would have beaten SHUF for free.
  * Caught on read-through when adding the parity task. `data.shuffled_labels` now permutes orbit labels so that about half agree with the truth.
* **Plain accuracy would have flattered a constant answer.** Held answer classes are unbalanced per world (8 of 25 and 18 of 27), so a constant answer scores up to 0.68. The metric is now balanced accuracy (exactly 0.50 for a constant).
* **An unstable metric.** The design's held-only kernel-target alignment is unstable with 26 prompts. It was replaced in cycle 1 by the held transfer gap, before any main world existed.
* **Surrogate miscalibration.** The tuning surrogate memorised about 10 times faster than the native store, which would have biased every tuning conclusion. Its readout lr was calibrated against the native FROZEN taught curve (0.005).
* **A failure mode no gate covers.** Code collapse (mean code overlap 0.24 → 0.78) was seen only because `mean_offdiag_jaccard` was logged. Lesson: log more meters than you gate on.
* **A claim written before its check (my own near miss).** I wrote "balanced accuracies recomputed by hand and match" into `cycles/CYCLE3.md` before running that check. I ran it before committing, and it matched. Rule: a sentence in a cycle note follows the output that supports it, never the other way round.
* **Overclaims in earlier summaries.** Earlier MiniFly summaries of mine overclaimed:
  * "v3/v4/v5.1 all stopped before arithmetic" was wrong for v3;
  * "lookup only" was too strong given V9.9A/B.

  An independent reviewer (Codex) caught them. Summaries drift toward the convenient story; a second reader is the check.
* **Reusing a development world.** D3 was used for surrogate tuning and then for the native dress rehearsal. It is recorded as a second development confirmation, not as evidence (`cycles/CYCLE3.md`).

## 2. Checks every future cycle should inherit
* **Testbed validity screen, before the design ships.** On the exact split and input code:
  * exact gradient must succeed, otherwise the data are insufficient;
  * frozen random features must fail, otherwise the task is separable.

  If either fails, the experiment cannot test the claim. Pattern: `diag_mlp.py`.
* **Arm set:**
  * BASE: baseline.
  * FROZEN: random features. Its codes must stay byte-identical, which is a gate.
  * SHUF: keeps the label marginals, roughly orthogonal to the truth, orbit-consistent.
  * ORACLE: an evaluator-supplied positive control for the plumbing, never a candidate; gated at 0.90.
* **Lock guard.**
  * `run.check_lock` refuses main worlds (M*) unless every sha256 matches the code on disk, including the imported native module, and the command-line settings equal `LOCK.json`.
  * Main worlds are seeded from their names and never generated before the lock.
  * Files: `run.py`, `make_lock.py`, `LOCK.json`, `main.sh`.
* **A mechanical decision rule.** Margins (+0.20 over FROZEN and over SHUF), gates and a multi-world verdict (4 of 4 supported, 3 of 4 partial) are applied by code (`summarize.py`), not by reading tables.
* **Recompute headline numbers** independently from the raw per-prompt outputs stored in every checkpoint.
* **Native invariants as unit checks** (`checks.py`, 41 checks):
  * identical births;
  * a probe leaves the state hash unchanged;
  * a zero-coefficient teach equals a passive event;
  * identity gate (distinct prompts give distinct codes);
  * teaching moves margins toward the target.

## 3. Where the three-cycle process wastes effort or lets errors through
* **Task validity was tested too late:** after the native implementation and a full native T-MOD run. It should be a cheap "cycle 0" before the design document is pushed.
* **Amendments are the main drift channel.** Amendment A1 (task, metric, SHUF, settings) was adopted after seeing development data. That was legitimate only because the main worlds were untouched and the amendment was committed before the lock. Require every amendment to be listed in `LOCK.json` with the path to the evidence that motivated it.
* **"Provisional" settings invite open-ended tuning.** Declare the tuning budget, the knobs and the worlds in advance.
* **The dress rehearsal was expensive** (about 1 CPU-hour) and largely repeated development confirmation. One full history plus short runs of the other arms would test the pipeline as well.

## 4. Accumulating learned knowledge instead of re-deriving it
* **Keep a findings ledger.** One row per finding: the claim; its conditions (input code, number of examples, H, passes); the evidence path; and its status (confirmed, retracted or superseded, never deleted). Rows from this experiment:
  * Random features plus a linear readout generalise separable byte tasks on FE-PERM traces. Mod-10 is not learnable from 74 examples even with BP. Parity is the minimal interaction testbed.
  * The sparse KC store memorises the taught set by about pass 20, which starves the local error signal. Feature learning needs roughly 100 or more passes at eta0 0.03.
  * The native store saturates the constant answer bits, while the informative bit's residual persists (about 0.2 at pass 20).
  * Surrogate readout lr 0.005 matches native memorisation speed.
* **Keep the screening fixtures as reusable tools:** `diag_mlp.py`, `surrogate.py`, `diag_margins.py`, and the task-validity table in `cycles/CYCLE2.md`. Future designs should start from them.

Status: the locked main run on M1–M4 is in progress. Its result and report will be pushed to this branch (`claude/charming-clarke-u9g850`).
