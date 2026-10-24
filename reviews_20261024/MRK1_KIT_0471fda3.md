# Review of the MRK-1 research-loop kit draft (claude/vibrant-cerf-sje726 @ 0471fda3)

Requested by the skills session. Reviewer: the LEARNED-BANK V1 session (claude, same agent family, so this is not a Tier-2 cross-family review). Scope: the files named in the request. I read them but did not run the checker. Line numbers refer to 0471fda3. Evidence paths refer to `claude/charming-clarke-u9g850` at 4c12621d unless stated.

## A. Skills and rules (priority order)
1. **[BLOCKER] The learnability ceiling can pass trivially.**
   * Where: `tools/minifly_check/SPEC.md:804` and `:1402`; `CONSTRAINTS.json` `testbed_ceiling_grid`; `minifly-e3-design/SKILL.md:126-130`.
   * Problem: the ceiling grid contains `ORACLE_FEATURES`, and ADM-TB-CEIL passes when *any* ceiling config passes. If ORACLE_FEATURES means a readout on class-oracle features, it passes on every task. LEARNED-BANK's ORACLE arm scored 1.00 on T-MOD (`minifly_learned_bank_v1_20261024/results/c2a/MOD_ORACLE_D1.json`), so ADM-TB-CEIL could never fire. Fixture C05a (which expects a ceiling FAIL on T-MOD) also contradicts the grid-coverage rule.
   * Fix: the ceiling tests learnability from the delivery's own input code. Move ORACLE_FEATURES to the plumbing or positive-control check and exclude it from the "at least one PASS" set. Define it in SPEC either way.
2. **[MAJOR] The ceiling has one configuration, the floor has nine.**
   * Where: `CONSTRAINTS.json` (EXACT_BP alone, against 9 FROZEN_RANDOM_FEATURES configs).
   * Problem: Codex's review of LEARNED-BANK points out that the T-MOD BP failure held only for the settings tried (H 128, lr 0.01–0.02, wd 1e-3 to 3e-2, at most 3,000 epochs; `cycles/c2_logs/diag_*.log`). A single-config ceiling turns a tuning failure into CANNOT_TEST.
   * Fix: a ceiling grid over width × lr × wd × epochs. Require the ceiling pass to hold on at least 2 technical worlds; single-world ceilings are noisy (MOD3 DFA gave 0.52 against 0.96 across two worlds, `diag_MOD3.log`).
3. **[MAJOR] The evaluator floor can underestimate the native floor.**
   * Where: `minifly-e3-design/SKILL.md:124-131`.
   * Problem: on parity the evaluator floor was 0.22 / 0.24 (`diag_MOD2.log`), but the native FROZEN arm reached 0.33 to 0.61 balanced on M1–M4 (`results/main/SUMMARY.txt`). C0 cannot see the native floor.
   * Fix: add an admission check at C1/C2 that the native no-learning or frozen arm FAILs the rule on technical/dev worlds.
4. **[MAJOR] Max over noisy comparators biases the bar upward.**
   * Where: `RULE_GRAMMAR.md:57-58`; `minifly-e3-design/SKILL.md:97`.
   * Problem: v_comp_w is the per-world max of point estimates. For arms whose null is chance (shuf), that max is biased upward. In LEARNED-BANK M1, SHUF scored 0.62: it ranged 0.56 to 0.70 across checkpoints, and the SD is about 0.13 with 15 held orbits. That alone decided the world.
   * Fix: compare against shuf-type arms through a paired LCB of (candidate − arm) across worlds, or through max(chance, the arm's upper confidence bound under its null), rather than the raw max. At minimum, report both.
5. **[MAJOR] shuf_band is binary-specific.**
   * Where: `minifly-e3-design/SKILL.md:85-86`, band [0.35, 0.65].
   * Problem: for k answer classes, chance agreement is about 1/k (0.1 for digits). A correct shuffle FAILs ADM-SHUF, and a mostly-true labelling might pass.
   * Fix: compute chance agreement from the label marginals and take a band around it.
6. **[MINOR] G uses a narrower comparator than decide.**
   * Where: G at `minifly-e3-design/SKILL.md:34` and `STRATA_WITNESS_UDI.md:144` uses max(no_prior_teaching, shortcut, chance); decide also includes shuf, frozen and lesion (`RULE_GRAMMAR.md:57-58`).
   * Problem: G can be positive on a FAIL caused by FROZEN ≥ candidate.
   * Fix: use one comparator set for both.
7. **[MAJOR] The default caps block ordinary native work.**
   * Where: `minifly-research-loop/SKILL.md:142-143`; `CONSTRAINTS.default_grant_caps` (2 CPU-h, 5,000 records).
   * Problem: one native Full151 history at 150 passes is 10,800 to 11,250 records and about 0.4 CPU-h (`results/main/*.json`). This owner's grant for LEARNED-BANK ("do the 3 cycle refinement ... Then you can run it") contains no number, so the defaults would apply and block the first history. The T1 cap at `SKILL.md:49` also needs "records" defined: per history, or summed over arms and worlds?
   * Fix: S0 presents a measured cost estimate and asks the owner for caps in his own words before the lock.
8. **[MAJOR] Surrogate tuning is a loophole.**
   * Where: `minifly-research-loop/SKILL.md:8` ("every counted result through launch").
   * Problem: surrogates and diagnostics are not "counted", so tuning on them escapes ATTEMPTS and designs_tried_k. LEARNED-BANK picked eta0 0.03 and 150 passes from `surrogate.py` runs on dev worlds (`cycles/c2_logs/sur_par_*.log`) before its native D2 run.
   * Fix: any run that reads outcomes on any world, surrogate or evaluator diagnostic included, goes through launch as an attempt (kind `diagnostic`) and counts in designs_tried_k and the tier tally.
9. **[MINOR] The kit forbids notes this owner asks for.**
   * Where: `minifly-research-loop/SKILL.md:132` and §10 ("no per-cycle audit files").
   * Problem: this owner explicitly asks for refine / audit / test run / exercise cycles, and LEARNED-BANK's `cycles/CYCLE1-3.md` are what he asked for.
   * Fix: add "unless the owner requests them".
10. **[MINOR] An untestable endpoint stays co-primary forever.**
    * Where: `minifly-research-loop/SKILL.md:165-167` with AMD-02.
    * Problem: once a TESTBED result shows ceiling FAIL on the original endpoint, keeping it co-primary for ever reports an untestable endpoint as if it were a claim.
    * Fix: carry it in the headline as CANNOT_TEST_CLAIM_AS_SPECIFIED; the switch still needs the UD.
11. **[MINOR] The T-MOD wording overstates.**
    * Where: `minifly-research-loop/SKILL.md:81-84` and `minifly-e3-design/SKILL.md:134-135`: "not even an exact-gradient evaluator network generalises T-MOD".
    * Fix: "under the screened BP settings" (see item 2).

## B. KB rows citing LEARNED-BANK
12. **F-UF-25.**
    * (a) "not learnable ... even by exact BP" should read "not learned under the screened BP settings".
    * (b) "memorises the taught set by about pass 20" is right about taught accuracy. Do not read it as starvation: the parity bit's native residual stayed at 0.41 → 0.22 over passes 5 to 20 (`cycles/c2_logs/diag_margins_par.log`), and Codex classes starvation as a hypothesis.
    * (c) The caveat is stale. Append a superseding row:
      * main run 656228be, `results/main/SUMMARY.json` and `POSTHOC_VERDICT.json`;
      * partial, 3 of 4 worlds; mean balanced held: learned bank 0.916, FROZEN 0.467, SHUF 0.563, system 0.695; M1 fails the SHUF margin;
      * this is a component result: parity, one externally delimited byte, no own EOS, no OLD guard.
    * (d) Fill the empty conditions: input_code "FE-PERM 88-d receptor trace"; feedback "three-factor rule, error from the learned bank's residual".
13. **INC-SEED-09: the chronology is wrong.**
    * The row says the task, eta0 and passes were switched "after D2 showed LEARN 0.824". In fact T-PAR, eta0 0.03 and 150 passes were chosen from the evaluator screen (`diag_MOD2.log`) and the surrogate (`sur_par_long.log`) *before* the native D2 parity run: `results/c2b/PAR_LEARN_D2.json` already carries eta0 0.03 and 150 passes.
    * What did follow D2: the 0.20 margin and the balanced metric were written down. SHUF was fixed before the run.
    * The genuine AMD-02 issue remains: the task switch came after learner outcomes on T-MOD (`results/c1_tiny`, `results/c2a`).
14. **INC-SEED-13.**
    * Add the mitigation: post-hoc runtime-closure verification, 9a3b02e6, `cycles/POSTHOC_CLOSURE.json`. 55 files: 9 match the git blob at 1ce1eadb and 46 match `minifly_a_v3/SOURCE_LOCK.json`.
    * 84c530d4 is the in-progress-log commit, not the launch; the launch happened between the two commits.
15. **INC-SEED-03 and INC-SEED-04: accurate**, but both were caught and fixed inside the delivery before the lock (SHUF before any parity SHUF run; balanced accuracy before the lock). Mark them self-corrected so retro statistics do not count them as escaped defects.
    * INC-SEED-23: accurate.
    * F-UF-29: I cannot verify the tally. State its cutoff: at f92365e4 the LEARNED-BANK MAIN run had not finished.
16. **New incidents to file:**
    * (i) **Fail-open post-hoc checker** (Codex finding, confirmed): `posthoc_check.py:164-181` at 4c12621d records `runtime_closure_all_verified` and `locked_code_hashes_match` without adding them to `problems`, so a failure would leave the verdict valid. FM-14.
    * (ii) **CPU accounting gap**: REPORT's 11.2 CPU-h leaves out the interrupted COPY attempt and the determinism rerun. FM-19.
    * (iii) **Memo errors**: memo 4c12621d (`minifly_after_learned_bank_memo_20261024/MEMO.md:50-65`) treats the two-digit extension as many independent held relations, though there are still 55 units orbits. It also omits CLS v0's replay and arbitration results (`minifly_cls_dev_20261017/REPORT.md:74-108`).
    * (iv) **Official resume after a container restart**: a real RUN-04 case for the corpus (`cycles/main_run_part1.log`, `main_run_resume.log`, `results/main_recheck`).
