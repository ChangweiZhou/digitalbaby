# Evidence ledger, 11 November 2026: review sections 9.59–9.82 and the coordinator note (section 2), every number recomputed from the receipts

Repository: `ChangweiZhou/digitalbaby` | Ref read: `df09c03a` (branch `claude/adoring-thompson-ip7x18`, a clone of the default branch's history) | Reviewed: 2026-11-11 | Reviewer: Claude Opus 5.5, job S1 of `OPUS_TASK_SIDE_JOBS_READ_ONLY_20261110.md` (read-only).

Script: `MOD97_LOCAL_RULE_20261106/ledger_recompute.py`. It uses the Python standard library only and reads JSON receipts only. It opens no `.pt` file and imports no model code. It trains nothing and writes nothing. To run it: `python3 -I MOD97_LOCAL_RULE_20261106/ledger_recompute.py` (add `--summary` for the counts only). Its full output is reproduced in Appendix A.

This ledger does not judge experimental design, because the two audits did that. It does not edit the review. Where the review and a receipt disagree, the ledger shows both values and does not pick one. It does not imply a checker gate, a seal or an independent confirmation. Everything here is a development result on a toy arithmetic harness at evidence level E2, held out by ordered pair, and none of it is a learning-core claim.

---

## What holds: the first screen

**Counts from the script (run of 11 November 2026, ref `df09c03a`): 319 checks. 277 MATCH; 2 WITHIN (a loose range that contains the data); 19 DIFFER; 21 NOT_RECOMPUTED.** A MATCH means the claimed number is reproduced from the receipts at the precision in which it is written. Exact ties such as 47.25 written as 47.2 count as matches.

**Most standing numbers reproduce from the receipts to the digit.** This covers every prediction verdict in 9.59 to 9.81, every per-seed table (9.61, 9.64, 9.66, 9.68, 9.70, 9.71, 9.77 and 9.81), every pass count and every Spearman coefficient. None of the 19 disagreements reverses a prediction verdict, and none changes a pass count.

What the receipts support, with the population each was measured on:

1. **Window learner under the sign rule, nothing frozen, drifting stream, raw reservoir 65,536.** It passes the family gate in **13 of 15** corrected-panel cloud seeds. On the `+-`/`-+` pairs alone it passes **2 of 15**, and on whole worked answers **9 of 15**. Adding the three development seeds re-run on corrected panels on the owner's machine gives **16 / 5 / 12 of 18** (9.75, 9.81, coordinator note row 1). The hard-pair gate passes **3 of 3** in the development seeds and **2 of 15** in the others.
2. **Replay share (E16-M3, seeds 400009 / 600013 / 300007, one life each).** At share 0 the hard pairs end near 1% (1.0–1.25) and the mod-97 ratio is 0.13–0.35. From share 0.25 to 0.75 the rule gains 29.6 / 36.9 points and backprop gains 9.8 / 14.8 (9.64). Admitted rows fall 1,769,488 → 1,179,680 → 589,872, so the effect of more replay is confounded with fewer fresh items (9.75 item 3).
3. **E16-M4 at checkpoint 5 (not at the end of life).** Every number in the 9.68 tables reproduces. The twelve receipts end at checkpoint 5 and have no completion record, so the predictions remain unjudged, as 9.68 says.
4. **E17 on the windowless two-bank learner (owner's machine, seeds 73129 / 93683 / 116089).** Every Phase 2, 3 and 4 table value reproduces. With credit through time at every layer and weight-sign feedback, the learner ends at **0.92–0.95** of its twin on mod-97 and **0.95–0.98** on mod-73. Its birth skills fall to **10.7 / 22.2 / 43.2**, and at 40 distractors addition is **18.25 / 9.25 / 28.25**. Backprop on the same pipeline reaches **0.98–1.02** of the twin (9.70, 9.75 item 5).
5. **Alignment at birth (E17_ALIGN_AT_BIRTH.json).** The first-bank projection gets cosine 0.09 / −0.06 / 0.05 under the sign rule as run and 0.08 / −0.01 / 0.26 with exact feedback. With credit through time it gets 0.81 / 0.78 / 0.55. Every cell of the 9.67 table reproduces.
6. **E18 births (cloud, seeds 73129 / 93683 / 116089, 24,576 updates).** Every arm value reproduces, and so does every prediction. B0, born with no backprop, reaches 0.96 / 0.96 / 0.94 of the backprop birth.
7. **E19 births.** The linear-write network born by backprop reaches 69.0 / 70.5 / 69.3. The forward-only rule reaches 43.2 / 45.8 / 54.5, which is 0.63 / 0.65 / 0.79 of that. The deep traces are worth +8.0 / +14.5 / +23.8 at the same decay. The ceiling arm ran at a different decay (0.30 against 0.25).
8. **E20-A stage 1 (window reference, backprop, cloud).** Both members of a twin are right in 63 / 80 / 130 of 300 twins (21.0 / 26.7 / 43.3%). With the state silenced the score is 0 / 0 / 1 of 600 items. Every P20 verdict reproduces.
9. **Exposure (9.79 item 3, fifteen corrected-panel seeds pooled).** The clean-item counts 361 / 125 / 136 / 1,656 and all eight accuracies reproduce from `E16_EXPOSURE_STRATA.json`.
10. **E16-F6 run B against run A.** All 15 files are identical in every field except CPU time, so run B is a repeat and not a second sample.

**What does not hold as written.** Each item below is small, and each one is listed in section 3 with both values:
- 9.79 says the registered mod-97 panel is "94–98%" items supervised elsewhere. That is true for add, max and min (94.0 / 97.9 / 97.7%). For subtraction it is **72.4%**.
- 9.67 says "`freeze: none`, `frozen: []` in every receipt". Only **6 of 18** drift-learner receipts carry both fields. No receipt records any freezing.
- 9.79 item 2 says the window learner's compositions with the state zeroed "fall to zero". In the receipts this holds for `+-`/`-+` on whole answers (0.0%). The comparison pairs reach up to 6.0% on whole answers and up to 28.2% on final answers.
- 9.66 says "every arm sits at 8–13 on the held pairs". The embeddings-frozen arm sits at **1.0–2.0**.
- Five table cells in 9.76, one difference in 9.71 and one mean in 9.60 are off by 0.1. Two CPU totals differ by about 1 s because they were summed from rounded per-run values.
- 9.81's P-CP.1 says "all nine ratios at or above 0.84". The lowest is **0.837**.

**What is not on file as a receipt.** This covers wall-clock times and forecasts, every claim made from saved weights (the embedding decay factor, row norms, the withdrawn strict re-score, the no-store single-operation comparison), the E18 parameter count, the E20-A version 1 panel census, and the next-steps lives of 9.82, which exist only as dry runs. Section 5 lists them.

**Statements that the review later corrected, but that still read uncorrected where they stand.** There are 13; section 4 lists them. They include the title of 9.70 ("keeps its birth"), the 9.76 table header ("Taught compositions"), 9.64 item 2 ("retention, not acquisition") and 9.66 item 2 ("has to be born by backprop and frozen"). The last row of the coordinator note's section 2 cites the strict re-score that 9.79 item 3 withdrew.

---

## 1. Status counts

| Status | Count | Meaning here |
|---|---:|---|
| **supported** | 279 | 277 checks MATCH and 2 are WITHIN (a range claim that contains every recomputed value without attaining its end points). |
| **partial** | 19 | DIFFER: the recomputed value disagrees with the written one, by any amount, or the claim holds for part of the stated population only. |
| **untested (not recomputed)** | 21 | The number is not recomputable from JSON receipts. The reason is given in each row. |
| **failed** | 0 | No standing prediction verdict was reversed by recomputation. |
| **invalidated** | 13 statements | Statements withdrawn or corrected by the review itself. These are listed in section 4 and are not counted among the 319 checks. |

Script summary line, pasted verbatim:

```
SUMMARY: 319 checks; MATCH 277; WITHIN 2; DIFFER 19; NOT_RECOMPUTED 21
DIFFER: 9.60-a'; 9.60-d; 9.61-h; 9.66-h; 9.62-j; 9.64-f; 9.67-l; 9.70-h'''; 9.75-j'(9); 9.76-a; 9.76-a; 9.76-c; 9.79-b(2); 9.79-e(3); 9.79-h'; 9.71-h; 9.71-j'''; 9.71-m; 9.81-f
WITHIN: 9.66-d'; 9.66-e'''
```

Definitions are those of the harness's own table scripts (`e16f7_gate.py`, `e16_audit_tables.py`, `e16m3_tables.py`, `summarize_exp16.py`):
- **operations** are exact answers at k = 12 distractors;
- a **family mean** is the mean of add, sub, max and min;
- **held compositions** are the mean of the final answers on `+-`, `-+`, `><` and `<>`;
- **hard pairs** are the mean of `+-` and `-+`, and the hard-pair ratio is the pooled learner count over the pooled reference count;
- **whole answer** is `exact` on the composition panels;
- the **family gate** requires all three family ratios to be at least 0.80.

**Machine.** No receipt records its machine. The machine named in each population below is the one the review states.

---

## 2. The ledger

Each section gives the exact population, followed by one row per claim. Check-ids are the script's, and the script output in Appendix A gives the claimed and recomputed values for every check. Some recurring terms:
- "Legacy" means the receipt predates the hashed format: it has no completion record and no panel hash.
- A ratio is "bound" when both receipts carry a completion record with a weight SHA-256 and the same `panel_sha256`.
- Whether each weight hash matches its `.pt` file was **not** checked, because no `.pt` was opened.

### 2.1 Section 9.59, E16-F6 (and the 9.60 run-B cross-check)

**Population.**

| | |
|---|---|
| Learner | `T_raw65536_drift_online`: window architecture, sign rule, `freeze` = none, raw reservoir 65,536 admitted online, replay share 0.5, wd 0.25 |
| Reference | `D_drift`: backprop on the joint stream (9.60) |
| Seeds | New: 500011, 600013, 700027, 800011, 900007, 1000003. Signature claims use the twelve seeds 73129 … 1000003. |
| Births | `e11_dev_M_{800011,900007,1000003}.json`, final step 24,576 |
| Receipts | `MOD97_LOCAL_RULE_20261106/exp45/e16_life_{T_raw65536_drift_online,D_drift}_<seed>.json`, plus the copies in `E16F6_RUN_B/exp45/` |
| Final stage | 7 stages, ending at F3.2 |
| Panels | Corrected ("keyed by split") for the nine seeds 200003 … 1000003; original for 73129, 93683, 116089 |
| Completion record / panel hash | None (legacy) |
| Machine | Two cloud containers (per the review) |

| Claim | Status | Note |
|---|---|---|
| Births at 89–98% at k = 12 (9.59-a) | supported | Development ops 89.0–98.3 |
| Twins at 90 / 85 / 90 or above; 500011 twin 88.4 on drift, against 84.0 under stages (b, c) | supported | 84.0 is `e16_life_D_500011.json` (83.96) |
| Learners pass in 5 of 6; composition ratios 0.83–0.89 (d, e) | supported | |
| 600013 ratios: compositions 0.705, mod-97 0.774; `+-`/`-+` at 42–48; max/min at 89; add/sub "near 50"; twin at 94 (f, g) | supported | add/sub 47.5 / 52.0 |
| P16F6.5: Spearman 0.217 (12 seeds) and 0.450 (9 seeds); four seeds above 80 at checkpoint 2 end at or below 65 (h, i) | supported | 300007, 500011, 800011, 900007 |
| 600013 dips at checkpoint 3 (47.2), not checkpoint 2 (79.9); life minima 47.2 / 62.0 against ≥ 66.7 (j, k) | supported | 47.25 exact tie |
| Drift gate 10 of 12 (l) | supported | |
| Run B = run A (9.60 item 4) | supported | 15 of 15 files are equal in every field except `cpu`; 0 of 15 are equal including `cpu` |
| Pace 4.7 h against 5.3 h; within 3% of forecast CPU (m) | untested | Logs and the brief only |

### 2.2 Section 9.60, the audit memo (items checked against receipts)

**Population.**

| | |
|---|---|
| Lives | The 21 raw-reservoir lives of 9 November: nine staged (`e16_life_T_raw65536_<seed>[_oldpanels].json` against `e16_life_D_<seed>.json`) and twelve drift |
| Other receipts | E16-L (`e16_life_T_raw65536_lossall_*`); E16-RP (`E16RP_SAME_WEIGHTS.json`; `e16_life_T_raw65536_93683[_oldpanels].json`); E17 Phase 1b twin (`e17_life_D_bank2_frz1_*`); E16-RC (`e16_life_T_raw{16384,32768,131072}_*`, with 65,536 taken from `*_oldpanels`) |
| Completion record / panel hash | Legacy throughout |

| Claim | Status | Note |
|---|---|---|
| Embedding rows = birth × 9.9387e-5 (a) | untested | Needs weights |
| …"which is AdamW's decay alone over 36,864 updates at 0.25 × 0.001" (a') | **partial** | (1 − 0.00025)^36864 = **9.932e-5**, against 9.9387e-5 written; per block 0.2152, which matches 0.215 |
| 36 of 47 rows get no gradient under the local arm; 2 under backprop (b) | supported | Taken from the Phase 2 receipts (field `zero_grad_first_update`). Phase 1 receipts carry no such field. |
| 21 lives: 18 / 14 / 15 / 4 pass; 17 of 21 fail on `+-` alone (c) | supported | |
| Comparison pairs, learner 89.5 against reference 98.7 (d) | **partial** | Reference **98.61** |
| Plus-minus 64.5 against 86; shortcut 69% (d', d'') | supported | |
| E16-L add/sub after block 1; under 12 after block 2; under 6 after block 3 (f) | supported | |
| E16-RP: 16.2 at block 6; 6.8 on the same weights; "about 9" run to run (g) | supported | |
| Frozen twin 89.4–95.0 by block (h) | supported | |
| E16-RC means 74.2 → 76.9 → 78.3 → 79.8 (i) | supported | |
| Exposure overlaps 40–52%; 47 / 44 / 21% (e) | untested | No JSON receipt |
| Scorecard 17 of 33; 1 of 12 (j) | untested | Counts over prediction files, which are not receipts |

### 2.3 Section 9.61, E17 Phase 2

**Population.**

| | |
|---|---|
| Architecture | `bank2` (windowless), embeddings outside decay (`bookdecay` off) |
| Seeds | 73129, 93683, 116089 |
| Arms | `T_bank2_nbd` (sign rule, freeze none); `T_bank2_nbd_frz1` (freeze bank1); `T_bank2_nbd_frz1_xfb_tup` (exact feedback, `traceup` proj); twin `D_bank2_nbd_frz1` (wd 0.30, against 0.25 for the learner arms) |
| Comparison receipts | Phase 1 `e17_life_T_bank2_raw65536_*` and `e17_life_T_bank2_frz1_raw65536_*` |
| Receipts | `exp45/e17_life_<arm>_<seed>.json` |
| Final stage | 7 stages |
| Completion record / panel hash | Legacy |
| Machine | Owner's Mac (per the review) |

| Claim | Status | Note |
|---|---|---|
| P17.2.3–P17.2.6, every value in the table, including the Phase 1 values with decay (a–d) | supported | |
| 36 of 47 rows without gradient (local); 2 (backprop) (e) | supported | All three seeds |
| final/birth = 1.0000; row norm 4.1 → 0.3 (f) | untested | Needs weights |
| Unfrozen arm: development ops 93–95 → 1–2; mod-97 never above 3 (g) | supported | |
| Frozen sign rule "holds 82–87 through the two mod-97 blocks" (h) | **partial** | 73129 block 1 = **81.4**; the range is 81.4–86.9 |
| Frozen arm falls to 51–54, ends at 59–62; twin at 90–92; add/sub; ratios 0.66–0.69 / 0.77–0.81 / 0.39–0.52 (h'–j) | supported | |
| XT arm within 3 of the twin and above it in two seeds; mod-73 2–3 above; add/sub 84–91; compositions 0.81–0.90; twin hard pairs (k, l) | supported | |
| CPU-seconds, all five arms (m) | supported | |
| Wall 2 h 29 min (n) | untested | Logs |

### 2.4 Section 9.66, E17 Phase 3

**Population.** As in 2.3, with arms `T_bank2_nbd_frz1_xfb` (exact feedback only), `T_bank2_nbd_frz1_tup` (sign feedback with `traceup` proj) and `T_bank2_nbd_frzb` (freeze book). The comparators are the Phase 2 receipts. Receipts are legacy, with 7 stages; machine is the owner's Mac.

| Claim | Status | Note |
|---|---|---|
| P17.3.1–P17.3.4, every value (a–d) | supported | |
| "hard pairs 0–4" (d') | supported (loose) | 0.2–2.3 |
| Trace alone at 0.89 / 0.94 / 0.93 of the twin; mod-73 0.97–1.00; add/sub (e–e'') | supported | |
| "loses 10–13 points during the composition blocks" (e''') | supported (loose) | 11.6–12.6 |
| Exact feedback alone adds 8–12; embeddings-frozen arm never above 2 (f, g) | supported | |
| "Every arm sits at 8–13 on the held pairs" (h) | **partial** | XF and TU arms 8.3–13.2; the **frzb arm is at 1.0–2.0** |
| Frozen twin 15–18; CPU-seconds (h', i) | supported | |
| Wall 1 h 35 min (j) | untested | Logs |

### 2.5 Section 9.62, E16-F7

**Population.**

| | |
|---|---|
| Seeds | New: 1100009, 1200007, 1300021, 1400017, 1500007, 1600033. Pooled claims use all eighteen drift seeds. |
| Learner | As in 2.1 |
| Reference | `D_drift` |
| Births | `e11_dev_M_<seed>.json` |
| Final stage | 7 stages |
| Panels | Corrected for the fifteen later seeds; original for the three development seeds (9.75 item 1) |
| Completion record / panel hash | Legacy |
| Machine | Cloud container (per the review) |

| Claim | Status | Note |
|---|---|---|
| P16F7.1–P16F7.5, every number (a–e) | supported | Lowest birth op is 86.3 (sub89, 1400017). 1600033 passes by 0.0048. The Spearman values are 0.109 (18 seeds) and −0.714 (6 seeds). |
| 16 of 18 family; 4 of 18 hard pairs (f) | supported as arithmetic | The pooled population mixes panel versions; see 9.75 item 1 and section 4 |
| Mean `+-`/`-+` 64.8 against 87.5; 1400017 S = 65.2 and final `+-` 79.0, the best of 18; S near 82 for 1200007 and 1300021; finals 49.7 / 51.5; late losses 15 / 30 (g–i) | supported | |
| Total CPU 79,218 (j) | **partial** | Unrounded sum **79,217.5**. The ranges per kind match. |
| Wall 5 h 37 min; machine (k) | untested | Logs |

### 2.6 Section 9.64, E16-M3

**Population.**

| | |
|---|---|
| Seeds | 400009, 600013, 300007 |
| Rule arms | `T_raw65536_drift_online{_rs0,_rs25,,_rs75}` |
| Backprop on the pipeline | `M_raw65536_drift_online{_rs25,,_rs75}`. M at share 0.5 for 400009 is E16-M2's receipt (CPU 3,871, a different job). |
| Twin | `D_drift` |
| Final stage | 7 stages |
| Panels | Corrected |
| Completion record / panel hash | Legacy |
| Machine | Cloud (per the review) |

| Claim | Status | Note |
|---|---|---|
| The share table, all 21 cells (a) | supported | |
| P16M3.1–P16M3.4 values (b–e) | supported | |
| "29 and 37 points across the range, against backprop's 10 and 15" (f) | **partial** | Rule **29.6** (written 29) / 36.9; backprop 9.8 / 14.8 |
| 400009 at 0.75: 0.93 / 0.87 / 0.97 and 0.82; 600013 0.796 and 0.66; 300007 0.88 (g) | supported | |
| Share 0 at about 1%, from peaks of 59–85, with mod-97 ratios 0.13–0.35; 400009 peak 57 → 41 (h, i) | supported | |
| CPU ranges; fourteen lives (j, k) | supported | |
| Wall 5.0 h (l) | untested | Logs |

### 2.7 Section 9.67, alignment at birth; Phase 4 prepared

**Population.**

| | |
|---|---|
| Alignment | `exp45/E17_ALIGN_AT_BIRTH.json`: births 73129, 93683, 116089; modes `sign`, `sign_tup`, `exact`, `exact_tup` and `sign_time` |
| Checks | `exp45/E17P4_CHECK.json`; dry runs in `exp45/archive_e17p4_dry/*_dry4_73129.json` (7 stages) |
| Comparator | `e17_life_D_bank2_nbd_*` |

| Claim | Status | Note |
|---|---|---|
| Every cell of the alignment table (a–f) | supported | "second bank's write, readout 0.65–0.81" covers layers lc2, lw2, lm, lm2 and l2. The output layer l3 is at 1.00. |
| Every layer at 0.45–0.81 under through-time credit (g) | supported | l3 excluded, as above |
| The option reproduces the plain rule to 2e-7, and equals the direct computation exactly; 48.7 / 54.7 / 49.4 ms (h) | supported | Max 2.2e-7 |
| Comparator 92.3 / 93.7 / 93.3 and 92.6 / 91.6 / 91.9 (i) | supported | |
| Dry runs pass seven stages; 36 of 47 against 2 (j) | supported | |
| "`freeze: none`, `frozen: []` in every receipt" for the E16 learner (l) | **partial** | **6 of 18** drift-learner receipts carry both fields. Six have no `freeze` field and twelve have no `frozen` field. None records any freezing. |
| Forecast 2 h 20 min (k) | untested | A forecast |

### 2.8 Section 9.70, E17 Phase 4

**Population.**

| | |
|---|---|
| Seeds | 73129, 93683, 116089 |
| Arms | `T_bank2_nbd_tall` (rule T, sign feedback, `traceup` all, freeze none, wd 0.25, rshare 0.5); `T_bank2_nbd_tup` (`traceup` proj); `M_bank2_nbd` (rule D, wd 0.25, rshare 0.5) |
| Twin | `D_bank2_nbd` (wd 0.3) |
| Births | Birth hash is identical across the three arms in each seed |
| Final stage | 7 stages |
| Completion record / panel hash | Legacy; these receipts carry no `projfb` field |
| Machine | Owner's Mac |

| Claim | Status | Note |
|---|---|---|
| The whole table: 9 rows × 9 values, plus the twin and Phase 2 summary rows (a–c) | supported | |
| P17.4.1–P17.4.5 values (a, d) | supported | |
| New arm at 91–93 in block 1; above the Phase 3 frozen arm in every seed (e) | supported | 90.7 rounds to 91 |
| Backprop 3–7 above on mod-97 and 3–6 on mod-73; add/sub 77–86 against 88–93; max/min 91–94 against 92–97; pipeline 0.98–1.02 (f, g) | supported | |
| One-layer trace 65–75, then 42–44 and 47–50 (h) | supported | |
| One-layer trace "loses 30–46 points when the compositions enter" (h''') | **partial** | Block 2 minus the minimum of blocks 3–4 gives **33.8–46.6**. Block 2 minus block 3 gives 29.9–35.7. Neither reading gives 30–46. |
| 36 of 47 against 2; development ops 10.7 / 22.2 / 43.2, 48.6 / 41.4 / 53.6 and 49.1 / 37.9 / 51.5; CPU (i–k) | supported | |
| Same births, seven stages; settings as described (l, m) | supported | |
| Wall 2 h 04 min (n) | untested | Logs |

### 2.9 Section 9.75, Codex's source audit and its errata

**Population.**
- Strata: the eighteen drift seeds, split by the `panels` field (2.1 and 2.5).
- Replay rows: the M3 receipts.
- E16-M2: `e16_life_{T_raw65536,D,M_raw65536,T_oracle}_600013.json`.
- Phase 4: as in 2.8.
- Checks: `AUDIT_RESPONSE_CHECKS.json`, `E17P4_INDEPENDENT_CHECK.json`. The split overlap in `AUDIT_RESPONSE_CHECKS.json` covers **three seeds only (73129, 400009, 1200007)**.

| Claim | Status | Note |
|---|---|---|
| Item 1: corrected panels 13 / 2 / 9 of 15; original 3 / 2 / 3 of 3 (a) | supported | |
| Item 2: fresh compositions arrive until about three quarters of the life, and keep being replayed (b) | partial (not fully recomputable) | Receipts give totals per block: 25,799 fresh F2 rows in block 5, none in block 6, and 71,178 replay F2 rows in block 6. The three-quarter point and the equal weights at checkpoint 4 are not recorded. |
| Item 3: 1,769,488 / 1,179,680 / 589,872 admitted (c) | supported | |
| Item 4: 25.2 against 24.6 at checkpoint zero; panel versions differ by arm (d) | supported | |
| Item 5a: 59.2 / 47.0 / 60.2 against 62.8 / 58.0 / 67.1; end 10.7 / 22.2 / 43.2 (e) | supported | |
| Item 5b: at 40 distractors, 18.2 / 9.2 / 28.2 and 16.8 / 25.8 / 19.0, against 77–86 at 12 (f) | supported | Exact ties: 18.25 / 9.25 / 28.25 and 16.75 / 25.75 |
| Item 5c: the two arms used different feedback maps (g) | untested | No `projfb` field in the Phase 4 receipts; this rests on source code |
| Item 5c: cosine 0.991–0.999; alignment 0.55 / 0.66 / 0.37 against 0.54 / 0.67 / 0.38 (g', g'') | supported | |
| Item 6: independent check to 2e-15 (h) | supported | Max 2.1e-15 |
| Item 6: the start-of-sequence row never receives a gradient (h') | untested | Not in the JSON |
| Item 7: B3 11 of 45; B0 / B1 / B2 45 (i) | supported | |
| Item 9: 25% reversal in the taught pool; 71–73% supervised as a sub-step (j, j'') | supported | Three seeds only |
| Item 9: "49–52% at birth" (j') | **partial** | **48.1–51.8%** (1200007 add89: 48.1) |
| Item 9: strict re-score, −1.2 to +1.7, ≤ 0.025, 1600033 0.805 → 0.798 (k) | **invalidated** (by 9.79 item 3); untested | Made from weights. The registered 0.805 is supported (0.8048). |
| Item 10: the store contributes nothing on single operations in all fifteen seeds (l) | untested | The cloud receipts have no no-store field; the comparison was made from weights |

### 2.10 Section 9.76, composition from memory

**Population.**

| Row | Receipts | Machine |
|---|---|---|
| Window reference | `e16_life_D_drift_{200003,300007,400009}` | Cloud |
| Window learner | `e16_life_T_raw65536_drift_online_{200003,300007,400009}` | Cloud |
| Windowless rows | `e17_life_{D_bank2_nbd, M_bank2_nbd_raw65536, T_bank2_nbd_tall_raw65536}_{73129,93683,116089}` | Mac |

Scores are whole answers (`exact`) on the `within`, `+-`/`-+` and `><`/`<>` panels. All receipts are legacy, with 7 stages.

| Claim | Status | Note |
|---|---|---|
| Table rows 1, 2 and 4 (a) | supported | |
| Table row 3, windowless reference (a) | **partial** | 116089 `+-`/`-+` is **2.83** (written 2.9); `><`/`<>` is **25.33** (written 25.4) |
| Table row 5, weight signs through time (a) | **partial** | 93683 `+-`/`-+` is **2.33** (written 2.4); `><`/`<>` is **20.33** (written 20.4) |
| Untaught pairs at 1–7% (b) | supported | 0.75–6.83 |
| "Below the 66–69% that ignoring one operator would give" (c) | **partial** | The receipts' `ignore_o1` ceilings on these panels are **65.0–70.0%** |

### 2.11 Section 9.79, Codex's complete audit and E19

**Population.**

| | |
|---|---|
| Item 3 | `E16_EXPOSURE_STRATA.json`: fifteen corrected-panel seeds pooled, 400 items per operation per seed. This is built from weights, but its counts are in JSON. |
| E19 | `exp45/e19_dev_kbank2_{bptt,r1,r2}_{73129,93683,116089}.json` (decay 0.30 / 0.25 / 0.25; step 24,576) |
| Gated reference | `e17_dev_bank2_nbd_*` |
| Machine | Cloud worker (E19) and owner's machine (reference), per the review |

| Claim | Status | Note |
|---|---|---|
| Item 1: windowless reference at 49–59% on `within` and 1–7% untaught (a) | supported | |
| Item 2: with the state zeroed, held compositions fall to zero (b) | **partial** | `silenced_compose` over 18 learners: `+-`/`-+` whole answers max **0.0%**; all four pairs max **6.0%** on whole answers; final answers max **28.2%**. The review names neither the metric nor the learners. |
| Item 3: counts 361 / 125 / 136 / 1,656 of 6,000; eight accuracies; reference values; max 5 points lower (c, d) | supported | |
| Item 3: "the registered mod-97 panel is 94–98% items supervised elsewhere" (e) | **partial** | add 94.0, max 97.9, min 97.7, **sub 72.4** |
| Item 4: 13 of 15 registered; 12 of 15 sensitivity (f) | supported / untested | 13 recomputed. 12 depends on the weights-based 0.798. |
| E20-A v1: 6.6% in view, 88.7% hidden (g) | untested | Markdown only |
| E19: 69.0 / 70.5 / 69.3; P19.3 43.2 / 45.8 / 54.5 and 0.63 / 0.65 / 0.79; P19.4 +8.0 / +14.5 / +23.8; decays; 220 against 103 ms (h–k) | supported | |
| "the gated network: 91–95" (h') | **partial** | Reference births **90.3–94.5** |

### 2.12 Section 9.71, E18 births

**Population.**

| Arm | Receipts | Settings |
|---|---|---|
| Reference | `e17_dev_bank2_nbd_*` | |
| B0 | `e17_dev_bank2_nbd_bT_tall_*` | top / T / all / 0.25 |
| B1 | `e17_dev_bank2_nbd_lh_*` | local / D / none / 0.30 |
| B2 | `e17_dev_bank2_nbd_lh_bT_tall_*` | local / T / all / 0.25 |
| B3 | `e17_dev_bank2_nbd_lh_bT_*` | local / T / none / 0.25 |

Seeds are 73129, 93683 and 116089; the final step is 24,576 for all twelve. Hard mean is (add89 + sub89) / 2 at k = 12 on the development panels. Receipts are legacy, as the review says. Machine: cloud (per the review); the reference is from the owner's machine.

| Claim | Status | Note |
|---|---|---|
| The whole table, 5 arms (a) | supported | |
| Settings and final step (b) | supported | |
| P18.1–P18.6 (c–g) | supported | |
| B0 − B2 = 12.7 / 21.0 / 18.4 (h) | **partial** | 116089 is **18.33** (84.67 − 66.33); 18.4 is the difference of the rounded values |
| B0 still rising 84 → 88, 88 → 91, 78 → 85 (i) | supported | |
| First bank's head about 20; upper bank adds about 60; additions at 27–30% (j–j'') | supported | |
| "The package ends about 10 points under the end-to-end reference" (j''') | **partial** | **10.5 / 12.3 / 13.2** (mean 12.0) |
| B3 rows unmoved; digit rows about 0.7 (k) | supported | |
| 657,153 parameters (l) | untested | Not in the receipts |
| 65,118 CPU-s (m) | **partial** | Unrounded sum **65,116.6**; 65,118 is the sum of the rounded per-birth values |
| Wall 4 h 39 min (n) | untested | Logs |

### 2.13 Section 9.81, the twelve repeats on corrected panels

**Population.**

| | |
|---|---|
| Drift seeds | 73129, 93683, 116089: `e16_life_T_raw65536_drift_online_cp_*` against `e16_life_D_drift_cp_*` |
| Staged seeds | 400009, 600013: `e16_life_{T_raw65536,M_raw65536,D}_cp_*` |
| Final stage | 7 stages |
| Completion record | **Present in all 12**, each with a weight SHA-256 |
| Panel hash | **Equal across the arms of each comparison** |
| Machine | Owner's Mac |

| Claim | Status | Note |
|---|---|---|
| Seven stages, completion record and weight hash; same panel hash; checkpoint zero at 24.17 and 24.58, identical across arms (a–c) | supported | The weight hash was not compared with the file |
| The drift table, 15 values; the staged table, 10 values (d, e) | supported | |
| P-CP.1 "all nine ratios at or above 0.84" (f) | **partial** | Minimum **0.837** (116089 mod-97). It rounds to 0.84 but is strictly below. |
| P-CP.4: store identical (g) | supported | |
| 16 / 5 / 12 of 18; 2 of 15; single operations reproduce to the digit; held pairs moved < 1 (max 0.79) (h–j) | supported | |
| CPU ranges (k) | supported | |
| Wall 2 h 17 min (l) | untested | Logs |

### 2.14 Section 9.77, E20-A stage 1

**Population.**

| | |
|---|---|
| Learner | Window reference by backprop, `e16_life_D_cm6_{73129,93683,116089}.json` (stream "blocks", `compgap` 6, store off) |
| Final stage | 7 stages |
| Completion record | Present |
| Birth | Admitted by `BIRTH_MANIFEST.json`, hash matches |
| Memory-panel hash (73129) | Equals `QUALIFY_E20_MEMORY_PANELS.json` |
| Machine | Cloud (per the review) |

| Claim | Status | Note |
|---|---|---|
| The whole table and P20.1–P20.3, including 420 of 600 (a–h) | supported | |
| Receipt integrity as listed (i) | supported | The weight hash was not compared with the file |
| Letter-free plus-minus panel at 69–81; hidden operand 42–66 (j, k) | supported | |
| "The reference taught without letters ended at 85–89" (j') | untested | No receipt is named. Candidates: staged D 84.5–88.2; D_drift 86.3–88.5; D_drift_cp 85.5–87.0. |

### 2.15 Section 9.68, E16-M4 (stopped)

**Population.**

| | |
|---|---|
| Receipts | `e16_life_T_raw65536_drift_online_rs75_<F7 seed>_killed.json` and `…_rs{88,94}_{300007,400009,600013}_killed.json` |
| Comparators | Share 0.5 and 0.75 lives, and `D_drift`, all **at checkpoint 5** |
| Final stage | Checkpoint 5 in all twelve (6 stages) |
| Completion record | **Absent** |
| Panels | Corrected |
| Machine | Cloud container that restarted |

| Claim | Status | Note |
|---|---|---|
| Both tables, 6 + 3 rows (a, g) | supported | |
| Twelve receipts end at checkpoint 5 without final weights (b) | supported | |
| 5 of 6 higher, 4 by 5 points or more; 3 of 6 against 1 of 6 at ≥ 0.80; family gate 6 against 4 (failures 1200007 at 0.73, 1600033 at 0.79); falls of 29.2 / 21.2 against 7.3 / 8.5 (c–f) | supported | |
| mod-73 is 2–6 lower at the two highest shares; no high-share life under the gate (h) | supported | Against share 0.75. Against share 0.5 the drop is 1.8–7.6. |
| P16M4.1–P16M4.5 | **untested** | Not judged, as 9.68 states |

### 2.16 Section 9.82, next steps

There is no numeric result. P-Z.1–3 (backprop and the rule at zero replay) and P-E.1–4 (every-byte lives) are **untested**: only dry receipts exist (`exp45/archive_nextsteps_dry/`). The rulings are decisions and fall outside this ledger.

### 2.17 Section 9.72, literature check

There is no receipt. The four verdicts on the coordinator's claims and the sources cited are **not traceable to a receipt** and were not checked here. That is job S2's and the literature workers' territory.

### 2.18 Coordinator note, section 2 ("What holds")

| Row | Status | Recomputed |
|---|---|---|
| 1. Window learner passes the family gate in 13 of 15; 2 of 15 on plus-minus; 9 of 15 whole | supported | 13 / 2 / 9. No receipt records freezing; 6 older receipts lack the `freeze` field. |
| 2. Retains nothing without replay | supported | Share 0 hard pairs 1.25 / 1.0 / 1.08 |
| 3. More replay gives better hard pairs, for the rule more than for backprop | supported | Rule +29.6 / +36.9, backprop +9.8 / +14.8. The allocation caveat stands (9.75 item 3). |
| 4. First bank lost through truncation in time, not weight signs | supported (as alignment at birth) | Projection cosine: exact 0.08 / −0.01 / 0.26; through time 0.81 / 0.78 / 0.55 |
| 5. Phase 4 at 0.92–0.95 of the reference; original skills at 11–43; fails at 40 distractors | supported | 0.919–0.952; 10.7–43.2; 9.25–28.25 at k = 40 |
| 6. Backprop on the pipeline reaches the reference; the remaining 3–7 points belong to weight signs | supported (numbers) | 0.984–1.018; 3.0–7.3 |
| 7. Forward-only credit matches backprop to rounding; the linear write costs about 22 points | supported (worker's check) / untested (coordinator's own) | `E19_CHECK.json`: 5.1e-16 (float64) and 2.6e-7 (float32); the coordinator's check is markdown only. The cost is 22.4 on average (22.2 / 24.0 / 21.0). |
| 8. 25% reversal overlap; 71–73% sub-step exposure; "did not inflate scores measurably; strict panels move ratios by at most 0.025" | supported (overlaps, three seeds) / **invalidated** (the strict-panel limit) | The strict re-score this row cites was withdrawn by 9.79 item 3, and the row does not say so (section 4) |

---

## 3. Every disagreement between the review and the receipts (both values; no side taken)

| # | Where | Claim as written | Receipts (recomputed) | Receipt(s) |
|---|---|---|---|---|
| 1 | 9.60 §3 | Decay factor 9.9387e-5 "which is AdamW's decay alone over 36,864 updates at 0.25 × 0.001" | (1 − 0.25 × 0.001)^36864 = **9.932e-5**. The measured factor needs weights. | Arithmetic |
| 2 | 9.60 §4 | Comparison pairs: reference **98.7** (learner 89.5) | Reference **98.61**; learner 89.49 | 21 raw-reservoir lives (2.2) |
| 3 | 9.61 item 2 | Frozen sign rule "holds **82–87** through the two mod-97 blocks" | **81.4–86.9** (73129 block 1: 81.4) | `e17_life_T_bank2_nbd_frz1_raw65536_*` |
| 4 | 9.66 item 3 | "Every arm sits at **8–13** on the held pairs" | XF 8.3–8.8; TU 12.0–13.25; **frzb 1.0–2.0** | `e17_life_T_bank2_nbd_{frz1_xfb,frz1_tup,frzb}_raw65536_*` |
| 5 | 9.62 Cost | **79,218** CPU-s over eighteen runs | **79,217.5** (unrounded sum) | `e11_dev_M_*`, F7 `T`/`D_drift` |
| 6 | 9.64 item 1 | Rule **29** and 37 points across shares 0.25–0.75; backprop 10 and 15 | Rule **29.6** and 36.9; backprop 9.8 and 14.8 | M3 receipts, 400009 and 600013 |
| 7 | 9.67 item 3 | "`freeze: none`, `frozen: []` in **every** receipt" | **6 of 18** carry both; 6 lack `freeze`; 12 lack `frozen`; none records freezing | 18 `T_raw65536_drift_online` receipts |
| 8 | 9.70 item 3 | One-layer trace "loses **30–46** points when the compositions enter" | Block 2 − min(blocks 3, 4): **33.8–46.6**; block 2 − block 3: 29.9–35.7 | `e17_life_T_bank2_nbd_tup_raw65536_*` |
| 9 | 9.75 item 9 | Reversal overlap at birth **49–52%** | **48.1–51.8%** (three seeds) | `AUDIT_RESPONSE_CHECKS.json` `split_overlap` |
| 10 | 9.76 table | Windowless reference 116089: untaught plus-minus **2.9**, comparison **25.4** | **2.83**, **25.33** (34 / 1200 and 304 / 1200) | `e17_life_D_bank2_nbd_116089.json` |
| 11 | 9.76 table | Weight signs through time 93683: plus-minus **2.4**, comparison **20.4** | **2.33**, **20.33** (28 / 1200 and 244 / 1200) | `e17_life_T_bank2_nbd_tall_raw65536_93683.json` |
| 12 | 9.76 text | "the **66–69%** that ignoring one operator would give" | `ignore_o1` ceilings on these panels: **65.0–70.0%** | `e17_life_D_bank2_nbd_*` stage 0 `shortcut_ceilings` |
| 13 | 9.79 item 2 | With the state zeroed, held compositions "fall to **zero**" | `+-`/`-+` whole answers 0.0%; all four pairs whole answers up to **6.0%**; final answers up to **28.2%** (18 drift learners) | `silenced_compose` in the 18 drift receipts |
| 14 | 9.79 item 3 | Registered mod-97 panel is "**94–98%** items whose relation is supervised somewhere" | add 94.0, max 97.9, min 97.7, **sub 72.4** | `E16_EXPOSURE_STRATA.json` |
| 15 | 9.79 E19 | "the gated network: **91–95**" | Reference births **90.3–94.5** | `e17_dev_bank2_nbd_*` |
| 16 | 9.71 item 3 | B0 − B2 = 12.7 / 21.0 / **18.4** | 12.67 / 21.00 / **18.33** | E18 births |
| 17 | 9.71 item 4 | Package ends "about **10** points under the end-to-end reference" | **10.5 / 12.3 / 13.2** (mean 12.0) | E18 births B1 and reference |
| 18 | 9.71 Cost | **65,118** CPU-s for twelve births | **65,116.6** unrounded (65,118 is the sum of the rounded values) | E18 births |
| 19 | 9.81 P-CP.1 | "all nine ratios **at or above 0.84**" | Minimum **0.837** (116089 mod-97) | `e16_life_*_drift*_cp_116089.json` |

The following are not counted as disagreements. The loose ranges (WITHIN) are 9.66 "hard pairs 0–4" (0.2–2.3) and 9.66 "loses 10–13 points" (11.6–12.6). The exact ties written as either neighbour (for example 47.25 written 47.2) are listed as MATCH.

---

## 4. Statements superseded later in the review but not marked where they stand

The review is append-only, so the corrections sit in later sections. Each statement below still reads as originally written.

| # | Where it stands | Statement as it reads | Superseded by |
|---|---|---|---|
| 1 | 9.62 item 1 | "The pass rate under drift is 16 of 18 on family means and 4 of 18 on the hard pairs alone" | 9.75 item 1 (two panel versions: 13/2/9 of 15 and 3/2/3 of 3); 9.81 item 1 (16 / **5** / 12 of 18 on one panel version; "2 of 15" is the honest estimate) |
| 2 | 9.64 heading and item 2 | "The share works through retention, not acquisition"; "by checkpoint 4 (the end of the composition blocks)" | 9.75 items 2 and 3 (an allocation result; "retention, not acquisition" withdrawn as a mechanism; checkpoint 4 is not the end of the compositions under drift) |
| 3 | 9.62 item 3 | The late `+-` loss read against "the window S looks at", as forgetting after the compositions | 9.75 item 2 (net change under continuing supervision) |
| 4 | 9.66 item 2 and Reading | "The first bank has to be born by backprop and frozen" | 9.67 item 2 (re-read as a statement about the rule as implemented); 9.70 item 1 (withdrawn) |
| 5 | 9.67, Phase 4 prepared | "Checked (`e17p4_check.py`) … equals a direct computation exactly" (presented as the check) | 9.75 item 6 (not independent; replaced by `e17p4_independent_check.py`) |
| 6 | 9.67 item 4 | The forward-only form "needs, for every synapse above a write, one trace per bank unit" | 9.72 item 3 (one per distinct upper rate for leaky layers with linear input) |
| 7 | 9.70 heading and item 1 | "the windowless learner keeps its birth with nothing frozen"; "the unfrozen learner does not lose its birth" | 9.75 item 5a ("overstated": birth operations fall to 59.2 / 47.0 / 60.2, then 10.7 / 22.2 / 43.2) |
| 8 | 9.70 item 3 | The one-layer against every-layer contrast read as time reach alone | 9.75 item 5c (the arms also differed in feedback map) |
| 9 | 9.75 item 9 | Strict re-score: "−1.2 to +1.7 … every ratio by at most 0.025 … did not inflate these scores measurably" | 9.79 item 3 (the re-score was not strict; withdrawn) |
| 10 | 9.75 item 4 | "the one-factor reading is withdrawn for that seed" (E16-M2, 600013) | 9.81 item 3 (lifted for this claim on the new same-panel receipts) |
| 11 | 9.76 table header and "Fact" paragraph | "Taught compositions"; "under backprop it half-learns the compositions it is taught" | 9.79 item 1 (`within` is new instances of taught pairs, not taught instances) |
| 12 | Coordinator note §1 | "under backprop it reaches 49–59% on the compositions it is taught" | 9.79 item 1, as in row 11 |
| 13 | Coordinator note §2, last row | "this did not inflate scores measurably; strict panels move ratios by at most 0.025" | 9.79 item 3 (the strict re-score was withdrawn); the row cites 9.75 only |

Two further statements fall outside the numeric ledger. First, "the owner's to decide" in 9.60, 9.67, 9.70, 9.71 and 9.79 was closed for three questions by the 9.82 addendum, which itself names those sections. Second, 9.76's hold on rule variants was changed by 9.82 ("Earlier decisions this changes").

---

## 5. Claims not traceable to a JSON receipt (marked "not recomputed")

| Claim | Where | Why |
|---|---|---|
| Wall-clock times, forecasts, "within N% of forecast", lock-release times, "4-core 2.10 GHz" | 9.59, 9.61, 9.62, 9.64, 9.66, 9.67, 9.70, 9.71, 9.81 | Recorded in logs and briefs. Receipts carry CPU time only, and no machine field. |
| Embedding rows = birth × 9.9387e-5 (measured); final/birth = 1.0000; row norm 4.1 → 0.3; signs −0.40 to +1.56 | 9.60, 9.61 | Need the saved weights (`.pt`), which this job does not open |
| Strict re-score (−1.2 to +1.7; ≤ 0.025; 1600033 → 0.798) | 9.75 item 9, 9.79 item 4, coordinator note row 8 | Made from weights; removed from `E16_AUDIT_TABLES.json`; withdrawn by 9.79 |
| Store contributes nothing on single operations, all fifteen seeds | 9.75 item 10 | The fifteen cloud receipts have no no-store field |
| Start-of-sequence row never receives a gradient | 9.75 item 6 | Not in `E17P4_INDEPENDENT_CHECK.json` |
| The two Phase 4 arms used raw against scaled sign maps | 9.75 item 5c | No `projfb` field in the Phase 4 receipts; this rests on source |
| Fresh compositions until three quarters of the life; equal weights at checkpoint 4 | 9.75 item 2 | Receipts hold totals per block only (consistent: fresh F2 in block 5, none in block 6) |
| Exposure overlaps 40–52%, 47 / 44 / 21% | 9.60 | Memo script output; no JSON |
| Prediction scorecard 17 of 33; 1 of 12 | 9.60 | Counts over prediction files |
| E20-A v1 panel census: 6.6% in view, 88.7% hidden | 9.79 | Only in `PROJECT_DOCUMENTS/AUDIT_RESPONSE_2_20261110.md` |
| E18 extra head, 657,153 parameters | 9.71 item 2 | No parameter count in the E18 birth receipts |
| "The reference taught without letters ended at 85–89" | 9.77 item 4 | No receipt named. Candidates: staged D 84.5–88.2; D_drift 86.3–88.5; D_drift_cp 85.5–87.0. |
| Coordinator's own independent E19 check (5.5e-16 to 5.8e-16) | Coordinator note row 7 | Recorded only in `exp45/E19_COORDINATOR_CHECK.md`. The worker's `E19_CHECK.json` is recomputed and agrees. |
| P-Z.1–3 and P-E.1–4 lives | 9.82 | Only dry receipts are on this branch (`exp45/archive_nextsteps_dry/`) |
| P16M4.1–5 | 9.68 | Lives stopped at checkpoint 5, as 9.68 says |
| Literature verdicts | 9.72 | No receipts; outside this job |
| Commit provenance: 8e958da, ac98a824, 1c7ebee, d96129c7, e78aa83e, 41f2ba2a, bec6d07f, cf2fea36, a3c0b675, e97fa270, 0ea40628, 0ea4ce45, 9f0ed126 | Several | Present in this clone's object store. Their content and order were not audited. **c5e2715** (Codex's first audit, 9.75) is **not in this clone**. |

---

## 6. Audit notes (template fields that apply)

- **Panels.** The three development seeds' original drift receipts are on the original composition panels, and the fifteen later seeds are on the corrected panels. The `_cp` repeats are on the corrected panels and carry a hash.
- **Bound comparisons.** Only the 9.81 `_cp` receipts and the 9.77 E20-A receipts carry a completion record and a panel hash. Every other number in this ledger rests on legacy receipts.
- **Machines.** No receipt names its machine. "One machine" statements, such as 9.81's, are the review's.
- **Repeats against independent samples.** E16-F6 run B is a repeat; its scores equal run A field for field. On the owner's machine the `_cp` drift repeats reproduce the original single-operation scores to the digit.
- **Matched baselines.** The E17 twins run at wd 0.30 against 0.25 for the arms. E19's `bptt` runs at decay 0.30 against 0.25 for `r1`/`r2`. Both facts are in the receipts. 9.79 already states the E19 consequence.

---

## Appendix A. Full script output (run of 11 November 2026, ref `df09c03a`)

```
claim-id | claimed | recomputed | match
9.59-a births 800011/900007/1000003 at k=12 (dev ops) | 89–98 | 89.0–98.3 | MATCH
9.59-b twins at 90/85/90 or above (six F6 seeds) | >= 90 / 85 / 90 | min 91.8 / 88.4 / 90.7 | MATCH
9.59-c 500011 twin compositions (drift) | 88.4 | 88.42 | MATCH
9.59-c' 500011 twin compositions under stages | 84.0 | 83.96 | MATCH
9.59-d learners >= 0.80 on all three families | 5 of 6 | 5 of 6 | MATCH
9.59-e composition ratios of the five passing seeds | 0.83–0.89 | 0.825–0.888 | MATCH
9.59-f 600013 compositions ratio, mod-97 ratio | 0.705 / 0.774 | 0.7051 / 0.7741 | MATCH
9.59-g 600013 +- and -+ (final) | 42–48 | 41.7–47.8 | MATCH
9.59-g' 600013 max/min pairs (><, <> mean, final) | 89 | 88.9 | MATCH
9.59-g'' 600013 add/sub mod 97 'near 50' | about 50 | 47.5–52.0 | MATCH
9.59-g''' 600013 twin 'untouched at 94' (twin mod-97 mean) | 94 | 94.1 | MATCH
9.59-h Spearman ck2 mod-97 vs final +- (12 seeds) | 0.217 | 0.2168 | MATCH
9.59-h' same, nine re-keyed-panel seeds | 0.450 | 0.4500 | MATCH
9.59-i seeds above 80 at ck2 ending <= 65 on +- | 4 | 4 (300007, 500011, 800011, 900007) | MATCH
9.59-j 600013 mod-97 at ck3, ck2 | 47.2 / 79.9 | 47.25 / 79.94 | MATCH
9.59-k life-minimum mod-97, failing seeds 600013, 400009 | 47.2 / 62.0 | 47.25 / 62.00 | MATCH
9.59-k' lowest life-minimum among the ten passing seeds | 66.7 | 66.69 | MATCH
9.59-l drift gate (family means) over twelve seeds | 10 of 12 | 10 of 12 | MATCH
9.60(4) run B = run A in twelve receipts and three births | bit-identical (15 files) | 15 of 15 equal in every field except 'cpu'; 0 of 15 equal including 'cpu'  [CPU-time fields differ, as expected between two containers] | MATCH
9.59-m pace 4.7 h against 5.3 forecast; within 3% of forecast CPU | 4.7 h / 5.3 h / 3% | not recomputed: wall time and forecasts are in logs and the brief, not in JSON receipts | NOT_RECOMPUTED
9.60-a embedding rows = birth x 9.9387e-5 (local arms, Phase 1) | 9.9387e-5 | not recomputed: needs the saved weights (.pt); this script opens no .pt file | NOT_RECOMPUTED
9.60-a' arithmetic: AdamW decay alone over 36,864 updates at 0.25 x 0.001 | 9.9387 | 9.93211  [value x 1e5] | DIFFER
9.60-a'' per block of 6,144 updates | 0.215 | 0.2152 | MATCH
9.60-b first update: local arm rows without gradient / rows in batch; backprop | 36 of 47; 2 | 36 of 47; 2 (seed 73129, Phase 2 receipts)  [the in-harness check is recorded in the Phase 2 receipts; Phase 1 receipts carry no such field] | MATCH
9.60-c raw-reservoir lives: count; pass family / whole / add-sub97 / +- -+ | 21; 18 / 14 / 15 / 4 | 21; 18 / 14 / 15 / 4 | MATCH
9.60-c' not met on +- pairs alone | 17 of 21 | 17 of 21 | MATCH
9.60-d comparison pairs (><,<>) learner / reference, mean of 21 | 89.5 / 98.7 | 89.49 / 98.61 | DIFFER
9.60-d' plus-minus pairs learner / reference, mean of 21 | 64.5 / 86 | 64.49 / 86.3 | MATCH
9.60-d'' single-operator shortcut on ><,<> (mean of best ceiling) | 69 | 68.8 | MATCH
9.60-e exposure overlaps 40-52% / 47, 44, 21% | 40–52%; 47 / 44 / 21% | not recomputed: from the 9 November memo's own script; no JSON receipt of these shares found | NOT_RECOMPUTED
9.60-f E16-L add/sub after block 1 (three seeds) | 46.8 / 40.0 / 29.8 / 22.5 / 21.5 / 14.2 | 46.75 / 40.00 / 29.75 / 22.50 / 21.50 / 14.25 | MATCH
9.60-f' E16-L add/sub under 12 after block 2, under 6 after block 3 | < 12; < 6 | max 11.8; max 3.5 | MATCH
9.60-g E16-RP 93683 +- at block 6, run on new panels minus run on old panels | 16.2 | 16.17 | MATCH
9.60-g' same weights, new minus old panel, +- (93683) | 6.8 | 6.83 | MATCH
9.60-g'' run-to-run part 'about 9 points' | 9 | 9.3 | MATCH
9.60-h frozen twin (E17 Phase 1b, D_bank2_frz1) mod-97 by block | 89.4–95.0 | 89.38–95.00 | MATCH
9.60-i E16-RC three-seed composition mean, 16,384 / 32,768 / 65,536 / 131,072 | 74.2 / 76.9 / 78.3 / 79.8 | 74.17 / 76.94 / 78.35 / 79.82  [65,536 = E16-R3 receipts (*_oldpanels), as in the E16-RC report] | MATCH
9.60-j prediction scorecard 17 of 33; 1 of 12 | 17 of 33; 1 of 12 | not recomputed: a count over prediction files, not receipts; outside this job's scope | NOT_RECOMPUTED
9.61-a P17.2.3 unfrozen sign rule F1 after block 1 | 2.8 / 1.6 / 1.5 | 2.75 / 1.56 / 1.50 | MATCH
9.61-a' Phase 1 (with decay) F1 after block 1 | 1.5 / 1.9 / 1.9 | 1.50 / 1.88 / 1.94 | MATCH
9.61-b P17.2.4 frozen sign rule F1 after block 2, block 6 | 82.1 / 86.9 / 83.9 / 59.4 / 62.4 / 61.2 | 82.13 / 86.94 / 83.94 / 59.38 / 62.44 / 61.25 | MATCH
9.61-b' with decay (Phase 1b) after block 2, block 6 | 21 / 15 / 32 / 20 / 18 / 17 | 21.3 / 15.3 / 31.8 / 19.9 / 17.6 / 17.2 | MATCH
9.61-b'' 73129 misses 60 by | 0.6 | 0.63 | MATCH
9.61-c P17.2.5 arm after block 2, block 6 | 93.7 / 93.6 / 95.0 / 87.5 / 90.1 / 91.9 | 93.69 / 93.56 / 95.00 / 87.50 / 90.06 / 91.88 | MATCH
9.61-c' frozen twin after block 2, block 6 | 94.1 / 95.0 / 93.1 / 90.2 / 90.0 / 91.5 | 94.13 / 95.00 / 93.13 / 90.19 / 90.00 / 91.50 | MATCH
9.61-d P17.2.6 held pairs XT arm; frozen twin | 13.4 / 14.3 / 15.0 / 14.9 / 17.8 / 18.1 | 13.42 / 14.29 / 15.00 / 14.88 / 17.75 / 18.13 | MATCH
9.61-d' XT arm +- ; -+ | 3.5 / 2.7 / 2.8 / 3.2 / 1.2 / 1.3 | 3.50 / 2.67 / 2.83 / 3.17 / 1.17 / 1.33 | MATCH
9.61-e zero-gradient rows on first update: unfrozen local arm; backprop | 36 of 47; 2 | 36/47; 36/47; 36/47 | 2; 2; 2 | MATCH
9.61-f final/birth = 1.0000 for unreached rows; digit-row norm 4.1 -> 0.3, final/birth -0.40 to +1.56 | 1.0000; 4.1 -> 0.3 | not recomputed: needs the saved weights (.pt) | NOT_RECOMPUTED
9.61-g unfrozen: development ops at birth | 93–95 | 93.2–94.8 | MATCH
9.61-g' unfrozen: development ops after block 1 | 1–2 | 1.1–2.0 | MATCH
9.61-g'' unfrozen: mod-97 never above 3 after birth | < 3 | max 2.8 | MATCH
9.61-h frozen sign rule F1 through blocks 1-2 | 82–87 | 81.4–86.9 | DIFFER
9.61-h' frozen sign rule low point in composition blocks (min of blocks 3-4) | 51–54 | 50.8–53.8 | MATCH
9.61-h'' frozen sign rule end | 59–62 | 59.4–62.4 | MATCH
9.61-h''' frozen twin end | 90–92 | 90.0–91.5 | MATCH
9.61-i frozen sign rule add / sub at end, three seeds | 41 / 24 / 46 / 28 / 46 / 23 | 41.3 / 24.0 / 45.5 / 27.8 / 45.8 / 23.0 | MATCH
9.61-j frozen sign rule ratio to frozen twin, mod-97 | 0.66–0.69 | 0.658–0.694 | MATCH
9.61-j' ... mod-73 | 0.77–0.81 | 0.765–0.805 | MATCH
9.61-j'' ... held compositions | 0.39–0.52 | 0.389–0.516 | MATCH
9.61-k XT arm mod-97 within 3 of frozen twin, above in two | |d| <= 3; 2 above | -2.7, 0.1, 0.4 | MATCH
9.61-k' XT arm mod-73 above twin by | 2–3 | 1.9–2.8 | MATCH
9.61-k'' XT arm add/sub at end | 84–91 | 84.3–91.0 | MATCH
9.61-k''' XT arm held compositions ratio to twin | 0.81–0.90 | 0.805–0.902 | MATCH
9.61-l frozen twin +- ; -+ | 2.5 / 10.7 / 7.3 / 2.5 / 1.3 / 2.0 | 2.50 / 10.67 / 7.33 / 2.50 / 1.33 / 2.00 | MATCH
9.61-m CPU-s per life, unfrozen sign rule | 7,475–7,515 | 7474.6–7515.1 | MATCH
9.61-m CPU-s per life, frozen sign rule | 6,853–6,923 | 6853.4–6922.5 | MATCH
9.61-m CPU-s per life, XT arm | 6,667–6,808 | 6667.1–6808.2 | MATCH
9.61-m CPU-s per life, frozen twin | 5,611–5,738 | 5611.1–5738.0 | MATCH
9.61-m CPU-s per life, unfrozen twin | 2,603–2,624 | 2603.3–2624.3 | MATCH
9.61-n wall 2 h 29 min; forecast; lock release 22:08 | 2 h 29 min | not recomputed: wall clock and lock are in logs, not JSON receipts | NOT_RECOMPUTED
9.66-a P17.3.1 exact feedback alone F1 after block 6 | 67.9 / 72.9 / 72.8 | 67.88 / 72.88 / 72.75 | MATCH
9.66-a' misses 'within 8 of frozen twin' in all three | all three > 8 below | 22.3, 17.1, 18.8 | MATCH
9.66-b P17.3.2 trace alone F1 after block 6 | 80.1 / 84.9 / 84.9 | 80.13 / 84.94 / 84.88 | MATCH
9.66-b' trace alone above frozen sign rule by | 21–24 | 20.8–23.6 | MATCH
9.66-b'' trace alone above exact-feedback-alone by '12' | 12–12 | 12.1–12.3 | MATCH
9.66-c P17.3.3 embeddings frozen: development ops after block 1 | 1.3 / 1.5 / 1.2 | 1.33 / 1.50 / 1.25 | MATCH
9.66-c' F1 after block 1 | 1.8 / 1.8 / 1.1 | 1.81 / 1.75 / 1.06 | MATCH
9.66-d P17.3.4 held pairs, feedback; trace | 8.8 / 8.3 / 8.8 / 12.2 / 13.2 / 12.0 | 8.75 / 8.33 / 8.75 / 12.21 / 13.25 / 11.96 | MATCH
9.66-d' hard pairs (+-, -+ each) in the three arms | 0–4 | 0.2–2.3 | WITHIN
9.66-e trace alone ratio to frozen twin, mod-97 | 0.89 / 0.94 / 0.93 | 0.888 / 0.944 / 0.928 | MATCH
9.66-e' ... mod-73 | 0.97–1.00 | 0.973–1.004 | MATCH
9.66-e'' trace alone add / sub | 78 / 66 / 84 / 77 / 83 / 72 | 78.3 / 65.5 / 84.0 / 76.8 / 82.8 / 72.0 | MATCH
9.66-e''' trace alone loss during composition blocks (block 2 - min(blocks 3, 4)) | 10–13 | 11.6–12.6 | WITHIN
9.66-f exact feedback alone adds to the sign rule (block 6) | 8–12 | 8.5–11.5 | MATCH
9.66-g embeddings frozen: mod-97 never above 2 after birth | <= 2 | max 1.9 | MATCH
9.66-h 'every arm sits at 8-13 on the held pairs' (XF, TU, FB) | 8–13 | 1.0–13.3  [XF and TU are 8.3-13.2; the embeddings-frozen arm FB is at 1.0-2.0] | DIFFER
9.66-h' frozen twin held pairs | 15–18 | 14.9–18.1 | MATCH
9.66-i CPU-s per life, exact feedback alone | 4,840–4,878 | 4840.0–4877.9 | MATCH
9.66-i CPU-s per life, trace alone | 5,129–5,152 | 5128.9–5151.8 | MATCH
9.66-i CPU-s per life, embeddings frozen | 5,574–5,602 | 5573.7–5601.8 | MATCH
9.66-j wall 1 h 35 min, forecast 2 h 15 | 1 h 35 min | not recomputed: wall clock is in logs, not JSON receipts | NOT_RECOMPUTED
9.62-a P16F7.1 lowest birth op (sub89, 1400017) | 86.3 | 86.33 | MATCH
9.62-a' the lowest is sub89 of 1400017 | sub89, 1400017 | ('sub89', 1400017) | MATCH
9.62-b P16F7.2 lowest twin means (mod-97 / comps / mod-73) | 93.7 / 91.2 / 92.2 | 93.69 / 91.21 / 92.25 | MATCH
9.62-c P16F7.3 at >= 0.80 on all three families | 6 of 6 | 6 of 6 | MATCH
9.62-c' 1600033 passes by (mod-97 ratio - 0.80) | 0.005 | 0.0048 | MATCH
9.62-d P16F7.4 Spearman S vs final +- over eighteen seeds | 0.109 | 0.1094 | MATCH
9.62-d' Spearman among the six new seeds (exploratory) | -0.71 | -0.714 | MATCH
9.62-d'' seeds with S below 65 are the two known failures | 400009, 600013 | 400009, 600013 | MATCH
9.62-e P16F7.5 new seeds passing on +-/-+ ; family gate | 1 of 6; 6 of 6 | 1 of 6; 6 of 6 | MATCH
9.62-f eighteen seeds: family means; hard pairs alone (mixed panel versions, see 9.75) | 16 of 18; 4 of 18 | 16 of 18; 4 of 18 | MATCH
9.62-g mean +-/-+ learner; twin (18 seeds) | 64.8 / 87.5 | 64.83 / 87.47 | MATCH
9.62-h 1400017 S; final +- (best of 18) | 65.2 / 79.0 | 65.19 / 79.00 | MATCH
9.62-h' 1400017 has the best final +- of all eighteen | best | best | MATCH
9.62-h'' S of 1200007, 1300021 ('near 82'); final +- | 82 / 82 / 49.7 / 51.5 | 81.8 / 82.1 / 49.67 / 51.50 | MATCH
9.62-i +- loss ck4 -> ck6, 1200007; 1300021 | 15 / 30 | 15.0 / 29.7 | MATCH
9.62-i' 1200007 passes family gate with +- at; failing 400009 +- | 49.7 / 48.0 | 49.67 / 48.00  [review writes 'level with the failing 400009' without a number; 48.0 is the receipt value] | MATCH
9.62-j total CPU-s, eighteen runs | 79,218 | 79217.5 | DIFFER
9.62-j' births CPU-s | 3,051–3,164 | 3051.4–3163.5 | MATCH
9.62-j'' learners CPU-s | 5,329–5,402 | 5328.8–5402.5 | MATCH
9.62-j''' twins CPU-s | 4,643–4,848 | 4642.8–4847.6 | MATCH
9.62-k 5 h 37 min wall; 4% over forecast; 4-core 2.10 GHz | 5 h 37 min | not recomputed: wall time and machine are in logs; receipts carry no machine field | NOT_RECOMPUTED
9.64-a share table, seed 400009 (rule 0/.25/.5/.75, backprop .25/.5/.75, twin) | 1.2 / 43.7 / 46.3 / 73.2 / 78.2 / 84.7 / 88.1 / 89.1 | 1.25 / 43.67 / 46.33 / 73.25 / 78.25 / 84.67 / 88.08 / 89.08 | MATCH
9.64-a share table, seed 600013 (rule 0/.25/.5/.75, backprop .25/.5/.75, twin) | 1.0 / 23.3 / 44.8 / 60.2 / 71.9 / 81.8 / 86.7 / 91.2 | 1.00 / 23.33 / 44.75 / 60.25 / 71.92 / 81.83 / 86.67 / 91.17 | MATCH
9.64-a share table, seed 300007 (rule 0/.25/.5/.75, twin) | 1.1 / 59.2 / 67.7 / 75.7 / 85.8 | 1.08 / 59.17 / 67.67 / 75.67 / 85.75 | MATCH
9.64-b P16M3.1 share-0 mod-97 ratios | 0.35 / 0.32 / 0.13 | 0.347 / 0.321 / 0.132 | MATCH
9.64-c P16M3.2 rule hard pairs 0.75 minus 0.5 (400009, 600013) | 27 / 15 | 26.9 / 15.5 | MATCH
9.64-d P16M3.3 backprop 0.5 minus 0.25 hard pairs (400009, 600013) | 6 / 10 | 6.4 / 9.9 | MATCH
9.64-e P16M3.4 300007 comp ratio at 0.25; hard 0.25 ; 0.75 | 0.796 / 59.2 / 75.7 | 0.7956 / 59.17 / 75.67 | MATCH
9.64-f rule hard pairs gain 0.25 -> 0.75 (400009, 600013); backprop | 29 / 37 / 10 / 15 | 29.6 / 36.9 / 9.8 / 14.8 | DIFFER
9.64-g 400009 at 0.75: family ratios; hard ratio | 0.93 / 0.87 / 0.97 / 0.82 | 0.928 / 0.869 / 0.967 / 0.822 | MATCH
9.64-g' 600013 at 0.75: comps ratio; hard ratio | 0.796 / 0.66 | 0.7965 / 0.661 | MATCH
9.64-g'' 300007 at 0.75 hard ratio | 0.88 | 0.882 | MATCH
9.64-h share 0 hard pairs at end ('about 1%') | 1–1 | 1.0–1.3 | MATCH
9.64-h' share 0 peak hard pairs (max of ck3, ck4) | 59–85 | 58.7–85.3 | MATCH
9.64-h'' share 0 mod-97 ratio range | 0.13–0.35 | 0.132–0.347 | MATCH
9.64-i 400009 hard-pair peak at ck3, share 0.5 ; 0.75 | 57 / 41 | 57.4 / 40.8 | MATCH
9.64-j rule lives CPU-s (this job's nine) | 5,132–5,441 | 5132.4–5440.8 | MATCH
9.64-j' backprop lives CPU-s (this job's five) | 4,665–4,733 | 4665.3–4733.3 | MATCH
9.64-k fourteen drift lives in the job | 14 | 14 | MATCH
9.64-l 5.0 h wall; 3-9% over forecast | 5.0 h | not recomputed: wall time is in logs, not JSON receipts | NOT_RECOMPUTED
9.67-a embeddings: sign as run ; exact ; through time | 0.13 / 0.28 / 0.32 / 0.20 / 0.33 / 0.36 / 0.51 / 0.77 / 0.53 | 0.132 / 0.281 / 0.320 / 0.196 / 0.330 / 0.359 / 0.505 / 0.768 / 0.531 | MATCH
9.67-b first layer: sign ; exact ; through time | 0.24 / 0.16 / 0.32 / 0.32 / 0.24 / 0.41 / 0.47 / 0.60 / 0.45 | 0.238 / 0.159 / 0.320 / 0.318 / 0.244 / 0.413 / 0.470 / 0.599 / 0.450 | MATCH
9.67-c first bank projection: sign ; trace ; exact ; through time | 0.09 / -0.06 / 0.05 / 0.54 / 0.61 / 0.34 / 0.08 / -0.01 / 0.26 / 0.81 / 0.78 / 0.55 | 0.090 / -0.059 / 0.052 / 0.540 / 0.608 / 0.344 / 0.083 / -0.006 / 0.260 / 0.814 / 0.780 / 0.549 | MATCH
9.67-d first bank write (lc, lw): sign | 0.55–0.67 | 0.548–0.666 | MATCH
9.67-d' first bank write: exact | 0.59–0.88 | 0.589–0.875 | MATCH
9.67-d'' first bank write: through time | 0.71–0.78 | 0.711–0.782 | MATCH
9.67-e second bank projection: sign | 0.52–0.57 | 0.520–0.572 | MATCH
9.67-e' second bank projection: trace | 0.78–0.80 | 0.780–0.797 | MATCH
9.67-e'' second bank projection: exact | 0.57–0.66 | 0.571–0.665 | MATCH
9.67-e''' second bank projection: through time | 0.78–0.80 | 0.781–0.798 | MATCH
9.67-f second bank write, readout: sign (lc2, lw2, lm, lm2, l2) | 0.65–0.81 | 0.649–0.813  [output layer l3 has cosine 1.00 and is not in this range] | MATCH
9.67-g every layer under through-time credit | 0.45–0.81 | 0.450–0.814  [excluding output layer l3 (1.00)] | MATCH
9.67-h Phase 4 option vs plain rule, max relative error ('2e-7') | 2 | 2.2  [x 1e7] | MATCH
9.67-h' option equals direct computation exactly | 0 | 0.0 | MATCH
9.67-h'' ms per update: new arm; plain rule; backprop | 48.7 / 54.7 / 49.4 | 48.70 / 54.73 / 49.43 | MATCH
9.67-i comparator D_bank2_nbd mod-97 ; mod-73 after block 6 | 92.3 / 93.7 / 93.3 / 92.6 / 91.6 / 91.9 | 92.31 / 93.69 / 93.31 / 92.56 / 91.63 / 91.88 | MATCH
9.67-j dry runs: seven stages; rows without gradient one-layer trace / new arm | 7; 36 of 47 / 2 | [7]; 36 / 2 (backprop 2) | MATCH
9.67-l window learner: 'freeze: none, frozen: [] in every receipt' (18 drift learners) | 18 of 18 | 6 of 18 carry both fields as stated; 6 have no freeze field, 12 no frozen field  [absent fields are older receipts written before the option existed; none records freezing] | DIFFER
9.67-k forecast 2 h 20 min | 2 h 20 min | not recomputed: a forecast, not a measurement | NOT_RECOMPUTED
9.70-a table row T_bank2_nbd_tall 73129 (blocks 1-6, mod-73, ratios) | 92.9 / 90.2 / 82.9 / 80.6 / 86.0 / 87.9 / 90.8 / 0.95 / 0.98 | 92.88 / 90.19 / 82.94 / 80.63 / 86.00 / 87.88 / 90.75 / 0.952 / 0.980 | MATCH
9.70-a table row T_bank2_nbd_tall 93683 (blocks 1-6, mod-73, ratios) | 92.1 / 93.0 / 84.9 / 82.0 / 82.2 / 86.1 / 86.9 / 0.92 / 0.95 | 92.06 / 93.00 / 84.88 / 82.00 / 82.19 / 86.06 / 86.88 / 0.919 / 0.948 | MATCH
9.70-a table row T_bank2_nbd_tall 116089 (blocks 1-6, mod-73, ratios) | 90.7 / 92.4 / 75.5 / 81.6 / 83.9 / 86.9 / 89.4 / 0.93 / 0.97 | 90.69 / 92.44 / 75.50 / 81.63 / 83.94 / 86.94 / 89.38 / 0.932 / 0.973 | MATCH
9.70-a table row T_bank2_nbd_tup 73129 (blocks 1-6, mod-73, ratios) | 65.0 / 65.5 / 35.6 / 31.8 / 41.9 / 44.2 / 49.9 / 0.48 / 0.54 | 65.00 / 65.50 / 35.63 / 31.75 / 41.94 / 44.25 / 49.88 / 0.479 / 0.539 | MATCH
9.70-a table row T_bank2_nbd_tup 93683 (blocks 1-6, mod-73, ratios) | 75.3 / 75.7 / 40.0 / 29.1 / 42.1 / 42.1 / 47.2 / 0.45 / 0.52 | 75.31 / 75.69 / 40.00 / 29.13 / 42.06 / 42.06 / 47.19 / 0.449 / 0.515 | MATCH
9.70-a table row T_bank2_nbd_tup 116089 (blocks 1-6, mod-73, ratios) | 74.9 / 74.2 / 39.2 / 37.6 / 41.4 / 42.6 / 46.9 / 0.46 / 0.51 | 74.88 / 74.19 / 39.19 / 37.56 / 41.44 / 42.63 / 46.94 / 0.457 / 0.511 | MATCH
9.70-a table row M_bank2_nbd 73129 (blocks 1-6, mod-73, ratios) | 93.6 / 93.2 / 87.6 / 89.4 / 92.2 / 90.9 / 93.5 / 0.98 / 1.01 | 93.56 / 93.25 / 87.56 / 89.38 / 92.19 / 90.88 / 93.50 / 0.984 / 1.010 | MATCH
9.70-a table row M_bank2_nbd 93683 (blocks 1-6, mod-73, ratios) | 94.5 / 93.2 / 90.8 / 86.1 / 92.8 / 92.3 / 93.2 / 0.99 / 1.02 | 94.50 / 93.25 / 90.75 / 86.13 / 92.75 / 92.31 / 93.25 / 0.985 / 1.018 | MATCH
9.70-a table row M_bank2_nbd 116089 (blocks 1-6, mod-73, ratios) | 93.8 / 94.0 / 90.2 / 92.4 / 93.4 / 94.2 / 93.2 / 1.01 / 1.01 | 93.81 / 94.00 / 90.25 / 92.44 / 93.44 / 94.25 / 93.19 / 1.010 / 1.014 | MATCH
9.70-b twin block 1 | 92.2–93.6 | 92.19–93.56 | MATCH
9.70-b' twin block 6 | 92.3–93.7 | 92.31–93.69 | MATCH
9.70-b'' twin mod-73 | 91.6–92.6 | 91.63–92.56 | MATCH
9.70-c sign rule as run (Phase 2) block 1 | 1.5–2.8 | 1.50–2.75 | MATCH
9.70-c' ... block 6 | 0.7–1.2 | 0.69–1.19 | MATCH
9.70-c'' ... mod-73 at 6 | 1.1–2.2 | 1.06–2.25 | MATCH
9.70-d P17.4.5 held pairs new arm | 12.8–14.7 | 12.79–14.71 | MATCH
9.70-d' ... one-layer | 7.7–9.0 | 7.71–8.96 | MATCH
9.70-d'' ... backprop | 14.0–18.5 | 14.00–18.46 | MATCH
9.70-d''' hard pairs (mean of +- and -+), three arms | 0.8–8.6 | 0.83–8.58 | MATCH
9.70-e new arm block 1 ('91-93') | 91–93 | 90.7–92.9 | MATCH
9.70-e' new arm ends above Phase 3 frozen trace (80.1/84.9/84.9) in every seed | every seed | 7.8, 1.1, 2.1 | MATCH
9.70-f backprop above new arm, mod-97 | 3–7 | 3.0–7.3 | MATCH
9.70-f' backprop above new arm, mod-73 | 3–6 | 2.8–6.4 | MATCH
9.70-f'' add/sub mod 97: new arm | 77–86 | 77.3–86.3 | MATCH
9.70-f''' add/sub mod 97: backprop | 88–93 | 87.5–92.8 | MATCH
9.70-g max/min mod 97: new arm | 91–94 | 91.0–94.3 | MATCH
9.70-g' max/min mod 97: backprop | 92–97 | 91.8–97.0 | MATCH
9.70-g'' pipeline cost to backprop (ratios to twin) | 0.98–1.02 | 0.984–1.018 | MATCH
9.70-h one-layer trace block 1 | 65–75 | 65.0–75.3 | MATCH
9.70-h' one-layer trace end mod-97 | 42–44 | 42.1–44.3 | MATCH
9.70-h'' one-layer trace end mod-73 | 47–50 | 46.9–49.9 | MATCH
9.70-h''' one-layer loss when compositions enter (block 2 - min(blocks 3, 4)) | 30–46 | 33.8–46.6  [block 2 - block 3 alone gives 29.9-35.7] | DIFFER
9.70-i rows without gradient: one-layer 36 of 47; new arm 2 | 36 / 2 | [36, 36, 36] / [2, 2, 2] | MATCH
9.70-j development ops after block 6: new arm ; backprop ; twin | 10.7 / 22.2 / 43.2 / 48.6 / 41.4 / 53.6 / 49.1 / 37.9 / 51.5 | 10.67 / 22.17 / 43.17 / 48.58 / 41.42 / 53.58 / 49.08 / 37.92 / 51.50 | MATCH
9.70-k CPU-s per life, new arm | 6,672–6,787 | 6671.9–6787.4 | MATCH
9.70-k CPU-s per life, one-layer | 7,034–7,068 | 7033.8–7068.5 | MATCH
9.70-k CPU-s per life, backprop | 6,615–6,732 | 6615.0–6732.4 | MATCH
9.70-l nine lives loaded the same three births, seven stages | same birth per seed; 7 stages | birth hashes per seed: 73129:1 distinct, 93683:1 distinct, 116089:1 distinct | MATCH
9.70-m settings: new arm sign/all/none frozen; one-layer sign/proj; backprop rule D wd 0.25 rshare 0.5; twin wd 0.3 | as described | T_bank2_nbd_tall: rule=T fb=sign traceup=all freeze=none wd=0.25 rshare=0.5 projfb=None; T_bank2_nbd_tup: rule=T fb=sign traceup=proj freeze=none wd=0.25 rshare=0.5 projfb=None; M_bank2_nbd: rule=D fb=sign traceup=none freeze=none wd=0.25 rshare=0.5 projfb=None; D_bank2_nbd: rule=D fb=sign traceup=none freeze=none wd=0.3 rshare=None projfb=None | MATCH
9.70-n wall 2 h 04 min; forecasts | 2 h 04 min | not recomputed: wall time and forecasts are in logs, not JSON receipts | NOT_RECOMPUTED
9.75-a corrected panels: family / hard pairs / whole answer | 13 of 15; 2 of 15; 9 of 15 | 13 of 15; 2 of 15; 9 of 15 | MATCH
9.75-a' original panels (73129, 93683, 116089): family / hard / whole | 3 of 3; 2 of 3; 3 of 3 | 3 of 3; 2 of 3; 3 of 3 | MATCH
9.75-b fresh compositions arrive after checkpoint 4 and stop before the last block | fresh F2 until ~3/4 of life | block 5 fresh/F2 rows 25,799; block 6 fresh/F2 rows 0; replay/F2 rows in block 6 71,178 (73129 _cp receipt)  [only per-block totals are recorded; the exact 'three quarters' point and the equal weights at checkpoint 4 are not recomputable] | MATCH
9.75-c admitted rows at share 0.25 / 0.5 / 0.75 (400009) | 1,769,488 / 1,179,680 / 589,872 | 1769488.0 / 1179680.0 / 589872.0 | MATCH
9.75-d E16-M2 600013 checkpoint-0 compositions: learner/reference (original panel) ; M/oracle (corrected) | 25.2 / 24.6 | 25.21 / 24.58 | MATCH
9.75-d' panel versions of the four E16-M2 600013 arms | T, D original; M, oracle corrected | T_raw65536=original, D=original, M_raw65536=corrected, T_oracle=corrected | MATCH
9.75-e(5a) Phase 4 development ops after block 1: new arm ; reference | 59.2 / 47.0 / 60.2 / 62.8 / 58.0 / 67.1 | 59.25 / 47.00 / 60.17 / 62.83 / 58.00 / 67.08 | MATCH
9.75-e'(5a) ... at the end (new arm) | 10.7 / 22.2 / 43.2 | 10.67 / 22.17 / 43.17 | MATCH
9.75-f(5b) new arm at 40 distractors: add ; sub (final answer) | 18.2 / 9.2 / 28.2 / 16.8 / 25.8 / 19.0 | 18.25 / 9.25 / 28.25 / 16.75 / 25.75 / 19.00 | MATCH
9.75-f'(5b) ... against 77-86 at 12 | 77–86 | 77.3–86.3 | MATCH
9.75-g(5c) one-layer arm relayed raw signs, every-layer arm signs scaled by the mean weight | raw / scaled | not recomputed: the Phase 4 receipts have no projfb field (added after them); the claim rests on source code, which this script does not read | NOT_RECOMPUTED
9.75-g'(5c) cosine raw vs scaled projection gradients | 0.991–0.999 | 0.9905–0.9988 | MATCH
9.75-g''(5c) alignment with true gradient, raw ; scaled (lp) | 0.55 / 0.66 / 0.37 / 0.54 / 0.67 / 0.38 | 0.553 / 0.656 / 0.374 / 0.540 / 0.674 / 0.383 | MATCH
9.75-h(6) independent check, max error ('2e-15') | 2 | 2.1  [x 1e15] | MATCH
9.75-h'(6) start-of-sequence row never receives a gradient | never | not recomputed: stated from the independent check's run; the JSON holds per-tensor errors only | NOT_RECOMPUTED
9.75-i(7) E18 census at init: B3 symbols with gradient; B0/B1/B2 | 11 of 45; 45 / 45 / 45 | 11 of 45; 45 / 45 / 45 | MATCH
9.75-j(9) reversal in taught pool, mod-97 and mod-73 ('25%'; measured on seeds 73129, 400009, 1200007 only) | about 25 | 24.5–25.8 | MATCH
9.75-j'(9) ... at birth (mod 89) | 49–52 | 48.1–51.8 | DIFFER
9.75-j''(9) held mod-97 pairs supervised as a sub-step | 71–73 | 71.1–72.7 | MATCH
9.75-k(9) strict re-score: mod-97 mean moves -1.2 to +1.7 (mean +0.2), ratios by <= 0.025; 1600033 0.805 -> 0.798 | -1.2..+1.7; 0.025; 0.798 | not recomputed: made from saved weights (.pt); the JSON part was removed from E16_AUDIT_TABLES.json; withdrawn by 9.79 item 3 | NOT_RECOMPUTED
9.75-k'(9) 1600033 registered mod-97 ratio (receipt) | 0.805 | 0.8048 | MATCH
9.75-l(10) receipt and no-store scores identical in all fifteen seeds (single ops, k = 12) | identical | not recomputed: the fifteen cloud receipts carry no no-store field; the comparison was made from weights | NOT_RECOMPUTED
9.76-a table row: window reference (D_drift) (within ; +- -+ ; >< <>, whole answers) | 92.8 / 91.5 / 92.5 / 88.8 / 85.2 / 88.8 / 98.3 / 96.2 / 98.0 | 92.83 / 91.50 / 92.50 / 88.83 / 85.25 / 88.75 / 98.33 / 96.25 / 98.00 | MATCH
9.76-a table row: window learner (T drift) (within ; +- -+ ; >< <>, whole answers) | 77.5 / 84.5 / 82.8 / 67.8 / 66.0 / 44.7 / 89.1 / 88.2 / 76.0 | 77.50 / 84.50 / 82.83 / 67.75 / 66.00 / 44.67 / 89.08 / 88.17 / 76.00 | MATCH
9.76-a table row: windowless reference (D_bank2_nbd) (within ; +- -+ ; >< <>, whole answers) | 48.7 / 58.5 / 51.7 / 0.8 / 6.8 / 2.9 / 22.2 / 23.1 / 25.4 | 48.67 / 58.50 / 51.67 / 0.75 / 6.83 / 2.83 / 22.25 / 23.08 / 25.33 | DIFFER
9.76-a table row: windowless backprop on pipeline (M_bank2_nbd) (within ; +- -+ ; >< <>, whole answers) | 36.3 / 42.5 / 40.2 / 0.8 / 3.8 / 1.8 / 19.0 / 20.0 / 26.8 | 36.33 / 42.50 / 40.17 / 0.75 / 3.75 / 1.83 / 19.00 / 20.00 / 26.75 | MATCH
9.76-a table row: windowless weight signs through time (T tall) (within ; +- -+ ; >< <>, whole answers) | 16.5 / 24.2 / 25.5 / 0.4 / 2.4 / 1.0 / 19.9 / 20.4 / 20.7 | 16.50 / 24.17 / 25.50 / 0.42 / 2.33 / 1.00 / 19.92 / 20.33 / 20.67 | DIFFER
9.76-b windowless reference, untaught plus-minus pairs ('1-7%') | 1–7 | 0.8–6.8 | MATCH
9.76-c 'ignoring one operator' shortcut on comparison pairs | 66–69 | 65.0–70.0  [shortcut_ceilings ignore_o1_use_a / _b, both pairs, three seeds] | DIFFER
9.79-a(1) windowless reference on 'within' (new instances of taught pairs) | 49–59 | 48.7–58.5 | MATCH
9.79-a'(1) ... on untaught pairs | 1–7 | 0.8–6.8 | MATCH
9.79-b(2) window learner's held compositions with the state zeroed fall to zero | zero | 18 drift learners, per pair: whole answer on +-,-+ max 0.0%, on all four pairs max 6.0%; final answer max 28.2%  [receipt field silenced_compose; the review does not say which metric or which learners] | DIFFER
9.79-c(3) clean items: addition / max / min / subtraction | 361 / 125 / 136 / 1,656 | 361.0 / 125.0 / 136.0 / 1656.0 | MATCH
9.79-c'(3) registered items per operation (15 seeds x 400) | 6,000 | 6000.0 | MATCH
9.79-d(3) learner clean ; all: add, sub, min, max | 75.6 / 74.2 / 73.7 / 73.1 / 94.9 / 95.9 / 89.6 / 94.8 | 75.62 / 74.25 / 73.73 / 73.07 / 94.85 / 95.88 / 89.60 / 94.85 | MATCH
9.79-d'(3) reference on clean items: add, sub, min, max | 90.0 / 90.5 / 97.8 / 97.6 | 90.03 / 90.46 / 97.79 / 97.60 | MATCH
9.79-d''(3) max: learner clean below all by | 5 | 5.3 | MATCH
9.79-e(3) registered mod-97 panel: share supervised somewhere in another form ('94-98%') | 94–98 | 72.4–97.9  [add 94.0, sub 72.4, max 97.9, min 97.7; the 94-98 range holds for add, max, min only] | DIFFER
9.79-f(4) corrected-panel family count: registered; with 1600033 at 0.798 | 13 of 15; 12 of 15 | 13 of 15; strict value not recomputable  [12 of 15 follows arithmetically if 1600033 falls below 0.80; the 0.798 itself is from weights] | MATCH
9.79-g E20-A v1 panel: first operand in view 6.6%, hidden 88.7% | 6.6% / 88.7% | not recomputed: measured on the withdrawn v1 panel; recorded only in AUDIT_RESPONSE_2 (markdown), no JSON receipt | NOT_RECOMPUTED
9.79-h E19 linear-write birth by backprop, hard mean | 69.0 / 70.5 / 69.3 | 69.00 / 70.50 / 69.33 | MATCH
9.79-h' the gated network ('91-95'): reference births e17_dev_bank2_nbd | 91–95 | 90.3–94.5 | DIFFER
9.79-i P19.3 forward-only rule with weight signs; ratio to bptt | 43.2 / 45.8 / 54.5 / 0.63 / 0.65 / 0.79 | 43.17 / 45.83 / 54.50 / 0.626 / 0.650 / 0.786 | MATCH
9.79-j P19.4 r1 minus r2 (no deep traces) | 8.0 / 14.5 / 23.8 | 8.00 / 14.50 / 23.83 | MATCH
9.79-j' settings: r1 and r2 same decay; bptt at a different decay | r1 = r2; bptt differs | bptt decay 0.3 / r1 decay 0.25 / r2 decay 0.25 | MATCH
9.79-k ms per update: rule ; backprop (mean of three) | 220 / 103 | 220.0 / 102.9 | MATCH
9.71-a table row ref (hard mean; max/min) | 91.2 / 94.5 / 90.3 / 96.0 / 95.2 / 95.5 | 91.17 / 94.50 / 90.33 / 96.00 / 95.17 / 95.50 | MATCH
9.71-a table row B0 (hard mean; at 20; max/min) | 87.5 / 90.7 / 84.7 / 84.5 / 88.0 / 74.7 / 98.8 / 97.8 / 95.7 | 87.50 / 90.67 / 84.67 / 84.50 / 88.00 / 74.67 / 98.83 / 97.83 / 95.67 | MATCH
9.71-a table row B1 (hard mean; at 20; max/min) | 80.7 / 82.2 / 77.2 / 79.2 / 73.3 / 73.5 / 93.8 / 95.7 / 94.5 | 80.67 / 82.17 / 77.17 / 79.17 / 73.33 / 73.50 / 93.83 / 95.67 / 94.50 | MATCH
9.71-a table row B2 (hard mean; at 20; max/min) | 74.8 / 69.7 / 66.3 / 71.2 / 64.3 / 62.2 / 96.0 / 96.7 / 93.8 | 74.83 / 69.67 / 66.33 / 71.17 / 64.33 / 62.17 / 96.00 / 96.67 / 93.83 | MATCH
9.71-a table row B3 (hard mean; max/min) | 1.2 / 0.8 / 0.8 / 0.8 / 2.0 / 2.8 | 1.17 / 0.83 / 0.83 / 0.83 / 2.00 / 2.83 | MATCH
9.71-b final step and package settings | 24,576; B0 top/T/all/0.25, B1 local/D/none/0.30, B2 local/T/all/0.25, B3 local/T/none/0.25 | steps [24576]; B0 top/T/all/0.25; B1 local/D/none/0.3; B2 local/T/all/0.25; B3 local/T/none/0.25 | MATCH
9.71-c P18.1 first bank's own head, add89 | 27.3 / 29.7 / 28.3 | 27.33 / 29.67 / 28.33 | MATCH
9.71-d P18.3 B0 / reference | 0.96 / 0.96 / 0.94 | 0.960 / 0.959 / 0.937 | MATCH
9.71-e P18.4 B0 - B1 | 6.8 / 8.5 / 7.5 | 6.83 / 8.50 / 7.50 | MATCH
9.71-f P18.5 B2 - B1 | -5.8 / -12.5 / -10.8 | -5.83 / -12.50 / -10.83 | MATCH
9.71-g P18.6 B3 max/min below B2's by | 91–95 | 91.0–95.2 | MATCH
9.71-h B0 - B2 | 12.7 / 21.0 / 18.4 | 12.67 / 21.00 / 18.33 | DIFFER
9.71-i B0 still rising at the last checkpoint (previous -> last) | 84 / 88 / 88 / 91 / 78 / 85 | 83.7 / 87.5 / 87.5 / 90.7 / 78.2 / 84.7 | MATCH
9.71-j first bank's own head, hard mean ('about 20') | about 20 | 19.5–20.0 | MATCH
9.71-j' upper bank adds ('about 60 points'): B1 hard mean minus head1 hard mean | about 60 | 57.7–62.2 | MATCH
9.71-j'' first bank alone answers additions | 27–30 | 27.3–29.7 | MATCH
9.71-j''' package (B1) under the end-to-end reference ('about 10 points') | about 10 | 10.5–13.2 | DIFFER
9.71-k B3 operator, letter, task-code rows unmoved; digit rows moved ~0.7 of a row norm | 0.00; 0.7 | operator [0.0, 0.0, 0.0], letter [0.0, 0.0, 0.0], task code [0.0, 0.0, 0.0]; digit [0.7005, 0.6797, 0.6886] | MATCH
9.71-l extra head of 657,153 parameters | 657,153 | not recomputed: the E18 birth receipts carry no parameter count | NOT_RECOMPUTED
9.71-m CPU-s for twelve births | 65,118 | 65116.6 | DIFFER
9.71-n 4 h 39 min wall; 5-6% over forecast | 4 h 39 min | not recomputed: wall time is in logs, not JSON receipts | NOT_RECOMPUTED
9.81-a twelve receipts: seven stages and a completion record with weight hash | 12 of 12 | 12 of 12  [that each weight hash matches its .pt file is not recomputed (no .pt opened)] | MATCH
9.81-b every comparison between receipts with the same panel hash | same hash | same hash | MATCH
9.81-c staged arms' checkpoint-0 composition score (400009; 600013) | 24.17 / 24.58 | 24.167 / 24.583 | MATCH
9.81-c' ... identical across the three arms of each seed | identical | {400009: ['24.17', '24.17', '24.17'], 600013: ['24.58', '24.58', '24.58']} | MATCH
9.81-d drift repeat 73129: ratios mod-97, held, mod-73; plus-minus; whole | 0.97 / 0.86 / 0.99 / 0.81 / 0.84 | 0.974 / 0.856 / 0.986 / 0.812 / 0.838 | MATCH
9.81-d drift repeat 93683: ratios mod-97, held, mod-73; plus-minus; whole | 0.93 / 0.87 / 0.96 / 0.82 / 0.84 | 0.933 / 0.874 / 0.962 / 0.815 / 0.843 | MATCH
9.81-d drift repeat 116089: ratios mod-97, held, mod-73; plus-minus; whole | 0.84 / 0.86 / 0.97 / 0.84 / 0.83 | 0.837 / 0.862 / 0.965 / 0.843 / 0.828 | MATCH
9.81-e staged held-pair ratio: learner (400009, 600013); backprop on pipeline | 0.83 / 0.64 / 0.98 / 0.94 | 0.830 / 0.644 / 0.977 / 0.940 | MATCH
9.81-e' plus-minus pairs learner / pipeline / reference (400009 ; 600013) | 67.4 / 86.8 / 90.4 / 46.6 / 80.9 / 89.2 | 67.42 / 86.75 / 90.42 / 46.58 / 80.92 / 89.17 | MATCH
9.81-f P-CP.1 'all nine ratios at or above 0.84' | >= 0.84 | min 0.837  [0.837 rounds to 0.84; strictly below] | DIFFER
9.81-g P-CP.4 held-pair mean with and without the store identical (five learner lives) | identical to two decimals | max |difference| 0.0000 | MATCH
9.81-h eighteen seeds on one panel version: family / plus-minus / whole | 16 / 5 / 12 of 18 | 16 / 5 / 12 of 18 | MATCH
9.81-i fifteen later seeds pass the hard-pair gate | 2 of 15 | 2 of 15 | MATCH
9.81-j repeats reproduce single-operation scores to the digit; held-pair means moved < 1 point | identical; < 1 | identical; max move 0.79 | MATCH
9.81-k CPU-s, learners | 6,275–6,307 | 6274.9–6306.9 | MATCH
9.81-k CPU-s, backprop on the pipeline | 5,718–5,742 | 5718.0–5741.9 | MATCH
9.81-k CPU-s, references | 5,675–5,807 | 5675.1–5806.6 | MATCH
9.81-l wall 2 h 17 min; lock released 16:44:44 | 2 h 17 min | not recomputed: wall clock and lock are in logs | NOT_RECOMPUTED
9.77-a plus-minus twins both right (of 300) | 63 / 80 / 130 | 63.0 / 80.0 / 130.0 | MATCH
9.77-a' ... per cent | 21.0 / 26.7 / 43.3 | 21.00 / 26.67 / 43.33 | MATCH
9.77-b same twins, single items right | 42.3 / 47.7 / 66.3 | 42.33 / 47.67 / 66.33 | MATCH
9.77-c in-view control (letters in front) | 78.3 / 70.0 / 80.7 | 78.33 / 70.00 / 80.67 | MATCH
9.77-d comparison twins both right | 63.7 / 58.7 / 56.0 | 63.67 / 58.67 / 56.00 | MATCH
9.77-e operand hidden, new instances ; taught triples | 72.3 / 72.3 / 85.7 / 82.7 / 78.3 / 84.3 | 72.33 / 72.33 / 85.67 / 82.67 / 78.33 / 84.33 | MATCH
9.77-f twins with the state zeroed, single items | 0.0 / 0.0 / 0.2 | 0.00 / 0.00 / 0.17 | MATCH
9.77-g P20.2 in-view minus single items | 36.0 / 22.3 / 14.3 | 36.00 / 22.33 / 14.33 | MATCH
9.77-h P20.3 one seed exactly on the mark (420 of 600) | 420 | 420.0 | MATCH
9.77-i seven stages, completion record, birth admitted by manifest, 73129 memory-panel hash = qualification, no store | all | all  [weight-hash-to-file match not recomputed (no .pt opened)] | MATCH
9.77-j registered letter-free plus-minus panel, whole answers | 69–81 | 68.7–80.7 | MATCH
9.77-j' 'the reference taught without letters ended at 85-89' (plus-minus, whole answers) | 85–89 | not recomputed: the review does not name the receipt; candidates: staged D 84.5–88.2; D_drift 86.3–88.5; D_drift_cp 85.5–87.0 | NOT_RECOMPUTED
9.77-k operand hidden, single items of untaught plus-minus pairs | 42–66 | 42.3–66.3 | MATCH
9.68-a ck5 seed 1100009: share 0.5, 0.75, reference, ratios | 62.2 / 71.1 / 89.7 / 0.69 / 0.79 | 62.17 / 71.08 / 89.67 / 0.693 / 0.793 | MATCH
9.68-a ck5 seed 1200007: share 0.5, 0.75, reference, ratios | 45.2 / 69.5 / 89.9 / 0.50 / 0.77 | 45.17 / 69.50 / 89.92 / 0.502 / 0.773 | MATCH
9.68-a ck5 seed 1300021: share 0.5, 0.75, reference, ratios | 69.6 / 76.4 / 88.9 / 0.78 / 0.86 | 69.58 / 76.42 / 88.92 / 0.783 / 0.859 | MATCH
9.68-a ck5 seed 1400017: share 0.5, 0.75, reference, ratios | 71.8 / 76.5 / 87.4 / 0.82 / 0.88 | 71.75 / 76.50 / 87.42 / 0.821 / 0.875 | MATCH
9.68-a ck5 seed 1500007: share 0.5, 0.75, reference, ratios | 71.3 / 70.2 / 90.2 / 0.79 / 0.78 | 71.33 / 70.17 / 90.17 / 0.791 / 0.778 | MATCH
9.68-a ck5 seed 1600033: share 0.5, 0.75, reference, ratios | 64.6 / 75.1 / 81.6 / 0.79 / 0.92 | 64.58 / 75.08 / 81.58 / 0.792 / 0.920 | MATCH
9.68-b twelve receipts end at checkpoint 5, no completion record | 12 at ck5 | 12 of 12 at ck5; completion record absent: True | MATCH
9.68-c higher at 0.75 in five of six; by >= 5 in four | 5; 4 | 5; 4 | MATCH
9.68-d at or above 0.80 of reference: share 0.75; share 0.5 | 3 of 6; 1 of 6 | 3 of 6; 1 of 6 | MATCH
9.68-e family gate at ck5: all six at 0.75; at 0.5 two fail (1200007, 1600033) | 6; 1200007, 1600033 | 6; 1200007, 1600033 | MATCH
9.68-e' 1200007 held-pair ratio at 0.5; 1600033 mod-97 ratio at 0.5 | 0.73 / 0.79 | 0.725 / 0.794 | MATCH
9.68-f +- fall ck4 -> ck5 in 1200007, 1300021 at share 0.5 ; at 0.75 | 29.2 / 21.2 / 7.3 / 8.5 | 29.17 / 21.17 / 7.33 / 8.50 | MATCH
9.68-g curve beyond 0.75, seed 400009 (hard; mod-73 at 0.5/0.75/0.875/0.9375) | 51.5 / 89.7 / 69.8 / 87.8 / 70.1 / 82.4 / 72.2 / 82.1 | 51.50 / 89.69 / 69.83 / 87.75 / 70.08 / 82.38 / 72.25 / 82.13 | MATCH
9.68-g curve beyond 0.75, seed 600013 (hard; mod-73 at 0.5/0.75/0.875/0.9375) | 46.2 / 80.8 / 56.9 / 80.4 / 69.2 / 77.4 / 73.9 / 78.3 | 46.17 / 80.81 / 56.92 / 80.38 / 69.17 / 77.44 / 73.92 / 78.31 | MATCH
9.68-g curve beyond 0.75, seed 300007 (hard; mod-73 at 0.5/0.75/0.875/0.9375) | 68.8 / 89.9 / 75.4 / 89.8 / 56.5 / 87.5 / 79.8 / 88.0 | 68.75 / 89.88 / 75.42 / 89.75 / 56.50 / 87.50 / 79.83 / 88.00 | MATCH
9.68-h mod-73 lower at the two highest shares than at 0.75 | 2–6 | 1.8–5.6  [against share 0.5 the drop is 1.8-7.6] | MATCH
9.68-h' at ck5 no high-share life under the family gate | none under | none under | MATCH
CN2-1 window learner, nothing frozen: family / plus-minus / whole on 15 corrected-panel seeds | 13 / 2 / 9 of 15; nothing frozen | 13 / 2 / 9 of 15; freeze fields ['None', 'none']  ['None' = field absent in older receipts; no receipt records any freezing] | MATCH
CN2-2 retains nothing without replay (share 0) | ~1% hard pairs; 0.13-0.35 mod-97 ratio | 1.3, 1.0, 1.1 | MATCH
CN2-3 more replay -> better hard pairs, rule more than backprop | rule gain > backprop gain | rule +29.6/+36.9, backprop +9.8/+14.8 | MATCH
CN2-4 first-bank projection: truncation in time, not weight signs (alignment at birth) | exact feedback no better; through-time much better | sign ['0.09', '-0.06', '0.05'], exact ['0.08', '-0.01', '0.26'], through time ['0.81', '0.78', '0.55'] | MATCH
CN2-5 Phase 4 new arm, mod-97 ratio to reference | 0.92–0.95 | 0.919–0.952 | MATCH
CN2-5' original skills fall to | 11–43 | 10.7–43.2 | MATCH
CN2-6 backprop on the pipeline above the new arm, mod-97 ('3-7 points') | 3–7 | 3.0–7.3 | MATCH
CN2-6' backprop on the pipeline / reference | 0.98–1.02 | 0.984–1.018 | MATCH
CN2-7 forward-only credit matches backprop to rounding (worker's E19_CHECK.json) | to rounding | max relative error 5.1e-16 in float64 (3 inits), 2.6e-07 in float32 (3 inits); all test1/test2 pass: True/True  [the coordinator's own independent check is recorded only in E19_COORDINATOR_CHECK.md (no JSON)] | MATCH
CN2-7' linear write costs about 22 points of birth accuracy under backprop (mean of three) | 22 | 22.4  [per seed 22.2, 24.0, 21.0; reference births on the owner's machine, E19 births on a cloud container] | MATCH
CN2-8 reversal overlap ('25%') | about 25 | 24.5–25.8 | MATCH
CN2-8' sub-step exposure | 71–73 | 71.1–72.7 | MATCH
CN2-8'' strict panels move ratios by at most 0.025; 'did not inflate scores' | <= 0.025 | not recomputed: from the strict re-score on weights, withdrawn by review 9.79 item 3; not in any JSON | NOT_RECOMPUTED

SUMMARY: 319 checks; MATCH 277; WITHIN 2; DIFFER 19; NOT_RECOMPUTED 21
DIFFER: 9.60-a'; 9.60-d; 9.61-h; 9.66-h; 9.62-j; 9.64-f; 9.67-l; 9.70-h'''; 9.75-j'(9); 9.76-a; 9.76-a; 9.76-c; 9.79-b(2); 9.79-e(3); 9.79-h'; 9.71-h; 9.71-j'''; 9.71-m; 9.81-f
WITHIN: 9.66-d'; 9.66-e'''
```
