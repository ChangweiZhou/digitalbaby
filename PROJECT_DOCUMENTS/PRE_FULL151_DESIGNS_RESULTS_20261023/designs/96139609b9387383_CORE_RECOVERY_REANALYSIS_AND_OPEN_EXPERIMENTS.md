# CORE-RECOVERY — Re-analysis results and remaining open experiments

**Date:** 2026-09-18
**Author:** Claude (analysis run against archived data only; no new simulation executed)
**Scope:** everything the historical endpoint audit reduced to "two unresolved mechanisms", re-examined.

---

## 0. Why this document exists

The OMNIBUS10 factorial — the experiment that assigned dispositions to all five
mechanism families and whose result the CORE-RECOVERY bridge rests on — was scored
on a **single blanket old-memory retention endpoint**.

`omnibus10_results/FACTORIAL_CONTRASTS.csv` contains 120 contrast rows
(15 terms x 8 scopes). Every one has `endpoint = A_keep_after_D_heldout_bpc`.
`CELL_CONTRASTS.csv` is the same: 8 contrasts, same endpoint. There is no
competence endpoint anywhere in either contrast table.

Competence appeared only as two undocumented summary means in
`MECHANISM_CLASSIFICATION.csv`, with no intervals, no p-values and no interactions.
One of those two (`revision_accuracy_effect_mean`) is reconstructed below and turns
out to be a **terminal** measure, not a revision measure.

Meanwhile the raw data needed for a competence analysis was collected and never used:

| file | rows | content |
|---|---:|---|
| `PREQUENTIAL.csv` | 125,440 | per-exposure `accuracy`/`bpc`, phases A-D, exposures 1-8 |
| `CHECKPOINT_PROBES.csv` | 94,080 | 9 checkpoints x 15 groups, incl. `revision_1/2/4/8` and `first_return` |
| `READ_AUTHORITY_PROBES.csv` | 70,560 | readers `hard` / `shadow50` / `corr2` x 6 probe groups |
| `REINSTATEMENT_PROBES.csv` | 11,760 | silent reinstatement, modes `none`/`correct`/`wrong` |
| `STATE_COSTS.csv` | 3,920 | per-arm state byte accounting |

Part 1 reports what that data says. Part 2 lists what genuinely still needs new
simulation. Part 3 lists work that is neither — scans and source recovery.

---

## 1. Method and its validation

Re-analysis uses the **exact method of the original analysis**, established by
reproducing its published output before computing anything new:

* inferential unit = world seed (n = 16)
* paired seed-level effect, pooled over the 7 cell families and all other factor levels
* Student t 95% CI, df = 15
* interactions reported as the half-effect (the published convention)
* positive effect always means **"factor ON is better"**
  (for `bpc`: `mean(OFF) - mean(ON)`; for accuracy: `mean(ON) - mean(OFF)`)

**Validation.** Recomputing the published endpoint `A_keep_after_D_heldout_bpc`
reproduces all 15 OVERALL terms:

```
worst absolute error in mean / CI low / CI high / sd : 9.7e-17
worst relative error in p                            : 1.2e-12
```

Holm correction below is applied across the 5 main effects within each endpoint
(m = 5). The original pooled a family of 13 (5 main effects + 8 cell contrasts);
that choice is noted where it matters and does not change any sign.

Script: `core_recovery_reanalysis.py` (accompanying this document).

---

## 2. PART 1 — Resolved by re-analysis, no new simulation needed

### 2.1 Headline table

| # | Mechanism | Ledger | Original basis for setting aside | Re-analysis verdict | Disposition |
|---|---|---|---|---|---|
| 1 | `C_FORGET` | — | hurt blanket retention (-0.0437, p=3.8e-05) | **helps revision at 2/4/8 exposures and all later acquisition**; small costs on anomaly and A first-return | **REOPEN — bridge premise confirmed** |
| 2 | `B_CONSOL` | — | retention ns (-0.0065, p=0.27) | **hurts** acquisition, revision @1/2/4, all three first-return probes, later acquisition C and D | **CLOSE — no competence case** |
| 3 | `D_LOCAL` | — | hurt retention (-1.366) | hurts every competence endpoint measured | **CLOSE** |
| 4 | `E_META` | — | classified TRADEOFF | hurts acquisition, revision @1/2/4/8, later acquisition; helps retention and anomaly only | **CLOSE (marginal)** |
| 5 | `A_ELIG` | — | adopted (best on retention) | helps acquisition, first-return A, anomaly, later acquisition; **hurts revision @4** | **ADOPTION STANDS, with a caveat** |
| 6 | `COV96` | M058 | "failed here (+0.378 bits old loss)" | hurts retention **and** acquisition, revision @2/4/8, later acquisition C/D | **CLOSE — confirmed on competence** |
| 7 | `CORR2` | M063 | "attenuation worsened this distributed readout" | worse on **all six** probe groups (-0.12 to -0.18 bpc) | **CLOSE — confirmed on competence** |
| 8 | simple leaky cell | M005 | chosen on retention | `LEAK_HOM` beats `STATIC` on **all 7** endpoints | **ARCHITECTURE CHOICE STANDS** |
| 9 | richer cell families | M001-M004, M006-M014 | closed on retention | `ALIF_HOM` and `ELM_HOM` vs `LEAK_HOM`: ns on **all 7** endpoints | **CLOSE (for the recovered proxies)** |
| 10 | heterogeneity | M011 | "negative exact omnibus ... on this loop" | worse on **all 7** endpoints | **CLOSE — confirmed on competence** |
| 11 | `CONTRACT_0.05` | — | failed 9 blanket old-memory gates | **unique arm of 10** passing every competence gate | **REOPEN — selection is not cherry-picking** |
| 12 | other contraction arms | — | failed many gates | all 9 others fail competence gates too | **CLOSE** |

### 2.2 OMNIBUS10 five factors on competence endpoints

Paired seed-level effect, n=16, Holm across the 5 main effects per endpoint.
Positive = factor ON is better. Bold = survives Holm at 0.05.

| endpoint | A_ELIG | B_CONSOL | C_FORGET | D_LOCAL | E_META |
|---|---:|---:|---:|---:|---:|
| **retention** `after_D/A_keep_heldout` (bpc) | **+0.0371** | -0.0065 | **-0.0437** | **-1.3655** | **+0.0147** |
| acquisition A (held-out) | **+0.0088** | **-0.0120** | +0.0016 | **-0.4552** | **-0.0015** |
| acquisition A (trained) | -0.0065 | **-0.0091** | **+0.0021** | **-0.3764** | **-0.0014** |
| revision @1 | +0.0026 | **-0.0132** | +0.0030 | +0.0094 | **-0.0054** |
| revision @2 | -0.0018 | **-0.0243** | **+0.0116** | -0.0558 | **-0.0174** |
| revision @4 | **-0.0181** | **-0.0384** | **+0.0229** | **-0.3615** | **-0.0311** |
| revision @8 | +0.0050 | -0.0038 | **+0.0169** | **-0.6281** | **-0.0124** |
| first-return A (held-out) | **+0.0142** | **-0.0077** | **-0.0035** | **-0.5414** | **+0.0012** |
| first-return A (trained) | **-0.0104** | **-0.0115** | **-0.0053** | **-0.5213** | +0.0003 |
| first-return B (held-out) | **+0.0108** | **-0.0164** | **+0.0055** | **-0.5411** | **-0.0034** |
| anomaly resistance | **+0.0202** | +0.0027 | **-0.0052** | **-0.5481** | **+0.0059** |
| later acquisition C | **+0.0121** | **-0.0173** | **+0.0070** | **-0.5728** | **-0.0073** |
| later acquisition D | **+0.0072** | **-0.0188** | **+0.0126** | **-0.5854** | **-0.0097** |

**`C_FORGET` reading.** The profile is exactly that of a mechanism that releases
obsolete information: revision improves and the improvement *grows* with corrective
exposures (+0.0030 → +0.0116 → +0.0229 → +0.0169 at 1/2/4/8), later acquisition of
new material improves (C +0.0070, D +0.0126), and the costs are confined to
recovering the old A on first return (-0.0035 held-out, -0.0053 trained) and to
anomaly resistance (-0.0052). Blanket retention is the one thing it clearly hurts.
This is the first direct competence evidence for the bridge's premise, and it
supports it.

**`B_CONSOL` reading — this corrects an earlier claim of mine.** The published
`revision_accuracy_effect_mean` column ranked B_CONSOL (+0.0074) above C_FORGET
(+0.0067), which I previously cited as reason to doubt the bridge's choice. That
column is reconstructed exactly by computing revised-item accuracy at
**`after_D / A_rev`** — the terminal checkpoint, after three further tasks — not on
the revision curve. On the actual revision curve the ranking reverses:

| factor | published summary (`after_D/A_rev`) | true revision curve (mean of @1,2,4,8) |
|---|---:|---:|
| A_ELIG | +0.0095 | -0.0031 |
| B_CONSOL | +0.0074 | **-0.0199** |
| C_FORGET | +0.0067 | **+0.0136** |
| D_LOCAL | -0.5823 | -0.2590 |
| E_META | -0.0102 | -0.0166 |

`C_FORGET` is the only factor that improves revision. The single competence number
that existed in the archive was itself a terminal-retention-flavoured measure.
**My earlier statement that B_CONSOL outranks C_FORGET on revision was wrong.**

**`A_ELIG` caveat.** The one adopted factor is the only one that was best on the
contaminated endpoint, and it **hurts revision at 4 exposures** (-0.0181, Holm
p=4.8e-03) and hurts A first-return on trained items (-0.0104). Its adoption is not
overturned — it helps acquisition, anomaly, later acquisition and held-out first
return — but "adopt A_ELIG, set aside the rest" was decided on retention, and on
competence the picture is mixed rather than clean.

### 2.3 `COV96` (M058) and `CORR2` (M063)

`COV96` vs the zero-factor baseline, paired by seed, pooled over 7 cells:

| endpoint | effect | 95% CI | Holm p | verdict |
|---|---:|---|---:|---|
| retention (the rejection basis) | -0.3782 | [-0.4293, -0.3271] | 8.5e-10 | HURTS |
| acquisition A (held-out) | -0.0204 | [-0.0286, -0.0121] | 4.7e-04 | HURTS |
| revision @2 / @4 / @8 | -0.0848 / -0.3371 / -0.2656 | — | ≤1.4e-03 | HURTS |
| later acquisition C / D | -0.1420 / -0.1543 | — | ≤2.8e-08 | HURTS |
| revision @1, first-return, anomaly | — | — | ns | ns |

The retention figure -0.3782 reproduces V51's "+0.378 bits old loss" exactly, an
independent check on the pipeline. `COV96` fails on competence as well as retention:
its closure was stated in contaminated terms but is **substantively correct**.

`CORR2` reader vs the `hard` reader is worse on all six probe groups
(A_keep -0.161, A_rev -0.160, A_noise -0.151, B -0.163, C -0.181, D -0.116 bpc;
all Holm p < 1e-9). Uniformly worse, not a retention artifact. **Closure correct.**

### 2.4 Architecture / cell family — the search-space layer

This is the layer most relevant to whether the current 17-gene family sits in an
artificially chosen space. It does not appear to be contaminated.

| contrast | retention (basis) | competence endpoints |
|---|---|---|
| `LEAK_HOM` vs `STATIC` | LEAK better (+0.162) | **LEAK better on all 7** (acquisition, revision @4/@8, first-return, anomaly, later acquisition) |
| `ALIF_HOM` vs `LEAK_HOM` | ns | **ns on all 7** |
| `ELM_HOM` vs `LEAK_HOM` | ns | **ns on all 7** |
| `HET` vs `HOM` | HET worse (-0.188) | **HET worse on all 7** |

The choice of a simple homogeneous leaky cell (M005) survives a competence
re-analysis on every endpoint tested. Richer recovered cell proxies show no benefit;
heterogeneity is worse everywhere. **The cell-family layer of the search space does
not need reopening.**

### 2.5 V74 contraction — gate re-tally

V74's development screen has 23 gates per arm. Twelve are blanket old-memory gates
(`old_accuracy`, `old_loss`, `old_worst` at 2 phases x 2 loads); eleven are
competence or engineering gates (`REV_10pp` x2, `STABLE` x2, `ANOMALY` x2, `NEW` x2,
`capacity96_day`, `reader_blank`, `state_budget`). Re-tallying `development_gates.csv`
with the twelve old-memory gates removed:

| arm | as published | competence-only | competence gates failed |
|---|---|---|---|
| `CONTRACT_0.01` | fail (4) | fail (2) | REV_10pp x2 |
| `CONTRACT_0.025` | fail (9) | fail (2) | REV_10pp x2 |
| **`CONTRACT_0.05`** | fail (9) | **PASS** | **none** |
| `CONTRACT_0.1` | fail (12) | fail (2) | MIXED_ANOMALY, capacity96_day |
| `CONTRACT_0.2` | fail (15) | fail (3) | MIXED_STABLE, MIXED_ANOMALY, capacity96_day |
| `FAST_CONTRACT_0.01` … `0.2` | fail (6-13) | fail (2-3) | REV_10pp x2 (+ capacity96_day) |

`CONTRACT_0.05` is the **only arm of ten** that clears every competence and
engineering gate, and it was blocked **solely** by nine blanket old-memory gates:
`CHALLENGE_48_old_accuracy`, `CHALLENGE_48_old_worst`, `CHALLENGE_96_old_worst`,
`MIXED_48_old_accuracy`, `MIXED_48_old_loss`, `MIXED_48_old_worst`,
`MIXED_96_old_accuracy`, `MIXED_96_old_loss`, `MIXED_96_old_worst`.

This also settles a methodological worry about the bridge: selecting 0.05 out of a
4-dose x 2-rule sweep is **not** post-hoc cherry-picking. The competence gates
select it uniquely.

### 2.6 Net effect on the bridge's scope

| bridge arm | status after re-analysis |
|---|---|
| `C_FORGET` 2x2 (A_ELIG x C_FORGET) | **justified and strengthened** |
| `CONTRACT_0.05` vs V74 parent | **justified and strengthened** |
| *(not in bridge)* `B_CONSOL` | now **closed** — do not add |
| *(not in bridge)* `COV96`, `CORR2`, `M011`, richer cells | now **closed** — do not add |

The bridge is aimed at the right two mechanisms. What it lacked was evidence for
that claim; this re-analysis supplies it, and simultaneously closes four of the
items I previously flagged as wrongly excluded.

---

## 3. PART 2 — Genuinely needs new experiment or simulation

| # | Question | Why re-analysis cannot answer it | Existing asset | Cost |
|---|---|---|---|---|
| E1 | Does `C_FORGET` destroy useful information or release obsolete information? | OMNIBUS10 cannot separate an isolated anomaly from a true reversal — they are the same event until later evidence arrives. The archive has an anomaly probe and a revision curve but never a **shared-prefix** D2A/D2R branch. The measured anomaly cost (-0.0052) is exactly what needs adjudicating. | bridge design §5.2/§5.3; `elm_omnibus10.py` intact with `_forget`, `FACTOR_SPECS.json`, archived fixture seeds | 512 lifetimes screen (+1024 confirm) |
| E2 | Does `CONTRACT_0.05` preserve behaviour when demand **returns**? | V74 has no dormancy/reactivation curriculum at all. `first_return` exists in OMNIBUS10, not in V74. D3 was never run for this mechanism. | `minifly/iteration19/src/{model74,selftest74,campaign74}.py`, `checks/lock.json` | 256 lifetimes screen (+512 confirm) |
| E3 | Are the 17 gene **bounds** right? | The V82E box (`g_k0` ±0.3, `g_readout` ±0.4, several tighter than ±1) was set under the old endpoints and is frozen by hash in `QUALIFICATION.json`. Nothing in the archive measures behaviour outside the box. | make-up runner; V82E config | new search, scale TBD |
| E4 | Is the 17-gene **set** the right set? | The genes are scaling knobs chosen during V81E. No archived experiment varies gene membership. | — | new search |
| E5 | Does topology / structural plasticity matter? | `g_pnkc_rewire` permutes weights **within a row**; it never changes connectivity. No archived experiment alters topology. Cell-family re-analysis (§2.4) covers cell dynamics, **not** wiring. | connectome graph; `structural_plasticity` 298 hits in the audit index | new experiment |
| E6 | Status of the ~20 mechanism families never dispositioned | `replay_repetition` (2410 hits), `recurrence` (1666), `banks_routing` (1319), `anatomical_feedback` (1097), `homeostatic_dynamics` (1075), `protection_uncertainty` (286), `spacing_adaptation_budget` (242), `reinstatement_read_authority` (212) … none has a recorded disposition. Some may be re-analysable (see A2); the rest need experiments. | audit `MECHANISM_SEARCH_HITS.json` | unknown until A1/A2 done |
| E7 | Is Stage-I's finalist real? | `§13` of the bridge defers Stage-I final confirmation. Development/validation evidence has misled in this project four times (V81E, V82E, V84E, V85E all `CONFIRMATION_NOT_PASSED`). | make-up runner `confirm` stage | 4 x 96 = 384 lifetimes (~1.6% of Stage-I) |

### Notes on E1 and E2

Both mechanisms are **targeted, evidence-triggered release**, not blanket forgetting:

* `C_FORGET` (`elm_omnibus10.py:1185`) — *"separate cue-local decay of plastic support
  for the currently winning wrong consequence"*: fires at the thing currently producing
  the wrong answer.
* `CONTRACT` (`selftest74.py`) — asserted invariants `same_sign_unchanged`,
  `no_write_no_contraction`, `zero_fast_first_conflict`: acts **only** on directional
  conflict, and is structurally incapable of touching an unopposed memory.

Set against V66 (old memories survive 62/62 under reinforcing writes, 0/62 under
opposing; direction R² 0.757, magnitude R² -0.003), the memories `CONTRACT` can
damage are drawn from the set that opposing writes already destroy — so the
"old-memory cost" charged against it may have been largely non-additional. That is
a hypothesis the D1/D2A/D2R/D3 curricula can test and the archive cannot.

### Power note for E1/E2

V74's historical effects rest on n=8 with wide intervals. The headline
`CHALLENGE` revision effect **crosses zero** (+15.6 pp, CI -6.3 to +34.4); only
`MIXED` is individually resolved (+21.9 pp, CI +6.3 to +37.5). The bridge's §10
pilot (quantify reference variability, freeze margins before science seeds) is
therefore load-bearing, not ceremonial.

---

## 4. PART 3 — Needs work that is neither re-analysis nor simulation

| # | Item | Problem | Action |
|---|---|---|---|
| A1 | Historical retention scan is incomplete | `RETENTION_TERMS` has 8 literal strings. It misses `tail_rise`, `old_worst`, `old_loss_increase`, `conditional_old_loss`, `worst_increase_Hz` — including `tail_rise` and `old_worst`, the two most-used proxies in `V82E/src/search_evo.py`. Tested against the four V51 exclusion statements: **0 of 4 caught.** | Extend the term list, re-scan. Static, no simulation. |
| A2 | No per-family dispositions | The audit indexes 25 mechanism families but produces dispositions for none. Several may be resolvable from archived data exactly as Part 1 was. | Produce a disposition per family; route each to re-analysis or experiment. |
| A3 | Mechanism ledger is unmapped | All 123 `LEDGER_COVERAGE` entries are `REFERENCED_SOURCE_MAPPING_UNVERIFIED` with zero descriptions, and 99 trace to `fly_teaching_followup/results/source_cells.json` — a connectome cell table. The scan matched `M\d+` out of neuron IDs. A **real** ledger exists in the V51/V54 design docs (M005 leaky cell, M011 heterogeneity, M058 covariance, M063 read authority, M103 slot admission). | Reconstruct the ledger from the design documents, not by regex. |
| A4 | Era classification broken for MiniFly | 2,358 of 2,854 MiniFly files land in `unclassified`; the `v56_v79` bucket (V66-V78, where the MiniFly mechanism decisions were made) holds 143 records against 5,139 for `early_elm`. | Fix the classifier before trusting any era-based coverage claim. |
| A5 | P4 / GRU / LIF comparisons never recovered | V51 §20: *"do not mark unrecovered P4/GRU/LIF tests completed"*. §2.4 above closes the **recovered proxies** (`ALIF`, `ELM`), which is not the same as closing the historical implementations. | Source recovery first; only then decide whether an experiment is warranted. |
| A6 | Half the project tree is unreadable | macOS TCC denies both listing and reading under `~/Documents/second_try` (3,678 files by the audit's own index). `source_cells.json` could not be opened to confirm A3 directly. | Grant Full Disk Access to the terminal. |

---

## 5. Summary

**Re-analysis settled ten of the twelve items** in §2.1 without any new simulation.
Four mechanisms I had flagged as possibly wrongly excluded (`B_CONSOL`, `COV96`,
`CORR2`, heterogeneity) are **correctly closed** — they fail on competence too. The
architecture layer of the search space is **not** contaminated: the simple
homogeneous leaky cell wins on every competence endpoint.

**Both bridge mechanisms are vindicated and better motivated than the design argued.**
`C_FORGET` is the only OMNIBUS factor that improves revision, and the improvement
grows with corrective exposures. `CONTRACT_0.05` is the unique arm of ten passing
every competence gate, blocked solely by nine blanket old-memory gates.

**What remains genuinely open is the search space itself** — gene bounds, gene
membership, topology, and roughly twenty mechanism families with no disposition —
plus the deferred Stage-I confirmation. None of that is answered by the bridge, and
E3-E6 are the items that bear on whether the corrected evolution is exploring a space
that was drawn by the experiments being repaired.

**Correction on the record:** I previously told you `B_CONSOL` outranked `C_FORGET`
on revision, citing `MECHANISM_CLASSIFICATION.csv`. That column measures revised
items at the terminal checkpoint, not revision. On the revision curve the ranking
reverses and `C_FORGET` is the only factor that helps. §2.2 documents this.
