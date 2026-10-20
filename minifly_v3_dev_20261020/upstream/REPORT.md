# MiniFly-CLS v0: native mushroom body as fast memory + learned slow core + sleep replay (2026-10-18)

**Development build and first gates. Not formal science, no adoption, no registry change.** Uses the three technical
worlds 202610143901–03 only (same single-digit split as the lineage comparison and the addition diagnosis).
Implements the proposal in `addition_difficulty_diagnosis_dev_20261017/MEMO.md` follow-up discussion. Decisions taken
with the user: the teaching signal may reach upstream layers (eligibility traces / feedback, backprop only as a
reference) and silent thinking steps are allowed. FE0 replace-or-keep was left to me: FE0 stays available as an arm,
and the order-keeping front end is the memory's default.

## What was built

**Fast memory (`fastmem.py`)** — the native Full151 mushroom body, unchanged inside: 16 compartments, each a canonical
newborn native fly; each compartment's value is one bit of the fixed 257-class C16 output code; only the fly's own
fast/slow value stores learn, through its own signed teach. Four changes around it, none inside the native equations:

1. **Order-keeping front end (FE-PERM).** Each fed byte first moves the receptor trace through one fixed permutation
   of the 88 receptor channels, then adds the byte's receptor pattern; decay 0.95 per byte step. Reordered prompts
   share 31% of KC code instead of FE0's 69%; addition-prompt pairs sharing > 80% of KCs drop from 41 to 1.
2. **One-sided KC homeostasis.** A KC firing more than 5× the native active fraction at teaching events raises its
   threshold; thresholds never fall. Before: 65 of a prompt's 184 KCs fired for > 80% of all prompts (the shared `+`
   and `=`). After: none; keys settle within a few passes (a two-sided version kept moving keys: same-prompt overlap
   0.48 between passes 1 and 8).
3. **Adaptation reset at record start**, in training and recall alike, so a fact is written and read under the same
   KC adaptation trajectory (native adaptation recovers over ~12 min, far longer than a record).
4. **Error-corrective teaching (delta).** Before each teaching event the memory reads its own current output for
   that context exactly as at recall; each compartment gets the native signed write with coefficient
   s = (tanh(z/T) − y)/2. Bits already right are not written again. This is the R3 residual idea from Package A.
   Without it, frequent targets (EOS ends every record; "1" starts every two-digit sum) accumulate in the ~1,770 KCs
   the native readout listens to (only 1,770 of 5,177 KCs have nonzero readout weight) and swamp every key.

**Slow core (`slowcore.py`)** — 64 leaky tanh units with fixed per-unit time constants, byte in / byte out, its own
257-way output, optional silent THINK steps after the prompt. Learning: **BPTT** (reference) or **e-prop** (local
eligibility traces × learning signal through fixed feedback weights). Decoupled weight decay = downscaling.
`test_slowcore.py`: BPTT matches finite differences; e-prop equals BPTT exactly without recurrence; aligned with it
with recurrence; both learn a small table.

**Coupling (`cls.py`)** — wake: the memory stores each record at once, and the record enters a bounded episode
cache (reservoir over all experience, or "recent" = only the current task). Sleep: replay from the cache into the slow
core with downscaling, optionally also into the memory (its own error-corrective write); the memory then rests 8 h.
Answer: each system's confidence divided by its median confidence on cached prompts; the higher wins.

## Results (`results/SUMMARY.json`; per-run files in `results/memory`, `slow`, `cls`, `crosstask`)

### G1 — memory: 80 taught addition facts through the byte stream (24 passes; 3 worlds)

| Memory configuration | Taught recalled | After 1 h rest | After 1 day rest |
|---|---|---|---|
| **Full** (FE-PERM, homeostasis, adaptation reset, delta) | **84–88%** | 81–86% | **81–85%** |
| FE0 front end instead | 78–85% | 72–83% | 71–79% |
| no homeostasis | 63–64% | 59–63% | 57–61% |
| no adaptation reset | 39–44% | 16–26% | 16–26% |
| plain signed teach (not error-corrective) | 1–3% | 1–3% | 1–3% |
| pure native logged teach | 1% | 1% | 1% |

Lineage native cores on the same facts: 0–8%. Full-design curves are still rising at pass 24 (75% at pass 16).

### G2 — order (memory level): 32 prompts that differ only in order, brackets or an early byte

`(2+3)*4=`, `2+3*4=`, `2*(3+4)=`, `2*3+4=`, `9-2=` vs `2-9=`, …: **FE-PERM 100%** (100% after a day) vs FE0 75% (66%).

### G3 — never-taught sums, slow core (3 worlds × 2 seeds; 20k full-batch steps)

| Rule | Downscaling | Thinking steps | Taught | Unseen sums |
|---|---|---|---|---|
| BPTT | 1 | 2 | 100% | **13%** (0–50%; mean best during training 20%) |
| BPTT | 1 | 0 | 100% | 0% |
| BPTT | 0 | 0 or 2 | 100% | 0% |
| e-prop | 0 or 1 | 0 or 2 | 91–100% (cell means) | **0%** |

For comparison the gated GRU reference of the diagnosis reached 6% (wd 0) and 44% (wd 1) in the same regime. **G3
fails**: this gate-free leaky core generalises worse than a gated one, and the local rule learns the taught table but
not the rule.

### G4 — keep and reuse: addition, then multiplication (same 80 pairs), 24 passes each, sleep after every pass

| Sleep replay | After addition: + taught (combined / memory / slow) | After multiplication: + kept (combined / memory / slow) | × taught (combined) |
|---|---|---|---|
| none ("recent" cache) | 95–98% / 81–88% / 100% | **4–6%** / 13–16% / 3% | 100% |
| into slow core | 86–89% / 81–88% / 78–86% | 53–59% / 13–16% / 56–73% | 88–95% |
| **into both systems** | 91–96% / 86–91% / 81–99% | **70–85%** / 60–68% / 74–89% | 85–90% |

Unseen products: 10–30% in every arm (mostly the operand-0/1 facts); unseen sums: 0–20%.

**Why the memory lost addition without replay into it** (`diag_crosstask.py`, memory only, 24 + 24 passes): the two
tasks' keys share about a third of their KCs (same-pair overlap 0.30 with homeostasis, 0.48 without), because the
digits are the same. The error-corrective write fixes what is being trained and overwrites unrehearsed facts on the
shared KCs (84% → 12.5% with homeostasis; 64% → 24% without). A day of plain rest costs only ~3 points (G1), so this is
interference, not decay. (An earlier explanation I gave, that homeostasis had suppressed the operator byte, was
wrong: the operator still separates the keys.)

## Verdict on the gates

* **G1 memory — largely met** (84–88% at 24 passes, still rising; target ≥ 95%). The native mushroom body can now be
  the memory. The decisive changes are the error-corrective write and the adaptation reset.
* **G2 order — met at the memory level** (100% vs 75%).
* **G3 unseen sums — not met** (best 13%; e-prop 0%).
* **G4 keep — met in part** (addition 70–85% kept while multiplication is learned, vs 4–6% without replay);
  **reuse — not shown** (unseen products only at the easy-rule level).

## Next steps

1. **Slow core generalisation.** Use gated units (the GRU reference generalised 44% in the same regime) and keep
   thinking steps (they mattered for BPTT: 0% → 13%). For the local rule: e-prop drops the credit that flows through
   recurrent connections, but combining the two operands needs exactly that credit, so the rule needs a richer
   eligibility (keeping some recurrent terms) or an architecture where the operands meet within one step.
2. **Memory to ≥ 95%**: more passes or a sharper code; check capacity with both tasks stored.
3. **Arbitration**: the combined answer is sometimes below the better single system (e.g. addition after ×:
   combined 53–59% vs slow 56–73% when only the slow core is replayed).

## Limits

3 technical worlds; 1–2 seeds per cell; no statistics. The four memory changes are new mechanisms around the native
fly (the native equations, B matrix and stores are untouched). The adaptation reset means the fly no longer carries
KC adaptation from one record to the next. The episode cache is a byte store, as in ledger item U047, not a native
mushroom-body function. Costs: memory ≈ 0.1 CPU s per record; slow core 12 ms (BPTT) / 32 ms (e-prop) per
full-batch step; one G4 life ≈ 15–18 CPU min. The native fly is imported read-only from `minifly_a_v3/src` and its
extracted `package/` (not in git), so the same package must be present to run this.

## Files

`fastmem.py`, `slowcore.py`, `cls.py`, `data.py`, `bootstrap.py`, `OUTPUT_CODEBOOK.json` (copied from the lineage
comparison); gates `gate_memory.py`, `gate_slow.py`, `gate_cls.py` with grids `grid_memory.py`, `grid_slow.py`,
`grid_cls.py`; `diag_crosstask.py`; `test_slowcore.py`; `summarize.py` → `results/SUMMARY.json`.
