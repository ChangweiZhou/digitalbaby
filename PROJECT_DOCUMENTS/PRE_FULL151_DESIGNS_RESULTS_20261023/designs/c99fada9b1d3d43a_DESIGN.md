# MiniFly V9.9A — confirm limited feature reuse and separate its decay

**Locked design before new-world evaluation, 26 September 2026.** This is a confirmation of the *same* E3 synthetic relation and frozen REF model used in [V9.8](../BYTE_CORE_V9_8_TRIAL/DESIGN.md). It does not search parameters, routers, banks, readouts, or relation families. Engineering regression uses historical world 1201 only; science worlds are fixed at 1301–1364, disjoint from all V9.8 outcomes. The V9.8 parent checkpoint, stream generator, old/new records, timing, feedback bits, native value learner, predictor, and probe implementation remain unchanged.

## Objective, exposure, and limits

The tested learner must use old training on six P–N symbol pairings to choose correctly on three *never-reinforced* pairings of those same six symbols. Every symbol is seen in two taught edges, in both byte positions equally often; each held-out edge is absent from the old and new teaching streams. A strict exact-taught-pair lookup has no answer for those edges. A learned one-bit class per symbol can solve the task, so success establishes only limited within-family feature reuse, not general compositional reasoning, NLP, or transfer to a different task. The task-solving class decoder is an analytic positive control, not a fly component.

For each world, `W` and `N` start at the exact same V9.2 checkpoint and receive identical old sensory bytes, outcome byte `!`, binary teaching labels and clock times. `W` permits the old native value writes; `N` disables them. The engineered byte predictor learns the same bytes in both and must have the same digest. Value choices are scored on read-only clones before any held-out feedback. The inherited native value circuit alone supplies the scored answer. The prior-teaching omission is the causal control for E3; the experimental class labels are randomized across worlds.

## Timeline and same-prestate descendants

Run V9.8 E3 old teaching: 12 bouts over 12 oriented taught cues (144 records). Probe trained and held-out pairs at `old_end`, then flush 24 h and probe at `old_day`. At `old_day`, clone each of `W` and `N` into three descendants with identical respective state:

| Descendant | From `old_day` to common final time | What it isolates |
|---|---|---|
| `R` | No new bytes; passive time to the exact final clock | time/passive persistence |
| `S` | Original 72 new-cohort records, including outcome and Teach calls, with **new native writes disabled** | added sensory exposure and nonplastic dynamics |
| `L` | The same 72 records with **new native writes enabled** | V9.8's ordinary continued-learning path |

`S` and `L` consume identical bytes with identical timing, so their engineered predictors must remain identical. All descendants are probed at the same final absolute time, 24 h after the original new-cohort end. `L` is also probed immediately after the new cohort for an exact V9.8 regression check. The `L` path must reproduce V9.8 world 1201's stored REF E3 pair margins at all four original probe stages before any new world may run. `R` differs in sensory history from `S`; therefore `S−R` is a sensory-plus-nonplastic contrast, while `L−S` isolates assignment of new write flags under a matched sensory stream. Neither is automatically a pure cellular mechanism estimate.

For any final descendant `X∈{R,S,L}`, define `Δ_X = choice(W_X, heldout) − choice(N_X, heldout)`. The **sole confirmatory primary** is the world mean `Δ_L` at the final delayed probe. A two-sided paired 95% Student-*t* interval over 64 world differences is positive if its lower bound is greater than zero, negative if its upper bound is below zero, otherwise unresolved. This one primary has no multiplicity adjustment. Report the absolute `W_L` and `N_L` choices and the trained-pair guard; a positive difference alone is not automatically useful behavior. No optional stopping, world substitution, or tuning is allowed.

The prespecified secondaries are final signed pair-margin `W−N` for each descendant; `Δ_R`, `Δ_S`, `Δ_L−Δ_S`, and `Δ_S−Δ_R`; old-end, old-day, and new-end choice/margins; and per-world sign counts. They receive descriptive paired intervals without promotion thresholds or multiplicity claims. The final signed margin is especially important because V9.8's `W−N` margin was positive in 12/12 worlds while choice improved in only 5/12. It cannot replace the declared behavioral primary after outcomes are observed.

## Precision and decision boundary

The V9.8 final world-level `Δ_L` sample standard deviation was 0.264, yielding an approximate two-sided 95% interval half-width of **6.6 percentage points** at n=64 if that variance persists. This is planning information from an exploratory sample, not guaranteed power; smaller true effects or larger variance may remain unresolved. A successful primary confirms limited E3 reuse on this fixed synthetic relation under continued learning. It does **not** promote a persistent learning core without another relation/task family and broader lifetime tests. An unresolved or negative primary is reported as such even if a margin secondary is positive.

The natural interpretations are conditional: `Δ_R` lost before new input suggests passive/expression loss; `Δ_R` retained but `Δ_S` lost suggests sensory/nonplastic disturbance; `Δ_S` retained but `Δ_L` lost suggests an effect of new writes. Quantization and interactions may keep these secondary contrasts unresolved. No HASH8 arm, BROADCAST8 arm, evolution search, or hybrid learner is run here.

## Engineering, audit, and compact delivery

Preflight must verify: V9.8 world 1201 `L` margins/choices exactly reproduce the archived REF E3 rows; branch clones match at `old_day`; `S` labels are inert when new write is off; `S` and `L` predictor states match after identical new bytes; held-out contexts are absent from both teaching schedules; every symbol's teaching exposure is balanced; read-only probes leave live state unchanged; and final clocks agree. The runner locks the design, code, relevant parent sources, archived V9.8 results, and checkpoint by SHA-256 before science. It writes atomic compact per-world metrics for resume, then final tables, source/acceptance checks, a manifest pointing to parent assets, and a report. No datasets, checkpoints, event logs, or prior archives are copied. The result bundle target is well below 10 MB.
