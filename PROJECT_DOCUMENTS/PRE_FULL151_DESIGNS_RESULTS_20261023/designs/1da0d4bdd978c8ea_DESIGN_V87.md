# MiniFly V87 — V84E core identification

## Question

V84E works as a bounded persistent-learning phenotype. V87 asks which parts are actually necessary, how robust the boundary-hit parameters are, and what the local write rule is doing. It does **not** evolve a new genome and does **not** declare minimality automatically.

## Anchor and invariants

- Anchor: exact V84E champion (`champion_model.json` SHA-256 `0b4b3b...3baf6`).
- Audited engine: V82E `model_evo.py` / `common_evo.py` hash-locked.
- No new memory bank, slot, replay trace, or persistent array.
- Nonstructural interventions copy the exact realized V84E PN→KC matrix after construction. This prevents the genome-hash jitter seed from contaminating gene ablations.
- The only interventions allowed to change B are `gene_REF_g_pnkc_weight`, `block_REF_representation`, and `REF`, because PN→KC weight restoration is part of those interventions.
- Every raw-PN history is shared across variants.
- Both fixed REF-Q and fixed V84E-Q observers are recorded, together with actual-only and candidate-coupled behavior.

## Variant panel

44 variants total:

1. `V84E`, `REF`.
2. Four block restorations: allocation; write/teaching; expression; representation/dynamics.
3. Fifteen single-gene restorations. `g_slow_tau` and `g_pnkc_rewire` are omitted because they are already exactly zero in V84E.
4. `g_support` dose: 0.25, 0.5, 0.75; endpoints supplied by V84E and `gene_REF_g_support`.
5. Direct active-fraction robustness: 2.5%, 3.0%, 4.0%; V84E is 3.5355% and REF-sparsity gives 5%.
6. `b_adapt` dose: -0.25, -0.5, -1.0; V84E is -0.75 and gene restoration gives 0.
7. Hidden-scale robustness: adaptation time constant ×0.5/×2; fast and slow allocation normalization ×0.5/×2.
8. Three matched shuffled-support controls: same number of newly taught rows and same added teaching-row pattern distribution, but relocated away from V84E's readout-aligned support geometry.
9. Small allocation × support × sparsity interaction panel.

## Life assay

Each history learns the same 168 identities in blocks ending at 48, 72, 96, 120, 144, and 168. Each pair receives six acquisition exposures. A one-day no-input interval follows each block. Immediate and post-day states are both read out.

This assay is not interpreted as requiring permanent memory. Its purposes are to detect saturation/freezing, continued acquisition, dynamic-range collapse, and gross regression of the V84E operating regime.

## Copy-only challenge probes

At loads 96 and 168, copies of the live state receive:

- repeated consistent reinforcement (`STABLE`),
- one contradictory outcome followed by consistent reinforcement (`ANOMALY1`),
- sustained reversal (`REV`),
- a genuinely new held-out association (`NEW`).

Matched no-write copies estimate collateral old-memory damage. These probes never alter the main lifetime trajectory.

## Internal mechanism probes

At loads 96 and 168, copied states record event-level:

- alpha write L1,
- slow/fast split,
- mass-weighted allocation fraction `z`, including p10/p90,
- pre-event adaptation,
- sign-aligned fast and slow local state,
- cosine overlap of slow writes for repeated-same, old-vs-new, and correct-vs-reversal events.

This instrumentation is evaluator-only and is never fed back into learning.

## Inference

History/seed is the independent unit. All contrast tables are paired against V84E. No scalar winner and no automatic prune list are produced. The output is intended to support a subsequent, explicit joint-pruning confirmation once the causal results are reviewed.
