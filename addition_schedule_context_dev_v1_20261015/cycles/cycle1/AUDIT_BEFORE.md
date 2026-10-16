# Cycle 1 — design and implementation audit before test

Scope is a technical exercise, capped at 96 attempted record applications across all three cycles. No production world may train.

Findings and design refinements:

1. NEW dose zero must be immediately before the first NEW record, not silently mapped to birth: BLOCKED T=16384, INTERLEAVED T=16. The common global HELD grid remains unchanged.
2. Preserve six histories with a literal same record-ID/target multiset; shuffle labels BEFORE merging the two streams.
3. Actual RMS displacement must be captured from the existing update; copying the updater is acceptable only with multi-step exact parent parity. First-step-only parity is insufficient because the initially zero output head hides internal gradients.
4. Private q/p must update AFTER prediction and target writing, shared FE0 unchanged; both interfaces use per-instance pure-function caches, and probes have empty isolated caches.
5. This cycle does not qualify the full execution pipeline or scientific capability. Checkpoint and failure/resume qualification follow in cycles 2 and 3.

Planned test: data/schedule invariants on all prospective rosters without model execution; real four-record technical six-history runs, three-step parent parity, and native W/NO_WRITE in both variants. All actual calls must reserve ledger entries first.
