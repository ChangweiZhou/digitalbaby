# A11-A v3.8.3 technical gate and resource note

Date: 2026-09-23. This is a qualification handoff, not a science authorization.

The independent source-event audit passed for K1/L1, K16/L1, K1/L4,
K16/L4, K8/L1, K8/L4 and K8/L16. It checked all emitted rows against the
frozen V88 event sources, identity and relevance mapping, endpoint tags,
timing, phase populations, and five exact comparisons through CORE_END.
Four mutations of an audited K16/L4 boundary window were rejected: deletion,
empty relevance, an added 330 seconds, and placement after block-0 probes.
These are window-level negative controls, not complete replays of mutated
learner lives. See the independent audit receipt in
technical_runs/a11_a_event_contract/A11_A_V383_INDEPENDENT_AUDIT.json.

The structural census now includes L-1 boundary rows per life. It records
30,185 / 115,286 / 455,690 main-path rows for L=1 / 4 / 16 and 60,958,272
rows across the planned 576 lives. It does not measure learner runtime, RSS,
output size, or final-bundle size.

Measured on-disk sizes of the seven technical CSV ledgers are:

| Cell | Rows | CSV bytes |
|---|---:|---:|
| K1/L1 | 30,185 | 9,895,963 |
| K16/L1 | 30,185 | 10,826,417 |
| K8/L1 | 30,185 | 10,307,759 |
| K1/L4 | 115,286 | 37,801,046 |
| K16/L4 | 115,286 | 41,568,518 |
| K8/L4 | 115,286 | 39,469,325 |
| K8/L16 | 455,690 | 157,392,284 |

Together they occupy 307,261,312 bytes (293.03 MiB). These are technical
qualification artifacts, not official A11 science outputs, so their size
alone does not prove a §40.3 science-output cap violation. It does show that
persisting full per-event CSVs for all 576 science lives would be incompatible
with the current 256 MiB peak incremental experiment-disk target. Scaling
the observed 328–361 CSV bytes per row to the planned 60,958,272 rows would
be roughly 18.6–20.5 GiB; that range is only a format projection, not a
measured science cost or a commitment to retain such CSVs.

Before the grid-instantiated flag changes or science starts, a separate
resource qualification must specify which event data are generated, streamed,
retained and discarded; demonstrate that the required audit evidence survives
this policy; measure or defensibly bound learner CPU, wall time, aggregate
RSS, peak incremental disk, persistent files and final bundle against §40;
and produce the required precision/resource plan and independent path-specific
authorization audit. The 193 additional V88 worlds used here are explicitly
technical fixtures and cannot be relabeled as A11 science worlds.

Current state remains `A11_A_GRID_NOT_INSTANTIATED`; science authorization
remains false. This status reflects the unresolved resource/authorization
gates, not a failure of the v3.8.3 event-ledger fixtures.
